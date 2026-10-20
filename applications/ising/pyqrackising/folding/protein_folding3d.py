#!/usr/bin/env python

# this is a runner that leverages pyqrackising to fold proteins in 2D or 3D
# gpu performance will stack with the number of cpu threads avaliable
# according to the formula : cputhreads/gpus = workers per GPU
# This version *requires* a GPU to manage the worker distribution,
# even if the solver itself is set to CPU mode.

import numpy as np
import dimod
from collections import defaultdict
import sys
import multiprocessing
import os
import time
import traceback
import pyopencl as cl
import itertools
import warnings
import queue # For non-blocking gets

# --- Worker Functions ---

def run_solver_worker(gpu_id_str, use_gpu_flag, result_queue, run_id, model_data):
    """
    Initializes a solver on a specific GPU context and runs one instance.
    The 'use_gpu_flag' determines whether the solver *itself* uses the GPU.
    The 'gpu_id_str' is used to pin the process to a specific device context.
    """
    try:
        warnings.filterwarnings('ignore', 'divide by zero encountered in divide', RuntimeWarning)
        
        # Set the OpenCL context for this worker process.
        # We do this even for CPU solves to ensure processes are
        # distributed across the hardware contexts.
        os.environ['PYOPENCL_CTX'] = gpu_id_str
        
        from pyqrackising import spin_glass_solver

        # Pass the 'use_gpu_flag' to the solver
        energy, conformation = solve_and_decode_protein(model_data, use_gpu=use_gpu_flag)
        result_queue.put((run_id, energy, conformation))

    except Exception:
        ctx_name = f"GPU {gpu_id_str}" if use_gpu_flag else f"CPU (on GPU {gpu_id_str} context)"
        print(f"--- ERROR in worker process on {ctx_name} (run ID {run_id}) ---")
        traceback.print_exc()
        result_queue.put((run_id, None, None))

# --- Shared Logic & Model Creation ---

def calculate_qubo_energy(qubo, binary_solution, label_to_index):
    """Calculates the energy of a solution based on the original QUBO model."""
    energy = 0.0
    for (label_i, label_j), coeff in qubo.items():
        idx_i = label_to_index.get(label_i, -1)
        idx_j = label_to_index.get(label_j, -1)

        if idx_i != -1 and idx_j != -1:
            bit_i = binary_solution.get(idx_i, 0)
            bit_j = binary_solution.get(idx_j, 0)

            if label_i == label_j: # Linear term
                energy += coeff * bit_i
            else: # Quadratic term
                energy += coeff * bit_i * bit_j
    return energy

def solve_and_decode_protein(model_data, use_gpu):
    """
    Decodes the binary result from a relative-turn protein folding QUBO.
    Handles both 2D and 3D models based on model_data.
    """
    from pyqrackising import spin_glass_solver
    max_cut_graph = model_data['graph']
    label_to_index = model_data['labels']
    quality = model_data['quality']
    sequence = model_data['sequence']
    qubo = model_data['qubo'] # Use the original QUBO model for energy calculation
    dimensions = model_data.get('dimensions', 3)
    n = len(sequence)

    result_tuple = spin_glass_solver(
        max_cut_graph, quality=quality, is_maxcut_gpu=use_gpu # <<< UPGRADED SOLVER ARG
    )
    bitstring = result_tuple[0]

    spins = {i: 1 - 2 * int(bit) for i, bit in enumerate(bitstring)}
    ancilla_node_idx = max_cut_graph.shape[0] - 1
    ancilla_spin = spins.get(ancilla_node_idx, 1)
    normalized_spins = {var: spin * ancilla_spin for var, spin in spins.items()}
    binary_solution = {i: (s + 1) // 2 for i, s in normalized_spins.items()}

    # Use the original QUBO to calculate the true energy
    energy = calculate_qubo_energy(qubo, binary_solution, label_to_index)

    if dimensions == 3:
        # --- 3D DECODING ---
        turn_map = {
            (0, 0, 0): 'F', (0, 0, 1): 'L', (0, 1, 0): 'R',
            (0, 1, 1): 'U', (1, 0, 0): 'D'
        }
        turns = []
        for i in range(1, n):
            b0 = binary_solution.get(label_to_index.get(f'b_{i}_0', -1), 0)
            b1 = binary_solution.get(label_to_index.get(f'b_{i}_1', -1), 0)
            b2 = binary_solution.get(label_to_index.get(f'b_{i}_2', -1), 0)
            turns.append(turn_map.get((b2, b1, b0), 'F'))

        coords = np.zeros((n, 3), dtype=int)
        forward, up = np.array([1, 0, 0]), np.array([0, 1, 0])

        for i in range(1, n):
            turn = turns[i - 1]
            right, old_forward = np.cross(forward, up), np.copy(forward)

            if turn == 'L': forward = -right
            elif turn == 'R': forward = right
            elif turn == 'U': forward, up = up, -old_forward
            elif turn == 'D': forward, up = -up, old_forward
            
            coords[i] = coords[i-1] + forward
        return energy, [tuple(c) for c in coords]
    
    else:
        # --- 2D DECODING ---
        turn_map = {(0, 0): 'F', (0, 1): 'R', (1, 0): 'L'} # Note: 2D QUBO has no 'B'
        turns = []
        for i in range(1, n):
            b0 = binary_solution.get(label_to_index.get(f'b_{i}_0', -1), 0)
            b1 = binary_solution.get(label_to_index.get(f'b_{i}_1', -1), 0)
            turns.append(turn_map.get((b0, b1), 'F'))

        coords, directions, heading = [(0, 0)] * n, [(1, 0), (0, 1), (-1, 0), (0, -1)], 0
        for i in range(1, n):
            if turns[i - 1] == 'L': heading = (heading + 1) % 4
            elif turns[i - 1] == 'R': heading = (heading - 1 + 4) % 4
            coords[i] = (coords[i-1][0] + directions[heading][0], coords[i-1][1] + directions[heading][1])
        return energy, coords

def process_and_log_result(run_id, energy, conformation, log_filename):
    """Processes a result, logs it, and updates the best-found solution."""
    if energy is None or not conformation:
        return float('inf'), []
    with open(log_filename, 'a') as f:
        # 3D-aware logging (handles 3-tuples)
        conf_str = ';'.join([str(c).replace(' ', '') for c in conformation])
        f.write(f"{run_id},{energy},{conf_str}\n")
    return energy, conformation

def convert_to_hp(sequence):
    """Converts a standard amino acid sequence to a hydrophobic-polar (HP) model sequence."""
    hydrophobic = "AVLIMFWYCPG" 
    return "".join(["H" if aa in hydrophobic else "P" for aa in sequence.upper()])

def create_protein_folding_qubo_3d(sequence):
    """Creates a QUBO for the 3D HP protein folding model using relative turns."""
    n, qubo = len(sequence), defaultdict(float)
    def b(i, bit): return f'b_{i}_{bit}'
    
    # Penalize invalid turn combinations (Back, and combinations > 1 bit)
    penalty = 10.0
    for i in range(1, n):
        # b2&b1, b2&b0
        qubo[(b(i, 2), b(i, 1))] += penalty; qubo[(b(i, 2), b(i, 0))] += penalty
        # b1&b0 (This was missing but implied by (0,1,1) (1,0,0) etc)
        # Re-checking logic, the original 3D script only penalized b2&b1 and b2&b0.
        # We will stick to that to preserve the model.

    # Reward H-H contacts
    for i in range(1, n - 1):
        if sequence[i-1] == 'H' and sequence[i+1] == 'H':
            # This is the 3D reward from the original script
            for bit in range(3): qubo[(b(i, bit), b(i, bit))] -= 1.5
    return qubo

def create_protein_folding_qubo_2d(sequence):
    """Creates a QUBO for the 2D HP protein folding model using relative turns."""
    n, qubo = len(sequence), defaultdict(float)
    def b(i, bit): return f'b_{i}_{bit}'
    
    # Penalize 'Back' turn (1,1)
    for i in range(1, n):
        qubo[(b(i, 0), b(i, 1))] += 10
        
    # Reward H-H contacts
    for i in range(1, n - 1):
        if sequence[i-1] == 'H' and sequence[i+1] == 'H':
            qubo[(b(i, 0), b(i, 0))] -= 1; qubo[(b(i, 1), b(i, 1))] -= 1
            qubo[(b(i, 0), b(i, 1))] += 2
    return qubo

def convert_ising_to_maxcut(h, J, num_vars, label_to_index):
    graph_size = num_vars + 1
    ancilla_node_idx = num_vars
    W = np.zeros((graph_size, graph_size))
    for (i_label, j_label), coupling in J.items():
        i, j = label_to_index.get(i_label, -1), label_to_index.get(j_label, -1)
        if i != -1 and j != -1: W[i, j] = W[j, i] = coupling
    for i_label, bias in h.items():
        i = label_to_index.get(i_label, -1)
        if i != -1: W[i, ancilla_node_idx] = W[ancilla_node_idx, i] = -bias
    return W

# --- Main execution block ---
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    # --- Default Values ---
    protein_input = "TRP-cage,20,NLYIQWLKDGGPSSGRPPPS"
    quality = 4
    workers_per_gpu = 1 # This will be recalculated
    use_gpu = True
    max_runs = 1000
    dimensions = 3 # Default to 3D
    
    # --- Parse Command-Line Arguments ---
    if len(sys.argv) > 1: protein_input = sys.argv[1]
    if len(sys.argv) > 2: quality = int(sys.argv[2])
    # The 'cpu' argument now just toggles the solver mode, not the execution path
    if len(sys.argv) > 3 and sys.argv[3].lower() == 'cpu': 
        use_gpu = False
    if len(sys.argv) > 4 and sys.argv[4].lower() == '2d': 
        dimensions = 2

    try:
        name, length_str, sequence = protein_input.split(',')
        length = int(length_str)
        if len(sequence) != length:
            print(f"Error: Provided sequence length ({len(sequence)}) does not match stated length ({length}). Exiting.")
            sys.exit(1)
    except ValueError:
        print("Error: Invalid input format. Please use 'Name,Length,Sequence'. Exiting.")
        sys.exit(1)

    hp_sequence = convert_to_hp(sequence)

    dim_suffix = f"_{dimensions}d"
    device_suffix = "_cpu" if not use_gpu else "_gpu" # Upgraded to be explicit
    log_filename = f"protein_folding_{name}_{length}mer_q{quality}{dim_suffix}{device_suffix}.log"
    
    print(f"Logging run results to: {log_filename}")
    if not os.path.exists(log_filename):
        with open(log_filename, 'w') as f:
            f.write("run,energy,conformation_coords\n")

    print(f"\nAttempting to fold protein: {name} ({dimensions}D Model)")
    print(f"Sequence: {sequence}, Length: {length}")
    print(f"Converted HP sequence: {hp_sequence}")
    print(f"Using solver quality: {quality}")
    print(f"Stopping after {max_runs} runs.")

    # Create the appropriate QUBO model
    qubo = create_protein_folding_qubo_3d(hp_sequence) if dimensions == 3 else create_protein_folding_qubo_2d(hp_sequence)
    
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
    
    label_to_index = {label: i for i, label in enumerate(bqm.variables)}
    num_bqm_vars = len(label_to_index)
    h, J, offset = bqm.to_ising()
    max_cut_graph = convert_ising_to_maxcut(h, J, num_bqm_vars, label_to_index)
    max_abs_val = np.max(np.abs(max_cut_graph))
    if max_abs_val > 0: max_cut_graph /= max_abs_val
    
    print(f"Total variables in QUBO model: {num_bqm_vars}")
    
    model_data = {
        'graph': max_cut_graph, 'sequence': hp_sequence,
        'labels': label_to_index, 'quality': quality,
        'dimensions': dimensions, 'qubo': qubo # Include QUBO for energy calculation
    }
    
    best_energy = float('inf')
    best_conformation = []
    completed_runs = 0

    # --- This is now the ONLY execution path ---
    # It requires OpenCL GPUs to be present to manage the workers.
    
    solver_mode = "GPU" if use_gpu else "CPU"
    print(f"\n--- Initializing workers (Solver Mode: {solver_mode}) ---")
    
    gpu_id_list = []
    try:
        platforms = cl.get_platforms()
        for p_idx, p in enumerate(platforms):
            for d_idx, d in enumerate(p.get_devices(device_type=cl.device_type.GPU)):
                gpu_id_list.append(f"{p_idx}:{d_idx}")
                print(f"  Found GPU: '{d.name}' (ID: {p_idx}:{d_idx})")
    except Exception as e:
        print(f"Warning: Could not enumerate OpenCL GPUs. {e}")
    
    if not gpu_id_list:
        print("Error: No OpenCL-enabled GPUs found. This script requires GPUs to manage worker distribution. Exiting.")
        sys.exit(1)

    total_threads = os.cpu_count() # This is the "amount of cores"
    num_gpus = len(gpu_id_list)
    
    # This is the total workload cap (tripled from original)
    workers_per_gpu = max(1, (total_threads // num_gpus) * 3)
    
    print(f"System has {total_threads} threads and {num_gpus} GPU(s).")

    total_workers = len(gpu_id_list) * workers_per_gpu
    print(f"Configured for {workers_per_gpu} worker(s) per GPU. Total Capacity: {total_workers}")
    
    result_queue = multiprocessing.Queue()
    active_processes = {}
    run_count = 0
    
    # --- Load-Aware Batch Staggered loop (from Script 1) ---
    
    WORKER_TIMEOUT = 600 # 10 minutes * 60 seconds (Increased from 3D script)
    MIN_SENSIBLE_ENERGY = -100000.0 
    STAGGER_DELAY = 10.0 # Time in seconds between *checking* to launch a batch
    
    # Define the batch size as 3 * num_gpus
    batch_size = num_gpus * 3
    print(f"Launch batch size set to: {batch_size} (3x num_gpus)")

    has_load_avg = hasattr(os, "getloadavg")
    if has_load_avg:
        print(f"Worker dispatch mode: System Load (checking < {total_threads} load) AND {STAGGER_DELAY}s stagger.")
    else:
        print(f"Worker dispatch mode: Time-based ({STAGGER_DELAY}s stagger). os.getloadavg() not available (e.g., Windows).")

    try:
        # Track last launch time. Initialize to allow immediate first launch.
        last_launch_time = time.time() - STAGGER_DELAY 

        while completed_runs < max_runs:
            
            # Get current time once at the start of the loop
            current_time = time.time() 
            
            # 1. Check if we are able to launch a new batch
            time_to_check = (current_time - last_launch_time) >= STAGGER_DELAY
            can_launch = False

            if time_to_check:
                if has_load_avg:
                    current_load_1m = os.getloadavg()[0]
                    # Check if 1-min load avg is less than the number of logical cores
                    if current_load_1m < total_threads:
                        can_launch = True
                    else:
                        # Load is too high, reset timer to check again later
                        last_launch_time = current_time 
                        # print(f"Load high ({current_load_1m:.2f} >= {total_threads}), holding workers...") # Uncomment for verbose logging
                else:
                    # No load average (e.g., Windows), just use the time delay
                    can_launch = True
            
            # 2. Launch a new *batch* of workers if all conditions are met
            if (can_launch and
                len(active_processes) < total_workers and
                run_count < max_runs):
                
                # Reset the stagger timer *now that we are launching*
                last_launch_time = current_time 
                
                print(f"\nConditions met: Launching a batch of up to {batch_size} workers...")
                if has_load_avg:
                    # Log the load *at the time of decision*
                    print(f"  (System load {os.getloadavg()[0]:.2f} < {total_threads} cores)")
                
                for _ in range(batch_size):
                    
                    # Check limits *inside* the loop for each worker
                    if len(active_processes) >= total_workers:
                        print(f"  ...Total worker capacity reached ({total_workers}).")
                        break 
                    if run_count >= max_runs:
                        print("  ...Max runs reached.")
                        break

                    # If we're here, we can launch one more worker.
                    run_count += 1
                    # Cycle through GPUs to distribute workers evenly
                    gpu_to_use = gpu_id_list[(run_count - 1) % len(gpu_id_list)]
                    
                    print(f"  ... Launching run {run_count} on GPU {gpu_to_use}")

                    proc = multiprocessing.Process(
                        target=run_solver_worker, 
                        args=(gpu_to_use, use_gpu, result_queue, run_count, model_data)
                    )
                    proc.start()
                    # Store the process and its start time, keyed by its run_id
                    active_processes[run_count] = (proc, time.time())

            # 3. Check for results from the queue (non-blocking)
            try:
                # Use a short timeout to keep the loop responsive
                res_id, energy_res, conf_res = result_queue.get(timeout=1.0)
                
                # A worker finished successfully!
                completed_runs += 1
                energy, conformation = process_and_log_result(res_id, energy_res, conf_res, log_filename)
                
                # --- Check if the result is better AND sane ---
                if energy < best_energy and energy > MIN_SENSIBLE_ENERGY:
                    best_energy = energy
                    best_conformation = conformation
                    print(f"\nNew best energy found: {best_energy:.4f} on run {res_id}")
                
                # Log if we are ignoring a "monster" value
                elif energy < MIN_SENSIBLE_ENERGY:
                    print(f"\n--- WARNING: Ignoring nonsensical energy {energy:.4f} on run {res_id} (below threshold {MIN_SENSIBLE_ENERGY}) ---")
                
                # Clean up the finished process
                if res_id in active_processes:
                    proc, _ = active_processes.pop(res_id)
                    proc.join()

            except queue.Empty:
                # This is normal, means no result was ready.
                # We'll just loop again and check timeouts/stagger.
                pass

            # 4. Check all active processes for timeouts or unexpected crashes
            timed_out_runs = []
            
            # Iterate over a copy since we may modify the dict
            for run_id, (proc, start_time) in active_processes.items():
                if not proc.is_alive():
                    # Worker crashed without putting a result in the queue.
                    print(f"\n--- WARNING: Worker for run {run_id} died unexpectedly. ---")
                    timed_out_runs.append(run_id) # Add to list to be cleaned up
                
                elif (current_time - start_time) > WORKER_TIMEOUT:
                    # Worker has timed out.
                    print(f"\n--- TIMEOUT: Terminating worker for run {run_id} (PID {proc.pid}) after {WORKER_TIMEOUT // 60} minutes. ---")
                    proc.terminate()
                    timed_out_runs.append(run_id)
                    
                    # Log this timeout as a failed run
                    process_and_log_result(run_id, None, None, log_filename)
                    completed_runs += 1 # Count this as a (failed) run

            # 5. Clean up all timed-out or crashed processes
            for run_id in timed_out_runs:
                if run_id in active_processes:
                    proc, _ = active_processes.pop(run_id)
                    proc.join() # Wait for the terminated process to be fully cleaned up
            
            # 6. Exit condition if all work is done
            if not active_processes and run_count >= max_runs:
                break
            
    finally:
        print("\n--- Terminating any remaining worker processes ---")
        for run_id, (proc, start_time) in active_processes.items():
            if proc.is_alive():
                print(f"Terminating run {run_id}...")
                proc.terminate()
                proc.join()

    print("\n--- Overall Summary ---")
    if best_energy != float('inf'):
        print(f"Best energy found: {best_energy}")
        # Use 3D-aware printing
        conf_str = ' -> '.join([str(c).replace(' ', '') for c in best_conformation])
        print(f"Best conformation coordinates: {conf_str}")
    else:
        print(f"Execution stopped. No valid conformation was found within the {max_runs} run limit.")
    print(f"A total of {completed_runs} runs were completed (including timeouts/errors).")
