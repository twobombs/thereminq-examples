# this is a wrapper that leverages ising models to factor
# derived from: https://arxiv.org/abs/2301.06738
# gemini25 - initial version

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

# --- Worker Functions ---

def run_solver_on_gpu(gpu_id_str, result_queue, run_id, model_data):
    """Initializes a solver on a specific GPU and runs one instance."""
    try:
        warnings.filterwarnings('ignore', 'divide by zero encountered in divide', RuntimeWarning)
        
        os.environ['PYOPENCL_CTX'] = gpu_id_str
        from pyqrackising import spin_glass_solver

        p, q = solve_and_decode(model_data, use_gpu=True)
        result_queue.put((run_id, p, q))

    except Exception:
        print(f"--- ERROR in worker process on GPU {gpu_id_str} (run ID {run_id}) ---")
        traceback.print_exc()
        result_queue.put((run_id, None, None))

def init_worker_cpu(shared_model_data):
    """Initializes global variables for each CPU worker process."""
    global worker_model_data
    worker_model_data = shared_model_data

def run_solver_on_cpu(run_id):
    """A single, independent solver instance run by a CPU worker process."""
    try:
        warnings.filterwarnings('ignore', 'divide by zero encountered in divide', RuntimeWarning)

        p, q = solve_and_decode(worker_model_data, use_gpu=False)
        return (run_id, p, q)
    except Exception:
        print(f"--- ERROR in CPU worker process (run ID {run_id}) ---")
        traceback.print_exc()
        return (run_id, None, None)

# --- Shared Logic ---

def solve_and_decode(model_data, use_gpu):
    """
    A single run of the solver and result decoding.
    Will now loop internally until a potentially valid (non-zero, non-trivial) 
    result is found.
    """
    from pyqrackising import spin_glass_solver
    # Unpack the model data
    max_cut_graph = model_data['graph']
    num_qubits_per_factor = model_data['nq']
    label_to_index = model_data['labels']
    quality = model_data['quality']
    # MODIFICATION: Unpack N to check if it's odd or even
    N_to_factor = model_data['N']

    # MODIFICATION: Use a `while True` loop that we explicitly break from
    while True:
        num_total_vars = max_cut_graph.shape[0]
        result_tuple = spin_glass_solver(
            max_cut_graph,
            quality=quality,
            is_base_maxcut_gpu=use_gpu
        )
        bitstring = result_tuple[0]
        spins = {i: 1 - 2 * int(bit) for i, bit in enumerate(bitstring)}
        ancilla_node_idx = num_total_vars - 1
        ancilla_spin = spins.get(ancilla_node_idx, 1)
        normalized_spins = {var: spin * ancilla_spin for var, spin in spins.items()}
        binary_solution = {i: (s + 1) // 2 for i, s in normalized_spins.items()}
        
        # Reset p and q before recalculating
        p, q = 0, 0
        for i in range(num_qubits_per_factor):
            if binary_solution.get(label_to_index.get(i), 0) == 1:
                p += 2**i
        for i in range(num_qubits_per_factor):
            if binary_solution.get(label_to_index.get(num_qubits_per_factor + i), 0) == 1:
                q += 2**i
        
        # --- MODIFICATION: New, more robust validation logic ---
        is_invalid = False
        reason = ""

        # Condition 1: Check for zero factors (solver failure)
        if p == 0 or q == 0:
            is_invalid = True
            reason = "zero factor"
        # Condition 2: Check for trivial factors of 1
        elif p == 1 or q == 1:
            is_invalid = True
            reason = "trivial factor of 1"
        # Condition 3: If N is odd, its factors must also be odd.
        # (p % 2 == 0) is true if p is even. We check this only if N is odd.
        elif (N_to_factor % 2 != 0) and (p % 2 == 0 or q % 2 == 0):
            is_invalid = True
            reason = "even factor for an odd target N"

        if is_invalid:
            print(f"(Worker {os.getpid()} got an invalid solution ({p}, {q}) - {reason}. Retrying...) ", end="", flush=True)
            # The 'continue' is implied, the loop will repeat
        else:
            # If we pass all checks, break the loop to return the result
            break
            
    return p, q

def process_and_log_result(run_id, p_res, q_res, N_to_factor, log_filename):
    """
    Processes a result from any worker, logs it, and returns True if successful.
    """
    if p_res is None:
        return False, (0, 0) # Signal failure

    factors = tuple(sorted((p_res, q_res)))
    product = factors[0] * factors[1]
    cost = (product - N_to_factor)**2
    
    with open(log_filename, 'a') as f:
        f.write(f"{run_id},{factors[0]},{factors[1]},{cost}\n")

    if product == N_to_factor:
        print("\nStatus: Success! Optimal solution found.")
        print(f"Result from Run {run_id}: p={factors[0]}, q={factors[1]}")
        print(f"Verification: {factors[0]} * {factors[1]} = {product} (Target N={N_to_factor})")
        return True, factors
    
    return False, factors


# --- (The following functions are used only in the main process) ---
def create_hubo_for_factorization(N, NQ):
    hubo = defaultdict(float)
    for l1 in range(NQ):
        for l2 in range(NQ):
            p_qubit, q_qubit = l1, NQ + l2
            coeff = 2**(2 * (l1 + l2)) - 2**(l1 + l2 + 1) * N
            hubo[tuple(sorted((p_qubit, q_qubit)))] += coeff
    for l3 in range(NQ):
        for l1 in range(NQ - 1):
            for l2 in range(l1 + 1, NQ):
                coeff = 2**(l1 + l2 + 2*l3 + 1)
                hubo[tuple(sorted((l1, l2, NQ + l3)))] += coeff
                hubo[tuple(sorted((l3, NQ + l1, NQ + l2)))] += coeff
    for l3 in range(NQ - 1):
        for l4 in range(l3 + 1, NQ):
            for l1 in range(NQ - 1):
                for l2 in range(l1 + 1, NQ):
                    coeff = 2**(l1 + l2 + l3 + l4 + 2)
                    hubo[tuple(sorted((l1, l2, NQ + l3, NQ + l4)))] += coeff
    return hubo

def convert_ising_to_maxcut(h, J, num_vars, label_to_index):
    graph_size = num_vars + 1
    ancilla_node_idx = num_vars
    W = np.zeros((graph_size, graph_size))
    for (i_label, j_label), coupling in J.items():
        i, j = label_to_index[i_label], label_to_index[j_label]
        W[i, j] = coupling
        W[j, i] = coupling
    for i_label, bias in h.items():
        i = label_to_index[i_label]
        W[i, ancilla_node_idx] = -bias
        W[ancilla_node_idx, i] = -bias
    return W

# --- Main execution block ---
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    # --- Default Values ---
    N_to_factor = 15
    quality = 4
    workers_per_gpu = 1
    use_gpu = True
    max_runs = 100 # *** NEW: Iteration limit ***

    # --- Parse Command-Line Arguments ---
    if len(sys.argv) > 1:
        try:
            N_to_factor = int(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid number '{sys.argv[1]}'. Please provide an integer.")
            sys.exit(1)

    if len(sys.argv) > 2:
        try:
            quality = int(sys.argv[2])
        except ValueError:
            print(f"Error: Invalid quality '{sys.argv[2]}'. Please provide an integer.")
            sys.exit(1)
    
    if len(sys.argv) > 3 and sys.argv[3].lower() == 'cpu':
        use_gpu = False

    device_suffix = "_cpu" if not use_gpu else ""
    log_filename = f"factor_landscape_{N_to_factor}_q{quality}{device_suffix}.log"
    print(f"Logging run results to: {log_filename}")
    if not os.path.exists(log_filename):
        with open(log_filename, 'w') as f:
            f.write("run,p,q,cost\n")

    num_qubits_per_factor = (N_to_factor.bit_length() // 2) + 1
    
    print(f"\nAttempting to factor N = {N_to_factor}")
    print(f"Using solver quality: {quality}")
    print(f"Calculated qubits per factor = {num_qubits_per_factor}")
    print(f"Stopping after {max_runs} iterations if no solution is found.") # *** NEW: Info message ***

    # --- Model Preparation (Done once) ---
    hubo = create_hubo_for_factorization(N_to_factor, num_qubits_per_factor)
    max_coeff = max(abs(c) for c in hubo.values()) if hubo else 1.0
    bqm = dimod.make_quadratic(hubo, strength=max_coeff * 2, vartype='BINARY')
    label_to_index = {label: i for i, label in enumerate(bqm.variables)}
    num_bqm_vars = len(label_to_index)
    h, J, offset = bqm.to_ising()
    max_cut_graph = convert_ising_to_maxcut(h, J, num_bqm_vars, label_to_index)
    max_abs_val = np.max(np.abs(max_cut_graph))
    if max_abs_val > 0:
        max_cut_graph /= max_abs_val
    
    print(f"Total variables after HUBO->QUBO reduction: {num_bqm_vars}")
    
    model_data = {
        'graph': max_cut_graph, 'nq': num_qubits_per_factor,
        'labels': label_to_index, 'quality': quality, 'N': N_to_factor
    }
    
    solution_found = False
    best_solution = (0, 0)
    completed_runs = 0 # *** NEW: Counter for completed runs ***

    # --- Conditional execution path (GPU vs CPU) ---
    if use_gpu:
        print("\n--- Running on available GPUs ---")
        gpu_id_list = []
        platforms = cl.get_platforms()
        for p_idx, p in enumerate(platforms):
            try:
                for d_idx, d in enumerate(p.get_devices(device_type=cl.device_type.GPU)):
                    gpu_id_list.append(f"{p_idx}:{d_idx}")
                    print(f"  Found GPU: '{d.name}' (ID: {p_idx}:{d_idx})")
            except cl.Error:
                continue
        
        if not gpu_id_list:
            print("Error: No OpenCL-enabled GPUs found.")
            sys.exit(1)

        total_workers = len(gpu_id_list) * workers_per_gpu
        print(f"Configured for {workers_per_gpu} worker(s) per GPU. Total: {total_workers}")
        
        result_queue = multiprocessing.Queue()
        active_processes = {}
        run_count = 0
        
        try:
            # *** MODIFIED: Loop condition now includes run limit ***
            while not solution_found and completed_runs < max_runs:
                while len(active_processes) < total_workers and run_count < max_runs:
                    run_count += 1
                    gpu_to_use = gpu_id_list[(run_count - 1) % len(gpu_id_list)]
                    proc = multiprocessing.Process(target=run_solver_on_gpu, args=(gpu_to_use, result_queue, run_count, model_data))
                    proc.start()
                    active_processes[proc.pid] = proc

                res_id, p_res, q_res = result_queue.get()
                completed_runs += 1 # *** NEW: Increment completed run counter ***
                solution_found, factors = process_and_log_result(res_id, p_res, q_res, N_to_factor, log_filename)
                if solution_found:
                    best_solution = factors

                finished_pids = [pid for pid, proc in active_processes.items() if not proc.is_alive()]
                for pid in finished_pids:
                    del active_processes[pid]
        
        finally:
            print("\n--- Terminating GPU worker processes ---")
            for pid, proc in active_processes.items():
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
    
    else: # CPU Path
        print("\n--- Running on CPU ---")
        total_threads = os.cpu_count()
        num_workers = max(1, int(total_threads * 0.25))
        print(f"System has {total_threads} threads. Using {num_workers} concurrent CPU worker(s).")

        initializer_args = (model_data,)
        
        with multiprocessing.Pool(processes=num_workers, initializer=init_worker_cpu, initargs=initializer_args) as pool:
            # *** MODIFIED: Use a finite range instead of an infinite count ***
            run_iterator = range(1, max_runs + 1)
            for run_id, p_res, q_res in pool.imap_unordered(run_solver_on_cpu, run_iterator):
                completed_runs += 1 # *** NEW: Increment completed run counter ***
                solution_found, factors = process_and_log_result(run_id, p_res, q_res, N_to_factor, log_filename)
                if solution_found:
                    best_solution = factors
                    pool.terminate()
                    break
    
    # --- Final Summary ---
    print("\n--- Overall Summary ---")
    if solution_found:
        print(f"Optimal solution p={best_solution[0]}, q={best_solution[1]} was found.")
    else:
        # *** MODIFIED: More informative message on failure ***
        print(f"Execution stopped. The optimal solution was not found within the {max_runs} run limit.")
    print(f"A total of {completed_runs} runs were completed.")
