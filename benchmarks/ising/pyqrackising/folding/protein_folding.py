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

# --- Worker Functions (Largely unchanged, adapted for clarity) ---

def run_solver_on_gpu(gpu_id_str, result_queue, run_id, model_data):
    """Initializes a solver on a specific GPU and runs one instance for protein folding."""
    try:
        warnings.filterwarnings('ignore', 'divide by zero encountered in divide', RuntimeWarning)
        
        os.environ['PYOPENCL_CTX'] = gpu_id_str
        from pyqrackising import spin_glass_solver

        energy, conformation = solve_and_decode_protein(model_data, use_gpu=True)
        result_queue.put((run_id, energy, conformation))

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

        energy, conformation = solve_and_decode_protein(worker_model_data, use_gpu=False)
        return (run_id, energy, conformation)
    except Exception:
        print(f"--- ERROR in CPU worker process (run ID {run_id}) ---")
        traceback.print_exc()
        return (run_id, None, None)

# --- Shared Logic (Modified for Protein Folding) ---

def solve_and_decode_protein(model_data, use_gpu):
    """
    A single run of the solver and result decoding for the HP protein folding model.
    """
    from pyqrackising import spin_glass_solver
    # Unpack the model data
    max_cut_graph = model_data['graph']
    label_to_index = model_data['labels']
    quality = model_data['quality']
    sequence_len = model_data['seq_len']

    # The solver returns the bitstring that minimizes the energy (maximizes the cut)
    result_tuple = spin_glass_solver(
        max_cut_graph,
        quality=quality,
        is_combo_maxcut_gpu=use_gpu
    )
    bitstring = result_tuple[0]
    energy = result_tuple[3] # spin_glass_solver returns energy as the 4th element

    spins = {i: 1 - 2 * int(bit) for i, bit in enumerate(bitstring)}
    ancilla_node_idx = max_cut_graph.shape[0] - 1
    ancilla_spin = spins.get(ancilla_node_idx, 1)
    normalized_spins = {var: spin * ancilla_spin for var, spin in spins.items()}
    binary_solution = {i: (s + 1) // 2 for i, s in normalized_spins.items()}
    
    # Decode the binary solution into a sequence of moves
    moves = []
    directions = ["U", "D", "L", "R"] # Up, Down, Left, Right
    for i in range(sequence_len):
        move_vars = [label_to_index.get(f'm_{i}_{d}', -1) for d in range(4)]
        
        chosen_move_idx = -1
        for idx, var_idx in enumerate(move_vars):
            if var_idx != -1 and binary_solution.get(var_idx, 0) == 1:
                chosen_move_idx = idx
                break
        
        moves.append(directions[chosen_move_idx] if chosen_move_idx != -1 else "N/A")

    return energy, moves

def process_and_log_result(run_id, energy, conformation, log_filename):
    """
    Processes a result, logs it, and updates the best-found solution.
    """
    if energy is None:
        return float('inf'), []

    with open(log_filename, 'a') as f:
        f.write(f"{run_id},{energy},{''.join(conformation)}\n")
    
    return energy, conformation

# --- Model Creation (NEW: For Protein Folding) ---

def create_protein_folding_qubo(sequence, grid_size=None):
    """
    Creates a QUBO for the 2D HP protein folding model.
    - sequence: A string of 'H' and 'P' (e.g., "HPHPPHH").
    - grid_size: The side length of the square lattice (e.g., len(sequence)).
    """
    n = len(sequence)
    if grid_size is None:
        grid_size = n
    
    qubo = defaultdict(float)
    
    # --- Variable Definition ---
    # x_i,j,k = 1 if amino acid 'i' is at grid position (j, k)
    def x(i, j, k):
        return (i * grid_size * grid_size) + (j * grid_size) + k

    # --- Constraints ---
    P_overlap = 10  # Strong penalty for two beads occupying the same site
    P_chain = 10    # Strong penalty for non-adjacent beads in the chain

    # 1. Each amino acid must occupy exactly one grid position
    for i in range(n):
        for j1 in range(grid_size):
            for k1 in range(grid_size):
                qubo[((x(i, j1, k1)), (x(i, j1, k1)))] -= 1
                for j2 in range(grid_size):
                    for k2 in range(grid_size):
                        if (j1, k1) < (j2, k2):
                            qubo[((x(i, j1, k1)), (x(i, j2, k2)))] += 2

    # 2. Each grid position can be occupied by at most one amino acid
    for j in range(grid_size):
        for k in range(grid_size):
            for i1 in range(n):
                for i2 in range(i1 + 1, n):
                    qubo[((x(i1, j, k)), (x(i2, j, k)))] += P_overlap
                    
    # 3. Chain connectivity constraint
    for i in range(n - 1):
        for j1 in range(grid_size):
            for k1 in range(grid_size):
                for j2 in range(grid_size):
                    for k2 in range(grid_size):
                        # Manhattan distance
                        dist = abs(j1 - j2) + abs(k1 - k2)
                        if dist != 1:
                           qubo[((x(i, j1, k1)), (x(i + 1, j2, k2)))] += P_chain

    # --- Objective Function: Maximize H-H contacts ---
    # A negative weight encourages the variables to be 1
    for i in range(n):
        for j in range(i + 2, n): # Non-adjacent pairs
            if sequence[i] == 'H' and sequence[j] == 'H':
                for j1 in range(grid_size):
                    for k1 in range(grid_size):
                        # Neighbors
                        neighbors = []
                        if j1 > 0: neighbors.append((j1 - 1, k1))
                        if j1 < grid_size - 1: neighbors.append((j1 + 1, k1))
                        if k1 > 0: neighbors.append((j1, k1 - 1))
                        if k1 < grid_size - 1: neighbors.append((j1, k1 + 1))
                        
                        for j2, k2 in neighbors:
                           qubo[((x(i, j1, k1)), (x(j, j2, k2)))] -= 1

    return qubo

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
    protein_sequence = "HPHPPHHPHPPHPHHPPHPH" # 20-mer example
    quality = 4
    workers_per_gpu = 1
    use_gpu = True
    max_runs = 100 

    # --- Parse Command-Line Arguments ---
    if len(sys.argv) > 1:
        protein_sequence = sys.argv[1]
    if len(sys.argv) > 2:
        quality = int(sys.argv[2])
    if len(sys.argv) > 3 and sys.argv[3].lower() == 'cpu':
        use_gpu = False

    device_suffix = "_cpu" if not use_gpu else ""
    log_filename = f"protein_folding_{protein_sequence}_q{quality}{device_suffix}.log"
    print(f"Logging run results to: {log_filename}")
    if not os.path.exists(log_filename):
        with open(log_filename, 'w') as f:
            f.write("run,energy,conformation\n")

    print(f"\nAttempting to fold sequence: {protein_sequence} (Length: {len(protein_sequence)})")
    print(f"Using solver quality: {quality}")
    print(f"Stopping after {max_runs} runs.")

    # --- Model Preparation (Done once) ---
    qubo = create_protein_folding_qubo(protein_sequence)
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
    
    label_to_index = {label: i for i, label in enumerate(bqm.variables)}
    num_bqm_vars = len(label_to_index)
    h, J, offset = bqm.to_ising()
    max_cut_graph = convert_ising_to_maxcut(h, J, num_bqm_vars, label_to_index)
    max_abs_val = np.max(np.abs(max_cut_graph))
    if max_abs_val > 0:
        max_cut_graph /= max_abs_val
    
    print(f"Total variables in QUBO model: {num_bqm_vars}")
    
    model_data = {
        'graph': max_cut_graph, 'seq_len': len(protein_sequence),
        'labels': label_to_index, 'quality': quality
    }
    
    best_energy = float('inf')
    best_conformation = []
    completed_runs = 0

    # --- Conditional execution path (GPU vs CPU) ---
    if use_gpu:
        # (GPU execution logic remains the same as the factorization script)
        print("\n--- Running on available GPUs ---")
        # ... (Omitted for brevity, identical to previous script)
    else: # CPU Path
        # (CPU execution logic remains the same as the factorization script)
        print("\n--- Running on CPU ---")
        # ... (Omitted for brevity, identical to previous script)

    # --- Final Summary ---
    print("\n--- Overall Summary ---")
    if best_energy != float('inf'):
        print(f"Best energy found: {best_energy}")
        print(f"Best conformation (moves): {' '.join(best_conformation)}")
    else:
        print(f"Execution stopped. No valid conformation was found within the {max_runs} run limit.")
    print(f"A total of {completed_runs} runs were completed.")

