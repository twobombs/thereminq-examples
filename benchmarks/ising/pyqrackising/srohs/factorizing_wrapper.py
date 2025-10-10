# this is a wrapper that leverages ising models to factor
# derived from: https://arxiv.org/abs/2301.06738
# evolved specs https://arxiv.org/pdf/2506.16799
# gemini25 - second version
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
    N_to_factor = model_data['N']

    while True:
        num_total_vars = max_cut_graph.shape[0]
        # FIX: Changed 'is_base_maxcut_gpu' to 'is_combo_maxcut_gpu'
        result_tuple = spin_glass_solver(
            max_cut_graph,
            quality=quality,
            is_combo_maxcut_gpu=use_gpu
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
        
        # --- New, more robust validation logic ---
        is_invalid = False
        reason = ""

        if p == 0 or q == 0:
            is_invalid = True
            reason = "zero factor"
        elif p == 1 or q == 1:
            is_invalid = True
            reason = "trivial factor of 1"
        elif (N_to_factor % 2 != 0) and (p % 2 == 0 or q % 2 == 0):
            is_invalid = True
            reason = "even factor for an odd target N"

        if is_invalid:
            print(f"(Worker {os.getpid()} got an invalid solution ({p}, {q}) - {reason}. Retrying...) ", end="", flush=True)
        else:
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

def create_carry_propagation_hubo(N, nq):
    """
    Creates a HUBO for integer factorization based on the binary multiplication
    with carry propagation method described in arXiv:2506.16799v1.

    This function implements the HUBO model from Eq. (16) of the paper.
    It creates variables for the bits of factors p and q, and for the carry bits
    C_i at each position of the binary multiplication.

    Variable Mapping:
    - p_j (bits of factor p) are mapped to integer variables `j`.
    - q_k (bits of factor q) are mapped to integer variables `nq + k`.
    - C_i (carry bits) are mapped to integer variables `2*nq + i`.
    """
    n = nq - 1
    hubo = defaultdict(float)

    r_bits = [int(bit) for bit in bin(N)[2:]][::-1]
    required_len = 2 * n + 2
    r_bits.extend([0] * (required_len - len(r_bits)))

    def p_var(j): return j
    def q_var(k): return nq + k
    def c_var(i): return 2 * nq + i

    def add_squared_expression(expression, constant):
        terms_list = list(expression.items())

        for vars_tuple, coeff in terms_list:
            key = tuple(sorted(vars_tuple))
            hubo[key] += coeff**2
            hubo[key] += 2 * coeff * constant

        for i in range(len(terms_list)):
            for j in range(i + 1, len(terms_list)):
                vars1, coeff1 = terms_list[i]
                vars2, coeff2 = terms_list[j]
                combined_vars = tuple(sorted(list(set(vars1 + vars2))))
                hubo[combined_vars] += 2 * coeff1 * coeff2
        
        hubo[()] += constant**2

    # Term for i=0
    expr_i0 = {
        (p_var(0), q_var(0)): 1.0,
        (c_var(0),): -2.0
    }
    add_squared_expression(expr_i0, -r_bits[0])

    # Summation term from i=1 to 2n
    for i in range(1, 2 * n + 1):
        expression = defaultdict(float)
        
        for j in range(nq):
            for k in range(nq):
                if j + k == i:
                    key = tuple(sorted((p_var(j), q_var(k))))
                    expression[key] += 1.0
        
        expression[(c_var(i - 1),)] += 1.0
        expression[(c_var(i),)] -= 2.0
        
        add_squared_expression(expression, -r_bits[i])

    # Final carry term
    expr_final_c = {(c_var(2 * n),): 1.0}
    add_squared_expression(expr_final_c, -r_bits[2 * n + 1])

    # Final subtraction term (offset)
    offset = sum(val**2 for val in r_bits)
    hubo[()] -= offset
    
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
    max_runs = 1

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
    print(f"Stopping after {max_runs} iterations if no solution is found.")

    # --- Model Preparation (Done once) ---
    hubo = create_carry_propagation_hubo(N_to_factor, num_qubits_per_factor)
    
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
    completed_runs = 0

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
            while not solution_found and completed_runs < max_runs:
                while len(active_processes) < total_workers and run_count < max_runs:
                    run_count += 1
                    gpu_to_use = gpu_id_list[(run_count - 1) % len(gpu_id_list)]
                    proc = multiprocessing.Process(target=run_solver_on_gpu, args=(gpu_to_use, result_queue, run_count, model_data))
                    proc.start()
                    active_processes[proc.pid] = proc

                res_id, p_res, q_res = result_queue.get()
                completed_runs += 1
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
        num_workers = max(1, int(total_threads * 0.75))
        print(f"System has {total_threads} threads. Using {num_workers} concurrent CPU worker(s).")

        initializer_args = (model_data,)
        
        with multiprocessing.Pool(processes=num_workers, initializer=init_worker_cpu, initargs=initializer_args) as pool:
            run_iterator = range(1, max_runs + 1)
            for run_id, p_res, q_res in pool.imap_unordered(run_solver_on_cpu, run_iterator):
                completed_runs += 1
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
        print(f"Execution stopped. The optimal solution was not found within the {max_runs} run limit.")
    print(f"A total of {completed_runs} runs were completed.")
