# this is a wrapper that leverages ising models to factor
# derived from: https://arxiv.org/abs/2301.06738
# evolved specs https://arxiv.org/pdf/2506.16799
# gemini25 - fixed version v3 (CLI landscape tweak)
# gemini30 - iterative update

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
    Loops internally until a potentially valid (non-zero, non-trivial) result is found.
    """
    from pyqrackising import spin_glass_solver
    
    max_cut_graph = model_data['graph']
    num_qubits_per_factor = model_data['nq']
    label_to_index = model_data['labels']
    quality = model_data['quality']
    N_to_factor = model_data['N']

    # Safety counter to prevent infinite loops if the landscape is too hard
    attempts = 0
    max_internal_retries = 5 

    while attempts < max_internal_retries:
        attempts += 1
        num_total_vars = max_cut_graph.shape[0]
        
        # Call solver with correct parameter name
        result_tuple = spin_glass_solver(
            max_cut_graph,
            quality=quality,
            is_maxcut_gpu=use_gpu 
        )
        
        bitstring = result_tuple[0]
        spins = {i: 1 - 2 * int(bit) for i, bit in enumerate(bitstring)}
        ancilla_node_idx = num_total_vars - 1
        ancilla_spin = spins.get(ancilla_node_idx, 1)
        normalized_spins = {var: spin * ancilla_spin for var, spin in spins.items()}
        binary_solution = {i: (s + 1) // 2 for i, s in normalized_spins.items()}
        
        # Decode p and q
        p, q = 0, 0
        for i in range(num_qubits_per_factor):
            if binary_solution.get(label_to_index.get(i), 0) == 1:
                p += 2**i
        for i in range(num_qubits_per_factor):
            if binary_solution.get(label_to_index.get(num_qubits_per_factor + i), 0) == 1:
                q += 2**i
        
        # Debug Print: View raw solver output before validation
        print(f" [DEBUG Worker {os.getpid()}] Found candidate: p={p}, q={q} (product={p*q})")

        # Validation
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
            reason = "even factor for odd N"

        if is_invalid:
            # If we hit max retries, return the last invalid result so it can be logged
            if attempts >= max_internal_retries:
                break
            continue # Retry immediately
        else:
            break # Valid solution found
            
    return p, q

def process_and_log_result(run_id, p_res, q_res, N_to_factor, log_filename):
    """
    Processes a result from any worker, logs it, and returns True if successful.
    """
    if p_res is None:
        return False, (0, 0)

    factors = tuple(sorted((p_res, q_res)))
    product = factors[0] * factors[1]
    cost = (product - N_to_factor)**2
    
    with open(log_filename, 'a') as f:
        f.write(f"{run_id},{factors[0]},{factors[1]},{cost}\n")

    if product == N_to_factor and factors[0] != 1 and factors[1] != 1:
        print(f"\nStatus: Success! Optimal solution found in Run {run_id}.")
        print(f"Factors: {factors[0]} * {factors[1]} = {product}")
        return True, factors
    
    return False, factors


# --- (HUBO Helper Functions) ---

def create_carry_propagation_hubo(N, nq):
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

    # i=0
    expr_i0 = {(p_var(0), q_var(0)): 1.0, (c_var(0),): -2.0}
    add_squared_expression(expr_i0, -r_bits[0])

    # i=1 to 2n
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

    # Final carry
    expr_final_c = {(c_var(2 * n),): 1.0}
    add_squared_expression(expr_final_c, -r_bits[2 * n + 1])

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
    N_to_factor = 91
    quality = 6
    workers_per_gpu = 2
    use_gpu = True
    max_runs = 1
    strength_multiplier = 1.5  # Default landscape tweak

    # --- Parse Arguments ---
    if len(sys.argv) > 1:
        try:
            N_to_factor = int(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid number '{sys.argv[1]}'")
            sys.exit(1)

    if len(sys.argv) > 2:
        try:
            quality = int(sys.argv[2])
        except ValueError:
            print(f"Error: Invalid quality '{sys.argv[2]}'")
            sys.exit(1)
    
    # Flexible parsing for optional args (device or strength) starting from index 3
    for arg in sys.argv[3:]:
        if arg.lower() == 'cpu':
            use_gpu = False
        else:
            try:
                strength_multiplier = float(arg)
            except ValueError:
                print(f"Warning: Ignoring unrecognized argument '{arg}'")

    device_suffix = "_cpu" if not use_gpu else ""
    log_filename = f"factor_landscape_{N_to_factor}_q{quality}{device_suffix}.log"
    print(f"Logging run results to: {log_filename}")
    if not os.path.exists(log_filename):
        with open(log_filename, 'w') as f:
            f.write("run,p,q,cost\n")

    num_qubits_per_factor = (N_to_factor.bit_length() // 2) + 1
    
    print(f"\nAttempting to factor N = {N_to_factor}")
    print(f"Using solver quality: {quality}")
    print(f"Penalty Strength Multiplier: {strength_multiplier}")
    print(f"Calculated qubits per factor = {num_qubits_per_factor}")

    # --- Model Preparation ---
    hubo = create_carry_propagation_hubo(N_to_factor, num_qubits_per_factor)
    
    max_coeff = max(abs(c) for c in hubo.values()) if hubo else 1.0
    
    # Applying the strength multiplier from CLI
    bqm = dimod.make_quadratic(hubo, strength=max_coeff * strength_multiplier, vartype='BINARY')
    
    label_to_index = {label: i for i, label in enumerate(bqm.variables)}
    num_bqm_vars = len(label_to_index)
    h, J, offset = bqm.to_ising()
    max_cut_graph = convert_ising_to_maxcut(h, J, num_bqm_vars, label_to_index)
    
    # Normalize
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

    if use_gpu:
        print("\n--- Running on available GPUs ---")
        gpu_id_list = []
        try:
            platforms = cl.get_platforms()
            for p_idx, p in enumerate(platforms):
                for d_idx, d in enumerate(p.get_devices(device_type=cl.device_type.GPU)):
                    gpu_id_list.append(f"{p_idx}:{d_idx}")
                    print(f"  Found GPU: '{d.name}' (ID: {p_idx}:{d_idx})")
        except Exception as e:
            print(f"OpenCL Error: {e}")
        
        if not gpu_id_list:
            print("Error: No OpenCL-enabled GPUs found.")
            sys.exit(1)

        total_workers = len(gpu_id_list) * workers_per_gpu
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

                if not active_processes:
                    break

                res_id, p_res, q_res = result_queue.get()
                completed_runs += 1
                solution_found, factors = process_and_log_result(res_id, p_res, q_res, N_to_factor, log_filename)
                if solution_found:
                    best_solution = factors

                # Cleanup finished processes
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
        num_workers = max(1, int(os.cpu_count() * 0.75))
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
    
    print("\n--- Overall Summary ---")
    if solution_found:
        print(f"Optimal solution p={best_solution[0]}, q={best_solution[1]} was found.")
    else:
        print(f"Execution stopped. The optimal solution was not found within the {max_runs} run limit.")
    print(f"A total of {completed_runs} runs were completed.")
