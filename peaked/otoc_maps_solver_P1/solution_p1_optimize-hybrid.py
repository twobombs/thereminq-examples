import multiprocessing as mp
import os
import re
import time
import numpy as np
from collections import Counter

# --- Configuration ---
# CRITICAL UPDATE: Scanning the "Fast Scrambler" regime (t < 1.0)
SCAN_RANGE = np.arange(0.01, 0.55, 0.01) 
CHUNK_SIZE = 5000
TOTAL_SHOTS_PER_POINT = 200_000

# --- Helper Functions ---
def get_hardware_config():
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        gpus = []
        for p in platforms:
            gpus.extend(p.get_devices(device_type=cl.device_type.GPU))
        num_gpus = len(gpus)
    except: num_gpus = 0
    num_cpus = mp.cpu_count()
    return num_gpus, num_cpus

def get_circuit_params(qasm_file):
    try:
        with open(qasm_file, 'r') as f: content = f.read()
    except: return 36, 1.54, 31
    n = int(re.search(r"qreg\s+q\[(\d+)\];", content).group(1))
    thetas = [float(t) for t in re.findall(r"u\((-?\d+\.?\d*e?-?\d*)", content)]
    avg_theta = np.mean([abs(t) for t in thetas]) if thetas else 1.57
    pairs = set()
    for q1, q2 in re.findall(r"cz\s+q\[(\d+)\],q\[(\d+)\]", content):
        pairs.add(tuple(sorted((int(q1), int(q2)))))
    z = int(round((2 * len(pairs)) / n))
    return n, avg_theta, z

def hybrid_worker(worker_id, device_type, device_id, task_queue, result_queue, params):
    if device_type == 'GPU':
        os.environ['PYOPENCL_CTX'] = f'0:{device_id}'
    else:
        os.environ['PYOPENCL_CTX'] = '' 
        os.environ['pyqrack_backend'] = 'cpu'

    try:
        from pyqrackising import generate_otoc_samples
    except ImportError: return

    while True:
        try:
            task = task_queue.get(timeout=0.1)
        except: break

        t_val, chunk_shots = task
        local_params = params.copy()
        local_params['t'] = t_val
        local_params['shots'] = chunk_shots
        
        try:
            raw_samples = generate_otoc_samples(**local_params)
            n = params['n_qubits']
            bitstrings = [format(s, f'0{n}b') for s in raw_samples]
            valid = [b for b in bitstrings if b != '0'*n and b != '1'*n]
            result_queue.put((t_val, Counter(valid)))
        except:
            result_queue.put((t_val, Counter()))

# --- Main Execution ---
if __name__ == "__main__":
    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    qasm_file = "P1_little_dimple.qasm"
    n, theta, z = get_circuit_params(qasm_file)
    num_gpus, num_cpus = get_hardware_config()
    
    # Strategy: Aggressive Parallelism
    num_gpu_workers = num_gpus * 2
    num_cpu_workers = max(0, num_cpus - num_gpu_workers)
    if num_gpus == 0: num_gpu_workers, num_cpu_workers = 0, num_cpus

    print(f"--- Micro-Time Scan (t=0.05 to 0.55) ---")
    print(f"Hypothesis: High connectivity (z={z}) compresses scrambling time.")
    
    manager = mp.Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    
    base_params = {
        "n_qubits": n, "J": -1.0, "h": 1.61803398875, "z": z,
        "theta": theta,
        "pauli_strings": ['I'*i + 'X' + 'I'*(n-1-i) for i in range(n)],
        "is_orbifold": True
    }
    
    total_chunks = 0
    chunks_per_point = TOTAL_SHOTS_PER_POINT // CHUNK_SIZE
    for t_val in SCAN_RANGE:
        for _ in range(chunks_per_point):
            task_queue.put((t_val, CHUNK_SIZE))
            total_chunks += 1

    processes = []
    for i in range(num_gpu_workers):
        gpu_id = i % num_gpus if num_gpus > 0 else 0
        p = mp.Process(target=hybrid_worker, args=(i, 'GPU', gpu_id, task_queue, result_queue, base_params))
        p.start()
        processes.append(p)
    for i in range(num_cpu_workers):
        p = mp.Process(target=hybrid_worker, args=(i+num_gpu_workers, 'CPU', 0, task_queue, result_queue, base_params))
        p.start()
        processes.append(p)

    results_by_t = {t: Counter() for t in SCAN_RANGE}
    processed_count = 0
    last_update = time.time()
    
    # Faster timeout since chunks are small and fast
    STALL_TIMEOUT = 10 

    print(f"Scanning {total_chunks} chunks...")

    while processed_count < total_chunks:
        try:
            t_ret, counts = result_queue.get(timeout=0.5)
            results_by_t[t_ret].update(counts)
            processed_count += 1
            last_update = time.time()
            if processed_count % 50 == 0:
                print(f"\rProgress: {processed_count}/{total_chunks}...", end="")
        except:
            if time.time() - last_update > STALL_TIMEOUT:
                print("\nStall detected. Stopping early.")
                break
    
    for p in processes:
        if p.is_alive(): p.terminate()

    print("\n\n--- Micro-Scan Results ---")
    print(f"{'Time (t)':<10} | {'Signal':<10} | {'Dominant Bitstring'}")
    print("-" * 65)
    
    best_t = 0
    max_signal = 0
    best_string = ""
    
    for t_val in sorted(SCAN_RANGE):
        counter = results_by_t[t_val]
        if not counter: continue
        winner, freq = counter.most_common(1)[0]
        
        # Highlight significant peaks
        marker = " <--- PEAK?" if freq > 10 else ""
        print(f"{t_val:<10.2f} | {freq:<10} | {winner}{marker}")
        
        if freq > max_signal:
            max_signal = freq
            best_t = t_val
            best_string = winner

    print("-" * 65)
    print(f"OPTIMAL t: {best_t}")
    print(f"WINNING BITSTRING: {best_string}")
    
    if max_signal > 5:
        with open("p1_micro_solution.txt", "w") as f:
            f.write(best_string)
        print("Solution saved to p1_micro_solution.txt")
    else:
        print("Signal still weak. Physics model parameters (J vs h) may need rescaling.")
