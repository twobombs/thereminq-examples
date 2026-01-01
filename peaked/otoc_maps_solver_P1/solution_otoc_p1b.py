# -*- coding: utf-8 -*-
import multiprocessing
import re
import time
import gc
import numpy as np
from collections import Counter
from pyqrackising import generate_otoc_samples

# --- Configuration ---
# The stable attractor found in the micro-scan (Target for verification)
TARGET_BITSTRING = "000000000110111111111111111111000110" 
TOTAL_SHOTS = 10_000_000
BATCH_SIZE = 50_000 

# Define the Stability Plateau
# We sample t continuously in this range to ensure physical robustness
T_MIN = 0.30
T_MAX = 0.45

# --- 1. Circuit & Physics ---
def get_physics_params(qasm_file):
    try:
        with open(qasm_file, 'r') as f: content = f.read()
    except: return 36, 1.54, 31

    n_match = re.search(r"qreg\s+q\[(\d+)\];", content)
    n = int(n_match.group(1)) if n_match else 36

    thetas = [float(t) for t in re.findall(r"u\((-?\d+\.?\d*e?-?\d*)", content)]
    avg_theta = np.mean([abs(t) for t in thetas]) if thetas else 1.57
    
    pairs = set()
    for q1, q2 in re.findall(r"cz\s+q\[(\d+)\],q\[(\d+)\]", content):
        pairs.add(tuple(sorted((int(q1), int(q2)))))
    z = int(round((2 * len(pairs)) / n)) if n > 0 else 4
    
    return n, avg_theta, z

# --- 2. Optimized Worker ---
def worker_task(args):
    """
    Runs simulation with time-averaging over the designated plateau.
    """
    params, shots = args
    local_params = params.copy()
    local_params['shots'] = shots
    
    # CRITICAL: Sample 't' continuously from the Stability Plateau
    # This averages results across the robust window [0.30, 0.45]
    local_params['t'] = np.random.uniform(T_MIN, T_MAX)
    
    try:
        # Run simulation
        raw_samples = generate_otoc_samples(**local_params)
        
        # Convert to bitstrings
        n = params['n_qubits']
        bitstrings = [format(s, f'0{n}b') for s in raw_samples]
        
        # Pre-filter thermal noise (00..0 and 11..1)
        valid = [b for b in bitstrings if b != '0'*n and b != '1'*n]
        return Counter(valid)
        
    except Exception:
        return Counter()

# --- 3. Main Loop ---
if __name__ == "__main__":
    qasm_file = "P1_little_dimple.qasm"
    n, theta, z = get_physics_params(qasm_file)
    
    # Physics Params
    params = {
        "n_qubits": n, "J": -1.0, "h": 1.61803398875, "z": z,
        "theta": theta,
        # 't' is chosen dynamically in the worker
        "pauli_strings": ['I'*i + 'X' + 'I'*(n-1-i) for i in range(n)],
        "is_orbifold": True
    }

    num_cpus = multiprocessing.cpu_count()
    num_tasks = TOTAL_SHOTS // BATCH_SIZE
    tasks = [(params, BATCH_SIZE) for _ in range(num_tasks)]
    
    print(f"--- Robust Stability Scan: {TOTAL_SHOTS:,} shots ---")
    print(f"Target Area: t in [{T_MIN}, {T_MAX}] (Ensemble Averaging)")
    print(f"Target String: {TARGET_BITSTRING}")
    print(f"Workers: {num_cpus}")
    
    global_counter = Counter()
    total_processed = 0
    start = time.time()
    
    # maxtasksperchild=10 cleans up memory leaks from the C++ extension
    with multiprocessing.Pool(processes=num_cpus, maxtasksperchild=10) as pool:
        
        for i, batch_counts in enumerate(pool.imap_unordered(worker_task, tasks)):
            if not batch_counts: continue
            
            global_counter.update(batch_counts)
            total_processed += sum(batch_counts.values())
            
            # Periodic status update
            if i % num_cpus == 0:
                elapsed = time.time() - start
                if global_counter:
                    leader, lead_count = global_counter.most_common(1)[0]
                    target_count = global_counter[TARGET_BITSTRING]
                    
                    # Manual GC to keep main process lean
                    if i % (num_cpus * 5) == 0: gc.collect()

                    match_status = "MATCH" if leader == TARGET_BITSTRING else "DIFF"
                    print(f"\rShots: {total_processed:,} | Leader: {lead_count} ({match_status}) | Target: {target_count} | {elapsed:.0f}s", end="")

    print(f"\n\n--- Final Result ---")
    if global_counter:
        # Analysis
        top_10 = global_counter.most_common(10)
        winner, count = top_10[0]
        
        print(f"Winner: {winner}")
        print(f"Count:  {count}")
        
        if winner == TARGET_BITSTRING:
            print("VERIFICATION SUCCESSFUL: The Dimple is robust across the time window.")
        else:
            print("NEW LEADER FOUND: The plateau might favor a different state.")
            
        print("\n--- Top 10 Candidates ---")
        print(f"{'Rank':<5} | {'Bitstring':<38} | {'Count':<10} | {'Rel. Prob':<10}")
        print("-" * 75)
        
        total_hits = sum(global_counter.values())
        
        for rank, (bitstring, freq) in enumerate(top_10, 1):
            rel_prob = freq / total_hits
            marker = " <--- TARGET" if bitstring == TARGET_BITSTRING else ""
            print(f"{rank:<5} | {bitstring} | {freq:<10} | {rel_prob:.6%}{marker}")

        # Save results
        with open("p1_final_verified.txt", "w") as f:
            f.write(winner)
            
        with open("p1_top10_candidates.txt", "w") as f:
            for rank, (bs, freq) in enumerate(top_10, 1):
                f.write(f"{rank},{bs},{freq}\n")
        
        print("\nResults saved to 'p1_final_verified.txt' and 'p1_top10_candidates.txt'")
    else:
        print("No valid signals found.")
