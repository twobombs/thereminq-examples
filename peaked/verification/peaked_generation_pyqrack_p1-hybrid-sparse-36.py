# -*- coding: utf-8 -*-
import os
import sys
import time
import argparse
import ctypes
import subprocess
import shutil
import numpy as np
from qiskit import QuantumCircuit, transpile

# --- CONFIGURATION for 32 QUBITS ---
sys.stdout.reconfigure(line_buffering=True)
# Force Dense Simulation (Crucial for correctness)
os.environ['QRACK_QUNIT_SEPARABILITY_THRESHOLD'] = '0.0'
# Disable Guards (Prevents random aborts)
os.environ['QRACK_DISABLE_QUNIT_FIDELITY_GUARD'] = '1'
# Enable Paging (Crucial for 32GB RAM)
os.environ['QRACK_MAX_PAGING'] = '1'

def check_gcc():
    if not shutil.which("gcc"):
        print("[CRITICAL] GCC not found. Run: apt-get install -y gcc")
        sys.exit(1)

def compile_final_filter():
    print("    [C-Engine] Compiling Optimized Atomic Filter...")
    
    # Standard Atomic Filter (No Debug Printfs to save speed)
    c_code = """
    #include <stdint.h>
    
    uint64_t* result_indices;
    double* result_probs;
    uint64_t* counter;
    uint64_t  max_hits;

    void setup_buffers(uint64_t* idx, double* prob, uint64_t* cnt, uint64_t max) {
        result_indices = idx;
        result_probs = prob;
        counter = cnt;
        max_hits = max;
    }

    // Return int to ensure ABI compatibility (bool)
    int filter_peak(uint64_t idx, double r, double i) {
        double prob = r*r + i*i;
        
        // Threshold 1.0e-9
        // For n32, background noise is ~1e-10, so this efficiently filters 99.9% of states.
        if (prob > 1.0e-9) {
            uint64_t c = __sync_fetch_and_add(counter, 1);
            if (c < max_hits) {
                result_indices[c] = idx;
                result_probs[c] = prob;
            }
        }
        return 1;
    }
    """
    
    with open("final_filter.c", "w") as f:
        f.write(c_code)
    
    try:
        subprocess.check_call(["gcc", "-O3", "-shared", "-fPIC", "-o", "final_filter.so", "final_filter.c"])
    except:
        sys.exit(1)
        
    return ctypes.CDLL("./final_filter.so")

def solve_n32_final(filename, target_string=None):
    print(f"SOLVING {filename}")
    print(f"    Mode: N32 ATOMIC (Paging On | Sep=0.0 | Threshold=1e-9)")
    
    check_gcc()
    c_lib = compile_final_filter()
    
    # 300 Million Buffer (Safe size)
    MAX_HITS = 300000000 
    print(f"    [Memory] Allocating 300M slots...")
    
    # Pinned memory arrays
    IdxArray = ctypes.c_uint64 * MAX_HITS
    ProbArray = ctypes.c_double * MAX_HITS
    idx_buf = IdxArray()
    prob_buf = ProbArray()
    counter = ctypes.c_uint64(0)
    
    c_lib.setup_buffers(
        ctypes.cast(idx_buf, ctypes.POINTER(ctypes.c_uint64)),
        ctypes.cast(prob_buf, ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(counter),
        ctypes.c_uint64(MAX_HITS)
    )

    try:
        from pyqrack import QrackSimulator, Qrack
    except ImportError:
        sys.exit(1)

    print(f"    [Qiskit] Loading {filename}...")
    qc = QuantumCircuit.from_qasm_file(filename)
    qc.remove_final_measurements() 
    qc_opt = transpile(qc, optimization_level=3, basis_gates=['u', 'cz', 'x'])
    new_ops = len(qc_opt.data)
    num_qubits = qc_opt.num_qubits
    
    print(f"    [PyQrack] Simulating {num_qubits} Qubits ({new_ops} gates)...")
    qsim = QrackSimulator(num_qubits)
    q_map = {q: i for i, q in enumerate(qc_opt.qubits)}
    
    t0 = time.time()
    for i, instruction in enumerate(qc_opt.data):
        op = instruction.operation
        qubits = [q_map[q] for q in instruction.qubits]
        if op.name == 'u': qsim.u(qubits[0], *op.params)
        elif op.name == 'cz': qsim.mcz([qubits[0]], qubits[1])
        elif op.name == 'x': qsim.x(qubits[0])
            
        if i % 2000 == 0 and i > 0:
            sys.stdout.write(f"\r    Progress: {i}/{new_ops}...")
    
    print("\n    [PyQrack] Normalizing (Syncing GPU)...")
    qsim.normalize()
    print(f"    Sim Time: {time.time()-t0:.2f}s")

    print("    [Scanner] Scanning State Vector (This may take 10-20 mins for 32q)...")
    t0 = time.time()
    
    # Define Callback signature (int return, u64, double, double args)
    CallbackType = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_ulonglong, ctypes.c_double, ctypes.c_double)
    c_callback = ctypes.cast(c_lib.filter_peak, CallbackType)
    
    raw_dump_fn = Qrack.qrack_lib.Dump
    raw_dump_fn.argtypes = [ctypes.c_ulonglong, CallbackType] 
    raw_dump_fn.restype = None
    
    try:
        raw_dump_fn(qsim.sid, c_callback)
    except Exception as e:
        print(f"    [ERROR] Scan failed: {e}")
        sys.exit(1)
        
    hits = counter.value
    print(f"    Scan Time: {time.time()-t0:.2f}s | Hits: {hits}")

    if hits == 0:
        print("    [Result] No peaks found.")
        return

    print("\nTOP CANDIDATES:")
    # Smart Sort (Only top valid hits)
    process_limit = min(hits, MAX_HITS)
    
    # Use Numpy views (zero copy)
    valid_probs = np.ctypeslib.as_array(prob_buf, shape=(process_limit,))
    valid_idxs = np.ctypeslib.as_array(idx_buf, shape=(process_limit,))
    
    k = min(10, process_limit)
    if process_limit > 1000000:
         print("    (Using fast partition sort...)")
         top_k_unsorted = np.argpartition(valid_probs, -k)[-k:]
         top_k_sorted = top_k_unsorted[np.argsort(valid_probs[top_k_unsorted])[::-1]]
    else:
         top_k_sorted = np.argsort(valid_probs)[::-1][:k]

    winner_str = ""
    for i in range(k):
        ptr = top_k_sorted[i]
        idx = valid_idxs[ptr]
        prob = valid_probs[ptr]
        
        bitstr = format(idx, f'0{num_qubits}b')
        print(f"RANK #{i+1}: {bitstr} | Prob: {prob:.11f}")
        if i == 0: winner_str = bitstr

    if target_string:
        print(f"\nMatch Check: {target_string == winner_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", default="little_dimple_n32_d200.qasm")
    parser.add_argument("--target", type=str)
    args = parser.parse_args()
    solve_n32_final(args.filename, args.target)
