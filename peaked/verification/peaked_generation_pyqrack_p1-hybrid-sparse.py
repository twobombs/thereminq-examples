# -*- coding: utf-8 -*-
import os
import time
import sys
import argparse
import ctypes
from qiskit import QuantumCircuit, transpile

# --- PYQRACK SETUP ---
try:
    from pyqrack import QrackSimulator, Qrack
except ImportError:
    print("Error: PyQrack not installed.")
    sys.exit(1)

# --- CTYPES CALLBACK SURGERY ---

# 1. Define Callback Type
# Signature: void (*)(unsigned long long, double, double)
DumpCallbackType = ctypes.CFUNCTYPE(None, ctypes.c_ulonglong, ctypes.c_double, ctypes.c_double)

# 2. Global Storage
sparse_collector = {}

# 3. The Python Function (Unclamped)
def my_dump_callback(idx, r, i):
    # Calculate probability
    prob = r*r + i*i
    
    # THRESHOLD: 1e-7
    # We filter out only the absolute noise floor (< 0.0000001).
    # Since you have 32GB RAM, we do not need a hard limit on the count.
    if prob > 1.0e-8: 
        sparse_collector[idx] = prob

# 4. Create Callback Instance
c_dump_callback = DumpCallbackType(my_dump_callback)


def solve_challenge_patched(filename, rounds, target_string=None):
    print(f"SOLVING {filename}")
    print(f"    Mode: Patched Sparse Oracle (Direct C++ Binding)")
    print(f"    Specs: Threadripper/32GB RAM Mode (No Limits)")
    print(f"    Filter: > 1e-8 (Noise Floor)")
    
    # --- LOAD ---
    print("    Loading QASM file...")
    t0 = time.time()
    qc = QuantumCircuit.from_qasm_file(filename)
    qc.remove_final_measurements() 
    
    orig_ops = len(qc.data)
    print(f"    Loaded in {time.time()-t0:.2f}s")
    
    # --- OPTIMIZE ---
    print("    Running Qiskit Optimization (Level 3)...")
    t0 = time.time()
    qc_opt = transpile(qc, optimization_level=3, basis_gates=['u', 'cz', 'x'])
    new_ops = len(qc_opt.data)
    print(f"    Optimization complete in {time.time()-t0:.2f}s (Gates: {orig_ops} -> {new_ops})")

    # --- SIMULATION ---
    num_qubits = qc_opt.num_qubits
    qsim = QrackSimulator(num_qubits)
    q_map = {q: i for i, q in enumerate(qc_opt.qubits)}
    
    print(f"    Executing {new_ops} gates on PyQrack...")
    
    for instruction in qc_opt.data:
        op = instruction.operation
        qubits = [q_map[q] for q in instruction.qubits]
        if op.name == 'u':
            qsim.u(qubits[0], *op.params)
        elif op.name == 'cz':
            qsim.mcz([qubits[0]], qubits[1])
        elif op.name == 'x':
            qsim.x(qubits[0])

    # --- ORACLE INSPECTION (DIRECT BINDING) ---
    print("\n    Invoking Direct C++ Dump...")
    
    sparse_collector.clear()
    start_scan = time.time()
    
    # Apply the ctypes argtype fix to bypass TypeError
    raw_dump_fn = Qrack.qrack_lib.Dump
    raw_dump_fn.argtypes = [ctypes.c_ulonglong, DumpCallbackType]
    raw_dump_fn.restype = None
    
    try:
        raw_dump_fn(qsim.sid, c_dump_callback)
    except Exception as e:
        print(f"    CRITICAL ERROR: {e}")
        sys.exit(1)

    scan_time = time.time() - start_scan
    count_found = len(sparse_collector)
    
    print(f"    Scan complete in {scan_time:.2f}s. Found {count_found} significant states.")
    
    if count_found == 0:
        print("    WARNING: No states found. The Peak is fainter than 1e-8.")
        return

    # --- RESULT ANALYSIS ---
    print("\n" + "="*60)
    print("THEORETICAL PEAK (SPARSE ORACLE)")
    print("="*60)
    
    # Sort by probability
    sorted_states = sorted(sparse_collector.items(), key=lambda x: x[1], reverse=True)
    
    winner_str = ""
    for i, (idx, prob) in enumerate(sorted_states[:10]):
        bitstr = format(idx, f'0{num_qubits}b')
        print(f"RANK #{i+1}")
        print(f"BITSTRING:   {bitstr}")
        print(f"PROBABILITY: {prob:.9f}")
        print("-" * 30)
        
        if i == 0:
            winner_str = bitstr

    print("="*60)
    
    # --- TARGET VERIFICATION ---
    if target_string:
        print("\nTARGET VERIFICATION")
        print(f"Expected: {target_string}")
        print(f"Found:    {winner_str}")
        if target_string == winner_str:
            print(">>> MATCH CONFIRMED <<<")
        else:
            print(">>> MISMATCH <<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", default="little_dimple_n32_d200.qasm")
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--target", type=str, help="Expected bitstring")
    
    args = parser.parse_args()
    solve_challenge_patched(args.filename, args.rounds, args.target)
