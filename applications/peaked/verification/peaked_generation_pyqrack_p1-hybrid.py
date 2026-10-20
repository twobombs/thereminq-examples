# -*- coding: utf-8 -*-
import os
import time
import sys
import argparse
import numpy as np
from qiskit import QuantumCircuit, transpile

# --- CONFIGURATION ---
# Disable aggressive memory compression to prevent hangs on Random Circuits
if 'QRACK_QUNIT_SEPARABILITY_THRESHOLD' in os.environ:
    del os.environ['QRACK_QUNIT_SEPARABILITY_THRESHOLD']
os.environ['QRACK_MAX_PAGING'] = '1'

try:
    from pyqrack import QrackSimulator
except ImportError as e:
    print("Error: PyQrack not installed.")
    sys.exit(1)

def solve_challenge_oracle(filename, rounds, target_string=None):
    print(f"SOLVING {filename}")
    print(f"    Mode: Full State Vector Inspection (Oracle Mode)")
    
    # 1. LOAD
    print("    Loading QASM file...")
    t0 = time.time()
    qc = QuantumCircuit.from_qasm_file(filename)
    qc.remove_final_measurements() 
    
    # Capture Original Stats
    orig_ops = len(qc.data)
    orig_depth = qc.depth()
    
    print(f"    Loaded in {time.time()-t0:.2f}s")
    
    # 2. OPTIMIZE & REPORT
    print("    Running Qiskit Optimization (Level 3)...")
    t0 = time.time()
    
    # Transpile to merge rotations and cancel inverses
    qc_opt = transpile(qc, optimization_level=3, basis_gates=['u', 'cz', 'x'])
    
    # Capture New Stats
    new_ops = len(qc_opt.data)
    new_depth = qc_opt.depth()
    
    # Calculate Compression metrics
    ops_reduction = (1 - new_ops / orig_ops) * 100
    depth_reduction = (1 - new_depth / orig_depth) * 100
    
    print(f"    Optimization complete in {time.time()-t0:.2f}s")
    print("\n    --- COMPRESSION REPORT ---")
    print(f"    Original Gates:  {orig_ops}")
    print(f"    Optimized Gates: {new_ops}")
    print(f"    GATE REDUCTION:  {ops_reduction:.2f}%  <-- Simulation Speedup")
    print(f"    --------------------------")
    print(f"    Original Depth:  {orig_depth}")
    print(f"    Optimized Depth: {new_depth}")
    print(f"    DEPTH REDUCTION: {depth_reduction:.2f}%")
    print(f"    --------------------------\n")
    
    # 3. SIMULATION
    start_sim = time.time()
    num_qubits = qc_opt.num_qubits
    qsim = QrackSimulator(num_qubits)
    q_map = {q: i for i, q in enumerate(qc_opt.qubits)}
    
    # Execute Gates
    total_gates = len(qc_opt.data)
    print(f"    Executing {total_gates} gates on PyQrack...")
    
    for i, instruction in enumerate(qc_opt.data):
        op = instruction.operation
        qubits = [q_map[q] for q in instruction.qubits]
        
        if op.name == 'u':
            qsim.u(qubits[0], *op.params)
        elif op.name == 'cz':
            qsim.mcz([qubits[0]], qubits[1])
        elif op.name == 'x':
            qsim.x(qubits[0])
            
        if i % 5000 == 0 and i > 0:
            sys.stdout.write(f"\r    Progress: {i}/{total_gates} gates...")
            sys.stdout.flush()

    sys.stdout.write(f"\r    Progress: {total_gates}/{total_gates} gates... Done.\n")
    
    # 4. ORACLE INSPECTION
    print("    Retrieving Probability Vector (GPU -> CPU)...")
    
    # Retrieve all probabilities
    all_qubits = list(range(num_qubits))
    probs = np.array(qsim.prob_all(all_qubits), dtype=np.float64)
    
    print("    Scanning State Vector for Peak...")
    max_idx = np.argmax(probs)
    max_prob = probs[max_idx]
    
    winner_str = format(max_idx, f'0{num_qubits}b')
    
    sim_time = time.time() - start_sim
    print(f"    Analysis complete in {sim_time:.2f}s.")

    print("\n" + "="*60)
    print("THEORETICAL PEAK (ARGMAX)")
    print("="*60)
    print(f"BITSTRING:   {winner_str}")
    print(f"PROBABILITY: {max_prob:.8f}")
    print("="*60)
    
    # --- TARGET VERIFICATION LOGIC ---
    if target_string:
        print("\n" + "="*60)
        print("TARGET VERIFICATION")
        print("="*60)
        print(f"EXPECTED: {target_string}")
        print(f"FOUND:    {winner_str}")
        
        if target_string == winner_str:
            print("RESULT:   SUCCESS - MATCH CONFIRMED")
        else:
            print("RESULT:   FAILURE - MISMATCH")
        print("="*60)

    # Optional: Still do sampling to show the noise contrast
    print(f"\n    Performing {rounds} Shot Sampling (Noise Check)...")
    
    # Normalize
    total_p = np.sum(probs)
    probs /= total_p
    
    shot_indices = np.random.choice(np.arange(len(probs)), size=rounds, p=probs)
    results = [format(idx, f'0{num_qubits}b') for idx in shot_indices]
    
    print(f"    Sampled {rounds} shots. Do any match the peak?")
    matches = sum(1 for res in results if res == winner_str)
    print(f"    Matches found: {matches}/{rounds}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", default="P1_steered_26q_200d.qasm")
    parser.add_argument("--rounds", type=int, default=10000000)
    parser.add_argument("--target", type=str, help="The expected bitstring to verify against")
    
    args = parser.parse_args()

    solve_challenge_oracle(args.filename, args.rounds, args.target)
