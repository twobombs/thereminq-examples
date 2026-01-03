# -*- coding: utf-8 -*-
import os
import time
import sys
import argparse
from collections import Counter
from qiskit import QuantumCircuit, transpile

# --- USER VERIFIED SETTINGS ---
# Confirmed to work on Dense 24-qubit circuits with >90% confidence.
os.environ['QRACK_QUNIT_SEPARABILITY_THRESHOLD'] = '0.1464466'
os.environ['QRACK_QUNIT_TOLERANCE'] = '1e-5'  # Tightened based on your success
os.environ['QRACK_DISABLE_QUNIT_FIDELITY_GUARD'] = '1'
os.environ['QRACK_MAX_PAGING'] = '1'

# Robust PyQrack Import
try:
    from pyqrack import QrackSimulator
except ImportError as e:
    sys.exit(1)

def execute_p1_once(qc, num_qubits):
    qsim = QrackSimulator(num_qubits)
    
    # Fast Map
    for instruction in qc.data:
        op = instruction.operation
        qubits = [qc.find_bit(q).index for q in instruction.qubits]
        
        if op.name == 'u':
            qsim.u(qubits[0], *op.params)
        elif op.name == 'cz':
            qsim.mcz([qubits[0]], qubits[1])
        elif op.name == 'x':
            qsim.x(qubits[0])
        elif op.name == 'barrier': 
            pass 
        elif op.name == 'measure':
            pass
        else:
            pass 

    measurements = qsim.measure_shots(list(range(num_qubits)), 1)
    return format(measurements[0], f'0{num_qubits}b')

def get_current_consensus(results, num_qubits):
    if not results: return "?" * num_qubits
    temp_bits = []
    for i in range(num_qubits):
        ones = sum(1 for bs in results if bs[i] == '1')
        zeros = len(results) - ones
        temp_bits.append('1' if ones > zeros else '0')
    return "".join(temp_bits)

def solve_challenge_cinematic(filename, rounds):
    print(f"SOLVING {filename} via Bitwise Consensus")
    print(f"   Settings: Thresh={os.environ['QRACK_QUNIT_SEPARABILITY_THRESHOLD']}, Tol={os.environ['QRACK_QUNIT_TOLERANCE']}")
    
    # 1. Load
    print("   Loading QASM file...")
    qc = QuantumCircuit.from_qasm_file(filename)
    num_qubits = qc.num_qubits
    original_depth = len(qc.data)
    print(f"   Circuit: {num_qubits} qubits, {original_depth} gates")

    # 2. Optimize (Standard Check)
    print("   Running Qiskit Optimization (Level 3)...")
    start_opt = time.time()
    qc_opt = transpile(qc, optimization_level=3, basis_gates=['u', 'cz', 'x'])
    new_depth = len(qc_opt.data)
    reduction = 100 * (1 - new_depth / original_depth) if original_depth > 0 else 0
    print(f"   Optimization Complete in {time.time() - start_opt:.2f}s.")
    print(f"   New Depth: {new_depth} gates (Reduction: {reduction:.1f}%)")
    
    if reduction < 1.0:
        print("   [Confirmed] Circuit is DENSE (Matches your test conditions).")

    print(f"   Target:  {rounds} Rounds\n")
    
    results = []
    start_sim = time.time()
    
    # --- CINEMATIC LOOP ---
    for i in range(1, rounds + 1):
        try:
            bs = execute_p1_once(qc_opt, num_qubits)
            results.append(bs)
            
            # Update Live Guess
            live_guess = get_current_consensus(results, num_qubits)
            sys.stdout.write(f"\r   Round {i:3d}/{rounds} | Consensus: {live_guess}")
            sys.stdout.flush()
            
        except Exception:
            # Skip failed shots (likely OOM protection kicking in)
            continue
            
    total_time = time.time() - start_sim
    print(f"\n\n   Done in {total_time:.2f}s.")
    
    if not results:
        print("CRITICAL FAILURE: No rounds completed.")
        return

    # --- FINAL REPORT ---
    print("\n   --- BITWISE CONFIDENCE REPORT ---")
    print("   Bit | 0s  | 1s  | Conf% | Choice")
    print("   ----------------------------------")
    
    final_bitstring = []
    
    for i in range(num_qubits):
        ones_count = sum(1 for bs in results if bs[i] == '1')
        zeros_count = len(results) - ones_count
        
        total = ones_count + zeros_count
        confidence = max(ones_count, zeros_count) / total * 100 if total > 0 else 0
        choice = '1' if ones_count > zeros_count else '0'
        
        final_bitstring.append(choice)
        
        marker = " "
        if confidence < 60: marker = "[LOW]"
        elif confidence < 80: marker = "[MED]"
        
        print(f"   {i:3d} | {zeros_count:3d} | {ones_count:3d} | {confidence:5.1f}% |   {choice}   {marker}")
            
    winner = "".join(final_bitstring)
    
    print("   ----------------------------------")
    print("\n" + "="*60)
    print(f"FINAL BITSTRING: {winner}")
    print("="*60)
    print(f"REVERSED (LSB):  {winner[::-1]}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", default="P1_little_dimple.qasm")
    parser.add_argument("--rounds", type=int, default=100)
    args = parser.parse_args()

    solve_challenge_cinematic(args.filename, args.rounds)
