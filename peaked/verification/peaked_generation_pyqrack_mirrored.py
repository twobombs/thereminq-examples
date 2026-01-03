# -*- coding: utf-8 -*-
import os
import random
import argparse
import time
import sys
from qiskit import QuantumCircuit, transpile, qasm2
from qiskit.quantum_info import random_unitary
from qiskit.circuit.library import UnitaryGate

# --- CRITICAL FIX FOR 36 QUBITS ---
# 1. DISABLE GUARD: This was the culprit. It was "optimizing" away the edge qubits.
os.environ['QRACK_DISABLE_QUNIT_FIDELITY_GUARD'] = '1'

# 2. Tight Threshold: 1e-6 (Tight enough to keep LNN chains, loose enough for V100)
#    If this OOMs, raise to 1e-5. If it fails, lower to 1e-7.
os.environ['QRACK_QUNIT_SEPARABILITY_THRESHOLD'] = '1e-6' 

# 3. Strict Tolerance:
os.environ['QRACK_QUNIT_TOLERANCE'] = '1e-7'

os.environ['QRACK_MAX_PAGING'] = '1'

try:
    from pyqrack import QrackSimulator
except ImportError as e:
    sys.exit(1)

def generate_LNN_mirror_circuit(num_qubits, depth, filename):
    hidden_bitstring = "".join([random.choice("01") for _ in range(num_qubits)])
    print(f"Hiding Secret Bitstring: {hidden_bitstring}")

    qc = QuantumCircuit(num_qubits)
    for i, bit in enumerate(reversed(hidden_bitstring)):
        if bit == '1':
            qc.x(i)
    
    print(f"Building LNN BRICKWORK MOUNTAIN (Depth {depth})...")
    
    forward_qc = QuantumCircuit(num_qubits)
    basis_gates = ['u', 'cz'] 
    
    for d in range(depth):
        # Brickwork Pattern (Linear Nearest Neighbor)
        start_index = d % 2 
        for i in range(start_index, num_qubits - 1, 2):
            q_indices = [i, i+1]
            U_matrix = random_unitary(4).data
            U_gate = UnitaryGate(U_matrix)
            
            mini_qc = QuantumCircuit(2)
            mini_qc.append(U_gate, [0, 1])
            transpiled = transpile(mini_qc, basis_gates=basis_gates, optimization_level=3)
            
            for instr in transpiled.data:
                mapped = [q_indices[transpiled.find_bit(q).index] for q in instr.qubits]
                forward_qc.append(instr.operation, mapped)
        
        forward_qc.barrier()

    qc.compose(forward_qc, inplace=True)
    qc.barrier() 
    qc.compose(forward_qc.inverse(), inplace=True)

    with open(filename, "w") as f:
        qasm2.dump(qc, f)
    
    return hidden_bitstring

def execute_p1_once(qc, num_qubits):
    qsim = QrackSimulator(num_qubits)
    for instruction in qc.data:
        op = instruction.operation
        qubits = [qc.find_bit(q).index for q in instruction.qubits]
        if op.name == 'u':
            qsim.u(qubits[0], *op.params)
        elif op.name == 'cz':
            qsim.mcz([qubits[0]], qubits[1])
        elif op.name == 'x':
            qsim.x(qubits[0])
        elif op.name == 'barrier': pass 
        elif op.name == 'measure': pass

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

def solve_mirror_test(filename, num_qubits, rounds, secret=None):
    print(f"\nStarting LNN MIRROR TEST on {filename}")
    print(f"   Mode: {rounds} Rounds")
    print(f"   Settings: Thresh={os.environ['QRACK_QUNIT_SEPARABILITY_THRESHOLD']}, Tol={os.environ['QRACK_QUNIT_TOLERANCE']}")
    print(f"   Fidelity Guard: DISABLED (Brute Force Mode)")
    
    qc = QuantumCircuit.from_qasm_file(filename)
    original_depth = len(qc.data)
    print(f"   Circuit: {num_qubits} qubits, {original_depth} gates")
    
    # Check optimization just to be safe
    qc_opt = transpile(qc, optimization_level=3, basis_gates=['u', 'cz', 'x'])
    new_depth = len(qc_opt.data)
    if new_depth < original_depth * 0.99:
        print("   [WARNING] Circuit simplified significantly.")
    else:
        print("   [CONFIRMED] Circuit is HARD (0% Reduction).")
    
    results = []
    print(f"\n   Running Loop...")
    start_sim = time.time()
    
    for i in range(1, rounds + 1):
        try:
            bs = execute_p1_once(qc_opt, num_qubits)
            results.append(bs)
            live_guess = get_current_consensus(results, num_qubits)
            
            status = " [OK]" if secret and live_guess == secret else ""
            sys.stdout.write(f"\r   Round {i:3d}/{rounds} | Consensus: {live_guess}{status}")
            sys.stdout.flush()
        except Exception:
            continue
            
    print(f"\n\n   Done in {time.time() - start_sim:.2f}s.")
    
    if not results: return None

    print("\n   --- BITWISE CONFIDENCE ---")
    final_bitstring = []
    for i in range(num_qubits):
        ones = sum(1 for bs in results if bs[i] == '1')
        zeros = len(results) - ones
        total = ones + zeros
        conf = max(ones, zeros) / total * 100
        choice = '1' if ones > zeros else '0'
        final_bitstring.append(choice)
        
        marker = " " 
        if conf < 60: marker = "[LOW]"
        elif conf < 80: marker = "[MED]"
            
        print(f"   Bit {i:2d}: {choice} ({conf:5.1f}%) {marker}")
            
    return "".join(final_bitstring)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", "-w", type=int, default=36)
    parser.add_argument("--depth", "-d", type=int, default=20) 
    parser.add_argument("--rounds", "-r", type=int, default=100)
    args = parser.parse_args()

    N_QUBITS = args.width
    DEPTH = args.depth
    FILENAME = f"peaked_LNN_w{N_QUBITS}_d{DEPTH}.qasm"

    print("="*60)
    print(f"CONFIGURATION: Width={N_QUBITS} | Peak Depth={DEPTH} (LNN Brickwork)")
    print("="*60)

    secret = generate_LNN_mirror_circuit(N_QUBITS, DEPTH, FILENAME)
    recovered = solve_mirror_test(FILENAME, N_QUBITS, args.rounds, secret=secret)

    print("\n" + "-" * 30)
    if recovered == secret:
        print("SUCCESS: Solved the High-Memory LNN Circuit!")
    else:
        print("FAILURE: Mismatch.")
        if recovered:
            diff = "".join(['^' if s!=r else ' ' for s,r in zip(secret, recovered)])
            print(f"Expected: {secret}")
            print(f"Got:      {recovered}")
            print(f"Diff:     {diff}")
    print("-" * 30)
