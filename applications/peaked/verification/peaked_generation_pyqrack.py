# -*- coding: utf-8 -*-
import os
import random
import argparse
import time
import sys
from qiskit import QuantumCircuit, transpile, qasm2
from qiskit.quantum_info import random_unitary
from qiskit.circuit.library import UnitaryGate

# --- SETTINGS (Testing your "High Memory" Hypothesis) ---
# We stick to your successful conservative settings. 
# If this crashes (OOM), we know we need to raise the Threshold.
os.environ['QRACK_QUNIT_SEPARABILITY_THRESHOLD'] = '0.001' 
os.environ['QRACK_QUNIT_TOLERANCE'] = '1e-6'
os.environ['QRACK_DISABLE_QUNIT_FIDELITY_GUARD'] = '1'
os.environ['QRACK_MAX_PAGING'] = '1'

# Robust PyQrack Import
try:
    from pyqrack import QrackSimulator
except ImportError as e:
    sys.exit(1)

def generate_full_density_circuit(num_qubits, depth, filename):
    hidden_bitstring = "".join([random.choice("01") for _ in range(num_qubits)])
    print(f"Hiding Secret Bitstring: {hidden_bitstring}")

    qc = QuantumCircuit(num_qubits)
    # 1. Encode Secret
    for i, bit in enumerate(reversed(hidden_bitstring)):
        if bit == '1':
            qc.x(i)
    
    print(f"Applying {depth} FULL-DENSITY layers (Parallel Gates)...")
    basis_gates = ['u', 'cz'] 
    
    # 2. Generate Dense Layers
    total_gates = 0
    for d in range(depth):
        # Create a random permutation of qubits to pair them up
        shuffled_indices = random.sample(range(num_qubits), num_qubits)
        
        # Apply N//2 gates in parallel (One Layer)
        for i in range(0, num_qubits - 1, 2):
            q_indices = [shuffled_indices[i], shuffled_indices[i+1]]
            
            U_matrix = random_unitary(4).data
            U_gate = UnitaryGate(U_matrix)
            
            # Decompose & Apply U
            mini_qc_U = QuantumCircuit(2)
            mini_qc_U.append(U_gate, [0, 1])
            transpiled_U = transpile(mini_qc_U, basis_gates=basis_gates, optimization_level=3)
            
            for instr in transpiled_U.data:
                mapped_qubits = [q_indices[transpiled_U.find_bit(q).index] for q in instr.qubits]
                qc.append(instr.operation, mapped_qubits)
            
            # Barrier inside the pair (Barrier obfuscation)
            qc.barrier(*q_indices)

            # Decompose & Apply U_dagger
            mini_qc_Ud = QuantumCircuit(2)
            mini_qc_Ud.append(U_gate.inverse(), [0, 1])
            transpiled_Ud = transpile(mini_qc_Ud, basis_gates=basis_gates, optimization_level=3)

            for instr in transpiled_Ud.data:
                mapped_qubits = [q_indices[transpiled_Ud.find_bit(q).index] for q in instr.qubits]
                qc.append(instr.operation, mapped_qubits)

            # Barrier after the pair
            qc.barrier(*q_indices)
            total_gates += 2 # Count logical blocks

    print(f"Generated {total_gates} logical obfuscation blocks.")
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
        elif op.name == 'barrier' or op.name == 'measure':
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

def solve_full_density(filename, num_qubits, rounds, secret=None):
    print(f"\nStarting FULL DENSITY TEST on {filename}")
    print(f"   Mode: {rounds} Rounds")
    print(f"   Settings: Thresh={os.environ['QRACK_QUNIT_SEPARABILITY_THRESHOLD']}, Tol={os.environ['QRACK_QUNIT_TOLERANCE']}")
    
    qc = QuantumCircuit.from_qasm_file(filename)
    original_depth = len(qc.data)
    print(f"   Circuit: {num_qubits} qubits, {original_depth} gates")

    print("   Running Qiskit Optimization (Level 3)...")
    start_opt = time.time()
    qc_opt = transpile(qc, optimization_level=3, basis_gates=['u', 'cz', 'x'])
    new_depth = len(qc_opt.data)
    reduction = 100 * (1 - new_depth / original_depth) if original_depth > 0 else 0
    print(f"   New Depth: {new_depth} gates (Reduction: {reduction:.1f}%)")
    
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
            # If we OOM here, the exception loop catches it, but the user sees it hang/fail
            continue
            
    print(f"\n\n   Done in {time.time() - start_sim:.2f}s.")
    
    if not results: return None

    # Report
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
    parser.add_argument("--width", "-w", type=int, default=34) 
    parser.add_argument("--depth", "-d", type=int, default=50) # Reduced default depth because parallel layers are heavy
    parser.add_argument("--rounds", "-r", type=int, default=100)
    args = parser.parse_args()

    # NOTE: Depth 50 here = 50 * (34/2) = 850 dense blocks ~ 5000+ gates
    # This roughly matches the complexity of Depth 200 in the linear version.
    
    N_QUBITS = args.width
    DEPTH = args.depth
    FILENAME = f"peaked_full_w{N_QUBITS}_d{DEPTH}.qasm"

    print("="*60)
    print(f"CONFIGURATION: Width={N_QUBITS} | Depth={DEPTH} (Full Density) | Rounds={args.rounds}")
    print("="*60)

    secret = generate_full_density_circuit(N_QUBITS, DEPTH, FILENAME)
    recovered = solve_full_density(FILENAME, N_QUBITS, args.rounds, secret=secret)

    print("\n" + "-" * 30)
    if recovered == secret:
        print("SUCCESS: Recovered secret from FULL DENSITY circuit!")
    else:
        print("FAILURE: Mismatch.")
        if recovered:
            diff = "".join(['^' if s!=r else ' ' for s,r in zip(secret, recovered)])
            print(f"Expected: {secret}")
            print(f"Got:      {recovered}")
            print(f"Diff:     {diff}")
    print("-" * 30)
