# -*- coding: utf-8 -*-
import argparse
import random
import numpy as np
import time

# Qiskit 1.0+ Imports
from qiskit import QuantumCircuit, qasm2
from qiskit.synthesis import OneQubitEulerDecomposer

# PyQrack for High-Performance Simulation
from pyqrack import QrackSimulator

def get_random_u_params():
    """Generates random theta, phi, lam for a U gate."""
    return np.random.uniform(0, 2 * np.pi, 3)

def _params_to_matrix(theta, phi, lam):
    """
    Manually reconstruct the U3 matrix from parameters.
    """
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    exp_phi = np.exp(1j * phi)
    exp_lam = np.exp(1j * lam)
    exp_phi_lam = np.exp(1j * (phi + lam))
    
    return np.array([
        [cos, -exp_lam * sin],
        [exp_phi * sin, exp_phi_lam * cos]
    ])

def get_natural_winner_pyqrack(qc, num_qubits):
    """
    Translates the Qiskit circuit directly to PyQrack commands
    to find the winning bitstring without using QASM text parsing.
    """
    ksim = QrackSimulator(num_qubits)
    
    # Create a map from Qiskit Qubit objects to integer indices
    q_map = {q: i for i, q in enumerate(qc.qubits)}
    
    for instruction in qc.data:
        op = instruction.operation
        name = op.name
        # Get integer indices for the qubits involved in this gate
        q_indices = [q_map[q] for q in instruction.qubits]
        
        if name == 'u':
            # PyQrack: u(qubit, theta, phi, lam)
            theta, phi, lam = map(float, op.params)
            ksim.u(q_indices[0], theta, phi, lam)
            
        elif name == 'cz':
            # PyQrack uses mcz(controls_list, target)
            ksim.mcz([q_indices[0]], q_indices[1])
            
    # Explicitly pass the list of all qubits to prob_all
    all_qubits = list(range(num_qubits))
    probs = ksim.prob_all(all_qubits)
    
    # Find the index of the maximum probability
    winner_idx = np.argmax(probs)
    return winner_idx, probs[winner_idx]

def build_and_steer_circuit(target_string, depth, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    num_qubits = len(target_string)
    qc = QuantumCircuit(num_qubits)
    
    final_layer_params = {} 
    
    print(f"1. Generating 'Little Dimple' backbone ({num_qubits} qubits, depth {depth})...")
    
    qubit_indices = list(range(num_qubits))

    # --- BUILD BACKBONE ---
    for layer in range(depth):
        random.shuffle(qubit_indices)
        for i in range(0, num_qubits - 1, 2):
            q_a = qubit_indices[i]
            q_b = qubit_indices[i+1]
            
            # Pre-entanglement
            qc.u(*get_random_u_params(), q_a)
            qc.u(*get_random_u_params(), q_b)
            
            # Entanglement
            qc.cz(q_a, q_b)
            
            # Post-entanglement (Save if last layer)
            params_a = get_random_u_params()
            params_b = get_random_u_params()
            
            if layer == depth - 1:
                final_layer_params[q_a] = params_a
                final_layer_params[q_b] = params_b
            else:
                qc.u(*params_a, q_a)
                qc.u(*params_b, q_b)

    for q in range(num_qubits):
        if q not in final_layer_params:
            final_layer_params[q] = get_random_u_params()

    # --- SIMULATION (OFFLOAD TO PYQRACK) ---
    print("2. Offloading simulation to PyQrack...")
    t0 = time.time()
    
    # Create temp circuit for simulation
    qc_sim = qc.copy()
    for q in range(num_qubits):
        qc_sim.u(*final_layer_params[q], q)
    
    # Use direct translation
    natural_winner_idx, win_prob = get_natural_winner_pyqrack(qc_sim, num_qubits)
    natural_winner_bin = format(natural_winner_idx, f'0{num_qubits}b')
    
    t1 = time.time()
    print(f"   Simulation complete in {t1-t0:.2f}s")
    print(f"   Natural Winner: {natural_winner_bin} (Prob: {win_prob:.6f})")
    print(f"   Target Winner:  {target_string}")

    # --- STEERING ---
    print("3. Applying steering corrections...")
    decomposer = OneQubitEulerDecomposer(basis='U3')
    
    for q in range(num_qubits):
        theta, phi, lam = final_layer_params[q]
        u_matrix = _params_to_matrix(theta, phi, lam)
        
        char_idx = (num_qubits - 1) - q
        
        target_bit = target_string[char_idx]
        natural_bit = natural_winner_bin[char_idx]
        
        if target_bit != natural_bit:
            x_mat = np.array([[0, 1], [1, 0]], dtype=complex)
            u_matrix = x_mat @ u_matrix
            
        u3_circuit = decomposer(u_matrix)
        new_params = u3_circuit.data[0].operation.params
        
        qc.u(*new_params, q)

    return qc

def generate_random_bitstring(length):
    return ''.join(random.choice('01') for _ in range(length))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild and Steer P1_little_dimple using PyQrack.")
    
    parser.add_argument("-q", "--qubits", type=int, default=24, 
                        help="Number of qubits. Default: 24")
    parser.add_argument("-d", "--depth", type=int, default=200, 
                        help="Circuit depth. Default: 200")
    
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed.")
    # Updated help text
    parser.add_argument("--qasm", action="store_true", 
                        help="Save QASM output to a file (P1_steered_{q}q_{d}d.qasm).")
    
    args = parser.parse_args()
    target_bitstring = generate_random_bitstring(args.qubits)

    print(f"--- Configuration ---")
    print(f"Qubits: {args.qubits}")
    print(f"Depth:  {args.depth}")
    print(f"Target: {target_bitstring}")
    print(f"---------------------")
    
    # 1. Build and Steer
    steered_circuit = build_and_steer_circuit(target_bitstring, args.depth, args.seed)
    
    # 2. Verify with PyQrack
    print("\n4. Verifying final result with PyQrack...")
    
    final_winner_idx, final_win_prob = get_natural_winner_pyqrack(steered_circuit, args.qubits)
    winner_str = format(final_winner_idx, f'0{args.qubits}b')
    
    print(f"\nFinal Result Peak: {winner_str}")
    print(f"Match Success:     {winner_str == target_bitstring}")

    if args.qasm:
        filename = f"P1_steered_{args.qubits}q_{args.depth}d.qasm"
        print(f"\n--- Saving QASM to {filename} ---")
        qasm_content = qasm2.dumps(steered_circuit)
        with open(filename, "w") as f:
            f.write(qasm_content)
        print("Done.")
