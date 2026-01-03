import numpy as np
from qiskit import QuantumCircuit
import random

def build_little_dimple_circuit(num_qubits, depth=650, seed=None):
    """
    Rebuilds the 'P1_little_dimple' circuit structure at a variable bitwidth.
    
    The original circuit consists of layers of random U gates and CZ gates applied
    to disjoint pairs of qubits. This script preserves that 'texture' and depth
    while allowing the number of qubits to change.
    
    Args:
        num_qubits (int): The width of the circuit (original was 36).
        depth (int): The number of entangling layers to apply (original approx 650).
        seed (int): Random seed for reproducibility.
    
    Returns:
        QuantumCircuit: The generated Qiskit circuit.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    qc = QuantumCircuit(num_qubits)
    
    # The 'Little Dimple' pattern observed:
    # Blocks of interaction: U(qA), U(qB), CZ(qA, qB), U(qA), U(qB)
    # applied to random disjoint pairs in layers.
    
    print(f"Generating 'Little Dimple' style circuit with {num_qubits} qubits and depth {depth}...")

    qubit_indices = list(range(num_qubits))

    for layer in range(depth):
        # In each layer, we shuffle qubits to form random disjoint pairs
        # This mimics the random connectivity observed in the file (e.g. CZ(1,3), CZ(0,8)...)
        random.shuffle(qubit_indices)
        
        # Iterate over pairs
        for i in range(0, num_qubits - 1, 2):
            q_a = qubit_indices[i]
            q_b = qubit_indices[i+1]
            
            # --- Pre-entanglement Rotations ---
            # Generate random parameters for U gates: theta, phi, lambda
            params_a_pre = np.random.uniform(0, 2 * np.pi, 3)
            params_b_pre = np.random.uniform(0, 2 * np.pi, 3)
            
            qc.u(*params_a_pre, q_a)
            qc.u(*params_b_pre, q_b)
            
            # --- Entanglement ---
            qc.cz(q_a, q_b)
            
            # --- Post-entanglement Rotations ---
            params_a_post = np.random.uniform(0, 2 * np.pi, 3)
            params_b_post = np.random.uniform(0, 2 * np.pi, 3)
            
            qc.u(*params_a_post, q_a)
            qc.u(*params_b_post, q_b)

    return qc

if __name__ == "__main__":
    # Example usage:
    # To match the original file: num_qubits=36
    # To verify variable bitwidth: change num_qubits to any integer (e.g., 10)
    
    target_width = 36  # Change this to desired variable bitwidth
    circuit = build_little_dimple_circuit(num_qubits=target_width)
    
    # Output statistics to verify it matches expectations
    print(f"\nCircuit Generated: {circuit.name}")
    print(f"Width: {circuit.num_qubits}")
    print(f"Total Depth: {circuit.depth()}")
    print(f"Gate Counts: {circuit.count_ops()}")
    
    # Optional: Export to QASM to verify format matches the input file
    # print(circuit.qasm())
