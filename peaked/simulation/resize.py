import numpy as np
from qiskit import QuantumCircuit
import qiskit.qasm2  # Required for Qiskit 1.0+ export
import random
import argparse
import matplotlib.pyplot as plt

def build_little_dimple_circuit(num_qubits, depth=650, seed=None):
    """
    Rebuilds the 'P1_little_dimple' circuit structure at a variable bitwidth.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    qc = QuantumCircuit(num_qubits)
    
    print(f"Generating 'Little Dimple' style circuit with {num_qubits} qubits and depth {depth}...")

    qubit_indices = list(range(num_qubits))

    for layer in range(depth):
        # Shuffle qubits to form random disjoint pairs
        random.shuffle(qubit_indices)
        
        # Iterate over pairs
        for i in range(0, num_qubits - 1, 2):
            q_a = qubit_indices[i]
            q_b = qubit_indices[i+1]
            
            # --- Pre-entanglement Rotations ---
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

def visualize_connectivity(circuit):
    """
    Plots a heatmap showing how often each pair of qubits interacts.
    """
    print("Generating connectivity heatmap...")
    n = circuit.num_qubits
    interaction_matrix = np.zeros((n, n))
    
    # Iterate over all instructions to find 2-qubit gates
    for instruction in circuit.data:
        if len(instruction.qubits) == 2:
            # Robustly find qubit indices
            q0_idx = circuit.find_bit(instruction.qubits[0]).index
            q1_idx = circuit.find_bit(instruction.qubits[1]).index
            
            # Increment count (symmetric)
            interaction_matrix[q0_idx, q1_idx] += 1
            interaction_matrix[q1_idx, q0_idx] += 1

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.imshow(interaction_matrix, origin='lower', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Number of Interactions (CZ)')
    plt.title(f'Connectivity Heatmap (Width={n}, Depth={circuit.depth()})')
    plt.xlabel('Qubit Index')
    plt.ylabel('Qubit Index')
    
    # Save plot
    plot_filename = f"connectivity_n{n}.png"
    plt.savefig(plot_filename)
    print(f"Connectivity plot saved to '{plot_filename}'")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a variable-width 'Little Dimple' quantum circuit.")

    # CLI Options
    parser.add_argument("-n", "--qubits", type=int, default=36, 
                        help="Number of qubits/width (default: 36)")
    parser.add_argument("-d", "--depth", type=int, default=650, 
                        help="Number of entangling layers (default: 650)")
    parser.add_argument("-s", "--seed", type=int, default=None, 
                        help="Random seed for reproducibility")
    parser.add_argument("-o", "--output", type=str, default=None, 
                        help="Custom output filename. Defaults to little_dimple_n{N}_d{D}.qasm")

    args = parser.parse_args()
    
    # 1. Build Circuit
    circuit = build_little_dimple_circuit(num_qubits=args.qubits, depth=args.depth, seed=args.seed)
    
    # 2. Output Statistics
    print(f"\n--- Circuit Generated: {circuit.name} ---")
    print(f"Width: {circuit.num_qubits}")
    print(f"Total Depth: {circuit.depth()}")
    print(f"Gate Counts: {circuit.count_ops()}")
    
    # 3. Visualize Connectivity
    visualize_connectivity(circuit)

    # 4. Save to QASM File
    if args.output:
        filename = args.output
    else:
        filename = f"little_dimple_n{args.qubits}_d{args.depth}.qasm"

    try:
        with open(filename, "w") as f:
            qiskit.qasm2.dump(circuit, f)
        print(f"Success: Circuit saved to '{filename}'")
    except Exception as e:
        print(f"Error saving file: {e}")
