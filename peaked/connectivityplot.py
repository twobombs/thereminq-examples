import argparse
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
import qiskit.qasm2
import os

def load_qasm_robust(filepath):
    """
    Loads a QASM file with fallback logic for missing gate definitions (like 'u').
    """
    with open(filepath, 'r') as f:
        qasm_str = f.read()

    try:
        # 1. Try standard load
        return qiskit.qasm2.loads(qasm_str)
    except qiskit.qasm2.QASM2Error as e:
        # If we get a specific error about 'u' being undefined, we try to patch it
        if "'u' is not defined" in str(e):
            print("Notice: 'u' gate undefined. Attempting to patch QASM header...")
            
            # Define the 'u' gate manually as a wrapper around the primitive 'U'
            patch = "gate u(theta,phi,lambda) q { U(theta,phi,lambda) q; }\n"
            
            # Find insertion point (after 'OPENQASM 2.0;' and includes)
            lines = qasm_str.splitlines()
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("include") or line.strip().startswith("OPENQASM"):
                    insert_idx = i + 1
            
            lines.insert(insert_idx, patch)
            patched_qasm = "\n".join(lines)
            
            # Retry loading
            return qiskit.qasm2.loads(patched_qasm)
        else:
            raise e

def visualize_connectivity(qasm_file):
    print(f"Loading circuit from '{qasm_file}'...")
    
    try:
        qc = load_qasm_robust(qasm_file)
    except Exception as e:
        print(f"CRITICAL ERROR loading QASM file: {e}")
        return

    n = qc.num_qubits
    print(f"Analyzing circuit: Width={n}, Depth={qc.depth()}")

    # Initialize N x N matrix
    interaction_matrix = np.zeros((n, n))
    interaction_count = 0
    
    for instruction in qc.data:
        # Check for 2-qubit gates
        if len(instruction.qubits) == 2:
            interaction_count += 1
            # Find indices
            q0_idx = qc.find_bit(instruction.qubits[0]).index
            q1_idx = qc.find_bit(instruction.qubits[1]).index
            
            # Symmetric update
            interaction_matrix[q0_idx, q1_idx] += 1
            interaction_matrix[q1_idx, q0_idx] += 1
            
    print(f"Found {interaction_count} two-qubit interactions.")

    # --- Plotting ---
    plt.figure(figsize=(10, 8))
    plt.imshow(interaction_matrix, origin='lower', cmap='viridis', interpolation='nearest')
    
    cbar = plt.colorbar()
    cbar.set_label('Interaction Count')
    
    plt.title(f'Connectivity Heatmap\nFile: {os.path.basename(qasm_file)} | Width: {n}')
    plt.xlabel('Qubit Index')
    plt.ylabel('Qubit Index')
    
    # Adjust ticks for readability
    step = max(1, n // 20)
    plt.xticks(range(0, n, step))
    plt.yticks(range(0, n, step))

    # Save
    base_name = os.path.splitext(os.path.basename(qasm_file))[0]
    output_filename = f"{base_name}_heatmap.png"
    plt.savefig(output_filename)
    print(f"Heatmap saved to '{output_filename}'")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a connectivity heatmap from a QASM file.")
    parser.add_argument("file", type=str, help="Path to the input .qasm file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
    else:
        visualize_connectivity(args.file)
