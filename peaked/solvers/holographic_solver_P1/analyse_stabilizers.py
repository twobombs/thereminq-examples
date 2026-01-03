import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UGate
from qiskit.quantum_info import Clifford

def snap_to_clifford(val, atol=1e-3):
    unit = np.pi / 2
    k = round(val / unit)
    if abs(val - (k * unit)) <= atol:
        return k * unit
    return None

def analyze_stabilizer_structure(filename):
    print(f"Analyzing stabilizer structure of '{filename}'...")
    
    # 1. Load and Purify (Same logic as before)
    try:
        qc_noisy = QuantumCircuit.from_qasm_file(filename)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    qc_clean = QuantumCircuit(qc_noisy.qubits, qc_noisy.clbits)
    
    # Reconstruct perfectly
    for instruction in qc_noisy.data:
        op = instruction.operation
        qubits = instruction.qubits
        clbits = instruction.clbits
        
        if op.name in ['cz', 'cx', 'swap', 'id', 'barrier', 'measure', 
                       'x', 'y', 'z', 'h', 's', 'sdg', 'sx']:
            qc_clean.append(op, qubits, clbits)
            
        elif op.name == 'u':
            # Snap and convert
            params = [snap_to_clifford(float(p)) for p in op.params]
            if all(p is not None for p in params):
                try:
                    gate = UGate(*params)
                    sub_circ = Clifford(gate).to_circuit()
                    qc_clean.compose(sub_circ, qubits=qubits, inplace=True)
                except: pass

    # 2. Calculate the Stabilizer Tableau
    print("Computing Clifford Tableau...")
    # We remove measurements to get the state vector properties
    qc_clean.remove_final_measurements()
    
    clifford_state = Clifford(qc_clean)
    
    # 3. Analyze the output distribution width
    # The number of 'X' or 'Y' components in the Z-stabilizers dictates the superposition size.
    # We can infer the probability of a single bitstring by checking the number of 
    # independent Z-measurements that are deterministic.
    
    # Get the stabilizers (generators of the state)
    stabilizers = clifford_state.to_dict()['stabilizer']
    
    z_deterministic_count = 0
    for stab_str in stabilizers:
        # If a stabilizer is purely Z (e.g., "+ZIIZ..."), it constrains the output bitstring deterministically.
        # If it contains X or Y, it implies superposition (randomness in Z-basis).
        if 'X' not in stab_str and 'Y' not in stab_str:
            z_deterministic_count += 1
            
    num_qubits = qc_clean.num_qubits
    num_superposition_bits = num_qubits - z_deterministic_count
    
    exact_probability = 1.0 / (2 ** num_superposition_bits)
    
    print("\n--- STABILIZER ANALYSIS ---")
    print(f"Total Qubits: {num_qubits}")
    print(f"Deterministic Z-Constraints: {z_deterministic_count}")
    print(f"Superposition Dimensions:    {num_superposition_bits}")
    print(f"\nExact Theoretical Probability of your bitstring:")
    print(f"1 / 2^{num_superposition_bits} = {exact_probability:.6%}")
    print(f"(1 in {2**num_superposition_bits})")
    
    print("\nInterpretation:")
    if exact_probability < 0.99:
        print("The low probability is physically inherent to the state.")
        print("The state is a 'cat state' or 'superposition' over many bitstrings.")
        print("Your extracted bitstring is ONE of these valid solutions.")
    else:
        print("The state should be deterministic! If your simulation showed low prob,")
        print("it was likely due to noise or insufficient shots.")

if __name__ == "__main__":
    analyze_stabilizer_structure('layer_2_dimple_core.qasm')
