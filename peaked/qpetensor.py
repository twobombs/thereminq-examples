import qiskit
import torch
import numpy as np
import warnings

# Suppress Qiskit warnings for cleaner output
warnings.filterwarnings('ignore')

def qasm_to_tensor(qasm_content: str, export_path: str = 'circuit_tensor.pt'):
    """
    Parses QASM content and converts it to a PyTorch tensor encoding the circuit structure.
    
    Tensor Schema per row: [Gate_ID, Qubit_Index_1, Qubit_Index_2, Parameter_Value]
    """
    
    # 1. Parse the QASM content
    try:
        circuit = qiskit.QuantumCircuit.from_qasm_str(qasm_content)
    except Exception as e:
        print(f"Error parsing QASM: {e}")
        return None

    # 2. Define a Mapping for Gate Types to Integers
    gate_map = {
        'x': 0,     # Pauli-X
        'h': 1,     # Hadamard
        'u1': 2,    # Phase Gate (parameterized)
        'cx': 3,    # Controlled-NOT
        'swap': 4,  # Swap Gate
        'barrier': 5 
    }

    # --- FIX START ---
    # Create a dictionary mapping Qubit objects to their integer indices
    # This works for both old and new Qiskit versions
    qubit_to_index = {qubit: i for i, qubit in enumerate(circuit.qubits)}
    # --- FIX END ---

    tensor_data = []

    # 3. Iterate through instructions to build the tensor
    for instruction in circuit.data:
        op = instruction.operation
        qubits = instruction.qubits
        
        # Get Gate ID
        gate_name = op.name
        if gate_name not in gate_map:
            continue
        
        gate_id = gate_map[gate_name]

        # --- FIX START ---
        # Look up the index using our dictionary instead of .index
        q1 = qubit_to_index[qubits[0]]
        q2 = qubit_to_index[qubits[1]] if len(qubits) > 1 else -1
        # --- FIX END ---

        # Extract Parameter (if any)
        param = float(op.params[0]) if len(op.params) > 0 else 0.0

        # Append to list: [Gate_ID, Q1, Q2, Param]
        tensor_data.append([gate_id, q1, q2, param])

    # 4. Convert to PyTorch Tensor
    circuit_tensor = torch.tensor(tensor_data, dtype=torch.float32)

    # 5. Export
    torch.save(circuit_tensor, export_path)
    
    print(f"--- Conversion Successful ---")
    print(f"Total Gates Processed: {circuit_tensor.shape[0]}")
    print(f"Tensor Shape: {circuit_tensor.shape}")
    print(f"Tensor saved to: {export_path}")
    
    return circuit_tensor

# --- Load your specific file content ---
# Ensure you are loading your actual 'qpe.qasm' file content here
with open('qpe.qasm', 'r') as f:
    qasm_source = f.read()

tensor_output = qasm_to_tensor(qasm_source)
