import torch
import qiskit
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def qasm_to_tensor_v2(qasm_file, output_file):
    print(f"--- Reading {qasm_file} ---")
    try:
        circuit = qiskit.QuantumCircuit.from_qasm_file(qasm_file)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Gate Map (Using your existing IDs)
    # ID 28 = u (3 params)
    # ID 17 = cz (0 params)
    gate_map = {
        'u': 28, 
        'cz': 17,
        'cx': 17, # Map CX to same ID if you want, or change to 15
        'rz': 11,
        'u3': 28
    }

    qubit_to_idx = {q: i for i, q in enumerate(circuit.qubits)}
    
    # New Tensor Shape: [GateID, Q1, Q2, Param1, Param2, Param3]
    tensor_data = []

    for instruction in circuit.data:
        op = instruction.operation
        name = op.name.lower()
        
        if name not in gate_map:
            # Fallback for unexpected gates
            continue
            
        gid = gate_map[name]
        
        # Get Qubits
        q1 = qubit_to_idx[instruction.qubits[0]]
        q2 = qubit_to_idx[instruction.qubits[1]] if len(instruction.qubits) > 1 else -1
        
        # Get Parameters (Support up to 3)
        params = [float(p) for p in op.params]
        # Pad with zeros if fewer than 3 params
        while len(params) < 3:
            params.append(0.0)
            
        # Store: [ID, Q1, Q2, P1, P2, P3]
        tensor_data.append([gid, q1, q2, params[0], params[1], params[2]])

    # Save
    tensor = torch.tensor(tensor_data, dtype=torch.float32)
    torch.save(tensor, output_file)
    print(f"Success! Saved {len(tensor)} gates to {output_file}")
    print("New Tensor Shape:", tensor.shape)

if __name__ == "__main__":
    # Point this to your uploaded QASM file
    qasm_to_tensor_v2('P1_little_dimple.qasm', 'circuit_tensor.pt')
