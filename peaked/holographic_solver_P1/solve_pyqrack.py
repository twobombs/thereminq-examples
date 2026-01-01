import numpy as np
from qiskit import QuantumCircuit
from pyqrack import QrackSimulator

def solve_with_pyqrack(filename):
    print(f"Loading '{filename}'...")
    try:
        # We use Qiskit to parse the QASM 2.0 file easily
        qc = QuantumCircuit.from_qasm_file(filename)
    except FileNotFoundError:
        print("Error: File not found. Ensure 'layer_2_dimple_core.qasm' exists.")
        return

    num_qubits = qc.num_qubits
    print(f"Circuit Depth: {len(qc)}")
    print(f"Simulating {num_qubits} qubits with PyQrack...")
    
    # Initialize PyQrack Simulator
    sim = QrackSimulator(num_qubits)
    
    # FIX: Create a map to look up qubit indices
    qubit_map = {bit: i for i, bit in enumerate(qc.qubits)}
    
    # Iterate through Qiskit gates and apply to PyQrack
    count = 0
    for instruction in qc.data:
        op = instruction.operation
        # FIX: Use the map instead of accessing .index
        qubits = [qubit_map[q] for q in instruction.qubits]
        
        # 1. Standard Stabilizer Gates
        if op.name == 'cz':
            sim.mcz([qubits[0]], qubits[1])
        elif op.name == 'cx':
            sim.mcx([qubits[0]], qubits[1])
        elif op.name == 'swap':
            sim.swap(qubits[0], qubits[1])
        elif op.name == 'h':
            sim.h(qubits[0])
        elif op.name == 'x':
            sim.x(qubits[0])
        elif op.name == 'y':
            sim.y(qubits[0])
        elif op.name == 'z':
            sim.z(qubits[0])
        elif op.name == 's':
            sim.s(qubits[0])
        elif op.name == 'sdg':
            sim.adjs(qubits[0])
            
        # 2. Parameterized U Gates (The tricky part)
        elif op.name == 'u':
            theta, phi, lam = [float(p) for p in op.params]
            sim.u(qubits[0], theta, phi, lam)
            
        # 3. Other handling
        elif op.name in ['barrier', 'id', 'measure']:
            pass # Ignore
        else:
            print(f"Warning: Unknown gate '{op.name}' skipped.")

        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} gates...")

    print("Simulation complete. Extracting peaked bitstring...")
    
    # Use m_all to get the peaked result directly
    measurements = sim.m_all()
    
    # Convert result (integer) to bitstring
    # PyQrack m_all returns a decimal integer
    result_val = measurements
    
    # Format to binary string
    bitstring = f"{result_val:0{num_qubits}b}"
    
    # Note on Endianness: 
    # Qiskit QASM usually orders registers q[35]...q[0]. 
    # If the integer is 1 (binary ...001), it means q[0]=1.
    # The string generated above puts MSB (q[35]) on the left, LSB (q[0]) on the right.
    
    print(f"\n>>> PEAKED VALUE FOUND <<<")
    print(f"Decimal: {result_val}")
    print(f"Bitstring: {bitstring}")
    
    with open("solution_pyqrack.txt", "w") as f:
        f.write(bitstring)

if __name__ == "__main__":
    solve_with_pyqrack('layer_2_dimple_core.qasm')
