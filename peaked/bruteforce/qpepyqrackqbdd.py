import qiskit
from pyqrack import QrackSimulator
import numpy as np
import warnings
import time

# Suppress warnings
warnings.filterwarnings('ignore')

QASM_FILE = 'P1_little_dimple.qasm'

def run_pyqrack_simulation():
    print(f"--- Loading {QASM_FILE} into PyQrack ---")
    
    # 1. Parse QASM
    try:
        circuit = qiskit.QuantumCircuit.from_qasm_file(QASM_FILE)
    except Exception as e:
        print(f"Error parsing QASM: {e}")
        return

    n_qubits = circuit.num_qubits
    print(f"Circuit has {n_qubits} qubits.")

    # 2. Initialize PyQrack Simulator
    # PyQrack will automatically pick your best GPU (Tesla V100 or Quadro)
    try:
        sim = QrackSimulator(n_qubits)
        print("PyQrack Simulator initialized.")
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    start_time = time.time()
    
    # 3. Execute Gates
    print("Executing gates...")
    count = 0
    
    # Pre-map qubits to indices to speed up loop
    qubit_map = {q: i for i, q in enumerate(circuit.qubits)}

    for instruction in circuit.data:
        op = instruction.operation
        name = op.name.lower()
        
        # Get integer indices for qubits
        qubits = [qubit_map[q] for q in instruction.qubits]
        params = [float(p) for p in op.params]

        # --- GATE MAPPING ---
        
        if name == 'u' or name == 'u3':
            # u(theta, phi, lambda)
            # PyQrack: u(target, theta, phi, lambda)
            sim.u(qubits[0], params[0], params[1], params[2])
            
        elif name == 'cz':
            # FIX: Use Multi-Controlled Z
            # mcz(controls_list, target_index)
            sim.mcz([qubits[0]], qubits[1])
            
        elif name == 'cx':
            # FIX: Use Multi-Controlled X
            # mcx(controls_list, target_index)
            sim.mcx([qubits[0]], qubits[1])
            
        elif name == 'rz':
            sim.rz(qubits[0], params[0])
            
        elif name == 'h':
            sim.h(qubits[0])
            
        elif name == 'x':
            sim.x(qubits[0])
            
        elif name == 'measure':
            pass
        
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} gates...")

    duration = time.time() - start_time
    print(f"Simulation finished in {duration:.4f} seconds.")

    # 4. Measure
    print("Measuring all qubits...")
    measurement = sim.m_all()
    
    # Convert to binary string (padding with zeros to n_qubits length)
    bitstring = format(measurement, f'0{n_qubits}b')
    
    print(f"\nMeasured Integer: {measurement}")
    print(f"Measured Bitstring: {bitstring}")
    
    # 5. Decode Phase
    # QPE Convention: Q1..Qn are Counting Qubits. Q0 is System.
    # We strip Q0 and interpret Q1 as MSB (0.5), Q2 as (0.25), etc.
    
    # The bitstring is printed High-Qubit to Low-Qubit (Q35...Q0)
    # We remove Q0 (the last char)
    counting_bits_str = bitstring[:-1] 
    
    # Now we have Q35...Q1
    # We iterate backwards from the end of the string (Q1) to the front (Q35)
    
    decimal_phase = 0.0
    print("\n--- Phase Interpretation ---")
    for i, bit in enumerate(reversed(counting_bits_str)):
        # i=0 corresponds to Q1 (1/2), i=1 to Q2 (1/4)...
        if bit == '1':
            decimal_phase += 1 / (2**(i + 1))
            
    print(f"Estimated Phase: {decimal_phase:.6f}")
    print(f"Angle (Radians): {decimal_phase * 2 * np.pi:.6f}")

if __name__ == "__main__":
    run_pyqrack_simulation()
