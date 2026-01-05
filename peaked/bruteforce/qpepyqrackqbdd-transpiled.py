import qiskit
from qiskit import transpile  # <--- Added import
from pyqrack import QrackSimulator
import numpy as np
import warnings
import time
import os
import csv

# Suppress warnings
warnings.filterwarnings('ignore')

QASM_FILE = 'P1_little_dimple.qasm'
CSV_FILE = 'qrack_results.csv'

def run_pyqrack_simulation():
    print(f"--- Loading {QASM_FILE} into PyQrack ---")
    
    # 1. Parse QASM
    try:
        circuit = qiskit.QuantumCircuit.from_qasm_file(QASM_FILE)
    except Exception as e:
        print(f"Error parsing QASM: {e}")
        return

    # 1.5 Transpilation Phase
    print("Transpiling circuit to compatible basis gates...")
    start_transpile = time.time()
    
    # We define the basis gates to match exactly what your loop handles below.
    # This prevents "Unhandled gate" warnings and optimizes the circuit.
    target_basis = ['u', 'u3', 'cu', 'cu3', 'cz', 'cx', 'cp', 'cu1', 'rz', 'h', 'x', 'swap']
    
    # optimization_level=3 gives the heaviest optimization (gate cancellation, etc.)
    circuit = transpile(circuit, basis_gates=target_basis, optimization_level=3)
    
    transpile_duration = time.time() - start_transpile
    print(f"Transpilation complete in {transpile_duration:.4f} seconds.")

    n_qubits = circuit.num_qubits
    print(f"Circuit has {n_qubits} qubits.")

    # 2. Initialize PyQrack Simulator with QBDD Enabled
    try:
        # isBinaryDecisionTree=True enables the QBDD engine
        sim = QrackSimulator(n_qubits, isBinaryDecisionTree=True)
        print("PyQrack Simulator initialized with QBDD engine.")
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    start_time = time.time()
    
    # 3. Execute Gates
    print("Executing gates...")
    count = 0
    qubit_map = {q: i for i, q in enumerate(circuit.qubits)}

    for instruction in circuit.data:
        op = instruction.operation
        name = op.name.lower()
        qubits = [qubit_map[q] for q in instruction.qubits]
        params = [float(p) for p in op.params]

        # --- GATE MAPPING ---
        if name == 'u' or name == 'u3':
            sim.u(qubits[0], params[0], params[1], params[2])
            
        elif name == 'cu' or name == 'cu3':
            sim.mcu([qubits[0]], qubits[1], params[0], params[1], params[2])
            
        elif name == 'cz':
            sim.mcz([qubits[0]], qubits[1])
            
        elif name == 'cx':
            sim.mcx([qubits[0]], qubits[1])
        
        elif name == 'cp' or name == 'cu1': 
            sim.mcu([qubits[0]], qubits[1], 0.0, 0.0, params[0])

        elif name == 'rz':
            sim.rz(qubits[0], params[0])
            
        elif name == 'h':
            sim.h(qubits[0])
            
        elif name == 'x':
            sim.x(qubits[0])
            
        elif name == 'swap':
            sim.swap(qubits[0], qubits[1])
            
        elif name == 'barrier' or name == 'measure':
            pass
            
        else:
            print(f"WARNING: Unhandled gate '{name}'")

        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} gates...")

    duration = time.time() - start_time
    print(f"Simulation finished in {duration:.4f} seconds.")

    # 4. Measure
    measurement = sim.m_all()
    bitstring = format(measurement, f'0{n_qubits}b')
    
    # 5. Decode Phase
    # Note: Ensure this slicing logic matches your specific algorithm's qubit ordering
    counting_bits_str = bitstring[:-1] 
    
    decimal_phase = 0.0
    for i, bit in enumerate(reversed(counting_bits_str)):
        if bit == '1':
            decimal_phase += 1 / (2**(i + 1))
            
    print(f"Estimated Phase: {decimal_phase:.6f}")

    # 6. Save Run to CSV
    sep_thresh = os.environ.get('QRACK_QUNIT_SEPARABILITY_THRESHOLD', 'N/A')
    paging_qb = os.environ.get('QRACK_MAX_PAGING_QB', 'N/A')
    
    file_exists = os.path.isfile(CSV_FILE)
    
    try:
        with open(CSV_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                # Updated header to track QBDD status
                writer.writerow(['QBDD_Enabled', 'Separability', 'MaxPagingQB', 'Duration_Sec', 'Phase', 'Measured_Int', 'Bitstring'])
            
            # Log 'True' for QBDD_Enabled
            writer.writerow(['True', sep_thresh, paging_qb, f"{duration:.4f}", f"{decimal_phase:.6f}", measurement, bitstring])
            print(f"Results appended to {CSV_FILE}")
    except Exception as e:
        print(f"Failed to write to CSV: {e}")

if __name__ == "__main__":
    run_pyqrack_simulation()
