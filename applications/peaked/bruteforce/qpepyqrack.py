import qiskit
from pyqrack import QrackSimulator
import numpy as np
import warnings
import time
import os
import csv

# Suppress warnings
warnings.filterwarnings('ignore')

QASM_FILE = 'layer_2_dimple_core.qasm'
CSV_FILE = 'qrack_results.csv'

def get_u3_matrix(theta, phi, lam):
    """
    Returns the 2x2 unitary matrix for a U3(theta, phi, lambda) gate.
    """
    cos_half = np.cos(theta / 2.0)
    sin_half = np.sin(theta / 2.0)
    
    u00 = cos_half
    u01 = -np.exp(1j * lam) * sin_half
    u10 = np.exp(1j * phi) * sin_half
    u11 = np.exp(1j * (phi + lam)) * cos_half
    
    return [u00, u01, u10, u11]

def load_qasm_robust(filename):
    """
    Reads QASM file and injects 'u' gate definition if missing
    to prevent 'u is not defined' errors.
    """
    with open(filename, 'r') as f:
        qasm_str = f.read()

    # Check if 'gate u(' is missing and 'u(' is used
    # We inject a standard definition for 'u' mapping it to primitive 'U'
    header_injection = "gate u(theta, phi, lambda) q { U(theta, phi, lambda) q; }\n"
    
    # If the file already has 'OPENQASM', we inject after the version line
    if "OPENQASM" in qasm_str:
        lines = qasm_str.split('\n')
        # Find the index of the version line
        for i, line in enumerate(lines):
            if line.strip().startswith("OPENQASM"):
                lines.insert(i + 1, header_injection)
                break
        qasm_str = "\n".join(lines)
    else:
        # If no header, just prepend
        qasm_str = header_injection + qasm_str

    return qiskit.QuantumCircuit.from_qasm_str(qasm_str)

def run_pyqrack_simulation():
    print(f"--- Loading {QASM_FILE} into PyQrack ---")
    
    # 1. Parse QASM (Robust Method)
    try:
        circuit = load_qasm_robust(QASM_FILE)
    except Exception as e:
        print(f"Error parsing QASM: {e}")
        return

    n_qubits = circuit.num_qubits
    print(f"Circuit has {n_qubits} qubits.")

    # 2. Initialize PyQrack Simulator
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
            mat = get_u3_matrix(params[0], params[1], params[2])
            sim.mcmul([qubits[0]], qubits[1], mat)
            
        elif name == 'cz':
            sim.mcz([qubits[0]], qubits[1])
            
        elif name == 'cx':
            sim.mcx([qubits[0]], qubits[1])
        
        elif name == 'cp' or name == 'cu1': 
            # Controlled-Phase: diag(1, e^i*lambda)
            mat = [1.0, 0.0, 0.0, np.exp(1j * params[0])]
            sim.mcmul([qubits[0]], qubits[1], mat)

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
    # Strip system qubit (Q0, last char) and reverse to get Q1 as MSB
    counting_bits_str = bitstring[:-1] 
    decimal_phase = 0.0
    for i, bit in enumerate(reversed(counting_bits_str)):
        if bit == '1':
            decimal_phase += 1 / (2**(i + 1))
            
    print(f"Estimated Phase: {decimal_phase:.6f}")

    # 6. Save Run to CSV
    sep_thresh = os.environ.get('QRACK_QUNIT_SEPARABILITY_THRESHOLD', 'N/A')
    paging_qb = os.environ.get('QRACK_MAX_PAGING_QB', 'N/A')
    
    try:
        file_exists = os.path.isfile(CSV_FILE)
        with open(CSV_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Separability', 'MaxPagingQB', 'Duration_Sec', 'Phase', 'Measured_Int', 'Bitstring'])
            writer.writerow([sep_thresh, paging_qb, f"{duration:.4f}", f"{decimal_phase:.6f}", measurement, bitstring])
            print(f"Results appended to {CSV_FILE}")
    except Exception as e:
        print(f"Failed to write to CSV: {e}")

if __name__ == "__main__":
    run_pyqrack_simulation()
