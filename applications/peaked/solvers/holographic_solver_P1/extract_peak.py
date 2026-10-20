import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import UGate
from qiskit.quantum_info import Clifford

def snap_to_clifford(val, atol=1e-3):
    """
    Returns the nearest exact multiple of pi/2 if within tolerance.
    """
    unit = np.pi / 2
    k = round(val / unit)
    exact_val = k * unit
    if abs(val - exact_val) <= atol:
        return exact_val
    return None

def extract_peaked_value(filename):
    print(f"Loading '{filename}'...")
    try:
        qc_noisy = QuantumCircuit.from_qasm_file(filename)
    except FileNotFoundError:
        print("Error: File not found. Please check the filename.")
        return

    print(f"Original Core Depth: {len(qc_noisy)}")
    
    # --- STEP 1: PURIFICATION & CONVERSION ---
    # We rebuild the circuit, converting 'u' gates to explicit 'h', 's', 'z' sequences
    qc_clean = QuantumCircuit(qc_noisy.qubits, qc_noisy.clbits)
    
    kept_count = 0
    removed_count = 0
    
    for instruction in qc_noisy.data:
        op = instruction.operation
        qubits = instruction.qubits
        clbits = instruction.clbits
        
        # 1. Keep Native Stabilizer Gates
        if op.name in ['cz', 'cx', 'swap', 'id', 'barrier', 'measure', 
                       'x', 'y', 'z', 'h', 's', 'sdg', 'sx']:
            qc_clean.append(op, qubits, clbits)
            kept_count += 1
            
        # 2. Convert 'u' Gates to Standard Cliffords
        elif op.name == 'u':
            # Snap parameters to perfect grid
            raw_params = [float(p) for p in op.params]
            snapped_params = [snap_to_clifford(p) for p in raw_params]
            
            if all(p is not None for p in snapped_params):
                try:
                    # Create exact UGate
                    u_gate = UGate(*snapped_params)
                    
                    # Convert to Clifford object (math validation)
                    cliff = Clifford(u_gate)
                    
                    # Decompose into standard gates (H, S, etc.)
                    # This removes the 'u' gate entirely!
                    sub_circ = cliff.to_circuit()
                    qc_clean.compose(sub_circ, qubits=qubits, inplace=True)
                    
                    kept_count += 1
                except Exception:
                    # Not a valid Clifford
                    removed_count += 1
            else:
                removed_count += 1
        else:
            removed_count += 1

    print(f"\n--- PURIFICATION REPORT ---")
    print(f"Thermal Gates Removed: {removed_count}")
    print(f"Clifford Gates Kept:   {kept_count}")

    # --- STEP 2: SIMULATION ---
    print("\nSimulating purified structure (Stabilizer Method)...")
    
    # Ensure measurement
    if qc_clean.num_clbits == 0:
        qc_clean.measure_all()
    
    # Run Simulation
    # No need for complex transpilation now because the circuit 
    # only contains standard gates (H, S, Z, CZ) which the simulator supports natively.
    sim = AerSimulator(method='stabilizer')
    
    try:
        qc_transpiled = transpile(qc_clean, sim)
        job = sim.run(qc_transpiled, shots=1024)
        counts = job.result().get_counts()
        
        # Extract Result
        peaked_bitstring = max(counts, key=counts.get)
        probability = counts[peaked_bitstring] / 1024.0
        
        print(f"\n>>> PEAKED VALUE BITSTRING <<<")
        print(f"Value: {peaked_bitstring}")
        print(f"Probability: {probability:.2%}")
        
        # Write to file
        with open("solution.txt", "w") as f:
            f.write(peaked_bitstring)
        print("Saved solution to 'solution.txt'")
        
    except Exception as e:
        print(f"Simulation Error: {e}")

if __name__ == "__main__":
    extract_peaked_value('layer_2_dimple_core.qasm')
