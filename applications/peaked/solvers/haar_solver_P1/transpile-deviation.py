from qiskit import QuantumCircuit, transpile, qasm2
import os
import time

# --- CONFIGURATION ---
INPUT_FILENAME = 'P1_purified.qasm'
OUTPUT_FILENAME = 'P1_final_collapsed.qasm'

def optimize_and_save():
    if not os.path.exists(INPUT_FILENAME):
        print(f"Error: File '{INPUT_FILENAME}' not found.")
        return

    print(f"Loading {INPUT_FILENAME}...")
    try:
        qc = QuantumCircuit.from_qasm_file(INPUT_FILENAME)
    except Exception as e:
        print(f"Failed to load QASM: {e}")
        return

    # Count original gates
    original_ops = qc.count_ops()
    total_original = sum(original_ops.values())
    print(f"Original Gate Count: {total_original}")

    print("Running Qiskit Transpiler (Optimization Level 3)...")
    start_time = time.time()
    
    # Level 3 collapses unitaries and cancels inverses
    qc_opt = transpile(qc, optimization_level=3)
    
    end_time = time.time()
    
    # Count new gates
    new_ops = qc_opt.count_ops()
    total_new = sum(new_ops.values())
    
    print(f"Optimization complete in {end_time - start_time:.2f}s")
    print(f"Gates: {total_original} -> {total_new}")
    print(f"Reduction: {(1 - total_new/total_original)*100:.1f}%")
    print("-" * 30)
    print(f"Final Gate Composition: {new_ops}")
    print("-" * 30)

    # Save using the modern qasm2 exporter
    print(f"Saving optimized circuit to {OUTPUT_FILENAME}...")
    with open(OUTPUT_FILENAME, 'w') as f:
        qasm2.dump(qc_opt, f)
        
    print("Done. You can now run this smaller file on your solver.")

if __name__ == "__main__":
    optimize_and_save()
