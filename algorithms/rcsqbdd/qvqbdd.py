import sys
import math
import random
from collections import Counter

import quimb.tensor as qtn
from pyqrack import QrackSimulator

def generate_qv_circuit(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    lcv_range = range(width)
    all_bits = list(lcv_range)

    circ = qtn.Circuit(width)
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            circ.apply_gate('U3', th, ph, lm, i)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            circ.apply_gate('CNOT', c, t)

    return circ

def run_qbdd_simulation(quimb_circ, width):
    """
    Translates the Quimb circuit to PyQrack (QBDD) and finds the
    most probable bitstring via sampling.
    """
    print("Initializing QBDD Simulator (PyQrack)...")
    qsim = QrackSimulator(width)

    # Translate and apply gates
    # We iterate through the Quimb circuit and apply equivalent PyQrack gates.
    for i, gate in enumerate(quimb_circ.gates):
        name = gate.label 
        params = gate.params
        qubits = gate.qubits

        if name == 'U3':
            # Ensure params are native floats to avoid C++ type errors
            qsim.u(qubits[0], float(params[0]), float(params[1]), float(params[2]))
        elif name == 'CNOT':
            qsim.mct([qubits[0]], qubits[1])
        else:
            # Fallback for other gates could be added here
            print(f"Warning: Gate {name} at index {i} not implemented. Skipping.")

    print("Simulation complete. QBDD built.")

    # Functionality: Find Best Guess for highest-probability bit string.
    # QBDD allows us to sample O(N) without building the 2^N vector.
    
    shots = 128 # Increased shots for better statistical resolution
    print(f"Sampling {shots} shots from QBDD to find mode...")
    
    # measure_shots returns a list of integers representing the basis states
    measurements = qsim.measure_shots(list(range(width)), shots)
    
    if not measurements:
        raise RuntimeError("No measurements returned from simulator.")

    # We construct a histogram of results to find the Mode (highest frequency outcome)
    counts = Counter(measurements)
    best_int_state, count = counts.most_common(1)[0]
    
    # Convert integer back to bit list [q0, q1, ...] for display
    best_bits = [(best_int_state >> i) & 1 for i in range(width)]

    # --- PROBABILITY ESTIMATION ---
    # Since we cannot dump the statevector (too large or fragile C-bindings),
    # we estimate the probability P(x) ~ Count(x) / Total_Shots.
    estimated_prob = count / shots
    prob_source = "Estimated (Frequency)"

    return tuple(best_bits), estimated_prob, count, prob_source

def main():
    if len(sys.argv) < 3:
        print("No args provided. Using defaults: width=10, depth=10")
        width = 10
        depth = 10
    else:
        width = int(sys.argv[1])
        depth = int(sys.argv[2])

    print(f"Generating random circuit (Width={width}, Depth={depth})...")
    qc = generate_qv_circuit(width, depth)

    try:
        best_bits, probability, hit_count, method = run_qbdd_simulation(qc, width)
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n--- Results leveraging QBDD ---")
    print(f"Most frequent bitstring: {best_bits}")
    print(f"Sampling frequency:      {hit_count}")
    print(f"Probability ({method}):  {probability:.6f}")
    
    # This tuple matches the structure requested in the original script
    print(f"Result tuple:            {(best_bits, probability)}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
