import math
import random
import sys
from pyqrack import QrackSimulator
from qiskit import QuantumCircuit

def bench_patch_xeb_explicit(width, depth):
    # 1. Configuration
    mid = width // 2
    mask_lower = (1 << mid) - 1 
    shots = 5000 
    
    print(f"Generating Patch Circuit: {width} qubits split at {mid}...")

    # 2. Build Circuits (Full + Split)
    full_rcs = QuantumCircuit(width)
    left_rcs = QuantumCircuit(mid)
    right_rcs = QuantumCircuit(width - mid)

    for d in range(depth):
        # Single Qubit Gates
        for i in range(width):
            angle = random.uniform(0, 2 * math.pi)
            full_rcs.h(i)
            full_rcs.rz(angle, i)
            
            if i < mid:
                left_rcs.h(i)
                left_rcs.rz(angle, i)
            else:
                right_rcs.h(i - mid)
                right_rcs.rz(angle, i - mid)

        # 2-Qubit Couplers
        all_bits = list(range(width))
        random.shuffle(all_bits)
        while len(all_bits) > 1:
            u = all_bits.pop()
            v = all_bits.pop()
            
            u_side = u < mid
            v_side = v < mid
            
            if u_side == v_side:
                full_rcs.cx(u, v)
                if u_side:
                    left_rcs.cx(u, v)
                else:
                    right_rcs.cx(u - mid, v - mid)

    # 3. Run 'Experiment' (Full Grid)
    print(f"Running Experiment on {width} qubits (sampling)...")
    sim_exp = QrackSimulator(width)
    sim_exp.run_qiskit_circuit(full_rcs)
    measured_shots = sim_exp.measure_shots(list(range(width)), shots)
    
    # 4. Run 'Ideal' Verification (Split Grids)
    # FIX: Pass the explicit list of qubits [0, 1, ... N-1] to prob_all()
    print(f"Running Ideal Left Patch ({mid} qubits)...")
    sim_left = QrackSimulator(mid)
    sim_left.run_qiskit_circuit(left_rcs)
    probs_left = sim_left.prob_all(list(range(mid))) 
    
    print(f"Running Ideal Right Patch ({width - mid} qubits)...")
    sim_right = QrackSimulator(width - mid)
    sim_right.run_qiskit_circuit(right_rcs)
    probs_right = sim_right.prob_all(list(range(width - mid)))

    # 5. Calculate Linear XEB
    print("Calculating XEB...")
    sum_probs = 0.0
    
    for k in measured_shots:
        k_lower = k & mask_lower
        k_upper = k >> mid
        
        # Multiply the probabilities of the independent halves
        p_total = probs_left[k_lower] * probs_right[k_upper]
        sum_probs += p_total

    mean_prob = sum_probs / shots
    n_pow = 2 ** width
    xeb = (n_pow * mean_prob) - 1

    return xeb

if __name__ == "__main__":
    w = 30
    d = 10
    if len(sys.argv) >= 3:
        w = int(sys.argv[1])
        d = int(sys.argv[2])

    try:
        score = bench_patch_xeb_explicit(w, d)
        print("\n" + "="*50)
        print(f"Patch XEB Score: {score:.5f}")
        print("="*50)
    except Exception as e:
        print(f"Error: {e}")
