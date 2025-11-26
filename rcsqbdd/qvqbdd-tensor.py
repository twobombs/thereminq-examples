import sys
import math
import random
import pickle  # <--- ADDED for saving
from collections import Counter

import numpy as np
import quimb.tensor as qtn
from pyqrack import QrackSimulator, Pauli

# ==============================================================================
# 1. ROBUST CIRCUIT GENERATOR
# ==============================================================================
def generate_robust_circuit(width, depth, seed=None):
    """
    Generates a random circuit using explicit RZ and RY rotations.
    Using explicit rotations avoids ambiguity in 'U3' gate definitions.
    """
    if seed is not None:
        random.seed(seed)
        
    lcv_range = range(width)
    all_bits = list(lcv_range)

    circ = qtn.Circuit(width)
    
    for d in range(depth):
        for i in lcv_range:
            # Random Euler Rotations (0.1 to 3.0 avoids 0/pi ambiguity)
            rz1 = random.uniform(0.1, 3.0)
            ry  = random.uniform(0.1, 3.0)
            rz2 = random.uniform(0.1, 3.0)
            circ.apply_gate('RZ', rz1, i)
            circ.apply_gate('RY', ry, i)
            circ.apply_gate('RZ', rz2, i)

        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            circ.apply_gate('CNOT', c, t)
            
    if seed is not None:
        random.seed() 
        
    return circ

# ==============================================================================
# 2. SIMULATION ENGINE
# ==============================================================================
def run_pyqrack_sim(circ, width, config, shots=None):
    """
    Runs the simulation on PyQrack using calibrated physics settings.
    """
    qsim = QrackSimulator(width)
    factor = config.get('factor', 1.0)
    sign   = config.get('sign', 1.0)
    
    for gate in circ.gates:
        name = gate.label 
        p = gate.params
        q = gate.qubits
        
        if name == 'RZ':
            # Apply: PauliZ, angle * factor * sign
            angle = float(p[0]) * factor * sign
            qsim.r(Pauli.PauliZ, angle, q[0])
        elif name == 'RY':
            # Apply: PauliY, angle * factor * sign
            angle = float(p[0]) * factor * sign
            qsim.r(Pauli.PauliY, angle, q[0])
        elif name == 'CNOT':
            qsim.mct([q[0]], q[1])
            
    if shots:
        # Return measurement counts for Production Run
        measurements = qsim.measure_shots(list(range(width)), shots)
        return Counter(measurements)
    else:
        # Return probability that Qubit 0 is in state |1> (For Calibration)
        return qsim.prob(0)

# ==============================================================================
# 3. ROBUST AUTO-CALIBRATION
# ==============================================================================
def calibrate_engine():
    print("--- Auto-Calibrating Physics Engine ---")
    config = {'factor': 1.0, 'sign': 1.0}

    # --- STEP 1: SCALING TEST ---
    # Does Input(PI) rotate |0> to |1>?
    qsim = QrackSimulator(1)
    qsim.r(Pauli.PauliY, math.pi, 0)
    prob_one = qsim.prob(0) 
    
    if prob_one > 0.99:
        print(f"1. Scaling: Standard (1.0) detected [Prob|1>={prob_one:.4f}]")
        config['factor'] = 1.0
    elif abs(prob_one - 0.5) < 0.1:
        print(f"1. Scaling: Half-Angle (0.5) detected [Prob|1>={prob_one:.4f}]")
        config['factor'] = 2.0 
    else:
        print(f"1. Scaling: Double-Angle (2.0) detected [Prob|1>={prob_one:.4f}]")
        config['factor'] = 0.5

    # --- STEP 2: ROTATION SIGN TEST ---
    # We use a 2-qubit circuit and check the marginal probability of Q0.
    print("2. Calibrating Rotation Sign...")
    
    w_cal, d_cal = 2, 2
    qc_cal = generate_robust_circuit(w_cal, d_cal, seed=42)
    
    # Calculate Ground Truth Marginal P(q0=1) using Quimb/Numpy
    psi = qc_cal.psi.to_dense().reshape((2, 2))
    # Sum prob over qubit 1 for cases where qubit 0 is |1> (Index 1)
    truth_marginal = np.sum(np.abs(psi[1, :])**2)
    
    def check_sign(s):
        cfg = {'factor': config['factor'], 'sign': s}
        # shots=None returns qsim.prob(0)
        return run_pyqrack_sim(qc_cal, w_cal, cfg, shots=None)

    prob_pos = check_sign(1.0)
    prob_neg = check_sign(-1.0)
    
    err_pos = abs(prob_pos - truth_marginal)
    err_neg = abs(prob_neg - truth_marginal)
    
    if err_neg < err_pos:
        print(f"   -> Detected Inverted Sign (-1.0). Error: {err_neg:.5f}")
        config['sign'] = -1.0
    else:
        print(f"   -> Detected Standard Sign (+1.0). Error: {err_pos:.5f}")
        config['sign'] = 1.0
        
    return config

# ==============================================================================
# 4. MAIN PIPELINE
# ==============================================================================
def main():
    if len(sys.argv) < 3:
        width = 10
        depth = 10
    else:
        width = int(sys.argv[1])
        depth = int(sys.argv[2])

    # 1. Calibrate
    try:
        config = calibrate_engine()
    except Exception as e:
        print(f"Calibration error: {e}. Using defaults.")
        config = {'factor': 1.0, 'sign': 1.0}

    # 2. Simulate
    print(f"\n--- Generating & Simulating (W={width}, D={depth}) ---")
    qc = generate_robust_circuit(width, depth)
    shots = 8192
    
    counts = run_pyqrack_sim(qc, width, config, shots=shots)
    if not counts: return 1
        
    best_int, count = counts.most_common(1)[0]
    est_prob = count / shots
    
    # Extract bits
    best_bits_raw = [(best_int >> i) & 1 for i in range(width)]
    
    print(f"\nMost Frequent Bitstring: {tuple(best_bits_raw)}")
    print(f"Estimated Probability:   {est_prob:.6f}")

    # 3. Create Tensor & Verify
    print(f"\n--- Creating Tensor & Verifying ---")
    
    mps_raw = qtn.MPS_computational_state(best_bits_raw)
    mps_rev = qtn.MPS_computational_state(best_bits_raw[::-1])
    
    # Contract <MPS|Psi>
    amp_raw = (mps_raw.H & qc.psi).contract(all, optimize='greedy')
    prob_raw = abs(amp_raw)**2
    
    amp_rev = (mps_rev.H & qc.psi).contract(all, optimize='greedy')
    prob_rev = abs(amp_rev)**2
    
    print(f"Exact Prob (Raw Order):  {prob_raw:.6f}")
    print(f"Exact Prob (Rev Order):  {prob_rev:.6f}")
    
    err_raw = abs(prob_raw - est_prob)
    err_rev = abs(prob_rev - est_prob)
    
    final_mps = None
    
    if err_rev < err_raw:
        print("\n-> MATCH: Reversed Endianness detected.")
        final_mps = mps_rev
        final_error = err_rev
    else:
        print("\n-> MATCH: Raw Endianness detected.")
        final_mps = mps_raw
        final_error = err_raw
        
    print(f"Final Mismatch:          {final_error:.6f}")
    
    # 4. Save Tensor
    if final_error < 0.05:
        print("\nSUCCESS: Tensor generated successfully.")
        
        print(final_mps)
        
        # --- SAVE TO DISK (FIXED) ---
        filename = "solution_tensor.pickle"
        try:
            # Use standard pickle for Quimb objects
            with open(filename, 'wb') as f:
                pickle.dump(final_mps, f)
                
            print(f"\n[SAVED] Tensor saved to disk as '{filename}'")
            print("To load later:")
            print("  import pickle")
            print(f"  with open('{filename}', 'rb') as f:")
            print("      t = pickle.load(f)")
        except Exception as e:
            print(f"\n[ERROR] Failed to save tensor: {e}")
            
    else:
        print("\nWARNING: High mismatch persists. Tensor not saved to avoid pollution.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
