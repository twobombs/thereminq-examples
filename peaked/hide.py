import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, OneQubitEulerDecomposer
from qiskit.circuit.library import UnitaryGate, XGate
import random

# --- CONFIGURATION ---
TARGET_BITSTRING = "10110"  # Define your desired winner here
BITWIDTH = len(TARGET_BITSTRING)
DEPTH = 20  # Reduced depth for demonstration; original was ~650
SEED = 42

def get_random_u_params():
    """Generates random theta, phi, lam for a U gate."""
    return np.random.uniform(0, 2 * np.pi, 3)

def build_and_steer_circuit(target_string, depth, seed=None):
    if seed:
        np.random.seed(seed)
        random.seed(seed)
    
    num_qubits = len(target_string)
    qc = QuantumCircuit(num_qubits)
    
    # 1. Build the Bulk of the Random Circuit (All layers except the very last rotations)
    # We do this so we can inject our steering logic into the final layer seamlessly.
    
    qubit_indices = list(range(num_qubits))
    
    # We will store the final rotations separately to apply corrections later
    final_layer_params = {} # Map qubit_index -> [theta, phi, lam]
    
    print(f"Building random circuit backbone ({num_qubits} qubits, {depth} layers)...")
    
    for layer in range(depth):
        random.shuffle(qubit_indices)
        
        for i in range(0, num_qubits - 1, 2):
            q_a = qubit_indices[i]
            q_b = qubit_indices[i+1]
            
            # Pre-entanglement Rotations
            qc.u(*get_random_u_params(), q_a)
            qc.u(*get_random_u_params(), q_b)
            
            # Entanglement
            qc.cz(q_a, q_b)
            
            # Post-entanglement Rotations (The "Dimple" pattern)
            # If this is the LAST layer, we don't add them to QC yet.
            # We calculate what they WOULD be, and store them.
            params_a = get_random_u_params()
            params_b = get_random_u_params()
            
            if layer == depth - 1:
                final_layer_params[q_a] = params_a
                final_layer_params[q_b] = params_b
            else:
                qc.u(*params_a, q_a)
                qc.u(*params_b, q_b)

    # 2. Handle odd qubit out in the last layer (if any)
    # The loop above skips the last qubit if width is odd. 
    # We must assign it a random final rotation too.
    for q in range(num_qubits):
        if q not in final_layer_params:
            final_layer_params[q] = get_random_u_params()

    # 3. Simulate the "Pre-Correction" State
    # We need to see where the circuit points BEFORE the final rotations roughly
    # (Technically we simulate up to the CZs, then apply the uncorrected final Us temporarily)
    
    print("Simulating to find natural peak...")
    # Create a temp circuit for simulation
    qc_sim = qc.copy()
    for q in range(num_qubits):
        p = final_layer_params[q]
        qc_sim.u(*p, q)
        
    # Get the statevector (Exponential cost: Limit to ~25 qubits)
    state = Statevector.from_instruction(qc_sim)
    probs = state.probabilities()
    
    # Identify the natural winner (The bitstring with highest probability)
    natural_winner_idx = np.argmax(probs)
    natural_winner_bin = format(natural_winner_idx, f'0{num_qubits}b')
    print(f"Natural Random Winner: {natural_winner_bin} (Prob: {probs[natural_winner_idx]:.4f})")
    
    # 4. Calculate Steering Corrections
    # We want TARGET_BITSTRING to win.
    # If natural is '0' and target is '1', we need to flip.
    # We apply this flip by modifying the FINAL U gate parameters.
    # U_new = X * U_old  (if flip needed)
    
    print(f"Steering towards target: {target_string}")
    decomposer = OneQubitEulerDecomposer(basis='U3')
    
    # We append the final layer to the REAL circuit now, with corrections
    # Note: iterating 0..N ensures we place gates in index order (aesthetic)
    for q in range(num_qubits):
        # 1. Reconstruct the intended random gate matrix
        theta, phi, lam = final_layer_params[q]
        u_gate = UnitaryGate(
            # Reconstruct U3 matrix from params
            # Qiskit's U gate logic: U(theta, phi, lam)
            # We can use a temporary circuit to get the matrix easily
            _params_to_matrix(theta, phi, lam)
        )
        
        # 2. Check if we need to flip this bit
        # Target string is big-endian (q0 is rightmost? Qiskit is usually Little-Endian)
        # Qiskit: q0 is rightmost bit. String "10" -> q1=1, q0=0.
        # Let's align string index to qubit index properly.
        # String "cn...c0"
        
        char_idx = (num_qubits - 1) - q  # Map q0 to last char of string
        target_bit = target_string[char_idx]
        natural_bit = natural_winner_bin[char_idx]
        
        final_unitary = u_gate.to_matrix()
        
        if target_bit != natural_bit:
            # Apply X gate correction: New_U = X . Old_U
            # Matrix multiplication order: X is applied AFTER the rotation
            x_mat = np.array([[0, 1], [1, 0]], dtype=complex)
            final_unitary = x_mat @ final_unitary
            
        # 3. Decompose back to U(theta, phi, lam)
        # decomposer returns tuple (theta, phi, lam)
        # We enforce 'U3' basis to match the file format 'u(...)'
        new_params = decomposer(final_unitary)
        # decomposer returns a circuit, we extract params
        # Actually easier: Qiskit 1.0+ decomposer returns DAG or circuit.
        # Let's use standard decomposition:
        
        qc.u(*new_params[0].params, q)

    return qc

def _params_to_matrix(theta, phi, lam):
    """Helper to get U3 matrix from params manually to avoid circuit overhead."""
    # Standard U3 matrix definition
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    exp_phi = np.exp(1j * phi)
    exp_lam = np.exp(1j * lam)
    exp_phi_lam = np.exp(1j * (phi + lam))
    
    return np.array([
        [cos, -exp_lam * sin],
        [exp_phi * sin, exp_phi_lam * cos]
    ])

if __name__ == "__main__":
    # Generate
    qc = build_and_steer_circuit(TARGET_BITSTRING, DEPTH, SEED)
    
    # Verify
    print("\nVerifying Result...")
    final_state = Statevector.from_instruction(qc)
    final_probs = final_state.probabilities()
    winner = np.argmax(final_probs)
    winner_str = format(winner, f'0{BITWIDTH}b')
    
    print(f"Final Circuit Peak:    {winner_str}")
    print(f"Target Was:            {TARGET_BITSTRING}")
    print(f"Success: {winner_str == TARGET_BITSTRING}")
    
    # print(qc.qasm())
