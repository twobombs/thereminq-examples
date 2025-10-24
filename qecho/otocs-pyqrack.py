# from https://arxiv.org/pdf/2510.19550
# gemini25 - draft

# --- Other imports and helper functions remain the same ---
import math
from pyqrack import QrackSimulator # Make sure pyqrack is installed

# --- rz, rx, ry, apply_zz_evolution, apply_dq_evolution, apply_trotter_step functions are unchanged ---
# (Include the previous definitions here)
# ... (rz definition) ...
# ... (rx definition) ...
# ... (ry definition) ...
# ... (apply_zz_evolution definition) ...
# ... (apply_dq_evolution definition) ...
# ... (apply_trotter_step definition) ...


def simulate_otoc_qbdd(n_qubits, couplings, time, n_steps, m_qubit, b_qubits):
    """
    Simulates the full OTOC protocol using the QBBD backend.

    Args:
        n_qubits (int): Total number of qubits (spins).
        couplings (dict): Dictionary mapping (q1, q2) tuples to coupling strengths.
                          Example: {(0, 1): {'zz': 0.5, 'dq': 1.2}, ...}
        time (float): Total forward/backward evolution time.
        n_steps (int): Number of Trotter steps.
        m_qubit (int): Index of the measurement qubit.
        b_qubits (list): List of indices for the butterfly qubits.

    Returns:
        float: The calculated OTOC value (expectation of Pauli-X on m_qubit).
    """
    # *** MODIFICATION HERE: Instantiate with QBBD backend ***
    # The exact flag might depend on the specific PyQrack version,
    # 'is_qbdd=True' is a common pattern. Check PyQrack docs if this fails.
    sim = QrackSimulator(n_qubits, is_qbdd=True)
    # **********************************************************

    dt = time / n_steps

    # 1. Initial State Preparation
    sim.x(m_qubit) # Pauli-X on measurement qubit

    # 2. Forward Evolution
    for _ in range(n_steps):
        apply_trotter_step(sim, couplings, dt, n_qubits, forward=True)

    # 3. Butterfly Operator
    for bq in b_qubits:
        sim.x(bq) # Example perturbation: a simple Pauli-X flip

    # 4. Backward Evolution
    for _ in range(n_steps):
        apply_trotter_step(sim, couplings, dt, n_qubits, forward=False)

    # 5. Measurement
    # Assuming exp_pauli_sum works with QBBD backend.
    # Define the Pauli string for <X> on m_qubit: [1.0, qubit_pauli_map]
    # qubit_pauli_map: list of N integers, 1=X at m_qubit, 0=I elsewhere
    pauli_op_map = [0] * n_qubits
    pauli_op_map[m_qubit] = 1 # 1 corresponds to Pauli X
    pauli_op = [1.0] + pauli_op_map
    
    # Note: The format for exp_pauli_sum might need adjustment based on PyQrack version.
    # The format [coeff, [p1, p2, ..., pn]] where pi is Pauli index (0=I, 1=X, 2=Y, 3=Z)
    # might be required by some versions. Let's stick to the previous PDF's format for now.
    # The format [1.0] + list(...) in the original PDF seems non-standard for typical exp_pauli_sum.
    # A more common format would be something like:
    # pauli_list = [[1.0, pauli_op_map]] # A list containing one term
    # otoc_value = sim.exp_pauli_sum(pauli_list)
    # Let's refine the Pauli op construction based on common conventions:
    pauli_list = []
    term_map = [0] * n_qubits # 0 = Pauli I
    term_map[m_qubit] = 1   # 1 = Pauli X
    pauli_list.append([1.0, term_map]) # Append the term [<coeff>, <pauli_indices>]

    otoc_value = sim.exp_pauli_sum(pauli_list) # Use the list-of-terms format

    return otoc_value

# --- Example Usage (Unchanged, but now calls simulate_otoc_qbdd) ---
if __name__ == '__main__':
    # Example usage for a 3-qubit system
    num_qubits = 3
    # Define couplings for the TARDIS-2 like Hamiltonian
    # H = d_01(YY-XX) + d_01(ZZ) + d_12(YY-XX) + d_12(ZZ) ...
    example_couplings = {
        (0, 1): {'zz': 1.5, 'dq': 0.8},
        (1, 2): {'zz': 1.2, 'dq': 1.0},
        (0, 2): {'zz': 0.5, 'dq': 0.3} # All-to-all coupling assumed here
    }
    measurement_qubit = 0
    butterfly_qubit_list = [2] # Perturb qubit 2
    total_time = 1.0 # Corresponds to 't' in C(t)
    trotter_steps = 20

    print(f"Simulating OTOC using QBBD backend for {num_qubits} qubits...")

    otoc = simulate_otoc_qbdd(
        n_qubits=num_qubits,
        couplings=example_couplings,
        time=total_time,
        n_steps=trotter_steps,
        m_qubit=measurement_qubit,
        b_qubits=butterfly_qubit_list
    )

    print(f"Simulated OTOC (QBDD) at time t={total_time}: {otoc}")
