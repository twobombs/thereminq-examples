# based on https://arxiv.org/abs/2506.04315
# from https://g.co/gemini/share/edea93889667

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.extensions import UnitaryGate
from pyqrack.qiskit_provider import PyQrackSimulator

# Helper function to create the unitary U_alpha for a rotation around an arbitrary axis n
def create_u_alpha(alpha, n):
    """
    Creates a Qiskit UnitaryGate for a rotation by angle alpha around axis n.

    Args:
        alpha (float): The rotation angle.
        n (np.array): A 3D numpy array representing the rotation axis [nx, ny, nz].

    Returns:
        UnitaryGate: The Qiskit gate for the operation U_alpha.
    """
    # Normalize the axis vector
    n = n / np.linalg.norm(n)
    nx, ny, nz = n[0], n[1], n[2]

    # Pauli matrices
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # The generator of the rotation
    h_matrix = (nx * pauli_x + ny * pauli_y + nz * pauli_z) / 2

    # The unitary operator U = exp(-i * alpha * H)
    # Note: The paper defines U_alpha = exp(-i * alpha * n . sigma / 2)
    # This corresponds to a rotation of 'alpha' radians.
    from scipy.linalg import expm
    unitary_matrix = expm(-1j * alpha * h_matrix)

    return UnitaryGate(unitary_matrix, label=f'U({alpha:.2f})')

# --- Main Simulation Function ---
def run_phase_estimation_protocols(rotation_axis, alpha_range):
    """
    Runs simulations for the three phase estimation protocols from the paper.

    1. Positronium Metrology (Entangled Qubit-Antiqubit)
    2. Entanglement-Free Sensing (Separable Qubit-Antiqubit)
    3. Agnostic Sensing (Standard Entangled Qubits)

    Args:
        rotation_axis (np.array): The axis [nx, ny, nz] for the U_alpha rotation.
        alpha_range (np.array): An array of alpha values to sweep over.

    Returns:
        dict: A dictionary containing the probability results for each protocol.
    """
    # Use the high-performance PyQrack simulator
    backend = PyQrackSimulator()
    results = {
        'positronium': [],
        'entanglement_free_q': [],
        'entanglement_free_aq': [],
        'agnostic': []
    }

    print(f"Simulating for rotation axis: {rotation_axis}")

    for alpha in alpha_range:
        # Create the U_alpha gate and its inverse (for the antiqubit)
        u_alpha_gate = create_u_alpha(alpha, rotation_axis)
        u_alpha_dagger_gate = u_alpha_gate.inverse()
        u_alpha_dagger_gate.label = f'U_dag({alpha:.2f})'


        # --- Protocol 1: Positronium Metrology ---
        # Prepare a singlet, apply U(a) to qubit and U_dag(a) to antiqubit, then measure in singlet basis.
        qc_pos = QuantumCircuit(2, 1)
        # Create the singlet state |Ψ⁻> = (|01> - |10>)/sqrt(2)
        qc_pos.x(0)
        qc_pos.h(0)
        qc_pos.cx(0, 1)
        qc_pos.z(0) # Phase flip for |Ψ⁻>
        qc_pos.barrier()
        # Apply the unitaries
        qc_pos.append(u_alpha_gate, [0])
        qc_pos.append(u_alpha_dagger_gate, [1])
        qc_pos.barrier()
        # To measure in the singlet basis, we "un-prepare" the singlet and measure q0.
        # If it's |1>, the original state was the singlet.
        qc_pos.z(0)
        qc_pos.cx(0, 1)
        qc_pos.h(0)
        qc_pos.measure(0, 0)
        # Transpile and run
        t_qc_pos = transpile(qc_pos, backend)
        counts_pos = backend.run(t_qc_pos, shots=8192).result().get_counts()
        results['positronium'].append(counts_pos.get('1', 0) / 8192)


        # --- Protocol 2: Optimal Entanglement-Free Sensing ---
        # Prepare |x+> on qubit and |z+> on antiqubit. Evolve and measure.
        qc_ef = QuantumCircuit(2, 2)
        # Prepare initial state |x+> |z+>
        qc_ef.h(0) # |x+>
        # q1 starts in |0> which is |z+>
        qc_ef.barrier()
        # Apply unitaries
        qc_ef.append(u_alpha_gate, [0])
        qc_ef.append(u_alpha_dagger_gate, [1])
        qc_ef.barrier()
        # Measure qubit in X basis, antiqubit in Z basis
        qc_ef.h(0)
        qc_ef.measure([0, 1], [0, 1])
        # Transpile and run
        t_qc_ef = transpile(qc_ef, backend)
        counts_ef = backend.run(t_qc_ef, shots=8192).result().get_counts()
        # Probability of |x+> on qubit is P(c0=0)
        prob_x_plus = (counts_ef.get('00', 0) + counts_ef.get('10', 0)) / 8192
        # Probability of |z+> on antiqubit is P(c1=0)
        prob_z_plus = (counts_ef.get('00', 0) + counts_ef.get('01', 0)) / 8192
        results['entanglement_free_q'].append(prob_x_plus)
        results['entanglement_free_aq'].append(prob_z_plus)


        # --- Protocol 3: Standard Agnostic Sensing (No Antiqubit) ---
        # Prepare singlet, apply U(a) to qubit, do nothing to ancilla.
        qc_agn = QuantumCircuit(2, 1)
        # Create singlet state
        qc_agn.x(0)
        qc_agn.h(0)
        qc_agn.cx(0, 1)
        qc_agn.z(0)
        qc_agn.barrier()
        # Apply unitary to only the first qubit
        qc_agn.append(u_alpha_gate, [0])
        qc_agn.barrier()
        # Measure in singlet basis
        qc_agn.z(0)
        qc_agn.cx(0, 1)
        qc_agn.h(0)
        qc_agn.measure(0, 0)
        # Transpile and run
        t_qc_agn = transpile(qc_agn, backend)
        counts_agn = backend.run(t_qc_agn, shots=8192).result().get_counts()
        results['agnostic'].append(counts_agn.get('1', 0) / 8192)

    return results

# --- Main Execution and Plotting ---
if __name__ == '__main__':
    # Define simulation parameters
    alpha_range = np.linspace(0, 2 * np.pi, 100)
    # The paper shows results for x, y, z axes. We'll pick one, e.g., the y-axis.
    # The key result of positronium metrology is its axis-independence.
    rotation_axis = np.array([0, 1, 0]) # Let's use the y-axis

    # Run the simulations
    sim_results = run_phase_estimation_protocols(rotation_axis, alpha_range)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('Simulation of Positronium Metrology vs. Competitor Strategies', fontsize=16)

    # Plot 1: Positronium and Agnostic Sensing
    ax1.plot(alpha_range, sim_results['positronium'], 'o-', label=r'Positronium Metrology: $P(|\Psi^-\rangle)$', markersize=4)
    ax1.plot(alpha_range, sim_results['agnostic'], 's--', label=r'Agnostic Sensing: $P(|\Psi^-\rangle)$', markersize=4)
    # Theoretical curves
    ax1.plot(alpha_range, np.cos(2 * alpha_range / 2)**2, 'gray', linestyle=':', label=r'Theory: $\cos^2(\alpha)$ for Positronium')
    ax1.plot(alpha_range, np.cos(alpha_range / 2)**2, 'gray', linestyle='-.', label=r'Theory: $\cos^2(\alpha/2)$ for Agnostic')
    ax1.set_ylabel('Probability $P(|\Psi^-\rangle)$', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.set_title('Entangled Protocols', fontsize=14)
    ax1.grid(True)

    # Plot 2: Entanglement-Free Protocol
    ax2.plot(alpha_range, sim_results['entanglement_free_q'], 'o-', label=r'Qubit: $P(|x+\rangle)$', markersize=4, color='purple')
    ax2.plot(alpha_range, sim_results['entanglement_free_aq'], 's--', label=r'Antiqubit: $P(|z+\rangle)$', markersize=4, color='orange')
    ax2.set_xlabel(r'Rotation Angle $\alpha$ (radians)', fontsize=12)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_title('Entanglement-Free Protocol', fontsize=14)
    ax2.grid(True)
    ax2.set_xticks(np.arange(0, 2*np.pi + 0.1, np.pi/2))
    ax2.set_xticklabels([r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])


    # --- Fisher Information Analysis ---
    print("\n--- Theoretical Maximum Fisher Information Analysis ---")
    # For P(alpha) = cos^2(k * alpha), the max Fisher Info is 4*k^2
    # The rotation gate definition corresponds to a rotation of 'alpha'
    # The evolution of the singlet state results in a probability dependence of:
    # Positronium: |<Ψ⁻|U(a)⊗U_dag(a)|Ψ⁻>|^2 = |<Ψ⁻|U(a)²⊗I|Ψ⁻>|^2 = cos^2(alpha)
    # Agnostic: |<Ψ⁻|U(a)⊗I|Ψ⁻>|^2 = cos^2(alpha/2)
    # The paper's U_alpha = e^(-i*alpha*H) results in P = cos^2(alpha) -> FI=4.
    # My U_alpha = e^(-i*(alpha/2)*n.sigma) -> a rotation of 'alpha'. So...
    # Positronium: U_alpha_total = U_2alpha -> P=cos^2(alpha) -> FI = 4
    # Agnostic: U_alpha -> P=cos^2(alpha/2) -> FI = 1
    print(f"Positronium Metrology (P ~ cos^2(alpha)): Max FI = 4")
    print(f"Agnostic Sensing (P ~ cos^2(alpha/2)): Max FI = 1")
    # The competitor strategy has a max average FI of 4/3 ≈ 1.33
    print(f"Entanglement-Free Sensing: Max Avg. FI = 4/3 ≈ 1.33")


    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

