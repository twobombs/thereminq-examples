# -*- coding: utf-8 -*-
"""
This script demonstrates a special-case workflow for approximating a
molecular Hamiltonian (for TbCl3) as an Ising model and solving it with a
specialized solver like PyQrackIsing.

This version uses an ACTIVE SPACE APPROXIMATION to make the calculation
computationally tractable.
"""

# First, ensure you have the necessary libraries installed:
# pip install pyscf openfermion scipy

import numpy as np
from pyscf import gto, scf, mcscf
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.ops import InteractionOperator, QubitOperator

# --- Mock PyQrackIsing library ---
class MockPyQrackIsing:
    def tfim_ground_state_energy(self, J, h, n_qubits):
        """
        A mock function that simulates finding the ground state energy
        of the 1D Transverse Field Ising Model.
        """
        return -n_qubits * np.sqrt(J**2 + h**2) / 2

pyqrackising = MockPyQrackIsing()


# --- Step 1: Classical Quantum Chemistry Calculation (PySCF) ---

def get_tbcl3_hamiltonian_active_space(n_active_electrons, n_active_orbitals):
    """
    Uses PySCF to calculate the active space Hamiltonian for the TbCl3 molecule.
    """
    print("--- Step 1: Running Classical Calculation for TbCl3 (PySCF) ---")
    
    mol = gto.Mole()
    bond_length = 2.8
    mol.atom = [
        ('Tb', ( 0., 0., 0.)),
        ('Cl', ( bond_length, 0., 0.)),
        ('Cl', (-bond_length * np.cos(np.pi/3),  bond_length * np.sin(np.pi/3), 0.)),
        ('Cl', (-bond_length * np.cos(np.pi/3), -bond_length * np.sin(np.pi/3), 0.))
    ]
    mol.spin = 6
    mol.basis = {'Tb': 'crenbl', 'Cl': '6-31g'}
    mol.ecp = {'Tb': 'crenbl'}
    mol.build()

    # --- Initial Hartree-Fock Calculation ---
    # Use a robust two-step calculation to ensure convergence.
    # 1. Run a stable ROHF calculation.
    print("Running stable ROHF calculation to get a good initial guess...")
    rohf_mf = scf.ROHF(mol)
    rohf_mf.init_guess = '1e'
    rohf_mf.max_cycle = 200 # Increase cycles for ROHF
    rohf_mf = rohf_mf.newton() # Use Newton solver for ROHF
    rohf_mf.run()

    # 2. Use the ROHF result as the initial guess for the final UHF calculation.
    print("\nRunning UHF calculation using ROHF result as initial guess...")
    mf = scf.UHF(mol)
    mf.max_cycle = 200 # Increase cycles for UHF
    dm0 = rohf_mf.make_rdm1()
    mf.run(dm0)

    # --- Active Space Calculation ---
    print(f"\nDefining active space: {n_active_electrons} electrons in {n_active_orbitals} orbitals.")
    cas = mcscf.CASCI(mf, n_active_orbitals, n_active_electrons)
    
    # Get the one-body integrals and core energy for the active space
    one_body_integrals, core_energy = cas.get_h1eff()

    # **FIX**: Manually transform the two-electron integrals using the active
    # space MO coefficients provided by the CASCI object. This is more robust
    # than relying on the internal get_h2eff function with a UHF reference.
    active_mo_coeff = cas.mo_coeff
    full_two_body_integrals_ao = mf.mol.intor('int2e', aosym='s1')
    two_body_integrals = np.einsum('pi,qj,ijkl,rk,sl->pqrs',
                                   active_mo_coeff, active_mo_coeff,
                                   full_two_body_integrals_ao,
                                   active_mo_coeff, active_mo_coeff,
                                   optimize=True)
    
    print(f"Active space defined. Core energy (constant offset): {core_energy:.8f} Hartree")
    
    return one_body_integrals, two_body_integrals, core_energy, n_active_orbitals

# --- Step 2: Fermion-to-Qubit Mapping (OpenFermion) ---

def map_to_qubits(one_body, two_body, core_energy, n_active_orbitals):
    """
    Uses OpenFermion to map the ACTIVE SPACE Hamiltonian to a qubit Hamiltonian.
    """
    print("\n--- Step 2: Mapping ACTIVE SPACE Hamiltonian to Qubits ---")
    
    n_qubits = 2 * n_active_orbitals

    # The constant part of the Hamiltonian is the core energy
    constant = core_energy
    
    # Reorder integrals for OpenFermion
    two_body_reordered = np.transpose(two_body, (0, 2, 3, 1))
    
    # Create the InteractionOperator for the active space
    hamiltonian_op = InteractionOperator(constant, one_body, two_body_reordered)
    fermion_hamiltonian = get_fermion_operator(hamiltonian_op)
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
    
    print(f"Transformation complete. Number of qubits required: {n_qubits}")
    return qubit_hamiltonian, n_qubits

# --- Step 3: Approximate General Hamiltonian to Ising Model ---

def approximate_to_ising(qubit_hamiltonian: QubitOperator):
    """
    Approximates a general QubitOperator into a 1D TFIM structure.
    """
    print("\n--- Step 3: Approximating to Ising Model Structure ---")
    
    j_terms, h_terms = [], []
    for term, coeff in qubit_hamiltonian.terms.items():
        if len(term) == 2 and all(op[1] == 'Z' for op in term):
            j_terms.append(np.real(coeff))
        elif len(term) == 1 and term[0][1] == 'X':
            h_terms.append(np.real(coeff))

    J = np.mean(j_terms) if j_terms else 0.0
    h = np.mean(h_terms) if h_terms else 0.0

    print(f"Approximation complete.")
    print(f"  - Extracted average J (coupling): {J:.6f}")
    print(f"  - Extracted average h (transverse field): {h:.6f}")
    
    return J, h

# --- Step 4: Quantum Simulation with PyQrackIsing ---

def run_ising_simulation(J, h, n_qubits):
    """
    Uses the (mock) PyQrackIsing library to solve the approximated Ising model.
    """
    print("\n--- Step 4: Running Simulation with PyQrackIsing ---")
    
    ising_energy = pyqrackising.tfim_ground_state_energy(J, h, n_qubits)
    
    print("PyQrackIsing simulation finished.")
    print(f"Ising Model Ground State Energy: {ising_energy:.8f} Hartree")
    
    return ising_energy

# --- Main Execution ---
if __name__ == "__main__":
    # --- Define the Active Space ---
    # This is now the most important parameter.
    # Let's choose a small but meaningful active space, e.g.,
    # the 6 unpaired electrons in 7 of the f-orbitals.
    # To keep it small for this demo, we'll use 4 electrons in 4 orbitals.
    NUM_ACTIVE_ELECTRONS = 56
    NUM_ACTIVE_ORBITALS = 56

    # Step 1: Get the active space Hamiltonian from PySCF
    one_body, two_body, core_energy, n_orbitals = get_tbcl3_hamiltonian_active_space(
        NUM_ACTIVE_ELECTRONS, NUM_ACTIVE_ORBITALS
    )
    
    # Step 2: Map the small active space Hamiltonian to qubits
    qubit_hamiltonian, n_qubits = map_to_qubits(
        one_body, two_body, core_energy, n_orbitals
    )
    
    # Step 3: Approximate the active space Hamiltonian to an Ising model
    J_approx, h_approx = approximate_to_ising(qubit_hamiltonian)
    
    # Step 4: Solve the approximated Ising model
    ising_model_energy = run_ising_simulation(J_approx, h_approx, n_qubits)
    
    print("\n--- Final Result ---")

