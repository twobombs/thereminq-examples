# Quantum chemistry example
#
# This script performs a classical simulation of a VQE-like algorithm
# to find the ground state energy of molecules.
#
# It uses OpenFermion to handle the quantum chemistry calculations
# and multiprocessing to speed up the optimization part of the simulation.
#
# from https://github.com/vm6502q/pyqrack-examples/blob/main/algorithms/clifford_vqe_min.py
# 
# gemini25 merged this into the existing import modules with parameters
#

import openfermion as of
from openfermionpyscf import run_pyscf
from openfermion.transforms import jordan_wigner
from openfermion.ops import FermionOperator

import multiprocessing
import numpy as np
import os

import pandas as pd
import ast

# --- Step 1: Data Import and Setup ---
# The script reads molecular geometry and ground state energy from a CSV file.

# Load the data from CSV
df = pd.read_csv('import_clifford_vqe_min.csv')

# Iterate over each molecule in the CSV
for index, row in df.iterrows():
    symbol = row['Symbol']
    element_name = row['Element Name']
    ground_state_energy = row['Ground State Energy ( compound )']
    geometry_str = row['Geometry']

    # Safely evaluate the string to a Python object. This is important
    # because the 'Geometry' column is a string representation of a list.
    try:
        geometry = ast.literal_eval(geometry_str)
    except (ValueError, SyntaxError):
        print(f"Could not parse geometry for {element_name} ({symbol}). Skipping.")
        continue

    print(f"Running VQE for {element_name} ({symbol})")
    print(f"Ground State Energy: {ground_state_energy}")
    print(f"Geometry: {geometry}")

    # --- Step 2: Quantum Chemistry Calculations ---
    # We use OpenFermion and PySCF to generate the molecular Hamiltonian.
    # This part of the code is a standard quantum chemistry workflow.
    basis = "sto-3g"
    multiplicity = 1
    charge = 0

    # Create a MolecularData object to store the molecule's information.
    molecule = of.MolecularData(geometry, basis, multiplicity, charge)
    
    # Run PySCF to get the molecular orbital energies and integrals.
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
    
    # Get the molecular Hamiltonian in the second quantization representation.
    fermionic_hamiltonian = molecule.get_molecular_hamiltonian()
    
    # Apply the Jordan-Wigner transformation to convert the Hamiltonian
    # into a qubit representation.
    qubit_hamiltonian = jordan_wigner(fermionic_hamiltonian)
    
    n_qubits = of.count_qubits(qubit_hamiltonian)
    
    # Refactored: Create a Z-only Hamiltonian by iterating through terms.
    # This is more explicit and robust, ensuring only Z-only terms are used
    # for the simplified VQE approximation.
    z_hamiltonian_terms = []
    z_qubits = set()
    for pauli_string, coeff in qubit_hamiltonian.terms.items():
        # Skip if any X or Y in term
        if any(op in ('X', 'Y') for _, op in pauli_string):
            continue
        
        qubits = [q for q, op in pauli_string]
        z_hamiltonian_terms.append({'qubits': qubits, 'coeff': coeff})
        z_qubits.update(qubits)

    z_qubits = sorted(list(z_qubits))

    # --- Step 3: Define VQE Parameters and Bootstrap Workers ---
    
    # The compute_energy function calculates the exact energy for a given
    # state represented by a list of 0s and 1s (theta_bits).
    def compute_energy(theta_bits, z_hamiltonian_terms):
        """
        Computes the exact expectation value of a Z-Hamiltonian on a
        computational basis state.
        
        Args:
            theta_bits: list of 0/1 integers representing the state.
            z_hamiltonian_terms: list of {'qubits': [], 'coeff': float} terms.

        Returns:
            energy (float)
        """
        energy = 0.0
        for term in z_hamiltonian_terms:
            value = 1
            for qubit in term['qubits']:
                # The expectation value of a Z operator on a |0> state is +1,
                # and on a |1> state it is -1.
                if theta_bits[qubit] == 1:
                    value *= -1
            energy += term['coeff'] * value
        return energy
    
    # The bootstrap_worker function is a multiprocessing worker that tests
    # a specific set of qubit flips and returns the best energy found
    # within that subset.
    def bootstrap_worker(args):
        z_hamiltonian_terms, theta_bits, indices = args
        best_energy = float('inf')
        best_flipped = None
        
        # This loop iterates through all 2^len(indices) combinations of 0 and 1.
        for i in range(1 << len(indices)):
            temp_theta_bits = theta_bits.copy()
            flipped_values = []
            for j in range(len(indices)):
                if (i >> j) & 1:
                    temp_theta_bits[indices[j]] = 1 if temp_theta_bits[indices[j]] == 0 else 0
                flipped_values.append(temp_theta_bits[indices[j]])
            
            energy = compute_energy(temp_theta_bits, z_hamiltonian_terms)
            
            if energy < best_energy:
                best_energy = energy
                best_flipped = flipped_values
            
        return indices, best_energy, best_flipped

    # --- Step 4: The Multiprocessing Optimization Loop ---
    # This function simulates the "classical" optimization part of VQE.
    # It uses a greedy approach, first trying 1-qubit flips, then 2-qubit flips.
    def multiprocessing_bootstrap(z_hamiltonian_terms, z_qubits):
        n_qubits = len(z_qubits)
        
        # Start with a random initial state (all 0s or a random bit string).
        best_theta_bits = np.zeros(n_qubits, dtype=int)
        min_energy = compute_energy(best_theta_bits, z_hamiltonian_terms)
        
        print(f"Initial Energy: {min_energy}")
        
        iter_count = 0
        improved = True
        
        # The main loop continues as long as the energy can be improved.
        while improved:
            improved = False
            
            # --- 1-Qubit Bootstrap ---
            print(f"\nIteration {iter_count + 1}: 1-Qubit Flips")
            theta_bits = best_theta_bits.copy()

            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                args = [(z_hamiltonian_terms, theta_bits, (z_qubits[i],)) for i in range(n_qubits)]
                results = pool.map(bootstrap_worker, args)
            
            results.sort(key=lambda r: r[1])
            
            if results[0][1] < min_energy:
                min_energy = results[0][1]
                indices, energy, flipped = results[0]
                best_theta_bits[indices[0]] = flipped[0]
                improved = True
                print(f"  Qubit {indices[0]} flip accepted. New energy: {min_energy}")
            else:
                print("  Qubit flips all rejected.")
            
            # --- 2-Qubit Bootstrap ---
            # This part is only run if the 1-qubit flips didn't improve the energy.
            if not improved:
                if n_qubits < 2:
                    break
                print(f"\nIteration {iter_count + 1}: 2-Qubit Flips")
                theta_bits = best_theta_bits.copy()
                
                with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                    args = []
                    for i in range(n_qubits):
                        for j in range(i + 1, n_qubits):
                            args.append((z_hamiltonian_terms, theta_bits, (z_qubits[i], z_qubits[j])))
                    results = pool.map(bootstrap_worker, args)

                results.sort(key=lambda r: r[1])
                indices, energy, flipped = results[0]
                
                if energy < min_energy:
                    min_energy = energy
                    for i in range(len(indices)):
                        best_theta_bits[indices[i]] = flipped[i]
                    improved = True
                    print(f"  Qubits {indices} flip accepted. New energy: {min_energy}")
                else:
                    print("  Qubit flips all rejected.")

            iter_count += 1
        
        return best_theta_bits, min_energy

    # --- Step 5: Run the Simulation and Print Results ---
    # This is the main execution block for each molecule.
    theta, min_energy = multiprocessing_bootstrap(z_hamiltonian_terms, z_qubits)

    print(f"\nFinal Results for {element_name} ({symbol}):")
    print("Optimal Parameters (theta):", theta)
    print("Minimum Energy (VQE):", min_energy)
    
    # --- Comparison Section ---
    print("\n--- Comparison with Ground State Energy ---")
    print(f"Ground State Energy (from CSV): {ground_state_energy}")
    difference = min_energy - ground_state_energy
    print(f"Difference (VQE - Ground State): {difference}")
    print("-" * 50)

