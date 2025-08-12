# Quantum chemistry example
# Developed with help from (OpenAI custom GPT) Elara
# (Requires OpenFermion)
# modified by Gemini25 to accomodate csv import

import openfermion as of
from openfermionpyscf import run_pyscf
from openfermion.transforms import jordan_wigner

import multiprocessing
import numpy as np
import os


# Step 0: Set environment variables (before running script)

# On command line or by .env file, you can set the following variables
# QRACK_DISABLE_QUNIT_FIDELITY_GUARD=1: For large circuits, automatically "elide," for approximation
# QRACK_NONCLIFFORD_ROUNDING_THRESHOLD=[0 to 1]: Sacrifices near-Clifford accuracy to reduce overhead
# QRACK_QUNIT_SEPARABILITY_THRESHOLD=[0 to 1]: Rounds to separable states more aggressively
# QRACK_QBDT_SEPARABILITY_THRESHOLD=[0 to 0.5]: Rounding for QBDD, if actually used


import pandas as pd
import ast

# Load the data from CSV
df = pd.read_csv('import_clifford_vqe_min.csv')

# Iterate over each molecule in the CSV
for index, row in df.iterrows():
    symbol = row['Symbol']
    element_name = row['Element Name']
    ground_state_energy = row['Ground State Energy ( compound )']
    geometry_str = row['Geometry']

    # Safely evaluate the string to a Python object
    try:
        geometry = ast.literal_eval(geometry_str)
    except (ValueError, SyntaxError):
        print(f"Could not parse geometry for {element_name} ({symbol}). Skipping.")
        continue

    print(f"Running VQE for {element_name} ({symbol})")
    print(f"Ground State Energy: {ground_state_energy}")
    print(f"Geometry: {geometry}")

    # The rest of your VQE logic goes here, using the 'geometry' variable
    basis = "sto-3g"
    multiplicity = 1
    charge = 0

    # Step 2: Run PySCF and get the Hamiltonian
    molecule = of.MolecularData(geometry, basis, multiplicity, charge)
    molecule = run_pyscf(molecule)
    hamiltonian = molecule.get_molecular_hamiltonian()
    z_hamiltonian = jordan_wigner(hamiltonian)

    # Step 3: Define QrackVQE parameters
    n_qubits = of.count_qubits(z_hamiltonian)
    z_qubits = list(range(n_qubits))

    # Initial ansatz angles (you can customize this)
    theta = np.zeros(n_qubits)

    # Step 4: Define the bootstrap workers for multiprocessing
    def bootstrap_worker(args):
        z_hamiltonian, theta, indices = args
        best_energy = float('inf')
        best_flipped = None
        for i in range(1, 1 << len(indices)):
            temp_theta = theta.copy()
            flipped = []
            for j in range(len(indices)):
                if (i >> j) & 1:
                    temp_theta[indices[j]] = np.pi
                    flipped.append(np.pi)
                else:
                    flipped.append(0)
            
            # This is a placeholder for your actual energy calculation function
            # You would replace this with your QrackVQE call
            energy = 0
            for op, coeff in z_hamiltonian.terms.items():
                val = 1
                for q, pauli in op:
                    if pauli == 'Z':
                        val *= (1 - 2 * (temp_theta[q] / np.pi))
                energy += coeff * val
            
            if energy < best_energy:
                best_energy = energy
                best_flipped = flipped
        
        return indices, best_energy, best_flipped


    # Step 5: Define the multiprocessing bootstrap function
    def multiprocessing_bootstrap(z_hamiltonian, initial_theta, z_qubits):
        n_qubits = len(z_qubits)
        best_theta = initial_theta.copy()
        
        # This is a placeholder for your actual energy calculation function
        min_energy = 0
        for op, coeff in z_hamiltonian.terms.items():
            val = 1
            for q, pauli in op:
                if pauli == 'Z':
                    val *= (1 - 2 * (best_theta[q] / np.pi))
            min_energy += coeff * val

        print(f"Initial Energy: {min_energy}")
        
        iter_count = 0
        improved = True
        while improved:
            improved = False
            print(f"\nIteration {iter_count + 1}:")
            theta = best_theta.copy()

            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                args = [(z_hamiltonian, theta, (z_qubits[i],)) for i in range(n_qubits)]
                results = pool.map(bootstrap_worker, args)

            results.sort(key=lambda r: r[1])
            
            improved_1qb = False
            for indices, energy, flipped in results:
                if energy < min_energy:
                    min_energy = energy
                    best_theta[indices[0]] = flipped[0]
                    improved_1qb = True
                    print(f"  Qubit {indices[0]} flip accepted. New energy: {min_energy}")
                else:
                    print(f"  Qubit {indices[0]} flip rejected.")
            
            if improved_1qb:
                improved = True
                iter_count += 1
                continue

            if n_qubits < 2:
                break

            print(f"\n2-Qubit Bootstrap Iteration {iter_count + 1}:")
            
            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                args = []
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        args.append((z_hamiltonian, theta, (z_qubits[i], z_qubits[j])))
                results = pool.map(bootstrap_worker, args)

            results.sort(key=lambda r: r[1])
            indices, energy, flipped = results[0]
            if energy < min_energy:
                min_energy = energy
                for i in range(len(indices)):
                    best_theta[indices[i]] = flipped[i]
                improved = True
                print(f"  Qubits {indices} flip accepted. New energy: {min_energy}")
            else:
                print(f"  Qubit flips all rejected.")

            iter_count += 1
            
        return best_theta, min_energy

    # Run threaded bootstrap
    theta, min_energy = multiprocessing_bootstrap(z_hamiltonian, theta, z_qubits)

    print(f"\nFinal Results for {element_name} ({symbol}):")
    print("Optimal Parameters (theta):", theta)
    print("Minimum Energy (VQE):", min_energy)
    
    # --- Comparison Section ---
    print("\n--- Comparison with Ground State Energy ---")
    print(f"Ground State Energy (from CSV): {ground_state_energy}")
    difference = min_energy - ground_state_energy
    print(f"Difference (VQE - Ground State): {difference}")
    print("-" * 50)
