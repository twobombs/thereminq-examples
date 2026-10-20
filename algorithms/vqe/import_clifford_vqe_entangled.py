import pandas as pd
import ast
import pennylane as qml
from pennylane import numpy as np
from openfermion.transforms import jordan_wigner
from openfermion import MolecularData, get_fermion_operator
from openfermionpyscf import run_pyscf
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='pyscf')
warnings.filterwarnings("ignore", category=FutureWarning)

def calculate_ground_state(geometry, basis, multiplicity, charge, molecule_name):
    """
    Calculates the ground state energy of a molecule using a VQE approach.
    """
    print(f"\n--- Starting calculation for {molecule_name} ---")
    try:
        # Define molecule and run classical PySCF calculation
        molecule = MolecularData(geometry, basis, multiplicity, charge)
        molecule = run_pyscf(molecule, run_scf=True, verbose=False)

        # Get the Hamiltonian operator
        fermionic_hamiltonian = get_fermion_operator(molecule.get_molecular_hamiltonian())
        
        # Map the operator to the qubit representation
        n_qubits = molecule.n_qubits
        qubit_hamiltonian = jordan_wigner(fermionic_hamiltonian)
        
        # Convert to a PennyLane Hamiltonian for VQE
        hamiltonian = qml.import_operator(qubit_hamiltonian, "openfermion")
        print(f"{molecule_name}: Using {n_qubits} qubits.")

        # Set up the quantum device
        dev = qml.device("default.qubit", wires=n_qubits)
        
        # Prepare the initial state (Hartree-Fock state)
        n_electrons = molecule.n_electrons
        hf_state = np.array([1 if i < n_electrons else 0 for i in range(n_qubits)])

        # Define the Quantum Circuit (VQE ansatz)
        @qml.qnode(dev)
        def circuit(params):
            qml.BasisState(hf_state, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
            return qml.expval(hamiltonian)

        # VQE optimizer setup
        n_layers = 2
        param_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        params = np.random.uniform(0, 2 * np.pi, size=param_shape, requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=0.1)
        num_steps = 40
        min_energy = molecule.hf_energy

        print(f"Starting VQE for {molecule_name} (Initial HF Energy: {min_energy:.8f} Ha)...")
        for step in range(num_steps):
            params, cost = opt.step_and_cost(circuit, params)
            if (step + 1) % 10 == 0:
                print(f"Step {step+1:2d}: Energy = {cost:.8f} Ha")
            min_energy = cost
        
        print(f"--- Finished calculation for {molecule_name}. Final VQE Energy: {min_energy:.8f} Ha ---")
        return min_energy

    except Exception as e:
        print(f"!!! ERROR calculating {molecule_name}: {e} !!!")
        return None

# --- Main Script Logic ---
df = pd.read_csv('import_clifford_vqe_min.csv')

# Use a smaller subset for this demonstration
df_subset = df.head(3)

results = []
for index, row in df_subset.iterrows():
    molecule_name = row['Element Name']
    expected_energy = row['Ground State Energy ( compound )']
    
    try:
        geometry = ast.literal_eval(row['Geometry'])
    except (ValueError, SyntaxError):
        print(f"Could not parse geometry for {molecule_name}")
        results.append({'Calculated Energy': None, 'Absolute Difference': None, 'Percentage Difference': None})
        continue

    calculated_energy = calculate_ground_state(
        geometry=geometry,
        basis="sto-3g",
        multiplicity=1,
        charge=0,
        molecule_name=molecule_name
    )
    
    # ***MODIFICATION: Immediate feedback is printed here***
    if calculated_energy is not None:
        abs_diff = abs(calculated_energy - expected_energy)
        pct_diff = (abs_diff / abs(expected_energy)) * 100 if abs(expected_energy) > 1e-9 else 0.0
        
        print(f"\nIMMEDIATE RESULT for {molecule_name}:")
        print(f"  > Expected Energy:  {expected_energy:.8f} Ha")
        print(f"  > Calculated Energy: {calculated_energy:.8f} Ha")
        print(f"  > Absolute Difference: {abs_diff:.8f} Ha")
        print(f"  > Percentage Difference: {pct_diff:.4f} %")
        
        results.append({
            'Calculated Energy': calculated_energy,
            'Absolute Difference': abs_diff,
            'Percentage Difference': pct_diff
        })
    else:
        results.append({'Calculated Energy': None, 'Absolute Difference': None, 'Percentage Difference': None})

# --- Display and Save Final Results ---
results_df = pd.DataFrame(results)
final_df = pd.concat([df_subset.reset_index(drop=True), results_df], axis=1)

final_df.rename(columns={'Ground State Energy ( compound )': 'Expected Energy (Ha)'}, inplace=True)

output_filename = 'vqe_comparison_results.csv'
final_df.to_csv(output_filename, index=False)

print(f"\n\n--- Final Summary Table ---")
print(f"Results saved to {output_filename}")
print(final_df[['Element Name', 'Expected Energy (Ha)', 'Calculated Energy', 'Absolute Difference', 'Percentage Difference']].to_string())
