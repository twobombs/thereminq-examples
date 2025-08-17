import pandas as pd
import ast
import pennylane as qml
from pennylane import numpy as np
from openfermion.transforms import jordan_wigner
from openfermion import MolecularData, get_fermion_operator
from openfermionpyscf import run_pyscf
import warnings
import multiprocessing
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='pyscf')
warnings.filterwarnings("ignore", category=FutureWarning)

def calculate_ground_state(geometry, basis, multiplicity, charge, molecule_name):
    """
    Calculates the ground state energy of a molecule using a VQE approach.
    This function will be executed in a separate process for each molecule.
    """
    print(f"--- [GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}] Starting calculation for {molecule_name} ---")
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
        print(f"[GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}] {molecule_name}: Using {n_qubits} qubits.")

        # --- GPU DEVICE SELECTION ---
        # This device will now correspond to the single GPU made visible
        # to this specific process by CUDA_VISIBLE_DEVICES.
        dev = qml.device("lightning.gpu", wires=n_qubits)
        
        # Prepare the initial state (Hartree-Fock state)
        n_electrons = molecule.n_electrons
        hf_state = np.array([1 if i < n_electrons else 0 for i in range(n_qubits)])

        # Define the Quantum Circuit (VQE ansatz)
        @qml.qnode(dev, diff_method="adjoint")
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

        print(f"[GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}] Starting VQE for {molecule_name} (Initial HF Energy: {min_energy:.8f} Ha)...")
        for step in range(num_steps):
            params, cost = opt.step_and_cost(circuit, params)
            if (step + 1) % 20 == 0: # Print less frequently for cleaner parallel output
                print(f"[GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}] {molecule_name} Step {step+1:2d}: Energy = {cost:.8f} Ha")
            min_energy = cost
        
        print(f"--- [GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}] Finished calculation for {molecule_name}. Final VQE Energy: {min_energy:.8f} Ha ---")
        return min_energy

    except Exception as e:
        print(f"!!! [GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}] ERROR calculating {molecule_name}: {e} !!!")
        return None

def worker_task(task_data):
    """
    Wrapper function for multiprocessing. It sets the visible GPU and calls the main calculation function.
    """
    # Unpack data for this task
    gpu_id, index, row = task_data
    
    # *** KEY STEP: Assign this worker process to a specific GPU ***
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    molecule_name = row['Element Name']
    expected_energy = row['Ground State Energy ( compound )']
    
    try:
        geometry = ast.literal_eval(row['Geometry'])
    except (ValueError, SyntaxError):
        print(f"Could not parse geometry for {molecule_name}")
        return {'Original Index': index, 'Calculated Energy': None, 'Absolute Difference': None, 'Percentage Difference': None}

    calculated_energy = calculate_ground_state(
        geometry=geometry,
        basis="sto-3g",
        multiplicity=1,
        charge=0,
        molecule_name=molecule_name
    )
    
    if calculated_energy is not None:
        abs_diff = abs(calculated_energy - expected_energy)
        pct_diff = (abs_diff / abs(expected_energy)) * 100 if abs(expected_energy) > 1e-9 else 0.0
        
        return {
            'Original Index': index,
            'Calculated Energy': calculated_energy,
            'Absolute Difference': abs_diff,
            'Percentage Difference': pct_diff
        }
    else:
        return {'Original Index': index, 'Calculated Energy': None, 'Absolute Difference': None, 'Percentage Difference': None}

# --- Main Script Logic ---
def main():
    # Ensure the CSV file is in the same directory as the script
    df = pd.read_csv('import_clifford_vqe_min.csv')

    # Use a smaller subset for demonstration, or the full df
    # df_subset = df.head(10) # Example: run the first 10 molecules
    df_subset = df

    # --- MULTIPROCESSING SETUP ---
    # Detect the number of available NVIDIA GPUs
    # This is a simple way; for robustness, a library like `pynvml` could be used.
    try:
        num_gpus = len(os.popen("nvidia-smi -L").readlines())
        if num_gpus == 0:
            raise ValueError("No NVIDIA GPUs detected.")
        print(f"\nFound {num_gpus} GPUs. Creating a pool of {num_gpus} worker processes.\n")
    except Exception:
        print("Could not detect NVIDIA GPUs. Defaulting to 1 worker.")
        num_gpus = 1

    # Create a list of tasks, assigning a GPU to each task in a round-robin fashion
    tasks = [(i % num_gpus, index, row) for i, (index, row) in enumerate(df_subset.iterrows())]

    # Use a multiprocessing Pool to execute the tasks in parallel
    with multiprocessing.Pool(processes=num_gpus) as pool:
        results = pool.map(worker_task, tasks)

    # --- Display and Save Final Results ---
    # Sort results to match the original DataFrame order
    results.sort(key=lambda x: x['Original Index'])

    results_df = pd.DataFrame(results).drop(columns='Original Index')
    final_df = pd.concat([df_subset.reset_index(drop=True), results_df], axis=1)

    final_df.rename(columns={'Ground State Energy ( compound )': 'Expected Energy (Ha)'}, inplace=True)

    output_filename = 'vqe_comparison_results_multigpu.csv'
    final_df.to_csv(output_filename, index=False)

    print(f"\n\n--- Final Summary Table ---")
    print(f"Results saved to {output_filename}")
    print(final_df[['Element Name', 'Expected Energy (Ha)', 'Calculated Energy', 'Absolute Difference', 'Percentage Difference']].to_string())

if __name__ == "__main__":
    # It's crucial to wrap the main execution logic in this block when using multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
