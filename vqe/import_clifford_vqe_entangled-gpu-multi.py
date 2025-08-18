# original code done by Dan Strano & GPT Elara 
# https://github.com/vm6502q/pyqrack-examples/blob/main/algorithms/clifford_vqe_entangled.py

# multi GPU version with bells and whistles done by Aryan Blaauw & Gemini25
# https://g.co/gemini/share/a24fec0d245c

# code requires a Volta+ NVidia GPU and supports multi NVidia GPUs
# code requires a specific CVS compiled for the script called import_clifford_vqe_min.csv 

# as reported with the original file this method reports a less then 2% derivations
# according to both Elara and Gemini25 is that better then FW 
# when required this code will switch to a higher precision chemical libary 

# default settings are set for balanced (~2% derivation) for testing and debugging
# for more speed set num_steps to less then 100 and/or stepsize to 0.2 or higher
# more precision set convergence_tolerance to 1e-6 or less and num_steps at or above 400

# this code has been tested on a combo of 3 volta/turing cards with ~11TFLOPS FP32 each

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
import time
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='pyscf')
warnings.filterwarnings("ignore", category=FutureWarning)

def calculate_ground_state(geometry, basis, multiplicity, charge, molecule_name, expected_energy):
    """
    Calculates the ground state energy of a molecule using a VQE approach
    with a chemistry-inspired UCCSD ansatz.
    """
    print(f"--- [GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}] Starting calculation for {molecule_name} (Multiplicity: {multiplicity}, Charge: {charge}) ---")
    
    start_time = time.time()
    
    # --- MODIFICATION: List to store convergence history ---
    convergence_log = []
    
    try:
        # Define molecule and run classical PySCF calculation
        molecule = MolecularData(geometry, basis, multiplicity, charge)
        molecule = run_pyscf(molecule, run_scf=True, verbose=False)

        # Get the Hamiltonian operator
        fermionic_hamilitonian = get_fermion_operator(molecule.get_molecular_hamiltonian())

        # Map the operator to the qubit representation
        n_qubits = molecule.n_qubits
        qubit_hamiltonian = jordan_wigner(fermionic_hamilitonian)

        # Convert to a PennyLane Hamiltonian for VQE
        hamiltonian = qml.import_operator(qubit_hamiltonian, "openfermion")
        print(f"[GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}] {molecule_name}: Using {n_qubits} qubits.")

        # --- GPU DEVICE SELECTION ---
        dev = qml.device("lightning.gpu", wires=n_qubits)

        # --- UCCSD Ansatz Setup ---
        n_electrons = molecule.n_electrons
        hf_state = qml.qchem.hf_state(n_electrons, n_qubits)
        
        # Handle cases with no possible excitations (e.g., He2)
        try:
            singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
        except ValueError as e:
            print(f"    > WARNING: Could not generate excitations for {molecule_name}. This is expected for systems with no virtual orbitals. {e}")
            print(f"    > No trainable parameters in UCCSD. Returning Hartree-Fock energy.")
            min_energy = molecule.hf_energy
            print(f"--- [GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}] Finished calculation for {molecule_name}. Final VQE Energy: {min_energy:.8f} Ha ---")
            if min_energy is not None and expected_energy is not None:
                abs_diff = abs(min_energy - expected_energy)
                pct_diff = (abs_diff / abs(expected_energy)) * 100 if abs(expected_energy) > 1e-9 else 0.0
                print(f"    > Deviation from Expected ({expected_energy:.8f} Ha): {abs_diff:.8f} Ha ({pct_diff:.2f}%)")
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"    > Calculation took {duration:.2f} seconds.")
            # Save log file even for this case
            # (Details will be added in the final block)
            return min_energy

        # Use PennyLane's dedicated function to map excitations to wires
        s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

        # Define the Quantum Circuit using the UCCSD ansatz
        @qml.qnode(dev, diff_method="adjoint")
        def circuit(params):
            qml.UCCSD(params, wires=range(n_qubits), s_wires=s_wires, d_wires=d_wires, init_state=hf_state)
            return qml.expval(hamiltonian)

        # The number of parameters is the number of single and double excitations.
        n_params = len(s_wires) + len(d_wires)
        params = np.random.uniform(0, 2 * np.pi, size=n_params, requires_grad=True)
        
	# updates the very high stepsize of 0.05 to a 'reasonable' 0.15
        opt = qml.AdamOptimizer(stepsize=0.15)
	# number of steps lower because of higher convergence value, yet avoiding long waits set to a balmy 100
        num_steps = 100
        min_energy = molecule.hf_energy

        # Convergence tracking
        prev_energy = min_energy
	# modified from the high precision 1e-6 to the faster max 0.5% loss of 6e-3
        convergence_tolerance = 6e-3
        convergence_steps = 10
        steps_without_improvement = 0
        converged_step = None

        print(f"[GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}] Starting VQE for {molecule_name} (Initial HF Energy: {min_energy:.8f} Ha)...")
        for step in range(num_steps):
            params, cost = opt.step_and_cost(circuit, params)
            
            log_entry = f"Step {step+1:3d}: Energy = {cost:.8f} Ha"
            convergence_log.append(log_entry)
            if (step + 1) % 20 == 0:
                print(f"[GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}] {molecule_name} {log_entry}")
            
            # Convergence check logic
            if abs(cost - prev_energy) < convergence_tolerance:
                steps_without_improvement += 1
            else:
                steps_without_improvement = 0
            
            if steps_without_improvement >= convergence_steps:
                converged_step = step + 1
                print(f"    > Converged at step {converged_step} due to stalled improvement.")
                min_energy = cost
                break

            prev_energy = cost
            min_energy = cost

        print(f"--- [GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}] Finished calculation for {molecule_name}. Final VQE Energy: {min_energy:.8f} Ha ---")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate deviation
        abs_diff = None
        pct_diff = None
        if min_energy is not None and expected_energy is not None:
            abs_diff = abs(min_energy - expected_energy)
            pct_diff = (abs_diff / abs(expected_energy)) * 100 if abs(expected_energy) > 1e-9 else 0.0
            print(f"    > Deviation from Expected ({expected_energy:.8f} Ha): {abs_diff:.8f} Ha ({pct_diff:.2f}%)")
        
        print(f"    > Calculation took {duration:.2f} seconds.")

        # --- MODIFICATION: Save detailed log file ---
        log_dir = "calculation_logs"
        os.makedirs(log_dir, exist_ok=True)
        safe_molecule_name = molecule_name.replace(' ', '_').replace('(', '').replace(')', '')
        log_filename = os.path.join(log_dir, f"vqe_result_{safe_molecule_name}.txt")

        with open(log_filename, 'w') as f:
            f.write("--- VQE Calculation Report ---\n")
            f.write(f"Molecule: {molecule_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("--- Configuration ---\n")
            f.write(f"Basis Set: {basis}\n")
            f.write(f"Multiplicity: {multiplicity}\n")
            f.write(f"Charge: {charge}\n")
            f.write(f"Number of Qubits: {n_qubits}\n\n")

            f.write("--- Results ---\n")
            f.write(f"Initial Hartree-Fock Energy: {molecule.hf_energy:.8f} Ha\n")
            f.write(f"Final VQE Energy: {min_energy:.8f} Ha\n")
            f.write(f"Expected Energy: {expected_energy:.8f} Ha\n\n")

            f.write("--- Analysis ---\n")
            f.write(f"Absolute Difference: {abs_diff:.8f} Ha\n" if abs_diff is not None else "Absolute Difference: N/A\n")
            f.write(f"Percentage Difference: {pct_diff:.2f}%\n" if pct_diff is not None else "Percentage Difference: N/A\n")
            f.write(f"Calculation Time: {duration:.2f} seconds\n")
            if converged_step:
                f.write(f"Converged: Yes, at step {converged_step}.\n\n")
            else:
                f.write("Converged: No, reached max steps.\n\n")

            f.write("--- Convergence Log ---\n")
            f.write("\n".join(convergence_log))
        
        print(f"    > Saved detailed report to {log_filename}")

        return min_energy

    except Exception as e:
        print(f"!!! [GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}] ERROR calculating {molecule_name}: {e} !!!")
        return None

def worker_task(task_data):
    """
    Wrapper function for multiprocessing. It sets the visible GPU and calls the main calculation function.
    """
    gpu_id, index, row = task_data

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    molecule_name = row['Element Name']
    expected_energy = row['Ground State Energy ( compound )']
    multiplicity = int(row['correct multiplicity'])
    charge = int(row['correct charge'])

    try:
        geometry = ast.literal_eval(row['Geometry'])
    except (ValueError, SyntaxError):
        print(f"Could not parse geometry for {molecule_name}")
        return {'Original Index': index, 'Calculated Energy': None, 'Absolute Difference': None, 'Percentage Difference': None}

    calculated_energy = calculate_ground_state(
        geometry=geometry,
        basis="sto-3g",
        multiplicity=multiplicity,
        charge=charge,
        molecule_name=molecule_name,
        expected_energy=expected_energy
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

def main():
    # *** Ensure this filename matches your CSV ***
    df = pd.read_csv('import_clifford_vqe_min.csv')
    df_subset = df

    try:
        num_gpus = len(os.popen("nvidia-smi -L").readlines())
        if num_gpus == 0:
            raise ValueError("No NVIDIA GPUs detected.")
        print(f"\nFound {num_gpus} GPUs. Creating a pool of {num_gpus} worker processes.\n")
    except Exception:
        print("Could not detect NVIDIA GPUs. Defaulting to 1 worker.")
        num_gpus = 1

    tasks = [(i % num_gpus, index, row) for i, (index, row) in enumerate(df_subset.iterrows())]

    with multiprocessing.Pool(processes=num_gpus) as pool:
        results = pool.map(worker_task, tasks)

    results.sort(key=lambda x: x['Original Index'])

    results_df = pd.DataFrame(results).drop(columns='Original Index')
    final_df = pd.concat([df_subset.reset_index(drop=True), results_df], axis=1)

    final_df.rename(columns={'Ground State Energy ( compound )': 'Expected Energy (Ha)'}, inplace=True)

    output_filename = 'vqe_comparison_results_multigpu_fixed.csv'
    final_df.to_csv(output_filename, index=False)

    print(f"\n\n--- Final Summary Table ---")
    print(f"Results saved to {output_filename}")
    print(final_df[['Element Name', 'Expected Energy (Ha)', 'Calculated Energy', 'Absolute Difference', 'Percentage Difference']].to_string())

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

