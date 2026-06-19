# original code done by Dan Strano & GPT Elara 
# https://github.com/vm6502q/pyqrack-examples/blob/main/algorithms/clifford_vqe_entangled.py

# multi GPU version with bells and whistles done by Aryan Blaauw & Gemini25
# Refactored for PyQrack OpenCL execution via Mesa/Rusticl (RADV)
# Includes dynamic hardware polling, guarded CSV parsing, HF fallback, and exact final energy

import ast
import multiprocessing
import os
import time
import warnings
from datetime import datetime

import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from openfermion.transforms import jordan_wigner
from openfermion import MolecularData, get_fermion_operator
from openfermionpyscf import run_pyscf

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='pyscf')
warnings.filterwarnings("ignore", category=FutureWarning)

def calculate_ground_state(geometry, basis, multiplicity, charge, molecule_name, expected_energy, ansatz_name, gpu_id):
    """
    Calculates the ground state energy of a molecule using a VQE approach.
    Allows for switching between different ansatz circuits.
    """
    print(f"--- [Rusticl Device {gpu_id}] Starting calculation for {molecule_name} (Ansatz: {ansatz_name}) ---")
    
    start_time = time.time()
    convergence_log = []
    
    try:
        # Handle isotopes like Deuterium (D) by treating them as Hydrogen (H)
        processed_geometry = [(atom[0].replace('D', 'H'), atom[1]) for atom in geometry]

        # Define molecule and run classical PySCF calculation
        molecule = MolecularData(processed_geometry, basis, multiplicity, charge)
        molecule = run_pyscf(molecule, run_scf=True, verbose=False)

        # Get the Hamiltonian operator
        fermionic_hamiltonian = get_fermion_operator(molecule.get_molecular_hamiltonian())

        # Map the operator to the qubit representation
        n_qubits = molecule.n_qubits
        qubit_hamiltonian = jordan_wigner(fermionic_hamiltonian)

        # Convert to a PennyLane Hamiltonian for VQE
        hamiltonian = qml.import_operator(qubit_hamiltonian, "openfermion")
        print(f"[Rusticl Device {gpu_id}] {molecule_name}: Using {n_qubits} qubits.")

        # --- PYQRACK OPENCL MESA DEVICE SELECTION ---
        # Route the device index explicitly for the OpenCL context rather than relying on env masks
        dev = qml.device("qrack.simulator", wires=n_qubits, device_id=int(gpu_id))

        # --- Ansatz Setup ---
        n_electrons = molecule.n_electrons
        hf_state = qml.qchem.hf_state(n_electrons, n_qubits)
        
        # --- Select the quantum circuit and prepare parameters ---
        if ansatz_name == 'Hardware-Efficient':
            n_layers = 4 
            param_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
            n_params = np.prod(param_shape)
            
            @qml.qnode(dev, diff_method="adjoint")
            def circuit(params):
                qml.BasisState(hf_state, wires=range(n_qubits))
                qml.StronglyEntanglingLayers(params.reshape(param_shape), wires=range(n_qubits))
                return qml.expval(hamiltonian)
        else: 
            ansatz_name = 'UCCSD'
            try:
                singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
            except ValueError as e:
                print(f"    > WARNING: Could not generate excitations for {molecule_name}. {e}")
                print(f"    > Falling back to uncorrelated Hartree-Fock energy (no VQE optimization was run).")
                hf_fallback_energy = molecule.hf_energy

                # Still write a log so this row is traceable as a fallback, not a real VQE result.
                log_dir = "calculation_logs"
                os.makedirs(log_dir, exist_ok=True)
                safe_molecule_name = molecule_name.replace(' ', '_').replace('(', '').replace(')', '')
                log_filename = os.path.join(log_dir, f"vqe_result_{safe_molecule_name}.txt")
                with open(log_filename, 'w') as f:
                    f.write("--- VQE Calculation Report ---\n")
                    f.write(f"Molecule: {molecule_name}\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("--- Status: FALLBACK (no VQE run) ---\n")
                    f.write(f"Reason: Could not generate UCCSD excitations: {e}\n")
                    f.write(f"Reported energy is the Hartree-Fock energy, NOT a VQE result.\n")
                    f.write(f"Hartree-Fock Energy: {hf_fallback_energy:.8f} Ha\n")
                    f.write(f"Expected Energy: {expected_energy:.8f} Ha\n")

                return hf_fallback_energy, "hf_fallback_no_uccsd"

            s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
            n_params = len(s_wires) + len(d_wires)
            
            @qml.qnode(dev, diff_method="adjoint")
            def circuit(params):
                qml.UCCSD(params, wires=range(n_qubits), s_wires=s_wires, d_wires=d_wires, init_state=hf_state)
                return qml.expval(hamiltonian)

        params = np.random.uniform(0, 2 * np.pi, size=n_params, requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=0.15)
        num_steps = 40
        min_energy = molecule.hf_energy

        # Convergence tracking
        prev_energy = min_energy
        convergence_tolerance = 1e-3
        convergence_steps = 10
        steps_without_improvement = 0
        converged_step = None

        print(f"[Rusticl Device {gpu_id}] Starting VQE for {molecule_name} (Initial HF Energy: {min_energy:.8f} Ha)...")
        for step in range(num_steps):
            params, cost = opt.step_and_cost(circuit, params)
            
            log_entry = f"Step {step+1:3d}: Energy = {cost:.8f} Ha"
            convergence_log.append(log_entry)
            if (step + 1) % 20 == 0:
                print(f"[Rusticl Device {gpu_id}] {molecule_name} {log_entry}")
            
            if abs(cost - prev_energy) < convergence_tolerance:
                steps_without_improvement += 1
            else:
                steps_without_improvement = 0
            
            if steps_without_improvement >= convergence_steps:
                converged_step = step + 1
                print(f"    > Converged at step {converged_step} due to stalled improvement.")
                break

            prev_energy = cost

        # Evaluate the true final energy with the converged parameters to fix the off-by-one quirk
        true_final_energy = float(circuit(params))

        print(f"--- [Rusticl Device {gpu_id}] Finished calculation for {molecule_name}. Final VQE Energy: {true_final_energy:.8f} Ha ---")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate deviation
        abs_diff = None
        pct_diff = None
        if true_final_energy is not None and expected_energy is not None:
            abs_diff = abs(true_final_energy - expected_energy)
            pct_diff = (abs_diff / abs(expected_energy)) * 100 if abs(expected_energy) > 1e-9 else 0.0
            print(f"    > Deviation from Expected ({expected_energy:.8f} Ha): {abs_diff:.8f} Ha ({pct_diff:.2f}%)")
        
        print(f"    > Calculation took {duration:.2f} seconds.")

        # Save detailed log file
        log_dir = "calculation_logs"
        os.makedirs(log_dir, exist_ok=True)
        safe_molecule_name = molecule_name.replace(' ', '_').replace('(', '').replace(')', '')
        log_filename = os.path.join(log_dir, f"vqe_result_{safe_molecule_name}.txt")

        with open(log_filename, 'w') as f:
            f.write("--- VQE Calculation Report ---\n")
            f.write(f"Molecule: {molecule_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("--- Configuration ---\n")
            f.write(f"Ansatz Type: {ansatz_name}\n")
            f.write(f"Basis Set: {basis}\n")
            f.write(f"Multiplicity: {multiplicity}\n")
            f.write(f"Charge: {charge}\n")
            f.write(f"Number of Qubits: {n_qubits}\n\n")

            f.write("--- Results ---\n")
            f.write(f"Initial Hartree-Fock Energy: {molecule.hf_energy:.8f} Ha\n")
            f.write(f"Final VQE Energy: {true_final_energy:.8f} Ha\n")
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

        return true_final_energy, "success"

    except Exception as e:
        print(f"!!! [Rusticl Device {gpu_id}] ERROR calculating {molecule_name}: {e} !!!")
        return None, "error"

def worker_task(task_data):
    """
    Wrapper function for multiprocessing. 
    Enforces the Mesa Rusticl ICD and sets the target OpenCL device via PyQrack environment bindings.
    """
    gpu_id, index, row = task_data

    # Enforce Rusticl and RADV dynamically for each spawned worker process
    os.environ["OCL_ICD_VENDORS"] = "rusticl.icd"
    os.environ["RUSTICL_ENABLE"] = "radv"
    
    # Directly force PyQrack's C++ backend to target the specific device index
    # to ensure tasks properly distribute if PennyLane drops the kwarg
    os.environ["QRACK_OCL_DEVICE"] = str(gpu_id)

    molecule_name = row.get('Element Name', f"row_{index}")

    def fail(reason):
        print(f"Skipping row {index} ({molecule_name}): {reason}")
        return {
            'Original Index': index,
            'Calculated Energy': None,
            'Absolute Difference': None,
            'Percentage Difference': None,
            'Status': f"data_error: {reason}",
        }

    # Guard all per-row data parsing
    try:
        expected_energy = float(row['Ground State Energy ( compound )'])
        multiplicity = int(row['correct multiplicity'])
        charge = int(row['correct charge'])
    except (KeyError, ValueError, TypeError) as e:
        return fail(f"invalid multiplicity/charge/expected energy ({e})")

    ansatz_name = 'UCCSD' if pd.isna(row.get('New Ansatz')) else row['New Ansatz']

    try:
        geometry = ast.literal_eval(row['Geometry'])
    except (ValueError, SyntaxError, KeyError) as e:
        return fail(f"could not parse geometry ({e})")

    try:
        calculated_energy, status = calculate_ground_state(
            geometry=geometry,
            basis="sto-3g",
            multiplicity=multiplicity,
            charge=charge,
            molecule_name=molecule_name,
            expected_energy=expected_energy,
            ansatz_name=ansatz_name,
            gpu_id=gpu_id
        )
    except Exception as e:
        return fail(f"unhandled exception in calculate_ground_state ({e})")

    if calculated_energy is not None:
        abs_diff = abs(calculated_energy - expected_energy)
        pct_diff = (abs_diff / abs(expected_energy)) * 100 if abs(expected_energy) > 1e-9 else 0.0

        return {
            'Original Index': index,
            'Calculated Energy': calculated_energy,
            'Absolute Difference': abs_diff,
            'Percentage Difference': pct_diff,
            'Status': status,
        }
    else:
        return {
            'Original Index': index,
            'Calculated Energy': None,
            'Absolute Difference': None,
            'Percentage Difference': None,
            'Status': status,
        }

def main():
    df = pd.read_csv('import_clifford_vqe_min.csv')
    df_subset = df

    # Dynamically poll the Rusticl platform for available devices
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        rusticl_platform = next((p for p in platforms if 'Rusticl' in p.name), platforms[0])
        num_gpus = len(rusticl_platform.get_devices())
        if num_gpus == 0:
            raise ValueError("No Rusticl devices found in PyOpenCL.")
        print(f"\nPyOpenCL detected {num_gpus} devices on the Rusticl platform.")
    except Exception as e:
        print(f"\nPyOpenCL polling failed or Rusticl not found ({e}). Defaulting to 4 instances.")
        num_gpus = 4

    # Set the number of workers per instance
    workers_per_gpu = 2
    num_processes = num_gpus * workers_per_gpu
    
    print(f"Configured for {num_gpus} OpenCL instances via Mesa/Rusticl. Creating a pool of {num_processes} worker processes ({workers_per_gpu} per instance).\n")

    tasks = [(i % num_gpus, index, row) for i, (index, row) in enumerate(df_subset.iterrows())]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(worker_task, tasks)

    results.sort(key=lambda x: x['Original Index'])

    results_df = pd.DataFrame(results).drop(columns='Original Index')
    final_df = pd.concat([df_subset.reset_index(drop=True), results_df], axis=1)

    final_df.rename(columns={'Ground State Energy ( compound )': 'Expected Energy (Ha)'}, inplace=True)

    output_filename = 'vqe_comparison_results_pyqrack_rusticl.csv'
    final_df.to_csv(output_filename, index=False)

    print(f"\n\n--- Final Summary Table ---")
    print(f"Results saved to {output_filename}")
    print(final_df[['Element Name', 'Expected Energy (Ha)', 'Calculated Energy', 'Absolute Difference', 'Percentage Difference', 'Status']].to_string())

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
