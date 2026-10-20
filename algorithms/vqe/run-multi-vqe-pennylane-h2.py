# Code using qml.qchem H (Jordan-Wigner), Manual UCCSD Ansatz
# *** Reverted to H2/STO-3G, keeping Qrack GPU Simulator ***
# WARNING: Ensure GPU/CUDA setup is correct for Qrack.

# original created by Dan Strano of the Unitary fund 
# https://github.com/vm6502q/pyqrack-examples/blob/main/vqe.py
# plot was added by Qwen2.5-Coder-32B-Instruct-Q5_K_S.gguf
# heavily modified and expanded by gemini 2.5
# experimental code - can change without notice

import os
import pennylane as qml
from pennylane import numpy as np
import numpy as original_np # Use original_np for non-autograd tasks
from autograd import value_and_grad
import pennylane.qchem
import matplotlib.pyplot as plt
import time
import cProfile
import pstats
import traceback

# --- Prerequisites ---
# Ensure you have the necessary libraries installed:
# pip install --upgrade pip
# pip install --prefer-binary openfermionpyscf pennylane-qchem matplotlib autograd pennylane-qrack
# For GPU: Requires NVIDIA GPU, compatible CUDA Toolkit, and Qrack compiled with GPU support.

# --- PySCF Import for FCI Calculation ---
try:
    import pyscf
    from pyscf import fci # Import fci for H2
    pyscf_imported = True
except ImportError:
    pyscf_imported = False
    print("Warning: PySCF not found. Cannot calculate FCI energy dynamically.")
    # Ensure PySCF is installed if needed for Hamiltonian generation/reference E

# --- Optimization Parameters (Restored for H2) ---
CONST_NUM_STEPS = 300
CONST_TOLERANCE = 1e-6
CONST_MAX_ITERATIONS = 300
CONST_DECAY_STEPS = 75
CONST_DECAY_RATE = 0.9
CONST_GRAD_NORM_TOL = 1e-6 # Restore stricter tolerance for H2
SEED = 42

# --- Ansatz definition (unchanged structure) ---
def ansatz_manual_uccsd(params, wires, hf_state, singles, doubles):
    """Applies the UCCSD ansatz manually using SingleExcitation and DoubleExcitation.

    Args:
        params (np.ndarray): Parameters (excitation amplitudes), ordered singles then doubles.
        wires (Sequence[int]): Wires to apply the ansatz on.
        hf_state (np.ndarray): Hartree-Fock state configuration.
        singles (list[list[int]]): List of single excitation wires.
        doubles (list[list[int]]): List of double excitation wires.
    """
    qml.BasisState(hf_state, wires=wires) # Prepare Hartree-Fock state

    param_idx = 0
    # Apply single excitations
    num_singles = len(singles)
    if num_singles > 0:
        for s_wires in singles:
            if len(s_wires) != 2:
                raise ValueError(f"Single excitation wires should have length 2, got {s_wires}")
            if param_idx >= len(params): raise IndexError("Not enough params for singles")
            qml.SingleExcitation(params[param_idx], wires=s_wires)
            param_idx += 1

    # Apply double excitations
    num_doubles = len(doubles)
    if num_doubles > 0:
        if param_idx + num_doubles > len(params):
            raise IndexError(f"Not enough params for doubles (needed {num_doubles}, have {len(params)-param_idx} left)")
        for d_wires in doubles:
            if len(d_wires) != 4:
                raise ValueError(f"Double excitation wires should have length 4, got {d_wires}")
            qml.DoubleExcitation(params[param_idx], wires=d_wires)
            param_idx += 1

    # Final check if all params were used (optional)
    if param_idx != len(params):
        print(f"Warning: Parameter length mismatch. Used {param_idx}, have {len(params)}.")

# --- Main execution block ---

# --- Molecule Definition: H2 ---
charge = 0
multiplicity = 1
symbols = ["H", "H"]

bohr_to_angstrom = 0.529177210903
angstrom_to_bohr = 1.0 / bohr_to_angstrom
bond_len_bohr = 1.39847
bond_len_ang = bond_len_bohr * bohr_to_angstrom

# Coordinates in Bohr for PennyLane qchem input
coordinates_bohr_list = [
    ["H", [0.0, 0.0, 0.0]],
    ["H", [0.0, 0.0, bond_len_bohr]],
]
geometry_bohr_array = original_np.array([atom[1] for atom in coordinates_bohr_list])

# --- Basis Set ---
basis = "sto-3g"
# No ECP needed for H2/sto-3g

# --- Active Space (Full space for H2/sto-3g) ---
electrons = 2
orbitals = 2 # Spatial orbitals
active_electrons = electrons # Use all electrons
active_orbitals = orbitals   # Use all orbitals
n_qubits = active_orbitals * 2 # 4 qubits for H2/sto-3g
# --------------------------------

print(f"\n--- Preparing H2 Calculation (Target: Qrack GPU) ---") # Updated Target
print(f"Symbols: {symbols}")
print(f"Basis: {basis}")
print(f"Active Space: Full ({active_electrons} electrons, {active_orbitals} spatial orbitals, {n_qubits} qubits)")
# ---------------------------------

# --- Profiling Setup ---
profiler = cProfile.Profile()
profiler.enable()

# --- Initialize variables ---
fci_energy = None # Renamed back from casci_energy
hamiltonian = None
# n_qubits defined above
actual_steps = 0
optimization_successful = False
energy_history = []
grad_norm_history = []
start_time = time.time()

# --- Main Logic Block with Error Handling ---
try:
    # --- Compute Hamiltonian using PennyLane qchem ---
    print(f"\nComputing Hamiltonian using PennyLane qchem (PySCF backend)...")
    hamiltonian, n_qubits_check = qml.qchem.molecular_hamiltonian(
        symbols,
        geometry_bohr_array, # Use numerical array
        charge=charge,
        mult=multiplicity,
        basis=basis,
        # No active space args needed if using full space (default), but specifying is fine too
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        mapping='jordan_wigner',
        method='pyscf'
    )
    if n_qubits_check != n_qubits:
         raise ValueError(f"Qubit number mismatch: expected {n_qubits}, got {n_qubits_check}")
    print(f"Successfully computed Hamiltonian for {n_qubits} qubits.")

    # --- Get Reference Energy (FCI using PySCF) ---
    fci_energy = None
    fci_energy_fallback = -1.13728383 # Restore H2 fallback value

    if pyscf_imported:
        try:
            print("\nAttempting to compute FCI energy explicitly using PySCF...")
            # Define molecule geometry for PySCF (in Angstrom)
            atom_pyscf_string = ""
            for symbol, coords_bohr in coordinates_bohr_list:
                coords_ang = [c * bohr_to_angstrom for c in coords_bohr]
                atom_pyscf_string += f"{symbol} {coords_ang[0]: .8f} {coords_ang[1]: .8f} {coords_ang[2]: .8f}; "
            atom_pyscf_string = atom_pyscf_string[:-2]

            mol = pyscf.gto.M(
                atom = atom_pyscf_string,
                basis = basis,
                # No ECP needed for H/F
                spin = multiplicity - 1,
                charge = charge,
                unit = 'Angstrom'
            )
            print("  PySCF Molecule object created.")

            # Run RHF
            mf = mol.RHF().run(verbose=0)
            print(f"  PySCF RHF energy: {mf.e_tot:.8f} Ha")

            # Run FCI (reverted from CASCI)
            print(f"  Running PySCF FCI...")
            # Use fci solver directly for full FCI
            fci_solver = fci.FCI(mf)
            fci_energy_pyscf = fci_solver.kernel()[0]
            print(f"Successfully computed FCI energy via PySCF: {fci_energy_pyscf:.8f} Ha")
            fci_energy = fci_energy_pyscf # Assign calculated energy

        except Exception as pyscf_err:
            print(f"\nWarning: An error occurred during PySCF FCI calculation: {pyscf_err}")
            traceback.print_exc(limit=2)
            fci_energy = fci_energy_fallback # Use fallback on error
            print(f"Using known reference FCI Energy = {fci_energy:.8f} Ha (fallback value)")
    else:
        # PySCF could not be imported
        fci_energy = fci_energy_fallback
        print(f"\nPySCF not available. Using known reference FCI Energy = {fci_energy:.8f} Ha (fallback value)")
    # --- <<< ---------------------------------------------------- >>> ---

    # --- Print Hamiltonian Info ---
    print("\n--- Hamiltonian (PennyLane qchem, Jordan-Wigner) ---") # Updated title
    num_terms = len(hamiltonian.ops) if hasattr(hamiltonian, 'ops') else 0
    print(f"(Hamiltonian has {num_terms} terms)")
    print("----------------------------------------------------\n")
    if num_terms == 0: raise ValueError("Hamiltonian generation failed.")

    # --- Get HF state for H2/sto-3g (4 qubits) ---
    hf_state = original_np.array([1, 1, 0, 0])
    print(f"Using HF state (qubit basis): {hf_state}")

    # --- Calculate Excitations for H2/sto-3g ---
    print(f"Calculating excitations using electrons={electrons}, n_qubits={n_qubits}...")
    singles, doubles = qml.qchem.excitations(electrons=electrons, orbitals=n_qubits)
    num_excitations = len(singles) + len(doubles)
    print(f"Computed {len(singles)} singles and {len(doubles)} doubles excitations.")
    print(f"Total UCCSD parameters: {num_excitations}") # Should be 2 for H2/sto-3g

    # --- Define Quantum Device USING Qrack Simulator with GPU ---
    print("\nAttempting to initialize Qrack GPU device...")
    print("  (Requires pennylane-qrack, NVIDIA GPU, CUDA, and Qrack built with GPU support)")
    use_gpu_flag = True
    try:
        dev = qml.device(
            "qrack.simulator",
            wires=n_qubits, # Now 4 wires
            use_gpu=use_gpu_flag
        )
        gpu_active_message = "(GPU presumably active)" if use_gpu_flag else "(GPU not requested)"
        print(f"Using device: {dev.name} {gpu_active_message} for {n_qubits} qubits")

    except Exception as qrack_err:
        print(f"\nCRITICAL ERROR: Could not load 'qrack.simulator' with use_gpu={use_gpu_flag}.")
        # ... (error messages remain the same) ...
        print(f"  Ensure 'pennylane-qrack' is installed.")
        if use_gpu_flag:
             print(f"  Ensure you have a compatible NVIDIA GPU, CUDA toolkit installed,")
             print(f"  and that Qrack was compiled/installed with GPU support.")
        print(f"Original error: {qrack_err}")
        traceback.print_exc(limit=2)
        raise SystemExit(1)

    # --- Define the Quantum Node (QNode) ---
    @qml.qnode(dev, diff_method="parameter-shift", interface="autograd")
    def scalar_circuit(params):
        ansatz_manual_uccsd(params, wires=range(n_qubits), hf_state=hf_state, singles=singles, doubles=doubles)
        return qml.expval(hamiltonian)

    # --- Optimizer Configuration ---
    initial_step_size = 0.05
    opt = qml.AdamOptimizer(stepsize=initial_step_size)
    print(f"Optimizer: Adam initial_step={initial_step_size}, decay_rate={CONST_DECAY_RATE}, decay_steps={CONST_DECAY_STEPS}")

    # --- Parameter Initialization ---
    theta = np.array(original_np.zeros(num_excitations), requires_grad=True)
    theta_shape = theta.shape
    total_num_params = num_excitations
    if total_num_params == 0: print("Warning: No UCCSD parameters generated.")
    else: print(f"Initialized {total_num_params} UCCSD parameters to zero.")

    # --- Calculate Initial Energy (HF Check) ---
    try:
        initial_energy = scalar_circuit(theta)
        print(f"\n---> Initial energy (params=0, H2, Qrack): {initial_energy:.8f} Ha <---\n") # Updated label
        expected_hf = -1.11675729 # Expected H2/sto-3g HF energy
        if abs(float(initial_energy) - expected_hf) > 0.001:
             print(f"Warning: Initial energy on Qrack differs significantly from expected HF energy (~ {expected_hf:.8f} Ha).")
        else:
             print(f"Initial energy calculation on Qrack matches expected HF energy.")
    except Exception as initial_calc_err:
        print(f"\n---> ERROR calculating initial energy on Qrack: {initial_calc_err} <---\n")
        traceback.print_exc(); raise

    # --- Data Storage & Timing ---
    print(f"Starting VQE optimization (H2, Qrack GPU) for max {CONST_NUM_STEPS} steps...") # Updated
    print(f"Convergence Criteria: Energy Tol={CONST_TOLERANCE}, Grad Norm Tol={CONST_GRAD_NORM_TOL}")

    # --- Value and Gradient Function ---
    value_and_grad_fn = value_and_grad(scalar_circuit, argnum=0)

    # --- Convergence Parameters ---
    previous_energy = float('inf')

    # --- Optimization Loop ---
    step_size = initial_step_size
    if total_num_params == 0:
        print("Skipping optimization loop.")
        energy_history.append(float(initial_energy)); grad_norm_history.append(0.0)
        actual_steps = 0; optimization_successful = True
    else:
        for step in range(CONST_NUM_STEPS):
            step_start_time = time.time()
            actual_steps += 1
            try:
                energy, grads = value_and_grad_fn(theta)
                current_energy = float(energy)
                grad_norm = float(original_np.linalg.norm(original_np.array(grads)))
                if not hasattr(grads, 'shape'): raise TypeError(f"Grads missing shape step {step+1}")
                if grads.shape != theta_shape: raise ValueError(f"Grad shape mismatch step {step+1}")

            except MemoryError as mem_err:
                 print(f"\nMEMORY ERROR during optimization step {step+1}: {mem_err}")
                 optimization_successful = False; break
            except Exception as step_err:
                 print(f"\nERROR during optimization step {step+1} on Qrack: {step_err}")
                 traceback.print_exc(); optimization_successful = False; break

            step_end_time = time.time()
            step_duration = step_end_time - step_start_time

            grad_norm_history.append(grad_norm)
            energy_history.append(current_energy)

            if step > 0 and step % CONST_DECAY_STEPS == 0:
                step_size *= CONST_DECAY_RATE
                opt = qml.AdamOptimizer(stepsize=step_size)
                print(f"  Step {step+1}: Decaying step size to {step_size:.6f}")

            theta = opt.apply_grad(grads, theta)

            # Print less frequently for potentially fast H2 calc
            if (step + 1) % 20 == 0 or step == 0:
                 print(f"Step {step + 1: >4}: Energy = {current_energy: .8f} Ha, Grad Norm = {grad_norm: .6f} (Took {step_duration:.3f} s)")

            # Convergence Checks
            energy_diff = abs(current_energy - previous_energy)
            if step > 0:
                if energy_diff < CONST_TOLERANCE: print(f"\nConvergence achieved: Energy change < tol at step {step+1}."); optimization_successful = True; break
                if grad_norm < CONST_GRAD_NORM_TOL: print(f"\nConvergence achieved: Grad norm < tol at step {step+1}."); optimization_successful = True; break
            if step >= CONST_MAX_ITERATIONS - 1: print(f"\nReached max allowed inner iterations ({CONST_MAX_ITERATIONS}). Stopping."); optimization_successful = True; break
            previous_energy = current_energy

        if actual_steps == CONST_NUM_STEPS and not optimization_successful: print(f"\nReached maximum VQE steps ({CONST_NUM_STEPS}) without meeting convergence tol."); optimization_successful = True

# --- Error Handling (Outer loop for setup errors) ---
except MemoryError as mem_err:
    print(f"\nMEMORY ERROR during setup/Hamiltonian generation: {mem_err}")
    optimization_successful = False
except Exception as main_exec_err:
    print(f"\nCRITICAL ERROR during main execution/setup: {main_exec_err}")
    traceback.print_exc()
    optimization_successful = False

# --- Disable Profiling ---
profiler.disable()

# --- Post-Optimization / Final Results ---
end_time = time.time(); total_time = end_time - start_time
print("\n" + "="*30)
print(f"Calculation Summary for H2/{basis} using Qrack GPU") # Updated
if optimization_successful and energy_history:
    final_energy = energy_history[-1]; final_grad_norm = grad_norm_history[-1] if grad_norm_history else float('nan')
    print(f"VQE Optimization Finished:")
    print(f"  Finished Steps:     {actual_steps}")
    print(f"  Final VQE Energy:   {final_energy: .8f} Ha")
    hf_ref_energy = float(initial_energy) if 'initial_energy' in locals() else None
    if hf_ref_energy is not None: print(f"  Initial HF Energy:  {hf_ref_energy:.8f} Ha")
    if fci_energy is not None:
        energy_error = abs(final_energy - fci_energy)
        # Check against fallback for H2
        fci_source = "(PySCF computed)" if abs(fci_energy - fci_energy_fallback) > 1e-9 else "(fallback)"
        print(f"  Reference FCI:      {fci_energy: .8f} Ha {fci_source}")
        print(f"  Absolute Error:     {energy_error: .8f} Ha ({energy_error*1000:.4f} mHa)")
    else: print(f"  Reference FCI:    Not Available")
    print(f"  Final Grad Norm:    {final_grad_norm: .6f}")
    print(f"  Total VQE Time:     {total_time:.2f} seconds") # Report in seconds for H2
    if actual_steps > 0: print(f"  Avg time/step:      {(total_time / actual_steps):.4f} seconds")
else:
    print("VQE Optimization did not complete successfully or was interrupted.")
    print(f"  Total Time Elapsed: {total_time:.2f} seconds")
print("="*30 + "\n")

# --- Visualization ---
if optimization_successful and energy_history and grad_norm_history and actual_steps > 0:
    print("Generating convergence plots...")
    try:
        plt.style.use('seaborn-v0_8-darkgrid'); plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(energy_history) + 1), energy_history, label='VQE Energy (UCCSD, QrackGPU)', marker='.', linestyle='-', markersize=4, color='forestgreen')
        if fci_energy is not None:
             fci_source = "(PySCF)" if abs(fci_energy - fci_energy_fallback) > 1e-9 else "(fallback)"
             plt.axhline(fci_energy, color='red', linestyle='--', label=f'FCI Energy ({fci_energy:.6f}) {fci_source}')
        if hf_ref_energy is not None: plt.axhline(hf_ref_energy, color='purple', linestyle=':', label=f'HF Energy ({hf_ref_energy:.6f})')
        window_size=min(10, actual_steps // 2 if actual_steps > 1 else 1)
        if len(energy_history) >= window_size and window_size > 0:
             moving_avg = original_np.convolve(original_np.array(energy_history), original_np.ones(window_size)/window_size, mode='valid')
             plt.plot(range(window_size, len(energy_history) + 1), moving_avg, label=f'Moving Avg ({window_size} steps)', linestyle=':', color='darkorange')
        plt.title(f'VQE Energy Convergence (H$_2$, {basis}, UCCSD, QrackGPU)') # Updated title
        plt.xlabel('Optimization Step'); plt.ylabel('Energy (Hartree)'); plt.legend(loc='best'); plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(grad_norm_history) + 1), grad_norm_history, label='Gradient Norm', marker='.', linestyle='-', markersize=4, color='crimson')
        plt.axhline(CONST_GRAD_NORM_TOL, color='gray', linestyle=':', label=f'Grad Tol ({CONST_GRAD_NORM_TOL:.1e})')
        plt.yscale('log'); plt.title('Gradient Norm Convergence (QrackGPU)')
        plt.xlabel('Optimization Step'); plt.ylabel('Gradient Norm (log scale)'); plt.legend(loc='upper right'); plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plot_filename = f"h2_{basis}_vqe_conv_qrackgpu_seed{SEED}.png" # Updated filename
        plt.savefig(plot_filename, dpi=300); print(f"Saved convergence plot to '{plot_filename}'")
    except Exception as plot_err: print(f"Warning: Could not generate/save plot: {plot_err}")
elif optimization_successful and actual_steps == 0: print("No optimization steps performed. No plot generated.")


# --- Profiling Results ---
print("\nProfiling Results:")
try:
    stats = pstats.Stats(profiler); stats.sort_stats(pstats.SortKey.CUMULATIVE); stats.print_stats(40)
except Exception as profile_err: print(f"Could not print profiling stats: {profile_err}")

print("\nScript finished.")

