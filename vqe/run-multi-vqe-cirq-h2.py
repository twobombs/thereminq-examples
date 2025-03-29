# Code using qml.qchem H (Jordan-Wigner), Manual UCCSD Ansatz
# *** Modified to use cirq.simulator AND calculate FCI explicitly via PySCF ***
# *** Includes fix for IndexError during PySCF geometry handling ***
# Current time: Wednesday, March 26, 2025 at 9:31 AM (Hengelo, Overijssel, Netherlands)

import os
import pennylane as qml
from pennylane import numpy as np
# Use original_np for things not needing autograd, like coordinates/linalg
import numpy as original_np
from autograd import value_and_grad
import pennylane.qchem
import matplotlib.pyplot as plt
import time
import cProfile
import pstats
import traceback

# --- Prerequisites ---
# Ensure you have the necessary libraries installed, preferably using binary wheels:
# pip install --upgrade pip
# pip install --prefer-binary openfermionpyscf pennylane-qchem pennylane-cirq cirq matplotlib autograd
# (Successfully installing openfermionpyscf likely installed pyscf, numpy, scipy via wheels)

# --- PySCF Import for FCI Calculation ---
try:
    import pyscf
    from pyscf import fci
    pyscf_imported = True
except ImportError:
    pyscf_imported = False
    print("Warning: PySCF not found. Cannot calculate FCI energy dynamically.")
    print("Ensure 'openfermionpyscf' or 'pyscf' installed correctly.")

# --- Optimization Parameters ---
CONST_NUM_STEPS = 300
CONST_TOLERANCE = 1e-6
CONST_MAX_ITERATIONS = 300 # Max iterations *within* the loop before stopping if no other convergence
CONST_DECAY_STEPS = 75
CONST_DECAY_RATE = 0.9
CONST_GRAD_NORM_TOL = 1e-6
SEED = 42

# --- Ansatz definition using manual UCCSD steps ---
def ansatz_manual_uccsd(params, wires, hf_state, singles, doubles):
    """Applies the UCCSD ansatz manually using SingleExcitation and DoubleExcitation.

    Args:
        params (np.ndarray): Parameters (excitation amplitudes), ordered singles then doubles.
        wires (Sequence[int]): Wires to apply the ansatz on.
        hf_state (np.ndarray): Hartree-Fock state configuration [1, 1, 0, 0].
        singles (list[list[int]]): List of single excitation wires, e.g., [[0, 2], [1, 3]].
        doubles (list[list[int]]): List of double excitation wires, e.g., [[0, 1, 2, 3]].
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

# --- Molecule Definition ---
multiplicity = 1
charge = 0
symbols = ["H", "H"]
# Coordinates used by PennyLane qchem (in Bohr) - Flattened [H1x, H1y, H1z, H2x, H2y, H2z]
coordinates_bohr = original_np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.39847])
basis = 'sto-3g'
electrons = 2
orbitals = 2

# --- Profiling Setup ---
print("\nStarting VQE Optimization with Profiling (Using Cirq Simulator)...")
profiler = cProfile.Profile()
profiler.enable()

# --- Initialize variables ---
fci_energy = None
hamiltonian = None
n_qubits = None
actual_steps = 0
optimization_successful = False
energy_history = []
grad_norm_history = []
start_time = time.time()

# --- Main Logic Block with Error Handling ---
try:
    # --- Direct Hamiltonian Calculation (Keep using PennyLane qchem) ---
    # This part remains the same, as it's needed for the VQE part
    print(f"Computing Hamiltonian directly using PennyLane qchem (Basis: {basis})...")
    hamiltonian, n_qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        coordinates_bohr, # Use Bohr coordinates here
        charge=charge,
        mult=multiplicity,
        basis=basis,
        active_electrons=electrons,
        active_orbitals=orbitals,
        mapping='jordan_wigner'
    )
    print(f"Successfully computed Hamiltonian for {n_qubits} qubits using qml.qchem (Jordan-Wigner).")

    # --- <<< Get Reference FCI Energy (Explicitly using PySCF with corrected geometry handling) >>> ---
    fci_energy_fallback = -1.13728383 # Define fallback value example value is for H2

    if pyscf_imported:
        try:
            print("\nAttempting to compute FCI energy explicitly using PySCF...")

            # --- CORRECTED GEOMETRY HANDLING ---
            # Define conversion factor
            bohr_to_angstrom = 0.529177210903

            # Reshape the flat array into per-atom coordinates (2 atoms, 3 coords each)
            # Ensure using original_np for this calculation if coordinates_bohr is original_np
            coords_2d_bohr = coordinates_bohr.reshape(2, 3) # Now [[0,0,0], [0,0,1.39847]]

            # Calculate distance vector between atom 1 and atom 0
            dist_vec_bohr = coords_2d_bohr[1, :] - coords_2d_bohr[0, :]

            # Calculate the norm (length) of the distance vector using original numpy
            bond_length_bohr = original_np.linalg.norm(dist_vec_bohr)

            # Convert bond length to Angstrom for PySCF
            bond_length_angstrom = bond_length_bohr * bohr_to_angstrom
            print(f"  Calculated bond length: {bond_length_bohr:.5f} Bohr / {bond_length_angstrom:.5f} Angstrom")

            # Build the PySCF molecule geometry string
            # Assuming one H is at origin, the other along Z axis
            atom_string = f'H 0 0 0; H 0 0 {bond_length_angstrom}'
            # --- END OF CORRECTED GEOMETRY HANDLING ---

            mol = pyscf.gto.M(
                atom = atom_string, # Use the calculated string
                basis = basis,
                spin = multiplicity - 1, # PySCF uses spin = 2S (0 for singlet)
                charge = charge,
                unit = 'Angstrom' # Make sure unit matches geometry
            )
            print("  PySCF Molecule object created.")

            # Run RHF (Restricted Hartree-Fock)
            mf = mol.RHF().run(verbose=0) # Use verbose=0 to suppress PySCF output
            print(f"  PySCF RHF energy: {mf.e_tot:.8f} Ha")

            # Run FCI
            cisolver = fci.FCI(mf)
            # kernel() returns tuple (E_FCI, CI_vector)
            fci_energy_pyscf, fci_wavefunction = cisolver.kernel()
            print(f"Successfully computed FCI energy via PySCF: {fci_energy_pyscf:.8f} Ha")
            fci_energy = fci_energy_pyscf # Assign to the main variable

        except Exception as pyscf_err:
            print(f"\nWarning: An error occurred during PySCF FCI calculation: {pyscf_err}")
            traceback.print_exc(limit=2) # Show a bit more traceback for debugging
            fci_energy = fci_energy_fallback # Use fallback on error
            print(f"Using known reference FCI Energy = {fci_energy:.8f} Ha (fallback value)")
    else:
        # PySCF could not be imported
        fci_energy = fci_energy_fallback
        print(f"\nPySCF not available. Using known reference FCI Energy = {fci_energy:.8f} Ha (fallback value)")
    # --- <<< ---------------------------------------------------- >>> ---


    # --- Print Hamiltonian Info ---
    print("\n--- Computed Hamiltonian (qml.qchem, Jordan-Wigner) ---")
    num_terms = len(hamiltonian.ops) if hasattr(hamiltonian, 'ops') else len(hamiltonian.coeffs)
    print(f"(Hamiltonian has {num_terms} terms)")
    print("---------------------------------------\n")

    # --- Get HF state and Excitations ---
    hf_state = original_np.array([1, 1, 0, 0]) # For H2 STO-3G: |1100>
    print(f"Using pre-defined HF state (qubit basis): {hf_state}")

    print(f"Calculating excitations using electrons={electrons}, spin-orbitals={n_qubits}...")
    singles, doubles = qml.qchem.excitations(electrons=electrons, orbitals=n_qubits)
    num_excitations = len(singles) + len(doubles)
    print(f"Computed singles: {singles}")
    print(f"Computed doubles: {doubles}")
    print(f"Computed {len(singles)} singles and {len(doubles)} doubles excitations.")
    print(f"Total UCCSD parameters: {num_excitations}")

    # --- Define Quantum Device USING Cirq Simulator ---
    try:
        dev = qml.device("cirq.simulator", wires=n_qubits)
        print(f"Using device: {dev.name} (via pennylane-cirq)")
    except qml.DeviceError as e:
        print(f"\nCRITICAL ERROR: Could not load 'cirq.simulator'. Ensure prerequisites are installed.")
        print(f"Original error: {e}")
        raise SystemExit(1)

    # --- Define the Quantum Node (QNode) using Cirq simulator ---
    # Uses PennyLane's numpy (np) for parameters to enable autograd
    @qml.qnode(dev, diff_method="parameter-shift", interface="autograd")
    def scalar_circuit(params):
        """Quantum circuit applying the UCCSD ansatz MANUALLY."""
        ansatz_manual_uccsd(params, wires=range(n_qubits), hf_state=hf_state, singles=singles, doubles=doubles)
        return qml.expval(hamiltonian)

    # --- Optimizer Configuration ---
    initial_step_size = 0.05
    opt = qml.AdamOptimizer(stepsize=initial_step_size)
    print(f"Optimizer: Adam with initial step size {initial_step_size}, decay rate {CONST_DECAY_RATE} every {CONST_DECAY_STEPS} steps.")

    # --- Parameter Initialization ---
    # Use PennyLane's numpy (np) for parameters that require gradients
    theta = np.array(original_np.zeros(num_excitations), requires_grad=True)
    theta_shape = theta.shape
    total_num_params = num_excitations
    if total_num_params == 0: print("Warning: No UCCSD parameters generated.")
    else: print(f"Initialized {total_num_params} UCCSD parameters to zero.")

    # --- Calculate and print initial energy (HF Check using Cirq simulator) ---
    try:
        initial_energy = scalar_circuit(theta) # Pass PennyLane numpy array
        print(f"\n---> Initial energy (params=0, JW H, cirq.simulator): {initial_energy:.8f} Ha <---\n")
        expected_hf = -1.11675729
        # Note: Accuracy might slightly differ between simulators due to floating point math
        if abs(float(initial_energy) - expected_hf) > 0.001:
            print(f"Warning: Initial energy on cirq.simulator differs significantly from expected HF energy (~ {expected_hf:.8f} Ha).")
        else:
            print(f"Initial energy calculation on cirq.simulator matches expected HF energy.")
    except Exception as initial_calc_err:
        print(f"\n---> ERROR calculating initial energy on cirq.simulator: {initial_calc_err} <---\n")
        traceback.print_exc()
        raise

    # --- Data Storage & Timing ---
    print(f"Starting VQE optimization on cirq.simulator for max {CONST_NUM_STEPS} steps...")
    print(f"Convergence Criteria: Energy Tol={CONST_TOLERANCE}, Grad Norm Tol={CONST_GRAD_NORM_TOL}")

    # --- Value and Gradient Function ---
    # Will compute gradient w.r.t the PennyLane numpy array 'theta'
    value_and_grad_fn = value_and_grad(scalar_circuit, argnum=0)

    # --- Convergence Parameters ---
    previous_energy = float('inf')

    # --- Optimization Loop ---
    step_size = initial_step_size
    # actual_steps initialized earlier

    if total_num_params == 0:
        print("Skipping optimization loop as there are no parameters.")
        # Use float() to ensure standard Python floats are stored if needed later
        energy_history.append(float(initial_energy))
        grad_norm_history.append(0.0)
        actual_steps = 0
        optimization_successful = True
    else:
        for step in range(CONST_NUM_STEPS):
            actual_steps += 1
            try:
                # Pass PennyLane numpy array 'theta' to the function
                energy, grads = value_and_grad_fn(theta)
                current_energy = float(energy) # Store as standard float

                # Gradient Handling and Checks
                # Ensure grads is a PennyLane numpy array before norm calculation if needed
                # Or convert to original_np if using original_np.linalg.norm
                if not isinstance(grads, (np.ndarray, original_np.ndarray)):
                    # Try converting to PennyLane numpy array first
                    try: grads_np = np.array(grads, requires_grad=False)
                    except Exception: grads_np = original_np.array(grads) # Fallback to original numpy
                else:
                    grads_np = grads # Already a numpy type

                # Use original_np for norm calculation as it doesn't need autograd
                grad_norm = float(original_np.linalg.norm(original_np.array(grads_np)))

                if not hasattr(grads_np, 'shape'): raise TypeError(f"Grads missing shape step {step+1}")
                if grads_np.shape != theta_shape: raise ValueError(f"Grad shape mismatch step {step+1}: {grads_np.shape} vs {theta_shape}")

            except Exception as step_err:
                print(f"\nERROR during optimization step {step+1}: {step_err}")
                traceback.print_exc()
                optimization_successful = False
                break # Stop optimization on error

            # Store history (use standard Python floats)
            grad_norm_history.append(grad_norm)
            energy_history.append(current_energy)

            # Optimizer Step Size Decay
            if step > 0 and step % CONST_DECAY_STEPS == 0:
                step_size *= CONST_DECAY_RATE
                opt = qml.AdamOptimizer(stepsize=step_size) # Re-init optimizer with new step size
                print(f"  Step {step+1}: Decaying step size to {step_size:.6f}")

            # Apply Gradient Update - opt.apply_grad expects PennyLane numpy arrays
            theta = opt.apply_grad(grads, theta) # Pass original grads from value_and_grad_fn

            # Print Progress
            if (step + 1) % 10 == 0 or step == 0:
                print(f"Step {step + 1: >4}: Energy = {current_energy: .8f} Ha, Grad Norm = {grad_norm: .6f}")

            # Convergence Checks
            energy_diff = abs(current_energy - previous_energy)
            if step > 0: # Avoid checking convergence on the very first step
                if energy_diff < CONST_TOLERANCE:
                    print(f"\nConvergence achieved: Energy change ({energy_diff:.2e}) < tolerance ({CONST_TOLERANCE}) at step {step+1}.")
                    optimization_successful = True
                    break
                if grad_norm < CONST_GRAD_NORM_TOL:
                    print(f"\nConvergence achieved: Gradient norm ({grad_norm:.2e}) < tolerance ({CONST_GRAD_NORM_TOL}) at step {step+1}.")
                    optimization_successful = True
                    break

            # Check against Max Iterations for convergence criteria (CONST_MAX_ITERATIONS)
            if step >= CONST_MAX_ITERATIONS - 1:
                 print(f"\nReached maximum allowed iterations for convergence check ({CONST_MAX_ITERATIONS}). Stopping.")
                 optimization_successful = True # Treat as finished if max iterations hit
                 break

            previous_energy = current_energy
        # --- End of the for loop ---

        # Final check if loop finished due to hitting CONST_NUM_STEPS without converging
        if actual_steps == CONST_NUM_STEPS and not optimization_successful:
             print(f"\nReached maximum number of steps ({CONST_NUM_STEPS}) without meeting convergence tolerance.")
             optimization_successful = True # Mark as finished, even if not fully converged by tolerance


# --- Error Handling (Outer loop for setup errors) ---
except Exception as main_exec_err:
    print(f"\nCRITICAL ERROR during main execution/setup: {main_exec_err}")
    traceback.print_exc()
    optimization_successful = False

# --- Disable Profiling ---
profiler.disable()

# --- Post-Optimization / Final Results ---
end_time = time.time()
total_time = end_time - start_time
print("\n" + "="*30)
if optimization_successful and energy_history:
    final_energy = energy_history[-1]
    final_grad_norm = grad_norm_history[-1] if grad_norm_history else float('nan')
    print(f"VQE Optimization Finished (Manual UCCSD Ansatz, JW H, cirq.simulator):")
    print(f"  Finished Steps:     {actual_steps}")
    print(f"  Final Energy:       {final_energy: .8f} Ha")
    hf_ref_energy = initial_energy if 'initial_energy' in locals() else None
    if hf_ref_energy is not None:
        print(f"  Initial HF Energy:  {hf_ref_energy:.8f} Ha") # Should be Pennylane numpy, cast to float if needed
    if fci_energy is not None:
        energy_error = abs(final_energy - fci_energy)
        # Check if the fallback value was used by comparing to the known fallback constant
        fci_source = "(PySCF computed)" if abs(fci_energy - fci_energy_fallback) > 1e-9 else "(fallback)"
        print(f"  Reference FCI:      {fci_energy: .8f} Ha {fci_source}")
        print(f"  Absolute Error:     {energy_error: .8f} Ha ({energy_error*1000:.4f} mHa)")
    print(f"  Final Grad Norm:    {final_grad_norm: .6f}")
    print(f"  Total VQE Time:     {total_time: .2f} seconds")
else:
    print("VQE Optimization did not complete successfully or was interrupted.")
    print(f"  Total Time Elapsed: {total_time: .2f} seconds")
print("="*30 + "\n")

# --- Visualization ---
if optimization_successful and energy_history and grad_norm_history and actual_steps > 0:
    print("Generating convergence plots...")
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(14, 6))
        # Energy Plot
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(energy_history) + 1), energy_history, label='VQE Energy (Manual UCCSD, JW, Cirq)', marker='.', linestyle='-', markersize=4, color='dodgerblue')
        if fci_energy is not None:
            fci_source = "(PySCF)" if abs(fci_energy - fci_energy_fallback) > 1e-9 else "(fallback)"
            fci_label = f'FCI Energy ({fci_energy:.6f}) {fci_source}'
            plt.axhline(fci_energy, color='red', linestyle='--', label=fci_label)
        # Ensure initial_energy is treated as a float for plotting if it exists
        hf_plot_energy = float(initial_energy) if 'initial_energy' in locals() and initial_energy is not None else None
        if hf_plot_energy is not None:
             plt.axhline(hf_plot_energy, color='purple', linestyle=':', label=f'HF Energy ({hf_plot_energy:.6f})')
        window_size=10
        if len(energy_history) >= window_size:
             # Use original_np for convolution if energy_history is standard list/array
             moving_avg = original_np.convolve(original_np.array(energy_history), original_np.ones(window_size)/window_size, mode='valid')
             plt.plot(range(window_size, len(energy_history) + 1), moving_avg, label=f'Moving Avg ({window_size} steps)', linestyle=':', color='darkorange')
        plt.title(f'VQE Energy Convergence (H$_2$, {basis}, Manual UCCSD, JW, cirq.simulator)')
        plt.xlabel('Optimization Step'); plt.ylabel('Energy (Hartree)')
        plt.legend(loc='best'); plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Gradient Norm Plot
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(grad_norm_history) + 1), grad_norm_history, label='Gradient Norm', marker='.', linestyle='-', markersize=4, color='mediumvioletred')
        plt.axhline(CONST_GRAD_NORM_TOL, color='gray', linestyle=':', label=f'Grad Tol ({CONST_GRAD_NORM_TOL:.1e})')
        plt.yscale('log'); plt.title('Gradient Norm Convergence (Manual UCCSD, JW, cirq.simulator)')
        plt.xlabel('Optimization Step'); plt.ylabel('Gradient Norm (log scale)')
        plt.legend(loc='upper right'); plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plot_filename = f"h2_{basis}_vqe_convergence_seed{SEED}_ManualUCCSD_JW_cirq_FCI_pyscf_fixed.png" # Updated filename
        plt.savefig(plot_filename, dpi=300)
        print(f"Saved convergence plot to '{plot_filename}'")
        # plt.show() # Uncomment to display plot interactively
    except Exception as plot_err:
        print(f"Warning: Could not generate/save plot: {plot_err}")
elif optimization_successful and actual_steps == 0:
    print("No optimization steps performed. No plot generated.")

# --- Profiling Results ---
print("\nProfiling Results:")
try:
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.CUMULATIVE) # Sort by cumulative time spent in function
    stats.print_stats(30) # Print top 30 entries
except Exception as profile_err:
    print(f"Could not print profiling stats: {profile_err}")

print("\nScript finished.")

