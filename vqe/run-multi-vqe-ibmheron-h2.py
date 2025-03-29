# Code using qml.qchem H (Jordan-Wigner), Manual UCCSD Ansatz
# *** Modified for H2/6-31G (8 qubits), targeting IBM Quantum Heron (ibm_brisbane) ***
# !!! EXTREME WARNING: Runs on REAL HARDWARE. Expect NOISE, SHOTS, QUEUE TIMES, COST. !!!
# !!! Requires IBM Quantum Account & API Token configured. !!!
# Current time: Friday, March 28, 2025 at 1:39 AM (Almelo, Overijssel, Netherlands)

import os
import pennylane as qml
from pennylane import numpy as np
import numpy as original_np # Use original_np for non-autograd tasks
# Removed: from autograd import value_and_grad # Not needed for SPSA VQE loop
import pennylane.qchem
import matplotlib.pyplot as plt
import time
import cProfile
import pstats
import traceback
import math # For sqrt

# --- Prerequisites ---
# Ensure you have the necessary libraries installed:
# pip install --upgrade pip
# pip install --prefer-binary openfermionpyscf pennylane-qchem matplotlib pennylane-qiskit qiskit-ibm-provider
# IBM Quantum account + API Token must be configured (e.g., via IBMProvider.save_account or environment variable)

# --- PySCF Import for FCI Calculation ---
try:
    import pyscf
    from pyscf import fci
    pyscf_imported = True
except ImportError:
    pyscf_imported = False
    print("Warning: PySCF not found. Cannot calculate reference energy dynamically.")

# --- IBM Provider Import and Setup ---
try:
    from qiskit_ibm_provider import IBMProvider
    # Load account using saved credentials or environment variables
    print("Attempting to load IBM Quantum Provider...")
    provider = IBMProvider()
    print("IBM Provider loaded successfully.")
    # Optional: List available backends to confirm target device exists
    # print("Available backends:", [b.name for b in provider.backends()])
except ImportError:
    print("CRITICAL ERROR: qiskit-ibm-provider not found. Install it: pip install qiskit-ibm-provider")
    provider = None
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load IBM Quantum Provider: {e}")
    print("  Ensure your API token is saved correctly (IBMProvider.save_account(token=...))")
    print("  or the QISKIT_IBM_TOKEN environment variable is set.")
    provider = None
# Exit if provider couldn't be loaded
if provider is None:
    raise SystemExit("Exiting due to IBM Provider loading failure.")


# --- Optimization Parameters ---
# Adjust steps for hardware - SPSA might need more iterations than gradient descent,
# but each step involves fewer quantum executions (typically 2).
# However, queue times dominate, so keep total steps modest initially.
CONST_NUM_STEPS = 50 # Total optimization steps for SPSA
CONST_TOLERANCE = 1e-3 # Looser tolerance suitable for noisy hardware
# CONST_MAX_ITERATIONS not directly applicable to SPSA loop structure
# CONST_DECAY_STEPS, CONST_DECAY_RATE not used by basic SPSA
# CONST_GRAD_NORM_TOL not applicable

# --- SPSA Specific Parameters (Using PennyLane defaults for simplicity) ---
# See PennyLane SPSAOptimizer docs for details if tuning is needed
spsa_a = 0.628 # Example default parameter scaling
spsa_c = 0.1   # Example step size scaling
spsa_A = 0.0
spsa_alpha = 0.602
spsa_gamma = 0.101

SEED = 42

# --- Hardware / Shot Settings ---
TARGET_BACKEND = "ibm_brisbane" # Example Heron R2 device
NUM_SHOTS = 4096 # Number of measurements for expectation value estimation

# --- Ansatz definition (unchanged structure) ---
def ansatz_manual_uccsd(params, wires, hf_state, singles, doubles):
    """Applies the UCCSD ansatz manually. (Implementation Skipped)"""
    qml.BasisState(hf_state, wires=wires)
    param_idx = 0; num_singles = len(singles); num_doubles = len(doubles)
    if num_singles > 0:
        for s_wires in singles:
            if len(s_wires) != 2: raise ValueError(f"Single wire len != 2: {s_wires}")
            if param_idx >= len(params): raise IndexError("Not enough params for singles")
            qml.SingleExcitation(params[param_idx], wires=s_wires); param_idx += 1
    if num_doubles > 0:
        if param_idx + num_doubles > len(params): raise IndexError(f"Not enough params for doubles")
        for d_wires in doubles:
            if len(d_wires) != 4: raise ValueError(f"Double wire len != 4: {d_wires}")
            qml.DoubleExcitation(params[param_idx], wires=d_wires); param_idx += 1
    if param_idx != len(params): print(f"Warning: Param length mismatch. Used {param_idx}, have {len(params)}.")

# --- Main execution block ---

# --- Molecule Definition: H2 / 6-31G (8 qubits) ---
charge = 0
multiplicity = 1
symbols = ["H", "H"]
bohr_to_angstrom = 0.529177210903
angstrom_to_bohr = 1.0 / bohr_to_angstrom
bond_len_bohr = 1.39847
bond_len_ang = bond_len_bohr * bohr_to_angstrom
coordinates_bohr_list = [ ["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, bond_len_bohr]], ]
geometry_bohr_array = original_np.array([atom[1] for atom in coordinates_bohr_list])
basis = "6-31g"
electrons = 2
# Expect 4 spatial orbitals -> 8 qubits
# --------------------------------

print(f"\n--- Preparing H2/6-31G Calculation (Target: IBMQ '{TARGET_BACKEND}') ---") # Updated Target
print(f"Symbols: {symbols}")
print(f"Basis: {basis}")
print(f"Electrons: {electrons}")
print(f"Target Backend: {TARGET_BACKEND}, Shots: {NUM_SHOTS}")
print(f"!!! WARNING: Running on REAL HARDWARE. Expect noise, queue times, cost. !!!")
# ---------------------------------

# --- Profiling Setup ---
profiler = cProfile.Profile()
profiler.enable()

# --- Initialize variables ---
fci_energy = None
hamiltonian = None
n_qubits = None # Will be set after Hamiltonian calculation
actual_steps = 0
optimization_successful = False
energy_history = []
# grad_norm_history removed
start_time = time.time()

# --- Main Logic Block with Error Handling ---
try:
    # --- Compute Hamiltonian using PennyLane qchem ---
    print(f"\nComputing Hamiltonian using PennyLane qchem (PySCF backend)...")
    hamiltonian, n_qubits_check = qml.qchem.molecular_hamiltonian(
        symbols, geometry_bohr_array, charge=charge, mult=multiplicity, basis=basis,
        mapping='jordan_wigner', method='pyscf'
    )
    n_qubits = n_qubits_check
    print(f"Successfully computed Hamiltonian for {n_qubits} qubits.")
    if n_qubits != 8:
        raise ValueError(f"ERROR: Expected 8 qubits for H2/6-31G, but got {n_qubits}!")
    else:
        print("  (Proceeding with 8 qubits as expected for H2/6-31G)")

    # --- Get Reference Energy (Full FCI using PySCF) ---
    fci_energy = None
    fci_energy_fallback = None # Lookup H2/6-31G FCI energy if needed (~ -1.15 Ha)
    if pyscf_imported:
        try:
            print("\nAttempting to compute FCI energy explicitly using PySCF...")
            atom_pyscf_string = "".join(f"{s} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}; " for s, c_bohr in coordinates_bohr_list for c in [np.array(c_bohr) * bohr_to_angstrom])[:-2]
            mol = pyscf.gto.M(atom=atom_pyscf_string, basis=basis, spin=multiplicity - 1, charge=charge, unit='Angstrom')
            print("  PySCF Molecule object created.")
            mf = mol.RHF().run(verbose=0)
            print(f"  PySCF RHF energy: {mf.e_tot:.8f} Ha")
            print(f"  Running PySCF Full FCI (for {electrons}e in {n_qubits} spin-orbitals)...")
            fci_solver = fci.FCI(mf)
            fci_energy_pyscf = fci_solver.kernel()[0]
            print(f"Successfully computed FCI energy via PySCF: {fci_energy_pyscf:.8f} Ha")
            fci_energy = fci_energy_pyscf
        except Exception as pyscf_err:
            print(f"\nWarning: An error occurred during PySCF FCI calculation: {pyscf_err}")
            traceback.print_exc(limit=2); fci_energy = fci_energy_fallback
            print(f"Could not compute reference FCI energy.")
    else: fci_energy = fci_energy_fallback; print(f"\nPySCF not available. Cannot compute reference FCI energy.")

    # --- Print Hamiltonian Info ---
    print("\n--- Hamiltonian (PennyLane qchem, Jordan-Wigner) ---")
    num_terms = len(hamiltonian.ops) if hasattr(hamiltonian, 'ops') else 0
    print(f"(Hamiltonian has {num_terms} terms)")
    print("----------------------------------------------------\n")
    if num_terms == 0: raise ValueError("Hamiltonian generation failed.")

    # --- Get HF state for H2 (2 electrons in 8 qubits) ---
    hf_state = original_np.array([1]*electrons + [0]*(n_qubits - electrons))
    print(f"Using HF state (qubit basis): {hf_state}")

    # --- Calculate Excitations for H2 (2e, 8 qubits) ---
    print(f"Calculating excitations using electrons={electrons}, n_qubits={n_qubits}...")
    singles, doubles = qml.qchem.excitations(electrons=electrons, orbitals=n_qubits)
    num_excitations = len(singles) + len(doubles)
    print(f"Computed {len(singles)} singles and {len(doubles)} doubles excitations.")
    print(f"Total UCCSD parameters: {num_excitations}") # Expect ~27

    # --- <<< Define Quantum Device using IBM Quantum Backend >>> ---
    print(f"\nAttempting to initialize IBM Quantum device '{TARGET_BACKEND}'...")
    try:
        # Note: provider loaded earlier
        dev = qml.device(
            "qiskit.ibmq",
            wires=n_qubits,
            backend=TARGET_BACKEND,
            provider=provider, # Pass the loaded provider object
            shots=NUM_SHOTS    # Specify number of shots for hardware
        )
        print(f"Using device: {dev.name} on backend '{TARGET_BACKEND}' with {NUM_SHOTS} shots.")

    except Exception as qiskit_err:
        print(f"\nCRITICAL ERROR: Could not load IBMQ device '{TARGET_BACKEND}'.")
        print(f"  Ensure the backend name is correct and you have access.")
        print(f"  Check pennylane-qiskit and qiskit-ibm-provider installation.")
        print(f"Original error: {qiskit_err}")
        traceback.print_exc(limit=2); raise SystemExit(1)
    # --- <<< ------------------------------------------------ >>> ---

    # --- Define the Quantum Node (QNode) ---
    # No diff_method specified, as SPSA optimizer doesn't use QNode's gradient calculation
    @qml.qnode(dev, interface="autograd")
    def scalar_circuit(params):
        """Cost function: evaluates energy on hardware"""
        ansatz_manual_uccsd(params, wires=range(n_qubits), hf_state=hf_state, singles=singles, doubles=doubles)
        # Expectation value based on finite shots
        return qml.expval(hamiltonian)

    # --- Optimizer Configuration (Switched to SPSA) ---
    # SPSA is generally better for noisy, shot-based optimization
    print(f"Optimizer: SPSA with max {CONST_NUM_STEPS} iterations (using default parameters).")
    opt = qml.SPSAOptimizer(maxiter=CONST_NUM_STEPS, c=spsa_c, a=spsa_a) # Pass max iterations here

    # --- Parameter Initialization ---
    theta = np.array(original_np.zeros(num_excitations), requires_grad=True)
    # SPSA works best with non-zero initial parameters sometimes, but start with zero for consistency
    # theta = np.array(original_np.random.normal(0, 0.1, num_excitations), requires_grad=True)
    print(f"Initialized {num_excitations} UCCSD parameters.")

    # --- Calculate Initial Energy (Check) ---
    # Run once to get initial estimate and check connection/compilation
    try:
        print("Calculating initial energy (may involve queue time)...")
        initial_energy = scalar_circuit(theta)
        print(f"\n---> Initial energy estimate (params=0, H2/6-31G, {TARGET_BACKEND}, {NUM_SHOTS} shots): {initial_energy:.8f} Ha <---\n")
        print(f"   (Compare to PySCF RHF: {mf.e_tot:.8f} Ha. Expect difference due to noise/shots.)")
    except Exception as initial_calc_err:
        print(f"\n---> ERROR calculating initial energy on {TARGET_BACKEND}: {initial_calc_err} <---\n")
        print(f"     This could be due to connection issues, transpilation problems, or hardware errors.")
        traceback.print_exc(); raise

    # --- Data Storage & Timing ---
    print(f"Starting VQE optimization (H2/6-31G {n_qubits} qubits, {TARGET_BACKEND}) for {CONST_NUM_STEPS} SPSA steps...")
    print(f"Convergence Criteria: Energy Tol={CONST_TOLERANCE} (Note: convergence hard on hardware)")
    print("!!! WARNING: Expect steps to be VERY SLOW due to QUEUE TIMES and hardware execution. !!!")

    # --- Convergence Parameters ---
    previous_energy = float(initial_energy) # Start with the estimated initial energy
    energy_history.append(previous_energy) # Log initial energy

    # --- Optimization Loop (Using SPSA) ---
    if total_num_params == 0:
        print("Skipping optimization loop.")
        actual_steps = 0; optimization_successful = True
    else:
        for step in range(CONST_NUM_STEPS): # Loop managed by maxiter in SPSAOptimizer essentially
            step_start_time = time.time()
            actual_steps += 1
            print(f"--- Starting VQE Step {step+1}/{CONST_NUM_STEPS} (SPSA) ---")
            try:
                # SPSA optimizer step takes the cost function and current parameters
                # It returns the updated parameters
                theta = opt.step(scalar_circuit, theta)

                # Calculate energy with new parameters for tracking (requires extra execution)
                # On hardware, might want to only do this every few steps to save cost/time
                if (step + 1) % 5 == 0 or step == CONST_NUM_STEPS - 1: # Check energy every 5 steps + last step
                     print(f"  Calculating energy for step {step+1}...")
                     current_energy = float(scalar_circuit(theta))
                     energy_history.append(current_energy)
                     print(f"  Energy at step {step+1}: {current_energy:.8f} Ha")
                else:
                     # Avoid calculating energy every step to save time/cost
                     # Append previous energy to keep list length consistent, or handle differently
                     current_energy = energy_history[-1] # Use last known energy
                     energy_history.append(current_energy)


            except Exception as step_err:
                 print(f"\nERROR during optimization step {step+1} on {TARGET_BACKEND}: {step_err}")
                 traceback.print_exc(); optimization_successful = False; break

            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            print(f"Step {step + 1: >4} completed. (Approx. Energy = {current_energy:.8f} Ha) (Took {step_duration:.2f} s including queue/run time)")

            # Convergence Check (basic energy check)
            energy_diff = abs(current_energy - previous_energy)
            if step > 0 and (step + 1) % 5 == 0: # Check convergence every 5 steps when energy is calculated
                 if energy_diff < CONST_TOLERANCE:
                     print(f"\nPotential Convergence: Energy change {energy_diff:.2e} < tolerance {CONST_TOLERANCE} between steps {step+1-5}-{step+1}.")
                     # Note: SPSA fluctuates, this isn't a guarantee of convergence.
                     # Might need more sophisticated checks (e.g., averaging over steps)
                     # For now, let SPSA run its full course decided by maxiter

            # Update previous_energy only when it's newly calculated
            if (step + 1) % 5 == 0 or step == CONST_NUM_STEPS - 1:
                previous_energy = current_energy

        # Mark as successful if loop completes without error (convergence not guaranteed)
        if actual_steps == CONST_NUM_STEPS:
             print(f"\nCompleted {CONST_NUM_STEPS} SPSA optimization steps.")
             optimization_successful = True

# --- Error Handling (Outer loop for setup errors) ---
except MemoryError as mem_err: # Less likely for H2
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
print(f"Calculation Summary for H2/{basis} ({n_qubits} qubits) using IBMQ '{TARGET_BACKEND}'") # Updated
if optimization_successful and energy_history:
    # Take the last *calculated* energy
    final_energy = energy_history[-1]
    # grad_norm is not available with SPSA
    print(f"VQE Optimization Attempt Finished:")
    print(f"  Finished Steps:     {actual_steps}")
    print(f"  Final VQE Energy:   {final_energy:.8f} Ha (Estimate from {NUM_SHOTS} shots, NOISY)")
    hf_ref_energy = float(initial_energy) if 'initial_energy' in locals() else None
    if hf_ref_energy is not None: print(f"  Initial Energy Est.:{hf_ref_energy:.8f} Ha (Noisy)")
    if fci_energy is not None:
        energy_error = abs(final_energy - fci_energy)
        fci_source = "(PySCF computed)" if fci_energy is not fci_energy_fallback else "(Not computed)"
        print(f"  Reference FCI:      {fci_energy: .8f} Ha {fci_source}")
        print(f"  Abs. Error vs FCI:  {energy_error: .8f} Ha ({energy_error*1000:.4f} mHa) (NOISY RESULT)")
    else: print(f"  Reference FCI:      Not Available")
    # print(f"  Final Grad Norm:    N/A (SPSA used)") # Removed grad norm
    print(f"  Total Wall Time:    {total_time / 60.0:.2f} minutes (includes queue time)")
    if actual_steps > 0: print(f"  Avg wall time/step: {(total_time / actual_steps):.2f} seconds")
else:
    print("VQE Optimization did not complete successfully or was interrupted.")
    print(f"  Total Time Elapsed: {total_time / 60.0:.2f} minutes")
print("="*30 + "\n")

# --- Visualization ---
# Remove gradient norm plot
if optimization_successful and energy_history and actual_steps > 0:
    print("Generating convergence plot (Energy only)...")
    # Filter out placeholder energies if energy wasn't calculated every step
    logged_steps = [i for i, e in enumerate(energy_history) if i==0 or (i+1-1)%5 == 0 or i == len(energy_history)-1] # Adjust indices based on logging frequency
    logged_energies = [energy_history[i] for i in logged_steps]
    plot_steps = np.array(logged_steps) + 1 # Convert step index to step number

    try:
        plt.style.use('seaborn-v0_8-darkgrid'); plt.figure(figsize=(8, 6)) # Single plot now
        # Energy Plot
        plt.plot(plot_steps, logged_energies, label=f'VQE Energy ({TARGET_BACKEND}, {NUM_SHOTS} shots)', marker='.', linestyle='-', markersize=6, color='dodgerblue') # Changed color
        if fci_energy is not None:
             fci_source = "(PySCF)" if fci_energy is not fci_energy_fallback else ""
             plt.axhline(fci_energy, color='red', linestyle='--', label=f'FCI Energy ({fci_energy:.6f}) {fci_source}')
        if hf_ref_energy is not None: plt.axhline(hf_ref_energy, color='purple', linestyle=':', label=f'Initial Energy ({hf_ref_energy:.6f})')

        plt.title(f'VQE Energy Convergence (H$_2$, {basis}, {n_qubits} qubits, UCCSD, {TARGET_BACKEND})') # Updated title
        plt.xlabel('Optimization Step (logged)'); plt.ylabel('Energy (Hartree)'); plt.legend(loc='best'); plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plot_filename = f"h2_{basis}_{n_qubits}q_vqe_conv_{TARGET_BACKEND}_seed{SEED}.png" # Updated filename
        plt.savefig(plot_filename, dpi=300); print(f"Saved convergence plot to '{plot_filename}'")
    except Exception as plot_err: print(f"Warning: Could not generate/save plot: {plot_err}")
elif optimization_successful and actual_steps == 0: print("No optimization steps performed. No plot generated.")


# --- Profiling Results ---
# Less useful for hardware runs due to waiting time, but keep for completeness
print("\nProfiling Results (Wall time dominated by job execution/queue):")
try:
    stats = pstats.Stats(profiler); stats.sort_stats(pstats.SortKey.CUMULATIVE); stats.print_stats(40)
except Exception as profile_err: print(f"Could not print profiling stats: {profile_err}")

print("\nScript finished.")
