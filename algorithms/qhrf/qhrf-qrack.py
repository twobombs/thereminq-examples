# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
# Ensure pyqrack is installed: pip install pyqrack
from pyqrack import QrackSimulator # Use QrackSimulator directly
import warnings
import time

print("--- Qrack Simulation (out_ket fix, NOISE DISABLED) ---")

# --- Simulation Parameters ---
t1_default_ns = 50000.0
t2_default_ns = 30000.0
gate_time_ns = 50.0
qhrf_improvement_factor = 5.0
t1_qhrf_ns = t1_default_ns * qhrf_improvement_factor
t2_qhrf_ns = t2_default_ns * qhrf_improvement_factor
t2_default_ns = min(t2_default_ns, 2 * t1_default_ns)
t2_qhrf_ns = min(t2_qhrf_ns, 2 * t1_qhrf_ns)
t1_default_s = t1_default_ns * 1e-9
t2_default_s = t2_default_ns * 1e-9
t1_qhrf_s = t1_qhrf_ns * 1e-9
t2_qhrf_s = t2_qhrf_ns * 1e-9
time_step_s = gate_time_ns * 1e-9
max_time_ns = t1_default_ns * 1.5
num_steps = int(round(max_time_ns / gate_time_ns))
time_points_ns = np.linspace(0, max_time_ns, num_steps + 1)
time_points_s = time_points_ns * 1e-9
initial_state_choice = '+'

print(f"Using defined defaults: T1={t1_default_ns} ns, T2={t2_default_ns} ns")
print(f"Simulation time step derived from gate time: {gate_time_ns} ns")

# --- Prepare Ideal State (Numpy array) ---
if initial_state_choice == '1':
    initial_state_label = '|1>'
    ideal_state_np = np.array([0, 1], dtype=complex)
elif initial_state_choice == '+':
    initial_state_label = '|+>'
    ideal_state_np = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
else:
    raise ValueError("initial_state_choice must be '1' or '+'")

# --- Helper Function for Fidelity ---
def calculate_fidelity(ideal_state_vector, sim_state_vector):
    """Calculates fidelity between two state vectors."""
    # Input sim_state_vector can now be list or numpy array
    sim_state_np = np.array(sim_state_vector, dtype=complex)
    sim_norm = np.linalg.norm(sim_state_np)
    if sim_norm < 1e-9: return 0.0
    normalized_sim_state = sim_state_np / sim_norm
    fidelity = np.abs(np.vdot(ideal_state_vector, normalized_sim_state))**2
    return fidelity

# --- Run Simulations ---
fidelities_default = []
fidelities_qhrf = []

print(f"\nSimulating for initial state {initial_state_label} using Qrack...")
print("!!! WARNING: Noise application (t1_kraus, t2_kraus) is COMMENTED OUT due to missing methods in pyqrack !!!")
print("!!!          Fidelity will likely remain constant. Update pyqrack recommended.           !!!")
# print(f"Default T1={t1_default_s:.3e} s, T2={t2_default_s:.3e} s") # Commented out as not used
# print(f"QHRF    T1={t1_qhrf_s:.3e} s, T2={t2_qhrf_s:.3e} s") # Commented out as not used
print(f"Time step = {time_step_s:.3e} s ({num_steps} steps)")

start_time = time.time()

# --- Simulation with Default Noise ---
print("\nRunning Default 'Simulation' (Noise Disabled)...")
qsim_default = QrackSimulator(1) # Use 1 qubit

# Prepare initial state
if initial_state_choice == '1':
    qsim_default.x(0)
elif initial_state_choice == '+':
    qsim_default.h(0)

# Record fidelity at each time step
for i in range(num_steps + 1):
    current_time_ns = time_points_ns[i]

    # Get statevector using out_ket and calculate fidelity
    try:
        # *** Using out_ket() based on previous test ***
        sv_default_amps = qsim_default.out_ket()
        # calculate_fidelity now handles conversion from list to numpy array
        fidelity = calculate_fidelity(ideal_state_np, sv_default_amps)
        fidelities_default.append(fidelity)
    except Exception as e:
        print(f"\nAn error occurred during out_ket or fidelity calculation: {e}")
        # Attempt to clean up before exiting
        del qsim_default
        exit()


    if i % (num_steps // 10 + 1) == 0:
         # Displaying fidelity even though it's expected to be constant
         print(f"  Default Time: {current_time_ns:.0f} ns ({i}/{num_steps}) - Fidelity: {fidelity:.4f}")

    # If not the last step, apply noise for the *next* time interval (time_step_s)
    if i < num_steps:
        # !!! NOISE APPLICATION COMMENTED OUT !!!
        # qsim_default.t1_kraus(0, time_step_s, t1_default_s)
        # qsim_default.t2_kraus(0, time_step_s, t2_default_s)
        pass # No operation for noise step

# --- Simulation with QHRF-Enhanced Noise ---
print("\nRunning QHRF 'Simulation' (Noise Disabled)...")
qsim_qhrf = QrackSimulator(1) # Use 1 qubit

# Prepare initial state
if initial_state_choice == '1':
    qsim_qhrf.x(0)
elif initial_state_choice == '+':
    qsim_qhrf.h(0)

# Record fidelity at each time step
for i in range(num_steps + 1):
    current_time_ns = time_points_ns[i]

    # Get statevector using out_ket and calculate fidelity
    try:
        # *** Using out_ket() based on previous test ***
        sv_qhrf_amps = qsim_qhrf.out_ket()
        fidelity = calculate_fidelity(ideal_state_np, sv_qhrf_amps)
        fidelities_qhrf.append(fidelity)
    except Exception as e:
        print(f"\nAn error occurred during out_ket or fidelity calculation: {e}")
        # Attempt to clean up before exiting
        del qsim_qhrf
        if 'qsim_default' in locals(): del qsim_default # Clean up both if needed
        exit()


    if i % (num_steps // 10 + 1) == 0:
         print(f"  QHRF Time: {current_time_ns:.0f} ns ({i}/{num_steps}) - Fidelity: {fidelity:.4f}")

    # If not the last step, apply noise for the next time interval
    if i < num_steps:
        # !!! NOISE APPLICATION COMMENTED OUT !!!
        # qsim_qhrf.t1_kraus(0, time_step_s, t1_qhrf_s)
        # qsim_qhrf.t2_kraus(0, time_step_s, t2_qhrf_s)
        pass # No operation for noise step

end_time = time.time()
print(f"\nSimulation finished in {end_time - start_time:.2f} seconds.")

# --- Plot Results ---
plt.figure(figsize=(10, 6))
# Plot simulation results (convert time back to microseconds for plotting)
# Expect fidelity to be ~constant since noise is off
plt.plot(time_points_ns / 1000, fidelities_default, 'o-', label=f'Default Params (NO NOISE) - Qrack Sim')
plt.plot(time_points_ns / 1000, fidelities_qhrf, 's-', label=f'QHRF Params (NO NOISE) - Qrack Sim')

# Add theoretical decay curves for comparison (using times in seconds now)
# These won't match the simulation plot as noise is off in simulation
theory_time_s = time_points_s
if initial_state_choice == '1':
    plt.plot(time_points_ns / 1000, np.exp(-theory_time_s / t1_default_s), '--', color='blue', alpha=0.3, label='Theory: T1 Decay (Default)')
    plt.plot(time_points_ns / 1000, np.exp(-theory_time_s / t1_qhrf_s), '--', color='orange', alpha=0.3, label='Theory: T1 Decay (QHRF)')
elif initial_state_choice == '+':
    plt.plot(time_points_ns / 1000, np.exp(-theory_time_s / t2_default_s), '--', color='blue', alpha=0.3, label='Theory: Approx T2 Decay (Default)')
    plt.plot(time_points_ns / 1000, np.exp(-theory_time_s / t2_qhrf_s), '--', color='orange', alpha=0.3, label='Theory: Approx T2 Decay (QHRF)')

plt.xlabel('Time (microseconds)')
plt.ylabel(f'Fidelity with Ideal State ({initial_state_label})')
plt.title('Qrack "Simulation" (NOISE DISABLED due to missing methods)')
plt.legend()
plt.grid(True)
plt.ylim(-0.05, 1.05)
plt.show()

# Clean up simulator objects (good practice)
print("Cleaning up QrackSimulator objects...")
del qsim_default
del qsim_qhrf
print("Cleanup complete.")
