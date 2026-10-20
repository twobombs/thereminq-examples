# @title Qiskit Simulation of QHRF Effect (Enhanced Coherence) - Fixed Transpiler Error
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile # Keep transpile import in case needed elsewhere, but not using it here
from qiskit_aer import AerSimulator
# Removed import for FakeManilaV2 as it might be deprecated/moved
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from qiskit.quantum_info import state_fidelity, Statevector
import warnings

# Suppress specific warnings if needed, e.g., about statevector method
warnings.filterwarnings('ignore', category=UserWarning, message='Simulation method.*')
# Suppress the specific UserWarning about backend/basis_gates incompatibility if it persists implicitly
warnings.filterwarnings('ignore', category=UserWarning, message='Providing `coupling_map` and/or `basis_gates`.*')


# --- Simulation Parameters ---
# Define plausible default T1/T2/gate times manually (in nanoseconds)
# These are just example values, adjust as needed.
t1_default_ns = 50000.0  # 50 microseconds
t2_default_ns = 30000.0  # 30 microseconds
gate_time_ns = 50.0      # 50 nanoseconds for a single-qubit gate

print(f"Using manually defined defaults: T1={t1_default_ns} ns, T2={t2_default_ns} ns, Gate Time={gate_time_ns} ns")


# Define improved T1/T2 times, simulating the QHRF effect (e.g., 5x improvement)
qhrf_improvement_factor = 5.0
t1_qhrf_ns = t1_default_ns * qhrf_improvement_factor
t2_qhrf_ns = t2_default_ns * qhrf_improvement_factor

# Ensure T2 <= 2*T1
t2_default_ns = min(t2_default_ns, 2 * t1_default_ns)
t2_qhrf_ns = min(t2_qhrf_ns, 2 * t1_qhrf_ns)

# Simulation time steps
max_time_ns = t1_default_ns * 1.5 # Simulate for ~1.5x the default T1
time_step_ns = gate_time_ns * 10 # Simulate in steps of 10 gate times
time_points_ns = np.arange(0, max_time_ns, time_step_ns)
num_steps = len(time_points_ns)

# Choose initial state: '1' or '+'
initial_state_choice = '+' # Options: '1', '+'

# --- Helper Function to Create Noise Model ---
def create_noise_model(t1_ns, t2_ns, gate_time_ns):
    """Creates a Qiskit NoiseModel with T1/T2 thermal relaxation."""
    noise_model = NoiseModel()
    # Add thermal relaxation error to idle qubits (during delay)
    # We model delay using identity gates, assuming they take one gate_time_ns
    thermal_error = thermal_relaxation_error(t1_ns, t2_ns, gate_time_ns)
    # Apply error to 'id' gate which represents the delay periods
    noise_model.add_quantum_error(thermal_error, 'id', [0])
    # Apply errors to other gates if they were used in state prep
    noise_model.add_quantum_error(thermal_error.copy(), 'x', [0]) # Need copy if applying same error type
    noise_model.add_quantum_error(thermal_error.copy(), 'h', [0])
    return noise_model

# --- Create Noise Models ---
noise_model_default = create_noise_model(t1_default_ns, t2_default_ns, gate_time_ns)
noise_model_qhrf = create_noise_model(t1_qhrf_ns, t2_qhrf_ns, gate_time_ns)

# --- Setup Simulator ---
# Using statevector simulator to get exact state after noise application
simulator = AerSimulator(method='statevector')

# --- Prepare Ideal State ---
ideal_circuit = QuantumCircuit(1, 1) # Keep classical bit here for consistency if needed later, though not strictly necessary for save_statevector
if initial_state_choice == '1':
    ideal_circuit.x(0)
    initial_state_label = '|1>'
    ideal_state = Statevector([0, 1])
elif initial_state_choice == '+':
    ideal_circuit.h(0)
    initial_state_label = '|+>'
    ideal_state = Statevector([1/np.sqrt(2), 1/np.sqrt(2)])
else:
    raise ValueError("initial_state_choice must be '1' or '+'")

# --- Run Simulations ---
fidelities_default = []
fidelities_qhrf = []

print(f"Simulating T1/T2 decay for initial state {initial_state_label}...")
print(f"Default T1={t1_default_ns:.2f} ns, T2={t2_default_ns:.2f} ns")
print(f"QHRF T1={t1_qhrf_ns:.2f} ns, T2={t2_qhrf_ns:.2f} ns")
print(f"Gate time (for delay step) = {gate_time_ns:.2f} ns")

for i in range(num_steps):
    current_time_ns = time_points_ns[i]
    # Ensure non-negative time and handle potential floating point inaccuracies
    if current_time_ns < 0: current_time_ns = 0
    num_delay_gates = int(round(current_time_ns / gate_time_ns)) # Number of identity gates to represent delay

    # Create circuit with initial state prep + delay
    qc = QuantumCircuit(1, name=f"qc_delay_{num_delay_gates}") # Classical bit removed
    if initial_state_choice == '1':
        qc.x(0)
    elif initial_state_choice == '+':
        qc.h(0)

    # Add identity gates to simulate time evolution (delay)
    for _ in range(num_delay_gates):
        qc.id(0) # Apply identity gate to qubit 0

    # Add save_statevector instruction
    qc.save_statevector(label='final_statevector')

    # --- Simulation with Default Noise ---
    # *** Run the original circuit directly, DO NOT transpile ***
    result_default = simulator.run(qc, noise_model=noise_model_default).result()
    statevector_default = result_default.data(0)['final_statevector']
    fidelity_default = state_fidelity(ideal_state, statevector_default)
    fidelities_default.append(fidelity_default)

    # --- Simulation with QHRF-Enhanced Noise ---
    # *** Run the original circuit directly, DO NOT transpile ***
    result_qhrf = simulator.run(qc, noise_model=noise_model_qhrf).result()
    statevector_qhrf = result_qhrf.data(0)['final_statevector']
    fidelity_qhrf = state_fidelity(ideal_state, statevector_qhrf)
    fidelities_qhrf.append(fidelity_qhrf)

    if i % (num_steps // 10 + 1) == 0: # Print progress
       print(f"  Time: {current_time_ns:.0f} ns ({i}/{num_steps}) - Fidelity (Default): {fidelity_default:.4f}, Fidelity (QHRF): {fidelity_qhrf:.4f}")


print("Simulation finished.")

# --- Plot Results ---
plt.figure(figsize=(10, 6))
plt.plot(time_points_ns / 1000, fidelities_default, 'o-', label=f'Default Noise (T1={t1_default_ns/1000:.1f}us, T2={t2_default_ns/1000:.1f}us)')
plt.plot(time_points_ns / 1000, fidelities_qhrf, 's-', label=f'QHRF-Enhanced Noise (T1={t1_qhrf_ns/1000:.1f}us, T2={t2_qhrf_ns/1000:.1f}us)')

# Add theoretical decay curves for comparison (optional, depends on state)
theory_time_us = time_points_ns / 1000
if initial_state_choice == '1':
    # T1 decay for |1> state: Fidelity = P(1) = exp(-t/T1)
    plt.plot(theory_time_us, np.exp(-time_points_ns / t1_default_ns), '--', color='blue', alpha=0.7, label='Theory: T1 Decay (Default)')
    plt.plot(theory_time_us, np.exp(-time_points_ns / t1_qhrf_ns), '--', color='orange', alpha=0.7, label='Theory: T1 Decay (QHRF)')
elif initial_state_choice == '+':
    # T2 decay for |+> state: Fidelity approx exp(-t/T2) (simplified view)
    plt.plot(theory_time_us, np.exp(-time_points_ns / t2_default_ns), '--', color='blue', alpha=0.7, label='Theory: T2 Decay (Default)')
    plt.plot(theory_time_us, np.exp(-time_points_ns / t2_qhrf_ns), '--', color='orange', alpha=0.7, label='Theory: T2 Decay (QHRF)')


plt.xlabel('Time (microseconds)')
plt.ylabel(f'Fidelity with Ideal State ({initial_state_label})')
plt.title('Simulated Qubit Coherence Enhancement (QHRF Effect)')
plt.legend()
plt.grid(True)
plt.ylim(0, 1.05)
plt.show()
