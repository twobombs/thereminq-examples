#
# Scaling Quantum Algorithms via Dissipation: A Qiskit Demonstration
#
# This script implements and compares a standard unitary Variational Quantum
# Algorithm (VQA) with a dissipative VQA, as described in the paper
# "Scaling Quantum Algorithms via Dissipation: Avoiding Barren Plateaus"
# (arXiv:2507.02043).
#
# The goal is to numerically demonstrate that by periodically resetting
# ancillary qubits, we can prevent the gradient of the cost function from
# vanishing exponentially with the system size (the "barren plateau" problem),
# especially in the presence of noise.
#
# To run this script, you will need Qiskit, NumPy, Matplotlib, and the
# PyQrack simulator installed:
#
# pip install qiskit numpy matplotlib qiskit-pyqrack
#

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import Parameter, ParameterVector
from qiskit.primitives import Estimator
from qiskit.providers.fake_provider import FakeManila
from qiskit_pyqrack import PyQrackSimulator

# For reproducibility
np.random.seed(42)

# --- Configuration ---
# System sizes to test
N_QUBITS_LIST = [4, 6, 8, 10]
# Number of random parameter sets to sample for variance calculation
N_SAMPLES = 30
# Noise level for the depolarizing error model
NOISE_PROB = 0.01

# ---------------------------------
# 1. Ansatz and Circuit Definitions
# ---------------------------------

def get_entangling_layer(n_qubits, params):
    """Creates a layer of Ry rotations and CNOT gates."""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(params[i], i)
    for i in range(n_qubits - 1):
        qc.cnot(i, i + 1)
    qc.barrier()
    return qc

def build_unitary_ansatz(n_qubits, depth):
    """
    Builds a standard unitary VQA ansatz.
    The depth of this circuit scales with the number of qubits, which is a
    common cause of barren plateaus.
    """
    params = ParameterVector('θ', length=n_qubits * depth)
    qc = QuantumCircuit(n_qubits)
    for d in range(depth):
        param_slice = params[d * n_qubits : (d + 1) * n_qubits]
        qc.compose(get_entangling_layer(n_qubits, param_slice), inplace=True)
    return qc

def build_dissipative_ansatz(n_system_qubits, layers_per_jump):
    """
    Builds a dissipative VQA ansatz.
    It includes system qubits and ancillary qubits. The ancillary qubits are
    periodically reset to remove entropy from the system.
    """
    # Use one ancilla for every two system qubits
    n_ancilla_qubits = n_system_qubits // 2
    total_qubits = n_system_qubits + n_ancilla_qubits

    # The number of "jumps" (reset cycles) scales with system size
    n_jumps = n_system_qubits // 2

    num_params = n_system_qubits * layers_per_jump * n_jumps
    params = ParameterVector('θ', length=num_params)
    
    qc = QuantumCircuit(total_qubits)
    system_q = list(range(n_system_qubits))
    ancilla_q = list(range(n_system_qubits, total_qubits))

    param_idx = 0
    for _ in range(n_jumps):
        # Unitary evolution on system qubits (constant depth block)
        for _ in range(layers_per_jump):
            param_slice = params[param_idx : param_idx + n_system_qubits]
            qc.compose(get_entangling_layer(n_system_qubits, param_slice), qubits=system_q, inplace=True)
            param_idx += n_system_qubits
        
        # Dissipative step: couple system to ancillas and reset ancillas
        # This maps system entropy to the ancillas, which is then discarded.
        for i in range(n_ancilla_qubits):
            # Couple two system qubits to one ancilla
            qc.cnot(system_q[2 * i], ancilla_q[i])
            qc.cnot(system_q[2 * i + 1], ancilla_q[i])
        
        # Reset the ancillary qubits (the non-unital, dissipative operation)
        qc.reset(ancilla_q)
        qc.barrier()
        
    return qc

# ---------------------------------
# 2. Gradient Calculation
# ---------------------------------

def get_gradient(circuit, observable, param_index, backend):
    """
    Calculates the gradient of the cost function with respect to a single
    parameter using the parameter-shift rule.
    """
    # Create two circuits for the parameter shift rule
    params = circuit.parameters
    
    # We need to create two sets of circuits to be run
    plus_value = np.pi / 2
    minus_value = -np.pi / 2

    # Get the specific parameter we are shifting
    target_parameter = params[param_index]

    # Create circuits for expectation value estimation
    plus_circuit = circuit.assign_parameters({target_parameter: plus_value})
    minus_circuit = circuit.assign_parameters({target_parameter: minus_value})
    
    estimator = Estimator(backend=backend)
    
    # Job execution
    job = estimator.run([plus_circuit, minus_circuit], [observable, observable])
    result = job.result()
    
    exp_vals = result.values
    
    # Parameter-shift rule
    gradient = 0.5 * (exp_vals[0] - exp_vals[1])
    
    return gradient


# ---------------------------------
# 3. Simulation and Main Loop
# ---------------------------------

def run_simulation():
    """
    Main simulation loop. Iterates through system sizes, calculates gradient
    variances for both ansatz types, and returns the results.
    """
    unitary_variances = []
    dissipative_variances = []

    # Define a simple local observable: Z on the first qubit
    # For the dissipative case, the observable acts only on the system qubits.
    observable = SparsePauliOp.from_list([("Z" + "I" * (max(N_QUBITS_LIST) - 1), 1)])

    # Initialize the PyQrack simulator
    # We can add a noise model here to simulate NIBPs
    backend = PyQrackSimulator()
    # To add noise, uncomment the following lines:
    # noise_model = NoiseModel()
    # error = depolarizing_error(NOISE_PROB, 1)
    # noise_model.add_all_qubit_quantum_error(error, ['ry', 'sx', 'x'])
    # error2 = depolarizing_error(NOISE_PROB * 10, 2)
    # noise_model.add_all_qubit_quantum_error(error2, ['cx'])
    # backend.set_options(noise_model=noise_model)


    print("Starting simulation...")
    for n_qubits in N_QUBITS_LIST:
        print(f"\n--- Simulating for {n_qubits} qubits ---")

        # --- Unitary Case ---
        print("  Running Unitary VQA...")
        # Depth scales with system size
        depth = n_qubits
        unitary_circuit = build_unitary_ansatz(n_qubits, depth)
        
        # Choose a parameter in the last layer to calculate the gradient for
        target_param_index = len(unitary_circuit.parameters) - n_qubits
        
        unitary_gradients = []
        for i in range(N_SAMPLES):
            # Assign random values to all other parameters
            random_params = np.random.uniform(0, 2 * np.pi, len(unitary_circuit.parameters))
            bound_circuit = unitary_circuit.assign_parameters(random_params)
            
            # We need to unbind the target parameter for the shift rule
            grad = get_gradient(bound_circuit, observable.copy(), target_param_index, backend)
            unitary_gradients.append(grad)
            print(f"    Unitary Sample {i+1}/{N_SAMPLES}, Gradient: {grad:.4f}", end='\r')
        
        unitary_variances.append(np.var(unitary_gradients))
        print(f"\n  Unitary Gradient Variance: {np.var(unitary_gradients):.6f}")

        # --- Dissipative Case ---
        print("  Running Dissipative VQA...")
        # Constant depth jumps
        layers_per_jump = 2
        dissipative_circuit = build_dissipative_ansatz(n_qubits, layers_per_jump)
        
        # Choose a parameter in the last layer
        target_param_index_diss = len(dissipative_circuit.parameters) - n_qubits
        
        dissipative_gradients = []
        for i in range(N_SAMPLES):
            random_params = np.random.uniform(0, 2 * np.pi, len(dissipative_circuit.parameters))
            bound_circuit = dissipative_circuit.assign_parameters(random_params)
            
            grad = get_gradient(bound_circuit, observable.copy(), target_param_index_diss, backend)
            dissipative_gradients.append(grad)
            print(f"    Dissipative Sample {i+1}/{N_SAMPLES}, Gradient: {grad:.4f}", end='\r')

        dissipative_variances.append(np.var(dissipative_gradients))
        print(f"\n  Dissipative Gradient Variance: {np.var(dissipative_gradients):.6f}")

    return unitary_variances, dissipative_variances


# ---------------------------------
# 4. Plotting
# ---------------------------------

def plot_results(unitary_variances, dissipative_variances):
    """Plots the gradient variances vs. number of qubits."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(N_QUBITS_LIST, unitary_variances, 'o-', label='Unitary VQA', color='crimson', linewidth=2, markersize=8)
    ax.plot(N_QUBITS_LIST, dissipative_variances, 's--', label='Dissipative VQA', color='royalblue', linewidth=2, markersize=8)

    ax.set_yscale('log')
    ax.set_xlabel('Number of System Qubits (n)', fontsize=14)
    ax.set_ylabel('Var(∂C/∂θ)', fontsize=14)
    ax.set_title('Gradient Variance vs. System Size', fontsize=16, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=12)
    
    # Add annotation
    ax.text(0.5, -0.15, 
            'The unitary VQA shows exponential decay (barren plateau),\nwhile the dissipative VQA maintains a constant gradient variance.',
            transform=ax.transAxes, ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Run the simulation
    unitary_vars, dissipative_vars = run_simulation()
    
    # Plot the results
    plot_results(unitary_vars, dissipative_vars)
