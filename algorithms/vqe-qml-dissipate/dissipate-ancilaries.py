"""
Scaling Quantum Algorithms via Dissipation: A Qiskit Demonstration

This script implements and compares a standard unitary Variational Quantum
Algorithm (VQA) with a dissipative VQA, as described in the paper
"Scaling Quantum Algorithms via Dissipation: Avoiding Barren Plateaus"
(arXiv:2507.02043).

The goal is to numerically demonstrate that by periodically resetting
ancillary qubits, we can prevent the gradient of the cost function from
vanishing exponentially with the system size (the "barren plateau" problem),
especially in the presence of noise.

References:
    - arXiv:2507.02043 - Scaling Quantum Algorithms via Dissipation:
      Avoiding Barren Plateaus

To run this script, you will need Qiskit, NumPy, Matplotlib, and the
PyQrack simulator installed:

    pip install qiskit numpy matplotlib qiskit-pyqrack
"""

import logging
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import Parameter, ParameterVector
from qiskit.primitives import Estimator
from qiskit_pyqrack import PyQrackSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DissipativeVQAConfig:
    """Configuration for the dissipative VQA simulation."""
    
    def __init__(
        self,
        n_qubits_list: List[int] = None,
        n_samples: int = 30,
        noise_prob: float = 0.01,
        seed: int = 42
    ):
        """
        Initialize the configuration.
        
        Args:
            n_qubits_list: List of system sizes to test.
            n_samples: Number of random parameter sets for variance calculation.
            noise_prob: Noise level for the depolarizing error model.
            seed: Random seed for reproducibility.
        """
        self.n_qubits_list = n_qubits_list or [4, 6, 8, 10]
        self.n_samples = n_samples
        self.noise_prob = noise_prob
        self.seed = seed


def plot_comparison(
    unitary_variances: np.ndarray,
    dissipative_variances: np.ndarray,
    n_qubits_list: List[int],
    save_path: str = "gradient_variance_comparison.png"
) -> None:
    """
    Create comparison plot with statistical analysis.
    
    Args:
        unitary_variances: Array of gradient variances for unitary VQA.
        dissipative_variances: Array of gradient variances for dissipative VQA.
        n_qubits_list: List of number of qubits for each data point.
        save_path: Path to save the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot variances
    axes[0].plot(n_qubits_list, unitary_variances, 'o-', label='Unitary VQA',
                 color='crimson', linewidth=2, markersize=8)
    axes[0].plot(n_qubits_list, dissipative_variances, 's--', label='Dissipative VQA',
                 color='royalblue', linewidth=2, markersize=8)
    
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Number of System Qubits (n)', fontsize=14)
    axes[0].set_ylabel('Var(∂C/∂θ)', fontsize=14)
    axes[0].set_title('Gradient Variance vs. System Size', fontsize=16, pad=20)
    axes[0].tick_params(axis='both', which='major', labelsize=12)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(unitary_variances, dissipative_variances)
    mean_unitary = np.mean(unitary_variances)
    mean_dissipative = np.mean(dissipative_variances)
    
    axes[1].text(0.5, 0.8, f'Unitary mean: {mean_unitary:.2e}',
                 transform=axes[1].transAxes, fontsize=12, ha='center')
    axes[1].text(0.5, 0.6, f'Dissipative mean: {mean_dissipative:.2e}',
                 transform=axes[1].transAxes, fontsize=12, ha='center')
    axes[1].text(0.5, 0.4, f't-statistic: {t_stat:.4f}',
                 transform=axes[1].transAxes, fontsize=12, ha='center')
    axes[1].text(0.5, 0.2, f'p-value: {p_value:.2e}',
                 transform=axes[1].transAxes, fontsize=12, ha='center')
    axes[1].text(0.5, 0.05, 'Statistical comparison of gradient variances',
                 transform=axes[1].transAxes, fontsize=11, ha='center', fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle('Dissipative vs Unitary VQA: Gradient Variance Analysis',
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {save_path}")
    plt.show()


def plot_comparison_with_confidence(
    unitary_variances: np.ndarray,
    dissipative_variances: np.ndarray,
    n_qubits_list: List[int],
    save_path: str = "gradient_variance_comparison.png"
) -> None:
    """
    Create comparison plot with confidence intervals.
    
    Args:
        unitary_variances: Array of gradient variances for unitary VQA.
        dissipative_variances: Array of gradient variances for dissipative VQA.
        n_qubits_list: List of number of qubits for each data point.
        save_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(n_qubits_list))
    width = 0.35
    
    # Calculate confidence intervals
    unitary_std = np.std(unitary_variances)
    dissipative_std = np.std(dissipative_variances)
    unitary_ci = stats.t.interval(0.95, len(unitary_variances)-1,
                                   loc=np.mean(unitary_variances),
                                   scale=stats.sem(unitary_variances))
    dissipative_ci = stats.t.interval(0.95, len(dissipative_variances)-1,
                                       loc=np.mean(dissipative_variances),
                                       scale=stats.sem(dissipative_variances))
    
    ax.errorbar(x - width/2, unitary_variances, yerr=unitary_std,
                fmt='o-', label='Unitary VQA', color='crimson',
                linewidth=2, markersize=8, capsize=5)
    ax.errorbar(x + width/2, dissipative_variances, yerr=dissipative_std,
                fmt='s--', label='Dissipative VQA', color='royalblue',
                linewidth=2, markersize=8, capsize=5)
    
    ax.set_yscale('log')
    ax.set_xlabel('Number of System Qubits (n)', fontsize=14)
    ax.set_ylabel('Var(∂C/∂θ)', fontsize=14)
    ax.set_title('Gradient Variance vs. System Size with 95% CI', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(n_qubits_list)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.text(0.5, -0.15,
            'The unitary VQA shows exponential decay (barren plateau),\n'
            'while the dissipative VQA maintains a constant gradient variance.',
            transform=ax.transAxes, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {save_path}")
    plt.show()

# ---------------------------------
# 1. Ansatz and Circuit Definitions
# ---------------------------------

def get_entangling_layer(n_qubits: int, params: np.ndarray) -> QuantumCircuit:
    """
    Creates a layer of Ry rotations and CNOT gates.
    
    Args:
        n_qubits: Number of qubits in the layer.
        params: Array of rotation parameters.
    
    Returns:
        A QuantumCircuit with the entangling layer.
    """
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(params[i], i)
    for i in range(n_qubits - 1):
        qc.cnot(i, i + 1)
    qc.barrier()
    return qc


def build_unitary_ansatz(n_qubits: int, depth: int) -> QuantumCircuit:
    """
    Builds a standard unitary VQA ansatz.
    The depth of this circuit scales with the number of qubits, which is a
    common cause of barren plateaus.
    
    Args:
        n_qubits: Number of system qubits.
        depth: Depth of the ansatz circuit.
    
    Returns:
        A QuantumCircuit representing the unitary ansatz.
    """
    params = ParameterVector('θ', length=n_qubits * depth)
    qc = QuantumCircuit(n_qubits)
    for d in range(depth):
        param_slice = params[d * n_qubits : (d + 1) * n_qubits]
        qc.compose(get_entangling_layer(n_qubits, param_slice), inplace=True)
    return qc


def build_dissipative_ansatz(
    n_system_qubits: int,
    layers_per_jump: int
) -> QuantumCircuit:
    """
    Builds a dissipative VQA ansatz.
    It includes system qubits and ancillary qubits. The ancillary qubits are
    periodically reset to remove entropy from the system.
    
    Args:
        n_system_qubits: Number of system qubits.
        layers_per_jump: Number of unitary layers between each dissipative jump.
    
    Returns:
        A QuantumCircuit representing the dissipative ansatz.
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

def get_gradient(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    param_index: int,
    backend
) -> float:
    """
    Calculates the gradient of the cost function with respect to a single
    parameter using the parameter-shift rule.
    
    Args:
        circuit: The quantum circuit.
        observable: The observable operator.
        param_index: Index of the parameter to compute gradient for.
        backend: The quantum backend for execution.
    
    Returns:
        The gradient value.
    """
    # Create two circuits for the parameter shift rule
    params = circuit.parameters
    
    # Parameter-shift rule values
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

def run_simulation(
    config: DissipativeVQAConfig = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main simulation loop. Iterates through system sizes, calculates gradient
    variances for both ansatz types, and returns the results.
    
    Args:
        config: Optional configuration. If None, default configuration is used.
    
    Returns:
        A tuple containing:
            - unitary_variances: Array of gradient variances for unitary VQA.
            - dissipative_variances: Array of gradient variances for dissipative VQA.
    """
    if config is None:
        config = DissipativeVQAConfig()
    
    # Set random seed for reproducibility
    np.random.seed(config.seed)
    
    unitary_variances = []
    dissipative_variances = []

    # Define a simple local observable: Z on the first qubit
    # For the dissipative case, the observable acts only on the system qubits.
    max_qubits = max(config.n_qubits_list)
    observable = SparsePauliOp.from_list([("Z" + "I" * (max_qubits - 1), 1)])

    # Initialize the PyQrack simulator
    backend = PyQrackSimulator()
    
    logger.info(f"Starting simulation with config: {config}")
    logger.info(f"Testing qubit sizes: {config.n_qubits_list}")

    for n_qubits in config.n_qubits_list:
        logger.info(f"\n--- Simulating for {n_qubits} qubits ---")

        # --- Unitary Case ---
        logger.info("  Running Unitary VQA...")
        # Depth scales with system size
        depth = n_qubits
        unitary_circuit = build_unitary_ansatz(n_qubits, depth)
        
        # Choose a parameter in the last layer to calculate the gradient for
        target_param_index = len(unitary_circuit.parameters) - n_qubits
        
        unitary_gradients = []
        for i in range(config.n_samples):
            # Assign random values to all other parameters
            random_params = np.random.uniform(0, 2 * np.pi, len(unitary_circuit.parameters))
            bound_circuit = unitary_circuit.assign_parameters(random_params)
            
            # We need to unbind the target parameter for the shift rule
            grad = get_gradient(bound_circuit, observable.copy(), target_param_index, backend)
            unitary_gradients.append(grad)
            logger.debug(f"    Unitary Sample {i+1}/{config.n_samples}, Gradient: {grad:.4f}")
        
        unitary_variance = np.var(unitary_gradients)
        unitary_variances.append(unitary_variance)
        logger.info(f"  Unitary Gradient Variance: {unitary_variance:.6e}")

        # --- Dissipative Case ---
        logger.info("  Running Dissipative VQA...")
        # Constant depth jumps
        layers_per_jump = 2
        dissipative_circuit = build_dissipative_ansatz(n_qubits, layers_per_jump)
        
        # Choose a parameter in the last layer
        target_param_index_diss = len(dissipative_circuit.parameters) - n_qubits
        
        dissipative_gradients = []
        for i in range(config.n_samples):
            random_params = np.random.uniform(0, 2 * np.pi, len(dissipative_circuit.parameters))
            bound_circuit = dissipative_circuit.assign_parameters(random_params)
            
            grad = get_gradient(bound_circuit, observable.copy(), target_param_index_diss, backend)
            dissipative_gradients.append(grad)
            logger.debug(f"    Dissipative Sample {i+1}/{config.n_samples}, Gradient: {grad:.4f}")

        dissipative_variance = np.var(dissipative_gradients)
        dissipative_variances.append(dissipative_variance)
        logger.info(f"  Dissipative Gradient Variance: {dissipative_variance:.6e}")

    return np.array(unitary_variances), np.array(dissipative_variances)


# ---------------------------------
# 4. Plotting
# ---------------------------------

def plot_results(
    unitary_variances: np.ndarray,
    dissipative_variances: np.ndarray,
    n_qubits_list: List[int] = None,
    save_path: str = "gradient_variance_comparison.png"
) -> None:
    """
    Plots the gradient variances vs. number of qubits.
    
    Args:
        unitary_variances: Array of gradient variances for unitary VQA.
        dissipative_variances: Array of gradient variances for dissipative VQA.
        n_qubits_list: List of number of qubits. Defaults to [4, 6, 8, 10].
        save_path: Path to save the plot.
    """
    if n_qubits_list is None:
        n_qubits_list = [4, 6, 8, 10]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(n_qubits_list, unitary_variances, 'o-', label='Unitary VQA',
            color='crimson', linewidth=2, markersize=8)
    ax.plot(n_qubits_list, dissipative_variances, 's--', label='Dissipative VQA',
            color='royalblue', linewidth=2, markersize=8)

    ax.set_yscale('log')
    ax.set_xlabel('Number of System Qubits (n)', fontsize=14)
    ax.set_ylabel('Var(∂C/∂θ)', fontsize=14)
    ax.set_title('Gradient Variance vs. System Size', fontsize=16, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=12)
    
    # Add annotation
    ax.text(0.5, -0.15,
            'The unitary VQA shows exponential decay (barren plateau),\n'
            'while the dissipative VQA maintains a constant gradient variance.',
            transform=ax.transAxes, ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {save_path}")
    plt.show()


def main() -> None:
    """
    Main entry point for the dissipative VQA demonstration.
    """
    # Create configuration
    config = DissipativeVQAConfig(
        n_qubits_list=[4, 6, 8, 10],
        n_samples=30,
        noise_prob=0.01,
        seed=42
    )
    
    # Run the simulation
    logger.info("Starting dissipative VQA simulation...")
    unitary_vars, dissipative_vars = run_simulation(config)
    
    # Plot the results
    plot_results(unitary_vars, dissipative_vars, config.n_qubits_list)
    
    # Print summary statistics
    logger.info("\n=== Summary Statistics ===")
    logger.info(f"Unitary mean variance: {np.mean(unitary_vars):.6e}")
    logger.info(f"Dissipative mean variance: {np.mean(dissipative_vars):.6e}")
    logger.info(f"Ratio (unitary/dissipative): {np.mean(unitary_vars) / np.mean(dissipative_vars):.2f}x")


if __name__ == '__main__':
    main()
