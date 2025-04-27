# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.linalg import expm

# Qiskit core imports
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import RealAmplitudes

# --- Attempting different primitive import paths due to environment issues ---
try:
    # Standard path for modern Qiskit
    from qiskit.primitives import Sampler, Estimator
    print("Using Sampler/Estimator from qiskit.primitives")
except ImportError:
    try:
        # Path for qiskit-aer primitives
        from qiskit_aer.primitives import Sampler, Estimator
        print("Using Sampler/Estimator from qiskit_aer.primitives")
    except ImportError:
        try:
            # Legacy path (less likely to work with modern algorithms)
            from qiskit.providers.aer.primitives import Sampler, Estimator
            print("Using Sampler/Estimator from qiskit.providers.aer.primitives (legacy)")
        except ImportError:
            print("ERROR: Could not find Sampler and Estimator. Please check your Qiskit installation and environment.")
            # Exit or raise error if primitives cannot be imported
            raise ImportError("Failed to import Sampler and Estimator from known Qiskit paths.")

from qiskit_aer import AerSimulator # Use Aer simulator

# Qiskit Algorithms imports
# Ensure you have run: pip install -U qiskit qiskit-aer qiskit_algorithms
from qiskit_algorithms.linear_solvers.numpy_linear_solver import NumPyLinearSolver
from qiskit_algorithms.linear_solvers.vqls import VQLS
from qiskit_algorithms.optimizers import SPSA


# --- 1. Problem Definition (Simulated FEM-like System) ---
# Define the size of the system (must be a power of 2 for simple qubit mapping)
N = 8 # Dimension of the matrix (requires log2(N) = 3 qubits)
num_qubits = int(np.log2(N))

# Create a representative sparse, Hermitian matrix A
# This is a simple banded matrix, inspired by FEM discretizations
# but not derived from a specific physics problem.
diagonals = [
    np.ones(N),
    -0.5 * np.ones(N - 1),
    -0.5 * np.ones(N - 1),
    0.1 * np.ones(N - 2),
    0.1 * np.ones(N - 2)
]
A_matrix = diags(diagonals, [0, 1, -1, 2, -2]).toarray()
# Ensure Hermiticity (A = A_dagger) - already symmetric here
# A_matrix = 0.5 * (A_matrix + A_matrix.conj().T) # General way

# Create a representative right-hand side vector b (simulating incident field)
b_vector = np.zeros(N, dtype=complex) # Ensure complex type
b_vector[0] = 1.0
b_vector[1] = 0.5j # Example complex value
b_vector = b_vector / np.linalg.norm(b_vector) # Normalize b

print("--- System Definition ---")
print(f"Matrix A (Dimension: {N}x{N}):")
# print(np.round(A_matrix, 3)) # Optionally print the matrix
print(f"\nVector b (Normalized):")
print(np.round(b_vector, 3))

# --- 2. Matrix Visualization ---
plt.figure(figsize=(6, 6))
plt.spy(A_matrix, markersize=5)
plt.title(f"Sparsity Pattern of Representative Matrix A ({N}x{N})")
plt.xlabel("Column Index")
plt.ylabel("Row Index")
plt.grid(True)
plt.show()

# --- 3. Classical Solution (for comparison) ---
# Use the updated import path
classical_solver = NumPyLinearSolver()
classical_solution = classical_solver.solve(A_matrix, b_vector)
# The solution state might be complex, handle accordingly
x_classical = classical_solution.state / np.linalg.norm(classical_solution.state) # Normalize
print("\n--- Classical Solution ---")
print("Classical solution vector x (Normalized):")
print(np.round(x_classical, 3))
# Fidelity calculation in NumPyLinearSolver might differ or be absent in newer versions
# print(f"Classical fidelity (if available): {classical_solution.fidelity:.4f}")


# --- 4. VQLS Implementation (Qiskit) ---
print("\n--- VQLS Setup ---")

# Choose an ansatz (variational circuit)
# RealAmplitudes is often a good starting point for real matrices,
# but EfficientSU2 might be better for complex ones. Let's use RealAmplitudes
# for simplicity, acknowledging it might not be optimal for this complex b.
# For N=8 (3 qubits), reps=3 gives 3*(3+1) = 12 parameters.
ansatz = RealAmplitudes(num_qubits=num_qubits, entanglement='linear', reps=3)
print(f"Using Ansatz: {ansatz.name} with {ansatz.num_parameters} parameters")
# ansatz.decompose().draw('mpl', style='iqx') # Uncomment to visualize circuit
# plt.show()

# Choose a classical optimizer
# SPSA is often good for noisy/simulated environments
# Use the updated import path
max_iterations = 150 # Increase for potentially better results, but slower
optimizer = SPSA(maxiter=max_iterations)
print(f"Using Optimizer: SPSA with max {max_iterations} iterations")

# Primitives for VQLS - Instantiate using the successfully imported classes
# The try-except block above should have defined Sampler and Estimator
estimator = Estimator()
sampler = Sampler()


# Initialize VQLS
# Use the updated import path
vqls = VQLS(
    estimator=estimator,
    ansatz=ansatz,
    optimizer=optimizer,
    sampler=sampler
)

print("\n--- Running VQLS (this may take a few minutes) ---")
# Solve the system using VQLS
# This runs the optimization loop
vqls_result = vqls.solve(A_matrix, b_vector)

# Extract the VQLS solution (approximation of the classical vector)
# VQLS result state is typically represented by the circuit
# To get the statevector, we need to simulate the circuit with optimal parameters
solution_circuit = vqls.ansatz.bind_parameters(vqls_result.optimal_point)
# Simulate the circuit using AerSimulator for the statevector
aer_sim = AerSimulator()
simulated_circuit = solution_circuit.copy() # Create a copy to avoid modifying the original
simulated_circuit.save_statevector() # Tell simulator to save the statevector
result = aer_sim.run(simulated_circuit).result()
solution_vector = result.get_statevector(simulated_circuit).data


# solution_vector = Statevector(solution_circuit).data # This might use BasicAer which can be slow
x_vqls = solution_vector / np.linalg.norm(solution_vector) # Normalize

print("\n--- VQLS Result ---")
print(f"VQLS Optimal Parameters: {np.round(vqls_result.optimal_point, 3)}")
# The attribute might be named differently, e.g., cost or objective_function_evals
# Check the actual result object attributes if this line causes an error
if hasattr(vqls_result, 'cost_function_evaluation'):
    print(f"VQLS Final Cost Function Value: {vqls_result.cost_function_evaluation:.4f}")
elif hasattr(vqls_result, 'cost'):
     print(f"VQLS Final Cost Function Value: {vqls_result.cost:.4f}")
else:
    print("Could not determine final cost function value attribute.")

print("VQLS solution vector x (Normalized):")
print(np.round(x_vqls, 3))

# Compare VQLS solution with classical solution
fidelity = np.abs(np.dot(x_classical.conj(), x_vqls))**2
print(f"\nFidelity between classical and VQLS solutions: {fidelity:.4f}")

# --- 5. RCS Calculation (Simplified) ---
# Define a representative far-field vector 'f'
# In a real scenario, this depends on the angle 
