# Import necessary libraries from Qiskit
import numpy as np
from qiskit import QuantumCircuit, transpile
# Note: BasicSimulator is deprecated, using primitives is recommended.
# from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import Statevector

# --- Qiskit Primitives Import ---
# Sampler is directly under qiskit.primitives in recent versions
from qiskit.primitives import Sampler # Use Sampler primitive for execution

# --- Qiskit Algorithms Import ---
# HHL is now in the qiskit_algorithms package
# Ensure you have installed it: pip install qiskit_algorithms
from qiskit_algorithms.linear_solvers.hhl import HHL

# --- 1. Define the Problem ---
# We want to solve Ax = b for x.
# Let's choose a simple 2x2 example:
# A = [[1, -1/3], [-1/3, 1]]
# b = [1, 0]
# The exact classical solution is x = [1.125, 0.375]

matrix = np.array([[1, -1/3], [-1/3, 1]])
vector = np.array([1, 0])

# --- 2. Instantiate the HHL Algorithm ---
# Qiskit's HHL class encapsulates the steps shown in the slide:
# - State preparation for 'vector' (b)
# - Quantum Phase Estimation (QPE) for 'matrix' (A)
# - Eigenvalue inversion using controlled rotation
# - Uncomputation and measurement
hhl_solver = HHL() # Uses standard multi-bit QPE by default

# --- 3. Solve the System ---
# The solve method constructs the circuit and runs it.
# It requires the matrix, the vector, and a quantum instance or primitive.
sampler = Sampler() # Use the Sampler primitive
solution = hhl_solver.solve(matrix, vector, sampler)

# --- 4. Analyze the Results ---
# The HHL algorithm doesn't directly output the full solution vector x.
# It prepares a quantum state |x> such that the amplitudes are proportional to x.
# We can get the estimated solution vector from the circuit's output statevector
# or by looking at the measurement probabilities (classical output).

# Get the full solution vector (requires simulation)
# The 'state' attribute might be a QuantumCircuit or a Statevector depending
# on the execution path. Accessing statevector directly might fail if it's
# just the circuit. The HHLResult object provides better ways.

# print("Full state vector from HHL circuit:")
# print(solution.state.get_statevector().data) # Might error if state is circuit

# Calculate the probability of measuring the ancilla qubit in state |1>
# This probability is related to the success of the eigenvalue inversion.
# Accessing probabilities directly from the result object is safer.
# prob_ancilla_1 = solution.state.probabilities_dict()['1'] # Might error
# print(f"\nProbability of measuring ancilla in |1>: {prob_ancilla_1:.4f}")

# The solution vector is proportional to the state in the |b> register
# *when the ancilla qubit is measured as |1>*.
# We need to normalize the relevant amplitudes.

# Extract amplitudes where the ancilla (last qubit added by HHL) is 1
# The HHLResult object simplifies accessing the solution.

print(f"\nEuclidean norm of the solution vector ||x||: {solution.euclidean_norm:.4f}")

# Get the classical solution vector (rescaled) from HHL result
classical_solution = np.real(solution.solution) # Use solution.solution
print("\nClassical solution vector from HHL (proportional to x):")
print(classical_solution)

# Compare with the expected solution (rescaled)
exact_solution = np.array([1.125, 0.375])
norm_exact = np.linalg.norm(exact_solution)
rescaled_exact = exact_solution / norm_exact

norm_hhl = np.linalg.norm(classical_solution)
# Handle potential zero norm if HHL fails
if norm_hhl > 1e-9:
    rescaled_hhl = classical_solution / norm_hhl # Normalize the HHL output
else:
    rescaled_hhl = classical_solution # Avoid division by zero
    print("\nWarning: HHL solution norm is close to zero.")


print("\nExact solution (normalized):")
print(rescaled_exact)
print("\nHHL solution (normalized):")
print(rescaled_hhl)

# --- 5. Show the Circuit (Optional) ---
# You can also inspect the quantum circuit built by HHL
hhl_circuit = solution.circuit
print(f"\nTotal number of qubits in HHL circuit: {hhl_circuit.num_qubits}")
print(f"Circuit depth: {hhl_circuit.depth()}")
# print("\nQuantum Circuit for HHL:")
# print(hhl_circuit.decompose(reps=3).draw(output='text', fold=-1)) # Decompose


# --- Discussion on 1-bit QPE (Slide 3) ---
# The code above uses standard QPE. 1-bit QPE (Iterative Phase Estimation - IPE)
# reduces resources by using only one ancilla qubit for phase estimation,
# measuring it, and feeding the result forward classically to refine the estimate
# over multiple iterations.
# While Qiskit Algorithms has tools for IPE (like qiskit_algorithms.IterativePhaseEstimation),
# integrating it directly into the HHL algorithm structure requires manual
# circuit construction and is significantly more complex than using the HHL class.
# The benefit, as shown in slide 3, is a large reduction in circuit depth and
# qubit count, crucial for near-term devices.

