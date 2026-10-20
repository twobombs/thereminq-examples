"""
HHL Algorithm Implementation

This module demonstrates the Harrow-Hassidim-Lloyd (HHL) algorithm for solving
linear systems of equations using quantum computing.

The HHL algorithm solves Ax = b for x, where A is a matrix and b is a vector.
It prepares a quantum state |x> proportional to the solution vector.

References:
    - Harrow, A. W., Hassidim, A., & Lloyd, S. (2009). Quantum algorithm for
      linear systems of equations. Physical Review Letters, 103(15), 150502.
    - Qiskit Algorithms: https://qiskit.org/ecosystem/algorithms/
"""

import logging
from typing import Tuple, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit_algorithms.linear_solvers.hhl import HHL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HHLProblem:
    """
    A class to represent and solve linear systems Ax = b using the HHL algorithm.
    
    Attributes:
        matrix: The coefficient matrix A (must be Hermitian and positive definite).
        vector: The right-hand side vector b.
    """
    
    def __init__(self, matrix: np.ndarray, vector: np.ndarray) -> None:
        """
        Initialize the HHL problem with a matrix and vector.
        
        Args:
            matrix: The coefficient matrix A. Must be Hermitian and positive definite.
            vector: The right-hand side vector b.
        
        Raises:
            ValueError: If the matrix is not Hermitian or positive definite,
                       or if the vector dimensions don't match the matrix.
        """
        self._validate_matrix(matrix)
        self._validate_vector(vector, matrix.shape[0])
        self.matrix = matrix
        self.vector = vector
        logger.info(f"HHL problem initialized: {matrix.shape[0]}x{matrix.shape[0]} system")
    
    def _validate_matrix(self, matrix: np.ndarray) -> None:
        """
        Validate that the matrix is Hermitian and positive definite.
        
        Args:
            matrix: The matrix to validate.
        
        Raises:
            ValueError: If the matrix is not Hermitian or positive definite.
        """
        matrix = np.asarray(matrix, dtype=complex)
        
        # Check if matrix is square
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square")
        
        # Check if matrix is Hermitian
        if not np.allclose(matrix, matrix.conj().T):
            raise ValueError("Matrix must be Hermitian (A = A†)")
        
        # Check if matrix is positive definite
        eigenvalues = np.linalg.eigvals(matrix)
        if not np.all(np.real(eigenvalues) > 0):
            raise ValueError("Matrix must be positive definite (all eigenvalues > 0)")
        
        logger.info(f"Matrix validated: eigenvalues = {np.real(eigenvalues)}")
    
    def _validate_vector(self, vector: np.ndarray, expected_size: int) -> None:
        """
        Validate that the vector has the correct dimensions.
        
        Args:
            vector: The vector to validate.
            expected_size: The expected size of the vector.
        
        Raises:
            ValueError: If the vector dimensions don't match the matrix.
        """
        vector = np.asarray(vector, dtype=complex)
        
        if vector.ndim != 1 or vector.shape[0] != expected_size:
            raise ValueError(
                f"Vector size {vector.shape[0]} doesn't match matrix size {expected_size}"
            )
        
        logger.info("Vector validated")
    
    def solve(
        self,
        sampler: Optional[Sampler] = None,
        max_iterations: int = 100
    ) -> Tuple[np.ndarray, float, int]:
        """
        Solve the linear system Ax = b using the HHL algorithm.
        
        Args:
            sampler: Optional Sampler primitive for execution. If None, a new one is created.
            max_iterations: Maximum number of iterations for the solver.
        
        Returns:
            A tuple containing:
                - classical_solution: The classical solution vector (normalized).
                - euclidean_norm: The Euclidean norm of the solution.
                - iterations: Number of iterations performed.
        
        Raises:
            RuntimeError: If the HHL solver fails to converge.
        """
        if sampler is None:
            sampler = Sampler()
        
        try:
            logger.info("Starting HHL solver...")
            hhl_solver = HHL(max_iterations=max_iterations)
            solution = hhl_solver.solve(self.matrix, self.vector, sampler)
            
            # Get the classical solution vector
            classical_solution = np.real(solution.solution)
            euclidean_norm = solution.euclidean_norm
            
            logger.info(f"HHL solver completed: ||x|| = {euclidean_norm:.4f}")
            return classical_solution, euclidean_norm, max_iterations
            
        except Exception as e:
            logger.error(f"HHL solver failed: {e}")
            raise RuntimeError(f"HHL solver failed: {e}")
    
    def get_exact_solution(self) -> np.ndarray:
        """
        Compute the exact classical solution using numpy.
        
        Returns:
            The exact solution vector x = A^(-1)b.
        """
        return np.linalg.solve(self.matrix, self.vector)


def main() -> None:
    """
    Main entry point for the HHL demonstration.
    
    Solves a 2x2 linear system Ax = b using the HHL algorithm and compares
    the quantum solution with the exact classical solution.
    """
    # Define the problem: Ax = b
    # A = [[1, -1/3], [-1/3, 1]]
    # b = [1, 0]
    # The exact classical solution is x = [1.125, 0.375]
    
    matrix = np.array([[1, -1/3], [-1/3, 1]], dtype=float)
    vector = np.array([1, 0], dtype=float)
    
    try:
        # Create and solve the HHL problem
        hhl_problem = HHLProblem(matrix, vector)
        
        # Get the exact classical solution for comparison
        exact_solution = hhl_problem.get_exact_solution()
        logger.info(f"Exact solution: {exact_solution}")
        
        # Solve using HHL
        hhl_solution, norm, iterations = hhl_problem.solve()
        
        # Normalize solutions for comparison
        norm_exact = np.linalg.norm(exact_solution)
        rescaled_exact = exact_solution / norm_exact
        
        if norm > 1e-9:
            rescaled_hhl = hhl_solution / norm
        else:
            rescaled_hhl = hhl_solution
            logger.warning("HHL solution norm is close to zero")
        
        # Display results
        print(f"\nEuclidean norm of the solution vector ||x||: {norm:.4f}")
        print(f"\nExact solution (normalized):")
        print(rescaled_exact)
        print("\nHHL solution (normalized):")
        print(rescaled_hhl)
        
        # Calculate similarity
        similarity = np.abs(np.dot(rescaled_exact, rescaled_hhl))
        print(f"\nSimilarity (inner product): {similarity:.4f}")
        
        # Show circuit information
        hhl_problem = HHLProblem(matrix, vector)
        sampler = Sampler()
        hhl_solver = HHL()
        solution = hhl_solver.solve(matrix, vector, sampler)
        hhl_circuit = solution.circuit
        print(f"\nTotal number of qubits in HHL circuit: {hhl_circuit.num_qubits}")
        print(f"Circuit depth: {hhl_circuit.depth()}")
        
    except ValueError as e:
        logger.error(f"Invalid problem setup: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"HHL computation failed: {e}")
        raise


if __name__ == "__main__":
    main()

