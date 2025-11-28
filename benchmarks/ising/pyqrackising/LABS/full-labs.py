# Spin Glass Ground State (considered NP-complete)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)
# Refactored for formatting and execution

try:
    from pyqrackising import spin_glass_solver
except ImportError:
    print("Error: pyqrackising not found. Please install it via pip or ensure it is in your path.")
    print("This script requires the 'pyqrackising' library to solve the generated Ising/QUBO model.")
    import sys
    sys.exit(1)

import numpy as np
import sys

# LABS spin glass
def labs_to_maxcut_adjacency(N: int, lam: float = 5.0) -> np.ndarray:
    """
    Construct a weighted adjacency matrix for the MAXCUT encoding of
    the Low-Autocorrelation Binary Sequence (LABS) problem.

    Args:
        N (int): length of the binary sequence.
        lam (float): penalty weight for enforcing y_{i,k} = s_i * s_{i+k}.

    Returns:
        np.ndarray: (M x M) adjacency matrix, where M = N + N*(N-1)/2
    """

    num_y = N * (N - 1) // 2
    M = N + num_y
    A = np.zeros((M, M))

    # Helper to map (i, k) to y-index
    def y_index(i, k):
        idx = 0
        for kk in range(1, N):
            if kk < k:
                idx += N - kk
            else:
                break
        return N + idx + i

    # Add autocorrelation quadratic terms (y_{i,k} y_{j,k})
    for k in range(1, N):
        for i in range(N - k):
            for j in range(i + 1, N - k):
                yi = y_index(i, k)
                yj = y_index(j, k)
                A[yi, yj] += 1.0
                A[yj, yi] += 1.0

    # Add constraint penalty terms (-2 λ y_{i,k} s_i s_{i+k})
    for k in range(1, N):
        for i in range(N - k):
            yi = y_index(i, k)
            si = i
            sj = i + k
            A[yi, si] += -2.0 * lam
            A[si, yi] += -2.0 * lam
            A[yi, sj] += -2.0 * lam
            A[sj, yi] += -2.0 * lam

    # No self-loops (diagonal remains zero)
    np.fill_diagonal(A, 0.0)

    # Normalize
    if A.max() != 0 or A.min() != 0:
        A /= max(A.max(), -A.min())

    return A


if __name__ == "__main__":
    # Parse command line arguments
    # Usage: python labs_solver.py [N_nodes] [Lambda] [Quality_Level]
    n_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    lam = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    quality = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    print(f"Generating LABS MAXCUT adjacency for N={n_nodes}, Lambda={lam}...")
    G_m = labs_to_maxcut_adjacency(n_nodes, lam)
    
    print(f"Matrix generated. Size: {G_m.shape[0]}x{G_m.shape[0]}")
    print("Solving ground state...")
    
    best_bitstring, best_cut_value, best_cut, best_energy = spin_glass_solver(
        G_m, 
        quality=quality, 
        is_spin_glass=False
    )

    print("-" * 40)
    print(f"Best cut: {best_cut_value}")
    
    # Extract original solution if needed (first N bits usually correspond to original spins)
    original_sequence = best_bitstring[:n_nodes] if len(best_bitstring) >= n_nodes else best_bitstring
    print(f"Original Sequence (first {n_nodes} bits): {original_sequence}")
    
    # MODIFIED: Always print the full raw bitstring regardless of length
    print(f"\nFull raw bitstring ({len(best_bitstring)} bits):")
    print(best_bitstring)
    
