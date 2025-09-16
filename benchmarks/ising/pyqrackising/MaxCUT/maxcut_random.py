# Random MAXCUT (for execution time tests)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)

from pyqrackising import spin_glass_solver
from numba import njit, prange
import numpy as np
import sys
import time


# Random MAXCUT adjacency matrix
def generate_adjacency(n_nodes=64, seed=None):
    if not (seed is None):
        np.random.seed(seed)

    G_m = np.empty((n_nodes, n_nodes), dtype=np.float64)

    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            weight = np.random.random()
            G_m[u, v] = weight
            G_m[v, u] = weight

    return G_m


if __name__ == "__main__":
    n_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    quality = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    correction_quality = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else None

    start = time.perf_counter()
    G_m = generate_adjacency(n_nodes=n_nodes, seed=seed)
    seconds = time.perf_counter() - start
    print(f"{seconds} seconds to initialize the adjacency matrix (statement of the problem itself)")

    print(f"Random seed: {seed}")
    print(f"Node count: {n_nodes}")
    start = time.perf_counter()
    bitstring, cut_value, cut, energy = spin_glass_solver(G_m, quality=quality, correction_quality=correction_quality)
    seconds = time.perf_counter() - start
    
    # The debug line you wanted to keep
    print(f"DEBUG - Raw bitstring variable: {bitstring}")

    print(f"Seconds to solution: {seconds}")
    
    # --- FINAL FIX ---
    # 1. Convert the result string to a list of characters (e.g., '101' -> ['1', '0', '1'])
    # 2. Convert that list into a NumPy array of integers (e.g., [1, 0, 1])
    bits_array = np.array(list(bitstring), dtype=int)
    
    # 3. Now, use this proper array to find the node sets
    nodes = np.arange(n_nodes)
    set_A = nodes[bits_array == 0]
    set_B = nodes[bits_array == 1]
    
    print(f"Set A nodes: {set_A.tolist()}")
    print(f"Set B nodes: {set_B.tolist()}")
    
    print(f"Cut weight: {cut_value}")
    print(
        "(The average randomized and symmetric weight between each and every node is about 0.5, from the range 0.0 to 1.0.)"
    )
