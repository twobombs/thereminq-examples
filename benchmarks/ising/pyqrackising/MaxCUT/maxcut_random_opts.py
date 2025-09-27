# Random MAXCUT (for execution time tests)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)
# Modified by Gemini to include argparse for command-line options

from pyqrackising import spin_glass_solver
from numba import njit, prange
import numpy as np
import argparse  # Import the argparse library
import time


# Random MAXCUT adjacency matrix
@njit
def generate_adjacency(n_nodes=64, seed=None):
    if not (seed is None):
        np.random.seed(seed)

    G_m = np.empty((n_nodes, n_nodes), dtype=np.float64)

    for u in prange(n_nodes):
        for v in range(u + 1, n_nodes):
            weight = np.random.random()
            G_m[u, v] = weight
            G_m[v, u] = weight

    return G_m


if __name__ == "__main__":
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(description="Run a MAXCUT simulation with PyQrackIsing.")
    parser.add_argument("-n", "--nodes", type=int, default=64, help="Number of nodes in the graph.")
    parser.add_argument("-q", "--quality", type=int, default=3, help="Quality of the QAOA simulation.")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed for graph generation.")
    parser.add_argument(
        "--use-alt-gpu", action="store_true", help="Enable the alternate all-GPU sampling method."
    )
    
    # 2. Parse the arguments from the command line
    args = parser.parse_args()

    start = time.perf_counter()
    G_m = generate_adjacency(n_nodes=args.nodes, seed=args.seed)
    seconds = time.perf_counter() - start
    print(f"{seconds} seconds to initialize the adjacency matrix (statement of the problem itself)")

    print(f"Random seed: {args.seed}")
    print(f"Node count: {args.nodes}")
    print(f"Alternate GPU Sampling: {args.use_alt_gpu}") # Acknowledges the setting

    start = time.perf_counter()
    # 3. Pass the new argument to the solver
    bitstring, cut_value, cut, energy = spin_glass_solver(
        G_m, quality=args.quality, is_alt_gpu_sampling=args.use_alt_gpu
    )
    seconds = time.perf_counter() - start

    print(f"Seconds to solution: {seconds}")
    print(f"Bipartite cut bit string: {bitstring}")
    print(f"Cut weight: {cut_value}")
    print(
        "(The average randomized and symmetric weight between each and every node is about 0.5, from the range 0.0 to 1.0.)"
    )
