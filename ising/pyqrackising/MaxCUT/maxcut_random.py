# Random MAXCUT (for execution time tests)
# from https://github.com/vm6502q/PyQrackIsing/blob/main/scripts/maxcut_random.py
# modified to produce the node paths and support CLI flags

from pyqrackising import spin_glass_solver
from numba import njit, prange
import numpy as np
import argparse
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
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Solve a random MAXCUT problem using PyQrackIsing.")
    
    # --- MODIFICATION START ---
    # Changed positional arguments to named, optional arguments
    parser.add_argument("--n-nodes", type=int, default=64, help="Number of nodes in the graph (default: 64)")
    parser.add_argument("--quality", type=int, default=1, help="Quality parameter for the solver (default: 1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: None)")
    # --- MODIFICATION END ---
    
    # Add the boolean flag for alternative GPU sampling
    parser.add_argument(
        "--is-alt-gpu",
        action="store_true",
        help="Use the alternative full-GPU sampling algorithm. Default is False."
    )
    
    # Parse the arguments from the command line
    args = parser.parse_args()

    start = time.perf_counter()
    G_m = generate_adjacency(n_nodes=args.n_nodes, seed=args.seed)
    seconds = time.perf_counter() - start
    print(f"{seconds} seconds to initialize the adjacency matrix (statement of the problem itself)")

    print(f"Random seed: {args.seed}")
    print(f"Node count: {args.n_nodes}")
    print(f"Using alternative GPU sampling: {args.is_alt_gpu}")
    
    start = time.perf_counter()
    # Call the solver (no changes needed here)
    bitstring, cut_value, cut, energy = spin_glass_solver(
        G_m, 
        quality=args.quality, 
        is_alt_gpu_sampling=args.is_alt_gpu
    )
    seconds = time.perf_counter() - start
    
    print(f"DEBUG - Raw bitstring variable: {bitstring}")
    print(f"Seconds to solution: {seconds}")
    
    bits_array = np.array(list(bitstring), dtype=int)
    
    nodes = np.arange(args.n_nodes)
    set_A = nodes[bits_array == 0]
    set_B = nodes[bits_array == 1]
    
    print(f"Set A nodes: {set_A.tolist()}")
    print(f"Set B nodes: {set_B.tolist()}")
    
    print(f"Cut weight: {cut_value}")
    print(
        "(The average randomized and symmetric weight between each and every node is about 0.5, from the range 0.0 to 1.0.)"
    )
