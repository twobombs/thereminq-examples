# -*- coding: utf-8 -*-
# from: https://github.com/vm6502q/PyQrackIsing/blob/main/scripts/tsp_benchmarks.py

import networkx as nx
import random
import time
import math
import pandas as pd
import csv
import os
import argparse

from PyQrackIsing import tsp_symmetric


# Generate a clustered TSP instance with n nodes in the unit square
def generate_clustered_tsp(n, clusters=4, spread=0.05, seed=42):
    """Generates a complete graph with nodes arranged in clusters."""
    random.seed(seed)
    points = {}
    centers = [(random.random(), random.random()) for _ in range(clusters)]
    for i in range(n):
        cx, cy = random.choice(centers)
        px = min(max(cx + random.uniform(-spread, spread), 0), 1)
        py = min(max(cy + random.uniform(-spread, spread), 0), 1)
        points[i] = (px, py)

    G = nx.complete_graph(n)
    for u, v in G.edges():
        x1, y1 = points[u]
        x2, y2 = points[v]
        G[u][v]["weight"] = math.hypot(x1 - x2, y1 - y2)
    return G, points


# Check if the returned path is a valid Hamiltonian cycle
def validate_tsp_solution(G, path):
    """Validates that the path visits every node exactly once."""
    return len(path) == len(G.nodes) + 1 and set(path[:-1]) == set(G.nodes)


# Benchmark framework for the PyQrackIsing TSP solver
def benchmark_pyqrackising(n_nodes=64, trials=3):
    """Runs and times the PyQrackIsing TSP solver."""
    results = []
    G, _ = generate_clustered_tsp(n_nodes)

    # Exclude numba JIT compilation overhead with a warm-up run
    # This ensures our timing is more accurate for the actual runs.
    print(f"Warming up solver for {n_nodes} nodes...")
    tsp_symmetric(G)
    print("Warm-up complete.")

    print(f"Running {trials} trials for {n_nodes} nodes...")
    for trial in range(trials):
        start_time = time.time()
        path, length = tsp_symmetric(G)
        end_time = time.time()
        
        # Store the time and length for this trial
        results.append((end_time - start_time, length))
        
        # Ensure the solution is valid
        assert validate_tsp_solution(G, path), f"Invalid PyQrackIsing solution in trial {trial}"
        print(f"  Trial {trial + 1}/{trials} complete. Time: {end_time - start_time:.4f}s, Length: {length:.6f}")

    return results


# --- Main Execution ---

# Setup CLI argument parsing
parser = argparse.ArgumentParser(description="Benchmark the PyQrackIsing TSP solver.")
parser.add_argument('node_sizes', type=int, nargs='*', 
                    help="A space-separated list of node sizes to test (e.g., 32 64 128). If none are given, a default set will be used.")
args = parser.parse_args()

# Define the problem sizes to test based on CLI input
if args.node_sizes:
    node_sizes = args.node_sizes
    print(f"Using custom node sizes from command line: {node_sizes}")
else:
    node_sizes = [32, 64, 128, 256]
    print(f"No node sizes provided. Using default set: {node_sizes}")

summary_data = []
# Create a dynamic CSV filename based on the node sizes being tested
node_sizes_str = "_".join(map(str, node_sizes))
csv_file = f"pyqrackising_benchmark_results_{node_sizes_str}.csv"

print("\n--- Starting PyQrackIsing TSP Benchmark ---")
print("-" * 45)

for n in node_sizes:
    # Run the benchmark for the current node size
    trial_results = benchmark_pyqrackising(n_nodes=n, trials=3)
    
    # Process the results for this size
    transposed = list(zip(*trial_results))
    avg_time = sum(transposed[0]) / len(transposed[0])
    min_length = min(transposed[1])
    
    # Store data for final summary
    summary_data.append({
        "Nodes": n,
        "Avg. Time (s)": avg_time,
        "Best Path Length": min_length,
    })

    # --- Intermediate Reporting ---
    print("\n--- Intermediate Result ---")
    print(f"Problem Size: {n} Nodes")
    print(f"  Average Time: {avg_time:.6f} seconds")
    print(f"  Best Length:  {min_length:.6f}")
    print("-" * 45)


# --- Final CSV Output ---
try:
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Nodes", "Avg. Time (s)", "Best Path Length"])
        writer.writeheader()
        writer.writerows(summary_data)
    print(f"\n Benchmark complete. Results saved to '{os.path.abspath(csv_file)}'")
except IOError as e:
    print(f"\n Error writing to CSV file: {e}")

# Display a final summary table in the console as well
summary_df = pd.DataFrame(summary_data)
print("\n--- Final Benchmark Summary ---")
print(summary_df.to_string(index=False))

