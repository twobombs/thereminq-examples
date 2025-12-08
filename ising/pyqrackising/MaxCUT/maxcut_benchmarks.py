# -*- coding: utf-8 -*-
# from: https://github.com/vm6502q/PyQrackIsing/blob/main/scripts/maxcut_benchmarks.py

import networkx as nx
import numpy as np
import time
import csv
import argparse

from PyQrackIsing import spin_glass_solver

# --- 1. Erdos-Renyi ---
def erdos_renyi_graph(n=128, p=0.5, seed=None):
    """Generates an Erdos-Renyi random graph."""
    return nx.erdos_renyi_graph(n, p, seed=seed)

# --- 2. Planted-partition ---
def planted_partition_graph(n=128, p_in=0.2, p_out=0.8, seed=None):
    """Generates a graph with a known community structure."""
    # Split into 2 equal communities
    sizes = [n // 2, n - n // 2]
    probs = [[p_in, p_out], [p_out, p_in]]
    return nx.stochastic_block_model(sizes, probs, seed=seed)

# --- 3. Hard instances (regular bipartite expander as example) ---
def hard_instance_graph(n=128, d=10, seed=None):
    """Generates a d-regular bipartite graph, often a challenging instance."""
    return nx.random_regular_graph(d, n, seed=seed)

def evaluate_cut_value(G, partition):
    """Compute cut value directly from the graph and a given partition."""
    cut = 0
    # Ensure partition is a tuple of two sets for verification
    if not (isinstance(partition, (tuple, list)) and len(partition) == 2 and all(isinstance(s, set) for s in partition)):
         raise TypeError("Partition must be a tuple or list of two sets.")

    partition_set1 = partition[0]
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        # Check if the edge crosses the partition
        if (u in partition_set1 and v not in partition_set1) or \
           (v in partition_set1 and u not in partition_set1):
            cut += w
    return cut

def benchmark_maxcut(generators, sizes=[64, 128, 256], seed=42, trials=1):
    """
    Benchmarks the Qrack spin glass solver for the Max-Cut problem.
    """
    results = {}
    for n in sizes:
        print(f"\n--- Benchmarking for size n={n} ---")
        results[n] = {}
        for key, value in generators.items():
            print(f"  Graph Type: {key}")
            results[n][key] = []
            for t in range(trials):
                # Generate random graph
                G = value[0](n=n, **(value[1]), seed=seed + t)

                results_dict = {}

                # --- Qrack solver ---
                start = time.time()
                _, cut_value, partition_from_solver, _ = spin_glass_solver(G)
                runtime = time.time() - start

                # Convert the solver's (list, list) tuple to a (set, set) tuple
                partition_for_eval = (set(partition_from_solver[0]), set(partition_from_solver[1]))

                # Pass the correctly formatted partition to the verification function
                verified = evaluate_cut_value(G, partition_for_eval)

                # Use a tolerance for floating point comparison
                assert np.isclose(cut_value, verified), f"Cut value mismatch: {cut_value} != {verified}"

                results_dict["Qrack"] = (cut_value, runtime)
                print(f"    Trial {t+1}/{trials}: Cut = {cut_value:.2f}, Time = {runtime:.4f}s")

                results[n][key].append(results_dict)

    return results

if __name__ == "__main__":
    # --- Set up CLI argument parsing ---
    parser = argparse.ArgumentParser(description="Run Max-Cut benchmarks with a Qrack solver.")
    parser.add_argument(
        '--sizes',
        type=int,
        nargs='+',
        default=[64, 128, 256],
        help='One or more graph sizes (number of nodes) to benchmark. Example: --sizes 32 64'
    )
    args = parser.parse_args()

    # Define the graph generators to test
    graph_generators = {
        "Erdos-Renyi": (erdos_renyi_graph, {"p": 0.5}),
        "Planted-partition": (planted_partition_graph, {"p_in": 0.2, "p_out": 0.8}),
        "Hard (bipartite expander)": (hard_instance_graph, {"d": 10}),
    }

    # Run the benchmark with the specified or default sizes
    benchmark_results = benchmark_maxcut(graph_generators, sizes=args.sizes)

    # Print the final results dictionary
    print("\n--- Final Results ---")
    print(benchmark_results)

    # --- Save results to a CSV file with a dynamic name ---
    sizes_str = "_".join(map(str, args.sizes))
    csv_file_path = f"maxcut_benchmark_results_{sizes_str}.csv"
    csv_header = ["Size", "GraphType", "Trial", "Solver", "CutValue", "Time"]

    with open(csv_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        
        for size, graph_data in benchmark_results.items():
            for graph_type, trials_list in graph_data.items():
                for trial_num, trial_result in enumerate(trials_list):
                    for solver, performance in trial_result.items():
                        cut = performance[0]
                        runtime = performance[1]
                        row = [size, graph_type, trial_num + 1, solver, cut, runtime]
                        writer.writerow(row)

    print(f"\nBenchmark results have been saved to {csv_file_path}")

