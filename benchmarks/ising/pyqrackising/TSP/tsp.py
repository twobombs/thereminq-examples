# Traveling Salesman Problem (considered NP-complete)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)

from PyQrackIsing import tsp_symmetric
import networkx as nx
import numpy as np
import sys


# Traveling Salesman Problem (normalized to longest segment)
def generate_tsp_graph(n_nodes=64, seed=None):
    if not (seed is None):
        np.random.seed(seed)
    G = nx.Graph()
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            G.add_edge(u, v, weight=np.random.random())
    return G


if __name__ == "__main__":
    # NP-complete TSP
    n_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    quality = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    correction_quality = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    multi_start = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else None
    G = generate_tsp_graph(n_nodes=n_nodes, seed=seed)
    best_circuit, best_path_length = tsp_symmetric(G, quality=quality, correction_quality=correction_quality, multi_start=multi_start)

    reconstructed_node_count = len(set(best_circuit))
    reconstructed_path_length = 0
    best_nodes = None
    for i in range(len(best_circuit) - 1):
        reconstructed_path_length += G[best_circuit[i]][best_circuit[i + 1]]["weight"]

    print(f"Random seed: {seed}")
    print(f"Path: {best_circuit}")
    print(f"Actual node count: {n_nodes}")
    print(f"Solution distinct node count: {reconstructed_node_count}")
    print(f"Claimed path length: {best_path_length}")
    print(f"Verified path length: {reconstructed_path_length}")
    print(
        "(The average randomized and normalized separation between each and every node is about 0.5.)"
    )
