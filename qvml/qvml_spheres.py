import re
import networkx as nx
import vedo
import argparse
import sys
from collections import defaultdict

def parse_tensor_info(filepath):
    """
    Parses a qvml.py output file to extract the list of tensors
    and their indices from the 'Contraction result' section.
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'", file=sys.stderr)
        return None

    match = re.search(r"Contraction result:(.*)", content, re.DOTALL)
    if not match:
        print("Error: Could not find 'Contraction result:' in the file.", file=sys.stderr)
        return None
    
    contraction_block = match.group(1)
    inds_matches = re.findall(r"inds=\((.*?)\)", contraction_block)
    
    if not inds_matches:
        print("Error: Could not parse any tensor indices from the file.", file=sys.stderr)
        return None

    all_indices = []
    for match_str in inds_matches:
        indices = [idx.strip().strip("'\"") for idx in match_str.split(',')]
        all_indices.append(indices)
        
    return all_indices

def build_graph(tensor_indices_list):
    """
    Builds a NetworkX graph from the parsed tensor data.
    """
    G = nx.Graph()
    index_to_tensors = defaultdict(list)

    for i, indices in enumerate(tensor_indices_list):
        tensor_id = f"T{i}"
        G.add_node(tensor_id, type='tensor', rank=len(indices), info=f"Tensor {i}\nRank: {len(indices)}")
        for index_label in indices:
            if index_label:
                index_to_tensors[index_label].append(tensor_id)

    for index_label, tensors in index_to_tensors.items():
        if len(tensors) == 2:
            G.add_edge(tensors[0], tensors[1], label=index_label)
        elif len(tensors) == 1:
            tensor_id = tensors[0]
            G.add_node(index_label, type='open_index', info=f"Open Index:\n{index_label}")
            G.add_edge(tensor_id, index_label, label=index_label)

    return G

def visualize_3d_vedo(G):
    """
    Creates and displays an interactive 3D plot of the tensor network
    graph using the Vedo library, with adjusted sphere sizes, dark mode,
    and 50% transparency for spheres.
    """
    if not G.nodes:
        print("Graph is empty, nothing to visualize.")
        return

    print("Computing 3D graph layout...")
    pos = nx.spring_layout(G, dim=3, seed=42, iterations=100)

    actors = []
    
    # Get the max rank to normalize the color map
    max_rank = max(nx.get_node_attributes(G, 'rank').values()) if G.nodes else 1

    for node, data in G.nodes(data=True):
        p = pos[node]
        if data['type'] == 'tensor':
            rank = data['rank']
            radius = 0.01 * rank + 0.03
            color = vedo.color_map(rank, 'viridis', vmin=1, vmax=max_rank)
            # Add alpha=0.5 for 50% transparency
            sphere = vedo.Sphere(pos=p, r=radius, c=color, alpha=0.5) 
        else: # 'open_index'
            # Add alpha=0.5 for 50% transparency
            sphere = vedo.Sphere(pos=p, r=0.04, c='red5', alpha=0.5) 
        
        sphere.name = data['info']
        actors.append(sphere)

    for u, v in G.edges():
        p1 = pos[u]
        p2 = pos[v]
        line = vedo.Line(p1, p2, c='grey', lw=2, alpha=0.7)
        actors.append(line)

    print("Launching 3D visualization window...")
    # --- MODIFIED PLOTTER FOR DARK MODE ---
    plt = vedo.Plotter(title='3D Tensor Network Visualization', axes=1, bg='black') # Set background to black
    plt.add(actors)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a contracted tensor network in 3D using Vedo."
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the simulation output file (e.g., qvml_results/qubits_19_depth_9.txt)"
    )
    
    args = parser.parse_args()
    
    tensor_indices = parse_tensor_info(args.filepath)
    if tensor_indices:
        graph = build_graph(tensor_indices)
        visualize_3d_vedo(graph)
