import re
import networkx as nx
import vedo
import argparse
import sys
import glob
from collections import defaultdict

def parse_tensor_info(filepath):
    """
    Parses a qvml.py output file to extract the list of tensors
    and their indices from the 'Contraction result:' section.
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'", file=sys.stderr)
        return None

    match = re.search(r"Contraction result:(.*)", content, re.DOTALL)
    if not match:
        print(f"Warning: Could not find 'Contraction result:' in file {filepath}", file=sys.stderr)
        return None
    
    contraction_block = match.group(1)
    inds_matches = re.findall(r"inds=\((.*?)\)", contraction_block)
    
    if not inds_matches:
        print(f"Warning: Could not parse any tensor indices from file {filepath}", file=sys.stderr)
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

def on_right_click(event, plotter):
    """
    Callback function to take a single high-resolution screenshot on right-click.
    """
    print("Right mouse button clicked! Taking high-resolution screenshot...")
    screenshot_filename = "tensor_network_screenshot.png"
    plotter.screenshot(filename=screenshot_filename, scale=5)
    print(f"Screenshot saved as {screenshot_filename}.")

def on_key_press(event, plotter, assemblies):
    """
    Callback function to handle key presses.
    If 'r' is pressed, starts a 360-degree rotation and screenshot sequence.
    """
    if event.key == 'r':
        print("'r' key pressed. Starting 360-degree rotation screenshot sequence...")
        
        # Define rotation parameters
        rotation_step = 5  # degrees per frame
        total_frames = 360 // rotation_step
        
        for i in range(total_frames):
            # Rotate every assembly
            for assembly in assemblies:
                assembly.rotate_z(rotation_step)
            
            # Force the scene to update
            plotter.render()
            
            # Save a numbered screenshot
            filename = f"rotation_frame_{i:03d}.png"
            plotter.screenshot(filename=filename, scale=5)
            print(f"Saved frame {i+1}/{total_frames}: {filename}")
        
        print("Rotation sequence complete.")


def main():
    """
    Main function to find, parse, and visualize multiple tensor networks in a logical grid.
    """
    parser = argparse.ArgumentParser(
        description="Visualize tensor networks in a 2D grid based on qubits (x-axis) and depth (y-axis).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--qubits', type=str, help="Specify the number of qubits (e.g., '10').")
    parser.add_argument('--depth', type=str, help="Specify the circuit depth (e.g., '8').")
    parser.add_argument('--dir', type=str, default='qvml_results', help="Directory containing the result files.")
    
    args = parser.parse_args()

    qubit_str = args.qubits if args.qubits else '*'
    depth_str = args.depth if args.depth else '*'
    
    search_pattern = f"{args.dir}/qubits_{qubit_str}_depth_{depth_str}.txt"
    files_to_load = sorted(glob.glob(search_pattern))

    if not files_to_load:
        print(f"No files found matching the pattern: {search_pattern}")
        sys.exit(0)
    
    if not args.qubits and not args.depth:
        print("No arguments provided. Visualizing all found networks...")
    print(f"Found {len(files_to_load)} matching files. Processing...")

    files_data = []
    all_qubits, all_depths = set(), set()
    for filepath in files_to_load:
        filename = filepath.split('/')[-1]
        try:
            q = int(filename.split('_')[1])
            d = int(filename.split('_')[3].split('.')[0])
            files_data.append({'filepath': filepath, 'qubits': q, 'depth': d})
            all_qubits.add(q)
            all_depths.add(d)
        except (IndexError, ValueError):
            print(f"Warning: Skipping malformed filename '{filename}'")
            continue

    if not files_data:
        print("No valid files to process.")
        sys.exit(0)

    min_qubits, max_qubits = min(all_qubits), max(all_qubits)
    min_depth, max_depth = min(all_depths), max(all_depths)
    
    all_assemblies = []
    static_actors = []
    grid_spacing = 4.0
    label_scale_individual = grid_spacing * 0.1
    label_scale_main = grid_spacing * 0.2

    for data in files_data:
        col = data['qubits'] - min_qubits
        row = data['depth'] - min_depth
        offset = (col * grid_spacing, -row * grid_spacing, 0)
        
        tensor_indices = parse_tensor_info(data['filepath'])
        if not tensor_indices: continue
        
        G = build_graph(tensor_indices)
        if not G.nodes: continue

        current_graph_actors = []
        pos = nx.spring_layout(G, dim=3, seed=42, iterations=100)
        
        for node, node_data in G.nodes(data=True):
            p = pos[node]
            # We apply the grid offset later, to the assembly
            
            if node_data['type'] == 'tensor':
                radius = 0.01 * node_data['rank'] + 0.03
                all_ranks = nx.get_node_attributes(G, 'rank').values()
                vmax = max(all_ranks) if all_ranks else 1
                color = vedo.color_map(node_data['rank'], 'viridis', vmin=1, vmax=vmax)
                sphere = vedo.Sphere(pos=p, r=radius, c=color)
            else:
                sphere = vedo.Sphere(pos=p, r=0.04, c='red5')
            
            sphere.name = node_data['info']
            current_graph_actors.append(sphere)

        for u, v in G.edges():
            line = vedo.Line(pos[u], pos[v], c='grey', lw=2, alpha=0.7)
            current_graph_actors.append(line)
        
        # Group all actors for this graph into one Assembly
        assembly = vedo.Assembly(current_graph_actors)
        assembly.pos(offset) # Move the entire assembly to its grid position
        all_assemblies.append(assembly)

    # Add non-rotating labels to a separate list
    for q_val in sorted(list(all_qubits)):
        col = q_val - min_qubits
        pos = (col * grid_spacing, grid_spacing * 0.6, 0)
        label = vedo.Text3D(f"Q={q_val}", pos=pos, s=label_scale_individual, justify='center')
        static_actors.append(label)

    for d_val in sorted(list(all_depths)):
        row = d_val - min_depth
        pos = (-grid_spacing * 0.6, -row * grid_spacing, 0)
        label = vedo.Text3D(f"D={d_val}", pos=pos, s=label_scale_individual, justify='center')
        label.rotate_z(90)
        static_actors.append(label)

    center_x = ((max_qubits - min_qubits) * grid_spacing) / 2.0
    center_y = (-(max_depth - min_depth) * grid_spacing) / 2.0
    qubit_axis_pos = (center_x, grid_spacing * 1.1, 0)
    qubit_axis_label = vedo.Text3D("QUBITS (WIDTH)", pos=qubit_axis_pos, s=label_scale_main, c='cyan', justify='center')
    static_actors.append(qubit_axis_label)
    depth_axis_pos = (-grid_spacing * 1.1, center_y, 0)
    depth_axis_label = vedo.Text3D("DEPTH", pos=depth_axis_pos, s=label_scale_main, c='cyan', justify='center')
    depth_axis_label.rotate_z(90)
    static_actors.append(depth_axis_label)

    print("Launching 3D visualization window...")
    print("\nHotkeys:")
    print("  - Right-click: Save a single high-resolution screenshot.")
    print("  - Press 'r':   Start a 360-degree rotation screenshot sequence.")
    
    plt = vedo.Plotter(title='Merged 3D Tensor Network Visualization', axes=1, bg='black')
    
    # Register callbacks
    plt.add_callback('right click', lambda ev: on_right_click(ev, plt))
    plt.add_callback('key press', lambda ev: on_key_press(ev, plt, all_assemblies))

    # Use splat operator (*) to unpack lists of actors
    plt.show(*all_assemblies, *static_actors, zoom='tight')


if __name__ == "__main__":
    main()
