import sys
import re
import os
import glob
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
import pyvista as pv
from scipy.spatial.distance import pdist, squareform

def parse_maxcut_worker(filepath):
    """Worker function to parse a single Max-Cut file."""
    basename = os.path.basename(filepath)
    match = re.search(r'macxut_n(\d+)_.*_s(\d+)\.txt', basename)
    if not match:
        return None

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        set_a_match = re.search(r'Set A nodes: \[([\d, ]+)\]', content)
        set_b_match = re.search(r'Set B nodes: \[([\d, ]+)\]', content)

        if not set_a_match or not set_b_match:
            return None

        def parse_node_list(match_group):
            return [int(node) for node in match_group.split(', ')]

        return {
            'nodes': int(match.group(1)),
            'seed': int(match.group(2)),
            'set_a': parse_node_list(set_a_match.group(1)),
            'set_b': parse_node_list(set_b_match.group(1)),
            'file': basename
        }
    except Exception as e:
        print(f"Warning: Could not parse {basename}. Error: {e}")
        return None

if __name__ == "__main__":
    # --- 1. SETUP AND PARALLEL FILE PARSING ---
    results_dir = 'results'
    file_paths = glob.glob(os.path.join(results_dir, 'macxut_*.txt'))
    if not file_paths:
        print(f"Error: No 'macxut_*.txt' files found in '{results_dir}'.")
        sys.exit(1)

    print(f"Found {len(file_paths)} result files. Parsing...")
    results_list = []
    with multiprocessing.Pool() as pool:
        for result in tqdm(pool.imap_unordered(parse_maxcut_worker, file_paths),
                           total=len(file_paths), ascii=True, desc="Parsing Files"):
            if result:
                results_list.append(result)

    if not results_list:
        print("Error: No valid Max-Cut data could be parsed from the files.")
        sys.exit(1)

    df = pd.DataFrame(results_list)
    print(f"\nSuccessfully parsed {len(df)} files.")

    # --- 2. PREPARE DATA FOR PLOTTING ---
    data_to_plot = df.loc[df['nodes'].idxmax()]
    print(f"Visualizing results from: {data_to_plot['file']}\n")

    n_nodes = data_to_plot['nodes']
    seed = data_to_plot['seed']
    set_a_nodes = data_to_plot['set_a']
    set_b_nodes = data_to_plot['set_b']

    np.random.seed(seed)
    coords = np.random.normal(size=(n_nodes, 3))
    coords /= np.linalg.norm(coords, axis=1, keepdims=True)

    coords_a = coords[set_a_nodes]
    coords_b = coords[set_b_nodes]
    
    # Set the global theme's font color for older PyVista versions
    pv.global_theme.font.color = 'white'

    # --- 3. PYVISTA SIDE-BY-SIDE VISUALIZATION ---
    print("Generating the side-by-side 3D plot with PyVista...")
    plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 800])
    plotter.set_background('black')
    
    color_limits = [0, n_nodes]

    # --- Plot for Set A ---
    plotter.subplot(0, 0)
    plotter.add_text("Set A", font_size=20, color='white')
    plotter.add_mesh(
        pv.PolyData(coords_a),
        scalars=set_a_nodes,
        cmap='viridis',
        clim=color_limits,
        point_size=5,             # CHANGED
        opacity=0.7,              # ADDED
        render_points_as_spheres=True,
        label="Set A Nodes",
        show_scalar_bar=False
    )
    plotter.show_bounds(location='outer', all_edges=True, color='white')

    # --- Plot for Set B ---
    plotter.subplot(0, 1)
    plotter.add_text("Set B", font_size=20, color='white')
    plotter.add_mesh(
        pv.PolyData(coords_b),
        scalars=set_b_nodes,
        cmap='viridis',
        clim=color_limits,
        point_size=5,             # CHANGED
        opacity=0.7,              # ADDED
        render_points_as_spheres=True,
        label="Set B Nodes"
    )
    plotter.add_scalar_bar(
        title="Node Index",
        fmt="%.0f"
    )
    plotter.show_bounds(location='outer', all_edges=True, color='white')

    plotter.link_views()
    print("\nShowing plot window. Close the window to exit.")
    plotter.show()
    print("PyVista plot window closed.")
