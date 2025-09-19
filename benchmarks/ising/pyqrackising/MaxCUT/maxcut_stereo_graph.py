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
            # Handles empty lists that might occur
            if not match_group.strip():
                return []
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

    # --- 2. PREPARE DATA FOR PLOTTING (MODIFIED TO PLOT ALL FILES) ---
    print("Aggregating coordinates for all parsed files. This may take a moment...")

    all_coords_a = []
    all_coords_b = []
    all_scalars_a = []
    all_scalars_b = []

    # Loop through every row in the DataFrame
    for index, row in tqdm(df.iterrows(), total=df.shape[0], ascii=True, desc="Processing Files"):
        n_nodes = row['nodes']
        seed = row['seed']
        set_a_nodes = row['set_a']
        set_b_nodes = row['set_b']

        # Generate coordinates for this specific graph
        np.random.seed(seed)
        coords = np.random.normal(size=(n_nodes, 3))
        # Avoid division by zero if a node is at the origin (highly unlikely)
        norm = np.linalg.norm(coords, axis=1, keepdims=True)
        np.divide(coords, norm, out=coords, where=norm!=0)


        # Append coordinates and scalars for Set A
        if set_a_nodes: # Ensure the list is not empty
            all_coords_a.append(coords[set_a_nodes])
            all_scalars_a.extend(set_a_nodes)

        # Append coordinates and scalars for Set B
        if set_b_nodes: # Ensure the list is not empty
            all_coords_b.append(coords[set_b_nodes])
            all_scalars_b.extend(set_b_nodes)

    # Combine the lists of arrays into single large arrays
    master_coords_a = np.vstack(all_coords_a)
    master_coords_b = np.vstack(all_coords_b)

    print(f"\nTotal points for Set A: {len(master_coords_a)}")
    print(f"Total points for Set B: {len(master_coords_b)}")

    # Set the global theme's font color for older PyVista versions
    pv.global_theme.font.color = 'white'

    # --- 3. PYVISTA SIDE-BY-SIDE VISUALIZATION (MODIFIED TO PLOT ALL FILES) ---
    print("Generating the side-by-side 3D plot with PyVista...")
    plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 800])
    plotter.set_background('black')

    # Determine a common color limit for the scalar bar
    max_node_index = df['nodes'].max()
    color_limits = [0, max_node_index]

    # --- Plot for Set A (All Files) ---
    plotter.subplot(0, 0)
    plotter.add_text("Set A (All Files)", font_size=20, color='white')
    plotter.add_mesh(
        pv.PolyData(master_coords_a),
        scalars=all_scalars_a,
        cmap='viridis',
        clim=color_limits,
        point_size=2,  # Reduced point size for clarity
        opacity=0.7,
        render_points_as_spheres=True,
        show_scalar_bar=False
    )
    plotter.show_bounds(location='outer', all_edges=True, color='white')

    # --- Plot for Set B (All Files) ---
    plotter.subplot(0, 1)
    plotter.add_text("Set B (All Files)", font_size=20, color='white')
    plotter.add_mesh(
        pv.PolyData(master_coords_b),
        scalars=all_scalars_b,
        cmap='viridis',
        clim=color_limits,
        point_size=2,  # Reduced point size for clarity
        opacity=0.7,
        render_points_as_spheres=True
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
