import sys
import re
import os
import glob
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
import pyvista as pv
from datetime import datetime

def parse_file_worker(filepath):
    """Worker function to parse a single file and return a dictionary."""
    basename = os.path.basename(filepath)
    # Regex to match the filename format
    match = re.search(r'tspmontecarlo_n(\d+)_ms(\d+)_kn(\d+)_s(\d+)\.txt', basename)
    if not match: return None
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        path_match = re.search(r'Path: \[([\d, ]+)\]', content)
        if not path_match: return None
        path_str = path_match.group(1)
        
        # Dictionary keys to reflect parameters
        return {
            'nodes': int(match.group(1)),
            'samples': int(match.group(2)),
            'neighbors': int(match.group(3)),
            'seed': int(match.group(4)),
            'path': [int(node) for node in path_str.split(', ')]
        }
    except Exception:
        return None

if __name__ == "__main__":
    # --- 1. SETUP AND PARALLEL FILE PARSING ---
    results_dir = 'results'
    # Glob pattern to find files
    file_paths = glob.glob(os.path.join(results_dir, 'tspmontecarlo_*.txt'))
    if not file_paths:
        print(f"Error: No 'tspmontecarlo_*.txt' files found in '{results_dir}'.")
        sys.exit(1)

    print(f"Parsing {len(file_paths)} files...")
    results_list = []
    with multiprocessing.Pool() as pool:
        for result in tqdm(pool.imap_unordered(parse_file_worker, file_paths), 
                           total=len(file_paths), ascii=True, desc="Parsing Files"):
            if result:
                results_list.append(result)

    df = pd.DataFrame(results_list)
    if df.empty:
        print("Error: No valid data could be parsed from the files.")
        sys.exit(1)

    # --- REVISED SNIPPET ---
    # Calculate and print the grand total of all nodes
    total_nodes_processed = df['nodes'].sum()
    print(f"\nData loaded for {len(df)} simulations.")
    print(f"   Grand total of all nodes processed across all files: {total_nodes_processed}")
    # --- END REVISED SNIPPET ---

    # --- 2. PREPARE DATA FOR PLOTTING ---
    print("\nPreparing data for stacked visualization...")
    
    # Sort by 'nodes' to ensure the z-stacking is ordered logically
    df_sorted = df.sort_values(by='nodes').reset_index(drop=True)
    
    # Use the parameters from the run with the most nodes to create base coordinates
    base_params = df.loc[df['nodes'].idxmax()]
    base_n_nodes = base_params['nodes']
    base_seed = base_params['seed']
    
    np.random.seed(int(base_seed))
    coords_2d = np.random.rand(int(base_n_nodes), 2)
    
    scaling_factor = 2
    coords_2d *= scaling_factor

    # --- 3. PYVISTA INTERACTIVE VISUALIZATION ---
    print("Generating the stacked 3D plot with PyVista...")

    pv.set_plot_theme("dark") 
    plotter = pv.Plotter(window_size=[1200, 800])

    # --- INTERACTIVE CALLBACKS ---
    def screenshot_callback(obj, event):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"visualizations/tsp_screenshot_{timestamp}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plotter.screenshot(filename, scale=3) 
        print(f"\nScreenshot saved to '{filename}'")

    plotter.iren.add_observer("RightButtonPressEvent", screenshot_callback)

    # --- PLOTTING LOGIC ---
    color_limits = [df['nodes'].min(), df['nodes'].max()]
    z_spacing = 0.005 / 2

    for index, row in tqdm(df_sorted.iterrows(), 
                           total=len(df_sorted), ascii=True, desc="Building Plot"):
        path = row['path']
        # Safety check in case a path has a node index higher than our base coordinate set
        if max(path) >= base_n_nodes:
            continue
        
        path_coords_2d = coords_2d[path, :]
        z_level = index * z_spacing
        path_coords_3d = np.hstack([path_coords_2d, np.full((len(path), 1), z_level)])
        
        line = pv.lines_from_points(path_coords_3d)
        scalar_values = np.full(line.n_points, row['nodes'])
        
        plotter.add_mesh(
            line,
            scalars=scalar_values,
            cmap='viridis_r',
            clim=color_limits,
            line_width=1,
            show_scalar_bar=False,
            opacity=0.01
        )

    plotter.add_scalar_bar(title="Number of Nodes", fmt="%.0f")
    
    plotter.camera_position = 'xy'
    plotter.camera.elevation = 60
    plotter.camera.zoom(1.2)
    
    plotter.show_bounds()

    print("\nShowing plot window. Right-click to save a screenshot. Close the window to exit.")
    plotter.show()

    print("PyVista plot window closed.")
