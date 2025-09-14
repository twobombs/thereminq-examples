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
    match = re.search(r'tsp_n(\d+)_q(\d+)_cq(\d+)_i(\d+)_s(\d+)\.txt', basename)
    if not match: return None
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        path_match = re.search(r'Path: \[([\d, ]+)\]', content)
        if not path_match: return None
        path_str = path_match.group(1)
        
        return {
            'nodes': int(match.group(1)),
            'quality': int(match.group(2)),
            'correction_quality': int(match.group(3)),
            'seed': int(match.group(5)),
            'path': [int(node) for node in path_str.split(', ')]
        }
    except Exception:
        return None

if __name__ == "__main__":
    # --- 1. SETUP AND PARALLEL FILE PARSING ---
    results_dir = 'results'
    file_paths = glob.glob(os.path.join(results_dir, 'tsp_*.txt'))
    if not file_paths:
        print(f"Error: No '.txt' files found in '{results_dir}'.")
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

    # --- 2. PREPARE DATA FOR PLOTTING ---
    print("Preparing data for stacked visualization...")
    
    df_sorted = df.sort_values(by='nodes').reset_index(drop=True)
    base_params = df.loc[df['nodes'].idxmax()]
    base_n_nodes = base_params['nodes']
    base_seed = base_params['seed']
    
    np.random.seed(base_seed)
    coords_2d = np.random.rand(base_n_nodes, 2)
    
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
    z_spacing = 0.005 / 4

    for index, row in tqdm(df_sorted.iterrows(), 
                           total=len(df_sorted), ascii=True, desc="Building Plot"):
        path = row['path']
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
            line_width=1, # FIX: Changed from 3 to 1 for thinner lines
            show_scalar_bar=False,
            opacity=0.5
        )

    plotter.add_scalar_bar(title="Number of Nodes", fmt="%.0f")
    
    plotter.camera_position = 'xy'
    plotter.camera.elevation = 60
    plotter.camera.zoom(1.2)
    
    plotter.show_bounds()

    print("\nShowing plot window. Right-click to save a screenshot. Close the window to exit.")
    plotter.show()

    print("PyVista plot window closed.")
