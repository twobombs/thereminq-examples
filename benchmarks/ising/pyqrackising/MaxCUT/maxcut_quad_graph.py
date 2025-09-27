import sys
import re
import os
import glob
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
import pyvista as pv
from scipy.spatial import KDTree
import datetime
from PIL import Image

def parse_maxcut_worker(filepath):
    """Worker function to parse a single Max-Cut file."""
    basename = os.path.basename(filepath)
    match = re.search(r'macxut_n(\d+)_.*_s(\d+)(_gpu-on)?\.txt', basename)
    if not match:
        return None

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        set_a_match = re.search(r'Set A nodes: \[([\d, ]*)\]', content)
        set_b_match = re.search(r'Set B nodes: \[([\d, ]*)\]', content)

        if not set_a_match or not set_b_match:
            return None

        def parse_node_list(match_group):
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

def calculate_cut_edges_worker(row_tuple):
    """Worker function to find cut edges using a fast KDTree."""
    index, row = row_tuple
    n_nodes = row['nodes']
    seed = row['seed']
    set_a = set(row['set_a'])
    set_b = set(row['set_b'])
    
    DISTANCE_THRESHOLD = 0.75 
    
    local_edge_points = []
    local_edge_scalars = []

    # Regenerate coordinates
    np.random.seed(seed)
    coords = np.random.normal(size=(n_nodes, 3))
    norm = np.linalg.norm(coords, axis=1, keepdims=True)
    np.divide(coords, norm, out=coords, where=norm!=0)

    tree = KDTree(coords)
    pairs = tree.query_pairs(r=DISTANCE_THRESHOLD)
    
    for i, j in pairs:
        is_in_a_i = i in set_a
        is_in_b_j = j in set_b
        is_in_b_i = i in set_b
        is_in_a_j = j in set_a
        
        if (is_in_a_i and is_in_b_j) or (is_in_b_i and is_in_a_j):
            local_edge_points.append(coords[i])
            local_edge_points.append(coords[j])
            local_edge_scalars.append(i)
            local_edge_scalars.append(j)
                    
    return local_edge_points, local_edge_scalars

def process_image_for_ffmpeg(numpy_array):
    """Converts a NumPy array to a Pillow Image and crops to even dimensions."""
    try:
        img = Image.fromarray(numpy_array)
        width, height = img.size
        if width % 2 != 0 or height % 2 != 0:
            new_width = width - (width % 2)
            new_height = height - (height % 2)
            img = img.crop((0, 0, new_width, new_height))
        return img
    except Exception as e:
        print(f"WARNING: Could not process image for even dimensions. Reason: {e}")
        return None

def process_log_files(file_paths, desc_suffix=""):
    """Parses, aggregates, and calculates edges for a given list of log files."""
    if not file_paths:
        return None, None, None, 0

    print(f"\n--- Processing {desc_suffix} Data ({len(file_paths)} files) ---")
    
    # 1. Parallel Parsing
    results_list = []
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(parse_maxcut_worker, file_paths),
                           total=len(file_paths), ascii=True, desc=f"Parsing {desc_suffix} Files"):
            if result:
                results_list.append(result)

    if not results_list:
        print(f"Warning: No valid data could be parsed for {desc_suffix}.")
        return None, None, None, 0

    df = pd.DataFrame(results_list)
    print(f"Successfully parsed {len(df)} {desc_suffix} files.")
    max_nodes = df['nodes'].max() if not df.empty else 0

    # 2. Aggregating Coordinates
    all_coords_a, all_coords_b = [], []
    all_scalars_a, all_scalars_b = [], []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], ascii=True, desc=f"Aggregating {desc_suffix} Coords"):
        np.random.seed(row['seed'])
        coords = np.random.normal(size=(row['nodes'], 3))
        norm = np.linalg.norm(coords, axis=1, keepdims=True)
        np.divide(coords, norm, out=coords, where=norm!=0)
        if row['set_a']:
            all_coords_a.append(coords[row['set_a']])
            all_scalars_a.extend(row['set_a'])
        if row['set_b']:
            all_coords_b.append(coords[row['set_b']])
            all_scalars_b.extend(row['set_b'])
    
    poly_a = pv.PolyData(np.vstack(all_coords_a)) if all_coords_a else None
    if poly_a: poly_a['Node Index'] = all_scalars_a
    
    poly_b = pv.PolyData(np.vstack(all_coords_b)) if all_coords_b else None
    if poly_b: poly_b['Node Index'] = all_scalars_b

    # 3. Calculating Cut Edges
    edge_points_lists, edge_scalars_lists = [], []
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        for points, scalars in tqdm(pool.imap_unordered(calculate_cut_edges_worker, df.iterrows(), chunksize=1),
                                      total=len(df), ascii=True, desc=f"Finding {desc_suffix} Cut Edges"):
            if points:
                edge_points_lists.append(points)
                edge_scalars_lists.append(scalars)
    
    edge_points = [p for sublist in edge_points_lists for p in sublist]
    edge_scalars = [s for sublist in edge_scalars_lists for s in sublist]

    # 4. Deduplicate and Create Edge Mesh
    cut_edges_mesh = None
    if edge_points:
        seen_edges = set()
        unique_edge_points, unique_edge_scalars = [], []
        for i in range(0, len(edge_scalars), 2):
            edge_tuple = tuple(sorted((edge_scalars[i], edge_scalars[i+1])))
            if edge_tuple not in seen_edges:
                seen_edges.add(edge_tuple)
                unique_edge_points.extend([edge_points[i], edge_points[i+1]])
                unique_edge_scalars.extend([edge_scalars[i], edge_scalars[i+1]])
        
        print(f"Found {len(seen_edges)} unique cut edges for {desc_suffix}.")
        
        edge_points_array = np.array(unique_edge_points)
        num_edges = len(seen_edges)
        lines_cell_array = np.hstack([np.full((num_edges, 1), 2), np.arange(num_edges * 2).reshape(-1, 2)])
        cut_edges_mesh = pv.PolyData(edge_points_array, lines=lines_cell_array)
        cut_edges_mesh['Node Index'] = unique_edge_scalars
    else:
        print(f"No cut edges found for {desc_suffix}.")

    return poly_a, poly_b, cut_edges_mesh, max_nodes

# Global variable to store mesh data
global_meshes = {}

def open_merged_view_all():
    """Callback for 'm': Opens a new window with all four views merged into one."""
    global global_meshes
    print("\n'm' key pressed. Opening new window with all data merged...")
    
    merged_plotter = pv.Plotter(window_size=[1280, 960], off_screen=False)
    merged_plotter.set_background('black')
    
    color_limits = global_meshes['color_limits']
    render_opts = dict(cmap='viridis', clim=color_limits, point_size=3, opacity=0.05, render_points_as_spheres=True)
    edge_opts = dict(cmap='viridis', clim=color_limits, line_width=1, opacity=0.05)

    merged_plotter.add_text("Merged View: All Results", font_size=20, color='white')

    # Add all meshes to this single plotter
    if global_meshes['poly_a_cpu']:
        merged_plotter.add_mesh(global_meshes['poly_a_cpu'], scalars='Node Index', show_scalar_bar=False, **render_opts)
    if global_meshes['poly_b_cpu']:
        merged_plotter.add_mesh(global_meshes['poly_b_cpu'], scalars='Node Index', show_scalar_bar=False, **render_opts)
    if global_meshes['edges_cpu']:
        merged_plotter.add_mesh(global_meshes['edges_cpu'], scalars='Node Index', **edge_opts)

    if global_meshes['poly_a_gpu']:
        merged_plotter.add_mesh(global_meshes['poly_a_gpu'], scalars='Node Index', show_scalar_bar=False, **render_opts)
    if global_meshes['poly_b_gpu']:
        merged_plotter.add_mesh(global_meshes['poly_b_gpu'], scalars='Node Index', **render_opts)
    if global_meshes['edges_gpu']:
        merged_plotter.add_mesh(global_meshes['edges_gpu'], scalars='Node Index', **edge_opts)

    merged_plotter.add_scalar_bar(title="Node Index", color="white")
    merged_plotter.show()
    print("Full merged view window closed.")

def open_merged_view_by_type():
    """Callback for '2': Opens a new window with CPU and GPU data merged separately."""
    global global_meshes
    print("\n'2' key pressed. Opening new window with merged CPU and GPU views...")
    
    # 1. Create a new 1x2 plotter
    merged_plotter = pv.Plotter(shape=(1, 2), window_size=[1920, 960], off_screen=False)
    merged_plotter.set_background('black')

    color_limits = global_meshes['color_limits']
    render_opts = dict(cmap='viridis', clim=color_limits, point_size=3, opacity=0.05, render_points_as_spheres=True)
    edge_opts = dict(cmap='viridis', clim=color_limits, line_width=1, opacity=0.05)

    # 2. Subplot 0: Merged CPU data
    merged_plotter.subplot(0, 0)
    merged_plotter.add_text("Merged CPU Results (A & B)", font_size=15, color='white')
    if global_meshes['poly_a_cpu']:
        merged_plotter.add_mesh(global_meshes['poly_a_cpu'], scalars='Node Index', show_scalar_bar=False, **render_opts)
    if global_meshes['poly_b_cpu']:
        merged_plotter.add_mesh(global_meshes['poly_b_cpu'], scalars='Node Index', show_scalar_bar=False, **render_opts)
    if global_meshes['edges_cpu']:
        merged_plotter.add_mesh(global_meshes['edges_cpu'], scalars='Node Index', **edge_opts)

    # 3. Subplot 1: Merged GPU data
    merged_plotter.subplot(0, 1)
    merged_plotter.add_text("Merged GPU Results (A & B)", font_size=15, color='white')
    if global_meshes['poly_a_gpu']:
        merged_plotter.add_mesh(global_meshes['poly_a_gpu'], scalars='Node Index', show_scalar_bar=False, **render_opts)
    if global_meshes['poly_b_gpu']:
        merged_plotter.add_mesh(global_meshes['poly_b_gpu'], scalars='Node Index', **render_opts)
    if global_meshes['edges_gpu']:
        merged_plotter.add_mesh(global_meshes['edges_gpu'], scalars='Node Index', **edge_opts)
    
    merged_plotter.link_views()
    merged_plotter.show()
    print("Two-way merged view window closed.")

def save_high_res_screenshot(plotter_obj, *args):
    """Callback to save screenshot of a given plotter."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"maxcut_visualization_{timestamp}.png"
        print(f"\nSaving screenshot to: {filename}")
        image_array = plotter_obj.screenshot(scale=3, return_img=True)
        processed_image = process_image_for_ffmpeg(image_array)
        if processed_image:
            processed_image.save(filename)
            print(f"Screenshot successfully saved.")
    except Exception as e:
        print(f"ERROR: Failed to save screenshot. Reason: {e}")

if __name__ == "__main__":
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print(f"Error: The '{results_dir}' directory does not exist.")
        sys.exit(1)

    all_files = glob.glob(os.path.join(results_dir, 'macxut_*.txt'))
    gpu_files = [f for f in all_files if f.endswith('_gpu-on.txt')]
    cpu_files = [f for f in all_files if not f.endswith('_gpu-on.txt')]
    
    if not cpu_files and not gpu_files:
        print(f"Error: No 'macxut_*.txt' files found in '{results_dir}'.")
        sys.exit(1)

    poly_a_cpu, poly_b_cpu, edges_cpu, max_nodes_cpu = process_log_files(cpu_files, "CPU")
    poly_a_gpu, poly_b_gpu, edges_gpu, max_nodes_gpu = process_log_files(gpu_files, "GPU")

    max_nodes_combined = max(max_nodes_cpu, max_nodes_gpu)

    # Store all processed meshes in a global dictionary
    global_meshes = {
        'poly_a_cpu': poly_a_cpu, 'poly_b_cpu': poly_b_cpu, 'edges_cpu': edges_cpu,
        'poly_a_gpu': poly_a_gpu, 'poly_b_gpu': poly_b_gpu, 'edges_gpu': edges_gpu,
        'color_limits': [0, max_nodes_combined]
    }

    # --- Setup the MAIN 2x2 plotter ---
    print("\nInitializing main 2x2 PyVista plotter...")
    plotter_2x2 = pv.Plotter(shape=(2, 2), window_size=[1920, 1080], off_screen=False)
    
    color_limits = global_meshes['color_limits']
    render_opts = dict(cmap='viridis', clim=color_limits, point_size=3, opacity=0.05, render_points_as_spheres=True)
    edge_opts = dict(cmap='viridis', clim=color_limits, line_width=1, opacity=0.05)

    # Plot 1: CPU Set A
    plotter_2x2.subplot(0, 0)
    if global_meshes['poly_a_cpu']:
        plotter_2x2.add_mesh(global_meshes['poly_a_cpu'], scalars='Node Index', show_scalar_bar=False, **render_opts)
        if global_meshes['edges_cpu']: plotter_2x2.add_mesh(global_meshes['edges_cpu'], scalars='Node Index', **edge_opts)
    plotter_2x2.add_text("CPU Results: Set A", font_size=15, color='white')

    # Plot 2: CPU Set B
    plotter_2x2.subplot(0, 1)
    if global_meshes['poly_b_cpu']:
        plotter_2x2.add_mesh(global_meshes['poly_b_cpu'], scalars='Node Index', show_scalar_bar=False, **render_opts)
        if global_meshes['edges_cpu']: plotter_2x2.add_mesh(global_meshes['edges_cpu'], scalars='Node Index', **edge_opts)
    plotter_2x2.add_text("CPU Results: Set B", font_size=15, color='white')

    # Plot 3: GPU Set A
    plotter_2x2.subplot(1, 0)
    if global_meshes['poly_a_gpu']:
        plotter_2x2.add_mesh(global_meshes['poly_a_gpu'], scalars='Node Index', show_scalar_bar=False, **render_opts)
        if global_meshes['edges_gpu']: plotter_2x2.add_mesh(global_meshes['edges_gpu'], scalars='Node Index', **edge_opts)
    plotter_2x2.add_text("GPU Results: Set A", font_size=15, color='white')

    # Plot 4: GPU Set B
    plotter_2x2.subplot(1, 1)
    if global_meshes['poly_b_gpu']:
        plotter_2x2.add_mesh(global_meshes['poly_b_gpu'], scalars='Node Index', **render_opts)
        if global_meshes['edges_gpu']: plotter_2x2.add_mesh(global_meshes['edges_gpu'], scalars='Node Index', **edge_opts)
    plotter_2x2.add_text("GPU Results: Set B", font_size=15, color='white')
    
    plotter_2x2.link_views()

    # --- Add interactive callbacks ---
    plotter_2x2.add_key_event('m', open_merged_view_all)
    plotter_2x2.add_key_event('2', open_merged_view_by_type)
    
    # Use a lambda to pass the plotter object to the save function
    plotter_2x2.track_click_position(callback=lambda *args: save_high_res_screenshot(plotter_2x2, *args), side='right')
    
    print("\nShowing main plot window. Close to exit.")
    print("- Press 'm' to merge all data into one new window.")
    print("- Press '2' to merge CPU/GPU data in a new two-panel window.")
    print("- Right-click to save a screenshot of this 2x2 view.")
    plotter_2x2.show()
    print("Main PyVista plot window closed.")
