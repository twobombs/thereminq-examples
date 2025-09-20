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
from scipy.spatial import KDTree
import datetime
from PIL import Image

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

    # --- OPTIMIZATION: Use a KDTree for fast neighbor search ---
    tree = KDTree(coords)
    pairs = tree.query_pairs(r=DISTANCE_THRESHOLD)
    
    # Find edges crossing from set A to set B
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


if __name__ == "__main__":
    # --- 1. SETUP AND PARALLEL FILE PARSING ---
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print(f"Error: The '{results_dir}' directory does not exist.")
        sys.exit(1)

    file_paths = glob.glob(os.path.join(results_dir, 'macxut_*.txt'))
    if not file_paths:
        print(f"Error: No 'macxut_*.txt' files found in '{results_dir}'.")
        sys.exit(1)

    print(f"Found {len(file_paths)} result files. Parsing in parallel...")
    results_list = []
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(parse_maxcut_worker, file_paths),
                           total=len(file_paths), ascii=True, desc="Parsing Files"):
            if result:
                results_list.append(result)

    if not results_list:
        print("Error: No valid Max-Cut data could be parsed from the files.")
        sys.exit(1)

    df = pd.DataFrame(results_list)
    print(f"\nSuccessfully parsed {len(df)} files.")

    # --- 2. AGGREGATE NODE COORDINATES ---
    print("Aggregating node coordinates for all parsed files...")
    all_coords_a, all_coords_b = [], []
    all_scalars_a, all_scalars_b = [], []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], ascii=True, desc="Aggregating Coords"):
        n_nodes, seed = row['nodes'], row['seed']
        set_a_nodes, set_b_nodes = row['set_a'], row['set_b']
        np.random.seed(seed)
        coords = np.random.normal(size=(n_nodes, 3))
        norm = np.linalg.norm(coords, axis=1, keepdims=True)
        np.divide(coords, norm, out=coords, where=norm!=0)

        if set_a_nodes:
            all_coords_a.append(coords[set_a_nodes])
            all_scalars_a.extend(set_a_nodes)
        if set_b_nodes:
            all_coords_b.append(coords[set_b_nodes])
            all_scalars_b.extend(set_b_nodes)

    master_coords_a = np.vstack(all_coords_a)
    master_coords_b = np.vstack(all_coords_b)

    # --- 3. IDENTIFY AND PREPARE CUT EDGES (PARALLEL) ---
    print("\nIdentifying cut edges in parallel...")
    edge_points_lists = []
    edge_scalars_lists = []
    num_workers = os.cpu_count() * 4
    print(f"Using {num_workers} worker processes for edge calculation...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        for points, scalars in tqdm(pool.imap_unordered(calculate_cut_edges_worker, df.iterrows(), chunksize=1),
                           total=len(df), ascii=True, desc="Finding Cut Edges"):
            if points:
                edge_points_lists.append(points)
                edge_scalars_lists.append(scalars)
    
    edge_points = [point for sublist in edge_points_lists for point in sublist]
    edge_scalars = [scalar for sublist in edge_scalars_lists for scalar in sublist]
    
    # --- 4. PYVISTA VISUALIZATION ---
    print("\nGenerating the 3D plot...")
    plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 800], off_screen=False)
    plotter.set_background('black')
    light = pv.Light(light_type='camera light', intensity=1.5)
    plotter.add_light(light)

    if edge_points:
        edge_points_array = np.array(edge_points)
        num_edges = len(edge_points_array) // 2
        lines_cell_array = np.hstack([np.full((num_edges, 1), 2), np.arange(num_edges * 2).reshape(-1, 2)])
        cut_edges_mesh = pv.PolyData(edge_points_array, lines=lines_cell_array)
        cut_edges_mesh['Node Index'] = edge_scalars
        print(f"Found {num_edges} unique cut edges to display.")
    else:
        cut_edges_mesh = None
        print("No cut edges found with the current distance threshold.")

    poly_a = pv.PolyData(master_coords_a)
    poly_a['Node Index'] = all_scalars_a
    poly_b = pv.PolyData(master_coords_b)
    poly_b['Node Index'] = all_scalars_b
    color_limits = [0, df['nodes'].max()]
    
    # Plot for Set A
    plotter.subplot(0, 0)
    plotter.add_mesh(poly_a, scalars='Node Index', cmap='viridis', clim=color_limits,
                     point_size=10, opacity=0.8, render_points_as_spheres=True,
                     show_scalar_bar=False)
    if cut_edges_mesh:
        plotter.add_mesh(cut_edges_mesh, scalars='Node Index', cmap='viridis',
                         clim=color_limits, line_width=2, opacity=0.5)
    plotter.add_text("Set A", font_size=20, color='white')

    # Plot for Set B
    plotter.subplot(0, 1)
    plotter.add_mesh(poly_b, scalars='Node Index', cmap='viridis', clim=color_limits,
                     point_size=10, opacity=0.8, render_points_as_spheres=True)
    if cut_edges_mesh:
        plotter.add_mesh(cut_edges_mesh, scalars='Node Index', cmap='viridis',
                         clim=color_limits, line_width=2, opacity=0.5)
    plotter.add_text("Set B", font_size=20, color='white')
    
    # Manually add the scalar bar as a final step
    plotter.add_scalar_bar(
        title="Node Index",
        fmt="%.0f",
        color="white",
        title_font_size=20,
        label_font_size=16,
        font_family="arial"
    )
    
    plotter.link_views()

    # --- INTERACTIVE CALLBACKS ---
    def save_high_res_screenshot(*args):
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"maxcut_visualization_{timestamp}.png"
            print(f"\nSaving screenshot to: {filename}")
            image_array = plotter.screenshot(scale=3, return_img=True)
            processed_image = process_image_for_ffmpeg(image_array)
            if processed_image:
                processed_image.save(filename)
                print(f"Screenshot successfully saved.")
        except Exception as e:
            print(f"ERROR: Failed to save screenshot. Reason: {e}")

    def record_rotation_animation():
        plotter.reset_camera(render=False)
        print("\n'r' key pressed! Starting animation recording...")
        n_steps = 360
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        animation_dir = f"animation_{timestamp}"
        os.makedirs(animation_dir, exist_ok=True)
        print(f"Saving {n_steps} frames to: {animation_dir}")
        
        for i in tqdm(range(n_steps), desc="Recording Frames", ascii=True):
            plotter.camera.azimuth += 360.0 / n_steps
            plotter.render()
            filename = os.path.join(animation_dir, f"frame_{i:03d}.png")
            image_array = plotter.screenshot(scale=3, return_img=True)
            processed_image = process_image_for_ffmpeg(image_array)
            if processed_image:
                processed_image.save(filename)
        print("Animation recording complete.")

    plotter.track_click_position(callback=save_high_res_screenshot, side='right')
    plotter.add_key_event('r', record_rotation_animation)

    print("\nShowing plot window. Close to exit.")
    print("- Right-click to save a high-resolution screenshot.")
    print("- Press 'r' to record a 360-degree animation.")
    plotter.show()
    print("PyVista plot window closed.")
