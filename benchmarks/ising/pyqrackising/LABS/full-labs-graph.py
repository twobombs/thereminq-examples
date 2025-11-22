import glob
import re
import math
import vedo
import vedo.pyplot as plt
import numpy as np
import os
import sys
import traceback
from scipy.spatial import Delaunay  # <--- The Fix: Standard scientific library

# --- Configuration ---
LOG_DIR = "labs_logs"
# --- End Configuration ---


def parse_logs(log_directory):
    """
    Parses log files with format: labs_run_N_{N}_L_{LAMBDA}.log
    """
    search_path = os.path.join(log_directory, "labs_run_N_*_L_*.log")
    log_files = glob.glob(search_path)
    
    if not log_files:
        print(f"Error: No matching log files found in '{log_directory}'")
        return []

    print(f"Found {len(log_files)} log files. Parsing...")
    
    data = []
    
    filename_regex = re.compile(r"labs_run_N_(\d+)_L_([\d.]+)\.log")
    time_regex = re.compile(r"real\s+(\d+)m([\d.]+)s")
    cut_regex = re.compile(r"Best cut.*:\s*([\d.e+-]+)")

    success_count = 0
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()

            name_match = filename_regex.search(log_file)
            time_match = time_regex.search(content)
            cut_match = cut_regex.search(content)

            if name_match and time_match and cut_match:
                n = int(name_match.group(1))
                l_val = float(name_match.group(2))
                
                minutes = float(time_match.group(1))
                seconds = float(time_match.group(2))
                total_time = (minutes * 60) + seconds
                
                cut_value = float(cut_match.group(1))
                
                data.append((n, l_val, total_time, cut_value))
                success_count += 1
        except Exception as e:
            pass

    print(f"Successfully parsed {success_count} out of {len(log_files)} files.")
    data.sort(key=lambda x: (x[0], x[1]))
    return data


def plot_3d_surface(data):
    """
    Visualizes the data as a 3D Surface using SciPy for robust triangulation.
    """
    print("Generating 3D Surface data...")
    if not data:
        raise ValueError("No data available to plot.")

    coords = []
    xy_points = [] # For triangulation (N, Lambda)
    time_scalars = []

    for n, l_val, time, cut in data:
        z_val = math.log10(cut) if cut > 0 else 0
        t_val = math.log(time) if time > 0 else 0
        
        coords.append([n, l_val, z_val])
        xy_points.append([n, l_val])
        time_scalars.append(t_val)

    coords_np = np.array(coords)
    xy_np = np.array(xy_points)
    time_np = np.array(time_scalars)

    # --- SCIPY FIX ---
    # 1. Use SciPy to calculate the triangles (faces) based on X,Y coordinates
    print("Triangulating points using SciPy...")
    tri = Delaunay(xy_np)
    faces = tri.simplices
    
    # 2. Create the Vedo Mesh manually using vertices and faces
    print("Constructing Mesh...")
    surf = vedo.Mesh([coords_np, faces])
    # -----------------

    # Color the surface by Time
    surf.cmap('viridis', time_np).alpha(0.9)
    surf.lw(1).lc('black') # Add wireframe lines

    plotter = vedo.Plotter(axes=0, bg='k', title="LABS Landscape")
    
    plotter.add(vedo.Text2D(
        "Surface Height = log10(Cut Value)\nColor = ln(Time)", 
        pos="top-center", c="white", s=1.0
    ))

    axes = vedo.Axes(
        surf,
        xtitle='N (Seq Length)',
        ytitle='Lambda',
        ztitle='log10(Cut)',
        c='white'
    )
    
    surf.add_scalarbar(title="ln(Time)", c='white')

    print("Displaying 3D Plot (Close window to continue)...")
    plotter.show(surf, axes, viewup='z')
    plotter.close()


def plot_2d_slices(data):
    """
    Plots N vs Cut Value, grouped by Lambda.
    """
    if not data:
        return

    print("Generating 2D Slices...")
    grouped = {}
    for n, l_val, time, cut in data:
        if l_val not in grouped:
            grouped[l_val] = ([], [])
        grouped[l_val][0].append(n)
        grouped[l_val][1].append(math.log10(cut) if cut > 0 else 0)

    plotter = vedo.Plotter(bg='k', title="2D Slices by Lambda")
    
    lines = []
    sorted_lambdas = sorted(grouped.keys())
    
    colors = ["red", "green", "blue", "yellow", "cyan", "magenta", "white", "orange"]

    for i, l_val in enumerate(sorted_lambdas):
        n_vals, cut_vals = grouped[l_val]
        c = colors[i % len(colors)]
        
        line = vedo.pyplot.plot(
            n_vals, cut_vals, 
            "-o", c=c
        )
        lines.append(line)

    if lines:
        lines[0].xtitle = "N"
        lines[0].ytitle = "log10(Cut Value)"
        
        plotter.add(vedo.Text2D("Cut Value vs N (By Lambda)", pos="top-center", c="white"))
        plotter.show(lines, zoom=1.2)
        plotter.close()


def main():
    all_data = parse_logs(LOG_DIR)
    
    if not all_data:
        print("Exiting: No data found.")
        sys.exit(1)

    # 1. Attempt 3D Plot
    try:
        print("1. Attempting 3D Surface Plot...")
        plot_3d_surface(all_data)
    except Exception:
        print("\n" + "!"*60)
        print("CRITICAL ERROR in 3D Plotting.")
        traceback.print_exc()
        print("!"*60)
        sys.exit(1)
    
    print("-" * 30)
    
    # 2. Attempt 2D Plot
    try:
        print("2. Attempting 2D Slice Plots...")
        plot_2d_slices(all_data)
    except Exception:
        print("\nCRITICAL ERROR in 2D Plotting:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
