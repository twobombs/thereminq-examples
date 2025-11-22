#!/usr/bin/env python3
import glob
import re
import math
import argparse
import os
import sys
import traceback
import numpy as np
import vedo
from scipy.spatial import Delaunay

# --- Configuration defaults ---
DEFAULT_LOG_DIR = "labs_logs"

def parse_logs(log_directory):
    """
    Parses log files. Returns a list of tuples: (n, lambda, avg_time, avg_cut).
    
    - If cut < 1.0 (failed/noise), force it to 1.0.
    - On the log10 scale, log10(1.0) = 0.
    - This ensures failed runs appear as "flat" points at zero height.
    """
    search_path = os.path.join(log_directory, "labs_run_N_*_L_*.log")
    log_files = glob.glob(search_path)
    
    if not log_files:
        print(f"Error: No matching log files found in '{log_directory}'")
        return None

    print(f"Found {len(log_files)} log files. Parsing...")
    
    raw_data = {}
    
    # Regex patterns
    filename_regex = re.compile(r"labs_run_N_(\d+)_L_([\d.]+)\.log")
    time_regex = re.compile(r"real\s+(\d+)m([\d.]+)s")
    cut_regex = re.compile(r"Best cut.*:\s*(-?[\d.eE+-]+)")

    success_count = 0
    forced_zero_count = 0
    
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

                # --- LOGIC: Force failures to floor ---
                if cut_value < 1.0:
                    cut_value = 1.0
                    forced_zero_count += 1
                # --------------------
                
                key = (n, l_val)
                if key not in raw_data:
                    raw_data[key] = {'time': [], 'cut': []}
                
                raw_data[key]['time'].append(total_time)
                raw_data[key]['cut'].append(cut_value)
                success_count += 1
                
        except Exception:
            pass

    print(f"Successfully parsed {success_count} files.")
    if forced_zero_count > 0:
        print(f"Note: {forced_zero_count} files had invalid cuts and were forced to Zero height (Cut=1.0).")
    print(f"Unique parameter configurations found: {len(raw_data)}")

    # Flatten and Average duplicates
    cleaned_data = []
    for (n, l_val), values in raw_data.items():
        avg_time = sum(values['time']) / len(values['time'])
        avg_cut = sum(values['cut']) / len(values['cut']) 
        cleaned_data.append((n, l_val, avg_time, avg_cut))

    # Sort by N then Lambda
    cleaned_data.sort(key=lambda x: (x[0], x[1]))
    return cleaned_data

def plot_3d_surface(data):
    print("Generating 3D Surface data...")
    
    coords = []
    xy_points = [] 
    time_scalars = []

    for n, l_val, time, cut in data:
        # cut is guaranteed >= 1.0 now, so z_val >= 0
        z_val = math.log10(cut)
        # Safety check for time
        t_val = math.log(time) if time > 1e-9 else -5
        
        coords.append([n, l_val, z_val])
        xy_points.append([n, l_val])
        time_scalars.append(t_val)

    coords_np = np.array(coords)
    xy_np = np.array(xy_points)
    time_np = np.array(time_scalars)

    # --- TRIANGULATION (SciPy) ---
    print("Triangulating points using SciPy...")
    if len(xy_np) < 3:
        print("Not enough points to triangulate surface.")
        return

    tri = Delaunay(xy_np)
    faces = tri.simplices
    
    print(f"Constructing Mesh ({len(faces)} faces)...")
    surf = vedo.Mesh([coords_np, faces])
    
    # Visuals: Color by Time (INVERTED: Bright=Fast, Dark=Slow)
    surf.cmap('viridis_r', time_np).alpha(0.9)
    
    # --- TRANSPARENT LINES FIX ---
    # 1. Set solid black color
    surf.lw(0.5).lc("black")
    
    # 2. Access the underlying VTK property via .properties
    try:
        surf.properties.SetEdgeOpacity(0.2)
    except AttributeError:
        # Fallback for very old vedo versions
        surf.GetProperty().SetEdgeOpacity(0.2)
    # -----------------------------

    # Plotter Setup
    plotter = vedo.Plotter(axes=0, bg='black', title="LABS Landscape")
    
    text_str = (
        "Landscape of Optimization\n"
        "Height: log10(Cut Value)\n"
        "Color: ln(Time in seconds)"
    )
    
    plotter.add(vedo.Text2D(text_str, pos="top-left", c="white", s=0.8))

    # Axes Setup
    axes = vedo.Axes(
        surf,
        xtitle='N (Seq Length)',
        ytitle='Lambda',
        ztitle='log10(Cut)',
        c='white'
    )
    
    surf.add_scalarbar(title="ln(Time)", c='white')

    print("Displaying 3D Plot (Close window to continue)...")
    plotter.show(surf, axes, viewup='z', interactive=True)
    plotter.close()

def main():
    parser = argparse.ArgumentParser(description="Plot LABS Logs (3D Only)")
    parser.add_argument("dir", nargs="?", default=DEFAULT_LOG_DIR, help="Directory containing log files")
    args = parser.parse_args()

    all_data = parse_logs(args.dir)
    
    if not all_data:
        sys.exit(1)

    # Attempt 3D Plot
    try:
        plot_3d_surface(all_data)
    except Exception:
        print("\nCRITICAL ERROR in 3D Plotting:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
