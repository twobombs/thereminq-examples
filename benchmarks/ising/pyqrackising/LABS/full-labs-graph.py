#!/usr/bin/env python3
import re
import math
import argparse
import sys
import traceback
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import vedo
from scipy.spatial import Delaunay

# --- Configuration defaults ---
DEFAULT_LOG_DIR = "labs_logs"

def parse_logs(log_directory: str) -> Optional[List[Tuple[int, float, float, float]]]:
    """
    Parses log files recursively.
    Returns a list of tuples: (n, lambda, avg_time, avg_cut).
    """
    log_dir_path = Path(log_directory)
    
    if not log_dir_path.exists():
        print(f"Error: Directory '{log_directory}' not found.")
        return None

    # CHANGE 1: Recursive search (rglob) to find logs inside subfolders (run_2025...)
    print(f"Scanning '{log_directory}' recursively for .log files...")
    all_log_files = list(log_dir_path.rglob("*.log"))
    
    if not all_log_files:
        print(f"Error: No .log files found in '{log_directory}' or its subdirectories.")
        return None

    print(f"Found {len(all_log_files)} potential log files. Parsing...")
    
    raw_data = {}
    
    # CHANGE 2: Updated Regex for new filename format: "N7_L4.6.log"
    # Break down:
    # N(\d+)        -> Matches "N" followed by digits (Group 1)
    # _L([-\d.]+)   -> Matches "_L" followed by float/negative (Group 2)
    filename_regex = re.compile(r"N(\d+)_L([-\d.]+)\.log")
    
    # Content Regex (Looks for "real XmYs" and "Best cut: X")
    time_regex = re.compile(r"real\s+(\d+)m([\d.]+)s")
    cut_regex = re.compile(r"Best cut.*:\s*(-?[\d.eE+-]+)")

    success_count = 0
    forced_zero_count = 0
    
    for log_file in all_log_files:
        try:
            # 1. Check Filename
            name_match = filename_regex.search(log_file.name)
            if not name_match:
                # File doesn't match N.._L.. pattern, skip it (might be system logs)
                continue

            n = int(name_match.group(1))
            l_val = float(name_match.group(2))

            # 2. Read Content
            # errors='replace' ensures we don't crash on weird binary characters
            content = log_file.read_text(encoding='utf-8', errors='replace')
            
            time_match = time_regex.search(content)
            cut_match = cut_regex.search(content)

            if time_match and cut_match:
                # Parse Time
                minutes = float(time_match.group(1))
                seconds = float(time_match.group(2))
                total_time = (minutes * 60) + seconds
                
                # Parse Cut
                cut_value = float(cut_match.group(1))

                # --- LOGIC: Zero Handling ---
                # The screenshot showed "Best cut: 0.0". 
                # We are plotting log10(cut). log10(0) crashes.
                # We force 0.0 -> 1.0, so the result is 0 on the Z-axis.
                if cut_value < 1.0:
                    cut_value = 1.0
                    forced_zero_count += 1
                # ----------------------------
                
                key = (n, l_val)
                if key not in raw_data:
                    raw_data[key] = {'time': [], 'cut': []}
                
                raw_data[key]['time'].append(total_time)
                raw_data[key]['cut'].append(cut_value)
                success_count += 1
                
        except (ValueError, OSError) as e:
            print(f"Warning: Could not process {log_file.name}: {e}")

    print(f"Successfully parsed {success_count} valid experiment logs.")
    if forced_zero_count > 0:
        print(f"Note: {forced_zero_count} runs had a Cut of 0.0 (or <1). They were set to 1.0 for Log scaling.")

    if not raw_data:
        print("No valid data found. Check your filenames match 'N{int}_L{float}.log'.")
        return None

    # Flatten and Average duplicates
    cleaned_data = []
    for (n, l_val), values in raw_data.items():
        avg_time = sum(values['time']) / len(values['time'])
        avg_cut = sum(values['cut']) / len(values['cut']) 
        cleaned_data.append((n, l_val, avg_time, avg_cut))

    cleaned_data.sort(key=lambda x: (x[0], x[1]))
    return cleaned_data

def plot_3d_surface(data: List[Tuple[int, float, float, float]]):
    print("Generating 3D Surface data...")
    
    coords = []
    xy_points = [] 
    time_scalars = []

    for n, l_val, time, cut in data:
        # Z-Axis: Log scale of the Cut
        z_val = math.log10(cut)
        
        # Color: Log scale of Time
        # Safety floor for very fast runs (avoid log(0))
        t_val = math.log(time) if time > 1e-9 else -5.0
        
        coords.append([n, l_val, z_val])
        xy_points.append([n, l_val])
        time_scalars.append(t_val)

    coords_np = np.array(coords)
    xy_np = np.array(xy_points)
    time_np = np.array(time_scalars)

    # --- TRIANGULATION ---
    print("Triangulating points...")
    if len(xy_np) < 3:
        print("Error: Not enough data points (<3) to build a 3D surface.")
        return

    tri = Delaunay(xy_np)
    faces = tri.simplices
    
    surf = vedo.Mesh([coords_np, faces])
    
    # --- VISUALS ---
    # cmap: viridis_r (reversed). Bright Yellow = Fast, Dark Purple = Slow
    surf.cmap('viridis_r', time_np).alpha(0.9)
    
    # Wireframe
    surf.lw(1).lc("black")
    
    # Edge Opacity Fix
    try:
        prop = surf.properties if hasattr(surf, 'properties') else surf.GetProperty()
        prop.SetEdgeOpacity(0.2)
    except AttributeError:
        pass 

    # Plotter Setup
    plotter = vedo.Plotter(axes=0, bg='black', title="LABS Optimization Landscape")
    
    text_str = (
        "Optimization Landscape\n"
        "Z-Axis: log10(Cut Value)\n"
        "Color: ln(Time)\n"
        "Bright = Fast, Dark = Slow"
    )
    
    plotter.add(vedo.Text2D(text_str, pos="top-left", c="white", s=0.8))

    # Custom Axes
    axes = vedo.Axes(
        surf,
        xtitle='N (Seq Length)',
        ytitle='Lambda',
        ztitle='log10(Cut)',
        c='white',
        xygrid=False 
    )
    
    surf.add_scalarbar(title="ln(Time)", c='white')

    print("Displaying 3D Plot...")
    plotter.show(surf, axes, viewup='z', interactive=True)
    plotter.close()

def main():
    parser = argparse.ArgumentParser(description="Plot LABS Logs (Recursive Search)")
    parser.add_argument("dir", nargs="?", default=DEFAULT_LOG_DIR, help="Parent directory containing run folders")
    args = parser.parse_args()

    all_data = parse_logs(args.dir)
    
    if not all_data:
        sys.exit(1)

    try:
        plot_3d_surface(all_data)
    except Exception:
        print("\nCRITICAL ERROR in 3D Plotting:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
