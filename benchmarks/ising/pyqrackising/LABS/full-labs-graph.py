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

# --- Configuration ---
DEFAULT_LOG_DIR = "labs_logs"

def parse_logs(log_directory: str) -> Optional[List[Tuple[int, float, float, float]]]:
    """
    Parses log files recursively.
    Returns list of (n, lambda, avg_time, avg_cut).
    """
    log_dir_path = Path(log_directory)
    
    if not log_dir_path.exists():
        print(f"Error: Directory '{log_directory}' not found.")
        return None

    print(f"Scanning '{log_directory}' recursively for .log files...")
    all_log_files = list(log_dir_path.rglob("*.log"))
    
    if not all_log_files:
        print(f"Error: No .log files found in '{log_directory}'.")
        return None

    print(f"Found {len(all_log_files)} log files. Parsing...")
    
    raw_data = {}
    
    # Regex: Enforces proper float format for Lambda (e.g., -4.6, 0.5, 10)
    filename_regex = re.compile(r"N(\d+)_L(-?\d+\.?\d*)\.log")
    
    # Content Regex
    time_regex = re.compile(r"real\s+(\d+)m([\d.]+)s")
    cut_regex = re.compile(r"Best cut.*:\s*(-?[\d.eE+-]+)")

    success_count = 0
    forced_zero_count = 0
    
    for log_file in all_log_files:
        try:
            # 1. Filename Check
            name_match = filename_regex.search(log_file.name)
            if not name_match:
                continue

            n = int(name_match.group(1))
            l_val = float(name_match.group(2))

            # 2. Content Read
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

                # Log-Scale Safety: Force 0 -> 1
                if cut_value < 1.0:
                    cut_value = 1.0
                    forced_zero_count += 1
                
                key = (n, l_val)
                if key not in raw_data:
                    raw_data[key] = {'time': [], 'cut': []}
                
                raw_data[key]['time'].append(total_time)
                raw_data[key]['cut'].append(cut_value)
                success_count += 1
                
        except Exception as e:
            print(f"Skipping {log_file.name}: {e}")

    print(f"Parsed {success_count} valid logs.")
    if forced_zero_count > 0:
        print(f"Note: {forced_zero_count} runs had Cut < 1.0 (set to 1.0).")

    if not raw_data:
        print("No valid data found.")
        return None

    # Average duplicates
    cleaned_data = []
    for (n, l_val), values in raw_data.items():
        avg_time = np.mean(values['time'])
        avg_cut = np.mean(values['cut']) 
        cleaned_data.append((n, l_val, avg_time, avg_cut))

    cleaned_data.sort(key=lambda x: (x[0], x[1]))
    return cleaned_data

def plot_3d_surface(data: List[Tuple[int, float, float, float]]):
    print("Generating 3D Surface...")
    
    coords = []
    xy_points = [] 
    time_scalars = []

    for n, l_val, time, cut in data:
        # Z: Log(Cut), Color: Log(Time)
        z_val = math.log10(cut)
        t_val = math.log(time) if time > 1e-9 else -5.0
        
        coords.append([n, l_val, z_val])
        xy_points.append([n, l_val])
        time_scalars.append(t_val)

    coords_np = np.array(coords)
    xy_np = np.array(xy_points)
    time_np = np.array(time_scalars)

    if len(xy_np) < 3:
        print("Error: Need at least 3 points for a surface.")
        return

    # --- TRIANGULATION ---
    tri = Delaunay(xy_np)
    faces = tri.simplices
    
    # Create Mesh
    surf = vedo.Mesh([coords_np, faces])
    
    # Map Colors (Scalars) to Mesh
    surf.cmap('viridis_r', time_np).alpha(0.9)
    surf.lw(0.1).lc("black") # Thin wireframe

    # --- VISUALS ---
    plotter = vedo.Plotter(axes=0, bg='black', title="LABS Optimization Landscape")
    
    text_str = (
        "Optimization Landscape\n"
        "Z: log10(Cut)\n"
        "Color: ln(Time)"
    )
    plotter.add(vedo.Text2D(text_str, pos="top-left", c="white", s=0.8))

    # Custom Axes
    axes = vedo.Axes(
        surf,
        xtitle='N (Sequence Length)',
        ytitle='Lambda (Hyperparam)',
        ztitle='log10(Cut Value)',
        c='white',
    )
    
    surf.add_scalarbar(title="ln(Time)  [Bright=Fast]", c='white')

    print("Displaying interactive plot...")
    # Removed 'points' from the show() call
    plotter.show(surf, axes, viewup='z', interactive=True)
    plotter.close()

def main():
    parser = argparse.ArgumentParser(description="Plot LABS Logs")
    parser.add_argument("dir", nargs="?", default=DEFAULT_LOG_DIR, help="Log directory")
    args = parser.parse_args()

    all_data = parse_logs(args.dir)
    
    if all_data:
        try:
            plot_3d_surface(all_data)
        except Exception:
            traceback.print_exc()

if __name__ == "__main__":
    main()
