import pandas as pd
import glob
import re
from vedo import Points, show, Axes  # <-- 'settings' is no longer imported
import numpy as np

# --- 1. Data Loading and Preparation ---

file_paths = glob.glob('factor_landscape*.log')
all_data = []

print(f"Found {len(file_paths)} files to process...")

for path in file_paths:
    match = re.search(r'_q(\d+)', path)
    if not match:
        print(f"Could not extract quality value from {path}. Skipping.")
        continue
    
    z_value = int(match.group(1))

    try:
        df = pd.read_csv(path)
        df['z'] = z_value
        all_data.append(df)
        print(f"Processed {path} with z-value = {z_value}")
    except Exception as e:
        print(f"Error reading {path}: {e}")

if not all_data:
    print("No data was loaded. Exiting.")
    exit()

master_df = pd.concat(all_data, ignore_index=True)

# --- 2. 3D Visualization with Vedo (with Manual Scaling) ---

points_coords = master_df[['p', 'q', 'z']].values
cost_values = master_df['cost'].values

plot_points = Points(points_coords)

bounds = plot_points.bounds()
x_range = bounds[1] - bounds[0]
y_range = bounds[3] - bounds[2]
z_range = bounds[5] - bounds[4]

if x_range == 0: x_range = 1
if y_range == 0: y_range = 1
if z_range == 0: z_range = 1

max_range = max(x_range, y_range, z_range)

sx = max_range / x_range
sy = max_range / y_range
sz = max_range / z_range

plot_points.scale([sx, sy, sz])

plot_points.cmap('viridis', cost_values, on='points').add_scalarbar('Cost')

# Make the axes and their labels white so they show up on a dark background
axs = Axes(plot_points, xtitle='p', ytitle='q', ztitle='Quality', c='white')

# Show the plot, manually setting the background color to black
show(
    plot_points,
    axs,
    "Factorization Landscape",
    viewup='z',
    bg='black'  # <-- Set background color here
).close()
