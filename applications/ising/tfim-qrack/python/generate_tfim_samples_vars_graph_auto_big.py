import os
import re
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

def parse_data(results_dir):
    """
    Scans a directory for log files, parses parameters from filenames,
    and computes the average magnetization from the samples in each file.
    """
    pattern = re.compile(r"W(\d+)_H([\d\.]+)_J(-?[\d\.]+)_T(-?[\d\.]+)\.log")
    
    parsed_data = []
    print(f"Scanning directory '{results_dir}'...")

    for filename in os.listdir(results_dir):
        match = pattern.match(filename)
        if not match:
            continue

        width, h, j, theta = match.groups()
        n_qubits = int(width)
        
        try:
            filepath = Path(results_dir) / filename
            with open(filepath, 'r') as f:
                lines = [line for line in f.readlines() if line.strip()]
                if len(lines) < 2:
                    continue
                sample_line = lines[-2] 
                samples = ast.literal_eval(sample_line)
        except (SyntaxError, ValueError, FileNotFoundError, IndexError) as e:
            print(f"Could not process file {filename}: {e}")
            continue

        magnetizations = []
        for s in samples:
            spins_down = s.bit_count()
            spins_up = n_qubits - spins_down
            m = (spins_up - spins_down) / n_qubits
            magnetizations.append(m)
        
        avg_magnetization = np.mean(magnetizations) if magnetizations else 0

        parsed_data.append({
            'W': n_qubits,
            'h': float(h),
            'J': float(j),
            'Theta': float(theta),
            'M': avg_magnetization
        })
    
    if not parsed_data:
        print("No data files found or processed. Exiting.")
        return None
        
    print(f"Found and processed {len(parsed_data)} data files.")
    return pd.DataFrame(parsed_data)

def plot_all_data_in_grid(df):
    """
    Automatically generates a single figure with a grid of 3D plots.
    Rows are determined by qubit width (W), columns by Theta.
    The final figure is saved to a high-resolution PNG file.
    """
    if df is None or df.empty:
        print("DataFrame is empty. Cannot plot.")
        return

    # Determine the grid dimensions from the unique values in the data
    available_widths = sorted(df['W'].unique())
    available_thetas = sorted(df['Theta'].unique())
    nrows = len(available_widths)
    ncols = len(available_thetas)

    if nrows == 0 or ncols == 0:
        print("Not enough data diversity to create a grid. Exiting.")
        return
        
    print(f"\nFound {nrows} widths and {ncols} thetas. Generating a {nrows}x{ncols} grid plot...")

    # Create the single large figure with a grid of subplots
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 5 * nrows),
        subplot_kw={'projection': '3d'},
        squeeze=False # Always return a 2D array for axes
    )

    fig.suptitle('TFIM Phase Diagram Grid', fontsize=24, y=0.98)

    # Loop through each cell in the grid (i=row, j=col)
    for i, w_choice in enumerate(available_widths):
        for j, theta_choice in enumerate(available_thetas):
            ax = axes[i, j]
            
            # Filter data for the specific W and Theta of this subplot
            subplot_df = df[(df['W'] == w_choice) & (np.isclose(df['Theta'], theta_choice))]
            
            ax.set_title(f'W = {w_choice}, θ = {theta_choice:.4f}', fontsize=12)
            ax.set_xlabel('J')
            ax.set_ylabel('h')
            ax.set_zlabel('<M>')

            if subplot_df.empty:
                ax.text2D(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
                continue

            # Prepare data for the 3D surface plot
            J_vals = sorted(subplot_df['J'].unique())
            h_vals = sorted(subplot_df['h'].unique())
            J_grid, H_grid = np.meshgrid(J_vals, h_vals)
            M_grid = subplot_df.pivot(index='h', columns='J', values='M').values
            
            # Plot the surface
            surf = ax.plot_surface(J_grid, H_grid, M_grid, cmap=cm.viridis, antialiased=False)
    
    # Adjust layout to prevent labels/titles from overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save the figure to a high-resolution PNG file
    output_filename = 'tfim_phase_diagram_grid.png'
    print(f"\nSaving figure to '{output_filename}' with high resolution (300 DPI)...")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Display the plot on screen
    print("Displaying combined plot... Close the plot window to exit.")
    plt.show()

if __name__ == "__main__":
    results_directory = "tfim_results"
    if not os.path.isdir(results_directory):
        print(f"Error: Directory '{results_directory}' not found.")
    else:
        master_df = parse_data(results_directory)
        if master_df is not None:
            plot_all_data_in_grid(master_df)
