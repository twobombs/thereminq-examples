# `generate_tfim_samples_vars_graph_auto_big_legenda_c.py`

```python
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
    Scans a directory for log files, parses parameters from filenames and file content,
    and computes the average magnetization from the samples in each file.
    Adapts to the new log format where parameters and samples are inside the file.
    """
    # Filename pattern is still needed for Theta (T) and for file discovery.
    pattern = re.compile(r"W(\d+)_H([\d\.]+)_J(-?[\d\.]+)_T(-?[\d\.]+)\.log")

    parsed_data = []
    print(f"Scanning directory '{results_dir}' for new log format...")

    for filename in os.listdir(results_dir):
        # Match filename to find relevant logs and get Theta
        match = pattern.match(filename)
        if not match:
            continue

        # Theta is the 4th captured group in the regex pattern
        theta_str = match.groups()[3]

        try:
            filepath = Path(results_dir) / filename
            with open(filepath, 'r') as f:
                content = f.read()

            # Use regex to find the line with parameters and extract them
            param_match = re.search(r"n_qubits:\s*(\d+).*J:\s*(-?[\d\.]+).*h:\s*([\d\.]+)", content)
            if not param_match:
                # Silently skip files that don't match the new internal format
                continue

            n_qubits = int(param_match.group(1))
            j = float(param_match.group(2))
            h = float(param_match.group(3))

            # Use regex to find the line with samples and extract the list string
            sample_match = re.search(r"Samples:\s*(\[.*\])", content)
            if not sample_match:
                continue

            # Safely evaluate the string to a Python list
            samples = ast.literal_eval(sample_match.group(1))
            theta = float(theta_str)

        except (SyntaxError, ValueError, FileNotFoundError, IndexError) as e:
            print(f"Could not process file {filename}: {e}")
            continue

        # Magnetization calculation logic is unchanged
        magnetizations = []
        for s in samples:
            spins_down = s.bit_count()
            spins_up = n_qubits - spins_down
            m = (spins_up - spins_down) / n_qubits
            magnetizations.append(m)

        avg_magnetization = np.mean(magnetizations) if magnetizations else 0

        # The data structure for the DataFrame is unchanged
        parsed_data.append({
            'W': n_qubits,
            'h': h,
            'J': j,
            'Theta': theta,
            'M': avg_magnetization
        })

    if not parsed_data:
        print("No data files matching the new format were found or processed. Exiting.")
        return None

    print(f"Found and processed {len(parsed_data)} data files.")
    return pd.DataFrame(parsed_data)

def plot_all_data_in_grid(df):
    """
    Automatically generates a single figure with a grid of 3D plots.
    Rows are determined by qubit width (W), columns by Theta.
    Adds a shared color bar representing the average magnetization <M>.
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

    # Initialize a variable to hold the surface plot object for the color bar
    surf = None

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

            # Plot the surface, capturing the last `surf` object for the color bar
            surf = ax.plot_surface(J_grid, H_grid, M_grid, cmap=cm.viridis, antialiased=False)

    # Adjust layout to prevent labels/titles from overlapping
    # The `rect` argument makes space for the suptitle and the color bar
    plt.tight_layout(rect=[0, 0.03, 0.95, 0.96])

    # Add a shared color bar (legend) for the entire figure if any plots were made
    if surf:
        cbar = fig.colorbar(surf, ax=axes.ravel().tolist(), shrink=0.7, aspect=25, pad=0.02)
        cbar.set_label('Average Magnetization <M>', size=14)

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
```
