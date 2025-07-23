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

def plot_all_data(df):
    """
    Automatically generates a separate 3D plot for each qubit width (W),
    with subplots for each available Theta value.
    """
    if df is None or df.empty:
        print("DataFrame is empty. Cannot plot.")
        return

    available_widths = sorted(df['W'].unique())
    print(f"\nFound data for qubit widths: {available_widths}. Generating plots...")

    # Loop 1: Create a separate figure for each width
    for w_choice in available_widths:
        df_for_width = df[df['W'] == w_choice]
        thetas_for_width = sorted(df_for_width['Theta'].unique())
        num_thetas = len(thetas_for_width)
        
        if num_thetas == 0:
            continue

        # Create a figure with a subplot for each theta value
        fig, axes = plt.subplots(
            1,
            num_thetas,
            figsize=(7 * num_thetas, 6),
            subplot_kw={'projection': '3d'}
        )
        
        # Handle the case of a single subplot, where 'axes' is not an array
        if num_thetas == 1:
            axes = [axes]

        fig.suptitle(f'TFIM Phase Diagram for {w_choice} Qubits', fontsize=16, y=0.98)

        # Loop 2: Create a subplot for each theta value
        for i, theta_choice in enumerate(thetas_for_width):
            ax = axes[i]
            subplot_df = df_for_width[np.isclose(df_for_width['Theta'], theta_choice)]

            if subplot_df.empty:
                ax.set_title(f'θ = {theta_choice:.4f}\n(No Data)')
                continue
            
            # Prepare data for the 3D surface plot
            J_vals = sorted(subplot_df['J'].unique())
            h_vals = sorted(subplot_df['h'].unique())
            J_grid, H_grid = np.meshgrid(J_vals, h_vals)
            M_grid = subplot_df.pivot(index='h', columns='J', values='M').values
            
            # Plot the surface
            surf = ax.plot_surface(J_grid, H_grid, M_grid, cmap=cm.viridis, antialiased=False)
            
            ax.set_xlabel('J (Ising Coupling)')
            ax.set_ylabel('h (Transverse Field)')
            ax.set_zlabel('<M>')
            ax.set_title(f'θ = {theta_choice:.4f}')
            fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

    print("\nDisplaying all plots... Close the plot windows to exit.")
    plt.show()

if __name__ == "__main__":
    results_directory = "tfim_results"
    if not os.path.isdir(results_directory):
        print(f"Error: Directory '{results_directory}' not found.")
    else:
        master_df = parse_data(results_directory)
        if master_df is not None:
            plot_all_data(master_df)
