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
                
                # --- THIS IS THE FIX ---
                # Ensure the file has enough lines for data and time
                if len(lines) < 2:
                    continue
                # The samples are on the second-to-last line
                sample_line = lines[-2] 
                samples = ast.literal_eval(sample_line)
                # --- END OF FIX ---

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

def plot_3d_surface(df):
    """
    Asks the user to select parameters and plots a 3D surface graph.
    """
    available_widths = sorted(df['W'].unique())
    available_thetas = sorted(df['Theta'].unique())

    print("\n--- Select Parameters for Plotting ---")
    print("Available qubit counts (W):", available_widths)
    w_choice = int(input("Enter the qubit count (W) you want to plot: "))
    
    print("\nAvailable Thetas (T):", [f"{t:.4f}" for t in available_thetas])
    theta_choice = float(input("Enter the Theta value you want to plot: "))

    if w_choice not in available_widths or not np.any(np.isclose(theta_choice, available_thetas)):
        print("Invalid selection. Please run the script again.")
        return

    plot_df = df[(df['W'] == w_choice) & (np.isclose(df['Theta'], theta_choice))]

    if plot_df.empty:
        print("No data available for the selected combination.")
        return

    J_vals = sorted(plot_df['J'].unique())
    h_vals = sorted(plot_df['h'].unique())
    J_grid, H_grid = np.meshgrid(J_vals, h_vals)
    M_grid = plot_df.pivot(index='h', columns='J', values='M').values

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(J_grid, H_grid, M_grid, cmap=cm.viridis, antialiased=False)

    ax.set_xlabel('J (Ising Coupling)', fontweight='bold')
    ax.set_ylabel('h (Transverse Field)', fontweight='bold')
    ax.set_zlabel('Average Magnetization <M>', fontweight='bold')
    ax.set_title(f'TFIM Phase Diagram for W={w_choice}, θ={theta_choice:.4f}', fontsize=16)
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Magnetization')
    
    print("\nDisplaying plot... Close the plot window to exit.")
    plt.show()


if __name__ == "__main__":
    results_directory = "tfim_results"
    if not os.path.isdir(results_directory):
        print(f"Error: Directory '{results_directory}' not found.")
    else:
        master_df = parse_data(results_directory)
        if master_df is not None:
            plot_3d_surface(master_df)
