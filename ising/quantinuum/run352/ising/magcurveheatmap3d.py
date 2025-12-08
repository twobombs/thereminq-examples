import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import ast

# --- Instructions ---
# 1. Create a file named 'fullog.txt' in the same directory as this script.
# 2. Copy and paste your log data into 'fullog.txt'. Each dictionary should be on a new line
#    and MUST contain 'width', 'depth', 'magnetization', and 'square_magnetization'.
#    Example:
#    {'width': 96, 'depth': 14, 'magnetization': 0.029, 'square_magnetization': 0.004, ...}
# 3. Run this Python script. It will display the two 3D plots and save them as a single PNG.

def create_separate_3d_plots_from_log(file_path='fullog.txt'):
    """
    Reads quantum simulation log data and generates two separate 3D surface plots
    in the same window: one for magnetization and one for square_magnetization.
    """
    try:
        # --- Data Loading and Processing (same as before) ---
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data.append(ast.literal_eval(line.strip()))
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Skipping malformed line: {line.strip()}\nError: {e}")
                    continue

        if not data:
            print("Error: No data was loaded. Please check 'fullog.txt'.")
            return

        df = pd.DataFrame(data)
        
        required_cols = ['width', 'depth', 'magnetization', 'square_magnetization']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: Log file must contain columns: {', '.join(required_cols)}")
            return

        # --- Pivot data for each plot ---
        pivot_mag = df.pivot_table(index='width', columns='depth', values='magnetization')
        pivot_sq_mag = df.pivot_table(index='width', columns='depth', values='square_magnetization')

        X = pivot_mag.columns.values
        Y = pivot_mag.index.values
        X, Y = np.meshgrid(X, Y)
        
        Z_mag = pivot_mag.values
        Z_sq_mag = pivot_sq_mag.values

        # --- Create a figure with two 3D subplots ---
        # plt.subplots creates a figure and a grid of subplots.
        # (1, 2) means 1 row, 2 columns.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': '3d'})
        
        # Add an overall title for the entire figure
        fig.suptitle('Magnetization vs. Square Magnetization', fontsize=20)

        # --- Plot 1: Magnetization ---
        surf1 = ax1.plot_surface(X, Y, Z_mag, cmap='viridis', edgecolor='none')
        ax1.set_title('Magnetization', fontsize=16)
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='Magnetization Value')
        
        # --- Plot 2: Square Magnetization ---
        surf2 = ax2.plot_surface(X, Y, Z_sq_mag, cmap='plasma', edgecolor='none')
        ax2.set_title('Square Magnetization', fontsize=16)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='Square Magnetization Value')

        # --- Configure labels and view for both plots ---
        for ax in [ax1, ax2]:
            ax.set_xlabel('Circuit Depth', fontsize=12, labelpad=10)
            ax.set_ylabel('Qubit Width', fontsize=12, labelpad=10)
            ax.set_zlabel('Value', fontsize=12, labelpad=10)
            # Set the same viewing angle for direct comparison
            ax.view_init(elev=20, azim=-65)

        # Adjust layout to prevent titles/labels from overlapping
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make room for suptitle

        # Save and show the plot
        plt.savefig('qubit_separate_3d_plots.png', dpi=300)
        print("Side-by-side 3D plots saved as 'qubit_separate_3d_plots.png'")
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    create_separate_3d_plots_from_log()
