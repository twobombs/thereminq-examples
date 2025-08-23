import os
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def parse_log_file(filepath):
    """
    Parses a VQE calculation log file to extract key metrics.
    """
    data = {}
    with open(filepath, 'r') as f:
        content = f.read()

        # Use regular expressions to find the data points
        molecule_match = re.search(r"Molecule: (.*)", content)
        qubits_match = re.search(r"Number of Qubits: (\d+)", content)
        time_match = re.search(r"Calculation Time: ([\d.]+) seconds", content)
        pct_diff_match = re.search(r"Percentage Difference: ([\d.]+)%", content)

        if all([molecule_match, qubits_match, time_match, pct_diff_match]):
            data['molecule'] = molecule_match.group(1).strip()
            data['qubits'] = int(qubits_match.group(1))
            data['time'] = float(time_match.group(1))
            data['pct_diff'] = float(pct_diff_match.group(1))
            return data
    return None

def visualize_results_3d():
    """
    Scans the calculation_logs directory, parses the files,
    and creates a 3D scatter plot of the results.
    """
    log_dir = "calculation_logs"
    if not os.path.isdir(log_dir):
        print(f"Error: Directory '{log_dir}' not found. Please run the main calculation script first.")
        return

    results = []
    for filename in os.listdir(log_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(log_dir, filename)
            data = parse_log_file(filepath)
            if data:
                results.append(data)

    if not results:
        print("No valid log files found to visualize.")
        return

    # Prepare data for plotting
    molecules = [res['molecule'] for res in results]
    qubits = np.array([res['qubits'] for res in results])
    times = np.array([res['time'] for res in results])
    pct_diffs = np.array([res['pct_diff'] for res in results])

    # Create the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a scatter plot
    scatter = ax.scatter(qubits, times, pct_diffs, c=pct_diffs, cmap='viridis', s=100, depthshade=True)

    # Add labels for each point
    for i, molecule in enumerate(molecules):
        ax.text(qubits[i], times[i], pct_diffs[i], f'  {molecule}', size=8, zorder=1, color='k')

    # Set labels and title
    ax.set_xlabel('Number of Qubits (Complexity)')
    ax.set_ylabel('Calculation Time (seconds)')
    ax.set_zlabel('Percentage Difference (Error %)')
    ax.set_title('VQE Performance: Complexity vs. Time vs. Accuracy')
    
    # Add a color bar
    cbar = fig.colorbar(scatter, shrink=0.5, aspect=10)
    cbar.set_label('Error Percentage')

    # --- MODIFICATION: Save the figure as a high-resolution PNG ---
    output_filename = 'vqe_performance_plot.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Graph saved as high-resolution PNG: {output_filename}")

    print("Displaying 3D plot. Close the plot window to exit.")
    plt.show()

if __name__ == "__main__":
    visualize_results_3d()

