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
    and creates a 3D scatter plot of the results with a legend, labels, and color bar.
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

    # --- MODIFICATION: Use different markers for each molecule for the legend ---
    unique_molecules = sorted(list(set(molecules)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    molecule_to_marker = {molecule: markers[i % len(markers)] for i, molecule in enumerate(unique_molecules)}

    # Create the 3D plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0.25, right=0.85) # Adjust plot area

    # Plot each molecule's data with its unique marker
    # The color is still determined by the percentage difference
    scatter_objects = []
    for molecule in unique_molecules:
        mask = [m == molecule for m in molecules]
        scatter = ax.scatter(
            qubits[mask], times[mask], pct_diffs[mask],
            c=pct_diffs[mask],  # Color by error
            cmap='viridis',
            s=150,
            marker=molecule_to_marker[molecule],
            label=molecule,
            depthshade=True
        )
        scatter_objects.append(scatter)

    # Add text labels directly to the points in the graph
    for i, molecule in enumerate(molecules):
        ax.text(qubits[i], times[i], pct_diffs[i], f'  {molecule}', size=8, zorder=1, color='k')

    # Set labels and title
    ax.set_xlabel('Number of Qubits (Complexity)')
    ax.set_ylabel('Calculation Time (seconds)')
    ax.set_zlabel('Percentage Difference (Error %)')
    ax.set_title('VQE Performance: Complexity vs. Time vs. Accuracy')
    
    # --- MODIFICATION: Add the legend on the left ---
    ax.legend(title='Molecules', loc='center left', bbox_to_anchor=(-0.4, 0.5))

    # --- MODIFICATION: Add the color bar on the right ---
    # We use the first scatter object to create the color bar, as they all share the same color map
    cbar = fig.colorbar(scatter_objects[0], ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Error Percentage')

    # Save the figure as a high-resolution PNG
    output_filename = 'vqe_performance_plot.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Graph saved as high-resolution PNG: {output_filename}")

    print("Displaying 3D plot. Close the plot window to exit.")
    plt.show()

if __name__ == "__main__":
    visualize_results_3d()
