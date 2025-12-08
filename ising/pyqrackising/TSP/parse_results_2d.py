import os
import re
import sys
import pandas as pd

# --- ADD THESE LINES ---
# Explicitly set the Matplotlib backend to an interactive one
import matplotlib
matplotlib.use('TkAgg')
# -----------------------

import matplotlib.pyplot as plt
import seaborn as sns

def parse_results(directory="results"):
    """
    Parses all .txt files in a directory to extract TSP experiment data.
    """
    data = []
    
    filename_pattern = re.compile(
        r"tsp_n(\d+)_q(\d+)_cq(\d+)_i(\d+)_s(\d+)\.txt"
    )

    path_length_pattern = re.compile(r"^Verified path length: ([\d.]+)", re.MULTILINE)
    node_count_pattern = re.compile(r"^Solution distinct node count: (\d+)", re.MULTILINE)

    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        return pd.DataFrame()

    print(f"Searching for result files in '{directory}'...")
    file_count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            match = filename_pattern.match(filename)
            if not match:
                continue

            params = match.groups()
            nodes, quality, c_quality, iterations, seed = map(int, params)

            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                content = f.read()
                
                path_length_match = path_length_pattern.search(content)
                node_count_match = node_count_pattern.search(content)

                if path_length_match and node_count_match:
                    verified_path_length = float(path_length_match.group(1))
                    distinct_nodes = int(node_count_match.group(1))
                    
                    data.append({
                        "nodes": nodes,
                        "quality": quality,
                        "correction_quality": c_quality,
                        "iterations": iterations,
                        "seed": seed,
                        "distinct_nodes": distinct_nodes,
                        "path_length": verified_path_length
                    })
                    file_count += 1

    if not data:
        print("No valid result files found.")
        return pd.DataFrame()

    print(f"Successfully parsed {file_count} files.")
    return pd.DataFrame(data)

def create_plot(df, output_filename="tsp_results_plot.png"):
    """
    Generates, saves, and displays a line plot of the TSP results.
    """
    if df.empty:
        print("DataFrame is empty, skipping plot generation.")
        return
        
    df['parameters'] = df.apply(lambda row: f"q={row['quality']}, cq={row['correction_quality']}", axis=1)
    
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(12, 8))
    
    ax = sns.lineplot(
        data=df,
        x='nodes',
        y='path_length',
        hue='parameters',
        style='parameters',
        markers=True,
        dashes=False
    )
    
    ax.set_title('TSP Path Length vs. Number of Nodes', fontsize=16)
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Verified Path Length', fontsize=12)
    
    plt.legend(title='Solver Parameters')
    
    plt.tight_layout()
    
    plt.savefig(output_filename)
    print(f"\nPlot saved to '{output_filename}'")
    
    # This should now open an interactive window
    plt.show()


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    
    results_df = parse_results(directory=results_dir)
    
    if not results_df.empty:
        print("\n--- Parsed Data Head ---")
        print(results_df.head())
        
        print("\n--- Summary Statistics for Path Length ---")
        summary = results_df.groupby(['quality', 'correction_quality', 'nodes'])['path_length'].describe()
        print(summary)
        
        create_plot(results_df)
      
