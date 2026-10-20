# display_heatmap.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys

def create_heatmap(filepath):
    """
    Loads data from a CSV file, cleans it, and displays a heatmap
    of computational cost vs. qubits and depth.

    Args:
        filepath (str): The path to the input CSV file.
    """
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded '{filepath}'.")
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.", file=sys.stderr)
        sys.exit(1)

    # --- 2. Clean Data ---
    # Drop rows with any missing values, which represent failed runs
    original_rows = len(df)
    df.dropna(inplace=True)
    
    # Ensure the core columns are converted to a numeric type
    for col in ['qubits', 'depth', 'cost']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop again in case the numeric conversion created new missing values
    df.dropna(subset=['qubits', 'depth', 'cost'], inplace=True)
    
    print(f"Removed {original_rows - len(df)} rows with missing data.")

    if df.empty:
        print("Error: No valid data remains after cleaning. Cannot generate plot.", file=sys.stderr)
        sys.exit(1)

    # --- 3. Reshape Data ---
    # Pivot the data to create a 2D grid required for the heatmap
    try:
        heatmap_data = df.pivot(index='depth', columns='qubits', values='cost')
    except Exception as e:
        print(f"Error creating pivot table: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 4. Generate and Display Plot ---
    print("Generating heatmap...")
    plt.figure(figsize=(18, 14))
    sns.heatmap(heatmap_data, annot=False, cmap='viridis', linewidths=.5)
    
    plt.title('Computational Cost vs. Qubits and Depth', fontsize=16)
    plt.xlabel('Number of Qubits', fontsize=12)
    plt.ylabel('Circuit Depth', fontsize=12)
    plt.tight_layout()

    print("Displaying plot. Close the window to exit the script.")
    plt.show()


if __name__ == "__main__":
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Generate and display a heatmap from a simulation summary CSV file."
    )
    # Add a required argument for the CSV file path
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the input CSV file (e.g., summary.csv)"
    )
    
    args = parser.parse_args()
    
    # Call the main function with the provided filename
    create_heatmap(args.csv_file)
