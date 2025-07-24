# -*- coding: utf-8 -*-
import os
import re
import ast
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# --- Configuration ---
# Set up basic logging to provide informative output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Core Logic ---

def _calculate_avg_magnetization(samples: List[int], n_qubits: int) -> float:
    """
    Calculates the average magnetization for a list of samples.

    Args:
        samples: A list of integer samples, where each integer represents a state.
        n_qubits: The total number of qubits.

    Returns:
        The calculated average magnetization. Returns 0 if samples are empty.
    """
    if not samples:
        return 0.0
    
    magnetizations = []
    for s in samples:
        # int.bit_count() efficiently counts set bits (spins_down in this context)
        spins_down = s.bit_count()
        spins_up = n_qubits - spins_down
        m = (spins_up - spins_down) / n_qubits
        magnetizations.append(m)
    
    return np.mean(magnetizations) if magnetizations else 0.0


def parse_single_file(filepath: Path) -> Optional[Dict[str, Any]]:
    """
    Parses a single log file to extract parameters and calculate average magnetization.

    This function is updated to handle the new log format where parameters
    are on the first line, comma-separated.

    Args:
        filepath: The Path object pointing to the log file.

    Returns:
        A dictionary containing the parsed data, or None if parsing fails.
    """
    # Filename pattern is used for discovering log files and extracting Theta (T)
    filename_pattern = re.compile(r"W(\d+)_H([\d\.]+)_J(-?[\d\.]+)_T(-?[\d\.]+)\.log")
    match = filename_pattern.match(filepath.name)
    if not match:
        logging.warning(f"Filename '{filepath.name}' does not match expected pattern. Skipping.")
        return None

    try:
        theta = float(match.group(4))
        # Open the file with UTF-8 encoding to be safe
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # --- MODIFIED PARAMETER PARSING ---
        # The new log format has all parameters on the first line, comma-separated.
        lines = content.split('\n', 1)
        if not lines:
            logging.warning(f"File '{filepath.name}' is empty. Skipping.")
            return None
        
        param_line = lines[0]
        params = {}
        # Split by comma, then by colon, to build a dictionary of parameters.
        for part in param_line.split(','):
            key_value = part.split(':')
            if len(key_value) == 2:
                key, value = key_value
                params[key.strip()] = value.strip()

        # Extract required parameters from the parsed dictionary
        if 'n_qubits' not in params or 'J' not in params or 'h' not in params:
            logging.warning(f"Could not find all required parameters (n_qubits, J, h) on the first line of '{filepath.name}'. Skipping.")
            return None

        n_qubits = int(params['n_qubits'])
        j_coupling = float(params['J'])
        h_field = float(params['h'])
        # --- END OF MODIFICATION ---

        # Regex to find the line with samples (this logic remains effective)
        sample_match = re.search(r"Samples:\s*(\[.*\])", content, re.MULTILINE)
        if not sample_match:
            logging.warning(f"Could not find samples in '{filepath.name}'. Skipping.")
            return None
        
        # Safely evaluate the string representation of the list
        samples = ast.literal_eval(sample_match.group(1))
        
        avg_magnetization = _calculate_avg_magnetization(samples, n_qubits)

        return {
            'W': n_qubits,
            'h': h_field,
            'J': j_coupling,
            'Theta': theta,
            'M': avg_magnetization
        }

    except (ValueError, SyntaxError, IndexError, KeyError) as e:
        logging.error(f"Could not process file {filepath.name}: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {filepath.name}: {e}")
        return None


def parse_directory(results_dir: str) -> Optional[pd.DataFrame]:
    """
    Scans a directory for log files, parses them, and returns a pandas DataFrame.
    """
    results_path = Path(results_dir)
    if not results_path.is_dir():
        logging.error(f"Error: Directory '{results_dir}' not found.")
        return None

    logging.info(f"Scanning directory '{results_dir}' for log files...")
    
    all_data = []
    for filepath in results_path.glob("*.log"):
        parsed_data = parse_single_file(filepath)
        if parsed_data:
            all_data.append(parsed_data)
    
    if not all_data:
        logging.warning("No data files were successfully processed. Exiting.")
        return None
        
    logging.info(f"Found and processed {len(all_data)} data files.")
    return pd.DataFrame(all_data)


def plot_all_data(df: pd.DataFrame, save_plots: bool = False, output_dir: str = "plots"):
    """
    Generates and displays/saves 3D plots for each qubit width (W) and Theta.
    """
    if df is None or df.empty:
        logging.warning("DataFrame is empty. Cannot generate plots.")
        return

    if save_plots:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        logging.info(f"Plots will be saved to '{output_dir}/'")

    available_widths = sorted(df['W'].unique())
    logging.info(f"Found data for qubit widths: {available_widths}. Generating plots...")

    for w_choice in available_widths:
        df_for_width = df[df['W'] == w_choice]
        thetas_for_width = sorted(df_for_width['Theta'].unique())
        
        num_thetas = len(thetas_for_width)
        if num_thetas == 0:
            continue

        fig, axes = plt.subplots(
            1, num_thetas, figsize=(7 * num_thetas, 6.5), subplot_kw={'projection': '3d'}
        )
        
        if num_thetas == 1:
            axes = [axes]  # Ensure axes is always iterable

        fig.suptitle(f'TFIM Phase Diagram for {w_choice} Qubits', fontsize=18, y=1.0)

        for i, theta_choice in enumerate(thetas_for_width):
            ax = axes[i]
            subplot_df = df_for_width[np.isclose(df_for_width['Theta'], theta_choice)]

            if subplot_df.empty:
                ax.set_title(f'T = {theta_choice:.4f}\n(No Data)')
                continue
            
            try:
                m_pivot = subplot_df.pivot_table(index='h', columns='J', values='M')
                J_vals = m_pivot.columns.values
                h_vals = m_pivot.index.values
                J_grid, H_grid = np.meshgrid(J_vals, h_vals)
                M_grid = m_pivot.values
                
                surf = ax.plot_surface(J_grid, H_grid, M_grid, cmap=cm.viridis, antialiased=False, rstride=1, cstride=1)
                
                ax.set_xlabel('J (Ising Coupling)', fontsize=10, labelpad=10)
                ax.set_ylabel('h (Transverse Field)', fontsize=10, labelpad=10)
                # --- CORRECTED LINE ---
                ax.set_zlabel(' M  (Avg. Magnetization)', fontsize=10, labelpad=10)
                ax.set_title(f'T = {theta_choice:.4f}', fontsize=12)
                fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)

            except Exception as e:
                logging.error(f"Failed to plot for W={w_choice}, Theta={theta_choice}: {e}")
                ax.set_title(f'T = {theta_choice:.4f}\n(Plotting Error)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_plots:
            plot_filename = output_path / f"tfim_plot_W{w_choice}.png"
            plt.savefig(plot_filename, dpi=300)
            logging.info(f"Saved plot to {plot_filename}")
        
    if not save_plots:
        logging.info("Displaying all plots. Close the plot windows to exit.")
        plt.show()


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse and plot Transverse Field Ising Model (TFIM) simulation data."
    )
    parser.add_argument(
        "results_dir",
        type=str,
        default="tfim_results",
        nargs='?', # Makes the argument optional
        help="Directory containing the .log files from the simulation."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save plots to files instead of displaying them."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="plots",
        help="Directory to save the output plots."
    )
    args = parser.parse_args()

    master_df = parse_directory(args.results_dir)
    if master_df is not None:
        plot_all_data(master_df, save_plots=args.save, output_dir=args.out)

