# -*- coding: utf-8 -*-
import time
import argparse
import os
import multiprocessing
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import math

# Assuming PyQrackIsing is installed and accessible
from PyQrackIsing import tfim_square_magnetization

# Helper function for float range
def frange(start, stop, step):
    i = start
    epsilon = 1e-9
    while i <= stop + epsilon:
        yield i
        i += step

# --- Worker function for the simulation (No changes needed here) ---
def run_simulation(params):
    """
    Runs a single simulation and returns the parameters and results.
    This worker does NOT perform any file I/O.
    """
    J, h, theta, t, n_qubits = params
    start_time = time.perf_counter()
    samples = None
    error_message = None

    try:
        samples = tfim_square_magnetization(
            J=J, h=h, z=4, theta=theta, t=t, n_qubits=n_qubits,
        )
    except Exception as e:
        error_message = f"ERROR: {e}"

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    return {
        "J": J, "h": h, "theta": theta, "t": t, "samples": samples,
        "elapsed_time": elapsed_time, "error": error_message
    }

# --- Main part of the script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TFIM simulations for a SINGLE Trotter step. Intended to be run in parallel for each step."
    )
    parser.add_argument('--log_dir', type=str, default='tfim_results_final',
                        help='Base directory to save aggregated CSV files.')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Maximum number of worker processes. Defaults to number of CPU cores.')
    parser.add_argument('--n_qubits', type=int, required=True,
                        help='The specific number of qubits for this simulation run.')
    parser.add_argument('--trotter_step', type=int, required=True,
                        help='The specific Trotter step (t) for this simulation run (e.g., 1 to 20).')

    args = parser.parse_args()

    # Create a specific directory for this n_qubits job
    job_log_dir = os.path.join(args.log_dir, f"nQ_{args.n_qubits}")
    os.makedirs(job_log_dir, exist_ok=True)

    # Determine number of workers
    max_workers = args.max_workers if args.max_workers else multiprocessing.cpu_count()
    
    # Define iteration ranges
    J_values = [round(x, 1) for x in frange(-2.0, 2.0, 0.1)]
    h_values = [round(x, 1) for x in frange(-2.0, 2.0, 0.1)]
    
    theta_values = [
        -math.pi / 2, -math.pi / 4, -math.pi / 18, 0.0,
        math.pi / 18, math.pi / 4, math.pi / 2,
    ]
    
    # Get fixed parameters from command line
    fixed_t = args.trotter_step
    fixed_n_qubits = args.n_qubits

    # Generate parameter combinations for the fixed t
    param_combinations_for_t = itertools.product(J_values, h_values, theta_values)
    tasks = [(J, h, theta, fixed_t, fixed_n_qubits) for J, h, theta in param_combinations_for_t]
    total_simulations = len(tasks)
    
    # This list will store all results for this specific job
    job_results = []
    
    simulation_count = 0
    start_total_time = time.perf_counter()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_simulation, task) for task in tasks]

        for future in as_completed(futures):
            simulation_count += 1
            try:
                result_data = future.result()
                
                if result_data["error"] is None:
                    samples_val = result_data["samples"]
                    if isinstance(samples_val, (list, tuple)):
                        samples_for_csv = ','.join(map(str, samples_val))
                    else:
                        samples_for_csv = str(samples_val)
                    
                    job_results.append({
                        "J": result_data["J"], "h": result_data["h"], 
                        "theta": result_data["theta"], "samples": samples_for_csv
                    })
                else:
                    # Still print errors
                    print(f"\n[t={fixed_t}] ERROR: {result_data['error']} for J={result_data['J']}, h={result_data['h']}")

                # MODIFIED: Simplified progress update
                elapsed = time.perf_counter() - start_total_time
                sims_per_sec = simulation_count / elapsed if elapsed > 0 else 0
                print(f"\rCompleted: {simulation_count}/{total_simulations} | {sims_per_sec:.2f} sims/s", end="", flush=True)

            except Exception as exc:
                print(f"\n[t={fixed_t}] An unexpected error occurred while retrieving a result: {exc}")

    # Add a final newline to clean up the progress bar line
    print()

    # --- Final Step: Write all collected data for this job to a single file ---
    output_filename = os.path.join(job_log_dir, f"t_{fixed_t}.csv")
    fieldnames = ["J", "h", "theta", "samples"]

    try:
        with open(output_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(job_results)
    except IOError as e:
        print(f"Error writing to file {output_filename}: {e}")
