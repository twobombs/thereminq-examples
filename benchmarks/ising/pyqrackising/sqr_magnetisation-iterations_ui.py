# -*- coding: utf-8 -*-
"""
An interactive 3D visualizer for the Transverse Field Ising Model (TFIM)
simulation output from iterations_cli.py.

This is a self-contained, single-file application.

This version uses multiprocessing to run the vedo plotter in a separate
process and a multiprocessing pool to run grid scans in parallel.
It can run a single J-h surface scan, a full Trotter scan over t,
a full Qubit scan over n_qubits, import pre-calculated CSV data, AND
salvage data from log files.

This version is optimized with batch processing for high-performance live rendering.

RUNTIME OPTIMIZATIONS:
- Reuses the multiprocessing.Pool across scan steps to eliminate process creation overhead.
- Sets a sensible maxtasksperchild to keep long-running workers fresh.
- MODIFIED to oversubscribe CPU workers by 10% for experimentation.
"""
import subprocess
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from vedo import Plotter, Points, Mesh, Text2D, ScalarBar
import threading
import queue
import sys
import multiprocessing as mp
import math

# Added for the import and salvage features
import os
import re
import pandas as pd
from collections import defaultdict
from tkinter import filedialog


# --- Simulation Runners ---

def _execute_simulation(params):
    """
    Core function to execute the CLI script and parse the output.
    This is a top-level function so it can be 'pickled' by multiprocessing.
    Returns a tuple: (result_type, data)
    """
    command = [sys.executable, 'sqr_magnetisation-iterations_cli.py']
    for key, value in params.items():
        command.extend([f'--{key}', str(value)])

    try:
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW

        result = subprocess.run(
            command, capture_output=True, text=True, check=True, creationflags=creationflags
        )

        output_lines = result.stdout.strip().split('\n')
        samples_line = ""
        time_line = "Calculation took: N/A"

        try:
            samples_index = output_lines.index("## Output Samples (Decimal Comma-Separated) ##")
            samples_line = output_lines[samples_index + 1]
        except (ValueError, IndexError):
            pass

        for line in output_lines:
            if line.startswith("Calculation took:"):
                time_line = line
                break

        if not samples_line:
            return ('error', "Could not find sample output in simulation results.")

        avg_magnetization = float(samples_line.replace(',', '.'))
        return ('success', (params, (avg_magnetization, time_line)))

    except FileNotFoundError:
        return ('error', "Error: 'sqr_magnetisation-iterations_cli.py' not found.")
    except subprocess.CalledProcessError as e:
        return ('error', f"Simulation failed with error:\n{e.stderr}")
    except Exception as e:
        return ('error', f"An unexpected error occurred: {e}")

def run_simulation_thread(params, result_queue):
    """
    Thread target for running a single simulation.
    """
    result_type, data = _execute_simulation(params)
    result_queue.put((result_type, data))

### NEW: INTEGRATED SALVAGE LOGIC ###
def _salvage_logic_task(result_queue):
    """
    This function contains the full logic for parsing log files and creating CSVs.
    It communicates its progress back to the GUI via the result_queue.
    Returns True on success, False on failure.
    """
    # --- Configuration ---
    LOG_DIRECTORY = './tfim_logs'
    OUTPUT_DIRECTORY = './tfim_results_final'
    # ---------------------

    def parse_log_file(filepath):
        """Parses a single log file to extract simulation data points."""
        line_regex = re.compile(
            r"Params:.*?{'J': ([-0-9.]+), 'h': ([-0-9.]+), 'z': ([-0-9.]+), 'theta': ([-0-9.]+), 't': ([-0-9.]+), 'n_qubits': ([-0-9.]+)}.*?Result: ([-0-9.]+)"
        )
        extracted_data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    match = line_regex.search(line)
                    if match:
                        try:
                            data_point = {
                                'J': float(match.group(1)),
                                'h': float(match.group(2)),
                                'z': int(match.group(3)),
                                'theta': float(match.group(4)),
                                't': int(match.group(5)),
                                'n_qubits': int(match.group(6)),
                                'samples': float(match.group(7))
                            }
                            extracted_data.append(data_point)
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            print(f"Error reading or parsing {os.path.basename(filepath)}: {e}") # Log to console
        return extracted_data

    # --- Main Salvage Logic ---
    if not os.path.isdir(LOG_DIRECTORY):
        result_queue.put(('error', f"Log directory not found at '{LOG_DIRECTORY}'"))
        return False

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    all_log_files = [f for f in os.listdir(LOG_DIRECTORY) if f.lower().endswith('.log')]

    if not all_log_files:
        result_queue.put(('status', f"No .log files found in '{LOG_DIRECTORY}'. Nothing to do."))
        return True

    result_queue.put(('status', f"Found {len(all_log_files)} log files to process..."))
    
    grouped_data = defaultdict(list)
    total_points_found = 0
    for i, filename in enumerate(all_log_files):
        result_queue.put(('status', f"Processing file {i+1}/{len(all_log_files)}: {filename}"))
        filepath = os.path.join(LOG_DIRECTORY, filename)
        data_from_file = parse_log_file(filepath)
        if data_from_file:
            for point in data_from_file:
                key = (point['n_qubits'], point['t'], point['z'], point['theta'])
                grouped_data[key].append({'J': point['J'], 'h': point['h'], 'samples': point['samples']})
            total_points_found += len(data_from_file)

    if not grouped_data:
        result_queue.put(('status', "Could not find any valid data points in the log files."))
        return True

    result_queue.put(('status', f"Parsed {total_points_found} points, creating {len(grouped_data)} CSV files..."))

    for key, data_points in grouped_data.items():
        n, t, z, theta = key
        output_subdir = os.path.join(OUTPUT_DIRECTORY, f"n{n}_z{z}_theta{theta:.2f}".replace('.', 'p'))
        os.makedirs(output_subdir, exist_ok=True)
        csv_filename = f"TFIM_scan_n{n}_t{t}.csv"
        output_path = os.path.join(output_subdir, csv_filename)
        
        df = pd.DataFrame(data_points)
        df.sort_values(by=['h', 'J'], inplace=True)
        df.to_csv(output_path, index=False, float_format='%.4f')

    result_queue.put(('status', "Log salvage complete. CSVs are ready to be imported."))
    return True

def run_salvage_worker(result_queue):
    """
    Thread worker to execute the in-process log salvage logic.
    """
    try:
        # Call the internal function instead of an external script
        _salvage_logic_task(result_queue)
    except Exception as e:
        error_msg = f"A critical error occurred during the salvage operation: {e}"
        result_queue.put(('error', error_msg))
#######################################

def run_import_worker(list_of_directories, result_queue):
    """
    Thread worker to import and render CSV files from a LIST of directories.
    This version is OPTIMIZED to avoid slow iteration for much faster loading.
    """
    try:
        total_dirs = len(list_of_directories)
        for dir_idx, directory_path in enumerate(list_of_directories):
            dir_name = os.path.basename(directory_path)
            
            # Sort files for deterministic order
            csv_files = sorted([f for f in os.listdir(directory_path) if f.lower().endswith('.csv')])
            if not csv_files:
                print(f"INFO: No CSV files found in '{dir_name}', skipping.")
                continue

            total_files_in_dir = len(csv_files)
            for i, filename in enumerate(csv_files):
                status_msg = f"Dir {dir_idx+1}/{total_dirs} | File {i+1}/{total_files_in_dir}: {filename}"
                result_queue.put(('status', status_msg))
                
                filepath = os.path.join(directory_path, filename)
                df = pd.read_csv(filepath)

                # Clean NaN values
                initial_rows = len(df)
                df.dropna(subset=['J', 'h', 'samples'], inplace=True)
                if len(df) < initial_rows:
                    print(f"INFO [{filename}]: Ignored {initial_rows - len(df)} rows with NaN values.")

                if not {'J', 'h', 'samples'}.issubset(df.columns):
                    result_queue.put(('error', f"CSV '{filename}' is missing required columns (J, h, samples). Skipping."))
                    continue

                # --- OPTIMIZATION START (Fix 1) ---
                # Avoid df.iterrows()! Get columns as lists for massive speedup.
                j_vals = df['J'].tolist()
                h_vals = df['h'].tolist()
                samples_vals = df['samples'].tolist()
                num_rows = len(df)
                # --- OPTIMIZATION END ---
                
                results_batch = []
                BATCH_SIZE = 2000 # Increased batch size is fine now
                
                # --- OPTIMIZATION START (Fix 1) ---
                # Loop through the lists using an index, which is extremely fast.
                for idx in range(num_rows):
                    params = {'J': j_vals[idx], 'h': h_vals[idx]}
                    avg_magnetization = samples_vals[idx]
                    time_line = "Imported from CSV"
                    results_batch.append((params, (avg_magnetization, time_line)))

                    # Send batch when full
                    if len(results_batch) >= BATCH_SIZE:
                        result_queue.put(('success_batch', results_batch.copy()))
                        results_batch.clear()
                # --- OPTIMIZATION END ---
                
                # Send any remaining data in the last batch
                if results_batch:
                    result_queue.put(('success_batch', results_batch.copy()))

                # Signal that this file is done and ready for saving/clearing
                base_filename = os.path.splitext(filename)[0]
                dataset_name = f"{dir_name}_{base_filename}"
                result_queue.put(('import_slice_complete', dataset_name))

        result_queue.put(('status', f"Full import of {total_dirs} directories complete."))

    except Exception as e:
        error_msg = f"An error occurred during import: {e}"
        print(error_msg)
        result_queue.put(('error', error_msg))

def run_view_worker(list_of_directories, result_queue, continue_event):
    """
    Thread worker to interactively VIEW CSV files one by one.
    Pauses after each file and waits for a signal to continue.
    """
    try:
        total_dirs = len(list_of_directories)
        for dir_idx, directory_path in enumerate(list_of_directories):
            dir_name = os.path.basename(directory_path)
            csv_files = sorted([f for f in os.listdir(directory_path) if f.lower().endswith('.csv')])
            if not csv_files:
                continue

            total_files_in_dir = len(csv_files)
            for i, filename in enumerate(csv_files):
                status_msg = f"Dir {dir_idx+1}/{total_dirs} | Loading {i+1}/{total_files_in_dir}: {filename}"
                result_queue.put(('status', status_msg))
                
                filepath = os.path.join(directory_path, filename)
                df = pd.read_csv(filepath)
                df.dropna(subset=['J', 'h', 'samples'], inplace=True)

                if not {'J', 'h', 'samples'}.issubset(df.columns):
                    continue
                
                j_vals = df['J'].tolist()
                h_vals = df['h'].tolist()
                samples_vals = df['samples'].tolist()
                num_rows = len(df)
                
                results_batch = []
                BATCH_SIZE = 2000
                for idx in range(num_rows):
                    params = {'J': j_vals[idx], 'h': h_vals[idx]}
                    avg_magnetization = samples_vals[idx]
                    time_line = "Imported from CSV"
                    results_batch.append((params, (avg_magnetization, time_line)))

                    if len(results_batch) >= BATCH_SIZE:
                        result_queue.put(('success_batch', results_batch.copy()))
                        results_batch.clear()
                
                if results_batch:
                    result_queue.put(('success_batch', results_batch.copy()))

                # NEW: Signal to render, then wait for the user to right-click in the plot
                base_filename = os.path.splitext(filename)[0]
                dataset_name = f"{dir_name}_{base_filename}"
                result_queue.put(('view_slice_and_wait', dataset_name))
                
                continue_event.clear()
                print(f"Displaying '{dataset_name}'. Waiting for right-click in plot window to continue...")
                result_queue.put(('status', f"Waiting for right-click to view next file..."))
                continue_event.wait() # This blocks until the plotter process sets the event
                
                # After continuing, send a command to clear the plot for the next slice
                result_queue.put(('clear', None))

        result_queue.put(('status', f"Interactive view of {total_dirs} directories complete."))

    except Exception as e:
        error_msg = f"An error occurred during interactive view: {e}"
        print(error_msg)
        result_queue.put(('error', error_msg))

def run_full_scan_worker(fixed_params, result_queue):
    """
    Thread worker to scan the full J-h plane IN PARALLEL.
    This version batches results for much faster rendering.
    """
    j_range = np.arange(-2.0, 2.0 + 0.1, 0.1)
    h_range = np.arange(-4.0, 4.0 + 0.1, 0.1)

    all_params = []
    for h_val in h_range:
        for j_val in j_range:
            current_params = fixed_params.copy()
            current_params['J'] = round(j_val, 2)
            current_params['h'] = round(h_val, 2)
            all_params.append(current_params)
    
    total_points = len(all_params)
    count = 0
    
    results_batch = []
    BATCH_SIZE = 100

    try:
        base_processes = mp.cpu_count()
        num_processes = int(base_processes + math.ceil(base_processes * 0.10))
        print(f"Oversubscribing CPU: Running with {num_processes} worker processes on a {base_processes}-thread system...")
        
        with mp.Pool(processes=num_processes, maxtasksperchild=1500) as pool:
            for result_type, data in pool.imap_unordered(_execute_simulation, all_params):
                count += 1
                status_msg = f"Scanning... {count}/{total_points}"
                result_queue.put(('status', status_msg))

                if result_type == 'success':
                    results_batch.append(data)
                    if len(results_batch) >= BATCH_SIZE:
                        result_queue.put(('success_batch', results_batch.copy()))
                        results_batch.clear()
                else:
                    result_queue.put((result_type, data))
            
            if results_batch:
                result_queue.put(('success_batch', results_batch.copy()))
                results_batch.clear()

        result_queue.put(('status', f"Scan complete. Plotted {total_points} points."))
    except Exception as e:
        error_msg = f"A multiprocessing error occurred: {e}"
        print(error_msg)
        result_queue.put(('error', error_msg))

def run_trotter_scan_worker(fixed_params, result_queue, start_t):
    """
    Thread worker to scan J-h planes for a range of Trotter steps 't'.
    This version batches results and REUSES the process pool for massive speedup.
    """
    end_t = start_t + 20
    trotter_range = range(start_t, end_t)

    j_range = np.arange(-2.0, 2.0 + 0.1, 0.1)
    h_range = np.arange(-4.0, 4.0 + 0.1, 0.1)
    total_planes = len(trotter_range)
    
    results_batch = []
    BATCH_SIZE = 100

    try:
        base_processes = mp.cpu_count()
        num_processes = int(base_processes + math.ceil(base_processes * 0.10))
        
        with mp.Pool(processes=num_processes, maxtasksperchild=1500) as pool:
            print(f"Oversubscribing CPU: Running with {num_processes} worker processes on a {base_processes}-thread system...")
            for i, t_val in enumerate(trotter_range):
                plane_params = fixed_params.copy()
                plane_params['t'] = t_val
                
                status_msg = f"Trotter Scan ({i+1}/{total_planes}): Starting plane for t={t_val}"
                result_queue.put(('status', status_msg))
                print(status_msg)

                all_params = []
                for h_val in h_range:
                    for j_val in j_range:
                        current_params = plane_params.copy()
                        current_params['J'] = round(j_val, 2)
                        current_params['h'] = round(h_val, 2)
                        all_params.append(current_params)
                
                total_points = len(all_params)
                count = 0
                
                for result_type, data in pool.imap_unordered(_execute_simulation, all_params):
                    count += 1
                    status_msg = f"Trotter Scan ({i+1}/{total_planes}, t={t_val}): {count}/{total_points}"
                    result_queue.put(('status', status_msg))
                    if result_type == 'success':
                        results_batch.append(data)
                        if len(results_batch) >= BATCH_SIZE:
                            result_queue.put(('success_batch', results_batch.copy()))
                            results_batch.clear()
                    else:
                        result_queue.put((result_type, data))
                
                if results_batch:
                    result_queue.put(('success_batch', results_batch.copy()))
                    results_batch.clear()

                result_queue.put(('trotter_slice_complete', t_val))

        result_queue.put(('status', "Full Trotter Scan complete."))
    except Exception as e:
        error_msg = f"A Trotter scan error occurred: {e}"
        print(error_msg)
        result_queue.put(('error', error_msg))

def run_qubit_scan_worker(fixed_params, result_queue, start_t, n_qubits_range):
    """
    Top-level worker to scan over N Qubits, containing Trotter scans.
    This version batches results and REUSES the process pool for massive speedup.
    """
    end_t = start_t + 20
    trotter_range = range(start_t, end_t)
    n_qubits_vals = range(int(n_qubits_range[0]), int(n_qubits_range[1]) + 1)

    j_range = np.arange(-2.0, 2.0 + 0.1, 0.1)
    h_range = np.arange(-4.0, 4.0 + 0.1, 0.1)

    total_qubit_steps = len(n_qubits_vals)
    total_trotter_steps = len(trotter_range)
    
    results_batch = []
    BATCH_SIZE = 100

    try:
        base_processes = mp.cpu_count()
        num_processes = int(base_processes + math.ceil(base_processes * 0.10))

        with mp.Pool(processes=num_processes, maxtasksperchild=1500) as pool:
            print(f"Oversubscribing CPU: Running with {num_processes} worker processes on a {base_processes}-thread system...")
            for q_idx, n_qubits_val in enumerate(n_qubits_vals):
                for t_idx, t_val in enumerate(trotter_range):
                    plane_params = fixed_params.copy()
                    plane_params['t'] = t_val
                    plane_params['n_qubits'] = n_qubits_val

                    status_msg = (f"Qubit Scan ({q_idx+1}/{total_qubit_steps}, n={n_qubits_val}) | "
                                  f"Trotter ({t_idx+1}/{total_trotter_steps}, t={t_val})")
                    result_queue.put(('status', status_msg))
                    print(status_msg)

                    all_params = []
                    for h_val in h_range:
                        for j_val in j_range:
                            current_params = plane_params.copy()
                            current_params['J'] = round(j_val, 2)
                            current_params['h'] = round(h_val, 2)
                            all_params.append(current_params)
                    
                    total_points = len(all_params)
                    count = 0
                    
                    for result_type, data in pool.imap_unordered(_execute_simulation, all_params):
                        count += 1
                        status_msg = (f"Qubit Scan ({q_idx+1}/{total_qubit_steps}, n={n_qubits_val}) | "
                                      f"Trotter ({t_idx+1}/{total_trotter_steps}, t={t_val}): {count}/{total_points}")
                        result_queue.put(('status', status_msg))
                        if result_type == 'success':
                            results_batch.append(data)
                            if len(results_batch) >= BATCH_SIZE:
                                result_queue.put(('success_batch', results_batch.copy()))
                                results_batch.clear()
                        else:
                            result_queue.put((result_type, data))
                    
                    if results_batch:
                        result_queue.put(('success_batch', results_batch.copy()))
                        results_batch.clear()
                    
                    result_queue.put(('trotter_slice_complete', (t_val, n_qubits_val)))

        result_queue.put(('status', "Full Qubit/Trotter Scan complete."))
    except Exception as e:
        error_msg = f"A Qubit scan error occurred: {e}"
        print(error_msg)
        result_queue.put(('error', error_msg))

# --- Vedo Plotter Process ---

def vedo_plotter_process(data_queue, param_specs, continue_event):
    """
    This function runs in a separate process to handle all vedo plotting.
    This version uses a throttled recreation approach for stability.
    """
    Z_AXIS_SCALE_FACTOR = 4.0

    j_range = (param_specs['J']['from_'], param_specs['J']['to'])
    h_range = (param_specs['h']['from_'], param_specs['h']['to'])
    z_range = (0.0, 1.0 * Z_AXIS_SCALE_FACTOR)

    plt = Plotter(
        axes=dict(
            xtitle='J (Coupling)', ytitle='h (Transverse Field)', ztitle='Avg. Measured Value',
            zlabel_size=0.02, ylabel_size=0.02, xlabel_size=0.02,
            xrange=j_range, yrange=h_range, zrange=z_range
        ),
        bg='black',
        title="TFIM Phase Visualizer"
    )

    plotted_points_data = []
    scan_points_data = {}
    
    point_cloud_actor = None
    surface_actor = None
    info_text_actor = Text2D("Initializing...", pos='top-left', s=1.1, bg='gray', alpha=0.7)
    scalar_bar_actor = None
    
    # --- FINAL FIX: Throttling parameters ---
    batch_counter = 0
    UPDATE_INTERVAL = 10 # Update the plot every 10 batches

    plt.show(info_text_actor, interactive=False)
    
    def _generate_and_add_surface(fixed_params=None, dataset_name=None):
        nonlocal surface_actor, scalar_bar_actor, point_cloud_actor
        
        if len(scan_points_data) < 4: return
        
        j_coords = sorted(list(set(p[0] for p in scan_points_data.keys())))
        h_coords = sorted(list(set(p[1] for p in scan_points_data.keys())))
        
        if len(j_coords) < 2 or len(h_coords) < 2: return

        vertices = []
        for h in h_coords:
            for j in j_coords:
                z_original = scan_points_data.get((j, h), 0.0)
                vertices.append([j, h, z_original * Z_AXIS_SCALE_FACTOR])
        
        faces = []
        nx, ny = len(j_coords), len(h_coords)
        for i in range(ny - 1):
            for j in range(nx - 1):
                p1, p2 = i * nx + j, i * nx + j + 1
                p3, p4 = (i + 1) * nx + j + 1, (i + 1) * nx + j
                faces.append([p1, p2, p3, p4])

        plt.remove(point_cloud_actor)
        point_cloud_actor = None
        plotted_points_data.clear()
        
        surface_actor = Mesh([vertices, faces]).lighting('glossy')
        z_values_surf = np.array(vertices)[:, 2]
        surface_actor.cmap('viridis', z_values_surf, vmin=0.0, vmax=1.0 * Z_AXIS_SCALE_FACTOR)
        
        plt.remove(scalar_bar_actor)
        scalar_bar_actor = ScalarBar(surface_actor, title="Avg. Measured Value", pos=((0.85, 0.4), (0.9, 0.9)))
        info_text_actor.text(f"Surface plot with {len(vertices)} vertices.\nScan complete.")
        
        plt.add(surface_actor, scalar_bar_actor)
        print("Surface mesh rendered.")

        # Handle saving for non-interactive modes
        if fixed_params:
            param_parts = [f"{key}={value}" for key, value in sorted(fixed_params.items())]
            filename = "TFIM_scan_" + "_".join(param_parts).replace(' ', '_') + ".png"
            plt.screenshot(filename, scale=3)
            print(f"Saved high-resolution screenshot to: {filename}")

    def update_plot(timer_id):
        nonlocal point_cloud_actor, surface_actor, info_text_actor, scalar_bar_actor
        nonlocal batch_counter

        try:
            while not data_queue.empty():
                command, data = data_queue.get_nowait()
                
                if command == 'add_batch':
                    batch_counter += 1
                    last_data_point = None
                    for params, (avg_z, time_str) in data:
                        new_point = (params['J'], params['h'], avg_z * Z_AXIS_SCALE_FACTOR)
                        plotted_points_data.append(new_point)
                        scan_points_data[(params['J'], params['h'])] = avg_z
                        last_data_point = (params, (avg_z, time_str))

                    # --- FINAL FIX: Throttled actor recreation for stability ---
                    if batch_counter % UPDATE_INTERVAL == 0 and plotted_points_data:
                        plt.remove(point_cloud_actor, scalar_bar_actor)

                        points_array = np.array(plotted_points_data)
                        z_values = points_array[:, 2]
                        
                        # Recreate the actor from scratch - this is the most stable method
                        point_cloud_actor = Points(points_array, r=5).cmap('viridis', z_values, vmin=0.0, vmax=1.0 * Z_AXIS_SCALE_FACTOR)
                        scalar_bar_actor = ScalarBar(point_cloud_actor, title="Avg. Measured Value", pos=((0.85, 0.4), (0.9, 0.9)))
                        plt.add(point_cloud_actor, scalar_bar_actor)

                    # Always update the text info so the GUI feels responsive
                    if last_data_point:
                        last_params, (last_avg_z, last_time_str) = last_data_point
                        info_text = (f"Plotted points: {len(plotted_points_data)}\n"
                                     f"Last point: J={last_params['J']:.2f}, h={last_params['h']:.2f}\n"
                                     f"Avg. Value: {last_avg_z:.4f}\n"
                                     f"{last_time_str}")
                        info_text_actor.text(info_text)

                elif command == 'view_slice_and_wait':
                    dataset_name = data
                    _generate_and_add_surface()
                    info_text_actor.text(f"Viewing: {dataset_name}\nRIGHT-CLICK in the window to continue.")
                    
                    def resume_callback(evt):
                        print("Right-click detected. Resuming scan.")
                        continue_event.set()
                        plt.remove_callback('RightButtonPress') # Clean up

                    plt.add_callback('RightButtonPress', resume_callback)

                elif command == 'scan_complete':
                    fixed_params = data
                    # Final render before creating surface
                    if plotted_points_data:
                        plt.remove(point_cloud_actor, scalar_bar_actor)
                        points_array = np.array(plotted_points_data)
                        z_values = points_array[:, 2]
                        point_cloud_actor = Points(points_array, r=5).cmap('viridis', z_values, vmin=0.0, vmax=1.0 * Z_AXIS_SCALE_FACTOR)
                        scalar_bar_actor = ScalarBar(point_cloud_actor, title="Avg. Measured Value", pos=((0.85, 0.4), (0.9, 0.9)))
                        plt.add(point_cloud_actor, scalar_bar_actor)
                        plt.render() # Force one last render of all points

                    _generate_and_add_surface(fixed_params)

                elif command == 'clear':
                    plotted_points_data.clear()
                    scan_points_data.clear()
                    batch_counter = 0
                    
                    plt.remove(point_cloud_actor, surface_actor, scalar_bar_actor)
                    point_cloud_actor, surface_actor, scalar_bar_actor = None, None, None
                    
                    info_text_actor.text("Plot cleared.")
                    plt.reset_camera()

                elif command == 'close':
                    plt.close()
                    return

        except Exception as e:
            import traceback
            print(f"Error in vedo process: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        
        plt.render()

    plt.add_callback('timer', update_plot)
    plt.timer_callback('create', dt=50)
    plt.interactive().close()

# --- GUI Class ---
class ControlPanel:
    def __init__(self, root, param_specs, data_queue, continue_event):
        self.root = root
        self.param_specs = param_specs
        self.data_queue = data_queue
        self.continue_event = continue_event
        self.sim_result_queue = queue.Queue()
        self.param_vars = {}
        self.scan_button, self.trotter_scan_button, self.qubit_scan_button = None, None, None
        self.import_button, self.view_button, self.salvage_button = None, None, None
        self.last_scan_params = {}

        self.root.title("TFIM Control Panel")
        self.root.geometry("400x850") # Increased height for new buttons
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Ready. Move a slider or start a scan.")
        self._create_controls(main_frame)
        
        self.root.after(100, self.check_sim_queue)

    def _create_controls(self, parent):
        slider_frame = ttk.LabelFrame(parent, text="Simulation Parameters", padding=10)
        slider_frame.pack(fill=tk.X, expand=True)
        
        for name, spec in self.param_specs.items():
            frame = ttk.Frame(slider_frame)
            frame.pack(fill=tk.X, pady=5)
            label = ttk.Label(frame, text=f"{name.replace('_', ' ').title()}:", width=12, anchor=tk.W)
            label.pack(side=tk.LEFT)

            var = tk.DoubleVar(value=spec['default'])
            self.param_vars[name] = var

            value_label = ttk.Label(frame, text=f"{var.get():.2f}", width=6)
            value_label.pack(side=tk.RIGHT, padx=(5, 0))

            command = lambda val, l=value_label, v=var, s=spec: l.config(text=f"{v.get():.2f}" if s['resolution'] < 1 else f"{int(v.get())}")

            slider = ttk.Scale(frame, from_=spec['from_'], to=spec['to'], orient=tk.HORIZONTAL, variable=var, command=command)
            slider.pack(fill=tk.X, expand=True)
            slider.bind("<ButtonRelease-1>", self.run_simulation_from_sliders)

        button_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        button_frame.pack(fill=tk.X, pady=20)

        self.scan_button = ttk.Button(button_frame, text="Run Full Scan (J vs h)", command=self.start_full_scan)
        self.scan_button.pack(fill=tk.X, expand=True, padx=5, pady=5, ipady=5)
        
        self.trotter_scan_button = ttk.Button(button_frame, text="Run Full Trotter Scan (t vs J,h)", command=self.start_trotter_scan)
        self.trotter_scan_button.pack(fill=tk.X, expand=True, padx=5, pady=5, ipady=5)

        self.qubit_scan_button = ttk.Button(button_frame, text="Run Full Qubit/Trotter Scan", command=self.start_qubit_scan)
        self.qubit_scan_button.pack(fill=tk.X, expand=True, padx=5, pady=5, ipady=5)
        
        self.import_button = ttk.Button(button_frame, text="Import All Directories", command=self.start_import_scan)
        self.import_button.pack(fill=tk.X, expand=True, padx=5, pady=5, ipady=5)

        self.view_button = ttk.Button(button_frame, text="View All Directories (Interactive)", command=self.start_view_scan)
        self.view_button.pack(fill=tk.X, expand=True, padx=5, pady=5, ipady=5)

        # Separator for visual distinction
        ttk.Separator(button_frame, orient='horizontal').pack(fill='x', pady=10)

        self.salvage_button = ttk.Button(button_frame, text="Salvage Logs to CSV", command=self.start_salvage_process)
        self.salvage_button.pack(fill=tk.X, expand=True, padx=5, pady=5, ipady=5)
        
        clear_button = ttk.Button(button_frame, text="Clear Plot", command=self.clear_plot)
        clear_button.pack(fill=tk.X, expand=True, padx=5, pady=5, ipady=5)

        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

    def run_simulation_from_sliders(self, event=None):
        current_params = {name: var.get() for name, var in self.param_vars.items()}
        for key in ['z', 't', 'n_qubits']:
            if key in current_params:
                current_params[key] = int(current_params[key])
        self.status_var.set("Running single simulation...")
        threading.Thread(target=run_simulation_thread, args=(current_params, self.sim_result_queue), daemon=True).start()
        
    def start_full_scan(self):
        if messagebox.askokcancel("Start Scan?", "This will run many simulations in parallel and render a surface. Continue?"):
            self._disable_buttons()
            self.clear_plot()
            fixed_params = {name: var.get() for name, var in self.param_vars.items()}
            fixed_params.pop('J', None); fixed_params.pop('h', None)
            for key in ['z', 't', 'n_qubits']:
                if key in fixed_params: fixed_params[key] = int(fixed_params[key])
            self.last_scan_params = fixed_params
            threading.Thread(target=run_full_scan_worker, args=(fixed_params, self.sim_result_queue), daemon=True).start()

    def start_trotter_scan(self):
        start_t = int(self.param_vars['t'].get())
        msg = f"This will run a 20-step scan from t={start_t} to t={start_t + 19}.\nThis will take a long time and save 20 images.\n\nContinue?"
        if messagebox.askokcancel("Start Trotter Scan?", msg):
            self._disable_buttons()
            self.clear_plot()
            fixed_params = {name: var.get() for name, var in self.param_vars.items()}
            fixed_params.pop('J', None); fixed_params.pop('h', None); fixed_params.pop('t', None)
            for key in ['z', 'n_qubits']:
                if key in fixed_params: fixed_params[key] = int(fixed_params[key])
            self.last_scan_params = fixed_params
            threading.Thread(target=run_trotter_scan_worker, args=(fixed_params, self.sim_result_queue, start_t), daemon=True).start()
            
    def start_qubit_scan(self):
        start_t = int(self.param_vars['t'].get())
        n_qubits_from = int(self.param_specs['n_qubits']['from_'])
        n_qubits_to = int(self.param_specs['n_qubits']['to'])
        msg = (f"This will run a full Trotter scan (starting from t={start_t}) "
               f"for EVERY qubit count from {n_qubits_from} to {n_qubits_to}.\n\n"
               "THIS WILL TAKE AN EXTREMELY LONG TIME AND GENERATE HUNDREDS OF IMAGES.\n\nContinue?")
        if messagebox.askokcancel("Start Full Qubit Scan?", msg, icon='warning'):
            self._disable_buttons()
            self.clear_plot()
            fixed_params = {name: var.get() for name, var in self.param_vars.items()}
            fixed_params.pop('J', None); fixed_params.pop('h', None); fixed_params.pop('t', None); fixed_params.pop('n_qubits', None)
            for key in ['z']:
                if key in fixed_params: fixed_params[key] = int(fixed_params[key])
            self.last_scan_params = fixed_params
            n_qubits_range = (n_qubits_from, n_qubits_to)
            threading.Thread(target=run_qubit_scan_worker, args=(fixed_params, self.sim_result_queue, start_t, n_qubits_range), daemon=True).start()
    
    def start_import_scan(self):
        """Callback for the 'Import All' button. Scans ALL subdirectories automatically."""
        base_path = './tfim_results_final'
        if not os.path.isdir(base_path):
            messagebox.showerror("Directory Not Found", f"The base directory '{base_path}' does not exist.")
            return

        try:
            subdirectories = sorted([os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
            if not subdirectories:
                messagebox.showinfo("No Directories Found", f"No subdirectories found to scan in '{base_path}'.")
                return
        except Exception as e:
            messagebox.showerror("Error Reading Directories", f"Could not read subdirectories from '{base_path}':\n{e}")
            return
            
        msg = (f"This will scan {len(subdirectories)} directories and render every CSV file found within them.\n\n"
               "This may generate a large number of images. Continue?")
        
        if messagebox.askokcancel("Start Full Import Scan?", msg):
            self._disable_buttons()
            self.clear_plot()
            self.status_var.set(f"Starting full import scan of {len(subdirectories)} directories...")
            threading.Thread(target=run_import_worker, args=(subdirectories, self.sim_result_queue), daemon=True).start()

    def start_view_scan(self):
        """Callback for the 'View All' button. Scans and pauses for each file."""
        base_path = './tfim_results_final'
        if not os.path.isdir(base_path):
            messagebox.showerror("Directory Not Found", f"The base directory '{base_path}' does not exist.")
            return

        try:
            subdirectories = sorted([os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
            if not subdirectories:
                messagebox.showinfo("No Directories Found", f"No subdirectories found to scan in '{base_path}'.")
                return
        except Exception as e:
            messagebox.showerror("Error Reading Directories", f"Could not read subdirectories from '{base_path}':\n{e}")
            return
            
        msg = (f"This will scan {len(subdirectories)} directories and render each CSV file one by one.\n\n"
               "You must RIGHT-CLICK in the plot window to advance to the next file.\n\nContinue?")
        
        if messagebox.askokcancel("Start Interactive View Scan?", msg):
            self._disable_buttons()
            self.clear_plot()
            self.status_var.set(f"Starting interactive view of {len(subdirectories)} directories...")
            threading.Thread(target=run_view_worker, args=(subdirectories, self.sim_result_queue, self.continue_event), daemon=True).start()

    def start_salvage_process(self):
        """Callback for the 'Salvage Logs' button."""
        msg = ("This will process raw log files from './tfim_logs' into clean CSVs "
               "in './tfim_results_final'. This is used to recover data from an "
               "incomplete or crashed simulation.\n\nContinue?")
        if messagebox.askokcancel("Start Log Salvage?", msg):
            self._disable_buttons()
            self.status_var.set("Starting log salvage process...")
            # Run the salvage worker in a separate thread
            threading.Thread(target=run_salvage_worker, args=(self.sim_result_queue,), daemon=True).start()

    def _disable_buttons(self):
        self.scan_button.config(state=tk.DISABLED)
        self.trotter_scan_button.config(state=tk.DISABLED)
        self.qubit_scan_button.config(state=tk.DISABLED)
        self.import_button.config(state=tk.DISABLED)
        self.view_button.config(state=tk.DISABLED)
        self.salvage_button.config(state=tk.DISABLED)

    def _enable_buttons(self):
        self.scan_button.config(state=tk.NORMAL)
        self.trotter_scan_button.config(state=tk.NORMAL)
        self.qubit_scan_button.config(state=tk.NORMAL)
        self.import_button.config(state=tk.NORMAL)
        self.view_button.config(state=tk.NORMAL)
        self.salvage_button.config(state=tk.NORMAL)

    def check_sim_queue(self):
        """Checks the queue for results from the simulation threads/processes."""
        try:
            result_type, data = self.sim_result_queue.get_nowait()

            if result_type == 'success_batch':
                self.data_queue.put(('add_batch', data))
            
            elif result_type == 'success':
                self.data_queue.put(('add_batch', [data]))

            elif result_type == 'error':
                self.status_var.set("Error: See message box for details.")
                self._enable_buttons()
                messagebox.showerror("Operation Error", data)
            
            elif result_type == 'status':
                self.status_var.set(data)
                if "complete" in data.lower() or "nothing to do" in data.lower():
                    self._enable_buttons()
                    # This handles the original "full scan" which terminates on a status message
                    if "scan complete" in data.lower():
                        self.data_queue.put(('scan_complete', self.last_scan_params))
            
            elif result_type == 'trotter_slice_complete':
                slice_params = self.last_scan_params.copy()
                if isinstance(data, tuple):
                    t_value, n_qubits_value = data
                    self.status_var.set(f"Slice for t={t_value}, n={n_qubits_value} complete. Saving.")
                    slice_params['t'], slice_params['n_qubits'] = int(t_value), int(n_qubits_value)
                else:
                    t_value = data
                    self.status_var.set(f"Trotter slice for t={t_value} complete. Saving image.")
                    slice_params['t'] = int(t_value)
                self.data_queue.put(('scan_complete', slice_params))
                self.data_queue.put(('clear', None))

            elif result_type == 'import_slice_complete':
                dataset_name = data
                self.status_var.set(f"Image for {dataset_name} complete. Saving.")
                slice_params = {'dataset': dataset_name}
                self.data_queue.put(('scan_complete', slice_params))
                self.data_queue.put(('clear', None))
            
            elif result_type == 'view_slice_and_wait':
                # This is a command from the worker to the plotter, just pass it through
                self.data_queue.put((result_type, data))
                
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_sim_queue)

    def clear_plot(self):
        self.data_queue.put(('clear', None))
        self.status_var.set("Plot cleared. Ready.")

    def on_closing(self):
        print("Closing application...")
        self.data_queue.put(('close', None))
        self.root.destroy()

# --- Main Execution ---
if __name__ == '__main__':
    mp.freeze_support()

    param_specs = {
        'J': {'from_': -2.0, 'to': 2.0, 'resolution': 0.1, 'default': -1.0},
        'h': {'from_': -4.0, 'to': 4.0, 'resolution': 0.1, 'default': 2.0},
        'z': {'from_': -8, 'to': 8, 'resolution': 1, 'default': 4},
        'theta': {'from_': -np.pi / 2, 'to': np.pi / 2, 'resolution': 0.01, 'default': 0.1745},
        't': {'from_': 1, 'to': 50, 'resolution': 1, 'default': 20},
        'n_qubits': {'from_': 4, 'to': 960, 'resolution': 1, 'default': 56},
    }
    
    data_queue = mp.Queue()
    continue_event = mp.Event() # Event for pause/resume functionality
    
    plot_process = mp.Process(target=vedo_plotter_process, args=(data_queue, param_specs, continue_event), daemon=True)
    plot_process.start()

    root = tk.Tk()
    app = ControlPanel(root, param_specs, data_queue, continue_event)
    root.mainloop()

    plot_process.join(timeout=2)
    if plot_process.is_alive():
        print("Terminating plotter process...")
        plot_process.terminate()
        
    print("Application closed.")
    
