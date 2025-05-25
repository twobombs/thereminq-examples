# original by Dan Strano https://github.com/vm6502q/pyqrack-examples/blob/main/ising/ising_ace_depth_series.py
# modified by gemini25 for automated iteration; tested settings can be found in ising_ace_depth_full_series.sh
# Further modified by Gemini (Bard) for parallel depth iterations and plotting fixes.
# GPL3

# Ising model Trotterization (modified for width and depth iteration)
# You likely want to specify environment variable QRACK_MAX_PAGING_QB=28

import math
import sys
import time
import os # For os.cpu_count()
import concurrent.futures

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate # RXGate is implicitly used by qc.rx
from qiskit.compiler import transpile

from pyqrack import QrackAceBackend


def factor_width(width, reverse=False):
    if width <= 0:
        raise Exception("ERROR: Width must be positive.")
    col_len = math.floor(math.sqrt(width))
    while col_len > 0:
        if (width % col_len) == 0:
            break
        col_len -= 1
    
    if col_len == 0 :
        raise Exception(f"ERROR: Can't factor width {width} appropriately, col_len reached 0 (unexpected).")
    # Allow 1xN or Nx1 factorization, useful for linear chains or specific tests
    # The original script would raise an error for prime widths.
    # If true 2D structure is strictly required and 1xN is not desired, the original check can be reinstated:
    # if col_len == 1 and width > 1: 
    #     raise Exception(f"ERROR: Width {width} is prime or has no suitable 2D factors other than 1 x width for this factorization method.")
    if width == 1 and col_len == 1:
        pass # Handles width 1

    row_len = width // col_len
    return (col_len, row_len) if reverse else (row_len, col_len)


def trotter_step(circ, qubits, lattice_shape, J, h, dt):
    n_rows, n_cols = lattice_shape

    # First half of transverse field term
    for q in qubits:
        circ.rx(h * dt / 2, q)

    # Layered RZZ interactions (simulate 2D nearest-neighbor coupling)
    def add_rzz_pairs(pairs):
        for q1, q2 in pairs:
            circ.append(RZZGate(2 * J * dt), [q1, q2])

    # Layer 1: horizontal pairs (even columns for even rows, or all if n_cols is small)
    horiz_pairs1 = [
        (r * n_cols + c, r * n_cols + (c + 1))
        for r in range(n_rows)
        for c in range(0, n_cols - 1, 2) # Ensure c+1 is within bounds
    ]
    if n_cols > 1: add_rzz_pairs(horiz_pairs1)


    # Layer 2: horizontal pairs (odd rows for odd columns)
    horiz_pairs2 = [
        (r * n_cols + c, r * n_cols + (c + 1))
        for r in range(n_rows)
        for c in range(1, n_cols - 1, 2) # Ensure c+1 is within bounds
    ]
    if n_cols > 1 : add_rzz_pairs(horiz_pairs2)


    # Layer 3: vertical pairs (even columns for even rows)
    vert_pairs1 = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c) # Periodic in row direction for last row
        for r in range(0, n_rows - 1 if n_rows > 1 else 0, 2) # Iterate up to n_rows-2 for r+1
        for c in range(n_cols)
    ]
    if n_rows == 1 and n_cols > 0 : # Handle single row case, no vertical pairs needed.
         pass
    elif n_rows > 1: add_rzz_pairs(vert_pairs1)


    # Layer 4: vertical pairs (odd columns for odd rows)
    vert_pairs2 = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c) # Periodic in row direction
        for r in range(1, n_rows - 1 if n_rows > 1 else 0, 2)
        for c in range(n_cols)
    ]
    if n_rows == 1 and n_cols > 0 : # Handle single row case
        pass
    elif n_rows > 1: add_rzz_pairs(vert_pairs2)
    
    # Second half of transverse field term
    for q in qubits:
        circ.rx(h * dt / 2, q)

    return circ

# This function will be executed in parallel for each depth point
def simulate_single_depth_point(
    width,
    d_step_count,
    qc_initial_state_qiskit,  # Qiskit QuantumCircuit object
    transpiled_step_unit_qiskit, # Qiskit QuantumCircuit object
    shots,
    # J, h, dt are global-like in the main script's context;
    # magnetization calculation here does not directly use them.
):
    """
    Simulates the system for a specific width and number of Trotter steps.
    Returns a dictionary containing results for this point.
    """
    task_start_time = time.perf_counter()
    local_experiment = None
    magnetization_val = None
    error_message = None

    try:
        local_experiment = QrackAceBackend(width)
        local_experiment.run_qiskit_circuit(qc_initial_state_qiskit)

        for _ in range(d_step_count):
            local_experiment.run_qiskit_circuit(transpiled_step_unit_qiskit)

        if d_step_count >= 4: # Condition for measurement
            experiment_samples = local_experiment.measure_shots(list(range(width)), shots)
            current_mag_sum = 0
            for sample_val in experiment_samples:
                for i in range(width):
                    if (sample_val >> i) & 1: # if the i-th bit is 1
                        current_mag_sum -= 1 # spin down
                    else:
                        current_mag_sum += 1 # spin up
            magnetization_val = current_mag_sum / (shots * width) # Average magnetization per site

    except Exception as e:
        error_message = str(e)
    finally:
        if local_experiment:
            del local_experiment # Ensure Qrack backend is released

    task_time_taken = time.perf_counter() - task_start_time
    return {
        "width": width,
        "depth": d_step_count,
        "magnetization": magnetization_val, # Will be None if d_step_count < 4 or error
        'task_seconds': round(task_time_taken, 4),
        "error": error_message
    }


def main():
    # Default values
    max_n_qubits_level_default = 56
    max_trotter_depth_level_default = 20
    shots_default = 1048576 # 2^20
    reverse_default = False
    try:
        num_parallel_workers_default = os.cpu_count() or 1
    except NotImplementedError:
        num_parallel_workers_default = 4 # Fallback

    # Argument parsing
    max_n_qubits_level = int(sys.argv[1]) if len(sys.argv) > 1 else max_n_qubits_level_default
    max_trotter_depth_level = int(sys.argv[2]) if len(sys.argv) > 2 else max_trotter_depth_level_default
    shots = int(sys.argv[3]) if len(sys.argv) > 3 else shots_default
    reverse_arg = sys.argv[4].lower() if len(sys.argv) > 4 else "false"
    reverse = reverse_arg not in ["0", "false", ""]
    num_parallel_workers = int(sys.argv[5]) if len(sys.argv) > 5 else num_parallel_workers_default


    print(f"Running simulation with max width up to {max_n_qubits_level} qubits.")
    print(f"Running simulation with max Trotter depth up to {max_trotter_depth_level} steps.")
    print(f"Number of shots per measurement: {shots}")
    print(f"Reverse factorization: {reverse}")
    print(f"Number of parallel workers for depth calculations: {num_parallel_workers}")
    print(f"WARNING: Parallel depth execution significantly increases total computation compared to sequential evolution.")


    J, h, dt = -1.0, 2.0, 0.25
    theta = -math.pi / 6

    basis_gates = [
        "rx", "ry", "rz", "h", "x", "y", "z", "sx", "sxdg",
        "s", "sdg", "t", "tdg", "cx", "cy", "cz", "swap", "iswap"
    ]

    overall_start_time = time.perf_counter()
    
    plot_data = {} # Stores data for plotting: {width: ([depths], [magnetizations])}
    all_results_log = [] # For detailed logging of each simulation point

    for current_width in range(4, max_n_qubits_level + 1):
        print(f"\n--- Processing Width: {current_width} ---")
        current_width_start_time = time.perf_counter()
        
        n_rows, n_cols = -1,-1
        try:
            n_rows, n_cols = factor_width(current_width, reverse)
            print(f"  Factorized into {n_rows} rows x {n_cols} cols.")
            if n_rows * n_cols != current_width: # Should be caught by factor_width's logic for primes
                print(f"  Warning: Factorization {n_rows}x{n_cols} ({n_rows*n_cols}) does not equal {current_width}. Skipping.")
                continue
        except Exception as e:
            print(f"  Skipping width {current_width}: {e}")
            continue

        # Initial state circuit for the current width
        qc_initial_state = QuantumCircuit(current_width, name=f"Init_W{current_width}")
        for q_idx in range(current_width):
            qc_initial_state.ry(theta, q_idx)

        # Base Trotter step circuit for the current width
        step_circuit_unit = QuantumCircuit(current_width, name=f"TrotterUnit_W{current_width}")
        trotter_step(step_circuit_unit, list(range(current_width)), (n_rows, n_cols), J, h, dt)
        
        transpiled_step_unit = None
        try:
            # Setting optimization_level=0 or 1 might be necessary if higher levels cause issues or are too slow.
            transpiled_step_unit = transpile(step_circuit_unit, basis_gates=basis_gates, optimization_level=1)
        except Exception as e:
            print(f"  Error transpiling step circuit for width {current_width}: {e}. Skipping this width.")
            continue
        
        # --- Parallel execution of depth iterations ---
        depth_results_for_this_width = [] # Temp list to store results for current width
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_parallel_workers) as executor:
            futures = []
            for d_step in range(1, max_trotter_depth_level + 1):
                futures.append(executor.submit(
                    simulate_single_depth_point,
                    current_width,
                    d_step,
                    qc_initial_state, # Passed by value (pickled)
                    transpiled_step_unit, # Passed by value (pickled)
                    shots
                ))

            print(f"  Submitted {len(futures)} depth tasks for width {current_width} to {num_parallel_workers} workers.")
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result() # This is the dict from simulate_single_depth_point
                    depth_results_for_this_width.append(result)
                    # Log individual results as they complete
                    progress_indicator = f"({i+1}/{len(futures)})"
                    if result["error"]:
                        print(f"    Completed W{result['width']} D{result['depth']}: Error - {result['error']} (TaskTime: {result['task_seconds']:.2f}s) {progress_indicator}")
                    elif result["magnetization"] is not None: # Depth >= 4 and no error
                        print(f"    Completed W{result['width']} D{result['depth']}: Mag={result['magnetization']:.6f} (TaskTime: {result['task_seconds']:.2f}s) {progress_indicator}")
                    else: # Depth < 4 and no error (no measurement taken)
                        print(f"    Completed W{result['width']} D{result['depth']}: No measurement (TaskTime: {result['task_seconds']:.2f}s) {progress_indicator}")

                except Exception as exc: # Should be rare if worker catches its own errors
                    # This catches errors in future.result() itself, e.g., if the task submission failed badly.
                    print(f"    Critical error retrieving result for a depth task (width {current_width}): {exc} ({i+1}/{len(futures)})")
        
        # Sort results by depth for consistent logging and plotting
        depth_results_for_this_width.sort(key=lambda r: r["depth"])

        # Aggregate results for plotting and detailed logging
        current_width_total_time = time.perf_counter() - current_width_start_time
        
        temp_depths_for_plot = []
        temp_mags_for_plot = []

        for res in depth_results_for_this_width:
            log_entry = {
                "width": res["width"],
                "depth": res["depth"],
                "magnetization": res["magnetization"], # can be None
                'task_seconds': res['task_seconds'],  # Time for this specific depth point calculation
                'error': res['error'] # can be None
            }
            all_results_log.append(log_entry)

            if res["magnetization"] is not None and not res["error"]: # Ensure depth >= 4, data exists, and no error
                temp_depths_for_plot.append(res["depth"])
                temp_mags_for_plot.append(res["magnetization"])
        
        if temp_depths_for_plot: # If any valid data was collected for plotting for this width
            plot_data[current_width] = (temp_depths_for_plot, temp_mags_for_plot)
        
        print(f"  --- Width {current_width} processed in {current_width_total_time:.2f} seconds ---")

    print(f"\nTotal simulation time: {time.perf_counter() - overall_start_time:.2f} seconds.")

    # --- Plotting ---
    if not plot_data:
        print("\nNo data available for plotting. Ensure max_trotter_depth_level >= 4 and simulations were successful.")
        return 0

    plt.figure(figsize=(15, 10))
    
    min_mag_overall = float('inf')
    max_mag_overall = float('-inf')

    for width_val, (depths, magnetizations) in plot_data.items():
        if depths and magnetizations: # Ensure there's data to plot for this width
            plt.plot(depths, magnetizations, marker='o', linestyle='-', label=f'{width_val} Qubits')
            # Calculate overall min/max magnetization for Y-axis scaling
            min_val_this_line = min(magnetizations)
            max_val_this_line = max(magnetizations)
            if min_val_this_line < min_mag_overall:
                min_mag_overall = min_val_this_line
            if max_val_this_line > max_mag_overall:
                max_mag_overall = max_val_this_line

    plt.title(f"Magnetization vs Trotter Depth\n(J={J}, h={h}, dt={dt}, shots={shots}, Workers={num_parallel_workers})")
    plt.xlabel("Trotter Depth (steps)")
    plt.ylabel("Average Magnetization per Site")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc='best')

    # Determine the actual range of unique integer depths plotted for x-axis formatting
    all_unique_depths_plotted = []
    if plot_data:
        temp_set = set()
        for depths_for_one_width, _ in plot_data.values(): # depths_for_one_width is a list of integers
            if depths_for_one_width: # Ensure the list is not empty
                temp_set.update(depths_for_one_width)
        if temp_set:
            all_unique_depths_plotted = sorted(list(temp_set))
    
    if all_unique_depths_plotted:
        min_d_plotted = all_unique_depths_plotted[0]
        max_d_plotted = all_unique_depths_plotted[-1]

        # Set x-limits to give a bit of space around integer data points
        plt.xlim(min_d_plotted - 0.5, max_d_plotted + 0.5)

        # Determine x-ticks based on the plotted depths
        tick_values = []
        if len(all_unique_depths_plotted) <= 20: # If 20 or fewer unique depths, show all as ticks
            tick_values = all_unique_depths_plotted
        else: # Otherwise, select a subset of ticks
            num_target_ticks = 15 # Aim for around 15 ticks
            stride = int(max(1, math.ceil(len(all_unique_depths_plotted) / num_target_ticks)))
            tick_values = all_unique_depths_plotted[::stride]
            # Ensure the last major point (max_d_plotted) is included if striding might have missed it
            if max_d_plotted not in tick_values:
                 # Add max_d_plotted if it's not already the last tick (can happen if stride perfectly lands before it)
                if not tick_values or max_d_plotted != tick_values[-1]:
                    tick_values.append(max_d_plotted)
                    tick_values = sorted(list(set(tick_values))) # Re-sort and unique

        if tick_values: # Ensure there are ticks to set
            plt.xticks(ticks=tick_values, labels=[str(int(tv)) for tv in tick_values])
    else:
        # Fallback for x-axis if no data was plotted (though script should exit earlier)
        pass

    # Y-limit calculation based on overall min/max magnetization
    if min_mag_overall != float('inf') and max_mag_overall != float('-inf'):
        padding = (max_mag_overall - min_mag_overall) * 0.05
        if padding == 0: padding = 0.1 # Ensure padding is not zero if all values are the same
        plt.ylim(min_mag_overall - padding, max_mag_overall + padding)
    
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    # Guard for ProcessPoolExecutor, especially important on Windows
    sys.exit(main())
