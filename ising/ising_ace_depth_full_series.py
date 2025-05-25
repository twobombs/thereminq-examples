# original by Dan Strano https://github.com/vm6502q/pyqrack-examples/blob/main/ising/ising_ace_depth_series.py
# modified by gemini25 for automated iteration; tested settings can be found in ising_ace_depth_full_series.sh
# GPL3 

# Ising model Trotterization (modified for width and depth iteration)
# You likely want to specify environment variable QRACK_MAX_PAGING_QB=28

import math
import sys
import time

# from collections import Counter # Counter is not used

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
# RZZGate and RXGate are used within trotter_step via circ.rx and circ.append(RZZGate(...))
from qiskit.circuit.library import RZZGate # RXGate is implicitly used by qc.rx

from qiskit.compiler import transpile

from pyqrack import QrackAceBackend


def factor_width(width, reverse=False):
    if width <= 0:
        raise Exception("ERROR: Width must be positive.")
    col_len = math.floor(math.sqrt(width))
    while col_len > 0: # Ensure col_len does not become zero or negative in the loop condition
        if (width % col_len) == 0: # Found a factor
            break
        col_len -= 1
    
    if col_len == 0 : # Should not happen if width is positive, means loop completed without break
         raise Exception(f"ERROR: Can't factor width {width} appropriately, col_len reached 0 (unexpected).")
    if col_len == 1 and width > 1: # If width is > 1 and largest factor <= sqrt(width) is 1, it's prime (or 1)
        raise Exception(f"ERROR: Width {width} is prime or has no suitable 2D factors other than 1 x width for this factorization method.")
    if width == 1 and col_len == 1: # Handle width 1 separately if needed, though loop starts at 4
        pass # Or raise specific exception if 1x1 is not desired.

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

    # Layer 1: horizontal pairs (even rows for even columns, or all if n_cols is small)
    horiz_pairs1 = [
        (r * n_cols + c, r * n_cols + (c + 1))
        for r in range(n_rows)
        for c in range(0, n_cols -1 , 2) # Ensure c+1 is within bounds
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
        for r in range(0, n_rows -1 , 2) # Iterate up to n_rows-2, so r+1 is n_rows-1 at most (before modulo)
        for c in range(n_cols)
    ]
    # Handle cases where n_rows might be small or odd, ensuring all connections
    # The original code might need more robust pairing for general n_rows, n_cols
    # This implementation assumes periodic boundaries are implicitly handled by RZZ layers if all pairs are covered.
    # The modulo operator on ((r + 1) % n_rows) handles periodic connection for vertical pairs.
    if n_rows > 1: add_rzz_pairs(vert_pairs1)


    # Layer 4: vertical pairs (odd columns for odd rows)
    vert_pairs2 = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c) # Periodic in row direction
        for r in range(1, n_rows - 1, 2)
        for c in range(n_cols)
    ]
    if n_rows > 1: add_rzz_pairs(vert_pairs2)
    
    # Note: For full nearest-neighbor coverage on a general 2D lattice with periodic boundary conditions (torus),
    # the pairing logic might need to be more exhaustive or ensure that all unique edges are covered
    # across the different layers. The current layering is based on the original script's pattern.

    # Second half of transverse field term
    for q in qubits:
        circ.rx(h * dt / 2, q)

    return circ


def main():
    # Default values
    max_n_qubits_level_default = 56 # Original n_qubits
    max_trotter_depth_level_default = 20 # Original depth
    shots_default = 1048576
    reverse_default = False

    # Argument parsing
    max_n_qubits_level = int(sys.argv[1]) if len(sys.argv) > 1 else max_n_qubits_level_default
    max_trotter_depth_level = int(sys.argv[2]) if len(sys.argv) > 2 else max_trotter_depth_level_default
    shots = int(sys.argv[3]) if len(sys.argv) > 3 else shots_default
    reverse = (sys.argv[4] not in ["0", "False", "false"]) if len(sys.argv) > 4 else reverse_default

    print(f"Running simulation with max width up to {max_n_qubits_level} qubits.")
    print(f"Running simulation with max Trotter depth up to {max_trotter_depth_level} steps.")
    print(f"Number of shots per measurement: {shots}")
    print(f"Reverse factorization: {reverse}")

    J, h, dt = -1.0, 2.0, 0.25  # System parameters
    theta = -math.pi / 6       # Initial Y rotation angle

    # BASIS GATES MODIFIED HERE: "rzz" is removed
    basis_gates = [
        "rx", "ry", "rz", "h", "x", "y", "z", "sx", "sxdg",
        "s", "sdg", "t", "tdg", "cx", "cy", "cz", "swap", "iswap"
    ]

    overall_start_time = time.perf_counter()
    
    # Store data for plotting: {width: ([depths], [magnetizations])}
    plot_data = {}
    all_results_log = [] # For detailed logging

    for current_width in range(4, max_n_qubits_level + 1):
        print(f"\n--- Processing Width: {current_width} ---")
        current_width_start_time = time.perf_counter()
        
        try:
            n_rows, n_cols = factor_width(current_width, reverse)
            print(f"  Factorized into {n_rows} rows x {n_cols} cols.")
            if n_rows * n_cols != current_width:
                 print(f"  Warning: Factorization {n_rows}x{n_cols} does not equal {current_width}. Skipping.")
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
        
        # Transpile the unit step once per width
        try:
            # Setting optimization_level=0 or 1 might be necessary if higher levels cause issues or are too slow.
            # Level 1 is a good default for light optimization.
            transpiled_step_unit = transpile(step_circuit_unit, basis_gates=basis_gates, optimization_level=1) 
        except Exception as e:
            print(f"  Error transpiling step circuit for width {current_width}: {e}. Skipping this width.")
            continue


        experiment = QrackAceBackend(current_width)
        experiment.run_qiskit_circuit(qc_initial_state) # Set initial state

        depths_for_plot_this_width = []
        magnetizations_for_plot_this_width = []

        for d_step_count in range(1, max_trotter_depth_level + 1):
            loop_iter_start_time = time.perf_counter()
            try:
                experiment.run_qiskit_circuit(transpiled_step_unit) # Apply one more Trotter step
            except Exception as e:
                print(f"    Error running transpiled step for width {current_width}, depth {d_step_count}: {e}")
                print(f"    Transpiled step circuit details: {transpiled_step_unit.count_ops()}")
                break # Stop processing this depth sequence for this width


            if d_step_count >= 4: # Collect data if Trotter depth is 4 or more
                try:
                    experiment_samples = experiment.measure_shots(list(range(current_width)), shots)
                except Exception as e:
                    print(f"    Error during measure_shots for width {current_width}, depth {d_step_count}: {e}")
                    break # Stop processing this depth sequence for this width

                magnetization = 0
                for sample_val in experiment_samples:
                    # Correctly calculate magnetization for current_width
                    # Each bit in sample_val corresponds to a qubit's state (0 or 1)
                    # We map 0 to spin +1 and 1 to spin -1
                    for i in range(current_width):
                        if (sample_val >> i) & 1: # if the i-th bit is 1
                            magnetization -= 1 # spin down
                        else:
                            magnetization += 1 # spin up
                
                magnetization /= (shots * current_width) # Average magnetization per site

                iter_time_taken = time.perf_counter() - loop_iter_start_time
                total_time_for_width_so_far = time.perf_counter() - current_width_start_time

                current_result = {
                    "width": current_width,
                    "depth": d_step_count,
                    "magnetization": magnetization,
                    'iter_seconds': round(iter_time_taken, 4),
                    'total_width_seconds': round(total_time_for_width_so_far, 2)
                }
                all_results_log.append(current_result)
                print(f"  {current_result}")

                depths_for_plot_this_width.append(d_step_count)
                magnetizations_for_plot_this_width.append(magnetization)
        
        if depths_for_plot_this_width: # If any data was collected for this width
            plot_data[current_width] = (depths_for_plot_this_width, magnetizations_for_plot_this_width)
        
        del experiment # Release Qrack backend for this width

    print(f"\nTotal simulation time: {time.perf_counter() - overall_start_time:.2f} seconds.")

    # --- Plotting ---
    if not plot_data:
        print("\nNo data available for plotting. Ensure max_trotter_depth_level >= 4.")
        return 0

    plt.figure(figsize=(15, 10)) # Adjusted figure size
    
    min_mag_overall = float('inf')
    max_mag_overall = float('-inf')

    for width_val, (depths, magnetizations) in plot_data.items():
        plt.plot(depths, magnetizations, marker='o', linestyle='-', label=f'{width_val} Qubits')
        if magnetizations: # Check if magnetizations list is not empty
            min_val_this_line = min(magnetizations)
            max_val_this_line = max(magnetizations)
            if min_val_this_line < min_mag_overall:
                min_mag_overall = min_val_this_line
            if max_val_this_line > max_mag_overall:
                max_mag_overall = max_val_this_line


    plt.title(f"Magnetization vs Trotter Depth\n(J={J}, h={h}, dt={dt}, shots={shots})")
    plt.xlabel("Trotter Depth (steps)")
    plt.ylabel("Average Magnetization per Site")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5) # Added more grid details
    plt.legend(loc='best')

    all_depths_plotted = sorted(list(set(d for width_data in plot_data.values() for d in width_data[0])))
    if all_depths_plotted:
        if len(all_depths_plotted) <= 20 : 
             plt.xticks(all_depths_plotted)
        else: # If too many, select a reasonable number of ticks or let matplotlib auto-decide
            # For example, show up to 10 ticks if many points
            step = max(1, len(all_depths_plotted) // 10)
            plt.xticks(all_depths_plotted[::step])


    if min_mag_overall != float('inf') and max_mag_overall != float('-inf'):
        padding = (max_mag_overall - min_mag_overall) * 0.05 
        if padding == 0: padding = 0.1 
        plt.ylim(min_mag_overall - padding, max_mag_overall + padding)
    
    plt.tight_layout() 
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())

