#!/usr/bin/env python3
# Out-of-Time-Order Correlator (OTOC) simulation using PyQrack QBDD backend
# Uses QBDD's get_unitary_fidelity method to calculate overlap.
# Includes pyopencl for automatic GPU detection (no fallback).
# Simulates a 2D lattice.
# Aryan Blaauw & Gemini, Oct 2025

import math
import time
from pyqrack import QrackSimulator
import multiprocessing
from itertools import cycle
# Add imports for suppressing C++ stdout/stderr
import os
import sys
import contextlib
# import re # <-- Removed regex import

# Attempt to import pyopencl for GPU detection
try:
    import pyopencl
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False
    pyopencl = None


# =IA. DO NOT EDIT THIS LINE.
# ==============================================================
# --- Helper functions for gate decompositions (Unchanged)---
# ==============================================================

def rz(sim, angle, q):
    """Pauli-Z rotation"""
    sim.r(3, angle, q) # Qrack uses Pauli enum: 1=X, 2=Y, 3=Z

def rx(sim, angle, q):
    """Pauli-X rotation"""
    sim.r(1, angle, q)

def ry(sim, angle, q):
    """Pauli-Y rotation"""
    sim.r(2, angle, q)

def apply_zz_evolution(sim, q1, q2, angle):
    """Applies exp(-i * angle * Z1 * Z2)."""
    sim.mcx([q1], q2)
    rz(sim, 2.0 * angle, q2)
    sim.mcx([q1], q2)

def apply_dq_evolution(sim, q1, q2, angle):
    """Applies exp(-i * angle * (YY - XX)) = exp(-i*angle*YY) * exp(i*angle*XX)"""
    # --- exp(i * angle * XX) ---
    angle_xx = angle
    sim.h(q1)
    sim.h(q2)
    sim.mcx([q1], q2)
    rz(sim, 2.0 * angle_xx, q2)
    sim.mcx([q1], q2)
    sim.h(q1)
    sim.h(q2)

    # --- exp(-i * angle * YY) ---
    angle_yy = -angle
    rx(sim, math.pi / 2.0, q1)
    rx(sim, math.pi / 2.0, q2)
    sim.mcx([q1], q2)
    rz(sim, 2.0 * angle_yy, q2)
    sim.mcx([q1], q2)
    rx(sim, -math.pi / 2.0, q1)
    rx(sim, -math.pi / 2.0, q2)

def apply_trotter_step(sim, couplings, dt, n_qubits, forward=True):
    """Applies one Trotter step for H = sum (Jzz Z.Z + Jdq(YY-XX))"""
    tstep = dt if forward else -dt
    # Apply ZZ terms first
    for (q1, q2), terms in couplings.items():
        if 'zz' in terms and terms['zz'] != 0:
            apply_zz_evolution(sim, q1, q2, terms['zz'] * tstep)
    # Apply DQ terms second
    for (q1, q2), terms in couplings.items():
        if 'dq' in terms and terms['dq'] != 0:
            apply_dq_evolution(sim, q1, q2, terms['dq'] * tstep)

# ==============================================================
# --- Context Manager for Suppressing C++ Output (Unchanged)---
# ==============================================================

@contextlib.contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects C++ stdout and stderr to devnull
       using low-level file descriptor duplication.

       Yields:
           int: The file descriptor for the original, un-redirected stdout.
    """
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    with open(os.devnull, 'w') as fnull:
        devnull_fd = fnull.fileno()
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        sys.stdout.flush()
        sys.stderr.flush()
    try:
        yield old_stdout_fd
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)


# ==============================================================
# --- OTOC via two-branch overlap using QBDD ---
# ==============================================================

def compute_otoc_overlap(n_qubits, couplings, time_t, n_steps, m_qubit, b_qubits, device_id):
    """
    Compute F(t) = <psi_A | psi_B> using the QBDD backend and its get_unitary_fidelity method.
    """
    def run_branch(seq_fn, dev_id):
        s = None
        init_exception = None
        with suppress_stdout_stderr(): # Suppress init output only
            try:
                # *** Use QBDD backend ***
                s = QrackSimulator(n_qubits, isBinaryDecisionTree=True)
                s.set_device(dev_id)
            except Exception as e:
                init_exception = e
        if init_exception:
            raise RuntimeError(f"QrackSimulator QBDD init failed on device {dev_id}: {init_exception}") from init_exception

        s.h(m_qubit) # Initial state: |+>_m |0>_other

        if n_steps == 0 or time_t == 0.0:
            dt_actual = 0.0
            steps = 0
        else:
            dt_actual = time_t / n_steps
            steps = n_steps

        # --- Forward evolution U(t) ---
        for i in range(steps):
            apply_trotter_step(s, couplings, dt_actual, n_qubits, forward=True)

        # --- Branch-specific sequence ---
        seq_fn(s)

        # --- Backward evolution U_dag(t) ---
        for i in range(steps):
            apply_trotter_step(s, couplings, dt_actual, n_qubits, forward=False)

        return s # Return the simulator object with the final QBDD state

    # Branch A sequence: Apply W then V
    def branchA_seq(s):
        for bq in b_qubits: s.x(bq)
        s.x(m_qubit)

    # Branch B sequence: Apply V then W
    def branchB_seq(s):
        s.x(m_qubit)
        for bq in b_qubits: s.x(bq)

    # --- Run both branches ---
    simA = run_branch(branchA_seq, device_id)
    simB = run_branch(branchB_seq, device_id)

    # ---- QBDD Overlap calculation ----
    try:
        # *** Use the QBDD get_unitary_fidelity method ***
        # Assuming this returns the complex overlap <A|B> or something convertible
        overlap_result = simA.get_unitary_fidelity(simB)

        # Ensure it's a Python complex type
        if not isinstance(overlap_result, complex):
             try:
                 # Check if it returns a tuple (real, imag)
                 if hasattr(overlap_result, '__len__') and len(overlap_result) == 2:
                     inner_product = complex(overlap_result[0], overlap_result[1])
                 else:
                     # Assume it might return just a real number (potentially fidelity^2)
                     # Warn the user if this happens, as OTOC needs complex overlap
                     try:
                         real_val = float(overlap_result)
                         if abs(real_val.imag) < 1e-9: # Check if it's effectively real
                              print(f"Warning: get_unitary_fidelity returned a non-complex value ({overlap_result}) for t={time_t}. Result might be fidelity^2, not complex overlap.", file=sys.stderr)
                         inner_product = complex(overlap_result)
                     except (TypeError, ValueError):
                          raise TypeError(f"Could not convert get_unitary_fidelity result '{overlap_result}' to float or complex.")

             except (TypeError, ValueError) as conv_err:
                  raise TypeError(f"Could not convert get_unitary_fidelity result '{overlap_result}' to complex. Error: {conv_err}")
        else:
             inner_product = overlap_result # It was already complex

        del simA
        del simB # Cleanup simulators
        return inner_product

    except AttributeError:
         # This would happen if 'get_unitary_fidelity' doesn't exist
         raise RuntimeError("QrackSimulator (QBDD) does not have a 'get_unitary_fidelity' method. Check PyQrack API.")
    except Exception as e:
        # Catch other errors, including potential memory errors
        if isinstance(e, MemoryError) or 'bad_alloc' in str(e):
             raise MemoryError(f"Memory allocation failed during QBDD get_unitary_fidelity for {n_qubits} qubits on device {device_id}. Error: {e}")
        raise RuntimeError(f"Failed to compute overlap using QBDD get_unitary_fidelity. Error: {e}.")

# ==============================================================
# --- Worker function for parallel processing (Unchanged) ---
# ==============================================================

def simulation_worker(t, n_steps, device_id, n_qubits, couplings, m_qubit, b_qubits):
    """Worker function for parallel processing one time point."""
    start = time.time()
    try:
        F = compute_otoc_overlap(n_qubits, couplings, t, n_steps, m_qubit, b_qubits, device_id)
        status = "Success"
    except MemoryError as me:
        F = None
        status = f"MemoryError on device {device_id} at t={t} ({n_qubits} qubits, QBDD): {me}"
        print(status, file=sys.stderr) # Print error directly
    except Exception as e:
        F = None
        status = f"Failed on device {device_id} at t={t} (QBDD): {e}"
        print(status, file=sys.stderr) # Print error directly
    end = time.time()
    wall_time = end - start
    return (t, n_steps, F, wall_time, status)

def simulation_worker_wrapper(args_tuple):
    """Helper to unpack arguments for pool.imap_unordered."""
    return simulation_worker(*args_tuple)

# ==============================================================
# --- Main routine (Unchanged, uses QBDD via worker) ---
# ==============================================================

if __name__ == "__main__":

    # --- Automatic GPU detection ---
    print("Detecting OpenCL GPUs...")
    base_gpus = []
    gpu_id_counter = 0
    if PYOPENCL_AVAILABLE:
        try:
            platforms = pyopencl.get_platforms()
            if not platforms: print("  No OpenCL platforms found.")
            for p_idx, p in enumerate(platforms):
                gpu_devices_on_platform = p.get_devices(device_type=pyopencl.device_type.GPU)
                if not gpu_devices_on_platform: print(f"  No GPUs found on platform {p_idx} ('{p.name}').")
                for d in gpu_devices_on_platform:
                    print(f"  Found GPU: '{d.name}' (Assigning sequential ID: {gpu_id_counter})")
                    base_gpus.append(gpu_id_counter)
                    gpu_id_counter += 1
        except AttributeError as ae:
             print(f"  PyOpenCL Attribute Error: {ae}.")
             base_gpus = [] # Reset on error
        except Exception as e:
            print(f"  Failed to detect OpenCL GPUs: {e}")
            base_gpus = [] # Reset on error
    else:
        print("  pyopencl not found. Cannot auto-detect GPUs.")
    num_gpus = len(base_gpus)
    print(f"  Detection complete. Found {num_gpus} GPUs with assigned IDs: {base_gpus}")
    if num_gpus == 0:
        print("Error: No GPUs detected. Cannot proceed.", file=sys.stderr)
        sys.exit(1)
    # --- End GPU detection ---

    # --- Grid and Qubit Setup ---
    grid_width = 8
    grid_height = 5
    num_qubits = grid_width * grid_height # 40 qubits

    # --- Hamiltonian Setup ---
    example_couplings = {}
    zz_strength = 0.0015 # Using very weak strengths
    dq_strength = 0.0008
    print(f"Building 2D lattice ({grid_width}x{grid_height}) Hamiltonian for {num_qubits} qubits...")
    for y in range(grid_height):
        for x in range(grid_width):
            q_current = y * grid_width + x
            if x < grid_width - 1:
                q_right = y * grid_width + (x + 1)
                example_couplings[(q_current, q_right)] = {"zz": zz_strength, "dq": dq_strength}
            if y < grid_height - 1:
                q_below = (y + 1) * grid_width + x
                example_couplings[(q_current, q_below)] = {"zz": zz_strength, "dq": dq_strength}

    measurement_qubit = 0 # Top-left
    butterfly_qubits = [num_qubits - 1] # Bottom-right (qubit 39)

    # --- Simulation Parameters ---
    fixed_dt = 0.01
    max_time = 10.0

    # *** Use QBDD Backend ***
    print(f"Using fixed Trotter step dt = {fixed_dt}")
    print(f"Simulating OTOC using QBDD backend for {num_qubits} qubits...")
    print(f"  V = X on qubit {measurement_qubit} (Top-Left)")
    print(f"  W = X on qubit {butterfly_qubits[0]} (Bottom-Right)")
    print("-" * 30)

    # --- Time sweep (parallel) ---
    num_time_points = 31
    time_points = [i * max_time / (num_time_points - 1) for i in range(num_time_points)]

    gpu_devices = base_gpus * 2
    device_cycle = cycle(gpu_devices)
    num_processes = len(gpu_devices) * 2
    if num_processes <= 0:
        print("Error: No worker processes configured.", file=sys.stderr)
        sys.exit(1)

    print(f"Starting parallel time sweep across {num_processes} workers ({len(base_gpus)} GPUs x 2 workers) on devices: {gpu_devices}...")
    # Print headers
    print(f"{'Time (t)':<10} | {'Trotter Steps':<14} | {'Re[Overlap]':<14} | {'Im[Overlap]':<14} | {'Wall Time (s)':<15}")
    print("-" * 80)
    sys.stdout.flush()

    # Prepare tasks
    tasks = []
    for t in time_points:
        n_steps = 0 if t == 0 else max(1, int(round(t / fixed_dt)))
        assigned_device = next(device_cycle)
        tasks.append((t, n_steps, assigned_device, num_qubits, example_couplings, measurement_qubit, butterfly_qubits))

    try:
        mp_context = multiprocessing.get_context('spawn')
    except Exception:
        print("Warning: Could not set context to 'spawn'. Using default.", file=sys.stderr)
        mp_context = multiprocessing

    all_results = {} # Store results

    # Wrap pool execution for suppression
    with suppress_stdout_stderr() as original_stdout_fd:
        with mp_context.Pool(processes=num_processes) as pool:
            results_iterator = pool.imap_unordered(simulation_worker_wrapper, tasks)

            for result in results_iterator:
                (t, n_steps, F, wall_time, status) = result
                all_results[t] = (n_steps, F, wall_time, status) # Store result

                if status == "Success" and F is not None:
                    output_line = f"{t:<10.3f} | {n_steps:<14} | {F.real:<14.6f} | {F.imag:<14.6f} | {wall_time:<15.4f}\n"
                else:
                    output_line = f"{t:<10.3f} | {n_steps:<14} | {'NaN':<14} | {'NaN':<14} | {wall_time:<15.4f} | {status}\n"
                # Write directly to original stdout
                os.write(original_stdout_fd, output_line.encode())

    # --- End of suppression block ---

    print("-" * 80)
    print("Time sweep complete.")
    # Optional: Print sorted results if needed
    # ...

