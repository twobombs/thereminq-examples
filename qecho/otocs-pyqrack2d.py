#!/usr/bin/env python3
# Out-of-Time-Order Correlator (OTOC) simulation using PyQrack default backend
# Includes pyopencl for automatic GPU detection (no fallback).
# Simulates a 2D lattice.
# Aryan Blaauw & Gemini, Oct 2025
# in active development

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
# --- Helper functions for gate decompositions ---
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
# --- Context Manager for Suppressing C++ Output ---
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
# --- OTOC via two-branch overlap ---
# ==============================================================

def compute_otoc_overlap(n_qubits, couplings, time_t, n_steps, m_qubit, b_qubits, device_id):
    """
    Compute F(t) = <psi_A | psi_B> where
    |psi_A> = U_dag(t) W U(t) V |psi>
    |psi_B> = U_dag(t) V U(t) W |psi>
    and |psi> = |+>_m |0>_other
    using two-branch overlap method.
    'device_id' specifies which GPU to run this instance on.
    """
    def run_branch(seq_fn, dev_id):
        s = None
        init_exception = None
        with suppress_stdout_stderr(): # Suppress init output only
            try:
                # Increased number of qubits here
                s = QrackSimulator(n_qubits)
                s.set_device(dev_id)
            except Exception as e:
                init_exception = e
        if init_exception:
            sys.stderr.write(f"Error during QrackSimulator init on device {dev_id}: {init_exception}\n")
            sys.stderr.flush()
            raise init_exception

        s.h(m_qubit) # Initial state: |+> on measurement qubit, |0> elsewhere

        if n_steps == 0 or time_t == 0.0:
            dt_actual = 0.0
            steps = 0
        else:
            # Use actual dt based on time_t and n_steps
            dt_actual = time_t / n_steps
            steps = n_steps

        # Forward evolution U(t)
        for i in range(steps):
            apply_trotter_step(s, couplings, dt_actual, n_qubits, forward=True)
            # Optional: Print progress within the loop for long runs
            # if (i + 1) % 100 == 0:
            #     print(f"  Device {dev_id}, t={time_t:.3f}: Forward step {i+1}/{steps}", file=sys.stderr)
            #     sys.stderr.flush()


        # Branch-specific sequence
        seq_fn(s)

        # Backward evolution U_dag(t)
        for i in range(steps):
            apply_trotter_step(s, couplings, dt_actual, n_qubits, forward=False)
            # Optional: Print progress within the loop for long runs
            # if (i + 1) % 100 == 0:
            #     print(f"  Device {dev_id}, t={time_t:.3f}: Backward step {i+1}/{steps}", file=sys.stderr)
            #     sys.stderr.flush()

        return s

    # Branch A sequence: Apply W then V
    def branchA_seq(s):
        for bq in b_qubits: s.x(bq)
        s.x(m_qubit)

    # Branch B sequence: Apply V then W
    def branchB_seq(s):
        s.x(m_qubit)
        for bq in b_qubits: s.x(bq)

    simA = run_branch(branchA_seq, device_id)
    simB = run_branch(branchB_seq, device_id)

    # ---- State overlap calculation ----
    try:
        svA = simA.out_ket()
        svB = simB.out_ket()
        if svA is None or svB is None: raise ValueError("out_ket returned None.")
        # Check size based on the potentially larger n_qubits
        if len(svA) != len(svB) or len(svA) != 1 << n_qubits: raise ValueError(f"State vector length mismatch. Expected {1<<n_qubits}, got {len(svA)}")
        inner_product = sum(complex(a).conjugate() * complex(b) for a, b in zip(svA, svB))
        del simA
        del simB # Cleanup simulators
        return inner_product
    except AttributeError:
         raise RuntimeError("Could not find '.out_ket()' method.")
    except Exception as e:
        # Catch memory errors explicitly if they become an issue
        if isinstance(e, MemoryError) or 'bad_alloc' in str(e):
             raise MemoryError(f"Memory allocation failed for {n_qubits} qubits on device {device_id}. Error: {e}")
        raise RuntimeError(f"Failed to compute overlap using .out_ket(). Error: {e}.")

# ==============================================================
# --- Worker function for parallel processing ---
# ==============================================================

def simulation_worker(t, n_steps, device_id, n_qubits, couplings, m_qubit, b_qubits):
    """Worker function for parallel processing one time point."""
    start = time.time()
    try:
        F = compute_otoc_overlap(n_qubits, couplings, t, n_steps, m_qubit, b_qubits, device_id)
        status = "Success"
    except MemoryError as me:
        F = None
        status = f"MemoryError on device {device_id} at t={t} ({n_qubits} qubits): {me}"
        print(status, file=sys.stderr) # Print error directly
    except Exception as e:
        F = None
        status = f"Failed on device {device_id} at t={t}: {e}"
        print(status, file=sys.stderr) # Print error directly
    end = time.time()
    wall_time = end - start
    return (t, n_steps, F, wall_time, status)

def simulation_worker_wrapper(args_tuple):
    """Helper to unpack arguments for pool.imap_unordered."""
    return simulation_worker(*args_tuple)

# ==============================================================
# --- Main routine ---
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
    grid_width = 4 # Increased grid size
    grid_height = 6
    num_qubits = grid_width * grid_height # Now 24 qubits

    # --- Hamiltonian Setup ---
    example_couplings = {}
    zz_strength = 0.0015 # Using very weak strengths
    dq_strength = 0.0008
    print(f"Building 2D lattice ({grid_width}x{grid_height}) Hamiltonian for {num_qubits} qubits...")
    for y in range(grid_height):
        for x in range(grid_width):
            q_current = y * grid_width + x
            # Connect to the right neighbor if it exists
            if x < grid_width - 1:
                q_right = y * grid_width + (x + 1)
                example_couplings[(q_current, q_right)] = {"zz": zz_strength, "dq": dq_strength}
            # Connect to the neighbor below if it exists
            if y < grid_height - 1:
                q_below = (y + 1) * grid_width + x
                example_couplings[(q_current, q_below)] = {"zz": zz_strength, "dq": dq_strength}

    measurement_qubit = 0 # Top-left
    butterfly_qubits = [num_qubits - 1] # Bottom-right (now qubit 39)

    # --- Simulation Parameters ---
    fixed_dt = 0.01 # Using smaller dt
    max_time = 10.0 # Using longer time

    print(f"Using fixed Trotter step dt = {fixed_dt}")
    print(f"Simulating OTOC using default SV backend for {num_qubits} qubits...")
    print(f"  V = X on qubit {measurement_qubit} (Top-Left)")
    print(f"  W = X on qubit {butterfly_qubits[0]} (Bottom-Right)")
    print("-" * 30)

    # --- Time sweep (parallel) ---
    num_time_points = 31 # Number of points including t=0
    time_points = [i * max_time / (num_time_points - 1) for i in range(num_time_points)]

    gpu_devices = base_gpus * 2 # Run 2 processes per GPU
    device_cycle = cycle(gpu_devices)
    num_processes = len(gpu_devices) * 2 # Correct calculation for number of processes
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


