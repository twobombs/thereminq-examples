#!/usr/bin/env python3
# Out-of-Time-Order Correlator (OTOC) simulation using PyQrack default backend
# Based on gemini25-draft, modified for correct OTOC evaluation via overlap
# Now includes pyopencl for automatic GPU detection (no fallback)
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

# Attempt to import pyopencl for GPU detection
try:
    # --- MODIFIED: Import the main module ---
    import pyopencl
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False
    pyopencl = None # Define pyopencl as None to avoid runtime errors in checks


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
    # Decomposition: H H CNOT Rz(2*theta) CNOT H H
    angle_xx = angle # Note: sign flipped compared to original comment for exp(+i...)
    sim.h(q1)
    sim.h(q2)
    sim.mcx([q1], q2)
    rz(sim, 2.0 * angle_xx, q2)
    sim.mcx([q1], q2)
    sim.h(q1)
    sim.h(q2)

    # --- exp(-i * angle * YY) ---
    # Decomposition: Rx(pi/2) Rx(pi/2) CNOT Rz(2*theta) CNOT Rx(-pi/2) Rx(-pi/2)
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
    # Apply terms sequentially (first-order Trotter)
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
    # Save original file descriptors
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    with open(os.devnull, 'w') as fnull:
        devnull_fd = fnull.fileno()
        # Redirect stdout (1) and stderr (2) to devnull
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        sys.stdout.flush() # Ensure Python buffers are flushed before redirect
        sys.stderr.flush()

    try:
        # Yield the original stdout file descriptor so Python can write to it
        yield old_stdout_fd
    finally:
        # Ensure Python buffers are flushed before restoring FDs
        sys.stdout.flush()
        sys.stderr.flush()
        # Restore original stdout and stderr
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        # Close the saved FDs
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)


# ==============================================================
# --- New OTOC via two-branch overlap ---
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
        # Using default State Vector backend as QBDD+out_ket reported issues.
        s = None
        init_exception = None

        # Suppress C++ stdout/stderr during QrackSimulator init and set_device
        with suppress_stdout_stderr():
            try:
                s = QrackSimulator(n_qubits) # Using default backend
                # set_device accepts the sequential integer ID
                s.set_device(dev_id)
            except Exception as e:
                init_exception = e # Store exception to re-raise later

        # Handle initialization exception (if any) after restoring stdout/stderr
        if init_exception:
            sys.stderr.write(f"Error during QrackSimulator init on device {dev_id}: {init_exception}\n")
            sys.stderr.flush()
            raise init_exception

        # Initial state: |+> on measurement qubit, |0> elsewhere
        s.h(m_qubit)

        # Calculate dt based on total time and total steps for this run
        if n_steps == 0 or time_t == 0.0:
            dt_actual = 0.0
            steps = 0
        else:
            # dt_actual might differ slightly from fixed_dt due to rounding n_steps
            dt_actual = time_t / n_steps
            steps = n_steps

        # Forward evolution U(t)
        for _ in range(steps):
            apply_trotter_step(s, couplings, dt_actual, n_qubits, forward=True)

        # Branch-specific sequence (W then V, or V then W)
        seq_fn(s)

        # Backward evolution U_dag(t)
        for _ in range(steps):
            apply_trotter_step(s, couplings, dt_actual, n_qubits, forward=False)

        return s

    # Branch A sequence: Apply W then V
    # Corresponds to U_dag W U V |psi>
    def branchA_seq(s):
        for bq in b_qubits: # Apply W first
            s.x(bq)
        s.x(m_qubit) # Apply V second

    # Branch B sequence: Apply V then W
    # Corresponds to U_dag V U W |psi>
    def branchB_seq(s):
        s.x(m_qubit) # Apply V first
        for bq in b_qubits: # Apply W second
            s.x(bq)

    simA = run_branch(branchA_seq, device_id)
    simB = run_branch(branchB_seq, device_id)

    # ---- State overlap calculation ----
    try:
        # Get state vectors using the out_ket() method
        svA = simA.out_ket()
        svB = simB.out_ket()

        if svA is None or svB is None:
            raise ValueError("simA.out_ket() or simB.out_ket() returned None.")
        if len(svA) != len(svB) or len(svA) != 1 << n_qubits:
             raise ValueError(f"State vector length mismatch or incorrect size. Got {len(svA)}, {len(svB)}, expected {1 << n_qubits}")

        # Calculate inner product <A|B> = sum(conj(a_i) * b_i)
        # Assuming out_ket returns complex numbers or types convertible to complex
        inner_product = sum(complex(a).conjugate() * complex(b) for a, b in zip(svA, svB))

        # Cleanup simulators
        del simA
        del simB

        return inner_product

    except AttributeError:
        # Fallback error message if even out_ket doesn't exist (unlikely based on context)
         raise RuntimeError("Could not find '.out_ket()' method on QrackSimulator. Check PyQrack API.")
    except Exception as e:
        # Catch other potential errors during overlap calculation
        raise RuntimeError(f"Failed to compute overlap using .out_ket(). Error: {e}. Check return type.")

# ==============================================================
# --- Worker function for parallel processing ---
# ==============================================================

def simulation_worker(t, n_steps, device_id, n_qubits, couplings, m_qubit, b_qubits):
    """Worker function for parallel processing one time point."""
    start = time.time()
    try:
        F = compute_otoc_overlap(n_qubits, couplings, t, n_steps, m_qubit, b_qubits, device_id)
        status = "Success"
    except Exception as e:
        # If simulation fails, record the error and return None for F
        F = None
        status = f"Failed on device {device_id} at t={t}: {e}"
        sys.stderr.write(status + "\n")
        sys.stderr.flush()

    end = time.time()
    wall_time = end - start
    # Return results including status for potential debugging
    return (t, n_steps, F, wall_time, status)

def simulation_worker_wrapper(args_tuple):
    """Helper to unpack arguments for pool.imap_unordered."""
    return simulation_worker(*args_tuple)

# ==============================================================
# --- Main routine ---
# ==============================================================

if __name__ == "__main__":

    # --- Automatic GPU detection via pyopencl ---
    print("Detecting OpenCL GPUs...")
    base_gpus = [] # This will be populated by detection
    gpu_id_counter = 0 # Sequential ID for PyQrack

    if PYOPENCL_AVAILABLE:
        try:
            # --- MODIFIED: Call on pyopencl module ---
            platforms = pyopencl.get_platforms()
            if not platforms:
                print("  No OpenCL platforms found.")
            for p_idx, p in enumerate(platforms):
                # --- MODIFIED: Use pyopencl.device_type ---
                gpu_devices_on_platform = p.get_devices(device_type=pyopencl.device_type.GPU)
                if not gpu_devices_on_platform:
                    print(f"  No GPUs found on platform {p_idx} ('{p.name}').")
                for d in gpu_devices_on_platform:
                    print(f"  Found GPU: '{d.name}' (Assigning sequential ID: {gpu_id_counter})")
                    base_gpus.append(gpu_id_counter)
                    gpu_id_counter += 1
        except AttributeError as ae:
             # Catch specific error if get_platforms is still missing (e.g. installation issue)
             print(f"  PyOpenCL Attribute Error during detection: {ae}. Is PyOpenCL installed correctly with OpenCL drivers?")
             base_gpus = []
        except Exception as e:
            print(f"  Failed to detect OpenCL GPUs: {e}")
            base_gpus = [] # Clear any partial list in case of error during detection
    else:
        print("  pyopencl not found. Cannot auto-detect GPUs.")

    num_gpus = len(base_gpus)
    print(f"  Detection complete. Found {num_gpus} GPUs with assigned IDs: {base_gpus}")

    # --- Exit if no GPUs are found ---
    if num_gpus == 0:
        print("Error: No GPUs detected via pyopencl. Cannot proceed.", file=sys.stderr)
        sys.exit(1) # Exit with a non-zero status code
    # --- End of GPU detection ---


    num_qubits = 20

    # --- Build 1D chain Hamiltonian ---
    example_couplings = {}
    # --- Using Very Weak interaction strengths ---
    zz_strength = 0.0015
    dq_strength = 0.0008

    print(f"Building 1D chain Hamiltonian for {num_qubits} qubits...")
    for i in range(num_qubits - 1):
        example_couplings[(i, i + 1)] = {"zz": zz_strength, "dq": dq_strength}

    measurement_qubit = 0
    butterfly_qubits = [num_qubits - 1] # Single butterfly qubit at the end
    # --- Using Smaller time step ---
    fixed_dt = 0.01 # Fixed time step for Trotterization

    print(f"Using fixed Trotter step dt = {fixed_dt}")
    print(f"Simulating OTOC using default SV backend for {num_qubits} qubits...")
    print(f"  V = X on qubit {measurement_qubit}")
    print(f"  W = X on qubit {butterfly_qubits[0]}")
    print("-" * 30)

    # --- Time sweep (parallel) ---
    # --- MODIFIED: Increased max_time ---
    max_time = 10.0 # Increased from 3.0
    # --- END MODIFICATION ---
    num_time_points = 31 # Number of points including t=0
    time_points = [i * max_time / (num_time_points -1) for i in range(num_time_points)] # Linear spacing

    # Run 2 processes per GPU to potentially increase utilization
    gpu_devices = base_gpus * 2
    device_cycle = cycle(gpu_devices)
    num_processes = len(gpu_devices)

    # --- Check if num_processes is zero ---
    if num_processes == 0:
         print("Error: Number of processes is zero (likely due to no GPUs found). Cannot start Pool.", file=sys.stderr)
         sys.exit(1)

    print(f"Starting parallel time sweep across {num_processes} workers ({len(base_gpus)} GPUs x 2 workers) on devices: {gpu_devices}...")
    # Print headers
    print(f"{'Time (t)':<10} | {'Trotter Steps':<14} | {'Re[Overlap]':<14} | {'Im[Overlap]':<14} | {'Wall Time (s)':<15}")
    print("-" * 80)
    sys.stdout.flush() # Ensure headers are printed before worker output starts

    # Prepare the list of tasks for the pool
    tasks = []
    for t in time_points:
        # Calculate number of steps for this time point based on fixed dt
        # Note: n_steps will be significantly larger due to increased max_time
        n_steps = 0 if t == 0 else max(1, int(round(t / fixed_dt)))
        assigned_device = next(device_cycle)
        tasks.append((t, n_steps, assigned_device, num_qubits, example_couplings, measurement_qubit, butterfly_qubits))

    # Use multiprocessing Pool to run tasks in parallel
    try:
        mp_context = multiprocessing.get_context('spawn')
    except Exception:
        print("Warning: Could not set multiprocessing context to 'spawn'. Using default.", file=sys.stderr)
        mp_context = multiprocessing

    all_results = {} # Store results for potential later use/sorting

    # Wrap the entire pool lifecycle in the suppressor
    with suppress_stdout_stderr() as original_stdout_fd:
        # Check num_processes again just before Pool creation
        if num_processes <= 0:
             # Write error using os.write as stdout is suppressed
             error_msg = "Error: Cannot create Pool with zero processes.\n".encode()
             os.write(original_stdout_fd, error_msg)
             # Need to handle exit carefully within context manager if possible,
             # otherwise raise an exception. Raising seems safer.
             raise ValueError("Cannot create multiprocessing Pool with zero processes.")

        with mp_context.Pool(processes=num_processes) as pool:
            # Use imap_unordered to get results as they complete
            results_iterator = pool.imap_unordered(simulation_worker_wrapper, tasks)

            for result in results_iterator:
                (t, n_steps, F, wall_time, status) = result
                all_results[t] = (n_steps, F, wall_time, status) # Store result

                if status == "Success" and F is not None:
                    # Format output line
                    output_line = f"{t:<10.3f} | {n_steps:<14} | {F.real:<14.6f} | {F.imag:<14.6f} | {wall_time:<15.4f}\n"
                else:
                    # Indicate failure in output
                    output_line = f"{t:<10.3f} | {n_steps:<14} | {'NaN':<14} | {'NaN':<14} | {wall_time:<15.4f} | {status}\n"

                # Write directly to the original stdout file descriptor
                os.write(original_stdout_fd, output_line.encode())

    # --- End of suppression block ---

    print("-" * 80)
    print("Time sweep complete.")

    # Optional: Print sorted results if desired (already printed unsorted)
    # ... (sorted print loop omitted for brevity) ...

