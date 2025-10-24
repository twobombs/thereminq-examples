#!/usr/bin/env python3
# Out-of-Time-Order Correlator (OTOC) simulation using PyQrack QBDD backend
# Based on gemini25-draft, modified for correct OTOC evaluation via overlap
# Aryan Blaauw & GPT-5, Oct 2025

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

# =IA. DO NOT EDIT THIS LINE.
# ==============================================================
# --- Helper functions for gate decompositions ---
# ==============================================================

def rz(sim, angle, q):
    """Pauli-Z rotation"""
    sim.r(3, angle, q)

def rx(sim, angle, q):
    """Pauli-X rotation"""
    sim.r(1, angle, q)

def ry(sim, angle, q):
    """Pauli-Y rotation"""
    sim.r(2, angle, q)

def apply_zz_evolution(sim, q1, q2, angle):
    """Applies e^(-i * angle * Z1 * Z2)."""
    sim.mcx([q1], q2)
    rz(sim, 2.0 * angle, q2)
    sim.mcx([q1], q2)

def apply_dq_evolution(sim, q1, q2, angle):
    """Applies e^(-i * angle * (YY - XX))"""
    # --- e^(i * angle * XX)
    angle_xx = -angle
    sim.h(q1)
    sim.h(q2)
    sim.mcx([q1], q2)
    rz(sim, 2.0 * angle_xx, q2)
    sim.mcx([q1], q2)
    sim.h(q1)
    sim.h(q2)
    # --- e^(-i * angle * YY)
    angle_yy = angle
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
    for (q1, q2), terms in couplings.items():
        if 'zz' in terms and terms['zz'] != 0:
            apply_zz_evolution(sim, q1, q2, terms['zz'] * tstep)
        if 'dq' in terms and terms['dq'] != 0:
            apply_dq_evolution(sim, q1, q2, terms['dq'] * tstep)

# ==============================================================
# --- New OTOC via two-branch overlap ---
# ==============================================================

@contextlib.contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull
       using low-level file descriptor duplication to capture C++ output.
       
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
    
    try:
        # --- CHANGED: Yield the *original* stdout file descriptor ---
        yield old_stdout_fd
    finally:
        # Restore original stdout and stderr
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        # Close the saved FDs
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)


def compute_otoc_overlap(n_qubits, couplings, time_t, n_steps, m_qubit, b_qubits, device_id):
    """
    Compute F(t) = <psi| V(t) W V(t) W |psi>
    using two-branch overlap method.
    'device_id' specifies which GPU to run this instance on.
    """
    def run_branch(seq_fn, dev_id):
        # We remove 'isBinaryDecisionTree=True' because the .out_ket() method
        # on the QBDD backend appears to be causing a 'std::bad_alloc' (out of memory)
        # error in the C++ library, even for a small 20-qubit system.
        # The default state vector backend should handle 20 qubits easily.
        
        s = None
        init_exception = None
        
        # --- Suppress C++ stdout/stderr during init ---
        # This context manager redirects file descriptors 1 (stdout) and 2 (stderr)
        # to /dev/null just for the QrackSimulator init and set_device call.
        with suppress_stdout_stderr():
            try:
                s = QrackSimulator(n_qubits) # , isBinaryDecisionTree=True)
                s.set_device(dev_id)
            except Exception as e:
                # Store exception to re-raise it after stdout is restored
                init_exception = e
        # ----------------------------------------------

        # --- Handle init exception (if any) with stdout restored ---
        if init_exception:
            # If init failed, print the error and re-raise it.
            # We must write to stderr directly as stdout might still be captured.
            sys.stderr.write(f"Error during QrackSimulator init on device {dev_id}: {init_exception}\n")
            sys.stderr.flush()
            raise init_exception
        # --------------------------------------------------

        s.h(m_qubit)  # prepare |+> on measurement qubit

        if n_steps == 0:
            dt = 0.0
            steps = 0
        else:
            dt = time_t / n_steps
            steps = n_steps

        # forward evolution
        for _ in range(steps):
            apply_trotter_step(s, couplings, dt, n_qubits, forward=True)

        # branch-specific sequence
        seq_fn(s)

        # backward evolution
        for _ in range(steps):
            apply_trotter_step(s, couplings, dt, n_qubits, forward=False)

        return s

    def branchA_seq(s):
        # apply V then W (order: V(t) W)
        s.x(m_qubit)
        for bq in b_qubits:
            s.x(bq)

    def branchB_seq(s):
        # apply W then V (order: W V(t))
        for bq in b_qubits:
            s.x(bq)
        s.x(m_qubit)

    simA = run_branch(branchA_seq, device_id)
    simB = run_branch(branchB_seq, device_id)

    # ---- State overlap ----
    # Adapt these lines based on your PyQrack version's API.
    # The 'dir()' output suggests 'out_ket()' is the method to get the state.
    try:
        svA = simA.out_ket()    # CHANGED: from get_statevector()
        svB = simB.out_ket()    # CHANGED: from get_statevector()

        # This calculation assumes svA and svB are iterables (like lists)
        # of complex numbers or types that can be cast to complex.
        if svA is None or svB is None:
             raise ValueError("simA.out_key() or simB.out_ket() returned None.")

        inner = sum(complex(a).conjugate() * complex(b) for a, b in zip(svA, svB))
        return inner

    except AttributeError as e:
        # This block might be hit if a *different* attribute is missing,
        # but the original get_statevector/state_inner_product are confirmed missing.
        raise RuntimeError(
            f"An AttributeError occurred: {e}. "
            "This is unexpected if 'out_ket' exists. "
            "Check PyQrack documentation."
        )
    except Exception as e:
        # Catch other errors, e.g., TypeError if out_ket() returns a string
        # and complex() fails.
        raise RuntimeError(
            f"Failed to compute overlap using .out_ket(). Error: {e}. "
            "Check the return type of .out_ket(). It might not be a list of complex numbers. "
            "Original tried methods 'get_statevector' and 'state_inner_product' are confirmed missing."
        )

# ==============================================================
# --- Worker function for parallel processing ---
# ==============================================================

def simulation_worker(t, n_steps, device_id, n_qubits, couplings, m_qubit, b_qubits):
    """Worker function for parallel processing one time point."""
    start = time.time()
    F = compute_otoc_overlap(n_qubits, couplings, t, n_steps, m_qubit, b_qubits, device_id)
    end = time.time()
    # Return results for printing
    return (t, n_steps, F, end - start)

def simulation_worker_wrapper(args_tuple):
    """Helper to unpack arguments for pool.imap_unordered."""
    return simulation_worker(*args_tuple)

# ==============================================================
# --- Main routine ---
# ==============================================================

if __name__ == "__main__":
    
    # --- Device detection logic removed. ---
    # --- Hardcoding 3 GPUs as requested. ---
    base_gpus = [0, 1, 2]
    
    num_qubits = 20

    # --- Build 1D chain Hamiltonian ---
    example_couplings = {}
    zz_strength = 0.15
    dq_strength = 0.08

    print(f"Building 1D chain Hamiltonian for {num_qubits} qubits...")
    for i in range(num_qubits - 1):
        example_couplings[(i, i + 1)] = {"zz": zz_strength, "dq": dq_strength}

    measurement_qubit = 0
    butterfly_qubits = [num_qubits - 1]
    dt = 0.05

    print(f"Using fixed Trotter step dt = {dt}")
    # --- CHANGED: Updated print to reflect removal of QBDD ---
    print(f"Simulating OTOC using default SV backend for {num_qubits} qubits...")
    print(f"  V = X on qubit {measurement_qubit}")
    print(f"  W = X on qubit {butterfly_qubits[0]}")
    print("-" * 30)

    # --- Time sweep (parallel) ---
    time_points = [i for i in range(0, 31)]  # 0 to 30

    # Define the list of GPU device IDs to use
    # Run 2 processes per GPU to increase utilization
    gpu_devices = base_gpus * 2 # Use devices [0,1,2] and run 2 threads per device
    device_cycle = cycle(gpu_devices)
    num_processes = len(gpu_devices)

    print(f"Starting parallel time sweep across {num_processes} workers ({len(base_gpus)} GPUs x 2 workers) on devices: {gpu_devices}...")
    # --- FIXED: Added padding and closing quote to the f-string ---
    print(f"{'Time (t)':<10} | {'Trotter Steps':<14} | {'Re[OTOC]':<14} | {'Im[OTOC]':<14} | {'Wall Time (s)':<15}")
    print("-" * 80)
    sys.stdout.flush() # Ensure headers are printed

    # Prepare the list of tasks for the pool
    tasks = []
    for t in time_points:
        n_steps = 0 if t == 0 else max(1, int(t / dt))
        # Assign a device from the cycle
        assigned_device = next(device_cycle)
        # Add arguments for starmap
        tasks.append((t, n_steps, assigned_device, num_qubits, example_couplings, measurement_qubit, butterfly_qubits))

    # Use multiprocessing Pool to run tasks in parallel
    # We set the start method to 'spawn' for GPU/CUDA compatibility
    try:
        mp_context = multiprocessing.get_context('spawn')
    except Exception:
        print("Warning: Could not set multiprocessing context to 'spawn'. Using default.")
        mp_context = multiprocessing

    # --- ADDED: Suppress stdout/stderr for the entire pool lifecycle ---
    # This prevents the C++ chatter when workers are destroyed at the end.
    # --- CHANGED: It now yields the original stdout file descriptor ---
    with suppress_stdout_stderr() as original_stdout_fd:
        with mp_context.Pool(processes=num_processes) as pool:
            # We use imap_unordered to get results as soon as they are ready,
            # rather than waiting for all jobs to finish (which 'starmap' does).
            # This will print results out of order, but shows progress.
            results_iterator = pool.imap_unordered(simulation_worker_wrapper, tasks)

            # --- CHANGED: Print results immediately using os.write ---
            # This writes *directly* to the original stdout file descriptor,
            # bypassing the redirect and printing immediately.
            for (t, n_steps, F, wall_time) in results_iterator:
                output_line = f"{t:<10.2f} | {n_steps:<14} | {F.real:<14.6f} | {F.imag:<14.6f} | {wall_time:<15.4f}\n"
                # Write the byte-encoded string to the original stdout FD
                os.write(original_stdout_fd, output_line.encode())

    # --- END of suppression block ---

    # Results are already printed, so we just print the footer.
    print("-" * 80)
    print("Time sweep complete.")

