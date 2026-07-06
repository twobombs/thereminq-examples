# -*- coding: us-ascii -*-
# 27-Qubit 3x3x3 Lattice & Macroscopic Grid Annealing
# High-Throughput Volumetric Engine with Statistical Variance Injection
# Nucleus qubit removed: pure nearest-neighbour 3D Ising topology (54 bonds)
#
# REVISION 11 -- ZERO-COPY KET TRANSFER (ROOT-CAUSE OOM FIX) & DEVICE PINNING
#
# This machine exposes TWO OpenCL devices under Mesa rusticl:
#   Device 0: AMD Radeon Pro VII / Instinct MI50 class (vega20, 16GB) -- compute
#   Device 1: NVIDIA CMP 50HX (10GB, nouveau)                        -- DO NOT USE
# All 16 workers pin every QRack device-selection env var to device 0.
#
# Each worker holds the quantum state as a numpy complex64 array in host RAM.
# The GPU sim is ephemeral:
#
#   acquire semaphore
#   -> QrackSimulator (fresh dense GPU sim)
#   -> fast_in_ket(sim, state)       # ZERO-COPY host->GPU (PCIe only)
#   -> Trotter + prob() reads
#   -> state = fast_out_ket(sim, n)  # ZERO-COPY GPU->host (PCIe only)
#   -> del sim
#   release semaphore
#
# ROOT CAUSE OF THE R10 OOM (Fix 17):
#   PyQrack's stock in_ket()/out_ket() route every transfer through
#   _qrack_complex_byref(), which for 2^27 amplitudes materialises:
#     - a list of 2^27 (real, imag) tuples:            ~14 GB
#     - a flattened 2*2^27-element list:                ~2 GB
#     - a 2*2^27-argument unpack tuple for ctypes(*a):  ~2 GB
#     - the ctypes float array itself:                  ~1 GB
#     - (out_ket also returns 2^27 Python complex:      ~5 GB)
#   Measured peak: ~20 GB per out_ket() call, ~24 GB per in_ket() call.
#   With semaphore=3, three workers spike simultaneously: 60-72 GB of
#   transient Python heap -- OOM on the 80 GB machine even at 16 patches.
#
#   Fix: bypass the list layer entirely. Qrack.qrack_lib.InKet/OutKet take a
#   raw float* (fppow=5 default build). numpy complex64 is interleaved
#   float32 (re, im) -- byte-identical to what the C API expects -- so we
#   pass state.ctypes.data_as(POINTER(c_float)) directly. Transfer overhead
#   drops from ~20-24 GB per call to ZERO extra allocation.
#
# CORRECTED MEMORY MODEL (Fix 19):
#   Default Qrack builds use fppow=5 (float32). The GPU state is complex64,
#   NOT complex128 as previously assumed. VRAM per active sim: ~1.1 GB.
#   LIVE_MB = 2200 is retained as a conservative cap (correct for fp64
#   builds, 2x headroom for the default fp32 build).
#   Host RAM: 16 workers x 1 GB state + process overhead = ~32 GB steady,
#   with no transfer spikes. ~48 GB headroom on the 80 GB machine.
#
# Fixes carried forward:
#   Fix 1:  make_sim forced dense engine
#   Fix 2:  Merged Rz layers
#   Fix 3:  ZZ as Rz+Rz+controlled-phase (mcmtrx path)
#   Fix 5:  Clone-free prob() measurement (within each GPU window)
#   Fix 6:  VRAM residency pre-check (semaphore-slot based)
#   Fix 7:  QRACK_MAX_ALLOC_MB tripwire
#   Fix 9:  Mean-field ZZ energy (avoids 162-kernel Mesa GPU reset)
#   Fix 10: Thread-safe pipe heartbeats & memory documentation
#   Fix 11: Heartbeat init drain & S2 Trotterisation
#
# New in R9:
#   Fix 12: Single MI50 hardware config (was erroneously two-GPU)
#   Fix 13: apply_zz mcmtrx path: phase was exp(+2i*theta), corrected to
#            exp(-2i*theta). The CX fallback was always correct; only the
#            mcmtrx fast path had the wrong sign.
#            Derivation: Rz1(a)*Rz2(a)*C-phase(ph) equals exp(-i*phi*ZZ) up
#            to global phase when a = 2*phi and ph = exp(-4i*phi). Since
#            theta (the argument to apply_zz) equals 2*phi, a = theta and
#            ph = exp(-2i*theta).
#   Fix 14: trotter_step_body S2 Rz half-step: was apply_rz(-2*hz*dt) (a full
#            hz step), corrected to apply_rz(-hz*dt) (a true half-step). The
#            R8 implementation doubled the hz contribution per Trotter step.
#   Fix 15: Init per-worker timeout raised from 120s to 300s.
#   Fix 16: Grid reduced from (4,4,2)=32 patches to (4,4,1)=16 patches.
#   Fix 17: Zero-copy in_ket/out_ket via Qrack pinvoke + numpy ctypes pointer.
#            Eliminates the ~20-24 GB per-call Python-heap spike inside
#            PyQrack's _qrack_complex_byref() (the actual R10 OOM cause).
#   Fix 18: Full QRack device pinning. The NVIDIA CMP 50HX is still visible
#            as OpenCL device 1 under rusticl/nouveau. QUnitMulti distributes
#            across ALL visible devices unless QRACK_QUNITMULTI_DEVICES is
#            set; nouveau allocations are host-RAM (GART) backed. Workers now
#            pin QRACK_QUNITMULTI_DEVICES and QRACK_QPAGER_DEVICES in
#            addition to QRACK_OCL_DEFAULT_DEVICE.
#   Fix 19: Corrected memory model comments for the default fppow=5 (fp32)
#            Qrack build: GPU state is complex64 (~1.1 GB/sim), not complex128.

import os
import gc
import csv
import time
import signal
import numpy as np
import multiprocessing as mp
import threading
from multiprocessing.connection import Connection, wait
from typing import List, Tuple, Dict, Any, Optional

# ==========================================
# 0. PYQRACK API SAFEGUARDS & GATES
# ==========================================
PX, PY, PZ = 1, 2, 3

def make_sim(QrackSimulator: Any, n: int) -> Any:
    """Force a plain dense QEngine. Strips unsupported kwargs on TypeError."""
    kwargs = dict(isTensorNetwork=False, isSchmidtDecompose=False,
                  isStabilizerHybrid=False, isBinaryDecisionTree=False)
    while True:
        try:
            return QrackSimulator(qubit_count=n, **kwargs)
        except TypeError as e:
            removed = False
            for k in list(kwargs):
                if k in str(e):
                    kwargs.pop(k); removed = True; break
            if not removed:
                return QrackSimulator(qubit_count=n)

def apply_h(sim: Any, q: int) -> None:
    if hasattr(sim, 'h'): sim.h(q)
    else: sim.mtrx([complex(1/np.sqrt(2),0), complex(1/np.sqrt(2),0),
                    complex(1/np.sqrt(2),0), complex(-1/np.sqrt(2),0)], q)

def apply_rx(sim: Any, theta: float, q: int) -> None:
    if hasattr(sim, 'r'): sim.r(PX, float(theta), q)
    else: sim.mtrx([complex(np.cos(theta/2),0), complex(0,-np.sin(theta/2)),
                    complex(0,-np.sin(theta/2)), complex(np.cos(theta/2),0)], q)

def apply_ry(sim: Any, theta: float, q: int) -> None:
    if hasattr(sim, 'r'): sim.r(PY, float(theta), q)
    else: sim.mtrx([complex(np.cos(theta/2),0), complex(-np.sin(theta/2),0),
                    complex(np.sin(theta/2),0), complex(np.cos(theta/2),0)], q)

def apply_rz(sim: Any, theta: float, q: int) -> None:
    if hasattr(sim, 'r'): sim.r(PZ, float(theta), q)
    else: sim.mtrx([complex(np.cos(theta/2),-np.sin(theta/2)), 0j,
                    0j, complex(np.cos(theta/2),np.sin(theta/2))], q)

def apply_cx(sim: Any, c: int, t: int) -> None:
    if hasattr(sim, 'cx'): sim.cx(c, t)
    elif hasattr(sim, 'mcx'):
        try: sim.mcx([c], t)
        except TypeError: sim.mcx([c], [t])
    else: raise RuntimeError("No CX gate available.")

def apply_zz(sim: Any, theta: float, q1: int, q2: int) -> None:
    """Implements exp(-i*(theta/2) * Z(q1)Z(q2)).

    mcmtrx fast path: Rz1(theta)*Rz2(theta)*C-phase(exp(-2i*theta)).
    Derivation: Rz1(a)*Rz2(a)*C-phase(ph) equals exp(-i*phi*ZZ) up to a
    global phase when a = 2*phi and ph = exp(-4i*phi). With theta = 2*phi
    (i.e. phi = theta/2): a = theta, ph = exp(-2i*theta).
    The global phase exp(-i*phi) = exp(-i*theta/2) has no observable effect.

    CX fallback: CX * Rz2(theta) * CX, which is always exact.
    """
    if hasattr(sim, 'mcmtrx'):
        apply_rz(sim, theta, q1)
        apply_rz(sim, theta, q2)
        # Fix 13: sign was +2i*theta (wrong), corrected to -2i*theta.
        ph = complex(np.cos(2.0*theta), -np.sin(2.0*theta))
        try:    sim.mcmtrx([q1], [complex(1,0), 0j, 0j, ph], q2)
        except TypeError: sim.mcmtrx([q1], [complex(1,0), 0j, 0j, ph], [q2])
    else:
        apply_cx(sim, q1, q2)
        apply_rz(sim, theta, q2)
        apply_cx(sim, q1, q2)

def trotter_step_body(sim: Any, num_qubits: int,
                      intra_edges: List[Tuple[int, int]],
                      J: float, hx: float, hz: float, dt: float,
                      steps: int) -> None:
    """Suzuki-Trotter S2 for H = -hx*sumX - hz*sumZ - J*sumZZ.

    Gate sequence per step:
        Rx(-hx*dt)      on all q   [half-step X:  exp(+i*hx*dt/2 * X)]
        Rz(-hz*dt)      on all q   [half-step Z:  exp(+i*hz*dt/2 * Z)]   <- Fix 14
        ZZ(-2*J*dt)     all edges  [full-step ZZ: exp(+i*J*dt * ZZ)]
        Rz(-hz*dt)      on all q   [half-step Z:  exp(+i*hz*dt/2 * Z)]   <- Fix 14
        Rx(-hx*dt)      on all q   [half-step X:  exp(+i*hx*dt/2 * X)]

    apply_rx(sim, theta, q) calls r(PX, theta) = exp(-i*theta/2 * X).
    apply_rz(sim, theta, q) calls r(PZ, theta) = exp(-i*theta/2 * Z).
    For a half-step of exp(+i*A*dt/2) on axis A with coefficient A, theta = -A*dt.
    R8 used -2*hz*dt for each Rz half-step (a full hz step each time, 2x overrotation).
    R9 corrects to -hz*dt so the two symmetric Rz calls together give exactly one
    full dt of hz evolution split as (dt/2 + dt/2).
    """
    for _ in range(steps):
        for q in range(num_qubits): apply_rx(sim, -hx*dt, q)
        for q in range(num_qubits): apply_rz(sim, -hz*dt, q)      # half-step
        for q1, q2 in intra_edges:  apply_zz(sim, -2.0*J*dt, q1, q2)
        for q in range(num_qubits): apply_rz(sim, -hz*dt, q)      # half-step
        for q in range(num_qubits): apply_rx(sim, -hx*dt, q)

# prob() is non-destructive on dense QEngine.
# make_sim() guarantees dense engine (isTensorNetwork=False, isStabilizerHybrid=False).
def z_means(sim: Any, qubits: List[int]) -> np.ndarray:
    return np.array([1.0 - 2.0*sim.prob(q) for q in qubits])

def x_means(sim: Any, qubits: List[int]) -> np.ndarray:
    out = np.empty(len(qubits))
    for i, q in enumerate(qubits):
        apply_h(sim, q); out[i] = 1.0 - 2.0*sim.prob(q); apply_h(sim, q)
    return out

def y_means(sim: Any, qubits: List[int]) -> np.ndarray:
    out = np.empty(len(qubits))
    for i, q in enumerate(qubits):
        apply_rx(sim, -np.pi/2, q); out[i] = 1.0 - 2.0*sim.prob(q)
        apply_rx(sim, np.pi/2, q)
    return out

def zz_means_meanfield(z_exp: np.ndarray, edges: List[Tuple[int,int]]) -> np.ndarray:
    """<ZiZj> ~ <Zi><Zj>. Avoids 162-kernel CX/prob/CX chain on Mesa rusticl."""
    return np.array([z_exp[q1]*z_exp[q2] for q1,q2 in edges])

# ==========================================
# 0b. ZERO-COPY STATE TRANSFER (Fix 17)
# ==========================================
# PyQrack's stock in_ket()/out_ket() convert through Python lists via
# _qrack_complex_byref(), spiking ~20-24 GB of transient heap per call at
# 2^27 amplitudes. The underlying C API (Qrack.qrack_lib.InKet/OutKet) takes
# a raw float* (fppow=5, default) or double* (fppow>=6). numpy complex64 /
# complex128 are interleaved (re, im) pairs -- byte-identical to that layout
# -- so we hand the C library a direct pointer into the numpy buffer.
#
# Verified against PyQrack source (qrack_simulator.py):
#   in_ket : Qrack.qrack_lib.InKet(self.sid, _qrack_complex_byref(ket))
#   out_ket: OutKet fills a ctypes array, then builds a list of complex
# and qrack_system.py:
#   fppow < 6 -> InKet/OutKet argtypes = [c_ulonglong, POINTER(c_float)]
#   fppow >= 6 -> POINTER(c_double)

import ctypes

# Lazy singleton: resolved on first call, INSIDE the worker process, AFTER the
# QRACK_* env vars are set. A module-level import would load libqrack (and
# open DRM fds) in the parent process, which we deliberately avoid.
_QRACK_PINVOKE: Dict[str, Any] = {}

def _get_qrack_pinvoke() -> Optional[Any]:
    if "q" not in _QRACK_PINVOKE:
        try:
            from pyqrack.qrack_system import Qrack
            _QRACK_PINVOKE["q"] = Qrack
        except Exception:
            _QRACK_PINVOKE["q"] = None
    return _QRACK_PINVOKE["q"]

def _state_dtype_and_ptr(qrack: Any) -> Tuple[Any, Any]:
    if qrack.fppow < 6:
        return np.complex64, ctypes.POINTER(ctypes.c_float)
    return np.complex128, ctypes.POINTER(ctypes.c_double)

def fast_in_ket(sim: Any, state: np.ndarray) -> None:
    """Zero-copy host->GPU state upload. Falls back to sim.in_ket on failure."""
    qrack = _get_qrack_pinvoke()
    if qrack is not None:
        dtype, cptr = _state_dtype_and_ptr(qrack)
        arr = np.ascontiguousarray(state, dtype=dtype)
        qrack.qrack_lib.InKet(sim.sid, arr.ctypes.data_as(cptr))
        sim._throw_if_error()
    else:
        sim.in_ket(state.tolist())  # slow path; only if pinvoke unavailable

def fast_out_ket(sim: Any, num_qubits: int) -> np.ndarray:
    """Zero-copy GPU->host state download. Falls back to sim.out_ket on failure."""
    qrack = _get_qrack_pinvoke()
    if qrack is not None:
        dtype, cptr = _state_dtype_and_ptr(qrack)
        buf = np.empty(1 << num_qubits, dtype=dtype)
        qrack.qrack_lib.OutKet(sim.sid, buf.ctypes.data_as(cptr))
        sim._throw_if_error()
        # fppow=5: buf is already complex64, return as-is (no copy).
        # fppow>=6: collapse complex128 -> complex64 for host storage.
        return buf if buf.dtype == np.complex64 else buf.astype(np.complex64)
    return np.array(sim.out_ket(), dtype=np.complex64)  # slow path

# ==========================================
# 1. TOPOLOGY
# ==========================================
def generate_27q_lattice_subvolume() -> Tuple[List[Tuple[int,int]], Dict[str,List[int]]]:
    lx, ly, lz = 3, 3, 3
    edges = []
    boundaries = {"+X":[], "-X":[], "+Y":[], "-Y":[], "+Z":[], "-Z":[]}
    for x in range(lx):
        for y in range(ly):
            for z in range(lz):
                idx = x*(ly*lz) + y*lz + z
                if x < lx-1: edges.append((idx, (x+1)*(ly*lz)+y*lz+z))
                if y < ly-1: edges.append((idx, x*(ly*lz)+(y+1)*lz+z))
                if z < lz-1: edges.append((idx, x*(ly*lz)+y*lz+(z+1)))
                if x == 0:    boundaries["-X"].append(idx)
                if x == lx-1: boundaries["+X"].append(idx)
                if y == 0:    boundaries["-Y"].append(idx)
                if y == ly-1: boundaries["+Y"].append(idx)
                if z == 0:    boundaries["-Z"].append(idx)
                if z == lz-1: boundaries["+Z"].append(idx)
    return edges, boundaries

# ==========================================
# 2. HOST-RAM SERIALISATION WORKER
# ==========================================
def persistent_island_worker_27q(
    gpu_device_id: int,
    patch_idx: int,
    num_qubits: int,
    intra_edges: List[Tuple[int,int]],
    boundaries: Dict[str,List[int]],
    cmd_pipe: Connection,
    gpu_semaphore: Any,
    seed: int,
    gpu_max_alloc_mb: int
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"
    os.environ["QRACK_MAX_ALLOC_MB"] = str(int(gpu_max_alloc_mb))
    # Fix 18: pin EVERY QRack device-selection path to the assigned GPU.
    # The NVIDIA CMP 50HX is still enumerated as OpenCL device 1 under
    # rusticl/nouveau; QUnitMulti will spread across all visible devices
    # unless explicitly pinned, and nouveau buffers are host-RAM backed.
    dev = str(gpu_device_id)
    os.environ["QRACK_OCL_DEFAULT_DEVICE"] = dev
    os.environ["QRACK_QUNITMULTI_DEVICES"] = dev
    os.environ["QRACK_QPAGER_DEVICES"] = dev
    # Fix 20: pin float precision to match the local Qrack build (-DFPPOW=5).
    # PyQrack does NOT query the library for precision -- it assumes fppow=5
    # unless QRACK_FPPOW overrides it. Pinning documents the contract: if the
    # library is ever rebuilt at FPPOW=6, update this (and rebuild awareness
    # that InKet/OutKet switch from float* to double*), or states will be
    # silently corrupted by both the stock API and the fast path.
    os.environ.setdefault("QRACK_FPPOW", "5")

    state: Optional[np.ndarray] = None  # complex64 array, ~1.07GB per worker
    pipe_lock = threading.Lock()
    stop_event = threading.Event()

    def safe_send(msg: Dict[str, Any]) -> None:
        with pipe_lock:
            try:
                cmd_pipe.send(msg)
            except (EOFError, OSError, BrokenPipeError):
                pass

    def heartbeat_loop() -> None:
        while not stop_event.wait(60.0):
            safe_send({"status": "HEARTBEAT", "patch_idx": patch_idx})

    hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    hb_thread.start()

    try:
        from pyqrack import QrackSimulator

        with gpu_semaphore:
            sim = make_sim(QrackSimulator, num_qubits)
            try:
                rng = np.random.default_rng(seed)
                for q in range(num_qubits):
                    apply_h(sim, q)
                    apply_rx(sim, rng.normal(0, 1e-5), q)
                    apply_rz(sim, rng.normal(0, 1e-5), q)
                state = fast_out_ket(sim, num_qubits)
            finally:
                del sim
        gc.collect()

        safe_send({"status": "READY", "patch_idx": patch_idx})

        all_boundary_qubits = sorted(list(set(
            q for face in boundaries.values() for q in face
        )))

        while True:
            if not cmd_pipe.poll(timeout=0.5): continue
            cmd = cmd_pipe.recv()
            action = cmd.get("action")
            if action == "SHUTDOWN": break

            try:
                # --------------------------------------------------
                if action == "EVOLVE_ONLY":
                    J     = cmd.get("J",   1.0)
                    hx    = cmd.get("hx",  0.5)
                    hz    = cmd.get("hz",  0.2)
                    dt    = cmd.get("dt",  0.05)
                    steps = cmd.get("steps", 1)

                    with gpu_semaphore:
                        sim = make_sim(QrackSimulator, num_qubits)
                        try:
                            fast_in_ket(sim, state)
                            trotter_step_body(sim, num_qubits, intra_edges,
                                              J, hx, hz, dt, steps)
                            state = fast_out_ket(sim, num_qubits)
                        finally:
                            del sim
                    gc.collect()
                    safe_send({"status": "EVOLVE_COMPLETE", "patch_idx": patch_idx})

                # --------------------------------------------------
                elif action == "EVOLVE_AND_MEASURE_STATISTICAL":
                    J     = cmd.get("J",   1.0)
                    hx    = cmd.get("hx",  0.5)
                    hz    = cmd.get("hz",  0.2)
                    dt    = cmd.get("dt",  0.05)
                    steps = cmd.get("steps", 1)

                    with gpu_semaphore:
                        sim = make_sim(QrackSimulator, num_qubits)
                        try:
                            fast_in_ket(sim, state)
                            trotter_step_body(sim, num_qubits, intra_edges,
                                              J, hx, hz, dt, steps)
                            Z_mean = z_means(sim, all_boundary_qubits)
                            X_mean = x_means(sim, all_boundary_qubits)
                            Y_mean = y_means(sim, all_boundary_qubits)
                            state = fast_out_ket(sim, num_qubits)
                        finally:
                            del sim
                    gc.collect()

                    Z_var = np.clip(1.0 - Z_mean**2, 0.0, 1.0)
                    X_var = np.clip(1.0 - X_mean**2, 0.0, 1.0)
                    Y_var = np.clip(1.0 - Y_mean**2, 0.0, 1.0)

                    payload = {
                        "qubits": all_boundary_qubits,
                        "means": {"X": X_mean, "Y": Y_mean, "Z": Z_mean},
                        "vars":  {"X": X_var,  "Y": Y_var,  "Z": Z_var}
                    }
                    safe_send({"status": "STEP_1_COMPLETE",
                               "patch_idx": patch_idx, "data": payload})

                # --------------------------------------------------
                elif action == "APPLY_KICKS_AND_MEASURE_ENERGY":
                    kicks  = cmd.get("kicks", {})
                    J_val  = cmd.get("J",   1.0)
                    hx_val = cmd.get("hx",  0.5)
                    hz_val = cmd.get("hz",  0.2)
                    all_q  = list(range(num_qubits))

                    with gpu_semaphore:
                        sim = make_sim(QrackSimulator, num_qubits)
                        try:
                            fast_in_ket(sim, state)

                            for raw_q, (kx, ky, kz) in kicks.items():
                                q = int(raw_q)
                                K = np.sqrt(kx**2 + ky**2 + kz**2)
                                if K > 0.0:
                                    c, s = np.cos(K/2.0), np.sin(K/2.0)
                                    nx, ny, nz = kx/K, ky/K, kz/K
                                    sim.mtrx([complex(c, -nz*s),
                                              complex(-ny*s, -nx*s),
                                              complex( ny*s, -nx*s),
                                              complex(c,  nz*s)], q)

                            z_exp = z_means(sim, all_q)
                            x_exp = x_means(sim, all_q)
                            # y_means intentionally omitted: Y is not in the TFIM
                            # local energy expression (-hz*Z - J*ZZ - hx*X).
                            state = fast_out_ket(sim, num_qubits)
                        finally:
                            del sim
                    gc.collect()

                    zz_exp = zz_means_meanfield(z_exp, intra_edges)
                    local_energy = (
                        -hz_val * float(np.sum(z_exp))
                        - J_val  * float(np.sum(zz_exp))
                        - hx_val * float(np.sum(x_exp))
                    )
                    safe_send({"status": "STEP_2_COMPLETE",
                               "patch_idx": patch_idx, "data": local_energy})

                # --------------------------------------------------
                elif action == "COMPUTE_BENCHMARKS":
                    try:
                        all_q = list(range(num_qubits))
                        with gpu_semaphore:
                            sim = make_sim(QrackSimulator, num_qubits)
                            try:
                                fast_in_ket(sim, state)
                                z_e = z_means(sim, all_q)
                                x_e = x_means(sim, all_q)
                                y_e = y_means(sim, all_q)
                                # No out_ket() needed: all prob() reads are
                                # non-destructive; state array is unchanged.
                            finally:
                                del sim
                        gc.collect()
                        avg_purity = float(np.mean(x_e**2 + y_e**2 + z_e**2))
                    except Exception as e:
                        print(f"Worker {patch_idx} benchmark failed: {e}")
                        avg_purity = 0.0

                    safe_send({"status": "BENCHMARKS_COMPUTED",
                               "patch_idx": patch_idx,
                               "data": {"grid_purity": avg_purity}})

            except Exception as e:
                safe_send({"status": "ERROR", "msg": str(e)})

    except (EOFError, OSError, BrokenPipeError): pass
    finally:
        stop_event.set()
        state = None
        gc.collect()
        try:
            with pipe_lock: cmd_pipe.close()
        except Exception: pass

# ==========================================
# 3. ORCHESTRATOR
# ==========================================
class VolumetricHadronEngine27Q:
    def __init__(
        self,
        gpu_allocation: List[int],
        semaphore_limits: Dict[int, int],
        gpu_alloc_caps_mb: Dict[int, int],
        grid: Tuple[int,int,int] = (2,2,2),
        master_seed: int = 42
    ):
        self._is_shutdown = False
        self.gpu_allocation = gpu_allocation

        self.grid_x, self.grid_y, self.grid_z = grid
        self.num_patches = self.grid_x * self.grid_y * self.grid_z

        if len(self.gpu_allocation) != self.num_patches:
            raise ValueError(
                f"GPU allocation list ({len(self.gpu_allocation)}) must match "
                f"total grid patches ({self.num_patches}).")

        self.qubits_per_patch = 27
        self.intra_edges, self.boundaries = generate_27q_lattice_subvolume()

        # VRAM check: only concurrent semaphore slots are live on GPU at once.
        # State lives in host RAM between windows; no persistent GPU residency.
        # Default Qrack build is fppow=5 (fp32): dense 2^27 sim = ~1.1 GB VRAM.
        # LIVE_MB = 2200 is kept as a conservative cap (exact for fp64 builds,
        # 2x headroom for the default fp32 build).
        LIVE_MB = 2200
        residency: Dict[int,int] = {}
        for g in self.gpu_allocation: residency[g] = residency.get(g,0) + 1
        for g, count in residency.items():
            cap = gpu_alloc_caps_mb.get(g)
            if cap is None: raise ValueError(f"No VRAM cap for GPU {g}.")
            concurrent = semaphore_limits.get(g, 1)
            need = concurrent * LIVE_MB
            if need > cap:
                raise ValueError(
                    f"GPU {g}: {concurrent} concurrent sims x {LIVE_MB}MB = "
                    f"{need}MB exceeds {cap}MB cap.")

        self.gpu_alloc_caps_mb = gpu_alloc_caps_mb

        self.patch_coords: Dict[int,Tuple[int,int,int]] = {}
        idx = 0
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                for z in range(self.grid_z):
                    self.patch_coords[idx] = (x, y, z); idx += 1

        self.coord_to_patch = {v:k for k,v in self.patch_coords.items()}

        self.ctx = mp.get_context('spawn')
        self.gpu_semaphores = {
            gid: self.ctx.Semaphore(lim)
            for gid, lim in semaphore_limits.items()
        }

        self.workers: List[mp.Process] = []
        self.pipes:   List[Connection] = []
        self.energy_history: List[Dict[str,Any]] = []

        self.csv_filename = "ground_state_energy_curve_27q_16p.csv"
        self._init_csv()

        total_sites = self.num_patches * self.qubits_per_patch
        host_gb = self.num_patches * 1.074  # complex64 host arrays

        print(f"Initializing 27Q Host-RAM Sliding Window Engine...")
        print(f"Grid: {grid} = {self.num_patches} patches x {self.qubits_per_patch}q "
              f"= {total_sites} total logical qubits")
        print(f"Host RAM: {self.num_patches} arrays x 1.07GB (complex64) = {host_gb:.1f}GB")
        for g, count in residency.items():
            conc = semaphore_limits.get(g, 1)
            print(f"GPU {g}: {count} workers, {conc} concurrent, "
                  f"{conc*LIVE_MB}MB / {gpu_alloc_caps_mb[g]}MB VRAM budget")

        master_rng  = np.random.default_rng(master_seed)
        patch_seeds = master_rng.integers(0, 2**31-1, size=self.num_patches)

        try:
            for p_idx in range(self.num_patches):
                parent_conn, child_conn = self.ctx.Pipe()
                assigned_gpu = self.gpu_allocation[p_idx]
                p = self.ctx.Process(
                    target=persistent_island_worker_27q,
                    args=(
                        assigned_gpu, p_idx, self.qubits_per_patch,
                        self.intra_edges, self.boundaries, child_conn,
                        self.gpu_semaphores[assigned_gpu],
                        int(patch_seeds[p_idx]),
                        self.gpu_alloc_caps_mb[assigned_gpu]
                    )
                )
                p.start()
                child_conn.close()
                self.workers.append(p)
                self.pipes.append(parent_conn)

            # Fix 15: timeout raised to 300s.
            # With 32 workers on a semaphore-3 gate, the last worker in the
            # queue may not acquire the semaphore until ~ceil(32/3)*15s ~= 165s
            # after the parent begins its sequential poll. 120s was too tight.
            _INIT_TIMEOUT_S = 300.0
            for i, pipe in enumerate(self.pipes):
                deadline_init = time.monotonic() + _INIT_TIMEOUT_S
                while True:
                    remaining = deadline_init - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"Worker {i} timeout during init.")
                    if not pipe.poll(timeout=remaining):
                        raise TimeoutError(f"Worker {i} timeout during init.")
                    msg = pipe.recv()
                    if msg.get("status") == "HEARTBEAT":
                        continue  # drain; worker is alive but semaphore-queued
                    if msg.get("status") != "READY":
                        raise RuntimeError(f"Worker {i} protocol failure.")
                    break

        except Exception as e:
            self.shutdown()
            raise RuntimeError(f"Engine init aborted: {e}") from e

    def _init_csv(self) -> None:
        try:
            with open(self.csv_filename, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=["Step","Anneal_Percent","Energy"]).writeheader()
        except Exception: pass

    def _append_to_csv(self, data: Dict[str,Any]) -> None:
        try:
            with open(self.csv_filename, mode='a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=["Step","Anneal_Percent","Energy"])
                w.writerow(data); f.flush(); os.fsync(f.fileno())
        except Exception: pass

    def sync_broadcast(
        self,
        action: str,
        kwargs_list: Optional[List[Dict]] = None,
        expected_status: Optional[str] = None,
        timeout_s: float = 1800.0
    ) -> Dict[int,Any]:
        if kwargs_list is None:
            kwargs_list = [{} for _ in range(self.num_patches)]

        send_errors: Dict[int,Exception] = {}
        error_lock = threading.Lock()

        def _send_one(pipe: Connection, msg: Dict[str,Any], i: int) -> None:
            try: pipe.send(msg)
            except Exception as e:
                with error_lock: send_errors[i] = e

        threads = []
        for i, pipe in enumerate(self.pipes):
            t = threading.Thread(target=_send_one,
                                 args=(pipe, {"action":action, **kwargs_list[i]}, i))
            t.start(); threads.append(t)
        for t in threads: t.join()

        if send_errors:
            self.shutdown()
            raise RuntimeError(f"Send failed: {send_errors}")

        results: Dict[int,Any] = {}
        pending = list(enumerate(self.pipes))
        deadline = time.monotonic() + timeout_s

        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timeout on '{action}' after {timeout_s:.0f}s")
            ready = wait([p for _,p in pending], timeout=min(remaining, 60.0))
            if not ready:
                elapsed = timeout_s - (deadline - time.monotonic())
                print(f"[WARN] Waiting on {len(pending)} workers "
                      f"({elapsed:.0f}s) for '{action}'...")
                continue

            still = []
            for i, pipe in pending:
                if pipe in ready:
                    try:
                        res = pipe.recv()
                        if res.get("status") == "HEARTBEAT":
                            still.append((i, pipe)); continue
                        if res.get("status") == "ERROR":
                            raise RuntimeError(f"Worker {i}: {res.get('msg')}")
                        if expected_status and res.get("status") != expected_status:
                            raise RuntimeError(
                                f"Worker {i}: expected {expected_status}, "
                                f"got {res.get('status')}")
                        results[i] = res
                    except (EOFError, OSError, BrokenPipeError):
                        raise RuntimeError(f"Worker {i} connection crashed.")
                else:
                    still.append((i, pipe))
            pending = still

        return results

    def anneal_to_ground_state(
        self,
        total_steps: int,
        dt: float,
        target_g_face: float,
        target_J: float,
        target_hx: float,
        target_hz: float,
        measure_every: int = 1
    ) -> None:
        print(f"\nStarting Adiabatic Anneal with Stochastic Injection...")
        if measure_every > 1:
            print(f"Measurement cadence: every {measure_every} steps")
        self.energy_history.clear()
        noise_rng = np.random.default_rng()
        effective_shots = 512.0

        for t in range(total_steps):
            s = t / max(1, (total_steps-1))
            current_hx     = (1.0-s)*3.0 + s*target_hx
            current_J      = s*target_J
            current_hz     = s*target_hz
            current_g_face = s*target_g_face
            is_measure     = (t % measure_every == 0) or (t == total_steps-1)

            step_payload = [
                {"J":current_J, "hx":current_hx, "hz":current_hz, "dt":dt, "steps":1}
                for _ in range(self.num_patches)
            ]

            if not is_measure:
                self.sync_broadcast("EVOLVE_ONLY", step_payload,
                                    expected_status="EVOLVE_COMPLETE")
                print(f"Step {t:03d} | Anneal: {s*100:05.1f}% | (evolve-only)")
                continue

            step1_res = self.sync_broadcast(
                "EVOLVE_AND_MEASURE_STATISTICAL", step_payload,
                expected_status="STEP_1_COMPLETE")

            patch_profiles = {p: res["data"] for p,res in step1_res.items()}
            kick_payloads = [
                {"kicks":{}, "J":current_J, "hx":current_hx, "hz":current_hz}
                for _ in range(self.num_patches)
            ]
            macroscopic_boundary_energy = 0.0
            scale = np.sqrt(dt / effective_shots)

            stochastic_noise: Dict[int,Dict[int,Tuple[float,float,float]]] = {}
            for p, prof in patch_profiles.items():
                n_bounds = len(prof["qubits"])
                X_noise = noise_rng.normal(0.0,1.0,n_bounds)*np.sqrt(prof["vars"]["X"])*scale
                Y_noise = noise_rng.normal(0.0,1.0,n_bounds)*np.sqrt(prof["vars"]["Y"])*scale
                Z_noise = noise_rng.normal(0.0,1.0,n_bounds)*np.sqrt(prof["vars"]["Z"])*scale
                for arr in (X_noise, Y_noise, Z_noise):
                    if not np.all(np.isfinite(arr)): arr[:] = 0.0
                stochastic_noise[p] = {
                    q: (X_noise[i], Y_noise[i], Z_noise[i])
                    for i,q in enumerate(prof["qubits"])
                }

            for p1, coord1 in self.patch_coords.items():
                x1, y1, z1 = coord1
                neighbors = {
                    "+X":(x1+1,y1,z1), "-X":(x1-1,y1,z1),
                    "+Y":(x1,y1+1,z1), "-Y":(x1,y1-1,z1),
                    "+Z":(x1,y1,z1+1), "-Z":(x1,y1,z1-1),
                }
                for dir1, coord2 in neighbors.items():
                    if not (0<=coord2[0]<self.grid_x and
                            0<=coord2[1]<self.grid_y and
                            0<=coord2[2]<self.grid_z): continue
                    p2 = self.coord_to_patch.get(coord2)
                    if p2 is None or p1 >= p2: continue

                    dir2 = dir1.replace("+","temp").replace("-","+").replace("temp","-")
                    face1_q = self.boundaries[dir1]
                    face2_q = self.boundaries[dir2]

                    prof1, noise1 = patch_profiles[p1], stochastic_noise[p1]
                    prof2, noise2 = patch_profiles[p2], stochastic_noise[p2]

                    q_to_i1 = {q:i for i,q in enumerate(prof1["qubits"])}
                    q_to_i2 = {q:i for i,q in enumerate(prof2["qubits"])}

                    ax2=ay2=az2=0.0
                    for q2 in face2_q:
                        i2 = q_to_i2[q2]
                        ax2 += prof2["means"]["X"][i2] + noise2[q2][0]
                        ay2 += prof2["means"]["Y"][i2] + noise2[q2][1]
                        az2 += prof2["means"]["Z"][i2] + noise2[q2][2]
                    n2 = max(1,len(face2_q)); ax2/=n2; ay2/=n2; az2/=n2

                    ax1=ay1=az1=0.0
                    for q1 in face1_q:
                        i1 = q_to_i1[q1]
                        ax1 += prof1["means"]["X"][i1] + noise1[q1][0]
                        ay1 += prof1["means"]["Y"][i1] + noise1[q1][1]
                        az1 += prof1["means"]["Z"][i1] + noise1[q1][2]
                    n1 = max(1,len(face1_q)); ax1/=n1; ay1/=n1; az1/=n1

                    interaction_E = (
                        -current_g_face
                        * (ax1*ax2 + ay1*ay2 + az1*az2)
                        * ((len(face1_q)+len(face2_q)) / 2.0)
                    )
                    macroscopic_boundary_energy += interaction_E

                    for q1 in face1_q:
                        k = kick_payloads[p1]["kicks"].get(q1, (0.,0.,0.))
                        kick_payloads[p1]["kicks"][q1] = (
                            k[0]+current_g_face*ax2,
                            k[1]+current_g_face*ay2,
                            k[2]+current_g_face*az2)
                    for q2 in face2_q:
                        k = kick_payloads[p2]["kicks"].get(q2, (0.,0.,0.))
                        kick_payloads[p2]["kicks"][q2] = (
                            k[0]+current_g_face*ax1,
                            k[1]+current_g_face*ay1,
                            k[2]+current_g_face*az1)

            step2_res = self.sync_broadcast(
                "APPLY_KICKS_AND_MEASURE_ENERGY", kick_payloads,
                expected_status="STEP_2_COMPLETE")
            bulk_energy = sum(r["data"] for r in step2_res.values())
            total_energy = bulk_energy + macroscopic_boundary_energy

            print(f"Step {t:03d} | Anneal: {s*100:05.1f}% | "
                  f"Total Setup Potential Energy: {total_energy:+.4f}")

            step_data = {"Step":t, "Anneal_Percent":s*100, "Energy":total_energy}
            self.energy_history.append(step_data)
            self._append_to_csv(step_data)

            if t == total_steps-1:
                print(f"         +-- Calculating Final Benchmarks...")
                bench_res = self.sync_broadcast(
                    "COMPUTE_BENCHMARKS", expected_status="BENCHMARKS_COMPUTED")
                avg_purity = (
                    sum(res["data"]["grid_purity"] for res in bench_res.values())
                    / self.num_patches
                )
                print(f"         +-- Avg Sub-Volume Purity: {avg_purity:.4f}")

    def shutdown(self) -> None:
        if self._is_shutdown: return
        self._is_shutdown = True
        print("\nShutting down Host-RAM Sliding Window Engine...")
        for pipe in getattr(self, 'pipes', []):
            try:
                if not pipe.closed:
                    while pipe.poll(timeout=0.05): pipe.recv()
                    pipe.send({"action": "SHUTDOWN"})
                    while pipe.poll(timeout=0.1): pipe.recv()
            except (OSError, BrokenPipeError, EOFError): pass
        time.sleep(0.5)
        for p in getattr(self, 'workers', []):
            try:
                p.join(timeout=3)
                if p.is_alive() and p.pid is not None:
                    os.kill(p.pid, signal.SIGKILL)
            except Exception: pass
        for pipe in getattr(self, 'pipes', []):
            try:
                if not pipe.closed: pipe.close()
            except (OSError, BrokenPipeError, EOFError): pass

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    mp.freeze_support()

    # Single AMD Instinct MI50 (16GB HBM2), device index overridable via env.
    gpu_0 = int(os.environ.get("WORMHOLE_GPU", "0"))

    # 4x4x1 = 16 patches, all routed to OpenCL device 0 (the AMD card).
    # VRAM: 3 concurrent sims x ~1.1 GB (fp32) = ~3.3 GB active.
    # Host RAM: 16 GB state + process overhead = ~32 GB steady, no transfer
    # spikes (zero-copy in_ket/out_ket, Fix 17). ~48 GB headroom on 80 GB.
    # Note: with Fix 17 the (4,4,2)=32-patch grid may be viable again
    # (~48-64 GB steady); verify with free -h during a 16-patch run first.
    target_grid = (4, 4, 1)
    explicit_gpu_allocation = [gpu_0] * 16

    semaphore_caps = {
        gpu_0: 3,
    }

    gpu_alloc_caps_mb = {
        gpu_0: 14000,
    }

    engine = VolumetricHadronEngine27Q(
        gpu_allocation=explicit_gpu_allocation,
        semaphore_limits=semaphore_caps,
        gpu_alloc_caps_mb=gpu_alloc_caps_mb,
        grid=target_grid,
        master_seed=1337
    )

    try:
        engine.anneal_to_ground_state(
            total_steps=100,
            dt=0.04,
            target_g_face=0.15,
            target_J=1.0,
            target_hx=0.5,
            target_hz=0.2,
            measure_every=1
        )
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    finally:
        engine.shutdown()
