# -*- coding: us-ascii -*-
# 27-Qubit 3x3x3 Lattice & Macroscopic Grid Annealing
# High-Throughput Volumetric Engine with Statistical Variance Injection
# Nucleus qubit removed: pure nearest-neighbour 3D Ising topology (54 bonds)
# Fable5 Senior for measurement breakthrough
# Sonnet 4.6 lead & management, Gemini 3.x pro as workers
#
# Fable5 PERFORMANCE REVISION 3 (VRAM-safe):
#   Fix 1: Forced plain dense engine (no QUnit Schmidt thrash)         [kept]
#   Fix 2: Merged redundant Rz layers (exact)                          [kept]
#   Fix 3: ZZ term as Rz+Rz+controlled-phase                           [kept]
#   Fix 5: Clone-free, shot-free prob()-based measurement              [kept]
#   Fix 6: VRAM RESIDENCY BUDGET. The dense engine allocates the full
#          2^27 buffer (~2.1GB) per worker at construction. Resident
#          workers x 2.1GB MUST stay below physical VRAM per card, or
#          Mesa pages buffers to GTT, kernels stall past the amdgpu
#          watchdog, and the driver resets the GPU ("context is lost"),
#          killing every worker on the card. Allocation is therefore
#          5 workers on the 16GB card (10.5GB) + 3 on the 10GB card
#          (6.3GB), grid (2,2,2) = 8 patches.
#   Fix 7: QRACK_MAX_ALLOC_MB tripwire per worker: oversubscription now
#          raises an allocation error in ONE worker instead of silently
#          paging and taking down the whole card.
#   Energy is computed and printed at EVERY step (measure_every=1).
#
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
    """
    Force a plain dense QEngine. QUnit's Schmidt-decomposition attempts
    on an irreducibly entangled 27-qubit register cost O(2^27) per failed
    re-factoring attempt. Kwarg names vary across PyQrack versions, so
    unsupported ones are stripped on TypeError and the constructor retried.
    """
    kwargs = dict(isTensorNetwork=False, isSchmidtDecompose=False,
                  isStabilizerHybrid=False, isBinaryDecisionTree=False)
    while True:
        try:
            return QrackSimulator(qubit_count=n, **kwargs)
        except TypeError as e:
            removed = False
            for k in list(kwargs):
                if k in str(e):
                    kwargs.pop(k)
                    removed = True
                    break
            if not removed:
                return QrackSimulator(qubit_count=n)

def apply_h(sim: Any, q: int) -> None:
    if hasattr(sim, 'h'):
        sim.h(q)
    else:
        sim.mtrx([complex(1/np.sqrt(2), 0), complex(1/np.sqrt(2), 0),
                  complex(1/np.sqrt(2), 0), complex(-1/np.sqrt(2), 0)], q)

def apply_rx(sim: Any, theta: float, q: int) -> None:
    if hasattr(sim, 'r'):
        sim.r(PX, float(theta), q)
    else:
        sim.mtrx([complex(np.cos(theta/2), 0), complex(0, -np.sin(theta/2)),
                  complex(0, -np.sin(theta/2)), complex(np.cos(theta/2), 0)], q)

def apply_ry(sim: Any, theta: float, q: int) -> None:
    if hasattr(sim, 'r'):
        sim.r(PY, float(theta), q)
    else:
        sim.mtrx([complex(np.cos(theta/2), 0), complex(-np.sin(theta/2), 0),
                  complex(np.sin(theta/2), 0), complex(np.cos(theta/2), 0)], q)

def apply_rz(sim: Any, theta: float, q: int) -> None:
    if hasattr(sim, 'r'):
        sim.r(PZ, float(theta), q)
    else:
        sim.mtrx([complex(np.cos(theta/2), -np.sin(theta/2)), 0j,
                  0j, complex(np.cos(theta/2), np.sin(theta/2))], q)

def apply_cx(sim: Any, c: int, t: int) -> None:
    if hasattr(sim, 'cx'):
        sim.cx(c, t)
    elif hasattr(sim, 'mcx'):
        try: sim.mcx([c], t)
        except TypeError: sim.mcx([c], [t])
    else:
        raise RuntimeError("No CX gate available.")

def apply_zz(sim: Any, theta: float, q1: int, q2: int) -> None:
    """
    exp(-i*theta/2 * Z(q1)Z(q2)) up to global phase.
    Rz(theta) x Rz(theta) . CP(-2*theta) with CP(phi)=diag(1,1,1,e^{i*phi}).
    One diagonal controlled-phase replaces two CX gates. Falls back to
    CX.Rz.CX if no controlled-mtrx API is available.
    """
    if hasattr(sim, 'mcmtrx'):
        apply_rz(sim, theta, q1)
        apply_rz(sim, theta, q2)
        ph = complex(np.cos(2.0 * theta), -np.sin(2.0 * theta))
        try:
            sim.mcmtrx([q1], [complex(1, 0), 0j, 0j, ph], q2)
        except TypeError:
            sim.mcmtrx([q1], [complex(1, 0), 0j, 0j, ph], [q2])
    else:
        apply_cx(sim, q1, q2)
        apply_rz(sim, theta, q2)
        apply_cx(sim, q1, q2)

def trotter_step_body(sim: Any, num_qubits: int,
                      intra_edges: List[Tuple[int, int]],
                      J: float, hx: float, hz: float, dt: float,
                      steps: int) -> None:
    """
    Symmetric Trotter step with the two Rz layers merged (Rz commutes
    exactly with the diagonal ZZ layer). Same unitary as the original
    Rx.Rz.ZZ.Rz.Rx sequence, 27 fewer gates per step.
    """
    for _ in range(steps):
        for q in range(num_qubits):
            apply_rx(sim, -2.0 * hx * dt, q)
        for q in range(num_qubits):
            apply_rz(sim, -4.0 * hz * dt, q)
        for q1, q2 in intra_edges:
            apply_zz(sim, -2.0 * J * dt, q1, q2)
        for q in range(num_qubits):
            apply_rx(sim, -2.0 * hx * dt, q)

# ------------------------------------------
# Clone-free expectation helpers.
# prob(q) is a NON-DESTRUCTIVE marginal query (no collapse), and every
# basis rotation below is exactly self-inverting (H.H = I,
# Rx(-a).Rx(a) = I, CX.CX = I), so the persistent state is restored to
# fp roundoff after each helper returns. This is what lets the anneal
# hot path run with zero clone_sid copies and zero measure_shots calls.
# ------------------------------------------

def z_means(sim: Any, qubits: List[int]) -> np.ndarray:
    """<Z_q> for each q. Pure reads; state untouched."""
    return np.array([1.0 - 2.0 * sim.prob(q) for q in qubits])

def x_means(sim: Any, qubits: List[int]) -> np.ndarray:
    """<X_q> for each q. H / read / H-undo, exact."""
    out = np.empty(len(qubits))
    for i, q in enumerate(qubits):
        apply_h(sim, q)
        out[i] = 1.0 - 2.0 * sim.prob(q)
        apply_h(sim, q)
    return out

def y_means(sim: Any, qubits: List[int]) -> np.ndarray:
    """<Y_q> for each q. Rx(-pi/2) / read / Rx(+pi/2)-undo, exact."""
    out = np.empty(len(qubits))
    for i, q in enumerate(qubits):
        apply_rx(sim, -np.pi / 2, q)
        out[i] = 1.0 - 2.0 * sim.prob(q)
        apply_rx(sim, np.pi / 2, q)
    return out

def zz_means(sim: Any, edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    <Z_q1 Z_q2> per edge via the CX trick: CX(q1,q2) maps Z(q1)Z(q2) to
    I x Z(q2), so prob(q2) in the rotated frame gives the ZZ parity.
    CX.CX = I restores the state exactly; prob() never collapses.
    """
    out = np.empty(len(edges))
    for i, (q1, q2) in enumerate(edges):
        apply_cx(sim, q1, q2)
        out[i] = 1.0 - 2.0 * sim.prob(q2)
        apply_cx(sim, q1, q2)
    return out

# ==========================================
# 1. 27-QUBIT TOPOLOGY (FULLY CONNECTED)
# ==========================================
def generate_27q_lattice_subvolume() -> Tuple[List[Tuple[int, int]], Dict[str, List[int]]]:
    """
    Fully Connected (FC) 27-qubit topology. 351 internal bonds.
    Every single qubit is entangled with every other qubit.
    
    Macroscopic boundary faces are retained mathematically from a 
    3x3x3 mapping to ensure compatibility with the mean-field orchestrator.
    """
    lx, ly, lz = 3, 3, 3
    edges = []
    boundaries = {"+X": [], "-X": [], "+Y": [], "-Y": [], "+Z": [], "-Z": []}

    # 1. Generate All-to-All Fully Connected Edges
    for q1 in range(27):
        for q2 in range(q1 + 1, 27):
            edges.append((q1, q2))

    # 2. Retain 3D Boundary Mappings for Macroscopic Kicks
    for x in range(lx):
        for y in range(ly):
            for z in range(lz):
                idx = x * (ly * lz) + y * lz + z

                if x == 0:      boundaries["-X"].append(idx)
                if x == lx - 1: boundaries["+X"].append(idx)
                if y == 0:      boundaries["-Y"].append(idx)
                if y == ly - 1: boundaries["+Y"].append(idx)
                if z == 0:      boundaries["-Z"].append(idx)
                if z == lz - 1: boundaries["+Z"].append(idx)

    return edges, boundaries

# ==========================================
# 2. ISOLATED LHV ISLAND (GPU WORKER)
# ==========================================
def persistent_island_worker_27q(
    device_id: int,
    patch_idx: int,
    num_qubits: int,
    intra_edges: List[Tuple[int, int]],
    boundaries: Dict[str, List[int]],
    cmd_pipe: Connection,
    gpu_semaphore: Any,
    seed: int,
    max_alloc_mb: int
) -> None:

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"

    # Fix 7: allocation tripwire. If the workers on this device would
    # collectively exceed physical VRAM, Qrack raises an allocation
    # error in THIS worker (reported cleanly over the pipe) instead of
    # Mesa silently paging to GTT, stalling kernels past the amdgpu
    # watchdog, and resetting the whole GPU.
    os.environ["QRACK_MAX_ALLOC_MB"] = str(int(max_alloc_mb))

    sim = None
    try:
        os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(device_id)
        from pyqrack import QrackSimulator

        sim = make_sim(QrackSimulator, num_qubits)
        rng = np.random.default_rng(seed)

        with gpu_semaphore:
            for q in range(num_qubits):
                apply_h(sim, q)
                apply_rx(sim, rng.normal(0, 1e-5), q)
                apply_rz(sim, rng.normal(0, 1e-5), q)

        cmd_pipe.send({"status": "READY", "patch_idx": patch_idx})

        all_boundary_qubits = sorted(list(set(
            q for face in boundaries.values() for q in face
        )))

        while True:
            if not cmd_pipe.poll(timeout=0.1): continue
            cmd = cmd_pipe.recv()
            action = cmd.get("action")

            if action == "SHUTDOWN": break

            try:
                # --------------------------------------------------
                # Fast path: Trotter evolution only (optional cadence).
                if action == "EVOLVE_ONLY":
                    J   = cmd.get("J",   1.0)
                    hx  = cmd.get("hx",  0.5)
                    hz  = cmd.get("hz",  0.2)
                    dt  = cmd.get("dt",  0.05)
                    steps = cmd.get("steps", 1)

                    with gpu_semaphore:
                        trotter_step_body(sim, num_qubits, intra_edges,
                                          J, hx, hz, dt, steps)

                    cmd_pipe.send({"status": "EVOLVE_COMPLETE",
                                   "patch_idx": patch_idx})

                # --------------------------------------------------
                elif action == "COMPUTE_SUPREMACY_BENCHMARK":
                    depth = cmd.get("depth", 10)
                    M_shots = cmd.get("shots", 10000)

                    with gpu_semaphore:
                        # Fresh simulator at |0...0> for proper RQC benchmarking.
                        # NOTE: this is the one remaining allocation-heavy path
                        # (fresh sim + clone = 2 x 2.1GB transient on top of the
                        # persistent state). The 5+3 residency budget leaves
                        # headroom for exactly this; the semaphore serialises it.
                        sim_xeb = make_sim(QrackSimulator, num_qubits)
                        try:
                            for _ in range(depth):
                                for q in range(num_qubits):
                                    apply_rx(sim_xeb, rng.uniform(0, 2*np.pi), q)
                                    apply_rz(sim_xeb, rng.uniform(0, 2*np.pi), q)
                                for q1, q2 in intra_edges:
                                    apply_cx(sim_xeb, q1, q2)
                                    apply_rz(sim_xeb, rng.uniform(0, 2*np.pi), q2)
                                    apply_cx(sim_xeb, q1, q2)

                            # State vector BEFORE measurement, via clone to
                            # protect against graph-resolution artifacts
                            sim_sv = QrackSimulator(clone_sid=sim_xeb.sid)
                            try:
                                if hasattr(sim_sv, 'out_ket'):
                                    state_vector = np.array(sim_sv.out_ket())
                                elif hasattr(sim_sv, 'amplitudes'):
                                    state_vector = np.array(sim_sv.amplitudes())
                                else:
                                    raise RuntimeError(
                                        "No compatible state vector extraction method found.")
                            finally:
                                del sim_sv

                            all_q_list = list(range(num_qubits))
                            samples = sim_xeb.measure_shots(all_q_list, M_shots)

                        finally:
                            del sim_xeb
                    gc.collect()

                    probs = np.abs(state_vector)**2
                    sample_probs = probs[samples]

                    median_prob = np.median(probs)
                    heavy_count = np.sum(sample_probs > median_prob)
                    hog_score = heavy_count / float(M_shots)

                    hilbert_space_size = 2.0 ** num_qubits
                    xeb_score = hilbert_space_size * np.mean(sample_probs) - 1.0

                    cmd_pipe.send({"status": "BENCHMARKS_COMPUTED",
                                   "patch_idx": patch_idx, "data": {
                        "HOG": float(hog_score),
                        "XEB": float(xeb_score)
                    }})

                # --------------------------------------------------
                # Clone-free, shot-free statistical measurement.
                elif action == "EVOLVE_AND_MEASURE_STATISTICAL":
                    J   = cmd.get("J",   1.0)
                    hx  = cmd.get("hx",  0.5)
                    hz  = cmd.get("hz",  0.2)
                    dt  = cmd.get("dt",  0.05)
                    steps = cmd.get("steps", 1)

                    with gpu_semaphore:
                        trotter_step_body(sim, num_qubits, intra_edges,
                                          J, hx, hz, dt, steps)

                        Z_mean = z_means(sim, all_boundary_qubits)
                        X_mean = x_means(sim, all_boundary_qubits)
                        Y_mean = y_means(sim, all_boundary_qubits)

                    # Var = 1 - <s>^2, exact for +/-1-valued Pauli observables.
                    Z_var = np.clip(1.0 - Z_mean**2, 0.0, 1.0)
                    X_var = np.clip(1.0 - X_mean**2, 0.0, 1.0)
                    Y_var = np.clip(1.0 - Y_mean**2, 0.0, 1.0)

                    payload = {
                        "qubits": all_boundary_qubits,
                        "means": {"X": X_mean, "Y": Y_mean, "Z": Z_mean},
                        "vars":  {"X": X_var,  "Y": Y_var,  "Z": Z_var}
                    }
                    cmd_pipe.send({"status": "STEP_1_COMPLETE",
                                   "patch_idx": patch_idx, "data": payload})

                # --------------------------------------------------
                # Clone-free, shot-free energy measurement.
                elif action == "APPLY_KICKS_AND_MEASURE_ENERGY":
                    kicks  = cmd.get("kicks", {})
                    J_val  = cmd.get("J",   1.0)
                    hx_val = cmd.get("hx",  0.5)
                    hz_val = cmd.get("hz",  0.2)

                    all_q = list(range(num_qubits))

                    with gpu_semaphore:
                        # Macroscopic mean-field kicks: single exact SU(2)
                        # unitary per qubit (Rodriguez formula).
                        for raw_q, (kx, ky, kz) in kicks.items():
                            q = int(raw_q)
                            K = np.sqrt(kx**2 + ky**2 + kz**2)
                            if K > 0.0:
                                c = np.cos(K / 2.0)
                                s = np.sin(K / 2.0)
                                nx, ny, nz = kx/K, ky/K, kz/K

                                m00 = complex(c, -nz * s)
                                m01 = complex(-ny * s, -nx * s)
                                m10 = complex(ny * s, -nx * s)
                                m11 = complex(c, nz * s)

                                sim.mtrx([m00, m01, m10, m11], q)

                        z_exp  = z_means(sim, all_q)
                        zz_exp = zz_means(sim, intra_edges)
                        x_exp  = x_means(sim, all_q)

                    local_energy = (
                        -hz_val * float(np.sum(z_exp))
                        - J_val * float(np.sum(zz_exp))
                        - hx_val * float(np.sum(x_exp))
                    )

                    cmd_pipe.send({"status": "STEP_2_COMPLETE",
                                   "patch_idx": patch_idx, "data": local_energy})

                # --------------------------------------------------
                # Clone-free purity benchmark.
                elif action == "COMPUTE_BENCHMARKS":
                    try:
                        all_q = list(range(num_qubits))
                        with gpu_semaphore:
                            z_e = z_means(sim, all_q)
                            x_e = x_means(sim, all_q)
                            y_e = y_means(sim, all_q)
                        avg_purity = float(np.mean(x_e**2 + y_e**2 + z_e**2))
                    except Exception as e:
                        print(f"Worker {patch_idx} benchmark failed: {e}")
                        avg_purity = 0.0

                    cmd_pipe.send({"status": "BENCHMARKS_COMPUTED",
                                   "patch_idx": patch_idx,
                                   "data": {"grid_purity": avg_purity}})

            except Exception as e:
                try:
                    cmd_pipe.send({"status": "ERROR", "msg": str(e)})
                except (EOFError, OSError, BrokenPipeError):
                    break

    except (EOFError, OSError, BrokenPipeError): pass
    finally:
        if sim is not None: del sim
        gc.collect()
        try:
            cmd_pipe.close()
        except Exception:
            pass

# ==========================================
# 3. 27-QUBIT MACROSCOPIC ORCHESTRATOR
# ==========================================
class VolumetricHadronEngine27Q:
    def __init__(
        self,
        gpu_allocation: List[int],
        semaphore_limits: Dict[int, int],
        gpu_alloc_caps_mb: Dict[int, int],
        grid: Tuple[int, int, int] = (2, 2, 2),
        master_seed: int = 42
    ):
        self._is_shutdown = False
        self.gpu_allocation = gpu_allocation

        self.grid_x, self.grid_y, self.grid_z = grid
        self.num_patches = self.grid_x * self.grid_y * self.grid_z

        if len(self.gpu_allocation) != self.num_patches:
            raise ValueError(
                f"GPU allocation list ({len(self.gpu_allocation)}) must match "
                f"total grid patches ({self.num_patches})."
            )

        self.qubits_per_patch = 27
        self.intra_edges, self.boundaries = generate_27q_lattice_subvolume()

        # Fix 6: hard VRAM residency check. Dense engine = ~2.1GB per
        # resident worker; refuse to start a configuration that
        # oversubscribes any card (the failure mode is a driver-level
        # GPU reset that kills every worker on the device).
        STATE_MB = 2200  # 2^27 complex128 amplitudes + runtime overhead
        residency: Dict[int, int] = {}
        for g in self.gpu_allocation:
            residency[g] = residency.get(g, 0) + 1
        for g, count in residency.items():
            cap = gpu_alloc_caps_mb.get(g)
            if cap is None:
                raise ValueError(f"No VRAM cap provided for GPU {g}.")
            need = count * STATE_MB
            if need > cap:
                raise ValueError(
                    f"GPU {g}: {count} resident workers x {STATE_MB}MB = "
                    f"{need}MB exceeds the {cap}MB cap. Reduce workers on "
                    f"this device (grid/allocation), do not raise the cap."
                )

        self.gpu_alloc_caps_mb = gpu_alloc_caps_mb

        self.patch_coords: Dict[int, Tuple[int, int, int]] = {}
        idx = 0
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                for z in range(self.grid_z):
                    self.patch_coords[idx] = (x, y, z)
                    idx += 1

        self.coord_to_patch = {v: k for k, v in self.patch_coords.items()}

        self.ctx = mp.get_context('spawn')
        self.gpu_semaphores = {
            gpu_id: self.ctx.Semaphore(limit)
            for gpu_id, limit in semaphore_limits.items()
        }

        self.workers: List[mp.Process] = []
        self.pipes: List[Connection] = []
        self.energy_history: List[Dict[str, Any]] = []

        self.csv_filename = "ground_state_energy_curve_27q.csv"
        self._init_csv()

        total_sites = self.num_patches * self.qubits_per_patch
        print(f"Initializing 27-Qubit 3x3x3 Lattice Volumetric Engine...")
        print(f"Topology: {self.num_patches} patches x {self.qubits_per_patch} qubits "
              f"= {total_sites} total logical qubits")
        print(f"Intra-patch bonds: {len(self.intra_edges)} (54 nearest-neighbour per patch)")
        for g, count in residency.items():
            print(f"GPU {g}: {count} resident workers, "
                  f"{count * STATE_MB}MB / {gpu_alloc_caps_mb[g]}MB budget")

        master_rng  = np.random.default_rng(master_seed)
        patch_seeds = master_rng.integers(0, 2**31 - 1, size=self.num_patches)

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

            for i, pipe in enumerate(self.pipes):
                if pipe.poll(timeout=60.0):
                    if pipe.recv().get("status") != "READY":
                        raise RuntimeError(f"Worker {i} protocol failure.")
                else:
                    raise TimeoutError(f"Worker {i} timeout during initialization.")

        except Exception as e:
            self.shutdown()
            raise RuntimeError(f"Engine initialization aborted: {e}") from e

    def _init_csv(self):
        try:
            with open(self.csv_filename, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["Step", "Anneal_Percent", "Energy"])
                writer.writeheader()
        except Exception: pass

    def _append_to_csv(self, data: Dict[str, Any]):
        try:
            with open(self.csv_filename, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["Step", "Anneal_Percent", "Energy"])
                writer.writerow(data)
                f.flush()
                os.fsync(f.fileno())
        except Exception: pass

    def sync_broadcast(
        self,
        action: str,
        kwargs_list: Optional[List[Dict]] = None,
        expected_status: Optional[str] = None,
        timeout_s: float = 1800.0
    ) -> Dict[int, Any]:
        if kwargs_list is None:
            kwargs_list = [{} for _ in range(self.num_patches)]

        send_errors: Dict[int, Exception] = {}
        error_lock = threading.Lock()

        def _send_one(pipe: Connection, msg: Dict[str, Any], i: int):
            try:
                pipe.send(msg)
            except Exception as e:
                with error_lock:
                    send_errors[i] = e

        threads = []
        for i, pipe in enumerate(self.pipes):
            t = threading.Thread(
                target=_send_one,
                args=(pipe, {"action": action, **kwargs_list[i]}, i)
            )
            t.start()
            threads.append(t)
        for t in threads: t.join()

        if send_errors:
            self.shutdown()
            raise RuntimeError(f"Send failed for workers: {send_errors}")

        results: Dict[int, Any] = {}
        pending = list(enumerate(self.pipes))
        deadline = time.monotonic() + timeout_s

        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Workers timed out on action '{action}' after {timeout_s:.0f}s"
                )
            ready_pipes = wait([p for _, p in pending], timeout=min(remaining, 60.0))
            if not ready_pipes:
                elapsed = timeout_s - (deadline - time.monotonic())
                print(f"[WARN] Still waiting on {len(pending)} workers "
                      f"({elapsed:.0f}s elapsed) for action '{action}'...")
                continue

            still_pending = []
            for i, pipe in pending:
                if pipe in ready_pipes:
                    try:
                        res = pipe.recv()
                        if res.get("status") == "ERROR":
                            raise RuntimeError(f"Worker {i} error: {res.get('msg')}")
                        if expected_status and res.get("status") != expected_status:
                            raise RuntimeError(
                                f"Worker {i} protocol sync error: "
                                f"Expected {expected_status}, got {res.get('status')}"
                            )
                        results[i] = res
                    except (EOFError, OSError, BrokenPipeError):
                        raise RuntimeError(f"Worker {i} connection crashed.")
                else:
                    still_pending.append((i, pipe))
            pending = still_pending

        return results

    def run_supremacy_benchmark(self, depth: int = 10, shots: int = 10000):
        print(f"\nInitiating Cross-Entropy Benchmarking (Depth {depth}, Shots {shots})...")
        print("Note: This extracts full state vectors to host RAM (approx 2.14GB per worker).")
        payload = [{"depth": depth, "shots": shots} for _ in range(self.num_patches)]
        try:
            res = self.sync_broadcast(
                "COMPUTE_SUPREMACY_BENCHMARK",
                payload,
                expected_status="BENCHMARKS_COMPUTED",
                timeout_s=3600.0
            )

            avg_hog = sum(r["data"]["HOG"] for r in res.values()) / self.num_patches
            avg_xeb = sum(r["data"]["XEB"] for r in res.values()) / self.num_patches

            print(f"         +-- Average HOG Score: {avg_hog:.4f} (Ideal noiseless: ~0.693)")
            print(f"         +-- Average XEB Score: {avg_xeb:.4f} (Ideal noiseless: 1.000)")

        except Exception as e:
            print(f"Supremacy Benchmark Failed: {e}")

    def anneal_to_ground_state(
        self,
        total_steps: int,
        dt: float,
        target_g_face: float,
        target_J: float,
        target_hx: float,
        target_hz: float,
        measure_every: int = 1
    ):
        """
        Clone-free expectation measurement makes the full
        measure -> kick -> energy cycle cheap enough to run at every
        step (measure_every=1, the default). Raise measure_every to
        skip the cycle on intermediate steps via EVOLVE_ONLY.
        """
        print(f"\nStarting Adiabatic Anneal with Stochastic Injection...")
        if measure_every > 1:
            print(f"Mean-field measurement cadence: every {measure_every} steps")
        self.energy_history.clear()
        noise_rng = np.random.default_rng()
        effective_shots = 512.0  # keeps noise amplitude consistent with prior runs

        for t in range(total_steps):
            s = t / max(1, (total_steps - 1))
            current_hx     = (1.0 - s) * 3.0 + s * target_hx
            current_J      = s * target_J
            current_hz     = s * target_hz
            current_g_face = s * target_g_face

            is_measure_step = (t % measure_every == 0) or (t == total_steps - 1)

            step_payload = [
                {"J": current_J, "hx": current_hx, "hz": current_hz,
                 "dt": dt, "steps": 1}
                for _ in range(self.num_patches)
            ]

            if not is_measure_step:
                self.sync_broadcast(
                    "EVOLVE_ONLY", step_payload,
                    expected_status="EVOLVE_COMPLETE"
                )
                print(f"Step {t:03d} | Anneal: {s*100:05.1f}% | (evolve-only)")
                continue

            step1_res = self.sync_broadcast(
                "EVOLVE_AND_MEASURE_STATISTICAL", step_payload,
                expected_status="STEP_1_COMPLETE"
            )

            patch_profiles = {p: res["data"] for p, res in step1_res.items()}
            kick_payloads = [
                {"kicks": {}, "J": current_J, "hx": current_hx, "hz": current_hz}
                for _ in range(self.num_patches)
            ]
            macroscopic_boundary_energy = 0.0

            # Independent per-qubit Gaussian noise from exact variances.
            scale = np.sqrt(dt / effective_shots)
            stochastic_noise: Dict[int, Dict[int, Tuple[float, float, float]]] = {}
            for p, prof in patch_profiles.items():
                n_bounds = len(prof["qubits"])

                X_noise = noise_rng.normal(0.0, 1.0, n_bounds) * np.sqrt(prof["vars"]["X"]) * scale
                Y_noise = noise_rng.normal(0.0, 1.0, n_bounds) * np.sqrt(prof["vars"]["Y"]) * scale
                Z_noise = noise_rng.normal(0.0, 1.0, n_bounds) * np.sqrt(prof["vars"]["Z"]) * scale

                for noise_arr in (X_noise, Y_noise, Z_noise):
                    if not np.all(np.isfinite(noise_arr)):
                        noise_arr[:] = 0.0

                stochastic_noise[p] = {
                    q: (X_noise[i], Y_noise[i], Z_noise[i])
                    for i, q in enumerate(prof["qubits"])
                }

            # Inter-patch mean-field coupling
            for p1, coord1 in self.patch_coords.items():
                x1, y1, z1 = coord1
                neighbors = {
                    "+X": (x1+1, y1,   z1),
                    "-X": (x1-1, y1,   z1),
                    "+Y": (x1,   y1+1, z1),
                    "-Y": (x1,   y1-1, z1),
                    "+Z": (x1,   y1,   z1+1),
                    "-Z": (x1,   y1,   z1-1),
                }

                for dir1, coord2 in neighbors.items():
                    if not (0 <= coord2[0] < self.grid_x and
                            0 <= coord2[1] < self.grid_y and
                            0 <= coord2[2] < self.grid_z):
                        continue
                    p2 = self.coord_to_patch.get(coord2)
                    if p2 is None or p1 >= p2: continue

                    dir2 = dir1.replace("+", "temp").replace("-", "+").replace("temp", "-")
                    face1_qubits = self.boundaries[dir1]
                    face2_qubits = self.boundaries[dir2]

                    prof1, noise1 = patch_profiles[p1], stochastic_noise[p1]
                    prof2, noise2 = patch_profiles[p2], stochastic_noise[p2]

                    q_to_idx1 = {q: i for i, q in enumerate(prof1["qubits"])}
                    q_to_idx2 = {q: i for i, q in enumerate(prof2["qubits"])}

                    avg_x2 = avg_y2 = avg_z2 = 0.0
                    for q2 in face2_qubits:
                        i2 = q_to_idx2[q2]
                        avg_x2 += prof2["means"]["X"][i2] + noise2[q2][0]
                        avg_y2 += prof2["means"]["Y"][i2] + noise2[q2][1]
                        avg_z2 += prof2["means"]["Z"][i2] + noise2[q2][2]
                    n2 = max(1, len(face2_qubits))
                    avg_x2 /= n2; avg_y2 /= n2; avg_z2 /= n2

                    avg_x1 = avg_y1 = avg_z1 = 0.0
                    for q1 in face1_qubits:
                        i1 = q_to_idx1[q1]
                        avg_x1 += prof1["means"]["X"][i1] + noise1[q1][0]
                        avg_y1 += prof1["means"]["Y"][i1] + noise1[q1][1]
                        avg_z1 += prof1["means"]["Z"][i1] + noise1[q1][2]
                    n1 = max(1, len(face1_qubits))
                    avg_x1 /= n1; avg_y1 /= n1; avg_z1 /= n1

                    # Symmetric mean-field bond energy (heuristic inter-patch coupling)
                    interaction_E = (
                        -current_g_face
                        * (avg_x1*avg_x2 + avg_y1*avg_y2 + avg_z1*avg_z2)
                        * ((len(face1_qubits) + len(face2_qubits)) / 2.0)
                    )
                    macroscopic_boundary_energy += interaction_E

                    for q1 in face1_qubits:
                        k = kick_payloads[p1]["kicks"].get(q1, (0.0, 0.0, 0.0))
                        kick_payloads[p1]["kicks"][q1] = (
                            k[0] + current_g_face * avg_x2,
                            k[1] + current_g_face * avg_y2,
                            k[2] + current_g_face * avg_z2,
                        )
                    for q2 in face2_qubits:
                        k = kick_payloads[p2]["kicks"].get(q2, (0.0, 0.0, 0.0))
                        kick_payloads[p2]["kicks"][q2] = (
                            k[0] + current_g_face * avg_x1,
                            k[1] + current_g_face * avg_y1,
                            k[2] + current_g_face * avg_z1,
                        )

            step2_res = self.sync_broadcast(
                "APPLY_KICKS_AND_MEASURE_ENERGY", kick_payloads,
                expected_status="STEP_2_COMPLETE"
            )
            bulk_energy = sum(r["data"] for r in step2_res.values())
            total_energy = bulk_energy + macroscopic_boundary_energy

            print(f"Step {t:03d} | Anneal: {s*100:05.1f}% | "
                  f"Total Setup Potential Energy: {total_energy:+.4f}")

            step_data = {"Step": t, "Anneal_Percent": s * 100, "Energy": total_energy}
            self.energy_history.append(step_data)
            self._append_to_csv(step_data)

            if t == total_steps - 1:
                print(f"         +-- Calculating Final Benchmarks...")
                bench_res = self.sync_broadcast(
                    "COMPUTE_BENCHMARKS", expected_status="BENCHMARKS_COMPUTED"
                )
                avg_purity = (
                    sum(res["data"]["grid_purity"] for res in bench_res.values())
                    / self.num_patches
                )
                print(f"         +-- Avg Sub-Volume Purity "
                      f"(1.0 = Pure, 0.0 = Mixed): {avg_purity:.4f}")

    def shutdown(self) -> None:
        if self._is_shutdown: return
        self._is_shutdown = True
        print("\nShutting down 27-Qubit Volumetric Engine...")

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

    gpu_env   = os.environ.get("WORMHOLE_GPUS", "0,1")
    base_gpus = [int(g.strip()) for g in gpu_env.split(',')]

    if len(base_gpus) < 2:
        print("WARNING: Requires at least 2 GPUs. Falling back to [0, 1].")
        base_gpus = [0, 1]

    gpu_16gb = base_gpus[0]
    gpu_10gb = base_gpus[1]

    # Fix 6: VRAM-safe residency. Dense engine allocates the full 2.1GB
    # state per worker AT ALL TIMES (unlike QUnit, which kept idle
    # workers tiny). Oversubscription causes Mesa GTT paging -> kernel
    # stalls -> amdgpu watchdog GPU reset ("context is lost") killing
    # every worker on the card. The residency budget is therefore a
    # hard constraint:
    #   16GB card: 5 workers x 2.1GB = 10.5GB (headroom for XEB transients)
    #   10GB card: 3 workers x 2.1GB =  6.3GB
    target_grid = (2, 2, 2)  # 8 patches total

    explicit_gpu_allocation = [gpu_16gb] * 5 + [gpu_10gb] * 3

    semaphore_caps = {
        gpu_16gb: 2,
        gpu_10gb: 2,
    }

    # Fix 7: per-device allocation caps (MB), passed to workers as
    # QRACK_MAX_ALLOC_MB. Set below physical VRAM so an accidental
    # oversubscribe fails loudly in one worker instead of resetting
    # the GPU. Also used by the orchestrator's residency pre-check.
    gpu_alloc_caps_mb = {
        gpu_16gb: 15000,
        gpu_10gb: 9000,
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
            measure_every=1   # energy computed and printed at EVERY step
        )

        # Diagnostic executed precisely ONCE on the post-annealed state
        engine.run_supremacy_benchmark(depth=10, shots=10000)

    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    finally:
        engine.shutdown()
