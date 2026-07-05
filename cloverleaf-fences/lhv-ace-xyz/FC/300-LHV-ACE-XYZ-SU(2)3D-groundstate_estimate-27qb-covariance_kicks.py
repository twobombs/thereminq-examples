# -*- coding: us-ascii -*-
# 27-Qubit 3x3x3 Lattice & Macroscopic Grid Annealing
# High-Throughput Volumetric Engine with Statistical Covariance Injection
# Nucleus qubit removed: pure nearest-neighbour 3D Ising topology (54 bonds)
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

def apply_h(sim: Any, q: int) -> None:
    if hasattr(sim, 'h'):
        sim.h(q)
    else:
        sim.mtrx([complex(1/np.sqrt(2), 0), complex(1/np.sqrt(2), 0),
                  complex(1/np.sqrt(2), 0), complex(-1/np.sqrt(2), 0)], [q])

def apply_rx(sim: Any, theta: float, q: int) -> None:
    if hasattr(sim, 'r'):
        sim.r(PX, float(theta), q)
    else:
        sim.mtrx([complex(np.cos(theta/2), 0), complex(0, -np.sin(theta/2)),
                  complex(0, -np.sin(theta/2)), complex(np.cos(theta/2), 0)], [q])

def apply_ry(sim: Any, theta: float, q: int) -> None:
    if hasattr(sim, 'r'):
        sim.r(PY, float(theta), q)
    else:
        sim.mtrx([complex(np.cos(theta/2), 0), complex(-np.sin(theta/2), 0),
                  complex(np.sin(theta/2), 0), complex(np.cos(theta/2), 0)], [q])

def apply_rz(sim: Any, theta: float, q: int) -> None:
    if hasattr(sim, 'r'):
        sim.r(PZ, float(theta), q)
    else:
        sim.mtrx([complex(np.cos(theta/2), -np.sin(theta/2)), 0j,
                  0j, complex(np.cos(theta/2), np.sin(theta/2))], [q])

def apply_cx(sim: Any, c: int, t: int) -> None:
    if hasattr(sim, 'cx'):
        sim.cx(c, t)
    elif hasattr(sim, 'mcx'):
        try: sim.mcx([c], t)
        except TypeError: sim.mcx([c], [t])
    else:
        raise RuntimeError("No CX gate available.")

# ==========================================
# 1. 27-QUBIT TOPOLOGY (3x3x3 LATTICE ONLY)
# ==========================================
def generate_27q_lattice_subvolume() -> Tuple[List[Tuple[int, int]], Dict[str, List[int]]]:
    """
    Pure 3x3x3 nearest-neighbour lattice. 27 qubits, 54 bonds.
    No nucleus qubit. Boundary faces are the outer planes of the cube.
    Qubit index: idx = x*(ly*lz) + y*lz + z, with lx=ly=lz=3.
    """
    lx, ly, lz = 3, 3, 3
    edges = []
    boundaries = {"+X": [], "-X": [], "+Y": [], "-Y": [], "+Z": [], "-Z": []}

    for x in range(lx):
        for y in range(ly):
            for z in range(lz):
                idx = x * (ly * lz) + y * lz + z

                if x < lx - 1: edges.append((idx, (x + 1) * (ly * lz) + y * lz + z))
                if y < ly - 1: edges.append((idx, x * (ly * lz) + (y + 1) * lz + z))
                if z < lz - 1: edges.append((idx, x * (ly * lz) + y * lz + (z + 1)))

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
    seed: int
) -> None:

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # NUMA Fix: Force single-threaded dispatch to prevent HyperTransport thrashing
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"

    sim = None
    try:
        os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(device_id)
        from pyqrack import QrackSimulator

        sim = QrackSimulator(qubit_count=num_qubits)
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
        num_bound = len(all_boundary_qubits)

        def extract_bits(samples: List[int], n_q: int) -> np.ndarray:
            arr = np.array(samples, dtype=np.uint64)[:, None]
            mask = np.uint64(1) << np.arange(n_q, dtype=np.uint64)
            bits = (arr & mask) > 0
            return 1.0 - 2.0 * bits.astype(float)

        while True:
            if not cmd_pipe.poll(timeout=0.1): continue
            cmd = cmd_pipe.recv()
            action = cmd.get("action")

            if action == "SHUTDOWN": break

            try:
                # --------------------------------------------------
                if action == "COMPUTE_SUPREMACY_BENCHMARK":
                    depth = cmd.get("depth", 10)
                    M_shots = cmd.get("shots", 10000)
                    try:
                        with gpu_semaphore:
                            sim_xeb = QrackSimulator(clone_sid=sim.sid)
                            try:
                                for _ in range(depth):
                                    for q in range(num_qubits):
                                        apply_rx(sim_xeb, rng.uniform(0, 2*np.pi), q)
                                        apply_rz(sim_xeb, rng.uniform(0, 2*np.pi), q)
                                    for q1, q2 in intra_edges:
                                        apply_cx(sim_xeb, q1, q2)
                                
                                all_q_list = list(range(num_qubits))
                                samples = sim_xeb.measure_shots(all_q_list, M_shots)
                                
                                state_vector = np.array(sim_xeb.amplitudes())
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

                        cmd_pipe.send({"status": "BENCHMARKS_COMPUTED", "patch_idx": patch_idx, "data": {
                            "HOG": float(hog_score),
                            "XEB": float(xeb_score)
                        }})

                    except Exception as e:
                        cmd_pipe.send({"status": "ERROR", "msg": f"XEB Failed: {str(e)}"})

                # --------------------------------------------------
                elif action == "EVOLVE_AND_MEASURE_STATISTICAL":
                    J   = cmd.get("J",   1.0)
                    hx  = cmd.get("hx",  0.5)
                    hz  = cmd.get("hz",  0.2)
                    dt  = cmd.get("dt",  0.05)
                    steps      = cmd.get("steps",      1)     # Minimised for speed
                    corr_shots = cmd.get("corr_shots", 512)   # Minimised for bandwidth

                    with gpu_semaphore:
                        for _ in range(steps):
                            for q in range(num_qubits): apply_rx(sim, -2.0 * hx * dt, q)
                            for q in range(num_qubits): apply_rz(sim, -2.0 * hz * dt, q)
                            for q1, q2 in intra_edges:
                                apply_cx(sim, q1, q2)
                                apply_rz(sim, -2.0 * J * dt, q2)
                                apply_cx(sim, q1, q2)
                            for q in range(num_qubits): apply_rz(sim, -2.0 * hz * dt, q)
                            for q in range(num_qubits): apply_rx(sim, -2.0 * hx * dt, q)

                        sim_z = QrackSimulator(clone_sid=sim.sid)
                        try:
                            z_samples = sim_z.measure_shots(all_boundary_qubits, corr_shots)
                        finally:
                            del sim_z

                        sim_x = QrackSimulator(clone_sid=sim.sid)
                        try:
                            for q in all_boundary_qubits: apply_h(sim_x, q)
                            x_samples = sim_x.measure_shots(all_boundary_qubits, corr_shots)
                        finally:
                            del sim_x

                        sim_y = QrackSimulator(clone_sid=sim.sid)
                        try:
                            for q in all_boundary_qubits: apply_rx(sim_y, -np.pi/2, q)
                            y_samples = sim_y.measure_shots(all_boundary_qubits, corr_shots)
                        finally:
                            del sim_y

                    gc.collect()

                    Z_mat  = extract_bits(z_samples, num_bound)
                    Z_mean = np.mean(Z_mat, axis=0)
                    Z_cov  = np.cov(Z_mat, rowvar=False) + np.eye(num_bound) * 1e-6

                    X_mat  = extract_bits(x_samples, num_bound)
                    X_mean = np.mean(X_mat, axis=0)
                    X_cov  = np.cov(X_mat, rowvar=False) + np.eye(num_bound) * 1e-6

                    Y_mat  = extract_bits(y_samples, num_bound)
                    Y_mean = np.mean(Y_mat, axis=0)
                    Y_cov  = np.cov(Y_mat, rowvar=False) + np.eye(num_bound) * 1e-6

                    payload = {
                        "qubits":     all_boundary_qubits,
                        "corr_shots": corr_shots,
                        "means": {"X": X_mean, "Y": Y_mean, "Z": Z_mean},
                        "covs":  {"X": X_cov,  "Y": Y_cov,  "Z": Z_cov}
                    }
                    cmd_pipe.send({"status": "STEP_1_COMPLETE",
                                   "patch_idx": patch_idx, "data": payload})

                # --------------------------------------------------
                elif action == "APPLY_KICKS_AND_MEASURE_ENERGY":
                    kicks  = cmd.get("kicks", {})
                    J_val  = cmd.get("J",   1.0)
                    hx_val = cmd.get("hx",  0.5)
                    hz_val = cmd.get("hz",  0.2)
                    local_energy = 0.0
                    
                    # Ensure energy measurements match the reduced statistical bandwidth
                    energy_shots = 512 

                    with gpu_semaphore:
                        for raw_q, (kx, ky, kz) in kicks.items():
                            q = int(raw_q)
                            # Consolidate 3-axis kick into a single exact SU(2) unitary
                            K = np.sqrt(kx**2 + ky**2 + kz**2)
                            if K > 0.0:
                                c = np.cos(K / 2.0)
                                s = np.sin(K / 2.0)
                                nx, ny, nz = kx/K, ky/K, kz/K
                                
                                m00 = complex(c, -nz * s)
                                m01 = complex(-ny * s, -nx * s)
                                m10 = complex(ny * s, -nx * s)
                                m11 = complex(c, nz * s)
                                
                                sim.mtrx([m00, m01, m10, m11], [q])

                        sim_z_zz = QrackSimulator(clone_sid=sim.sid)
                        try:
                            zz_samples = sim_z_zz.measure_shots(
                                list(range(num_qubits)), energy_shots)
                        finally:
                            del sim_z_zz

                        sim_x = QrackSimulator(clone_sid=sim.sid)
                        try:
                            for q in range(num_qubits): apply_h(sim_x, q)
                            x_samples = sim_x.measure_shots(
                                list(range(num_qubits)), energy_shots)
                        finally:
                            del sim_x

                    mask = np.uint64(1) << np.arange(num_qubits, dtype=np.uint64)

                    arr_z   = np.array(zz_samples, dtype=np.uint64)[:, None]
                    spins_z = 1.0 - 2.0 * ((arr_z & mask) > 0).astype(float)

                    for q in range(num_qubits):
                        local_energy += -hz_val * np.mean(spins_z[:, q])
                    for q1, q2 in intra_edges:
                        local_energy += -J_val * np.mean(spins_z[:, q1] * spins_z[:, q2])

                    arr_x   = np.array(x_samples, dtype=np.uint64)[:, None]
                    spins_x = 1.0 - 2.0 * ((arr_x & mask) > 0).astype(float)

                    for q in range(num_qubits):
                        local_energy += -hx_val * np.mean(spins_x[:, q])

                    cmd_pipe.send({"status": "STEP_2_COMPLETE",
                                   "patch_idx": patch_idx, "data": local_energy})

                # --------------------------------------------------
                elif action == "COMPUTE_BENCHMARKS":
                    try:
                        purity_sum = 0.0
                        with gpu_semaphore:
                            for q in range(num_qubits):
                                sim_q = QrackSimulator(clone_sid=sim.sid)
                                try:
                                    z_exp = 1.0 - 2.0 * sim_q.prob(q)
                                    apply_h(sim_q, q)
                                    x_exp = 1.0 - 2.0 * sim_q.prob(q)
                                    apply_h(sim_q, q)
                                    apply_rx(sim_q, -np.pi/2, q)
                                    y_exp = 1.0 - 2.0 * sim_q.prob(q)
                                    purity_sum += x_exp**2 + y_exp**2 + z_exp**2
                                finally:
                                    del sim_q
                        avg_purity = purity_sum / float(num_qubits)
                    except Exception as e:
                        print(f"Worker {patch_idx} benchmark failed: {e}")
                        avg_purity = 0.0

                    gc.collect()
                    cmd_pipe.send({"status": "BENCHMARKS_COMPUTED",
                                   "patch_idx": patch_idx,
                                   "data": {"grid_purity": float(avg_purity)}})

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
                        self.gpu_semaphores[assigned_gpu], int(patch_seeds[p_idx])
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
            
            print(f"         +-- Average HOG Score: {avg_hog:.4f} (Ideal noiseless: ~0.846)")
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
        target_hz: float
    ):
        print(f"\nStarting Adiabatic Anneal with Stochastic Injection...")
        self.energy_history.clear()
        noise_rng = np.random.default_rng()

        for t in range(total_steps):
            s = t / max(1, (total_steps - 1))
            current_hx     = (1.0 - s) * 3.0 + s * target_hx
            current_J      = s * target_J
            current_hz     = s * target_hz
            current_g_face = s * target_g_face

            step_payload = [
                {"J": current_J, "hx": current_hx, "hz": current_hz,
                 "dt": dt, "steps": 1, "corr_shots": 512}  # Optimised steps & shots
                for _ in range(self.num_patches)
            ]
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

            stochastic_noise: Dict[int, Dict[int, Tuple[float, float, float]]] = {}
            for p, prof in patch_profiles.items():
                n_bounds = len(prof["qubits"])
                scale = np.sqrt(dt / prof["corr_shots"])

                # CPU FPU Fix: Swap method='svd' to method='cholesky' for strict O(N^3) speedup
                X_noise = noise_rng.multivariate_normal(
                    np.zeros(n_bounds), prof["covs"]["X"], method='cholesky') * scale
                Y_noise = noise_rng.multivariate_normal(
                    np.zeros(n_bounds), prof["covs"]["Y"], method='cholesky') * scale
                Z_noise = noise_rng.multivariate_normal(
                    np.zeros(n_bounds), prof["covs"]["Z"], method='cholesky') * scale

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

            if t % 5 == 0 or t == total_steps - 1:
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

    target_grid = (2, 2, 4)  # 16 patches total

    # Symmetric Load Balancing for quad-socket NUMA architecture.
    # Eliminates the global barrier wait by equalising the worker count.
    explicit_gpu_allocation = [gpu_16gb] * 8 + [gpu_10gb] * 8

    # Generous semaphores to ensure GPUs are never starved.
    # VRAM footprint validated at ~1.1GB continuous per worker.
    semaphore_caps = {
        gpu_16gb: 7,
        gpu_10gb: 6,
    }

    engine = VolumetricHadronEngine27Q(
        gpu_allocation=explicit_gpu_allocation,
        semaphore_limits=semaphore_caps,
        grid=target_grid,
        master_seed=1337
    )

    try:
        engine.anneal_to_ground_state(
            total_steps=100,
            dt=0.04,  # Doubled to compensate for fewer inner steps
            target_g_face=0.15,
            target_J=1.0,
            target_hx=0.5,
            target_hz=0.2
        )
        
        # Diagnostic executed precisely ONCE on the post-annealed state
        engine.run_supremacy_benchmark(depth=10, shots=10000)
        
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    finally:
        engine.shutdown()
