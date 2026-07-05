# -*- coding: us-ascii -*-
# 28-Qubit 3x3x3 + Nucleus Lattice & Macroscopic Grid Annealing
# High-Throughput Volumetric Engine with Statistical Covariance Injection
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
        sim.mtrx([complex(np.cos(-theta/2), np.sin(-theta/2)), 0j, 
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
# 1. 28-QUBIT TOPOLOGY (3x3x3 + NUCLEUS)
# ==========================================
def generate_28q_nucleus_subvolume() -> Tuple[List[Tuple[int, int]], Dict[str, List[int]]]:
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
                
                if x == 0: boundaries["-X"].append(idx)
                if x == lx - 1: boundaries["+X"].append(idx)
                if y == 0: boundaries["-Y"].append(idx)
                if y == ly - 1: boundaries["+Y"].append(idx)
                if z == 0: boundaries["-Z"].append(idx)
                if z == lz - 1: boundaries["+Z"].append(idx)
                
    nucleus_idx = 27
    for grid_idx in range(27):
        edges.append((nucleus_idx, grid_idx))
                
    return edges, boundaries

# ==========================================
# 2. ISOLATED LHV ISLAND (GPU WORKER)
# ==========================================
def persistent_island_worker_28q(
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
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
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
        
        all_boundary_qubits = sorted(list(set([q for face in boundaries.values() for q in face])))
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
                if action == "EVOLVE_AND_MEASURE_STATISTICAL":
                    J, hx, hz = cmd.get("J", 1.0), cmd.get("hx", 0.5), cmd.get("hz", 0.2)
                    dt, steps = cmd.get("dt", 0.05), cmd.get("steps", 2)
                    corr_shots = cmd.get("corr_shots", 1024)
                    
                    with gpu_semaphore:
                        for _ in range(steps):
                            for q in range(num_qubits): apply_rx(sim, -hx * dt, q)
                            for q in range(num_qubits): apply_rz(sim, -hz * dt, q)
                            for q1, q2 in intra_edges:
                                apply_cx(sim, q1, q2)
                                apply_rz(sim, -2.0 * J * dt, q2)
                                apply_cx(sim, q1, q2)
                            for q in range(num_qubits): apply_rz(sim, -hz * dt, q)
                            for q in range(num_qubits): apply_rx(sim, -hx * dt, q)
                            
                    with gpu_semaphore:
                        sim_z = QrackSimulator(clone_sid=sim.sid)
                        try:
                            z_samples = sim_z.measure_shots(all_boundary_qubits, corr_shots)
                        finally:
                            del sim_z
                    gc.collect()
                    Z_mat = extract_bits(z_samples, num_bound)
                    Z_mean = np.mean(Z_mat, axis=0)
                    Z_cov = np.cov(Z_mat, rowvar=False) + np.eye(num_bound) * 1e-6
                    
                    with gpu_semaphore:
                        sim_x = QrackSimulator(clone_sid=sim.sid)
                        try:
                            for q in all_boundary_qubits: apply_h(sim_x, q)
                            x_samples = sim_x.measure_shots(all_boundary_qubits, corr_shots)
                        finally:
                            del sim_x
                    gc.collect()
                    X_mat = extract_bits(x_samples, num_bound)
                    X_mean = np.mean(X_mat, axis=0)
                    X_cov = np.cov(X_mat, rowvar=False) + np.eye(num_bound) * 1e-6

                    with gpu_semaphore:
                        sim_y = QrackSimulator(clone_sid=sim.sid)
                        try:
                            for q in all_boundary_qubits: apply_rx(sim_y, -np.pi/2, q)
                            y_samples = sim_y.measure_shots(all_boundary_qubits, corr_shots)
                        finally:
                            del sim_y
                    gc.collect()
                    Y_mat = extract_bits(y_samples, num_bound)
                    Y_mean = np.mean(Y_mat, axis=0)
                    Y_cov = np.cov(Y_mat, rowvar=False) + np.eye(num_bound) * 1e-6
                            
                    payload = {
                        "qubits": all_boundary_qubits,
                        "corr_shots": corr_shots,
                        "means": {"X": X_mean, "Y": Y_mean, "Z": Z_mean},
                        "covs": {"X": X_cov, "Y": Y_cov, "Z": Z_cov}
                    }
                    cmd_pipe.send({"status": "STEP_1_COMPLETE", "patch_idx": patch_idx, "data": payload})

                elif action == "APPLY_KICKS_AND_MEASURE_ENERGY":
                    kicks = cmd.get("kicks", {})
                    J_val, hx_val, hz_val = cmd.get("J", 1.0), cmd.get("hx", 0.5), cmd.get("hz", 0.2)
                    local_energy = 0.0
                    
                    with gpu_semaphore:
                        # 1. Apply macroscopic kicks to the primary persistent state
                        for raw_q, (kx, ky, kz) in kicks.items():
                            q = int(raw_q)
                            if kx != 0.0: apply_rx(sim, kx, q)
                            if ky != 0.0: apply_ry(sim, ky, q)
                            if kz != 0.0: apply_rz(sim, kz, q)
                            
                        # 2. Instantiate ephemeral clone strictly for destructive energy measurement
                        sim_e = QrackSimulator(clone_sid=sim.sid)
                        try:
                            # Z terms - Native Z Basis
                            for q in range(num_qubits):
                                local_energy += -hz_val * (1.0 - 2.0 * sim_e.prob(q))
                                
                            # ZZ terms - Native Z Basis (CX pairs restore entanglement cleanly)
                            for q1, q2 in intra_edges:
                                apply_cx(sim_e, q1, q2)
                                local_energy += -J_val * (1.0 - 2.0 * sim_e.prob(q2))
                                apply_cx(sim_e, q1, q2)
                                
                            # X terms - Destructively rotate to X Basis last
                            for q in range(num_qubits):
                                apply_h(sim_e, q)
                                local_energy += -hx_val * (1.0 - 2.0 * sim_e.prob(q))
                                
                        finally:
                            del sim_e
                            
                    cmd_pipe.send({"status": "STEP_2_COMPLETE", "patch_idx": patch_idx, "data": local_energy})

                elif action == "COMPUTE_BENCHMARKS":
                    try:
                        purity_sum = 0.0
                        with gpu_semaphore:
                            for q in range(num_qubits):
                                z_exp = 1.0 - 2.0 * sim.prob(q)
                                apply_h(sim, q)
                                x_exp = 1.0 - 2.0 * sim.prob(q)
                                apply_h(sim, q)
                                apply_rx(sim, -np.pi/2, q)
                                y_exp = 1.0 - 2.0 * sim.prob(q)
                                apply_rx(sim, np.pi/2, q) # Y-basis restoration guaranteed
                                purity_sum += (x_exp**2 + y_exp**2 + z_exp**2)
                                
                        avg_purity = purity_sum / float(num_qubits)
                    except Exception as e:
                        print(f"Worker {patch_idx} benchmark failed: {e}")
                        avg_purity = 0.0
                    
                    gc.collect() 
                    cmd_pipe.send({"status": "BENCHMARKS_COMPUTED", "patch_idx": patch_idx, "data": {
                        "grid_purity": float(avg_purity)
                    }})

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
            cmd_pipe.close() # Plugs the worker process resource leak
        except Exception:
            pass

# ==========================================
# 3. 28-QUBIT MACROSCOPIC ORCHESTRATOR
# ==========================================
class VolumetricHadronEngine28Q:
    def __init__(self, gpu_allocation: List[int], semaphore_limits: Dict[int, int], grid: Tuple[int, int, int] = (2, 2, 2), master_seed: int = 42):
        self._is_shutdown = False
        self.gpu_allocation = gpu_allocation
        
        self.grid_x, self.grid_y, self.grid_z = grid
        self.num_patches = self.grid_x * self.grid_y * self.grid_z
        
        if len(self.gpu_allocation) != self.num_patches:
            raise ValueError(f"GPU allocation list ({len(self.gpu_allocation)}) must match total grid patches ({self.num_patches}).")
            
        self.qubits_per_patch = 28
        self.intra_edges, self.boundaries = generate_28q_nucleus_subvolume()
        
        self.patch_coords = {}
        idx = 0
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                for z in range(self.grid_z):
                    self.patch_coords[idx] = (x, y, z)
                    idx += 1

        self.coord_to_patch = {v: k for k, v in self.patch_coords.items()}

        self.ctx = mp.get_context('spawn')
        self.gpu_semaphores = {gpu_id: self.ctx.Semaphore(limit) for gpu_id, limit in semaphore_limits.items()}
        
        self.workers: List[mp.Process] = []
        self.pipes: List[Connection] = []
        self.energy_history = []
        
        self.csv_filename = "ground_state_energy_curve.csv"
        self._init_csv()

        total_sites = self.num_patches * self.qubits_per_patch
        print(f"Initializing Asymmetric High-Throughput 28-Qubit Engine...")
        print(f"Total logical capacity configured: {total_sites} total qubits")
        
        master_rng = np.random.default_rng(master_seed)
        patch_seeds = master_rng.integers(0, 2**31 - 1, size=self.num_patches)
        
        try:
            for p_idx in range(self.num_patches):
                parent_conn, child_conn = self.ctx.Pipe()
                assigned_gpu = self.gpu_allocation[p_idx]
                
                p = self.ctx.Process(
                    target=persistent_island_worker_28q,
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
                if pipe.poll(timeout=45.0):
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

    def sync_broadcast(self, action: str, kwargs_list: Optional[List[Dict]] = None, expected_status: Optional[str] = None) -> Dict[int, Any]:
        if kwargs_list is None: kwargs_list = [{} for _ in range(self.num_patches)]
        
        send_errors = {}
        error_lock = threading.Lock()
        
        def _send_one(pipe: Connection, msg: Dict[str, Any], idx: int):
            try:
                pipe.send(msg)
            except Exception as e:
                with error_lock:
                    send_errors[idx] = e

        threads = []
        for i, pipe in enumerate(self.pipes): 
            t = threading.Thread(target=_send_one, args=(pipe, {"action": action, **kwargs_list[i]}, i))
            t.start()
            threads.append(t)
            
        for t in threads: t.join()
            
        if send_errors:
            self.shutdown()
            raise RuntimeError(f"Send failed for workers: {send_errors}")
            
        results = {}
        pending = list(enumerate(self.pipes)) 
        deadline = time.monotonic() + 300.0
        
        while pending:
            timeout = deadline - time.monotonic()
            if timeout <= 0: raise TimeoutError(f"Workers timed out on action '{action}'")
                
            # Restrict wait timeout to 60s for periodic progress logging
            ready_pipes = wait([p for _, p in pending], timeout=min(timeout, 60.0))
            if not ready_pipes: 
                elapsed = 300.0 - (deadline - time.monotonic())
                print(f"[WARN] Still waiting on {len(pending)} workers ({elapsed:.0f}s elapsed) for action '{action}'...")
                continue
                
            still_pending = []
            for idx, pipe in pending:
                if pipe in ready_pipes:
                    try:
                        res = pipe.recv()
                        if res.get("status") == "ERROR": 
                            raise RuntimeError(f"Worker {idx} error: {res.get('msg')}")
                        if expected_status and res.get("status") != expected_status:
                            raise RuntimeError(f"Worker {idx} protocol sync error: Expected {expected_status}, got {res.get('status')}")
                        results[idx] = res
                    except (EOFError, OSError, BrokenPipeError): 
                        raise RuntimeError(f"Worker {idx} connection crashed.")
                else: 
                    still_pending.append((idx, pipe))
            pending = still_pending
        return results

    def anneal_to_ground_state(self, total_steps: int, dt: float, target_g_face: float, target_J: float, target_hx: float, target_hz: float):
        print(f"Starting Adiabatic Anneal with Stochastic Injection...")
        self.energy_history.clear()
        
        noise_rng = np.random.default_rng()
        
        for t in range(total_steps):
            s = t / max(1, (total_steps - 1))
            current_hx = (1.0 - s) * 3.0 + s * target_hx
            current_J = s * target_J
            current_hz = s * target_hz
            current_g_face = s * target_g_face
            
            step_payload = [{"J": current_J, "hx": current_hx, "hz": current_hz, "dt": dt, "steps": 2, "corr_shots": 1024} for _ in range(self.num_patches)]
            step1_res = self.sync_broadcast("EVOLVE_AND_MEASURE_STATISTICAL", step_payload, expected_status="STEP_1_COMPLETE")
            
            patch_profiles = {p: res["data"] for p, res in step1_res.items()}
            kick_payloads = [{"kicks": {}, "J": current_J, "hx": current_hx, "hz": current_hz} for _ in range(self.num_patches)]
            macroscopic_boundary_energy = 0.0
            
            stochastic_noise = {}
            for p, prof in patch_profiles.items():
                n_bounds = len(prof["qubits"])
                scale = np.sqrt(dt / prof["corr_shots"])
                
                X_noise = noise_rng.multivariate_normal(np.zeros(n_bounds), prof["covs"]["X"], method='svd') * scale
                Y_noise = noise_rng.multivariate_normal(np.zeros(n_bounds), prof["covs"]["Y"], method='svd') * scale
                Z_noise = noise_rng.multivariate_normal(np.zeros(n_bounds), prof["covs"]["Z"], method='svd') * scale
                
                stochastic_noise[p] = {
                    q: (X_noise[i], Y_noise[i], Z_noise[i]) 
                    for i, q in enumerate(prof["qubits"])
                }

            for p1, coord1 in self.patch_coords.items():
                x1, y1, z1 = coord1
                neighbors = {
                    "+X": (x1+1, y1, z1), "-X": (x1-1, y1, z1),
                    "+Y": (x1, y1+1, z1), "-Y": (x1, y1-1, z1),
                    "+Z": (x1, y1, z1+1), "-Z": (x1, y1, z1-1)
                }
                
                for dir1, coord2 in neighbors.items():
                    if not (0 <= coord2[0] < self.grid_x and 0 <= coord2[1] < self.grid_y and 0 <= coord2[2] < self.grid_z):
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
                    
                    avg_x2, avg_y2, avg_z2 = 0.0, 0.0, 0.0
                    for q2 in face2_qubits:
                        idx2 = q_to_idx2[q2]
                        avg_x2 += prof2["means"]["X"][idx2] + noise2[q2][0]
                        avg_y2 += prof2["means"]["Y"][idx2] + noise2[q2][1]
                        avg_z2 += prof2["means"]["Z"][idx2] + noise2[q2][2]
                        
                    n2 = max(1, len(face2_qubits))
                    avg_x2 /= n2; avg_y2 /= n2; avg_z2 /= n2
                    
                    avg_x1, avg_y1, avg_z1 = 0.0, 0.0, 0.0
                    for q1 in face1_qubits:
                        idx1 = q_to_idx1[q1]
                        avg_x1 += prof1["means"]["X"][idx1] + noise1[q1][0]
                        avg_y1 += prof1["means"]["Y"][idx1] + noise1[q1][1]
                        avg_z1 += prof1["means"]["Z"][idx1] + noise1[q1][2]
                        
                    n1 = max(1, len(face1_qubits))
                    avg_x1 /= n1; avg_y1 /= n1; avg_z1 /= n1
                    
                    # Physically robust mean-field bond energy calculation (Symmetric)
                    interaction_E = -current_g_face * (avg_x1*avg_x2 + avg_y1*avg_y2 + avg_z1*avg_z2) * ((len(face1_qubits) + len(face2_qubits)) / 2.0)
                    macroscopic_boundary_energy += interaction_E
                    
                    for q1 in face1_qubits:
                        curr_k1 = kick_payloads[p1]["kicks"].get(q1, (0.0, 0.0, 0.0))
                        kick_payloads[p1]["kicks"][q1] = (
                            curr_k1[0] + current_g_face * avg_x2,
                            curr_k1[1] + current_g_face * avg_y2,
                            curr_k1[2] + current_g_face * avg_z2
                        )
                        
                    for q2 in face2_qubits:
                        curr_k2 = kick_payloads[p2]["kicks"].get(q2, (0.0, 0.0, 0.0))
                        kick_payloads[p2]["kicks"][q2] = (
                            curr_k2[0] + current_g_face * avg_x1,
                            curr_k2[1] + current_g_face * avg_y1,
                            curr_k2[2] + current_g_face * avg_z1
                        )

            step2_res = self.sync_broadcast("APPLY_KICKS_AND_MEASURE_ENERGY", kick_payloads, expected_status="STEP_2_COMPLETE")
            bulk_energy = sum([r["data"] for r in step2_res.values()])
            
            total_energy = bulk_energy + macroscopic_boundary_energy
            print(f"Step {t:03d} | Anneal: {s*100:05.1f}% | Total Setup Potential Energy: {total_energy:+.4f}")
            
            step_data = {"Step": t, "Anneal_Percent": s*100, "Energy": total_energy}
            self.energy_history.append(step_data)
            
            if t % 5 == 0 or t == total_steps - 1:
                self._append_to_csv(step_data)
            
            if t == total_steps - 1:
                print(f"         +-- Calculating Final Benchmarks...")
                bench_res = self.sync_broadcast("COMPUTE_BENCHMARKS", expected_status="BENCHMARKS_COMPUTED")
                avg_purity = sum(res["data"]["grid_purity"] for res in bench_res.values()) / self.num_patches
                print(f"         +-- Avg Total Sub-Volume Purity (1.0 = Pure, 0.0 = Mixed): {avg_purity:.4f}")

    def shutdown(self) -> None:
        if self._is_shutdown: return
        self._is_shutdown = True
        print("\nShutting down Asymmetric 28-Qubit Volumetric Engine...")
        
        # Hardened double-drain pipe sequence to prevent race conditions during shutdown
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
                if p.is_alive(): 
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
    
    gpu_env = os.environ.get("WORMHOLE_GPUS", "0,1") 
    base_gpus = [int(g.strip()) for g in gpu_env.split(',')]
    
    if len(base_gpus) < 2:
        print("WARNING: Asymmetric execution requires at least 2 GPUs. Falling back to default [0, 1].")
        base_gpus = [0, 1]
        
    gpu_16gb = base_gpus[0]
    gpu_10gb = base_gpus[1]
    
    target_grid = (2, 2, 2)
    
    explicit_gpu_allocation = [gpu_16gb] * 5 + [gpu_10gb] * 3
    
    semaphore_caps = {
        gpu_16gb: 1, 
        gpu_10gb: 1   
    }
    
    engine = VolumetricHadronEngine28Q(
        gpu_allocation=explicit_gpu_allocation,
        semaphore_limits=semaphore_caps,
        grid=target_grid,
        master_seed=1337
    )

    try:
        engine.anneal_to_ground_state(
            total_steps=100, 
            dt=0.02, 
            target_g_face=0.15,
            target_J=1.0, 
            target_hx=0.5, 
            target_hz=0.2
        )
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    finally:
        engine.shutdown()
