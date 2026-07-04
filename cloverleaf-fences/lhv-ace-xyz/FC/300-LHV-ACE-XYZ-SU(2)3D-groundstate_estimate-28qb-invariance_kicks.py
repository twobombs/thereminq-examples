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
        
        # Sort to guarantee stable covariance matrix indexing
        all_boundary_qubits = sorted(list(set([q for face in boundaries.values() for q in face])))
        num_bound = len(all_boundary_qubits)

        def extract_bits(samples: List[int], n_q: int) -> np.ndarray:
            """Convert integer shots to an array of +1/-1 eigenvalues."""
            arr = np.array(samples, dtype=np.uint32)[:, None]
            mask = 1 << np.arange(n_q, dtype=np.uint32)
            bits = (arr & mask) > 0
            return 1.0 - 2.0 * bits 

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
                        # 1. Strang Splitting (Heavy Entanglement Phase)
                        for _ in range(steps):
                            for q in range(num_qubits): apply_rx(sim, -hx * dt, q)
                            for q in range(num_qubits): apply_rz(sim, -hz * dt, q)
                            for q1, q2 in intra_edges:
                                apply_cx(sim, q1, q2)
                                apply_rz(sim, -2.0 * J * dt, q2)
                                apply_cx(sim, q1, q2)
                            for q in range(num_qubits): apply_rz(sim, -hz * dt, q)
                            for q in range(num_qubits): apply_rx(sim, -hx * dt, q)
                            
                        # 2. Statistical Correlation Profiling (Z-Basis)
                        z_samples = sim.measure_shots(all_boundary_qubits, corr_shots)
                        Z_mat = extract_bits(z_samples, num_bound)
                        Z_mean = np.mean(Z_mat, axis=0)
                        Z_cov = np.cov(Z_mat, rowvar=False) + np.eye(num_bound) * 1e-6
                        
                        # X-Basis Profile
                        for q in all_boundary_qubits: apply_h(sim, q)
                        x_samples = sim.measure_shots(all_boundary_qubits, corr_shots)
                        X_mat = extract_bits(x_samples, num_bound)
                        X_mean = np.mean(X_mat, axis=0)
                        X_cov = np.cov(X_mat, rowvar=False) + np.eye(num_bound) * 1e-6
                        for q in all_boundary_qubits: apply_h(sim, q) # Reverse

                        # Y-Basis Profile
                        for q in all_boundary_qubits: apply_rx(sim, np.pi/2, q)
                        y_samples = sim.measure_shots(all_boundary_qubits, corr_shots)
                        Y_mat = extract_bits(y_samples, num_bound)
                        Y_mean = np.mean(Y_mat, axis=0)
                        Y_cov = np.cov(Y_mat, rowvar=False) + np.eye(num_bound) * 1e-6
                        for q in all_boundary_qubits: apply_rx(sim, -np.pi/2, q) # Reverse
                            
                    payload = {
                        "qubits": all_boundary_qubits,
                        "means": {"X": X_mean, "Y": Y_mean, "Z": Z_mean},
                        "covs": {"X": X_cov, "Y": Y_cov, "Z": Z_cov}
                    }
                    cmd_pipe.send({"status": "STEP_1_COMPLETE", "patch_idx": patch_idx, "data": payload})

                elif action == "APPLY_KICKS_AND_MEASURE_ENERGY":
                    kicks = cmd.get("kicks", {})
                    J_val, hx_val, hz_val = cmd.get("J", 1.0), cmd.get("hx", 0.5), cmd.get("hz", 0.2)
                    local_energy = 0.0
                    
                    with gpu_semaphore:
                        # 1. Apply Stochastic Macroscopic Kicks
                        for raw_q, (kx, ky, kz) in kicks.items():
                            q = int(raw_q)
                            if kx != 0.0: apply_rx(sim, kx, q)
                            if ky != 0.0: apply_ry(sim, ky, q)
                            if kz != 0.0: apply_rz(sim, kz, q)
                            
                        # 2. Measure Bulk Energy (in-place)
                        for q in range(num_qubits):
                            local_energy += -hz_val * (1.0 - 2.0 * sim.prob(q))
                            apply_h(sim, q)
                            local_energy += -hx_val * (1.0 - 2.0 * sim.prob(q))
                            apply_h(sim, q)
                            
                        for q1, q2 in intra_edges:
                            apply_cx(sim, q1, q2)
                            local_energy += -J_val * (1.0 - 2.0 * sim.prob(q2))
                            apply_cx(sim, q1, q2)
                            
                    cmd_pipe.send({"status": "STEP_2_COMPLETE", "patch_idx": patch_idx, "data": local_energy})

                # COMPUTE_BENCHMARKS remains functionally identical to the previous version
                elif action == "COMPUTE_BENCHMARKS":
                    shots = cmd.get("shots", 2048) 
                    avg_purity, expected_xeb, expected_hog = 0.0, 0.0, 0.0
                    
                    try:
                        purity_sum = 0.0
                        with gpu_semaphore:
                            for q in range(27):
                                z_exp = 1.0 - 2.0 * sim.prob(q)
                                apply_h(sim, q)
                                x_exp = 1.0 - 2.0 * sim.prob(q)
                                apply_h(sim, q)
                                apply_rx(sim, np.pi/2, q)
                                y_exp = 1.0 - 2.0 * sim.prob(q)
                                apply_rx(sim, -np.pi/2, q)
                                purity_sum += (x_exp**2 + y_exp**2 + z_exp**2)
                                
                            ideal_probs_list = sim.out_probs()
                            raw_samples = sim.measure_shots(list(range(num_qubits)), shots)
                            
                        avg_purity = purity_sum / 27.0
                        
                        if raw_samples and isinstance(raw_samples[0], (list, tuple)):
                            samples_arr = np.array(raw_samples, dtype=np.uint32)
                            powers = 1 << np.arange(num_qubits, dtype=np.uint32)
                            samples = samples_arr.dot(powers)
                        else:
                            samples = np.array(raw_samples, dtype=np.uint32)
                        
                        ideal_probs = np.array(ideal_probs_list, dtype=np.float64)
                        n_pow = len(ideal_probs)
                        u_u = 1.0 / n_pow
                        
                        counts_arr = np.bincount(samples, minlength=n_pow)
                        denom = np.sum((ideal_probs - u_u) ** 2)
                        numer = np.sum((ideal_probs - u_u) * ((counts_arr / shots) - u_u))
                        expected_xeb = float(numer / denom) if denom > 0 else 0.0
                        
                        threshold = np.median(ideal_probs)
                        is_heavy = ideal_probs > threshold
                        sum_hog_counts = np.sum(counts_arr[is_heavy])
                        expected_hog = float(sum_hog_counts / shots)
                        
                    except Exception as e:
                        print(f"Worker {patch_idx} benchmark failed: {e}")
                    
                    gc.collect() 
                    cmd_pipe.send({"status": "BENCHMARKS_COMPUTED", "patch_idx": patch_idx, "data": {
                        "grid_purity": float(avg_purity), "xeb": expected_xeb, "hog": expected_hog
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

# ==========================================
# 3. 28-QUBIT MACROSCOPIC ORCHESTRATOR
# ==========================================
class VolumetricHadronEngine28Q:
    def __init__(self, device_ids: List[int], grid: Tuple[int, int, int] = (2, 2, 4), master_seed: int = 42):
        self._is_shutdown = False
        self.device_ids = device_ids
        
        self.grid_x, self.grid_y, self.grid_z = grid
        self.num_patches = self.grid_x * self.grid_y * self.grid_z
        
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
        self.gpu_semaphores = {gpu_id: self.ctx.Semaphore(2) for gpu_id in set(self.device_ids)}
        
        self.workers: List[mp.Process] = []
        self.pipes: List[Connection] = []
        self.energy_history = []
        
        self.csv_filename = "ground_state_energy_curve.csv"
        self._init_csv()

        total_sites = self.num_patches * self.qubits_per_patch
        print(f"Initializing High-Throughput 28-Qubit Engine with Statistical Covariance Profiles...")
        
        master_rng = np.random.default_rng(master_seed)
        patch_seeds = master_rng.integers(0, 2**31 - 1, size=self.num_patches)
        
        for p_idx in range(self.num_patches):
            parent_conn, child_conn = self.ctx.Pipe()
            assigned_gpu = self.device_ids[p_idx % len(self.device_ids)]
            
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
                    self.shutdown(); raise RuntimeError("Worker error.")
            else: 
                self.shutdown(); raise TimeoutError("Worker timeout.")

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

    def sync_broadcast(self, action: str, kwargs_list: Optional[List[Dict]] = None) -> Dict[int, Any]:
        if kwargs_list is None: kwargs_list = [{}] * self.num_patches
        for i, pipe in enumerate(self.pipes): 
            pipe.send({"action": action, **kwargs_list[i]})
            
        results = {}
        pending = list(enumerate(self.pipes)) 
        deadline = time.monotonic() + 300.0
        
        while pending:
            timeout = deadline - time.monotonic()
            if timeout <= 0: raise TimeoutError(f"Workers timed out on action '{action}'")
                
            ready_pipes = wait([p for _, p in pending], timeout=timeout)
            if not ready_pipes: continue
                
            still_pending = []
            for idx, pipe in pending:
                if pipe in ready_pipes:
                    try:
                        res = pipe.recv()
                        if res.get("status") == "ERROR": 
                            raise RuntimeError(f"Worker {idx} error: {res.get('msg')}")
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
        
        for t in range(total_steps):
            s = t / max(1, (total_steps - 1))
            current_hx = (1.0 - s) * 3.0 + s * target_hx
            current_J = s * target_J
            current_hz = s * target_hz
            current_g_face = s * target_g_face
            
            # Step 1: Statistical Sampling
            step1_res = self.sync_broadcast("EVOLVE_AND_MEASURE_STATISTICAL", [
                {"J": current_J, "hx": current_hx, "hz": current_hz, "dt": dt, "steps": 2, "corr_shots": 1024}
            ] * self.num_patches)
            
            patch_profiles = {p: res["data"] for p, res in step1_res.items()}
            kick_payloads = [{"kicks": {}, "J": current_J, "hx": current_hx, "hz": current_hz} for _ in range(self.num_patches)]
            macroscopic_boundary_energy = 0.0
            
            # Generate multivariate stochastic fluctuations based on measured exact covariances
            stochastic_noise = {}
            for p, prof in patch_profiles.items():
                n_bounds = len(prof["qubits"])
                # Langevin dynamics: Sample correlated noise from the covariance matrices
                X_noise = np.random.multivariate_normal(np.zeros(n_bounds), prof["covs"]["X"])
                Y_noise = np.random.multivariate_normal(np.zeros(n_bounds), prof["covs"]["Y"])
                Z_noise = np.random.multivariate_normal(np.zeros(n_bounds), prof["covs"]["Z"])
                
                stochastic_noise[p] = {
                    q: (X_noise[i], Y_noise[i], Z_noise[i]) 
                    for i, q in enumerate(prof["qubits"])
                }

            # Map the kicks topographically
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
                    if p2 is None: continue 
                    
                    dir2 = dir1.replace("+", "temp").replace("-", "+").replace("temp", "-")
                    face1_qubits = self.boundaries[dir1]
                    face2_qubits = self.boundaries[dir2]
                    
                    prof2 = patch_profiles[p2]
                    noise2 = stochastic_noise[p2]
                    
                    # Compute mean field + statistical dispersion per face
                    avg_x, avg_y, avg_z = 0.0, 0.0, 0.0
                    for q2 in face2_qubits:
                        idx2 = prof2["qubits"].index(q2)
                        # Mean Field
                        avg_x += prof2["means"]["X"][idx2]
                        avg_y += prof2["means"]["Y"][idx2]
                        avg_z += prof2["means"]["Z"][idx2]
                        
                        # Add Correlated Fluctuation Component
                        avg_x += noise2[q2][0]
                        avg_y += noise2[q2][1]
                        avg_z += noise2[q2][2]
                        
                    n2 = max(1, len(face2_qubits))
                    avg_x /= n2; avg_y /= n2; avg_z /= n2
                    
                    interaction_E = 0.0
                    prof1 = patch_profiles[p1]
                    for q1 in face1_qubits:
                        curr_k = kick_payloads[p1]["kicks"].get(q1, (0.0, 0.0, 0.0))
                        kick_payloads[p1]["kicks"][q1] = (
                            curr_k[0] + current_g_face * avg_x,
                            curr_k[1] + current_g_face * avg_y,
                            curr_k[2] + current_g_face * avg_z
                        )
                        idx1 = prof1["qubits"].index(q1)
                        interaction_E += -current_g_face * (prof1["means"]["X"][idx1]*avg_x + 
                                                            prof1["means"]["Y"][idx1]*avg_y + 
                                                            prof1["means"]["Z"][idx1]*avg_z)
                        
                    if p1 < p2: macroscopic_boundary_energy += interaction_E

            step2_res = self.sync_broadcast("APPLY_KICKS_AND_MEASURE_ENERGY", kick_payloads)
            bulk_energy = sum([r["data"] for r in step2_res.values()])
            
            total_energy = bulk_energy + macroscopic_boundary_energy
            print(f"Step {t:03d} | Anneal: {s*100:05.1f}% | Total Setup Potential Energy: {total_energy:+.4f}")
            
            step_data = {"Step": t, "Anneal_Percent": s*100, "Energy": total_energy}
            self.energy_history.append(step_data)
            
            if t % 5 == 0 or t == total_steps - 1:
                self._append_to_csv(step_data)
            
            if t == total_steps - 1:
                print(f"         +-- Calculating Final Benchmarks...")
                bench_res = {}
                for p_idx, pipe in enumerate(self.pipes):
                    pipe.send({"action": "COMPUTE_BENCHMARKS", "shots": 2048})
                    if not wait([pipe], timeout=300.0):
                        self.shutdown(); raise TimeoutError(f"Worker {p_idx} benchmark timeout.")
                    try:
                        res = pipe.recv()
                        if res.get("status") == "ERROR": self.shutdown(); raise RuntimeError(res.get('msg'))
                        bench_res[p_idx] = res
                    except (EOFError, OSError, BrokenPipeError):
                        self.shutdown(); raise RuntimeError(f"Worker {p_idx} crashed.")

                avg_purity = sum(res["data"]["grid_purity"] for res in bench_res.values()) / self.num_patches
                avg_xeb = sum(res["data"]["xeb"] for res in bench_res.values()) / self.num_patches
                avg_hog = sum(res["data"]["hog"] for res in bench_res.values()) / self.num_patches
                
                print(f"         +-- Avg XEB: {avg_xeb:.4f} | Avg HOG: {avg_hog:.4f}")
                print(f"         +-- Avg Grid Purity (1.0 = Pure, 0.0 = Mixed): {avg_purity:.4f}")

    def shutdown(self) -> None:
        if self._is_shutdown: return
        self._is_shutdown = True
        print("\nShutting down 28-Qubit Volumetric Engine...")
        for pipe in getattr(self, 'pipes', []):
            try:
                if not pipe.closed: 
                    while pipe.poll(): pipe.recv()
                    pipe.send({"action": "SHUTDOWN"})
            except (OSError, BrokenPipeError, EOFError): pass
        for p in getattr(self, 'workers', []):
            try:
                p.join(timeout=2)
                if p.is_alive(): p.terminate()
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
    
    gpu_env = os.environ.get("WORMHOLE_GPUS", "0,1,2,3") 
    base_gpus = [int(g.strip()) for g in gpu_env.split(',')]
    
    target_grid = (2, 2, 4)
    total_requested_nodes = target_grid[0] * target_grid[1] * target_grid[2]
    AVAILABLE_GPUS = [base_gpus[i % len(base_gpus)] for i in range(total_requested_nodes)]
    
    engine = VolumetricHadronEngine28Q(
        device_ids=AVAILABLE_GPUS, 
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
