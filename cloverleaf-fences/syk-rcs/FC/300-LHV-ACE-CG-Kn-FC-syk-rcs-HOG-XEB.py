# -*- coding: us-ascii -*-
import os
import gc
import time
import signal
import collections
import numpy as np
import multiprocessing as mp
from multiprocessing.connection import Connection, wait
from typing import List, Tuple, Dict, Any

# ==========================================
# 0. PYQRACK API SAFEGUARDS & GATES
# ==========================================
PX, PY, PZ = 1, 2, 3

def apply_h(sim: Any, q: int) -> None:
    if hasattr(sim, 'h'): 
        sim.h(q)
    else: 
        sim.mtrx([
            complex(1/np.sqrt(2), 0), complex(1/np.sqrt(2), 0), 
            complex(1/np.sqrt(2), 0), complex(-1/np.sqrt(2), 0)
        ], [q])

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
    else: 
        sim.mcx([c], t)

# ==========================================
# 1. HOLOGRAPHIC TOPOLOGY (BULK/BOUNDARY)
# ==========================================
def get_complete_intra_edges(num_qubits: int = 25) -> List[Tuple[int, int]]:
    edges = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            edges.append((i, j))
    return edges

def get_holographic_topology(
    num_patches: int = 12, 
    qubits_per_patch: int = 25, 
    boundary_size: int = 4
) -> Tuple[List[List[int]], List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    
    total_qubits = num_patches * qubits_per_patch
    patches = [[] for _ in range(num_patches)]
    fence_edges = []
    
    for idx in range(total_qubits):
        patch_idx = idx // qubits_per_patch
        patches[patch_idx].append(idx)

    for p1 in range(num_patches):
        for p2 in range(p1 + 1, num_patches):
            for b1 in range(boundary_size):
                for b2 in range(boundary_size):
                    fence_edges.append(((p1, b1), (p2, b2)))

    return patches, fence_edges

# ==========================================
# 2. ISOLATED LHV ISLAND (GPU WORKER)
# ==========================================
def persistent_island_worker(
    device_id: int, 
    patch_idx: int, 
    num_qubits: int, 
    intra_edges: List[Tuple[int, int]], 
    boundary_qubits: List[int],
    cmd_pipe: Connection
) -> None:
    
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    
    sim = None
    try:
        os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(device_id)
        os.environ["QRACK_QPAGER_DEVICES"] = str(device_id)

        from pyqrack import QrackSimulator
        
        sim = QrackSimulator(qubit_count=num_qubits)
        
        for q in range(num_qubits):
            apply_h(sim, q)
            
        cmd_pipe.send({"status": "READY", "patch_idx": patch_idx})
        rotation_gates = [apply_rx, apply_ry, apply_rz]

        while True:
            try:
                if cmd_pipe.poll(timeout=0.1):
                    cmd = cmd_pipe.recv()
                else:
                    continue
            except (EOFError, OSError, BrokenPipeError):
                break
                
            action = cmd.get("action")

            if action == "SHUTDOWN":
                break

            try:
                if action == "RCS_CHUNK":
                    seed = cmd.get("seed")
                    depth = cmd.get("depth", 1)
                    rng = np.random.default_rng(seed)

                    for _ in range(depth):
                        for q in range(num_qubits):
                            gate_idx = rng.integers(0, 3)
                            theta = rng.uniform(-np.pi, np.pi)
                            rotation_gates[gate_idx](sim, theta, q)
                        
                        for q1, q2 in intra_edges:
                            apply_cx(sim, q1, q2)
                    
                    cmd_pipe.send({"status": "CHUNK_COMPLETE", "patch_idx": patch_idx})

                elif action == "MEASURE_BOUNDARY_BLOCH":
                    bloch_vectors = {}
                    for q in boundary_qubits:
                        # Z Expectation
                        z_exp = 1.0 - 2.0 * sim.prob(q)
                        
                        # X Expectation (H -> measure Z -> H)
                        apply_h(sim, q)
                        x_exp = 1.0 - 2.0 * sim.prob(q)
                        apply_h(sim, q)
                        
                        # Y Expectation (Rx(-pi/2) -> measure Z -> Rx(pi/2))
                        apply_rx(sim, -np.pi/2, q)
                        y_exp = 1.0 - 2.0 * sim.prob(q)
                        apply_rx(sim, np.pi/2, q)

                        bloch_vectors[q] = (float(x_exp), float(y_exp), float(z_exp))
                        
                    cmd_pipe.send({"status": "BLOCH_EXTRACTED", "patch_idx": patch_idx, "data": bloch_vectors})

                elif action == "APPLY_LHV_KICKS":
                    kicks = cmd.get("kicks", {})
                    for raw_q, (kx, ky, kz) in kicks.items():
                        q = int(raw_q)
                        if kx != 0.0: apply_rx(sim, kx, q)
                        if ky != 0.0: apply_ry(sim, ky, q)
                        if kz != 0.0: apply_rz(sim, kz, q)
                    cmd_pipe.send({"status": "KICKS_APPLIED", "patch_idx": patch_idx})

                elif action == "MEASURE_MAGNETIZATION":
                    total_z = sum((1.0 - 2.0 * sim.prob(q)) for q in range(num_qubits))
                    cmd_pipe.send({"status": "MAGNETIZATION_MEASURED", "patch_idx": patch_idx, "data": total_z})
                
                elif action == "COMPUTE_BENCHMARKS":
                    shots = cmd.get("shots", 8192)
                    
                    try:
                        samples = sim.measure_shots(list(range(num_qubits)), shots)
                        counts = dict(collections.Counter(samples))
                        
                        if hasattr(sim, 'out_probs'):
                            ideal_probs = np.asarray(sim.out_probs(), dtype=np.float64)
                        else:
                            raise RuntimeError("Missing out_probs()")

                        n_pow = len(ideal_probs)
                        u_u = 1.0 / n_pow
                        
                        obs_probs = np.zeros(n_pow, dtype=np.float64)
                        for k, count in counts.items():
                            obs_probs[k] = count / shots

                        denom = np.sum((ideal_probs - u_u) ** 2)
                        numer = np.sum((ideal_probs - u_u) * (obs_probs - u_u))
                        expected_xeb = numer / denom if denom > 0 else 0.0

                        threshold = np.median(ideal_probs)
                        heavy_mask = ideal_probs > threshold
                        expected_hog = np.sum(obs_probs[heavy_mask])

                        del ideal_probs, obs_probs, heavy_mask

                    except Exception as e:
                        expected_xeb, expected_hog = 0.0, 0.0
                        print(f"Worker {patch_idx} benchmark failed: {e}")
                    
                    gc.collect() 
                    cmd_pipe.send({
                        "status": "BENCHMARKS_COMPUTED", 
                        "patch_idx": patch_idx, 
                        "data": {"xeb": float(expected_xeb), "hog": float(expected_hog)}
                    })

            except Exception as inner_e:
                try:
                    cmd_pipe.send({"status": "ERROR", "msg": str(inner_e)})
                except:
                    break
                
    except (EOFError, OSError, BrokenPipeError):
        pass
    finally:
        if sim is not None:
            del sim
        gc.collect()

# ==========================================
# 3. LHV WORMHOLE ORCHESTRATOR 
# ==========================================
class LHVWormholeEngine:
    def __init__(self, device_ids: List[int]):
        self._is_shutdown = False
        self.device_ids = device_ids
        
        self.patches, self.fence_edges = get_holographic_topology(boundary_size=4)
        self.intra_patch_edges = get_complete_intra_edges()
        
        self.num_patches = len(self.patches)
        self.boundary_map = {i: {} for i in range(self.num_patches)}
        
        for (pA, qA), (pB, qB) in self.fence_edges:
            self.boundary_map[pA].setdefault(qA, []).append((pB, qB))
            self.boundary_map[pB].setdefault(qB, []).append((pA, qA))

        self.ctx = mp.get_context('spawn')
        self.workers = []
        self.pipes = []

        print(f"Initializing {self.num_patches} LHV Islands (25 Qubits/Patch)...")
        for p_idx in range(self.num_patches):
            parent_conn, child_conn = self.ctx.Pipe()
            boundary_qubits = list(self.boundary_map[p_idx].keys())
            
            p = self.ctx.Process(
                target=persistent_island_worker,
                args=(
                    self.device_ids[p_idx % len(self.device_ids)], 
                    p_idx, 
                    len(self.patches[p_idx]), 
                    self.intra_patch_edges, 
                    boundary_qubits, 
                    child_conn
                )
            )
            p.start()
            self.workers.append(p)
            self.pipes.append(parent_conn)
            
        for i, pipe in enumerate(self.pipes):
            if pipe.poll(timeout=45.0): 
                msg = pipe.recv()
                if msg.get("status") != "READY":
                    self.shutdown()
                    raise RuntimeError(f"Worker {i} error: {msg.get('msg', '')}")
            else:
                self.shutdown()
                raise TimeoutError(f"Worker {i} timed out.")

    def sync_broadcast(self, action: str, kwargs_list: List[Dict] = None, timeout_secs: float = 300.0) -> Dict[int, Any]:
        if kwargs_list is None:
            kwargs_list = [{}] * self.num_patches
            
        for i, pipe in enumerate(self.pipes):
            payload = {"action": action}
            payload.update(kwargs_list[i])
            pipe.send(payload)
            
        results = {}
        pending = list(enumerate(self.pipes)) 
        deadline = time.monotonic() + timeout_secs
        
        while pending:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                self.shutdown()
                raise TimeoutError(f"Workers {[i for i, _ in pending]} timed out.")
                
            active_pipes = [p for _, p in pending]
            ready_pipes = wait(active_pipes, timeout=timeout)
            
            if not ready_pipes: continue
                
            still_pending = []
            for idx, pipe in pending:
                if pipe in ready_pipes:
                    res = pipe.recv()
                    if res.get("status") == "ERROR":
                        self.shutdown()
                        raise RuntimeError(f"Worker {idx} error: {res.get('msg')}")
                    results[idx] = res
                else:
                    still_pending.append((idx, pipe))
            pending = still_pending
            
        return results

    def evolve(self, total_time_steps: int, depth_per_step: int, coupling_strength: float):
        print(f"\nStarting Tuned XYZ LHV Evolution...")
        print(f"Steps: {total_time_steps} | RCS Depth: {depth_per_step} | g: {coupling_strength}\n")
        
        main_rng = np.random.default_rng()

        for t in range(total_time_steps):
            seeds = main_rng.integers(0, 2**32, size=self.num_patches)
            self.sync_broadcast("RCS_CHUNK", [{"seed": int(seeds[i]), "depth": depth_per_step} for i in range(self.num_patches)])
            
            # Extract full (X, Y, Z) projections from all boundaries
            xyz_results = self.sync_broadcast("MEASURE_BOUNDARY_BLOCH")
            patch_bloch = {p_idx: res["data"] for p_idx, res in xyz_results.items()}

            kick_payloads = [{"kicks": {}} for _ in range(self.num_patches)]
            
            for pA in range(self.num_patches):
                for qA, neighbors in self.boundary_map[pA].items():
                    n_neighbors = len(neighbors)
                    kx, ky, kz = 0.0, 0.0, 0.0
                    
                    for (pB, qB) in neighbors:
                        xB, yB, zB = patch_bloch[pB][qB]
                        
                        # SDRP Style Tuned XYZ Communication 
                        kx += (coupling_strength * xB) / np.sqrt(n_neighbors)
                        ky += (coupling_strength * yB) / np.sqrt(n_neighbors)
                        kz += (coupling_strength * zB) / np.sqrt(n_neighbors)
                        
                    kick_payloads[pA]["kicks"][qA] = (kx, ky, kz)

            self.sync_broadcast("APPLY_LHV_KICKS", kick_payloads)

            # Analyze full inner product <V_A • V_B> cross-correlation
            cross_corr = 0.0
            edge_count = 0
            seen = set()
            
            for pA in range(self.num_patches):
                for qA, neighbors in self.boundary_map[pA].items():
                    for (pB, qB) in neighbors:
                        key = (min((pA, qA), (pB, qB)), max((pA, qA), (pB, qB)))
                        if key not in seen:
                            seen.add(key)
                            vA = patch_bloch[pA][qA]
                            vB = patch_bloch[pB][qB]
                            
                            # Dot product of the two Bloch vectors
                            dot_prod = vA[0]*vB[0] + vA[1]*vB[1] + vA[2]*vB[2]
                            cross_corr += dot_prod
                            edge_count += 1
                        
            avg_corr = cross_corr / edge_count if edge_count > 0 else 0.0
            print(f"Step {t:03d} | Boundary XYZ Correlation (V_A • V_B): {avg_corr:+.4f}")
            
            if t == total_time_steps - 1:
                print(f"         +-- Calculating Final Benchmarks...")
                bench_res = self.sync_broadcast("COMPUTE_BENCHMARKS", [{"shots": 8192} for _ in range(self.num_patches)])
                avg_xeb = sum(res["data"]["xeb"] for res in bench_res.values()) / self.num_patches
                avg_hog = sum(res["data"]["hog"] for res in bench_res.values()) / self.num_patches
                print(f"         +-- Avg XEB: {avg_xeb:.4f} | Avg HOG: {avg_hog:.4f}")

    def shutdown(self):
        if self._is_shutdown: return
        self._is_shutdown = True
        
        print("\nShutting down LHV Islands...")
        for pipe in getattr(self, 'pipes', []):
            try:
                if not pipe.closed: pipe.send({"action": "SHUTDOWN"})
            except: pass
                
        for p in getattr(self, 'workers', []):
            p.join(timeout=5)
            if p.is_alive(): p.terminate()
                
        for pipe in getattr(self, 'pipes', []):
            try:
                if not pipe.closed: pipe.close()
            except: pass

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    gpu_env = os.environ.get("WORMHOLE_GPUS", "0,1,2,3,4,5") 
    base_gpus = [int(g.strip()) for g in gpu_env.split(',')]
    
    num_patches = 12
    AVAILABLE_GPUS = [base_gpus[i // 2 % len(base_gpus)] for i in range(num_patches)]
    
    engine = LHVWormholeEngine(device_ids=AVAILABLE_GPUS)

    try:
        engine.evolve(
            total_time_steps=10, 
            depth_per_step=1, 
            coupling_strength=0.15 
        )
    finally:
        engine.shutdown()
