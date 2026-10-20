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
    elif hasattr(sim, 'mcx'):
        try:
            sim.mcx([c], t)
        except TypeError:
            sim.mcx([c], [t])
    else:
        raise RuntimeError("No CX gate available")

# ==========================================
# 1. HOLOGRAPHIC TOPOLOGY (BULK/BOUNDARY)
# ==========================================
def get_intra_edges(num_qubits: int = 25, topology: str = "FC") -> List[Tuple[int, int]]:
    edges = []
    if topology == "FC":
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                edges.append((i, j))
    elif topology == "RING":
        for i in range(num_qubits):
            edges.append((i, (i + 1) % num_qubits))
    elif topology == "STAR":
        for i in range(1, num_qubits):
            edges.append((0, i))
    else:
        raise ValueError(f"Unknown topology type: {topology}")
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
    cmd_pipe: Connection,
    gpu_lock: Any
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
        
        with gpu_lock:
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
                    drop_rate = cmd.get("drop_rate", 0.0) 
                    rng = np.random.default_rng(seed)

                    with gpu_lock:
                        for _ in range(depth):
                            for q in range(num_qubits):
                                gate_idx = rng.integers(0, 3)
                                theta = rng.uniform(-np.pi, np.pi)
                                rotation_gates[gate_idx](sim, theta, q)
                            
                            for q1, q2 in intra_edges:
                                if rng.random() > drop_rate:
                                    apply_cx(sim, q1, q2)
                    
                    cmd_pipe.send({"status": "CHUNK_COMPLETE", "patch_idx": patch_idx})

                elif action == "MEASURE_BOUNDARY_BLOCH":
                    bloch_vectors = {}
                    with gpu_lock:
                        for q in boundary_qubits:
                            z_exp = 1.0 - 2.0 * sim.prob(q)
                            
                            apply_h(sim, q)
                            x_exp = 1.0 - 2.0 * sim.prob(q)
                            apply_h(sim, q)
                            
                            apply_rx(sim, np.pi/2, q)
                            y_exp = 1.0 - 2.0 * sim.prob(q)
                            apply_rx(sim, -np.pi/2, q)

                            bloch_vectors[q] = (float(x_exp), float(y_exp), float(z_exp))
                        
                    cmd_pipe.send({"status": "BLOCH_EXTRACTED", "patch_idx": patch_idx, "data": bloch_vectors})

                elif action == "APPLY_LHV_KICKS":
                    kicks = cmd.get("kicks", {})
                    with gpu_lock:
                        for raw_q, (kx, ky, kz) in kicks.items():
                            q = int(raw_q)
                            if kx != 0.0: apply_rx(sim, kx, q)
                            if ky != 0.0: apply_ry(sim, ky, q)
                            if kz != 0.0: apply_rz(sim, kz, q)
                    cmd_pipe.send({"status": "KICKS_APPLIED", "patch_idx": patch_idx})

                elif action == "COMPUTE_BENCHMARKS":
                    shots = cmd.get("shots", 2048) 
                    
                    # FIX: Initialize all variables before the try block to prevent UnboundLocalError
                    avg_purity = 0.0
                    expected_xeb = 0.0
                    expected_hog = 0.0
                    purity_sum = 0.0
                    
                    try:
                        with gpu_lock:
                            for q in boundary_qubits:
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
                            
                        avg_purity = purity_sum / len(boundary_qubits) if boundary_qubits else 0.0
                        
                        # Note: This matrix dot product assumes PyQrack outputs little-endian bitstrings
                        # where qubit 0 is the least significant bit (LSB).
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
                    cmd_pipe.send({
                        "status": "BENCHMARKS_COMPUTED", 
                        "patch_idx": patch_idx, 
                        "data": {
                            "boundary_purity": float(avg_purity),
                            "xeb": expected_xeb,
                            "hog": expected_hog
                        }
                    })

            except Exception as inner_e:
                try:
                    cmd_pipe.send({"status": "ERROR", "msg": str(inner_e)})
                except (EOFError, OSError, BrokenPipeError):
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
    def __init__(self, device_ids: List[int], intra_topology: str = "FC", boundary_size: int = 4):
        self._is_shutdown = False
        self.device_ids = device_ids
        
        self.patches, self.fence_edges = get_holographic_topology(boundary_size=boundary_size)
        self.intra_patch_edges = get_intra_edges(num_qubits=25, topology=intra_topology)
        self.num_patches = len(self.patches)
        self.boundary_map = {i: {} for i in range(self.num_patches)}
        
        for (pA, bA), (pB, bB) in self.fence_edges:
            self.boundary_map[pA].setdefault(bA, []).append((pB, bB))
            self.boundary_map[pB].setdefault(bB, []).append((pA, bA))

        self.ctx = mp.get_context('spawn')
        
        self.gpu_locks = {gpu_id: self.ctx.Lock() for gpu_id in set(self.device_ids)}
        
        self.workers = []
        self.pipes = []

        print(f"Initializing {self.num_patches} LHV Islands (25 Qubits/Patch) with '{intra_topology}' Topology...")
        for p_idx in range(self.num_patches):
            parent_conn, child_conn = self.ctx.Pipe()
            boundary_qubits = list(self.boundary_map[p_idx].keys())
            assigned_gpu = self.device_ids[p_idx % len(self.device_ids)]
            
            p = self.ctx.Process(
                target=persistent_island_worker,
                args=(
                    assigned_gpu, 
                    p_idx, 
                    len(self.patches[p_idx]), 
                    self.intra_patch_edges, 
                    boundary_qubits, 
                    child_conn,
                    self.gpu_locks[assigned_gpu]
                )
            )
            p.start()
            
            # FIX: Explicitly close the child connection in the parent to prevent FD leaks
            child_conn.close()
            
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
                    try:
                        res = pipe.recv()
                        if res.get("status") == "ERROR":
                            self.shutdown()
                            raise RuntimeError(f"Worker {idx} error: {res.get('msg')}")
                        results[idx] = res
                    except (EOFError, OSError):
                        self.shutdown()
                        raise RuntimeError(f"Worker {idx} connection crashed during {action}.")
                else:
                    still_pending.append((idx, pipe))
            pending = still_pending
            
        return results

    def evolve(self, total_time_steps: int, depth_per_step: int, coupling_strength: float, drop_rate: float = 0.0):
        print(f"\nStarting Tuned XYZ LHV Evolution...")
        print(f"Steps: {total_time_steps} | RCS Depth: {depth_per_step} | g: {coupling_strength}")
        print(f"Scrambling Drop Rate: {drop_rate*100}% of internal CX gates skipped.\n")
        
        main_rng = np.random.default_rng()

        for t in range(total_time_steps):
            seeds = main_rng.integers(0, 2**32, size=self.num_patches)
            self.sync_broadcast("RCS_CHUNK", [
                {"seed": int(seeds[i]), "depth": depth_per_step, "drop_rate": drop_rate} 
                for i in range(self.num_patches)
            ])
            
            xyz_results = self.sync_broadcast("MEASURE_BOUNDARY_BLOCH")
            patch_bloch = {p_idx: res["data"] for p_idx, res in xyz_results.items()}

            kick_payloads = [{"kicks": {}} for _ in range(self.num_patches)]
            
            for pA in range(self.num_patches):
                for bA, neighbors in self.boundary_map[pA].items():
                    n_neighbors = len(neighbors)
                    kx, ky, kz = 0.0, 0.0, 0.0
                    
                    for (pB, bB) in neighbors:
                        xB, yB, zB = patch_bloch[pB][bB]
                        
                        kx += (coupling_strength * xB) / np.sqrt(n_neighbors)
                        ky += (coupling_strength * yB) / np.sqrt(n_neighbors)
                        kz += (coupling_strength * zB) / np.sqrt(n_neighbors)
                        
                    kick_payloads[pA]["kicks"][bA] = (kx, ky, kz)

            self.sync_broadcast("APPLY_LHV_KICKS", kick_payloads)

            cross_corr = 0.0
            edge_count = 0
            seen = set()
            
            for pA in range(self.num_patches):
                for bA, neighbors in self.boundary_map[pA].items():
                    for (pB, bB) in neighbors:
                        key = (min((pA, bA), (pB, bB)), max((pA, bA), (pB, bB)))
                        if key not in seen:
                            seen.add(key)
                            vA = patch_bloch[pA][bA]
                            vB = patch_bloch[pB][bB]
                            
                            dot_prod = vA[0]*vB[0] + vA[1]*vB[1] + vA[2]*vB[2]
                            cross_corr += dot_prod
                            edge_count += 1
                        
            avg_corr = cross_corr / edge_count if edge_count > 0 else 0.0
            print(f"Step {t:03d} | Boundary XYZ Correlation (V_A * V_B): {avg_corr:+.4f}")
            
            if t == total_time_steps - 1:
                print(f"         +-- Calculating Final Benchmarks...")
                
                # FIX: Stagger benchmarks sequentially to prevent a 3GB RAM spike.
                # Instead of a single sync_broadcast, we iterate one by one.
                bench_res = {}
                for p_idx, pipe in enumerate(self.pipes):
                    pipe.send({"action": "COMPUTE_BENCHMARKS", "shots": 2048})
                    
                    # Wait for just this worker to finish and garbage collect
                    ready = wait([pipe], timeout=300.0)
                    if not ready:
                        self.shutdown()
                        raise TimeoutError(f"Worker {p_idx} timed out during benchmarks.")
                        
                    try:
                        res = pipe.recv()
                        if res.get("status") == "ERROR":
                            self.shutdown()
                            raise RuntimeError(f"Worker {p_idx} error: {res.get('msg')}")
                        bench_res[p_idx] = res
                    except (EOFError, OSError):
                        self.shutdown()
                        raise RuntimeError(f"Worker {p_idx} crashed during COMPUTE_BENCHMARKS.")

                avg_purity = sum(res["data"]["boundary_purity"] for res in bench_res.values()) / self.num_patches
                avg_xeb = sum(res["data"]["xeb"] for res in bench_res.values()) / self.num_patches
                avg_hog = sum(res["data"]["hog"] for res in bench_res.values()) / self.num_patches
                
                print(f"         +-- Avg XEB: {avg_xeb:.4f} | Avg HOG: {avg_hog:.4f}")
                print(f"         +-- Avg Boundary Purity (1.0 = Pure, 0.0 = Mixed): {avg_purity:.4f}")

    def shutdown(self):
        if self._is_shutdown: return
        self._is_shutdown = True
        
        print("\nShutting down LHV Islands...")
        for pipe in getattr(self, 'pipes', []):
            try:
                if not pipe.closed:
                    # FIX: Drain the pipe before sending SHUTDOWN to avoid blocking
                    while pipe.poll():
                        pipe.recv()
                    pipe.send({"action": "SHUTDOWN"})
            except (OSError, BrokenPipeError, EOFError):
                pass
                
        for p in getattr(self, 'workers', []):
            try:
                p.join(timeout=5)
                if p.is_alive(): 
                    p.terminate()
            except Exception:
                pass
                
        for pipe in getattr(self, 'pipes', []):
            try:
                if not pipe.closed: pipe.close()
            except (OSError, BrokenPipeError):
                pass

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    gpu_env = os.environ.get("WORMHOLE_GPUS", "0,1") 
    base_gpus = [int(g.strip()) for g in gpu_env.split(',')]
    
    num_patches = 12
    
    # FIX: Clean round-robin GPU assignment prevents uneven loading on arbitrary patch counts
    AVAILABLE_GPUS = [base_gpus[i % len(base_gpus)] for i in range(num_patches)]
    
    engine = LHVWormholeEngine(device_ids=AVAILABLE_GPUS, intra_topology="FC", boundary_size=4)

    try:
        engine.evolve(
            total_time_steps=10, 
            depth_per_step=1, 
            coupling_strength=0.5,
            drop_rate=0.8 
        )
    finally:
        engine.shutdown()
