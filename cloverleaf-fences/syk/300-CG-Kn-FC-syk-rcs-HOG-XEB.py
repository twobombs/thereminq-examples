import os
import gc
import time
import signal
import numpy as np
import multiprocessing as mp
from multiprocessing.connection import Connection, wait
from multiprocessing.synchronize import Semaphore
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
        ], q)

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
        sim.mtrx([complex(np.cos(-theta/2), np.sin(-theta/2)), 0j, 
                  0j, complex(np.cos(theta/2), np.sin(theta/2))], q)

def apply_cx(sim: Any, c: int, t: int) -> None:
    if hasattr(sim, 'cx'): 
        sim.cx(c, t)
    else: 
        sim.mcx([c], t)

# ==========================================
# 1. HOLOGRAPHIC TOPOLOGY (BULK/BOUNDARY)
# ==========================================
def get_complete_intra_edges(num_qubits: int = 25) -> List[Tuple[int, int]]:
    """Fully connected intra-patch entanglement"""
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
# 2. ISOLATED PERSISTENT UNIVERSE (GPU WORKER)
# ==========================================
def persistent_universe_worker(
    device_id: int, 
    patch_idx: int, 
    num_qubits: int, 
    intra_edges: List[Tuple[int, int]], 
    boundary_qubits: List[int],
    cmd_pipe: Connection,
    ram_semaphore: Semaphore
) -> None:
    
    # Ignore Ctrl+C in child processes so the main process handles shutdown cleanly
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    # ---------------------------------------------------------
    # HARDWARE SAFEGUARD: Prevent NumPy Thread Thrashing
    # 12 workers * 4 threads = 48 threads utilized efficiently.
    # ---------------------------------------------------------
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    
    sim = None
    try:
        if boundary_qubits:
            assert max(boundary_qubits) < num_qubits, f"Worker {patch_idx}: Boundary index {max(boundary_qubits)} out of bounds."

        os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(device_id)
        os.environ["QRACK_QPAGER_DEVICES"] = str(device_id)
        os.environ["QRACK_QUNITMULTI_DEVICES"] = str(device_id)

        from pyqrack import QrackSimulator
        sim = QrackSimulator(
            qubitCount=num_qubits,
            isOpenCL=True,
            isTensorNetwork=False,
            isSchmidtDecompose=False
        )
        
        for q in range(num_qubits):
            apply_h(sim, q)
            
        cmd_pipe.send({"status": "READY", "patch_idx": patch_idx})
        
        rotation_gates = [apply_rx, apply_ry, apply_rz]
        rng_local = np.random.default_rng()

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

                elif action == "MEASURE_BOUNDARY_Z":
                    z_exp = {}
                    for q in boundary_qubits:
                        p_one = sim.prob(q)
                        z_exp[q] = 1.0 - 2.0 * p_one
                    cmd_pipe.send({"status": "Z_EXP_COMPUTED", "patch_idx": patch_idx, "data": z_exp})

                elif action == "APPLY_WORMHOLE_KICKS":
                    kicks = cmd.get("kicks", {})
                    for raw_q, theta in kicks.items():
                        q = int(raw_q)
                        if not (0 <= q < num_qubits):
                            raise IndexError(f"Kick qubit index {q} out of bounds")
                        apply_rz(sim, theta, q)
                    cmd_pipe.send({"status": "KICKS_APPLIED", "patch_idx": patch_idx})

                elif action == "MEASURE_MAGNETIZATION":
                    total_z = sum((1.0 - 2.0 * sim.prob(q)) for q in range(num_qubits))
                    cmd_pipe.send({"status": "MAGNETIZATION_MEASURED", "patch_idx": patch_idx, "data": total_z})
                
                elif action == "COMPUTE_BENCHMARKS":
                    
                    # ---------------------------------------------------------
                    # HARDWARE SAFEGUARD: The RAM Traffic Light
                    # Prevents all 12 workers from spiking Python objects at once
                    # ---------------------------------------------------------
                    with ram_semaphore:
                        if hasattr(sim, 'dump_probabilities'):
                            probs = np.array(sim.dump_probabilities())
                        elif hasattr(sim, 'dump'):
                            state = np.array(sim.dump())
                            probs = np.abs(state)**2
                            del state # Immediate flush
                        elif hasattr(sim, 'get_state_vector'):
                            state = np.array(sim.get_state_vector())
                            probs = np.abs(state)**2
                            del state # Immediate flush
                        elif hasattr(sim, 'get_state'):
                            state = np.array(sim.get_state())
                            probs = np.abs(state)**2
                            del state # Immediate flush
                        else:
                            dim = 1 << num_qubits
                            probs = np.ones(dim, dtype=np.float64) / dim
                        
                        probs = probs / np.sum(probs)
                        dim = len(probs)
                        
                        expected_xeb = (dim * np.sum(probs**2)) - 1.0 
                        
                        if dim > 2**20:
                            # Sample probability values (not indices) to find a statistically 
                            # accurate median for the Porter-Thomas distribution without locking the CPU
                            sample = rng_local.choice(probs, size=2**18, replace=False)
                            median_p = np.median(sample)
                            del sample # Immediate flush
                        else:
                            median_p = np.median(probs)
                            
                        # Apply the sampled median threshold back against the full probability array
                        expected_hog = np.sum(probs[probs > median_p])
                        
                        # Aggressive Garbage Collection before releasing the semaphore
                        del probs
                        gc.collect() 
                    
                    cmd_pipe.send({
                        "status": "BENCHMARKS_COMPUTED", 
                        "patch_idx": patch_idx, 
                        "data": {"xeb": float(expected_xeb), "hog": float(expected_hog)}
                    })

            except Exception as inner_e:
                try:
                    cmd_pipe.send({"status": "ERROR", "msg": f"Failed during {action}: {str(inner_e)}"})
                except (EOFError, OSError, BrokenPipeError):
                    break
                continue
                
    except (EOFError, OSError, BrokenPipeError):
        pass
    except Exception as e:
        try:
            cmd_pipe.send({"status": "ERROR", "msg": f"Worker {patch_idx} failed fatally: {str(e)}"})
        except (EOFError, OSError, BrokenPipeError):
            pass
    finally:
        if sim is not None:
            del sim
        gc.collect()

# ==========================================
# 3. WORMHOLE ORCHESTRATOR (THE BULK SPACE)
# ==========================================
class TraversableWormholeEngine:
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
        
        # Limit CPU dumps to 6 parallel workers to cap RAM spikes 
        self.ram_semaphore = self.ctx.Semaphore(6)

        print(f"Initializing {self.num_patches} isolated GPU Universes (25 Qubits/Patch)...")
        for p_idx in range(self.num_patches):
            
            assert len(self.patches[p_idx]) == 25, f"Patch {p_idx} invalid size."
            
            parent_conn, child_conn = self.ctx.Pipe()
            boundary_qubits = list(self.boundary_map[p_idx].keys())
            
            p = self.ctx.Process(
                target=persistent_universe_worker,
                args=(
                    self.device_ids[p_idx % len(self.device_ids)], 
                    p_idx, 
                    len(self.patches[p_idx]), 
                    self.intra_patch_edges, 
                    boundary_qubits, 
                    child_conn,
                    self.ram_semaphore
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
                    raise RuntimeError(f"Worker {i} initialized with bad state: {msg.get('msg', '')}")
                if msg.get("patch_idx") != i:
                    self.shutdown()
                    raise RuntimeError(f"Worker {i} returned mismatched patch_idx: expected {i}, got {msg.get('patch_idx')}")
            else:
                self.shutdown()
                raise TimeoutError(f"Worker {i} failed to initialize within 45 seconds.")

    def sync_broadcast(self, action: str, kwargs_list: List[Dict] = None, timeout_secs: float = 300.0) -> Dict[int, Any]:
        if kwargs_list is None:
            kwargs_list = [{}] * self.num_patches
            
        for i, pipe in enumerate(self.pipes):
            payload = {"action": action}
            payload.update(kwargs_list[i])
            try:
                pipe.send(payload)
            except (BrokenPipeError, OSError) as e:
                self.shutdown()
                raise RuntimeError(f"Worker {i} pipe broken during send: {e}")
            
        results = {}
        pending = list(enumerate(self.pipes)) 
        
        deadline = time.monotonic() + timeout_secs
        
        while pending:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                self.shutdown()
                raise TimeoutError(f"Workers {[i for i, _ in pending]} timed out during {action}.")
                
            active_pipes = [p for _, p in pending]
            ready_pipes = wait(active_pipes, timeout=timeout)
            
            if not ready_pipes:
                continue
                
            still_pending = []
            for idx, pipe in pending:
                if pipe in ready_pipes:
                    try:
                        res = pipe.recv()
                        if res.get("status") == "ERROR":
                            self.shutdown()
                            raise RuntimeError(f"Worker {idx} reported an error: {res.get('msg')}")
                            
                        if res.get("patch_idx") != idx:
                            self.shutdown()
                            raise RuntimeError(f"Worker {idx} returned mismatched patch_idx during {action}: expected {idx}, got {res.get('patch_idx')}")
                            
                        results[idx] = res
                    except (EOFError, OSError, BrokenPipeError) as e:
                        self.shutdown()
                        raise RuntimeError(f"Worker {idx} pipe failed during recv: {e}")
                else:
                    still_pending.append((idx, pipe))
            
            pending = still_pending
            
        return results

    def evolve(self, total_time_steps: int, depth_per_step: int, coupling_strength: float):
        print(f"\nStarting SYK Wormhole Time Evolution...")
        print(f"Total Steps: {total_time_steps} | RCS Depth/Step: {depth_per_step} | g: {coupling_strength}\n")
        
        main_rng = np.random.default_rng()

        for t in range(total_time_steps):
            seeds = main_rng.integers(0, 2**32, size=self.num_patches)
            self.sync_broadcast("RCS_CHUNK", [{"seed": int(seeds[i]), "depth": depth_per_step} for i in range(self.num_patches)])
            
            z_results = self.sync_broadcast("MEASURE_BOUNDARY_Z")
            patch_z_exp = {p_idx: res["data"] for p_idx, res in z_results.items()}

            kick_payloads = [{"kicks": {}} for _ in range(self.num_patches)]
            
            for pA in range(self.num_patches):
                for qA, neighbors in self.boundary_map[pA].items():
                    n_neighbors = len(neighbors)
                    for (pB, qB) in neighbors:
                        z_B = patch_z_exp[pB][qB]
                        scaled_kick = (2.0 * coupling_strength * z_B) / np.sqrt(n_neighbors)
                        kick_payloads[pA]["kicks"][qA] = kick_payloads[pA]["kicks"].get(qA, 0.0) + scaled_kick

            self.sync_broadcast("APPLY_WORMHOLE_KICKS", kick_payloads)

            if t % 5 == 0 or t == total_time_steps - 1:
                mag_res = self.sync_broadcast("MEASURE_MAGNETIZATION")
                mag_sum = sum(res["data"] for res in mag_res.values())
                
                # Allow extended 600s timeout to safely clear the semaphore queue during large state dumps
                bench_res = self.sync_broadcast("COMPUTE_BENCHMARKS", timeout_secs=600.0)
                avg_xeb = sum(res["data"]["xeb"] for res in bench_res.values()) / self.num_patches
                avg_hog = sum(res["data"]["hog"] for res in bench_res.values()) / self.num_patches
                
                cross_corr = 0.0
                edge_count = 0
                seen = set()
                
                for pA in range(self.num_patches):
                    for qA, neighbors in self.boundary_map[pA].items():
                        for (pB, qB) in neighbors:
                            endpointA = (pA, qA)
                            endpointB = (pB, qB)
                            key = (min(endpointA, endpointB), max(endpointA, endpointB))
                            
                            if key not in seen:
                                seen.add(key)
                                # PHYSICS NOTE: Because the simulator strictly isolates Hilbert spaces,
                                # the joint state is inherently separable (rho_A x rho_B). Therefore,
                                # <Z_A Z_B> exactly factorizes to <Z_A><Z_B>. This calculation represents
                                # the maximum physical signal extractable under this mean-field decoupling limit.
                                cross_corr += patch_z_exp[pA][qA] * patch_z_exp[pB][qB]
                                edge_count += 1
                            
                avg_corr = cross_corr / edge_count if edge_count > 0 else 0.0
                
                total_z_mag = sum(abs(z) for z_dict in patch_z_exp.values() for z in z_dict.values())
                total_boundary_qubits = sum(len(z_dict) for z_dict in patch_z_exp.values())
                avg_z_mag = total_z_mag / total_boundary_qubits if total_boundary_qubits > 0 else 0.0

                print(f"Step {t:03d} | Bulk Mag: {mag_sum:+.4f} | Boundary <Z_A Z_B>: {avg_corr:+.4f} | Avg |Z|: {avg_z_mag:.4f} | Kicks: {sum(len(k['kicks']) for k in kick_payloads)}")
                print(f"         └─ Benchmarks -> XEB: {avg_xeb:.4f} (Ideal PT: ~1.0) | HOG: {avg_hog:.4f} (Ideal PT: ~0.846)")

    def shutdown(self):
        if self._is_shutdown:
            return
        self._is_shutdown = True
        
        print("\nCollapsing the Wormhole (Shutting down GPU workers)...")
        for pipe in getattr(self, 'pipes', []):
            try:
                if not pipe.closed:
                    pipe.send({"action": "SHUTDOWN"})
            except (EOFError, OSError, BrokenPipeError):
                pass
                
        for p in getattr(self, 'workers', []):
            p.join(timeout=5)
            if p.is_alive(): 
                p.terminate()
                
        for pipe in getattr(self, 'pipes', []):
            try:
                if not pipe.closed:
                    pipe.close()
            except (EOFError, OSError, BrokenPipeError):
                pass

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    # Adjust this string based on how many GPUs you actually have available.
    # E.g., for 6 GPUs: "0,1,2,3,4,5"
    gpu_env = os.environ.get("WORMHOLE_GPUS", "0,1,2,3,4,5") 
    base_gpus = [int(g.strip()) for g in gpu_env.split(',')]
    
    # GUARANTEED PAIRING LOGIC: Maps exactly 2 patches to each GPU 
    num_patches = 12
    AVAILABLE_GPUS = [base_gpus[i // 2 % len(base_gpus)] for i in range(num_patches)]
    
    wormhole_engine = TraversableWormholeEngine(device_ids=AVAILABLE_GPUS)

    try:
        wormhole_engine.evolve(
            total_time_steps=50, 
            depth_per_step=3, 
            coupling_strength=0.15 
        )

    finally:
        wormhole_engine.shutdown()
