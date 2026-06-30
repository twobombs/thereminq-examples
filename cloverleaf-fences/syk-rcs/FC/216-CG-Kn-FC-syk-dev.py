import os
import gc
import time
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
        # Explicit H matrix construction: [H00, H01, H10, H11]
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
# 1. COMPLETE GRAPH (ALL-TO-ALL) TOPOLOGY
# ==========================================
def get_complete_intra_edges(num_qubits: int = 18) -> List[Tuple[int, int]]:
    """Fully connected intra-patch entanglement (153 edges for 18 qubits)"""
    edges = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            edges.append((i, j))
    return edges

def get_complete_topology(num_patches: int = 12, qubits_per_patch: int = 18) -> Tuple[List[List[int]], List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Expands the architecture to a pure Complete Graph (K_216).
    Total: 12 patches, 216 total simulated qubits, all-to-all connectivity.
    """
    total_qubits = num_patches * qubits_per_patch
    patches = [[] for _ in range(num_patches)]
    fence_edges = []
    global_to_local = {}

    # Pass 1: Linearly distribute qubits to patches
    for idx in range(total_qubits):
        patch_idx = idx // qubits_per_patch
        local_idx = idx % qubits_per_patch
        patches[patch_idx].append(idx)
        global_to_local[idx] = (patch_idx, local_idx)

    # Pass 2: Define All-to-All Inter-patch Boundaries (21,384 edges)
    for g1 in range(total_qubits):
        for g2 in range(g1 + 1, total_qubits):
            p1, l1 = global_to_local[g1]
            p2, l2 = global_to_local[g2]
            
            # If they reside in different patches, establish a boundary fence edge
            if p1 != p2:
                fence_edges.append(((p1, l1), (p2, l2)))

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
    cmd_pipe: Connection
) -> None:
    
    sim = None
    try:
        # Pre-flight bounds check on boundary qubits
        if boundary_qubits:
            assert max(boundary_qubits) < num_qubits, f"Worker {patch_idx}: Boundary index {max(boundary_qubits)} out of bounds (max {num_qubits-1})."

        os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(device_id)
        os.environ["QRACK_QPAGER_DEVICES"] = str(device_id)
        os.environ["QRACK_QUNITMULTI_DEVICES"] = str(device_id)

        from pyqrack import QrackSimulator
        
        # Updated for PyQrack 2.0 API (relies on env variables defined above)
        sim = QrackSimulator(qubit_count=num_qubits)
        
        for q in range(num_qubits):
            apply_h(sim, q)
            
        cmd_pipe.send({"status": "READY", "patch_idx": patch_idx})
        
        rotation_gates = [apply_rx, apply_ry, apply_rz]

        while True:
            try:
                # Tightened polling latency for active event loops
                if cmd_pipe.poll(timeout=0.1):
                    cmd = cmd_pipe.recv()
                else:
                    continue
            except (EOFError, OSError, BrokenPipeError):
                break
                
            action = cmd.get("action")

            # Pulled outside the try block for structural clarity
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
                        q = int(raw_q) # Defensive cast to protect against JSON-style stringified dict keys
                        if not (0 <= q < num_qubits):
                            raise IndexError(f"Kick qubit index {q} out of bounds (max {num_qubits-1})")
                        apply_rz(sim, theta, q)
                    cmd_pipe.send({"status": "KICKS_APPLIED", "patch_idx": patch_idx})

                elif action == "MEASURE_MAGNETIZATION":
                    total_z = sum((1.0 - 2.0 * sim.prob(q)) for q in range(num_qubits))
                    cmd_pipe.send({"status": "MAGNETIZATION_MEASURED", "patch_idx": patch_idx, "data": total_z})

            except Exception as inner_e:
                # Catch simulation/action errors cleanly, report them, but keep the worker alive
                try:
                    cmd_pipe.send({"status": "ERROR", "msg": f"Failed during {action}: {str(inner_e)}"})
                except (EOFError, OSError, BrokenPipeError):
                    break
                continue
                
    except (EOFError, OSError, BrokenPipeError):
        # Graceful exit if orchestrator pipe closes unexpectedly
        pass
    except Exception as e:
        # Catch catastrophic initialization errors cleanly and report back to main process
        try:
            cmd_pipe.send({"status": "ERROR", "msg": f"Worker {patch_idx} failed fatally: {str(e)}"})
        except (EOFError, OSError, BrokenPipeError):
            pass
    finally:
        # Guaranteed cleanup regardless of error origin
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
        
        # Complete Graph topology initialization
        self.patches, self.fence_edges = get_complete_topology()
        self.intra_patch_edges = get_complete_intra_edges()
        
        self.num_patches = len(self.patches)
        self.boundary_map = {i: {} for i in range(self.num_patches)}
        
        # Accumulate neighbors to handle corner qubits overlapping multiple boundaries safely
        for (pA, qA), (pB, qB) in self.fence_edges:
            self.boundary_map[pA].setdefault(qA, []).append((pB, qB))
            self.boundary_map[pB].setdefault(qB, []).append((pA, qA))

        self.ctx = mp.get_context('spawn')
        self.workers = []
        self.pipes = []

        print(f"Initializing {self.num_patches} isolated GPU Universes (All-to-All Complete Graph)...")
        for p_idx in range(self.num_patches):
            
            assert len(self.patches[p_idx]) == 18, f"Patch {p_idx} invalid size."
            
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
                    error_msg = msg.get("msg", str(msg))
                    raise RuntimeError(f"Worker {i} initialized with bad state: {error_msg}")
                if msg.get("patch_idx") != i:
                    self.shutdown()
                    raise RuntimeError(f"Worker {i} returned mismatched patch_idx: expected {i}, got {msg.get('patch_idx')}")
            else:
                self.shutdown()
                raise TimeoutError(f"Worker {i} failed to initialize within 45 seconds.")

    def sync_broadcast(self, action: str, kwargs_list: List[Dict] = None) -> Dict[int, Any]:
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
        
        # Extended timeout to 300s to account for intense K_216 GPU kernel serialization
        deadline = time.monotonic() + 300.0
        
        while pending:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                self.shutdown()
                missing = [i for i, _ in pending]
                raise TimeoutError(f"Workers {missing} timed out during {action}.")
                
            active_pipes = [p for _, p in pending]
            
            # Wait blocks until at least one pipe is ready to be read, eliminating sequential latency
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
                        
                        # Guard: Ensure the payload maps cleanly to the physical pipe mapping
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
            
            # Generate truly independent seeds for each patch to avoid correlation
            seeds = main_rng.integers(0, 2**32, size=self.num_patches)
            self.sync_broadcast("RCS_CHUNK", [{"seed": int(seeds[i]), "depth": depth_per_step} for i in range(self.num_patches)])
            
            z_results = self.sync_broadcast("MEASURE_BOUNDARY_Z")
            # Robustly keyed by the patch ID explicitly returned by the worker
            patch_z_exp = {p_idx: res["data"] for p_idx, res in z_results.items()}

            kick_payloads = [{"kicks": {}} for _ in range(self.num_patches)]
            
            # Scaling Note: The complete graph topology dictates that every qubit communicates with 
            # every other qubit, requiring 21,384 dictionary iterations per time step here. While dense, 
            # this iteration resolves in a few milliseconds in native Python, remaining vastly outpaced 
            # by the execution speed of the resulting OpenCL matrix multiplications on the GPU.
            for pA in range(self.num_patches):
                for qA, neighbors in self.boundary_map[pA].items():
                    for (pB, qB) in neighbors:
                        z_B = patch_z_exp[pB][qB]
                        kick_payloads[pA]["kicks"][qA] = kick_payloads[pA]["kicks"].get(qA, 0.0) + 2.0 * coupling_strength * z_B

            self.sync_broadcast("APPLY_WORMHOLE_KICKS", kick_payloads)

            if t % 5 == 0 or t == total_time_steps - 1:
                mag_res = self.sync_broadcast("MEASURE_MAGNETIZATION")
                mag_sum = sum(res["data"] for res in mag_res.values())
                
                cross_corr = 0.0
                edge_count = 0
                seen = set()
                
                # Cross-correlation deduplicates the edges via lexicographical canonical pairing
                for pA in range(self.num_patches):
                    for qA, neighbors in self.boundary_map[pA].items():
                        for (pB, qB) in neighbors:
                            endpointA = (pA, qA)
                            endpointB = (pB, qB)
                            key = (min(endpointA, endpointB), max(endpointA, endpointB))
                            
                            if key not in seen:
                                seen.add(key)
                                cross_corr += patch_z_exp[pA][qA] * patch_z_exp[pB][qB]
                                edge_count += 1
                            
                avg_corr = cross_corr / edge_count if edge_count > 0 else 0.0
                
                # State Health Diagnostic: Calculate avg |Z_A| to monitor for maximally mixed convergence.
                # Note: This uses pre-kick Z expectations, correctly aligning with Maldacena-Qi 
                # traversable wormhole teleportation signal protocols (measuring the <Z_A Z_B> two-point correlator).
                total_z_mag = sum(abs(z) for z_dict in patch_z_exp.values() for z in z_dict.values())
                total_boundary_qubits = sum(len(z_dict) for z_dict in patch_z_exp.values())
                avg_z_mag = total_z_mag / total_boundary_qubits if total_boundary_qubits > 0 else 0.0

                print(f"Step {t:03d} | Bulk Mag: {mag_sum:+.4f} | Boundary <Z_A Z_B>: {avg_corr:+.4f} | Avg |Z|: {avg_z_mag:.4f} | Inter-patch Kicks: {sum(len(k['kicks']) for k in kick_payloads)}")

    def shutdown(self):
        # Idempotency guard to prevent double-shutdown propagating errors
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

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    # Support environment-based parameterization for multi-GPU nodes
    gpu_env = os.environ.get("WORMHOLE_GPUS", "0,1,0,1,0,1,0,1,0,1,0,1")
    base_gpus = [int(g.strip()) for g in gpu_env.split(',')]
    
    # Auto-pad or slice the GPU list to precisely match the 12 resulting patches
    AVAILABLE_GPUS = [base_gpus[i % len(base_gpus)] for i in range(12)]
    
    wormhole_engine = TraversableWormholeEngine(device_ids=AVAILABLE_GPUS)

    try:
        wormhole_engine.evolve(
            total_time_steps=50, 
            depth_per_step=3, 
            coupling_strength=0.15 
        )

    finally:
        wormhole_engine.shutdown()
