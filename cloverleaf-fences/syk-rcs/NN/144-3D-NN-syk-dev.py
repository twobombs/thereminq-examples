import os
import gc
import numpy as np
import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import List, Tuple, Dict, Any

# ==========================================
# 0. PYQRACK API SAFEGUARDS & GATES
# ==========================================
PX, PY, PZ = 1, 2, 3

def apply_h(sim: Any, q: int) -> None:
    if hasattr(sim, 'h'): 
        sim.h(q)
    else: 
        sim.mtrx([complex(1/np.sqrt(2), 0)] * 3 + [complex(-1/np.sqrt(2), 0)], [q])

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
# 1. 3D TOPOLOGY & BOUNDARY ROUTING
# ==========================================
def get_3x6_edges() -> List[Tuple[int, int]]:
    """Intra-patch entanglement for 18 qubits (Base QPU Node)"""
    edges = []
    for r in range(3):
        for c in range(6):
            idx = r * 6 + c
            if c < 5: edges.append((idx, idx + 1))
            if r < 2: edges.append((idx, idx + 6))
    return edges

def get_3d_topology() -> Tuple[List[List[Tuple[int, int]]], List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Expands the architecture to a 3D stack (2 layers of 4 patches).
    Total: 8 patches, 144 total simulated qubits.
    """
    patches = [[] for _ in range(8)]
    fence_edges = []
    global_to_local = {}

    # Pass 1: Populate global_to_local and patches to ensure all keys exist
    for z in range(2): 
        for r in range(6):
            for c in range(12):
                idx = z * 72 + r * 12 + c
                patch_r = r // 3
                patch_c = c // 6
                # Patches 0-3 are Bottom Layer, Patches 4-7 are Top Layer
                patch_idx = z * 4 + (patch_r * 2 + patch_c) 
                local_idx = len(patches[patch_idx])
                patches[patch_idx].append((idx, local_idx))
                global_to_local[idx] = (patch_idx, local_idx)

    # Pass 2: Define 2D Lateral Fences and 3D Vertical Fences
    for z in range(2):
        for r in range(6):
            for c in range(12):
                g1 = z * 72 + r * 12 + c
                
                # 2D Lateral Boundaries (East-West / North-South inside a layer)
                if c == 5:
                    g2_east = z * 72 + r * 12 + (c + 1)
                    fence_edges.append((global_to_local[g1], global_to_local[g2_east]))
                if r == 2:
                    g2_south = z * 72 + (r + 1) * 12 + c
                    fence_edges.append((global_to_local[g1], global_to_local[g2_south]))
                
                # 3D Vertical Boundaries (Partial FC interconnect between Z-layers)
                if z == 0 and (r % 2 == 0 and c % 3 == 0):
                    g2_up = (z + 1) * 72 + r * 12 + c
                    fence_edges.append((global_to_local[g1], global_to_local[g2_up]))

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
    
    # Pre-flight bounds check on boundary qubits
    try:
        if boundary_qubits:
            assert max(boundary_qubits) < num_qubits, f"Worker {patch_idx}: Boundary index {max(boundary_qubits)} out of bounds (max {num_qubits-1})."
    except AssertionError as e:
        cmd_pipe.send({"status": "ERROR", "msg": str(e)})
        return

    os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(device_id)
    os.environ["QRACK_QPAGER_DEVICES"] = str(device_id)
    os.environ["QRACK_QUNITMULTI_DEVICES"] = str(device_id)

    # Delay READY signal until GPU context is fully allocated and initialized
    try:
        from pyqrack import QrackSimulator
        sim = QrackSimulator(
            qubitCount=num_qubits,
            isOpenCL=True,
            isTensorNetwork=False,
            isSchmidtDecompose=False
        )
        
        for q in range(num_qubits):
            apply_h(sim, q)
            
        cmd_pipe.send({"status": "READY"})
    except Exception as e:
        cmd_pipe.send({"status": "ERROR", "msg": f"Worker {patch_idx} init failed: {e}"})
        return

    rotation_gates = [apply_rx, apply_ry, apply_rz]

    try:
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

            if action == "SHUTDOWN":
                break

            elif action == "RCS_CHUNK":
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
                
                cmd_pipe.send({"status": "CHUNK_COMPLETE"})

            elif action == "MEASURE_BOUNDARY_Z":
                z_exp = {}
                for q in boundary_qubits:
                    p_one = sim.prob(q)
                    z_exp[q] = 1.0 - 2.0 * p_one
                cmd_pipe.send({"status": "Z_EXP_COMPUTED", "data": z_exp})

            elif action == "APPLY_WORMHOLE_KICKS":
                kicks = cmd.get("kicks", {})
                for q, theta in kicks.items():
                    apply_rz(sim, theta, q)
                cmd_pipe.send({"status": "KICKS_APPLIED"})

            elif action == "MEASURE_MAGNETIZATION":
                total_z = sum((1.0 - 2.0 * sim.prob(q)) for q in range(num_qubits))
                cmd_pipe.send({"status": "MAGNETIZATION_MEASURED", "data": total_z})

    finally:
        del sim
        gc.collect()

# ==========================================
# 3. WORMHOLE ORCHESTRATOR (THE BULK SPACE)
# ==========================================
class TraversableWormholeEngine:
    def __init__(self, device_ids: List[int]):
        self.device_ids = device_ids
        self.patches, self.fence_edges = get_3d_topology()
        self.intra_patch_edges = get_3x6_edges()
        
        self.num_patches = len(self.patches)
        self.boundary_map = {i: {} for i in range(self.num_patches)}
        
        # Accumulate neighbors to handle corner qubits overlapping multiple boundaries safely
        for (pA, qA), (pB, qB) in self.fence_edges:
            self.boundary_map[pA].setdefault(qA, []).append((pB, qB))
            self.boundary_map[pB].setdefault(qB, []).append((pA, qA))

        self.ctx = mp.get_context('spawn')
        self.workers = []
        self.pipes = []

        print(f"Initializing {self.num_patches} isolated GPU Universes (3D Stack)...")
        for p_idx in range(self.num_patches):
            
            assert len(self.patches[p_idx]) == 18, f"Patch {p_idx} invalid size. Intra-patch edges assume 3x6 topologies."
            
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
            if pipe.poll(timeout=30.0): # Increased timeout slightly to account for parallel GPU cold starts
                msg = pipe.recv()
                if msg.get("status") != "READY":
                    self.shutdown()
                    error_msg = msg.get("msg", str(msg))
                    raise RuntimeError(f"Worker {i} initialized with bad state: {error_msg}")
            else:
                self.shutdown()
                raise TimeoutError(f"Worker {i} failed to initialize within 30 seconds.")

    def sync_broadcast(self, action: str, kwargs_list: List[Dict] = None) -> List[Any]:
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
            
        results = []
        for i, pipe in enumerate(self.pipes):
            if pipe.poll(timeout=120.0):
                results.append(pipe.recv())
            else:
                self.shutdown()
                raise TimeoutError(f"Worker {i} timed out during {action}.")
        return results

    def evolve(self, total_time_steps: int, depth_per_step: int, coupling_strength: float):
        print(f"\nStarting 3D SYK Wormhole Time Evolution...")
        print(f"Total Steps: {total_time_steps} | RCS Depth/Step: {depth_per_step} | g: {coupling_strength}\n")

        for t in range(total_time_steps):
            
            step_seed = int(np.random.randint(0, 1000000)) 
            
            self.sync_broadcast("RCS_CHUNK", [{"seed": step_seed + i, "depth": depth_per_step} for i in range(self.num_patches)])
            
            z_results = self.sync_broadcast("MEASURE_BOUNDARY_Z")
            patch_z_exp = {i: res["data"] for i, res in enumerate(z_results)}

            kick_payloads = [{"kicks": {}} for _ in range(self.num_patches)]
            
            # Kicks are accumulated over all boundaries a qubit touches
            for pA in range(self.num_patches):
                for qA, neighbors in self.boundary_map[pA].items():
                    for (pB, qB) in neighbors:
                        z_B = patch_z_exp[pB][qB]
                        kick_payloads[pA]["kicks"][qA] = kick_payloads[pA]["kicks"].get(qA, 0.0) + 2.0 * coupling_strength * z_B

            self.sync_broadcast("APPLY_WORMHOLE_KICKS", kick_payloads)

            if t % 5 == 0 or t == total_time_steps - 1:
                mag_res = self.sync_broadcast("MEASURE_MAGNETIZATION")
                mag_sum = sum(res["data"] for res in mag_res)
                
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

                print(f"Step {t:03d} | Bulk Mag: {mag_sum:+.4f} | Boundary <Z_A Z_B>: {avg_corr:+.4f} | 2D/3D Kicks: {sum(len(k['kicks']) for k in kick_payloads)}")

    def shutdown(self):
        print("\nCollapsing the Wormhole (Shutting down GPU workers)...")
        for pipe in self.pipes:
            try:
                if not pipe.closed:
                    pipe.send({"action": "SHUTDOWN"})
                    
                    # Extended drain loop for straggling responses
                    while pipe.poll(timeout=0.5):
                        pipe.recv()
            except (EOFError, OSError, BrokenPipeError):
                pass
                
        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive(): 
                p.terminate()

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    AVAILABLE_GPUS = [0, 0, 0, 0, 0, 0, 0, 0] 
    
    wormhole_engine = TraversableWormholeEngine(device_ids=AVAILABLE_GPUS)

    try:
        wormhole_engine.evolve(
            total_time_steps=50, 
            depth_per_step=3, 
            coupling_strength=0.15 
        )

    finally:
        wormhole_engine.shutdown()
      
