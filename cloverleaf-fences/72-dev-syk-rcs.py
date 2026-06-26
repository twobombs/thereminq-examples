import os
import gc
import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Dict, Any

# ==========================================
# 0. PYQRACK API SAFEGUARDS & GATES
# ==========================================
try:
    from pyqrack import Pauli
    PX = getattr(Pauli, 'PauliX', getattr(Pauli, 'X', 1))
    PY = getattr(Pauli, 'PauliY', getattr(Pauli, 'Y', 2))
    PZ = getattr(Pauli, 'PauliZ', getattr(Pauli, 'Z', 3))
except ImportError:
    PX, PY, PZ = 1, 2, 3

def apply_h(sim: Any, q: int) -> None:
    if hasattr(sim, 'h'): sim.h(q)
    else: sim.mtrx([complex(1/np.sqrt(2), 0)] * 3 + [complex(-1/np.sqrt(2), 0)], [q])

def apply_rx(sim: Any, theta: float, q: int) -> None:
    if hasattr(sim, 'r'): sim.r(PX, float(theta), q)
    else: sim.mtrx([complex(np.cos(theta/2), 0), complex(0, -np.sin(theta/2)), complex(0, -np.sin(theta/2)), complex(np.cos(theta/2), 0)], [q])

def apply_ry(sim: Any, theta: float, q: int) -> None:
    if hasattr(sim, 'r'): sim.r(PY, float(theta), q)
    else: sim.mtrx([complex(np.cos(theta/2), 0), complex(-np.sin(theta/2), 0), complex(np.sin(theta/2), 0), complex(np.cos(theta/2), 0)], [q])

def apply_rz(sim: Any, theta: float, q: int) -> None:
    if hasattr(sim, 'r'): sim.r(PZ, float(theta), q)
    else: sim.mtrx([complex(np.cos(-theta/2), np.sin(-theta/2)), 0j, 0j, complex(np.cos(theta/2), np.sin(theta/2))], [q])

def apply_cx(sim: Any, c: int, t: int) -> None:
    if hasattr(sim, 'cx'): sim.cx(c, t)
    else: sim.mcx([c], t)

# ==========================================
# 1. TOPOLOGY & BOUNDARY ROUTING
# ==========================================
def get_3x6_edges() -> List[Tuple[int, int]]:
    """Intra-patch entanglement for 18 qubits"""
    edges = []
    for r in range(3):
        for c in range(6):
            idx = r * 6 + c
            if c < 5: edges.append((idx, idx + 1))
            if r < 2: edges.append((idx, idx + 6))
    return edges

def get_topology() -> Tuple[List[List[Tuple[int, int]]], List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    patches = [[] for _ in range(4)]
    fence_edges = []

    for r in range(6):
        for c in range(12):
            idx = r * 12 + c
            patch_r = r // 3
            patch_c = c // 6
            patch_idx = patch_r * 2 + patch_c
            local_idx = len(patches[patch_idx])
            patches[patch_idx].append((idx, local_idx))

    global_to_local = {}
    for p_idx, patch in enumerate(patches):
        for global_idx, local_idx in patch:
            global_to_local[global_idx] = (p_idx, local_idx)

    for r in range(6):
        for c in range(12):
            g1 = r * 12 + c
            if c == 5:
                g2 = r * 12 + (c + 1)
                fence_edges.append((global_to_local[g1], global_to_local[g2]))
            if r == 2:
                g2 = (r + 1) * 12 + c
                fence_edges.append((global_to_local[g1], global_to_local[g2]))

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
    cmd_pipe: mp.connection.Connection
) -> None:
    """
    Runs an infinite event loop, maintaining the VRAM state vector.
    Receives RCS chunks, executes them, and applies ER=EPR coupling kicks.
    """
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

    # Fast indexed lookup array for the inner RCS loop
    rotation_gates = [apply_rx, apply_ry, apply_rz]

    for q in range(num_qubits):
        apply_h(sim, q)

    cmd_pipe.send({"status": "READY"})

    try:
        while True:
            if cmd_pipe.poll(timeout=1.0):
                cmd = cmd_pipe.recv()
            else:
                continue
                
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
        self.patches, self.fence_edges = get_topology()
        self.intra_patch_edges = get_3x6_edges()

        self.boundary_map = {i: {} for i in range(4)}
        for (pA, qA), (pB, qB) in self.fence_edges:
            self.boundary_map[pA][qA] = (pB, qB)
            self.boundary_map[pB][qB] = (pA, qA)

        self.ctx = mp.get_context('spawn')
        self.workers = []
        self.pipes = []

        print("Initializing isolated GPU Universes...")
        for p_idx in range(4):
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
            if pipe.poll(timeout=15.0):
                msg = pipe.recv()
                if msg.get("status") != "READY":
                    raise RuntimeError(f"Worker {i} initialized with bad state: {msg}")
            else:
                self.shutdown()
                raise TimeoutError(f"Worker {i} failed to initialize within 15 seconds.")

    def sync_broadcast(self, action: str, kwargs_list: List[Dict] = None) -> List[Any]:
        if kwargs_list is None:
            kwargs_list = [{}] * 4
            
        for i, pipe in enumerate(self.pipes):
            payload = {"action": action}
            payload.update(kwargs_list[i])
            pipe.send(payload)
            
        results = []
        for pipe in self.pipes:
            if pipe.poll(timeout=120.0):
                results.append(pipe.recv())
            else:
                raise TimeoutError(f"Worker process timed out during {action}.")
        return results

    def evolve(self, total_time_steps: int, depth_per_step: int, coupling_strength: float):
        print(f"\nStarting SYK Traversable Wormhole Time Evolution...")
        print(f"Total Steps: {total_time_steps} | RCS Depth/Step: {depth_per_step} | g: {coupling_strength}\n")

        for t in range(total_time_steps):
            step_seed = np.random.randint(0, 1000000)
            
            self.sync_broadcast("RCS_CHUNK", [{"seed": step_seed + i, "depth": depth_per_step} for i in range(4)])
            
            z_results = self.sync_broadcast("MEASURE_BOUNDARY_Z")
            patch_z_exp = {i: res["data"] for i, res in enumerate(z_results)}

            kick_payloads = [{"kicks": {}} for _ in range(4)]
            for pA in range(4):
                for qA, (pB, qB) in self.boundary_map[pA].items():
                    z_B = patch_z_exp[pB][qB]
                    theta_kick = 2.0 * coupling_strength * z_B
                    kick_payloads[pA]["kicks"][qA] = theta_kick

            self.sync_broadcast("APPLY_WORMHOLE_KICKS", kick_payloads)

            if t % 5 == 0 or t == total_time_steps - 1:
                mag_res = self.sync_broadcast("MEASURE_MAGNETIZATION")
                mag_sum = sum(res["data"] for res in mag_res)
                
                cross_corr = 0.0
                edge_count = 0
                for pA in range(4):
                    for qA, (pB, qB) in self.boundary_map[pA].items():
                        if pA < pB: 
                            cross_corr += patch_z_exp[pA][qA] * patch_z_exp[pB][qB]
                            edge_count += 1
                            
                avg_corr = cross_corr / edge_count if edge_count > 0 else 0.0

                print(f"Step {t:03d} | Bulk Mag: {mag_sum:+.4f} | Boundary <Z_A Z_B> [pre-kick]: {avg_corr:+.4f} | Kicks: {sum(len(k['kicks']) for k in kick_payloads)}")

    def shutdown(self):
        print("\nCollapsing the Wormhole (Shutting down GPU workers)...")
        for pipe in self.pipes:
            try:
                pipe.send({"action": "SHUTDOWN"})
            except:
                pass
        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive(): p.terminate()

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    AVAILABLE_GPUS = [0, 0, 0, 0] 
    
    wormhole_engine = TraversableWormholeEngine(device_ids=AVAILABLE_GPUS)

    try:
        wormhole_engine.evolve(
            total_time_steps=50, 
            depth_per_step=3, 
            coupling_strength=0.15 
        )

    finally:
        wormhole_engine.shutdown()
