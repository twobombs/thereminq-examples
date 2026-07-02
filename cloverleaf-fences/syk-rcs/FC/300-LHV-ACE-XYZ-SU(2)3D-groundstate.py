import os
import gc
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
# 1. 3D LATTICE TOPOLOGY (SUB-VOLUME)
# ==========================================
def generate_3d_subvolume(lx: int = 3, ly: int = 3, lz: int = 2) -> Tuple[List[Tuple[int, int]], Dict[str, List[int]]]:
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
                
    return edges, boundaries

# ==========================================
# 2. 3D ISOLATED LHV ISLAND (GPU WORKER)
# ==========================================
def persistent_island_worker_3d(
    device_id: int, 
    patch_idx: int, 
    num_qubits: int, 
    intra_edges: List[Tuple[int, int]], 
    boundaries: Dict[str, List[int]],
    cmd_pipe: Connection,
    seed: int
) -> None:
    
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    
    sim = None
    try:
        os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(device_id)
        from pyqrack import QrackSimulator
        sim = QrackSimulator(qubit_count=num_qubits)
        
        np.random.seed(seed)
        
        for q in range(num_qubits):
            apply_h(sim, q)
            apply_rx(sim, np.random.normal(0, 1e-5), q)
            apply_rz(sim, np.random.normal(0, 1e-5), q)
            
        cmd_pipe.send({"status": "READY", "patch_idx": patch_idx})
        all_boundary_qubits = list(set([q for face in boundaries.values() for q in face]))

        while True:
            if not cmd_pipe.poll(timeout=0.1): continue
            cmd = cmd_pipe.recv()
            action = cmd.get("action")
            
            if action == "SHUTDOWN": break

            try:
                if action == "EVOLVE_HADRONS":
                    J, hx, hz = cmd.get("J", 1.0), cmd.get("hx", 0.5), cmd.get("hz", 0.2)
                    dt, steps = cmd.get("dt", 0.05), cmd.get("steps", 2)
                    
                    for _ in range(steps):
                        for q in range(num_qubits): apply_rx(sim, -2.0 * hx * dt, q)
                        for q in range(num_qubits): apply_rz(sim, -2.0 * hz * dt, q)
                        for q1, q2 in intra_edges:
                            apply_cx(sim, q1, q2)
                            apply_rz(sim, -2.0 * J * dt, q2)
                            apply_cx(sim, q1, q2)
                            
                    cmd_pipe.send({"status": "HADRONS_EVOLVED", "patch_idx": patch_idx})

                elif action == "MEASURE_ENERGY":
                    J_val, hx_val, hz_val = cmd.get("J", 1.0), cmd.get("hx", 0.5), cmd.get("hz", 0.2)
                    local_energy = 0.0
                    
                    # N2 LIMITATION: This per-qubit cloning strategy introduces O(E + 2N) memory 
                    # overhead per step. It is viable up to ~22 qubits/patch on standard hardware.
                    # Scaling further requires staged measurement or native backend Pauli expectation tracking.
                    for q in range(num_qubits):
                        clone_x = sim.clone()
                        apply_h(clone_x, q)
                        local_energy += -hx_val * (1.0 - 2.0 * clone_x.prob(q))
                        del clone_x
                        
                        clone_z = sim.clone()
                        local_energy += -hz_val * (1.0 - 2.0 * clone_z.prob(q))
                        del clone_z
                        
                    for q1, q2 in intra_edges:
                        # N1 LIMITATION: The CNOT proxy method below is mathematically approximate 
                        # for highly entangled states. It provides a heuristic <ZZ> expectation but 
                        # lacks the precision of full state tomography or pauli_expectation_eigenvalues.
                        clone_zz = sim.clone()
                        apply_cx(clone_zz, q1, q2)
                        local_energy += -J_val * (1.0 - 2.0 * clone_zz.prob(q2))
                        del clone_zz
                        
                    cmd_pipe.send({"status": "ENERGY_MEASURED", "patch_idx": patch_idx, "data": local_energy})

                elif action == "MEASURE_BOUNDARY_BLOCH":
                    bloch_vectors = {}
                    for q in all_boundary_qubits:
                        clone_z = sim.clone()
                        z_exp = 1.0 - 2.0 * clone_z.prob(q)
                        del clone_z
                        
                        clone_x = sim.clone()
                        apply_h(clone_x, q)
                        x_exp = 1.0 - 2.0 * clone_x.prob(q)
                        del clone_x
                        
                        clone_y = sim.clone()
                        apply_rx(clone_y, -np.pi/2, q) 
                        y_exp = 1.0 - 2.0 * clone_y.prob(q)
                        del clone_y
                        
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

            except Exception as e:
                cmd_pipe.send({"status": "ERROR", "msg": str(e)})
                
    except (EOFError, OSError): pass
    finally:
        if sim is not None: del sim
        gc.collect()

# ==========================================
# 3. 3D HIERARCHICAL ORCHESTRATOR
# ==========================================
class HierarchicalHadronEngine3D:
    def __init__(self, device_ids: List[int], grid: Tuple[int, int, int] = (2, 2, 2), master_seed: int = 42):
        self._is_shutdown = False
        self.device_ids = device_ids
        
        self.grid_x, self.grid_y, self.grid_z = grid
        self.num_patches = self.grid_x * self.grid_y * self.grid_z
        
        self.lx, self.ly, self.lz = 3, 3, 2
        self.qubits_per_patch = self.lx * self.ly * self.lz
        self.intra_edges, self.boundaries = generate_3d_subvolume(self.lx, self.ly, self.lz)
        
        self.patch_coords = {}
        idx = 0
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                for z in range(self.grid_z):
                    self.patch_coords[idx] = (x, y, z)
                    idx += 1

        self.coord_to_patch = {v: k for k, v in self.patch_coords.items()}

        self.ctx = mp.get_context('spawn')
        self.workers: List[mp.Process] = []
        self.pipes: List[Connection] = []

        total_sites = self.num_patches * self.qubits_per_patch
        print(f"Initializing 3D Minimum Energy Setup ({self.num_patches} patches, {self.qubits_per_patch} qubits/patch -> {total_sites} total qubits)...")
        
        np.random.seed(master_seed)
        patch_seeds = np.random.randint(0, 2**31 - 1, size=self.num_patches)
        
        for p_idx in range(self.num_patches):
            parent_conn, child_conn = self.ctx.Pipe()
            p = self.ctx.Process(
                target=persistent_island_worker_3d,
                args=(self.device_ids[p_idx % len(self.device_ids)], p_idx, 
                      self.qubits_per_patch, self.intra_edges, self.boundaries, child_conn, int(patch_seeds[p_idx]))
            )
            p.start()
            child_conn.close() 
            
            self.workers.append(p)
            self.pipes.append(parent_conn)
            
        for i, pipe in enumerate(self.pipes):
            if pipe.poll(timeout=45.0):
                if pipe.recv().get("status") != "READY": raise RuntimeError("Worker error.")
            else: raise TimeoutError("Worker timeout.")

    def sync_broadcast(self, action: str, kwargs_list: Optional[List[Dict]] = None) -> Dict[int, Any]:
        if kwargs_list is None: kwargs_list = [{}] * self.num_patches
        for i, pipe in enumerate(self.pipes): pipe.send({"action": action, **kwargs_list[i]})
            
        results = {}
        for idx, pipe in enumerate(self.pipes):
            if not pipe.poll(timeout=30.0):
                raise TimeoutError(f"Worker {idx} timed out on action '{action}'")
            
            msg = pipe.recv()
            if msg.get("status") == "ERROR":
                raise RuntimeError(f"Worker {idx} execution failed: {msg.get('msg')}")
                
            results[idx] = msg
        return results

    def anneal_to_ground_state(self, total_steps: int, dt: float, target_g_face: float, target_J: float, target_hx: float, target_hz: float):
        print(f"Starting Adiabatic Anneal -> Target: J={target_J}, hx={target_hx}, hz={target_hz}\n")
        
        for t in range(total_steps):
            s = t / max(1, (total_steps - 1))
            current_hx = (1.0 - s) * 3.0 + s * target_hx
            current_J = s * target_J
            current_hz = s * target_hz
            current_g_face = s * target_g_face
            
            self.sync_broadcast("EVOLVE_HADRONS", [{"J": current_J, "hx": current_hx, "hz": current_hz, "dt": dt, "steps": 2}] * self.num_patches)
            
            patch_bloch = {p: res["data"] for p, res in self.sync_broadcast("MEASURE_BOUNDARY_BLOCH").items()}
            kick_payloads = [{"kicks": {}} for _ in range(self.num_patches)]
            macroscopic_boundary_energy = 0.0
            
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
                    
                    avg_x, avg_y, avg_z = 0.0, 0.0, 0.0
                    for q2 in face2_qubits:
                        v2 = patch_bloch[p2][q2]
                        avg_x += v2[0]; avg_y += v2[1]; avg_z += v2[2]
                        
                    n2 = max(1, len(face2_qubits))
                    avg_x /= n2; avg_y /= n2; avg_z /= n2
                    
                    interaction_E = 0.0
                    for q1 in face1_qubits:
                        # N3 Note: Kicks are accumulated unconditionally. This correctly ensures
                        # bilateral interaction, as both (p1->p2) and (p2->p1) apply fields to each other.
                        curr_k = kick_payloads[p1]["kicks"].get(q1, (0.0, 0.0, 0.0))
                        kick_payloads[p1]["kicks"][q1] = (
                            curr_k[0] + current_g_face * avg_x,
                            curr_k[1] + current_g_face * avg_y,
                            curr_k[2] + current_g_face * avg_z
                        )
                        v1 = patch_bloch[p1][q1]
                        interaction_E += -current_g_face * (v1[0]*avg_x + v1[1]*avg_y + v1[2]*avg_z)
                        
                    if p1 < p2:
                        macroscopic_boundary_energy += interaction_E

            if current_g_face > 0.0:
                self.sync_broadcast("APPLY_LHV_KICKS", kick_payloads)
            
            energy_res = self.sync_broadcast("MEASURE_ENERGY", [{"J": current_J, "hx": current_hx, "hz": current_hz}] * self.num_patches)
            bulk_energy = sum([r["data"] for r in energy_res.values()])
            
            total_energy = bulk_energy + macroscopic_boundary_energy
            print(f"Step {t:03d} | Anneal: {s*100:05.1f}% | Total Setup Potential Energy: {total_energy:+.4f}")

    def shutdown(self) -> None:
        if self._is_shutdown: return
        self._is_shutdown = True
        for pipe in getattr(self, 'pipes', []):
            try: 
                pipe.send({"action": "SHUTDOWN"})
                while pipe.poll(timeout=0.05): pipe.recv()
            except: pass
        for p in getattr(self, 'workers', []):
            try:
                p.join(timeout=2)
                if p.is_alive(): p.terminate()
            except: pass

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    mp.freeze_support()
    
    gpu_env = os.environ.get("WORMHOLE_GPUS", "0,1,2,3") 
    base_gpus = [int(g.strip()) for g in gpu_env.split(',')]
    
    target_grid = (2, 2, 2)
    total_requested_nodes = target_grid[0] * target_grid[1] * target_grid[2]
    
    AVAILABLE_GPUS = [base_gpus[i % len(base_gpus)] for i in range(total_requested_nodes)]
    
    engine = HierarchicalHadronEngine3D(
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
        pass
    finally:
        engine.shutdown()
