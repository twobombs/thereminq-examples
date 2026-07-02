# -*- coding: us-ascii -*-
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
    """
    Generates internal edges for a 3D cubic lattice and identifies the qubits 
    living on the 6 boundary faces for LHV stitching.
    Max qubits: lx * ly * lz <= 25
    """
    edges = []
    boundaries = {"+X": [], "-X": [], "+Y": [], "-Y": [], "+Z": [], "-Z": []}
    
    for x in range(lx):
        for y in range(ly):
            for z in range(lz):
                idx = x * (ly * lz) + y * lz + z
                
                # 3D Nearest Neighbor Links
                if x < lx - 1: edges.append((idx, (x + 1) * (ly * lz) + y * lz + z))
                if y < ly - 1: edges.append((idx, x * (ly * lz) + (y + 1) * lz + z))
                if z < lz - 1: edges.append((idx, x * (ly * lz) + y * lz + (z + 1)))
                
                # Identify Surface Qubits
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
    cmd_pipe: Connection
) -> None:
    
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    
    sim = None
    try:
        os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(device_id)
        from pyqrack import QrackSimulator
        sim = QrackSimulator(qubit_count=num_qubits)
        
        # Initialize 3D Vacuum State Superposition
        for q in range(num_qubits):
            apply_h(sim, q)
            
        cmd_pipe.send({"status": "READY", "patch_idx": patch_idx})
        
        # Flatten all boundary qubits for efficient measurement
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
                        # 3D Flux Tube ZZ Couplings
                        for q1, q2 in intra_edges:
                            apply_cx(sim, q1, q2)
                            apply_rz(sim, -2.0 * J * dt, q2)
                            apply_cx(sim, q1, q2)
                            
                    cmd_pipe.send({"status": "HADRONS_EVOLVED", "patch_idx": patch_idx})

                elif action == "MEASURE_BOUNDARY_BLOCH":
                    bloch_vectors = {}
                    for q in all_boundary_qubits:
                        z_exp = 1.0 - 2.0 * sim.prob(q)
                        apply_h(sim, q)
                        x_exp = 1.0 - 2.0 * sim.prob(q)
                        apply_h(sim, q)
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

                elif action == "MEASURE_DYNAMICS":
                    total_z = sum((1.0 - 2.0 * sim.prob(q)) for q in range(num_qubits))
                    cmd_pipe.send({"status": "DYNAMICS_MEASURED", "patch_idx": patch_idx, "data": total_z / num_qubits})

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
    def __init__(self, device_ids: List[int]):
        self._is_shutdown = False
        self.device_ids = device_ids
        
        # 3D Spatial Geometry: A 3x3x2 sub-volume per patch (18 qubits)
        self.lx, self.ly, self.lz = 3, 3, 2
        self.qubits_per_patch = self.lx * self.ly * self.lz
        self.intra_edges, self.boundaries = generate_3d_subvolume(self.lx, self.ly, self.lz)
        
        # Macroscopic Grid: Arrange 8 patches into a 2x2x2 "Atom"
        self.num_patches = 8 
        
        # Mapping spatial coordinates of patches in the 2x2x2 macroscopic grid
        self.patch_coords = {
            0: (0,0,0), 1: (1,0,0), 2: (0,1,0), 3: (1,1,0),
            4: (0,0,1), 5: (1,0,1), 6: (0,1,1), 7: (1,1,1)
        }

        self.ctx = mp.get_context('spawn')
        self.workers: List[mp.Process] = []
        self.pipes: List[Connection] = []

        print(f"Initializing {self.num_patches} LHV Patches ({self.qubits_per_patch} qubits/patch)...")
        print(f"Macroscopic Lattice: 2x2x2 patches -> Total Volume: {self.lx*2}x{self.ly*2}x{self.lz*2}")
        
        for p_idx in range(self.num_patches):
            parent_conn, child_conn = self.ctx.Pipe()
            p = self.ctx.Process(
                target=persistent_island_worker_3d,
                args=(self.device_ids[p_idx % len(self.device_ids)], p_idx, 
                      self.qubits_per_patch, self.intra_edges, self.boundaries, child_conn)
            )
            p.start()
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
            results[idx] = pipe.recv()
        return results

    def evolve(self, total_steps: int, dt: float, g_face: float):
        print(f"\nStarting 3D Spatial Propagation | Face-to-Face Coupling: {g_face}\n")
        
        for t in range(total_steps):
            # 1. Internal 3D Dynamics
            self.sync_broadcast("EVOLVE_HADRONS", [{"J": 1.0, "hx": 0.5, "hz": 0.2, "dt": dt, "steps": 2}] * self.num_patches)
            
            # 2. Extract 2D Surface States
            patch_bloch = {p: res["data"] for p, res in self.sync_broadcast("MEASURE_BOUNDARY_BLOCH").items()}
            kick_payloads = [{"kicks": {}} for _ in range(self.num_patches)]
            
            # 3. 3D Face-to-Face Stitching (Macroscopic Routing)
            for p1, coord1 in self.patch_coords.items():
                x1, y1, z1 = coord1
                
                # Check neighbors in 6 directions
                neighbors = {
                    "+X": (x1+1, y1, z1), "-X": (x1-1, y1, z1),
                    "+Y": (x1, y1+1, z1), "-Y": (x1, y1-1, z1),
                    "+Z": (x1, y1, z1+1), "-Z": (x1, y1, z1-1)
                }
                
                for dir1, coord2 in neighbors.items():
                    # Find which patch is at coord2
                    p2 = next((k for k, v in self.patch_coords.items() if v == coord2), None)
                    if p2 is None: continue # Open macroscopic boundary
                    
                    # Map the opposing face (e.g., +X of p1 connects to -X of p2)
                    dir2 = dir1.replace("+", "temp").replace("-", "+").replace("temp", "-")
                    
                    face1_qubits = self.boundaries[dir1]
                    face2_qubits = self.boundaries[dir2]
                    
                    # Apply Mean-Field kick from opposing face to local face
                    # Averaging the state of the neighbor's face to apply as a uniform field
                    avg_x, avg_y, avg_z = 0.0, 0.0, 0.0
                    for q2 in face2_qubits:
                        v2 = patch_bloch[p2][q2]
                        avg_x += v2[0]; avg_y += v2[1]; avg_z += v2[2]
                        
                    n2 = max(1, len(face2_qubits))
                    avg_x /= n2; avg_y /= n2; avg_z /= n2
                    
                    for q1 in face1_qubits:
                        # Add kick to existing kicks (since corner qubits belong to multiple faces)
                        curr_k = kick_payloads[p1]["kicks"].get(q1, (0.0, 0.0, 0.0))
                        kick_payloads[p1]["kicks"][q1] = (
                            curr_k[0] + g_face * avg_x,
                            curr_k[1] + g_face * avg_y,
                            curr_k[2] + g_face * avg_z
                        )

            self.sync_broadcast("APPLY_LHV_KICKS", kick_payloads)
            
            # 4. Measure 3D Bulk Density
            dyn_res = self.sync_broadcast("MEASURE_DYNAMICS")
            avg_mag = np.mean([r["data"] for r in dyn_res.values()])
            print(f"Step {t:03d} | Macroscopic 3D Bulk Density: {avg_mag:+.4f}")

    def shutdown(self) -> None:
        if self._is_shutdown: return
        self._is_shutdown = True
        for pipe in getattr(self, 'pipes', []):
            try: pipe.send({"action": "SHUTDOWN"})
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
    
    # Simulating on 1 to 4 GPUs
    gpu_env = os.environ.get("WORMHOLE_GPUS", "0,1,2,3") 
    base_gpus = [int(g.strip()) for g in gpu_env.split(',')]
    
    # 8 patches distributed across available GPUs
    AVAILABLE_GPUS = [base_gpus[i % len(base_gpus)] for i in range(8)]
    
    engine = HierarchicalHadronEngine3D(device_ids=AVAILABLE_GPUS)

    try:
        engine.evolve(total_steps=20, dt=0.05, g_face=0.15)
    except KeyboardInterrupt:
        pass
    finally:
        engine.shutdown()
