# -*- coding: us-ascii -*-
import os
import gc
import time
import signal
import numpy as np
import multiprocessing as mp
from multiprocessing.connection import Connection, wait
from typing import List, Tuple, Dict, Any

# ==========================================
# 0. PYQRACK API SAFEGUARDS & GATES
# ==========================================
PX, PY, PZ = 1, 2, 3

def apply_h(sim: Any, q: int) -> None:
    if hasattr(sim, 'h'): sim.h(q)
    else: sim.mtrx([complex(1/np.sqrt(2), 0), complex(1/np.sqrt(2), 0), complex(1/np.sqrt(2), 0), complex(-1/np.sqrt(2), 0)], [q])

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
    elif hasattr(sim, 'mcx'):
        try: sim.mcx([c], t)
        except TypeError: sim.mcx([c], [t])
    else: raise RuntimeError("No CX gate available")

# ==========================================
# 1. LATTICE TOPOLOGY
# ==========================================
def get_intra_edges(num_qubits: int = 25, topology: str = "RING") -> List[Tuple[int, int]]:
    """Generates the internal (1+1)D spatial lattice for meson propagation."""
    edges = []
    if topology == "RING":
        for i in range(num_qubits):
            edges.append((i, (i + 1) % num_qubits))
    elif topology == "LINE":
        for i in range(num_qubits - 1):
            edges.append((i, i + 1))
    return edges

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
        
        # Initialize to uniform superposition
        for q in range(num_qubits):
            apply_h(sim, q)
            
        cmd_pipe.send({"status": "READY", "patch_idx": patch_idx})
        rotation_gates = [apply_rx, apply_ry, apply_rz]

        while True:
            try:
                if cmd_pipe.poll(timeout=0.1): cmd = cmd_pipe.recv()
                else: continue
            except (EOFError, OSError, BrokenPipeError):
                break
                
            action = cmd.get("action")
            if action == "SHUTDOWN": break

            try:
                if action == "EVOLVE_HADRONS":
                    # Simulating the SU(2) LSH String-Breaking Proxy
                    J = cmd.get("J", 1.0)
                    hx = cmd.get("hx", 0.5)
                    hz = cmd.get("hz", 0.2)
                    dt = cmd.get("dt", 0.05)
                    steps = cmd.get("steps", 2)
                    
                    for _ in range(steps):
                        # Transverse Field (Kinetic / Pair creation)
                        for q in range(num_qubits): rotation_gates[0](sim, -2.0 * hx * dt, q)
                        # Longitudinal Field (Mass / String tension linear potential)
                        for q in range(num_qubits): rotation_gates[2](sim, -2.0 * hz * dt, q)
                        # ZZ Coupling (Flux tubes / String binding)
                        for q1, q2 in intra_edges:
                            apply_cx(sim, q1, q2)
                            rotation_gates[2](sim, -2.0 * J * dt, q2)
                            apply_cx(sim, q1, q2)
                            
                    cmd_pipe.send({"status": "HADRONS_EVOLVED", "patch_idx": patch_idx})

                elif action == "MEASURE_BOUNDARY_BLOCH":
                    bloch_vectors = {}
                    for q in boundary_qubits:
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
                    # Tracks the macroscopic breathing modes of the mesonic string
                    total_z = sum((1.0 - 2.0 * sim.prob(q)) for q in range(num_qubits))
                    cmd_pipe.send({"status": "DYNAMICS_MEASURED", "patch_idx": patch_idx, "data": total_z / num_qubits})

            except Exception as inner_e:
                try: cmd_pipe.send({"status": "ERROR", "msg": str(inner_e)})
                except (EOFError, OSError, BrokenPipeError): break
                
    except (EOFError, OSError, BrokenPipeError): pass
    finally:
        if sim is not None: del sim
        gc.collect()

# ==========================================
# 3. HIERARCHICAL ORCHESTRATOR
# ==========================================
class HierarchicalHadronEngine:
    def __init__(self, device_ids: List[int], boundary_size: int = 4):
        self._is_shutdown = False
        self.device_ids = device_ids
        
        # Structural Hierarchy
        self.num_patches = 12
        self.patches_per_atom = 3
        self.num_atoms = self.num_patches // self.patches_per_atom
        
        # Atom Dictionary (e.g., Atom 0 contains patches [0, 1, 2])
        self.atoms = {i: list(range(i * self.patches_per_atom, (i + 1) * self.patches_per_atom)) 
                      for i in range(self.num_atoms)}
                      
        self.boundary_qubits = list(range(boundary_size))
        self.intra_patch_edges = get_intra_edges(num_qubits=25, topology="RING")

        self.ctx = mp.get_context('spawn')
        self.workers, self.pipes = [], []

        print(f"Initializing {self.num_patches} LHV Hadrons ({self.num_atoms} Atoms | 1 Molecule)...")
        for p_idx in range(self.num_patches):
            parent_conn, child_conn = self.ctx.Pipe()
            p = self.ctx.Process(
                target=persistent_island_worker,
                args=(
                    self.device_ids[p_idx % len(self.device_ids)], 
                    p_idx, 25, self.intra_patch_edges, self.boundary_qubits, child_conn
                )
            )
            p.start()
            self.workers.append(p)
            self.pipes.append(parent_conn)
            
        for i, pipe in enumerate(self.pipes):
            if pipe.poll(timeout=45.0): 
                msg = pipe.recv()
                if msg.get("status") != "READY": raise RuntimeError(f"Worker {i} error.")
            else: raise TimeoutError(f"Worker {i} timed out.")

    def sync_broadcast(self, action: str, kwargs_list: List[Dict] = None, timeout_secs: float = 300.0) -> Dict[int, Any]:
        if kwargs_list is None: kwargs_list = [{}] * self.num_patches
        for i, pipe in enumerate(self.pipes):
            payload = {"action": action}
            payload.update(kwargs_list[i])
            pipe.send(payload)
            
        results = {}
        pending = list(enumerate(self.pipes)) 
        deadline = time.monotonic() + timeout_secs
        
        while pending:
            timeout = deadline - time.monotonic()
            if timeout <= 0: raise TimeoutError("Workers timed out.")
                
            ready_pipes = wait([p for _, p in pending], timeout=timeout)
            if not ready_pipes: continue
                
            still_pending = []
            for idx, pipe in pending:
                if pipe in ready_pipes:
                    try:
                        res = pipe.recv()
                        if res.get("status") == "ERROR": raise RuntimeError(f"Worker {idx} error: {res.get('msg')}")
                        results[idx] = res
                    except (EOFError, OSError): raise RuntimeError(f"Worker {idx} connection crashed.")
                else: still_pending.append((idx, pipe))
            pending = still_pending
        return results

    def evolve(self, total_time_steps: int, dt: float, steps_per_dt: int, g_atom: float, g_mol: float):
        print(f"\nStarting Hierarchical Propagation (FC/CG)...")
        print(f"Steps: {total_time_steps} | Atom FC Coupling: {g_atom} | Mol CG Coupling: {g_mol}\n")
        
        for t in range(total_time_steps):
            # 1. Hadron Dynamics (1D LSH Proxy)
            self.sync_broadcast("EVOLVE_HADRONS", [{"J": 1.0, "hx": 0.5, "hz": 0.2, "dt": dt, "steps": steps_per_dt}] * self.num_patches)
            
            # 2. Extract Boundary Signatures
            xyz_results = self.sync_broadcast("MEASURE_BOUNDARY_BLOCH")
            patch_bloch = {p_idx: res["data"] for p_idx, res in xyz_results.items()}

            kick_payloads = [{"kicks": {}} for _ in range(self.num_patches)]
            atom_cg_vectors = {}
            
            # 3A. Coarse-Grain (CG) Calculation for Molecular Mapping
            for atom_idx, patch_list in self.atoms.items():
                cg_vec = np.zeros(3)
                count = 0
                for p in patch_list:
                    for b, v in patch_bloch[p].items():
                        cg_vec += np.array(v)
                        count += 1
                atom_cg_vectors[atom_idx] = cg_vec / max(1, count)
                
            # 3B. Distribute Hierarchical Kicks
            for atom_idx, patch_list in self.atoms.items():
                neighbor_atom = (atom_idx + 1) % self.num_atoms
                mol_cg_vec = atom_cg_vectors[neighbor_atom]
                
                for pA in patch_list:
                    for bA in self.boundary_qubits:
                        kx, ky, kz = 0.0, 0.0, 0.0
                        
                        # --- ATOM LEVEL: Fully-Connected (FC) ---
                        # Patch A binds tightly to all boundaries of other patches inside the Atom
                        fc_neighbors = 0
                        for pB in patch_list:
                            if pA == pB: continue
                            for bB, vB in patch_bloch[pB].items():
                                kx += g_atom * vB[0]
                                ky += g_atom * vB[1]
                                kz += g_atom * vB[2]
                                fc_neighbors += 1
                                
                        if fc_neighbors > 0:
                            kx /= np.sqrt(fc_neighbors)
                            ky /= np.sqrt(fc_neighbors)
                            kz /= np.sqrt(fc_neighbors)
                            
                        # --- MOLECULE LEVEL: Coarse-Grained (CG) ---
                        # The patch interacts with the averaged mean-field state of the neighboring Atom
                        kx += g_mol * mol_cg_vec[0]
                        ky += g_mol * mol_cg_vec[1]
                        kz += g_mol * mol_cg_vec[2]
                        
                        kick_payloads[pA]["kicks"][bA] = (kx, ky, kz)
                        
            self.sync_broadcast("APPLY_LHV_KICKS", kick_payloads)
            
            # 4. Measure Hadronic Breathing Modes (Proxy)
            dyn_res = self.sync_broadcast("MEASURE_DYNAMICS")
            avg_mag = np.mean([r["data"] for r in dyn_res.values()])
            cg_str = ", ".join(f"{v:+.2f}" for v in atom_cg_vectors[0])
            print(f"Step {t:03d} | Avg String Density: {avg_mag:+.4f} | Atom 0 CG Vector: [{cg_str}]")

    def shutdown(self):
        if self._is_shutdown: return
        self._is_shutdown = True
        print("\nShutting down Hierarchical Engine...")
        for pipe in getattr(self, 'pipes', []):
            try:
                if not pipe.closed: pipe.send({"action": "SHUTDOWN"})
            except: pass
        for p in getattr(self, 'workers', []):
            try:
                p.join(timeout=5)
                if p.is_alive(): p.terminate()
            except: pass

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    gpu_env = os.environ.get("WORMHOLE_GPUS", "0,1,2,3,4,5") 
    base_gpus = [int(g.strip()) for g in gpu_env.split(',')]
    
    # Using 12 Patches -> 4 Atoms -> 1 Molecule
    num_patches = 12
    patches_per_gpu = max(1, num_patches // len(base_gpus))
    AVAILABLE_GPUS = [base_gpus[(i // patches_per_gpu) % len(base_gpus)] for i in range(num_patches)]
    
    engine = HierarchicalHadronEngine(device_ids=AVAILABLE_GPUS, boundary_size=4)

    try:
        engine.evolve(
            total_time_steps=20, 
            dt=0.05,
            steps_per_dt=2,
            g_atom=0.25,  # Strong local binding (Fully-Connected)
            g_mol=0.05    # Weak structural binding (Coarse-Grained)
        )
    finally:
        engine.shutdown()
