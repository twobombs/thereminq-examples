# -*- coding: us-ascii -*-
# Inspired by (1+1)D SU(2) LSH Dynamics (arXiv:2602.18080v1)

import os
import gc
import time
import signal
import collections
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
    elif topology == "FC":
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                edges.append((i, j))
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

                elif action == "EVOLVE_HADRONS":
                    # Simulating the (1+1)D SU(2) Loop-String-Hadron (LSH) proxy
                    # Reference: arXiv:2602.18080v1
                    J = cmd.get("J", 1.0)
                    hx = cmd.get("hx", 0.5)
                    hz = cmd.get("hz", 0.2)
                    dt = cmd.get("dt", 0.05)
                    steps = cmd.get("steps", 2)
                    
                    with gpu_lock:
                        for _ in range(steps):
                            # Upgraded to Strang Splitting (2nd-Order Lie-Trotter)
                            for q in range(num_qubits): apply_rx(sim, -hx * dt, q)
                            for q in range(num_qubits): apply_rz(sim, -hz * dt, q)
                            
                            for q1, q2 in intra_edges:
                                apply_cx(sim, q1, q2)
                                apply_rz(sim, -2.0 * J * dt, q2)
                                apply_cx(sim, q1, q2)
                                
                            for q in range(num_qubits): apply_rz(sim, -hz * dt, q)
                            for q in range(num_qubits): apply_rx(sim, -hx * dt, q)
                                
                    cmd_pipe.send({"status": "HADRONS_EVOLVED", "patch_idx": patch_idx})

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

                elif action == "MEASURE_DYNAMICS":
                    with gpu_lock:
                        total_z = sum((1.0 - 2.0 * sim.prob(q)) for q in range(num_qubits))
                    cmd_pipe.send({"status": "DYNAMICS_MEASURED", "patch_idx": patch_idx, "data": total_z / num_qubits})

                elif action == "COMPUTE_BENCHMARKS":
                    shots = cmd.get("shots", 2048) 
                    
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
# 3. HIERARCHICAL ORCHESTRATOR
# ==========================================
class HierarchicalHadronEngine:
    def __init__(self, device_ids: List[int], intra_topology: str = "RING", boundary_size: int = 4):
        self._is_shutdown = False
        self.device_ids = device_ids
        
        # Structural Hierarchy
        self.num_patches = 12
        self.patches_per_atom = 3
        
        assert self.num_patches % self.patches_per_atom == 0, \
            f"Error: {self.num_patches} patches not evenly divisible by {self.patches_per_atom}"
            
        self.num_atoms = self.num_patches // self.patches_per_atom
        
        # Atom Dictionary (e.g., Atom 0 contains patches [0, 1, 2])
        self.atoms = {i: list(range(i * self.patches_per_atom, (i + 1) * self.patches_per_atom)) 
                      for i in range(self.num_atoms)}
                      
        self.boundary_qubits = list(range(boundary_size))
        self.intra_patch_edges = get_intra_edges(num_qubits=25, topology=intra_topology)

        self.ctx = mp.get_context('spawn')
        self.gpu_locks = {gpu_id: self.ctx.Lock() for gpu_id in set(self.device_ids)}
        
        self.workers: List[mp.Process] = []
        self.pipes: List[Connection] = []

        print(f"Initializing {self.num_patches} LHV Hadrons ({self.num_atoms} Atoms | 1 Molecule)...")
        for p_idx in range(self.num_patches):
            parent_conn, child_conn = self.ctx.Pipe()
            assigned_gpu = self.device_ids[p_idx % len(self.device_ids)]
            
            p = self.ctx.Process(
                target=persistent_island_worker,
                args=(
                    assigned_gpu, 
                    p_idx, 
                    25, 
                    self.intra_patch_edges, 
                    self.boundary_qubits, 
                    child_conn,
                    self.gpu_locks[assigned_gpu]
                )
            )
            p.start()
            
            child_conn.close()
            
            self.workers.append(p)
            self.pipes.append(parent_conn)
            
        for i, pipe in enumerate(self.pipes):
            if pipe.poll(timeout=45.0): 
                msg = pipe.recv()
                if msg.get("status") != "READY": 
                    self.shutdown()
                    raise RuntimeError(f"Worker {i} error: {msg.get('msg')}")
            else: 
                self.shutdown()
                raise TimeoutError(f"Worker {i} timed out.")

    def sync_broadcast(self, action: str, kwargs_list: Optional[List[Dict]] = None, timeout_secs: float = 300.0) -> Dict[int, Any]:
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
                raise TimeoutError("Workers timed out.")
                
            ready_pipes = wait([p for _, p in pending], timeout=timeout)
            if not ready_pipes: 
                continue
                
            still_pending = []
            for idx, pipe in pending:
                if pipe in ready_pipes:
                    try:
                        res = pipe.recv()
                        if res.get("status") == "ERROR": 
                            raise RuntimeError(f"Worker {idx} error: {res.get('msg')}")
                        results[idx] = res
                    except (EOFError, OSError): 
                        raise RuntimeError(f"Worker {idx} connection crashed.")
                else: 
                    still_pending.append((idx, pipe))
            pending = still_pending
        return results

    def evolve(self, total_time_steps: int, dt: float, steps_per_dt: int, g_atom: float, g_mol: float, depth_per_step: int = 1, drop_rate: float = 0.0, hadron_hz: float = 0.2) -> None:
        print(f"\nStarting Hierarchical Propagation (FC/CG)...")
        print(f"Steps: {total_time_steps} | RCS Depth: {depth_per_step} | Drop Rate: {drop_rate*100}%")
        print(f"Atom FC Coupling: {g_atom} | Mol CG Coupling: {g_mol}")
        print(f"Hadron Physics (1D LSH): {steps_per_dt} steps per dt ({dt}) -> J=1.0, hx=0.5, hz={hadron_hz}\n")
        
        main_rng = np.random.default_rng()
        
        for t in range(total_time_steps):
            
            # 1. Random Circuit Scrambling
            if depth_per_step > 0:
                seeds = main_rng.integers(0, 2**32, size=self.num_patches)
                self.sync_broadcast("RCS_CHUNK", [
                    {"seed": int(seeds[i]), "depth": depth_per_step, "drop_rate": drop_rate} 
                    for i in range(self.num_patches)
                ])

            # 2. Hadron Dynamics (1D LSH Proxy)
            if steps_per_dt > 0:
                self.sync_broadcast("EVOLVE_HADRONS", [{"J": 1.0, "hx": 0.5, "hz": hadron_hz, "dt": dt, "steps": steps_per_dt}] * self.num_patches)
            
            # 3. Extract Boundary Signatures
            xyz_results = self.sync_broadcast("MEASURE_BOUNDARY_BLOCH")
            patch_bloch = {p_idx: res["data"] for p_idx, res in xyz_results.items()}

            kick_payloads = [{"kicks": {}} for _ in range(self.num_patches)]
            atom_cg_vectors = {}
            
            # 4A. Coarse-Grain (CG) Calculation for Molecular Mapping
            for atom_idx, patch_list in self.atoms.items():
                cg_vec = np.zeros(3)
                count = 0
                for p in patch_list:
                    for b, v in patch_bloch[p].items():
                        cg_vec += np.array(v)
                        count += 1
                atom_cg_vectors[atom_idx] = cg_vec / max(1, count)
                
            # 4B. Distribute Hierarchical Kicks
            for atom_idx, patch_list in self.atoms.items():
                neighbor_atom = (atom_idx + 1) % self.num_atoms
                mol_cg_vec = atom_cg_vectors[neighbor_atom]
                
                for pA in patch_list:
                    for bA in self.boundary_qubits:
                        kx, ky, kz = 0.0, 0.0, 0.0
                        
                        # --- ATOM LEVEL: Fully-Connected (FC) ---
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
                        kx += g_mol * mol_cg_vec[0]
                        ky += g_mol * mol_cg_vec[1]
                        kz += g_mol * mol_cg_vec[2]
                        
                        kick_payloads[pA]["kicks"][bA] = (kx, ky, kz)
                        
            self.sync_broadcast("APPLY_LHV_KICKS", kick_payloads)
            
            # 5. Measure Hadronic Breathing Modes (Proxy)
            dyn_res = self.sync_broadcast("MEASURE_DYNAMICS")
            avg_mag = np.mean([r["data"] for r in dyn_res.values()])
            cg_str = ", ".join(f"{v:+.2f}" for v in atom_cg_vectors[0])
            print(f"Step {t:03d} | Avg String Density: {avg_mag:+.4f} | Atom 0 CG Vector: [{cg_str}]")

            if t == total_time_steps - 1:
                print(f"         +-- Calculating Final Benchmarks...")
                
                bench_res = {}
                for p_idx, pipe in enumerate(self.pipes):
                    pipe.send({"action": "COMPUTE_BENCHMARKS", "shots": 2048})
                    
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

    def shutdown(self) -> None:
        if self._is_shutdown: return
        self._is_shutdown = True
        print("\nShutting down Hierarchical Engine...")
        
        for pipe in getattr(self, 'pipes', []):
            try:
                if not pipe.closed:
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
                if not pipe.closed: 
                    pipe.close()
            except (OSError, BrokenPipeError, EOFError): 
                pass

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    mp.freeze_support()
    
    gpu_env = os.environ.get("WORMHOLE_GPUS", "0,1") 
    base_gpus = [int(g.strip()) for g in gpu_env.split(',')]
    
    # 12 Patches -> 4 Atoms -> 1 Molecule
    num_patches = 12
    AVAILABLE_GPUS = [base_gpus[i % len(base_gpus)] for i in range(num_patches)]
    
    # "RING" naturally sets up the 1D periodic boundary spatial lattice
    engine = HierarchicalHadronEngine(device_ids=AVAILABLE_GPUS, intra_topology="RING", boundary_size=4)

    try:
        engine.evolve(
            total_time_steps=20, 
            dt=0.05,
            steps_per_dt=2,
            g_atom=0.25,  
            g_mol=0.20,   
            depth_per_step=1,
            drop_rate=0.8,
            hadron_hz=0.8 
        )
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    finally:
        engine.shutdown()
