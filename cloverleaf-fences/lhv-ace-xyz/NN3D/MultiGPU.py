# -*- coding: us-ascii -*-
# 27-Qubit 3x3x3 Lattice & Macroscopic Grid Annealing
# High-Throughput Volumetric Engine with Statistical Variance Injection
#
# REVISION 58 - MULTI-GPU DISTRIBUTED IPC ARCHITECTURE (PROD GOLD MASTER)
#
# MULTI-GPU STRATEGY:
# - Partitions the total patches across independent worker processes.
# - Each worker is strictly pinned to a distinct OpenCL device ID.
# - PyQrack is explicitly imported *inside* the worker loop to guarantee isolation.
# - The master process orchestrates pipe-buffered IPC, aggregating boundary profiles,
#   calculating the macroscopic mean-field kicks, and scattering the payloads back.
#
# BUGFIXES & REFINEMENTS:
# - Updated orchestrator initialization log to explicitly calculate and display the 
#   total number of logical qubits across the entire distributed grid.
# - Corrected Y-axis sign probe to use native sim.s(0) instead of r(PZ, pi/2), 
#   ensuring preparation of |+i> rather than |-i> to prevent double-negative masking.
# - Hardened Dynamic Convention Autodetect: Isolated Z, X, and Y verification tracks 
#   into fresh, discrete simulator instances to eliminate mutation cross-contamination.
# - Added magnitude assertions (< 1e-4 deviation from unity) before sign evaluation.
# - Optimized observables to use PyQrack's native pauli_expectation.
# - Disabled QUnit Fidelity Guard to bypass a fatal PyQrack v2.0.0 C++ bounds index bug.
# - Trotter step explicitly calculates dt_half for Strang Splitting mathematical clarity.

import os
import sys
import gc
import csv
import json
import time
import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Dict, Any

# --- GLOBAL CONFIGURATION ---
GRID_X, GRID_Y, GRID_Z = 4, 4, 3
TOTAL_PATCHES = GRID_X * GRID_Y * GRID_Z
QUBITS_PER_PATCH = 27

# Set this to the number of distinct GPU devices you want to target.
WORKER_GPUS = 6 

# =====================================================================
# ENVIRONMENT - set before pyqrack import
# =====================================================================
# DISABLED FIDELITY GUARD to prevent QUnit::Prob C++ bounds corruption
os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"
os.environ["QRACK_QUNIT_SEPARABILITY_THRESHOLD"] = "1e-7"

# =====================================================================
# PURE FUNCTIONS (Math & Topology - independent of Qrack context)
# =====================================================================
def generate_27q_lattice_subvolume() -> Tuple[List[Tuple[int, int]], Dict[str, List[int]]]:
    lx, ly, lz = 3, 3, 3
    edges: List[Tuple[int, int]] = []
    boundaries: Dict[str, List[int]] = {
        "+X": [], "-X": [], "+Y": [], "-Y": [], "+Z": [], "-Z": []
    }
    
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

# =====================================================================
# WORKER PROCESS LOGIC
# =====================================================================
def gpu_worker_process(rank: int, assigned_patches: List[int], conn: mp.connection.Connection, dt: float, total_steps: int, target_J: float, target_hx: float, target_hz: float, measure_every: int):
    """
    Isolated worker process. Env vars must be set BEFORE importing PyQrack 
    to ensure the C++ backend binds to the correct OpenCL device.
    """
    os.environ["PYQRACK_SHARED_LIB_PATH"] = "/usr/local/lib/qrack/libqrack_pinvoke.so"
    os.environ["OCL_ICD_PLATFORM_SORT"] = "none"
    os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(rank)
    os.environ["QRACK_QPAGER_DEVICES"] = "-1"
    os.environ["QRACK_QUNITMULTI_DEVICES"] = "-1"
    os.environ["QRACK_FPPOW"] = "5"
    os.environ["QRACK_MAX_ALLOC_MB"] = "64000"
    os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"
    
    import pyqrack
    from pyqrack import QrackSimulator
    
    PX, PY, PZ = 1, 2, 3
    
    # Declare sims early to prevent NameError in finally block on catastrophic startup failure
    sims = {}
    
    try:
        # --- HARDENED SIGN CONVENTION AUTODETECT ---
        # Fresh, isolated engines ensure zero mutation leak. Magnitude checks
        # guard against underflow / rounding corruption prior to parsing signs.
        
        # 1. Z-axis: Fresh qubit is |0>. Standard physics expectation <Z> = +1.0.
        _sim_z = QrackSimulator(qubit_count=1)
        qrack_z = _sim_z.pauli_expectation([0], [PZ])
        assert abs(abs(qrack_z) - 1.0) < 1e-4, f"Fatal: PZ probe returned invalid magnitude: {qrack_z}"
        SIGN_Z = 1.0 if qrack_z > 0 else -1.0
        del _sim_z
        
        # 2. X-axis: Apply H to get |+>. Standard physics expectation <X> = +1.0.
        _sim_x = QrackSimulator(qubit_count=1)
        _sim_x.h(0)
        qrack_x = _sim_x.pauli_expectation([0], [PX])
        assert abs(abs(qrack_x) - 1.0) < 1e-4, f"Fatal: PX probe returned invalid magnitude: {qrack_x}"
        SIGN_X = 1.0 if qrack_x > 0 else -1.0
        del _sim_x
        
        # 3. Y-axis: Apply H then S to get |+i>. Standard physics expectation <Y> = +1.0.
        _sim_y = QrackSimulator(qubit_count=1)
        _sim_y.h(0)
        _sim_y.s(0)
        qrack_y = _sim_y.pauli_expectation([0], [PY])
        assert abs(abs(qrack_y) - 1.0) < 1e-4, f"Fatal: PY probe returned invalid magnitude: {qrack_y}"
        SIGN_Y = 1.0 if qrack_y > 0 else -1.0
        del _sim_y
        # -------------------------------------------
        
        def apply_h(sim, q): sim.h(q)
        def apply_rx(sim, theta, q): sim.r(PX, float(theta), q)
        def apply_rz(sim, theta, q): sim.r(PZ, float(theta), q)
        def apply_zz(sim, theta, q1, q2):
            sim.mcx([q1], q2); apply_rz(sim, 2.0 * theta, q2); sim.mcx([q1], q2)
            
        def trotter_step_body(sim, num_qubits, intra_edges, J, hx, hz, dt_local, steps):
            """
            True 2nd-order Strang Splitting: e^(i hx dt/2 X) * e^(i hz dt Z + i J dt ZZ) * e^(i hx dt/2 X)
            """
            # Hoisted invariants
            dt_half = dt_local / 2.0
            theta_x = -2.0 * hx * dt_half
            theta_z = -2.0 * hz * dt_local
            theta_zz = -J * dt_local

            for _ in range(steps):
                # --- A/2 (Transverse field half-step: dt/2) ---
                for q in range(num_qubits): apply_rx(sim, theta_x, q)
                
                # --- B (Longitudinal field + Interactions full-step: dt) ---
                for q in range(num_qubits): apply_rz(sim, theta_z, q)
                for q1, q2 in intra_edges: apply_zz(sim, theta_zz, q1, q2)
                
                # --- A/2 (Transverse field half-step: dt/2) ---
                for q in range(num_qubits): apply_rx(sim, theta_x, q)

        # Observables dynamically corrected to standard physics convention
        def z_means(sim, qubits):
            return np.array([SIGN_Z * sim.pauli_expectation([q], [PZ]) for q in qubits])
        
        def x_means(sim, qubits):
            return np.array([SIGN_X * sim.pauli_expectation([q], [PX]) for q in qubits])
            
        def y_means(sim, qubits):
            return np.array([SIGN_Y * sim.pauli_expectation([q], [PY]) for q in qubits])
            
        def zz_means_meanfield(z_exp, edges):
            return np.array([z_exp[q1] * z_exp[q2] for q1, q2 in edges])
            
        def apply_kicks(sim, kicks, dt_local, m_every):
            if not kicks: return
            for raw_q, (kx, ky, kz) in kicks.items():
                q = int(raw_q)
                kx *= 2.0 * dt_local * m_every; ky *= 2.0 * dt_local * m_every; kz *= 2.0 * dt_local * m_every
                K = np.sqrt(kx**2 + ky**2 + kz**2)
                if K > 0.0:
                    c, s = np.cos(K / 2.0), np.sin(K / 2.0)
                    nx, ny, nz = kx / K, ky / K, kz / K
                    sim.mtrx([complex(c, -nz * s), complex(-ny * s, -nx * s), complex(ny * s, -nx * s), complex(c, nz * s)], q)

        intra_edges, boundaries = generate_27q_lattice_subvolume()
        
        for p in assigned_patches:
            sim = QrackSimulator(qubit_count=QUBITS_PER_PATCH)
            for q in range(QUBITS_PER_PATCH): apply_h(sim, q)
            sims[p] = sim

        kick_payloads = {p: {} for p in assigned_patches}
    
        for t in range(total_steps):
            s = t / max(1, (total_steps - 1))
            current_hx = (1.0 - s) * 3.0 + s * target_hx
            current_J = s * target_J
            current_hz = s * target_hz
            is_measure = (t % measure_every == 0) or (t == total_steps - 1)
            
            patch_data_to_master = {}
            
            for p in assigned_patches:
                sim = sims[p]
                if kick_payloads[p]:
                    apply_kicks(sim, kick_payloads[p], dt, measure_every)
                    
                trotter_step_body(sim, QUBITS_PER_PATCH, intra_edges, current_J, current_hx, current_hz, dt, 1)
                
                if is_measure:
                    all_q = list(range(QUBITS_PER_PATCH))
                    state = {"Z": z_means(sim, all_q), "X": x_means(sim, all_q), "Y": y_means(sim, all_q)}
                    zz_exp = zz_means_meanfield(state["Z"], intra_edges)
                    bulk_e = -current_hz * float(np.sum(state["Z"])) - current_J * float(np.sum(zz_exp)) - current_hx * float(np.sum(state["X"]))
                    
                    patch_data_to_master[p] = {"state": state, "bulk_energy": bulk_e}
            
            if is_measure:
                conn.send(patch_data_to_master)
                kick_payloads = conn.recv()
            else:
                kick_payloads = {p: {} for p in assigned_patches}
                
    finally:
        sims.clear()
        gc.collect()
        conn.close()

# =====================================================================
# MASTER ORCHESTRATOR
# =====================================================================
class MultiGpuHadronEngine:
    def __init__(self, master_seed: int = 1337):
        self.intra_edges, self.boundaries = generate_27q_lattice_subvolume()
        self.all_boundary_qubits = sorted(set(q for face in self.boundaries.values() for q in face))
        
        self.patch_coords = {}
        idx = 0
        for x in range(GRID_X):
            for y in range(GRID_Y):
                for z in range(GRID_Z):
                    self.patch_coords[idx] = (x, y, z)
                    idx += 1
        self.coord_to_patch = {v: k for k, v in self.patch_coords.items()}
        
        self.lattice_history = []
        
        self.energy_csv = "ground_state_energy_curve_multi.csv"
        self.profiles_csv = "boundary_profiles_multi.csv"
        self.state_dump_file = "macroscopic_lattice_states.npy"
        self.config_file = "lattice_config.json"
        
        self._init_files()
        
        self.worker_assignments = [[] for _ in range(WORKER_GPUS)]
        for i in range(TOTAL_PATCHES):
            self.worker_assignments[i % WORKER_GPUS].append(i)
            
        total_qubits = TOTAL_PATCHES * QUBITS_PER_PATCH
        print(f"Distributed Orchestrator Initialized: {TOTAL_PATCHES} patches ({total_qubits} total logical qubits) over {WORKER_GPUS} OpenCL GPUs.")
        for rank, p_list in enumerate(self.worker_assignments):
            if p_list: print(f"  -> GPU {rank}: {len(p_list)} patches")

    def _init_files(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump({"grid_x": GRID_X, "grid_y": GRID_Y, "grid_z": GRID_Z, "num_patches": TOTAL_PATCHES, "qubits_per_patch": QUBITS_PER_PATCH}, f)
            with open(self.energy_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=["Step", "Anneal_Percent", "Bulk_Energy", "Boundary_Energy", "Total_Energy"]).writeheader()
            with open(self.profiles_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=["Step", "Patch", "Face", "X_mean", "Y_mean", "Z_mean"]).writeheader()
        except Exception: pass

    def _log_csvs(self, step, anneal, bulk, bound, total, patch_profiles):
        try:
            with open(self.energy_csv, mode='a', newline='') as f:
                csv.DictWriter(f, fieldnames=["Step", "Anneal_Percent", "Bulk_Energy", "Boundary_Energy", "Total_Energy"]).writerow(
                    {"Step": step, "Anneal_Percent": anneal, "Bulk_Energy": bulk, "Boundary_Energy": bound, "Total_Energy": total})
            
            with open(self.profiles_csv, mode='a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=["Step", "Patch", "Face", "X_mean", "Y_mean", "Z_mean"])
                for p, prof in patch_profiles.items():
                    q_to_i = {q: i for i, q in enumerate(prof["qubits"])}
                    for face_name, face_qubits in self.boundaries.items():
                        if not face_qubits: continue
                        xm = float(np.mean([prof["means"]["X"][q_to_i[q]] for q in face_qubits]))
                        ym = float(np.mean([prof["means"]["Y"][q_to_i[q]] for q in face_qubits]))
                        zm = float(np.mean([prof["means"]["Z"][q_to_i[q]] for q in face_qubits]))
                        w.writerow({"Step": step, "Patch": p, "Face": face_name, "X_mean": xm, "Y_mean": ym, "Z_mean": zm})
        except Exception: pass

    def run(self, total_steps: int, dt: float, target_g_face: float, target_J: float, target_hx: float, target_hz: float, measure_every: int = 1):
        assert measure_every >= 1, "measure_every must be a positive integer"
        
        active_ranks = [rank for rank in range(WORKER_GPUS) if self.worker_assignments[rank]]
        
        workers = []
        pipes = []
        
        print("Spawning GPU workers...")
        for rank in active_ranks:
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(target=gpu_worker_process, args=(
                rank, self.worker_assignments[rank], child_conn, dt, total_steps, target_J, target_hx, target_hz, measure_every
            ))
            p.start()
            workers.append(p)
            pipes.append(parent_conn)
            
        print("All workers active. Commencing distributed anneal...")
        noise_rng = np.random.default_rng()
        effective_shots = 512.0

        try:
            for t in range(total_steps):
                t0 = time.perf_counter()
                s = t / max(1, (total_steps - 1))
                current_g_face = s * target_g_face
                is_measure = (t % measure_every == 0) or (t == total_steps - 1)
                
                if not is_measure:
                    print(f"Step {t:03d} | Evolve-only")
                    continue
                
                patch_full_states = {}
                bulk_energy = 0.0
                for conn in pipes:
                    data = conn.recv()
                    for p, payload in data.items():
                        patch_full_states[p] = payload["state"]
                        bulk_energy += payload["bulk_energy"]
                        
                step_state = np.zeros((TOTAL_PATCHES, 27, 3))
                patch_profiles = {}
                for p, state in patch_full_states.items():
                    step_state[p, :, 0] = state["X"]
                    step_state[p, :, 1] = state["Y"]
                    step_state[p, :, 2] = state["Z"]
                    
                    bq = self.all_boundary_qubits
                    patch_profiles[p] = {
                        "qubits": bq,
                        "means": {"X": state["X"][bq], "Y": state["Y"][bq], "Z": state["Z"][bq]},
                        "vars": {
                            "X": np.clip(1.0 - state["X"][bq]**2, 0.0, 1.0),
                            "Y": np.clip(1.0 - state["Y"][bq]**2, 0.0, 1.0),
                            "Z": np.clip(1.0 - state["Z"][bq]**2, 0.0, 1.0),
                        }
                    }
                self.lattice_history.append(step_state)

                next_kick_payloads = {p: {} for p in range(TOTAL_PATCHES)}
                macroscopic_boundary_energy = 0.0
                scale = np.sqrt(dt * measure_every / effective_shots)
                stochastic_noise = {}
                
                for p, prof in patch_profiles.items():
                    n_b = len(prof["qubits"])
                    xn = noise_rng.normal(0.0, 1.0, n_b) * np.sqrt(prof["vars"]["X"]) * scale
                    yn = noise_rng.normal(0.0, 1.0, n_b) * np.sqrt(prof["vars"]["Y"]) * scale
                    zn = noise_rng.normal(0.0, 1.0, n_b) * np.sqrt(prof["vars"]["Z"]) * scale
                    stochastic_noise[p] = {q: (xn[i], yn[i], zn[i]) for i, q in enumerate(prof["qubits"])}

                for p1, coord1 in self.patch_coords.items():
                    x1, y1, z1 = coord1
                    neighbors = {
                        "+X": (x1+1, y1, z1), "-X": (x1-1, y1, z1),
                        "+Y": (x1, y1+1, z1), "-Y": (x1, y1-1, z1),
                        "+Z": (x1, y1, z1+1), "-Z": (x1, y1, z1-1),
                    }
                    
                    for dir1, coord2 in neighbors.items():
                        p2 = self.coord_to_patch.get(coord2)
                        if p2 is None or p1 >= p2: continue
                            
                        dir2 = dir1.replace("+", "temp").replace("-", "+").replace("temp", "-")
                        face1_q, face2_q = self.boundaries[dir1], self.boundaries[dir2]
                        prof1, noise1 = patch_profiles[p1], stochastic_noise[p1]
                        prof2, noise2 = patch_profiles[p2], stochastic_noise[p2]
                        
                        q_to_i1 = {q: i for i, q in enumerate(prof1["qubits"])}
                        q_to_i2 = {q: i for i, q in enumerate(prof2["qubits"])}
                        
                        ax2 = np.mean([prof2["means"]["X"][q_to_i2[q]] + noise2[q][0] for q in face2_q])
                        ay2 = np.mean([prof2["means"]["Y"][q_to_i2[q]] + noise2[q][1] for q in face2_q])
                        az2 = np.mean([prof2["means"]["Z"][q_to_i2[q]] + noise2[q][2] for q in face2_q])
                        
                        ax1 = np.mean([prof1["means"]["X"][q_to_i1[q]] + noise1[q][0] for q in face1_q])
                        ay1 = np.mean([prof1["means"]["Y"][q_to_i1[q]] + noise1[q][1] for q in face1_q])
                        az1 = np.mean([prof1["means"]["Z"][q_to_i1[q]] + noise1[q][2] for q in face1_q])
                        
                        macroscopic_boundary_energy += current_g_face * (ax1*ax2 + ay1*ay2 + az1*az2) * ((len(face1_q) + len(face2_q)) / 2.0)
                        
                        for q1f in face1_q:
                            k = next_kick_payloads[p1].get(q1f, (0., 0., 0.))
                            next_kick_payloads[p1][q1f] = (k[0] + current_g_face * ax2, k[1] + current_g_face * ay2, k[2] + current_g_face * az2)
                        for q2f in face2_q:
                            k = next_kick_payloads[p2].get(q2f, (0., 0., 0.))
                            next_kick_payloads[p2][q2f] = (k[0] + current_g_face * ax1, k[1] + current_g_face * ay1, k[2] + current_g_face * az1)

                total_energy = bulk_energy + macroscopic_boundary_energy
                print(f"Step {t:03d} | E: {total_energy:+.4f} | {time.perf_counter() - t0:.2f}s")
                self._log_csvs(t, s * 100, bulk_energy, macroscopic_boundary_energy, total_energy, patch_profiles)
                
                for i, rank in enumerate(active_ranks):
                    worker_payload = {p: next_kick_payloads[p] for p in self.worker_assignments[rank]}
                    pipes[i].send(worker_payload)

            np.save(self.state_dump_file, np.array(self.lattice_history))
            print(f"--- Full 3D Lattice History dumped to {self.state_dump_file} ---")

        finally:
            print("Terminating GPU workers...")
            for p in workers:
                p.join(timeout=2)
                if p.is_alive(): p.terminate()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    engine = MultiGpuHadronEngine(master_seed=1337)
    try:
        engine.run(
            total_steps=100, dt=0.04, target_g_face=0.15, target_J=1.0, target_hx=0.5, target_hz=0.2, measure_every=1
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
