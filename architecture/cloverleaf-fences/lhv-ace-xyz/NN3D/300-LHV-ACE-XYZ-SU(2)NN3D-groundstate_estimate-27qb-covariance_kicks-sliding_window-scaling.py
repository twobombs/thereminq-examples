# -*- coding: us-ascii -*-
# 27-Qubit 3x3x3 Lattice & Macroscopic Grid Annealing
# High-Throughput Volumetric Engine with Statistical Variance Injection
#
# REVISION 44 - DYNAMIC VOLUME CONFIGURATION
#
# ARCHITECTURE UPDATES:
# - Automatically exports grid dimensions to lattice_config.json to allow 
#   dynamic scaling in downstream visualization scripts.
# - Carries forward all S2 Trotter fixes, exact ZZ-parity logic, and 
#   the single-pass VRAM paging architecture.

import os
import sys
import gc
import csv
import json
import time
import numpy as np
from typing import List, Tuple, Dict, Any

# =====================================================================
# ENVIRONMENT - set before pyqrack import
# =====================================================================
QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
os.environ["PYQRACK_SHARED_LIB_PATH"] = QRACK_LIB_PATH
os.environ["OCL_ICD_PLATFORM_SORT"] = "none"
os.environ["QRACK_OCL_DEFAULT_DEVICE"] = os.environ.get("WORMHOLE_GPU", "0")
os.environ["QRACK_QPAGER_DEVICES"] = "-1"
os.environ["QRACK_QUNITMULTI_DEVICES"] = "-1"
os.environ["QRACK_FPPOW"] = "5"
os.environ["QRACK_MAX_ALLOC_MB"] = "64000"
os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "0"
os.environ["QRACK_QUNIT_SEPARABILITY_THRESHOLD"] = "1e-7"

from pyqrack import QrackSimulator

# =====================================================================
# GATES
# =====================================================================
PX, PY, PZ = 1, 2, 3

def apply_h(sim: QrackSimulator, q: int) -> None:
    sim.h(q)

def apply_rx(sim: QrackSimulator, theta: float, q: int) -> None:
    sim.r(PX, float(theta), q)

def apply_rz(sim: QrackSimulator, theta: float, q: int) -> None:
    sim.r(PZ, float(theta), q)

def apply_zz(sim: QrackSimulator, theta: float, q1: int, q2: int) -> None:
    sim.cnot(q1, q2)
    apply_rz(sim, 2.0 * theta, q2)
    sim.cnot(q1, q2)

def trotter_step_body(sim: QrackSimulator, num_qubits: int, intra_edges: List[Tuple[int, int]], J: float, hx: float, hz: float, dt: float, steps: int) -> None:
    for _ in range(steps):
        for q in range(num_qubits):
            apply_rx(sim, -hx * dt, q)
        for q in range(num_qubits):
            apply_rz(sim, -2.0 * hz * dt, q)
        for q1, q2 in intra_edges:
            apply_zz(sim, -J * dt, q1, q2)
        for q in range(num_qubits):
            apply_rx(sim, -hx * dt, q)

def z_means(sim: QrackSimulator, qubits: List[int]) -> np.ndarray:
    return np.array([1.0 - 2.0 * sim.prob(q) for q in qubits])

def x_means(sim: QrackSimulator, qubits: List[int]) -> np.ndarray:
    out = np.empty(len(qubits))
    for i, q in enumerate(qubits):
        apply_h(sim, q)
        out[i] = 1.0 - 2.0 * sim.prob(q)
        apply_h(sim, q)
    return out

def y_means(sim: QrackSimulator, qubits: List[int]) -> np.ndarray:
    out = np.empty(len(qubits))
    for i, q in enumerate(qubits):
        apply_rx(sim, np.pi / 2, q)
        out[i] = 1.0 - 2.0 * sim.prob(q)
        apply_rx(sim, -np.pi / 2, q)
    return out

def zz_means_exact(sim: QrackSimulator, edges: List[Tuple[int, int]]) -> np.ndarray:
    out = np.empty(len(edges))
    for i, (q1, q2) in enumerate(edges):
        sim.cnot(q1, q2)
        out[i] = 1.0 - 2.0 * sim.prob(q2)
        sim.cnot(q1, q2)
    return out

# =====================================================================
# TOPOLOGY
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
                if x < lx - 1:
                    edges.append((idx, (x + 1) * (ly * lz) + y * lz + z))
                if y < ly - 1:
                    edges.append((idx, x * (ly * lz) + (y + 1) * lz + z))
                if z < lz - 1:
                    edges.append((idx, x * (ly * lz) + y * lz + (z + 1)))
                
                if x == 0:
                    boundaries["-X"].append(idx)
                if x == lx - 1:
                    boundaries["+X"].append(idx)
                if y == 0:
                    boundaries["-Y"].append(idx)
                if y == ly - 1:
                    boundaries["+Y"].append(idx)
                if z == 0:
                    boundaries["-Z"].append(idx)
                if z == lz - 1:
                    boundaries["+Z"].append(idx)
                    
    return edges, boundaries

# =====================================================================
# ORCHESTRATOR
# =====================================================================
class VolumetricHadronEngine27Q:
    def __init__(self, grid: Tuple[int, int, int] = (4, 4, 1), master_seed: int = 42):
        self.grid_x, self.grid_y, self.grid_z = grid
        self.num_patches = self.grid_x * self.grid_y * self.grid_z
        self.qubits_per_patch = 27
        self.intra_edges, self.boundaries = generate_27q_lattice_subvolume()
        self.all_boundary_qubits = sorted(set(q for face in self.boundaries.values() for q in face))
        
        self.patch_coords: Dict[int, Tuple[int, int, int]] = {}
        idx = 0
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                for z in range(self.grid_z):
                    self.patch_coords[idx] = (x, y, z)
                    idx += 1
        self.coord_to_patch = {v: k for k, v in self.patch_coords.items()}
        
        self.energy_history: List[Dict[str, Any]] = []
        self.lattice_history: List[np.ndarray] = []
        
        self.energy_csv = "ground_state_energy_curve_27q.csv"
        self.profiles_csv = "boundary_profiles_27q.csv"
        self.rng_state_file = "rng_state_27q.json"
        self.state_dump_file = "macroscopic_lattice_states.npy"
        self.config_file = "lattice_config.json"
        
        self._init_csvs()
        
        # EXPORT CONFIGURATION FOR VISUALIZER
        try:
            with open(self.config_file, 'w') as f:
                json.dump({
                    "grid_x": self.grid_x, 
                    "grid_y": self.grid_y, 
                    "grid_z": self.grid_z,
                    "num_patches": self.num_patches,
                    "qubits_per_patch": self.qubits_per_patch
                }, f)
        except Exception as e:
            print(f"Failed to write config file: {e}")
        
        print(f"Initializing 27Q Resident-Simulator Engine...")
        t0 = time.perf_counter()
        self.sims: List[QrackSimulator] = []
        for p in range(self.num_patches):
            sim = QrackSimulator(qubit_count=self.qubits_per_patch)
            for q in range(self.qubits_per_patch):
                apply_h(sim, q)
            self.sims.append(sim)
        print(f"All {self.num_patches} simulators resident. Init: {time.perf_counter() - t0:.1f}s")

    def _init_csvs(self) -> None:
        try:
            with open(self.energy_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=["Step", "Anneal_Percent", "Bulk_Energy", "Boundary_Energy", "Total_Energy"]).writeheader()
            with open(self.profiles_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=["Step", "Patch", "Face", "X_mean", "Y_mean", "Z_mean"]).writeheader()
        except Exception:
            pass

    def _append_energy_csv(self, data: Dict[str, Any]) -> None:
        try:
            with open(self.energy_csv, mode='a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=["Step", "Anneal_Percent", "Bulk_Energy", "Boundary_Energy", "Total_Energy"])
                w.writerow(data)
        except Exception:
            pass

    def _append_profiles_csv(self, step: int, patch_profiles: Dict[int, Dict[str, Any]]) -> None:
        try:
            with open(self.profiles_csv, mode='a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=["Step", "Patch", "Face", "X_mean", "Y_mean", "Z_mean"])
                for p, prof in patch_profiles.items():
                    q_to_i = {q: i for i, q in enumerate(prof["qubits"])}
                    for face_name, face_qubits in self.boundaries.items():
                        if not face_qubits: continue
                        x_mean = float(np.mean([prof["means"]["X"][q_to_i[q]] for q in face_qubits]))
                        y_mean = float(np.mean([prof["means"]["Y"][q_to_i[q]] for q in face_qubits]))
                        z_mean = float(np.mean([prof["means"]["Z"][q_to_i[q]] for q in face_qubits]))
                        w.writerow({"Step": step, "Patch": p, "Face": face_name, "X_mean": x_mean, "Y_mean": y_mean, "Z_mean": z_mean})
        except Exception:
            pass

    def _measure_patch_full(self, sim: QrackSimulator) -> Dict[str, np.ndarray]:
        """Measures X, Y, Z for all 27 qubits in a single pass."""
        all_q = list(range(self.qubits_per_patch))
        return {
            "Z": z_means(sim, all_q),
            "X": x_means(sim, all_q),
            "Y": y_means(sim, all_q)
        }

    def _apply_kicks(self, sim: QrackSimulator, kicks: Dict[int, Tuple[float, float, float]], dt: float, measure_every: int) -> None:
        if not kicks: return
        for raw_q, (kx, ky, kz) in kicks.items():
            q = int(raw_q)
            kx *= 2.0 * dt * measure_every
            ky *= 2.0 * dt * measure_every
            kz *= 2.0 * dt * measure_every
            K = np.sqrt(kx**2 + ky**2 + kz**2)
            if K > 0.0:
                c, s = np.cos(K / 2.0), np.sin(K / 2.0)
                nx, ny, nz = kx / K, ky / K, kz / K
                sim.mtrx([
                    complex(c, -nz * s), complex(-ny * s, -nx * s),
                    complex(ny * s, -nx * s), complex(c, nz * s)
                ], q)

    def anneal_to_ground_state(self, total_steps: int, dt: float, target_g_face: float, target_J: float, target_hx: float, target_hz: float, measure_every: int = 1) -> None:
        print(f"\nStarting Adiabatic Anneal...")
        noise_rng = np.random.default_rng()
        try:
            with open(self.rng_state_file, 'w') as f:
                json.dump(noise_rng.bit_generator.state, f)
        except: pass
            
        effective_shots = 512.0
        kick_payloads = {p: {} for p in range(self.num_patches)}
        
        for t in range(total_steps):
            t0 = time.perf_counter()
            s = t / max(1, (total_steps - 1))
            current_hx = (1.0 - s) * 3.0 + s * target_hx
            current_J = s * target_J
            current_hz = s * target_hz
            current_g_face = s * target_g_face
            is_measure = (t % measure_every == 0) or (t == total_steps - 1)
            
            patch_full_states = {}
            bulk_energy = 0.0
            
            # --- THE SINGLE PASS ---
            for p in range(self.num_patches):
                sim = self.sims[p]
                if kick_payloads[p]:
                    self._apply_kicks(sim, kick_payloads[p], dt, measure_every)
                    
                trotter_step_body(sim, self.qubits_per_patch, self.intra_edges, current_J, current_hx, current_hz, dt, 1)
                
                if is_measure:
                    # Measure the entire 27 qubit patch
                    state = self._measure_patch_full(sim)
                    patch_full_states[p] = state
                    
                    # Calculate bulk energy using the full state array
                    zz_exp = zz_means_exact(sim, self.intra_edges)
                    bulk_e = -current_hz * float(np.sum(state["Z"])) - current_J * float(np.sum(zz_exp)) - current_hx * float(np.sum(state["X"]))
                    bulk_energy += bulk_e
                    
            kick_payloads = {p: {} for p in range(self.num_patches)}
            if not is_measure:
                print(f"Step {t:03d} | Evolve-only | {time.perf_counter() - t0:.1f}s")
                continue
                
            # Restructure full state into boundaries and state tensor dump
            step_state = np.zeros((self.num_patches, 27, 3))
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
            self._append_profiles_csv(t, patch_profiles)
                
            next_kick_payloads = {p: {} for p in range(self.num_patches)}
            macroscopic_boundary_energy = 0.0
            scale = np.sqrt(dt * measure_every / effective_shots)
            stochastic_noise = {}
            
            for p, prof in patch_profiles.items():
                n_bounds = len(prof["qubits"])
                X_noise = noise_rng.normal(0.0, 1.0, n_bounds) * np.sqrt(prof["vars"]["X"]) * scale
                Y_noise = noise_rng.normal(0.0, 1.0, n_bounds) * np.sqrt(prof["vars"]["Y"]) * scale
                Z_noise = noise_rng.normal(0.0, 1.0, n_bounds) * np.sqrt(prof["vars"]["Z"]) * scale
                stochastic_noise[p] = {q: (X_noise[i], Y_noise[i], Z_noise[i]) for i, q in enumerate(prof["qubits"])}
                
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
                        
            kick_payloads = next_kick_payloads
            total_energy = bulk_energy + macroscopic_boundary_energy
            
            print(f"Step {t:03d} | E: {total_energy:+.4f} | {time.perf_counter() - t0:.1f}s")
            self._append_energy_csv({"Step": t, "Anneal_Percent": s * 100, "Bulk_Energy": bulk_energy, "Boundary_Energy": macroscopic_boundary_energy, "Total_Energy": total_energy})
            
            if t == total_steps - 1:
                try:
                    np.save(self.state_dump_file, np.array(self.lattice_history))
                    print(f"--- Full 3D Lattice History dumped to {self.state_dump_file} ---")
                except Exception as e:
                    print(f"Failed to dump lattice history: {e}")

    def shutdown(self) -> None:
        for _ in range(len(self.sims)): del self.sims[0]
        gc.collect()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

if __name__ == "__main__":
    # Change the grid argument here, and the visualizer will adapt automatically.
    engine = VolumetricHadronEngine27Q(grid=(4, 4, 3), master_seed=1337)
    try:
        engine.anneal_to_ground_state(
            total_steps=100, dt=0.04, target_g_face=0.15, target_J=1.0, target_hx=0.5, target_hz=0.2, measure_every=1
        )
    except KeyboardInterrupt: pass
    finally: engine.shutdown()
