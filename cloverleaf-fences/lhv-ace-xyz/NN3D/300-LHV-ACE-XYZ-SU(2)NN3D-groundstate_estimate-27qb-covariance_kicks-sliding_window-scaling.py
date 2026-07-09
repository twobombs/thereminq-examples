# -*- coding: us-ascii -*-
# 27-Qubit 3x3x3 Lattice & Macroscopic Grid Annealing
# High-Throughput Volumetric Engine with Statistical Variance Injection
#
# REVISION 37 - SINGLE-PROCESS RESIDENT SIMULATORS (CLEAN STDOUT)
#
# DESIGN CORRECTION:
# The user story is: "evolve independent 27-qubit patches with periodic
# mean-field boundary coupling, approximating a larger stitched lattice."
# Nothing in that story requires statevectors to leave the simulator.
#
# OpenCL already performs buffer residency management: allocations that
# do not fit on-device and are not referenced by an executing kernel are
# migrated to host by the runtime, transparently. QRack builds on this.
#
# ONE-PASS ARCHITECTURE: 
# To halve the PCIe paging penalty, boundary kicks evaluated at time t 
# are stored in host memory and applied at the start of time t+1. 
# Exactly one VRAM page-in per patch.
#
# DYNAMIC SCHMIDT SPLITTING & COMPRESSION:
# Microscopic initialization jitter has been removed to allow lossless 
# BDD compression. QUnit Fidelity Guard is RE-ENABLED with a separability 
# threshold of 1e-7, allowing Qrack to dynamically tear apart weakly 
# entangled statevectors to bypass TTM oversubscription limits.
#
# CARRIED FORWARD:
# Environment isolation (DRI_PRIME, RUSTICL*, FPPOW=5) set once.
# Trotter S2 symmetry, ZZ mcmtrx decomposition, mean-field ZZ readout.
# Stochastic variance injection on boundary measurements.
# Breadcrumb trail: Energy split, per-face profiles, RNG state dump.
# Scorched-earth OS shutdown to prevent AMD driver VRAM locks.
# Per-patch initialization stdout suppressed for cleaner scaling logs.
#
# MEMORY BUDGET:
# 48 patches (4x4x3) x 1 GB = 48 GB theoretical statevector footprint.
# Routed to a single 8 GB Radeon Pro V340 Vega die. 
# Relies on Qrack dynamic compression + OpenCL TTM to manage load.

import os
import sys

# =====================================================================
# ENVIRONMENT - set before pyqrack import
# =====================================================================
QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
os.environ["PYQRACK_SHARED_LIB_PATH"] = QRACK_LIB_PATH
os.environ["DRI_PRIME"] = "pci-0000_44_00_0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OCL_ICD_PLATFORM_SORT"] = "none"
os.environ["RUSTICL_ENABLE"] = "radeonsi"
os.environ["RUSTICL_ALLOW_SVM"] = "0"
os.environ["MESA_VK_DEVICE_SELECT"] = "amd"

# --- QRACK OPTIMIZATION SETTINGS ---
os.environ["QRACK_OCL_DEFAULT_DEVICE"] = os.environ.get("WORMHOLE_GPU", "0")
os.environ["QRACK_QPAGER_DEVICES"] = "-1"
os.environ["QRACK_QUNITMULTI_DEVICES"] = "-1"
os.environ["QRACK_FPPOW"] = "5"
os.environ["QRACK_MAX_ALLOC_MB"] = "64000"

# Re-enable the dynamic separation guard
os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "0"
# Tell Qrack: "If tearing the statevector apart loses less than 1e-7 fidelity, do it."
os.environ["QRACK_QUNIT_SEPARABILITY_THRESHOLD"] = "1e-7"
# -------------------------------------------

import gc
import csv
import json
import time
import numpy as np
from typing import List, Tuple, Dict, Any
from pyqrack import QrackSimulator

# =====================================================================
# GATES
# =====================================================================
PX, PY, PZ = 1, 2, 3

def apply_h(sim: Any, q: int) -> None:
    sim.h(q)

def apply_rx(sim: Any, theta: float, q: int) -> None:
    sim.r(PX, float(theta), q)

def apply_rz(sim: Any, theta: float, q: int) -> None:
    sim.r(PZ, float(theta), q)

def apply_zz(sim: Any, theta: float, q1: int, q2: int) -> None:
    apply_rz(sim, theta, q1)
    apply_rz(sim, theta, q2)
    ph = complex(np.cos(2.0 * theta), -np.sin(2.0 * theta))
    try:
        sim.mcmtrx([q1], [complex(1, 0), 0j, 0j, ph], q2)
    except TypeError:
        sim.mcmtrx([q1], [complex(1, 0), 0j, 0j, ph], [q2])

def trotter_step_body(sim: Any, num_qubits: int, intra_edges: List[Tuple[int, int]], J: float, hx: float, hz: float, dt: float, steps: int) -> None:
    for _ in range(steps):
        for q in range(num_qubits):
            apply_rx(sim, -hx * dt, q)
        for q in range(num_qubits):
            apply_rz(sim, -hz * dt, q)
        for q1, q2 in intra_edges:
            apply_zz(sim, -2.0 * J * dt, q1, q2)
        for q in range(num_qubits):
            apply_rz(sim, -hz * dt, q)
        for q in range(num_qubits):
            apply_rx(sim, -hx * dt, q)

def z_means(sim: Any, qubits: List[int]) -> np.ndarray:
    return np.array([1.0 - 2.0 * sim.prob(q) for q in qubits])

def x_means(sim: Any, qubits: List[int]) -> np.ndarray:
    out = np.empty(len(qubits))
    for i, q in enumerate(qubits):
        apply_h(sim, q)
        out[i] = 1.0 - 2.0 * sim.prob(q)
        apply_h(sim, q)
    return out

def y_means(sim: Any, qubits: List[int]) -> np.ndarray:
    out = np.empty(len(qubits))
    for i, q in enumerate(qubits):
        apply_rx(sim, np.pi / 2, q)
        out[i] = 1.0 - 2.0 * sim.prob(q)
        apply_rx(sim, -np.pi / 2, q)
    return out

def zz_means_meanfield(z_exp: np.ndarray, edges: List[Tuple[int, int]]) -> np.ndarray:
    return np.array([z_exp[q1] * z_exp[q2] for q1, q2 in edges])

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
# ORCHESTRATOR - single process, resident simulators
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
        
        # Breadcrumb files
        self.energy_csv = "ground_state_energy_curve_27q.csv"
        self.profiles_csv = "boundary_profiles_27q.csv"
        self.rng_state_file = "rng_state_27q.json"
        self._init_csvs()
        
        total_sites = self.num_patches * self.qubits_per_patch
        vram_gb = self.num_patches * (1 << self.qubits_per_patch) * 8 / 1024 ** 3
        
        print(f"Initializing 27Q Resident-Simulator Engine...")
        print(f"Grid: {grid} = {self.num_patches} patches x {self.qubits_per_patch}q = {total_sites} total logical qubits")
        print(f"Total theoretical statevector footprint: {vram_gb:.1f} GB")
        print(f"Dynamic Separability Threshold set to 1e-7 to actively compress VRAM.")
        
        print("Allocating resident simulators...")
        t0 = time.perf_counter()
        self.sims: List[Any] = []
        
        for p in range(self.num_patches):
            sim = QrackSimulator(qubit_count=self.qubits_per_patch)
            # Starting in a pure |+> state allows Qrack's BDD layer to perfectly compress the state.
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
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass

    def _append_profiles_csv(self, step: int, patch_profiles: Dict[int, Dict[str, Any]]) -> None:
        """Averages scalar arrays across each geometric face per patch and logs them."""
        try:
            with open(self.profiles_csv, mode='a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=["Step", "Patch", "Face", "X_mean", "Y_mean", "Z_mean"])
                for p, prof in patch_profiles.items():
                    q_to_i = {q: i for i, q in enumerate(prof["qubits"])}
                    for face_name, face_qubits in self.boundaries.items():
                        if not face_qubits:
                            continue
                        x_mean = float(np.mean([prof["means"]["X"][q_to_i[q]] for q in face_qubits]))
                        y_mean = float(np.mean([prof["means"]["Y"][q_to_i[q]] for q in face_qubits]))
                        z_mean = float(np.mean([prof["means"]["Z"][q_to_i[q]] for q in face_qubits]))
                        w.writerow({
                            "Step": step, "Patch": p, "Face": face_name,
                            "X_mean": x_mean, "Y_mean": y_mean, "Z_mean": z_mean
                        })
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass

    def _measure_boundary_profile(self, sim: Any) -> Dict[str, Any]:
        """Expectation values on boundary qubits. Scalar reads only - no statevector leaves the device."""
        Z_mean = z_means(sim, self.all_boundary_qubits)
        X_mean = x_means(sim, self.all_boundary_qubits)
        Y_mean = y_means(sim, self.all_boundary_qubits)
        
        return {
            "qubits": self.all_boundary_qubits,
            "means": {"X": X_mean, "Y": Y_mean, "Z": Z_mean},
            "vars": {
                "X": np.clip(1.0 - X_mean ** 2, 0.0, 1.0),
                "Y": np.clip(1.0 - Y_mean ** 2, 0.0, 1.0),
                "Z": np.clip(1.0 - Z_mean ** 2, 0.0, 1.0),
            },
        }

    def _apply_kicks(self, sim: Any, kicks: Dict[int, Tuple[float, float, float]], dt: float, measure_every: int) -> None:
        """Apply boundary kicks as single-qubit rotations directly on the resident simulator."""
        if not kicks:
            return
            
        for raw_q, (kx, ky, kz) in kicks.items():
            q = int(raw_q)
            
            # Apply time scaling to the impulse components
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

    def _read_bulk_energy(self, sim: Any, J: float, hx: float, hz: float) -> float:
        """Read bulk energy expectations directly from resident simulator."""
        all_q = list(range(self.qubits_per_patch))
        z_exp = z_means(sim, all_q)
        x_exp = x_means(sim, all_q)
        zz_exp = zz_means_meanfield(z_exp, self.intra_edges)
        
        return (-hz * float(np.sum(z_exp)) - J * float(np.sum(zz_exp)) - hx * float(np.sum(x_exp)))

    def anneal_to_ground_state(self, total_steps: int, dt: float, target_g_face: float, target_J: float, target_hx: float, target_hz: float, measure_every: int = 1) -> None:
        print(f"\nStarting Adiabatic Anneal (resident-simulator path)...")
        if measure_every > 1:
            print(f"Measurement cadence: every {measure_every} steps")
            
        self.energy_history.clear()
        noise_rng = np.random.default_rng()
        
        # Dump RNG state for reproducibility
        try:
            with open(self.rng_state_file, 'w') as f:
                json.dump(noise_rng.bit_generator.state, f)
            print(f"RNG state checkpointed to {self.rng_state_file}")
        except Exception as e:
            print(f"Failed to dump RNG state: {e}")
            
        effective_shots = 512.0
        
        # Initialize an empty payload dictionary for the very first step
        kick_payloads = {p: {} for p in range(self.num_patches)}
        
        for t in range(total_steps):
            t0 = time.perf_counter()
            s = t / max(1, (total_steps - 1))
            current_hx = (1.0 - s) * 3.0 + s * target_hx
            current_J = s * target_J
            current_hz = s * target_hz
            current_g_face = s * target_g_face
            is_measure = (t % measure_every == 0) or (t == total_steps - 1)
            
            patch_profiles = {}
            bulk_energy = 0.0
            
            # ---------------------------------------------------------
            # THE SINGLE PASS (One VRAM load/unload cycle per patch)
            # ---------------------------------------------------------
            for p in range(self.num_patches):
                sim = self.sims[p]
                
                # 1. Apply boundary coupling from previous step (Trotter A)
                if kick_payloads[p]:
                    self._apply_kicks(sim, kick_payloads[p], dt, measure_every)
                    kick_payloads[p] = {} # Clear payload after application
                    
                # 2. Evolve bulk (Trotter B)
                trotter_step_body(
                    sim, self.qubits_per_patch, self.intra_edges,
                    current_J, current_hx, current_hz, dt, 1
                )
                
                # 3. Read profiles and energy synchronously on the cohesive state
                if is_measure:
                    patch_profiles[p] = self._measure_boundary_profile(sim)
                    bulk_energy += self._read_bulk_energy(sim, current_J, current_hx, current_hz)
                    
            # ---------------------------------------------------------
            # HOST MATH (No VRAM touches)
            # ---------------------------------------------------------
            if not is_measure:
                print(f"Step {t:03d} | Anneal: {s*100:05.1f}% | (evolve-only, {time.perf_counter() - t0:.1f}s)")
                continue
                
            # Log face-averaged profiles to secondary CSV
            self._append_profiles_csv(t, patch_profiles)
                
            next_kick_payloads = {p: {} for p in range(self.num_patches)}
            macroscopic_boundary_energy = 0.0
            
            # Noise variance scaling relative to elapsed measure_every dt
            scale = np.sqrt(dt * measure_every / effective_shots)
            stochastic_noise: Dict[int, Dict[int, Tuple[float, float, float]]] = {}
            
            for p, prof in patch_profiles.items():
                n_bounds = len(prof["qubits"])
                X_noise = noise_rng.normal(0.0, 1.0, n_bounds) * np.sqrt(prof["vars"]["X"]) * scale
                Y_noise = noise_rng.normal(0.0, 1.0, n_bounds) * np.sqrt(prof["vars"]["Y"]) * scale
                Z_noise = noise_rng.normal(0.0, 1.0, n_bounds) * np.sqrt(prof["vars"]["Z"]) * scale
                
                for arr in (X_noise, Y_noise, Z_noise):
                    if not np.all(np.isfinite(arr)):
                        arr[:] = 0.0
                        
                stochastic_noise[p] = {q: (X_noise[i], Y_noise[i], Z_noise[i]) for i, q in enumerate(prof["qubits"])}
                
            for p1, coord1 in self.patch_coords.items():
                x1, y1, z1 = coord1
                neighbors = {
                    "+X": (x1 + 1, y1, z1), "-X": (x1 - 1, y1, z1),
                    "+Y": (x1, y1 + 1, z1), "-Y": (x1, y1 - 1, z1),
                    "+Z": (x1, y1, z1 + 1), "-Z": (x1, y1, z1 - 1),
                }
                
                for dir1, coord2 in neighbors.items():
                    if not (0 <= coord2[0] < self.grid_x and 0 <= coord2[1] < self.grid_y and 0 <= coord2[2] < self.grid_z):
                        continue
                        
                    p2 = self.coord_to_patch.get(coord2)
                    if p2 is None or p1 >= p2:
                        continue
                        
                    dir2 = dir1.replace("+", "temp").replace("-", "+").replace("temp", "-")
                    face1_q = self.boundaries[dir1]
                    face2_q = self.boundaries[dir2]
                    prof1, noise1 = patch_profiles[p1], stochastic_noise[p1]
                    prof2, noise2 = patch_profiles[p2], stochastic_noise[p2]
                    
                    q_to_i1 = {q: i for i, q in enumerate(prof1["qubits"])}
                    q_to_i2 = {q: i for i, q in enumerate(prof2["qubits"])}
                    
                    ax2 = ay2 = az2 = 0.0
                    for q2 in face2_q:
                        i2 = q_to_i2[q2]
                        ax2 += prof2["means"]["X"][i2] + noise2[q2][0]
                        ay2 += prof2["means"]["Y"][i2] + noise2[q2][1]
                        az2 += prof2["means"]["Z"][i2] + noise2[q2][2]
                    n2 = max(1, len(face2_q))
                    ax2 /= n2; ay2 /= n2; az2 /= n2
                    
                    ax1 = ay1 = az1 = 0.0
                    for q1f in face1_q:
                        i1 = q_to_i1[q1f]
                        ax1 += prof1["means"]["X"][i1] + noise1[q1f][0]
                        ay1 += prof1["means"]["Y"][i1] + noise1[q1f][1]
                        az1 += prof1["means"]["Z"][i1] + noise1[q1f][2]
                    n1 = max(1, len(face1_q))
                    ax1 /= n1; ay1 /= n1; az1 /= n1
                    
                    # Note: +g_face is retained here, making boundary interactions antiferromagnetic
                    interaction_E = current_g_face * (ax1 * ax2 + ay1 * ay2 + az1 * az2) * ((len(face1_q) + len(face2_q)) / 2.0)
                    macroscopic_boundary_energy += interaction_E
                    
                    for q1f in face1_q:
                        k = next_kick_payloads[p1].get(q1f, (0., 0., 0.))
                        next_kick_payloads[p1][q1f] = (k[0] + current_g_face * ax2, k[1] + current_g_face * ay2, k[2] + current_g_face * az2)
                        
                    for q2f in face2_q:
                        k = next_kick_payloads[p2].get(q2f, (0., 0., 0.))
                        next_kick_payloads[p2][q2f] = (k[0] + current_g_face * ax1, k[1] + current_g_face * ay1, k[2] + current_g_face * az1)
                        
            # Store generated payloads for the next sequential pass
            kick_payloads = next_kick_payloads
                
            total_energy = bulk_energy + macroscopic_boundary_energy
            elapsed = time.perf_counter() - t0
            print(f"Step {t:03d} | Anneal: {s*100:05.1f}% | Bulk E: {bulk_energy:+.4f} | Bound E: {macroscopic_boundary_energy:+.4f} | Total: {total_energy:+.4f} | {elapsed:.1f}s")
            
            step_data = {
                "Step": t, "Anneal_Percent": s * 100,
                "Bulk_Energy": bulk_energy, "Boundary_Energy": macroscopic_boundary_energy, "Total_Energy": total_energy
            }
            self.energy_history.append(step_data)
            self._append_energy_csv(step_data)
            
            if t == total_steps - 1:
                print(f"--- Calculating Final Benchmarks ---")
                purities = []
                all_q = list(range(self.qubits_per_patch))
                for p in range(self.num_patches):
                    sim = self.sims[p]
                    z_e = z_means(sim, all_q)
                    x_e = x_means(sim, all_q)
                    y_e = y_means(sim, all_q)
                    purities.append(float(np.mean(x_e ** 2 + y_e ** 2 + z_e ** 2)))
                avg_purity = sum(purities) / self.num_patches
                print(f"--- Avg Sub-Volume Purity: {avg_purity:.4f}")

    def shutdown(self) -> None:
        print("\nInitiating scorched-earth VRAM teardown...")
        # 1. Explicitly delete the simulators to trigger PyQrack C++ destructors
        for _ in range(len(self.sims)):
            del self.sims[0]
            
        # 2. Force Python garbage collection
        gc.collect()
        
        print("Hardware contexts released. Terminating process.")
        
        # 3. Bypass standard sys.exit() and gracefully force the OS to reap the PID instantly.
        # This prevents lingering C++ threads or locked OpenCL queues from zombifying the VRAM.
        os._exit(0)

# =====================================================================
# EXECUTION
# =====================================================================
if __name__ == "__main__":
    # 48 Patches (4x4x3 grid) = 1,296 logical qubits.
    engine = VolumetricHadronEngine27Q(grid=(4, 4, 3), master_seed=1337)
    try:
        engine.anneal_to_ground_state(
            total_steps=100,
            dt=0.04,
            target_g_face=0.15,
            target_J=1.0,
            target_hx=0.5,
            target_hz=0.2,
            measure_every=1
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        engine.shutdown()
    finally:
        engine.shutdown()
