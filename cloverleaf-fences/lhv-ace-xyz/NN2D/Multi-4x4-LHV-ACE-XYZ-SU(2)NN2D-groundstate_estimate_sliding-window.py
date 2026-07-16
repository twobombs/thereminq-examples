# -*- coding: us-ascii -*-
# 16-Qubit 4x4 Flat-Tile Macroscopic Grid Annealing (16 Patches, 256 Qubits Total)
# High-Throughput Planar Engine with Statistical Variance Injection
#
# REVISION 88-F - 4x4 FLAT TILE VARIANT (of Rev 88)
#
# CHANGES (Rev 88-F):
# - GEOMETRY: Per-patch lattice reduced from 3x3x3 (27q) to a flat 4x4 (16q)
#   square lattice. Intra-patch edges are 2D nearest-neighbor only (24 edges).
# - BOUNDARIES: Faces reduced to {+X, -X, +Y, -Y} (4 qubits each; the 12
#   perimeter qubits carry boundary coupling, the interior 2x2 does not).
#   Corner qubits belong to two faces, as before.
# - MACRO GRID: 4x4 patch plane (16 patches, 256 qubits). Inter-patch
#   coupling is planar; all Z-direction neighbor logic removed.
# - MEMORY: A 16-qubit statevector is 2^16 amplitudes: 512 KiB at fp32,
#   256 KiB at fp16 (FPPOW=4 build). All 16 patches are trivially
#   VRAM-resident; GTT/PCIe paging is a non-issue at this scale, so
#   QRACK_MAX_ALLOC_MB is left generous but is no longer load-bearing.
#
# Inherited from Rev 88:
# - STOCHASTIC SCALING: scale = sqrt(dt / effective_shots), no measure_every.
# Inherited from Rev 87:
# - CONTINUOUS COUPLING: kick payloads persist across non-measure steps.
# - INTEGRATION SCALING: no m_every multiplier in apply_kicks.
# - FIDELITY WARNING: one-time stderr alert for missing fidelity binding.

import os
import sys
import gc
import csv
import json
import time
import math
import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Dict, Any

# --- GLOBAL CONFIGURATION ---
GRID_X, GRID_Y = 4, 4                      # macroscopic patch plane
TOTAL_PATCHES = GRID_X * GRID_Y            # 16
QUBITS_PER_PATCH = 16                      # flat 4x4 tile

# Topography tuning for raw statevectors
GPUS_AVAILABLE = 1
WORKERS_PER_GPU = 4  # 16q patches are cheap; 4 workers x 4 patches each
TOTAL_WORKERS = GPUS_AVAILABLE * WORKERS_PER_GPU

# =====================================================================
# ENVIRONMENT - set before pyqrack import
# =====================================================================
os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"


# =====================================================================
# PURE FUNCTIONS (Math & Topology)
# =====================================================================
def generate_16q_flat_tile() -> Tuple[List[Tuple[int, int]], Dict[str, List[int]]]:
    """4x4 planar square lattice. idx = x * ly + y (row-major in x)."""
    lx, ly = 4, 4
    edges: List[Tuple[int, int]] = []
    boundaries: Dict[str, List[int]] = {
        "+X": [], "-X": [], "+Y": [], "-Y": []
    }

    for x in range(lx):
        for y in range(ly):
            idx = x * ly + y
            if x < lx - 1: edges.append((idx, (x + 1) * ly + y))
            if y < ly - 1: edges.append((idx, x * ly + (y + 1)))

            if x == 0: boundaries["-X"].append(idx)
            if x == lx - 1: boundaries["+X"].append(idx)
            if y == 0: boundaries["-Y"].append(idx)
            if y == ly - 1: boundaries["+Y"].append(idx)

    return edges, boundaries


# =====================================================================
# WORKER PROCESS LOGIC
# =====================================================================
def gpu_worker_process(
    rank: int,
    workers_per_gpu: int,
    assigned_patches: List[int],
    conn: mp.connection.Connection,
    dt: float,
    total_steps: int,
    initial_hx: float,
    target_J: float,
    target_hx: float,
    target_hz: float,
    measure_every: int
) -> None:
    os.environ["PYQRACK_SHARED_LIB_PATH"] = "/usr/local/lib/qrack/libqrack_pinvoke.so"
    os.environ["OCL_ICD_PLATFORM_SORT"] = "none"

    # Map multiple ranks to the same physical GPU device index
    physical_gpu_index = rank // workers_per_gpu
    os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(physical_gpu_index)

    # Bind QPager to the assigned device to enable driver-level PCIe paging
    os.environ["QRACK_QPAGER_DEVICES"] = str(physical_gpu_index)
    os.environ["QRACK_QUNITMULTI_DEVICES"] = str(physical_gpu_index)

    # 16q statevectors are tiny (<=512 KiB fp32 / 256 KiB fp16); cap is
    # retained for hygiene but is not a constraint at this tile size.
    alloc_mb = 64000 // workers_per_gpu
    os.environ["QRACK_MAX_ALLOC_MB"] = str(alloc_mb)
    os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"

    import pyqrack
    from pyqrack import QrackSimulator

    sims = {}

    try:
        # --- PAULI CODE AUTODETECT ---
        _THRESH = 0.5

        # Z Probe
        _probe_z = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
        vals0_z = {}
        for _code in range(8):
            try: vals0_z[_code] = _probe_z.pauli_expectation([0], [_code])
            except Exception: pass
        _probe_z.x(0)
        PZ, SIGN_Z = None, None
        for _code, v0 in vals0_z.items():
            try: v1 = _probe_z.pauli_expectation([0], [_code])
            except Exception: continue
            if abs(v0) > _THRESH and abs(v1) > _THRESH and (v0 * v1) < 0:
                PZ = _code
                SIGN_Z = 1.0 if v0 > 0 else -1.0
                break
        if PZ is None:
            raise RuntimeError("Fatal: could not autodetect PZ code")
        del _probe_z

        # X Probe
        _probe_x = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
        _probe_x.h(0)
        vals0_x = {}
        for _code in range(8):
            if _code == PZ: continue
            try: vals0_x[_code] = _probe_x.pauli_expectation([0], [_code])
            except Exception: pass
        _probe_x.z(0)
        PX, SIGN_X = None, None
        for _code, v0 in vals0_x.items():
            try: v1 = _probe_x.pauli_expectation([0], [_code])
            except Exception: continue
            if abs(v0) > _THRESH and abs(v1) > _THRESH and (v0 * v1) < 0:
                PX = _code
                SIGN_X = 1.0 if v0 > 0 else -1.0
                break
        if PX is None:
            raise RuntimeError("Fatal: could not autodetect PX code")
        del _probe_x

        # Y Probe
        _probe_y = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
        _c, _s = math.cos(math.pi / 4.0), math.sin(math.pi / 4.0)
        _probe_y.mtrx([complex(_c, 0), complex(0, _s), complex(0, _s), complex(_c, 0)], 0)
        vals0_y = {}
        for _code in range(8):
            if _code in (PX, PZ): continue
            try: vals0_y[_code] = _probe_y.pauli_expectation([0], [_code])
            except Exception: pass
        _probe_y.z(0)
        PY, SIGN_Y = None, None
        for _code, v0 in vals0_y.items():
            try: v1 = _probe_y.pauli_expectation([0], [_code])
            except Exception: continue
            if abs(v0) > _THRESH and abs(v1) > _THRESH and (v0 * v1) < 0:
                PY = _code
                SIGN_Y = 1.0 if v0 > 0 else -1.0
                break
        if PY is None:
            raise RuntimeError("Fatal: could not autodetect PY code")
        del _probe_y
        # ----------------------------

        # --- ANGLE CONVENTION AUTODETECT ---
        _sim_mag = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
        _sim_mag.r(PX, math.pi, 0)
        mag_check = _sim_mag.pauli_expectation([0], [PZ])
        _corrected = SIGN_Z * mag_check
        if abs(_corrected + 1.0) < 0.1:
            ANGLE_SCALE = 1.0
        elif abs(_corrected - 1.0) < 0.1:
            ANGLE_SCALE = 0.5
        else:
            raise RuntimeError(
                f"Fatal: r(PX,pi) returned ambiguous SIGN_Z*<Z> = {_corrected:.6f}; "
                f"expected ~+1.0 or ~-1.0"
            )
        del _sim_mag
        # -------------------------------------------------------

        def apply_h(sim: QrackSimulator, q: int) -> None: 
            sim.h(q)

        def apply_rx(sim: QrackSimulator, theta: float, q: int) -> None:
            sim.r(PX, float(theta) * ANGLE_SCALE, q)

        def apply_ry(sim: QrackSimulator, theta: float, q: int) -> None:
            sim.r(PY, float(theta) * ANGLE_SCALE, q)

        def apply_rz(sim: QrackSimulator, theta: float, q: int) -> None:
            sim.r(PZ, float(theta) * ANGLE_SCALE, q)

        def apply_zz(sim: QrackSimulator, theta: float, q1: int, q2: int) -> None:
            sim.mcx([q1], q2)
            apply_rz(sim, 2.0 * theta, q2)
            sim.mcx([q1], q2)

        def trotter_step_body(sim: QrackSimulator, num_qubits: int, intra_edges: List[Tuple[int, int]], 
                              J: float, hx: float, hz: float, dt_local: float) -> None:
            dt_half = dt_local / 2.0

            theta_x  = -2.0 * hx * dt_half
            theta_z  = -2.0 * hz * dt_local
            theta_zz = -J * dt_local

            for q in range(num_qubits): apply_rx(sim, theta_x, q)
            for q in range(num_qubits): apply_rz(sim, theta_z, q)
            for q1, q2 in intra_edges: apply_zz(sim, theta_zz, q1, q2)
            for q in range(num_qubits): apply_rx(sim, theta_x, q)

        def z_means(sim: QrackSimulator, qubits: List[int]) -> np.ndarray:
            return np.array([SIGN_Z * float(sim.pauli_expectation([q], [PZ])) for q in qubits])

        def x_means(sim: QrackSimulator, qubits: List[int]) -> np.ndarray:
            return np.array([SIGN_X * float(sim.pauli_expectation([q], [PX])) for q in qubits])

        def y_means(sim: QrackSimulator, qubits: List[int]) -> np.ndarray:
            return np.array([SIGN_Y * float(sim.pauli_expectation([q], [PY])) for q in qubits])

        def zz_means_meanfield(z_exp: np.ndarray, edges: List[Tuple[int, int]]) -> np.ndarray:
            return np.array([z_exp[q1] * z_exp[q2] for q1, q2 in edges])

        def apply_kicks(sim: QrackSimulator, kicks: Dict[int, Tuple[float, float, float]], dt_local: float) -> None:
            if not kicks: return
            for raw_q, (kx, ky, kz) in kicks.items():
                q = int(raw_q)

                # Continuous evolution across all steps: m_every multiplier removed
                coef = -2.0 * dt_local

                theta_x = kx * coef
                theta_y = ky * coef
                theta_z = kz * coef

                if abs(theta_x) > 1e-12: apply_rx(sim, theta_x, q)
                if abs(theta_y) > 1e-12: apply_ry(sim, theta_y, q)
                if abs(theta_z) > 1e-12: apply_rz(sim, theta_z, q)

        intra_edges, boundaries = generate_16q_flat_tile()

        for p in assigned_patches:
            sim = QrackSimulator(
                qubit_count=QUBITS_PER_PATCH,
                is_binary_decision_tree=False,
                is_stabilizer_hybrid=False,
                is_gpu=True,
            )
            for q in range(QUBITS_PER_PATCH): apply_h(sim, q)
            sims[p] = sim

            # --- ALLOCATION SMOKE TEST (kept for parity; trivial at 16q) ---
            try:
                _ = sim.pauli_expectation([0], [PZ])
            except Exception as e:
                raise RuntimeError(f"Fatal: GPU allocation failed on patch {p}. Driver error: {e}")

        kick_payloads = {p: {} for p in assigned_patches}
        _warned_fidelity = False

        for t in range(total_steps):
            s = t / max(1, (total_steps - 1))
            current_hx = (1.0 - s) * initial_hx + s * target_hx
            current_J  = s * target_J
            current_hz = s * target_hz
            is_measure = (t % measure_every == 0) or (t == total_steps - 1)

            patch_data_to_master = {}

            for p in assigned_patches:
                sim = sims[p]
                if kick_payloads[p]:
                    apply_kicks(sim, kick_payloads[p], dt)

                t_start_trotter = time.perf_counter()
                trotter_step_body(sim, QUBITS_PER_PATCH, intra_edges,
                                  current_J, current_hx, current_hz, dt)
                t_lat_trotter = time.perf_counter() - t_start_trotter

                t_lat_tomo = 0.0
                if is_measure:
                    t_start_tomo = time.perf_counter()
                    all_q = list(range(QUBITS_PER_PATCH))
                    state = {
                        "Z": z_means(sim, all_q),
                        "X": x_means(sim, all_q),
                        "Y": y_means(sim, all_q),
                    }
                    zz_exp = zz_means_meanfield(state["Z"], intra_edges)
                    bulk_e = (-current_hz * float(np.sum(state["Z"]))
                              - current_J  * float(np.sum(zz_exp))
                              - current_hx * float(np.sum(state["X"])))
                    t_lat_tomo = time.perf_counter() - t_start_tomo

                    try:
                        fidelity = float(sim.get_unitary_fidelity())
                    except AttributeError:
                        fidelity = 1.0  # Fallback if API lacks binding
                        if not _warned_fidelity:
                            print(f"[Worker {rank}] Warning: sim.get_unitary_fidelity() not found. Upgrade PyQrack for true fidelity tracking.", file=sys.stderr)
                            _warned_fidelity = True

                    patch_data_to_master[p] = {
                        "state": state,
                        "meanfield_bulk_energy": bulk_e,
                        "lat_trotter_ms": t_lat_trotter * 1000.0,
                        "lat_tomo_ms": t_lat_tomo * 1000.0,
                        "unitary_fidelity": fidelity
                    }

            if is_measure:
                conn.send(patch_data_to_master)
                kick_payloads = conn.recv()

            # Kick payloads are intentionally retained across non-measure
            # intermediate steps (continuous boundary field, Rev 87).

    finally:
        for p in list(sims.keys()):
            _s = sims.pop(p)
            del _s
        sims.clear()
        gc.collect()
        conn.close()


# =====================================================================
# MASTER ORCHESTRATOR
# =====================================================================
class MultiGpuHadronEngine:
    def __init__(self, master_seed: int = 1337) -> None:
        self.master_seed = master_seed
        self.intra_edges, self.boundaries = generate_16q_flat_tile()
        self.all_boundary_qubits = sorted(set(q for face in self.boundaries.values() for q in face))
        self._bq_to_idx = {q: i for i, q in enumerate(self.all_boundary_qubits)}
        self._bq_arr    = np.array(self.all_boundary_qubits, dtype=np.intp)

        # Planar macroscopic layout: patch index <-> (x, y)
        self.patch_coords = {}
        idx = 0
        for x in range(GRID_X):
            for y in range(GRID_Y):
                self.patch_coords[idx] = (x, y)
                idx += 1
        self.coord_to_patch = {v: k for k, v in self.patch_coords.items()}

        self.lattice_history  = []
        self.energy_csv       = "meanfield_ground_state_energy_curve_multi.csv"
        self.profiles_csv     = "boundary_profiles_multi.csv"
        self.state_dump_file  = "macroscopic_lattice_states.npy"
        self.config_file      = "lattice_config.json"

        self._init_files()

        self.worker_assignments = [[] for _ in range(TOTAL_WORKERS)]
        for i in range(TOTAL_PATCHES):
            self.worker_assignments[i % TOTAL_WORKERS].append(i)

    def _init_files(self) -> None:
        try:
            with open(self.config_file, 'w') as f:
                json.dump({"grid_x": GRID_X, "grid_y": GRID_Y, "grid_z": 1,
                           "num_patches": TOTAL_PATCHES,
                           "qubits_per_patch": QUBITS_PER_PATCH,
                           "tile_geometry": "4x4_flat"}, f)
            with open(self.energy_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Anneal_Percent", "MeanField_Bulk_Energy",
                    "MeanField_Boundary_Energy", "MeanField_Total_Energy",
                    "Min_Unitary_Fidelity"
                ]).writeheader()
            with open(self.profiles_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Patch", "Face", "X_mean", "Y_mean", "Z_mean"
                ]).writeheader()
        except Exception as e:
            print(f"[CSV] Warning: Setup configuration write failed: {e}", file=sys.stderr)

    def _log_csvs(self, step: int, anneal: float, bulk: float, bound: float, total: float, min_fidelity: float, patch_profiles: Dict[int, Any]) -> None:
        try:
            with open(self.energy_csv, mode='a', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Anneal_Percent", "MeanField_Bulk_Energy",
                    "MeanField_Boundary_Energy", "MeanField_Total_Energy",
                    "Min_Unitary_Fidelity"
                ]).writerow({"Step": step, "Anneal_Percent": anneal,
                             "MeanField_Bulk_Energy": bulk, "MeanField_Boundary_Energy": bound,
                             "MeanField_Total_Energy": total,
                             "Min_Unitary_Fidelity": min_fidelity})

            with open(self.profiles_csv, mode='a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=[
                    "Step", "Patch", "Face", "X_mean", "Y_mean", "Z_mean"
                ])
                for p, prof in patch_profiles.items():
                    for face_name, face_qubits in self.boundaries.items():
                        if not face_qubits: continue
                        xm = float(np.mean([prof["means"]["X"][self._bq_to_idx[q]] for q in face_qubits]))
                        ym = float(np.mean([prof["means"]["Y"][self._bq_to_idx[q]] for q in face_qubits]))
                        zm = float(np.mean([prof["means"]["Z"][self._bq_to_idx[q]] for q in face_qubits]))
                        w.writerow({"Step": step, "Patch": p, "Face": face_name,
                                    "X_mean": xm, "Y_mean": ym, "Z_mean": zm})
        except Exception as e:
            print(f"[CSV] Warning: Log write failed: {e}", file=sys.stderr)

    def run(self, total_steps: int, dt: float, initial_hx: float, target_g_face: float,
            target_J: float, target_hx: float, target_hz: float,
            measure_every: int = 1, effective_shots: float = 512.0) -> None:

        if total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        if measure_every < 1:
            raise ValueError("measure_every must be a positive integer")

        total_qubits = TOTAL_PATCHES * QUBITS_PER_PATCH
        print(f"[Engine] {TOTAL_PATCHES} flat patches ({GRID_X}x{GRID_Y}), {total_qubits} qubits, "
              f"{GPUS_AVAILABLE} GPUs ({WORKERS_PER_GPU} workers/GPU), {total_steps} steps")

        active_ranks = [r for r in range(TOTAL_WORKERS) if self.worker_assignments[r]]

        workers = []
        pipes   = []

        for rank in active_ranks:
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=gpu_worker_process,
                args=(rank, WORKERS_PER_GPU, self.worker_assignments[rank], child_conn,
                      dt, total_steps, initial_hx, target_J, target_hx, target_hz, measure_every)
            )
            p.start()
            child_conn.close()
            workers.append(p)
            pipes.append(parent_conn)

        try:
            for t in range(total_steps):
                s = t / max(1, (total_steps - 1))
                current_g_face = s * target_g_face
                is_measure = (t % measure_every == 0) or (t == total_steps - 1)

                if not is_measure:
                    continue

                t0 = time.perf_counter()

                # --- GATHER ---
                patch_full_states = {}
                bulk_energy = 0.0
                max_lat_trotter = 0.0
                max_lat_tomo = 0.0
                min_fidelity = 1.0

                for conn in pipes:
                    try:
                        data = conn.recv()
                    except EOFError:
                        raise RuntimeError("Worker IPC connection lost.")
                    for p, payload in data.items():
                        patch_full_states[p] = payload["state"]
                        bulk_energy += payload["meanfield_bulk_energy"]
                        max_lat_trotter = max(max_lat_trotter, payload["lat_trotter_ms"])
                        max_lat_tomo = max(max_lat_tomo, payload["lat_tomo_ms"])
                        min_fidelity = min(min_fidelity, payload.get("unitary_fidelity", 1.0))

                if len(patch_full_states) != TOTAL_PATCHES:
                    raise RuntimeError(
                        f"Fatal: IPC gather incomplete. "
                        f"Expected {TOTAL_PATCHES} patches, got {len(patch_full_states)}."
                    )

                # --- BUILD PROFILES ---
                step_state = np.zeros((TOTAL_PATCHES, QUBITS_PER_PATCH, 3))
                patch_profiles = {}
                bq = self._bq_arr

                for p, state in patch_full_states.items():
                    step_state[p, :, 0] = state["X"]
                    step_state[p, :, 1] = state["Y"]
                    step_state[p, :, 2] = state["Z"]
                    patch_profiles[p] = {
                        "means": {
                            "X": state["X"][bq],
                            "Y": state["Y"][bq],
                            "Z": state["Z"][bq],
                        },
                        "vars": {
                            "X": np.clip(1.0 - state["X"][bq]**2, 0.0, 1.0),
                            "Y": np.clip(1.0 - state["Y"][bq]**2, 0.0, 1.0),
                            "Z": np.clip(1.0 - state["Z"][bq]**2, 0.0, 1.0),
                        }
                    }

                self.lattice_history.append(step_state.copy())

                if len(self.lattice_history) % 10 == 0:
                    try:
                        np.save(self.state_dump_file, np.array(self.lattice_history))
                    except Exception as e:
                        print(f"[Checkpoint] Warning: Failed to save: {e}", file=sys.stderr)

                # --- COMPUTE KICKS & BOUNDARY ENERGY ---
                next_kick_payloads = {p: {} for p in range(TOTAL_PATCHES)}
                macroscopic_boundary_energy = 0.0

                scale = np.sqrt(dt / effective_shots)
                stochastic_noise = {}
                n_b = len(self.all_boundary_qubits)

                for p in range(TOTAL_PATCHES):
                    prof = patch_profiles[p]
                    rng_p = np.random.default_rng([self.master_seed, t, p])

                    xn = rng_p.normal(0.0, 1.0, n_b) * np.sqrt(prof["vars"]["X"]) * scale
                    yn = rng_p.normal(0.0, 1.0, n_b) * np.sqrt(prof["vars"]["Y"]) * scale
                    zn = rng_p.normal(0.0, 1.0, n_b) * np.sqrt(prof["vars"]["Z"]) * scale

                    stochastic_noise[p] = {
                        q: (xn[i], yn[i], zn[i])
                        for i, q in enumerate(self.all_boundary_qubits)
                    }

                for p1, coord1 in self.patch_coords.items():
                    x1, y1 = coord1
                    neighbors = {
                        "+X": (x1 + 1, y1), "-X": (x1 - 1, y1),
                        "+Y": (x1, y1 + 1), "-Y": (x1, y1 - 1),
                    }

                    for dir1, coord2 in neighbors.items():
                        p2 = self.coord_to_patch.get(coord2)
                        if p2 is None or p1 >= p2: continue

                        dir2    = dir1.replace("+", "temp").replace("-", "+").replace("temp", "-")
                        face1_q = self.boundaries[dir1]
                        face2_q = self.boundaries[dir2]
                        prof1, noise1 = patch_profiles[p1], stochastic_noise[p1]
                        prof2, noise2 = patch_profiles[p2], stochastic_noise[p2]

                        ax2 = np.mean([prof2["means"]["X"][self._bq_to_idx[q]] + noise2[q][0] for q in face2_q])
                        ay2 = np.mean([prof2["means"]["Y"][self._bq_to_idx[q]] + noise2[q][1] for q in face2_q])
                        az2 = np.mean([prof2["means"]["Z"][self._bq_to_idx[q]] + noise2[q][2] for q in face2_q])

                        ax1 = np.mean([prof1["means"]["X"][self._bq_to_idx[q]] + noise1[q][0] for q in face1_q])
                        ay1 = np.mean([prof1["means"]["Y"][self._bq_to_idx[q]] + noise1[q][1] for q in face1_q])
                        az1 = np.mean([prof1["means"]["Z"][self._bq_to_idx[q]] + noise1[q][2] for q in face1_q])

                        macroscopic_boundary_energy += (
                            -current_g_face
                            * (ax1*ax2 + ay1*ay2 + az1*az2)
                            * ((len(face1_q) + len(face2_q)) / 2.0)
                        )

                        for q1f in face1_q:
                            k = next_kick_payloads[p1].get(q1f, (0., 0., 0.))
                            next_kick_payloads[p1][q1f] = (
                                k[0] + current_g_face * ax2,
                                k[1] + current_g_face * ay2,
                                k[2] + current_g_face * az2,
                            )
                        for q2f in face2_q:
                            k = next_kick_payloads[p2].get(q2f, (0., 0., 0.))
                            next_kick_payloads[p2][q2f] = (
                                k[0] + current_g_face * ax1,
                                k[1] + current_g_face * ay1,
                                k[2] + current_g_face * az1,
                            )

                total_energy = bulk_energy + macroscopic_boundary_energy
                print(f"Step {t:03d} | E: {total_energy:+.4f} | Lat(Trot/Tomo): {max_lat_trotter:5.1f}/{max_lat_tomo:5.1f}ms | Fid: {min_fidelity:.5f} | {time.perf_counter() - t0:.2f}s")
                self._log_csvs(t, s * 100, bulk_energy,
                               macroscopic_boundary_energy, total_energy, min_fidelity, patch_profiles)

                # --- SCATTER ---
                for i, w_rank in enumerate(active_ranks):
                    worker_payload = {p: next_kick_payloads[p]
                                      for p in self.worker_assignments[w_rank]}
                    pipes[i].send(worker_payload)

        finally:
            for conn in pipes:
                try: conn.close()
                except Exception: pass

            if self.lattice_history:
                try:
                    np.save(self.state_dump_file, np.array(self.lattice_history))
                    print(f"\n[Master] Dumped history matrix to {self.state_dump_file}")
                except Exception as e:
                    print(f"\n[Master] Failed to save lattice history: {e}", file=sys.stderr)

            for p in workers:
                p.join(timeout=15)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=3)
                    if p.is_alive():
                        try: p.kill()
                        except Exception: pass


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    engine = MultiGpuHadronEngine(master_seed=1337)
    try:
        engine.run(
            total_steps=100,
            dt=0.04,
            initial_hx=3.0,
            target_g_face=0.15,
            target_J=1.0,
            target_hx=0.5,
            target_hz=0.2,
            measure_every=1,
            effective_shots=512.0
        )
    except KeyboardInterrupt:
        pass
