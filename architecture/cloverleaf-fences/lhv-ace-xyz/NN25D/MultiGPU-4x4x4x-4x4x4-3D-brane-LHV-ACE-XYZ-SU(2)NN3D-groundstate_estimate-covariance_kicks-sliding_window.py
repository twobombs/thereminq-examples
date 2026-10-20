# -*- coding: us-ascii -*-
# 16-Qubit 4x4 Brane Tiles -> 4x4x4 Brane-Stack Blocks -> 4x4x4 Block Lattice
# (256 Patches, 4096 Qubits Total)
# Layered Planar Engine with Site-Resolved Inter-Brane AND Inter-Block Coupling
#
# REVISION 88-I - BLOCK LATTICE VARIANT
#
# CHANGES (Rev 88-I):
# 1. FIXED BUG: Moved ket_cache checkpointing to AFTER apply_kicks.
#    Previously, a mid-Trotter context loss would resurrect the state to
#    its pre-kick snapshot, dropping the continuous kicks for that step.
# 2. PERF: Removed the redundant outer pauli_expectation drain from the
#    Trotter CHUNK=1 loop, as apply_zz is now self-draining. Saves ~18k
#    drains per step.
# 3. MATH: Symmetrized the hz term in the Strang splitting to achieve true
#    O(dt^3) per-step error: Rx(dt/2) -> Rz(dt/2) -> ZZ(dt) -> Rz(dt/2) -> Rx(dt/2).

import os
import sys
import gc
import csv
import json
import time
import math
import numpy as np
import multiprocessing as mp
import multiprocessing.connection
from typing import List, Tuple, Dict, Any

# --- GLOBAL CONFIGURATION ---
BLOCK_GRID_X = 4
BLOCK_GRID_Y = 4
BLOCK_GRID_Z = 4
BRANES_PER_BLOCK = 4
GLOBAL_Z = BLOCK_GRID_Z * BRANES_PER_BLOCK  # 16

TOTAL_PATCHES = BLOCK_GRID_X * BLOCK_GRID_Y * GLOBAL_Z  # 256
QUBITS_PER_PATCH = 16
TOTAL_QUBITS = TOTAL_PATCHES * QUBITS_PER_PATCH          # 4096

PERIODIC_X = False
PERIODIC_Y = False
PERIODIC_Z = False

GPUS_AVAILABLE = 6
WORKERS_PER_GPU = 24
TOTAL_WORKERS = GPUS_AVAILABLE * WORKERS_PER_GPU

RING_RESET_LAT_THRESHOLD_MS = 5.0
INTER_PATCH_YIELD_EVERY = 16
KICK_THETA_THRESHOLD = 1e-6

# =====================================================================
# ENVIRONMENT - set before pyqrack import
# =====================================================================
os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"


# =====================================================================
# PURE FUNCTIONS (Math & Topology)
# =====================================================================
def generate_16q_brane_tile() -> Tuple[List[Tuple[int, int]], List[int]]:
    lx, ly = 4, 4
    edges: List[Tuple[int, int]] = []

    for x in range(lx):
        for y in range(ly):
            idx = x * ly + y
            if x < lx - 1: edges.append((idx, (x + 1) * ly + y))
            if y < ly - 1: edges.append((idx, x * ly + (y + 1)))

    brane_sites = list(range(lx * ly))
    return edges, brane_sites


def patch_id(tx: int, ty: int, z: int) -> int:
    return (tx * BLOCK_GRID_Y + ty) * GLOBAL_Z + z


def patch_coords(p: int) -> Tuple[int, int, int]:
    z = p % GLOBAL_Z
    rest = p // GLOBAL_Z
    return rest // BLOCK_GRID_Y, rest % BLOCK_GRID_Y, z


def build_interfaces() -> List[Tuple[int, int, np.ndarray, np.ndarray, str]]:
    z_i1 = np.arange(QUBITS_PER_PATCH)
    z_i2 = z_i1
    x_i1 = np.array([12 + y for y in range(4)])
    x_i2 = np.array([y for y in range(4)])
    y_i1 = np.array([x * 4 + 3 for x in range(4)])
    y_i2 = np.array([x * 4 for x in range(4)])

    interfaces: List[Tuple[int, int, np.ndarray, np.ndarray, str]] = []

    for tx in range(BLOCK_GRID_X):
        for ty in range(BLOCK_GRID_Y):
            for z in range(GLOBAL_Z):
                p1 = patch_id(tx, ty, z)

                # --- Z neighbor ---
                if z < GLOBAL_Z - 1:
                    kind = "Z_INTRA" if (z + 1) % BRANES_PER_BLOCK != 0 else "Z_INTER"
                    interfaces.append((p1, patch_id(tx, ty, z + 1), z_i1, z_i2, kind))
                elif PERIODIC_Z and GLOBAL_Z > 2:
                    interfaces.append((p1, patch_id(tx, ty, 0), z_i1, z_i2, "Z_INTER"))

                # --- X neighbor ---
                if tx < BLOCK_GRID_X - 1:
                    interfaces.append((p1, patch_id(tx + 1, ty, z), x_i1, x_i2, "XY"))
                elif PERIODIC_X and BLOCK_GRID_X > 2:
                    interfaces.append((p1, patch_id(0, ty, z), x_i1, x_i2, "XY"))

                # --- Y neighbor ---
                if ty < BLOCK_GRID_Y - 1:
                    interfaces.append((p1, patch_id(tx, ty + 1, z), y_i1, y_i2, "XY"))
                elif PERIODIC_Y and BLOCK_GRID_Y > 2:
                    interfaces.append((p1, patch_id(tx, 0, z), y_i1, y_i2, "XY"))

    return interfaces


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
    os.environ["AMD_OPENCL_FORCE_COMPUTE_QUEUE"] = "1"

    physical_gpu_index = rank // workers_per_gpu
    os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(physical_gpu_index)
    os.environ["QRACK_QPAGER_DEVICES"] = str(physical_gpu_index)
    os.environ["QRACK_QUNITMULTI_DEVICES"] = str(physical_gpu_index)

    alloc_mb = 7500 // max(1, workers_per_gpu)
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
        gc.collect()

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
        gc.collect()

        # Y Probe
        try:
            _s_test = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
            _s_test.s(0)
            del _s_test
            gc.collect()
            _USE_S_GATE = True
        except Exception:
            try: del _s_test
            except NameError: pass
            gc.collect()
            _USE_S_GATE = False

        _probe_y = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
        _probe_y.h(0)
        if _USE_S_GATE:
            _probe_y.s(0)
        else:
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
        gc.collect()
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
                "Fatal: r(PX,pi) returned ambiguous SIGN_Z*<Z> = " +
                "{:.6f}".format(_corrected) +
                "; expected ~+1.0 or ~-1.0"
            )
        del _sim_mag
        gc.collect()
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
            _ = sim.pauli_expectation([0], [PZ])
            apply_rz(sim, 2.0 * theta, q2)
            _ = sim.pauli_expectation([0], [PZ])
            sim.mcx([q1], q2)
            _ = sim.pauli_expectation([0], [PZ])

        def trotter_step_body(sim: QrackSimulator, num_qubits: int,
                              intra_edges: List[Tuple[int, int]],
                              J: float, hx: float, hz: float,
                              dt_local: float) -> None:
            dt_half = dt_local / 2.0

            theta_x  = -2.0 * hx * dt_half
            theta_z  = -2.0 * hz * dt_half
            theta_zz = -J * dt_local

            for q in range(num_qubits): apply_rx(sim, theta_x, q)
            for q in range(num_qubits): apply_rz(sim, theta_z, q)

            for q1, q2 in intra_edges:
                apply_zz(sim, theta_zz, q1, q2)

            for q in range(num_qubits): apply_rz(sim, theta_z, q)
            for q in range(num_qubits): apply_rx(sim, theta_x, q)

        def z_means(sim: QrackSimulator, qubits: List[int]) -> np.ndarray:
            return np.array([SIGN_Z * float(sim.pauli_expectation([q], [PZ]))
                             for q in qubits])

        def x_means(sim: QrackSimulator, qubits: List[int]) -> np.ndarray:
            return np.array([SIGN_X * float(sim.pauli_expectation([q], [PX]))
                             for q in qubits])

        def y_means(sim: QrackSimulator, qubits: List[int]) -> np.ndarray:
            return np.array([SIGN_Y * float(sim.pauli_expectation([q], [PY]))
                             for q in qubits])

        def zz_means_meanfield(z_exp: np.ndarray,
                               edges: List[Tuple[int, int]]) -> np.ndarray:
            return np.array([z_exp[q1] * z_exp[q2] for q1, q2 in edges])

        def apply_kicks(sim: QrackSimulator,
                        kicks: Dict[int, Tuple[float, float, float]],
                        dt_local: float) -> None:
            if not kicks: return
            coef = -2.0 * dt_local
            items = list(kicks.items())

            KICK_CHUNK = 1
            for i in range(0, len(items), KICK_CHUNK):
                for raw_q, (kx, ky, kz) in items[i:i + KICK_CHUNK]:
                    q = int(raw_q)
                    theta_x = kx * coef
                    theta_y = ky * coef
                    theta_z = kz * coef
                    if abs(theta_x) > KICK_THETA_THRESHOLD: apply_rx(sim, theta_x, q)
                    if abs(theta_y) > KICK_THETA_THRESHOLD: apply_ry(sim, theta_y, q)
                    if abs(theta_z) > KICK_THETA_THRESHOLD: apply_rz(sim, theta_z, q)
                _ = sim.pauli_expectation([0], [PZ])

        _CONTEXT_LOST_STRINGS = (
            "context is lost",
            "CS has cancelled",
            "CL_OUT_OF_RESOURCES",
            "CL_INVALID_COMMAND_QUEUE",
        )

        def is_context_loss(exc: Exception) -> bool:
            msg = str(exc).lower()
            return any(s.lower() in msg for s in _CONTEXT_LOST_STRINGS)

        intra_edges, _brane_sites = generate_16q_brane_tile()

        ket_cache: Dict[int, np.ndarray] = {}

        def resurrect_sim(p: int) -> QrackSimulator:
            if p in sims:
                try:
                    _old = sims.pop(p)
                    del _old
                except Exception:
                    pass
            gc.collect()
            sim_new = QrackSimulator(
                qubit_count=QUBITS_PER_PATCH,
                is_binary_decision_tree=False,
                is_stabilizer_hybrid=False,
                is_gpu=True,
            )
            if p in ket_cache:
                try:
                    sim_new.in_ket(ket_cache[p].tolist())
                    print("[Worker " + str(rank) + "] patch " + str(p) +
                          " resurrected from ket checkpoint.", file=sys.stderr)
                except Exception as restore_err:
                    print("[Worker " + str(rank) + "] patch " + str(p) +
                          " ket restore failed (" + str(restore_err) +
                          "); reinitialising to |+>^16.", file=sys.stderr)
                    for q in range(QUBITS_PER_PATCH):
                        apply_h(sim_new, q)
            else:
                for q in range(QUBITS_PER_PATCH):
                    apply_h(sim_new, q)
                print("[Worker " + str(rank) + "] patch " + str(p) +
                      " resurrected to |+>^16 (no ket checkpoint).",
                      file=sys.stderr)
            sims[p] = sim_new
            return sim_new

        for p in assigned_patches:
            sim = QrackSimulator(
                qubit_count=QUBITS_PER_PATCH,
                is_binary_decision_tree=False,
                is_stabilizer_hybrid=False,
                is_gpu=True,
            )
            for q in range(QUBITS_PER_PATCH): apply_h(sim, q)
            sims[p] = sim

            try:
                _ = sim.pauli_expectation([0], [PZ])
            except Exception as e:
                raise RuntimeError(
                    "Fatal: GPU allocation failed on patch " + str(p) +
                    ". Driver error: " + str(e)
                )
            try:
                ket_cache[p] = np.array(sim.out_ket(), dtype=np.complex64)
            except Exception:
                pass

        kick_payloads = {p: {} for p in assigned_patches}
        _warned_fidelity = False

        for t in range(total_steps):
            s = t / max(1, (total_steps - 1))
            current_hx = (1.0 - s) * initial_hx + s * target_hx
            current_J  = s * target_J
            current_hz = s * target_hz
            is_measure = (t % measure_every == 0) or (t == total_steps - 1)

            patch_data_to_master = {}

            for p_idx, p in enumerate(assigned_patches):
                sim = sims[p]

                # 1. APPLY KICKS BEFORE CHECKPOINT
                if kick_payloads[p]:
                    try:
                        apply_kicks(sim, kick_payloads[p], dt)
                    except RuntimeError as _ke:
                        if is_context_loss(_ke):
                            print("[Worker " + str(rank) + "] step " + str(t) +
                                  " patch " + str(p) +
                                  " context lost in apply_kicks; resurrecting.",
                                  file=sys.stderr)
                            sim = resurrect_sim(p)
                            try:
                                apply_kicks(sim, kick_payloads[p], dt)
                            except Exception:
                                pass
                        else:
                            raise

                # 2. CHECKPOINT STATE AFTER KICKS, BEFORE TROTTER
                try:
                    ket_cache[p] = np.array(sim.out_ket(), dtype=np.complex64)
                except Exception:
                    pass

                t_start_trotter = time.perf_counter()
                try:
                    trotter_step_body(sim, QUBITS_PER_PATCH, intra_edges,
                                      current_J, current_hx, current_hz, dt)
                except RuntimeError as _te:
                    if is_context_loss(_te):
                        print("[Worker " + str(rank) + "] step " + str(t) +
                              " patch " + str(p) +
                              " context lost in trotter; resurrecting.",
                              file=sys.stderr)
                        sim = resurrect_sim(p)
                        try:
                            trotter_step_body(sim, QUBITS_PER_PATCH, intra_edges,
                                              current_J, current_hx, current_hz, dt)
                        except Exception:
                            pass
                    else:
                        raise

                try:
                    _ = sim.pauli_expectation([0], [PZ])
                except RuntimeError as _be:
                    if is_context_loss(_be):
                        sim = resurrect_sim(p)
                    else:
                        raise

                if (p_idx + 1) % INTER_PATCH_YIELD_EVERY == 0:
                    time.sleep(0)

                t_lat_trotter = time.perf_counter() - t_start_trotter

                t_lat_tomo = 0.0
                if is_measure:
                    t_start_tomo = time.perf_counter()
                    all_q = list(range(QUBITS_PER_PATCH))
                    try:
                        state = {
                            "Z": z_means(sim, all_q),
                            "X": x_means(sim, all_q),
                            "Y": y_means(sim, all_q),
                        }
                    except RuntimeError as _me:
                        if is_context_loss(_me):
                            print("[Worker " + str(rank) + "] step " + str(t) +
                                  " patch " + str(p) +
                                  " context lost in tomo; resurrecting.",
                                  file=sys.stderr)
                            sim = resurrect_sim(p)
                            state = {
                                "Z": z_means(sim, all_q),
                                "X": x_means(sim, all_q),
                                "Y": y_means(sim, all_q),
                            }
                        else:
                            raise
                    zz_exp = zz_means_meanfield(state["Z"], intra_edges)
                    bulk_e = (-current_hz * float(np.sum(state["Z"]))
                              - current_J  * float(np.sum(zz_exp))
                              - current_hx * float(np.sum(state["X"])))
                    t_lat_tomo = time.perf_counter() - t_start_tomo

                    try:
                        fidelity = float(sim.get_unitary_fidelity())
                    except (AttributeError, RuntimeError):
                        fidelity = 1.0
                        if not _warned_fidelity:
                            print(
                                "[Worker " + str(rank) + "] Warning: "
                                "get_unitary_fidelity() failed or not found. "
                                "Upgrade PyQrack.",
                                file=sys.stderr
                            )
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
        self.intra_edges, self.brane_sites = generate_16q_brane_tile()
        self.n_sites = len(self.brane_sites)

        self.patch_coords = {p: patch_coords(p) for p in range(TOTAL_PATCHES)}
        self.interfaces = build_interfaces()

        n_by_kind = {"Z_INTRA": 0, "Z_INTER": 0, "XY": 0}
        for _, _, _, _, kind in self.interfaces:
            n_by_kind[kind] += 1
        self.n_by_kind = n_by_kind

        self.lattice_history  = []
        self.energy_csv       = "meanfield_ground_state_energy_curve_multi.csv"
        self.profiles_csv     = "boundary_profiles_multi.csv"
        self.state_dump_file  = "macroscopic_lattice_states.npy"
        self.config_file      = "lattice_config.json"

        self._energy_fields = [
            "Step", "Anneal_Percent", "MeanField_Bulk_Energy",
            "E_Z_Intra", "E_Z_Inter", "E_XY",
            "MeanField_Boundary_Energy", "MeanField_Total_Energy",
            "Min_Unitary_Fidelity", "Ring_Reset"
        ]
        self._profile_fields = [
            "Step", "Patch", "Tx", "Ty", "Z", "Block_Z", "Layer",
            "Face", "X_mean", "Y_mean", "Z_mean"
        ]

        self._init_files()

        self.worker_assignments = [[] for _ in range(TOTAL_WORKERS)]
        for i in range(TOTAL_PATCHES):
            self.worker_assignments[i % TOTAL_WORKERS].append(i)

    def _init_files(self) -> None:
        try:
            with open(self.config_file, 'w') as f:
                json.dump({
                    "grid_x": BLOCK_GRID_X, "grid_y": BLOCK_GRID_Y, "grid_z": GLOBAL_Z,
                    "block_grid": [BLOCK_GRID_X, BLOCK_GRID_Y, BLOCK_GRID_Z],
                    "branes_per_block": BRANES_PER_BLOCK,
                    "num_patches": TOTAL_PATCHES,
                    "qubits_per_patch": QUBITS_PER_PATCH,
                    "total_qubits": TOTAL_QUBITS,
                    "tile_geometry": "4x4_brane_block_lattice",
                    "periodic": [PERIODIC_X, PERIODIC_Y, PERIODIC_Z],
                    "interfaces_by_kind": self.n_by_kind,
                    "patch_id_convention": "p = (tx * grid_y + ty) * grid_z + z",
                    "state_dump_shape": ["n_steps", "patch", "qubit", "XYZ"]
                }, f)
            with open(self.energy_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=self._energy_fields).writeheader()
            with open(self.profiles_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=self._profile_fields).writeheader()
        except Exception as e:
            print("[CSV] Warning: Setup configuration write failed: " + str(e),
                  file=sys.stderr)

    def _log_csvs(self, step: int, anneal: float, bulk: float,
                  e_by_kind: Dict[str, float], bound: float, total: float,
                  min_fidelity: float, patch_profiles: Dict[int, Any],
                  ring_reset: bool) -> None:
        try:
            with open(self.energy_csv, mode='a', newline='') as f:
                csv.DictWriter(f, fieldnames=self._energy_fields).writerow({
                    "Step": step, "Anneal_Percent": anneal,
                    "MeanField_Bulk_Energy": bulk,
                    "E_Z_Intra": e_by_kind["Z_INTRA"],
                    "E_Z_Inter": e_by_kind["Z_INTER"],
                    "E_XY": e_by_kind["XY"],
                    "MeanField_Boundary_Energy": bound,
                    "MeanField_Total_Energy": total,
                    "Min_Unitary_Fidelity": min_fidelity,
                    "Ring_Reset": 1 if ring_reset else 0
                })

            with open(self.profiles_csv, mode='a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self._profile_fields)
                for p, prof in patch_profiles.items():
                    tx, ty, z = self.patch_coords[p]
                    w.writerow({
                        "Step": step, "Patch": p,
                        "Tx": tx, "Ty": ty, "Z": z,
                        "Block_Z": z // BRANES_PER_BLOCK,
                        "Layer": z % BRANES_PER_BLOCK,
                        "Face": "BRANE",
                        "X_mean": float(np.mean(prof["means"]["X"])),
                        "Y_mean": float(np.mean(prof["means"]["Y"])),
                        "Z_mean": float(np.mean(prof["means"]["Z"]))
                    })
        except Exception as e:
            print("[CSV] Warning: Log write failed: " + str(e), file=sys.stderr)

    def run(self, total_steps: int, dt: float, initial_hx: float,
            target_g_intra_z: float, target_g_inter_z: float, target_g_xy: float,
            target_J: float, target_hx: float, target_hz: float,
            measure_every: int = 1, effective_shots: float = 512.0) -> None:

        if total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        if measure_every < 1:
            raise ValueError("measure_every must be a positive integer")

        print(
            "[Engine] " + str(BLOCK_GRID_X) + "x" + str(BLOCK_GRID_Y) + "x" +
            str(BLOCK_GRID_Z) + " block lattice x " + str(BRANES_PER_BLOCK) +
            " branes/block = " + str(TOTAL_PATCHES) + " patches, " +
            str(TOTAL_QUBITS) + " qubits | interfaces: " +
            str(self.n_by_kind["Z_INTRA"]) + " Z-intra, " +
            str(self.n_by_kind["Z_INTER"]) + " Z-inter, " +
            str(self.n_by_kind["XY"]) + " XY | " +
            str(GPUS_AVAILABLE) + " GPUs (" + str(WORKERS_PER_GPU) +
            " workers/GPU), " + str(total_steps) + " steps"
        )
        print(
            "[Engine] Rev 88-I: intra-ZZ drain (seq-delta=1), ket checkpoint "
            "+ sim resurrection on context loss, self-draining apply_zz / KICK_CHUNK=1, "
            "KICK_THETA_THRESHOLD=" + str(KICK_THETA_THRESHOLD) + ", "
            "QRACK_MAX_ALLOC_MB=7500/worker."
        )

        active_ranks = [r for r in range(TOTAL_WORKERS)
                        if self.worker_assignments[r]]

        workers = []
        pipes   = []

        for rank in active_ranks:
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=gpu_worker_process,
                args=(rank, WORKERS_PER_GPU, self.worker_assignments[rank],
                      child_conn, dt, total_steps, initial_hx,
                      target_J, target_hx, target_hz, measure_every)
            )
            p.start()
            child_conn.close()
            workers.append(p)
            pipes.append(parent_conn)

        try:
            for t in range(total_steps):
                s = t / max(1, (total_steps - 1))
                g_now = {
                    "Z_INTRA": s * target_g_intra_z,
                    "Z_INTER": s * target_g_inter_z,
                    "XY":      s * target_g_xy,
                }
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
                        max_lat_trotter = max(max_lat_trotter,
                                              payload["lat_trotter_ms"])
                        max_lat_tomo = max(max_lat_tomo,
                                          payload["lat_tomo_ms"])
                        min_fidelity = min(min_fidelity,
                                          payload.get("unitary_fidelity", 1.0))

                if len(patch_full_states) != TOTAL_PATCHES:
                    raise RuntimeError(
                        "Fatal: IPC gather incomplete. Expected " +
                        str(TOTAL_PATCHES) + " patches, got " +
                        str(len(patch_full_states)) + "."
                    )

                # --- BUILD PROFILES ---
                step_state = np.zeros((TOTAL_PATCHES, QUBITS_PER_PATCH, 3))
                patch_profiles = {}

                for p, state in patch_full_states.items():
                    step_state[p, :, 0] = state["X"]
                    step_state[p, :, 1] = state["Y"]
                    step_state[p, :, 2] = state["Z"]
                    patch_profiles[p] = {
                        "means": {
                            "X": state["X"].copy(),
                            "Y": state["Y"].copy(),
                            "Z": state["Z"].copy(),
                        },
                        "vars": {
                            "X": np.clip(1.0 - state["X"]**2, 0.0, 1.0),
                            "Y": np.clip(1.0 - state["Y"]**2, 0.0, 1.0),
                            "Z": np.clip(1.0 - state["Z"]**2, 0.0, 1.0),
                        }
                    }

                self.lattice_history.append(step_state.copy())

                if len(self.lattice_history) % 10 == 0:
                    try:
                        np.save(self.state_dump_file,
                                np.array(self.lattice_history))
                    except Exception as e:
                        print("[Checkpoint] Warning: Failed to save: " + str(e),
                              file=sys.stderr)

                ring_reset = max_lat_tomo > RING_RESET_LAT_THRESHOLD_MS

                # --- COMPUTE KICKS & INTERFACE ENERGY ---
                scale = np.sqrt(dt / effective_shots)
                n_s = self.n_sites
                AXES = ("X", "Y", "Z")

                noisy_field = {}
                for p in range(TOTAL_PATCHES):
                    prof = patch_profiles[p]
                    rng_p = np.random.default_rng([self.master_seed, t, p])
                    noisy_field[p] = {
                        ax: prof["means"][ax]
                            + rng_p.normal(0.0, 1.0, n_s)
                            * np.sqrt(prof["vars"][ax]) * scale
                        for ax in AXES
                    }

                kick_acc = {p: np.zeros((n_s, 3)) for p in range(TOTAL_PATCHES)}
                e_by_kind = {"Z_INTRA": 0.0, "Z_INTER": 0.0, "XY": 0.0}

                for p1, p2, i1, i2, kind in self.interfaces:
                    g = g_now[kind]
                    if g == 0.0:
                        continue
                    f1, f2 = noisy_field[p1], noisy_field[p2]
                    m1 = patch_profiles[p1]["means"]
                    m2 = patch_profiles[p2]["means"]

                    dot = 0.0
                    for a, ax in enumerate(AXES):
                        dot += float(np.sum(m1[ax][i1] * m2[ax][i2]))
                        kick_acc[p1][i1, a] += g * f2[ax][i2]
                        kick_acc[p2][i2, a] += g * f1[ax][i1]
                    e_by_kind[kind] += -g * dot

                macroscopic_boundary_energy = sum(e_by_kind.values())

                next_kick_payloads = {}
                _coef = -2.0 * dt
                _thresh = KICK_THETA_THRESHOLD
                for p in range(TOTAL_PATCHES):
                    acc = kick_acc[p]
                    payload = {}
                    for q in range(n_s):
                        ax_vals = acc[q]
                        if (abs(ax_vals[0] * _coef) > _thresh or
                                abs(ax_vals[1] * _coef) > _thresh or
                                abs(ax_vals[2] * _coef) > _thresh):
                            payload[q] = (float(ax_vals[0]),
                                          float(ax_vals[1]),
                                          float(ax_vals[2]))
                    next_kick_payloads[p] = payload

                total_energy = bulk_energy + macroscopic_boundary_energy
                reset_tag = "  RING_RESET" if ring_reset else ""
                print(
                    "Step {:03d} | E: {:+.4f} "
                    "(Zi {:+.3f} / Ze {:+.3f} / XY {:+.3f}) "
                    "| Lat(Trot/Tomo): {:5.1f}/{:5.1f}ms "
                    "| Fid: {:.5f} | {:.2f}s{}".format(
                        t, total_energy,
                        e_by_kind["Z_INTRA"], e_by_kind["Z_INTER"],
                        e_by_kind["XY"],
                        max_lat_trotter, max_lat_tomo,
                        min_fidelity,
                        time.perf_counter() - t0,
                        reset_tag
                    )
                )
                self._log_csvs(
                    t, s * 100, bulk_energy, e_by_kind,
                    macroscopic_boundary_energy, total_energy,
                    min_fidelity, patch_profiles, ring_reset
                )

                # --- SCATTER ---
                for i, w_rank in enumerate(active_ranks):
                    worker_payload = {
                        p: next_kick_payloads[p]
                        for p in self.worker_assignments[w_rank]
                    }
                    pipes[i].send(worker_payload)

        finally:
            for conn in pipes:
                try: conn.close()
                except Exception: pass

            if self.lattice_history:
                try:
                    np.save(self.state_dump_file,
                            np.array(self.lattice_history))
                    print("\n[Master] Dumped history matrix to " +
                          self.state_dump_file)
                except Exception as e:
                    print("\n[Master] Failed to save lattice history: " +
                          str(e), file=sys.stderr)

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
            target_g_intra_z=0.12,
            target_g_inter_z=0.06,
            target_g_xy=0.06,
            target_J=1.0,
            target_hx=0.5,
            target_hz=0.2,
            measure_every=1,
            effective_shots=512.0
        )
    except KeyboardInterrupt:
        pass
