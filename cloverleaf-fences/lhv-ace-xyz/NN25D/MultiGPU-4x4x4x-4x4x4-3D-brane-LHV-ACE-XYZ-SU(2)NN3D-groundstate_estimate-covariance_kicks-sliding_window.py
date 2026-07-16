# -*- coding: us-ascii -*-
# 16-Qubit 4x4 Brane Tiles -> 4x4x4 Brane-Stack Blocks -> 4x4x4 Block Lattice
# (256 Patches, 4096 Qubits Total)
# Layered Planar Engine with Site-Resolved Inter-Brane AND Inter-Block Coupling
#
# REVISION 88-C - BLOCK LATTICE VARIANT (of Rev 88-B)
#
# CHANGES (Rev 88-C):
# - MACRO GEOMETRY: The single 1x1x4 brane stack of Rev 88-B (one 4x4x4
#   64-qubit "block") is promoted to the unit cell. A 4x4x4 lattice of
#   these blocks is built: 64 blocks x 4 branes/block = 256 patches,
#   256 x 16 = 4096 qubits. Equivalently, the full system is a 16x16x16
#   qubit cube tiled into 4x4x1 branes, with exact 16q statevector
#   dynamics inside each brane and site-resolved mean-field exchange
#   across every tiling seam.
# - PATCH ADDRESSING: A patch (brane) is addressed by (tx, ty, z) where
#   tx, ty in [0,4) are the in-plane block coordinates (one tile spans a
#   block's full XY footprint) and z in [0,16) is the global layer index.
#   Block coords: (bx, by, bz) = (tx, ty, z // 4), layer l = z % 4.
#   Flat id: p = (tx * 4 + ty) * 16 + z.
# - THREE COUPLING CLASSES (all annealed with s, all site-resolved,
#   full XX+YY+ZZ vector kicks inherited from Rev 87/88):
#     Z_INTRA : brane <-> brane inside a block   (the Rev 88-B g_face)
#               pairs (i, i), 16 sites/interface, 3 per block column slot
#     Z_INTER : top brane of block <-> bottom brane of block above
#               pairs (i, i), 16 sites/interface (geometrically identical
#               to Z_INTRA; only the coupling constant differs)
#     XY      : lateral tile-edge coupling between in-plane neighbor
#               blocks at the same layer. +X face (x=3, idx 12..15) of p1
#               pairs with -X face (x=0, idx 0..3) of p2 at matching y;
#               +Y face (idx 4x+3) pairs with -Y face (idx 4x) at
#               matching x. 4 sites/interface.
#   Setting g_intra_z == g_inter_z == g_xy recovers a uniform 16^3 cube;
#   g_inter_z, g_xy << g_intra_z makes the block a bound "hadron" of
#   branes weakly coupled to its neighbors.
# - INTERFACE COUNT (open boundaries): 240 Z (192 intra + 48 inter),
#   192 X, 192 Y = 624 interfaces. PERIODIC_X/Y/Z close each macro
#   dimension into a ring (guarded against size-2 double counting).
# - INTERFACE ENERGY: E = -g_kind * sum_pairs (s1_i . s2_j) per
#   interface, logged per coupling class (E_Z_Intra, E_Z_Inter, E_XY)
#   plus the combined boundary total. As in 88-B, logged energies are
#   noiseless monitoring quantities; dynamics follow the noisy fields.
# - KICK ACCUMULATION: per-patch (16, 3) numpy accumulators replace the
#   dict-merge loop; a bulk brane receives Z kicks on all 16 sites from
#   both z-neighbors plus lateral kicks on its 12 edge sites (corners
#   see 2 lateral contributions). Converted to the sparse {q: (kx,ky,kz)}
#   payload format the worker already understands (unchanged worker-side
#   apply_kicks).
# - RETUNING NOTE: g values are NOT comparable to Rev 88-B's 0.15. A
#   bulk qubit now sums kicks from 2 aligned z-partners (vs at most 2 in
#   the 4-stack, but every brane is now double-sided except the global
#   top/bottom) and edge qubits add 1-2 lateral partners. Start
#   conservative; defaults below are deliberately soft.
# - SCALE NOTE: 256 x 16q statevectors = 128 MiB fp32 total; VRAM is a
#   non-issue. The cost center is tomography: 48 pauli_expectation calls
#   per patch per measure step (12,288 calls system-wide) serialized
#   over TOTAL_WORKERS pipes. Raise measure_every or WORKERS_PER_GPU if
#   wall time per step is dominated by Lat(Tomo).
#
# Inherited from Rev 88-B:
# - Flat 4x4 intra-patch tile: 2D nearest-neighbor edges only (24 edges).
# - Site-resolved (not face-averaged) inter-brane exchange along Z.
# - Per-site statistical variance injection, scale = sqrt(dt / shots).
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
import multiprocessing.connection  # explicit: mp.connection annotation below
from typing import List, Tuple, Dict, Any

# --- GLOBAL CONFIGURATION ---
BLOCK_GRID_X = 4                           # blocks along X
BLOCK_GRID_Y = 4                           # blocks along Y
BLOCK_GRID_Z = 4                           # blocks along Z
BRANES_PER_BLOCK = 4                       # 4x4 tiles stacked inside a block
GLOBAL_Z = BLOCK_GRID_Z * BRANES_PER_BLOCK # 16 global brane layers

TOTAL_PATCHES = BLOCK_GRID_X * BLOCK_GRID_Y * GLOBAL_Z  # 256
QUBITS_PER_PATCH = 16                      # flat 4x4 tile (the brane)
TOTAL_QUBITS = TOTAL_PATCHES * QUBITS_PER_PATCH         # 4096

PERIODIC_X = False                         # True -> torus along block X
PERIODIC_Y = False                         # True -> torus along block Y
PERIODIC_Z = False                         # True -> torus along global Z

# Topography tuning for raw statevectors
GPUS_AVAILABLE = 1
WORKERS_PER_GPU = 20  # 256 branes -> evolved sequentially
TOTAL_WORKERS = GPUS_AVAILABLE * WORKERS_PER_GPU

# =====================================================================
# ENVIRONMENT - set before pyqrack import
# =====================================================================
os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"


# =====================================================================
# PURE FUNCTIONS (Math & Topology)
# =====================================================================
def generate_16q_brane_tile() -> Tuple[List[Tuple[int, int]], List[int]]:
    """4x4 planar square lattice. idx = x * ly + y (row-major in x).

    Returns (intra_edges, brane_sites). Every qubit is a brane site: the
    whole tile is the Z-interface. Site index i in one brane is
    geometrically aligned with site index i in the branes above/below.
    """
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
    """Flat patch index for tile (tx, ty) at global layer z."""
    return (tx * BLOCK_GRID_Y + ty) * GLOBAL_Z + z


def patch_coords(p: int) -> Tuple[int, int, int]:
    """Inverse of patch_id: returns (tx, ty, z)."""
    z = p % GLOBAL_Z
    rest = p // GLOBAL_Z
    return rest // BLOCK_GRID_Y, rest % BLOCK_GRID_Y, z


def build_interfaces() -> List[Tuple[int, int, np.ndarray, np.ndarray, str]]:
    """All coupled seams as (p1, p2, idx1, idx2, kind).

    idx1[k] on p1 pairs with idx2[k] on p2. Kinds:
      Z_INTRA - adjacent branes inside one block
      Z_INTER - adjacent branes across a block boundary along Z
      XY      - lateral tile-edge seams between in-plane neighbor blocks
    """
    z_i1 = np.arange(QUBITS_PER_PATCH)                 # (i, i), 16 sites
    z_i2 = z_i1
    x_i1 = np.array([12 + y for y in range(4)])        # +X face of p1
    x_i2 = np.array([y for y in range(4)])             # -X face of p2
    y_i1 = np.array([x * 4 + 3 for x in range(4)])     # +Y face of p1
    y_i2 = np.array([x * 4 for x in range(4)])         # -Y face of p2

    interfaces: List[Tuple[int, int, np.ndarray, np.ndarray, str]] = []

    for tx in range(BLOCK_GRID_X):
        for ty in range(BLOCK_GRID_Y):
            for z in range(GLOBAL_Z):
                p1 = patch_id(tx, ty, z)

                # --- Z neighbor (brane stacking) ---
                if z < GLOBAL_Z - 1:
                    kind = "Z_INTRA" if (z + 1) % BRANES_PER_BLOCK != 0 else "Z_INTER"
                    interfaces.append((p1, patch_id(tx, ty, z + 1), z_i1, z_i2, kind))
                elif PERIODIC_Z and GLOBAL_Z > 2:
                    # Wrap seam is a block boundary by construction
                    interfaces.append((p1, patch_id(tx, ty, 0), z_i1, z_i2, "Z_INTER"))

                # --- X neighbor (lateral block seam) ---
                if tx < BLOCK_GRID_X - 1:
                    interfaces.append((p1, patch_id(tx + 1, ty, z), x_i1, x_i2, "XY"))
                elif PERIODIC_X and BLOCK_GRID_X > 2:
                    interfaces.append((p1, patch_id(0, ty, z), x_i1, x_i2, "XY"))

                # --- Y neighbor (lateral block seam) ---
                if ty < BLOCK_GRID_Y - 1:
                    interfaces.append((p1, patch_id(tx, ty + 1, z), y_i1, y_i2, "XY"))
                elif PERIODIC_Y and BLOCK_GRID_Y > 2:
                    interfaces.append((p1, patch_id(tx, 0, z), y_i1, y_i2, "XY"))

    return interfaces


# =====================================================================
# WORKER PROCESS LOGIC (unchanged from Rev 88-B except patch count)
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

    # 256 x 16q statevectors are 128 MiB fp32 system-wide; the cap is
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

        intra_edges, _brane_sites = generate_16q_brane_tile()

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

            # IPC SYNC INVARIANT: The worker calculates `is_measure` identically to the master.
            # It only blocks on send/recv when `is_measure` is True, ensuring lockstep.
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
        self.intra_edges, self.brane_sites = generate_16q_brane_tile()
        self.n_sites = len(self.brane_sites)  # 16; site i aligns with site i above/below

        # Block-lattice macroscopic layout
        self.patch_coords = {p: patch_coords(p) for p in range(TOTAL_PATCHES)}

        # All coupled seams with precomputed site-index arrays
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
            "Min_Unitary_Fidelity"
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
            print(f"[CSV] Warning: Setup configuration write failed: {e}", file=sys.stderr)

    def _log_csvs(self, step: int, anneal: float, bulk: float,
                  e_by_kind: Dict[str, float], bound: float, total: float,
                  min_fidelity: float, patch_profiles: Dict[int, Any]) -> None:
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
                    "Min_Unitary_Fidelity": min_fidelity})

            # One row per brane: the whole tile is the face (Face="BRANE",
            # Rev 88-B schema, extended with lattice coordinates).
            with open(self.profiles_csv, mode='a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self._profile_fields)
                for p, prof in patch_profiles.items():
                    tx, ty, z = self.patch_coords[p]
                    w.writerow({"Step": step, "Patch": p,
                                "Tx": tx, "Ty": ty, "Z": z,
                                "Block_Z": z // BRANES_PER_BLOCK,
                                "Layer": z % BRANES_PER_BLOCK,
                                "Face": "BRANE",
                                "X_mean": float(np.mean(prof["means"]["X"])),
                                "Y_mean": float(np.mean(prof["means"]["Y"])),
                                "Z_mean": float(np.mean(prof["means"]["Z"]))})
        except Exception as e:
            print(f"[CSV] Warning: Log write failed: {e}", file=sys.stderr)

    def run(self, total_steps: int, dt: float, initial_hx: float,
            target_g_intra_z: float, target_g_inter_z: float, target_g_xy: float,
            target_J: float, target_hx: float, target_hz: float,
            measure_every: int = 1, effective_shots: float = 512.0) -> None:

        if total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        if measure_every < 1:
            raise ValueError("measure_every must be a positive integer")

        print(f"[Engine] {BLOCK_GRID_X}x{BLOCK_GRID_Y}x{BLOCK_GRID_Z} block lattice "
              f"x {BRANES_PER_BLOCK} branes/block = {TOTAL_PATCHES} patches, "
              f"{TOTAL_QUBITS} qubits | interfaces: "
              f"{self.n_by_kind['Z_INTRA']} Z-intra, {self.n_by_kind['Z_INTER']} Z-inter, "
              f"{self.n_by_kind['XY']} XY | "
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
                g_now = {
                    "Z_INTRA": s * target_g_intra_z,
                    "Z_INTER": s * target_g_inter_z,
                    "XY":      s * target_g_xy,
                }
                is_measure = (t % measure_every == 0) or (t == total_steps - 1)

                # IPC SYNC INVARIANT: Master only enters this block (and blocks on recv)
                # on measure steps, perfectly matching the worker's blocking condition.
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
                # The whole tile is the brane face, so profiles carry the full
                # per-site arrays (index i == lattice site i).
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
                        np.save(self.state_dump_file, np.array(self.lattice_history))
                    except Exception as e:
                        print(f"[Checkpoint] Warning: Failed to save: {e}", file=sys.stderr)

                # --- COMPUTE KICKS & INTERFACE ENERGY (site-resolved) ---
                scale = np.sqrt(dt / effective_shots)
                n_s = self.n_sites
                AXES = ("X", "Y", "Z")

                # Per-site noisy brane fields: s_p = means + N(0,1)*sqrt(var)*scale
                noisy_field = {}
                for p in range(TOTAL_PATCHES):
                    prof = patch_profiles[p]
                    rng_p = np.random.default_rng([self.master_seed, t, p])
                    noisy_field[p] = {
                        ax: prof["means"][ax]
                            + rng_p.normal(0.0, 1.0, n_s) * np.sqrt(prof["vars"][ax]) * scale
                        for ax in AXES
                    }

                # (16, 3) kick accumulators per patch; converted to the
                # sparse dict payload the worker expects afterwards.
                kick_acc = {p: np.zeros((n_s, 3)) for p in range(TOTAL_PATCHES)}
                e_by_kind = {"Z_INTRA": 0.0, "Z_INTER": 0.0, "XY": 0.0}

                for p1, p2, i1, i2, kind in self.interfaces:
                    g = g_now[kind]
                    if g == 0.0:
                        continue
                    f1, f2 = noisy_field[p1], noisy_field[p2]
                    m1 = patch_profiles[p1]["means"]
                    m2 = patch_profiles[p2]["means"]

                    # NOTE: interface energy is a noiseless monitoring quantity.
                    # The dynamics are driven by noisy_field kicks (Langevin
                    # exploration). Intentionally decoupled (Rev 88-B note).
                    dot = 0.0
                    for a, ax in enumerate(AXES):
                        dot += float(np.sum(m1[ax][i1] * m2[ax][i2]))
                        # Per-site mean-field kicks: paired sites only.
                        # While a corner site accumulates from multiple interfaces over the loop,
                        # index arrays are unique per interface call, making numpy += safe here.
                        kick_acc[p1][i1, a] += g * f2[ax][i2]
                        kick_acc[p2][i2, a] += g * f1[ax][i1]
                    e_by_kind[kind] += -g * dot

                macroscopic_boundary_energy = sum(e_by_kind.values())

                next_kick_payloads = {}
                for p in range(TOTAL_PATCHES):
                    acc = kick_acc[p]
                    next_kick_payloads[p] = {
                        q: (float(acc[q, 0]), float(acc[q, 1]), float(acc[q, 2]))
                        for q in range(n_s) if np.any(np.abs(acc[q]) > 0.0)
                    }

                total_energy = bulk_energy + macroscopic_boundary_energy
                print(f"Step {t:03d} | E: {total_energy:+.4f} "
                      f"(Zi {e_by_kind['Z_INTRA']:+.3f} / Ze {e_by_kind['Z_INTER']:+.3f} / XY {e_by_kind['XY']:+.3f}) "
                      f"| Lat(Trot/Tomo): {max_lat_trotter:5.1f}/{max_lat_tomo:5.1f}ms "
                      f"| Fid: {min_fidelity:.5f} | {time.perf_counter() - t0:.2f}s")
                self._log_csvs(t, s * 100, bulk_energy, e_by_kind,
                               macroscopic_boundary_energy, total_energy,
                               min_fidelity, patch_profiles)

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
            # NOTE: not comparable to Rev 88-B's 0.15. A bulk brane is now
            # double-sided along Z and edge sites add lateral partners, so
            # per-qubit accumulated kick strength is larger for the same g.
            # Hierarchy below binds branes into blocks (g_intra) with weaker
            # inter-block glue (g_inter_z, g_xy). Set all three equal to
            # recover a uniform mean-field-tiled 16x16x16 cube.
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
