# -*- coding: us-ascii -*-
# 16-Qubit 4x4 Brane-Stack Macroscopic Annealing (4 Patches, 64 Qubits Total)
# Layered Planar Engine with Site-Resolved Inter-Brane Coupling
#
# REVISION 88-B - BRANE STACK VARIANT (of Rev 88-F)
#
# CHANGES (Rev 88-B):
# - MACRO GEOMETRY: The 4x4 planar patch grid (16 patches) is replaced by a
#   vertical stack of 4 patches along Z (1x1x4). Each patch remains a flat
#   4x4 16-qubit tile; total system is 4 x 16 = 64 qubits.
# - BRANE SEMANTICS: Each 2D tile IS the boundary. There are no perimeter
#   faces: all 16 qubits of a tile constitute the +Z and -Z interface
#   simultaneously. The tile plays the role of a brane whose worldvolume
#   couples to the adjacent branes above and below it.
# - SITE-RESOLVED COUPLING: Inter-brane mean-field exchange is per-site,
#   not face-averaged. Qubit (x, y) in layer k couples only to qubit (x, y)
#   in layers k+1 / k-1:
#       kick on q in p from p' = g_face * < sigma_q' >   (X, Y, Z components)
#   Face-averaging (Rev 88-F) would erase all in-plane structure across the
#   gap; per-site coupling preserves transverse locality through the stack,
#   acting as mean-field-decoupled J_perp bonds in an emergent 3rd dimension.
# - INTERFACE ENERGY: E_boundary = -g_face * sum_i (s1_i . s2_i) per
#   interface (16 site terms; full XX+YY+ZZ vector coupling inherited from
#   the Rev 87/88 kick design). 3 open interfaces by default; set
#   PERIODIC_Z = True for a closed ring of branes.
# - NOISE: Statistical variance injection is per-site (16 independent
#   samples per axis per brane), same scale = sqrt(dt / effective_shots).
# - RETUNING NOTE: target_g_face is NOT numerically comparable to Rev 88-F.
#   Per-qubit kick magnitude is similar, but each interface now sums 16
#   site dot-products instead of one face-averaged product x avg face size.
#
# Inherited from Rev 88-F:
# - Flat 4x4 intra-patch tile: 2D nearest-neighbor edges only (24 edges).
# - 16q statevector = 512 KiB fp32 / 256 KiB fp16 (FPPOW=4); all patches
#   trivially VRAM-resident, QRACK_MAX_ALLOC_MB retained for hygiene only.
# Inherited from Rev 88:
# - STOCHASTIC SCALING: scale = sqrt(dt / effective_shots), no measure_every.
# Inherited from Rev 87:
# - CONTINUOUS COUPLING: kick payloads persist across non-measure steps.
# - INTEGRATION SCALING: no m_every multiplier in apply_kicks.
# - FIDELITY WARNING: one-time stderr alert for missing fidelity binding.
#
# FIXES IN THIS VERSION:
# - Applied H-sandwich (H.RX.H) substitution in apply_zz to bypass deterministic
#   ACO Z-rotation shader hang on gfx1013 at step 039.
# - Guaranteed ring drain flush (pauli_expectation) in trotter_step_body and apply_kicks.
# - Native h(0) + s(0) probe initialization to bypass mtrx JIT overflow on rusticl
#   without introducing an angle convention race condition.
# - Scaled back to 1 worker per GPU for baseline stability testing.
# - Adjusted VRAM allocation ceiling to 7500MB to avoid GTT spillover on Oberon BC-250.
# - Forced GC collection on all probe sim failure/success paths to prevent rusticl context leaks.
# - Broadened get_unitary_fidelity catch to (AttributeError, RuntimeError) for C-layer pinvoke.
# - AMD_OPENCL_FORCE_COMPUTE_QUEUE=1 set in worker env to enable in-place ring reset
#   recovery on CYAN_SKILLFISH (gfx1013). Without this, gfx_0.0.0 ring timeout kills
#   the OpenCL context; with it, the driver recovers in-place (~15ms tomo latency spike
#   at step ~39) and execution continues. Validated on Oberon BC-250, Mesa 26.1.4.
# - Ring reset events flagged in stdout as RING_RESET when tomo latency > 5ms.
#   Allows post-hoc correlation of energy discontinuities with driver recovery events.

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
GRID_Z = 4                                 # branes stacked along Z
TOTAL_PATCHES = GRID_Z                     # 4
QUBITS_PER_PATCH = 16                      # flat 4x4 tile (the brane)
PERIODIC_Z = False                         # True -> close the stack into a ring

# Topography tuning for raw statevectors
GPUS_AVAILABLE = 1
WORKERS_PER_GPU = 1  # 4 branes -> all patches to 1 worker; prevent concurrent GFX ring saturation
TOTAL_WORKERS = GPUS_AVAILABLE * WORKERS_PER_GPU

# Tomo latency threshold above which a step is flagged as a ring reset recovery event.
# gfx_0.0.0 ring resets on CYAN_SKILLFISH produce ~15ms tomo spikes; normal is ~0.4ms.
RING_RESET_LAT_THRESHOLD_MS = 5.0

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
    whole tile is the interface, so there is no perimeter/interior split.
    Site index i in one brane is geometrically aligned with site index i
    in the branes above and below it.
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

    # Enable in-place ring reset recovery on CYAN_SKILLFISH (gfx1013).
    # Without this, an amdgpu gfx_0.0.0 ring timeout at anneal step ~39 kills
    # the OpenCL context fatally. With it, the driver resets and resumes in ~15ms.
    # Validated on Oberon BC-250, Mesa rusticl 26.1.4. No effect on CUDA targets.
    os.environ["AMD_OPENCL_FORCE_COMPUTE_QUEUE"] = "1"

    # Map multiple ranks to the same physical GPU device index
    physical_gpu_index = rank // workers_per_gpu
    os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(physical_gpu_index)

    # Bind QPager to the assigned device to enable driver-level PCIe paging
    os.environ["QRACK_QPAGER_DEVICES"] = str(physical_gpu_index)
    os.environ["QRACK_QUNITMULTI_DEVICES"] = str(physical_gpu_index)

    # 16q statevectors are tiny (<=512 KiB fp32 / 256 KiB fp16); cap is
    # retained for hygiene but is not a constraint at this tile size.
    # Tuned to 7500 MB for Oberon BC-250 to prevent GTT spillover.
    alloc_mb = 7500 // workers_per_gpu
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
        # s() availability check: s() is a native Clifford gate in pyqrack v2.7.1.
        # Confirmed present on gfx906/gfx1013 rusticl builds. Fallback retained for
        # portability to builds where the gate is absent.
        try:
            _s_test = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
            _s_test.s(0)
            del _s_test
            gc.collect()  # Force rusticl context teardown before next sim
            _USE_S_GATE = True
        except Exception:
            try: del _s_test
            except NameError: pass
            gc.collect()
            _USE_S_GATE = False

        _probe_y = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)

        # Angle-convention-agnostic Y eigenstate preparation.
        # H|0> = |+>, S|+> = |+i> (the +1 eigenstate of Y).
        # H and S are Clifford gates with no rotation parameter, so this path
        # is independent of the ANGLE_SCALE convention resolved below.
        _probe_y.h(0)
        if _USE_S_GATE:
            _probe_y.s(0)
        else:
            # Fallback: r(PZ, pi/2) -- angle-convention-dependent.
            # Safe only when ANGLE_SCALE=1.0. Dead path on gfx906/gfx1013
            # builds that support s(); retained for portability only.
            _probe_y.r(PZ, math.pi / 2.0, 0)

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
            # ZZ(theta) = CNOT . RZ(2*theta) . CNOT
            # RZ decomposed as H.RX.H to avoid the ACO shader path that hangs
            # on gfx1013 at anneal fractions ~0.38 (step ~39 of 100).
            # H.RX(phi).H == RZ(phi) exactly; uses the X-rotation shader instead.
            sim.mcx([q1], q2)
            sim.h(q2)
            apply_rx(sim, 2.0 * theta, q2)
            sim.h(q2)
            sim.mcx([q1], q2)

        def trotter_step_body(sim: QrackSimulator, num_qubits: int,
                              intra_edges: List[Tuple[int, int]],
                              J: float, hx: float, hz: float,
                              dt_local: float) -> None:
            dt_half = dt_local / 2.0

            theta_x  = -2.0 * hx * dt_half
            theta_z  = -2.0 * hz * dt_local
            theta_zz = -J * dt_local

            for q in range(num_qubits): apply_rx(sim, theta_x, q)
            for q in range(num_qubits): apply_rz(sim, theta_z, q)

            # CHUNK=8: 3 cmds/edge x 8 DWORDs/cmd x 8 edges = 192 DWORDs/chunk.
            # gfx1013 ring buffer ~4096 DWORDs. Safe headroom on CYAN_SKILLFISH.
            # Flush after each chunk with a non-collapsing blocking sync:
            # pauli_expectation does not project the statevector (QEngineCL).
            CHUNK = 8
            for i in range(0, len(intra_edges), CHUNK):
                for q1, q2 in intra_edges[i:i + CHUNK]:
                    apply_zz(sim, theta_zz, q1, q2)
                _ = sim.pauli_expectation([0], [PZ])

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

            # KICK_CHUNK=8: flush after every 8 qubits to prevent ring starvation
            # at late anneal steps where all 3 kick components are nonzero.
            KICK_CHUNK = 8
            for i in range(0, len(items), KICK_CHUNK):
                for raw_q, (kx, ky, kz) in items[i:i + KICK_CHUNK]:
                    q = int(raw_q)
                    theta_x = kx * coef
                    theta_y = ky * coef
                    theta_z = kz * coef
                    if abs(theta_x) > 1e-12: apply_rx(sim, theta_x, q)
                    if abs(theta_y) > 1e-12: apply_ry(sim, theta_y, q)
                    if abs(theta_z) > 1e-12: apply_rz(sim, theta_z, q)
                _ = sim.pauli_expectation([0], [PZ])

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

            # Allocation smoke test -- trivial at 16q but kept for parity
            try:
                _ = sim.pauli_expectation([0], [PZ])
            except Exception as e:
                raise RuntimeError(
                    "Fatal: GPU allocation failed on patch " + str(p) +
                    ". Driver error: " + str(e)
                )

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
                    except (AttributeError, RuntimeError):
                        # AttributeError: binding absent in this pyqrack build.
                        # RuntimeError: C-layer pinvoke failure (v2.7.1 API).
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

            # Kick payloads intentionally retained across non-measure steps
            # (continuous boundary field, Rev 87).

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

        # Brane-stack macroscopic layout: patch index == layer z
        self.patch_coords = {p: (0, 0, p) for p in range(TOTAL_PATCHES)}

        # Ordered list of coupled interfaces (p_lower, p_upper)
        self.interfaces: List[Tuple[int, int]] = [
            (z, z + 1) for z in range(GRID_Z - 1)
        ]

        # Guard: For GRID_Z=2, a periodic wrap causes topological collapse.
        # The top-to-bottom interface is identical to bottom-to-top; skip to
        # avoid double-counting interface coupling strength.
        if PERIODIC_Z and GRID_Z > 2:
            self.interfaces.append((GRID_Z - 1, 0))

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
                json.dump({
                    "grid_x": 1, "grid_y": 1, "grid_z": GRID_Z,
                    "num_patches": TOTAL_PATCHES,
                    "qubits_per_patch": QUBITS_PER_PATCH,
                    "tile_geometry": "4x4_brane_stack",
                    "periodic_z": PERIODIC_Z,
                    "state_dump_shape": ["n_steps", "patch", "qubit", "XYZ"]
                }, f)
            with open(self.energy_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Anneal_Percent", "MeanField_Bulk_Energy",
                    "MeanField_Boundary_Energy", "MeanField_Total_Energy",
                    "Min_Unitary_Fidelity", "Ring_Reset"
                ]).writeheader()
            with open(self.profiles_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Patch", "Face", "X_mean", "Y_mean", "Z_mean"
                ]).writeheader()
        except Exception as e:
            print("[CSV] Warning: Setup configuration write failed: " + str(e),
                  file=sys.stderr)

    def _log_csvs(self, step: int, anneal: float, bulk: float, bound: float,
                  total: float, min_fidelity: float,
                  patch_profiles: Dict[int, Any],
                  ring_reset: bool) -> None:
        try:
            with open(self.energy_csv, mode='a', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Anneal_Percent", "MeanField_Bulk_Energy",
                    "MeanField_Boundary_Energy", "MeanField_Total_Energy",
                    "Min_Unitary_Fidelity", "Ring_Reset"
                ]).writerow({
                    "Step": step,
                    "Anneal_Percent": anneal,
                    "MeanField_Bulk_Energy": bulk,
                    "MeanField_Boundary_Energy": bound,
                    "MeanField_Total_Energy": total,
                    "Min_Unitary_Fidelity": min_fidelity,
                    "Ring_Reset": 1 if ring_reset else 0
                })

            # One row per brane: the whole tile is the face. Face tagged "BRANE"
            # (schema note: +X/-X/+Y/-Y face rows of Rev 88-F no longer exist).
            with open(self.profiles_csv, mode='a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=[
                    "Step", "Patch", "Face", "X_mean", "Y_mean", "Z_mean"
                ])
                for p, prof in patch_profiles.items():
                    w.writerow({
                        "Step": step, "Patch": p, "Face": "BRANE",
                        "X_mean": float(np.mean(prof["means"]["X"])),
                        "Y_mean": float(np.mean(prof["means"]["Y"])),
                        "Z_mean": float(np.mean(prof["means"]["Z"]))
                    })
        except Exception as e:
            print("[CSV] Warning: Log write failed: " + str(e), file=sys.stderr)

    def run(self, total_steps: int, dt: float, initial_hx: float,
            target_g_face: float, target_J: float, target_hx: float,
            target_hz: float, measure_every: int = 1,
            effective_shots: float = 512.0) -> None:

        if total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        if measure_every < 1:
            raise ValueError("measure_every must be a positive integer")

        total_qubits = TOTAL_PATCHES * QUBITS_PER_PATCH
        print(
            "[Engine] " + str(TOTAL_PATCHES) +
            " stacked branes (1x1x" + str(GRID_Z) +
            (", periodic" if PERIODIC_Z else ", open ends") +
            "), " + str(total_qubits) + " qubits, " +
            str(GPUS_AVAILABLE) + " GPUs (" +
            str(WORKERS_PER_GPU) + " workers/GPU), " +
            str(total_steps) + " steps"
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

                # --- RING RESET DETECTION ---
                # A tomo latency spike above RING_RESET_LAT_THRESHOLD_MS indicates
                # the amdgpu driver performed an in-place gfx_0.0.0 ring reset
                # and recovered. The statevector is preserved; the step is valid
                # but flagged for post-hoc correlation with energy discontinuities.
                ring_reset = max_lat_tomo > RING_RESET_LAT_THRESHOLD_MS

                # --- COMPUTE KICKS & INTER-BRANE ENERGY (site-resolved) ---
                next_kick_payloads = {p: {} for p in range(TOTAL_PATCHES)}
                macroscopic_boundary_energy = 0.0

                scale = np.sqrt(dt / effective_shots)
                n_s = self.n_sites

                # Per-site noisy brane fields: s_p = means + N(0,1)*sqrt(var)*scale
                noisy_field = {}
                for p in range(TOTAL_PATCHES):
                    prof = patch_profiles[p]
                    rng_p = np.random.default_rng([self.master_seed, t, p])
                    noisy_field[p] = {
                        ax: prof["means"][ax]
                            + rng_p.normal(0.0, 1.0, n_s)
                            * np.sqrt(prof["vars"][ax]) * scale
                        for ax in ("X", "Y", "Z")
                    }

                for p1, p2 in self.interfaces:
                    f1, f2 = noisy_field[p1], noisy_field[p2]
                    f1_mean = patch_profiles[p1]["means"]
                    f2_mean = patch_profiles[p2]["means"]

                    # macroscopic_boundary_energy is a noiseless monitoring quantity.
                    # Dynamics are driven by noisy_field kicks (Langevin exploration).
                    # These are intentionally decoupled: logged energy tracks the
                    # mean-field potential; trajectory follows the stochastic update.
                    dot = float(np.sum(
                        f1_mean["X"] * f2_mean["X"]
                        + f1_mean["Y"] * f2_mean["Y"]
                        + f1_mean["Z"] * f2_mean["Z"]
                    ))
                    macroscopic_boundary_energy += -current_g_face * dot

                    # Per-site mean-field kicks: each qubit sees only its aligned
                    # partner in the adjacent brane. Uses noisy_field for Langevin
                    # variance injection.
                    for q in range(n_s):
                        k = next_kick_payloads[p1].get(q, (0., 0., 0.))
                        next_kick_payloads[p1][q] = (
                            k[0] + current_g_face * f2["X"][q],
                            k[1] + current_g_face * f2["Y"][q],
                            k[2] + current_g_face * f2["Z"][q],
                        )
                        k = next_kick_payloads[p2].get(q, (0., 0., 0.))
                        next_kick_payloads[p2][q] = (
                            k[0] + current_g_face * f1["X"][q],
                            k[1] + current_g_face * f1["Y"][q],
                            k[2] + current_g_face * f1["Z"][q],
                        )

                total_energy = bulk_energy + macroscopic_boundary_energy
                reset_tag = "  RING_RESET" if ring_reset else ""
                print(
                    "Step {:03d} | E: {:+.4f} | "
                    "Lat(Trot/Tomo): {:5.1f}/{:5.1f}ms | "
                    "Fid: {:.5f} | {:.2f}s{}".format(
                        t, total_energy,
                        max_lat_trotter, max_lat_tomo,
                        min_fidelity,
                        time.perf_counter() - t0,
                        reset_tag
                    )
                )
                self._log_csvs(
                    t, s * 100, bulk_energy,
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
            # NOTE: not comparable to Rev 88-F's 0.15. Per-qubit kick strength
            # is similar, but interface energy now sums 16 site dot-products.
            # Start conservative; g here acts as an effective J_perp.
            target_g_face=0.15,
            target_J=1.0,
            target_hx=0.5,
            target_hz=0.2,
            measure_every=1,
            effective_shots=512.0
        )
    except KeyboardInterrupt:
        pass
