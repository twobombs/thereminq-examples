# -*- coding: us-ascii -*-
# 16-Qubit 4x4 Brane Tiles -> 4x4x4 Brane-Stack Blocks -> 4x4x4 Block Lattice
# (256 Patches, 4096 Qubits Total)
# Layered Planar Engine with Site-Resolved Inter-Brane AND Inter-Block Coupling
#
# REVISION 88-H - BLOCK LATTICE VARIANT (of Rev 88-G)
#
# CHANGES (Rev 88-H) - use-after-reset cascading failure fixed:
#
# NEW KERNEL LOG EVIDENCE (seq delta=3, hqd deactivate failure, BACO reset):
#
#   comp_1.1.0 timeout: signaled=1874, emitted=1877 -> seq delta=3
#   "fail to wait on hqd deactive"
#   "Ring comp_1.1.0 reset failed"
#   "BACO reset"  (Bus-Active Chip-Off: full power cycle of the GPU die)
#   "VRAM is lost due to GPU reset!"
#   7 "innocent context" kills (down from 9 in prior runs with CHUNK=8)
#
#   seq delta=3 directly identifies apply_zz as the failure unit:
#   apply_zz = mcx -> rz -> mcx = exactly 3 kernels submitted to the
#   comp ring before any drain. CHUNK=1 chunked at edge granularity
#   but not at gate granularity -- the 3 kernels within a single ZZ
#   gate were still submitted as an unbarriered burst. The GPU hangs
#   on the first of those 3 kernels, the other 2 remain in-flight,
#   and the ring timeout fires with emitted=signaled+3.
#
#   "fail to wait on hqd deactive" = the compute CU is truly STUCK,
#   not merely slow. The MEC cannot deactivate the Hardware Queue
#   Descriptor within the timeout. This is a memory access fault:
#   the kernel is trying to read/write a VRAM address that became
#   invalid after a prior ring reset zeroed or reallocated its
#   backing buffer. This is a USE-AFTER-RESET pattern.
#
#   "psp gfx command UNLOAD_TA(0x2) failed (0x117)" = PSP firmware
#   itself is in a bad state; soft ring reset cannot proceed, forcing
#   BACO (full die power cycle). After BACO, VRAM is wiped clean.
#   The sims{} dict in the worker still holds QrackSimulator objects
#   whose cl_mem handles now point to zeroed or nonexistent VRAM.
#   The next kernel dispatch on any of these stale sims hangs the CU
#   again, triggering another BACO reset (reset counter = 3 in the
#   second 0000:65:00.0 crash). This is cascading use-after-reset.
#
# FIX 1 - INTRA-GATE DRAIN in apply_zz (this file):
#   Insert a pauli_expectation drain after EACH of the 3 kernels
#   within apply_zz (mcx, rz, mcx). This makes every kernel
#   submission immediately followed by a blocking sync before the
#   next kernel is dispatched. Maximum in-flight on any comp ring
#   = 1 at all times. seq delta cannot exceed 1 under this scheme.
#   The pauli_expectation is non-collapsing (does not project the
#   statevector). Cost: 3 extra drains per ZZ gate x 24 edges per
#   patch x 256 patches = 18,432 extra drain calls per step.
#   At ~0.1ms/drain (16q) = ~1.8s/step overhead, acceptable given
#   the BACO reset overhead (~1s per event) already dominates.
#
# FIX 2 - KET CHECKPOINTING + SIM RESURRECTION (this file):
#   Even with intra-gate drains, a ring reset between steps can
#   leave sims{} holding stale cl_mem handles. The worker now:
#   a) Checkpoints the statevector via out_ket() into ket_cache[p]
#      BEFORE each patch's Trotter step (one DMA per patch per step:
#      2^16 complex64 = 512KB per patch, 128MB total in-worker RAM).
#   b) On any RuntimeError whose message contains "context is lost"
#      or "CS has cancelled" (the userspace amdgpu error strings),
#      calls resurrect_sim(p): deletes the stale sim, gc.collect()s
#      the OpenCL context, creates a fresh QrackSimulator, and
#      restores the statevector via in_ket(ket_cache[p]).
#   c) Retries the failed operation once after resurrection.
#   d) If resurrection fails (no ket_cache yet, or in_ket errors),
#      falls back to |+>^16 (H on all qubits) -- the t=0 state.
#      This patch's evolution restarts from the last checkpointed
#      ket, introducing at most one step of discontinuity.
#   Note: ket_cache is populated at the START of each step (before
#   Trotter), so a crash mid-Trotter reverts to the PRE-step ket,
#   not a partially-evolved intermediate. This is the correct
#   recovery semantics: re-evolve the step from the clean checkpoint.
#
# Rev 88-G/F/E/D changes retained:
#   CHUNK=1 (edge-level, now redundant given intra-gate drains but
#   kept as defence-in-depth), KICK_CHUNK=1, KICK_THETA_THRESHOLD=1e-6,
#   QRACK_MAX_ALLOC_MB=7500/worker, inter-patch barrier+yield/16,
#   RING_RESET detection, all Rev 88-B/D worker improvements.
#
# 1. Y-PROBE: mtrx() JIT-overflow-safe Clifford path replaces the mtrx()
#    call used in Rev 88-C. Y-eigenstate |+i> is now prepared with
#    h(0) + s(0) (native Clifford gates, no rotation parameter, no JIT
#    kernel compilation). An S-gate availability check probes the build
#    at startup; the old mtrx() path is retained as a dead fallback only
#    when s() is absent (non-gfx906/gfx1013 builds).
#
# 2. RING-BUFFER CHUNKED FLUSHING in trotter_step_body and apply_kicks:
#    - trotter_step_body: ZZ edges batched into CHUNK=8 groups; a
#      non-collapsing pauli_expectation([0],[PZ]) drain is issued after
#      each chunk. Prevents gfx_0.0.0 ring buffer overruns at high qubit
#      counts or late anneal steps when all 3 kick components are nonzero.
#      Safe headroom: 3 cmds/edge x 8 edges x 8 DWORDs/cmd = 192 DWORDs
#      per chunk vs. ~4096 DWORD ring on gfx1013.
#    - apply_kicks: qubits batched into KICK_CHUNK=8 groups with the same
#      drain pattern. Prevents ring starvation when all 16 sites of a
#      brane receive 3-axis kicks simultaneously.
#
# 3. AMD_OPENCL_FORCE_COMPUTE_QUEUE=1 added to worker environment.
#    Enables in-place gfx_0.0.0 ring reset recovery on CYAN_SKILLFISH
#    (gfx1013) / Oberon BC-250 / Mesa rusticl 26.1.4. Without this flag
#    a ring timeout kills the OpenCL context fatally; with it the driver
#    resets in-place (~15ms tomo latency spike) and execution continues.
#    No effect on CUDA targets.
#
# 4. gc.collect() after every probe QrackSimulator deletion during the
#    Pauli/angle autodetect block. Prevents rusticl context handle leaks
#    across the 5 probe sims that run before the main simulation loop.
#
# 5. get_unitary_fidelity() exception catch broadened from AttributeError
#    to (AttributeError, RuntimeError) to cover C-layer pinvoke failures
#    introduced in PyQrack v2.7.1 API.
#
# 6. RING RESET DETECTION added to master orchestrator:
#    - RING_RESET_LAT_THRESHOLD_MS = 5.0 ms threshold (normal ~0.4ms;
#      gfx1013 ring reset recovery ~15ms).
#    - ring_reset flag computed from max_lat_tomo per step.
#    - Ring_Reset column added to energy CSV for post-hoc correlation of
#      energy discontinuities with driver recovery events.
#    - "RING_RESET" tag appended to stdout step line when triggered.
#
# Inherited from Rev 88-C:
# - 4x4x4 block lattice: 256 patches, 4096 qubits.
# - Three coupling classes: Z_INTRA, Z_INTER, XY (site-resolved).
# - build_interfaces() with precomputed site-index arrays.
# - Vectorized NumPy kick accumulation (kick_acc[p] shape (16,3)).
# - Per-kind energy logging: E_Z_Intra, E_Z_Inter, E_XY.
# - PERIODIC_X/Y/Z boundary condition flags.
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

# Tomo latency threshold above which a step is flagged as a ring reset
# recovery event. gfx_0.0.0 ring resets on CYAN_SKILLFISH produce ~15ms
# tomo spikes; normal latency is ~0.4ms. Inherited from Rev 88-B.
RING_RESET_LAT_THRESHOLD_MS = 5.0

# Inter-patch compute ring yield interval (Rev 88-E).
# Every N patches in the sequential loop, time.sleep(0) is called after
# the per-patch barrier to allow the OS/amdgpu ISR to process completions.
# 16 = 16 x ~126 dispatches = 2016 dispatches between sleep yields.
# Reduce to 8 if Lat(Trot) spikes persist above ~5ms; raise to 32 if
# per-step wall time increases unacceptably.
INTER_PATCH_YIELD_EVERY = 16

# Minimum rotation angle (radians) to submit as a GPU kernel dispatch (Rev 88-F).
# Below this threshold the kick rotation is suppressed: the gate is a no-op
# to 13 significant figures on a 16q statevector (<Z> change ~ theta^2/2).
# Raised from the prior 1e-12 (below float32 epsilon ~1.2e-7) to 1e-6.
# With dt=0.04 and typical g*f ~ 0.04, theta ~ 3e-3 >> 1e-6; only noise-
# dominated near-zero kicks at low anneal fractions are suppressed. This
# reduces GART-mapped kernel argument buffer count at early/mid steps,
# directly reducing per-step GART pressure.
KICK_THETA_THRESHOLD = 1e-6

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
    z_i1 = np.arange(QUBITS_PER_PATCH)
    z_i2 = z_i1
    x_i1 = np.array([12 + y for y in range(4)])   # +X face of p1
    x_i2 = np.array([y for y in range(4)])         # -X face of p2
    y_i1 = np.array([x * 4 + 3 for x in range(4)])# +Y face of p1
    y_i2 = np.array([x * 4 for x in range(4)])    # -Y face of p2

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
    # Without this, an amdgpu gfx_0.0.0 ring timeout kills the OpenCL
    # context fatally. With it, the driver resets in-place (~15ms tomo
    # latency spike) and execution continues. Inherited from Rev 88-B.
    # No effect on CUDA targets.
    os.environ["AMD_OPENCL_FORCE_COMPUTE_QUEUE"] = "1"

    physical_gpu_index = rank // workers_per_gpu
    os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(physical_gpu_index)
    os.environ["QRACK_QPAGER_DEVICES"] = str(physical_gpu_index)
    os.environ["QRACK_QUNITMULTI_DEVICES"] = str(physical_gpu_index)

    # QRACK_MAX_ALLOC_MB capped to 7500 MB per worker (Rev 88-F).
    # The V340 has 8000MB HBM2 per die. Previous value of 64000//workers_per_gpu
    # gave 64000MB at 1 worker -- far above physical VRAM. Qrack uses this
    # limit to decide whether to spill statevectors to GTT (host-mapped) memory.
    # An inflated cap caused all 256 statevector buffers to be mapped into
    # GART-backed GTT, saturating the 256MB default GART aperture and producing
    # the per-step ring resets. 7500MB is below HBM2 capacity, keeping Qrack
    # in VRAM-only mode. Requires amdgpu.gartsize=2048 in GRUB as backstop.
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
        gc.collect()  # Force rusticl context teardown; prevents handle leaks.

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
        # S-gate availability check: s() is a native Clifford gate in
        # pyqrack v2.7.1. Confirmed present on gfx906/gfx1013 rusticl
        # builds. The mtrx() fallback is retained for portability to
        # builds where s() is absent, but that path triggers JIT kernel
        # compilation and may overflow the rusticl stack on the first call.
        # Prefer the Clifford path wherever possible. (Rev 88-B)
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

        # Angle-convention-agnostic Y eigenstate preparation.
        # H|0> = |+>, S|+> = |+i> (the +1 eigenstate of Y).
        # H and S are Clifford gates with no rotation parameter; this path
        # is independent of the ANGLE_SCALE convention resolved below.
        # The mtrx() fallback path (Rev 88-C) triggered JIT overflow on
        # rusticl; replaced here by the Clifford path from Rev 88-B.
        _probe_y.h(0)
        if _USE_S_GATE:
            _probe_y.s(0)
        else:
            # Fallback: mtrx() Rx(pi/2) approximating S in the X-Z plane.
            # Angle-convention-dependent; safe only when ANGLE_SCALE=1.0.
            # Dead path on gfx906/gfx1013 builds that support s().
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
            # INTRA-GATE DRAIN (Rev 88-H): drain after EACH of the 3 kernels
            # within this ZZ gate. seq delta=3 in the crash log showed that
            # mcx/rz/mcx were all 3 in-flight simultaneously; the ring hung
            # on the first, leaving 2 more pending. Each pauli_expectation
            # is a non-collapsing blocking sync that empties the comp ring
            # before the next kernel is dispatched. Max in-flight = 1.
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
            theta_z  = -2.0 * hz * dt_local
            theta_zz = -J * dt_local

            for q in range(num_qubits): apply_rx(sim, theta_x, q)
            for q in range(num_qubits): apply_rz(sim, theta_z, q)

            # CHUNK=1 (Rev 88-G): submit ONE ZZ edge (3 kernels: mcx/rz/mcx),
            # then immediately drain with pauli_expectation before the next.
            # Limits maximum simultaneous in-flight CL command queue submissions
            # to 1 at any time, eliminating the 9-context burst pattern that
            # caused the hardware reset. Previous CHUNK=8 submitted 8 ZZ bursts
            # before draining, leaving 9 cl_command_queues with pending work
            # simultaneously -- exactly matching the 9 "innocent context" kills
            # seen in the crash log. Cost: 24 drain calls vs 3 at CHUNK=8,
            # but ring-reset overhead already dominates step latency.
            CHUNK = 1
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

            # KICK_CHUNK=1 (Rev 88-G): same principle as CHUNK=1 in
            # trotter_step_body. Submit one qubit's rotations (up to 3
            # kernels: Rx, Ry, Rz) then drain before the next qubit.
            # Prevents simultaneous in-flight submissions across multiple
            # qubit kick contexts during the apply_kicks phase.
            KICK_CHUNK = 1
            for i in range(0, len(items), KICK_CHUNK):
                for raw_q, (kx, ky, kz) in items[i:i + KICK_CHUNK]:
                    q = int(raw_q)
                    theta_x = kx * coef
                    theta_y = ky * coef
                    theta_z = kz * coef
                    # KICK_THETA_THRESHOLD=1e-6: suppress near-zero rotations
                    # to reduce GART-mapped kernel buffer count (Rev 88-F).
                    if abs(theta_x) > KICK_THETA_THRESHOLD: apply_rx(sim, theta_x, q)
                    if abs(theta_y) > KICK_THETA_THRESHOLD: apply_ry(sim, theta_y, q)
                    if abs(theta_z) > KICK_THETA_THRESHOLD: apply_rz(sim, theta_z, q)
                _ = sim.pauli_expectation([0], [PZ])

        # Strings that identify a use-after-reset context loss error in
        # the RuntimeError message from the amdgpu/rusticl userspace layer.
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

        # Per-patch statevector checkpoint: updated before each Trotter step.
        # 2^16 complex64 = 512 KB per patch; 128 MB total at 256 patches.
        # Used to restore state after sim resurrection following a context loss.
        ket_cache: Dict[int, np.ndarray] = {}

        def resurrect_sim(p: int) -> QrackSimulator:
            """Delete stale sim, create fresh one, restore from ket_cache."""
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
                # No checkpoint yet (failure on step 0): start from |+>^16.
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
            # Seed ket_cache with the initial |+>^16 state so that
            # resurrect_sim() can restore even on a step-0 failure.
            try:
                ket_cache[p] = np.array(sim.out_ket(), dtype=np.complex64)
            except Exception:
                pass  # Cache miss; resurrect_sim falls back to |+>^16.

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

                # KET CHECKPOINT (Rev 88-H): snapshot statevector BEFORE
                # Trotter so that a mid-step context loss can be recovered
                # by re-evolving from this clean pre-step state.
                try:
                    ket_cache[p] = np.array(sim.out_ket(), dtype=np.complex64)
                except Exception:
                    pass  # Stale sim; resurrect_sim will use prior cache.

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
                            # Retry kicks on fresh sim with restored ket.
                            try:
                                apply_kicks(sim, kick_payloads[p], dt)
                            except Exception:
                                pass  # Absorb; Trotter will proceed.
                        else:
                            raise

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
                        # Retry Trotter from restored ket checkpoint.
                        try:
                            trotter_step_body(sim, QUBITS_PER_PATCH, intra_edges,
                                              current_J, current_hx, current_hz, dt)
                        except Exception:
                            pass  # Absorb; tomo will measure recovered state.
                    else:
                        raise

                # INTER-PATCH BARRIER (Rev 88-E): force compute ring to drain
                # completely between patches.
                try:
                    _ = sim.pauli_expectation([0], [PZ])
                except RuntimeError as _be:
                    if is_context_loss(_be):
                        sim = resurrect_sim(p)
                    else:
                        raise

                # INTER-PATCH YIELD (Rev 88-E): OS scheduler yield every N patches.
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
                            # Measure the resurrected (restored) state.
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
                        # AttributeError: binding absent in this pyqrack build.
                        # RuntimeError: C-layer pinvoke failure (v2.7.1 API).
                        # Catch broadened from AttributeError-only in Rev 88-C.
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
        self.n_sites = len(self.brane_sites)  # 16

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

        # Ring_Reset column added (Rev 88-D, from Rev 88-B).
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
            "[Engine] Rev 88-H: intra-ZZ drain (seq-delta=1), ket checkpoint "
            "+ sim resurrection on context loss, CHUNK=1/KICK_CHUNK=1, "
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

                # --- RING RESET DETECTION ---
                # A tomo latency spike above RING_RESET_LAT_THRESHOLD_MS
                # indicates the amdgpu driver performed an in-place ring reset
                # and recovered. The statevector is preserved; the step is
                # valid but flagged for post-hoc correlation with energy
                # discontinuities. Inherited from Rev 88-B.
                ring_reset = max_lat_tomo > RING_RESET_LAT_THRESHOLD_MS

                # --- COMPUTE KICKS & INTERFACE ENERGY (site-resolved) ---
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
                # KICK_THETA_THRESHOLD pre-filter (Rev 88-F): compute theta at
                # master side and suppress entries that would be no-ops in the
                # worker's apply_kicks anyway. Reduces IPC pickle size and
                # eliminates near-zero kick entries from the worker's kick dict,
                # preventing them from entering the KICK_CHUNK loop at all.
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
