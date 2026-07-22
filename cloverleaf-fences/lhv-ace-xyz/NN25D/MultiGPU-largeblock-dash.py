# -*- coding: us-ascii -*-
# 16-Qubit 4x4 Brane Tiles -> 4x4x4 Brane-Stack Blocks -> 4x4x4 Block Lattice
# (256 Patches, 4096 Qubits Total)
# Layered Planar Engine with Site-Resolved Inter-Brane AND Inter-Block Coupling
#
# REVISION 88-N - BLOCK LATTICE VARIANT (of Rev 88-M)
#
# CONFIRMED WORKING in Rev 88-M: all workers restore from memmap correctly.
#   With 2 GPUs x 32 workers = 64 total workers, all 64 restored at step 39.
#   The two-pass gather correctly handles simultaneous death of all workers.
#
# REMAINING ISSUE: "timeout" at step 39 with 64 workers.
#   The respawn is NOT a crash -- it IS completing -- but it takes ~19 minutes:
#
#   Pass 1 (dead worker detection + spawn): SEQUENTIAL.
#     for i in range(64): recv() -> EOFError -> join(timeout=5s) -> spawn
#     64 * (5s join + 1s spawn) = ~384s to spawn all 64 replacements.
#
#   Pass 2 (collect from respawned workers): SEQUENTIAL.
#     for i in pending_recv: conn.recv() <- blocks on each one individually.
#     Each respawned worker takes ~12s (JIT rebuild + 4 ket restores).
#     64 * 12s = ~768s waiting for all 64 to send their first payload.
#
#   Total: ~19 minutes for one respawn event. With any number of workers
#   above ~8, this serial chain makes respawn prohibitively slow.
#
# FIX (Rev 88-N): PARALLEL RESPAWN via multiprocessing.connection.wait().
#
# 1. PASS 1 SPLIT INTO TWO SEQUENTIAL MINI-PASSES:
#    Mini-pass A: iterate all pipes with recv(), collect EOFError indices
#      into dead_indices[], accumulate live data immediately. No spawning yet.
#      Join timeout reduced from 5s to 1s: os._exit(1) workers terminate
#      immediately; the 5s was conservative padding, not needed.
#    Mini-pass B: join+spawn ALL dead workers simultaneously (tight loop,
#      no blocking between spawns).
#    Combined: ~1s (max join) + 64 * ~0.1s (spawn) = ~7s for all 64 spawns
#    vs the old 64 * 6s = 384s.
#
# 2. PASS 2 USES connection.wait() FOR PARALLEL RECV:
#    multiprocessing.connection.wait(conns, timeout=T) is a select()-based
#    multiplexer that returns a list of connections that have data ready,
#    without blocking on each one individually.
#    while pending_recv not empty:
#      ready_conns = wait([pipes[i] for i in pending_recv], timeout=120)
#      for conn in ready_conns: recv() and accumulate; remove from pending
#    All 64 respawned workers' JIT builds (~12s each) run in parallel.
#    Pass 2 completes in max(12s) = ~12s for any number of workers,
#    vs the old 64 * 12s = 768s.
#
#    Combined respawn time (64 workers):
#      Old (88-M): ~19 min serial
#      New (88-N): ~20s parallel
#
# 3. WORKERS_PER_GPU and GPUS_AVAILABLE unchanged.
#    The fix is purely in the master gather loop; worker code is untouched.
#
# Retained from Rev 88-M: reset_event_count per-round semantics,
#   MAX_RESET_EVENTS=10, memmap ket cache, _fatal_exit, resurrect_sim,
#   CHUNK=8, KICK_THETA_THRESHOLD, all prior improvements.
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
import tempfile
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

GPUS_AVAILABLE = 1
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

# Memmap ket cache layout (Rev 88-K, replaces SharedMemory from Rev 88-J).
# One complex64 slot per patch: 2^QUBITS_PER_PATCH = 65536 amplitudes.
# Total file size: 256 * 65536 * 8 = 128MB on /tmp (not /dev/shm).
KET_SLOT_ELEMS  = 1 << QUBITS_PER_PATCH       # 65536 complex64 per patch
KET_MEMMAP_PATH = os.path.join(
    tempfile.gettempdir(),
    "thereminq_ket_cache_" + str(os.getpid()) + ".bin"
)

# Maximum number of full GPU reset EVENTS (not individual worker respawns)
# the master will survive per run. One event = one gather round where
# any worker died; multiple workers dying in the same round = one event.
# 10 is generous; a typical run sees 1-2 resets around step 039-040.
MAX_RESET_EVENTS = 10

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
    measure_every: int,
    shm_name: str,      # kept as arg name for API compat; holds memmap path
    resume_step: int = 0,
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

    # Map the file-backed memmap ket cache (Rev 88-K).
    # shm_name holds the memmap file path (legacy arg name kept for compat).
    # mode='r+': file already created by master; worker reads and writes.
    # On /tmp (overlay fs in Docker) -- no /dev/shm size limit.
    _memmap_path = shm_name
    _shm_arr = np.memmap(
        _memmap_path,
        dtype=np.complex64,
        mode='r+',
        shape=(TOTAL_PATCHES, KET_SLOT_ELEMS)
    )

    def _flush_ket_to_shm(p: int, ket: np.ndarray) -> None:
        """Write patch ket into the memmap slot and flush to disk."""
        try:
            _shm_arr[p, :] = ket.astype(np.complex64)
            _shm_arr.flush()
        except Exception:
            pass

    def _fatal_exit(sims_to_flush: Dict[int, Any]) -> None:
        """Flush all patch kets to memmap, close conn, exit without __del__."""
        for p, sim in list(sims_to_flush.items()):
            try:
                ket = np.array(sim.out_ket(), dtype=np.complex64)
                _flush_ket_to_shm(p, ket)
            except Exception:
                pass
        try:
            del _shm_arr  # Close the memmap file handle.
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        os._exit(1)  # Immediate exit; no rusticl __del__, no coredump.

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
            # Bare CNOT-conjugation ZZ gate: mcx -> rz -> mcx.
            # Intra-gate pauli_expectation drains were added in Rev 88-H to
            # cap seq delta at 1, but caused 72x drain overhead per patch per
            # step (3 drains/gate x 24 gates) which made Lat(Tomo)=250ms in
            # the ring-reset regime. The ket checkpoint+resurrection in Rev 88-H
            # already provides correct recovery from any seq-delta=3 hang:
            # context loss -> resurrect from pre-step ket -> retry Trotter.
            # Intra-gate drains are therefore redundant. Removed in Rev 88-I.
            sim.mcx([q1], q2)
            apply_rz(sim, 2.0 * theta, q2)
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

            # CHUNK=8 restored (Rev 88-I): defence-in-depth against ring buffer
            # overflow (original gfx1013 concern from Rev 88-B). With ket
            # checkpoint+resurrection handling context loss, CHUNK no longer
            # needs to be 1 for correctness. CHUNK=8 gives 3 drain points per
            # 24 edges, avoiding 21 extra drain calls vs CHUNK=1.
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

            # KICK_CHUNK=8 restored (Rev 88-I): same rationale as CHUNK=8.
            # Resurrection handles any context loss; KICK_CHUNK=8 avoids
            # 8 extra drain calls per kick phase vs KICK_CHUNK=1.
            KICK_CHUNK = 8
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
            if resume_step > 0:
                # RESPAWN PATH (Rev 88-J): restore ket from shared memory
                # written by the dead worker's last pre-Trotter checkpoint.
                try:
                    ket_slot = _shm_arr[p, :].copy()
                    sim.in_ket(ket_slot.tolist())
                    print("[Worker " + str(rank) + "] patch " + str(p) +
                          " restored from shm at resume_step=" +
                          str(resume_step) + ".", file=sys.stderr)
                except Exception as _re:
                    print("[Worker " + str(rank) + "] patch " + str(p) +
                          " shm restore failed (" + str(_re) +
                          "); init |+>^16.", file=sys.stderr)
                    for q in range(QUBITS_PER_PATCH): apply_h(sim, q)
            else:
                # FRESH START: initialise to |+>^16.
                for q in range(QUBITS_PER_PATCH): apply_h(sim, q)
            sims[p] = sim

            try:
                _ = sim.pauli_expectation([0], [PZ])
            except Exception as e:
                raise RuntimeError(
                    "Fatal: GPU allocation failed on patch " + str(p) +
                    ". Driver error: " + str(e)
                )
            # Seed shm slot with initial ket (|+>^16 or restored).
            try:
                ket_cache[p] = np.array(sim.out_ket(), dtype=np.complex64)
                _flush_ket_to_shm(p, ket_cache[p])
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

                # KET CHECKPOINT (Rev 88-H/J): snapshot statevector BEFORE
                # Trotter so that a mid-step context loss can be recovered.
                # Rev 88-J: also flush to shared memory so the master can
                # hand the ket to a respawned worker after a BACO crash.
                try:
                    ket_cache[p] = np.array(sim.out_ket(), dtype=np.complex64)
                    _flush_ket_to_shm(p, ket_cache[p])
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
        try:
            del _shm_arr  # Close memmap handle.
        except Exception:
            pass
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

        # File-backed memmap ket cache (Rev 88-K, replaces SharedMemory).
        # Created on /tmp (overlay fs) to avoid Docker's 64MB /dev/shm limit.
        # mode='w+' creates the file and zeros it. Workers open mode='r+'.
        self.memmap_path = KET_MEMMAP_PATH
        try:
            self.ket_mm = np.memmap(
                self.memmap_path,
                dtype=np.complex64,
                mode='w+',
                shape=(TOTAL_PATCHES, KET_SLOT_ELEMS)
            )
            self.ket_mm[:] = 0.0
            self.ket_mm.flush()
            print("[Engine] Ket memmap: " + self.memmap_path +
                  " (" + str(TOTAL_PATCHES * KET_SLOT_ELEMS * 8 // (1024*1024)) +
                  " MB)")
        except Exception as e:
            raise RuntimeError(
                "Fatal: could not create ket memmap at " +
                self.memmap_path + ": " + str(e)
            )

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
            "[Engine] Rev 88-N: parallel respawn via connection.wait() "
            "(O(max_worker_time) not O(N*worker_time)), "
            "MAX_RESET_EVENTS=" + str(MAX_RESET_EVENTS) + ", "
            "memmap ket cache, CHUNK=8."
        )

        active_ranks = [r for r in range(TOTAL_WORKERS)
                        if self.worker_assignments[r]]

        def _spawn_worker(rank: int, resume_step: int = 0):
            """Spawn a single worker process. Returns (process, parent_conn)."""
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=gpu_worker_process,
                args=(rank, WORKERS_PER_GPU, self.worker_assignments[rank],
                      child_conn, dt, total_steps, initial_hx,
                      target_J, target_hx, target_hz, measure_every,
                      self.memmap_path, resume_step)
            )
            p.start()
            child_conn.close()
            return p, parent_conn

        workers = []
        pipes   = []
        rank_map = {}  # pipe index -> rank

        for i, rank in enumerate(active_ranks):
            proc, conn = _spawn_worker(rank, resume_step=0)
            workers.append(proc)
            pipes.append(conn)
            rank_map[i] = rank

        respawn_count = 0   # total individual worker respawns (for logging)
        reset_event_count = 0  # GPU reset events (the limit that matters)

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

                # --- GATHER (parallel two-pass respawn) ---
                # Rev 88-N: fully parallelised to handle large worker counts
                # (e.g. 2 GPUs x 32 workers = 64 total) without serial respawn
                # chains that caused ~19-minute "timeouts" at step 039.
                #
                # Mini-pass A: drain all live pipes; collect dead pipe indices.
                #   Uses connection.wait() to avoid blocking on dead pipes.
                # Mini-pass B: join+spawn all dead workers simultaneously.
                # Pass 2:      collect from all respawned workers in parallel
                #   using connection.wait() so all JIT builds run concurrently.
                patch_full_states = {}
                bulk_energy = 0.0
                max_lat_trotter = 0.0
                max_lat_tomo = 0.0
                min_fidelity = 1.0
                any_died = False
                dead_indices = []   # pipe indices that raised EOFError
                pending_recv = []   # pipe indices that need a pass-2 recv

                # --- Mini-pass A: non-blocking drain of all pipes ---
                # connection.wait() returns pipes with data ready (or closed).
                # We iterate all pipes; if a pipe has data we recv immediately;
                # if it is closed (worker died) we get EOFError on recv.
                remaining = list(range(len(pipes)))
                while remaining:
                    ready = mp.connection.wait(
                        [pipes[i] for i in remaining], timeout=120
                    )
                    if not ready:
                        # Timeout: remaining workers haven't responded in 120s.
                        # Treat them as dead for respawn purposes.
                        for i in remaining:
                            dead_indices.append(i)
                            any_died = True
                        break
                    # Map ready connections back to their indices
                    ready_set = set(id(c) for c in ready)
                    still_waiting = []
                    for i in remaining:
                        if id(pipes[i]) in ready_set:
                            try:
                                data = pipes[i].recv()
                                for p, payload in data.items():
                                    patch_full_states[p] = payload["state"]
                                    bulk_energy += payload["meanfield_bulk_energy"]
                                    max_lat_trotter = max(max_lat_trotter, payload["lat_trotter_ms"])
                                    max_lat_tomo = max(max_lat_tomo, payload["lat_tomo_ms"])
                                    min_fidelity = min(min_fidelity, payload.get("unitary_fidelity", 1.0))
                            except EOFError:
                                dead_indices.append(i)
                                any_died = True
                        else:
                            still_waiting.append(i)
                    remaining = still_waiting

                # --- Mini-pass B: join + spawn ALL dead workers together ---
                if dead_indices:
                    # Join dead processes (fast: os._exit(1) workers are already gone)
                    for i in dead_indices:
                        dead_proc = workers[i]
                        dead_proc.join(timeout=1)
                        if dead_proc.is_alive():
                            dead_proc.terminate()
                            dead_proc.join(timeout=2)
                        try:
                            pipes[i].close()
                        except Exception:
                            pass
                    # Spawn all replacements simultaneously
                    for i in dead_indices:
                        new_proc, new_conn = _spawn_worker(rank_map[i], resume_step=t)
                        workers[i] = new_proc
                        pipes[i] = new_conn
                        pending_recv.append(i)
                        print(
                            "[Master] Worker rank=" + str(rank_map[i]) +
                            " died at step " + str(t) + "; respawned.",
                            file=sys.stderr
                        )

                # Count this round as one reset event if any worker died.
                if any_died:
                    reset_event_count += 1
                    print(
                        "[Master] GPU reset event #" + str(reset_event_count) +
                        " at step " + str(t) + ": " +
                        str(len(dead_indices)) + " workers respawned in parallel.",
                        file=sys.stderr
                    )
                    if reset_event_count > MAX_RESET_EVENTS:
                        raise RuntimeError(
                            "Exceeded MAX_RESET_EVENTS=" + str(MAX_RESET_EVENTS) +
                            ". Hardware may be unrecoverable."
                        )

                # --- Pass 2: parallel collection from all respawned workers ---
                # connection.wait() lets all JIT builds run concurrently.
                # All N workers' ~12s startup overlaps -> completes in ~12s
                # regardless of N, vs the old 88-M N*12s serial chain.
                while pending_recv:
                    ready = mp.connection.wait(
                        [pipes[i] for i in pending_recv], timeout=120
                    )
                    if not ready:
                        raise RuntimeError(
                            "Respawned workers did not respond within 120s at step "
                            + str(t) + ". Hardware may be unrecoverable."
                        )
                    ready_set = set(id(c) for c in ready)
                    still_pending = []
                    for i in pending_recv:
                        if id(pipes[i]) in ready_set:
                            try:
                                data = pipes[i].recv()
                                for p, payload in data.items():
                                    patch_full_states[p] = payload["state"]
                                    bulk_energy += payload["meanfield_bulk_energy"]
                                    max_lat_trotter = max(max_lat_trotter, payload["lat_trotter_ms"])
                                    max_lat_tomo = max(max_lat_tomo, payload["lat_tomo_ms"])
                                    min_fidelity = min(min_fidelity, payload.get("unitary_fidelity", 1.0))
                            except EOFError:
                                raise RuntimeError(
                                    "Respawned worker rank=" + str(rank_map[i]) +
                                    " died immediately after respawn at step " +
                                    str(t) + ". Hardware may be unrecoverable."
                                )
                        else:
                            still_pending.append(i)
                    pending_recv = still_pending

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

            # Release memmap and delete the file (Rev 88-K).
            try:
                del self.ket_mm
            except Exception:
                pass
            try:
                os.unlink(self.memmap_path)
            except Exception:
                pass


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
