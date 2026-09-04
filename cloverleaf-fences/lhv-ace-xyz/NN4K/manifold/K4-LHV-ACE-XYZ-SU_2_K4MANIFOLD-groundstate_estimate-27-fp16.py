# -*- coding: us-ascii -*-
# K4 All-Boundary Manifold Cluster Annealing
# (4 Tri-Hole Manifolds x 15 Qubits, 6 Junctions, 30 Unique Qubits)
# High-Throughput Volumetric Engine with Statistical Variance Injection
#
# REVISION 310 - HANG PREVENTION REFINEMENTS (PART 2)
# CHANGES vs Rev 309:
# - Worker tomo: Guarded zz_exact call to prevent hanging on a known-broken GPU.
#
# CHANGES vs Rev 308:
# - Worker reinit: Moved sim deletion and gc.collect() BEFORE the cooldown sleep.
# - Worker sanity fail: Replaced broken sim with a dummy to ensure fast-fail on next step.
#
# CHANGES vs Rev 307:
# - Master gather: Replaced blocking recv() with mpc.wait() + timeout.
# - Worker reinit: Fast single-qubit sanity check to prevent infinite kernel hang.
# - Worker cooldown: Exponential backoff for consecutive driver timeouts.
#
# CHANGES vs Rev 305:
# - Master gather: NaN detection + last-good-state substitution.
# - Noise injection: np.where NaN guard on variance.
# - apply_kicks: isfinite guard on all three components.
# - Worker tomo: NaN detection + sim reinit to |+>^15 + Y_arr guard.
# - bulk_energy: skip non-finite contributions in gather.
#
# WHY 15 NOT 16:
#   K4 requires each manifold to have exactly LOCAL_EDGES=3 local edges
#   (degree-3 complete graph). QUBITS_PER_MANIFOLD must equal
#   LOCAL_EDGES * QUBITS_PER_EDGE and therefore be divisible by 3.
#   16 is not divisible by 3. 15 (3x5) is the nearest valid point
#   below 16 and is the smallest size that keeps C_CORNER=2 meaningful
#   for the Moebius twist pairing.
#
# FP16 PRECISION BUDGET AT 15 QUBITS:
#   Raw statevector: 2^15 * 4 bytes (complex<half>) = 128 KB per manifold.
#   fp16 mantissa (10 bits) gives amplitude resolution ~1e-3.
#   Expectation value rounding: ~1/sqrt(2^15) * fp16_eps ~ 0.001.
#   Shot noise floor: sqrt(dt/effective_shots) = sqrt(0.04/512) ~ 0.009.
#   Rounding is ~9x below shot noise -> fp16 does not dominate noise budget.
#
# MANIFOLD_VRAM_MB: 1024 -> 256.
#
# TOPOLOGY: tri-hole manifold with all-boundary property unchanged.
#   Each manifold: 3 portal rings of 10 qubits, 18 intra-manifold rim edges.
#   Every qubit belongs to exactly 2 portals. No bulk.
#   K4 wiring: 6 junctions x 5 qubits each.
#   Moebius twist: corners [1,0], mid-edges [4,3,2].
#   Racetrack loop and chord junctions: unchanged.
#
# ALL OTHER PHYSICS UNCHANGED FROM Rev 304:
#   fp16 FPPOW=4 Qrack library, FP16_LIB env override.
#   zz_exact two-body correlator, probe step IPC sync, swap asymmetry fix.
#   Stochastic scaling, continuous coupling, Pauli autodetect.

import os
import sys
import gc
import csv
import json
import time
import math
import numpy as np
import multiprocessing as mp
import multiprocessing.connection as mpc
from typing import List, Tuple, Dict, Any

# --- GLOBAL CONFIGURATION ---
C_CORNER  = 2                                    # corner qubits per intra-manifold edge
E_MIDEDGE = 3                                    # mid-edge qubits per intra-manifold edge
QUBITS_PER_EDGE     = C_CORNER + E_MIDEDGE       # 5
LOCAL_EDGES         = 3                           # 3 portals -> 3 intra edges
QUBITS_PER_MANIFOLD = LOCAL_EDGES * QUBITS_PER_EDGE   # 15
N_RIM = 2 * QUBITS_PER_EDGE                      # 10 slots per portal ring

N_MANIFOLDS    = 4                               # K4
NEIGHBOURS     = [[j for j in range(N_MANIFOLDS) if j != i]
                  for i in range(N_MANIFOLDS)]
JUNCTION_PAIRS = [(i, j) for i in range(N_MANIFOLDS)
                  for j in range(i + 1, N_MANIFOLDS)]   # 6 junctions
N_JUNCTIONS    = len(JUNCTION_PAIRS)

LOOP_CYCLE     = [0, 1, 2, 3]
LOOP_JUNCTIONS = [tuple(sorted((LOOP_CYCLE[i], LOOP_CYCLE[(i + 1) % 4])))
                  for i in range(4)]
CHORD_JUNCTIONS = [jp for jp in JUNCTION_PAIRS if jp not in LOOP_JUNCTIONS]

MOEBIUS_PAIRS = [(0, 1), (2, 3)]
TWIST_GLUING  = True

GPUS_AVAILABLE   = 4
WORKERS_PER_GPU  = 1
TOTAL_WORKERS    = GPUS_AVAILABLE * WORKERS_PER_GPU

# fp16: 2^15 * 4 bytes (complex<half>) = 128 KB raw; 256 MB cap is ample.
MANIFOLD_VRAM_MB = 256

# fp16 library (FPPOW=4 build). Override with QRACK_FP16_LIB if needed.
FP16_LIB = os.environ.get(
    "QRACK_FP16_LIB",
    "/usr/local/lib/qrack/libqrack_pinvoke.so"
)

UNIQUE_QUBITS = (N_MANIFOLDS * QUBITS_PER_MANIFOLD
                 - N_JUNCTIONS * QUBITS_PER_EDGE)   # 30

# Expected intra-manifold rim edge count: 3 * (QUBITS_PER_EDGE + 1)
_EXPECTED_RIM_EDGES = 3 * (QUBITS_PER_EDGE + 1)     # 18

GATHER_TIMEOUT_S    = 30.0
NAN_COOLDOWN_BASE_S = 1.5

assert QUBITS_PER_MANIFOLD <= 29, "29-qubit fp16 ceiling for a 2 GB statevector"
assert N_MANIFOLDS - 1 <= LOCAL_EDGES, "K_N needs degree N-1 <= 3 local edges"
assert TOTAL_WORKERS >= N_MANIFOLDS, "need at least one worker per manifold"

# =====================================================================
# ENVIRONMENT - set before pyqrack import
# =====================================================================
os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"

# =====================================================================
# PURE FUNCTIONS (Math & Topology)
# =====================================================================
def edge_local_qubits(k: int) -> List[int]:
    """Local qubit indices of intra-manifold edge k (corners then mid-edge)."""
    base = k * QUBITS_PER_EDGE
    return list(range(base, base + QUBITS_PER_EDGE))


def portal_ring(p: int) -> List[int]:
    """The 10 local qubits of portal p, in ring order."""
    return edge_local_qubits((p - 1) % LOCAL_EDGES) + edge_local_qubits(p)


def generate_manifold_topology() -> Tuple[List[Tuple[int, int]],
                                          Dict[int, List[int]]]:
    """Tri-hole manifold: 3 portal rings of 10 qubits, 15 qubits total.
    Intra-manifold rim edge count: 3*(QUBITS_PER_EDGE+1) = 18."""
    edges = set()
    for p in range(LOCAL_EDGES):
        ring = portal_ring(p)
        for s in range(len(ring)):
            a, b = ring[s], ring[(s + 1) % len(ring)]
            edges.add((min(a, b), max(a, b)))
    intra_edges = sorted(edges)

    if len(intra_edges) != _EXPECTED_RIM_EDGES:
        raise RuntimeError(
            f"Fatal: expected {_EXPECTED_RIM_EDGES} intra-manifold rim edges, "
            f"got {len(intra_edges)}")
    seen_q = set()
    for a, b in intra_edges:
        seen_q.add(a); seen_q.add(b)
    if len(seen_q) != QUBITS_PER_MANIFOLD:
        raise RuntimeError(
            f"Fatal: rim edges cover {len(seen_q)} qubits, "
            f"expected {QUBITS_PER_MANIFOLD}")
    in_portals: Dict[int, int] = {q: 0 for q in range(QUBITS_PER_MANIFOLD)}
    for p in range(LOCAL_EDGES):
        for q in portal_ring(p):
            in_portals[q] += 1
    if any(v != 2 for v in in_portals.values()):
        raise RuntimeError("Fatal: all-boundary property violated "
                           "(qubit not in exactly 2 portals)")
    adj: Dict[int, set] = {q: set() for q in range(QUBITS_PER_MANIFOLD)}
    for a, b in intra_edges:
        adj[a].add(b); adj[b].add(a)
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v); stack.append(v)
    if len(seen) != QUBITS_PER_MANIFOLD:
        raise RuntimeError("Fatal: manifold coupling graph is disconnected")

    edge_groups = {k: edge_local_qubits(k) for k in range(LOCAL_EDGES)}
    return intra_edges, edge_groups


def junction_slot(manifold: int, neighbour: int) -> int:
    return NEIGHBOURS[manifold].index(neighbour)


def junction_pairing(twist: bool) -> List[int]:
    """Moebius half-twist: corners [1,0], mid-edges [4,3,2] for 5-qubit edges."""
    if not twist:
        return list(range(QUBITS_PER_EDGE))
    corners = list(range(C_CORNER))[::-1]
    midedge = [C_CORNER + i for i in range(E_MIDEDGE)][::-1]
    return corners + midedge


def build_junction_table(twist: bool) -> List[Dict[str, Any]]:
    pairing = junction_pairing(twist)
    table = []
    for ji, (ma, mb) in enumerate(JUNCTION_PAIRS):
        ka = junction_slot(ma, mb)
        kb = junction_slot(mb, ma)
        qa_list = edge_local_qubits(ka)
        qb_list = edge_local_qubits(kb)
        bonds = [(qa_list[i], qb_list[pairing[i]])
                 for i in range(QUBITS_PER_EDGE)]
        table.append({
            "index": ji,
            "ma": ma, "mb": mb,
            "slot_a": ka, "slot_b": kb,
            "qubits_a": qa_list, "qubits_b": qb_list,
            "bonds": bonds,
            "label": f"J{ma}{mb}",
        })
    used: Dict[int, List[int]] = {m: [] for m in range(N_MANIFOLDS)}
    for rec in table:
        used[rec["ma"]].append(rec["slot_a"])
        used[rec["mb"]].append(rec["slot_b"])
    for m, slots in used.items():
        if sorted(slots) != list(range(LOCAL_EDGES)):
            raise RuntimeError(
                f"Fatal: manifold {m} local edge usage {sorted(slots)} "
                f"!= {list(range(LOCAL_EDGES))}")
    return table


def compute_kicks_and_junction_energy(
        profiles: Dict[int, Dict[str, np.ndarray]],
        noise: Dict[int, Dict[int, Tuple[float, float, float]]],
        junctions: List[Dict[str, Any]],
        g_junction: float
        ) -> Tuple[Dict[int, Dict[int, Tuple[float, float, float]]],
                   float, Dict[int, np.ndarray]]:
    kicks: Dict[int, Dict[int, Tuple[float, float, float]]] = {
        m: {} for m in range(N_MANIFOLDS)}

    def site_vec(m, q):
        prof = profiles[m]
        nx, ny, nz = noise[m].get(q, (0.0, 0.0, 0.0))
        return (float(prof["X"][q]) + nx,
                float(prof["Y"][q]) + ny,
                float(prof["Z"][q]) + nz)

    def add_kick(m, q, g, vec):
        k = kicks[m].get(q, (0.0, 0.0, 0.0))
        kicks[m][q] = (k[0] + g * vec[0],
                       k[1] + g * vec[1],
                       k[2] + g * vec[2])

    energy = 0.0
    junction_vectors: Dict[int, np.ndarray] = {}

    for rec in junctions:
        ma, mb = rec["ma"], rec["mb"]
        acc    = np.zeros(3)
        n_sites = 0
        for (qa, qb) in rec["bonds"]:
            va = site_vec(ma, qa)
            vb = site_vec(mb, qb)
            energy += -g_junction * (va[0] * vb[0]
                                     + va[1] * vb[1]
                                     + va[2] * vb[2])
            add_kick(ma, qa, g_junction, vb)
            add_kick(mb, qb, g_junction, va)
            acc += np.array(va) + np.array(vb)
            n_sites += 2
        junction_vectors[rec["index"]] = acc / max(1, n_sites)

    return kicks, energy, junction_vectors


def loop_momentum_amplitudes(junction_vectors: Dict[int, np.ndarray],
                             loop_indices: List[int]) -> np.ndarray:
    n    = len(loop_indices)
    data = np.array([junction_vectors[ji] for ji in loop_indices])
    amps = np.zeros((n, 3))
    for k in range(n):
        phase = np.exp(-2j * math.pi * k * np.arange(n) / n)
        for c in range(3):
            amps[k, c] = abs(np.dot(phase, data[:, c])) / n
    return amps


def swap_asymmetry(profiles: Dict[int, Dict[str, np.ndarray]],
                   junctions: List[Dict[str, Any]],
                   pairs: List[Tuple[int, int]]) -> Dict[str, float]:
    out     = {}
    by_pair = {(r["ma"], r["mb"]): r for r in junctions}
    for (ma, mb) in pairs:
        rec = by_pair.get((min(ma, mb), max(ma, mb)))
        if rec is None:
            continue
        diffs = []
        for (qa, qb) in rec["bonds"]:
            if rec["ma"] == ma:
                pa, pb, ia, ib = profiles[ma], profiles[mb], qa, qb
            else:
                pa, pb, ia, ib = profiles[mb], profiles[ma], qa, qb
            for comp in ("X", "Y", "Z"):
                diffs.append(float(pa[comp][ia]) - float(pb[comp][ib]))
        arr = np.array(diffs)
        out[f"M{ma}M{mb}"] = float(np.sqrt(np.mean(arr ** 2)))
    return out


# =====================================================================
# WORKER PROCESS LOGIC
# =====================================================================
def gpu_worker_process(
    rank: int,
    workers_per_gpu: int,
    assigned_manifolds: List[int],
    conn,
    dt: float,
    total_steps: int,
    initial_hx: float,
    target_J: float,
    target_hx: float,
    target_hz: float,
    measure_every: int,
    probe_step: int,
    fp16_lib: str
):
    os.environ["PYQRACK_SHARED_LIB_PATH"]         = fp16_lib
    os.environ["OCL_ICD_PLATFORM_SORT"]           = "none"

    physical_gpu_index = rank // workers_per_gpu
    os.environ["QRACK_OCL_DEFAULT_DEVICE"]        = str(physical_gpu_index)
    os.environ["QRACK_QPAGER_DEVICES"]            = str(physical_gpu_index)
    os.environ["QRACK_QUNITMULTI_DEVICES"]        = str(physical_gpu_index)

    alloc_mb = MANIFOLD_VRAM_MB * max(1, len(assigned_manifolds))
    os.environ["QRACK_MAX_ALLOC_MB"]              = str(alloc_mb)
    os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"

    import pyqrack
    from pyqrack import QrackSimulator

    sims = {}

    try:
        # --- PAULI CODE AUTODETECT ---
        _THRESH = 0.5

        _probe_z = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
        vals0_z  = {}
        for _code in range(8):
            try: vals0_z[_code] = _probe_z.pauli_expectation([0], [_code])
            except Exception: pass
        _probe_z.x(0)
        PZ, SIGN_Z = None, None
        for _code, v0 in vals0_z.items():
            try: v1 = _probe_z.pauli_expectation([0], [_code])
            except Exception: continue
            if abs(v0) > _THRESH and abs(v1) > _THRESH and (v0 * v1) < 0:
                PZ    = _code
                SIGN_Z = 1.0 if v0 > 0 else -1.0
                break
        if PZ is None:
            raise RuntimeError("Fatal: could not autodetect PZ code")
        del _probe_z

        _probe_x = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
        _probe_x.h(0)
        vals0_x  = {}
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
                PX    = _code
                SIGN_X = 1.0 if v0 > 0 else -1.0
                break
        if PX is None:
            raise RuntimeError("Fatal: could not autodetect PX code")
        del _probe_x

        _probe_y = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
        _c, _s   = math.cos(math.pi / 4.0), math.sin(math.pi / 4.0)
        _probe_y.mtrx([complex(_c, 0), complex(0, _s),
                       complex(0, _s), complex(_c, 0)], 0)
        vals0_y  = {}
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
                PY    = _code
                SIGN_Y = 1.0 if v0 > 0 else -1.0
                break
        if PY is None:
            raise RuntimeError("Fatal: could not autodetect PY code")
        del _probe_y

        # --- ANGLE CONVENTION AUTODETECT ---
        _sim_mag   = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
        _sim_mag.r(PX, math.pi, 0)
        _corrected = SIGN_Z * _sim_mag.pauli_expectation([0], [PZ])
        if abs(_corrected + 1.0) < 0.1:
            ANGLE_SCALE = 1.0
        elif abs(_corrected - 1.0) < 0.1:
            ANGLE_SCALE = 0.5
        else:
            raise RuntimeError(
                f"Fatal: r(PX,pi) returned ambiguous SIGN_Z*<Z> = {_corrected:.6f}; "
                f"expected ~+1.0 or ~-1.0")
        del _sim_mag

        def apply_h(sim, q): sim.h(q)

        def apply_rx(sim, theta, q):
            sim.r(PX, float(theta) * ANGLE_SCALE, q)

        def apply_ry(sim, theta, q):
            sim.r(PY, float(theta) * ANGLE_SCALE, q)

        def apply_rz(sim, theta, q):
            sim.r(PZ, float(theta) * ANGLE_SCALE, q)

        def apply_zz(sim, theta, q1, q2):
            sim.mcx([q1], q2); apply_rz(sim, 2.0 * theta, q2); sim.mcx([q1], q2)

        def trotter_step_body(sim, num_qubits, intra_edges, J, hx, hz, dt_local):
            dt_half  = dt_local / 2.0
            theta_x  = -2.0 * hx * dt_half
            theta_z  = -2.0 * hz * dt_local
            theta_zz = -J * dt_local
            for q in range(num_qubits):    apply_rx(sim, theta_x, q)
            for q in range(num_qubits):    apply_rz(sim, theta_z, q)
            for q1, q2 in intra_edges:     apply_zz(sim, theta_zz, q1, q2)
            for q in range(num_qubits):    apply_rx(sim, theta_x, q)

        def z_means(sim, qubits):
            return np.array([SIGN_Z * float(sim.pauli_expectation([q], [PZ]))
                             for q in qubits])

        def x_means(sim, qubits):
            return np.array([SIGN_X * float(sim.pauli_expectation([q], [PX]))
                             for q in qubits])

        def y_means(sim, qubits):
            return np.array([SIGN_Y * float(sim.pauli_expectation([q], [PY]))
                             for q in qubits])

        def zz_exact(sim, edges):
            return np.array([float(sim.pauli_expectation([q1, q2], [PZ, PZ]))
                             for q1, q2 in edges])

        def apply_kicks(sim, kicks, dt_local):
            if not kicks: return
            for raw_q, (kx, ky, kz) in kicks.items():
                q    = int(raw_q)
                coef = -2.0 * dt_local
                if not math.isfinite(kx): kx = 0.0   # NaN guard
                if not math.isfinite(ky): ky = 0.0
                if not math.isfinite(kz): kz = 0.0
                if abs(kx * coef) > 1e-12: apply_rx(sim, kx * coef, q)
                if abs(ky * coef) > 1e-12: apply_ry(sim, ky * coef, q)
                if abs(kz * coef) > 1e-12: apply_rz(sim, kz * coef, q)

        intra_edges, _edge_groups = generate_manifold_topology()

        for m in assigned_manifolds:
            sim = QrackSimulator(
                qubit_count=QUBITS_PER_MANIFOLD,
                is_binary_decision_tree=False,
                is_stabilizer_hybrid=False,
                is_gpu=True,
            )
            for q in range(QUBITS_PER_MANIFOLD): apply_h(sim, q)
            sims[m] = sim
            try:
                _ = sim.pauli_expectation([0], [PZ])
            except Exception as e:
                raise RuntimeError(
                    f"Fatal: VRAM allocation failed on manifold {m} "
                    f"(rank {rank}, gpu {physical_gpu_index}): {e}")

        kick_payloads    = {m: {} for m in assigned_manifolds}
        _warned_fidelity = False
        
        nan_streak = {m: 0 for m in assigned_manifolds}

        for t in range(total_steps):
            s          = t / max(1, (total_steps - 1))
            current_hx = (1.0 - s) * initial_hx + s * target_hx
            current_J  = s * target_J
            current_hz = s * target_hz
            is_measure = ((t % measure_every == 0)
                          or (t == total_steps - 1)
                          or (t == probe_step))

            manifold_data_to_master = {}

            for m in assigned_manifolds:
                sim = sims[m]
                if kick_payloads[m]:
                    apply_kicks(sim, kick_payloads[m], dt)

                t_start_trotter = time.perf_counter()
                trotter_step_body(sim, QUBITS_PER_MANIFOLD, intra_edges,
                                  current_J, current_hx, current_hz, dt)
                t_lat_trotter = time.perf_counter() - t_start_trotter

                t_lat_tomo = 0.0
                if is_measure:
                    t_start_tomo = time.perf_counter()
                    all_q  = list(range(QUBITS_PER_MANIFOLD))
                    
                    Z_arr = z_means(sim, all_q)
                    X_arr = x_means(sim, all_q)
                    Y_arr = y_means(sim, all_q)
                    
                    if not (np.all(np.isfinite(Z_arr)) and np.all(np.isfinite(X_arr)) and np.all(np.isfinite(Y_arr))):
                        nan_streak[m] += 1
                        print(f"[Worker {rank}] NaN tomo on M{m} at t={t}; reinit to |+>^{QUBITS_PER_MANIFOLD}",
                              file=sys.stderr)
                        
                        try:
                            del sims[m]
                        except KeyError:
                            pass
                        gc.collect()
                        
                        if nan_streak[m] > 1:
                            sleep_s = NAN_COOLDOWN_BASE_S * nan_streak[m]
                            print(f"[Worker {rank}] NaN streak {nan_streak[m]} on M{m}; cooldown {sleep_s:.1f}s", file=sys.stderr)
                            time.sleep(sleep_s)
                            
                        sim = QrackSimulator(qubit_count=QUBITS_PER_MANIFOLD,
                                             is_binary_decision_tree=False,
                                             is_stabilizer_hybrid=False, is_gpu=True)
                        for q in range(QUBITS_PER_MANIFOLD):
                            apply_h(sim, q)
                        sims[m] = sim
                        
                        _sanity = float(sim.pauli_expectation([0], [PX]))
                        if not math.isfinite(_sanity) or abs(_sanity - SIGN_X) > 0.5:
                            print(f"[Worker {rank}] GPU still broken after reinit M{m} t={t}; sleeping 5s and returning NaN", file=sys.stderr)
                            time.sleep(5.0)
                            
                            try:
                                del sims[m]
                            except KeyError:
                                pass
                            gc.collect()
                            
                            sim = QrackSimulator(qubit_count=QUBITS_PER_MANIFOLD,
                                                 is_binary_decision_tree=False,
                                                 is_stabilizer_hybrid=False, is_gpu=True)
                            sims[m] = sim
                            
                            Z_arr = np.full(QUBITS_PER_MANIFOLD, float('nan'))
                            X_arr = np.full(QUBITS_PER_MANIFOLD, float('nan'))
                            Y_arr = np.full(QUBITS_PER_MANIFOLD, float('nan'))
                        else:
                            Z_arr = z_means(sim, all_q)
                            X_arr = x_means(sim, all_q)
                            Y_arr = y_means(sim, all_q)
                    else:
                        nan_streak[m] = 0
                        
                    state = {"Z": Z_arr, "X": X_arr, "Y": Y_arr}
                    
                    if np.all(np.isfinite(Z_arr)):
                        zz_exp = zz_exact(sim, intra_edges)
                    else:
                        zz_exp = np.zeros(len(intra_edges))
                        
                    bulk_e = (-current_hz * float(np.sum(state["Z"]))
                              - current_J  * float(np.sum(zz_exp))
                              - current_hx * float(np.sum(state["X"])))
                    t_lat_tomo = time.perf_counter() - t_start_tomo

                    try:
                        fidelity = float(sim.get_unitary_fidelity())
                    except AttributeError:
                        fidelity = 1.0
                        if not _warned_fidelity:
                            print(f"[Worker {rank}] Warning: "
                                  f"sim.get_unitary_fidelity() not found.",
                                  file=sys.stderr)
                            _warned_fidelity = True

                    manifold_data_to_master[m] = {
                        "state":             state,
                        "exact_bulk_energy": bulk_e,
                        "lat_trotter_ms":    t_lat_trotter * 1000.0,
                        "lat_tomo_ms":       t_lat_tomo    * 1000.0,
                        "unitary_fidelity":  fidelity,
                    }

            if is_measure:
                conn.send(manifold_data_to_master)
                kick_payloads = conn.recv()

    finally:
        for m in list(sims.keys()):
            _s = sims.pop(m)
            del _s
        sims.clear()
        gc.collect()
        conn.close()


# =====================================================================
# MASTER ORCHESTRATOR
# =====================================================================
class K4ManifoldEngine:
    def __init__(self, master_seed: int = 1337, twist: bool = TWIST_GLUING):
        self.master_seed = master_seed
        self.twist       = twist
        self.intra_edges, self.edge_groups = generate_manifold_topology()
        self.junctions        = build_junction_table(twist)
        self.junction_by_pair = {(r["ma"], r["mb"]): r for r in self.junctions}
        self.loop_indices     = [self.junction_by_pair[jp]["index"]
                                 for jp in LOOP_JUNCTIONS]
        self.chord_indices    = [self.junction_by_pair[jp]["index"]
                                 for jp in CHORD_JUNCTIONS]

        self.lattice_history  = []
        self.energy_csv       = "k4_exact_ground_state_energy_curve.csv"
        self.profiles_csv     = "k4_junction_profiles.csv"
        self.loop_csv         = "k4_loop_momentum_spectrum.csv"
        self.state_dump_file  = "k4_manifold_states.npy"
        self.config_file      = "k4_manifold_config.json"

        self._init_files()

        self.worker_assignments = [[] for _ in range(TOTAL_WORKERS)]
        for m in range(N_MANIFOLDS):
            self.worker_assignments[m % TOTAL_WORKERS].append(m)

    def _get_last_good(self, m):
        if self.lattice_history:
            last = self.lattice_history[-1]
            return {"X": last[m, :, 0].copy(),
                    "Y": last[m, :, 1].copy(),
                    "Z": last[m, :, 2].copy()}
        return {"X": np.zeros(QUBITS_PER_MANIFOLD),
                "Y": np.zeros(QUBITS_PER_MANIFOLD),
                "Z": np.ones(QUBITS_PER_MANIFOLD)}

    def _init_files(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump({
                    "topology":            "K4_trihole_manifold_cluster",
                    "revision":            310,
                    "precision":           "fp16_FPPOW4",
                    "n_manifolds":         N_MANIFOLDS,
                    "qubits_per_manifold": QUBITS_PER_MANIFOLD,
                    "corner_per_edge":     C_CORNER,
                    "midedge_per_edge":    E_MIDEDGE,
                    "qubits_per_junction": QUBITS_PER_EDGE,
                    "n_junctions":         N_JUNCTIONS,
                    "unique_qubits":       UNIQUE_QUBITS,
                    "intra_edges":         len(self.intra_edges),
                    "twist_gluing":        self.twist,
                    "loop_junctions":      [f"J{a}{b}" for a, b in LOOP_JUNCTIONS],
                    "chord_junctions":     [f"J{a}{b}" for a, b in CHORD_JUNCTIONS],
                    "moebius_pairs":       [f"M{a}M{b}" for a, b in MOEBIUS_PAIRS],
                    "gpus":                GPUS_AVAILABLE,
                    "manifold_vram_mb":    MANIFOLD_VRAM_MB,
                    "fp16_lib":            FP16_LIB,
                }, f)
            with open(self.energy_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Anneal_Percent", "Exact_Bulk_Energy",
                    "Junction_Energy", "Total_Energy",
                    "Min_Unitary_Fidelity", "Swap_Asym_M0M1", "Swap_Asym_M2M3"
                ]).writeheader()
            with open(self.profiles_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Junction", "Kind", "X_mean", "Y_mean", "Z_mean"
                ]).writeheader()
            with open(self.loop_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Loop_Momentum", "Amp_X", "Amp_Y", "Amp_Z"
                ]).writeheader()
        except Exception as e:
            print(f"[CSV] Warning: setup write failed: {e}", file=sys.stderr)

    def _log_csvs(self, step, anneal, bulk, junction_e, total, min_fidelity,
                  junction_vectors, loop_amps, asym):
        try:
            with open(self.energy_csv, mode='a', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Anneal_Percent", "Exact_Bulk_Energy",
                    "Junction_Energy", "Total_Energy",
                    "Min_Unitary_Fidelity", "Swap_Asym_M0M1", "Swap_Asym_M2M3"
                ]).writerow({
                    "Step": step, "Anneal_Percent": anneal,
                    "Exact_Bulk_Energy": bulk,
                    "Junction_Energy":   junction_e,
                    "Total_Energy":      total,
                    "Min_Unitary_Fidelity": min_fidelity,
                    "Swap_Asym_M0M1": asym.get("M0M1", float('nan')),
                    "Swap_Asym_M2M3": asym.get("M2M3", float('nan'))})

            with open(self.profiles_csv, mode='a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=[
                    "Step", "Junction", "Kind", "X_mean", "Y_mean", "Z_mean"])
                for rec in self.junctions:
                    v    = junction_vectors[rec["index"]]
                    kind = "loop" if rec["index"] in self.loop_indices else "chord"
                    w.writerow({"Step": step, "Junction": rec["label"],
                                "Kind": kind,
                                "X_mean": float(v[0]),
                                "Y_mean": float(v[1]),
                                "Z_mean": float(v[2])})

            with open(self.loop_csv, mode='a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=[
                    "Step", "Loop_Momentum", "Amp_X", "Amp_Y", "Amp_Z"])
                for k in range(loop_amps.shape[0]):
                    w.writerow({"Step": step, "Loop_Momentum": k,
                                "Amp_X": float(loop_amps[k, 0]),
                                "Amp_Y": float(loop_amps[k, 1]),
                                "Amp_Z": float(loop_amps[k, 2])})
        except Exception as e:
            print(f"[CSV] Warning: log write failed: {e}", file=sys.stderr)

    def run(self, total_steps: int, dt: float, initial_hx: float,
            target_g_junction: float, target_J: float, target_hx: float,
            target_hz: float, measure_every: int = 1,
            effective_shots: float = 512.0,
            probe_step: int = -1, probe_amplitude: float = 0.0,
            probe_junction: int = 0):

        if total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        if measure_every < 1:
            raise ValueError("measure_every must be a positive integer")
        if not (0 <= probe_junction < N_JUNCTIONS):
            raise ValueError("probe_junction out of range")
        if probe_step >= 0 and probe_step >= total_steps:
            raise ValueError(
                f"probe_step {probe_step} >= total_steps {total_steps}")

        print(f"[Engine] K4 Rev310 fp16 15q | "
              f"{N_MANIFOLDS} manifolds x {QUBITS_PER_MANIFOLD} qubits | "
              f"{N_JUNCTIONS} junctions x {QUBITS_PER_EDGE} qubits | "
              f"{UNIQUE_QUBITS} unique qubits | "
              f"rim_edges={_EXPECTED_RIM_EDGES} | "
              f"gluing={'MOEBIUS' if self.twist else 'TRIVIAL'} | "
              f"{GPUS_AVAILABLE} GPUs | {total_steps} steps")

        active_ranks = [r for r in range(TOTAL_WORKERS)
                        if self.worker_assignments[r]]
        workers = []
        pipes   = []

        for rank in active_ranks:
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=gpu_worker_process,
                args=(rank, WORKERS_PER_GPU, self.worker_assignments[rank],
                      child_conn, dt, total_steps, initial_hx, target_J,
                      target_hx, target_hz, measure_every, probe_step,
                      FP16_LIB)
            )
            p.start()
            child_conn.close()
            workers.append(p)
            pipes.append(parent_conn)

        try:
            for t in range(total_steps):
                s = t / max(1, (total_steps - 1))
                current_g_junction = s * target_g_junction
                is_measure = ((t % measure_every == 0)
                              or (t == total_steps - 1)
                              or (t == probe_step))

                if not is_measure:
                    continue

                t0 = time.perf_counter()

                # --- GATHER ---
                manifold_states = {}
                bulk_energy     = 0.0
                max_lat_trotter = 0.0
                max_lat_tomo    = 0.0
                min_fidelity    = 1.0

                pending = list(zip(pipes, active_ranks))
                deadline = time.perf_counter() + GATHER_TIMEOUT_S

                while pending:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        for conn, w_rank in pending:
                            print(f"[Master] TIMEOUT rank {w_rank}; substituting last-good",
                                  file=sys.stderr)
                            for m in self.worker_assignments[w_rank]:
                                manifold_states[m] = self._get_last_good(m)
                        break
                        
                    ready = mpc.wait([c for c, _ in pending], timeout=min(remaining, 5.0))
                    
                    for conn in ready:
                        w_rank = next(r for c, r in pending if c is conn)
                        try:
                            data = conn.recv()
                            for m, payload in data.items():
                                state = payload["state"]
                                if any(np.any(~np.isfinite(state[c])) for c in ("X", "Y", "Z")):
                                    print(f"[Master] NaN M{m} step {t}; substituting last-good", file=sys.stderr)
                                    state = self._get_last_good(m)
                                    
                                manifold_states[m] = state
                                bulk_energy += payload["exact_bulk_energy"] if math.isfinite(
                                    payload["exact_bulk_energy"]) else 0.0
                                max_lat_trotter = max(max_lat_trotter, payload["lat_trotter_ms"])
                                max_lat_tomo    = max(max_lat_tomo,    payload["lat_tomo_ms"])
                                min_fidelity    = min(min_fidelity,    payload.get("unitary_fidelity", 1.0))
                                
                        except EOFError:
                            print(f"[Master] EOF rank {w_rank}; substituting last-good", file=sys.stderr)
                            for m in self.worker_assignments[w_rank]:
                                manifold_states[m] = self._get_last_good(m)
                                
                        pending = [(c, r) for c, r in pending if c is not conn]

                for m in range(N_MANIFOLDS):
                    if m not in manifold_states:
                        manifold_states[m] = self._get_last_good(m)

                # --- HISTORY ---
                step_state = np.zeros((N_MANIFOLDS, QUBITS_PER_MANIFOLD, 3))
                for m, state in manifold_states.items():
                    step_state[m, :, 0] = state["X"]
                    step_state[m, :, 1] = state["Y"]
                    step_state[m, :, 2] = state["Z"]
                self.lattice_history.append(step_state.copy())

                if len(self.lattice_history) % 10 == 0:
                    try:
                        np.save(self.state_dump_file,
                                np.array(self.lattice_history))
                    except Exception as e:
                        print(f"[Checkpoint] Warning: {e}", file=sys.stderr)

                # --- STOCHASTIC NOISE ---
                scale = np.sqrt(dt / effective_shots)
                noise = {m: {} for m in range(N_MANIFOLDS)}
                for rec in self.junctions:
                    rng_j = np.random.default_rng(
                        [self.master_seed, t, rec["index"]])
                    for side, mm, qlist in (("a", rec["ma"], rec["qubits_a"]),
                                            ("b", rec["mb"], rec["qubits_b"])):
                        prof = manifold_states[mm]
                        idx  = np.array(qlist, dtype=np.intp)
                        px = prof["X"][idx]; py = prof["Y"][idx]; pz = prof["Z"][idx]
                        vx = np.where(np.isfinite(px), np.clip(1.0 - px ** 2, 0.0, 1.0), 1.0)
                        vy = np.where(np.isfinite(py), np.clip(1.0 - py ** 2, 0.0, 1.0), 1.0)
                        vz = np.where(np.isfinite(pz), np.clip(1.0 - pz ** 2, 0.0, 1.0), 1.0)
                        n    = len(qlist)
                        xn   = rng_j.normal(0.0, 1.0, n) * np.sqrt(vx) * scale
                        yn   = rng_j.normal(0.0, 1.0, n) * np.sqrt(vy) * scale
                        zn   = rng_j.normal(0.0, 1.0, n) * np.sqrt(vz) * scale
                        for i, q in enumerate(qlist):
                            noise[mm][q] = (xn[i], yn[i], zn[i])

                # --- KICKS & JUNCTION ENERGY ---
                kicks, junction_energy, junction_vectors = \
                    compute_kicks_and_junction_energy(
                        manifold_states, noise, self.junctions,
                        current_g_junction)

                # --- PROBE ---
                if probe_step >= 0 and t == probe_step and probe_amplitude != 0.0:
                    rec    = self.junctions[probe_junction]
                    qa, qb = rec["bonds"][0]
                    for (mm, qq) in ((rec["ma"], qa), (rec["mb"], qb)):
                        k = kicks[mm].get(qq, (0.0, 0.0, 0.0))
                        kicks[mm][qq] = (k[0] + probe_amplitude, k[1], k[2])
                    print(f"[Probe] step {t}: injected {probe_amplitude:+.4f} "
                          f"at {rec['label']} bond 0")

                # --- RACETRACK DIAGNOSTICS ---
                loop_amps = loop_momentum_amplitudes(junction_vectors,
                                                     self.loop_indices)
                asym      = swap_asymmetry(manifold_states, self.junctions,
                                           MOEBIUS_PAIRS)

                total_energy = bulk_energy + junction_energy
                asym_str = " ".join(
                    f"{k}:{v:.4f}" for k, v in sorted(asym.items()))
                print(f"Step {t:03d} | E: {total_energy:+.4f} "
                      f"(bulk {bulk_energy:+.3f} / junc {junction_energy:+.3f}) | "
                      f"Loop k1: {np.linalg.norm(loop_amps[1]):.4f} | "
                      f"Asym {asym_str} | "
                      f"Lat(Trot/Tomo): {max_lat_trotter:5.1f}/{max_lat_tomo:5.1f}ms | "
                      f"Fid: {min_fidelity:.5f} | {time.perf_counter() - t0:.2f}s")
                self._log_csvs(t, s * 100, bulk_energy, junction_energy,
                               total_energy, min_fidelity, junction_vectors,
                               loop_amps, asym)

                # --- SCATTER ---
                for i, w_rank in enumerate(active_ranks):
                    pipes[i].send({m: kicks[m]
                                   for m in self.worker_assignments[w_rank]})

        finally:
            for conn in pipes:
                try: conn.close()
                except Exception: pass

            if self.lattice_history:
                try:
                    np.save(self.state_dump_file,
                            np.array(self.lattice_history))
                    print(f"\n[Master] Dumped history to {self.state_dump_file}")
                except Exception as e:
                    print(f"\n[Master] Failed to save history: {e}",
                          file=sys.stderr)

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

    engine = K4ManifoldEngine(master_seed=1337, twist=TWIST_GLUING)
    try:
        engine.run(
            total_steps=100,
            dt=0.04,
            initial_hx=3.0,
            target_g_junction=0.15,
            target_J=1.0,
            target_hx=0.5,
            target_hz=0.2,
            measure_every=1,
            effective_shots=512.0,
            probe_step=-1,
            probe_amplitude=0.5,
            probe_junction=0
        )
    except KeyboardInterrupt:
        pass
