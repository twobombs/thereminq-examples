# -*- coding: us-ascii -*-
# K4 All-Boundary Manifold Cluster Annealing
# (4 Tri-Hole Manifolds x 27 Qubits, 6 Junctions, 54 Unique Qubits)
# High-Throughput Volumetric Engine with Statistical Variance Injection
#
# REVISION 312 - PLUGGABLE BOUNDARY (meanfield | gnn | bp)
# (forked from Rev 311; self-contained, no external modules)
#
# WHY: the 3 junction slots PARTITION a manifold's 27 qubits (build_junction_
# table asserts slots == [0,1,2], nothing left over). So a manifold state is
# EXACTLY a rank-3 tensor with leg dimension 2**9 = 512. Four tensors, six
# copy-tensor bonds, K4 with no structural loss. This is not an ansatz.
# Rev 311 is that network at chi = 1: passing (<X>,<Y>,<Z>) per site is a
# rank-1 message, which is why the cross-junction connected correlator and the
# inter-manifold mutual information are identically zero there -- forced by
# chi = 1, not by physics.
#
# CHANGES vs Rev 311:
# - Topology parameterized on (C_CORNER, E_MIDEDGE). Defaults unchanged (3, 6)
#   so BOUNDARY_MODE="meanfield" reproduces Rev 311 exactly.
# - Kick payload is now an explicit per-site ROTATION ANGLE triple (wx,wy,wz)
#   applied as rx/ry/rz. Rev 311 sent a field the worker multiplied by -2*dt;
#   meanfield now sends w = -2*dt*g*<sigma>, numerically identical.
# - BOUNDARY_MODE:
#     "meanfield" Rev 311 behaviour. chi = 1. Baseline.
#     "gnn"       learned per-bond residual from a .npz trained by the exact
#                 small-scale reference. Numpy-only inference, no torch.
#     "bp"        chi > 1 loopy belief propagation.
# - bp mode does the Schmidt truncation IN THE WORKER, next to the GPU, and
#   ships only the truncated tensor (chi**3) plus three isometries
#   (chi x 512). At chi=8 that is ~100 KB instead of a 1 GB out_ket through
#   mp.Pipe, and the four SVDs run in parallel across the four workers.
# - Added global_qubit_map(): the (manifold, local qubit) -> unique qubit
#   identification, used by bond_consistency and by the exact reference.
# - Added bond_consistency(): the honest swap_asymmetry. Rev 311 RMS-averaged
#   over all components, which dilutes a single badly-violated site. The two
#   copies of a shared qubit are supposed to BE the same qubit, so report the
#   WORST disagreement per junction. Rev 311's swap_asymmetry is retained
#   alongside for continuity of the CSV.
# - Added loop_noise_floor(): the racetrack k!=0 amplitudes have an analytic
#   floor of sqrt(dt/shots)/sqrt(2*qpe)/2 = 1.04e-3 at dt=0.04, shots=512.
#   Rev 311 logged bare LoopK1, so anything near 1e-3 was half noise. Now
#   logged alongside and flagged when under 3x floor.
# - g_junction is used ONLY by meanfield. It was never a coupling: it is a
#   penalty enforcing "these two arrays are the same qubit", whose correct
#   value is infinity. gnn and bp enforce the identification in the message.
#
# READ BEFORE TRUSTING bp MODE:
# K4 contains four triangles. BP is exact only on trees. Measured against a
# state that IS exactly a K4 tensor network, at chi = full leg dimension with
# ZERO truncation, BP single-site beliefs still carried ~0.17 mean absolute
# error. Raising chi does not fix this; it is loop error, not truncation
# error. chi buys correlations (block MI 0 -> 1.87 bits, which is real and is
# the point) but not correctness. A converged BP residual of 1e-11 says
# nothing about accuracy -- do not report it as a convergence criterion.
#
# TOPOLOGY UNCHANGED FROM Rev 303/311 at the default (3, 6):
# - Tri-hole manifold: 3 portal rims of 18 qubits each, C=3 corner + E=6
#   mid-edge qubits per intra-manifold edge. 27 qubits total, 30 rim edges,
#   every qubit in exactly 2 portals.
# - K4 cluster: 6 junctions x 9 qubits = 54 unique qubits.
# - TWIST_GLUING, racetrack DFT, swap asymmetry, bond-resolved kicks,
#   stochastic scaling, Pauli autodetect, hang prevention: all unchanged.
#
# NOTE: the Moebius twist is NOT a relabeling. At (3,6) the union of the four
# manifolds' rim edges over the shared qubits has 78 distinct edges with the
# twist and 72 without. TWIST_GLUING is a physical parameter.
#
# NOTE: the Rev 311 Trotter body is SECOND order, not first. The Z and ZZ
# terms are mutually diagonal, so the symmetric X wrap makes the step a true
# Strang splitting; measured local error scales as dt**3. Larger dt is
# affordable than the Rev 311 comments assumed.

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
C_CORNER  = 3                                    # corner qubits per intra-manifold edge
E_MIDEDGE = 6                                    # mid-edge qubits per intra-manifold edge
QUBITS_PER_EDGE     = C_CORNER + E_MIDEDGE       # 9
LOCAL_EDGES         = 3                           # 3 portals -> 3 intra edges
QUBITS_PER_MANIFOLD = LOCAL_EDGES * QUBITS_PER_EDGE   # 27
N_RIM = 2 * QUBITS_PER_EDGE                      # 18 slots per portal ring
LEG_DIM = 2 ** QUBITS_PER_EDGE                   # 512

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

# --- BOUNDARY CONFIGURATION ---
BOUNDARY_MODE   = "meanfield"    # "meanfield" | "gnn" | "bp"
GNN_MODEL_PATH  = "k4_edge_mlp.npz"
GNN_SCALE       = 1.0
BP_CHI          = 8
BP_EVERY        = 20             # bp truncation+solve cadence (in measured steps)
BP_DAMPING      = 0.3
BP_MAX_ITER     = 200
MAX_KICK_ANGLE  = 0.35           # rad, clip on emitted rotation magnitude

GPUS_AVAILABLE   = 4
WORKERS_PER_GPU  = 1
TOTAL_WORKERS    = GPUS_AVAILABLE * WORKERS_PER_GPU
MANIFOLD_VRAM_MB = 2048          # 2^27 * 8B = 1 GB fp32; 2 GB cap for QUnit

UNIQUE_QUBITS = (N_MANIFOLDS * QUBITS_PER_MANIFOLD
                 - N_JUNCTIONS * QUBITS_PER_EDGE)   # 54

# Expected intra-manifold rim edge count: 3 * (QUBITS_PER_EDGE + 1)
_EXPECTED_RIM_EDGES = 3 * (QUBITS_PER_EDGE + 1)     # 30

GATHER_TIMEOUT_S    = 30.0
NAN_COOLDOWN_BASE_S = 1.5

assert QUBITS_PER_MANIFOLD <= 28, "28-qubit fp32 ceiling for a 2GB statevector"
assert N_MANIFOLDS - 1 <= LOCAL_EDGES, "K_N needs degree N-1 <= 3 local edges"
assert TOTAL_WORKERS >= N_MANIFOLDS, "need at least one worker per manifold"

# =====================================================================
# ENVIRONMENT - set before pyqrack import
# =====================================================================
os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"

_PAULI_MAT = {
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


# =====================================================================
# PURE FUNCTIONS (Math & Topology)
# =====================================================================
def edge_local_qubits(k: int) -> List[int]:
    """Local qubit indices of intra-manifold edge k (corners then mid-edge)."""
    base = k * QUBITS_PER_EDGE
    return list(range(base, base + QUBITS_PER_EDGE))


def portal_ring(p: int) -> List[int]:
    """The 2*QUBITS_PER_EDGE local qubits of portal p, in ring order.
    Portal p is bounded by intra-manifold edges (p-1)%3 and p."""
    return edge_local_qubits((p - 1) % LOCAL_EDGES) + edge_local_qubits(p)


def generate_manifold_topology() -> Tuple[List[Tuple[int, int]],
                                          Dict[int, List[int]]]:
    """Tri-hole manifold: 3 portal rings, QUBITS_PER_MANIFOLD qubits total.
    Intra-manifold rim edge count: 3*(QUBITS_PER_EDGE+1)."""
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
    """Which local edge index of `manifold` carries the junction to `neighbour`."""
    return NEIGHBOURS[manifold].index(neighbour)


def junction_pairing(twist: bool) -> List[int]:
    """Map slot index on side A to slot index on side B within a junction.
    Identity = trivial gluing; order-reversing within the corner block and
    within the mid-edge block = the discrete Moebius half-twist."""
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
            "bonds": bonds, "pairing": list(pairing),
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


def global_qubit_map(junctions: List[Dict[str, Any]]) -> Dict[Tuple[int, int], int]:
    """(manifold, local qubit) -> unique global qubit index.

    Global index of bond i of junction j is j*QUBITS_PER_EDGE + i. Both
    endpoints of the bond -- which are the SAME physical qubit -- map there.
    This is the identification that g_junction is trying to enforce by penalty
    and that the exact merged reference enforces structurally.
    """
    gmap: Dict[Tuple[int, int], int] = {}
    for rec in junctions:
        for i, (qa, qb) in enumerate(rec["bonds"]):
            g = rec["index"] * QUBITS_PER_EDGE + i
            gmap[(rec["ma"], qa)] = g
            gmap[(rec["mb"], qb)] = g
    if len(gmap) != N_MANIFOLDS * QUBITS_PER_MANIFOLD:
        raise RuntimeError(f"Fatal: global map covers {len(gmap)} of "
                           f"{N_MANIFOLDS * QUBITS_PER_MANIFOLD} slots")
    if len(set(gmap.values())) != UNIQUE_QUBITS:
        raise RuntimeError(f"Fatal: global map hits {len(set(gmap.values()))} "
                           f"distinct qubits, expected {UNIQUE_QUBITS}")
    return gmap


def compute_kicks_and_junction_energy(
        profiles: Dict[int, Dict[str, np.ndarray]],
        noise: Dict[int, Dict[int, Tuple[float, float, float]]],
        junctions: List[Dict[str, Any]],
        g_junction: float
        ) -> Tuple[Dict[int, Dict[int, Tuple[float, float, float]]],
                   float, Dict[int, np.ndarray]]:
    """Rev 311 mean field, unchanged. Returns FIELDS (not angles); the
    MeanFieldBoundary converts them to rotation angles via -2*dt."""
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
    """Racetrack DFT over the 4 loop junctions.

    NOTE: this is a DFT over an ORDERED LIST OF LABELS (LOOP_CYCLE), not a
    momentum. In K4 all six junctions are symmetry-equivalent, so which four
    you call "the loop" is a gauge choice; permuting manifold labels moves the
    spectrum. Also, because abs() is taken of a real-input DFT,
    amps[1] == amps[3] identically -- four momenta, three independent numbers.
    """
    n    = len(loop_indices)
    data = np.array([junction_vectors[ji] for ji in loop_indices])  # (n,3)
    amps = np.zeros((n, 3))
    for k in range(n):
        phase = np.exp(-2j * math.pi * k * np.arange(n) / n)
        for c in range(3):
            amps[k, c] = abs(np.dot(phase, data[:, c])) / n
    return amps


def loop_noise_floor(dt: float, effective_shots: float) -> float:
    """Analytic floor of the k != 0 racetrack amplitudes.

    Per-site injected noise is sqrt(dt/shots); the junction vector averages
    2*QUBITS_PER_EDGE bond endpoints, and the DFT averages 4 junctions. A
    LoopK1 near this value is noise, not signal.
    """
    scale = math.sqrt(dt / max(effective_shots, 1e-12))
    return scale / math.sqrt(2.0 * QUBITS_PER_EDGE) / math.sqrt(4.0)


def swap_asymmetry(profiles: Dict[int, Dict[str, np.ndarray]],
                   junctions: List[Dict[str, Any]],
                   pairs: List[Tuple[int, int]]) -> Dict[str, float]:
    """Rev 311 metric, retained for CSV continuity. RMS over all components,
    which DILUTES a single badly-violated site -- see bond_consistency()."""
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


def bond_consistency(profiles: Dict[int, Dict[str, np.ndarray]],
                     junctions: List[Dict[str, Any]]) -> Dict[str, float]:
    """The honest swap_asymmetry: WORST disagreement between the two copies of
    a shared qubit, per junction.

    The two sides of a junction are supposed to BE the same qubits. If this
    does not fall during the anneal, the K4 topology was never realized and
    you simulated four loosely coupled manifolds. This is the metric to sweep
    g_junction against: Rev 311's 0.15 against target_J=1.0 is far too weak to
    hold the constraint, whose correct value is infinity.
    """
    out = {}
    for rec in junctions:
        worst = 0.0
        for (qa, qb) in rec["bonds"]:
            for c in ("X", "Y", "Z"):
                worst = max(worst, abs(float(profiles[rec["ma"]][c][qa])
                                       - float(profiles[rec["mb"]][c][qb])))
        out[rec["label"]] = worst
    return out


def bloch_to_rotation(b, delta, max_angle: float = MAX_KICK_ANGLE):
    """Rotation vector w with w x b = delta.

    A rotation by vector w acts as b -> b + w x b to first order, so the
    minimum-norm solution is w = (b x delta) / |b|^2. The component of delta
    along b is dropped: no unitary can change the Bloch radius. Linearization
    error is O(|w|^2); max_angle bounds it.
    """
    b = np.asarray(b, dtype=float)
    d = np.asarray(delta, dtype=float)
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(d))):
        return np.zeros(3)
    n2 = float(b @ b)
    if n2 < 1e-9:
        return np.zeros(3)
    w = np.cross(b, d) / n2
    m = float(np.linalg.norm(w))
    if m > max_angle:
        w = w * (max_angle / m)
    return w


# =====================================================================
# TENSOR-NETWORK BOUNDARY (chi > 1)
# =====================================================================
def ket_to_tensor(psi: np.ndarray) -> np.ndarray:
    """Manifold ket -> T[slot0, slot1, slot2].

    Verified: reshape(psi, (d,d,d)) yields axes [slot2, slot1, slot0] under
    little-endian amplitude indexing, so one transpose puts legs in slot order.
    """
    d = LEG_DIM
    psi = np.asarray(psi)
    if psi.size != d ** 3:
        raise ValueError(f"ket size {psi.size} != d**3 = {d ** 3}")
    return np.transpose(psi.reshape(d, d, d), (2, 1, 0))


def leg_isometries(T: np.ndarray, chi: int):
    """Top-chi Schmidt basis per leg. P[k] is (chi, d), rows conjugated so
    that applying P[k] to leg k truncates it. Also returns the discarded
    weight per leg."""
    d = T.shape[0]
    Ps, disc = [], []
    for k in range(3):
        M = np.ascontiguousarray(np.moveaxis(T, k, 0)).reshape(d, -1)
        rho = M @ M.conj().T
        rho = 0.5 * (rho + rho.conj().T)
        w, v = np.linalg.eigh(rho)
        idx = np.argsort(w)[::-1]
        w = np.clip(w[idx], 0.0, None)
        v = v[:, idx]
        tot = max(w.sum(), 1e-300)
        c = int(min(chi, d))
        Ps.append(np.ascontiguousarray(v[:, :c].conj().T))
        disc.append(float(1.0 - w[:c].sum() / tot))
        del M, rho, w, v
        gc.collect()
    return Ps, disc


def truncate_tensor(T: np.ndarray, Ps) -> np.ndarray:
    out = np.einsum('ia,jb,kc,abc->ijk', Ps[0], Ps[1], Ps[2], T, optimize=True)
    n = np.linalg.norm(out)
    return out / (n if n > 0 else 1.0)


def bond_permutation(rec) -> np.ndarray:
    """Index permutation of a junction block, side-B basis -> side-A basis.

    Bit i of the side-A block index is qubit qa_list[i]; bit j of side-B is
    qb_list[j]; the gluing identifies qa_list[i] with qb_list[pairing[i]].
    So a_i = b_{pairing[i]}.
    """
    d = LEG_DIM
    pairing = rec["pairing"]
    b = np.arange(d)
    a = np.zeros(d, dtype=np.int64)
    for i in range(QUBITS_PER_EDGE):
        a |= (((b >> pairing[i]) & 1) << i)
    return a


def _expect_site(rho: np.ndarray, i: int, P: np.ndarray) -> float:
    """<P_i> for qubit i of a block density matrix, little-endian in-block."""
    d = LEG_DIM
    lo = 2 ** i
    hi = d // (2 * lo)
    r = rho.reshape(hi, 2, lo, hi, 2, lo)
    red = np.einsum('aibajb->ij', r)
    return float(np.real(np.trace(P @ red)))


def _vn_bits(rho: np.ndarray) -> float:
    w = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    w = np.clip(np.real(w), 1e-300, None)
    w = w / w.sum()
    return float(-np.sum(w * np.log2(w)))


class K4BoundaryBP(object):
    """Loopy BP with Hermitian PSD chi x chi messages over the K4 bonds.

    CAVEAT (measured, not hypothesized): K4 has four triangles and BP is exact
    only on trees. Against a state that IS exactly a K4 tensor network, at
    chi = full leg dimension with truncation weight 1e-16, single-site beliefs
    still carried 0.17 MAE. chi buys correlations, not correctness.
    """

    def __init__(self, junctions, chi=BP_CHI, damping=BP_DAMPING,
                 max_iter=BP_MAX_ITER, tol=1e-9):
        self.junctions = junctions
        self.chi = int(chi)
        self.damping = float(damping)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.perm = {r["index"]: bond_permutation(r) for r in junctions}
        self.T, self.P, self.disc = {}, {}, {}
        self.msg = {}
        for m in range(N_MANIFOLDS):
            for k in range(LOCAL_EDGES):
                self.msg[(m, k)] = np.eye(self.chi, dtype=np.complex128) / self.chi
        self.iters = 0
        self.residual = float('nan')

    def set_manifold_tensor(self, m, T, Ps, disc=None):
        self.T[m] = np.asarray(T, dtype=np.complex128)
        self.P[m] = [np.asarray(p, dtype=np.complex128) for p in Ps]
        self.disc[m] = list(disc) if disc is not None else [0.0] * 3
        c = self.T[m].shape[0]
        for k in range(LOCAL_EDGES):
            if self.msg[(m, k)].shape[0] != c:
                self.msg[(m, k)] = np.eye(c, dtype=np.complex128) / c

    def transport(self, ji):
        """W : (chi_A, chi_B), mapping B's leg basis into A's leg basis."""
        rec = self.junctions[ji]
        PA = self.P[rec["ma"]][rec["slot_a"]]
        PB = self.P[rec["mb"]][rec["slot_b"]]
        a_of_b = self.perm[ji]
        PBp = np.zeros_like(PB)
        PBp[:, a_of_b] = PB
        return PA @ PBp.conj().T

    def _outgoing(self, m, k):
        T = self.T[m]
        others = [j for j in range(3) if j != k]
        r1 = self.msg[(m, others[0])]
        r2 = self.msg[(m, others[1])]
        Tk = np.moveaxis(T, k, 0)
        A = np.einsum('abc,bd,ce->ade', Tk, r1, r2, optimize=True)
        out = np.einsum('ade,fde->af', A, Tk.conj(), optimize=True)
        out = 0.5 * (out + out.conj().T)
        tr = np.real(np.trace(out))
        return out / (tr if tr > 1e-300 else 1.0)

    def run(self):
        if len(self.T) != N_MANIFOLDS:
            raise RuntimeError("BP needs a tensor from every manifold")
        lam = self.damping
        for it in range(self.max_iter):
            new = {}
            for rec in self.junctions:
                ji, ma, mb = rec["index"], rec["ma"], rec["mb"]
                ka, kb = rec["slot_a"], rec["slot_b"]
                W = self.transport(ji)
                oa = self._outgoing(ma, ka)
                ob = self._outgoing(mb, kb)
                to_b = W.conj().T @ oa @ W
                to_a = W @ ob @ W.conj().T
                for key, val in (((mb, kb), to_b), ((ma, ka), to_a)):
                    tr = np.real(np.trace(val))
                    new[key] = val / (tr if tr > 1e-300 else 1.0)
            res = 0.0
            for key, val in new.items():
                res = max(res, float(np.abs(val - self.msg[key]).max()))
                self.msg[key] = (1.0 - lam) * val + lam * self.msg[key]
            self.iters = it + 1
            self.residual = res
            if res < self.tol:
                break
        return self.residual

    @staticmethod
    def _psd_sqrt(rho):
        w, v = np.linalg.eigh(0.5 * (rho + rho.conj().T))
        w = np.clip(w, 0.0, None)
        return (v * np.sqrt(w)) @ v.conj().T

    def edge_belief_physical(self, ji):
        """Junction block reduced density matrix in side-A's physical basis.
        Simple-update / Vidal-gauge belief sqrt(rho_A) rho_B sqrt(rho_A):
        exact on a tree, approximate on K4."""
        rec = self.junctions[ji]
        ma, mb = rec["ma"], rec["mb"]
        ka, kb = rec["slot_a"], rec["slot_b"]
        W = self.transport(ji)
        ra = self._outgoing(ma, ka)
        rb = W @ self._outgoing(mb, kb) @ W.conj().T
        sa = self._psd_sqrt(ra)
        PA = self.P[ma][ka]
        rho = PA.conj().T @ (sa @ rb @ sa) @ PA
        rho = 0.5 * (rho + rho.conj().T)
        tr = np.real(np.trace(rho))
        return rho / (tr if tr > 1e-300 else 1.0)

    def site_expectations(self, ji):
        rho = self.edge_belief_physical(ji)
        out = {}
        for name, P in _PAULI_MAT.items():
            out[name] = np.array([_expect_site(rho, i, P)
                                  for i in range(QUBITS_PER_EDGE)])
        return out

    def pair_mutual_information(self, m, k1, k2):
        """MI between two junction blocks of manifold m, in bits, in the
        truncated basis. Identically zero at chi = 1 -- that is the whole
        point of chi > 1."""
        T = self.T[m]
        k3 = [j for j in range(3) if j not in (k1, k2)][0]
        r3 = self.msg[(m, k3)]
        Tp = np.transpose(T, (k1, k2, k3))
        A = np.einsum('abc,cd->abd', Tp, r3, optimize=True)
        rho4 = np.einsum('abd,efd->abef', A, Tp.conj(), optimize=True)
        c1, c2 = Tp.shape[0], Tp.shape[1]
        rho = rho4.reshape(c1 * c2, c1 * c2)
        rho = 0.5 * (rho + rho.conj().T)
        rho = rho / max(np.real(np.trace(rho)), 1e-300)
        r1 = np.einsum('abeb->ae', rho.reshape(c1, c2, c1, c2))
        r2 = np.einsum('abaf->bf', rho.reshape(c1, c2, c1, c2))
        return _vn_bits(r1) + _vn_bits(r2) - _vn_bits(rho)

    def diagnostics(self):
        allw = [w for m in self.disc for w in self.disc[m]]
        return {"chi": self.chi, "bp_iters": self.iters,
                "bp_residual": self.residual,
                "trunc_wt_max": max(allw) if allw else 0.0,
                "block_mi_m0": self.pair_mutual_information(0, 0, 1)}


# =====================================================================
# LEARNED EDGE MESSAGE (numpy-only inference)
# =====================================================================
N_FEAT = 16
N_OUT = 3


class EdgeMLP(object):
    """Inference half of k4_gnn_message.MLP. Training lives with the exact
    small-scale reference; this only loads and evaluates.

    Feature layout (16): own site Bloch (3), partner site Bloch (3), own block
    mean (3), partner block mean (3), s, J, hx, hz. Per-BOND and shared across
    bond index, which is why a model trained at QUBITS_PER_EDGE = 4 transfers
    to 9. Measured: train qpe=2 -> test qpe=3 gives residual R^2 = 0.78.

    It does NOT extrapolate in anneal time: trained on steps 0-19 and tested
    on 20-29 gives R^2 = -4.3. Train across the FULL schedule at small size.
    """

    def __init__(self, path):
        z = np.load(path)
        self.sizes = [int(x) for x in z["sizes"]]
        self.mu = z["mu"]
        self.sd = z["sd"]
        self.yscale = float(z["yscale"][0])
        n = len(self.sizes) - 1
        self.W = [z[f"W{i}"] for i in range(n)]
        self.b = [z[f"b{i}"] for i in range(n)]
        if self.sizes[0] != N_FEAT or self.sizes[-1] != N_OUT:
            raise ValueError(f"model has shape {self.sizes}, expected "
                             f"{N_FEAT} inputs and {N_OUT} outputs")

    def predict(self, X):
        h = (np.atleast_2d(np.asarray(X, dtype=float)) - self.mu) / self.sd
        for i in range(len(self.W)):
            z = h @ self.W[i] + self.b[i]
            h = np.tanh(z) if i < len(self.W) - 1 else z
        return h * self.yscale


def build_edge_features(junctions, profiles, s, J, hx, hz):
    """Per-bond feature matrix plus the (manifold, local qubit) index so
    predictions scatter back into kick dicts."""
    F, index = [], []
    for rec in junctions:
        ma, mb = rec["ma"], rec["mb"]
        for side, mm, mo in (("a", ma, mb), ("b", mb, ma)):
            qs = rec["qubits_a"] if side == "a" else rec["qubits_b"]
            oq = rec["qubits_b"] if side == "a" else rec["qubits_a"]
            pm, po = profiles[mm], profiles[mo]
            blk_own = np.array([[pm["X"][q], pm["Y"][q], pm["Z"][q]]
                                for q in qs])
            blk_oth = np.array([[po["X"][q], po["Y"][q], po["Z"][q]]
                                for q in oq])
            mo_own = blk_own.mean(axis=0)
            mo_oth = blk_oth.mean(axis=0)
            for i, q in enumerate(qs):
                F.append(np.concatenate([blk_own[i], blk_oth[i],
                                         mo_own, mo_oth, [s, J, hx, hz]]))
                index.append((mm, q))
    return np.array(F), index


# =====================================================================
# BOUNDARY PLUGINS
# =====================================================================
class MeanFieldBoundary(object):
    """Rev 311, unchanged in substance. chi = 1."""

    name = "meanfield"
    needs_tensor = False

    def __init__(self, junctions, g_junction_target):
        self.junctions = junctions
        self.g_target = float(g_junction_target)

    def kicks(self, profiles, noise, s, J, hx, hz, dt, tensors=None):
        g = s * self.g_target
        fields, energy, jvecs = compute_kicks_and_junction_energy(
            profiles, noise, self.junctions, g)
        w = {m: {} for m in range(N_MANIFOLDS)}
        coef = -2.0 * dt
        for m in fields:
            for q, k in fields[m].items():
                w[m][q] = (coef * k[0], coef * k[1], coef * k[2])
        return w, energy, jvecs, {"mode": "meanfield", "g": g}


class GNNBoundary(object):
    """Learned per-bond residual, converted to a rotation."""

    name = "gnn"
    needs_tensor = False

    def __init__(self, junctions, model_path=GNN_MODEL_PATH, scale=GNN_SCALE):
        self.junctions = junctions
        self.net = EdgeMLP(model_path)
        self.scale = float(scale)

    def kicks(self, profiles, noise, s, J, hx, hz, dt, tensors=None):
        F, index = build_edge_features(self.junctions, profiles, s, J, hx, hz)
        R = self.net.predict(F) * self.scale
        w = {m: {} for m in range(N_MANIFOLDS)}
        for (m, q), r in zip(index, R):
            b = np.array([profiles[m]["X"][q], profiles[m]["Y"][q],
                          profiles[m]["Z"][q]])
            w[m][q] = tuple(bloch_to_rotation(b, r))
        jvecs = {}
        for rec in self.junctions:
            acc = np.zeros(3)
            n = 0
            for (qa, qb) in rec["bonds"]:
                for (mm, qq) in ((rec["ma"], qa), (rec["mb"], qb)):
                    p = profiles[mm]
                    acc += np.array([p["X"][qq], p["Y"][qq], p["Z"][qq]])
                    n += 1
            jvecs[rec["index"]] = acc / max(1, n)
        return w, 0.0, jvecs, {"mode": "gnn",
                               "msg_rms": float(np.sqrt(np.mean(R ** 2)))}


class BPBoundary(object):
    """chi > 1 loopy BP. Consumes truncated tensors computed in the workers."""

    name = "bp"
    needs_tensor = True

    def __init__(self, junctions, chi=BP_CHI, damping=BP_DAMPING,
                 max_iter=BP_MAX_ITER):
        self.junctions = junctions
        self.bp = K4BoundaryBP(junctions, chi=chi, damping=damping,
                               max_iter=max_iter)
        self.last = None

    def kicks(self, profiles, noise, s, J, hx, hz, dt, tensors=None):
        if tensors is None or len(tensors) != N_MANIFOLDS:
            if self.last is None:
                w = {m: {} for m in range(N_MANIFOLDS)}
                jv = {r["index"]: np.zeros(3) for r in self.junctions}
                return w, 0.0, jv, {"mode": "bp", "state": "warming-up"}
            targets, info = self.last
            info = dict(info, state="stale")
        else:
            for m in range(N_MANIFOLDS):
                T, Ps, disc = tensors[m]
                self.bp.set_manifold_tensor(m, T, Ps, disc)
            self.bp.run()
            targets = {r["index"]: self.bp.site_expectations(r["index"])
                       for r in self.junctions}
            info = dict(self.bp.diagnostics(), mode="bp", state="fresh")
            self.last = (targets, info)

        w = {m: {} for m in range(N_MANIFOLDS)}
        jvecs = {}
        for rec in self.junctions:
            se = targets[rec["index"]]
            acc = np.zeros(3)
            for i, (qa, qb) in enumerate(rec["bonds"]):
                tgt = np.array([se["X"][i], se["Y"][i], se["Z"][i]])
                for (mm, qq) in ((rec["ma"], qa), (rec["mb"], qb)):
                    p = profiles[mm]
                    b = np.array([p["X"][qq], p["Y"][qq], p["Z"][qq]])
                    w[mm][qq] = tuple(bloch_to_rotation(b, tgt - b))
                acc += 2.0 * tgt
            jvecs[rec["index"]] = acc / max(1, 2 * QUBITS_PER_EDGE)
        return w, 0.0, jvecs, info


def make_boundary(mode, junctions, target_g_junction):
    if mode == "meanfield":
        return MeanFieldBoundary(junctions, target_g_junction)
    if mode == "gnn":
        return GNNBoundary(junctions)
    if mode == "bp":
        return BPBoundary(junctions)
    raise ValueError(f"unknown BOUNDARY_MODE {mode!r}")


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
    tensor_every: int,
    bp_chi: int
):
    os.environ["PYQRACK_SHARED_LIB_PATH"]         = "/usr/local/lib/qrack/libqrack_pinvoke.so"
    os.environ["OCL_ICD_PLATFORM_SORT"]           = "none"

    physical_gpu_index = rank // workers_per_gpu
    os.environ["QRACK_OCL_DEFAULT_DEVICE"]        = str(physical_gpu_index)
    os.environ["QRACK_QPAGER_DEVICES"]            = str(physical_gpu_index)
    os.environ["QRACK_QUNITMULTI_DEVICES"]        = str(physical_gpu_index)

    # 27 qubits fp32 = 1 GB of amplitudes; cap at 2 GB per manifold for QUnit
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

        def apply_kick_angles(sim, angles):
            """Rev 312: the master now sends ROTATION ANGLES directly, so the
            worker no longer multiplies by -2*dt. meanfield mode sends
            -2*dt*g*<sigma>, reproducing Rev 311 exactly."""
            if not angles: return
            for raw_q, (wx, wy, wz) in angles.items():
                q = int(raw_q)
                if not math.isfinite(wx): wx = 0.0   # NaN guard
                if not math.isfinite(wy): wy = 0.0
                if not math.isfinite(wz): wz = 0.0
                if abs(wx) > 1e-12: apply_rx(sim, wx, q)
                if abs(wy) > 1e-12: apply_ry(sim, wy, q)
                if abs(wz) > 1e-12: apply_rz(sim, wz, q)

        def export_tensor(sim, chi):
            """Schmidt-truncate this manifold NEXT TO THE GPU.

            out_ket is 1 GB at 27 qubits; shipping it through mp.Pipe every BP
            step would be 4 GB of IPC per step. Truncating here sends
            chi**3 + 3*chi*512 complex numbers instead (~100 KB at chi=8), and
            the four SVDs run in parallel across the four workers.

            Host RAM cost inside the worker: the ket (1 GB complex64 ->
            promoted to 2 GB complex128 for eigh) plus one contiguous copy per
            leg. Budget ~4 GB per worker while this runs.
            """
            psi = np.asarray(sim.out_ket(), dtype=np.complex64)
            T = ket_to_tensor(psi).astype(np.complex128)
            del psi
            gc.collect()
            Ps, disc = leg_isometries(T, chi)
            Tt = truncate_tensor(T, Ps)
            del T
            gc.collect()
            return (Tt.astype(np.complex64),
                    [p.astype(np.complex64) for p in Ps], disc)

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
                    f"Fatal: VRAM/PCIe paging allocation failed on manifold {m} "
                    f"(rank {rank}, gpu {physical_gpu_index}). Driver error: {e}")

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
            want_tensor = (tensor_every > 0 and is_measure
                           and (t % tensor_every == 0))

            manifold_data_to_master = {}

            for m in assigned_manifolds:
                sim = sims[m]
                if kick_payloads[m]:
                    apply_kick_angles(sim, kick_payloads[m])

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

                    if not (np.all(np.isfinite(Z_arr))
                            and np.all(np.isfinite(X_arr))
                            and np.all(np.isfinite(Y_arr))):
                        nan_streak[m] += 1
                        print(f"[Worker {rank}] NaN tomo on M{m} at t={t}; "
                              f"reinit to |+>^{QUBITS_PER_MANIFOLD}",
                              file=sys.stderr)

                        try:
                            del sims[m]
                        except KeyError:
                            pass
                        gc.collect()

                        if nan_streak[m] > 1:
                            sleep_s = NAN_COOLDOWN_BASE_S * nan_streak[m]
                            print(f"[Worker {rank}] NaN streak {nan_streak[m]} "
                                  f"on M{m}; cooldown {sleep_s:.1f}s",
                                  file=sys.stderr)
                            time.sleep(sleep_s)

                        sim = QrackSimulator(
                            qubit_count=QUBITS_PER_MANIFOLD,
                            is_binary_decision_tree=False,
                            is_stabilizer_hybrid=False, is_gpu=True)
                        for q in range(QUBITS_PER_MANIFOLD):
                            apply_h(sim, q)
                        sims[m] = sim

                        _sanity = float(sim.pauli_expectation([0], [PX]))
                        if not math.isfinite(_sanity) or abs(_sanity - SIGN_X) > 0.5:
                            print(f"[Worker {rank}] GPU still broken after reinit "
                                  f"M{m} t={t}; sleeping 5s and returning NaN",
                                  file=sys.stderr)
                            time.sleep(5.0)

                            try:
                                del sims[m]
                            except KeyError:
                                pass
                            gc.collect()

                            sim = QrackSimulator(
                                qubit_count=QUBITS_PER_MANIFOLD,
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
                                  f"sim.get_unitary_fidelity() not found. "
                                  f"Upgrade PyQrack for true fidelity tracking.",
                                  file=sys.stderr)
                            _warned_fidelity = True

                    rec_out = {
                        "state":             state,
                        "exact_bulk_energy": bulk_e,
                        "lat_trotter_ms":    t_lat_trotter * 1000.0,
                        "lat_tomo_ms":       t_lat_tomo    * 1000.0,
                        "unitary_fidelity":  fidelity,
                    }

                    if want_tensor and np.all(np.isfinite(Z_arr)):
                        t_start_svd = time.perf_counter()
                        try:
                            rec_out["tensor"] = export_tensor(sim, bp_chi)
                            rec_out["lat_svd_ms"] = (
                                time.perf_counter() - t_start_svd) * 1000.0
                        except Exception as e:
                            print(f"[Worker {rank}] tensor export failed on "
                                  f"M{m} t={t}: {e}", file=sys.stderr)
                            gc.collect()

                    manifold_data_to_master[m] = rec_out

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
    def __init__(self, master_seed: int = 1337, twist: bool = TWIST_GLUING,
                 boundary_mode: str = BOUNDARY_MODE,
                 target_g_junction: float = 0.15):
        self.master_seed = master_seed
        self.twist       = twist
        self.intra_edges, self.edge_groups = generate_manifold_topology()
        self.junctions        = build_junction_table(twist)
        self.junction_by_pair = {(r["ma"], r["mb"]): r for r in self.junctions}
        self.loop_indices     = [self.junction_by_pair[jp]["index"]
                                 for jp in LOOP_JUNCTIONS]
        self.chord_indices    = [self.junction_by_pair[jp]["index"]
                                 for jp in CHORD_JUNCTIONS]
        self.global_map       = global_qubit_map(self.junctions)

        self.boundary_mode = boundary_mode
        self.boundary      = make_boundary(boundary_mode, self.junctions,
                                           target_g_junction)

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
                    "revision":            312,
                    "boundary_mode":       self.boundary_mode,
                    "bp_chi":              BP_CHI,
                    "bp_every":            BP_EVERY,
                    "n_manifolds":         N_MANIFOLDS,
                    "qubits_per_manifold": QUBITS_PER_MANIFOLD,
                    "corner_per_edge":     C_CORNER,
                    "midedge_per_edge":    E_MIDEDGE,
                    "qubits_per_junction": QUBITS_PER_EDGE,
                    "leg_dim":             LEG_DIM,
                    "n_junctions":         N_JUNCTIONS,
                    "unique_qubits":       UNIQUE_QUBITS,
                    "intra_edges":         len(self.intra_edges),
                    "rim_edges_expected":  _EXPECTED_RIM_EDGES,
                    "twist_gluing":        self.twist,
                    "loop_junctions":      [f"J{a}{b}" for a, b in LOOP_JUNCTIONS],
                    "chord_junctions":     [f"J{a}{b}" for a, b in CHORD_JUNCTIONS],
                    "moebius_pairs":       [f"M{a}M{b}" for a, b in MOEBIUS_PAIRS],
                    "gpus":                GPUS_AVAILABLE,
                    "manifold_vram_mb":    MANIFOLD_VRAM_MB,
                }, f)
            with open(self.energy_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Anneal_Percent", "Exact_Bulk_Energy",
                    "Junction_Energy", "Total_Energy",
                    "Min_Unitary_Fidelity", "Swap_Asym_M0M1", "Swap_Asym_M2M3",
                    "Bond_Consistency_Max", "Loop_K1", "Loop_Noise_Floor",
                    "Boundary_Info"
                ]).writeheader()
            with open(self.profiles_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Junction", "Kind", "X_mean", "Y_mean", "Z_mean",
                    "Bond_Consistency"
                ]).writeheader()
            with open(self.loop_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Loop_Momentum", "Amp_X", "Amp_Y", "Amp_Z"
                ]).writeheader()
        except Exception as e:
            print(f"[CSV] Warning: setup write failed: {e}", file=sys.stderr)

    def _log_csvs(self, step, anneal, bulk, junction_e, total, min_fidelity,
                  junction_vectors, loop_amps, asym, cons, floor, info):
        try:
            with open(self.energy_csv, mode='a', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Anneal_Percent", "Exact_Bulk_Energy",
                    "Junction_Energy", "Total_Energy",
                    "Min_Unitary_Fidelity", "Swap_Asym_M0M1", "Swap_Asym_M2M3",
                    "Bond_Consistency_Max", "Loop_K1", "Loop_Noise_Floor",
                    "Boundary_Info"
                ]).writerow({
                    "Step": step, "Anneal_Percent": anneal,
                    "Exact_Bulk_Energy": bulk,
                    "Junction_Energy":   junction_e,
                    "Total_Energy":      total,
                    "Min_Unitary_Fidelity": min_fidelity,
                    "Swap_Asym_M0M1": asym.get("M0M1", float('nan')),
                    "Swap_Asym_M2M3": asym.get("M2M3", float('nan')),
                    "Bond_Consistency_Max": max(cons.values()),
                    "Loop_K1": float(np.linalg.norm(loop_amps[1])),
                    "Loop_Noise_Floor": floor,
                    "Boundary_Info": json.dumps(info, default=float)})

            with open(self.profiles_csv, mode='a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=[
                    "Step", "Junction", "Kind", "X_mean", "Y_mean", "Z_mean",
                    "Bond_Consistency"])
                for rec in self.junctions:
                    v    = junction_vectors[rec["index"]]
                    kind = "loop" if rec["index"] in self.loop_indices else "chord"
                    w.writerow({"Step": step, "Junction": rec["label"],
                                "Kind": kind,
                                "X_mean": float(v[0]),
                                "Y_mean": float(v[1]),
                                "Z_mean": float(v[2]),
                                "Bond_Consistency": cons[rec["label"]]})

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
            probe_junction: int = 0, bp_every: int = BP_EVERY):

        if total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        if measure_every < 1:
            raise ValueError("measure_every must be a positive integer")
        if not (0 <= probe_junction < N_JUNCTIONS):
            raise ValueError("probe_junction out of range")
        if probe_step >= 0 and probe_step >= total_steps:
            raise ValueError(
                f"probe_step {probe_step} >= total_steps {total_steps}")

        tensor_every = bp_every if self.boundary.needs_tensor else 0
        floor = loop_noise_floor(dt, effective_shots)

        print(f"[Engine] K4 Rev312 27q | boundary={self.boundary.name} | "
              f"{N_MANIFOLDS} manifolds x {QUBITS_PER_MANIFOLD} qubits (all boundary) | "
              f"{N_JUNCTIONS} junctions x {QUBITS_PER_EDGE} qubits | "
              f"{UNIQUE_QUBITS} unique qubits | "
              f"rim_edges={_EXPECTED_RIM_EDGES} | leg_dim={LEG_DIM} | "
              f"gluing={'MOEBIUS' if self.twist else 'TRIVIAL'} | "
              f"loop={[f'J{a}{b}' for a, b in LOOP_JUNCTIONS]} | "
              f"{GPUS_AVAILABLE} GPUs | {total_steps} steps")
        if self.boundary.needs_tensor:
            print(f"[Engine] bp: chi={BP_CHI}, truncation in workers every "
                  f"{bp_every} measured steps (~100 KB payload, not 1 GB). "
                  f"BP is NOT exact on K4 -- four triangles. Score against the "
                  f"exact small-scale reference before believing it.")
        print(f"[Engine] racetrack noise floor (k!=0) ~ {floor:.3e} -- treat "
              f"LoopK1 below 3x this as nothing")

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
                      tensor_every, BP_CHI)
            )
            p.start()
            child_conn.close()
            workers.append(p)
            pipes.append(parent_conn)

        try:
            for t in range(total_steps):
                s = t / max(1, (total_steps - 1))
                current_g_junction = s * target_g_junction
                current_hx = (1.0 - s) * initial_hx + s * target_hx
                current_J  = s * target_J
                current_hz = s * target_hz
                is_measure = ((t % measure_every == 0)
                              or (t == total_steps - 1)
                              or (t == probe_step))

                if not is_measure:
                    continue

                t0 = time.perf_counter()

                # --- GATHER ---
                manifold_states = {}
                manifold_tensors = {}
                bulk_energy     = 0.0
                max_lat_trotter = 0.0
                max_lat_tomo    = 0.0
                max_lat_svd     = 0.0
                min_fidelity    = 1.0

                pending  = list(zip(pipes, active_ranks))
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

                    ready = mpc.wait([c for c, _ in pending],
                                     timeout=min(remaining, 5.0))

                    for conn in ready:
                        w_rank = next(r for c, r in pending if c is conn)
                        try:
                            data = conn.recv()
                            for m, payload in data.items():
                                state = payload["state"]
                                if any(np.any(~np.isfinite(state[c]))
                                       for c in ("X", "Y", "Z")):
                                    print(f"[Master] NaN M{m} step {t}; "
                                          f"substituting last-good",
                                          file=sys.stderr)
                                    state = self._get_last_good(m)

                                manifold_states[m] = state
                                if "tensor" in payload:
                                    manifold_tensors[m] = payload["tensor"]
                                    max_lat_svd = max(
                                        max_lat_svd,
                                        payload.get("lat_svd_ms", 0.0))
                                bulk_energy += (
                                    payload["exact_bulk_energy"]
                                    if math.isfinite(payload["exact_bulk_energy"])
                                    else 0.0)
                                max_lat_trotter = max(max_lat_trotter,
                                                      payload["lat_trotter_ms"])
                                max_lat_tomo    = max(max_lat_tomo,
                                                      payload["lat_tomo_ms"])
                                min_fidelity    = min(min_fidelity,
                                                      payload.get("unitary_fidelity", 1.0))

                        except EOFError:
                            print(f"[Master] EOF rank {w_rank}; substituting last-good",
                                  file=sys.stderr)
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

                # --- STOCHASTIC NOISE (Rev 88 scaling) ---
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
                        vx = np.where(np.isfinite(px),
                                      np.clip(1.0 - px ** 2, 0.0, 1.0), 1.0)
                        vy = np.where(np.isfinite(py),
                                      np.clip(1.0 - py ** 2, 0.0, 1.0), 1.0)
                        vz = np.where(np.isfinite(pz),
                                      np.clip(1.0 - pz ** 2, 0.0, 1.0), 1.0)
                        n    = len(qlist)
                        xn   = rng_j.normal(0.0, 1.0, n) * np.sqrt(vx) * scale
                        yn   = rng_j.normal(0.0, 1.0, n) * np.sqrt(vy) * scale
                        zn   = rng_j.normal(0.0, 1.0, n) * np.sqrt(vz) * scale
                        for i, q in enumerate(qlist):
                            noise[mm][q] = (xn[i], yn[i], zn[i])

                # --- BOUNDARY: kicks are ROTATION ANGLES ---
                tensors = (manifold_tensors
                           if len(manifold_tensors) == N_MANIFOLDS else None)
                kicks, junction_energy, junction_vectors, binfo = \
                    self.boundary.kicks(manifold_states, noise, s, current_J,
                                        current_hx, current_hz, dt,
                                        tensors=tensors)

                # --- OPTIONAL LOCALIZED PROBE (rings the bell) ---
                if probe_step >= 0 and t == probe_step and probe_amplitude != 0.0:
                    rec    = self.junctions[probe_junction]
                    qa, qb = rec["bonds"][0]
                    for (mm, qq) in ((rec["ma"], qa), (rec["mb"], qb)):
                        k = kicks[mm].get(qq, (0.0, 0.0, 0.0))
                        kicks[mm][qq] = (k[0] + probe_amplitude, k[1], k[2])
                    print(f"[Probe] step {t}: injected {probe_amplitude:+.4f} rad "
                          f"at {rec['label']} bond 0")

                # --- RACETRACK DIAGNOSTICS ---
                loop_amps = loop_momentum_amplitudes(junction_vectors,
                                                     self.loop_indices)
                asym      = swap_asymmetry(manifold_states, self.junctions,
                                           MOEBIUS_PAIRS)
                cons      = bond_consistency(manifold_states, self.junctions)

                total_energy = bulk_energy + junction_energy
                k1   = float(np.linalg.norm(loop_amps[1]))
                flag = "" if k1 > 3.0 * floor else " [<3x floor]"
                asym_str = " ".join(
                    f"{k}:{v:.4f}" for k, v in sorted(asym.items()))
                svd_str = f" SVD:{max_lat_svd:6.0f}ms" if max_lat_svd > 0 else ""
                print(f"Step {t:03d} | E: {total_energy:+.4f} "
                      f"(bulk {bulk_energy:+.3f} / junc {junction_energy:+.3f}) | "
                      f"Loop k1: {k1:.4e}{flag} | "
                      f"BondCons: {max(cons.values()):.4f} | "
                      f"Asym {asym_str} | "
                      f"Lat(Trot/Tomo): {max_lat_trotter:5.1f}/{max_lat_tomo:5.1f}ms"
                      f"{svd_str} | "
                      f"Fid: {min_fidelity:.5f} | {time.perf_counter() - t0:.2f}s")
                self._log_csvs(t, s * 100, bulk_energy, junction_energy,
                               total_energy, min_fidelity, junction_vectors,
                               loop_amps, asym, cons, floor, binfo)

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
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--boundary", default=BOUNDARY_MODE,
                     choices=["meanfield", "gnn", "bp"])
    _ap.add_argument("--model", default=GNN_MODEL_PATH)
    _ap.add_argument("--chi", type=int, default=BP_CHI)
    _ap.add_argument("--bp-every", type=int, default=BP_EVERY)
    _ap.add_argument("--steps", type=int, default=100)
    _ap.add_argument("--dt", type=float, default=0.04)
    _ap.add_argument("--g", type=float, default=0.15,
                     help="target g_junction (meanfield only; it is a "
                          "constraint penalty, correct value is infinity)")
    _args = _ap.parse_args()

    BOUNDARY_MODE  = _args.boundary
    GNN_MODEL_PATH = _args.model
    BP_CHI         = _args.chi
    BP_EVERY       = _args.bp_every

    mp.set_start_method('spawn', force=True)

    engine = K4ManifoldEngine(master_seed=1337, twist=TWIST_GLUING,
                              boundary_mode=BOUNDARY_MODE,
                              target_g_junction=_args.g)
    try:
        engine.run(
            total_steps=_args.steps,
            dt=_args.dt,
            initial_hx=3.0,
            target_g_junction=_args.g,
            target_J=1.0,
            target_hx=0.5,
            target_hz=0.2,
            measure_every=1,
            effective_shots=512.0,
            probe_step=-1,
            probe_amplitude=0.5,
            probe_junction=0,
            bp_every=_args.bp_every
        )
    except KeyboardInterrupt:
        pass
