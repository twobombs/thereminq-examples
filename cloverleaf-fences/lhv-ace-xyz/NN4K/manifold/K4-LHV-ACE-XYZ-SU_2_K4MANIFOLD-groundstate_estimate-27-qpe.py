# -*- coding: us-ascii -*-
# K4 Distributed Composite Quantum Phase Estimation Benchmark
# (4 Tri-Hole Manifolds, 6 Junctions, shared Hilbert space)
#
# SINGLE FILE. No external modules, so spawn workers need nothing on PYTHONPATH.
# Merged from k4_topology, k4_backend, k4_regions, k4_qpe, k4_exact_reference
# and the benchmark driver.
#
# ===================================================================
# THE REDUCTION
# ===================================================================
# For the single-ancilla information-theory QPE of arXiv:2505.09133,
#
#     P(m | k, beta) = ( 1 + (-1)^m Re[ e^{i beta} <Phi|U^k|Phi> ] ) / 2
#
# so the ENTIRE workload -- every k, every beta, the estimator, the energy -- is
# a function of one complex number per k: the Loschmidt amplitude
# a_k = <Phi|U^k|Phi>. There is no ancilla to simulate and no controlled-U to
# distribute.
#
# This matters because a shared ancilla under controlled-U entangles with all
# four manifolds at once, and four independent pure manifold states cannot hold
# that. Distributing the QPE CIRCUIT is not an approximation with a measurable
# error; it is structurally impossible. Distributing the AMPLITUDE is both
# possible and cheap.
#
# ===================================================================
# THE COMPOSITE ESTIMATOR
# ===================================================================
# a_k is a GLOBAL observable, so the kick-based boundaries (meanfield, gnn, bp)
# cannot produce it: they steer local Bloch vectors and the composite has no
# global state to overlap with. The boundary moves out of the evolution and into
# the estimator, along K4's own region decomposition:
#
#     a_k ~ prod_m a_k^(m) / prod_j a_k^(j)
#
# Each region evolves ALONE. No kicks, no messages, no ket transport. The only
# communication is one complex scalar per region per k: 10 numbers.
#
# ===================================================================
# MEASURED (both against an exact merged reference)
# ===================================================================
#   12 qubits (C=1,E=1):  exact -26.094 | product -52.086 (-99.6%)
#                                       | bethe   -26.929 ( -3.2%)
#   18 qubits (C=1,E=2):  exact -41.038 | product +47.235 (+215%, wrong sign)
#                                       | bethe   -41.226 ( -0.46%)
#
# The Bethe error SHRINKS with size. Junction blocks are a smaller fraction of
# the system as qpe grows, so loop error matters relatively less. This is the
# opposite of how the kick-based boundaries behaved.
#
# ===================================================================
# TWO BUGS THIS FILE FIXES
# ===================================================================
# 1. The global Hamiltonian was multiplicity-weighted: an edge contributed by
#    two manifolds got weight 2, to match Rev 311's sum-over-manifolds bulk
#    energy. That is double counting. Each edge of the shared system is ONE
#    physical bond, weight 1. At (3,6) with twist there are 78 distinct edges,
#    not 120 manifold-edge instances. Pass legacy_multiplicity=True to
#    reproduce the old numbers for diffing against past runs.
#
# 2. The obvious region assignment -- junction regions carry every edge internal
#    to their block -- silently DELETES edges. At (3,6) twelve edges come out
#    with counting weight zero. Junction regions must carry only the
#    multiplicity-2 edges plus the single-site terms of their qubits.
#
#    With that fixed:  H = sum_m H_m - sum_j H_j, exactly.
#    4 manifold regions (+1), 6 junction regions (-1). Bethe/Kikuchi on a
#    bipartite region graph. Verified weight-1 on every edge and every site at
#    (1,1) (1,2) (2,2) (2,3) (3,6).
#
# ===================================================================
# WHAT IS NOT GUARANTEED
# ===================================================================
# The region decomposition is EXACT for the Hamiltonian. It is an
# APPROXIMATION for the amplitude: K4 has four triangles and Bethe is exact only
# on trees. Score it against the exact reference at (1,1)/(1,2)/(2,2). At (3,6)
# there is no exact reference and product-vs-bethe divergence is the only
# self-consistency signal available.
#
# ALIASING: E = -phase/dt wraps at |E|*dt = pi. The slope readout survives by
# unwrapping; estimate_energy() resolves the branch and will silently return the
# wrong energy past the limit. Use dt <= 0.06 at 12 qubits, <= 0.04 at 18, and
# scale down further at 54. check_aliasing() warns.
#
# GPU AMPLITUDE EXTRACTION: for |Phi> = |+>^n, a_k is obtained locally in the
# worker as vdot(psi_0, psi_k) via out_ket -- ~1 GB of HOST RAM per evaluation
# at 27 qubits but ZERO pipe traffic, since the worker returns the scalar.
# Evaluate only at the k values needed. The Hadamard-test alternative needs a
# 28th qubit and doubles VRAM; not worth it.

import os
import sys
import csv
import json
import math
import argparse
import numpy as np
from typing import List, Tuple, Dict, Any


# =====================================================================
# TOPOLOGY
# =====================================================================
# K4 tri-hole manifold topology, parameterized on (C_CORNER, E_MIDEDGE).
#
# Extracted from Rev 311 and made size-agnostic so that the exact small-scale
# reference and the large deployed engine share one source of truth.
#
# Key structural facts (asserted, not assumed):
#   * QUBITS_PER_EDGE = C_CORNER + E_MIDEDGE
#   * A manifold has exactly LOCAL_EDGES = 3 slots, each of QUBITS_PER_EDGE
#     qubits, and they partition the manifold. There is no interior.
#   * Every qubit lies in exactly 2 of the 3 portals (all-boundary property).
#   * K4 has 6 junctions; each manifold spends all 3 slots on junctions.
#   * Therefore each manifold state is EXACTLY a rank-3 tensor with leg
#     dimension 2**QUBITS_PER_EDGE. This is not an ansatz.
#   * UNIQUE_QUBITS = N_JUNCTIONS * QUBITS_PER_EDGE.

LOCAL_EDGES = 3
N_MANIFOLDS = 4
NEIGHBOURS = [[j for j in range(N_MANIFOLDS) if j != i]
              for i in range(N_MANIFOLDS)]
JUNCTION_PAIRS = [(i, j) for i in range(N_MANIFOLDS)
                  for j in range(i + 1, N_MANIFOLDS)]
N_JUNCTIONS = len(JUNCTION_PAIRS)

LOOP_CYCLE = [0, 1, 2, 3]
LOOP_JUNCTIONS = [tuple(sorted((LOOP_CYCLE[i], LOOP_CYCLE[(i + 1) % 4])))
                  for i in range(4)]
CHORD_JUNCTIONS = [jp for jp in JUNCTION_PAIRS if jp not in LOOP_JUNCTIONS]
MOEBIUS_PAIRS = [(0, 1), (2, 3)]


class K4Topology(object):
    """All topology for one choice of (C_CORNER, E_MIDEDGE, twist)."""

    def __init__(self, c_corner=3, e_midedge=6, twist=True):
        if c_corner < 1 or e_midedge < 1:
            raise ValueError("c_corner and e_midedge must both be >= 1")
        self.c_corner = int(c_corner)
        self.e_midedge = int(e_midedge)
        self.twist = bool(twist)

        self.qpe = self.c_corner + self.e_midedge
        self.qpm = LOCAL_EDGES * self.qpe
        self.n_rim = 2 * self.qpe
        self.unique_qubits = N_JUNCTIONS * self.qpe
        self.expected_rim_edges = 3 * (self.qpe + 1)

        self.intra_edges, self.edge_groups = self._generate_manifold_topology()
        self.junctions = self._build_junction_table()
        self.junction_by_pair = {(r["ma"], r["mb"]): r for r in self.junctions}
        self.loop_indices = [self.junction_by_pair[jp]["index"]
                             for jp in LOOP_JUNCTIONS]
        self.chord_indices = [self.junction_by_pair[jp]["index"]
                              for jp in CHORD_JUNCTIONS]
        self.global_map = self._build_global_map()
        self.global_edges = self._build_global_edges()

    # -- intra-manifold ------------------------------------------------
    def edge_local_qubits(self, k):
        base = k * self.qpe
        return list(range(base, base + self.qpe))

    def portal_ring(self, p):
        return (self.edge_local_qubits((p - 1) % LOCAL_EDGES)
                + self.edge_local_qubits(p))

    def _generate_manifold_topology(self):
        edges = set()
        for p in range(LOCAL_EDGES):
            ring = self.portal_ring(p)
            for s in range(len(ring)):
                a, b = ring[s], ring[(s + 1) % len(ring)]
                edges.add((min(a, b), max(a, b)))
        intra_edges = sorted(edges)

        if len(intra_edges) != self.expected_rim_edges:
            raise RuntimeError(
                "Fatal: expected %d intra-manifold rim edges, got %d"
                % (self.expected_rim_edges, len(intra_edges)))
        seen_q = set()
        for a, b in intra_edges:
            seen_q.add(a)
            seen_q.add(b)
        if len(seen_q) != self.qpm:
            raise RuntimeError("Fatal: rim edges cover %d qubits, expected %d"
                               % (len(seen_q), self.qpm))
        in_portals = dict((q, 0) for q in range(self.qpm))
        for p in range(LOCAL_EDGES):
            for q in self.portal_ring(p):
                in_portals[q] += 1
        if any(v != 2 for v in in_portals.values()):
            raise RuntimeError("Fatal: all-boundary property violated")
        adj = dict((q, set()) for q in range(self.qpm))
        for a, b in intra_edges:
            adj[a].add(b)
            adj[b].add(a)
        seen = set([0])
        stack = [0]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(seen) != self.qpm:
            raise RuntimeError("Fatal: manifold coupling graph is disconnected")
        groups = dict((k, self.edge_local_qubits(k)) for k in range(LOCAL_EDGES))
        return intra_edges, groups

    # -- junctions -----------------------------------------------------
    def junction_slot(self, manifold, neighbour):
        return NEIGHBOURS[manifold].index(neighbour)

    def junction_pairing(self):
        """Slot-index map from side A to side B within a junction.

        Identity = trivial gluing. Order-reversal within the corner block and
        within the mid-edge block = the discrete Moebius half-twist.
        """
        if not self.twist:
            return list(range(self.qpe))
        corners = list(range(self.c_corner))[::-1]
        midedge = [self.c_corner + i for i in range(self.e_midedge)][::-1]
        return corners + midedge

    def _build_junction_table(self):
        pairing = self.junction_pairing()
        table = []
        for ji, (ma, mb) in enumerate(JUNCTION_PAIRS):
            ka = self.junction_slot(ma, mb)
            kb = self.junction_slot(mb, ma)
            qa_list = self.edge_local_qubits(ka)
            qb_list = self.edge_local_qubits(kb)
            bonds = [(qa_list[i], qb_list[pairing[i]]) for i in range(self.qpe)]
            table.append({
                "index": ji, "ma": ma, "mb": mb,
                "slot_a": ka, "slot_b": kb,
                "qubits_a": qa_list, "qubits_b": qb_list,
                "bonds": bonds, "pairing": list(pairing),
                "label": "J%d%d" % (ma, mb),
            })
        used = dict((m, []) for m in range(N_MANIFOLDS))
        for rec in table:
            used[rec["ma"]].append(rec["slot_a"])
            used[rec["mb"]].append(rec["slot_b"])
        for m, slots in used.items():
            if sorted(slots) != list(range(LOCAL_EDGES)):
                raise RuntimeError(
                    "Fatal: manifold %d local edge usage %s != %s"
                    % (m, sorted(slots), list(range(LOCAL_EDGES))))
        return table

    # -- global (shared Hilbert space) map ------------------------------
    def _build_global_map(self):
        """(manifold, local_qubit) -> global unique-qubit index.

        Global index of bond i of junction j is j * qpe + i. Both endpoints of
        the bond -- which are the SAME physical qubit -- map there.
        """
        gmap = {}
        for rec in self.junctions:
            for i, (qa, qb) in enumerate(rec["bonds"]):
                g = rec["index"] * self.qpe + i
                gmap[(rec["ma"], qa)] = g
                gmap[(rec["mb"], qb)] = g
        if len(gmap) != N_MANIFOLDS * self.qpm:
            raise RuntimeError("Fatal: global map covers %d of %d slots"
                               % (len(gmap), N_MANIFOLDS * self.qpm))
        if len(set(gmap.values())) != self.unique_qubits:
            raise RuntimeError("Fatal: global map hits %d distinct qubits, "
                               "expected %d"
                               % (len(set(gmap.values())), self.unique_qubits))
        for g in range(self.unique_qubits):
            n = sum(1 for v in gmap.values() if v == g)
            if n != 2:
                raise RuntimeError("Fatal: global qubit %d claimed %d times "
                                   "(expected exactly 2)" % (g, n))
        return gmap

    def _build_global_edges(self):
        """Union of all four manifolds' rim edges, in global indices.

        Returned as a sorted list of (ga, gb, multiplicity). Multiplicity > 1
        means two manifolds contribute the same physical bond; the exact
        Hamiltonian keeps the weight so that it matches the sum over
        manifolds used by the mean-field engine.
        """
        counts = {}
        for m in range(N_MANIFOLDS):
            for (a, b) in self.intra_edges:
                ga = self.global_map[(m, a)]
                gb = self.global_map[(m, b)]
                if ga == gb:
                    raise RuntimeError("Fatal: self-loop in global edge set")
                key = (min(ga, gb), max(ga, gb))
                counts[key] = counts.get(key, 0) + 1
        return sorted((a, b, c) for (a, b), c in counts.items())

    # -- helpers --------------------------------------------------------
    def global_block(self, junction_index):
        """The unique-qubit indices carried by one junction."""
        j = junction_index
        return list(range(j * self.qpe, (j + 1) * self.qpe))

    def manifold_global_qubits(self, m):
        return [self.global_map[(m, q)] for q in range(self.qpm)]

    def summary(self):
        return {
            "c_corner": self.c_corner, "e_midedge": self.e_midedge,
            "qubits_per_edge": self.qpe, "qubits_per_manifold": self.qpm,
            "rim_edges": len(self.intra_edges),
            "rim_edges_expected": self.expected_rim_edges,
            "unique_qubits": self.unique_qubits,
            "global_edges": len(self.global_edges),
            "global_edge_multiplicities": sorted(set(
                c for _, _, c in self.global_edges)),
            "leg_dim": 2 ** self.qpe,
            "twist": self.twist,
        }


REV311 = dict(c_corner=3, e_midedge=6, twist=True)
SMALL = dict(c_corner=2, e_midedge=2, twist=True)
TINY = dict(c_corner=1, e_midedge=1, twist=True)


# =====================================================================
# STATEVECTOR BACKENDS
# =====================================================================
# Statevector backends with a single shared gate convention.
#
# Convention (both backends): R_P(a) = exp(-i * a * P / 2), little-endian
# amplitude indexing (bit q of the amplitude index is qubit q).
#
# Under this convention the Rev 311 Trotter body is a correct SECOND-order
# (Strang) splitting of
#
#     H = -J * sum_<ab> Z_a Z_b  -  hx * sum_a X_a  -  hz * sum_a Z_a
#
# with theta_x = -hx*dt applied twice (symmetric wrap), theta_z = -2*hz*dt,
# theta_zz = -J*dt through the CNOT-RZ-CNOT identity. It is second order and not
# first order because the Z and ZZ terms are mutually diagonal, so the symmetric
# X wrap makes the whole step exp(-iA dt/2) exp(-iB dt) exp(-iA dt/2). Verified
# numerically: local error scales as dt**3.
#
# NumpyStatevector exists so the topology, leg-ordering and boundary code can be
# validated without a GPU. QrackStatevector is the production path and keeps the
# Rev 311 Pauli-code / angle-scale autodetect.

_SQ2 = 1.0 / math.sqrt(2.0)
_H = np.array([[_SQ2, _SQ2], [_SQ2, -_SQ2]], dtype=np.complex128)
_PAULI = {
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def _rot(pauli, angle):
    c = math.cos(0.5 * angle)
    s = math.sin(0.5 * angle)
    return c * np.eye(2, dtype=np.complex128) - 1j * s * _PAULI[pauli]


class NumpyStatevector(object):
    """Dense reference simulator. Memory: 2**n * 16 bytes (complex128)."""

    def __init__(self, n, dtype=np.complex128):
        self.n = int(n)
        self.dtype = dtype
        self.psi = np.zeros((2,) * self.n, dtype=dtype)
        self.psi[(0,) * self.n] = 1.0

    def _axis(self, q):
        # little-endian qubit q  <->  tensor axis (n-1-q)
        return self.n - 1 - int(q)

    def _apply1(self, u, q):
        ax = self._axis(q)
        self.psi = np.moveaxis(
            np.tensordot(u.astype(self.dtype), self.psi, axes=([1], [ax])),
            0, ax)

    def h(self, q):
        self._apply1(_H, q)

    def rx(self, a, q):
        self._apply1(_rot("X", a), q)

    def ry(self, a, q):
        self._apply1(_rot("Y", a), q)

    def rz(self, a, q):
        self._apply1(_rot("Z", a), q)

    def zz(self, theta, q1, q2):
        """exp(-i * theta * Z_q1 Z_q2), matching CNOT-RZ(2t)-CNOT."""
        a1, a2 = self._axis(q1), self._axis(q2)
        sh = [1] * self.n
        sh[a1] = 2
        z1 = np.array([1.0, -1.0]).reshape(sh)
        sh = [1] * self.n
        sh[a2] = 2
        z2 = np.array([1.0, -1.0]).reshape(sh)
        self.psi = self.psi * np.exp(-1j * theta * z1 * z2)

    def expect1(self, pauli, q):
        ax = self._axis(q)
        out = np.moveaxis(
            np.tensordot(_PAULI[pauli].astype(self.dtype), self.psi,
                         axes=([1], [ax])), 0, ax)
        return float(np.real(np.vdot(self.psi, out)))

    def expectzz(self, q1, q2):
        a1, a2 = self._axis(q1), self._axis(q2)
        sh = [1] * self.n
        sh[a1] = 2
        z1 = np.array([1.0, -1.0]).reshape(sh)
        sh = [1] * self.n
        sh[a2] = 2
        z2 = np.array([1.0, -1.0]).reshape(sh)
        p = np.abs(self.psi) ** 2
        return float(np.sum(p * z1 * z2))

    def ket(self):
        return self.psi.reshape(-1).copy()

    def set_ket(self, v):
        v = np.asarray(v, dtype=self.dtype).reshape((2,) * self.n)
        nrm = np.linalg.norm(v)
        if nrm <= 0.0:
            raise ValueError("set_ket got a zero vector")
        self.psi = v / nrm

    def rdm(self, qubits):
        """Reduced density matrix on the given qubits, in little-endian order
        within the subsystem (qubits[0] is the least significant bit)."""
        qubits = list(qubits)
        keep = [self._axis(q) for q in qubits]
        rest = [a for a in range(self.n) if a not in keep]
        # order kept axes most-significant-first so that the flattened
        # subsystem index is little-endian in qubits[]
        order = [self._axis(q) for q in reversed(qubits)]
        t = np.transpose(self.psi, order + rest)
        d = 2 ** len(qubits)
        t = t.reshape(d, -1)
        return t @ t.conj().T


class QrackStatevector(object):
    """PyQrack backend. Autodetects Pauli codes and the angle convention the
    same way Rev 311 does, then exposes the NumpyStatevector interface."""

    def __init__(self, n, gpu=True, device=None):
        if device is not None:
            os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(device)
            os.environ["QRACK_QPAGER_DEVICES"] = str(device)
            os.environ["QRACK_QUNITMULTI_DEVICES"] = str(device)
        from pyqrack import QrackSimulator
        self._QS = QrackSimulator
        self.n = int(n)
        self._detect()
        self.sim = QrackSimulator(qubit_count=self.n,
                                  is_binary_decision_tree=False,
                                  is_stabilizer_hybrid=False, is_gpu=gpu)

    def _detect(self):
        QS = self._QS
        thresh = 0.5

        def find(prep, flip, exclude):
            p = QS(qubit_count=1, is_binary_decision_tree=False)
            prep(p)
            v0 = {}
            for code in range(8):
                if code in exclude:
                    continue
                try:
                    v0[code] = p.pauli_expectation([0], [code])
                except Exception:
                    pass
            flip(p)
            for code, a in v0.items():
                try:
                    b = p.pauli_expectation([0], [code])
                except Exception:
                    continue
                if abs(a) > thresh and abs(b) > thresh and a * b < 0:
                    del p
                    return code, (1.0 if a > 0 else -1.0)
            del p
            return None, None

        self.PZ, self.SZ = find(lambda p: None, lambda p: p.x(0), set())
        if self.PZ is None:
            raise RuntimeError("Fatal: could not autodetect PZ code")
        self.PX, self.SX = find(lambda p: p.h(0), lambda p: p.z(0), {self.PZ})
        if self.PX is None:
            raise RuntimeError("Fatal: could not autodetect PX code")
        c, s = math.cos(math.pi / 4.0), math.sin(math.pi / 4.0)
        self.PY, self.SY = find(
            lambda p: p.mtrx([complex(c, 0), complex(0, s),
                              complex(0, s), complex(c, 0)], 0),
            lambda p: p.z(0), {self.PX, self.PZ})
        if self.PY is None:
            raise RuntimeError("Fatal: could not autodetect PY code")

        m = QS(qubit_count=1, is_binary_decision_tree=False)
        m.r(self.PX, math.pi, 0)
        corr = self.SZ * m.pauli_expectation([0], [self.PZ])
        if abs(corr + 1.0) < 0.1:
            self.ANGLE_SCALE = 1.0
        elif abs(corr - 1.0) < 0.1:
            self.ANGLE_SCALE = 0.5
        else:
            raise RuntimeError(
                "Fatal: r(PX,pi) gave ambiguous SIGN_Z*<Z> = %.6f" % corr)
        del m
        self._code = {"X": self.PX, "Y": self.PY, "Z": self.PZ}
        self._sign = {"X": self.SX, "Y": self.SY, "Z": self.SZ}

    def h(self, q):
        self.sim.h(int(q))

    def _r(self, pauli, a, q):
        self.sim.r(self._code[pauli], float(a) * self.ANGLE_SCALE, int(q))

    def rx(self, a, q):
        self._r("X", a, q)

    def ry(self, a, q):
        self._r("Y", a, q)

    def rz(self, a, q):
        self._r("Z", a, q)

    def zz(self, theta, q1, q2):
        q1, q2 = int(q1), int(q2)
        self.sim.mcx([q1], q2)
        self.rz(2.0 * theta / self.ANGLE_SCALE, q2)
        self.sim.mcx([q1], q2)

    def expect1(self, pauli, q):
        return self._sign[pauli] * float(
            self.sim.pauli_expectation([int(q)], [self._code[pauli]]))

    def expectzz(self, q1, q2):
        return float(self.sim.pauli_expectation(
            [int(q1), int(q2)], [self.PZ, self.PZ]))

    def ket(self):
        return np.asarray(self.sim.out_ket(), dtype=np.complex128)

    def set_ket(self, v):
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        nrm = np.linalg.norm(v)
        if nrm <= 0.0:
            raise ValueError("set_ket got a zero vector")
        self.sim.in_ket((v / nrm).tolist())

    def fidelity(self):
        try:
            return float(self.sim.get_unitary_fidelity())
        except AttributeError:
            return 1.0


def make_backend(n, kind="numpy", **kw):
    if kind == "numpy":
        return NumpyStatevector(n, **kw)
    if kind == "qrack":
        return QrackStatevector(n, **kw)
    raise ValueError("unknown backend %r" % (kind,))


def trotter_step(sim, qubits, edges, J, hx, hz, dt, edge_weight=None):
    """Rev 311 trotter_step_body, generalized to explicit qubit lists and
    optional per-edge weights (used by the exact engine where two manifolds
    can contribute the same physical bond)."""
    tx = -hx * dt
    tz = -2.0 * hz * dt
    for q in qubits:
        sim.rx(tx, q)
    for q in qubits:
        sim.rz(tz, q)
    for i, (q1, q2) in enumerate(edges):
        w = 1.0 if edge_weight is None else float(edge_weight[i])
        sim.zz(-J * dt * w, q1, q2)
    for q in qubits:
        sim.rx(tx, q)


# =====================================================================
# REGION DECOMPOSITION
# =====================================================================
# Corrected K4 Hamiltonian and its Bethe/Kikuchi region decomposition.
#
# TWO BUGS THIS FIXES (both mine, from k4_exact_reference.py):
#
# 1. The global Hamiltonian was multiplicity-weighted: an edge contributed by two
#    manifolds got weight 2, to match Rev 311's sum-over-manifolds bulk energy.
#    That is double counting. Each edge of the shared system is ONE physical bond
#    and gets weight 1. At (3,6) with twist there are 78 distinct edges, not 120
#    manifold-edge instances.
#
# 2. The obvious region assignment -- junction regions carry every edge internal
#    to their block -- silently DELETES edges. At (3,6) twelve edges come out with
#    counting weight zero. The correct assignment gives junction regions only the
#    multiplicity-2 edges (the genuinely double-counted ones) plus the single-site
#    terms of their qubits.
#
# With that fixed, K4 decomposes exactly:
#
#     H  =  sum_m H_m  -  sum_j H_j
#
#     4 manifold regions, counting number +1, qpm qubits each
#     6 junction regions, counting number -1, qpe qubits each
#
# which is Bethe/Kikuchi on a bipartite region graph. Verified weight-1 on every
# edge and every site at (1,1) (1,2) (2,2) (2,3) (3,6).
#
# This is what makes distributed QPE work: the Loschmidt amplitude factorizes
# along the same regions, so manifolds exchange nothing during evolution -- one
# complex scalar each, per k, at readout.
#
# It remains an APPROXIMATION for the amplitude even though it is exact for the
# Hamiltonian: K4 has four triangles and Bethe is exact only on trees.

class K4Regions(object):
    def __init__(self, topo):
        self.topo = topo
        self.blocks = [set(topo.global_block(j)) for j in range(N_JUNCTIONS)]
        self.unit_edges = [(a, b) for a, b, _ in topo.global_edges]
        self.heavy = set((a, b) for a, b, c in topo.global_edges if c == 2)
        gm = topo.global_map

        self.manifold_edges, self.manifold_qubits = [], []
        for m in range(N_MANIFOLDS):
            self.manifold_edges.append(
                [(gm[(m, a)], gm[(m, b)]) for a, b in topo.intra_edges])
            self.manifold_qubits.append([gm[(m, q)] for q in range(topo.qpm)])

        self.junction_edges, self.junction_qubits = [], []
        for j in range(N_JUNCTIONS):
            self.junction_qubits.append(sorted(self.blocks[j]))
            self.junction_edges.append(
                [e for e in sorted(self.heavy)
                 if e[0] in self.blocks[j] and e[1] in self.blocks[j]])
        self.verify()

    def verify(self):
        tot = {}
        for ed in self.manifold_edges:
            for a, b in ed:
                k = (min(a, b), max(a, b))
                tot[k] = tot.get(k, 0) + 1
        for ed in self.junction_edges:
            for a, b in ed:
                k = (min(a, b), max(a, b))
                tot[k] = tot.get(k, 0) - 1
        bad = sorted(set(tot.values()) - set([1]))
        if bad:
            raise RuntimeError("region edge weights %s != 1; decomposition is "
                               "not a partition of H" % bad)
        if len(tot) != len(self.unit_edges):
            raise RuntimeError("regions cover %d edges, H has %d"
                               % (len(tot), len(self.unit_edges)))
        site = dict((q, 0) for q in range(self.topo.unique_qubits))
        for qs in self.manifold_qubits:
            for q in qs:
                site[q] += 1
        for qs in self.junction_qubits:
            for q in qs:
                site[q] -= 1
        bad = sorted(set(site.values()) - set([1]))
        if bad:
            raise RuntimeError("region site weights %s != 1" % bad)
        return True

    def region_specs(self):
        """[(label, counting_number, n_qubits, local_edges, global_qubits)]

        Local edges are re-indexed into each region's own simulator, so a region
        runs standalone on a worker with no global context.
        """
        out = []
        for m in range(N_MANIFOLDS):
            qs = sorted(self.manifold_qubits[m])
            loc = dict((g, i) for i, g in enumerate(qs))
            out.append(("M%d" % m, +1, len(qs),
                        [(loc[a], loc[b]) for a, b in self.manifold_edges[m]],
                        qs))
        for j in range(N_JUNCTIONS):
            qs = self.junction_qubits[j]
            loc = dict((g, i) for i, g in enumerate(qs))
            out.append((self.topo.junctions[j]["label"], -1, len(qs),
                        [(loc[a], loc[b]) for a, b in self.junction_edges[j]],
                        qs))
        return out

    def summary(self):
        return {"unique_qubits": self.topo.unique_qubits,
                "distinct_edges": len(self.unit_edges),
                "multiplicity_2_edges": len(self.heavy),
                "manifold_qubits": self.topo.qpm,
                "junction_qubits": self.topo.qpe,
                "verified": True}


def hamiltonian_expectation(sim, edges, qubits, J, hx, hz):
    """<H> for H = -J sum ZZ - hx sum X - hz sum Z, all weights 1."""
    e = 0.0
    for a, b in edges:
        e += -J * sim.expectzz(a, b)
    for q in qubits:
        e += -hx * sim.expect1("X", q) - hz * sim.expect1("Z", q)
    return float(e)


# =====================================================================
# DISTRIBUTED COMPOSITE QPE
# =====================================================================
# Distributed Composite Quantum Phase Estimation for K4.
#
# THE REDUCTION
# -------------
# For the single-ancilla information-theory QPE of arXiv:2505.09133,
#
#     P(m | k, beta) = ( 1 + (-1)^m * Re[ e^{i beta} <Phi|U^k|Phi> ] ) / 2
#
# so the ENTIRE workload -- every k, every beta, the estimator, the energy -- is a
# function of one complex number per k: the Loschmidt amplitude
#
#     a_k = <Phi| U^k |Phi>.
#
# There is no ancilla to simulate and no controlled-U to distribute. This matters
# because a shared ancilla under controlled-U entangles with all four manifolds at
# once, and four independent pure manifold states cannot hold that. Distributing
# the QPE circuit itself is not an approximation with a measurable error; it is
# structurally impossible. Distributing the AMPLITUDE is both possible and cheap.
#
# THE COMPOSITE ESTIMATOR
# -----------------------
# a_k is a GLOBAL observable, so the kick-based boundaries (meanfield, gnn, bp)
# cannot produce it -- they steer local Bloch vectors and the composite has no
# global state to overlap with. The boundary has to move out of the evolution and
# into the estimator, along the region decomposition of k4_regions:
#
#     a_k^bethe = prod_m a_k^(m) / prod_j a_k^(j)
#
# Each region evolves ALONE. No kicks, no messages, no ket transport. The only
# communication is one complex scalar per region per k: 10 numbers, against 1 GB
# kets for BP.
#
# WHAT IS AND IS NOT GUARANTEED
# -----------------------------
# The region decomposition is EXACT for the Hamiltonian (verified weight-1 on
# every edge and site at five sizes). It is an APPROXIMATION for the amplitude,
# because K4 has four triangles and Bethe is exact only on trees. Measured at
# qpe=2: naive product over manifolds gives 100% energy error, Bethe gives ~3%.
# Score it against the exact reference at (1,1)/(1,2)/(2,2); at (3,6) there is no
# exact reference and product-vs-bethe divergence is the only self-consistency
# signal you have.
#
# AMPLITUDE EXTRACTION ON GPU
# ---------------------------
# For |Phi> = |+>^n, a_k = <Phi|U^k|Phi> is obtained locally in the worker as
# vdot(psi_0, psi_k) via out_ket. That is ~1 GB of HOST RAM per evaluation at 27
# qubits but ZERO pipe traffic: the worker sends back the scalar. Evaluate only at
# the k values you need (kmax is ~12, not 100). The Hadamard-test alternative
# needs a 28th qubit and doubles VRAM; it is not worth it here.

# =====================================================================
# AMPLITUDES
# =====================================================================
def prepare_plus(sim):
    for q in range(sim.n):
        sim.h(q)


def loschmidt_series(n, edges, J, hx, hz, dt, kmax, backend="numpy",
                     prep=prepare_plus, **bk):
    """a_k = <Phi|U^k|Phi> for k = 1..kmax, U = one Trotter step.

    All edges carry weight 1 -- this is the corrected Hamiltonian. Do not pass
    multiplicity weights here.
    """
    sim = make_backend(n, backend, **bk)
    prep(sim)
    psi0 = sim.ket().copy()
    out = np.zeros(kmax, dtype=complex)
    for k in range(kmax):
        trotter_step(sim, range(n), edges, J, hx, hz, dt)
        out[k] = complex(np.vdot(psi0, sim.ket()))
    return out


def region_amplitudes(regions, J, hx, hz, dt, kmax, backend="numpy", **bk):
    """a_k for every region, keyed by label. Regions are independent: this is
    the loop you would run on four GPU workers plus six tiny CPU sims."""
    out = {}
    for label, cnum, nq, edges, _g in regions.region_specs():
        out[label] = (cnum, loschmidt_series(nq, edges, J, hx, hz, dt, kmax,
                                             backend=backend, **bk))
    return out


def combine_regions(amps, mode="bethe"):
    """Composite amplitude from per-region amplitudes.

    mode="bethe"   : prod a^(+1) / prod a^(-1)   -- the full region estimator
    mode="product" : manifolds only, no junction correction. This double counts
                     every shared qubit and is the naive baseline; it is what
                     you get if you forget that the regions overlap.
    """
    num = None
    den = None
    for label, (cnum, a) in amps.items():
        if cnum > 0:
            num = a.copy() if num is None else num * a
        elif mode == "bethe":
            den = a.copy() if den is None else den * a
    if num is None:
        raise RuntimeError("no +1 regions")
    if den is None:
        return num
    safe = np.where(np.abs(den) < 1e-12, 1e-12, den)
    return num / safe


# =====================================================================
# INFORMATION-THEORY QPE (arXiv:2505.09133 Sec. II A, Eqs 4-7, 15)
# =====================================================================
def sample_outcomes(a_k, kmax, ns, rng, decoh_a=0.0, decoh_b=0.0):
    """Draw Ns measurement outcomes from P(m|k,beta).

    P(m|k,beta) = (1 + (-1)^m Re[e^{i beta} a_k]) / 2, optionally damped by the
    paper's decoherence model exp(-a*k - b) to emulate hardware noise.
    """
    ks = rng.integers(1, kmax + 1, size=ns)
    betas = rng.choice([0.0, math.pi / 2.0], size=ns)
    damp = np.exp(-decoh_a * ks - decoh_b) if (decoh_a or decoh_b) else 1.0
    amp = a_k[ks - 1]
    p0 = 0.5 * (1.0 + damp * np.real(np.exp(1j * betas) * amp))
    p0 = np.clip(p0, 0.0, 1.0)
    ms = (rng.random(ns) > p0).astype(int)
    return ks, betas, ms


def phase_posterior(ks, betas, ms, grid, decoh_a=0.0, decoh_b=0.0):
    """log Q(phi) = sum_l log[(1 + e^{-a k - b} cos(k phi + beta - m pi))/2].

    Noise-aware form (Eq. 15) when decoh_a/b are given, plain (Eq. 6) otherwise.
    """
    phi = grid[None, :]
    k = ks[:, None]
    arg = k * phi + betas[:, None] - ms[:, None] * math.pi
    damp = (np.exp(-decoh_a * k - decoh_b) if (decoh_a or decoh_b) else 1.0)
    q = 0.5 * (1.0 + damp * np.cos(arg))
    return np.log(np.clip(q, 1e-300, None)).sum(axis=0)


def estimate_phase(a_k, kmax, ns, rng, grid_n=4001, decoh_a=0.0, decoh_b=0.0):
    grid = np.linspace(0.0, 2.0 * math.pi, grid_n, endpoint=False)
    ks, betas, ms = sample_outcomes(a_k, kmax, ns, rng, decoh_a, decoh_b)
    lq = phase_posterior(ks, betas, ms, grid, decoh_a, decoh_b)
    return float(grid[int(np.argmax(lq))])


def estimate_energy(a_k, kmax, dt, ns, rng, repeats=40, **kw):
    """Circular-mean phase estimate over bootstrap repeats -> energy.

    U = e^{-iH dt} has eigenvalues e^{i phi} with phi = -E dt, so E = -phi/dt,
    branch-resolved into (-pi/dt, pi/dt]. ALIASING: pick dt so that |E| dt < pi
    for the eigenvalues you care about, or the estimate wraps silently.
    """
    # cost is repeats * ns * grid_n; keep the product under ~1e8
    phis = np.array([estimate_phase(a_k, kmax, ns, rng, **kw)
                     for _ in range(repeats)])
    mean = np.angle(np.mean(np.exp(1j * phis)))
    holevo = 1.0 / max(abs(np.mean(np.exp(1j * phis))) ** 2, 1e-300) - 1.0
    if mean > math.pi:
        mean -= 2.0 * math.pi
    return -mean / dt, math.sqrt(max(holevo, 0.0)) / dt


def energy_from_slope(a_k, dt):
    """Cheap alternative readout: unwrap arg(a_k) and fit the slope. Uses no
    sampling, so it separates BOUNDARY error from SHOT noise."""
    kk = np.arange(1, len(a_k) + 1) * dt
    return float(-np.polyfit(kk, np.unwrap(np.angle(a_k)), 1)[0])


def amplitude_decay(a_k):
    """|a_k| falls from Trotter dephasing alone. This is the classical analogue
    of the paper's decoherence parameter q, and it is what actually bounds kmax:
    the signal dies before the boundary approximation does."""
    return np.abs(a_k)


def fit_decay(a_k):
    """Fit |a_k| ~ exp(-lam k); returns lam."""
    k = np.arange(1, len(a_k) + 1)
    y = np.log(np.clip(np.abs(a_k), 1e-12, None))
    return float(-np.polyfit(k, y, 1)[0])


# =====================================================================
# EXACT MERGED REFERENCE
# =====================================================================
# Exact K4 on a genuinely shared Hilbert space.
#
# The four manifolds are merged into ONE simulator of UNIQUE_QUBITS qubits via
# K4Topology.global_map. There is no g_junction here at all: the identification
# of the two copies of each shared qubit is enforced structurally, which is the
# g -> infinity limit of the Rev 311 penalty.
#
# At (c_corner, e_midedge) = (2, 2) this is 24 qubits: 268 MB as complex128,
# 134 MB as complex64, trivial for a V340 and fine on CPU. That is the whole
# point -- it gives ground truth for observables the mean field cannot produce
# even in principle:
#
#   * mutual information I(M_i : M_j) between manifolds, which is identically
#     zero in the mean field because the sims are independent
#   * connected correlators across a junction
#   * the true energy, against which the mean-field energy can be scored
#
# It also exports the training set for k4_gnn_message.

def schedule(t, total_steps, initial_hx, target_J, target_hx, target_hz):
    s = t / max(1, (total_steps - 1))
    return (s,
            (1.0 - s) * initial_hx + s * target_hx,
            s * target_J,
            s * target_hz)


def _vn_bits(rho):
    w = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    w = np.clip(np.real(w), 1e-300, None)
    w = w / w.sum()
    return float(-np.sum(w * np.log2(w)))


class K4ExactEngine(object):
    def __init__(self, topo, backend="numpy", dtype=np.complex128,
                 legacy_multiplicity=False, **bk):
        self.topo = topo
        self.n = topo.unique_qubits
        self.backend = backend
        self.dtype = dtype
        self.bk = bk
        self.reset_simulator()
        
        # CORRECTED (see k4_regions.py): weight 1 per DISTINCT edge. The old
        # multiplicity weighting double-counted every bond internal to a
        # junction block, to match Rev 311's sum-over-manifolds bulk energy.
        # That made the Hamiltonian inconsistent with its own region
        # decomposition. Pass legacy_multiplicity=True only to reproduce the
        # old (wrong) numbers.
        self.edges = [(a, b) for a, b, _ in topo.global_edges]
        self.weights = ([c for _, _, c in topo.global_edges]
                        if legacy_multiplicity
                        else [1] * len(self.edges))
        self.history = []

    def reset_simulator(self):
        """Defensive reset to ensure a fresh simulator state for each run."""
        if self.backend == "numpy":
            self.sim = make_backend(self.n, "numpy", dtype=self.dtype)
        else:
            self.sim = make_backend(self.n, self.backend, **self.bk)

    def prepare_plus(self):
        for q in range(self.n):
            self.sim.h(q)

    def site_vectors(self):
        """(n, 3) array of <X>,<Y>,<Z> per unique qubit."""
        out = np.zeros((self.n, 3))
        for q in range(self.n):
            out[q, 0] = self.sim.expect1("X", q)
            out[q, 1] = self.sim.expect1("Y", q)
            out[q, 2] = self.sim.expect1("Z", q)
        return out

    def energy(self, J, hx, hz, sv=None):
        """Exact <H> with H = -J sum_w ZZ - hx sum X - hz sum Z, summed over
        UNIQUE qubits (not over manifold copies). The Rev 311 bulk_e sums over
        4 x qpm = 2 x unique sites and is therefore twice this on the
        single-site terms."""
        if sv is None:
            sv = self.site_vectors()
        e = -hx * float(sv[:, 0].sum()) - hz * float(sv[:, 2].sum())
        for (a, b), w in zip(self.edges, self.weights):
            e += -J * w * self.sim.expectzz(a, b)
        return float(e)

    def _entropy(self, qubits):
        """von Neumann entropy of a subset, in bits. Uses the complement when
        it is smaller -- the global state is pure, so S(A) = S(A_complement).
        Without this, I(M0:M1) at qpe=4 would need a 2**16 x 2**16 matrix."""
        qubits = sorted(qubits)
        comp = [q for q in range(self.n) if q not in set(qubits)]
        use = qubits if len(qubits) <= len(comp) else comp
        if not use:
            return 0.0
        return _vn_bits(self.sim.rdm(use))

    def manifold_mutual_information(self, ma, mb):
        """I(M_a : M_b) in bits. The two manifolds SHARE one junction block,
        so this is computed on their exclusive qubits only -- otherwise the
        shared qubits make it trivially large."""
        ga = set(self.topo.manifold_global_qubits(ma))
        gb = set(self.topo.manifold_global_qubits(mb))
        ea = sorted(ga - gb)
        eb = sorted(gb - ga)
        return (self._entropy(ea) + self._entropy(eb)
                - self._entropy(ea + eb))

    def block_mutual_information(self, j1, j2):
        b1 = self.topo.global_block(j1)
        b2 = self.topo.global_block(j2)
        return (self._entropy(b1) + self._entropy(b2)
                - self._entropy(b1 + b2))

    def run(self, total_steps=60, dt=0.04, initial_hx=3.0, target_J=1.0,
            target_hx=0.5, target_hz=0.2, measure_every=1, verbose=True,
            collect_mi=True):
        self.reset_simulator()
        self.prepare_plus()
        self.history = []
        for t in range(total_steps):
            s, hx, J, hz = schedule(t, total_steps, initial_hx, target_J,
                                    target_hx, target_hz)
            pre = self.site_vectors()
            trotter_step(self.sim, range(self.n), self.edges, J, hx, hz, dt,
                         edge_weight=self.weights)
            post = self.site_vectors()

            if t % measure_every == 0 or t == total_steps - 1:
                rec = {"step": t, "s": s, "J": J, "hx": hx, "hz": hz,
                       "energy": self.energy(J, hx, hz, post),
                       "pre": pre, "post": post}
                if collect_mi:
                    rec["mi_manifold"] = dict(
                        ("M%dM%d" % (a, b),
                         self.manifold_mutual_information(a, b))
                        for a, b in MOEBIUS_PAIRS)
                    rec["mi_block_loop"] = self.block_mutual_information(
                        self.topo.loop_indices[0], self.topo.loop_indices[1])
                self.history.append(rec)
                if verbose:
                    mi = rec.get("mi_manifold", {})
                    mstr = " ".join("%s:%.4f" % (k, v)
                                    for k, v in sorted(mi.items()))
                    print("Step %03d | s=%.2f | E=%+.5f | MI %s"
                          % (t, s, rec["energy"], mstr))
        return self.history


def free_step_product(topo, sv_manifold, J, hx, hz, dt):
    """Evolve ONE manifold, as an isolated product state with the given Bloch
    vectors, through one Trotter step using only its own rim edges.

    This is exactly what the Rev 311 mean field does with zero kick. The
    difference between this and the exact evolution is the residual the
    message function has to learn.
    """
    qpm = topo.qpm
    sim = make_backend(qpm, "numpy")
    for q in range(qpm):
        _prepare_bloch(sim, q, sv_manifold[q])
    trotter_step(sim, range(qpm), topo.intra_edges, J, hx, hz, dt)
    out = np.zeros((qpm, 3))
    for q in range(qpm):
        out[q, 0] = sim.expect1("X", q)
        out[q, 1] = sim.expect1("Y", q)
        out[q, 2] = sim.expect1("Z", q)
    return out


def _prepare_bloch(sim, q, vec):
    """Prepare the pure state whose Bloch vector points along vec (renormalized
    to the sphere; mixedness cannot be represented in a statevector)."""
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    r = math.sqrt(x * x + y * y + z * z)
    if r < 1e-12:
        sim.h(q)
        return
    x, y, z = x / r, y / r, z / r
    theta = math.acos(max(-1.0, min(1.0, z)))
    phi = math.atan2(y, x)
    sim.ry(theta, q)
    sim.rz(phi, q)


def build_dataset(topo, history, dt):
    """Turn an exact run into (features, target) pairs for the edge function.

    For each measured step and each manifold, the free product-state step is
    computed from the exact pre-step Bloch vectors. The target is

        residual = exact_post - free_post

    per site, which is the entire effect the mean field is missing: the rim
    edges contributed by neighbouring manifolds on shared qubits, plus the
    aggregate entanglement correction insofar as it is a function of the Bloch
    vectors.

    Features are per-BOND and shared across bond index, so the learned function
    transfers from qpe = 4 (training) to qpe = 9 (deployment).
    """
    gmap = topo.global_map
    F, Y = [], []
    for rec in history:
        pre, post = rec["pre"], rec["post"]
        J, hx, hz, s = rec["J"], rec["hx"], rec["hz"], rec["s"]
        pre_m = {}
        for m in range(N_MANIFOLDS):
            pre_m[m] = np.array([pre[gmap[(m, q)]] for q in range(topo.qpm)])
        free_m = dict((m, free_step_product(topo, pre_m[m], J, hx, hz, dt))
                      for m in range(N_MANIFOLDS))
        for jrec in topo.junctions:
            ma, mb = jrec["ma"], jrec["mb"]
            for side, mm, mo, qs in (("a", ma, mb, jrec["qubits_a"]),
                                     ("b", mb, ma, jrec["qubits_b"])):
                own = pre_m[mm]
                oth = pre_m[mo]
                blk_own = np.array([own[q] for q in qs])
                oq = (jrec["qubits_b"] if side == "a" else jrec["qubits_a"])
                blk_oth = np.array([oth[q] for q in oq])
                mo_own = blk_own.mean(axis=0)
                mo_oth = blk_oth.mean(axis=0)
                for i, q in enumerate(qs):
                    g = gmap[(mm, q)]
                    resid = post[g] - free_m[mm][q]
                    F.append(np.concatenate([
                        blk_own[i], blk_oth[i], mo_own, mo_oth,
                        [s, J, hx, hz]]))
                    Y.append(resid)
    return np.array(F), np.array(Y)


# =====================================================================
# ALIASING GUARD
# =====================================================================
def check_aliasing(energy_scale, dt, label=""):
    """E = -phase/dt wraps at |E|*dt = pi.

    energy_scale should be a bound on |E| for the eigenvalues you care about;
    the trivial bound J*n_edges + (hx+hz)*n_qubits is fine. Returns the
    recommended dt ceiling and prints a warning when the current dt exceeds it.
    """
    dt_max = math.pi / max(abs(energy_scale), 1e-12)
    if dt > dt_max:
        print("[ALIAS] %sdt=%.4f exceeds pi/|E|=%.4f. estimate_energy() will "
              "resolve the wrong branch; the slope readout still unwraps "
              "correctly." % (label + " " if label else "", dt, dt_max),
              file=sys.stderr)
    return dt_max


def energy_scale_bound(edges, n_qubits, J, hx, hz):
    return abs(J) * len(edges) + (abs(hx) + abs(hz)) * n_qubits


# =====================================================================
# BENCHMARK DRIVER
# =====================================================================
def run_benchmark(c=1, e=2, kmax=12, dt=0.05, J=1.0, hx=2.0, hz=0.2,
                  ns=500, repeats=40, seed=1337, backend="numpy",
                  do_exact=True, csv_out=None):
    """Compare exact / product / bethe on the same H and the same |Phi>."""
    topo = K4Topology(c, e, True)
    reg = K4Regions(topo)
    rng = np.random.default_rng(seed)

    print("[QPE] %s" % json.dumps(reg.summary()))
    print("[QPE] H = -J sum_ZZ - hx sum_X - hz sum_Z  (weight 1 per distinct "
          "edge)  J=%.3f hx=%.3f hz=%.3f dt=%.4f kmax=%d"
          % (J, hx, hz, dt, kmax))
    check_aliasing(energy_scale_bound(reg.unit_edges, topo.unique_qubits,
                                      J, hx, hz), dt, "global")

    amps = region_amplitudes(reg, J, hx, hz, dt, kmax, backend="numpy")
    a_prod = combine_regions(amps, "product")
    a_beth = combine_regions(amps, "bethe")

    series = {}
    if do_exact:
        series["exact"] = loschmidt_series(topo.unique_qubits, reg.unit_edges,
                                           J, hx, hz, dt, kmax,
                                           backend=backend)
    series["product"] = a_prod
    series["bethe"] = a_beth

    ref = series.get("exact", a_beth)
    pr = np.unwrap(np.angle(ref))
    pp = np.unwrap(np.angle(a_prod))
    pb = np.unwrap(np.angle(a_beth))
    if do_exact:
        print("\n  k   |a_exact|   arg_exact     product (err)       bethe (err)")
        for k in range(kmax):
            print("  %2d   %7.4f   %+9.4f   %+9.4f (%+8.4f)  %+9.4f (%+8.4f)"
                  % (k + 1, abs(ref[k]), pr[k], pp[k], pp[k] - pr[k],
                     pb[k], pb[k] - pr[k]))
    else:
        print("\n  k   |a_bethe|   arg_bethe     product (divergence)")
        for k in range(kmax):
            print("  %2d   %7.4f   %+9.4f   %+9.4f (%+8.4f)"
                  % (k + 1, abs(a_beth[k]), pb[k], pp[k], pp[k] - pb[k]))

    print("\n%-9s %12s %12s %12s %10s" % ("estimator", "E_slope", "E_qpe",
                                          "sigma_qpe", "decay_lam"))
    rows, e_ref = [], None
    for name in ("exact", "product", "bethe"):
        if name not in series:
            continue
        a = series[name]
        es = energy_from_slope(a, dt)
        eq, sq = estimate_energy(a, kmax, dt, ns, rng, repeats=repeats)
        lam = fit_decay(a)
        if name == "exact":
            e_ref = es
        rows.append((name, es, eq, sq, lam))
        print("%-9s %12.5f %12.5f %12.5f %10.4f" % (name, es, eq, sq, lam))
    if e_ref is not None:
        print("\nrelative error in E_slope (boundary error, no shot noise):")
        for name, es, eq, sq, lam in rows:
            if name != "exact":
                print("   %-8s %+8.2f%%" % (name,
                                            100.0 * (es - e_ref) / abs(e_ref)))

    n_reg = N_MANIFOLDS + N_JUNCTIONS
    comm = n_reg * kmax * 16
    ket_bytes = N_MANIFOLDS * (2 ** topo.qpm) * 8
    print("\ncommunication: %d complex scalars = %d bytes for the whole run; "
          "regions exchange NOTHING during evolution" % (n_reg * kmax, comm))
    print("               one BP ket round trip at this size is %d bytes "
          "(ratio %.4g)" % (ket_bytes, ket_bytes / max(comm, 1)))

    path = csv_out or ("k4_qpe_benchmark_C%dE%d.csv" % (c, e))
    try:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["k"] + sum([["abs_" + n, "arg_" + n]
                                    for n in series], []))
            for k in range(kmax):
                row = [k + 1]
                for n in series:
                    row += [abs(series[n][k]), float(np.angle(series[n][k]))]
                w.writerow(row)
        print("\nwrote %s" % path)
    except Exception as exc:
        print("[CSV] write failed: %s" % exc, file=sys.stderr)
    return series


# =====================================================================
# SELF-TEST (no GPU)
# =====================================================================
def selftest():
    print("region decomposition:")
    for cfg in [(1, 1), (1, 2), (2, 2), (2, 3), (3, 6)]:
        r = K4Regions(K4Topology(cfg[0], cfg[1], True))
        s = r.summary()
        print("  C=%d E=%d -> %2d qubits, %2d distinct edges (%2d of mult 2); "
              "4x%dq(+1) 6x%dq(-1): VERIFIED"
              % (cfg[0], cfg[1], s["unique_qubits"], s["distinct_edges"],
                 s["multiplicity_2_edges"], s["manifold_qubits"],
                 s["junction_qubits"]))

    print("H == sum_m H_m - sum_j H_j on random states:")
    topo = K4Topology(1, 1, True)
    reg = K4Regions(topo)
    rng = np.random.default_rng(0)
    J, hx, hz = 1.0, 2.0, 0.2
    worst = 0.0
    for _ in range(3):
        s = NumpyStatevector(topo.unique_qubits)
        v = rng.normal(size=2 ** topo.unique_qubits) \
            + 1j * rng.normal(size=2 ** topo.unique_qubits)
        s.set_ket(v)
        he = hamiltonian_expectation(s, reg.unit_edges,
                                     range(topo.unique_qubits), J, hx, hz)
        hm = sum(hamiltonian_expectation(s, reg.manifold_edges[m],
                                         reg.manifold_qubits[m], J, hx, hz)
                 for m in range(N_MANIFOLDS))
        hj = sum(hamiltonian_expectation(s, reg.junction_edges[j],
                                         reg.junction_qubits[j], J, hx, hz)
                 for j in range(N_JUNCTIONS))
        worst = max(worst, abs(he - (hm - hj)))
    print("  max |H_exact - (sum_m - sum_j)| = %.2e" % worst)
    assert worst < 1e-9, "region decomposition does not reproduce H"

    print("trotter order (dt**3 local error => second-order Strang):")
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    errs = []
    for dt in (0.2, 0.1, 0.05):
        s = NumpyStatevector(4)
        for q in range(4):
            s.h(q)
        v0 = s.ket().copy()
        trotter_step(s, range(4), edges, 0.83, 1.31, 0.47, dt)
        s2 = NumpyStatevector(4)
        s2.set_ket(v0)
        n = 64
        for _ in range(n):
            trotter_step(s2, range(4), edges, 0.83, 1.31, 0.47, dt / n)
        errs.append(np.abs(s.ket() - s2.ket()).max())
    order = math.log(errs[0] / errs[2]) / math.log(4.0)
    print("  order = %.2f" % order)
    assert order > 2.5

    print("QPE benchmark at 12 qubits:")
    run_benchmark(c=1, e=1, kmax=12, dt=0.05, ns=300, repeats=20,
                  csv_out="/dev/null")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="K4 Distributed Composite QPE Benchmark")
    sub = ap.add_subparsers(dest="cmd")

    b = sub.add_parser("qpe", help="run the QPE benchmark")
    b.add_argument("--c", type=int, default=1)
    b.add_argument("--e", type=int, default=2)
    b.add_argument("--kmax", type=int, default=12)
    b.add_argument("--dt", type=float, default=0.05)
    b.add_argument("--J", type=float, default=1.0)
    b.add_argument("--hx", type=float, default=2.0)
    b.add_argument("--hz", type=float, default=0.2)
    b.add_argument("--ns", type=int, default=500)
    b.add_argument("--repeats", type=int, default=40)
    b.add_argument("--seed", type=int, default=1337)
    b.add_argument("--backend", default="numpy")
    b.add_argument("--no-exact", action="store_true",
                   help="skip the merged reference (mandatory at C=3 E=6)")

    r = sub.add_parser("regions", help="verify the region decomposition")

    x = sub.add_parser("exact", help="exact anneal + GNN dataset export")
    x.add_argument("--c", type=int, default=2)
    x.add_argument("--e", type=int, default=2)
    x.add_argument("--steps", type=int, default=60)
    x.add_argument("--dt", type=float, default=0.04)
    x.add_argument("--backend", default="numpy")
    x.add_argument("--no-mi", action="store_true")
    x.add_argument("--out", default="k4_exact_dataset.npz")
    x.add_argument("--legacy-multiplicity", action="store_true",
                   help="reproduce the old double-counted Hamiltonian")

    s = sub.add_parser("selftest", help="no-GPU checks of every claim")

    a = ap.parse_args()
    if a.cmd == "qpe":
        run_benchmark(a.c, a.e, a.kmax, a.dt, a.J, a.hx, a.hz, a.ns,
                      a.repeats, a.seed, a.backend, not a.no_exact)
    elif a.cmd == "regions":
        for cfg in [(1, 1), (1, 2), (2, 2), (2, 3), (3, 6)]:
            rr = K4Regions(K4Topology(cfg[0], cfg[1], True))
            print("C=%d E=%d -> %s" % (cfg[0], cfg[1],
                                       json.dumps(rr.summary())))
    elif a.cmd == "exact":
        topo = K4Topology(a.c, a.e, True)
        print("[Exact] %s" % json.dumps(topo.summary()))
        eng = K4ExactEngine(topo, backend=a.backend,
                            legacy_multiplicity=a.legacy_multiplicity)
        hist = eng.run(total_steps=a.steps, dt=a.dt, collect_mi=not a.no_mi)
        F, Y = build_dataset(topo, hist, a.dt)
        np.savez_compressed(a.out, F=F, Y=Y,
                            energy=np.array([h["energy"] for h in hist]),
                            step=np.array([h["step"] for h in hist]),
                            c_corner=a.c, e_midedge=a.e, dt=a.dt)
        print("[Exact] dataset %s: F%s Y%s" % (a.out, F.shape, Y.shape))
    elif a.cmd == "selftest":
        selftest()
    else:
        ap.print_help()
