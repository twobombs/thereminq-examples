#!/usr/bin/env python3
"""
nighthawk_ghz.py -- Layer 1a (GHZ populations + MQC) and Layer 2 (idle-window refocusing) of
"Exact Refocusing of Diagonal Crosstalk for Shor Order Finding on a 10 x 12 Square-Lattice Processor"
(the formal proof; numpy reference shrs_nighthawk_stealth.py), run on Nighthawk registers under
QrackAceBackend, so the paper's statements meet ACE's seams.

Companion of nighthawk_qrack.py, imported as a library (registers, ACE presets and placement, the
coupler classes, the worker pool). Gates are exact; the only departures from the paper's model are
ACE's seam approximation and its sampler. An exact QrackSimulator reference runs beside every ACE run.

Paper objects as implemented here (section numbers of the paper)
  Lemma 1     the register's couplers join the sublattices A = {r + c even}, B = {r + c odd} of the device
              lattice (checked for every register)
  Def. 2      idle drift H_D = sum J_ij Z_i Z_j + sum (dw_i + eps_i) Z_i, with eta = 0 (A4): J_ij = j_zz x
              U(0.75, 1.25) per coupler (seeded), dw_q = 0.022 ((q mod 3) - 1), eps_i ~ N(0, sigma_qs^2)
              quasi-static: one draw per realisation (--repeats); defaults are the paper's section 8 values
  Def. 4      a block with G = I (Lemma 3): one idle window V; --blocks L of them after the preparation.
              No depolarising channel: ACE has no global D_p (A2); lambda from Lemma 3 then measures the
              population loss ACE's seams cause, as an effective survival, exact only under A2
  Def. 5      Walsh sequence (--layer2 dd, the anti-OTOC layer, route B): U(T/4) X_B U(T/4) X_A U(T/4) X_B
              U(T/4) X_A in time order; Theorem 1: W_T(H_D) = I for every J, dw, eps
  Def. 6      compensation (--layer2 compensated, route A): e^{+iT H_hat} e^{-iT H}, H_hat from the true
              J, dw (an oracle Layer 1b: Proposition 1 then leaves the eps term only)
  Lemma 3     lambda per block = [(P0 + P1 - 2/d) / (1 - 2/d)]^(1/L)
  Lemma 4     MQC: Tr[M(phi) rho] at phi_m = pi m / (n + 1), m = 0 .. 2n + 1; the DFT coefficient at n is
              c = rho_{0..0,1..1}; C_parity = 2|c|, Phi_coh = arg c, F_GHZ = (P0 + P1 + C) / 2 = max over
              theta of <GHZ_theta|rho|GHZ_theta>. 2n + 2 settings, not Clifford: units up to --mqc-max
  Cor. 4.1    sum dw = -Phi / (2 L T) (when the unit is the whole register: no outside couplings)
  Cor. 2.1    a GHZ state is blind to every Z_i Z_j inside it
  Def. 8      the OTOC is not estimated here (shrs_nighthawk_stealth.py computes it for the 2 x 4 plaquette);
              Corollary 7.1 (refocused OTOC constant) is what --layer2 dd tests through the GHZ numbers
Not in the paper: the Clifford witness F >= P0 + P1 + P(even X parity) - 1, used for chip-wide states too
wide for MQC, and the seam maps.

Drift on ACE. e^{-i phi Z_i Z_j} is applied as CX_ij e^{-i phi Z_j} CX_ij, so a coupling across a seam goes
through ACE's seam machinery, as the circuit's own CXs do. A coupling from a GHZ unit to a qubit outside every
unit is applied as e^{-i phi z_j Z_i} on the unit qubit, with z_j = +-1 that qubit's Z eigenvalue, tracked
through the Walsh pulses: the same unitary, exactly, without merging units. Couplings among outside qubits are
global phases. Units of one circuit never share a coupler under --layer2 (tiles are spread into groups),
because a ZZ between two GHZ units entangles them and the register would no longer factorise.

Commands
  chip      one GHZ state over every qubit of the register: H on a root, then CX along a spanning tree
            (--trees bfs: breadth-first from the centre; seam: fewest CXs through a seam, Kruskal). With
            the witness readout (Clifford) and no Layer 2 the patches run as stabilizer hybrids, so large
            registers fit (measured: the two-patch ladder up to nighthawk2p57; not nighthawk61r2 on
            PyQrack 2.29.5 / Qrack 10.13.2, whose sampler leaves the tableau). With MQC or Layer 2 the
            patches are dense: small registers only.
  tiles     disjoint k-qubit GHZ chains (--unit k) along lattice rows and columns, every orientation and
            offset; per unit Layer 1a numbers, per qubit the mean over its units, by seam distance.
  plot      the map of a records file (--in).
  selftest  Lemma 3, Lemma 4 and the witness against numpy; the readout gates, Theorem 1, Corollaries 2.1
            and 4.1, and the outside-coupling identity on the exact simulator.

Per-unit hook (--layers): pre / after_cx / post inside a unit's preparation. fold:F[:all] adds F CX pairs
after every seam CX (every CX): ACE's seam error scaled 2F + 1 times.

Records (--out, JSONL): one ghz_map per run, one ghz per state (chip: per tree; tiles: per unit) with the
reference implementation's Layer 1a names (P0, P1, C_parity, Phi_coh, F_GHZ, lambda_per_block,
sum_dw_estimate) for ACE and, prefixed exact_, for the exact reference.

Examples
  python3 nighthawk_ghz.py selftest
  python3 nighthawk_ghz.py tiles --register nighthawk61r2 --unit 3 --repeats 4 --jobs 8 --out ghz61.jsonl
  python3 nighthawk_ghz.py tiles --register nighthawk61r2 --unit 3 --layer2 dd --blocks 3 --repeats 8 \\
          --jobs 8 --out ghz61_dd.jsonl
  python3 nighthawk_ghz.py chip  --register nighthawk2p57 --shots 4096 --repeats 5 --jobs 8 --out ghz57.jsonl
  python3 nighthawk_ghz.py chip  --register nighthawk2p16 --layer2 unmitigated --out ghz16.jsonl
  python3 nighthawk_ghz.py plot  --in ghz61.jsonl --register nighthawk61r2 --png ghz61.png

Environment: no CUDA. CPU by default; --gpu runs ACE and the reference on the OpenCL device
QRACK_OCL_DEFAULT_DEVICE names (--ocl-sync: QRACK_OCL_SYNC=1, Qrack >= 10.13.2).
"""

import os

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):
    os.environ["PYQRACK_SHARED_LIB_PATH"] = QRACK_LIB_PATH

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))          # nighthawk_qrack.py next to this file (also under python -I)

import nighthawk_qrack as nq                 # noqa: E402

COLUMNS = 10          # nighthawk_qrack's device lattice: 12 rows x 10 columns (the paper writes 10 x 12)
LAYER2_MODES = ("none", "unmitigated", "compensated", "dd")


# ================================================================== register, placement, classes

class Register:
    """A register as ACE places it: layout, couplers, per qubit its ACE simulators (two or more for a
    seam qubit), per coupler its class, lattice cell and sublattice, seam distances."""

    def __init__(self, repo, reg, brep=False):
        if reg != "release" and reg not in nq.REGISTERS:
            nq.load_register_files()
        self.repo, self.name, self.brep = repo, reg, bool(brep)
        self.lay = nq.Layout(repo) if reg == "release" else nq.RegisterLayout(repo, reg)
        self.n = self.lay.n
        ra = nq.shor9_run_args(reg, repo, False, self.brep)          # CPU settings: placement only
        nq.apply_register_preset(ra, self.lay)
        self.map, sids, idx, row = nq.shor9_register_map(self.lay, ra)
        probe = nq.AceEngine(self.lay, self.n, ra)
        self.size, self.lrc, self.lrr, self.torus, self.opts = probe.size, probe.lrc, probe.lrr, probe.torus, probe.opts
        self.widths = list(probe.widths)
        del probe
        self.idx = list(idx)
        self.sids = [sids[idx[q]] for q in range(self.n)]
        self.seam = {q for q in range(self.n) if len(self.sids[q]) > 1}
        self.couplers = [tuple(c) for c in nq.logical_couplers(self.lay, self.n)]
        self.adj = collections.defaultdict(set)
        for a, b in self.couplers:
            self.adj[a].add(b)
            self.adj[b].add(a)
        self.cell = {q: (p // COLUMNS, p % COLUMNS) for q, p in enumerate(self.lay.l2p[:self.n])}
        self.sub = [(r + c) % 2 for r, c in (self.cell[q] for q in range(self.n))]      # Lemma 1: 0 = A, 1 = B
        bad = [(a, b) for a, b in self.couplers if self.sub[a] == self.sub[b]]
        if bad:
            raise SystemExit(f"ghz: {reg} is not bipartite under r + c parity (Lemma 1): couplers {bad[:5]}")
        self.seam_dist = self._bfs_dist(self.seam) if self.seam else {q: math.inf for q in range(self.n)}

    def region(self, q):
        return "seam" if q in self.seam else f"p{next(iter(self.sids[q]))}"

    def cls(self, a, b):
        sa, sb = self.sids[a], self.sids[b]
        return "exact" if len(sa) == 1 and sa == sb else "replica" if sa & sb else "cross"

    def _bfs_dist(self, sources, nodes=None):
        nodes = set(range(self.n)) if nodes is None else set(nodes)
        dist = {q: 0 for q in sources if q in nodes}
        frontier = list(dist)
        while frontier:
            nxt = []
            for u in frontier:
                for v in sorted(self.adj[u]):
                    if v in nodes and v not in dist:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        return {q: dist.get(q, math.inf) for q in nodes}

    def centre(self, nodes=None, adj=None):
        """The node of least eccentricity (ties: not a seam qubit, then the lowest index)."""
        adj = self.adj if adj is None else adj
        nodes = sorted(range(self.n) if nodes is None else nodes)
        best = None
        for s in nodes:
            dist, frontier = {s: 0}, [s]
            while frontier:
                nxt = []
                for u in frontier:
                    for v in adj[u]:
                        if v not in dist:
                            dist[v] = dist[u] + 1
                            nxt.append(v)
                frontier = nxt
            if len(dist) < len(nodes):
                continue
            key = (max(dist.values()), s in self.seam, s)
            if best is None or key < best[0]:
                best = (key, s)
        if best is None:
            raise SystemExit(f"ghz: the couplers of {self.name} do not connect its qubits")
        return best[1]

    def backend(self, gpu=False, stab=True):
        """A fresh QrackAceBackend on the register's placement and ACE options."""
        return nq.ace_backend(self.size, long_range_columns=self.lrc, long_range_rows=self.lrr, is_torus=self.torus,
                              is_gpu=bool(gpu), is_stabilizer_hybrid=bool(stab), **self.opts)

    def describe(self):
        return (f"{self.name} ({self.n} qubits, {len(self.seam)} seam), ACE {self.lrc}x{self.lrr}"
                f"{' torus' if self.torus else ''} on {self.size} sites, simulator widths {self.widths}")


_REGS = {}


def register(repo, reg, brep=False):
    key = (repo, reg, bool(brep))
    if key not in _REGS:
        _REGS[key] = Register(repo, reg, brep)
    return _REGS[key]


# ================================================================== Layer 2 model: Definition 2 (eta = 0)

class Drift:
    """H_D of Definition 2 on a register: J_ij per coupler (j_zz x U(0.75, 1.25), seeded as the reference
    does), dw_q = 0.022 ((q mod 3) - 1), quasi-static eps ~ N(0, sigma_qs^2) per realisation, window T."""

    def __init__(self, couplers, n, sub, j_zz=0.065, sigma_qs=0.015, t_idle=4.4, seed=42):
        self.couplers, self.n, self.sub = [tuple(c) for c in couplers], n, list(sub)
        self.j_zz, self.sigma, self.T, self.seed = float(j_zz), float(sigma_qs), float(t_idle), int(seed)
        self.J = self.j_zz * np.random.default_rng(self.seed).uniform(0.75, 1.25, len(self.couplers))
        self.dw = np.array([0.022 * ((q % 3) - 1.0) for q in range(n)])

    def eps(self, realisation):
        """The quasi-static detunings of one realisation (the same for ACE and the exact reference)."""
        if self.sigma == 0:
            return np.zeros(self.n)
        return np.random.default_rng([self.seed, 7919, int(realisation)]).normal(0.0, self.sigma, self.n)

    def describe(self):
        return dict(j_zz=self.j_zz, sigma_qs=self.sigma, t_idle=self.T, seed=self.seed, eta=0.0)


class Probe:
    """What the drift acts on in one circuit: the units' qubits, couplings inside a unit (gates), couplings
    from a unit qubit to a qubit outside every unit (z-signed local rotations), none between two units."""

    def __init__(self, drift, units):
        self.drift = drift
        owner = {}
        for k, u in enumerate(units):
            for q in u["qubits"]:
                owner[q] = k
        self.qubits = sorted(owner)
        self.inside, self.outside = [], []
        for (a, b), J in zip(drift.couplers, drift.J):
            ia, ib = owner.get(a), owner.get(b)
            if ia is not None and ib is not None:
                if ia != ib:
                    raise RuntimeError(f"ghz: units {ia} and {ib} share coupler {a}-{b}; Layer 2 needs them apart")
                self.inside.append((a, b, float(J)))
            elif ia is not None:
                self.outside.append((a, b, float(J)))
            elif ib is not None:
                self.outside.append((b, a, float(J)))
        self.idle = sorted({b for _, b, _ in self.outside})


def drift_segment(g, probe, dt, eps, z, sign=1.0, with_eps=True):
    """e^{-i sign dt H} on the probe: H = H_D with (dw + eps) or, with_eps False, with dw only (H_hat).
    z: the Z eigenvalue (+1 / -1) of every outside qubit, as the Walsh pulses leave it."""
    d = probe.drift
    for a, b, J in probe.inside:
        g.cx(a, b)
        g.zrot(b, sign * J * dt)
        g.cx(a, b)
    for a, b, J in probe.outside:
        g.zrot(a, sign * J * dt * z[b])
    for q in probe.qubits:
        g.zrot(q, sign * (d.dw[q] + (eps[q] if with_eps else 0.0)) * dt)


def walsh_pulse(g, probe, z, lattice):
    """X on every qubit of sublattice `lattice` (0 = A, 1 = B): the unit qubits get the gate, the outside
    qubits (Z eigenstates throughout) get their sign flipped."""
    sub = probe.drift.sub
    for q in probe.qubits:
        if sub[q] == lattice:
            g.x(q)
    for b in probe.idle:
        if sub[b] == lattice:
            z[b] = -z[b]


def idle_block(g, probe, mode, eps):
    """One idle window V of Definition 4 (G = I) in the given Layer 2 mode."""
    T = probe.drift.T
    z = {b: 1.0 for b in probe.idle}                     # outside qubits start in |0>
    if mode == "unmitigated":
        drift_segment(g, probe, T, eps, z)
    elif mode == "compensated":                          # Definition 6: e^{+iT H_hat} e^{-iT H}
        drift_segment(g, probe, T, eps, z)
        drift_segment(g, probe, T, eps, z, sign=-1.0, with_eps=False)
    elif mode == "dd":                                   # Definition 5, time order U X_B U X_A U X_B U X_A
        for lat in (1, 0, 1, 0):
            drift_segment(g, probe, T / 4, eps, z)
            walsh_pulse(g, probe, z, lat)
        assert all(v == 1.0 for v in z.values())
    elif mode != "none":
        raise ValueError(mode)


# ================================================================== GHZ units: trees and chains

def unit_from_tree(R, root, parent_of, order):
    """A unit (a dict, JSON-ready): qubits, root, tree edges parent -> child in preparation order, each
    edge's coupler class, the tree's depth."""
    depth = {root: 0}
    edges = []
    for c in order:
        if c == root:
            continue
        p = parent_of[c]
        depth[c] = depth[p] + 1
        edges.append([p, c])
    qubits = [root] + [c for _, c in edges]
    return dict(qubits=qubits, root=root, edges=edges, classes=[R.cls(p, c) for p, c in edges],
                depth=max(depth.values()))


def tree_bfs(R, root):
    parent, order, frontier = {root: None}, [root], [root]
    while frontier:
        nxt = []
        for u in frontier:
            for v in sorted(R.adj[u]):
                if v not in parent:
                    parent[v] = u
                    order.append(v)
                    nxt.append(v)
        frontier = nxt
    if len(order) != R.n:
        raise SystemExit(f"ghz: {R.name} is not connected from qubit {root}")
    return unit_from_tree(R, root, parent, order)


def tree_seam(R, root=None):
    """Spanning tree with the fewest CXs through a seam: Kruskal with inside-patch couplers first (ties
    toward the register's centre), then seam couplers, then couplers between simulators with no seam.
    Rooted at the tree's own centre unless root is given."""
    weight = {"exact": 0, "replica": 1, "cross": 1000}
    dc = R._bfs_dist({R.centre()})
    parent = list(range(R.n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    tadj = collections.defaultdict(set)
    for a, b in sorted(R.couplers, key=lambda e: (weight[R.cls(*e)], max(dc[e[0]], dc[e[1]]), e)):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            tadj[a].add(b)
            tadj[b].add(a)
    root = R.centre(adj=tadj) if root is None else root
    par, order, frontier = {root: None}, [root], [root]
    while frontier:
        nxt = []
        for u in frontier:
            for v in sorted(tadj[u]):
                if v not in par:
                    par[v] = u
                    order.append(v)
                    nxt.append(v)
        frontier = nxt
    if len(order) != R.n:
        raise SystemExit(f"ghz: {R.name} is not connected")
    return unit_from_tree(R, root, par, order)


TREES = dict(bfs=lambda R, root: tree_bfs(R, R.centre() if root is None else root),
             seam=lambda R, root: tree_seam(R, root))


def chain_unit(R, qs):
    """A GHZ chain on consecutive coupled qubits qs: root in the middle, CXs outwards along the chain."""
    m = (len(qs) - 1) // 2
    order = [qs[m]] + [qs[i] for k in range(1, len(qs)) for i in (m - k, m + k) if 0 <= i < len(qs)]
    parent = {}
    for i in range(len(qs)):
        if i < m:
            parent[qs[i]] = qs[i + 1]
        elif i > m:
            parent[qs[i]] = qs[i - 1]
    return unit_from_tree(R, qs[m], parent, order)


def tilings(R, k, orients=("rows", "cols"), offsets=None):
    """[(orientation, offset, [units])]: k-qubit chains along every lattice row (or column) of the register,
    split where a site is missing or two neighbours are not coupled, cut into chunks of k from the offset."""
    pos = {q: R.cell[q] for q in range(R.n)}
    out = []
    for ori in orients:
        line = collections.defaultdict(list)
        for q, (r, c) in pos.items():
            line[r if ori == "rows" else c].append(q)
        runs = []
        for key in sorted(line):
            qs = sorted(line[key], key=lambda q: pos[q][1] if ori == "rows" else pos[q][0])
            run = [qs[0]]
            for a, b in zip(qs, qs[1:]):
                step = (pos[b][1] - pos[a][1]) if ori == "rows" else (pos[b][0] - pos[a][0])
                if step == 1 and b in R.adj[a]:
                    run.append(b)
                else:
                    runs.append(run)
                    run = [b]
            runs.append(run)
        for f in (range(k) if offsets is None else offsets):
            units = [chain_unit(R, run[i:i + k]) for run in runs for i in range(f, len(run) - k + 1, k)]
            if units:
                out.append((ori, f, units))
    return out


def spread(R, units):
    """Split units into groups with no coupler between two units of a group (greedy colouring), for Layer 2."""
    sets = [set(u["qubits"]) for u in units]
    touch = [set().union(*(R.adj[q] for q in s)) | s for s in sets]
    groups = []
    for i, s in enumerate(sets):
        for gr in groups:
            if all(not (touch[i] & sets[j]) for j in gr):
                gr.append(i)
                break
        else:
            groups.append([i])
    return [[units[i] for i in gr] for gr in groups]


def unit_info(R, u):
    regions = collections.Counter(R.region(q) for q in u["qubits"])
    d = min(R.seam_dist[q] for q in u["qubits"])
    kind = "seam" if d == 0 else "near seam" if d == 1 else "bulk"
    return dict(regions=dict(regions), cx_classes=dict(collections.Counter(u["classes"])),
                seam_distance=(None if d == math.inf else int(d)), kind=kind,
                cells=[list(R.cell[q]) for q in u["qubits"]])


# ================================================================== per-unit layers

class Gates:
    """Logical-qubit gates on a backend (ACE through its placement, or the exact simulator)."""

    def __init__(self, sim, mapq):
        self.sim, self.q = sim, mapq

    def h(self, q):
        self.sim.h(self.q[q])

    def x(self, q):
        self.sim.x(self.q[q])

    def z(self, q):
        self.sim.z(self.q[q])

    def s(self, q):
        self.sim.s(self.q[q])

    def sdg(self, q):
        self.sim.adjs(self.q[q])

    def u(self, q, th, ph, lm):
        self.sim.u(self.q[q], th, ph, lm)

    def zrot(self, q, phi):
        """e^{-i phi Z} up to a global phase: diag(1, e^{2 i phi})."""
        self.sim.u(self.q[q], 0.0, 0.0, 2.0 * phi)

    def cx(self, c, t):
        self.sim.mcx([self.q[c]], self.q[t])


class Layer:
    """Base per-unit layer: nothing. A layer acts on one unit at three points of its preparation."""
    name = "none"

    def pre(self, g, unit):
        pass

    def after_cx(self, g, unit, p, c, cls):
        pass

    def post(self, g, unit):
        pass


class Fold(Layer):
    """fold:F[:all]: F extra CX pairs after every seam CX (or every CX). Logically the identity; on ACE
    each pair passes the seam twice more, so its error is scaled 2F + 1 times."""

    def __init__(self, F=1, scope="seam"):
        self.F, self.scope = int(F), scope
        self.name = f"fold:{self.F}" + (":all" if scope == "all" else "")

    def after_cx(self, g, unit, p, c, cls):
        if self.scope == "all" or cls != "exact":
            for _ in range(2 * self.F):
                g.cx(p, c)


LAYERS = {
    "none": lambda *args: Layer(),
    "fold": lambda F="1", scope="seam": Fold(int(F), scope),
}


def parse_layers(spec):
    """'none', 'fold:2', 'fold:1:all+...' -> [Layer]."""
    out = []
    for part in (spec or "none").split("+"):
        name, *args = part.strip().split(":")
        if name not in LAYERS:
            raise SystemExit(f"ghz: unknown layer {name!r}; known: {', '.join(sorted(LAYERS))}")
        lay = LAYERS[name](*args)
        if lay.name != "none":
            out.append(lay)
    return out


def layers_name(layers):
    return "+".join(L.name for L in layers) or "none"


def parse_layer2(spec):
    s = (spec or "none").strip().lower()
    s = {"walsh": "dd", "anti-otoc": "dd", "echo": "dd", "compensation": "compensated"}.get(s, s)
    if s not in LAYER2_MODES:
        raise SystemExit(f"ghz: unknown --layer2 {spec!r}; known: {', '.join(LAYER2_MODES)} (walsh = anti-otoc = dd)")
    return s


# ================================================================== circuit: preparation and readout

def prepare(g, unit, layers):
    for L in layers:
        L.pre(g, unit)
    g.h(unit["root"])
    for (p, c), cls in zip(unit["edges"], unit["classes"]):
        g.cx(p, c)
        for L in layers:
            L.after_cx(g, unit, p, c, cls)
    for L in layers:
        L.post(g, unit)


def readout(g, unit, setting):
    """Basis change before a Z measurement of every qubit.
      z     none (populations P0, P1)
      qM    MQC (Lemma 4): M(phi) = cos(phi) X + sin(phi) Y on every qubit, phi = pi M / (k + 1),
            M = 0 .. 2k + 1: H Rz(-phi) = u(pi/2, 0, pi - phi). q0 is the X basis.
      xrA   phase A pi/2 on the root (I, S-dagger, Z, S), then H on all: Clifford, any size"""
    if setting == "z":
        return
    k = len(unit["qubits"])
    if setting.startswith("q"):
        phi = int(setting[1:]) * math.pi / (k + 1)
        for q in unit["qubits"]:
            g.u(q, math.pi / 2, 0.0, math.pi - phi)
        return
    if setting.startswith("xr"):
        a = int(setting[2:]) % 4
        r = unit["root"]
        (lambda: None, lambda: g.sdg(r), lambda: g.z(r), lambda: g.s(r))[a]()
        for q in unit["qubits"]:
            g.h(q)
        return
    raise ValueError(setting)


def settings_for(k, estimator, mqc_max):
    """Readout settings for a k-qubit unit: mqc (z, q0 .. q(2k+1)), witness (z, xr0 .. xr3), or both."""
    est = estimator if estimator != "auto" else ("mqc" if k <= mqc_max else "witness")
    s = ["z"]
    if est in ("mqc", "both"):
        s += [f"q{m}" for m in range(2 * k + 2)]
    if est in ("witness", "both"):
        s += ["xr0", "xr1", "xr2", "xr3"]
    return s


def resolve_stabilizer(a, mode, settings, layer2):
    """--stabilizer auto: tableau patches for a chip-wide state with Clifford readout and no Layer 2; dense
    otherwise (tiles: QUnit keeps every chain separate, so dense is small)."""
    if a.stabilizer != "auto":
        return a.stabilizer == "on"
    return mode == "chip" and layer2 == "none" and all(s == "z" or s.startswith("xr") or s == "q0" for s in settings)


def ideal_parity(setting, k):
    """The parity every shot of |GHZ_0> gives (+1 / -1), None where it is random."""
    if setting in ("xr0", "q0"):
        return 1
    if setting == "xr2":
        return -1
    if setting == f"q{k + 1}":
        return -1 if k % 2 else 1
    return None


def _task(spec):
    """One circuit: a fresh backend (ACE, or the exact simulator), every unit prepared, the per-unit layers,
    Layer 2 (blocks of idle windows on the probe), the readout setting, `shots` samples of all register
    qubits (bit i = logical qubit i)."""
    R = register(spec["repo"], spec["reg"], spec["brep"])
    layers = parse_layers(spec["layers"])
    if spec["exact"]:
        from pyqrack import QrackSimulator
        sim = QrackSimulator(R.n, is_gpu=spec["gpu"], is_stabilizer_hybrid=spec["stab"])
        g = Gates(sim, list(range(R.n)))
    else:
        sim = R.backend(spec["gpu"], spec["stab"])
        g = Gates(sim, R.idx)
    for u in spec["units"]:
        prepare(g, u, layers)
    l2 = spec.get("layer2") or {}
    if l2.get("mode", "none") != "none":
        drift = Drift(R.couplers, R.n, R.sub, l2["j_zz"], l2["sigma_qs"], l2["t_idle"], l2["seed"])
        probe = Probe(drift, spec["units"])
        eps = drift.eps(spec["realisation"])
        for _ in range(int(l2["blocks"])):
            idle_block(g, probe, l2["mode"], eps)
    for u in spec["units"]:
        readout(g, u, spec["setting"])
    if spec["exact"]:
        shots = exact_shots(sim, R.n, spec["shots"], spec["stab"])
    else:
        try:
            shots = sim.measure_shots(list(R.idx), spec["shots"])
        except RuntimeError as err:
            if spec["stab"]:
                raise RuntimeError(f"ghz: ACE's sampler failed on a stabilizer-hybrid backend ({err}); a forced "
                                   f"seam-replica measurement had probability 0, which a tableau refuses: "
                                   f"rerun with --stabilizer off (dense patches)") from err
            raise
    del sim
    return [int(x) for x in shots]


def exact_shots(sim, n, shots, stab):
    """Samples of the exact simulator. With the stabilizer hybrid every shot is a clone measured qubit by
    qubit, which stays a tableau: QrackSimulator.measure_shots converts a state that is random in the
    measured basis to a dense ket first (2^28 amplitudes took 3 GB at 28 qubits). Dense and n <= 32:
    measure_shots (it returns bits 0-31 only, PyQrack 2.28-2.29)."""
    if not stab and n <= 32:
        return sim.measure_shots(list(range(n)), shots)
    out = []
    for _ in range(shots):
        c = sim.clone()
        out.append(sum(int(c.m(q)) << q for q in range(n)))
        del c
    return out


# ================================================================== estimators (Layer 1a)

def bit_matrix(shots, n):
    """shots (ints, bit i = qubit i) -> uint8 array shots x n."""
    nb = max(1, (n + 7) // 8)
    buf = b"".join(int(x).to_bytes(nb, "little") for x in shots)
    a = np.frombuffer(buf, dtype=np.uint8).reshape(len(shots), nb)
    return np.unpackbits(a, axis=1, bitorder="little")[:, :n]


def parity_mean(b):
    """<(-1)^(sum of bits)> over the shots of a shots x k bit array, and its standard error."""
    p = 1.0 - 2.0 * (b.sum(axis=1) & 1)
    m = float(p.mean())
    return m, math.sqrt(max(0.0, 1.0 - m * m) / len(p))


def estimate(data, k, blocks=1, t_idle=None):
    """Layer 1a numbers of one k-qubit unit from {setting: shots x k bits} (unit order: root first), in the
    reference implementation's names. blocks: L of Lemma 3; t_idle: T for Corollary 4.1 (None: no windows)."""
    out = {}
    d = 2.0 ** k
    if "z" in data:
        b = data["z"]
        S = len(b)
        w = b.sum(axis=1)
        P0, P1 = float(np.mean(w == 0)), float(np.mean(w == k))
        pop = P0 + P1
        out.update(P0=P0, P1=P1, pop=pop, pop_se=math.sqrt(max(pop * (1 - pop), 0.0) / S),
                   hamming=np.bincount(w, minlength=k + 1).tolist())
        lam_total = max(1e-6, (pop - 2 / d) / (1 - 2 / d))                     # Lemma 3
        out["lambda_per_block"] = lam_total ** (1.0 / max(1, int(blocks)))
    par = {s: parity_mean(v) for s, v in data.items() if s != "z"}
    out["parity"] = {s: round(m, 6) for s, (m, _) in par.items()}
    M = 2 * k + 2
    if "pop" in out and all(f"q{m}" in par for m in range(M)):                 # Lemma 4
        phis = math.pi * np.arange(M) / (k + 1)
        sig = np.array([par[f"q{m}"][0] for m in range(M)])
        var = np.array([par[f"q{m}"][1] ** 2 for m in range(M)])
        c = complex(np.mean(sig * np.exp(-1j * k * phis)))
        se_c = math.sqrt(var.sum()) / M                                          # per real component, about
        out.update(c_re=c.real, c_im=c.imag, C_parity=2 * abs(c), Phi_coh=math.atan2(c.imag, c.real),
                   F_GHZ=0.5 * (out["P0"] + out["P1"] + 2 * abs(c)),
                   F0=0.5 * (out["P0"] + out["P1"]) + c.real)                    # fidelity to GHZ_0
        out["F_GHZ_se"] = 0.5 * math.hypot(out["pop_se"], 2 * se_c)
        if t_idle:
            out["sum_dw_estimate"] = -out["Phi_coh"] / (2 * max(1, int(blocks)) * t_idle)   # Corollary 4.1
    x = par.get("q0") or par.get("xr0")
    if x is not None and "pop" in out:                                          # Clifford witness (not in the paper)
        out["x_parity"] = x[0]
        out["F_witness"] = out["pop"] + (1.0 + x[0]) / 2.0 - 1.0
        out["F_witness_se"] = math.hypot(out["pop_se"], x[1] / 2.0)
    if all(f"xr{a}" in par for a in range(4)):
        re_ = (par["xr0"][0] - par["xr2"][0]) / 2.0
        im_ = (par["xr1"][0] - par["xr3"][0]) / 2.0
        out["C_x"] = math.hypot(re_, im_)          # every x / x-bar coherence pair, not only 0..0 / 1..1
        out["phase_x"] = math.atan2(im_, re_)
    return out


def headline(e):
    """The unit's fidelity: F_GHZ (Lemma 4) when measured, else the witness bound."""
    return ("F_GHZ", e["F_GHZ"]) if "F_GHZ" in e else ("F_witness", e.get("F_witness"))


def z_breaks(R, unit, bz):
    """Z-basis map of a unit (bz: shots x unit qubits, unit order): per tree edge and per coupler inside
    the unit the share of shots whose two qubits differ, per qubit the share where it differs from the
    unit's majority (ties: the root's bit)."""
    pos = {q: i for i, q in enumerate(unit["qubits"])}
    k = len(unit["qubits"])
    w = bz.sum(axis=1)
    maj = np.where(2 * w == k, bz[:, 0], (2 * w > k).astype(np.uint8))
    site = {q: float(np.mean(bz[:, pos[q]] != maj)) for q in unit["qubits"]}
    edges = [[p, c, R.cls(p, c), float(np.mean(bz[:, pos[p]] != bz[:, pos[c]]))] for p, c in unit["edges"]]
    qs = set(unit["qubits"])
    cpl = [[a, b, R.cls(a, b), float(np.mean(bz[:, pos[a]] != bz[:, pos[b]]))]
           for a, b in R.couplers if a in qs and b in qs]
    return site, edges, cpl


def score_unit(R, unit, per_rep, k, blocks=1, t_idle=None, maps=True):
    """Pooled (the realisations averaged, as the paper averages rho over mu) and per-realisation numbers."""
    cols = unit["qubits"]
    pooled = {s: np.concatenate([d[s][:, cols] for d in per_rep]) for s in per_rep[0]}
    est = estimate(pooled, k, blocks, t_idle)
    reps = [estimate({s: d[s][:, cols] for s in d}, k, blocks, t_idle) for d in per_rep]
    for key in ("F_GHZ", "F0", "F_witness", "C_parity", "Phi_coh", "pop"):
        vals = [r[key] for r in reps if key in r]
        if len(vals) > 1:
            est[key + "_runs"] = [round(v, 6) for v in vals]
            est[key + "_spread"] = [round(min(vals), 6), round(max(vals), 6)]
    if maps and "z" in pooled:
        site, edges, cpl = z_breaks(R, unit, pooled["z"])
        est.update(site_break={str(q): round(v, 6) for q, v in site.items()},
                   edge_break=[[p, c, cl, round(v, 6)] for p, c, cl, v in edges],
                   coupler_break=[[a, b, cl, round(v, 6)] for a, b, cl, v in cpl])
    return est


def check_exact(units, setting, shots_bits, parity_ideal):
    """Shots of the exact reference that are not ideal: all-equal in Z per unit (any diagonal drift keeps
    populations, Lemma 3's proof), and, where the idle unitary is the identity (no Layer 2, Theorem 1's
    echo, oracle compensation without eps), the deterministic MQC / X parities."""
    bad = 0
    for u in units:
        b = shots_bits[:, u["qubits"]]
        k = len(u["qubits"])
        if setting == "z":
            w = b.sum(axis=1)
            bad += int(np.sum((w != 0) & (w != k)))
        elif parity_ideal:
            want = ideal_parity(setting, k)
            if want is not None:
                p = 1 - 2 * (b.sum(axis=1).astype(np.int64) & 1)      # signed: the unsigned sum would wrap
                bad += int(np.sum(p != want))
    return bad


# ================================================================== running

def layer2_spec(a):
    mode = parse_layer2(a.layer2)
    if mode == "none":
        return dict(mode="none", blocks=0)
    return dict(mode=mode, blocks=int(a.blocks), j_zz=float(a.j_zz), sigma_qs=float(a.sigma_qs),
                t_idle=float(a.t_idle), seed=int(a.seed), eta=0.0)


def identity_idle(l2):
    """True when the ideal idle unitary is the identity: no Layer 2, the Walsh echo (Theorem 1), or oracle
    compensation without quasi-static spread (Proposition 1 with eps = 0)."""
    m = l2["mode"]
    return m in ("none", "dd") or (m == "compensated" and l2.get("sigma_qs", 0) == 0)


def run_tasks(a, R, groups, settings, label, l2):
    """Every (group of units, setting, realisation) on ACE and on the exact simulator. groups: [(name,
    [units])]. Returns ({name: [ {setting: bits} per realisation ]} for ACE and for exact, bad, seconds)."""
    reps = max(1, int(a.repeats))
    ex_reps = reps if l2["mode"] != "none" else 1
    base = dict(repo=a.repo, reg=R.name, brep=bool(a.ace_boundary_rep), layers=a.layers, gpu=bool(a.gpu),
                stab=a.stab, layer2=l2)
    calls, labels, keys = [], [], []
    for name, units in groups:
        for s in settings:
            for r in range(reps):
                calls.append((_task, (dict(base, units=units, setting=s, shots=a.shots, exact=False, realisation=r),)))
                labels.append(f"ACE {label} {name} {s} realisation {r + 1}/{reps}")
                keys.append((name, s, r, False))
            if not a.no_exact:
                for r in range(ex_reps):
                    calls.append((_task, (dict(base, units=units, setting=s, shots=a.exact_shots, exact=True,
                                               realisation=r),)))
                    labels.append(f"exact {label} {name} {s} realisation {r + 1}/{ex_reps}")
                    keys.append((name, s, r, True))
    pool, workers = nq.shor_pool(a.jobs, len(calls))
    every = float(a.progress or 0)
    if every:
        print(f"# ghz plan: {len(calls)} circuits ({len(groups)} groups x {len(settings)} settings x {reps} "
              f"ACE realisations" + (f" + exact x {ex_reps}" if not a.no_exact else "") + f"), {workers} process(es)",
              flush=True)
    t0 = time.time()
    try:
        res = nq.shor_pool_map(pool, calls, labels, every, what="ghz")
    finally:
        if pool is not None:
            pool.shutdown()
    data = {name: [dict() for _ in range(reps)] for name, _ in groups}
    exact = {name: [dict() for _ in range(ex_reps)] for name, _ in groups}
    unit_of = dict(groups)
    bad = 0
    pid = identity_idle(l2)
    for (name, s, r, ex), shots in zip(keys, res):
        bits = bit_matrix(shots, R.n)
        if ex:
            bad += check_exact(unit_of[name], s, bits, pid)
            exact[name][r][s] = bits
        else:
            data[name][r][s] = bits
    return data, (None if a.no_exact else exact), (None if a.no_exact else bad), time.time() - t0


def header(a, R, what, l2):
    stab = "stabilizer hybrid" if a.stab else "dense"
    dev = ("OpenCL" + (" (QRACK_OCL_SYNC=1)" if os.environ.get("QRACK_OCL_SYNC") == "1" else "")) if a.gpu else "CPU"
    lay2 = ("Layer 2 none" if l2["mode"] == "none" else
            f"Layer 2 {l2['mode']}: {l2['blocks']} blocks, T {l2['t_idle']}, J {l2['j_zz']} x U(0.75, 1.25), "
            f"sigma_qs {l2['sigma_qs']}, seed {l2['seed']}, eta 0")
    return (f"{what} on {R.describe()}; {a.shots} shots x {a.repeats} realisation(s) per setting, {lay2}, "
            f"layers {a.layers or 'none'}, {stab}, {dev}"
            + (", boundary repetition code" if a.ace_boundary_rep else "") + f"; {nq.qrack_versions()}")


def base_record(a, R, l2):
    return dict(register=R.name, n=R.n, shots=a.shots, repeats=a.repeats, layers=layers_name(parse_layers(a.layers)),
                layer2=l2, estimator=a.estimator, stabilizer=bool(a.stab), boundary_rep=bool(a.ace_boundary_rep),
                device="gpu" if a.gpu else "cpu", ocl_sync=os.environ.get("QRACK_OCL_SYNC") == "1",
                qrack=nq.qrack_versions(), ace=dict(lrc=R.lrc, lrr=R.lrr, torus=bool(R.torus), width=R.size,
                                                   widths=R.widths))


def exact_fields(R, unit, per_rep, k, l2):
    """The exact reference's Layer 1a numbers for a unit, prefixed exact_ (no maps)."""
    if not per_rep:
        return {}
    e = score_unit(R, unit, per_rep, k, max(1, l2["blocks"]), l2.get("t_idle") if l2["mode"] != "none" else None,
                   maps=False)
    keep = ("P0", "P1", "C_parity", "Phi_coh", "F_GHZ", "F0", "lambda_per_block", "sum_dw_estimate", "F_witness",
            "x_parity", "C_x", "phase_x")
    return {"exact_" + key: e[key] for key in keep if key in e}


def write_records(a, R, recs, bad):
    if not a.out or bad:
        return
    when = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(a.out, "a") as fh:
        fh.write(json.dumps(dict(test="ghz_map", register=R.name, when=when, seam=sorted(R.seam), sublattice=R.sub,
                                 seam_distance={str(q): (None if d == math.inf else int(d))
                                                for q, d in R.seam_dist.items()}, **R.map)) + "\n")
        for r in recs:
            fh.write(json.dumps(dict(r, when=when)) + "\n")
    print(f"  appended {len(recs)} records and the register map to {a.out}")


def stop_on_bad(a, bad):
    if bad:
        msg = (f"ghz: the exact reference gave {bad} non-ideal shots, so this run's numbers are not recorded")
        if a.gpu:
            msg += "; on OpenCL use Qrack >= 10.13.2 with --ocl-sync, or drop --gpu (nighthawk_qrack README 5.3)"
        raise SystemExit(msg)


def fmt(v, nd=3):
    return "  -  " if v is None else f"{v:.{nd}f}"


def ascii_map(R, value, nd=2):
    """The register on its lattice, one cell per site: the value, '*' after a seam qubit, '.' for no site."""
    cells = {R.cell[q]: q for q in range(R.n)}
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    w = nd + 3
    lines = ["      " + "".join(f"c{c:<{w}}" for c in range(min(cols), max(cols) + 1))]
    for r in range(min(rows), max(rows) + 1):
        s = f"  r{r:<3}"
        for c in range(min(cols), max(cols) + 1):
            q = cells.get((r, c))
            if q is None:
                s += " " * (w // 2) + "." + " " * (w - w // 2)
            else:
                v = value.get(q)
                s += (f"{v:.{nd}f}" if v is not None else "-" * (nd + 2)) + ("*" if q in R.seam else " ")
        lines.append(s)
    return "\n".join(lines)


def print_unit_line(name, e, ex):
    key, F = headline(e)
    s = (f"{name:<6} P0+P1 {fmt(e.get('pop'))}  ")
    if "F_GHZ" in e:
        s += (f"C {fmt(e['C_parity'])}  Phi {fmt(e['Phi_coh'], 2):>5}  F_GHZ {e['F_GHZ']:.3f} +/- {e['F_GHZ_se']:.3f}  "
              f"F0 {e['F0']:.3f}  lambda/block {fmt(e.get('lambda_per_block'), 4)}")
        if "sum_dw_estimate" in e:
            s += f"  sum dw est {e['sum_dw_estimate']:+.4f}"
    if "F_witness" in e:
        s += f"  F >= {e['F_witness']:.3f} (witness)"
    if "C_x" in e:
        s += f"  C_x {e['C_x']:.3f} phase_x {e['phase_x']:+.2f}"
    if ex:
        s += ("   | exact: " + "  ".join(f"{k_[6:]} {v:.3f}" for k_, v in ex.items()
                                         if k_ in ("exact_F_GHZ", "exact_Phi_coh", "exact_F_witness", "exact_P0")))
    print(s)


# ================================================================== chip

def cmd_chip(a):
    R = register(a.repo, a.register, a.ace_boundary_rep)
    l2 = layer2_spec(a)
    names = [t.strip() for t in a.trees.split(",") if t.strip()]
    for t in names:
        if t not in TREES:
            raise SystemExit(f"ghz: unknown tree {t!r}; known: {', '.join(TREES)}")
    trees = [(t, TREES[t](R, a.root)) for t in names]
    settings = settings_for(R.n, a.estimator, a.mqc_max)
    a.stab = resolve_stabilizer(a, "chip", settings, l2["mode"])
    if not a.stab and R.n > 30 and not a.force:
        raise SystemExit(f"ghz: {R.n} qubits with dense patches (MQC readout or Layer 2) do not fit: the GHZ state spans "
                         f"each patch ({R.widths}); use a register of <= 30 qubits, the witness readout without "
                         f"Layer 2, or --force")
    print(header(a, R, f"GHZ over all {R.n} qubits", l2))
    print("tree  root  depth  CX exact/replica/cross  readout settings: " + " ".join(settings))
    for t, u in trees:
        c = collections.Counter(u["classes"])
        print(f"{t:<5} {u['root']:>4}  {u['depth']:>5}  {c.get('exact', 0):>5}/{c.get('replica', 0)}/{c.get('cross', 0)}")
    data, exact, bad, secs = run_tasks(a, R, [(t, [u]) for t, u in trees], settings, "chip", l2)
    stop_on_bad(a, bad)
    L, T = max(1, l2["blocks"]), (l2.get("t_idle") if l2["mode"] != "none" else None)
    print()
    recs = []
    for t, u in trees:
        e = score_unit(R, u, data[t], R.n, L, T)
        ex = exact_fields(R, u, exact[t] if exact else None, R.n, l2)
        print_unit_line(t, e, ex)
        recs.append(dict(base_record(a, R, l2), test="ghz", mode="chip", tree=t, **u,
                         cx_classes=dict(collections.Counter(u["classes"])), **e, **ex))
    if l2["mode"] != "none":
        d = Drift(R.couplers, R.n, R.sub, l2["j_zz"], l2["sigma_qs"], l2["t_idle"], l2["seed"])
        print(f"  Corollary 4.1 reference: sum dw over the register {float(d.dw.sum()):+.4f} (eps averages to 0)")
    for t, u in trees:
        e = next(r for r in recs if r["tree"] == t)
        worst = sorted(e.get("edge_break", []), key=lambda x: -x[3])[:8]
        print(f"\n{t}: tree edges that break the Z pattern most (parent child class rate): "
              + ", ".join(f"{p}-{c} {cl} {v:.3f}" for p, c, cl, v in worst))
        by = collections.defaultdict(list)
        for p, c, cl, v in e.get("edge_break", []):
            by[cl].append(v)
        print("  mean break rate by edge class: " + ", ".join(f"{cl} {np.mean(v):.4f} ({len(v)})"
                                                         for cl, v in sorted(by.items())))
        print(f"  share of shots where each site differs from the majority ({t} tree; * seam qubit):")
        print(ascii_map(R, {int(q): v for q, v in e.get("site_break", {}).items()}, 3))
    if bad is not None:
        print(f"\nexact reference (QrackSimulator, {R.n} qubits, {a.exact_shots} shots per setting and realisation): "
              + ("every checked shot ideal" if bad == 0 else f"{bad} shots NOT ideal"))
    print(f"seconds: {secs:.1f}")
    write_records(a, R, recs, bad)


# ================================================================== tiles

def cmd_tiles(a):
    R = register(a.repo, a.register, a.ace_boundary_rep)
    l2 = layer2_spec(a)
    k = int(a.unit)
    if k < 2:
        raise SystemExit("ghz: --unit must be 2 or more")
    offs = None if a.offsets in (None, "all") else [int(x) for x in a.offsets.split(",")]
    tl = tilings(R, k, tuple(x.strip() for x in a.orient.split(",")), offs)
    if not tl:
        raise SystemExit(f"ghz: no chain of {k} coupled qubits along a lattice line of {R.name}")
    settings = settings_for(k, a.estimator, a.mqc_max)
    a.stab = resolve_stabilizer(a, "tiles", settings, l2["mode"])
    groups = []
    for ori, f, units in tl:
        parts = spread(R, units) if l2["mode"] != "none" else [units]
        for j, part in enumerate(parts):
            groups.append((f"{ori}{f}" + (f".{j}" if len(parts) > 1 else ""), part))
    nunits = sum(len(u) for _, u in groups)
    print(header(a, R, f"{k}-qubit GHZ chains tiled", l2))
    print(f"{len(groups)} circuit groups ({', '.join(f'{g} {len(u)}' for g, u in groups)} units), {nunits} units; "
          f"readout settings: {' '.join(settings)}")
    data, exact, bad, secs = run_tasks(a, R, groups, settings, "tiles", l2)
    stop_on_bad(a, bad)
    L, T = max(1, l2["blocks"]), (l2.get("t_idle") if l2["mode"] != "none" else None)
    recs = []
    for g, units in groups:
        for j, u in enumerate(units):
            e = score_unit(R, u, data[g], k, L, T)
            ex = exact_fields(R, u, exact[g] if exact else None, k, l2)
            recs.append(dict(base_record(a, R, l2), test="ghz", mode="tiles", unit_size=k, tiling=g, unit=j, **u,
                             **unit_info(R, u), **e, **ex))
    key = headline(recs[0])[0]
    print(f"\nby seam distance ({key}; 0 = the unit holds a seam qubit): units, mean, min, max"
          + (", exact reference mean" if exact else ""))
    by = collections.defaultdict(list)
    for r in recs:
        by[r["seam_distance"]].append(r)
    for d in sorted(by, key=lambda x: (x is None, x)):
        v = [r[key] for r in by[d]]
        xv = [r.get("exact_" + key) for r in by[d] if r.get("exact_" + key) is not None]
        print(f"  {str(d):>4}  {len(v):>4}  {np.mean(v):.3f}  {min(v):.3f}  {max(v):.3f}"
              + (f"   {np.mean(xv):.3f}" if xv else ""))
    if "Phi_coh" in recs[0]:
        ph = [r["Phi_coh"] for r in recs if r["seam_distance"] not in (0, None)]
        xph = [r["exact_Phi_coh"] for r in recs if r["seam_distance"] not in (0, None) and "exact_Phi_coh" in r]
        if ph:
            print(f"  Phi_coh off the seam: ACE mean {np.mean(ph):+.3f}, spread {np.std(ph):.3f}"
                  + (f"; exact mean {np.mean(xph):+.3f}" if xph else ""))
    print(f"\nlowest {min(10, len(recs))} units ({key}, regions, CX exact/replica/cross):")
    for r in sorted(recs, key=lambda r: r[key])[:10]:
        c = r["cx_classes"]
        print(f"  {r['tiling']:>8} #{r['unit']:<3} qubits {r['qubits']}  {r[key]:.3f}  "
              f"{', '.join(f'{x} {y}' for x, y in sorted(r['regions'].items()))}  "
              f"{c.get('exact', 0)}/{c.get('replica', 0)}/{c.get('cross', 0)}")
    per_q = collections.defaultdict(list)
    for r in recs:
        for q in r["qubits"]:
            per_q[q].append(r[key])
    print(f"\nper site: mean {key} of the units holding it (* seam qubit):")
    print(ascii_map(R, {q: float(np.mean(v)) for q, v in per_q.items()}, 2))
    if bad is not None:
        print(f"\nexact reference (QrackSimulator, {R.n} qubits, {a.exact_shots} shots per setting, group and "
              f"realisation): " + ("every checked shot ideal" if bad == 0 else f"{bad} shots NOT ideal"))
    print(f"seconds: {secs:.1f}  (shot noise alone on {key}: about +/- "
          f"{np.mean([r.get(key + '_se', 0) for r in recs]):.3f} per unit)")
    write_records(a, R, recs, bad)


# ================================================================== plot

# one blue ramp, light -> dark, for magnitude (the reference palette's sequential blue, steps 100-700)
LOSS_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8984"
CLASS_COL = {"exact": "#b9b8b2", "replica": "#eb6834", "cross": "#8f4fd1"}    # inside / through a seam / no seam


def read_records(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


def _l2name(r):
    l2 = r.get("layer2") or {}
    return l2.get("mode", "none") if isinstance(l2, dict) else str(l2)


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    recs = read_records(a.inp)
    maps = [r for r in recs if r.get("test") == "ghz_map" and (not a.register or r["register"] == a.register)]
    if not maps:
        raise SystemExit(f"ghz plot: no ghz_map record{' for ' + a.register if a.register else ''} in {a.inp}")
    mp = maps[-1]
    reg = mp["register"]
    ghz = [r for r in recs if r.get("test") == "ghz" and r["register"] == reg
           and (not a.layers or r.get("layers") == a.layers) and (not a.layer2 or _l2name(r) == parse_layer2(a.layer2))]
    if not ghz:
        raise SystemExit(f"ghz plot: no ghz records for {reg}")
    tiles = [r for r in ghz if r["mode"] == "tiles"]
    chips = {}
    for r in ghz:
        if r["mode"] == "chip":
            chips[r["tree"]] = r                               # the latest run per tree
    sites = {s[0]: s for s in mp["sites"]}
    seam = set(mp.get("seam", []))
    xy = {q: (s[2], -s[1]) for q, s in sites.items()}
    cmap = LinearSegmentedColormap.from_list("loss", LOSS_RAMP)

    panels = []                                                # (title, {qubit: loss}, colour-bar label, tree edges)
    if tiles:
        key = headline(tiles[0])[0]
        per_q = collections.defaultdict(list)
        for r in tiles:
            for q in r["qubits"]:
                per_q[q].append(r[key])
        panels.append((f"{tiles[0]['unit_size']}-qubit GHZ chains, Layer 2 {_l2name(tiles[0])}: loss per site\n"
                       f"1 - mean {key} of the units holding it", {q: 1.0 - float(np.mean(v)) for q, v in per_q.items()},
                       f"1 - {key}", None))
    for t, r in sorted(chips.items()):
        k_, F = headline(r)
        panels.append((f"{r['n']}-qubit GHZ, {t} tree (thick: its CXs), {k_} {F:.3f}"
                       f"\nshare of Z shots where a site differs from the majority",
                       {int(q): v for q, v in r.get("site_break", {}).items()}, "break rate",
                       {tuple(sorted(e)) for e in r["edges"]}))
    npan = len(panels) + (1 if tiles else 0)
    ncol = min(2, npan)
    nrow = -(-npan // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 5.0 * nrow), squeeze=False)
    axes = list(axes.flat)
    for ax in axes[npan:]:
        ax.axis("off")
    if tiles:                                                  # the box plot goes last
        axes = axes[:len(panels)] + [axes[npan - 1]]
    vmax = max([max(v.values() or [0]) for _, v, _, _ in panels] + [0.05])
    for ax, (title, val, clabel, tree) in zip(axes, panels):
        segs, cols, lws = [], [], []
        for a_, b_, cl in mp["couplers"]:
            segs.append([xy[a_], xy[b_]])
            cols.append(CLASS_COL.get(cl, INK_MUTED))
            on_tree = tree is None or tuple(sorted((a_, b_))) in tree
            lws.append((3.0 if cl != "exact" else 2.0) if on_tree else 0.8)
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, zorder=1, capstyle="round"))
        qs = sorted(val)
        sc = ax.scatter([xy[q][0] for q in qs], [xy[q][1] for q in qs], c=[val[q] for q in qs], cmap=cmap,
                        vmin=0.0, vmax=vmax, s=300, zorder=2,
                        edgecolors=[INK if q in seam else "#c9c8c2" for q in qs],
                        linewidths=[2.4 if q in seam else 1.0 for q in qs])
        for q in qs:
            ax.text(xy[q][0], xy[q][1], f"{max(val[q], 0.0):.2f}", ha="center", va="center", fontsize=6.5, zorder=3,
                    color="#fcfcfb" if val[q] / vmax > 0.5 else INK)
        cb = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.02)
        cb.set_label(clabel, color=INK_2, fontsize=8)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, colors=INK_2)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_aspect("equal")
        ax.margins(0.08)
        ax.axis("off")
    axes[0].legend(handles=[Line2D([], [], color=CLASS_COL["exact"], lw=2, label="coupler inside one simulator"),
                            Line2D([], [], color=CLASS_COL["replica"], lw=3, label="coupler through a seam"),
                            Line2D([], [], marker="o", ls="", markersize=9, markerfacecolor="#fcfcfb",
                                   markeredgecolor=INK, markeredgewidth=2, label="seam qubit")],
                   loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=7.5)
    if tiles:
        ax = axes[-1]
        key = headline(tiles[0])[0]
        by = collections.defaultdict(list)
        for r in tiles:
            by[r["seam_distance"]].append(r[key])
        ds = sorted(d for d in by if d is not None)
        bp = ax.boxplot([by[d] for d in ds], positions=range(len(ds)), widths=0.45, patch_artist=True,
                        medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=INK_2),
                        capprops=dict(color=INK_2), flierprops=dict(marker="o", markersize=4,
                                                                   markerfacecolor=INK_MUTED, markeredgecolor="none"))
        for b in bp["boxes"]:
            b.set(facecolor="#9ec5f4", edgecolor=INK_2)
        ax.set_xticks(range(len(ds)), [f"{d}\n({len(by[d])} units)" for d in ds], fontsize=8, color=INK_2)
        ax.set_xlabel("unit's distance to the nearest seam qubit (couplers)", color=INK_2, fontsize=8.5)
        ax.set_ylabel(key, color=INK_2, fontsize=8.5)
        ax.axhline(0.5, color=INK_MUTED, lw=0.8, ls="--")
        ax.text(len(ds) - 0.5, 0.51, "0.5: genuine entanglement above", ha="right", va="bottom", fontsize=7,
                color=INK_MUTED)
        ax.set_ylim(min(0.0, min(min(v) for v in by.values()) - 0.05), 1.05)
        ax.set_title(f"{key} by seam distance", fontsize=8.5, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(INK_MUTED)
        ax.tick_params(colors=INK_2, labelsize=8)
    fig.suptitle(f"{reg} on ACE ({mp['ace']['lrc']}x{mp['ace']['lrr']}), Layer 2 {_l2name(ghz[-1])}, per-unit layers "
                 f"{ghz[-1].get('layers', 'none')}; loss below 0 (F_GHZ above 1: shot noise) shown as 0; "
                 f"{ghz[-1].get('qrack', '')}", fontsize=9, color=INK)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    out = a.png or f"ghz_{reg}.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


# ================================================================== selftest

def _np_ghz_rho(k, theta=0.0, p_flip=0.0, p_deph=0.0):
    """numpy density matrix: GHZ with phase theta, then an independent bit flip on qubit 0 with p_flip and a
    global dephasing of the 0..0 / 1..1 coherence by (1 - p_deph)."""
    d = 2 ** k
    psi = np.zeros(d, complex)
    psi[0], psi[-1] = 1 / math.sqrt(2), np.exp(1j * theta) / math.sqrt(2)
    rho = np.outer(psi, psi.conj())
    rho[0, -1] *= (1 - p_deph)
    rho[-1, 0] *= (1 - p_deph)
    X0 = np.kron(np.array([[0, 1], [1, 0]]), np.eye(d // 2))
    return (1 - p_flip) * rho + p_flip * X0 @ rho @ X0


def _np_readout_probs(rho, k, setting):
    """Outcome probabilities (index bit i = qubit i) after readout(setting) on a numpy rho (qubit 0 = root,
    the most significant factor of the kron products)."""
    I2, H = np.eye(2), np.array([[1, 1], [1, -1]]) / math.sqrt(2)

    def u(th, ph, lm):
        return np.array([[math.cos(th / 2), -np.exp(1j * lm) * math.sin(th / 2)],
                         [np.exp(1j * ph) * math.sin(th / 2), np.exp(1j * (ph + lm)) * math.cos(th / 2)]])
    if setting == "z":
        ops = [I2] * k
    elif setting.startswith("xr"):
        ph = [np.eye(2), np.diag([1, -1j]), np.diag([1, -1]), np.diag([1, 1j])][int(setting[2:])]
        ops = [H @ ph] + [H] * (k - 1)
    else:
        phi = int(setting[1:]) * math.pi / (k + 1)
        ops = [u(math.pi / 2, 0, math.pi - phi)] * k
    U = ops[0]
    for o in ops[1:]:
        U = np.kron(U, o)
    pr = np.real(np.diag(U @ rho @ U.conj().T))
    out = np.zeros(2 ** k)
    for idx_ in range(2 ** k):                          # kron index: qubit 0 is the high bit; ours: bit 0
        b = [(idx_ >> (k - 1 - i)) & 1 for i in range(k)]
        out[sum(bit << i for i, bit in enumerate(b))] = pr[idx_]
    return out


def _bits_from_probs(p, k, S=0):
    """An exact 'bit matrix' stand-in: every outcome repeated in proportion, for the estimators' arithmetic."""
    S = S or 400_000
    counts = np.round(p * S).astype(int)
    rows = np.repeat(np.arange(len(p)), counts)
    return ((rows[:, None] >> np.arange(k)) & 1).astype(np.uint8)


class _Lattice:
    """A small rows x cols grid for the exact-simulator checks (no ACE, no release needed)."""

    def __init__(self, rows, cols):
        self.n = rows * cols
        self.couplers = [(r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1)] + \
                        [(r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols)]
        self.sub = [(q // cols + q % cols) % 2 for q in range(self.n)]


def _ket(sim, n):
    return np.array(sim.out_ket(), dtype=complex)[: 2 ** n]


def cmd_selftest(a):
    from pyqrack import QrackSimulator
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    print("[1] Lemma 4 (MQC) and the witness against numpy density matrices (k = 3, 4, 5)")
    for k in (3, 4, 5):
        for th, pf, pd in ((0.0, 0.0, 0.0), (0.7, 0.0, 0.0), (0.0, 0.1, 0.0), (0.3, 0.05, 0.4)):
            rho = _np_ghz_rho(k, th, pf, pd)
            ghz = np.zeros(2 ** k)
            ghz[0] = ghz[-1] = 1 / math.sqrt(2)
            F0_true = float(np.real(ghz @ rho @ ghz))
            thetas = np.linspace(-math.pi, math.pi, 2001)
            Fmax = max(0.5 * float(np.real(rho[0, 0] + rho[-1, -1] + 2 * np.exp(1j * t) * rho[0, -1])) for t in thetas)
            data = {s: _bits_from_probs(_np_readout_probs(rho, k, s), k) for s in settings_for(k, "both", 99)}
            e = estimate(data, k)
            c = rho[0, -1]
            check(f"k={k} theta={th} flip={pf} dephase={pd}",
                  abs(e["c_re"] - c.real) < 2e-3 and abs(e["c_im"] - c.imag) < 2e-3
                  and abs(e["F_GHZ"] - Fmax) < 2e-3 and abs(e["F0"] - F0_true) < 2e-3
                  and e["F_witness"] <= F0_true + 2e-3,
                  f"c {e['c_re']:+.4f}{e['c_im']:+.4f}i / {c.real:+.4f}{c.imag:+.4f}i, F_GHZ {e['F_GHZ']:.4f} / "
                  f"max_theta {Fmax:.4f}, F0 {e['F0']:.4f} / {F0_true:.4f}, witness {e['F_witness']:.4f}")

    print("[2] Lemma 4 on any rho: the DFT at n returns rho_{0..0,1..1} (random states, k = 4)")
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(20):
        A = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
        rho = A @ A.conj().T
        rho /= np.trace(rho)
        data = {s: _bits_from_probs(_np_readout_probs(rho, 4, s), 4) for s in settings_for(4, "mqc", 99)}
        e = estimate(data, 4)
        worst = max(worst, abs(complex(e["c_re"], e["c_im"]) - rho[0, -1]))
    check("20 random density matrices", worst < 2e-3, f"largest |c_hat - c| {worst:.1e}")

    print("[3] Lemma 3: lambda from the populations of a depolarised GHZ (k = 3, L = 3)")
    for lam in (0.94, 0.7):
        d = 8
        rho = _np_ghz_rho(3)
        for _ in range(3):
            rho = lam * rho + (1 - lam) * np.eye(d) / d
        e = estimate({"z": _bits_from_probs(_np_readout_probs(rho, 3, "z"), 3)}, 3, blocks=3)
        check(f"lambda {lam}", abs(e["lambda_per_block"] - lam) < 2e-3, f"estimate {e['lambda_per_block']:.4f}")

    print("[4] readout gates on Qrack against numpy (k = 3, GHZ with phase 0.5)")
    k = 3
    rho = _np_ghz_rho(k, 0.5)
    for s in settings_for(k, "both", 99):
        sim = QrackSimulator(k, is_gpu=False)
        g = Gates(sim, list(range(k)))
        u = dict(qubits=[0, 1, 2], root=0, edges=[[0, 1], [0, 2]], classes=["exact", "exact"], depth=1)
        prepare(g, u, [])
        sim.u(0, 0.0, 0.0, 0.5)                      # phase 0.5 on |1..1>
        readout(g, u, s)
        pq = np.array([sim.prob_perm([0, 1, 2], [bool((x >> i) & 1) for i in range(k)]) for x in range(2 ** k)])
        pn = _np_readout_probs(rho, k, s)
        check(f"setting {s}", np.max(np.abs(pq - pn)) < 1e-5, f"max |dp| {np.max(np.abs(pq - pn)):.2e}")

    print("[5] Theorem 1 on the exact simulator: W_T(H_D) = I (2 x 3 grid, random state, random J, dw, eps)")
    lat = _Lattice(2, 3)
    for seed in (1, 2, 3):
        drift = Drift(lat.couplers, lat.n, lat.sub, j_zz=0.3, sigma_qs=0.2, t_idle=4.4, seed=seed)
        unit = dict(qubits=list(range(lat.n)))
        probe = Probe(drift, [unit])
        r_ = np.random.default_rng(seed)
        sim = QrackSimulator(lat.n, is_gpu=False)
        g = Gates(sim, list(range(lat.n)))
        for q in range(lat.n):
            g.u(q, *r_.uniform(0, math.pi, 3))
        for a_, b_ in lat.couplers[:4]:
            g.cx(a_, b_)
        before = _ket(sim, lat.n)
        idle_block(g, probe, "dd", drift.eps(seed))
        after = _ket(sim, lat.n)
        fid = abs(np.vdot(before, after)) ** 2
        check(f"seed {seed}", abs(fid - 1) < 1e-6, f"|<psi|W_T|psi>|^2 = {fid:.9f}")
        idle_block(g, probe, "unmitigated", drift.eps(seed))
        check(f"seed {seed}: unmitigated is not the identity (control)", abs(np.vdot(before, _ket(sim, lat.n))) ** 2 < 0.999)

    print("[6] Corollary 2.1 and 4.1 on the exact simulator: a register-wide GHZ under unmitigated drift")
    drift = Drift(lat.couplers, lat.n, lat.sub, j_zz=0.3, sigma_qs=0.0, t_idle=4.4, seed=5)
    unit = dict(qubits=list(range(lat.n)))
    probe = Probe(drift, [unit])
    for with_dw in (False, True):
        if not with_dw:
            drift.dw = np.zeros(lat.n)
        else:                                            # the paper's formula sums to 0 on 2 x 3: shift it
            drift.dw = np.array([0.022 * ((q % 3) - 1.0) + 0.011 for q in range(lat.n)])
        sim = QrackSimulator(lat.n, is_gpu=False)
        g = Gates(sim, list(range(lat.n)))
        g.h(0)
        for q in range(1, lat.n):
            g.cx(0, q)
        L = 3
        for _ in range(L):
            idle_block(g, probe, "unmitigated", drift.eps(0))
        v = _ket(sim, lat.n)
        c = v[0] * np.conj(v[-1])
        want = -2 * L * drift.T * float(drift.dw.sum())
        dphi = (np.angle(c) - want + math.pi) % (2 * math.pi) - math.pi
        check("ZZ only: coherence untouched" if not with_dw else "ZZ + dw: arg c = -2 L T sum dw",
              abs(abs(c) - 0.5) < 1e-6 and abs(dphi) < 1e-6,
              f"|c| {abs(c):.6f}, arg c {np.angle(c):+.6f}, expected {((want + math.pi) % (2 * math.pi)) - math.pi:+.6f}")

    print("[7] the outside-coupling identity: e^{-i phi Z_a Z_b} with b in |0> or |1> = e^{-i phi z_b Z_a}")
    for zb in (0, 1):
        sims = []
        for exact_gate in (True, False):
            sim = QrackSimulator(2, is_gpu=False)
            g = Gates(sim, [0, 1])
            g.u(0, 1.1, 0.3, 0.7)
            if zb:
                g.x(1)
            if exact_gate:
                g.cx(0, 1)
                g.zrot(1, 0.37)
                g.cx(0, 1)
            else:
                g.zrot(0, 0.37 * (1 - 2 * zb))
            sims.append(_ket(sim, 2))
        fid = abs(np.vdot(sims[0], sims[1])) ** 2
        check(f"z_b = {1 - 2 * zb:+d}", abs(fid - 1) < 1e-6, f"overlap {fid:.9f}")      # fp32 Qrack

    print("[8] fold is the identity on the exact simulator")
    sim = QrackSimulator(4, is_gpu=False)
    g = Gates(sim, list(range(4)))
    u = dict(qubits=[1, 0, 2, 3], root=1, edges=[[1, 0], [1, 2], [2, 3]], classes=["exact", "replica", "exact"],
             depth=2)
    prepare(g, u, parse_layers("fold:2:all"))
    p = [sim.prob_perm([0, 1, 2, 3], [bool((x >> i) & 1) for i in range(4)]) for x in range(16)]
    check("fold:2:all keeps the GHZ populations", abs(p[0] - 0.5) < 1e-6 and abs(p[15] - 0.5) < 1e-6)
    print("selftest " + ("passed" if ok else "FAILED"))
    if not ok:
        raise SystemExit(1)


# ================================================================== command line

def make_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo", default=None, help="rcs-nighthawk clone (default: auto-detect, $NIGHTHAWK_REPO)")
    common.add_argument("--register", default="nighthawk61r2",
                        help="release, any register of nighthawk_qrack.py (nighthawk61r2, nighthawk2p45, nighthawk88e, "
                             "...) or one saved by its layoutsearch (fav5p112, ...); its ACE preset placement is used")
    common.add_argument("--shots", type=int, default=4096, help="shots per readout setting and realisation")
    common.add_argument("--repeats", type=int, default=3,
                        help="realisations per setting: a fresh ACE backend and, with Layer 2, a fresh eps draw each "
                             "(the paper's K)")
    common.add_argument("--estimator", choices=["auto", "mqc", "witness", "both"], default="auto",
                        help="auto: Lemma 4's MQC up to --mqc-max qubits, the Clifford witness above")
    common.add_argument("--mqc-max", dest="mqc_max", type=int, default=12,
                        help="largest unit the MQC readout (2k + 2 settings, not Clifford) runs on with --estimator auto")
    common.add_argument("--layer2", default="none",
                        help="idle-window refocusing after the preparation: none, unmitigated, compensated (route A, "
                             "oracle H_hat), dd (route B, Walsh echo: the anti-OTOC layer; also 'walsh', 'anti-otoc')")
    common.add_argument("--blocks", type=int, default=3, help="Layer 2: idle windows L (the paper's GHZ blocks)")
    common.add_argument("--t-idle", dest="t_idle", type=float, default=4.4, help="Layer 2: window T (time units)")
    common.add_argument("--j-zz", dest="j_zz", type=float, default=0.065, help="Layer 2: mean J_ij (rad per time unit)")
    common.add_argument("--sigma-qs", dest="sigma_qs", type=float, default=0.015,
                        help="Layer 2: quasi-static detuning spread (rad per time unit)")
    common.add_argument("--seed", type=int, default=42, help="Layer 2: seed of the J_ij draw and the eps realisations")
    common.add_argument("--layers", default="none", help="per-unit hook: none, fold:F, fold:F:all, joined by '+'")
    common.add_argument("--exact-shots", dest="exact_shots", type=int, default=128,
                        help="shots of the exact reference per setting and realisation")
    common.add_argument("--no-exact", dest="no_exact", action="store_true", help="skip the exact reference")
    common.add_argument("--stabilizer", choices=["auto", "on", "off"], default="auto",
                        help="ACE's patch simulators as stabilizer hybrids. auto: on for chip with Clifford-only readout "
                             "and no Layer 2 (a GHZ across a patch is otherwise 2^width amplitudes), off otherwise")
    common.add_argument("--force", action="store_true", help="chip: run dense patches above 30 qubits anyway")
    common.add_argument("--ace-boundary-rep", dest="ace_boundary_rep", action="store_true",
                        help="ACE's is_boundary_repetition_code (off unless given, as in nighthawk_qrack.py)")
    common.add_argument("--gpu", action="store_true", help="ACE and the reference on OpenCL (QRACK_OCL_DEFAULT_DEVICE)")
    common.add_argument("--ocl-sync", dest="ocl_sync", action="store_true", help="with --gpu: QRACK_OCL_SYNC=1")
    common.add_argument("--jobs", type=int, default=1, help="worker processes, one circuit each (0 = every core)")
    common.add_argument("--progress", type=float, default=30.0, help="seconds between heartbeats (0 = off)")
    common.add_argument("--out", default=None, help="append JSONL records (ghz_map once, then ghz per state)")

    c = sub.add_parser("chip", parents=[common], help="one GHZ state over the whole register")
    c.add_argument("--trees", default="bfs,seam", help="comma list of bfs, seam")
    c.add_argument("--root", type=int, default=None, help="root qubit (default: the tree's centre)")

    t = sub.add_parser("tiles", parents=[common], help="k-qubit GHZ chains tiled over the register")
    t.add_argument("--unit", type=int, default=3, help="qubits per GHZ chain")
    t.add_argument("--orient", default="rows,cols", help="rows, cols or both")
    t.add_argument("--offsets", default="all", help="chain offsets along a line: all, or a comma list")

    p = sub.add_parser("plot", help="the map of a records file")
    p.add_argument("--in", dest="inp", required=True, help="JSONL written by chip / tiles --out")
    p.add_argument("--register", default=None, help="which register (default: the last map in the file)")
    p.add_argument("--layers", default=None, help="only records with this per-unit layer stack")
    p.add_argument("--layer2", default=None, help="only records with this Layer 2 mode")
    p.add_argument("--png", default=None, help="output image (default ghz_<register>.png)")

    sub.add_parser("selftest", help="paper identities against numpy and the exact simulator")
    return ap


def main():
    a = make_parser().parse_args()
    if getattr(a, "ocl_sync", False):
        os.environ["QRACK_OCL_SYNC"] = "1"
    if a.cmd in ("chip", "tiles"):
        nq.ensure_isolated(a)
    dict(chip=cmd_chip, tiles=cmd_tiles, plot=cmd_plot, selftest=cmd_selftest)[a.cmd](a)


if __name__ == "__main__":
    main()
