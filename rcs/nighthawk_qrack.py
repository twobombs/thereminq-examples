#!/usr/bin/env python3
"""
nighthawk_qrack.py -- PyQrack audit of the fidelity estimators behind arXiv:2609.28657
(BlueQubit, random-circuit sampling on IBM Nighthawk r2 / ibm_phoenix), without a QPU.

Subcommands
  seamgap   data only (numpy): CZ counts of every released circuit family, the fit read
            at depth vs at gate count, ln F regression on cycles and CZ count, and the
            per-cycle error budget.  Run against a clone of BlueQubitDev/rcs-nighthawk.
  run       statevector emulation on nested windows of the real 61-qubit coupler graph
            (--layout data/layout.json) or of a synthetic grid (--grid 6x6).
            One size (--sizes 36) is a single echo run; a range (--sizes 27-36) is a
            scaling sweep.  Resumable JSONL; --summarize prints the tables.
  selftest  the validation checks (noiseless exactness, Loschmidt echo vs explicit
            overlap, twirl control, analytic vs hybrid vs trajectory stochastic noise,
            native compilation at the poles, ACE).

Estimators per (window size N, depth d, instance)
  fwd_full     noisy U_full(d), then IDEAL U_full(d)^dag, read P(input string)
               = |<psi_ideal|psi_noisy>|^2 on ONE statevector: the truth for the circuit
               that would be sampled
  fwd_pp       same for the pseudo-patched (seam-thinned) circuit
  mirror       noisy U_pp(d/2), NOISY U_pp(d/2)^dag; pseudo-patch rotates over the K
               partitions every cycle (paper, Appendix C); random weight-N/2 inputs
  mirror_full  same with U_full
  patched      product over the K patches of the normalised patch XEB, Eq. (1)-(2), from
               exact probabilities of each patch (small separate simulators)
  ace          the SAME Haar/CZ mirror as mirror_full (same gates, same input string),
               with IDEAL gates, run on QrackAceBackend(--lrc, --lrr) and read from shots,
               as in vm6502q/pyqrack-examples rcs/mirror_nn_qab.py.  The exact noiseless
               echo is 1, so 1 - fidelity_ace is ACE's own seam (elision) error.  Logged:
               fidelity_ace, hamming_weight_ace (popcount, upstream-compatible),
               hamming_dist_ace (distance from the input string; equal to the weight with
               --zero-input), bulk_to_boundary (from the backend's private _unpack; NaN if
               that API is gone), seam_cz (CZs touching an ACE boundary qubit), pyqrack
               (installed version).  ACE lays window-local indices 0..N-1 on its own grid,
               so couplers of --layout windows are placed where ACE puts those indices,
               not where the device has them.  Needs qiskit (the circuit goes in through
               run_qiskit_circuit, as upstream).  Not bound by QRACK_MAX_CPU_QB: ace alone
               can run the full 61-qubit window.

Geometry
  Windows are grown from the centre of the base graph, so the N-window contains the
  (N-1)-window.  The device (coherent errors, per-element error rates, readout) is drawn
  once over the whole base graph and every window inherits its slice; Haar gates are
  drawn once per instance over the whole base graph, so a qubit runs the same gates in
  every window that contains it.  K-patch partitions: balanced to +/-1 qubit, connected,
  minimum cut, five kept (the paper's Appendix B procedure).

Noise (--preset r2, default; --preset ideal zeroes everything; flags override)
  stochastic  RB error per gate converted to a Pauli channel by (d+1)/d:
                CZ 1.9e-3 -> 2.375e-3; SU(2) = 2 SX x 2.5e-4 x 3/2 -> 7.5e-4
              idle 6.34e-4 per qubit per cycle (residual of the paper's 0.872/cycle fit)
              lognormal spread per qubit/coupler, sigma --spread
  coherent    ASSUMED, not reported: CZ phase 0.02+/-0.01 rad, SX over-rotation
              (2+/-1)e-3; optional idle ZZ on undriven couplers
  native      every SU(2) runs as RZ.SX.RZ.SX.RZ with the device SX, forward and
              recompiled inverse alike; every CZ is the same physical CZ.  Angles come
              from u_angles, which is well defined at the poles (diagonal/antidiagonal)
  twirl       --twirl none|mirror|all; frames are merged into the neighbouring SU(2)
              BEFORE native compilation, as Qiskit does
  readout     P(1->0) 1.3%, P(0->1) 0.2%, measurement-twirled (symmetrised) by default
  --stochastic analytic (default): in a scrambling circuit any Pauli error sends the
              overlap to ~0, so F = F_coherent (one deterministic run) x P(no error).
              That fails near the circuit's ends, where an error has not yet spread
              (fwd: the input side; mirror: both the input and the output side).
              --edge-layers L samples the errors of those first/last L noisy cycles by
              trajectory and keeps the analytic factor for the bulk only (--traj runs per
              point).  Default 0 = the pure analytic model, so older JSONL files resume.
              Pick L with selftest [4], which compares every L against full trajectories.
              --stochastic trajectory samples the errors everywhere instead.
  The ace mode ignores the device model: it isolates the simulator's approximation.

Scaling summary: per-cycle decay b(N) per estimator is fitted as u*N + v*(CZ per cycle)
(non-negative) and evaluated at the 61-qubit experiment's counts (layout base only).
Seam table: ACE's seam loss b_ace next to the pseudo-patch's seam gain
b_mirror_full - b_mirror (echo) and b_fwd_full - b_fwd_pp (forward), all per cycle.
ACE points at or below 3/shots are dropped from its decay fit; a b fitted with dropped
depths is marked *.

Big states: --cpu (is_gpu=False); export QRACK_MAX_CPU_QB=36 and QRACK_MAX_ALLOC_MB;
leave QRACK_QUNIT_SEPARABILITY_THRESHOLD unset.  No CUDA; OpenCL via
QRACK_OCL_DEFAULT_DEVICE when not --cpu.

Examples
  python nighthawk_qrack.py seamgap --repo rcs-nighthawk
  python nighthawk_qrack.py selftest --layout rcs-nighthawk/data/layout.json
  QRACK_MAX_CPU_QB=36 python nighthawk_qrack.py run --cpu \
      --layout rcs-nighthawk/data/layout.json --sizes 27-36 --depths 8 16 24 32 36 \
      --instances 2 --twirl none --edge-layers 3 --out r2_scaling.jsonl
  python nighthawk_qrack.py run --layout rcs-nighthawk/data/layout.json \
      --sizes 27-36 --modes ace,mirror_full,mirror --lrc 4 --lrr 4 --out r2_scaling.jsonl
  python nighthawk_qrack.py run --layout rcs-nighthawk/data/layout.json \
      --summarize --out r2_scaling.jsonl
"""
import os

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
os.environ["PYQRACK_SHARED_LIB_PATH"] = QRACK_LIB_PATH

import argparse
import collections
import gc
import hashlib
import itertools
import json
import re
import statistics as st
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PRESETS = {
    "r2": dict(cz_rb=1.9e-3, sx_rb=2.5e-4, idle=6.34e-4, ro10=0.013, ro01=0.002,
               cz_phase=(0.02, 0.01), sx_overrot=(2e-3, 1e-3), zz_idle=(0.0, 0.0), spread=0.4),
    "ideal": dict(cz_rb=0.0, sx_rb=0.0, idle=0.0, ro10=0.0, ro01=0.0,
                  cz_phase=(0.0, 0.0), sx_overrot=(0.0, 0.0), zz_idle=(0.0, 0.0), spread=0.0),
}
MODES = ["fwd_full", "fwd_pp", "mirror", "mirror_full", "patched", "ace"]   # append only: index seeds


# ================================================================== seamgap (numpy only)
def cmd_seamgap(a):
    MED_CZ = 1.9e-3 * 5 / 4          # RB -> Pauli channel, (d+1)/d with d = 4
    MED_SX = 2.5e-4 * 3 / 2          # d = 2
    repo = Path(a.repo)
    man = json.loads((repo / "data/circuits/manifest.json").read_text())["circuits"]
    tab = collections.defaultdict(list)
    for c in man:
        kind = c["kind"] if c["kind"] != "patched" else "K" + c["file"].split("/")[1].lstrip("K")
        n = len(re.findall(r"^\s*cz\b", (repo / "data/circuits" / c["qasm3"]).read_text(), re.M))
        assert n == c["cz"], f"manifest mismatch in {c['file']}"
        tab[(kind, int(re.search(r"d(\d+)", c["file"]).group(1)))].append(n)
    mean = {k: st.mean(v) for k, v in tab.items()}
    kinds = ["full", "mirror", "K3", "K4"]
    print("== CZ counts (mean [min-max]); gaps relative to the sampled full circuit ==")
    print("depth\t" + "\t".join(kinds) + "\tfull-mirror\tfull-K3")
    for d in sorted({d for _, d in tab}):
        cells = [f"{st.mean(tab[(k, d)]):.0f}[{min(tab[(k, d)])}-{max(tab[(k, d)])}]"
                 if (k, d) in tab else "-" for k in kinds]
        gap = lambda k: (f"{mean[('full', d)] - mean[(k, d)]:.0f} "
                         f"({(mean[('full', d)] - mean[(k, d)]) / mean[('full', d)]:.1%})"
                         if (k, d) in mean and ("full", d) in mean else "-")
        print(f"{d}\t" + "\t".join(cells) + f"\t{gap('mirror')}\t{gap('K3')}")

    res = json.loads((repo / "data/results/fidelity_vs_depth.json").read_text())
    pts, fit = res["points"], res["fit"]
    A, f = fit["prefactor"], fit["fidelity_per_cycle"]
    d0 = 36
    czpc = mean[("mirror", d0)] / d0
    deq = mean[("full", d0)] / czpc
    missing = mean[("full", d0)] - mean[("mirror", d0)]
    print(f"\n== fit F(d) = {A:.4f} x {f:.4f}^d ==")
    print(f"at d={d0}: {A * f ** d0:.3e} (headline);  at gate-count-equivalent d={deq:.1f}: "
          f"{A * f ** deq:.3e} (all decay charged to CZ)")
    for eps in (1e-3, MED_CZ, 3e-3):
        print(f"  {missing:.0f} missing CZ at Pauli eps {eps:.2e}: {A * f ** d0 * (1 - eps) ** missing:.3e}")

    fam = {"mirror": "mirror", "3-patch": "K3", "4-patch": "K4"}
    rows = [(fk, int(ds), mean[(fam[fk], int(ds))], np.log(v["fidelity"]), v["se"] / v["fidelity"])
            for fk, pk in pts.items() for ds, v in pk.items() if int(ds) >= a.dmin]
    fams = sorted({r[0] for r in rows})
    print(f"\n== ln F = a + beta*cycles + gamma*CZ, d >= {a.dmin} ==")
    for off in (False, True):
        X = np.array([[d, c] + ([float(fk == g) for g in fams] if off else [1.0]) for fk, d, c, _, _ in rows])
        w = np.array([1 / s for *_, s in rows])
        y = np.array([lf for *_, lf, _ in rows])
        Xw, yw = X * w[:, None], y * w
        b, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        r = yw - Xw @ b
        cov = np.linalg.inv(Xw.T @ Xw) * (r @ r / max(len(y) - X.shape[1], 1))
        e = np.sqrt(np.diag(cov))
        print(f"{'family offsets' if off else 'common intercept':17s} beta {b[0]:+.4f}+/-{e[0]:.4f}  "
              f"gamma {b[1]:+.5f}+/-{e[1]:.5f}   (Pauli CZ error would give {np.log(1 - MED_CZ):+.5f})")
    print("\n== K=3 vs K=4 at equal depth ==\ndepth\tF3/F4\tdCZ\timplied eps_CZ")
    for d in sorted(int(x) for x in pts["3-patch"] if x in pts["4-patch"]):
        r3 = pts["3-patch"][str(d)]["fidelity"] / pts["4-patch"][str(d)]["fidelity"]
        dc = mean[("K3", d)] - mean[("K4", d)]
        print(f"{d}\t{r3:.3f}\t{dc:.0f}\t{-np.log(r3) / dc:+.5f}")
    dec = -np.log(f)
    bcz, b1q = czpc * MED_CZ, 61 * 2 * MED_SX
    print(f"\n== per-cycle budget (Pauli rates) ==\nfitted {dec:.4f} | CZ {bcz:.4f} | 1q {b1q:.4f} | "
          f"unexplained {dec - bcz - b1q:.4f} ({(dec - bcz - b1q) / dec:.0%})")


# ================================================================== one-qubit algebra
I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.diag([1, -1]).astype(complex)
PAULI = {"i": I2, "x": PX, "y": PY, "z": PZ}
P1 = ("x", "y", "z")
P2 = [p for p in itertools.product("ixyz", repeat=2) if p != ("i", "i")]


def rz(t):
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])


def rx(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def haar_su2(rng):
    phi, lam = rng.uniform(0, 2 * np.pi, 2)
    return rz(phi) @ rx(np.arccos(rng.uniform(-1, 1))) @ rz(lam)


def u_angles(M, tol=1e-12):
    """(theta, phi, lambda) of Qiskit's U gate equal to M up to global phase.
    Robust at the poles, where an angle of a zero entry would be arbitrary."""
    c, s = abs(M[0, 0]), abs(M[1, 0])
    th = 2 * np.arctan2(s, c)
    g = np.angle(M[0, 0]) if c > tol else np.angle(M[1, 0])
    ph = np.angle(M[1, 0]) - g if s > tol else 0.0
    la = np.angle(-M[0, 1]) - g if s > tol else np.angle(M[1, 1]) - g - ph
    return float(th), float(ph), float(la)


def native(M, sx):
    """Device realisation of target SU(2) M: RZ(ph+pi).SX.RZ(th+pi).SX.RZ(la),
    with the angles of u_angles (well defined at the poles)."""
    th, ph, la = u_angles(M)
    return rz(ph + np.pi) @ sx @ rz(th + np.pi) @ sx @ rz(la)


def flat(m):
    return [complex(m[0, 0]), complex(m[0, 1]), complex(m[1, 0]), complex(m[1, 1])]


def same_up_to_phase(A, B):
    return 1 - abs(np.trace(A.conj().T @ B)) / 2


# ================================================================== graphs
class BaseGraph:
    """The graph windows are cut from: the released 61-qubit layout or a synthetic grid."""

    def __init__(self, layout=None, grid=None, K=3):
        if layout:
            L = json.load(open(layout))
            self.n = L["num_qubits"]
            self.colours = {k: [tuple(e) for e in L["matchings"][k]] for k in L["schedule"]}
            cols = L["device_lattice"]["cols"]
            self.labels = L["logical_to_physical"]
            rr, cc = L["subgrid"]["rows"], L["subgrid"]["cols"]
            cen = ((rr[0] + rr[1]) / 2, (cc[0] + cc[1]) / 2)
            self.dist = np.array([np.hypot(p // cols - cen[0], p % cols - cen[1]) for p in self.labels])
            if str(K) not in L["partitions"]:
                raise SystemExit(f"layout has no K={K} partitions (has {sorted(L['partitions'])})")
            bnd = [len(p["boundary_edges"]) for p in L["partitions"][str(K)]]
            ncoup = sum(len(v) for v in self.colours.values())
            self.czpc_full = ncoup / 4
            self.czpc_pp = (ncoup - np.mean(bnd)) / 4
            self.is_layout = True
        else:
            R, C = (int(x) for x in grid.lower().split("x"))
            self.n = R * C
            col = {k: [] for k in "ABCD"}
            for r in range(R):
                for c in range(C - 1):
                    col["A" if c % 2 == 0 else "B"].append((r * C + c, r * C + c + 1))
            for r in range(R - 1):
                for c in range(C):
                    col["C" if r % 2 == 0 else "D"].append((r * C + c, (r + 1) * C + c))
            self.colours = col
            self.labels = list(range(self.n))
            self.dist = np.array([np.hypot(q // C - (R - 1) / 2, q % C - (C - 1) / 2) for q in range(self.n)])
            self.is_layout = False
        self.adj = defaultdict(set)
        for es in self.colours.values():
            for u, v in es:
                self.adj[u].add(v)
                self.adj[v].add(u)

    def window(self, N):
        if N > self.n:
            raise SystemExit(f"window {N} larger than base graph ({self.n})")
        start = int(np.argmin(self.dist))
        win, inside = [start], {start}
        while len(win) < N:
            front = {v for u in win for v in self.adj[u] if v not in inside}
            v = min(front, key=lambda x: (round(self.dist[x], 6), x))
            win.append(v)
            inside.add(v)
        win = sorted(win)
        new = {q: i for i, q in enumerate(win)}
        colours = {k: [(new[u], new[v]) for u, v in es if u in new and v in new]
                   for k, es in self.colours.items()}
        return win, colours


def connected(nodes, adj):
    nodes = set(int(x) for x in nodes)
    if not nodes:
        return False
    s = next(iter(nodes))
    seen, stack = {s}, [s]
    while stack:
        for v in adj[stack.pop()]:
            if v in nodes and v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(nodes)


def dcut(lab, u, v, adj):
    """Change in cut size if u and v (in different parts) swap labels.  The u-v edge
    itself stays cut either way, so it is skipped."""
    pu, pv, d = lab[u], lab[v], 0
    for w in adj[u]:
        if w != v:
            d += int(lab[w] != pv) - int(lab[w] != pu)
    for w in adj[v]:
        if w != u:
            d += int(lab[w] != pu) - int(lab[w] != pv)
    return d


def partitions(n, edges, K, rng, keep=5, restarts=600):
    """Balanced (+/-1), connected K-partitions of minimum cut: random growth + swap search.
    Swaps are scored incrementally (dcut); acceptance is the same as recomputing the cut,
    so a given rng gives the same partitions as before."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    sizes = [n // K + (1 if i < n % K else 0) for i in range(K)]
    cut = lambda lab: sum(lab[u] != lab[v] for u, v in edges)
    found = {}
    for _ in range(restarts):
        lab = -np.ones(n, dtype=int)
        seeds = rng.choice(n, K, replace=False)
        members = [[int(s)] for s in seeds]
        for p, s in enumerate(seeds):
            lab[s] = p
        while (lab < 0).any():
            grew = False
            for p in rng.permutation(K):
                if len(members[p]) >= sizes[p]:
                    continue
                front = {v for u in members[p] for v in adj[u] if lab[v] < 0}
                if not front:
                    continue
                v = max(front, key=lambda x: (sum(lab[y] == p for y in adj[x]), rng.random()))
                lab[v] = p
                members[p].append(v)
                grew = True
            if not grew:
                break
        if (lab < 0).any():
            continue
        best, improved = cut(lab), True
        while improved:
            improved = False
            bnd = [u for u in range(n) if any(lab[v] != lab[u] for v in adj[u])]
            rng.shuffle(bnd)
            for u, v in itertools.product(bnd, bnd):
                if lab[u] == lab[v]:
                    continue
                dc = dcut(lab, u, v, adj)
                if dc >= 0:
                    continue
                pu, pv = lab[u], lab[v]
                lab[u], lab[v] = pv, pu
                if connected(np.flatnonzero(lab == pu), adj) and connected(np.flatnonzero(lab == pv), adj):
                    best, improved = best + dc, True
                    break
                lab[u], lab[v] = pu, pv
        if all(connected(np.flatnonzero(lab == p), adj) for p in range(K)):
            key = frozenset(frozenset(np.flatnonzero(lab == p).tolist()) for p in range(K))
            found[key] = best
    out = []
    for key, c in sorted(found.items(), key=lambda kv: kv[1])[:keep]:
        patches = sorted(sorted(s) for s in key)
        lab = np.zeros(n, dtype=int)
        for p, s in enumerate(patches):
            lab[s] = p
        out.append(dict(patches=patches, cut=int(c), boundary={e for e in edges if lab[e[0]] != lab[e[1]]}))
    return out


# ================================================================== device
class Device:
    """Static device: coherent parameters (statevector engine), per-element stochastic
    Pauli rates and readout.  Draw once over the base graph, then window()/patch()."""

    def __init__(self, n, colours, a, rng):
        self.n = n
        self.edges = sorted({e for es in colours.values() for e in es})
        s = a.spread
        ln = lambda med, size: med * np.exp(rng.normal(0, s, size) - s * s / 2)
        (ms, ss), (md, sd), (mz, sz) = a.sx_overrot, a.cz_phase, a.zz_idle
        kap = rng.normal(ms, ss, n) if (ms or ss) else np.zeros(n)
        self.sx = [rx(np.pi / 2 * (1 + k)) for k in kap]
        self.cz_diag = {e: ([1, 0, 0, -np.exp(1j * rng.normal(md, sd))] if (md or sd) else None)
                        for e in self.edges}
        self.zz_diag = ({e: [1, 0, 0, np.exp(1j * rng.normal(mz, sz))] for e in self.edges}
                        if (mz or sz) else {})
        self.e_cz = dict(zip(self.edges, ln(a.cz_rb * 5 / 4, len(self.edges))))
        self.e_1q = ln(2 * a.sx_rb * 3 / 2, n)
        self.e_idle = ln(a.idle, n)
        self.ro10, self.ro01 = ln(a.ro10, n), ln(a.ro01, n)
        self.meas_twirl = a.meas_twirl
        self.traj = False                 # sample stochastic errors everywhere in the engine?

    def _slice(self, qubits, colours):
        new = {q: i for i, q in enumerate(qubits)}
        rm = lambda d: {(new[u], new[v]): x for (u, v), x in d.items() if u in new and v in new}
        v = Device.__new__(Device)
        v.n = len(qubits)
        v.edges = sorted({e for es in colours.values() for e in es}) if colours else sorted(rm(self.e_cz))
        v.sx = [self.sx[q] for q in qubits]
        v.cz_diag, v.zz_diag, v.e_cz = rm(self.cz_diag), rm(self.zz_diag), rm(self.e_cz)
        v.e_1q, v.e_idle = self.e_1q[qubits], self.e_idle[qubits]
        v.ro10, v.ro01 = self.ro10[qubits], self.ro01[qubits]
        v.meas_twirl, v.traj = self.meas_twirl, self.traj
        return v

    window = _slice
    patch = _slice

    def ro_err(self, x0=None):
        if self.meas_twirl or x0 is None:
            return (self.ro10 + self.ro01) / 2
        return np.where(np.asarray(x0) == 1, self.ro10, self.ro01)

    def p_noerr(self, cycles, n_layers):
        lnp = sum(np.log1p(-self.e_cz[e]) for _, czs in cycles for e in czs)
        return float(np.exp(lnp + n_layers * np.sum(np.log1p(-self.e_1q) + np.log1p(-self.e_idle))))


# ================================================================== circuits
def cycles_for(gates, colours, drop):
    return [(layer, [e for e in colours["ABCD"[t % 4]] if e not in drop(t)]) for t, layer in enumerate(gates)]


def forward_steps(cyc):
    out = []
    for layer, czs in cyc:
        out += [("u", layer), ("cz", czs), ("cycle", None)]
    return out


def inverse_steps(cyc):
    out = []
    for layer, czs in reversed(cyc):
        out += [("cz", czs), ("u", [m.conj().T for m in layer]), ("cycle", None)]
    return out


def edge_set(n_noisy, L):
    """Global indices of the first and last L noisy cycles (only the first L are 'edge' in
    the forward modes, whose last noisy cycle is followed by an ideal inverse)."""
    L = max(0, min(L, n_noisy))
    return frozenset(range(L)) | frozenset(range(n_noisy - L, n_noisy))


# ================================================================== engine
class Runner:
    """One QrackSimulator, reset between runs.  halves = [(steps, noisy, twirl), ...].
    Stochastic Pauli errors are sampled on noisy steps when dev.traj is set, or when the
    step's global cycle index (counted across halves) is in `edge`."""

    def __init__(self, n, dev, cpu):
        from pyqrack import QrackSimulator
        self.n, self.dev = n, dev
        self.sim = QrackSimulator(n, is_gpu=not cpu)
        self.n_fold = 0

    def _pauli(self, p, q):
        if p != "i":
            getattr(self.sim, p)(q)

    def run(self, x0, halves, rng, want="survival", edge=frozenset()):
        sim, n, dev = self.sim, self.n, self.dev
        sim.reset_all()
        self.n_fold = 0                        # CZs twirled by an explicit (unmerged) frame
        pending = [I2] * n                     # post-CZ twirl correction, folded forward
        prep = [PX if b else I2 for b in x0]   # input string, folded into the first layer
        first = True
        c_glob = 0                             # cycle index across all halves
        for steps, noisy, twirl in halves:
            pre = None
            for i, (kind, pay) in enumerate(steps):
                stoch = noisy and (dev.traj or c_glob in edge)
                if kind == "u":
                    frame = [I2] * n
                    # next CZ layer, skipping "cycle" markers: inverse halves run
                    # cz, u, cycle, cz, ... so steps[i+1] after a u is never the cz
                    nxt = next((st_ for st_ in steps[i + 1:] if st_[0] != "cycle"), None)
                    if noisy and twirl and nxt is not None and nxt[0] == "cz":
                        pre = {}
                        for u, v in nxt[1]:
                            pa = P1[rng.integers(3)] if rng.random() < 0.75 else "i"
                            pb = P1[rng.integers(3)] if rng.random() < 0.75 else "i"
                            pre[(u, v)] = (pa, pb)
                            frame[u], frame[v] = PAULI[pa], PAULI[pb]
                    for q in range(n):
                        t = frame[q] @ pay[q] @ pending[q]
                        if first:
                            t = t @ prep[q]
                        pending[q] = I2
                        sim.mtrx(flat(native(t, dev.sx[q]) if noisy else t), q)
                        if stoch and rng.random() < dev.e_1q[q]:
                            self._pauli(P1[rng.integers(3)], q)
                    first = False
                elif kind == "cz":
                    if first:
                        for q in range(n):
                            if x0[q]:
                                sim.x(q)
                        first = False
                    for u, v in pay:
                        pa = pb = "i"
                        if noisy and twirl:
                            if pre is not None and (u, v) in pre:
                                pa, pb = pre[(u, v)]
                            else:                   # fold point: explicit ideal pre-frame
                                self.n_fold += 1
                                pa = P1[rng.integers(3)] if rng.random() < 0.75 else "i"
                                pb = P1[rng.integers(3)] if rng.random() < 0.75 else "i"
                                self._pauli(pa, u)
                                self._pauli(pb, v)
                        diag = dev.cz_diag.get((u, v)) if noisy else None
                        if diag is None:
                            sim.mcz([u], v)
                        else:
                            sim.mcmtrx([u], diag, v)
                        if noisy and twirl:         # CZ (Pa x Pb) CZ, up to phase
                            pending[u] = pending[u] @ PAULI[pa] @ (PZ if pb in "xy" else I2)
                            pending[v] = pending[v] @ PAULI[pb] @ (PZ if pa in "xy" else I2)
                        if stoch and rng.random() < dev.e_cz[(u, v)]:
                            qa, qb = P2[rng.integers(15)]
                            self._pauli(qa, u)
                            self._pauli(qb, v)
                    pre = None
                    if noisy and dev.zz_diag:
                        driven = set(pay)
                        for e in dev.edges:
                            if e not in driven:
                                sim.mcmtrx([e[0]], dev.zz_diag[e], e[1])
                else:
                    if stoch:
                        for q in np.flatnonzero(rng.random(n) < dev.e_idle):
                            self._pauli(P1[rng.integers(3)], int(q))
                    c_glob += 1
            for q in range(n):
                if not np.allclose(pending[q], I2):
                    sim.mtrx(flat(pending[q]), q)
                    pending[q] = I2
        if want == "probs":
            return np.array(sim.out_probs(), dtype=np.float64)
        if want == "ket":
            return np.array(sim.out_ket())
        return float(sim.prob_perm(list(range(n)), [bool(b) for b in x0]))


# ================================================================== ACE (QrackAceBackend)
def pyqrack_version():
    try:
        import importlib.metadata as im
        for dist in ("pyqrack", "pyqrack-cpu", "pyqrack-complex128", "pyqrack-cpu-complex128"):
            try:
                return im.version(dist)
            except im.PackageNotFoundError:
                continue
    except ImportError:
        pass
    try:
        import pyqrack
        return str(getattr(pyqrack, "__version__", "unknown"))
    except ImportError:
        return "unknown"


def ace_kwargs(a):
    kw = dict(long_range_columns=a.lrc, long_range_rows=a.lrr)
    if a.ace_torus:
        kw["is_torus"] = True
    return kw


def ace_shots(a, N):
    return a.ace_shots or (1 << min(13, N + 2))       # upstream default


def ace_cfg(cfg, a):
    return f"{cfg}/ace-{a.lrc}x{a.lrr}{'-torus' if a.ace_torus else ''}-s{a.ace_shots or 'auto'}"


def ace_mirror(N, x0, steps, a, shots):
    """Ideal circuit on QrackAceBackend via run_qiskit_circuit (the path mirror_nn_qab.py
    uses), sampled with measure_shots.  Returns the logged ACE fields."""
    from pyqrack import QrackAceBackend
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(N)
    for q in np.flatnonzero(x0):
        qc.x(int(q))
    czs = []
    for kind, pay in steps:
        if kind == "u":
            for q, m in enumerate(pay):
                qc.u(*u_angles(m), q)
        elif kind == "cz":
            for u, v in pay:
                qc.cz(int(u), int(v))
                czs.append((u, v))
    sim = QrackAceBackend(N, **ace_kwargs(a))
    n = sim.num_qubits()
    try:                                  # private API: degrade to NaN if it changes
        bnd = {lq for lq in range(n) if len(sim._unpack(lq)) > 1}
        b2b = (n - len(bnd)) / len(bnd) if bnd else float("inf")
        n_bnd, seam = len(bnd), sum(1 for u, v in czs if u in bnd or v in bnd)
    except (AttributeError, TypeError):
        b2b, n_bnd, seam = float("nan"), None, None
    sim.run_qiskit_circuit(qc, shots=0)
    counts = Counter(int(s) for s in sim.measure_shots(list(range(N)), shots))
    x0i = sum(1 << int(q) for q in np.flatnonzero(x0))
    F = counts.get(x0i, 0) / shots
    return dict(fidelity_ace=F, se_ace=float(np.sqrt(F * (1 - F) / shots)),
                hamming_weight_ace=sum(s.bit_count() * c for s, c in counts.items()) / shots,
                hamming_dist_ace=sum((s ^ x0i).bit_count() * c for s, c in counts.items()) / shots,
                bulk_to_boundary=b2b, ace_boundary_qubits=n_bnd, seam_cz=seam,
                lrc=a.lrc, lrr=a.lrr, ace_torus=bool(a.ace_torus), shots=shots,
                pyqrack=pyqrack_version())


# ================================================================== estimators
def norm_xeb(p, q):
    D = p.size
    return (D * np.dot(p, q) - 1) / (D * np.dot(p, p) - 1)


def confuse(probs, dev):
    """Per-qubit readout channel on a little-endian probability vector (qubit q = bit q)."""
    k = dev.n
    t = probs.reshape([2] * k)
    e_sym = (dev.ro10 + dev.ro01) / 2
    for q in range(k):
        e10, e01 = (e_sym[q], e_sym[q]) if dev.meas_twirl else (dev.ro10[q], dev.ro01[q])
        M = np.array([[1 - e01, e10], [e01, 1 - e10]])
        ax = k - 1 - q
        t = np.moveaxis(np.tensordot(M, t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)


def measure_point(mode, N, d, inst, tj, ctx, a):
    """One (mode, depth, instance, trajectory) point.  Returns the JSONL record."""
    dev, gates, colours, parts, bnds = ctx["dev"], ctx["gates"], ctx["colours"], ctx["parts"], ctx["bnds"]
    # ace shares mirror_full's seed, hence its input string: the two are paired
    seed_mode = "mirror_full" if mode == "ace" else mode
    trng = np.random.default_rng([a.seed, N, inst, d, MODES.index(seed_mode), tj])
    full = cycles_for(gates[:d], colours, lambda t: set())
    pp = cycles_for(gates[:d], colours, lambda t: bnds[t % len(bnds)])
    h = d // 2
    L = 0 if dev.traj else int(getattr(a, "edge_layers", 0) or 0)
    rec = dict(N=N, instance=inst, depth=d, mode=mode, traj=tj, cfg=ctx["cfg"])
    x0 = np.zeros(N, dtype=int)
    if not a.zero_input:
        x0[trng.choice(N, N // 2, replace=False)] = 1
    tw_f, tw_m = a.twirl == "all", a.twirl in ("all", "mirror")
    if mode in ("fwd_full", "fwd_pp"):
        cyc = full if mode == "fwd_full" else pp
        edge = frozenset(range(min(L, len(cyc))))           # input side only
        Fc = ctx["big"].run(x0, [(forward_steps(cyc), True, tw_f), (inverse_steps(cyc), False, False)],
                            trng, edge=edge)
        bulk = [c for i, c in enumerate(cyc) if i not in edge]
        P = 1.0 if dev.traj else dev.p_noerr(bulk, len(bulk))
        ro = float(np.prod(1 - dev.ro_err()))
        rec.update(F_coh=Fc, P_noerr=P, F=Fc * P, F_ro=Fc * P * ro, cz=sum(len(c) for _, c in cyc))
    elif mode in ("mirror", "mirror_full"):
        half = (cycles_for(gates[:h], colours, lambda t: bnds[t % len(bnds)]) if mode == "mirror"
                else cycles_for(gates[:h], colours, lambda t: set()))
        edge = edge_set(2 * h, L)                            # input and output side
        Fc = ctx["big"].run(x0, [(forward_steps(half), True, tw_m), (inverse_steps(half), True, tw_m)],
                            trng, edge=edge)
        bulk = [half[i] for i in range(h) if i not in edge] + \
               [half[h - 1 - j] for j in range(h) if h + j not in edge]
        P = 1.0 if dev.traj else dev.p_noerr(bulk, len(bulk))
        ro = float(np.prod(1 - dev.ro_err(x0)))
        rec.update(F_coh=Fc, P_noerr=P, F=Fc * P, F_ro=Fc * P * ro, cz=2 * sum(len(c) for _, c in half))
    elif mode == "ace":
        half = cycles_for(gates[:h], colours, lambda t: set())
        r = ace_mirror(N, x0, forward_steps(half) + inverse_steps(half), a, ace_shots(a, N))
        rec.update(r)
        rec.update(F=r["fidelity_ace"], F_ro=r["fidelity_ace"], cz=2 * sum(len(c) for _, c in half),
                   cfg=ctx["ace_cfg"])
    else:
        part = parts[(inst + tj) % len(parts)]
        Fh, czs = 1.0, 0
        for qs in part["patches"]:
            new = {q: i for i, q in enumerate(qs)}
            pcol = {k: [(new[u], new[v]) for u, v in es if u in new and v in new] for k, es in colours.items()}
            pdev = dev.patch(qs, pcol)
            pcyc = cycles_for([[layer[q] for q in qs] for layer in gates[:d]], pcol, lambda t: set())
            key = tuple(qs)
            if key not in ctx["patch_runners"]:
                ctx["patch_runners"][key] = Runner(len(qs), pdev, cpu=True)
            pr = ctx["patch_runners"][key]
            pr.dev = pdev
            zero = np.zeros(len(qs), dtype=int)
            p_id = pr.run(zero, [(forward_steps(pcyc), False, False)], trng, want="probs")
            p_no = pr.run(zero, [(forward_steps(pcyc), True, False)], trng, want="probs")
            P = 1.0 if dev.traj else pdev.p_noerr(pcyc, len(pcyc))
            q_meas = confuse(P * p_no + (1 - P) / p_no.size, pdev)
            Fh *= norm_xeb(p_id, q_meas)
            czs += sum(len(c) for _, c in pcyc)
        rec.update(F=Fh, F_ro=Fh, cz=czs, cut=part["cut"], K=len(part["patches"]))
    if mode not in ("ace", "patched") and L:
        rec["edge_layers"] = L
    return rec


# ================================================================== run / summarize
def fill_preset(a):
    for k, v in PRESETS[a.preset].items():
        if getattr(a, k) is None:
            setattr(a, k, v)


def cfg_hash(a):
    # ACE settings are deliberately NOT hashed here (they go in the ace records' own cfg
    # tag), so JSONL files written before the ace mode existed still resume.
    keys = ["layout", "grid", "preset", "cz_rb", "sx_rb", "idle", "ro10", "ro01", "cz_phase",
            "sx_overrot", "zz_idle", "spread", "meas_twirl", "twirl", "stochastic", "patches",
            "seed", "device_seed", "zero_input"]
    d = {k: getattr(a, k) for k in keys}
    if a.twirl != "none":
        d["engine"] = "twirl-lookahead-2"   # inverse-half frames now merged: new numbers
    if getattr(a, "edge_layers", 0) and a.stochastic == "analytic":
        d["edge_layers"] = a.edge_layers    # only when set: L=0 keeps the old hashes
    blob = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def parse_sizes(s):
    out = []
    for part in s.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    return out


def load_records(path, cfgs):
    recs, other = {}, 0
    if path and os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r.get("cfg") not in cfgs:
                other += 1
                continue
            recs[(r["N"], r["instance"], r["depth"], r["mode"], r["traj"])] = r
    if other:
        print(f"# note: {other} records in {path} belong to another configuration and are ignored")
    return recs


def _decay(by, N, e, floor=0.0):
    """Per-cycle decay from a log-linear fit over depths whose mean F exceeds floor.
    Returns (b or None, number of depths dropped)."""
    pts = sorted((d, float(np.mean(v))) for (n_, d, e_), v in by.items() if n_ == N and e_ == e)
    good = [(d, np.log(m)) for d, m in pts if m > floor]
    dropped = len(pts) - len(good)
    if len(good) < 2:
        return None, dropped
    dd, yy = np.array(good).T
    return float(-np.polyfit(dd, yy, 1)[0]), dropped


def summarize(recs, base, K):
    lab = lambda m: f"K{K}" if m == "patched" else m
    by = defaultdict(list)
    czpc = defaultdict(list)
    b2b = {}
    shots = {}
    versions = set()
    for r in recs.values():
        by[(r["N"], r["depth"], r["mode"])].append(r["F_ro"])
        czpc[(r["N"], r["mode"])].append(r["cz"] / r["depth"])
        if r["mode"] == "ace":
            b2b[r["N"]] = r["bulk_to_boundary"]
            shots[r["N"]] = min(shots.get(r["N"], r["shots"]), r["shots"])
            versions.add(r.get("pyqrack", "unrecorded"))
    if len(versions) > 1:
        print(f"# WARNING: ace records come from several PyQrack versions: {sorted(versions)}")
    Ns = sorted({k[0] for k in by})
    ests = [m for m in MODES if any(k[2] == m for k in by)]
    rat_ests = [e for e in ests if e not in ("fwd_full", "ace")]    # ace is ideal: no ratio to noisy truth
    print("\n== means per point (readout included; readout-free F in the JSONL; ace is noiseless) ==")
    for N in Ns:
        print(f"\nN = {N}" + (f"   ACE bulk/boundary {b2b[N]:.3g}" if N in b2b else ""))
        print("d    " + "".join(f"{lab(e):>13s}" for e in ests) +
              "".join(f"{lab(e) + '/full':>18s}" for e in rat_ests))
        for d in sorted({k[1] for k in by if k[0] == N}):
            m = {e: np.mean(by[(N, d, e)]) for e in ests if by.get((N, d, e))}
            row = "".join(f"{m[e]:13.5f}" if e in m else f"{'-':>13s}" for e in ests)
            rat = "".join(f"{m[e] / m['fwd_full']:18.3f}" if e in m and m.get("fwd_full") else f"{'-':>18s}"
                          for e in rat_ests)
            print(f"{d:<4} {row}{rat}")
    if len(Ns) < 2 and "ace" not in ests:
        return
    print("\n== per-cycle decay b(N) = -dlnF/dd  /  CZ per cycle ==")
    print("N    " + "".join(f"{lab(e):>20s}" for e in ests))
    rows = defaultdict(list)
    bmap, dropmap = {}, {}
    for N in Ns:
        cells = []
        for e in ests:
            floor = 3.0 / shots[N] if e == "ace" and N in shots else 0.0
            bN, drop = _decay(by, N, e, floor)
            if bN is not None:
                c = float(np.mean(czpc[(N, e)]))
                rows[e].append((N, c, bN))
                bmap[(N, e)], dropmap[(N, e)] = bN, drop
                cells.append(f"{bN:9.4f}{'*' if drop else ' '}/ {c:6.2f}  ")
            else:
                cells.append(f"{'-':>20s}")
        print(f"{N:<4} " + "".join(cells))
    if any(dropmap.values()):
        print("(* fitted with depths dropped at or below 3/shots; the fit covers shallow depths only)")
    if "ace" in ests:
        g = lambda N, x, y: (f"{bmap[(N, x)] - bmap[(N, y)]:+14.4f}"
                             if (N, x) in bmap and (N, y) in bmap else f"{'-':>14s}")
        print("\n== seam terms, per cycle: ACE elision loss vs pseudo-patch thinning gain ==")
        print("(b_ace > 0: ACE under-reports; gains > 0: pseudo-patch over-reports)")
        print(f"N    {'bulk/bnd':>9s}{'b_ace':>11s}{'mfull-mirror':>14s}{'ffull-fwd_pp':>14s}{'F_ace(dmax)':>13s}")
        for N in Ns:
            ds = [d for (n_, d, e) in by if n_ == N and e == "ace"]
            fa = f"{np.mean(by[(N, max(ds), 'ace')]):13.4f}" if ds else f"{'-':>13s}"
            ba = (f"{bmap[(N, 'ace')]:10.4f}{'*' if dropmap[(N, 'ace')] else ' '}"
                  if (N, "ace") in bmap else f"{'-':>11s}")
            bb = f"{b2b[N]:9.3g}" if N in b2b else f"{'-':>9s}"
            print(f"{N:<4} {bb}{ba}{g(N, 'mirror_full', 'mirror')}{g(N, 'fwd_full', 'fwd_pp')}{fa}")
    if len(Ns) < 2 or not base.is_layout:
        return
    print("\n== fitted b = u*N + v*CZpc (non-negative), evaluated at the 61-qubit experiment ==")
    target = {"fwd_full": base.czpc_full, "mirror_full": base.czpc_full,
              "fwd_pp": base.czpc_pp, "mirror": base.czpc_pp, "patched": base.czpc_pp}
    pred = {}
    for e in ests:
        if e not in target or len(rows[e]) < 2:
            continue
        A = np.array([[N, c] for N, c, _ in rows[e]])
        y = np.array([b for *_, b in rows[e]])
        (u, v), *_ = np.linalg.lstsq(A, y, rcond=None)
        if u < 0 or v < 0:
            cand = []
            for j in (0, 1):
                cj = max(float(A[:, j] @ y / (A[:, j] @ A[:, j])), 0.0)
                cand.append((np.sum((y - cj * A[:, j]) ** 2), j, cj))
            _, j, cj = min(cand)
            u, v = (cj, 0.0) if j == 0 else (0.0, cj)
        pred[e] = u * 61 + v * target[e]
        print(f"{lab(e):12s} u {u:.3e}/qubit  v {v:.3e}/CZ  b(61) {pred[e]:.4f}  "
              f"per-cycle {np.exp(-pred[e]):.4f}   (paper fit 0.8717)")
    if "fwd_full" in pred:
        for e in ests:
            if e != "fwd_full" and e in pred:
                print(f"{lab(e):12s} / truth at 61 qubits, d=36: {np.exp(-36 * (pred[e] - pred['fwd_full'])):.3f}")
    if len(Ns) < 5 or max(Ns) - min(Ns) < 6:
        print("(few or narrowly spaced sizes: the 61-qubit numbers are indicative only)")


def build_ctx(base, dev_base, N, a, cfg, need_big):
    win, colours = base.window(N)
    edges = sorted({e for es in colours.values() for e in es})
    parts = partitions(N, edges, a.patches, np.random.default_rng([a.seed, N]))
    if not parts:
        raise SystemExit(f"no connected balanced K={a.patches} partition for N={N}")
    dev = dev_base.window(win, colours)
    ctx = dict(win=win, colours=colours, parts=parts, bnds=[p["boundary"] for p in parts],
               dev=dev, cfg=cfg, ace_cfg=ace_cfg(cfg, a), patch_runners={},
               big=Runner(N, dev, a.cpu) if need_big else None)
    return ctx


def cmd_run(a):
    fill_preset(a)
    a.stochastic = a.stochastic or "analytic"
    cfg = cfg_hash(a)
    base = BaseGraph(a.layout, a.grid, a.patches)
    recs = load_records(a.out, {cfg, ace_cfg(cfg, a)})
    if a.summarize:
        summarize(recs, base, a.patches)
        return
    modes = a.modes.split(",")
    bad = set(modes) - set(MODES)
    if bad:
        raise SystemExit(f"unknown modes {bad}; choose from {MODES}")
    if "ace" in modes:
        try:
            from pyqrack import QrackAceBackend  # noqa: F401
            import qiskit  # noqa: F401
        except ImportError as err:
            raise SystemExit(f"ace mode needs QrackAceBackend and qiskit: {err}")
    dev_base = Device(base.n, base.colours, a, np.random.default_rng(a.device_seed))
    dev_base.traj = a.stochastic == "trajectory"
    hybrid = bool(a.edge_layers) and not dev_base.traj
    fh = open(a.out, "a")
    dmax = max(a.depths)
    print(f"# config {cfg}: preset {a.preset}, twirl {a.twirl}, stochastic {a.stochastic}"
          f"{f' (edge layers {a.edge_layers})' if hybrid else ''}, K={a.patches}")
    if "ace" in modes:
        print(f"# ace tag {ace_cfg(cfg, a)}, pyqrack {pyqrack_version()}")
    for N in parse_sizes(a.sizes):
        ctx = build_ctx(base, dev_base, N, a, cfg, any(m not in ("patched", "ace") for m in modes))
        print(f"\n# N={N}: base labels {[base.labels[q] for q in ctx['win']]}")
        print(f"# {len(ctx['dev'].edges)} couplers; K={a.patches} cuts {[p['cut'] for p in ctx['parts']]}, "
              f"patch sizes {[len(s) for s in ctx['parts'][0]['patches']]}")
        for inst in range(a.instances):
            irng = np.random.default_rng([a.seed, inst])
            gb = [[haar_su2(irng) for _ in range(base.n)] for _ in range(dmax)]
            ctx["gates"] = [[layer[q] for q in ctx["win"]] for layer in gb]
            for d in a.depths:
                for mode in modes:
                    twirled = (mode.startswith("fwd") and a.twirl == "all") or \
                              (mode.startswith("mirror") and a.twirl in ("all", "mirror"))
                    sampled = dev_base.traj or twirled or (hybrid and mode not in ("patched", "ace"))
                    ntraj = 1 if mode == "ace" else (a.traj if sampled else 1)
                    for tj in range(ntraj):
                        key = (N, inst, d, mode, tj)
                        if key in recs:
                            continue
                        t0 = time.time()
                        rec = measure_point(mode, N, d, inst, tj, ctx, a)
                        rec["seconds"] = round(time.time() - t0, 3)
                        fh.write(json.dumps(rec, default=float) + "\n")
                        fh.flush()
                        recs[key] = rec
                        extra = (f" hw {rec['hamming_weight_ace']:.2f} hd {rec['hamming_dist_ace']:.2f} "
                                 f"b/b {rec['bulk_to_boundary']:.3g} seamCZ {rec['seam_cz']}"
                                 if mode == "ace" else "")
                        print(f"N {N} inst {inst} d {d:>3} {mode:<11s} traj {tj:<3} CZ {rec['cz']:>5} "
                              f"F {rec['F']:.6f} F_ro {rec['F_ro']:.6f}{extra} ({rec['seconds']:.1f}s)",
                              flush=True)
        del ctx
        gc.collect()
    summarize(recs, base, a.patches)


# ================================================================== selftest
def cmd_selftest(a):
    base = BaseGraph(a.layout, a.grid)
    N = min(a.n, base.n)
    ok = True

    def mk(preset, **over):
        ns = argparse.Namespace(**{k: None for k in PRESETS["r2"]}, preset=preset, meas_twirl=True,
                                twirl="none", patches=3, seed=7, cpu=True, zero_input=False,
                                stochastic="analytic", traj=1, layout=a.layout, grid=a.grid, device_seed=5,
                                lrc=4, lrr=4, ace_torus=False, ace_shots=None, edge_layers=0)
        fill_preset(ns)
        for k, v in over.items():
            setattr(ns, k, v)
        return ns

    def ctx_for(ns, traj_mode=False, modes_big=True):
        dev_base = Device(base.n, base.colours, ns, np.random.default_rng(ns.device_seed))
        dev_base.traj = traj_mode
        c = build_ctx(base, dev_base, N, ns, "selftest", modes_big)
        irng = np.random.default_rng([ns.seed, 0])
        gb = [[haar_su2(irng) for _ in range(base.n)] for _ in range(24)]
        c["gates"] = [[layer[q] for q in c["win"]] for layer in gb]
        return c

    print(f"[1] noiseless exactness, N={N}, statevector modes, twirl all")
    ns = mk("ideal", twirl="all")
    c = ctx_for(ns)
    worst = 0.0
    for mode in MODES:
        if mode == "ace":
            continue
        for d in (4, 8):
            for tj in range(2):
                worst = max(worst, abs(1 - measure_point(mode, N, d, 0, tj, c, ns)["F"]))
    print(f"    max |1 - F| = {worst:.2e}  {'OK' if worst < 1e-4 else 'FAIL'}")
    ok &= worst < 1e-4

    print("[2] Loschmidt echo vs explicit two-ket overlap (coherent device)")
    ns = mk("r2", cz_phase=(0.08, 0.04), sx_overrot=(0.02, 0.01), zz_idle=(0.02, 0.01))
    c = ctx_for(ns)
    cyc = cycles_for(c["gates"][:12], c["colours"], lambda t: set())
    x0 = np.zeros(N, dtype=int)
    x0[::3] = 1
    r, rng = c["big"], np.random.default_rng(1)
    F_l = r.run(x0, [(forward_steps(cyc), True, False), (inverse_steps(cyc), False, False)], rng)
    k1 = r.run(x0, [(forward_steps(cyc), False, False)], rng, want="ket")
    k2 = r.run(x0, [(forward_steps(cyc), True, False)], rng, want="ket")
    F_e = abs(np.vdot(k1, k2)) ** 2
    print(f"    echo {F_l:.8f}  overlap {F_e:.8f}  diff {abs(F_l - F_e):.1e}  "
          f"{'OK' if abs(F_l - F_e) < 1e-5 else 'FAIL'}")
    ok &= abs(F_l - F_e) < 1e-5

    print("[3] twirl control: CZ phase only, twirl all -> mirror tracks forward")
    ns = mk("ideal", cz_phase=(0.15, 0.0), twirl="all")
    c = ctx_for(ns)
    fw = np.mean([measure_point("fwd_pp", N, 12, 0, t, c, ns)["F"] for t in range(a.traj)])
    mi = np.mean([measure_point("mirror", N, 12, 0, t, c, ns)["F"] for t in range(a.traj)])
    print(f"    fwd_pp {fw:.4f}  mirror {mi:.4f}  ratio {mi / fw:.3f}  {'OK' if abs(mi / fw - 1) < 0.08 else 'CHECK'}")
    half = cycles_for(c["gates"][:6], c["colours"], lambda t: c["bnds"][t % len(c["bnds"])])
    measure_point("mirror", N, 12, 0, 0, c, ns)
    want = len(half[-1][1])                  # only the first inverse CZ layer lacks a u before it
    print(f"    unmerged frames {c['big'].n_fold} (fold layer has {want} CZ)  "
          f"{'OK' if c['big'].n_fold == want else 'FAIL'}")
    ok &= c["big"].n_fold == want

    print(f"[4] stochastic model vs per-element Pauli trajectories (r2 rates, no coherent), "
          f"d={a.d4}, {a.traj4} runs")
    print("    (edge layers L: errors of the first/last L noisy cycles sampled, bulk analytic;"
          " L=0 is the pure analytic model)")
    ns = mk("r2", cz_phase=(0.0, 0.0), sx_overrot=(0.0, 0.0))
    ca, ct = ctx_for(ns, False), ctx_for(ns, True)
    for mode in ("fwd_full", "mirror"):
        tr = np.array([measure_point(mode, N, a.d4, 0, t, ct, ns)["F"] for t in range(a.traj4)])
        se_t = tr.std(ddof=1) / np.sqrt(len(tr))
        print(f"    {mode:9s} full trajectories {tr.mean():.4f} +/- {se_t:.4f}")
        pick = None
        for L in range(0, a.max_edge + 1):
            ns.edge_layers = L
            runs = 1 if L == 0 else a.traj4
            hy = np.array([measure_point(mode, N, a.d4, 0, t, ca, ns)["F"] for t in range(runs)])
            se_h = hy.std(ddof=1) / np.sqrt(len(hy)) if len(hy) > 1 else 0.0
            se = np.hypot(se_t, se_h)
            z = (tr.mean() - hy.mean()) / max(se, 1e-12)
            if pick is None and abs(z) < 1:
                pick = L
            print(f"      L={L}  {hy.mean():.4f} +/- {se_h:.4f}   traj - model {z:+.1f} sigma   "
                  f"(model/traj {hy.mean() / tr.mean():.3f})")
        ns.edge_layers = 0
        print(f"      smallest L within 1 sigma: {pick if pick is not None else f'> {a.max_edge}'}")

    print("[5] native compilation and U(theta,phi,lambda) conversion, including the poles")
    rng = np.random.default_rng(3)
    poles = [I2, PX, PY, PZ, rz(0.7), rz(-2.9), rx(np.pi) @ rz(0.3), PX @ rz(1.1), 1j * PZ]
    mats = [haar_su2(rng) for _ in range(200)] + poles
    worst_u, worst_n = 0.0, 0.0
    sx_ideal = rx(np.pi / 2)
    for M in mats:
        th, ph, la = u_angles(M)
        U = np.array([[np.cos(th / 2), -np.exp(1j * la) * np.sin(th / 2)],
                      [np.exp(1j * ph) * np.sin(th / 2), np.exp(1j * (ph + la)) * np.cos(th / 2)]])
        worst_u = max(worst_u, same_up_to_phase(U, M))
        worst_n = max(worst_n, same_up_to_phase(native(M, sx_ideal), M))
    print(f"    u_angles: max 1 - |tr(U^dag M)|/2 = {worst_u:.1e}  {'OK' if worst_u < 1e-10 else 'FAIL'}")
    print(f"    native (ideal SX): max 1 - |tr(N^dag M)|/2 = {worst_n:.1e}  {'OK' if worst_n < 1e-10 else 'FAIL'}")
    ok &= worst_u < 1e-10 and worst_n < 1e-10

    print("[6] partitions: incremental swap score vs recomputed cut")
    prng = np.random.default_rng(11)
    edges6 = sorted({e for es in base.window(N)[1].values() for e in es})
    adj6 = defaultdict(set)
    for u, v in edges6:
        adj6[u].add(v)
        adj6[v].add(u)
    cut6 = lambda lab: sum(lab[u] != lab[v] for u, v in edges6)
    bad6 = 0
    for _ in range(300):
        lab = prng.integers(0, 3, N)
        u, v = prng.choice(N, 2, replace=False)
        if lab[u] == lab[v]:
            continue
        before = cut6(lab)
        dc = dcut(lab, u, v, adj6)
        lab[u], lab[v] = lab[v], lab[u]
        bad6 += (cut6(lab) - before) != dc
    print(f"    mismatches {bad6}  {'OK' if bad6 == 0 else 'FAIL'}")
    ok &= bad6 == 0

    print("[7] ACE mirror on QrackAceBackend")
    try:
        from pyqrack import QrackAceBackend  # noqa: F401
        import qiskit  # noqa: F401
    except ImportError as err:
        print(f"    skipped: {err}")
    else:
        c = ctx_for(mk("ideal"), modes_big=False)
        print(f"    pyqrack {pyqrack_version()}")
        for lr in (N, 4, 2):
            ns = mk("ideal", lrc=lr, lrr=lr, ace_shots=1024)
            rec = measure_point("ace", N, 12, 0, 0, dict(c, ace_cfg="selftest"), ns)
            print(f"    lrc=lrr={lr:<3} bulk/boundary {rec['bulk_to_boundary']:.3g}  "
                  f"fidelity_ace {rec['fidelity_ace']:.4f}  hamming dist {rec['hamming_dist_ace']:.2f}  "
                  f"seam CZ {rec['seam_cz']}")
            if rec["ace_boundary_qubits"] == 0:     # no seams: ACE must be exact, bit order included
                good = rec["fidelity_ace"] > 0.99
                print(f"    boundary-free echo {'OK' if good else 'FAIL'}")
                ok &= good
    print("selftest", "PASSED" if ok else "FAILED")


# ================================================================== CLI
def pair(s):
    x, y = (float(v) for v in s.split(","))
    return x, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seamgap", help="data-only analysis of the released repo")
    s.add_argument("--repo", default=".")
    s.add_argument("--dmin", type=int, default=20)

    def geometry(p):
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--layout", help="data/layout.json of the BlueQubit release")
        g.add_argument("--grid", help="synthetic RxC lattice, e.g. 6x6")

    r = sub.add_parser("run", help="statevector emulation (echo run or scaling sweep)")
    geometry(r)
    r.add_argument("--sizes", default="27-36", help="e.g. 36, 27-36 or 27,30,33,36")
    r.add_argument("--depths", type=int, nargs="+", default=[8, 16, 24, 32, 36])
    r.add_argument("--instances", type=int, default=2)
    r.add_argument("--modes", default="fwd_full,fwd_pp,mirror,patched",
                   help=f"comma list from {MODES}")
    r.add_argument("--patches", type=int, default=3, help="K for the patched estimator and pseudo-patch")
    r.add_argument("--preset", choices=list(PRESETS), default="r2")
    for k in ("cz_rb", "sx_rb", "idle", "ro10", "ro01", "spread"):
        r.add_argument("--" + k.replace("_", "-"), type=float, default=None)
    for k in ("cz_phase", "sx_overrot", "zz_idle"):
        r.add_argument("--" + k.replace("_", "-"), type=pair, default=None, help="mean,std")
    r.add_argument("--no-meas-twirl", dest="meas_twirl", action="store_false")
    r.add_argument("--twirl", choices=["none", "mirror", "all"], default="mirror")
    r.add_argument("--stochastic", choices=["analytic", "trajectory"], default="analytic")
    r.add_argument("--edge-layers", type=int, default=0,
                   help="analytic mode: sample errors of the first/last L noisy cycles (selftest [4] picks L)")
    r.add_argument("--traj", type=int, default=8, help="runs per point when twirled, hybrid or trajectory")
    r.add_argument("--zero-input", action="store_true", help="all-zeros input instead of weight N/2")
    r.add_argument("--lrc", type=int, default=4, help="ace: QrackAceBackend long_range_columns")
    r.add_argument("--lrr", type=int, default=4, help="ace: QrackAceBackend long_range_rows")
    r.add_argument("--ace-torus", action="store_true", help="ace: pass is_torus=True")
    r.add_argument("--ace-shots", type=int, default=None, help="ace: shots (default 2^min(13, N+2))")
    r.add_argument("--cpu", action="store_true", help="is_gpu=False, for big states")
    r.add_argument("--seed", type=int, default=2609)
    r.add_argument("--device-seed", type=int, default=120)
    r.add_argument("--out", default="nighthawk_qrack.jsonl")
    r.add_argument("--summarize", action="store_true")

    t = sub.add_parser("selftest", help="validation checks at small size")
    geometry(t)
    t.add_argument("--n", type=int, default=12)
    t.add_argument("--traj", type=int, default=40)
    t.add_argument("--d4", type=int, default=12, help="depth for the stochastic-model check [4]")
    t.add_argument("--traj4", type=int, default=400, help="runs per point in [4] (needs many: F is near-binary per run)")
    t.add_argument("--max-edge", type=int, default=4, help="largest L tried in [4]")

    a = ap.parse_args()
    {"seamgap": cmd_seamgap, "run": cmd_run, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
