#!/usr/bin/env python3
"""
nighthawk_qrack.py -- clean-qubit PyQrack companion to arXiv:2609.28657
(BlueQubit, random-circuit sampling on IBM Nighthawk r2 / ibm_phoenix, 61 qubits).

No noise model. The simulator stands in for the QPU with ideal gates; the only
error left in any number this script prints is shot noise and, for the ace
backend, QrackAceBackend's own seam approximation. Everything else is aligned
gate-for-gate with the paper and the BlueQubitDev/rcs-nighthawk release:

  circuits   regenerated from data/layout.json exactly as rcs/circuits.py does:
             SplitMix64 hash of (seed, instance, cycle, qubit, k) -> Haar angles,
             applied as rz(phi) rx(theta) rz(lambda); colours A,B,C,D in the fixed
             order; seeds 2025 + k*1000003 (full, patched) and 2025/instance k
             (mirror); `verify` checks every one of the 233 released QASM files.
  patches    the five released K=3 and K=4 partitions (layout.json), not re-searched.
  mirror     forward d/2 cycles pseudo-patched through the five K=3 boundary sets,
             advancing one partition per cycle (rotate_every=1), then the exact
             inverse; the ten released input strings (char q -> logical qubit q);
             shots split evenly over the inputs; optional Pauli-frame gate
             twirling (64 randomisations in the paper). Twirling is a gate identity,
             so it adds no noise: on the ace backend it only randomises ACE's
             coherent seam error so the two mirror halves cannot cancel it.
  patched    Eq. (1)-(2): per-patch linear XEB normalised by its ideal XEB, product
             over patches, second-order delta-method standard error.
  combine    inverse-variance mean over instances; unweighted log10 exponential fit
             of the mirror points (analysis/fidelity_vs_depth.py).

Subcommands
  verify    regenerate all circuits and compare with data/circuits/*.qasm (no Qrack).
  hwxeb     re-score the measured ibm_phoenix bitstrings against PyQrack ideal patch
            distributions and rebuild the paper's F(d) points, fit and collision
            ratios; every number is checked against data/results/*.json.
  run       clean-qubit emulation of the released circuits (--backend exact|ace),
            scored with the same estimators; the summary sets the simulator's
            per-cycle fidelity next to the device's 0.872. --sizes sweeps the
            register (first-n truncation, the release's `e < n` convention),
            resumable per (n, point), and fits b(N) = u*N + v*CZ/cycle to 61.
            Every bitstring drawn is kept: one npz per point under
            <out>_shots/<cfg>/points, merged after each run into .../release/ in the release's data/
            layout, so BlueQubit's analysis scripts read it unchanged. --families full
            draws the 10 x 100k samples of the d36 circuit (stored, not scorable).
            --gpus 0-5 --per-gpu 3 spreads the points over 18 workers through a
            claim-file queue (QRACK_OCL_DEFAULT_DEVICE per worker); a killed worker's
            point is taken over, and the same command resumes.
  selftest  conventions and exactness checks (seconds).
  seamgap   data-only analysis of the release (numpy), unchanged.

Backends
  exact   QrackSimulator. Patched circuits factorise, and QUnit keeps the patches
          separate, so 61-qubit patched circuits run exactly. Mirror circuits
          entangle all 61 qubits after one ABCD sweep: use --n to truncate to the
          first n logical qubits (the release's own `e < n` convention).
  ace     QrackAceBackend, noise=0. The 61 logical qubits are placed at their true
          positions on an 8x8 ACE register (rows 1-8, cols 2-9 of the device), so
          every coupler is nearest-neighbour in ACE's grid; the three dropped sites
          idle in |0>. is_torus=False (the device patch is not a torus).

Environment: no CUDA. OpenCL via QRACK_OCL_DEFAULT_DEVICE, or --cpu.
For big exact states: QRACK_MAX_CPU_QB, QRACK_MAX_ALLOC_MB.

Examples
  python nighthawk_qrack.py verify   --repo rcs-nighthawk
  python nighthawk_qrack.py selftest --repo rcs-nighthawk
  python nighthawk_qrack.py hwxeb    --repo rcs-nighthawk --cpu
  python nighthawk_qrack.py run      --repo rcs-nighthawk --backend ace --families mirror \\
         --depths 4 8 16 24 36 --out clean_ace.jsonl
  python nighthawk_qrack.py run      --repo rcs-nighthawk --backend ace --families mirror \\
         --sizes 27-36,61 --depths 4 6 8 12 --out clean_ace.jsonl
  python nighthawk_qrack.py run      --repo rcs-nighthawk --backend exact --families patched \\
         --depths 20 36 --out clean_exact.jsonl
  python nighthawk_qrack.py run      --repo rcs-nighthawk --summarize --out clean_ace.jsonl
"""

import os

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):
    os.environ["PYQRACK_SHARED_LIB_PATH"] = QRACK_LIB_PATH

import argparse
import collections
import sys
import hashlib
import json
import math
import re
import statistics as st
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ================================================================== constants of the release
BASE_SEED = 2025
INSTANCE_SEED_STRIDE = 1000003
MIRROR_SHOTS = {4: 30_000, 6: 30_000, 8: 30_000, 10: 45_000, 12: 45_000, 14: 45_000, 16: 60_000,
                18: 60_000, 20: 72_000, 24: 96_000, 28: 120_000, 32: 120_000, 36: 240_000, 40: 360_000}
SAMPLE_PUBS, SAMPLE_SHOTS_PER_PUB = 10, 100_000
PATCHED_SHOTS = {20: 2_400, 24: 4_800, 28: 12_000, 32: 24_000, 36: 36_000, 40: 54_000}
GATE_TWIRLS = 64
FULL_DEPTHS = list(range(4, 41, 4))
_MASK64 = (1 << 64) - 1


# ================================================================== angle hash (rcs/circuits.py)
def _splitmix64(x):
    x = (int(x) + 0x9E3779B97F4A7C15) & _MASK64
    z = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def u01_from_key(*parts):
    x = 0
    for p in parts:
        x = _splitmix64(x ^ (int(p) & _MASK64))
    return ((x >> 11) & ((1 << 53) - 1)) / float(1 << 53)


def haar_angles(seed, instance, cycle, qubit):
    """(phi, theta, lambda), applied as rz(phi), rx(theta), rz(lambda)."""
    phi = 2.0 * math.pi * u01_from_key(seed, instance, cycle, qubit, 1)
    z = 1.0 - 2.0 * u01_from_key(seed, instance, cycle, qubit, 2)
    theta = math.acos(min(1.0, max(-1.0, z)))
    lam = 2.0 * math.pi * u01_from_key(seed, instance, cycle, qubit, 3)
    return phi, theta, lam


def instance_seed(instance, kind, base=BASE_SEED):
    return base + instance * INSTANCE_SEED_STRIDE if kind in ("full", "patched") else base


# ================================================================== one-qubit algebra
I2 = np.eye(2, dtype=complex)


def rz(t):
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])


def rx(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def gate(ang):
    phi, th, lam = ang
    return rz(lam) @ rx(th) @ rz(phi)            # matrix of rz(phi) then rx then rz(lam)


def u_angles(M, tol=1e-12):
    """(theta, phi, lambda) of the U gate equal to M up to global phase; pole-safe."""
    c, s = abs(M[0, 0]), abs(M[1, 0])
    th = 2 * np.arctan2(s, c)
    g = np.angle(M[0, 0]) if c > tol else np.angle(M[1, 0])
    ph = np.angle(M[1, 0]) - g if s > tol else 0.0
    la = np.angle(-M[0, 1]) - g if s > tol else np.angle(M[1, 1]) - g - ph
    return float(th), float(ph), float(la)


def flat(m):
    return [complex(x) for x in m.reshape(-1)]


# ================================================================== layout
REPO_NAME = "rcs-nighthawk"
REPO_URL = "https://github.com/BlueQubitDev/rcs-nighthawk"


def _is_repo(p):
    return p is not None and (Path(p) / "data/layout.json").is_file()


def find_repo(repo=None):
    """--repo if given; else $NIGHTHAWK_REPO; else a folder holding data/layout.json, or an
    rcs-nighthawk clone, in the working dir, next to this script, or up to 3 parents up."""
    if repo:
        if _is_repo(repo):
            return Path(repo)
        raise SystemExit(f"--repo {repo}: no data/layout.json there. Clone it with\n  git clone {REPO_URL} {repo}")
    here, script = Path.cwd(), Path(__file__).resolve().parent
    cands = [os.environ.get("NIGHTHAWK_REPO")]
    for base in (here, script):
        for p in [base, *list(base.parents)[:3]]:
            cands += [p, p / REPO_NAME]
    for c in cands:
        if _is_repo(c):
            return Path(c)
    raise SystemExit(f"cannot find the BlueQubit release (data/layout.json). Either clone it next to this script:\n"
                     f"  git clone {REPO_URL} {script / REPO_NAME}\n"
                     f"or pass --repo /path/to/{REPO_NAME}, or set NIGHTHAWK_REPO.")


class Layout:
    def __init__(self, repo):
        self.repo = find_repo(repo)
        L = json.loads((self.repo / "data/layout.json").read_text())
        self.n = L["num_qubits"]
        self.l2p = L["logical_to_physical"]
        self.schedule = L["schedule"]
        self.matchings = {k: [tuple(e) for e in v] for k, v in L["matchings"].items()}
        self.base_seed = L["base_seed"]
        self.instances = L["instances"]
        self.depths = L["depths"]
        self.inputs = L["mirror_input_strings"]
        self.partitions = {}
        for K, plist in L["partitions"].items():
            self.partitions[int(K)] = [dict(
                boundary=frozenset(tuple(sorted(e)) for e in p["boundary_edges"]),
                patches=[sorted(v) for _, v in sorted(p["patch_qubits"].items(), key=lambda kv: int(kv[0]))])
                for p in plist]
        cols = L["device_lattice"]["cols"]
        (r0, r1), (c0, c1) = L["subgrid"]["rows"], L["subgrid"]["cols"]
        self.grid = (r1 - r0 + 1, c1 - c0 + 1)
        self.pos = [(p // cols - r0, p % cols - c0) for p in self.l2p]
        assert self.base_seed == BASE_SEED


def build_cycles(lay, ncyc, seed, inst, removed=frozenset(), rotate=None, n=None):
    """[(angles[q], cz_edges)] per cycle, as build_rcs_circuit with removed_edges /
    make_pseudo_patch_edge_filter(rotate, rotate_every=1)."""
    n = n or lay.n
    out = []
    for c in range(ncyc):
        ang = [haar_angles(seed, inst, c, q) for q in range(n)]
        col = lay.schedule[c % len(lay.schedule)]
        es = [e for e in lay.matchings[col] if e[0] < n and e[1] < n and tuple(sorted(e)) not in removed]
        if rotate:
            cut = rotate[c % len(rotate)]
            es = [e for e in es if tuple(sorted(e)) not in cut]
        out.append((ang, es))
    return out


def mirror_cycles(lay, depth, inst, n=None):
    rot = [p["boundary"] for p in lay.partitions[3]]
    return build_cycles(lay, depth // 2, instance_seed(inst, "mirror"), inst, rotate=rot, n=n)


def patched_cycles(lay, K, depth, j, inst):
    return build_cycles(lay, depth, instance_seed(inst, "patched"), inst,
                        removed=lay.partitions[K][j]["boundary"])


def full_cycles(lay, depth):
    return build_cycles(lay, depth, instance_seed(0, "full"), 0)


# ================================================================== verify (no Qrack)
_OP = re.compile(r"^\s*(rz|rx|cz)\s*(?:\(([^)]*)\))?\s+q\[(\d+)\](?:\s*,\s*q\[(\d+)\])?\s*;", re.M)


def qasm_ops(text):
    out = []
    for g, arg, a, b in _OP.findall(text):
        if g == "cz":
            out.append(("cz", int(a), int(b)))
        else:
            try:
                out.append((g, int(a), float(arg)))
            except ValueError:           # rx(_theta_q_): mirror preparation parameter
                continue
    return out


def ops_forward(cyc):
    out = []
    for ang, es in cyc:
        for q, (phi, th, lam) in enumerate(ang):
            out += [("rz", q, phi), ("rx", q, th), ("rz", q, lam)]
        out += [("cz", a, b) for a, b in es]
    return out


def ops_inverse(cyc):
    out = []
    for g in reversed(ops_forward(cyc)):
        out.append(g if g[0] == "cz" else (g[0], g[1], -g[2]))
    return out


def same_ops(A, B, tol):
    if len(A) != len(B):
        return False, f"length {len(A)} vs {len(B)}"
    for i, (x, y) in enumerate(zip(A, B)):
        if x[0] != y[0] or x[1] != y[1]:
            return False, f"op {i}: {x} vs {y}"
        if x[0] == "cz":
            if x[2] != y[2]:
                return False, f"op {i}: {x} vs {y}"
        elif abs(math.remainder(x[2] - y[2], 2 * math.pi)) > tol:
            return False, f"op {i}: angle {x[2]!r} vs {y[2]!r}"
    return True, ""


def released_circuits(lay):
    for d in lay.depths["mirror"]:
        for i in range(lay.instances):
            cyc = mirror_cycles(lay, d, i)
            yield f"mirror/d{d:02d}_instance{i}.qasm", ops_forward(cyc) + ops_inverse(cyc)
    for K in (3, 4):
        for d in lay.depths["patched"]:
            for j in range(len(lay.partitions[K])):
                for i in range(lay.instances):
                    yield (f"patched/K{K}/d{d}/partition{j}_instance{i}.qasm",
                           ops_forward(patched_cycles(lay, K, d, j, i)))
    for d in FULL_DEPTHS:
        yield f"full/d{d:02d}_logical.qasm", ops_forward(full_cycles(lay, d))


def cmd_verify(a):
    lay = Layout(a.repo)
    root = lay.repo / "data/circuits"
    man = {c["qasm3"]: c for c in json.loads((root / "manifest.json").read_text())["circuits"]}
    n_ok = n_bad = 0
    for rel, mine in released_circuits(lay):
        theirs = qasm_ops((root / rel).read_text())
        ok, why = same_ops(mine, theirs, a.tol)
        ncz = sum(1 for g in mine if g[0] == "cz")
        if ok and rel in man and man[rel]["cz"] != ncz:
            ok, why = False, f"CZ count {ncz} vs manifest {man[rel]['cz']}"
        n_ok += ok
        n_bad += not ok
        if not ok:
            print(f"DIFFERENT {rel}: {why}")
    in_repo = {k for k in man if not k.endswith("_executed.qasm")}
    missed = in_repo - {rel for rel, _ in released_circuits(lay)}
    print(f"{n_ok} identical, {n_bad} different (angle tol {a.tol:g} rad)"
          + (f"; not regenerated: {sorted(missed)}" if missed else ""))
    raise SystemExit(1 if n_bad or missed else 0)


# ================================================================== engines
PAULIS = ("i", "x", "y", "z")


class ExactEngine:
    """QrackSimulator on the logical register."""

    def __init__(self, n, cpu):
        from pyqrack import QrackSimulator
        self.n = n
        self.sim = QrackSimulator(n, is_gpu=not cpu)

    def reset(self):
        self.sim.reset_all()

    def g1(self, q, M):
        self.sim.mtrx(flat(M), q)

    def pauli(self, p, q):
        if p != "i":
            getattr(self.sim, p)(q)

    def cz(self, a, b):
        self.sim.mcz([a], b)

    def shots(self, s):
        return np.array(self.sim.measure_shots(list(range(self.n)), s), dtype=np.uint64)

    def prob_bits(self, bits):
        return float(self.sim.prob_perm(list(range(self.n)), [bool(b) for b in bits]))


class AceEngine:
    """QrackAceBackend on the device geometry: logical q -> its (row, col) of the 8x8
    subgrid, so couplers are nearest-neighbour in ACE's grid. One backend per worker and
    size, reset in place between runs."""

    def __init__(self, lay, n, a):
        self.lay, self.n, self.a = lay, n, a
        R, C = lay.grid
        self.idx = [r * C + c for r, c in lay.pos[:n]]
        self.size = R * C
        from pyqrack import QrackAceBackend
        # ACE on rusticl/Vega10 hung compute rings and forced GPU resets: CPU unless --ace-gpu
        self.sim = QrackAceBackend(self.size, long_range_columns=a.lrc, long_range_rows=a.lrr,
                                   is_torus=False, is_gpu=(not a.cpu) and getattr(a, "ace_gpu", False))
        rl, cl = self.sim.get_row_length(), self.sim.get_column_length()
        if {rl, cl} != {R, C}:
            raise SystemExit(f"ACE chose a {rl}x{cl} grid for {self.size} qubits, expected {R}x{C}")

    def reset(self):
        """Back to |0...0> in place: measure all, flip the ones. No reallocation, so no
        new device buffers or uploads per twirl (checked exact: every P(1) = 0 after)."""
        v = self.sim.m_all()
        for q in range(self.size):
            if (v >> q) & 1:
                self.sim.x(q)

    def g1(self, q, M):
        self.sim.u(self.idx[q], *u_angles(M))

    def pauli(self, p, q):
        if p != "i":
            getattr(self.sim, p)(self.idx[q])

    def cz(self, a, b):
        self.sim.cz(self.idx[a], self.idx[b])

    def shots(self, s):
        return np.array(self.sim.measure_shots(self.idx, s), dtype=np.uint64)

    prob_bits = None


_GATE = {"a": None}


def gated(fn):
    """Run an OpenCL initialisation (first context + JIT, backend allocation) under one
    machine-wide lock, at least --init-gap seconds after the previous one finished, so
    workers never initialise at the same time over the cards' shared PCIe links."""
    a = _GATE["a"]
    if a is None or a.cpu or not a.init_gap:
        return fn()
    import fcntl
    lock = shots_root(a) / "initgate.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            txt = fh.read().strip()
            wait = (float(txt) if txt else 0.0) + a.init_gap - time.time()
            if wait > 0:
                time.sleep(wait)
            out = fn()
            fh.seek(0)
            fh.truncate()
            fh.write(f"{time.time():.3f}")
            fh.flush()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    return out


def make_engine(lay, n, a):
    return gated(lambda: AceEngine(lay, n, a) if a.backend == "ace" else ExactEngine(n, a.cpu))


def cz_layer(eng, es, rng):
    """CZ layer, Pauli-frame twirled when rng is given: P_a x P_b before, and after it
    CZ (P_a x P_b) CZ = (P_a Z^[P_b in XY]) x (P_b Z^[P_a in XY]) up to phase. Exact."""
    for a_, b_ in es:
        if rng is None:
            eng.cz(a_, b_)
            continue
        pa, pb = PAULIS[rng.integers(4)], PAULIS[rng.integers(4)]
        eng.pauli(pa, a_)
        eng.pauli(pb, b_)
        eng.cz(a_, b_)
        eng.pauli(pa, a_)
        eng.pauli(pb, b_)
        if pb in "xy":
            eng.pauli("z", a_)
        if pa in "xy":
            eng.pauli("z", b_)


def apply_forward(eng, cyc, rng=None):
    for ang, es in cyc:
        for q, t in enumerate(ang):
            eng.g1(q, gate(t))
        cz_layer(eng, es, rng)


def apply_inverse(eng, cyc, rng=None):
    for ang, es in reversed(cyc):
        cz_layer(eng, list(reversed(es)), rng)
        for q in reversed(range(len(ang))):
            eng.g1(q, gate(ang[q]).conj().T)


# ================================================================== estimators (rcs/estimators.py)
def extract_bits(shots, qubits):
    shots = np.asarray(shots, dtype=np.uint64)
    out = np.zeros(shots.shape, dtype=np.int64)
    for j, q in enumerate(qubits):
        out |= ((shots >> np.uint64(q)) & np.uint64(1)).astype(np.int64) << j
    return out


def ideal_xeb(p):
    return float(p.size * np.dot(p, p) - 1.0)


def _delta_se(means, cov):
    k = len(means)
    grad = np.array([np.prod(np.delete(means, i)) for i in range(k)])
    hess = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            hess[i, j] = hess[j, i] = np.prod(np.delete(means, [i, j]))
    a = cov @ hess
    return math.sqrt(max(float(grad @ cov @ grad) + 0.5 * float(np.trace(a @ a)), 0.0))


def patched_xeb(shots, probs, patches):
    """Eq. (1)-(2): prod_r mean_s[(2^n_r p_r(x_r) - 1) / (2^n_r sum p_r^2 - 1)]."""
    if len(shots) < 2:
        raise ValueError("need at least two shots")
    norms = [ideal_xeb(p) for p in probs]
    scores = np.array([(p.size * p[extract_bits(shots, qs)] - 1.0) / nz
                       for p, qs, nz in zip(probs, patches, norms)])
    means = scores.mean(axis=1)
    cov = np.atleast_2d(np.cov(scores, bias=True)) / len(shots)
    return dict(fidelity=float(np.prod(means)), se=_delta_se(means, cov),
                patch_fidelity=[float(m) for m in means], patch_ideal_xeb=norms, shots=int(len(shots)))


def inverse_variance_mean(v, se):
    v, se = np.asarray(v, float), np.maximum(np.asarray(se, float), 1e-12)
    w = 1 / se ** 2
    return float(np.sum(w * v) / np.sum(w)), float(math.sqrt(1 / np.sum(w)))


def fit_exp(depths, F):
    s, i = np.polyfit(np.asarray(depths, float), np.log10(np.asarray(F, float)), 1)
    return float(10 ** i), float(10 ** s)


# ================================================================== ideal patch distributions
def patch_probs(cyc, patches, cpu):
    """Exact distribution of every patch sub-circuit (qubit j of patch = sorted[j]),
    little-endian as Qiskit / out_probs; renormalised in float64."""
    from pyqrack import QrackSimulator
    out = []
    for qs in patches:
        loc = {q: j for j, q in enumerate(qs)}
        if _GATE.get("warm"):
            sim = QrackSimulator(len(qs), is_gpu=not cpu)
        else:
            sim = gated(lambda: QrackSimulator(len(qs), is_gpu=not cpu))
            _GATE["warm"] = True
        for ang, es in cyc:
            for q in qs:
                sim.mtrx(flat(gate(ang[q])), loc[q])
            for a_, b_ in es:
                if a_ in loc and b_ in loc:
                    sim.mcz([loc[a_]], loc[b_])
                elif (a_ in loc) != (b_ in loc):
                    raise RuntimeError(f"CZ {a_}-{b_} crosses a patch boundary")
        p = np.asarray(sim.out_probs(), dtype=np.float64)
        out.append(p / p.sum())
        del sim
    return out


def cached_patch_probs(cache, key, cyc, patches, cpu):
    if cache:
        f = Path(cache) / (key.replace("/", "_") + ".npz")
        if f.exists():
            with np.load(f) as z:
                return [z[f"p{r}"].astype(np.float64) for r in range(len(patches))]
    probs = patch_probs(cyc, patches, cpu)
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, **{f"p{r}": p.astype(np.float32) for r, p in enumerate(probs)})
    return probs


# ================================================================== hwxeb: the paper's numbers, Qrack as reference
def cmd_hwxeb(a):
    lay = Layout(a.repo)
    ref = {(r["K"], r["depth"], r["partition"], r["instance"]): r
           for r in json.loads((lay.repo / "data/results/patch_xeb.json").read_text())}
    fvd = json.loads((lay.repo / "data/results/fidelity_vs_depth.json").read_text())
    depths = a.depths or lay.depths["patched"]
    recs, worst_f, worst_i, t0 = [], 0.0, 0.0, time.time()
    for K in a.K:
        for d in depths:
            shots = np.load(lay.repo / f"data/counts/patched_K{K}_d{d}.npz")
            done = []
            for j, part in enumerate(lay.partitions[K]):
                for i in range(lay.instances):
                    if a.limit and len(recs) >= a.limit:
                        break
                    probs = cached_patch_probs(a.cache, f"K{K}_d{d}_p{j}_i{i}", patched_cycles(lay, K, d, j, i),
                                               part["patches"], a.cpu)
                    r = patched_xeb(shots[f"partition{j}_instance{i}"], probs, part["patches"])
                    rr = ref.get((K, d, j, i))
                    if rr:
                        worst_f = max(worst_f, abs(r["fidelity"] - rr["fidelity"]) / max(rr["se"], 1e-12))
                        ideal_ref = sorted(rr["patch_ideal_xeb"][str(k)] for k in range(K))
                        worst_i = max(worst_i, max(abs(x - y) for x, y in zip(sorted(r["patch_ideal_xeb"]), ideal_ref)))
                    rec = dict(K=K, depth=d, partition=j, instance=i, **r)
                    recs.append(rec)
                    done.append(rec)
            if done:
                F, se = inverse_variance_mean([r["fidelity"] for r in done], [r["se"] for r in done])
                cr = np.mean([x for r in done for x in r["patch_ideal_xeb"]])
                pr = fvd["points"][f"{K}-patch"].get(str(d), {})
                print(f"K={K} d={d:>2}: {len(done):>2} circuits  F = {F:.3e} +/- {se:.1e}"
                      f"  (release {pr.get('fidelity', float('nan')):.3e})  collision ratio {cr:.4f}"
                      f"  (release {fvd['collision_ratio'][f'K{K}'].get(str(d), float('nan')):.4f})"
                      f"  [{time.time() - t0:.0f}s]", flush=True)
    print(f"\nworst |F_qrack - F_release| = {worst_f:.3f} sigma; worst |ideal XEB diff| = {worst_i:.2e}"
          f"  (PyQrack float32 wheels agree to ~1e-5; {'OK' if worst_f < 0.05 and worst_i < 1e-3 else 'CHECK'})")

    ms = json.loads((lay.repo / "data/counts/mirror_survival.json").read_text())["depths"]
    pts = {}
    for d, e in ms.items():
        h, s = np.asarray(e["hits"], float).sum(-1), np.asarray(e["shots"], float).sum(-1)
        p = h / s
        pts[int(d)] = inverse_variance_mean(p, np.sqrt(np.maximum(p * (1 - p), 1e-12) / s))
    dd = sorted(pts)
    A, f = fit_exp(dd, [pts[d][0] for d in dd])
    print(f"\nmirror fit from the hardware counts: F(d) = {A:.4f} x {f:.4f}^d "
          f"(release {fvd['fit']['prefactor']:.4f} x {fvd['fit']['fidelity_per_cycle']:.4f}^d); "
          f"F(36) = {A * f ** 36:.3e}; error/qubit/cycle {-math.log(f) / lay.n:.2e}")
    if a.out:
        Path(a.out).write_text(json.dumps(recs, indent=1))
        print("wrote", a.out)


# ================================================================== run: clean-qubit emulation
def cfg_tag(a):
    """Everything but the register size: records of every n share one tag, so a sweep
    can be extended with more sizes and resumes per (n, point)."""
    keys = dict(backend=a.backend, shots=a.shots, twirls=a.twirls,
                lrc=a.lrc if a.backend == "ace" else None, lrr=a.lrr if a.backend == "ace" else None,
                exact_probs=a.exact_probs and a.backend == "exact")
    return a.backend + "-" + hashlib.sha1(json.dumps(keys, sort_keys=True).encode()).hexdigest()[:8]


def parse_sizes(s):
    """'36', '27-36', '27,30,33-36' -> sorted unique sizes."""
    out = set()
    for part in str(s).split(","):
        lo, _, hi = part.strip().partition("-")
        out.update(range(int(lo), int(hi) + 1) if hi else [int(lo)])
    return sorted(out)


def mirror_czpc(lay, n, d):
    """CZ gates per cycle of the truncated pseudo-patched mirror (both halves)."""
    cyc = mirror_cycles(lay, d, 0, n=n)
    return sum(len(es) for _, es in cyc) / max(len(cyc), 1)


def n_shots(a, family, d):
    if a.shots == "paper":
        return (MIRROR_SHOTS if family == "mirror" else PATCHED_SHOTS)[d]
    return int(a.shots)


def run_mirror(lay, eng, d, inst, a):
    """Survival over the ten released inputs, shots split evenly; twirled runs split
    each input's shots further over the randomisations (as SamplerV2 does)."""
    n = eng.n
    cyc = mirror_cycles(lay, d, inst, n=n)
    per_input = max(1, n_shots(a, "mirror", d) // len(lay.inputs))
    R = a.twirls if a.backend == "ace" else 0       # twirling is an identity for exact gates
    hits, shots, keep = [], [], {}
    for s, string in enumerate(lay.inputs):
        bits = [int(c) for c in string[:n]]
        target = sum(b << q for q, b in enumerate(bits))
        h = tot = 0
        pr = 0.0
        got, tw = [], []
        for r in range(max(R, 1)):
            rng = np.random.default_rng([BASE_SEED, d, inst, s, r]) if R else None
            k = per_input // max(R, 1) + (1 if r < per_input % max(R, 1) else 0)
            if k == 0:
                continue
            eng.reset()
            for q, b in enumerate(bits):
                if b:
                    eng.pauli("x", q)
            apply_forward(eng, cyc, rng)
            apply_inverse(eng, cyc, rng)
            if eng.prob_bits and a.exact_probs:
                pr += eng.prob_bits(bits) / max(R, 1)
            else:
                sh = eng.shots(k)
                got.append(sh)
                tw.append(np.full(k, r, dtype=np.uint16))
                h += int(np.count_nonzero(sh == np.uint64(target)))
                tot += k
        if eng.prob_bits and a.exact_probs:
            hits.append(pr)
            shots.append(0)
        else:
            hits.append(h)
            shots.append(tot)
            keep[f"input{s}"] = np.concatenate(got) if got else np.zeros(0, np.uint64)
            keep[f"input{s}_twirl"] = np.concatenate(tw) if tw else np.zeros(0, np.uint16)
    if eng.prob_bits and a.exact_probs:
        return dict(fidelity=float(np.mean(hits)), se=0.0, survival=hits, shots=0)
    p = sum(hits) / sum(shots)
    return dict(fidelity=p, se=math.sqrt(max(p * (1 - p), 1e-12) / sum(shots)),
                hits=hits, shots=int(sum(shots)), shots_per_input=shots, _shots=keep)


def run_patched(lay, eng, K, d, j, inst, a):
    part = lay.partitions[K][j]
    cyc = patched_cycles(lay, K, d, j, inst)
    probs = cached_patch_probs(a.cache, f"K{K}_d{d}_p{j}_i{inst}", cyc, part["patches"], a.cpu)
    eng.reset()
    apply_forward(eng, cyc)
    sh = eng.shots(n_shots(a, "patched", d))
    return dict(**patched_xeb(sh, probs, part["patches"]), _shots={"shots": sh})


def run_full(lay, eng, d, pub, a):
    """One pub of the unpatched forward circuit (release seed 2025, instance 0), as the
    10 x 100k-shot pubs of the 10^6-sample run. Not scorable at 61 qubits: stored only."""
    cyc = build_cycles(lay, d, instance_seed(0, "full"), 0, n=eng.n)
    eng.reset()
    apply_forward(eng, cyc)
    k = SAMPLE_SHOTS_PER_PUB if a.shots == "paper" else int(a.shots)
    sh = eng.shots(k)
    return dict(fidelity=None, se=None, shots=int(k), _shots={"shots": sh})


# ================================================================== bitstring storage
def shots_root(a):
    """<shots-dir or out stem_shots>/<cfg tag>: configurations never mix in one release/."""
    base = Path(a.shots_dir) if a.shots_dir else Path(a.out).with_name(Path(a.out).stem + "_shots")
    return base / cfg_tag(a)


def point_file(a, n, fam, K, d, j, i):
    return shots_root(a) / "points" / f"n{n}" / f"{fam}_K{K}_d{d:02d}_p{j}_i{i}.npz"


def save_point(path, arrays):
    """Atomic write, so an interrupted run never leaves a half file behind a record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in arrays.items()})
    os.replace(tmp, path)


def pack_shots(recs, lay, a):
    """Merge the per-point files into the release's data/ layout under <shots>/release:
    counts/patched_K{K}_d{d}.npz (partition{j}_instance{i}, sorted uint64),
    counts/mirror_survival.json (hits[i][s] / shots[i][s]) plus mirror_shots.npz with every
    raw mirror shot and its twirl index, samples/full_d{d}.npz (shots, sorted).
    Truncated registers get an _n{n} suffix; bit q is logical qubit q throughout."""
    rel = shots_root(a) / "release"
    groups, missing = defaultdict(dict), 0
    for (n, fam, K, d, j, i), r in sorted(recs.items(), key=lambda kv: kv[0]):
        f = point_file(a, n, fam, K, d, j, i)
        if not f.exists():
            missing += fam != "mirror" or not a.exact_probs or a.backend == "ace"
            continue
        groups[(n, fam, K, d)][(j, i)] = (f, r)
    written = []
    for (n, fam, K, d), pts in groups.items():
        sfx = "" if n == lay.n else f"_n{n}"
        if fam == "patched":
            out = rel / "counts" / f"patched_K{K}_d{d}{sfx}.npz"
            arrays = {}
            for (j, i), (f, _) in pts.items():
                with np.load(f) as z:
                    arrays[f"partition{j}_instance{i}"] = np.sort(z["shots"].astype(np.uint64))
        elif fam == "full":
            out = rel / "samples" / f"full_d{d}{sfx}.npz"
            parts = []
            for _, (f, _) in sorted(pts.items()):
                with np.load(f) as z:
                    parts.append(z["shots"].astype(np.uint64))
            arrays = {"shots": np.sort(np.concatenate(parts))}
        else:
            out = rel / "counts" / f"mirror_shots_d{d:02d}{sfx}.npz"
            arrays = {}
            for (_, i), (f, _) in pts.items():
                with np.load(f) as z:
                    arrays.update({f"instance{i}_{k}": z[k] for k in z.files})
        save_point(out, arrays)
        written.append(out)
    mirror = defaultdict(dict)
    for (n, fam, K, d), pts in groups.items():
        if fam == "mirror":
            for (_, i), (_, r) in sorted(pts.items()):
                mirror[n].setdefault(str(d), {"instances": [], "hits": [], "shots": []})
                e = mirror[n][str(d)]
                e["instances"].append(i)
                e["hits"].append(r["hits"])
                e["shots"].append(r["shots_per_input"])
    for n, depths in mirror.items():
        sfx = "" if n == lay.n else f"_n{n}"
        out = rel / "counts" / f"mirror_survival{sfx}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"description": "Clean-qubit emulation (" + a.backend + "): hits[i][s] of "
                                   "shots[i][s] executions of instance instances[i] prepared in input string s "
                                   "returned that string. Depth = cycles of U U^dagger.", "depths": depths}, indent=1))
        written.append(out)
    if written:
        print(f"# bitstrings: {len(written)} release-format files under {rel}")
    if missing:
        print(f"# note: {missing} records have no stored bitstrings (written before storage existed)")


def record_files(out):
    """The main --out file plus every worker's <stem>.w<pid>.jsonl beside it."""
    o = Path(out)
    return [o] + sorted(o.parent.glob(o.stem + ".w*.jsonl"))


def load_jsonl(path, tag, n_default):
    recs = {}
    for f in record_files(path) if path else []:
        if not f.exists():
            continue
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:          # a line cut by a killed worker
                continue
            if r.get("cfg") == tag:
                r.setdefault("n", n_default)
                recs[(r["n"], r["family"], r["K"], r["depth"], r["partition"], r["instance"])] = r
    return recs


# ================================================================== work queue (claim files)
def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def claim_file(a, key):
    n, fam, K, d, j, i = key
    return shots_root(a) / "claims" / f"n{n}_{fam}_K{K}_d{d:02d}_p{j}_i{i}"


def _write_atomic(path, text):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def try_claim(path, max_attempts=2, retry_failed=False):
    """O_EXCL claim holding 'pid:attempt'. 'done' claims are final. A claim whose worker
    died (OOM kill, memory cap, segfault) is taken over, up to max_attempts workers in
    total; after that the point is marked 'failed: ...' so the remaining workers do not
    crash on it one after another. --retry-failed clears that."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}:{getattr(try_claim, 'next_attempt', 1)}".encode())
            os.close(fd)
            try_claim.next_attempt = 1
            return True
        except FileExistsError:
            pass
        try:
            txt = path.read_text().strip()
        except FileNotFoundError:
            continue
        if txt == "done":
            return False
        if txt.startswith("failed"):
            if not retry_failed:
                return False
            att = 0
        else:
            pid_s, _, att_s = txt.partition(":")
            if not pid_s.isdigit():
                return False                        # being written right now
            if _pid_alive(int(pid_s)):
                return False
            att = int(att_s) if att_s.isdigit() else 1
            if att >= max_attempts:
                _write_atomic(path, f"failed: {att} worker(s) died on this point (memory cap, OOM or crash)")
                return False
        try:
            os.rename(path, path.with_name(path.name + f".stale{os.getpid()}"))
        except FileNotFoundError:
            pass
        try_claim.next_attempt = att + 1
    return False


def failed_points(a):
    root = shots_root(a) / "claims"
    out = []
    if root.exists():
        for f in sorted(root.iterdir()):
            if f.is_file() and "." not in f.name:
                try:
                    t = f.read_text().strip()
                except OSError:
                    continue
                if t.startswith("failed"):
                    out.append((f.name, t))
    return out


def mark_done(path):
    _write_atomic(path, "done")


def _point(v):
    """Inverse-variance mean of (F, se) pairs; plain mean if any se is 0 (exact probs)."""
    if all(s > 0 for _, s in v):
        return inverse_variance_mean(*zip(*v))
    return float(np.mean([x for x, _ in v])), 0.0


def _decay(pts, floor):
    """Per-cycle decay b = -dlnF/dd from a log-linear fit over depths with F > floor.
    Returns (b or None, prefactor, number of depths dropped)."""
    good = [(d, F) for d, F in sorted(pts.items()) if F > floor]
    dropped = len(pts) - len(good)
    if len(good) < 2:
        return None, None, dropped
    A, f = fit_exp([d for d, _ in good], [F for _, F in good])
    return -math.log(f), A, dropped


def report_failed(a):
    bad = failed_points(a)
    if bad:
        print(f"\n# {len(bad)} point(s) marked failed (rerun with --retry-failed to try again):")
        for name, why in bad:
            print(f"#   {name}: {why}")


def summarize(recs, lay, a):
    report_failed(a)
    fvd = json.loads((lay.repo / "data/results/fidelity_vs_depth.json").read_text())
    by = defaultdict(list)
    min_shots = {}
    n_full = defaultdict(int)
    for r in recs.values():
        if r["family"] == "full":
            n_full[(r["n"], r["depth"])] += r["shots"]
            continue
        fam = "mirror" if r["family"] == "mirror" else f"{r['K']}-patch"
        by[(r["n"], fam, r["depth"])].append((r["fidelity"], r["se"]))
        if r["family"] == "mirror" and r.get("shots"):
            min_shots[r["n"]] = min(min_shots.get(r["n"], r["shots"]), r["shots"])
    for (n, d), k in sorted(n_full.items()):
        print(f"full circuit n={n} d={d}: {k:,} stored samples (no score: not verifiable at this size)")
    sizes = sorted({k[0] for k in by})
    b_rows, fit_depths = [], set()
    for n in sizes:
        fams = [f for f in ("mirror", "3-patch", "4-patch") if any(k[0] == n and k[1] == f for k in by)]
        full = n == lay.n
        print(f"\n== clean-qubit emulation ({a.backend}), n = {n}"
              + (" -- the experiment's register, compared with ibm_phoenix ==" if full else " (first-n truncation) =="))
        print("family    d   F_sim" + ("            F_hardware     F_sim/F_hw" if full else ""))
        mpts = {}
        for fam in fams:
            for d in sorted(k[2] for k in by if k[0] == n and k[1] == fam):
                F, se = _point(by[(n, fam, d)])
                if fam == "mirror":
                    mpts[d] = F
                hw = fvd["points"][fam].get(str(d), {}).get("fidelity") if full else None
                print(f"{fam:8s} {d:>3}  {F:.5f}+/-{se:.1e}  " + (f"{hw:.3e}      {F / hw:10.1f}" if hw else ""))
        floor = 3.0 / min_shots[n] if n in min_shots else 0.0
        b, A, dropped = _decay(mpts, floor)
        if b is not None:
            czpc = float(np.mean([mirror_czpc(lay, n, d) for d in mpts]))
            fit_depths.update(mpts)
            b_rows.append((n, czpc, b, dropped))
            print(f"mirror fit F(d) = {A:.4f} x {math.exp(-b):.5f}^d"
                  + (f"  vs device {fvd['fit']['prefactor']:.4f} x {fvd['fit']['fidelity_per_cycle']:.4f}^d" if full else "")
                  + (f"  ({dropped} depth(s) at or below 3/shots dropped)" if dropped else ""))

    if not b_rows:
        return
    b_dev = -math.log(fvd["fit"]["fidelity_per_cycle"])
    print(f"\n== per-cycle decay b(N) of the {a.backend} simulator (mirror) ==")
    print("N    CZ/cycle   b(N)       per-cycle F   err/qubit/cycle")
    for n, c, b, dr in b_rows:
        print(f"{n:<4} {c:8.2f}   {b:9.5f}{'*' if dr else ' '} {math.exp(-b):10.5f}    {b / n:.2e}")
    print(f"device (ibm_phoenix, N=61): b = {b_dev:.5f}, per-cycle {math.exp(-b_dev):.4f}, err/qubit/cycle {b_dev / lay.n:.2e}")
    if any(dr for *_, dr in b_rows):
        print("(* fitted with deep depths dropped at or below 3/shots: shallow depths only)")
    if a.backend == "exact":
        print("(exact backend: b = 0 up to shot noise by construction -- a pipeline control)")
        return
    if len(b_rows) >= 2:
        X = np.array([[n, c] for n, c, _, _ in b_rows])
        y = np.array([b for _, _, b, _ in b_rows])
        (u, v), *_ = np.linalg.lstsq(X, y, rcond=None)
        if u < 0 or v < 0:                      # non-negative: best single-term fit
            cands = []
            for col in (0, 1):
                cj = max(float(X[:, col] @ y / (X[:, col] @ X[:, col])), 0.0)
                cands.append((float(np.sum((y - cj * X[:, col]) ** 2)), col, cj))
            _, col, cj = min(cands)
            u, v = (cj, 0.0) if col == 0 else (0.0, cj)
        c61 = float(np.mean([mirror_czpc(lay, lay.n, d) for d in sorted(fit_depths)]))
        b61 = u * lay.n + v * c61
        meas = next((b for n, _, b, _ in b_rows if n == lay.n), None)
        print(f"\nfit b = u*N + v*CZpc (non-negative): u {u:.3e}/qubit, v {v:.3e}/CZ"
              "  (N and CZ/cycle are nearly collinear: trust b(61), not u and v separately)")
        print(f"at N=61 ({c61:.2f} CZ/cycle): b {b61:.5f}, per-cycle {math.exp(-b61):.5f}, "
              f"F(36) {math.exp(-36 * b61):.3e}" + (f"; measured at 61: b {meas:.5f}" if meas is not None else "")
              + f";  sim/device decay ratio {b61 / b_dev:.2f} ({'cleaner' if b61 < b_dev else 'noisier'} than ibm_phoenix)")
        if len(b_rows) < 5 or max(n for n, *_ in b_rows) - min(n for n, *_ in b_rows) < 6:
            print("(few or narrowly spaced sizes: the 61-qubit numbers are indicative only)")


_WATCH = {"claim": None}


def start_rss_watchdog(limit_gb, period=2.0):
    """Real memory cap: a daemon thread reads this process's resident set from
    /proc/self/status; above limit_gb it marks the point being worked on as failed and
    exits, before the kernel OOM killer picks a victim. Unlike ulimit -v it does not
    count the address space Qrack's thread pool reserves but never touches."""
    import threading

    def rss_gb():
        try:
            for line in open("/proc/self/status"):
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1048576
        except OSError:
            pass
        return 0.0

    def loop():
        while True:
            time.sleep(period)
            r = rss_gb()
            if r > limit_gb:
                cf = _WATCH["claim"]
                msg = f"failed: resident memory {r:.2f} GB above --max-rss-gb {limit_gb:g}"
                if cf is not None:
                    try:
                        _write_atomic(cf, msg)
                    except OSError:
                        pass
                print(f"# {msg}; worker exits", flush=True)
                os._exit(3)

    threading.Thread(target=loop, daemon=True).start()


def parse_gpus(s):
    return parse_sizes(s) if s else []


def build_jobs(lay, a, sizes, fams):
    insts = range(a.instances if a.instances is not None else lay.instances)
    jobs = []
    for n in sizes:
        for fam in fams:
            if fam == "patched" and n != lay.n:
                continue
            ds = a.depths or lay.depths[{"mirror": "mirror", "patched": "patched", "full": "full_sampled"}[fam]]
            for d in ds:
                if fam == "full":
                    jobs += [(n, "full", 0, d, p, 0) for p in range(a.pubs)]
                elif fam == "mirror":
                    jobs += [(n, "mirror", 0, d, 0, i) for i in insts]
                else:
                    for K in a.K:
                        parts = range(len(lay.partitions[K])) if a.partitions is None else range(a.partitions)
                        jobs += [(n, "patched", K, d, j, i) for j in parts for i in insts]
    # longest first, so the big points do not end up as a lone tail
    return sorted(jobs, key=lambda k: -(k[0] * k[3] * (MIRROR_SHOTS.get(k[3], 1) if k[1] == "mirror" else 1)))


def _strip_launch_args(argv):
    out, skip = [], False
    for t in argv:
        if skip:
            skip = False
            continue
        if t in ("--gpus", "--per-gpu"):
            skip = True
            continue
        if t.startswith("--gpus=") or t.startswith("--per-gpu=") or t == "--retry-failed":
            continue                                # the launcher clears failed marks once
        out.append(t)
    return out


def launch(a, lay, tag, jobs, recs):
    """Spawn len(gpus) x per_gpu workers (QRACK_OCL_DEFAULT_DEVICE per worker), watch the
    record files, then pack the bitstrings and summarise once they are all done."""
    import subprocess
    gpus = parse_gpus(a.gpus)
    logdir = Path(a.out).with_name(Path(a.out).stem + "_logs")
    logdir.mkdir(parents=True, exist_ok=True)
    child = [sys.executable, os.path.abspath(__file__)] + _strip_launch_args(sys.argv[1:]) + ["--worker"]
    procs = []
    order = [(g, k) for k in range(a.per_gpu) for g in gpus]      # round-robin over the cards
    print(f"# launching {len(order)} workers (GPUs {gpus} x {a.per_gpu}), one every {a.stagger:g} s; "
          f"OpenCL inits serialised {a.init_gap:g} s apart; logs in {logdir}/", flush=True)
    for idx, (g, k) in enumerate(order):
        if idx:
            time.sleep(a.stagger)
        env = dict(os.environ, QRACK_OCL_DEFAULT_DEVICE=str(g))
        log = open(logdir / f"gpu{g}_job{k}.txt", "a")
        procs.append((g, k, subprocess.Popen(child, env=env, stdout=log, stderr=subprocess.STDOUT)))
        print(f"#   started gpu{g} job{k}", flush=True)
    total = len(jobs)
    t0, last, respawns = time.time(), -1, 0

    def open_points():
        """Points neither recorded nor marked done/failed: still worth a worker."""
        have = load_jsonl(a.out, tag, lay.n)
        out = []
        for k in jobs:
            if k in have:
                continue
            try:
                t = claim_file(a, k).read_text().strip()
            except OSError:
                t = ""
            if t == "done" or t.startswith("failed"):
                continue
            out.append(k)
        return out

    try:
        while True:
            alive = [p for *_, p in procs if p.poll() is None]
            # a worker the memory watchdog stopped (exit 3) is replaced while points remain;
            # other exits (crash loops, 3 failures in a row) are not
            for idx, (g, k, p) in enumerate(procs):
                if p.poll() == 3 and respawns < total and open_points():
                    env = dict(os.environ, QRACK_OCL_DEFAULT_DEVICE=str(g))
                    log = open(logdir / f"gpu{g}_job{k}.txt", "a")
                    procs[idx] = (g, k, subprocess.Popen(child, env=env, stdout=log, stderr=subprocess.STDOUT))
                    respawns += 1
                    print(f"#   gpu{g} job{k} stopped by the memory watchdog; replacement started", flush=True)
                    alive.append(procs[idx][2])
            if not alive:
                break
            time.sleep(30)
            done = sum(1 for k in jobs if k in load_jsonl(a.out, tag, lay.n))
            if done != last:
                el = time.time() - t0
                new = done - len([k for k in jobs if k in recs])
                eta = f", ~{(total - done) * el / new / 3600:.1f} h left" if new > 0 else ""
                print(f"# {done}/{total} points done, {sum(p.poll() is None for *_, p in procs)} workers alive"
                      f" [{el / 60:.0f} min{eta}]", flush=True)
                last = done
    except KeyboardInterrupt:
        print("# stopping workers (finished points are kept; rerun the same command to resume)")
        for *_, p in procs:
            p.terminate()
        raise SystemExit(130)
    bad = [(g, k, p.returncode) for g, k, p in procs if p.returncode and p.returncode != 3]
    if bad:
        print(f"# workers with errors (see logs): {bad}")
    recs = load_jsonl(a.out, tag, lay.n)
    pack_shots(recs, lay, a)
    summarize(recs, lay, a)


def cmd_run(a):
    lay = Layout(a.repo)
    tag = cfg_tag(a)
    recs = load_jsonl(a.out, tag, lay.n)
    if a.summarize or a.pack:
        if a.pack:
            pack_shots(recs, lay, a)
        if a.summarize:
            summarize(recs, lay, a)
        return
    if a.sizes and a.n:
        raise SystemExit("give --sizes or --n, not both")
    sizes = parse_sizes(a.sizes) if a.sizes else [a.n or lay.n]
    if not all(2 <= n <= lay.n for n in sizes):
        raise SystemExit(f"sizes must lie in 2..{lay.n}")
    fams = a.families.split(",")
    jobs = build_jobs(lay, a, sizes, fams)
    todo = [k for k in jobs if k not in recs]
    if a.retry_failed and not a.worker:            # once, here; workers never retry failed points
        cleared = 0
        for k in todo:
            cf = claim_file(a, k)
            try:
                if cf.read_text().strip().startswith("failed"):
                    os.rename(cf, cf.with_name(cf.name + f".stale{os.getpid()}"))
                    cleared += 1
            except OSError:
                pass
        print(f"# --retry-failed: {cleared} failed point(s) cleared for another attempt", flush=True)
    if not a.worker:
        if a.backend == "exact" and a.exact_probs and "mirror" in fams:
            print("# note: exact mirror reads survival with prob_perm, so it draws no bitstrings; "
                  "--no-exact-probs samples (and stores) them")
        if "patched" in fams and any(n != lay.n for n in sizes):
            print("# note: patched circuits use the released 61-qubit partitions; run only at n = 61")
        print(f"# cfg {tag}: backend {a.backend}, sizes {sizes}, shots {a.shots}, "
              f"twirls {a.twirls if a.backend == 'ace' else 0}"
              f"{', lrc/lrr ' + str(a.lrc) + '/' + str(a.lrr) if a.backend == 'ace' else ''}")
        print(f"# {len(jobs) - len(todo)} of {len(jobs)} points already done", flush=True)
        if a.gpus:
            if todo:
                return launch(a, lay, tag, jobs, recs)
            return summarize(recs, lay, a)
    out = Path(a.out)
    out_file = out.with_name(f"{out.stem}.w{os.getpid()}.jsonl") if a.worker else out
    fh = open(out_file, "a")
    _GATE["a"] = a
    if a.max_rss_gb:
        start_rss_watchdog(a.max_rss_gb)
    engines, fails, remaining = {}, 0, list(todo)
    while remaining:
        cur = next(iter(engines), None)          # stay on the current size: no re-initialisation
        key = next((k for k in remaining if k[0] == cur), remaining[0])
        remaining.remove(key)
        n, fam, K, d, j, i = key
        cf = claim_file(a, key)
        if not try_claim(cf, a.max_attempts):
            continue
        if key in load_jsonl(a.out, tag, lay.n):   # finished by another worker since we started
            mark_done(cf)
            continue
        _WATCH["claim"] = cf
        if n not in engines:
            engines.clear()                         # one engine alive per worker
            try:
                engines[n] = make_engine(lay, n, a)
            except Exception as err:                # e.g. exact backend beyond QRACK_MAX_CPU_QB
                print(f"# n = {n}: cannot build the {a.backend} engine ({err}); skipped", flush=True)
                cf.unlink(missing_ok=True)
                fails += 1
                if a.worker and fails >= 3:
                    raise SystemExit(f"# worker stops after {fails} failures in a row (GPU state?)")
                continue
        eng = engines[n]
        t0 = time.time()
        try:
            r = (run_mirror(lay, eng, d, i, a) if fam == "mirror" else
                 run_full(lay, eng, d, j, a) if fam == "full" else run_patched(lay, eng, K, d, j, i, a))
        except Exception as err:                    # deterministic (e.g. QRACK_MAX_CPU_QB): do not retry
            print(f"# n = {n} {fam} d{d} inst {i}: failed ({err}); marked failed", flush=True)
            _write_atomic(cf, f"failed: {type(err).__name__}: {err}"[:500])
            engines.clear()                         # do not reuse a backend that failed mid-run
            fails += 1
            if a.worker and fails >= 3:
                raise SystemExit(f"# worker stops after {fails} failures in a row (GPU state?)")
            continue
        fails = 0
        arrays = r.pop("_shots", None)
        if arrays:                                  # bitstrings first, record second, claim last
            f = point_file(a, n, fam, K, d, j, i)
            save_point(f, arrays)
            r["shots_file"] = str(f)
        rec = dict(cfg=tag, n=n, family=fam, K=K, depth=d, partition=j, instance=i,
                   seconds=round(time.time() - t0, 2), **r)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
        mark_done(cf)
        recs[key] = rec
        score = f"F {r['fidelity']:.5f} +/- {r['se']:.1e}" if r["fidelity"] is not None else f"{r['shots']:,} samples"
        print(f"n {n:>2} {fam:7s} K{K} d{d:>3} part {j} inst {i}  {score}  ({rec['seconds']}s)", flush=True)
    if a.worker:
        return
    recs = load_jsonl(a.out, tag, lay.n)
    pack_shots(recs, lay, a)
    summarize(recs, lay, a)


# ================================================================== selftest
def cmd_selftest(a):
    from pyqrack import QrackSimulator
    lay = Layout(a.repo)
    ok = True

    def report(name, good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"{name}: {msg}  {'OK' if good else 'FAIL'}")

    rel = "mirror/d08_instance1.qasm"
    cyc = mirror_cycles(lay, 8, 1)
    good, why = same_ops(ops_forward(cyc) + ops_inverse(cyc),
                         qasm_ops((lay.repo / "data/circuits" / rel).read_text()), 1e-9)
    report("[1] angle hash + pseudo-patch rotation vs released " + rel, good, why or "identical")

    th, ph, la = 0.7, 1.3, -2.1
    s1, s2 = QrackSimulator(1, is_gpu=False), QrackSimulator(1, is_gpu=False)
    s1.u(0, th, ph, la)
    M = gate(haar_angles(BASE_SEED, 0, 0, 0))
    s2.mtrx(flat(M), 0)
    s1.reset_all()
    s1.u(0, *u_angles(M))
    ov = abs(np.vdot(np.array(s1.out_ket()), np.array(s2.out_ket())))
    report("[2] u_angles -> Qrack u == Haar matrix", abs(1 - ov) < 1e-5, f"overlap {ov:.7f}")

    ns = argparse.Namespace(backend="exact", cpu=True, exact_probs=True, shots="1000", twirls=0, n=a.n)
    eng = ExactEngine(a.n, True)
    F = run_mirror(lay, eng, 12, 0, ns)["fidelity"]
    report(f"[3] exact mirror, first {a.n} logical qubits, d=12, 10 inputs", abs(1 - F) < 1e-4, f"F = {F:.7f}")

    eng.reset()
    rng = np.random.default_rng(3)
    bits = [int(c) for c in lay.inputs[0][:a.n]]
    for q, b in enumerate(bits):
        if b:
            eng.pauli("x", q)
    cyc = mirror_cycles(lay, 12, 0, n=a.n)
    apply_forward(eng, cyc, rng)
    apply_inverse(eng, cyc, rng)
    F = eng.prob_bits(bits)
    report("[4] Pauli-frame twirl is an identity (exact)", abs(1 - F) < 1e-4, f"F = {F:.7f}")

    K, d, j, i = 3, 20, 0, 0
    part = lay.partitions[K][j]
    probs = patch_probs(patched_cycles(lay, K, d, j, i), part["patches"], True)
    rr = next(r for r in json.loads((lay.repo / "data/results/patch_xeb.json").read_text())
              if (r["K"], r["depth"], r["partition"], r["instance"]) == (K, d, j, i))
    mine, theirs = sorted(ideal_xeb(p) for p in probs), sorted(rr["patch_ideal_xeb"].values())
    diff = max(abs(x - y) for x, y in zip(mine, theirs))
    report("[5] PyQrack patch ideal XEB vs release (Aer, double)", diff < 1e-3, f"max diff {diff:.1e}")

    shots = np.load(lay.repo / f"data/counts/patched_K{K}_d{d}.npz")[f"partition{j}_instance{i}"]
    r = patched_xeb(shots, probs, part["patches"])
    report("[6] Eq.(1)-(2) on the hardware counts vs release", abs(r["fidelity"] - rr["fidelity"]) < 0.05 * rr["se"],
           f"{r['fidelity']:.5f} vs {rr['fidelity']:.5f} (se {rr['se']:.1e})")

    samp = np.zeros(4000, dtype=np.uint64)
    g = np.random.default_rng(9)
    for qs, p in zip(part["patches"], probs):
        x = g.choice(p.size, size=samp.size, p=p)
        for jj, q in enumerate(qs):
            samp |= ((x >> jj) & 1).astype(np.uint64) << np.uint64(q)
    r = patched_xeb(samp, probs, part["patches"])
    report("[7] ideal samples score F = 1 within 4 se", abs(1 - r["fidelity"]) < 4 * r["se"],
           f"F = {r['fidelity']:.4f} +/- {r['se']:.4f}")

    try:
        ae = AceEngine(lay, lay.n, argparse.Namespace(lrc=a.lrc, lrr=a.lrr, cpu=True))
        print(f"[8] ACE register {ae.size} qubits, grid {ae.sim.get_row_length()}x{ae.sim.get_column_length()}, "
              f"is_torus=False; logical->ACE map is nearest-neighbour for all couplers: "
              f"{all(abs(lay.pos[u][0] - lay.pos[v][0]) + abs(lay.pos[u][1] - lay.pos[v][1]) == 1 for es in lay.matchings.values() for u, v in es)}")
    except ImportError as err:
        print(f"[8] ACE not available: {err}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


# ================================================================== seamgap (numpy only, unchanged)
def cmd_seamgap(a):
    MED_CZ = 1.9e-3 * 5 / 4          # RB -> Pauli channel, (d+1)/d with d = 4
    MED_SX = 2.5e-4 * 3 / 2          # d = 2
    repo = find_repo(a.repo)
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


# ================================================================== CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seamgap", help="data-only analysis of the released repo")
    s.add_argument("--repo", default=None, help="rcs-nighthawk clone (default: auto-detect)")
    s.add_argument("--dmin", type=int, default=20)

    v = sub.add_parser("verify", help="regenerate all circuits and compare with the released QASM")
    v.add_argument("--repo", default=None, help="rcs-nighthawk clone (default: auto-detect)")
    v.add_argument("--tol", type=float, default=1e-9, help="angle tolerance, rad")

    h = sub.add_parser("hwxeb", help="re-score the ibm_phoenix bitstrings with PyQrack ideal patches")
    h.add_argument("--repo", default=None, help="rcs-nighthawk clone (default: auto-detect)")
    h.add_argument("--K", type=int, nargs="+", default=[3, 4])
    h.add_argument("--depths", type=int, nargs="+", default=None)
    h.add_argument("--limit", type=int, default=0, help="stop after this many circuits (0 = all 180)")
    h.add_argument("--cache", default=None, help="directory for ideal patch distributions (float32 npz)")
    h.add_argument("--cpu", action="store_true")
    h.add_argument("--out", default=None, help="write per-circuit records (patch_xeb.json format)")

    r = sub.add_parser("run", help="clean-qubit emulation of the released circuits")
    r.add_argument("--repo", default=None, help="rcs-nighthawk clone (default: auto-detect)")
    r.add_argument("--backend", choices=["exact", "ace"], default="ace")
    r.add_argument("--families", default="mirror,patched",
                   help="comma list of mirror, patched, full (full: unscored samples of the d36 circuit)")
    r.add_argument("--pubs", type=int, default=SAMPLE_PUBS, help="full: pubs of --shots each (paper 10 x 100k)")
    r.add_argument("--shots-dir", dest="shots_dir", default=None,
                   help="where bitstrings go (default: <out stem>_shots/); a <cfg> subfolder holds points/ + release/")
    r.add_argument("--pack", action="store_true", help="only rebuild the release-format files from stored points")
    r.add_argument("--depths", type=int, nargs="+", default=None, help="default: the release's depths")
    r.add_argument("--K", type=int, nargs="+", default=[3, 4])
    r.add_argument("--instances", type=int, default=None, help="default 3, as released")
    r.add_argument("--partitions", type=int, default=None, help="use the first j partitions (default all 5)")
    r.add_argument("--shots", default="paper",
                   help="'paper' (Appendix D budgets; full: 100k per pub) or shots per circuit / pub")
    r.add_argument("--twirls", type=int, default=GATE_TWIRLS, help="ace mirror Pauli-frame randomisations")
    r.add_argument("--exact-probs", dest="exact_probs", action=argparse.BooleanOptionalAction, default=True,
                   help="exact backend mirror: read survival with prob_perm instead of sampling")
    r.add_argument("--sizes", default=None,
                   help="register sizes to sweep, first-n truncation: 36, 27-36 or 20,27-36 (mirror; patched at 61 only)")
    r.add_argument("--n", type=int, default=None, help="single size, same as --sizes n")
    r.add_argument("--lrc", type=int, default=4, help="ace: long_range_columns")
    r.add_argument("--lrr", type=int, default=4, help="ace: long_range_rows")
    r.add_argument("--cache", default=None, help="directory for ideal patch distributions")
    r.add_argument("--cpu", action="store_true", help="is_gpu=False everywhere")
    r.add_argument("--ace-gpu", dest="ace_gpu", action="store_true",
                   help="run QrackAceBackend on OpenCL too (off by default: it wedged Vega10/rusticl cards)")
    r.add_argument("--out", default="nighthawk_clean.jsonl")
    r.add_argument("--summarize", action="store_true")
    r.add_argument("--gpus", default=None,
                   help="launch workers on these OpenCL devices, e.g. 0-5 or 0,2,4; pairs with --per-gpu")
    r.add_argument("--per-gpu", dest="per_gpu", type=int, default=3, help="workers per GPU (default 3)")
    r.add_argument("--stagger", type=float, default=20.0, help="seconds between worker starts (default 20)")
    r.add_argument("--init-gap", dest="init_gap", type=float, default=10.0,
                   help="min seconds between any two OpenCL initialisations, machine-wide (default 10; 0 = off)")
    r.add_argument("--max-attempts", dest="max_attempts", type=int, default=2,
                   help="workers that may die on one point before it is marked failed (default 2)")
    r.add_argument("--retry-failed", dest="retry_failed", action="store_true",
                   help="try points marked failed again")
    r.add_argument("--max-rss-gb", dest="max_rss_gb", type=float, default=0,
                   help="per-worker resident-memory cap: above it the point is marked failed and the worker "
                        "exits (0 = off). Use this instead of ulimit -v or QRACK_MAX_CPU_QB")
    r.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    t = sub.add_parser("selftest", help="conventions and exactness checks")
    t.add_argument("--repo", default=None, help="rcs-nighthawk clone (default: auto-detect)")
    t.add_argument("--n", type=int, default=14, help="register for the exact mirror checks")
    t.add_argument("--lrc", type=int, default=4)
    t.add_argument("--lrr", type=int, default=4)

    if len(sys.argv) == 1:
        ap.print_help()
        print("""
quick start (the BlueQubit release is found next to this script, in the working dir,
up to 3 parents up, via $NIGHTHAWK_REPO or --repo):
  git clone https://github.com/BlueQubitDev/rcs-nighthawk
  python3 nighthawk_qrack.py verify                                 # circuits vs released QASM
  python3 nighthawk_qrack.py selftest                               # conventions, ~1 min
  python3 nighthawk_qrack.py hwxeb    --cache xeb_cache
  python3 nighthawk_qrack.py run      --backend ace --families mirror \\
          --sizes 27-36,61 --depths 4 6 8 12 --out clean_ace.jsonl
help per command: python3 nighthawk_qrack.py <command> -h""")
        return
    a = ap.parse_args()
    dict(seamgap=cmd_seamgap, verify=cmd_verify, hwxeb=cmd_hwxeb, run=cmd_run, selftest=cmd_selftest)[a.cmd](a)


if __name__ == "__main__":
    main()
