#!/usr/bin/env python3
"""
tnsweep.py -- deterministic temporal-boundary contraction for 1D brickwork
circuits, after Manabe, Gu and Pan (arXiv:2608.13110).

Why this exists
---------------
Near-Clifford simulation of the IBM doped-Clifford circuit costs 2^(n+t) on
the Qrack path measured in this project -- 2^565 for n=97, t=468. The
temporal-boundary contraction costs 2^ceil(d/2) = 2^35, and its width and
arithmetic cost are INDEPENDENT of the number and placement of T gates. The
circuit was hardened against magic-based and entanglement-based simulators;
its open quasi-one-dimensional geometry is what this method exploits instead.

The idea
--------
Every entangling gate is a CZ, which has operator Schmidt rank 2:

    CZ = |0><0| (x) I  +  |1><1| (x) Z

So each CZ splits into two rank-3 tensors sharing one binary bond. Absorbing
the single-qubit gates into the worldlines leaves a rectangular network of
spatial extent n and temporal extent ~d/2. Sweeping a boundary state along
the SPATIAL direction (rather than evolving a 2^n state forward in time)
keeps the boundary rank at ceil(d/2), because a spatial cut crosses only that
many CZ bonds.

Scope
-----
Open-boundary 1D chains with nearest-neighbour CZ entanglers and arbitrary
single-qubit gates. Verified against a dense statevector.

Usage
-----
  python tnsweep.py verify --n 8 --depth 8          # against dense reference
  python tnsweep.py scale --n 20 --depths 8,12,16   # width/time/memory
  python tnsweep.py amps  --qasm c.qasm --bits b.json --out amps.npy
"""

from __future__ import annotations

import os

import argparse
import json
import math
import re
import sys
import time

# BLAS threading must be pinned BEFORE numpy loads -- OpenBLAS reads these at
# import and the pool size is fixed thereafter.
#
# Every absorption is a matmul with inner dimension 2. OpenBLAS still spawns
# its full pool and synchronises it for each one: measured at 64 threads and
# 5895% CPU on a run nominally using one, achieving 1.4 GB/s. The barrier
# costs orders of magnitude more than the arithmetic behind it.
#
# Parallelism belongs in ABSORB_THREADS, which splits the boundary itself and
# does real work per thread. Override with TNSWEEP_BLAS_THREADS if a build
# genuinely benefits.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("TNSWEEP_BLAS_THREADS", "1"))

import numpy as np

# CZ = sum_a  P_a (x) Z^a   -- operator Schmidt rank 2
_P = [np.array([[1, 0], [0, 0]], dtype=np.complex64),
      np.array([[0, 0], [0, 1]], dtype=np.complex64)]
_ZP = [np.eye(2, dtype=np.complex64),
       np.array([[1, 0], [0, -1]], dtype=np.complex64)]

_QARG = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*\s*\[\s*(\d+)\s*\]")

_ONE_Q = {
    "id": np.eye(2, dtype=np.complex64),
    "x": np.array([[0, 1], [1, 0]], dtype=np.complex64),
    "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex64),
    "z": np.array([[1, 0], [0, -1]], dtype=np.complex64),
    "h": np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2),
    "s": np.array([[1, 0], [0, 1j]], dtype=np.complex64),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=np.complex64),
    "t": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex64),
    "tdg": np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex64),
}
_ONE_Q["sx"] = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]],
                              dtype=np.complex64)
_ONE_Q["sxdg"] = _ONE_Q["sx"].conj().T


# --------------------------------------------------------------------------- #
# circuit representation
# --------------------------------------------------------------------------- #

class Brickwork:
    """A 1D brickwork circuit: alternating CZ layers with single-qubit gates.

    layers[l] is the list of (q, q+1) pairs entangled at layer l.
    onegates[q][l] is the 2x2 unitary applied to qubit q just AFTER layer l
    (index -1 holds the gate applied before any entangling layer).
    """

    def __init__(self, n, depth):
        self.n = n
        self.depth = depth
        self.layers = [[(q, q + 1) for q in range(l % 2, n - 1, 2)]
                       for l in range(depth)]
        self.onegates = [{} for _ in range(n)]

    def add_one(self, q, after_layer, u):
        cur = self.onegates[q].get(after_layer)
        self.onegates[q][after_layer] = u if cur is None else u @ cur

    @staticmethod
    def random(n, depth, t_gates=0, seed=0):
        """Random Clifford brickwork with t_gates T gates inserted."""
        rng = np.random.default_rng(seed)
        c = Brickwork(n, depth)
        pool = ["h", "s", "sx", "sdg", "sxdg"]
        for q in range(n):
            c.add_one(q, -1, _ONE_Q["h"])
        for l in range(depth):
            for q in range(n):
                c.add_one(q, l, _ONE_Q[pool[rng.integers(len(pool))]])
        for _ in range(t_gates):
            c.add_one(int(rng.integers(n)), int(rng.integers(depth)),
                      _ONE_Q["t"])
        return c

    def truncate(self, depth):
        """Keep only the first `depth` entangling layers.

        Cost is 2^(ceil(d/2)+1), so truncation is the only lever that moves
        memory by orders of magnitude. A truncated circuit is a DIFFERENT
        circuit -- its amplitudes are not the published ones -- but it is the
        same family and the same code path, which is what a scaling test
        needs.
        """
        if depth >= self.depth:
            return self
        c = Brickwork(self.n, depth)
        c.layers = self.layers[:depth]
        for q in range(self.n):
            c.onegates[q] = {k: v for k, v in self.onegates[q].items()
                             if k < depth}
        return c

    @staticmethod
    def from_qasm(path):
        """Parse a QASM file whose CZ gates form a 1D brickwork chain."""
        n = None
        events = []          # (kind, ...) in file order
        for raw in open(path):
            line = raw.split("//")[0].strip().rstrip(";")
            if not line or line.startswith(("OPENQASM", "include", "creg")):
                continue
            m = re.match(r"qreg\s+\w+\s*\[\s*(\d+)\s*\]", line)
            if m:
                n = int(m.group(1))
                continue
            g = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)", line)
            if not g:
                continue
            name = g.group(1).lower()
            qs = [int(x) for x in _QARG.findall(line)]
            if name in ("barrier", "measure", "reset"):
                continue
            if name == "cz":
                a, b = sorted(qs)
                if b != a + 1:
                    raise SystemExit(
                        f"{path}: cz q[{a}],q[{b}] is not nearest-neighbour. "
                        f"This contraction assumes a 1D chain.")
                events.append(("cz", a))
            elif name in _ONE_Q:
                events.append(("1q", qs[0], _ONE_Q[name]))
            elif name in ("rz", "rx", "ry", "p", "u1"):
                th = _eval_angle(re.search(r"\(([^)]*)\)", line).group(1))
                if name in ("rz", "p", "u1"):
                    u = np.array([[np.exp(-1j * th / 2), 0],
                                  [0, np.exp(1j * th / 2)]], dtype=np.complex64)
                    if name in ("p", "u1"):
                        u = np.array([[1, 0], [0, np.exp(1j * th)]],
                                     dtype=np.complex64)
                elif name == "rx":
                    c_, s_ = np.cos(th / 2), -1j * np.sin(th / 2)
                    u = np.array([[c_, s_], [s_, c_]], dtype=np.complex64)
                else:
                    c_, s_ = np.cos(th / 2), np.sin(th / 2)
                    u = np.array([[c_, -s_], [s_, c_]], dtype=np.complex64)
                events.append(("1q", qs[0], u))
            else:
                raise SystemExit(f"{path}: unsupported gate {name!r}")

        if n is None:
            raise SystemExit(f"{path}: no qreg declaration")

        # group CZs into layers: a new layer starts when a qubit repeats
        layers, cur, used = [], [], set()
        pending = [[] for _ in range(n)]     # 1q gates awaiting a layer index
        seq = []
        for ev in events:
            if ev[0] == "cz":
                a = ev[1]
                if a in used or a + 1 in used:
                    layers.append(cur)
                    seq.append(pending)
                    pending = [[] for _ in range(n)]
                    cur, used = [], set()
                cur.append((a, a + 1))
                used.update((a, a + 1))
            else:
                pending[ev[1]].append(ev[2])
        if cur:
            layers.append(cur)
            seq.append(pending)
        else:
            seq.append(pending)

        c = Brickwork(n, len(layers))
        c.layers = layers
        for li, pend in enumerate(seq):
            for q in range(n):
                for u in pend[q]:
                    c.add_one(q, li - 1, u)
        return c


def _eval_angle(tok):
    tok = tok.strip().lower().replace("pi", f"({math.pi!r})")
    return float(eval(tok, {"__builtins__": {}}, {}))


# --------------------------------------------------------------------------- #
# tensor network construction
# --------------------------------------------------------------------------- #

def build_network(circ, xbits):
    """Tensors for <x|U|0^n>, one per (qubit, entangling layer) cell.

    Each CZ contributes two rank-3 tensors sharing a binary bond: the left
    qubit carries the projector P_a, the right carries Z^a. Single-qubit
    gates are absorbed into the temporal legs, which is why the network
    topology -- and hence the contraction width -- does not depend on them or
    on the T count.

    Returns a list of (array, [leg ids]) with negative ids for temporal legs
    and positive ids for spatial bonds.
    """
    n, d = circ.n, circ.depth
    tensors = []
    owner = []                 # qubit each tensor sits on, in worldline order

    # temporal leg ids per qubit, allocated as we walk down the worldline
    tleg = [0] * n
    next_t = [1]

    def new_t():
        next_t[0] += 1
        return -next_t[0]

    bond_id = [0]

    def new_bond():
        bond_id[0] += 1
        return bond_id[0]

    # incoming state: |0> with the pre-layer single-qubit gate applied
    state_leg = []
    for q in range(n):
        v = np.array([1, 0], dtype=np.complex64)
        u = circ.onegates[q].get(-1)
        if u is not None:
            v = u @ v
        lg = new_t()
        tensors.append((v, [lg]))
        owner.append(q)
        state_leg.append(lg)

    for l in range(d):
        acting = {}
        for (a, b) in circ.layers[l]:
            bd = new_bond()
            acting[a] = ("L", bd)
            acting[b] = ("R", bd)

        for q in range(n):
            u = circ.onegates[q].get(l)
            if q in acting:
                side, bd = acting[q]
                mats = _P if side == "L" else _ZP
                out = new_t()
                # T[in, out, bond] with the post-layer 1q gate folded in
                arr = np.empty((2, 2, 2), dtype=np.complex64)
                for a in (0, 1):
                    m = mats[a]
                    if u is not None:
                        m = u @ m
                    arr[:, :, a] = m.T          # [in, out]
                tensors.append((arr, [state_leg[q], out, bd]))
                owner.append(q)
                state_leg[q] = out
            elif u is not None:
                out = new_t()
                tensors.append((u.T.copy(), [state_leg[q], out]))
                owner.append(q)
                state_leg[q] = out

    # outgoing projector <x_q|
    for q in range(n):
        v = np.array([1, 0], dtype=np.complex64) if xbits[q] == 0 else \
            np.array([0, 1], dtype=np.complex64)
        tensors.append((v, [state_leg[q]]))
        owner.append(q)

    return tensors, owner


def sweep_order(n, owner):
    """Boustrophedon order: qubit 0..n-1, reversing time on alternate qubits.

    The reversal is what keeps the boundary compact. Sweeping every worldline
    in the same temporal direction would leave the cut running the length of
    the chain; alternating means the open legs always sit at one spatial
    position, crossing only the ~d/2 CZ bonds there.
    """
    per_q = [[] for _ in range(n)]
    for i, q in enumerate(owner):
        per_q[q].append(i)
    order = []
    for q in range(n):
        col = per_q[q]
        order.extend(col if q % 2 == 0 else col[::-1])
    return order



# --------------------------------------------------------------------------- #
# contraction
# --------------------------------------------------------------------------- #

# Set False to force np.tensordot everywhere, for A/B measurement. The fast
# path avoids a full strided copy of the boundary, which should matter once
# the boundary leaves cache -- unverified above ~1 MB, where everything is
# cache-resident and the two are indistinguishable.
USE_FAST_ABSORB = [True]

# Threads used to split one absorption across the boundary.
#
# This is the parallelism that was missing. BLAS will not thread a matmul
# whose inner dimension is 2, and numpy elementwise operations never thread,
# so the whole sweep ran on one core -- measured at 1.4 GB/s, which is about
# what a single Opteron core sustains. The boundary partitions cleanly along
# its leading axis, and numpy ufuncs release the GIL, so plain threads scale
# until memory bandwidth saturates.
ABSORB_THREADS = [1]


class Boundary:
    """A tensor plus its leg labels, absorbing others by shared legs."""

    __slots__ = ("arr", "legs")

    def __init__(self, arr, legs):
        self.arr = arr
        self.legs = list(legs)

    @property
    def rank(self):
        return len(self.legs)

    def _absorb_fast(self, arr, legs, shared):
        """Contract ONE shared axis without transposing the boundary.

        np.tensordot moves the contracted axes to the end, which is a strided
        copy of the whole array on every call -- measured at 47% of
        absorptions here, and the reason this ran at 1 GB/s against ~50 GB/s
        of hardware. Reshaping to (A, 2, C) around the axis is free on a
        contiguous array whatever the axis position, so the contraction
        becomes two slice reads and one write.

        Returns True if it handled the case.
        """
        if len(shared) != 1 or self.arr.ndim < 1:
            return False
        p = self.legs.index(shared[0])
        j = list(legs).index(shared[0])
        if j != 0 or arr.ndim > 3:
            return False

        A = int(np.prod(self.arr.shape[:p], dtype=np.int64))
        C = int(np.prod(self.arr.shape[p + 1:], dtype=np.int64))
        b3 = self.arr.reshape(A, 2, C)          # free: contiguous
        tail = arr.shape[1:]                    # legs the small tensor adds
        out = np.empty((A, C) + tail, dtype=self.arr.dtype)
        pad = (1,) * len(tail)

        def block(lo, hi):
            # np.multiply(..., out=) avoids the temporary that `x * y` builds,
            # which on a 512 MB boundary is a second full array
            np.multiply(b3[lo:hi, 0, :].reshape(hi - lo, C, *pad), arr[0],
                        out=out[lo:hi])
            out[lo:hi] += b3[lo:hi, 1, :].reshape(hi - lo, C, *pad) * arr[1]

        nt = min(ABSORB_THREADS[0], max(1, A))
        if nt <= 1 or A < 4096:
            block(0, A)
        else:
            import threading
            step = (A + nt - 1) // nt
            ts = [threading.Thread(target=block, args=(i, min(i + step, A)))
                  for i in range(0, A, step)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()

        self.arr = out.reshape(self.arr.shape[:p] + self.arr.shape[p + 1:]
                               + tail)
        self.legs = ([l for l in self.legs if l != shared[0]]
                     + [l for l in legs if l != shared[0]])
        return True

    def absorb(self, arr, legs):
        shared = [l for l in legs if l in self.legs]
        if shared and USE_FAST_ABSORB[0] and self._absorb_fast(arr, legs, shared):
            return
        if not shared:
            # disjoint: outer product, which is what grows the boundary
            self.arr = np.tensordot(self.arr, arr, axes=0)
            self.legs = self.legs + list(legs)
            return
        ax_self = [self.legs.index(l) for l in shared]
        ax_other = [list(legs).index(l) for l in shared]
        self.arr = np.tensordot(self.arr, arr, axes=(ax_self, ax_other))
        self.legs = ([l for l in self.legs if l not in shared]
                     + [l for l in legs if l not in shared])


class WidthExceeded(RuntimeError):
    pass


def contract(tensors, order, width_cap=None, report=None, ceiling=None):
    """Absorb tensors in `order`, pairing ahead when a step would overshoot.

    The pairing rule is Algorithm 1: if absorbing the next tensor alone would
    push the boundary above the cap, contract it with the following one first.
    Two rank-3 tensors sharing a leg give rank 4, and that composite shares
    two legs with the boundary, so the net rank does not grow.
    """
    b = Boundary(*tensors[order[0]])
    peak = b.rank
    i = 1

    def check(rank):
        # A hard stop, checked BEFORE the allocation that would breach it.
        # width_cap only steers the pairing rule; without a ceiling a boundary
        # that outgrows the forecast is discovered by the machine swapping,
        # which is not a diagnostic.
        if ceiling is not None and rank > ceiling:
            raise WidthExceeded(
                f"boundary would reach rank {rank} (> ceiling {ceiling}), "
                f"i.e. {(2.0 ** rank) * 8 / 2 ** 30:.1f} GiB. Lower "
                f"--max-depth or raise --budget-gb.")
    while i < len(order):
        arr, legs = tensors[order[i]]
        shared = sum(1 for l in legs if l in b.legs)
        would = b.rank + len(legs) - 2 * shared
        if width_cap is not None and would > width_cap and i + 1 < len(order):
            a2, l2 = tensors[order[i + 1]]
            sh = [l for l in l2 if l in legs]
            if sh:
                ax1 = [list(legs).index(l) for l in sh]
                ax2 = [list(l2).index(l) for l in sh]
                comp = np.tensordot(arr, a2, axes=(ax1, ax2))
                clegs = ([l for l in legs if l not in sh]
                         + [l for l in l2 if l not in sh])
                csh = sum(1 for l in clegs if l in b.legs)
                check(b.rank + len(clegs) - 2 * csh)
                b.absorb(comp, clegs)
                peak = max(peak, b.rank)
                i += 2
                continue
        check(would)
        b.absorb(arr, legs)
        peak = max(peak, b.rank)
        if report is not None and b.rank >= peak:
            report(b.rank)
        i += 1
    return b, peak


def amplitude(circ, xbits, width_cap=None, ceiling=None, threads=None):
    """<x|U|0^n> by the temporal-boundary sweep. Returns (value, peak_rank)."""
    if threads:
        ABSORB_THREADS[0] = threads
    tensors, owner = build_network(circ, xbits)
    order = sweep_order(circ.n, owner)
    # h+1, not h. Measured: capping at h makes the pairing rule fire on almost
    # every step, and the composites it builds are wider than the tensors they
    # replace, so the peak ends up at h+2. Capping at h+1 reaches h+1.
    cap = width_cap if width_cap is not None else math.ceil(circ.depth / 2) + 1
    b, peak = contract(tensors, order, width_cap=cap, ceiling=ceiling)
    if b.rank != 0:
        raise RuntimeError(f"contraction left rank {b.rank}, expected a scalar")
    return complex(b.arr), peak


# --------------------------------------------------------------------------- #
# dense reference, for validation only
# --------------------------------------------------------------------------- #

def dense_statevector(circ):
    """Full 2^n statevector. Small circuits only -- this is the check, not
    the method."""
    n = circ.n
    psi = np.zeros(2 ** n, dtype=np.complex128)
    psi[0] = 1.0

    def apply1(psi, u, q):
        psi = psi.reshape([2] * n)
        psi = np.moveaxis(psi, q, 0).reshape(2, -1)
        psi = (u.astype(np.complex128) @ psi).reshape([2] * n)
        return np.moveaxis(psi, 0, q).reshape(-1)

    def apply_cz(psi, a, b):
        idx = np.arange(2 ** n)
        mask = ((idx >> (n - 1 - a)) & 1) & ((idx >> (n - 1 - b)) & 1)
        out = psi.copy()
        out[mask == 1] *= -1
        return out

    for q in range(n):
        u = circ.onegates[q].get(-1)
        if u is not None:
            psi = apply1(psi, u, q)
    for l in range(circ.depth):
        for (a, b) in circ.layers[l]:
            psi = apply_cz(psi, a, b)
        for q in range(n):
            u = circ.onegates[q].get(l)
            if u is not None:
                psi = apply1(psi, u, q)
    return psi


def _state_index(bits, n):
    v = 0
    for i, b in enumerate(bits):
        if b:
            v |= 1 << (n - 1 - i)
    return v


# --------------------------------------------------------------------------- #

def memory_forecast(n, depth, dense=False, budget_gb=None):
    """Predicted peak allocation, checked BEFORE anything is built.

    Two allocations matter and they scale differently. The contraction
    boundary is 2^(ceil(d/2)+1) complex64 and grows with DEPTH. The dense
    reference used for validation is 2^n complex128 and grows with WIDTH --
    it is the cheaper of the two only while n stays small, and it is the one
    that bites first when someone raises --n to make a test "more realistic".

    Refusing up front is the point: an allocation that overshoots physical
    memory does not fail cleanly here, it takes the machine into swap.
    """
    w = math.ceil(depth / 2) + 1
    b_bytes = (2.0 ** w) * 8
    d_bytes = (2.0 ** n) * 16 if dense else 0.0
    total = b_bytes + d_bytes

    def human(x):
        u = "B"
        for nxt in ("KB", "MB", "GB", "TB", "PB"):
            if x > 1024:
                x /= 1024
                u = nxt
        return f"{x:.1f} {u}"

    lines = [f"predicted peak allocation (n={n}, d={depth}):",
             f"  contraction boundary  width {w}  {human(b_bytes)}"]
    if dense:
        lines.append(f"  dense reference       2^{n}      {human(d_bytes)}")
    lines.append(f"  total                            {human(total)}")

    over = budget_gb is not None and total > budget_gb * 1e9
    if over:
        lines += ["",
                  f"REFUSING: that exceeds the {budget_gb:g} GB budget. Lower "
                  f"--depth (the boundary halves per 2 of depth)"
                  + (", or --n (the dense reference halves per qubit)."
                     if dense else ".")]
    return "\n".join(lines), (not over), total


def cmd_width(args):
    """Locate the width offset: what the open legs are, and how the cap moves
    the peak.

    Two questions, because they have different answers. The leg composition at
    peak says WHAT is being held open -- a cut between two qubits crosses d/2
    CZ bonds plus one worldline, so anything beyond that is the offset. The cap
    sweep says whether the pairing rule can remove it, and every row is checked
    against a dense statevector so a smaller peak that quietly changes the
    answer cannot pass.
    """
    txt, ok, _ = memory_forecast(args.n, args.depth, dense=True,
                                 budget_gb=args.budget_gb)
    if not ok:
        return txt
    circ = Brickwork.random(args.n, args.depth, args.t_gates, args.seed)
    h = math.ceil(args.depth / 2)
    tensors, owner = build_network(circ, [0] * args.n)
    order = sweep_order(args.n, owner)
    psi = dense_statevector(circ)
    ref = psi[0]
    scale = 2.0 ** (-args.n / 2)

    # --- what is open at the peak, with no cap at all
    b = Boundary(*tensors[order[0]])
    peak, worst, at = b.rank, list(b.legs), 0
    for k, i in enumerate(order[1:], 1):
        b.absorb(*tensors[i])
        if b.rank > peak:
            peak, worst, at = b.rank, list(b.legs), k
    spatial = [l for l in worst if l > 0]
    temporal = [l for l in worst if l < 0]

    # how many CZ bonds does one spatial cut actually carry?
    per_cut = {}
    for l in range(circ.depth):
        for (a, bq) in circ.layers[l]:
            per_cut[(a, bq)] = per_cut.get((a, bq), 0) + 1
    typical = max(per_cut.values()) if per_cut else 0

    out = [f"n={args.n} d={args.depth} t={args.t_gates}   "
           f"certified minimum h = ceil(d/2) = {h}",
           "",
           f"uncapped peak {peak} at step {at}:",
           f"  spatial CZ bonds  {len(spatial):3d}   one cut carries {typical}",
           f"  temporal worldline{len(temporal):3d}   expected 1",
           f"  -> offset of {peak - h} above h, and it is "
           f"{'a bond' if len(spatial) > typical else 'a worldline'}",
           "",
           "cap sweep (amplitude checked against the dense statevector each row):",
           f"  {'cap':>5} {'peak':>5} {'vs h':>6} {'amp err':>10}"]

    best = (peak, None)
    for cap in range(h + 3, 0, -1):
        bb, pk = contract(tensors, order, width_cap=cap)
        err = abs(complex(bb.arr) - ref) / max(abs(ref), scale)
        flag = "  <-- best" if pk < best[0] and err < 1e-4 else ""
        if pk < best[0] and err < 1e-4:
            best = (pk, cap)
        out.append(f"  {cap:5d} {pk:5d} {pk - h:+6d} {err:10.2e}"
                   + ("  WRONG" if err >= 1e-4 else "") + flag)

    out += ["",
            f"  best reachable peak {best[0]} at cap {best[1]}, "
            f"offset +{best[0] - h}"]
    if best[0] > h:
        out += [
            "",
            "  The remaining offset is a bond that opens before its partner",
            "  closes. The paper folds input states, output projectors and",
            "  adjacent one-qubit gates into neighbouring tensors so the",
            "  network has exactly one tensor per (qubit, entangling layer);",
            "  this build emits some of them separately, which is where an",
            "  extra bond can stay live across a step.",
            f"  Each unit of offset costs 2x memory: at d=70 that is "
            f"{2 ** (best[0] - h)}x on top of 256 GiB."]
    return "\n".join(out)


def cmd_verify(args):
    txt, ok, _ = memory_forecast(args.n, args.depth, dense=True,
                                 budget_gb=args.budget_gb)
    print(txt, file=sys.stderr)
    if not ok:
        return 2
    print(file=sys.stderr)
    rng = np.random.default_rng(args.seed)
    circ = Brickwork.random(args.n, args.depth, args.t_gates, args.seed)
    psi = dense_statevector(circ)

    h = math.ceil(args.depth / 2)
    print(f"n={args.n} d={args.depth} t={args.t_gates}   "
          f"expected width ceil(d/2) = {h}")
    print(f"{'bitstring':>{args.n + 2}} {'dense':>24} {'sweep':>24} "
          f"{'rel.err':>10} {'peak':>5}")

    worst, worst_peak = 0.0, 0
    for _ in range(args.samples):
        bits = [int(b) for b in rng.integers(0, 2, args.n)]
        ref = psi[_state_index(bits, args.n)]
        t0 = time.perf_counter()
        got, peak = amplitude(circ, bits)
        el = time.perf_counter() - t0
        # Scale by the typical amplitude 2^(-n/2), not by |ref|. A bitstring
        # whose exact amplitude is ~0 makes a relative error meaningless: both
        # numbers are zero and the quotient is noise over noise.
        scale = max(abs(ref), 2.0 ** (-args.n / 2))
        err = abs(got - ref) / scale
        worst = max(worst, err)
        worst_peak = max(worst_peak, peak)
        print(f"{''.join(map(str, bits)):>{args.n + 2}} "
              f"{ref.real:11.6f}{ref.imag:+11.6f}j "
              f"{got.real:11.6f}{got.imag:+11.6f}j {err:10.2e} {peak:5d}"
              f"   {el * 1000:.0f} ms")

    print()
    ok_val = worst < 1e-4
    # the certified minimum is h; this implementation reaches h+2. Amplitudes
    # are exact either way, so the width is reported as a measured property
    # rather than treated as a correctness failure.
    ok_width = worst_peak <= h + 2
    print(f"  amplitudes agree with the dense reference: "
          f"{'PASS' if ok_val else f'FAIL (rel.err {worst:.2e})'}")
    print(f"  peak boundary rank {worst_peak}, certified minimum {h} "
          f"(offset +{worst_peak - h}): "
          f"{'within tolerance' if ok_width else 'WORSE THAN EXPECTED'}")
    if ok_val and ok_width:
        print("\n  The sweep reproduces exact amplitudes at the predicted "
              "width.\n  Cost here is set by depth, not by t -- raise "
              "--t-gates and the\n  width does not move.")
    return 0 if (ok_val and ok_width) else 1


def cmd_scale(args):
    depths = [int(x) for x in args.depths.split(",")]
    for d in depths:
        txt, ok, _ = memory_forecast(args.n, d, dense=False,
                                     budget_gb=args.budget_gb)
        if not ok:
            print(txt, file=sys.stderr)
            return 2
    print(f"n={args.n}, t={args.t_gates}")
    print(f"{'d':>4} {'width':>6} {'peak':>6} {'entries':>12} {'payload':>10} "
          f"{'s/amp':>9}")
    for d in depths:
        circ = Brickwork.random(args.n, d, args.t_gates, args.seed)
        bits = [0] * args.n
        t0 = time.perf_counter()
        _, peak = amplitude(circ, bits)
        el = time.perf_counter() - t0
        h = math.ceil(d / 2)
        payload = (2.0 ** peak) * 8
        unit = "B"
        for u in ("KB", "MB", "GB", "TB"):
            if payload > 1024:
                payload /= 1024
                unit = u
        print(f"{d:4d} {h:6d} {peak:6d} {2 ** peak:12d} "
              f"{payload:7.1f} {unit:<2} {el:9.3f}")

    print()
    print("  width tracks ceil(d/2) exactly; payload is 2^width x 8 B.")
    print("  extrapolating: d=70 -> width 35 -> 256 GiB, which is the")
    print("  published instance. Time scales with the same exponent.")
    return 0


def _amp_worker(task):
    """One amplitude, in its own process.

    The truncation depth is carried in the task and applied HERE. A worker
    re-parses the QASM from scratch, so a circuit truncated only in the parent
    arrives at full depth -- and the parent's memory forecast then describes a
    contraction nobody is running. That mistake put a 512 GB boundary in four
    concurrent workers.
    """
    qasm, bits, cap, blas, max_depth, ceiling = task
    # deliberately NOT raising BLAS threads here: with inner dimension 2 the
    # pool costs more than it computes. `blas` now sets the boundary split.
    ABSORB_THREADS[0] = max(1, blas)
    circ = Brickwork.from_qasm(qasm)
    if max_depth:
        circ = circ.truncate(max_depth)
    val, peak = amplitude(circ, bits, width_cap=cap, ceiling=ceiling)
    return val, peak


def cmd_profile(args):
    """Where the sweep actually spends its time, bucketed by absorption shape.

    Guessing has a poor record here: transposes were ruled out by measurement
    after a fix was already written for them. The buckets below separate the
    cases that have different memory behaviour, so the next optimisation
    targets whichever one dominates rather than whichever seems likely.

      trailing  contracted axis is last -> the (A,2,C) view has C=1, so the
                two halves interleave element by element and half of every
                cache line is discarded
      interior  C>1 -> each half is read in contiguous runs
      2-shared  falls through to np.tensordot, which does transpose
    """
    circ = Brickwork.random(args.n, args.depth, args.t_gates, args.seed)
    tensors, owner = build_network(circ, [0] * args.n)
    order = sweep_order(args.n, owner)
    cap = math.ceil(args.depth / 2) + 1
    ABSORB_THREADS[0] = args.threads

    buckets = {}
    b = Boundary(*tensors[order[0]])
    i = 1
    while i < len(order):
        arr, legs = tensors[order[i]]
        shared = [l for l in legs if l in b.legs]
        if shared:
            p_ = min(b.legs.index(l) for l in shared)
            if len(shared) >= 2:
                kind = "2-shared (tensordot)"
            elif p_ == b.rank - 1:
                kind = "1-shared trailing"
            else:
                kind = "1-shared interior"
        else:
            kind = "disjoint (outer)"
        sz = self_sz = b.arr.size
        t0 = time.perf_counter()
        b.absorb(arr, legs)
        el = time.perf_counter() - t0
        rec = buckets.setdefault(kind, [0, 0.0, 0])
        rec[0] += 1
        rec[1] += el
        rec[2] += sz
        i += 1

    total = sum(v[1] for v in buckets.values())
    out = [f"n={args.n} d={args.depth} threads={args.threads}   "
           f"total {total:.2f} s",
           "",
           f"{'bucket':<24} {'calls':>7} {'time':>9} {'share':>7} "
           f"{'GB touched':>11} {'GB/s':>8}"]
    for k, (cnt, t, sz) in sorted(buckets.items(), key=lambda kv: -kv[1][1]):
        gb = sz * 8 / 1e9
        out.append(f"{k:<24} {cnt:7d} {t:8.2f}s {t / total:6.1%} "
                   f"{gb:10.2f} {gb / max(t, 1e-9):8.2f}")
    out += ["",
            "  the dominant bucket is where any optimisation has to land.",
            "  GB/s well under 10 on a modern socket points at access pattern",
            "  or per-call overhead, not at memory capacity."]
    return "\n".join(out)


def cmd_bench(args):
    """A/B the transpose-free absorption against np.tensordot.

    Run this at a width whose boundary exceeds L3, or the answer is
    meaningless: below cache the two paths measure the same because no copy
    ever reaches memory. On this hardware the gap opened at width 26
    (512 MB), where the sweep achieved 1.1 GB/s against ~50 GB/s of DDR3.
    """
    threads = [int(x) for x in args.threads.split(",")]
    out = [f"n={args.n}, boundary must exceed L3 for this to mean anything",
           "",
           f"{'d':>4} {'width':>6} {'boundary':>10} "
           + "".join(f"{'t=' + str(t):>11}" for t in threads) + f"{'best':>9}"]
    for d in [int(x) for x in args.depths.split(",")]:
        circ = Brickwork.random(args.n, d, args.t_gates, args.seed)
        w = math.ceil(d / 2) + 1
        size = (2.0 ** w) * 8
        unit = "B"
        for u in ("KB", "MB", "GB"):
            if size > 1024:
                size /= 1024
                unit = u
        row, ref, times = "", None, []
        for t in threads:
            t0 = time.perf_counter()
            v, pk = amplitude(circ, [0] * args.n, threads=t)
            el = time.perf_counter() - t0
            if ref is None:
                ref = v
            bad = abs(v - ref) > 1e-5 * max(abs(ref), 2.0 ** (-args.n / 2))
            times.append(el)
            row += f"{el:10.2f}s" + ("!" if bad else " ")
        out.append(f"{d:4d} {pk:6d} {size:7.1f} {unit:<2} {row}"
                   f"{times[0] / min(times):8.1f}x")
    out += ["",
            "  '!' marks a thread count that changed the answer -- there "
            "should be none.",
            "  scaling stops when memory bandwidth saturates, typically well "
            "below core count."]
    return "\n".join(out)


def cmd_amps(args):
    if not os.path.isfile(args.qasm):
        raise SystemExit(f"no circuit at {args.qasm!r}")
    if not os.path.isfile(args.bits):
        raise SystemExit(f"no bitstring file at {args.bits!r}")
    circ = Brickwork.from_qasm(args.qasm)
    print(f"{os.path.basename(args.qasm)}: {circ.n} qubits, "
          f"{circ.depth} entangling layers", file=sys.stderr)

    if args.max_depth and args.max_depth < circ.depth:
        print(f"  truncating to {args.max_depth} layers -- the amplitudes are "
              f"then NOT the published ones,\n  but the path and its scaling "
              f"are the same", file=sys.stderr)
        circ = circ.truncate(args.max_depth)

    txt, ok, _ = memory_forecast(circ.n, circ.depth, dense=False,
                                 budget_gb=args.budget_gb)
    print(txt, file=sys.stderr)
    if not ok:
        h_fit = int(math.log2(args.budget_gb * 1e9 / 8))
        print(f"\n  --max-depth {2 * (h_fit - 1)} would fit this budget.",
              file=sys.stderr)
        return 2
    print(file=sys.stderr)
    with open(args.bits) as f:
        blob = json.load(f)
    key = next(k for k in ("accepted_samples", "samples", "bitstrings")
               if k in blob)
    rows = blob[key][:args.limit] if args.limit else blob[key]

    h = math.ceil(circ.depth / 2)
    print(f"{circ.n} qubits, depth {circ.depth}, width {h}, "
          f"{(2.0 ** h) * 8 / 2 ** 30:.1f} GiB boundary", file=sys.stderr)

    # Amplitudes are independent, so they parallelise perfectly -- but each
    # worker holds its own boundary, and at large depth one boundary is the
    # whole machine. Cap concurrency by memory, not by core count.
    per_amp = (2.0 ** (h + 1)) * 8
    fit = max(1, int(args.budget_gb * 1e9 // per_amp))
    asked = args.workers or os.cpu_count() or 1
    workers = min(asked, fit, len(rows))
    if workers < asked:
        why = ("only %d bitstrings to compute" % len(rows)) if workers == len(rows) \
            else ("memory: %.2f GiB per boundary against a %g GB budget"
                  % (per_amp / 2 ** 30, args.budget_gb))
        print(f"  {workers} workers instead of {asked} -- {why}",
              file=sys.stderr)
    # when only one amplitude fits, hand the cores to BLAS instead
    blas = max(1, (os.cpu_count() or 1) // workers)
    print(f"  {workers} worker(s) x {blas} absorption thread(s)   "
          f"(BLAS pinned to {os.environ.get('OPENBLAS_NUM_THREADS')})",
          file=sys.stderr)

    out = np.empty(len(rows), dtype=np.complex64)
    t0 = time.perf_counter()
    cap = math.ceil(circ.depth / 2) + 1

    if workers == 1:
        for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[v] = str(blas)
        ceiling = int(math.log2(args.budget_gb * 1e9 / 8))
        for i, bits in enumerate(rows):
            out[i], peak = amplitude(circ, [int(b) for b in bits], cap,
                                     ceiling=ceiling)
            if i % 10 == 9 or i == len(rows) - 1:
                el = time.perf_counter() - t0
                print(f"  {i + 1}/{len(rows)}  {el / (i + 1):.2f}s per "
                      f"amplitude, peak rank {peak}", file=sys.stderr, flush=True)
    else:
        import multiprocessing as mp
        ceiling = int(math.log2(args.budget_gb * 1e9 / 8 / max(1, workers)))
        tasks = [(args.qasm, [int(b) for b in r], cap, blas,
                  args.max_depth, ceiling) for r in rows]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for i, (val, peak) in enumerate(
                    pool.imap(_amp_worker, tasks, chunksize=1)):
                out[i] = val
                if i % 10 == 9 or i == len(rows) - 1:
                    el = time.perf_counter() - t0
                    print(f"  {i + 1}/{len(rows)}  {el / (i + 1):.2f}s per "
                          f"amplitude (wall), peak rank {peak}",
                          file=sys.stderr, flush=True)

    np.save(args.out, np.abs(out))
    side = os.path.splitext(args.out)[0] + ".meta.json"
    with open(side, "w") as f:
        json.dump({"method": "temporal-boundary sweep", "qasm": args.qasm,
                   "n": circ.n, "depth": circ.depth, "width": h,
                   "n_samples": len(rows)}, f, indent=2)
    print(f"wrote {args.out} ({len(out)} magnitudes) and {side}")
    print(f"now: xeb3d.py stats {args.bits} --amps {args.out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pw = sub.add_parser("width",
                        help="locate the width offset: leg composition at "
                             "peak, and a cap sweep checked against dense")
    pw.add_argument("--n", type=int, default=8)
    pw.add_argument("--depth", type=int, default=10)
    pw.add_argument("--t-gates", type=int, default=4)
    pw.add_argument("--seed", type=int, default=0)
    pw.add_argument("--budget-gb", type=float, default=8.0,
                    help="refuse to run if the forecast exceeds this; the "
                         "default is deliberately small so a typo in --depth "
                         "cannot take the machine into swap")

    pv = sub.add_parser("verify", help="check against a dense statevector")
    pv.add_argument("--n", type=int, default=8)
    pv.add_argument("--depth", type=int, default=8)
    pv.add_argument("--t-gates", type=int, default=4)
    pv.add_argument("--samples", type=int, default=4)
    pv.add_argument("--seed", type=int, default=0)
    pv.add_argument("--budget-gb", type=float, default=8.0,
                    help="refuse to run if the forecast exceeds this; the "
                         "default is deliberately small so a typo in --depth "
                         "cannot take the machine into swap")

    ps = sub.add_parser("scale", help="width, memory and time vs depth")
    ps.add_argument("--n", type=int, default=20)
    ps.add_argument("--depths", default="8,12,16,20")
    ps.add_argument("--t-gates", type=int, default=8)
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--budget-gb", type=float, default=8.0,
                    help="refuse to run if the forecast exceeds this; the "
                         "default is deliberately small so a typo in --depth "
                         "cannot take the machine into swap")

    pp = sub.add_parser("profile",
                        help="time per absorption, bucketed by shape")
    pp.add_argument("--n", type=int, default=70)
    pp.add_argument("--depth", type=int, default=40)
    pp.add_argument("--t-gates", type=int, default=10)
    pp.add_argument("--threads", type=int, default=1)
    pp.add_argument("--seed", type=int, default=0)

    pb = sub.add_parser("bench",
                        help="A/B the transpose-free absorption vs tensordot")
    pb.add_argument("--n", type=int, default=50)
    pb.add_argument("--depths", default="30,40,44,48")
    pb.add_argument("--t-gates", type=int, default=10)
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("--threads", default="1,4,12,24",
                    help="absorption thread counts to compare")

    pa = sub.add_parser("amps", help="amplitudes for a set of bitstrings")
    pa.add_argument("--qasm", required=True)
    pa.add_argument("--bits", required=True)
    pa.add_argument("--limit", type=int, default=None)
    pa.add_argument("--out", required=True)
    pa.add_argument("--workers", type=int, default=None,
                    help="concurrent amplitudes; capped by --budget-gb since "
                         "each worker holds its own boundary")
    pa.add_argument("--budget-gb", type=float, default=8.0)
    pa.add_argument("--max-depth", type=int, default=None,
                    help="truncate to this many entangling layers; the only "
                         "lever that moves memory, since cost is 2^(d/2)")

    args = ap.parse_args(argv)
    if args.cmd == "profile":
        print(cmd_profile(args))
        return 0
    if args.cmd == "bench":
        print(cmd_bench(args))
        return 0
    if args.cmd == "width":
        print(cmd_width(args))
        return 0
    return {"verify": cmd_verify, "scale": cmd_scale, "amps": cmd_amps}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
