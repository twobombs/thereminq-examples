#!/usr/bin/env python3
"""
nighthawk_cloverfield.py -- several cloverleaf chips woven by the loom into one virtual chip.

The cloverleaf (nighthawk_qrack.py) emulates one IBM Nighthawk chip as a few ACE patches
joined by seams. The cloverfield puts two or more of those chips side by side and connects
them with inter-chip couplers, the way IBM plans to link Nighthawk chips with l-couplers:
one large register, one circuit, one mirror survival.

Architecture
  chip worker   one process per chip, isolated to one OpenCL device (or the CPU) exactly as
                nighthawk_qrack's workers are. It holds that chip's simulator (its ACE preset,
                or an exact QrackSimulator for small chips) and nothing else: it never sees
                another chip. It executes gate batches and answers marginal and probability
                queries over a pipe.
  loom          the coordinator. It generates the circuit of the whole field with the
                release's own recipe (rect_matchings + the K = 3 pseudo-patch rotation
                + the input-string recipe, on the global register), cuts every cycle into
                per-chip batches, and does all IO between the chips. Intra-chip CZs go to
                their chip. A link CZ (a on chip A, b on chip B) is woven by the loom.
                --loom ghost (default): as ACE replicates a boundary qubit in both patches
                next to a seam, every chip also holds a ghost of each partner its links touch
                (a ghost of b on chip A, of a on chip B, on a free ACE site coupled to the
                anchor). Every single-qubit gate and twirl Pauli of a qubit goes to all its
                copies; the link CZ is applied exactly on both chips (a with ghost(b), ghost(a)
                with b), so entanglement crosses the link; the partner's own CZs with its chip
                neighbours reach the ghost as the mean-field phase pi * P(1) of the neighbour.
                Links are only placed where both chips have ghost room under their width caps.
                --loom mean: no ghosts; the CZ's mean-field decoupling on both sides,
                    exp(i pi n_a n_b)  ~  exp(i pi (n_a <n_b> + <n_a> n_b)),
                a phase pi*P_b(1) on a and pi*P_a(1) on b; chips stay in a product state.
                --loom sample: a Z kick drawn from P(1) instead of the phase.
                All are exact when a partner is in a computational basis state. CZ layers are
                diagonal, so marginals read before a layer hold for all of it: one round trip
                per chip per layer that needs a marginal, all chips in parallel. Records carry
                ghost_drift, the mean |P(1) of a ghost - P(1) of its real qubit|.
  estimator     mirror survival with nighthawk_qrack's prob estimator on every chip, the
                field's survival the product of the chips' probabilities of their part of the
                prepared string (ghosts traced out). With ghosts, each chip's part already
                carries what entanglement across the link did to it; the product drops only
                the correlation between the chips' outcomes, which a perfect mirror returns to
                zero.

Field
  --chips NAME[,NAME...] or NAME*K   chip registers: any nighthawk_qrack register
                (release, nighthawk88, nighthawk88d, nighthawk88e, layoutsearch-saved ones)
                with its stored ACE preset, or rect:r0-r1,c0-c1 (a rectangle of the 12 x 10
                lattice, broken qubits dropped) for small exact or --chip-ace chips
  --arrangement RxC  chips on a grid, row-major (default 1 x K)
  --links K|all      inter-chip couplers per adjacent pair: K facing border-site pairs,
                evenly spaced, or every one (--link-rows/--link-cols pick them by hand);
                each takes a CZ colour (A-D) free on both of its qubits, so a cycle still
                applies one colour and every qubit meets at most one CZ per cycle
  --gtt GB           VRAM oversubscription per chip worker: up to GB of host memory its GPU
                maps over PCIe (Qrack's ceiling QRACK_MAX_ALLOC_MB raised to --vram-gb + GB);
                ghost room is then judged by the VRAM / GTT split instead of the register's
                width caps, so wider patches and more ghost links fit on 8 GB dies

Subcommands
  plan          the field: chips, qubits, couplers per colour, links, ACE presets and
                memory per worker, devices needed; --png draws it
  run           mirror circuits on the field (--depths, --instances, --twirls), loom +
                chip workers; --gpus/--chips-per-gpu place the chip workers, --parallel
                runs several looms at once; resumable JSONL records (nighthawk_graph.py
                reads them as mirror records); --status, --summarize
  linksearch    the same field at several link counts (--link-counts 0 2 4 all), short
                mirror runs each: survival against number of links
  layoutsearch  nighthawk_qrack's register + placement search for one chip, unchanged
                (chips are cloverleaves; the loom does not change them)
  selftest      field construction; the loom's op stream for one chip against nighthawk_qrack's
                engine calls (exact); the woven CZ against exact cases; CZ.CZ = I across a
                link (ghost: exact, mean-field: fails); the loom's error on exact chips

Examples
  python3 nighthawk_cloverfield.py plan --chips nighthawk88e*2 --links 4 --png field.png
  python3 nighthawk_cloverfield.py run  --chips nighthawk88e*2 --links 4 --theta nnqab \\
          --depths 4 6 8 10 12 --ace-gpu --gpus 0-5 --out field88e2.jsonl
  python3 nighthawk_cloverfield.py run  --chips nighthawk88e*2 --links 4 --out field88e2.jsonl --summarize
  python3 nighthawk_cloverfield.py linksearch --chips release*2 --link-counts 0 2 4 8 \\
          --depths 4 8 --twirls 4 --cpu
  python3 nighthawk_cloverfield.py selftest
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import queue
import struct
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH) and "PYQRACK_SHARED_LIB_PATH" not in os.environ:
    os.environ["PYQRACK_SHARED_LIB_PATH"] = QRACK_LIB_PATH

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nighthawk_qrack as nq  # noqa: E402

COLOURS = ("A", "B", "C", "D")
# the experiment's own register as a chip: Dan's three-patch ACE register (README Sec. 2.8)
CHIP_PRESETS = {
    "release": dict(width=75, lrc=4, lrr=5, torus=True, exclude="74,59,44,73,69,54,68,67,64,63,62"),
}


# ================================================================== chips
class ChipLayout:
    """A chip's register: its sites on the 12 x 10 lattice and its CZ colours, from the
    release's rect_matchings/remove_qubits (as nighthawk_qrack's RegisterLayout, without
    the K = 3, 4 partitions, which the loom computes for the whole field)."""

    def __init__(self, repo, name, rows, cols, drop, n):
        self.repo = nq.find_repo(repo)
        L = json.loads((self.repo / "data/layout.json").read_text())
        rl = nq._load_release_module(self.repo, "layout")
        (r0, r1), (c0, c1) = rows, cols
        R, C = r1 - r0 + 1, c1 - c0 + 1
        cols_dev = L["device_lattice"]["cols"]
        if drop is not None:
            gone = sorted({(p // cols_dev - r0) * C + (p % cols_dev - c0) for p in drop})
        else:
            gone = list(range(n, R * C))
        keep = [i for i in range(R * C) if i not in set(gone)]
        if len(keep) != n:
            raise SystemExit(f"chip {name}: {len(keep)} sites, expected {n}")
        m = rl.remove_qubits(rl.rect_matchings(R, C), gone)
        self.name, self.n = name, n
        self.l2p = [(r0 + i // C) * cols_dev + (c0 + i % C) for i in keep]
        self.matchings = {k: [tuple(e) for e in v] for k, v in m.items()}
        self.schedule = L["schedule"]
        self.grid = (R, C)
        self.pos = [(i // C, i % C) for i in keep]
        self.base_seed = L["base_seed"]


def parse_chip(tok, repo, broken_policy="disable"):
    """Chip spec dict from one --chips token."""
    if tok.startswith("rect:"):
        rr, cc = tok[5:].split(",")
        r0, r1 = map(int, rr.split("-"))
        c0, c1 = map(int, cc.split("-"))
        R_, C_, broken, _ = nq.lattice_info(repo)
        drop = [r * C_ + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                if (r, c) in broken and broken_policy == "disable"]
        n = (r1 - r0 + 1) * (c1 - c0 + 1) - len(drop)
        return dict(kind="rect", name=tok, rows=[r0, r1], cols=[c0, c1], drop=drop, n=n)
    if tok == "release":
        return dict(kind="release", name="release")
    if tok not in nq.REGISTERS:
        raise SystemExit(f"--chips: unknown register {tok} (known: release, rect:r0-r1,c0-c1, "
                         + ", ".join(sorted(nq.REGISTERS)) + ")")
    s = nq.REGISTERS[tok]
    return dict(kind="register", name=tok, rows=list(s["rows"]), cols=list(s["cols"]),
                drop=s.get("drop"), n=s["n"])


def chip_layout(spec, repo):
    if spec["kind"] == "release":
        return nq.Layout(repo)
    if spec["kind"] == "rect" and spec["name"] not in nq.REGISTERS:
        nq.REGISTERS[spec["name"]] = dict(n=spec["n"], rows=spec["rows"], cols=spec["cols"], drop=spec["drop"],
                                          about="cloverfield rect chip", ace=None)
    return ChipLayout(repo, spec["name"], spec["rows"], spec["cols"], spec["drop"], spec["n"])


def chip_engine_args(spec, opts):
    """Namespace for nighthawk_qrack's engines: the chip register's ACE preset (stored
    placement), Dan's register for the release, or --chip-ace for rect chips."""
    a = argparse.Namespace(backend=opts["backend"], cpu=opts["cpu"], ace_gpu=opts["ace_gpu"],
                           ace_host_pointer=opts.get("ace_host_pointer", False), ace_gpu_shots=False,
                           ace_layout="grid", ace_tiling=False, ace_max_width=None, lrc=4, lrr=4,
                           ace_torus=False, ace_transpile=False, ace_width=None, ace_exclude=None,
                           transpile_engine="exact", transpile_max_width=None, transpile_max_wide=None,
                           transpile_allow_cross=False, transpile_seed=0, transpile_trials=0, transpile_calls=0,
                           gpu_isolation="auto", init_gap=0)
    if opts["backend"] != "ace":
        return a
    if spec["kind"] == "register":
        ace = nq.REGISTERS[spec["name"]]["ace"]
        a.ace_transpile, a.transpile_engine = True, "exact"
        a.ace_width, a.lrc, a.lrr, a.ace_torus = ace["width"], ace["lrc"], ace["lrr"], ace["torus"]
        a.transpile_max_width, a.transpile_max_wide = ace.get("max_width"), ace.get("max_wide")
        return a
    if spec["kind"] == "release":
        p = CHIP_PRESETS["release"]
        a.ace_transpile, a.ace_width, a.lrc, a.lrr = True, p["width"], p["lrc"], p["lrr"]
        a.ace_torus, a.ace_exclude = p["torus"], p["exclude"]
        return a
    ca = opts.get("chip_ace")
    if not ca:
        raise SystemExit(f"chip {spec['name']}: --backend ace on a rect chip needs --chip-ace SIZE:LRC:LRR[:t]")
    f = ca.split(":")
    a.ace_transpile, a.ace_width, a.lrc, a.lrr = True, int(f[0]), int(f[1]), int(f[2])
    a.ace_torus = len(f) > 3 and f[3] in ("t", "torus")
    a.transpile_max_width, a.transpile_max_wide = opts.get("max_width"), opts.get("max_wide")
    return a


# ================================================================== ghost room
class GhostRoom:
    """Where a chip can hold ghosts: its ACE placement, the sites ACE couples, and the patch
    widths, under the width caps (the chip register's preset caps, or --ghost-max-width).
    take(anchor) reserves a free site coupled to the anchor's site, or returns None when
    every such site would break the caps. Exact chips have room for any number."""

    def __init__(self, spec, opts):
        from collections import Counter
        self.exact = opts["backend"] != "ace"
        self.sites = []
        if self.exact:
            return
        a = chip_engine_args(spec, opts)
        size = int(a.ace_width)
        if spec["kind"] == "register":
            site = list(nq.REGISTERS[spec["name"]]["ace"]["site"])
            row = nq.layout_row(size, a.lrc, a.lrr, a.ace_torus, {})
        else:
            lay = nq.Layout(None) if spec["kind"] == "release" else chip_layout(spec, None)
            row = nq.layout_row(size, a.lrc, a.lrr, a.ace_torus, {})
            site = nq.transpile_layout(lay, lay.n, a, row, size)
        self.size, self.site = size, site
        self.edges = {tuple(sorted(e)) for e in nq.ace_coupling_edges(row, size, a.ace_torus, {}, ())}
        self.sids = [frozenset(s_ for s_, _ in u) for u in row["unpack"]]
        self.used = set(site)
        self.width = Counter(sid for x in site for sid in self.sids[x])
        self.patches = sorted({next(iter(s_)) for s_ in self.sids if len(s_) == 1})
        # admission: explicit --ghost-max-width caps; with --gtt the die split (VRAM + host-mapped
        # memory) instead of the register's own width caps; otherwise those caps (one 8 GB die)
        self.mem = (float(opts.get("vram_gb") or 8.0), float(opts["gtt"])) if opts.get("gtt") is not None else None
        if opts.get("ghost_max_width"):
            W, N = opts.get("ghost_max_width"), opts.get("ghost_max_wide")
        elif self.mem is None:
            W, N = a.transpile_max_width, a.transpile_max_wide
        else:
            W, N = None, None
        self.W, self.N = W, N

    def widths(self, extra=()):
        from collections import Counter
        w = Counter(self.width)
        for x in extra:
            for s_ in self.sids[x]:
                w[s_] += 1
        return [w[p_] + 1 for p_ in self.patches]

    def take(self, anchor):
        if self.exact:
            self.sites.append(None)
            return True
        sa = self.site[anchor]
        cands = [x for x in range(self.size) if x not in self.used and (min(x, sa), max(x, sa)) in self.edges]
        ok = []
        for x in cands:
            w = self.widths([x])
            if self.W and (max(w) > self.W or (self.N is not None and sum(1 for v in w if v == self.W) > self.N)):
                continue
            if self.mem is not None and not nq.die_split(w, *self.mem)["fits"]:
                continue
            ok.append((len(self.sids[x]) > 1, max(w), x))
        if not ok:
            return False
        x = min(ok)[2]
        self.used.add(x)
        for s_ in self.sids[x]:
            self.width[s_] += 1
        self.sites.append(x)
        return True

    def gib(self):
        if self.exact:
            return None
        g = [2 ** w * 8 / 2 ** 30 for w in self.widths()]
        return sum(g) + max(g)

    def split(self, vram_gb=8.0, gtt_gb=0.0):
        return None if self.exact else nq.die_split(self.widths(), vram_gb, gtt_gb)


# ================================================================== the field
class Field:
    """Chips on a grid plus the links between neighbours; a Layout-like object for
    nighthawk_qrack's circuit generation (n, schedule, matchings, partitions, inputs)."""

    def __init__(self, repo, chip_specs, arrangement=None, links="all", link_rows=None, link_cols=None,
                 ghost=None):
        self.repo = nq.find_repo(repo)
        L = json.loads((self.repo / "data/layout.json").read_text())
        self.R_dev, self.C_dev = L["device_lattice"]["rows"], L["device_lattice"]["cols"]
        self.specs = chip_specs
        k = len(chip_specs)
        gr, gc = arrangement or (1, k)
        if gr * gc != k:
            raise SystemExit(f"--arrangement {gr}x{gc} holds {gr * gc} chips, --chips gives {k}")
        self.arr = (gr, gc)
        self.lays = [chip_layout(s, repo) for s in chip_specs]
        self.chip_n = [lay.n for lay in self.lays]
        self.off = [int(x) for x in np.cumsum([0] + self.chip_n[:-1])]
        self.n = sum(self.chip_n)
        self.chip_of = []
        for c, nn in enumerate(self.chip_n):
            self.chip_of += [c] * nn
        self.local = [q - self.off[self.chip_of[q]] for q in range(self.n)]
        self.cells = [[(p // self.C_dev, p % self.C_dev) for p in lay.l2p] for lay in self.lays]
        self.schedule = L["schedule"]
        self.base_seed = L["base_seed"]
        self.instances = L["instances"]
        self.depths = L["depths"]
        # intra-chip couplers, global indices
        self.matchings = {col: [] for col in COLOURS}
        for c, lay in enumerate(self.lays):
            for col, es in lay.matchings.items():
                self.matchings[col] += [(u + self.off[c], v + self.off[c]) for u, v in es]
        used = [set() for _ in range(self.n)]
        for col, es in self.matchings.items():
            for u, v in es:
                used[u].add(col)
                used[v].add(col)
        # links between grid neighbours; with ghost=engine opts, only where both chips have ghost room
        self.rooms = [GhostRoom(s, ghost) for s in chip_specs] if ghost else None
        self.ghost_sites = [[] for _ in chip_specs]       # (partner global, anchor local, ACE site or None)
        self.links, self.skipped = [], []
        for c in range(k):
            ci, cj = divmod(c, gc)
            for dc, horiz in (((ci, cj + 1), True), ((ci + 1, cj), False)):
                if dc[0] >= gr or dc[1] >= gc:
                    continue
                d = dc[0] * gc + dc[1]
                self._link_pair(c, d, horiz, links, link_rows if horiz else link_cols, used)
        for (u, v, col) in self.links:
            self.matchings[col].append((u, v))
        self.link_set = {(min(u, v), max(u, v)) for u, v, _ in self.links}
        # release recipe on the global register: K = 3 pseudo-patch cuts and the inputs
        pt = nq._load_release_module(self.repo, "patching")
        edges = sorted({tuple(sorted(e)) for es in self.matchings.values() for e in es})
        self.edges = edges
        try:
            parts = pt.make_patches(self.n, edges, 3, seed=self.base_seed, balance_tol=1, top_n=5)
        except Exception:                           # registers too small for three patches (selftest)
            parts = []
        self.partitions = {3: [dict(boundary=frozenset(tuple(sorted(e)) for e in b),
                                    patches=[sorted(v) for _, v in sorted(pq.items(), key=lambda kv: int(kv[0]))])
                               for b, pq in parts]}
        rng = np.random.default_rng(seed=self.base_seed)
        self.inputs = []
        for _ in range(len(L["mirror_input_strings"])):
            bits = np.array(["0"] * self.n)
            bits[rng.choice(self.n, size=self.n // 2, replace=False)] = "1"
            self.inputs.append("".join(bits))
        # drawing / plan geometry: chip tiles side by side with a one-column gap
        self.pos = []
        for c in range(k):
            ci, cj = divmod(c, gc)
            for (r, cc) in self.cells[c]:
                self.pos.append((ci * (self.R_dev + 1) + r, cj * (self.C_dev + 1) + cc))
        self.grid = (gr * (self.R_dev + 1), gc * (self.C_dev + 1))
        self.name = "cloverfield"

    def _link_pair(self, c, d, horiz, links, chosen, used):
        """Facing border sites of chips c (left/top) and d (right/bottom): per row (horizontal
        neighbours) c's rightmost and d's leftmost site, per column (vertical) c's lowest and
        d's highest; K of them evenly spaced, each on a colour free at both ends."""
        A, B = self.cells[c], self.cells[d]
        lines = sorted({r for r, _ in A} & {r for r, _ in B}) if horiz else \
            sorted({cc for _, cc in A} & {cc for _, cc in B})
        cand = []
        for t in lines:
            if horiz:
                ia = max((i for i, p in enumerate(A) if p[0] == t), key=lambda i: A[i][1])
                ib = min((i for i, p in enumerate(B) if p[0] == t), key=lambda i: B[i][1])
            else:
                ia = max((i for i, p in enumerate(A) if p[1] == t), key=lambda i: A[i][0])
                ib = min((i for i, p in enumerate(B) if p[1] == t), key=lambda i: B[i][0])
            cand.append((t, ia + self.off[c], ib + self.off[d]))
        if chosen:
            want = [x for x in cand if x[0] in set(chosen)]
            rest = []
        elif links == "all":
            want, rest = cand, []
        else:
            K = int(links)
            if K <= 0 or not cand:
                return
            pick = sorted({int(round(x)) for x in np.linspace(0, len(cand) - 1, min(K, len(cand)))})
            want = [cand[i] for i in pick]
            rest = [x for i, x in enumerate(cand) if i not in set(pick)]
        target = len(want)
        pref = ("A", "B", "C", "D") if horiz else ("C", "D", "A", "B")
        placed = 0
        for t, u, v in want + rest:
            if placed >= target:
                break
            free = [col for col in pref if col not in used[u] and col not in used[v]]
            if not free:
                self.skipped.append((c, d, t, u, v))
                continue
            if self.rooms is not None:
                ra, rb = self.rooms[c], self.rooms[d]
                if not ra.take(self.local[u]):
                    self.skipped.append((c, d, t, u, v))
                    continue
                if not rb.take(self.local[v]):
                    # undo the reservation on c
                    x = ra.sites.pop()
                    if x is not None:
                        ra.used.discard(x)
                        for s_ in ra.sids[x]:
                            ra.width[s_] -= 1
                    self.skipped.append((c, d, t, u, v))
                    continue
                self.ghost_sites[c].append((v, self.local[u], ra.sites[-1]))
                self.ghost_sites[d].append((u, self.local[v], rb.sites[-1]))
            used[u].add(free[0])
            used[v].add(free[0])
            self.links.append((u, v, free[0]))
            placed += 1

    def ghost_plan(self):
        """--loom ghost: per chip, the partner qubits it holds a replica of -- for every link
        (u, v), a ghost of v on u's chip anchored at u, and a ghost of u on v's chip anchored
        at v -- as [(partner global index, anchor local index)]. Ghost j of chip c has local
        index chip_n[c] + j."""
        if self.rooms is not None:
            return [[(q, anc, site) for q, anc, site in gs] for gs in self.ghost_sites]
        plan = [[] for _ in self.lays]
        for u, v, _ in self.links:
            for real, partner in ((u, v), (v, u)):
                c = self.chip_of[real]
                if all(g[0] != partner for g in plan[c]):
                    plan[c].append((partner, self.local[real], None))
        return plan

    def chip_bits(self, string):
        return [[int(ch) for ch in string[self.off[c]:self.off[c] + self.chip_n[c]]] for c in range(len(self.lays))]

    def describe(self):
        per = {col: (sum(1 for u, v in es if (min(u, v), max(u, v)) not in self.link_set),
                     sum(1 for u, v in es if (min(u, v), max(u, v)) in self.link_set))
               for col, es in self.matchings.items()}
        return dict(n=self.n, chips=[s["name"] for s in self.specs], chip_n=self.chip_n, arrangement=list(self.arr),
                    links=[[u, v, col] for u, v, col in self.links], couplers_per_colour=per,
                    skipped_links=len(self.skipped))

    def tag(self):
        return hashlib.sha1(json.dumps(dict(chips=self.specs, arr=self.arr,
                                            links=[[u, v, c] for u, v, c in self.links]),
                                       sort_keys=True).encode()).hexdigest()[:10]


# ================================================================== pipe framing
def _send(fh, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    fh.write(struct.pack("<Q", len(data)))
    fh.write(data)
    fh.flush()


def _recv(fh):
    head = fh.read(8)
    if len(head) < 8:
        raise EOFError("pipe closed")
    (size,) = struct.unpack("<Q", head)
    data = fh.read(size)
    if len(data) < size:
        raise EOFError("pipe closed mid-message")
    return pickle.loads(data)


# ================================================================== chip worker (isolated process)
def place_ghosts(eng, anchors, given=None):
    """ACE sites for ghost qubits: for each anchor (the chip's own link qubit), a free site
    that ACE couples to the anchor's site (so the woven CZ between them is applied inside a
    simulator, exactly), preferring bulk sites, then the narrowest resulting patch.
    Returns the sites and the patch widths (incl. ancilla) with the ghosts in."""
    from collections import Counter
    row = nq.layout_row(eng.size, eng.lrc, eng.lrr, eng.torus, eng.opts)
    edges = {tuple(sorted(e)) for e in nq.ace_coupling_edges(row, eng.size, eng.torus, eng.opts, ())}
    sids = [frozenset(s for s, _ in u) for u in row["unpack"]]
    used = set(eng.idx)
    width = Counter(sid for x in eng.idx for sid in sids[x])
    sites = []
    for j, a_ in enumerate(anchors):
        sa = eng.idx[a_]
        if given is not None:                            # the loom's reservation (GhostRoom): check, take
            x = given[j]
            if x in used or (min(x, sa), max(x, sa)) not in edges:
                raise RuntimeError(f"ghost site {x} is taken or not coupled to anchor site {sa}")
            used.add(x)
            sites.append(x)
            for s_ in sids[x]:
                width[s_] += 1
            continue
        cands = [x for x in range(eng.size) if x not in used and (min(x, sa), max(x, sa)) in edges]
        if not cands:
            raise RuntimeError(f"no free ACE site couples to the link qubit at site {sa}: the chip's ACE register "
                               f"has no room for its ghost (use --loom mean, or a register with spare sites)")
        x = min(cands, key=lambda x_: (len(sids[x_]) > 1, max(width[s_] + 1 for s_ in sids[x_]), x_))
        used.add(x)
        sites.append(x)
        for s_ in sids[x]:
            width[s_] += 1
    patches = sorted({next(iter(s_)) for s_ in sids if len(s_) == 1})
    return sites, [width[p_] + 1 for p_ in patches]


class ChipSim:
    """The engine of one chip, driven by op batches (also used in-process by the selftest)."""

    def __init__(self, spec, repo, opts):
        self.spec = spec
        self.lay = chip_layout(spec, repo)
        self.n = self.lay.n
        self.backend = opts["backend"]
        a = chip_engine_args(spec, opts)
        anchors = list(opts.get("ghost_anchors") or [])
        given = list(opts.get("ghost_sites") or [])
        self.ghost = dict(count=len(anchors))
        if self.backend == "ace":
            self.eng = nq.AceEngine(self.lay, self.n, a)
            if anchors:
                sites, widths = place_ghosts(self.eng, anchors, given if given and None not in given else None)
                self.eng.idx = list(self.eng.idx) + sites   # ghost j = logical n + j; prob_target still reads n
                self.ghost.update(sites=sites, widths=widths)
        else:
            self.eng = nq.ExactEngine(self.n + len(anchors), opts["cpu"])
            self.eng.n = self.n                             # prob_bits marginalises the ghosts

    def info(self):
        g = self.eng.geometry() if self.backend == "ace" else {}
        return dict(n=self.n, pid=os.getpid(), geometry={k: v for k, v in g.items() if k != "ace_map"},
                    ace_map=g.get("ace_map"), device=os.environ.get("NIGHTHAWK_ISOLATED"), ghosts=self.ghost,
                    alloc_mb=os.environ.get("QRACK_MAX_ALLOC_MB"), gtt_gb=os.environ.get("NIGHTHAWK_GTT_GB"))

    def marg(self, q):
        if self.backend == "ace":
            return float(self.eng.sim.prob(self.eng.idx[q]))
        return float(self.eng.sim.prob(q))

    def batch(self, ops, query):
        e = self.eng
        for op in ops:
            k = op[0]
            if k == "g":
                M = nq.gate(op[2])
                e.g1(op[1], M.conj().T if op[3] else M)
            elif k == "p":
                e.pauli(op[1], op[2])
            elif k == "cz":
                e.cz(op[1], op[2])
            elif k == "ph":
                e.g1(op[1], np.diag([1.0, np.exp(1j * op[2])]))
            elif k == "reset":
                e.reset()
            else:
                raise ValueError(f"unknown op {k}")
        return [self.marg(q) for q in query]

    def prob(self, bits, seed):
        if self.backend != "ace":
            return float(self.eng.prob_bits(bits)), None
        try:
            return float(self.eng.prob_target(bits, np.random.default_rng(seed))), None
        except nq.ProbTargetError as err:
            self.eng.rebuild()
            return 0.0, float(err.args[0])


def cmd_chipworker(a):
    """Started by the loom with the environment of one device. Protocol (length-prefixed
    pickles on stdin/stdout): init, batch(ops, query) -> marginals, prob(bits, seed), quit.
    stdout is taken over for the protocol; anything Qrack prints goes to stderr."""
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    inp = sys.stdin.buffer
    sim = None
    while True:
        try:
            msg = _recv(inp)
        except EOFError:
            return
        try:
            k = msg[0]
            if k == "init":
                nq.THETA_DIST["mode"] = msg[3]
                sim = ChipSim(msg[1], msg[2], msg[4])
                _send(out, ("ok", sim.info()))
            elif k == "batch":
                _send(out, ("m", sim.batch(msg[1], msg[2])))
            elif k == "prob":
                _send(out, ("p",) + sim.prob(msg[1], msg[2]))
            elif k == "quit":
                _send(out, ("bye",))
                return
        except Exception:
            _send(out, ("err", traceback.format_exc()))


class ChipProc:
    """The loom's handle on one chip worker process."""

    def __init__(self, index, spec, repo, opts, env, theta):
        self.index, self.spec = index, spec
        cmd = [sys.executable, os.path.abspath(__file__), "chipworker"]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        _send(self.p.stdin, ("init", spec, repo, theta, opts))
        self.info = self.get("ok")[1]

    def put(self, msg):
        _send(self.p.stdin, msg)

    def get(self, want):
        r = _recv(self.p.stdout)
        if r[0] == "err":
            raise RuntimeError(f"chip {self.index} ({self.spec['name']}): {r[1]}")
        if r[0] != want:
            raise RuntimeError(f"chip {self.index}: expected {want}, got {r[0]}")
        return r

    def close(self):
        try:
            self.put(("quit",))
            self.get("bye")
        except Exception:
            pass
        try:
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


class LocalChip:
    """In-process stand-in for ChipProc (selftest)."""

    def __init__(self, index, spec, repo, opts, theta):
        nq.THETA_DIST["mode"] = theta
        self.index, self.spec, self.sim = index, spec, ChipSim(spec, repo, opts)
        self.info, self._r = self.sim.info(), None

    def put(self, msg):
        k = msg[0]
        self._r = ("m", self.sim.batch(msg[1], msg[2])) if k == "batch" else ("p",) + self.sim.prob(msg[1], msg[2])

    def get(self, want):
        return self._r

    def close(self):
        pass


# ================================================================== the loom
class Loom:
    """Weaves one circuit run over the chips. Holds one handle per chip; all IO between
    chips passes through here.

    Every qubit has a real copy on its own chip and, with --loom ghost, a ghost on each chip
    it is linked to. Every single-qubit gate and twirl Pauli goes to all copies. A CZ (u, v)
    is applied on every chip that holds a copy of u or v: exactly where both are held (an
    intra-chip CZ on the qubits' own chip; a link CZ between the real qubit and the partner's
    ghost, on both chips), and as the mean-field phase pi * P(1) of the absent qubit where
    only one is held (its marginal read from the absent qubit's real copy). With --loom mean
    there are no ghosts, so a link CZ is the mean-field phase on both sides. --loom sample
    draws a Z kick from P(1) instead of the phase."""

    def __init__(self, field, chips, mode="mean"):
        self.f, self.chips, self.mode = field, chips, mode
        self.round_trips = 0
        self.ghost_of = {}                               # global q -> [(chip, local ghost index)]
        if mode == "ghost":
            for c, plan in enumerate(field.ghost_plan()):
                for j, (q, _, _) in enumerate(plan):
                    self.ghost_of.setdefault(q, []).append((c, field.chip_n[c] + j))
        self.drift = []

    def reps(self, q):
        return [(self.f.chip_of[q], self.f.local[q])] + self.ghost_of.get(q, [])

    def _emit1(self, queues, q, op_head, *tail):
        for c, l_ in self.reps(q):
            queues[c].append((op_head, l_) + tail if op_head != "p" else ("p", tail[0], l_))

    def _flush(self, queues, query):
        """Send every chip its pending ops plus a marginal query; collect in parallel."""
        for c, ch in enumerate(self.chips):
            ch.put(("batch", queues[c], query[c]))
        res = [ch.get("m")[1] for ch in self.chips]
        for q_ in queues:
            q_.clear()
        self.round_trips += 1
        return res

    def _layer(self, es, rng, queues, srng):
        f = self.f
        need = set()                                     # qubits whose real marginal some chip needs
        for u, v in es:
            ru, rv = {c for c, _ in self.reps(u)}, {c for c, _ in self.reps(v)}
            if ru - rv:
                need.add(v)
            if rv - ru:
                need.add(u)
        marg = {}
        ghosted = sorted({q for e in es for q in e if q in self.ghost_of}) if need else []
        if need:
            query = [[] for _ in self.chips]
            for q in sorted(need | set(ghosted)):
                query[f.chip_of[q]].append(("r", q, f.local[q]))
            for q in ghosted:                            # ghost drift: replica vs real, same moment
                for c, l_ in self.ghost_of[q]:
                    query[c].append(("g", q, l_))
            res = self._flush(queues, [[x[2] for x in qs] for qs in query])
            gm = []
            for c, qs in enumerate(query):
                for (kind, q, _), p in zip(qs, res[c]):
                    if kind == "r":
                        marg[q] = p
                    else:
                        gm.append((q, p))
            self.drift += [abs(p - marg[q]) for q, p in gm]
        for a_, b_ in es:
            pa, pb = (nq.PAULIS[rng.integers(4)], nq.PAULIS[rng.integers(4)]) if rng is not None else ("i", "i")
            if pa != "i":
                self._emit1(queues, a_, "p", pa)
            if pb != "i":
                self._emit1(queues, b_, "p", pb)
            ra, rb = dict(self.reps(a_)), dict(self.reps(b_))
            for c in sorted(set(ra) | set(rb)):
                if c in ra and c in rb:
                    queues[c].append(("cz", ra[c], rb[c]))
                    continue
                here, other, po = (ra[c], b_, pb) if c in ra else (rb[c], a_, pa)
                p1 = 1.0 - marg[other] if po in "xy" else marg[other]
                if self.mode == "sample":
                    if srng.random() < p1:
                        queues[c].append(("p", "z", here))
                else:
                    queues[c].append(("ph", here, math.pi * p1))
            if pa != "i":
                self._emit1(queues, a_, "p", pa)
            if pb != "i":
                self._emit1(queues, b_, "p", pb)
            if pb in "xy":
                self._emit1(queues, a_, "p", "z")
            if pa in "xy":
                self._emit1(queues, b_, "p", "z")

    def run(self, cyc, bits_per_chip, rng, prob_seed, srng=None, tick=None):
        """One mirror circuit run (forward, then the exact inverse), ops in nighthawk_qrack's
        order (apply_forward / apply_inverse / cz_layer, same twirl draws); returns the
        product over chips of the probability of their part of the prepared string, and the
        partial products of chips whose backend raised."""
        f = self.f
        queues = [[("reset",)] for _ in self.chips]
        for c, bits in enumerate(bits_per_chip):
            for q, b in enumerate(bits):
                if b:
                    self._emit1(queues, f.off[c] + q, "p", "x")
        for ang, es in cyc:
            for q, t in enumerate(ang):
                self._emit1(queues, q, "g", tuple(t), False)
            self._layer(es, rng, queues, srng)
            if tick:
                tick()
        for ang, es in reversed(cyc):
            self._layer(list(reversed(es)), rng, queues, srng)
            for q in reversed(range(len(ang))):
                self._emit1(queues, q, "g", tuple(ang[q]), True)
            if tick:
                tick()
        self._flush(queues, [[] for _ in self.chips])
        for c, ch in enumerate(self.chips):
            ch.put(("prob", bits_per_chip[c], prob_seed + ([c] if len(self.chips) > 1 else [])))
        P, errs = 1.0, []
        for ch in self.chips:
            r = ch.get("p")
            P *= r[1]
            if r[2] is not None:
                errs.append(r[2])
        return P, errs


def run_point(field, loom, d, inst, a, log=None):
    """One (depth, instance) point: every input x twirl, as run_mirror_prob."""
    cyc = nq.mirror_cycles(field, d, inst, n=field.n)
    R = max(a.twirls, 1) if a.backend == "ace" else 1
    per_run, per_input, errors = [], [], []
    t0, last, done = time.time(), time.time(), 0
    total = len(field.inputs) * R
    for s, string in enumerate(field.inputs):
        bits = field.chip_bits(string)
        ps = []
        for r in range(R):
            rng = np.random.default_rng([nq.BASE_SEED, d, inst, s, r]) if (a.twirls and a.backend == "ace") else None
            srng = np.random.default_rng([nq.BASE_SEED, d, inst, s, r, 11])
            P, errs = loom.run(cyc, bits, rng, [nq.BASE_SEED, d, inst, s, r, 7], srng)
            ps.append(P)
            if errs:
                errors.append(float(np.prod(errs)))
            done += 1
            if log and time.time() - last > 60:
                el = time.time() - t0
                log(f"# progress d{d} i{inst}: run {done}/{total}, {el / 60:.1f} min, "
                    f"ETA {el / done * (total - done) / 60:.1f} min")
                last = time.time()
        per_run += ps
        per_input.append(float(np.mean(ps)))
    x = np.array(per_run)
    se = float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0
    out = dict(fidelity=float(x.mean()), se=se, survival=per_input, shots=0, runs=len(x), estimator="prob",
               seconds=round(time.time() - t0, 2))
    if loom.drift:
        out["ghost_drift"] = float(np.mean(loom.drift))      # mean |P(1) of a ghost - P(1) of its real qubit|
        loom.drift = []
    if errors:
        out.update(prob_errors=len(errors), prob_error_bound=float(sum(errors) / len(x)))
    return out


# ================================================================== run / status / summarize
def build_field(a):
    nq.load_register_files()
    import re
    toks = []
    # rect:r0-r1,c0-c1 contains a comma itself: take those tokens whole
    for t in re.findall(r"rect:\d+-\d+,\d+-\d+(?:\*\d+)?|[^,]+", a.chips):
        if "*" in t:
            name, k = t.rsplit("*", 1)
            toks += [name] * int(k)
        else:
            toks.append(t)
    specs = [parse_chip(t, a.repo, getattr(a, "broken", "disable")) for t in toks]
    arr = tuple(map(int, a.arrangement.lower().split("x"))) if a.arrangement else None
    rows = [int(x) for x in a.link_rows.split(",")] if getattr(a, "link_rows", None) else None
    cols = [int(x) for x in a.link_cols.split(",")] if getattr(a, "link_cols", None) else None
    ghost = engine_opts(a) if getattr(a, "loom", "mean") == "ghost" and hasattr(a, "backend") else None
    return Field(a.repo, specs, arr, a.links, rows, cols, ghost=ghost)


def engine_opts(a):
    return dict(backend=a.backend, cpu=a.cpu, ace_gpu=a.ace_gpu, ace_host_pointer=getattr(a, "ace_host_pointer", False),
                chip_ace=getattr(a, "chip_ace", None), max_width=getattr(a, "chip_max_width", None),
                max_wide=getattr(a, "chip_max_wide", None), ghost_max_width=getattr(a, "ghost_max_width", None),
                ghost_max_wide=getattr(a, "ghost_max_wide", None), gtt=getattr(a, "gtt", None),
                vram_gb=getattr(a, "vram_gb", 8.0))


def cfg_tag(field, a):
    keys = dict(field=field.tag(), backend=a.backend, twirls=a.twirls if a.backend == "ace" else 0,
                theta=a.theta, loom=a.loom, chip_ace=getattr(a, "chip_ace", None))
    return "field-" + hashlib.sha1(json.dumps(keys, sort_keys=True).encode()).hexdigest()[:8]


def load_records(path, tag):
    recs = {}
    try:
        for line in Path(path).read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("cfg") == tag and r.get("family") == "mirror":
                recs[(r["depth"], r["instance"])] = r
    except OSError:
        pass
    return recs


def loom_env():
    """The loom builds no simulator: it runs with OpenCL hidden, and keeps the user's
    original environment for the workers (nighthawk_qrack's base_env restores it)."""
    if os.environ.get("NIGHTHAWK_LOOM") == "1":
        return
    env = dict(os.environ)
    env["NIGHTHAWK_ORIG_ENV"] = json.dumps({k: env.get(k) for k in nq._ISO_KEYS})
    env["NIGHTHAWK_ISOLATED"] = "loom"
    env["NIGHTHAWK_LOOM"] = "1"
    os.makedirs("/tmp/no-icd", exist_ok=True)
    env["OCL_ICD_VENDORS"] = "/tmp/no-icd"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


def worker_env_for(a, device):
    if a.cpu or device is None:
        env = nq.base_env()
        os.makedirs("/tmp/no-icd", exist_ok=True)
        env["OCL_ICD_VENDORS"] = "/tmp/no-icd"
        env["NIGHTHAWK_ISOLATED"] = "cpu"
        return env
    env = nq.worker_env(device, argparse.Namespace(gpu_isolation=a.gpu_isolation))
    if getattr(a, "gtt", None):
        nq.gtt_env(env, float(a.vram_gb or 8.0), float(a.gtt))
    return env


def start_chips(field, a, loom_index):
    gpus = nq.parse_gpus(a.gpus) if a.gpus else []
    plan = field.ghost_plan() if a.loom == "ghost" else [[] for _ in field.specs]
    chips = []
    for c, spec in enumerate(field.specs):
        dev = None
        if gpus and not a.cpu:
            slot = loom_index * len(field.specs) + c
            dev = gpus[(slot // max(1, a.chips_per_gpu)) % len(gpus)]
        opts = dict(engine_opts(a), ghost_anchors=[anc for _, anc, _ in plan[c]],
                    ghost_sites=[site for _, _, site in plan[c]])
        chips.append(ChipProc(c, spec, a.repo, opts, worker_env_for(a, dev), a.theta))
    return chips


def cmd_run(a):
    nq.THETA_DIST["mode"] = a.theta
    field = build_field(a)
    tag = cfg_tag(field, a)
    depths = a.depths or [4, 6, 8, 10, 12]
    insts = range(a.instances if a.instances is not None else field.instances)
    keys = [(d, i) for d in depths for i in insts]
    recs = load_records(a.out, tag)
    failed_path = Path(a.out).with_suffix(".failed.json")
    try:
        failed = set(tuple(x) for x in json.loads(failed_path.read_text()).get(tag, []))
    except (OSError, ValueError):
        failed = set()
    if a.retry_failed:
        failed = set()
    desc = field.describe()
    if a.status or a.summarize:
        print(f"# {tag} ({a.out}): {len(recs)} of {len(keys)} points done"
              + (f", {len(failed)} failed" if failed else ""))
        if a.status:
            for k in keys:
                r = recs.get(k)
                print(f"  n{field.n} mirror d{k[0]} i{k[1]}  " + (f"done     F = {r['fidelity']:.4g}" if r else
                                                                ("failed" if k in failed else "pending")))
        if a.summarize:
            summarize(field, recs, depths)
        return
    print(f"# cloverfield {tag}: {len(field.specs)} chips {desc['chips']} ({desc['arrangement'][0]}x"
          f"{desc['arrangement'][1]}), {field.n} qubits, {len(field.links)} links"
          + (f" ({len(field.skipped)} facing pairs skipped: no common free colour" + (", or no ghost room under the caps" if field.rooms else "") + ")" if field.skipped else "")
          + f", loom {a.loom}, backend {a.backend}, theta {a.theta}", flush=True)
    todo = queue.Queue()
    for k in sorted(keys, key=lambda k: (-k[0], k[1]) if a.order == "deep" else k):
        if k not in recs and k not in failed:
            todo.put(k)
    if todo.empty():
        print("# nothing to do")
        summarize(field, recs, depths)
        return
    gpus = nq.parse_gpus(a.gpus) if a.gpus else []
    looms = a.parallel or (max(1, len(gpus) * max(1, a.chips_per_gpu) // len(field.specs)) if gpus and not a.cpu else 1)
    lock = threading.Lock()
    fails = dict()

    def log(msg):
        with lock:
            print(msg, flush=True)

    def worker(li):
        chips = None
        try:
            while True:
                try:
                    k = todo.get_nowait()
                except queue.Empty:
                    return
                for attempt in (1, 2):
                    try:
                        if chips is None:
                            chips = start_chips(field, a, li)
                            log(f"# loom {li}: chips " + ", ".join(
                                f"{c.spec['name']} pid {c.info['pid']} on {c.info['device'] or 'cpu'}"
                                + (f" [GTT {c.info['gtt_gb']} GB, QRACK_MAX_ALLOC_MB={c.info['alloc_mb']}]" if c.info.get("gtt_gb") else "")
                                + (f" ({c.info['ghosts']['count']} ghosts"
                                   + (f", patch widths {c.info['ghosts']['widths']}" if c.info['ghosts'].get('widths') else "")
                                   + ")" if c.info.get("ghosts", {}).get("count") else "") for c in chips))
                        loom = Loom(field, chips, a.loom)
                        res = run_point(field, loom, k[0], k[1], a, log)
                        rec = dict(cfg=tag, n=field.n, family="mirror", K=0, depth=k[0], partition=0, instance=k[1],
                                   theta=a.theta, loom=a.loom, twirls=a.twirls if a.backend == "ace" else 0,
                                   backend=a.backend, field=desc, round_trips=loom.round_trips,
                                   chips_geometry=[c.info["geometry"] for c in chips],
                                   chips_map=[c.info["ace_map"] for c in chips],
                                   chips_ghosts=[c.info.get("ghosts") for c in chips], **res)
                        with lock:
                            with open(a.out, "a") as fh:
                                fh.write(json.dumps(rec) + "\n")
                        log(f"n {field.n} mirror d{k[0]:>3} inst {k[1]}  F {res['fidelity']:.5g} +/- {res['se']:.1e}"
                            + (f"  ghost drift {res['ghost_drift']:.3g}" if "ghost_drift" in res else "")
                            + f"  ({res['seconds']} s, loom {li})")
                        break
                    except Exception as err:
                        log(f"# loom {li}: d{k[0]} i{k[1]} attempt {attempt} failed: {err}")
                        if chips:
                            for c in chips:
                                try:
                                    c.p.kill()
                                except Exception:
                                    pass
                        chips = None
                        if attempt == 2:
                            with lock:
                                fails[k] = str(err)[:500]
        finally:
            if chips:
                for c in chips:
                    c.close()

    th = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(looms)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if fails:
        try:
            allf = json.loads(failed_path.read_text())
        except (OSError, ValueError):
            allf = {}
        allf[tag] = sorted(set(map(tuple, allf.get(tag, []))) | set(fails))
        failed_path.write_text(json.dumps(allf))
        print(f"# {len(fails)} point(s) failed twice, listed in {failed_path} (--retry-failed reruns them)")
    summarize(field, load_records(a.out, tag), depths)


def summarize(field, recs, depths):
    print(f"\n== cloverfield, n = {field.n}, {len(field.specs)} chips, {len(field.links)} links ==")
    print("d     instances                          weighted          plain mean")
    for d in sorted(depths):
        rs = [recs[k] for k in sorted(recs) if k[0] == d]
        if not rs:
            continue
        F = np.array([r["fidelity"] for r in rs])
        se = np.array([max(r["se"], 1e-12) for r in rs])
        w = 1 / se ** 2
        wm, wse = float((w * F).sum() / w.sum()), float(1 / math.sqrt(w.sum()))
        inst = "  ".join(f"{r['fidelity']:.4g}" for r in rs)
        print(f"{d:<5} {inst:<34} {wm:.5g}+/-{wse:.1e}   {F.mean():.5g}")


# ================================================================== plan
def chip_memory(spec, opts):
    if opts["backend"] != "ace" or spec["kind"] == "rect" and not opts.get("chip_ace"):
        return None, None
    a = chip_engine_args(spec, opts)
    if spec["kind"] == "register":
        lay = nq.RegisterLayout(None, spec["name"])
    elif spec["kind"] == "release":
        lay = nq.Layout(None)
    else:
        lay = chip_layout(spec, None)
    row = nq.layout_row(int(a.ace_width), a.lrc, a.lrr, a.ace_torus, {})
    site = nq.transpile_layout(lay, lay.n, a, row, int(a.ace_width))
    sids = [frozenset(s for s, _ in u) for u in row["unpack"]]
    patches = sorted({next(iter(s)) for s in sids if len(s) == 1})
    widths = [sum(1 for x in site if p in sids[x]) + 1 for p in patches]
    gib = [2 ** w * 8 / 2 ** 30 for w in widths]
    return widths, sum(gib) + max(gib)


def cmd_plan(a):
    field = build_field(a)
    desc = field.describe()
    print(f"cloverfield: {len(field.specs)} chips {desc['chips']} on a {desc['arrangement'][0]} x "
          f"{desc['arrangement'][1]} grid, {field.n} qubits")
    print("colour   intra-chip CZ   link CZ")
    for col in COLOURS:
        i_, l_ = desc["couplers_per_colour"][col]
        print(f"  {col}      {i_:>8}        {l_:>4}")
    print(f"links: {len(field.links)}, loom {a.loom}" + (f" ({len(field.skipped)} facing pairs skipped: no colour free "
                                         f"at both ends{', or no ghost room under the caps' if field.rooms else ''})"
                                         if field.skipped else ""))
    for u, v, col in field.links:
        cu, cv = field.chip_of[u], field.chip_of[v]
        print(f"  chip {cu} site {field.cells[cu][field.local[u]]} -- chip {cv} site {field.cells[cv][field.local[v]]}"
              f"  colour {col}")
    opts = engine_opts(a)
    total, splits = [], []
    for c, spec in enumerate(field.specs):
        widths, gb = chip_memory(spec, opts)
        ng = 0
        if field.rooms is not None and not field.rooms[c].exact:
            widths, gb, ng = field.rooms[c].widths(), field.rooms[c].gib(), len(field.rooms[c].sites)
        total.append(gb or 0)
        sp = nq.die_split(widths, a.vram_gb or 8.0, a.gtt or 0.0) if widths else None
        if sp:
            splits.append(sp)
        print(f"chip {c}: {spec['name']}, {field.chip_n[c]} qubits"
              + (f" + {ng} ghosts" if ng else "")
              + (f", patch widths {widths} (incl. ancilla), peak about {gb:.1f} GiB fp32" if widths else "")
              + (f"\n        on one die: {nq.describe_split(sp, a.vram_gb or 8.0, a.gtt or 0.0)}" if sp else ""))
    if any(total):
        per = a.chips_per_gpu
        ok = all(sp["fits"] for sp in splits) and per == 1 if splits else max(total) * per <= a.die_gb
        gtt_host = sum(sp["gtt"] for sp in splits)
        print(f"one loom = {len(field.specs)} chip workers, {per} chip(s) per {a.vram_gb or a.die_gb:g} GB device: "
              f"{'fits' if ok else 'DOES NOT FIT'}"
              + (f" (host memory mapped as GTT: {gtt_host:.1f} GiB per loom)" if gtt_host else "")
              + f"; devices per loom: {math.ceil(len(field.specs) / per)}")
    if a.png:
        draw_field(field, a.png)
        print(f"drawn: {a.png}")


def draw_field(field, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    H, W = field.grid
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * W, 1.0 + 0.55 * H), dpi=110)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1, W)
    ax.set_ylim(H, -1)
    for col, es in field.matchings.items():
        for u, v in es:
            (r1, c1), (r2, c2) = field.pos[u], field.pos[v]
            lk = (min(u, v), max(u, v)) in field.link_set
            ax.plot([c1, c2], [r1, r2], color="#0b0b0b" if lk else "#cfcfca", lw=3.2 if lk else 1.2, zorder=1 if not lk else 3)
    for q, (r, c) in enumerate(field.pos):
        ax.scatter([c], [r], s=60, color=cols[field.chip_of[q] % len(cols)], zorder=4, edgecolors="white", linewidths=1)
    ax.set_title(f"cloverfield: {len(field.specs)} chips, {field.n} qubits, {len(field.links)} links (black)",
                 fontsize=12, loc="left")
    fig.savefig(path, bbox_inches="tight", facecolor="white")


# ================================================================== linksearch
def cmd_linksearch(a):
    nq.THETA_DIST["mode"] = a.theta
    rows = []
    for k in a.link_counts:
        a.links = k
        field = build_field(a)
        chips = start_chips(field, a, 0)
        try:
            loom = Loom(field, chips, a.loom)
            res = {}
            for d in a.depths:
                vals = [run_point(field, loom, d, i, a)["fidelity"] for i in range(a.instances)]
                res[d] = float(np.mean(vals))
        finally:
            for c in chips:
                c.close()
        score = math.exp(sum(math.log(max(v, 1e-9)) for v in res.values()) / len(res))
        rows.append((k, len(field.links), res, score))
        print(f"links {k:>4} ({len(field.links)} placed): " + "  ".join(f"d{d} {v:.4g}" for d, v in res.items())
              + f"  score {score:.4g}", flush=True)
    print("\nlinks  placed  " + "  ".join(f"d{d:<8}" for d in a.depths) + "  score")
    for k, nl, res, sc in rows:
        print(f"{str(k):<6} {nl:>6}  " + "  ".join(f"{res[d]:<9.4g}" for d in a.depths) + f"  {sc:.4g}")


# ================================================================== selftest
def cmd_selftest(a):
    ok = True

    def report(name, good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"{name}: {msg}  {'OK' if good else 'FAIL'}", flush=True)

    nq.load_register_files()
    nq.THETA_DIST["mode"] = "nnqab"
    # [1] field construction
    fa = argparse.Namespace(repo=a.repo, chips="nighthawk88e*2", arrangement=None, links="4", link_rows=None, link_cols=None)
    f1, f2 = build_field(fa), build_field(fa)
    match_ok = all(len({q for e in es for q in e}) == 2 * len(es) for es in f1.matchings.values())
    same = f1.inputs == f2.inputs and [p["boundary"] for p in f1.partitions[3]] == [p["boundary"] for p in f2.partitions[3]]
    report("[1] field nighthawk88e*2, 4 links: every colour a matching, deterministic inputs and K = 3 cuts",
           match_ok and same and len(f1.links) == 4, f"{f1.n} qubits, {len(f1.links)} links, "
           f"{sum(len(es) for es in f1.matchings.values())} couplers")
    # [2] IO equivalence. (a) exact and RNG-free: the op stream the loom sends one chip (no
    # links) equals, gate for gate, what nighthawk_qrack's run_mirror_prob sends its engine;
    # (b) the same point through a real worker process, within Qrack's own unseeded spread.
    fa1 = argparse.Namespace(repo=a.repo, chips="nighthawk88e", arrangement=None, links="0", link_rows=None, link_cols=None)
    f = build_field(fa1)
    lay = nq.RegisterLayout(a.repo, "nighthawk88e")

    class RecEngine:                                 # stands in for nighthawk_qrack's engine
        def __init__(self):
            self.ops, self.n = [], lay.n

        def reset(self):
            self.ops.append(("reset",))

        def pauli(self, p_, q):
            if p_ != "i":
                self.ops.append(("p", p_, q))

        def g1(self, q, M):
            self.ops.append(("g", q, np.asarray(M)))

        def cz(self, a_, b_):
            self.ops.append(("cz", a_, b_))

        def prob_target(self, bits, rng):
            self.ops.append(("prob", tuple(bits)))
            return 0.5

    class RecChip:                                   # stands in for a chip worker
        def __init__(self):
            self.ops, self._r = [], None

        def put(self, msg):
            if msg[0] == "batch":
                for op in msg[1]:
                    if op[0] == "g":
                        M = nq.gate(op[2])
                        self.ops.append(("g", op[1], M.conj().T if op[3] else M))
                    else:
                        self.ops.append(op)
                self._r = ("m", [0.5] * len(msg[2]))
            else:
                self.ops.append(("prob", tuple(msg[1])))
                self._r = ("p", 0.5, None)

        def get(self, want):
            return self._r

    na = chip_engine_args(f.specs[0], dict(backend="ace", cpu=True, ace_gpu=False))
    na.twirls, na.mirror_estimator = 2, "prob"
    rec_e, rec_c = RecEngine(), RecChip()
    nq.run_mirror_prob(lay, rec_e, 4, 0, na, lay.n, nq.mirror_cycles(lay, 4, 0, n=lay.n), 2)
    ra = argparse.Namespace(backend="ace", cpu=True, ace_gpu=False, twirls=2, gpus=None, chips_per_gpu=1,
                            gpu_isolation="auto", theta="nnqab", loom="mean", repo=a.repo)
    run_point(f, Loom(f, [rec_c], "mean"), 4, 0, ra)

    def same(x, y):
        if x[0] != y[0] or len(x) != len(y):
            return False
        if x[0] == "g":
            return x[1] == y[1] and np.allclose(x[2], y[2], atol=1e-12)
        return x == y
    eq = len(rec_e.ops) == len(rec_c.ops) and all(same(x, y) for x, y in zip(rec_e.ops, rec_c.ops))
    report("[2a] one chip, no links: loom op stream == nighthawk_qrack's engine calls (d4, 20 runs)",
           eq, f"{len(rec_c.ops)} vs {len(rec_e.ops)} ops, identical" if eq else
           f"{len(rec_c.ops)} vs {len(rec_e.ops)} ops, first difference at "
           f"{next((i for i, (x, y) in enumerate(zip(rec_e.ops, rec_c.ops)) if not same(x, y)), min(len(rec_e.ops), len(rec_c.ops)))}")
    chips = start_chips(f, ra, 0)
    try:
        loom_res = run_point(f, Loom(f, chips, "mean"), 4, 0, ra)["fidelity"]
    finally:
        for c in chips:
            c.close()
    eng = nq.AceEngine(lay, lay.n, na)
    ref = nq.run_mirror_prob(lay, eng, 4, 0, na, lay.n, nq.mirror_cycles(lay, 4, 0, n=lay.n), 2)["fidelity"]
    report("[2b] the same point through a worker process vs nighthawk_qrack in-process (Qrack's RNG is unseeded: "
           "~1e-4 between processes)", abs(loom_res - ref) < 1e-3, f"{loom_res:.6f} vs {ref:.6f}")
    # [3] the woven CZ against exact cases (two 1-qubit-wide exact chips, in-process)
    R_, C_, broken, _ = nq.lattice_info(a.repo)
    fe = Field(a.repo, [parse_chip("rect:0-0,0-0", a.repo), parse_chip("rect:0-0,0-0", a.repo)], (1, 2), "all")
    chips = [LocalChip(i, s, a.repo, dict(backend="exact", cpu=True, ace_gpu=False), "nnqab") for i, s in enumerate(fe.specs)]
    lm = Loom(fe, chips, "mean")
    plus = (0.0, math.pi / 2, math.pi / 2)          # rz(pi/2) rx(pi/2) |0> = |+> up to phase

    def xval(prep_b):
        qs = [[("reset",)], [("reset",)]]
        qs[0].append(("g", 0, plus, False))
        if prep_b == "1":
            qs[1].append(("p", "x", 0))
        if prep_b == "+":
            qs[1].append(("g", 0, plus, False))
        lm._layer([(0, 1)], None, qs, None)
        lm._flush(qs, [[], []])
        s = chips[0].sim.eng.sim
        s.h(0)
        return 1.0 - 2.0 * float(s.prob(0))
    xs = {b: xval(b) for b in ("0", "1", "+")}
    report("[3] woven CZ: <X> of |+> after CZ with |0>, |1>, |+> (exact: +1, -1, 0)",
           abs(xs["0"] - 1) < 1e-6 and abs(xs["1"] + 1) < 1e-6 and abs(xs["+"]) < 1e-6,
           ", ".join(f"{k}: {v:+.6f}" for k, v in xs.items()))
    # [4] CZ twice is the identity: the ghost carries the entanglement of the first CZ, the
    # mean-field weave cannot (it turns |+>|+> into |->|-> after two "CZ"s)
    def czcz(mode):
        plan = fe.ghost_plan() if mode == "ghost" else [[], []]
        chs = [LocalChip(i, s_, a.repo, dict(backend="exact", cpu=True, ace_gpu=False,
                                              ghost_anchors=[x for _, x, _ in plan[i]]), "nnqab")
               for i, s_ in enumerate(fe.specs)]
        lm2 = Loom(fe, chs, mode)
        qs = [[("reset",)], [("reset",)]]
        for q in (0, 1):
            lm2._emit1(qs, q, "g", plus, False)
        lm2._layer([(0, 1)], None, qs, None)
        lm2._layer([(0, 1)], None, qs, None)
        lm2._flush(qs, [[], []])
        s_ = chs[0].sim.eng.sim
        s_.h(0)
        return 1.0 - 2.0 * float(s_.prob(0))
    xg, xm = czcz("ghost"), czcz("mean")
    report("[4] CZ.CZ = I on |+>|+> across the link: <X> of the first qubit (exact +1)",
           abs(xg - 1) < 1e-6, f"ghost {xg:+.6f}, mean-field {xm:+.6f}")
    # [5] the loom's error alone: exact chips, mirror survival (1 without links, < 1 with)
    ea = argparse.Namespace(backend="exact", cpu=True, ace_gpu=False, twirls=0, gpus=None, chips_per_gpu=1,
                            gpu_isolation="auto", theta="nnqab", loom="mean", repo=a.repo)
    out = {}
    for nl, mode in (("0", "ghost"), ("2", "mean"), ("2", "ghost"), ("all", "mean"), ("all", "ghost")):
        ea.loom = mode
        fx = Field(a.repo, [parse_chip("rect:8-10,0-2", a.repo)] * 2, (1, 2), nl)
        ch = start_chips(fx, ea, 0)
        try:
            out[nl, mode] = [run_point(fx, Loom(fx, ch, mode), d, 0, ea)["fidelity"] for d in (8, 12, 16)]
        finally:
            for c in ch:
                c.close()
    fmt = lambda v: " / ".join(f"{x:.4f}" for x in v)
    good = all(abs(v - 1) < 1e-6 for v in out["0", "ghost"]) and \
        all(g >= m - 1e-9 for k in ("2", "all") for g, m in zip(out[k, "ghost"], out[k, "mean"]))
    report("[5] exact chips rect:8-10,0-2 x 2 (18 qubits), loom error alone, d8 / d12 / d16; ghost >= mean-field",
           good, f"0 links {fmt(out['0', 'ghost'])};  2 links mean {fmt(out['2', 'mean'])}, ghost "
                 f"{fmt(out['2', 'ghost'])};  all links mean {fmt(out['all', 'mean'])}, ghost {fmt(out['all', 'ghost'])}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


# ================================================================== CLI
def add_field_args(p):
    p.add_argument("--repo", default=None, help="rcs-nighthawk clone (default: auto-detect)")
    p.add_argument("--chips", default="nighthawk88e*2",
                   help="chip registers, comma list, NAME*K repeats (default nighthawk88e*2): release, nighthawk88, "
                        "nighthawk88d, nighthawk88e, layoutsearch-saved registers, or rect:r0-r1,c0-c1")
    p.add_argument("--arrangement", default=None, help="RxC chip grid, row-major (default 1 x number of chips)")
    p.add_argument("--links", default="4", help="inter-chip couplers per adjacent chip pair: K or all (default 4)")
    p.add_argument("--link-rows", dest="link_rows", default=None,
                   help="horizontal neighbours: lattice rows that get a link, e.g. 2,5,8 (overrides --links)")
    p.add_argument("--link-cols", dest="link_cols", default=None, help="vertical neighbours: lattice columns")
    p.add_argument("--broken", choices=["disable", "use"], default="disable", help="rect chips: broken qubits")
    p.add_argument("--backend", choices=["ace", "exact"], default="ace",
                   help="chip simulators: each chip's ACE preset (default), or exact (small chips)")
    p.add_argument("--chip-ace", dest="chip_ace", default=None, help="rect chips with --backend ace: SIZE:LRC:LRR[:t]")
    p.add_argument("--chip-max-width", dest="chip_max_width", type=int, default=None)
    p.add_argument("--chip-max-wide", dest="chip_max_wide", type=int, default=None)
    p.add_argument("--cpu", action="store_true", help="chip workers on the CPU (OpenCL hidden)")
    p.add_argument("--ace-gpu", dest="ace_gpu", action="store_true", help="chip workers run ACE on their OpenCL device")
    p.add_argument("--ace-host-pointer", dest="ace_host_pointer", action="store_true")
    p.add_argument("--gpus", default=None, help="OpenCL devices for chip workers, e.g. 0-5")
    p.add_argument("--chips-per-gpu", dest="chips_per_gpu", type=int, default=1,
                   help="chip workers per device (default 1: one Nighthawk chip per 8 GB die)")
    p.add_argument("--gpu-isolation", dest="gpu_isolation", choices=["auto", "off"], default="auto")
    p.add_argument("--theta", choices=["haar", "nnqab"], default="haar")
    p.add_argument("--loom", choices=["ghost", "mean", "sample"], default="ghost",
                   help="how the loom weaves a link CZ: ghost (default): each chip holds a replica of its link "
                        "partners, the link CZ is exact on both chips and carries entanglement, the partner's "
                        "other CZs reach the ghost as mean-field phases; mean: mean-field phases on both sides, "
                        "no entanglement across; sample: sampled Z kicks instead of the phases")
    p.add_argument("--twirls", type=int, default=nq.GATE_TWIRLS, help="Pauli-frame randomisations (ace)")
    p.add_argument("--ghost-max-width", dest="ghost_max_width", type=int, default=None,
                   help="--loom ghost: patch width cap with the ghosts in (default: the chip register's own caps, "
                        "e.g. 28 with one patch at 28 for nighthawk88e); links whose ghost would break it are skipped")
    p.add_argument("--ghost-max-wide", dest="ghost_max_wide", type=int, default=None)
    p.add_argument("--gtt", "-gtt", dest="gtt", type=float, default=None, metavar="GB",
                   help="VRAM oversubscription: each chip worker may use up to GB of host memory that its GPU maps "
                        "over PCIe (GTT). Qrack keeps a patch state in VRAM when 3x its size fits the die, else in "
                        "host memory; QRACK_MAX_ALLOC_MB is raised to VRAM + GB. Ghost room is then judged by this "
                        "split instead of the register's width caps, so wider patches and more ghost links fit")
    p.add_argument("--vram-gb", dest="vram_gb", type=float, default=8.0, help="VRAM per chip worker's die (default 8)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "chipworker":
        return cmd_chipworker(None)
    if len(sys.argv) > 1 and sys.argv[1] == "layoutsearch":
        sys.argv = [str(HERE / "nighthawk_qrack.py")] + sys.argv[1:]
        return nq.main()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="mirror circuits on the cloverfield")
    add_field_args(r)
    r.add_argument("--depths", type=int, nargs="+", default=None, help="default 4 6 8 10 12")
    r.add_argument("--instances", type=int, default=None, help="default 3, as released")
    r.add_argument("--parallel", type=int, default=None,
                   help="looms at once (default: devices x chips-per-gpu / chips, or 1 on CPU)")
    r.add_argument("--order", choices=["deep", "shallow"], default="shallow")
    r.add_argument("--out", default="cloverfield.jsonl")
    r.add_argument("--status", action="store_true")
    r.add_argument("--summarize", action="store_true")
    r.add_argument("--retry-failed", dest="retry_failed", action="store_true")
    p = sub.add_parser("plan", help="the field: chips, couplers, links, memory, devices")
    add_field_args(p)
    p.add_argument("--die-gb", dest="die_gb", type=float, default=8.0)
    p.add_argument("--png", default=None, help="draw the field to this PNG")
    ls = sub.add_parser("linksearch", help="survival against the number of links")
    add_field_args(ls)
    ls.add_argument("--link-counts", dest="link_counts", nargs="+", default=["0", "2", "4", "all"])
    ls.add_argument("--depths", type=int, nargs="+", default=[4, 8])
    ls.add_argument("--instances", type=int, default=1)
    sub.add_parser("layoutsearch", help="nighthawk_qrack's chip-level register + placement search (same options)")
    t = sub.add_parser("selftest", help="field, IO equivalence, woven CZ, loom error")
    t.add_argument("--repo", default=None)
    a = ap.parse_args()
    if a.cmd in ("run", "linksearch", "selftest", "plan"):
        loom_env()
    dict(run=cmd_run, plan=cmd_plan, linksearch=cmd_linksearch, selftest=cmd_selftest)[a.cmd](a)


if __name__ == "__main__":
    main()
