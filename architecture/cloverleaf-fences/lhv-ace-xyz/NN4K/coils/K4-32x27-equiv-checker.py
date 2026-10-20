# -*- coding: us-ascii -*-
# K4-coil-check.py -- can a tiling carry three disjoint coils?
#
# THE QUESTION
# =====================================================================
# A coil is a non-contractible cycle that threads the lattice through
# TREE sites only, never touching a loop site. Three of them, one per
# torus direction, pairwise vertex-disjoint, give a 3x3 twist-response
# tensor instead of a scalar -- and for a genuine Fermi surface that
# tensor is anisotropic, with principal axes that report the Fermi
# surface orientation.
#
# Why tree sites only: a coil that ran through a loop site would perturb
# that block's flux operator when twisted. Threading past the registers
# rather than through them is the whole point. In a --bury tiling every
# loop site has all three bonds inside its block, so every seam bond
# already lands on a tree site; the remaining question is whether the
# WITHIN-block hops can also avoid the loops, and whether three such
# cycles can be found that share no vertex.
#
# HOW WINDING IS DECIDED
# =====================================================================
# Every bond of this lattice displaces the cell coordinate by 0 or by
# +1 along exactly one axis (checked at load). So a closed walk's total
# displacement is 6*w with w the winding vector, and non-contractible
# means w != 0. No homology machinery needed, and the check is exact
# rather than a heuristic about "going all the way round".
#
# THE MODEL
# =====================================================================
# One binary circulation per direction on the tree-site subgraph:
# flow conservation at every node, at most one outgoing arc per node
# (so the support is a disjoint union of simple cycles), total
# displacement pinned to 6 along its own axis and 0 along the others,
# and at most one direction using any given node. Minimise total
# length, because a shorter coil perturbs fewer trees.
#
# A circulation with the right total winding could in principle be
# several cycles rather than one. That is not forbidden in the model --
# forbidding it needs subtour elimination and costs more than it is
# worth -- so the components are extracted afterwards and reported. A
# result with more than one component per direction is still a valid
# answer to a slightly different question, and it says so.
#
#   ./K4-coil-check.py bury-0.json
#   ./K4-coil-check.py bury-*.json equiv-bury.json
#   ./K4-coil-check.py --seconds 300 bury-0.json

import argparse
import collections
import glob
import importlib.util
import json
import os
import sys

from ortools.sat.python import cp_model

L = 6
AXES = ("x", "y", "z")


def find_engine(explicit):
    cands = [explicit] if explicit else []
    if not explicit:
        for pat in ("*Kitaev-single*.py", "*Kitaev*single*.py",
                    "*rystalstacks*.py"):
            for d in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
                for p in sorted(glob.glob(os.path.join(d, pat))):
                    if os.path.abspath(p) != os.path.abspath(__file__):
                        cands.append(p)
    for p in cands:
        if not os.path.exists(p):
            continue
        try:
            spec = importlib.util.spec_from_file_location("k4e", p)
            mod = importlib.util.module_from_spec(spec)
            argv, sys.argv = sys.argv, ["k4e"]
            try:
                spec.loader.exec_module(mod)
            finally:
                sys.argv = argv
        except Exception:
            continue
        if hasattr(mod, "srs_bonds"):
            print("found engine: %s" % os.path.basename(p))
            return mod
    raise SystemExit("could not find the K4 engine; pass --engine")


def lattice(k4):
    bonds, idx = k4.srs_bonds(L, 1)
    n = len(idx)
    adj = collections.defaultdict(set)
    disp = {}
    cube = L ** 3

    def cell(i):
        return ((i % cube) // (L * L), (i % (L * L)) // L, i % L)

    for i, j, _c in bonds:
        adj[i].add(j)
        adj[j].add(i)
        d = tuple(((b - a + L // 2) % L) - L // 2
                  for a, b in zip(cell(i), cell(j)))
        nz = [k for k in range(3) if d[k]]
        if len(nz) > 1 or (nz and abs(d[nz[0]]) != 1):
            raise SystemExit("bond %d-%d displaces by %s; the winding "
                             "argument assumes single-axis unit steps" %
                             (i, j, d))
        disp[(i, j)] = d
        disp[(j, i)] = tuple(-x for x in d)
    return {v: sorted(adj[v]) for v in range(n)}, disp, n


def loop_sites(blocks, adj):
    """The cycle of each block, by iterated leaf-stripping."""
    out = set()
    for b in blocks:
        S = set(b)
        sub = {v: [w for w in adj[v] if w in S] for v in S}
        core = set(S)
        ch = True
        while ch:
            ch = False
            for v in list(core):
                if sum(1 for w in sub[v] if w in core) < 2:
                    core.discard(v)
                    ch = True
        out |= core
    return out


def solve_coils(adj, disp, tree, seconds, workers, ndir=3):
    arcs = []
    for u in sorted(tree):
        for v in adj[u]:
            if v in tree:
                arcs.append((u, v))
    m = cp_model.CpModel()
    x = [{a: m.NewBoolVar("x%d_%d_%d" % (k, a[0], a[1])) for a in arcs}
         for k in range(ndir)]
    out_of = collections.defaultdict(list)
    in_to = collections.defaultdict(list)
    for a in arcs:
        out_of[a[0]].append(a)
        in_to[a[1]].append(a)
    use = [{} for _ in range(ndir)]
    for k in range(ndir):
        for v in sorted(tree):
            o = sum(x[k][a] for a in out_of[v])
            i = sum(x[k][a] for a in in_to[v])
            m.Add(o == i)
            u = m.NewBoolVar("u%d_%d" % (k, v))
            m.Add(o == u)                      # 0 or 1 outgoing: simple
            use[k][v] = u
        for ax in range(3):
            m.Add(sum(disp[a][ax] * x[k][a] for a in arcs)
                  == (L if ax == k else 0))
    for v in sorted(tree):
        m.Add(sum(use[k][v] for k in range(ndir)) <= 1)
    m.Minimize(sum(x[k][a] for k in range(ndir) for a in arcs))

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = seconds
    s.parameters.num_workers = workers
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return s.StatusName(st), None
    res = []
    for k in range(ndir):
        used = [a for a in arcs if s.Value(x[k][a])]
        nxt = {a[0]: a[1] for a in used}
        comps = []
        seen = set()
        for st0 in nxt:
            if st0 in seen:
                continue
            cyc, v = [], st0
            while v not in seen:
                seen.add(v)
                cyc.append(v)
                v = nxt[v]
            comps.append(cyc)
        res.append(comps)
    return s.StatusName(st), res


def report(path, adj, disp, n, seconds, workers):
    d = json.load(open(path))
    blocks = [list(b) for b in d["blocks"]]
    core = loop_sites(blocks, adj)
    tree = set(range(n)) - core
    blk_of = {v: i for i, b in enumerate(blocks) for v in b}
    seam = [(u, v) for u in range(n) for v in adj[u]
            if u < v and blk_of[u] != blk_of[v]]
    on_seam = sum(1 for u, v in seam if u in core or v in core)
    print("\n=== %s ===" % os.path.basename(path))
    print("  loop sites %d, tree sites %d, seam bonds %d (%d touching a loop)"
          % (len(core), len(tree), len(seam), on_seam))

    name, res = solve_coils(adj, disp, tree, seconds, workers)
    if res is None:
        print("  three disjoint coils: %s -- nothing concluded" % name
              if name == "UNKNOWN" else
              "  three disjoint coils: NONE EXIST (%s)" % name)
        return
    total = 0
    for k, comps in enumerate(res):
        ln = sum(len(c) for c in comps)
        total += ln
        blks = len({blk_of[v] for c in comps for v in c})
        note = "" if len(comps) == 1 else \
            "  [%d components, not a single winding]" % len(comps)
        print("  coil %s: %3d sites across %2d of 32 blocks%s"
              % (AXES[k], ln, blks, note))
    per = collections.Counter()
    for comps in res:
        for c in comps:
            for v in c:
                per[blk_of[v]] += 1
    print("  total %d tree sites used of %d; per-block max %d of 17"
          % (total, len(tree), max(per.values()) if per else 0))
    print("  status %s -- three disjoint coils EXIST, none touching a loop"
          % name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--engine")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    k4 = find_engine(a.engine)
    adj, disp, n = lattice(k4)
    print("lattice: %d sites; every bond is a unit step on one axis" % n)
    for f in a.files:
        report(f, adj, disp, n, a.seconds, a.workers)


if __name__ == "__main__":
    main()