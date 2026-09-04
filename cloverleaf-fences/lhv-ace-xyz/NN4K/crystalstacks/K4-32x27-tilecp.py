# -*- coding: us-ascii -*-
# tilecp.py -- Rev 2.
#
# Can the L=6 hyperoctagon lattice be partitioned into 32 racetrack
# blocks of 27 sites, each a 10-loop with a 17-site tree hanging off it?
#
# The counting says maybe. 32 x 27 = 864 = every site, and
# 32 x 27 internal + (32 x 27)/2 shared = 864 + 432 = 1296 = every
# bond. Both close exactly. That is a necessary condition and not a
# construction, which is what this script is for.
#
# WHAT CHANGED IN REV 2
# =====================================================================
# Rev 1 asked a yes/no question with an all-or-nothing model and, after
# 900 seconds, had returned neither answer. Six changes, in rough order
# of how much they matter.
#
#   1. MAXIMISE, DO NOT DECIDE. Sites may now be left unassigned and
#      the objective is the number of complete blocks. A timeout now
#      yields a lower bound (blocks actually built, and verified) AND
#      an upper bound (CP-SAT's objective bound) instead of nothing.
#      Optimum 32 proves the tiling; a proven optimum below 32
#      disproves it; anything else brackets it. Rev 1 could only
#      report the middle case as silence, which is what it did.
#
#   2. AN INDEPENDENT VERIFIER. verify_blocks() re-derives everything
#      from the lattice: 27 sites, pairwise disjoint, exactly 27
#      induced edges, connected, exactly one cycle, that cycle of
#      length 10 and equal to a genuine elementary loop. It shares no
#      code with the model. A model this fiddly should not be trusted
#      on its own say-so.
#
#   3. A SELF-TEST WITH A KNOWN ANSWER. --selftest runs the identical
#      encoding on the induced subgraph of one known-good block with
#      K=1, where the only possible answer is that block. L=6 is the
#      smallest lattice whose site count 27 divides, so there is no
#      natural small instance; this manufactures one. Run it before
#      trusting any long run.
#
#   4. FULL REIFICATION OF ok[e]. Rev 1 constrained only
#      ok => justified and leaned on a global count for the converse.
#      Correct, but it propagates badly and it is exactly the kind of
#      cleverness that hides an off-by-one. Now an iff, both ways.
#
#   5. A COMPLETE WARM START, AND CHECKPOINTING. --hint seeds the
#      search from a greedy packing (25-28 blocks unaided). The hint
#      sets EVERY variable -- loop choice, parent pointers, depths,
#      edge justification -- not just the block labels: a partial hint
#      left the solver reconstructing the rest and it reported no
#      feasible solution at all inside 200s. A callback verifies and
#      writes every improving solution to disk, so a long run is
#      informative while it runs rather than only at the end.
#
#   6. A LOWER BOUND, AND DEAD WEIGHT REMOVED.
#      --lb pins the objective from below, by default at the greedy
#      hint size, so the search hunts for an improvement rather than
#      rediscovering 28 blocks from nothing. --lb 32 turns the whole
#      thing back into a decision problem, and there INFEASIBLE is a
#      proof that no tiling exists -- probably the mode you want for a
#      long run.
#      Rev 1's negative channelling, blk[v] != k enforced on
#      y[v][k].Not(), was 27648 constraints implied by ExactlyOne(y[v])
#      plus the positive direction; it is gone. Redundant-but-
#      propagating constraints replace it (loop sites per block,
#      internal edges), behind --no-redundant so you can measure
#      whether they earn their keep.
#
# A NOTE ON HARDWARE
# =====================================================================
# Rev 1's inconclusive 900-second run was made on ONE core. This model
# has roughly 10^5 booleans and CP-SAT parallelises well; --workers now
# defaults to the machine's core count. Do not read anything into a
# single-core timeout.
#
# STANDING CAVEATS
# =====================================================================
#   - A block is REQUIRED to contain a full elementary 10-loop. The
#     lattice has girth 10, but a unicyclic 27-site subgraph could
#     carry a 12- or 14-cycle instead. Those are excluded by fiat here
#     because the racetrack is the point. For the weaker question --
#     can it be tiled by unicyclic 27-blocks at all -- drop the loop
#     machinery and constrain induced edges to 27.
#   - Connectivity is not asserted, it is forced. Every non-loop site
#     carries a parent at strictly smaller depth, so its chain
#     terminates on a loop site of its own block. Cheaper and safer
#     than any flow or cut encoding.
#   - Symmetry across the K interchangeable labels is broken by value
#     precedence over sites. That is the standard linear-size
#     encoding, not necessarily the best one here; --no-symmetry turns
#     it off for comparison.
#   - Blocks are required to contain a loop, so a proven optimum below
#     K disproves THIS tiling, not every 27-site tiling.
#
# Requires ortools. The lattice comes from the K4 engine.

import argparse
import collections
import json
import os
import pickle
import sys
import time

from ortools.sat.python import cp_model

ELL = 10
DEFAULT_B = 27
CACHE = "/tmp/k4_lattice_cache.pkl"


# =====================================================================
# PART 1 -- LATTICE
# =====================================================================

def load_lattice(engine, L, cache=CACHE):
    """(adj, loops, n) for the L^3 hyperoctagon torus. Enumerating the
    elementary 10-loops is the expensive step, so it is cached."""
    key = "%s-L%d" % (os.path.basename(engine), L)
    if cache and os.path.exists(cache):
        try:
            blob = pickle.load(open(cache, "rb"))
            if blob.get("key") == key:
                return blob["adj"], blob["loops"], blob["n"]
        except Exception:
            pass
    import importlib.util
    spec = importlib.util.spec_from_file_location("k4engine", engine)
    k4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k4)
    bonds, idx = k4.srs_bonds(L, 1)
    a = collections.defaultdict(list)
    for i, j, c in bonds:
        a[i].append(j)
        a[j].append(i)
    adj = {v: sorted(a[v]) for v in range(len(idx))}
    uniq, _, _, _ = k4.elementary_loops(L, ELL)
    loops = [tuple(p) for p in uniq.values()]
    if cache:
        pickle.dump({"key": key, "adj": adj, "loops": loops,
                     "n": len(idx)}, open(cache, "wb"))
    return adj, loops, len(idx)


def edge_index(adj):
    edges = sorted({(min(u, v), max(u, v)) for u in adj for v in adj[u]})
    return edges, {e: i for i, e in enumerate(edges)}


# =====================================================================
# PART 2 -- INDEPENDENT VERIFIER
# =====================================================================
# Shares nothing with the model. If the encoding is wrong, this is what
# catches it.

def verify_blocks(blocks, adj, loops, B=DEFAULT_B, ell=ELL):
    """(ok, [problems]). Each block must be 27 sites, disjoint from the
    others, with exactly 27 induced edges, connected, and carrying one
    cycle which is a genuine elementary ell-loop."""
    bad, seen = [], {}
    loopset = {frozenset(p) for p in loops}
    for bi, blk in enumerate(blocks):
        S = set(blk)
        if len(S) != B:
            bad.append("block %d: %d sites, want %d" % (bi, len(S), B))
            continue
        for v in S:
            if v in seen:
                bad.append("site %d in blocks %d and %d" % (v, seen[v], bi))
            seen[v] = bi
        ie = [(u, w) for u in S for w in adj[u] if w in S and u < w]
        if len(ie) != B:
            bad.append("block %d: %d induced edges, want %d"
                       % (bi, len(ie), B))
            continue
        nb = collections.defaultdict(list)
        for u, w in ie:
            nb[u].append(w)
            nb[w].append(u)
        start = next(iter(S))
        comp, stack = {start}, [start]
        while stack:
            v = stack.pop()
            for w in nb[v]:
                if w not in comp:
                    comp.add(w)
                    stack.append(w)
        if len(comp) != B:
            bad.append("block %d: disconnected, %d of %d reachable"
                       % (bi, len(comp), B))
            continue
        # 27 vertices, 27 edges, connected -> exactly one cycle.
        # Peel leaves; what survives is that cycle.
        deg = {v: len(nb[v]) for v in S}
        rem = set(S)
        leaves = [v for v in rem if deg[v] == 1]
        while leaves:
            v = leaves.pop()
            if v not in rem:
                continue
            rem.discard(v)
            for w in nb[v]:
                if w in rem:
                    deg[w] -= 1
                    if deg[w] == 1:
                        leaves.append(w)
        if len(rem) != ell:
            bad.append("block %d: cycle length %d, want %d"
                       % (bi, len(rem), ell))
        elif frozenset(rem) not in loopset:
            bad.append("block %d: its %d-cycle is not an elementary loop"
                       % (bi, ell))
    return (not bad), bad


# =====================================================================
# PART 3 -- GREEDY PACKING (hint source, and a baseline)
# =====================================================================

def greedy_blocks(adj, loops, B, rng=None, limit=None):
    import random
    rng = rng or random.Random(0)
    order = list(loops)
    rng.shuffle(order)
    free, out = set(adj), []
    for p in order:
        if limit and len(out) >= limit:
            break
        if not set(p) <= free:
            continue
        cur = set(p)
        while len(cur) < B:
            fr = [w for v in cur for w in adj[v]
                  if w in free and w not in cur
                  and sum(1 for x in adj[w] if x in cur) == 1]
            if not fr:
                break
            cur.add(rng.choice(sorted(fr)))
        if len(cur) == B:
            out.append(sorted(cur))
            free -= cur
    return out


def best_greedy(adj, loops, B, restarts, target):
    import random
    best = []
    for s in range(restarts):
        g = greedy_blocks(adj, loops, B, random.Random(s), limit=target)
        if len(g) > len(best):
            best = g
            if len(best) >= target:
                break
    return best


# =====================================================================
# PART 4 -- THE MODEL
# =====================================================================

def build(adj, loops, K, B, ell=ELL, symmetry=True, redundant=True):
    """Blocks 0..K-1 plus an 'unassigned' class K. Maximises the count
    of complete blocks. Returns (model, vars)."""
    n = len(adj)
    edges, eid = edge_index(adj)
    loops_at = collections.defaultdict(list)
    loops_with = collections.defaultdict(list)
    for i, p in enumerate(loops):
        for v in p:
            loops_at[v].append(i)
        for r in range(ell):
            a, b = p[r], p[(r + 1) % ell]
            loops_with[eid[(min(a, b), max(a, b))]].append(i)

    m = cp_model.CpModel()
    x = [m.NewBoolVar("x%d" % i) for i in range(len(loops))]
    used = [m.NewBoolVar("u%d" % k) for k in range(K)]
    blk = [m.NewIntVar(0, K, "b%d" % v) for v in range(n)]
    y = [[m.NewBoolVar("y%d_%d" % (v, k)) for k in range(K + 1)]
         for v in range(n)]
    un = [y[v][K] for v in range(n)]
    lp = [m.NewBoolVar("lp%d" % v) for v in range(n)]
    d = [m.NewIntVar(0, B - ell, "d%d" % v) for v in range(n)]
    ok = [m.NewBoolVar("ok%d" % e) for e in range(len(edges))]
    par = {v: [m.NewBoolVar("p%d_%d" % (v, j)) for j in range(len(adj[v]))]
           for v in range(n)}
    nb = m.NewIntVar(0, K, "nblocks")

    for v in range(n):
        m.AddExactlyOne(y[v])
        for k in range(K + 1):
            # positive direction only; ExactlyOne supplies the converse
            m.Add(blk[v] == k).OnlyEnforceIf(y[v][k])
        m.AddAtMostOne([x[i] for i in loops_at[v]])
        m.Add(lp[v] == sum(x[i] for i in loops_at[v]))
        m.AddImplication(lp[v], un[v].Not())
        m.Add(d[v] == 0).OnlyEnforceIf(lp[v])
        m.Add(sum(par[v]) == 0).OnlyEnforceIf(lp[v])
        m.Add(sum(par[v]) == 0).OnlyEnforceIf(un[v])
        m.Add(sum(par[v]) == 1).OnlyEnforceIf([lp[v].Not(), un[v].Not()])
        for j, u in enumerate(adj[v]):
            m.Add(blk[v] == blk[u]).OnlyEnforceIf(par[v][j])
            m.Add(d[v] == d[u] + 1).OnlyEnforceIf(par[v][j])

    m.Add(nb == sum(used))
    m.Add(sum(x) == nb)
    for k in range(K):
        m.Add(sum(y[v][k] for v in range(n)) == B * used[k])
    for k in range(K - 1):
        m.Add(used[k] >= used[k + 1])

    for i, p in enumerate(loops):
        for r in range(ell):
            m.Add(blk[p[r]] == blk[p[(r + 1) % ell]]).OnlyEnforceIf(x[i])

    for e, (u, v) in enumerate(edges):
        ju, jv = adj[v].index(u), adj[u].index(v)
        just = [x[i] for i in loops_with[e]] + [par[v][ju], par[u][jv]]
        m.AddBoolOr([ok[e].Not()] + just)              # ok => justified
        for t in just:
            m.AddImplication(t, ok[e])                 # justified => ok
        # a chord may not join two sites of the same assigned block
        m.Add(blk[u] != blk[v]).OnlyEnforceIf([ok[e].Not(), un[u].Not()])

    if redundant:
        m.Add(sum(lp) == ell * nb)
        m.Add(sum(ok) == B * nb)
        ly = [[m.NewBoolVar("ly%d_%d" % (v, k)) for k in range(K)]
              for v in range(n)]
        for v in range(n):
            for k in range(K):
                m.AddImplication(ly[v][k], y[v][k])
                m.AddImplication(ly[v][k], lp[v])
                m.AddBoolOr([y[v][k].Not(), lp[v].Not(), ly[v][k]])
        for k in range(K):
            m.Add(sum(ly[v][k] for v in range(n)) == ell * used[k])

    if symmetry and K > 1:
        cc = [m.NewIntVar(0, K - 1, "c%d" % v) for v in range(n)]
        mx = [m.NewIntVar(0, K - 1, "m%d" % v) for v in range(n)]
        for v in range(n):
            m.Add(cc[v] == blk[v]).OnlyEnforceIf(un[v].Not())
            m.Add(cc[v] == 0).OnlyEnforceIf(un[v])
        m.Add(mx[0] == cc[0])
        m.Add(blk[0] == 0).OnlyEnforceIf(un[0].Not())
        for v in range(1, n):
            m.Add(blk[v] <= mx[v - 1] + 1).OnlyEnforceIf(un[v].Not())
            m.AddMaxEquality(mx[v], [mx[v - 1], cc[v]])

    m.Maximize(nb)
    return m, {"x": x, "blk": blk, "nb": nb, "used": used, "y": y,
               "lp": lp, "d": d, "ok": ok, "par": par, "eid": eid,
               "loops": loops, "adj": adj, "ell": ell,
               "n": n, "K": K, "B": B, "edges": edges}


def block_structure(S, adj, ell=ELL):
    """(cycle, parent, depth) for one block: peel leaves to expose the
    unique cycle, then root a BFS forest on it."""
    S = set(S)
    nb = {v: [w for w in adj[v] if w in S] for v in S}
    deg = {v: len(nb[v]) for v in S}
    rem = set(S)
    leaves = [v for v in rem if deg[v] == 1]
    while leaves:
        v = leaves.pop()
        if v not in rem:
            continue
        rem.discard(v)
        for w in nb[v]:
            if w in rem:
                deg[w] -= 1
                if deg[w] == 1:
                    leaves.append(w)
    parent, depth = {}, {v: 0 for v in rem}
    fr = sorted(rem)
    while fr:
        nx = []
        for v in fr:
            for w in nb[v]:
                if w not in depth:
                    depth[w] = depth[v] + 1
                    parent[w] = v
                    nx.append(w)
        fr = nx
    return rem, parent, depth


def add_hint(m, V, blocks):
    """A COMPLETE warm start. Rev 1 hinted blk alone and left every
    auxiliary variable -- loop choice, parent pointers, depths, edge
    justification -- for the solver to reconstruct, which on one core
    it never managed inside the budget. Hint all of it or none."""
    n, K, ell = V["n"], V["K"], V["ell"]
    loop_id = {frozenset(p): i for i, p in enumerate(V["loops"])}
    blocks = blocks[:K]
    where, isloop, par_of, dep = {}, set(), {}, {}
    chosen = set()
    for k, S in enumerate(blocks):
        cyc, parent, depth = block_structure(S, V["adj"], ell)
        i = loop_id.get(frozenset(cyc))
        if i is None:
            return 0                       # not a racetrack block; skip hint
        chosen.add(i)
        for v in S:
            where[v] = k
        isloop |= set(cyc)
        par_of.update(parent)
        dep.update(depth)
    for i in range(len(V["loops"])):
        m.AddHint(V["x"][i], 1 if i in chosen else 0)
    for k in range(K):
        m.AddHint(V["used"][k], 1 if k < len(blocks) else 0)
    just = set()
    for v in range(n):
        k = where.get(v, K)
        m.AddHint(V["blk"][v], k)
        for kk in range(K + 1):
            m.AddHint(V["y"][v][kk], 1 if kk == k else 0)
        m.AddHint(V["lp"][v], 1 if v in isloop else 0)
        m.AddHint(V["d"][v], dep.get(v, 0))
        p = par_of.get(v)
        for j, u in enumerate(V["adj"][v]):
            hit = 1 if (p is not None and u == p) else 0
            m.AddHint(V["par"][v][j], hit)
            if hit:
                just.add(V["eid"][(min(u, v), max(u, v))])
    for i in chosen:
        p = V["loops"][i]
        for r in range(ell):
            a, b = p[r], p[(r + 1) % ell]
            just.add(V["eid"][(min(a, b), max(a, b))])
    for e in range(len(V["edges"])):
        m.AddHint(V["ok"][e], 1 if e in just else 0)
    m.AddHint(V["nb"], len(blocks))
    return len(blocks)


def extract(solver, V):
    part = collections.defaultdict(list)
    for v in range(V["n"]):
        k = solver.Value(V["blk"][v])
        if k < V["K"]:
            part[k].append(v)
    return [sorted(p) for _, p in sorted(part.items())]


class Progress(cp_model.CpSolverSolutionCallback):
    def __init__(self, V, adj, loops, out):
        super().__init__()
        self.V, self.adj, self.loops, self.out = V, adj, loops, out
        self.best = -1

    def on_solution_callback(self):
        k = int(self.Value(self.V["nb"]))
        if k <= self.best:
            return
        self.best = k
        blocks = extract(self, self.V)
        ok, bad = verify_blocks(blocks, self.adj, self.loops, self.V["B"])
        print("  [%8.1fs] %3d blocks, %4d/%d sites, verifier %s"
              % (self.WallTime(), k, k * self.V["B"], self.V["n"],
                 "pass" if ok else "FAIL: " + "; ".join(bad[:2])))
        sys.stdout.flush()
        if self.out:
            json.dump({"n_blocks": k, "verified": ok, "problems": bad,
                       "wall": self.WallTime(), "blocks": blocks},
                      open(self.out, "w"), indent=1)


# =====================================================================
# PART 5 -- SELF-TEST
# =====================================================================

def selftest(adj, loops, B):
    """Run the identical encoding on the induced subgraph of one known
    block with K=1. The only valid answer is that block."""
    print("=" * 70)
    print("SELF-TEST: one known block, K=1, induced subgraph")
    print("=" * 70)
    g = greedy_blocks(adj, loops, B, limit=1)
    if not g:
        print("  no seed block available -- inconclusive")
        return False
    S = set(g[0])
    ren = {v: i for i, v in enumerate(sorted(S))}
    sadj = {ren[v]: sorted(ren[w] for w in adj[v] if w in S) for v in S}
    sloops = [tuple(ren[v] for v in p) for p in loops if set(p) <= S]
    print("  subgraph: %d sites, %d edges, %d elementary %d-loops"
          % (len(sadj), sum(len(a) for a in sadj.values()) // 2,
             len(sloops), ELL))
    m, V = build(sadj, sloops, 1, B)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = 120.0
    s.parameters.num_search_workers = max(1, os.cpu_count() or 1)
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("  status %s -- ENCODING IS BROKEN, this must be solvable"
              % s.StatusName(st))
        return False
    got = extract(s, V)
    ok, bad = verify_blocks(got, sadj, sloops, B)
    hit = len(got) == 1 and set(got[0]) == set(range(len(sadj)))
    print("  status %s, blocks %d, verifier %s, recovers the block: %s"
          % (s.StatusName(st), len(got), "pass" if ok else "FAIL",
             "yes" if hit else "no"))
    for b in bad[:3]:
        print("    ! %s" % b)
    return ok and hit


# =====================================================================
# PART 6 -- CLI
# =====================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Racetrack-block tiling of the hyperoctagon lattice")
    ap.add_argument("--engine", default="K4-Chrystalstacks-Kitaev-single.py",
                    help="path to the K4 engine, which supplies the lattice")
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = every core on the machine")
    ap.add_argument("--out", default="tiling.json")
    ap.add_argument("--hint", type=int, default=200, metavar="RESTARTS",
                    help="greedy restarts for the warm start; 0 disables")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-symmetry", action="store_true")
    ap.add_argument("--no-redundant", action="store_true")
    ap.add_argument("--lb", type=int, default=-1, metavar="N",
                    help="require at least N blocks. Default: the greedy "
                         "hint size. --lb 32 makes it a pure decision "
                         "problem, where INFEASIBLE disproves the tiling")
    ap.add_argument("--log", action="store_true", help="CP-SAT search log")
    ns = ap.parse_args(argv)
    workers = ns.workers or max(1, os.cpu_count() or 1)

    adj, loops, n = load_lattice(ns.engine, ns.L)
    edges, _ = edge_index(adj)
    print("lattice L=%d: %d sites, %d bonds, %d elementary %d-loops"
          % (ns.L, n, len(edges), len(loops), ELL))
    if n % ns.B:
        print("%d does not divide %d -- no exact tiling is possible"
              % (ns.B, n))
        return 1
    K, B = n // ns.B, ns.B
    print("target: %d blocks x %d sites; %d internal + %d shared bonds"
          % (K, B, K * B, K * B // 2))
    print("")

    if ns.selftest:
        return 0 if selftest(adj, loops, B) else 2

    hint = []
    if ns.hint:
        t = time.time()
        hint = best_greedy(adj, loops, B, ns.hint, K)
        ok, bad = verify_blocks(hint, adj, loops, B)
        print("greedy hint: %d blocks in %.1fs, verifier %s"
              % (len(hint), time.time() - t, "pass" if ok else "FAIL"))
        for b in bad[:3]:
            print("  ! %s" % b)
        if not ok:
            hint = []

    t0 = time.time()
    m, V = build(adj, loops, K, B, ELL,
                 symmetry=not ns.no_symmetry,
                 redundant=not ns.no_redundant)
    lb = ns.lb if ns.lb >= 0 else len(hint)
    if lb > 0:
        m.Add(V["nb"] >= lb)
        print("lower bound: requiring at least %d blocks" % lb)
    hinted = add_hint(m, V, hint) if hint else 0
    if hint and not hinted:
        print("  hint rejected: greedy blocks are not racetrack blocks")
    print("model built in %.1fs (hinted %d blocks); solving up to %.0fs"
          " on %d worker%s"
          % (time.time() - t0, hinted, ns.seconds, workers,
             "" if workers == 1 else "s"))

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = ns.seconds
    s.parameters.num_search_workers = workers
    s.parameters.log_search_progress = ns.log
    s.parameters.repair_hint = True
    cb = Progress(V, adj, loops, ns.out)
    st = s.Solve(m, cb)

    best = cb.best if cb.best >= 0 else 0
    bound = int(s.BestObjectiveBound())
    print("")
    print("status %s after %.1fs" % (s.StatusName(st), s.WallTime()))
    print("blocks: best found %d, upper bound %d, target %d"
          % (best, bound, K))
    if st == cp_model.INFEASIBLE:
        print("INFEASIBLE at lb=%d: at most %d racetrack blocks fit." % (lb, lb - 1))
        if lb >= K:
            print("NO TILING EXISTS.")
    elif best == K:
        print("TILING EXISTS. %s holds the %d blocks." % (ns.out, K))
    elif bound < K:
        print("NO TILING EXISTS: proven upper bound %d < %d." % (bound, K))
    else:
        print("OPEN: bracketed in [%d, %d]. Run longer, or compare with"
              " --no-symmetry / --no-redundant." % (max(best, lb), bound))
    return 0


if __name__ == "__main__":
    sys.exit(main())
