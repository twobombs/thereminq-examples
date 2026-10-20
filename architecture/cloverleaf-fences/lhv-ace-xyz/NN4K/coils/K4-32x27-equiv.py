# -*- coding: us-ascii -*-
# k4-equiv-tile.py -- the 32x27 racetrack tiling of the L=6 hyperoctagon,
# searched under translation equivariance instead of head-on.
#
# WHY THIS IS A DIFFERENT SCRIPT AND NOT A FLAG ON tilecp.py
# =====================================================================
# tilecp.py assigns 864 sites to 32 interchangeable labels and spends
# its presolve discovering that the labels are interchangeable. This
# asks a smaller question whose answer implies an answer to that one.
#
# Fix a subgroup H of the translation group acting freely on the sites.
# Ask only for tilings whose BLOCK SET is H-invariant: the 32 blocks are
# the H-images of 32/|H| representatives. Then
#
#   - 864/|H| site-orbits, and the representative blocks contain
#     exactly one site from each. So the whole thing lives on the
#     quotient graph, which for |H|=4 is 216 nodes, not 864.
#   - 32/|H| labels, not 32. For |H|=4 that is 8.
#   - the 1296 candidate 10-loops become 324 loop-orbits.
#
# The lift is a gain (voltage) graph. Two lattice sites u,v project to
# orbits r,r' with a gain gamma = t_u ^ t_v in H, and they land in the
# SAME block iff their orbits carry the same class AND their gauge
# values differ by exactly gamma. Blocks are H-images of each other, so
# verifying one verifies all 32.
#
# WHAT IS AND IS NOT PROVED BY A RUN OF THIS
# =====================================================================
# INFEASIBLE here means no H-EQUIVARIANT tiling exists. It does not
# mean no tiling exists. That remains tilecp.py --lb 32's job and this
# does not replace it.
#
# FEASIBLE here is a genuine 32x27 tiling of the full lattice, and a
# better one than an asymmetric solution would be: 32/|H| distinct
# block types rather than 32, so 32/|H| kernels and calibrations.
#
# WHICH SUBGROUP, AND WHY NOT THE OBVIOUS ONE
# =====================================================================
# The full Z_2^3 of half-period translations -- <(3,0,0),(0,3,0),
# (0,0,3)>, order 8 -- is ruled out, and cheaply. Its 1296 loops fall
# into 162 orbits of exactly 8; an equivariant tiling needs 4 of those
# pairwise site-disjoint after closure; the maximum is 3, proven
# OPTIMAL. Four block types is not available. Do not spend time on it.
#
# Order 4 is the maximum that survives, and barely: every Z_2^2
# subgroup admits at most 9 disjoint loop-orbit closures against the 8
# required. One spare. That tightness is the point -- loop selection is
# nearly forced here, where in the 32-label model it is wide open.
#
# THE BURIED-LOOP OPTION  (--bury)
# =====================================================================
# In a cubic graph a loop site spends two bonds on the loop and has one
# left. --bury forces that third bond inside the block, so no loop site
# touches the seam and the flux operator is one layer in from the
# boundary. Measured on this lattice: all 1296 loops are chordless with
# 10 distinct off-loop feeders, so every loop CAN be buried and the
# option filters nothing at selection time. It is a real restriction on
# the tiling, though: it fixes 20 of a block's 27 sites the moment the
# loop is chosen, leaving 7 free. Off by default because it makes the
# question strictly harder; on if you want the flux registers seam-safe.
#
# VERIFICATION
# =====================================================================
# A solution is lifted back to all 864 sites and checked by code that
# shares nothing with the model: 32 blocks, pairwise disjoint, covering,
# 27 sites each, exactly 27 induced edges, connected, exactly one cycle,
# that cycle of length 10 and a genuine elementary loop of the lattice.
# The equivariance is checked too -- the block set must be closed under
# every element of H -- because a model that quietly broke its own
# symmetry would still pass a plain tiling check.
#
# FOUR MODES, IN THE ORDER YOU SHOULD RUN THEM
# =====================================================================
# The first three cost seconds and any of them can end the search, so
# none of them is optional before a long run.
#
#   --prune     Closure structure and packing bounds. Are there even
#               enough pairwise-disjoint loops to build 32 blocks from?
#               A maximum below the block count settles the question
#               outright, and there is no point discovering that after
#               twelve hours of CP-SAT.
#
#   --scan      Every half-period translation subgroup, with the
#               largest equivariant loop family each can carry. This is
#               what rules out Z_2^3 in about a second, and what says
#               order 4 is the ceiling.
#
#   --selftest  The quotient checked against the lattice it came from.
#               Runs automatically before every solve; this flag just
#               stops afterwards. See VERIFICATION below for why it is
#               not optional.
#
#   (default)   Build and solve the equivariant model.
#
#   ./K4-32x27-equiv.py --check          # confirm the embedded answer
#   ./K4-32x27-equiv.py --prune
#   ./K4-32x27-equiv.py --scan
#   ./K4-32x27-equiv.py --seconds 600
#   ./K4-32x27-equiv.py --bury --seconds 600
#   ./K4-32x27-equiv.py --gens 3,0,0:0,0,3 --workers 32
#
# MEASURED ON L=6
# =====================================================================
#   all 1296 loops chordless, 10 distinct feeders, closure exactly 20
#   32 x 20 = 640 of 864, leaving 224 = 32 x 7
#   max disjoint 10-loops   >= 70 (ceiling 86)     need 32
#   max disjoint closures   >= 36 (ceiling 43)     need 32
#   |H|=8   max 3 orbits, OPTIMAL                  need 4   IMPOSSIBLE
#   |H|=4   max 9 orbits, OPTIMAL, all 7 subgroups need 8   ok
#   |H|=2   max 18 orbits                          need 16  ok
#
# So selection is nowhere near tight in the general problem -- the
# difficulty is entirely in the tree completion -- but it is nearly
# forced under order 4, one orbit of slack. That is the whole reason
# this file is worth running.
#
# Requires ortools and numpy. The lattice comes from the K4 engine.

import argparse
import collections
import glob
import importlib.util
import itertools
import json
import os
import sys
import time

import numpy as np
from ortools.sat.python import cp_model

L_DEFAULT = 6
ELL = 10
B_DEFAULT = 27


# =====================================================================
# PART 1 -- LATTICE
# =====================================================================

def find_engine(explicit):
    cands = [explicit] if explicit else []
    if not explicit:
        here = os.path.dirname(os.path.abspath(__file__))
        me = os.path.abspath(__file__)
        for pat in ("*Kitaev-single*.py", "*Kitaev*single*.py",
                    "*[CK]rystalstacks*Kitaev*.py", "*rystalstacks*.py"):
            for d in (here, os.getcwd()):
                for p in sorted(glob.glob(os.path.join(d, pat))):
                    if os.path.abspath(p) != me and p not in cands:
                        cands.append(p)
    for p in cands:
        if not os.path.exists(p):
            continue
        try:
            spec = importlib.util.spec_from_file_location("k4engine", p)
            m = importlib.util.module_from_spec(spec)
            argv, sys.argv = sys.argv, ["k4engine"]
            try:
                spec.loader.exec_module(m)
            finally:
                sys.argv = argv
        except Exception:
            continue
        if hasattr(m, "srs_bonds") and hasattr(m, "elementary_loops"):
            if not explicit:
                print("found engine: %s" % os.path.basename(p))
            return m
    raise SystemExit(
        "could not find the K4 lattice engine.\n"
        "  it must define srs_bonds and elementary_loops\n"
        "  examined: %s\n"
        "  pass it explicitly with --engine <file>"
        % (", ".join(os.path.basename(c) for c in cands) or "nothing"))


def build_lattice(k4, L):
    bonds, idx = k4.srs_bonds(L, 1)
    n = len(idx)
    a = collections.defaultdict(set)
    for i, j, c in bonds:
        a[i].add(j)
        a[j].add(i)
    adj = {v: sorted(a[v]) for v in range(n)}
    bad = [v for v in adj if len(adj[v]) != 3]
    if bad:
        raise SystemExit("lattice is not cubic at %d sites" % len(bad))
    uniq, _, _, _ = k4.elementary_loops(L, ELL)
    loops = [tuple(p) for p in uniq.values()]
    return adj, loops, n


# =====================================================================
# PART 2 -- THE GROUP, THE ORBITS, THE GAIN GRAPH
# =====================================================================
# Site index in this engine is v*L^3 + c0*L^2 + c1*L + c2, with v the
# K4 sublattice index and c the cell. Translations act on c only, which
# is what makes the quotient a graph rather than a mess.

def make_coords(L):
    cube = L ** 3

    def unsite(i):
        return (i // cube, ((i % cube) // (L * L), (i % (L * L)) // L, i % L))

    def site(v, c):
        return v * cube + c[0] * L * L + c[1] * L + c[2]

    return unsite, site


def close_group(gens, L):
    S = {(0, 0, 0)}
    while True:
        new = {tuple((a[k] + g[k]) % L for k in range(3))
               for a in S for g in gens}
        if new <= S:
            return sorted(S)
        S |= new


def group_bits(H, gens, L):
    """Index every element of H by its coordinates over the generators,
    so the group operation is XOR on those bits. Only valid when H is
    elementary abelian, which is checked."""
    for h in H:
        d = tuple((2 * x) % L for x in h)
        if d != (0, 0, 0):
            raise SystemExit("subgroup is not elementary abelian (2-torsion "
                             "required); element %s has order > 2" % (h,))
    k = len(gens)
    if len(H) != 2 ** k:
        gens = [g for g in H if g != (0, 0, 0)]
        indep = []
        span = {(0, 0, 0)}
        for g in gens:
            if g not in span:
                indep.append(g)
                span = {tuple((a[i] + b[i]) % L for i in range(3))
                        for a in span for b in ((0, 0, 0), g)} | span
        gens = indep
        k = len(gens)
    tbl = {}
    for bits in range(2 ** k):
        e = (0, 0, 0)
        for j in range(k):
            if bits >> j & 1:
                e = tuple((e[i] + gens[j][i]) % L for i in range(3))
        tbl[e] = bits
    if len(tbl) != len(H):
        raise SystemExit("could not index the subgroup by its generators")
    return tbl, k


def quotient(adj, n, H, htbl, L):
    """Orbits, per-site (orbit, phase), and the quotient gain graph."""
    unsite, site = make_coords(L)
    orb_of = [-1] * n
    phase_of = [0] * n
    orbits = []
    for i in range(n):
        if orb_of[i] >= 0:
            continue
        v, c = unsite(i)
        oid = len(orbits)
        mem = []
        for h in H:
            j = site(v, tuple((c[k] + h[k]) % L for k in range(3)))
            if orb_of[j] >= 0 and orb_of[j] != oid:
                raise SystemExit("H does not act freely on the sites")
            orb_of[j] = oid
            phase_of[j] = htbl[h]
            mem.append(j)
        if len(set(mem)) != len(H):
            raise SystemExit("H does not act freely on the sites")
        orbits.append(sorted(mem))

    # Quotient edges carry a gain. Parallel edges between the same orbit
    # pair are kept separately: they have different gains and at most
    # one of them can ever be internal.
    qe = {}
    for u in range(n):
        for v in adj[u]:
            if u > v:
                continue
            r, rp = orb_of[u], orb_of[v]
            g = phase_of[u] ^ phase_of[v]
            key = (r, rp, g) if r <= rp else (rp, r, g)
            qe.setdefault(key, 0)
            qe[key] += 1
    edges = sorted(qe)
    return orbits, orb_of, phase_of, edges


def loop_orbits(loops, orb_of, phase_of, H):
    """One representative per H-orbit of loops, as (orbits, relative
    phases). Loops in one H-orbit differ only by a global phase, so the
    relative pattern is the invariant."""
    seen = {}
    out = []
    for lp in loops:
        rs = [orb_of[v] for v in lp]
        if len(set(rs)) != len(lp):
            continue                   # would need two sites of one orbit
        ph = [phase_of[v] for v in lp]
        pairs = sorted(zip(rs, ph))
        base = pairs[0][1]
        key = tuple((r, p ^ base) for r, p in pairs)
        if key in seen:
            continue
        seen[key] = 1
        out.append(([r for r, _ in pairs],
                    [p ^ base for _, p in pairs],
                    lp))
    return out


def feeders(adj, lp):
    """The one off-loop neighbour of each loop site, or None if the loop
    is chorded or two sites share a feeder."""
    S = set(lp)
    out = []
    for v in lp:
        third = [w for w in adj[v] if w not in S]
        if len(third) != 1:
            return None
        out.append(third[0])
    return out if len(set(out)) == len(lp) else None


# =====================================================================
# PART 3 -- THE MODEL
# =====================================================================

def build_model(adj, orbits, orb_of, phase_of, edges, lorbs, K, B, kbits,
                bury, no_redundant):
    m = cp_model.CpModel()
    R = len(orbits)
    E = len(edges)

    # class one-hot: which of the K block types this orbit belongs to
    yc = [[m.NewBoolVar("yc%d_%d" % (r, i)) for i in range(K)]
          for r in range(R)]
    for r in range(R):
        m.AddExactlyOne(yc[r])
    for i in range(K):
        m.Add(sum(yc[r][i] for r in range(R)) == B)

    # gauge: kbits booleans per orbit, the H-value of its representative
    gb = [[m.NewBoolVar("g%d_%d" % (r, k)) for k in range(kbits)]
          for r in range(R)]
    # Gauge is only ever meaningful up to a global shift per block, so
    # pin orbit 0 and let the solver keep the K-1 it actually needs.
    for k in range(kbits):
        m.Add(gb[0][k] == 0)

    # per-edge predicates
    same_g = []            # gauge difference equals the edge's gain
    same_c = []            # both endpoints carry the same class
    internal = []
    pair = []              # pair[e][i]: both endpoints in class i
    for e, (r, rp, g) in enumerate(edges):
        sg = m.NewBoolVar("sg%d" % e)
        lits = []
        for k in range(kbits):
            want = (g >> k) & 1
            b = m.NewBoolVar("eq%d_%d" % (e, k))
            # b <=> (gb[r][k] XOR gb[rp][k]) == want
            if want:
                m.AddBoolOr([gb[r][k], gb[rp][k], b.Not()])
                m.AddBoolOr([gb[r][k].Not(), gb[rp][k].Not(), b.Not()])
                m.AddBoolOr([gb[r][k], gb[rp][k].Not(), b])
                m.AddBoolOr([gb[r][k].Not(), gb[rp][k], b])
            else:
                m.AddBoolOr([gb[r][k], gb[rp][k].Not(), b.Not()])
                m.AddBoolOr([gb[r][k].Not(), gb[rp][k], b.Not()])
                m.AddBoolOr([gb[r][k], gb[rp][k], b])
                m.AddBoolOr([gb[r][k].Not(), gb[rp][k].Not(), b])
            lits.append(b)
        m.AddMinEquality(sg, lits) if len(lits) > 1 else m.Add(sg == lits[0])
        same_g.append(sg)

        pr = []
        for i in range(K):
            p = m.NewBoolVar("p%d_%d" % (e, i))
            m.AddMultiplicationEquality(p, [yc[r][i], yc[rp][i]])
            pr.append(p)
        pair.append(pr)
        sc = m.NewBoolVar("sc%d" % e)
        m.Add(sum(pr) == 1).OnlyEnforceIf(sc)
        m.Add(sum(pr) == 0).OnlyEnforceIf(sc.Not())
        same_c.append(sc)

        it = m.NewBoolVar("in%d" % e)
        m.AddMultiplicationEquality(it, [sg, sc])
        internal.append(it)

    # a block has 27 sites and, being unicyclic, exactly 27 internal edges
    for i in range(K):
        ei = []
        for e in range(E):
            x = m.NewBoolVar("ei%d_%d" % (e, i))
            m.AddMultiplicationEquality(x, [internal[e], pair[e][i]])
            ei.append(x)
        m.Add(sum(ei) == B)

    # ---- loop selection ------------------------------------------------
    NL = len(lorbs)
    y = [m.NewBoolVar("y%d" % l) for l in range(NL)]
    m.Add(sum(y) == K)
    inloop = [m.NewBoolVar("il%d" % r) for r in range(R)]
    byorb = collections.defaultdict(list)
    for l, (rs, ph, _) in enumerate(lorbs):
        for r in rs:
            byorb[r].append(l)
    for r in range(R):
        m.Add(sum(y[l] for l in byorb[r]) == inloop[r])
    for i in range(K):
        # exactly one selected loop per class, anchored on its first orbit
        z = []
        for l, (rs, ph, _) in enumerate(lorbs):
            t = m.NewBoolVar("z%d_%d" % (l, i))
            m.AddMultiplicationEquality(t, [y[l], yc[rs[0]][i]])
            z.append(t)
        m.AddExactlyOne(z)
    for l, (rs, ph, _) in enumerate(lorbs):
        r0, p0 = rs[0], ph[0]
        for r, p in zip(rs[1:], ph[1:]):
            for i in range(K):
                m.Add(yc[r][i] == yc[r0][i]).OnlyEnforceIf(y[l])
            for k in range(kbits):
                if ((p ^ p0) >> k) & 1:
                    m.Add(gb[r][k] + gb[r0][k] == 1).OnlyEnforceIf(y[l])
                else:
                    m.Add(gb[r][k] == gb[r0][k]).OnlyEnforceIf(y[l])

    # ---- trees: every non-loop orbit hangs off a parent one step nearer
    inc = collections.defaultdict(list)
    for e, (r, rp, g) in enumerate(edges):
        if r != rp:
            inc[r].append((e, rp))
            inc[rp].append((e, r))
    depth = [m.NewIntVar(0, B - ELL, "d%d" % r) for r in range(R)]
    par = {}
    for r in range(R):
        ps = []
        for e, rp in inc[r]:
            p = m.NewBoolVar("par%d_%d" % (r, e))
            par[(r, e)] = p
            m.AddImplication(p, internal[e])
            m.Add(depth[r] == depth[rp] + 1).OnlyEnforceIf(p)
            ps.append(p)
        m.Add(sum(ps) == 1).OnlyEnforceIf(inloop[r].Not())
        m.Add(sum(ps) == 0).OnlyEnforceIf(inloop[r])
        m.Add(depth[r] == 0).OnlyEnforceIf(inloop[r])
        m.Add(depth[r] >= 1).OnlyEnforceIf(inloop[r].Not())

    # ---- redundant but propagating -------------------------------------
    if not no_redundant:
        for i in range(K):
            li = []
            for r in range(R):
                t = m.NewBoolVar("li%d_%d" % (r, i))
                m.AddMultiplicationEquality(t, [inloop[r], yc[r][i]])
                li.append(t)
            m.Add(sum(li) == ELL)

    return m, dict(yc=yc, gb=gb, y=y, internal=internal, inloop=inloop,
                   depth=depth, edges=edges)


def add_bury(m, V, adj, orbits, orb_of, phase_of, edges, lorbs, kbits):
    """Force each loop site's third bond inside its own block."""
    epos = {}
    for e, (r, rp, g) in enumerate(edges):
        epos[(r, rp, g)] = e
        epos[(rp, r, g)] = e
    y, yc, gb = V["y"], V["yc"], V["gb"]
    K = len(yc[0])
    dropped = 0
    for l, (rs, ph, lp) in enumerate(lorbs):
        fd = feeders(adj, lp)
        if fd is None:
            m.Add(y[l] == 0)
            dropped += 1
            continue
        ok = True
        for v, f in zip(lp, fd):
            r, rf = orb_of[v], orb_of[f]
            g = phase_of[v] ^ phase_of[f]
            if r == rf or (r, rf, g) not in epos:
                ok = False
                break
            e = epos[(r, rf, g)]
            m.AddImplication(y[l], V["internal"][e])
        if not ok:
            m.Add(y[l] == 0)
            dropped += 1
    return dropped


# =====================================================================
# PART 4 -- LIFT AND VERIFY  (shares no code with the model)
# =====================================================================

def lift(sol_class, sol_gauge, orbits, orb_of, phase_of, K, H, htbl):
    """Rebuild all 32 blocks in the full lattice from the quotient
    solution. Block (i,h) = sites whose orbit has class i and whose
    phase differs from the orbit's gauge by h."""
    blocks = collections.defaultdict(list)
    for oid, mem in enumerate(orbits):
        i = sol_class[oid]
        g = sol_gauge[oid]
        for v in mem:
            blocks[(i, phase_of[v] ^ g)].append(v)
    return [sorted(b) for _, b in sorted(blocks.items())]


def verify(blocks, adj, n, loops, K_total, B, ell=ELL):
    loopset = {frozenset(p) for p in loops}
    seen = set()
    if len(blocks) != K_total:
        return False, "expected %d blocks, got %d" % (K_total, len(blocks))
    for bi, blk in enumerate(blocks):
        S = set(blk)
        if len(S) != B:
            return False, "block %d has %d sites" % (bi, len(S))
        if S & seen:
            return False, "block %d overlaps an earlier block" % bi
        seen |= S
        eds = [(u, v) for u in S for v in adj[u] if u < v and v in S]
        if len(eds) != B:
            return False, "block %d has %d induced edges, want %d" \
                % (bi, len(eds), B)
        sub = collections.defaultdict(list)
        for u, v in eds:
            sub[u].append(v)
            sub[v].append(u)
        st, comp = [blk[0]], {blk[0]}
        while st:
            u = st.pop()
            for w in sub[u]:
                if w not in comp:
                    comp.add(w)
                    st.append(w)
        if len(comp) != B:
            return False, "block %d is not connected (%d of %d)" \
                % (bi, len(comp), B)
        core = {v for v in S if len(sub[v]) >= 2}
        changed = True
        while changed:
            changed = False
            for v in list(core):
                if sum(1 for w in sub[v] if w in core) < 2:
                    core.discard(v)
                    changed = True
        if len(core) != ell:
            return False, "block %d cycle has %d sites, want %d" \
                % (bi, len(core), ell)
        if frozenset(core) not in loopset:
            return False, "block %d cycle is not an elementary loop" % bi
    if len(seen) != n:
        return False, "blocks cover %d of %d sites" % (len(seen), n)
    return True, "ok"


def verify_equivariance(blocks, H, orb_helpers, L):
    unsite, site = orb_helpers
    bs = {frozenset(b) for b in blocks}
    for h in H:
        for b in bs:
            img = frozenset(
                site(unsite(v)[0],
                     tuple((c + d) % L for c, d in zip(unsite(v)[1], h)))
                for v in b)
            if img not in bs:
                return False, "block set is not closed under %s" % (h,)
    return True, "ok"


# =====================================================================

def selftest(adj, loops, n, orbits, orb_of, phase_of, edges, lorbs, H, htbl, L):
    """Does the quotient faithfully represent the lattice?

    Everything downstream rests on one claim: that (orbit, phase) is an
    exact recoding of a site, and that the gain on a quotient edge is
    exactly the phase difference of its lifts. If that is wrong the model
    will still solve something, just not this problem, and verify() would
    be checking a lift built by the same wrong rule. So this reconstructs
    the lattice FROM the quotient and compares against the real one.
    """
    unsite, site = make_coords(L)
    bad = []

    # 1. (orbit, phase) is a bijection onto sites
    seen = {}
    for v in range(n):
        k = (orb_of[v], phase_of[v])
        if k in seen:
            bad.append("sites %d and %d share (orbit,phase) %s"
                       % (seen[k], v, k))
        seen[k] = v
    if len(seen) != n:
        bad.append("(orbit,phase) covers %d of %d sites" % (len(seen), n))

    # 2. every lattice edge is a quotient edge with the right gain, and
    #    every quotient edge lifts to exactly |H| lattice edges
    eset = {}
    for e, (r, rp, g) in enumerate(edges):
        eset[(r, rp, g)] = eset.get((r, rp, g), 0) + 1
    lift_count = collections.Counter()
    for u in range(n):
        for v in adj[u]:
            if u > v:
                continue
            r, rp = orb_of[u], orb_of[v]
            g = phase_of[u] ^ phase_of[v]
            key = (r, rp, g) if r <= rp else (rp, r, g)
            if key not in eset:
                bad.append("lattice edge (%d,%d) has no quotient edge" % (u, v))
            lift_count[key] += 1
    for key, c in lift_count.items():
        if c != len(H):
            bad.append("quotient edge %s lifts to %d edges, want %d"
                       % (key, c, len(H)))
    if len(lift_count) != len(edges):
        bad.append("%d quotient edges but %d were hit by lifting"
                   % (len(edges), len(lift_count)))

    # 3. loop-orbits lift back to genuine elementary loops, all |H| of them
    loopset = {frozenset(p) for p in loops}
    if len(lorbs) * len(H) != len(loops):
        bad.append("%d loop-orbits x %d != %d loops"
                   % (len(lorbs), len(H), len(loops)))
    rep = {}
    for v in range(n):
        rep.setdefault(orb_of[v], {})[phase_of[v]] = v
    for l, (rs, ph, lp) in enumerate(lorbs):
        got = set()
        for gauge in range(len(H)):
            S = frozenset(rep[r][p ^ gauge] for r, p in zip(rs, ph))
            if S not in loopset:
                bad.append("loop-orbit %d at gauge %d is not a loop" % (l, gauge))
            got.add(S)
        if len(got) != len(H):
            bad.append("loop-orbit %d lifts to %d distinct loops, want %d"
                       % (l, len(got), len(H)))

    # 4. the block-count arithmetic the model asserts
    K = (n // B_DEFAULT) // len(H)
    if len(orbits) != K * B_DEFAULT:
        bad.append("%d orbits but %d types x %d sites" % (len(orbits), K,
                                                          B_DEFAULT))
    if len(edges) != K * B_DEFAULT + (len(orbits) * 3 - 2 * K * B_DEFAULT) // 2:
        bad.append("quotient edge count %d is not internal+seam" % len(edges))

    print("selftest: %d checks failed" % len(bad))
    for b in bad[:10]:
        print("  " + b)
    if not bad:
        print("  (orbit,phase) bijective; every quotient edge lifts to %d;"
              % len(H))
        print("  all %d loop-orbits lift to %d genuine elementary loops each;"
              % (len(lorbs), len(H)))
        print("  %d orbits = %d types x %d, %d quotient edges."
              % (len(orbits), K, B_DEFAULT, len(edges)))
    return not bad


# =====================================================================
# PART 5 -- THE SOLUTIONS, IN THE FILE
# =====================================================================
# Both tilings this script found are embedded here, so it is
# self-contained and a fresh checkout can confirm the result without
# re-running anything.
#
#   'bury'   every loop site has all three bonds inside its own block.
#            No flux operator touches a seam anywhere in the lattice;
#            all 432 seam bonds emanate from tree sites. This is the
#            one to use.
#   'plain'  the same tiling question without that requirement. Kept
#            for comparison: its buried counts run 1 to 7 per block,
#            so its flux operators DO sit on seams. Strictly worse
#            architecturally, and it was no easier to find -- burying
#            pins 20 of 27 sites the moment a loop is chosen, and that
#            propagates harder than it costs.
#
# Both are equivariant under <(3,0,0),(0,3,0)>, so there are 8 distinct
# block types rather than 32.
#
# Site indices only mean anything relative to how the engine numbers
# sites, so --check VERIFIES them against the freshly built lattice and
# reports a reason rather than trusting them. An embedded constant that
# silently disagreed with the lattice would be worse than none.

SOLUTIONS = {
    # (L, B, variant): the 32 blocks, as site indices into the lattice
    # as THIS engine numbers them. Verified before use; see below.
    (6, 27, 'bury'): (
    (0,5,41,180,185,186,191,216,221,246,371,396,401,407,432,437,473,617,622,623,648,649,654,684,689,834,839),
    (72,77,78,83,108,113,149,263,288,293,299,324,329,354,509,514,515,540,545,581,726,731,756,757,762,792,797),
    (18,23,59,198,203,204,209,228,234,239,389,414,419,425,450,455,491,635,640,641,666,667,672,702,707,852,857),
    (90,95,96,101,126,131,167,281,306,311,317,336,342,347,527,532,533,558,563,599,744,749,774,775,780,810,815),
    (1,2,156,181,182,187,188,218,248,367,397,398,403,433,434,588,593,618,619,650,651,656,804,805,830,835,836),
    (48,73,74,79,80,109,110,259,289,290,295,326,356,480,485,510,511,541,542,696,697,722,727,728,758,759,764),
    (19,20,174,199,200,205,206,230,236,385,415,416,421,451,452,606,611,636,637,668,669,674,822,823,848,853,854),
    (66,91,92,97,98,127,128,277,307,308,313,338,344,498,503,528,529,559,560,714,715,740,745,746,776,777,782),
    (75,76,81,82,111,112,179,261,291,292,297,328,358,395,512,513,543,544,580,610,724,729,730,760,761,766,827),
    (3,4,71,183,184,189,190,220,250,287,369,399,400,405,435,436,472,502,620,621,652,653,658,719,832,837,838),
    (93,94,99,100,129,130,161,279,309,310,315,340,346,377,530,531,561,562,592,598,742,747,748,778,779,784,809),
    (21,22,53,201,202,207,208,232,238,269,387,417,418,423,453,454,484,490,638,639,670,671,676,701,850,855,856),
    (25,26,31,32,61,62,210,241,242,247,272,278,426,427,457,462,463,493,494,499,524,679,680,710,711,741,828),
    (102,133,134,139,140,169,170,318,319,349,350,355,380,386,565,570,571,601,602,607,632,720,787,788,818,819,849),
    (7,8,13,14,43,44,192,223,224,229,254,260,408,409,439,444,445,475,476,481,506,661,662,692,693,723,846),
    (84,115,116,121,122,151,152,300,301,331,332,337,362,368,547,552,553,583,584,589,614,738,769,770,800,801,831),
    (135,136,141,142,171,172,321,351,352,357,382,383,388,567,568,572,573,603,604,634,783,789,790,820,821,826,851),
    (27,28,33,34,63,64,243,244,249,274,275,280,429,459,460,464,465,495,496,526,675,681,682,712,713,718,743),
    (117,118,123,124,153,154,303,333,334,339,364,365,370,549,550,554,555,585,586,616,765,771,772,802,803,808,833),
    (9,10,15,16,45,46,225,226,231,256,257,262,411,441,442,446,447,477,478,508,657,663,664,694,695,700,725),
    (114,119,120,125,150,155,294,305,330,335,341,360,366,546,551,556,557,582,587,612,763,767,768,773,798,799,829),
    (6,11,12,17,42,47,222,227,233,252,258,402,413,438,443,448,449,474,479,504,655,659,660,665,690,691,721),
    (132,137,138,143,168,173,312,323,348,353,359,378,384,564,569,574,575,600,605,630,781,785,786,791,816,817,847),
    (24,29,30,35,60,65,240,245,251,270,276,420,431,456,461,466,467,492,497,522,673,677,678,683,708,709,739),
    (144,145,146,175,176,211,212,325,361,390,391,392,422,428,458,576,577,613,642,643,644,793,794,824,859,860,861),
    (36,37,38,67,68,103,104,217,253,282,283,284,314,320,468,469,505,534,535,536,566,685,686,716,751,752,753),
    (157,158,162,163,164,193,194,343,372,373,374,379,404,410,440,594,595,624,625,626,631,806,811,812,841,842,843),
    (49,50,54,55,56,85,86,235,264,265,266,271,296,302,486,487,516,517,518,523,548,698,703,704,733,734,735),
    (147,148,177,178,213,214,215,327,363,393,394,424,430,578,579,608,609,615,645,646,647,795,796,825,858,862,863),
    (39,40,69,70,105,106,107,219,255,285,286,316,322,470,471,500,501,507,537,538,539,687,688,717,750,754,755),
    (159,160,165,166,195,196,197,345,375,376,381,406,412,590,591,596,597,627,628,629,633,807,813,814,840,844,845),
    (51,52,57,58,87,88,89,237,267,268,273,298,304,482,483,488,489,519,520,521,525,699,705,706,732,736,737),
    ),
    (6, 27, 'plain'): (
    (0,5,30,180,186,190,191,197,246,251,370,396,401,407,413,432,437,618,622,623,629,648,649,653,834,839,845),
    (72,78,82,83,89,108,113,138,262,288,293,299,305,354,359,510,514,515,521,540,545,726,731,737,756,757,761),
    (12,18,23,198,204,208,209,215,228,233,388,414,419,425,431,450,455,636,640,641,647,666,667,671,852,857,863),
    (90,96,100,101,107,120,126,131,280,306,311,317,323,336,341,528,532,533,539,558,563,744,749,755,774,775,779),
    (87,92,93,97,128,148,149,297,303,307,308,328,338,365,524,529,549,559,560,580,735,741,745,746,766,776,797),
    (20,40,41,195,200,201,205,220,230,257,405,411,415,416,441,451,452,472,632,637,658,668,689,843,849,853,854),
    (74,75,79,105,110,166,167,289,290,315,321,346,356,383,506,511,541,542,567,598,723,727,728,753,758,784,815),
    (2,58,59,182,183,187,213,238,248,275,397,398,423,429,433,434,459,490,614,619,650,676,707,831,835,836,861),
    (84,85,94,95,99,130,270,295,300,301,309,310,516,522,526,527,531,561,562,732,733,738,739,743,747,748,778),
    (22,192,193,202,203,207,378,403,408,409,417,418,453,454,624,630,634,635,639,670,840,841,846,847,851,855,856),
    (76,77,81,102,103,112,252,291,292,313,318,319,504,508,509,513,534,543,544,720,721,725,729,730,750,751,760),
    (4,184,185,189,210,211,360,399,400,421,426,427,435,436,612,616,617,621,642,652,828,829,833,837,838,858,859),
    (73,134,139,140,169,170,171,320,349,350,355,381,385,505,570,571,572,601,602,633,722,787,788,817,818,819,850),
    (26,31,32,61,62,63,181,241,242,247,273,277,428,462,463,464,493,494,525,613,679,680,709,710,711,742,830),
    (91,116,121,122,151,152,153,302,331,332,337,363,367,523,552,553,554,583,584,615,740,769,770,799,800,801,832),
    (8,13,14,43,44,45,199,223,224,229,255,259,410,444,445,446,475,476,507,631,661,662,691,692,693,724,848),
    (123,129,154,155,156,161,162,333,334,339,364,366,371,372,377,555,585,586,593,771,772,777,802,803,804,809,810),
    (15,21,46,47,48,53,54,225,226,231,256,258,263,264,269,447,477,478,485,663,664,669,694,695,696,701,702),
    (111,141,144,172,173,174,179,351,352,357,382,384,389,390,395,573,603,604,611,759,789,790,792,820,821,822,827),
    (3,33,36,64,65,66,71,243,244,249,274,276,281,282,287,465,495,496,503,651,681,682,684,712,713,714,719),
    (34,35,60,67,68,69,70,240,245,250,279,466,467,492,497,498,499,500,501,502,678,683,708,715,716,717,718),
    (142,143,168,175,176,177,178,348,353,358,387,574,575,600,605,606,607,608,609,610,786,791,816,823,824,825,826),
    (16,17,42,49,50,51,52,222,227,232,261,448,449,474,479,480,481,482,483,484,660,665,690,697,698,699,700),
    (124,125,150,157,158,159,160,330,335,340,369,556,557,582,587,588,589,590,591,592,768,773,798,805,806,807,808),
    (1,7,37,38,98,104,216,217,218,253,254,278,283,284,314,439,468,469,470,530,535,536,566,656,685,686,752),
    (109,115,145,146,206,212,324,325,326,361,362,386,391,392,422,458,547,576,577,578,638,643,644,764,793,794,860),
    (19,25,55,56,80,86,234,235,236,260,265,266,271,272,296,457,486,487,488,512,517,518,548,674,703,704,734),
    (127,133,163,164,188,194,342,343,344,368,373,374,379,380,404,440,565,594,595,596,620,625,626,782,811,812,842),
    (6,10,11,135,165,196,221,345,375,376,402,406,412,438,442,443,473,597,627,628,654,655,659,783,813,814,844),
    (27,57,88,114,118,119,237,267,268,294,298,304,329,489,519,520,546,550,551,581,675,705,706,736,762,763,767),
    (24,28,29,117,147,214,239,327,393,394,420,424,430,456,460,461,491,579,645,646,672,673,677,765,795,796,862),
    (9,39,106,132,136,137,219,285,286,312,316,322,347,471,537,538,564,568,569,599,657,687,688,754,780,781,785),
    ),
}


# =====================================================================
# PART 6 -- CHEAP PRUNING, BEFORE ANY LONG RUN
# =====================================================================
# Three questions that cost seconds and can end the search outright.
# Run them before committing hours to anything.
#
#   --prune  closure structure and packing bounds
#   --scan   which translation subgroups can carry an equivariant tiling
#
# Measured on L=6, and this is why --scan defaults the way it does:
# the full Z_2^3 admits at most 3 disjoint loop-orbit closures against
# the 4 required, proven OPTIMAL. Every Z_2^2 admits 9 against 8.

def packing_masks(n, sets):
    M = np.zeros((len(sets), n), dtype=bool)
    for i, s in enumerate(sets):
        M[i, list(s)] = True
    return M


def max_packing(M, label, target, tl, workers):
    """Largest number of pairwise site-disjoint rows of M."""
    m = cp_model.CpModel()
    x = [m.NewBoolVar("x%d" % i) for i in range(M.shape[0])]
    for v in range(M.shape[1]):
        own = np.nonzero(M[:, v])[0]
        if len(own) > 1:
            m.AddAtMostOne([x[i] for i in own])
    m.Maximize(sum(x))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = tl
    s.parameters.num_workers = workers
    st = s.Solve(m)
    ok = st in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    best = int(s.ObjectiveValue()) if ok else -1
    print("  %-30s %-9s best %3d  bound %5.1f  need %3d  %s"
          % (label, s.StatusName(st), best, s.BestObjectiveBound(), target,
             "ok" if best >= target else
             "IMPOSSIBLE" if st == cp_model.OPTIMAL else "?"))
    return st, best


def cmd_prune(adj, loops, n, K_total, B, tl, workers):
    print("\n=== closure structure ===")
    # A loop site spends two bonds on its loop and has one left. Burying
    # the loop means that third bond stays inside the block, so the
    # closure -- loop plus feeders -- is what a block must contain.
    chord = dup = 0
    cl = []
    for lp in loops:
        f = feeders(adj, lp)
        if f is None:
            S = set(lp)
            if any(len([w for w in adj[v] if w not in S]) != 1 for v in lp):
                chord += 1
            else:
                dup += 1
            cl.append(None)
        else:
            cl.append(frozenset(set(lp) | set(f)))
    good = [c for c in cl if c is not None]
    szs = collections.Counter(len(c) for c in good)
    print("  chorded, or a feeder landing on the loop : %d" % chord)
    print("  two loop sites sharing one feeder        : %d" % dup)
    print("  loops with a clean closure               : %d of %d"
          % (len(good), len(loops)))
    print("  closure sizes                            : %s" % dict(szs))
    if good:
        cs = len(next(iter(good)))
        print("  arithmetic: %d x %d = %d of %d sites, %d left = %d x %d"
              % (K_total, cs, K_total * cs, n, n - K_total * cs, K_total,
                 (n - K_total * cs) // K_total))

    print("\n=== packing bounds (a maximum below the block count ends it) ===")
    print("  counting ceilings: loops %d, closures %d"
          % (n // ELL, n // (len(next(iter(good))) if good else ELL)))
    max_packing(packing_masks(n, [set(p) for p in loops]),
                "disjoint %d-loops" % ELL, K_total, tl, workers)
    if good:
        max_packing(packing_masks(n, good), "disjoint closures", K_total,
                    tl, workers)


def cmd_scan(adj, loops, n, K_total, B, L, tl, workers):
    """Which translation subgroups could carry an equivariant tiling?

    Necessary condition only, and cheap: the K_total/|H| representative
    loops must have pairwise disjoint closures, and each orbit's own
    |H| closures must be disjoint from each other. If that fails there
    is no equivariant tiling and the model need never be built.
    """
    unsite, site = make_coords(L)
    half = L // 2
    if half * 2 != L:
        print("scan needs an even L for half-period translations")
        return
    cands = [t for t in itertools.product((0, half), repeat=3)
             if t != (0, 0, 0)]
    cl = {}
    for i, lp in enumerate(loops):
        f = feeders(adj, lp)
        cl[i] = frozenset(set(lp) | set(f)) if f else frozenset(lp)
    key = {frozenset(lp): i for i, lp in enumerate(loops)}

    def orbits_for(H, htbl):
        orb, out = {}, []
        for i, lp in enumerate(loops):
            if i in orb:
                continue
            o = set()
            for h in H:
                img = frozenset(
                    site(unsite(v)[0],
                         tuple((c + d) % L for c, d in zip(unsite(v)[1], h)))
                    for v in lp)
                o.add(key[img])
            for j in o:
                orb[j] = len(out)
            out.append(sorted(o))
        return out

    tried = set()
    print("\n=== translation subgroups ===")
    print("  %-30s %-9s %s" % ("subgroup", "status", "verdict"))
    for k in (3, 2, 1):
        for gens in itertools.combinations(cands, k):
            H = close_group(gens, L)
            if len(H) != 2 ** k or frozenset(H) in tried:
                continue
            tried.add(frozenset(H))
            if K_total % len(H):
                continue
            need = K_total // len(H)
            try:
                htbl, _ = group_bits(H, list(gens), L)
            except SystemExit:
                continue
            orbs = orbits_for(H, htbl)
            masks = []
            for o in orbs:
                if len(o) != len(H):
                    continue
                acc = np.zeros(n, dtype=bool)
                good = True
                for i in o:
                    mm = np.zeros(n, dtype=bool)
                    mm[list(cl[i])] = True
                    if (acc & mm).any():
                        good = False
                        break
                    acc |= mm
                if good:
                    masks.append(acc)
            if not masks:
                print("  |H|=%d %-22s %-9s no self-disjoint orbit"
                      % (len(H), str(gens[0]) if k == 1 else "%d gens" % k,
                         "-"))
                continue
            max_packing(np.array(masks),
                        "|H|=%d  %s" % (len(H), ",".join(
                            "".join(str(x) for x in g) for g in gens)),
                        need, tl, workers)


# =====================================================================
# PART 7 -- EXHAUSTIVE ENUMERATION OF LOOP FAMILIES
# =====================================================================
# Under order-4 equivariance the loop selection is nearly forced: 9
# pairwise-disjoint orbit closures exist and 8 are needed. Enumerating
# the whole selection space takes about 20 seconds on one core and
# yields 324 families for L=6. That is small enough to settle by
# exhaustion.
#
# So instead of one model that returns UNKNOWN, this runs 324 small
# ones. Each fixes the 8 loop-orbits -- pinning 80 of 216 orbits and,
# with --bury, another 80 -- and solves only the tree completion.
#
# The exhaustion argument is only as good as its weakest subproblem: if
# any family times out at UNKNOWN, nothing is proved, and the run says
# so rather than reporting a proof it does not have. Raise
# --family-seconds and rerun those.
#
# Shard across processes with --shard i --shards n; families are
# independent and share nothing.

def loop_footprints(adj, lorbs, orbits, orb_of, phase_of, H, n):
    """Per loop-orbit, the union of the closures of all |H| of its
    lifts, or None if those closures overlap each other -- an orbit
    that collides with its own image can never be used."""
    rep = {}
    for v in range(n):
        rep.setdefault(orb_of[v], {})[phase_of[v]] = v
    out = []
    for l, (rs, ph, lp) in enumerate(lorbs):
        acc = set()
        ok = True
        for gauge in range(len(H)):
            S = [rep[r][p ^ gauge] for r, p in zip(rs, ph)]
            f = feeders(adj, S)
            if f is None:
                ok = False
                break
            m = set(S) | set(f)
            if acc & m:
                ok = False
                break
            acc |= m
        out.append(acc if ok else None)
    return out


def enumerate_families(foot, K, cap=None):
    """Every set of K loop-orbits whose footprints are pairwise
    disjoint. Bitmask DFS -- the conflict graph is dense, which is
    what keeps the count small."""
    use = [(l, f) for l, f in enumerate(foot) if f is not None]
    bits = []
    for l, f in use:
        b = 0
        for v in f:
            b |= 1 << v
        bits.append(b)
    N = len(use)
    conf = []
    for a in range(N):
        c = 0
        for b in range(N):
            if a != b and (bits[a] & bits[b]):
                c |= 1 << b
        conf.append(c)
    fams = []

    def dfs(start, chosen, banned):
        if len(chosen) == K:
            fams.append([use[i][0] for i in chosen])
            return
        if cap and len(fams) >= cap:
            return
        for a in range(start, N):
            if banned >> a & 1:
                continue
            if N - a < K - len(chosen):
                break
            dfs(a + 1, chosen + [a], banned | conf[a])

    dfs(0, [], 0)
    return N, fams


def cmd_enumerate(a, adj, loops, n, K_total, orbits, orb_of, phase_of,
                  edges, lorbs, K, kbits, H, htbl):
    t0 = time.time()
    foot = loop_footprints(adj, lorbs, orbits, orb_of, phase_of, H, n)
    N, fams = enumerate_families(foot, K, a.max_families or None)
    print("\n=== enumeration ===")
    print("  usable loop-orbits      : %d of %d" % (N, len(lorbs)))
    print("  valid %d-orbit families  : %d   (%.1fs)"
          % (K, len(fams), time.time() - t0))
    if not fams:
        print("\nNO EQUIVARIANT TILING: no family of %d disjoint loop-orbits "
              "exists." % K)
        return
    mine = [(i, f) for i, f in enumerate(fams) if i % a.shards == a.shard]
    if a.shards > 1:
        print("  shard %d of %d           : %d families"
              % (a.shard, a.shards, len(mine)))

    solved = unknown = 0
    for pos, (fi, fam) in enumerate(mine):
        m, V = build_model(adj, orbits, orb_of, phase_of, edges, lorbs, K,
                           a.B, kbits, a.bury, a.no_redundant)
        if a.bury:
            add_bury(m, V, adj, orbits, orb_of, phase_of, edges, lorbs, kbits)
        chosen = set(fam)
        for l in range(len(lorbs)):
            m.Add(V["y"][l] == (1 if l in chosen else 0))
        # Classes are interchangeable, so pinning family member k to
        # class k is free symmetry breaking, not an assumption.
        for k, l in enumerate(sorted(fam)):
            m.Add(V["yc"][lorbs[l][0][0]][k] == 1)
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = a.family_seconds
        s.parameters.num_workers = a.workers
        st = s.Solve(m)
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            cls = [next(i for i in range(K) if s.Value(V["yc"][r][i]))
                   for r in range(len(orbits))]
            gau = [sum(s.Value(V["gb"][r][k]) << k for k in range(kbits))
                   for r in range(len(orbits))]
            blocks = lift(cls, gau, orbits, orb_of, phase_of, K, H, htbl)
            ok, why = verify(blocks, adj, n, loops, K_total, a.B)
            eq, ewhy = verify_equivariance(blocks, H, make_coords(a.L), a.L)
            print("\nfamily %d: SOLVED   verifier %s (%s)   equivariance %s (%s)"
                  % (fi, "PASS" if ok else "FAIL", why,
                     "PASS" if eq else "FAIL", ewhy))
            if ok and eq:
                print("\nTILING EXISTS: %d blocks of %d, %d distinct types"
                      % (K_total, a.B, K))
                json.dump({"L": a.L, "B": a.B, "n_blocks": len(blocks),
                           "n_types": K, "family": fi, "bury": a.bury,
                           "verified": True, "blocks": blocks},
                          open(a.out, "w"))
                print("written to %s" % a.out)
                return
            print("  verification failed; this is a bug, not a result.")
            return
        if st == cp_model.INFEASIBLE:
            solved += 1
        else:
            unknown += 1
            print("  family %d: UNKNOWN at %.0fs -- exhaustion is not "
                  "available unless this closes" % (fi, a.family_seconds))
        if (pos + 1) % 20 == 0 or pos + 1 == len(mine):
            print("  %d/%d families: %d refuted, %d timed out  (%.0fs)"
                  % (pos + 1, len(mine), solved, unknown, time.time() - t0))

    print("")
    if unknown:
        print("INCONCLUSIVE: %d of %d families refuted, %d timed out."
              % (solved, len(mine), unknown))
        print("Nothing is proved while any family is UNKNOWN. Raise")
        print("--family-seconds and rerun.")
    elif a.shards > 1:
        print("SHARD %d COMPLETE: all %d of its families refuted."
              % (a.shard, len(mine)))
        print("Exhaustion needs every shard to report this.")
    else:
        print("NO EQUIVARIANT TILING under this subgroup%s: all %d loop"
              % (" with --bury" if a.bury else "", len(fams)))
        print("families enumerated and every one refuted.")
        print("This does NOT refute the general %dx%d tiling."
              % (K_total, a.B))



def cmd_check(adj, loops, n, K_total, B, L, want=None):
    """Verify the embedded solutions against this lattice."""
    keys = [k for k in SOLUTIONS if k[0] == L and k[1] == B
            and (want is None or k[2] == want)]
    if not keys:
        print("no embedded solution for L=%d B=%d %s" % (L, B, want or ""))
        return
    for k in sorted(keys):
        blocks = [list(b) for b in SOLUTIONS[k]]
        print("\n=== embedded solution %r ===" % (k,))
        ok, why = verify(blocks, adj, n, loops, K_total, B)
        print("  tiling        : %s (%s)" % ("PASS" if ok else "FAIL", why))
        if not ok:
            print("  the constant does not match this lattice; not used.")
            continue
        H = close_group([(3, 0, 0), (0, 3, 0)], L)
        eq, ewhy = verify_equivariance(blocks, H, make_coords(L), L)
        print("  equivariance  : %s (%s)" % ("PASS" if eq else "FAIL", ewhy))
        # Buried loop sites, re-derived here rather than trusted: a loop
        # site is buried when all three of its bonds stay in its block.
        cnt = collections.Counter()
        seam = 0
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
            cnt[sum(1 for v in core if len(sub[v]) == 3)] += 1
            seam += sum(3 - len(sub[v]) for v in S)
        print("  buried loop sites per block: %s" % dict(cnt))
        print("  seam bonds: %d (%d per block)" % (seam // 2, seam // len(blocks)))
        if set(cnt) == {ELL}:
            print("  every flux operator is interior; no loop site on a seam.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine")
    ap.add_argument("-L", type=int, default=L_DEFAULT)
    ap.add_argument("-B", type=int, default=B_DEFAULT)
    ap.add_argument("--gens", default="3,0,0:0,3,0",
                    help="translation generators, colon separated")
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--bury", action="store_true")
    ap.add_argument("--no-redundant", action="store_true")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="check the quotient against the lattice, then stop")
    ap.add_argument("--prune", action="store_true",
                    help="closure structure and packing bounds, then stop")
    ap.add_argument("--scan", action="store_true",
                    help="which subgroups admit an equivariant tiling, "
                         "then stop")
    ap.add_argument("--check", nargs="?", const="all",
                    choices=["all", "bury", "plain"],
                    help="verify the embedded solutions, then stop")
    ap.add_argument("--enumerate", action="store_true", dest="enumerate_",
                    help="enumerate every loop family and solve each")
    ap.add_argument("--family-seconds", type=float, default=120.0,
                    help="time limit per family in --enumerate")
    ap.add_argument("--max-families", type=int, default=0,
                    help="stop enumerating after this many (0 = all)")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--prune-seconds", type=float, default=60.0,
                    help="time limit per packing bound in --prune/--scan")
    ap.add_argument("--out", default="equiv.json")
    a = ap.parse_args()

    print("ortools %s" % getattr(cp_model, "__version__", "?"))
    k4 = find_engine(a.engine)
    adj, loops, n = build_lattice(k4, a.L)
    K_total = n // a.B
    print("lattice: %d sites, %d bonds, %d %d-loops, %d blocks of %d"
          % (n, sum(len(x) for x in adj.values()) // 2, len(loops), ELL,
             K_total, a.B))

    if a.check:
        cmd_check(adj, loops, n, K_total, a.B, a.L,
                  None if a.check == "all" else a.check)
        return
    if a.prune:
        cmd_prune(adj, loops, n, K_total, a.B, a.prune_seconds, a.workers)
        return
    if a.scan:
        cmd_scan(adj, loops, n, K_total, a.B, a.L, a.prune_seconds, a.workers)
        return

    gens = [tuple(int(x) for x in g.split(",")) for g in a.gens.split(":")]
    H = close_group(gens, a.L)
    if K_total % len(H):
        raise SystemExit("|H|=%d does not divide the block count %d"
                         % (len(H), K_total))
    htbl, kbits = group_bits(H, gens, a.L)
    K = K_total // len(H)
    orbits, orb_of, phase_of, edges = quotient(adj, n, H, htbl, a.L)
    lorbs = loop_orbits(loops, orb_of, phase_of, H)
    print("H order %d -> %d site-orbits, %d quotient edges, %d loop-orbits"
          % (len(H), len(orbits), len(edges), len(lorbs)))
    print("searching for %d representative blocks (%d after lifting)"
          % (K, K_total))

    if not selftest(adj, loops, n, orbits, orb_of, phase_of, edges, lorbs,
                    H, htbl, a.L):
        raise SystemExit("the quotient does not faithfully represent the "
                         "lattice; not solving.")
    if a.selftest:
        return
    if a.enumerate_:
        cmd_enumerate(a, adj, loops, n, K_total, orbits, orb_of, phase_of,
                      edges, lorbs, K, kbits, H, htbl)
        return

    t0 = time.time()
    m, V = build_model(adj, orbits, orb_of, phase_of, edges, lorbs, K, a.B,
                       kbits, a.bury, a.no_redundant)
    if a.bury:
        d = add_bury(m, V, adj, orbits, orb_of, phase_of, edges, lorbs, kbits)
        print("--bury: %d of %d loop-orbits cannot be buried and are excluded"
              % (d, len(lorbs)))
    print("model built in %.1fs" % (time.time() - t0))

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = a.seconds
    s.parameters.num_workers = a.workers
    s.parameters.log_search_progress = a.log
    st = s.Solve(m)
    name = s.StatusName(st)
    print("\nstatus: %s   (%.1fs, %d workers)"
          % (name, s.WallTime(), a.workers))

    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        cls = [next(i for i in range(K) if s.Value(V["yc"][r][i]))
               for r in range(len(orbits))]
        gau = [sum(s.Value(V["gb"][r][k]) << k for k in range(kbits))
               for r in range(len(orbits))]
        blocks = lift(cls, gau, orbits, orb_of, phase_of, K, H, htbl)
        ok, why = verify(blocks, adj, n, loops, K_total, a.B)
        print("independent verifier: %s (%s)" % ("PASS" if ok else "FAIL", why))
        eq, ewhy = verify_equivariance(blocks, H, make_coords(a.L), a.L)
        print("equivariance check  : %s (%s)" % ("PASS" if eq else "FAIL", ewhy))
        if ok and eq:
            print("\nTILING EXISTS: %d blocks of %d, %d distinct block types"
                  % (K_total, a.B, K))
            json.dump({"L": a.L, "B": a.B, "gens": gens, "n_blocks": len(blocks),
                       "n_types": K, "bury": a.bury, "verified": True,
                       "blocks": blocks}, open(a.out, "w"))
            print("written to %s" % a.out)
    elif st == cp_model.INFEASIBLE:
        print("\nNO EQUIVARIANT TILING under H = %s (order %d)%s."
              % (gens, len(H), " with --bury" if a.bury else ""))
        print("This does NOT refute the general 32x27 tiling. Only")
        print("tilecp.py --lb 32 returning INFEASIBLE does that.")
    else:
        print("\nUNKNOWN: nothing is concluded. Raise --seconds.")


if __name__ == "__main__":
    main()
