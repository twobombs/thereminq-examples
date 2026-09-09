# -*- coding: us-ascii -*-
# K4-manifest.py -- turn a verified tiling into a circuit manifest.
#
# WHAT THIS IS FOR
# =====================================================================
# The tiling JSON says which sites belong together. A simulator needs
# rather more: local qubit indices, which Pauli sits on each bond, the
# flux operator as an actual Pauli word, and which qubit pairs face
# each other across a seam. This emits that, and checks it round-trips
# against the lattice rather than asking anyone to trust it.
#
# WHY THE KITAEV MODEL SUITS A QUBIT SIMULATOR HERE
# =====================================================================
# It is natively a spin model: one qubit per site, each bond a
# two-qubit Pauli coupling picked by the bond's colour. No Jordan-
# Wigner, so none of the non-local strings a 3D fermionic mapping
# would inflict. Every site carries exactly one bond of each colour
# (checked at load), which is what makes the flux operator well
# defined.
#
# A block is 27 qubits. That is 2^27 amplitudes -- about 1.1 GB at
# complex64, so a block state vector fits a mid-range card with room
# for ancillae. Equivariance means there are 8 distinct block types
# rather than 32, so eight simulations cover the lattice.
#
# THE FLUX OPERATOR
# =====================================================================
# For an even loop, W is the product over loop sites of the Pauli
# matching the colour of that site's bond that is NOT on the loop. In
# a --bury tiling that bond is internal, so W is a 10-qubit Pauli word
# entirely inside one block, measurable with one ancilla and no
# cross-seam anything. All 32 loops here share a colour profile of
# 2/4/4, so the word has the same shape everywhere.
#
# WHAT THIS DOES NOT GIVE YOU
# =====================================================================
# 864 qubits is not state-vector simulable at any block size, and this
# does not change that. It gives 27-qubit patches plus an explicit
# account of what couples them. Note also that the pure Kitaev ground
# state is Gaussian and exact diagonalization on the Majoranas beats
# any qubit simulation of the same question -- the reason to go to
# circuits is the parts that are NOT Gaussian: seam repair with
# ancillae, gauge-protection terms, noise and open-system dynamics.
#
#   ./K4-manifest.py equiv-bury.json -o manifest.json
#   ./K4-manifest.py bury-0.json --no-coils
#   ./K4-manifest.py --validate-only manifest.json
#   ./K4-manifest.py --coils-only bury-*.json
#
# THE COILS
# =====================================================================
# A coil is a non-contractible cycle threading TREE sites only, never a
# loop site, so that twisting it perturbs no flux register. Three of
# them, one per torus direction, pairwise vertex-disjoint, give a 3x3
# twist-response tensor rather than a scalar -- and for a genuine Fermi
# surface that tensor is anisotropic, with principal axes reporting the
# Fermi surface orientation. A gapped state cannot fake that: its
# response is exponentially small in every direction alike.
#
# Winding is exact, not a heuristic. Every bond of this lattice moves
# the cell coordinate by 0 or +1 along exactly one axis (checked at
# load), so a closed walk's total displacement is 6*w.
#
# --coils-only reports the question per tiling without writing
# anything, which is how to choose between several verified tilings:
# they differ in how much tree budget the coils cost and how much of
# the lattice they sample. Measured on the buried tilings, all of them
# carry three disjoint coils of 18 sites -- the geodesic floor, since a
# winding crosses 6 cells at 3 hops each -- using about a tenth of the
# 544 tree sites and leaving every block most of its 17 free.
#
# In a buried tiling not one of the 432 seam bonds touches a loop site.
# In an unburied one about 188 do, and the coils then have to route
# around flux operators sitting on the boundary.

import argparse
import collections
import glob
import importlib.util
import json
import os
import sys

L = 6
PAULI = "XYZ"


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
    colour = {}
    disp = {}
    cube = L ** 3

    def cell(i):
        return ((i % cube) // (L * L), (i % (L * L)) // L, i % L)

    for i, j, c in bonds:
        adj[i].add(j)
        adj[j].add(i)
        colour[(i, j)] = colour[(j, i)] = c
        d = tuple(((b - a + L // 2) % L) - L // 2
                  for a, b in zip(cell(i), cell(j)))
        disp[(i, j)] = d
        disp[(j, i)] = tuple(-x for x in d)
    adj = {v: sorted(adj[v]) for v in range(n)}
    # One bond of each colour per site. Without this the flux operator
    # is not well defined and everything below is meaningless.
    for v in range(n):
        cs = sorted(colour[(v, w)] for w in adj[v])
        if cs != [0, 1, 2]:
            raise SystemExit("site %d has bond colours %s, want one of each"
                             % (v, cs))
    return adj, colour, disp, n, cell


def block_cycle(S, adj):
    sub = {v: [w for w in adj[v] if w in S] for v in S}
    core = set(S)
    ch = True
    while ch:
        ch = False
        for v in list(core):
            if sum(1 for w in sub[v] if w in core) < 2:
                core.discard(v)
                ch = True
    return core, sub


def order_cycle(core, sub):
    start = min(core)
    out = [start]
    prev = None
    cur = start
    while True:
        nxt = [w for w in sub[cur] if w in core and w != prev]
        if not nxt:
            return None
        cur, prev = nxt[0], cur
        if cur == start:
            return out
        out.append(cur)


def block_types(blocks, cell, gens):
    """Blocks related by a translation in H are the same type."""
    cube = L ** 3

    def unsite(i):
        return (i // cube, cell(i))

    def site(v, c):
        return v * cube + c[0] * L * L + c[1] * L + c[2]

    H = {(0, 0, 0)}
    while True:
        new = {tuple((a[k] + g[k]) % L for k in range(3))
               for a in H for g in gens}
        if new <= H:
            break
        H |= new
    key = {frozenset(b): i for i, b in enumerate(blocks)}
    typ = {}
    t = 0
    for i, b in enumerate(blocks):
        if i in typ:
            continue
        for h in sorted(H):
            img = frozenset(
                site(unsite(v)[0], tuple((c + d) % L
                                         for c, d in zip(unsite(v)[1], h)))
                for v in b)
            if img in key:
                typ[key[img]] = t
        t += 1
    return typ, t


def build(blocks, adj, colour, disp, n, cell, gens):
    blk_of = {v: i for i, b in enumerate(blocks) for v in b}
    typ, ntypes = block_types(blocks, cell, gens)
    out_blocks = []
    for bi, b in enumerate(blocks):
        S = set(b)
        sites = sorted(S)
        q = {v: k for k, v in enumerate(sites)}
        core, sub = block_cycle(S, adj)
        ring = order_cycle(core, sub)
        if ring is None or len(ring) != 10:
            raise SystemExit("block %d has no clean 10-cycle" % bi)
        word = ""
        for v in ring:
            on = {ring[(ring.index(v) + 1) % 10], ring[ring.index(v) - 1]}
            third = [w for w in adj[v] if w not in on]
            if len(third) != 1:
                raise SystemExit("block %d loop site %d is chorded" % (bi, v))
            word += PAULI[colour[(v, third[0])]]
        ib = sorted((q[u], q[v], colour[(u, v)])
                    for u in S for v in adj[u] if u < v and v in S)
        out_blocks.append({
            "index": bi, "type": typ[bi],
            "sites": sites,
            "loop": [q[v] for v in ring],
            "loop_sites": ring,
            "loop_pauli": word,
            "internal_bonds": ib,
        })
    seams = []
    for u in range(n):
        for v in adj[u]:
            if u >= v or blk_of[u] == blk_of[v]:
                continue
            bi, bj = blk_of[u], blk_of[v]
            seams.append([bi, out_blocks[bi]["sites"].index(u),
                          bj, out_blocks[bj]["sites"].index(v),
                          colour[(u, v)]])
    return out_blocks, seams, ntypes


def solve_coils(adj, disp, tree, seconds, workers):
    from ortools.sat.python import cp_model
    arcs = [(u, v) for u in sorted(tree) for v in adj[u] if v in tree]
    m = cp_model.CpModel()
    x = [{a: m.NewBoolVar("") for a in arcs} for _ in range(3)]
    out_of, in_to = collections.defaultdict(list), collections.defaultdict(list)
    for a in arcs:
        out_of[a[0]].append(a)
        in_to[a[1]].append(a)
    use = [{} for _ in range(3)]
    for k in range(3):
        for v in sorted(tree):
            u = m.NewBoolVar("")
            m.Add(sum(x[k][a] for a in out_of[v]) == u)
            m.Add(sum(x[k][a] for a in in_to[v]) == u)
            use[k][v] = u
        for ax in range(3):
            m.Add(sum(disp[a][ax] * x[k][a] for a in arcs)
                  == (L if ax == k else 0))
    for v in sorted(tree):
        m.Add(sum(use[k][v] for k in range(3)) <= 1)
    m.Minimize(sum(x[k][a] for k in range(3) for a in arcs))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = seconds
    s.parameters.num_workers = workers
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return s.StatusName(st), None
    # The circulation's support is a disjoint union of simple cycles.
    # With the winding pinned it is normally one, but nothing forbids
    # several, so the components are extracted and the caller decides.
    res = []
    for k in range(3):
        nxt = {a[0]: a[1] for a in arcs if s.Value(x[k][a])}
        comps, seen = [], set()
        for v0 in sorted(nxt):
            if v0 in seen:
                continue
            cyc, v = [], v0
            while v not in seen:
                seen.add(v)
                cyc.append(v)
                v = nxt[v]
            comps.append(cyc)
        res.append(comps)
    return s.StatusName(st), res


# =====================================================================
# VALIDATION -- the manifest re-derived from the lattice, not trusted
# =====================================================================

def validate(man, adj, colour, disp, n):
    bad = []
    blocks = man["blocks"]
    seen = collections.Counter()
    for b in blocks:
        S = b["sites"]
        if len(S) != len(set(S)):
            bad.append("block %d repeats a site" % b["index"])
        for v in S:
            seen[v] += 1
        if len(b["internal_bonds"]) != len(S):
            bad.append("block %d: %d internal bonds for %d sites"
                       % (b["index"], len(b["internal_bonds"]), len(S)))
        for qi, qj, c in b["internal_bonds"]:
            u, v = S[qi], S[qj]
            if v not in adj[u]:
                bad.append("block %d: %d-%d is not a bond" % (b["index"], u, v))
            elif colour[(u, v)] != c:
                bad.append("block %d: bond %d-%d colour %d, manifest says %d"
                           % (b["index"], u, v, colour[(u, v)], c))
        ring = b["loop_sites"]
        if len(ring) != 10 or set(ring) - set(S):
            bad.append("block %d: bad loop" % b["index"])
        else:
            for k in range(10):
                if ring[(k + 1) % 10] not in adj[ring[k]]:
                    bad.append("block %d: loop is not a cycle" % b["index"])
                    break
            if [S[q] for q in b["loop"]] != ring:
                bad.append("block %d: loop indices disagree with sites"
                           % b["index"])
            w = ""
            for k, v in enumerate(ring):
                on = {ring[(k + 1) % 10], ring[k - 1]}
                third = [x for x in adj[v] if x not in on]
                w += PAULI[colour[(v, third[0])]] if len(third) == 1 else "?"
            if w != b["loop_pauli"]:
                bad.append("block %d: flux word %s, re-derived %s"
                           % (b["index"], b["loop_pauli"], w))
    if set(seen) != set(range(n)) or set(seen.values()) != {1}:
        bad.append("blocks do not cover every site exactly once")
    real = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                real.add((u, v))
    ms = set()
    for bi, qi, bj, qj, c in man["seams"]:
        u, v = blocks[bi]["sites"][qi], blocks[bj]["sites"][qj]
        if bi == bj:
            bad.append("seam entry inside one block")
        if v not in adj[u]:
            bad.append("seam %d-%d is not a bond" % (u, v))
        elif colour[(u, v)] != c:
            bad.append("seam %d-%d colour mismatch" % (u, v))
        ms.add((min(u, v), max(u, v)))
    internal = sum(len(b["internal_bonds"]) for b in blocks)
    if len(ms) != len(man["seams"]):
        bad.append("seam list has duplicates")
    if internal + len(ms) != len(real):
        bad.append("internal %d + seam %d != %d bonds"
                   % (internal, len(ms), len(real)))
    loops = {v for b in blocks for v in b["loop_sites"]}
    for bi, qi, bj, qj, c in man["seams"]:
        if blocks[bi]["sites"][qi] in loops or blocks[bj]["sites"][qj] in loops:
            bad.append("a seam bond touches a loop site (not a buried tiling)")
            break
    for ax, path in sorted(man.get("coils", {}).items()):
        tot = [0, 0, 0]
        for k, v in enumerate(path):
            w = path[(k + 1) % len(path)]
            if w not in adj[v]:
                bad.append("coil %s is not a cycle" % ax)
                break
            for t in range(3):
                tot[t] += disp[(v, w)][t]
            if v in loops:
                bad.append("coil %s touches a loop site" % ax)
                break
        else:
            want = [L if AX == ax else 0 for AX in ("x", "y", "z")]
            if tot != want:
                bad.append("coil %s winds %s, want %s" % (ax, tot, want))
    cs = list(man.get("coils", {}).values())
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            if set(cs[i]) & set(cs[j]):
                bad.append("coils share a site")
    return bad


def coil_report(path, adj, colour, disp, n, seconds, workers):
    """What the coil question looks like for one tiling, without
    building a manifest. Useful for choosing between several verified
    tilings: they differ in how much tree budget the coils cost and how
    much of the lattice they sample."""
    d = json.load(open(path))
    blocks = [list(b) for b in d["blocks"]]
    blk_of = {v: i for i, b in enumerate(blocks) for v in b}
    loops = set()
    for b in blocks:
        core, _ = block_cycle(set(b), adj)
        loops |= core
    tree = set(range(n)) - loops
    seam = [(u, v) for u in range(n) for v in adj[u]
            if u < v and blk_of[u] != blk_of[v]]
    on_loop = sum(1 for u, v in seam if u in loops or v in loops)
    print("\n=== %s ===" % os.path.basename(path))
    print("  loop sites %d, tree sites %d, seam bonds %d (%d touching a loop)"
          % (len(loops), len(tree), len(seam), on_loop))
    st, res = solve_coils(adj, disp, tree, seconds, workers)
    if res is None:
        print("  three disjoint coils: %s" %
              ("NONE EXIST" if st == "INFEASIBLE" else
               "%s -- nothing concluded, raise --seconds" % st))
        return
    per = collections.Counter()
    for k, comps in enumerate(res):
        ln = sum(len(c) for c in comps)
        spans = len({blk_of[v] for c in comps for v in c})
        for c in comps:
            for v in c:
                per[blk_of[v]] += 1
        note = "" if len(comps) == 1 else \
            "  [%d components, not one winding]" % len(comps)
        print("  coil %s: %3d sites across %2d of %d blocks%s"
              % ("xyz"[k], ln, spans, len(blocks), note))
    print("  %d tree sites used of %d; busiest block %d of %d"
          % (sum(per.values()), len(tree), max(per.values()) if per else 0,
             len(blocks[0]) - 10))
    print("  status %s%s" % (st, "" if st == "OPTIMAL" else
                             " -- feasible, not proven minimal"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tiling", nargs="+")
    ap.add_argument("-o", "--out", default="manifest.json")
    ap.add_argument("--engine")
    ap.add_argument("--gens", default="3,0,0:0,3,0")
    ap.add_argument("--no-coils", action="store_true")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--validate-only", action="store_true",
                    help="treat the inputs as manifests and check them")
    ap.add_argument("--coils-only", action="store_true",
                    help="report the coil question per tiling, write nothing")
    a = ap.parse_args()
    k4 = find_engine(a.engine)
    adj, colour, disp, n, cell = lattice(k4)
    print("lattice: %d sites, one bond of each colour per site" % n)

    if a.validate_only:
        for f in a.tiling:
            man = json.load(open(f))
            bad = validate(man, adj, colour, disp, n)
            print("%-24s %d problems" % (os.path.basename(f), len(bad)))
            for b in bad[:10]:
                print("  " + b)
        return

    if a.coils_only:
        for f in a.tiling:
            coil_report(f, adj, colour, disp, n, a.seconds, a.workers)
        return

    if len(a.tiling) > 1:
        raise SystemExit("export takes one tiling; use --coils-only or "
                         "--validate-only for several")
    d = json.load(open(a.tiling[0]))
    blocks = [list(b) for b in d["blocks"]]
    gens = [tuple(int(x) for x in g.split(",")) for g in a.gens.split(":")]
    ob, seams, ntypes = build(blocks, adj, colour, disp, n, cell, gens)
    man = {"L": L, "n_sites": n, "n_blocks": len(ob), "n_types": ntypes,
           "qubits_per_block": len(ob[0]["sites"]),
           "source": os.path.basename(a.tiling[0]),
           "pauli_by_colour": list(PAULI),
           "blocks": ob, "seams": seams}

    if not a.no_coils:
        loops = {v for b in ob for v in b["loop_sites"]}
        tree = set(range(n)) - loops
        st, res = solve_coils(adj, disp, tree, a.seconds, a.workers)
        if res and all(len(c) == 1 for c in res):
            res = [c[0] for c in res]
            man["coils"] = {"x": res[0], "y": res[1], "z": res[2]}
            print("coils: %s, lengths %s%s"
                  % (st, [len(r) for r in res],
                     "" if st == "OPTIMAL" else
                     "  -- not proven minimal; 18 each is the geodesic "
                     "floor, raise --seconds"))
        elif res:
            print("coils: %s but a direction came back as %s components, "
                  "not one winding -- omitted from the manifest"
                  % (st, [len(c) for c in res]))
        else:
            print("coils: %s -- omitted from the manifest" % st)

    bad = validate(man, adj, colour, disp, n)
    print("validation: %d problems" % len(bad))
    for b in bad[:10]:
        print("  " + b)
    if bad:
        raise SystemExit("manifest not written")
    # The word is read off starting at the loop's lowest site index, so
    # two blocks with the same flux operator print different strings.
    # The rotation class is what is actually invariant.
    def canon(w):
        return min(w[i:] + w[:i] for i in range(len(w)))
    words = collections.Counter(canon(b["loop_pauli"]) for b in ob)
    print("%d blocks, %d types, %d qubits each, %d seam bonds"
          % (len(ob), ntypes, len(ob[0]["sites"]), len(seams)))
    print("flux words up to rotation: %d  %s" % (len(words), dict(words)))
    print("  (%d raw strings; they differ only in where the loop is entered)"
          % len({b["loop_pauli"] for b in ob}))
    json.dump(man, open(a.out, "w"))
    print("written to %s (%.0f KB)" % (a.out, os.path.getsize(a.out) / 1024))


AX = None
if __name__ == "__main__":
    main()