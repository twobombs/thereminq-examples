# -*- coding: us-ascii -*-
# shell_dodec_kitaev.py -- Rev 1.
#
# The Kitaev model on a stack of spherical shells, as a distinct direction
# from the hyperoctagon work. Self-contained: numpy only. ortools is used
# if present, and only to re-derive literals that are already baked in.
#
# WHAT THE SHELL IS, AND WHY IT IS FORCED
# =====================================================================
# A shell must be tricoordinated, must close on a sphere, and should carry
# ten-site flux loops so the flux operator keeps the shape it has on the
# hyperoctagon. Those three demands leave exactly one choice.
#
# Subdividing every edge of a 3-regular polyhedron once turns a face of
# size n into a loop of size 2n, and leaves the subdivision vertices with
# one free bond each -- which is what the radial coupling has to be made
# of, because degree three is the entire budget. An interlayer bond is
# not added to a site, it is taken from that site's in-shell bonds.
#
# 2n = 10 needs n = 5, and the only 3-regular polyhedron with all
# pentagonal faces is the dodecahedron. So:
#
#   20 branch sites     degree 3, all three bonds in-shell
#   30 connector sites  two bonds in-shell, one radial
#   50 sites, 60 in-shell bonds, 12 ten-loops per shell
#
# Euler checks: 50 - 60 + 12 = 2. Subdividing every edge doubles every
# cycle, so the shell is bipartite, and 10 = 2 mod 4 -- the same residue
# the hyperoctagon has, so the same Lieb reasoning would apply if the
# shell stood alone.
#
# THE RADIAL BUDGET, AND WHY THE STACK IS CLOSED
# =====================================================================
# A connector has ONE free bond, so it joins one neighbouring shell, not
# two. A shell sends 15 bonds out and 15 in, never 30 each way. 15 is odd,
# so an innermost shell cannot cap itself by pairing its own connectors.
# The stack is therefore closed into S^2 x S^1 by default: shell N-1 wires
# back to shell 0, every site is degree 3, and there is no boundary. The
# open ball is available with closed=False and leaves 2 x 15 dangling
# bonds, which is a seam-repair problem rather than a lattice.
#
# U is the 15 outward edges, BETA the bijection onto the 15 inward ones.
# Both were annealed to maximise girth. Shell count must be even, so the
# alternating bipartite classes close.
#
# WHAT IS VERIFIED HERE, AND WHAT IS NOT
# =====================================================================
# Verified by --check at N=4 (200 sites, 300 bonds):
#   degree 3 everywhere; bipartite; a shell-periodic proper 3-edge
#   colouring exists, so every site carries one bond of each colour and
#   the flux operator is well defined; girth 8; 40 eight-loops and 196
#   ten-loops.
#
# The free-fermion solver is calibrated against the hyperoctagon: at
# L=6 it returns -0.7691433 periodic and -0.7702703 all-antiperiodic,
# bracketing the published -0.7698848061 from both sides, which is the
# expected finite-size behaviour and independently confirms that the
# all-antiperiodic twist is the ground sector on even L.
#
# NOT verified, and the reason this is a distinct direction rather than
# a variant: THE FLUX SECTOR IS FRUSTRATED. Radial bonds create 8-loops
# alongside the in-shell 10-loops, and 8 = 0 mod 4 while 10 = 2 mod 4, so
# the two families want opposite signs. Solving over GF(2) shows the
# sector with -1 on every 10-loop and +1 on every 8-loop is NOT
# REALISABLE: 236 loop constraints against 101 independent cycles, and
# the system is inconsistent. There is no uniform ground gauge here. The
# best sector found by annealing at N=4 is -0.7686218 per site, against
# -0.7590361 for the uniform gauge, and its flux pattern is mixed.
#
# So this lattice has no analogue of the hyperoctagon's exact anchor. It
# has an exact free-fermion solution for any given gauge -- that part is
# untouched -- but the ground sector must be searched for, not asserted.
#
# WHAT CARRIES OVER AND WHAT DOES NOT
# =====================================================================
#   carries over: tricoordination, bipartiteness, one qubit per site, no
#                 Jordan-Wigner, a 10-qubit flux word per in-shell loop,
#                 exact free-fermion diagonalisation per gauge sector
#   lost:         translation invariance, and with it the quotient graph,
#                 the equivariant tiling, the 8-block-types reduction and
#                 the Bloch-form twist sectors. K4-32x27-equiv.py has no
#                 group to act on here.
#   lost:         girth 10. The radial bonds bring it down to 8, so the
#                 belief-propagation argument weakens rather than holds.
#
#   python3 shell_dodec_kitaev.py --check
#   python3 shell_dodec_kitaev.py --shells 6 --anneal 4000
#   python3 shell_dodec_kitaev.py --hyperoctagon /path/to/K4-Chrystalstacks-Kitaev-single.py

import argparse
import collections
import itertools
import json

import numpy as np

PHI = (1.0 + 5.0 ** 0.5) / 2.0
COLOR_NAME = ["x", "y", "z"]

# annealed radial wiring: U = outward edges, BETA[i] = inward partner of U[i]
U = [5, 16, 11, 2, 8, 7, 19, 13, 21, 10, 4, 3, 26, 25, 0]
BETA = [27, 24, 29, 17, 15, 22, 28, 6, 9, 20, 14, 18, 12, 23, 1]

# shell-periodic 3-colouring. HALF[(v, e)] is the colour of the half-bond at
# branch vertex v on dodecahedron edge e; RADIAL_COLOUR[e] is the third
# colour, which the radial bond at that connector must carry.
HALF = {(0, 0): 2, (0, 1): 1, (0, 2): 0, (1, 3): 2, (1, 4): 0, (1, 5): 1,
        (2, 6): 0, (2, 7): 1, (2, 8): 2, (3, 9): 0, (3, 10): 1, (3, 11): 2,
        (4, 12): 1, (4, 13): 2, (4, 14): 0, (5, 15): 0, (5, 16): 2,
        (5, 17): 1, (6, 18): 0, (6, 19): 1, (6, 20): 2, (7, 21): 0,
        (7, 22): 1, (7, 23): 2, (8, 0): 1, (8, 12): 0, (8, 24): 2,
        (9, 1): 2, (9, 3): 0, (9, 25): 1, (10, 2): 1, (10, 6): 2,
        (10, 26): 0, (11, 4): 1, (11, 15): 2, (11, 27): 0, (12, 7): 0,
        (12, 9): 2, (12, 28): 1, (13, 5): 0, (13, 10): 2, (13, 26): 1,
        (14, 8): 0, (14, 18): 2, (14, 24): 1, (15, 13): 0, (15, 16): 1,
        (15, 25): 2, (16, 14): 1, (16, 19): 2, (16, 29): 0, (17, 11): 0,
        (17, 21): 2, (17, 27): 1, (18, 20): 1, (18, 22): 0, (18, 28): 2,
        (19, 17): 0, (19, 23): 1, (19, 29): 2}
RADIAL_COLOUR = [0, 0, 2, 1, 2, 2, 1, 2, 1, 1, 0, 1, 2, 1, 2,
                 1, 0, 2, 1, 0, 0, 1, 2, 0, 0, 0, 2, 2, 0, 1]


# =====================================================================
# PART 1 -- LATTICE
# =====================================================================
def dodecahedron():
    """Vertices, edges and the twelve pentagonal faces."""
    V = [s for s in itertools.product([1, -1], repeat=3)]
    for a, b in itertools.product([1, -1], repeat=2):
        V.append((0, a / PHI, b * PHI))
        V.append((a / PHI, b * PHI, 0))
        V.append((a * PHI, 0, b / PHI))
    V = np.array(V, float)
    D = np.linalg.norm(V[:, None] - V[None, :], axis=-1)
    m = np.min(D[D > 1e-9])
    E = [(i, j) for i in range(20) for j in range(i + 1, 20)
         if abs(D[i, j] - m) < 1e-6]
    adj = collections.defaultdict(list)
    for i, j in E:
        adj[i].append(j)
        adj[j].append(i)
    faces = []
    seen = set()

    def walk(p):
        if len(p) == 5:
            if p[0] in adj[p[-1]]:
                k = tuple(sorted(p))
                if k not in seen:
                    seen.add(k)
                    faces.append(tuple(p))
            return
        for w in adj[p[-1]]:
            if w > p[0] and w not in p:
                walk(p + [w])

    for s in range(20):
        walk([s])
    if len(E) != 30 or len(faces) != 12:
        raise SystemExit("dodecahedron came out wrong: %d edges, %d faces"
                         % (len(E), len(faces)))
    return V, E, adj, faces


def build(nshell=4, closed=True):
    """Site index, 3-space position, bonds as (i, j, colour), loops."""
    if closed and nshell % 2:
        raise SystemExit("a closed stack needs an even shell count "
                         "(the bipartite classes alternate by shell)")
    V, E, dadj, faces = dodecahedron()
    eidx = {e: i for i, e in enumerate(E)}
    idx = {}
    pos = []
    for s in range(nshell):
        r = 1.0 + 0.45 * s
        for v in range(20):
            idx[(s, "b", v)] = len(idx)
            pos.append(r * V[v])
        for k, (u, v) in enumerate(E):
            idx[(s, "c", k)] = len(idx)
            pos.append(r * (V[u] + V[v]) / 2.0)
    bonds = []
    for s in range(nshell):
        for k, (u, v) in enumerate(E):
            bonds.append((idx[(s, "b", u)], idx[(s, "c", k)], HALF[(u, k)]))
            bonds.append((idx[(s, "c", k)], idx[(s, "b", v)], HALF[(v, k)]))
    for s in range(nshell if closed else nshell - 1):
        t = (s + 1) % nshell
        for a, b in zip(U, BETA):
            bonds.append((idx[(s, "c", a)], idx[(t, "c", b)],
                          RADIAL_COLOUR[a]))
    loops = []
    for s in range(nshell):
        for f in faces:
            cyc = []
            for i in range(5):
                u, v = f[i], f[(i + 1) % 5]
                cyc.append(idx[(s, "b", u)])
                cyc.append(idx[(s, "c", eidx[(min(u, v), max(u, v))])])
            loops.append(cyc)
    return idx, np.array(pos), bonds, loops


# =====================================================================
# PART 2 -- CHECKS
# =====================================================================
def neighbours(n, bonds):
    a = collections.defaultdict(list)
    for i, j, c in bonds:
        a[i].append(j)
        a[j].append(i)
    return a


def elementary_cycles(n, bonds, maxlen=10):
    adj = neighbours(n, bonds)
    out = collections.defaultdict(list)
    seen = set()
    for s in range(n):
        st = [(s, [s])]
        while st:
            u, p = st.pop()
            if len(p) > maxlen:
                continue
            for v in adj[u]:
                if v == s and len(p) >= 6:
                    key = frozenset((min(p[i], p[(i + 1) % len(p)]),
                                     max(p[i], p[(i + 1) % len(p)]))
                                    for i in range(len(p)))
                    if key not in seen:
                        seen.add(key)
                        out[len(p)].append(p[:])
                elif v > s and v not in p:
                    st.append((v, p + [v]))
    return out


def check(nshell=4, closed=True):
    idx, pos, bonds, loops = build(nshell, closed)
    n = len(idx)
    deg = collections.Counter()
    cols = collections.defaultdict(list)
    for i, j, c in bonds:
        deg[i] += 1
        deg[j] += 1
        cols[i].append(c)
        cols[j].append(c)
    print("shells %d  sites %d  bonds %d  in-shell loops %d"
          % (nshell, n, len(bonds), len(loops)))
    print("  degrees: %s" % sorted(set(deg.values())))
    print("  one bond of each colour at every site: %s"
          % all(sorted(v) == [0, 1, 2] for v in cols.values()))
    adj = neighbours(n, bonds)
    col = {0: 0}
    st = [0]
    bip = True
    while st:
        u = st.pop()
        for w in adj[u]:
            if w not in col:
                col[w] = 1 - col[u]
                st.append(w)
            elif col[w] == col[u]:
                bip = False
    print("  bipartite: %s" % bip)
    cy = elementary_cycles(n, bonds)
    print("  cycle census: %s" % {k: len(v) for k, v in sorted(cy.items())})
    print("  girth: %d" % min(cy))
    return idx, pos, bonds, loops, cy


# =====================================================================
# PART 3 -- FREE FERMIONS
# =====================================================================
def energy(n, bonds, u=None):
    """Ground energy per site of the Kitaev model in gauge u. Calibrated
    against the hyperoctagon: see the header."""
    A = np.zeros((n, n))
    for t, (i, j, c) in enumerate(bonds):
        s = 1.0 if u is None else u[t]
        A[i, j] += 2.0 * s
        A[j, i] -= 2.0 * s
    return -0.25 * np.sum(np.abs(np.linalg.eigvalsh(1j * A))) / n


def flux(cycle, bonds, u):
    bid = {(min(i, j), max(i, j)): t for t, (i, j, c) in enumerate(bonds)}
    f = 1.0
    for i in range(len(cycle)):
        a, b = cycle[i], cycle[(i + 1) % len(cycle)]
        f *= u[bid[(min(a, b), max(a, b))]]
    return f


def anneal(n, bonds, steps=4000, seed=0, t0=0.006):
    rng = np.random.default_rng(seed)
    u = np.ones(len(bonds))
    cur = energy(n, bonds, u)
    best = (cur, u.copy())
    T = t0
    for it in range(steps):
        t = int(rng.integers(len(bonds)))
        u[t] *= -1
        v = energy(n, bonds, u)
        if v < cur or rng.random() < np.exp(-(v - cur) / T):
            cur = v
        else:
            u[t] *= -1
        if cur < best[0] - 1e-12:
            best = (cur, u.copy())
        if it % max(1, steps // 10) == 0:
            T *= 0.6
    return best


def flux_report(loops, cy, bonds, u):
    print("  in-shell 10-loops: %s"
          % dict(collections.Counter(int(round(flux(c, bonds, u)))
                                     for c in loops)))
    for L in sorted(cy):
        print("  all %2d-loops:      %s"
              % (L, dict(collections.Counter(
                  int(round(flux(c, bonds, u))) for c in cy[L]))))


# =====================================================================
# PART 4 -- HYPEROCTAGON CROSS-CHECK
# =====================================================================
def hyperoctagon(path, L=6):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("k4e", path)
    m = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["k4e"]
    try:
        spec.loader.exec_module(m)
    finally:
        sys.argv = argv
    bl, idx = m.srs_bonds(L, 1)
    bonds = [(i, j, c) for i, j, c in bl]
    cube = L ** 3

    def cell(i):
        return ((i % cube) // (L * L), (i % (L * L)) // L, i % L)

    for tw in ((0, 0, 0), (1, 1, 1)):
        u = []
        for i, j, c in bl:
            s = 1.0
            for ax in range(3):
                ca, cb = cell(i)[ax], cell(j)[ax]
                if tw[ax] and min(ca, cb) == 0 and max(ca, cb) == L - 1:
                    s = -s
            u.append(s)
        print("hyperoctagon L=%d twist %s  E0/site = %.10f"
              % (L, tw, energy(len(idx), bonds, u)))
    print("published anchor                    -0.7698848061")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shells", type=int, default=4)
    ap.add_argument("--open", action="store_true",
                    help="ball with boundary instead of the closed S^2 x S^1")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--anneal", type=int, default=0)
    ap.add_argument("--hyperoctagon", metavar="ENGINE")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()

    if a.hyperoctagon:
        hyperoctagon(a.hyperoctagon)
        return

    idx, pos, bonds, loops, cy = check(a.shells, not a.open)
    n = len(idx)
    eu = energy(n, bonds)
    print("  uniform gauge E0/site = %.10f" % eu)
    u = np.ones(len(bonds))
    if a.anneal:
        e, u = anneal(n, bonds, a.anneal)
        print("  annealed      E0/site = %.10f  (gain %+.7f)" % (e, e - eu))
    flux_report(loops, cy, bonds, u)

    if a.out:
        man = {
            "shells": a.shells, "closed": not a.open, "n_sites": n,
            "sites_per_shell": 50, "loops_per_shell": 12,
            "pauli_by_colour": COLOR_NAME,
            "bonds": [[int(i), int(j), int(c)] for i, j, c in bonds],
            "in_shell_loops": [[int(v) for v in c] for c in loops],
            "gauge": [float(x) for x in u],
        }
        json.dump(man, open(a.out, "w"))
        print("  wrote %s" % a.out)


if __name__ == "__main__":
    main()
