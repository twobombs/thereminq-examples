# -*- coding: us-ascii -*-
# kitaev_hyperoctagon.py -- Rev 2.
#
# The Kitaev model on the hyperoctagon lattice = (10,3)-a = srs =
# Laves' graph of girth ten = the K4 crystal = the maximal abelian
# cover of K4. Exact free-fermion anchor for the K4 manifold engine.
#
# Merges kitaev_srs.py (Rev 1), srs_flux.py and srs_kappa.py into one
# file, and fixes the boundary conditions.
#
# WHAT CHANGED IN REV 2
# =====================================================================
# Rev 1 computed everything in the uniform u = +1 gauge and flagged the
# flux sector as unresolved. It is resolved:
#
#   - Every elementary 10-loop already carries W = -1 in the uniform
#     gauge. ell = 10 = 2 mod 4, so with A_jk = 2 J u_jk the loop
#     amplitude carries i^10 = -1 and W = -1 IS zero flux. The local
#     sector is Lieb-optimal as it stands and is not a free parameter.
#   - The three Wilson lines of the torus are the only remaining
#     freedom: 8 sectors, not 2^(E-V+1). Verified: the twists leave
#     W = -1 on all 384 ten-loops at L=4, so they are orthogonal to
#     local flux rather than merely coincident with it.
#   - The ground sector on even L is the all-antiperiodic twist
#     (1,1,1). It reproduces, exactly, the best energy found by a
#     2^129 greedy search.
#   - In Bloch form a twist t is a HALF-STEP k-GRID OFFSET,
#     k_a = 2 pi (n_a + t_a / 2) / L. Verified to 0.0 against
#     real-space Majorana diagonalisation at L = 2..6 and all sectors.
#
#   Anchor: E0/site = -0.769884... in the thermodynamic limit.
#   Anchor on EVEN L only -- odd L is not bipartite and is
#   twist-insensitive (all 8 sectors degenerate).
#
# The loop enumerator is Aryan's srs_flux.py, folded in unchanged in
# substance.
#
# STANDING CAVEATS
# =====================================================================
#   - The colour cyclic order used by the kappa term is a convention
#     chosen here, not derived from the standard realisation. It is
#     self-consistent (mirror == kappa -> -kappa, verified to 0.0). On
#     a different tricoordinated net it must be re-fixed against that
#     lattice's embedding.
#   - Handedness does NOT appear in the energy. (10,3)-a is bipartite
#     and the sublattice gauge forces E(+kappa) = E(-kappa) exactly.
#     A chiral observable has to be built from eigenvectors.
#
# numpy only.

import sys
import math
import cmath
import argparse
import itertools
import collections
import numpy as np

# =====================================================================
# PART 1 -- LATTICE
# =====================================================================
NEIGHBOURS = [[j for j in range(4) if j != i] for i in range(4)]
_CHORD = {(1, 2): 0, (1, 3): 1, (2, 3): 2}

# The three perfect matchings of K4. These, NOT the junction slot
# indices, are the Kitaev bond colours: slot_a != slot_b on three of
# the six K4 edges, so a slot is a per-side label and a colour is a
# per-bond one. The matchings lift to a proper 3-edge-colouring of the
# whole crystal.
COLOR_CLASS = {(0, 1): 0, (2, 3): 0,      # x
               (0, 2): 1, (1, 3): 1,      # y
               (0, 3): 2, (1, 2): 2}      # z
COLOR_NAME = ["x", "y", "z"]

# Rev 315 cross-reference:
#   MOEBIUS_PAIRS   = [(0,1),(2,3)]        -> the x colour class
#   CHORD_JUNCTIONS = [(0,2),(1,3)]        -> the y colour class
#   LOOP_JUNCTIONS  = (0,1)(1,2)(2,3)(0,3) -> alternating x z x z


def edge_shift(u, v):
    key = (min(u, v), max(u, v))
    if key not in _CHORD:
        return (0, 0, 0)
    axis = _CHORD[key]
    s = [0, 0, 0]
    s[axis] = 1 if u < v else -1
    return tuple(s)


def junction_slot(m, n):
    return NEIGHBOURS[m].index(n)


def bond_list():
    """The six K4 edges as (v, w, shift, colour), v < w."""
    return [(v, w, edge_shift(v, w), COLOR_CLASS[(v, w)])
            for v, w in itertools.combinations(range(4), 2)]


def srs_sites(L):
    cells = list(itertools.product(range(L), repeat=3))
    idx = {}
    for v in range(4):
        for c in cells:
            idx[(v, c)] = len(idx)
    return idx, cells


def srs_bonds(L, chirality=1):
    """Every bond of the L^3 torus as (i, j, colour)."""
    idx, cells = srs_sites(L)
    out = []
    for v, w, d, col in bond_list():
        if chirality < 0:
            d = tuple(-x for x in d)
        for c in cells:
            nc = tuple((ci + di) % L for ci, di in zip(c, d))
            out.append((idx[(v, c)], idx[(w, nc)], col))
    return out, idx


def is_bipartite(L):
    bonds, idx = srs_bonds(L)
    adj = collections.defaultdict(list)
    for i, j, c in bonds:
        adj[i].append(j)
        adj[j].append(i)
    col = {0: 1}
    st = [0]
    while st:
        u = st.pop()
        for w in adj[u]:
            if w not in col:
                col[w] = -col[u]
                st.append(w)
            elif col[w] == col[u]:
                return False, None
    return True, np.array([col[i] for i in range(len(idx))], dtype=float)


def color_report(maxL=3):
    print("=" * 74)
    print("BOND COLOURING: slots are per-side, colours are per-bond")
    print("=" * 74)
    print("%-9s %7s %7s %8s   %s"
          % ("K4 edge", "slot_a", "slot_b", "colour", "verdict"))
    print("-" * 74)
    bad = 0
    for v, w, d, col in bond_list():
        ka, kb = junction_slot(v, w), junction_slot(w, v)
        if ka != kb:
            bad += 1
        print("%-9s %7d %7d %8s   %s"
              % ("(%d,%d)" % (v, w), ka, kb, COLOR_NAME[col],
                 "ok" if ka == kb else "MISMATCH"))
    print("-" * 74)
    print("%d of 6 edges mismatch, so the colouring must come from the" % bad)
    print("three perfect matchings of K4, not from the slot index.")
    print("")
    for L in range(1, maxL + 1):
        bonds, idx = srs_bonds(L)
        seen = collections.defaultdict(list)
        for i, j, col in bonds:
            seen[i].append(col)
            seen[j].append(col)
        ok = all(sorted(v) == [0, 1, 2] for v in seen.values())
        bip, _ = is_bipartite(L)
        print("  L=%d  %4d sites  proper 3-colouring %-5s  bipartite %s"
              % (L, len(idx), ok, bip))
    return True


# =====================================================================
# PART 2 -- Z2 GAUGE FIELD: LOOPS AND WILSON LINES
# =====================================================================
def elementary_loops(L, ell=10):
    """All ell-cycles of the L^3 torus, deduped by edge set. Returns
    (loops, edge_id_map, bonds, idx)."""
    bonds, idx = srs_bonds(L, 1)
    eid = {}
    adj = collections.defaultdict(list)
    for b, (i, j, c) in enumerate(bonds):
        eid[(i, j)] = (b, +1)
        eid[(j, i)] = (b, -1)
        adj[i].append(j)
        adj[j].append(i)
    uniq = {}
    for s in range(len(idx)):
        stack = [(s, [s], {s})]
        while stack:
            v, path, seen = stack.pop()
            if len(path) == ell:
                if s in adj[v]:
                    key = frozenset(
                        (min(path[t], path[(t + 1) % ell]),
                         max(path[t], path[(t + 1) % ell]))
                        for t in range(ell))
                    if len(key) == ell:
                        uniq.setdefault(key, tuple(path))
                continue
            for w in adj[v]:
                if w > s and w not in seen:
                    stack.append((w, path + [w], seen | {w}))
    return uniq, eid, bonds, idx


def loop_flux(path, u, eid):
    """W = product of traversal-signed u around the loop."""
    ell = len(path)
    w = 1.0
    for t in range(ell):
        b, sgn = eid[(path[t], path[(t + 1) % ell])]
        w *= sgn * u[b]
    return w


def wrap_flags(L):
    """Per bond, which of the three torus directions it wraps."""
    idx, cells = srs_sites(L)
    flags = []
    for v, w, d, col in bond_list():
        for c in cells:
            flags.append(tuple(1 if (c[a] + d[a] < 0 or c[a] + d[a] >= L)
                               else 0 for a in range(3)))
    return np.array(flags)


def wilson_u(L, twist):
    """u for a given (t1,t2,t3) twist of the boundary conditions.
    twist = (0,0,0) is periodic, (1,1,1) all-antiperiodic."""
    fl = wrap_flags(L)
    u = np.ones(fl.shape[0])
    for a in range(3):
        if twist[a]:
            u *= np.where(fl[:, a] == 1, -1.0, 1.0)
    return u


GROUND_TWIST = (1, 1, 1)


def flux_report(L=4, ell=10):
    uniq, eid, bonds, idx = elementary_loops(L, ell)
    n = len(idx)
    print("=" * 74)
    print("LOCAL FLUX IS NOT A FREE PARAMETER")
    print("=" * 74)
    print("L=%d  sites %d  bonds %d" % (L, n, len(bonds)))
    print("elementary %d-loops : %d  (%d through each site)"
          % (ell, len(uniq), ell * len(uniq) // n))
    print("cycle rank E-V+1   : %d  -- the loop data is %dx redundant"
          % (len(bonds) - n + 1, len(uniq) // max(1, len(bonds) - n + 1)))
    print("")
    print("W over the %d-loops, per Wilson sector:" % ell)
    for sec in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]:
        u = wilson_u(L, sec)
        d = collections.Counter(int(round(loop_flux(p, u, eid)))
                                for p in uniq.values())
        print("  twist %s : %s" % (sec, dict(d)))
    print("")
    print("ell = %d = 2 mod 4, so Lieb-optimal flux is zero; with" % ell)
    print("A_jk = 2 J u_jk the loop amplitude carries i^%d = -1, so zero" % ell)
    print("flux means W = -1. The uniform gauge is already there, and the")
    print("Wilson twists do not disturb it -- they are orthogonal to the")
    print("local sector, not merely coincident with it.")
    print("")
    print("So the search space is 8, not 2^%d." % (len(bonds) - n + 1))
    return True


# =====================================================================
# PART 3 -- MAJORANA SOLVER
# =====================================================================
# H = -sum_<jk>_a J_a sigma^a_j sigma^a_k = (i/4) sum A_jk c_j c_k
# with A_jk = 2 J_a u_jk, and E0 = -(1/2) sum of the positive
# eigenvalues of iA.

CYCLIC = ((0, 1), (1, 2), (2, 0))       # colour order x -> y -> z -> x


def majorana_matrix(L, J=(1.0, 1.0, 1.0), u=None, chirality=1,
                    twist=None):
    """A for the L^3 torus. If u is None it is built from `twist`
    (default: the ground sector)."""
    bonds, idx = srs_bonds(L, chirality)
    if u is None:
        u = wilson_u(L, GROUND_TWIST if twist is None else twist)
    n = len(idx)
    A = np.zeros((n, n))
    for b, (i, j, col) in enumerate(bonds):
        t = 2.0 * J[col] * float(u[b])
        A[i, j] += t
        A[j, i] -= t
    return A, bonds, idx


def _colour_neighbours(L, chirality=1):
    bonds, idx = srs_bonds(L, chirality)
    nb = {}
    for b, (i, j, col) in enumerate(bonds):
        nb.setdefault(i, {})[col] = (j, b, +1.0)
        nb.setdefault(j, {})[col] = (i, b, -1.0)
    return nb, bonds, idx


def add_kappa(A, L, kappa, u, chirality=1, order=+1):
    """Kitaev's three-spin term as second-neighbour Majorana hopping,
    amplitude kappa u_kj u_kl. `order` sets the cyclic colour sense
    around each site; order = -1 is the mirror."""
    if kappa == 0.0:
        return A
    nb, bonds, idx = _colour_neighbours(L, chirality)
    pairs = CYCLIC if order > 0 else tuple((b, a) for a, b in CYCLIC)
    for k in range(len(idx)):
        for ca, cb in pairs:
            j, bj, sj = nb[k][ca]
            l, bl, sl = nb[k][cb]
            t = 2.0 * kappa * (sj * u[bj]) * (sl * u[bl])
            A[j, l] += t
            A[l, j] -= t
    return A


def free_fermion_energy(A):
    ev = np.linalg.eigvalsh(1j * A)
    return -0.5 * float(np.sum(ev[ev > 1e-12]))


def energy_real_space(L, J=(1.0, 1.0, 1.0), twist=None, kappa=0.0,
                      chirality=1, order=+1):
    u = wilson_u(L, GROUND_TWIST if twist is None else twist)
    A, bonds, idx = majorana_matrix(L, J, u=u, chirality=chirality)
    A = add_kappa(A, L, kappa, u, chirality, order)
    return free_fermion_energy(A) / len(idx)


# ---- Bloch form. A twist is a half-step k-grid offset. ----
def bloch_matrix(k, J=(1.0, 1.0, 1.0), chirality=1):
    f = np.zeros((4, 4), dtype=np.complex128)
    for v, w, d, col in bond_list():
        if chirality < 0:
            d = tuple(-x for x in d)
        ph = cmath.exp(1j * (k[0] * d[0] + k[1] * d[1] + k[2] * d[2]))
        f[v, w] += 2.0 * J[col] * ph
        f[w, v] -= 2.0 * J[col] * ph.conjugate()
    return f


def bloch_bands(k, J=(1.0, 1.0, 1.0), chirality=1):
    return np.linalg.eigvalsh(1j * bloch_matrix(k, J, chirality))


def energy_bloch(M, J=(1.0, 1.0, 1.0), twist=None, chirality=1):
    """E0 per site from an M^3 k-grid. twist t shifts the grid by t/2
    of a step per axis, which is exactly the Wilson sector."""
    t = GROUND_TWIST if twist is None else twist
    offs = tuple(x / 2.0 for x in t)
    tot = 0.0
    for n in itertools.product(range(M), repeat=3):
        k = tuple(2.0 * math.pi * (n[a] + offs[a]) / M for a in range(3))
        tot += float(np.sum(np.abs(bloch_bands(k, J, chirality))))
    return -0.25 * tot / (M ** 3) / 4.0


# =====================================================================
# PART 4 -- VALIDATION
# =====================================================================
_P = {0: np.array([[0, 1], [1, 0]], dtype=np.complex128),
      1: np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
      2: np.array([[1, 0], [0, -1]], dtype=np.complex128)}


def _kron_pair(n, col, a, b):
    m = np.array([[1.0 + 0.0j]])
    for i in range(n):
        m = np.kron(m, _P[col] if (i == a or i == b) else np.eye(2))
    return m


def validate_report(J=(1.0, 1.0, 1.0)):
    print("=" * 74)
    print("VALIDATION 1 -- SPIN ED vs MAJORANA AT L = 1")
    print("=" * 74)
    bonds, idx = srs_bonds(1)
    n = len(idx)
    H = np.zeros((2 ** n, 2 ** n), dtype=np.complex128)
    for i, j, col in bonds:
        H -= J[col] * _kron_pair(n, col, i, j)
    e_ed = float(np.linalg.eigvalsh(H)[0])
    best = None
    for signs in itertools.product((1, -1), repeat=len(bonds)):
        A, _, _ = majorana_matrix(1, J, u=np.array(signs, dtype=float))
        e = free_fermion_energy(A)
        if best is None or e < best:
            best = e
    print("spin ED ground energy                  : %.12f" % e_ed)
    print("free fermion, best of %d gauge sectors : %.12f"
          % (2 ** len(bonds), best))
    print("agreement                              : %.3e"
          % abs(e_ed - best))
    ok1 = abs(e_ed - best) < 1e-9
    print("VERDICT: %s -- conventions pinned, A_jk = 2 J u_jk,"
          % ("PASS" if ok1 else "FAIL"))
    print("         E0 = -(1/2) sum of positive eigenvalues of iA")
    print("")

    print("=" * 74)
    print("VALIDATION 2 -- WILSON TWIST == HALF-STEP k-GRID OFFSET")
    print("=" * 74)
    print("%4s %-10s %18s %18s %10s"
          % ("L", "twist", "real space", "Bloch M=L", "diff"))
    print("-" * 74)
    worst = 0.0
    for L in (2, 4, 6):
        for sec in [(0, 0, 0), (1, 0, 0), (1, 1, 1)]:
            rs = energy_real_space(L, J, twist=sec)
            bl = energy_bloch(L, J, twist=sec)
            worst = max(worst, abs(rs - bl))
            print("%4d %-10s %18.12f %18.12f %10.1e"
                  % (L, str(sec), rs, bl, abs(rs - bl)))
    print("-" * 74)
    print("worst disagreement: %.1e  VERDICT: %s"
          % (worst, "PASS" if worst < 1e-10 else "FAIL"))
    return ok1 and worst < 1e-10


def anchor_report(J=(1.0, 1.0, 1.0)):
    print("=" * 74)
    print("THE ANCHOR -- GROUND SECTOR, ISOTROPIC J = (1,1,1)")
    print("=" * 74)
    print("Local flux fixed at W = -1 (Lieb-optimal). Wilson twist (1,1,1).")
    print("")
    print("%4s %8s %10s %20s %20s"
          % ("L", "sites", "bipartite", "E0/site ground", "E0/site periodic"))
    print("-" * 74)
    for L in (2, 3, 4, 5, 6):
        bip, _ = is_bipartite(L)
        eg = energy_real_space(L, J, twist=GROUND_TWIST)
        ep = energy_real_space(L, J, twist=(0, 0, 0))
        tag = "" if bip else "   <- odd L, skip"
        print("%4d %8d %10s %20.12f %20.12f%s"
              % (L, 4 * L ** 3, bip, eg, ep, tag))
    print("-" * 74)
    print("Odd L is not bipartite and every twist is degenerate there.")
    print("Anchor on EVEN L only.")
    print("")
    print("Thermodynamic limit, antiperiodic k-grid:")
    print("%6s %22s %14s" % ("M", "E0/site", "delta"))
    prev = None
    for M in (4, 8, 16, 32, 64, 96):
        e = energy_bloch(M, J)
        print("%6s %22.12f %14s"
              % (M, e, "" if prev is None else "%.2e" % abs(e - prev)))
        prev = e
    print("-" * 74)
    print("ANCHOR: E0/site = %.10f" % prev)
    print("")
    print("The antiperiodic grid never samples k = 0, where the Majorana")
    print("Fermi surface sits, so convergence is smooth rather than")
    print("sawtoothed. That is the practical reason to carry the twist")
    print("rather than average over sectors.")
    return prev


# =====================================================================
# PART 5 -- MAJORANA FERMI SURFACE
# =====================================================================
def fermi_report(J=(1.0, 1.0, 1.0), c=1.2):
    print("=" * 74)
    print("GAPLESS LOCUS, tolerance delta = %.2f / M" % c)
    print("=" * 74)
    print("delta MUST scale as 1/M. At fixed delta the shell has fixed")
    print("volume fraction and the count grows as M^3 for a surface, a")
    print("line and a point alike, which measures nothing.")
    print("")

    def count(M, JJ):
        n = 0
        ks = 2.0 * math.pi * (np.arange(M) + 0.5) / M
        d = c / M
        for kx in ks:
            for ky in ks:
                for kz in ks:
                    if np.min(np.abs(bloch_bands((kx, ky, kz), JJ))) < d:
                        n += 1
        return n

    Ms = (16, 24, 32, 48, 64)
    print("%-18s %6s %6s %6s %6s %6s %9s"
          % ("J", "M=16", "24", "32", "48", "64", "fitted d"))
    print("-" * 74)
    for label, JJ in (("(1,1,1)", J), ("(1,1,0.5)", (1.0, 1.0, 0.5)),
                      ("(1,1,0)", (1.0, 1.0, 0.0))):
        cs = [count(M, JJ) for M in Ms]
        good = [(m, x) for m, x in zip(Ms, cs) if x > 0]
        d_fit = (float(np.polyfit(np.log([g[0] for g in good]),
                                  np.log([g[1] for g in good]), 1)[0])
                 if len(good) >= 2 else float('nan'))
        print("%-18s %6d %6d %6d %6d %6d %9.2f"
              % (label, cs[0], cs[1], cs[2], cs[3], cs[4], d_fit))
    print("-" * 74)
    print("d ~ 2 is a Fermi SURFACE, ~1 a nodal line, ~0 points. An")
    print("extensive gapless manifold in 3D with an exact spectrum is")
    print("what makes this lattice worth the trouble.")
    return True


# =====================================================================
# PART 6 -- KAPPA AND HANDEDNESS
# =====================================================================
def kappa_report(Ls=(2, 4, 6), kappas=(0.05, 0.15, 0.3)):
    print("=" * 74)
    print("HANDEDNESS -- AND WHY THE ENERGY CANNOT SEE IT")
    print("=" * 74)
    print("%4s %8s %19s %19s %10s"
          % ("L", "kappa", "E(+kappa)", "E(-kappa)", "split"))
    print("-" * 74)
    for L in Ls:
        for kp in kappas:
            ep = energy_real_space(L, kappa=+kp)
            em = energy_real_space(L, kappa=-kp)
            print("%4d %8.3f %19.12f %19.12f %10.1e"
                  % (L, kp, ep, em, abs(ep - em)))
    print("-" * 74)
    print("")
    L = 4
    u = wilson_u(L, GROUND_TWIST)
    print("Is reversing the colour cyclic order the same as kappa -> -kappa?")
    print("%8s %19s %19s %10s"
          % ("kappa", "E(order=-1,+k)", "E(order=+1,-k)", "diff"))
    for kp in kappas:
        a = energy_real_space(L, kappa=+kp, order=-1)
        b = energy_real_space(L, kappa=-kp, order=+1)
        print("%8.3f %19.12f %19.12f %10.1e" % (kp, a, b, abs(a - b)))
    print("")
    print("Yes. So the mirror is implemented correctly.")
    print("")
    bip, s = is_bipartite(L)
    P = np.diag(s)
    print("The obstruction, made explicit. (10,3)-a is bipartite; let")
    print("P = diag(+1 on one sublattice, -1 on the other). Nearest-")
    print("neighbour terms cross sublattices and flip under P, the")
    print("second-neighbour kappa term does not:")
    for kp in kappas:
        Ap = add_kappa(majorana_matrix(L, u=u)[0].copy(), L, +kp, u)
        Am = add_kappa(majorana_matrix(L, u=u)[0].copy(), L, -kp, u)
        print("   kappa=%.2f   max |P A(+k) P^T + A(-k)| = %.1e"
              % (kp, float(np.abs(P @ Ap @ P.T + Am).max())))
    print("")
    print("iA's spectrum is symmetric under A -> -A, so E(+kappa) and")
    print("E(-kappa) are equal by an exact gauge transformation.")
    print("Bipartiteness therefore forbids a handedness signal in ANY")
    print("function of the Majorana spectrum: energy, gap, DOS, specific")
    print("heat, Fermi surface shape. The crystal is chiral; the")
    print("spectrum cannot see it. A chiral observable must be built")
    print("from eigenvectors -- Chern number of the kappa-gapped bands,")
    print("or the Majorana thermal Hall response, both odd in kappa.")
    print("")
    print("What kappa does do:")
    print("%8s %16s %18s" % ("kappa", "min |eps|", "E0/site"))
    for kp in (0.0, 0.05, 0.1, 0.2, 0.4):
        A = add_kappa(majorana_matrix(L, u=u)[0].copy(), L, kp, u)
        ev = np.linalg.eigvalsh(1j * A)
        print("%8.3f %16.9f %18.12f"
              % (kp, float(np.min(np.abs(ev))),
                 free_fermion_energy(A) / (4 * L ** 3)))
    return True


def chirality_report():
    print("=" * 74)
    print("CHIRALITY IS NOT IN THE SPECTRUM EITHER")
    print("=" * 74)
    for L in (2, 4):
        a = energy_real_space(L, chirality=+1)
        b = energy_real_space(L, chirality=-1)
        print("L=%d  E0/site  right %.12f  left %.12f  diff %.1e"
              % (L, a, b, abs(a - b)))
    print("")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        k = rng.uniform(-math.pi, math.pi, 3)
        worst = max(worst, float(np.abs(
            np.sort(bloch_bands(tuple(k), chirality=-1))
            - np.sort(bloch_bands(tuple(-k), chirality=+1))).max()))
    print("max |bands_-(k) - bands_+(-k)| over 200 random k : %.1e" % worst)
    print("")
    print("Negating the chord shifts is literally k -> -k. It is a")
    print("relabelling, so it measures one crystal's lack of inversion")
    print("symmetry twice rather than comparing two crystals. The real")
    print("mirror is the colour-order reversal in Part 6.")
    return True


# =====================================================================
# RACETRACK LOOP AMPLITUDES
# =====================================================================
# The Rev 315 Racetrack Loop DFT, taken over a real loop of this
# lattice, with an exact reference to benchmark against.
#
# Ground-state Majorana correlator: <c_j c_k> = delta_jk + i G_jk with
#
#     G = -i sgn(iA)          real antisymmetric
#
# and E0 = (1/4) Tr(A G), which is checked against the spectral form.
#
# Per bond of a loop define the racetrack amplitude
#
#     b_r = (traversal-signed u) * G[p_r, p_{r+1}]
#
# Under a gauge change c_j -> s_j c_j BOTH u_jk and G_jk pick up the
# same factor s_j s_k, so b_r is gauge invariant BOND BY BOND, not only
# around the closed loop. That is what makes it a benchmark: an engine
# in a different gauge, with a different junction_pairing convention,
# must still reproduce these numbers exactly.
#
# The DFT of b over the loop index is the racetrack spectrum; the
# circulation prod_r b_r is one gauge-invariant scalar per loop.

def correlation_matrix(A):
    """G = -i sgn(iA). Ground-state Majorana correlator."""
    ev, U = np.linalg.eigh(1j * A)
    S = (U * np.sign(ev)) @ U.conj().T
    G = -1j * S
    return np.real(0.5 * (G - G.T))


def racetrack_amplitudes(L, twist=None, ell=10, J=(1.0, 1.0, 1.0),
                         kappa=0.0, order=+1):
    tw = GROUND_TWIST if twist is None else twist
    u = wilson_u(L, tw)
    A, bonds, idx = majorana_matrix(L, J, u=u)
    A = add_kappa(A, L, kappa, u, 1, order)
    G = correlation_matrix(A)
    uniq, eid, _, _ = elementary_loops(L, ell)
    loops = list(uniq.values())
    colmap = {}
    for i, j, c in bonds:
        colmap[(i, j)] = c
        colmap[(j, i)] = c
    amps, circ, words = [], [], []
    for p in loops:
        b = np.empty(ell)
        w = []
        for r in range(ell):
            a, bb = p[r], p[(r + 1) % ell]
            eb, sgn = eid[(a, bb)]
            b[r] = sgn * u[eb] * G[a, bb]
            w.append(COLOR_NAME[colmap[(a, bb)]])
        amps.append(b)
        circ.append(float(np.prod(b)))
        words.append("".join(w))
    return np.array(amps), np.array(circ), words, loops, A, G, idx


def racetrack_report(L=4, ell=10, kappa=0.0):
    print("=" * 78)
    print("RACETRACK LOOP AMPLITUDES   L=%d, %d-loops, kappa=%.2f"
          % (L, ell, kappa))
    print("=" * 78)
    print("b_r = (signed u) * G[p_r, p_r+1] -- gauge invariant per bond.")
    print("")
    ref = None
    for tw in (GROUND_TWIST, (1, 1, 0), (1, 0, 0), (0, 0, 0)):
        amps, circ, words, loops, A, G, idx = racetrack_amplitudes(
            L, tw, ell, kappa=kappa)
        e_spec = free_fermion_energy(A) / len(idx)
        e_corr = 0.25 * float(np.trace(A @ G)) / len(idx)
        F = np.abs(np.fft.fft(amps, axis=1) / ell).mean(axis=0)
        if ref is None:
            ref = F
        print("twist %s   E0/site %.12f   (from G: %.12f, diff %.1e)"
              % (str(tw), e_spec, e_corr, abs(e_spec - e_corr)))
        print("   loops %d   colour word %s   unique words %d"
              % (len(loops), words[0], len(set(words))))
        print("   |b|        mean %.6f  min %.6f  max %.6f"
              % (np.abs(amps).mean(), np.abs(amps).min(),
                 np.abs(amps).max()))
        print("   circulation mean %+.6e   signs %s"
              % (circ.mean(),
                 dict(collections.Counter(np.sign(circ).astype(int)))))
        print("   |DFT_q|    %s"
              % " ".join("%.5f" % x for x in F))
        print("   drift vs ground sector  %.3e"
              % float(np.abs(F - ref).max()))
        print("")
    print("-" * 78)
    print("The q spectrum is the benchmark. An engine running its")
    print("Racetrack Loop DFT on a genuine 10-loop of this lattice has to")
    print("reproduce these magnitudes; the twist rows tell you how much")
    print("of any deviation is boundary condition rather than method")
    print("error, which is the distinction the K4 racetrack never had.")
    return True


def racetrack_scaling(ell=10):
    print("=" * 78)
    print("RACETRACK SPECTRUM vs L, GROUND SECTOR")
    print("=" * 78)
    print("%4s %8s %8s %11s %13s   %s"
          % ("L", "sites", "loops", "mean |b|", "circulation",
             "|DFT_q| q=0..4"))
    print("-" * 78)
    for L in (2, 4, 6):
        amps, circ, words, loops, A, G, idx = racetrack_amplitudes(L, None,
                                                                    ell)
        if not loops:
            print("%4d %8d %8d   no %d-loops on this torus"
                  % (L, len(idx), 0, ell))
            continue
        F = np.abs(np.fft.fft(amps, axis=1) / ell).mean(axis=0)
        print("%4d %8d %8d %11.6f %13.3e   %s"
              % (L, len(idx), len(loops), np.abs(amps).mean(), circ.mean(),
                 " ".join("%.5f" % x for x in F[:5])))
    print("-" * 78)
    print("Converged values are the anchor at whatever L the engine")
    print("reaches. Note L=2 has girth 6, so its 10-loops wrap the torus")
    print("and are not bulk objects -- read L=4 and L=6.")
    return True


# =====================================================================
# BORDER TENSORS: WHAT chi THE JUNCTION BLOCKS ACTUALLY NEED
# =====================================================================
# Rev 315's border object is T[slot0,slot1,slot2] with leg dimension
# 512, truncated to chi by leg_isometries() -- the top-chi eigenvectors
# of the leg reduced density matrix. BP_CHI defaults to 8, so the
# junction block is carried at rank 8 out of 512.
#
# Whether 8 is enough was previously unanswerable: there was no exact
# border state to compare against. There is now. For free fermions the
# reduced state of a region R is fixed by the correlation matrix
# restricted to R: eigenvalues of i G_R come in pairs +-nu, the mode
# occupations are (1 +- nu)/2, and the Schmidt spectrum is the product
# measure over those independent modes. Exact, no sampling.
#
# PARITY WARNING. A region of n sites carries n matter Majoranas, and a
# fermionic subsystem needs an even count. Odd n leaves an unpaired
# zero mode -- checked: n=9 has exactly one, n=11 has one, n=8 has
# none. So a 9-qubit leg is not a clean fermionic cut in this anchor
# and its entropy sits BELOW the 8-site value rather than above it.
# Even leg sizes are parity-clean; B=24 (legs of 8) and B=30 (legs of
# 10) are, B=27 (legs of 9) is not.
#
# The numbers below are the matter sector only. The Z2 gauge sector
# adds a boundary counting term, so treat them as a floor on chi.

def border_entropy(L=6, sizes=(2, 4, 6, 8, 9, 10, 12, 14), seeds=9,
                   J=(1.0, 1.0, 1.0)):
    u = wilson_u(L, GROUND_TWIST)
    A, bonds, idx = majorana_matrix(L, J, u=u)
    G = correlation_matrix(A)
    adj = collections.defaultdict(list)
    for i, j, c in bonds:
        adj[i].append(j)
        adj[j].append(i)

    def ball(seed, n):
        cur, seen = [seed], {seed}
        while len(cur) < n:
            fr = [w for v in cur for w in adj[v] if w not in seen]
            if not fr:
                break
            cur.append(fr[0])
            seen.add(fr[0])
        return cur

    def h2(p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    print("=" * 78)
    print("EXACT BORDER ENTANGLEMENT AND THE chi IT DEMANDS   L=%d" % L)
    print("=" * 78)
    print("%5s %6s %10s %11s %11s %11s"
          % ("n", "zero", "S (bits)", "chi @ 0.99", "chi @ 0.999",
             "chi @ 0.9999"))
    print("-" * 78)
    step = max(1, len(idx) // seeds)
    for n in sizes:
        Ss, c2, c3, c4, zm = [], [], [], [], 0
        for seed in range(0, len(idx), step):
            R = ball(seed, n)
            if len(R) < n:
                continue
            ev = np.linalg.eigvalsh(1j * G[np.ix_(R, R)])
            zm = int((np.abs(ev) < 1e-9).sum())
            nu = ev[ev > 1e-12]
            Ss.append(float(sum(h2((1 + v) / 2) for v in nu)))
            w = np.array([1.0])
            for v in nu:
                w = np.concatenate([w * (1 + v) / 2, w * (1 - v) / 2])
            w = np.sort(w)[::-1]
            w /= w.sum()
            cw = np.cumsum(w)
            c2.append(int(np.searchsorted(cw, 0.99) + 1))
            c3.append(int(np.searchsorted(cw, 0.999) + 1))
            c4.append(int(np.searchsorted(cw, 0.9999) + 1))
        mark = "  <- Rev 315 leg" if n == 9 else ""
        print("%5d %6d %10.4f %11.1f %11.1f %11.1f%s"
              % (n, zm, np.mean(Ss), np.mean(c2), np.mean(c3),
                 np.mean(c4), mark))
    print("-" * 78)
    print("BP_CHI = 8 sits at roughly the 99 percent level for a 9-site")
    print("leg and short of 99.9. That is about one percent of Schmidt")
    print("weight discarded per leg per truncation, which compounds over")
    print("steps and is a floor no amount of BP iteration removes.")
    print("")
    print("The n=9 row is lower than n=8 because of the unpaired zero")
    print("mode, not because a 9-site cut is cheaper. Read n=8 and n=10")
    print("as the bracket.")
    return True


# =====================================================================
# BEAMLINE: FUNNELLING DATA THROUGH THE RACETRACK
# =====================================================================
# Can the racetrack be driven like an accelerator -- inject, circulate,
# collide, detect? Three of those four, exactly, and the fourth not at
# all. Worth being precise about which.
#
# INJECT. A single Pauli flips u on a bond and creates a flux pair, so
# it kicks the gauge sector as well as the matter sector. The
# flux-preserving injector is the BOND operator sigma^a_i sigma^a_j on
# a correctly coloured bond, which in Majorana form is i u_ij c_i c_j:
# pure matter, leaves every W untouched. Exponentiated it is a rotation
# in the (i,j) Majorana plane, so the state stays Gaussian and the
# correlation matrix transforms as G -> R G R^T with R = exp(2 theta K).
#
# CIRCULATE. Heisenberg evolution of Majoranas is dc/dt = A c, so
# c(t) = exp(A t) c(0) with exp(A t) real orthogonal, and
# G(t) = exp(A t) G exp(-A t). Exact, and it costs one eigendecomposition
# of an N x N real antisymmetric matrix. No Trotter, no sampling.
#
# DETECT. Read the racetrack amplitudes b_r(t) back out. Same
# gauge-invariant observable as the static benchmark, now time resolved.
#
# COLLIDE. No. The model is free -- quadratic in Majoranas even with
# the kappa term -- so counter-propagating packets pass through each
# other. The measured non-additivity below scales linearly in the
# injection amplitude, i.e. it is the two injection rotations failing
# to commute, not a force between beams. A real collision needs a
# non-Kitaev term (Heisenberg, say), and that is exactly the point
# where the exact anchor is lost. Accelerator yes, collider no.
#
# CONFINEMENT. On the bare lattice the packet disperses into the 3D
# bulk within a couple of hopping times. Boosting J on the ten
# racetrack bonds builds a beam pipe, and since that is still a
# quadratic Hamiltonian it stays exactly solvable.

def _expm_antisym(M, t):
    """exp(M t) for real antisymmetric M, via eigh(iM). Real orthogonal."""
    ev, U = np.linalg.eigh(1j * M)
    return np.real(U @ np.diag(np.exp(-1j * ev * t)) @ U.conj().T)


def _kick_rotation(n, i, j, theta):
    K = np.zeros((n, n))
    K[i, j] = 1.0
    K[j, i] = -1.0
    return _expm_antisym(K, 2.0 * theta)


def beamline_setup(L=4, ell=10, twist=None, pipe=1.0, J=(1.0, 1.0, 1.0)):
    """Returns (A, G0, loop, loop_bonds, eid, u, idx). `pipe` multiplies
    J on the racetrack bonds -- the beam pipe."""
    tw = GROUND_TWIST if twist is None else twist
    u = wilson_u(L, tw)
    A, bonds, idx = majorana_matrix(L, J, u=u)
    uniq, eid, _, _ = elementary_loops(L, ell)
    loop = list(uniq.values())[0]
    lb = set()
    for r in range(ell):
        a, b = loop[r], loop[(r + 1) % ell]
        lb.add((min(a, b), max(a, b)))
    if pipe != 1.0:
        for a, b in lb:
            A[a, b] *= pipe
            A[b, a] *= pipe
    G0 = correlation_matrix(A)
    return A, G0, loop, sorted(lb), eid, u, idx


def beamline_report(L=4, ell=10, theta=0.3,
                    pipes=(1.0, 2.0, 4.0, 8.0, 16.0),
                    times=(0.5, 1.0, 2.0, 4.0)):
    print("=" * 78)
    print("BEAMLINE   L=%d, %d-loop, injection theta=%.2f" % (L, ell, theta))
    print("=" * 78)
    print("Injector: bond operator on loop bond 0, flux preserving.")
    print("Retention = fraction of |G(t)-G0|^2 still on the loop sites.")
    print("")
    print("%8s %11s %11s %11s %11s"
          % ("pipe J", "t=%.1f" % times[0], "t=%.1f" % times[1],
             "t=%.1f" % times[2], "t=%.1f" % times[3]))
    print("-" * 78)
    for g in pipes:
        A, G0, loop, lb, eid, u, idx = beamline_setup(L, ell, pipe=g)
        n = len(idx)
        R = _kick_rotation(n, loop[0], loop[1], theta)
        Gk = R @ G0 @ R.T
        row = []
        for t in times:
            O = _expm_antisym(A, t)
            d = (O @ Gk @ O.T - G0) ** 2
            row.append(d[np.ix_(loop, loop)].sum() / max(d.sum(), 1e-30))
        print("%8.0f %11.3f %11.3f %11.3f %11.3f" % (g, *row))
    print("-" * 78)
    print("Bare lattice leaks into the bulk within a hopping time. A")
    print("pipe of 8 holds ~90 percent indefinitely, 16 holds ~97.")
    print("")

    # arrival profile in the confined pipe
    A, G0, loop, lb, eid, u, idx = beamline_setup(L, ell, pipe=8.0)
    n = len(idx)
    R = _kick_rotation(n, loop[0], loop[1], theta)
    Gk = R @ G0 @ R.T
    print("ARRIVAL AROUND THE RING, pipe J = 8")
    print("%8s   %s" % ("t", "  ".join("b%d" % r for r in range(ell))))
    print("-" * 78)
    for t in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8):
        O = _expm_antisym(A, t)
        Gt = O @ Gk @ O.T
        prof = [abs(Gt[loop[r], loop[(r + 1) % ell]]
                    - G0[loop[r], loop[(r + 1) % ell]]) for r in range(ell)]
        print("%8.2f   %s" % (t, "  ".join("%.3f" % x for x in prof)))
    print("-" * 78)
    print("The packet splits and runs both ways round the ring, which is")
    print("what a Majorana bilinear injection does -- it has no momentum")
    print("bias. A one-way beam needs a chiral injector, i.e. two kicks")
    print("phased a quarter period apart on adjacent bonds.")
    print("")

    print("COLLIDER TEST: two counter-propagating packets")
    print("%10s %22s" % ("theta", "relative non-additivity"))
    print("-" * 78)
    A, G0, loop, lb, eid, u, idx = beamline_setup(L, ell, pipe=4.0)
    n = len(idx)
    for th in (0.4, 0.2, 0.1, 0.05, 0.025):
        Ra = _kick_rotation(n, loop[0], loop[1], th)
        Rb = _kick_rotation(n, loop[ell // 2], loop[ell // 2 + 1], th)
        O = _expm_antisym(A, 2.0)
        ea = O @ (Ra @ G0 @ Ra.T) @ O.T - G0
        eb = O @ (Rb @ G0 @ Rb.T) @ O.T - G0
        eab = O @ (Rb @ (Ra @ G0 @ Ra.T) @ Rb.T) @ O.T - G0
        rel = np.abs(eab - (ea + eb)).max() / max(np.abs(eab).max(), 1e-30)
        print("%10.3f %22.3e" % (th, rel))
    print("-" * 78)
    print("Non-additivity is linear in theta, so it vanishes with the")
    print("beams rather than surviving as an interaction. The packets are")
    print("transparent to each other. That is what free fermions means,")
    print("and it is the price of having an exact anchor at all.")
    return True


# =====================================================================
# EXPORT: A RACETRACK MANIFOLD YOU CAN BUILD TODAY
# =====================================================================
# A block of B sites with three legs of B/3 is unicyclic (derived
# earlier: cut = B, internal = B, cyclomatic = 1). At B = 27 that is a
# 10-cycle plus 17 tree sites. So a single 27-qubit manifold holds one
# elementary racetrack of this lattice and nothing is left over.
#
# That means the racetrack is buildable NOW, at one qubit per site,
# without the blocking problem being solved. The blocking only gates
# assembling many manifolds into a lattice; it does not gate putting a
# racetrack inside one.
#
# The catch, quantified in `export`: an ISOLATED 27-block with its 27
# border bonds simply dropped gets b_r wrong by about 30 percent and
# smears the DFT across every harmonic. Open boundaries are not a small
# perturbation here -- half the block's bonds are border bonds.
#
# The fix does not need neighbours either. For a Gaussian state the
# exact reduced state of the block IS G restricted to the block, so the
# export ships that 27x27 matrix as the target. An engine holding 27
# qubits is scored directly against it: no legs, no junction table, no
# lattice assembly, no blocking.

def export_racetrack(L=4, B=27, ell=10, path=None, seed=3):
    import json
    u = wilson_u(L, GROUND_TWIST)
    A, bonds, idx = majorana_matrix(L, u=u)
    G = correlation_matrix(A)
    inv = {v: k for k, v in idx.items()}
    adj = collections.defaultdict(list)
    colmap = {}
    for b, (i, j, c) in enumerate(bonds):
        adj[i].append(j)
        adj[j].append(i)
        colmap[(i, j)] = c
        colmap[(j, i)] = c
    uniq, eid, _, _ = elementary_loops(L, ell)
    loop = list(uniq.values())[0]

    cur, edges = set(loop), ell
    while len(cur) < B:
        fr = [w for v in cur for w in adj[v]
              if w not in cur and sum(1 for x in adj[w] if x in cur) == 1]
        if not fr:
            break
        cur.add(sorted(fr)[0])
        edges += 1
    blk = sorted(cur)
    q = {s: i for i, s in enumerate(blk)}

    internal, border = [], []
    for b, (i, j, c) in enumerate(bonds):
        if i in cur and j in cur:
            internal.append({"qa": q[i], "qb": q[j],
                             "pauli": COLOR_NAME[c], "u": float(u[b])})
        elif (i in cur) != (j in cur):
            inside = i if i in cur else j
            border.append({"q": q[inside], "pauli": COLOR_NAME[c],
                           "u": float(u[b])})

    Gblk = G[np.ix_(blk, blk)]
    Ai = np.zeros((B, B))
    for t in internal:
        w = 2.0 * t["u"]
        Ai[t["qa"], t["qb"]] += w
        Ai[t["qb"], t["qa"]] -= w
    Giso = correlation_matrix(Ai)

    br_t, br_i = [], []
    for r in range(ell):
        a, bb = loop[r], loop[(r + 1) % ell]
        eb, sgn = eid[(a, bb)]
        br_t.append(float(sgn * u[eb] * G[a, bb]))
        br_i.append(float(sgn * u[eb] * Giso[q[a], q[bb]]))
    Ft = np.abs(np.fft.fft(br_t) / ell)
    Fi = np.abs(np.fft.fft(br_i) / ell)

    print("=" * 78)
    print("RACETRACK MANIFOLD EXPORT   L=%d, B=%d, %d-loop" % (L, B, ell))
    print("=" * 78)
    print("block            : %d qubits, %d internal bonds, cyclomatic %d"
          % (len(blk), len(internal), len(internal) - len(blk) + 1))
    print("border half-edges: %d  (= B, as required for three B/3 legs)"
          % len(border))
    print("racetrack qubits : %s" % [q[s] for s in loop])
    print("colour word      : %s"
          % "".join(COLOR_NAME[colmap[(loop[r], loop[(r + 1) % ell])]]
                    for r in range(ell)))
    print("")
    print("b_r, exact (lattice) : %s"
          % " ".join("%.5f" % x for x in br_t))
    print("b_r, isolated block  : %s"
          % " ".join("%.5f" % x for x in br_i))
    print("max deviation        : %.4f  = %.1f%% of the exact value"
          % (max(abs(a - b) for a, b in zip(br_t, br_i)),
             100 * max(abs(a - b) for a, b in zip(br_t, br_i))
             / abs(np.mean(br_t))))
    print("|DFT| exact          : %s" % " ".join("%.5f" % x for x in Ft))
    print("|DFT| isolated       : %s" % " ".join("%.5f" % x for x in Fi))
    print("")
    print("Dropping the border bonds is not a small perturbation: half")
    print("the block's bonds ARE border bonds. Score against the exported")
    print("G_block instead -- it is the exact reduced state of these 27")
    print("sites in the lattice ground state, and it needs no neighbours.")

    spec = {
        "lattice": "hyperoctagon (10,3)-a / srs / K4 crystal",
        "torus_L": L, "twist": list(GROUND_TWIST),
        "anchor_E0_per_site": -0.7698848061,
        "n_qubits": B,
        "qubit_to_site": [{"q": q[s], "k4_label": inv[s][0],
                           "cell": list(inv[s][1])} for s in blk],
        "internal_bonds": internal,
        "border_half_edges": border,
        "racetrack": {"qubits": [q[s] for s in loop],
                      "colour_word": "".join(
                          COLOR_NAME[colmap[(loop[r], loop[(r + 1) % ell])]]
                          for r in range(ell)),
                      "b_r_exact": br_t,
                      "dft_magnitude_exact": Ft.tolist()},
        "G_block_exact": Gblk.tolist(),
        "note": ("H = -sum_bonds J_p sigma^p_a sigma^p_b with p from "
                 "'pauli'. u is the Z2 gauge field; only loop products "
                 "are physical. G_block is the exact ground-state "
                 "Majorana correlator restricted to these 27 sites."),
    }
    if path:
        with open(path, "w") as fh:
            json.dump(spec, fh, indent=1)
        print("")
        print("written: %s" % path)
    return spec


# =====================================================================
# CLI
# =====================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Kitaev on the hyperoctagon / (10,3)-a / K4 crystal")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("colour", help="bond colouring and bipartiteness")
    f = sub.add_parser("flux", help="local flux and Wilson sectors")
    f.add_argument("--L", type=int, default=4)
    sub.add_parser("validate", help="spin ED and twist == k-offset")
    sub.add_parser("anchor", help="ground-sector E0/site and the limit")
    r = sub.add_parser("racetrack", help="loop amplitude benchmark")
    r.add_argument("--L", type=int, default=4)
    r.add_argument("--kappa", type=float, default=0.0)
    bl = sub.add_parser("beamline", help="inject, circulate, read out")
    bl.add_argument("--L", type=int, default=4)
    bl.add_argument("--theta", type=float, default=0.3)
    ex = sub.add_parser("export", help="emit a 27-qubit racetrack spec")
    ex.add_argument("--L", type=int, default=4)
    ex.add_argument("--out", type=str, default="racetrack_manifold.json")
    sub.add_parser("border", help="exact border entropy and chi demand")
    sub.add_parser("fermi", help="gapless locus dimension")
    sub.add_parser("kappa", help="three-spin term and handedness")
    sub.add_parser("chirality", help="enantiomorph relabelling")
    sub.add_parser("all")
    ns = ap.parse_args(argv)
    cmd = ns.cmd or "all"
    if cmd in ("colour", "all"):
        color_report()
        print("")
    if cmd in ("flux", "all"):
        flux_report(getattr(ns, "L", 4))
        print("")
    if cmd in ("validate", "all"):
        validate_report()
        print("")
    if cmd in ("anchor", "all"):
        anchor_report()
        print("")
    if cmd in ("fermi", "all"):
        fermi_report()
        print("")
    if cmd in ("kappa", "all"):
        kappa_report()
        print("")
    if cmd in ("chirality", "all"):
        chirality_report()
        print("")
    if cmd == "export":
        export_racetrack(L=getattr(ns, "L", 4),
                         path=getattr(ns, "out", None))
    if cmd == "beamline":
        beamline_report(L=getattr(ns, "L", 4),
                        theta=getattr(ns, "theta", 0.3))
    if cmd in ("border", "all"):
        border_entropy()
        print("")
    if cmd in ("racetrack", "all"):
        racetrack_report(L=getattr(ns, "L", 4),
                         kappa=getattr(ns, "kappa", 0.0))
        print("")
        racetrack_scaling()
    return 0


if __name__ == "__main__":
    sys.exit(main())
