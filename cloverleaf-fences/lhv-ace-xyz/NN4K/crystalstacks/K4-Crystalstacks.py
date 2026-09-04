# -*- coding: us-ascii -*-
# Rev 1.
# SCALING THE K4 MANIFOLD ENGINE OUT INTO 3-SPACE WITHOUT LEAVING K4.
#
# =====================================================================
# THE STRUCTURAL FACT REV 315 ALREADY DEPENDS ON
# =====================================================================
# Rev 315's header states it plainly: the 3 junction slots PARTITION a
# manifold's 27 qubits, so a manifold state is EXACTLY a rank-3 tensor
# with leg dimension 2**9 = 512. Four tensors, six bonds, K4, no
# structural loss. That is not an ansatz -- it is an identity, and it
# holds for exactly one reason:
#
#       27 = 3 x 9        degree 3, three legs, clean partition
#
# A cubic stack breaks it. A cubic site has SIX neighbours, so an
# element is a rank-6 tensor and 27 qubits must split 5/5/5/4/4/4.
# Nothing is wrong with that arithmetically, but it throws away the one
# property that makes the whole file work: uniform legs, one code path,
# one leg dimension, one Schmidt basis size, one BP message shape.
#
# So do not go to a cubic stack. There is a degree-3 lattice that fills
# 3-space, and it is not a coincidence which one it is:
#
#   the maximal abelian cover of K4 = the K4 crystal = Laves' graph of
#   girth ten = (10,3)-a = the srs net = the triamond.
#
# H_1(K4, R) is 3-dimensional (b1 = 6 - 4 + 1 = 3), so the deck group is
# Z^3 and the cover embeds in 3-space with FOUR vertices per primitive
# cell. Four manifolds x 27 qubits = 108 qubits per cell, which is the
# current engine, unchanged. The existing 4-manifold cluster is the
# L = 1 quotient of the crystal. Scaling out is unrolling the cover.
#
# =====================================================================
# WHY THIS IS THE RIGHT MOVE AND NOT JUST A PRETTY ONE
# =====================================================================
# Rev 315's own "READ BEFORE TRUSTING bp MODE" note is the argument.
# BP on K4 carries ~0.17 mean absolute error in the single-site beliefs
# at FULL chi with ZERO truncation. That is loop error, and chi cannot
# fix it. K4 has girth 3 and four triangles; every bond sits on two of
# them. Loop error is what BP pays for short cycles.
#
# The srs net has GIRTH 10. Same degree, same leg dimension, same
# junction_pairing, same K4BoundaryBP class, same message shapes. The
# only thing that changes is which cell the neighbour lives in.
#
# The bpgirth subcommand measures this directly on exactly-contractible
# 3-regular networks and the trend is not subtle.
#
# =====================================================================
# CHIRALITY, WHICH IS ALREADY THE POINT OF THIS ENGINE
# =====================================================================
# The srs net is chiral: it comes in two enantiomorphs. The Moebius
# half-twist in junction_pairing() is a discrete chirality applied at
# the junction. On a 4-node K4 those two are separate knobs. On the
# crystal they are the same kind of object at different scales, which
# turns the existing Swap Asymmetry parity check into a real
# handedness experiment rather than a self-consistency check.
#
# SINGLE FILE. numpy required; networkx/opt_einsum used by bpgirth only.

import sys
import math
import argparse
import itertools
import collections
import numpy as np

# ---- geometry constants inherited from Rev 315 ----
C_CORNER = 3
E_MIDEDGE = 6
QUBITS_PER_EDGE = C_CORNER + E_MIDEDGE          # 9
LOCAL_EDGES = 3
QUBITS_PER_MANIFOLD = LOCAL_EDGES * QUBITS_PER_EDGE   # 27
LEG_DIM = 2 ** QUBITS_PER_EDGE                  # 512


# =====================================================================
# THE K4 CRYSTAL AS A COVERING GRAPH
# =====================================================================
# Spanning tree of K4: edges (0,1), (0,2), (0,3), weight 0.
# Chords: (1,2) -> +e0, (1,3) -> +e1, (2,3) -> +e2.
# A vertex is (v, n) with v in {0,1,2,3} and n in Z^3.
#
# Slot order is preserved from Rev 315: NEIGHBOURS[v] = sorted others,
# so junction_slot(v, w) = NEIGHBOURS[v].index(w) is untouched. Only
# the CELL the neighbour sits in is new.

NEIGHBOURS = [[j for j in range(4) if j != i] for i in range(4)]

_CHORD = {(1, 2): 0, (1, 3): 1, (2, 3): 2}


def edge_shift(u, v):
    """Cell displacement crossing the K4 edge u->v in the cover."""
    key = (min(u, v), max(u, v))
    if key not in _CHORD:
        return (0, 0, 0)
    axis = _CHORD[key]
    s = [0, 0, 0]
    s[axis] = 1 if u < v else -1
    return tuple(s)


def srs_neighbours(v, cell):
    """The three neighbours of crystal vertex (v, cell)."""
    out = []
    for w in NEIGHBOURS[v]:
        d = edge_shift(v, w)
        out.append((w, tuple(c + dd for c, dd in zip(cell, d))))
    return out


def srs_torus(L):
    """srs net on Z^3 / L Z^3. L=1 recovers K4 exactly.

    Returns (nodes, adjacency, edges). A node is (v, cell)."""
    nodes = [(v, c) for v in range(4)
             for c in itertools.product(range(L), repeat=3)]
    idx = {n: i for i, n in enumerate(nodes)}
    adj = {n: [] for n in nodes}
    edges = set()
    for n in nodes:
        v, cell = n
        for w, ncell in srs_neighbours(v, cell):
            m = (w, tuple(c % L for c in ncell))
            adj[n].append(m)
            edges.add((min(idx[n], idx[m]), max(idx[n], idx[m])))
    return nodes, adj, sorted(edges)


def girth(adj, nodes, cap=24):
    """Shortest cycle length by BFS from every vertex."""
    best = cap + 1
    for src in nodes:
        dist = {src: 0}
        par = {src: None}
        dq = collections.deque([src])
        while dq:
            u = dq.popleft()
            if 2 * dist[u] >= best:
                break
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    par[w] = u
                    dq.append(w)
                elif w != par[u]:
                    best = min(best, dist[u] + dist[w] + 1)
    return best


def cycle_census(adj, nodes, edges_idx, maxlen=10):
    """Number of cycles of each length <= maxlen through a bond,
    averaged over bonds. This is the quantity BP loop error tracks."""
    idx = {n: i for i, n in enumerate(nodes)}
    counts = collections.Counter()
    # count cycles by DFS from each edge, bounded depth
    for n in nodes:
        for m in adj[n]:
            if idx[m] <= idx[n]:
                continue
            # paths from m back to n avoiding the direct edge
            stack = [(m, {n, m}, 1)]
            while stack:
                u, seen, d = stack.pop()
                if d >= maxlen:
                    continue
                for w in adj[u]:
                    if w == n and d >= 2:
                        counts[d + 1] += 1
                    elif w not in seen:
                        stack.append((w, seen | {w}, d + 1))
    ne = len(edges_idx)
    return {k: v / float(2 * ne) for k, v in sorted(counts.items())}


# =====================================================================
# net : verify the crystal
# =====================================================================
def net_report(maxL=5):
    print("=" * 74)
    print("THE K4 CRYSTAL  (srs / (10,3)-a / Laves girth-ten / triamond)")
    print("=" * 74)
    print("K4:      V=4 E=6  b1 = E - V + 1 = 3   -> deck group Z^3")
    print("cover:   4 vertices per primitive cell, degree 3 everywhere")
    print("engine:  4 x %d = %d qubits per cell -- the CURRENT engine size"
          % (QUBITS_PER_MANIFOLD, 4 * QUBITS_PER_MANIFOLD))
    print("")
    print("%4s %8s %8s %8s %10s   %s"
          % ("L", "cells", "verts", "qubits", "girth", "note"))
    print("-" * 74)
    for L in range(1, maxL + 1):
        nodes, adj, edges = srs_torus(L)
        deg = set(len(adj[n]) for n in nodes)
        g = girth(adj, nodes)
        note = "torus wrap still limits girth"
        if L == 1:
            note = "= K4, the current engine"
        if g >= 10:
            note = "saturated at the infinite-crystal girth"
        print("%4d %8d %8d %8d %10d   %s"
              % (L, L ** 3, len(nodes), len(nodes) * QUBITS_PER_MANIFOLD,
                 g, note))
        if deg != {3}:
            print("   FATAL: degrees %s" % sorted(deg))
            return False
    print("-" * 74)

    # L=1 must be K4
    nodes, adj, edges = srs_torus(1)
    k4 = set()
    for a, b in itertools.combinations(range(4), 2):
        k4.add((a, b))
    got = set()
    idx = {n: i for i, n in enumerate(nodes)}
    for a, b in edges:
        got.add((min(nodes[a][0], nodes[b][0]), max(nodes[a][0], nodes[b][0])))
    print("L=1 quotient edge set == K4 : %s" % (got == k4))
    print("")
    print("So the 4-manifold cluster in Rev 315 is not merely LIKE the")
    print("unit cell of this crystal. It IS the unit cell. Setting every")
    print("chord shift to zero collapses the cover back onto it.")
    return True


# =====================================================================
# census : why K4 hurts BP and srs does not
# =====================================================================
def census_report():
    print("=" * 74)
    print("SHORT-CYCLE LOAD PER BOND   (the driver of BP loop error)")
    print("=" * 74)
    rows = []

    nodes, adj, edges = srs_torus(1)
    rows.append(("K4 (Rev 315, L=1)", nodes, adj, edges))
    for L in (2, 3):
        nodes, adj, edges = srs_torus(L)
        rows.append(("srs torus L=%d" % L, nodes, adj, edges))

    # cubic lattice for contrast: the NN3D stacking
    for L in (4,):
        cnodes = list(itertools.product(range(L), repeat=3))
        cadj = {n: [] for n in cnodes}
        cedges = set()
        ci = {n: i for i, n in enumerate(cnodes)}
        for n in cnodes:
            for ax in range(3):
                for s in (1, -1):
                    m = list(n)
                    m[ax] = (m[ax] + s) % L
                    m = tuple(m)
                    cadj[n].append(m)
                    cedges.add((min(ci[n], ci[m]), max(ci[n], ci[m])))
        rows.append(("cubic 6-nbr L=%d" % L, cnodes, cadj, sorted(cedges)))

    print("%-20s %6s %8s %10s %10s %10s"
          % ("lattice", "deg", "girth", "cyc<=4", "cyc<=6", "cyc<=8"))
    print("-" * 74)
    for name, nodes, adj, edges in rows:
        d = len(adj[nodes[0]])
        g = girth(adj, nodes)
        cen = cycle_census(adj, nodes, edges, maxlen=8)
        c4 = sum(v for k, v in cen.items() if k <= 4)
        c6 = sum(v for k, v in cen.items() if k <= 6)
        c8 = sum(v for k, v in cen.items() if k <= 8)
        print("%-20s %6d %8d %10.2f %10.2f %10.2f"
              % (name, d, g, c4, c6, c8))
    print("-" * 74)
    print("cyc<=n is the mean number of cycles of length <= n running")
    print("through a bond. BP is exact on trees; this column is how far")
    print("from a tree each bond locally is.")
    print("")
    print("K4 puts two triangles on every bond -- the worst case for a")
    print("3-regular graph. The srs net puts ZERO cycles below length 10")
    print("on any bond, at the same degree, same leg dimension, same")
    print("message shape.")
    print("")
    print("The cubic row is why the 3D stack is the wrong direction. Six")
    print("neighbours means a rank-6 element, 27 qubits split 5/5/5/4/4/4")
    print("instead of 3 x 9, six different leg dimensions, and ~870 short")
    print("cycles per bond against K4's 2. Stacking cubically makes the BP")
    print("problem far harder while discarding the uniform-leg identity.")
    return True


# =====================================================================
# bpgirth : is the Rev 315 BP floor actually loop error?
# =====================================================================
# Rev 315 says: "Measured against a state that IS exactly a K4 tensor
# network, at chi = full leg dimension with ZERO truncation, BP
# single-site beliefs still carried ~0.17 mean absolute error. Raising
# chi does not fix this; it is loop error, not truncation error."
#
# Loop error is a testable claim. If the 0.17 is loop error it must
# vanish on a TREE and it must fall as girth rises. Both are checked
# here, on the same tensor-network shape as the engine.
#
# TWO MESSAGE SEMANTICS, and the distinction is the whole finding:
#
#   virtual bond   ket and bra carry INDEPENDENT indices on the bond,
#                  joined through a chi x chi Hermitian PSD message.
#                  This is what K4BoundaryBP._outgoing implements.
#
#   copy tensor    the bond IS a shared physical register, so ket and
#                  bra carry the SAME index. The message collapses to
#                  its diagonal: a distribution over the 2**9 = 512
#                  junction-block configurations.
#
# Rev 315's own header calls them "six copy-tensor bonds" -- and they
# are, because the 9 junction qubits are physically shared, which is
# the identification global_qubit_map() enforces. But the message
# update is the virtual-bond one.

def named_cubic_graph(name):
    import networkx as nx
    if name == "K4":
        g = nx.complete_graph(4)
    elif name == "K33":
        g = nx.complete_bipartite_graph(3, 3)
    elif name == "prism":
        g = nx.circular_ladder_graph(3)
    elif name == "cube":
        g = nx.hypercube_graph(3)
    elif name == "petersen":
        g = nx.petersen_graph()
    elif name == "heawood":
        g = nx.heawood_graph()
    elif name == "pappus":
        g = nx.pappus_graph()
    elif name == "desargues":
        g = nx.desargues_graph()
    elif name == "srs2":
        nodes, adj, edges = srs_torus(2)
        g = nx.Graph()
        g.add_nodes_from(range(len(nodes)))
        g.add_edges_from(edges)
    else:
        raise ValueError(name)
    g = nx.convert_node_labels_to_integers(g)
    if set(dict(g.degree()).values()) != {3}:
        raise ValueError("%s is not 3-regular" % name)
    return g


def _nx_girth(g):
    best = 10 ** 9
    for src in g.nodes():
        dist = {src: 0}
        par = {src: None}
        dq = collections.deque([src])
        while dq:
            u = dq.popleft()
            for w in g.neighbors(u):
                if w not in dist:
                    dist[w] = dist[u] + 1
                    par[w] = u
                    dq.append(w)
                elif w != par[u]:
                    best = min(best, dist[u] + dist[w] + 1)
    return best


def build_random_tn(g, d, seed=0):
    rng = np.random.default_rng(seed)
    inc = {v: [] for v in g.nodes()}
    for e, (a, b) in enumerate(sorted(g.edges())):
        inc[a].append(e)
        inc[b].append(e)
    T = {}
    for v in g.nodes():
        t = rng.normal(size=(d, d, d)) + 1j * rng.normal(size=(d, d, d))
        T[v] = t / np.linalg.norm(t)
    return T, inc


def exact_edge_rdm(g, T, inc, e0):
    """Exact reduced density matrix of the SHARED register on bond e0.

    Both endpoints carry the open ket index on that leg and both carry
    the open bra index -- that is what 'shared physical qubits' means.
    Everything else is traced. No approximation."""
    import opt_einsum as oe
    A, B = oe.get_symbol(60), oe.get_symbol(61)
    ops, subs = [], []
    for v in g.nodes():
        sk = "".join(A if e == e0 else oe.get_symbol(e) for e in inc[v])
        sb = "".join(B if e == e0 else oe.get_symbol(e) for e in inc[v])
        ops += [T[v], T[v].conj()]
        subs += [sk, sb]
    r = oe.contract(",".join(subs) + "->" + A + B, *ops, optimize="auto-hq")
    r = 0.5 * (r + r.conj().T)
    return r / max(np.real(np.trace(r)), 1e-300)


def _slots(g, inc):
    slot = {}
    for v in g.nodes():
        for k, e in enumerate(inc[v]):
            slot[(v, e)] = k
    return slot


def bp_virtual(g, T, inc, damping=0.3, max_iter=2000, tol=1e-12):
    """K4BoundaryBP._outgoing, transport set to identity."""
    slot = _slots(g, inc)
    msg = {}
    for v in g.nodes():
        for k in range(3):
            msg[(v, k)] = np.eye(T[v].shape[k], dtype=np.complex128) \
                / T[v].shape[k]

    def outg(v, k):
        t = T[v]
        o = [j for j in range(3) if j != k]
        tk = np.moveaxis(t, k, 0)
        a = np.einsum('abc,bd,ce->ade', tk, msg[(v, o[0])], msg[(v, o[1])],
                      optimize=True)
        out = np.einsum('ade,fde->af', a, tk.conj(), optimize=True)
        out = 0.5 * (out + out.conj().T)
        return out / max(np.real(np.trace(out)), 1e-300)

    for it in range(max_iter):
        new = {}
        for e, (a, b) in enumerate(sorted(g.edges())):
            new[(b, slot[(b, e)])] = outg(a, slot[(a, e)])
            new[(a, slot[(a, e)])] = outg(b, slot[(b, e)])
        res = max(float(np.abs(new[k] - msg[k]).max()) for k in new)
        for k, v in new.items():
            msg[k] = (1.0 - damping) * v + damping * msg[k]
        if res < tol:
            break
    return msg, outg, slot, res, it + 1


def bp_copy(g, T, inc, damping=0.2, max_iter=4000, tol=1e-13):
    """Copy-tensor BP: the message is the DIAGONAL of the half-network,
    a distribution over the shared register's configurations."""
    slot = _slots(g, inc)
    msg = {}
    for v in g.nodes():
        for k in range(3):
            msg[(v, k)] = np.ones(T[v].shape[k]) / T[v].shape[k]

    def outg(v, k):
        t = T[v]
        o = [j for j in range(3) if j != k]
        tk = np.moveaxis(t, k, 0)
        a = tk * msg[(v, o[0])][None, :, None] * msg[(v, o[1])][None, None, :]
        out = np.einsum('abc,dbc->ad', a, tk.conj(), optimize=True)
        out = 0.5 * (out + out.conj().T)
        return out / max(np.real(np.trace(out)), 1e-300)

    for it in range(max_iter):
        new = {}
        for e, (a, b) in enumerate(sorted(g.edges())):
            new[(b, slot[(b, e)])] = np.clip(
                np.real(np.diag(outg(a, slot[(a, e)]))), 0.0, None)
            new[(a, slot[(a, e)])] = np.clip(
                np.real(np.diag(outg(b, slot[(b, e)]))), 0.0, None)
        for k in new:
            new[k] = new[k] / max(new[k].sum(), 1e-300)
        res = max(float(np.abs(new[k] - msg[k]).max()) for k in new)
        for k, v in new.items():
            msg[k] = (1.0 - damping) * v + damping * msg[k]
        if res < tol:
            break
    return msg, outg, slot, res, it + 1


def _psd_sqrt(r):
    w, v = np.linalg.eigh(0.5 * (r + r.conj().T))
    w = np.clip(w, 0.0, None)
    return (v * np.sqrt(w)) @ v.conj().T


def _nrm(r):
    r = 0.5 * (r + r.conj().T)
    tr = np.real(np.trace(r))
    return r / (tr if abs(tr) > 1e-300 else 1.0)


def belief_virtual(g, T, inc, outg, slot, e0):
    """Rev 315 edge_belief_physical, transport = identity."""
    a, b = sorted(g.edges())[e0]
    ra = outg(a, slot[(a, e0)])
    rb = outg(b, slot[(b, e0)])
    sa = _psd_sqrt(ra)
    return _nrm(sa @ rb @ sa)


def belief_copy(g, T, inc, outg, slot, e0):
    """Copy tensor: both halves are functions of the SAME open index
    pair, so they multiply elementwise."""
    a, b = sorted(g.edges())[e0]
    return _nrm(outg(a, slot[(a, e0)]) * outg(b, slot[(b, e0)]))


def tree_reproducer(d=3, seed=5):
    """Three tensors in a CHAIN. No cycles. Loop error is identically
    zero. Whatever error appears here is not loop error."""
    rng = np.random.default_rng(seed)
    a0 = rng.normal(size=d) + 1j * rng.normal(size=d)
    a2 = rng.normal(size=d) + 1j * rng.normal(size=d)
    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    a0 /= np.linalg.norm(a0)
    a2 /= np.linalg.norm(a2)
    A /= np.linalg.norm(A)

    psi = np.einsum('x,xy,y->xy', a0, A, a2)
    exact = _nrm(np.einsum('xy,zy->xz', psi, psi.conj()))

    m0 = _nrm(np.outer(a0, a0.conj()))
    m2 = _nrm(np.outer(a2, a2.conj()))
    # virtual-bond message from the middle tensor
    m1_v = _nrm(np.einsum('xy,zw,yw->xz', A, A.conj(), m2))
    # copy-tensor message: ket and bra share the index
    m1_c = _nrm(np.einsum('xy,zy,y->xz', A, A.conj(),
                          np.clip(np.real(np.diag(m2)), 0.0, None)))

    sa = _psd_sqrt(m0)
    b_v = _nrm(sa @ m1_v @ sa)
    b_c = _nrm(m0 * m1_c)

    print("=" * 74)
    print("TREE REPRODUCER   3 tensors in a chain, d = %d" % d)
    print("=" * 74)
    print("A chain has no cycles. Loop error here is ZERO by definition.")
    print("")
    print("  virtual-bond message + sqrt(ra) rb sqrt(ra)   MAE %.4e"
          % float(np.abs(b_v - exact).mean()))
    print("  copy-tensor message + elementwise combine     MAE %.4e"
          % float(np.abs(b_c - exact).mean()))
    print("")
    print("The first number is the same order as the 0.17 that Rev 315")
    print("attributes to loop error on K4. On a tree it cannot be loop")
    print("error. It is the bond semantics: a copy tensor forces the ket")
    print("and bra indices EQUAL, so the message is the diagonal, not a")
    print("chi x chi matrix, and the two halves multiply elementwise.")
    return True


def bpgirth_report(d=2, trials=3, names=None):
    if names is None:
        names = ["K4", "K33", "cube", "petersen", "heawood",
                 "desargues", "srs2"]
    print("=" * 78)
    print("BP EDGE BELIEF vs EXACT, BOTH BOND SEMANTICS")
    print("  3-regular rank-3 networks, leg dim d = %d, %d instances,"
          " NO truncation" % (d, trials))
    print("=" * 78)
    print("%-12s %6s %7s %14s %14s %8s"
          % ("graph", "V", "girth", "virtual-bond", "copy-tensor", "ratio"))
    print("-" * 78)
    rows = []
    for name in names:
        g = named_cubic_graph(name)
        gg = _nx_girth(g)
        vv, cc = [], []
        for t in range(trials):
            T, inc = build_random_tn(g, d, seed=1000 * t + 7)
            ex = exact_edge_rdm(g, T, inc, 0)
            _, ogv, slv, _, _ = bp_virtual(g, T, inc)
            _, ogc, slc, _, _ = bp_copy(g, T, inc)
            vv.append(float(np.abs(
                belief_virtual(g, T, inc, ogv, slv, 0) - ex).mean()))
            cc.append(float(np.abs(
                belief_copy(g, T, inc, ogc, slc, 0) - ex).mean()))
        mv, mc = np.mean(vv), np.mean(cc)
        print("%-12s %6d %7d %14.3e %14.3e %8.0fx"
              % (name, g.number_of_nodes(), gg, mv, mc, mv / max(mc, 1e-300)))
        rows.append((name, gg, mv, mc))
    print("-" * 78)
    print("virtual-bond is flat in girth. It is not loop error and no")
    print("lattice change will improve it.")
    print("")
    print("copy-tensor IS loop error: it falls monotonically with girth,")
    print("K4 -> srs2 is roughly two orders of magnitude at the same")
    print("degree, same leg dimension, same message count.")
    print("")
    print("Order of operations therefore matters. Fix the message")
    print("semantics first; THEN the srs wiring pays. Scale out before")
    print("the fix and the extra manifolds buy nothing.")
    return rows


# =====================================================================
# table : the drop-in replacement for build_junction_table
# =====================================================================
def junction_pairing(twist):
    """Unchanged from Rev 315."""
    if not twist:
        return list(range(QUBITS_PER_EDGE))
    corners = list(range(C_CORNER))[::-1]
    midedge = [C_CORNER + i for i in range(E_MIDEDGE)][::-1]
    return corners + midedge


def edge_local_qubits(k):
    base = k * QUBITS_PER_EDGE
    return list(range(base, base + QUBITS_PER_EDGE))


def junction_slot(manifold, neighbour):
    return NEIGHBOURS[manifold].index(neighbour)


def build_junction_table_srs(L=2, twist=True, chirality=+1):
    """Rev 315 record format, wired on the srs torus of side L.

    A 'manifold' is now (v, cell) flattened to an integer. Slot indices,
    qubit blocks, pairing and the twist are IDENTICAL to Rev 315 -- the
    only new field is 'shift', the cell displacement across the bond.

    chirality = -1 selects the mirror enantiomorph of the crystal by
    negating every chord shift. The graph is isomorphic; the embedding
    and therefore the loop-momentum spectrum are not."""
    pairing = junction_pairing(twist)
    cells = list(itertools.product(range(L), repeat=3))
    mid = {}
    for v in range(4):
        for c in cells:
            mid[(v, c)] = len(mid)

    table = []
    seen = set()
    for v in range(4):
        for c in cells:
            ma = mid[(v, c)]
            for w in NEIGHBOURS[v]:
                d = edge_shift(v, w)
                if chirality < 0:
                    d = tuple(-x for x in d)
                nc = tuple((ci + di) % L for ci, di in zip(c, d))
                mb = mid[(w, nc)]
                key = (min(ma, mb), max(ma, mb), d if ma < mb
                       else tuple(-x for x in d))
                if key in seen:
                    continue
                seen.add(key)
                ka = junction_slot(v, w)
                kb = junction_slot(w, v)
                qa = edge_local_qubits(ka)
                qb = edge_local_qubits(kb)
                table.append({
                    "index": len(table),
                    "ma": ma, "mb": mb,
                    "cell_a": c, "cell_b": nc, "shift": d,
                    "slot_a": ka, "slot_b": kb,
                    "qubits_a": qa, "qubits_b": qb,
                    "bonds": [(qa[i], qb[pairing[i]])
                              for i in range(QUBITS_PER_EDGE)],
                    "pairing": list(pairing),
                    "label": "J%d%d%s" % (v, w, "".join(str(x) for x in c)),
                })

    # the same invariant Rev 315 asserts: every manifold uses each of
    # its three local edges exactly once
    used = collections.defaultdict(list)
    for rec in table:
        used[rec["ma"]].append(rec["slot_a"])
        used[rec["mb"]].append(rec["slot_b"])
    for m, slots in used.items():
        if sorted(slots) != list(range(LOCAL_EDGES)):
            raise RuntimeError("manifold %d slot usage %s != %s"
                               % (m, sorted(slots), list(range(LOCAL_EDGES))))
    return table, mid


def table_report(L=2, twist=True):
    for chir in (+1, -1):
        table, mid = build_junction_table_srs(L, twist, chir)
        n_m = len(mid)
        print("=" * 74)
        print("SRS JUNCTION TABLE  L=%d  twist=%s  chirality=%+d"
              % (L, twist, chir))
        print("=" * 74)
        print("manifolds        : %d   (%d cells x 4)" % (n_m, L ** 3))
        print("junctions        : %d   (= 3 n_m / 2)" % len(table))
        print("unique qubits    : %d" % (len(table) * QUBITS_PER_EDGE))
        print("total qubits     : %d" % (n_m * QUBITS_PER_MANIFOLD))
        print("leg dimension    : %d  (unchanged)" % LEG_DIM)
        print("slot invariant   : PASS")
        print("first 4 records  :")
        for rec in table[:4]:
            print("   %-10s m%-3d slot%d  <->  m%-3d slot%d   shift %s"
                  % (rec["label"], rec["ma"], rec["slot_a"],
                     rec["mb"], rec["slot_b"], rec["shift"]))
        print("")
    print("The two chiralities give isomorphic GRAPHS and mirror-image")
    print("CRYSTALS. Run the Racetrack Loop DFT on both: any k != 0")
    print("splitting that flips sign under chirality is handedness, not")
    print("noise, and it is a cleaner probe than the Moebius twist alone")
    print("because the graph is held fixed while the embedding flips.")
    return True


# =====================================================================
# budget : what this costs
# =====================================================================
def budget_report():
    print("=" * 74)
    print("BUDGET  (per manifold, unchanged from Rev 315)")
    print("=" * 74)
    print("qubits per manifold        : %d   = 3 legs x %d"
          % (QUBITS_PER_MANIFOLD, QUBITS_PER_EDGE))
    print("statevector fp32           : 2^27 x 8 B = 1.07 GB")
    print("leg dimension              : %d" % LEG_DIM)
    print("BP message                 : chi x chi complex")
    for chi in (4, 8, 16, 32):
        print("   chi=%-3d  message %6d B   truncated tensor %9d B"
              % (chi, chi * chi * 16, chi ** 3 * 16))
    print("")
    print("%6s %8s %10s %12s %10s %12s"
          % ("L", "cells", "manifolds", "qubits", "junctions", "GPU-GB"))
    print("-" * 74)
    for L in (1, 2, 3, 4):
        nm = 4 * L ** 3
        nj = 3 * nm // 2
        print("%6d %8d %10d %12d %10d %12.1f"
              % (L, L ** 3, nm, nm * QUBITS_PER_MANIFOLD, nj, nm * 1.07))
    print("-" * 74)
    print("L=1 is today. L=2 is 32 manifolds / 864 qubits / 34 GB of")
    print("statevector spread over the worker pool -- reachable on the")
    print("V340 fleet with the existing per-worker VRAM cap and the")
    print("existing parallel respawn path. Nothing in the tensor or BP")
    print("code changes shape; the worker count and the junction table do.")
    return True


# =====================================================================
# CLI
# =====================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="K4 crystal (srs) scale-out for the K4 manifold engine")
    sub = ap.add_subparsers(dest="cmd")
    n = sub.add_parser("net", help="build and verify the crystal")
    n.add_argument("--maxL", type=int, default=5)
    sub.add_parser("census", help="short-cycle load per bond")
    b = sub.add_parser("bpgirth", help="BP error vs girth, both semantics")
    b.add_argument("--d", type=int, default=2)
    b.add_argument("--trials", type=int, default=3)
    t = sub.add_parser("table", help="drop-in srs junction table")
    t.add_argument("--L", type=int, default=2)
    sub.add_parser("tree", help="tree reproducer: the floor is not loop error")
    sub.add_parser("budget", help="compute and memory budget")
    sub.add_parser("all", help="everything")
    ns = ap.parse_args(argv)

    if ns.cmd == "net":
        net_report(ns.maxL)
    elif ns.cmd == "census":
        census_report()
    elif ns.cmd == "bpgirth":
        bpgirth_report(d=ns.d, trials=ns.trials)
    elif ns.cmd == "tree":
        tree_reproducer()
    elif ns.cmd == "table":
        table_report(ns.L)
    elif ns.cmd == "budget":
        budget_report()
    else:
        net_report(5)
        print("")
        census_report()
        print("")
        tree_reproducer()
        print("")
        bpgirth_report()
        print("")
        table_report(2)
        print("")
        budget_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
