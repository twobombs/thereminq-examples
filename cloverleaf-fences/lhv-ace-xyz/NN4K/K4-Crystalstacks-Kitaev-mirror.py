# in the style of minor mending - a mirrored Kitaev Chrystalstacks racetrack circuit
# -*- coding: us-ascii -*-
# srs_racetrack_ring.py -- the racetrack at MANIFOLD scale, as constants.
#
# Everything the engine needs is a module-level literal below. No JSON,
# no file IO, no generation step at import. `from srs_racetrack_ring
# import RING_NODES, RING_JUNCTIONS, BEAM_PORTS` and wire it.
#
# The generator is kept at the bottom under `verify`, which regenerates
# the ring from K4-Crystalstacks and asserts it matches the literals.
# Run it after touching junction_pairing() or edge_shift().
#
# TWO RACETRACKS, TWO SCALES. DO NOT CONFUSE THEM.
# =====================================================================
#   spin scale      one qubit per lattice site; a unicyclic 27-block is
#                   a 10-cycle plus 17 tree sites and fits in ONE
#                   manifold. Has an exact free-fermion anchor.
#   manifold scale  THIS FILE. Each ring node IS a 27-qubit manifold,
#                   each junction identifies 9 qubits with 9. Ten
#                   manifolds, 270 qubits. Needs no blocking: the
#                   manifolds are the srs nodes, which is what
#                   K4-Crystalstacks was built for.
#
# WHY DEGREE 3 HANDS YOU THE BEAM PORTS
# =====================================================================
# A cycle through a trivalent node consumes two of its three bonds, so a
# 10-ring uses 20 slots and leaves EXACTLY TEN FREE, one per manifold.
# That is the injection/extraction port, and the count is forced by the
# coordination number rather than designed in. Degree 4 would leave two
# free legs per node and no canonical port; a cubic stack, four.
#
# INVARIANTS, ALL CHECKED BY `verify`
# =====================================================================
#   * RING_SHIFT_SUM is (0,0,0): the ring is contractible, a bulk object
#     at any L >= 4, not a torus wrap.
#   * every manifold uses exactly 2 of 3 slots in-ring.
#   * junction_pairing(twist=True) is an INVOLUTION, [2,1,0,8,7,6,5,4,3],
#     so the composed pairing around the ring is the identity for any
#     EVEN ring and the raw pairing for any odd one. Girth 10 is even,
#     so every elementary ring of this lattice closes. The Moebius twist
#     is topologically consistent here for free.

# =====================================================================
# THE RING
# =====================================================================
ELL = 10
QUBITS_PER_EDGE = 9
QUBITS_PER_MANIFOLD = 27
LEG_DIM = 512
TWIST = True

# junction_pairing(twist=True). Involution: PAIRING[PAIRING[i]] == i.
PAIRING = [2, 1, 0, 8, 7, 6, 5, 4, 3]

# slot k of a manifold owns these local qubit indices
SLOT_QUBITS = [[0, 1, 2, 3, 4, 5, 6, 7, 8],
               [9, 10, 11, 12, 13, 14, 15, 16, 17],
               [18, 19, 20, 21, 22, 23, 24, 25, 26]]

# ring manifold m -> (K4 label, cell) on the crystal
RING_NODES = [
    (0, (0, 0, 0)),
    (3, (0, 0, 0)),
    (2, (0, 0, -1)),
    (1, (-1, 0, -1)),
    (3, (-1, 1, -1)),
    (0, (-1, 1, -1)),
    (2, (-1, 1, -1)),
    (3, (-1, 1, 0)),
    (1, (-1, 0, 0)),
    (2, (0, 0, 0)),
]

RING_SHIFT_SUM = (0, 0, 0)
COLOUR_WORD = "zxzyzyxyzy"

# Rev 315 junction record format, plus 'shift' and 'colour'.
RING_JUNCTIONS = [
    dict(index=0, ma=0, mb=1, slot_a=2, slot_b=0, shift=(0, 0, 0), colour='z', label='R00'),
    dict(index=1, ma=1, mb=2, slot_a=2, slot_b=2, shift=(0, 0, -1), colour='x', label='R01'),
    dict(index=2, ma=2, mb=3, slot_a=1, slot_b=1, shift=(-1, 0, 0), colour='z', label='R02'),
    dict(index=3, ma=3, mb=4, slot_a=2, slot_b=1, shift=(0, 1, 0), colour='y', label='R03'),
    dict(index=4, ma=4, mb=5, slot_a=0, slot_b=2, shift=(0, 0, 0), colour='z', label='R04'),
    dict(index=5, ma=5, mb=6, slot_a=1, slot_b=0, shift=(0, 0, 0), colour='y', label='R05'),
    dict(index=6, ma=6, mb=7, slot_a=2, slot_b=2, shift=(0, 0, 1), colour='x', label='R06'),
    dict(index=7, ma=7, mb=8, slot_a=1, slot_b=2, shift=(0, -1, 0), colour='y', label='R07'),
    dict(index=8, ma=8, mb=9, slot_a=1, slot_b=1, shift=(1, 0, 0), colour='z', label='R08'),
    dict(index=9, ma=9, mb=0, slot_a=0, slot_b=1, shift=(0, 0, 0), colour='y', label='R09'),
]

# the one free slot per ring manifold: inject here, extract here
BEAM_PORTS = [
    dict(manifold=0, slot=0),
    dict(manifold=1, slot=1),
    dict(manifold=2, slot=0),
    dict(manifold=3, slot=0),
    dict(manifold=4, slot=2),
    dict(manifold=5, slot=0),
    dict(manifold=6, slot=1),
    dict(manifold=7, slot=0),
    dict(manifold=8, slot=0),
    dict(manifold=9, slot=2),
]

TOTAL_QUBITS = ELL * QUBITS_PER_MANIFOLD          # 270
SHARED_QUBITS = ELL * QUBITS_PER_EDGE             # 90
PORT_QUBITS = len(BEAM_PORTS) * QUBITS_PER_EDGE   # 90


def junction_bonds(rec):
    """Global qubit identification for one junction: 9 pairs
    (manifold ma local qubit, manifold mb local qubit)."""
    qa = SLOT_QUBITS[rec["slot_a"]]
    qb = SLOT_QUBITS[rec["slot_b"]]
    return [(qa[i], qb[PAIRING[i]]) for i in range(QUBITS_PER_EDGE)]


def port_qubits(port):
    return SLOT_QUBITS[port["slot"]]



# =====================================================================
# GEOMETRY: THE STANDARD (ALBANESE) REALIZATION
# =====================================================================
# Combinatorics alone cannot see handedness -- the mirror ring is
# graph-isomorphic to the original, and every slot, pairing and colour
# comes out identical. A chirality probe needs POSITIONS.
#
# The standard realization of the maximal abelian cover: project each
# edge indicator onto the cycle space of K4 and read the result in an
# orthonormal basis of that 3-dimensional space. All six projections
# have norm^2 = 1/2, so every bond has the same length 1/sqrt(2) and
# the ring closes exactly. This is the geometry, not a drawing.

EDGE_VECTORS = {
    (0, 1): (0.000000000, 0.000000000, 0.707106781),
    (0, 2): (0.500000000, 0.353553391, -0.353553391),
    (0, 3): (-0.500000000, -0.353553391, -0.353553391),
    (1, 2): (0.166666667, -0.589255651, 0.353553391),
    (1, 3): (-0.166666667, 0.589255651, 0.353553391),
    (2, 3): (0.666666667, -0.235702260, 0.000000000),
}
BOND_LENGTH = 0.707106781

RING_POSITIONS = [
    (0.000000000, 0.000000000, 0.000000000),
    (-0.500000000, -0.353553391, -0.353553391),
    (-1.166666667, -0.117851130, -0.353553391),
    (-1.333333333, 0.471404521, -0.707106781),
    (-1.500000000, 1.060660172, -0.353553391),
    (-1.000000000, 1.414213562, 0.000000000),
    (-0.500000000, 1.767766953, -0.353553391),
    (0.166666667, 1.532064693, -0.353553391),
    (0.333333333, 0.942809042, -0.707106781),
    (0.500000000, 0.353553391, -0.353553391),
]
MIRROR_POSITIONS = [(x, y, -z) for (x, y, z) in RING_POSITIONS]

# discrete torsion, sum over the ring of det[e_r, e_r+1, e_r+2].
# A pseudoscalar: reflection sends it to minus itself.
TORSION = +0.5
MIRROR_TORSION = -0.5


# =====================================================================
# THE MIRROR CIRCUIT
# =====================================================================
# chirality = -1 negates every chord shift. The ring nodes reflect, the
# junction shifts negate, and NOTHING ELSE CHANGES -- same manifold
# indices, same slots, same pairing, same colour word, same holonomy.
#
# That is the whole point. The mirror is a KNOWN-NULL differential: run
# both circuits and every observable must agree bit for bit, because the
# two are related by a relabelling. Any divergence is engine error, and
# because the mirror moves which cell each manifold sits in, it stresses
# exactly the junction_pairing and slot-indexing paths where the bugs
# have historically been.
#
# The null test alone would be vacuous -- an engine that ignores
# geometry entirely passes it. TORSION is the paired must-DIFFER check.

MIRROR_NODES = [
    (0, (0, 0, 0)),
    (3, (0, 0, 0)),
    (2, (0, 0, 1)),
    (1, (1, 0, 1)),
    (3, (1, -1, 1)),
    (0, (1, -1, 1)),
    (2, (1, -1, 1)),
    (3, (1, -1, 0)),
    (1, (1, 0, 0)),
    (2, (0, 0, 0)),
]

MIRROR_JUNCTIONS = [
    dict(index=0, ma=0, mb=1, slot_a=2, slot_b=0, shift=(0, 0, 0), colour='z', label='M00'),
    dict(index=1, ma=1, mb=2, slot_a=2, slot_b=2, shift=(0, 0, 1), colour='x', label='M01'),
    dict(index=2, ma=2, mb=3, slot_a=1, slot_b=1, shift=(1, 0, 0), colour='z', label='M02'),
    dict(index=3, ma=3, mb=4, slot_a=2, slot_b=1, shift=(0, -1, 0), colour='y', label='M03'),
    dict(index=4, ma=4, mb=5, slot_a=0, slot_b=2, shift=(0, 0, 0), colour='z', label='M04'),
    dict(index=5, ma=5, mb=6, slot_a=1, slot_b=0, shift=(0, 0, 0), colour='y', label='M05'),
    dict(index=6, ma=6, mb=7, slot_a=2, slot_b=2, shift=(0, 0, -1), colour='x', label='M06'),
    dict(index=7, ma=7, mb=8, slot_a=1, slot_b=2, shift=(0, 1, 0), colour='y', label='M07'),
    dict(index=8, ma=8, mb=9, slot_a=1, slot_b=1, shift=(-1, 0, 0), colour='z', label='M08'),
    dict(index=9, ma=9, mb=0, slot_a=0, slot_b=1, shift=(0, 0, 0), colour='y', label='M09'),
]

MIRROR_PORTS = list(BEAM_PORTS)          # identical, by construction
ISOMORPHISM = list(range(ELL))           # manifold m <-> mirror m


# =====================================================================
# THE VERIFICATION LOOP
# =====================================================================
# Call `differential` every step with the two runs' observables. It
# returns (max_residual, per_key), and `verdict` turns that into a
# pass/fail against a tolerance. Nothing here needs the anchor, the
# Kitaev Hamiltonian, or a converged ground state -- it is a symmetry
# test, valid from step zero.
#
# NULL_KEYS   must agree to tolerance between run and mirror-run.
# The must-DIFFER side is the torsion, checked once at setup by
# `verify`, since geometry does not evolve.

NULL_TOL = 1e-10


def _flat(x):
    import numpy as np
    return np.asarray(x, dtype=float).ravel()


def differential(obs, obs_mirror, iso=None):
    """obs / obs_mirror: dict key -> per-manifold array or scalar.
    Returns (max_residual, {key: residual})."""
    import numpy as np
    iso = ISOMORPHISM if iso is None else iso
    per = {}
    for k in sorted(set(obs) & set(obs_mirror)):
        a = np.asarray(obs[k], dtype=float)
        b = np.asarray(obs_mirror[k], dtype=float)
        if a.shape != b.shape:
            per[k] = float("inf")
            continue
        if a.ndim >= 1 and a.shape[0] == ELL:
            b = b[list(iso)]
        per[k] = float(np.abs(a - b).max()) if a.size else 0.0
    missing = set(obs) ^ set(obs_mirror)
    for k in missing:
        per[k] = float("inf")
    return (max(per.values()) if per else 0.0), per


def verdict(obs, obs_mirror, tol=NULL_TOL, verbose=True):
    res, per = differential(obs, obs_mirror)
    if verbose:
        print("%-28s %14s  %s" % ("observable", "residual", "status"))
        print("-" * 60)
        for k in sorted(per):
            print("%-28s %14.3e  %s"
                  % (k, per[k], "ok" if per[k] <= tol else "DIVERGED"))
        print("-" * 60)
        print("max residual %.3e   tol %.1e   %s"
              % (res, tol, "PASS" if res <= tol else "FAIL"))
    return res <= tol, res, per


def self_test():
    """Synthetic loop: a clean run passes, a run with one junction's
    pairing corrupted fails, so the test is known to have teeth."""
    import numpy as np
    rng = np.random.default_rng(0)
    print("=" * 60)
    print("VERIFICATION LOOP SELF-TEST")
    print("=" * 60)
    base = {"energy": rng.normal(size=ELL),
            "leg_entropy": rng.random((ELL, 3)),
            "bp_residual": rng.random(ELL) * 1e-6}
    print("\nclean mirror run:")
    ok1, r1, _ = verdict(base, {k: v.copy() for k, v in base.items()})
    bad = {k: v.copy() for k, v in base.items()}
    bad["leg_entropy"][4] = bad["leg_entropy"][4][::-1]
    print("\nmirror run with manifold 4's legs permuted:")
    ok2, r2, _ = verdict(base, bad)
    print("\nself-test %s" % ("PASS" if (ok1 and not ok2) else "FAIL"))
    return ok1 and not ok2


# =====================================================================
# verify -- regenerate from K4-Crystalstacks and assert the literals
# =====================================================================
def verify():
    import collections
    import k4_crystalstacks as K4

    ok = True

    def check(name, got, want):
        global_ok = (got == want)
        print("  %-34s %s" % (name, "PASS" if global_ok else
                              "FAIL  got %s want %s" % (got, want)))
        return global_ok

    print("=" * 74)
    print("VERIFY RACETRACK RING AGAINST K4-Crystalstacks")
    print("=" * 74)

    ok &= check("junction_pairing(True)",
                K4.junction_pairing(True), PAIRING)
    ok &= check("pairing is an involution",
                [PAIRING[PAIRING[i]] for i in range(9)], list(range(9)))
    ok &= check("slot qubit blocks",
                [K4.edge_local_qubits(k) for k in range(3)], SLOT_QUBITS)

    def nbrs(node):
        v, cell = node
        return [(w, tuple(c + d for c, d in
                          zip(cell, K4.edge_shift(v, w))))
                for w in K4.NEIGHBOURS[v]]

    adj_ok = True
    for r in range(ELL):
        a, b = RING_NODES[r], RING_NODES[(r + 1) % ELL]
        if b not in nbrs(a):
            adj_ok = False
    ok &= check("ring is a closed walk on the crystal", adj_ok, True)
    ok &= check("ring nodes distinct", len(set(RING_NODES)), ELL)

    tot = [0, 0, 0]
    for r in range(ELL):
        a, b = RING_NODES[r], RING_NODES[(r + 1) % ELL]
        for i in range(3):
            tot[i] += b[1][i] - a[1][i]
    ok &= check("shift sum (contractible)", tuple(tot), RING_SHIFT_SUM)

    slot_ok, used = True, collections.defaultdict(list)
    for rec in RING_JUNCTIONS:
        a, b = RING_NODES[rec["ma"]], RING_NODES[rec["mb"]]
        if K4.junction_slot(a[0], b[0]) != rec["slot_a"]:
            slot_ok = False
        if K4.junction_slot(b[0], a[0]) != rec["slot_b"]:
            slot_ok = False
        used[rec["ma"]].append(rec["slot_a"])
        used[rec["mb"]].append(rec["slot_b"])
    ok &= check("slot indices match junction_slot()", slot_ok, True)
    ok &= check("every manifold uses exactly 2 of 3 slots",
                sorted(len(v) for v in used.values()), [2] * ELL)

    port_ok = True
    for p in BEAM_PORTS:
        free = [k for k in range(3) if k not in used[p["manifold"]]]
        if free != [p["slot"]]:
            port_ok = False
    ok &= check("beam ports are the free slots", port_ok, True)

    net = list(range(QUBITS_PER_EDGE))
    for _ in range(ELL):
        net = [PAIRING[x] for x in net]
    ok &= check("pairing holonomy around the ring",
                net, list(range(QUBITS_PER_EDGE)))

    word = "".join(rec["colour"] for rec in RING_JUNCTIONS)
    ok &= check("colour word", word, COLOUR_WORD)

    # ---- mirror circuit ----
    def nbrs_m(node):
        v, cell = node
        return [(w, tuple(c - d for c, d in
                          zip(cell, K4.edge_shift(v, w))))
                for w in K4.NEIGHBOURS[v]]

    ok &= check("mirror nodes are the reflected ring",
                MIRROR_NODES,
                [(v, tuple(-x for x in c)) for v, c in RING_NODES])
    ok &= check("mirror ring closed under chirality=-1",
                all(MIRROR_NODES[(r + 1) % ELL] in nbrs_m(MIRROR_NODES[r])
                    for r in range(ELL)), True)
    ok &= check("mirror slots identical (null)",
                [(r["slot_a"], r["slot_b"]) for r in MIRROR_JUNCTIONS],
                [(r["slot_a"], r["slot_b"]) for r in RING_JUNCTIONS])
    ok &= check("mirror colour word identical (null)",
                "".join(r["colour"] for r in MIRROR_JUNCTIONS), COLOUR_WORD)
    ok &= check("mirror shifts negated",
                [r["shift"] for r in MIRROR_JUNCTIONS],
                [tuple(-x for x in r["shift"]) for r in RING_JUNCTIONS])
    ok &= check("mirror bonds identical (null)",
                [junction_bonds(r) for r in MIRROR_JUNCTIONS],
                [junction_bonds(r) for r in RING_JUNCTIONS])

    # ---- geometry, the must-DIFFER side ----
    import numpy as np

    def torsion(Q):
        Q = np.asarray(Q, dtype=float)
        E = [Q[(r + 1) % ELL] - Q[r] for r in range(ELL)]
        return sum(float(np.linalg.det(np.array(
            [E[r], E[(r + 1) % ELL], E[(r + 2) % ELL]])))
            for r in range(ELL))

    Lg = [float(np.linalg.norm(np.array(RING_POSITIONS[(r + 1) % ELL])
                               - np.array(RING_POSITIONS[r])))
          for r in range(ELL)]
    ok &= check("all bonds equal length",
                max(abs(x - BOND_LENGTH) for x in Lg) < 1e-8, True)
    ok &= check("ring closes geometrically",
                abs(float(np.linalg.norm(
                    np.array(RING_POSITIONS[0])
                    - np.array(RING_POSITIONS[-1]))) - BOND_LENGTH)
                < 1e-8, True)
    ok &= check("torsion",
                abs(torsion(RING_POSITIONS) - TORSION) < 1e-8, True)
    ok &= check("mirror torsion FLIPS (not a null)",
                abs(torsion(MIRROR_POSITIONS) - MIRROR_TORSION) < 1e-8,
                True)
    print("      torsion %+0.9f   mirror %+0.9f   ratio %+0.6f"
          % (torsion(RING_POSITIONS), torsion(MIRROR_POSITIONS),
             torsion(MIRROR_POSITIONS) / torsion(RING_POSITIONS)))

    bonds = [junction_bonds(rec) for rec in RING_JUNCTIONS]
    ok &= check("bonds per junction",
                sorted(set(len(b) for b in bonds)), [QUBITS_PER_EDGE])

    print("")
    print("  manifolds %d x %d = %d qubits, %d shared, %d port"
          % (ELL, QUBITS_PER_MANIFOLD, TOTAL_QUBITS, SHARED_QUBITS,
             PORT_QUBITS))
    print("")
    print("VERDICT: %s" % ("ALL PASS" if ok else "FAILURES ABOVE"))
    return ok


if __name__ == "__main__":
    import sys
    a = sys.argv[1:] or ["all"]
    good = True
    if a[0] in ("verify", "all"):
        good &= verify()
        print("")
    if a[0] in ("selftest", "all"):
        good &= self_test()
    sys.exit(0 if good else 1)
