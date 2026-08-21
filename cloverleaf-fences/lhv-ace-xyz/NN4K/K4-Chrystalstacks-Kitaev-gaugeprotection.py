# -*- coding: us-ascii -*-
# flux_diag.py -- Rev 2.
#
# eps_W: the flux-sector analogue of Andrade et al.'s gauge-violation
# scalar <eps_G> = sum_j <G_j>^2 / N_g (arXiv:2608.02756, Fig. 2), for
# Kitaev on the hyperoctagon / (10,3)-a / K4 crystal.
#
# WHAT CHANGED IN REV 2
# =====================================================================
# Rev 1 shipped the stabiliser table as JSON. Gone. There is no file
# handoff and no parser: the engine imports this module, builds a
# FluxMonitor once, and calls it per sweep. Nothing is serialised in
# the loop.
#
#     import flux_diag
#     mon = flux_diag.FluxMonitor(L=4, block_of=my_site_to_manifold)
#     w = {key: engine.expectation(ops)              # your contractor
#          for key, ops in mon.targets(site_to_qubit)}
#     print(mon.line(w))                             # one log line
#
# `mon.targets` hands back (key, {qubit: pauli}) in the ENGINE's own
# qubit indexing. `pauli_expectation` is provided if the engine holds
# a statevector and has nothing better. For a build that must not
# import the anchor at all, `emit_table` writes a standalone .py
# holding the same table as literals -- Python, not data.
#
# WHY THIS IS A USABLE DIAGNOSTIC AND b_r IS NOT A SUBSTITUTE
# =====================================================================
# The racetrack amplitudes b_r are gauge invariant bond by bond and
# are the sharper benchmark, but they are STATE amplitudes: they mix
# "the flux sector drifted" with "the sector is right and the
# amplitudes are wrong". eps_W separates those. It is one scalar per
# sweep, it is a stabiliser expectation so it is basis- and
# convention-free, and its exact target is known: -1 on every
# elementary 10-loop, because ell = 10 = 2 mod 4 makes W = -1 the
# Lieb-optimal zero-flux value and the Wilson twists do not disturb it.
#
# WHAT THE ENGINE HAS TO MEASURE
# =====================================================================
#     W_p = prod_{r=1..ell} sigma^{gamma_r}_{j_r}
#
# with gamma_r the DANGLING colour at loop site j_r -- the one of the
# three bond colours at j_r that the loop does not use. Always defined
# on a tricoordinated lattice, since the two loop bonds at a site have
# distinct colours.
#
# SIGN RULE (derived, then pinned by ED -- see self_test)
# =====================================================================
# In the Majorana representation sigma^a = i b^a c with D_j = 1,
#
#     W_p = (-1)^ell * (prod_r eps_r) * prod_r u_{j_r, j_{r+1}}
#
# where eps_r = +1 if the bond colour ADVANCES cyclically at site j_r
# (alpha_r = alpha_{r-1} + 1 mod 3) and -1 if it retreats. So the map
# from the anchor's `loop_flux` (a u-product) to the spin stabiliser is
# loop dependent, not a global constant:
#
#     <W_p>_spin = s_p * loop_flux(p),  s_p = (-1)^(ell + n_ret + 1)
#
# For ell = 10 the closure condition n_adv - n_ret = 0 mod 3 allows
# n_ret in {2, 5, 8}, so s_p is NOT automatically uniform. On the real
# lattice at L >= 4 it is: every elementary 10-loop has n_ret = 5, so
# s_p = +1 and the target is <W_p> = -1 throughout. At L = 2 the torus
# is too small and the "10-loops" wrap; s_p is mixed there. L >= 4 is
# enforced, not advised.
#
# HOW TO READ THE TWO NUMBERS
# =====================================================================
#   eps_W_intra  -- loops wholly inside a manifold. Contracted without
#                   crossing a message bond. Nonzero here means the
#                   block state itself is wrong: truncation, chi, the
#                   junction tensor. BP is not implicated.
#   eps_W_inter  -- loops crossing a manifold boundary. These are the
#                   ONLY ones that route through _outgoing. Nonzero
#                   here with eps_W_intra ~ 0 is a boundary-semantics
#                   signature: a copy tensor carries the stabiliser
#                   through a shared index, a virtual bond does not,
#                   and the flux leaks.
#
# A single scalar cannot tell a bond-semantics bug from a rank
# truncation bug; the pair can. Caveat both ways: the intra channel
# holds only ONE loop per 27-block (27 = 10-cycle + 17 tree sites), so
# it is a coarse pass/fail, and a straddling W_p is measured through
# the very message passing under test, so a dirty inter channel is
# consistent with a bond fault rather than proof of one. `suspects`
# is the discriminator: a semantics fault pushes every loop through a
# given junction the same way, truncation noise is diffuse.
#
# numpy only. Imports the anchor as `k4kitaev`.

import sys
import collections
import itertools
import numpy as np

import k4kitaev as K


# =====================================================================
# STABILISERS
# =====================================================================
def _colour_map(bonds):
    cm = {}
    for b, (i, j, c) in enumerate(bonds):
        cm[(i, j)] = c
        cm[(j, i)] = c
    return cm


def loop_sign(path, colmap):
    """s_p with <W_p>_spin = s_p * (u-product around the loop)."""
    n = len(path)
    al = [colmap[(path[r], path[(r + 1) % n])] for r in range(n)]
    ret = sum(1 for r in range(n) if (al[r] - al[(r - 1) % n]) % 3 == 2)
    return +1 if (n + ret) % 2 else -1


def pauli_string(path, colmap):
    """[(site, colour)] -- the dangling colour at each loop site."""
    n = len(path)
    out = []
    for r, site in enumerate(path):
        c1 = colmap[(site, path[(r - 1) % n])]
        c2 = colmap[(site, path[(r + 1) % n])]
        if c1 == c2:
            raise ValueError("loop reuses a colour at site %d" % site)
        out.append((site, 3 - c1 - c2))
    return out


def stabilizers(L=4, ell=10, twist=None):
    """One record per elementary loop:
         key     frozenset of loop sites (order-free handle)
         path    traversal order
         pauli   [(site, colour)] -- what the engine measures
         word    colour word of the loop bonds, for cross-checking
         sign    s_p, the loop-dependent u-product to spin map
         target  exact <W_p> in the anchor ground sector
    """
    if L < 4:
        raise ValueError("L=%d wraps the torus; s_p is not uniform and "
                         "the targets are meaningless. Use L >= 4." % L)
    uniq, eid, bonds, idx = K.elementary_loops(L, ell)
    colmap = _colour_map(bonds)
    u = K.wilson_u(L, K.GROUND_TWIST if twist is None else twist)
    out = []
    for path in uniq.values():
        s = loop_sign(path, colmap)
        out.append({
            "key": frozenset(path),
            "path": tuple(path),
            "pauli": pauli_string(path, colmap),
            "word": "".join(K.COLOR_NAME[colmap[(path[r],
                                                 path[(r + 1) % ell])]]
                            for r in range(ell)),
            "sign": s,
            "target": float(s * K.loop_flux(path, u, eid)),
        })
    return out, (u, eid, bonds, idx, colmap)


def w_from_u(stabs, u, eid):
    """Exact <W_p> for any Z2 gauge field. Used to seed known errors."""
    return {st["key"]: st["sign"] * K.loop_flux(st["path"], u, eid)
            for st in stabs}


# =====================================================================
# THE SCALAR
# =====================================================================
def eps_W(stabs, measured, keys=None):
    """mean over loops of ((<W> - target) / 2)^2.

    0.0 = exactly in sector. 1.0 = every loop maximally wrong
    (sign-flipped). A single flipped bond flips the ell loops through
    it, so on the L=4 lattice one bad bond reads 10/384 = 0.026.
    """
    sel = [st for st in stabs if keys is None or st["key"] in keys]
    if not sel:
        return float("nan"), 0
    acc = 0.0
    for st in sel:
        w = measured.get(st["key"])
        if w is None:
            raise KeyError("no <W> supplied for loop %s" % (st["path"],))
        acc += ((float(w) - st["target"]) / 2.0) ** 2
    return acc / len(sel), len(sel)


def eps_W_split(stabs, measured, block_of):
    """intra / inter eps_W given a site -> manifold-id map.

    block_of may omit sites; anything unmapped counts as its own block,
    so a partial map still classifies correctly.
    """
    intra, inter = set(), set()
    for st in stabs:
        blocks = {block_of.get(s, ("_", s)) for s in st["path"]}
        (intra if len(blocks) == 1 else inter).add(st["key"])
    ei, ni = eps_W(stabs, measured, intra)
    eo, no = eps_W(stabs, measured, inter)
    return {"intra": ei, "n_intra": ni, "inter": eo, "n_inter": no}


def worst_bonds(stabs, measured, top=8):
    """Localise. Score each bond by the mean violation of the loops
    through it; a single bad bond lights up its own ell loops and
    nothing else, so the top of this list IS the suspect bond."""
    score = collections.defaultdict(list)
    for st in stabs:
        v = ((float(measured[st["key"]]) - st["target"]) / 2.0) ** 2
        p = st["path"]
        for r in range(len(p)):
            a, b = p[r], p[(r + 1) % len(p)]
            score[(min(a, b), max(a, b))].append(v)
    return sorted(((float(np.mean(v)), len(v), k)
                   for k, v in score.items()), reverse=True)[:top]


# =====================================================================
# BLOCK GROWTH (mirrors export_racetrack, so the block is the same one)
# =====================================================================
def grow_block(L=4, B=27, ell=10):
    uniq, eid, bonds, idx = K.elementary_loops(L, ell)
    adj = collections.defaultdict(list)
    for b, (i, j, c) in enumerate(bonds):
        adj[i].append(j)
        adj[j].append(i)
    loop = list(uniq.values())[0]
    cur = set(loop)
    while len(cur) < B:
        fr = [w for v in cur for w in adj[v]
              if w not in cur and sum(1 for x in adj[w] if x in cur) == 1]
        if not fr:
            break
        cur.add(sorted(fr)[0])
    return sorted(cur), tuple(loop)


# =====================================================================
# MEASUREMENT HELPER
# =====================================================================
_PAULI = {0: np.array([[0, 1], [1, 0]], dtype=complex),
          1: np.array([[0, -1j], [1j, 0]], dtype=complex),
          2: np.array([[1, 0], [0, -1]], dtype=complex)}


def pauli_expectation(psi, ops, n_qubits=None):
    """<psi| prod_q sigma^{ops[q]}_q |psi> without forming the operator.

    psi: complex array of length 2**n, qubit 0 the most significant
    axis (numpy reshape order, i.e. np.kron order).
    ops: {qubit_index: colour} with colour in {0,1,2} = x,y,z.

    Applies one 2x2 at a time along its own axis, so the peak cost is
    one copy of the state. At 27 qubits that is 2 GB in complex128 and
    1 GB in complex64 -- cast first if the engine can afford it, the
    stabiliser is a sign so it can.
    """
    n = int(round(np.log2(psi.size))) if n_qubits is None else n_qubits
    if 2 ** n != psi.size:
        raise ValueError("psi has %d amplitudes, not 2**%d"
                         % (psi.size, n))
    phi = psi.reshape((2,) * n)
    for q, col in sorted(ops.items()):
        phi = np.moveaxis(np.tensordot(_PAULI[col], phi,
                                       axes=([1], [q])), 0, q)
    return complex(np.vdot(psi.reshape(-1), phi.reshape(-1)))


# =====================================================================
# THE OBJECT THE ENGINE ACTUALLY HOLDS
# =====================================================================
class FluxMonitor(object):
    """Build once, call per sweep. No serialisation anywhere.

        mon = FluxMonitor(L=4, block_of=site_to_manifold)
        w = {key: engine.expectation(ops)
             for key, ops in mon.targets(site_to_qubit)}
        print(mon.line(w))
    """

    def __init__(self, L=4, ell=10, twist=None, block_of=None):
        self.L, self.ell = L, ell
        self.stabs, aux = stabilizers(L, ell, twist)
        self.u, self.eid, self.bonds, self.idx, self.colmap = aux
        self.by_key = {st["key"]: st for st in self.stabs}
        self.block_of = dict(block_of or {})
        self.history = []
        self._last = {}

    # -- what to measure -------------------------------------------
    def targets(self, site_to_qubit=None, keys=None):
        """[(key, {qubit: colour})] in the engine's own indexing.
        Loops touching a site absent from the map are dropped, so a
        single-manifold map yields exactly that manifold's loops."""
        out = []
        for st in self.stabs:
            if keys is not None and st["key"] not in keys:
                continue
            if site_to_qubit is None:
                out.append((st["key"], {s: c for s, c in st["pauli"]}))
            elif all(s in site_to_qubit for s, _ in st["pauli"]):
                out.append((st["key"],
                            {site_to_qubit[s]: c for s, c in st["pauli"]}))
        return out

    def intra_keys(self):
        return {st["key"] for st in self.stabs
                if len({self.block_of.get(s, ("_", s))
                        for s in st["path"]}) == 1}

    def target_of(self, key):
        return self.by_key[key]["target"]

    # -- what came back --------------------------------------------
    def update(self, measured, label=None):
        w = {k: (v.real if isinstance(v, complex) else float(v))
             for k, v in measured.items()}
        seen = [st for st in self.stabs if st["key"] in w]
        total, n = eps_W(seen, w)
        rec = {"label": label, "n": n, "eps_W": total}
        if self.block_of:
            rec.update(eps_W_split(seen, w, self.block_of))
        rec["worst"] = max((abs(w[st["key"]] - st["target"]), st["path"])
                           for st in seen)
        self.history.append(rec)
        self._last = w
        return rec

    def suspects(self, top=6):
        return worst_bonds([st for st in self.stabs
                            if st["key"] in self._last], self._last, top)

    def line(self, measured=None, label=None):
        r = self.update(measured, label) if measured is not None \
            else self.history[-1]
        s = "eps_W %.3e (%d)" % (r["eps_W"], r["n"])
        if "intra" in r:
            s += "  intra %.3e (%d)  inter %.3e (%d)" % (
                r["intra"], r["n_intra"], r["inter"], r["n_inter"])
        s += "  worst |dW| %.3f" % r["worst"][0]
        return (("[%s] " % r["label"]) if r["label"] else "") + s

    def drift(self):
        """eps_W per sweep. Smooth growth is leakage; a step is a bug
        that switched on."""
        return [r["eps_W"] for r in self.history]


# =====================================================================
# STANDALONE TABLE (Python source, for a build with no anchor import)
# =====================================================================
def emit_table(path="flux_table.py", L=4, ell=10, B=27):
    """Write the stabilisers as Python literals plus a self-contained
    eps_W. No JSON, no parser, no runtime dependency on this module or
    on k4kitaev -- `import flux_table; flux_table.eps_W(w)`."""
    stabs, _ = stabilizers(L, ell)
    blk, loop = grow_block(L, B, ell)
    inb = set(blk)
    rows = ["    (%s, %s, %+d.0, %s)," %
            (tuple(st["path"]),
             [(s, K.COLOR_NAME[c]) for s, c in st["pauli"]],
             st["target"], set(st["path"]) <= inb)
            for st in stabs]
    src = [
        "# -*- coding: us-ascii -*-",
        "# GENERATED by flux_diag.emit_table -- do not hand edit.",
        "# Kitaev (10,3)-a flux stabilisers, L=%d, ell=%d." % (L, ell),
        "# Row: (loop_sites, [(site, pauli)], target_W, inside_block)",
        "",
        "BLOCK_SITES = %s" % (blk,),
        "RACETRACK = %s" % (list(loop),),
        "",
        "LOOPS = [",
    ] + rows + [
        "]",
        "",
        "",
        "def eps_W(measured, only_inside=None):",
        "    \"\"\"measured: {frozenset(loop_sites): <W>}.\"\"\"",
        "    acc = 0.0",
        "    n = 0",
        "    for sites, _pauli, tgt, inside in LOOPS:",
        "        if only_inside is not None and inside != only_inside:",
        "            continue",
        "        w = measured.get(frozenset(sites))",
        "        if w is None:",
        "            continue",
        "        acc += ((float(w) - tgt) / 2.0) ** 2",
        "        n += 1",
        "    return (acc / n if n else float('nan')), n",
        "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(src) + "\n")
    print("written: %s  (%d loops, %d wholly inside the %d-block)"
          % (path, len(stabs),
             sum(1 for st in stabs if set(st["path"]) <= inb), B))
    return path


# =====================================================================
# SELF TEST
# =====================================================================
def _ring_ed(alpha):
    """Exact-diagonalise an ell-site ring whose bond colours are
    `alpha`, and return the sign s with <W>_spin = s * prod(u).
    Independent of the lattice: a pure test of the colour rule."""
    n = len(alpha)
    gam = [3 - alpha[(r - 1) % n] - alpha[r] for r in range(n)]

    def kr(ops):
        M = np.array([[1.0 + 0j]])
        for i in range(n):
            M = np.kron(M, ops.get(i, np.eye(2)))
        return M

    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for r in range(n):
        H -= kr({r: _PAULI[alpha[r]], (r + 1) % n: _PAULI[alpha[r]]})
    W = kr({r: _PAULI[gam[r]] for r in range(n)})
    assert np.linalg.norm(H @ W - W @ H) < 1e-9
    e = {}
    for s in (+1, -1):
        Pr = 0.5 * (np.eye(2 ** n) + s * W)
        e[s] = np.linalg.eigvalsh(Pr @ H @ Pr
                                  + 1e6 * (np.eye(2 ** n) - Pr))[0]
    ff = {}
    for sg in itertools.product((1, -1), repeat=n):
        A = np.zeros((n, n))
        for r in range(n):
            A[r, (r + 1) % n] += 2.0 * sg[r]
            A[(r + 1) % n, r] -= 2.0 * sg[r]
        f = int(np.prod(sg))
        ff[f] = min(ff.get(f, 1e9), K.free_fermion_energy(A))
    return +1 if abs(e[+1] - ff[+1]) < 1e-8 else -1


def self_test(L=4, ell=10):
    ok = True
    print("=" * 74)
    print("SELF TEST 1 -- SIGN RULE vs RING ED")
    print("=" * 74)
    cases = [[0, 1, 0, 2], [0, 1, 0, 1], [0, 1, 2, 0, 1, 2],
             [0, 1, 0, 1, 0, 1], [0, 1, 2, 0, 1, 2, 0, 2],
             [0, 1] * 5, [0, 1, 2, 0, 1, 2, 0, 1, 2, 1]]
    uniq, eid, bonds, idx = K.elementary_loops(L, ell)
    colmap = _colour_map(bonds)
    p0 = list(uniq.values())[0]
    cases.append([colmap[(p0[r], p0[(r + 1) % ell])] for r in range(ell)])
    print("%-26s %4s %8s %8s" % ("colour word", "ell", "ED", "rule"))
    print("-" * 74)
    for al in cases:
        n = len(al)
        ret = sum(1 for r in range(n)
                  if (al[r] - al[(r - 1) % n]) % 3 == 2)
        rule = +1 if (n + ret) % 2 else -1
        ed = _ring_ed(al)
        ok = ok and (ed == rule)
        print("%-26s %4d %8d %8d %s"
              % ("".join(K.COLOR_NAME[a] for a in al), n, ed, rule,
                 "" if ed == rule else "  <-- MISMATCH"))
    print("-" * 74)
    print("VERDICT: %s\n" % ("PASS" if ok else "FAIL"))

    print("=" * 74)
    print("SELF TEST 2 -- ANCHOR SITS AT eps_W = 0")
    print("=" * 74)
    blk, loop = grow_block(L, 27, ell)
    mon = FluxMonitor(L=L, ell=ell, block_of={s: 0 for s in blk})
    print("L=%d  sites %d  bonds %d  elementary %d-loops %d"
          % (L, len(mon.idx), len(mon.bonds), ell, len(mon.stabs)))
    print("loop signs s_p : %s"
          % dict(collections.Counter(st["sign"] for st in mon.stabs)))
    print("target <W_p>   : %s"
          % dict(collections.Counter(int(round(st["target"]))
                                     for st in mon.stabs)))
    clean = w_from_u(mon.stabs, mon.u, mon.eid)
    print("%s" % mon.line(clean, "clean"))
    ok2 = mon.history[-1]["eps_W"] < 1e-24
    ok = ok and ok2
    print("VERDICT: %s\n" % ("PASS" if ok2 else "FAIL"))

    print("=" * 74)
    print("SELF TEST 3 -- SEEDED SINGLE-BOND ERROR")
    print("=" * 74)
    bad = 17
    ub = mon.u.copy()
    ub[bad] = -ub[bad]
    meas = w_from_u(mon.stabs, ub, mon.eid)
    print("flipped u on bond %d (%s colour, sites %d-%d)"
          % (bad, K.COLOR_NAME[mon.bonds[bad][2]],
             mon.bonds[bad][0], mon.bonds[bad][1]))
    print("%s" % mon.line(meas, "1 bad bond"))
    flipped = sum(1 for st in mon.stabs
                  if abs(meas[st["key"]] - st["target"]) > 1e-9)
    print("loops flipped  : %d of %d   (expect ell = %d)"
          % (flipped, len(mon.stabs), ell))
    print("eps_W expected : %d/%d = %.6f"
          % (ell, len(mon.stabs), ell / float(len(mon.stabs))))
    seed_key = (min(mon.bonds[bad][0], mon.bonds[bad][1]),
                max(mon.bonds[bad][0], mon.bonds[bad][1]))
    for sc, nl, k in mon.suspects(top=4):
        print("    %-14s loops %2d  score %.4f%s"
              % (str(k), nl, sc,
                 "  <-- the seeded bond" if k == seed_key else ""))
    ok3 = (flipped == ell
           and abs(mon.history[-1]["eps_W"]
                   - ell / float(len(mon.stabs))) < 1e-12
           and mon.suspects(top=1)[0][2] == seed_key)
    ok = ok and ok3
    print("VERDICT: %s\n" % ("PASS" if ok3 else "FAIL"))

    print("=" * 74)
    print("SELF TEST 4 -- INTRA vs INTER SPLIT ON A 27-SITE MANIFOLD")
    print("=" * 74)
    print("block: %d sites, racetrack %s" % (len(blk), list(loop)))
    print("loops wholly inside : %d" % len(mon.intra_keys()))
    lset = {(min(loop[r], loop[(r + 1) % ell]),
             max(loop[r], loop[(r + 1) % ell])) for r in range(ell)}
    inb = [b for b, (i, j, c) in enumerate(mon.bonds)
           if (min(i, j), max(i, j)) in lset]
    outb = [b for b, (i, j, c) in enumerate(mon.bonds)
            if (i in set(blk)) != (j in set(blk))]
    for name, blist in (("racetrack bond", inb), ("border bond", outb)):
        ue = mon.u.copy()
        ue[blist[0]] = -ue[blist[0]]
        print("%s" % mon.line(w_from_u(mon.stabs, ue, mon.eid), name))
    print("")

    print("=" * 74)
    print("SELF TEST 5 -- targets() AND pauli_expectation()")
    print("=" * 74)
    q = {s: i for i, s in enumerate(blk)}
    tg = mon.targets(q)
    print("loops measurable inside the block: %d" % len(tg))
    key, ops = tg[0]
    print("racetrack stabiliser : %s"
          % " ".join("%s%d" % (K.COLOR_NAME[c], b)
                     for b, c in sorted(ops.items())))
    print("target <W_p>         : %+.1f" % mon.target_of(key))
    rng = np.random.default_rng(0)
    n = 8
    psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    psi /= np.linalg.norm(psi)
    toy = {0: 0, 3: 1, 5: 2, 7: 0}
    M = np.array([[1.0 + 0j]])
    for i in range(n):
        M = np.kron(M, _PAULI[toy[i]] if i in toy else np.eye(2))
    a = pauli_expectation(psi, toy, n)
    b = complex(np.vdot(psi, M @ psi))
    print("sparse vs dense kron : %.3e" % abs(a - b))
    ok5 = abs(a - b) < 1e-12 and len(tg) == 1
    ok = ok and ok5
    print("VERDICT: %s\n" % ("PASS" if ok5 else "FAIL"))

    print("ALL: %s" % ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "table":
        emit_table(sys.argv[2] if len(sys.argv) > 2 else "flux_table.py")
    else:
        self_test()
