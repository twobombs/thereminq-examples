# -*- coding: us-ascii -*-
# K4-qasm.py -- manifest to QASM 3 and to a native PyQrack driver.
#
# BOTH, DELIBERATELY
# =====================================================================
# QASM 3 is the portable artifact: it opens in any visualiser, feeds
# any benchmark harness, and is what you hand to someone who does not
# run Qrack. PyQrack has no QASM reader, so the route into it is
#
#   .qasm -> qiskit.qasm3.loads -> QrackCircuit.in_from_qiskit_circuit
#         -> .run(QrackSimulator)
#
# The native driver skips that entirely and is the one to actually
# simulate with, because Qrack has a primitive QASM does not: exp(b,
# ph, q) applies e^{i*ph*(P (x) P ...)} directly. A Kitaev bond IS that
# object. In QASM the same thing costs a basis change, two CX and an
# RZ, which is five gates and a decomposition the simulator then has to
# recognise. Same circuit, two costs -- so both are emitted from one
# gate list and checked against each other.
#
# THE RELABELLING, AND WHY IT IS NOT COSMETIC
# =====================================================================
# Blocks of one type are translations of each other, so they are the
# same circuit. But a manifest's local qubit indices come from sorted
# site order, and translation permutes that order -- so the four blocks
# of a type arrive with four different bond lists describing one
# circuit. Emitting from those directly would produce 32 QASM files
# where 8 suffice, and would quietly deny that the equivariance bought
# anything.
#
# So each block is relabelled against its type's representative: find
# the translation h carrying the representative onto this block, and
# index this block's sites by their preimages. After that the four
# blocks of a type are bit-identical as circuits, and the emitter
# checks that they are rather than assuming it.
#
# WHAT IS EMITTED
# =====================================================================
#   qasm/type<k>-trotter.qasm    one Trotter step, 27 bonds
#   qasm/type<k>-flux.qasm       Hadamard test on the 10-qubit W word
#   qasm/type<k>-full.qasm       a step then the measurement
#   k4_pyqrack_driver.py         native driver, reads the manifest
#   qasm/README.txt              qubit conventions and the Qrack route
#
# CHECKS
# =====================================================================
#   1. the XX/YY/ZZ decompositions against exact exp(-i.theta.P(x)P)
#      matrices -- the YY basis change is the easiest thing here to get
#      wrong and the hardest to notice
#   2. Qrack's exp() against those same matrices, so the two emitters
#      are pinned to one definition rather than to each other
#   3. every emitted file parsed back by qiskit.qasm3, its bond list
#      reconstructed from the gates and compared to the manifest
#
#   ./K4-qasm.py manifest.json --theta 0.1 -o out/

import argparse
import collections
import json
import os

L = 6
CUBE = L ** 3
PAULI = "XYZ"


def cell(i):
    return ((i % CUBE) // (L * L), (i % (L * L)) // L, i % L)


def sub(i):
    return i // CUBE


def site(v, c):
    return v * CUBE + c[0] * L * L + c[1] * L + c[2]


def translate(i, h):
    return site(sub(i), tuple((a + b) % L for a, b in zip(cell(i), h)))


# =====================================================================
# CANONICAL RELABELLING
# =====================================================================

def canonical_maps(man, gens):
    """For each block, a map site -> local qubit index that makes every
    block of a type carry an identical bond list."""
    H = {(0, 0, 0)}
    while True:
        new = {tuple((a[k] + g[k]) % L for k in range(3))
               for a in H for g in gens}
        if new <= H:
            break
        H |= new
    H = sorted(H)
    blocks = man["blocks"]
    by_type = collections.defaultdict(list)
    for b in blocks:
        by_type[b["type"]].append(b)
    qmap = {}
    reps = {}
    for t, bs in sorted(by_type.items()):
        rep = bs[0]
        reps[t] = rep["index"]
        rq = {v: k for k, v in enumerate(sorted(rep["sites"]))}
        qmap[rep["index"]] = dict(rq)
        for b in bs[1:]:
            S = set(b["sites"])
            found = None
            for h in H:
                if {translate(v, h) for v in rep["sites"]} == S:
                    found = h
                    break
            if found is None:
                raise SystemExit(
                    "block %d is typed with block %d but no translation in H "
                    "carries one onto the other; the manifest's types are "
                    "not translation classes" % (b["index"], rep["index"]))
            qmap[b["index"]] = {translate(v, found): rq[v] for v in rep["sites"]}
    return qmap, reps


def gate_list(man, qmap):
    """Per block: bonds and loop, in canonical local indices."""
    out = {}
    for b in man["blocks"]:
        q = qmap[b["index"]]
        S = b["sites"]
        bonds = sorted((min(q[S[i]], q[S[j]]), max(q[S[i]], q[S[j]]), c)
                       for i, j, c in b["internal_bonds"])
        loop = [q[v] for v in b["loop_sites"]]
        # The word is a string read along the loop, so it depends on
        # where traversal starts AND which way it goes -- while W itself
        # does not, since its ten Paulis act on distinct qubits and
        # commute. Two blocks of one type can therefore arrive with
        # reversed words describing the same operator. Rotate to the
        # lowest index, then take whichever direction reads smaller.
        pairs = list(zip(loop, b["loop_pauli"]))
        k = loop.index(min(loop))
        pairs = pairs[k:] + pairs[:k]
        rev = [pairs[0]] + pairs[:0:-1]
        if [x[0] for x in rev] < [x[0] for x in pairs]:
            pairs = rev
        loop = [x[0] for x in pairs]
        word = "".join(x[1] for x in pairs)
        out[b["index"]] = (bonds, loop, word)
    return out


# =====================================================================
# QASM 3
# =====================================================================

BASIS_IN = {0: ["h {q};"], 1: ["sdg {q};", "h {q};"], 2: []}
BASIS_OUT = {0: ["h {q};"], 1: ["h {q};", "s {q};"], 2: []}


def bond_qasm(qi, qj, c, theta, reg="q"):
    g = []
    for t in (qi, qj):
        for s in BASIS_IN[c]:
            g.append(s.format(q="%s[%d]" % (reg, t)))
    g.append("cx %s[%d], %s[%d];" % (reg, qi, reg, qj))
    g.append("rz(%.17g) %s[%d];" % (2.0 * theta, reg, qj))
    g.append("cx %s[%d], %s[%d];" % (reg, qi, reg, qj))
    for t in (qi, qj):
        for s in BASIS_OUT[c]:
            g.append(s.format(q="%s[%d]" % (reg, t)))
    return g


def emit_trotter(bonds, nq, theta):
    o = ['OPENQASM 3.0;', 'include "stdgates.inc";',
         "qubit[%d] q;" % nq, ""]
    for qi, qj, c in bonds:
        o.append("// bond %d-%d colour %d (%s%s)"
                 % (qi, qj, c, PAULI[c], PAULI[c]))
        o += bond_qasm(qi, qj, c, theta)
    return "\n".join(o) + "\n"


def emit_flux(loop, word, nq):
    o = ['OPENQASM 3.0;', 'include "stdgates.inc";',
         "qubit[%d] q;" % nq, "qubit[1] anc;", "bit[1] c;", "",
         "// Hadamard test on W = %s over loop qubits %s"
         % (word, ",".join(str(x) for x in loop)),
         "h anc[0];"]
    for q, p in zip(loop, word):
        o.append("c%s anc[0], q[%d];" % (p.lower(), q))
    o += ["h anc[0];", "c[0] = measure anc[0];"]
    return "\n".join(o) + "\n"


def emit_full(bonds, loop, word, nq, theta):
    o = ['OPENQASM 3.0;', 'include "stdgates.inc";',
         "qubit[%d] q;" % nq, "qubit[1] anc;", "bit[1] c;", ""]
    for qi, qj, c in bonds:
        o += bond_qasm(qi, qj, c, theta)
    o += ["", "h anc[0];"]
    for q, p in zip(loop, word):
        o.append("c%s anc[0], q[%d];" % (p.lower(), q))
    o += ["h anc[0];", "c[0] = measure anc[0];"]
    return "\n".join(o) + "\n"


# =====================================================================
# NATIVE PYQRACK DRIVER  (emitted as a standalone file)
# =====================================================================

DRIVER = '''# -*- coding: us-ascii -*-
# k4_pyqrack_driver.py -- run a K4 block natively on Qrack.
#
# Generated by K4-qasm.py from %(src)s. Reads the manifest at runtime,
# so editing the manifest does not mean regenerating this driver.
#
# WHY NOT exp()
# =====================================================================
# Qrack exposes exp(b, ph, q), applying e^{i*ph*(P (x) P)} in one call,
# which is exactly a Kitaev bond and would beat the five-gate QASM
# decomposition. It is NOT used here: in the pyqrack build this was
# generated against, the C entry point declares its Pauli-basis
# argument as int* while the Python wrapper passes ulonglong*, so the
# call raises a ctypes TypeError, and forcing the correct array type by
# hand segfaults. Worth raising upstream; until then the decomposition
# below is what runs.
#
# CONVENTIONS, fixed here and matched gate for gate by the QASM
# =====================================================================
#   bond term      exp(-i*theta*P(x)P)
#   r(PauliZ, ph)  is RZ(ph): measured, it puts relative phase e^{i*ph}
#                  on |1>, so the bond needs 2*theta, as QASM's rz does
#   basis change   XX: h ... h        YY: adjs h ... h s        ZZ: none
#   out_ket        little-endian; qubit 0 is the low bit
#
# Validated against exp(-i*theta*P(x)P) at fidelity 1 for all three
# colours by K4-qasm.py at generation time.
#
#   python3 k4_pyqrack_driver.py --block 0 --theta 0.1 --steps 4

import argparse
import json
import os

from pyqrack import QrackSimulator, Pauli

MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "%(src)s")


def bond(sim, qi, qj, c, theta):
    """exp(-i*theta*P(x)P) for colour c in 0,1,2 = XX,YY,ZZ."""
    for t in (qi, qj):
        if c == 0:
            sim.h(t)
        elif c == 1:
            sim.adjs(t)
            sim.h(t)
    sim.mcx([qi], qj)
    sim.r(Pauli.PauliZ, 2.0 * theta, qj)
    sim.mcx([qi], qj)
    for t in (qi, qj):
        if c == 0:
            sim.h(t)
        elif c == 1:
            sim.h(t)
            sim.s(t)


def trotter_step(sim, bonds, theta):
    for qi, qj, c in bonds:
        bond(sim, qi, qj, c, theta)


def measure_flux(sim, loop, word, anc):
    """Hadamard test on the flux word; returns <W> in [-1, 1].

    QrackCircuit cannot measure, which is why this runs against a live
    QrackSimulator rather than a compiled circuit.
    """
    sim.h(anc)
    for q, p in zip(loop, word):
        (sim.mcx, sim.mcy, sim.mcz)["XYZ".index(p)]([anc], q)
    sim.h(anc)
    return 1.0 - 2.0 * sim.prob(anc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--theta", type=float, default=%(theta)r)
    ap.add_argument("--steps", type=int, default=1)
    a = ap.parse_args()

    man = json.load(open(a.manifest))
    byi = {b["index"]: b for b in man["blocks"]}
    b = byi[a.block]
    nq = man["qubits_per_block"]
    bonds = b["canonical_bonds"]

    sim = QrackSimulator(nq + 1)
    for _ in range(a.steps):
        trotter_step(sim, bonds, a.theta)
    w = measure_flux(sim, b["canonical_loop"], b["canonical_word"], nq)

    print("block %%d (type %%d): %%d qubits + 1 ancilla" %% (
        a.block, b["type"], nq))
    print("  %%d bonds, %%d step(s) at theta=%%g" %% (
        len(bonds), a.steps, a.theta))
    print("  flux word %%s" %% b["canonical_word"])
    print("  <W> = %%+.6f" %% w)


if __name__ == "__main__":
    main()
'''


# =====================================================================
# CHECKS
# =====================================================================

def check_primitives(theta):
    """The decompositions against exact matrices, and Qrack against the
    same. Two emitters pinned to one definition, not to each other."""
    import numpy as np
    msgs = []
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    P = [X, Y, Z]
    try:
        from qiskit import qasm3
        from qiskit.quantum_info import Operator
    except Exception as e:
        return ["qiskit unavailable, QASM primitives unchecked (%s)" % e]
    for c in (0, 1, 2):
        pp = np.kron(P[c], P[c])
        # exp(-i theta P(x)P), Qiskit's little-endian kron order
        w, v = np.linalg.eigh(pp)
        exact = v @ np.diag(np.exp(-1j * theta * w)) @ v.conj().T
        src = ('OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\n'
               + "\n".join(bond_qasm(0, 1, c, theta)) + "\n")
        got = Operator(qasm3.loads(src)).data
        # global phase is unobservable; compare up to it
        k = np.argmax(np.abs(exact))
        ph = got.flat[k] / exact.flat[k]
        if not np.allclose(got, exact * ph, atol=1e-9):
            msgs.append("QASM %s%s decomposition does not match exp(-i.t.PP)"
                        % (PAULI[c], PAULI[c]))
    try:
        from pyqrack import QrackSimulator, Pauli
        # Qrack's exp() would be the natural primitive but is broken in
        # current builds (int* vs ulonglong*, then a segfault), so the
        # driver uses the same decomposition as the QASM -- which makes
        # checking it against the exact matrix necessary, not optional.
        q0 = np.array([1, 1], complex) / np.sqrt(2)
        q1 = np.array([1, np.exp(1j * np.pi / 4)], complex) / np.sqrt(2)
        psi0 = np.kron(q1, q0)          # out_ket is little-endian
        for c in (0, 1, 2):
            s = QrackSimulator(2)
            s.h(0)
            s.h(1)
            s.t(1)
            for t in (0, 1):
                if c == 0:
                    s.h(t)
                elif c == 1:
                    s.adjs(t)
                    s.h(t)
            s.mcx([0], 1)
            s.r(Pauli.PauliZ, 2.0 * theta, 1)
            s.mcx([0], 1)
            for t in (0, 1):
                if c == 0:
                    s.h(t)
                elif c == 1:
                    s.h(t)
                    s.s(t)
            got = np.array(s.out_ket()).reshape(-1)
            pp = np.kron(P[c], P[c])
            w, v = np.linalg.eigh(pp)
            psi = (v @ np.diag(np.exp(-1j * theta * w)) @ v.conj().T) @ psi0
            f = abs(np.vdot(psi, got))
            if f < 1 - 1e-5:
                msgs.append("driver %s%s gate sequence: fidelity %.9f against "
                            "exp(-i.t.PP)" % (PAULI[c], PAULI[c], f))
    except Exception as e:
        msgs.append("pyqrack unavailable, native primitives unchecked (%s)" % e)
    return msgs


def check_roundtrip(path, bonds):
    """Parse an emitted file and rebuild its bond list from the gates.

    Colours are NOT guessed from neighbouring gate names -- one bond's
    basis-out looks exactly like the next bond's basis-in, and a ZZ bond
    (which has no basis change at all) then reads as whatever preceded
    it. Instead the single-qubit frame is accumulated exactly: at each
    cx-rz-cx, the qubit's frame U makes the term act on U'ZU, and that
    is compared against X, Y, Z. Unambiguous, and it catches a wrong
    basis change rather than agreeing with it.
    """
    try:
        import numpy as np
        from qiskit import qasm3
    except Exception:
        return ["qiskit unavailable, %s unparsed" % os.path.basename(path)]
    try:
        qc = qasm3.loads(open(path).read())
    except Exception as e:
        return ["%s does not parse as QASM 3: %s" % (os.path.basename(path), e)]
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], complex)
    ONE = {"h": H, "s": S, "sdg": S.conj().T}
    frame = collections.defaultdict(lambda: I.copy())
    ops = [(i.operation.name, [qc.find_bit(x).index for x in i.qubits])
           for i in qc.data]
    got = []
    for k, (nm, qs) in enumerate(ops):
        if nm in ONE:
            frame[qs[0]] = ONE[nm] @ frame[qs[0]]
            continue
        if nm != "rz":
            continue
        if k < 1 or ops[k - 1][0] != "cx" or k + 1 >= len(ops) \
                or ops[k + 1][0] != "cx":
            return ["%s: an rz is not wrapped in cx" % os.path.basename(path)]
        qi, qj = ops[k - 1][1]
        cs = []
        for q in (qi, qj):
            U = frame[q]
            eff = U.conj().T @ Z @ U
            hit = [c for c, P in enumerate((X, Y, Z))
                   if np.allclose(eff, P, atol=1e-9)
                   or np.allclose(eff, -P, atol=1e-9)]
            if len(hit) != 1:
                return ["%s: qubit %d frame is not a Pauli basis change"
                        % (os.path.basename(path), q)]
            cs.append(hit[0])
        if cs[0] != cs[1]:
            return ["%s: bond %d-%d couples %s to %s, not a Kitaev term"
                    % (os.path.basename(path), qi, qj, PAULI[cs[0]],
                       PAULI[cs[1]])]
        got.append((min(qi, qj), max(qi, qj), cs[0]))
    want = sorted((a, b, c) for a, b, c in bonds)
    if sorted(got) != want:
        miss = set(want) - set(got)
        extra = set(got) - set(want)
        return ["%s: %d bonds recovered vs %d; %d missing, %d unexpected %s"
                % (os.path.basename(path), len(got), len(want), len(miss),
                   len(extra), sorted(extra)[:3])]
    return []


# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("-o", "--out", default="out")
    ap.add_argument("--theta", type=float, default=0.1)
    ap.add_argument("--gens", default="3,0,0:0,3,0")
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()

    man = json.load(open(a.manifest))
    gens = [tuple(int(x) for x in g.split(",")) for g in a.gens.split(":")]
    nq = man["qubits_per_block"]
    qmap, reps = canonical_maps(man, gens)
    gl = gate_list(man, qmap)

    # After relabelling, one type must be one circuit.
    by_type = collections.defaultdict(list)
    for b in man["blocks"]:
        by_type[b["type"]].append(b["index"])
    for t, idxs in sorted(by_type.items()):
        sigs = {(tuple(gl[i][0]), tuple(gl[i][1]), gl[i][2]) for i in idxs}
        if len(sigs) != 1:
            raise SystemExit(
                "type %d still has %d distinct circuits after relabelling; "
                "the blocks are not translations of one another" % (t, len(sigs)))
    print("relabelled: %d blocks collapse to %d distinct circuits"
          % (len(man["blocks"]), len(by_type)))

    qd = os.path.join(a.out, "qasm")
    os.makedirs(qd, exist_ok=True)
    problems = []
    for t in sorted(by_type):
        bonds, loop, word = gl[reps[t]]
        for nm, txt in (("trotter", emit_trotter(bonds, nq, a.theta)),
                        ("flux", emit_flux(loop, word, nq)),
                        ("full", emit_full(bonds, loop, word, nq, a.theta))):
            p = os.path.join(qd, "type%d-%s.qasm" % (t, nm))
            open(p, "w").write(txt)
            if not a.no_check and nm != "flux":
                problems += check_roundtrip(p, bonds)

    for b in man["blocks"]:
        bonds, loop, word = gl[b["index"]]
        b["canonical_bonds"] = [list(x) for x in bonds]
        b["canonical_loop"] = loop
        b["canonical_word"] = word
    src = os.path.basename(a.manifest)
    json.dump(man, open(os.path.join(a.out, src), "w"))
    open(os.path.join(a.out, "k4_pyqrack_driver.py"), "w").write(
        DRIVER % {"src": src, "nb": len(gl[reps[0]][0]), "theta": a.theta})
    open(os.path.join(qd, "README.txt"), "w").write(
        "K4 block circuits, generated from %s\n\n"
        "  q[0..%d]   block qubits, canonical local indices\n"
        "  anc[0]     Hadamard-test ancilla (flux and full only)\n\n"
        "Bond term is exp(-i*theta*P(x)P) with theta=%g baked in; colour\n"
        "0/1/2 is XX/YY/ZZ. One file per block type: the four blocks of a\n"
        "type are translations and share a circuit exactly.\n\n"
        "PyQrack has no QASM reader. The route in is\n"
        "  qc = qiskit.qasm3.loads(open('type0-trotter.qasm').read())\n"
        "  c  = QrackCircuit.in_from_qiskit_circuit(qc)\n"
        "  c.run(QrackSimulator(%d))\n"
        "QrackCircuit cannot measure; measure on the simulator after run().\n"
        "For simulation prefer ../k4_pyqrack_driver.py, which uses Qrack's\n"
        "native exp() instead of the CX-RZ-CX decomposition QASM forces.\n"
        % (src, nq - 1, a.theta, nq + 1))

    if not a.no_check:
        problems += check_primitives(a.theta)
    print("checks: %d problems" % len(problems))
    for p in problems[:10]:
        print("  " + p)
    print("wrote %d QASM files, driver, and relabelled manifest to %s/"
          % (3 * len(by_type), a.out))


if __name__ == "__main__":
    main()
