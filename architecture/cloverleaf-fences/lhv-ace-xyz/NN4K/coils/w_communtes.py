# w_commutes.py [MANIFEST.json | typeN-full.qasm ...] -- does each block's
# flux word W commute with every bond term of that block?
#
# A .qasm argument is read directly: bonds from each cx-rz-cx and its
# basis frame, the word from the Hadamard test -- no manifest needed.
#
# Needs the RELABELLED manifest K4-qasm.py writes into its -o directory
# (it adds canonical_bonds / canonical_loop / canonical_word). With no
# argument, every such manifest in the current directory is checked.
#
# P_c (x) P_c anticommutes with W at each loop site whose Pauli differs
# from P_c; the term commutes with W iff that count is even.
import glob, json, sys

KEYS = ("canonical_bonds", "canonical_loop", "canonical_word")


def relabelled(path):
    try:
        man = json.load(open(path))
        return man if all(k in man["blocks"][0] for k in KEYS) else None
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def check(path, man):
    print("== %s" % path)
    bad_total = 0
    for b in man["blocks"]:
        pauli_at = dict(zip(b["canonical_loop"], b["canonical_word"]))
        bad = [(i, j, "XYZ"[c]) for i, j, c in b["canonical_bonds"]
               if sum(1 for k in (i, j)
                      if k in pauli_at and pauli_at[k] != "XYZ"[c]) % 2]
        bad_total += len(bad)
        print("block %d (type %d): %d/%d bonds commute with W=%s%s" % (
            b["index"], b["type"], len(b["canonical_bonds"]) - len(bad),
            len(b["canonical_bonds"]), b["canonical_word"],
            "" if not bad else "  anticommuting: %s" % bad[:4]))
    print("W conserved by every Trotter step" if not bad_total else
          "W NOT conserved: %d anticommuting bond terms" % bad_total)
    return bad_total


def from_qasm(path):
    """Recover bonds and flux word from a K4-qasm.py 'full' circuit.

    Bond colour is read from each qubit's accumulated single-qubit frame
    at every cx-rz-cx, exactly as K4-qasm.py's own round-trip check does:
    the term acts on U'ZU, which is compared with X, Y, Z. The word is
    read from the Hadamard test's controlled Paulis on the ancilla.
    """
    import re
    import numpy as np
    H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    S = np.diag([1, 1j])
    ONE = {"h": H, "s": S, "sdg": S.conj().T}
    P = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]),
         np.diag([1, -1]).astype(complex)]
    ops = []
    src = re.sub(r"//[^\n]*", "", open(path).read())      # drop comments
    for st in src.replace("\n", " ").split(";"):
        st = st.strip()
        m = re.match(r"(h|s|sdg)\s+q\[(\d+)\]$", st)
        if m:
            ops.append((m.group(1), (int(m.group(2)),)))
            continue
        m = re.match(r"cx\s+q\[(\d+)\],\s*q\[(\d+)\]$", st)
        if m:
            ops.append(("cx", (int(m.group(1)), int(m.group(2)))))
            continue
        m = re.match(r"rz\([^)]*\)\s+q\[(\d+)\]$", st)
        if m:
            ops.append(("rz", (int(m.group(1)),)))
            continue
        m = re.match(r"c([xyz])\s+anc\[0\],\s*q\[(\d+)\]$", st)
        if m:
            ops.append(("flux", (int(m.group(2)), m.group(1).upper())))
    frame = {}
    bonds, loop, word = [], [], ""
    for k, (nm, qs) in enumerate(ops):
        if nm in ONE:
            frame[qs[0]] = ONE[nm] @ frame.get(qs[0], np.eye(2))
        elif nm == "flux":
            loop.append(qs[0])
            word += qs[1]
        elif nm == "rz" and 0 < k < len(ops) - 1 and ops[k - 1][0] == "cx" \
                and ops[k + 1][0] == "cx":
            qi, qj = ops[k - 1][1]
            cs = []
            for q in (qi, qj):
                U = frame.get(q, np.eye(2))
                eff = U.conj().T @ P[2] @ U
                hit = [c for c in range(3) if np.allclose(eff, P[c], atol=1e-9)
                       or np.allclose(eff, -P[c], atol=1e-9)]
                cs.append(hit[0] if len(hit) == 1 else None)
            if None in cs or cs[0] != cs[1]:
                raise SystemExit("%s: bond %d-%d is not a single Kitaev term"
                                 % (path, qi, qj))
            bonds.append([qi, qj, cs[0]])
    if not bonds or not loop:
        return None
    return {"blocks": [{"index": 0, "type": 0, "canonical_bonds": bonds,
                        "canonical_loop": loop, "canonical_word": word}]}


paths = sys.argv[1:]
if not paths:
    paths = [p for p in sorted(glob.glob("*.json")) if relabelled(p)]
    if not paths:
        sys.exit("usage: python3 w_commutes.py MANIFEST.json\n"
                 "No relabelled manifest found in this directory. Pass the\n"
                 "one K4-qasm.py wrote into its -o output directory.")
rc = 0
for p in paths:
    man = from_qasm(p) if p.endswith(".qasm") else relabelled(p)
    if man is None and p.endswith(".qasm"):
        print("== %s: no bonds or no Hadamard-test flux word found; use a\n"
              "   K4-qasm.py 'full' circuit (typeN-full.qasm)" % p)
        rc = 1
        continue
    if man is None:
        print("== %s: not a relabelled manifest (no %s). Use the copy\n"
              "   K4-qasm.py writes into its -o output directory." % (p, "/".join(KEYS)))
        rc = 1
        continue
    rc |= check(p, man) > 0
sys.exit(rc)