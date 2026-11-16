# -*- coding: us-ascii -*-
# stellarator-qasm.py -- Rev 1.
#
# Emit OpenQASM 3.0 circuits for each of the 4 canonical 27-qubit block types
# of the 40 x 27 stellarator manifest (`manifest.json`), modelled directly on
# NN4K/coils/K4-qasm.py.
#
# For each block type k in 0..3, emits:
#   - qasm/type<k>-trotter.qasm : one first-order Kitaev Trotter layer
#   - qasm/type<k>-flux.qasm    : Hadamard-test readout of the 8-site canonical
#                                 flux stabiliser W_p on ancilla q[27]
#   - qasm/type<k>-full.qasm    : N Trotter steps followed by W_p readout
#
#   python3 stellarator-qasm.py --manifest manifest.json --steps 2 --dt 0.1 -o qasm

import argparse
import json
import os


def basis_change_qasm(pauli_char, qa, qb, inverse=False):
    """OpenQASM 3 basis rotation mapping sigma^a_qa sigma^a_qb -> Z_qa Z_qb."""
    lines = []
    p = pauli_char.lower()
    if p == "x":
        lines.append("h q[%d];" % qa)
        lines.append("h q[%d];" % qb)
    elif p == "y":
        if not inverse:
            lines.append("rx(1.5707963267948966) q[%d];" % qa)
            lines.append("rx(1.5707963267948966) q[%d];" % qb)
        else:
            lines.append("rx(-1.5707963267948966) q[%d];" % qa)
            lines.append("rx(-1.5707963267948966) q[%d];" % qb)
    return lines


def bond_step_qasm(qa, qb, pauli_char, theta):
    """5-gate decomposition of exp(-i * theta * sigma^a_qa sigma^a_qb)."""
    lines = []
    lines.extend(basis_change_qasm(pauli_char, qa, qb, inverse=False))
    lines.append("cx q[%d], q[%d];" % (qa, qb))
    lines.append("rz(%.12f) q[%d];" % (2.0 * theta, qb))
    lines.append("cx q[%d], q[%d];" % (qa, qb))
    lines.extend(basis_change_qasm(pauli_char, qa, qb, inverse=True))
    return lines


def flux_readout_qasm(loop_pauli_local, anc=27):
    """Hadamard test on ancilla q[anc] for W_p = prod (q_loc, colour)."""
    lines = ["h q[%d];" % anc]
    for q_loc, col in loop_pauli_local:
        gate = ("cx", "cy", "cz")[int(col)]
        lines.append("%s q[%d], q[%d];" % (gate, anc, int(q_loc)))
    lines.append("h q[%d];" % anc)
    lines.append("c[0] = measure q[%d];" % anc)
    return lines


def emit_qasm_for_type(bt, steps=2, dt=0.1, out_dir="qasm"):
    os.makedirs(out_dir, exist_ok=True)
    k = int(bt["type_id"])

    # 1. Single Trotter step
    trotter_lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        "// Stellarator 27-qubit manifold Type %d -- 1 Trotter step (dt=%.4f)" % (k, dt),
        "qubit[27] q;",
    ]
    for ib in bt["internal_bonds"]:
        theta = -float(dt) * float(ib["weight"]) * float(ib["u"])
        trotter_lines.extend(
            bond_step_qasm(int(ib["qa"]), int(ib["qb"]), ib["pauli"], theta)
        )
    trotter_path = os.path.join(out_dir, "type%d-trotter.qasm" % k)
    with open(trotter_path, "w") as fh:
        fh.write("\n".join(trotter_lines) + "\n")

    # 2. Flux stabiliser Hadamard test
    flux_lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        "// Stellarator 27-qubit manifold Type %d -- W_p (%s) readout"
        % (k, bt["loop_pauli_word"]),
        "qubit[28] q;",
        "bit[1] c;",
    ]
    flux_lines.extend(flux_readout_qasm(bt["loop_pauli_local"], anc=27))
    flux_path = os.path.join(out_dir, "type%d-flux.qasm" % k)
    with open(flux_path, "w") as fh:
        fh.write("\n".join(flux_lines) + "\n")

    # 3. Full circuit (N Trotter steps + flux readout)
    full_lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        "// Stellarator 27-qubit manifold Type %d -- %d Trotter steps + W_p readout"
        % (k, steps),
        "qubit[28] q;",
        "bit[1] c;",
    ]
    for _ in range(steps):
        for ib in bt["internal_bonds"]:
            theta = -float(dt) * float(ib["weight"]) * float(ib["u"])
            full_lines.extend(
                bond_step_qasm(int(ib["qa"]), int(ib["qb"]), ib["pauli"], theta)
            )
    full_lines.extend(flux_readout_qasm(bt["loop_pauli_local"], anc=27))
    full_path = os.path.join(out_dir, "type%d-full.qasm" % k)
    with open(full_path, "w") as fh:
        fh.write("\n".join(full_lines) + "\n")

    return trotter_path, flux_path, full_path


def main():
    ap = argparse.ArgumentParser(
        description="Emit OpenQASM 3 circuits for the 4 stellarator block types")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("-o", "--out", default="qasm")
    a = ap.parse_args()

    with open(a.manifest, "r") as fh:
        manifest = json.load(fh)

    print("=" * 78)
    print("PHASE 4 (QASM) -- OPENQASM 3.0 EMITTER FOR 4 CANONICAL BLOCK TYPES")
    print("=" * 78)
    for bt in manifest["block_types"]:
        tp, fp, flp = emit_qasm_for_type(bt, steps=a.steps, dt=a.dt, out_dir=a.out)
        print("  Type %d: wrote %s, %s, %s" % (bt["type_id"], tp, fp, flp))
    print("=" * 78)


if __name__ == "__main__":
    main()
