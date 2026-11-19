# -*- coding: us-ascii -*-
# stellarator-qasm.py -- Rev 2.
#
# Emit OpenQASM 3.0 circuits for each of the 4 canonical 27-qubit block types
# of the 40 x 27 stellarator manifest (`manifest.json`), modelled directly on
# NN4K/coils/K4-qasm.py.
#
# Uses strictly 27 qubits (`qubit[27] q;`) per manifold with ancilla-free
# in-place Clifford parity folding onto the first loop qubit for all internal
# loop stabilisers W_p, plus optional Sigma_keep seam-repair bond layers.
#
# For each block type k in 0..3, emits:
#   - qasm/type<k>-trotter.qasm : one first-order Kitaev Trotter layer (+ Sigma_keep)
#   - qasm/type<k>-flux.qasm    : ancilla-free in-place parity-fold readout of all
#                                 internal loop stabilisers W_p on 27 qubits
#   - qasm/type<k>-full.qasm    : N Trotter steps followed by multi-cycle W_p readout
#
#   python3 stellarator-qasm.py --manifest manifest.json --steps 2 --dt 0.1 -o qasm

import argparse
import json
import os


def single_basis_qasm(pauli_col, q, inverse=False):
    """Basis rotation mapping Pauli col (0=X, 1=Y, 2=Z) on q[q] to Z."""
    c = int(pauli_col)
    if c == 0:
        return ["h q[%d];" % q]
    if c == 1:
        if not inverse:
            return ["rx(1.5707963267948966) q[%d];" % q]
        return ["rx(-1.5707963267948966) q[%d];" % q]
    return []


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


def flux_readout_inplace_qasm(loop_pauli_local, bit_idx=0, uncompute=True):
    """Ancilla-free in-place parity fold for W_p = prod (q_loc, colour) on 27 qubits."""
    q0 = int(loop_pauli_local[0][0])
    lines = []
    for q_loc, col in loop_pauli_local:
        lines.extend(single_basis_qasm(col, int(q_loc), inverse=False))
    for q_loc, _ in loop_pauli_local[1:]:
        lines.append("cx q[%d], q[%d];" % (int(q_loc), q0))
    lines.append("c[%d] = measure q[%d];" % (bit_idx, q0))
    if uncompute:
        for q_loc, _ in reversed(loop_pauli_local[1:]):
            lines.append("cx q[%d], q[%d];" % (int(q_loc), q0))
        for q_loc, col in reversed(loop_pauli_local):
            lines.extend(single_basis_qasm(col, int(q_loc), inverse=True))
    return lines


def emit_qasm_for_type(bt, steps=2, dt=0.1, out_dir="qasm"):
    os.makedirs(out_dir, exist_ok=True)
    k = int(bt["type_id"])
    int_loops = bt.get("internal_loops") or [{
        "cycle_index": 0,
        "pauli_local": bt["loop_pauli_local"],
        "pauli_word": bt["loop_pauli_word"],
    }]
    n_loops = len(int_loops)

    # 1. Single Trotter step (internal bonds + Sigma_keep seam repair)
    trotter_lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        "// Stellarator 27-qubit manifold Type %d -- 1 Trotter step + Sigma_keep (dt=%.4f)" % (k, dt),
        "qubit[27] q;",
    ]
    for ib in bt["internal_bonds"]:
        theta = -float(dt) * float(ib["weight"]) * float(ib["u"])
        trotter_lines.extend(
            bond_step_qasm(int(ib["qa"]), int(ib["qb"]), ib["pauli"], theta)
        )
    for sb in bt.get("sigma_repair_bonds", []):
        if sb.get("is_internal_bond", False):
            theta_s = -0.25 * float(dt) * float(sb["sigma_ab"])
            trotter_lines.extend(
                bond_step_qasm(int(sb["qa"]), int(sb["qb"]), sb["pauli"], theta_s)
            )
    trotter_path = os.path.join(out_dir, "type%d-trotter.qasm" % k)
    with open(trotter_path, "w") as fh:
        fh.write("\n".join(trotter_lines) + "\n")

    # 2. Ancilla-free in-place flux stabiliser readout across all internal cycles
    flux_lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        "// Stellarator 27-qubit manifold Type %d -- ancilla-free in-place W_p readout (%d cycles)"
        % (k, n_loops),
        "qubit[27] q;",
        "bit[%d] c;" % n_loops,
    ]
    for c_idx, lp in enumerate(int_loops):
        flux_lines.append("// Cycle %d (%s)" % (c_idx, lp["pauli_word"]))
        flux_lines.extend(
            flux_readout_inplace_qasm(lp["pauli_local"], bit_idx=c_idx, uncompute=True)
        )
    flux_path = os.path.join(out_dir, "type%d-flux.qasm" % k)
    with open(flux_path, "w") as fh:
        fh.write("\n".join(flux_lines) + "\n")

    # 3. Full circuit (N Trotter steps + ancilla-free multi-cycle flux readout)
    full_lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        "// Stellarator 27-qubit manifold Type %d -- %d Trotter steps + %d-cycle W_p readout"
        % (k, steps, n_loops),
        "qubit[27] q;",
        "bit[%d] c;" % n_loops,
    ]
    for _ in range(steps):
        for ib in bt["internal_bonds"]:
            theta = -float(dt) * float(ib["weight"]) * float(ib["u"])
            full_lines.extend(
                bond_step_qasm(int(ib["qa"]), int(ib["qb"]), ib["pauli"], theta)
            )
        for sb in bt.get("sigma_repair_bonds", []):
            if sb.get("is_internal_bond", False):
                theta_s = -0.25 * float(dt) * float(sb["sigma_ab"])
                full_lines.extend(
                    bond_step_qasm(int(sb["qa"]), int(sb["qb"]), sb["pauli"], theta_s)
                )
    for c_idx, lp in enumerate(int_loops):
        full_lines.extend(
            flux_readout_inplace_qasm(lp["pauli_local"], bit_idx=c_idx, uncompute=True)
        )
    full_path = os.path.join(out_dir, "type%d-full.qasm" % k)
    with open(full_path, "w") as fh:
        fh.write("\n".join(full_lines) + "\n")

    return trotter_path, flux_path, full_path


def main():
    ap = argparse.ArgumentParser(
        description="Emit 27-qubit OpenQASM 3 circuits for the 4 stellarator block types")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("-o", "--out", default="qasm")
    a = ap.parse_args()

    with open(a.manifest, "r") as fh:
        manifest = json.load(fh)

    print("=" * 78)
    print("PHASE 4 (QASM) -- 27-QUBIT OPENQASM 3.0 EMITTER (ANCILLA-FREE PARITY FOLD)")
    print("=" * 78)
    for bt in manifest["block_types"]:
        tp, fp, flp = emit_qasm_for_type(bt, steps=a.steps, dt=a.dt, out_dir=a.out)
        print("  Type %d (%d cycles): wrote %s, %s, %s"
              % (bt["type_id"], bt.get("n_internal_loops", 1), tp, fp, flp))
    print("=" * 78)


if __name__ == "__main__":
    main()

