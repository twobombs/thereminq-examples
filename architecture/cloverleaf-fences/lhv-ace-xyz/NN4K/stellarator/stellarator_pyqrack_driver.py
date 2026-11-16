# -*- coding: us-ascii -*-
# stellarator_pyqrack_driver.py -- Rev 1.
#
# Native PyQrack 27+1 qubit block driver, entanglement-Hamiltonian seam
# repair (Sigma = A_ent - A_int), and FluxMonitor verification gate (Phase 4)
# for the 40 x 27 W7-X-like stellarator Kitaev lattice.
#
# Modelled on NN4K/coils/k4_pyqrack_driver.py and
# NN4K/coils/qasm-pyqrack-seam-backend.py.
#
#   python3 stellarator_pyqrack_driver.py --manifest manifest.json --check
#   python3 stellarator_pyqrack_driver.py --manifest manifest.json --block-type 0 --steps 2

import argparse
import json
import math
import os
import sys

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):
    os.environ.setdefault("PYQRACK_SHARED_LIB_PATH", QRACK_LIB_PATH)

import numpy as np
from pyqrack import Pauli, QrackSimulator

import stellarator_kitaev as SK

HAS_OPENCL_SO = os.path.exists(QRACK_LIB_PATH)
DEFAULT_MAX_KET_QUBITS = 27 if HAS_OPENCL_SO else 14


# =====================================================================
# PART 1 -- 5-GATE BOND DECOMPOSITION & HADAMARD FLUX READOUT ON PYQRACK
# =====================================================================
def apply_basis_change(sim, pauli_col, qa, qb, inverse=False):
    """Rotate sigma^{pauli_col} into computational Z basis using pure
    Clifford gates (h, adjs, s) so Qrack's stabilizer-hybrid QUnit stays
    compressed until non-Clifford RZ rotations are applied."""
    for t in (qa, qb):
        if pauli_col == 0:
            sim.h(t)
        elif pauli_col == 1:
            if not inverse:
                sim.adjs(t)
                sim.h(t)
            else:
                sim.h(t)
                sim.s(t)


def apply_bond_step(sim, qa, qb, pauli_col, theta, non_clifford=True):
    """Apply exp(-i * theta * sigma^col_qa sigma^col_qb) using the 5-gate
    CNOT + Rz decomposition (avoids pyqrack exp() ctypes mismatch)."""
    apply_basis_change(sim, pauli_col, qa, qb, inverse=False)
    sim.mcx([qa], qb)
    if non_clifford:
        sim.r(Pauli.PauliZ, 2.0 * float(theta), qb)
    sim.mcx([qa], qb)
    apply_basis_change(sim, pauli_col, qa, qb, inverse=True)


def prepare_loop_sector_state(sim, loop_pauli_local, target_sign):
    """Prepare the 27-qubit register in a clean eigenstate of the canonical
    loop stabiliser W_p = prod_{(q, c)} sigma^c_q with eigenvalue `target_sign`
    (+1 or -1). Because every internal Kitaev bond commutes with W_p, all
    subsequent Trotter steps preserve <W_p> = target_sign."""
    for q_loc, col in loop_pauli_local:
        q = int(q_loc)
        c = int(col)
        if c == 0:
            sim.h(q)
        elif c == 1:
            sim.h(q)
            sim.s(q)
    if target_sign < 0 and loop_pauli_local:
        q0, c0 = int(loop_pauli_local[0][0]), int(loop_pauli_local[0][1])
        if c0 in (0, 1):
            sim.z(q0)
        else:
            sim.x(q0)


def measure_flux_hadamard(sim, loop_pauli_local, anc=27):
    """Non-destructive Hadamard test on ancilla qubit `anc` for
    W_p = prod_{(q, c)} sigma^c_q. Returns <W_p> in [-1, +1]."""
    if sim.m(anc):
        sim.x(anc)
    sim.h(anc)
    for q_loc, col in loop_pauli_local:
        q = int(q_loc)
        c = int(col)
        if c == 0:
            sim.mcx([anc], q)
        elif c == 1:
            sim.mcy([anc], q)
        else:
            sim.mcz([anc], q)
    sim.h(anc)
    p1 = float(sim.prob(anc))
    if sim.m(anc):
        sim.x(anc)
    return 1.0 - 2.0 * p1


def run_block_pyqrack(block_spec, steps=2, dt=0.08, max_ket_qubits=DEFAULT_MAX_KET_QUBITS):
    """Run a 28-qubit (27 site + 1 ancilla) PyQrack simulator on a single
    manifold block, executing `steps` Trotter layers and measuring <W_p>."""
    sim = QrackSimulator(28)
    target = float(block_spec["loop_target"])
    loop_pauli = block_spec["loop_pauli_local"]
    active_ket = set(range(min(27, int(max_ket_qubits))))

    prepare_loop_sector_state(sim, loop_pauli, target)
    w_init = measure_flux_hadamard(sim, loop_pauli, anc=27)

    for _ in range(steps):
        for ib in block_spec["internal_bonds"]:
            qa, qb = int(ib["qa"]), int(ib["qb"])
            theta = -float(dt) * float(ib["weight"]) * float(ib["u"])
            non_cliff = (qa in active_ket and qb in active_ket)
            apply_bond_step(sim, qa, qb, int(ib["colour"]), theta,
                            non_clifford=non_cliff)

    w_final = measure_flux_hadamard(sim, loop_pauli, anc=27)
    eps_w = ((w_final - target) / 2.0) ** 2
    return {
        "w_init": w_init,
        "w_final": w_final,
        "target": target,
        "eps_W": eps_w,
    }


# =====================================================================
# PART 2 -- SEAM REPAIR OPERATOR (Sigma = A_ent - A_int) & BUDGET CHECK
# =====================================================================
def sparsify_antisymmetric(Sigma, keep=42):
    """Keep the top `keep` upper-triangular entries (i < j) of a real
    antisymmetric matrix Sigma by magnitude |Sigma_{ij}|."""
    n = Sigma.shape[0]
    tri = []
    for i in range(n):
        for j in range(i + 1, n):
            val = float(Sigma[i, j])
            if abs(val) > 1e-15:
                tri.append((abs(val), i, j, val))
    tri.sort(reverse=True)
    S_keep = np.zeros_like(Sigma)
    for _, i, j, val in tri[:keep]:
        S_keep[i, j] = val
        S_keep[j, i] = -val
    return S_keep


def evaluate_seam_repair(manifest, keep=42):
    """Gate 4 verification: compare bare open block (A_int) vs magnitude-
    sparsified seam-repaired block (A_int + Sigma_keep) against the exact
    1080-site ground-state correlator G_exact across all 4 block types."""
    n_sites = int(manifest["n_sites"])
    bonds = [tuple(b) for b in manifest["bonds"]]
    u_g = np.array(manifest["gauge"], dtype=float)
    w_g = np.array(manifest["weights"], dtype=float)

    A_full = SK.majorana_matrix(n_sites, bonds, u=u_g, bond_weights=w_g)
    G_full = SK.correlation_matrix(A_full)

    results = []
    for k in range(4):
        blk = manifest["blocks"][k]  # module 0, block type k
        sites = blk["sites"]
        idx_arr = np.ix_(sites, sites)
        G_exact = G_full[idx_arr]
        A_int = A_full[idx_arr]
        A_ent = SK.entanglement_hamiltonian(G_exact)

        Sigma = A_ent - A_int
        Sigma_keep = sparsify_antisymmetric(Sigma, keep=keep)
        A_rep = A_int + Sigma_keep

        G_bare = SK.correlation_matrix(A_int)
        G_rep = SK.thermal_correlator(A_rep, beta=1.0)

        # Evaluate RMS error on internal bonds of block k
        bare_diffs = []
        rep_diffs = []
        exact_vals = []
        for ib in blk["internal_bonds"]:
            qa, qb = int(ib["qa"]), int(ib["qb"])
            g_ex = float(G_exact[qa, qb])
            g_ba = float(G_bare[qa, qb])
            g_re = float(G_rep[qa, qb])
            exact_vals.append(g_ex)
            bare_diffs.append(g_ba - g_ex)
            rep_diffs.append(g_re - g_ex)

        rms_ex = float(np.sqrt(np.mean(np.square(exact_vals))))
        rel_bare = float(np.sqrt(np.mean(np.square(bare_diffs)))) / max(rms_ex, 1e-12)
        rel_rep = float(np.sqrt(np.mean(np.square(rep_diffs)))) / max(rms_ex, 1e-12)

        results.append({
            "type_id": k,
            "n_internal_bonds": len(blk["internal_bonds"]),
            "keep": keep,
            "rel_err_bare": rel_bare,
            "rel_err_repaired": rel_rep,
        })
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Native PyQrack 27-qubit block driver & seam repair verifier")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--block-type", type=int, default=None)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--keep", type=int, default=42,
                    help="number of upper-triangular Sigma_ij entries kept (default 42)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    with open(a.manifest, "r") as fh:
        manifest = json.load(fh)

    print("=" * 78)
    print("PHASE 4 GATE (PYQRACK DRIVER & SEAM REPAIR) -- 27-QUBIT MANIFOLDS")
    print("=" * 78)
    print("  QRACK_LIB_PATH = %s" % QRACK_LIB_PATH)

    seam_res = evaluate_seam_repair(manifest, keep=a.keep)
    mean_rep = float(np.mean([r["rel_err_repaired"] for r in seam_res]))
    print("\n  Seam-Repair Budget Check (keep = %d off-diagonal pairs, mean %.3f%% <= 2.50%%):"
          % (a.keep, 100.0 * mean_rep))
    print("  %-8s %10s %18s %18s %10s"
          % ("Type", "int_bonds", "bare err (open)", "repaired (Sigma)", "verdict"))
    print("  " + "-" * 70)
    seam_ok = (mean_rep <= 0.025)
    for r in seam_res:
        ok = r["rel_err_repaired"] <= 0.030
        seam_ok = seam_ok and ok
        print("  type %-3d %10d %17.2f%% %17.3f%% %10s"
              % (r["type_id"], r["n_internal_bonds"],
                 100.0 * r["rel_err_bare"], 100.0 * r["rel_err_repaired"],
                 "PASS" if ok else "FAIL"))

    types_to_run = [a.block_type] if a.block_type is not None else [0, 1, 2, 3]
    print("\n  Native PyQrack 28-Qubit (27 + 1 Ancilla) Trotter & FluxMonitor Check:")
    print("  %-8s %12s %12s %12s %14s %10s"
          % ("Type", "Pauli word", "target W_p", "meas <W_p>", "eps_W", "verdict"))
    print("  " + "-" * 70)
    qrack_ok = True
    for k in types_to_run:
        bt = manifest["block_types"][k]
        res = run_block_pyqrack(bt, steps=a.steps, dt=a.dt)
        ok = res["eps_W"] < 1e-6
        qrack_ok = qrack_ok and ok
        print("  type %-3d %12s %+12.4f %+12.6f %14.4e %10s"
              % (k, bt["loop_pauli_word"], res["target"],
                 res["w_final"], res["eps_W"], "PASS" if ok else "FAIL"))

    all_ok = seam_ok and qrack_ok
    print("=" * 78)
    print("PHASE 4 GATE VERDICT: %s" % ("PASS" if all_ok else "FAIL"))
    print("=" * 78)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
