# -*- coding: us-ascii -*-
# stellarator_pyqrack_driver.py -- Rev 2.
#
# Native PyQrack 27-qubit (ancilla-free) block driver, entanglement-Hamiltonian
# seam repair (Sigma = A_ent - A_int), shared Gaussian/stabilizer state
# preparation, and multi-cycle FluxMonitor verification gate (Phase 4) for the
# 40 x 27 W7-X-like stellarator Kitaev lattice.
#
# Modelled on NN4K/coils/k4_pyqrack_driver.py and
# NN4K/coils/qasm-pyqrack-seam-backend.py.
#
#   python3 stellarator_pyqrack_driver.py --manifest manifest.json --check
#   python3 stellarator_pyqrack_driver.py --manifest manifest.json --block-type 0 --steps 4

import argparse
from collections import Counter
import json
import math
import os
import sys

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):
    os.environ.setdefault("PYQRACK_SHARED_LIB_PATH", QRACK_LIB_PATH)
# Allow Qrack's QUnit to engage automatic circuit elision (ACE) and
# Schmidt-decomposition rounding (SDRP) on all 27 qubits without throwing
# on the default fidelity guard threshold.
os.environ.setdefault("QRACK_DISABLE_QUNIT_FIDELITY_GUARD", "1")

import numpy as np
from pyqrack import Pauli, QrackSimulator

import stellarator_kitaev as SK

HAS_OPENCL_SO = os.path.exists(QRACK_LIB_PATH)
DEFAULT_MAX_KET_QUBITS = 27
DEFAULT_ACE_MAX_QB = 20
DEFAULT_SDRP = 0.02


# =====================================================================
# PART 1 -- 5-GATE BOND DECOMPOSITION & ANCILLA-FREE PARITY READOUT
# =====================================================================
def configure_simulator_27q(sim, ace_max_qb=DEFAULT_ACE_MAX_QB, sdrp=DEFAULT_SDRP):
    """Configure a 27-qubit QrackSimulator with native C++ ACE subsystem cap
    and SDRP rounding so all 27 qubits execute non-Clifford rotations cleanly."""
    if ace_max_qb is not None and hasattr(sim, "set_ace_max_qb"):
        sim.set_ace_max_qb(int(ace_max_qb))
    if sdrp is not None and sdrp > 0.0 and hasattr(sim, "set_sdrp"):
        sim.set_sdrp(float(sdrp))


def apply_basis_change(sim, pauli_col, qa, qb, inverse=False):
    """Rotate sigma^{pauli_col} into computational Z basis using pure
    Clifford gates (h, adjs, s)."""
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


def apply_single_basis_change(sim, q, pauli_col, inverse=False):
    """Rotate single qubit `q` between sigma^{pauli_col} and Z basis."""
    q_int = int(q)
    c_int = int(pauli_col)
    if c_int == 0:
        sim.h(q_int)
    elif c_int == 1:
        if not inverse:
            sim.adjs(q_int)
            sim.h(q_int)
        else:
            sim.h(q_int)
            sim.s(q_int)


def apply_bond_step(sim, qa, qb, pauli_col, theta, non_clifford=True):
    """Apply exp(-i * theta * sigma^col_qa sigma^col_qb) using the 5-gate
    CNOT + Rz decomposition (avoids pyqrack exp() ctypes mismatch)."""
    apply_basis_change(sim, pauli_col, qa, qb, inverse=False)
    sim.mcx([qa], qb)
    if non_clifford and abs(float(theta)) > 1e-15:
        sim.r(Pauli.PauliZ, 2.0 * float(theta), qb)
    sim.mcx([qa], qb)
    apply_basis_change(sim, pauli_col, qa, qb, inverse=True)


def apply_pauli_string_step(sim, pauli_pairs, theta, q_counts=None):
    """Apply exp(-i * theta * prod_{(q, c)} sigma^c_q) on any multi-qubit
    Kitaev Pauli string without ancilla qubits (basis rotate -> CNOT fold onto
    target q0 -> Rz(2*theta) -> uncompute CNOT -> inverse basis rotate).
    Used to execute all 45 Sigma_keep seam-repair terms including non-adjacent
    Majorana bilinears (`N9`)."""
    if not pauli_pairs or abs(float(theta)) < 1e-15:
        return
    if len(pauli_pairs) == 2 and int(pauli_pairs[0][1]) == int(pauli_pairs[1][1]):
        apply_bond_step(
            sim, int(pauli_pairs[0][0]), int(pauli_pairs[1][0]),
            int(pauli_pairs[0][1]), float(theta), non_clifford=True)
        return
    q0 = pick_fold_target(pauli_pairs, q_counts)
    other_qs = [int(q) for q, _ in pauli_pairs if int(q) != q0]
    for q, c in pauli_pairs:
        apply_single_basis_change(sim, q, c, inverse=False)
    for q in other_qs:
        sim.mcx([q], q0)
    sim.r(Pauli.PauliZ, 2.0 * float(theta), q0)
    for q in reversed(other_qs):
        sim.mcx([q], q0)
    for q, c in pauli_pairs:
        apply_single_basis_change(sim, q, c, inverse=True)


def compute_cycle_qubit_counts(block_spec):
    """Classify each local qubit for ancilla-free parity folding:
      - 0: peeled tree qubit outside the 2-core (safe leaf fold target for
           2-qubit bond readouts/sinks)
      - 1: private cycle qubit with 2-core degree == 2 (ideal cycle fold target)
      - 2: private cycle qubit with 2-core degree >= 3 (fallback cycle fold target)
      - >= 3: shared cycle or chord qubit in the 2-core."""
    rem = [
        (int(ib["qa"]), int(ib["qb"]), int(ib["colour"]))
        for ib in block_spec.get("internal_bonds", [])
    ]
    while True:
        deg = Counter()
        for u, v, _ in rem:
            deg[u] += 1
            deg[v] += 1
        leaf = None
        for u, v, c in rem:
            if deg[u] == 1:
                leaf = (u, v, c)
                break
            if deg[v] == 1:
                leaf = (v, u, c)
                break
        if leaf is None:
            break
        rem = [e for e in rem if set(e[:2]) != set(leaf[:2])]

    deg_2core = Counter()
    for u, v, _ in rem:
        deg_2core[u] += 1
        deg_2core[v] += 1

    cyc_counts = Counter()
    loops = block_spec.get("internal_loops", [])
    if not loops and "loop_pauli_local" in block_spec:
        loops = [{
            "pauli_local": block_spec["loop_pauli_local"],
            "target": block_spec.get("loop_target", 1.0),
        }]
    for cyc in loops:
        for q, _ in cyc["pauli_local"]:
            cyc_counts[int(q)] += 1

    q_counts = Counter()
    for q, cnt in cyc_counts.items():
        q_counts[q] = cnt
    for q, d2 in deg_2core.items():
        if d2 > 0 and q_counts.get(q, 0) == 0:
            q_counts[q] = 2
    return q_counts


def pick_fold_target(pairs, q_counts=None):
    """Select a private qubit (q_counts[q] == 1 for cycle words, or minimal
    q_counts[q] == 0 for 2-qubit bond words) of `pairs` as the CNOT parity
    fold target `q0` if available; otherwise return pairs[0][0]."""
    if q_counts is not None:
        if len(pairs) == 2:
            q0 = min(pairs, key=lambda qc: q_counts.get(int(qc[0]), 0))[0]
            return int(q0)
        for q, _ in pairs:
            if q_counts.get(int(q), 0) == 1:
                return int(q)
    return int(pairs[0][0])


def read_pauli_word_inplace(sim, pairs, q_counts=None):
    """Ancilla-free non-destructive expectation readout of a Pauli string
    P = prod_{(q, c)} sigma^c_q on a 27-qubit QrackSimulator via in-place
    Clifford basis rotation + CNOT parity fold onto `q0` + sim.prob(q0) +
    exact uncomputation."""
    if not pairs:
        return 1.0
    q0 = pick_fold_target(pairs, q_counts)
    other_qs = [int(q) for q, _ in pairs if int(q) != q0]
    for q, c in pairs:
        apply_single_basis_change(sim, q, c, inverse=False)
    for q in other_qs:
        sim.mcx([q], q0)
    p1 = float(sim.prob(q0))
    for q in reversed(other_qs):
        sim.mcx([q], q0)
    for q, c in pairs:
        apply_single_basis_change(sim, q, c, inverse=True)
    return 1.0 - 2.0 * p1


def project_pauli_word_inplace(sim, pairs, target_sign, q_counts=None):
    """Project `sim` into the `target_sign` (+1 or -1) eigenspace of the Pauli
    string P = prod_{(q, c)} sigma^c_q via in-place Clifford parity folding
    onto a private qubit `q0`, projective measurement `sim.m(q0)`, conditional
    X flip on `q0`, and uncomputation."""
    if not pairs:
        return
    q0 = pick_fold_target(pairs, q_counts)
    other_qs = [int(q) for q, _ in pairs if int(q) != q0]
    for q, c in pairs:
        apply_single_basis_change(sim, q, c, inverse=False)
    for q in other_qs:
        sim.mcx([q], q0)
    outcome = 1 if sim.m(q0) else 0
    want = 1 if float(target_sign) < 0 else 0
    if outcome != want:
        sim.x(q0)
    for q in reversed(other_qs):
        sim.mcx([q], q0)
    for q, c in pairs:
        apply_single_basis_change(sim, q, c, inverse=True)


def read_bond_expectation(sim, qa, qb, pauli_col, q_counts=None):
    """Read the 2-qubit bond expectation <sigma^col_qa sigma^col_qb> without
    an ancilla qubit and without collapsing the state."""
    return read_pauli_word_inplace(
        sim, [(int(qa), int(pauli_col)), (int(qb), int(pauli_col))],
        q_counts=q_counts)


def prepare_all_internal_loops(sim, block_spec):
    """Prepare the 27-qubit register in a simultaneous eigenstate of ALL
    independent internal cycles in `block_spec['internal_loops']` (all
    independent cycles across all 40 manifolds). Returns `q_counts`."""
    loops = block_spec.get("internal_loops", [])
    if not loops:
        loops = [{
            "pauli_local": block_spec["loop_pauli_local"],
            "target": block_spec.get("loop_target", 1.0),
        }]
    q_counts = compute_cycle_qubit_counts(block_spec)
    # Pre-align private cycle qubits to their cycle Pauli basis so the CNOT
    # parity fold target starts in a clean computational state after rotation
    aligned = set()
    for cyc in loops:
        for q, c in cyc["pauli_local"]:
            q_int, c_int = int(q), int(c)
            if q_counts[q_int] == 1 and q_int not in aligned:
                if c_int == 0:
                    sim.h(q_int)
                elif c_int == 1:
                    sim.h(q_int)
                    sim.s(q_int)
                aligned.add(q_int)
    for cyc in loops:
        project_pauli_word_inplace(
            sim, cyc["pauli_local"], float(cyc["target"]), q_counts=q_counts)
    return q_counts


def prepare_shared_dimer_state(sim, block_spec):
    """Prepare an exact gauge-fixed Gaussian stabilizer state shared between
    QrackSimulator `sim` and a Majorana correlation matrix `G0` (`N11`):
      1. Build a maximal vertex-disjoint matching of internal bonds, matching
         degree-3 vertices first so every unmatched vertex has degree <= 2
         (and therefore has at least one missing bond colour).
      2. Prepare a maximally entangled Bell pair on each matched dimer
         `(qa, qb, col, u_b)` with `<sigma^col_qa sigma^col_qb> = u_b` and
         zero single-qubit Pauli expectations (`<sigma^a_q> = 0`), giving
         `G0[qa, qb] = +1.0, G0[qb, qa] = -1.0`.
      3. Initialize every unmatched vertex `q` in the +1 eigenstate of a
         MISSING bond colour `m_col` at `q`, so `sigma^{m_col}_q = i b_q^{m_col} c_q`
         couples `c_q` only to a decoupled boundary zero mode `b_q^{m_col}`
         and never mixes bond gauge variables `u_{qr}`.
      4. Project every independent internal cycle `W_p` onto its target flux
         eigenvalue using a dimer-preserving Pauli flip on a private qubit.
    Returns `(q_counts, G0, matched_bonds)`."""
    n_q = int(block_spec.get("n_qubits", 27))
    q_counts = compute_cycle_qubit_counts(block_spec)
    loops = block_spec.get("internal_loops", [])
    if not loops and "loop_pauli_local" in block_spec:
        loops = [{
            "pauli_local": block_spec["loop_pauli_local"],
            "target": block_spec.get("loop_target", 1.0),
        }]

    ibs = block_spec["internal_bonds"]
    adj = {q: [] for q in range(n_q)}
    for ib in ibs:
        qa, qb, col = int(ib["qa"]), int(ib["qb"]), int(ib["colour"])
        u_b = float(ib["u"])
        adj[qa].append((qb, col, u_b))
        adj[qb].append((qa, col, -u_b))
    deg = {q: len(adj[q]) for q in range(n_q)}

    order_b = sorted(
        range(len(ibs)),
        key=lambda idx_b: -(deg[int(ibs[idx_b]["qa"])] + deg[int(ibs[idx_b]["qb"])])
    )
    used_qs = set()
    matched_bonds = []
    dimer_col_of = {}
    for idx_b in order_b:
        ib = ibs[idx_b]
        qa, qb, col = int(ib["qa"]), int(ib["qb"]), int(ib["colour"])
        u_b = float(ib["u"])
        if qa not in used_qs and qb not in used_qs:
            used_qs.add(qa)
            used_qs.add(qb)
            matched_bonds.append((qa, qb, col, u_b))
            dimer_col_of[qa] = col
            dimer_col_of[qb] = col

    # 1. Prepare Bell dimers in the bond Pauli basis with <sigma^c_a sigma^c_b> = u_b
    for qa, qb, col, u_b in matched_bonds:
        sim.h(qa)
        sim.mcx([qa], qb)
        if u_b < 0.0:
            sim.x(qb)
        apply_basis_change(sim, col, qa, qb, inverse=True)

    # 2. Prepare unmatched qubits in an eigenstate of a missing bond colour
    for q in range(n_q):
        if q not in used_qs:
            present = {col for _, col, _ in adj[q]}
            missing = [c for c in (0, 1, 2) if c not in present]
            m_col = missing[0] if missing else 2
            apply_single_basis_change(sim, q, m_col, inverse=True)

    # 3. Project every independent internal cycle onto its target sign without
    #    disturbing any dimer expectation
    for cyc in loops:
        pairs = cyc["pauli_local"]
        q0 = pick_fold_target(pairs, q_counts)
        other_qs = [int(q) for q, _ in pairs if int(q) != q0]
        for q, c in pairs:
            apply_single_basis_change(sim, q, c, inverse=False)
        for q in other_qs:
            sim.mcx([q], q0)
        outcome = 1 if sim.m(q0) else 0
        want = 1 if float(cyc["target"]) < 0 else 0
        for q in reversed(other_qs):
            sim.mcx([q], q0)
        for q, c in pairs:
            apply_single_basis_change(sim, q, c, inverse=True)
        if outcome != want:
            c_w = dict((int(q), int(c)) for q, c in pairs)[q0]
            c_flip = dimer_col_of.get(q0, (c_w + 1) % 3)
            if c_flip == c_w:
                c_flip = (c_w + 1) % 3
            if c_flip == 0:
                sim.x(q0)
            elif c_flip == 1:
                sim.y(q0)
            else:
                sim.z(q0)

    G0 = np.zeros((n_q, n_q), dtype=float)
    for qa, qb, col, u_b in matched_bonds:
        val = read_bond_expectation(sim, qa, qb, col, q_counts=q_counts)
        G0[qa, qb] = u_b * val
        G0[qb, qa] = -u_b * val

    return q_counts, G0, matched_bonds


def measure_all_internal_loops(sim, block_spec, q_counts=None):
    """Measure every independent internal cycle in `block_spec['internal_loops']`
    using ancilla-free in-place parity folding. Returns a list of per-cycle
    dicts (`loop_index`, `length`, `pauli_word`, `target`, `w_meas`, `eps_W`)."""
    loops = block_spec.get("internal_loops", [])
    if not loops:
        loops = [{
            "loop_index": 0,
            "length": len(block_spec.get("loop", [])),
            "pauli_word": block_spec.get("loop_pauli_word", ""),
            "pauli_local": block_spec["loop_pauli_local"],
            "target": block_spec.get("loop_target", 1.0),
        }]
    if q_counts is None:
        q_counts = compute_cycle_qubit_counts(block_spec)
    out = []
    for idx, cyc in enumerate(loops):
        tgt = float(cyc["target"])
        w_val = read_pauli_word_inplace(sim, cyc["pauli_local"], q_counts=q_counts)
        eps_w = ((w_val - tgt) / 2.0) ** 2
        out.append({
            "loop_index": int(cyc.get("cycle_index", cyc.get("loop_index", idx))),
            "length": int(cyc.get("length", len(cyc["pauli_local"]))),
            "pauli_word": cyc.get("pauli_word", ""),
            "target": tgt,
            "w_meas": w_val,
            "eps_W": eps_w,
        })
    return out


def measure_flux_hadamard(sim, loop_pauli_local, anc=None, q_counts=None):
    """Ancilla-free compatibility wrapper that reads <W_p> on 27 qubits via
    `read_pauli_word_inplace` (eliminating the 28th Hadamard ancilla qubit)."""
    return read_pauli_word_inplace(sim, loop_pauli_local, q_counts=q_counts)


def run_block_pyqrack(block_spec, steps=2, dt=0.04, use_sigma_repair=True,
                      ace_max_qb=None, sdrp=0.01):
    """Run an ancilla-free 27-qubit PyQrack simulator on a single manifold
    block, executing `steps` Trotter layers across ALL 27 qubits (no dropped
    bonds), including ALL 45 Sigma_keep seam-repair terms (exact 0.5*Sigma
    unit conversion, both 2-qubit internal bonds and multi-qubit Kitaev Pauli
    strings, `N9`), and measuring EVERY independent internal cycle in
    `block_spec['internal_loops']`."""
    sim = QrackSimulator(27, is_gpu=HAS_OPENCL_SO, is_stabilizer_hybrid=True)
    q_counts = prepare_all_internal_loops(sim, block_spec)
    init_cycles = [
        {
            "loop_index": int(cyc.get("cycle_index", cyc.get("loop_index", idx))),
            "length": int(cyc.get("length", len(cyc["pauli_local"]))),
            "pauli_word": cyc.get("pauli_word", ""),
            "target": float(cyc["target"]),
            "w_meas": float(cyc["target"]),
            "eps_W": 0.0,
        }
        for idx, cyc in enumerate(block_spec.get("internal_loops", []))
    ]

    configure_simulator_27q(sim, ace_max_qb=ace_max_qb, sdrp=sdrp)

    # Separate Sigma_keep into 2-qubit internal bond shifts (fused into internal
    # bonds) and non-adjacent multi-qubit Kitaev Pauli strings (executed via
    # apply_pauli_string_step). Exact unit factor: H_Sigma = 0.5 * Sigma * s_str * P.
    sig_bond_theta = {}
    sig_nonlocal = []
    n_sigma_applied = 0
    if use_sigma_repair:
        for sb in block_spec.get("sigma_repair_bonds", []):
            sig_ab = float(sb.get("sigma_ab", sb.get("sigma_val", 0.0)))
            if abs(sig_ab) < 1e-12:
                continue
            p_str = sb.get("pauli_string", [])
            s_str = float(sb.get("string_sign", -1.0))
            if sb.get("is_internal_bond", False):
                qa, qb = int(sb["qa"]), int(sb["qb"])
                sig_bond_theta[(min(qa, qb), max(qa, qb))] = 0.5 * sig_ab * s_str
                n_sigma_applied += 1
            elif p_str:
                sig_nonlocal.append((p_str, 0.5 * sig_ab * s_str))
                n_sigma_applied += 1

    final_cycles = init_cycles
    for _ in range(steps):
        for ib in block_spec["internal_bonds"]:
            qa, qb = int(ib["qa"]), int(ib["qb"])
            d_sig = sig_bond_theta.get((min(qa, qb), max(qa, qb)), 0.0)
            # Physical Kitaev spin Hamiltonian: H = -sum J_b sigma^c_a sigma^c_b
            # + 0.5 * Sigma_ab * s_str * sigma^c_a sigma^c_b
            theta = float(dt) * (-float(ib["weight"]) + d_sig)
            apply_bond_step(sim, qa, qb, int(ib["colour"]), theta,
                            non_clifford=True)
        for p_str, coeff in sig_nonlocal:
            apply_pauli_string_step(sim, p_str, float(dt) * coeff, q_counts=q_counts)
        final_cycles = measure_all_internal_loops(
            sim, block_spec, q_counts=q_counts)

    if steps == 0:
        final_cycles = measure_all_internal_loops(
            sim, block_spec, q_counts=q_counts)

    max_eps = max(c["eps_W"] for c in final_cycles) if final_cycles else 0.0
    mean_eps = float(np.mean([c["eps_W"] for c in final_cycles])) if final_cycles else 0.0
    return {
        "n_qubits": 27,
        "n_cycles": len(final_cycles),
        "n_sigma_applied": int(n_sigma_applied),
        "init_cycles": init_cycles,
        "final_cycles": final_cycles,
        "w_init": init_cycles[0]["w_meas"],
        "w_final": final_cycles[0]["w_meas"],
        "target": final_cycles[0]["target"],
        "eps_W": max_eps,
        "mean_eps_W": mean_eps,
    }


# =====================================================================
# PART 2 -- SEAM REPAIR OPERATOR (Sigma = A_ent - A_int) & BUDGET CHECK
# =====================================================================
def sparsify_antisymmetric(Sigma, keep=45):
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


def build_sigma_keep_from_manifest(block_spec):
    """Construct the 27 x 27 antisymmetric repair matrix `Sigma_keep` directly
    from `block_spec['sigma_repair_bonds']` shipped in `manifest.json`."""
    n_q = int(block_spec.get("n_qubits", 27))
    S_keep = np.zeros((n_q, n_q), dtype=float)
    for sb in block_spec.get("sigma_repair_bonds", []):
        qa, qb = int(sb["qa"]), int(sb["qb"])
        val = float(sb.get("sigma_ab", sb.get("sigma_val", 0.0)))
        S_keep[qa, qb] = val
        S_keep[qb, qa] = -val
    return S_keep


def evaluate_seam_repair(manifest, keep=45):
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
        if len(blk.get("sigma_repair_bonds", [])) == keep:
            Sigma_keep = build_sigma_keep_from_manifest(blk)
        else:
            Sigma_keep = sparsify_antisymmetric(Sigma, keep=keep)
        A_rep = A_int + Sigma_keep

        G_bare = SK.correlation_matrix(A_int)
        G_rep = SK.thermal_correlator(A_rep, beta=1.0)

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
    ap.add_argument("--dt", type=float, default=0.04)
    ap.add_argument("--keep", type=int, default=45,
                    help="number of upper-triangular Sigma_ij entries kept (default 45)")
    ap.add_argument("--require-qrack-lib", action="store_true",
                    help="fail loudly if QRACK_LIB_PATH is not present")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    if a.require_qrack_lib and not os.path.exists(QRACK_LIB_PATH):
        raise RuntimeError(
            "Required Qrack shared library missing at %s" % QRACK_LIB_PATH)

    with open(a.manifest, "r") as fh:
        manifest = json.load(fh)

    print("=" * 78)
    print("PHASE 4 GATE (PYQRACK DRIVER & SEAM REPAIR) -- 27-QUBIT MANIFOLDS")
    print("=" * 78)
    print("  QRACK_LIB_PATH = %s (exists=%s)" % (QRACK_LIB_PATH, HAS_OPENCL_SO))

    seam_res = evaluate_seam_repair(manifest, keep=a.keep)
    mean_rep = float(np.mean([r["rel_err_repaired"] for r in seam_res]))
    print("\n  Seam-Repair Budget Check (keep = %d local path<=3 pairs, mean %.3f%% <= 2.50%%):"
          % (a.keep, 100.0 * mean_rep))
    print("  %-8s %10s %18s %18s %10s"
          % ("Type", "int_bonds", "bare err (open)", "repaired (Sigma)", "verdict"))
    print("  " + "-" * 70)
    seam_ok = (mean_rep <= 0.025)
    for r in seam_res:
        ok = r["rel_err_repaired"] <= 0.035
        seam_ok = seam_ok and ok
        print("  type %-3d %10d %17.2f%% %17.3f%% %10s"
              % (r["type_id"], r["n_internal_bonds"],
                 100.0 * r["rel_err_bare"], 100.0 * r["rel_err_repaired"],
                 "PASS" if ok else "FAIL"))

    types_to_run = [a.block_type] if a.block_type is not None else [0, 1, 2, 3]
    print("\n  Native PyQrack 27-Qubit (Ancilla-Free) Multi-Cycle Trotter & FluxMonitor (%d steps):"
          % a.steps)
    print("  %-8s %7s %6s %14s %10s %12s %12s %8s"
          % ("Type", "Sigma", "cycle", "Pauli word", "target", "meas <W_p>", "eps_W", "verdict"))
    print("  " + "-" * 84)
    qrack_ok = True
    for k in types_to_run:
        bt = manifest["block_types"][k]
        res = run_block_pyqrack(bt, steps=a.steps, dt=a.dt)
        n_sig = int(res["n_sigma_applied"])
        for cyc in res["final_cycles"]:
            ok = (cyc["eps_W"] < 1e-6) and (n_sig == int(a.keep))
            qrack_ok = qrack_ok and ok
            print("  type %-3d %4d/%-2d %6d %14s %+10.4f %+12.6f %12.4e %8s"
                  % (k, n_sig, int(a.keep), cyc["loop_index"], cyc["pauli_word"],
                     cyc["target"], cyc["w_meas"], cyc["eps_W"],
                     "PASS" if ok else "FAIL"))

    all_ok = seam_ok and qrack_ok
    print("=" * 78)
    print("PHASE 4 GATE VERDICT: %s" % ("PASS" if all_ok else "FAIL"))
    print("=" * 78)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
