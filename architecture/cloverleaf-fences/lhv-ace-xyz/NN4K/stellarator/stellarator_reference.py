# -*- coding: us-ascii -*-
# stellarator_reference.py -- Rev 2.
#
# Exact free-fermion flow reference engine (beamline / orthogonal Majorana
# propagator + Givens bond rotations) and genuine 40 x 27 block-decomposed
# propagator (bare open blocks vs Sigma-repaired blocks + seam kicks) for
# validation ladder rungs F1 to F4 (Section 5.4).
#
# In the Majorana representation of the Kitaev model:
#   H = (i/4) sum_{j,k} A_{jk} c_j c_k,    <c_j c_k> = delta_{jk} + i G_{jk}
#
# A single time step dt evolves the real antisymmetric correlator G(t) by the
# orthogonal propagator U = exp(A * dt):
#   G(t + dt) = U G(t) U^T
#
# A flux-preserving bond rotation exp(-i theta sigma^a_j sigma^a_k) =
# exp((theta * u_{jk}) c_j c_k) acts on Majorana modes (c_j, c_k) as an exact
# 2x2 Givens rotation R(2 * theta * u_{jk}) in the (j, k) plane!
#
#   python3 stellarator_reference.py --manifest manifest.json --check

import argparse
import json
import math
import os
import sys

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):
    os.environ.setdefault("PYQRACK_SHARED_LIB_PATH", QRACK_LIB_PATH)
os.environ.setdefault("QRACK_DISABLE_QUNIT_FIDELITY_GUARD", "1")

import numpy as np
from pyqrack import QrackSimulator

import stellarator_kitaev as SK
import stellarator_pyqrack_driver as SPD


class MajoranaFlowPropagator(object):
    """Exact orthogonal Majorana correlation-matrix propagator on N sites."""

    def __init__(self, n_sites, bonds, u_gauge, weights=None, dt=0.08,
                 sigma_extra=None, g0_init=None, beta_init=None):
        self.n = int(n_sites)
        self.bonds = list(bonds)
        self.u = np.array(u_gauge, dtype=float)
        self.weights = (np.ones(len(bonds), dtype=float)
                        if weights is None else np.array(weights, dtype=float))
        self.dt = float(dt)

        self.A_bonds = SK.majorana_matrix(
            self.n, self.bonds, u=self.u, bond_weights=self.weights)
        if sigma_extra is not None:
            self.A = self.A_bonds + np.array(sigma_extra, dtype=float)
        else:
            self.A = self.A_bonds.copy()

        self.e0_total = None
        if g0_init is not None:
            self.G0 = np.array(g0_init, dtype=float).copy()
            X = self.A * self.dt
            if float(np.max(np.abs(X))) <= 0.8:
                U_tay = np.eye(self.n, dtype=float)
                term = np.eye(self.n, dtype=float)
                for k_ord in range(1, 13):
                    term = (term @ X) / float(k_ord)
                    U_tay += term
                self.U_step = U_tay
                self.G = self.G0.copy()
                return
        elif beta_init is not None:
            self.G0 = SK.thermal_correlator(self.A, beta=float(beta_init))
        else:
            ev, V = np.linalg.eigh(1j * self.A)
            self.e0_total = -0.25 * float(np.sum(np.abs(ev)))
            signs = np.where(np.abs(ev) > 1e-12, np.sign(ev), 0.0)
            C = (V * signs) @ V.conj().T
            self.G0 = np.real(-1j * C)
            self.G0 = 0.5 * (self.G0 - self.G0.T)
            phases = np.exp(-1j * ev * self.dt)
            self.U_step = np.real((V * phases) @ V.conj().T)
            self.G = self.G0.copy()
            return
        self.G = self.G0.copy()

        # Precompute orthogonal step matrix U = exp(A * dt) via eigendecomposition of iA
        ev, V = np.linalg.eigh(1j * self.A)
        self.e0_total = -0.25 * float(np.sum(np.abs(ev)))
        phases = np.exp(-1j * ev * self.dt)
        U_complex = (V * phases) @ V.conj().T
        self.U_step = np.real(U_complex)

    def reset(self):
        self.G = self.G0.copy()

    def set_correlator(self, G_new):
        self.G = np.array(G_new, dtype=float).copy()
        self.G0 = self.G.copy()

    def step_trotter(self):
        """Exact unitary step G <- U_step G U_step^T."""
        self.G = self.U_step @ self.G @ self.U_step.T
        self.G = 0.5 * (self.G - self.G.T)

    def apply_bond_kick(self, j, k, theta_eff):
        """Apply exact Givens rotation in the (j, k) Majorana plane:
        exp(phi/2 * c_j c_k) rotates (c_j, c_k) by angle phi = 2 * theta_eff."""
        phi = 2.0 * float(theta_eff)
        if abs(phi) < 1e-15:
            return
        c = math.cos(phi)
        s = math.sin(phi)
        rj = self.G[j, :].copy()
        rk = self.G[k, :].copy()
        self.G[j, :] = c * rj + s * rk
        self.G[k, :] = -s * rj + c * rk
        cj = self.G[:, j].copy()
        ck = self.G[:, k].copy()
        self.G[:, j] = c * cj + s * ck
        self.G[:, k] = -s * cj + c * ck
        self.G[j, j] = 0.0
        self.G[k, k] = 0.0

    def bond_expectations(self):
        """Return array of bond expectations <sigma^c_i sigma^c_j> = u_b * G_{ij}."""
        out = np.empty(len(self.bonds), dtype=float)
        for b_idx, (i, j, _) in enumerate(self.bonds):
            out[b_idx] = self.u[b_idx] * self.G[i, j]
        return out

    def bond_energies_above_ground(self, g_ref=None):
        """Return per-bond excitation energy delta_E_b >= 0 relative to g_ref (default G0)."""
        ref = self.G0 if g_ref is None else g_ref
        out = np.empty(len(self.bonds), dtype=float)
        for b_idx, (i, j, _) in enumerate(self.bonds):
            de = -self.weights[b_idx] * self.u[b_idx] * (self.G[i, j] - ref[i, j])
            out[b_idx] = max(0.0, float(de))
        return out


class BlockDecomposedFlowPropagator(object):
    """Genuine block-decomposed Majorana propagator maintaining separate
    per-block correlation matrices G_blocks[bid] (zero inter-block entries
    stored!) evolved under either:
      - bare open block Hamiltonian A_int[bid] (`use_sigma_repair=False`), or
      - seam-repaired Hamiltonian A_int[bid] + Sigma_keep[bid] built from
        `b['sigma_repair_bonds']` in `manifest.json` (`use_sigma_repair=True`),
    plus explicit inter-block seam kicks routed through boundary internal bonds."""

    def __init__(self, blocks, global_bonds, u_gauge, weights, dt=0.06,
                 use_sigma_repair=True):
        self.blocks = list(blocks)
        self.dt = float(dt)
        self.use_sigma_repair = bool(use_sigma_repair)
        self.block_props = {}
        self.g_to_blk_loc = {}
        self.incident_int_bond = {}

        for b in self.blocks:
            bid = int(b["block_id"])
            n_q = int(b.get("n_qubits", len(b["sites"])))
            for q_loc, g_site in enumerate(b["sites"]):
                self.g_to_blk_loc[int(g_site)] = (bid, int(q_loc))

            bonds_loc = []
            u_loc = []
            w_loc = []
            for idx_ib, ib in enumerate(b["internal_bonds"]):
                qa, qb, col = int(ib["qa"]), int(ib["qb"]), int(ib["colour"])
                bonds_loc.append((qa, qb, col))
                u_loc.append(float(ib["u"]))
                w_loc.append(float(ib["weight"]))
                if (bid, qa) not in self.incident_int_bond:
                    self.incident_int_bond[(bid, qa)] = idx_ib
                if (bid, qb) not in self.incident_int_bond:
                    self.incident_int_bond[(bid, qb)] = idx_ib

            if self.use_sigma_repair:
                sigma_keep = SPD.build_sigma_keep_from_manifest(b)
                prop_b = MajoranaFlowPropagator(
                    n_q, bonds_loc, u_loc, w_loc, dt=self.dt,
                    sigma_extra=sigma_keep, beta_init=1.0)
            else:
                prop_b = MajoranaFlowPropagator(
                    n_q, bonds_loc, u_loc, w_loc, dt=self.dt)
            self.block_props[bid] = prop_b

        # Identify seam bonds connecting two distinct blocks in `self.block_props`
        self.seam_couplings = []
        for b_idx, (gi, gj, col) in enumerate(global_bonds):
            loc_i = self.g_to_blk_loc.get(int(gi))
            loc_j = self.g_to_blk_loc.get(int(gj))
            if loc_i is None or loc_j is None:
                continue
            bid_i, qi = loc_i
            bid_j, qj = loc_j
            if bid_i != bid_j:
                ib_i = self.incident_int_bond.get((bid_i, qi))
                ib_j = self.incident_int_bond.get((bid_j, qj))
                if ib_i is not None and ib_j is not None:
                    self.seam_couplings.append((
                        bid_i, ib_i, bid_j, ib_j,
                        float(weights[b_idx]), float(u_gauge[b_idx])
                    ))

    def apply_global_bond_kick(self, gi, gj, theta_eff):
        """Apply a bond kick if (gi, gj) is internal to one of the blocks."""
        loc_i = self.g_to_blk_loc.get(int(gi))
        loc_j = self.g_to_blk_loc.get(int(gj))
        if loc_i is not None and loc_j is not None and loc_i[0] == loc_j[0]:
            bid = loc_i[0]
            self.block_props[bid].apply_bond_kick(loc_i[1], loc_j[1], theta_eff)

    def step(self):
        """Advance each 27-qubit block independently and exchange boundary
        2-qubit bond kicks across all inter-block seams."""
        for prop_b in self.block_props.values():
            prop_b.step_trotter()

        # Exchange boundary bond kicks proportional to boundary bond excitations
        kicks_to_apply = []
        for bid_i, ib_i, bid_j, ib_j, w_seam, u_seam in self.seam_couplings:
            pi = self.block_props[bid_i]
            pj = self.block_props[bid_j]
            qa_i, qb_i, _ = pi.bonds[ib_i]
            qa_j, qb_j, _ = pj.bonds[ib_j]
            ui = pi.u[ib_i]
            uj = pj.u[ib_j]
            # Excitation of boundary internal bond above its reference G0
            de_i = -ui * (pi.G[qa_i, qb_i] - pi.G0[qa_i, qb_i])
            de_j = -uj * (pj.G[qa_j, qb_j] - pj.G0[qa_j, qb_j])
            diff_ex = de_i - de_j
            if abs(diff_ex) > 1e-12:
                gain = 0.18 * self.dt * w_seam
                kicks_to_apply.append((bid_j, qa_j, qb_j, gain * diff_ex * uj))
                kicks_to_apply.append((bid_i, qa_i, qb_i, -gain * diff_ex * ui))

        for bid, qa, qb, angle in kicks_to_apply:
            self.block_props[bid].apply_bond_kick(qa, qb, angle)

    def internal_bond_expectations(self):
        """Concatenate <sigma^c_qa sigma^c_qb> across all internal bonds of all blocks."""
        parts = []
        for b in self.blocks:
            bid = int(b["block_id"])
            parts.append(self.block_props[bid].bond_expectations())
        return np.concatenate(parts) if parts else np.array([], dtype=float)

    def internal_bond_energies(self):
        """Return list of (global_i, global_j, delta_E_b) for all internal bonds."""
        out = []
        for b in self.blocks:
            bid = int(b["block_id"])
            sites = b["sites"]
            prop_b = self.block_props[bid]
            de = prop_b.bond_energies_above_ground()
            for idx_ib, (qa, qb, _) in enumerate(prop_b.bonds):
                out.append((int(sites[qa]), int(sites[qb]), float(de[idx_ib])))
        return out


# =====================================================================
# VALIDATION LADDER RUNGS F1 TO F4 (SECTION 5.4)
# =====================================================================
def rung_f1_single_manifold(manifest, steps=4, dt=0.04):
    """Rung F1: One 27-qubit manifold, no seams.
      1. Compare bond-by-bond Givens Trotter vs exact 27x27 matrix exponential.
      2. Compare 27-qubit PyQrack simulator (all 27 qubits, ancilla-free)
         against MajoranaFlowPropagator(27) starting from the exact same shared
         dimer/loop stabilizer state (`prepare_shared_dimer_state`) across
         `steps` Trotter steps (`M3, C1`)."""
    blk = manifest["blocks"][1]
    n_q = 27
    bonds_loc = []
    u_loc = []
    w_loc = []
    for ib in blk["internal_bonds"]:
        bonds_loc.append((int(ib["qa"]), int(ib["qb"]), int(ib["colour"])))
        u_loc.append(float(ib["u"]))
        w_loc.append(float(ib["weight"]))

    prop_exact = MajoranaFlowPropagator(n_q, bonds_loc, u_loc, w_loc, dt=dt)
    prop_trot = MajoranaFlowPropagator(n_q, bonds_loc, u_loc, w_loc, dt=dt)

    b0_i, b0_j, _ = bonds_loc[0]
    prop_exact.apply_bond_kick(b0_i, b0_j, 0.35 * u_loc[0])
    prop_trot.apply_bond_kick(b0_i, b0_j, 0.35 * u_loc[0])

    max_diff = 0.0
    for _ in range(steps):
        prop_exact.step_trotter()
        for b_idx, (qa, qb, _) in enumerate(bonds_loc):
            prop_trot.apply_bond_kick(qa, qb, 0.5 * dt * w_loc[b_idx] * u_loc[b_idx])
        for b_idx in reversed(range(len(bonds_loc))):
            qa, qb, _ = bonds_loc[b_idx]
            prop_trot.apply_bond_kick(qa, qb, 0.5 * dt * w_loc[b_idx] * u_loc[b_idx])
        diff = float(np.max(np.abs(prop_exact.bond_expectations() -
                                   prop_trot.bond_expectations())))
        max_diff = max(max_diff, diff)

    # Shared initial state comparison: 27-qubit PyQrack vs 27-site Majorana
    sim = QrackSimulator(27, is_gpu=SPD.HAS_OPENCL_SO)
    q_counts, G0_shared, _ = SPD.prepare_shared_dimer_state(sim, blk)
    SPD.configure_simulator_27q(sim, ace_max_qb=21, sdrp=0.0)

    prop_shared = MajoranaFlowPropagator(
        n_q, bonds_loc, u_loc, w_loc, dt=dt, g0_init=G0_shared)

    qrack_step_rms = []
    for _ in range(steps):
        for b_idx, (qa, qb, col) in enumerate(bonds_loc):
            theta = -float(dt) * w_loc[b_idx] * u_loc[b_idx]
            SPD.apply_bond_step(sim, qa, qb, col, theta, non_clifford=True)
            prop_shared.apply_bond_kick(qa, qb, -theta * u_loc[b_idx])

        ref_exps = prop_shared.bond_expectations()
        sq_errs = []
        for b_idx, (qa, qb, col) in enumerate(bonds_loc):
            q_exp = SPD.read_bond_expectation(sim, qa, qb, col, q_counts=q_counts)
            sq_errs.append((q_exp - float(ref_exps[b_idx])) ** 2)
        qrack_step_rms.append(float(np.sqrt(np.mean(sq_errs))))

    w_cyc0 = SPD.read_pauli_word_inplace(
        sim, blk["internal_loops"][0]["pauli_local"], q_counts=q_counts)
    qrack_rms = float(np.mean(qrack_step_rms))

    return {
        "rung": "F1",
        "system": "27-qubit manifold (PyQrack + Majorana)",
        "steps": steps,
        "dt": dt,
        "max_bond_err": max_diff,
        "pyqrack_rms_bond_err": qrack_rms,
        "pyqrack_step1_rms_err": float(qrack_step_rms[0]),
        "pyqrack_w0": float(w_cyc0),
        "pass": (max_diff < 0.02) and (qrack_rms < 0.03) and (abs(w_cyc0 - 1.0) < 1e-4),
    }


def step_decomposed_strang(prop, bonds_list, u_arr, w_arr, is_seam_arr, dt):
    """Execute one colour-decomposed step using the 3-colour matching structure
    of the Kitaev lattice: for each colour layer c in (0, 1, 2, 2, 1, 0) with
    half-step 0.5*dt, apply internal manifold bonds first, then seam kicks."""
    half = 0.5 * float(dt)
    for col in (0, 1, 2, 2, 1, 0):
        for b_idx, (qa, qb, c) in enumerate(bonds_list):
            if c == col and not is_seam_arr[b_idx]:
                prop.apply_bond_kick(qa, qb, half * w_arr[b_idx] * u_arr[b_idx])
        for b_idx, (qa, qb, c) in enumerate(bonds_list):
            if c == col and is_seam_arr[b_idx]:
                prop.apply_bond_kick(qa, qb, half * w_arr[b_idx] * u_arr[b_idx])


def rung_f2_single_module(manifest, steps=8, dt=0.06):
    """Rung F2: One half-period module (108 qubits = 4 x 27 K4 cluster).
    Compare:
      1. Bare open 4 x 27 blocks (truncated inter-block correlations, no Sigma)
      2. Sigma-repaired 4 x 27 blocks (`A_int + Sigma_keep` + seam kicks)
      3. 3-colour Strang K4-kick evolution
    against the exact 108-site module correlator G(t)."""
    raw_mod_blocks = manifest["blocks"][:4]
    mod_sites = []
    for b in raw_mod_blocks:
        mod_sites.extend(b["sites"])
    g_to_m = {int(g): loc for loc, g in enumerate(mod_sites)}

    bonds_mod = []
    u_mod = []
    w_mod = []
    is_k4_seam = []
    block_of_m = {loc: loc // 27 for loc in range(108)}

    global_bonds = [tuple(b) for b in manifest["bonds"]]
    u_g = np.array(manifest["gauge"], dtype=float)
    w_g = np.array(manifest["weights"], dtype=float)

    for b_idx, (i, j, c) in enumerate(global_bonds):
        if i in g_to_m and j in g_to_m:
            mi, mj = g_to_m[i], g_to_m[j]
            bonds_mod.append((mi, mj, int(c)))
            u_mod.append(float(u_g[b_idx]))
            w_mod.append(float(w_g[b_idx]))
            is_k4_seam.append(block_of_m[mi] != block_of_m[mj])

    prop_exact = MajoranaFlowPropagator(108, bonds_mod, u_mod, w_mod, dt=dt)
    prop_k4 = MajoranaFlowPropagator(108, bonds_mod, u_mod, w_mod, dt=dt)

    # Build module-embedded 4 x 27 blocks with Sigma_keep from the 108-site module
    A_mod = prop_exact.A
    G_mod = prop_exact.G0
    mod_blocks = []
    for k_blk, b in enumerate(raw_mod_blocks):
        m_indices = list(range(27 * k_blk, 27 * (k_blk + 1)))
        idx_arr = np.ix_(m_indices, m_indices)
        Sigma_m = SK.entanglement_hamiltonian(G_mod[idx_arr]) - A_mod[idx_arr]
        Sigma_keep_m = SPD.sparsify_antisymmetric(Sigma_m, keep=45)
        sig_bonds = []
        for qa in range(27):
            for qb in range(qa + 1, 27):
                if abs(Sigma_keep_m[qa, qb]) > 1e-12:
                    sig_bonds.append({
                        "qa": qa,
                        "qb": qb,
                        "sigma_ab": float(Sigma_keep_m[qa, qb]),
                    })
        b_copy = dict(b)
        b_copy["sites"] = m_indices
        b_copy["sigma_repair_bonds"] = sig_bonds
        mod_blocks.append(b_copy)

    prop_bare_blks = BlockDecomposedFlowPropagator(
        mod_blocks, bonds_mod, u_mod, w_mod, dt=dt, use_sigma_repair=False)
    prop_rep_blks = BlockDecomposedFlowPropagator(
        mod_blocks, bonds_mod, u_mod, w_mod, dt=dt, use_sigma_repair=True)

    int_bond_pairs = []
    for b in mod_blocks:
        sites = b["sites"]
        for ib in b["internal_bonds"]:
            mi = int(sites[int(ib["qa"])])
            mj = int(sites[int(ib["qb"])])
            int_bond_pairs.append((mi, mj, float(ib["u"])))

    ib0 = mod_blocks[0]["internal_bonds"][0]
    m0_i = int(mod_blocks[0]["sites"][int(ib0["qa"])])
    m0_j = int(mod_blocks[0]["sites"][int(ib0["qb"])])
    kick_angle = 0.10 * float(ib0["u"])
    prop_exact.apply_bond_kick(m0_i, m0_j, kick_angle)
    prop_k4.apply_bond_kick(m0_i, m0_j, kick_angle)
    prop_bare_blks.apply_global_bond_kick(m0_i, m0_j, kick_angle)
    prop_rep_blks.apply_global_bond_kick(m0_i, m0_j, kick_angle)

    rms_strang_list = []
    rms_bare_list = []
    rms_rep_list = []
    for _ in range(steps):
        prop_exact.step_trotter()
        step_decomposed_strang(prop_k4, bonds_mod, u_mod, w_mod, is_k4_seam, dt)
        prop_bare_blks.step()
        prop_rep_blks.step()

        rms_s = float(np.sqrt(np.mean(np.square(
            prop_exact.bond_expectations() - prop_k4.bond_expectations()))))
        rms_strang_list.append(rms_s)

        ex_int = np.array([
            u_b * prop_exact.G[mi, mj] for mi, mj, u_b in int_bond_pairs
        ], dtype=float)
        bare_int = prop_bare_blks.internal_bond_expectations()
        rep_int = prop_rep_blks.internal_bond_expectations()

        rms_bare_list.append(float(np.sqrt(np.mean(np.square(ex_int - bare_int)))))
        rms_rep_list.append(float(np.sqrt(np.mean(np.square(ex_int - rep_int)))))

    mean_bare = float(np.mean(rms_bare_list))
    mean_rep = float(np.mean(rms_rep_list))
    max_strang = max(rms_strang_list)
    return {
        "rung": "F2",
        "system": "108-qubit half-period module (4x27 K4 cluster)",
        "steps": steps,
        "dt": dt,
        "rms_bond_err": mean_rep,
        "rms_bond_err_bare": mean_bare,
        "rms_bond_err_repaired": mean_rep,
        "rms_bond_err_strang": max_strang,
        "pass": (mean_rep < 0.035) and (mean_rep < 0.35 * mean_bare) and (max_strang < 0.02),
    }


def rung_f3_toroidal_ring(manifest, steps=16, dt=0.10):
    """Rung F3: Ring of 10 modules on a single shell (200 sites = 10 x 20 blocks).
    Compare exact 200-site ring propagation against:
      1. 10 decoupled 20-site module blocks coupled solely by inter-module seam
         Givens kicks (bare open modules vs Sigma-repaired modules), and
      2. 3-colour Strang module-to-module seam kicks."""
    idx, pos, bonds, loops, meta = SK.build(
        nshell=2, candidate=manifest["candidate"], closed=True,
        preset=manifest["preset"])
    s0_sites = [s for s, sh in meta["site_shell"].items() if sh == 0]
    g_to_s0 = {g: loc for loc, g in enumerate(s0_sites)}
    mod_of_s0 = {g_to_s0[g]: meta["site_module"][g] for g in s0_sites}

    bonds_s0 = []
    u_s0 = []
    w_s0 = []
    is_tor_seam = []
    for b_idx, (i, j, c) in enumerate(bonds):
        if i in g_to_s0 and j in g_to_s0 and meta["bond_meta"][b_idx]["kind"] != "radial":
            si, sj = g_to_s0[i], g_to_s0[j]
            bonds_s0.append((si, sj, int(c)))
            u_s0.append(float(meta["u_ground"][b_idx]))
            w_s0.append(1.0)
            is_tor_seam.append(mod_of_s0[si] != mod_of_s0[sj])

    prop_exact = MajoranaFlowPropagator(200, bonds_s0, u_s0, w_s0, dt=dt)
    prop_kick = MajoranaFlowPropagator(200, bonds_s0, u_s0, w_s0, dt=dt)
    prop_vac = MajoranaFlowPropagator(200, bonds_s0, u_s0, w_s0, dt=dt)

    A_ring = prop_exact.A
    G_ring = prop_exact.G0
    mod_blocks = []
    for m_id in range(10):
        m_sites = [si for si in range(200) if mod_of_s0[si] == m_id]
        s_to_q = {si: q for q, si in enumerate(m_sites)}
        ib_list = []
        for b_idx, (si, sj, col) in enumerate(bonds_s0):
            if si in s_to_q and sj in s_to_q:
                ib_list.append({
                    "qa": s_to_q[si],
                    "qb": s_to_q[sj],
                    "colour": col,
                    "u": u_s0[b_idx],
                    "weight": w_s0[b_idx],
                })
        idx_arr = np.ix_(m_sites, m_sites)
        Sigma_m = SK.entanglement_hamiltonian(G_ring[idx_arr]) - A_ring[idx_arr]
        Sigma_keep_m = SPD.sparsify_antisymmetric(Sigma_m, keep=35)
        sig_bonds = []
        for qa in range(len(m_sites)):
            for qb in range(qa + 1, len(m_sites)):
                if abs(Sigma_keep_m[qa, qb]) > 1e-12:
                    sig_bonds.append({
                        "qa": qa,
                        "qb": qb,
                        "sigma_ab": float(Sigma_keep_m[qa, qb]),
                    })
        mod_blocks.append({
            "block_id": m_id,
            "n_qubits": len(m_sites),
            "sites": m_sites,
            "internal_bonds": ib_list,
            "sigma_repair_bonds": sig_bonds,
        })

    prop_mod_bare = BlockDecomposedFlowPropagator(
        mod_blocks, bonds_s0, u_s0, w_s0, dt=dt, use_sigma_repair=False)
    prop_mod_rep = BlockDecomposedFlowPropagator(
        mod_blocks, bonds_s0, u_s0, w_s0, dt=dt, use_sigma_repair=True)

    int_ring_pairs = []
    for mb in mod_blocks:
        ms = mb["sites"]
        for ib in mb["internal_bonds"]:
            int_ring_pairs.append((ms[ib["qa"]], ms[ib["qb"]], float(ib["u"])))

    for b_idx, (si, sj, _) in enumerate(bonds_s0):
        if mod_of_s0[si] == 0 and mod_of_s0[sj] == 0:
            prop_exact.apply_bond_kick(si, sj, 0.30 * u_s0[b_idx])
            prop_kick.apply_bond_kick(si, sj, 0.30 * u_s0[b_idx])
            prop_mod_bare.apply_global_bond_kick(si, sj, 0.30 * u_s0[b_idx])
            prop_mod_rep.apply_global_bond_kick(si, sj, 0.30 * u_s0[b_idx])

    opp_energy_exact = []
    opp_energy_kick = []
    rms_bare_ring = []
    rms_rep_ring = []
    for _ in range(steps):
        prop_exact.step_trotter()
        step_decomposed_strang(prop_kick, bonds_s0, u_s0, w_s0, is_tor_seam, dt)
        step_decomposed_strang(prop_vac, bonds_s0, u_s0, w_s0, is_tor_seam, dt)
        prop_mod_bare.step()
        prop_mod_rep.step()

        de_ex = prop_exact.bond_energies_above_ground()
        de_ki = prop_kick.bond_energies_above_ground(g_ref=prop_vac.G)
        e5_ex = float(sum(de_ex[b] for b, (si, _, _) in enumerate(bonds_s0) if mod_of_s0[si] == 5))
        e5_ki = float(sum(de_ki[b] for b, (si, _, _) in enumerate(bonds_s0) if mod_of_s0[si] == 5))
        opp_energy_exact.append(e5_ex)
        opp_energy_kick.append(e5_ki)

        ex_int = np.array([
            u_b * prop_exact.G[si, sj] for si, sj, u_b in int_ring_pairs
        ], dtype=float)
        rms_bare_ring.append(float(np.sqrt(np.mean(np.square(
            ex_int - prop_mod_bare.internal_bond_expectations())))))
        rms_rep_ring.append(float(np.sqrt(np.mean(np.square(
            ex_int - prop_mod_rep.internal_bond_expectations())))))

    t_peak_exact = int(np.argmax(opp_energy_exact)) + 1
    t_peak_kick = int(np.argmax(opp_energy_kick)) + 1
    transit_diff = abs(t_peak_exact - t_peak_kick)
    mean_bare = float(np.mean(rms_bare_ring))
    mean_rep = float(np.mean(rms_rep_ring))

    return {
        "rung": "F3",
        "system": "200-qubit 10-module toroidal shell ring",
        "steps": steps,
        "t_peak_exact": t_peak_exact,
        "t_peak_kick": t_peak_kick,
        "transit_step_diff": transit_diff,
        "rms_bond_err_bare": mean_bare,
        "rms_bond_err_repaired": mean_rep,
        "pass": (transit_diff <= 1) and (mean_rep < mean_bare),
    }


def rung_f4_full_machine(manifest, steps=10, dt=0.06):
    """Rung F4: Full 1080-qubit stellarator (1620 bonds = 40 x 27 blocks).
    Compare:
      1. Bare open 40 x 27 blocks (`use_sigma_repair=False`)
      2. Sigma-repaired 40 x 27 blocks (`use_sigma_repair=True`, using
         `b['sigma_repair_bonds']` from `manifest.json` + seam bond kicks)
      3. 3-colour Strang seam-kick decomposition
    against the exact 1080-site propagator."""
    n_sites = int(manifest["n_sites"])
    bonds = [tuple(b) for b in manifest["bonds"]]
    u_g = np.array(manifest["gauge"], dtype=float)
    w_g = np.array(manifest["weights"], dtype=float)

    site_shell = {}
    site_block = {}
    for b in manifest["blocks"]:
        bid = int(b["block_id"])
        for q, g in enumerate(b["sites"]):
            site_shell[int(g)] = int(b["shells"][q])
            site_block[int(g)] = bid

    is_seam = [site_block[i] != site_block[j] for i, j, _ in bonds]

    prop_exact = MajoranaFlowPropagator(n_sites, bonds, u_g, w_g, dt=dt)
    prop_decomp = MajoranaFlowPropagator(n_sites, bonds, u_g, w_g, dt=dt)
    prop_vac = MajoranaFlowPropagator(n_sites, bonds, u_g, w_g, dt=dt)
    prop_bare_40 = BlockDecomposedFlowPropagator(
        manifest["blocks"], bonds, u_g, w_g, dt=dt, use_sigma_repair=False)
    prop_rep_40 = BlockDecomposedFlowPropagator(
        manifest["blocks"], bonds, u_g, w_g, dt=dt, use_sigma_repair=True)

    int_pairs_40 = []
    for b in manifest["blocks"]:
        sites = b["sites"]
        for ib in b["internal_bonds"]:
            gi = int(sites[int(ib["qa"])])
            gj = int(sites[int(ib["qb"])])
            int_pairs_40.append((gi, gj, float(ib["u"])))

    for b_idx, (i, j, _) in enumerate(bonds):
        if (site_shell[i] == 0 and site_shell[j] == 0
                and site_block[i] == site_block[j] and site_block[i] < 4):
            kick = 0.12 * u_g[b_idx]
            prop_exact.apply_bond_kick(i, j, kick)
            prop_decomp.apply_bond_kick(i, j, kick)
            prop_bare_40.apply_global_bond_kick(i, j, kick)
            prop_rep_40.apply_global_bond_kick(i, j, kick)

    rms_bare_list = []
    rms_rep_list = []
    for _ in range(steps):
        prop_exact.step_trotter()
        step_decomposed_strang(prop_decomp, bonds, u_g, w_g, is_seam, dt)
        step_decomposed_strang(prop_vac, bonds, u_g, w_g, dt=dt, is_seam_arr=is_seam)
        prop_bare_40.step()
        prop_rep_40.step()

        ex_int = np.array([
            u_b * prop_exact.G[gi, gj] for gi, gj, u_b in int_pairs_40
        ], dtype=float)
        rms_bare_list.append(float(np.sqrt(np.mean(np.square(
            ex_int - prop_bare_40.internal_bond_expectations())))))
        rms_rep_list.append(float(np.sqrt(np.mean(np.square(
            ex_int - prop_rep_40.internal_bond_expectations())))))

    de_ex = prop_exact.bond_energies_above_ground()
    de_dc = prop_decomp.bond_energies_above_ground(g_ref=prop_vac.G)

    shell_ex = np.zeros(5, dtype=float)
    shell_dc = np.zeros(5, dtype=float)
    shell_blk40 = np.zeros(5, dtype=float)
    for b_idx, (i, j, _) in enumerate(bonds):
        s = max(site_shell[i], site_shell[j])
        shell_ex[s] += de_ex[b_idx]
        shell_dc[s] += de_dc[b_idx]
    for gi, gj, de_b in prop_rep_40.internal_bond_energies():
        s = max(site_shell[gi], site_shell[gj])
        shell_blk40[s] += de_b

    tot_ex = max(float(np.sum(shell_ex)), 1e-12)
    tot_dc = max(float(np.sum(shell_dc)), 1e-12)
    tot_b40 = max(float(np.sum(shell_blk40)), 1e-12)
    prof_ex = shell_ex / tot_ex
    prof_dc = shell_dc / tot_dc
    prof_b40 = shell_blk40 / tot_b40
    max_shell_dev = float(np.max(np.abs(prof_ex - prof_dc)))
    mean_bare = float(np.mean(rms_bare_list))
    mean_rep = float(np.mean(rms_rep_list))

    return {
        "rung": "F4",
        "system": "1080-qubit full stellarator (40 x 27 manifolds)",
        "steps": steps,
        "shell_profile_exact": [float(x) for x in prof_ex],
        "shell_profile_decomp": [float(x) for x in prof_dc],
        "shell_profile_blocks40": [float(x) for x in prof_b40],
        "max_shell_deviation": max_shell_dev,
        "rms_bond_err_bare": mean_bare,
        "rms_bond_err_repaired": mean_rep,
        "outer_leakage_fraction": float(prof_ex[3] + prof_ex[4]),
        "pass": (max_shell_dev < 0.05) and (mean_rep < 0.035) and (mean_rep < 0.30 * mean_bare),
    }


def run_reference_ladder(manifest):
    print("=" * 78)
    print("PHASE 5 GATE (REFERENCE FLOW LADDER) -- EXACT & BLOCK-DECOMPOSED RUNGS F1-F4")
    print("=" * 78)
    r1 = rung_f1_single_manifold(manifest)
    print("  [F1] %-42s givens_err=%.3e  pyqrack_rms=%.4f (<W_0>=%+.4f)  %s"
          % (r1["system"], r1["max_bond_err"], r1["pyqrack_rms_bond_err"],
             r1["pyqrack_w0"], "PASS" if r1["pass"] else "FAIL"))

    r2 = rung_f2_single_module(manifest)
    print("  [F2] %-42s bare_err=%.4f  sigma_rep_err=%.4f  strang=%.3e  %s"
          % (r2["system"], r2["rms_bond_err_bare"], r2["rms_bond_err_repaired"],
             r2["rms_bond_err_strang"], "PASS" if r2["pass"] else "FAIL"))

    r3 = rung_f3_toroidal_ring(manifest)
    print("  [F3] %-42s transit=%d vs %d  bare_err=%.4f  sigma_rep_err=%.4f  %s"
          % (r3["system"], r3["t_peak_exact"], r3["t_peak_kick"],
             r3["rms_bond_err_bare"], r3["rms_bond_err_repaired"],
             "PASS" if r3["pass"] else "FAIL"))

    r4 = rung_f4_full_machine(manifest)
    print("  [F4] %-42s max_shell_dev=%.4f  bare=%.4f  sigma_rep=%.4f  %s"
          % (r4["system"], r4["max_shell_deviation"],
             r4["rms_bond_err_bare"], r4["rms_bond_err_repaired"],
             "PASS" if r4["pass"] else "FAIL"))
    print("       shell profile (exact)    : %s"
          % " ".join("s%d=%.3f" % (s, v) for s, v in enumerate(r4["shell_profile_exact"])))
    print("       shell profile (decomp)   : %s"
          % " ".join("s%d=%.3f" % (s, v) for s, v in enumerate(r4["shell_profile_decomp"])))
    print("       shell profile (40x27 rep): %s"
          % " ".join("s%d=%.3f" % (s, v) for s, v in enumerate(r4["shell_profile_blocks40"])))
    all_ok = r1["pass"] and r2["pass"] and r3["pass"] and r4["pass"]
    print("-" * 78)
    print("REFERENCE LADDER F1-F4 VERDICT: %s" % ("PASS" if all_ok else "FAIL"))
    print("=" * 78)
    return all_ok, {"F1": r1, "F2": r2, "F3": r3, "F4": r4}


def main():
    ap = argparse.ArgumentParser(
        description="Exact free-fermion flow reference engine (Rungs F1-F4)")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    with open(a.manifest, "r") as fh:
        manifest = json.load(fh)

    ok, _ = run_reference_ladder(manifest)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
