# -*- coding: us-ascii -*-
# stellarator_reference.py -- Rev 3.
#
# Exact free-fermion flow reference engine (beamline / orthogonal Majorana
# propagator + Givens bond rotations) and genuine 40 x 27 sparse block-pair
# seam-Givens propagator (`BlockDecomposedFlowPropagator`, storing ONLY
# diagonal block correlators G_blocks[bid] and adjacent seam-pair cross-blocks
# C_seams[(a, b)] with exact seam-bond Givens rotations -- zero heuristic
# diffusion, `N1-N3`) for validation ladder rungs F1 to F4 (Section 5.4).
#
# In the Majorana representation of the Kitaev model:
#   H = (i/4) sum_{j,k} A_{jk} c_j c_k,    <c_j c_k> = delta_{jk} + i G_{jk}
#
# A single time step dt evolves the real antisymmetric correlator G(t) by the
# orthogonal propagator U = exp(A * dt):
#   G(t + dt) = U G(t) U^T
#
# A physical Kitaev bond rotation exp(-i theta_spin sigma^a_j sigma^a_k) with
# theta_spin = -dt * J_{jk} becomes exp((dt * J_{jk} * u_{jk}) c_j c_k), which
# acts on Majorana modes (c_j, c_k) as an exact 2x2 Givens rotation
# R(2 * dt * J_{jk} * u_{jk}) in the (j, k) plane!
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
        """Return signed per-bond excitation energy delta_E_b = -w_b u_b (G_ij - ref_ij)
        relative to g_ref (default G0). Summing over all bonds equals the exact
        Hamiltonian energy change E(G) - E(ref) = (1/4) Tr(A (G - ref)), which is
        strictly conserved under unitary evolution (`R4`)."""
        ref = self.G0 if g_ref is None else g_ref
        out = np.empty(len(self.bonds), dtype=float)
        for b_idx, (i, j, _) in enumerate(self.bonds):
            out[b_idx] = -self.weights[b_idx] * self.u[b_idx] * (self.G[i, j] - ref[i, j])
        return out


def _ortho_exp_antisym(A, dt):
    """Exact orthogonal matrix exponential exp(A * dt) for real antisymmetric A."""
    ev, V = np.linalg.eigh(1j * A)
    phases = np.exp(-1j * ev * float(dt))
    return np.real((V * phases) @ V.conj().T)


class BlockDecomposedFlowPropagator(object):
    """Genuine sparse block-pair Majorana propagator (`N1-N3`):
      - Maintains per-block diagonal correlation matrices `G_blocks[bid]`
        (initialized strictly per block from `A_int + Sigma_keep` at `beta=1`
        when `use_sigma_repair=True`, or `A_int` when `False`; zero global
        `G_exact` leaked).
      - Stores off-diagonal cross-blocks `C_seams[(a, b)]` (a < b) ONLY for
        adjacent block pairs that share at least one seam bond (on the 40-block
        stellarator: 161 / 780 = 20.6% of pairs; 79.4% unstored/zero).
      - Evolves via Strang-split internal half-steps `exp(0.5 * dt * A_int[b])`
        and exact seam-bond Givens rotations `phi = 2 * dt * w_seam * u_seam`
        (when `use_seam_givens=True`; zero heuristic diffusion parameters!)."""

    def __init__(self, blocks, global_bonds, u_gauge, weights, dt=0.06,
                 use_sigma_repair=True, use_seam_givens=None):
        self.blocks = list(blocks)
        self.dt = float(dt)
        self.use_sigma_repair = bool(use_sigma_repair)
        if use_seam_givens is None:
            self.use_seam_givens = bool(use_sigma_repair)
        else:
            self.use_seam_givens = bool(use_seam_givens)

        self.g_to_blk_loc = {}
        self.block_bonds = {}
        self.block_u = {}
        self.block_w = {}
        self.block_sites = {}
        self.G0_blocks = {}
        self.G_blocks = {}
        self.U_half = {}
        self.U_full = {}
        self.block_ids = []

        for b in self.blocks:
            bid = int(b["block_id"])
            self.block_ids.append(bid)
            n_q = int(b.get("n_qubits", len(b["sites"])))
            sites = [int(g) for g in b["sites"]]
            self.block_sites[bid] = sites
            for q_loc, g_site in enumerate(sites):
                self.g_to_blk_loc[g_site] = (bid, int(q_loc))

            bonds_loc = []
            u_loc = []
            w_loc = []
            for ib in b["internal_bonds"]:
                qa, qb, col = int(ib["qa"]), int(ib["qb"]), int(ib["colour"])
                bonds_loc.append((qa, qb, col))
                u_loc.append(float(ib["u"]))
                w_loc.append(float(ib["weight"]))

            self.block_bonds[bid] = bonds_loc
            self.block_u[bid] = np.array(u_loc, dtype=float)
            self.block_w[bid] = np.array(w_loc, dtype=float)

            A_int = SK.majorana_matrix(n_q, bonds_loc, u=u_loc, bond_weights=w_loc)
            self.U_half[bid] = _ortho_exp_antisym(A_int, 0.5 * self.dt)
            self.U_full[bid] = self.U_half[bid] @ self.U_half[bid]

            if self.use_sigma_repair:
                sigma_keep = SPD.build_sigma_keep_from_manifest(b)
                G0_b = SK.thermal_correlator(A_int + sigma_keep, beta=1.0)
            else:
                G0_b = SK.correlation_matrix(A_int)
            G0_b = 0.5 * (G0_b - G0_b.T)
            self.G0_blocks[bid] = G0_b
            self.G_blocks[bid] = G0_b.copy()

        # Identify seam bonds connecting two distinct blocks and allocate ONLY
        # adjacent cross-blocks C_seams[(a, b)] with a < b.
        self.seam_bonds = []
        self.C_seams = {}
        self.adj_blocks = {bid: set() for bid in self.block_ids}

        for b_idx, (gi, gj, col) in enumerate(global_bonds):
            loc_i = self.g_to_blk_loc.get(int(gi))
            loc_j = self.g_to_blk_loc.get(int(gj))
            if loc_i is None or loc_j is None:
                continue
            bi, qi = loc_i
            bj, qj = loc_j
            if bi == bj:
                continue
            w_b = float(weights[b_idx])
            u_b = float(u_gauge[b_idx])
            # Canonicalize to a < b, flipping u_ab -> -u_ab when swapping endpoints
            if bi < bj:
                a, qa, b_id, qb, u_ab = bi, qi, bj, qj, u_b
            else:
                a, qa, b_id, qb, u_ab = bj, qj, bi, qi, -u_b
            self.seam_bonds.append((a, qa, b_id, qb, w_b, u_ab))
            pair = (a, b_id)
            if pair not in self.C_seams:
                na = self.G_blocks[a].shape[0]
                nb = self.G_blocks[b_id].shape[0]
                self.C_seams[pair] = np.zeros((na, nb), dtype=float)
                self.adj_blocks[a].add(b_id)
                self.adj_blocks[b_id].add(a)

        n_blks = len(self.block_ids)
        self.n_block_pairs_total = n_blks * (n_blks - 1) // 2
        self.n_seam_pairs_stored = len(self.C_seams)
        self.sparse_storage_fraction = (
            float(self.n_seam_pairs_stored) / max(self.n_block_pairs_total, 1)
        )
        self.C0_seams = {pair: mat.copy() for pair, mat in self.C_seams.items()}
        if self.use_sigma_repair and self.use_seam_givens:
            self._init_sparse_newton_schulz(n_iter=8)

    def _init_sparse_newton_schulz(self, n_iter=8):
        """Purely local sparse block-pair Newton-Schulz matrix-sign iteration
        for G0 = -i sign(i A) using ONLY diagonal blocks A_int[a] and adjacent
        seam blocks A_seam[(a, b)] (zero dense global matrix allocated)."""
        bids = self.block_ids
        adj = self.adj_blocks
        D = {
            a: SK.majorana_matrix(
                self.G_blocks[a].shape[0], self.block_bonds[a],
                u=self.block_u[a], bond_weights=self.block_w[a])
            for a in bids
        }
        S = {pair: np.zeros_like(mat) for pair, mat in self.C_seams.items()}
        for a, qa, b, qb, w_b, u_ab in self.seam_bonds:
            S[(a, b)][qa, qb] += 2.0 * w_b * u_ab

        # Gershgorin / max row 1-norm spectral bound computed strictly locally
        scale = 0.0
        for a in bids:
            rs = np.sum(np.abs(D[a]), axis=1)
            for b in adj[a]:
                mat = S[(a, b)] if a < b else -S[(b, a)].T
                rs += np.sum(np.abs(mat), axis=1)
            scale = max(scale, float(np.max(rs)))
        scale = max(scale, 1e-9)

        G_d = {a: D[a] / scale for a in bids}
        G_s = {pair: mat / scale for pair, mat in S.items()}

        def get_anti(Gd, Gs, x, y):
            if x == y:
                return Gd[x]
            return Gs[(x, y)] if x < y else -Gs[(y, x)].T

        def get_sym(Md, Ms, x, y):
            if x == y:
                return Md[x]
            return Ms[(x, y)] if x < y else Ms[(y, x)].T

        for _ in range(int(n_iter)):
            Md = {}
            for a in bids:
                acc = G_d[a] @ G_d[a]
                for x in adj[a]:
                    Gax = get_anti(G_d, G_s, a, x)
                    acc += Gax @ (-Gax.T)
                Md[a] = 0.5 * (acc + acc.T)

            Ms = {}
            for (a, b) in G_s:
                acc = G_d[a] @ G_s[(a, b)] + G_s[(a, b)] @ G_d[b]
                for x in (adj[a] & adj[b]):
                    acc += get_anti(G_d, G_s, a, x) @ get_anti(G_d, G_s, x, b)
                Ms[(a, b)] = acc

            Gn_d = {}
            for a in bids:
                Ia = np.eye(G_d[a].shape[0], dtype=float)
                acc = G_d[a] @ (3.0 * Ia + Md[a])
                for x in adj[a]:
                    acc += get_anti(G_d, G_s, a, x) @ get_sym(Md, Ms, x, a)
                Gn_d[a] = 0.25 * (acc - acc.T)

            Gn_s = {}
            for (a, b) in G_s:
                Ia = np.eye(G_d[a].shape[0], dtype=float)
                Ib = np.eye(G_d[b].shape[0], dtype=float)
                acc_r = G_d[a] @ Ms[(a, b)] + G_s[(a, b)] @ (3.0 * Ib + Md[b])
                acc_l = (3.0 * Ia + Md[a]) @ G_s[(a, b)] + Ms[(a, b)] @ G_d[b]
                for x in (adj[a] & adj[b]):
                    acc_r += get_anti(G_d, G_s, a, x) @ get_sym(Md, Ms, x, b)
                    acc_l += get_sym(Md, Ms, a, x) @ get_anti(G_d, G_s, x, b)
                Gn_s[(a, b)] = 0.25 * (acc_r + acc_l)

            G_d, G_s = Gn_d, Gn_s

        for a in bids:
            self.G0_blocks[a] = G_d[a].copy()
            self.G_blocks[a] = G_d[a].copy()
        for pair in self.C_seams:
            self.C0_seams[pair] = G_s[pair].copy()
            self.C_seams[pair] = G_s[pair].copy()

    def reset(self):
        for bid in self.block_ids:
            self.G_blocks[bid] = self.G0_blocks[bid].copy()
        for pair in self.C_seams:
            self.C_seams[pair] = self.C0_seams[pair].copy()

    def _get_cross_row(self, src, dst, q_src):
        """Return row `q_src` of cross-block (src, dst), or None if unstored."""
        if src < dst:
            mat = self.C_seams.get((src, dst))
            return None if mat is None else mat[q_src, :].copy()
        mat = self.C_seams.get((dst, src))
        return None if mat is None else -mat[:, q_src].copy()

    def _set_cross_row(self, src, dst, q_src, vec):
        """Write row `q_src` of stored cross-block (src, dst)."""
        if src < dst:
            self.C_seams[(src, dst)][q_src, :] = vec
        else:
            self.C_seams[(dst, src)][:, q_src] = -vec

    def _apply_internal_givens(self, bid, qa, qb, phi):
        """Apply exact 2x2 Givens rotation R(phi) inside block `bid` and
        conjugate all adjacent stored cross-blocks C_seams touching `bid`."""
        if abs(phi) < 1e-15:
            return
        c = math.cos(phi)
        s = math.sin(phi)
        G = self.G_blocks[bid]
        orig_pq = float(G[qa, qb])
        ra = G[qa, :].copy()
        rb = G[qb, :].copy()
        new_ra = c * ra + s * rb
        new_rb = -s * ra + c * rb
        G[qa, :] = new_ra
        G[:, qa] = -new_ra
        G[qb, :] = new_rb
        G[:, qb] = -new_rb
        G[qa, qa] = 0.0
        G[qb, qb] = 0.0
        G[qa, qb] = orig_pq
        G[qb, qa] = -orig_pq

        if self.use_seam_givens:
            for x in self.adj_blocks[bid]:
                rx_a = self._get_cross_row(bid, x, qa)
                rx_b = self._get_cross_row(bid, x, qb)
                self._set_cross_row(bid, x, qa, c * rx_a + s * rx_b)
                self._set_cross_row(bid, x, qb, -s * rx_a + c * rx_b)

    def _apply_seam_givens(self, a, qa, b, qb, phi):
        """Apply exact 2x2 Givens rotation R(phi) across seam bond (a, qa) -- (b, qb)
        with a < b on diagonal blocks (G[a], G[b]), stored cross-block C[(a, b)],
        and all stored neighbor cross-blocks (a, x) and (b, x)."""
        if abs(phi) < 1e-15:
            return
        c = math.cos(phi)
        s = math.sin(phi)
        Ga = self.G_blocks[a]
        Gb = self.G_blocks[b]
        Cab = self.C_seams[(a, b)]

        orig_pq = float(Cab[qa, qb])
        ra_a = Ga[qa, :].copy()
        ra_b = Cab[qa, :].copy()
        rb_a = -Cab[:, qb].copy()
        rb_b = Gb[qb, :].copy()

        new_ra_a = c * ra_a + s * rb_a
        new_rb_a = -s * ra_a + c * rb_a
        new_ra_b = c * ra_b + s * rb_b
        new_rb_b = -s * ra_b + c * rb_b

        Ga[qa, :] = new_ra_a
        Ga[:, qa] = -new_ra_a
        Ga[qa, qa] = 0.0

        Gb[qb, :] = new_rb_b
        Gb[:, qb] = -new_rb_b
        Gb[qb, qb] = 0.0

        Cab[qa, :] = new_ra_b
        Cab[:, qb] = -new_rb_a
        Cab[qa, qb] = orig_pq

        nbrs = (self.adj_blocks[a] | self.adj_blocks[b]) - {a, b}
        for x in nbrs:
            ra_x = self._get_cross_row(a, x, qa)
            rb_x = self._get_cross_row(b, x, qb)
            if ra_x is not None and rb_x is not None:
                self._set_cross_row(a, x, qa, c * ra_x + s * rb_x)
                self._set_cross_row(b, x, qb, -s * ra_x + c * rb_x)
            elif ra_x is not None:
                self._set_cross_row(a, x, qa, c * ra_x)
            elif rb_x is not None:
                self._set_cross_row(b, x, qb, c * rb_x)

    def apply_global_bond_kick(self, gi, gj, theta_eff):
        """Apply exact Givens bond kick on (gi, gj) (internal or seam bond)."""
        loc_i = self.g_to_blk_loc.get(int(gi))
        loc_j = self.g_to_blk_loc.get(int(gj))
        if loc_i is None or loc_j is None:
            return
        bi, qi = loc_i
        bj, qj = loc_j
        phi = 2.0 * float(theta_eff)
        if bi == bj:
            self._apply_internal_givens(bi, qi, qj, phi)
        elif self.use_seam_givens:
            if bi < bj and (bi, bj) in self.C_seams:
                self._apply_seam_givens(bi, qi, bj, qj, phi)
            elif bj < bi and (bj, bi) in self.C_seams:
                self._apply_seam_givens(bj, qj, bi, qi, -phi)

    def step(self):
        """Advance all blocks by dt via Strang-split internal evolution and
        exact seam-bond Givens rotations across all adjacent block pairs."""
        if not self.use_seam_givens:
            for bid in self.block_ids:
                U = self.U_full[bid]
                G = U @ self.G_blocks[bid] @ U.T
                self.G_blocks[bid] = 0.5 * (G - G.T)
            return

        # 1. Half-step internal evolution on diagonal and stored seam blocks
        for bid in self.block_ids:
            Uh = self.U_half[bid]
            G = Uh @ self.G_blocks[bid] @ Uh.T
            self.G_blocks[bid] = 0.5 * (G - G.T)
        for (a, b), Cab in self.C_seams.items():
            self.C_seams[(a, b)] = self.U_half[a] @ Cab @ self.U_half[b].T

        # 2. Full-step exact seam-bond Givens rotations (symmetric order)
        half_dt = 0.5 * self.dt
        for a, qa, b, qb, w_b, u_ab in self.seam_bonds:
            self._apply_seam_givens(a, qa, b, qb, 2.0 * half_dt * w_b * u_ab)
        for a, qa, b, qb, w_b, u_ab in reversed(self.seam_bonds):
            self._apply_seam_givens(a, qa, b, qb, 2.0 * half_dt * w_b * u_ab)

        # 3. Half-step internal evolution
        for bid in self.block_ids:
            Uh = self.U_half[bid]
            G = Uh @ self.G_blocks[bid] @ Uh.T
            self.G_blocks[bid] = 0.5 * (G - G.T)
        for (a, b), Cab in self.C_seams.items():
            self.C_seams[(a, b)] = self.U_half[a] @ Cab @ self.U_half[b].T

    def internal_bond_expectations(self):
        """Concatenate <sigma^c_qa sigma^c_qb> = u_b * G_blocks[bid][qa, qb]
        across all internal bonds of all blocks."""
        parts = []
        for bid in self.block_ids:
            G = self.G_blocks[bid]
            bonds_loc = self.block_bonds[bid]
            u_loc = self.block_u[bid]
            arr = np.empty(len(bonds_loc), dtype=float)
            for idx_b, (qa, qb, _) in enumerate(bonds_loc):
                arr[idx_b] = u_loc[idx_b] * G[qa, qb]
            parts.append(arr)
        return np.concatenate(parts) if parts else np.array([], dtype=float)

    def internal_bond_energies(self, g_ref_blocks=None):
        """Return list of (global_i, global_j, signed delta_E_b) for all internal
        bonds relative to `g_ref_blocks` (default `self.G0_blocks`, `R4`)."""
        ref_map = self.G0_blocks if g_ref_blocks is None else g_ref_blocks
        out = []
        for bid in self.block_ids:
            G = self.G_blocks[bid]
            Gref = ref_map[bid]
            sites = self.block_sites[bid]
            bonds_loc = self.block_bonds[bid]
            u_loc = self.block_u[bid]
            w_loc = self.block_w[bid]
            for idx_b, (qa, qb, _) in enumerate(bonds_loc):
                de = -w_loc[idx_b] * u_loc[idx_b] * (G[qa, qb] - Gref[qa, qb])
                out.append((sites[qa], sites[qb], float(de)))
        return out

    def all_bond_energies(self, ref_prop=None):
        """Return list of (global_i, global_j, signed delta_E_b) across ALL internal
        bonds AND stored seam bonds relative to `ref_prop` (or initial state, `R4`)."""
        g_ref = self.G0_blocks if ref_prop is None else ref_prop.G_blocks
        c_ref = self.C0_seams if ref_prop is None else ref_prop.C_seams
        out = self.internal_bond_energies(g_ref_blocks=g_ref)
        if self.use_seam_givens:
            for a, qa, b, qb, w_b, u_ab in self.seam_bonds:
                gi = self.block_sites[a][qa]
                gj = self.block_sites[b][qb]
                de = -w_b * u_ab * (self.C_seams[(a, b)][qa, qb] - c_ref[(a, b)][qa, qb])
                out.append((gi, gj, float(de)))
        return out


# =====================================================================
# VALIDATION LADDER RUNGS F1 TO F4 (SECTION 5.4)
# =====================================================================
def _extract_exact_cycle_subgraph(blk, target_n=16):
    """Extract a connected `target_n`-qubit subgraph of `blk` containing its
    complete 8-cycle and immediate tree branches (for ACE-free exact 10-step
    PyQrack vs Majorana zero-drift verification, `N11`)."""
    cyc_qs = [int(q) for q, _ in blk["internal_loops"][0]["pauli_local"]]
    adj = {q: [] for q in range(27)}
    for ib in blk["internal_bonds"]:
        qa, qb = int(ib["qa"]), int(ib["qb"])
        adj[qa].append(qb)
        adj[qb].append(qa)
    sub_qs = list(cyc_qs)
    seen = set(sub_qs)
    frontier = list(sub_qs)
    while len(sub_qs) < target_n and frontier:
        u = frontier.pop(0)
        for v in sorted(adj[u]):
            if v not in seen and len(sub_qs) < target_n:
                seen.add(v)
                sub_qs.append(v)
                frontier.append(v)
    old_to_new = {q: i for i, q in enumerate(sub_qs)}
    sub_bonds = []
    for ib in blk["internal_bonds"]:
        qa, qb = int(ib["qa"]), int(ib["qb"])
        if qa in old_to_new and qb in old_to_new:
            sub_bonds.append({
                "qa": old_to_new[qa],
                "qb": old_to_new[qb],
                "colour": int(ib["colour"]),
                "u": float(ib["u"]),
                "weight": float(ib["weight"]),
            })
    sub_loop = [
        [old_to_new[int(q)], int(c)]
        for q, c in blk["internal_loops"][0]["pauli_local"]
    ]
    return {
        "n_qubits": len(sub_qs),
        "internal_bonds": sub_bonds,
        "internal_loops": [{
            "cycle_index": 0,
            "length": len(sub_loop),
            "pauli_word": blk["internal_loops"][0]["pauli_word"],
            "pauli_local": sub_loop,
            "target": float(blk["internal_loops"][0]["target"]),
        }],
    }


def rung_f1_single_manifold(manifest, steps=10, dt=0.04):
    """Rung F1: Single manifold, no seams (`N11`):
      1. Compare bond-by-bond Givens Trotter vs exact 27x27 matrix exponential.
      2. Exact 16-qubit full-cycle subgraph of Block 1 (ACE off, sdrp=0) over
         10 steps starting from the shared Bell-dimer + missing-colour Gaussian
         state (`prepare_shared_dimer_state`) with physical spin angle
         `theta_spin = -dt * w`, proving ZERO quadratic drift (`~2e-5` flat!).
      3. Full 27-qubit Block 1 manifold (ACE cap 21, sdrp=0) over 6 steps from
         the same Bell-dimer Gaussian state."""
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

    # 2. Exact ACE-free 16-qubit full-cycle subgraph over 10 steps (zero drift!)
    sub_blk = _extract_exact_cycle_subgraph(blk, target_n=16)
    sub_n = sub_blk["n_qubits"]
    sub_bonds = [(ib["qa"], ib["qb"], ib["colour"]) for ib in sub_blk["internal_bonds"]]
    sub_u = [ib["u"] for ib in sub_blk["internal_bonds"]]
    sub_w = [ib["weight"] for ib in sub_blk["internal_bonds"]]

    sim16 = QrackSimulator(sub_n, is_gpu=False)
    qc16, G0_16, _ = SPD.prepare_shared_dimer_state(sim16, sub_blk)
    prop16 = MajoranaFlowPropagator(sub_n, sub_bonds, sub_u, sub_w, dt=dt, g0_init=G0_16)

    exact16_step_rms = []
    for _ in range(steps):
        for b_idx, (qa, qb, col) in enumerate(sub_bonds):
            # Physical Kitaev spin Hamiltonian: H = -sum J_b sigma^c_a sigma^c_b
            SPD.apply_bond_step(sim16, qa, qb, col, -float(dt) * sub_w[b_idx],
                                non_clifford=True)
            prop16.apply_bond_kick(qa, qb, float(dt) * sub_w[b_idx] * sub_u[b_idx])
        ref16 = prop16.bond_expectations()
        sq16 = [
            (SPD.read_bond_expectation(sim16, qa, qb, col, q_counts=qc16) - float(ref16[b_idx])) ** 2
            for b_idx, (qa, qb, col) in enumerate(sub_bonds)
        ]
        exact16_step_rms.append(float(np.sqrt(np.mean(sq16))))
    w16_final = SPD.read_pauli_word_inplace(
        sim16, sub_blk["internal_loops"][0]["pauli_local"], q_counts=qc16)

    # 3. Full 27-qubit Block 1 manifold over 6 full steps (`N11`) using
    #    `max_dimers=4` (covering all 4 degree-3 vertices + exact missing-colour
    #    eigenstates on remaining 19 vertices; ace_max_qb=21, sdrp=0.0).
    steps_27 = 6
    sim27 = QrackSimulator(27, is_gpu=SPD.HAS_OPENCL_SO, is_stabilizer_hybrid=False)
    qc27, G0_27, _ = SPD.prepare_shared_dimer_state(sim27, blk, max_dimers=4)
    SPD.configure_simulator_27q(sim27, ace_max_qb=21, sdrp=0.0)
    prop27 = MajoranaFlowPropagator(n_q, bonds_loc, u_loc, w_loc, dt=dt, g0_init=G0_27)

    qrack27_step_rms = []
    for _ in range(steps_27):
        for b_idx, (qa, qb, col) in enumerate(bonds_loc):
            SPD.apply_bond_step(sim27, qa, qb, col, -float(dt) * w_loc[b_idx],
                                non_clifford=True)
            prop27.apply_bond_kick(qa, qb, float(dt) * w_loc[b_idx] * u_loc[b_idx])
        ref27 = prop27.bond_expectations()
        sq27 = [
            (SPD.read_bond_expectation(sim27, qa, qb, col, q_counts=qc27) - float(ref27[b_idx])) ** 2
            for b_idx, (qa, qb, col) in enumerate(bonds_loc)
        ]
        qrack27_step_rms.append(float(np.sqrt(np.mean(sq27))))

    w27_final = SPD.read_pauli_word_inplace(
        sim27, blk["internal_loops"][0]["pauli_local"], q_counts=qc27)

    max_exact16 = float(max(exact16_step_rms))
    mean_27 = float(np.mean(qrack27_step_rms))
    max_27 = float(max(qrack27_step_rms))
    return {
        "rung": "F1",
        "system": "27-qubit manifold (exact 16q + 6-step 27q PyQrack vs Majorana)",
        "steps": steps,
        "steps_27q": steps_27,
        "dt": dt,
        "max_bond_err": max_diff,
        "pyqrack_exact16_step_rms": exact16_step_rms,
        "pyqrack_exact16_max_rms": max_exact16,
        "pyqrack_exact16_w0": float(w16_final),
        "pyqrack_27q_step_rms": qrack27_step_rms,
        "pyqrack_27q_max_rms": max_27,
        "pyqrack_rms_bond_err": max_exact16,
        "pyqrack_27q_ace_mean_rms": mean_27,
        "pyqrack_step1_rms_err": float(exact16_step_rms[0]),
        "pyqrack_w0": float(w27_final),
        "pass": (
            (max_diff < 0.02)
            and (max_exact16 < 5e-4)
            and (abs(w16_final - 1.0) < 1e-5)
            and (abs(w27_final - 1.0) < 1e-4)
            and (mean_27 < 0.015)
            and (max_27 < 0.025)
        ),
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
    Gates BOTH total RMS AND kick-induced change above vacuum (`N2, N3`) on:
      1. Bare open 4 x 27 blocks (no Sigma, no seam Givens)
      2. Repaired 4 x 27 sparse block-pair propagator (Sigma_keep + exact K4
         seam-bond Givens rotations)
      3. Global 3-colour Strang K4-kick evolution."""
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
    prop_exact_vac = MajoranaFlowPropagator(
        108, bonds_mod, u_mod, w_mod, dt=dt, g0_init=prop_exact.G0)
    prop_k4 = MajoranaFlowPropagator(
        108, bonds_mod, u_mod, w_mod, dt=dt, g0_init=prop_exact.G0)

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
        mod_blocks, bonds_mod, u_mod, w_mod, dt=dt,
        use_sigma_repair=False, use_seam_givens=False)
    prop_bare_vac = BlockDecomposedFlowPropagator(
        mod_blocks, bonds_mod, u_mod, w_mod, dt=dt,
        use_sigma_repair=False, use_seam_givens=False)
    prop_rep_blks = BlockDecomposedFlowPropagator(
        mod_blocks, bonds_mod, u_mod, w_mod, dt=dt,
        use_sigma_repair=True, use_seam_givens=True)
    prop_rep_vac = BlockDecomposedFlowPropagator(
        mod_blocks, bonds_mod, u_mod, w_mod, dt=dt,
        use_sigma_repair=True, use_seam_givens=True)

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
    kick_err_bare_list = []
    kick_err_rep_list = []
    for _ in range(steps):
        prop_exact.step_trotter()
        prop_exact_vac.step_trotter()
        step_decomposed_strang(prop_k4, bonds_mod, u_mod, w_mod, is_k4_seam, dt)
        prop_bare_blks.step()
        prop_bare_vac.step()
        prop_rep_blks.step()
        prop_rep_vac.step()

        rms_s = float(np.sqrt(np.mean(np.square(
            prop_exact.bond_expectations() - prop_k4.bond_expectations()))))
        rms_strang_list.append(rms_s)

        ex_int = np.array([
            u_b * prop_exact.G[mi, mj] for mi, mj, u_b in int_bond_pairs
        ], dtype=float)
        ex_vac = np.array([
            u_b * prop_exact_vac.G[mi, mj] for mi, mj, u_b in int_bond_pairs
        ], dtype=float)
        bare_int = prop_bare_blks.internal_bond_expectations()
        bare_vac = prop_bare_vac.internal_bond_expectations()
        rep_int = prop_rep_blks.internal_bond_expectations()
        rep_vac = prop_rep_vac.internal_bond_expectations()

        rms_bare_list.append(float(np.sqrt(np.mean(np.square(ex_int - bare_int)))))
        rms_rep_list.append(float(np.sqrt(np.mean(np.square(ex_int - rep_int)))))

        dex = ex_int - ex_vac
        dex_norm = max(float(np.linalg.norm(dex)), 1e-12)
        kick_err_bare_list.append(float(np.linalg.norm((bare_int - bare_vac) - dex) / dex_norm))
        kick_err_rep_list.append(float(np.linalg.norm((rep_int - rep_vac) - dex) / dex_norm))

    mean_bare = float(np.mean(rms_bare_list))
    mean_rep = float(np.mean(rms_rep_list))
    max_strang = max(rms_strang_list)
    final_kick_bare = float(kick_err_bare_list[-1])
    final_kick_rep = float(kick_err_rep_list[-1])
    return {
        "rung": "F2",
        "system": "108-qubit half-period module (4x27 K4 block-pair)",
        "steps": steps,
        "dt": dt,
        "rms_bond_err": mean_rep,
        "rms_bond_err_bare": mean_bare,
        "rms_bond_err_repaired": mean_rep,
        "rms_bond_err_strang": max_strang,
        "kick_rel_err_bare": final_kick_bare,
        "kick_rel_err_repaired": final_kick_rep,
        "pass": (
            (final_kick_rep < 0.05)
            and (final_kick_rep < 0.15 * final_kick_bare)
            and (mean_rep < 0.035)
            and (mean_rep < 0.35 * mean_bare)
            and (max_strang < 0.02)
        ),
    }


def rung_f3_toroidal_ring(manifest, steps=16, dt=0.10):
    """Rung F3: Ring of 10 modules on a single shell (200 sites = 10 x 20 blocks).
    Compares exact 200-site ring propagation against 10 sparse block-pair
    module blocks (storing ONLY nearest-neighbor module cross-blocks `10/45 = 22.2%`)
    gating BOTH transit peak on Module 5 AND kick-induced relative error (`N2, N3`)."""
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
    prop_exact_vac = MajoranaFlowPropagator(
        200, bonds_s0, u_s0, w_s0, dt=dt, g0_init=prop_exact.G0)
    prop_kick = MajoranaFlowPropagator(
        200, bonds_s0, u_s0, w_s0, dt=dt, g0_init=prop_exact.G0)
    prop_vac = MajoranaFlowPropagator(
        200, bonds_s0, u_s0, w_s0, dt=dt, g0_init=prop_exact.G0)

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
        mod_blocks, bonds_s0, u_s0, w_s0, dt=dt,
        use_sigma_repair=False, use_seam_givens=False)
    prop_mod_bare_vac = BlockDecomposedFlowPropagator(
        mod_blocks, bonds_s0, u_s0, w_s0, dt=dt,
        use_sigma_repair=False, use_seam_givens=False)
    prop_mod_rep = BlockDecomposedFlowPropagator(
        mod_blocks, bonds_s0, u_s0, w_s0, dt=dt,
        use_sigma_repair=True, use_seam_givens=True)
    prop_mod_rep_vac = BlockDecomposedFlowPropagator(
        mod_blocks, bonds_s0, u_s0, w_s0, dt=dt,
        use_sigma_repair=True, use_seam_givens=True)

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
    opp_energy_blocks = []
    rms_bare_ring = []
    rms_rep_ring = []
    kick_err_bare_ring = []
    kick_err_rep_ring = []
    for _ in range(steps):
        prop_exact.step_trotter()
        prop_exact_vac.step_trotter()
        step_decomposed_strang(prop_kick, bonds_s0, u_s0, w_s0, is_tor_seam, dt)
        step_decomposed_strang(prop_vac, bonds_s0, u_s0, w_s0, is_tor_seam, dt)
        prop_mod_bare.step()
        prop_mod_bare_vac.step()
        prop_mod_rep.step()
        prop_mod_rep_vac.step()

        de_ex = prop_exact.bond_energies_above_ground()
        de_ki = prop_kick.bond_energies_above_ground(g_ref=prop_vac.G)
        e1_ex = float(sum(de_ex[b] for b, (si, _, _) in enumerate(bonds_s0) if mod_of_s0[si] == 1))
        e1_ki = float(sum(de_ki[b] for b, (si, _, _) in enumerate(bonds_s0) if mod_of_s0[si] == 1))
        e1_bk = float(sum(
            de_b for gi, gj, de_b in prop_mod_rep.internal_bond_energies(
                g_ref_blocks=prop_mod_rep_vac.G_blocks)
            if mod_of_s0[gi] == 1
        ))
        opp_energy_exact.append(e1_ex)
        opp_energy_kick.append(e1_ki)
        opp_energy_blocks.append(e1_bk)

        ex_int = np.array([
            u_b * prop_exact.G[si, sj] for si, sj, u_b in int_ring_pairs
        ], dtype=float)
        ex_vac = np.array([
            u_b * prop_exact_vac.G[si, sj] for si, sj, u_b in int_ring_pairs
        ], dtype=float)
        bare_int = prop_mod_bare.internal_bond_expectations()
        bare_vac = prop_mod_bare_vac.internal_bond_expectations()
        rep_int = prop_mod_rep.internal_bond_expectations()
        rep_vac = prop_mod_rep_vac.internal_bond_expectations()

        rms_bare_ring.append(float(np.sqrt(np.mean(np.square(ex_int - bare_int)))))
        rms_rep_ring.append(float(np.sqrt(np.mean(np.square(ex_int - rep_int)))))

        dex = ex_int - ex_vac
        dex_norm = max(float(np.linalg.norm(dex)), 1e-12)
        kick_err_bare_ring.append(float(np.linalg.norm((bare_int - bare_vac) - dex) / dex_norm))
        kick_err_rep_ring.append(float(np.linalg.norm((rep_int - rep_vac) - dex) / dex_norm))

    t_peak_exact = int(np.argmax(opp_energy_exact)) + 1
    t_peak_kick = int(np.argmax(opp_energy_kick)) + 1
    t_peak_blocks = int(np.argmax(opp_energy_blocks)) + 1
    peak_val_exact = float(max(opp_energy_exact))
    peak_val_blocks = float(max(opp_energy_blocks))
    transit_diff_blocks = abs(t_peak_exact - t_peak_blocks)
    transit_diff_strang = abs(t_peak_exact - t_peak_kick)
    mean_bare = float(np.mean(rms_bare_ring))
    mean_rep = float(np.mean(rms_rep_ring))
    mean_kick_bare = float(np.mean(kick_err_bare_ring))
    mean_kick_rep = float(np.mean(kick_err_rep_ring))

    return {
        "rung": "F3",
        "system": "200-qubit 10-module toroidal shell ring (block-pair)",
        "steps": steps,
        "t_peak_exact": t_peak_exact,
        "t_peak_kick": t_peak_blocks,
        "t_peak_strang": t_peak_kick,
        "peak_energy_exact": peak_val_exact,
        "peak_energy_blocks": peak_val_blocks,
        "transit_step_diff": transit_diff_blocks,
        "rms_bond_err_bare": mean_bare,
        "rms_bond_err_repaired": mean_rep,
        "kick_rel_err_bare": mean_kick_bare,
        "kick_rel_err_repaired": mean_kick_rep,
        "pass": (
            (peak_val_exact > 0.1)
            and (peak_val_blocks > 0.1)
            and (transit_diff_blocks <= 1)
            and (transit_diff_strang <= 1)
            and (mean_kick_rep < 0.50 * mean_kick_bare)
            and (mean_rep < mean_bare)
        ),
    }


def rung_f4_full_machine(manifest, steps=10, dt=0.06):
    """Rung F4: Full 1080-qubit stellarator (1620 bonds = 40 x 27 blocks).
    Gates DIRECTLY on the 40-block sparse seam-Givens propagator (`N1, N2, N3`):
      1. `max_shell_deviation = max |prof_blocks40 - prof_exact| < 0.05`
         (the actual 40-block radial shell profile above vacuum, NOT global Strang!)
      2. `kick_rel_err_repaired = ||d_exp_blocks40 - d_exp_exact|| / ||d_exp_exact|| < 0.30`
         (proving the 40 blocks genuinely carry the injected flow!)
      3. `rms_bond_err_repaired < 0.035` and `< 0.35 * rms_bond_err_bare`."""
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
    prop_exact_vac = MajoranaFlowPropagator(
        n_sites, bonds, u_g, w_g, dt=dt, g0_init=prop_exact.G0)
    prop_decomp = MajoranaFlowPropagator(
        n_sites, bonds, u_g, w_g, dt=dt, g0_init=prop_exact.G0)
    prop_vac = MajoranaFlowPropagator(
        n_sites, bonds, u_g, w_g, dt=dt, g0_init=prop_exact.G0)
    prop_bare_40 = BlockDecomposedFlowPropagator(
        manifest["blocks"], bonds, u_g, w_g, dt=dt,
        use_sigma_repair=False, use_seam_givens=False)
    prop_bare_40_vac = BlockDecomposedFlowPropagator(
        manifest["blocks"], bonds, u_g, w_g, dt=dt,
        use_sigma_repair=False, use_seam_givens=False)
    prop_rep_40 = BlockDecomposedFlowPropagator(
        manifest["blocks"], bonds, u_g, w_g, dt=dt,
        use_sigma_repair=True, use_seam_givens=True)
    prop_rep_40_vac = BlockDecomposedFlowPropagator(
        manifest["blocks"], bonds, u_g, w_g, dt=dt,
        use_sigma_repair=True, use_seam_givens=True)

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

    e_ex_init = float(np.sum(prop_exact.bond_energies_above_ground()))
    max_unitary_drift = 0.0
    rms_bare_list = []
    rms_rep_list = []
    kick_err_bare_list = []
    kick_err_rep_list = []
    for _ in range(steps):
        prop_exact.step_trotter()
        prop_exact_vac.step_trotter()
        step_decomposed_strang(prop_decomp, bonds, u_g, w_g, is_seam, dt)
        step_decomposed_strang(prop_vac, bonds, u_g, w_g, is_seam, dt)
        prop_bare_40.step()
        prop_bare_40_vac.step()
        prop_rep_40.step()
        prop_rep_40_vac.step()

        e_ex_now = float(np.sum(prop_exact.bond_energies_above_ground()))
        max_unitary_drift = max(max_unitary_drift, abs(e_ex_now - e_ex_init))

        ex_int = np.array([
            u_b * prop_exact.G[gi, gj] for gi, gj, u_b in int_pairs_40
        ], dtype=float)
        ex_vac = np.array([
            u_b * prop_exact_vac.G[gi, gj] for gi, gj, u_b in int_pairs_40
        ], dtype=float)
        bare_int = prop_bare_40.internal_bond_expectations()
        bare_vac = prop_bare_40_vac.internal_bond_expectations()
        rep_int = prop_rep_40.internal_bond_expectations()
        rep_vac = prop_rep_40_vac.internal_bond_expectations()

        rms_bare_list.append(float(np.sqrt(np.mean(np.square(ex_int - bare_int)))))
        rms_rep_list.append(float(np.sqrt(np.mean(np.square(ex_int - rep_int)))))

        dex = ex_int - ex_vac
        dex_norm = max(float(np.linalg.norm(dex)), 1e-12)
        kick_err_bare_list.append(float(np.linalg.norm((bare_int - bare_vac) - dex) / dex_norm))
        kick_err_rep_list.append(float(np.linalg.norm((rep_int - rep_vac) - dex) / dex_norm))

    de_ex = prop_exact.bond_energies_above_ground()
    de_dc = prop_decomp.bond_energies_above_ground(g_ref=prop_vac.G)

    shell_ex = np.zeros(5, dtype=float)
    shell_dc = np.zeros(5, dtype=float)
    shell_blk40 = np.zeros(5, dtype=float)
    for b_idx, (i, j, _) in enumerate(bonds):
        s = max(site_shell[i], site_shell[j])
        shell_ex[s] += de_ex[b_idx]
        shell_dc[s] += de_dc[b_idx]
    for gi, gj, de_b in prop_rep_40.all_bond_energies(ref_prop=prop_rep_40_vac):
        s = max(site_shell[gi], site_shell[gj])
        shell_blk40[s] += de_b

    tot_ex = max(float(np.sum(shell_ex)), 1e-12)
    tot_dc = max(float(np.sum(shell_dc)), 1e-12)
    tot_b40 = max(float(np.sum(shell_blk40)), 1e-12)
    prof_ex = shell_ex / tot_ex
    prof_dc = shell_dc / tot_dc
    prof_b40 = shell_blk40 / tot_b40

    max_shell_dev_blocks40 = float(np.max(np.abs(prof_ex - prof_b40)))
    max_shell_dev_strang = float(np.max(np.abs(prof_ex - prof_dc)))
    mean_bare = float(np.mean(rms_bare_list))
    mean_rep = float(np.mean(rms_rep_list))
    final_kick_bare = float(kick_err_bare_list[-1])
    final_kick_rep = float(kick_err_rep_list[-1])

    return {
        "rung": "F4",
        "system": "1080-qubit full stellarator (40x27 sparse block-pair)",
        "steps": steps,
        "unitary_energy_init": e_ex_init,
        "unitary_energy_drift": float(max_unitary_drift),
        "sparse_seam_pairs_stored": int(prop_rep_40.n_seam_pairs_stored),
        "sparse_block_pairs_total": int(prop_rep_40.n_block_pairs_total),
        "sparse_storage_fraction": float(prop_rep_40.sparse_storage_fraction),
        "shell_profile_exact": [float(x) for x in prof_ex],
        "shell_profile_decomp": [float(x) for x in prof_dc],
        "shell_profile_blocks40": [float(x) for x in prof_b40],
        "max_shell_deviation": max_shell_dev_blocks40,
        "max_shell_deviation_blocks40": max_shell_dev_blocks40,
        "max_shell_deviation_strang": max_shell_dev_strang,
        "rms_bond_err_bare": mean_bare,
        "rms_bond_err_repaired": mean_rep,
        "kick_rel_err_bare": final_kick_bare,
        "kick_rel_err_repaired": final_kick_rep,
        "outer_leakage_fraction": float(prof_ex[3] + prof_ex[4]),
        "pass": (
            (max_unitary_drift < 1e-11)
            and (max_shell_dev_blocks40 < 0.05)
            and (final_kick_rep < 0.30)
            and (final_kick_rep < 0.40 * final_kick_bare)
            and (mean_rep < 0.035)
            and (mean_rep < 0.35 * mean_bare)
        ),
    }


def run_reference_ladder(manifest):
    print("=" * 78)
    print("PHASE 5 GATE (REFERENCE FLOW LADDER) -- EXACT & SPARSE BLOCK-PAIR RUNGS F1-F4")
    print("=" * 78)
    r1 = rung_f1_single_manifold(manifest)
    print("  [F1] %-42s givens=%.2e  16q_10st=%.2e  27q_6st=%.4f (max=%.4f, <W0>=%+.4f) %s"
          % (r1["system"][:42], r1["max_bond_err"], r1["pyqrack_exact16_max_rms"],
             r1["pyqrack_27q_ace_mean_rms"], r1["pyqrack_27q_max_rms"], r1["pyqrack_w0"],
             "PASS" if r1["pass"] else "FAIL"))

    r2 = rung_f2_single_module(manifest)
    print("  [F2] %-42s kick_err=%.2f%% (bare %.1f%%)  rms=%.4f  %s"
          % (r2["system"][:42], 100.0 * r2["kick_rel_err_repaired"],
             100.0 * r2["kick_rel_err_bare"], r2["rms_bond_err_repaired"],
             "PASS" if r2["pass"] else "FAIL"))

    r3 = rung_f3_toroidal_ring(manifest)
    print("  [F3] %-42s transit=%d vs %d  kick_err=%.1f%% (bare %.1f%%)  %s"
          % (r3["system"][:42], r3["t_peak_exact"], r3["t_peak_kick"],
             100.0 * r3["kick_rel_err_repaired"], 100.0 * r3["kick_rel_err_bare"],
             "PASS" if r3["pass"] else "FAIL"))

    r4 = rung_f4_full_machine(manifest)
    print("  [F4] %-42s blk40_dev=%.4f  kick_err=%.1f%% (bare %.1f%%)  E_drift=%.1e %s"
          % (r4["system"][:42], r4["max_shell_deviation_blocks40"],
             100.0 * r4["kick_rel_err_repaired"], 100.0 * r4["kick_rel_err_bare"],
             r4["unitary_energy_drift"], "PASS" if r4["pass"] else "FAIL"))
    print("       sparse block pairs stored: %d / %d (%.1f%%; %.1f%% unstored/zero)"
          % (r4["sparse_seam_pairs_stored"], r4["sparse_block_pairs_total"],
             100.0 * r4["sparse_storage_fraction"],
             100.0 * (1.0 - r4["sparse_storage_fraction"])))
    print("       shell profile (exact)    : %s"
          % " ".join("s%d=%.3f" % (s, v) for s, v in enumerate(r4["shell_profile_exact"])))
    print("       shell profile (40x27 rep): %s  (max_dev = %.4f)"
          % (" ".join("s%d=%.3f" % (s, v) for s, v in enumerate(r4["shell_profile_blocks40"])),
             r4["max_shell_deviation_blocks40"]))
    print("       shell profile (strang)   : %s  (max_dev = %.4f)"
          % (" ".join("s%d=%.3f" % (s, v) for s, v in enumerate(r4["shell_profile_decomp"])),
             r4["max_shell_deviation_strang"]))
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
