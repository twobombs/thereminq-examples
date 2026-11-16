# -*- coding: us-ascii -*-
# stellarator_reference.py -- Rev 1.
#
# Exact free-fermion flow reference engine (beamline / orthogonal Majorana
# propagator + Givens bond rotations) for validation ladder rungs F1 to F4
# (Section 5.4).
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
import sys

import numpy as np

import stellarator_kitaev as SK


class MajoranaFlowPropagator(object):
    """Exact orthogonal Majorana correlation-matrix propagator on N sites."""

    def __init__(self, n_sites, bonds, u_gauge, weights=None, dt=0.08):
        self.n = int(n_sites)
        self.bonds = list(bonds)
        self.u = np.array(u_gauge, dtype=float)
        self.weights = (np.ones(len(bonds), dtype=float)
                        if weights is None else np.array(weights, dtype=float))
        self.dt = float(dt)

        self.A = SK.majorana_matrix(
            self.n, self.bonds, u=self.u, bond_weights=self.weights)
        self.G0 = SK.correlation_matrix(self.A)
        self.G = self.G0.copy()

        # Precompute orthogonal step matrix U = exp(A * dt) via eigendecomposition of iA
        ev, V = np.linalg.eigh(1j * self.A)
        # Since iA = V diag(ev) V^dagger, A = -i V diag(ev) V^dagger,
        # so exp(A * dt) = V diag(exp(-i * ev * dt)) V^dagger (real orthogonal)
        phases = np.exp(-1j * ev * self.dt)
        U_complex = (V * phases) @ V.conj().T
        self.U_step = np.real(U_complex)

    def reset(self):
        self.G = self.G0.copy()

    def step_trotter(self):
        """Exact unitary step G <- U_step G U_step^T."""
        self.G = self.U_step @ self.G @ self.U_step.T
        self.G = 0.5 * (self.G - self.G.T)

    def apply_bond_kick(self, j, k, theta_eff):
        """Apply exact Givens rotation in the (j, k) Majorana plane:
        exp(phi/2 * c_j c_k) rotates (c_j, c_k) by angle phi = 2 * theta_eff."""
        phi = 2.0 * float(theta_eff)
        c = math.cos(phi)
        s = math.sin(phi)
        # Update rows j, k
        rj = self.G[j, :].copy()
        rk = self.G[k, :].copy()
        self.G[j, :] = c * rj + s * rk
        self.G[k, :] = -s * rj + c * rk
        # Update cols j, k
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


# =====================================================================
# VALIDATION LADDER RUNGS F1 TO F4 (SECTION 5.4)
# =====================================================================
def rung_f1_single_manifold(manifest, steps=8, dt=0.06):
    """Rung F1: One 27-qubit manifold, no seams. Compare first-order bond-by-bond
    Givens Trotter sequence against exact matrix exponential U = exp(A_int * dt)."""
    blk = manifest["blocks"][0]
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

    # Inject a core-bond packet on bond 0
    b0_i, b0_j, _ = bonds_loc[0]
    prop_exact.apply_bond_kick(b0_i, b0_j, 0.35 * u_loc[0])
    prop_trot.apply_bond_kick(b0_i, b0_j, 0.35 * u_loc[0])

    max_diff = 0.0
    for _ in range(steps):
        prop_exact.step_trotter()
        # Symmetric Strang / Trotter bond-by-bond Givens sweep
        for b_idx, (qa, qb, _) in enumerate(bonds_loc):
            prop_trot.apply_bond_kick(qa, qb, 0.5 * dt * w_loc[b_idx] * u_loc[b_idx])
        for b_idx in reversed(range(len(bonds_loc))):
            qa, qb, _ = bonds_loc[b_idx]
            prop_trot.apply_bond_kick(qa, qb, 0.5 * dt * w_loc[b_idx] * u_loc[b_idx])
        diff = float(np.max(np.abs(prop_exact.bond_expectations() -
                                   prop_trot.bond_expectations())))
        max_diff = max(max_diff, diff)

    return {
        "rung": "F1",
        "system": "27-qubit manifold (Type 0)",
        "steps": steps,
        "dt": dt,
        "max_bond_err": max_diff,
        "pass": max_diff < 0.02,
    }


def step_decomposed_strang(prop, bonds_list, u_arr, w_arr, is_seam_arr, dt):
    """Execute one block-decomposed step using the 3-colour matching structure
    of the Kitaev lattice: for each colour layer c in (0, 1, 2, 2, 1, 0) with
    half-step 0.5*dt, apply internal manifold bonds first, then seam kicks."""
    half = 0.5 * float(dt)
    for col in (0, 1, 2, 2, 1, 0):
        # 1. Internal manifold bonds of colour `col`
        for b_idx, (qa, qb, c) in enumerate(bonds_list):
            if c == col and not is_seam_arr[b_idx]:
                prop.apply_bond_kick(qa, qb, half * w_arr[b_idx] * u_arr[b_idx])
        # 2. Inter-manifold seam kicks of colour `col`
        for b_idx, (qa, qb, c) in enumerate(bonds_list):
            if c == col and is_seam_arr[b_idx]:
                prop.apply_bond_kick(qa, qb, half * w_arr[b_idx] * u_arr[b_idx])


def rung_f2_single_module(manifest, steps=8, dt=0.06):
    """Rung F2: One half-period module (108 qubits = 4 x 27 K4 cluster).
    Compare decoupled K4-kick evolution against exact 108-site module G(t)."""
    mod_sites = []
    for bid in range(4):
        mod_sites.extend(manifest["blocks"][bid]["sites"])
    g_to_m = {g: loc for loc, g in enumerate(mod_sites)}

    bonds_mod = []
    u_mod = []
    w_mod = []
    is_k4_seam = []
    block_of_m = {loc: loc // 27 for loc in range(108)}

    for b_idx, (i, j, c) in enumerate(manifest["bonds"]):
        if i in g_to_m and j in g_to_m:
            mi, mj = g_to_m[i], g_to_m[j]
            bonds_mod.append((mi, mj, int(c)))
            u_mod.append(float(manifest["gauge"][b_idx]))
            w_mod.append(float(manifest["weights"][b_idx]))
            is_k4_seam.append(block_of_m[mi] != block_of_m[mj])

    prop_exact = MajoranaFlowPropagator(108, bonds_mod, u_mod, w_mod, dt=dt)
    prop_k4 = MajoranaFlowPropagator(108, bonds_mod, u_mod, w_mod, dt=dt)

    b0_i, b0_j, _ = bonds_mod[0]
    prop_exact.apply_bond_kick(b0_i, b0_j, 0.35 * u_mod[0])
    prop_k4.apply_bond_kick(b0_i, b0_j, 0.35 * u_mod[0])

    rms_diffs = []
    for _ in range(steps):
        prop_exact.step_trotter()
        step_decomposed_strang(prop_k4, bonds_mod, u_mod, w_mod, is_k4_seam, dt)
        rms = float(np.sqrt(np.mean(np.square(
            prop_exact.bond_expectations() - prop_k4.bond_expectations()))))
        rms_diffs.append(rms)

    max_rms = max(rms_diffs)
    return {
        "rung": "F2",
        "system": "108-qubit half-period module (K4 cluster)",
        "steps": steps,
        "dt": dt,
        "rms_bond_err": max_rms,
        "pass": max_rms < 0.02,
    }


def rung_f3_toroidal_ring(manifest, steps=16, dt=0.10):
    """Rung F3: Ring of 10 modules on a single shell (200 sites).
    Measure toroidal pulse circulation and transit time agreement between
    exact ring propagation and module-to-module seam kicks."""
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

    for b_idx, (si, sj, _) in enumerate(bonds_s0):
        if mod_of_s0[si] == 0 and mod_of_s0[sj] == 0:
            prop_exact.apply_bond_kick(si, sj, 0.30 * u_s0[b_idx])
            prop_kick.apply_bond_kick(si, sj, 0.30 * u_s0[b_idx])

    opp_energy_exact = []
    opp_energy_kick = []
    for _ in range(steps):
        prop_exact.step_trotter()
        step_decomposed_strang(prop_kick, bonds_s0, u_s0, w_s0, is_tor_seam, dt)
        step_decomposed_strang(prop_vac, bonds_s0, u_s0, w_s0, is_tor_seam, dt)

        de_ex = prop_exact.bond_energies_above_ground()
        de_ki = prop_kick.bond_energies_above_ground(g_ref=prop_vac.G)
        e5_ex = float(sum(de_ex[b] for b, (si, _, _) in enumerate(bonds_s0) if mod_of_s0[si] == 5))
        e5_ki = float(sum(de_ki[b] for b, (si, _, _) in enumerate(bonds_s0) if mod_of_s0[si] == 5))
        opp_energy_exact.append(e5_ex)
        opp_energy_kick.append(e5_ki)

    t_peak_exact = int(np.argmax(opp_energy_exact)) + 1
    t_peak_kick = int(np.argmax(opp_energy_kick)) + 1
    transit_diff = abs(t_peak_exact - t_peak_kick)

    return {
        "rung": "F3",
        "system": "200-qubit 10-module toroidal shell ring",
        "steps": steps,
        "t_peak_exact": t_peak_exact,
        "t_peak_kick": t_peak_kick,
        "transit_step_diff": transit_diff,
        "pass": transit_diff <= 1,
    }


def rung_f4_full_machine(manifest, steps=10, dt=0.06):
    """Rung F4: Full 1080-qubit stellarator (1620 bonds).
    Inject a core excitation on shell 0 / magnetic axis and compare the radial
    energy profile across shells s = 0..4 between exact 1080-site propagation
    and block-decomposed (40 x 27 + K4 + toroidal seam kick) propagation."""
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

    for b_idx, (i, j, _) in enumerate(bonds):
        if site_shell[i] == 0 and site_shell[j] == 0 and site_block[i] < 4:
            prop_exact.apply_bond_kick(i, j, 0.25 * u_g[b_idx])
            prop_decomp.apply_bond_kick(i, j, 0.25 * u_g[b_idx])

    for _ in range(steps):
        prop_exact.step_trotter()
        step_decomposed_strang(prop_decomp, bonds, u_g, w_g, is_seam, dt)
        step_decomposed_strang(prop_vac, bonds, u_g, w_g, is_seam, dt)

    de_ex = prop_exact.bond_energies_above_ground()
    de_dc = prop_decomp.bond_energies_above_ground(g_ref=prop_vac.G)

    shell_ex = np.zeros(5, dtype=float)
    shell_dc = np.zeros(5, dtype=float)
    for b_idx, (i, j, _) in enumerate(bonds):
        s = max(site_shell[i], site_shell[j])
        shell_ex[s] += de_ex[b_idx]
        shell_dc[s] += de_dc[b_idx]

    tot_ex = max(float(np.sum(shell_ex)), 1e-12)
    tot_dc = max(float(np.sum(shell_dc)), 1e-12)
    prof_ex = shell_ex / tot_ex
    prof_dc = shell_dc / tot_dc
    max_shell_dev = float(np.max(np.abs(prof_ex - prof_dc)))

    return {
        "rung": "F4",
        "system": "1080-qubit full stellarator (40 x 27 manifolds)",
        "steps": steps,
        "shell_profile_exact": [float(x) for x in prof_ex],
        "shell_profile_decomp": [float(x) for x in prof_dc],
        "max_shell_deviation": max_shell_dev,
        "outer_leakage_fraction": float(prof_ex[3] + prof_ex[4]),
        "pass": max_shell_dev < 0.05,
    }


def run_reference_ladder(manifest):
    print("=" * 78)
    print("PHASE 5 GATE (REFERENCE FLOW LADDER) -- EXACT FREE-FERMION RUNGS F1 TO F4")
    print("=" * 78)
    r1 = rung_f1_single_manifold(manifest)
    print("  [F1] %-42s max_bond_err = %.4e   %s"
          % (r1["system"], r1["max_bond_err"], "PASS" if r1["pass"] else "FAIL"))

    r2 = rung_f2_single_module(manifest)
    print("  [F2] %-42s rms_bond_err = %.4e   %s"
          % (r2["system"], r2["rms_bond_err"], "PASS" if r2["pass"] else "FAIL"))

    r3 = rung_f3_toroidal_ring(manifest)
    print("  [F3] %-42s transit peak = %d vs %d  %s"
          % (r3["system"], r3["t_peak_exact"], r3["t_peak_kick"],
             "PASS" if r3["pass"] else "FAIL"))

    r4 = rung_f4_full_machine(manifest)
    print("  [F4] %-42s max_shell_dev = %.4f  %s"
          % (r4["system"], r4["max_shell_deviation"], "PASS" if r4["pass"] else "FAIL"))
    print("       shell profile (exact) : %s"
          % " ".join("s%d=%.3f" % (s, v) for s, v in enumerate(r4["shell_profile_exact"])))
    print("       shell profile (decomp): %s"
          % " ".join("s%d=%.3f" % (s, v) for s, v in enumerate(r4["shell_profile_decomp"])))
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
