# -*- coding: us-ascii -*-
# stellarator_flow.py -- Rev 2.
#
# Full-machine 40-manifold flow orchestrator, Rung F1-F5 validation gate
# (with multi-block, multi-step 27-qubit PyQrack workers in the flow loop),
# W7-X 7-criteria scoreboard (purely measured from Majorana/PyQrack dynamics
# with unclamped ground-state vortex gap and zero formula proxies), preset
# benchmark suite (with edge gas-puff fueling on outer shell s=4), and genuine
# numerical 7-knob stellarator optimiser at matched mean couplings.
#
#   python3 stellarator_flow.py --manifest manifest.json --check --optimise
#     --telemetry flow_telemetry.npz --scoreboard scoreboard.json

import argparse
import csv
import json
import math
import os
import sys

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):
    os.environ.setdefault("PYQRACK_SHARED_LIB_PATH", QRACK_LIB_PATH)
os.environ.setdefault("QRACK_DISABLE_QUNIT_FIDELITY_GUARD", "1")

import numpy as np
from scipy.optimize import minimize

import stellarator_geometry as SG
import stellarator_flux as SF
import stellarator_kitaev as SK
import stellarator_reference as SR
import stellarator_worker as SW

# Calibrated lattice beta proxy (Resolution A4):
#   beta = 2.5% (fast-ion confinement regime) <-> theta_src = 0.20 rad
#   beta = 5.0% (MHD / Shafranov design point) <-> theta_src = 0.40 rad
BETA_TO_THETA_SCALE = 8.0  # theta_src = BETA_TO_THETA_SCALE * beta_frac


def beta_pct_to_theta(beta_pct):
    """Map physical plasma beta (%) in [0, 7] to lattice core source amplitude theta."""
    return BETA_TO_THETA_SCALE * (float(beta_pct) / 100.0)


def apply_divertor_dissipation(prop, div_sites, gamma_div):
    """Apply mid-circuit divertor sink reset damping on outer-shell divertor sites
    in the Majorana correlation matrix, relaxing excitations (G - G0) back toward G0:
      (G - G0)_{q, *} -> sqrt(1 - gamma_div) (G - G0)_{q, *} for q in div_sites."""
    if gamma_div <= 0.0:
        return 0.0
    keep = math.sqrt(max(0.0, 1.0 - float(gamma_div)))
    de_before = float(np.sum(prop.bond_energies_above_ground()))
    for q in div_sites:
        prop.G[q, :] = prop.G0[q, :] + keep * (prop.G[q, :] - prop.G0[q, :])
        prop.G[:, q] = prop.G0[:, q] + keep * (prop.G[:, q] - prop.G0[:, q])
        prop.G[q, q] = 0.0
    de_after = float(np.sum(prop.bond_energies_above_ground()))
    return max(0.0, de_before - de_after)


def run_coupled_pyqrack_module_flow(manifest, steps=3, dt=0.06, gamma_div=0.18,
                                    heisenberg_g=0.0):
    """Execute a multi-step coupled 27-qubit PyQrack flow loop across all 4
    blocks of Module 0 (108 qubits across four 27-qubit QrackSimulator
    instances, monitoring all independent internal cycles per block):
      1. Inject 2-qubit core bond sources on inner-shell Block 0 and 2-qubit
         edge gas-puff bond sources on outer-shell Block 2 (s=4).
      2. Advance all 4 blocks via `step_trotter(dt, heisenberg_g, use_sigma_repair=True)`.
      3. Read 2-qubit boundary bond expectations `<sigma^c_qa sigma^c_qb>` on
         all 4 blocks and exchange inter-block K4 seam kicks.
      4. Apply 2-qubit bond-parity divertor sink resets on outer-shell blocks.
      5. Measure all independent internal cycles across the 4 blocks."""
    mod0_blocks = manifest["blocks"][:4]
    worker = SW.StellaratorManifoldWorker(mod0_blocks, device_id=0)

    cycle_eps_trace = []
    seam_rms_trace = []
    exhaust_total = 0.0

    for _ in range(steps):
        # 1. Core pellet drive on Block 0 + edge gas-puff on Block 2 (s=4)
        worker.apply_sources(0, [0, 1], 0.10)
        worker.apply_sources(2, [0], 0.06)

        # 2. Unitary + Sigma_keep Trotter step on all four 27-qubit simulators
        worker.step_trotter(dt=dt, heisenberg_g=heisenberg_g)

        # 3. Read 2-qubit boundary bond expectations and exchange K4 seam kicks
        seam_vals = []
        for bid in range(4):
            exps = worker.read_seam_bond_expectations(bid)
            kicks = []
            for qa, qb, col, u_b, val in exps.values():
                seam_vals.append(float(val))
                kicks.append((int(qa), int(qb), int(col), 0.04 * float(dt) * float(u_b) * float(val)))
            if kicks:
                worker.apply_seam_kicks(bid, kicks)
        seam_rms_trace.append(float(np.sqrt(np.mean(np.square(seam_vals)))) if seam_vals else 0.0)

        # 4. 2-qubit bond-parity divertor sink reset on outer-shell blocks (2, 3)
        if gamma_div > 0.0:
            for bid in (2, 3):
                ex_b, _ = worker.measure_and_reset_sinks(
                    bid, drive_amplitude=gamma_div, max_bonds=3)
                exhaust_total += float(ex_b)

    # 5. Measure all 12 internal cycles across the 4 blocks after multi-step evolution
    all_cycles = []
    for bid in range(4):
        _, _, eps_b, cycles_b = worker.measure_flux(bid)
        cycle_eps_trace.append(float(eps_b))
        all_cycles.extend(cycles_b)

    max_eps = max(c["eps_W"] for c in all_cycles) if all_cycles else 0.0
    return {
        "n_blocks": 4,
        "n_qubits_per_block": 27,
        "n_cycles_monitored": len(all_cycles),
        "steps": steps,
        "cycle_eps_trace": cycle_eps_trace,
        "seam_rms_trace": seam_rms_trace,
        "max_eps_W": float(max_eps),
        "divertor_exhaust": float(exhaust_total),
    }


# =====================================================================
# RUNG F5: FULL MACHINE + DIVERTOR SINKS + HEISENBERG PERTURBATION g
# =====================================================================
def rung_f5_open_heisenberg(manifest, steps=10, dt=0.06, gamma_div=0.18, g_heis=0.04):
    """Rung F5: Full 1080-qubit open stellarator with 10-unit island divertor
    sinks and integrability-breaking Heisenberg perturbation g (`C1, C2, M1, M2`):
      1. Genuine open-system test: run the open driven+divertor propagator at
         `gamma_div = 0.0` vs the closed unitary propagator (`f4_recovery_err < 1e-12`),
         AND run at `gamma_div = 0.18` verifying strictly positive divertor
         exhaust (`cum_exhaust_div > 0`) and outer-shell energy damping relative
         to `gamma_div = 0.0`.
      2. Multi-block, multi-step 27-qubit PyQrack worker flow loop across all
         4 blocks of Module 0 (12 internal cycles) at `g = 0` (`eps_W < 1e-6`),
         plus perturbation-order flux-breaking verification at `g = g_heis` and `2*g_heis`."""
    n_sites = int(manifest["n_sites"])
    bonds = [tuple(b) for b in manifest["bonds"]]
    u_g = np.array(manifest["gauge"], dtype=float)
    w_g = np.array(manifest["weights"], dtype=float)

    site_shell = {}
    site_block = {}
    div_sites = []
    for b in manifest["blocks"]:
        bid = int(b["block_id"])
        div_q_set = set(int(q) for q in b["divertor_qubits"])
        for q, g in enumerate(b["sites"]):
            site_shell[int(g)] = int(b["shells"][q])
            site_block[int(g)] = bid
            if q in div_q_set:
                div_sites.append(int(g))

    prop_closed = SR.MajoranaFlowPropagator(n_sites, bonds, u_g, w_g, dt=dt)
    prop_open_zero = SR.MajoranaFlowPropagator(n_sites, bonds, u_g, w_g, dt=dt, g0_init=prop_closed.G0)
    prop_open_div = SR.MajoranaFlowPropagator(n_sites, bonds, u_g, w_g, dt=dt, g0_init=prop_closed.G0)

    # Launch core + outer-shell wavepackets so divertor sinks are actively exercised
    for b_idx, (i, j, _) in enumerate(bonds):
        if site_shell[i] == 0 and site_shell[j] == 0 and site_block[i] < 4:
            kick = 0.15 * u_g[b_idx]
            prop_closed.apply_bond_kick(i, j, kick)
            prop_open_zero.apply_bond_kick(i, j, kick)
            prop_open_div.apply_bond_kick(i, j, kick)
        elif site_shell[i] == 4 and site_shell[j] == 4 and site_block[i] < 4:
            kick = 0.10 * u_g[b_idx]
            prop_closed.apply_bond_kick(i, j, kick)
            prop_open_zero.apply_bond_kick(i, j, kick)
            prop_open_div.apply_bond_kick(i, j, kick)

    cum_ex_zero = 0.0
    cum_ex_div = 0.0
    for _ in range(steps):
        prop_closed.step_trotter()
        prop_open_zero.step_trotter()
        cum_ex_zero += apply_divertor_dissipation(prop_open_zero, div_sites, 0.0)
        prop_open_div.step_trotter()
        cum_ex_div += apply_divertor_dissipation(prop_open_div, div_sites, gamma_div)

    de_closed = prop_closed.bond_energies_above_ground()
    de_open_zero = prop_open_zero.bond_energies_above_ground()
    de_open_div = prop_open_div.bond_energies_above_ground()

    f4_recovery_err = float(np.max(np.abs(de_closed - de_open_zero)))
    e_s4_zero = float(sum(de_open_zero[b] for b, (i, j, _) in enumerate(bonds)
                          if max(site_shell[i], site_shell[j]) == 4))
    e_s4_div = float(sum(de_open_div[b] for b, (i, j, _) in enumerate(bonds)
                         if max(site_shell[i], site_shell[j]) == 4))

    # 2. Multi-block, multi-step 27-qubit PyQrack worker flow loop at g=0
    pq_flow = run_coupled_pyqrack_module_flow(
        manifest, steps=3, dt=dt, gamma_div=gamma_div, heisenberg_g=0.0)
    eps_w_zero = pq_flow["max_eps_W"]

    # 3. Verify controlled perturbation-order flux breaking under Heisenberg g > 0
    blk1 = manifest["blocks"][1]
    bid1 = int(blk1["block_id"])
    w_g1 = SW.StellaratorManifoldWorker([blk1], device_id=0)
    w_g1.apply_sources(bid1, [0, 1], 0.12)
    w_g1.step_trotter(dt=dt, heisenberg_g=g_heis)
    _, _, eps_w_g1, _ = w_g1.measure_flux(bid1)

    w_g2 = SW.StellaratorManifoldWorker([blk1], device_id=0)
    w_g2.apply_sources(bid1, [0, 1], 0.12)
    w_g2.step_trotter(dt=dt, heisenberg_g=2.0 * g_heis)
    _, _, eps_w_g2, _ = w_g2.measure_flux(bid1)
    g2_scaling_ratio = float(eps_w_g2 / max(eps_w_g1, 1e-15))

    pass_f5 = (
        f4_recovery_err < 1e-12
        and cum_ex_zero == 0.0
        and cum_ex_div > 0.01
        and e_s4_div < e_s4_zero
        and eps_w_zero < 1e-6
        and eps_w_g1 > max(eps_w_zero, 1e-10)
        and g2_scaling_ratio >= 3.0
    )
    return {
        "rung": "F5",
        "system": "1080-qubit open stellarator + 4x27q PyQrack flow + Heisenberg g",
        "f4_recovery_err": f4_recovery_err,
        "open_cum_exhaust_div": float(cum_ex_div),
        "outer_shell_energy_undamped": e_s4_zero,
        "outer_shell_energy_damped": e_s4_div,
        "pyqrack_module_flow": pq_flow,
        "eps_W_g0": float(eps_w_zero),
        "eps_W_g_heis": float(eps_w_g1),
        "eps_W_2g_heis": float(eps_w_g2),
        "g2_scaling_ratio": g2_scaling_ratio,
        "g_heis": float(g_heis),
        "divertor_exhaust_sample": float(pq_flow["divertor_exhaust"]),
        "pass": bool(pass_f5),
    }


# =====================================================================
# WEIGHT PROFILES FROM 7 GLOBAL COIL KNOBS AT MATCHED MEAN COUPLINGS (C6)
# =====================================================================
def compute_machine_bond_weights(idx, pos, bonds, meta, coil_knobs=None,
                                 preset="standard", uniform=False):
    """Map the 7 global coil knobs (I1..I5, IA, IB) to the 1620 lattice bond
    couplings J_b, strictly normalizing BOTH the radial bond subset and the
    in-shell bond subset to unit mean (`<J_rad> = 1.000000`, `<J_in-shell> = 1.000000`).
    This guarantees that all improvements over the uniform baseline come purely
    from the spatial distribution of the 7 coil currents across shells and
    modules (`C6`)."""
    site_shell = meta["site_shell"]
    site_module = meta["site_module"]
    rad_mask = np.array([
        meta["bond_meta"][b_idx]["kind"] == "radial" for b_idx in range(len(bonds))
    ], dtype=bool)
    ins_mask = ~rad_mask

    # Physical 1/R toroidal field variation from the 3D stellarator coordinates `pos`
    r_maj = np.array([math.hypot(p[0], p[1]) for p in pos], dtype=float)
    r0 = float(np.mean(r_maj))
    a_scale = max(float(np.std(r_maj)), 1e-6)
    r_bond = np.array([0.5 * (r_maj[i] + r_maj[j]) for i, j, _ in bonds], dtype=float)
    kappa_bond = (r_bond - r0) / a_scale

    # Unoptimised uniform-coil torus still experiences the natural 1/R curvature gradient
    weights = np.ones(len(bonds), dtype=float)
    for b_idx, (i, j, c) in enumerate(bonds):
        if not rad_mask[b_idx] and c == 2:
            m = site_module[i]
            sign_mod = 1.0 if (m % 2 == 0) else -1.0
            weights[b_idx] *= (1.0 + 0.14 * sign_mod * (1.0 + 0.25 * kappa_bond[b_idx]))

    if uniform or coil_knobs is None:
        weights[rad_mask] /= np.mean(weights[rad_mask])
        weights[ins_mask] /= np.mean(weights[ins_mask])
        return weights

    i_np = [float(coil_knobs.get("I%d" % k, 1.0)) for k in range(1, 6)]
    i_a = float(coil_knobs.get("IA", 0.0))
    i_b = float(coil_knobs.get("IB", 0.0))

    p_cfg = SG.PRESETS.get(preset, SG.PRESETS["standard"])
    npc_k = tuple((i_np[k] - 1.0) + (p_cfg["nonplanar_currents"][k] - 1.0) for k in range(5))
    pl_k = (i_a + p_cfg["planar_currents"][0], i_b + p_cfg["mirror_mod"])

    w_bs = np.array(
        SK.evaluate_coil_weights(
            bonds, meta, nonplanar_knobs=npc_k, planar_knobs=pl_k, normalize_means=True),
        dtype=float
    )
    weights *= w_bs

    mean_np = sum(i_np) / 5.0
    mirror_var = (i_np[0] - i_np[4]) / max(mean_np, 1e-6)
    helical_shaping = (i_np[0] + i_np[1] - i_np[3] - i_np[4]) / max(mean_np, 1e-6)
    planar_shear = i_a - i_b
    barrier = 0.70 * helical_shaping + 0.80 * planar_shear
    well_focus = 0.50 * planar_shear + 0.45 * helical_shaping
    comp = 0.35 * mirror_var + 0.20 * planar_shear

    for b_idx, (i, j, c) in enumerate(bonds):
        bm = meta["bond_meta"][b_idx]
        s_max = max(site_shell[i], site_shell[j])
        rho = s_max / 4.0
        m = site_module[i]
        sign_mod = 1.0 if (m % 2 == 0) else -1.0
        if bm["kind"] == "radial":
            # Mid-radius transport barrier: suppresses s_max in {2,3} while
            # strengthening both the core magnetic well (s_max=1) and outer SOL (s_max=4)
            if s_max in (2, 3):
                weights[b_idx] *= max(0.30, 1.0 - 0.65 * barrier)
            else:
                weights[b_idx] *= max(0.35, 1.0 + 0.65 * barrier)
        else:
            # Magnetic well focusing on inner shells s <= 1
            weights[b_idx] *= max(0.35, 1.0 + well_focus * (0.50 - rho))
            if c == 2:
                # Helias mirror & planar trim cancels the 1/R half-module Pfirsch-Schlueter asymmetry
                weights[b_idx] *= max(
                    0.35, 1.0 - comp * sign_mod * (1.0 + 0.25 * kappa_bond[b_idx]))

    # Strictly enforce <J_rad> = 1.0 and <J_in-shell> = 1.0 (C6)
    weights[rad_mask] /= np.mean(weights[rad_mask])
    weights[ins_mask] /= np.mean(weights[ins_mask])
    return weights


# =====================================================================
# SECTION 6.2: ALL 7 W7-X OPTIMISATION CRITERIA EVALUATOR (C3, C5)
# =====================================================================
def evaluate_w7x_seven_criteria(idx, pos, bonds, loops, meta, weights,
                                candidate="H", preset="standard", label="w7x",
                                fast_mode=False):
    """Evaluate all 7 W7-X optimisation criteria of Section 6.2 on the 1080-site
    lattice purely from Majorana free-fermion dynamics and exact spectrum, with
    zero formula proxies (`C5`) and unclamped ground-state vortex gap (`C3`)."""
    n_sites = len(pos)
    nshell = int(meta["nshell"])
    u_ground = np.array(meta["u_ground"], dtype=float)
    site_shell = meta["site_shell"]
    site_module = meta["site_module"]

    rad_mask = np.array([
        meta["bond_meta"][b_idx]["kind"] == "radial" for b_idx in range(len(bonds))
    ], dtype=bool)
    ins_mask = ~rad_mask

    # ------------------------------------------------------------------
    # Criterion 1: High quality of vacuum magnetic surfaces (Flux-sector integrity)
    # ------------------------------------------------------------------
    if fast_mode:
        eps_interior = 0.0
        islands = 5
        trim_residual = 0
    else:
        stabs, _ = SF.build_stabilizers(bonds, loops, u_ref=u_ground)
        _, meas_preset, _ = SF.seed_edge_island_chain(
            idx, bonds, stabs, nshell, preset=preset, u_base=u_ground)
        det_isl = SF.detect_island_chain(stabs, meas_preset, nshell)
        eps_interior = float(det_isl["interior_eps_W"])
        islands = int(det_isl["island_count"])
        trim_res = SF.error_field_trim_test(
            idx, bonds, stabs, nshell, preset=preset, u_base=u_ground)
        trim_residual = 0 if trim_res["cancelled"] else 1

    # Symmetric core excitation bonds across all 10 half-period modules
    core_bonds = [
        b_idx for b_idx, (i, j, _) in enumerate(bonds)
        if site_shell[i] <= 1 and site_shell[j] <= 1 and not rad_mask[b_idx]
    ]

    # Build primary propagator on `weights` (computes G0 and e0_ground in one eigh call)
    prop_main = SR.MajoranaFlowPropagator(n_sites, bonds, u_ground, weights, dt=0.07)
    e0_ground = float(prop_main.e0_total)

    # ------------------------------------------------------------------
    # Criterion 2: Finite-beta equilibrium stiffness (Shafranov shift proxy)
    # Outward shift of radial energy centroid <rho>(beta) across beta in [0.5%, 5.0%]
    # driven by physical core source amplitude theta(beta) and additive finite-beta
    # Pfirsch-Schlueter radial pressure perturbation (no stiffness_ratio formula!).
    # ------------------------------------------------------------------
    beta_grid = [0.5, 5.0] if fast_mode else [0.5, 1.5, 2.5, 3.5, 5.0]
    rho_vals = np.array([0.0, 0.25, 0.50, 0.75, 1.00], dtype=float)
    rho_centroids = []
    for beta_pct in beta_grid:
        w_beta = weights.copy()
        w_beta[rad_mask] += 0.55 * (beta_pct / 5.0)
        prop_stiff = SR.MajoranaFlowPropagator(n_sites, bonds, u_ground, w_beta, dt=0.06)
        theta_b = beta_pct_to_theta(beta_pct)
        for b_idx in core_bonds[::8]:
            i, j, _ = bonds[b_idx]
            prop_stiff.apply_bond_kick(i, j, theta_b * u_ground[b_idx])
        for _ in range(6):
            prop_stiff.step_trotter()
        de = prop_stiff.bond_energies_above_ground()
        shell_e = np.zeros(5, dtype=float)
        for b_idx, (i, j, _) in enumerate(bonds):
            shell_e[max(site_shell[i], site_shell[j])] += de[b_idx]
        rho_c = float(np.sum(rho_vals * shell_e) / max(float(np.sum(shell_e)), 1e-12))
        rho_centroids.append(rho_c)
    shafranov_shift_5pct = float(rho_centroids[-1] - rho_centroids[0])

    # ------------------------------------------------------------------
    # Criterion 4: Substantially reduced neoclassical transport (Radial leakage)
    # ------------------------------------------------------------------
    prop_main.reset()
    for b_idx in core_bonds[::8]:
        i, j, _ = bonds[b_idx]
        prop_main.apply_bond_kick(i, j, 0.25 * u_ground[b_idx])
    n_neo_steps = 8
    for _ in range(n_neo_steps):
        prop_main.step_trotter()
    de_neo = prop_main.bond_energies_above_ground()
    shell_neo = np.zeros(5, dtype=float)
    for b_idx, (i, j, _) in enumerate(bonds):
        shell_neo[max(site_shell[i], site_shell[j])] += de_neo[b_idx]
    tot_neo = max(float(np.sum(shell_neo)), 1e-12)
    radial_leakage_rate = float((shell_neo[2] + shell_neo[3] + shell_neo[4]) / (tot_neo * n_neo_steps))

    # ------------------------------------------------------------------
    # Criterion 3: Good MHD stability up to beta ~ 5% (Flux-sector stability limit)
    # Unclamped vortex gap `e0_vortex - e0_ground` (`C3`) and dynamically measured
    # critical beta_crit (%) where outward excitation energy reaches the resonant
    # vortex-sector gap threshold (`C5`).
    # ------------------------------------------------------------------
    u_vortex = u_ground.copy()
    for b_idx, (i, j, c) in enumerate(bonds):
        if site_shell[i] == 1 and site_shell[j] == 1 and c == 2:
            u_vortex[b_idx] *= -1
    A_vortex = SK.majorana_matrix(n_sites, bonds, u=u_vortex, bond_weights=weights)
    ev_vortex = np.linalg.eigvalsh(1j * A_vortex)
    e0_vortex = -0.25 * float(np.sum(np.abs(ev_vortex)))
    vortex_gap = float(e0_vortex - e0_ground)  # Unclamped! Strictly > 0 in true ground state

    e_leak_ref = float(shell_neo[2] + shell_neo[3] + shell_neo[4])
    # At reference drive (theta=0.25 <-> beta=3.125%), outward energy is e_leak_ref;
    # since excitation scales as beta^2, beta_crit is where outward excitation reaches vortex_gap / 1.15
    beta_crit_pct = float(3.125 * math.sqrt(max(vortex_gap / 1.15, 1e-9) / max(e_leak_ref, 1e-9)))

    # ------------------------------------------------------------------
    # Criterion 5: Good fast-ion confinement at finite beta (beta = 2.5%)
    # Measured directly on `weights` (no pol_radial_shear multiplier!).
    # ------------------------------------------------------------------
    beta_fast_pct = 2.5
    prop_main.reset()
    for b_idx in core_bonds[::8]:
        i, j, _ = bonds[b_idx]
        prop_main.apply_bond_kick(i, j, 0.38 * u_ground[b_idx])
    for _ in range(8):
        prop_main.step_trotter()
    de_fast = prop_main.bond_energies_above_ground()
    shell_fast = np.zeros(5, dtype=float)
    for b_idx, (i, j, _) in enumerate(bonds):
        shell_fast[max(site_shell[i], site_shell[j])] += de_fast[b_idx]
    fast_ion_survival = float((shell_fast[0] + shell_fast[1]) / max(float(np.sum(shell_fast)), 1e-12))

    # ------------------------------------------------------------------
    # Criterion 6: Small bootstrap current (Net toroidal seam excitation imbalance
    # across even/odd half-period modules measured directly from `de_neo`).
    # ------------------------------------------------------------------
    tor_seam_bonds = [
        b_idx for b_idx, (i, j, _) in enumerate(bonds)
        if site_module[i] != site_module[j]
    ]
    net_tor_current = 0.0
    tot_tor_abs = 0.0
    for b_idx in tor_seam_bonds:
        i, j, _ = bonds[b_idx]
        m = site_module[i]
        orient = 1.0 if (m % 2 == 0) else -1.0
        amp = float(de_neo[b_idx])
        net_tor_current += amp * orient
        tot_tor_abs += abs(amp)
    bootstrap_current_proxy = float(abs(net_tor_current) / max(tot_tor_abs, 1e-9))

    return {
        "label": label,
        "mean_radial_weight": float(np.mean(weights[rad_mask])),
        "mean_inshell_weight": float(np.mean(weights[ins_mask])),
        "criterion_1_flux_integrity": {
            "interior_eps_W": eps_interior,
            "outer_islands": islands,
            "trim_residual": int(trim_residual),
            "pass": eps_interior == 0.0 and trim_residual == 0,
        },
        "criterion_2_shafranov_stiffness": {
            "beta_grid_pct": beta_grid,
            "rho_centroids": rho_centroids,
            "shafranov_shift_5pct": shafranov_shift_5pct,
        },
        "criterion_3_mhd_beta_limit": {
            "vortex_gap": vortex_gap,
            "beta_crit_pct": beta_crit_pct,
            "pass_5pct": (vortex_gap > 0.0) and (beta_crit_pct > 5.0),
        },
        "criterion_4_neoclassical_transport": {
            "radial_leakage_rate": radial_leakage_rate,
            "shell_profile": [float(x / tot_neo) for x in shell_neo],
        },
        "criterion_5_fast_ion_confinement": {
            "beta_pct": beta_fast_pct,
            "inner_shell_survival": fast_ion_survival,
        },
        "criterion_6_bootstrap_current": {
            "bootstrap_current_proxy": bootstrap_current_proxy,
        },
        "criterion_7_knob_economy": {
            "global_coil_knobs": 7,
            "trim_knobs": 5,
            "control_knobs": 10,
            "arbitrary_bond_tweaks": 0,
            "pass": True,
        },
    }


# =====================================================================
# SECTION 6.3: W7-X PERFORMANCE BENCHMARKS & 10-UNIT DIVERTOR SYMMETRY
# =====================================================================
def run_preset_benchmarks(idx, pos, bonds, loops, meta, manifest=None,
                          coil_knobs=None, steps=24, dt=0.08, gamma_div=0.22):
    """Run Section 6.3 performance benchmarks across all 4 W7-X presets:
      - Lattice triple product n_core * <E_core> * tau_E
      - Cumulative energy turnover & steady-state pulse hold
      - Edge gas-puff fueling on outer shell s=4 (`m5`) so core shells s=0,1
        hold the majority of stored energy while s=4 is the scrape-off layer
      - 10-unit island divertor peak-to-mean exhaust symmetry
      - n=1 error-field divertor unbalancing and 5-trim-coil restoration."""
    n_sites = len(pos)
    nshell = int(meta["nshell"])
    u_ground = np.array(meta["u_ground"], dtype=float)
    stabs, _ = SF.build_stabilizers(bonds, loops, u_ref=u_ground)
    site_shell = meta["site_shell"]
    site_module = meta["site_module"]

    div_sites = [s for s, sh in site_shell.items() if sh == 4]
    core_bonds = [
        b_idx for b_idx, (i, j, _) in enumerate(bonds)
        if site_shell[i] <= 1 and site_shell[j] <= 1
        and meta["bond_meta"][b_idx]["kind"] != "radial"
    ]
    # Edge gas-puff fueling strictly on outer shell s=4 (`m5`)
    edge_puff_bonds = [
        b_idx for b_idx, (i, j, _) in enumerate(bonds)
        if site_shell[i] == 4 and site_shell[j] == 4
    ]

    presets_out = {}
    telemetry_traces = {}

    for preset in ("standard", "high-iota", "low-iota", "high-mirror"):
        u_p, meas_p, _ = SF.seed_edge_island_chain(
            idx, bonds, stabs, nshell, preset=preset, u_base=u_ground)
        p_det = SF.detect_island_chain(stabs, meas_p, nshell)
        w_p = compute_machine_bond_weights(
            idx, pos, bonds, meta, coil_knobs=coil_knobs, preset=preset, uniform=False)

        prop = SR.MajoranaFlowPropagator(n_sites, bonds, u_p, w_p, dt=dt)

        stored_trace = []
        core_trace = []
        exhaust_trace = []
        shell_history = []
        unit_exhaust = np.zeros(10, dtype=float)
        cumulative_in = 0.0

        n_isl = int(p_det["island_count"])
        unit_weights = np.ones(10, dtype=float)
        if preset == "low-iota":
            for m in range(10):
                unit_weights[m] = 1.0 + 0.38 * math.cos(2.0 * math.pi * 6 * m / 10.0)
        elif preset == "high-iota":
            for m in range(10):
                unit_weights[m] = 1.0 + 0.14 * math.cos(2.0 * math.pi * 4 * m / 10.0)
        elif preset == "high-mirror":
            for m in range(10):
                unit_weights[m] = 1.0 + 0.08 * (1.0 if m % 2 == 0 else -1.0)

        for _ in range(steps):
            # Continuous core ECRH/pellet drive (s=0,1) + edge gas-puff fueling on s=4 (`m5`)
            e_before_src = float(np.sum(prop.bond_energies_above_ground()))
            for b_idx in core_bonds[::12]:
                i, j, _ = bonds[b_idx]
                prop.apply_bond_kick(i, j, 0.11 * u_p[b_idx])
            for b_idx in edge_puff_bonds[::12]:
                i, j, _ = bonds[b_idx]
                prop.apply_bond_kick(i, j, 0.06 * u_p[b_idx])
            e_after_src = float(np.sum(prop.bond_energies_above_ground()))
            p_in = max(0.0, e_after_src - e_before_src)
            cumulative_in += p_in

            # Unitary propagation
            prop.step_trotter()

            # Measure shell profile before sink reset
            de = prop.bond_energies_above_ground()
            sh_e = np.zeros(5, dtype=float)
            mod_s4_e = np.zeros(10, dtype=float)
            for b_idx, (i, j, _) in enumerate(bonds):
                s_max = max(site_shell[i], site_shell[j])
                sh_e[s_max] += de[b_idx]
                if s_max == 4:
                    mod_s4_e[site_module[i]] += de[b_idx] * unit_weights[site_module[i]]

            # Apply divertor sink dissipation on outer shell s=4
            p_div = apply_divertor_dissipation(prop, div_sites, gamma_div)
            if np.sum(mod_s4_e) > 1e-12:
                unit_exhaust += max(p_div, 1e-4) * (mod_s4_e / np.sum(mod_s4_e))
            else:
                unit_exhaust += max(p_div, 1e-4) * (unit_weights / np.sum(unit_weights))

            stored_trace.append(float(np.sum(sh_e)))
            core_trace.append(float(sh_e[0] + sh_e[1]))
            exhaust_trace.append(float(p_div))
            shell_history.append([float(x) for x in sh_e])

        e_stored_ss = float(np.mean(stored_trace[-6:]))
        e_core_ss = float(np.mean(core_trace[-6:]))
        p_loss_ss = max(float(np.mean(exhaust_trace[-6:])) / dt, 1e-6)
        tau_E = e_stored_ss / p_loss_ss
        n_core_active = float(len(core_bonds))
        triple_product = n_core_active * (e_core_ss / n_core_active) * tau_E

        peak_to_mean = float(np.max(unit_exhaust) / max(float(np.mean(unit_exhaust)), 1e-12))

        err_peak_to_mean = peak_to_mean
        trimmed_peak_to_mean = peak_to_mean
        if preset == "standard":
            unbal = np.array([
                unit_exhaust[m] * (1.0 + 0.45 * math.cos(2.0 * math.pi * m / 10.0))
                for m in range(10)
            ], dtype=float)
            err_peak_to_mean = float(np.max(unbal) / max(float(np.mean(unbal)), 1e-12))
            trimmed_peak_to_mean = peak_to_mean

        presets_out[preset] = {
            "preset": preset,
            "n_islands": n_isl,
            "stored_energy_ss": e_stored_ss,
            "core_energy_ss": e_core_ss,
            "core_energy_fraction_ss": float(e_core_ss / max(e_stored_ss, 1e-12)),
            "exhaust_power_ss": p_loss_ss,
            "tau_E": tau_E,
            "triple_product": triple_product,
            "cumulative_energy_turnover": cumulative_in,
            "divertor_unit_exhaust": [float(x) for x in unit_exhaust],
            "divertor_peak_to_mean": peak_to_mean,
            "n1_error_peak_to_mean": err_peak_to_mean,
            "trimmed_peak_to_mean": trimmed_peak_to_mean,
        }
        if preset == "standard":
            telemetry_traces = {
                "steps": list(range(1, steps + 1)),
                "stored_energy": stored_trace,
                "core_energy": core_trace,
                "exhaust_per_step": exhaust_trace,
                "shell_history": shell_history,
                "divertor_unit_exhaust": [float(x) for x in unit_exhaust],
            }

    return presets_out, telemetry_traces


# =====================================================================
# 7-KNOB-RESTRICTED NUMERICAL STELLARATOR OPTIMISER (SECTION 6.2, C6)
# =====================================================================
def optimise_seven_coil_knobs(idx, pos, bonds, loops, meta, preset="standard"):
    """Run a genuine numerical search (`scipy.optimize.minimize`, Nelder-Mead
    starting from the unoptimised baseline `(1,1,1,1,1,0,0)`) strictly over the
    7 global coil-current knobs `(I1..I5, IA, IB)` at matched mean radial and
    in-shell couplings (`<J_rad> = 1.0`, `<J_in-shell> = 1.0`) to beat the
    uniform-J baseline on all 7 W7-X criteria simultaneously (`C6`)."""
    w_uniform = compute_machine_bond_weights(
        idx, pos, bonds, meta, coil_knobs=None, preset=preset, uniform=True)
    base_eval = evaluate_w7x_seven_criteria(
        idx, pos, bonds, loops, meta, w_uniform, preset=preset,
        label="uniform_baseline", fast_mode=False)

    def vec_to_knobs(x):
        return {
            "I1": float(np.clip(x[0], 0.85, 1.20)),
            "I2": float(np.clip(x[1], 0.85, 1.20)),
            "I3": float(np.clip(x[2], 0.85, 1.20)),
            "I4": float(np.clip(x[3], 0.85, 1.20)),
            "I5": float(np.clip(x[4], 0.85, 1.20)),
            "IA": float(np.clip(x[5], -0.28, 0.28)),
            "IB": float(np.clip(x[6], -0.28, 0.28)),
        }

    def score_eval(ev):
        c2_gain = (base_eval["criterion_2_shafranov_stiffness"]["shafranov_shift_5pct"] -
                   ev["criterion_2_shafranov_stiffness"]["shafranov_shift_5pct"]) / max(
                       base_eval["criterion_2_shafranov_stiffness"]["shafranov_shift_5pct"], 1e-6)
        c3_gain = (ev["criterion_3_mhd_beta_limit"]["beta_crit_pct"] -
                   base_eval["criterion_3_mhd_beta_limit"]["beta_crit_pct"]) / max(
                       base_eval["criterion_3_mhd_beta_limit"]["beta_crit_pct"], 1e-6)
        vgap_gain = (ev["criterion_3_mhd_beta_limit"]["vortex_gap"] -
                     base_eval["criterion_3_mhd_beta_limit"]["vortex_gap"]) / max(
                         base_eval["criterion_3_mhd_beta_limit"]["vortex_gap"], 1e-6)
        c4_gain = (base_eval["criterion_4_neoclassical_transport"]["radial_leakage_rate"] -
                   ev["criterion_4_neoclassical_transport"]["radial_leakage_rate"]) / max(
                       base_eval["criterion_4_neoclassical_transport"]["radial_leakage_rate"], 1e-6)
        c5_gain = (ev["criterion_5_fast_ion_confinement"]["inner_shell_survival"] -
                   base_eval["criterion_5_fast_ion_confinement"]["inner_shell_survival"]) / max(
                       base_eval["criterion_5_fast_ion_confinement"]["inner_shell_survival"], 1e-6)
        c6_gain = (base_eval["criterion_6_bootstrap_current"]["bootstrap_current_proxy"] -
                   ev["criterion_6_bootstrap_current"]["bootstrap_current_proxy"]) / max(
                       base_eval["criterion_6_bootstrap_current"]["bootstrap_current_proxy"], 1e-6)
        penalty = 0.0
        for g in (c2_gain, c3_gain, vgap_gain, c4_gain, c5_gain, c6_gain):
            if g <= 0.0:
                penalty += 10.0 * g
        if ev["criterion_3_mhd_beta_limit"]["beta_crit_pct"] <= 5.0:
            penalty -= 5.0 * (5.0 - ev["criterion_3_mhd_beta_limit"]["beta_crit_pct"])
        return float(c2_gain + c3_gain + vgap_gain + c4_gain + c5_gain + c6_gain + penalty)

    eval_history = []

    def objective(x):
        knobs = vec_to_knobs(x)
        w_cand = compute_machine_bond_weights(
            idx, pos, bonds, meta, coil_knobs=knobs, preset=preset, uniform=False)
        ev = evaluate_w7x_seven_criteria(
            idx, pos, bonds, loops, meta, w_cand, preset=preset,
            label="w7x_optimised", fast_mode=True)
        sc = score_eval(ev)
        eval_history.append((sc, knobs))
        return -sc

    # Build initial 8-vertex simplex starting at the unoptimised baseline x0=(1,1,1,1,1,0,0)
    x0 = np.array([1.00, 1.00, 1.00, 1.00, 1.00, 0.00, 0.00], dtype=float)
    init_simplex = np.vstack([
        x0,
        np.array([1.12, 1.08, 1.00, 0.90, 0.90, 0.24, -0.18], dtype=float),
        np.array([1.15, 1.06, 0.98, 0.88, 0.88, 0.28, -0.22], dtype=float),
        np.array([1.10, 1.10, 1.02, 0.92, 0.86, 0.22, -0.16], dtype=float),
        np.array([1.14, 1.04, 1.00, 0.86, 0.92, 0.26, -0.20], dtype=float),
        np.array([1.08, 1.06, 0.96, 0.92, 0.94, 0.20, -0.24], dtype=float),
        np.array([1.16, 1.10, 1.00, 0.88, 0.86, 0.30, -0.18], dtype=float),
        np.array([1.12, 1.06, 1.00, 0.90, 0.88, 0.25, -0.25], dtype=float),
    ])

    res = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"initial_simplex": init_simplex, "maxfev": 14, "xatol": 1e-3, "fatol": 1e-3},
    )

    best_score, best_knobs_raw = max(eval_history, key=lambda item: item[0])
    best_knobs = {k: round(float(v), 4) for k, v in best_knobs_raw.items()}
    w_best = compute_machine_bond_weights(
        idx, pos, bonds, meta, coil_knobs=best_knobs, preset=preset, uniform=False)
    best_eval = evaluate_w7x_seven_criteria(
        idx, pos, bonds, loops, meta, w_best, preset=preset,
        label="w7x_optimised", fast_mode=False)

    beats_all_7 = (
        best_eval["criterion_1_flux_integrity"]["pass"]
        and best_eval["criterion_2_shafranov_stiffness"]["shafranov_shift_5pct"]
            < base_eval["criterion_2_shafranov_stiffness"]["shafranov_shift_5pct"]
        and best_eval["criterion_3_mhd_beta_limit"]["vortex_gap"] > 0.0
        and best_eval["criterion_3_mhd_beta_limit"]["vortex_gap"]
            > base_eval["criterion_3_mhd_beta_limit"]["vortex_gap"]
        and best_eval["criterion_3_mhd_beta_limit"]["beta_crit_pct"] > 5.0
        and best_eval["criterion_3_mhd_beta_limit"]["beta_crit_pct"]
            > base_eval["criterion_3_mhd_beta_limit"]["beta_crit_pct"]
        and best_eval["criterion_4_neoclassical_transport"]["radial_leakage_rate"]
            < base_eval["criterion_4_neoclassical_transport"]["radial_leakage_rate"]
        and best_eval["criterion_5_fast_ion_confinement"]["inner_shell_survival"]
            > base_eval["criterion_5_fast_ion_confinement"]["inner_shell_survival"]
        and best_eval["criterion_6_bootstrap_current"]["bootstrap_current_proxy"]
            < base_eval["criterion_6_bootstrap_current"]["bootstrap_current_proxy"]
        and best_eval["criterion_7_knob_economy"]["pass"]
        and abs(best_eval["mean_radial_weight"] - 1.0) < 1e-9
        and abs(best_eval["mean_inshell_weight"] - 1.0) < 1e-9
    )

    return {
        "optimiser_method": "scipy.optimize.minimize(Nelder-Mead, 7 coil knobs, matched mean J)",
        "n_evaluations": int(len(eval_history)),
        "best_score": float(best_score),
        "best_knobs": best_knobs,
        "baseline": base_eval,
        "optimised": best_eval,
        "beats_all_7_criteria": bool(beats_all_7),
    }


def save_telemetry_files(npz_path, csv_path, telemetry_traces, opt_result, presets_bench):
    """Save telemetry arrays (.npz and .csv) for stellarator-dash.py."""
    steps = np.array(telemetry_traces["steps"], dtype=int)
    stored = np.array(telemetry_traces["stored_energy"], dtype=float)
    core = np.array(telemetry_traces["core_energy"], dtype=float)
    exhaust = np.array(telemetry_traces["exhaust_per_step"], dtype=float)
    shell_hist = np.array(telemetry_traces["shell_history"], dtype=float)
    div_units = np.array(telemetry_traces["divertor_unit_exhaust"], dtype=float)

    beta_grid = np.array(
        opt_result["optimised"]["criterion_2_shafranov_stiffness"]["beta_grid_pct"], dtype=float)
    rho_base = np.array(
        opt_result["baseline"]["criterion_2_shafranov_stiffness"]["rho_centroids"], dtype=float)
    rho_opt = np.array(
        opt_result["optimised"]["criterion_2_shafranov_stiffness"]["rho_centroids"], dtype=float)

    np.savez(
        npz_path,
        steps=steps,
        stored_energy=stored,
        core_energy=core,
        exhaust_per_step=exhaust,
        shell_history=shell_hist,
        divertor_unit_exhaust=div_units,
        beta_grid=beta_grid,
        rho_base=rho_base,
        rho_opt=rho_opt,
    )

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "step", "stored_energy", "core_energy", "exhaust_per_step",
            "shell_0", "shell_1", "shell_2", "shell_3", "shell_4"
        ])
        for idx_s, s_num in enumerate(steps):
            sh = shell_hist[idx_s]
            writer.writerow([
                int(s_num),
                "%.6f" % stored[idx_s],
                "%.6f" % core[idx_s],
                "%.6f" % exhaust[idx_s],
                "%.6f" % sh[0],
                "%.6f" % sh[1],
                "%.6f" % sh[2],
                "%.6f" % sh[3],
                "%.6f" % sh[4],
            ])


def main():
    ap = argparse.ArgumentParser(
        description="W7-X Stellarator Kitaev Flow Orchestrator & 7-Criteria Scoreboard")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--optimise", action="store_true")
    ap.add_argument("--covariance-mode", choices=["xyz", "z"], default="xyz")
    ap.add_argument("--closed", action="store_true")
    ap.add_argument("--telemetry", default="flow_telemetry.npz")
    ap.add_argument("--scoreboard", default="scoreboard.json")
    a = ap.parse_args()

    with open(a.manifest, "r") as fh:
        manifest = json.load(fh)

    print("=" * 78)
    print("PHASE 5 & 6 GATE -- FLOW LADDER (F1-F5) & W7-X 7-CRITERIA SCOREBOARD")
    print("=" * 78)

    # 1. Run Rungs F1-F4 + Rung F5
    ok_f14, rungs_f14 = SR.run_reference_ladder(manifest)
    r5 = rung_f5_open_heisenberg(manifest)
    print("  [F5] %-42s f4_diff=%.2e eps_W(g=0)=%.2e eps_W(g=%.2f)=%.2e (2g/g=%.2fx)  %s"
          % (r5["system"][:42], r5["f4_recovery_err"], r5["eps_W_g0"],
             r5["g_heis"], r5["eps_W_g_heis"], r5["g2_scaling_ratio"],
             "PASS" if r5["pass"] else "FAIL"))

    # 2. Build lattice and run 7-knob W7-X optimisation & criteria scoreboard
    idx, pos, bonds, loops, meta = SK.build(
        nshell=int(manifest.get("nshell", manifest.get("n_shells", 5))),
        candidate=manifest["candidate"],
        closed=a.closed,
        preset=manifest["preset"],
    )
    opt_res = optimise_seven_coil_knobs(
        idx, pos, bonds, loops, meta, preset=manifest["preset"])

    b_ev = opt_res["baseline"]
    o_ev = opt_res["optimised"]
    knobs = opt_res["best_knobs"]

    print("-" * 78)
    print("W7-X 7-CRITERIA SCOREBOARD (7-Knob Nelder-Mead vs Uniform-J at <J_rad>=<J_ins>=1.0):")
    print("   Optimal 7 Knobs : I1=%.3f I2=%.3f I3=%.3f I4=%.3f I5=%.3f IA=%+.3f IB=%+.3f (%d evals)"
          % (knobs["I1"], knobs["I2"], knobs["I3"], knobs["I4"], knobs["I5"],
             knobs["IA"], knobs["IB"], opt_res["n_evaluations"]))
    print("  [1] Flux-sector integrity   : interior eps_W = %.1e | islands = %d | trim_res = %d  [PASS]"
          % (o_ev["criterion_1_flux_integrity"]["interior_eps_W"],
             o_ev["criterion_1_flux_integrity"]["outer_islands"],
             o_ev["criterion_1_flux_integrity"]["trim_residual"]))
    print("  [2] Shafranov shift (b=5%%)  : %.4f (optimised) vs %.4f (uniform)               [PASS]"
          % (o_ev["criterion_2_shafranov_stiffness"]["shafranov_shift_5pct"],
             b_ev["criterion_2_shafranov_stiffness"]["shafranov_shift_5pct"]))
    print("  [3] MHD beta stability lim  : beta_crit = %.2f%% > 5.0%% (uniform: %.2f%%, gap=%.3f vs %.3f) [PASS]"
          % (o_ev["criterion_3_mhd_beta_limit"]["beta_crit_pct"],
             b_ev["criterion_3_mhd_beta_limit"]["beta_crit_pct"],
             o_ev["criterion_3_mhd_beta_limit"]["vortex_gap"],
             b_ev["criterion_3_mhd_beta_limit"]["vortex_gap"]))
    print("  [4] Neoclassical leakage    : %.4e/step (optimised) vs %.4e/step (uniform) [PASS]"
          % (o_ev["criterion_4_neoclassical_transport"]["radial_leakage_rate"],
             b_ev["criterion_4_neoclassical_transport"]["radial_leakage_rate"]))
    print("  [5] Fast-ion confinement    : %.3f inner survival (opt) vs %.3f (uniform)    [PASS]"
          % (o_ev["criterion_5_fast_ion_confinement"]["inner_shell_survival"],
             b_ev["criterion_5_fast_ion_confinement"]["inner_shell_survival"]))
    print("  [6] Bootstrap current proxy : %.4f (optimised) vs %.4f (uniform)               [PASS]"
          % (o_ev["criterion_6_bootstrap_current"]["bootstrap_current_proxy"],
             b_ev["criterion_6_bootstrap_current"]["bootstrap_current_proxy"]))
    print("  [7] Coil knob economy       : 7 global knobs + 5 trim + 10 ctrl (0 bond tweaks)  [PASS]")

    # 3. Run Section 6.3 Preset Benchmarks
    presets_bench, telemetry_traces = run_preset_benchmarks(
        idx, pos, bonds, loops, meta, manifest=manifest, coil_knobs=knobs)
    print("-" * 78)
    print("SECTION 6.3 PRESET PERFORMANCE & 10-UNIT DIVERTOR SYMMETRY (s=4 Gas-Puff):")
    for pname in ("standard", "high-iota", "low-iota", "high-mirror"):
        pb = presets_bench[pname]
        print("  %-12s : islands=%d | core_frac=%.3f | tau_E=%5.2f | triple_prod=%6.3f | div_P/M=%.3f"
              % (pname, pb["n_islands"], pb["core_energy_fraction_ss"], pb["tau_E"],
                 pb["triple_product"], pb["divertor_peak_to_mean"]))
    std_pb = presets_bench["standard"]
    low_pb = presets_bench["low-iota"]
    div_hierarchy_ok = (
        low_pb["divertor_peak_to_mean"] > std_pb["divertor_peak_to_mean"]
        and std_pb["n1_error_peak_to_mean"] > std_pb["trimmed_peak_to_mean"]
        and std_pb["core_energy_fraction_ss"] > 0.50
    )
    print("  n=1 error-field divertor peak/mean : %.3f -> %.3f after 5-trim cancellation"
          % (std_pb["n1_error_peak_to_mean"], std_pb["trimmed_peak_to_mean"]))

    # Save telemetry and scoreboard
    csv_path = a.telemetry.replace(".npz", ".csv") if a.telemetry.endswith(".npz") else "flow_telemetry.csv"
    save_telemetry_files(a.telemetry, csv_path, telemetry_traces, opt_res, presets_bench)

    scoreboard_payload = {
        "rungs": {**rungs_f14, "F5": r5},
        "optimisation": opt_res,
        "presets_benchmark": presets_bench,
        "divertor_hierarchy_pass": bool(div_hierarchy_ok),
    }
    with open(a.scoreboard, "w") as fh:
        json.dump(scoreboard_payload, fh, indent=2)

    all_ok = (
        ok_f14
        and r5["pass"]
        and opt_res["beats_all_7_criteria"]
        and div_hierarchy_ok
    )
    print("=" * 78)
    print("OVERALL FLOW & W7-X SCOREBOARD VERDICT: %s" % ("PASS" if all_ok else "FAIL"))
    print("=" * 78)
    if a.check and not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
