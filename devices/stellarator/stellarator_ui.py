# -*- coding: us-ascii -*-
"""stellarator_ui.py -- native live dashboard for the W7-X Kitaev stellarator.

A PySide6 (Qt) desktop window that runs parameter sweeps in a background
thread and shows every step while it is being computed: parameter forms with
sweep ranges, matplotlib charts (step trace, sweep result or heatmap, detail),
a results table, a log, and a QPainter 3D view of the 1080 sites, 1620 bonds,
85 coils and 10 divertor units coloured by the running job.

The physics comes from stellarator.py in the same folder; this file holds only
the dashboard (job definitions, job runner and window).

    python3 stellarator_ui.py                          # open the window
    python3 stellarator_ui.py --selftest shot.png      # headless check + screenshot
    python3 stellarator_ui.py --selftest shot.png --selftest-all
    python3 stellarator.py ui                          # same, via the main file

Jobs (each sweepable over one parameter, the coil landscape over two):
  ness      core-to-divertor Lindblad NESS pulse on the 1080-site machine
  coils     3D Biot-Savart coil-knob landscape (fast vacuum metrics)
  criteria  full W7-X 7-criteria evaluation at given coil knobs
  manifold  PyQrack 16q / 27q manifold flow vs exact Majorana reference
  link      DEJMPS distilled-ebit link budget vs LO-QPD sampling cost

Requirements: stellarator.py (and its numpy / scipy / pyqrack stack),
matplotlib, PySide6 (pip install PySide6).
"""

import argparse
import collections
import csv
import json
import math
import os
import sys
import threading
import time

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):
    os.environ.setdefault("PYQRACK_SHARED_LIB_PATH", QRACK_LIB_PATH)
os.environ.setdefault("QRACK_DISABLE_QUNIT_FIDELITY_GUARD", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from stellarator import (  # noqa: E402  physics engine (same folder)
    BiotSavartCoilModel,
    EBIT_STORAGE_SHARE,
    HAS_OPENCL_SO,
    LAM_PAR_TRIM,
    MajoranaFlowPropagator,
    PRESET_ISLANDS,
    QrackSimulator,
    _extract_exact_cycle_subgraph,
    _parallel_reduction_pct,
    _run_divertor_pulse,
    apply_bond_step,
    build,
    build_stabilizers,
    configure_simulator_27q,
    dejmps_link_budget,
    evaluate_w7x_seven_criteria,
    extract_divertor_target_map,
    generate_coil_filaments,
    prepare_shared_dimer_state,
    read_bond_expectation,
    read_pauli_word_inplace,
    seed_edge_island_chain,
)


class _JobCancelled(Exception):
    pass


UI_PRESETS = ["standard", "high-iota", "low-iota", "high-mirror"]
UI_KNOB_PARAMS = [
    {"key": "I1", "label": "I1 non-planar", "type": "float", "min": 0.65, "max": 1.35, "default": 1.0, "group": "coils"},
    {"key": "I2", "label": "I2 non-planar", "type": "float", "min": 0.65, "max": 1.35, "default": 1.0, "group": "coils"},
    {"key": "I3", "label": "I3 non-planar", "type": "float", "min": 0.65, "max": 1.35, "default": 1.0, "group": "coils"},
    {"key": "I4", "label": "I4 non-planar", "type": "float", "min": 0.65, "max": 1.35, "default": 1.0, "group": "coils"},
    {"key": "I5", "label": "I5 non-planar", "type": "float", "min": 0.65, "max": 1.35, "default": 1.0, "group": "coils"},
    {"key": "IA", "label": "IA planar", "type": "float", "min": -0.35, "max": 0.35, "default": 0.0, "group": "coils"},
    {"key": "IB", "label": "IB planar", "type": "float", "min": -0.35, "max": 0.35, "default": 0.0, "group": "coils"},
]
UI_KNOB_KEYS = [p["key"] for p in UI_KNOB_PARAMS]

UI_JOBS = collections.OrderedDict([
    ("ness", {
        "label": "NESS pulse",
        "title": "Core-to-divertor Lindblad NESS pulse (1080 sites)",
        "about": ("Heats the 20 axis sites with the CPTP creation bath and cools the 60 divertor "
                  "targets with the zero-T bath; streams stored, core and exhaust energy each step. "
                  "About 0.4 s per step."),
        "params": [
            {"key": "preset", "label": "Preset", "type": "select", "options": UI_PRESETS, "default": "standard"},
            {"key": "field", "label": "Error field", "type": "select",
             "options": ["clean", "error", "trimmed"], "default": "clean"},
            {"key": "islands", "label": "Edge islands", "type": "select",
             "options": ["preset", "none"], "default": "preset"},
            {"key": "coils", "label": "Coil currents", "type": "select",
             "options": ["optimised", "baseline", "custom"], "default": "optimised"},
            {"key": "steps", "label": "Steps", "type": "int", "min": 5, "max": 600, "default": 135},
            {"key": "dt", "label": "dt", "type": "float", "min": 0.05, "max": 3.0, "default": 1.2},
            {"key": "gamma_heat", "label": "gamma_heat", "type": "float", "min": 0.0, "max": 1.0, "default": 0.14},
            {"key": "gamma_div", "label": "gamma_div", "type": "float", "min": 0.0, "max": 4.0, "default": 0.95},
            {"key": "error_amp", "label": "Error amp", "type": "float", "min": 0.0, "max": 60.0, "default": 15.0},
            {"key": "lam_par", "label": "lambda_par", "type": "float", "min": 0.0, "max": 20.0, "default": LAM_PAR_TRIM},
        ] + UI_KNOB_PARAMS,
        "step_metrics": ["stored", "core", "exhaust", "injected", "cum_in", "cum_ex", "ledger_err"],
        "summary_metrics": ["tau_E", "e_stored_ss", "core_fraction", "ex_in_ratio", "S_vN",
                            "tor_p2m", "per_p2m", "pol_p2m", "confinement_product", "ledger_err",
                            "edge_rms_reduction_pct", "bond_par_reduction_pct"],
        "default_step_metric": "stored",
        "default_summary_metric": "tau_E",
    }),
    ("coils", {
        "label": "Coil landscape",
        "title": "3D Biot-Savart coil-knob landscape",
        "about": ("Vacuum field metrics from the 50 non-planar + 20 planar coil filaments for each "
                  "knob setting, as relative change vs the uniform baseline. The score is the plain "
                  "mean of the six relative improvements (not the Powell objective). About 2 ms per "
                  "point, so 2D sweeps of a few thousand points are interactive."),
        "params": [
            {"key": "preset", "label": "Preset", "type": "select", "options": UI_PRESETS, "default": "standard"},
        ] + UI_KNOB_PARAMS,
        "two_d": True,
        "step_metrics": [],
        "summary_metrics": ["w_mag_rel", "ripple_rel", "geo_asym_rel", "out_trap_ratio_rel",
                            "rad_asym_rel", "rad_outboard_rel", "std_w", "score"],
        "default_summary_metric": "score",
    }),
    ("criteria", {
        "label": "7 criteria",
        "title": "Full W7-X 7-criteria evaluation",
        "about": ("Polished ground sector, Shafranov shift, lattice MHD limit, neoclassical leakage, "
                  "fast-ion survival and geodesic asymmetry at the given coil knobs. About 65 s per "
                  "point; progress is reported per criterion."),
        "params": [
            {"key": "preset", "label": "Preset", "type": "select", "options": UI_PRESETS, "default": "standard"},
            {"key": "coils", "label": "Coil currents", "type": "select",
             "options": ["optimised", "baseline", "custom"], "default": "optimised"},
        ] + UI_KNOB_PARAMS,
        "step_metrics": [],
        "summary_metrics": ["shafranov_shift_5pct", "beta_crit_pct", "lambda_crit", "min_gap_5pct",
                            "n_unstable_5pct", "ripple_eps_B", "leak_iso", "leak_out",
                            "fast_ion_survival", "trap_ratio", "ps_dipole", "geodesic_asym"],
        "default_summary_metric": "beta_crit_pct",
    }),
    ("manifold", {
        "label": "PyQrack manifold",
        "title": "PyQrack manifold flow vs exact Majorana reference",
        "about": ("Bell-dimer + missing-colour Gaussian state on one manifest block, Trotterised "
                  "with the 5-gate bond decomposition, every bond read out each step and compared "
                  "with the exact Majorana propagator. 16q: about 0.06 s per step; 27q: about 5 s."),
        "params": [
            {"key": "qubits", "label": "Width", "type": "select",
             "options": ["16 qubits", "27 qubits"], "default": "16 qubits"},
            {"key": "block", "label": "Block", "type": "int", "min": 0, "max": 39, "default": 1},
            {"key": "steps", "label": "Steps", "type": "int", "min": 1, "max": 200, "default": 20},
            {"key": "dt", "label": "dt", "type": "float", "min": 0.005, "max": 0.5, "default": 0.04},
            {"key": "kick", "label": "Initial kick", "type": "float", "min": 0.0, "max": 1.5, "default": 0.0},
            {"key": "ace_max_qb", "label": "ACE max qb (27q)", "type": "int", "min": 8, "max": 27, "default": 23},
            {"key": "sdrp", "label": "SDRP (27q)", "type": "float", "min": 0.0, "max": 0.3, "default": 0.0},
        ],
        "step_metrics": ["rms", "max_err", "mean_bond_qrack", "mean_bond_exact"],
        "summary_metrics": ["final_rms", "max_rms", "mean_rms", "final_max_err", "W0"],
        "default_step_metric": "rms",
        "default_summary_metric": "max_rms",
    }),
    ("link", {
        "label": "Ebit link budget",
        "title": "Distilled-ebit link budget vs LO-QPD sampling",
        "about": ("Exact DEJMPS rounds, raw ebit cost and pair-memory requirement for N teleported "
                  "seam cuts at total link infidelity delta, against the LO-QPD shot count "
                  "(|cos 2theta| + 3|sin 2theta|)^(2N) for the same cuts. Instant."),
        "params": [
            {"key": "n_ebits", "label": "Ebits per shot", "type": "int", "min": 1, "max": 1e6,
             "default": 5200, "log": True},
            {"key": "F0", "label": "Raw ebit F0", "type": "float", "min": 0.5, "max": 0.99999, "default": 0.99},
            {"key": "target_delta", "label": "Total infidelity delta", "type": "float", "min": 1e-5,
             "max": 0.5, "default": 0.01, "log": True},
            {"key": "storage_share", "label": "Storage share", "type": "float", "min": 0.01, "max": 0.99,
             "default": EBIT_STORAGE_SHARE},
            {"key": "t_round_us", "label": "Classical round (us)", "type": "float", "min": 0.001,
             "max": 1e4, "default": 1.0, "log": True},
            {"key": "theta", "label": "Cut angle theta (rad)", "type": "float", "min": 1e-4, "max": 0.7854,
             "default": 0.05},
            {"key": "shots", "label": "Target shots", "type": "float", "min": 1, "max": 1e9,
             "default": 1e4, "log": True},
        ],
        "step_metrics": [],
        "summary_metrics": ["dejmps_rounds", "log10_raw_per_distilled", "log10_raw_total",
                            "log10_Tmem_over_tround", "log10_Tmem_s", "log10_eps_req",
                            "log10_shots_LO_QPD", "ft_required"],
        "default_summary_metric": "log10_raw_total",
    }),
])


class LiveMachine(object):
    """Lazily built, shared machine context for the dashboard jobs."""

    def __init__(self, manifest_path, scoreboard_path):
        self.manifest_path = manifest_path
        self.scoreboard_path = scoreboard_path
        self._lock = threading.Lock()
        self._ctx = None
        self._b_ext = {}
        self._u_isl = {}
        self._geom = None
        self.last_frame_t = 0.0

    def ctx(self, log=None):
        with self._lock:
            if self._ctx is None:
                if log:
                    log("Building 1080-site lattice and Biot-Savart coil model ...")
                with open(self.manifest_path, "r") as fh:
                    manifest = json.load(fh)
                idx, pos, bonds, loops, meta = build(
                    nshell=int(manifest.get("nshell", manifest.get("n_shells", 5))),
                    candidate=manifest["candidate"], closed=False, preset=manifest["preset"])
                n_sites = len(pos)
                u_ground = np.array(meta["u_ground"], dtype=float)
                stabs, _ = build_stabilizers(bonds, loops, u_ref=u_ground)
                site_shell = meta["site_shell"]
                div_units_sites, div_pol_of_site = extract_divertor_target_map(idx)
                core_sites = sorted(
                    int(idx[k]) for k in idx if isinstance(k, tuple) and k[0] == "core")
                bs_model = BiotSavartCoilModel(pos, bonds, meta, n_pts=48)
                opt_knobs = None
                try:
                    with open(self.scoreboard_path, "r") as fh:
                        opt_knobs = json.load(fh)["optimisation"]["best_knobs"]
                except Exception:
                    opt_knobs = None
                self._ctx = {
                    "manifest": manifest, "idx": idx, "pos": pos, "bonds": bonds,
                    "loops": loops, "meta": meta, "n_sites": n_sites,
                    "nshell": int(meta["nshell"]), "u_ground": u_ground, "stabs": stabs,
                    "site_shell": site_shell, "div_units_sites": div_units_sites,
                    "div_pol_of_site": div_pol_of_site, "core_sites": core_sites,
                    "edge_sites": [s for s in range(n_sites) if site_shell[s] >= 3],
                    "bs_model": bs_model, "opt_knobs": opt_knobs,
                    "bs_base": {p: bs_model.physics_metrics(preset=p, uniform=True) for p in UI_PRESETS},
                }
                site_block = [-1] * n_sites
                for blk in manifest["blocks"]:
                    for st_ in blk["sites"]:
                        site_block[int(st_)] = int(blk["block_id"])
                self._ctx["site_block"] = site_block
                self._ctx["seam_bonds"] = [
                    k for k, b in enumerate(bonds)
                    if site_block[b[0]] != site_block[b[1]] and site_block[b[0]] >= 0 and site_block[b[1]] >= 0]
                if log:
                    log("Machine ready: %d sites, %d bonds, optimised knobs %s"
                        % (n_sites, len(bonds), "loaded" if opt_knobs else "not found (using baseline)"))
            return self._ctx

    def geometry(self):
        """Static 3D geometry for the dashboard as numpy arrays: sites,
        bonds, shells, modules, blocks, core and divertor sites, seam bonds,
        divertor-unit centroids and all coil filaments."""
        c = self.ctx()
        with self._lock:
            if self._geom is None:
                n = c["n_sites"]
                P = np.asarray(c["pos"], dtype=float)
                bonds = [(int(b[0]), int(b[1]), int(b[2])) for b in c["bonds"]]
                bi = np.array([b[0] for b in bonds], dtype=int)
                bj = np.array([b[1] for b in bonds], dtype=int)
                incident = [[] for _ in range(n)]
                for k, (i, j, _) in enumerate(bonds):
                    incident[i].append(k)
                    incident[j].append(k)
                coils = []
                for fam, lst in generate_coil_filaments(n_pts=40).items():
                    for k, cl in enumerate(lst):
                        xyz = np.asarray(cl["xyz"], dtype=float)
                        coils.append({"fam": fam, "k": k, "type": int(cl.get("type", 0)), "xyz": xyz})
                coil_phi = np.array([
                    math.atan2(float(cl["xyz"][:, 1].sum()), float(cl["xyz"][:, 0].sum())) % (2 * math.pi)
                    for cl in coils])
                self._geom = {
                    "n": n, "P": P,
                    "phi": np.mod(np.arctan2(P[:, 1], P[:, 0]), 2 * math.pi),
                    "bonds": bonds, "bi": bi, "bj": bj,
                    "bcolour": np.array([b[2] for b in bonds], dtype=int),
                    "shell": np.array(_ui_per_site(c["site_shell"], n), dtype=int),
                    "module": np.array(_ui_per_site(c["meta"]["site_module"], n), dtype=int),
                    "block": np.array(c["site_block"], dtype=int),
                    "core": set(int(x) for x in c["core_sites"]),
                    "div": set(int(x) for u in c["div_units_sites"] for x in u),
                    "units": np.array([P[u].mean(axis=0) for u in c["div_units_sites"]]),
                    "seam_set": set(c["seam_bonds"]),
                    "incident": [np.array(v, dtype=int) for v in incident],
                    "coils": coils, "coil_phi": coil_phi,
                }
            return self._geom

    def external_field(self, amp):
        key = round(float(amp), 9)
        with self._lock:
            if key not in self._b_ext:
                self._b_ext[key] = self._ctx["bs_model"].external_busbar_error_field(amp=float(amp), n_pts=48)
            return self._b_ext[key]

    def island_gauge(self, preset):
        with self._lock:
            if preset not in self._u_isl:
                c = self._ctx
                u_p, _, _ = seed_edge_island_chain(
                    c["idx"], c["bonds"], c["stabs"], c["nshell"], preset=preset, u_base=c["u_ground"])
                self._u_isl[preset] = u_p
            return self._u_isl[preset]


def _ui_knobs(ctx, p):
    mode = p.get("coils", "custom")
    if mode == "baseline":
        return None
    if mode == "optimised" and ctx["opt_knobs"]:
        return {k: float(ctx["opt_knobs"][k]) for k in UI_KNOB_KEYS}
    if mode == "optimised":
        return None
    return {k: float(p[k]) for k in UI_KNOB_KEYS}


def _ui_per_site(m, n):
    """Per-site integer list from a list/array or a {site: value} mapping."""
    if isinstance(m, dict):
        return [int(m.get(i, m.get(str(i), -1))) for i in range(n)]
    return [int(x) for x in m]


def _ui_vals(a, nd=4):
    return [round(float(x), nd) for x in np.asarray(a, dtype=float).ravel()]


def _ui_coil_frame(bs, knobs, preset, trim=None, ctrl=None):
    """Coil currents for the 3D view: per family, with the preset nominal
    currents for colouring deviations."""
    i_np, i_pc = bs.currents_from_knobs(coil_knobs=knobs, preset=preset, uniform=knobs is None)
    n_np, n_pc = bs.currents_from_knobs(coil_knobs=None, preset=preset, uniform=True)
    return {
        "np": _ui_vals(i_np), "pc": _ui_vals(i_pc), "np0": _ui_vals(n_np), "pc0": _ui_vals(n_pc),
        "trim": _ui_vals(trim if trim is not None else np.zeros(5)),
        "ctrl": _ui_vals(ctrl if ctrl is not None else np.zeros(10)),
    }


def _ui_island_bonds(c, u_p):
    return [int(k) for k in np.nonzero(np.asarray(u_p) != np.asarray(c["u_ground"]))[0]]


def _ui_point_ness(machine, p, emit_step, check):
    c = machine.ctx()
    bs = c["bs_model"]
    preset = p["preset"]
    knobs = _ui_knobs(c, p)
    u_p = machine.island_gauge(preset) if p["islands"] == "preset" else c["u_ground"]
    extra = {}
    tr = ct = None
    if p["field"] == "clean":
        w_p, _, _, _ = bs.preset_bond_weights_and_field(coil_knobs=knobs, preset=preset, uniform=False)
    else:
        B_ext_sites, B_ext_bonds = machine.external_field(p["error_amp"])
        if p["field"] == "error":
            w_p, _, _, _ = bs.preset_bond_weights_and_field(
                coil_knobs=knobs, preset=preset, uniform=False,
                b_ext_sites=B_ext_sites, b_ext_bonds=B_ext_bonds)
        else:
            i_np, i_pc = bs.currents_from_knobs(coil_knobs=knobs, preset=preset, uniform=False)
            B0b = bs.field_bonds(i_np, i_pc)
            bhat = B0b / np.maximum(np.linalg.norm(B0b, axis=1, keepdims=True), 1e-12)
            tr, ct, rb, ra, pr = bs.solve_divertor_error_mitigation(
                c["edge_sites"], B_ext_sites, reg=0.5,
                b_hat_bonds=bhat, b_ext_bonds=B_ext_bonds, lam_par=float(p["lam_par"]))
            if float(p["lam_par"]) == 0.0:
                pr = _parallel_reduction_pct(bs, bhat, B_ext_bonds, tr, ct)
            extra["edge_rms_reduction_pct"] = float(100.0 * (1.0 - ra / max(rb, 1e-12)))
            extra["bond_par_reduction_pct"] = float(pr)
            w_p, _, _, _ = bs.preset_bond_weights_and_field(
                coil_knobs=knobs, preset=preset, uniform=False, trim_amps=tr, ctrl_amps=ct,
                b_ext_sites=B_ext_sites, b_ext_bonds=B_ext_bonds)
    steps = int(p["steps"])
    emit_step(None, steps, None, keep=True, frame={
        "label": "NESS %s, %s field, %s islands" % (preset, p["field"], p["islands"]),
        "coils": _ui_coil_frame(bs, knobs, preset, tr, ct),
        "isl_bonds": _ui_island_bonds(c, u_p),
        "weights": _ui_vals(w_p),
        "bond": None, "bond_ids": None, "highlight": [], "seams": [], "units": [0.0] * 10,
    })

    def _cb(step_idx, d):
        check()
        be = d.pop("bond_energy")
        ue = np.asarray(d["unit_exhaust"], dtype=float)
        emit_step(step_idx, steps, d, frame={
            "bond": _ui_vals(be), "bond_mode": "energy",
            "units": _ui_vals(ue / max(float(np.mean(ue)), 1e-12), 3),
        })

    run = _run_divertor_pulse(
        c["n_sites"], c["bonds"], u_p, w_p, c["site_shell"],
        c["div_units_sites"], c["div_pol_of_site"], c["core_sites"],
        steps=steps, dt=float(p["dt"]), gamma_heat=float(p["gamma_heat"]),
        gamma_div=float(p["gamma_div"]), on_step=_cb)
    out = {
        "tau_E": run["tau_E"],
        "e_stored_ss": run["e_stored_ss"],
        "core_fraction": float(run["e_core_ss"] / max(abs(run["e_stored_ss"]), 1e-12)),
        "ex_in_ratio": run["steady_state_ex_in_ratio"],
        "S_vN": run["von_neumann_entropy_nats"],
        "tor_p2m": run["tor_p2m"],
        "per_p2m": run["per_p2m"],
        "pol_p2m": run["pol_p2m"],
        "confinement_product": run["confinement_product"],
        "ledger_err": run["max_ledger_err"],
        "unit_norm": [float(x) for x in run["unit_norm"]],
        "pol_exhaust": [float(x) for x in run["pol_exhaust"]],
        "n_islands": int(PRESET_ISLANDS.get(preset, 5)) if p["islands"] == "preset" else 0,
    }
    out.update(extra)
    return out


def _ui_point_coils(machine, p, emit_step, check):
    c = machine.ctx()
    check()
    knobs = {k: float(p[k]) for k in UI_KNOB_KEYS}
    m = c["bs_model"].physics_metrics(coil_knobs=knobs, preset=p["preset"], uniform=False)
    now = time.time()
    if p.get("_last") or now - machine.last_frame_t > 0.12:
        machine.last_frame_t = now
        w_k, _, _, _ = c["bs_model"].preset_bond_weights_and_field(
            coil_knobs=knobs, preset=p["preset"], uniform=False)
        emit_step(None, 0, None, frame={
            "label": "coil knobs " + ", ".join("%s=%.3f" % (k, knobs[k]) for k in UI_KNOB_KEYS),
            "coils": _ui_coil_frame(c["bs_model"], knobs, p["preset"]),
            "bond": _ui_vals(w_k), "bond_mode": "weight", "bond_ids": None,
            "weights": None, "isl_bonds": [], "highlight": [], "seams": [], "units": None,
        })
    b = c["bs_base"][p["preset"]]

    def rel(key, sign=1.0):
        return float(sign * 100.0 * (m[key] - b[key]) / max(abs(b[key]), 1e-12))

    out = {
        "w_mag_rel": rel("w_mag"),
        "ripple_rel": rel("ripple"),
        "geo_asym_rel": rel("geo_asym"),
        "out_trap_ratio_rel": rel("out_trap_ratio"),
        "rad_asym_rel": rel("rad_asym"),
        "rad_outboard_rel": rel("rad_outboard"),
        "std_w": float(m["std_w"]),
        "raw": {k: float(v) for k, v in m.items()},
    }
    # Same improvement directions as the Powell objective: deeper well, lower
    # ripple / geodesic / radial asymmetry, higher outboard trapping.
    out["score"] = float(np.mean([
        out["w_mag_rel"], -out["ripple_rel"], -out["geo_asym_rel"], out["out_trap_ratio_rel"],
        -out["rad_asym_rel"], -out["rad_outboard_rel"]]))
    return out


def _ui_point_criteria(machine, p, emit_step, check, emit_log):
    c = machine.ctx()
    knobs = _ui_knobs(c, p)
    w_k, _, _, _ = c["bs_model"].preset_bond_weights_and_field(
        coil_knobs=knobs, preset=p["preset"], uniform=knobs is None)
    emit_step(None, 0, None, keep=True, frame={
        "label": "7 criteria, %s, coils %s" % (p["preset"], p.get("coils", "custom")),
        "coils": _ui_coil_frame(c["bs_model"], knobs, p["preset"]),
        "bond": _ui_vals(w_k), "bond_mode": "weight", "bond_ids": None, "weights": None,
        "isl_bonds": [], "highlight": [], "seams": [], "units": None,
    })
    phases = ["ground sector polish", "C1 flux integrity", "C2 Shafranov shift", "C3 MHD stability",
              "C4 neoclassical transport", "C5 fast-ion confinement", "C6 bootstrap / PS"]
    state = {"k": 0}
    emit_step(0, len(phases), {"phase": phases[0]})

    def _prog(label):
        check()
        state["k"] += 1
        emit_step(state["k"], len(phases), {"phase": label})
        emit_log("  running %s" % label)

    ev = evaluate_w7x_seven_criteria(
        c["idx"], c["pos"], c["bonds"], c["loops"], c["meta"], c["bs_model"],
        coil_knobs=knobs, preset=p["preset"], uniform=(knobs is None and p.get("coils") == "baseline"),
        label="ui", progress=_prog)
    emit_step(len(phases), len(phases), {"phase": "done"})
    c3 = ev["criterion_3_mhd_beta_limit"]
    c4 = ev["criterion_4_neoclassical_transport"]
    c5 = ev["criterion_5_fast_ion_confinement"]
    c6 = ev["criterion_6_bootstrap_current"]
    return {
        "shafranov_shift_5pct": float(ev["criterion_2_shafranov_stiffness"]["shafranov_shift_5pct"]),
        "beta_crit_pct": float(c3["beta_crit_pct"]),
        "lambda_crit": float(c3["dimensionless_critical_load_x_crit"]),
        "min_gap_5pct": float(c3["min_vison_gap_5pct"]),
        "n_unstable_5pct": int(c3["n_unstable_bonds_5pct"]),
        "ripple_eps_B": float(c4["helical_ripple_eps_B"]),
        "leak_iso": float(c4["radial_leakage_rate"]),
        "leak_out": float(c4["outboard_leakage_rate"]),
        "fast_ion_survival": float(c5["inner_shell_survival"]),
        "trap_ratio": float(c5["outboard_trapping_ratio"]),
        "ps_dipole": float(c6["pfirsch_schlueter_seam_dipole"]),
        "geodesic_asym": float(c6["geodesic_field_asymmetry"]),
        "interior_eps_W": float(ev["criterion_1_flux_integrity"]["interior_eps_W"]),
        "outer_islands": int(ev["criterion_1_flux_integrity"]["outer_islands"]),
        "lowering_flips": int(ev["ground_sector_polish"]["n_lowering_single_flips"]),
        "mean_radial_weight": float(ev["mean_radial_weight"]),
        "mean_inshell_weight": float(ev["mean_inshell_weight"]),
    }


def _ui_point_manifold(machine, p, emit_step, check):
    manifest = machine.ctx()["manifest"]
    blk = manifest["blocks"][int(np.clip(int(p["block"]), 0, len(manifest["blocks"]) - 1))]
    dt = float(p["dt"])
    steps = int(p["steps"])
    blk_sites = [int(x) for x in blk["sites"]]
    if str(p["qubits"]).startswith("16"):
        b = _extract_exact_cycle_subgraph(blk, target_n=16)
        n_q = int(b["n_qubits"])
        sim = QrackSimulator(n_q, is_gpu=False)
        qc, G0, _ = prepare_shared_dimer_state(sim, b)
    else:
        b = blk
        n_q = 27
        sim = QrackSimulator(27, is_gpu=HAS_OPENCL_SO, is_stabilizer_hybrid=False)
        qc, G0, _ = prepare_shared_dimer_state(sim, b, max_dimers=4)
        configure_simulator_27q(sim, ace_max_qb=int(p["ace_max_qb"]), sdrp=float(p["sdrp"]))
    bl = [(int(ib["qa"]), int(ib["qb"]), int(ib["colour"])) for ib in b["internal_bonds"]]
    u = [float(ib["u"]) for ib in b["internal_bonds"]]
    w = [float(ib["weight"]) for ib in b["internal_bonds"]]
    prop = MajoranaFlowPropagator(n_q, bl, u, w, dt=dt, g0_init=G0)
    bond_ids = [ib.get("bond_id") for ib in b["internal_bonds"]]
    keep_k = [k for k, bid in enumerate(bond_ids) if bid is not None]
    used_sites = ([blk_sites[q] for q in b["sub_qs"]] if "sub_qs" in b else blk_sites)
    emit_step(None, steps, None, keep=True, frame={
        "label": "PyQrack block %d, %d qubits" % (int(blk["block_id"]), n_q),
        "highlight": used_sites, "bond_ids": [int(bond_ids[k]) for k in keep_k],
        "bond": None, "bond_mode": "error", "isl_bonds": [], "seams": [], "units": None, "weights": None,
    })
    kick = float(p["kick"])
    if kick != 0.0 and bl:
        qa, qb, col = bl[0]
        apply_bond_step(sim, qa, qb, col, -kick, non_clifford=True)
        prop.apply_bond_kick(qa, qb, kick * u[0])
    rms_hist = []
    max_hist = []
    for st in range(1, steps + 1):
        check()
        for k, (qa, qb, col) in enumerate(bl):
            apply_bond_step(sim, qa, qb, col, -dt * w[k], non_clifford=True)
            prop.apply_bond_kick(qa, qb, dt * w[k] * u[k])
        ref = prop.bond_expectations()
        vq = [float(read_bond_expectation(sim, qa, qb, col, q_counts=qc)) for (qa, qb, col) in bl]
        ve = [float(ref[k]) for k in range(len(bl))]
        err = np.abs(np.array(vq) - np.array(ve))
        rms = float(np.sqrt(np.mean(err ** 2)))
        rms_hist.append(rms)
        max_hist.append(float(np.max(err)))
        emit_step(st, steps, {
            "rms": rms, "max_err": float(np.max(err)),
            "mean_bond_qrack": float(np.mean(vq)), "mean_bond_exact": float(np.mean(ve)),
            "bond_qrack": vq, "bond_exact": ve,
        }, frame={
            "bond": [float("%.3e" % err[k]) for k in keep_k], "bond_mode": "error",
            "bond_ids": [int(bond_ids[k]) for k in keep_k],
        })
    W0 = None
    loops_l = b.get("internal_loops") or []
    if loops_l:
        try:
            W0 = float(read_pauli_word_inplace(sim, loops_l[0]["pauli_local"], q_counts=qc))
        except Exception:
            W0 = None
    del sim
    return {
        "n_qubits": n_q,
        "n_bonds": len(bl),
        "final_rms": rms_hist[-1],
        "max_rms": float(max(rms_hist)),
        "mean_rms": float(np.mean(rms_hist)),
        "final_max_err": max_hist[-1],
        "W0": W0,
    }


def _ui_point_link(machine, p, emit_step, check):
    check()
    c = machine.ctx()
    if p.get("_first"):
        emit_step(None, 0, None, keep=True, frame={
            "label": "%d inter-block seam bonds = teleported cuts per Trotter step" % len(c["seam_bonds"]),
            "seams": c["seam_bonds"], "bond": None, "bond_ids": None, "highlight": [],
            "isl_bonds": [], "units": None, "weights": None, "coils": None,
        })
    n_eb = max(1, int(round(float(p["n_ebits"]))))
    bud = dejmps_link_budget(float(p["F0"]), n_eb, target_delta=float(p["target_delta"]),
                             storage_share=float(p["storage_share"]))
    th = float(p["theta"])
    g_cut = abs(math.cos(2.0 * th)) + 3.0 * abs(math.sin(2.0 * th))
    shots = float(p["shots"])
    log10_lo = 2.0 * n_eb * math.log10(max(g_cut, 1.0)) + math.log10(shots)
    raw_total = shots * float(n_eb) * bud["raw_ebits_per_distilled"]
    t_mem_s = bud["required_T_mem_over_t_round"] * float(p["t_round_us"]) * 1e-6
    return {
        "n_ebits": n_eb,
        "eps_req": bud["eps_req"],
        "log10_eps_req": float(math.log10(bud["eps_req"])),
        "dejmps_rounds": bud["dejmps_rounds"],
        "reached_target": bud["reached_target"],
        "F_distilled": bud["F_distilled"],
        "raw_per_distilled": bud["raw_ebits_per_distilled"],
        "log10_raw_per_distilled": float(math.log10(max(bud["raw_ebits_per_distilled"], 1e-300))),
        "log10_raw_total": float(math.log10(max(raw_total, 1e-300))),
        "Tmem_over_tround": bud["required_T_mem_over_t_round"],
        "log10_Tmem_over_tround": float(math.log10(bud["required_T_mem_over_t_round"])),
        "Tmem_s": t_mem_s,
        "log10_Tmem_s": float(math.log10(max(t_mem_s, 1e-300))),
        "Gamma_LO_per_cut": g_cut,
        "log10_shots_LO_QPD": float(log10_lo),
        "ft_required": 1 if bud["eps_req"] < 1e-3 else 0,
    }


def _ui_sweep_values(sw):
    lo, hi = float(sw["min"]), float(sw["max"])
    n = max(1, min(int(sw.get("n", 5)), 4096))
    if n == 1:
        return [lo]
    if sw.get("scale") == "log" and lo > 0 and hi > 0:
        return [float(v) for v in np.logspace(math.log10(lo), math.log10(hi), n)]
    return [float(v) for v in np.linspace(lo, hi, n)]


class LiveJobRunner(object):
    """Runs one sweep at a time in a background thread and keeps an event log
    that Server-Sent-Event clients replay and follow."""

    def __init__(self, machine):
        self.machine = machine
        self.cond = threading.Condition()
        self.events = []
        self.job_id = 0
        self.thread = None
        self.cancel = threading.Event()
        self.running = False
        self.last_results = None
        self._last_frame_idx = None

    def emit(self, ev):
        with self.cond:
            ev["seq"] = len(self.events)
            ev["job_id"] = self.job_id
            ev["t"] = time.time()
            self.events.append(ev)
            self.cond.notify_all()

    def log(self, msg):
        self.emit({"type": "log", "msg": str(msg)})

    def emit_frame(self, i_pt, frame, keep=False):
        """3D-view frame (per-bond values, coil currents, highlights). Only
        the latest transient frame keeps its payload in the replay log, so a
        long run does not hold every per-bond array in memory; `keep` frames
        (static per-point overlays) are always kept."""
        with self.cond:
            if not keep and self._last_frame_idx is not None:
                old = self.events[self._last_frame_idx]
                old.pop("frame", None)
                old["stale"] = True
            ev = {"type": "frame", "i": i_pt, "keep": bool(keep), "frame": frame,
                  "seq": len(self.events), "job_id": self.job_id, "t": time.time()}
            self.events.append(ev)
            if not keep:
                self._last_frame_idx = len(self.events) - 1
            self.cond.notify_all()

    def status(self):
        with self.cond:
            return {"running": self.running, "job_id": self.job_id, "n_events": len(self.events)}

    def start(self, req):
        job = req.get("job")
        if job not in UI_JOBS:
            raise ValueError("unknown job %r" % (job,))
        with self.cond:
            if self.running:
                raise RuntimeError("a job is already running")
            self.running = True
            self.job_id += 1
            self.events = []
            self._last_frame_idx = None
            self.cancel.clear()
        self.thread = threading.Thread(target=self._run, args=(req,), daemon=True)
        self.thread.start()
        return self.job_id

    def stop(self):
        self.cancel.set()

    def _check(self):
        if self.cancel.is_set():
            raise _JobCancelled()

    def _run(self, req):
        job = req["job"]
        spec = UI_JOBS[job]
        params = {}
        for pd in spec["params"]:
            v = req.get("params", {}).get(pd["key"], pd["default"])
            if pd["type"] == "int":
                v = int(round(float(v)))
            elif pd["type"] == "float":
                v = float(v)
            params[pd["key"]] = v
        sweeps = [s for s in (req.get("sweep"), req.get("sweep2")) if s and s.get("key")]
        if not spec.get("two_d"):
            sweeps = sweeps[:1]
        axes = []
        for s_ax in sweeps:
            vals = _ui_sweep_values(s_ax)
            pd_ax = next((q for q in spec["params"] if q["key"] == s_ax["key"]), None)
            if pd_ax is None:
                raise ValueError("unknown sweep parameter %r" % (s_ax["key"],))
            if pd_ax["type"] == "int":
                seen = []
                for v in vals:
                    iv = float(int(round(v)))
                    if iv not in seen:
                        seen.append(iv)
                vals = seen
            axes.append((s_ax["key"], vals))
        scales = {s["key"]: ("log" if s.get("scale") == "log" else "lin") for s in sweeps}
        if any(k in UI_KNOB_KEYS for k, _ in axes) and "coils" in params:
            params["coils"] = "custom"
        grid = [()]
        for _, vals in axes:
            grid = [g + (v,) for g in grid for v in vals]
        t0 = time.time()
        self.emit({"type": "job_start", "job": job, "params": params,
                   "axes": [{"key": k, "values": v, "scale": scales[k]} for k, v in axes],
                   "n_points": len(grid)})
        results = []
        try:
            self.machine.ctx(log=self.log)
            for i_pt, coords in enumerate(grid):
                self._check()
                p = dict(params)
                for (k, _), v in zip(axes, coords):
                    pd = next(q for q in spec["params"] if q["key"] == k)
                    p[k] = int(round(v)) if pd["type"] == "int" else float(v)
                coord_d = {k: p[k] for k, _ in axes}
                p["_first"] = (i_pt == 0)
                p["_last"] = (i_pt == len(grid) - 1)
                self.emit({"type": "point_start", "i": i_pt, "coords": coord_d})
                t_pt = time.time()

                def emit_step(step, n_steps, data, frame=None, keep=False, _i=i_pt):
                    if step is not None:
                        self.emit({"type": "step", "i": _i, "step": int(step),
                                   "n_steps": int(n_steps), "data": data})
                    if frame is not None:
                        self.emit_frame(_i, frame, keep=keep)

                if job == "ness":
                    summ = _ui_point_ness(self.machine, p, emit_step, self._check)
                elif job == "coils":
                    summ = _ui_point_coils(self.machine, p, emit_step, self._check)
                elif job == "criteria":
                    summ = _ui_point_criteria(self.machine, p, emit_step, self._check, self.log)
                elif job == "manifold":
                    summ = _ui_point_manifold(self.machine, p, emit_step, self._check)
                else:
                    summ = _ui_point_link(self.machine, p, emit_step, self._check)
                rec = {"i": i_pt, "coords": coord_d, "summary": summ, "seconds": time.time() - t_pt}
                results.append(rec)
                self.emit(dict({"type": "point_done"}, **rec))
            self.emit({"type": "job_done", "seconds": time.time() - t0, "n_points": len(results)})
        except _JobCancelled:
            self.emit({"type": "job_cancelled", "seconds": time.time() - t0, "n_points": len(results)})
        except Exception as exc:
            import traceback
            self.emit({"type": "job_error", "error": "%s: %s" % (type(exc).__name__, exc),
                       "trace": traceback.format_exc()})
        finally:
            self.last_results = {"job": job, "params": params,
                                 "axes": [{"key": k, "values": v} for k, v in axes],
                                 "points": results}
            with self.cond:
                self.running = False
                self.cond.notify_all()


def _ui_schema(machine):
    """Job schema with coil-knob defaults set to the optimised knobs from
    scoreboard.json when available, so `custom` starts at the optimum."""
    knobs = None
    try:
        with open(machine.scoreboard_path, "r") as fh:
            knobs = json.load(fh)["optimisation"]["best_knobs"]
    except Exception:
        knobs = None
    out = json.loads(json.dumps(UI_JOBS))
    if knobs:
        for spec in out.values():
            for pd in spec["params"]:
                if pd["key"] in UI_KNOB_KEYS and pd["key"] in knobs:
                    pd["default"] = float(knobs[pd["key"]])
    return out


def _ui_json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


UI_C = {
    "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "s4": "#c98500", "s5": "#d55181", "s6": "#0ca30c",
    "grid": "#2a2a28", "axis": "#45443f", "t1": "#f0efec", "t2": "#c3c2b7", "t3": "#8f8e85",
    "surf": "#1a1a19", "surf2": "#232321", "page": "#121211", "border": "#2e2e2b", "mid": "#383835",
    "good": "#0ca30c", "warn": "#fab219", "crit": "#d03b3b",
}
UI_SEQ = ["#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec",
          "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"]
UI_DIV_NEG = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"]   # far -> near
UI_DIV_POS = ["#e66767", "#d03b3b", "#a52f2f", "#7a2424"]   # far -> near
UI_ORANGE = ["#2b1a12", "#4d2414", "#7a3217", "#a8421c", "#d95926", "#e87d4f", "#f2a684", "#fbd3c0"]
UI_CAT8 = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
UI_MODRAMP = ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6", "#86b6ef", "#5598e7",
              "#2a78d6", "#1c5cab"]
UI_LABELS = {
    "stored": "Stored energy", "core": "Core energy (s0+s1)", "exhaust": "Exhaust per step",
    "injected": "Injected per step", "cum_in": "Cumulative injected", "cum_ex": "Cumulative exhaust",
    "ledger_err": "Ledger error", "tau_E": "tau_E", "e_stored_ss": "Steady stored energy",
    "core_fraction": "Core fraction", "ex_in_ratio": "Exhaust / input", "S_vN": "S_vN (nats)",
    "tor_p2m": "10-unit peak/mean", "per_p2m": "5-period peak/mean", "pol_p2m": "Poloidal peak/mean",
    "confinement_product": "E_core x tau_E", "edge_rms_reduction_pct": "Edge field reduction %",
    "bond_par_reduction_pct": "Parallel field reduction %", "w_mag_rel": "Magnetic well (rel %)",
    "ripple_rel": "Helical ripple (rel %)", "geo_asym_rel": "Geodesic asymmetry (rel %)",
    "out_trap_ratio_rel": "Outboard trapping (rel %)", "rad_asym_rel": "Radial asymmetry (rel %)",
    "rad_outboard_rel": "Outboard radial (rel %)", "std_w": "Bond weight std",
    "score": "Mean improvement score (%)", "shafranov_shift_5pct": "Shafranov shift (beta 5%)",
    "beta_crit_pct": "beta_crit %", "lambda_crit": "lambda_crit", "min_gap_5pct": "Min flip gap (beta 5%)",
    "n_unstable_5pct": "Unstable bonds (beta 5%)", "ripple_eps_B": "Ripple eps_B",
    "leak_iso": "Isotropic leakage", "leak_out": "Outboard leakage", "fast_ion_survival": "Fast-ion survival",
    "trap_ratio": "Trapping ratio", "ps_dipole": "PS dipole", "geodesic_asym": "Geodesic asymmetry",
    "rms": "RMS bond error", "max_err": "Max bond error", "mean_bond_qrack": "Mean bond (PyQrack)",
    "mean_bond_exact": "Mean bond (exact)", "final_rms": "Final RMS", "max_rms": "Max RMS",
    "mean_rms": "Mean RMS", "final_max_err": "Final max error", "W0": "W_0 loop",
    "dejmps_rounds": "DEJMPS rounds", "log10_raw_per_distilled": "log10 raw / distilled",
    "log10_raw_total": "log10 raw ebits total", "log10_Tmem_over_tround": "log10 T_mem / t_round",
    "log10_Tmem_s": "log10 T_mem (s)", "log10_eps_req": "log10 eps_req",
    "log10_shots_LO_QPD": "log10 shots LO-QPD", "ft_required": "FT logical ebits needed",
}
UI_LOGY = {"rms", "max_err", "ledger_err", "final_rms", "max_rms", "mean_rms", "final_max_err"}
UI_PROFILE = {
    "coils": ("Metric profile along X", ["w_mag_rel", "ripple_rel", "geo_asym_rel", "out_trap_ratio_rel",
                                          "rad_asym_rel", "rad_outboard_rel"], "change vs uniform baseline (%)"),
    "link": ("Distilled-link cost (log10)", ["log10_raw_total", "log10_raw_per_distilled",
                                              "log10_Tmem_over_tround"], "log10 of count or ratio"),
}
UI_PHASES = ["ground sector polish", "C1 flux integrity", "C2 Shafranov shift", "C3 MHD stability",
             "C4 neoclassical transport", "C5 fast-ion confinement", "C6 bootstrap / PS"]


def _ui_lab(k):
    return UI_LABELS.get(k, k)


def _ui_fmt(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "true" if v else "false"
    if not isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, float) and not math.isfinite(v):
        return "-"
    a = abs(v)
    if a == 0:
        return "0"
    if isinstance(v, int) and a < 1e6:
        return str(v)
    if a >= 1e5 or a < 1e-3:
        return "%.2e" % v
    if a >= 100:
        return "%.1f" % v
    if a >= 1:
        return "%.3f" % v
    return "%.4f" % v


def _ui_dur(s):
    if not math.isfinite(s):
        return "-"
    if s < 1:
        return "< 1 s"
    if s < 90:
        return "%d s" % round(s)
    if s < 5400:
        return "%d min" % round(s / 60.0)
    return "%.1f h" % (s / 3600.0)


def _ui_hex_rgb(hx):
    n = int(hx[1:], 16)
    return ((n >> 16) & 255, (n >> 8) & 255, n & 255)


def _ui_lerp_hex(ramp, t):
    """Continuous interpolation along a hex ramp; returns an (r, g, b) tuple."""
    if not math.isfinite(t):
        t = 0.0
    t = min(1.0, max(0.0, t))
    if len(ramp) == 1:
        return _ui_hex_rgb(ramp[0])
    f = t * (len(ramp) - 1)
    i = min(len(ramp) - 2, int(math.floor(f)))
    u = f - i
    a, b = _ui_hex_rgb(ramp[i]), _ui_hex_rgb(ramp[i + 1])
    return tuple(int(round(a[k] + (b[k] - a[k]) * u)) for k in range(3))


def _ui_div_rgb(t):
    """Diverging blue - gray - red for t in [-1, 1]."""
    if t < 0:
        return _ui_lerp_hex([UI_C["mid"]] + UI_DIV_NEG[::-1], -t)
    return _ui_lerp_hex([UI_C["mid"]] + UI_DIV_POS[::-1], t)


def _ui_run_qt(a, machine, runner):
    """Build and run the PySide6 dashboard window (or a headless self-test)."""
    from PySide6 import QtCore, QtGui, QtWidgets
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.colors import LinearSegmentedColormap

    C = UI_C
    CM_SEQ = LinearSegmentedColormap.from_list("ui_seq", UI_SEQ)
    CM_TRACE = LinearSegmentedColormap.from_list("ui_trace", UI_SEQ[2:])
    CM_DIV = LinearSegmentedColormap.from_list("ui_div", UI_DIV_NEG + [C["mid"]] + UI_DIV_POS[::-1])

    def _safe_alpha(qc_, x):
        x = float(x)
        qc_.setAlphaF(min(1.0, max(0.0, x)) if math.isfinite(x) else 1.0)

    def qcol(rgb_or_hex, alpha=255):
        if isinstance(rgb_or_hex, str):
            c = QtGui.QColor(rgb_or_hex)
        else:
            c = QtGui.QColor(*rgb_or_hex)
        c.setAlpha(int(alpha))
        return c

    def style_ax(ax, xlabel=None, ylabel=None):
        ax.set_facecolor(C["surf"])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(C["axis"])
        ax.tick_params(colors=C["t3"], labelsize=8, length=3)
        ax.grid(True, color=C["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        if xlabel:
            ax.set_xlabel(xlabel, color=C["t2"], fontsize=9)
        if ylabel:
            ax.set_ylabel(ylabel, color=C["t2"], fontsize=9)

    def empty_msg(ax, msg):
        ax.set_facecolor(C["surf"])
        ax.set_axis_off()
        ax.text(0.5, 0.5, msg, color=C["t3"], ha="center", va="center", fontsize=10, transform=ax.transAxes)

    # ------------------------------------------------------------------
    # Parameter form for one job
    # ------------------------------------------------------------------
    class ParamForm(QtWidgets.QWidget):
        changed = QtCore.Signal()

        def __init__(self, job, spec):
            super().__init__()
            self.job, self.spec = job, spec
            self.editors, self.sweepX, self.sweepY, self.ranges = {}, {}, {}, {}
            lay = QtWidgets.QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(2)
            for pd in spec["params"]:
                box = QtWidgets.QFrame()
                box.setObjectName("prm")
                bl = QtWidgets.QVBoxLayout(box)
                bl.setContentsMargins(0, 5, 0, 5)
                bl.setSpacing(3)
                row = QtWidgets.QHBoxLayout()
                name = pd["label"] + ("  (custom coils)" if pd.get("group") == "coils" and any(
                    q["key"] == "coils" for q in spec["params"]) else "")
                lab = QtWidgets.QLabel(name)
                lab.setObjectName("prmName")
                row.addWidget(lab, 1)
                if pd["type"] == "select":
                    ed = QtWidgets.QComboBox()
                    ed.addItems(pd["options"])
                    ed.setCurrentText(str(pd["default"]))
                    ed.currentTextChanged.connect(self.changed.emit)
                else:
                    ed = self._spin(pd, pd["default"])
                    ed.valueChanged.connect(self.changed.emit)
                ed.setFixedWidth(104)
                row.addWidget(ed)
                self.editors[pd["key"]] = ed
                if pd["type"] != "select":
                    cx = QtWidgets.QCheckBox("sweep")
                    cx.setToolTip("Sweep this parameter (X axis)")
                    cx.toggled.connect(lambda on, k=pd["key"]: self._toggle(k, "x", on))
                    row.addWidget(cx)
                    self.sweepX[pd["key"]] = cx
                    if spec.get("two_d"):
                        cy = QtWidgets.QCheckBox("Y")
                        cy.setToolTip("Second sweep axis (heatmap)")
                        cy.toggled.connect(lambda on, k=pd["key"]: self._toggle(k, "y", on))
                        row.addWidget(cy)
                        self.sweepY[pd["key"]] = cy
                bl.addLayout(row)
                if pd["type"] != "select":
                    rw = QtWidgets.QWidget()
                    rl = QtWidgets.QHBoxLayout(rw)
                    rl.setContentsMargins(12, 0, 0, 0)
                    lo, hi = self._default_range(pd)
                    s_lo, s_hi = self._spin(pd, lo), self._spin(pd, hi)
                    s_n = QtWidgets.QSpinBox()
                    s_n.setRange(1, 4096)
                    s_n.setValue(self._default_n(two_d=False))
                    s_log = QtWidgets.QCheckBox("log")
                    s_log.setChecked(bool(pd.get("log")))
                    for wdg in (s_lo, s_hi):
                        wdg.setFixedWidth(92)
                    s_n.setFixedWidth(60)
                    for t_, wdg in (("from", s_lo), ("to", s_hi), ("n", s_n)):
                        lb = QtWidgets.QLabel(t_)
                        lb.setObjectName("muted")
                        rl.addWidget(lb)
                        rl.addWidget(wdg)
                    rl.addWidget(s_log)
                    rl.addStretch(1)
                    s_n.valueChanged.connect(self.changed.emit)
                    rw.setVisible(False)
                    bl.addWidget(rw)
                    self.ranges[pd["key"]] = (rw, s_lo, s_hi, s_n, s_log)
                lay.addWidget(box)
            lay.addStretch(1)

        def _spin(self, pd, val):
            if pd["type"] == "int":
                sp = QtWidgets.QSpinBox()
                sp.setRange(int(pd["min"]), int(pd["max"]))
                sp.setValue(int(round(float(val))))
            else:
                sp = QtWidgets.QDoubleSpinBox()
                dec = 6 if float(pd["min"]) < 1e-3 and float(pd["min"]) > 0 else 4
                sp.setDecimals(dec)
                sp.setRange(float(pd["min"]), float(pd["max"]))
                span = float(pd["max"]) - float(pd["min"])
                sp.setSingleStep(10 ** math.floor(math.log10(max(span, 1e-9)) - 2))
                sp.setValue(float(val))
            sp.setKeyboardTracking(False)
            return sp

        def _default_range(self, pd):
            lo, hi = float(pd["min"]), float(pd["max"])
            if not pd.get("log"):
                span = hi - lo
                lo, hi = lo + 0.15 * span, hi - 0.15 * span
            if pd["type"] == "int":
                lo, hi = round(lo), round(hi)
            return float("%.4g" % lo), float("%.4g" % hi)

        def _default_n(self, two_d):
            if two_d:
                return 21
            return {"coils": 41, "link": 60, "criteria": 3}.get(self.job, 5)

        def _toggle(self, key, axis, on):
            boxes = self.sweepX if axis == "x" else self.sweepY
            other = self.sweepY if axis == "x" else self.sweepX
            if on:
                for k, cb in boxes.items():
                    if k != key and cb.isChecked():
                        cb.blockSignals(True)
                        cb.setChecked(False)
                        cb.blockSignals(False)
                if key in other and other[key].isChecked():
                    other[key].blockSignals(True)
                    other[key].setChecked(False)
                    other[key].blockSignals(False)
                if axis == "y":
                    self.ranges[key][3].setValue(self._default_n(True))
            for k, (rw, _, _, _, _) in self.ranges.items():
                rw.setVisible((k in self.sweepX and self.sweepX[k].isChecked())
                              or (k in self.sweepY and self.sweepY[k].isChecked()))
            self.changed.emit()

        def _sweep(self, boxes):
            for k, cb in boxes.items():
                if cb.isChecked():
                    _, lo, hi, n, lg = self.ranges[k]
                    return {"key": k, "min": float(lo.value()), "max": float(hi.value()),
                            "n": int(n.value()), "scale": "log" if lg.isChecked() else "lin"}
            return None

        def request(self):
            params = {}
            for pd in self.spec["params"]:
                ed = self.editors[pd["key"]]
                params[pd["key"]] = ed.currentText() if pd["type"] == "select" else ed.value()
            return {"job": self.job, "params": params,
                    "sweep": self._sweep(self.sweepX), "sweep2": self._sweep(self.sweepY)}

        def n_points(self):
            r = self.request()
            n = 1
            for s_ in (r["sweep"], r["sweep2"]):
                if s_:
                    n *= max(1, s_["n"])
            return n, r

    # ------------------------------------------------------------------
    # Matplotlib chart panel with a metric selector
    # ------------------------------------------------------------------
    class ChartPanel(QtWidgets.QFrame):
        def __init__(self, title, nrows=1, ncols=1, height=280):
            super().__init__()
            self.setObjectName("card")
            lay = QtWidgets.QVBoxLayout(self)
            lay.setContentsMargins(10, 8, 10, 8)
            hd = QtWidgets.QHBoxLayout()
            self.title = QtWidgets.QLabel(title)
            self.title.setObjectName("cardTitle")
            hd.addWidget(self.title, 1)
            self.hint = QtWidgets.QLabel("")
            self.hint.setObjectName("muted")
            hd.addWidget(self.hint)
            self.metric_box = QtWidgets.QComboBox()
            hd.addWidget(self.metric_box)
            lay.addLayout(hd)
            self.fig = Figure(figsize=(5, 3), dpi=100, facecolor=C["surf"])
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.canvas.setMinimumHeight(height)
            lay.addWidget(self.canvas, 1)
            self.legend = QtWidgets.QLabel("")
            self.legend.setObjectName("legend")
            self.legend.setWordWrap(True)
            lay.addWidget(self.legend)

        def set_metrics(self, keys, default):
            cur = self.metric_box.currentData()
            self.metric_box.blockSignals(True)
            self.metric_box.clear()
            for k in keys:
                self.metric_box.addItem(_ui_lab(k), k)
            want = cur if cur in keys else default
            if want in keys:
                self.metric_box.setCurrentIndex(keys.index(want))
            self.metric_box.setVisible(bool(keys))
            self.metric_box.blockSignals(False)

        def key(self):
            return self.metric_box.currentData()

    # ------------------------------------------------------------------
    # QPainter 3D machine view
    # ------------------------------------------------------------------
    class View3D(QtWidgets.QWidget):
        def __init__(self, geo, controls):
            super().__init__()
            self.g = geo
            self.ctl = controls
            self.setMinimumHeight(460)
            self.setMouseTracking(True)
            self.frame = {}
            self.frame_step = None
            self.status_text = ""
            self.reset_view(update=False)
            self._drag = None
            self._proj = None
            self.spin_timer = QtCore.QTimer(self)
            self.spin_timer.timeout.connect(self._spin)

        def reset_view(self, update=True):
            self.yaw, self.pitch, self.dist = -0.55, 0.6, 14.5
            self.pan = [0.0, 0.0]
            self.target = np.zeros(3)
            if update:
                self.update()

        def focus(self):
            g, f3 = self.g, self.frame
            ids = []
            if f3.get("highlight"):
                ids = list(f3["highlight"])
            elif f3.get("isl_bonds"):
                b = g["bonds"][f3["isl_bonds"][0]]
                ids = [int(b[0]), int(b[1])]
            if ids:
                self.target = g["P"][ids].mean(axis=0)
            elif f3.get("units"):
                self.target = g["units"][int(np.argmax(f3["units"]))]
            else:
                return
            self.ctl["span"].setValue(360)
            self.yaw = -math.atan2(self.target[1], self.target[0]) - math.pi / 2
            self.pitch, self.dist, self.pan = 0.45, 5.5, [0.0, 0.0]
            self.update()

        def _spin(self):
            self.yaw -= 0.006
            self.update()

        def set_spin(self, on):
            if on:
                self.spin_timer.start(33)
            else:
                self.spin_timer.stop()

        def reset_frame(self):
            self.frame, self.frame_step = {}, None
            self.update()

        def merge_frame(self, fr, step):
            for k, v in fr.items():
                self.frame[k] = v
            if "bond" in fr:
                self.frame_step = step
            self.update()

        # -- interaction --
        def mousePressEvent(self, e):
            p = e.position()
            self._drag = (p.x(), p.y(), bool(e.modifiers() & QtCore.Qt.ShiftModifier)
                          or e.button() == QtCore.Qt.RightButton)

        def mouseReleaseEvent(self, e):
            self._drag = None

        def mouseDoubleClickEvent(self, e):
            self.reset_view()

        def wheelEvent(self, e):
            self.dist = float(np.clip(self.dist * math.pow(1.0012, -e.angleDelta().y()), 2.5, 60.0))
            self.update()

        def mouseMoveEvent(self, e):
            p = e.position()
            if self._drag:
                x0, y0, pan = self._drag
                dx, dy = p.x() - x0, p.y() - y0
                self._drag = (p.x(), p.y(), pan)
                if pan:
                    self.pan[0] += dx
                    self.pan[1] += dy
                else:
                    self.yaw -= dx * 0.008
                    self.pitch = float(np.clip(self.pitch + dy * 0.006, 0.02, 1.55))
                self.update()
                return
            if self._proj is None:
                return
            sx, sy, vis = self._proj
            d2 = (sx - p.x()) ** 2 + (sy - p.y()) ** 2
            d2[~vis] = 1e18
            i = int(np.argmin(d2))
            if d2[i] > 64:
                QtWidgets.QToolTip.hideText()
                return
            g = self.g
            txt = "site %d  shell s%d  module %d  block %d" % (i, g["shell"][i], g["module"][i], g["block"][i])
            if i in g["core"]:
                txt += "\ncore axis site"
            if i in g["div"]:
                txt += "\ndivertor target site"
            sv = self.site_value(i)
            if sv is not None:
                txt += "\n%s: %s" % (sv[1], _ui_fmt(sv[0]))
            QtWidgets.QToolTip.showText(e.globalPosition().toPoint(), txt, self)

        # -- data helpers --
        def live_values(self):
            f, mode = self.frame, self.ctl["colour"].currentData()
            if mode == "weight":
                w = f.get("weights") or (f.get("bond") if f.get("bond_mode") == "weight" else None)
                return (w, None, "weight", "Coupling J") if w else None
            if mode != "live" or not f.get("bond"):
                return None
            lab = {"energy": "Excess bond energy", "weight": "Coupling J",
                   "error": "PyQrack bond error"}.get(f.get("bond_mode"), "value")
            return (f["bond"], f.get("bond_ids"), f.get("bond_mode"), lab)

        def bond_array(self, lv):
            vals = np.full(len(self.g["bonds"]), np.nan)
            v, ids = np.asarray(lv[0], dtype=float), lv[1]
            if ids:
                vals[np.asarray(ids, dtype=int)] = v
            else:
                vals[:len(v)] = v
            return vals

        def scale(self, lv, vals):
            v = vals[np.isfinite(vals)]
            if not len(v):
                return None
            if lv[2] == "weight":
                m = max(1e-6, float(np.max(np.abs(v - 1.0))))
                return ("div", 1.0, m)
            if lv[2] == "error":
                lo = math.log10(max(1e-9, float(np.quantile(v, 0.02))))
                hi = math.log10(max(1e-9, float(np.max(v))))
                return ("log", lo, max(hi, lo + 0.5))
            lo = min(0.0, float(np.quantile(v, 0.02)))
            hi = max(float(np.quantile(v, 0.995)), lo + 1e-6)
            return ("seq", lo, hi)

        def colour_t(self, sc, x):
            if sc[0] == "div":
                return _ui_div_rgb((x - sc[1]) / sc[2])
            if sc[0] == "log":
                return _ui_lerp_hex(UI_SEQ, (math.log10(max(x, 1e-12)) - sc[1]) / (sc[2] - sc[1]))
            return _ui_lerp_hex(UI_ORANGE, (x - sc[1]) / (sc[2] - sc[1]))

        def site_value(self, i):
            lv = self.live_values()
            if not lv:
                return None
            vals = self.bond_array(lv)
            inc = self.g["incident"][i]
            v = vals[inc]
            v = v[np.isfinite(v)]
            if not len(v):
                return None
            return float(np.mean(v)), "%s (mean of %d bonds)" % (lv[3], len(v))

        def legend_html(self):
            return getattr(self, "_legend", "")

        # -- drawing --
        def paintEvent(self, ev):
            g = self.g
            w, h = self.width(), self.height()
            qp = QtGui.QPainter(self)
            qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
            grad = QtGui.QRadialGradient(w / 2, h / 2, max(w, h) * 0.7)
            grad.setColorAt(0.0, QtGui.QColor("#1d1d1b"))
            grad.setColorAt(1.0, QtGui.QColor("#161615"))
            qp.fillRect(self.rect(), grad)
            cyw, syw = math.cos(self.yaw), math.sin(self.yaw)
            cp, sp = math.cos(self.pitch), math.sin(self.pitch)
            f = min(0.46 * w * 14.5 / 6.3, 0.44 * h * 14.5 / (6.3 * math.sin(0.6) + 1.0))
            ox, oy = w / 2 + self.pan[0], h / 2 + self.pan[1]
            T = self.target

            def project(P):
                X = P[:, 0] - T[0]
                Y = P[:, 1] - T[1]
                Z = P[:, 2] - T[2]
                x1 = X * cyw - Y * syw
                y1 = X * syw + Y * cyw
                y2 = y1 * cp - Z * sp
                z2 = y1 * sp + Z * cp
                d = np.maximum(0.3, self.dist + y2)
                return ox + f * x1 / d, oy - f * z2 / d, d

            phi0 = math.radians(self.ctl["phi0"].value())
            span = math.radians(self.ctl["span"].value())

            def in_win(a_):
                if span >= 2 * math.pi - 1e-6:
                    return np.ones_like(a_, dtype=bool)
                return np.mod(a_ - phi0 + 4 * math.pi, 2 * math.pi) <= span

            sx, sy, dep = project(g["P"])
            shells_on = np.array([cb.isChecked() for cb in self.ctl["shells"]])
            vis = shells_on[g["shell"]] & in_win(g["phi"])
            self._proj = (sx, sy, vis)
            dmin, dmax = float(dep.min()), float(dep.max())

            def depth_a(d):
                return float(min(1.0, max(0.15, 1.0 - 0.6 * (d - dmin) / max(1e-3, dmax - dmin))))

            mode = self.ctl["colour"].currentData()
            f3 = self.frame
            legend = []

            # 1. coils coloured by current deviation from the preset nominal
            if self.ctl["coils"].isChecked():
                cf = f3.get("coils")
                devs = []
                for c in g["coils"]:
                    if not cf:
                        devs.append(None)
                    elif c["fam"] == "non_planar":
                        devs.append(cf["np"][c["type"]] - cf["np0"][c["type"]])
                    elif c["fam"] == "planar":
                        devs.append(cf["pc"][c["type"]] - cf["pc0"][c["type"]])
                    elif c["fam"] == "trim":
                        devs.append(cf["trim"][c["k"]])
                    else:
                        devs.append(cf["ctrl"][c["k"]])
                fam_max = {}
                for c, dv in zip(g["coils"], devs):
                    if dv is not None:
                        fam_max[c["fam"]] = max(fam_max.get(c["fam"], 0.0), abs(dv))
                widths = {"non_planar": 1.7, "planar": 1.2, "trim": 2.6, "control": 1.2}
                win_c = in_win(g["coil_phi"])
                for k, (c, dv) in enumerate(zip(g["coils"], devs)):
                    if not win_c[k]:
                        continue
                    cx, cy_, cd = project(c["xyz"])
                    if dv is None:
                        col, al = qcol("#4a4945"), 0.35
                    else:
                        floor_ = 0.05 if c["fam"] in ("non_planar", "planar") else 1e-3
                        col, al = qcol(_ui_div_rgb(dv / max(fam_max.get(c["fam"], 0.0), floor_))), 0.55
                    _safe_alpha(col, al * depth_a(float(cd.mean())))
                    pen = QtGui.QPen(col, widths[c["fam"]])
                    qp.setPen(pen)
                    qp.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(float(a_), float(b_)) for a_, b_ in zip(cx, cy_)]))
                legend.append("Coils: current vs preset nominal (blue below, red above)" if cf
                              else "Coils: 50 non-planar, 20 planar, 5 trim, 10 control")

            # 2. bonds
            lv = self.live_values() if mode in ("live", "weight") else None
            sc, vals = None, None
            if lv:
                vals = self.bond_array(lv)
                sc = self.scale(lv, vals)
            bi, bj, bcol = g["bi"], g["bj"], g["bcolour"]
            bvis = vis[bi] & vis[bj]
            if self.ctl["bonds"].isChecked():
                bdep = 0.5 * (dep[bi] + dep[bj])
                layer = np.clip(np.nan_to_num(3 * (bdep - dmin) / max(1e-3, dmax - dmin)), 0, 2).astype(int)
                keys = {}
                for k in np.nonzero(bvis)[0]:
                    if mode == "shell":
                        col = UI_CAT8[max(g["shell"][bi[k]], g["shell"][bj[k]])]
                    elif mode == "module":
                        col = UI_MODRAMP[g["module"][bi[k]] % 10]
                    elif mode == "block":
                        col = "#f0efec" if k in g["seam_set"] else UI_CAT8[max(0, g["block"][bi[k]]) % 4]
                    elif mode == "colour":
                        col = UI_CAT8[bcol[k] % 3]
                    elif sc is not None and math.isfinite(vals[k]):
                        rgb = self.colour_t(sc, float(vals[k]))
                        col = "#%02x%02x%02x" % tuple(int(round(c_ / 8.0) * 8) if c_ < 248 else 255 for c_ in rgb)
                    else:
                        col = None
                    hot = col is not None
                    key = (col or "#3a3a37", int(layer[k]), hot)
                    keys.setdefault(key, []).append(k)
                for (col, lay_, hot) in sorted(keys.keys(), key=lambda t: -t[1]):
                    qc_ = qcol(col)
                    _safe_alpha(qc_, (1.0 if hot else 0.55) * (1 - 0.22 * lay_))
                    qp.setPen(QtGui.QPen(qc_, 1.7 if hot else 1.0))
                    ks = keys[(col, lay_, hot)]
                    qp.drawLines([QtCore.QLineF(float(sx[bi[k]]), float(sy[bi[k]]), float(sx[bj[k]]), float(sy[bj[k]]))
                                  for k in ks])

            # 3. overlays
            if self.ctl["over"].isChecked():
                def draw_set(ids, col, lw, alpha=1.0):
                    qc_ = qcol(col)
                    _safe_alpha(qc_, alpha)
                    qp.setPen(QtGui.QPen(qc_, lw))
                    qp.drawLines([QtCore.QLineF(float(sx[bi[k]]), float(sy[bi[k]]), float(sx[bj[k]]), float(sy[bj[k]]))
                                  for k in ids if bvis[k]])
                if f3.get("seams"):
                    draw_set(f3["seams"], "#f0efec", 1.2, 0.75)
                    legend.append("%d seam bonds (1 ebit per cut per step)" % len(f3["seams"]))
                if f3.get("isl_bonds"):
                    qp.setPen(QtGui.QPen(qcol(C["s5"]), 2.0))
                    qp.setBrush(QtCore.Qt.NoBrush)
                    for k in f3["isl_bonds"]:
                        if bvis[k]:
                            qp.drawEllipse(QtCore.QPointF((sx[bi[k]] + sx[bj[k]]) / 2, (sy[bi[k]] + sy[bj[k]]) / 2), 7, 7)
                    draw_set(f3["isl_bonds"], C["s5"], 3.0)
                    legend.append("%d edge-island flux defects (flipped u)" % len(f3["isl_bonds"]))
                if f3.get("highlight"):
                    qp.setPen(QtGui.QPen(qcol(C["s2"]), 1.5))
                    qp.setBrush(QtCore.Qt.NoBrush)
                    for i in f3["highlight"]:
                        if vis[i]:
                            qp.drawEllipse(QtCore.QPointF(float(sx[i]), float(sy[i])), 5, 5)
                    legend.append("%d PyQrack qubits" % len(f3["highlight"]))

            # 4. sites (far to near, grouped by colour and size)
            if self.ctl["sites"].isChecked():
                order = np.nonzero(vis)[0]
                order = order[np.argsort(-dep[order])]
                qp.setPen(QtCore.Qt.NoPen)
                for i in order:
                    col, r, a0 = "#8f8e85", 1.1, (0.55 if mode in ("live", "weight") else 0.9)
                    if mode == "shell":
                        col = UI_CAT8[g["shell"][i]]
                    elif mode == "block":
                        col = UI_CAT8[max(0, g["block"][i]) % 4]
                    elif mode == "module":
                        col = UI_MODRAMP[g["module"][i] % 10]
                    if i in g["core"]:
                        col, r, a0 = C["s3"], 2.0, 0.95
                    elif i in g["div"]:
                        col, r, a0 = C["s6"], 1.8, 0.95
                    qc_ = qcol(col)
                    _safe_alpha(qc_, a0 * depth_a(float(dep[i])))
                    qp.setBrush(qc_)
                    rr = max(0.8, r * 14.5 / float(dep[i]))
                    qp.drawEllipse(QtCore.QPointF(float(sx[i]), float(sy[i])), rr, rr)

            # 5. divertor units sized by exhaust share
            if self.ctl["div"].isChecked():
                u = f3.get("units")
                ux, uy, _ = project(g["units"])
                uphi = np.mod(np.arctan2(g["units"][:, 1], g["units"][:, 0]), 2 * math.pi)
                win_u = in_win(uphi)
                fnt = qp.font()
                fnt.setPointSizeF(7.5)
                qp.setFont(fnt)
                for k in range(len(g["units"])):
                    if not win_u[k]:
                        continue
                    v = u[k] if u else None
                    r = 5.0 if v is None else 3 + 7 * max(0.0, v)
                    fill = qcol(C["s3"])
                    _safe_alpha(fill, 0.18 if v is None else min(0.75, 0.15 + 0.35 * v))
                    qp.setBrush(fill)
                    qp.setPen(QtGui.QPen(qcol(C["s3"]), 1.2))
                    qp.drawEllipse(QtCore.QPointF(float(ux[k]), float(uy[k])), r, r)
                    qp.setPen(qcol(C["t2"]))
                    txt = "D%d" % (k + 1) + ("" if v is None else " %.2f" % v)
                    qp.drawText(QtCore.QPointF(float(ux[k]) - 14, float(uy[k]) - r - 3), txt)
                legend.append("divertor units (size = exhaust / mean)")

            # HUD
            qp.setPen(qcol(C["t3"]))
            fnt = qp.font()
            fnt.setPointSizeF(8.5)
            qp.setFont(fnt)
            hud = f3.get("label") or ""
            if self.status_text:
                hud += ("\n" if hud else "") + self.status_text
            qp.drawText(QtCore.QRectF(10, 6, w - 20, 40), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, hud)
            qp.end()

            # legend text
            parts = []
            if mode == "shell":
                parts.append("Bonds and sites by flux shell s0 to s4")
            elif mode == "block":
                parts.append("Bonds by 27-qubit block type 0 to 3, seams white")
            elif mode == "module":
                parts.append("Bonds by half-period module")
            elif mode == "colour":
                parts.append("Kitaev bond colour x (blue), y (orange), z (aqua)")
            elif sc is not None:
                lo_ = 10 ** sc[1] if sc[0] == "log" else (sc[1] - sc[2] if sc[0] == "div" else sc[1])
                hi_ = 10 ** sc[2] if sc[0] == "log" else (sc[1] + sc[2] if sc[0] == "div" else sc[2])
                ramp = {"div": "blue-gray-red", "log": "dark to light blue (log)", "seq": "dark to light orange"}[sc[0]]
                parts.append("%s %s to %s (%s)" % (lv[3], _ui_fmt(lo_), _ui_fmt(hi_), ramp))
            elif mode == "live":
                parts.append("Run a job to colour the machine with live data")
            else:
                parts.append("Run a NESS, coil or 7-criteria job to load coupling weights")
            self._legend = "   |   ".join(parts + legend)
            lg = self.ctl.get("legend")
            if lg is not None and lg.text() != self._legend:
                lg.setText(self._legend)

    # ------------------------------------------------------------------
    # Main window
    # ------------------------------------------------------------------
    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("W7-X Kitaev Stellarator - Live (stellarator_ui.py)")
            self.schema = _ui_schema(machine)
            self.S = None
            self.ev_pos, self.ev_job = 0, -1
            self.dirty_charts = False
            self.last_chart_draw = 0.0
            self.last_status = None

            root = QtWidgets.QWidget()
            self.setCentralWidget(root)
            rl = QtWidgets.QVBoxLayout(root)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(0)
            rl.addWidget(self._header())
            split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            rl.addWidget(split, 1)
            split.addWidget(self._sidebar())
            split.addWidget(self._main_area())
            split.setStretchFactor(1, 1)
            split.setSizes([440, 1160])

            self.poll_timer = QtCore.QTimer(self)
            self.poll_timer.timeout.connect(self.poll)
            self.poll_timer.start(100)
            self.on_job_changed()
            self.render_all()

        # -- layout pieces --
        def _header(self):
            hd = QtWidgets.QFrame()
            hd.setObjectName("header")
            l = QtWidgets.QHBoxLayout(hd)
            l.setContentsMargins(14, 8, 14, 8)
            t = QtWidgets.QVBoxLayout()
            a_ = QtWidgets.QLabel("W7-X Kitaev Stellarator - Live")
            a_.setObjectName("h1")
            b_ = QtWidgets.QLabel("stellarator_ui.py - 1080 sites - 40 x 27-qubit manifolds")
            b_.setObjectName("muted")
            t.addWidget(a_)
            t.addWidget(b_)
            l.addLayout(t)
            self.pill = QtWidgets.QLabel("idle")
            self.pill.setObjectName("pill")
            l.addWidget(self.pill)
            pv = QtWidgets.QVBoxLayout()
            self.bar_pts = QtWidgets.QProgressBar()
            self.bar_step = QtWidgets.QProgressBar()
            self.bar_step.setObjectName("stepbar")
            for b in (self.bar_pts, self.bar_step):
                b.setRange(0, 1000)
                b.setTextVisible(False)
                b.setFixedHeight(6)
                pv.addWidget(b)
            self.prog_text = QtWidgets.QLabel("No job yet")
            self.prog_text.setObjectName("muted")
            pv.addWidget(self.prog_text)
            l.addLayout(pv, 1)
            return hd

        def _sidebar(self):
            outer = QtWidgets.QFrame()
            outer.setObjectName("sidebar")
            ol = QtWidgets.QVBoxLayout(outer)
            ol.setContentsMargins(12, 10, 12, 10)
            self.job_box = QtWidgets.QComboBox()
            for j, spec in self.schema.items():
                self.job_box.addItem(spec["label"], j)
            ol.addWidget(self.job_box)
            self.about = QtWidgets.QLabel("")
            self.about.setWordWrap(True)
            self.about.setObjectName("about")
            ol.addWidget(self.about)
            self.forms = QtWidgets.QStackedWidget()
            self.form_by_job = {}
            for j, spec in self.schema.items():
                fm = ParamForm(j, spec)
                fm.changed.connect(self.update_estimate)
                self.form_by_job[j] = fm
                self.forms.addWidget(fm)
            sc = QtWidgets.QScrollArea()
            sc.setWidgetResizable(True)
            sc.setWidget(self.forms)
            sc.setFrameShape(QtWidgets.QFrame.NoFrame)
            sc.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            ol.addWidget(sc, 1)
            self.estimate = QtWidgets.QLabel("")
            self.estimate.setObjectName("muted")
            ol.addWidget(self.estimate)
            r1 = QtWidgets.QHBoxLayout()
            self.run_btn = QtWidgets.QPushButton("Run")
            self.run_btn.setObjectName("run")
            self.stop_btn = QtWidgets.QPushButton("Stop")
            r1.addWidget(self.run_btn)
            r1.addWidget(self.stop_btn)
            ol.addLayout(r1)
            r2 = QtWidgets.QHBoxLayout()
            self.csv_btn = QtWidgets.QPushButton("Export CSV")
            self.json_btn = QtWidgets.QPushButton("Export JSON")
            r2.addWidget(self.csv_btn)
            r2.addWidget(self.json_btn)
            ol.addLayout(r2)
            self.job_box.currentIndexChanged.connect(self.on_job_changed)
            self.run_btn.clicked.connect(self.run_job)
            self.stop_btn.clicked.connect(runner.stop)
            self.csv_btn.clicked.connect(self.export_csv)
            self.json_btn.clicked.connect(self.export_json)
            return outer

        def _main_area(self):
            sc = QtWidgets.QScrollArea()
            sc.setWidgetResizable(True)
            sc.setFrameShape(QtWidgets.QFrame.NoFrame)
            body = QtWidgets.QWidget()
            body.setObjectName("mainbody")
            sc.setWidget(body)
            g = QtWidgets.QGridLayout(body)
            g.setContentsMargins(12, 12, 12, 12)
            g.setSpacing(12)
            g.addWidget(self._view3d_card(), 0, 0, 1, 2)
            self.trace = ChartPanel("Live step trace")
            self.sweep = ChartPanel("Sweep result")
            self.detail = ChartPanel("Detail", height=240)
            self.detail.metric_box.setVisible(False)
            self.kv = QtWidgets.QTableWidget(0, 2)
            self.kv.horizontalHeader().setVisible(False)
            self.kv.verticalHeader().setVisible(False)
            self.kv.horizontalHeader().setStretchLastSection(True)
            self.detail_stack = QtWidgets.QStackedWidget()
            dl = self.detail.layout()
            pos_ = dl.indexOf(self.detail.canvas)
            dl.removeWidget(self.detail.canvas)
            self.detail_stack.addWidget(self.detail.canvas)
            self.detail_stack.addWidget(self.kv)
            dl.insertWidget(pos_, self.detail_stack, 1)
            self.detail_stack.setMinimumHeight(240)
            self.trace.metric_box.currentIndexChanged.connect(self.mark_dirty)
            self.sweep.metric_box.currentIndexChanged.connect(self.mark_dirty)
            g.addWidget(self.trace, 1, 0)
            g.addWidget(self.sweep, 1, 1)
            g.addWidget(self.detail, 2, 0)
            tcard = QtWidgets.QFrame()
            tcard.setObjectName("card")
            tl = QtWidgets.QVBoxLayout(tcard)
            th = QtWidgets.QHBoxLayout()
            tt = QtWidgets.QLabel("Results table")
            tt.setObjectName("cardTitle")
            th.addWidget(tt, 1)
            self.table_hint = QtWidgets.QLabel("")
            self.table_hint.setObjectName("muted")
            th.addWidget(self.table_hint)
            tl.addLayout(th)
            self.table = QtWidgets.QTableWidget(0, 0)
            self.table.verticalHeader().setVisible(False)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            tl.addWidget(self.table, 1)
            g.addWidget(tcard, 2, 1)
            lcard = QtWidgets.QFrame()
            lcard.setObjectName("card")
            ll = QtWidgets.QVBoxLayout(lcard)
            lt = QtWidgets.QLabel("Log")
            lt.setObjectName("cardTitle")
            ll.addWidget(lt)
            self.log = QtWidgets.QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(500)
            self.log.setFixedHeight(150)
            ll.addWidget(self.log)
            g.addWidget(lcard, 3, 0, 1, 2)
            g.setColumnStretch(0, 1)
            g.setColumnStretch(1, 1)
            g.setRowStretch(4, 1)
            return sc

        def _view3d_card(self):
            card = QtWidgets.QFrame()
            card.setObjectName("card")
            v = QtWidgets.QVBoxLayout(card)
            hd = QtWidgets.QHBoxLayout()
            t = QtWidgets.QLabel("3D machine view")
            t.setObjectName("cardTitle")
            hd.addWidget(t, 1)
            hint = QtWidgets.QLabel("drag to rotate - wheel to zoom - shift-drag to pan - double-click to reset")
            hint.setObjectName("muted")
            hd.addWidget(hint)
            v.addLayout(hd)
            bar = QtWidgets.QHBoxLayout()
            ctl = {}
            ctl["colour"] = QtWidgets.QComboBox()
            for lab, key in (("Live job data", "live"), ("Coupling J (coil field)", "weight"), ("Flux shell", "shell"),
                             ("27-qubit block type", "block"), ("Half-period module", "module"),
                             ("Kitaev bond colour x/y/z", "colour")):
                ctl["colour"].addItem(lab, key)
            bar.addWidget(QtWidgets.QLabel("Colour"))
            bar.addWidget(ctl["colour"])
            bar.addWidget(QtWidgets.QLabel("Shells"))
            ctl["shells"] = []
            for k in range(5):
                cb = QtWidgets.QCheckBox("s%d" % k)
                cb.setChecked(True)
                ctl["shells"].append(cb)
                bar.addWidget(cb)
            for key, lab, lo, hi, val in (("phi0", "Cut start", 0, 360, 0), ("span", "Toroidal span", 20, 360, 360)):
                bar.addWidget(QtWidgets.QLabel(lab))
                sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                sl.setRange(lo, hi)
                sl.setValue(val)
                sl.setFixedWidth(110)
                ctl[key] = sl
                bar.addWidget(sl)
            bar.addStretch(1)
            v.addLayout(bar)
            bar2 = QtWidgets.QHBoxLayout()
            for key, lab, on in (("bonds", "bonds", True), ("sites", "sites", True), ("coils", "coils", True),
                                 ("div", "divertor", True), ("over", "islands / seams / block", True),
                                 ("spin", "rotate", False)):
                cb = QtWidgets.QCheckBox(lab)
                cb.setChecked(on)
                ctl[key] = cb
                bar2.addWidget(cb)
            self.focus_btn = QtWidgets.QPushButton("Focus job region")
            self.reset_btn = QtWidgets.QPushButton("Reset view")
            bar2.addWidget(self.focus_btn)
            bar2.addWidget(self.reset_btn)
            bar2.addStretch(1)
            v.addLayout(bar2)
            ctl["legend"] = QtWidgets.QLabel("")
            ctl["legend"].setObjectName("legend")
            ctl["legend"].setWordWrap(True)
            self.view = View3D(machine.geometry(), ctl)
            v.addWidget(self.view, 1)
            v.addWidget(ctl["legend"])
            upd = self.view.update
            ctl["colour"].currentIndexChanged.connect(upd)
            for cb in ctl["shells"]:
                cb.toggled.connect(upd)
            for key in ("phi0", "span"):
                ctl[key].valueChanged.connect(upd)
            for key in ("bonds", "sites", "coils", "div", "over"):
                ctl[key].toggled.connect(upd)
            ctl["spin"].toggled.connect(self.view.set_spin)
            self.focus_btn.clicked.connect(self.view.focus)
            self.reset_btn.clicked.connect(lambda: self.view.reset_view())
            return card

        # -- actions --
        def current_job(self):
            return self.job_box.currentData()

        def on_job_changed(self, *args):
            j = self.current_job()
            self.forms.setCurrentWidget(self.form_by_job[j])
            spec = self.schema[j]
            self.about.setText("<b>%s</b><br>%s" % (spec["title"], spec["about"]))
            if not self.S or not self.S["running"]:
                self.set_metric_lists(j)
                self.mark_dirty()
            self.update_estimate()

        def set_metric_lists(self, job):
            spec = self.schema[job]
            self.trace.set_metrics(spec["step_metrics"], spec.get("default_step_metric"))
            self.sweep.set_metrics(spec["summary_metrics"], spec.get("default_summary_metric"))

        def update_estimate(self):
            j = self.current_job()
            fm = self.form_by_job[j]
            n, r = fm.n_points()
            p = r["params"]
            per = {"ness": 0.35 * float(p.get("steps", 135)), "coils": 0.003, "criteria": 62.0,
                   "manifold": (5.0 if str(p.get("qubits", "")).startswith("27") else 0.07) * float(p.get("steps", 20)),
                   "link": 0.001}[j]
            self.estimate.setText("%d point%s - estimated %s" % (n, "" if n == 1 else "s", _ui_dur(n * per)))

        def run_job(self):
            req = self.form_by_job[self.current_job()].request()
            try:
                runner.start(req)
            except (ValueError, RuntimeError) as exc:
                self.log_line("Run refused: %s" % exc, err=True)

        def export_csv(self):
            S = self.S
            if not S:
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", "stellarator_ui_%s.csv" % S["job"], "CSV (*.csv)")
            if not path:
                return
            self.write_csv(path)

        def write_csv(self, path):
            S = self.S
            keys = self.schema[S["job"]]["summary_metrics"]
            axk = [ax_["key"] for ax_ in S["axes"]]
            with open(path, "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["point"] + axk + keys + ["seconds"])
                for i in S["order"]:
                    P = S["points"][i]
                    if P["summary"] is None:
                        continue
                    wr.writerow([i + 1] + [P["coords"][k] for k in axk] + [P["summary"].get(k) for k in keys]
                                + [P.get("seconds")])
            self.log_line("Saved %s" % path)

        def export_json(self):
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export JSON", "stellarator_ui_results.json", "JSON (*.json)")
            if not path:
                return
            with open(path, "w") as fh:
                json.dump(runner.last_results or {}, fh, indent=1, default=_ui_json_default)
            self.log_line("Saved %s" % path)

        def log_line(self, msg, err=False):
            ts = time.strftime("%H:%M:%S")
            if err:
                self.log.appendHtml("<span style='color:%s'>%s  %s</span>" % (C["crit"], ts, msg.replace("<", "&lt;").replace("\n", "<br>")))
            else:
                self.log.appendPlainText("%s  %s" % (ts, msg))

        def mark_dirty(self, *args):
            self.dirty_charts = True

        def closeEvent(self, e):
            runner.stop()
            super().closeEvent(e)

        # -- event handling --
        def poll(self):
            with runner.cond:
                if runner.job_id != self.ev_job:
                    self.ev_job, self.ev_pos = runner.job_id, 0
                batch = runner.events[self.ev_pos:]
                self.ev_pos += len(batch)
            for ev in batch:
                self.on_event(ev)
            now = time.time()
            if self.S and self.S["running"]:
                self.render_header()
            if self.dirty_charts and now - self.last_chart_draw > 0.3:
                self.render_all()
            self.run_btn.setEnabled(not runner.status()["running"])

        def on_event(self, ev):
            t = ev["type"]
            if t == "job_start":
                self.S = {"job": ev["job"], "job_id": ev["job_id"], "params": ev["params"], "axes": ev["axes"],
                          "n_points": ev["n_points"], "points": {}, "order": [], "cur": -1, "step": 0,
                          "n_steps": 0, "status": "running", "running": True, "t0": ev["t"], "t_last": ev["t"],
                          "last_step": None, "phases": [], "done": 0}
                self.view.reset_frame()
                self.view.status_text = ""
                if self.current_job() != ev["job"]:
                    self.job_box.setCurrentIndex(list(self.schema.keys()).index(ev["job"]))
                self.set_metric_lists(ev["job"])
                axs = " - sweep " + " x ".join("%s [%d]" % (a_["key"], len(a_["values"])) for a_ in ev["axes"]) if ev["axes"] else ""
                self.log_line("Job %d started: %s - %d point(s)%s" % (ev["job_id"], self.schema[ev["job"]]["title"], ev["n_points"], axs))
                self.mark_dirty()
                return
            S = self.S
            if not S or ev["job_id"] != S["job_id"]:
                return
            S["t_last"] = ev["t"]
            if t == "point_start":
                S["cur"] = ev["i"]
                S["points"][ev["i"]] = {"coords": ev["coords"], "steps": [], "summary": None}
                S["order"].append(ev["i"])
                S["step"], S["phases"] = 0, []
            elif t == "step":
                P = S["points"].get(ev["i"])
                if P is None:
                    return
                S["step"], S["n_steps"], S["last_step"] = ev["step"], ev["n_steps"], ev["data"]
                if S["job"] == "criteria":
                    S["phases"].append((ev["data"]["phase"], ev["t"]))
                else:
                    keep = {k: ev["data"].get(k) for k in self.schema[S["job"]]["step_metrics"]}
                    P["steps"].append((ev["step"], keep))
                self.view.status_text = "step %d / %d" % (ev["step"], ev["n_steps"]) if S["job"] != "criteria" else ""
            elif t == "frame":
                if ev.get("frame"):
                    self.view.merge_frame(ev["frame"], S["step"])
                return
            elif t == "point_done":
                P = S["points"].get(ev["i"])
                if P is None:
                    return
                P["summary"], P["seconds"] = ev["summary"], ev["seconds"]
                S["done"] += 1
                if S["job"] in ("ness", "manifold", "criteria"):
                    keys = {"ness": ["tau_E", "ex_in_ratio", "per_p2m"], "manifold": ["max_rms", "final_rms", "W0"],
                            "criteria": ["beta_crit_pct", "shafranov_shift_5pct", "fast_ion_survival"]}[S["job"]]
                    self.log_line("Point %d/%d %s done in %.1f s: %s" % (
                        ev["i"] + 1, S["n_points"], self.coord_text(ev["coords"]), ev["seconds"],
                        ", ".join("%s %s" % (_ui_lab(k), _ui_fmt(ev["summary"].get(k))) for k in keys)))
            elif t == "log":
                self.log_line(ev["msg"])
            elif t in ("job_done", "job_cancelled", "job_error"):
                S["running"] = False
                S["status"] = {"job_done": "done", "job_cancelled": "cancelled", "job_error": "error"}[t]
                if t == "job_error":
                    self.log_line("ERROR %s\n%s" % (ev["error"], ev.get("trace", "")), err=True)
                else:
                    self.log_line("Job %d %s after %s (%d point(s))" % (S["job_id"], S["status"], _ui_dur(ev["seconds"]), ev["n_points"]))
                self.render_header()
            self.mark_dirty()

        @staticmethod
        def coord_text(c):
            return "(" + ", ".join("%s=%s" % (k, _ui_fmt(v)) for k, v in c.items()) + ")" if c else ""

        # -- rendering --
        def render_all(self):
            self.last_chart_draw = time.time()
            self.dirty_charts = False
            self.render_header()
            self.render_trace()
            self.render_sweep()
            self.render_detail()
            self.render_table()

        def render_header(self):
            S = self.S
            if not S:
                self.pill.setText("idle")
                self.pill.setProperty("state", "idle")
                return
            st = S["status"]
            if st != self.last_status:
                self.pill.setText(st)
                self.pill.setProperty("state", st)
                self.pill.style().unpolish(self.pill)
                self.pill.style().polish(self.pill)
                self.last_status = st
            frac = (S["done"] + (S["step"] / S["n_steps"] if S["running"] and S["n_steps"] else 0)) / max(1, S["n_points"])
            self.bar_pts.setValue(int(1000 * (frac if S["running"] else S["done"] / max(1, S["n_points"]))))
            self.bar_step.setValue(int(1000 * S["step"] / S["n_steps"]) if S["n_steps"] else 0)
            el = (time.time() if S["running"] else S["t_last"]) - S["t0"]
            eta = el * (1 - frac) / frac if S["running"] and frac > 0.002 else 0
            cur = S["points"].get(S["cur"])
            txt = "Job %d - point %d / %d %s" % (S["job_id"], min(S["done"] + (1 if S["running"] else 0), S["n_points"]),
                                                S["n_points"], self.coord_text(cur["coords"]) if cur else "")
            if S["n_steps"]:
                txt += "    %s %d / %d" % ("phase" if S["job"] == "criteria" else "step", S["step"], S["n_steps"])
            txt += "    elapsed %s" % _ui_dur(el) + ("    ETA %s" % _ui_dur(eta) if eta else "")
            self.prog_text.setText(txt)

        def axis_info(self):
            S = self.S
            if not S or not S["axes"]:
                return None
            a0 = S["axes"][0]
            v = a0["values"]
            return {"key": a0["key"], "lo": min(v), "hi": max(v), "log": a0.get("scale") == "log" and min(v) > 0}

        def render_trace(self):
            pnl, S = self.trace, self.S
            job = S["job"] if S else self.current_job()
            spec = self.schema[job]
            fig = pnl.fig
            fig.clear()
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.17)
            if job == "criteria":
                pnl.title.setText("Evaluation phases")
                self.draw_phases(ax)
                pnl.legend.setText("")
            elif not spec["step_metrics"]:
                self.draw_profile(ax, job, pnl)
            else:
                pnl.title.setText("Live step trace")
                m = pnl.key() or spec["default_step_metric"]
                if not S:
                    empty_msg(ax, "Set parameters and press Run")
                    pnl.legend.setText("")
                else:
                    ai = self.axis_info()
                    style_ax(ax, "step", _ui_lab(m))
                    drew = False
                    for i in S["order"]:
                        P = S["points"][i]
                        if not P["steps"]:
                            continue
                        xs = [r[0] for r in P["steps"]]
                        ys = [np.nan if r[1].get(m) is None else r[1][m] for r in P["steps"]]
                        running = i == S["cur"] and S["running"]
                        if running:
                            col, lw, z = C["s2"], 2.4, 5
                        elif ai:
                            v = P["coords"][ai["key"]]
                            t_ = ((math.log10(v) - math.log10(ai["lo"])) / max(1e-12, math.log10(ai["hi"]) - math.log10(ai["lo"]))
                                  if ai["log"] else (v - ai["lo"]) / max(1e-12, ai["hi"] - ai["lo"]))
                            col, lw, z = CM_TRACE(t_), 1.6, 3
                        else:
                            col, lw, z = C["s1"], 1.8, 3
                        ax.plot(xs, ys, color=col, lw=lw, zorder=z, solid_capstyle="round")
                        drew = True
                    if m in UI_LOGY:
                        ax.set_yscale("log")
                    if not drew:
                        empty_msg(ax, "First step is running")
                    leg = ("%s %s to %s (dark to light)" % (_ui_lab(ai["key"]), _ui_fmt(ai["lo"]), _ui_fmt(ai["hi"]))
                           if ai else _ui_lab(m))
                    if S["running"]:
                        leg += "   |   orange: running point"
                    pnl.legend.setText(leg)
            pnl.canvas.draw_idle()

        def draw_phases(self, ax):
            S = self.S
            if not S:
                empty_msg(ax, "Run the 7-criteria evaluation to follow its phases here")
                return
            seen = [p[0] for p in S["phases"] if p[0] != "done"]
            cur_done = S["cur"] >= 0 and S["points"].get(S["cur"], {}).get("summary") is not None
            durs, cols = [], []
            for k, lb in enumerate(UI_PHASES):
                if lb in seen:
                    idx = seen.index(lb)
                    t_end = S["phases"][idx + 1][1] if idx + 1 < len(S["phases"]) else (
                        S["t_last"] if cur_done else time.time())
                    durs.append(t_end - S["phases"][idx][1])
                    cols.append(C["s2"] if (idx == len(seen) - 1 and S["running"] and not cur_done) else C["s1"])
                else:
                    durs.append(0.0)
                    cols.append(C["mid"])
            y = np.arange(len(UI_PHASES))[::-1]
            style_ax(ax, "seconds", None)
            ax.barh(y, durs, color=cols, height=0.62)
            ax.set_yticks(y)
            ax.set_yticklabels(UI_PHASES, color=C["t2"], fontsize=8)
            ax.grid(True, axis="x", color=C["grid"])
            ax.grid(False, axis="y")
            for yy, d in zip(y, durs):
                if d > 0:
                    ax.text(d, yy, " %.1f s" % d, va="center", color=C["t2"], fontsize=8)
            ax.figure.subplots_adjust(left=0.36)

        def draw_profile(self, ax, job, pnl):
            title, keys, ylabel = UI_PROFILE[job]
            pnl.title.setText(title)
            S = self.S
            if not S or not S["axes"]:
                empty_msg(ax, "Sweep a parameter to see the profile" if S else "Set parameters and press Run")
                pnl.legend.setText("")
                return
            ai = self.axis_info()
            kx = ai["key"]
            pts = [S["points"][i] for i in S["order"] if S["points"][i]["summary"] is not None]
            note = ""
            if len(S["axes"]) == 2 and pts:
                ky = S["axes"][1]["key"]
                yv = pts[-1]["coords"][ky]
                pts = [P for P in pts if P["coords"][ky] == yv]
                note = " at %s = %s" % (_ui_lab(ky), _ui_fmt(yv))
            pts.sort(key=lambda P: P["coords"][kx])
            style_ax(ax, _ui_lab(kx) + note, ylabel)
            xs = [P["coords"][kx] for P in pts]
            for n_, k in enumerate(keys):
                ax.plot(xs, [P["summary"].get(k) for P in pts], color=UI_CAT8[n_], lw=2, label=_ui_lab(k))
            if ai["log"]:
                ax.set_xscale("log")
            if pts:
                lg = ax.legend(fontsize=7, frameon=False, labelcolor=C["t2"], ncol=2, loc="best")
                lg.set_zorder(10)
            pnl.legend.setText("")

        def render_sweep(self):
            pnl, S = self.sweep, self.S
            fig = pnl.fig
            fig.clear()
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.17)
            job = S["job"] if S else self.current_job()
            m = pnl.key() or self.schema[job]["default_summary_metric"]
            if not S:
                empty_msg(ax, "Results appear here point by point")
                pnl.legend.setText("")
                pnl.canvas.draw_idle()
                return
            done = [S["points"][i] for i in S["order"] if S["points"][i]["summary"] is not None]
            if len(S["axes"]) == 2:
                pnl.title.setText("Sweep result (heatmap)")
                xs, ys = S["axes"][0]["values"], S["axes"][1]["values"]
                grid = np.full((len(ys), len(xs)), np.nan)
                xi = {v: k for k, v in enumerate(xs)}
                yi = {v: k for k, v in enumerate(ys)}
                for P in done:
                    a_, b_ = xi.get(P["coords"][S["axes"][0]["key"]]), yi.get(P["coords"][S["axes"][1]["key"]])
                    if a_ is not None and b_ is not None:
                        val = P["summary"].get(m)
                        grid[b_, a_] = np.nan if val is None else val
                style_ax(ax, _ui_lab(S["axes"][0]["key"]), _ui_lab(S["axes"][1]["key"]))
                ax.grid(False)
                fin = grid[np.isfinite(grid)]
                if len(fin):
                    lo, hi = float(fin.min()), float(fin.max())
                    if lo < 0 < hi:
                        mm = max(abs(lo), abs(hi))
                        im = ax.pcolormesh(xs, ys, grid, cmap=CM_DIV, vmin=-mm, vmax=mm, shading="nearest")
                    else:
                        im = ax.pcolormesh(xs, ys, grid, cmap=CM_SEQ, vmin=lo, vmax=hi if hi > lo else lo + 1e-9, shading="nearest")
                    cb = fig.colorbar(im, ax=ax, pad=0.02)
                    cb.ax.tick_params(colors=C["t3"], labelsize=7)
                    cb.outline.set_edgecolor(C["axis"])
                    pnl.legend.setText("%s: %d / %d points" % (_ui_lab(m), len(done), S["n_points"]))
            elif not S["axes"]:
                pnl.title.setText("Result")
                ax.set_axis_off()
                if done:
                    v = done[0]["summary"].get(m)
                    ax.text(0.5, 0.55, _ui_fmt(v), color=C["t1"], fontsize=30, ha="center", va="center",
                            transform=ax.transAxes, fontweight="bold")
                    ax.text(0.5, 0.32, _ui_lab(m), color=C["t2"], fontsize=11, ha="center", transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, "Single-point run: value appears when it finishes", color=C["t3"],
                            ha="center", va="center", transform=ax.transAxes)
                pnl.legend.setText("")
            else:
                pnl.title.setText("Sweep result")
                ai = self.axis_info()
                pts = sorted(done, key=lambda P: P["coords"][ai["key"]])
                xs = [P["coords"][ai["key"]] for P in pts]
                ys = [np.nan if P["summary"].get(m) is None else P["summary"][m] for P in pts]
                style_ax(ax, _ui_lab(ai["key"]), _ui_lab(m))
                if xs:
                    ax.plot(xs, ys, color=C["s1"], lw=2, marker="o" if len(xs) <= 80 else None, ms=5,
                            mec=C["surf"], mew=1.5)
                else:
                    empty_msg(ax, "First point is running")
                if ai["log"]:
                    ax.set_xscale("log")
                if m in UI_LOGY:
                    ax.set_yscale("log")
                pnl.legend.setText("%s vs %s - %d / %d points" % (_ui_lab(m), _ui_lab(ai["key"]), len(done), S["n_points"]))
            pnl.canvas.draw_idle()

        def render_detail(self):
            S, pnl = self.S, self.detail
            job = S["job"] if S else self.current_job()
            fig = pnl.fig
            fig.clear()
            if job in ("ness", "manifold") or not S:
                self.detail_stack.setCurrentIndex(0)
            if not S:
                pnl.title.setText("Detail")
                empty_msg(fig.add_subplot(111), "Shell energies and divertor loads (NESS), bond agreement (PyQrack),\nlast result for the other jobs")
                pnl.canvas.draw_idle()
                return
            d = S["last_step"]
            if job == "ness":
                pnl.title.setText("Shell energy and divertor load (current step)")
                pnl.hint.setText("step %d" % S["step"] if d else "")
                ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
                fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.2, wspace=0.3)
                style_ax(ax1, "shell", "excess energy")
                style_ax(ax2, "divertor unit", "cumulative / mean")
                if d:
                    ax1.bar(["s0", "s1", "s2", "s3", "s4"], d["shell"], color=C["s1"], width=0.85)
                    ue = np.asarray(d["unit_exhaust"], dtype=float)
                    ax2.bar(["D%d" % (k + 1) for k in range(10)], ue / max(float(np.mean(ue)), 1e-12),
                            color=C["s3"], width=0.85)
                    ax2.tick_params(axis="x", labelsize=7)
                ax1.grid(False, axis="x")
                ax2.grid(False, axis="x")
            elif job == "manifold":
                pnl.title.setText("Bond expectations: PyQrack vs exact (current step)")
                pnl.hint.setText("step %d - dashed = perfect agreement" % S["step"] if d else "")
                ax = fig.add_subplot(111)
                fig.subplots_adjust(left=0.14, right=0.97, top=0.95, bottom=0.17)
                style_ax(ax, "exact Majorana <sigma sigma>", "PyQrack <sigma sigma>")
                if d and d.get("bond_exact"):
                    xe, yq = np.asarray(d["bond_exact"]), np.asarray(d["bond_qrack"])
                    lo, hi = float(min(xe.min(), yq.min())), float(max(xe.max(), yq.max()))
                    ax.plot([lo, hi], [lo, hi], ls="--", color=C["axis"], lw=1)
                    ax.scatter(xe, yq, s=26, color=C["s2"], edgecolors=C["surf"], linewidths=1.2, zorder=3)
            else:
                pnl.title.setText({"criteria": "Last 7-criteria result", "coils": "Last coil point",
                                   "link": "Last link budget"}[job])
                self.detail_stack.setCurrentIndex(1)
                last = [S["points"][i] for i in S["order"] if S["points"][i]["summary"] is not None]
                pnl.hint.setText(self.coord_text(last[-1]["coords"]) if last else "")
                rows = []
                if last:
                    s_ = last[-1]["summary"]
                    rows = [(_ui_lab(k), _ui_fmt(v)) for k, v in s_.items() if not isinstance(v, (dict, list))]
                    rows += [("%s (abs)" % k, _ui_fmt(v)) for k, v in (s_.get("raw") or {}).items()]
                self.kv.setRowCount(len(rows))
                for r_, (k, v) in enumerate(rows):
                    it_k = QtWidgets.QTableWidgetItem(k)
                    it_v = QtWidgets.QTableWidgetItem(v)
                    it_v.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                    self.kv.setItem(r_, 0, it_k)
                    self.kv.setItem(r_, 1, it_v)
                self.kv.resizeColumnToContents(0)
            pnl.canvas.draw_idle()

        def render_table(self):
            S = self.S
            if not S:
                self.table.setRowCount(0)
                self.table.setColumnCount(0)
                self.table_hint.setText("")
                return
            keys = self.schema[S["job"]]["summary_metrics"]
            axk = [a_["key"] for a_ in S["axes"]]
            hdr = ["#"] + [_ui_lab(k) for k in axk] + [_ui_lab(k) for k in keys] + ["s"]
            if self.table.columnCount() != len(hdr):
                self.table.setColumnCount(len(hdr))
            self.table.setHorizontalHeaderLabels(hdr)
            order = S["order"][-400:]
            self.table_hint.setText(("showing last 400 of %d" % len(S["order"])) if len(S["order"]) > 400 else "%d point(s)" % len(S["order"]))
            self.table.setRowCount(len(order))
            for r_, i in enumerate(order):
                P = S["points"][i]
                running = i == S["cur"] and S["running"] and P["summary"] is None
                vals = [str(i + 1)] + [_ui_fmt(P["coords"][k]) for k in axk]
                vals += [(_ui_fmt(P["summary"].get(k)) if P["summary"] else ("..." if running else "")) for k in keys]
                vals += ["%.1f" % P["seconds"] if P.get("seconds") else ""]
                for c_, v in enumerate(vals):
                    it = QtWidgets.QTableWidgetItem(v)
                    it.setTextAlignment((QtCore.Qt.AlignLeft if c_ == 0 else QtCore.Qt.AlignRight) | QtCore.Qt.AlignVCenter)
                    if running:
                        it.setForeground(QtGui.QBrush(QtGui.QColor(C["s2"])))
                    self.table.setItem(r_, c_, it)
            self.table.resizeColumnsToContents()
            if S["running"]:
                self.table.scrollToBottom()

    QSS = """
    QWidget { background: %(page)s; color: %(t1)s; font-size: 12px; }
    QFrame#header { background: %(surf)s; border-bottom: 1px solid %(border)s; }
    QFrame#sidebar { background: %(surf)s; border-right: 1px solid %(border)s; }
    QFrame#sidebar QWidget { background: %(surf)s; }
    QWidget#mainbody { background: %(page)s; }
    QFrame#card { background: %(surf)s; border: 1px solid %(border)s; border-radius: 8px; }
    QFrame#card QWidget { background: %(surf)s; }
    QFrame#prm { border-top: 1px solid %(border)s; }
    QLabel#h1 { font-size: 15px; font-weight: 600; }
    QLabel#cardTitle { font-size: 13px; font-weight: 600; }
    QLabel#muted, QLabel#legend { color: %(t3)s; font-size: 11px; }
    QLabel#about { color: %(t2)s; font-size: 11px; }
    QLabel#prmName { color: %(t2)s; }
    QLabel#pill { border: 1px solid %(border)s; border-radius: 9px; padding: 2px 10px; font-weight: 600; color: %(t2)s; }
    QLabel#pill[state="running"] { color: %(s1)s; border-color: %(s1)s; }
    QLabel#pill[state="done"] { color: %(good)s; border-color: %(good)s; }
    QLabel#pill[state="cancelled"] { color: %(warn)s; border-color: %(warn)s; }
    QLabel#pill[state="error"] { color: %(crit)s; border-color: %(crit)s; }
    QProgressBar { background: %(surf2)s; border: none; border-radius: 3px; }
    QProgressBar::chunk { background: %(s1)s; border-radius: 3px; }
    QProgressBar#stepbar::chunk { background: %(s3)s; }
    QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTableWidget {
        background: %(page)s; border: 1px solid %(border)s; border-radius: 5px; padding: 3px 5px; }
    QHeaderView::section { background: %(surf)s; color: %(t2)s; border: none; border-bottom: 1px solid %(border)s; padding: 4px; }
    QTableWidget { gridline-color: %(border)s; }
    QPushButton { background: %(surf2)s; border: 1px solid %(border)s; border-radius: 6px; padding: 6px 10px; font-weight: 600; }
    QPushButton:hover { border-color: %(s1)s; }
    QFrame#sidebar QPushButton { background: %(surf2)s; }
    QFrame#sidebar QPushButton#run { background: %(s1)s; border-color: %(s1)s; color: white; }
    QFrame#sidebar QPushButton#run:disabled { background: #24476f; color: #9fb4cc; }
    QFrame#card QPushButton { background: %(surf2)s; }
    QFrame#card QComboBox, QFrame#card QPlainTextEdit, QFrame#card QTableWidget,
    QFrame#sidebar QComboBox, QFrame#sidebar QSpinBox, QFrame#sidebar QDoubleSpinBox { background: %(page)s; }
    QCheckBox { color: %(t2)s; spacing: 5px; }
    QCheckBox::indicator { width: 13px; height: 13px; border: 1px solid #55544f; border-radius: 3px; background: %(page)s; }
    QCheckBox::indicator:checked { background: %(s1)s; border-color: %(s1)s; }
    QSlider::groove:horizontal { height: 4px; background: %(surf2)s; border-radius: 2px; }
    QSlider::sub-page:horizontal { background: %(s1)s; border-radius: 2px; }
    QSlider::handle:horizontal { background: %(t1)s; width: 12px; margin: -5px 0; border-radius: 6px; }
    QScrollArea { border: none; }
    QSplitter::handle { background: %(border)s; }
    QToolTip { background: #0d0d0c; color: %(t1)s; border: 1px solid %(border)s; padding: 4px; }
    """ % C

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    win = MainWindow()
    win.resize(1600, 1780 if a.selftest else 1000)
    win.show()
    if a.selftest:
        rc = _ui_selftest(app, win, a, runner)
    else:
        rc = app.exec()
    win.poll_timer.stop()
    win.view.spin_timer.stop()
    win.deleteLater()
    app.processEvents()
    return rc


def _ui_selftest(app, win, a, runner):
    """Headless check: run a short NESS sweep (and optionally more jobs),
    then save window screenshots next to `a.selftest`."""
    base, ext = os.path.splitext(a.selftest)
    ext = ext or ".png"
    plan = [
        ("ness", {"steps": 30, "field": "trimmed"}, {"key": "gamma_div", "min": 0.6, "max": 1.6, "n": 2, "scale": "lin"}, None, ""),
        ("coils", {}, {"key": "I1", "min": 0.8, "max": 1.2, "n": 21, "scale": "lin"},
         {"key": "I3", "min": 0.8, "max": 1.2, "n": 15, "scale": "lin"}, "_coils"),
        ("manifold", {"steps": 12, "kick": 0.4}, {"key": "dt", "min": 0.02, "max": 0.12, "n": 3, "scale": "lin"}, None, "_manifold"),
        ("link", {}, {"key": "n_ebits", "min": 1, "max": 100000, "n": 40, "scale": "log"}, None, "_link"),
    ]
    if not a.selftest_all:
        plan = plan[:1]
    saved = []
    for job, params, sw, sw2, suffix in plan:
        fm = win.form_by_job[job]
        req = fm.request()
        req["params"].update(params)
        req["sweep"], req["sweep2"] = sw, sw2
        runner.start(req)
        t0 = time.time()
        while True:
            app.processEvents()
            time.sleep(0.05)
            st = runner.status()
            if not st["running"] and time.time() - t0 > 0.5:
                break
            if time.time() - t0 > 600:
                break
        for _ in range(30):
            app.processEvents()
            time.sleep(0.02)
        win.poll()
        win.render_all()
        if job == "manifold":
            win.view.focus()
        app.processEvents()
        path = base + suffix + ext
        win.grab().save(path)
        saved.append(path)
        if job == "manifold":
            win.view.reset_view()
    for p in saved:
        print("saved %s" % p, flush=True)
    win.close()
    app.processEvents()
    return 0

def main_ui(argv=None):
    ap = argparse.ArgumentParser(prog="stellarator_ui.py",
                                 description="Native live dashboard (PySide6): parameter sweeps with 2D charts and a 3D machine view.")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--scoreboard", default="scoreboard.json")
    ap.add_argument("--selftest", default=None, metavar="PNG",
                    help="run a short NESS sweep headless (QT_QPA_PLATFORM=offscreen) and save a screenshot")
    ap.add_argument("--selftest-all", action="store_true",
                    help="with --selftest: also run coil, PyQrack and link jobs, one screenshot each")
    a = ap.parse_args(argv)
    try:
        import PySide6  # noqa: F401
    except ImportError:
        print("The ui subcommand needs PySide6:  pip install PySide6", file=sys.stderr)
        sys.exit(1)
    if a.selftest and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    machine = LiveMachine(a.manifest, a.scoreboard)
    print("Building machine context ...", flush=True)
    machine.ctx()
    runner = LiveJobRunner(machine)
    try:
        rc = _ui_run_qt(a, machine, runner)
    finally:
        runner.stop()
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    main_ui()
