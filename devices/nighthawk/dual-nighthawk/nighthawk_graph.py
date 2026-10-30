#!/usr/bin/env python3
"""
nighthawk_graph.py -- matplotlib views of everything nighthawk_qrack.py produces.

Reads, for every configuration found (backend / shots / twirls / ACE seams):
  run records    <out>.jsonl and every worker's <out>.w<pid>.jsonl
  bitstrings     <out>_shots/<cfg>/points/n<n>/*.npz (mirror, patched, full)
  hwxeb output   the --out JSON of `nighthawk_qrack.py hwxeb` (optional)
  the release    data/results/fidelity_vs_depth.json, patch_xeb.json, layout.json

Figures (one window each; --save writes PNGs too):
  1  fidelity vs depth at 61 qubits: ibm_phoenix mirror / 3-patch / 4-patch + fit,
     the clean-qubit runs, and the PyQrack re-score of the hardware counts
  2  mirror decay per register size (the --sizes sweep)
  3  per-cycle decay b(N), the u*N + v*CZ/cycle fit and its value at 61 qubits
  4  mirror bitstrings: Hamming distance from the prepared string, per-qubit error
     rate on the device grid, survival per twirl randomisation and per input string
  5  patched and full bitstrings: per-patch XEB, per-circuit XEB, per-qubit P(1) on the
     grid, Hamming-weight distribution against the uniform binomial
  6  release + hwxeb: collision ratios, PyQrack vs release per-circuit XEB, residuals
  7  run cost: seconds per point against register size and depth
  8  fxeb: forward XEB against depth, one panel per register size, every configuration
  9  fxeb: forward XEB against register size, one panel per depth
 10  fxeb: per-cycle decay b(N) per configuration, extrapolated to 61 qubits against
     ibm_phoenix's b
 11  fxeb: what predicts the XEB -- every ACE configuration at one (n, d) against its
     coupler split (exact / replica / cross), seam qubits used, simulators used,
     widest simulator and effective bulk-to-boundary ratio of the placed qubits
     (nn_qab.py's B-to-B), with the rank correlation of each
 12  hamming: Dan Strano's mirror Hamming signal 1 - 2h against depth, one panel per
     register size, folds lambda = 1 / 3 / 5, with his ACE calculator at the depth of U
 13  hamming: measured / calculator against register size per depth, and the effective
     folding exponent q (s(lambda) = s(1)^(lambda^q)) against his gain exponent 1.606
 14  shor9: decoded logical-Z and logical-X success against code size (Shor-9, -16, -25,
     -36, ...) per layout and setup (PyQrack version, boundary repetition code, OpenCL),
     with the backend-to-backend standard error and every fresh backend as a dot
 15  shor9: per logical state, the share of outcomes the exact state never gives and the
     total variation distance to the exact distribution against its shot-noise floor
 16  shor9 --register: every unit's decoded Z / X and its Z-basis impossible share against
     the encoder CXs through the seam, by placement (one patch, touches seam)
 17  shor9 --register on the lattice: per register, code size and setup, the qubits by ACE
     patch, seam qubits and seam lines (labelled with the patches they join), couplers by class (exact / replica / cross), block leaders, and each
     unit's square filled by its weaker logical basis and labelled Z / X / impossible share.
     The map is the run's shor9_map record; older runs' maps are rebuilt (release + PyQrack)

 18  ghz tiles (nighthawk_qrack.py ghz tiles) on the lattice: per register, unit size and Layer 2 mode, each site
     coloured by the mean fidelity of the units holding it (F_GHZ of the formal proof's Lemma 4, else the
     Clifford witness), seam qubits and lines, couplers by class
 19  ghz tiles: mean fidelity by placement (off the seam; holding a seam qubit with k CXs through the seam),
     one bar per Layer 2 mode (none, unmitigated, compensated, dd = Walsh echo), the exact reference as a tick
 20  ghz tiles: every unit's ACE fidelity against its exact reference, one panel per Layer 2 mode, by placement
 21  ghz chip (nighthawk_qrack.py ghz chip): per register, tree and Layer 2 mode, each site's share of Z shots that
     differ from the state's majority, the tree's CXs, the breaking ones labelled, the root
 22  engines (nighthawk_qrack.py ghz stack / tiles with --engine loom|bridge, --box L H): mean F_GHZ per placement
     (inside a chip or face; across the loom, the bridge, a box edge direct / relayed, through the box ring's
     wrap cut) against QrackBridge max_bonds, one panel per geometry, the loom's hard cut at the left
 23  shor across QrackBridge (nighthawk_qrack.py ghz shor): decoded logical Z and X per geometry and code size
     against max_bonds, by how the cut meets the code (inside one circuit, between blocks, through blocks,
     through the wrap cut), with the majority-over-block-signs prediction min(m, max_bonds)
 24  dual Nighthawk R1 (nighthawk_qrack.py ghz dual, arXiv:2610.29986): every state's fidelity under each model
     (hard cut, exact bridge, constituent budget, measured distant CZ) with IBM's raw and mitigated values
 25  dual Nighthawk R1 heatmap (nighthawk_qrack.py ghz dualmap): both 12 x 10 modules (B turned 180 degrees), the
     L-qubits and their cables labelled with the distant-CZ fidelity, each site the lowest F_GHZ of the
     3-qubit chains through it, one panel per model
 26  Layer 3, Shor order finding (nighthawk_qrack.py ghz shoralg) on QrackBridge: per geometry, N, a and counting width,
     the per-shot factoring success (the order found where r is odd) of every layout and placement against max_bonds,
     the ideal circuit and a uniform guess as lines, the bonds the rule needs (ceil(log2 r) + 1 across the registers)
 27  the same, the total variation distance to the ideal distribution (shot-noise floors dotted)
 28  Layer 3 on the dual Nighthawk R1 (shoralg --dual): every layout's success under each model, with its routing counts

3D (--grid3d DIR, or the viewer's "Open shor9 / GHZ registers in 3D" button): view 17 as a rotatable page per
register and code size, written to DIR with an index; each unit a column whose height is the logical
information it kept, setups side by side, and a switch that pulls the patches apart into ACE's
simulators. With ghz records also one page per register (views 18 and 21 in three dimensions): every site a
column, its height the mean GHZ fidelity of the tiled chains through it or, for a chip-wide state, the share
of shots where it agrees with the majority, (v - 0.5) / 0.5; a switch per Layer 2 mode and chip-wide tree,
ACE or the exact reference, and the chains through a clicked site. three.js comes from jsDelivr; --three DIR uses a local npm `three` package and --serve
serves the pages on 127.0.0.1 (module scripts do not load from file://).

shor9 records (`nighthawk_qrack.py shor9 --out`) are found in the harvested *.jsonl or given
with --shor; views 14-16 need no BlueQubit release. --table FILE also writes
FILE_shor9.csv (one row per code, layout and setup, and one per register unit, with the top
wrong patterns) and prints the shor9 summary.

ghz records (`nighthawk_qrack.py ghz chip|tiles --out`) are found in the harvested *.jsonl or given with --ghz;
views 18-21 need no BlueQubit release (the register map comes from the run's ghz_map record). --table FILE
also writes FILE_ghz.csv (one row per state: Layer 1a numbers in the reference implementation's names, ACE
and exact, the Layer 2 model, placement, and the engine: engine, loom_read, chip_backend, chips, max_bonds,
truncation, lazy_bonds, box, box_edge, box_cuts, chips_spanned, inside_zz, stack_order, stack_pass,
seam_class, seam_plan) and prints the ghz summary.

shor_bridge, dual_r1 and dual_map records (nighthawk_qrack.py ghz shor / dual / dualmap --out) are read the same
way; --table FILE writes FILE_shorbridge.csv (one row per unit: layout, hops, wrap, bonds kept for Z and X,
decoded and raw per logical state), FILE_dual.csv (one row per state and model, with IBM's values) and
FILE_dualmap.csv (one row per chain and model) and, for shor_alg records, FILE_shoralg.csv (one row per circuit), and
prints their summaries. The setup label of every engine
run names its parameters ("QrackBridge max_bonds 2, box 5x5x5 ring, code first, inside ZZ by stabiliser").

Harvest: with no --out, every *.jsonl in the working directory is read (worker files
are folded into their main file), so every sweep collected so far is set side by side.
Configurations are labelled from what their records say (lrc/lrr, torus, tiled, shots)
instead of by config hash. --table FILE writes every point of every configuration as
CSV (with the coupler split, recomputed where older records lack it) and prints a
per-(n, d) comparison; then it exits.

XEB points in views 8-11 are the plain mean over instances, with the standard error
from the instance scatter (instances are different circuits, and their spread is far
larger than shot noise); nighthawk_qrack.py's summary uses inverse-variance weights.

Viewer (default when a display is available): a Tk window with the list of views on
the left, the selected figure on the right with matplotlib's zoom/pan toolbar, and
selectors for configuration, register size and depth. Reload re-reads every file;
Auto-reload refreshes every 60 s, so it can follow a sweep that is still running.
Save PNG writes the current view, Save all writes every view.

Usage
  python3 nighthawk_graph.py                                               # viewer, every *.jsonl here
  python3 nighthawk_graph.py --out clean_ace.jsonl --hwxeb hwxeb.json      # viewer, chosen files
  python3 nighthawk_graph.py --save plots                                  # PNGs only
  python3 nighthawk_graph.py --table harvest.csv                           # CSV + comparison table
  python3 nighthawk_graph.py --ghz ghz61.jsonl --figs 18-21 --save plots   # the ghz views as PNGs
  python3 nighthawk_graph.py --ghz box_mb2.jsonl --ghz shor9_box5_mb2.jsonl --ghz dual_r1.jsonl \
          --ghz dualmap.jsonl --figs 18-25 --save plots                    # bridge, box, shor, dual R1
  python3 nighthawk_graph.py --grid3d nighthawk_3d --open                  # shor9 registers in 3D
  python3 nighthawk_graph.py --grid3d nighthawk_3d --three node_modules/three --serve   # the same, offline
Without a display (no DISPLAY / WAYLAND_DISPLAY) it writes PNGs to ./nighthawk_plots.
"""

import os

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):
    os.environ["PYQRACK_SHARED_LIB_PATH"] = QRACK_LIB_PATH

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_nh():
    p = Path(__file__).resolve().parent / "nighthawk_qrack.py"
    if not p.exists():
        raise SystemExit(f"nighthawk_qrack.py not found next to this script ({p.parent})")
    spec = importlib.util.spec_from_file_location("nighthawk_qrack", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nh = load_nh()

FAM_STYLE = {"mirror": ("o", "C0"), "3-patch": ("s", "C1"), "4-patch": ("D", "C2")}
LABELS = {}                 # cfg tag -> readable label, filled by Store.reload()


def lab(cfg):
    """Readable label of a configuration, with its tag for traceability."""
    return f"{LABELS[cfg]} [{cfg}]" if cfg in LABELS else cfg


ENSEMBLE = {}               # cfg tag -> "haar" | "nnqab", filled by Store.reload()
THETA = {"show": "haar"}    # --theta: which ensemble views 8-11 compare


def ensemble(recs):
    """Single-qubit ensemble of a configuration: the records' own `theta` (written since
    it was added), else inferred for fxeb records from the ideal XEB at d >= 8 -- the
    nn_qab ensemble's outputs are far more concentrated (ideal XEB ~ 16 at d = 12 against
    ~ 1.6 for Haar circuits), so > 5 marks it. Mirror-only records without `theta` are
    taken as Haar, the only ensemble they were run with before the field existed."""
    rs = list(recs.values())
    th = {r["theta"] for r in rs if r.get("theta")}
    if th:
        return th.pop() if len(th) == 1 else "mixed"
    ideal = [r["ideal_xeb"] for r in rs if r.get("family") == "fxeb" and r.get("depth", 0) >= 8
             and r.get("ideal_xeb") is not None]
    return "nnqab" if ideal and float(np.median(ideal)) > 5 else "haar"


def describe(cfg, recs, lay):
    """Label a configuration from what its records say rather than by its hash."""
    lab_ = _describe(cfg, recs, lay)
    return lab_ + ("" if ENSEMBLE.get(cfg, "haar") == "haar" else f" [theta {ENSEMBLE[cfg]}]")


def _describe(cfg, recs, lay):
    backend = cfg.split("-")[0]
    rs = list(recs.values())
    fams = sorted({r["family"] for r in rs})
    shots = sorted({r.get("shots") for r in rs if r["family"] == "fxeb" and r.get("shots")})
    extra = (f", {shots[0]} shots" if len(shots) == 1 and shots[0] != nh.FXEB_SHOTS else "")
    if backend != "ace":
        return f"{backend} {'/'.join(fams)}{extra}"
    geos = {(r.get("ace_lrc"), r.get("ace_lrr"), bool(r.get("ace_torus")), bool(r.get("ace_tiling")))
            for r in rs if "ace_lrc" in r}
    regs = {r.get("ace_register") for r in rs if "ace_register" in r}
    if not geos:
        return f"ace (no geometry in records){extra}"
    if len(geos) > 1:
        lab_ = "ace max-width (layout per size)"
        if all(g[3] for g in geos):
            lab_ += " tiled"
    else:
        lrc, lrr, torus, tiled = next(iter(geos))
        vers = {r.get("ace_tiling_version") for r in rs if r.get("ace_tiling")}
        tv = f" tiled v{vers.pop()}" if len(vers) == 1 and None not in vers else " tiled"
        lab_ = f"ace {lrc}/{lrr}" + (" torus" if torus else "") + (tv if tiled else " untiled")
        txr = [r for r in rs if r.get("ace_transpile")]
        if txr:
            r0 = txr[0]
            lab_ = (f"ace {lrc}/{lrr}" + (" torus" if torus else "") + f" {r0.get('ace_transpile_engine', 'anneal')}"
                    + f" w{r0.get('ace_register')}" + (f" ex{len(r0['ace_exclude'])}" if r0.get("ace_exclude") else "")
                    + (" xok" if r0.get("ace_transpile_allow_cross") else ""))
    if any(g != lay.grid[0] * lay.grid[1] for g in regs):
        lab_ += " strip"
    for key, val, tok in (("ace_boundary_rep", True, " rep"), ("ace_error_detection", False, " noED"),
                          ("ace_crossbars", False, " noXbar")):
        if any(r.get(key) == val for r in rs):
            lab_ += tok
    return lab_ + extra


# ================================================================== data
def read_records(out):
    """{cfg: {key: record}} over the main file and all worker files, any configuration."""
    by = defaultdict(dict)
    for f in nh.record_files(out):
        if not f.exists():
            continue
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "cfg" not in r:
                continue
            f_ = r.get("fidelity")
            if isinstance(f_, float) and not math.isfinite(f_):   # a broken run's record (NaN): ignored
                continue
            r.setdefault("n", 61)
            by[r["cfg"]][(r["n"], r["family"], r["K"], r["depth"], r["partition"], r["instance"])] = r
    return by


def shots_base(out, shots_dir):
    return Path(shots_dir) if shots_dir else Path(out).with_name(Path(out).stem + "_shots")


def point_npz(base, cfg, key):
    n, fam, K, d, j, i = key
    return base / cfg / "points" / f"n{n}" / f"{fam}_K{K}_d{d:02d}_p{j}_i{i}.npz"


def popcount64(x):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.uint64))
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def bit_matrix(x, n):
    x = np.asarray(x, dtype=np.uint64)
    return ((x[:, None] >> np.arange(n, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8)


def point_means(recs, n, fam_of):
    """{(fam, d): (F, se)} over instances/partitions, inverse-variance as in the summary."""
    acc = defaultdict(list)
    for (rn, fam, K, d, j, i), r in recs.items():
        if rn != n or r.get("fidelity") is None or fam in ("hamming", "vqe"):   # hamming: views 12-13; vqe: energies
            continue
        acc[(fam_of(fam, K), d)].append((r["fidelity"], r["se"]))
    out = {}
    for k, v in acc.items():
        out[k] = nh._point(v)
    return out


def fam_label(fam, K):
    return "mirror" if fam == "mirror" else f"{K}-patch"


def logsafe(F, floor):
    return max(F, floor)


# ================================================================== figures
def fig_fidelity(plt, lay, fvd, data, hw_recs):
    fig, ax = plt.subplots(figsize=(9, 6))
    d = np.arange(2, 42)
    A, f = fvd["fit"]["prefactor"], fvd["fit"]["fidelity_per_cycle"]
    ax.plot(d, A * f ** d, "k--", lw=1, label=f"device fit {A:.3f} x {f:.4f}^d")
    for fam, pts in fvd["points"].items():
        m, c = FAM_STYLE[fam]
        ds = sorted(int(k) for k in pts)
        ax.errorbar(ds, [pts[str(k)]["fidelity"] for k in ds], [pts[str(k)]["se"] for k in ds],
                    fmt=m, color=c, ms=5, capsize=2, label=f"ibm_phoenix {fam}")
    if hw_recs:
        acc = defaultdict(list)
        for r in hw_recs:
            acc[(r["K"], r["depth"])].append((r["fidelity"], r["se"]))
        for K in (3, 4):
            ks = sorted(k for k in acc if k[0] == K)
            if ks:
                F = [nh.inverse_variance_mean(*zip(*acc[k]))[0] for k in ks]
                ax.plot([k[1] for k in ks], F, FAM_STYLE[f"{K}-patch"][0], mfc="none", ms=11,
                        color=FAM_STYLE[f"{K}-patch"][1], label=f"PyQrack re-score {K}-patch (hwxeb)")
    floor = 1e-5
    for ci, (cfg, recs) in enumerate(sorted(data.items())):
        pm = point_means(recs, lay.n, fam_label)
        for fam in ("mirror", "3-patch", "4-patch"):
            ks = sorted(k for k in pm if k[0] == fam)
            if not ks:
                continue
            m, _ = FAM_STYLE[fam]
            F = [pm[k][0] for k in ks]
            se = [pm[k][1] for k in ks]
            ds_ = [k[1] for k in ks]
            col = f"C{3 + ci}"
            ax.errorbar(ds_, [logsafe(x, floor) for x in F], se, fmt=m + "-", color=col, ms=6, lw=1,
                        capsize=2, label=f"clean {fam} [{lab(cfg)}]")
            low = [(dd, x) for dd, x in zip(ds_, F) if x <= floor]
            if low:
                ax.plot([x for x, _ in low], [floor] * len(low), "v", color=col, ms=9)
    ax.set_yscale("log")
    ax.set_xlabel("depth d (cycles)")
    ax.set_ylabel("fidelity estimate")
    ax.set_title("1  Fidelity vs depth, 61 qubits (v = at or below plot floor)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    return fig


def fig_sweep(plt, lay, fvd, data):
    cfgs = [c for c, r in sorted(data.items()) if any(k[1] == "mirror" for k in r)]
    if not cfgs:
        return None
    fig, axes = plt.subplots(1, len(cfgs), figsize=(7 * len(cfgs), 5.5), squeeze=False)
    for ax, cfg in zip(axes[0], cfgs):
        recs = data[cfg]
        sizes = sorted({k[0] for k in recs if k[1] == "mirror"})
        cmap = plt.get_cmap("viridis")
        shots = [r["shots"] for k, r in recs.items() if k[1] == "mirror" and r.get("shots")]
        floor = 3.0 / min(shots) if shots else 1e-6
        for si, n in enumerate(sizes):
            pm = point_means(recs, n, fam_label)
            ks = sorted(k for k in pm if k[0] == "mirror")
            col = cmap(si / max(len(sizes) - 1, 1))
            ax.errorbar([k[1] for k in ks], [max(pm[k][0], floor / 3) for k in ks], [pm[k][1] for k in ks],
                        fmt="o-", color=col, ms=4, capsize=2, label=f"n={n}")
        hw = fvd["points"]["mirror"]
        ds = sorted(int(k) for k in hw)
        ax.plot(ds, [hw[str(k)]["fidelity"] for k in ds], "k:", lw=1, label="ibm_phoenix, n=61")
        ax.axhline(floor, color="r", ls="--", lw=0.8, label="3/shots (fit floor)")
        ax.set_yscale("log")
        ax.set_xlabel("depth d")
        ax.set_ylabel("mirror survival")
        ax.set_title(f"2  Mirror decay by register size [{lab(cfg)}]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    return fig


def fig_bN(plt, lay, fvd, data):
    rows_by = {}
    for cfg, recs in sorted(data.items()):
        sizes = sorted({k[0] for k in recs if k[1] == "mirror"})
        rows = []
        for n in sizes:
            pm = point_means(recs, n, fam_label)
            mpts = {k[1]: v[0] for k, v in pm.items() if k[0] == "mirror"}
            shots = [r["shots"] for k, r in recs.items() if k[0] == n and k[1] == "mirror" and r.get("shots")]
            b, A, dropped = nh._decay(mpts, 3.0 / min(shots) if shots else 0.0)
            if b is not None:
                cz = float(np.mean([nh.mirror_czpc(lay, n, d) for d in mpts]))
                rows.append((n, cz, b, dropped, sorted(mpts)))
        if rows:
            rows_by[cfg] = rows
    if not rows_by:
        return None
    b_dev = -math.log(fvd["fit"]["fidelity_per_cycle"])
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.axhline(b_dev, color="k", ls="--", lw=1, label=f"ibm_phoenix b = {b_dev:.3f} (per-cycle 0.872)")
    for ci, (cfg, rows) in enumerate(rows_by.items()):
        col = f"C{ci}"
        N = [r[0] for r in rows]
        b = [r[2] for r in rows]
        ax.plot(N, b, "o", color=col, label=f"b(N) [{lab(cfg)}]")
        for n, _, bb, dr, _ in rows:
            if dr:
                ax.annotate("*", (n, bb), color=col, fontsize=12)
        if len(rows) >= 2:
            X = np.array([[r[0], r[1]] for r in rows])
            y = np.array(b)
            (u, v), *_ = np.linalg.lstsq(X, y, rcond=None)
            if u < 0 or v < 0:
                best = None
                for col_i in (0, 1):
                    cj = max(float(X[:, col_i] @ y / (X[:, col_i] @ X[:, col_i])), 0.0)
                    err = float(np.sum((y - cj * X[:, col_i]) ** 2))
                    if best is None or err < best[0]:
                        best = (err, col_i, cj)
                u, v = (best[2], 0.0) if best[1] == 0 else (0.0, best[2])
            depths = sorted({d for r in rows for d in r[4]})
            grid = np.arange(min(N), lay.n + 1)
            cz = [float(np.mean([nh.mirror_czpc(lay, int(g), d) for d in depths])) for g in grid]
            fit = u * grid + v * np.array(cz)
            ax.plot(grid, fit, "-", color=col, lw=1, label=f"u*N + v*CZpc, u={u:.2e}, v={v:.2e}")
            ax.plot([lay.n], [fit[-1]], "*", color=col, ms=14,
                    label=f"b(61) = {fit[-1]:.3f} -> per-cycle {math.exp(-fit[-1]):.3f}")
    ax.set_xlabel("register size N (first-n truncation)")
    ax.set_ylabel("per-cycle decay b = -d ln F / dd")
    ax.set_title("3  Per-cycle decay of the simulator vs size (* = fit used shallow depths only)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    sec = ax.secondary_yaxis("right", functions=(lambda b: np.exp(-np.asarray(b)),
                                                 lambda f: -np.log(np.clip(np.asarray(f), 1e-12, None))))
    sec.set_ylabel("per-cycle fidelity e^-b")
    return fig


def grid_image(lay, n, values):
    R, C = lay.grid
    img = np.full((R, C), np.nan)
    for q in range(n):
        r, c = lay.pos[q]
        img[r, c] = values[q]
    return img


def fig_mirror_bits(plt, lay, data, base, want_n, want_d):
    for cfg, recs in sorted(data.items()):
        mk = [k for k in recs if k[1] == "mirror" and point_npz(base, cfg, k).exists()]
        if not mk:
            continue
        n = want_n if want_n in {k[0] for k in mk} else max(k[0] for k in mk)
        ks = sorted(k for k in mk if k[0] == n)
        depths = sorted({k[3] for k in ks})
        targets = [[int(c) for c in s[:n]] for s in lay.inputs]
        tint = [sum(b << q for q, b in enumerate(t)) for t in targets]
        ham, perq_err, twirl_surv, input_surv = {}, {}, defaultdict(list), defaultdict(lambda: np.zeros((2, 10)))
        for k in ks:
            d = k[3]
            with np.load(point_npz(base, cfg, k)) as z:
                for s in range(len(lay.inputs)):
                    if f"input{s}" not in z.files:
                        continue
                    x = z[f"input{s}"].astype(np.uint64)
                    tw = z[f"input{s}_twirl"]
                    if x.size == 0:
                        continue
                    diff = x ^ np.uint64(tint[s])
                    ham.setdefault(d, []).append(popcount64(diff))
                    e = bit_matrix(diff, n).sum(axis=0)
                    pe = perq_err.setdefault(d, [np.zeros(n), 0])
                    pe[0] += e
                    pe[1] += x.size
                    hit = diff == 0
                    for r in np.unique(tw):
                        twirl_surv[d].append(float(hit[tw == r].mean()))
                    input_surv[d][0, s] += hit.sum()
                    input_surv[d][1, s] += x.size
        fig, axs = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(f"4  Mirror bitstrings, n = {n} [{lab(cfg)}]")
        ax = axs[0, 0]
        cmap = plt.get_cmap("plasma")
        bins = np.arange(n + 2) - 0.5
        for di, d in enumerate(depths):
            h = np.concatenate(ham[d])
            ax.hist(h, bins=bins, density=True, histtype="step", lw=1.4,
                    color=cmap(di / max(len(depths) - 1, 1)), label=f"d={d} ({h.size:,} shots)")
        from math import comb
        kk = np.arange(n + 1)
        ax.plot(kk, [comb(n, int(x)) / 2 ** n for x in kk], "k:", lw=1, label="scrambled: Binomial(n, 1/2)")
        ax.set_ylim(bottom=0.3 / max(np.concatenate(ham[d]).size for d in depths), top=1.5)
        ax.set_xlabel("Hamming distance from the prepared string")
        ax.set_ylabel("fraction of shots")
        ax.set_yscale("log")
        ax.set_title("error weight per shot (0 = survived)")
        ax.legend(fontsize=7)
        ax = axs[0, 1]
        dsel = want_d if want_d in perq_err else next((d for d in depths if perq_err[d][0].sum() > 0), depths[0])
        rate = perq_err[dsel][0] / max(perq_err[dsel][1], 1)
        img = grid_image(lay, n, rate)
        im = ax.imshow(img, cmap="magma", vmin=0, vmax=max(0.5, np.nanmax(img)))
        for q in range(n):
            r, c = lay.pos[q]
            ax.text(c, r, str(q), ha="center", va="center", fontsize=6,
                    color="w" if rate[q] < 0.3 else "k")
        fig.colorbar(im, ax=ax, fraction=0.046, label="P(flip)")
        ax.set_title(f"per-qubit flip rate on the grid, d={dsel}")
        ax.set_xticks(range(lay.grid[1]))
        ax.set_yticks(range(lay.grid[0]))
        ax = axs[1, 0]
        try:
            ax.boxplot([twirl_surv[d] for d in depths], tick_labels=[str(d) for d in depths], showmeans=True)
        except TypeError:                           # matplotlib < 3.9
            ax.boxplot([twirl_surv[d] for d in depths], labels=[str(d) for d in depths], showmeans=True)
        ax.set_xlabel("depth d")
        ax.set_ylabel("survival of one twirl randomisation")
        ax.set_title("spread over Pauli-frame randomisations (inputs x instances pooled)")
        ax = axs[1, 1]
        for di, d in enumerate(depths):
            hs = input_surv[d]
            with np.errstate(invalid="ignore"):
                p = hs[0] / hs[1]
            ax.plot(range(10), p, "o-", color=cmap(di / max(len(depths) - 1, 1)), label=f"d={d}")
        ax.set_xticks(range(10))
        ax.set_xlabel("released input string s")
        ax.set_ylabel("survival")
        ax.set_title("survival per input string")
        ax.legend(fontsize=7)
        fig.tight_layout()
        yield cfg, fig


def fig_patched_full(plt, lay, data, base, hw_recs):
    for cfg, recs in sorted(data.items()):
        pk = sorted(k for k in recs if k[1] == "patched")
        fk = sorted(k for k in recs if k[1] == "full" and point_npz(base, cfg, k).exists())
        if not pk and not fk:
            continue
        fig, axs = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(f"5  Patched and full-circuit bitstrings, n = 61 [{lab(cfg)}]")
        ax = axs[0, 0]
        if pk:
            labels, x0 = [], 0
            for k in pk:
                pf = recs[k]["patch_fidelity"]
                for r, v in enumerate(pf):
                    ax.bar(x0 + r * 0.25, v, 0.25, color=f"C{r}", label=f"patch {r}" if x0 == 0 else None)
                labels.append((x0 + 0.25 * (len(pf) - 1) / 2, f"K{k[2]} d{k[3]}\np{k[4]} i{k[5]}"))
                x0 += 1.3
            ax.set_xticks([p for p, _ in labels])
            ax.set_xticklabels([t for _, t in labels], fontsize=6)
            ax.axhline(1, color="k", lw=0.6)
            ax.set_ylabel("normalised patch XEB, Eq. (1)")
            ax.set_title("per-patch XEB of the simulator's samples")
            ax.legend(fontsize=7)
        else:
            ax.set_axis_off()
        ax = axs[0, 1]
        if pk or hw_recs:
            for K, m in ((3, "s"), (4, "D")):
                kk = [k for k in pk if k[2] == K]
                if kk:
                    ax.errorbar([k[3] + 0.3 * (k[4] - 2) / 2 for k in kk], [recs[k]["fidelity"] for k in kk],
                                [recs[k]["se"] for k in kk], fmt=m, color=f"C{K}", ms=5, capsize=2,
                                label=f"clean, K={K}")
                hh = [r for r in hw_recs if r["K"] == K]
                if hh:
                    ax.errorbar([r["depth"] - 0.6 for r in hh], [r["fidelity"] for r in hh], [r["se"] for r in hh],
                                fmt=m, mfc="none", color="gray", ms=5, capsize=2, label=f"ibm_phoenix, K={K} (hwxeb)")
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_xlabel("depth d")
            ax.set_ylabel("patched XEB, Eq. (2)")
            ax.set_title("per-circuit patched XEB (partitions offset)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        else:
            ax.set_axis_off()
        if fk:
            depths = sorted({k[3] for k in fk})
            dsel = max(depths)
            x = []
            for k in fk:
                if k[3] == dsel:
                    with np.load(point_npz(base, cfg, k)) as z:
                        x.append(z["shots"].astype(np.uint64))
            x = np.concatenate(x)
            n = fk[0][0]
            p1 = bit_matrix(x, n).mean(axis=0)
            ax = axs[1, 0]
            dev = max(0.05, float(np.max(np.abs(p1 - 0.5))))
            im = ax.imshow(grid_image(lay, n, p1), cmap="coolwarm", vmin=0.5 - dev, vmax=0.5 + dev)
            for q in range(n):
                r, c = lay.pos[q]
                ax.text(c, r, f"{p1[q]:.2f}", ha="center", va="center", fontsize=6)
            fig.colorbar(im, ax=ax, fraction=0.046, label="P(1)")
            ax.set_title(f"full circuit d={dsel}: per-qubit P(1), {x.size:,} samples "
                         f"(se ~ {0.5 / math.sqrt(x.size):.3f})")
            ax = axs[1, 1]
            w = popcount64(x)
            from math import comb
            ax.hist(w, bins=np.arange(n + 2) - 0.5, density=True, histtype="stepfilled", alpha=0.6,
                    label=f"samples d={dsel}")
            kk = np.arange(n + 1)
            ax.plot(kk, [comb(n, int(v)) / 2 ** n for v in kk], "k:", label="uniform: Binomial(n, 1/2)")
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.3 / w.size, top=1)
            ax.set_xlabel("Hamming weight")
            ax.set_ylabel("fraction of samples")
            ax.set_title("Hamming-weight distribution (Porter-Thomas outputs look uniform here)")
            ax.legend(fontsize=7)
        else:
            axs[1, 0].set_axis_off()
            axs[1, 1].set_axis_off()
        fig.tight_layout()
        yield cfg, fig


def fig_release(plt, lay, fvd, hw_recs):
    ref = {(r["K"], r["depth"], r["partition"], r["instance"]): r
           for r in json.loads((lay.repo / "data/results/patch_xeb.json").read_text())}
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("6  Release data and the PyQrack re-score of the hardware counts (hwxeb)")
    ax = axs[0]
    for K in (3, 4):
        cr = fvd["collision_ratio"][f"K{K}"]
        ds = sorted(int(k) for k in cr)
        ax.plot(ds, [cr[str(d)] for d in ds], FAM_STYLE[f"{K}-patch"][0] + "-",
                color=FAM_STYLE[f"{K}-patch"][1], label=f"release K={K}")
        if hw_recs:
            acc = defaultdict(list)
            for r in hw_recs:
                if r["K"] == K:
                    acc[r["depth"]] += list(r["patch_ideal_xeb"])
            if acc:
                dd = sorted(acc)
                ax.plot(dd, [np.mean(acc[d]) for d in dd], FAM_STYLE[f"{K}-patch"][0], mfc="none", ms=12,
                        color=FAM_STYLE[f"{K}-patch"][1], label=f"PyQrack K={K}")
    ax.axhline(1, color="k", lw=0.6)
    ax.set_xlabel("depth d")
    ax.set_ylabel("collision ratio 2^n sum p^2 - 1")
    ax.set_title("anticoncentration (1 = Porter-Thomas)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    if hw_recs:
        pairs = [(ref[(r["K"], r["depth"], r["partition"], r["instance"])], r) for r in hw_recs
                 if (r["K"], r["depth"], r["partition"], r["instance"]) in ref]
        ax = axs[1]
        for K in (3, 4):
            pp = [(a["fidelity"], b["fidelity"]) for a, b in pairs if a["K"] == K]
            if pp:
                ax.plot(*zip(*pp), FAM_STYLE[f"{K}-patch"][0], color=FAM_STYLE[f"{K}-patch"][1], ms=4,
                        label=f"K={K}")
        lim = [min(a["fidelity"] for a, _ in pairs), max(a["fidelity"] for a, _ in pairs)]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel("release F (Aer)")
        ax.set_ylabel("PyQrack F")
        ax.set_title(f"per-circuit patched XEB, {len(pairs)} circuits")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax = axs[2]
        res = [(b["fidelity"] - a["fidelity"]) / max(a["se"], 1e-12) for a, b in pairs]
        ax.hist(res, bins=30)
        ax.set_xlabel("(F_PyQrack - F_release) / se")
        ax.set_title(f"residuals: max |r| = {max(abs(x) for x in res):.2e} sigma")
    else:
        for ax in axs[1:]:
            ax.text(0.5, 0.5, "no hwxeb output given\n(--hwxeb FILE)", ha="center", va="center")
            ax.set_axis_off()
    fig.tight_layout()
    return fig


def fig_cost(plt, data):
    fig, ax = plt.subplots(figsize=(8, 5))
    any_ = False
    for ci, (cfg, recs) in enumerate(sorted(data.items())):
        for fam, m in (("mirror", "o"), ("patched", "s"), ("full", "^"), ("fxeb", "P")):
            rr = [r for k, r in recs.items() if k[1] == fam and r.get("seconds") is not None]
            if rr:
                any_ = True
                sc = ax.scatter([r["n"] for r in rr], [r["seconds"] for r in rr], c=[r["depth"] for r in rr],
                                marker=m, cmap="viridis", s=25, edgecolors=f"C{ci}",
                                label=f"{fam} [{lab(cfg)}]")
    if not any_:
        plt.close(fig)
        return None
    fig.colorbar(sc, ax=ax, label="depth d")
    ax.set_yscale("log")
    ax.set_xlabel("register size n")
    ax.set_ylabel("seconds per point")
    ax.set_title("7  Run cost per point (plan worker counts from this)")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    return fig


# ================================================================== fxeb: harvest and comparison
def fxeb_points(recs):
    """{(n, d): stats} over instances: plain mean, standard error from the instance
    scatter (shot se when there is one instance), shot noise, spread, HOG, ideal XEB."""
    acc = defaultdict(list)
    for k, r in recs.items():
        if k[1] == "fxeb" and r.get("fidelity") is not None:
            acc[(k[0], k[3])].append(r)
    out = {}
    for key, v in acc.items():
        F = np.array([r["fidelity"] for r in v], float)
        se = np.array([r["se"] for r in v], float)
        k = len(v)
        spread = float(F.std(ddof=1)) if k > 1 else float("nan")
        shot = float(math.sqrt(np.sum(se ** 2)) / k)
        out[key] = dict(F=float(F.mean()), err=(spread / math.sqrt(k)) if k > 1 else shot, shot=shot, k=k,
                        spread=spread, hog=float(np.mean([r.get("hog", np.nan) for r in v])),
                        ideal=float(np.mean([r.get("ideal_xeb", np.nan) for r in v])), recs=v)
    return out


def fxeb_cfgs(data):
    """Configurations with fxeb points, restricted to the ensemble chosen with --theta
    (default haar): XEB values of different ensembles are not comparable."""
    return [c for c, r in sorted(data.items()) if any(k[1] == "fxeb" for k in r)
            and (THETA["show"] == "all" or ENSEMBLE.get(c, "haar") == THETA["show"])]


def cfg_colors(cfgs):
    return {c: f"C{i % 10}" for i, c in enumerate(cfgs)}


def fxeb_decay(pts, n, dmin):
    """Per-cycle decay b = -d ln F / dd at size n, weighted fit over depths >= dmin whose
    XEB is resolved (F > 2 x its error). Returns (b, b_err, depths used) or None."""
    good = sorted((d, p) for (nn, d), p in pts.items() if nn == n and d >= dmin and p["F"] > 2 * p["err"])
    if len(good) < 2:
        return None
    d = np.array([x for x, _ in good], float)
    F = np.array([p["F"] for _, p in good])
    w = (F / np.array([p["err"] for _, p in good])) ** 2
    W = np.diag(w)
    X = np.column_stack([np.ones_like(d), d])
    cov = np.linalg.inv(X.T @ W @ X)
    beta = cov @ X.T @ W @ np.log(F)
    return float(-beta[1]), float(math.sqrt(cov[1, 1])), [int(x) for x in d]


def fxeb_czpc(lay, n, depths):
    """CZ gates per cycle of the forward fxeb circuit on the first n qubits."""
    c = [sum(len(es) for _, es in nh.fxeb_cycles(lay, d, 0, n)) / d for d in depths]
    return float(np.mean(c))


def nonneg_fit(X, y):
    """y ~ X @ (u, v) with u, v >= 0 (best single term if the free fit goes negative)."""
    (u, v), *_ = np.linalg.lstsq(X, y, rcond=None)
    if u < 0 or v < 0:
        best = None
        for j in (0, 1):
            cj = max(float(X[:, j] @ y / (X[:, j] @ X[:, j])), 0.0)
            err = float(np.sum((y - cj * X[:, j]) ** 2))
            if best is None or err < best[0]:
                best = (err, j, cj)
        u, v = (best[2], 0.0) if best[1] == 0 else (0.0, best[2])
    return float(u), float(v)


def all_handles(axes):
    """Legend entries of every panel, each label once."""
    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    return list(seen.values()), list(seen.keys())


def fxeb_axes(ax, top):
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_ylim(top=max(top * 1.6, 0.05))
    ax.set_ylabel("forward XEB")
    ax.grid(True, which="both", alpha=0.3)


def fig_fxeb_depth(plt, lay, fvd, data):
    cfgs = fxeb_cfgs(data)
    if not cfgs:
        return None
    col = cfg_colors(cfgs)
    pts = {c: fxeb_points(data[c]) for c in cfgs}
    sizes = sorted({n for c in cfgs for n, _ in pts[c]})
    nc = min(4, len(sizes))
    nr = math.ceil(len(sizes) / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.9 * nr + 0.9), squeeze=False)
    A, f = fvd["fit"]["prefactor"], fvd["fit"]["fidelity_per_cycle"]
    dmax = max(d for c in cfgs for _, d in pts[c])
    dd = np.arange(2, dmax + 5)
    top = max(p["F"] + p["err"] for c in cfgs for p in pts[c].values())
    for ax, n in zip(axs.flat, sizes):
        for ci, c in enumerate(cfgs):
            ks = sorted(d for nn, d in pts[c] if nn == n)
            if not ks:
                continue
            P = [pts[c][(n, d)] for d in ks]
            off = (ci - (len(cfgs) - 1) / 2) * 0.15
            ax.errorbar([d + off for d in ks], [p["F"] for p in P], [p["err"] for p in P], fmt="o-",
                        color=col[c], ms=4, lw=1, capsize=2, label=lab(c))
        ax.plot(dd, A * f ** dd, "k--", lw=0.8, label="ibm_phoenix fit (61 q)")
        fxeb_axes(ax, top)
        ax.set_xlim(0, dmax + 4)
        ax.set_title(f"n = {n}", fontsize=10)
        ax.set_xlabel("depth d")
    for ax in list(axs.flat)[len(sizes):]:
        ax.set_axis_off()
    h, l = all_handles(axs.flat)
    fig.legend(h, l, loc="lower center", ncol=min(3, len(l)), fontsize=7)
    fig.suptitle("8  Forward XEB vs depth per register size (error bars: instance scatter)")
    fig.tight_layout(rect=(0, 0.04 + 0.025 * math.ceil(len(l) / 3), 1, 0.97))
    return fig


def fig_fxeb_size(plt, lay, fvd, data):
    cfgs = fxeb_cfgs(data)
    if not cfgs:
        return None
    col = cfg_colors(cfgs)
    pts = {c: fxeb_points(data[c]) for c in cfgs}
    depths = sorted({d for c in cfgs for _, d in pts[c]})
    top = max(p["F"] + p["err"] for c in cfgs for p in pts[c].values())
    fig, axs = plt.subplots(1, len(depths), figsize=(4.6 * len(depths), 4.8), squeeze=False)
    for ax, d in zip(axs[0], depths):
        for c in cfgs:
            ns = sorted(n for n, dd in pts[c] if dd == d)
            if ns:
                P = [pts[c][(n, d)] for n in ns]
                ax.errorbar(ns, [p["F"] for p in P], [p["err"] for p in P], fmt="o-", color=col[c], ms=4,
                            lw=1, capsize=2, label=lab(c))
        fxeb_axes(ax, top)
        ax.set_title(f"d = {d}", fontsize=10)
        ax.set_xlabel("register size n")
    h, l = all_handles(axs.flat)
    fig.legend(h, l, loc="lower center", ncol=min(3, len(l)), fontsize=7)
    fig.suptitle("9  Forward XEB vs register size per depth")
    fig.tight_layout(rect=(0, 0.05 + 0.03 * math.ceil(len(l) / 3), 1, 0.95))
    return fig


def fig_fxeb_bN(plt, lay, fvd, data, dmin):
    cfgs = fxeb_cfgs(data)
    rows = {}
    for c in cfgs:
        pts = fxeb_points(data[c])
        rr = []
        for n in sorted({n for n, _ in pts}):
            fit = fxeb_decay(pts, n, dmin)
            if fit:
                rr.append((n, fxeb_czpc(lay, n, fit[2]), *fit))
        if rr:
            rows[c] = rr
    if not rows:
        return None
    col = cfg_colors(cfgs)
    b_dev = -math.log(fvd["fit"]["fidelity_per_cycle"])
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.axhline(b_dev, color="k", ls="--", lw=1, label=f"ibm_phoenix b = {b_dev:.3f} (per-cycle 0.872)")
    for c, rr in rows.items():
        N = np.array([r[0] for r in rr], float)
        b = np.array([r[2] for r in rr])
        ax.errorbar(N, b, [r[3] for r in rr], fmt="o", color=col[c], capsize=2, label=f"b(N) {lab(c)}")
        if len(rr) >= 2:
            u, v = nonneg_fit(np.column_stack([N, [r[1] for r in rr]]), b)
            depths = sorted({d for r in rr for d in r[4]})
            grid = np.arange(int(N.min()), lay.n + 1)
            cz = np.array([fxeb_czpc(lay, int(g), depths) for g in grid])
            fit = u * grid + v * cz
            ax.plot(grid, fit, "-", color=col[c], lw=1)
            ax.plot([lay.n], [fit[-1]], "*", color=col[c], ms=14,
                    label=f"  -> b(61) {fit[-1]:.3f}, per-cycle {math.exp(-fit[-1]):.3f}"
                          f" ({'cleaner' if fit[-1] < b_dev else 'noisier'} than ibm_phoenix)")
    ax.set_xlabel("register size N (first-n truncation)")
    ax.set_ylabel("per-cycle decay b = -d ln XEB / dd")
    ax.set_title(f"10  Forward-XEB decay per cycle vs size, fit over resolved depths >= {dmin}, extrapolated to 61")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    sec = ax.secondary_yaxis("right", functions=(lambda x: np.exp(-np.asarray(x)),
                                                 lambda y: -np.log(np.clip(np.asarray(y), 1e-12, None))))
    sec.set_ylabel("per-cycle fidelity e^-b")
    return fig


_GEO = {}


def with_b2b(out, n):
    """Effective bulk-to-boundary ratio of the placed qubits: logical qubits on bulk sites
    over logical qubits on seam sites -- nn_qab.py's B-to-B ratio, counted over the sites
    the circuit actually occupies rather than over the whole register (no seam qubit used:
    the bulk count itself, as the finite upper end)."""
    s = out.get("seam_used")
    out["eff_b2b"] = None if s is None else ((n - s) / s if s else float(n))
    return out


def geometry_stats(lay, rec, n):
    """Coupler split and placement facts of the ACE configuration a record ran with, at
    size n: from the record where it has them, recomputed with nighthawk_qrack's own
    layout and tiling code otherwise. The recomputed placement is checked against the
    record's ace_map; if they differ (older tiling version), only the record's own
    counts are used."""
    key = (rec.get("ace_register"), rec.get("ace_lrc"), rec.get("ace_lrr"), bool(rec.get("ace_torus")),
           bool(rec.get("ace_tiling")), n, rec.get("ace_map"))
    if key in _GEO:
        return _GEO[key]
    reg, lrc, lrr, torus, tiled, _, rmap = key
    out = dict(exact=None, replica=None, cross=None, seam_used=None, sims_used=None,
               widest=(rec.get("ace_widths") or [None])[0], map_ok=None)
    if rec.get("ace_couplers"):
        out.update(rec["ace_couplers"])
    if rec.get("ace_seam_used") is not None:            # written by newer runs: nothing to recompute
        out.update(seam_used=rec["ace_seam_used"], sims_used=rec.get("ace_sims_used"), map_ok=True)
        _GEO[key] = with_b2b(out, n)
        return out
    if None not in (reg, lrc, lrr):
        try:
            opts = nh.ace_opts(argparse.Namespace(**{k: rec[k] for k in
                                                     ("ace_boundary_rep", "ace_error_detection", "ace_crossbars")
                                                     if k in rec}))
            row = nh.layout_row(reg, lrc, lrr, torus, opts)
            layout = "grid" if reg == lay.grid[0] * lay.grid[1] else "strip"
            # older tiled records carry no tiling version: try each until the placement hash matches
            versions = [rec["ace_tiling_version"]] if rec.get("ace_tiling_version") else [2, 3]
            for ver in versions if tiled else [None]:
                idx = nh.place(lay, n, argparse.Namespace(ace_layout=layout, ace_tiling=tiled,
                                                          ace_tiling_version=ver), row, reg)
                ok = rmap is None or hashlib.sha1(json.dumps(idx).encode()).hexdigest()[:10] == rmap
                if ok:
                    break
            out["map_ok"] = ok
            out["widest"] = row["max_width"]
            if ok:
                sids = [frozenset(s for s, _ in u) for u in row["unpack"]]
                ex, rp, cr = nh.coupler_classes(sids, idx, nh.logical_couplers(lay, n))
                out.update(exact=ex, replica=rp, cross=cr,
                           seam_used=sum(1 for s in idx if len(sids[s]) > 1),
                           sims_used=len(set().union(*[sids[s] for s in idx if len(sids[s]) == 1])))
        except Exception as err:                    # no PyQrack here: record values only
            out["error"] = str(err)
    _GEO[key] = with_b2b(out, n)
    return out


def rank_corr(x, y):
    """Spearman rank correlation (average ranks for ties)."""
    def ranks(v):
        v = np.asarray(v, float)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2
            i = j + 1
        return r
    if len(x) < 3:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


PREDICTORS = [("exact", "couplers inside one simulator"), ("replica", "couplers through a seam replica"),
              ("cross", "couplers across simulators"), ("seam_used", "logical qubits on seam sites"),
              ("sims_used", "simulators holding bulk qubits"), ("widest", "widest internal simulator"),
              ("eff_b2b", "effective bulk-to-boundary ratio (nn_qab B-to-B, placed qubits)")]


def pick_nd(data, want_n, want_d):
    """(n, d) for view 11: the requested pair, else the pair most ACE configurations
    share, preferring d >= 8 (shallower outputs are too concentrated to rank by)."""
    count = defaultdict(int)
    for c in fxeb_cfgs(data):
        if c.startswith("ace"):
            for key in fxeb_points(data[c]):
                count[key] += 1
    if not count:
        return None
    if want_n and want_d and (want_n, want_d) in count:
        return want_n, want_d
    cands = [k for k in count if (not want_n or k[0] == want_n) and (not want_d or k[1] == want_d)] or list(count)
    return max(cands, key=lambda k: (count[k], k[1] >= 8, -abs(k[1] - 8), k[0]))


def fig_fxeb_predict(plt, lay, data, want_n, want_d):
    nd = pick_nd(data, want_n, want_d)
    if nd is None:
        return None
    n, d = nd
    cfgs = [c for c in fxeb_cfgs(data) if c.startswith("ace")]
    col = cfg_colors(fxeb_cfgs(data))
    rows = []
    for c in cfgs:
        p = fxeb_points(data[c]).get((n, d))
        if p:
            rows.append((c, p, geometry_stats(lay, p["recs"][0], n)))
    if not rows:
        return None
    ncol = math.ceil(len(PREDICTORS) / 2)
    fig, axs = plt.subplots(2, ncol, figsize=(5 * ncol, 9), squeeze=False)
    for ax in list(axs.flat)[len(PREDICTORS):]:
        ax.set_axis_off()
    for ax, (key, title) in zip(axs.flat, PREDICTORS):
        xs, ys = [], []
        for c, p, g in rows:
            if g.get(key) is None:
                continue
            ax.errorbar(g[key], p["F"], p["err"], fmt="o", color=col[c], ms=7, capsize=3, label=lab(c))
            ax.annotate(LABELS.get(c, c).replace("ace ", ""), (g[key], p["F"]), fontsize=6,
                        xytext=(4, 3), textcoords="offset points")
            xs.append(g[key])
            ys.append(p["F"])
        rho = rank_corr(xs, ys)
        rtxt = f"rank corr {rho:+.2f}" if rho == rho else "rank corr n/a (needs 3+ distinct)"
        ax.set_title(f"{title}\n{rtxt} over {len(xs)} configs" if xs else f"{title}\n(no data)", fontsize=9)
        ax.set_xlabel(key)
        ax.set_ylabel(f"forward XEB, n={n} d={d}")
        ax.grid(True, alpha=0.3)
    stale = [lab(c) for c, _, g in rows if g.get("map_ok") is False]
    fig.suptitle(f"11  What predicts the XEB: ACE configurations at n = {n}, d = {d}"
                 + (f"\n(placement recomputed differently for: {', '.join(stale)} -- record counts only)"
                    if stale else ""), fontsize=11)
    fig.tight_layout()
    return fig


def harvest_rows(store):
    """One row per point of every configuration, with its readable label and the
    coupler split of ACE points."""
    rows = []
    for c, recs in sorted(store.data.items()):
        for k, r in sorted(recs.items(), key=lambda kv: kv[0]):
            n, fam, K, d, j, i = k
            row = dict(cfg=c, label=LABELS.get(c, c), family=fam, n=n, K=K, depth=d, partition=j, instance=i,
                       fidelity=r.get("fidelity"), se=r.get("se"), xeb_linear=r.get("xeb_linear"),
                       ideal_xeb=r.get("ideal_xeb"), hog=r.get("hog"), shots=r.get("shots"),
                       seconds=r.get("seconds"), lrc=r.get("ace_lrc"), lrr=r.get("ace_lrr"),
                       torus=r.get("ace_torus"), tiled=r.get("ace_tiling"), register=r.get("ace_register"))
            if c.startswith("ace") and fam == "fxeb":
                g = geometry_stats(store.lay, r, n)
                row.update({key: g.get(key) for key, _ in PREDICTORS})
            rows.append(row)
    return rows


def write_table(store, path):
    rows = harvest_rows(store)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} points of {len(store.data)} configurations to {path}")
    cfgs = fxeb_cfgs(store.data)
    if not cfgs:
        return
    pts = {c: fxeb_points(store.data[c]) for c in cfgs}
    keys = sorted({k for c in cfgs for k in pts[c]})
    width = max(len(LABELS.get(c, c)) for c in cfgs)
    print(f"\nforward XEB, mean over instances +/- instance-scatter se (k = instances); "
          f"ensemble: {THETA['show']} (--theta)")
    for n, d in keys:
        here = sorted(((pts[c][(n, d)], c) for c in cfgs if (n, d) in pts[c]), key=lambda x: -x[0]["F"])
        print(f"\nn = {n}, d = {d}")
        for p, c in here:
            print(f"  {LABELS.get(c, c):{width}s}  {p['F']:+.4f} +/- {p['err']:.4f}  (k={p['k']})")
    print(f"\nper-cycle decay b over resolved depths >= {store.a.fxeb_dmin} (ibm_phoenix: "
          f"{-math.log(store.fvd['fit']['fidelity_per_cycle']):.4f})")
    for c in cfgs:
        for n in sorted({n for n, _ in pts[c]}):
            fit = fxeb_decay(pts[c], n, store.a.fxeb_dmin)
            if fit:
                print(f"  {LABELS.get(c, c):{width}s}  n={n:<3} b {fit[0]:.4f} +/- {fit[1]:.4f}  depths {fit[2]}")


# ================================================================== figure factory
# ================================================================== hamming (views 12-13)
def hamming_points(data):
    """{(n, d, lam): dict(F, err, k, seam, patches)} over every configuration's hamming records
    (plain mean over circuits, error = instance scatter / sqrt(k))."""
    acc, geo = defaultdict(dict), {}
    for recs in data.values():
        for (n, fam, K, d, j, i), r in recs.items():
            if fam != "hamming" or r.get("fidelity") is None:
                continue
            lam = int(r.get("fold", K))
            acc[(n, d, lam)][i] = float(r["fidelity"])
            if "ace_seam_used" in r:
                geo[n] = (int(r["ace_seam_used"]), int(r.get("ace_sims_used", 2)))
    out = {}
    for (n, d, lam), v in acc.items():
        x = np.array(list(v.values()))
        err = float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0
        seam, P = geo.get(n, (0, 2))
        out[(n, d, lam)] = dict(F=float(x.mean()), err=err, k=len(x), seam=seam, patches=P)
    return out


def dan_of(n, seam, P, d):
    return nh.dan_xeb(n - seam, seam, P, d) if seam else float("nan")


def fig_hamming(plt, data):
    pts = hamming_points(data)
    if not pts:
        return None
    sizes = sorted({k[0] for k in pts})
    nc = min(4, len(sizes))
    nr = math.ceil(len(sizes) / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.6 * nr + 0.9), squeeze=False)
    mk = {1: ("o", "C0", 1.0), 3: ("s", "C1", 0.75), 5: ("D", "C2", 0.75)}
    dmax = max(k[1] for k in pts)
    for ax, n in zip(axs.flat, sizes):
        seam = next(pts[k]["seam"] for k in pts if k[0] == n)
        P = next(pts[k]["patches"] for k in pts if k[0] == n)
        for lam, (m, c, al) in mk.items():
            ks = sorted(k[1] for k in pts if k[0] == n and k[2] == lam)
            if not ks:
                continue
            off = {1: 0.0, 3: 0.18, 5: 0.36}[lam]
            ax.errorbar([d + off for d in ks], [pts[(n, d, lam)]["F"] for d in ks], [pts[(n, d, lam)]["err"] for d in ks],
                        fmt=m + ("-" if lam == 1 else ""), color=c, ms=4.5, lw=1.2, capsize=2, alpha=al,
                        label=f"$\\lambda$ = {lam}" + ("" if lam == 1 else " (folded)"))
        if seam:
            dd = np.linspace(2, dmax + 2, 60)
            ax.plot(dd, [dan_of(n, seam, P, d) for d in dd], "k--", lw=1.0, label="Dan's calculator at d")
            x = ((n - seam) / (seam * P)) ** nh.DAN_MODEL["power"]
            ax.set_title(f"n = {n}: {seam} seam qubits, {P} patches, x = {x:.2f}", fontsize=9.5)
        else:
            ax.set_title(f"n = {n}", fontsize=9.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(2, dmax + 2)
        ax.set_xlabel("depth d of U (mirror runs 2*d*$\\lambda$ layers)")
        ax.set_ylabel("signal 1 - 2h")
        ax.grid(True, alpha=0.3)
    for ax in list(axs.flat)[len(sizes):]:
        ax.set_axis_off()
    h, l = all_handles(axs.flat)
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8)
    fig.suptitle("12  Dan Strano's mirror Hamming signal vs his ACE calculator "
                 "(--families hamming --protocol nnqab; error bars: circuit scatter)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return fig


def fig_hamming_ratio(plt, data):
    pts = hamming_points(data)
    if not pts:
        return None
    sizes = sorted({k[0] for k in pts})
    depths = sorted({k[1] for k in pts})
    cmap = plt.get_cmap("viridis")
    col = {d: cmap(0.12 + 0.76 * i / max(1, len(depths) - 1)) for i, d in enumerate(depths)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for d in depths:
        ns = [n for n in sizes if (n, d, 1) in pts and pts[(n, d, 1)]["seam"]]
        dan = [dan_of(n, pts[(n, d, 1)]["seam"], pts[(n, d, 1)]["patches"], d) for n in ns]
        r = [pts[(n, d, 1)]["F"] / v for n, v in zip(ns, dan)]
        e = [pts[(n, d, 1)]["err"] / v for n, v in zip(ns, dan)]
        if ns:
            a1.errorbar(ns, r, e, fmt="o-", color=col[d], ms=4.5, lw=1.2, capsize=2, label=f"d = {d}")
    a1.axhline(1, color="k", lw=0.9)
    a1.axhspan(0.85, 1.15, color="gray", alpha=0.12, lw=0)
    for n in sizes:
        seam = next(pts[k]["seam"] for k in pts if k[0] == n)
        a1.annotate(f"{seam}", (n, 0.02), xycoords=("data", "axes fraction"), ha="center", fontsize=7, color="gray")
    a1.set_xlabel("register size n (gray numbers: seam qubits)")
    a1.set_ylabel("measured / Dan's calculator")
    a1.set_title("Measured signal / calculator, $\\lambda$ = 1 (band: +/-15 %)", fontsize=10)
    a1.grid(True, alpha=0.3)
    a1.legend(fontsize=8, ncol=2)
    for d in depths:
        xs, qs = [], []
        for n in sizes:
            s1 = pts.get((n, d, 1), {}).get("F")
            for lam in (3, 5):
                s = pts.get((n, d, lam), {}).get("F")
                if s1 and s and 0 < s < 1 and 0 < s1 <= 0.9:     # near 1 the log-log ratio is noise
                    xs.append(n + (0.25 if lam == 5 else -0.25))
                    qs.append(math.log(math.log(s) / math.log(s1)) / math.log(lam))
        if xs:
            a2.plot(xs, qs, "o", color=col[d], ms=4.5, label=f"d = {d}")
    a2.axhline(nh.DAN_MODEL["depth_power"], color="k", ls="--", lw=1.0)
    a2.text(sizes[0], nh.DAN_MODEL["depth_power"] + 0.04, "Dan's gain exponent 1.606", fontsize=8)
    a2.axhline(0, color="gray", lw=0.6)
    a2.set_ylim(-0.4, 1.9)
    a2.set_xlabel("register size n (left of n: from $\\lambda$ = 3, right: from $\\lambda$ = 5; points with s(1) > 0.9 left out)")
    a2.set_ylabel("effective folding exponent q")
    a2.set_title("Folding: s($\\lambda$) = s(1)^($\\lambda$^q)", fontsize=10)
    a2.grid(True, alpha=0.3)
    a2.legend(fontsize=8, ncol=2)
    fig.suptitle("13  Hamming ladder against Dan's calculator, and how folding scales ACE's loss")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# ================================================================== shor9 (views 14-16)
# Records of `nighthawk_qrack.py shor9 --out`: one per layout (standalone register: seam_leaders,
# patch_blocks, exact_reference) or one per unit (--register). A setup is what changes ACE's answer:
# the PyQrack version (QrackAceBackend is PyQrack's Python code), the boundary repetition code and
# OpenCL. Records written before `boundary_rep` was stored ran with PyQrack's own default, which is on
# from PyQrack 2.29.
SHOR_STATES = ("0", "1", "+", "-")
SHOR_LAYOUTS = ("seam_leaders", "patch_blocks")


def read_shor(paths, test="shor9"):
    """Every record of this test (shor9, or shor9_map: the register as placed) in these files, once each
    (the same file may be harvested and given)."""
    out, seen = [], set()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict) or r.get("test") != test:
                continue
            key = (r.get("when"), r.get("register"), r.get("unit"), r.get("layout"), r.get("code"), r.get("qrack"),
                   r.get("seconds"))
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
    return out


def shor_versions(r):
    import re
    q = r.get("qrack") or ""
    m1, m2 = re.search(r"pyqrack ([\w.?]+)", q), re.search(r"Qrack ([\w.?]+)", q)
    return (m1.group(1) if m1 else "?", m2.group(1) if m2 else "?")


def shor_brep(r):
    if "boundary_rep" in r:
        return bool(r["boundary_rep"])
    try:
        return tuple(int(x) for x in shor_versions(r)[0].split(".")[:2]) >= (2, 29)
    except ValueError:
        return False


def shor_setup(r):
    """(long label, short label) of a record's setup."""
    pv, lv = shor_versions(r)
    lng = f"pyqrack {pv}, Qrack {lv}" if r.get("qrack") else "versions not recorded"
    sht = f"py {pv}" if r.get("qrack") else "py ?"
    if shor_brep(r):
        lng, sht = lng + ", boundary code", sht + " brep"
        if r.get("code_authority") == "colocated":                   # nighthawk_qrack.py ghz --code-authority
            lng, sht = lng + " (co-located)", sht + " coloc"
        if r.get("stack_pass") in ("V", "C", "E") and r.get("code_schedule") is False:   # --code-schedule off
            lng, sht = lng + ", preparation order", sht + " prep"
    eng = engine_setup(r)                                              # nighthawk_qrack.py ghz --engine loom / bridge, --box
    if eng:
        lng, sht = lng + ", " + eng[0], sht + " " + eng[1]
    if r.get("stack_order") == "code-echo":                            # nighthawk_qrack.py ghz stack --order code-echo
        lng, sht = lng + ", code first", sht + " ce"
    l2 = r.get("layer2")
    if isinstance(l2, dict) and l2.get("inside_zz") == "stabilizer":   # nighthawk_qrack.py ghz --inside-zz stabilizer
        lng, sht = lng + ", inside ZZ by stabiliser", sht + " zzstab"
    if r.get("device") == "gpu":
        lng += ", OpenCL" + (" (sync)" if r.get("ocl_sync") else "")
        sht += " ocl"
    return lng, sht


def engine_setup(r):
    """(long, short) label of a nighthawk_qrack.py ghz record's seam engine, or None on ACE: the loom (hard cut or ghost,
    exact or ACE chips) or QrackBridge (max_bonds, truncation, lazy bonds), on chips or on the closed box."""
    lo = r.get("loom")
    if not isinstance(lo, dict):
        return None
    read = lo.get("read", "ghost")
    if read == "bridge":
        bp = lo.get("bridge") or {}
        lng = f"QrackBridge max_bonds {bp.get('max_bonds', '?')}"
        sht = f"qb{bp.get('max_bonds', '?')}"
        if bp.get("truncation", "greedy") != "greedy":
            lng, sht = lng + f" {bp['truncation']}", sht + bp["truncation"][0]
        if bp.get("lazy_bonds"):
            lng, sht = lng + f", lazy {bp['lazy_bonds']}", sht + f"l{bp['lazy_bonds']}"
        if lo.get("box"):
            lng, sht = lng + f", box {lo['box']['L']}x{lo['box']['L']}x{lo['box']['H']} ring", sht + " box"
        return lng, sht
    lng = "loom " + ("hard cut" if read == "cut" else "ghost band")
    sht = "loom" + ("cut" if read == "cut" else "ghost")
    if lo.get("backend") == "ace":
        lng, sht = lng + ", ACE chips", sht + "+ace"
    return lng, sht


def engine_record_bonds(r):
    """max_bonds of a bridge record, -1 for the loom's hard cut, None otherwise (x of view 22)."""
    lo = r.get("loom")
    if not isinstance(lo, dict):
        return None
    if lo.get("read") == "bridge":
        return (lo.get("bridge") or {}).get("max_bonds")
    return -1 if lo.get("read") == "cut" else None


def _mean_err(v):
    v = [float(x) for x in v if x is not None]
    if not v:
        return float("nan"), float("nan")
    return float(np.mean(v)), (float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0)


def _pool_dist(cs):
    """Distribution comparisons of several records pooled by their shot counts."""
    cs = [c for c in cs if c]
    if not cs:
        return None
    N = sum(c.get("shots", 1) for c in cs)
    w = lambda k: sum(c.get(k, 0.0) * c.get("shots", 1) for c in cs) / N
    top = defaultdict(float)
    dec = {}
    for c in cs:
        for t in c.get("top_wrong", []):
            top[t["pattern"]] += t["share"] * c.get("shots", 1)
            dec[t["pattern"]] = t.get("decodes")
    tw = [dict(pattern=p, share=v / N, decodes=dec[p]) for p, v in sorted(top.items(), key=lambda kv: -kv[1])[:3]]
    return dict(tvd=w("tvd"), tvd_floor=w("tvd_floor"), impossible=w("impossible"), shots=N,
                support=cs[0].get("support"), top_wrong=tw)


def shor_standalone(recs):
    """{(setup, m, layout): statistics} over the standalone-register records: per-backend logical Z / X
    (the |0_L>/|1_L> and |+_L>/|-_L> backends paired by index, as the script prints its spread), the mean
    with the backend-to-backend standard error, decoded and raw success per state, pooled distribution
    comparison per state, seconds; plus 'exact' per (setup, m): every exact-reference record at 1.0."""
    acc = defaultdict(lambda: dict(zs=[], xs=[], dec=defaultdict(list), raw=defaultdict(list),
                                   dist=defaultdict(list), secs=[], records=0, shots=0, label=""))
    exact = defaultdict(list)
    for r in recs:
        if r.get("register") or "size" not in r:
            continue
        lng, sht = shor_setup(r)
        m = int(r["size"])
        if r.get("layout") == "exact_reference":
            exact[(lng, m)].append(all(abs(float(r.get(f"decoded_{s}", 0)) - 1) < 1e-12 for s in SHOR_STATES))
            continue
        a = acc[(lng, m, r.get("layout"))]
        a["label"] = sht
        runs = r.get("runs") or {}
        if all(runs.get(s) for s in SHOR_STATES):
            a["zs"] += [(b0[1] + b1[1]) / 2 for b0, b1 in zip(runs["0"], runs["1"])]
            a["xs"] += [(bp[1] + bm[1]) / 2 for bp, bm in zip(runs["+"], runs["-"])]
        else:
            a["zs"].append(r["logical_z"])
            a["xs"].append(r["logical_x"])
        for s in SHOR_STATES:
            a["dec"][s].append(r.get(f"decoded_{s}"))
            a["raw"][s].append(r.get(f"raw_{s}"))
            a["dist"][s].append((r.get("distribution") or {}).get(s))
        a["secs"].append(r.get("seconds"))
        a["records"] += 1
        a["shots"] += int(r.get("shots", 0)) * max(1, int(r.get("repeats", 1)))
    out = {}
    for k, a in acc.items():
        z, ez = _mean_err(a["zs"])
        x, ex = _mean_err(a["xs"])
        out[k] = dict(z=z, ez=ez, x=x, ex=ex, zs=a["zs"], xs=a["xs"], label=a["label"], records=a["records"],
                      backends=len(a["xs"]), shots=a["shots"],
                      dec={s: _mean_err(v)[0] for s, v in a["dec"].items()},
                      raw={s: _mean_err(v)[0] for s, v in a["raw"].items()},
                      dist={s: _pool_dist(v) for s, v in a["dist"].items()},
                      seconds=_mean_err(a["secs"])[0],
                      exact=(all(exact[(k[0], k[1])]) if exact.get((k[0], k[1])) else None))
    return out


def shor_units(recs):
    """{(setup, register, m, place): statistics} over the --register records, pooled over runs. Units are
    matched by their square on the lattice (rows, columns, leaders), not their number: the packing can
    number the same squares differently from run to run."""
    acc = defaultdict(list)
    for r in recs:
        if r.get("register") and r.get("unit") is not None:
            rw, cl = r.get("rows") or [0, 0], r.get("cols") or [0, 0]
            place = f"r{rw[0]}-{rw[1]} c{cl[0]}-{cl[1]}" + ("" if r.get("leaders", "cols") == "cols" else " rows")
            acc[(shor_setup(r)[0], r["register"], int(r.get("size", 3)), place)].append(r)
    out = {}
    for k, rs in acc.items():
        cx = rs[0].get("cx_classes") or {}
        out[k] = dict(z=_mean_err([r["logical_z"] for r in rs])[0], x=_mean_err([r["logical_x"] for r in rs])[0],
                      rows=rs[0].get("rows"), cols=rs[0].get("cols"), blocks=rs[0].get("blocks"),
                      kind=rs[0].get("kind", "?"), kinds=sorted({r.get("kind", "?") for r in rs}), replica=int(cx.get("replica", 0)), cross=int(cx.get("cross", 0)),
                      exact_cx=int(cx.get("exact", 0)), regions=rs[0].get("regions", {}), runs=len(rs),
                      label=shor_setup(rs[0])[1],
                      dist={s: _pool_dist([(r.get("distribution") or {}).get(s) for r in rs]) for s in SHOR_STATES})
    return out


def shor_colors(plt, setups):
    cmap = plt.get_cmap("tab10")
    return {s: cmap(i % 10) for i, s in enumerate(sorted(setups))}


def fig_shor_size(plt, recs):
    st = shor_standalone(recs)
    if not st:
        return None
    setups = sorted({k[0] for k in st})
    col = shor_colors(plt, setups)
    sty = {"seam_leaders": ("o", "-"), "patch_blocks": ("s", "--")}
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2))
    rng = np.random.default_rng(1)
    combos = sorted({(k[0], k[2]) for k in st})
    for basis, ax in zip(("z", "x"), axs):
        for ci, (setup, layout) in enumerate(combos):
            ms = sorted(k[1] for k in st if k[0] == setup and k[2] == layout)
            off = (ci - (len(combos) - 1) / 2) * 0.07
            mk, ls = sty.get(layout, ("^", ":"))
            ys = [st[(setup, m, layout)][basis] for m in ms]
            es = [st[(setup, m, layout)]["e" + basis] for m in ms]
            ax.errorbar([m + off for m in ms], ys, es, fmt=mk + ls, color=col[setup], ms=6, lw=1.3, capsize=3,
                        label=f"{layout}, {setup}")
            for m in ms:                                    # every fresh backend, jittered
                v = st[(setup, m, layout)][basis + "s"]
                ax.plot(m + off + rng.uniform(-0.025, 0.025, len(v)), v, mk, color=col[setup], ms=2.5, alpha=0.3)
        ax.axhline(1.0, color="k", lw=0.8)
        ax.axhline(0.5, color="gray", ls=":", lw=1.0)
        ax.text(0.01, 0.505, "0.5: no logical information", transform=ax.get_yaxis_transform(), fontsize=7,
                color="gray", va="bottom")
        allm = sorted({k[1] for k in st})
        ax.set_xticks(allm)
        ax.set_xticklabels([f"Shor-{m * m}\nm = {m}" for m in allm], fontsize=8)
        ax.set_ylim(0.3, 1.05)
        ax.set_xlabel("code [[m^2, 1, m]] on an m x m two-patch ACE register")
        ax.set_ylabel(f"decoded logical-{basis.upper()} success")
        ax.set_title(f"logical {basis.upper()}  (" + ("|0_L>, |1_L>: H then Z, majority of block signs"
                     if basis == "z" else "|+_L>, |-_L>: Z, parity of block majorities") + ")", fontsize=9.5)
        ax.grid(True, alpha=0.3)
    h, l = all_handles(axs)
    fig.legend(h, l, loc="lower center", ncol=min(3, len(l)), fontsize=8)
    bad = sorted({f"Shor-{k[1] ** 2} ({k[0]})" for k, v in st.items() if v["exact"] is False})
    fig.suptitle("14  Shor's code on ACE: decoded logical success per code size (bars: backend-to-backend "
                 "standard error; small dots: single fresh backends)"
                 + (f"\nEXACT REFERENCE NOT 1.0: {', '.join(bad)}" if bad else ""), fontsize=10.5)
    fig.tight_layout(rect=(0, 0.05 + 0.025 * math.ceil(len(l) / 3), 1, 0.95))
    return fig


def fig_shor_dist(plt, recs):
    st = shor_standalone(recs)
    if not st:
        return None
    layouts = [L for L in SHOR_LAYOUTS if any(k[2] == L for k in st)]
    fig, axs = plt.subplots(2, len(layouts), figsize=(7.2 * len(layouts), 7.6), squeeze=False)
    cs = {"0": "C0", "1": "C9", "+": "C3", "-": "C1"}
    w = 0.19
    for j, layout in enumerate(layouts):
        keys = sorted((k for k in st if k[2] == layout), key=lambda k: (k[1], k[0]))
        xs = np.arange(len(keys))
        for i, s in enumerate(SHOR_STATES):
            dd = [st[k]["dist"].get(s) for k in keys]
            pos = xs + (i - 1.5) * w
            axs[0, j].bar(pos, [100 * d["impossible"] if d else np.nan for d in dd], w, color=cs[s], label=f"|{s}_L>")
            axs[1, j].bar(pos, [d["tvd"] if d else np.nan for d in dd], w, color=cs[s], label=f"|{s}_L>")
            axs[1, j].plot(pos, [d["tvd_floor"] if d else np.nan for d in dd], "_", color="k", ms=9, mew=1.6,
                           label="floor (exact samples)" if i == 0 else None)
        for ax in axs[:, j]:
            ax.set_xticks(xs)
            ax.set_xticklabels([f"Shor-{k[1] ** 2}\n{st[k]['label']}" for k in keys], fontsize=7.5)
            ax.grid(True, axis="y", alpha=0.3)
        axs[0, j].set_ylabel("outcomes the exact state never gives (%)")
        axs[0, j].set_title(f"{layout}: impossible outcomes (Z-basis |+-_L>: broken blocks; X-basis |01_L>: "
                            "wrong block parity)", fontsize=8.5)
        axs[1, j].set_ylabel("total variation distance to exact")
        axs[1, j].set_ylim(0, 1.05)
        axs[1, j].set_title(f"{layout}: distance to the exact distribution (tick: shot-noise floor)", fontsize=8.5)
    h, l = all_handles(axs.flat)
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8)
    fig.suptitle("15  Shor's code on ACE: each logical state's outcome histogram against the exact distribution "
                 "(py = PyQrack version, brep = boundary code, ocl = OpenCL)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return fig


def fig_shor_units(plt, recs):
    un = shor_units(recs)
    if not un:
        return None
    kinds = {"one patch": ("C2", "o"), "touches seam": ("C3", "s"), "two patches": ("C4", "D")}
    setups = sorted({k[0] for k in un})
    fills = {s: ("full" if i == 0 else "none") for i, s in enumerate(setups)}
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16.5, 5.4))
    rng = np.random.default_rng(2)
    for k, u in sorted(un.items()):
        setup, reg, m, unit = k
        c, mk = kinds.get(u["kind"], ("C7", "^"))
        xv = u["replica"] + u["cross"] + rng.uniform(-0.12, 0.12)
        fs = fills[setup]
        for ax, y in ((a1, u["z"]), (a2, u["x"])):
            ax.plot(xv, y, mk, color=c, ms=6.5, fillstyle=fs, alpha=0.85)
            if y < 0.97:
                ax.annotate(f"{reg} {unit}", (xv, y), fontsize=6, color="gray", xytext=(3, -3),
                            textcoords="offset points")
        imp = [u["dist"][s]["impossible"] for s in ("+", "-") if u["dist"].get(s)]
        if imp:
            a3.plot(xv, 100 * float(np.mean(imp)), mk, color=c, ms=6.5, fillstyle=fs, alpha=0.85)
    for ax, t in ((a1, "decoded logical Z"), (a2, "decoded logical X")):
        ax.set_ylim(0.3, 1.04)
        ax.axhline(0.5, color="gray", ls=":", lw=1.0)
        ax.set_ylabel(t)
    a3.set_ylabel("Z-basis readout (|+-_L>): impossible outcomes (%)")
    from matplotlib.ticker import MaxNLocator
    for ax in (a1, a2, a3):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("encoder CXs through the seam (replica + cross)")
        ax.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=c, marker=mk, ls="", label=kd) for kd, (c, mk) in kinds.items()
          if any(u["kind"] == kd for u in un.values())]
    hs += [Line2D([], [], color="k", marker="o", ls="", fillstyle=fills[s], label=s) for s in setups]
    fig.legend(handles=hs, loc="lower center", ncol=len(hs), fontsize=8)
    regs = sorted({(k[1], k[2]) for k in un})
    fig.suptitle("16  Shor's code tiled over registers (" + ", ".join(f"{r} Shor-{m * m}" for r, m in regs)
                 + "): every unit against its seam load; one-patch units off zero = ACE's whole-block flips",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    return fig


_SHOR_MAPS = {}


def shor_register_map(store, reg):
    """The register as ACE placed it for shor9 (sites, regions, coupler classes): the newest shor9_map record,
    else rebuilt with nighthawk_qrack.py (needs the release and PyQrack; cached), else None."""
    recs = [r for r in store.shor_maps if r.get("register") == reg]
    if recs:
        return max(recs, key=lambda r: r.get("when", ""))
    if reg not in _SHOR_MAPS:
        try:
            nh.load_register_files()
            lay = nh.Layout(store.a.repo) if reg == "release" else nh.RegisterLayout(store.a.repo, reg)
            ra = nh.shor9_run_args(reg, store.a.repo)
            nh.apply_register_preset(ra, lay)
            _SHOR_MAPS[reg] = nh.shor9_register_map(lay, ra)[0]
        except (SystemExit, Exception) as err:          # no release / PyQrack / register file here
            print(f"# view 17: no map of {reg} ({str(err).splitlines()[0] if str(err) else type(err).__name__}); "
                  "units drawn on the bare lattice")
            _SHOR_MAPS[reg] = None
    return _SHOR_MAPS[reg]


SEAM_COLOR = "#7b3294"


def draw_seam_line(ax, sites):
    """The seams as bands: seam qubits joined to their lattice neighbours (diagonals too, since a seam
    steps across the square lattice) that join the same patches, each seam labelled with those patches
    (ACE's crossbar simulator, which every seam qubit also lives in, left out).
    Maps written before the joined patches were stored draw one band through every adjacent seam qubit."""
    # a seam qubit is held by the patches it joins and by ACE's shared crossbar simulator, which holds no
    # bulk qubit: only bulk-holding simulators (patches) name the seam
    sites = [st for st in sites if isinstance(st[3], str)]       # loom / bridge / box maps: chips, not ACE regions
    bulk = {int(st[3][1:]) for st in sites if st[3] != "seam" and st[3][1:].isdigit()}
    seam = [(int(st[1]), int(st[2]), tuple(sorted(x for x in st[4] if x in bulk)) if len(st) > 4 else None)
            for st in sites if st[3] == "seam"]
    if not seam:
        return
    same = lambda a, b: a[2] is None or b[2] is None or a[2] == b[2]
    parent = list(range(len(seam)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(seam)):
        for j in range(i + 1, len(seam)):
            a, b = seam[i], seam[j]
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 and same(a, b):
                ax.plot([a[1], b[1]], [a[0], b[0]], "-", color=SEAM_COLOR, lw=9, alpha=0.28, solid_capstyle="round",
                        zorder=0.5)
                parent[find(i)] = find(j)
    for r, c, _ in seam:
        ax.plot(c, r, "o", ms=14, color=SEAM_COLOR, alpha=0.28, mew=0, zorder=0.5)
    comps = defaultdict(list)
    for i, st in enumerate(seam):
        comps[find(i)].append(st)
    best = {}                                                       # one label per patch pair: its longest stretch
    for members in comps.values():
        joins = tuple(sorted({x for m_ in members if m_[2] for x in m_[2]}))
        if joins and len(members) > len(best.get(joins, [])):
            best[joins] = members
    for joins, members in best.items():
        r, c = max(members, key=lambda m_: (m_[0], -m_[1]))[:2]              # label at the seam's lower end
        ax.text(c + 0.32, r + 0.38, "|".join(f"p{x}" for x in joins), fontsize=7, color=SEAM_COLOR, zorder=6,
                fontweight="bold")


def fig_shor_grid(plt, store):
    """The registers on the Nighthawk lattice with their Shor units: qubits by ACE patch, seam qubits, couplers
    by class, each unit's square filled by its weaker decoded logical basis and labelled with Z, X and the
    Z-basis impossible share."""
    un = shor_units(store.shor)
    if not un:
        return None
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    panels = sorted({(k[1], k[2], k[0]) for k in un})
    nc = min(3, len(panels))
    nr = math.ceil(len(panels) / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(6.0 * nc, 6.9 * nr + 0.8), squeeze=False)
    cmap, norm = plt.get_cmap("RdYlGn"), Normalize(0.5, 1.0)
    pal = plt.get_cmap("Pastel1")
    ccol = {"exact": ("#bdbdbd", 1.0), "replica": ("C1", 2.0), "cross": ("C3", 2.4)}
    kcol = {"one patch": "C2", "touches seam": "C3", "two patches": "C4"}
    for ax, (reg, m, setup) in zip(axs.flat, panels):
        units = {k: u for k, u in un.items() if k[1] == reg and k[2] == m and k[0] == setup}
        mp = shor_register_map(store, reg)
        C = (mp or {}).get("columns", 10)
        pos = {int(st[0]): (int(st[1]), int(st[2])) for st in (mp or {}).get("sites", [])}
        R = max([12] + [r + 1 for r, _ in pos.values()] + [int(u["rows"][1]) + 1 for u in units.values() if u["rows"]])
        gr, gc = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
        ax.plot(gc.ravel(), gr.ravel(), ".", color="#dddddd", ms=3, zorder=0)       # the lattice
        if mp:
            for a_, b_, cl in mp["couplers"]:
                (r1, c1), (r2, c2) = pos[int(a_)], pos[int(b_)]
                col, lw = ccol.get(cl, ("k", 1.0))
                ax.plot([c1, c2], [r1, r2], "-", color=col, lw=lw, zorder=1)
            regs = sorted({st[3] for st in mp["sites"] if st[3] != "seam"},
                          key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
            pcol = {rg: pal(i % 9) for i, rg in enumerate(regs)}
            draw_seam_line(ax, mp["sites"])
            for q, r, c, rg, *_ in mp["sites"]:
                if rg == "seam":
                    ax.plot(c, r, "D", ms=7, mfc="white", mec="k", mew=1.4, zorder=3)
                else:
                    ax.plot(c, r, "o", ms=7.5, mfc=pcol[rg], mec="#777777", mew=0.6, zorder=3)
            for rg in regs:                                                         # patch names at their centre
                pts = np.array([(st[1], st[2]) for st in mp["sites"] if st[3] == rg], dtype=float)
                ax.text(pts[:, 1].mean(), pts[:, 0].mean(), rg, fontsize=8, color="#555555", ha="center",
                        va="center", zorder=2, alpha=0.8)
        zs, xs = [], []
        for (_, _, _, place), u in sorted(units.items()):
            if not u["rows"] or not u["cols"]:
                continue
            r0, c0 = int(u["rows"][0]), int(u["cols"][0])
            v = min(u["z"], u["x"])
            zs.append(u["z"])
            xs.append(u["x"])
            ax.add_patch(Rectangle((c0 - 0.42, r0 - 0.42), m - 0.16, m - 0.16, facecolor=cmap(norm(v)), alpha=0.42,
                                   edgecolor=kcol.get(u["kind"], "k"), lw=2.2, zorder=2))
            imp = [u["dist"][s_]["impossible"] for s_ in ("+", "-") if u["dist"].get(s_)]
            txt = f"Z {u['z']:.3f}\nX {u['x']:.3f}" + (f"\nimp {100 * float(np.mean(imp)):.1f}%" if imp else "")
            ax.text(c0 + (m - 1) / 2, r0 + (m - 1) / 2, txt, ha="center", va="center", fontsize=7.5 if m == 3 else 8.5,
                    zorder=5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
            if pos and u.get("blocks"):
                for b in u["blocks"]:                                               # block leaders
                    if int(b[0]) in pos:
                        r, c = pos[int(b[0])]
                        ax.plot(c, r, "o", ms=12, mfc="none", mec="k", mew=1.2, zorder=4)
        ax.set_xlim(-0.7, C - 0.3)
        ax.set_ylim(R - 0.3, -0.7)
        ax.set_aspect("equal")
        ax.set_xticks(range(C))
        ax.set_yticks(range(R))
        ax.tick_params(labelsize=7)
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        ttl = f"{reg}, Shor-{m * m}: {len(zs)} units"
        if zs:
            ttl += f", mean Z {np.mean(zs):.3f} / X {np.mean(xs):.3f}"
        ax.set_title(ttl + f"\n{setup}" + ("" if mp else "  (no register map: bare lattice)"), fontsize=8.5)
    for ax in list(axs.flat)[len(panels):]:
        ax.set_axis_off()
    hs = [Line2D([], [], color=ccol[k][0], lw=ccol[k][1], label=f"coupler {k}") for k in ("exact", "replica", "cross")]
    hs += [Line2D([], [], marker="o", ls="", mfc=pal(0), mec="#777777", ms=7, label="qubit (colour: ACE patch)"),
           Line2D([], [], marker="D", ls="", mfc="white", mec="k", ms=7, label="seam qubit"),
           Line2D([], [], color=SEAM_COLOR, lw=8, alpha=0.3, solid_capstyle="round", label="seam line (patches joined)"),
           Line2D([], [], marker="o", ls="", mfc="none", mec="k", ms=11, label="block leader")]
    hs += [Line2D([], [], color=c, lw=2.2, label=f"unit {k}") for k, c in kcol.items()
           if any(u["kind"] == k for u in un.values())]
    fig.legend(handles=hs, loc="lower center", ncol=min(5, len(hs)), fontsize=8)
    fig.suptitle("17  Shor units on the Nighthawk lattice: ACE patches, seam, coupler classes, and each unit's "
                 "measured logical Z / X (imp: Z-basis outcomes the exact state never gives; single-bit ones are "
                 "corrected by the inner code)", fontsize=10.5)
    fig.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.06 + 0.03 * math.ceil(len(hs) / 5))
    cax = fig.add_axes([0.91, 0.3, 0.014, 0.45])
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="unit fill: weaker decoded logical basis, min(Z, X)")
    return fig


def shor_rows(store):
    rows = []
    for (setup, m, layout), v in sorted(shor_standalone(store.shor).items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        row = dict(kind="standalone", setup=setup, code=f"shor{m * m}", size=m, layout=layout, register="",
                   unit="", records=v["records"], backends=v["backends"], logical_z=v["z"], err_z=v["ez"],
                   logical_x=v["x"], err_x=v["ex"], exact_ok=v["exact"], seconds=v["seconds"])
        for s in SHOR_STATES:
            d = v["dist"].get(s) or {}
            row.update({f"decoded_{s}": v["dec"].get(s), f"raw_{s}": v["raw"].get(s), f"tvd_{s}": d.get("tvd"),
                        f"floor_{s}": d.get("tvd_floor"), f"impossible_{s}": d.get("impossible"),
                        f"top_wrong_{s}": "; ".join(f"{t['pattern']} {100 * t['share']:.2f}%" for t in d.get("top_wrong", []))})
        rows.append(row)
    for (setup, reg, m, unit), u in sorted(shor_units(store.shor).items()):
        row = dict(kind="unit", setup=setup, code=f"shor{m * m}", size=m, layout=u["kind"], register=reg, unit=unit,
                   kinds_seen="/".join(u["kinds"]),
                   records=u["runs"], logical_z=u["z"], logical_x=u["x"], cx_exact=u["exact_cx"],
                   cx_replica=u["replica"], cx_cross=u["cross"])
        for s in SHOR_STATES:
            d = u["dist"].get(s) or {}
            row.update({f"tvd_{s}": d.get("tvd"), f"floor_{s}": d.get("tvd_floor"), f"impossible_{s}": d.get("impossible"),
                        f"top_wrong_{s}": "; ".join(f"{t['pattern']} {100 * t['share']:.2f}%" for t in d.get("top_wrong", []))})
        rows.append(row)
    return rows


def write_shor_table(store, path):
    rows = shor_rows(store)
    if not rows:
        return False
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {len(rows)} shor9 rows to {path}")
    print_shor_summary(store)
    return True


def print_shor_summary(store):
    st = shor_standalone(store.shor)
    if st:
        print("\nshor9, standalone register: decoded logical Z / X (+/- backend-to-backend s.e.), backends, exact reference")
        for (setup, m, layout), v in sorted(st.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
            ex = {True: "1.000", False: "NOT 1.0", None: "-"}[v["exact"]]
            print(f"  Shor-{m * m:<3} {layout:<13} {v['z']:.3f} +/- {v['ez']:.3f} / {v['x']:.3f} +/- {v['ex']:.3f}"
                  f"  {v['backends']:>4}  exact {ex}  [{setup}]")
    un = shor_units(store.shor)
    if un:
        print("\nshor9 units on registers: decoded logical Z / X, encoder CXs exact/replica/cross")
        for (setup, reg, m, unit), u in sorted(un.items()):
            print(f"  {reg:<14} Shor-{m * m:<3} {unit:<13} {u['kind']:<13} {u['z']:.3f} / {u['x']:.3f}  "
                  f"{u['exact_cx']}/{u['replica']}/{u['cross']}  [{setup}]")


# ================================================================== ghz (views 18-21)
# Records of `nighthawk_qrack.py ghz chip|tiles --out`: one ghz per state (chip: per tree; tiles: per unit) and one
# ghz_map per run (the register as placed, the shor9_map format plus the sublattice). The formal proof's Layer 1a
# numbers in the reference implementation's names (P0, P1, C_parity, Phi_coh, F_GHZ, lambda_per_block), the
# same prefixed exact_ from the exact reference, and `layer2` (mode none / unmitigated / compensated / dd).
# A unit's fidelity is F_GHZ (Lemma 4, MQC) where it was measured, else the Clifford witness F_witness.
GHZ_MODES = ("none", "unmitigated", "compensated", "dd")
GHZ_MODE_LABEL = {"none": "no Layer 2", "unmitigated": "unmitigated", "compensated": "compensated (oracle)",
                  "dd": "dd (Walsh echo)"}
GHZ_MODE_COLOR = {"none": "#7f7f7f", "unmitigated": "C3", "compensated": "C1", "dd": "C0"}


def read_ghz(paths, test="ghz"):
    """Every ghz (or ghz_map) record in these files, once each (the same file may be harvested and given). Layer V's
    control records of nighthawk_qrack.py ghz stack (pass 2's circuit rerun without the code) are left out: they repeat
    pass 2; so are --order code-echo's pass C control records (they repeat pass 1). Its corrected records (boundary_rep true) stay, under their own setup (", boundary code")."""
    out, seen = [], set()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict) or r.get("test") != test:
                continue
            if r.get("stack_pass") in ("V", "C") and not r.get("boundary_rep"):
                continue
            key = (r.get("when"), r.get("register"), r.get("mode"), r.get("tiling"), r.get("unit"), r.get("tree"),
                   ghz_mode(r), r.get("layers"), r.get("qrack"), tuple(r.get("qubits") or ()), r.get("stack_pass"),
                   r.get("boundary_rep"), r.get("code_authority"), r.get("code_schedule"),
                   (r.get("layer2") or {}).get("inside_zz") if isinstance(r.get("layer2"), dict) else None)
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
    return out


def ghz_mode(r):
    l2 = r.get("layer2")
    return (l2.get("mode", "none") if isinstance(l2, dict) else (l2 or "none")) if l2 is not None else "none"


def ghz_F(r, prefix=""):
    """(name, value) of a record's fidelity: F_GHZ (Lemma 4) where measured, else the witness bound. Records of
    nighthawk_qrack.py ghz's first version name them f_exact (the fidelity to GHZ_0, from k angles) and f_witness."""
    for k in ("F_GHZ", "F_witness", "f_exact", "f_witness"):
        v = r.get(prefix + k)
        if v is not None:
            return k, float(v)
    return None, None


def ghz_chips_of(r):
    """The chips (loom / bridge) or faces (box) a unit's qubits sit on, from the record's loom map."""
    lo = r.get("loom") or {}
    ends = list(np.cumsum(lo.get("chip_n") or []))
    if not ends:
        return set()
    return {next(i for i, e in enumerate(ends) if int(q) < e) for q in r.get("qubits", [])}


def ghz_place(r):
    """(sort key, label) of a tile's placement: off the seam, or holding a seam qubit with k CXs through it; with
    nighthawk_qrack.py ghz's seam engines: inside one chip (or box face), or across the loom / the bridge (on the box:
    a direct ring cut, or relayed through a face, and whether the ring's wrap cut is among them)."""
    lo = r.get("loom")
    if isinstance(lo, dict):
        bx = r.get("box")
        if bx is not None and lo.get("box"):
            n_ = len(bx.get("cuts", []))
            if n_ == 0:
                return (0, 0), "inside a face"
            wrap = len(lo["box"].get("ring", [])) - 1 in bx.get("cuts", [])
            return (2, n_ + (0.5 if wrap else 0)), ("across a box edge, " + ("direct" if bx.get("hops", n_) <= 1 else
                                                                               "relayed") + (", wrap cut" if wrap else ""))
        chips = ghz_chips_of(r)
        if len(chips) > 1:
            read = lo.get("read", "ghost")
            return (2, 0), ("across the bridge" if read == "bridge" else
                            "across the loom (hard cut)" if read == "cut" else "across the loom (ghost band)")
        if lo.get("backend") != "ace":
            return (0, 0), "inside a chip"
    d = r.get("seam_distance")
    if d is None:
        return (-1, 0), "no seam"
    if d >= 1:
        return (0, 0), "off the seam"
    c = r.get("cx_classes") or {}
    k = int(c.get("replica", 0)) + int(c.get("cross", 0))
    return (1, k), f"on the seam, {k} seam CX" + ("" if k == 1 else "s")


def ghz_tiles(recs):
    """{(register, unit size, layers, setup): {mode: [records]}} of the tiles records."""
    out = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r.get("mode") == "tiles":
            out[(r["register"], int(r.get("unit_size", 0)), r.get("layers", "none"), shor_setup(r)[0])][ghz_mode(r)].append(r)
    return out


def ghz_chips(recs):
    """{(register, tree, mode, layers, setup): [records, newest last]} of the chip records."""
    out = defaultdict(list)
    for r in recs:
        if r.get("mode") == "chip":
            out[(r["register"], r.get("tree", "?"), ghz_mode(r), r.get("layers", "none"), shor_setup(r)[0])].append(r)
    for v in out.values():
        v.sort(key=lambda r: r.get("when", ""))
    return out


def ghz_register_map(store, reg):
    """The register as nighthawk_qrack.py ghz placed it: the newest ghz_map record, else the shor9 map (same
    placement: the register's ACE preset), else None."""
    recs = [r for r in store.ghz_maps if r.get("register") == reg]
    if recs:
        return max(recs, key=lambda r: r.get("when", ""))
    return shor_register_map(store, reg)


def _ghz_lattice(ax, mp, R=None):
    """The register's couplers by class, seam bands and an empty lattice; returns {qubit: (row, col)}."""
    ccol = {"exact": ("#bdbdbd", 1.0), "replica": ("C1", 2.0), "cross": ("C3", 2.4)}
    pos = {int(st[0]): (int(st[1]), int(st[2])) for st in (mp or {}).get("sites", [])}
    rows = [r for r, _ in pos.values()] or [0]
    cols = [c for _, c in pos.values()] or [0]
    lo = (mp or {}).get("loom") or {}
    links = {frozenset(map(int, x)) for x in lo.get("links", [])}
    for a_, b_, cl in (mp or {}).get("couplers", []):
        (r1, c1), (r2, c2) = pos[int(a_)], pos[int(b_)]
        if lo.get("box") and abs(r1 - r2) + abs(c1 - c2) != 1:
            continue                                   # a box edge the unfolded net cuts open: not drawable
        col, lw = ccol.get(cl, ("k", 1.0))
        if frozenset((int(a_), int(b_))) in links:     # a loom link or a bridged coupler between chips / faces
            col, lw = (LINK_COLOR, 3.2)
        ax.plot([c1, c2], [r1, r2], "-", color=col, lw=lw, zorder=1)
    if mp:
        draw_seam_line(ax, mp["sites"])
    ax.set_xlim(min(cols) - 0.7, max(cols) + 0.7)
    ax.set_ylim(max(rows) + 0.7, min(rows) - 0.7)
    ax.set_aspect("equal")
    ax.set_xticks(range(min(cols), max(cols) + 1))
    ax.set_yticks(range(min(rows), max(rows) + 1))
    ax.tick_params(labelsize=7)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    return pos


LINK_COLOR = "#1d1d1b"


def _ghz_seam(mp):
    """Seam qubits of a register map: the ACE sites marked seam, or the map's own seam list (loom maps)."""
    out = {int(st[0]) for st in (mp or {}).get("sites", []) if st[3] == "seam"}
    return out | {int(q) for q in (mp or {}).get("seam", [])}


def _ghz_legend(fig, extra=()):
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color="#bdbdbd", lw=1.0, label="coupler exact (inside one simulator)"),
          Line2D([], [], color="C1", lw=2.0, label="coupler replica (through a seam)"),
          Line2D([], [], color="C3", lw=2.4, label="coupler cross"),
          Line2D([], [], marker="D", ls="", mfc="white", mec="k", ms=7, label="seam qubit"),
          Line2D([], [], color=SEAM_COLOR, lw=8, alpha=0.3, solid_capstyle="round", label="seam line (patches joined)"),
          Line2D([], [], color=LINK_COLOR, lw=3.2, label="link between chips (loom / bridge)")]
    fig.legend(handles=hs + list(extra), loc="lower center", ncol=min(6, len(hs) + len(extra)), fontsize=8)


def fig_ghz_map(plt, store):
    """18: tiles on the lattice: per site the mean fidelity of the units holding it, per register, unit size,
    Layer 2 mode and setup."""
    tl = ghz_tiles(store.ghz)
    if not tl:
        return None
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    panels = [(key, m) for key, by in sorted(tl.items()) for m in GHZ_MODES if by.get(m)]
    nc = min(4, len(panels))
    nr = math.ceil(len(panels) / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(5.2 * nc, 5.6 * nr + 0.8), squeeze=False)
    cmap, norm = plt.get_cmap("RdYlGn"), Normalize(0.5, 1.0)
    for ax, ((reg, k, layers, setup), m) in zip(axs.flat, panels):
        rs = tl[(reg, k, layers, setup)][m]
        mp = ghz_register_map(store, reg)
        pos = _ghz_lattice(ax, mp)
        seam = _ghz_seam(mp)
        per_q = defaultdict(list)
        for r in rs:
            for q in r.get("qubits", []):
                per_q[int(q)].append(ghz_F(r)[1])
        key = ghz_F(rs[0])[0]
        for q, v in per_q.items():
            if q not in pos:
                continue
            r_, c_ = pos[q]
            val = float(np.mean(v))
            ax.plot(c_, r_, "D" if q in seam else "o", ms=13 if q in seam else 14, mfc=cmap(norm(val)),
                    mec="k" if q in seam else "#777777", mew=1.4 if q in seam else 0.6, zorder=3)
            ax.text(c_, r_, f"{min(val, 1.0):.2f}", ha="center", va="center", fontsize=6, zorder=4)
        off = [ghz_F(r)[1] for r in rs if ghz_place(r)[0][0] == 0]
        on = [ghz_F(r)[1] for r in rs if ghz_place(r)[0][0] >= 1]
        runs = len({r.get("when") for r in rs})
        ttl = f"{reg}, {k}-qubit chains, {GHZ_MODE_LABEL.get(m, m)}"
        if layers not in (None, "none"):
            ttl += f", {layers}"
        ttl += (f"\n{len(rs)} units ({runs} run{'s' if runs != 1 else ''}): {key} off the seam "
                f"{np.mean(off) if off else float('nan'):.3f}, on it {np.mean(on) if on else float('nan'):.3f}\n"
                + textwrap.fill(setup, 64, break_on_hyphens=False))
        ax.set_title(ttl + ("" if mp else "  (no register map)"), fontsize=8)
    for ax in list(axs.flat)[len(panels):]:
        ax.set_axis_off()
    _ghz_legend(fig)
    fig.suptitle("18  GHZ tiles on the Nighthawk lattice: per site, the mean fidelity of the units holding it "
                 "(F_GHZ = (P0 + P1 + C)/2, Lemma 4; values above 1 are shot noise, shown as 1)", fontsize=10.5)
    fig.subplots_adjust(left=0.04, right=0.9, top=0.9, bottom=0.1, hspace=0.35)
    cax = fig.add_axes([0.92, 0.3, 0.012, 0.45])
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="site: mean fidelity of its units")
    return fig


def ghz_place_stats(rs, prefix=""):
    """{placement label: (sort key, mean, s.e. over units, units)} of a set of tile records."""
    by = defaultdict(list)
    keys = {}
    for r in rs:
        k, lab_ = ghz_place(r)
        v = ghz_F(r, prefix)[1]
        if v is not None:
            by[lab_].append(v)
            keys[lab_] = k
    return {lab_: (keys[lab_],) + _mean_err(v) + (len(v),) for lab_, v in by.items()}


def fig_ghz_modes(plt, store):
    """19: per register and unit size, the mean fidelity by placement, one bar per Layer 2 mode, with the exact
    reference as a black tick: where ACE follows the paper and where the seam takes over."""
    tl = ghz_tiles(store.ghz)
    if not tl:
        return None
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    keys = sorted(tl)
    nc = min(2, len(keys))
    nr = math.ceil(len(keys) / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(7.4 * nc, 4.8 * nr + 0.8), squeeze=False)
    for ax, key in zip(axs.flat, keys):
        by = tl[key]
        modes = [m for m in GHZ_MODES if by.get(m)]
        stats = {m: ghz_place_stats(by[m]) for m in modes}
        xstats = {m: ghz_place_stats(by[m], "exact_") for m in modes}
        places = sorted({p for s in stats.values() for p in s}, key=lambda p: next(s[p][0] for s in stats.values() if p in s))
        w = 0.8 / max(1, len(modes))
        for i, m in enumerate(modes):
            for j, p in enumerate(places):
                if p not in stats[m]:
                    continue
                _, mu, se, cnt = stats[m][p]
                x = j - 0.4 + w * (i + 0.5)
                ax.bar(x, mu, w * 0.9, yerr=se, color=GHZ_MODE_COLOR.get(m, "k"), alpha=0.85, capsize=2,
                       error_kw=dict(lw=0.8))
                if p in xstats[m]:
                    ax.plot([x - w * 0.42, x + w * 0.42], [xstats[m][p][1]] * 2, "-", color="k", lw=1.6, zorder=4)
        cnt_of = {p_: max(s[p_][3] for s in stats.values() if p_ in s) for p_ in places}
        ax.set_xticks(range(len(places)), [f"{p_}\n({cnt_of[p_]} unit records)" for p_ in places], fontsize=8)
        ax.axhline(0.5, color="#888888", lw=0.8, ls="--")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("mean unit fidelity (F_GHZ, or the witness)")
        reg, k, layers, setup = key
        ax.set_title(f"{reg}, {k}-qubit chains" + (f", {layers}" if layers not in (None, "none") else "")
                     + f"\n{setup}", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axs.flat)[len(keys):]:
        ax.set_axis_off()
    hs = [Patch(color=GHZ_MODE_COLOR[m], alpha=0.85, label=GHZ_MODE_LABEL[m]) for m in GHZ_MODES
          if any(tl[k_].get(m) for k_ in keys)]
    hs.append(Line2D([], [], color="k", lw=1.6, label="exact reference (same eps draws)"))
    fig.legend(handles=hs, loc="lower center", ncol=len(hs), fontsize=8)
    fig.suptitle("19  GHZ tiles by placement and Layer 2 mode (bars: ACE, mean +/- s.e. over units; tick: exact "
                 "reference; dashed: 0.5, genuine entanglement above)", fontsize=10.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.16, hspace=0.45)
    return fig


def fig_ghz_exact(plt, store):
    """20: every unit's ACE fidelity against its exact reference, one panel per Layer 2 mode, coloured by
    placement: on the diagonal ACE reproduces the paper's model; below it the seam costs."""
    rs = [r for r in store.ghz if r.get("mode") == "tiles" and ghz_F(r, "exact_")[1] is not None]
    if not rs:
        return None
    modes = [m for m in GHZ_MODES if any(ghz_mode(r) == m for r in rs)]
    places = sorted({ghz_place(r) for r in rs})
    pal = ["C2", "C1", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C0"]
    pcol = {}
    for _, lab_ in places:                               # one colour per placement label, cycling
        pcol.setdefault(lab_, pal[len(pcol) % len(pal)])
    regs = sorted({r["register"] for r in rs})
    mk = {reg: "osD^vP*X<>ph"[i % 12] for i, reg in enumerate(regs)}
    nc = min(4, len(modes))
    nr = math.ceil(len(modes) / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(4.6 * nc, 4.6 * nr + 0.9), squeeze=False)
    from matplotlib.lines import Line2D
    for ax, m in zip(axs.flat, modes):
        sub = [r for r in rs if ghz_mode(r) == m]
        for r in sub:
            ax.plot(ghz_F(r, "exact_")[1], ghz_F(r)[1], mk[r["register"]], color=pcol[ghz_place(r)[1]], ms=4.5,
                    alpha=0.7, mew=0)
        ax.plot([0, 1.05], [0, 1.05], "-", color="#888888", lw=0.8)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_aspect("equal")
        ax.set_xlabel("exact reference")
        ax.set_ylabel("ACE / loom / QrackBridge read")
        off = [ghz_F(r)[1] - ghz_F(r, "exact_")[1] for r in sub if ghz_place(r)[0][0] == 0]
        on = [ghz_F(r)[1] - ghz_F(r, "exact_")[1] for r in sub if ghz_place(r)[0][0] >= 1]
        ax.set_title(f"{GHZ_MODE_LABEL.get(m, m)}: {len(sub)} units\nACE - exact: off the seam "
                     f"{np.mean(off) if off else float('nan'):+.3f}, on it {np.mean(on) if on else float('nan'):+.3f}",
                     fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axs.flat)[len(modes):]:
        ax.set_axis_off()
    hs = [Line2D([], [], marker="o", ls="", color=pcol[lab_], label=lab_) for _, lab_ in places]
    hs += [Line2D([], [], marker=mk[reg], ls="", color="#555555", label=reg) for reg in regs]
    fig.legend(handles=hs, loc="lower center", ncol=min(6, len(hs)), fontsize=8)
    fig.suptitle("20  GHZ tiles, the engine's read (ACE, loom or QrackBridge) against the exact reference per unit (F_GHZ, or the witness)", fontsize=10.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.2, wspace=0.35, hspace=0.45)
    return fig


def fig_ghz_chip(plt, store):
    """21: chip-wide GHZ states (newest run per register, tree, Layer 2 mode, setup): per site the share of
    Z shots where it differs from the state's majority, the tree's CXs drawn thick, the breaking ones labelled."""
    ch = ghz_chips(store.ghz)
    if not ch:
        return None
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    keys = sorted(ch, key=lambda k_: (k_[0], GHZ_MODES.index(k_[2]) if k_[2] in GHZ_MODES else 9, k_[1], k_[3], k_[4]))
    nc = min(4, len(keys))
    nr = math.ceil(len(keys) / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(5.2 * nc, 5.6 * nr + 0.8), squeeze=False)
    cmap, norm = plt.get_cmap("Reds"), Normalize(0.0, 0.5)
    for ax, key in zip(axs.flat, keys):
        reg, tree, m, layers, setup = key
        r = ch[key][-1]
        mp = ghz_register_map(store, reg)
        pos = _ghz_lattice(ax, mp)
        seam = _ghz_seam(mp)
        for p_, c_, cl, v in r.get("edge_break", []):
            if int(p_) in pos and int(c_) in pos:
                (r1, c1), (r2, c2) = pos[int(p_)], pos[int(c_)]
                ax.plot([c1, c2], [r1, r2], "-", color="C1" if cl != "exact" else "#555555", lw=3.4, zorder=1.5,
                        solid_capstyle="round")
                if v >= 0.05:
                    ax.text((c1 + c2) / 2, (r1 + r2) / 2, f"{v:.2f}", fontsize=6.5, color="C3", ha="center",
                            va="center", zorder=5, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
        for q, v in (r.get("site_break") or {}).items():
            q = int(q)
            if q in pos:
                r_, c_ = pos[q]
                ax.plot(c_, r_, "D" if q in seam else "o", ms=11, mfc=cmap(norm(v)), mec="k" if q in seam else "#777777",
                        mew=1.4 if q in seam else 0.6, zorder=3)
        if r.get("root") is not None and int(r["root"]) in pos:
            r_, c_ = pos[int(r["root"])]
            ax.plot(c_, r_, "*", ms=13, color="k", zorder=4)
        kF, F = ghz_F(r)
        xk, xF = ghz_F(r, "exact_")
        cx = r.get("cx_classes") or {}
        ttl = (f"{reg}, {tree} tree, {GHZ_MODE_LABEL.get(m, m)}" + (f", {layers}" if layers not in (None, "none") else "")
               + f"\nP0+P1 {float(r.get('pop', np.nan)):.3f}, {kF} {F:.3f}" + (f" (exact {xF:.3f})" if xF is not None else "")
               + f"; CX exact/seam {cx.get('exact', 0)}/{cx.get('replica', 0) + cx.get('cross', 0)}"
               + (f"; {len(ch[key])} runs" if len(ch[key]) > 1 else "") + f"\n{setup}")
        ax.set_title(ttl, fontsize=8)
    for ax in list(axs.flat)[len(keys):]:
        ax.set_axis_off()
    _ghz_legend(fig, [Line2D([], [], color="#555555", lw=3.4, label="tree CX (thick)"),
                      Line2D([], [], marker="*", ls="", color="k", ms=11, label="root"),
                      Line2D([], [], color="C3", lw=0, marker="$0.47$", ms=16, label="tree CX break rate >= 0.05")])
    fig.suptitle("21  Chip-wide GHZ: per site the share of Z shots where it differs from the state's majority "
                 "(newest run per register, tree, Layer 2 mode and setup)", fontsize=10.5)
    fig.subplots_adjust(left=0.04, right=0.9, top=0.88, bottom=0.12, hspace=0.4)
    cax = fig.add_axes([0.92, 0.3, 0.012, 0.45])
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="site: differs from the majority (Z)")
    return fig


# ================================================================== the seam engines, Shor across the bridge, the dual Nighthawk R1
# (views 22-25). Records of nighthawk_qrack.py ghz: ghz / ghz_stack with `loom` (engine loom or bridge: chips, read cut /
# ghost / bridge, chip backend, bridge max_bonds / truncation / lazy_bonds, box L, H, ring, edges) and per unit `box`
# (faces, edge, ring cuts); shor_bridge + shor_map (shor); dual_r1 (dual), dual_map + dual_map_layout (dualmap).

def read_tests(paths, test, keyf):
    """Every record of this test in these files, once each by keyf(record)."""
    out, seen = [], set()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict) or r.get("test") != test:
                continue
            k = keyf(r)
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
    return out


def engine_geometry(r):
    """The geometry a seam-engine record ran on: the box, or the chips joined (and how)."""
    lo = r.get("loom") or {}
    if lo.get("box"):
        return f"box {lo['box']['L']}x{lo['box']['L']}x{lo['box']['H']} (six faces on a ring)"
    return " + ".join(lo.get("chips", [])) + (" (ACE chips)" if lo.get("backend") == "ace" else "")


PASS_STYLE = {1: ("-", "o", "pass 1, no drift"), 2: ("--", "s", "pass 2, Walsh echo"),
              "E": ("--", "s", "pass E, Walsh echo"), "V": (":", "^", "pass V"), "C": (":", "^", "pass C")}
PLACE_COLOR = ["#7f7f7f", "C0", "C1", "C3", "C4", "C5", "C2", "C6"]


def fig_bridge_capacity(plt, store):
    """22: GHZ fidelity across the seam engines against the bridge's capacity: per geometry (chips or the box), the
    mean F_GHZ of each placement (inside a chip / face, across the bridge, on the box direct, relayed, through the
    wrap cut) against max_bonds, the loom's hard cut at the left; one line style per stack pass, hollow markers for
    --inside-zz stabilizer."""
    rs = [r for r in store.ghz if engine_record_bonds(r) is not None and r.get("mode") == "tiles"]
    if not rs:
        return None
    geos = sorted({engine_geometry(r) for r in rs})
    nc = min(2, len(geos))
    nr = math.ceil(len(geos) / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(7.6 * nc, 4.8 * nr + 1.0), squeeze=False)
    from matplotlib.lines import Line2D
    places_all = sorted({ghz_place(r) for r in rs})
    pcol = {lab_: PLACE_COLOR[i % len(PLACE_COLOR)] for i, (_, lab_) in enumerate(places_all)}
    for ax, geo in zip(axs.flat, geos):
        sub = [r for r in rs if engine_geometry(r) == geo]
        acc = defaultdict(list)
        for r in sub:
            zz = ((r.get("layer2") or {}).get("inside_zz", "gate") if isinstance(r.get("layer2"), dict) else "gate")
            acc[(ghz_place(r), r.get("stack_pass", 1), zz, engine_record_bonds(r))].append(ghz_F(r)[1])
        lines = defaultdict(list)
        for (pl, ps, zz, mb), v in acc.items():
            lines[(pl, ps, zz)].append((mb, float(np.mean(v)), _mean_err(v)[1]))
        for (pl, ps, zz), pts in sorted(lines.items(), key=lambda kv: (kv[0][0][0], str(kv[0][1]), kv[0][2])):
            pts.sort()
            ls, mk, _ = PASS_STYLE.get(ps, ("-", "o", str(ps)))
            col = pcol[pl[1]]
            ax.errorbar([p_[0] for p_ in pts], [p_[1] for p_ in pts], yerr=[p_[2] for p_ in pts], fmt=mk, ls=ls,
                        color=col, mfc=("white" if zz == "stabilizer" else col), ms=6, lw=1.3, capsize=2, alpha=0.9)
        mbs = sorted({k[3] for k in acc})
        ax.set_xticks(mbs, ["loom\nhard cut" if m_ == -1 else str(m_) for m_ in mbs], fontsize=8)
        ax.axhline(0.5, color="#888888", lw=0.8, ls="--")
        ax.set_ylim(0.4, 1.06)
        ax.set_xlabel("QrackBridge max_bonds (bond qubits per cut)")
        ax.set_ylabel("mean F_GHZ of the units")
        ax.set_title(geo, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axs.flat)[len(geos):]:
        ax.set_axis_off()
    hs = [Line2D([], [], color=pcol[lab_], lw=2, label=lab_) for _, lab_ in places_all]
    passes = sorted({r.get("stack_pass", 1) for r in rs}, key=str)
    hs += [Line2D([], [], color="#555555", ls=PASS_STYLE.get(p_, ("-",))[0], marker=PASS_STYLE.get(p_, ("", "o"))[1],
                  label=PASS_STYLE.get(p_, ("", "", str(p_)))[2]) for p_ in passes]
    hs.append(Line2D([], [], color="#555555", marker="o", mfc="white", ls="", label="--inside-zz stabilizer"))
    fig.legend(handles=hs, loc="lower center", ncol=min(5, len(hs)), fontsize=8)
    fig.suptitle("22  GHZ chains across the seam engines: mean F_GHZ by placement against QrackBridge's capacity "
                 "(left: the loom's hard cut; dashed: 0.5, the separable bound)", fontsize=10.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.12 + 0.03 * math.ceil(len(hs) / 5), hspace=0.45)
    return fig


def shor_majority(m, carried):
    """Decoded logical Z when `carried` of m block signs survive and the rest are coin flips (ties count 1/2)."""
    lost, p = m - carried, 0.0
    for k in range(lost + 1):
        right = carried + k
        p += math.comb(lost, k) / 2 ** lost * (1.0 if 2 * right > m else 0.5 if 2 * right == m else 0.0)
    return p


def shor_bridge_class(r, detail=False):
    """inside one circuit / cut between blocks / cut through blocks, plus the box ring's wrap cut (where the bridge
    falls short); detail adds direct / relayed on the box (which give the same numbers)."""
    c = {"inside": "inside one circuit", "blocks whole": "cut between blocks", "blocks split": "cut through blocks"}
    lab_ = c.get(r.get("layout"), r.get("layout", "?"))
    if r.get("hops") and detail:
        lab_ += ", relayed" if r["hops"] > 1 else ", direct"
    if r.get("wrap"):
        lab_ += ", through the wrap cut"
    return lab_


SHOR_BRIDGE_COLOR = {"inside one circuit": "#7f7f7f", "cut between blocks": "C0", "cut through blocks": "C1",
                     "cut between blocks, through the wrap cut": "C9", "cut through blocks, through the wrap cut": "C3"}


def fig_shor_bridge(plt, store):
    """23: Shor's code across QrackBridge (nighthawk_qrack.py ghz shor): per geometry and code size, the mean decoded
    logical Z (solid) and logical X (dashed) of each way the cut meets the code against max_bonds, with the majority
    count over the block signs for the cut through blocks (dotted)."""
    rs = store.shor_bridge
    if not rs:
        return None
    keys = sorted({(engine_geometry(r), int(r.get("size", 3))) for r in rs}, key=lambda k_: (k_[1], k_[0]))
    nc = min(3, len(keys))
    nr = math.ceil(len(keys) / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(5.6 * nc, 4.4 * nr + 1.0), squeeze=False)
    from matplotlib.lines import Line2D
    classes = sorted({shor_bridge_class(r) for r in rs}, key=lambda c_: list(SHOR_BRIDGE_COLOR).index(c_)
                     if c_ in SHOR_BRIDGE_COLOR else 99)
    ccol = {c_: SHOR_BRIDGE_COLOR.get(c_, PLACE_COLOR[i % len(PLACE_COLOR)]) for i, c_ in enumerate(classes)}
    for ax, (geo, m) in zip(axs.flat, keys):
        sub = [r for r in rs if engine_geometry(r) == geo and int(r.get("size", 3)) == m]
        acc = defaultdict(lambda: defaultdict(list))
        for r in sub:
            mb = (r.get("loom") or {}).get("bridge", {}).get("max_bonds")
            for key in ("logical_Z", "logical_X"):
                if r.get(key) is not None:
                    acc[(shor_bridge_class(r), key)][mb].append(r[key])
        for (c_, key), by in sorted(acc.items()):
            mbs = sorted(by)
            ax.plot(mbs, [np.mean(by[x]) for x in mbs], ("-o" if key == "logical_Z" else "--s"), color=ccol[c_], ms=5,
                    lw=1.3, mfc=(ccol[c_] if key == "logical_Z" else "white"))
        mbs = sorted({(r.get("loom") or {}).get("bridge", {}).get("max_bonds") for r in sub})
        xs = np.arange(min(mbs), max(mbs) + 1)
        ax.plot(xs, [shor_majority(m, min(m, int(x))) for x in xs], ":", color="k", lw=1.2)
        ax.axhline(0.5, color="#888888", lw=0.8, ls="--")
        ax.set_ylim(0.4, 1.05)
        ax.set_xticks(xs)
        ax.set_xlabel("QrackBridge max_bonds")
        ax.set_ylabel("decoded success")
        ax.set_title(f"Shor-{m * m} on {geo}\n{len({r['unit'] for r in sub})} placements", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axs.flat)[len(keys):]:
        ax.set_axis_off()
    hs = [Line2D([], [], color=ccol[c_], lw=2, label=c_) for c_ in classes]
    hs += [Line2D([], [], color="#555555", ls="-", marker="o", label="logical Z (|0_L>, |1_L>)"),
           Line2D([], [], color="#555555", ls="--", marker="s", mfc="white", label="logical X (|+_L>, |-_L>)"),
           Line2D([], [], color="k", ls=":", label="majority over block signs, min(m, max_bonds) carried")]
    fig.legend(handles=hs, loc="lower center", ncol=min(4, len(hs)), fontsize=8)
    fig.suptitle("23  Shor's code across QrackBridge: decoded logical Z and X against the bridge's capacity, by how "
                 "the cut meets the code (dashed grey: 0.5, no logical information)", fontsize=10.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.12 + 0.035 * math.ceil(len(hs) / 4), hspace=0.5)
    return fig


# ================================================================== Layer 3: Shor order finding (nighthawk_qrack.py ghz shoralg)
SHA_LAYOUTS = ("inside", "split", "direct", "countsplit", "worksplit")
SHA_LAYOUT_LABEL = dict(inside="inside one circuit", split="counting | work, fan-out ancilla",
                        direct="counting | work, direct", countsplit="cut through the counting register",
                        worksplit="cut through the work register")
SHA_LAYOUT_COLOR = dict(inside="#7f7f7f", split="C0", direct="C1", countsplit="C2", worksplit="C4")


def sha_place(r):
    """The placement of a shor_alg record: the side pair, and on the box direct / relayed / through the wrap cut."""
    if r.get("engine") == "dual":
        return f"link {r.get('link')}"
    f = " / ".join(r.get("faces") or [])
    if r.get("hops"):
        f += " (" + ("relayed" if r["hops"] > 1 else "direct") + (", wrap cut" if r.get("wrap") else "") + ")"
    return f


def sha_case(r):
    return (r.get("geometry"), int(r["N"]), int(r["a"]), int(r["t"]))


def fig_shor_alg(plt, store, metric="success"):
    """26 (success) / 27 (tvd): Layer 3 across QrackBridge (nighthawk_qrack.py ghz shoralg), newest run per case: per
    geometry, N, a and counting width, the per-shot factoring success (the paper's rule) or the distance to the ideal
    distribution of every layout and placement against max_bonds; the ideal circuit's and a uniform guess's success
    as lines, the bond count the rule says the split layouts need as a mark."""
    rs = [r for r in store.shor_alg if r.get("engine") == "bridge"]
    if not rs:
        return None
    newest = {}
    for r in rs:
        newest[sha_case(r)] = max(newest.get(sha_case(r), ""), r.get("when", ""))
    rs = [r for r in rs if r.get("when") == newest[sha_case(r)]]
    keys = sorted(newest, key=lambda k: (k[0], k[1], k[2], k[3]))
    nc = min(3, len(keys))
    nr = math.ceil(len(keys) / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(5.4 * nc, 4.0 * nr + 1.2), squeeze=False)
    from matplotlib.lines import Line2D
    styles = ["-", "--", ":", "-."]
    places = []
    for ax, key in zip(axs.flat, keys):
        sub = [r for r in rs if sha_case(r) == key]
        r0 = sub[0]
        m_ = metric if metric != "success" or r0["exact"]["success"] > 0 else "order"     # odd r: no factor
        by = defaultdict(dict)
        for r in sub:
            if m_ in r:
                by[(r["layout"], sha_place(r))][int(r["max_bonds"])] = r
        pls = sorted({k[1] for k in by})
        for (lay, pl), d in sorted(by.items(), key=lambda kv: (SHA_LAYOUTS.index(kv[0][0]), kv[0][1])):
            if pl not in places:
                places.append(pl)
            mbs = sorted(d)
            ls = styles[pls.index(pl) % len(styles)] if lay != "inside" else "-"
            ax.plot(mbs, [d[x][m_] for x in mbs], ls, marker="o", ms=4, color=SHA_LAYOUT_COLOR.get(lay, "k"),
                    lw=1.4)
            if metric == "tvd":
                fl = [d[x].get("tvd_floor") for x in mbs if d[x].get("tvd_floor")]
                if fl:
                    ax.axhline(max(fl), color=SHA_LAYOUT_COLOR.get(lay, "k"), lw=0.6, ls=":")
        if metric == "success":
            ax.axhline(r0["exact"][m_], color="k", lw=1.0, ls="--")
            ax.axhline(r0["uniform"][m_], color="#888888", lw=1.0, ls=":")
            ax.set_ylim(-0.02, max(1.0, r0["exact"][m_] + 0.1))
            ax.set_ylabel("per-shot factoring success" if m_ == "success" else "order found (odd r: no factor)")
        else:
            ax.set_ylim(-0.02, 1.0)
            ax.set_ylabel("TVD to the ideal distribution")
        need = r0.get("bonds_needed") if r0.get("layout") in ("split", "direct") else next(
            (r.get("bonds_needed") for r in sub if r.get("layout") in ("split", "direct")), None)
        if need:
            ax.axvspan(need - 0.2, need + 0.2, color="#d9d8d3", zorder=0)
        ax.set_xlabel("QrackBridge max_bonds")
        ax.set_xticks(sorted({int(r["max_bonds"]) for r in sub}))
        ax.set_title(f"{key[0]}\nN {key[1]}, a {key[2]} (order {r0['r']}), {key[3]} counting + {r0['n_work']} work",
                     fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axs.flat)[len(keys):]:
        ax.set_axis_off()
    hs = [Line2D([], [], color=SHA_LAYOUT_COLOR[k], lw=2, label=SHA_LAYOUT_LABEL[k]) for k in SHA_LAYOUTS
          if any(r["layout"] == k for r in rs)]
    if len(places) > 1:
        hs += [Line2D([], [], color="#555555", ls=styles[i % len(styles)], label=p_) for i, p_ in enumerate(
            sorted({sha_place(r) for r in rs if r["layout"] != "inside"}))]
    if metric == "success":
        hs += [Line2D([], [], color="k", ls="--", label="ideal circuit"),
               Line2D([], [], color="#888888", ls=":", label="uniform guess")]
    from matplotlib.patches import Patch
    hs.append(Patch(color="#d9d8d3", label="bonds the rule needs (counting | work)"))
    fig.legend(handles=hs, loc="lower center", ncol=min(4, len(hs)), fontsize=8)
    what = "per-shot factoring success" if metric == "success" else "total variation distance to the ideal distribution"
    fig.suptitle(f"{26 if metric == 'success' else 27}  Layer 3 (Shor order finding) across QrackBridge: {what} "
                 "against the bridge's capacity, per layout and placement", fontsize=10.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86 if nr == 1 else 0.9,
                        bottom=0.1 + 0.035 * math.ceil(len(hs) / 4), hspace=0.55)
    return fig


def fig_shor_alg_dual(plt, store):
    """28: Layer 3 on the dual Nighthawk R1 (nighthawk_qrack.py ghz shoralg --dual), newest run per case: per N, a and
    counting width, every layout's per-shot factoring success under each model, with the ideal circuit and a
    uniform guess."""
    rs = [r for r in store.shor_alg if r.get("engine") == "dual" and "success" in r]
    if not rs:
        return None
    newest = {}
    for r in rs:
        newest[sha_case(r)] = max(newest.get(sha_case(r), ""), r.get("when", ""))
    rs = [r for r in rs if r.get("when") == newest[sha_case(r)]]
    keys = sorted(newest)
    fig, axs = plt.subplots(len(keys), 1, figsize=(10.5, 1.4 + 1.25 * sum(
        len({r["layout"] for r in rs if sha_case(r) == k}) for k in keys)), squeeze=False)
    for ax, key in zip(axs.flat, keys):
        sub = [r for r in rs if sha_case(r) == key]
        lays = [x for x in SHA_LAYOUTS if any(r["layout"] == x for r in sub)]
        for i, lay in enumerate(lays):
            y = len(lays) - 1 - i
            ax.axhline(y, color="#e4e2dc", lw=0.8, zorder=0)
            for r in sub:
                if r["layout"] == lay:
                    lab_, col, mk = DUAL_MODEL_STYLE.get(r["model"], (r["model"], "k", "o"))
                    ax.plot(r["success"], y, mk, color=col, ms=8, zorder=3,
                            label=lab_ if not any(h.get_label() == lab_ for h in ax.lines) else None)
            rr = next(r for r in sub if r["layout"] == lay)
            ax.text(1.005, y, f"{rr['route']['distant']} distant, {rr['route']['local_cx']} local CX, "
                              f"{rr['route']['swaps']} SWAPs", transform=ax.get_yaxis_transform(), fontsize=7,
                    va="center")
        r0 = sub[0]
        ax.axvline(r0["exact"]["success"], color="k", lw=1.0, ls="--")
        ax.axvline(r0["uniform"]["success"], color="#888888", lw=1.0, ls=":")
        ax.set_yticks(range(len(lays)), [SHA_LAYOUT_LABEL[x] for x in lays][::-1], fontsize=8.5)
        ax.set_xlim(-0.02, max(0.6, r0["exact"]["success"] + 0.05))
        ax.set_title(f"N {key[1]}, a {key[2]} (order {r0['r']}), {key[3]} counting + {r0['n_work']} work; distant "
                     f"CNOTs through {r0.get('link')}; dashed: ideal, dotted: uniform guess", fontsize=8.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axs.flat[-1].set_xlabel("per-shot factoring success (the paper's rule)")
    hs, ls_ = [], []
    for ax in axs.flat:
        for h in ax.lines:
            if h.get_label() and not h.get_label().startswith("_") and h.get_label() not in ls_:
                hs.append(h)
                ls_.append(h.get_label())
    fig.legend(hs, ls_, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"28  Layer 3 on the dual Nighthawk R1 (arXiv:2610.29986): factoring success per layout and model "
                 f"(local CZ error {r0.get('r_cz')}, {max(r.get('trajectories', 1) for r in rs)} trajectories)",
                 fontsize=10)
    fig.subplots_adjust(left=0.24, right=0.8, top=0.9, bottom=0.14, hspace=0.6)
    return fig


def sha_rows(store):
    out = []
    for r in store.shor_alg:
        row = dict(when=r.get("when"), geometry=r.get("geometry"), engine=r.get("engine"), model=r.get("model", ""),
                   N=r.get("N"), a=r.get("a"), r=r.get("r"), t=r.get("t"), n_work=r.get("n_work"),
                   qubits=r.get("qubits"), layout=r.get("layout"), placement=sha_place(r),
                   synth=" ".join(r.get("synth") or []), max_bonds=r.get("max_bonds"), bonds_needed=r.get("bonds_needed"),
                   cx=r.get("cx"), cross_cx=r.get("cross"), success=r.get("success"), order=r.get("order"),
                   tvd=r.get("tvd"), tvd_floor=r.get("tvd_floor"), fidelity=r.get("fidelity"), zero=r.get("zero"),
                   peak_bonds=r.get("peak_bonds"), truncations=r.get("truncations"),
                   fidelity_estimate=r.get("fidelity_estimate"), readout=r.get("readout"),
                   vanished=r.get("vanished", ""),
                   ideal_success=(r.get("exact") or {}).get("success"),
                   uniform_success=(r.get("uniform") or {}).get("success"), qrack=r.get("qrack"))
        if r.get("engine") == "dual":
            row.update(link=r.get("link"), **{f"route_{k}": v for k, v in (r.get("route") or {}).items()},
                       trajectories=r.get("trajectories"), r_cz=r.get("r_cz"))
        out.append(row)
    return out


DUAL_MODEL_STYLE = {"hardcut": ("hard cut (bridge, max_bonds 0)", "#8a8984", "o"),
                    "ideal": ("bridge, exact", "#b9b8b2", "o"), "exact": ("QrackSimulator, exact", "#b9b8b2", "s"),
                    "budget": ("bridge + constituent budget", "C1", "o"),
                    "measured": ("bridge + measured distant CZ", "C0", "o")}
DUAL_MODELS = ("hardcut", "ideal", "exact", "budget", "measured")


def fig_dual(plt, store):
    """24: the dual Nighthawk R1 (nighthawk_qrack.py ghz dual, arXiv:2610.29986): the paper's Bell and GHZ states under each
    model (newest run), against IBM's raw and readout-mitigated values."""
    if not store.dual:
        return None
    when = max(r["when"] for r in store.dual)
    rows = [r for r in store.dual if r["when"] == when]
    models = [m_ for m_ in DUAL_MODELS if m_ in rows[0]]
    if "ideal" in models and "exact" in models:
        models.remove("exact")
    n = len(rows)
    fig, ax = plt.subplots(figsize=(10.5, 1.6 + 0.5 * n))
    for i, r in enumerate(rows):
        y = n - 1 - i
        ax.axhline(y, color="#e4e2dc", lw=0.8, zorder=0)
        for mo in models:
            lab_, col, mk = DUAL_MODEL_STYLE[mo]
            ax.errorbar(r[mo]["F"], y, xerr=r[mo].get("se") or None, fmt=mk, color=col, ms=7, zorder=3,
                        label=lab_ if i == 0 else None)
        ib = r.get("ibm")
        if ib:
            first = not any(q.get("ibm") for q in rows[:i])
            ax.errorbar(ib["mitigated"], y, xerr=ib["mitigated_se"], fmt="D", color="k", ms=7, zorder=4,
                        label="IBM, readout-mitigated" if first else None)
            ax.errorbar(ib["unmitigated"], y, xerr=ib["unmitigated_se"], fmt="D", mfc="white", color="k", ms=7,
                        zorder=4, label="IBM, raw" if first else None)
    ax.set_yticks(range(n), [r["state"] for r in rows][::-1], fontsize=9)
    ax.axvline(0.5, color="#888888", lw=0.8, ls="--")
    ax.set_xlim(0.45, 1.01)
    ax.set_xlabel("state fidelity F (models: no readout error)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32 - 0.6 / n), ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"24  Dual Nighthawk R1 (arXiv:2610.29986): the paper's states under the bridge's models (run {when}, "
                 f"{rows[0].get('trajectories')} trajectories, local CZ error {rows[0].get('local_cz_error')})",
                 fontsize=10)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.9, bottom=0.12 + 1.1 / (1.6 + 0.5 * n))
    return fig


def fig_dual_map(plt, store):
    """25: GHZ fidelity over both modules of the dual Nighthawk R1 and their L-couplers (nighthawk_qrack.py ghz dualmap,
    newest run): per model, each site the lowest F_GHZ of the 3-qubit chains through it, module B drawn rotated by
    180 degrees facing A, the L-qubits and cables between them."""
    if not store.dual_map:
        return None
    when = max(r["when"] for r in store.dual_map)
    recs = [r for r in store.dual_map if r["when"] == when]
    lay = next((r for r in store.dual_layout if r["when"] == when), None)
    if lay is None:
        return None
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    models = list(lay.get("models") or recs[0]["F"].keys())
    pos = {int(st[0]): (int(st[1]), int(st[2])) for st in lay["sites"]}
    fig, axs = plt.subplots(1, len(models), figsize=(4.4 * len(models) + 1.2, 11.5), squeeze=False)
    cmap, norm = plt.get_cmap("RdYlGn"), Normalize(0.5, 1.0)
    for ax, mo in zip(axs.flat, models):
        per = defaultdict(list)
        for r in recs:
            for q in r["qubits"]:
                per[int(q)].append(r["F"][mo])
        for q, (r_, c_) in pos.items():
            if q >= 240:
                ax.plot(c_, r_, "s", ms=9, mfc="#d6d4cd", mec="#777777", mew=0.6, zorder=3)
                ax.text(c_, r_, str(q), fontsize=4.5, ha="center", va="center", zorder=4)
                continue
            v = min(per.get(q, [float("nan")]))
            ax.plot(c_, r_, "o", ms=10.5, mfc=cmap(norm(v)), mec="#777777", mew=0.5, zorder=3)
            ax.text(c_, r_, f"{v:.2f}", fontsize=4.6, ha="center", va="center", zorder=4)
        for L in lay["links"]:
            (r1, c1), (r2, c2) = pos[L["la"]], pos[L["lb"]]
            ax.plot([c1, c2], [r1, r2], "-", color=LINK_COLOR, lw=2.4, zorder=2)
            ax.text((c1 + c2) / 2 + 0.25, (r1 + r2) / 2, f"{100 * L['dcz']:.1f}", fontsize=6, va="center")
            for a_, b_ in ((L["a"], L["la"]), (L["lb"], L["b"])):
                (r1, c1), (r2, c2) = pos[a_], pos[b_]
                ax.plot([c1, c2], [r1, r2], "-", color="#999999", lw=1.0, zorder=1)
        lat = [r["F"][mo] for r in recs if not r.get("link")]
        lk = [np.mean([r["F"][mo] for r in recs if r.get("link") == L["name"]]) for L in lay["links"]]
        ax.set_title(f"{mo}\nlattice {np.mean(lat):.3f} · links " + " ".join(f"{v:.2f}" for v in lk), fontsize=8.5)
        ax.set_xlim(-0.8, 9.8)
        ax.set_ylim(26.8, -0.8)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(f"25  Dual Nighthawk R1: lowest F_GHZ per site of the 3-qubit chains through it, both modules and their "
                 f"L-couplers (run {when}; cables labelled with their distant-CZ fidelity)", fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.9, top=0.92, bottom=0.02, wspace=0.05)
    cax = fig.add_axes([0.92, 0.3, 0.012, 0.4])
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="site: lowest F_GHZ of its chains")
    return fig


def engine_rows(store):
    """CSV rows: shor_bridge per unit, dual_r1 per state and model, dual_map per chain."""
    sb, du, dm = [], [], []
    for r in store.shor_bridge:
        lo = r.get("loom") or {}
        row = dict(when=r.get("when"), register=r.get("register"), geometry=engine_geometry(r), size=r.get("size"),
                   unit=r.get("unit"), layout=r.get("layout"), shor_class=r.get("shor_class"),
                   blocks_split=r.get("blocks_split"), cross_outer=r.get("cross_outer"),
                   cross_inner=r.get("cross_inner"), chips=" ".join(r.get("chips", [])), hops=r.get("hops", ""),
                   wrap=r.get("wrap", ""), max_bonds=(lo.get("bridge") or {}).get("max_bonds"),
                   truncation=(lo.get("bridge") or {}).get("truncation"), logical_Z=r.get("logical_Z"),
                   logical_X=r.get("logical_X"), bonds_Z=r.get("bonds_Z"), bonds_X=r.get("bonds_X"),
                   qubits=" ".join(str(q) for q in r.get("qubits", [])), shots=r.get("shots"), qrack=r.get("qrack"))
        for s_, d in (r.get("states") or {}).items():
            row.update({f"decoded_{s_}": d.get("decoded"), f"raw_{s_}": d.get("raw"), f"impossible_{s_}": d.get("impossible"),
                        f"exact_{s_}": d.get("exact_decoded"), f"vanished_{s_}": d.get("vanished")})
        sb.append(row)
    for r in store.dual:
        for mo in DUAL_MODELS:
            if mo in r:
                ib = r.get("ibm") or {}
                du.append(dict(when=r.get("when"), state=r.get("state"), kind=r.get("kind"),
                               qubits=" ".join(str(q) for q in r.get("qubits", [])), model=mo, F=r[mo]["F"],
                               se=r[mo].get("se"), peak_bonds=r[mo].get("peak_bonds"),
                               local_cz_error=r.get("local_cz_error"), trajectories=r.get("trajectories"),
                               ibm_mitigated=ib.get("mitigated"), ibm_raw=ib.get("unmitigated")))
    for r in store.dual_map:
        row = dict(when=r.get("when"), unit=r.get("unit"), tiling=r.get("tiling"), link=r.get("link") or "",
                   qubits=" ".join(str(q) for q in r.get("qubits", [])), trajectories=r.get("trajectories"),
                   local_cz_error=r.get("local_cz_error"))
        row.update({f"F_{mo}": v for mo, v in (r.get("F") or {}).items()})
        dm.append(row)
    return sb, du, dm


def write_engine_tables(store, path):
    wrote = False
    for rows, tag in zip(list(engine_rows(store)) + [sha_rows(store)], ("shorbridge", "dual", "dualmap", "shoralg")):
        if not rows:
            continue
        cols = []
        for r in rows:
            cols += [k for k in r if k not in cols]
        out = str(Path(path).with_name(Path(path).stem + f"_{tag}.csv"))
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
        print(f"wrote {len(rows)} {tag} rows to {out}")
        wrote = True
    print_engine_summary(store)
    return wrote


def print_engine_summary(store):
    rs = [r for r in store.ghz if engine_record_bonds(r) is not None and r.get("mode") == "tiles"]
    if rs:
        print("\nghz across the seam engines: mean F_GHZ by placement (units), per geometry, max_bonds (-1: loom hard "
              "cut), pass and inside ZZ")
        acc = defaultdict(list)
        for r in rs:
            zz = (r.get("layer2") or {}).get("inside_zz", "gate") if isinstance(r.get("layer2"), dict) else "gate"
            acc[(engine_geometry(r), engine_record_bonds(r), str(r.get("stack_pass", 1)), zz, ghz_place(r))].append(
                ghz_F(r)[1])
        for k in sorted(acc, key=lambda k_: (k_[0], k_[1], k_[2], k_[3], k_[4][0])):
            print(f"  {k[0]:<42} mb {k[1]:>2}  pass {k[2]:<2} {k[3]:<10} {k[4][1]:<34} {np.mean(acc[k]):.3f} "
                  f"({len(acc[k])})")
    if store.shor_bridge:
        print("\nShor across QrackBridge: decoded logical Z / X by geometry, size, max_bonds and class (units)")
        acc = defaultdict(list)
        for r in store.shor_bridge:
            acc[(engine_geometry(r), r.get("size"), (r.get("loom") or {}).get("bridge", {}).get("max_bonds"),
                 shor_bridge_class(r, True))].append(r)
        for k in sorted(acc, key=lambda k_: (k_[1], k_[0], -int(k_[2] or 0), k_[3])):
            v = acc[k]
            z = [r["logical_Z"] for r in v if r.get("logical_Z") is not None]
            x = [r["logical_X"] for r in v if r.get("logical_X") is not None]
            print(f"  Shor-{k[1] * k[1]:<3} {k[0]:<40} mb {k[2]:>2}  {k[3]:<38} {np.mean(z) if z else float('nan'):.3f} / "
                  f"{np.mean(x) if x else float('nan'):.3f} ({len(v)})")
    if store.shor_alg:
        print("\nLayer 3 (Shor order finding): per-shot factoring success (TVD to the ideal distribution) by geometry, "
              "N, a, t, layout, placement and max_bonds / model")
        for r in sorted(store.shor_alg, key=lambda r: (r.get("geometry"), r["N"], r["a"], r["t"], r["engine"],
                                                       SHA_LAYOUTS.index(r["layout"]), sha_place(r),
                                                       str(r.get("model", "")), r.get("max_bonds") or 0)):
            if "success" not in r:
                continue
            tag = (r.get("model") or "") if r["engine"] == "dual" else ("exact" if r["engine"] == "exact" else
                                                                         f"mb {r.get('max_bonds')}")
            print(f"  {r.get('geometry', '')[:34]:<35} N {r['N']:>2} a {r['a']:>2} t {r['t']:>2} {r['layout']:<11}"
                  f"{sha_place(r)[:30]:<31}{tag:<10}{r['success']:.3f} ({r['tvd']:.3f})")
    if store.dual:
        when = max(r["when"] for r in store.dual)
        print(f"\ndual Nighthawk R1 (run {when}): F per model, IBM mitigated / raw")
        for r in [r for r in store.dual if r["when"] == when]:
            ib = r.get("ibm") or {}
            print(f"  {r['state']:<30} " + "  ".join(f"{mo} {r[mo]['F']:.3f}" for mo in DUAL_MODELS if mo in r)
                  + (f"   IBM {ib['mitigated']:.4f} / {ib['unmitigated']:.4f}" if ib else ""))


GHZ_CSV_KEYS = ("P0", "P1", "pop", "C_parity", "Phi_coh", "F_GHZ", "F0", "F_witness", "lambda_per_block",
                "sum_dw_estimate", "C_x", "phase_x")


def ghz_rows(store):
    rows = []
    for r in sorted(store.ghz, key=lambda r: (r.get("register", ""), r.get("mode", ""), ghz_mode(r), r.get("when", ""),
                                              str(r.get("tiling", r.get("tree", ""))), r.get("unit", 0))):
        l2 = r.get("layer2") if isinstance(r.get("layer2"), dict) else {}
        cx = r.get("cx_classes") or {}
        row = dict(kind=r.get("mode"), register=r.get("register"), setup=shor_setup(r)[0], when=r.get("when"),
                   layer2=ghz_mode(r), blocks=l2.get("blocks", 0), t_idle=l2.get("t_idle"), j_zz=l2.get("j_zz"),
                   sigma_qs=l2.get("sigma_qs"), seed=l2.get("seed"), layers=r.get("layers"),
                   tree=r.get("tree", ""), tiling=r.get("tiling", ""), unit=r.get("unit", ""),
                   unit_size=r.get("unit_size", r.get("n")), qubits=" ".join(str(q) for q in r.get("qubits", [])),
                   placement=ghz_place(r)[1] if r.get("mode") == "tiles" else "chip",
                   seam_distance=r.get("seam_distance"), cx_exact=cx.get("exact", 0), cx_replica=cx.get("replica", 0),
                   cx_cross=cx.get("cross", 0), shots=r.get("shots"), realisations=r.get("repeats"),
                   estimator=r.get("estimator"), inside_zz=l2.get("inside_zz", "gate"),
                   stack_order=r.get("stack_order", ""), stack_pass=r.get("stack_pass", ""),
                   seam_class=r.get("seam_class", ""), seam_plan=r.get("seam_plan", ""))
        lo = r.get("loom") if isinstance(r.get("loom"), dict) else {}
        bp, bx = lo.get("bridge") or {}, r.get("box") or {}
        row.update(engine=("ace" if not lo else "bridge" if lo.get("read") == "bridge" else "loom"),
                   loom_read=lo.get("read", ""), chip_backend=lo.get("backend", ""),
                   chips=" + ".join(lo.get("chips", [])), max_bonds=bp.get("max_bonds", ""),
                   truncation=bp.get("truncation", ""), lazy_bonds=bp.get("lazy_bonds", ""),
                   box=(f"{lo['box']['L']}x{lo['box']['L']}x{lo['box']['H']}" if lo.get("box") else ""),
                   box_edge=bx.get("edge") or "", box_cuts=" ".join(str(c) for c in bx.get("cuts", [])),
                   chips_spanned=len(ghz_chips_of(r)) if lo else "")
        row.update({k: r.get(k) for k in GHZ_CSV_KEYS})
        row.update({"exact_" + k: r.get("exact_" + k) for k in GHZ_CSV_KEYS})
        rows.append(row)
    return rows


def write_ghz_table(store, path):
    rows = ghz_rows(store)
    if not rows:
        return False
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {len(rows)} ghz rows to {path}")
    print_ghz_summary(store)
    return True


def print_ghz_summary(store):
    tl = ghz_tiles(store.ghz)
    if tl:
        print("\nghz tiles: mean unit fidelity (F_GHZ, or the witness) by placement and Layer 2 mode, ACE / exact, units")
        for key in sorted(tl):
            reg, k, layers, setup = key
            print(f"  {reg}, {k}-qubit chains" + (f", {layers}" if layers not in (None, "none") else "") + f"  [{setup}]")
            for m in GHZ_MODES:
                rs = tl[key].get(m)
                if not rs:
                    continue
                st, xs = ghz_place_stats(rs), ghz_place_stats(rs, "exact_")
                cells = [f"{p}: {st[p][1]:.3f}" + (f" / {xs[p][1]:.3f}" if p in xs else "") + f" ({st[p][3]})"
                         for p in sorted(st, key=lambda p: st[p][0])]
                print(f"    {GHZ_MODE_LABEL.get(m, m):<21} " + "; ".join(cells))
    ch = ghz_chips(store.ghz)
    if ch:
        print("\nghz chip-wide (newest run): P0+P1, fidelity, exact reference, CX exact/seam")
        for key in sorted(ch):
            reg, tree, m, layers, setup = key
            r = ch[key][-1]
            kF, F = ghz_F(r)
            xk, xF = ghz_F(r, "exact_")
            cx = r.get("cx_classes") or {}
            print(f"  {reg:<14} {tree:<5} {GHZ_MODE_LABEL.get(m, m):<21} {float(r.get('pop', np.nan)):.3f}  "
                  f"{kF} {F:.3f}" + (f"  exact {xF:.3f}" if xF is not None else "")
                  + f"  {cx.get('exact', 0)}/{cx.get('replica', 0) + cx.get('cross', 0)}  [{setup}]")


# ================================================================== shor9 in 3D (--grid3d, viewer button)
# A self-contained page per register and code size: the register on the Nighthawk lattice (ACE patches, seam
# lines, couplers by class, block leaders) with every Shor unit as a column whose height is the logical
# information it kept, (decoded success - 0.5) / 0.5; setups side by side; a switch that pulls the patches
# apart into ACE's simulators (the seam qubits stay in the crossbar and reappear as copies in each patch
# they join). three.js 0.160.0 from jsDelivr, or a local copy (--three DIR: an npm `three` package) served
# from a local web server (--serve). The page text is ASCII here: entities in HTML, u-escapes in JS.
THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0/"
SHOR3D_HEAD = r"""<title>__TITLE__</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* Layout: poster header and figure row on top, the 3D stage as the page's one large surface, controls and units beside it (below it on phones). */
:root {
  --bg: #fcfcfb; --panel: #f3f2ef; --stage: #f0efeb; --ink: #0b0b0b; --ink2: #52514e; --muted: #8a8984; --grid: #d9d8d3; --line: #e4e3df;
  --p0: #2a78d6; --p1: #eb6834; --pc0: #2a78d6; --pc1: #eb6834; --pc2: #8a5cc7; --pc3: #d39a12; --pc4: #2f9e8f; --pc5: #c2477d; --cross: #c0392b; --seam: #1baf7a; --focus: #2a78d6;
  --display: "Inter", "Helvetica Neue", Arial, sans-serif; --body: "Inter", "Helvetica Neue", Arial, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
  --r: 10px;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #121315; --panel: #1b1c1f; --stage: #0d0e10; --ink: #ecebe7; --ink2: #b8b6b0; --muted: #807e78; --grid: #34353a; --line: #2a2b2f;
  --p0: #5b9cf0; --p1: #f2875a; --pc0: #5b9cf0; --pc1: #f2875a; --pc2: #a982e0; --pc3: #e3b23a; --pc4: #4fbcad; --pc5: #e06b9f; --cross: #ef6b5c; --seam: #2fd093; --focus: #7fb2f5; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #121315; --panel: #1b1c1f; --stage: #0d0e10; --ink: #ecebe7; --ink2: #b8b6b0; --muted: #807e78; --grid: #34353a; --line: #2a2b2f;
  --p0: #5b9cf0; --p1: #f2875a; --pc0: #5b9cf0; --pc1: #f2875a; --pc2: #a982e0; --pc3: #e3b23a; --pc4: #4fbcad; --pc5: #e06b9f; --cross: #ef6b5c; --seam: #2fd093; --focus: #7fb2f5; color-scheme: dark }
html, body { height: auto; }
body { background: var(--bg); color: var(--ink); font-family: var(--body); font-size: 14px; line-height: 1.45;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 22px 28px; }
.wrap { max-width: 1440px; margin: 0 auto; display: grid; gap: 18px; }
header { display: grid; gap: 6px; }
.eyebrow { font-size: 12px; color: var(--muted); letter-spacing: .02em; }
h1 { margin: 0; font-family: var(--display); font-weight: 800; font-size: clamp(30px, 4.2vw, 46px); letter-spacing: -.02em; line-height: 1.05; text-wrap: balance; }
h1 span { font-weight: 500; color: var(--ink2); font-size: .5em; letter-spacing: 0; margin-left: .5em; }
.sub { margin: 0; color: var(--ink2); max-width: 75ch; }
.figs { display: grid; grid-template-columns: repeat(auto-fit, minmax(132px, 1fr)); gap: 8px; }
.fig { background: var(--panel); border-radius: 8px; padding: 10px 12px; display: grid; gap: 2px; min-width: 0; }
.fig b { font-size: 21px; font-weight: 800; font-variant-numeric: tabular-nums; letter-spacing: -.01em; }
.fig > span { font-size: 11.5px; color: var(--ink2); }
.main { display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: 18px; align-items: start; }
@media (max-width: 980px) { .main { grid-template-columns: minmax(0, 1fr); } }
.stage { position: relative; background: var(--stage); border-radius: var(--r); overflow: hidden; height: clamp(380px, 72vh, 820px); min-width: 0; touch-action: none; }
.stage canvas { display: block; }
.labels { position: absolute; inset: 0; pointer-events: none; }
.qlab { font: 600 9.5px/1 var(--mono); color: #fff; text-shadow: 0 0 2px rgba(0,0,0,.65); pointer-events: none; }
.ulab { font: 600 12px/1.25 var(--body); color: var(--ink); background: var(--bg); border: 1.5px solid var(--ink2); border-radius: 6px; padding: 4px 7px;
  white-space: nowrap; pointer-events: auto; cursor: pointer; font-variant-numeric: tabular-nums; box-shadow: 0 1px 3px rgba(0,0,0,.12); }
.ulab.sel { outline: 2px solid var(--focus); outline-offset: 2px; }
.ulab em { font-style: normal; color: var(--ink2); font-weight: 500; }
.llab { font: 600 11px/1.2 var(--body); color: var(--ink2); letter-spacing: .03em; text-transform: uppercase; pointer-events: none; white-space: nowrap; }
.hint { position: absolute; right: 12px; top: 10px; font-size: 11.5px; color: var(--muted); pointer-events: none; text-align: right; }
.legend { position: absolute; left: 12px; bottom: 10px; display: flex; flex-wrap: wrap; gap: 6px 14px; max-width: calc(100% - 24px); font-size: 11.5px; color: var(--ink2); pointer-events: none; }
.legend i { display: inline-block; width: 11px; height: 11px; border-radius: 50%; vertical-align: -1px; margin-right: 5px; }
.legend i.bar { border-radius: 2px; height: 4px; width: 16px; vertical-align: 2px; }
.legend i.ring { background: transparent; border: 2px dashed var(--ink2); }
.nogl { position: absolute; inset: 0; display: grid; place-items: center; padding: 24px; text-align: center; color: var(--ink2); }
aside { display: grid; gap: 14px; min-width: 0; }
.ctl { display: grid; gap: 6px; }
.ctl > .lbl { font-size: 11.5px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
.seg { display: flex; background: var(--panel); border-radius: 8px; padding: 3px; gap: 3px; }
.seg button { flex: 1 1 0; min-width: 0; border: 0; background: transparent; color: var(--ink2); font: 500 12.5px/1.25 var(--body); padding: 7px 8px; border-radius: 6px; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--bg); color: var(--ink); font-weight: 650; box-shadow: 0 1px 2px rgba(0,0,0,.12); }
.sw { display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--ink); cursor: pointer; }
.sw input { width: 34px; height: 20px; appearance: none; background: var(--grid); border-radius: 999px; position: relative; cursor: pointer; flex: none; margin: 0; transition: background .15s; }
.sw input::after { content: ""; position: absolute; top: 2px; left: 2px; width: 16px; height: 16px; border-radius: 50%; background: var(--bg); transition: transform .15s; box-shadow: 0 1px 2px rgba(0,0,0,.25); }
.sw input:checked { background: var(--seam); }
.sw input:checked::after { transform: translateX(14px); }
.btn { border: 1px solid var(--grid); background: var(--bg); color: var(--ink); border-radius: 8px; padding: 7px 12px; font: 500 12.5px var(--body); cursor: pointer; justify-self: start; }
button:focus-visible, input:focus-visible, .ucard:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.units { display: grid; gap: 8px; }
.ucard { text-align: left; border: 1px solid var(--line); background: var(--bg); color: var(--ink); border-radius: 9px; padding: 10px 12px; display: grid; gap: 6px; cursor: pointer; font: inherit; }
.ucard[aria-pressed="true"] { border-color: var(--focus); box-shadow: 0 0 0 1px var(--focus); }
.urow { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.uid { font-weight: 800; font-size: 15px; }
.chip { font-size: 11px; padding: 2px 7px; border-radius: 999px; background: var(--panel); color: var(--ink2); }
.chip.seam { color: var(--seam); }
.sq { font: 400 11.5px var(--mono); color: var(--muted); margin-left: auto; }
.meters { display: grid; grid-template-columns: 14px 1fr 46px; gap: 3px 8px; align-items: center; font-size: 12px; font-variant-numeric: tabular-nums; }
.meters b { font-weight: 600; color: var(--ink2); font-size: 11px; }
.meter { height: 6px; background: var(--panel); border-radius: 3px; position: relative; overflow: hidden; }
.meter span { position: absolute; inset: 0 auto 0 0; border-radius: 3px; }
.meter::after { content: ""; position: absolute; left: 50%; top: -1px; bottom: -1px; width: 1px; background: var(--muted); opacity: .6; }
.umeta { font-size: 11.5px; color: var(--ink2); display: flex; gap: 10px; flex-wrap: wrap; font-variant-numeric: tabular-nums; }
.note { font-size: 12px; color: var(--ink2); margin: 0; max-width: 60ch; }
footer { font-size: 11.5px; color: var(--muted); }
@media (prefers-reduced-motion: reduce) { .sw input, .sw input::after { transition: none; } }
</style>
"""
SHOR3D_BODY = r"""<div class="wrap">
  <header>
    <div class="eyebrow" id="t-eyebrow"></div>
    <h1 id="t-h1"></h1>
    <p class="sub" id="t-sub"></p>
  </header>

  <section class="figs" aria-label="Register figures" id="figs"></section>

  <section class="main">
    <div class="stage" id="stage" aria-label="3D view of the register">
      <div class="labels" id="labels"></div>
      <div class="hint">Drag to rotate &#xb7; scroll or pinch to zoom &#xb7; right-drag to pan</div>
      <div class="legend" aria-hidden="true" id="legend"></div>
    </div>

    <aside>
      <div class="ctl">
        <span class="lbl" id="l-setup">Setup</span>
        <div class="seg" role="group" aria-labelledby="l-setup" id="seg-setup"></div>
      </div>
      <div class="ctl">
        <span class="lbl" id="l-theme">Theme</span>
        <div class="seg" role="group" aria-labelledby="l-theme" id="seg-theme">
          <button type="button" id="th-auto" data-t="auto" aria-pressed="true">Auto</button>
          <button type="button" id="th-light" data-t="light" aria-pressed="false">Light</button>
          <button type="button" id="th-dark" data-t="dark" aria-pressed="false">Dark</button>
        </div>
      </div>
      <div class="ctl">
        <span class="lbl" id="l-height">Column height</span>
        <div class="seg" role="group" aria-labelledby="l-height" id="seg-height">
          <button type="button" id="h-x" data-h="x" aria-pressed="true">Logical X</button>
          <button type="button" id="h-z" data-h="z" aria-pressed="false">Logical Z</button>
          <button type="button" id="h-min" data-h="min" aria-pressed="false">Weaker</button>
        </div>
      </div>
      <div class="ctl">
        <label class="sw" for="sw-split"><input type="checkbox" id="sw-split"> Pull the patches apart</label>
        <label class="sw" for="sw-num"><input type="checkbox" id="sw-num" checked> Qubit numbers</label>
        <label class="sw" for="sw-turn"><input type="checkbox" id="sw-turn"> Turn slowly</label>
        <button type="button" class="btn" id="b-reset">Reset view</button>
      </div>
      <p class="note" id="split-note">Pulled apart, each patch drops to its own layer: that is the dense simulator ACE keeps for it. The seam qubits stay on the device plane in the crossbar and reappear as copies in every patch layer they join, linked by green pillars. Couplers drawn in the text colour (black, or white in dark mode) run through such a copy.</p>
      <div class="ctl">
        <span class="lbl">Units</span>
        <div class="units" id="units"></div>
      </div>
    </aside>
  </section>

  <footer id="t-foot"></footer>
</div>
"""
SHOR3D_JS = r"""import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const D = JSON.parse(document.getElementById("d").textContent);
const stage = document.getElementById("stage");
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const M = D.m;            // code size: m x m units
const H = 3.0;            // column height at decoded success 1.0
const LAYER = 2.6;        // drop between layers when pulled apart
const R = 0.26;           // qubit radius
const COLS = D.columns;

// ------------------------------------------------------------------ data
const sites = D.sites.map(([q, r, c, region, sims]) => ({ q, r, c, region, sims: sims || [] }));
const ROWS = Math.max(12, ...sites.map(s => s.r + 1));
const bulkIds = [...new Set(sites.filter(s => s.region !== "seam").map(s => +s.region.slice(1)))].sort((a, b) => a - b);
const hasSims = sites.some(s => s.sims.length);
const pidx = sim => bulkIds.indexOf(sim);                    // patch number 0, 1, ... in display order
const layerOf = sim => (bulkIds.includes(sim) ? pidx(sim) + 1 : 0);   // 0: device plane / crossbar
const ptok = sim => `--pc${pidx(sim) % 6}`;
const joins = s => s.sims.filter(v => bulkIds.includes(v));
const byQ = new Map(sites.map(s => [s.q, s]));
const home = q => { const s = byQ.get(q); return s.region === "seam" ? null : +s.region.slice(1); };
const px = c => c - (COLS - 1) / 2, pz = r => r - (ROWS - 1) / 2;
const state = { setup: 0, h: "x", split: 0, splitTarget: 0, sel: null, nums: true };
const KEY = `nh3d-${D.register}`;
try { const s = JSON.parse(localStorage.getItem(KEY) || "{}"); Object.assign(state, { h: s.h || "x", nums: s.nums ?? true }); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify({ h: state.h, nums: state.nums })); } catch (e) {} };

// ------------------------------------------------------------------ text
const S0 = D.setups[0];
const nUnits = D.places.length, nSeam = sites.filter(s => s.region === "seam").length;
const cnt = { exact: 0, replica: 0, cross: 0 };
for (const [, , k] of D.couplers) cnt[k] = (cnt[k] || 0) + 1;
const bulkCounts = bulkIds.map(p => sites.filter(s => s.region === `p${p}`).length);
const words = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
const w = k => (k < words.length ? words[k] : String(k));
document.getElementById("t-eyebrow").textContent = `rcs/nighthawk \u00b7 nighthawk_qrack.py shor9 --register ${D.register}`
  + (M !== 3 ? ` --size ${M}` : "") + ` \u00b7 stored ACE placement, lrc ${D.ace.lrc} \u00b7 lrr ${D.ace.lrr} \u00b7 nighthawk_graph.py view 17 in three dimensions`;
document.getElementById("t-h1").innerHTML = `${D.register}<span>Shor-${M * M} units on the Nighthawk lattice</span>`;
document.getElementById("t-sub").textContent = `${w(nUnits).replace(/^./, c => c.toUpperCase())} unit${nUnits === 1 ? "" : "s"} of Shor's `
  + `[[${M * M}, 1, ${M}]] code tiled over a ${D.n}-qubit register cut into ${w(bulkIds.length)} ACE patches. Each unit rises from its `
  + `${M} \u00d7 ${M} square: the column's height is the logical information it kept, so a flat square has none left (0.5) and a `
  + `full-height column decodes perfectly (1.0).`;
document.getElementById("figs").innerHTML = [
  [D.n, "qubits"], [bulkIds.length, `ACE patches (${bulkCounts.join(" + ")} bulk)`], [nSeam, "seam qubits"],
  [`${cnt.exact} / ${cnt.replica} / ${cnt.cross}`, "exact / replica / cross couplers"],
  [`${nUnits} \u00b7 ${nUnits * M * M}`, "units \u00b7 qubits used"], ['<span id="f-mean">\u2013</span>', "mean decoded Z / X"],
  ["1.000", "exact reference, every unit"],
].map(([v, k]) => `<div class="fig"><b>${v}</b><span>${k}</span></div>`).join("");
document.getElementById("t-foot").textContent = `Source: twobombs/thereminq-examples rcs/nighthawk, nighthawk_qrack.py shor9 --register ${D.register}`
  + ` (unit records and the shor9_map record of the placement); ` + D.setups.map(s => `${s.label}: ${s.backends} fresh backend${s.backends === 1 ? "" : "s"} \u00d7 ${s.shots} shots per rotation`).join("; ")
  + `; logical states rotated over 4 runs. Height = (decoded success \u2212 0.5) / 0.5. imp: share of |+_L\u27e9, |\u2212_L\u27e9 outcomes the exact state never gives; `
  + `the inner code corrects the single-bit ones. Written by nighthawk_graph.py --grid3d.`;
document.getElementById("legend").innerHTML = bulkIds.map(p => `<span><i style="background:var(${ptok(p)})"></i>patch ${pidx(p)}</span>`).join("")
  + `<span><i style="background:linear-gradient(90deg,var(--pc1) 50%,var(--pc0) 50%);box-shadow:0 0 0 2px var(--ink)"></i>seam qubit</span>`
  + `<span><i class="bar" style="background:var(--seam);opacity:.55;height:7px"></i>seam line</span>`
  + `<span><i class="bar" style="background:var(--ink)"></i>replica coupler</span>`
  + (cnt.cross ? `<span><i class="bar" style="background:var(--cross)"></i>cross coupler</span>` : "")
  + `<span><i class="bar" style="background:var(--pc0);opacity:.55"></i>exact coupler</span><span><i class="ring"></i>block leader</span>`;
if (!hasSims) {
  const sw = document.getElementById("sw-split"); sw.disabled = true;
  document.getElementById("split-note").textContent = "This register map does not say which simulators hold each seam qubit (written before nighthawk_qrack.py stored it); rerun shor9 --register once to pull the patches apart.";
}

// ------------------------------------------------------------------ renderer
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
} catch (e) {
  stage.insertAdjacentHTML("beforeend", '<div class="nogl">This view needs WebGL, which this browser or device does not offer. nighthawk_graph.py view 17 shows the same register flat.</div>');
  throw e;
}
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
stage.prepend(renderer.domElement);
const labels = new CSS2DRenderer({ element: document.getElementById("labels") });
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 400);
const span = Math.max(ROWS, COLS);
const HOME = { pos: new THREE.Vector3(-0.29 * span, 1.04 * span, 1.21 * span), target: new THREE.Vector3(0.6, 0.3, 0.6) };
camera.position.copy(HOME.pos);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(HOME.target);
controls.enableDamping = !reduce;
controls.maxPolarAngle = Math.PI * 0.92;
controls.minDistance = 5; controls.maxDistance = 4 * span + LAYER * bulkIds.length * 2;
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8984, 1.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(-6, 14, 9); scene.add(sun);

const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const themed = [];   // [material, token]
const mat = (token, opts = {}, kind = "std") => {
  const m = kind === "line" ? new THREE.LineBasicMaterial(opts) : kind === "basic" ? new THREE.MeshBasicMaterial(opts)
    : new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.0, ...opts });
  themed.push([m, token]); return m;
};

// ------------------------------------------------------------------ geometry helpers
const UP = new THREE.Vector3(0, 1, 0);
const cylGeo = new THREE.CylinderGeometry(1, 1, 1, 10, 1);
function placeCyl(mesh, a, b, r) {
  const d = new THREE.Vector3().subVectors(b, a), len = d.length();
  mesh.position.copy(a).addScaledVector(d, 0.5);
  mesh.scale.set(r, Math.max(len, 1e-4), r);
  if (len > 1e-6) mesh.quaternion.setFromUnitVectors(UP, d.normalize());
}
const sphGeo = new THREE.SphereGeometry(R, 28, 18);
const halfA = new THREE.SphereGeometry(R + 0.03, 28, 18, 0, Math.PI);
const halfB = new THREE.SphereGeometry(R + 0.03, 28, 18, Math.PI, Math.PI);
const ringGeo = new THREE.TorusGeometry(R + 0.07, 0.035, 8, 40);
const leaderGeo = new THREE.TorusGeometry(R + 0.17, 0.022, 6, 40);
const ease = t => t * t * (3 - 2 * t);

// ------------------------------------------------------------------ the lattice (device plane)
{
  const pts = [];
  for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
    if (c + 1 < COLS) pts.push(px(c), 0, pz(r), px(c + 1), 0, pz(r));
    if (r + 1 < ROWS) pts.push(px(c), 0, pz(r), px(c), 0, pz(r + 1));
  }
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  scene.add(new THREE.LineSegments(g, mat("--grid", {}, "line")));
  const taken = new Set(sites.map(s => s.r * 10 + s.c));
  const dotGeo = new THREE.SphereGeometry(0.06, 10, 8), dotMat = mat("--muted", { transparent: true, opacity: 0.55 }, "basic");
  const xMat = mat("--muted", {}, "basic"), xGeo = new THREE.BoxGeometry(0.42, 0.04, 0.06);
  for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
    const p = r * 10 + c;
    if (taken.has(p)) continue;
    if ([17, 55, 62].includes(p)) {
      for (const a of [Math.PI / 4, -Math.PI / 4]) { const m = new THREE.Mesh(xGeo, xMat); m.position.set(px(c), 0.02, pz(r)); m.rotation.y = a; scene.add(m); }
    } else { const d = new THREE.Mesh(dotGeo, dotMat); d.position.set(px(c), 0, pz(r)); scene.add(d); }
  }
}

// ------------------------------------------------------------------ qubits (and seam copies)
const nodes = [];
const qlabels = [];
const qLayer = q => { const h = home(q); return h == null ? 0 : layerOf(h); };
for (const s of sites) {
  const x = px(s.c), z = pz(s.r);
  const el = document.createElement("div"); el.className = "qlab"; el.textContent = s.q; qlabels.push(el);
  if (s.region !== "seam") {
    const p = +s.region.slice(1);
    const m = new THREE.Mesh(sphGeo, mat(ptok(p)));
    m.position.set(x, 0, z); scene.add(m); m.add(new CSS2DObject(el));
    nodes.push({ obj: m, layer: layerOf(p) });
    continue;
  }
  const j = joins(s);
  const g = new THREE.Group(); g.position.set(x, 0, z);
  g.add(new THREE.Mesh(halfA, mat(j.length ? ptok(j[j.length - 1]) : "--pc1")), new THREE.Mesh(halfB, mat(j.length ? ptok(j[0]) : "--pc0")));
  const ring = new THREE.Mesh(ringGeo, mat("--ink")); ring.rotation.x = Math.PI / 2; g.add(ring);
  g.add(new CSS2DObject(el)); scene.add(g);
  nodes.push({ obj: g, layer: 0 });
  for (const sim of j) {                                       // its copy in each patch it joins
    const cg = new THREE.Group();
    const cm = mat(ptok(sim), { transparent: true, opacity: 0 });
    cg.add(new THREE.Mesh(sphGeo, cm));
    const rm = mat("--seam", { transparent: true, opacity: 0 }); const rr = new THREE.Mesh(ringGeo, rm); rr.rotation.x = Math.PI / 2; cg.add(rr);
    cg.position.set(x, 0, z); scene.add(cg);
    nodes.push({ obj: cg, layer: layerOf(sim), copy: [cm, rm] });
  }
  if (j.length) {
    const pm = mat("--seam", { transparent: true, opacity: 0, depthWrite: false });
    const pillar = new THREE.Mesh(cylGeo, pm); scene.add(pillar);
    nodes.push({ pillar, pm, x, z, depth: LAYER * Math.max(...j.map(layerOf)) });
  }
}

// ------------------------------------------------------------------ couplers (each end in its own layer)
const edges = [];
for (const [a, b, cls] of D.couplers) {
  const la = home(a) != null ? qLayer(a) : home(b) != null ? qLayer(b) : 0;
  const lb = home(b) != null ? qLayer(b) : home(a) != null ? qLayer(a) : 0;
  const bulk = home(a) != null ? home(a) : home(b);
  const token = cls === "replica" ? "--ink" : cls === "cross" ? "--cross" : (bulk != null ? ptok(bulk) : "--ink");
  const m = new THREE.Mesh(cylGeo, mat(token, cls === "exact" ? { transparent: true, opacity: 0.55 } : {}));
  scene.add(m);
  edges.push({ m, a, b, la, lb, r: cls === "exact" ? 0.04 : 0.055 });
}
const qPos = (q, layer, t) => { const s = byQ.get(q); return new THREE.Vector3(px(s.c), -LAYER * layer * t, pz(s.r)); };

// ------------------------------------------------------------------ seam lines (device plane)
{
  const seam = sites.filter(s => s.region === "seam");
  const lm = mat("--seam", { transparent: true, opacity: 0.32, depthWrite: false });
  const blobGeo = new THREE.SphereGeometry(0.46, 20, 12);
  for (let i = 0; i < seam.length; i++) {
    const blob = new THREE.Mesh(blobGeo, lm); blob.scale.y = 0.25;
    blob.position.set(px(seam[i].c), -0.02, pz(seam[i].r)); scene.add(blob);
    for (let k = i + 1; k < seam.length; k++) {
      const A = seam[i], B = seam[k];
      if (Math.max(Math.abs(A.r - B.r), Math.abs(A.c - B.c)) === 1 && joins(A).join() === joins(B).join()) {
        const t = new THREE.Mesh(cylGeo, lm);
        placeCyl(t, new THREE.Vector3(px(A.c), -0.02, pz(A.r)), new THREE.Vector3(px(B.c), -0.02, pz(B.r)), 0.2);
        scene.add(t);
      }
    }
  }
}

// ------------------------------------------------------------------ layer planes (pulled apart)
const planes = [];
if (hasSims) {
  const rs = sites.map(s => s.r), cs = sites.map(s => s.c);
  const wd = Math.max(...cs) - Math.min(...cs) + 1.4, dp = Math.max(...rs) - Math.min(...rs) + 1.4;
  const cx = (px(Math.max(...cs)) + px(Math.min(...cs))) / 2, cz = (pz(Math.max(...rs)) + pz(Math.min(...rs))) / 2;
  const layers = [["crossbar: the seam qubits", "--seam"], ...bulkIds.map(p => [`patch ${pidx(p)} simulator`, ptok(p)])];
  layers.forEach(([nm, tok], i) => {
    const pm = mat(tok, { transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide });
    const pl = new THREE.Mesh(new THREE.PlaneGeometry(wd, dp), pm); pl.rotation.x = -Math.PI / 2; pl.position.set(cx, -LAYER * i - 0.03, cz);
    scene.add(pl);
    const el = document.createElement("div"); el.className = "llab"; el.textContent = nm; el.style.opacity = 0;
    const lab = new CSS2DObject(el); lab.position.set(cx - wd / 2 + 0.15, -LAYER * i, cz + dp / 2 + 0.15); lab.center.set(0, 0); scene.add(lab);
    planes.push({ pm, el, i });
  });
}

// ------------------------------------------------------------------ units (columns)
const STOPS = [[0, "#c8352e"], [0.25, "#ee8a3c"], [0.5, "#f1cf55"], [0.75, "#86c06a"], [1, "#1f9a5a"]];
function vcol(v) {
  const t = Math.min(1, Math.max(0, (v - 0.5) / 0.5));
  for (let i = 1; i < STOPS.length; i++) if (t <= STOPS[i][0]) {
    const [t0, c0] = STOPS[i - 1], [t1, c1] = STOPS[i];
    return new THREE.Color(c0).lerp(new THREE.Color(c1), (t - t0) / (t1 - t0));
  }
  return new THREE.Color(STOPS[STOPS.length - 1][1]);
}
const units = [];
const boxGeo = new THREE.BoxGeometry(1, 1, 1), boxEdges = new THREE.EdgesGeometry(boxGeo);
const leaderMat = mat("--ink", { transparent: true, opacity: 0.85 });
for (const u of D.places) {
  const r0 = u.rows[0], c0 = u.cols[0];
  const cx = px(c0 + (M - 1) / 2), cz = pz(r0 + (M - 1) / 2), side = M - 0.14;
  const fill = new THREE.MeshStandardMaterial({ transparent: true, opacity: 0.34, depthWrite: false, roughness: 0.7 });
  const box = new THREE.Mesh(boxGeo, fill); box.renderOrder = 2;
  const em = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.95 });
  const edge = new THREE.LineSegments(boxEdges, em);
  const gm = mat("--muted", { transparent: true, opacity: 0.35 }, "line");
  const ghost = new THREE.LineSegments(boxEdges, gm);
  ghost.scale.set(side, H, side); ghost.position.set(cx, H / 2, cz);
  const baseMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.22, depthWrite: false, side: THREE.DoubleSide });
  const base = new THREE.Mesh(new THREE.PlaneGeometry(side, side), baseMat); base.rotation.x = -Math.PI / 2; base.position.set(cx, 0.015, cz);
  scene.add(box, edge, ghost, base);
  box.userData.unit = u.id;
  const el = document.createElement("div"); el.className = "ulab";
  el.addEventListener("click", () => select(state.sel === u.id ? null : u.id));
  const lab = new CSS2DObject(el); scene.add(lab);
  for (const b of u.blocks) {
    const lg = new THREE.Mesh(leaderGeo, leaderMat);
    lg.rotation.x = Math.PI / 2; const s = byQ.get(b[0]); lg.position.set(px(s.c), 0, pz(s.r)); scene.add(lg);
    nodes.push({ obj: lg, layer: qLayer(b[0]) });
  }
  units.push({ id: u.id, cx, cz, side, box, edge, fill, em, ghost, gm, base, baseMat, el, lab, h: 0, hTarget: 0 });
}

// ------------------------------------------------------------------ values and UI
const metric = u => (state.h === "x" ? u.x : state.h === "z" ? u.z : Math.min(u.x, u.z));
const fmt = v => v.toFixed(3);
const unitData = id => D.setups[state.setup].units.find(u => u.id === id);
function applyValues(instant) {
  for (const U of units) {
    const u = unitData(U.id);
    if (!u) { U.hTarget = 0.03; if (instant) U.h = U.hTarget; U.el.textContent = `${U.id} not measured in this setup`; continue; }
    const v = metric(u);
    U.hTarget = Math.max(0.03, H * (v - 0.5) / 0.5);
    if (instant) U.h = U.hTarget;
    const col = vcol(v);
    U.fill.color.copy(col); U.em.color.copy(col).multiplyScalar(0.72); U.baseMat.color.copy(col);
    U.el.innerHTML = `${U.id} <em>Z</em> ${fmt(u.z)} <em>\u00b7 X</em> ${fmt(u.x)}`;
    U.el.style.borderColor = "#" + col.clone().multiplyScalar(0.8).getHexString();
  }
  const s = D.setups[state.setup];
  const mz = s.units.reduce((a, u) => a + u.z, 0) / s.units.length, mx = s.units.reduce((a, u) => a + u.x, 0) / s.units.length;
  document.getElementById("f-mean").textContent = `${fmt(mz)} / ${fmt(mx)}`;
  renderCards();
}
function setupName(s) {
  const py = (s.label.match(/pyqrack ([\d.]+)/) || [])[1] || "?";
  return `PyQrack ${py}` + (s.label.includes("boundary code") ? " + boundary code" : "") + (s.label.includes("OpenCL") ? " \u00b7 OpenCL" : "");
}
const segSetup = document.getElementById("seg-setup");
D.setups.forEach((s, i) => {
  const b = document.createElement("button"); b.type = "button"; b.id = `setup-${i}`; b.textContent = setupName(s);
  b.setAttribute("aria-pressed", i === state.setup); b.title = `${s.label} \u00b7 ${s.backends} fresh backends \u00d7 ${s.shots} shots per rotation`;
  b.addEventListener("click", () => { state.setup = i; [...segSetup.children].forEach((x, k) => x.setAttribute("aria-pressed", k === i)); applyValues(reduce); });
  segSetup.appendChild(b);
});
document.querySelectorAll("#seg-height button").forEach(b => {
  b.setAttribute("aria-pressed", b.dataset.h === state.h);
  b.addEventListener("click", () => {
    state.h = b.dataset.h; save();
    document.querySelectorAll("#seg-height button").forEach(x => x.setAttribute("aria-pressed", x === b));
    applyValues(reduce);
  });
});
const swSplit = document.getElementById("sw-split"), swNum = document.getElementById("sw-num");
swSplit.addEventListener("change", () => { state.splitTarget = swSplit.checked ? 1 : 0; if (reduce) state.split = state.splitTarget; });
swNum.checked = state.nums;
const applyNums = () => qlabels.forEach(el => (el.style.visibility = state.nums ? "visible" : "hidden"));
swNum.addEventListener("change", () => { state.nums = swNum.checked; save(); applyNums(); });
applyNums();
let lastT = 0;
document.getElementById("b-reset").addEventListener("click", () => {
  const dy = -LAYER * 0.45 * bulkIds.length * lastT, z = 1 + 0.09 * bulkIds.length * lastT;
  controls.target.copy(HOME.target).y += dy;
  camera.position.copy(HOME.pos).sub(HOME.target).multiplyScalar(z).add(controls.target); controls.update();
});

function renderCards() {
  const box = document.getElementById("units");
  box.innerHTML = "";
  for (const u of D.setups[state.setup].units) {
    const c = document.createElement("button"); c.type = "button"; c.className = "ucard"; c.id = `card-${u.id}`;
    c.setAttribute("aria-pressed", state.sel === u.id);
    const bar = v => `<div class="meter"><span style="width:${(v * 100).toFixed(1)}%;background:#${vcol(v).getHexString()}"></span></div>`;
    const regs = Object.entries(u.regions).map(([k, v]) => `${k} ${v}`).join(", ");
    c.innerHTML = `<div class="urow"><span class="uid">${u.id}</span><span class="chip ${u.kind === "one patch" ? "" : "seam"}">${u.kind}</span>`
      + `<span class="sq">r${u.rows[0]}\u2013${u.rows[1]} c${u.cols[0]}\u2013${u.cols[1]}</span></div>`
      + `<div class="meters"><b>Z</b>${bar(u.z)}<span>${fmt(u.z)}</span><b>X</b>${bar(u.x)}<span>${fmt(u.x)}</span></div>`
      + `<div class="umeta"><span>${regs}</span><span>CX ${u.cx[0]}/${u.cx[1]}/${u.cx[2]}</span>`
      + `<span>imp ${u.imp == null ? "\u2013" : (100 * u.imp).toFixed(1) + " %"}</span></div>`;
    c.addEventListener("click", () => select(state.sel === u.id ? null : u.id));
    box.appendChild(c);
  }
}
function select(id) {
  state.sel = id;
  document.querySelectorAll(".ucard").forEach(c => c.setAttribute("aria-pressed", c.id === `card-${id}`));
  for (const U of units) { const on = U.id === id; U.el.classList.toggle("sel", on); U.fill.emissive.set(on ? 0x333333 : 0x000000); }
}
const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
renderer.domElement.addEventListener("click", e => {
  const r = renderer.domElement.getBoundingClientRect();
  ptr.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ptr, camera);
  const hit = ray.intersectObjects(units.map(U => U.box), false)[0];
  if (hit) select(hit.object.userData.unit);
});

// ------------------------------------------------------------------ theme
function applyTheme() {
  for (const [m, token] of themed) m.color.set(css(token) || "#888");
  scene.background = new THREE.Color(css("--stage") || "#f0efeb");
}
new MutationObserver(applyTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
// Theme switch: Auto keeps whatever the page started with (the system setting, or a host's own choice)
const hostTheme = document.documentElement.getAttribute("data-theme");
function setTheme(t, keep) {
  if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
  else if (hostTheme) document.documentElement.setAttribute("data-theme", hostTheme);
  else document.documentElement.removeAttribute("data-theme");
  document.querySelectorAll("#seg-theme button").forEach(b => b.setAttribute("aria-pressed", b.dataset.t === t));
  if (keep) { try { localStorage.setItem("nh3d-theme", t); } catch (e) {} }
}
document.querySelectorAll("#seg-theme button").forEach(b => b.addEventListener("click", () => setTheme(b.dataset.t, true)));
let savedTheme = "auto";
try { savedTheme = localStorage.getItem("nh3d-theme") || "auto"; } catch (e) {}
setTheme(savedTheme, false);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);

// ------------------------------------------------------------------ frame
function layout() {
  const t = ease(state.split);
  const follow = LAYER * 0.45 * bulkIds.length * (t - lastT);   // keep the stack in view, keep the user's own pan
  controls.target.y -= follow; camera.position.y -= follow;
  const zoom = f => 1 + 0.09 * bulkIds.length * f;               // and step back as the stack grows
  if (t !== lastT) camera.position.sub(controls.target).multiplyScalar(zoom(t) / zoom(lastT)).add(controls.target);
  lastT = t;
  for (const n of nodes) {
    if (n.pillar) {
      n.pm.opacity = 0.55 * t; n.pillar.visible = t > 0.01;
      placeCyl(n.pillar, new THREE.Vector3(n.x, 0, n.z), new THREE.Vector3(n.x, -n.depth * t, n.z), 0.05);
      continue;
    }
    n.obj.position.y = -LAYER * n.layer * t;
    if (n.copy) { n.copy[0].opacity = t; n.copy[1].opacity = t; n.obj.visible = t > 0.01; }
  }
  for (const e of edges) placeCyl(e.m, qPos(e.a, e.la, t), qPos(e.b, e.lb, t), e.r);
  for (const p of planes) { p.pm.opacity = (p.i === 0 ? 0.10 : 0.08) * t; p.el.style.opacity = t; }
  for (const U of units) {
    U.box.scale.set(U.side, U.h, U.side); U.box.position.set(U.cx, U.h / 2 + 0.02, U.cz);
    U.edge.scale.copy(U.box.scale); U.edge.position.copy(U.box.position);
    U.fill.opacity = 0.34 * (1 - 0.8 * t); U.em.opacity = 0.95 * (1 - 0.65 * t); U.gm.opacity = 0.35 * (1 - 0.7 * t);
    U.lab.position.set(U.cx, Math.max(U.h, 0.4) + 0.45, U.cz);
  }
}
function resize() {
  const wd = stage.clientWidth, ht = stage.clientHeight;
  renderer.setSize(wd, ht, false); renderer.domElement.style.width = wd + "px"; renderer.domElement.style.height = ht + "px";
  labels.setSize(wd, ht); camera.aspect = wd / ht; camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);
let prev = performance.now();
function frame(now) {
  const dt = Math.min(0.1, Math.max(0, (now - prev) / 1000)); prev = now;
  const step = dt / (0.6 + 0.15 * bulkIds.length);
  if (Math.abs(state.split - state.splitTarget) > step) state.split += Math.sign(state.splitTarget - state.split) * step;
  else state.split = state.splitTarget;
  const k = 1 - Math.exp(-dt / 0.12);
  for (const U of units) if (Math.abs(U.h - U.hTarget) > 1e-3) U.h += (U.hTarget - U.h) * k; else U.h = U.hTarget;
  layout();
  controls.update();
  renderer.render(scene, camera); labels.render(scene, camera);
  requestAnimationFrame(frame);
}
// turning: the switch, and a hook for scripted captures (camera at a fixed angle about the target)
const swTurn = document.getElementById("sw-turn");
controls.autoRotateSpeed = -0.8;
swTurn.addEventListener("change", () => { controls.autoRotate = swTurn.checked; });
const sph = new THREE.Spherical();
let theta0 = null;
window.nh3d = {
  camera, controls,
  orbit(deg) {
    sph.setFromVector3(camera.position.clone().sub(controls.target));
    if (theta0 === null) theta0 = sph.theta;
    sph.theta = theta0 + (deg * Math.PI) / 180;
    camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(sph));
    controls.update();
  },
  frame: () => new Promise(res => requestAnimationFrame(() => requestAnimationFrame(res))),
};
applyTheme(); resize(); applyValues(true); select(null);
requestAnimationFrame(frame);
window.__ready = true;
"""


def _setup_rank(setup):
    return ("boundary code" in setup, "OpenCL" in setup, setup)


def shor3d_payload(store, reg, m):
    """Everything the 3D page of one register and code size shows: the placement (sites with their ACE
    simulators, couplers by class), the union of unit squares measured on it, and per setup each unit's
    decoded Z / X, Z-basis impossible share, encoder CX classes; None without a register map."""
    mp = shor_register_map(store, reg)
    if not mp:
        return None
    un = {k: u for k, u in shor_units(store.shor).items() if k[1] == reg and k[2] == m}
    if not un:
        return None
    places = {}
    for (_, _, _, place), u in un.items():
        places.setdefault(place, dict(rows=u["rows"], cols=u["cols"], blocks=u["blocks"]))
    order = sorted(places, key=lambda p: (places[p]["rows"][0], places[p]["cols"][0]))
    ids = {p: f"U{i + 1}" for i, p in enumerate(order)}
    setups = []
    for su in sorted({k[0] for k in un}, key=_setup_rank):
        rs = [r for r in store.shor if r.get("register") == reg and int(r.get("size", 3)) == m and shor_setup(r)[0] == su]
        units = []
        for (s_, _, _, place), u in sorted(un.items(), key=lambda kv: ids[kv[0][3]]):
            if s_ != su:
                continue
            imp = [u["dist"][x]["impossible"] for x in ("+", "-") if u["dist"].get(x)]
            units.append(dict(id=ids[place], rows=u["rows"], cols=u["cols"], z=round(float(u["z"]), 4),
                              x=round(float(u["x"]), 4), kind=u["kind"], regions=u["regions"],
                              cx=[u["exact_cx"], u["replica"], u["cross"]],
                              imp=round(float(np.mean(imp)), 4) if imp else None))
        setups.append(dict(label=su, backends=max([int(r.get("repeats", 1)) for r in rs] or [1]),
                           shots=rs[0].get("shots") if rs else None, units=units))
    sites = [list(st[:4]) + [list(st[4]) if len(st) > 4 else []] for st in mp["sites"]]
    return dict(register=reg, m=int(m), n=int(mp.get("n", len(sites))), columns=int(mp.get("columns", 10)),
                ace=mp.get("ace") or dict(lrc="?", lrr="?"), sites=sites, couplers=mp["couplers"],
                places=[dict(id=ids[p], **places[p]) for p in order], setups=setups)


def shor3d_html(payload, three_base=THREE_CDN, document=True):
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")
    title = f"{payload['register']} Shor-{payload['m'] ** 2} in 3D"
    imports = json.dumps({"imports": {"three": three_base + "build/three.module.js",
                                      "three/addons/": three_base + "examples/jsm/"}})
    if not document:                                   # a fragment for a host that adds its own document
        return (SHOR3D_HEAD.replace("__TITLE__", title) + SHOR3D_BODY
                + '<script type="application/json" id="d">' + data + "</script>\n"
                + '<script type="importmap">' + imports + "</script>\n"
                + '<script type="module">\n' + SHOR3D_JS + "</script>\n")
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            "<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}"
            "body{margin:0}[hidden]{display:none!important}</style>\n"
            + SHOR3D_HEAD.replace("__TITLE__", title) + "</head><body>\n" + SHOR3D_BODY
            + '<script type="application/json" id="d">' + data + "</script>\n"
            + '<script type="importmap">' + imports + "</script>\n"
            + '<script type="module">\n' + SHOR3D_JS + "</script>\n</body></html>\n")


def _copy_three(src, outdir):
    """Copy the three files the page imports out of an npm `three` package (offline use, with --serve)."""
    src = Path(src)
    need = ["build/three.module.js", "examples/jsm/controls/OrbitControls.js", "examples/jsm/renderers/CSS2DRenderer.js"]
    missing = [p for p in need if not (src / p).exists()]
    if missing:
        raise SystemExit(f"--three {src}: not an npm three package (missing {', '.join(missing)}); "
                         "get one with: npm pack three@0.160.0 && tar xzf three-0.160.0.tgz")
    import shutil
    for p in need:
        (Path(outdir) / "three" / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / p, Path(outdir) / "three" / p)
    return "./three/"                                   # import maps need ./ (a bare name is ignored)


def write_shor3d(store, outdir, three=None):
    """One page per register and code size with shor9 --register records, one per register with ghz records,
    plus index.html; returns the list of written pages (the index first when there are several)."""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    base = THREE_CDN
    if three:
        base = three if str(three).startswith(("http://", "https://")) else _copy_three(three, outdir)
        if not base.endswith("/"):
            base += "/"
    pages = []
    for reg, m in sorted({(k[1], k[2]) for k in shor_units(store.shor)}):
        pl = shor3d_payload(store, reg, m)
        if pl is None:
            print(f"# 3D: no register map of {reg}; skipped")
            continue
        p = Path(outdir) / f"{reg}_shor{m * m}_3d.html"
        p.write_text(shor3d_html(pl, base))
        pages.append(p)
        print("wrote", p)
    for reg in sorted({r["register"] for r in store.ghz}):          # nighthawk_qrack.py ghz records: one page per register
        pl = ghz3d_payload(store, reg)
        if pl is None:
            print(f"# 3D: no register map of {reg} for its ghz records; skipped")
            continue
        p = Path(outdir) / f"{reg}_ghz_3d.html"
        p.write_text(ghz3d_html(pl, base))
        pages.append(p)
        print("wrote", p)
    if len(pages) > 1:
        idx = Path(outdir) / "index.html"
        rows = "".join(f'<li><a href="{p.name}">{p.stem.replace("_3d", "").replace("_", " ")}</a></li>' for p in pages)
        idx.write_text("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>nighthawk in 3D</title>"
                       "<style>body{font:15px/1.5 Inter,Helvetica,Arial,sans-serif;margin:32px;background:#fcfcfb;color:#0b0b0b}"
                       "a{color:#2a78d6}@media(prefers-color-scheme:dark){body{background:#121315;color:#ecebe7}a{color:#7fb2f5}}</style>"
                       f"</head><body><h1>nighthawk in 3D</h1><ul>{rows}</ul></body></html>\n")
        pages.insert(0, idx)
    return pages


def serve_dir(folder, port=0):
    """Serve a folder on 127.0.0.1 from a daemon thread (module scripts do not load from file://);
    returns the base URL."""
    import functools
    import http.server
    import threading
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **k):
            pass
    h = functools.partial(Quiet, directory=str(folder))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", int(port)), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}/"


def open_shor3d(store, folder=None, three=None, serve=False, port=0):
    """Write the 3D pages and open the index (or the one page) in the default browser."""
    import tempfile
    import webbrowser
    folder = folder or tempfile.mkdtemp(prefix="nighthawk_3d_")
    pages = write_shor3d(store, folder, three)
    if not pages:
        return None
    if serve or (three and not str(three).startswith(("http://", "https://"))):
        url = serve_dir(folder, port) + pages[0].name
    else:
        url = pages[0].resolve().as_uri()
    webbrowser.open(url)
    return url


# ================================================================== ghz in 3D (--grid3d, viewer button)
# A page per register with nighthawk_qrack.py ghz records, on the shor9 page's scene (the lattice, ACE patches, seam
# lines, couplers by class, the switch that pulls the patches apart into ACE's simulators, theme, camera): its
# JavaScript is SHOR3D_JS cut at its section markers, with the text, the columns and the controls replaced.
# Every site is a column: for tiles its height is the mean GHZ fidelity of the units holding it (views 18-19),
# for a chip-wide state the share of Z shots where the site agrees with the state's majority (view 21); both
# are drawn as (v - 0.5) / 0.5, so a flat site has nothing left above the 0.5 line. A data-set switch picks
# the Layer 2 mode (or the chip-wide tree), a height switch ACE or the exact reference; clicking a site lists
# the states that hold it.
def _js_cut(src, start=None, end=None):
    i = src.index(start) if start else 0
    j = src.index(end) if end else len(src)
    return src[i:j]


_M_TEXT = "// ------------------------------------------------------------------ text"
_M_RENDER = "// ------------------------------------------------------------------ renderer"
_M_UNITS = "// ------------------------------------------------------------------ units (columns)"
_M_THEME = "// ------------------------------------------------------------------ theme"

GHZ3D_TEXT_JS = r"""// ------------------------------------------------------------------ text
const nSeam = sites.filter(s => s.region === "seam").length;
const cnt = { exact: 0, replica: 0, cross: 0 };
for (const [, , k] of D.couplers) cnt[k] = (cnt[k] || 0) + 1;
const bulkCounts = bulkIds.map(p => sites.filter(s => s.region === `p${p}`).length);
document.getElementById("t-eyebrow").textContent = `rcs/nighthawk · nighthawk_qrack.py ghz --register ${D.register}`
  + ` · stored ACE placement, lrc ${D.ace.lrc} · lrr ${D.ace.lrr} · nighthawk_graph.py views 18 and 21 in three dimensions`;
document.getElementById("t-h1").innerHTML = `${D.register}<span>GHZ states on the Nighthawk lattice</span>`;
document.getElementById("t-sub").textContent = `Every site of the ${D.n}-qubit register rises by what the GHZ states holding it kept: `
  + `for tiled chains the mean GHZ fidelity of the chains through the site, for a chip-wide state the share of shots where the site `
  + `agrees with the state's majority. A flat site is at 0.5 or below (no genuine entanglement left); a full column is 1.0. `
  + `The register is cut into ${bulkIds.length} ACE patches; the seam qubits sit on the green line.`;
document.getElementById("figs").innerHTML = [
  [D.n, "qubits"], [bulkIds.length, `ACE patches (${bulkCounts.join(" + ")} bulk)`], [nSeam, "seam qubits"],
  [`${cnt.exact} / ${cnt.replica} / ${cnt.cross}`, "exact / replica / cross couplers"],
  ['<span id="f-units">–</span>', '<span id="f-units-l">states</span>'],
  ['<span id="f-mean">–</span>', '<span id="f-mean-l">off the seam / on it</span>'],
  ['<span id="f-exact">–</span>', "exact reference, same"],
].map(([v, k]) => `<div class="fig"><b>${v}</b><span>${k}</span></div>`).join("");
document.getElementById("t-foot").textContent = `Source: twobombs/thereminq-examples rcs/nighthawk, nighthawk_qrack.py ghz --register ${D.register}`
  + ` (ghz records and the ghz_map record of the placement); ` + D.datasets.map(d => `${d.label}: ${d.runs} run${d.runs === 1 ? "" : "s"}, `
  + `${d.shots} shots × ${d.realisations} realisation${d.realisations === 1 ? "" : "s"} per setting`).join("; ")
  + `. Fidelity: F_GHZ = (P0 + P1 + C)/2 of the formal proof's Lemma 4 where measured, else the Clifford witness; values above 1 are shot noise, `
  + `drawn as 1. Height = (value − 0.5) / 0.5. Written by nighthawk_graph.py --grid3d.`;
document.getElementById("legend").innerHTML = bulkIds.map(p => `<span><i style="background:var(${ptok(p)})"></i>patch ${pidx(p)}</span>`).join("")
  + `<span><i style="background:linear-gradient(90deg,var(--pc1) 50%,var(--pc0) 50%);box-shadow:0 0 0 2px var(--ink)"></i>seam qubit</span>`
  + `<span><i class="bar" style="background:var(--seam);opacity:.55;height:7px"></i>seam line</span>`
  + `<span><i class="bar" style="background:var(--ink)"></i>replica coupler</span>`
  + (cnt.cross ? `<span><i class="bar" style="background:var(--cross)"></i>cross coupler</span>` : "")
  + `<span><i class="bar" style="background:var(--pc0);opacity:.55"></i>exact coupler</span>`
  + `<span><i class="bar" style="background:linear-gradient(90deg,#c8352e,#f1cf55,#1f9a5a);height:7px"></i>column: 0.5 → 1.0</span>`;
if (!hasSims) {
  const sw = document.getElementById("sw-split"); sw.disabled = true;
  document.getElementById("split-note").textContent = "This register map does not say which simulators hold each seam qubit; rerun nighthawk_qrack.py ghz once to pull the patches apart.";
}

"""

GHZ3D_UNITS_JS = r"""// ------------------------------------------------------------------ site columns
const STOPS = [[0, "#c8352e"], [0.25, "#ee8a3c"], [0.5, "#f1cf55"], [0.75, "#86c06a"], [1, "#1f9a5a"]];
function vcol(v) {
  const t = Math.min(1, Math.max(0, (v - 0.5) / 0.5));
  for (let i = 1; i < STOPS.length; i++) if (t <= STOPS[i][0]) {
    const [t0, c0] = STOPS[i - 1], [t1, c1] = STOPS[i];
    return new THREE.Color(c0).lerp(new THREE.Color(c1), (t - t0) / (t1 - t0));
  }
  return new THREE.Color(STOPS[STOPS.length - 1][1]);
}
const units = [];                 // one column per site; the frame loop animates them as it does Shor units
const boxGeo = new THREE.BoxGeometry(1, 1, 1), boxEdges = new THREE.EdgesGeometry(boxGeo);
const SIDE = 0.6;
for (const s of sites) {
  const cx = px(s.c), cz = pz(s.r);
  const fill = new THREE.MeshStandardMaterial({ transparent: true, opacity: 0.34, depthWrite: false, roughness: 0.7 });
  const box = new THREE.Mesh(boxGeo, fill); box.renderOrder = 2;
  const em = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.95 });
  const edge = new THREE.LineSegments(boxEdges, em);
  const gm = mat("--muted", { transparent: true, opacity: 0.35 }, "line");
  const ghost = new THREE.LineSegments(boxEdges, gm); ghost.visible = false;
  const baseMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.22, depthWrite: false, side: THREE.DoubleSide });
  const base = new THREE.Mesh(new THREE.PlaneGeometry(SIDE, SIDE), baseMat); base.rotation.x = -Math.PI / 2; base.position.set(cx, 0.015, cz);
  scene.add(box, edge, ghost, base);
  box.userData.unit = s.q;
  const el = document.createElement("div"); el.className = "ulab";
  el.addEventListener("click", () => select(state.sel === s.q ? null : s.q));
  const lab = new CSS2DObject(el); lab.visible = false; scene.add(lab);   // CSS2DRenderer shows it only when visible
  units.push({ id: s.q, cx, cz, side: SIDE, box, edge, fill, em, ghost, gm, base, baseMat, el, lab, h: 0, hTarget: 0 });
}

// ------------------------------------------------------------------ values and UI
const fmt = v => (v == null ? "–" : v.toFixed(3));
const ds = () => D.datasets[state.setup];
const siteVal = (q, which) => { const v = ds().sites[q]; return v ? v[which || (state.h === "exact" ? "exact" : "ace")] : null; };
const isChip = () => ds().kind === "chip";
const GREY = new THREE.Color("#9a9993");
function applyValues(instant) {
  const d = ds();
  for (const U of units) {
    const v = siteVal(U.id);
    U.hTarget = v == null ? 0.03 : Math.max(0.03, H * (Math.min(v, 1) - 0.5) / 0.5);
    if (instant) U.h = U.hTarget;
    const col = v == null ? GREY : vcol(v);
    U.fill.color.copy(col); U.em.color.copy(col).multiplyScalar(0.72); U.baseMat.color.copy(col);
    const a = siteVal(U.id, "ace"), e = siteVal(U.id, "exact");
    U.el.innerHTML = `q${U.id} <em>${isChip() ? "agrees" : "F"}</em> ${fmt(a)}`
      + (e != null ? ` <em>· exact</em> ${fmt(e)}` : "");
    U.el.style.borderColor = "#" + col.clone().multiplyScalar(0.8).getHexString();
  }
  document.getElementById("f-units").textContent = isChip() ? `${d.chip.tree} tree` : `${d.n_units} · ${d.runs}`;
  document.getElementById("f-units-l").textContent = isChip() ? `chip-wide state, root q${d.chip.root}` : `${d.k}-qubit chains · runs`;
  document.getElementById("f-mean").textContent = isChip() ? `${fmt(d.chip.pop)} · ${fmt(d.chip.F)}` : `${fmt(d.off)} / ${fmt(d.on)}`;
  document.getElementById("f-mean-l").textContent = isChip() ? `P0 + P1 · ${d.chip.kF}` : `mean ${d.key}: off the seam / on it`;
  document.getElementById("f-exact").textContent = isChip() ? fmt(d.chip.xF) : `${fmt(d.xoff)} / ${fmt(d.xon)}`;
  const hx = document.getElementById("h-exact"); hx.disabled = isChip(); hx.title = isChip() ? "no per-site exact numbers for a chip-wide state (its exact reference is ideal or dephased as a whole)" : "";
  if (isChip() && state.h === "exact") { state.h = "ace"; document.querySelectorAll("#seg-height button").forEach(x => x.setAttribute("aria-pressed", x.dataset.h === "ace")); applyValues(instant); return; }
  renderCards();
}
const segSetup = document.getElementById("seg-setup");
D.datasets.forEach((d, i) => {
  const b = document.createElement("button"); b.type = "button"; b.id = `setup-${i}`; b.textContent = d.short;
  b.setAttribute("aria-pressed", i === state.setup); b.title = d.label;
  b.addEventListener("click", () => { state.setup = i; [...segSetup.children].forEach((x, k) => x.setAttribute("aria-pressed", k === i)); applyValues(reduce); });
  segSetup.appendChild(b);
});
document.querySelectorAll("#seg-height button").forEach(b => {
  b.setAttribute("aria-pressed", b.dataset.h === state.h);
  b.addEventListener("click", () => {
    if (b.disabled) return;
    state.h = b.dataset.h; save();
    document.querySelectorAll("#seg-height button").forEach(x => x.setAttribute("aria-pressed", x === b));
    applyValues(reduce);
  });
});
const swSplit = document.getElementById("sw-split"), swNum = document.getElementById("sw-num");
swSplit.addEventListener("change", () => { state.splitTarget = swSplit.checked ? 1 : 0; if (reduce) state.split = state.splitTarget; });
swNum.checked = state.nums;
const applyNums = () => qlabels.forEach(el => (el.style.visibility = state.nums ? "visible" : "hidden"));
swNum.addEventListener("change", () => { state.nums = swNum.checked; save(); applyNums(); });
applyNums();
let lastT = 0;
document.getElementById("b-reset").addEventListener("click", () => {
  const dy = -LAYER * 0.45 * bulkIds.length * lastT, z = 1 + 0.09 * bulkIds.length * lastT;
  controls.target.copy(HOME.target).y += dy;
  camera.position.copy(HOME.pos).sub(HOME.target).multiplyScalar(z).add(controls.target); controls.update();
});

const bar = v => (v == null ? `<div class="meter"></div>` :
  `<div class="meter"><span style="width:${(Math.min(1, Math.max(0, v)) * 100).toFixed(1)}%;background:#${vcol(v).getHexString()}"></span></div>`);
function siteCard(q, box, pressed) {
  const s = byQ.get(q), a = siteVal(q, "ace"), e = siteVal(q, "exact"), n = (ds().sites[q] || {}).n || 0;
  const c = document.createElement("button"); c.type = "button"; c.className = "ucard"; c.id = `card-${q}`;
  c.setAttribute("aria-pressed", !!pressed);
  c.innerHTML = `<div class="urow"><span class="uid">q${q}</span><span class="chip ${s.region === "seam" ? "seam" : ""}">${s.region === "seam" ? "seam qubit" : "patch " + pidx(+s.region.slice(1))}</span>`
    + `<span class="sq">r${s.r} c${s.c}</span></div>`
    + `<div class="meters"><b>A</b>${bar(a)}<span>${fmt(a)}</span>` + (isChip() ? "" : `<b>E</b>${bar(e)}<span>${fmt(e)}</span>`) + `</div>`
    + `<div class="umeta"><span>${isChip() ? "agrees with the majority (Z)" : `mean of ${n} chain${n === 1 ? "" : "s"}`}</span><span>A ACE · E exact</span></div>`;
  c.addEventListener("click", () => select(state.sel === q ? null : q));
  box.appendChild(c);
}
function renderCards() {
  const box = document.getElementById("units");
  box.innerHTML = "";
  const d = ds();
  const lab = document.getElementById("l-units");
  if (isChip()) {
    const ch = d.chip;
    const c = document.createElement("div"); c.className = "ucard";
    c.innerHTML = `<div class="urow"><span class="uid">${ch.tree} tree</span><span class="chip">root q${ch.root}</span><span class="sq">depth ${ch.depth}</span></div>`
      + `<div class="meters"><b>P</b>${bar(ch.pop)}<span>${fmt(ch.pop)}</span><b>F</b>${bar(ch.F)}<span>${fmt(ch.F)}</span></div>`
      + `<div class="umeta"><span>P = P0 + P1 · F = ${ch.kF}</span><span>exact ${fmt(ch.xF)}</span><span>CX ${ch.cx[0]}/${ch.cx[1]}/${ch.cx[2]}</span></div>`
      + `<div class="umeta"><span>breaking tree CXs: ${ch.edges.filter(e => e[3] >= 0.05).map(e => `${e[0]}–${e[1]} ${e[3].toFixed(2)}`).join(", ") || "none"}</span></div>`;
    box.appendChild(c);
  }
  if (state.sel != null) {
    lab.textContent = isChip() ? "Selected site" : "Selected site and the chains through it";
    siteCard(state.sel, box, true);
    if (!isChip()) {
      for (const u of d.units.filter(u => u.qubits.includes(state.sel))) {
        const c = document.createElement("div"); c.className = "ucard";
        c.innerHTML = `<div class="urow"><span class="uid">${u.qubits.join("–")}</span><span class="chip ${u.place.startsWith("on") ? "seam" : ""}">${u.place}</span>`
          + `<span class="sq">${u.tiling}</span></div>`
          + `<div class="meters"><b>A</b>${bar(u.F)}<span>${fmt(u.F)}</span><b>E</b>${bar(u.xF)}<span>${fmt(u.xF)}</span></div>`
          + `<div class="umeta"><span>C ${fmt(u.C)}</span><span>Φ ${u.Phi == null ? "–" : u.Phi.toFixed(2)}</span><span>CX ${u.cx[0]}/${u.cx[1]}/${u.cx[2]}</span></div>`;
        box.appendChild(c);
      }
    }
    return;
  }
  lab.textContent = "Lowest sites (click one, or a column)";
  const order = sites.map(s => s.q).filter(q => siteVal(q) != null).sort((a, b) => siteVal(a) - siteVal(b)).slice(0, 8);
  for (const q of order) siteCard(q, box, false);
}
function select(id) {
  state.sel = id;
  for (const U of units) { const on = U.id === id; U.el.classList.toggle("sel", on); U.lab.visible = on; U.fill.emissive.set(on ? 0x333333 : 0x000000); }
  renderCards();
}
const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
renderer.domElement.addEventListener("click", e => {
  const r = renderer.domElement.getBoundingClientRect();
  ptr.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ptr, camera);
  const hit = ray.intersectObjects(units.map(U => U.box), false)[0];
  if (hit) select(hit.object.userData.unit);
});

"""


def _ghz3d_parts():
    """GHZ3D_JS, GHZ3D_BODY and GHZ3D_HEAD from the shor9 page's (checked once: a change to SHOR3D_* that
    moves a marker stops here, not in the browser)."""
    pre = _js_cut(SHOR3D_JS, None, _M_TEXT)
    for a, b in (('const KEY = `nh3d-${D.register}`;', 'const KEY = `nh3d-ghz-${D.register}`;'),
                 ('const state = { setup: 0, h: "x",', 'const state = { setup: 0, h: "ace",'),
                 ('Object.assign(state, { h: s.h || "x", nums: s.nums ?? true });',
                  'Object.assign(state, { h: ["ace", "exact"].includes(s.h) ? s.h : "ace", nums: s.nums ?? true });')):
        if pre.count(a) != 1:
            raise RuntimeError(f"ghz 3D: the shor9 page's script changed ({a!r})")
        pre = pre.replace(a, b)
    scene = _js_cut(SHOR3D_JS, _M_RENDER, _M_UNITS).replace(
        "nighthawk_graph.py view 17 shows the same register flat.", "nighthawk_graph.py views 18 and 21 show the same register flat.")
    tail = _js_cut(SHOR3D_JS, _M_THEME, None)
    js = pre + GHZ3D_TEXT_JS + scene + GHZ3D_UNITS_JS + tail
    body = SHOR3D_BODY
    for a, b in (('<span class="lbl" id="l-setup">Setup</span>\n        <div class="seg" role="group" aria-labelledby="l-setup" id="seg-setup"></div>',
                  '<span class="lbl" id="l-setup">Data set</span>\n        <div class="seg wrap" role="group" aria-labelledby="l-setup" id="seg-setup"></div>'),
                 ('''          <button type="button" id="h-x" data-h="x" aria-pressed="true">Logical X</button>
          <button type="button" id="h-z" data-h="z" aria-pressed="false">Logical Z</button>
          <button type="button" id="h-min" data-h="min" aria-pressed="false">Weaker</button>''',
                  '''          <button type="button" id="h-ace" data-h="ace" aria-pressed="true">ACE</button>
          <button type="button" id="h-exact" data-h="exact" aria-pressed="false">Exact reference</button>'''),
                 ('<span class="lbl">Units</span>', '<span class="lbl" id="l-units">Sites</span>'),
                 ('aria-label="3D view of the register"', 'aria-label="3D view of the register and its GHZ states"')):
        if body.count(a) != 1:
            raise RuntimeError(f"ghz 3D: the shor9 page's layout changed ({a[:40]!r})")
        body = body.replace(a, b)
    head = SHOR3D_HEAD.replace("</style>", ".seg.wrap { flex-wrap: wrap; }\n.seg.wrap button { flex: 1 1 44%; }\n"
                               ".seg button:disabled { opacity: .45; cursor: not-allowed; }\n</style>", 1)
    return js, body, head


GHZ_MODE_SHORT = {"none": "none", "unmitigated": "unmitigated", "compensated": "compensated", "dd": "dd (echo)"}


def ghz3d_payload(store, reg):
    """Everything the GHZ 3D page of one register shows: the placement and one data set per tiles group (unit size,
    per-unit layers, Layer 2 mode, setup) and per chip-wide state (tree, mode, setup; the newest run); None
    without a register map or records."""
    mp = ghz_register_map(store, reg)
    if not mp:
        return None
    datasets = []
    tl = {k: v for k, v in ghz_tiles(store.ghz).items() if k[0] == reg}
    ch = {k: v for k, v in ghz_chips(store.ghz).items() if k[0] == reg}
    if not tl and not ch:
        return None
    setups = {k[3] for k in tl} | {k[4] for k in ch}
    sizes = {k[1] for k in tl}
    r4 = lambda v: None if v is None or not math.isfinite(float(v)) else round(float(v), 4)
    for (_, k, layers, setup) in sorted(tl, key=lambda k_: (k_[1], k_[2], _setup_rank(k_[3]))):
        for m in GHZ_MODES:
            rs = tl[(reg, k, layers, setup)].get(m)
            if not rs:
                continue
            per, xper = defaultdict(list), defaultdict(list)
            for r in rs:
                F, xF = ghz_F(r)[1], ghz_F(r, "exact_")[1]
                for q in r.get("qubits", []):
                    per[int(q)].append(F)
                    if xF is not None:
                        xper[int(q)].append(xF)
            st, xs = ghz_place_stats(rs), ghz_place_stats(rs, "exact_")
            off = [v[1] for p, v in st.items() if v[0][0] == 0]
            on = [r_ for r_ in rs if ghz_place(r_)[0][0] >= 1]
            xon = [ghz_F(r_, "exact_")[1] for r_ in on if ghz_F(r_, "exact_")[1] is not None]
            xoff = [v[1] for p, v in xs.items() if v[0][0] == 0]
            extra = (f", {layers}" if layers not in (None, "none") else "") + (f" · {setup}" if len(setups) > 1 else "")
            datasets.append(dict(
                kind="tiles", mode=m, k=k, key=ghz_F(rs[0])[0], runs=len({r_.get("when") for r_ in rs}),
                shots=rs[0].get("shots"), realisations=max(int(r_.get("repeats") or 1) for r_ in rs), n_units=len(rs),
                label=f"{k}-qubit chains, {GHZ_MODE_LABEL.get(m, m)}{extra}",
                short=(f"{k}q · " if len(sizes) > 1 else "") + GHZ_MODE_SHORT.get(m, m)
                      + (" · " + layers if layers not in (None, "none") else "")
                      + (" · " + shor_setup(rs[0])[1] if len(setups) > 1 else ""),
                off=r4(off[0]) if off else None, on=r4(np.mean([ghz_F(r_)[1] for r_ in on])) if on else None,
                xoff=r4(xoff[0]) if xoff else None, xon=r4(np.mean(xon)) if xon else None,
                sites={str(q): dict(ace=r4(np.mean(v)), exact=r4(np.mean(xper[q])) if xper.get(q) else None, n=len(v))
                       for q, v in per.items()},
                units=[dict(qubits=[int(q) for q in r_.get("qubits", [])], F=r4(ghz_F(r_)[1]), xF=r4(ghz_F(r_, "exact_")[1]),
                            C=r4(r_.get("C_parity", r_.get("coherence"))), Phi=r4(r_.get("Phi_coh", r_.get("phase"))),
                            cx=[int((r_.get("cx_classes") or {}).get(x, 0)) for x in ("exact", "replica", "cross")],
                            place=ghz_place(r_)[1], tiling=str(r_.get("tiling", ""))) for r_ in rs]))
    for key in sorted(ch, key=lambda k_: (GHZ_MODES.index(k_[2]) if k_[2] in GHZ_MODES else 9, k_[1], k_[3], _setup_rank(k_[4]))):
        _, tree, m, layers, setup = key
        r = ch[key][-1]
        kF, F = ghz_F(r)
        cx = r.get("cx_classes") or {}
        extra = (f", {layers}" if layers not in (None, "none") else "") + (f" · {setup}" if len(setups) > 1 else "")
        datasets.append(dict(
            kind="chip", mode=m, k=int(r.get("n", 0)), key=kF, runs=len(ch[key]), shots=r.get("shots"),
            realisations=int(r.get("repeats") or 1), n_units=1,
            label=f"chip-wide GHZ, {tree} tree, {GHZ_MODE_LABEL.get(m, m)}{extra}",
            short=f"chip {tree} · {GHZ_MODE_SHORT.get(m, m)}" + (" · " + shor_setup(r)[1] if len(setups) > 1 else ""),
            sites={str(int(q)): dict(ace=r4(1.0 - float(v)), exact=None, n=1) for q, v in (r.get("site_break") or {}).items()},
            units=[], chip=dict(tree=tree, root=r.get("root"), depth=r.get("depth"), pop=r4(r.get("pop")), kF=kF, F=r4(F),
                                xF=r4(ghz_F(r, "exact_")[1]), cx=[int(cx.get(x, 0)) for x in ("exact", "replica", "cross")],
                                edges=sorted([[int(a_), int(b_), cl, round(float(v), 4)] for a_, b_, cl, v in r.get("edge_break", [])],
                                             key=lambda e: -e[3]))))
    if (mp.get("loom") or {}).get("chip_n"):            # loom / bridge / box maps: the chips (faces) are the patches
        ends = list(np.cumsum(mp["loom"]["chip_n"]))
        seam = _ghz_seam(mp)
        chip = lambda q: next(i for i, e in enumerate(ends) if int(q) < e)          # noqa: E731
        sites = [[int(st[0]), int(st[1]), int(st[2]), "seam" if int(st[0]) in seam else f"p{chip(st[0])}",
                  [chip(st[0])]] for st in mp["sites"]]
    else:
        sites = [list(st[:4]) + [list(st[4]) if len(st) > 4 else []] for st in mp["sites"]]
    cols_ = mp.get("columns") or (max(int(st[2]) for st in mp["sites"]) + 1)
    return dict(kind="ghz", register=reg, m=1, n=int(mp.get("n", len(sites))), columns=int(cols_),
                ace=mp.get("ace") or dict(lrc="?", lrr="?"), sites=sites, couplers=mp["couplers"], datasets=datasets)


def ghz3d_html(payload, three_base=THREE_CDN, document=True):
    js, body, head = _ghz3d_parts()
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")
    title = f"{payload['register']} GHZ in 3D"
    imports = json.dumps({"imports": {"three": three_base + "build/three.module.js",
                                      "three/addons/": three_base + "examples/jsm/"}})
    inner = (head.replace("__TITLE__", title) + ("</head><body>\n" if document else "") + body
             + '<script type="application/json" id="d">' + data + "</script>\n"
             + '<script type="importmap">' + imports + "</script>\n"
             + '<script type="module">\n' + js + "</script>\n")
    if not document:
        return inner
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            "<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}"
            "body{margin:0}[hidden]{display:none!important}</style>\n" + inner + "</body></html>\n")


class FigFactory:
    """The slice of pyplot the builders use, returning standalone Figures: they embed in
    the Tk viewer or save as PNG without pyplot ever opening a window."""

    def subplots(self, nrows=1, ncols=1, figsize=None, squeeze=True, **kw):
        from matplotlib.figure import Figure
        fig = Figure(figsize=figsize)
        return fig, fig.subplots(nrows, ncols, squeeze=squeeze, **kw)

    def get_cmap(self, name):
        import matplotlib
        return matplotlib.colormaps[name]

    def close(self, fig):
        pass


VIEWS = [
    ("1  Fidelity vs depth (61 q)", "fid"),
    ("2  Mirror decay per size", "sweep"),
    ("3  Per-cycle decay b(N)", "bN"),
    ("4  Mirror bitstrings", "mbits"),
    ("5  Patched + full bitstrings", "pbits"),
    ("6  Release + hwxeb", "release"),
    ("7  Run cost", "cost"),
    ("8  fxeb: XEB vs depth", "fxd"),
    ("9  fxeb: XEB vs size", "fxn"),
    ("10 fxeb: decay b(N) -> 61", "fxb"),
    ("11 fxeb: what predicts XEB", "fxp"),
    ("12 hamming: signal vs calculator", "hm"),
    ("13 hamming: ratio + folding", "hmr"),
    ("14 shor9: logical success vs size", "shz"),
    ("15 shor9: outcomes vs exact", "shd"),
    ("16 shor9: units on registers", "shu"),
    ("17 shor9: units on the grid", "shg"),
    ("18 ghz: tiles on the grid", "ghm"),
    ("19 ghz: placement x Layer 2", "ghl"),
    ("20 ghz: ACE vs exact", "ghx"),
    ("21 ghz: chip-wide break map", "ghc"),
    ("22 ghz: across loom / bridge vs bonds", "brc"),
    ("23 shor: across QrackBridge vs bonds", "shb"),
    ("24 dual R1: models vs IBM", "dua"),
    ("25 dual R1: fidelity map", "dum"),
    ("26 Shor algorithm: success vs bonds", "sas"),
    ("27 Shor algorithm: TVD vs bonds", "sat"),
    ("28 Shor algorithm: dual R1 models", "sad"),
]
SHOR_VIEWS = {"shz", "shd", "shu", "shg"}
GHZ_VIEWS = {"ghm": fig_ghz_map, "ghl": fig_ghz_modes, "ghx": fig_ghz_exact, "ghc": fig_ghz_chip}
ENGINE_VIEWS = {"brc": fig_bridge_capacity, "shb": fig_shor_bridge, "dua": fig_dual, "dum": fig_dual_map,
                "sas": fig_shor_alg, "sat": lambda plt, st: fig_shor_alg(plt, st, "tvd"), "sad": fig_shor_alg_dual}
PER_CFG = {"mbits", "pbits"}


class Store:
    """Everything the figures read; reload() re-reads all files (live sweeps)."""

    def __init__(self, a):
        self.a = a
        try:                                        # the release: views 1-13; shor9 and ghz views need none of it
            self.lay = nh.Layout(a.repo)
            self.fvd = json.loads((self.lay.repo / "data/results/fidelity_vs_depth.json").read_text())
        except SystemExit as err:
            print(f"# no BlueQubit release ({str(err).splitlines()[0] if str(err) else 'not found'}): "
                  "views 1-13 are empty; give --repo for them")
            self.lay, self.fvd = None, None
        self.reload()

    def reload(self):
        data, bases = {}, {}
        for out in self.a.out:
            for cfg, recs in read_records(out).items():
                if self.a.cfg and cfg not in self.a.cfg:
                    continue
                data.setdefault(cfg, {}).update(recs)
                bases[cfg] = shots_base(out, self.a.shots_dir)
        self.data, self.bases = data, bases
        files = list(self.a.out) + list(getattr(self.a, "shor", []) or []) + list(getattr(self.a, "ghz", []) or [])
        self.shor = read_shor(files)
        self.shor_maps = read_shor(files, "shor9_map")
        self.ghz = read_ghz(files)
        self.ghz_maps = read_ghz(files, "ghz_map")
        self.shor_bridge = read_tests(files, "shor_bridge", lambda r: (r.get("when"), r.get("register"), r.get("size"),
                                                                        r.get("unit")))
        self.dual = read_tests(files, "dual_r1", lambda r: (r.get("when"), r.get("state")))
        self.dual_map = read_tests(files, "dual_map", lambda r: (r.get("when"), r.get("unit")))
        self.dual_layout = read_tests(files, "dual_map_layout", lambda r: (r.get("when"),))
        self.shor_alg = read_tests(files, "shor_alg", lambda r: (r.get("when"), r.get("geometry"), r.get("engine"),
                                                                 r.get("N"), r.get("a"), r.get("t"), r.get("layout"),
                                                                 tuple(r.get("faces") or ()), r.get("max_bonds"),
                                                                 r.get("model")))
        ENSEMBLE.clear()
        ENSEMBLE.update({c: ensemble(r) for c, r in data.items()})
        LABELS.clear()
        LABELS.update({c: describe(c, r, self.lay) for c, r in data.items()} if self.lay is not None else
                      {c: c for c in data})
        self.hw = []
        for f in self.a.hwxeb:
            try:
                self.hw += json.loads(Path(f).read_text())
            except (OSError, json.JSONDecodeError) as err:
                print(f"hwxeb file {f}: {err}")
        self.stamp = __import__("time").strftime("%H:%M:%S")

    def summary(self):
        return (f"{', '.join(f'{c}: {len(r)} points' for c, r in sorted(self.data.items())) or 'no run records'}"
                f"; hwxeb circuits: {len(self.hw)}; shor9 records: {len(self.shor)}; ghz records: {len(self.ghz)}"
                f"; shor_bridge records: {len(self.shor_bridge)}; dual_r1 rows: {len(self.dual)}; dual_map chains: "
                f"{len(self.dual_map)}; shor_alg circuits: {len(self.shor_alg)}"
                f"; read {self.stamp}")

    def mirror_sizes(self, cfg):
        """Sizes with mirror or fxeb points (of cfg, or of every configuration)."""
        recs = [self.data.get(cfg, {})] if cfg else list(self.data.values())
        return sorted({k[0] for r in recs for k in r if k[1] in ("mirror", "fxeb")})

    def mirror_depths(self, cfg, n):
        recs = [self.data.get(cfg, {})] if cfg else list(self.data.values())
        return sorted({k[3] for r in recs for k in r if k[1] in ("mirror", "fxeb") and k[0] == n})

    def build(self, view, cfg=None, n=None, depth=None):
        """One Figure for a view (None if it has no data for this selection)."""
        F, lay, fvd = FigFactory(), self.lay, self.fvd
        if view in GHZ_VIEWS:
            return GHZ_VIEWS[view](F, self)
        if view in ENGINE_VIEWS:
            return ENGINE_VIEWS[view](F, self)
        if view in SHOR_VIEWS:
            if view == "shg":
                return fig_shor_grid(F, self)
            return {"shz": fig_shor_size, "shd": fig_shor_dist, "shu": fig_shor_units}[view](F, self.shor)
        if lay is None:
            return None
        data = {cfg: self.data[cfg]} if cfg in self.data else self.data
        if view == "fid":
            return fig_fidelity(F, lay, fvd, data, self.hw)
        if view == "sweep":
            return fig_sweep(F, lay, fvd, data) if data else None
        if view == "bN":
            return fig_bN(F, lay, fvd, data) if data else None
        if view == "release":
            return fig_release(F, lay, fvd, self.hw)
        if view == "cost":
            return fig_cost(F, data) if data else None
        if view == "fxd":
            return fig_fxeb_depth(F, lay, fvd, data)
        if view == "fxn":
            return fig_fxeb_size(F, lay, fvd, data)
        if view == "fxb":
            return fig_fxeb_bN(F, lay, fvd, data, self.a.fxeb_dmin)
        if view == "fxp":
            return fig_fxeb_predict(F, lay, data, n, depth)
        if view == "hm":
            return fig_hamming(F, data)
        if view == "hmr":
            return fig_hamming_ratio(F, data)
        c = cfg if cfg in self.data else next(iter(sorted(self.data)), None)
        if c is None:
            return None
        gen = (fig_mirror_bits(F, lay, {c: self.data[c]}, self.bases[c], n, depth) if view == "mbits"
               else fig_patched_full(F, lay, {c: self.data[c]}, self.bases[c], self.hw))
        return next((fig for _, fig in gen), None)


def save_all(store, folder, figs=None):
    Path(folder).mkdir(parents=True, exist_ok=True)
    for i, (label, view) in enumerate(VIEWS, 1):
        if figs and i not in figs:
            continue
        cfgs = sorted(store.data) if view in PER_CFG else [None]
        for cfg in cfgs:
            fig = store.build(view, cfg, store.a.n, store.a.depth)
            if fig is None:
                continue
            p = Path(folder) / f"{i}_{view}{'_' + cfg if cfg else ''}.png"
            fig.savefig(p, dpi=130, bbox_inches="tight")
            print("saved", p)


# ================================================================== Tk viewer
def run_viewer(store):
    import tkinter as tk
    from tkinter import ttk, filedialog
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    root = tk.Tk()
    root.title("nighthawk_qrack viewer")
    root.geometry("1500x950")

    left = ttk.Frame(root, padding=6)
    left.pack(side=tk.LEFT, fill=tk.Y)
    right = ttk.Frame(root)
    right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    ttk.Label(left, text="View").pack(anchor="w")
    lb = tk.Listbox(left, height=len(VIEWS), width=30, exportselection=False, activestyle="none")
    for label, _ in VIEWS:
        lb.insert(tk.END, label)
    lb.pack(anchor="w", pady=(0, 10))

    ALL = "(all configurations)"
    v_cfg, v_n, v_d = tk.StringVar(), tk.StringVar(), tk.StringVar()
    ttk.Label(left, text="Configuration").pack(anchor="w")
    cb_cfg = ttk.Combobox(left, textvariable=v_cfg, state="readonly", width=40)
    cb_cfg.pack(anchor="w", pady=(0, 8))
    ttk.Label(left, text="Register size n (views 4, 11)").pack(anchor="w")
    cb_n = ttk.Combobox(left, textvariable=v_n, state="readonly", width=28)
    cb_n.pack(anchor="w", pady=(0, 8))
    ttk.Label(left, text="Depth (views 4, 11)").pack(anchor="w")
    cb_d = ttk.Combobox(left, textvariable=v_d, state="readonly", width=28)
    cb_d.pack(anchor="w", pady=(0, 12))

    v_auto = tk.BooleanVar(value=False)
    btns = ttk.Frame(left)
    btns.pack(anchor="w", fill=tk.X)
    status = tk.StringVar()
    ttk.Label(left, textvariable=status, wraplength=230, foreground="#555").pack(anchor="w", pady=(12, 0))

    holder = {"canvas": None, "toolbar": None, "fig": None}

    def view_key():
        sel = lb.curselection()
        return VIEWS[sel[0] if sel else 0][1]

    def cfg_of(choice):
        return None if choice in ("", ALL) else choice.rsplit("[", 1)[-1].rstrip("]")

    def refresh_choices():
        cfgs = sorted(store.data, key=lambda c: (LABELS.get(c, c), c))
        cb_cfg["values"] = [ALL] + [lab(c) for c in cfgs]
        if v_cfg.get() not in cb_cfg["values"]:
            v_cfg.set(lab(cfgs[0]) if len(cfgs) == 1 else ALL)
        c = cfg_of(v_cfg.get())
        sizes = store.mirror_sizes(c) if c else []
        cb_n["values"] = [str(x) for x in sizes]
        if v_n.get() not in cb_n["values"]:
            v_n.set(str(store.a.n) if store.a.n in sizes else (str(sizes[-1]) if sizes else ""))
        ds = store.mirror_depths(c, int(v_n.get())) if v_n.get() else []
        cb_d["values"] = ["auto"] + [str(x) for x in ds]
        if v_d.get() not in cb_d["values"]:
            v_d.set(str(store.a.depth) if store.a.depth in ds else "auto")

    def draw(*_):
        refresh_choices()
        view = view_key()
        cfg = cfg_of(v_cfg.get())
        n = int(v_n.get()) if v_n.get() else None
        d = int(v_d.get()) if v_d.get() not in ("", "auto") else None
        root.config(cursor="watch")
        root.update_idletasks()
        try:
            fig = store.build(view, cfg, n, d)
        except Exception as err:                    # keep the viewer alive on a bad file
            fig = None
            msg = f"error: {err}"
        else:
            msg = "no data for this view and selection" if fig is None else ""
        if fig is None:
            fig = FigFactory().subplots(figsize=(8, 5))[0]
            fig.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12)
        if holder["canvas"] is not None:
            holder["toolbar"].destroy()
            holder["canvas"].get_tk_widget().destroy()
        canvas = FigureCanvasTkAgg(fig, master=right)
        toolbar = NavigationToolbar2Tk(canvas, right, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()
        holder.update(canvas=canvas, toolbar=toolbar, fig=fig)
        status.set(store.summary())
        root.config(cursor="")

    def reload(*_):
        store.reload()
        draw()

    def save_png():
        p = filedialog.asksaveasfilename(defaultextension=".png", initialfile=f"{view_key()}.png",
                                         filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if p:
            holder["fig"].savefig(p, dpi=130, bbox_inches="tight")
            status.set(f"saved {p}")

    def save_every():
        d = filedialog.askdirectory(initialdir=os.getcwd())
        if d:
            save_all(store, d)
            status.set(f"saved all views to {d}")

    def tick():
        if v_auto.get():
            reload()
        root.after(60_000, tick)

    ttk.Button(btns, text="Reload", command=reload).pack(fill=tk.X)
    ttk.Checkbutton(btns, text="Auto-reload every 60 s", variable=v_auto).pack(anchor="w", pady=4)
    ttk.Button(btns, text="Save PNG...", command=save_png).pack(fill=tk.X)
    ttk.Button(btns, text="Save all views...", command=save_every).pack(fill=tk.X, pady=(4, 0))

    def export_table():
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="harvest.csv",
                                         filetypes=[("CSV", "*.csv")])
        if p:
            if store.data and store.lay is not None:
                write_table(store, p)
            sp = str(Path(p).with_name(Path(p).stem + "_shor9.csv"))
            gp = str(Path(p).with_name(Path(p).stem + "_ghz.csv"))
            status.set(f"wrote {p}" + (f" and {sp}" if write_shor_table(store, sp) else "")
                       + (f" and {gp}" if write_ghz_table(store, gp) else ""))

    ttk.Button(btns, text="Export table (CSV)...", command=export_table).pack(fill=tk.X, pady=(4, 0))

    def open_3d():
        try:
            url = open_shor3d(store, None, store.a.three)
        except (SystemExit, Exception) as err:          # keep the viewer alive
            status.set(f"3D: {err}")
            return
        status.set(f"3D view: {url}" if url else "3D: no shor9 --register or ghz records with a register map")

    ttk.Button(btns, text="Open shor9 / GHZ registers in 3D", command=open_3d).pack(fill=tk.X, pady=(4, 0))

    lb.bind("<<ListboxSelect>>", draw)
    for cb in (cb_cfg, cb_n, cb_d):
        cb.bind("<<ComboboxSelected>>", draw)
    root.bind("<F5>", reload)
    start = max(0, min(len(VIEWS) - 1, (min(store.a.figs_set) if store.a.figs_set else 1) - 1))
    lb.select_set(start)
    root.after(60_000, tick)
    root.after(50, draw)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


# ================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", action="append", default=[],
                    help="run record file(s) given to nighthawk_qrack.py run --out (repeatable)")
    ap.add_argument("--hwxeb", action="append", default=[], help="hwxeb --out JSON file(s)")
    ap.add_argument("--grid3d", nargs="?", const="nighthawk_3d", default=None, metavar="DIR",
                    help="write a rotatable 3D page per register and code size of the shor9 --register records "
                         "(view 17 in three dimensions) and per register of the ghz records (views 18 and 21) to DIR "
                         "(default ./nighthawk_3d), with index.html; then exit")
    ap.add_argument("--three", default=None, metavar="DIR_OR_URL",
                    help="--grid3d / viewer: three.js 0.160 from here instead of jsDelivr: an npm `three` package "
                         "directory (copied next to the pages; offline, needs --serve) or a URL ending in /")
    ap.add_argument("--serve", nargs="?", type=int, const=8017, default=None, metavar="PORT",
                    help="--grid3d: serve the pages on http://127.0.0.1:PORT (default 8017) and open them in the "
                         "browser; Ctrl-C stops")
    ap.add_argument("--open", action="store_true", help="--grid3d: open the pages in the default browser")
    ap.add_argument("--shor", action="append", default=[],
                    help="shor9 --out file(s) (views 14-16); harvested *.jsonl are searched for shor9 records too")
    ap.add_argument("--ghz", action="append", default=[],
                    help="nighthawk_qrack.py ghz --out file(s) (views 18-28: ghz, stack, shor, dual, dualmap, shoralg); harvested *.jsonl "
                         "are searched for these records too")
    ap.add_argument("--repo", default=None, help="rcs-nighthawk clone (default: auto-detect)")
    ap.add_argument("--shots-dir", dest="shots_dir", default=None, help="as given to run --shots-dir")
    ap.add_argument("--cfg", action="append", default=None, help="only these config tags")
    ap.add_argument("--n", type=int, default=None, help="register size for view 4 (default: largest)")
    ap.add_argument("--depth", type=int, default=None, help="depth for the per-qubit grid in view 4")
    ap.add_argument("--figs", default=None, help="PNG mode: which views, e.g. 1-3,6; viewer: the first one shown")
    ap.add_argument("--save", default=None, help="write PNGs to this directory instead of opening the viewer")
    ap.add_argument("--table", default=None,
                    help="write every point of every configuration to this CSV, print the comparison, exit")
    ap.add_argument("--theta", choices=["haar", "nnqab", "all"], default="haar",
                    help="views 8-11 and the --table comparison: single-qubit ensemble to compare "
                         "(default haar, the paper's circuits; all = both, labelled)")
    ap.add_argument("--fxeb-dmin", dest="fxeb_dmin", type=int, default=8,
                    help="views 10-11 and --table: smallest depth used in the decay fit (default 8; "
                         "shallower outputs are too concentrated)")
    a = ap.parse_args()
    THETA["show"] = a.theta
    if not a.out:                                   # harvest: every record file here
        import re
        # a run launched with --gpus writes only worker files (<stem>.w<pid>.jsonl) until it
        # finishes, so a run in progress is found through its worker files' stem as well
        stems = set()
        for p in Path(".").glob("*.jsonl"):
            m = re.match(r"^(.*)\.w\d+\.jsonl$", p.name)
            stems.add(m.group(1) + ".jsonl" if m else p.name)
        a.out = sorted(stems)
        if a.out:
            print(f"harvesting {len(a.out)} record files: {', '.join(a.out)}")
    if not a.out and not a.hwxeb and not a.shor and not a.ghz:
        ap.print_help()
        return
    a.figs_set = set(nh.parse_sizes(a.figs)) if a.figs else set()
    if a.grid3d:
        import matplotlib
        matplotlib.use("Agg")
        store = Store(a)
        print(store.summary())
        pages = write_shor3d(store, a.grid3d, a.three)
        if not pages:
            print("no shor9 --register or ghz records with a register map: nothing to draw in 3D")
            return
        import webbrowser
        if a.serve is not None or (a.three and not a.three.startswith(("http://", "https://"))):
            url = serve_dir(a.grid3d, a.serve or 8017) + pages[0].name
            print(f"serving {a.grid3d} at {url} (Ctrl-C stops)")
            webbrowser.open(url)
            try:
                while True:
                    __import__("time").sleep(3600)
            except KeyboardInterrupt:
                return
        if a.open:
            webbrowser.open(pages[0].resolve().as_uri())
        return
    if a.table:
        import matplotlib
        matplotlib.use("Agg")
        store = Store(a)
        print(store.summary())
        if store.data and store.lay is not None:
            write_table(store, a.table)
        if store.shor:
            write_shor_table(store, str(Path(a.table).with_name(Path(a.table).stem + "_shor9.csv")))
        if store.ghz:
            write_ghz_table(store, str(Path(a.table).with_name(Path(a.table).stem + "_ghz.csv")))
        if store.shor_bridge or store.dual or store.dual_map or store.shor_alg or any(engine_record_bonds(r) is not None
                                                                     for r in store.ghz):
            write_engine_tables(store, a.table)
        return

    import matplotlib
    display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) or sys.platform in ("win32", "darwin")
    tk_ok = importlib.util.find_spec("tkinter") is not None
    if a.save or not display or not tk_ok:
        matplotlib.use("Agg")
        store = Store(a)
        print(store.summary())
        if not a.save:
            a.save = "nighthawk_plots"
            print("no display" if not display else "tkinter missing (apt install python3-tk)",
                  "-> writing PNGs to ./nighthawk_plots")
        save_all(store, a.save, a.figs_set or None)
        return
    matplotlib.use("TkAgg")
    store = Store(a)
    print(store.summary())
    run_viewer(store)


if __name__ == "__main__":
    main()
