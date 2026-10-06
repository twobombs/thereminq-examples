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
    return lab_ + ("" if ENSEMBLE.get(cfg, "haar") == "haar" else f" [θ {ENSEMBLE[cfg]}]")


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
        if rn != n or r.get("fidelity") is None:
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
]
PER_CFG = {"mbits", "pbits"}


class Store:
    """Everything the figures read; reload() re-reads all files (live sweeps)."""

    def __init__(self, a):
        self.a = a
        self.lay = nh.Layout(a.repo)
        self.fvd = json.loads((self.lay.repo / "data/results/fidelity_vs_depth.json").read_text())
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
        ENSEMBLE.clear()
        ENSEMBLE.update({c: ensemble(r) for c, r in data.items()})
        LABELS.clear()
        LABELS.update({c: describe(c, r, self.lay) for c, r in data.items()})
        self.hw = []
        for f in self.a.hwxeb:
            try:
                self.hw += json.loads(Path(f).read_text())
            except (OSError, json.JSONDecodeError) as err:
                print(f"hwxeb file {f}: {err}")
        self.stamp = __import__("time").strftime("%H:%M:%S")

    def summary(self):
        return (f"{', '.join(f'{c}: {len(r)} points' for c, r in sorted(self.data.items())) or 'no run records'}"
                f"; hwxeb circuits: {len(self.hw)}; read {self.stamp}")

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
            write_table(store, p)
            status.set(f"wrote {p}")

    ttk.Button(btns, text="Export table (CSV)...", command=export_table).pack(fill=tk.X, pady=(4, 0))

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
    if not a.out and not a.hwxeb:
        ap.print_help()
        return
    a.figs_set = set(nh.parse_sizes(a.figs)) if a.figs else set()
    if a.table:
        import matplotlib
        matplotlib.use("Agg")
        store = Store(a)
        print(store.summary())
        write_table(store, a.table)
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
