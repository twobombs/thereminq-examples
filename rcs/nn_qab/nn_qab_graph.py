#!/usr/bin/env python3
"""
plot3d.py -- interactive 3D view of the sweep output, plus CSV export.

Axes are chosen to show the thing the dataset was built to test:

    x  bulk-to-boundary ratio   (the regressor)
    y  width                    (the confound)
    z  XEB (or HOG)

In the original five series, width and B-to-B correlate at r = 0.84, so a
2D XEB-vs-ratio plot cannot distinguish "ratio drives XEB" from "width
drives XEB". Putting width on its own axis makes the confound visible, and
the width-28 replicate at ratio 2.5 -- same ratio as width 14, double the
seam -- is the point that breaks the collinearity. If ratio is what
matters, those two series means sit at the same height, and the dashed
connector between them is flat.

Interactive: drag inside the plot to rotate, scroll to zoom. The panel on
the right toggles layers, switches metric, sets the viewpoint, and writes
the CSVs and a PNG.

CSV output (always written, also on demand from the UI):
    <prefix>-runs.csv     every individual run from every series, one row
                          each, with its geometry -- the tidy long form
    <prefix>-summary.csv  one row per series: n, mean, sd, se, 95% CI, and
                          the residual against each fitted model

Reads the per-run CSVs written by `nn_qab.py merge` (runs/w*.csv).

Usage:
    python3 plot3d.py                        # render + CSVs, no window
    python3 plot3d.py --show                 # interactive UI
    python3 plot3d.py --runs runs --prefix xeb3d
    python3 plot3d.py --metric hog --elev 12 --azim -75
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d')


# Column order for the exported per-run CSV. Anything else found in the
# input is appended after these, so a schema change upstream is not lost.
PREFERRED = [
    "width", "long_range_columns", "long_range_rows",
    "bulk_to_boundary", "boundary_qubits", "bulk_qubits",
    "seed", "depth", "shots", "swap_mode", "resolved_swap_mode",
    "xeb_ace", "hog_ace",
    "ace_seconds", "ideal_seconds", "stats_seconds",
    "qrack_lib", "qrack_device_ace", "qrack_device_ideal",
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(runs_dir):
    """One entry per series, keeping the raw rows so export stays lossless."""
    series = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "w*.csv"))):
        rows, xebs, hogs, ratios = [], [], [], []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    xeb = float(row["xeb_ace"])
                    ratio = float(row["bulk_to_boundary"])
                except (KeyError, ValueError):
                    continue
                hogs.append(float(row.get("hog_ace", "nan") or "nan"))
                xebs.append(xeb)
                ratios.append(ratio)
                rows.append(row)
        if not rows:
            continue

        series.append({
            "path": path,
            "rows": rows,
            "width": int(rows[0]["width"]),
            "lrc": int(rows[0]["long_range_columns"]),
            "lrr": int(rows[0]["long_range_rows"]),
            "ratio": float(np.mean(ratios)),
            "xeb": np.asarray(xebs),
            "hog": np.asarray(hogs),
        })
    series.sort(key=lambda s: (s["ratio"], s["width"]))
    return series


def stats_for(series, metric):
    """Attach mean/sd/se for the selected metric, in place."""
    for s in series:
        v = s[metric]
        v = v[np.isfinite(v)]
        n = len(v)
        s["n"] = n
        s["mean"] = float(v.mean()) if n else float("nan")
        s["sd"] = float(v.std(ddof=1)) if n > 1 else 0.0
        s["se"] = s["sd"] / np.sqrt(n) if n > 1 else 0.0
    return series


def loocv(x_cols, y):
    """Leave-one-out RMSE for an OLS fit; an intercept is added here.

    With only a handful of series means, in-sample R^2 always improves when
    you add a term, so LOOCV is the honest way to ask whether the extra
    degree of freedom earns its place.
    """
    X = np.column_stack([np.ones(len(y))] + list(x_cols))
    n = len(y)
    if n <= X.shape[1]:
        return float("nan")
    err = []
    for i in range(n):
        keep = np.arange(n) != i
        beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        err.append(y[i] - X[i] @ beta)
    return float(np.sqrt(np.mean(np.square(err))))


def fit_models(series):
    """OLS on the series means: ratio only, and ratio + width."""
    ratio = np.array([s["ratio"] for s in series])
    width = np.array([s["width"] for s in series], dtype=float)
    mean = np.array([s["mean"] for s in series])

    out = {"ratio": ratio, "width": width, "mean": mean,
           "line": None, "plane": None, "corr": float("nan")}

    if len(series) >= 3:
        A = np.column_stack([np.ones(len(mean)), ratio])
        beta, *_ = np.linalg.lstsq(A, mean, rcond=None)
        out["line"] = (beta, loocv([ratio], mean))

    if len(series) >= 4 and np.ptp(width) > 0:
        B = np.column_stack([np.ones(len(mean)), ratio, width])
        beta, *_ = np.linalg.lstsq(B, mean, rcond=None)
        out["plane"] = (beta, loocv([ratio, width], mean))
        out["corr"] = float(np.corrcoef(ratio, width)[0, 1])

    return out


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_runs(series, path):
    fields = list(PREFERRED)
    for s in series:
        for k in s["rows"][0]:
            if k not in fields:
                fields.append(k)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for s in series:
            for row in s["rows"]:
                w.writerow(row)
    return sum(len(s["rows"]) for s in series)


def export_summary(series, fit, metric, path):
    head = ["width", "long_range_columns", "long_range_rows",
            "bulk_to_boundary", "n", f"mean_{metric}", f"sd_{metric}",
            "se", "ci95_lo", "ci95_hi",
            "fit_ratio_pred", "fit_ratio_resid",
            "fit_plane_pred", "fit_plane_resid"]

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for s in series:
            lo = s["mean"] - 1.96 * s["se"]
            hi = s["mean"] + 1.96 * s["se"]
            row = [s["width"], s["lrc"], s["lrr"], f"{s['ratio']:.6f}",
                   s["n"], f"{s['mean']:.10f}", f"{s['sd']:.10f}",
                   f"{s['se']:.10f}", f"{lo:.10f}", f"{hi:.10f}"]

            if fit["line"] is not None:
                b = fit["line"][0]
                pred = b[0] + b[1] * s["ratio"]
                row += [f"{pred:.10f}", f"{s['mean'] - pred:.10f}"]
            else:
                row += ["", ""]

            if fit["plane"] is not None:
                b = fit["plane"][0]
                pred = b[0] + b[1] * s["ratio"] + b[2] * s["width"]
                row += [f"{pred:.10f}", f"{s['mean'] - pred:.10f}"]
            else:
                row += ["", ""]

            w.writerow(row)
    return len(series)


def report(series, fit, metric):
    label = metric.upper()
    print(f"{'width':>6} {'lrc':>4} {'lrr':>4} {'B-to-B':>7} "
          f"{'n':>5} {'mean ' + label:>12} {'sd':>10} {'SE':>10}")
    for s in series:
        print(f"{s['width']:6d} {s['lrc']:4d} {s['lrr']:4d} {s['ratio']:7.2f} "
              f"{s['n']:5d} {s['mean']:12.6f} {s['sd']:10.6f} {s['se']:10.6f}")

    if fit["line"] is not None:
        b, cv = fit["line"]
        print(f"\nratio only    : {label} = {b[0]:+.5f} {b[1]:+.5f}*ratio"
              f"    LOOCV RMSE {cv:.5f}")
    if fit["plane"] is not None:
        b, cv = fit["plane"]
        print(f"ratio + width : {label} = {b[0]:+.5f} {b[1]:+.5f}*ratio "
              f"{b[2]:+.5f}*width    LOOCV RMSE {cv:.5f}")
        r = fit["corr"]
        print(f"corr(ratio, width) = {r:+.3f}"
              + ("   <- collinear; width term is not identifiable"
                 if abs(r) > 0.8 else ""))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

class Explorer:
    LAYERS = ["runs", "plane", "error bars", "labels", "equal-ratio"]

    def __init__(self, series, metric, elev, azim, interactive):
        self.series = series
        self.metric = metric
        self.visible = {k: True for k in self.LAYERS}
        self.interactive = interactive

        self.fig = plt.figure(figsize=(13, 8) if interactive else (11, 8))
        rect = [0.02, 0.05, 0.70, 0.90] if interactive else [0.03, 0.05, 0.94, 0.90]
        self.ax = self.fig.add_axes(rect, projection="3d")
        self.elev, self.azim = elev, azim
        self.status = None
        self.draw()

        if interactive:
            self._build_panel()

    # -- drawing ---------------------------------------------------------
    def draw(self):
        ax = self.ax
        ax.clear()

        series = stats_for(self.series, self.metric)
        self.fit = fit_models(series)
        label = self.metric.upper()

        colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(series)))
        ratio = self.fit["ratio"]
        width = self.fit["width"]

        if self.visible["plane"] and self.fit["plane"] is not None:
            b = self.fit["plane"][0]
            gx = np.linspace(ratio.min() - 0.3, ratio.max() + 0.3, 24)
            gy = np.linspace(width.min() - 2, width.max() + 2, 24)
            GX, GY = np.meshgrid(gx, gy)
            ax.plot_surface(GX, GY, b[0] + b[1] * GX + b[2] * GY,
                            alpha=0.16, color="tab:orange", linewidth=0,
                            antialiased=True, shade=False)

        vals = [s[self.metric][np.isfinite(s[self.metric])] for s in series]
        zfloor = min(0.0, float(min(v.min() for v in vals)))

        for s, c in zip(series, colors):
            v = s[self.metric]
            v = v[np.isfinite(v)]

            if self.visible["runs"]:
                # Jitter along width only, so the ratio axis stays exact.
                jit = np.random.default_rng(s["width"]).normal(0, 0.28, len(v))
                ax.scatter(np.full(len(v), s["ratio"]), s["width"] + jit, v,
                           s=7, color=c, alpha=0.16, edgecolors="none",
                           depthshade=False)

            if self.visible["error bars"]:
                ax.plot([s["ratio"]] * 2, [s["width"]] * 2,
                        [s["mean"] - s["se"], s["mean"] + s["se"]],
                        color=c, lw=2.4, solid_capstyle="round", zorder=5)

            ax.plot([s["ratio"]] * 2, [s["width"]] * 2, [zfloor, s["mean"]],
                    color=c, lw=0.8, alpha=0.45, ls=":")
            ax.scatter([s["ratio"]], [s["width"]], [s["mean"]], s=95, color=c,
                       edgecolors="black", linewidths=0.8, depthshade=False,
                       zorder=6,
                       label=f"w{s['width']}  B/B {s['ratio']:.1f}  n={s['n']}")

            if self.visible["labels"]:
                ax.text(s["ratio"], s["width"], s["mean"] + s["se"] + 0.012,
                        f"{s['mean']:.3f}", fontsize=8, ha="center")

        if self.visible["equal-ratio"]:
            groups = {}
            for s in series:
                groups.setdefault(round(s["ratio"], 6), []).append(s)
            for g in groups.values():
                if len(g) < 2:
                    continue
                g.sort(key=lambda s: s["width"])
                ax.plot([x["ratio"] for x in g], [x["width"] for x in g],
                        [x["mean"] for x in g], color="crimson", lw=1.6,
                        ls="--", zorder=7)
                mw = float(np.mean([x["width"] for x in g]))
                mz = float(np.mean([x["mean"] for x in g]))
                span = max(x["mean"] for x in g) - min(x["mean"] for x in g)
                ax.text(g[0]["ratio"], mw, mz - max(0.07, span * 0.6),
                        "same ratio,\ndifferent width", color="crimson",
                        fontsize=8, ha="center", va="top")

        ax.set_xlabel("bulk-to-boundary ratio")
        ax.set_ylabel("width (qubits)")
        ax.set_zlabel(label)
        ax.set_title(f"{label} vs bulk-to-boundary and width")
        ax.view_init(elev=self.elev, azim=self.azim)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        self.fig.canvas.draw_idle()

    # -- widgets ---------------------------------------------------------
    def _build_panel(self):
        f = self.fig

        f.text(0.755, 0.945, "layers", fontsize=9, weight="bold")
        ax_layers = f.add_axes([0.755, 0.775, 0.20, 0.16])
        ax_layers.set_frame_on(False)
        self.w_layers = CheckButtons(ax_layers, self.LAYERS,
                                     [True] * len(self.LAYERS))
        self.w_layers.on_clicked(self._toggle_layer)

        f.text(0.755, 0.735, "metric", fontsize=9, weight="bold")
        ax_metric = f.add_axes([0.755, 0.640, 0.20, 0.085])
        ax_metric.set_frame_on(False)
        self.w_metric = RadioButtons(ax_metric, ("xeb", "hog"),
                                     active=0 if self.metric == "xeb" else 1)
        self.w_metric.on_clicked(self._set_metric)

        f.text(0.755, 0.590, "viewpoint", fontsize=9, weight="bold")
        ax_elev = f.add_axes([0.795, 0.545, 0.155, 0.025])
        ax_azim = f.add_axes([0.795, 0.505, 0.155, 0.025])
        self.s_elev = Slider(ax_elev, "elev", -90, 90, valinit=self.elev)
        self.s_azim = Slider(ax_azim, "azim", -180, 180, valinit=self.azim)
        self.s_elev.on_changed(self._set_view)
        self.s_azim.on_changed(self._set_view)

        ax_csv = f.add_axes([0.755, 0.410, 0.20, 0.045])
        self.b_csv = Button(ax_csv, "save CSV")
        self.b_csv.on_clicked(lambda _evt: self._save_csv())

        ax_png = f.add_axes([0.755, 0.350, 0.20, 0.045])
        self.b_png = Button(ax_png, "save PNG")
        self.b_png.on_clicked(lambda _evt: self._save_png())

        self.status = f.text(0.755, 0.300, "", fontsize=8, va="top",
                             wrap=True, color="tab:green")
        f.text(0.755, 0.10,
               "drag in the plot to rotate\nscroll to zoom",
               fontsize=8, color="0.4", va="bottom")

    def _toggle_layer(self, label):
        self.visible[label] = not self.visible[label]
        self.draw()

    def _set_metric(self, label):
        self.metric = label
        self.draw()

    def _set_view(self, _val):
        self.elev = self.s_elev.val
        self.azim = self.s_azim.val
        self.ax.view_init(elev=self.elev, azim=self.azim)
        self.fig.canvas.draw_idle()

    def _say(self, msg):
        print(msg)
        if self.status is not None:
            self.status.set_text(msg)
            self.fig.canvas.draw_idle()

    def _save_csv(self):
        runs_path = f"{self.prefix}-runs.csv"
        sum_path = f"{self.prefix}-summary.csv"
        n = export_runs(self.series, runs_path)
        k = export_summary(self.series, self.fit, self.metric, sum_path)
        self._say(f"wrote {runs_path} ({n} runs)\nwrote {sum_path} ({k} series)")

    def _save_png(self):
        path = f"{self.prefix}.png"
        # Hide the widget panel in the saved image.
        bbox = self.ax.get_tightbbox().transformed(
            self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(path, dpi=160, bbox_inches=bbox.expanded(1.08, 1.08))
        self._say(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs", help="directory holding w*.csv")
    p.add_argument("--prefix", default="xeb3d",
                   help="output prefix for .png / -runs.csv / -summary.csv")
    p.add_argument("--show", action="store_true",
                   help="open the interactive window")
    p.add_argument("--metric", default="xeb", choices=("xeb", "hog"))
    p.add_argument("--elev", type=float, default=24.0)
    p.add_argument("--azim", type=float, default=-58.0)
    p.add_argument("--dpi", type=int, default=160)
    args = p.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    series = load(args.runs)
    if not series:
        print(f"no w*.csv found under {args.runs}/ "
              f"(run: nn_qab.py merge --out runs/wNN --csv runs/wNN.csv)",
              file=sys.stderr)
        return 1

    ex = Explorer(series, args.metric, args.elev, args.azim, args.show)
    ex.prefix = args.prefix

    report(stats_for(series, args.metric), ex.fit, args.metric)
    print()
    ex._save_csv()

    if args.show:
        backend = matplotlib.get_backend()
        if backend.lower() in ("agg", "template"):
            print(f"\nbackend is {backend!r}: no display available, so no "
                  f"window will open.\nInstall a GUI backend (python3-tk "
                  f"for TkAgg) or run without --show.", file=sys.stderr)
        plt.show()
    else:
        ex._save_png()

    return 0


if __name__ == "__main__":
    sys.exit(main())
