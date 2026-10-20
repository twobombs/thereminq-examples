#!/usr/bin/env python3
"""
xeb3d.py -- interactive 3D viewer for cross-entropy benchmarking sample dumps.

Reads a pair of .npy files:

    <prefix>_xeb_amplitudes.npy   (S,)       ideal amplitudes / magnitudes / probs
    <prefix>_xeb_bitstrings.npy   (S, N)     measured bitstrings, one row per shot

and gives you five rotatable 3D views plus a live fidelity readout.

Views
-----
  lattice        qubits placed on an nx*ny*nz grid, coloured by P(1) marginal or
                 by the currently selected shot's bit pattern
  cloud          the S shots embedded in 3D by PCA of the centred bit matrix,
                 coloured by log10(D*p) -- shows whether heavy bitstrings clump
  corr           connected ZZ correlation matrix <Z_i Z_j> - <Z_i><Z_j> as a surface
  porter         Porter-Thomas histogram of x = D*p in 3D bars, with the
                 depolarising-model curve (1-F)e^-x + F*x*e^-x overlaid
  landscape      joint density of (Hamming weight, log10 x) as a surface

Controls
--------
  radio buttons   switch view
  slider          select shot index (highlighted in lattice / cloud)
  check boxes     lattice shows selected shot; log colour scale; theory overlay
  save button     dump the current view to PNG
  left/right      step the shot selector
  1..5            jump to a view

Usage
-----
  python xeb3d.py                                  # auto-discovers *_xeb_*.npy in cwd
  python xeb3d.py --amps A.npy --bits B.npy
  python xeb3d.py --dims 7,5,2                     # qubit lattice shape (product must be >= N)
  python xeb3d.py --save-all out/                  # headless: render every view to PNG
  python xeb3d.py --stats                          # print the fidelity table and exit

No dependencies beyond numpy + matplotlib.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

EULER_GAMMA = 0.5772156649015329
LN2 = math.log(2.0)

# --------------------------------------------------------------------------- #
# Matplotlib Shim Lazy Loader
# --------------------------------------------------------------------------- #

plt_colors = None
plt_cm = None

def _ensure_mpl():
    """Bind the colour helpers used inside the renderers, matplotlib-version safe.
    Lazily evaluated to support direct Viewer instantiations without going through main.
    """
    global plt_colors, plt_cm
    if plt_colors is not None and plt_cm is not None:
        return
        
    try:
        import matplotlib.colors as _c
    except ImportError as e:
        raise ImportError("xeb3d requires matplotlib; install it with: pip install matplotlib") from e
        
    plt_colors = _c

    class _CmapShim:
        @staticmethod
        def get_cmap(name):
            try:
                import matplotlib as mpl
                return mpl.colormaps[name]           # matplotlib >= 3.6
            except Exception:
                import matplotlib.cm as cm
                return cm.get_cmap(name)             # older matplotlib

    plt_cm = _CmapShim()


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #

@dataclass
class XEBData:
    bits: np.ndarray        # (S, N) bool, True == measured 1
    p: np.ndarray           # (S,) ideal probability of each sampled bitstring
    n_qubits: int
    n_shots: int
    amp_kind: str           # how the amplitude file was interpreted
    amps_path: str
    bits_path: str

    def __post_init__(self):
        self._weights: np.ndarray | None = None

    @property
    def dim(self) -> float:
        return 2.0 ** self.n_qubits

    @property
    def x(self) -> np.ndarray:
        """Normalised ideal probability D*p. Porter-Thomas mean is 1."""
        return self.dim * self.p

    @property
    def weights(self) -> np.ndarray:
        if self._weights is None:
            self._weights = self.bits.sum(axis=1)
        return self._weights


def _interpret_amplitudes(a: np.ndarray, n_qubits: int) -> tuple[np.ndarray, str]:
    """Figure out whether the file holds amplitudes, magnitudes or probabilities.

    The test is which interpretation puts the mean of D*p near unity, since a
    Porter-Thomas sample has E[D*p] between 1 and 2 for any plausible fidelity.
    """
    dim = 2.0 ** n_qubits

    if np.iscomplexobj(a):
        return np.abs(a) ** 2, "complex amplitudes -> |a|^2"

    a = np.asarray(a, dtype=np.float64)
    if np.any(a < 0):
        # signed real amplitudes
        return a ** 2, "signed real amplitudes -> a^2"

    as_prob = float(np.mean(a) * dim)
    as_mag = float(np.mean(a ** 2) * dim)

    # score = how far log10(mean D*p) sits from the expected O(1)
    score_prob = abs(math.log10(as_prob)) if as_prob > 0 else float("inf")
    score_mag = abs(math.log10(as_mag)) if as_mag > 0 else float("inf")

    if score_mag <= score_prob:
        return a ** 2, f"magnitudes |a| -> a^2 (mean D*p = {as_mag:.3f})"
    return a, f"probabilities (mean D*p = {as_prob:.3f})"


def _normalise_bits(b: np.ndarray, n_shots: int) -> np.ndarray:
    """Coerce the bitstring array to (S, N) bool with True == measured 1."""
    b = np.asarray(b)
    if b.ndim != 2:
        raise ValueError(f"bitstring array must be 2-D, got shape {b.shape}")

    if b.shape[0] == n_shots and b.shape[1] == n_shots:
        warnings.warn(
            f"Bitstring array is square ({n_shots}x{n_shots}). Assuming rows are shots. "
            "If this is incorrect, transpose the array before loading.",
            stacklevel=3
        )
    elif b.shape[0] != n_shots and b.shape[1] == n_shots:
        b = b.T
        
    if b.shape[0] != n_shots:
        raise ValueError(
            f"bitstring rows ({b.shape[0]}) do not match amplitude count ({n_shots})"
        )

    if b.dtype == np.bool_:
        return b
    if b.min() < 0:
        # +-1 eigenvalue convention: z = +1 is |0>, z = -1 is |1>
        return b < 0
    return b.astype(bool)


def load_xeb(amps_path: str, bits_path: str) -> XEBData:
    amps = np.load(amps_path, allow_pickle=False).ravel()
    raw_bits = np.load(bits_path, allow_pickle=False)

    n_shots = amps.size
    bits = _normalise_bits(raw_bits, n_shots)
    n_qubits = bits.shape[1]

    p, kind = _interpret_amplitudes(amps, n_qubits)

    return XEBData(
        bits=bits,
        p=p,
        n_qubits=n_qubits,
        n_shots=n_shots,
        amp_kind=kind,
        amps_path=amps_path,
        bits_path=bits_path,
    )


def autodiscover(directory: str = ".") -> tuple[str, str]:
    amps = sorted(glob.glob(os.path.join(directory, "*amplitude*.npy")))
    bits = sorted(glob.glob(os.path.join(directory, "*bitstring*.npy")))
    if not amps or not bits:
        raise SystemExit(
            "could not auto-discover *amplitude*.npy / *bitstring*.npy in "
            f"{os.path.abspath(directory)!r}; pass --amps and --bits explicitly"
        )
    return amps[0], bits[0]


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def fidelity_table(d: XEBData) -> dict[str, float]:
    """Three independent fidelity estimators under a global depolarising model.

    Sampling distribution is assumed to be  F*p_ideal + (1-F)/D, which gives
        linear XEB   F = E[D*p] - 1
        log XEB      F = E[ln(D*p)] + gamma
        HOG          F = (HOG - 1/2) / (ln2 / 2)
    Agreement between the three is the evidence that the noise really is
    depolarising-like; divergence points at coherent or leakage error.
    """
    x = d.x
    pos = x[x > 0]

    lin = float(x.mean() - 1.0)
    log = float(np.log(pos).mean() + EULER_GAMMA) if pos.size else float("nan")
    hog = float((x > LN2).mean())
    f_hog = (hog - 0.5) / (LN2 / 2.0)

    # standard error on the linear estimator
    lin_sem = float(x.std(ddof=1) / math.sqrt(x.size))

    return {
        "F_linear": lin,
        "F_linear_sem": lin_sem,
        "F_log": log,
        "HOG": hog,
        "F_HOG": f_hog,
        "mean_x": float(x.mean()),
        "var_x": float(x.var(ddof=1)),
        "median_x": float(np.median(x)),
        # O(S * N * log S) reduction to find unique rows; expensive but cached locally
        "unique_frac": float(len(np.unique(d.bits, axis=0)) / d.n_shots),
        "mean_weight": float(d.weights.mean()),
    }


def stats_text(d: XEBData, t: dict[str, float]) -> str:
    return (
        f"qubits N = {d.n_qubits}    shots S = {d.n_shots}    D = 2^{d.n_qubits}\n"
        f"amplitudes read as: {d.amp_kind}\n"
        f"\n"
        f"F  linear XEB   {t['F_linear']:.4f}  +/- {t['F_linear_sem']:.4f}\n"
        f"F  log  XEB     {t['F_log']:.4f}\n"
        f"F  from HOG     {t['F_HOG']:.4f}   (HOG = {t['HOG']:.4f})\n"
        f"\n"
        f"mean D*p  {t['mean_x']:.4f}     var  {t['var_x']:.4f}\n"
        f"median    {t['median_x']:.4f}     unique shots {t['unique_frac']*100:.1f}%\n"
        f"mean Hamming weight  {t['mean_weight']:.2f} / {d.n_qubits}"
    )


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def factor_3d(n: int) -> tuple[int, int, int]:
    """Pick the most cube-like (nx, ny, nz) with nx*ny*nz >= n.
    
    Searches a slight envelope of padded volumes to find a geometry that 
    factors cleanly without extreme skews (e.g., snaps N=127 to 8x4x4 instead
    of a massive stick). lattice_positions only generates coordinates up to n, 
    leaving 'ghost' slots in the visualization grid.
    """
    if n < 1:
        raise ValueError(f"Lattice size must be >= 1, got {n}")

    # Search envelope: allow up to 25% padding or at least 15 extra slots
    max_v = max(n + 15, int(n * 1.25))
    
    best = None
    for v in range(n, max_v + 1):
        for nx in range(1, int(round(v ** (1 / 3))) + 3):
            if nx == 0 or v % nx:
                continue
            rem = v // nx
            for ny in range(nx, int(math.isqrt(rem)) + 2):
                if ny == 0 or rem % ny:
                    continue
                nz = rem // ny
                dims = tuple(sorted((nx, ny, nz), reverse=True))
                spread = max(dims) - min(dims)
                ghosts = v - n
                # Score candidates by minimal ghost slots (weighted heavily) and spread
                score = (ghosts * 5) + spread
                if best is None or score < best[0]:
                    best = (score, dims)
                    
    if best is not None:
        return best[1]  # type: ignore[return-value]

    # Extreme fallback if search somehow misses
    side = int(math.ceil(n ** (1 / 3)))
    return (side, side, int(math.ceil(n / (side * side))))


def lattice_positions(n: int, dims: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = dims
    idx = np.arange(n)
    xs = idx % nx
    ys = (idx // nx) % ny
    zs = idx // (nx * ny)
    return np.column_stack([xs, ys, zs]).astype(float)


def pca3(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (coords, explained_variance_ratio)."""
    m = bits.astype(np.float64)
    m -= m.mean(axis=0, keepdims=True)
    # economy SVD; S x N with S ~ 1e3-1e5 is cheap
    _, s, vt = np.linalg.svd(m, full_matrices=False)
    k = min(3, vt.shape[0])
    coords = m @ vt[:k].T
    if k < 3:
        coords = np.column_stack([coords, np.zeros((coords.shape[0], 3 - k))])
    var = (s ** 2) / max(1, m.shape[0] - 1)
    total = var.sum()
    explained = var[:3] / total if total > 0 else np.zeros(3)
    return coords, explained


def zz_correlation(bits: np.ndarray) -> np.ndarray:
    z = 1.0 - 2.0 * bits.astype(np.float64)   # |0> -> +1, |1> -> -1
    mean = z.mean(axis=0)
    cov = (z.T @ z) / z.shape[0] - np.outer(mean, mean)
    return cov


# --------------------------------------------------------------------------- #
# views
# --------------------------------------------------------------------------- #

class Viewer:
    VIEWS = ("lattice", "cloud", "corr", "porter", "landscape")

    def __init__(self, data: XEBData, dims: tuple[int, int, int] | None = None,
                 cmap: str = "viridis"):
        self.d = data
        self.stats = fidelity_table(data)
        self.dims = dims or factor_3d(data.n_qubits)
        self.cmap = cmap

        self.view = "lattice"
        self.shot = 0
        self.show_shot = False
        self.log_colour = True
        self.show_theory = True
        self.is_interactive = False

        # precompute the expensive bits once
        self.pos = lattice_positions(data.n_qubits, self.dims)
        self.marg = data.bits.mean(axis=0)
        
        self._cloud = None
        self._cloud_explained = None
        self._corr = None

        self.fig = None
        self.ax = None
        self.cax = None
        self._widgets = []

    # -- lazily cached derived products ------------------------------------ #

    @property
    def cloud(self) -> np.ndarray:
        if self._cloud is None:
            self._cloud, self._cloud_explained = pca3(self.d.bits)
        return self._cloud

    @property
    def corr(self) -> np.ndarray:
        if self._corr is None:
            self._corr = zz_correlation(self.d.bits)
        return self._corr

    # -- individual renderers ---------------------------------------------- #

    def _colour_values(self) -> np.ndarray:
        x = np.clip(self.d.x, 1e-12, None)
        return np.log10(x) if self.log_colour else self.d.x

    def draw_lattice(self, ax):
        d = self.d
        nx, ny, nz = self.dims
        pos = self.pos

        if self.show_shot:
            vals = d.bits[self.shot].astype(float)
            vmin, vmax, label = 0.0, 1.0, f"bit value, shot {self.shot}"
            sizes = 40 + 160 * vals
        else:
            vals = self.marg
            spread = max(0.02, float(np.abs(vals - 0.5).max()))
            vmin, vmax = 0.5 - spread, 0.5 + spread
            label = "P(1) marginal"
            sizes = 40 + 400 * np.abs(vals - 0.5)

        # faint nearest-neighbour bonds so the geometry is legible
        segs = []
        index = {tuple(p): i for i, p in enumerate(pos.astype(int))}
        for (px, py, pz), i in index.items():
            for step in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                nb = (px + step[0], py + step[1], pz + step[2])
                j = index.get(nb)
                if j is not None:
                    segs.append((pos[i], pos[j]))
        for a, b in segs:
            ax.plot(*zip(a, b), color="0.75", lw=0.6, alpha=0.5, zorder=1)

        sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=vals, s=sizes,
                        cmap=self.cmap, vmin=vmin, vmax=vmax,
                        edgecolors="k", linewidths=0.4, depthshade=False, zorder=3)

        if d.n_qubits <= 96:
            for i, (px, py, pz) in enumerate(pos):
                ax.text(px, py, pz + 0.22, str(i), fontsize=5.5,
                        color="0.25", ha="center")

        ax.set_title(f"qubit lattice  {nx}x{ny}x{nz}", fontsize=11)
        ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
        self._set_box(ax, (nx, ny, nz))
        return sc, label

    def draw_cloud(self, ax):
        c = self.cloud
        vals = self._colour_values()
        sc = ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=vals, s=7,
                        cmap=self.cmap, alpha=0.75, depthshade=False,
                        linewidths=0)
        sel = c[self.shot]
        ax.scatter(*sel, s=200, facecolors="none", edgecolors="crimson",
                   linewidths=1.8, depthshade=False, zorder=5)

        assert self._cloud_explained is not None
        expl = self._cloud_explained
        # for structureless bits every PC carries 1/N of the variance, so the
        # printed fractions only mean something against that baseline
        ax.set_title(
            "shot cloud, PCA of bit matrix\n"
            f"isotropic null = 1/N = {100.0 / self.d.n_qubits:.2f}% per component",
            fontsize=10,
        )
        ax.set_xlabel(f"PC1 ({expl[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({expl[1]*100:.1f}%)")
        ax.set_zlabel(f"PC3 ({expl[2]*100:.1f}%)")
        label = "log10 D*p" if self.log_colour else "D*p"
        return sc, label

    def draw_corr(self, ax):
        c = self.corr
        n = c.shape[0]
        
        if n == 1:
            ax.set_title("connected ZZ correlations: 1 qubit -- nothing to show", fontsize=11)
            ax.set_axis_off()
            return None, None
            
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        lim = max(1e-6, float(np.abs(c - np.diag(np.diag(c))).max()))
        cc = c.copy()
        np.fill_diagonal(cc, 0.0)   # the diagonal is 1 - <Z>^2 and swamps the scale

        surf = ax.plot_surface(ii, jj, cc, cmap="coolwarm", vmin=-lim, vmax=lim,
                               linewidth=0, antialiased=True, rstride=1, cstride=1)

        # +/- 2 sigma shot-noise planes: with S shots the sampling scatter on an
        # uncorrelated pair is 1/sqrt(S), so anything inside these planes is noise
        sigma = 1.0 / math.sqrt(self.d.n_shots)
        if self.show_theory:
            gx, gy = np.meshgrid([0, n - 1], [0, n - 1])
            for sign in (+1, -1):
                ax.plot_surface(gx, gy, np.full_like(gx, sign * 2 * sigma,
                                                dtype=float),
                                color="0.35", alpha=0.13, linewidth=0,
                                shade=False)

        ax.set_title(
            "connected ZZ correlations  <Z_i Z_j> - <Z_i><Z_j>\n"
            f"shot noise 1 sigma = {sigma:.4f}   |   max |C| off-diagonal = {lim:.4f}",
            fontsize=10,
        )
        ax.set_xlabel("qubit i"), ax.set_ylabel("qubit j"), ax.set_zlabel("C_ij")
        ax.set_zlim(-lim, lim)
        
        if self.is_interactive:
            ax.text2D(0.05, 0.95, "Shot selector not used in this view",
                      transform=ax.transAxes, color="0.4", fontsize=9)
                  
        return surf, "C_ij"

    def draw_porter(self, ax):
        _ensure_mpl()
        
        x = self.d.x
        hi = float(np.percentile(x, 99.5))
        edges = np.linspace(0, max(hi, 1.0), 41)
        dens, _ = np.histogram(x, bins=edges, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        bin_width = edges[1] - edges[0]

        norm = plt_colors.Normalize(vmin=0, vmax=max(dens.max(), 1e-12))
        cmap = plt_cm.get_cmap(self.cmap)
        ax.bar3d(centres - bin_width / 2, np.zeros_like(centres), np.zeros_like(centres),
                 bin_width * 0.9, 0.5, dens, color=cmap(norm(dens)),
                 shade=True, alpha=0.9, zorder=1)

        if self.show_theory:
            f_raw = self.stats["F_linear"]
            f = np.clip(f_raw, 0.0, 1.0)
            t = np.linspace(edges[0], edges[-1], 400)
            model = (1 - f) * np.exp(-t) + f * t * np.exp(-t)
            # both curves sit at the mid-depth of the bars, otherwise parallax
            # makes the visual comparison meaningless
            # curves sit at the mid-depth of the bar wall so the comparison is
            # parallax-free; render() disables computed_zorder for this view so
            # the explicit zorder below actually wins over the bar collection
            mid = np.full_like(t, 0.25)
            
            label_text = f"depolarising model, F={f:.3f}" + (" (clamped)" if f != f_raw else "")
            
            ax.plot(t, mid, model, color="crimson", lw=2.2, zorder=20,
                    label=label_text)
            ax.plot(t, mid, t * np.exp(-t), color="k", lw=1.3, ls="--",
                    zorder=20, label="ideal Porter-Thomas, F=1")
            ax.legend(loc="upper right", fontsize=8)

        ax.set_title("Porter-Thomas distribution of x = D*p", fontsize=11)
        ax.set_xlabel("x = D*p"), ax.set_ylabel(""), ax.set_zlabel("density")
        ax.set_yticks([])
        ax.set_ylim(-0.15, 0.65)
        ax.view_init(elev=20, azim=-68)
        return None, None

    def draw_landscape(self, ax):
        d = self.d
        w = d.weights
        lx = np.log10(np.clip(d.x, 1e-6, None))

        if lx.min() == lx.max():
            ax.set_title("landscape: all D*p identical -- nothing to show", fontsize=11)
            ax.set_axis_off()
            return None, None

        if w.min() == w.max():
            ax.set_title("landscape: all shots have identical Hamming weight -- nothing to show", fontsize=11)
            ax.set_axis_off()
            return None, None

        wbins = np.arange(w.min() - 0.5, w.max() + 1.5, 1.0)
        if wbins.size > 40:
            # Trades perfect half-integer edge alignment for a bounded bin count on large N
            wbins = np.linspace(w.min() - 0.5, w.max() + 0.5, 40)
        xbins = np.linspace(lx.min(), lx.max(), 32)

        h, _, _ = np.histogram2d(w, lx, bins=[wbins, xbins])
        wc = 0.5 * (wbins[:-1] + wbins[1:])
        xc = 0.5 * (xbins[:-1] + xbins[1:])
        ww, xx = np.meshgrid(wc, xc, indexing="ij")

        surf = ax.plot_surface(ww, xx, h, cmap=self.cmap, linewidth=0,
                               antialiased=True, rstride=1, cstride=1)
        ax.set_title("joint density: Hamming weight vs log10 D*p", fontsize=11)
        ax.set_xlabel("Hamming weight")
        ax.set_ylabel("log10 D*p")
        ax.set_zlabel("counts")
        return surf, "counts"

    # -- plumbing ----------------------------------------------------------- #

    @staticmethod
    def _set_box(ax, dims):
        span = max(dims)
        ax.set_xlim(-0.5, dims[0] - 0.5)
        ax.set_ylim(-0.5, dims[1] - 0.5)
        ax.set_zlim(-0.5, dims[2] - 0.5)
        ax.set_box_aspect([d / span for d in dims])

    def render(self, ax, cax=None):
        ax.clear()
        if cax is not None:
            cax.clear()
            cax.set_axis_off()

        # matplotlib sorts 3D artists by mean depth, which puts the bar3d
        # collection in front of the theory lines; opt out for that view only
        ax.computed_zorder = self.view != "porter"

        fn = {
            "lattice": self.draw_lattice,
            "cloud": self.draw_cloud,
            "corr": self.draw_corr,
            "porter": self.draw_porter,
            "landscape": self.draw_landscape,
        }[self.view]

        mappable, label = fn(ax)
        if mappable is not None and cax is not None and self.fig is not None:
            cax.set_axis_on()
            self.fig.colorbar(mappable, cax=cax, label=label or "")

    # -- interactive app ----------------------------------------------------- #

    def show(self):
        self.is_interactive = True
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

        self.fig = plt.figure(figsize=(14, 8.5))
        self.fig.canvas.manager.set_window_title(
            f"xeb3d  --  {os.path.basename(self.d.amps_path)}"
        )

        self.ax = self.fig.add_axes([0.28, 0.10, 0.60, 0.84], projection="3d")
        self.cax = self.fig.add_axes([0.91, 0.28, 0.015, 0.42])
        self.cax.set_axis_off()

        # ---- control column
        ax_radio = self.fig.add_axes([0.02, 0.68, 0.20, 0.24])
        ax_radio.set_title("view", fontsize=9, loc="left")
        radio = RadioButtons(ax_radio, self.VIEWS, active=0)

        ax_check = self.fig.add_axes([0.02, 0.52, 0.20, 0.13])
        ax_check.set_title("options", fontsize=9, loc="left")
        check = CheckButtons(
            ax_check,
            ["show selected shot", "log colour scale", "theory overlay"],
            [self.show_shot, self.log_colour, self.show_theory],
        )

        ax_slider = self.fig.add_axes([0.04, 0.45, 0.17, 0.03])
        slider = Slider(ax_slider, "shot", 0, max(0, self.d.n_shots - 1),
                        valinit=0, valstep=1)

        ax_btn = self.fig.add_axes([0.04, 0.38, 0.08, 0.045])
        button = Button(ax_btn, "save PNG")

        ax_txt = self.fig.add_axes([0.02, 0.02, 0.22, 0.33])
        ax_txt.set_axis_off()
        txt = ax_txt.text(0.0, 1.0, stats_text(self.d, self.stats),
                          va="top", ha="left", fontsize=8.5, family="monospace")

        self._widgets = [radio, check, slider, button, txt]

        def refresh():
            self.render(self.ax, self.cax)
            self.fig.canvas.draw_idle()

        def on_view(label):
            self.view = label
            refresh()

        def on_check(label):
            if label == "show selected shot":
                self.show_shot = not self.show_shot
            elif label == "log colour scale":
                self.log_colour = not self.log_colour
            elif label == "theory overlay":
                self.show_theory = not self.show_theory
            else:
                warnings.warn(f"Unknown checkbox label: {label}", stacklevel=2)
            refresh()

        def on_slide(val):
            self.shot = int(val)
            if self.view in ("lattice", "cloud"):
                refresh()

        def on_save(_evt):
            out = f"xeb3d_{self.view}.png"
            self.fig.savefig(out, dpi=180, bbox_inches="tight")
            print(f"wrote {out}")

        def on_key(evt):
            if evt.key in ("right", "left"):
                step = 1 if evt.key == "right" else -1
                slider.set_val(int(np.clip(self.shot + step, 0, self.d.n_shots - 1)))
            elif evt.key in "12345":
                idx = int(evt.key) - 1
                if idx < len(self.VIEWS):
                    radio.set_active(idx)

        radio.on_clicked(on_view)
        check.on_clicked(on_check)
        slider.on_changed(on_slide)
        button.on_clicked(on_save)
        self.fig.canvas.mpl_connect("key_press_event", on_key)

        refresh()
        plt.show()

    def save_all(self, outdir: str, dpi: int = 170):
        import matplotlib.pyplot as plt

        os.makedirs(outdir, exist_ok=True)
        written = []
        
        was_interactive = self.is_interactive
        self.is_interactive = False
        
        for view in self.VIEWS:
            self.view = view
            self.fig = plt.figure(figsize=(10, 7.5))
            ax = self.fig.add_axes([0.05, 0.06, 0.80, 0.88], projection="3d")
            cax = self.fig.add_axes([0.89, 0.25, 0.02, 0.5])
            cax.set_axis_off()
            self.render(ax, cax)
            path = os.path.join(outdir, f"xeb3d_{view}.png")
            self.fig.savefig(path, dpi=dpi)
            plt.close(self.fig)
            written.append(path)
            
        self.fig = None
        self.view = "lattice"
        self.is_interactive = was_interactive
        return written


# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amps", help="path to the amplitudes .npy")
    ap.add_argument("--bits", help="path to the bitstrings .npy")
    ap.add_argument("--dir", default=".", help="directory to auto-discover in")
    ap.add_argument("--dims", help="qubit lattice as nx,ny,nz (product must be >= N)")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--save-all", metavar="OUTDIR",
                    help="render every view to PNG and exit (no display needed)")
    ap.add_argument("--stats", action="store_true",
                    help="print the fidelity table and exit")
    args = ap.parse_args(argv)

    amps = args.amps
    bits = args.bits
    if not (amps and bits):
        amps, bits = autodiscover(args.dir)

    data = load_xeb(amps, bits)

    dims = None
    if args.dims:
        dims = tuple(int(v) for v in args.dims.replace("x", ",").split(","))
        if len(dims) != 3 or math.prod(dims) < data.n_qubits:
            raise SystemExit(
                f"--dims must be three integers whose product is >= {data.n_qubits}"
            )

    if args.stats:
        print(stats_text(data, fidelity_table(data)))
        return 0

    if args.save_all:
        import matplotlib
        matplotlib.use("Agg")
        viewer = Viewer(data, dims, args.cmap)
        for path in viewer.save_all(args.save_all):
            print(f"wrote {path}")
        print()
        print(stats_text(data, viewer.stats))
        return 0

    Viewer(data, dims, args.cmap).show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
