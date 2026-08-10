#!/usr/bin/env python3
"""
xeb3d.py -- one-file toolkit for cross-entropy benchmarking of sampled circuits.

Three jobs that have to happen in order, in one file:

    tcount   count non-Clifford gates in a QASM and say whether per-amplitude
             strong simulation is even feasible before you spend cluster time
    probs    strong-simulate the ideal probability of each accepted sample,
             producing the amplitude array that XEB requires
    view     interactive 3D viewer + fidelity readout over samples and amps

The split matters. A weak simulator (QrackStabilizer with stochastic T-gate
rounding, e.g. dcs_post_select) emits bitstrings and nothing else. XEB needs
p_ideal(x) for every sampled x, which is a different computation on a
different simulator and cannot be recovered from samples after the fact. If
you only have samples, `view` still runs -- it disables the fidelity panel
rather than inventing a number.

Inputs
------
  .npy pair      <prefix>_xeb_amplitudes.npy  (S,)    ideal amps/mags/probs
                 <prefix>_xeb_bitstrings.npy  (S, N)  one row per shot
  result .json   {"accepted_samples": [[0,1,...], ...], ...}, probabilities
                 read from an "ideal_probs"-style key if present, else --amps

Amplitude files are sniffed: complex, signed real, magnitudes and outright
probabilities are all accepted, chosen by which reading puts mean(D*p) near
unity. Bitstrings may be bool, 0/1, +-1, or transposed.

Views
-----
  lattice      qubits on an nx*ny*nz grid, coloured by P(1) marginal or by the
               selected shot's bit pattern
  cloud        shots embedded by PCA of the centred bit matrix, coloured by
               log10(D*p), against the 1/N isotropic null
  corr         connected <Z_i Z_j> - <Z_i><Z_j> surface with +-2 sigma
               shot-noise planes
  porter       Porter-Thomas histogram of x = D*p with the depolarising model
               (1-F)e^-x + F*x*e^-x overlaid
               -- without probabilities: Hamming weight vs Binomial(N, 1/2)
  landscape    joint density of (Hamming weight, log10 x)
               -- without probabilities: weight vs shot index, which is where
               RNG state duplicated across forked workers shows up

Controls
--------
  radio buttons / 1..5   switch view
  slider / left,right    select shot (highlighted in lattice and cloud)
  check boxes            show selected shot; log colour; theory overlay
  save button            dump the current view to PNG

Usage
-----
  python xeb3d.py view result.json                     # samples only
  python xeb3d.py view result.json --amps amps.npy
  python xeb3d.py view --amps A.npy --bits B.npy
  python xeb3d.py view --save-all out/                 # headless PNG dump
  python xeb3d.py stats result.json
  python xeb3d.py tcount circuit.qasm
  python xeb3d.py probs circuit.qasm result.json --limit 1 --workers 32

The subcommand may be omitted; a bare path is treated as `view`.

view/stats/tcount need only numpy + matplotlib. probs needs pyqrack + qiskit.
"""

from __future__ import annotations

import argparse
import glob
import re
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

EULER_GAMMA = 0.5772156649015329
LN2 = math.log(2.0)


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #

@dataclass
class XEBData:
    bits: np.ndarray             # (S, N) bool, True == measured 1
    p: np.ndarray | None         # (S,) ideal prob of each sample, or None
    n_qubits: int
    n_shots: int
    amp_kind: str                # how the amplitude file was interpreted
    amps_path: str | None
    bits_path: str
    meta: dict | None = None     # run metadata carried through from JSON

    @property
    def has_probs(self) -> bool:
        return self.p is not None

    @property
    def dim(self) -> float:
        return 2.0 ** self.n_qubits

    @property
    def x(self) -> np.ndarray:
        """Normalised ideal probability D*p. Porter-Thomas mean is 1."""
        if self.p is None:
            raise ValueError(
                "no ideal probabilities loaded -- XEB is undefined from "
                "bitstrings alone. Supply an amplitudes file with --amps."
            )
        return self.dim * self.p

    @property
    def weights(self) -> np.ndarray:
        return self.bits.sum(axis=1)


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

    if b.shape[0] != n_shots and b.shape[1] == n_shots:
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


# keys a sampler might plausibly use for the bitstring array / probabilities
_BIT_KEYS = ("accepted_samples", "samples", "bitstrings", "bits", "shots_data")
_PROB_KEYS = ("ideal_probs", "ideal_probabilities", "amplitudes",
              "ideal_amplitudes", "probs", "probabilities")


def load_json(json_path: str, amps_path: str | None = None) -> XEBData:
    """Load a sampler result JSON (e.g. dcs_post_select_result.json).

    Bitstrings come from the JSON. Ideal probabilities come from the JSON too
    if the sampler wrote them, otherwise from a separate --amps .npy, otherwise
    not at all -- in which case the viewer runs in samples-only mode and the
    XEB panel is disabled rather than silently invented.
    """
    import json

    with open(json_path) as f:
        blob = json.load(f)

    if not isinstance(blob, dict):
        raise ValueError(f"{json_path}: expected a JSON object at top level")

    key = next((k for k in _BIT_KEYS if k in blob), None)
    if key is None:
        raise ValueError(
            f"{json_path}: no bitstring array found (looked for "
            f"{', '.join(_BIT_KEYS)})"
        )
    raw = np.asarray(blob[key])
    if raw.size == 0:
        raise ValueError(f"{json_path}: {key!r} is empty -- zero accepted shots")
        
    if raw.ndim == 2 and raw.shape[0] < raw.shape[1]:
        print("warning: JSON array has more columns than rows, assuming transposed (N, S) -> (S, N)", file=sys.stderr)
        raw = raw.T

    n_shots = raw.shape[0]
    bits = _normalise_bits(raw, n_shots)
    n_qubits = bits.shape[1]

    p = None
    kind = "none -- samples only, no ideal probabilities"
    src = None

    pkey = next((k for k in _PROB_KEYS if k in blob), None)
    if pkey is not None:
        p, kind = _interpret_amplitudes(np.asarray(blob[pkey]).ravel(), n_qubits)
        kind = f"{kind}  [from JSON key {pkey!r}]"
        src = f"{json_path}:{pkey}"
    elif amps_path:
        amps = np.load(amps_path, allow_pickle=False).ravel()
        if amps.size != n_shots:
            raise ValueError(
                f"amplitude count ({amps.size}) does not match accepted shots "
                f"({n_shots}); these are not from the same run"
            )
        p, kind = _interpret_amplitudes(amps, n_qubits)
        src = amps_path

    meta = {k: v for k, v in blob.items() if k not in _BIT_KEYS + _PROB_KEYS}

    return XEBData(bits=bits, p=p, n_qubits=n_qubits, n_shots=n_shots,
                   amp_kind=kind, amps_path=src, bits_path=json_path, meta=meta)


def load_any(bits_path: str, amps_path: str | None = None) -> XEBData:
    """Dispatch on the bitstring container: .json result blob or .npy array."""
    if bits_path.lower().endswith(".json"):
        return load_json(bits_path, amps_path)
    if amps_path is None:
        raise SystemExit(
            "a .npy bitstring file needs a matching --amps file; pass a "
            "sampler result .json instead if you only have samples"
        )
    return load_xeb(amps_path, bits_path)


def autodiscover(directory: str = ".") -> tuple[str, str]:
    amps = sorted(glob.glob(os.path.join(directory, "*amplitude*.npy")))
    bits = sorted(glob.glob(os.path.join(directory, "*bitstring*.npy")))
    if not amps or not bits:
        raise SystemExit(
            "could not auto-discover *amplitude*.npy / *bitstring*.npy in "
            f"{os.path.abspath(directory)!r}; pass --amps and --bits explicitly"
        )
        
    def prefix(p): 
        return re.sub(r'(amplitude|bitstring).*', '', os.path.basename(p))
        
    for a in amps:
        pa = prefix(a)
        match = [b for b in bits if prefix(b) == pa]
        if match:
            return a, match[0]
            
    print("warning: falling back to alphabetical pairing in autodiscover", file=sys.stderr)
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
        "unique_frac": float(len(np.unique(d.bits, axis=0)) / d.n_shots),
        "mean_weight": float(d.weights.mean()),
    }


def sample_health(d: XEBData) -> str:
    """Checks that need only bitstrings: marginal bias and pairwise correlation
    measured against the finite-sample floor. A parallel sampler that shares
    RNG state across workers shows up here as duplicate rows or an inflated
    correlation tail, so it is worth reading even when probabilities exist."""
    s_, n = d.n_shots, d.n_qubits
    marg_sigma = 0.5 / math.sqrt(s_)
    max_bias = float(np.abs(d.bits.mean(axis=0) - 0.5).max()) / marg_sigma

    c = zz_correlation(d.bits)
    np.fill_diagonal(c, 0.0)
    max_corr = float(np.abs(c).max()) * math.sqrt(s_)

    dup = s_ - len(np.unique(d.bits, axis=0))
    w = d.weights

    return (
        f"marginal bias    max {max_bias:.2f} sigma\n"
        f"ZZ correlation   max {max_corr:.2f} sigma over {n*(n-1)//2} pairs\n"
        f"duplicate shots  {dup}\n"
        f"Hamming weight   {w.mean():.2f} +/- {w.std():.2f}"
        f"   (binomial {n/2:.1f} +/- {math.sqrt(n)/2:.2f})"
    )


def stats_text(d: XEBData, t: dict[str, float] | None) -> str:
    head = (
        f"qubits N = {d.n_qubits}    shots S = {d.n_shots}    D = 2^{d.n_qubits}\n"
        f"amplitudes read as: {d.amp_kind}\n"
    )

    if d.meta:
        interesting = ("qasm_path", "shots", "accepted", "acceptance_rate",
                       "workers", "seed", "shots_per_second")
        rows = [f"{k} = {d.meta[k]}" for k in interesting if k in d.meta]
        if rows:
            head += "\nrun: " + "\n     ".join(rows) + "\n"

    if t is None:
        return (
            head
            + "\nXEB UNAVAILABLE -- no ideal probabilities.\n"
              "Linear/log XEB and HOG all require p_ideal(x) for each\n"
              "sampled x; bitstrings alone do not determine them.\n"
              "\n" + sample_health(d)
        )

    return head + "\n" + _fidelity_block(d, t) + "\n\n" + sample_health(d)


def _fidelity_block(d: XEBData, t: dict[str, float]) -> str:
    return (
        f"F  linear XEB   {t['F_linear']:.4f}  +/- {t['F_linear_sem']:.4f}\n"
        f"F  log   XEB    {t['F_log']:.4f}\n"
        f"F  from HOG     {t['F_HOG']:.4f}   (HOG = {t['HOG']:.4f})\n"
        f"\n"
        f"mean D*p  {t['mean_x']:.4f}     var  {t['var_x']:.4f}\n"
        f"median    {t['median_x']:.4f}"
    )


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #

def factor_3d(n: int) -> tuple[int, int, int]:
    """Pick the most cube-like (nx, ny, nz) with nx*ny*nz == n, else pad."""
    best = None
    for nx in range(1, int(round(n ** (1 / 3))) + 3):
        if nx == 0 or n % nx:
            continue
        rem = n // nx
        for ny in range(nx, int(math.isqrt(rem)) + 2):
            if ny == 0 or rem % ny:
                continue
            nz = rem // ny
            dims = tuple(sorted((nx, ny, nz), reverse=True))
            spread = max(dims) - min(dims)
            if best is None or spread < best[0]:
                best = (spread, dims)
    if best is not None:
        return best[1]  # type: ignore[return-value]

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
    m = bits.astype(np.float64)
    m -= m.mean(axis=0, keepdims=True)
    
    n_shots, n_qubits = m.shape
    
    # Accelerated PCA computation: Use covariance matrix when S >> N to drop
    # quadratic complexity. Avoids hanging on high shot volumes.
    if n_shots > n_qubits:
        cov = (m.T @ m)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # eigh returns ascending order; reverse to get primary components
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        k = min(3, n_qubits)
        vt = eigvecs[:, :k].T
        s = np.sqrt(np.maximum(eigvals, 0))
    else:
        # economy SVD fallback for edge cases
        _, s, vt = np.linalg.svd(m, full_matrices=False)
        k = min(3, vt.shape[0])

    coords = m @ vt[:k].T
    if k < 3:
        coords = np.column_stack([coords, np.zeros((coords.shape[0], 3 - k))])
        
    var = (s ** 2) / max(1, n_shots - 1)
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
        self.stats = fidelity_table(data) if data.has_probs else None
        self.dims = dims or factor_3d(data.n_qubits)
        self.cmap = cmap

        self.view = "lattice"
        self.shot = 0
        self.show_shot = False
        self.log_colour = True
        self.show_theory = True

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

    def _colour_values(self) -> tuple[np.ndarray, str]:
        if not self.d.has_probs:
            return self.d.weights.astype(float), "Hamming weight"
        x = np.clip(self.d.x, 1e-12, None)
        if self.log_colour:
            return np.log10(x), "log10 D*p"
        return self.d.x, "D*p"

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
        vals, label = self._colour_values()
        sc = ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=vals, s=7,
                        cmap=self.cmap, alpha=0.75, depthshade=False,
                        linewidths=0)
        sel = c[self.shot]
        ax.scatter(*sel, s=200, facecolors="none", edgecolors="crimson",
                   linewidths=1.8, depthshade=False, zorder=5)

        expl = self._cloud_explained if self._cloud_explained is not None else np.zeros(3)
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
        return sc, label

    def draw_corr(self, ax):
        c = self.corr
        n = c.shape[0]
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
        return surf, "C_ij"

    def draw_porter(self, ax):
        if not self.d.has_probs:
            return self._draw_weight_hist(ax)
        x = self.d.x
        hi = float(np.percentile(x, 99.5))
        edges = np.linspace(0, max(hi, 1.0), 41)
        dens, _ = np.histogram(x, bins=edges, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        w = edges[1] - edges[0]

        norm = plt_colors.Normalize(vmin=0, vmax=max(dens.max(), 1e-12))
        cmap = plt_cm.get_cmap(self.cmap)
        ax.bar3d(centres - w / 2, np.zeros_like(centres), np.zeros_like(centres),
                 w * 0.9, 0.5, dens, color=cmap(norm(dens)),
                 shade=True, alpha=0.9, zorder=1)

        if self.show_theory:
            f = np.clip((self.stats or {}).get("F_linear", 0.0), 0.0, 1.0)
            t = np.linspace(edges[0], edges[-1], 400)
            model = (1 - f) * np.exp(-t) + f * t * np.exp(-t)
            # both curves sit at the mid-depth of the bars, otherwise parallax
            # makes the visual comparison meaningless
            # curves sit at the mid-depth of the bar wall so the comparison is
            # parallax-free; render() disables computed_zorder for this view so
            # the explicit zorder below actually wins over the bar collection
            mid = np.full_like(t, 0.25)
            ax.plot(t, mid, model, color="crimson", lw=2.2, zorder=20,
                    label=f"depolarising model, F={f:.3f}")
            ax.plot(t, mid, t * np.exp(-t), color="k", lw=1.3, ls="--",
                    zorder=20, label="ideal Porter-Thomas, F=1")
            ax.legend(loc="upper right", fontsize=8)

        ax.set_title("Porter-Thomas distribution of x = D*p", fontsize=11)
        ax.set_xlabel("x = D*p"), ax.set_ylabel(""), ax.set_zlabel("density")
        ax.set_yticks([])
        ax.set_ylim(-0.15, 0.65)
        ax.view_init(elev=20, azim=-68)
        return None, None

    def _draw_weight_hist(self, ax):
        """Samples-only substitute for the Porter-Thomas view.

        Without p_ideal the only distribution available is over Hamming
        weights, which for a scrambling circuit should be Binomial(N, 1/2).
        This is a necessary-but-far-from-sufficient check: it catches a dead
        or biased sampler, it says nothing whatsoever about fidelity."""
        d = self.d
        w = d.weights
        edges = np.arange(w.min() - 0.5, w.max() + 1.5, 1.0)
        dens, _ = np.histogram(w, bins=edges, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])

        norm = plt_colors.Normalize(vmin=0, vmax=max(dens.max(), 1e-12))
        cmap = plt_cm.get_cmap(self.cmap)
        ax.bar3d(centres - 0.45, np.zeros_like(centres), np.zeros_like(centres),
                 0.9, 0.5, dens, color=cmap(norm(dens)), shade=True,
                 alpha=0.9, zorder=1)

        if self.show_theory:
            from math import comb
            t = np.arange(edges[0] + 0.5, edges[-1] + 0.5)
            binom = np.array([comb(d.n_qubits, int(k)) for k in t],
                             dtype=float) * 2.0 ** (-d.n_qubits)
            ax.plot(t, np.full_like(t, 0.25), binom, color="crimson", lw=2.2,
                    zorder=20, label=f"Binomial({d.n_qubits}, 1/2)")
            ax.legend(loc="upper right", fontsize=8)

        ax.set_title(
            "Hamming weight distribution  (no ideal probabilities loaded)\n"
            "sampler sanity check only -- this is NOT a fidelity measure",
            fontsize=10,
        )
        ax.set_xlabel("Hamming weight"), ax.set_zlabel("density")
        ax.set_yticks([])
        ax.set_ylim(-0.15, 0.65)
        ax.view_init(elev=20, azim=-68)
        return None, None

    def draw_landscape(self, ax):
        d = self.d
        if not d.has_probs:
            return self._draw_drift(ax)
        w = d.weights
        lx = np.log10(np.clip(d.x, 1e-6, None))

        wbins = np.arange(w.min() - 0.5, w.max() + 1.5, 1.0)
        if wbins.size > 40:
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

    def _draw_drift(self, ax):
        """Samples-only substitute for the joint-density view.

        Hamming weight distribution as a function of position in the shot
        sequence. For a correctly parallelised sampler this is flat: shots are
        independent draws regardless of which worker produced them. A ridge
        that shifts or repeats across batches is the signature of duplicated
        RNG state after fork, which is the failure mode a process pool most
        easily introduces and which per-shot statistics alone will not show."""
        d = self.d
        w = d.weights
        # size both axes so the mean cell holds ~5 counts; at a few hundred
        # shots unit-width weight bins give a 1-count moonscape you cannot read
        target = 5.0
        nb = int(np.clip(round(math.sqrt(d.n_shots / target)), 4, 20))
        nw = int(np.clip(round(d.n_shots / (nb * target)), 4, 24))
        bedges = np.linspace(0, d.n_shots, nb + 1)
        wedges = np.linspace(w.min() - 0.5, w.max() + 0.5, nw + 1)

        idx = np.arange(d.n_shots)
        h, _, _ = np.histogram2d(idx, w, bins=[bedges, wedges])
        bc = 0.5 * (bedges[:-1] + bedges[1:])
        wc = 0.5 * (wedges[:-1] + wedges[1:])
        bb, ww = np.meshgrid(bc, wc, indexing="ij")

        surf = ax.plot_surface(bb, ww, h, cmap=self.cmap, linewidth=0,
                               antialiased=True, rstride=1, cstride=1)
        ax.set_title("Hamming weight vs position in shot sequence\n"
                     "flat ridge == independent shots across workers",
                     fontsize=10)
        ax.set_xlabel("shot index")
        ax.set_ylabel("Hamming weight")
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
        global plt_colors, plt_cm
        if plt_colors is None:
            _install_mpl_shims()
            
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
        if mappable is not None and cax is not None:
            cax.set_axis_on()
            self.fig.colorbar(mappable, cax=cax, label=label or "")

    # -- interactive app ----------------------------------------------------- #

    def show(self):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

        self.fig = plt.figure(figsize=(14, 8.5))
        
        # fix: allow bare bitstring sets without throwing a NoneType error
        title_source = self.d.amps_path or self.d.bits_path
        self.fig.canvas.manager.set_window_title(
            f"xeb3d  --  {os.path.basename(title_source)}"
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
            if label.startswith("show"):
                self.show_shot = not self.show_shot
            elif label.startswith("log"):
                self.log_colour = not self.log_colour
            else:
                self.show_theory = not self.show_theory
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
        return written


# --------------------------------------------------------------------------- #
# non-Clifford gate counting -- feasibility gate for everything below
# --------------------------------------------------------------------------- #

_CLIFFORD_ANGLES = re.compile(
    r"\s*-?\s*(pi\s*/\s*2|pi|0(?:\.0*)?|2\s*\*?\s*pi|3\s*\*?\s*pi\s*/\s*2)\s*", re.IGNORECASE
)
_PARAMETRIC = ("rz", "rx", "ry", "p", "u1", "u2", "u3", "u", "crz", "cp", "cu1")


def count_nonclifford(qasm_path: str) -> tuple[int, dict[str, int]]:
    """Count T-like gates in a QASM file.

    T and Tdg always count. Rotations count only when the angle is not a
    multiple of pi/2 -- a circuit emitted by a transpiler is often full of
    rz(pi/2), and treating those as non-Clifford inflates the estimate to the
    point of being useless.
    """
    counts: dict[str, int] = {}
    nonclifford = 0

    with open(qasm_path) as f:
        for line in f:
            line = line.split("//")[0].strip()
            if not line or line.startswith(
                ("OPENQASM", "include", "gate ", "qreg", "creg", "}")
            ):
                continue
            m = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)", line)
            if not m:
                continue
            g = m.group(1).lower()
            counts[g] = counts.get(g, 0) + 1

            if g in ("t", "tdg"):
                nonclifford += 1
            elif g in _PARAMETRIC:
                ang = re.search(r"\(([^)]*)\)", line)
                if ang and not _CLIFFORD_ANGLES.fullmatch(ang.group(1)):
                    nonclifford += 1

    return nonclifford, counts


def tcount_report(qasm_path: str) -> str:
    t, counts = count_nonclifford(qasm_path)
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:12]
    # approximate stabilizer-rank scaling, chi ~ 2^(0.23 t)
    chi = 0.23 * t
    verdict = ("tractable" if chi < 30 else
               "borderline -- benchmark one sample first" if chi < 45 else
               "INTRACTABLE: per-amplitude strong simulation will not finish")
    return (
        f"{qasm_path}\n"
        f"  gates: {', '.join(f'{g}={c}' for g, c in top)}\n"
        f"  non-Clifford (T-like) count t = {t}\n"
        f"  approximate stabilizer rank chi ~ 2^(0.23t) = 2^{chi:.1f}\n"
        f"  per-amplitude strong simulation: {verdict}"
    )


# --------------------------------------------------------------------------- #
# ideal probabilities -- the half a weak simulator never computes
# --------------------------------------------------------------------------- #

def _collapse(sim, q, bit):
    """Force qubit q to `bit`. Requires pyqrack's force_m for the chain rule."""
    fn = getattr(sim, "force_m", None)
    if fn is not None:
        return fn(q, bool(bit))
        
    raise RuntimeError(
        "Modern pyqrack with force_m is required for the chain rule. "
        "m() is probabilistic and cannot guarantee the required post-selection path."
    )


def log_prob_of(sim_factory, bits, n_data, n_ancilla):
    """log p(x, syndrome=0) and log P(syndrome=0) for one sample.

    Chain rule rather than a state-vector amplitude lookup, since a 97-qubit
    ket cannot be materialised:

        p(x) = prod_i p(x_i | x_0 .. x_{i-1})

    read off prob(q) and then collapsed with force_m(q, x_i). Accumulated in
    log space -- p is O(2^-70) and underflows float64 around qubit 50.

    Ancillas are forced to 0 first, so the result is the post-selected
    probability p(x | s=0) = p(x, s=0) / P(s=0), which is the distribution
    the accepted samples are actually drawn from.
    """
    sim = sim_factory()
    try:
        log_p = 0.0
        log_syn = 0.0

        for q in range(n_data, n_data + n_ancilla):
            p0 = max(1.0 - float(sim.prob(q)), 0.0)
            if p0 <= 0.0:
                return -math.inf, -math.inf
            log_syn += math.log(p0)
            _collapse(sim, q, 0)

        for i in range(n_data):
            p1 = float(sim.prob(i))
            pi = p1 if bits[i] else 1.0 - p1
            if pi <= 0.0:
                return -math.inf, log_syn
            log_p += math.log(pi)
            _collapse(sim, i, int(bits[i]))

        return log_p, log_syn
    finally:
        del sim


def compute_ideal_probs(qasm_path, result_json, n_data, n_ancilla,
                        out_path, limit=None, use_clone=True, force=False, workers=1):
    """Strong-simulate p_ideal for each accepted sample and save magnitudes.

    Only tractable while the T-count keeps the state near-Clifford;
    QrackSimulator drifts toward dense representation as doping rises. Run
    `tcount` on the QASM first -- at the reference circuit's 468 T gates this
    will not finish, and that intractability is the advantage claim rather
    than a bug to route around.
    """
    import json
    import time

    # feasibility gate first: a job that cannot finish should not start
    t_gates, _ = count_nonclifford(qasm_path)
    chi = 0.23 * t_gates
    print(f"non-Clifford count t = {t_gates}, chi ~ 2^{chi:.1f}", file=sys.stderr)
    if chi >= 45 and not force:
        raise SystemExit(
            f"\nrefusing to start: at t = {t_gates} the stabilizer rank is\n"
            f"~2^{chi:.1f} and per-amplitude strong simulation will not finish.\n"
            f"This is the hardness the circuit was designed to have, not a\n"
            f"configuration problem. Options: run a less-doped instance of the\n"
            f"same family to validate the pipeline, or use the syndrome-based\n"
            f"certificate the circuit provides instead of XEB.\n"
            f"Pass --force to override."
        )

    with open(result_json) as f:
        blob = json.load(f)

    key = next((k for k in _BIT_KEYS if k in blob), None)
    if key is None:
        raise SystemExit(f"{result_json}: no bitstring array found")
    samples = np.asarray(blob[key], dtype=np.uint8)
    if samples.shape[1] != n_data:
        raise SystemExit(
            f"samples have {samples.shape[1]} columns, --n-data is {n_data}"
        )
    if limit:
        samples = samples[:limit]

    try:
        from pyqrack import QrackSimulator
        from qiskit import QuantumCircuit
    except ImportError as e:
        raise SystemExit(
            f"the `probs` subcommand needs pyqrack and qiskit ({e}). "
            "view/stats/tcount do not."
        )

    n_qubits = n_data + n_ancilla
    qc = QuantumCircuit.from_qasm_file(qasm_path)
    if qc.num_qubits != n_qubits:
        raise SystemExit(f"circuit has {qc.num_qubits} qubits, expected {n_qubits}")

    print(f"strong-simulating {len(samples)} samples on {n_qubits} qubits",
          file=sys.stderr)

    base = None
    if use_clone:
        t0 = time.perf_counter()
        base = QrackSimulator(n_qubits)
        base.run_qiskit_circuit(qc, shots=0)
        if not hasattr(base, "clone"):
            print("  no clone() here; re-running the circuit per sample",
                  file=sys.stderr)
            del base
            base = None
        else:
            print(f"  circuit applied once in {time.perf_counter() - t0:.1f}s, "
                  f"cloning per sample", file=sys.stderr)

    def factory():
        if base is not None:
            return base.clone()
        sim = QrackSimulator(n_qubits)
        sim.run_qiskit_circuit(qc, shots=0)
        return sim

    log_ps = np.empty(len(samples))
    log_syn = np.empty(len(samples))
    t0 = time.perf_counter()

    if workers > 1 and base is not None:
        import concurrent.futures
        import queue
        
        actual_workers = min(workers, len(samples))
        print(f"  running PyQrack concurrent worker pool with {actual_workers} threads", file=sys.stderr)
        
        # Pre-clone serially to avoid C++ concurrent read tearing on the base object
        worker_bases = queue.Queue()
        for _ in range(actual_workers):
            worker_bases.put(base.clone())
            
        def safe_factory():
            wb = worker_bases.get()
            try:
                return wb.clone()
            finally:
                worker_bases.put(wb)
        
        def _sim_task(args_tuple):
            k, row = args_tuple
            lp, ls = log_prob_of(safe_factory, row, n_data, n_ancilla)
            return k, lp, ls

        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as pool:
            futures = {pool.submit(_sim_task, (k, row)): k for k, row in enumerate(samples)}
            completed = 0
            for fut in concurrent.futures.as_completed(futures):
                k, lp, ls = fut.result()
                log_ps[k] = lp
                log_syn[k] = ls
                completed += 1
                if completed % 10 == 0 or completed == len(samples):
                    el = time.perf_counter() - t0
                    print(f"  {completed}/{len(samples)}  {el / completed:.2f}s per sample, "
                          f"eta {(len(samples) - completed) * el / completed / 60:.1f} min",
                          file=sys.stderr, flush=True)

        # Proactively release VRAM backings without waiting on GC
        while not worker_bases.empty():
            wb = worker_bases.get_nowait()
            del wb

    else:
        for k, row in enumerate(samples):
            log_ps[k], log_syn[k] = log_prob_of(factory, row, n_data, n_ancilla)
            if k % 10 == 9 or k == len(samples) - 1:
                el = time.perf_counter() - t0
                print(f"  {k + 1}/{len(samples)}  {el / (k + 1):.2f}s per sample, "
                      f"eta {(len(samples) - k - 1) * el / (k + 1) / 60:.1f} min",
                      file=sys.stderr, flush=True)

    log_post = log_ps - log_syn
    amps = np.exp(0.5 * log_post)      # xeb3d reads magnitudes
    np.save(out_path, amps)

    x = np.exp(log_post + n_data * math.log(2.0))
    print()
    print(f"P(syndrome=0) = {math.exp(log_syn.mean()):.6f} "
          f"(mean over samples; 1.0 means code-preserving doping held)")
    print(f"mean D*p = {x.mean():.4f}   ->  linear XEB F = {x.mean() - 1:.4f}")
    print(f"wrote {out_path}  ({len(amps)} magnitudes)")
    print(f"\nnow: python xeb3d.py view {result_json} --amps {out_path}")
    return amps


# --------------------------------------------------------------------------- #

SUBCOMMANDS = ("view", "stats", "tcount", "probs")


def _build_data(args) -> XEBData:
    bits = args.input or args.bits
    amps = args.amps

    if bits is None:
        if amps is None:
            amps, bits = autodiscover(args.dir)
        else:
            raise SystemExit("--amps given without a bitstring source")

    data = load_any(bits, amps)

    if not data.has_probs:
        print(
            "note: no ideal probabilities in this input, so XEB is undefined.\n"
            "      Geometry and sampler-health views work; the fidelity panel\n"
            "      and probability colouring are disabled. Use the `probs`\n"
            "      subcommand to compute them, if the T-count allows it.\n",
            file=sys.stderr,
        )
    return data


def _parse_dims(spec, n_qubits):
    if not spec:
        return None
    dims = tuple(int(v) for v in spec.replace("x", ",").split(","))
    if len(dims) != 3 or math.prod(dims) != n_qubits:
        raise SystemExit(f"--dims must be three integers multiplying to {n_qubits}")
    return dims


def _add_input_args(p):
    p.add_argument("input", nargs="?",
                   help="sampler result .json, or omit to auto-discover .npy")
    p.add_argument("--amps", help="ideal amplitudes/probabilities .npy")
    p.add_argument("--bits", help="bitstrings .npy or result .json")
    p.add_argument("--dir", default=".", help="directory to auto-discover in")


def main(argv=None):
    # piping into head/less should exit quietly, not dump a BrokenPipeError
    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, AttributeError, ValueError):
        pass

    argv = list(sys.argv[1:] if argv is None else argv)

    # a bare path means `view`; keeps the older single-purpose invocation working
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "view")
    elif argv and argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv.insert(0, "view")
    elif not argv:
        argv = ["view"]

    ap = argparse.ArgumentParser(
        prog="xeb3d.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("view", help="interactive 3D viewer",
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_input_args(pv)
    pv.add_argument("--dims", help="qubit lattice as nx,ny,nz (product == N)")
    pv.add_argument("--cmap", default="viridis")
    pv.add_argument("--save-all", metavar="OUTDIR",
                    help="render every view to PNG and exit (no display needed)")
    pv.add_argument("--dpi", type=int, default=170)

    ps = sub.add_parser("stats", help="print the fidelity and health table")
    _add_input_args(ps)

    pt = sub.add_parser("tcount", help="count non-Clifford gates in a QASM")
    pt.add_argument("qasm_path")

    pp = sub.add_parser("probs", help="strong-simulate ideal probabilities")
    pp.add_argument("qasm_path")
    pp.add_argument("result_json")
    pp.add_argument("--n-data", type=int, default=70)
    pp.add_argument("--n-ancilla", type=int, default=27)
    pp.add_argument("--out", default="dcs_amplitudes.npy")
    pp.add_argument("--limit", type=int, default=None,
                    help="only do the first K samples; run with 1 first, the "
                         "per-sample cost is the whole feasibility question")
    pp.add_argument("--no-clone", dest="use_clone", action="store_false",
                    help="re-run the circuit per sample instead of cloning")
    pp.add_argument("--force", action="store_true",
                    help="run even when the T-count says it cannot finish")
    pp.add_argument("--workers", type=int, default=1,
                    help="Number of concurrent workers for strong simulation. "
                         "Scales well on multi-core systems when cloning is supported.")

    args = ap.parse_args(argv)

    if args.cmd == "tcount":
        print(tcount_report(args.qasm_path))
        return 0

    if args.cmd == "probs":
        compute_ideal_probs(args.qasm_path, args.result_json, args.n_data,
                            args.n_ancilla, args.out, args.limit,
                            args.use_clone, args.force, args.workers)
        return 0

    data = _build_data(args)

    if args.cmd == "stats":
        print(stats_text(data, fidelity_table(data) if data.has_probs else None))
        return 0

    dims = _parse_dims(args.dims, data.n_qubits)

    if args.save_all:
        import matplotlib
        matplotlib.use("Agg")
        viewer = Viewer(data, dims, args.cmap)
        for path in viewer.save_all(args.save_all, args.dpi):
            print(f"wrote {path}")
        print()
        print(stats_text(data, viewer.stats))
        return 0

    Viewer(data, dims, args.cmap).show()
    return 0


def _install_mpl_shims():
    """Bind the colour helpers used inside the renderers, matplotlib-version safe."""
    global plt_colors, plt_cm
    import matplotlib.colors as plt_colors  # noqa: F811

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


plt_colors = None
plt_cm = None


if __name__ == "__main__":
    sys.exit(main())
