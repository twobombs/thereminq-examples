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
  python xeb3d.py probs circuit.qasm result.json --limit 1

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


def load_json(json_path: str, amps_path: str | None = None,
              transpose: bool = False) -> XEBData:
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

    # Rows are shots by construction -- "accepted_samples" is a list of samples,
    # one per accepted shot. Do NOT infer transposition from shape: a run with
    # a realistic acceptance rate legitimately has far fewer accepted shots than
    # qubits (the reference experiment accepts ~6e-4), and guessing from
    # shape < shape would silently reinterpret 12 shots of 70 qubits as 70
    # shots of 12 and compute XEB against D = 2^12.
    if transpose:
        raw = raw.T
    elif raw.ndim == 2 and raw.shape[0] < raw.shape[1] and amps_path:
        # only when an amplitude file provides independent evidence
        n_amp = np.load(amps_path, allow_pickle=False).size
        if n_amp == raw.shape[1] and n_amp != raw.shape[0]:
            print(f"note: amplitude count {n_amp} matches columns not rows; "
                  f"transposing bitstrings", file=sys.stderr)
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


def load_any(bits_path: str, amps_path: str | None = None,
             transpose: bool = False) -> XEBData:
    """Dispatch on the bitstring container: .json result blob or .npy array."""
    if bits_path.lower().endswith(".json"):
        return load_json(bits_path, amps_path, transpose)
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

    if n_shots > n_qubits:
        cov = (m.T @ m)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        k = min(3, n_qubits)
        vt = eigvecs[:, :k].T
        s = np.sqrt(np.maximum(eigvals, 0))
    else:
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

_NUMERIC_EXPR = re.compile(r"[0-9eE.+\-*/() ]*")


def _eval_angle(tok: str) -> float | None:
    """Evaluate one QASM angle expression to radians, or None if unparseable."""
    tok = tok.strip().lower().replace("pi", f"({math.pi!r})")
    if not tok or not _NUMERIC_EXPR.fullmatch(tok):
        return None
    try:
        return float(eval(tok, {"__builtins__": {}}, {}))
    except Exception:
        return None


def _angles_are_clifford(param_group: str) -> bool:
    """True when every parameter is a multiple of pi/2.

    Numeric, not symbolic. A transpiler emits pi/2 as often as
    1.5707963267948966, and pattern-matching only the symbolic spelling
    counts the numeric one as a T gate -- which now matters, because an
    inflated t makes `probs` refuse a job that would have finished.
    Unparseable expressions count as non-Clifford: overcounting is the safe
    direction for a feasibility gate.
    """
    for tok in param_group.split(","):
        ang = _eval_angle(tok)
        if ang is None or abs(math.remainder(ang, math.pi / 2)) > 1e-9:
            return False
    return True


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
                if ang and not _angles_are_clifford(ang.group(1)):
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
    """Force qubit q to `bit`, using whichever API this pyqrack exposes."""
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
                        out_path, limit=None, use_clone=True, force=False,
                        workers=1, allow_circuit_mismatch=False):
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

    # The samples were drawn from one specific circuit. Computing p_ideal from
    # a different one yields mean D*p ~ 1 and F ~ 0 with no error raised
    # anywhere -- and a near-zero XEB is exactly what a genuinely bad run looks
    # like, so the mistake is invisible in the result.
    recorded = blob.get("qasm_path")
    if recorded and os.path.basename(recorded) != os.path.basename(qasm_path):
        msg = (f"circuit mismatch:\n"
               f"  samples were drawn from : {os.path.basename(recorded)}\n"
               f"  amplitudes requested from: {os.path.basename(qasm_path)}\n"
               f"Amplitudes from a different circuit give F ~ 0 and look like a\n"
               f"failed run rather than an error.")
        if allow_circuit_mismatch:
            print(f"warning: {msg}", file=sys.stderr)
        else:
            raise SystemExit(
                msg + "\nPass --allow-circuit-mismatch if this is deliberate "
                      "(e.g. the file was renamed)."
            )

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

    if workers > 1 and base is None:
        print("  warning: --workers needs clone(); running serially",
              file=sys.stderr)

    if workers > 1 and base is not None:
        # NOTE: this shares one Qrack library instance across threads. Verify
        # against a serial run on a few samples before trusting a long one --
        # dcs_post_select_paralel.py deliberately uses processes for this
        # reason, and these sims are far heavier than a stabilizer tableau.
        import concurrent.futures
        import queue

        actual_workers = min(workers, len(samples))
        print(f"  thread pool with {actual_workers} workers", file=sys.stderr)

        # pre-clone serially: concurrent reads of one base object are not
        # known to be safe on the C++ side
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

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=actual_workers) as pool:
            futures = [pool.submit(_sim_task, (k, row))
                       for k, row in enumerate(samples)]
            completed = 0
            for fut in concurrent.futures.as_completed(futures):
                k, lp, ls = fut.result()
                log_ps[k] = lp
                log_syn[k] = ls
                completed += 1
                if completed % 10 == 0 or completed == len(samples):
                    el = time.perf_counter() - t0
                    print(f"  {completed}/{len(samples)}  "
                          f"{el / completed:.2f}s per sample, "
                          f"eta {(len(samples) - completed) * el / completed / 60:.1f} min",
                          file=sys.stderr, flush=True)

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

    # provenance sidecar: which circuit and which sample set these belong to
    side = os.path.splitext(out_path)[0] + ".meta.json"
    with open(side, "w") as f:
        json.dump({"qasm_path": qasm_path,
                   "qasm_basename": os.path.basename(qasm_path),
                   "t_count": t_gates,
                   "result_json": result_json,
                   "n_samples": int(len(amps)),
                   "n_data": n_data,
                   "n_ancilla": n_ancilla}, f, indent=2)

    x = np.exp(log_post + n_data * math.log(2.0))
    print()
    print(f"P(syndrome=0) = {math.exp(log_syn.mean()):.6f} "
          f"(mean over samples; 1.0 means code-preserving doping held)")
    print(f"mean D*p = {x.mean():.4f}   ->  linear XEB F = {x.mean() - 1:.4f}")
    print(f"wrote {out_path}  ({len(amps)} magnitudes)")
    print(f"wrote {side}  (provenance)")
    print(f"\nnow: python xeb3d.py view {result_json} --amps {out_path}")
    return amps


# --------------------------------------------------------------------------- #
# de-doping -- build intermediate rungs of a validation ladder
# --------------------------------------------------------------------------- #

def _snap_angle(ang: float) -> float:
    """Nearest multiple of pi/2."""
    return round(ang / (math.pi / 2)) * (math.pi / 2)


def dedope(qasm_path: str, keep: int, out_path: str,
           keep_from: str | None = None, seed: int | None = None) -> dict:
    """Emit a copy of the circuit with only `keep` non-Clifford gates left.

    Every other doped gate has its angle snapped to the nearest multiple of
    pi/2, which makes it Clifford while preserving the gate, its position and
    the entire interaction graph. The result is a circuit of the same family
    and connectivity at lower doping -- the rungs between the tractable
    calibration point and the intractable claim.

    The kept gates are spread evenly through the circuit rather than taken
    from the front, so the doping stays distributed the way the original is.

    This is NOT bit-identical to how the vendor generated their own reduced
    files; they may re-randomise rather than snap. Check before relying on it:
    dedope the full circuit down to the vendor's doping level and compare
    against their file (see `--verify-against`).
    """
    lines = open(qasm_path).read().splitlines(keepends=True)

    # locate every non-Clifford gate: (line index, kind)
    doped = []
    for i, raw in enumerate(lines):
        line = raw.split("//")[0].strip()
        if not line or line.startswith(
                ("OPENQASM", "include", "gate ", "qreg", "creg", "}")):
            continue
        m = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)", line)
        if not m:
            continue
        g = m.group(1).lower()
        if g in ("t", "tdg"):
            doped.append((i, g))
        elif g in _PARAMETRIC:
            ang = re.search(r"\(([^)]*)\)", line)
            if ang and not _angles_are_clifford(ang.group(1)):
                doped.append((i, g))

    total = len(doped)
    if keep > total:
        raise SystemExit(f"circuit has only {total} non-Clifford gates, "
                         f"cannot keep {keep}")

    # map line index -> (gate ordinal, qubit), so a keep-set lifted from
    # another file can be matched positionally rather than by line number
    fp = circuit_fingerprint(qasm_path)
    pos_of_line = {}
    ordinal_iter = iter(fp["doped"])
    for (i, _g) in doped:
        pos_of_line[i] = next(ordinal_iter, None)

    if keep_from:
        anchor = set(circuit_fingerprint(keep_from)["doped"])
        keep_idx = {i for i in pos_of_line if pos_of_line[i] in anchor}
        matched = len(keep_idx)
        if matched != len(anchor):
            print(f"warning: only {matched}/{len(anchor)} of the reference's "
                  f"doped positions exist in this circuit; the two files may "
                  f"not share a skeleton", file=sys.stderr)
        if keep and keep < matched:
            raise SystemExit(
                f"--keep {keep} is below the {matched} positions carried by "
                f"--keep-from; a nested ladder cannot drop reference gates"
            )
        if keep and keep > matched:
            # extend the reference set evenly through the gates it left out,
            # so higher rungs are supersets of the vendor's file
            rest = [i for i in pos_of_line if i not in keep_idx]
            extra = keep - matched
            step = len(rest) / extra if extra else 1
            keep_idx |= {rest[min(len(rest) - 1, int(j * step))]
                         for j in range(extra)}
        keep = len(keep_idx)          # report what was actually kept
    elif seed is not None:
        # independent random draw -- the construction the vendor appears to
        # have used, since their own doping levels share no positions
        import random
        rng = random.Random(seed)
        keep_idx = {doped[i][0]
                    for i in rng.sample(range(total), min(keep, total))}
    elif keep <= 0:
        keep_idx = set()
    else:
        step = total / keep
        keep_idx = {doped[min(total - 1, int(j * step))][0] for j in range(keep)}

    out = list(lines)
    for i, g in doped:
        if i in keep_idx:
            continue
        raw = out[i]
        if g in ("t", "tdg"):
            # T is rz(pi/4); pi/4 is equidistant from 0 and pi/2, so snap up
            # to S/Sdg, preserving the gate rather than deleting it
            out[i] = re.sub(r"\b" + g + r"\b",
                            "s" if g == "t" else "sdg", raw, count=1)
        else:
            ang = re.search(r"\(([^)]*)\)", raw)
            snapped = []
            for tok in ang.group(1).split(","):
                v = _eval_angle(tok)
                snapped.append(repr(_snap_angle(v)) if v is not None else tok)
            out[i] = raw[:ang.start(1)] + ",".join(snapped) + raw[ang.end(1):]

    with open(out_path, "w") as f:
        f.writelines(out)

    got, _ = count_nonclifford(out_path)
    return {"source": qasm_path, "out": out_path, "t_source": total,
            "t_requested": keep, "t_actual": got}


def circuit_fingerprint(qasm_path: str) -> dict:
    """Structural fingerprint: the actual edge multiset and the positions of
    the doped gates, not just their counts.

    Counting two-qubit gates cannot distinguish two different wirings, and is
    useless for validating dedope specifically, since dedope only rewrites
    single-qubit gates and therefore inherits the source's edge count no
    matter what. What has to be compared is which qubits are coupled and where
    the surviving non-Clifford gates sit.
    """
    from collections import Counter

    edges = Counter()
    doped = []
    ordinal = 0

    with open(qasm_path) as f:
        for raw in f:
            line = raw.split("//")[0].strip()
            if not line or line.startswith(
                    ("OPENQASM", "include", "gate ", "qreg", "creg", "}")):
                continue
            m = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)", line)
            if not m:
                continue
            g = m.group(1).lower()
            if g in ("barrier", "measure", "reset"):
                continue
            qs = [int(x) for x in _QARG.findall(line)]
            if not qs:
                continue
            ordinal += 1

            if g in _TWO_QUBIT and len(qs) >= 2:
                for q in qs[1:]:
                    edges[(min(qs[0], q), max(qs[0], q))] += 1
            elif len(qs) == 1:
                nc = (g in ("t", "tdg")) or (
                    g in _PARAMETRIC
                    and (lambda a: a and not _angles_are_clifford(a.group(1)))(
                        re.search(r"\(([^)]*)\)", line)))
                if nc:
                    doped.append((ordinal, qs[0]))

    return {"edges": edges, "doped": doped, "n_gates": ordinal}


def compare_circuits(a_path: str, b_path: str) -> str:
    fa, fb = circuit_fingerprint(a_path), circuit_fingerprint(b_path)

    same_edges = fa["edges"] == fb["edges"]
    same_doping = fa["doped"] == fb["doped"]
    overlap = len(set(fa["doped"]) & set(fb["doped"]))
    n_a, n_b = len(fa["doped"]), len(fb["doped"])

    lines = [
        f"    skeleton: {len(fa['edges'])} vs {len(fb['edges'])} distinct "
        f"coupled pairs, edge multiset "
        f"{'IDENTICAL' if same_edges else 'DIFFERENT'}",
        f"    doped positions: {overlap}/{max(n_a, n_b, 1)} shared "
        f"({'identical' if same_doping else 'differ'})",
    ]

    if same_edges and same_doping:
        lines.append("    -> reconstruction reproduces the reference file")
    elif same_edges:
        lines += [
            "    -> same skeleton, different choice of which gates stay doped.",
            "       A valid ladder over the same topology and doping level,",
            "       but NOT the vendor's circuits -- say so in any writeup.",
            "       Placement varying across rungs arguably tests a certificate",
            "       harder than holding it fixed.",
        ]
    else:
        lines.append("    -> different topology; not the same circuit family")
    return "\n".join(lines)


def dedope_report(qasm_path, keep, out_path, verify_against=None,
                  keep_from=None, seed=None) -> str:
    r = dedope(qasm_path, keep, out_path, keep_from, seed)
    chi = 0.23 * r["t_actual"]
    lines = [
        f"{r['source']}  (t = {r['t_source']})",
        f"  -> {r['out']}  t = {r['t_actual']}  chi ~ 2^{chi:.1f}",
    ]
    if r["t_actual"] != r["t_requested"]:
        lines.append(f"  WARNING: asked for t = {r['t_requested']}, got "
                     f"{r['t_actual']}; snapping may have altered a gate the "
                     f"counter classifies differently")

    if verify_against:
        ref_t, _ = count_nonclifford(verify_against)
        lines += [
            "",
            f"  verify against {verify_against}:",
            f"    t: {r['t_actual']} vs {ref_t}"
            f"  {'match' if r['t_actual'] == ref_t else 'DIFFER'}",
            compare_circuits(out_path, verify_against),
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# circuit structure -- does the T-count factor into independent patches?
# --------------------------------------------------------------------------- #

_TWO_QUBIT = ("cz", "cx", "cy", "ch", "swap", "iswap", "crz", "cp", "cu1",
              "cu3", "cu", "rzz", "rxx", "ryy", "czz", "ecr", "dcx")
_QARG = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*\s*\[\s*(\d+)\s*\]")


def circuit_structure(qasm_path: str) -> dict:
    """Connected components of the two-qubit interaction graph, with the
    non-Clifford count carried by each.

    Stabilizer rank is 2^(0.23t) for t non-Clifford gates *in one entangled
    block*. If the doping localises into components with no two-qubit path
    between them, the state is a tensor product and the cost drops from
    2^(0.23 sum t_i) to the sum of 2^(0.23 t_i) -- which is the difference
    between intractable and routine at high doping. This decides whether that
    route is open, without running anything.
    """
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    per_qubit_t: dict[int, int] = {}
    n_edges = 0
    max_q = -1

    with open(qasm_path) as f:
        for line in f:
            line = line.split("//")[0].strip()
            if not line or line.startswith(
                    ("OPENQASM", "include", "gate ", "qreg", "creg", "}")):
                continue
            m = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)", line)
            if not m:
                continue
            g = m.group(1).lower()
            if g in ("barrier", "measure", "reset"):
                continue

            qs = [int(x) for x in _QARG.findall(line)]
            if not qs:
                continue
            max_q = max(max_q, *qs)
            for q in qs:
                find(q)

            if g in _TWO_QUBIT and len(qs) >= 2:
                n_edges += 1
                for q in qs[1:]:
                    union(qs[0], q)
            elif len(qs) == 1:
                nc = (g in ("t", "tdg")) or (
                    g in _PARAMETRIC
                    and (lambda a: a and not _angles_are_clifford(a.group(1)))(
                        re.search(r"\(([^)]*)\)", line))
                )
                if nc:
                    per_qubit_t[qs[0]] = per_qubit_t.get(qs[0], 0) + 1

    comps: dict[int, list[int]] = {}
    for q in list(parent):
        comps.setdefault(find(q), []).append(q)

    out = []
    for root, qubits in sorted(comps.items(), key=lambda kv: -len(kv[1])):
        t = sum(per_qubit_t.get(q, 0) for q in qubits)
        out.append({"qubits": sorted(qubits), "n_qubits": len(qubits), "t": t})

    return {"n_qubits": max_q + 1, "n_two_qubit_gates": n_edges,
            "components": out,
            "t_total": sum(c["t"] for c in out)}


def structure_report(qasm_path: str) -> str:
    st = circuit_structure(qasm_path)
    comps = st["components"]
    lines = [
        f"{qasm_path}",
        f"  {st['n_qubits']} qubits, {st['n_two_qubit_gates']} two-qubit gates",
        f"  {len(comps)} independent component(s), t_total = {st['t_total']}",
        "",
    ]
    for i, c in enumerate(comps):
        chi = 0.23 * c["t"]
        lines.append(f"  component {i}: {c['n_qubits']:3d} qubits, "
                     f"t = {c['t']:4d}, chi ~ 2^{chi:.1f}")

    mono = 0.23 * st["t_total"]
    if len(comps) == 1:
        lines += [
            "",
            "  Single entangled block: the doping does not factor, so cost is",
            f"  the monolithic 2^{mono:.1f}. Splitting by component is not",
            "  available here -- any decomposition would have to cut through",
            "  entanglement (tensor-network contraction across a time or space",
            "  cut), not just separate components.",
        ]
    else:
        split = sum(2 ** (0.23 * c["t"]) for c in comps)
        lines += [
            "",
            f"  monolithic : 2^{mono:.1f}",
            f"  factored   : 2^{math.log2(split):.1f}  (sum over components)",
            "  The state is a tensor product across components; simulate each",
            "  separately and multiply the per-component probabilities.",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# noisy sampling -- give post-selection something real to filter
# --------------------------------------------------------------------------- #

def _grid_shape(n: int) -> tuple[int, int]:
    """Most square factor pair of n."""
    best = (1, n)
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            best = max(best, (i, n // i), key=lambda t: min(t))
    return best


def ace_grid_advice(n_qubits: int) -> tuple[int, tuple[int, int]]:
    """Pick a padded qubit count that gives ACE a usable 2D grid.

    ACE tiles qubits into a grid and elides couplings between patches. A prime
    qubit count factors as 1 x n -- a line, not a lattice -- and the patch
    structure degenerates, taking long_range_columns/rows with it. 70 + 27 = 97
    is prime, so this matters for exactly the circuit at hand.
    """
    r, c = _grid_shape(n_qubits)
    if min(r, c) > 1:
        return n_qubits, (r, c)
    for pad in range(n_qubits + 1, n_qubits + 12):
        r, c = _grid_shape(pad)
        if min(r, c) >= 4:          # want a real 2D patch structure
            return pad, (r, c)
    return n_qubits, (1, n_qubits)


def run_noisy_postselect(qasm_path, n_data, n_ancilla, shots, noise,
                         out_path, long_range_columns=4, long_range_rows=4,
                         pad_to=None, progress_every=25):
    """Post-selected sampling on QrackAceBackend with a physical noise model.

    The difference from dcs_post_select: QrackStabilizer's only error source is
    T-gate rounding, which the code-preserving doping is constructed to commute
    with -- so acceptance is 100% and post-selection filters nothing. ACE
    injects error on the Clifford gates themselves, which is what the checks
    were built to detect, so acceptance drops below 1 and the filtering becomes
    measurable.

    Writes two files with the same schema: `out_path` holding the accepted
    shots, and `<out>.raw.json` holding every shot regardless of syndrome.
    Running `probs` and `stats` on both gives the fidelity before and after
    post-selection, which is the quantity the syndrome certificate predicts.

    ACE is approximate and has its own elision error. Characterise it first:
    run with --noise 0 and XEB the result against exact amplitudes from
    `probs`. That residual is ACE's floor, and every later number sits on top
    of it.
    """
    import json
    import time

    n_qubits = n_data + n_ancilla
    padded, (rows, cols) = ace_grid_advice(n_qubits)
    if pad_to:
        padded = pad_to
        rows, cols = _grid_shape(padded)

    if padded != n_qubits:
        print(f"note: {n_qubits} qubits factors as {_grid_shape(n_qubits)}; "
              f"padding to {padded} for a {rows}x{cols} ACE grid "
              f"({padded - n_qubits} idle)", file=sys.stderr)
    if min(rows, cols) == 1:
        print("warning: ACE grid is a line, not a lattice; patch elision is "
              "degenerate and results will not mean much", file=sys.stderr)

    try:
        from pyqrack import QrackAceBackend
        from qiskit import QuantumCircuit
    except ImportError as e:
        raise SystemExit(f"`noisy` needs pyqrack and qiskit ({e})")

    qc = QuantumCircuit.from_qasm_file(qasm_path)
    if qc.num_qubits != n_qubits:
        raise SystemExit(f"circuit has {qc.num_qubits} qubits, expected {n_qubits}")

    accepted, raw = [], []
    t0 = time.perf_counter()

    for shot in range(shots):
        # fresh instance per shot: the noise realisation has to be an
        # independent draw, so cloning a post-circuit state is not an option
        # here the way it is for the noiseless reference
        sim = QrackAceBackend(padded,
                              long_range_columns=long_range_columns,
                              long_range_rows=long_range_rows,
                              noise=noise)
        try:
            sim.run_qiskit_circuit(qc, shots=0)
            syndrome = [sim.m(q) for q in range(n_data, n_qubits)]
            data = [sim.m(q) for q in range(n_data)]
            raw.append(data)
            if all(s == 0 for s in syndrome):
                accepted.append(data)
        finally:
            del sim

        if progress_every and (shot + 1) % progress_every == 0:
            el = time.perf_counter() - t0
            print(f"  {shot + 1}/{shots}, {len(accepted)} accepted "
                  f"({100 * len(accepted) / (shot + 1):.1f}%), "
                  f"{el / (shot + 1):.2f}s per shot", file=sys.stderr, flush=True)

    elapsed = time.perf_counter() - t0
    common = {
        "qasm_path": qasm_path,
        "shots": shots,
        "noise": noise,
        "backend": "QrackAceBackend",
        "ace_grid": [rows, cols],
        "padded_qubits": padded,
        "long_range_columns": long_range_columns,
        "long_range_rows": long_range_rows,
        "elapsed_seconds": elapsed,
    }

    with open(out_path, "w") as f:
        json.dump({**common, "accepted": len(accepted),
                   "acceptance_rate": len(accepted) / shots,
                   "accepted_samples": accepted}, f)

    raw_path = os.path.splitext(out_path)[0] + ".raw.json"
    with open(raw_path, "w") as f:
        json.dump({**common, "accepted": len(raw), "acceptance_rate": 1.0,
                   "post_selected": False, "accepted_samples": raw}, f)

    print(f"\n{len(accepted)}/{shots} accepted "
          f"({100 * len(accepted) / shots:.2f}%) at noise={noise}")
    print(f"wrote {out_path} and {raw_path}")
    print("\nnext, on each file:")
    print(f"  python xeb3d.py probs {qasm_path} <file> --out <file>.npy")
    print("  python xeb3d.py stats <file> --amps <file>.npy")
    print("the difference in F is what post-selection bought you")
    return accepted, raw


# --------------------------------------------------------------------------- #
# ladder driver -- sweep circuits, tabulate fidelity before/after post-selection
# --------------------------------------------------------------------------- #

_LADDER_COLUMNS = ("circuit", "t", "chi", "shots", "accepted", "accept_rate",
                   "F_raw", "F_post", "lift", "note")


def _ladder_row(circuit, t, chi, **kw):
    row = {c: "" for c in _LADDER_COLUMNS}
    row.update(circuit=os.path.basename(circuit), t=t, chi=f"2^{chi:.1f}")
    row.update({k: v for k, v in kw.items() if k in row})
    return row


def format_ladder(rows) -> str:
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) if rows else len(c)
              for c in _LADDER_COLUMNS}
    out = ["  ".join(c.ljust(widths[c]) for c in _LADDER_COLUMNS),
           "  ".join("-" * widths[c] for c in _LADDER_COLUMNS)]
    for r in rows:
        out.append("  ".join(str(r[c]).ljust(widths[c])
                             for c in _LADDER_COLUMNS))
    return "\n".join(out)


def run_ladder(patterns, n_data, n_ancilla, shots, noise, outdir,
               dry_run=False, resume=True, chi_limit=45.0, **ace_kw):
    """Sweep a set of circuits: noisy sample -> exact amplitudes -> XEB, both
    before and after post-selection.

    The quantity of interest is the last column. F_raw is the fidelity of the
    noisy sampler against the exact ideal distribution; F_post is the same
    after discarding non-zero-syndrome shots. Their difference is what the
    checks bought, and it is the number a syndrome certificate has to predict.
    Running this across a doping ladder shows whether that prediction holds as
    t climbs toward the regime where no exact reference exists.

    Each circuit is independent: one failure records a note and the sweep
    continues. With resume=True, stages whose outputs already exist are
    skipped, so a long sweep can be restarted without redoing finished work.
    """
    import csv
    import glob as _glob

    circuits = sorted({c for pat in patterns for c in _glob.glob(pat)})
    if not circuits:
        raise SystemExit(f"no circuits matched {patterns}")
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for circuit in circuits:
        stem = os.path.join(outdir, os.path.splitext(os.path.basename(circuit))[0])
        t, _ = count_nonclifford(circuit)
        chi = 0.23 * t

        if chi >= chi_limit:
            rows.append(_ladder_row(circuit, t, chi,
                                    note="skipped: no exact reference possible"))
            continue
        if dry_run:
            rows.append(_ladder_row(circuit, t, chi, shots=shots,
                                    note=f"would write {stem}.json"))
            continue

        try:
            acc_json = f"{stem}.json"
            raw_json = f"{stem}.raw.json"
            if not (resume and os.path.exists(acc_json)
                    and os.path.exists(raw_json)):
                run_noisy_postselect(circuit, n_data, n_ancilla, shots, noise,
                                     acc_json, **ace_kw)

            results = {}
            for tag, path in (("post", acc_json), ("raw", raw_json)):
                amps = f"{stem}.{tag}.npy"
                if not (resume and os.path.exists(amps)):
                    compute_ideal_probs(circuit, path, n_data, n_ancilla, amps)
                data = load_json(path, amps)
                results[tag] = (fidelity_table(data), data)

            f_raw = results["raw"][0]["F_linear"]
            f_post = results["post"][0]["F_linear"]
            meta = results["post"][1].meta or {}
            n_acc = meta.get("accepted", results["post"][1].n_shots)

            rows.append(_ladder_row(
                circuit, t, chi, shots=shots, accepted=n_acc,
                accept_rate=f"{n_acc / shots:.4f}",
                F_raw=f"{f_raw:.4f}", F_post=f"{f_post:.4f}",
                lift=f"{f_post - f_raw:+.4f}"))

        except SystemExit as e:
            rows.append(_ladder_row(circuit, t, chi, note=f"failed: {e}"))
        except Exception as e:  # keep the sweep alive
            rows.append(_ladder_row(circuit, t, chi,
                                    note=f"failed: {type(e).__name__}: {e}"))

        print(format_ladder(rows), file=sys.stderr)
        print(file=sys.stderr)

    csv_path = os.path.join(outdir, "ladder.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_LADDER_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    return rows, csv_path


# --------------------------------------------------------------------------- #

SUBCOMMANDS = ("view", "stats", "tcount", "probs", "noisy", "structure", "dedope", "compare", "ladder")


def _build_data(args) -> XEBData:
    bits = args.input or args.bits
    amps = args.amps

    if bits is None:
        if amps is None:
            amps, bits = autodiscover(args.dir)
        else:
            raise SystemExit("--amps given without a bitstring source")

    data = load_any(bits, amps, getattr(args, "transpose", False))

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
    p.add_argument("--transpose", action="store_true",
                   help="bitstring array is (N, S) rather than (S, N)")


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

    pn = sub.add_parser("noisy", help="post-selected sampling on ACE with noise")
    pn.add_argument("qasm_path")
    pn.add_argument("--n-data", type=int, default=70)
    pn.add_argument("--n-ancilla", type=int, default=27)
    pn.add_argument("--shots", type=int, default=500)
    pn.add_argument("--noise", type=float, default=0.0,
                    help="ACE noise parameter; start at 0 to measure ACE's own "
                         "elision error against exact amplitudes")
    pn.add_argument("--pad-to", type=int, default=None,
                    help="override the padded qubit count for the ACE grid")
    pn.add_argument("--long-range-columns", type=int, default=4)
    pn.add_argument("--long-range-rows", type=int, default=4)
    pn.add_argument("--out", default="dcs_ace_result.json")

    pl = sub.add_parser("ladder", help="sweep circuits and tabulate fidelity")
    pl.add_argument("circuits", nargs="+", help="QASM paths or globs")
    pl.add_argument("--n-data", type=int, default=70)
    pl.add_argument("--n-ancilla", type=int, default=27)
    pl.add_argument("--shots", type=int, default=500)
    pl.add_argument("--noise", type=float, default=0.0)
    pl.add_argument("--outdir", default="ladder_runs")
    pl.add_argument("--chi-limit", type=float, default=45.0,
                    help="skip circuits whose stabilizer rank exceeds 2^this")
    pl.add_argument("--dry-run", action="store_true",
                    help="show the plan and the per-circuit cost, run nothing")
    pl.add_argument("--no-resume", dest="resume", action="store_false",
                    help="recompute stages whose outputs already exist")

    pt = sub.add_parser("tcount", help="count non-Clifford gates in a QASM")
    pt.add_argument("qasm_path")

    pd = sub.add_parser("dedope",
                        help="reduce doping to build intermediate ladder rungs")
    pd.add_argument("qasm_path")
    pd.add_argument("--keep-from", default=None,
                    help="lift the keep-set from this QASM, reproducing its "
                         "doping exactly; with --keep, extend it to a superset")
    pd.add_argument("--keep", type=int, default=0,
                    help="how many non-Clifford gates to leave in place")
    pd.add_argument("--out", required=True)
    pd.add_argument("--seed", type=int, default=None,
                    help="draw the keep-set at random from this seed, matching "
                         "the vendor's apparent construction; use several "
                         "seeds at one t to measure placement sensitivity")
    pd.add_argument("--verify-against", default=None,
                    help="a vendor file at the same doping level, to check "
                         "this reconstruction is faithful")

    pc = sub.add_parser("compare", help="structural comparison of two QASMs")
    pc.add_argument("qasm_a")
    pc.add_argument("qasm_b")

    pst = sub.add_parser("structure",
                         help="component decomposition of the interaction graph")
    pst.add_argument("qasm_path")

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
                    help="concurrent workers; needs clone(). Validate against "
                         "a serial run before trusting a long one")
    pp.add_argument("--allow-circuit-mismatch", action="store_true",
                    help="proceed when the QASM differs from the one recorded "
                         "in the result JSON")

    args = ap.parse_args(argv)

    if args.cmd == "noisy":
        run_noisy_postselect(args.qasm_path, args.n_data, args.n_ancilla,
                             args.shots, args.noise, args.out,
                             args.long_range_columns, args.long_range_rows,
                             args.pad_to)
        return 0

    if args.cmd == "ladder":
        rows, csv_path = run_ladder(args.circuits, args.n_data, args.n_ancilla,
                                    args.shots, args.noise, args.outdir,
                                    args.dry_run, args.resume, args.chi_limit)
        print(format_ladder(rows))
        print(f"\nwrote {csv_path}")
        return 0

    if args.cmd == "tcount":
        print(tcount_report(args.qasm_path))
        return 0

    if args.cmd == "dedope":
        if not args.keep and not args.keep_from:
            raise SystemExit("dedope needs --keep or --keep-from")
        if args.keep_from and args.seed is not None:
            raise SystemExit("--keep-from and --seed are alternative ways to "
                             "choose the keep-set; pick one")
        print(dedope_report(args.qasm_path, args.keep, args.out,
                            args.verify_against, args.keep_from, args.seed))
        return 0

    if args.cmd == "compare":
        print(f"{args.qasm_a}\n{args.qasm_b}")
        print(compare_circuits(args.qasm_a, args.qasm_b))
        return 0

    if args.cmd == "structure":
        print(structure_report(args.qasm_path))
        return 0

    if args.cmd == "probs":
        compute_ideal_probs(args.qasm_path, args.result_json, args.n_data,
                            args.n_ancilla, args.out, args.limit,
                            args.use_clone, args.force, args.workers,
                            args.allow_circuit_mismatch)
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
