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

# --------------------------------------------------------------------------- #
# Qrack shared library location
# --------------------------------------------------------------------------- #
# PyQrack resolves libqrack_pinvoke.so through PYQRACK_SHARED_LIB_PATH, and it
# does so with ctypes at *import* time -- setting the variable after `import
# pyqrack` has no effect. Every pyqrack import in this file is deliberately
# deferred into a function so that this resolution can run first.

QRACK_LIB_CANDIDATES = (
    "/usr/local/lib/qrack/libqrack_pinvoke.so",
    "/usr/local/lib/libqrack_pinvoke.so",
    "/usr/lib/qrack/libqrack_pinvoke.so",
    "/usr/lib/libqrack_pinvoke.so",
    "/usr/lib/x86_64-linux-gnu/libqrack_pinvoke.so",
)


# Qubit-allocation caps, read by Qrack at library load. QUnit compares the
# entangled subsystem width against these; when the circuit exceeds them it
# concludes it must elide, reports "needed to engage ACE" and the fidelity
# guard throws -- even though the near-Clifford layer underneath could hold
# the state in polynomial space. Removing the cap removes the reason to elide.
# -1 means unlimited, which is what the Mitiq CDR near-Clifford tutorial uses.
# QRACK_MAX_CPU_QB caps how many qubits one CPU engine may hold, and it is
# also the width at which Qrack hands work to the GPU. Setting it to -1
# (unlimited) therefore pins everything to the CPU -- which is not what is
# wanted once the real cost is 2^(n+t) dense amplitudes. Left unset by
# default; pass --max-cpu-qb to override.
QRACK_ENV_DEFAULTS = {
    "QRACK_MAX_PAGING_QB": "-1",
}


def loaded_qrack_lib():
    """Which libqrack_pinvoke.so is actually mapped into this process.

    PYQRACK_SHARED_LIB_PATH states an intent; /proc/self/maps states what the
    loader did. They differ when another copy is found first -- a pip-bundled
    one under dist-packages, or an ld.so.cache entry -- and results computed
    against different builds are not comparable even though nothing errors.
    """
    seen = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "libqrack" in line:
                    path = line.rstrip().split(" ", 5)[-1].strip()
                    if path and path not in seen:
                        seen.append(path)
    except OSError:
        return []
    return seen


def verify_qrack_lib(quiet=False):
    """Compare the mapped library against the requested one."""
    want = os.environ.get("PYQRACK_SHARED_LIB_PATH")
    got = loaded_qrack_lib()
    if not got:
        return None
    real_want = os.path.realpath(want) if want else None
    real_got = [os.path.realpath(g) for g in got]

    if real_want and real_want not in real_got:
        parent = os.path.dirname(want)
        print(
            f"\nWARNING: Qrack library mismatch\n"
            f"  requested : {want}\n"
            f"  loaded    : {got[0]}\n"
            f"\n"
            f"  pyqrack ships its own libqrack_pinvoke.so and falls back to it\n"
            f"  when PYQRACK_SHARED_LIB_PATH does not resolve. Some revisions\n"
            f"  expect the DIRECTORY containing the library rather than the\n"
            f"  file itself, in which case a file path is silently ignored.\n"
            f"\n"
            f"  Check which form this build wants:\n"
            f"    grep -n -B3 -A6 PYQRACK_SHARED_LIB_PATH \\\n"
            f"      $(python -c 'import pyqrack.qrack_system.qrack_system as m; "
            f"print(m.__file__)')\n"
            f"\n"
            f"  Then either:\n"
            f"    export PYQRACK_SHARED_LIB_PATH={parent}\n"
            f"  or replace the bundled copy outright:\n"
            f"    mv {os.path.dirname(got[0])}/libqrack_pinvoke.so{{,.bundled}}\n"
            f"    ln -s {want} {os.path.dirname(got[0])}/libqrack_pinvoke.so\n"
            f"\n"
            f"  Until they match, every number produced here came from the\n"
            f"  library named on the 'loaded' line.\n",
            file=sys.stderr)
    elif not quiet:
        print(f"qrack loaded: {got[0]}", file=sys.stderr)
    return got[0]


def configure_qrack_env(overrides=None, quiet=False):
    """Set Qrack's allocation caps before the library loads.

    Not the same as QRACK_DISABLE_QUNIT_FIDELITY_GUARD: that one permits
    elision and silently returns approximate amplitudes, while this removes
    the width limit that makes elision look necessary in the first place.
    Existing values in the environment are respected.
    """
    applied = {}
    for k, v in {**QRACK_ENV_DEFAULTS, **(overrides or {})}.items():
        if os.environ.get(k) is None:
            os.environ[k] = str(v)
        applied[k] = os.environ[k]
    if not quiet:
        print("qrack env: " + "  ".join(f"{k}={v}" for k, v in applied.items()),
              file=sys.stderr)
    return applied


def configure_qrack_lib(path=None, quiet=False):
    """Point PyQrack at a specific libqrack_pinvoke.so.

    Accepts a file or the directory containing it. An explicit path is
    honoured as given; otherwise an already-set PYQRACK_SHARED_LIB_PATH wins,
    and failing that the usual install locations are probed. A locally built
    Qrack (custom FPPOW, ACE tuning, GPU flags) commonly lives outside the
    default prefix, so pinning this matters: silently loading a different
    build than intended changes numerical results without any error.
    """
    if "pyqrack" in sys.modules and not quiet:
        print("warning: pyqrack is already imported; the library path is "
              "fixed and this call has no effect", file=sys.stderr)

    chosen = None
    if path:
        cand = os.path.expanduser(path)
        if os.path.isdir(cand):
            cand = os.path.join(cand, "libqrack_pinvoke.so")
        if not os.path.isfile(cand):
            raise SystemExit(f"no libqrack_pinvoke.so at {cand!r}")
        chosen = cand
        # Revisions differ on whether this variable names the file or the
        # directory holding it. Setting the file path and letting
        # verify_qrack_lib() check what was actually mapped is safer than
        # guessing, because the fallback is silent.
        os.environ.setdefault("PYQRACK_SHARED_LIB_DIR", os.path.dirname(cand))
    elif os.environ.get("PYQRACK_SHARED_LIB_PATH"):
        chosen = os.environ["PYQRACK_SHARED_LIB_PATH"]
    else:
        chosen = next((c for c in QRACK_LIB_CANDIDATES if os.path.isfile(c)),
                      None)

    if chosen:
        os.environ["PYQRACK_SHARED_LIB_PATH"] = chosen
        if not quiet:
            try:
                size = os.path.getsize(chosen) / (1024 * 1024)
                import time as _t
                when = _t.strftime("%Y-%m-%d",
                                   _t.localtime(os.path.getmtime(chosen)))
                print(f"qrack: {chosen}  ({size:.1f} MB, built {when})",
                      file=sys.stderr)
            except OSError:
                print(f"qrack: {chosen}", file=sys.stderr)
    return chosen


def _build_stamp():
    """Short hash of this file, printed at startup.

    Long debugging sessions involve copying the script around; a run against a
    stale copy looks like a fix that did not work, and the only tell is
    output that should have changed and did not. The stamp makes that visible
    instead of inferred.
    """
    try:
        import hashlib
        with open(os.path.abspath(__file__), "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()[:8]
        mt = os.path.getmtime(os.path.abspath(__file__))
        import time as _t
        return f"{h} ({_t.strftime('%H:%M:%S', _t.localtime(mt))})"
    except Exception:
        return "unknown"


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
    for label, path in (("amplitude", amps_path), ("bitstring", bits_path)):
        if not os.path.isfile(path):
            raise SystemExit(f"no {label} file at {path!r}")
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
        if not os.path.isfile(amps_path):
            raise SystemExit(
                f"no amplitude file at {amps_path!r}. `probs` writes it at the "
                f"end of the run, so this is normal if that run is still going "
                f"or was interrupted."
            )
        amps = np.load(amps_path, allow_pickle=False).ravel()
        if amps.size < n_shots:
            # `probs --limit N` computes the FIRST N samples in order, so a
            # short amplitude file pairs with the leading N bitstrings. Confirm
            # against the provenance sidecar rather than assuming: a file that
            # is short for any other reason must not be silently truncated to
            # fit.
            side = os.path.splitext(amps_path)[0] + ".meta.json"
            claimed = None
            if os.path.isfile(side):
                try:
                    claimed = json.load(open(side)).get("n_samples")
                except Exception:
                    claimed = None
            if claimed == amps.size:
                print(f"note: {amps.size} amplitudes for {n_shots} accepted "
                      f"shots; pairing with the first {amps.size} (a "
                      f"--limit {amps.size} run, per the sidecar)",
                      file=sys.stderr)
                bits = bits[:amps.size]
                n_shots = amps.size
            else:
                raise ValueError(
                    f"amplitude count ({amps.size}) is short of accepted shots "
                    f"({n_shots}) and {os.path.basename(side)} does not "
                    f"confirm a --limit run. Refusing to guess which samples "
                    f"these are."
                )
        elif amps.size != n_shots:
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

    body = head + "\n" + _fidelity_block(d, t)

    # The three estimators coincide only under P = F*p + (1-F)/D. Wide
    # disagreement is not noise in one of them -- it says that model does not
    # describe the sampling, and the XEB number should not be read as a
    # fidelity at all.
    vals = [t["F_linear"], t["F_log"], t["F_HOG"]]
    spread = max(vals) - min(vals)
    if spread > 0.3:
        body += (
            f"\n\nESTIMATORS DISAGREE by {spread:.2f}. They coincide only "
            f"when the samples come from\nF*p_ideal + (1-F)/D. This spread "
            f"means that model does not hold, so none\nof these is a "
            f"fidelity."
        )
        if t["HOG"] < 0.2:
            body += (
                f"\nHOG = {t['HOG']:.3f} against 0.5 for an ideal "
                f"distribution: {100 * (1 - t['HOG']):.0f}% of samples sit in\n"
                f"the low-probability tail of the reference. Median D*p = "
                f"{t['median_x']:.4f} against a\nmean of {t['mean_x']:.4f} -- "
                f"the mean is carried by a few outliers while the bulk has\n"
                f"almost no support. The reference distribution is "
                f"concentrated away from the\nsamples rather than being a "
                f"noisy version of the right one."
            )
    return body + "\n\n" + sample_health(d)


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


_COST_MODEL = ["auto"]


def effective_cost(t: int, n_qubits: int, model=None) -> tuple[float, str]:
    """log2 of the cheaper of the two representations, and which one wins.

    Stabilizer rank grows as 2^(0.23t) without bound, but a dense statevector
    is always available at 2^n. Above t = n/0.23 the rank decomposition is the
    *more* expensive representation and quoting it overstates the cost -- a
    hybrid simulator will have fallen back to dense long before. The honest
    ceiling is the minimum.
    """
    model = model or _COST_MODEL[0]

    # "dense" is the empirical model for a build whose near-Clifford path does
    # not engage: measured peak RSS tracks 2^n regardless of t, so planning
    # against 2^(0.23t) produces patches that are cheap on paper and
    # unaffordable in practice. Verify with `probs --limit 2` before trusting
    # either model -- the right one is a property of the build, not the theory.
    if model == "dense":
        return float(n_qubits), "dense statevector (measured behaviour)"

    # Measured on this build: the exact near-Clifford path allocates 2^t
    # terms of 16 bytes, matching observation to the byte (t=35 -> 512 GB).
    # That is the naive t-injection decomposition, not the Bravyi-Gosset
    # 2^(0.23t) stabilizer rank -- and it does not fall back to dense when
    # t > n, so a 12-qubit patch that would fit in 64 KB as a statevector
    # instead asks for half a terabyte. Plan against t, not qubits.
    if model == "tinject":
        return float(t), "exact near-Clifford, 2^t terms (measured)"

    # Measured on this build, matching three independent allocations to the
    # byte: the exact near-Clifford path holds 2^t stabilizer terms, each a
    # full 2^n statevector of 8-byte complex amplitudes. Width and doping both
    # count, additively in the exponent -- which is why a 25-qubit patch with
    # only t=15 still asks for 8 TB while a 4-qubit patch with t=18 runs.
    if model == "nt":
        return float(n_qubits + t), "2^(n+t) terms x amplitudes (measured)"

    rank = 0.23 * t
    if rank < n_qubits:
        return rank, "near-Clifford (stabilizer rank)"
    return float(n_qubits), "dense statevector (rank has saturated)"


def tcount_report(qasm_path: str) -> str:
    t, counts = count_nonclifford(qasm_path)
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:12]
    st = circuit_structure(qasm_path)
    n = st["n_qubits"]

    rank = 0.23 * t
    eff, which = effective_cost(t, n)

    # per-term cost matters as much as the term count: a stabilizer tableau at
    # n qubits is ~n^2/4 bytes, so 2^30 terms is terabytes, not "large"
    term_bytes = (2 * n * (2 * n + 1)) / 8
    total = eff + math.log2(term_bytes if which.startswith("near") else 16)
    for label, exp in (("KB", 10), ("MB", 20), ("GB", 30), ("TB", 40),
                       ("PB", 50), ("EB", 60)):
        if total < exp + 10:
            mem = f"{2 ** (total - exp):.1f} {label}"
            break
    else:
        mem = f"2^{total:.0f} bytes"

    lines = [
        f"{qasm_path}",
        f"  gates: {', '.join(f'{g}={c}' for g, c in top)}",
        f"  {n} qubits, non-Clifford (T-like) count t = {t}",
        f"  stabilizer rank  2^(0.23t) = 2^{rank:.1f}",
        f"  statevector      2^n       = 2^{n}",
        f"  -> cheaper route: {which}, 2^{eff:.1f}",
        f"  -> if materialised: ~{mem}",
    ]

    if eff < 20:
        lines.append("  per-amplitude strong simulation: tractable")
    elif eff < 27:
        lines.append("  per-amplitude strong simulation: feasible, memory-heavy "
                     "-- time one sample before committing")
    else:
        lines.append("  per-amplitude strong simulation: OUT OF REACH")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# ideal probabilities -- the half a weak simulator never computes
# --------------------------------------------------------------------------- #

# Qrack exposes several internal representations and not all support the
# operations the chain rule needs. QrackSimulator defaults to
# isTensorNetwork=True, and QTensorNetwork does not serve arbitrary
# mid-circuit prob() queries -- which is why the reference sampler reaches for
# QrackStabilizer instead. For near-Clifford work the combination below keeps
# the state in stabilizer form while still allowing prob()/force_m().
_SIM_CLASS = ["QrackSimulator"]
_WARNED_FLAGS: set = set()
_LIB_VERIFIED = False
_ACE_MAX_QB = [None]
_PROTECT_CHECKS = [True]
_NDATA = [70]
_NANC = [27]


def warn_if_gpu_disabled(preset):
    """Flag the two independent ways work gets pinned to the CPU."""
    reasons = []
    if SIM_PRESETS.get(preset, {}).get("is_gpu") is False:
        reasons.append(f"preset {preset!r} sets is_gpu=False")
    cap = os.environ.get("QRACK_MAX_CPU_QB")
    if cap in ("-1", "0"):
        reasons.append(f"QRACK_MAX_CPU_QB={cap} (unlimited CPU width, so "
                       f"Qrack never hands off to the GPU)")
    if reasons:
        print("note: GPU will not be used -- " + "; ".join(reasons),
              file=sys.stderr)


def _warn_if_guard_disabled():
    if os.environ.get("QRACK_DISABLE_QUNIT_FIDELITY_GUARD"):
        print("WARNING: QRACK_DISABLE_QUNIT_FIDELITY_GUARD is set. That does "
              "not make simulation exact -- it lets QUnit elide silently and "
              "return APPROXIMATE amplitudes with no error. Unset it for any "
              "run whose output is used as an exact reference.",
              file=sys.stderr)

# PyQrack's constructor takes snake_case keywords (is_stabilizer_hybrid, not
# isStabilizerHybrid). The camelCase spellings are silently ignored by
# **kwargs-free signatures, so a preset written that way is a no-op that looks
# like it worked -- which is exactly what happened here.
SIM_PRESETS = {
    # is_schmidt_decompose is absent from the v2.7 constructor but present in
    # some builds; the kwarg filter drops it silently where it is not.
    # Near-Clifford work is tableau manipulation on the CPU; the GPU path
    # allocates dense subsystem state and offers nothing here. On a stack
    # where QUnit is allocating gigabytes for a t=5 circuit, the accelerator
    # is the thing pulling it toward a dense representation.
    # is_near_clifford_tableau_writer is the flag that decides whether the
    # near-Clifford path engages at all. Measured by exhaustive kwarg search:
    # with it, prob() returns in 0.5s; without it, Qrack times out or throws
    # on the same circuit, whatever the other flags are set to.
    # GPU is ON here. It was off for several revisions on the theory that
    # near-Clifford work is CPU-side tableau manipulation -- but the measured
    # cost law is 2^(n+t) dense complex amplitudes swept per gate, which is
    # exactly what an accelerator is for. Note that QRACK_MAX_CPU_QB=-1 also
    # pins work to the CPU by removing the width at which Qrack hands off, so
    # both have to be right before the GPU sees anything.
    "near-clifford": dict(is_near_clifford_tableau_writer=True,
                          is_stabilizer_hybrid=True,
                          is_schmidt_decompose_multi=False),
    "near-clifford-cpu": dict(is_near_clifford_tableau_writer=True,
                              is_stabilizer_hybrid=True,
                              is_schmidt_decompose_multi=False,
                              is_gpu=False),
    "tableau": dict(is_stabilizer_hybrid=True,
                    is_near_clifford_tableau_writer=True,
                    is_gpu=False),
    "hybrid":        dict(is_stabilizer_hybrid=True),
    "schmidt":       dict(is_stabilizer_hybrid=True,
                          is_schmidt_decompose_multi=True),
    "plain":         dict(is_stabilizer_hybrid=False),
    "default":       dict(),
}

def make_simulator(n_qubits, preset="near-clifford", cls=None):
    """Construct a simulator, passing only kwargs this pyqrack accepts.

    Flag names have come and gone across pyqrack revisions, so the constructor
    signature is inspected rather than assumed; an unsupported flag is dropped
    with a note instead of raising.
    """
    import inspect

    if cls is None:
        import pyqrack
        cls = getattr(pyqrack, _SIM_CLASS[0])

    want = SIM_PRESETS.get(preset, SIM_PRESETS["near-clifford"])
    try:
        accepted = set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):
        accepted = set(want) | {"qubitCount"}

    kw = {k: v for k, v in want.items() if k in accepted}
    dropped = sorted(set(want) - set(kw))
    if dropped and tuple(dropped) not in _WARNED_FLAGS:
        _WARNED_FLAGS.add(tuple(dropped))
        print(f"note: this pyqrack does not accept {dropped}; continuing "
              f"without (preset {preset!r} is weakened)", file=sys.stderr)

    sim = cls(n_qubits, **kw)

    global _LIB_VERIFIED
    if not _LIB_VERIFIED:
        _LIB_VERIFIED = True
        verify_qrack_lib()

    return apply_sim_settings(sim, n_qubits)


_SIM_TUNING = ["full"]


def apply_sim_settings(sim, n_qubits, tuning=None):
    """Post-construction settings that must be reapplied after every clone().

    clone() builds a new simulator from the source's state; it does not carry
    across settings applied through setter methods. Anything configured on a
    base and then cloned per sample is silently lost on the clone -- which is
    how a base that constructs and runs fine produces a clone that throws on
    the first prob().
    """
    # Exact near-Clifford is the default per the API docs, but the whole point
    # of this pipeline is exact amplitudes, so assert it rather than inherit
    # it. The approximate path is faster and would silently produce numbers
    # that are not p_ideal.
    tuning = tuning or _SIM_TUNING[0]
    if tuning == "none":
        return sim

    # set_reactive_separate(False) was added on a hunch and never justified:
    # reactive separation is what lets Qrack factor a state into independent
    # subsystems, so switching it off pushes toward a monolithic
    # representation. It is off the default path now and only applied under
    # tuning="full".
    wanted = [("set_use_exact_near_clifford", True)]
    if tuning == "full":
        wanted.append(("set_reactive_separate", False))

    for meth, arg in wanted:
        fn = getattr(sim, meth, None)
        if fn is not None:
            try:
                fn(arg)
            except Exception:
                pass

    # Raise the ACE cap to the full width. set_ace_max_qb caps the maximum
    # entangled subsystem, replacing wider entangling gates with gate shadows
    # -- i.e. eliding. Below the circuit's entanglement, QUnit reports "needed
    # to engage ACE" and the fidelity guard throws.
    #
    # The advertised remedy, QRACK_DISABLE_QUNIT_FIDELITY_GUARD=1, silences
    # the guard rather than removing the need: the run proceeds WITH elision
    # and returns approximate amplitudes with no error raised. For an exact
    # reference that is the worst outcome, since the error it introduces is
    # exactly the F_elide this pipeline exists to measure. Lift the cap; keep
    # the guard as the tripwire.
    cap = _ACE_MAX_QB[0] if _ACE_MAX_QB[0] is not None else n_qubits
    fn = getattr(sim, "set_ace_max_qb", None) if tuning == "full" else None
    if fn is not None:
        try:
            fn(cap)
        except Exception:
            pass
    return sim


_KWARG_CHILD = r"""
import json, os, resource, sys, time

# Import before touching rlimits. Loading pyqrack pulls in the OpenCL stack,
# which maps a very large virtual address range -- often tens of GB of VA for
# a few MB of real memory. An RLIMIT_AS set afterwards is therefore already
# exceeded, and the next allocation fails instantly: Qiskit's Rust core
# raises pyo3 PanicException, or the allocator aborts with SIGABRT. Neither
# says anything about the simulator.
try:
    import pyqrack
    from qiskit import QuantumCircuit
except BaseException as e:
    print(json.dumps({"ok": False, "err": "import: " + str(e)[:40], "s": 0.0}))
    sys.exit(0)

def va_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return None

cap_gb = float(sys.argv[3])
if cap_gb > 0:
    used = va_gb() or 0.0
    lim = int(max(cap_gb, used + 2.0) * 1024 ** 3)
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS,
                           (lim, hard if hard != resource.RLIM_INFINITY else lim))
    except Exception:
        pass

kw = json.loads(sys.argv[4])

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

t0 = time.perf_counter()
try:
    cls = getattr(pyqrack, sys.argv[5])
    qc = QuantumCircuit.from_qasm_file(sys.argv[1])
    # baseline AFTER the libraries and circuit are loaded: peak RSS is
    # dominated by the interpreter, qiskit and the OpenCL driver, so only the
    # delta across simulation says anything about representation cost
    base_rss = rss_mb()
    n = int(sys.argv[2])
    sim = cls(n, **kw)
    fn = getattr(sim, "set_use_exact_near_clifford", None)
    if fn:
        try:
            fn(True)
        except Exception:
            pass
    sim.run_qiskit_circuit(qc, shots=0)

    # Exercise BOTH access patterns the pipeline uses. prob() alone is cheap
    # and certifies configurations that then collapse under the chain rule:
    # force_m() mutates the state and can drive a near-Clifford
    # representation dense, and prob_perm() is the alternative that never
    # collapses at all. Timing them separately shows which is affordable.
    t_prob = time.perf_counter()
    for q in range(max(0, n - 2), n):
        float(sim.prob(q))
    t_prob = time.perf_counter() - t_prob

    t_perm = -1.0
    if hasattr(sim, "prob_perm"):
        _t = time.perf_counter()
        try:
            float(sim.prob_perm(list(range(min(n, 8))), [False] * min(n, 8)))
            t_perm = time.perf_counter() - _t
        except Exception:
            t_perm = -1.0

    t_force = time.perf_counter()
    for q in range(max(0, n - 4), n):
        float(sim.prob(q))
        sim.force_m(q, False)
    t_force = time.perf_counter() - t_force

    print(json.dumps({"ok": True, "s": time.perf_counter() - t0,
                      "t_prob": t_prob, "t_perm": t_perm,
                      "t_force": t_force,
                      "rss": rss_mb() - base_rss, "base": base_rss,
                      "va": va_gb()}))
except MemoryError:
    print(json.dumps({"ok": False, "err": "MemoryError", "s": time.perf_counter() - t0}))
# PanicException subclasses BaseException, so `except Exception` misses it
except BaseException as e:
    print(json.dumps({"ok": False,
                      "err": type(e).__name__ + ": " + str(e).split(chr(10))[0][:28],
                      "s": time.perf_counter() - t0}))
"""


def kwarg_search(qasm_path, n_qubits, sim_class="QrackSimulator",
                 mem_gb=0, timeout=60, quick=False):
    """Try every combination of the boolean kwargs this build accepts, each in
    an isolated subprocess under a hard memory cap.

    Isolation is not optional here. When Qrack decides on a dense
    representation at this width it requests terabytes, and on failure it
    dereferences the null result rather than raising -- a segfault that takes
    the whole enumeration down with it. RLIMIT_AS makes the allocation fail at
    a survivable size, and the subprocess boundary means a crash costs one row
    instead of the run. Without the cap a combination that nearly fits can
    push the machine into swap.
    """
    import inspect
    import itertools
    import json
    import subprocess
    import tempfile

    # A circuit wider than the simulator drives every gate past the end of the
    # register: Qrack raises on some paths and corrupts memory on others, and
    # the resulting table looks like a finding about flags. Refuse rather than
    # report it.
    if not os.path.isfile(qasm_path):
        raise SystemExit(f"no such circuit: {qasm_path}")
    width = circuit_structure(qasm_path)["n_qubits"]
    if n_qubits < width:
        raise SystemExit(
            f"{os.path.basename(qasm_path)} uses {width} qubits but "
            f"--n-qubits is {n_qubits}. Every gate above index {n_qubits - 1} "
            f"would be out of range, and the failures that produces say "
            f"nothing about the simulator.\n"
            f"To test patch width, point this at a patch circuit:\n"
            f"  xeb3d.py patch <full.qasm> --outdir patches/ --target 27\n"
            f"  xeb3d.py doctor --kwarg-search patches/patch0.qasm"
        )
    if n_qubits > width:
        print(f"note: circuit uses {width} qubits, simulating {n_qubits}; "
              f"the extra {n_qubits - width} stay idle", file=sys.stderr)

    import pyqrack

    cls = getattr(pyqrack, sim_class)
    sig = inspect.signature(cls.__init__).parameters
    knobs = [k for k in ("is_stabilizer_hybrid", "is_schmidt_decompose_multi",
                         "is_gpu", "is_near_clifford_tableau_writer",
                         "is_binary_decision_tree", "is_sparse")
             if k in sig]

    child = os.path.join(tempfile.mkdtemp(), "child.py")
    with open(child, "w") as f:
        f.write(_KWARG_CHILD)

    rows = []
    combos = [c for c in itertools.product([False, True], repeat=len(knobs))
              if not (dict(zip(knobs, c)).get("is_binary_decision_tree")
                      and dict(zip(knobs, c)).get("is_stabilizer_hybrid"))]

    # Most-promising first: stabilizer_hybrid on and gpu off is the
    # near-Clifford configuration, and an early hit means the rest of the grid
    # can be abandoned. --quick keeps only that neighbourhood.
    def rank(c):
        kw = dict(zip(knobs, c))
        return (not kw.get("is_stabilizer_hybrid", False),
                kw.get("is_gpu", False),
                sum(c))

    combos.sort(key=rank)
    if quick:
        combos = combos[:8]

    cap_note = (f"{mem_gb} GB cap" if mem_gb > 0 else
                "no rlimit (the kernel already refuses absurd requests; "
                "subprocess isolation contains the crash)")
    print(f"{len(combos)} combinations, {cap_note}, {timeout}s timeout each.",
          file=sys.stderr)
    if not sys.stdout.isatty():
        print("stdout is piped -- rows stream as they complete, but a pager "
              "or `tail` will hold them until the end. Drop the pipe (or use "
              "`| tee`) to watch progress.", file=sys.stderr)

    on = lambda kw: ",".join(k.replace("is_", "") for k, v in kw.items()
                             if v) or "(all off)"
    header = (f"{'enabled flags':<56} {'result':<26} {'s':>6} {'RSS MB':>8}")
    print(f"kwarg search: {sim_class} at {n_qubits} qubits on "
          f"{os.path.basename(qasm_path)}\n")
    print(header, flush=True)

    for i, combo in enumerate(combos):
        kw = dict(zip(knobs, combo))
        try:
            r = subprocess.run(
                [sys.executable, child, qasm_path, str(n_qubits), str(mem_gb),
                 json.dumps(kw), sim_class],
                capture_output=True, text=True, timeout=timeout,
                env={**os.environ})
            if r.returncode < 0:
                rows.append((kw, f"CRASH (signal {-r.returncode})", 0.0, 0.0))
            else:
                line = [l for l in r.stdout.splitlines() if l.startswith("{")]
                if line:
                    d = json.loads(line[-1])
                    res = "OK" if d.get("ok") else d.get("err", "?")
                    if d.get("ok"):
                        tf, tp = d.get("t_force", 0.0), d.get("t_perm", -1.0)
                        res = (f"OK force_m={tf:.2f}s"
                               + (f" perm={tp:.3f}s" if tp >= 0 else
                                  " perm=n/a"))
                    rows.append((kw, res, d.get("s", 0.0), d.get("rss", 0.0)))
                else:
                    # a silent child is a harness fault, not a result -- say
                    # what the process actually did rather than logging "no
                    # output" for every row
                    tail = (r.stderr.strip().splitlines() or ["(no stderr)"])[-1]
                    rows.append((kw, f"rc={r.returncode}: {tail[:24]}",
                                 0.0, 0.0))
        except subprocess.TimeoutExpired:
            rows.append((kw, f"TIMEOUT (>{timeout}s)", float(timeout), 0.0))
        kw_, res_, el_, rss_ = rows[-1]
        if res_.startswith("CRASH") and kw_.get("is_near_clifford_tableau_writer"):
            res_ = res_ + " [tableau_writer]"
            rows[-1] = (kw_, res_, el_, rss_)
        # stream each row as it lands; a search this slow is useless if the
        # first result only appears after the last one
        print(f"{on(kw_):<56} {res_:<26} {el_:6.1f} {rss_:8.0f}", flush=True)
        if res_.startswith("OK") and rss_ < 512:
            print(f"\n-> cheap success on {on(kw_)}; stopping the sweep",
                  flush=True)
            break

    out = [""]
    ok = [r for r in rows if r[1].startswith("OK")]
    out.append("")
    if ok:
        best = min(ok, key=lambda r: (r[3], r[2]))
        out.append(f"cheapest working configuration: {on(best[0])}")
        out.append("  Compare the force_m and perm timings above: if force_m "
                   "dominates, use --method permutation, which never collapses "
                   "the state.")
        out.append(f"  {best[2]:.1f}s, {best[3]:.0f} MB added over baseline")

        # which single flags separate success from failure?
        okset = [r[0] for r in ok]
        badset = [r[0] for r in rows if not r[1].startswith("OK")]
        decisive = []
        for k in knobs:
            if okset and badset and all(kw.get(k) for kw in okset) and \
                    not any(kw.get(k) for kw in badset):
                decisive.append(f"{k}=True")
            elif okset and badset and not any(kw.get(k) for kw in okset) and \
                    all(kw.get(k) for kw in badset):
                decisive.append(f"{k}=False")
        if decisive:
            out.append(f"  decisive flag(s): {', '.join(decisive)} -- present "
                       f"in every success and absent from every failure; the "
                       f"rest make no difference here.")
    else:
        partial = quick and len(rows) < 40
        harness = [r for r in rows if r[1].startswith("rc=")]
        if partial:
            out.append(f"None of the {len(rows)} configurations tried under "
                       f"--quick worked. That is NOT a conclusion about the "
                       f"build: --quick only covers stabilizer_hybrid with "
                       f"the GPU off. Re-run without --quick for the full "
                       f"grid before drawing one.")
        elif len(harness) == len(rows):
            out.append("Every child failed before reporting -- this is a "
                       "harness fault, not a result about Qrack. The rc/stderr "
                       "column says what happened.")
        else:
            out.append(f"No accepted kwarg combination supports prob() at "
                       f"{n_qubits} qubits.")
            if n_qubits > 40:
                out.append("  Next: try patch width (--n-qubits 31), which is "
                           "what `probs --patches` needs.")
            else:
                out.append("  At this width that is a strong result: even the "
                           "patched route is closed on this build.")
    return "\n".join(out)


def doctor(n_max=97, n_ancilla_frac=0.28, sim_class="QrackSimulator",
           sim_preset="near-clifford"):
    """Report the installed build, then bisect on width to find where prob()
    starts failing.

    The C++ exception carries no detail, so the useful question is not "which
    flag is wrong" but "at what width does this stop working, and does it
    depend on entanglement or on qubit count alone". A hard width limit and a
    circuit-dependent one call for different responses, and guessing at flags
    cannot tell them apart.
    """
    import inspect
    import tempfile

    out = []
    try:
        import pyqrack
        ver = getattr(pyqrack, "__version__", "(no __version__)")
        out.append(f"pyqrack {ver}  from {os.path.dirname(pyqrack.__file__)}")
    except ImportError as e:
        return f"pyqrack not importable: {e}"

    out.append(f"library:  {os.environ.get('PYQRACK_SHARED_LIB_PATH', '(default)')}")
    for k in ("QRACK_MAX_CPU_QB", "QRACK_MAX_PAGING_QB", "QRACK_FPPOW",
              "QRACK_DISABLE_QUNIT_FIDELITY_GUARD", "QRACK_QUNIT_SEPARABILITY_THRESHOLD",
              "QRACK_QBDT_SEPARABILITY_THRESHOLD"):
        v = os.environ.get(k)
        if v is not None:
            out.append(f"env:      {k}={v}")

    cls = getattr(pyqrack, sim_class)
    try:
        sig = inspect.signature(cls.__init__)
        out += ["", f"{sim_class}.__init__ accepts:"]
        for name, par in sig.parameters.items():
            if name in ("self",):
                continue
            out.append(f"  {name} = {par.default!r}")
    except (TypeError, ValueError):
        out.append(f"  (signature unavailable)")

    out += ["", "relevant methods:"]
    for m in ("prob", "prob_perm", "force_m", "clone", "set_ace_max_qb",
              "set_use_exact_near_clifford", "set_sdrp", "set_ncrp",
              "set_t_injection", "set_reactive_separate", "out_probs",
              "get_unitary_fidelity", "separate", "try_separate_tolerance"):
        out.append(f"  {m:<28} {'yes' if hasattr(cls, m) else 'NO'}")

    # width sweep on a circuit of the same shape as the real one: linear
    # nearest-neighbour CZ chain, a few layers, a handful of T gates
    def build(n, t_gates, depth=6):
        L = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];"]
        for _ in range(depth):
            for q in range(n):
                L.append(f"h q[{q}];")
            for q in range(n - 1):
                L.append(f"cz q[{q}],q[{q + 1}];")
        for i in range(t_gates):
            L.append(f"t q[{i % n}];")
        for q in range(n):
            L.append(f"h q[{q}];")
        path = os.path.join(tempfile.mkdtemp(), f"w{n}.qasm")
        open(path, "w").write("\n".join(L) + "\n")
        return path

    from qiskit import QuantumCircuit

    def works(n, t_gates):
        try:
            qc = QuantumCircuit.from_qasm_file(build(n, t_gates))
            sim = make_simulator(n, sim_preset, cls)
            sim.run_qiskit_circuit(qc, shots=0)
            for q in range(min(4, n)):
                float(sim.prob(q))
            del sim
            return True, ""
        except Exception as e:
            return False, str(e).split("\n")[0][:52]

    out += ["", f"width sweep ({sim_class}, preset {sim_preset}):",
            "  n     t=0        t=5        t=75"]
    widths = sorted({8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, n_max})
    last_ok = {}
    for n in widths:
        if n > n_max:
            continue
        cells = []
        for t in (0, 5, 75):
            ok, msg = works(n, t)
            cells.append("ok  " if ok else "FAIL")
            if ok:
                last_ok[t] = n
        out.append(f"  {n:<4}  {cells[0]:<9}  {cells[1]:<9}  {cells[2]}")

    out += ["", "widest working width by doping:"]
    for t in (0, 5, 75):
        out.append(f"  t={t:<3} -> {last_ok.get(t, 'none')}")

    if last_ok.get(0) == last_ok.get(5) == last_ok.get(75) and last_ok.get(0):
        out.append("\n  Limit is independent of doping, so it is a width or "
                   "entanglement ceiling in the layer stack, not near-Clifford "
                   "capacity.")
    elif last_ok.get(0, 0) > last_ok.get(75, 0):
        out.append("\n  Limit falls as doping rises, so it is near-Clifford "
                   "capacity and the ceiling moves with t.")
    return "\n".join(out)


def probe_simulator(qasm_path, n_qubits, n_probe=4):
    """Find a simulator configuration that survives the chain rule.

    The failure this exists for is opaque: prob() raises a generic
    'C++ library raised exception' with no indication of which representation
    refused the call. Trying the configurations directly is faster than
    reading it out of a traceback.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit.from_qasm_file(qasm_path)
    rows = []

    for name in ("near-clifford", "hybrid", "plain", "default"):
        for cls_name in ("QrackSimulator", "QrackStabilizer"):
            try:
                import pyqrack
                cls = getattr(pyqrack, cls_name, None)
                if cls is None:
                    rows.append((name, cls_name, "absent", ""))
                    continue

                sim = make_simulator(n_qubits, name, cls)
                sim.run_qiskit_circuit(qc, shots=0)
                for q in range(min(n_probe, n_qubits)):
                    float(sim.prob(q))
                has_force = hasattr(sim, "force_m")
                if has_force:
                    sim.force_m(0, False)

                # clone() may rebuild with default flags, silently undoing the
                # preset; probs uses it per sample, so test it separately
                clone_note = "clone=n"
                if hasattr(sim, "clone"):
                    try:
                        c = apply_sim_settings(sim.clone(), n_qubits)
                        float(c.prob(min(n_probe, n_qubits) - 1))
                        del c
                        clone_note = "clone=y"
                    except Exception:
                        clone_note = "clone=BROKEN(use --no-clone)"
                del sim
                rows.append((name, cls_name, "OK",
                             f"force_m={'y' if has_force else 'n'} "
                             f"{clone_note}"))
            except Exception as e:
                msg = str(e).split("\n")[0][:60]
                rows.append((name, cls_name, "fail", msg))

    out = [f"library: {os.environ.get('PYQRACK_SHARED_LIB_PATH', '(pyqrack default)')}",
           f"caps:    QRACK_MAX_CPU_QB={os.environ.get('QRACK_MAX_CPU_QB', '(default)')} "
           f"QRACK_MAX_PAGING_QB={os.environ.get('QRACK_MAX_PAGING_QB', '(default)')}",
           "",
           "preset          class             result  notes"]
    for r in rows:
        out.append(f"{r[0]:<15} {r[1]:<17} {r[2]:<7} {r[3]}")

    ok = [r for r in rows if r[2] == "OK" and "force_m=y" in r[3]]
    out.append("")
    if ok:
        best = next((r for r in ok if "clone=y" in r[3]), ok[0])
        flags = f"--sim-preset {best[0]}"
        if "clone=y" not in best[3]:
            flags += " --no-clone"
        out.append(f"use: {flags}")
        if best[1] != "QrackSimulator":
            out.append(f"  (and QrackSimulator failed here -- {best[1]} worked; "
                       f"tell me and I will switch the class)")
    else:
        out.append("no configuration supports prob() + force_m() on this build;"
                   " the chain rule cannot run here")
    return "\n".join(out)


def _peak_rss_mb():
    """Peak resident set size in MB, or None where unavailable.

    Reported in-process so cost measurement does not depend on GNU time,
    which is a separate package from the bash `time` builtin and is missing
    from most slim containers.
    """
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS bytes
        return kb / 1024.0 if sys.platform != "darwin" else kb / (1024.0 ** 2)
    except Exception:
        return None


def _cost_report(n_samples, elapsed, label=""):
    per = elapsed / max(n_samples, 1)
    rss = _peak_rss_mb()
    lines = [
        f"\ncost{' ' + label if label else ''}:",
        f"  {n_samples} samples in {elapsed:.1f}s  ({per:.3f}s per sample)",
    ]
    if rss is not None:
        lines.append(f"  peak RSS {rss:.0f} MB")
    for n in (500, 2000):
        lines.append(f"  -> {n} samples would take {per * n / 60:.1f} min "
                     f"({per * n * 2 / 60:.1f} min for accepted + raw)")
    return "\n".join(lines)


def _statevector(qasm_path, n):
    """Dense reference statevector. Small circuits only -- this exists to give
    the chain rule something independent to be checked against."""
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0

    H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    SDG = S.conj().T
    T = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex)
    TDG = T.conj().T
    ONE = {"h": H, "x": X, "y": Y, "z": Z, "s": S, "sdg": SDG,
           "t": T, "tdg": TDG}

    # Endianness has to be one convention throughout. In reshape([2]*n), axis
    # j carries weight 2^(n-1-j), so using axis q puts qubit q at 2^(n-1-q) --
    # matching apply_cz and the state-index construction in selftest. Using
    # axis (n-1-q) instead silently mirrors single-qubit gates, which is
    # invisible on symmetric circuits and only shows on bitstrings that the
    # mirror does not map to themselves.
    def apply1(psi, u, q):
        psi = psi.reshape([2] * n)
        psi = np.moveaxis(psi, q, 0).reshape(2, -1)
        psi = (u @ psi).reshape([2] * n)
        return np.moveaxis(psi, 0, q).reshape(-1)

    def apply_cz(psi, a, b):
        idx = np.arange(2 ** n)
        mask = ((idx >> (n - 1 - a)) & 1) & ((idx >> (n - 1 - b)) & 1)
        out = psi.copy()
        out[mask == 1] *= -1
        return out

    for raw in open(qasm_path):
        line = raw.split("//")[0].strip().rstrip(";")
        if not line or line.startswith(("OPENQASM", "include", "qreg", "creg")):
            continue
        m = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)", line)
        if not m:
            continue
        g = m.group(1).lower()
        qs = [int(x) for x in _QARG.findall(line)]
        if g in ONE:
            psi = apply1(psi, ONE[g], qs[0])
        elif g == "rz":
            th = _eval_angle(re.search(r"\(([^)]*)\)", line).group(1))
            u = np.array([[np.exp(-1j * th / 2), 0],
                          [0, np.exp(1j * th / 2)]], dtype=complex)
            psi = apply1(psi, u, qs[0])
        elif g == "cz":
            psi = apply_cz(psi, qs[0], qs[1])
        elif g == "cx":
            psi = apply1(psi, H, qs[1])
            psi = apply_cz(psi, qs[0], qs[1])
            psi = apply1(psi, H, qs[1])
        elif g in ("barrier", "measure", "reset"):
            continue
        else:
            raise SystemExit(f"selftest reference cannot handle gate {g!r}")
    return psi


def selftest(n_qubits=6, n_ancilla=2, n_t=3, seed=0, sim_preset="near-clifford",
             sim_class="QrackSimulator", trials=3, tol=1e-6, method="chain"):
    """Check the chain rule against a dense statevector on a small circuit.

    This is the check that matters before trusting any amplitude this tool
    produces. A weak simulator answers prob() from one stochastic rounding of
    the non-Clifford gates, which is a perfectly plausible number that is not
    p_ideal -- and nothing downstream would reveal it. Two things are tested:
    agreement with the exact statevector, and reproducibility across repeated
    runs, since a rounding-based backend gives a different answer each time.
    """
    import random
    import tempfile

    rng = random.Random(seed)
    n_data = n_qubits - n_ancilla
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n_qubits}];"]
    for _ in range(3):
        for q in range(n_qubits):
            lines.append(f"h q[{q}];")
        for q in range(n_qubits - 1):
            lines.append(f"cz q[{q}],q[{q + 1}];")
        for q in range(n_qubits):
            lines.append(f"s q[{q}];")
    # The T gates must be followed by a basis change. T and CZ are both
    # diagonal in Z, so doping placed after the final H layer alters phases
    # only and leaves every measurement probability untouched -- the test then
    # compares against a uniform distribution and passes even if non-Clifford
    # handling is completely broken.
    for _ in range(n_t):
        lines.append(f"t q[{rng.randrange(n_qubits)}];")
    for q in range(n_qubits):
        lines.append(f"h q[{q}];")
    for q in range(n_qubits - 1):
        lines.append(f"cz q[{q}],q[{q + 1}];")
    for _ in range(n_t):
        lines.append(f"t q[{rng.randrange(n_qubits)}];")
    for q in range(n_qubits):
        lines.append(f"h q[{q}];")

    path = os.path.join(tempfile.mkdtemp(), "selftest.qasm")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    psi = _statevector(path, n_qubits)
    probs = np.abs(psi) ** 2

    # exact p(x | ancillas = 0), matching what log_prob_generic computes
    idx = np.arange(2 ** n_qubits)
    anc_zero = np.ones(len(idx), dtype=bool)
    for a in range(n_data, n_qubits):
        anc_zero &= ((idx >> (n_qubits - 1 - a)) & 1) == 0
    p_syn = probs[anc_zero].sum()

    from qiskit import QuantumCircuit
    import pyqrack
    cls = getattr(pyqrack, sim_class)
    qc = QuantumCircuit.from_qasm_file(path)

    def factory():
        sim = make_simulator(n_qubits, sim_preset, cls)
        sim.run_qiskit_circuit(qc, shots=0)
        return sim

    cond = probs[anc_zero] / p_syn if p_syn > 0 else probs[anc_zero]
    uniform = float(cond.max() - cond.min()) < 1e-12

    out = [f"selftest: {n_qubits} qubits ({n_data} data + {n_ancilla} ancilla), "
           f"t={n_t}, class={sim_class}, preset={sim_preset}, "
           f"method={method}",
           f"  P(syndrome=0) exact = {p_syn:.6f}"]
    if uniform:
        out.append("  WARNING: the reference distribution is UNIFORM, so this "
                   "test cannot detect a non-Clifford handling error at all")
    out.append("")

    fn = log_prob_perm if method == "permutation" else log_prob_generic

    # Enumerate the whole data space when it is small enough. Two independent
    # consistency checks come out of it, and between them they localise the
    # fault far better than four sampled bitstrings can:
    #   * does the chain-rule distribution sum to 1? if yes it is internally
    #     consistent and the disagreement is in the reference mapping, not the
    #     simulator
    #   * which bitstrings disagree? a scattered few points at a bad
    #     conditional is a different fault from a uniform offset
    full = n_data <= 12
    chain_all = np.zeros(2 ** n_data) if full else None

    if full:
        for k in range(2 ** n_data):
            bits = [(k >> (n_data - 1 - i)) & 1 for i in range(n_data)]
            lp, _ = fn(factory, range(n_data, n_qubits),
                       [(i, bits[i]) for i in range(n_data)])
            chain_all[k] = math.exp(lp) if lp > -math.inf else 0.0

        # Build the reference under both qubit orderings and report which the
        # backend matches, rather than asserting one. Qiskit and Qrack both
        # index qubit 0 as least significant, but the QASM here is written
        # with explicit indices and the mapping is worth confirming, not
        # assuming -- an endianness error is a permutation of the
        # distribution, so it preserves normalisation and hides from every
        # check except a direct comparison.
        def build_ref(big_endian):
            r = np.zeros(2 ** n_data)
            for k in range(2 ** n_data):
                state = 0
                for i in range(n_data):
                    if (k >> (n_data - 1 - i)) & 1:
                        state |= 1 << ((n_qubits - 1 - i) if big_endian else i)
                r[k] = probs[state] / p_syn if p_syn > 0 else 0.0
            return r

        cands = {"qubit0=MSB": build_ref(True), "qubit0=LSB": build_ref(False)}
        scored = {k: float(np.abs(v - chain_all).max())
                  for k, v in cands.items()}
        best = min(scored, key=scored.get)
        ref_all = cands[best]
        if len(set(np.round(list(scored.values()), 12))) > 1:
            out.append(f"  qubit ordering: {best} "
                       f"(max dev {scored[best]:.2e}; other convention "
                       f"{scored[max(scored, key=scored.get)]:.2e})")

        n_bad = int((np.abs(chain_all - ref_all)
                     > 1e-4 * np.maximum(ref_all, 1e-12)).sum())
        out += [
            f"  chain-rule distribution sums to {chain_all.sum():.6f} "
            f"(exact reference sums to {ref_all.sum():.6f})",
            f"  disagreeing bitstrings: {n_bad}/{2 ** n_data}",
            "",
        ]

    samples = [tuple(rng.randrange(2) for _ in range(n_data)) for _ in range(4)]
    worst = 0.0
    spread = 0.0

    for bits in samples:
        state = 0
        for i, b in enumerate(bits):
            if b:
                state |= 1 << (n_qubits - 1 - i)
        exact = probs[state] / p_syn if p_syn > 0 else 0.0

        got = []
        for _ in range(trials):
            lp, ls = fn(factory, range(n_data, n_qubits),
                        [(i, bits[i]) for i in range(n_data)])
            got.append(math.exp(lp) if lp > -math.inf else 0.0)

        err = abs(got[0] - exact) / max(exact, 1e-12)
        var = (max(got) - min(got)) / max(float(np.mean(got)), 1e-12)
        worst = max(worst, err)
        spread = max(spread, var)
        out.append(f"  x={''.join(map(str, bits))}  exact={exact:.6e}  "
                   f"got={got[0]:.6e}  rel.err={err:.2e}  spread={var:.2e}")

    out.append("")
    ok_exact = worst < 1e-4 and not uniform
    # A single-precision build carries ~1e-6 relative noise; only an O(1)
    # spread indicates stochastic rounding rather than float32.
    stochastic = spread > 1e-3
    ok_stable = spread < 1e-5

    out.append(f"  agreement with statevector: "
               f"{'PASS' if ok_exact else f'FAIL (rel.err {worst:.2e})'}")
    out.append(f"  reproducible across runs:   "
               f"{'PASS' if ok_stable else f'FAIL (spread {spread:.2e})'}"
               + ("  [O(1): stochastic rounding]" if stochastic
                  else "  [~1e-6: float32 precision]" if spread > 1e-9 else ""))
    out.append("")

    if ok_exact and ok_stable:
        out.append("  This configuration produces exact ideal probabilities.")
    elif stochastic:
        out.append("  NOT exact: the backend answers prob() from a stochastic "
                   "rounding of the non-Clifford gates. This is weak "
                   "simulation; amplitudes from it are not p_ideal.")
    elif full and abs(chain_all.sum() - 1.0) < 1e-3:
        out.append("  The chain rule sums to 1, so it is internally consistent "
                   "and the simulator is normalising correctly. The "
                   "disagreement is then in the reference mapping or in "
                   "specific conditionals, NOT in the backend. Re-run with "
                   "--n-t 0: if a pure-Clifford circuit also disagrees, the "
                   "fault is bit ordering.")
    else:
        out.append("  Deterministic but wrong, and the distribution does not "
                   "normalise -- the conditionals themselves are off.")
    return "\n".join(out)


def _sim_fail(op, q, sim, extra=""):
    return RuntimeError(
        f"Qrack raised on {op}(q={q}). The C++ exception carries no detail, "
        f"but the usual causes are:\n"
        f"  * the representation does not serve this call -- QTensorNetwork "
        f"(isTensorNetwork=True, the default) does not answer arbitrary "
        f"mid-circuit prob() queries\n"
        f"  * clone() rebuilt the simulator with default flags, so the preset "
        f"applied to the base never reached this instance -- retry with "
        f"--no-clone\n"
        f"  * the operation needs a subsystem wider than QRACK_MAX_CPU_QB / "
        f"QRACK_MAX_PAGING_QB allow\n"
        f"Run `probe` to test the configurations directly.{extra}"
    )


def _prob(sim, q):
    try:
        return float(sim.prob(q))
    except RuntimeError as e:
        raise _sim_fail("prob", q, sim, f"\n  original: {e}") from e


def _collapse(sim, q, bit):
    """Force qubit q to `bit`, using whichever API this pyqrack exposes."""
    fn = getattr(sim, "force_m", None)
    if fn is not None:
        return fn(q, bool(bit))
    raise RuntimeError(
        "Modern pyqrack with force_m is required for the chain rule. "
        "m() is probabilistic and cannot guarantee the required post-selection path."
    )


def log_prob_generic(sim_factory, ancilla_idx, data_pairs):
    """log p(x | s=0) and log P(s=0) over explicit qubit lists.

    NOTE the conditioning. The ancillas are forced to 0 BEFORE any data qubit
    is read, so every data probability in the chain is already conditioned on
    the zero syndrome and the returned log_p is p(x | s=0), not the joint
    p(x, s=0). Subtracting log_syn from it would divide by P(s=0) twice.
    The joint, if ever needed, is log_p + log_syn.

    Chain rule rather than a state-vector amplitude lookup:

        p(x) = prod_i p(x_i | x_0 .. x_{i-1})

    read off prob(q) and collapsed with force_m(q, x_i), accumulated in log
    space since p is O(2^-70) and underflows float64 around qubit 50.

    Takes explicit indices because patch-local numbering interleaves data and
    ancilla qubits -- the contiguous [0, n_data) convention only holds for the
    unpatched circuit.
    """
    sim = sim_factory()
    try:
        log_p = 0.0
        log_syn = 0.0

        for q in ancilla_idx:
            p0 = max(1.0 - _prob(sim, q), 0.0)
            if p0 <= 0.0:
                return -math.inf, -math.inf
            log_syn += math.log(p0)
            _collapse(sim, q, 0)

        for q, bit in data_pairs:
            p1 = _prob(sim, q)
            pi = p1 if bit else 1.0 - p1
            if pi <= 0.0:
                return -math.inf, log_syn
            log_p += math.log(pi)
            _collapse(sim, q, int(bit))

        return log_p, log_syn
    finally:
        del sim


def log_prob_perm(sim_factory, ancilla_idx, data_pairs):
    """Same quantity as log_prob_generic, via prob_perm() instead of a chain.

    prob_perm(q, c) returns the joint probability that qubits q take truth
    values c, so two calls replace ~n sequential prob()/force_m() pairs:
    p(x, s=0) over everything, and P(s=0) over the ancillas alone. Faster, and
    with no forced-measurement state mutation to get wrong. Returns
    log p(x | s=0) and log P(s=0), matching log_prob_generic.
    """
    sim = sim_factory()
    try:
        anc = list(ancilla_idx)
        qs = anc + [q for q, _ in data_pairs]
        cs = [False] * len(anc) + [bool(b) for _, b in data_pairs]

        p_joint = float(sim.prob_perm(qs, cs))
        p_syn = float(sim.prob_perm(anc, [False] * len(anc))) if anc else 1.0

        if p_syn <= 0.0:
            return -math.inf, -math.inf
        if p_joint <= 0.0:
            return -math.inf, math.log(p_syn)
        return math.log(p_joint) - math.log(p_syn), math.log(p_syn)
    finally:
        del sim


def log_prob_of(sim_factory, bits, n_data, n_ancilla):
    """Unpatched case: data are [0, n_data), ancillas follow.

    Returns log p(x | s=0) and log P(s=0); see log_prob_generic.
    """
    return log_prob_generic(
        sim_factory,
        range(n_data, n_data + n_ancilla),
        [(i, bits[i]) for i in range(n_data)],
    )


def patch_plan(manifest_path):
    """Per-patch qubit maps: (qasm, n_qubits, ancilla_locals, data_locals).

    data_locals is [(original_data_index, local_index)], so a sample indexed by
    original data qubit can be routed into each patch's local numbering.
    """
    import json
    man = json.load(open(manifest_path))

    # Validate every patch file, including manifests written by an earlier
    # revision: a stale or mis-renumbered patch set is indistinguishable from
    # a simulator problem once it reaches Qrack.
    for pch in man["patches"]:
        if not os.path.isfile(pch["file"]):
            raise SystemExit(f"missing patch circuit {pch['file']}")
        declared = None
        refs = []
        for raw in open(pch["file"]):
            line = raw.split("//")[0].strip()
            m2 = re.match(r"qreg\s+\w+\s*\[\s*(\d+)\s*\]", line)
            if m2:
                declared = int(m2.group(1))
                continue          # the size, not a qubit reference
            if line.lstrip().startswith("creg"):
                continue
            refs += [int(x) for x in _QARG.findall(line)]
        if declared is None:
            raise SystemExit(f"{pch['file']}: no qreg declaration")
        if refs and max(refs) >= declared:
            raise SystemExit(
                f"{pch['file']}: declares qreg q[{declared}] but references "
                f"q[{max(refs)}]. Regenerate with `patch`; this file was "
                f"written by a revision with a renumbering bug."
            )
        if declared != pch["n_qubits"]:
            raise SystemExit(
                f"{pch['file']}: declares q[{declared}] but the manifest says "
                f"{pch['n_qubits']} qubits -- manifest and circuits disagree."
            )

    plan = []
    for pch in man["patches"]:
        local = {q: i for i, q in enumerate(pch["qubits"])}
        plan.append({
            "file": pch["file"],
            "n_qubits": pch["n_qubits"],
            "ancilla_locals": [local[q] for q in pch["intact_ancillas"]],
            "dropped_ancillas": [q for q in pch["ancillas"]
                                 if q not in pch["intact_ancillas"]],
            "data_locals": [(q, local[q]) for q in pch["data"]],
        })
    return man, plan


def compute_ideal_probs(qasm_path, result_json, n_data, n_ancilla,
                        out_path, limit=None, use_clone=True, force=False,
                        workers=1, allow_circuit_mismatch=False,
                        patches=None, sim_preset="near-clifford",
                        method="chain"):
    """Strong-simulate p_ideal for each accepted sample and save magnitudes.

    Only tractable while the T-count keeps the state near-Clifford;
    QrackSimulator drifts toward dense representation as doping rises. Run
    `tcount` on the QASM first -- at the reference circuit's 468 T gates this
    will not finish, and that intractability is the advantage claim rather
    than a bug to route around.
    """
    import json
    import time

    # Validate inputs before anything expensive: loading Qrack, applying a
    # circuit and spawning workers all happen before the samples are read, so
    # a missing file surfaces late and after a wait.
    for label, path in (("result JSON", result_json), ("circuit", qasm_path),
                        ("patch manifest", patches)):
        if path and not os.path.isfile(path):
            raise SystemExit(
                f"no {label} at {path!r}.\n"
                + ("The sampler writes this; check it finished and that --out "
                   "matched this path." if label == "result JSON" else "")
            )

    if patches:
        return _patched_ideal_probs(patches, result_json, n_data, out_path,
                                    limit, sim_preset, method, workers)

    # feasibility gate first: a job that cannot finish should not start
    t_gates, _ = count_nonclifford(qasm_path)
    chi, which = effective_cost(t_gates,
                                circuit_structure(qasm_path)["n_qubits"])
    print(f"non-Clifford count t = {t_gates}, cost 2^{chi:.1f} via {which}",
          file=sys.stderr)
    if chi >= 27 and not force:
        raise SystemExit(
            f"\nrefusing to start: at t = {t_gates} the cheapest available\n"
            f"representation is 2^{chi:.1f} ({which}) and per-amplitude strong\n"
            f"simulation will not finish.\n"
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

    print(f"strong-simulating {len(samples)} samples on {n_qubits} qubits "
          f"[{sim_preset}]", file=sys.stderr)

    base = None
    if use_clone:
        t0 = time.perf_counter()
        base = make_simulator(n_qubits, sim_preset)
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
            return apply_sim_settings(base.clone(), n_qubits)
        sim = make_simulator(n_qubits, sim_preset)
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
            worker_bases.put(apply_sim_settings(base.clone(), n_qubits))

        def safe_factory():
            wb = worker_bases.get()
            try:
                return apply_sim_settings(wb.clone(), n_qubits)
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

    # log_ps is already p(x | s=0): the ancillas were forced before any data
    # qubit was read, so the conditioning is baked into the chain. Subtracting
    # log_syn here would apply P(s=0) a second time -- a clean factor of
    # 1/P(s=0) on every amplitude, which selftest catches and nothing else
    # would have.
    log_post = log_ps
    amps = np.exp(0.5 * log_post)      # xeb3d reads magnitudes
    np.save(out_path, amps)

    # provenance sidecar: which circuit and which sample set these belong to
    side = os.path.splitext(out_path)[0] + ".meta.json"
    with open(side, "w") as f:
        json.dump({"qasm_path": qasm_path,
                   "qrack_lib": (loaded_qrack_lib() or [None])[0],
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
    print(_cost_report(len(samples), time.perf_counter() - t0))
    print(f"\nnow: python xeb3d.py view {result_json} --amps {out_path}")
    return amps


# --------------------------------------------------------------------------- #
# patch / elide -- split the circuit where the interaction graph is weakest
# --------------------------------------------------------------------------- #

def _components_without(edges, n_qubits, removed):
    """Connected components of the interaction graph minus `removed` edges."""
    adj = {}
    for (a, b) in edges:
        if (a, b) in removed:
            continue
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    seen, comps = set(), []
    for q in range(n_qubits):
        if q in seen:
            continue
        stack, comp = [q], []
        seen.add(q)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


def ancilla_protected_edges(edges, n_qubits, n_data, n_ancilla):
    """Edges that must not be cut, or a check loses part of its support.

    An ancilla measures a stabilizer over the data qubits it couples to. On a
    tree there is exactly one path from the ancilla to each partner, so cutting
    any edge on those paths detaches part of the support and the syndrome
    becomes a fragment -- measurable, plausible, and meaningless. Marking them
    up front is better than discovering afterwards which check was destroyed.
    """
    adj = {}
    for (a, b) in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    partners = {}
    for (a, b) in edges:
        for x, y in ((a, b), (b, a)):
            if x >= n_data:
                partners.setdefault(x, set()).add(y)

    protected = set()
    for anc in range(n_data, n_data + n_ancilla):
        targets = partners.get(anc, set())
        if not targets:
            continue
        # BFS from the ancilla, recording the parent edge of each node, then
        # walk back from every partner
        parent, seen, stack = {anc: None}, {anc}, [anc]
        while stack:
            u = stack.pop()
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    parent[v] = u
                    stack.append(v)
        for tgt in targets:
            cur = tgt
            while cur is not None and parent.get(cur) is not None:
                p_ = parent[cur]
                protected.add((min(cur, p_), max(cur, p_)))
                if p_ == anc:
                    break
                cur = p_
    return protected


def partition_by_capacity(edges, n_qubits, per_q, cap, weight="t",
                          protected=None):
    """Cut a tree so that every component carries at most `cap` weight.

    Greedy bottom-up: root anywhere, DFS, accumulate each subtree's weight,
    and cut the edge to the parent as soon as a subtree would exceed the cap.
    Whatever is cut off is a finished component under the cap; the parent
    continues with the remainder.

    This is the right algorithm when the constraint is a per-part capacity
    rather than balance. Greedily minimising the largest part stalls the
    moment no single edge improves it -- which happens as soon as the weight
    clusters, and doping does cluster. The capacity walk cannot stall: it
    either meets the cap or names the qubit that makes it impossible.

    Returns (removed_edges, infeasible_qubits).
    """
    adj = {}
    for (a, b) in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    bad = [q for q in range(n_qubits) if per_q.get(q, 0) > cap]
    if bad:
        return set(), bad

    root = next(iter(adj), 0)
    order, parent, seen = [], {root: None}, {root}
    stack = [root]
    while stack:
        u = stack.pop()
        order.append(u)
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                parent[v] = u
                stack.append(v)

    protected = protected or set()
    removed = set()
    over = []
    acc = {q: float(per_q.get(q, 0)) for q in seen}
    for u in reversed(order):          # children before parents
        p = parent[u]
        if p is None:
            continue
        e = (min(u, p), max(u, p))
        if acc[p] + acc[u] > cap and e not in protected:
            removed.add(e)                        # detach the finished subtree
        else:
            # either it fits, or the edge carries a check and must be kept:
            # merging past the cap is the lesser evil against a broken syndrome
            if acc[p] + acc[u] > cap:
                over.append(e)
            acc[p] += acc[u]
    return removed, over


def cut_analysis(qasm_path: str, max_cuts: int = 12, top: int = 5,
                 target: float = 27.0) -> dict:
    """Find edge cuts that split the circuit into cheaper independent patches.

    Patching replaces the joint amplitude with a product over patches:
    p_patch(x) = prod_i p_i(x_i). Cost drops from one 2^(0.23 t) to a sum of
    them. The price is every two-qubit gate crossing a cut, which is simply
    deleted -- so the patched circuit computes the ideal distribution of a
    *different* circuit, and how different is the whole question.

    The graph structure decides whether this is viable. A tree has a bridge at
    every edge, so a single-edge cut splits it; a densely connected graph needs
    many edges removed for the same split and elides far more.
    """
    fp = circuit_fingerprint(qasm_path)
    st = circuit_structure(qasm_path)
    edges = fp["edges"]                       # Counter[(a,b)] -> gate instances
    n = st["n_qubits"]
    per_q = st["per_qubit_t"]

    n_edges = len(edges)
    cycles = n_edges - n + len(st["components"])
    is_tree = cycles == 0

    def score(removed):
        comps = _components_without(edges, n, removed)
        if len(comps) < 2:
            return None
        elided = sum(edges[e] for e in removed)
        patches = []
        for comp in comps:
            t = sum(per_q.get(q, 0) for q in comp)
            cost, which = effective_cost(t, len(comp))
            patches.append({"n_qubits": len(comp), "t": t,
                            "cost": cost, "route": which})
        worst = max(p["cost"] for p in patches)
        return {"removed": sorted(removed), "elided_gates": elided,
                "n_patches": len(comps), "patches": patches,
                "worst_cost": worst,
                "total_cost": math.log2(sum(2 ** p["cost"] for p in patches))}

    def patch_of(comp):
        t = sum(per_q.get(q, 0) for q in comp)
        cost, which = effective_cost(t, len(comp))
        return {"n_qubits": len(comp), "t": t, "cost": cost, "route": which}

    # under tinject the binding constraint is doping per patch, so cuts that
    # split qubits evenly but leave one patch carrying most of the T gates are
    # useless -- the bisection scores on cost, which now tracks t

    # single-edge cuts first; on a tree every edge is a bridge
    singles = [r for r in (score({e}) for e in edges) if r]
    singles.sort(key=lambda r: (r["worst_cost"], r["elided_gates"]))

    # Recursive bisection to a cost target. The naive greedy -- minimise the
    # worst patch over all edges -- happily shaves a single leaf qubit off the
    # largest patch, books that as an improvement and stalls. Always bisect the
    # *most expensive* patch instead, and score candidates by how balanced the
    # resulting halves are.
    chosen: set = set()

    # Under tinject the cost is 2^t per part, so the constraint is a capacity
    # on t and the capacity walk solves it directly. Under the other models
    # cost depends on both t and width, so fall back to bisection.
    if _COST_MODEL[0] in ("tinject", "nt"):
        cap = float(target)
        # under "nt" a part costs 2^(n+t), so each qubit contributes 1 for
        # itself plus its own doping -- the subtree weight is then exactly
        # n_sub + t_sub and the capacity walk bounds the real cost
        w = ({q: 1 + per_q.get(q, 0) for q in range(n)}
             if _COST_MODEL[0] == "nt" else per_q)
        prot = (ancilla_protected_edges(edges, n, _NDATA[0], _NANC[0])
                if _PROTECT_CHECKS[0] else set())
        chosen, over = partition_by_capacity(edges, n, w, cap, protected=prot)
        if prot:
            print(f"  {len(prot)} edges protected (carry a check); "
                  f"{len(over)} places had to exceed the cap to keep a "
                  f"syndrome intact", file=sys.stderr)
        greedy = score(chosen) if chosen else None
        full_t = st["t_total"]
        full_cost, full_route = effective_cost(full_t, n)
        return {"n_qubits": n, "n_edges": n_edges, "cycles": cycles,
                "target": target, "is_tree": is_tree, "t_total": full_t,
                "full_cost": full_cost, "full_route": full_route,
                "gate_instances": sum(edges.values()),
                "singles": singles[:top], "greedy": greedy}

    for _ in range(max_cuts):
        comps = _components_without(edges, n, chosen)
        ranked = sorted(((patch_of(c)["cost"], c) for c in comps),
                        key=lambda kv: -kv[0])
        worst_cost, worst_comp = ranked[0]
        if worst_cost <= target or len(worst_comp) < 2:
            break

        inside = set(worst_comp)
        best_edge, best_score = None, None
        for e in edges:
            if e in chosen or e[0] not in inside or e[1] not in inside:
                continue
            halves = [c for c in _components_without(edges, n, chosen | {e})
                      if set(c) <= inside]
            if len(halves) != 2:
                continue
            # minimise the larger half, tie-break on fewer elided gates
            sc = (max(patch_of(h)["cost"] for h in halves), edges[e])
            if best_score is None or sc < best_score:
                best_edge, best_score = e, sc

        if best_edge is None or best_score[0] >= worst_cost:
            break
        chosen.add(best_edge)

    greedy = score(chosen) if chosen else None

    full_t = st["t_total"]
    full_cost, full_route = effective_cost(full_t, n)

    return {"n_qubits": n, "n_edges": n_edges, "cycles": cycles,
            "target": target,
            "is_tree": is_tree, "t_total": full_t,
            "full_cost": full_cost, "full_route": full_route,
            "gate_instances": sum(edges.values()),
            "singles": singles[:top], "greedy": greedy}


def cut_report(qasm_path: str, max_cuts: int = 12,
               target: float = 27.0) -> str:
    r = cut_analysis(qasm_path, max_cuts, target=target)
    L = [
        f"{qasm_path}",
        f"  {r['n_qubits']} qubits, {r['n_edges']} distinct coupled pairs, "
        f"{r['gate_instances']} two-qubit gate instances",
        f"  independent cycles: {r['cycles']}"
        + ("  -> TREE: every edge is a bridge" if r["is_tree"] else ""),
        f"  unpatched: t = {r['t_total']}, cost 2^{r['full_cost']:.1f} "
        f"({r['full_route']})",
        "",
    ]

    if not r["singles"]:
        L.append("  no single-edge cut splits this circuit")
    else:
        L.append("  best single-edge cuts (by cost of the worst patch):")
        for c in r["singles"]:
            (a, b) = c["removed"][0]
            parts = " + ".join(f"{p['n_qubits']}q/t={p['t']}/2^{p['cost']:.1f}"
                               for p in c["patches"])
            L.append(f"    cut q{a}-q{b}: elides {c['elided_gates']:4d} gates "
                     f"-> {parts}")

    g = r["greedy"]
    if g and len(g["removed"]) >= 1:
        L += ["",
              f"  recursive bisection to a 2^{r['target']:.0f} target: "
              f"{len(g['removed'])} cuts, {g['elided_gates']} gates elided",
              f"    {g['n_patches']} patches, largest 2^{g['worst_cost']:.1f}"]
        for pch in sorted(g["patches"], key=lambda x: -x["cost"]):
            nb = (2 ** pch["cost"]) * 16
            unit = "B"
            for u in ("KB", "MB", "GB", "TB", "PB"):
                if nb > 1024:
                    nb /= 1024
                    unit = u
            L.append(f"      {pch['n_qubits']:3d}q  t={pch['t']:4d}  "
                     f"2^{pch['cost']:.1f}  ({nb:.1f} {unit})")
        if g["worst_cost"] > r["target"]:
            L.append(f"    NOT reached: largest patch is still "
                     f"2^{g['worst_cost']:.1f}")

    best = g or (r["singles"][0] if r["singles"] else None)
    if best:
        saved = r["full_cost"] - best["worst_cost"]
        L += [
            "",
            f"  cost of the largest patch: 2^{best['worst_cost']:.1f} "
            f"(down 2^{saved:.1f} from unpatched)",
            "",
            "  CAVEAT: patched amplitudes are the ideal distribution of a",
            "  DIFFERENT circuit. Scoring samples against them gives",
            "      XEB_measured = F_true * F_elide",
            "  so the number means nothing until F_elide is known. Measure it",
            "  where both routes are computable (the low-t rungs): run `probs`",
            "  exactly and patched on the same circuit, and take the ratio.",
            "  Only if F_elide is stable and predictable across rungs does",
            "  extrapolating it to the hard circuit carry any weight.",
        ]
    return "\n".join(L)


def emit_patches(qasm_path, cuts, outdir, n_data, n_ancilla, prefix="patch"):
    """Write one QASM per patch, plus a manifest tying them back together.

    Gates crossing a cut are deleted; qubits are renumbered densely within each
    patch. The product of the patch distributions approximates the joint one:
        p_patch(x) = prod_i p_i(x restricted to patch i)

    Ancilla integrity is checked, not assumed. A syndrome qubit whose check
    gates straddle a cut measures a fragment of its stabilizer and its outcome
    is meaningless -- post-selecting on it would silently corrupt the
    acceptance condition, which is exactly the quantity under study. Those
    ancillas are reported and must be dropped from the syndrome test.
    """
    import json

    fp = circuit_fingerprint(qasm_path)
    n = circuit_structure(qasm_path)["n_qubits"]
    removed = {tuple(sorted(e)) for e in cuts}

    comps = _components_without(fp["edges"], n, removed)
    comps.sort(key=lambda c: c[0])
    owner = {q: i for i, comp in enumerate(comps) for q in comp}

    # which qubits does each ancilla couple to?
    touches: dict[int, set] = {}
    for (a, b) in fp["edges"]:
        for x, y in ((a, b), (b, a)):
            if x >= n_data:
                touches.setdefault(x, set()).add(y)

    intact, broken = [], []
    for anc in range(n_data, n_data + n_ancilla):
        partners = touches.get(anc, set())
        if not partners:
            broken.append(anc)
        elif all(owner.get(q) == owner.get(anc) for q in partners):
            intact.append(anc)
        else:
            broken.append(anc)

    os.makedirs(outdir, exist_ok=True)
    lines = open(qasm_path).read().splitlines()
    manifest = {"source": qasm_path, "cuts": [list(e) for e in sorted(removed)],
                "n_data": n_data, "n_ancilla": n_ancilla,
                "intact_ancillas": intact, "broken_ancillas": broken,
                "patches": []}

    for i, comp in enumerate(comps):
        local = {q: j for j, q in enumerate(comp)}
        out_lines = [l for l in lines[:2]] + [f"qreg q[{len(comp)}];"]

        for raw in lines:
            line = raw.split("//")[0].strip()
            if not line or line.startswith(
                    ("OPENQASM", "include", "qreg", "creg", "gate ", "}")):
                continue
            m = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)", line)
            if not m or m.group(1).lower() in ("barrier", "measure", "reset"):
                continue
            qs = [int(x) for x in _QARG.findall(line)]
            if not qs or any(owner.get(q) != i for q in qs):
                continue      # outside this patch, or crossing a cut
            out_lines.append(
                _QARG.sub(lambda mm: f"q[{local[int(mm.group(1))]}]", line))

        path = os.path.join(outdir, f"{prefix}{i}.qasm")
        with open(path, "w") as f:
            f.write("\n".join(out_lines) + "\n")

        # A gate referencing an index at or above the declared register makes
        # the simulator size itself to that index instead of the patch width,
        # which shows up as an allocation wildly larger than the patch should
        # need. Check the artifact rather than trusting the renumbering.
        # skip the declaration itself: "qreg q[25];" contains q[25], which is
        # the register SIZE, not a reference to qubit 25
        refs = [int(m2) for line in out_lines
                if not line.lstrip().startswith(("qreg", "creg"))
                for m2 in _QARG.findall(line)]
        if refs and max(refs) >= len(comp):
            raise SystemExit(
                f"{path}: declares qreg q[{len(comp)}] but references "
                f"q[{max(refs)}]. The renumbering is wrong and the emitted "
                f"circuit is invalid."
            )

        t, _ = count_nonclifford(path)
        manifest["patches"].append({
            "file": path, "qubits": comp, "n_qubits": len(comp), "t": t,
            "data": [q for q in comp if q < n_data],
            "ancillas": [q for q in comp if q >= n_data],
            "intact_ancillas": [q for q in comp if q in intact],
        })

    mpath = os.path.join(outdir, f"{prefix}_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest, mpath


def patch_report(qasm_path, outdir, n_data, n_ancilla, target=27.0,
                 max_cuts=12, cuts_from=None) -> str:
    if cuts_from:
        # Reuse a cut set found on another circuit. Every rung of a doping
        # ladder shares the source's interaction graph exactly, so the same
        # edges are cuttable everywhere -- and only by holding the cuts fixed
        # does F_elide across rungs measure how one elision degrades with
        # doping, rather than comparing different elisions at each rung.
        import json
        ref = json.load(open(cuts_from))
        removed = [tuple(e) for e in ref["cuts"]]
        fp = circuit_fingerprint(qasm_path)
        missing = [e for e in removed if tuple(sorted(e)) not in fp["edges"]]
        if missing:
            raise SystemExit(
                f"cuts {missing} are not edges of {qasm_path}; the reference "
                f"manifest describes a different circuit topology"
            )
        elided = sum(fp["edges"][tuple(sorted(e))] for e in removed)
        src = f" (cuts from {os.path.basename(cuts_from)})"
    else:
        r = cut_analysis(qasm_path, max_cuts, target=target)
        g = r["greedy"]
        if not g:
            raise SystemExit(
                f"{qasm_path}\n"
                f"  already below the 2^{target:.0f} target, so no cut is "
                f"needed and no patch files were written. Patching here would "
                f"be a no-op and F_elide would come out trivially 1.\n"
                f"  For a ladder rung, apply the SAME cuts as the hard "
                f"circuit instead of cost-driven ones:\n"
                f"    xeb3d.py patch {os.path.basename(qasm_path)} "
                f"--outdir <dir>/ --cuts-from <refcut>/patch_manifest.json\n"
                f"  (felide.sh derives that manifest from the undoped circuit "
                f"and reuses it for every rung.)"
            )
        removed, elided, src = g["removed"], g["elided_gates"], ""

    man, mpath = emit_patches(qasm_path, removed, outdir, n_data, n_ancilla)

    L = [f"{qasm_path}{src}",
         f"  {len(removed)} cuts, {elided} gates elided",
         ""]
    for pch in man["patches"]:
        cost, _ = effective_cost(pch["t"], pch["n_qubits"])
        L.append(f"  {os.path.basename(pch['file'])}: {pch['n_qubits']:3d}q  "
                 f"t={pch['t']:4d}  2^{cost:.1f}   "
                 f"{len(pch['data'])} data, "
                 f"{len(pch['intact_ancillas'])}/{len(pch['ancillas'])} "
                 f"ancillas intact")

    n_ok, n_bad = len(man["intact_ancillas"]), len(man["broken_ancillas"])
    L += ["", f"  syndrome checks: {n_ok}/{n_ancilla} intact, {n_bad} broken "
              f"by cuts"]
    if n_bad:
        L += [f"    broken: {man['broken_ancillas']}",
              "    Their check gates straddle a cut, so they measure a fragment",
              "    of their stabilizer and the outcome is meaningless. Post-",
              "    select on the intact set only; the acceptance rate is then",
              "    NOT comparable to the unpatched run, which used all "
              f"{n_ancilla}."]
    L += ["", f"  wrote {mpath}"]
    return "\n".join(L)


def budget_report(qasm_path=None, ram_gb=64, hours=24, shots=500,
                  ref_nt=24, ref_s_per_sample=39.0, n_qubits=None,
                  t_gates=None):
    """What n+t is reachable given RAM and a time budget, and what that buys.

    Both memory and runtime scale as 2^(n+t) on this build, so RAM raises the
    ceiling logarithmically while the clock rises exponentially. The binding
    constraint is almost always time: doubling the exponent is a rounding
    error in GB and a factor of two in months.

    Calibrated against a measured point rather than a model -- pass --ref-nt
    and --ref-seconds from a run that completed.
    """
    if qasm_path:
        t_gates, _ = count_nonclifford(qasm_path)
        n_qubits = circuit_structure(qasm_path)["n_qubits"]
    n_qubits = n_qubits or 97
    t_gates = t_gates or 117
    weight = n_qubits + t_gates

    nt_mem = int(math.floor(math.log2(ram_gb * (1024 ** 3) / 8)))
    budget_s = hours * 3600.0
    nt_time = ref_nt
    while (ref_s_per_sample * 2 ** (nt_time + 1 - ref_nt)) * shots <= budget_s:
        nt_time += 1
    nt = min(nt_mem, nt_time)

    parts = max(1, math.ceil(weight / nt))
    per_sample = ref_s_per_sample * 2 ** (nt - ref_nt)

    lines = [
        f"circuit: {n_qubits} qubits, t = {t_gates}  (weight n+t = {weight})",
        f"budget:  {ram_gb} GB, {hours} h, {shots} shots",
        f"calibration: n+t={ref_nt} measured at {ref_s_per_sample:.0f} s/sample",
        "",
        f"  memory allows  n+t <= {nt_mem}",
        f"  time allows    n+t <= {nt_time}",
        f"  binding limit  n+t <= {nt}  ({'time' if nt_time < nt_mem else 'memory'})",
        "",
        f"  -> {parts} patches, {parts - 1} cuts, ~{(parts - 1) * 30} gates elided",
        f"  -> {per_sample:.0f} s per sample, "
        f"{per_sample * shots / 3600:.1f} h for {shots} shots",
    ]

    if nt >= weight:
        lines += ["", "  The whole circuit fits: no cuts needed, and the exact "
                      "monolithic amplitude is available. F_elide is "
                      "measurable here."]
    else:
        # how much RAM would it take to skip cutting entirely?
        need_gb = (2.0 ** weight) * 8 / (1024 ** 3)
        need_s = ref_s_per_sample * 2 ** (weight - ref_nt)
        lines += ["", f"  Running it whole would need "
                      f"2^{weight} x 8 B = {need_gb:.3g} GB and "
                      f"{need_s / 3600:.3g} h per sample.",
                  "  More RAM raises the ceiling by log2(GB); runtime doubles "
                  "per unit of n+t. Time binds first in every realistic case."]
    return "\n".join(lines)


def exact_sample(qasm_path, n_data, n_ancilla, shots, out_path, seed=0):
    """Draw shots from the circuit's EXACT post-selected distribution.

    A weak simulator (QrackStabilizer with stochastic T-gate rounding) does
    not sample p_ideal -- it samples an approximation, and the gap grows with
    doping. That breaks the F_elide measurement at its root: XEB_exact is
    supposed to equal F_true, and Cauchy-Schwarz guarantees
    E_{x~p}[D p(x)] = D sum(p^2) >= 1, so exact XEB can never be negative.
    Observing a negative one proves the samples did not come from p.

    Below ~20 qubits the full statevector is affordable, so the honest move is
    to sample from it directly. Then F_true = 1 by construction, XEB_exact
    measures only the anticoncentration of the circuit, and the ratio
    XEB_patched / XEB_exact isolates F_elide with nothing else mixed in.
    """
    import json

    n = n_data + n_ancilla
    if n > 24:
        raise SystemExit(
            f"{n} qubits: the exact distribution is 2^{n} amplitudes. This "
            f"path is for small circuits; use the weak sampler above ~24."
        )

    psi = _statevector(qasm_path, n)
    probs = np.abs(psi) ** 2

    idx = np.arange(2 ** n)
    keep = np.ones(len(idx), dtype=bool)
    for a in range(n_data, n):
        keep &= ((idx >> (n - 1 - a)) & 1) == 0
    p_syn = float(probs[keep].sum())
    if p_syn <= 0:
        raise SystemExit("zero-syndrome subspace has no support")

    cond = probs[keep] / p_syn
    states = idx[keep]

    rng = np.random.default_rng(seed)
    drawn = rng.choice(len(states), size=shots, p=cond)
    picked = states[drawn]

    samples = [[int((st >> (n - 1 - i)) & 1) for i in range(n_data)]
               for st in picked]

    result = {
        "qasm_path": qasm_path,
        "shots": shots,
        "accepted": shots,
        "acceptance_rate": 1.0,
        "sampler": "exact statevector",
        "P_syndrome_zero": p_syn,
        "seed": seed,
        "accepted_samples": samples,
    }
    with open(out_path, "w") as f:
        json.dump(result, f)

    # what XEB_exact must come out as, for comparison against the real run
    d = 2.0 ** n_data
    predicted = float(d * np.sum(cond ** 2) - 1.0)
    return {"path": out_path, "P_syn": p_syn, "shots": shots,
            "predicted_xeb": predicted,
            "support": int((cond > 1e-12).sum()), "dim": len(states)}


def exact_sample_report(**kw):
    r = exact_sample(**kw)
    lines = [
        f"wrote {r['path']}  ({r['shots']} shots, exact statevector)",
        f"  P(syndrome=0) = {r['P_syn']:.6f}",
        f"  support: {r['support']}/{r['dim']} states carry probability",
        f"  predicted XEB_exact = {r['predicted_xeb']:.4f}",
    ]
    if r["predicted_xeb"] < 0.3:
        lines += ["",
                  "  That is low. XEB assumes an anticoncentrated (Porter-",
                  "  Thomas) ideal distribution, where D*sum(p^2) - 1 is near 1.",
                  "  A near-Clifford circuit concentrates instead, and XEB stops",
                  "  being a meaningful fidelity there. Raise --t-gates or the",
                  "  depth until this approaches 1 before measuring F_elide."]
    else:
        lines += ["",
                  "  Sampled from p_ideal, so F_true = 1 by construction and",
                  "  XEB_patched / XEB_exact is F_elide with nothing else in it."]
    return "\n".join(lines)


def ideal_sample(qasm_path, n_data, n_ancilla, shots, out_path,
                 sim_preset="near-clifford", seed=None):
    """Draw shots from the circuit's EXACT distribution, not a weak simulation.

    Why this matters for F_elide. The ratio XEB_patched / XEB_exact is only
    well conditioned when the denominator is well away from zero, and a weak
    simulator's own T-rounding drives F_true down -- measured at 0.146 +/-
    0.14 for a 13-qubit circuit at t=9, which is one sigma from zero and makes
    the ratio meaningless.

    Sampling from the ideal distribution sets F_true = 1 by construction, so
    XEB_patched IS F_elide and no division is needed. This is only available
    because the circuit is small enough to simulate exactly -- which is the
    whole point of the synthetic instance.

    Note these samples are NOT post-selected: measure_shots reads the data
    register of the ideal state. With code-preserving checks the syndrome
    would be zero anyway; without them, post-selection is not part of what
    F_elide measures.
    """
    import json

    try:
        from pyqrack import QrackSimulator  # noqa: F401
        from qiskit import QuantumCircuit
    except ImportError as e:
        raise SystemExit(f"`idealsample` needs pyqrack and qiskit ({e})")

    n_qubits = n_data + n_ancilla
    qc = QuantumCircuit.from_qasm_file(qasm_path)
    if qc.num_qubits != n_qubits:
        raise SystemExit(f"circuit has {qc.num_qubits} qubits, expected {n_qubits}")

    sim = make_simulator(n_qubits, sim_preset)
    if seed is not None:
        fn = getattr(sim, "seed", None)
        if fn is not None:
            fn(seed)
    sim.run_qiskit_circuit(qc, shots=0)

    fn = getattr(sim, "measure_shots", None)
    if fn is None:
        raise SystemExit("this pyqrack has no measure_shots(); cannot sample "
                         "the ideal distribution directly")
    raw = fn(list(range(n_data)), shots)

    # measure_shots returns packed integers over the requested qubit list, in
    # the order given; bit i of the value is qubit i of that list
    rows = [[(int(v) >> i) & 1 for i in range(n_data)] for v in raw]

    result = {
        "qasm_path": qasm_path,
        "sampler": "ideal (measure_shots on the exact state)",
        "post_selected": False,
        "shots": shots,
        "accepted": len(rows),
        "acceptance_rate": 1.0,
        "accepted_samples": rows,
    }
    with open(out_path, "w") as f:
        json.dump(result, f)

    w = np.asarray(rows).sum(axis=1)
    print(f"wrote {out_path}: {len(rows)} ideal samples over {n_data} qubits")
    print(f"  Hamming weight {w.mean():.2f} +/- {w.std():.2f} "
          f"(binomial {n_data / 2:.1f} +/- {math.sqrt(n_data) / 2:.2f})")
    print(f"  unique bitstrings {len(set(map(tuple, rows)))}/{len(rows)}")
    print("\nF_true = 1 by construction, so a patched XEB against these "
          "samples\nis F_elide directly -- no ratio, no small denominator.")
    return rows


def synth_dcs(n_data, n_ancilla, depth, t_gates, out_path, seed=0,
              check_span=3):
    """Build a small DCS-style circuit: tree topology, interleaved checks.

    The point is a circuit where the monolithic exact amplitude fits, so
    F_elide can be measured by comparing patched against exact on the same
    samples. On this build that means n + t <= ~28 for the WHOLE circuit,
    which the reference family cannot reach at any doping.

    Structure mirrors the reference: data qubits on a line, each ancilla
    coupled to a short span of neighbouring data qubits, doping spread over
    the data register, checks repeated through the depth.
    """
    import random

    rng = random.Random(seed)
    n = n_data + n_ancilla
    L = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];"]

    # ancilla a covers data qubits [start, start+check_span)
    spans = []
    for a in range(n_ancilla):
        start = min(n_data - check_span,
                    int(a * (n_data - check_span) / max(1, n_ancilla - 1))
                    if n_ancilla > 1 else 0)
        spans.append((n_data + a, list(range(start, start + check_span))))

    doped = 0
    for d in range(depth):
        for q in range(n_data):
            L.append(f"h q[{q}];")
        for q in range(n_data - 1):
            L.append(f"cz q[{q}],q[{q + 1}];")

        # doping spread evenly through the layers
        want = round(t_gates * (d + 1) / depth) - doped
        for _ in range(max(0, want)):
            L.append(f"t q[{rng.randrange(n_data)}];")
            doped += 1

        # checks: ancilla picks up the parity of its span, Clifford throughout
        for anc, qs in spans:
            L.append(f"h q[{anc}];")
            for q in qs:
                L.append(f"cz q[{anc}],q[{q}];")
            L.append(f"h q[{anc}];")

    for q in range(n_data):
        L.append(f"h q[{q}];")

    with open(out_path, "w") as f:
        f.write("\n".join(L) + "\n")

    t_actual, _ = count_nonclifford(out_path)
    st = circuit_structure(out_path)
    cost = st["n_qubits"] + t_actual
    mem = (2.0 ** cost) * 8
    unit = "B"
    for u in ("KB", "MB", "GB", "TB"):
        if mem > 1024:
            mem /= 1024
            unit = u
    return {"path": out_path, "n_qubits": st["n_qubits"], "t": t_actual,
            "n_t": cost, "mem": f"{mem:.1f} {unit}",
            "edges": len(circuit_fingerprint(out_path)["edges"]),
            "two_qubit": st["n_two_qubit_gates"]}


def synth_report(**kw):
    r = synth_dcs(**kw)
    lines = [
        f"wrote {r['path']}",
        f"  {r['n_qubits']} qubits, t = {r['t']}, "
        f"{r['two_qubit']} two-qubit gates over {r['edges']} pairs",
        f"  monolithic cost 2^(n+t) = 2^{r['n_t']} = {r['mem']}",
    ]
    # GPU suitability is about shape, not size: cost is 2^t kernels of 2^n
    # amplitudes, so a wide/lightly-doped circuit gives few large kernels
    # (what an accelerator wants) while a narrow/heavily-doped one gives many
    # tiny ones (pure dispatch overhead).
    term = (2.0 ** r["n_qubits"]) * 8
    tb, tu = term, "B"
    for u in ("KB", "MB", "GB"):
        if tb > 1024:
            tb /= 1024
            tu = u
    lines.append(f"  shape: {2 ** r['t']} kernels of {tb:.0f} {tu} "
                 f"(2^t terms x 2^n amplitudes)")
    if r["n_qubits"] >= 20 and r["t"] <= 8:
        lines.append("  wide and lightly doped: the GPU should engage here. "
                     "Time it against --sim-preset near-clifford-cpu.")
    elif r["n_qubits"] < 20:
        lines.append(f"  only {r['n_qubits']} qubits, so the GPU will not "
                     f"engage -- Qrack keeps narrow states on the CPU.")
    else:
        lines.append(f"  t={r['t']} means {2 ** r['t']} kernels; dispatch "
                     f"overhead will dominate. Lower t for a GPU test.")

    if r["n_t"] <= 28:
        lines += ["", "  Both routes are available here: run `probs` exactly "
                      "and patched on the same samples, and their ratio is "
                      "F_elide -- the quantity the reference family cannot "
                      "yield at any doping."]
    else:
        lines += ["", f"  Too large: 2^{r['n_t']} will not fit. Reduce "
                      f"--n-data or --t-gates until n+t <= 28."]
    return "\n".join(lines)


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
            "components": out, "per_qubit_t": per_qubit_t,
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


def _patch_worker(task):
    """Evaluate one patch across a slice of the samples, in its own process.

    Parallelising over patches rather than samples means each worker builds
    its simulator once and reuses it, instead of paying construction on every
    task. Patches are independent circuits by construction, so there is no
    cross-talk to synchronise.

    pyqrack is imported here, never in the parent: a forked child would
    otherwise inherit an already-initialised library, including its RNG state
    and any device handles. The spawn context avoids that, and importing
    inside the worker keeps it true either way.
    """
    (pch, idx, rows, sim_preset, method, lib_path, env) = task

    for k, v in (env or {}).items():
        os.environ.setdefault(k, v)
    if lib_path:
        os.environ["PYQRACK_SHARED_LIB_PATH"] = lib_path

    from qiskit import QuantumCircuit

    qc = QuantumCircuit.from_qasm_file(pch["file"])
    n = pch["n_qubits"]

    base = make_simulator(n, sim_preset)
    base.run_qiskit_circuit(qc, shots=0)
    can_clone = hasattr(base, "clone")

    def factory():
        if can_clone:
            return apply_sim_settings(base.clone(), n)
        sim = make_simulator(n, sim_preset)
        sim.run_qiskit_circuit(qc, shots=0)
        return sim

    fn = log_prob_perm if method == "permutation" else log_prob_generic
    out = []
    for pairs in rows:
        lp, ls = fn(factory, pch["ancilla_locals"], pairs)
        out.append((lp, ls))
    return pch["file"], idx, out


def _patched_ideal_probs(manifest_path, result_json, n_data, out_path,
                         limit=None, sim_preset="near-clifford",
                         method="chain", workers=1):
    """Amplitudes as a product over patches: p(x) = prod_i p_i(x_i).

    These are the ideal probabilities of the *patched* circuit, not the real
    one. Scoring samples against them gives F_true * F_elide, so the output is
    only interpretable once F_elide has been measured on a circuit where the
    exact route is also computable.
    """
    import json
    import time

    man, plan = patch_plan(manifest_path)

    with open(result_json) as f:
        blob = json.load(f)
    key = next((k for k in _BIT_KEYS if k in blob), None)
    samples = np.asarray(blob[key], dtype=np.uint8)
    if limit:
        samples = samples[:limit]

    try:
        from pyqrack import QrackSimulator
        from qiskit import QuantumCircuit
    except ImportError as e:
        raise SystemExit(f"patched `probs` needs pyqrack and qiskit ({e})")

    # Echo the manifest before simulating. Inferring patch contents from
    # crash sizes has been unreliable; the run should state what it is about
    # to do, in the same output as the failure.
    print(f"\nmanifest {manifest_path}", file=sys.stderr)
    print(f"  source {os.path.basename(man.get('source', '?'))}, "
          f"{len(man.get('cuts', []))} cuts", file=sys.stderr)
    print(f"  {'patch':<16} {'qubits':>6} {'t':>5} {'2^t x16B':>12} "
          f"{'data':>5} {'anc':>4}", file=sys.stderr)
    worst = 0
    for pch, meta in zip(plan, man["patches"]):
        t = meta.get("t", 0)
        worst = max(worst, t)
        nb = (2.0 ** t) * 16
        unit = "B"
        for u in ("KB", "MB", "GB", "TB", "PB"):
            if nb > 1024:
                nb /= 1024
                unit = u
        print(f"  {os.path.basename(meta['file']):<16} {meta['n_qubits']:>6} "
              f"{t:>5} {nb:>9.1f} {unit:<2} "
              f"{len(meta['data']):>5} {len(meta['ancillas']):>4}",
              file=sys.stderr)
    print(f"  max t = {worst}", file=sys.stderr)
    widest = max((m["n_qubits"] for m in man["patches"]), default=0)
    if widest < 20:
        print(f"  note: widest patch is {widest} qubits, so the GPU will not "
              f"engage -- Qrack keeps narrow states on the CPU, and the "
              f"2^(n+t) cost here is 2^t small operations rather than one wide "
              f"statevector. Parallelism across patches and samples is the "
              f"lever; the accelerator is not.", file=sys.stderr)
    if worst > 20:
        print(f"  WARNING: a patch with t={worst} needs 2^{worst} terms if "
              f"Qrack takes the exact near-Clifford path. Re-cut with "
              f"--cost-model tinject --target 18.", file=sys.stderr)
    print(file=sys.stderr)

    covered = sorted(q for pch in plan for q, _ in pch["data_locals"])
    if covered != list(range(n_data)):
        raise SystemExit(
            f"patches cover {len(covered)} data qubits, expected {n_data}; "
            "the manifest does not match this sample set"
        )
    dropped = [q for pch in plan for q in pch["dropped_ancillas"]]
    if dropped:
        print(f"warning: {len(dropped)} ancillas broken by cuts and excluded "
              f"from the syndrome ({dropped}); the acceptance condition is "
              f"weaker than the unpatched run's", file=sys.stderr)

    bases = []
    for pch in (plan if workers <= 1 or len(plan) <= 1 else []):
        sim = make_simulator(pch["n_qubits"], sim_preset)
        sim.run_qiskit_circuit(
            QuantumCircuit.from_qasm_file(pch["file"]), shots=0)
        bases.append(sim if hasattr(sim, "clone") else None)
        print(f"  {os.path.basename(pch['file'])}: {pch['n_qubits']}q applied",
              file=sys.stderr)

    prob_fn = log_prob_perm if method == "permutation" else log_prob_generic

    log_ps = np.zeros(len(samples))
    log_syn = np.zeros(len(samples))
    t0 = time.perf_counter()

    if workers > 1 and len(plan) > 1:
        import multiprocessing as mp

        # One task per patch leaves the pool idle behind the slowest patch:
        # cost varies as 2^(n+t), so the largest can run 30x the smallest and
        # every other core waits on it. Split each patch's samples into chunks
        # instead, with more chunks where the patch is expensive, so the tail
        # is one chunk rather than one whole patch.
        costs = [2.0 ** (meta["n_qubits"] + meta.get("t", 0))
                 for meta in man["patches"]]
        total = sum(costs) or 1.0

        tasks = []
        for pch, meta, cost in zip(plan, man["patches"], costs):
            share = max(1, int(round(workers * cost / total)))
            n_chunks = max(1, min(share, len(samples)))
            size = math.ceil(len(samples) / n_chunks)
            for start in range(0, len(samples), size):
                sl = list(range(start, min(start + size, len(samples))))
                rows = [[(loc, int(samples[k][orig]))
                         for orig, loc in pch["data_locals"]] for k in sl]
                tasks.append((pch, sl, rows, sim_preset, method,
                              os.environ.get("PYQRACK_SHARED_LIB_PATH"),
                              {k: os.environ[k] for k in
                               ("QRACK_MAX_CPU_QB", "QRACK_MAX_PAGING_QB")
                               if k in os.environ}))

        # heaviest first: a long task started last sets the wall time
        tasks.sort(key=lambda tk: -(2.0 ** (tk[0]["n_qubits"])))
        n_w = min(workers, len(tasks))
        print(f"  {len(tasks)} tasks over {n_w} workers "
              f"({len(plan)} patches x sample chunks)", file=sys.stderr)

        ctx = mp.get_context("spawn")
        done_n = 0
        with ctx.Pool(processes=n_w) as pool:
            for name, idx, res in pool.imap_unordered(_patch_worker, tasks):
                for k, (lp, ls) in zip(idx, res):
                    log_ps[k] += lp
                    log_syn[k] += ls
                done_n += 1
                print(f"  {done_n}/{len(tasks)} {os.path.basename(name)}"
                      f"[{idx[0]}:{idx[-1] + 1}] "
                      f"({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
    else:
        for k, row in enumerate(samples):
            for pch, base in zip(plan, bases):
                def factory(_p=pch, _b=base):
                    if _b is not None:
                        return apply_sim_settings(_b.clone(), _p["n_qubits"])
                    sim = make_simulator(_p["n_qubits"], sim_preset)
                    sim.run_qiskit_circuit(
                        QuantumCircuit.from_qasm_file(_p["file"]), shots=0)
                    return sim

                lp, ls = prob_fn(
                    factory, _p_anc(pch),
                    [(loc, row[orig]) for orig, loc in pch["data_locals"]])
                log_ps[k] += lp
                log_syn[k] += ls

            if k % 10 == 9 or k == len(samples) - 1:
                el = time.perf_counter() - t0
                print(f"  {k + 1}/{len(samples)}  {el / (k + 1):.2f}s per sample",
                      file=sys.stderr, flush=True)

    log_post = log_ps          # already conditioned; see log_prob_generic
    amps = np.exp(0.5 * log_post)
    np.save(out_path, amps)

    n_zero = int((log_post == -math.inf).sum())
    if n_zero:
        frac = n_zero / len(log_post)
        print(f"\n{n_zero}/{len(log_post)} samples ({frac:.0%}) have EXACTLY "
              f"zero probability under the patched circuit.", file=sys.stderr)
        if frac > 0.9:
            tt = [p_["t"] for p_ in man["patches"]]
            print(
                "\nThis is a property of the method, not a failure of the run.\n"
                f"The patches carry t = {tt}. Eliding the cross-cut gates left\n"
                "them at or near zero doping, and a stabilizer state has support\n"
                "on a coset of 2^k basis states with exact zeros everywhere else.\n"
                "Samples drawn from the full circuit fall outside that support\n"
                "almost surely, so p_patched(x) = 0 and XEB collapses to -1.\n"
                "\n"
                "Patching therefore carries no information at low doping: the\n"
                "patched circuit is MORE Clifford than the original, and the\n"
                "gap widens as t falls. Rungs with enough doping left in each\n"
                "patch after the cut are the only ones where F_elide is\n"
                "measurable -- at t=468 the patches carry t = 79..145, which is\n"
                "a different regime entirely.",
                file=sys.stderr)

    side = os.path.splitext(out_path)[0] + ".meta.json"
    with open(side, "w") as f:
        json.dump({"patched": True, "method": method,
                   "qrack_lib": (loaded_qrack_lib() or [None])[0],
                   "zero_probability_samples": n_zero,
                   "patch_t": [p_["t"] for p_ in man["patches"]],
                   "manifest": manifest_path,
                   "source": man["source"], "cuts": man["cuts"],
                   "result_json": result_json, "n_samples": int(len(amps)),
                   "broken_ancillas": man["broken_ancillas"]}, f, indent=2)

    x = np.exp(log_post + n_data * math.log(2.0))
    print(f"\nmean D*p = {x.mean():.4f}  ->  patched XEB = {x.mean() - 1:.4f}")
    print("this is F_true * F_elide, NOT a fidelity; divide by F_elide "
          "measured on a rung where the exact route also runs")
    print(f"wrote {out_path} and {side}")
    print(_cost_report(len(samples), time.perf_counter() - t0, "(patched)"))
    if (time.perf_counter() - t0) / max(len(samples), 1) > 10:
        if method == "chain":
            print("\nThat is slow for these widths. The chain rule spends most "
                  "of its time in force_m, which can expand the representation "
                  "on every collapse; prob_perm asks for the joint probability "
                  "in one call and never forces a measurement. Try "
                  "--method permutation.", file=sys.stderr)
        else:
            rss = _peak_rss_mb() or 0.0
            if rss > 8000:
                print("\nSlow, and peak RSS is in the tens of GB for circuits "
                      "this narrow -- Qrack is holding dense state.",
                      file=sys.stderr)
            else:
                print(f"\nSlow, but peak RSS is only {rss:.0f} MB, so this is "
                      f"compute-bound rather than memory-bound. Runtime scales "
                      f"as 2^(n+t) like the memory does; the lever is a lower "
                      f"--target (more, smaller patches), not more RAM.",
                      file=sys.stderr)
    return amps


def _p_anc(pch):
    return pch["ancilla_locals"]


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
        chi, _which = effective_cost(t, circuit_structure(circuit)["n_qubits"])

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

SUBCOMMANDS = ("view", "stats", "tcount", "probs", "noisy", "structure",
               "dedope", "compare", "ladder", "cut", "patch", "synth",
               "budget", "sample", "idealsample", "probe", "selftest",
               "doctor",
               "env")


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

    # Strip --qrack-lib first, wherever it appears. It has to come out before
    # the bare-path heuristic below, which sees a leading "-" and would insert
    # the default subcommand ahead of the flag.
    _qrack_lib_arg = None
    for i, tok in enumerate(list(argv)):
        if tok == "--qrack-lib" and i + 1 < len(argv):
            _qrack_lib_arg = argv[i + 1]
            del argv[i:i + 2]
            break
        if tok.startswith("--qrack-lib="):
            _qrack_lib_arg = tok.split("=", 1)[1]
            del argv[i]
            break

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
    ap.add_argument("--qrack-lib", default=None,
                    help="path to libqrack_pinvoke.so (or its directory); "
                         "sets PYQRACK_SHARED_LIB_PATH before pyqrack loads")
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
    pl.add_argument("--chi-limit", type=float, default=27.0,
                    help="skip circuits whose cheapest representation exceeds "
                         "2^this; 2^27 is already tens of GB at ~100 qubits")
    pl.add_argument("--dry-run", action="store_true",
                    help="show the plan and the per-circuit cost, run nothing")
    pl.add_argument("--no-resume", dest="resume", action="store_false",
                    help="recompute stages whose outputs already exist")

    pse = sub.add_parser("selftest",
                         help="validate the chain rule against a statevector")
    pse.add_argument("--n-qubits", type=int, default=6)
    pse.add_argument("--n-ancilla", type=int, default=2)
    pse.add_argument("--n-t", type=int, default=3)
    pse.add_argument("--seed", type=int, default=0)
    pse.add_argument("--sim-class", default="QrackSimulator",
                     choices=("QrackSimulator", "QrackStabilizer"))
    pse.add_argument("--max-cpu-qb", default=None,
                    help="QRACK_MAX_CPU_QB; -1 (default) lifts the per-engine "
                         "qubit cap so QUnit has no reason to elide")
    pse.add_argument("--ace-max-qb", type=int, default=None,
                    help="max entangled subsystem before ACE elision; "
                         "defaults to the full circuit width so no elision "
                         "occurs")
    pse.add_argument("--sim-preset", default="near-clifford",
                     choices=sorted(SIM_PRESETS))
    pse.add_argument("--method", default="chain",
                     choices=("chain", "permutation"))

    penv = sub.add_parser("env",
                          help="print the exports that pin other scripts "
                               "(e.g. the sampler) to this Qrack build")

    pdoc = sub.add_parser("doctor",
                          help="report the installed build and find the width "
                               "at which prob() starts failing")
    pdoc.add_argument("--n-max", type=int, default=97)
    pdoc.add_argument("--kwarg-search", metavar="QASM", default=None,
                      help="enumerate every accepted boolean kwarg on this "
                           "circuit at full width")
    pdoc.add_argument("--n-qubits", type=int, default=None,
                      help="simulator width; defaults to the circuit's own")
    pdoc.add_argument("--mem-gb", type=float, default=0,
                      help="optional RLIMIT_AS per attempt, in GB. Off by "
                           "default: the OpenCL stack maps tens of GB of "
                           "virtual address space, so a low cap kills the "
                           "child before it simulates anything. Subprocess "
                           "isolation is what contains a crash.")
    pdoc.add_argument("--timeout", type=int, default=60)
    pdoc.add_argument("--quick", action="store_true",
                      help="only the eight most promising combinations")
    pdoc.add_argument("--sim-class", default="QrackSimulator",
                      choices=("QrackSimulator", "QrackStabilizer"))
    pdoc.add_argument("--sim-preset", default="near-clifford",
                      choices=sorted(SIM_PRESETS))

    pr = sub.add_parser("probe",
                        help="find a Qrack config that supports the chain rule")
    pr.add_argument("qasm_path")
    pr.add_argument("--n-data", type=int, default=70)
    pr.add_argument("--n-ancilla", type=int, default=27)
    pr.add_argument("--max-cpu-qb", default=None,
                    help="QRACK_MAX_CPU_QB; -1 (default) lifts the per-engine "
                         "qubit cap so QUnit has no reason to elide")
    pr.add_argument("--ace-max-qb", type=int, default=None)

    pb = sub.add_parser("budget",
                        help="what n+t is reachable for a given RAM and time "
                             "budget, and how many cuts that implies")
    pb.add_argument("qasm_path", nargs="?")
    pb.add_argument("--ram-gb", type=float, default=64)
    pb.add_argument("--hours", type=float, default=24)
    pb.add_argument("--shots", type=int, default=500)
    pb.add_argument("--ref-nt", type=int, default=24)
    pb.add_argument("--ref-seconds", type=float, default=39.0)

    pes = sub.add_parser("sample",
                         help="draw shots from the exact post-selected "
                              "distribution via a numpy statevector, and "
                              "report the XEB it must produce")
    pes.add_argument("qasm_path")
    pes.add_argument("--n-data", type=int, default=13)
    pes.add_argument("--n-ancilla", type=int, default=0)
    pes.add_argument("--shots", type=int, default=200)
    pes.add_argument("--seed", type=int, default=0)
    pes.add_argument("--out", required=True)

    pis = sub.add_parser("idealsample",
                         help="sample the exact distribution, so F_true = 1")
    pis.add_argument("qasm_path")
    pis.add_argument("--n-data", type=int, default=13)
    pis.add_argument("--n-ancilla", type=int, default=0)
    pis.add_argument("--shots", type=int, default=200)
    pis.add_argument("--seed", type=int, default=None)
    pis.add_argument("--sim-preset", default="near-clifford",
                     choices=sorted(SIM_PRESETS))
    pis.add_argument("--out", required=True)

    psy = sub.add_parser("synth",
                         help="build a small DCS-style circuit where the "
                              "monolithic amplitude is computable")
    psy.add_argument("--n-data", type=int, default=12)
    psy.add_argument("--n-ancilla", type=int, default=4)
    psy.add_argument("--depth", type=int, default=8)
    psy.add_argument("--t-gates", type=int, default=12)
    psy.add_argument("--check-span", type=int, default=3)
    psy.add_argument("--seed", type=int, default=0)
    psy.add_argument("--out", required=True)

    pt = sub.add_parser("tcount", help="count non-Clifford gates in a QASM")
    pt.add_argument("qasm_path")
    pt.add_argument("--cost-model", choices=("auto", "dense", "tinject", "nt"), default="auto",
                     help="cost law to plan against. 'auto' = 2^(0.23t) "
                          "stabilizer rank; 'dense' = 2^n; 'tinject' = 2^t, "
                          "which is what this Qrack build actually allocates "
                          "for exact near-Clifford. Confirm with `probs "
                          "--limit 2` before trusting any of them.")

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

    pcut = sub.add_parser("cut", help="find patch cuts of the interaction graph")
    pcut.add_argument("qasm_path")
    pcut.add_argument("--allow-broken-checks", action="store_true",
                     help="permit cuts through a check's support; by default "
                          "such edges are protected even at the cost of a "
                          "larger patch")
    pcut.add_argument("--cost-model", choices=("auto", "dense", "tinject", "nt"), default="auto",
                     help="cost law to plan against. 'auto' = 2^(0.23t) "
                          "stabilizer rank; 'dense' = 2^n; 'tinject' = 2^t, "
                          "which is what this Qrack build actually allocates "
                          "for exact near-Clifford. Confirm with `probs "
                          "--limit 2` before trusting any of them.")
    pcut.add_argument("--max-cuts", type=int, default=12)
    pcut.add_argument("--n-data", type=int, default=70)
    pcut.add_argument("--n-ancilla", type=int, default=27)
    pcut.add_argument("--target", type=float, default=27.0,
                      help="bisect until every patch is at or below 2^this")

    ppa = sub.add_parser("patch", help="emit patched circuits from the cuts")
    ppa.add_argument("qasm_path")
    ppa.add_argument("--allow-broken-checks", action="store_true",
                     help="permit cuts through a check's support; by default "
                          "such edges are protected even at the cost of a "
                          "larger patch")
    ppa.add_argument("--cost-model", choices=("auto", "dense", "tinject", "nt"), default="auto",
                     help="cost law to plan against. 'auto' = 2^(0.23t) "
                          "stabilizer rank; 'dense' = 2^n; 'tinject' = 2^t, "
                          "which is what this Qrack build actually allocates "
                          "for exact near-Clifford. Confirm with `probs "
                          "--limit 2` before trusting any of them.")
    ppa.add_argument("--outdir", default="patches")
    ppa.add_argument("--n-data", type=int, default=70)
    ppa.add_argument("--n-ancilla", type=int, default=27)
    ppa.add_argument("--target", type=float, default=27.0)
    ppa.add_argument("--max-cuts", type=int, default=12)
    ppa.add_argument("--cuts-from", default=None,
                     help="reuse the cut set from another manifest; required "
                          "when comparing F_elide across a doping ladder, so "
                          "every rung elides exactly the same gates")

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
    pp.add_argument("--sim-class", default="QrackSimulator",
                    choices=("QrackSimulator", "QrackStabilizer"),
                    help="QrackStabilizer answers prob() from a stochastic "
                         "rounding of the non-Clifford gates -- weak "
                         "simulation, verified by selftest to vary by O(1) "
                         "between identical calls. Do not use it for "
                         "amplitudes.")
    pp.add_argument("--sim-tuning", default="exact",
                    choices=("none", "exact", "full"),
                    help="post-construction settings. 'none' = constructor "
                         "flags only (matches a bare QrackSimulator call); "
                         "'exact' = also assert exact near-Clifford; 'full' = "
                         "also set_reactive_separate(False) and "
                         "set_ace_max_qb(). Bisect with these when a circuit "
                         "runs standalone but not here.")
    pp.add_argument("--method", default="chain",
                    choices=("chain", "permutation"),
                    help="permutation uses prob_perm (2 calls/sample) instead "
                         "of the ~n-call chain; validate with selftest first")
    pp.add_argument("--max-cpu-qb", default=None,
                    help="QRACK_MAX_CPU_QB; -1 (default) lifts the per-engine "
                         "qubit cap so QUnit has no reason to elide")
    pp.add_argument("--ace-max-qb", type=int, default=None,
                    help="max entangled subsystem before ACE elision; "
                         "defaults to the full circuit width so no elision "
                         "occurs")
    pp.add_argument("--sim-preset", default="near-clifford",
                    choices=sorted(SIM_PRESETS),
                    help="Qrack representation; run `probe` if prob() raises")
    pp.add_argument("--patches", default=None,
                    help="patch manifest from `patch`; computes the product "
                         "over patches instead of the exact amplitude")
    pp.add_argument("--allow-circuit-mismatch", action="store_true",
                    help="proceed when the QASM differs from the one recorded "
                         "in the result JSON")

    args = ap.parse_args(argv)
    lib = _qrack_lib_arg or getattr(args, "qrack_lib", None)

    if args.cmd in ("probs", "noisy", "probe", "selftest", "doctor", "env",
                    "idealsample"):
        print(f"xeb3d build {_build_stamp()}", file=sys.stderr)
        configure_qrack_lib(lib)
        configure_qrack_env(
            {"QRACK_MAX_CPU_QB": args.max_cpu_qb} if
            getattr(args, "max_cpu_qb", None) is not None else None)
        _warn_if_guard_disabled()
        if getattr(args, "sim_preset", None):
            warn_if_gpu_disabled(args.sim_preset)
        if getattr(args, "ace_max_qb", None):
            _ACE_MAX_QB[0] = args.ace_max_qb
    elif lib:
        configure_qrack_lib(lib)

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

    if args.cmd == "selftest":
        print(selftest(args.n_qubits, args.n_ancilla, args.n_t, args.seed,
                       args.sim_preset, args.sim_class, method=args.method))
        return 0

    if args.cmd == "env":
        configure_qrack_lib(lib, quiet=True)
        configure_qrack_env(quiet=True)
        print("# source this, or prefix any script that imports pyqrack")
        for k in ("PYQRACK_SHARED_LIB_PATH", "QRACK_MAX_CPU_QB",
                  "QRACK_MAX_PAGING_QB"):
            if os.environ.get(k):
                print(f'export {k}="{os.environ[k]}"')
        print("\n# the sampler is a separate process and does NOT inherit "
              "these unless exported")
        return 0

    if args.cmd == "doctor":
        if args.kwarg_search:
            n_q = args.n_qubits or circuit_structure(args.kwarg_search)["n_qubits"]
            print(kwarg_search(args.kwarg_search, n_q,
                               args.sim_class, args.mem_gb, args.timeout,
                               args.quick))
        else:
            print(doctor(args.n_max, sim_class=args.sim_class,
                         sim_preset=args.sim_preset))
        return 0

    if args.cmd == "probe":
        print(probe_simulator(args.qasm_path, args.n_data + args.n_ancilla))
        return 0

    if args.cmd == "budget":
        print(budget_report(args.qasm_path, args.ram_gb, args.hours,
                            args.shots, args.ref_nt, args.ref_seconds))
        return 0

    if args.cmd == "sample":
        print(exact_sample_report(qasm_path=args.qasm_path,
                                  n_data=args.n_data, n_ancilla=args.n_ancilla,
                                  shots=args.shots, out_path=args.out,
                                  seed=args.seed))
        return 0

    if args.cmd == "idealsample":
        ideal_sample(args.qasm_path, args.n_data, args.n_ancilla, args.shots,
                     args.out, args.sim_preset, args.seed)
        return 0

    if args.cmd == "synth":
        print(synth_report(n_data=args.n_data, n_ancilla=args.n_ancilla,
                           depth=args.depth, t_gates=args.t_gates,
                           out_path=args.out, seed=args.seed,
                           check_span=args.check_span))
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

    if getattr(args, "cost_model", None):
        _COST_MODEL[0] = args.cost_model
    if getattr(args, "n_data", None):
        _NDATA[0] = args.n_data
    if getattr(args, "n_ancilla", None):
        _NANC[0] = args.n_ancilla
    _PROTECT_CHECKS[0] = not getattr(args, "allow_broken_checks", False)

    if args.cmd == "cut":
        print(cut_report(args.qasm_path, args.max_cuts, args.target))
        return 0

    if args.cmd == "patch":
        print(patch_report(args.qasm_path, args.outdir, args.n_data,
                           args.n_ancilla, args.target, args.max_cuts,
                           args.cuts_from))
        return 0

    if args.cmd == "structure":
        print(structure_report(args.qasm_path))
        return 0

    if args.cmd == "probs":
        _SIM_CLASS[0] = args.sim_class
        _SIM_TUNING[0] = args.sim_tuning
        compute_ideal_probs(args.qasm_path, args.result_json, args.n_data,
                            args.n_ancilla, args.out, args.limit,
                            args.use_clone, args.force, args.workers,
                            args.allow_circuit_mismatch, args.patches,
                            args.sim_preset, args.method)
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
