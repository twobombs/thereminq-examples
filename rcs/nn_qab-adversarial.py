# Nearest-neighbor RCS: adversarial-hardened measurement harness.
#
# Extends nn_qab.py (Dan Strano and Claude) with the controls needed to
# separate "approximate simulation" from "XEB spoofing" in the sense of
# Gao, Kalinowski, Chou, Lukin, Barak & Choi, arXiv:2112.01657.
#
# example of a run:
# python3 nn_qab_adversarial.py 14 12 3 2 --reps 20 --consensus 3 --json out.json
# full example:
# nn_qab_adversarial.py bench 14 12 3 2 --reps 20 --bases zxy --json out.json
# nn_qab_adversarial.py run --runs 4 --reps 50 14 12 3 2 --bases zxy --csv summary.csv
# nn_qab_adversarial.py pool 'runs/*.json' --metric xeb
#
# Arms scored against the SAME ideal distribution and the SAME circuit:
#   ace       -- QrackAceBackend as shipped (the method under test)
#   severed   -- identical, with seam reconciliation disabled: elision
#                WITHOUT mending. This is the local spoofer control.
#   ceiling   -- `shots` samples drawn from the exact ideal distribution.
#                The maximum any simulator can score at this shot budget.
#   null      -- ace counts scored against an INDEPENDENT circuit's ideal
#                probabilities. Empirical demonstration of p-blindness:
#                a method exploiting knowledge of p cannot sit at zero here.
#
# Metrics per arm:
#   xeb          -- nn_qab.py's estimator (XEB normalized by the measured
#                   second moment rather than assuming Porter-Thomas)
#   xeb_google   -- 2^N sum(p q) - 1, directly comparable to the literature
#   xeb_tail     -- same estimator restricted to bitstrings with p below
#                   the median. Head-tilted spoofers have no tail structure.
#   hog          -- heavy output generation probability
#   hellinger    -- sum(sqrt(p q)), classical fidelity; shot-biased, but the
#                   bias is common-mode at fixed shot count, so read it
#                   against the ceiling arm
#   tvd          -- total variation distance, same caveat
#   zz_err_seam  -- mean |<Z_i Z_j>_ace - <Z_i Z_j>_ideal| over coupled pairs
#                   touching a seam
#   zz_err_bulk  -- the same over coupled pairs entirely inside a patch.
#                   If seam ~ bulk, the severed-circuit analogy is dead.
#   z_err_seam / z_err_bulk -- single-qubit version, cheaper and noisier
#
# Repetitions produce mean, sd, sem and a bootstrap 95% CI per metric per
# arm, so every point carries a range bar.
#
# THREE MODES (see --help):
#   bench  one run
#   run    R independent seeded bench runs as separate processes, then pool
#   pool   pool bench JSON files that already exist
#
# The bare positional form still means bench:
#   python3 nn_qab_adversarial.py 14 12 3 2 --reps 20 --bases zxy
#
# WHY `run` EXISTS. Two bench runs of the identical command at n=20 gave
# paired ace-minus-severed differences of +0.1376 (t=+3.98, sign 17/20,
# p=0.0026) and -0.0407 (t=-1.59, sign 9/20, p=0.82). Cochran's Q on those
# two put I^2 at 94%: the within-run standard error is anticonservative, so
# a single run's t-test will manufacture a result the next run reverses.
# Treat the RUN as the unit of analysis and quote the random-effects line.

import argparse
import glob
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time

from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# Qrack shared library resolution
# ---------------------------------------------------------------------------
# PyQrack resolves PYQRACK_SHARED_LIB_PATH at *import* time (ctypes.CDLL), so
# this block must run before `from pyqrack import ...` -- setting it later has
# no effect on which .so is bound.
#
# ThereminQ container builds place a locally-compiled libqrack_pinvoke.so at
# /usr/local/lib/qrack/ rather than relying on the copy bundled inside the
# pyqrack wheel. Override the default with the QRACK_LIB_PATH env var.
#
# NOTE: this only selects which file ctypes opens. If libqrack_pinvoke.so links
# against a sibling libqrack.so (or OpenCL/CUDA runtimes) in the same directory,
# the dynamic loader must also be able to find those. Setting LD_LIBRARY_PATH
# from inside Python is too late -- the loader reads it at process start. Either
# build with `-Wl,-rpath,/usr/local/lib/qrack` or export it in the container
# entrypoint:
#
#     export LD_LIBRARY_PATH=/usr/local/lib/qrack:${LD_LIBRARY_PATH}
QRACK_LIB_PATH = os.environ.get(
    "QRACK_LIB_PATH", "/usr/local/lib/qrack/libqrack_pinvoke.so"
)
if os.path.isfile(QRACK_LIB_PATH):
    os.environ["PYQRACK_SHARED_LIB_PATH"] = QRACK_LIB_PATH
    QRACK_LIB_SOURCE = QRACK_LIB_PATH
else:
    # Fall back to whatever ships with the installed pyqrack wheel.
    os.environ.pop("PYQRACK_SHARED_LIB_PATH", None)
    QRACK_LIB_SOURCE = "pyqrack-bundled"
    print(
        f"warning: {QRACK_LIB_PATH} not found; using bundled pyqrack library",
        file=sys.stderr,
    )

from pyqrack import QrackSimulator, QrackAceBackend  # noqa: E402


# ---------------------------------------------------------------------------
# Geometry (identical to nn_qab.py -- do not "improve", results must compare)
# ---------------------------------------------------------------------------

def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (row_len, col_len)


def boundary_qubits(sim):
    """Logical qubits with more than one hardware replica, i.e. the ones
    sitting on a patch seam. ACE's own _unpack() is the source of truth.
    """
    return [lq for lq in range(sim.num_qubits()) if len(sim._unpack(lq)) > 1]


def bulk_to_boundary_ratio(sim):
    n = sim.num_qubits()
    boundary = len(boundary_qubits(sim))
    bulk = n - boundary
    return bulk / boundary if boundary else float("inf")


# ---------------------------------------------------------------------------
# Gate wrappers (verbatim from nn_qab.py)
# ---------------------------------------------------------------------------

def u(sim, q, th, ph, lm):
    sim.u(q, th, ph, lm)


def cx(sim, q1, q2):
    sim.mcx([q1], q2)


def cy(sim, q1, q2):
    sim.mcy([q1], q2)


def cz(sim, q1, q2):
    sim.mcz([q1], q2)


def acx(sim, q1, q2):
    sim.macx([q1], q2)


def acy(sim, q1, q2):
    sim.macy([q1], q2)


def acz(sim, q1, q2):
    sim.macz([q1], q2)


def swap_native(sim, q1, q2):
    sim.swap(q1, q2)


def swap_cnot(sim, q1, q2):
    if random.getrandbits(1):
        q1, q2 = q2, q1
    sim.mcx([q1], q2)
    sim.mcx([q2], q1)
    sim.mcx([q1], q2)


def iswap_native(sim, q1, q2):
    sim.iswap(q1, q2)


def iswap_cnot(sim, q1, q2):
    swap_cnot(sim, q1, q2)
    sim.mcz([q1], q2)
    sim.s(q1)
    sim.s(q2)


def iiswap_native(sim, q1, q2):
    sim.adjiswap(q1, q2)


def iiswap_cnot(sim, q1, q2):
    sim.adjs(q2)
    sim.adjs(q1)
    sim.mcz([q1], q2)
    swap_cnot(sim, q1, q2)


def pswap_native(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.swap(q1, q2)


def pswap_cnot(sim, q1, q2):
    sim.mcz([q1], q2)
    swap_cnot(sim, q1, q2)


def mswap_native(sim, q1, q2):
    sim.swap(q1, q2)
    sim.mcz([q1], q2)


def mswap_cnot(sim, q1, q2):
    swap_cnot(sim, q1, q2)
    sim.mcz([q1], q2)


def nswap_native(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.swap(q1, q2)
    sim.mcz([q1], q2)


def nswap_cnot(sim, q1, q2):
    sim.mcz([q1], q2)
    swap_cnot(sim, q1, q2)
    sim.mcz([q1], q2)


NATIVE_GATES = (
    swap_native, pswap_native, mswap_native, nswap_native,
    iswap_native, iiswap_native, cx, cy, cz, acx, acy, acz,
)

CNOT_GATES = (
    swap_cnot, pswap_cnot, mswap_cnot, nswap_cnot,
    iswap_cnot, iiswap_cnot, cx, cy, cz, acx, acy, acz,
)

SWAP_RATIO_THRESHOLD = 7.0


def run_circuit(sim, circ):
    for g in circ:
        g[0](sim, *g[1:])


def build_circuit(width, depth, two_bit_gates):
    """Returns (gate_list, coupled_pairs). coupled_pairs is the set of
    (i, j) that actually receive a two-qubit gate, which is what the
    correlator classification below is built on -- classifying by lattice
    adjacency instead would count pairs the circuit never couples.
    """
    lcv_range = range(width)
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    row_len, col_len = factor_width(width)

    qc = []
    pairs = set()

    for _ in range(depth):
        for i in lcv_range:
            th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
            # Keep it Haar-random towards the poles:
            th = math.asin(th / math.pi)
            qc.append((u, i, th, ph, lm))

        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1)
                temp_col = temp_col + (1 if (gate & 1) else 0)

                if temp_row < 0:
                    temp_row = temp_row + row_len
                if temp_col < 0:
                    temp_col = temp_col + col_len
                if temp_row >= row_len:
                    temp_row = temp_row - row_len
                if temp_col >= col_len:
                    temp_col = temp_col - col_len

                b1 = col * row_len + row
                b2 = temp_col * row_len + temp_row

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                qc.append((g, b1, b2))
                pairs.add((min(b1, b2), max(b1, b2)))

    return qc, sorted(pairs)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def counts_to_sparse(counts, shots):
    """(index array, probability array) for observed outcomes only."""
    if not counts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    idx = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
    val = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
    return idx, val / shots


def score(ideal_probs, counts, shots, tail_mask, denom_full, denom_tail,
          tail_center_sum, median_p, u_u):
    """All distribution-level metrics for one arm, in one pass.

    ideal_probs is the dense exact distribution; counts is sparse. The
    xeb numerator only touches observed outcomes because (q - u_u)
    contributes a constant over the unobserved remainder, which is folded
    in analytically -- this keeps the cost O(observed) not O(2^N).
    """
    idx, q_obs = counts_to_sparse(counts, shots)
    p_obs = ideal_probs[idx] if len(idx) else np.zeros(0)

    # sum over ALL x of (p - u)(q - u)
    #   = sum_obs (p - u)(q - u) + sum_unobs (p - u)(0 - u)
    #   = sum_obs (p - u) q   - u * sum_all (p - u)
    #   = sum_obs (p - u) q            [since sum_all (p - u) == 0]
    numer = float(np.sum((p_obs - u_u) * q_obs))
    xeb = numer / denom_full

    # Google-normalized XEB: 2^N sum(p q) - 1
    n_pow = len(ideal_probs)
    xeb_google = float(n_pow * np.sum(p_obs * q_obs) - 1.0)

    # Tail-restricted: only bitstrings with p below the median. A method
    # that concentrates mass on the head of p scores ~0 here.
    #
    # NOTE the -u_u term cannot be dropped the way it can above. Over the
    # full support sum(p - u_u) == 0, which is what lets the dense
    # remainder vanish; over a strict subset it does not, so the constant
    # sum_tail(p - u_u) has to be carried explicitly or the whole arm
    # picks up a large negative offset. Calibrate against the ceiling arm,
    # which must land at 1.0 for this to be right.
    if len(idx):
        in_tail = tail_mask[idx]
        numer_tail = float(np.sum((p_obs[in_tail] - u_u) * q_obs[in_tail]))
        numer_tail -= u_u * tail_center_sum
    else:
        numer_tail = -u_u * tail_center_sum
    xeb_tail = numer_tail / denom_tail if denom_tail > 0 else float("nan")

    hog = float(np.sum(q_obs[p_obs > median_p])) if len(idx) else 0.0

    # Classical fidelity and TVD. Both are shot-biased at these budgets;
    # read them relative to the ceiling arm, never in absolute terms.
    hellinger = float(np.sum(np.sqrt(p_obs * q_obs))) if len(idx) else 0.0
    obs_p_mass = float(np.sum(p_obs)) if len(idx) else 0.0
    tvd = 0.5 * (float(np.sum(np.abs(p_obs - q_obs))) + (1.0 - obs_p_mass))

    return {
        "xeb": xeb,
        "xeb_google": xeb_google,
        "xeb_tail": xeb_tail,
        "hog": hog,
        "hellinger": hellinger,
        "tvd": tvd,
    }


def sample_ideal(ideal_probs, shots, rng, chunk=1 << 20):
    """Draw `shots` samples from the exact distribution without building a
    full cumulative array (which would double peak memory at width 26+).
    Sorted uniforms are walked against a chunked running sum.
    """
    u_sorted = np.sort(rng.random(shots))
    out = np.empty(shots, dtype=np.int64)
    running = 0.0
    k = 0
    n_pow = len(ideal_probs)
    for start in range(0, n_pow, chunk):
        if k >= shots:
            break
        block = ideal_probs[start:start + chunk]
        cum = np.cumsum(block) + running
        running = cum[-1]
        take = np.searchsorted(cum, u_sorted[k:], side="right")
        n_here = int(np.searchsorted(take, len(block), side="left"))
        if n_here:
            out[k:k + n_here] = start + take[:n_here]
            k += n_here
    if k < shots:  # float drift on the last edge
        out[k:] = n_pow - 1
    return dict(Counter(out.tolist()))


# ---------------------------------------------------------------------------
# Correlators
# ---------------------------------------------------------------------------

def rotate_to_basis(sim, width, basis):
    """Rotate every qubit so a computational-basis measurement reads out
    the requested Pauli. Verified against a Bell state: <XX>=+1, <YY>=-1,
    <ZZ>=+1 on both QrackSimulator and QrackAceBackend.

    Z-basis correlators are DIAGONAL and therefore blind to relative
    phase. If a repair mechanism is restoring coherence between replicas
    rather than populations, ZZ cannot see it by construction -- which is
    why running at least one off-diagonal basis is not optional when the
    question is whether seam repair does physical work.
    """
    if basis == "z":
        return
    if basis == "x":
        for q in range(width):
            sim.h(q)
    elif basis == "y":
        for q in range(width):
            sim.adjs(q)
            sim.h(q)
    else:
        raise ValueError(f"unknown basis {basis!r}")


def ideal_correlators(ideal_probs, pairs, singles, chunk=1 << 20):
    """Exact <Z_i> and <Z_i Z_j> from the dense ideal distribution,
    computed chunked so we never materialize 2^N-sized bit arrays.
    """
    zz = np.zeros(len(pairs), dtype=np.float64)
    z = np.zeros(len(singles), dtype=np.float64)
    n_pow = len(ideal_probs)
    for start in range(0, n_pow, chunk):
        block = ideal_probs[start:start + chunk]
        idx = np.arange(start, start + len(block), dtype=np.int64)
        for k, q in enumerate(singles):
            s = 1.0 - 2.0 * ((idx >> q) & 1)
            z[k] += float(np.dot(block, s))
        for k, (i, j) in enumerate(pairs):
            s = 1.0 - 2.0 * (((idx >> i) ^ (idx >> j)) & 1)
            zz[k] += float(np.dot(block, s))
    return z, zz


def sampled_correlators(counts, shots, pairs, singles):
    idx, q = counts_to_sparse(counts, shots)
    if not len(idx):
        return np.zeros(len(singles)), np.zeros(len(pairs))
    z = np.array([float(np.dot(q, 1.0 - 2.0 * ((idx >> s) & 1)))
                  for s in singles])
    zz = np.array([float(np.dot(q, 1.0 - 2.0 * (((idx >> i) ^ (idx >> j)) & 1)))
                   for (i, j) in pairs])
    return z, zz


def classify_pairs(pairs, boundary_set, max_per_class=None, rng=None):
    """A coupled pair is 'seam' if either endpoint carries replicas."""
    seam = [p for p in pairs if p[0] in boundary_set or p[1] in boundary_set]
    seam_set = set(seam)
    bulk = [p for p in pairs if p not in seam_set]
    if max_per_class and rng is not None:
        if len(seam) > max_per_class:
            seam = [seam[i] for i in
                    rng.choice(len(seam), max_per_class, replace=False)]
        if len(bulk) > max_per_class:
            bulk = [bulk[i] for i in
                    rng.choice(len(bulk), max_per_class, replace=False)]
    return seam, bulk


# ---------------------------------------------------------------------------
# One repetition
# ---------------------------------------------------------------------------

def make_backend(width, lrc, lrr, sever, sever_mode="correct"):
    """sever_mode selects WHICH repair mechanism the ablation removes:

      correct  -- monkeypatch _correct() to a no-op. _correct() already
                  early-returns on bulk qubits (len(hq) == 1), so this
                  removes ONLY the seam reconciliation between replicas.
      edetect  -- construct with is_error_detection=False, disabling the
                  detect-and-post-select gadget instead.
      both     -- both of the above.

    Use more than one. The ACE approximation error has several sources
    (shadow couplings, replica drift, the detection gadget) and if
    "severed" tracks "ace" under one mode that may only mean you disabled
    the mechanism that was not carrying the load.
    """
    sim = QrackAceBackend(
        width, long_range_columns=lrc, long_range_rows=lrr, is_torus=True,
        is_error_detection=not (sever and sever_mode in ("edetect", "both")),
    )
    if sever and sever_mode in ("correct", "both"):
        sim._correct = lambda *a, **k: None
    return sim


def shift_index(i, row_shift, col_shift, row_len, col_len):
    row = i % row_len
    col = i // row_len
    return ((col + col_shift) % col_len) * row_len + (row + row_shift) % row_len


def one_rep(width, depth, lrc, lrr, shots, rng, do_null=True,
            consensus_instances=0, corr_pairs_max=16,
            sever_mode="correct", bases=("z",)):
    row_len, col_len = factor_width(width)
    n_pow = 1 << width

    probe = make_backend(width, lrc, lrr, False, sever_mode)
    bset = set(boundary_qubits(probe))
    ratio = bulk_to_boundary_ratio(probe)
    del probe

    two_bit_gates = (CNOT_GATES if ratio >= SWAP_RATIO_THRESHOLD
                     else NATIVE_GATES)

    qc, pairs = build_circuit(width, depth, two_bit_gates)

    # ---- exact ground truth -------------------------------------------
    t0 = time.perf_counter()
    sim_ideal = QrackSimulator(width)
    run_circuit(sim_ideal, qc)
    ideal_probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
    t_ideal = time.perf_counter() - t0
    # NOTE sim_ideal is deliberately kept alive until the correlator block
    # below, so each extra basis can be reached by clone-and-rotate rather
    # than re-running the circuit. Peak memory is one statevector plus one
    # clone plus one probability array; at large width, drop to
    # --bases z if that is tight.

    u_u = 1.0 / n_pow
    centered = ideal_probs - u_u
    denom_full = float(np.dot(centered, centered))
    median_p = float(np.median(ideal_probs))
    tail_mask = ideal_probs < median_p
    denom_tail = float(np.dot(centered[tail_mask], centered[tail_mask]))
    tail_center_sum = float(np.sum(centered[tail_mask]))

    # ---- decoy ground truth for the null band -------------------------
    decoy_probs = None
    if do_null:
        qc_decoy, _ = build_circuit(width, depth, two_bit_gates)
        sd = QrackSimulator(width)
        run_circuit(sd, qc_decoy)
        decoy_probs = np.asarray(sd.out_probs(), dtype=np.float64)
        del sd
        dc = decoy_probs - u_u
        decoy_denom = float(np.dot(dc, dc))
        decoy_median = float(np.median(decoy_probs))
        decoy_tail = decoy_probs < decoy_median
        decoy_denom_tail = float(np.dot(dc[decoy_tail], dc[decoy_tail]))
        decoy_tail_center = float(np.sum(dc[decoy_tail]))

    # ---- correlator pair selection ------------------------------------
    seam_pairs, bulk_pairs = classify_pairs(
        pairs, bset, max_per_class=corr_pairs_max, rng=rng
    )
    seam_singles = sorted(bset)[:corr_pairs_max]
    bulk_singles = [q for q in range(width) if q not in bset][:corr_pairs_max]
    all_pairs = seam_pairs + bulk_pairs
    all_singles = seam_singles + bulk_singles

    # ---- arms ----------------------------------------------------------
    # Z-basis counts drive every distribution-level metric (XEB is defined
    # in the computational basis). Extra bases are used for correlators
    # only, so the cost is one extra measure_shots per arm per basis.
    arms = {}
    backends = {}
    timings = {"ideal": t_ideal}

    for name, sever in (("ace", False), ("severed", True)):
        t = time.perf_counter()
        sim = make_backend(width, lrc, lrr, sever, sever_mode)
        run_circuit(sim, qc)
        counts = dict(Counter(sim.measure_shots(list(range(width)), shots)))
        timings[name] = time.perf_counter() - t
        arms[name] = counts
        backends[name] = (sim, sever)

    ceiling_counts = sample_ideal(ideal_probs, shots, rng)
    arms["ceiling"] = ceiling_counts
    timings["ceiling"] = 0.0

    # ---- correlators, per basis ---------------------------------------
    corr = {}          # corr[arm][basis] = (z_err_seam, z_err_bulk, zz_s, zz_b)
    for name in arms:
        corr[name] = {}

    for basis in bases:
        cl = sim_ideal.clone()
        rotate_to_basis(cl, width, basis)
        probs_b = np.asarray(cl.out_probs(), dtype=np.float64)
        del cl
        z_ideal, zz_ideal = ideal_correlators(probs_b, all_pairs, all_singles)

        basis_counts = {}
        if basis == "z":
            basis_counts["ace"] = arms["ace"]
            basis_counts["severed"] = arms["severed"]
            basis_counts["ceiling"] = arms["ceiling"]
        else:
            for name, (sim, sever) in backends.items():
                c = sim.clone()
                if sever and sever_mode in ("correct", "both"):
                    # clone() constructs a fresh backend, so the severing
                    # monkeypatch does NOT survive it. Re-apply, or the
                    # off-diagonal bases silently measure a repaired arm.
                    c._correct = lambda *a, **k: None
                rotate_to_basis(c, width, basis)
                basis_counts[name] = dict(
                    Counter(c.measure_shots(list(range(width)), shots))
                )
                del c
            basis_counts["ceiling"] = sample_ideal(probs_b, shots, rng)

        ns, nb = len(seam_pairs), len(bulk_pairs)
        nss = len(seam_singles)
        for name, cnts in basis_counts.items():
            z_s, zz_s = sampled_correlators(cnts, shots, all_pairs, all_singles)
            corr[name][basis] = {
                f"{basis}{basis}_err_seam": (
                    float(np.mean(np.abs(zz_s[:ns] - zz_ideal[:ns])))
                    if ns else float("nan")),
                f"{basis}{basis}_err_bulk": (
                    float(np.mean(np.abs(zz_s[ns:] - zz_ideal[ns:])))
                    if nb else float("nan")),
                f"{basis}_err_seam": (
                    float(np.mean(np.abs(z_s[:nss] - z_ideal[:nss])))
                    if nss else float("nan")),
                f"{basis}_err_bulk": (
                    float(np.mean(np.abs(z_s[nss:] - z_ideal[nss:])))
                    if len(bulk_singles) else float("nan")),
            }
        del probs_b

    del sim_ideal
    for sim, _ in backends.values():
        del sim
    backends.clear()

    # ---- optional: index-shifted consensus, reported PER INSTANCE ------
    per_instance = []
    if consensus_instances:
        for inst in range(consensus_instances):
            smap = [shift_index(i, inst, inst, row_len, col_len)
                    for i in range(width)]
            inst_circ = []
            for g in qc:
                if g[0] is u:
                    _, i, th, ph, lm = g
                    inst_circ.append((u, smap[i], th, ph, lm))
                else:
                    fn, q1, q2 = g
                    inst_circ.append((fn, smap[q1], smap[q2]))
            s = make_backend(width, lrc, lrr, False, sever_mode)
            run_circuit(s, inst_circ)
            raw = s.measure_shots(list(range(width)), max(1, shots // consensus_instances))
            unshifted = []
            for smp in raw:
                r = 0
                for orig_i, shifted_i in enumerate(smap):
                    r |= ((smp >> shifted_i) & 1) << orig_i
                unshifted.append(r)
            c = dict(Counter(unshifted))
            per_instance.append(score(
                ideal_probs, c, len(unshifted), tail_mask,
                denom_full, denom_tail, tail_center_sum, median_p, u_u
            ))

    # ---- scoring -------------------------------------------------------
    results = {}
    for name, counts in arms.items():
        m = score(ideal_probs, counts, shots, tail_mask,
                  denom_full, denom_tail, tail_center_sum, median_p, u_u)
        for basis in bases:
            m.update(corr[name][basis])
        m["seconds"] = timings.get(name, float("nan"))
        results[name] = m

    # null band: ace's own counts, scored against an unrelated circuit
    if do_null:
        results["null"] = score(
            decoy_probs, arms["ace"], shots, decoy_tail,
            decoy_denom, decoy_denom_tail, decoy_tail_center,
            decoy_median, u_u
        )
        results["null"]["seconds"] = 0.0

    if per_instance:
        for k, m in enumerate(per_instance):
            results[f"consensus_inst{k}"] = m
            results[f"consensus_inst{k}"]["seconds"] = 0.0

    meta = {
        "qrack_lib": QRACK_LIB_SOURCE,
        "bulk_to_boundary": ratio,
        "n_boundary": len(bset),
        "n_bulk": width - len(bset),
        "n_seam_pairs": len(seam_pairs),
        "n_bulk_pairs": len(bulk_pairs),
        "ideal_seconds": t_ideal,
    }
    return results, meta


# ---------------------------------------------------------------------------
# Aggregation with range bars
# ---------------------------------------------------------------------------

def bootstrap_ci(vals, n_boot=2000, alpha=0.05, rng=None):
    v = np.asarray([x for x in vals if not math.isnan(x)], dtype=np.float64)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0)
    draws = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return (float(np.quantile(draws, alpha / 2)),
            float(np.quantile(draws, 1 - alpha / 2)))


def aggregate(reps, rng):
    out = {}
    arm_names = reps[0].keys()
    for arm in arm_names:
        metrics = reps[0][arm].keys()
        out[arm] = {}
        for m in metrics:
            vals = [r[arm][m] for r in reps]
            clean = [v for v in vals if not math.isnan(v)]
            if not clean:
                out[arm][m] = {"mean": float("nan")}
                continue
            lo, hi = bootstrap_ci(clean, rng=rng)
            sd = statistics.stdev(clean) if len(clean) > 1 else 0.0
            out[arm][m] = {
                "mean": statistics.mean(clean),
                "sd": sd,
                "sem": sd / math.sqrt(len(clean)) if clean else float("nan"),
                "ci_lo": lo,
                "ci_hi": hi,
                "n": len(clean),
            }
    return out


BASE_METRICS = ["xeb", "xeb_google", "xeb_tail", "hog", "hellinger", "tvd"]


def row_metrics(bases):
    rows = list(BASE_METRICS)
    for b in bases:
        rows += [f"{b}{b}_err_seam", f"{b}{b}_err_bulk",
                 f"{b}_err_seam", f"{b}_err_bulk"]
    return rows + ["seconds"]


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------
# ace and severed run the SAME circuit in every repetition, so they are
# paired observations. Comparing their independent bootstrap CIs throws
# that pairing away and badly understates the evidence: at width 14 the
# unpaired intervals nearly touch while the paired test gives t ~ 4 and a
# sign test at p ~ 0.003 on the same 20 reps. Always read this table for
# ace-vs-severed; the per-arm table is for magnitudes, not comparisons.

def paired_stats(reps, arm_a, arm_b, metric, rng, n_boot=20000):
    d = []
    for r in reps:
        if arm_a not in r or arm_b not in r:
            return None
        va, vb = r[arm_a].get(metric), r[arm_b].get(metric)
        if va is None or vb is None or math.isnan(va) or math.isnan(vb):
            continue
        d.append(va - vb)
    if len(d) < 2:
        return None
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    sd = float(d.std(ddof=1))
    sem = sd / math.sqrt(n) if sd > 0 else float("nan")
    draws = rng.choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    wins = int((d > 0).sum())
    p_sign = 2.0 * sum(math.comb(n, k) for k in range(max(wins, n - wins), n + 1))
    p_sign = min(1.0, p_sign / (2 ** n))
    return {
        "mean_diff": float(d.mean()),
        "sd": sd,
        "sem": sem,
        "t": float(d.mean() / sem) if sem and not math.isnan(sem) else float("nan"),
        "dz": float(d.mean() / sd) if sd > 0 else float("nan"),
        "ci_lo": float(np.quantile(draws, 0.025)),
        "ci_hi": float(np.quantile(draws, 0.975)),
        "wins": wins,
        "n": n,
        "p_sign": p_sign,
    }


def print_paired(reps, args, rng, arm_a="ace", arm_b="severed"):
    if arm_a not in reps[0] or arm_b not in reps[0]:
        return
    print(f"paired {arm_a} - {arm_b} (same circuit each rep)")
    print("  sign convention: xeb/xeb_tail/hog/hellinger are SCORES, so a "
          "positive diff favours " + arm_a + ".")
    print("  tvd and every *_err_* row are ERRORS, so a NEGATIVE diff "
          "favours " + arm_a + ".")
    print("metric".ljust(16) + "mean diff".ljust(12) + "95% CI".ljust(22)
          + "t".ljust(8) + "dz".ljust(7) + "sign".ljust(9) + "p")
    print("-" * 82)
    for m in row_metrics(args.bases):
        if m == "seconds":
            continue
        st = paired_stats(reps, arm_a, arm_b, m, rng)
        if st is None:
            continue
        print(m.ljust(16)
              + f"{st['mean_diff']:+.4f}".ljust(12)
              + f"[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}]".ljust(22)
              + f"{st['t']:+.2f}".ljust(8)
              + f"{st['dz']:+.2f}".ljust(7)
              + f"{st['wins']}/{st['n']}".ljust(9)
              + f"{st['p_sign']:.4f}")
    print()


def print_table(agg, meta, args):
    print()
    print(f"width={args.width} depth={args.depth} lrc={args.lrc} lrr={args.lrr} "
          f"shots={args.shots} reps={args.reps} sever_mode={args.sever_mode} "
          f"bases={''.join(args.bases)}")
    print(f"qrack_lib={meta['qrack_lib']}")
    print(f"bulk={meta['n_bulk']} boundary={meta['n_boundary']} "
          f"B-to-B={meta['bulk_to_boundary']:.4g}  "
          f"seam_pairs={meta['n_seam_pairs']} bulk_pairs={meta['n_bulk_pairs']}")
    print()
    arms = [a for a in ("ace", "severed", "ceiling", "null") if a in agg]
    arms += [a for a in agg if a.startswith("consensus_inst")]
    w = 26
    header = "metric".ljust(16) + "".join(a.ljust(w) for a in arms)
    print(header)
    print("-" * len(header))
    for m in row_metrics(args.bases):
        row = m.ljust(16)
        for a in arms:
            s = agg[a].get(m)
            if not s or math.isnan(s.get("mean", float("nan"))):
                row += "--".ljust(w)
            else:
                row += f"{s['mean']:+.4f} [{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}]".ljust(w)
        print(row)
    print()
    if "ace" in agg and "ceiling" in agg:
        c = agg["ceiling"]["xeb"]["mean"]
        if c:
            print(f"ace xeb as fraction of achievable ceiling: "
                  f"{agg['ace']['xeb']['mean'] / c:.3f}")
        ch = agg["ceiling"]["hellinger"]["mean"]
        if ch:
            print(f"ace hellinger as fraction of ceiling:       "
                  f"{agg['ace']['hellinger']['mean'] / ch:.3f}")
            print(f"severed hellinger as fraction of ceiling:   "
                  f"{agg['severed']['hellinger']['mean'] / ch:.3f}")
    print()


def bench_main(argv):
    ap = argparse.ArgumentParser(prog="nn_qab_adversarial.py bench")
    ap.add_argument("width", type=int)
    ap.add_argument("depth", type=int)
    ap.add_argument("lrc", type=int, nargs="?", default=4)
    ap.add_argument("lrr", type=int, nargs="?", default=4)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--shots", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-null", action="store_true")
    ap.add_argument("--consensus", type=int, default=0,
                    help="number of index-shifted instances to score "
                         "INDIVIDUALLY (0 disables)")
    ap.add_argument("--corr-pairs", type=int, default=16)
    ap.add_argument("--sever-mode", choices=("correct", "edetect", "both"),
                    default="correct",
                    help="which repair mechanism the severed arm removes")
    ap.add_argument("--bases", type=str, default="z",
                    help="Pauli bases for correlators, e.g. zxy. Z alone is "
                         "diagonal and cannot see phase repair. Each extra "
                         "basis costs one measure_shots per ACE arm per rep.")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    if args.shots is None:
        args.shots = 1 << min(10, args.width + 2)
    args.bases = tuple(dict.fromkeys(args.bases.lower()))
    for b in args.bases:
        if b not in ("x", "y", "z"):
            ap.error(f"--bases must contain only x, y, z (got {b!r})")
    if args.seed is not None:
        random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    reps, meta = [], None
    for r in range(args.reps):
        res, meta = one_rep(
            args.width, args.depth, args.lrc, args.lrr, args.shots, rng,
            do_null=not args.no_null,
            consensus_instances=args.consensus,
            corr_pairs_max=args.corr_pairs,
            sever_mode=args.sever_mode,
            bases=args.bases,
        )
        reps.append(res)
        print(f"  rep {r + 1}/{args.reps} "
              f"ace_xeb={res['ace']['xeb']:+.4f} "
              f"sev_xeb={res['severed']['xeb']:+.4f} "
              f"ceil_xeb={res['ceiling']['xeb']:+.4f}"
              + (f" null_xeb={res['null']['xeb']:+.4f}" if 'null' in res else ""),
              file=sys.stderr)

    agg = aggregate(reps, rng)
    print_table(agg, meta, args)
    print_paired(reps, args, rng)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "meta": meta,
                       "aggregate": agg, "reps": reps}, f, indent=2)
        print(f"wrote {args.json}")

    return 0



# ---------------------------------------------------------------------------
# Distributions without scipy (the container may not have it)
# ---------------------------------------------------------------------------

def _gammap_series(a, x, itmax=500, eps=3e-12):
    ap, s, d = a, 1.0 / a, 1.0 / a
    for _ in range(itmax):
        ap += 1.0
        d *= x / ap
        s += d
        if abs(d) < abs(s) * eps:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammaq_cf(a, x, itmax=500, eps=3e-12, fpmin=1e-300):
    b, c, d = x + 1.0 - a, 1.0 / fpmin, 1.0 / (x + 1.0 - a)
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_sf(x, k):
    """Upper tail of the chi-square distribution. Used for Cochran's Q."""
    if x <= 0:
        return 1.0
    a, xx = k / 2.0, x / 2.0
    if xx < a + 1.0:
        return 1.0 - _gammap_series(a, xx)
    return _gammaq_cf(a, xx)


def norm_sf2(z):
    """Two-sided normal tail."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def sign_p(wins, n):
    if n == 0:
        return float("nan")
    k = max(wins, n - wins)
    return min(1.0, 2.0 * sum(math.comb(n, i)
                              for i in range(k, n + 1)) / (2 ** n))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

GEOMETRY_KEYS = ("width", "depth", "lrc", "lrr", "shots", "sever_mode", "bases")


def load_runs(paths):
    runs = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        if "reps" not in d or "args" not in d:
            print(f"skipping {p}: not a harness JSON", file=sys.stderr)
            continue
        runs.append({
            "path": p,
            "args": d["args"],
            "meta": d.get("meta", {}),
            "reps": d["reps"],
            "seed": d["args"].get("seed"),
        })
    if not runs:
        sys.exit("no usable JSON files")

    # Pooling runs with different geometry is meaningless. Refuse rather
    # than silently averaging incomparable things.
    ref = {k: runs[0]["args"].get(k) for k in GEOMETRY_KEYS}
    bad = []
    for r in runs[1:]:
        cur = {k: r["args"].get(k) for k in GEOMETRY_KEYS}
        if cur != ref:
            bad.append((r["path"], cur))
    if bad:
        print("refusing to pool: geometry/settings differ from "
              f"{runs[0]['path']} ({ref})", file=sys.stderr)
        for path, cur in bad:
            print(f"  {path}: {cur}", file=sys.stderr)
        sys.exit(1)

    seeds = [r["seed"] for r in runs]
    if any(s is None for s in seeds):
        print("warning: at least one run has no --seed; it cannot be "
              "reproduced if it turns out to be an outlier", file=sys.stderr)
    elif len(set(seeds)) != len(seeds):
        print(f"warning: duplicate seeds {seeds} -- runs are not independent",
              file=sys.stderr)
    return runs


def paired_diffs(rep_list, arm_a, arm_b, metric):
    out = []
    for r in rep_list:
        if arm_a not in r or arm_b not in r:
            continue
        va, vb = r[arm_a].get(metric), r[arm_b].get(metric)
        if va is None or vb is None:
            continue
        if isinstance(va, float) and math.isnan(va):
            continue
        if isinstance(vb, float) and math.isnan(vb):
            continue
        out.append(va - vb)
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Meta-analysis
# ---------------------------------------------------------------------------

def meta_analyze(per_run):
    """per_run: list of (mean, se, n). Returns fixed and random effects."""
    usable = [(m, se, n) for (m, se, n) in per_run
              if se and se > 0 and not math.isnan(se)]
    if len(usable) < 1:
        return None
    m = np.array([x[0] for x in usable])
    se = np.array([x[1] for x in usable])
    w = 1.0 / se ** 2

    fe = float((w * m).sum() / w.sum())
    fe_se = float(math.sqrt(1.0 / w.sum()))

    k = len(usable)
    Q = float((w * (m - fe) ** 2).sum())
    df = k - 1
    q_p = chi2_sf(Q, df) if df > 0 else float("nan")
    i2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 and df > 0 else 0.0

    if df > 0:
        c = w.sum() - (w ** 2).sum() / w.sum()
        tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0
    else:
        tau2 = 0.0

    wr = 1.0 / (se ** 2 + tau2)
    re = float((wr * m).sum() / wr.sum())
    re_se = float(math.sqrt(1.0 / wr.sum()))

    return {
        "k": k, "fe": fe, "fe_se": fe_se,
        "Q": Q, "df": df, "Q_p": q_p, "I2": i2, "tau2": tau2,
        "re": re, "re_se": re_se,
    }


def analyze_metric(runs, metric, arm_a, arm_b, rng, n_boot=20000):
    per_run, all_d = [], []
    rows = []
    for r in runs:
        d = paired_diffs(r["reps"], arm_a, arm_b, metric)
        if len(d) < 2:
            continue
        n = len(d)
        sd = float(d.std(ddof=1))
        se = sd / math.sqrt(n)
        wins = int((d > 0).sum())
        per_run.append((float(d.mean()), se, n))
        all_d.append(d)
        rows.append({
            "path": os.path.basename(r["path"]), "seed": r["seed"],
            "n": n, "mean": float(d.mean()), "sd": sd, "se": se,
            "t": float(d.mean() / se) if se > 0 else float("nan"),
            "wins": wins, "p_sign": sign_p(wins, n),
        })
    if not rows:
        return None

    pooled = np.concatenate(all_d)
    n = len(pooled)
    sd = float(pooled.std(ddof=1))
    naive_se = sd / math.sqrt(n)
    wins = int((pooled > 0).sum())
    draws = rng.choice(pooled, size=(n_boot, n), replace=True).mean(axis=1)

    meta = meta_analyze(per_run)
    return {
        "metric": metric, "rows": rows, "meta": meta,
        "pooled": {
            "n": n, "mean": float(pooled.mean()), "sd": sd,
            "naive_se": naive_se,
            "t": float(pooled.mean() / naive_se) if naive_se > 0 else float("nan"),
            "dz": float(pooled.mean() / sd) if sd > 0 else float("nan"),
            "ci_lo": float(np.quantile(draws, 0.025)),
            "ci_hi": float(np.quantile(draws, 0.975)),
            "wins": wins, "p_sign": sign_p(wins, n),
        },
    }


def arm_means(runs, arm, metric):
    vals = []
    for r in runs:
        for rep in r["reps"]:
            v = rep.get(arm, {}).get(metric)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                vals.append(v)
    return np.asarray(vals, dtype=np.float64)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

ERROR_METRICS_PREFIXES = ("tvd",)


def is_error_metric(m):
    return m.startswith(ERROR_METRICS_PREFIXES) or "_err_" in m


def default_metrics(runs):
    seen, ordered = set(), []
    for rep in runs[0]["reps"]:
        for arm in ("ace", "severed"):
            for k in rep.get(arm, {}):
                if k == "seconds" or k in seen:
                    continue
                seen.add(k)
                ordered.append(k)
        break
    return ordered


def report(runs, metrics, arm_a, arm_b, rng, csv_path=None):
    a = runs[0]["args"]
    print()
    print(f"pooled {len(runs)} run(s): width={a.get('width')} "
          f"depth={a.get('depth')} lrc={a.get('lrc')} lrr={a.get('lrr')} "
          f"shots={a.get('shots')} sever_mode={a.get('sever_mode')} "
          f"bases={''.join(a.get('bases') or 'z')}")
    print(f"seeds: {[r['seed'] for r in runs]}")
    print(f"qrack_lib: {runs[0]['meta'].get('qrack_lib', '?')}")
    print()

    csv_rows = []
    for metric in metrics:
        res = analyze_metric(runs, metric, arm_a, arm_b, rng)
        if res is None:
            continue
        meta, pl = res["meta"], res["pooled"]
        direction = "lower is better" if is_error_metric(metric) else "higher is better"

        print(f"=== {metric}  ({arm_a} - {arm_b}; {direction}) ===")
        print("  run".ljust(22) + "seed".ljust(9) + "n".ljust(6)
              + "mean".ljust(11) + "se".ljust(10) + "t".ljust(8) + "sign")
        for row in res["rows"]:
            print("  " + row["path"][:20].ljust(20)
                  + str(row["seed"]).ljust(9)
                  + str(row["n"]).ljust(6)
                  + f"{row['mean']:+.4f}".ljust(11)
                  + f"{row['se']:.4f}".ljust(10)
                  + f"{row['t']:+.2f}".ljust(8)
                  + f"{row['wins']}/{row['n']}")

        if meta:
            fe_lo, fe_hi = meta["fe"] - 1.96 * meta["fe_se"], meta["fe"] + 1.96 * meta["fe_se"]
            re_lo, re_hi = meta["re"] - 1.96 * meta["re_se"], meta["re"] + 1.96 * meta["re_se"]
            print(f"  fixed-effect   {meta['fe']:+.4f} "
                  f"[{fe_lo:+.4f},{fe_hi:+.4f}]  "
                  f"z={meta['fe'] / meta['fe_se']:+.2f} "
                  f"p={norm_sf2(meta['fe'] / meta['fe_se']):.4f}")
            if meta["df"] > 0:
                print(f"  heterogeneity  Q={meta['Q']:.2f} df={meta['df']} "
                      f"p={meta['Q_p']:.4f}  I^2={meta['I2']:.0f}%  "
                      f"tau^2={meta['tau2']:.5f}")
            print(f"  random-effects {meta['re']:+.4f} "
                  f"[{re_lo:+.4f},{re_hi:+.4f}]  "
                  f"z={meta['re'] / meta['re_se']:+.2f} "
                  f"p={norm_sf2(meta['re'] / meta['re_se']):.4f}")
            if meta["df"] > 0 and meta["I2"] >= 50:
                print("  ^ runs are heterogeneous (I^2 >= 50%). Quote the "
                      "random-effects line, not the fixed-effect one.")

        print(f"  all reps       {pl['mean']:+.4f} "
              f"[{pl['ci_lo']:+.4f},{pl['ci_hi']:+.4f}] "
              f"n={pl['n']} t={pl['t']:+.2f} dz={pl['dz']:+.2f} "
              f"sign={pl['wins']}/{pl['n']} p={pl['p_sign']:.4f}")

        # power: reps needed for 80% power at the observed effect size
        if pl["sd"] > 0 and abs(pl["mean"]) > 0:
            need = math.ceil((2.8 * pl["sd"] / abs(pl["mean"])) ** 2)
            print(f"  reps for 80% power at the observed effect: {need} "
                  f"(have {pl['n']})")
        print()

        csv_rows.append({
            "metric": metric,
            "runs": len(res["rows"]),
            "n_reps": pl["n"],
            "fe": meta["fe"] if meta else "",
            "fe_se": meta["fe_se"] if meta else "",
            "I2": meta["I2"] if meta else "",
            "tau2": meta["tau2"] if meta else "",
            "re": meta["re"] if meta else "",
            "re_se": meta["re_se"] if meta else "",
            "Q_p": meta["Q_p"] if meta else "",
            "pooled_mean": pl["mean"],
            "pooled_ci_lo": pl["ci_lo"],
            "pooled_ci_hi": pl["ci_hi"],
            "sign_wins": pl["wins"],
            "sign_p": pl["p_sign"],
        })

    # absolute arm levels, for the bands that do not need a paired test
    print("=== arm levels (all reps pooled) ===")
    arms = ["ace", "severed", "ceiling", "null"]
    arms += sorted({k for r in runs for rep in r["reps"] for k in rep
                    if k.startswith("consensus_inst")})
    hdr = "metric".ljust(16) + "".join(x.ljust(20) for x in arms)
    print(hdr)
    print("-" * len(hdr))
    for metric in metrics:
        line = metric.ljust(16)
        any_val = False
        for arm in arms:
            v = arm_means(runs, arm, metric)
            if len(v) == 0:
                line += "--".ljust(20)
                continue
            any_val = True
            se = v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0.0
            line += f"{v.mean():+.4f}+-{se:.4f}".ljust(20)
        if any_val:
            print(line)
    print()

    if csv_path and csv_rows:
        import csv as _csv
        with open(csv_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"wrote {csv_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def do_run(args, extra):
    script = os.path.abspath(__file__)
    os.makedirs(args.outdir, exist_ok=True)
    paths = []
    for i in range(args.runs):
        seed = args.seed0 + i
        out = os.path.join(args.outdir, f"run_seed{seed}.json")
        if os.path.isfile(out) and args.skip_existing:
            print(f"[{i + 1}/{args.runs}] seed {seed}: exists, skipping")
            paths.append(out)
            continue
        # Each run is a SEPARATE process on purpose. Fresh Qrack/OpenCL
        # state per run is part of what between-run heterogeneity is
        # meant to capture; running them in-process would hide exactly
        # the variance this analysis exists to measure.
        cmd = [sys.executable, script, "bench",
               str(args.width), str(args.depth), str(args.lrc), str(args.lrr),
               "--reps", str(args.reps), "--seed", str(seed),
               "--json", out] + list(extra)
        print(f"[{i + 1}/{args.runs}] seed {seed}: {' '.join(cmd)}",
              file=sys.stderr)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"  run failed (exit {r.returncode}); continuing",
                  file=sys.stderr)
            continue
        paths.append(out)
    return paths



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

SUBCOMMANDS = ("bench", "run", "pool")

USAGE = """usage:
  nn_qab_adversarial.py bench WIDTH DEPTH [LRC] [LRR] [--reps N] [--bases zxy]
                              [--shots N] [--seed N] [--consensus N]
                              [--sever-mode correct|edetect|both] [--json OUT]
        One run. Arms: ace, severed, ceiling, null. Prints the per-arm table
        and the paired ace-minus-severed table.

  nn_qab_adversarial.py run --runs R --reps N WIDTH DEPTH [LRC] [LRR] [...]
        R independent seeded bench runs as separate processes, then pools
        them. Extra flags pass through to bench.

  nn_qab_adversarial.py pool FILES... [--metric M] [--csv OUT]
        Pool existing bench JSON files.

A single run at n=20 is NOT enough to compare ace against severed: two such
runs produced +0.1376 (t=+3.98, p=0.003) and -0.0407 (t=-1.59, p=0.82) on the
identical command, I^2 = 94%. Use `run --runs 4 --reps 50` and quote the
random-effects line."""


def run_main(argv):
    ap = argparse.ArgumentParser(prog="nn_qab_adversarial.py run")
    ap.add_argument("width", type=int)
    ap.add_argument("depth", type=int)
    ap.add_argument("lrc", type=int, nargs="?", default=4)
    ap.add_argument("lrr", type=int, nargs="?", default=4)
    ap.add_argument("--runs", type=int, default=4,
                    help="independent seeded runs; >=3 before between-run "
                         "heterogeneity means anything")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--outdir", type=str, default="runs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-pool", action="store_true")
    ap.add_argument("--metric", action="append", default=None)
    ap.add_argument("--csv", type=str, default=None)
    args, extra = ap.parse_known_args(argv)

    paths = do_run(args, extra)
    if args.no_pool or not paths:
        return 0
    runs = load_runs(paths)
    metrics = args.metric or default_metrics(runs)
    report(runs, metrics, "ace", "severed", np.random.default_rng(0), args.csv)
    return 0


def pool_main(argv):
    ap = argparse.ArgumentParser(prog="nn_qab_adversarial.py pool")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--metric", action="append", default=None)
    ap.add_argument("--arm-a", type=str, default="ace")
    ap.add_argument("--arm-b", type=str, default="severed")
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args(argv)

    files = []
    for f in args.files:
        files.extend(sorted(glob.glob(f)) or [f])
    runs = load_runs(files)
    metrics = args.metric or default_metrics(runs)
    report(runs, metrics, args.arm_a, args.arm_b,
           np.random.default_rng(0), args.csv)
    return 0


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    if argv[0] in SUBCOMMANDS:
        cmd, rest = argv[0], argv[1:]
    else:
        # Bare positional form kept working on purpose: the earlier runs in
        # this project were launched as `nn_qab_adversarial.py 14 12 3 2 ...`
        # and those command lines are quoted in the notes and the JSON.
        cmd, rest = "bench", argv
    return {"bench": bench_main, "run": run_main, "pool": pool_main}[cmd](rest)


if __name__ == "__main__":
    sys.exit(main())
