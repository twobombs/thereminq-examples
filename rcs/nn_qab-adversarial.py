# Nearest-neighbor RCS: adversarial-hardened measurement harness.
#
# Extends nn_qab.py (Dan Strano and Claude) with the controls needed to
# separate "approximate simulation" from "XEB spoofing" in the sense of
# Gao, Kalinowski, Chou, Lukin, Barak & Choi, arXiv:2112.01657.
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
# Usage:
#   python3 nn_qab_adversarial.py WIDTH DEPTH [LRC] [LRR] [--reps N]
#          [--shots N] [--seed N] [--no-null] [--consensus] [--json OUT]

import argparse
import json
import math
import random
import statistics
import sys
import time

from collections import Counter

import numpy as np
from pyqrack import QrackSimulator, QrackAceBackend


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
            sever_mode="correct"):
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
    del sim_ideal
    t_ideal = time.perf_counter() - t0

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
    z_ideal, zz_ideal = ideal_correlators(ideal_probs, all_pairs, all_singles)

    # ---- arms ----------------------------------------------------------
    arms = {}
    timings = {"ideal": t_ideal}

    for name, sever in (("ace", False), ("severed", True)):
        t = time.perf_counter()
        sim = make_backend(width, lrc, lrr, sever, sever_mode)
        run_circuit(sim, qc)
        counts = dict(Counter(sim.measure_shots(list(range(width)), shots)))
        timings[name] = time.perf_counter() - t
        arms[name] = (counts, sim)

    ceiling_counts = sample_ideal(ideal_probs, shots, rng)
    arms["ceiling"] = (ceiling_counts, None)
    timings["ceiling"] = 0.0

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
    for name, (counts, sim) in arms.items():
        m = score(ideal_probs, counts, shots, tail_mask,
                  denom_full, denom_tail, tail_center_sum, median_p, u_u)
        z_s, zz_s = sampled_correlators(counts, shots, all_pairs, all_singles)
        ns, nb = len(seam_pairs), len(bulk_pairs)
        m["zz_err_seam"] = (float(np.mean(np.abs(zz_s[:ns] - zz_ideal[:ns])))
                            if ns else float("nan"))
        m["zz_err_bulk"] = (float(np.mean(np.abs(zz_s[ns:] - zz_ideal[ns:])))
                            if nb else float("nan"))
        nss = len(seam_singles)
        m["z_err_seam"] = (float(np.mean(np.abs(z_s[:nss] - z_ideal[:nss])))
                           if nss else float("nan"))
        m["z_err_bulk"] = (float(np.mean(np.abs(z_s[nss:] - z_ideal[nss:])))
                           if len(bulk_singles) else float("nan"))
        m["seconds"] = timings.get(name, float("nan"))
        results[name] = m

    # null band: ace's own counts, scored against an unrelated circuit
    if do_null:
        results["null"] = score(
            decoy_probs, arms["ace"][0], shots, decoy_tail,
            decoy_denom, decoy_denom_tail, decoy_tail_center,
            decoy_median, u_u
        )
        results["null"]["seconds"] = 0.0

    if per_instance:
        for k, m in enumerate(per_instance):
            results[f"consensus_inst{k}"] = m
            results[f"consensus_inst{k}"]["seconds"] = 0.0

    meta = {
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


ROW_METRICS = [
    "xeb", "xeb_google", "xeb_tail", "hog",
    "hellinger", "tvd",
    "zz_err_seam", "zz_err_bulk", "z_err_seam", "z_err_bulk",
    "seconds",
]


def print_table(agg, meta, args):
    print()
    print(f"width={args.width} depth={args.depth} lrc={args.lrc} lrr={args.lrr} "
          f"shots={args.shots} reps={args.reps} sever_mode={args.sever_mode}")
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
    for m in ROW_METRICS:
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


def main():
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    if args.shots is None:
        args.shots = 1 << min(10, args.width + 2)
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

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "meta": meta,
                       "aggregate": agg, "reps": reps}, f, indent=2)
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
