# sdrp_zne_batch.py
#
# SDRP-as-noise-knob ZNE over Dan's patch geometry series (the "noise-aware
# transpilation data collection" sheet layout), using PyQrack QrackAceBackend
# and Mitiq inference factories.
#
# Method:
#   * Circuit family mirrors pyqrack-examples qem/mitiq_nn_hamming_weight.py
#     and rcs/expressive_nearest_neighbor: nearest-neighbor coupler RCS on
#     factor_width(width) grid, gateSequence [0,3,2,1,2,1,0,3], the 12
#     two-qubit gate set, uniform [0, 2*pi) u3 angles.
#   * Experiment backend: QrackAceBackend(width, long_range_columns,
#     long_range_rows) with set_sdrp(sdrp). Geometry per series is taken
#     straight from the sheet: (lrc, lrr, width) = (2,2,10), (3,2,14),
#     (4,3,27), (5,2,22), (6,2,26); B-to-B ratio = 2 * lrc.
#   * Ideal reference: full QrackSimulator at sdrp = 0, queried sparsely via
#     prob_perm on sampled bitstrings only (no out_probs DMA bursts).
#   * Observable: shot-based linear XEB per circuit,
#         F = 2^n * mean_i P_ideal(x_i) - 1
#     over shots x_i drawn from the SDRP'd patch simulation.
#   * ZNE: the SAME seeded circuit ensemble is replayed at every SDRP point
#     (paired design; circuit-to-circuit fluctuation cancels in the trend).
#     Series-mean XEB per SDRP point is extrapolated to sdrp -> 0 with
#     Linear, Richardson, and Exp factories via run_classical. Ensemble
#     averaging over random circuits also smooths the per-circuit SDRP
#     "staircase" response, which is what makes the extrapolation viable.
#   * Validation: optional direct run at sdrp = 0 (patches still on), the
#     exact target the extrapolation should reproduce.
#
# SDRP points default to g * {1/3, 2/3, 1} with g = (1 - 1/sqrt(2)) / 2,
# the sheet's "golden value" (~0.146447), i.e. sdrp ~ {0.049, 0.098, 0.146}.
#
# Usage:
#   python3 sdrp_zne_batch.py [--series 10,14,22] [--circuits 100]
#       [--depth 12] [--shots 1024] [--seed 42] [--cpu] [--no-exact]
#       [--scales 0.3333,0.6667,1.0] [--out sdrp_zne]
#
# Output: one CSV per series (per-circuit XEB at each SDRP point, sheet-like
# columns) plus a summary block per series on stdout.
#
# Mitiq is under GPL 3.0; as a combined work, treat this file as GPL 3.0.
# Dependencies: pyqrack, mitiq, numpy. US-ASCII only. CPython 3.9+.

import argparse
import csv
import math
import random
import sys

import numpy as np

from pyqrack import QrackSimulator, QrackAceBackend
from mitiq.zne.inference import LinearFactory, RichardsonFactory, ExpFactory

GOLDEN_SDRP = (1.0 - 1.0 / math.sqrt(2.0)) / 2.0

# Sheet geometry: width -> (long_range_columns, long_range_rows, ratio)
SERIES_TABLE = {
    10: (2, 2, 4),
    14: (3, 2, 6),
    27: (4, 3, 8),
    22: (5, 2, 10),
    26: (6, 2, 12),
}

TWO_BIT_GATES = ("swap", "pswap", "mswap", "nswap", "iswap", "iiswap",
                 "cx", "cy", "cz", "acx", "acy", "acz")


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        raise ValueError("cannot simulate prime-number width")
    return (row_len, col_len)


def build_circuit(width, depth, rng):
    # Mirrors the pyqrack-examples nearest-neighbor RCS generator.
    gate_sequence = [0, 3, 2, 1, 2, 1, 0, 3]
    row_len, col_len = factor_width(width)
    gates = []
    for _ in range(depth):
        for i in range(width):
            th = rng.uniform(0.0, 2.0 * math.pi)
            ph = rng.uniform(0.0, 2.0 * math.pi)
            lm = rng.uniform(0.0, 2.0 * math.pi)
            gates.append(("u", i, th, ph, lm))
        gate = gate_sequence.pop(0)
        gate_sequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row + (1 if (gate & 2) else -1)
                temp_col = col + (1 if (gate & 1) else 0)
                if temp_row < 0:
                    temp_row += row_len
                if temp_col < 0:
                    temp_col += col_len
                if temp_row >= row_len:
                    temp_row -= row_len
                if temp_col >= col_len:
                    temp_col -= col_len
                b1 = col * row_len + row
                b2 = temp_col * row_len + temp_row
                if (b1 >= width) or (b2 >= width):
                    continue
                gates.append(("g2", rng.choice(TWO_BIT_GATES), b1, b2))
    return gates


# ----------------------------------------------------------------------------
# Gate replay adapters (QrackAceBackend and QrackSimulator differ on
# controlled-gate call conventions).
# ----------------------------------------------------------------------------


def apply_two_bit_ace(sim, name, q1, q2):
    if name == "swap":
        sim.swap(q1, q2)
    elif name == "iswap":
        sim.iswap(q1, q2)
    elif name == "iiswap":
        sim.adjiswap(q1, q2)
    elif name == "pswap":
        sim.mcz(q1, q2)
        sim.swap(q1, q2)
    elif name == "mswap":
        sim.swap(q1, q2)
        sim.mcz(q1, q2)
    elif name == "nswap":
        sim.mcz(q1, q2)
        sim.swap(q1, q2)
        sim.mcz(q1, q2)
    elif name == "cx":
        sim.mcx(q1, q2)
    elif name == "cy":
        sim.mcy(q1, q2)
    elif name == "cz":
        sim.mcz(q1, q2)
    elif name == "acx":
        sim.macx(q1, q2)
    elif name == "acy":
        sim.macy(q1, q2)
    elif name == "acz":
        sim.macz(q1, q2)
    else:
        raise ValueError("unknown two-bit gate: " + name)


def apply_two_bit_sim(sim, name, q1, q2):
    if name == "swap":
        sim.swap(q1, q2)
    elif name == "iswap":
        sim.iswap(q1, q2)
    elif name == "iiswap":
        sim.adjiswap(q1, q2)
    elif name == "pswap":
        sim.mcz([q1], q2)
        sim.swap(q1, q2)
    elif name == "mswap":
        sim.swap(q1, q2)
        sim.mcz([q1], q2)
    elif name == "nswap":
        sim.mcz([q1], q2)
        sim.swap(q1, q2)
        sim.mcz([q1], q2)
    elif name == "cx":
        sim.mcx([q1], q2)
    elif name == "cy":
        sim.mcy([q1], q2)
    elif name == "cz":
        sim.mcz([q1], q2)
    elif name == "acx":
        sim.macx([q1], q2)
    elif name == "acy":
        sim.macy([q1], q2)
    elif name == "acz":
        sim.macz([q1], q2)
    else:
        raise ValueError("unknown two-bit gate: " + name)


def replay(sim, gates, is_ace):
    for g in gates:
        if g[0] == "u":
            sim.u(g[1], g[2], g[3], g[4])
        elif is_ace:
            apply_two_bit_ace(sim, g[1], g[2], g[3])
        else:
            apply_two_bit_sim(sim, g[1], g[2], g[3])


# ----------------------------------------------------------------------------
# Per-circuit XEB at one SDRP point
# ----------------------------------------------------------------------------


def xeb_at_sdrp(gates, width, lrc, lrr, sdrp, shots, ideal_sim, is_gpu):
    ace = QrackAceBackend(width, long_range_columns=lrc,
                          long_range_rows=lrr, is_gpu=is_gpu)
    if sdrp > 0.0:
        ace.set_sdrp(sdrp)
    replay(ace, gates, is_ace=True)
    samples = ace.measure_shots(list(range(width)), shots)
    n_pow = float(1 << width)
    qubits = list(range(width))
    prob_cache = {}
    total = 0.0
    for s in samples:
        p = prob_cache.get(s)
        if p is None:
            bits = [bool((s >> i) & 1) for i in range(width)]
            p = ideal_sim.prob_perm(qubits, bits)
            prob_cache[s] = p
        total += p
    return n_pow * (total / len(samples)) - 1.0


# ----------------------------------------------------------------------------
# Series driver
# ----------------------------------------------------------------------------


def run_series(width, depth, n_circuits, shots, scales, base_sdrp,
               with_exact, seed, is_gpu, out_prefix):
    lrc, lrr, ratio = SERIES_TABLE[width]
    sdrp_points = [base_sdrp * s for s in scales]
    labels = ["sdrp=%.6f" % p for p in sdrp_points]
    if with_exact:
        labels.append("sdrp=0 (exact)")

    print("=" * 72)
    print("series: width=%d lrc=%d lrr=%d ratio=%d depth=%d"
          % (width, lrc, lrr, ratio, depth))
    print("circuits=%d shots=%d points=%s"
          % (n_circuits, shots,
             ", ".join("%.6f" % p for p in sdrp_points)
             + (", 0 (exact)" if with_exact else "")))

    columns = [[] for _ in labels]
    csv_path = "%s_w%d_r%d.csv" % (out_prefix, width, ratio)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["circuit"] + labels)
        for c in range(n_circuits):
            rng = random.Random((seed * 1000003 + width * 1009 + c) & 0x7FFFFFFF)
            gates = build_circuit(width, depth, rng)
            ideal = QrackSimulator(width)
            replay(ideal, gates, is_ace=False)
            row = [c]
            for k, sdrp in enumerate(sdrp_points):
                v = xeb_at_sdrp(gates, width, lrc, lrr, sdrp, shots,
                                ideal, is_gpu)
                columns[k].append(v)
                row.append("%.10f" % v)
            if with_exact:
                v = xeb_at_sdrp(gates, width, lrc, lrr, 0.0, shots,
                                ideal, is_gpu)
                columns[len(sdrp_points)].append(v)
                row.append("%.10f" % v)
            ideal.reset_all()
            writer.writerow(row)
            f.flush()
            print("  circuit %3d/%d: %s"
                  % (c + 1, n_circuits,
                     "  ".join("%+.4f" % columns[k][-1]
                               for k in range(len(labels)))))

    # Summary statistics per point
    print("-" * 72)
    means = []
    for k, lab in enumerate(labels):
        a = np.array(columns[k])
        sem = a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else 0.0
        means.append(float(a.mean()))
        print("  %-18s mean=%+.6f  median=%+.6f  sem=%.6f  n=%d"
              % (lab, a.mean(), np.median(a), sem, len(a)))

    # ZNE on series means: executor is a lookup keyed by scale factor.
    mean_by_scale = dict(zip(scales, means[:len(scales)]))

    def executor(scale_factor):
        return mean_by_scale[scale_factor]

    print("  ZNE extrapolation of series-mean XEB to sdrp -> 0:")
    factories = [
        ("Linear", LinearFactory(scale_factors=list(scales))),
        ("Richardson", RichardsonFactory(scale_factors=list(scales))),
        ("Exp", ExpFactory(scale_factors=list(scales), asymptote=None)),
    ]
    exact_mean = means[len(scales)] if with_exact else None
    for name, factory in factories:
        try:
            factory.run_classical(executor)
            est = factory.reduce()
            if exact_mean is None:
                print("    %-10s estimate=%+.6f" % (name, est))
            else:
                print("    %-10s estimate=%+.6f  target=%+.6f  abs_err=%.6f"
                      % (name, est, exact_mean, abs(est - exact_mean)))
        except Exception as exc:
            print("    %-10s FAILED: %s" % (name, exc))
    print("  wrote %s" % csv_path)
    return csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="10,14,27,22,26",
                    help="comma-separated widths from the sheet table")
    ap.add_argument("--circuits", type=int, default=100)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scales", default="0.333333,0.666667,1.0",
                    help="scale factors applied to base SDRP")
    ap.add_argument("--base-sdrp", type=float, default=GOLDEN_SDRP)
    ap.add_argument("--cpu", action="store_true",
                    help="run QrackAceBackend with is_gpu=False")
    ap.add_argument("--no-exact", action="store_true",
                    help="skip the direct sdrp=0 validation column")
    ap.add_argument("--out", default="sdrp_zne")
    args = ap.parse_args()

    widths = [int(w) for w in args.series.split(",")]
    scales = [float(s) for s in args.scales.split(",")]
    for w in widths:
        if w not in SERIES_TABLE:
            raise ValueError("width %d not in sheet series table %s"
                             % (w, sorted(SERIES_TABLE)))

    print("base SDRP (golden value): %.6f" % args.base_sdrp)
    for w in widths:
        run_series(w, args.depth, args.circuits, args.shots, scales,
                   args.base_sdrp, not args.no_exact, args.seed,
                   not args.cpu, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
