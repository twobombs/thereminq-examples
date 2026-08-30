#!/usr/bin/env python3
# Nearest-neighbor RCS: Automatic circuit elision
# Original By Dan Strano and (Anthropic) Claude.
# https://github.com/vm6502q/pyqrack-examples/blob/main/rcs/nn_qab.py
#
# rights and license remain for the algo by Dan Strano et al
#
# modifications are done for runtime environment requirements
# within the ThereminQ container ecosystem ( parallel - sweep )
# 
# 
# ---------------------------------------------------------------------------
# MODES
# ---------------------------------------------------------------------------
#   (legacy)  nn_qab.py WIDTH DEPTH [LRC] [LRR] [SWAP_MODE]
#             One self-contained run: ACE, then exact reference, then stats.
#             Argument order and printed output are unchanged from the
#             original script, so existing shell wrappers keep working.
#
#   run       Same thing with named flags, plus an optional --seed.
#
#   ace       Pass A of a parallel sweep. Builds the circuit from a seed,
#             runs QrackAceBackend, saves the shot counts AND the circuit.
#             Patch sims are small (<= ~20 qubits at width 28), so this
#             belongs on a small card or on CPU, not on the one GPU that
#             can hold the reference state vector.
#
#   ideal     Pass B. Reloads the SAME circuit, runs the exact
#             QrackSimulator reference, computes XEB/HOG against the saved
#             counts. This is the stage that needs the big card, and it is
#             ~90% of the wall time.
#
#   merge     Collects per-seed results into one CSV, prints n/mean/stdev.
#
# Splitting ace from ideal keeps the big GPU busy with the only work that
# needs it instead of idling through the ACE run. Sweep workers coordinate
# through O_EXCL lock files: launch N copies of the same command and they
# divide the seed pool between them. Finished seeds are skipped, so an
# interrupted sweep resumes where it stopped.
#
#   nn_qab.py ace   --width 28 --depth 12 --lrc 3 --lrr 4 \
#                   --seeds 0-99 --out runs/w28
#   nn_qab.py ideal --seeds 0-99 --out runs/w28
#   nn_qab.py merge --out runs/w28 --csv w28.csv

import argparse
import csv
import ctypes
import gc
import json
import math
import os
import random
import statistics
import sys
import time

from collections import Counter

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

import numpy as np
from pyqrack import QrackSimulator, QrackAceBackend
from pyqrack.qrack_system import Qrack


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (row_len, col_len)


def bulk_to_boundary_ratio(sim):
    """Empirically measured bulk-to-boundary ratio for an already-
    constructed QrackAceBackend, via its own _unpack() (same source of
    truth used internally, rather than re-deriving the geometry rules by
    hand). Returns float('inf') if there are zero boundary qubits (a
    valid, if edge-case, config), rather than raising on the divide.
    """
    n = sim.num_qubits()
    boundary = sum(1 for lq in range(n) if len(sim._unpack(lq)) > 1)
    bulk = n - boundary
    return bulk / boundary if boundary else float("inf")


def make_ace(width, lrc, lrr):
    """Build the ACE backend and report its measured geometry.

    Built once and reused for the actual run. The original constructed a
    throwaway backend for the ratio (with an explicit is_torus=True, which
    is already the QrackAceBackend default) and then a second one after
    circuit construction. Geometrically the two were identical, but at
    width 28 the throwaway is not free, and reusing this instance
    guarantees the reported bulk_to_boundary describes the backend that
    was actually benchmarked.
    """
    sim = QrackAceBackend(
        width, long_range_columns=lrc, long_range_rows=lrr, is_torus=True
    )
    n_log = sim.num_qubits()
    boundary = sum(1 for lq in range(n_log) if len(sim._unpack(lq)) > 1)
    return sim, boundary, n_log - boundary, bulk_to_boundary_ratio(sim)


# ---------------------------------------------------------------------------
# Probability extraction
# ---------------------------------------------------------------------------

def out_probs_np(sim):
    """Zero-copy replacement for QrackSimulator.out_probs().

    The stock method allocates [0.0] * 2**n (a Python list), fills a ctypes
    buffer, then does list(probs) -- which materializes 2**n *distinct*
    Python float objects. At width 28 that is ~8.6 GB for the list alone,
    ~14 GB peak across the three copies, before np.asarray() makes a fourth.

    Here OutProbs writes straight into a numpy buffer of the matching real1
    dtype (float32 for the default fp32 build, float64 if QRACK_FPPOW=6).
    Peak for width 28: 1.07 GB instead of ~14 GB. Values are bit-identical.
    """
    n_pow = 1 << sim.num_qubits()
    if Qrack.fppow < 6:
        c_type, np_type = ctypes.c_float, np.float32
    else:
        c_type, np_type = ctypes.c_double, np.float64
    buf = np.empty(n_pow, dtype=np_type)
    Qrack.qrack_lib.OutProbs(sim.sid, buf.ctypes.data_as(ctypes.POINTER(c_type)))
    sim._throw_if_error()
    return buf


# ---------------------------------------------------------------------------
# Gate wrappers
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


# --- swap-family gates: native (QrackAceBackend.swap(), the _correct()-
# wrapped fast/sandwiched-shadow implementation) vs. cnot (manual 3-CNOT
# decomposition, going through the ordinary _cpauli-wrapped cx() path
# instead) -- two full sets of wrappers, selected between in bench_qrack()
# based on the swap_mode argument. Each _cnot variant mirrors the actual
# QrackAceBackend.swap()/iswap()/adjiswap() class-method gate sequences
# exactly, just using 3 explicit cx() calls in place of a single swap()
# call, so the two modes differ ONLY in how the swap itself is realized,
# not in the surrounding cz/s/adjs structure of the compound gates.

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


def run_circuit(sim, circ):
    for g in circ:
        g[0](sim, *g[1:])


GATE_SETS = {
    "swap": (
        swap_native, pswap_native, mswap_native, nswap_native,
        iswap_native, iiswap_native, cx, cy, cz, acx, acy, acz,
    ),
    "cnot": (
        swap_cnot, pswap_cnot, mswap_cnot, nswap_cnot,
        iswap_cnot, iiswap_cnot, cx, cy, cz, acx, acy, acz,
    ),
}

# Name -> callable, for rehydrating a serialized circuit.
_BY_NAME = {u.__name__: u}
for _set in GATE_SETS.values():
    for _g in _set:
        _BY_NAME[_g.__name__] = _g

SWAP_RATIO_THRESHOLD = 7.0


def resolve_swap_mode(ratio, swap_mode):
    if swap_mode not in ("auto", "swap", "cnot"):
        raise ValueError('swap_mode must be one of "auto", "swap", "cnot"')
    if swap_mode != "auto":
        return swap_mode
    return "cnot" if ratio >= SWAP_RATIO_THRESHOLD else "swap"


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------

def build_circuit(width, depth, two_bit_gates, seed=None):
    """The circuit the original bench_qrack() built, optionally pinned to
    a seed so a sweep's two passes can reproduce it exactly."""
    if seed is not None:
        random.seed(seed)

    lcv_range = range(width)
    row_len, col_len = factor_width(width)

    # Nearest-neighbor couplers:
    gate_sequence = [0, 3, 2, 1, 2, 1, 0, 3]
    qc = []

    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
            # Keep it Haar-random towards the poles:
            th = math.asin(th / math.pi)
            qc.append((u, i, th, ph, lm))

        # Nearest-neighbor couplers:
        ############################
        gate = gate_sequence.pop(0)
        gate_sequence.append(gate)
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

    return qc


def serialize(qc):
    return [[g[0].__name__] + list(g[1:]) for g in qc]


def deserialize(rows):
    return [tuple([_BY_NAME[r[0]]] + list(r[1:])) for r in rows]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calc_stats(ideal_probs, counts, shots):
    """Vectorized XEB / HOG.

    Identical arithmetic to the original scalar loop (agrees to ~1e-16), but
    the pure-Python version is O(2**n) interpreted iterations plus
    statistics.median(), which sorts 2**n numpy scalars into a Python list.
    At width 28 that is roughly 7-8 minutes and ~9 GB of transient list per
    call; here it is a couple of seconds.

    ideal_probs is consumed in place (centered) to avoid a second 2.1 GB
    temporary -- pass a copy if the caller still needs it.
    """
    ideal = np.asarray(ideal_probs, dtype=np.float64)

    threshold = float(np.median(ideal))
    u_u = float(ideal.mean())

    # Sparse shot histogram -> dense probability vector.
    idx = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
    val = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
    patch = np.zeros_like(ideal)
    patch[idx] = val / shots

    hog_prob = float(patch[ideal > threshold].sum())

    ideal -= u_u                                   # in place: no extra copy
    denom = float(ideal @ ideal)
    numer = float(ideal @ patch) - u_u * float(ideal.sum())

    xeb = numer / denom
    return xeb, hog_prob


# ---------------------------------------------------------------------------
# Mode: single self-contained run
# ---------------------------------------------------------------------------

def bench_qrack(width, depth, lrc=4, lrr=4, swap_mode="auto", seed=None):
    all_bits = list(range(width))
    shots = 1 << min(10, width + 2)

    t_circ = time.perf_counter()

    sim, boundary_qubits, bulk_qubits, ratio = make_ace(width, lrc, lrr)
    resolved_swap_mode = resolve_swap_mode(ratio, swap_mode)

    qc = build_circuit(width, depth, GATE_SETS[resolved_swap_mode], seed)

    # -----------------------------------------------------------------------
    # Method: QrackAceBackend
    # -----------------------------------------------------------------------
    run_circuit(sim, qc)
    ace_counts = dict(Counter(sim.measure_shots(all_bits, shots)))
    del sim
    gc.collect()

    t_ace = time.perf_counter()
    print(f"ace_seconds: {t_ace - t_circ:.4f}")

    # -----------------------------------------------------------------------
    # Ideal ground truth via QrackSimulator
    # -----------------------------------------------------------------------
    sim_ideal = QrackSimulator(width)
    run_circuit(sim_ideal, qc)
    ideal_probs = out_probs_np(sim_ideal).astype(np.float64)
    del sim_ideal
    gc.collect()

    t_ideal = time.perf_counter()
    print(f"ideal_seconds: {t_ideal - t_ace:.4f}")

    xeb_ace, hog_ace = calc_stats(ideal_probs, ace_counts, shots)
    del ideal_probs
    gc.collect()

    t_stats = time.perf_counter()
    print(f"stats_seconds: {t_stats - t_ideal:.4f}")

    return {
        "width":              width,
        "depth":              depth,
        "long_range_columns": lrc,
        "long_range_rows":    lrr,
        "boundary_qubits":    boundary_qubits,
        "bulk_qubits":        bulk_qubits,
        "bulk_to_boundary":   ratio,
        "swap_mode":          swap_mode,
        "resolved_swap_mode": resolved_swap_mode,
        "seed":               seed,
        "qrack_lib":          QRACK_LIB_SOURCE,
        "xeb_ace":            xeb_ace,
        "hog_ace":            hog_ace,
    }


def run_single(args):
    result = bench_qrack(
        args.width, args.depth, args.lrc, args.lrr, args.swap_mode, args.seed
    )
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0


# ---------------------------------------------------------------------------
# Sweep: worker coordination
# ---------------------------------------------------------------------------

def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def claim(path):
    """Atomically claim a unit of work. False if another worker has it."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)          # atomic: readers never see a partial file


# ---------------------------------------------------------------------------
# Sweep pass A: ACE
# ---------------------------------------------------------------------------

def run_ace(args):
    ace_dir = os.path.join(args.out, "ace")
    os.makedirs(ace_dir, exist_ok=True)

    for seed in parse_seeds(args.seeds):
        done = os.path.join(ace_dir, f"{seed:06d}.json")
        lock = done + ".lock"
        if os.path.exists(done) or not claim(lock):
            continue

        try:
            t0 = time.perf_counter()
            width, depth = args.width, args.depth
            shots = 1 << min(10, width + 2)

            sim, boundary, bulk, ratio = make_ace(width, args.lrc, args.lrr)
            resolved = resolve_swap_mode(ratio, args.swap_mode)

            qc = build_circuit(width, depth, GATE_SETS[resolved], seed)
            run_circuit(sim, qc)
            counts = dict(Counter(sim.measure_shots(list(range(width)), shots)))
            del sim
            gc.collect()

            write_json(done, {
                "seed": seed,
                "width": width,
                "depth": depth,
                "long_range_columns": args.lrc,
                "long_range_rows": args.lrr,
                "boundary_qubits": boundary,
                "bulk_qubits": bulk,
                "bulk_to_boundary": ratio,
                "swap_mode": args.swap_mode,
                "resolved_swap_mode": resolved,
                "shots": shots,
                "ace_seconds": time.perf_counter() - t0,
                "counts": {str(k): v for k, v in counts.items()},
                "circuit": serialize(qc),
            })
            print(f"ace seed={seed} {time.perf_counter() - t0:.1f}s", flush=True)
        except Exception as e:
            release(lock)
            print(f"ace seed={seed} FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            if args.stop_on_error:
                raise

    return 0


# ---------------------------------------------------------------------------
# Sweep pass B: exact reference + statistics
# ---------------------------------------------------------------------------

def run_ideal(args):
    ace_dir = os.path.join(args.out, "ace")
    xeb_dir = os.path.join(args.out, "xeb")
    os.makedirs(xeb_dir, exist_ok=True)

    for seed in parse_seeds(args.seeds):
        src = os.path.join(ace_dir, f"{seed:06d}.json")
        if not os.path.exists(src):
            continue                      # pass A hasn't reached this seed yet
        done = os.path.join(xeb_dir, f"{seed:06d}.json")
        lock = done + ".lock"
        if os.path.exists(done) or not claim(lock):
            continue

        try:
            t0 = time.perf_counter()
            with open(src) as f:
                rec = json.load(f)

            qc = deserialize(rec["circuit"])
            counts = {int(k): v for k, v in rec["counts"].items()}

            sim_ideal = QrackSimulator(rec["width"])
            run_circuit(sim_ideal, qc)
            ideal_probs = out_probs_np(sim_ideal).astype(np.float64)
            del sim_ideal
            gc.collect()

            t_ideal = time.perf_counter()
            xeb, hog = calc_stats(ideal_probs, counts, rec["shots"])
            del ideal_probs
            gc.collect()

            out = {k: v for k, v in rec.items()
                   if k not in ("counts", "circuit")}
            out.update({
                "xeb_ace": xeb,
                "hog_ace": hog,
                "ideal_seconds": t_ideal - t0,
                "stats_seconds": time.perf_counter() - t_ideal,
                "qrack_lib": QRACK_LIB_SOURCE,
            })
            write_json(done, out)
            print(f"ideal seed={seed} xeb={xeb:.6f} "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
        except Exception as e:
            release(lock)
            print(f"ideal seed={seed} FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            if args.stop_on_error:
                raise

    return 0


# ---------------------------------------------------------------------------
# Sweep: merge
# ---------------------------------------------------------------------------

FIELDS = [
    "seed", "width", "depth", "long_range_columns", "long_range_rows",
    "boundary_qubits", "bulk_qubits", "bulk_to_boundary",
    "swap_mode", "resolved_swap_mode", "shots",
    "xeb_ace", "hog_ace",
    "ace_seconds", "ideal_seconds", "stats_seconds", "qrack_lib",
]


def run_merge(args):
    xeb_dir = os.path.join(args.out, "xeb")
    rows = []
    for name in sorted(os.listdir(xeb_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(xeb_dir, name)) as f:
            rows.append(json.load(f))

    if not rows:
        print("no results yet", file=sys.stderr)
        return 1

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    xebs = [r["xeb_ace"] for r in rows]
    print(f"n={len(xebs)}  mean={statistics.mean(xebs):.10f}", end="")
    if len(xebs) > 1:
        print(f"  stdev={statistics.stdev(xebs):.10f}", end="")
    print(f"  -> {args.csv}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

SUBCOMMANDS = ("run", "ace", "ideal", "merge")


def build_parser():
    p = argparse.ArgumentParser(
        prog="nn_qab.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Nearest-neighbor RCS with automatic circuit elision.",
        epilog=(
            "legacy form (unchanged):\n"
            "  nn_qab.py WIDTH DEPTH [LRC=4] [LRR=4] [SWAP_MODE=auto]\n"
        ),
    )
    sub = p.add_subparsers(dest="mode", required=True)

    def geom(sp):
        sp.add_argument("--width", type=int, required=True)
        sp.add_argument("--depth", type=int, required=True)
        sp.add_argument("--lrc", type=int, default=4,
                        help="long_range_columns (default 4)")
        sp.add_argument("--lrr", type=int, default=4,
                        help="long_range_rows (default 4)")
        sp.add_argument("--swap-mode", default="auto",
                        choices=("auto", "swap", "cnot"))

    r = sub.add_parser("run", help="one self-contained run")
    geom(r)
    r.add_argument("--seed", type=int, default=None,
                   help="pin the circuit (default: unseeded)")
    r.set_defaults(func=run_single)

    a = sub.add_parser("ace", help="sweep pass A: ACE + shot counts")
    geom(a)
    a.add_argument("--seeds", required=True, help="e.g. 0-99 or 0,5,7-9")
    a.add_argument("--out", required=True)
    a.add_argument("--stop-on-error", action="store_true")
    a.set_defaults(func=run_ace)

    b = sub.add_parser("ideal", help="sweep pass B: exact reference + XEB")
    b.add_argument("--seeds", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--stop-on-error", action="store_true")
    b.set_defaults(func=run_ideal)

    m = sub.add_parser("merge", help="collect per-seed results into a CSV")
    m.add_argument("--out", required=True)
    m.add_argument("--csv", required=True)
    m.set_defaults(func=run_merge)

    return p


def main():
    argv = sys.argv[1:]

    # Legacy positional form, kept so existing wrappers don't have to change.
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        if len(argv) < 2:
            raise RuntimeError(
                "Usage: python3 nn_qab.py [width] [depth] "
                "[long_range_columns=4] [long_range_rows=4] "
                "[swap_mode=auto|swap|cnot]\n"
                "   or: python3 nn_qab.py {run,ace,ideal,merge} --help"
            )
        args = argparse.Namespace(
            width=int(argv[0]),
            depth=int(argv[1]),
            lrc=int(argv[2]) if len(argv) > 2 else 4,
            lrr=int(argv[3]) if len(argv) > 3 else 4,
            swap_mode=argv[4] if len(argv) > 4 else "auto",
            seed=None,
        )
        return run_single(args)

    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
