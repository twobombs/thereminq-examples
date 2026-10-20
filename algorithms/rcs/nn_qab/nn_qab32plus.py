#!/usr/bin/env python3
# Nearest-neighbor RCS: Automatic circuit elision
# Original By Dan Strano and (Anthropic) Claude.
# https://github.com/vm6502q/pyqrack-examples/blob/main/rcs/nn_qab.py
#
# rights and license remain for this code for Dan Strano et al
# modifications are done for environment variable requirements
# within the ThereminQ container ecosystem
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
#             Patch sims are small (<= 21 qubits at width 28, <= 26 at
#             width 35, each including its detection ancilla), so this
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
# All modes, including the legacy form, accept --device N to pin Qrack to a
# single OpenCL device (default 0). See the "Device selection" block below.
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

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
# Qrack binds its device configuration when the shared library is loaded, at
# `from pyqrack import ...` below -- long before argparse runs. So the device
# has to be settled here, by environment variable, or not at all.
#
# Qrack uses ALL detected OpenCL devices by default. On a mixed box that means
# the ACE patch simulators get scattered across cards, and a single bad device
# can take down a run that had no business touching it. Default to pinning
# everything to device 0, which is the device Qrack lists first in the
# "OpenCL device #n:" block it prints at startup.
#
# Precedence: --device N  >  $QRACK_DEVICE  >  0. Any of the three underlying
# variables you set yourself is left alone, so an explicit multi-device list
# (e.g. QRACK_QPAGER_DEVICES=4.0,4.1 for segment-level load balancing) still
# works as written.

_DEVICE_VARS = (
    "QRACK_OCL_DEFAULT_DEVICE",
    "QRACK_QPAGER_DEVICES",
    "QRACK_QUNITMULTI_DEVICES",
)


def _take_early_arg(name):
    """Pull `--name VALUE` / `--name=VALUE` out of sys.argv before argparse.

    Removed from argv entirely so neither the subcommand parser nor the
    legacy positional form has to know about it.
    """
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i] == name and (i + 1) < len(argv):
            value = argv[i + 1]
            del argv[i:i + 2]
            return value
        if argv[i].startswith(name + "="):
            value = argv[i].split("=", 1)[1]
            del argv[i]
            return value
    return None


# A comma list (--device 1,2,3,4,5) pages one simulator across several
# devices: the list goes to QRACK_QPAGER_DEVICES / QRACK_QUNITMULTI_DEVICES,
# and its first entry becomes QRACK_OCL_DEFAULT_DEVICE, which takes a single
# device only.

def _device_env(spec):
    first = spec.split(",")[0].split(".")[0]
    return {
        "QRACK_OCL_DEFAULT_DEVICE": first,
        "QRACK_QPAGER_DEVICES":     spec,
        "QRACK_QUNITMULTI_DEVICES": spec,
    }


_device_flag = _take_early_arg("--device")
if _device_flag is not None:
    QRACK_DEVICE = _device_flag
    for _var, _val in _device_env(QRACK_DEVICE).items():
        os.environ[_var] = _val                  # flag overrides the env
else:
    QRACK_DEVICE = os.environ.get("QRACK_DEVICE", "0")
    for _var, _val in _device_env(QRACK_DEVICE).items():
        os.environ.setdefault(_var, _val)

QRACK_DEVICE_ENV = {_var: os.environ[_var] for _var in _DEVICE_VARS}

print(
    "qrack_device: "
    + ", ".join(f"{k}={v}" for k, v in QRACK_DEVICE_ENV.items()),
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

# ---------------------------------------------------------------------------
# Reference-simulator engines
# ---------------------------------------------------------------------------
# Constructor kwargs for the exact/approximate reference. The ACE backend is
# untouched by this -- only the QrackSimulator that provides ground truth.
#
# MEASURED on this circuit family (depth-12 nearest-neighbour RCS, CPU):
#
#     width   statevector   qbdd      ratio
#        12        0.01 s    2.56 s     256x
#        14        0.04 s    8.65 s     216x
#        16        0.19 s   55.95 s     294x
#
# QBDD is not merely slower, it scales worse: 3.4x then 6.5x per +2 qubits
# against the state vector's steady 4x (= 2**2, as expected). It also does
# not reproduce state-vector amplitudes -- up to 6.3% relative error per
# permutation, median 0.5%, unchanged by QRACK_QBDT_SEPARABILITY_THRESHOLD=0,
# so it is fp32 accumulation through the tree rather than branch rounding.
#
# That is what a decision diagram does on a circuit built to have no
# structure: RCS at depth 12 is maximally entangling by construction, which
# is the whole point of it as a benchmark. QBDD, sparse truncation and MPS
# all exploit structure this circuit deliberately destroys.
#
# The flag is here so the result can be re-tested on GPU, where the constant
# factors differ even if the scaling exponent should not.
ENGINES = {
    "statevector": {},
    "qbdd":        {"is_binary_decision_tree": True},
    "sparse":      {"is_sparse": True},
    "stabilizer":  {"is_stabilizer_hybrid": True},
    "cpu":         {"is_gpu": False},
}


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
    """XEB / HOG in one pass over the probability vector, no extra copies.

    Identical arithmetic to the original scalar loop (agrees to ~1e-16), but
    the naive vectorization materializes three more 2**n arrays -- a float64
    cast, a dense shot vector, and a centered copy -- which is what puts a
    ~5x multiplier on the memory ceiling. Past width 32 that multiplier is
    the thing that stops you, not the state vector itself.

    The dense shot vector is avoidable because counts has at most `shots`
    nonzero entries. Writing q_b for the shot frequency and mu = 1/N:

        numer = SUM_all (p-mu)(q-mu)
              = SUM_all (p-mu) q  -  mu * SUM_all (p-mu)
              = SUM_sampled (p-mu) q  -  mu * (SUM_all p  -  1)

    since q vanishes off the sample and sums to 1. Only the <= 1024 sampled
    amplitudes are ever touched. Likewise

        denom = SUM (p-mu)^2 = SUM p^2 - 2 mu SUM p + N mu^2

    needs two reductions, both accumulated in float64 regardless of the
    buffer dtype, and no centered copy. HOG is a lookup over the same
    sampled indices.

    Peak memory is therefore the probability vector alone: 64 GiB at width
    34 in fp32, where the previous path wanted well over 300 GiB.

    ideal_probs is read, and permuted in place by the median if it is
    writable -- pass a copy if the caller still needs the original order.
    """
    p = np.asarray(ideal_probs)
    n_pow = p.size
    mu = 1.0 / n_pow

    # Gather the sampled amplitudes FIRST. np.median(overwrite_input=True)
    # partitions in place, which permutes the array and destroys the
    # bitstring -> index correspondence; anything positional has to happen
    # before it.
    idx = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
    q = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
    q /= shots
    p_s = p[idx].astype(np.float64)

    total = float(p.sum(dtype=np.float64))
    sum_sq = float(np.einsum("i,i->", p, p, dtype=np.float64))
    denom = sum_sq - 2.0 * mu * total + n_pow * mu * mu

    numer = float(((p_s - mu) * q).sum()) - mu * (total - 1.0)

    # Last: overwrite_input spares numpy a full copy to partition.
    threshold = float(np.median(p, overwrite_input=p.flags.writeable))
    hog_prob = float(q[p_s > threshold].sum())

    return numer / denom, hog_prob


# ---------------------------------------------------------------------------
# Amplitude-only statistics (widths where 2**n probabilities won't fit)
# ---------------------------------------------------------------------------
# calc_stats() needs the whole probability vector for SUM p^2 (the XEB
# denominator) and for the median (the HOG threshold). Past ~width 34 that
# vector -- on top of the state vector it is read from -- no longer fits in
# 320 GB of host RAM, and at width 36 the state vector alone is 512 GiB.
#
# The sampled amplitudes, though, are only <= 1024 numbers, and Qrack can
# hand them over without materializing anything: PermutationProb with a
# full mask iterates over the 2**(n - popcount(mask)) = 1 unmasked
# subspace, so each lookup is O(1). Checked bit-identical to out_probs()
# at widths 20 and 24, ~10 us per lookup.
#
# The two full-vector quantities are then replaced by their Porter-Thomas
# values, which is what "linear XEB" is:
#
#     SUM p^2  ->  2/N      so  denom -> 1/N,  xeb -> N * SUM_s q p  -  1
#     median   ->  ln2/N
#
# This is the standard estimator, but it is NOT the same estimator as
# calc_stats(). To keep the series comparable across the switch, exact
# mode also computes the linear value from the same sampled amplitudes and
# records both (xeb_linear / hog_linear) -- the overlap widths calibrate
# the change of estimator instead of hiding it.

def sampled_probs(sim, counts):
    """Ideal probabilities of the sampled bitstrings only. Index bit i is
    qubit i, the same convention as measure_shots() and out_probs()."""
    n = sim.num_qubits()
    qubits = list(range(n))
    idx = list(counts.keys())
    return idx, np.array(
        [sim.prob_perm(qubits, [bool((k >> i) & 1) for i in range(n)])
         for k in idx],
        dtype=np.float64,
    )


def calc_stats_linear(p_s, counts_in_order, shots, width):
    """Linear (Porter-Thomas) XEB and HOG from sampled amplitudes only.

    p_s and counts_in_order must be aligned: p_s[i] is the ideal
    probability of the bitstring that was observed counts_in_order[i] times.
    """
    n_pow = float(1 << width)
    q = np.asarray(counts_in_order, dtype=np.float64) / shots
    p_s = np.asarray(p_s, dtype=np.float64)
    xeb = n_pow * float((q * p_s).sum()) - 1.0
    hog = float(q[p_s > math.log(2.0) / n_pow].sum())
    return xeb, hog


# ---------------------------------------------------------------------------
# Host-memory guard
# ---------------------------------------------------------------------------
# At width 34-35 an over-subscribed reference doesn't fail politely: it
# drives the box into swap or wakes the OOM killer, and on a shared host
# that can take other work down with it. Estimate first, refuse early.
#
# Bytes per amplitude, fp32 build (doubled for QRACK_FPPOW=6):
#     state vector        8   (complex64)
#     probability vector  4   (exact stats only)
# The GPU engines also stage the ket through host memory on OutProbs, so
# exact mode is budgeted at 12 B/amp regardless of engine. Linear mode on a
# GPU engine keeps the state on the card and needs no large host buffer.
# These are estimates; --no-mem-check disables the guard.

EXIT_REFUSED = 3                   # memory guard refused; see nn_qab.sh

MEM_BASE_BYTES = 1 << 30           # interpreter, numpy, Qrack, driver


def _meminfo(key):
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def mem_available(allow_swap=False):
    avail = _meminfo("MemAvailable")
    if avail is not None and allow_swap:
        avail += _meminfo("SwapFree") or 0
    return avail


# Swap traffic per reference run, from the kernel's own counters. At width
# 36 most of the state vector lives in swap and every gate pass rewrites it,
# so this is the number that tracks NVMe wear; it is recorded per seed.
def swap_pages():
    out = {}
    try:
        with open("/proc/vmstat") as f:
            for line in f:
                k, v = line.split()
                if k in ("pswpin", "pswpout"):
                    out[k] = int(v)
    except OSError:
        pass
    return out.get("pswpin", 0), out.get("pswpout", 0)


PAGE_BYTES = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def host_bytes_needed(width, stats, engine):
    amp = 1 << width
    scale = 2 if Qrack.fppow >= 6 else 1
    on_host_sv = engine in ("cpu",)
    if stats == "exact":
        per_amp = 12
    else:
        per_amp = 8 if on_host_sv else 0
    return MEM_BASE_BYTES + per_amp * scale * amp


def resolve_stats(width, stats, engine, sdrp, mem_check, allow_swap=False):
    """Pick exact/linear for this width and refuse if neither fits."""
    if sdrp is not None:
        # Schmidt rounding makes the footprint unpredictable, so no guard.
        # And under SDRP, exact stats would be self-defeating: OutProbs
        # composes every shard back into one 2**n vector, which is exactly
        # the allocation the rounding was there to avoid.
        return "linear" if stats == "auto" else stats
    if not mem_check:
        return "exact" if stats == "auto" else stats

    avail = mem_available(allow_swap)
    if avail is None:
        return "exact" if stats == "auto" else stats

    # With swap counted, auto still tries exact first. Once the state vector
    # spills, the extra probability vector is a small cost next to the gate
    # passes; and at 35 the state vector stays in RAM and only the probs
    # spill. Exact rows also carry xeb_linear, so exact is strictly more.
    candidates = ("exact", "linear") if stats == "auto" else (stats,)
    for mode in candidates:
        need = host_bytes_needed(width, mode, engine)
        if need <= avail:
            ram = mem_available(False)
            if ram is not None and need > ram:
                print(f"note: width {width} {mode} stats will put "
                      f"~{(need - ram) / 2**30:.0f} GiB in swap",
                      file=sys.stderr, flush=True)
            return mode

    need = host_bytes_needed(width, candidates[-1], engine)
    what = "RAM + swap" if allow_swap else "RAM"
    raise MemoryError(
        f"width {width} ({candidates[-1]} stats, engine {engine}) needs "
        f"~{need / 2**30:.0f} GiB host memory, {avail / 2**30:.0f} GiB "
        f"{what} available"
    )


# ---------------------------------------------------------------------------
# Mode: single self-contained run
# ---------------------------------------------------------------------------

def make_reference(width, engine="statevector", sdrp=None):
    """The exact (or, with SDRP, approximate) ground-truth simulator.

    With SDRP set, Qrack rounds Schmidt decompositions and can report an
    estimate of its own fidelity afterwards. That turns the reference from
    exact into approximate-with-a-number-attached, which is the only option
    left once 2**n amplitudes no longer fit -- past roughly width 30 on a
    10 GiB card. Record the fidelity alongside the XEB or the comparison
    is not interpretable.
    """
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {sorted(ENGINES)}")
    sim = QrackSimulator(width, **ENGINES[engine])
    if sdrp is not None:
        sim.set_sdrp(sdrp)
    return sim


def reference_stats(qc, width, counts, shots, engine="statevector",
                    sdrp=None, stats="exact"):
    """Run the reference for one circuit and score the ACE counts against it.

    Always computes the linear estimator from the sampled amplitudes (cheap,
    O(shots)); additionally computes the exact full-vector estimator when
    stats == "exact". The primary xeb/hog are whichever `stats` names.
    """
    t0 = time.perf_counter()
    swin0, swout0 = swap_pages()
    sim = make_reference(width, engine, sdrp)
    run_circuit(sim, qc)
    fidelity = sim.get_unitary_fidelity() if sdrp is not None else None

    idx, p_s = sampled_probs(sim, counts)
    ideal_probs = out_probs_np(sim) if stats == "exact" else None
    del sim
    gc.collect()
    t_ideal = time.perf_counter()

    xeb_lin, hog_lin = calc_stats_linear(
        p_s, [counts[k] for k in idx], shots, width)
    if ideal_probs is not None:
        xeb, hog = calc_stats(ideal_probs, counts, shots)
        del ideal_probs
        gc.collect()
    else:
        xeb, hog = xeb_lin, hog_lin

    # Host-wide counters: anything else swapping at the same time lands here
    # too, which is one more reason the big profile runs one worker.
    swin1, swout1 = swap_pages()

    return {
        "xeb_ace":          xeb,
        "hog_ace":          hog,
        "xeb_linear":       xeb_lin,
        "hog_linear":       hog_lin,
        "stats_mode":       stats,
        "unitary_fidelity": fidelity,
        "ideal_seconds":    t_ideal - t0,
        "stats_seconds":    time.perf_counter() - t_ideal,
        "swap_in_gib":      (swin1 - swin0) * PAGE_BYTES / 2**30,
        "swap_out_gib":     (swout1 - swout0) * PAGE_BYTES / 2**30,
    }


def bench_qrack(width, depth, lrc=4, lrr=4, swap_mode="auto", seed=None,
                engine="statevector", sdrp=None, stats="auto", mem_check=True,
                allow_swap=False):
    all_bits = list(range(width))
    shots = 1 << min(10, width + 2)

    # Refuse before the (possibly long) ACE pass, not after it.
    stats = resolve_stats(width, stats, engine, sdrp, mem_check, allow_swap)

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
    ref = reference_stats(qc, width, ace_counts, shots, engine, sdrp, stats)
    print(f"ideal_seconds: {ref['ideal_seconds']:.4f}")
    print(f"stats_seconds: {ref['stats_seconds']:.4f}")

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
        "engine":             engine,
        "sdrp":               sdrp,
        "unitary_fidelity":   ref["unitary_fidelity"],
        "qrack_lib":          QRACK_LIB_SOURCE,
        "qrack_device":       QRACK_DEVICE,
        "stats_mode":         ref["stats_mode"],
        "xeb_ace":            ref["xeb_ace"],
        "hog_ace":            ref["hog_ace"],
        "xeb_linear":         ref["xeb_linear"],
        "hog_linear":         ref["hog_linear"],
    }


def run_single(args):
    result = bench_qrack(
        args.width, args.depth, args.lrc, args.lrr, args.swap_mode, args.seed,
        getattr(args, "engine", "statevector"), getattr(args, "sdrp", None),
        getattr(args, "stats", "auto"), getattr(args, "mem_check", True),
        getattr(args, "allow_swap", False),
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
                "qrack_device_ace": QRACK_DEVICE,
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

            stats = resolve_stats(rec["width"], args.stats, args.engine,
                                  args.sdrp, args.mem_check, args.allow_swap)
            ref = reference_stats(qc, rec["width"], counts, rec["shots"],
                                  args.engine, args.sdrp, stats)
            xeb = ref["xeb_ace"]

            out = {k: v for k, v in rec.items()
                   if k not in ("counts", "circuit")}
            out.update(ref)
            out.update({
                "engine": args.engine,
                "sdrp": args.sdrp,
                "qrack_lib": QRACK_LIB_SOURCE,
                "qrack_device_ideal": QRACK_DEVICE,
            })
            write_json(done, out)
            print(f"ideal seed={seed} xeb={xeb:.6f} ({stats}) "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
        except MemoryError as e:
            # Every seed in this directory has the same width, so this will
            # not get better on the next one. Stop instead of spinning.
            # Exit code 3, not 2: argparse already uses 2 for usage errors.
            release(lock)
            print(f"ideal seed={seed} REFUSED: {e}", file=sys.stderr,
                  flush=True)
            return EXIT_REFUSED
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
    "ace_seconds", "ideal_seconds", "stats_seconds",
    "engine", "sdrp", "unitary_fidelity",
    "qrack_lib", "qrack_device_ace", "qrack_device_ideal",
    # appended last: nn_qab.sh reads columns 8 and 12 by position
    "stats_mode", "xeb_linear", "hog_linear",
    "swap_in_gib", "swap_out_gib",
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
            "\n"
            "device selection (any mode, incl. the legacy form):\n"
            "  --device N     pin Qrack to OpenCL device N (default 0)\n"
            "  $QRACK_DEVICE  same, via the environment\n"
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

    def stats_flags(sp):
        sp.add_argument("--stats", default="auto",
                        choices=("auto", "exact", "linear"),
                        help="exact: full-vector XEB/HOG (needs 2**n probs "
                             "in host RAM). linear: Porter-Thomas XEB/HOG "
                             "from sampled amplitudes only. auto (default): "
                             "exact if it fits in MemAvailable, else linear, "
                             "else refuse.")
        sp.add_argument("--no-mem-check", dest="mem_check",
                        action="store_false",
                        help="skip the host-memory guard")
        sp.add_argument("--allow-swap", action="store_true",
                        help="count SwapFree in the guard's budget; per-seed "
                             "swap traffic is recorded either way")

    r = sub.add_parser("run", help="one self-contained run")
    geom(r)
    r.add_argument("--seed", type=int, default=None,
                   help="pin the circuit (default: unseeded)")
    r.add_argument("--engine", default="statevector", choices=sorted(ENGINES),
                   help="reference simulator engine (default statevector)")
    r.add_argument("--sdrp", type=float, default=None,
                   help="Schmidt-decomposition rounding parameter; makes the "
                        "reference approximate and records its fidelity")
    stats_flags(r)
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
    b.add_argument("--engine", default="statevector", choices=sorted(ENGINES),
                   help="reference simulator engine (default statevector)")
    b.add_argument("--sdrp", type=float, default=None,
                   help="Schmidt-decomposition rounding parameter; makes the "
                        "reference approximate and records its fidelity")
    stats_flags(b)
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
            engine="statevector",
            sdrp=None,
            stats="auto",
            mem_check=True,
            allow_swap=False,
        )
        return run_single(args)

    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
