# Semiclassical Shor factoring engine for the QFT-cosmos condensate.
#
# Layout (Beauregard / Parker-Plenio style):
#   * ONE control qubit, recycled 2n times with measurement + feed-forward.
#     This is the semiclassical inverse QFT: the counting register never exists
#     as a state vector and needs no seams or Bell pairs at all.
#   * ONE work patch: n-qubit register q, held as a full state vector. This is
#     the part that must stay "very close to a full SV": it holds a superposition
#     over up to r values a^k mod N, and its entanglement across any cut can be
#     ~n/2 ebits, so it cannot be split into classically seamed patches.
#   * Controlled U^(2^j) = controlled in-place modular multiplication by
#     A = a^(2^j) mod N, applied as one exact permutation on (ctrl, q) via
#     QrackSimulator.hash(): identity if ctrl=0, x -> A*x mod N if ctrl=1
#     (x >= N left fixed). No scratch register, so 1 + n qubits total.
#     (mcmuln/mcdivn were dropped: in pyqrack-cpu 2.10-2.25.2 muln does not
#     return A*x mod N for many inputs, e.g. N=21, A=11, x=0 -> 11.)
#   * Bit order: the step using U^(2^(t-1)) reads the LSB of m first, so the
#     k-th measured bit has weight 2^k (same recurrence as chain_sample(inverse)).
#
# Every run is checked two ways:
#   * classically: the order candidate r must satisfy a^r = 1 mod N, and any
#     reported factor must divide N (exact at any size);
#   * statistically: each measured m gets its exact QPE probability given the true
#     order (computed classically here, so only for small N) -> log-likelihood gap
#     against ideal draws, as in the condensate patch checks.
#
# Usage:
#   python3 shor_semiclassical.py 15 21 35 143 221 --shots 32 [--no-gpu] [--seed 1]
#   python3 shor_semiclassical.py 3127 --t 24 --shots 16

import os

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):  # pyqrack reads this env var at import time
    os.environ.setdefault("PYQRACK_SHARED_LIB_PATH", QRACK_LIB_PATH)

import argparse, math, sys, time
from fractions import Fraction

import numpy as np
from pyqrack import QrackSimulator


def order_classical(a, N):
    x, r = a % N, 1
    while x != 1:
        x = (x * a) % N
        r += 1
    return r


def qpe_probs(r, t, ms):
    """Exact P(m) for QPE with t bits on the uniform mixture of phases s/r."""
    T = 1 << t
    s = np.arange(r)[None, :] / r
    d = s - np.asarray(ms, dtype=float)[:, None] / T
    d = d - np.round(d)
    num = np.sin(np.pi * T * d)
    den = T * np.sin(np.pi * d)
    k = np.where(np.abs(den) < 1e-15, 1.0, (num / np.where(den == 0, 1, den)) ** 2)
    return k.mean(axis=1)


def sample_ideal_m(r, t, rng, shots):
    """Draw m from the ideal distribution: pick s, then m from the kernel around s/r*2^t."""
    T = 1 << t
    out = []
    for _ in range(shots):
        s = rng.integers(r)
        c = s * T / r
        cand = np.arange(math.floor(c) - 64, math.floor(c) + 66) % T
        d = s / r - cand / T
        d = d - np.round(d)
        den = T * np.sin(np.pi * d)
        w = np.where(np.abs(den) < 1e-15, 1.0, (np.sin(np.pi * T * d) / np.where(den == 0, 1, den)) ** 2)
        out.append(int(rng.choice(cand, p=w / w.sum())))
    return out


def ctrl_mul_table(A, N, n):
    """Permutation on (ctrl, q), ctrl = LSB: identity if ctrl=0, x -> A*x mod N if ctrl=1.
    Values x >= N are left fixed, so the table is a bijection."""
    t = list(range(1 << (n + 1)))
    for x in range(N):
        t[1 | (x << 1)] = 1 | ((A * x % N) << 1)
    return t


def run_once(N, a, t, rng, use_gpu, tables):
    n = N.bit_length()
    ctrl = 0
    reg = list(range(1 + n))            # ctrl + q; no scratch register needed
    sim = QrackSimulator(1 + n, is_gpu=use_gpu)
    sim.x(1)  # work register q = |1>

    m = 0
    R = 0.0
    for step, j in enumerate(range(t - 1, -1, -1)):  # largest power first
        if sim.m(ctrl):             # recycle the control qubit
            sim.x(ctrl)
        sim.h(ctrl)
        A = pow(a, 1 << j, N)
        if A != 1:
            sim.hash(reg, tables[A])
        if R:
            sim.u(ctrl, 0.0, 0.0, -2 * math.pi * R)  # semiclassical inverse-QFT correction
        sim.h(ctrl)
        b = sim.m(ctrl)
        m |= b << step              # U^(2^(t-1)) reads the LSB of m first
        R = (R + b / 2) / 2
    return m


def order_from_m(m, t, N, a):
    if m == 0:
        return None
    frac = Fraction(m, 1 << t).limit_denominator(N)
    r = frac.denominator
    for mult in range(1, 4):  # small multiples recover r when gcd(s, r) > 1
        if pow(a, r * mult, N) == 1:
            return r * mult
    return None


def factors_from_order(a, r, N):
    if r % 2:
        return None
    x = pow(a, r // 2, N)
    if x == N - 1:
        return None
    f = math.gcd(x - 1, N)
    if 1 < f < N:
        return sorted((f, N // f))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("N", type=int, nargs="+")
    ap.add_argument("--shots", type=int, default=16)
    ap.add_argument("--t", type=int, default=None, help="phase bits (default 2n)")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    for N in args.N:
        if N % 2 == 0:
            print({"N": N, "note": "even, factor 2"})
            continue
        n = N.bit_length()
        t = args.t or 2 * n
        a = int(rng.integers(2, N - 1))
        while math.gcd(a, N) != 1:
            a = int(rng.integers(2, N - 1))
        r_true = order_classical(a, N)

        t0 = time.perf_counter()
        tables = {}
        for j in range(t):
            A = pow(a, 1 << j, N)
            if A != 1 and A not in tables:
                tables[A] = ctrl_mul_table(A, N, n)
        ms = [run_once(N, a, t, rng, not args.no_gpu, tables) for _ in range(args.shots)]
        dt = time.perf_counter() - t0

        orders = [order_from_m(m, t, N, a) for m in ms]
        found = [r for r in orders if r]
        facs = None
        for r in found:
            facs = factors_from_order(a, r, N)
            if facs:
                break
        ok = facs is not None and facs[0] * facs[1] == N

        ideal = sample_ideal_m(r_true, t, rng, args.shots)
        lp_meas = np.log(np.maximum(qpe_probs(r_true, t, ms), 1e-300))
        lp_ideal = np.log(np.maximum(qpe_probs(r_true, t, ideal), 1e-300))
        gap = lp_meas - lp_ideal

        print({
            "N": N, "a": a, "qubits": 1 + n, "t": t, "shots": args.shots,
            "seconds_per_shot": round(dt / args.shots, 4),
            "true_order": r_true,
            "order_found_rate": round(len(found) / args.shots, 3),
            "ideal_order_found_rate": round(sum(1 for m in ideal if order_from_m(m, t, N, a)) / args.shots, 3),
            "factors": facs, "verified": ok,
            "loglik_gap_vs_ideal": round(float(gap.mean()), 3),
            "z": round(float(gap.mean() / (gap.std(ddof=1) / math.sqrt(len(gap)))), 2) if len(gap) > 1 and gap.std() > 0 else 0.0,
        })
    return 0


if __name__ == "__main__":
    sys.exit(main())
