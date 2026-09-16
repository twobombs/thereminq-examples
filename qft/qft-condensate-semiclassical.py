# Semiclassical Shor factoring engine for the QFT-cosmos condensate.
#
# Layout (Beauregard / Parker-Plenio style):
#   * ONE control qubit, recycled 2n times with measurement + feed-forward.
#     This is the semiclassical inverse QFT: the counting register never exists
#     as a state vector and needs no seams or Bell pairs at all.
#   * ONE work patch: n-qubit register q plus n-qubit scratch o (o is |0> between
#     steps), held as a full state vector. This is the part that must stay
#     "very close to a full SV": it holds a superposition over up to r values
#     a^k mod N, and its entanglement across any cut can be ~n/2 ebits, so it
#     cannot be split into classically seamed patches.
#   * Controlled U^(2^j) = controlled in-place modular multiplication by
#     A = a^(2^j) mod N, using Qrack's permutation-level arithmetic:
#         mcmuln(A)  : o <- A*q mod N            (if control)
#         cswap      : q <-> o                   (if control)
#         mcdivn(A^-1): o <- o - A^-1 * q mod N  (if control)  -> o back to 0
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


def run_once(N, a, t, rng, use_gpu):
    n = N.bit_length()
    ctrl = 0
    q = list(range(1, 1 + n))
    o = list(range(1 + n, 1 + 2 * n))
    sim = QrackSimulator(1 + 2 * n, is_gpu=use_gpu)
    sim.x(q[0])  # work register = |1>

    bits = {}
    R = 0.0
    for j in range(t - 1, -1, -1):  # largest power first
        if sim.m(ctrl):             # recycle the control qubit
            sim.x(ctrl)
        sim.h(ctrl)
        A = pow(a, 1 << j, N)
        if A != 1:
            Ainv = pow(A, -1, N)
            sim.mcmuln(A, [ctrl], N, q, o)
            sim.cswap([ctrl], q[0], o[0]) if n == 1 else [sim.cswap([ctrl], qi, oi) for qi, oi in zip(q, o)]
            sim.mcdivn(Ainv, [ctrl], N, q, o)
        if R:
            sim.u(ctrl, 0.0, 0.0, -2 * math.pi * R)  # semiclassical inverse-QFT correction
        sim.h(ctrl)
        b = sim.m(ctrl)
        bits[j] = b
        R = (R + b / 2) / 2
    return sum(b << j for j, b in bits.items())


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
        ms = [run_once(N, a, t, rng, not args.no_gpu) for _ in range(args.shots)]
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
            "N": N, "a": a, "qubits": 1 + 2 * n, "t": t, "shots": args.shots,
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
