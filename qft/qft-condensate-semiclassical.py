# Semiclassical Shor factoring engine (standalone; shares the chain_sample
# recurrence with qft-cosmos-classical-condensate.py but is not wired into it).
#
# Layout:
#   * ONE control qubit, recycled t times with measurement + feed-forward.
#     This is the semiclassical inverse QFT (Griffiths & Niu 1996; single-control
#     layout as in Parker & Plenio 2000): the counting register never exists as a
#     state vector.
#   * ONE work register q of n qubits, held as a full state vector. It holds a
#     superposition over up to r values a^k mod N and cannot be cut into
#     classically seamed patches.
#   * Controlled U^(2^j) = controlled in-place modular multiplication by
#     A = a^(2^j) mod N, applied as ONE ORACLE PERMUTATION on (ctrl, q) via
#     QrackSimulator.hash(): identity if ctrl=0, x -> A*x mod N if ctrl=1
#     (x >= N left fixed). Hence 1 + n simulated qubits. This is NOT a gate-level
#     circuit: a reversible modular multiplier needs ancillae (Beauregard 2003:
#     2n+3 qubits in total), so "qubits" in the output is not a hardware count.
#     (mcmuln/mcdivn are not used: on pyqrack 2.25.2, muln(11, 21) on x=0 yields 11.)
#   * Bit order: the step using U^(2^(t-1)) reads the LSB of m first, so the
#     k-th measured bit has weight 2^k.
#
# Every base a is checked two ways:
#   * classically: the order candidate must satisfy a^r = 1 mod N, and any
#     reported factor must divide N (exact at any size);
#   * statistically, against the exact QPE distribution for the true order
#     (computed classically, so small N only):
#       - generic r: log-likelihood gap of measured m vs ideal draws (+ z);
#       - r | 2^t: the ideal distribution is uniform on the r multiples of 2^t/r,
#         so the gap is identically 0 and carries no information. A support +
#         uniformity check is reported instead.
# Unlucky bases (odd order, or a^(r/2) = -1 mod N) are reported as such and the
# next base is tried (--max-a), so a correct simulation is never reported as a
# failed one just because of the base.
#
# Usage:
#   python3 qft-condensate-semiclassical.py 15 21 35 143 221 --shots 32 [--no-gpu] [--seed 1]
#   python3 qft-condensate-semiclassical.py 3127 --t 24 --shots 16 --max-a 4

import os

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):  # pyqrack reads this env var at import time
    os.environ.setdefault("PYQRACK_SHARED_LIB_PATH", QRACK_LIB_PATH)

import argparse, ctypes, math, sys, time
from fractions import Fraction

import numpy as np
from pyqrack import QrackSimulator

try:  # private pyqrack handle, used only to pass pre-packed hash tables (see HashTable)
    from pyqrack.qrack_system import Qrack as _Qrack
except ImportError:
    _Qrack = None

MAX_WORK_QUBITS = 31  # table has 2^(n+1) uint64 entries; n=31 is already 34 GiB


# ---------------------------------------------------------------- number theory

def is_prime(N):
    """Deterministic Miller-Rabin for N < 3.3e24."""
    if N < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for p in small:
        if N % p == 0:
            return N == p
    d, s = N - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for b in small:
        x = pow(b, d, N)
        if x in (1, N - 1):
            continue
        for _ in range(s - 1):
            x = x * x % N
            if x == N - 1:
                break
        else:
            return False
    return True


def perfect_power(N):
    """Return (b, k) with b**k == N and k >= 2, or None."""
    for k in range(N.bit_length(), 1, -1):
        b = round(N ** (1.0 / k))
        for c in (b - 1, b, b + 1):
            if c > 1 and c ** k == N:
                return c, k
    return None


def classical_precheck(N):
    """Inputs order finding cannot (or need not) handle. Returns a result dict or None."""
    if N < 4:
        return {"N": N, "note": "nothing to factor"}
    if is_prime(N):
        return {"N": N, "note": "prime, nothing to factor"}
    if N % 2 == 0:
        return {"N": N, "factors": [2, N // 2], "note": "even, factor 2 (no quantum step)"}
    pp = perfect_power(N)
    if pp:
        b, k = pp
        return {"N": N, "factors": [b, N // b],
                "note": f"perfect power {b}^{k}, split classically"}
    return None


def order_classical(a, N):
    x, r = a % N, 1
    while x != 1:
        x = (x * a) % N
        r += 1
    return r


def base_is_usable(a, r, N):
    """Classical ground truth: does order r of base a yield a factor?"""
    return r % 2 == 0 and pow(a, r // 2, N) != N - 1


def order_from_m(m, t, N, a):
    if m == 0:
        return None
    r = Fraction(m, 1 << t).limit_denominator(N).denominator
    for mult in range(1, 4):  # small multiples recover r when gcd(s, r) > 1
        if pow(a, r * mult, N) == 1:
            return r * mult
    return None


def factors_from_order(a, r, N):
    if not base_is_usable(a, r, N):
        return None
    f = math.gcd(pow(a, r // 2, N) - 1, N)
    if 1 < f < N:
        return sorted((f, N // f))
    return None


# ---------------------------------------------------------------- QPE statistics

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
    half = min(64, T // 2)  # keep the window free of duplicate residues
    out = []
    for _ in range(shots):
        s = rng.integers(r)
        c = s * T / r
        cand = np.arange(math.floor(c) - half + 1, math.floor(c) + half + 1) % T
        d = s / r - cand / T
        d = d - np.round(d)
        den = T * np.sin(np.pi * d)
        w = np.where(np.abs(den) < 1e-15, 1.0, (np.sin(np.pi * T * d) / np.where(den == 0, 1, den)) ** 2)
        out.append(int(rng.choice(cand, p=w / w.sum())))
    return out


def statistical_check(r, t, ms, ideal):
    T = 1 << t
    shots = len(ms)
    if T % r == 0:
        # Ideal distribution: uniform on the r multiples of T/r. The log-likelihood
        # gap is identically 0 here, so test support and uniformity directly.
        step = T // r
        on = [m // step for m in ms if m % step == 0]
        res = {"check": "support+uniformity", "off_support": shots - len(on)}
        expected = len(on) / r
        if r > 1 and expected >= 5:
            counts = np.bincount(on, minlength=r)
            res["chi2"] = round(float(((counts - expected) ** 2 / expected).sum()), 2)
            res["dof"] = r - 1
        else:
            res["chi2"] = None
            res["note"] = "too few shots per outcome for a uniformity test"
        return res

    lp_meas = np.log(np.maximum(qpe_probs(r, t, ms), 1e-300))
    lp_ideal = np.log(np.maximum(qpe_probs(r, t, ideal), 1e-300))
    gap = lp_meas - lp_ideal
    res = {"check": "loglik_gap", "loglik_gap_vs_ideal": round(float(gap.mean()), 3)}
    sd = gap.std(ddof=1) if shots > 1 else 0.0
    res["z"] = round(float(gap.mean() / (sd / math.sqrt(shots))), 2) if sd > 0 else None
    if shots < 30:
        res["note"] = "z is only indicative below ~30 shots (skewed log-likelihoods)"
    return res


# ---------------------------------------------------------------- oracle tables

class HashTable:
    """Controlled x -> A*x mod N permutation on (ctrl, q), ctrl = LSB.

    Built once with numpy. QrackSimulator.hash() re-packs its table into bytes in a
    Python double loop on every call, which dominates run time for n >~ 12; when the
    fast path is available (and passes the self-test in make_applier) the packed
    buffer is built once here and handed to the C library directly.
    """

    def __init__(self, A, N, n):
        if n > MAX_WORK_QUBITS:
            raise SystemExit(f"N has {n} bits; oracle table would need 2^{n + 1} entries")
        nq = n + 1
        t = np.arange(1 << nq, dtype=np.uint64)
        x = np.arange(N, dtype=np.uint64)
        t[(x << np.uint64(1)) | np.uint64(1)] = ((np.uint64(A) * x) % np.uint64(N) << np.uint64(1)) | np.uint64(1)
        self.values = t
        c = (nq - 1) // 8 + 1  # bytes per entry, little-endian (pyqrack's _to_ubyte layout)
        self._buf = np.ascontiguousarray(t.astype("<u8").view(np.uint8).reshape(-1, 8)[:, :c]).reshape(-1)
        self.cbuf = (ctypes.c_ubyte * self._buf.size).from_buffer(self._buf)

    def nbytes(self):
        return self.values.nbytes + self._buf.nbytes


def _hash_fast(sim, reg, reg_c, table):
    _Qrack.qrack_lib.Hash(sim.sid, len(reg), reg_c, table.cbuf)
    sim._throw_if_error()


def _hash_slow(sim, reg, reg_c, table):
    sim.hash(reg, table.values.tolist())


def make_applier(use_gpu):
    """Pick the fast hash path only if it reproduces sim.hash() exactly."""
    if _Qrack is None:
        return _hash_slow, "pyqrack.hash (slow path: private handle unavailable)"
    N, A, n = 21, 11, 5
    tab = HashTable(A, N, n)
    reg = list(range(n + 1))
    reg_c = (ctypes.c_ulonglong * len(reg))(*reg)
    kets = []
    for fn in (_hash_slow, _hash_fast):
        sim = QrackSimulator(n + 1, is_gpu=use_gpu)
        for q in reg:
            sim.u(q, 0.3 + 0.2 * q, 0.1 * q, 0.0)  # generic product state
        try:
            fn(sim, reg, reg_c, tab)
        except Exception:
            return _hash_slow, "pyqrack.hash (slow path: fast path raised)"
        kets.append(np.array(sim.out_ket()))
    if abs(abs(np.vdot(kets[0], kets[1])) - 1.0) < 1e-5:  # equal up to global phase
        return _hash_fast, "prepacked table (self-test passed)"
    return _hash_slow, "pyqrack.hash (slow path: fast path self-test failed)"


# ---------------------------------------------------------------- engine

def run_once(sim, reg, reg_c, a, N, t, tables, apply_hash):
    ctrl = 0
    sim.reset_all()
    sim.x(1)  # work register q = |1>
    m = 0
    R = 0.0
    for step, j in enumerate(range(t - 1, -1, -1)):  # largest power first
        if sim.m(ctrl):             # recycle the control qubit
            sim.x(ctrl)
        sim.h(ctrl)
        A = pow(a, 1 << j, N)
        if A != 1:
            apply_hash(sim, reg, reg_c, tables[A])
        if R:
            sim.u(ctrl, 0.0, 0.0, -2 * math.pi * R)  # semiclassical inverse-QFT correction
        sim.h(ctrl)
        b = sim.m(ctrl)
        m |= b << step              # U^(2^(t-1)) reads the LSB of m first
        R = (R + b / 2) / 2
    return m


def try_base(N, a, t, shots, rng, use_gpu, apply_hash):
    n = N.bit_length()
    r_true = order_classical(a, N)
    usable = base_is_usable(a, r_true, N)

    t0 = time.perf_counter()
    tables = {}
    for j in range(t):
        A = pow(a, 1 << j, N)
        if A != 1 and A not in tables:
            tables[A] = HashTable(A, N, n)
    t_tables = time.perf_counter() - t0

    reg = list(range(1 + n))
    reg_c = (ctypes.c_ulonglong * len(reg))(*reg)
    sim = QrackSimulator(1 + n, is_gpu=use_gpu)
    # Seeds Qrack's measurement RNG only on builds without hardware RNG; the PyPI wheel
    # uses RDRAND, so there --seed fixes the bases and ideal draws but not the measured m.
    sim.seed(int(rng.integers(1, 2**31)))
    t1 = time.perf_counter()
    ms = [run_once(sim, reg, reg_c, a, N, t, tables, apply_hash) for _ in range(shots)]
    t_shots = time.perf_counter() - t1

    orders = [order_from_m(m, t, N, a) for m in ms]
    found = [r for r in orders if r]
    facs = None
    for r in found:
        facs = factors_from_order(a, r, N)
        if facs:
            break

    ideal = sample_ideal_m(r_true, t, rng, shots)
    if facs:
        outcome = "factored"
    elif not usable:
        outcome = "base_unusable"      # odd order or a^(r/2) = -1: not a simulation failure
    elif not found:
        outcome = "order_not_found"    # possible for an exact sampler at low shot counts
    else:
        outcome = "unexpected"         # usable base + order found must factor; investigate
    return {
        "a": a, "true_order": r_true, "base_usable": usable, "outcome": outcome,
        "order_found_rate": round(len(found) / shots, 3),
        "ideal_order_found_rate": round(sum(1 for m in ideal if order_from_m(m, t, N, a)) / shots, 3),
        "seconds_tables": round(t_tables, 3),
        "table_mb": round(sum(tb.nbytes() for tb in tables.values()) / 2**20, 1),
        "seconds_per_shot": round(t_shots / shots, 4),
        **statistical_check(r_true, t, ms, ideal),
    }, facs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("N", type=int, nargs="+")
    ap.add_argument("--shots", type=int, default=16)
    ap.add_argument("--t", type=int, default=None, help="phase bits (default 2n)")
    ap.add_argument("--max-a", type=int, default=8, help="bases to try per N before giving up")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    use_gpu = not args.no_gpu
    apply_hash, hash_path = make_applier(use_gpu)

    for N in args.N:
        pre = classical_precheck(N)
        if pre:
            print(pre)
            continue
        n = N.bit_length()
        t = args.t or 2 * n
        tries, tried, facs = [], set(), None
        gcd_hits = 0
        while len(tries) < args.max_a and len(tried) < N - 3:
            a = int(rng.integers(2, N - 1))
            if a in tried:
                continue
            tried.add(a)
            if math.gcd(a, N) != 1:
                gcd_hits += 1          # a lucky classical factor; skip to exercise the quantum step
                continue
            info, facs = try_base(N, a, t, args.shots, rng, use_gpu, apply_hash)
            tries.append(info)
            if facs:
                break
        ok = facs is not None and facs[0] * facs[1] == N
        if ok:
            status = "factored"
        elif tries and all(x["outcome"] == "base_unusable" for x in tries):
            status = "simulation_ok_all_bases_unusable"
        else:
            status = "not_factored"
        print({
            "N": N, "qubits": 1 + n, "qubits_note": "simulated (oracle permutation), not a gate-level count",
            "t": t, "shots_per_base": args.shots, "hash_path": hash_path,
            "factors": facs, "verified": ok, "status": status,
            "bases_tried": len(tries), "coprime_gcd_hits_skipped": gcd_hits,
            "tries": tries,
        })
    return 0


if __name__ == "__main__":
    sys.exit(main())
