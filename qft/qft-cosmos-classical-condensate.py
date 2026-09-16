# QFT-cosmos "condensate": a circuit of circuits with rotating seams.
#
# A ring of W qubits is cut into patches of p qubits. Each layer:
#   * patch boundaries rotate by --shift (so every seam moves each layer),
#   * every qubit is re-prepared as U(theta, phi, lambda)|y>, where y is its
#     measured bit from the previous layer (classical feed-forward across seams),
#   * every patch runs QFT or IQFT (random, like Qrack's test_cosmology),
#   * every patch is measured.
#
# Because each patch sees a product-state input and ends in a Z measurement, its
# output distribution factorises into a chain of exact conditionals. So:
#   * the whole condensate is sampled exactly at arbitrary width x depth,
#     cost O(W * layers * shots), no state vector anywhere;
#   * every patch outcome gets its exact ln P, so every patch is checkable;
#   * --check K runs K random patches per layer on Qrack (a p-qubit state
#     vector per shot), feeds Qrack's outcomes into the condensate, and scores them;
#   * --fault-rate injects a random RY error into checked patches to measure detection.
#
# Honest scope: the seams here are classical (measure -> re-prepare). That is exactly
# what makes arbitrary width and depth checkable, and it also means the condensate as a
# whole is classically easy. Coherent seams (no mid-layer measurement, or ebit
# telegates between layers) make entanglement grow each layer; then exact checking is
# limited to light-cone spot-checks whose reference size grows with depth.
#
# Conventions verified against pyqrack (TVD ~1e-7):
#   QFT  |x> = 2^{-n/2} sum_y exp(-2 pi i x.rev(y)/2^n) |y>
#   IQFT |z> = 2^{-n/2} sum_x exp(+2 pi i x.rev(z)/2^n) |x>
#   U(th, ph, lm)|0> ~ [cos(th/2), e^{i ph} sin(th/2)],  U|1> ~ [-sin(th/2), e^{i ph} cos(th/2)]
#
# Usage:
#   python3 qft_condensate.py --width 1000000 --patch 8 --layers 10 --shots 64
#   python3 qft_condensate.py --width 96 --patch 8 --layers 6 --shots 256 --check 3 --fault-rate 0.5

import argparse, math, sys, time
import numpy as np

LN2 = math.log(2)


def chain_sample(a, b, rng, inverse, forced=None):
    """a, b: (p, shots) complex. Exact sample + ln P for QFT (inverse=False) or IQFT."""
    p, shots = a.shape
    phase = np.zeros(shots)
    lnp = np.zeros(shots)
    bits = np.zeros((p, shots), dtype=np.uint8)
    order = range(p) if inverse else range(p - 1, -1, -1)
    sign = 1.0 if inverse else -1.0
    for k in order:
        w = b[k] * np.exp(sign * 2j * np.pi * phase)
        p0 = 0.5 * np.abs(a[k] + w) ** 2
        p1 = 0.5 * np.abs(a[k] - w) ** 2
        s = p0 + p1
        p0, p1 = p0 / s, p1 / s
        y = (rng.random(shots) < p1) if forced is None else forced[k].astype(bool)
        bits[k] = y
        lnp += np.log(np.where(y, p1, p0))
        phase = (phase + y / 2) / 2 if inverse else (y / 2 + phase) / 2
    return bits, lnp


def prepared_states(y, th, ph):
    """Per-qubit, per-shot amplitudes of U(th, ph, .)|y>. y: (p, shots); th, ph: (p,)."""
    c, s = np.cos(th / 2)[:, None], np.sin(th / 2)[:, None]
    e = np.exp(1j * ph)[:, None]
    a = np.where(y == 1, -s, c).astype(complex)
    b = np.where(y == 1, e * c, e * s)
    return a, b


def run_on_qrack(y, th, ph, lm, inverse, rng, fault_rate):
    """Run one patch per shot on Qrack. Returns bits (p, shots) and a per-shot fault flag."""
    from pyqrack import QrackSimulator
    p, shots = y.shape
    bits = np.zeros((p, shots), dtype=np.uint8)
    faults = np.zeros(shots, dtype=bool)
    qs = list(range(p))
    for t in range(shots):
        sim = QrackSimulator(p, is_gpu=USE_GPU)
        for q in qs:
            if y[q, t]:
                sim.x(q)
            sim.u(q, th[q], ph[q], lm[q])
        (sim.iqft if inverse else sim.qft)(qs)
        if rng.random() < fault_rate:
            faults[t] = True
            sim.u(int(rng.integers(p)), float(rng.uniform(0.5, 1.5) * math.pi / 2), 0.0, 0.0)
        v = sim.measure_shots(qs, 1)[0]
        bits[:, t] = [(v >> q) & 1 for q in qs]
    return bits, faults


def main():
    global USE_GPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--shift", type=int, default=None, help="seam rotation per layer (default patch//2)")
    ap.add_argument("--shots", type=int, default=128)
    ap.add_argument("--check", type=int, default=0, help="patches per layer run on Qrack and scored")
    ap.add_argument("--fault-rate", type=float, default=0.0)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    USE_GPU = not a.no_gpu

    W, p, L = a.width, a.patch, a.layers
    if W % p:
        raise SystemExit("--width must be a multiple of --patch")
    shift = p // 2 if a.shift is None else a.shift
    rng = np.random.default_rng(a.seed)
    npatch = W // p

    y = np.zeros((W, a.shots), dtype=np.uint8)  # measured bits, fed forward
    z_total = np.zeros(a.shots)                 # sum over all patches of (p ln2 + ln P)
    checks = {"clean": [], "faulty": []}
    t0 = time.perf_counter()

    for layer in range(L):
        offset = (layer * shift) % p
        th = 4 * np.pi * rng.random(W)
        ph = 2 * np.pi * rng.random(W)
        lm = 2 * np.pi * rng.random(W)
        inverse = rng.random(npatch) < 0.5
        checked = set(rng.choice(npatch, size=min(a.check, npatch), replace=False).tolist()) if a.check else set()

        # All patches of one kind at once: gather (npatch_k * p, shots) and reshape.
        idx = (offset + np.arange(W)) % W                    # ring positions in patch order
        new_y = np.empty_like(y)
        for kind in (False, True):
            ms = [m for m in range(npatch) if inverse[m] == kind and m not in checked]
            if not ms:
                continue
            q = np.concatenate([idx[m * p:(m + 1) * p] for m in ms])
            A, B = prepared_states(y[q], th[q], ph[q])
            # stack patches as extra "shots" axis groups: loop over patch position within patch
            A = A.reshape(len(ms), p, -1).transpose(1, 0, 2).reshape(p, -1)
            B = B.reshape(len(ms), p, -1).transpose(1, 0, 2).reshape(p, -1)
            bits, lnp = chain_sample(A, B, rng, kind)
            bits = bits.reshape(p, len(ms), -1).transpose(1, 0, 2).reshape(len(ms) * p, -1)
            new_y[q] = bits
            z_total += (p * LN2 + lnp).reshape(len(ms), -1).sum(axis=0)

        for m in checked:
            q = idx[m * p:(m + 1) * p]
            bits, faults = run_on_qrack(y[q], th[q], ph[q], lm[q], inverse[m], rng, a.fault_rate)
            A, B = prepared_states(y[q], th[q], ph[q])
            _, lnp_meas = chain_sample(A, B, rng, inverse[m], forced=bits)
            _, lnp_ideal = chain_sample(A, B, rng, inverse[m])       # same inputs, ideal draw
            z_meas = p * LN2 + lnp_meas
            z_ideal = p * LN2 + lnp_ideal
            for t in range(a.shots):
                checks["faulty" if faults[t] else "clean"].append(z_meas[t] - z_ideal[t])
            new_y[q] = bits
            z_total += z_meas

        y = new_y

    dt = time.perf_counter() - t0
    out = {
        "width": W, "patch": p, "layers": L, "shift": shift, "shots": a.shots,
        "patch_runs": npatch * L, "seconds": round(dt, 3),
        "mean_total_log_gain_per_patch": float(z_total.mean() / (npatch * L)),
    }
    for k, v in checks.items():
        if v:
            v = np.array(v)
            out[f"check_{k}"] = {
                "n": int(v.size),
                "mean_gap_vs_ideal": float(v.mean()),
                "z_score": float(v.mean() / (v.std(ddof=1) / math.sqrt(v.size))) if v.size > 1 else None,
            }
    print(out)
    return 0


USE_GPU = True

if __name__ == "__main__":
    sys.exit(main())

