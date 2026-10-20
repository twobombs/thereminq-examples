import itertools
import math
import random
import sys
import time

import numpy as np


def factor_width(width, is_transpose=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len
    return (col_len, row_len) if is_transpose else (row_len, col_len)


def int_to_bitstring(integer, length):
    return bin(integer)[2:].zfill(length)


def closeness_like_bits(perm, n_rows, n_cols):
    """
    Compute closeness-of-like-bits metric C(state) for a given bitstring on an LxL toroidal grid.
    """
    if n_rows * n_cols == 0:
        return 0.0
    bitstring = list(int_to_bitstring(perm, n_rows * n_cols))
    grid = np.array(bitstring).reshape((n_rows, n_cols))
    total_edges = 0
    like_count = 0
    for i in range(n_rows):
        for j in range(n_cols):
            s = grid[i, j]
            s_right = grid[i, (j + 1) % n_cols]
            like_count += 1 if s == s_right else -1
            total_edges += 1
            s_down = grid[(i + 1) % n_rows, j]
            like_count += 1 if s == s_down else -1
            total_edges += 1
    return like_count / total_edges if total_edges > 0 else 0.0


def expected_closeness_weight(n_rows, n_cols, hamming_weight):
    L = n_rows * n_cols
    if L < 2:
        return 0.0
    try:
        same_pairs = math.comb(hamming_weight, 2) + math.comb(L - hamming_weight, 2)
        total_pairs = math.comb(L, 2)
    except ValueError:
        return 0.0 # Occurs if hamming_weight > L
    if total_pairs == 0:
        return 0.0
    mu_k = same_pairs / total_pairs
    return 2 * mu_k - 1


def main():
    # --- Default Parameters ---
    n_qubits = 16
    depth = 20
    shots = 100
    t2 = 1
    omega = 1.5
    J = -1.0
    h = 2.0
    dt = 0.25
    theta = math.pi / 18
    delta_theta = None

    # --- Parse CLI Arguments ---
    if len(sys.argv) > 1: n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2: depth = int(sys.argv[2])
    if len(sys.argv) > 3: dt = float(sys.argv[3])
    if len(sys.argv) > 4: shots = int(sys.argv[4])
    if len(sys.argv) > 5: J = float(sys.argv[5])
    if len(sys.argv) > 6: h = float(sys.argv[6])
    if len(sys.argv) > 7: theta = float(sys.argv[7])
    if len(sys.argv) > 8: delta_theta = float(sys.argv[8])

    print(f"Running with: n_qubits={n_qubits}, J={J}, h={h}, theta={theta}, delta_theta={delta_theta}")

    omega *= math.pi
    n_rows, n_cols = factor_width(n_qubits, False)
    qubits = list(range(n_qubits))

    if delta_theta is None:
        z = 4
        if np.isclose(J, 0):
            theta_c = math.pi / 2 if h >= 0 else -math.pi / 2
        else:
            asin_arg = h / (z * J)
            asin_arg = max(-1.0, min(1.0, asin_arg))
            theta_c = math.asin(asin_arg)
        delta_theta = theta - theta_c

    start = time.perf_counter()
    bias = []
    t = depth * dt
    if np.isclose(h, 0):
        bias.append(1)
        bias += n_qubits * [0]
    elif np.isclose(J, 0):
        bias = (n_qubits + 1) * [1 / (n_qubits + 1)] if n_qubits > -1 else []
    else:
        sin_delta_theta = math.sin(delta_theta)
        p = (
            ((2 ** (abs(J / h) - 1)) * (1 + sin_delta_theta * math.cos(J * omega * t + theta) / ((1 + math.sqrt(t / t2)) if t2 > 0 else 1)) - 0.5)
            if t2 > 0 else (2 ** abs(J / h))
        )
        if p >= 1024:
            bias.append(1)
            bias += n_qubits * [0]
        else:
            tot_n = 0
            for q in range(n_qubits + 1):
                try:
                    n = 1 / (n_qubits * (2 ** (p * q)))
                except (OverflowError, ZeroDivisionError):
                    n = 0
                if n == float("inf"):
                    tot_n = 1; bias = [1.0] + [0.0] * n_qubits; break
                bias.append(n)
                tot_n += n
            if tot_n > 0:
                for q in range(n_qubits + 1):
                    bias[q] /= tot_n
            else: # Failsafe
                bias = [1.0] + [0.0] * n_qubits

    if J > 0 and bias:
        bias.reverse()

    thresholds = []
    tot_prob = 0
    if not bias: # Failsafe if bias calculation failed
        bias = [1.0] + [0.0] * n_qubits
    for q_prob in bias:
        tot_prob += q_prob
        thresholds.append(tot_prob)
    if thresholds:
        thresholds[-1] = 1.0

    samples = []
    for s in range(shots):
        mag_prob = random.random()
        m = 0
        while m < len(thresholds) and thresholds[m] < mag_prob:
            m += 1
        m = min(m, n_qubits) # Ensure m is a valid hamming weight

        if m == 0:
            samples.append(0)
            continue
        
        try:
            num_combos = math.comb(n_qubits, m)
            if num_combos > 1_000_000:
                print(f"Warning: High number of combinations ({num_combos}) for hamming weight {m}. May be slow or use a lot of memory.", file=sys.stderr)
        except (ValueError, OverflowError):
            samples.append(0) # Failsafe
            continue

        combinations_iter = itertools.combinations(qubits, m)
        states, weights = [], []
        norm_factor = 1.0 + expected_closeness_weight(n_rows, n_cols, m)
        if np.isclose(norm_factor, 0): # Avoid division by zero
            chosen_combo = random.choice(list(combinations_iter))
            samples.append(sum(1 << pos for pos in chosen_combo))
            continue

        for combo in combinations_iter:
            state_int = sum(1 << pos for pos in combo)
            states.append(state_int)
            weights.append((1.0 + closeness_like_bits(state_int, n_rows, n_cols)) / norm_factor)
        
        if states:
            if sum(weights) > 0:
                chosen_state = random.choices(states, weights=weights, k=1)[0]
                samples.append(chosen_state)
            else: # All weights are zero, pick uniformly
                samples.append(random.choice(states))

    seconds = time.perf_counter() - start
    print(samples)
    print("Seconds: " + str(seconds))

if __name__ == "__main__":
    sys.exit(main())
