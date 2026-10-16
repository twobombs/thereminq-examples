# Results: `rcs/nighthawk`

Every measured number of the experiments, grouped by experiment. The procedure behind each is in [METHODOLOGY.md](METHODOLOGY.md) (experiment IDs E1–E15 in its §9); usage is in the [README](README.md). State as of 16 October 2026.

| Convention | |
|---|---|
| device reference | `ibm_phoenix`, 61 qubits: the paper's mirror fit F(d) = 0.326 × 0.8717^d (b_dev = 0.137 per cycle); per-depth points from the release's counts |
| θ | Haar = the released ensemble; `nn_qab` = variant ensemble, not comparable value for value with Haar or with the device |
| means | "inverse-variance" from `run --summarize`; "plain" = mean over instances with instance-scatter error |
| fxeb | forward XEB against an exact reference, 4096 shots per instance unless noted |
| mirror | depth d = d/2 forward + d/2 inverse cycles; `prob` estimator unless noted |

## Index

| § | Experiment | Answer |
|---|---|---|
| 1 | E1, E2: reproducing the release | circuits identical; hardware numbers reproduced |
| 2 | E3: layout or placement | placement: 0.008 → 0.321 at n = 27, d = 8 |
| 3 | E4, E5: size, depth, tiling, ACE switches | b = 0.23–0.32 per cycle at n = 27–29; no switch lowers it |
| 4 | E6: ensemble | `nn_qab` scores ≈ 2× Haar; placement ×2.85 |
| 5 | cost | reference-dominated `fxeb`; 6–11 h per deep 61-qubit mirror point |
| 6 | E7, E8: 61 qubits, three patches | 2.3× the device at d = 4–6, 1/160 at d = 20 |
| 7 | seam cost and E9 | ≈ 0.025 per replica coupler per cycle; two patches beat three at 49 qubits |
| 8 | E10: 88 qubits | `nighthawk88e` 10–20× `nighthawk88d` |
| 9 | E11: two patches at 61 qubits | level with the device at d = 4–6 on its own circuits |
| 10 | E12: ladder vs Dan's calculator | holds at 28 qubits, overestimates above |
| 11 | E13: cloverfield | ghost link 1.000 / 0.989 / 0.656 vs mean-field 0.773 / 0.773 / 0.067 |
| 12 | E14, E15: OpenCL and GTT | three defects, fixed; GTT ceiling lifted |
| 13 | tool validation | GPU = CPU; layoutsearch matches the exact engine |
| 14 | not yet measured | open points |

---

## 1. Reproducing the release (E1, E2)

| Check | Result |
|---|---|
| `verify`: every released circuit regenerated | identical (operations, angles within 10⁻⁹ rad, CZ counts) |
| selftest [1] angle hash + pseudo-patch rotation vs `mirror/d08_instance1.qasm` | identical |
| selftest [2] Qrack `u` vs Haar matrix | overlap 1.0000001 |
| selftest [3] exact mirror, 14 qubits, d = 12, 10 inputs | F = 1.0000000 |
| selftest [4] twirl identity, exact | F = 0.9999998 |
| selftest [5] PyQrack patch ideal XEB vs release (Aer, double) | max difference 1.8 × 10⁻⁸ |
| selftest [6] Eq. (1)–(2) on the hardware counts | 0.02555 vs 0.02555 (σ 3.9 × 10⁻³) |
| selftest [7] ideal samples | F = 0.9562 ± 0.0379 |
| selftest [9] register recipe on the release's window | inputs and K = 3, 4 partitions identical |
| `hwxeb` | per (K, d) within 0.05 σ of the release; float32 ideal XEB within ≈ 10⁻⁵ |

---

## 2. Layout and placement at n = 27 (E3)

| lrc/lrr | XEB d = 8, device placement | XEB d = 8, tiled v2 | XEB d = 12, tiled v2 | effective bulk / seam qubits |
|---|---|---|---|---|
| 4/4 | 0.008 | **0.321** | **0.182** | 8.0 |
| 4/3 | 0.036 | 0.227 | 0.153 | 8.0 |
| 2/7 | 0.046 | 0.154 | 0.136 | 5.8 |
| 1/7 | 0.077 | 0.145 | 0.109 | 3.5 |
| 2/2 | −0.032 | 0.117 | 0.091 | 0.7 |

(Inverse-variance means, 3 instances.)

| Spearman ρ with XEB (10 configurations, d = 8) | ρ |
|---|---|
| simulators holding bulk qubits | −0.84 |
| cross couplers | −0.80 |
| exact couplers | +0.74 |
| logical qubits on seams | −0.40 |
| replica couplers | −0.21 |
| widest simulator | −0.15 |

| Layout alone (no tiling), 8 × 8 window | Couplers on a seam (of 102) or coupler split | Simulators |
|---|---|---|
| 4/4 (default) | 69 | ≤ 23 qubits |
| 5/4 (best with every simulator ≤ 33) | 68 | ≤ 33 |
| 2/7 torus | 57 | 33 / 33 / 25 / 25, 128 GiB |
| device placement, 4/4, n = 61 | exact / replica / cross 33 / 30 / 39 | — |
| tiled v2, 4/4, n = 61 | 61 / 41 / 0 | — |

Tiling removes every cross coupler at unchanged memory, and the widest simulator does not predict the XEB. Once cross couplers are gone, the ordering follows the qubits on seams.

---

## 3. Size, depth, tiling version and ACE switches (E4, E5)

| n | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 |
|---|---|---|---|---|---|---|---|---|
| v2: exact/replica/cross, seam/sims | 31/10/0, 3/2 | 23/20/0, 8/3 | 27/18/0, 5/3 | 28/19/0, 6/3 | 29/20/0, 7/3 | 22/28/0, 11/3 | 34/18/0, 6/3 | 35/19/0, 6/3 |
| v3: exact/replica/cross, seam/sims | 32/9/0, 3/2 | 30/13/0, 4/3 | 33/12/0, 4/2 | 34/13/0, 5/2 | 35/14/0, 5/3 | 35/15/0, 6/3 | 35/17/0, 5/3 | 36/18/0, 5/3 |

| n | seam/sims v3 vs v2 | d = 12: v3 / v2 | d = 16: v3 / v2 | d = 20: v3 / v2 | b over d = 12–20 (v3) |
|---|---|---|---|---|---|
| 27 | 3/2 vs 3/2 | 0.165 / 0.146 | 0.062 / 0.063 | 0.026 / 0.033 | 0.23 |
| 28 | 4/3 vs 8/3 | **0.147 / 0.040** | **0.051 / 0.012** | 0.016 / 0.021 | 0.27 |
| 29 | 4/2 vs 5/3 | 0.140 / 0.114 | 0.030 / 0.034 | 0.011 / 0.015 | 0.32 |
| 30 (v2 only) | 6/3 | — / 0.069 | — / 0.001 | — / −0.001 | — |

(Inverse-variance means, 3 instances, same circuits and one shared reference per point; σ ≈ 0.009; d = 20 at the 4096-shot floor.)

| Decay at n = 27 (v3) | b per cycle |
|---|---|
| d = 8–12 | 0.19 |
| d = 16–20 | 0.25 |
| device at n = 61 | 0.137 |

| ACE switch: paired difference from the default (5 circuits) | n = 27, d = 12 | n = 27, d = 16 | n = 28, d = 12 | n = 28, d = 16 |
|---|---|---|---|---|
| boundary repetition code on | −0.007 ± 0.010 | +0.009 ± 0.008 | −0.012 ± 0.011 | −0.001 ± 0.008 |
| seam error detection off | +0.008 ± 0.006 | +0.007 ± 0.009 | +0.029 ± 0.014 | +0.011 ± 0.005 |
| crossbars off | +0.001 ± 0.012 | +0.002 ± 0.009 | −0.020 ± 0.009 | −0.002 ± 0.005 |

| Error detection off, effect on the decay | n = 27 | n = 28 |
|---|---|---|
| b, d = 12 → 16, default | 0.27 | 0.26 |
| b, d = 12 → 16, error detection off | 0.25 | 0.26 |
| combined XEB offset | +0.010 ± 0.003 | |

Halving the seam qubits (n = 28) raises the XEB 3.6–4.3-fold, and where the seam counts agree the two tilings coincide. No ACE switch lowers the decay per cycle; error detection off only adds a constant offset at n = 27–28 (at 61 qubits it costs, §9).

---

## 4. The `nn_qab` ensemble (E6; n = 27, d = 12)

| Configuration | Circuits | XEB (plain mean) | HOG |
|---|---|---|---|
| Nighthawk window, 4/4 tiled v3 | 20 | **0.435 ± 0.037** | 0.665 |
| Nighthawk window, 4/4 device placement | 20 | 0.153 ± 0.014 | 0.500 |
| `nn_qab` 26 qubits, B-to-B 5.5 (best series) | 100 | 0.241 | — |
| `nn_qab` 27 qubits, two patches, B-to-B 3.5 | 100 | 0.185 | — |
| `nn_qab` 27 qubits, three patches, B-to-B 2.0 | 100 | 0.111 | — |
| Haar ensemble, 4/4 tiled v2 (inverse-variance) | 3 | 0.182 | — |

| Same runs, by mean | inverse-variance | plain |
|---|---|---|
| tiled | 0.343 | 0.435 ± 0.037 |
| untiled | 0.125 | 0.153 ± 0.014 |

| Ideal XEB at d = 12 | |
|---|---|
| `nn_qab` ensemble | ≈ 16 |
| Haar ensemble | ≈ 1.5 |

- **Placement:** within the `nn_qab` ensemble, placement alone gives ×2.85.
- **Geometry:** tiled ACE on the Nighthawk window beats every `nn_qab` geometry by ≥ ×1.8. Part of that gap may be the CZ-only gate set.
- **HOG at chance:** with the device placement, HOG sits at chance while the XEB stays positive, so a few heavy bitstrings carry the score.

---

## 5. Cost

| Point | Hardware | Time |
|---|---|---|
| `fxeb`, n = 34, d = 20 | 96-thread EPYC | ≈ 6.2 × 10³ s |
| `fxeb`, tiled 4/4, n = 27 | CPU | 1–4 × 10² s |
| `fxeb`, tiled 2/7, n = 27 | CPU | ≈ 1.8 × 10³ s |
| `fxeb`, two variants, n = 27–28 / n = 29 | CMP 50HX, 3 / 2 workers | ≈ 10 / 13 min per worker |
| exact reference alone, n = 27 | CMP 50HX | ≈ 10 s |
| mirror, n = 61, untiled, shots, d = 4–20 | EPYC | 0.65–4.5 × 10³ s |
| mirror, n = 61, three patches, `prob`, 640 runs, d = 16 / 20 | EPYC, ≈ 16 threads per point | ≈ 6 h / 11 h |
| mirror, 45-qubit two-patch, 10 runs, d = 8 / 12 / 16 / 20 | 2-core container | 64 / 340 / 671 / 1361 s |
| mirror, `nighthawk2p40`, 10 runs, d = 16 | 2-core container | 157 s |
| mirror, `nighthawk61r2`, 80 runs per instance, d = 4 / 6 | 2-core container | ≈ 3 / 5 s |
| ACE mirror on the patched OpenCL build vs `QRACK_OCL_SYNC=1` | PoCL | 185 s vs 198 s |

---

## 6. 61 qubits, three patches (E7, E8)

Placements of the release circuit on the 75-site register (lrc 4 lrr 5), with a first mirror check (`nn_qab`, one instance, 1000 shots, 4 twirls):

| Exclusions | Engine | exact / replica / cross | Seam qubits | Memory | d = 4 | d = 6 |
|---|---|---|---|---|---|---|
| none, 64 sites lrc 4 lrr 4 | anneal | 63 / 38 / 1 | 12 | 0.05 GiB | — | — |
| none | exact | 73 / 29 / 0 | 8 | 0.77 GiB | 0.121 ± 0.010 | 0.056 ± 0.007 |
| first 7 | anneal, `xok` | 76 / 24 / 2 | 8 | 0.66 GiB | — | — |
| 11 | anneal, `xok` | 76 / 25 / 1 | 8 | 0.44 GiB | 0.897 ± 0.010 | 0.118 ± 0.010 |
| 11, site 69 reopened | anneal | 72 / 30 / 0 | 9 | 0.50 GiB | 0.799 ± 0.013 | 0.086 ± 0.009 |
| **11** | **exact, proven optimal** | **74 / 28 / 0** | **9 (4 / 3 / 2)** | **0.50 GiB** | **0.646 ± 0.015** | **0.388 ± 0.015** |
| 11 | vf2 / plain anneal | none within the bound / refused (cross coupler 34–41) | — | — | — | — |
| 64 sites, lrc 4 lrr 7, error detection off (two patches) | anneal | 84 / 18 / 0 | 6 | 192 GiB | — | — |

| Bound | Value |
|---|---|
| boundary qubits of any balanced three-way split of the Nighthawk graph | ≥ 9 (proven) |
| VF2 at n = 27 | 9–10 qubits on seams (17 exact / 24 replica on 64 sites) vs 3 tiled |
| VF2, n = 18 smoke test | 0.26 vs 0.60–0.69 annealed |

Sweep on the exact placement (`nn_qab`, `prob`, 10 inputs × 64 twirls per instance, EPYC CPU):

| d | i0 | i1 | i2 | inverse-variance | plain mean | `ibm_phoenix` | ACE / device (plain) | boundary CZs (forward half) |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.630 | 0.292 | 0.550 | 0.495 ± 0.007 | 0.491 | 0.214 | 2.3 | 11 |
| 6 | 0.364 | 0.260 | 0.388 | 0.338 ± 0.006 | 0.337 | 0.154 | 2.2 | 18 |
| 8 | 0.0126 | 0.0726 | 0.0461 | 0.0309 ± 0.0016 | 0.0438 | 0.109 | 0.40 | 24 |
| 10 | 0.0119 | 0.0590 | 0.0471 | 0.0305 ± 0.0016 | 0.0393 | 0.081 | 0.49 | 27 |
| 12 | 0.0108 | 0.0632 | 0.0452 | 0.0238 ± 0.0013 | 0.0397 | 0.062 | 0.64 | 27 |
| 16 | failed (C++ exception) | 1.38 × 10⁻³ | 5.3 × 10⁻⁴ | 5.9 × 10⁻⁴ | 9.5 × 10⁻⁴ | 0.030 | 1/32 | 40 |
| 20 | 4.6 × 10⁻⁵ | 2.1 × 10⁻⁴ | — | — | 1.3 × 10⁻⁴ | 0.021 | 1/160 | 49 |

![61-qubit mirror, every instance](mirror61_dan_instances_d4-20.png)

| Fit | Value |
|---|---|
| d = 8 → 12 | +3 boundary CZs, survival flat |
| d = 12 → 20 | +22 boundary CZs, factor ≈ 310: 0.26 nats per boundary CZ |
| instance 1, d = 12–16 / 16–20 | 0.29 / 0.21 nats per boundary CZ |
| single exponential (`--summarize`) | 5.30 × 0.586^d (b ≈ 0.53; exceeds 1 at d = 0) |
| power law, log(−ln F) vs log d | p = 1.59 (d = 4–20), 1.28 (d ≥ 8) |
| power law against cumulative boundary CZs | exponent 1.81 |
| Dan's model at this register (52 bulk, 9 boundary, 3 patches) | XEB(12) = 0.215 |

| PyQrack version effect (d = 6, CPU) | Shift |
|---|---|
| instance 1, 2.28.1 vs the release before | −0.007 |
| instance 2 | +0.005 |

The survival steps with the boundary-CZ count rather than with depth. Instance 0 runs on a lower curve from d = 8 on and pulls the inverse-variance mean below the plain mean. The circuits use the `nn_qab` ensemble, so the comparison with the device's Haar points is one of shape.

---

## 7. Seam cost extrapolated to 61 qubits; two vs three patches at 49 (E9)

| n (tiled v3) | b over d = 12–20 | replica couplers | b per replica coupler |
|---|---|---|---|
| 27 | 0.23 | 9 | 0.026 |
| 28 | 0.27 | 13 | 0.021 |
| 29 | 0.32 | 12 | 0.027 |
| device at n = 27–29 (2.25 × 10⁻³ per qubit-cycle) | ≈ 0.06 | — | — |

| Split of the 8 × 8 window | Couplers between simulators | Estimated b at 61 | Measured |
|---|---|---|---|
| tiled 4/4, five simulators | ≈ 40 | ≈ 1.0 | — |
| three large patches | ≈ 16 | ≈ 0.4 | ≈ 0.53 (§6) |
| two halves | ≈ 8 | ≈ 0.2 | §9 |
| device | — | 0.137 | — |

| 49-qubit sub-register, same qubits | Patch widths | Boundary qubits | exact / replica / cross | Mirror d = 4 / 6 (`nn_qab`, 20 runs) |
|---|---|---|---|---|
| 55 sites, lrc 5 lrr 4, flat | 28 / 27 | 4 | 64 / 15 / 0 | 0.760 / 0.706 |
| 75 sites, lrc 4 lrr 5, torus | 23 / 23 / 12 | 6 | 58 / 21 / 0 | 0.206 / 0.234 |

---

## 8. 88 qubits on one 8 GB die (E10)

| Patches | Logical qubits | Boundary qubits | Widest patch incl. ancilla |
|---|---|---|---|
| 2 | 49 | 4 | 28 |
| 3 (experiment's register) | 61 | 9 | 25 |
| 3 | 68 | 11 | 28 |
| 4, widths ≤ 27 | 87 | 16 | 27 |
| 4 (`nighthawk88`) | 88 | 17 | 28 |
| 4, disabled sites on the seams (`nighthawk88d`, `nighthawk88e`) | 88 | 11 | 28 |

![nighthawk88d](nighthawk88d_four_patch_seams.png)

| `nn_qab`, 3 instances × 40 runs, inverse-variance | Boundary couplers per A–D cycle | d = 4 | d = 6 |
|---|---|---|---|
| `nighthawk88` | 55 | 0.047 ± 0.009 | 0.0082 ± 0.0020 |
| `nighthawk88d` | 42 | 0.030 ± 0.005 | 0.0073 ± 0.0018 |
| `nighthawk88e` | 36 | **0.287 ± 0.020** | **0.143 ± 0.012** |

| Instance scatter at d = 4 | 0.018–0.20 |
|---|---|

`nighthawk88e` has 14 % fewer boundary couplers than `nighthawk88d` and scores 10–20× higher. So where the seams sit relative to the mirrors' rotating pseudo-patch cuts matters too, not only how many couplers cross them.

---

## 9. Two patches at 61 qubits (E11)

| Split of the release register | Boundary qubits | Couplers through the boundary | Circuit CZs through the boundary, Σ d = 4–20 | Widths incl. ancilla |
|---|---|---|---|---|
| three patches (§6) | 9 | 28 | 196 | ≤ 25 |
| two patches, coupler objective | 6 (proven minimum, balanced) | 18 | 135 | 35 / 34 |
| two patches, circuit-CZ objective (`nighthawk61r2`) | 6 | 19 | 128 | 35 / 34 |
| two patches, unbalanced | 6 | 18 | 120 | 33 / 36 |

| Boundary CZs per mirror (forward half), d | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 36 |
|---|---|---|---|---|---|---|---|---|---|
| three patches | 11 | 18 | 24 | 27 | 27 | 40 | 49 | 62 | 93 |
| `nighthawk61r2` | 8 | 12 | 14 | 16 | 19 | 25 | 34 | 41 | 67 |
| `nighthawk61w` (3-qubit waist) | 1 | 3 | 6 | 6 | 7 | 12 | 13 | 19 | 27 |

| Register | θ | Error detection | Runs per point | d = 4 | d = 6 |
|---|---|---|---|---|---|
| three patches (§6) | `nn_qab` | on | 640 | 0.495 | 0.338 |
| `nighthawk61r2` | `nn_qab` | on | 240 | 0.522 ± 0.019 | 0.354 ± 0.016 |
| `nighthawk61r2` | `nn_qab` | off | 240 | 0.448 ± 0.017 | 0.207 ± 0.012 |
| `nighthawk61r2` | Haar | on | 20 (1 instance) | 0.332 ± 0.099 | 0.100 ± 0.062 |
| `nighthawk61r2` | Haar | on | 240 | **0.226 ± 0.021** (plain 0.254) | **0.128 ± 0.014** (plain 0.147) |
| `nighthawk61w` | `nn_qab` and Haar | on | 240 | 1.00000 | 1.00000 |
| `ibm_phoenix` | Haar | — | — | 0.214 | 0.154 |

(Inverse-variance means unless noted.)

| `nighthawk61r2`, Haar, per instance | d = 4 | d = 6 |
|---|---|---|
| i0 | 0.312 ± 0.045 | 0.177 ± 0.037 |
| i1 | 0.137 ± 0.031 | 0.083 ± 0.019 |
| i2 | 0.312 ± 0.041 | 0.180 ± 0.024 |

45-qubit waist analogue (rows 2–11 × cols 0–4, 3-qubit seam, widths 26 / 24, Haar, 1 instance × 10 runs):

| d | 8 | 12 | 16 | 20 |
|---|---|---|---|---|
| boundary CZs | 5 | 6 | 10 | 11 |
| survival | 0.899 ± 0.035 | 0.883 ± 0.036 | 0.164 ± 0.10 | 0.124 ± 0.074 |
| `ibm_phoenix`, 61 qubits | 0.109 | 0.062 | 0.030 | 0.021 |
| per-run survivals | 1.00, 0.78, 0.78, 1.00, 0.97, 1.00, 0.75, 0.97, 0.78, 0.97 (d = 8) | 0.78, 1.00, 1.00, 0.78, 0.78, 0.97, 1.00, 1.00, 0.75, 0.78 (d = 12) | 0.13, 0.01, 0.47, 0.00, 0.00, 0.00, 1.00, 0.02, 0.00, 0.00 (d = 16) | — |

| Same analogue, `nn_qab` | d = 8 | d = 12 | d = 16 |
|---|---|---|---|
| survival | 0.600 ± 0.13 | 0.530 ± 0.13 | C++ exception |

| Comparison | `nighthawk61r2` | `nighthawk61w` |
|---|---|---|
| qubits, CZs at d = 20 | 61, 221 | 61, 221 |
| narrowest cut | 8 (IBM's window) | 3 |
| what a win shows | ACE against the device on the device's register | ACE on a two-chip register with a narrow link |

---

## 10. Two-patch ladder against Dan's calculator (E12)

| n | 28 | 32 | 36 | 40 | 45 | 49 | 53 | 57 | 61 |
|---|---|---|---|---|---|---|---|---|---|
| register | `2p28` | `2p32` | `2p36` | `2p40` | `2p45` | `2p49` | `2p53` | `2p57` | `61r2` |
| boundary qubits (proven minimum) | 3 | 3 | 3 | 3 | 4 | 4 | 5 | 5 | 6 |
| patch widths incl. ancilla | 15/18 | 18/19 | 20/21 | 22/23 | 26/25 | 27/28 | 30/30 | 32/32 | 35/34 |
| couplers through the boundary | 11 | 11 | 12 | 12 | 12 | 15 | 15 | 18 | 19 |
| circuit CZs through the boundary, Σ d = 4–20 | 70 | 80 | 95 | 78 | 79 | 107 | 122 | 125 | 128 |
| boundary CZs at d = 20 | 18 | 20 | 25 | 21 | 20 | 29 | 34 | 34 | 34 |
| Dan's XEB(12) | 0.372 | 0.409 | 0.444 | 0.476 | 0.425 | 0.450 | 0.408 | 0.429 | 0.396 |
| Dan's XEB at d = 4 / 20 / 36 | 0.844 / 0.106 / 0.0031 | 0.858 / 0.132 / 0.0054 | 0.870 / 0.158 / 0.0088 | 0.881 / 0.186 / 0.0132 | 0.864 / 0.143 / 0.0068 | 0.872 / 0.163 / 0.0095 | 0.857 / 0.130 / 0.0053 | 0.865 / 0.146 / 0.0071 | 0.853 / 0.122 / 0.0045 |

Measured mirror survival (plain means, error detection on, CPU; 3 instances × 10 runs to n = 36, 1 × 10 at 40–45), with Dan's prediction in brackets:

| n | d = 4 | 6 | 8 | 10 | 12 | 16 | 20 | 36 | fitted p | fitted XEB(12) |
|---|---|---|---|---|---|---|---|---|---|---|
| **`nn_qab` θ** | | | | | | | | | | |
| 28 | 0.593 (0.844) | 0.637 (0.722) | 0.543 (0.597) | 0.546 (0.478) | 0.462 (0.372) | 0.354 (0.208) | 0.058 (0.106) | 1.9 × 10⁻⁴ (0.0031) | 1.68 | 0.360 |
| 32 | 0.730 (0.858) | 0.679 (0.746) | 0.260 (0.628) | 0.247 (0.514) | 0.212 (0.409) | 0.135 (0.242) | 0.082 (0.132) | 2.4 × 10⁻⁴ (0.0054) | 1.44 | 0.217 |
| 36 | 0.765 (0.870) | 0.642 (0.766) | 0.259 (0.655) | 0.157 (0.546) | 0.058 (0.444) | 0.019 (0.276) | 2.8 × 10⁻⁴ (0.158) | 8.9 × 10⁻⁵ (0.0088) | 1.68 | 0.100 |
| 40 | 0.989 (0.881) | 0.986 (0.784) | 0.319 (0.679) | 0.059 (0.575) | 0.092 (0.476) | 0.034 (0.308) | 0.0088 (0.186) | 1.6 × 10⁻⁵ (0.0132) | 2.87 | 0.306 |
| 45 | 0.748 (0.864) | 0.856 (0.755) | 0.600 (0.640) | 0.087 (0.528) | 0.103 (0.425) | 0.067 (0.257) | 0.0010 (0.143) | — | 2.89 | 0.166 |
| **Haar θ** | | | | | | | | | | |
| 28 | 0.278 | 0.196 | 0.087 | 0.054 | 0.056 | 0.056 | 9.3 × 10⁻⁴ | 2.7 × 10⁻⁵ | 1.02 | 0.039 |
| 32 | 0.327 | 0.146 | 0.024 | 0.039 | 0.013 | 0.026 | 0.0037 | 2.7 × 10⁻⁵ | 0.81 | 0.021 |
| 36 | 0.397 | 0.142 | 0.052 | 0.0015 | 0.0041 | 4.0 × 10⁻⁵ | 4.7 × 10⁻⁴ | 2.7 × 10⁻⁶ | 1.01 | 0.005 |
| 40 | 0.849 | 0.772 | 0.152 | 0.060 | 0.0046 | 0.0012 | 1.5 × 10⁻⁶ | 4.2 × 10⁻⁸ | 2.14 | 0.043 |
| 45 | 0.319 | 0.048 | 0.0078 | 0.0096 | 0.0028 | 0.0055 | — | — | 0.86 | 0.003 |
| **Dan's calculator** | | | | | | | | | 1.606 | 0.37–0.48 |

(Fitted p and XEB(12): Dan's form, log(−ln F) linear in log(d/12), from d = 6 on. Dan's predictions are the same for both ensembles.)

| Measured / Dan at d = 20, `nn_qab` | 28 | 32 | 36 | 40 | 45 |
|---|---|---|---|---|---|
| ratio | 0.55 | 0.62 | 0.0018 | 0.048 | 0.0071 |

| Forward XEB, `nighthawk2p28`, `nn_qab`, 1 instance | d = 4 | d = 12 |
|---|---|---|
| `fxeb` against the exact reference | 0.885 ± 0.044 | 0.338 ± 0.016 |
| Dan's calculator | 0.844 | 0.372 |
| fxeb / Dan | 1.05 | 0.91 |
| mirror survival, same register | 0.593 | 0.462 |
| ideal XEB | 1581 | 11.2 |
| HOG | 0.958 | 0.626 |

| Dan's calculator at the release register, against measurement | d = 4 | d = 6 |
|---|---|---|
| prediction (`nighthawk61r2`: 55 bulk, 6 boundary, 2 patches) | 0.853 | 0.737 |
| measured, Haar (plain) | 0.254 | 0.147 |
| measured, `nn_qab` | 0.522 | 0.354 |

| Dan's spreadsheet itself | Value |
|---|---|
| depth power refitted from its depth series (d = 14–22) | 1.610 (sheet: 1.60569) |
| leave-one-depth-out | 1.584 / 1.592 / 1.587 / 1.660 / 1.629 (leaving out d = 22 / 20 / 18 / 16 / 14) |
| worked example (52 / 7 / 2, d = 36) | 0.0019774 (sheet and `danmodel`) |
| fit form vs calculator form at d = 36 | 0.00252 vs 0.00198 |

- **At 28 qubits, the edge of the calculator's data, it holds.** It agrees in its own variable (XEB within 10 %), and its p and XEB(12) match the fit to the measurements.
- **Above 28 it overestimates more and more with size and depth.** Its x grows with the bulk, while the measured XEB(12) falls and p rises to ≈ 2.9.
- **Under Haar the losses are 10–100× larger.** The fit on `nn_qab` circuits does not carry over.

---

## 11. Cloverfield (E13)

| Loom error alone: two exact 9-qubit chips, 2 links (selftest [5]) | d = 8 | d = 12 | d = 16 |
|---|---|---|---|
| no links | 1 | 1 | 1 |
| mean-field | 0.773 | 0.773 | 0.067 |
| ghost | **1.000** | **0.989** | **0.656** |

| Loom algebra | Result |
|---|---|
| op stream, one chip, no links (d = 4, 20 runs) | 20 618 ops identical to nighthawk_qrack |
| worker process vs in-process | 0.375971 vs 0.375882 |
| woven CZ on \|+⟩ against \|0⟩, \|1⟩, \|+⟩ | ⟨X⟩ = +1, −1, 0 |
| CZ·CZ across a link | ghost +1; mean-field −1 |
| loom process memory | ≈ 45 MiB |

| Two `nighthawk88e` chips, 176 qubits, 3 ghost links (`nn_qab`, 3 instances × 20 runs, plain) | d = 4 | d = 6 |
|---|---|---|
| ghost | 0.079 | 0.017 |
| mean-field (same links) | 0.056 | 0.015 |

| Links that fit | VRAM only | `--gtt 8` | `--gtt 16` |
|---|---|---|---|
| two `nighthawk88e` chips, ghost | 3 (5.8 GiB per chip; widths 25 / 28 / 27 / 26) | 5 (2.8 GiB VRAM + 8 GiB GTT; 25 / 28 / 29 / 26) | 6 (a 30-wide patch: 2.8 GiB VRAM + 16 GiB GTT) |
| two `nighthawk88e` chips, mean-field | 8 | — | — |
| four chips, 2 × 2, ghost | 10 (6.5 GiB per chip) | 14 (4.5 GiB VRAM + 8 GiB GTT each) | — |

---

## 12. Qrack's OpenCL engine and VRAM oversubscription (E14, E15)

| 16-qubit reproducer, PoCL, Qrack 10.13.1 | prob(q3) / prob(q10) |
|---|---|
| CPU | 0.414847 / 0.430348 |
| OpenCL, stock | NaN / NaN, NaN / NaN, segfault (three runs) |
| OpenCL, `DISABLE_OOO_OCL` | 1.0 / 1.0 (wrong) |
| OpenCL, synchronous dispatch | 0.414847 / 0.430348 |

| Defect | Evidence |
|---|---|
| A: async writes from out-of-scope stack arrays | fixed together with B: correct |
| B: `USE_HOST_PTR` memory freed while kernels queued | fixed together with A: correct; either alone: wrong |
| C: callback on an empty event | `CL_INVALID_EVENT` in 1099 of 1099 registrations |

| Patched build (`qrack_ocl_async_fix.patch` on 10.13.2) | Result |
|---|---|
| reproducer | exact in 5 of 5 runs |
| ACE mirror, d = 4 and 6, 3 instances × 640 runs, vs CPU on the same build | within 3 × 10⁻⁴ |
| wall time vs `QRACK_OCL_SYNC=1` | 185 s vs 198 s |

| `--gtt` ceiling test (PoCL, loom worker environment) | Result |
|---|---|
| 24-qubit entangled state (128 MiB) under a 61 MB ceiling | refused |
| same under 563 MB | runs |

| Qrack CPU qubit ceiling (`QRACK_MAX_CPU_QB = 10`, 12 entangled qubits) | Result |
|---|---|
| unset | runs, P = 0.5 |
| set | "QUnit needed to engage automatic circuit elision" exception |
| `QRACK_MAX_PAGE_QB` / `QRACK_MAX_PAGING_QB` = 10 | run |

---

## 13. Tool validation

| Check | Result |
|---|---|
| tiled ACE, n = 27, d = 12 / 16 / 20: CPU vs CMP 50HX (± 0.009) | 0.158 / 0.063 / 0.024 vs 0.146 / 0.063 / 0.033 |
| concurrent references on one 10 GB card | 10 workers at n = 27 and 3 at n = 28 failed; 1 worker completed |
| `layoutsearch`, release register (75 sites) vs exact engine (selftest [10]) | 8 / 8 boundary qubits, proven optimal in ≈ 30 s; 26 vs 29 couplers through a boundary |
| `layoutsearch` n = 88, `convex`, warm start from `nighthawk88d` | 11 boundary qubits kept; 42 → 36 couplers (`nighthawk88e`); diverse second candidate 13 / 38 |
| `layoutsearch` n = 49, two vs three patches, `rect --trim 2`, d = 4, 6 | 3 vs 6 boundary qubits; score 0.94 vs 0.59 |
| `layoutsearch` n = 49, `convex` | a two-patch register with 1 boundary qubit (2 couplers): a neck, excluded by `rect` |
| `layoutsearch` n = 28, cap ≥ n | everything in one patch (0 boundary qubits): ladder capped at n/2 + 5 |
| `layoutsearch` n = 61, `rect --trim 6`, two patches, cap 34 | 2–3 boundary qubits (waist registers, e.g. `nighthawk61w`) |
| `--save-best` round trip | stored placement applied; survival identical (1.000, 0.8867) |
| `auto` geometries, n = 49, two patches | 3 geometries in ≈ 1 min |
| boundary-CZ counter on the three-patch register | 11 / 18 / 24 / 27 / 27 / 40 / 49: §6's column exactly |
| `release:61`, `nighthawk61r2` vs the release | mirror CZs, angles and inputs identical at d = 4, 8, 20 |
| ACE placements with error detection on vs off | identical site maps (widths 43 / 43 / 8 vs 42 / 42 / 7 on 77 sites) |
| `--swap-gb` config tag | unchanged (`ace-3189ad58` with and without) |
| `fxeb` on a register | runs from n ≤ 36 (`nighthawk2p28`: 104 s at d = 4, 464 s at d = 12 incl. reference) |
| existing configurations after each change | tags unchanged; selftest ALL OK |

---

## 14. Not yet measured

| Point | Status |
|---|---|
| three patches, 61 qubits, d = 16 instance 0 | failed: CPU C++ exception, not yet traced (also seen at 45 qubits, `nn_qab`, d = 16) |
| three patches, 61 qubits, d = 20 instance 2 | not run |
| three patches, 61 qubits, d = 24 | dropped (≈ 11 h per d = 20 point already) |
| `nighthawk61r2`, d ≥ 8, error detection on | needs RAM + swap on the EPYC (worst case 513 of 536 GiB) |
| `nighthawk61w`, d ≥ 8 | needs the 64 GiB patches (EPYC) |
| ladder n = 49, 53, 57; n ≥ 40 with 3 instances | EPYC |
| 45-qubit ladder, Haar, d = 20 | not finished |
| `fxeb` ladder, n = 32 | EPYC (48 GiB reference) |
| `nighthawk88d` / `88e` depth sweep | six V340 dies |
| tiling v3 vs v2 at n = 30–32 | CPU path |
| host-mapped (`--gtt`) patch speed on the V340s | not measured |

Runtime screenshots:

<img width="970" height="403" alt="image" src="https://github.com/user-attachments/assets/717d8289-2a12-4de4-892d-77d5a161f955" />
<img width="1114" height="327" alt="image" src="https://github.com/user-attachments/assets/b84b32ce-4fae-434b-b06a-62b38d32c4b2" />
