# Results: `rcs/nighthawk`

Every measured number of the experiments, grouped by experiment. The procedure behind each is in [METHODOLOGY.md](METHODOLOGY.md) (experiment IDs E1–E19 in its §9); usage is in the [README](README.md). State as of 20 October 2026.

## Index

| § | Experiment | Answer |
|---|---|---|
| [1](#1-reproducing-the-release-e1-e2) | E1, E2: reproducing the release | circuits identical; hardware numbers reproduced |
| [2](#2-layout-and-placement-at-n--27-e3) | E3: layout or placement | placement: 0.008 → 0.321 at n = 27, d = 8 |
| [3](#3-size-depth-tiling-version-and-ace-switches-e4-e5) | E4, E5: size, depth, tiling, ACE switches | b = 0.23–0.32 per cycle at n = 27–29; no switch lowers it |
| [4](#4-the-nn_qab-ensemble-e6-n--27-d--12) | E6: ensemble | `nn_qab` scores ≈ 2× Haar; placement ×2.85 |
| [5](#5-cost) | cost | reference-dominated `fxeb`; 6–11 h per deep 61-qubit mirror point |
| [6](#6-61-qubits-three-patches-e7-e8) | E7, E8: 61 qubits, three patches | 2.3× the device at d = 4–6, 1/160 at d = 20 |
| [7](#7-seam-cost-extrapolated-to-61-qubits-two-vs-three-patches-at-49-e9) | seam cost and E9 | ≈ 0.025 per replica coupler per cycle; two patches beat three at 49 qubits |
| [8](#8-88-qubits-on-one-8-gb-die-e10) | E10: 88 qubits | `nighthawk88e` 10–20× `nighthawk88d` |
| [9](#9-two-patches-at-61-qubits-e11) | E11: two patches at 61 qubits | level with the device at d = 4–6 on its own circuits |
| [10](#10-two-patch-ladder-against-dans-calculator-e12-e17-e18) | E12, E17: ladder vs Dan's calculator | in his protocol (XEB): 1.03× at 16 qubits to 0.40× at 28 (d = 12) |
| [10.1](#101-dans-circuits-past-the-xeb-ceiling-his-mirror-hamming-weight-e18) | E18: his mirror Hamming weight, 16–57 qubits | 0.95–1.11× at d = 4–6 from 36 to 57 qubits; at d = 12 0.58–0.63× at 36–40, 0.91× at 45 (4-qubit seam); folding exponent 0.04–0.41 against his 1.606 |
| [10.2](#102-vqe-on-the-two-patch-ladder-e19) | E19: VQE energy, 16–28 qubits | the angles transfer (88–99.7 % of E₀'s correlation energy); ACE keeps 70–97 % of it at 16 qubits, less with depth from 20 on; the seam acts as local dephasing, never below E₀ |
| [11](#11-cloverfield-e13) | E13: cloverfield | ghost link 1.000 / 0.989 / 0.656 vs mean-field 0.773 / 0.773 / 0.067 |
| [12](#12-qracks-opencl-engine-and-vram-oversubscription-e14-e15) | E14, E15: OpenCL and GTT | three defects, fixed; GTT ceiling lifted |
| [13](#13-tool-validation) | tool validation | GPU = CPU; layoutsearch matches the exact engine |
| [14](#14-why-dans-calculator-and-our-numbers-differ-e16) | E16: Dan's calculator vs our numbers | the observable: XEB equals a seam-cut sampler's XEB, mirror survival does not |
| [15](#15-not-yet-measured) | not yet measured | open points |

| Convention | |
|---|---|
| device reference | `ibm_phoenix`, 61 qubits: the paper's mirror fit F(d) = 0.326 × 0.8717^d (b_dev = 0.137 per cycle); per-depth points from the release's counts |
| θ | Haar = the released ensemble; `nn_qab` = variant ensemble, not comparable value for value with Haar or with the device |
| means | "inverse-variance" from `run --summarize`; "plain" = mean over instances with instance-scatter error |
| fxeb | forward XEB against an exact reference, 4096 shots per instance unless noted |
| mirror | depth d = d/2 forward + d/2 inverse cycles; `prob` estimator unless noted |

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

## 10. Two-patch ladder against Dan's calculator (E12, E17, E18)

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

In Dan's protocol (`run --families fxeb --protocol nnqab`: forward XEB, sin θ-uniform single-qubit gates, his twelve two-qubit gates on the Nighthawk couplers, 2^min(13, n+2) shots; plain means over circuits, CPU), against his calculator:

| Register | n | Seam qubits | Couplers through the boundary | Circuits | d = 12 | d = 16 | d = 20 | Fitted p | Fitted XEB(12) |
|---|---|---|---|---|---|---|---|---|---|
| `nighthawk2p16` | 16 | 3 | 6 | 10 | 0.241 ± 0.041 (0.235) | 0.178 ± 0.023 (0.100) | 0.096 ± 0.019 (0.037) | 0.97 | 0.250 |
| `nighthawk2p20` | 20 | 3 | 9 | 10 | 0.200 ± 0.028 (0.285) | 0.133 ± 0.012 (0.136) | 0.079 ± 0.014 (0.058) | 0.89 | 0.203 |
| `nighthawk2p24` | 24 | 3 | 8 | 10 | 0.224 ± 0.019 (0.330) | 0.111 ± 0.015 (0.172) | 0.027 ± 0.007 (0.081) | 1.70 | 0.234 |
| `nighthawk2p28` | 28 | 3 | 11 | 3 | 0.148 ± 0.020 (0.372) | — | 0.025 ± 0.009 (0.106) | 1.29 | 0.148 |
| **Dan's calculator** | | | | | | | | 1.606 | 0.235–0.372 |

| Measured / Dan, protocol `nnqab` | 16 | 20 | 24 | 28 |
|---|---|---|---|---|
| d = 12 | 1.03 | 0.70 | 0.68 | 0.40 |
| d = 16 | 1.78 | 0.98 | 0.64 | — |
| d = 20 | 2.58 | 1.36 | 0.34 | 0.24 |

| Same registers, CZ-only gates (default protocol, 3 instances × 4096 shots) | 16 | 20 | 24 | 28 |
|---|---|---|---|---|
| d = 12 | 0.455 | 0.451 | 0.642 | 0.539 |
| d = 20 | 0.341 | 0.256 | 0.172 | 0.219 |

| Seam qubits vs couplers through the boundary | Seam qubits | Couplers through the boundary | Ratio |
|---|---|---|---|
| his strips, two patches (10, 14, 22, 26 qubits) | 4 | 12 | 3.0 |
| his 27 qubits, two / three patches | 6 / 9 | 18 / 30 | 3.0 / 3.3 |
| our ladder, 16 / 20 / 24 / 28 qubits | 3 | 6 / 9 / 8 / 11 | 2.0 / 3.0 / 2.7 / 3.7 |

Mirror survival (the `ibm_phoenix` comparison proxy, not Dan's observable; plain means, error detection on, CPU; 3 instances × 10 runs to n = 36, 1 × 10 at 40–45), with Dan's prediction in brackets:

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

- **In Dan's protocol the ladder meets his calculator at 16 qubits** (1.03× at d = 12). Its depth power, 0.89–1.70 over 16–28 qubits, brackets his 1.606.
- **The level parts with size.** At a fixed 3-qubit seam his x grows with the bulk, so his XEB(12) rises from 0.235 to 0.372 between 16 and 28 qubits; the measured XEB(12) stays at 0.15–0.25. The seam qubits touch as many couplers per seam qubit as on his strips (ratio 2–3.7 against 3.0), so the difference is the 2D patch geometry against his 2-wide strips, on which the bulk term was fitted.
- **For Nighthawk registers his calculator therefore applies with a measured XEB(12)**: his depth power with the anchor of this table. The CZ-only circuits score 2–4× higher XEB than his gate set on the same registers.
- **Mirror survival is a different observable** (§14) and is kept for the comparison with `ibm_phoenix`. Under Haar angles its losses are 10–100× larger.

### 10.1 Dan's circuits past the XEB ceiling: his mirror Hamming weight (E18)

`run --families hamming --protocol nnqab`: his circuits (the E17 gates and θ) as U U†, folded (U U†)^λ, signal 1 − 2h with h the exact mean fraction of ones (METHODOLOGY §6.4); plain means over circuits, error detection on; 16–32 container CPU (2 cores, 7 GiB), 36–61 on the EPYC (10 circuits; 3 at 61; 45–61 still filling). Dan's calculator in brackets, at the depth d of U.

| n | Circuits | d = 4 | 6 | 8 | 10 | 12 | Fitted p | Fitted signal(12) |
|---|---|---|---|---|---|---|---|---|
| 16 | 6 | 0.990 ± 0.010 (0.780) | 0.829 ± 0.085 (0.621) | 0.571 ± 0.147 (0.469) | 0.494 ± 0.133 (0.339) | 0.288 ± 0.107 (0.235) | 2.59 | 0.283 |
| 20 | 6 | 0.887 ± 0.064 (0.807) | 0.707 ± 0.069 (0.662) | 0.633 ± 0.088 (0.520) | 0.439 ± 0.123 (0.392) | 0.275 ± 0.039 (0.285) | 1.93 | 0.305 |
| 24 | 6 | 0.914 ± 0.026 (0.827) | 0.823 ± 0.053 (0.695) | 0.806 ± 0.051 (0.561) | 0.546 ± 0.082 (0.438) | 0.344 ± 0.076 (0.330) | 2.58 | 0.396 |
| 28 | 6 | 0.882 ± 0.037 (0.844) | 0.670 ± 0.102 (0.722) | 0.539 ± 0.090 (0.597) | 0.272 ± 0.073 (0.478) | 0.137 ± 0.031 (0.372) | 2.38 | 0.147 |
| 32 | 6 | 0.904 ± 0.027 (0.858) | 0.641 ± 0.087 (0.746) | 0.537 ± 0.085 (0.628) | 0.309 ± 0.062 (0.514) | 0.210 ± 0.033 (0.409) | 1.90 | 0.213 |
| 36 | 10 | 0.890 ± 0.029 (0.870) | 0.728 ± 0.065 (0.766) | 0.551 ± 0.065 (0.655) | 0.332 ± 0.052 (0.546) | 0.257 ± 0.054 (0.444) | 2.18 | 0.231 |
| 40 | 10 | 0.897 ± 0.028 (0.881) | 0.766 ± 0.051 (0.784) | 0.590 ± 0.061 (0.679) | 0.369 ± 0.048 (0.575) | 0.301 ± 0.065 (0.476) | 2.26 | 0.267 |
| 45 (4 seam qubits) | 10 / 10 / 9 / 9 / 5 | 0.959 ± 0.013 (0.864) | 0.802 ± 0.041 (0.755) | 0.759 ± 0.044 (0.640) | 0.450 ± 0.030 (0.528) | 0.385 ± 0.061 (0.425) | 2.34 | 0.374 |
| 49 (4 seam qubits) | 10 / 9 / 8 / 8 | 0.858 ± 0.038 (0.872) | 0.762 ± 0.044 (0.769) | 0.595 ± 0.067 (0.660) | 0.387 ± 0.040 (0.551) | running | 2.44 (d = 4–10) | 0.235 |
| 53 (5 seam qubits) | 10 / 8 / 7 | 0.935 ± 0.014 (0.857) | 0.740 ± 0.059 (0.745) | 0.657 ± 0.056 (0.626) | running | | 1.15 (d = 4–8, not yet meaningful) | 0.512 |
| 57 (5 seam qubits) | 10 / 1 | 0.902 ± 0.021 (0.865) | 0.661 (0.757), 1 circuit | running | | | | |
| 61 (`nighthawk61r2`, 6 seam qubits) | 3 / 1 | 0.927 ± 0.011 (0.853) | 0.730 (0.737), 1 circuit | running | | | | |
| **Dan's calculator** | | | | | | | 1.606 | 0.235–0.476 |

| Measured / Dan, Hamming signal | 16 | 20 | 24 | 28 | 32 | 36 | 40 | 45 | 49 | 53 | 57 | 61 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d = 4 | 1.27 | 1.10 | 1.11 | 1.04 | 1.05 | 1.02 | 1.02 | 1.11 | 0.98 | 1.09 | 1.04 | 1.09 (3 circuits) |
| d = 6 | 1.33 | 1.07 | 1.18 | 0.93 | 0.86 | 0.95 | 0.98 | 1.06 | 0.99 | 0.99 | 0.87 (1 circuit) | 0.99 (1 circuit) |
| d = 8 | 1.22 | 1.22 | 1.44 | 0.90 | 0.86 | 0.84 | 0.87 | 1.19 | 0.90 | 1.05 (7 circuits) | — | — |
| d = 10 | 1.46 | 1.12 | 1.25 | 0.57 | 0.60 | 0.61 | 0.64 | 0.85 | 0.70 | — | — | — |
| d = 12 | 1.23 | 0.96 | 1.04 | 0.37 | 0.51 | 0.58 | 0.63 | 0.91 (5 circuits) | — | — | — | — |
| Dan at the mirror's 2d layers instead, d = 12 | 0.0121 | 0.0219 | 0.0344 | 0.0492 | 0.0660 | 0.0846 | 0.1046 | 0.0739 | 0.0882 | 0.0651 | 0.0760 | 0.0595 |

Coupler load: couplers through the boundary per seam qubit, sorted. Dan's strips, on which his calculator was fitted, have 12 couplers on 4 seam qubits (3.0).

| Measured / Dan, by coupler load | 16 | 24 | 20 | 45 | 28 | 32 | 36 | 40 |
|---|---|---|---|---|---|---|---|---|
| seam qubits | 3 | 3 | 3 | 4 | 3 | 3 | 3 | 3 |
| couplers through the boundary | 6 | 8 | 9 | 12 | 11 | 11 | 12 | 12 |
| couplers per seam qubit | 2.0 | 2.7 | 3.0 | 3.0 | 3.7 | 3.7 | 4.0 | 4.0 |
| Hamming signal, d = 8 | 1.22 | 1.44 | 1.22 | 1.19 | 0.90 | 0.86 | 0.84 | 0.87 |
| Hamming signal, d = 10 | 1.46 | 1.25 | 1.12 | 0.85 | 0.57 | 0.60 | 0.61 | 0.64 |
| Hamming signal, d = 12 | 1.23 | 1.04 | 0.96 | 0.91 | 0.37 | 0.51 | 0.58 | 0.63 |
| forward XEB (E17), d = 12 | 1.03 | 0.68 | 0.70 | — | 0.40 | — | — | — |

Prediction from this split, recorded 2026-10-19 with only 49 at d = 10 measured (8 circuits):

| Register | Seam qubits | Couplers | Per seam qubit | Predicted measured / Dan, d = 10–12 | Measured |
|---|---|---|---|---|---|
| `2p49` | 4 | 15 | 3.75 | 0.4–0.65 | d = 8: 0.90; d = 10: 0.70 ± 0.07 (8 circuits) |
| `2p53` | 5 | 15 | 3.0 | 0.85–1.2 | d = 8: 1.05 ± 0.09 (7 circuits; 3 lost to NaN) |
| `2p57` | 5 | 18 | 3.6 | 0.4–0.65 | — |
| `61r2` | 6 | 19 | 3.17 | between the two groups: locates the threshold | d = 4: 1.09 (3 circuits); d = 6: 0.99 (1 circuit) |

| Same registers, container (6 circuits; 4 at 40) vs EPYC (10 circuits), d = 12 | 36 | 40 |
|---|---|---|
| container | 0.157 ± 0.012 | 0.202 ± 0.050 |
| EPYC | 0.257 ± 0.054 | 0.301 ± 0.065 |
| circuit-to-circuit spread (std) on the EPYC | 0.17 | 0.21 |

| d = 12, where both exist | 16 | 20 | 24 | 28 |
|---|---|---|---|---|
| forward XEB, E17 | 0.241 | 0.200 | 0.224 | 0.148 |
| Hamming signal, E18 | 0.288 | 0.275 | 0.344 | 0.137 |
| ratio | 1.20 | 1.38 | 1.54 | 0.93 |

Folding, signal at λ = 1 / 3 / 5 and Dan's Richardson extrapolation (ln signal in u = λ^1.60569, to λ = 0):

| n | d = 4 | d = 8 | d = 12 |
|---|---|---|---|
| 16 | 0.990 / 0.977 / 0.977 → 0.994 | 0.571 / 0.635 / 0.628 → 0.553 | 0.288 / 0.353 / 0.273 → 0.267 |
| 20 | 0.887 / 0.933 / 0.884 → 0.871 | 0.633 / 0.520 / 0.569 → 0.677 | 0.275 / 0.118 / 0.103 → 0.352 |
| 24 | 0.914 / 0.934 / 0.890 → 0.905 | 0.806 / 0.793 / 0.806 → 0.811 | 0.344 / 0.457 / 0.365 → 0.311 |
| 28 | 0.882 / 0.901 / 0.884 → 0.875 | 0.539 / 0.589 / 0.521 → 0.521 | 0.137 / 0.165 / 0.166 → 0.129 |
| 32 | 0.904 / 0.919 / 0.916 → 0.899 | 0.537 / 0.530 / 0.514 → 0.539 | 0.210 / 0.157 / 0.156 → 0.229 |
| 36 (EPYC) | 0.890 / 0.907 / 0.902 → 0.885 | 0.551 / 0.501 / 0.527 → 0.570 | 0.257 / 0.170 / 0.185 → 0.294 |
| 40 (EPYC) | 0.897 / 0.914 / 0.920 → 0.892 | 0.590 / 0.555 / 0.484 → 0.595 | 0.301 / 0.173 / 0.127 → 0.350 |
| 45 (EPYC) | 0.959 / 0.956 / 0.951 → 0.960 | 0.759 / 0.668 / 0.637 → 0.787 | 0.385 / 0.222 / 0.203 → 0.453 (5 / 10 / 6 circuits) |
| 49 (EPYC) | 0.858 / 0.809 / 0.906 → 0.880 | 0.595 / 0.541 / 0.551 → 0.613 (8 / 7 / 7 circuits) | running |
| 53 (EPYC) | 0.935 / 0.922 / 0.932 → 0.940 | λ = 1 only: 0.657 (7 circuits); d = 6: 0.740 / 0.733 / 0.688 → 0.739 (8 / 7 / 6 circuits) | running |
| 57 (EPYC) | 0.902 / 0.916 / 0.920 → 0.897 | running | running |
| 61 (EPYC) | λ = 1 only: 0.927 (3 circuits) | running | running |

| Effective folding exponent q, s(λ) = s(1)^(λ^q), EPYC | d = 4–6 | d = 8 | d = 10 | d = 12 |
|---|---|---|---|---|
| 36 | −0.16 – 0.0 | 0.04–0.13 | 0.06–0.08 | 0.13–0.24 |
| 40 | −0.17 – 0.03 | 0.10–0.20 | 0.14–0.16 | 0.34–0.35 |
| 45 | 0.02–0.11 | 0.31–0.35 | 0.26–0.27 | 0.32–0.41 |
| his gain exponent | 1.606 | 1.606 | 1.606 | 1.606 |

| Seconds per circuit, container CPU | 16 | 20 | 24 | 28 | 32 | 36 | 40 |
|---|---|---|---|---|---|---|---|
| ACE simulator widths | 17 / 17 / 5 | 19 / 19 / 5 | 21 / 21 / 7 / 5 | 22 / 22 / 4 | 25 / 25 / 5 | 29 / 28 / 6 | 28 / 28 / 4 |
| d = 12, λ = 1 | 0.1 | 0.2 | 0.6 | 2.7 | 5.8 | 25 | 108 |
| d = 12, λ = 5 | 0.3 | 1.3 | 3.4 | 14 | 33 | 145 | — |

| `nighthawk61r2` in this protocol | Value |
|---|---|
| planner's estimate per point (error detection on) | ≈ 385 GiB: `--swap-gb 238 --mem-budget-gb 530` |
| Dan's calculator, d = 4 / 6 / 8 / 10 / 12 | 0.853 / 0.737 / 0.617 / 0.501 / 0.396 |
| measured, λ = 1 | d = 4: 0.927 ± 0.011 (3 circuits), 1.09×; d = 6: 0.730 (1 circuit), 0.99× |

- **His Hamming weight reaches where his XEB cannot.** It needs no 2ⁿ reference; at d = 12 it tracks the E17 forward XEB within 0.93–1.54× on the same registers.
- **The ladder conforms to his calculator at d ≤ 6 at every size** (0.86–1.33×), and at all depths at 16–24 qubits. **From 28 qubits on it falls below with depth**: 0.37–0.63× at d = 12 at 28–40 qubits (3-qubit seam), while the calculator keeps rising with the bulk. **At 45 qubits (4-qubit seam) it is back within 0.85–1.19× at every depth** (d = 12 on 5 of 10 circuits). The fitted depth power is 1.8–2.6 against his 1.606.
- **Coupler load, not size, separates the two groups at d ≥ 10.** The four registers at ≤ 3.0 couplers per seam qubit (16, 20, 24, 45; his strips are 3.0) stay at 0.85–1.46×; the four at 3.7–4.0 (28–40) sit at 0.37–0.64×. 24 → 28 qubits keeps the 3-qubit seam and adds 3 couplers, and the ratio at d = 12 drops from 1.04 to 0.37. The threshold was read off these eight registers (a perfect split by chance: 1 in 70), so 49, 53 and 57 are the test. The forward XEB of E17 splits less cleanly (20 and 24 at 0.70 and 0.68), and 28 and 32 rest on 6 container circuits, which sat ≈ 0.1 low at 36 and 40.
- **Ten circuits are the minimum per point.** The circuit-to-circuit spread at d = 12 is 0.17–0.21 (std); with 4–6 circuits the container's 36- and 40-qubit points sat 0.10 below the EPYC's 10-circuit means.
- **Folding barely amplifies ACE's error.** At d ≤ 6 the signal does not change with λ; at d ≥ 8 it falls with an effective exponent q = 0.04–0.41, against his gain exponent 1.606, which would give s^(3^1.606) ≈ s^5.8 (0.0095 instead of the measured 0.342 at 45 qubits, d = 10). His Richardson step therefore lands within −1 % to +16 % of the λ = 1 value: the exponent describes hardware noise, which grows with every folded gate, and ACE's seam error grows far more slowly.

### 10.2 VQE on the two-patch ladder (E19)

`run --families vqe` (METHODOLOGY §6.5): transverse-field Ising, J = 1, h = 3.04, on each register's couplers; Hamiltonian-variational circuit with p layers, angles optimised once in numpy on `nighthawk2p16` and used unchanged at every size; energies shot-free on ACE (error detection on), container CPU. Energies per site; the product state \|+⟩ⁿ has −3.04.

| n | p | ACE | same circuit, numpy | ACE − circuit | E₀ (Lanczos) | Anderson bound | correlation energy: ACE / circuit | circuit / E₀ |
|---|---|---|---|---|---|---|---|---|
| 16 | 1 | −3.13613 | −3.13897 | +0.00284 | −3.14831 | −3.29948 | 0.971 | 0.914 |
| 16 | 2 | −3.11392 | −3.14725 | +0.03333 | | | 0.689 | 0.990 |
| 16 | 3 | −3.14346 | −3.14796 | +0.00451 | | | 0.958 | 0.997 |
| 16 | 4 | −3.11600 | −3.14823 | +0.03222 | | | 0.702 | 0.999 |
| 20 | 1 | −3.12996 | −3.15015 | +0.02019 | −3.16385 | −3.36385 | 0.817 | 0.889 |
| 20 | 2 | −3.01539 | −3.16128 | +0.14589 | | | −0.203 | 0.979 |
| 20 | 3 | −3.00150 | −3.16280 | +0.16130 | | | −0.314 | 0.992 |
| 20 | 4 | −2.91542 | −3.16346 | +0.24803 | | | −1.009 | 0.997 |
| 24 | 1 | −3.12793 | −3.15451 | +0.02658 | −3.16974 | −3.38940 | 0.768 | 0.883 |
| 24 | 2 | −3.14761 | −3.16674 | +0.01913 | | | 0.849 | 0.977 |
| 24 | 3 | −3.09706 | −3.16852 | +0.07147 | | | 0.444 | 0.991 |
| 24 | 4 | −3.07928 | −3.16928 | +0.09001 | | | 0.304 | 0.997 |
| 28 | 1 / 2 / 3 / 4 | −3.12312 / −3.11716 / −3.05474 / −2.98625 | above the container's 26-qubit numpy limit | | | −3.42253 | | |

Where the error sits, p = 4 (mean over the seam's and the bulk's qubits and couplers, ACE against the numpy state):

| Register | ⟨X⟩ bulk | ⟨X⟩ seam qubits | ⟨ZZ⟩ bulk couplers | ⟨ZZ⟩ seam couplers |
|---|---|---|---|---|
| `nighthawk2p16` (seam q4, q10, q15) | 0.955 / 0.959 | **0.834 / 0.968** | 0.158 / 0.182 | **0.288 / 0.183** |
| `nighthawk2p20` (q3, q10, q17) | 0.933 / 0.955 | **0.677 / 0.941** | 0.185 / 0.187 | **0.245 / 0.198** |
| `nighthawk2p24` (q11, q18, q23) | 0.921 / 0.949 | **0.732 / 0.954** | 0.197 / 0.193 | **0.286 / 0.193** |

- **The angles transfer.** Optimised on 16 qubits, the same circuit captures 88–99.7 % of the exact correlation energy at 20 and 24 qubits (circuit / E₀), rising with p: the circuit family is sound at every size the numpy reference reaches.
- **ACE keeps 70–97 % of the circuit's correlation energy at 16 qubits, and loses it with depth from 20 on.** At 20 qubits, p ≥ 2, its energy is above the product state's; at 24 it keeps 30–85 %. More layers make it worse, not better.
- **The seam acts like local dephasing.** The seam qubits' ⟨X⟩ falls to 0.68–0.83 (exact 0.94–0.97) while ⟨ZZ⟩ on the seam couplers rises to 0.24–0.29 (exact 0.18–0.20): coherence at the seam turns into classical correlation. The bulk stays within 0.03. Every energy stays above E₀ and above the bound (no `below_e0`, no `below_bound`): what ACE returns is consistent with a physical, partly decohered state, not an inconsistent one.
- **This observable separates registers differently from the Hamming signal.** `nighthawk2p20` (3.0 couplers per seam qubit) follows Dan's calculator on the Hamming signal but is the worst here: each layer puts two CX through every seam coupler, plus one more in the readout, so the seam's share of the circuit is larger than in Dan's gate set.

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
| `hamming` on GPU, `nighthawk2p45`, six V340 dies × 3 workers (`--ace-gpu`) | 4–9 W per 100 W card: launch and read-back latency and PCIe traffic dominate; reverted to CPU. The 45-qubit file from that attempt held NaN signals (in 4 of its 13 depth × fold groups) and 9 failed points (worker deaths, C++ exceptions): rerun on CPU |
| memory in use, `hamming` at 57 qubits, d = 4–6 (EPYC) | ≈ 25 GB per worker, against the planner's worst case of ≈ 81 GiB (patches 2³² × 8 B each fully entangled, plus the compose transient): shallow circuits leave the patches partly separable |
| `hamming` (selftest [11]) | exact backend signal 1.000000 at λ = 1, 3; each of the twelve gates' inverses within 1.2 × 10⁻⁷ of identity |
| `fxeb` on a register | runs from n ≤ 36 (`nighthawk2p28`: 104 s at d = 4, 464 s at d = 12 incl. reference) |
| existing configurations after each change | tags unchanged; selftest ALL OK |

---

## 14. Why Dan's calculator and our numbers differ (E16)

Dan's calculator (§10) is fitted on `nn_qab.py`; our ladder ran `nighthawk_qrack.py --theta nnqab`. The two procedures differ in these ways:

| Factor | `nn_qab.py` (Dan) | `nighthawk_qrack.py --theta nnqab` (ours) | Tested |
|---|---|---|---|
| observable | forward XEB, 8192 shots | mirror survival (ladder); forward XEB (`fxeb`), 4096 shots | yes |
| two-qubit gates | uniformly random from 12 Cliffords (SWAP, CZ·SWAP, CZ·iSWAP, CZ·iSWAP†, iSWAP, iSWAP†, CX, CY, CZ and the three anti-controlled); SWAP, CZ·iSWAP and CZ·iSWAP† do not entangle | CZ only | yes |
| lattice | `factor_width` strip (10 → 5 × 2, 22 → 11 × 2, 26 → 13 × 2), torus brick pattern, sequence 0 3 2 1 2 1 0 3 | Nighthawk square lattice, colours A B C D, open edges; mirrors pseudo-patched | yes (both lattices measured) |
| ACE register | ACE's own geometry, `is_torus` default (on), error detection on | stored two-patch placement, flat, error detection on | — |
| single-qubit gates | `u(θ, φ, λ)`, sin θ uniform, φ, λ uniform | rz(φ) rx(θ) rz(λ), sin θ uniform, φ, λ uniform: the same distribution | by construction |
| XEB formula | Σ_b (p_b − 1/N) q_b / Σ_b (p_b − 1/N)² | (N · mean_s p − 1) / (N Σp² − 1): algebraically identical | by construction |
| depth | d forward layers | mirror: d/2 forward + d/2 inverse | — |
| PyQrack | 2.28.2 | 2.28.1 | yes |
| circuits per point | 200, unseeded | 3 instances (fxeb), 3 × 10 runs (mirror) | — |

Reproduction of his sheet with his procedure on our PyQrack (d = 12, 20 circuits):

| Series | his sheet (200 circuits) | ours |
|---|---|---|
| 10 qubits, lrc 2 lrr 2 | 0.0939 | 0.085 ± 0.040 |
| 14 qubits, lrc 3 lrr 2 | 0.1587 | 0.168 ± 0.016 |
| 22 qubits, lrc 5 lrr 2 | 0.2547 | 0.255 ± 0.015 |
| 14 qubits, d = 20 | 0.0143 | 0.017 ± 0.004 |
| 22 qubits, d = 20 | 0.0360 | 0.046 ± 0.008 |

The "cut sampler" draws each patch from its exact marginal, independently of the other patch, with seam qubits assigned to their first patch. Its XEB against the ideal distribution is computed exactly. It is what XEB scores when every correlation across the seam is lost and nothing else is wrong.

Dan's circuits on his strips (20 circuits per cell; 8 at 26 qubits; mirror twirled with the exact Clifford conjugation, d/2 + d/2; exact backend gives survival 1.000):

| n | d | Dan's calculator | XEB, his gates | cut-sampler XEB | XEB, CZ only | mirror, his gates | mirror, CZ only |
|---|---|---|---|---|---|---|---|
| 10 | 12 | 0.098 | 0.085 ± 0.040 | 0.370 ± 0.015 | 0.537 ± 0.059 | 0.018 ± 0.007 | 0.608 ± 0.078 |
| 10 | 20 | 0.005 | −0.028 ± 0.012 | 0.273 ± 0.007 | 0.127 ± 0.018 | 0.005 ± 0.003 | — |
| 14 | 12 | 0.153 | 0.168 ± 0.016 | 0.286 ± 0.019 | 0.549 ± 0.036 | 0.013 ± 0.004 | 0.103 ± 0.033 |
| 14 | 20 | 0.014 | 0.017 ± 0.004 | 0.088 ± 0.003 | 0.225 ± 0.025 | < 10⁻³ | — |
| 22 | 12 | 0.241 | 0.255 ± 0.015 | 0.271 ± 0.013 | 0.558 ± 0.025 | 0.007 ± 0.003 | 0.077 ± 0.020 |
| 22 | 20 | 0.040 | 0.046 ± 0.008 | 0.048 ± 0.006 | 0.297 ± 0.018 | < 10⁻³ | — |
| 26 | 12 | 0.279 | 0.253 ± 0.022 | — | — | 0.004 ± 0.002 | — |

Our Nighthawk two-patch registers, 3 seam qubits each (release prefixes; `nn_qab` θ; fxeb 3 instances × 4096 shots; mirror 3 × 10 runs; cut sampler on the same forward circuits):

| Register | n | d | Dan's calculator | fxeb | cut-sampler XEB | mirror |
|---|---|---|---|---|---|---|
| `release:16`, 2 patches 10 / 11 | 16 | 12 | 0.235 | 0.455 ± 0.035 | 0.465 ± 0.017 | 1.000 ± 0.000 |
| | 16 | 20 | 0.037 | 0.341 ± 0.016 | 0.338 ± 0.034 | 0.910 ± 0.044 |
| `release:20`, 13 / 12 | 20 | 12 | 0.285 | 0.451 ± 0.052 | 0.514 ± 0.008 | 0.652 ± 0.030 |
| | 20 | 20 | 0.058 | 0.256 ± 0.070 | 0.301 ± 0.033 | 0.355 ± 0.055 |
| `release:24`, 16 / 12 | 24 | 12 | 0.330 | 0.642 ± 0.173 | 0.632 ± 0.075 | 0.386 ± 0.028 |
| | 24 | 20 | 0.081 | 0.172 ± 0.009 | 0.329 ± 0.031 | 0.134 ± 0.045 |
| `nighthawk2p28` | 28 | 12 | 0.372 | 0.539 ± 0.194 | 0.404 ± 0.027 | 0.462 ± 0.054 |
| | 28 | 20 | 0.106 | 0.219 ± 0.052 | 0.253 ± 0.055 | 0.058 ± 0.048 |

| Same circuit, fxeb / cut-sampler XEB | instance 0 | instance 1 | instance 2 |
|---|---|---|---|
| 16 qubits, d = 12 | 0.430 / 0.437 | 0.523 / 0.497 | 0.410 / 0.461 |
| 16 qubits, d = 20 | 0.328 / 0.333 | 0.320 / 0.400 | 0.373 / 0.281 |
| 20 qubits, d = 12 | 0.505 / 0.511 | 0.500 / 0.529 | 0.348 / 0.501 |
| 20 qubits, d = 20 | 0.275 / 0.298 | 0.368 / 0.360 | 0.127 / 0.247 |
| 24 qubits, d = 12 | 0.612 / 0.655 | 0.955 / 0.750 | 0.359 / 0.491 |
| 28 qubits, d = 12 | 0.338 / 0.455 | 0.927 / 0.393 | 0.352 / 0.364 |

| Factor | Effect on the delta | Evidence |
|---|---|---|
| **observable** | **dominant.** ACE's forward XEB sits at the cut-sampler level, so it cannot see whether correlations survive the seam. It stays flat or rises as the bulk grows. Mirror survival falls with size on both lattices | his strips: XEB 0.085 → 0.253, mirror 0.018 → 0.004 (n = 10 → 26, d = 12); our registers: fxeb ≈ cut-sampler XEB at every size, mirror 1.000 → 0.462 (n = 16 → 28) |
| gate set | shifts the level. His 12-gate Clifford set lowers ACE's XEB 2–6× and its mirror survival 10–30× against CZ only on the same lattice. Our CZ-only fxeb therefore lies above his calculator at 16–28 qubits | his strips, CZ-only columns |
| lattice and register | secondary. The same XEB-vs-mirror split appears on his torus strips and on our open Nighthawk patches | both tables |
| PyQrack version, environment | none | sheet reproduced within 1σ |
| XEB formula, single-qubit ensemble | none | identical by construction |

- **The calculator's variable is a weak-link variable.** x = (bulk / (boundary · patches))^(2/3) measures how weak the cut is relative to the bulk. XEB that equals the cut sampler's XEB rises with it. This is the mechanism by which a sampler that drops weak links scores on XEB (Gao et al., README ref. [5]). So the calculator predicts ACE's XEB correctly, but at these sizes that XEB does not measure ACE's fidelity across the seam.
- **Mirror survival cannot be passed by a cut sampler.** It falls with size at a fixed 3-qubit seam, and it is the proxy `ibm_phoenix` is compared on at 61 qubits. The ladder's gap to the calculator above 28 qubits is the gap between XEB and survival.
- **Neither proxy is the fidelity itself.** At 16 qubits survival is 1.000 while XEB is 0.455: a mirror absorbs a few boundary CZs exactly (as `nighthawk61w` at d = 4–6). At larger sizes it is the stricter of the two.

## 15. Not yet measured

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
| Hamming ladder n = 45–57 and `nighthawk61r2` (E18) | EPYC: README §7 recipe; 57 needs 64 GiB patches, 61 RAM + swap (≈ 385 GiB) |
| VQE ladder n = 28–61 (E19) | EPYC: README §7 recipe; numpy circuit reference to 28–30 there (`--vqe-exact-max`, 4–16 GiB), Lanczos E₀ to 26–28 |
| Hamming ladder 16–32 with 10 circuits per point | 6 measured in the container (36–45 have 10 on the EPYC) |
| Hamming, 45 qubits: 9 failed points (worker deaths, C++ exceptions) and 4 NaN records | rerun with `--retry-failed` |
| Hamming, C++ exceptions and worker deaths still open | 36: d = 10 λ = 5 i8 (2 workers died); 49: d = 8 λ = 5 i8, d = 10 λ = 1 i2, λ = 3 i1 and i3, λ = 5 i0; 57: d = 6 λ = 1 i1. Retryable with `--retry-failed`; NaN points may also pass on a retry |
| Hamming, 49 qubits, instance 4 at d = 6 | ACE returns P(q = 1) = NaN on two bulk qubits of patch 1 (q45, q47) at λ = 1, 3 and 5; reproduced on a second machine (PyQrack 2.28.1, CPU): a property of this circuit on ACE, not of a device or a run. Excluded; the d = 6 points have 9 circuits |
| Hamming, NaN circuits at 45–57 qubits | 45: instance 2 (d = 8, 10); 49: instance 4 (d = 6–10, all folds), 5 and 7 (d = 8); 53 at d = 6: instances 1, 6 (λ = 1), 1, 5, 6 (λ = 3), 1, 2, 5, 6 (λ = 5), at d = 8: 1 (q48), 6 (NaN on 10 of 53 qubits, from q31: the first to spread past the seam's neighbourhood) and 8 (q48) at λ = 1, 1 (q45) at λ = 3; 57 at d = 6: instances 2, 4 (λ = 1). The NaN always appears on patch 1's bulk or on the lowest seam qubit (45: q38, q40; 49: q45, q47; 53: q45, q48; 57: q51, q55). Not every NaN is permanent: 53 d = 6 λ = 3 instances 3 and 4 failed in an earlier pass and have records now; 49 instance 4 fails on two machines. Instances 1 and 6 of 53 fail at every depth and fold measured so far: a property of those circuits. A small surviving set can move a lot: 53, d = 6, λ = 5 read 0.591 on 3 circuits and 0.688 on 6; 53, d = 8 read 0.581 on 4 and 0.657 on 7. Points with exclusions carry their circuit count and are not used in fits |
| `nighthawk88d` / `88e` depth sweep | six V340 dies |
| tiling v3 vs v2 at n = 30–32 | CPU path |
| host-mapped (`--gtt`) patch speed on the V340s | not measured |

Runtime screenshots:

<img width="970" height="403" alt="image" src="https://github.com/user-attachments/assets/717d8289-2a12-4de4-892d-77d5a161f955" />
<img width="1114" height="327" alt="image" src="https://github.com/user-attachments/assets/b84b32ce-4fae-434b-b06a-62b38d32c4b2" />
