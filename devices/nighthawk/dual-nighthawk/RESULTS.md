# Results: `rcs/nighthawk`

Every measured number of the experiments, grouped by experiment. The procedure behind each is in [METHODOLOGY.md](METHODOLOGY.md) (experiment IDs E1–E36 in its §9); usage is in the [README](README.md). State as of 29 October 2026.

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
| [10.3](#103-shors-9-qubit-code-across-a-seam-e20) | E20: Shor's code (9 to 36 qubits) on a two-patch ACE register, and tiled over 61 and 112 qubits | outer layer across the seam: logical Z 1.000, X 0.49; inner blocks across it: Z 0.50, X 0.67; tiled: decided by each unit's replica CXs (Z 0.73, X 0.44 at seven); in-patch units lose 2–7 % of X in ACE's sampler |
| [10.4](#104-ghz-states-across-the-seams-e21) | E21: GHZ Layer 1a on two-patch registers (16, 28, 57, 61 qubits), chip-wide and tiled | every 3-qubit unit at distance ≥ 1 from a seam qubit: F_GHZ 1.000; units holding one: 0.77 (1 seam CX), 0.43 (2); chip-wide the coherence does not cross the seam; fold:1 changes 0.638 → 0.622 |
| [10.5](#105-layer-2-across-the-seams-walsh-echo-compensation-e22) | E22: the formal proof's Layer 2 (Walsh echo, compensation) on the same tiles | off the seam ACE = exact in every mode, the echo restores 1.000; on the seam every mode costs (61: 0.773 → 0.671 dd, 0.426 → 0.240 on 2-seam-CX units) |
| [10.6](#106-seam-error-extraction-and-zne-e23) | E23: seam error extracted from passes 1–2, ZNE over seam scales λ = 1, 5, 9, a fresh Layer 2 check | the first crossing keeps 0.39 of the coherence on 28 and 61 qubits; no amplifier scales the loss, so ZNE returns pass 2 (−0.02 … +0.01), less than a rerun moves it (0.04–0.06) |
| [10.7](#107-seam-error-correction-syndrome-replica-repetition-code-the-third-measurement-e24) | E24: seam syndrome from passes 1–2, the replica repetition code with its schedule, the corrected third measurement | every unit one simulator holds is corrected to the exact reference (28: 0.720 → 1.001; 61: 0.683 → 0.999, 0.481 → 1.002), +0.28–0.34 against a rerun's 0.09; units across two simulators keep no coherence (≤ 0.06), F ≤ ½ |
| [10.8](#108-pyqracks-ace-source-the-authority-rule-the-crossbar-the-gadget-e25) | E25: PyQrack's lineage authority, a co-located variant, the crossbar, the error-detection gadget | the lineage rule strands a seam partner's authority on the crossbar ([48, 40, 56] 0.463 in the preparation's order); the co-located step corrects every replica unit in any order (1.000–1.003); seam-to-seam GHZ 0.99–1.01 through the crossbar; the gadget does not make the one-sided loss |
| [10.9](#109-the-code-before-the-echo-e26) | E26: the stack with Layer S before Layer 2 (`--order code-echo`) | the code takes every replica unit to the exact value without the drift (0.770 → 1.000) and the echo keeps it there (1.000); the cut units stay at the rerun level (0.357), then lose to the drift (0.216) |
| [10.10](#1010-the-walsh-echo-window-by-window-e27) | E27: GHZ fidelity against the number of Walsh-echo windows, by seam class | off the seam 1.000 at every L; replica units step down over the first two windows (0.78 → 0.72) and stay flat to L = 8; cut units fall in the first window to near the scrambled floor; the code at L = 8: replica 0.999 |
| [10.11](#1011-protecting-the-cut-units-from-layer-2-e28) | E28: how the drift's inside ZZ is put through ACE; `--inside-zz stabilizer` | dropping the inside ZZ (a global phase on the GHZ) keeps the cut units at their no-drift level (0.353 against 0.205 as gates) and removes the replica units' echo dip (0.764 against 0.683); the ½ ceiling stays |
| [10.12](#1012-the-loom-as-the-seam-ghost-band-e29) | E29: `nighthawk61r2`'s qubits as two exact chips joined by the cloverfield's ghost loom | every 3-chain across a link lies in one chip's ghost band and reads 1.000 there, 0.265 on its own chips; the ghost band moves the cut, it does not correct across it (§10.14) |
| [10.13](#1013-two-patch-ace-chips-joined-by-the-loom-ghost-band-e30) | E30: two `nighthawk61r2` cloverleaves joined by the ghost loom, 122 qubits | ACE's code takes ACE's replica units to 1.00; the loom's 3-chains read 1.00 inside a ghost band (§10.14); the 14 ACE-cut units fall to 0.21 under the echo, 0.36 with seam protection |
| [10.14](#1014-the-loom-as-a-hard-cut-and-what-the-ghost-band-does-e31) | E31: the loom as a hard cut; the ghost band explained | across a hard cut every GHZ unit sits at the separable bound 0.50 in every pass, code, echo and protection alike (sd 0.008); the ghost loom's 1.00 is a two-column replica band simulating a unit whole on one chip, and 5-chains wider than the band stay at 0.26 |
| [10.15](#1015-qrackbridge-in-place-of-the-loom-e32) | E32: the two chips joined by PyQrack's QrackBridge instead of the loom, `max_bonds` 6 → 0 | the bridge carries entanglement: exact at 6 (every unit across it 1.00); below the demand it carries max_bonds whole units per circuit and leaves the rest separable (0.50): means 0.85 / 0.74 at 2 / 1; at 0 it is the hard cut (0.51) |
| [10.16](#1016-a-closed-box-six-faces-on-a-qrackbridge-ring-e33) | E33: a closed 5 × 5 × 5 box, base, four walls and lid as six QrackBridge patches on a ring, `max_bonds` 6 → 0 | every box edge carries GHZ chains: exact at capacity, relayed through a third face or not; below it the inner cuts ration whole units as two chips do (0.75 / 0.63 at 2 / 1 against budgets 0.750 / 0.625), the ring's wrap cut falls short (0.67 / 0.61) and at 0–1 with bridged drift gates the ring zeroed the state |
| [10.17](#1017-shors-code-across-qrackbridge-e34) | E34: Shor's [[m², 1, m]] code instead of GHZ across QrackBridge, Shor-9 on two chips and the box, Shor-16 and -25 across the chips, `max_bonds` 6 → 0 | a cut between the code's blocks needs one bond (for \|±_L⟩), a cut through them one per block it splits (for \|0/1_L⟩); short of that the bridge drops whole blocks and the decoder repairs them as a majority vote predicts (Shor-9 at 1 bond 0.745 against 0.750, Shor-16 at 2 / 1 0.874 / 0.688 against 0.875 / 0.688); at 0 it gives E20's ACE-seam pattern on the side that needs coherence |
| [10.18](#1018-ibms-two-module-nighthawk-r1-on-the-bridge-e35) | E35: IBM's two Nighthawk R1 modules and four L-couplers (arXiv:2610.29986) on QrackBridge, its Bell and GHZ states under four models against its Table 1 | the hard cut caps every state at 0.50; the benchmarked distant CZ predicts the readout-mitigated Bell 115–233 and GHZ-4/6 to 1.5–3 points (0.954 / 0.944 / 0.938 against 0.939 / 0.917 / 0.909); Bell 117–231 and the simultaneous pairs fall outside the gate model |
| [10.19](#1019-layer-3-shor-order-finding-across-the-configurations-e36) | E36: Layer 3, the paper's Shor order finding (N 15, a 7) and N 21 at orders 2, 3, 6, as gates across QrackBridge (two chips, the box), and on the dual Nighthawk R1 | a cut between the registers needs ⌈log₂ r⌉ + 1 bonds (4 for r = 6, 3 for r = 3; exact from there on the chips and at every box placement); a power-of-two order needs none: the paper's N = 15 gives its ideal ½ through a hard cut, so it cannot show a link; on the dual R1 with the benchmarked distant CZ 0.414 against ½ (t 4), N 21 at the uniform guess |
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

### 10.3 Shor's 9-qubit code across a seam (E20)

`shor9 --shots 1024 --repeats 6` (METHODOLOGY §6.6): 3 × 3 ACE grid, `long_range_columns = 1`, flat; seam column q1, q4, q7 (ACE's constructor defaults otherwise); container CPU. Success rate raw / decoded, mean over 6 fresh backends per logical state; shot noise alone ± 0.011 per logical basis.

| Layout | Crosses the seam | \|0_L⟩ | \|1_L⟩ | \|+_L⟩ | \|−_L⟩ | logical Z (decoded) | logical X (decoded) |
|---|---|---|---|---|---|---|---|
| `seam_leaders`, blocks [1, 0, 2] [4, 5, 3] [7, 6, 8] | every inner bit-flip block | 0.505 / 0.503 | 0.498 / 0.495 | 0.612 / 0.691 | 0.487 / 0.647 | 0.499 (0.448–0.532) | 0.669 (0.631–0.700) |
| `patch_blocks`, blocks [4, 1, 7] [0, 5, 6] [2, 3, 8] | the outer phase-flip layer only | 0.925 / 1.000 | 0.893 / 1.000 | 0.477 / 0.496 | 0.491 / 0.477 | 1.000 (1.000–1.000) | 0.486 (0.402–0.544) |
| exact `QrackSimulator` | — | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 | 1.000 |

(Ranges in brackets: backend to backend. A single 2048-shot run gave `seam_leaders` X 0.686 and 0.534 on two occasions, and `patch_blocks` \|0_L⟩ raw 0.779 and 0.634.)

- **What a CX through the seam creates is lost; what lives inside one region survives.** In `patch_blocks` each block's sign is local to one region, so logical Z decodes to 1.000 every time, while the parity tying the three blocks together was made by the outer CXs through the seam: logical X 0.49, no information. In `seam_leaders` each block's GHZ phase was made by CXs through the seam: logical Z 0.50; the outer layer is seam-to-seam, which ACE applies exactly, and logical X keeps 0.67.
- **Decoding cannot repair it.** The code corrects one error per block; the seam removes a correlation that spans whole blocks.
- **ACE's seam is not deterministic.** The same circuit on a fresh backend gives a logical success that ranges over 0.07–0.14 (except `patch_blocks` Z, always 1.000), 6–13× the shot noise; a single run is not a measurement.

Units tiled over whole registers (`shor9 --register`), decoded success; every encoder CX a coupler of the register; exact reference 1.000 for every unit and state on both:

| Register, run | Unit | Rows × cols | Regions | Encoder CX exact / replica / cross | Logical Z | Logical X |
|---|---|---|---|---|---|---|
| `nighthawk61r2` (2 patches), 1024 shots × 3 backends | 0 | 1–3 × 2–4 | patch 1 | 8 / 0 / 0 | 1.000 | 0.941 |
| | 1 | 2–4 × 7–9 | patch 0 | 8 / 0 / 0 | 1.000 | 0.946 |
| | 2 | 6–8 × 3–5 | patch 1 7, seam 2 | 5 / 3 / 0 | 1.000 | 0.922 |
| | 3 | 6–8 × 6–8 | patch 0 8, seam 1 | 7 / 1 / 0 | 1.000 | 1.000 |
| `fav5p112` (5 patches), 512 shots × 1 backend | 0 | 0–2 × 4–6 | patch 1 8, seam 1 | 7 / 1 / 0 | 1.000 | 1.000 |
| | 1 | 1–3 × 0–2 | patch 0 | 8 / 0 / 0 | 1.000 | 0.930 |
| | 2 | 2–4 × 7–9 | patch 2 8, seam 1 | 7 / 1 / 0 | 1.000 | 1.000 |
| | 3 | 5–7 × 6–8 | patches 2, 3; seam 2 | 4 / 4 / 0 | 0.975 | 0.902 |
| | 4 | 6–8 × 3–5 | patches 2, 4; seam 5 | 1 / 7 / 0 | 0.733 | 0.441 |
| | 5 | 7–9 × 0–2 | patch 4 | 8 / 0 / 0 | 1.000 | 0.980 |
| | 6 | 9–11 × 7–9 | patch 3 | 8 / 0 / 0 | 1.000 | 0.954 |

Outcome distributions against the exact one (numpy), `nighthawk61r2`, 1024 shots × 3 backends pooled per unit and state:

| Unit | Placement | \|0_L⟩, \|1_L⟩ (X basis): TVD / floor, impossible | \|+_L⟩, \|−_L⟩ (Z basis): TVD / floor, impossible | most frequent impossible outcomes (Z basis) | decoder |
|---|---|---|---|---|---|
| 0, 1 | one patch | 0.050–0.062 / 0.056–0.058, 0.0 % | 0.056–0.067 / 0.012–0.013, 5.4–5.5 % | 000 111 000, 111 111 111 (\|+_L⟩); 111 111 000, 000 111 111 (\|−_L⟩): one whole block flipped | fails on all of them |
| 2 | 7 patch, 2 seam; 3 replica CX | 0.260–0.269, 18–23 % | 0.325–0.331, 32–33 % | 111 000 011, 000 111 011, 000 000 100 | fails on some |
| 3 | 8 patch, 1 seam; 1 replica CX | 0.208–0.260, 20–26 % | 0.385–0.449, 39 % | 111 110 111, 000 110 000, 111 001 111 | recovers them (X 1.000) |

The 3 × 3 test (`shor9 --shots 2048 --repeats 3`): `patch_blocks` \|0_L⟩, \|1_L⟩ at TVD 0.066–0.086 (floor 0.040), 3–8 % impossible, all decoded; \|+_L⟩, \|−_L⟩ at 0.74–0.75 with 111 111 111 (35 %) and 000 111 111 (39 %) on top; `seam_leaders` at 0.59–0.92 in every state.

- **Two kinds of error, told apart by the patterns.** Inside one patch ACE's sampler flips a whole block (000 ↔ 111) in about 5 % of the Z-basis shots and nothing else: the X-basis readout of the same unit sits at the shot-noise floor. A whole-block flip is exactly what the code cannot correct. At the seam the wrong outcomes are mostly single bits inside one block (111 110 111, 000 001 000), which the inner bit-flip code removes: unit 3 decodes to 1.000 with 39 % of its shots impossible.

Larger codes (`shor9 --size m`, standalone two-patch ACE register, decoded success; ties count 0.5; exact reference 1.000 for every size and state):

| Code | ACE geometry | Shots × backends | `seam_leaders` Z / X | `patch_blocks` Z / X |
|---|---|---|---|---|
| Shor-9 | 3 × 3, lrc 1 | 1024 × 6 | 0.499 / 0.669 | 1.000 / 0.486 |
| Shor-16 | 4 × 4, lrc 2, lrr 3 | 1024 × 1 | 0.496 / 0.614 | — (even m) |
| Shor-25 | 5 × 5, lrc 2, lrr 4 | 1024 × 2 | 0.477 / 0.545 | 1.000 / 0.515 |
| Shor-36 | 6 × 6, lrc 3, lrr 5 | 1024 × 1 | 0.493 / 0.528 | — (even m) |

Shor-16 units tiled over whole registers (`--register --size 4`, 512 shots × 1 backend):

| Register | Unit | Regions | Encoder CX exact / replica / cross | Logical Z | Logical X |
|---|---|---|---|---|---|
| `nighthawk61r2` | 0 | patch 1 14, seam 2 | 13 / 2 / 0 | 1.000 | 1.000 |
| | 1 | patch 0 15, seam 1 | 14 / 1 / 0 | 1.000 | 1.000 |
| `fav5p112` | 0 | patch 0 15, seam 1 | 14 / 1 / 0 | 1.000 | 1.000 |
| | 1 | patches 1, 2; seam 2 | 11 / 4 / 0 | 0.955 | 0.809 |
| | 2 | patches 2, 3; seam 2 | 11 / 4 / 0 | 0.960 | 0.831 |
| | 3 | patch 4 | 15 / 0 / 0 | 1.000 | 0.951 |

- **A bigger code does not buy back what the seam takes.** On the standalone register every inner block of `seam_leaders` crosses the seam, and its logical X falls with size (0.67, 0.61, 0.55, 0.53 from Shor-9 to Shor-36) while Z stays at 0.5: more qubits per block means more replica CXs per block, and the seam's loss is not the independent single-qubit error the larger distance corrects. With the outer layer crossing (`patch_blocks`, odd m) logical Z stays exact and X stays at no information at every size.
- **Placed on the chip, Shor-16 tracks its replica CXs** as Shor-9 does: 1–2 replica CXs leave both bases at 1.000; 4 cost 0.04 in Z and 0.17–0.19 in X; the unit wholly inside one patch shows the sampler's whole-block flips again (X 0.951).

PyQrack version, ACE's boundary code and OpenCL. `QrackAceBackend` is PyQrack's Python code on top of the Qrack library, and PyQrack 2.29 changed its default `is_boundary_repetition_code` from off to on, with a new implementation (the seam replica that received the real gate decides; ACE's seam error detection is skipped while it is on). Library and wrapper swapped with `PYQRACK_SHARED_LIB_PATH` (Shor-9, standalone register, CPU, 2048 shots, 6 backends per state; decoded logical Z / X):

| PyQrack | Qrack library | Boundary code | `seam_leaders` | `seam_leaders` X, backend to backend | `patch_blocks` |
|---|---|---|---|---|---|
| 2.28.1 | 10.13.2 | off (its default) | 0.495 / 0.678 | 0.622–0.711 | 1.000 / 0.507 |
| 2.28.1 | 10.13.2 | on | 0.502 / 0.648 | 0.556–0.698 | 1.000 / 0.500 |
| 2.29.4 | 10.13.2 | off | 0.496 / 0.663 | 0.634–0.708 | 1.000 / 0.515 |
| 2.29.4 | 10.13.2 | on (its default) | 0.491 / 0.833 | 0.500–1.000 | 1.000 / 0.517 |
| 2.29.4 | 10.13.1 | on (its default) | 0.503 / 0.833 | 0.500–1.000 | 1.000 / 0.506 |

The EPYC (Qrack 10.13.2 library under PyQrack 2.28.1, boundary code off; CPU, 16 processes per job, 4096 shots × 10 backends per state; a range where the job ran more than once):

| Code | Runs | `seam_leaders` Z | `seam_leaders` X | `patch_blocks` Z / X | Exact reference | Seconds per run |
|---|---|---|---|---|---|---|
| Shor-9 | 3 | 0.488–0.508 | 0.651–0.679 | 1.000 / 0.496–0.513 | 1.000 | 351–1122 |
| Shor-16 | 3 | 0.501–0.514 | 0.590–0.601 | — (even m) | 1.000 | 312–830 |
| Shor-25 | 1 | 0.503 | 0.542 | 1.000 / 0.496 | 1.000 | 695 |
| Shor-36 | 1 | 0.493 | 0.527 | — (even m) | 1.000 | 4956 |

OpenCL validity (PoCL 3.0, CPU device, in the container; 1024 shots; the 10.13.2 rows under PyQrack 2.29.4, so with its boundary code on):

| Run | Qrack library | Shor-9 `seam_leaders` | Shor-16 `seam_leaders` | Shor-36 `seam_leaders` | Exact reference | Seconds (9 / 16 / 36) |
|---|---|---|---|---|---|---|
| CPU | 10.13.2 | 0.510 / 0.875 (4 backends: X 0.500–1.000) | 0.505 / 0.529 | 0.487 / 0.512 | 1.000 | 7 / 5 / 139 |
| PoCL, `--gpu --ocl-sync` | 10.13.2 | 0.489 / 1.000 (1 backend) | 0.517 / 0.536 | 0.482 / 0.493 | 1.000 | 8 / 12 / 94 |
| PoCL, `--gpu` (asynchronous) | 10.13.2 | 0.513 / 1.000 | C++ exception | C++ exception | — | 7 / — / — |
| PoCL, `--gpu` | 10.13.1 | 0.501 / 0.681 | exact reference wrong (\|−_L⟩ 0.53, then 0.00): stopped | — | — | 15 / — / — |

- **The earlier tables hold on the current library.** With the boundary code off, Qrack 10.13.2 gives Shor-9 to Shor-36 `seam_leaders` X of 0.65–0.68, 0.59–0.60, 0.54 and 0.53 on the EPYC against 0.67, 0.61, 0.55 and 0.53 above; the library version does not move ACE's results.
- **PyQrack 2.29's boundary code is what lifts the seam.** It takes Shor-9 `seam_leaders` X from 0.66 to 0.83, each backend landing at 1.000 or 0.5, and the seam units of `nighthawk61r2` from 0.92 to 1.000; PyQrack 2.28.1's boundary code changes nothing. At Shor-36 it does not help (0.512 against 0.528). The script now passes all three ACE switches explicitly (`ace_backend`), off unless `--ace-boundary-rep`, so a run means the same on either PyQrack; records carry `boundary_rep` and both versions.
- **The in-patch whole-block flips are in both PyQrack versions,** with the boundary code on or off: they come from ACE's sampling, not from the seam handling.
- **PoCL runs `shor9` correctly with `--ocl-sync` on the 10.13.2 library,** at every size, within ACE's backend-to-backend spread of the CPU. Without the sync switch, or on 10.13.1, the asynchronous-dispatch defect of §12 returns: C++ exceptions, or an exact reference that is wrong. The run stops on a wrong exact reference and records nothing.

- **The test spans the chip.** 4 units on 36 of `nighthawk61r2`'s 61 qubits, 7 units on 63 of `fav5p112`'s 112 (the 3 × 3 squares that fit between the holes), all in one circuit; Qrack keeps the unentangled units apart, so memory stays small (61 qubits: 0.05 GB peak).
- **Seam load decides, unit by unit.** Units with one replica CX keep both logical bases; 3–4 replica CXs cost up to 0.1; the unit sitting on five seam qubits with seven replica CXs falls to Z 0.73 and X 0.44.
- **Units inside one patch lose 2–7 % of logical X, though every CX is exact.** ACE's sampler returns odd block parities (e.g. all three blocks 111) about 5 % of the time for a unit wholly in one patch, on every fresh backend; the qubits' marginals are exactly 0.5. Logical Z is untouched. This is ACE's sampling path, not the seam.

### 10.4 GHZ states across the seams (E21)

`nighthawk_ghz.py` (README §3.6, METHODOLOGY §6.8): the formal proof's Layer 1a on each register's ACE preset placement, no Layer 2. Container, CPU, PyQrack 2.29.5 / Qrack 10.13.2 (PyPI), boundary repetition code off; exact reference (`QrackSimulator`, 128 shots per setting): every deterministic shot ideal in every run below. F_GHZ = (P0 + P1 + C)/2 from Lemma 4's MQC (2k + 2 phases); F0 = fidelity to GHZ_0; the witness is the Clifford bound F ≥ P0 + P1 + P(even X parity) − 1.

3-qubit chains tiled over the register (`tiles --unit 3`: rows and columns, offsets 0–2), 256 shots × 4 backends per setting (61) or × 2 (28):

| Register | Units | Off the seam (distance ≥ 1) | Holding a seam qubit, 1 seam CX | Holding one, 2 seam CXs (both patches) |
|---|---|---|---|---|
| `nighthawk2p28` (3 seam qubits) | 30 | 1.002 (16 units) | 0.767 (9) | 0.353 (5) |
| `nighthawk61r2` (6) | 82 | 1.000 (58) | 0.773 (15) | 0.426 (9) |

(mean F_GHZ; exact reference 0.996–1.002 in every row; λ per block from Lemma 3: 1.000 / 0.80–0.83 / 0.55–0.62)

| `nighthawk61r2`, earlier run (F0, 3 angles, 512 × 2) | Seam distance 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| layers none | 0.638 (24) | 1.000 (23) | 1.000 (21) | 1.000 (11) | 1.000 (3) |
| `fold:1` | 0.622 (24) | 1.000 | 1.000 | 1.000 | 1.000 |

One GHZ over the whole register (`chip`, stabilizer-hybrid patches, witness readout), 512 shots × 2 backends per setting:

| Register | Tree | Root | Depth | CX exact / seam | P0 + P1, run 1 / run 2 | F ≥ (witness) | Seam CXs that break (Z) |
|---|---|---|---|---|---|---|---|
| `nighthawk2p28` | bfs | 10 | 5 | 21 / 6 | 0.293 / — | −0.19 | 3 of 6 at 0.45–0.47 |
| | seam | 10 | 6 | 23 / 4 | 0.513 / — | 0.02 | 2 of 4 at 0.47–0.48 |
| `nighthawk2p57` | bfs | 26 | 7 | 44 / 12 | 0.267 / 0.000 | −0.24 / −0.50 | 4 of 12 at 0.47–0.50 |
| | seam | 9 | 10 | 50 / 6 | 0.000 / 0.005 | −0.51 / −0.46 | 3 of 6 at 0.47–0.52 |
| `nighthawk2p16` (MQC, 256 × 4) | bfs | 11 | 5 | 10 / 5 | 0.583 | F_GHZ 0.293 | |
| | seam | 3 | 6 | 11 / 4 | 0.456 | F_GHZ 0.238 | |

- **The loss sits on the seam qubits, not around them.** Every unit that only couples to a seam qubit (distance 1) is at F_GHZ 1.000 on both registers, phase 0.00 ± 0.01; the per-site map's halo around each seam column comes from sites sharing units with seam qubits.
- **A breaking seam CX cuts off its whole subtree.** Only some seam CXs break, each at 0.45–0.52, into a seam qubit or out of one; every qubit behind such a CX then flips with it (the map's blocks of equal break rate). The other seam CXs break at 0.000–0.010, and no CX inside one simulator breaks (0.0000 on every run).
- **Chip-wide, the state does not survive the seam.** ⟨X…X⟩ and C are ≈ 0 in every chip-wide run: the coherence does not cross. Populations vary from run to run by more than the trees differ (`nighthawk2p57` bfs 0.267, then 0.000), so fewer seam crossings are not shown to help.
- **Folding barely moves it.** Three times the seam CXs (`fold:1`) lower the seam units from 0.638 to 0.622 and leave every other unit at 1.000: the seam's loss behaves as a decorrelation at a crossing, not as an error per gate.
- **`nighthawk61r2` chip-wide is not measured** on this build: its stabilizer patches leave the tableau in ACE's sampler (README §3.6, §8).

### 10.5 Layer 2 across the seams: Walsh echo, compensation (E22)

The formal proof's Layer 2 on the same tiles: after the preparation, L = 3 idle windows of T = 4.4 under H_D (Definition 2, η = 0; J_ij = 0.065 × U(0.75, 1.25), δω_q = 0.022 ((q mod 3) − 1), σ_qs = 0.015, seed 42: the paper's §8 model), in each mode; 256 shots × K = 4 realisations (fresh backend and ε) per setting; exact reference with the same ε draws. Mean F_GHZ (F0 in brackets where the phase matters):

| `nighthawk61r2` units | none | unmitigated | compensated (oracle Ĥ) | dd (Walsh echo) | exact: unmit. / comp. / dd |
|---|---|---|---|---|---|
| off the seam (58) | 1.000 | 0.901 (F0 0.581) | 0.900 (F0 0.877) | **1.001** | 0.901 / 0.900 / 1.002 |
| holding a seam qubit, 1 seam CX (15) | 0.773 | 0.650 | 0.627 | 0.671 | 0.880 / 0.884 / 0.999 |
| holding one, 2 seam CXs (9) | 0.426 | 0.304 | 0.274 | 0.240 | 0.886 / 0.887 / 1.000 |

| `nighthawk2p28` units | none | unmitigated | compensated | dd | exact: unmit. / comp. / dd |
|---|---|---|---|---|---|
| off the seam (16) | 1.002 | 0.913 (F0 0.634) | 0.911 (F0 0.888) | **1.000** | 0.910 / 0.913 / 1.001 |
| 1 seam CX (9) | 0.767 | 0.688 | 0.654 | 0.724 | 0.889 / 0.890 / 0.999 |
| 2 seam CXs (5) | 0.353 | 0.257 | 0.194 | 0.183 | 0.897 / 0.898 / 0.997 |

One GHZ over `nighthawk2p16` (MQC, dense patches): P0 + P1 bfs / seam tree 0.583 / 0.456 without Layer 2; 0.262 / 0.308 unmitigated; 0.279 / 0.295 compensated; **0.000** / 0.298 dd; the exact reference 1.000 under dd, F_GHZ 0.67 under the other two.

- **Off the seam, ACE reproduces the paper.** In every mode ACE equals the exact reference within shot noise, and the Walsh echo returns F_GHZ = 1.000 with the phase at 0.00 ± 0.01 (Theorem 1, Corollary 7.1). Unmitigated drift leaves F_GHZ at 0.90 (the ε dephasing over K = 4 draws) and F0 at 0.58 (the deterministic phase of δω and of the couplings to idle neighbours); compensation removes that phase (F0 0.88) but not ε, as Proposition 1 says.
- **On the seam, every Layer 2 mode costs more than it refocuses.** The drift's couplings through the seam are themselves seam CXs on ACE, so each window adds crossings (§10.6: their count alone does not lower F; their rotations do): units holding a seam qubit fall below their no-drift values in every mode, and most where a window passes the drift most often (dd: four passes; compensated: two; unmitigated: one) on the two-seam-CX units. The exact reference keeps them at 0.88–1.00.
- **Chip-wide, the echo makes it worse on ACE.** With every seam coupler inside the state, the dd windows break the 16-qubit bfs GHZ completely (P0 + P1 = 0.000); the exact reference under the same echo is ideal.
- **Corollary 4.1 needs the paper's K.** With K = 4 and σ_qs = 0.015, the ε phase spread over L T = 13.2 is 2LTσ√k ≈ 0.7 rad per 3-qubit unit (1.6 rad at 16 qubits), larger than the δω signal; `sum_dw_estimate` needs K ≈ 256 as in the paper's calibration.

### 10.6 Seam error extraction and ZNE (E23)

`nighthawk_ghz.py stack` in its first version, now `stack --zne-k 1,2` (README §3.6, METHODOLOGY §6.8): pass 1 (Layer 1a), pass 2 (Layer 2 dd, the §8 model, L = 3), the seam's error read from passes 1 and 2, then ZNE with every seam crossing scaled by λ = 1, 5, 9, and a rerun of pass 2 with fresh ε and backends (the column "Rerun" below; since §10.7 Layer V is the corrected measurement). 256 shots × K = 4 per setting and λ, exact reference 128 shots; container CPU, PyQrack 2.29.5 / Qrack 10.13.2. Mean F_GHZ (Lemma 4):

| Register, units | Pass 1 (L1a) | Pass 2 (L2 dd) | Rerun (L2) | ZNE linear | ZNE exp | Exact, pass 2 |
|---|---|---|---|---|---|---|
| `nighthawk2p28`, off the seam (16) | 1.001 | 1.000 | — | — | — | 1.002 |
| `nighthawk2p28`, 1 seam CX (9) | 0.766 | 0.737 | 0.723 | 0.714 | 0.779 (8 of 9) | 1.000 |
| `nighthawk2p28`, 2 seam CXs (5) | 0.360 | 0.185 | 0.156 | 0.164 | undefined | 0.996 |
| `nighthawk61r2`, off the seam (58) | 1.001 | 0.999 | — | — | — | 1.000 |
| `nighthawk61r2`, 1 seam CX (15) | 0.761 | 0.672 | 0.699 | 0.671 | 0.765 (13 of 15) | 1.000 |
| `nighthawk61r2`, 2 seam CXs (9) | 0.424 | 0.262 | 0.222 | 0.232 | 0.417 (3 of 9) | 0.996 |
| `nighthawk2p16` chip, seam tree (4 seam CXs) | 0.159 | 0.149 | 0.226 | 0.173 | 0.181 | 1.002 |
| `nighthawk2p16` chip, bfs tree (5 seam CXs) | 0.230 | 0.004 | 0.002 | 0.003 | undefined | 0.999 |

(ZNE exp is defined only where the population excess and \|c\| stay positive at every λ; its mean covers those units, the higher ones, so it reads high.)

| Layer S extraction: kept against exact, log(kept) = a + b·n | P0 + P1 first crossing | per further crossing | Coherence first crossing | per further crossing |
|---|---|---|---|---|
| `nighthawk2p28` (28 unit-passes, n = 1…50) | 0.867 | 0.986 | 0.391 | 0.965 |
| `nighthawk61r2` (48, n = 1…50) | 0.773 | 0.984 | 0.390 | 0.975 |
| `nighthawk2p16` chip (4, n = 4…149) | 0.409 | 0.979 | 0.014 | 0.997 |

| Seam units: F at λ = 1 / 5 / 9 | Mean | Slope per λ, mean (range) |
|---|---|---|
| `nighthawk2p28`, 1 seam CX | 0.737 / 0.715 / 0.750 | +0.002 (−0.004 … +0.010) |
| `nighthawk2p28`, 2 seam CXs | 0.185 / 0.135 / 0.161 | −0.003 (−0.018 … +0.008) |
| `nighthawk61r2`, 1 seam CX | 0.672 / 0.726 / 0.711 | +0.005 (−0.007 … +0.048) |
| `nighthawk61r2`, 2 seam CXs | 0.262 / 0.143 / 0.224 | −0.005 (−0.044 … +0.021) |

| Amplifier check, `nighthawk2p28` tiles, no drift (256 × 2), mean F_GHZ | none | `fold:1` / `fold:3` (×3, ×7) | `zzfold:1` / `2` / `4` (×5, ×9, ×17) | `gfold:1` / `2` (×3, ×5) |
|---|---|---|---|---|
| 1 seam CX (9 units) | 0.780 | 0.763 / 0.760 | 0.753 / 0.771 / 0.759 | 0.740 / 0.776 |
| 2 seam CXs (5 units) | 0.371 | 0.357 / 0.355 | 0.317 / 0.326 / 0.292 | 0.363 / 0.346 |

| ZNE (linear) − pass 2 against the rerun spread \|rerun − pass 2\| | ZNE − pass 2, mean (range) | \|rerun − pass 2\|, mean (max) |
|---|---|---|
| `nighthawk2p28` (14 seam units) | −0.022 (−0.234 … +0.014) | 0.056 (0.196) |
| `nighthawk61r2` (24) | −0.012 (−0.077 … +0.036) | 0.059 (0.332) |
| `nighthawk2p16` chip (2) | +0.012 (−0.001 … +0.024) | 0.040 (0.077) |

- **The seam's loss is set at the first crossing.** Layer S finds the same coherence share after the first crossing on both tiled registers, 0.391 and 0.390; each further crossing costs 1.4–3.5 % against that. A unit's F_GHZ sits at one of a few levels (≈ 0.9, ≈ 0.6, ≈ 0.35 with no drift) whatever the amplification.
- **No amplifier scales it.** Seventeen times the seam crossings (`zzfold:4`) move the seam units by 0.02–0.08; CX folding and global folding move them less. In Layer S an amplified point moves from λ = 1 about as far as a rerun does (median \|F(λ) − F(1)\| 0.024 / 0.041 against median \|rerun − pass 2\| 0.038 / 0.026 on 28 / 61 qubits) and in no consistent direction: the mean slope per λ is +0.002 to +0.005 on 1-seam-CX units, −0.003 to −0.005 on 2-seam-CX units.
- **ZNE therefore returns pass 2.** Linear ZNE differs from pass 2 by −0.022 to +0.012 on average, less than a fresh run differs from pass 2 (0.040–0.059): it corrects nothing that a rerun does not change as much. The exponential model is undefined on most 2-seam-CX units (their populations fall under 2/d) and its class means are selected high.
- **Pass 2's extra loss is not the crossing count.** The per-crossing rate between passes 1 and 2 (0.965–0.975 for coherence) would predict a further fall at λ = 5 and 9; the amplified runs show none. What Layer 2 adds on the seam comes from what its windows do there, the ZZ rotations and the X pulses on seam replicas, not from how many crossings they make (this revises §10.5's reading). Identities of the same crossings cost nothing.
- **Chip-wide, nothing to extrapolate from.** On `nighthawk2p16` the coherence kept after the first crossing is 0.014: the bfs state under the echo is gone at every λ (0.004 / 0.004 / 0.007); the seam tree keeps 0.15–0.23, again within its rerun spread.

### 10.7 Seam error correction: syndrome, replica repetition code, the third measurement (E24)

`nighthawk_ghz.py stack` (README §3.6, METHODOLOGY §6.8): pass 1 (Layer 1a), pass 2 (Layer 2 dd, the §8 model, L = 3), Layer S (the seam's syndrome from passes 1 and 2, every unit classed and planned; the replica repetition code with its schedule on the replica units that lost F), Layer V (pass 2's circuit again, realisations 1000–1003: the code units under the correction, the cut units without it as the control). 256 shots × K = 4 per setting, exact reference 128 shots; container CPU, PyQrack 2.29.5 / Qrack 10.13.2. Mean F_GHZ (Lemma 4; chip: MQC over 16 qubits):

| Register, units | Seam class, plan | Pass 1 (L1a) | Pass 2 (L2 dd) | V (L2 + correction) | Exact, pass 2 |
|---|---|---|---|---|---|
| `nighthawk2p28`, off the seam (16) | bulk | 1.000 | 1.000 | — | 1.000 |
| `nighthawk2p28`, 1 seam CX (9) | replica, code | 0.777 | 0.720 | **1.001** | 1.000 |
| `nighthawk2p28`, 2 seam CXs (5) | cut, control | 0.358 | 0.222 | 0.162 | 0.998 |
| `nighthawk61r2`, off the seam (58) | bulk | 0.999 | 1.001 | — | 1.000 |
| `nighthawk61r2`, 1 seam CX (15) | replica, code | 0.769 | 0.683 | **0.999** | 1.002 |
| `nighthawk61r2`, 2 seam CXs, two seam qubits (2) | replica, code | 0.625 | 0.481 | **1.002** | 1.002 |
| `nighthawk61r2`, 2 seam CXs (7) | cut, control | 0.350 | 0.228 | 0.154 | 1.003 |
| `nighthawk2p16` chip, seam tree (4 seam CXs) | cut, control | 0.225 | 0.149 | 0.206 | 0.999 |
| `nighthawk2p16` chip, bfs tree (5 seam CXs) | cut, control | 0.068 | 0.135 | 0.081 | 0.997 |

| What V changes | Code units: V − pass 2, mean (range) | Control units: \|V − pass 2\|, mean (max) | Control: V at most; coherence |
|---|---|---|---|
| `nighthawk2p28` | +0.280 (+0.102 … +0.572); exact − V −0.001 | 0.093 (0.243) | 0.259; 0.024 |
| `nighthawk61r2` | +0.340 (+0.084 … +0.913); exact − V +0.002 | 0.088 (0.194) | 0.263; 0.024 |
| `nighthawk2p16` chip | — | 0.055 (0.057) | 0.206; 0.007 |

| Layer S syndrome, pass 1 / pass 2 | Units | P0 + P1 kept | Coherence kept | P1 − P0 | Most-breaking qubit is a seam qubit |
|---|---|---|---|---|---|
| `nighthawk2p28`, replica | 9 | 0.868 / 0.794 | 0.689 / 0.647 | −0.156 / +0.110 | 4 of 4 / 3 of 4 |
| `nighthawk2p28`, cut | 5 | 0.661 / 0.401 | 0.055 / 0.043 | −0.021 / −0.024 | 0 of 5 / 0 of 5 |
| `nighthawk61r2`, replica | 17 | 0.853 / 0.756 | 0.655 / 0.562 | −0.151 / −0.064 | 8 of 9 / 7 of 9 |
| `nighthawk61r2`, cut | 7 | 0.664 / 0.417 | 0.035 / 0.039 | −0.043 / +0.038 | 0 of 7 / 0 of 7 |
| `nighthawk2p16` chip, cut | 2 | 0.283 / 0.278 | 0.010 / 0.007 | ± 0.28 | 0 of 2 / 1 of 2 |

(Most-breaking qubit: counted over the units whose highest rate of leaving the unit's Z majority exceeds 0.02. P1 − P0 is a mean over units of both signs; per unit it reaches ± 0.3.)

| Tomography, `nighthawk2p28`, no drift (27 Pauli settings × 2048 shots; exact 1024) | F(GHZ₀) | P0 / P1 | c = ρ₀…₀,₁…₁ | Error weight in the GHZ basis |
|---|---|---|---|---|
| [19, 18, 20], seam qubit 18 a leaf (replica), ACE, two backends | 0.579 / 0.699 | 0.16–0.19 / 0.51–0.55 | 0.22 / 0.35 | 0.29–0.30 on 010 ± (the seam qubit flipped), from the 0…0 branch |
| the same, boundary repetition code | 1.000 | 0.498 / 0.502 | 0.500 | none above 0.01 |
| [11, 10, 12], seam qubit 11 the root between the patches (cut), ACE | 0.296 / 0.297 | 0.30 / 0.29 | 0.000 / 0.002 | 000 + and − equal (0.30 each): no coherence; 0.28 on 010 (q10), 0.16 on 110 (q12) |
| the same, boundary repetition code | 0.329 | 0.33 / 0.33 | −0.003 | as without |
| exact reference, both | 1.000 | 0.49–0.51 | 0.500 | — |

| The code's schedule, `nighthawk61r2` [48, 40, 56] (48→40 seam to seam, 48→56 into patch 1), dd, realisations 1000–1003, two repeats | F_GHZ |
|---|---|
| boundary repetition code, the preparation's order (48→40 first) | 0.446 / 0.230 (per realisation 0.05–0.61) |
| boundary repetition code, the code's schedule (48→56 first) | 1.000 / 1.005 |
| no code, realisations 0–2: no drift / dd | 0.654 / 0.429 |

- **The replica error is a bit flip, and a code removes it.** Where one simulator holds the unit, the seam qubit's Z value leaves the unit's (most-breaking qubit a seam qubit in 22 of 26 cases over both registers), mostly from one branch (P0 0.16–0.19 against P1 0.51–0.55 in the tomography), with part of the coherence kept (0.56–0.69). ACE's replica repetition code with authority decoding takes all 26 replica units, Layer 2 included, to the exact reference: 1.001, 0.999 and 1.002 against 1.000–1.002. That is +0.28 to +0.34 on average over pass 2, three to four times what a rerun moves a unit (0.09), and 0.22–0.25 over pass 1.
- **The code needs its schedule.** A CX between two seam qubits is matched in every simulator and clears the decoder's authority record; met before the seam qubit's partner in the host simulator, it leaves no replica to trust ([48, 40, 56]: 0.23–0.45 under the code). Reordering the tree so a qubit's CXs into one simulator come first (the same state: CXs from one control commute) gives 1.000 there, and every replica unit of both registers is corrected.
- **The cut class is beyond any code on ACE.** A unit no simulator holds keeps 0.02–0.06 of its coherence in every pass (tomography: c = 0.000, GHZ⁺ and GHZ⁻ equal), because ACE joins its simulators only through replicas and the classical shadow: no state of two simulators is entangled between them. F_GHZ ≤ (P0 + P1)/2 ≤ ½ for any decoder; measured at most 0.26. The boundary code leaves it there (0.30 → 0.33 in the tomography), and so does the chip-wide GHZ (coherence 0.007–0.010), which on a register of two patches is always cut.
- **Against §10.6.** ZNE moved F by −0.02 … +0.01 because the seam's loss is set at the first crossing; the code removes it outright on the replica class. What remains of the seam's GHZ loss after Layer S is exactly the cut class, the coherence ACE does not carry between simulators.
- **Layer 2 on a corrected seam.** Under the code the Walsh echo is back to Theorem 1 on the seam units: V (pass 2's dd circuit) = exact on every replica unit, where pass 2 without the code had fallen to 0.48–0.72.

### 10.8 PyQrack's ACE source: the authority rule, the crossbar, the gadget (E25)

PyQrack's main branch (unitaryfoundation/pyqrack, last change to `qrack_ace_backend.py` on 21 October 2026) is byte-identical to the 2.29.5 used here. Its newest change replaced the fresh-copy authority of the boundary code with lineage tracking (`_update_lineage`): a replica stays the decoder's authority while every coupling it took part in was a real gate with a clean replica of the partner, and when two boundary qubits meet, their crossbar pair becomes the sole authority. `nighthawk_ghz.py --code-authority colocated` adds one step after it: a boundary qubit whose authority moves to a simulator takes the authority of every boundary qubit it has coupled with to that qubit's replica in the same simulator. `stack --code-schedule off` keeps the preparation's CX order. 256 shots × K = 4 per setting, Layer V on realisations 1000–1003; container CPU, PyQrack 2.29.5 / Qrack 10.13.2. Layer V, mean F_GHZ of the replica units under the code (E24: lineage with the schedule):

| Register, replica units | Lineage + schedule (E24) | Lineage, preparation order | Co-located, preparation order | Co-located + schedule | Exact |
|---|---|---|---|---|---|
| `nighthawk2p28`, 1 seam CX (9) | 1.001 | 1.000 | 1.000 | 1.002 | 1.000–1.003 |
| `nighthawk61r2`, 1 seam CX (15) | 0.999 | 1.001 | 1.001 | 1.002 | 0.998–1.002 |
| `nighthawk61r2`, 2 seam CXs, two seam qubits (2) | 1.002 | **0.732** | 1.001 | 0.999 | 0.990–1.002 |
| of these, [48, 40, 56] (48→40 seam to seam first) | 1.000 | **0.463** | 1.003 | 0.994 | 0.990–1.001 |
| `nighthawk61r2`, code units: V − pass 2, mean | +0.340 | +0.320 | +0.338 | +0.346 | — |

The cut units run without the code in every configuration: V 0.15–0.22, coherence 0.03–0.05, the control as in §10.7.

| In-process probe, `nighthawk61r2`, preparation order, F_GHZ (no drift / echo) | No code | Code, lineage | Code, co-located |
|---|---|---|---|
| [48, 40, 56] | 0.648 / 0.157 | 0.780 / 0.584 | 0.998 / 1.003 |
| [18, 10, 26] | 0.658 / 0.570 | 0.999 / 1.002 | 1.001 / 0.997 |
| seam pair [18, 26] | 0.935 / 0.822 | 1.007 / 0.999 | 1.002 / 0.996 |
| seam pair [40, 48] | 0.927 / 0.770 | 1.001 / 0.993 | 0.997 / 1.004 |
| all 17 replica units, echo, 2 realisations: mean (lowest) | — | 0.970 (0.469, [48, 40, 56]) | 0.999 (0.989) |
| all 7 cut units, echo: mean | — | 0.160 | 0.147 |

| Z populations of a seam unit, no code, 6 backends × 256 shots | Error detection | P000 | P111 | Seam qubit flipped (010 / 101) |
|---|---|---|---|---|
| `nighthawk61r2` [19, 18, 20] | on | 0.514 | 0.157 | 0.06 / 0.27 |
| | off | 0.232 | 0.377 | 0.27 / 0.12 |
| `nighthawk61r2` [58, 57, 59] | on | 0.367 | 0.244 | 0.11 / 0.28 |
| | off | 0.247 | 0.434 | 0.21 / 0.11 |
| `nighthawk61r2` [11, 10, 12], cut | on / off | 0.354 / 0.363 | 0.327 / 0.310 | patches decorrelate (§10.7) |

- **The lineage rule strands a partner on the crossbar.** A CX between two seam qubits runs in every simulator, and the crossbar pair becomes the sole authority of both. When one of them then couples into a patch, its authority moves there; its partner's stays on the crossbar, whose copy is correlated with a stale copy of the first. [48, 40, 56] in the preparation's order: 0.463 through the stack, 0.58 in the probe, against 1.00 for every other replica unit.
- **The co-located step removes it, and with it the need for the schedule.** Moving the partner's authority to its copy in the same simulator, which shares the whole history, corrects [48, 40, 56] in the preparation's order (1.003) and every replica class of both registers (means 1.000–1.001); with the schedule as well it changes nothing (means 0.999–1.002). It costs nothing on the other classes. The schedule of §10.7 works around the same stranding from the circuit side; the co-located step fixes it in the decoder, for any order.
- **The crossbar carries a seam-to-seam GHZ.** Pairs of seam qubits reach 0.99–1.01 under the code with the echo: the crossbar simulator holds a replica of every boundary qubit, so seam-to-seam circuits stay coherent across the whole boundary, as the source says. Bulk qubits have no crossbar replica, so the cut class (§10.7) stays.
- **The error-detection gadget does not make the one-sided loss.** With it on or off, the seam qubit leaves its unit in about a third of shots, and which branch loses changes from backend to backend (P000 0.51 against P111 0.16 with it on, 0.23 against 0.38 off): a per-backend reconcile choice, not the gadget's forced ancilla measurement.
- **Two readings of the source, not run.** `measure_shots_consensus` forces every qubit to its majority marginal across clones, which erases a GHZ's correlations (every marginal is ½): an XEB tool. `QrackAceBackend` turns `is_boundary_repetition_code` on by default since 2.29; `nighthawk_qrack.py` passes it off for comparability, so the uncorrected passes here are that choice, not ACE's default.

### 10.9 The code before the echo (E26)

`nighthawk_ghz.py stack --order code-echo` (README §3.6, METHODOLOGY §6.8) flips Layers 2 and S of §10.7: pass 1 (Layer 1a), Layer S from pass 1's syndrome alone, pass C (pass 1's circuit, realisations 1000–1003: the replica units under the code with its schedule, the cut units without as the control; bulk units keep pass 1), pass E (Layer 2 dd on every unit, the §8 model, L = 3, realisations 0–3 as in §10.7's pass 2, the code kept on the replica units). The drift arrives with its echo in both orders, so pass E runs §10.7's Layer V circuit on the replica units. `nighthawk61r2`, 256 shots × K = 4 per setting, exact reference 128 shots; container CPU, PyQrack 2.29.5 / Qrack 10.13.2; authority `lineage`. Mean F_GHZ (Lemma 4):

| Units | Seam class, plan | Pass 1 (L1a) | C (L1a + code) | E (code + L2 dd) | Exact, E |
|---|---|---|---|---|---|
| off the seam (58) | bulk | 1.000 | 1.000 (pass 1) | 1.000 | 1.000 |
| 1 seam CX (15) | replica, code | 0.770 | **1.000** | **1.000** | 1.001 |
| 2 seam CXs, two seam qubits (2) | replica, code | 0.666 | **0.997** | **0.999** | 0.998 |
| 2 seam CXs (7) | cut, control | 0.365 | 0.357 | 0.216 | 1.001 |

| The two orders, `nighthawk61r2`, class means | Second measurement | Third measurement |
|---|---|---|
| echo, then code (§10.7): replica (17) | 0.660 (echo, no code) | 1.000 (V) |
| code, then echo: replica (17) | **1.000** (C, code, no drift) | 1.000 (E) |
| echo, then code: cut (7) | 0.228 | 0.154 (V, realisations 1000–1003) |
| code, then echo: cut (7) | 0.357 (C, no drift) | 0.216 (E, realisations 0–3; §10.7 pass 2 on the same draws: 0.228) |

| Per unit | Pass 1 | C | E |
|---|---|---|---|
| [48, 40, 56] (48→40 seam to seam; the code's schedule) | 0.669 | 0.993 | 1.003 |
| [18, 10, 26] | 0.663 | 1.000 | 0.994 |
| replica units, range | 0.59–0.94 | 0.993–1.008 | 0.994–1.005 |
| cut units: coherence (C_parity), mean | — | 0.046 | 0.040 |

- **Pass 1 alone plans the same correction.** Every replica unit lost more than 0.02 and 3σ in pass 1 (17 of 17 planned code, as with passes 1 and 2 in §10.7); the syndrome's classes are those of §10.7 (most-breaking qubit a seam qubit in 8 of 9). The seam-crossing fit is not identifiable from pass 1: its units have 1 or 2 crossings, one count per class, so `seam_model` now needs three counts and reports none.
- **The code works without the echo, and the echo works on the code.** C takes the replica class from 0.770 to 1.000 before any drift is applied; E adds the drift and the Walsh echo and stays at 1.000 (E − C +0.000, exact − E +0.001). Theorem 1 holds on a corrected seam whichever layer comes first; with the code first, the echo never sees the replica error, so the dip of §10.7's pass 2 (0.660) does not occur.
- **The cut class follows the drift, not the order.** Without drift a rerun moves it by 0.018 (C 0.357 against pass 1 0.365); with drift and echo it falls to 0.216, the same as §10.7's pass 2 on the same ε draws (0.228). Coherence stays at 0.04–0.05 throughout: no order of these layers gives ACE coherence between simulators.
- **The final measurement does not depend on the order.** Replica 1.000 in V and in E, bulk 1.000 in both; the order decides only what the middle measurement shows and which passes the syndrome is read from.

### 10.10 The Walsh echo window by window (E27)

`nighthawk_ghz.py sweep` (README §3.6; run as `tiles` once per L, which is what `sweep` does): `nighthawk2p28`, 3-qubit chains, Layer 2 dd with L = 0 (none) to 8 windows of four Walsh segments each, the §8 model (T = 4.4); L = 8 once more under the boundary repetition code (lineage authority, the preparation's order). 256 shots × K = 4 per setting, ACE only (the exact reference under dd is ideal by Theorem 1, checked in §10.5); container CPU, PyQrack 2.29.5 / Qrack 10.13.2. Mean over units:

| L (windows) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 8, code |
|---|---|---|---|---|---|---|---|---|---|
| off the seam (16), F_GHZ | 1.001 | 0.999 | 1.000 | 1.000 | 1.001 | 1.001 | 0.999 | 1.000 | 0.998 |
| replica (9), F_GHZ | 0.779 | 0.753 | 0.720 | 0.722 | 0.724 | 0.714 | 0.722 | 0.705 | **0.999** |
| replica, C = 2\|c\| | 0.697 | 0.642 | 0.621 | 0.622 | 0.613 | 0.621 | 0.608 | 0.596 | 0.998 |
| replica, P0 + P1 | 0.862 | 0.865 | 0.819 | 0.823 | 0.835 | 0.806 | 0.836 | 0.814 | 1.000 |
| cut (5), F_GHZ | 0.355 | 0.150 | 0.217 | 0.189 | 0.180 | 0.168 | 0.142 | 0.163 | 0.173 |
| cut, P0 + P1 | 0.666 | 0.270 | 0.406 | 0.339 | 0.323 | 0.306 | 0.261 | 0.285 | 0.282 |
| cut, C | 0.043 | 0.030 | 0.029 | 0.039 | 0.037 | 0.030 | 0.023 | 0.041 | 0.065 |

- **Off the seam the echo is exact in every window.** F_GHZ 1.000 from L = 0 to 8: each window refocuses its own drift (Theorem 1), so nothing carries into the next.
- **The replica loss is a step, not a decay.** 0.779 → 0.720 over the first two windows, then 0.705–0.724 to L = 8, although every window adds 8 seam crossings per unit (64 at L = 8). A fixed loss per crossing, as §10.6's fit across units suggests, would take L = 8 far below L = 2. Within one unit the seam qubit's replicas disagree once; further crossings add no more.
- **The cut units fall to near the scrambled floor in the first window.** P0 + P1 0.666 → 0.270 at L = 1, then 0.26–0.41, against 2/8 = 0.25 for Z bits with no correlation left; the coherence is 0.02–0.04 at every L. The odd / even swing (0.270 at L = 1, 0.406 at L = 2) is within the backend-to-backend spread of 5 units × 4 backends.
- **With the code the plateau is gone.** At L = 8 the replica units are at 0.999 and the cut units where they are without it (0.173).

### 10.11 Protecting the cut units from Layer 2 (E28)

A ZZ coupling between two qubits of a GHZ unit is a global phase on the GHZ (Z_iZ_j stabilises it, and every Walsh-flipped image of it: Corollary 2.1), yet on ACE it is two CXs through the seam wherever the coupling crosses one, and before every such CX ACE reconciles the seam qubit's replicas. `nighthawk_ghz.py probe` (README §3.6; run as a standalone script with the same circuits before it was folded in) puts the cut units of `nighthawk61r2` (all 7, every spread tiling) through Layer 2 dd, L = 3, the §8 model, with the inside ZZ applied several ways; 256 shots × K = 4 per setting, exact reference 128 shots; container CPU, PyQrack 2.29.5 / Qrack 10.13.2:

| Cut units, `nighthawk61r2` (7) | F_GHZ | P0 + P1 | C | Per unit | Exact |
|---|---|---|---|---|---|
| no Layer 2 | 0.355 | 0.666 | 0.044 | 0.32–0.37 | — |
| inside ZZ as CX · Rz · CX (`gate`, as in §10.5–10.9) | 0.205 | 0.379 | 0.030 | 0.05–0.30 | 1.001 |
| the same, CX target on the seam qubit | 0.133 | 0.222 | 0.043 | 0.07–0.22 | — |
| the same, CX target on the bulk qubit | 0.258 | 0.480 | 0.036 | 0.17–0.34 | — |
| **inside ZZ dropped (`stab`)** | **0.353** | **0.669** | 0.038 | **0.34–0.37** | **1.001** |
| chain preparation from one end, no Layer 2 | 0.332 | 0.646 | 0.017 | 0.32–0.34 | — |
| chain preparation, inside ZZ as gates | 0.276 | 0.532 | 0.019 | 0.20–0.34 | — |
| chain preparation, inside ZZ dropped | 0.333 | 0.650 | 0.016 | 0.32–0.35 | 1.002 |

(A first run of the same variants with the units grouped by tiling gave gate 0.187, target on the seam qubit 0.168, on the bulk qubit 0.242, dropped 0.358, chain with gates 0.214: the gate variants scatter by ± 0.03–0.06 between runs, the dropped variants by ± 0.005.)

`nighthawk_ghz.py --inside-zz stabilizer` makes the dropped variant the stack's rule. `stack` (§10.7's settings, `echo-code`, lineage authority with the code's schedule), mean F_GHZ:

| Register, units | Seam class, plan | Pass 1 | Pass 2 (dd), gates (§10.7) | Pass 2 (dd), dropped | V, gates (§10.7) | V, dropped | Exact |
|---|---|---|---|---|---|---|---|
| `nighthawk2p28`, off the seam (16) | bulk | 1.000 | 1.000 | 1.001 | — | — | 0.997 |
| `nighthawk2p28`, 1 seam CX (9) | replica, code | 0.770 | 0.720 | **0.781** | 1.001 | 0.999 | 1.001 |
| `nighthawk2p28`, 2 seam CXs (5) | cut, control | 0.361 | 0.222 | **0.355** | 0.162 | **0.345** | 1.001 |
| `nighthawk61r2`, off the seam (58) | bulk | 1.000 | 1.001 | 1.000 | — | — | 1.001 |
| `nighthawk61r2`, 1 seam CX (15) | replica, code | 0.771 | 0.683 | **0.764** | 0.999 | 1.001 | 1.001 |
| `nighthawk61r2`, 2 seam CXs, two seam qubits (2) | replica, code | 0.654 | 0.481 | **0.668** | 1.002 | 0.999 | 0.999 |
| `nighthawk61r2`, 2 seam CXs (7) | cut, control | 0.347 | 0.228 | **0.355** | 0.154 | **0.362** | 1.000 |

With the code first (`--order code-echo`, §10.9), `nighthawk61r2`, the same settings:

| Units | Seam class, plan | Pass 1 | C (code) | E (code + dd), gates (§10.9) | E (code + dd), dropped | Exact, E |
|---|---|---|---|---|---|---|
| off the seam (58) | bulk | 1.000 | 1.000 (pass 1) | 1.000 | 1.001 | 1.000 |
| 1 seam CX (15) | replica, code | 0.773 | 0.999 | 1.000 | 1.000 | 0.998 |
| 2 seam CXs, two seam qubits (2) | replica, code | 0.626 | 0.994 | 0.999 | 1.004 | 0.999 |
| 2 seam CXs (7) | cut, control | 0.356 | 0.353 | 0.216 | **0.352** | 1.003 |

(Pass 1 and C are from the run with the inside ZZ dropped; §10.9's are within 0.01, except pass 1 of the two units with two seam qubits, 0.666 there: two units, four backends each. With the code first and the inside ZZ dropped, no class loses fidelity to Layer 2: E − C +0.002 on the replica units, −0.001 on the cut units.)

(Pass 1 is from this run; §10.7's pass 1 is within 0.03 of it in every row. Cut units in V: \|V − pass 2\| 0.011 / 0.018, coherence 0.028 / 0.046 on `nighthawk2p28` / `nighthawk61r2`.)

- **The drop under Layer 2 comes from the inside ZZ, and dropping it removes the drop.** With the inside ZZ dropped, the cut units keep their no-drift level through the echo (0.353 against 0.355; per unit 0.34–0.37), on ACE and in the stack on both registers (0.345–0.362 against 0.154–0.228 with the gates). The exact reference is the same with and without it (1.001, 1.002): the same circuit.
- **The ½ ceiling stays.** C = 0.016–0.044 in every variant, so F_GHZ ≤ (P0 + P1)/2 ≈ 0.33–0.36. What the stabiliser rule protects is the level the preparation leaves, not coherence between simulators: that still needs a unit held by one simulator (§10.7) or the crossbar (§10.8).
- **The replica units lose their echo dip too.** Pass 2 under dd is at pass 1's level (0.764 against 0.771, 0.668 against 0.654 on `nighthawk61r2`; 0.781 against 0.770 on `nighthawk2p28`), where the gates took it to 0.683, 0.481, 0.720. The code still takes them to the exact value in V; the stabiliser rule does not replace it, since the replica error is set in the preparation.
- **Which qubit the CX targets matters, the preparation's root does not help.** Target on the bulk qubit 0.258, on the seam qubit 0.133, as the coupling lists it 0.205: neither choice keeps the no-drift level. Preparing the GHZ from one end instead of from the seam qubit gives 0.332 without drift, no better than the seam root (0.355).
- **Scope.** The rule holds because the units are GHZ states, which is all `nighthawk_ghz.py` prepares; with a payload inside the unit (Layer 3) the inside ZZ is not a phase. On hardware the coupling is already invisible to a GHZ (Corollary 2.1); the rule only keeps ACE from turning that identity into seam crossings.

### 10.12 The loom as the seam, ghost band (E29)

> **Read with §10.14.** These runs use the ghost loom (`--loom-read ghost`, the only mode then). Each chip also simulates its partners' anchor qubits, so a 3-chain across a link is simulated whole on one chip; "Layer S's code" below reads it there. Nothing crosses the loom: as a hard cut (`--loom-read cut`, now the default) the same units sit at 0.50 in every pass, and 5-chains that no ghost band holds stay at 0.26 here too.

`nighthawk_ghz.py stack --engine loom --order code-echo` (README §3.6, METHODOLOGY §6.8): `nighthawk61r2`'s 61 qubits as two chips of 30 and 31 qubits (`rect:1-8,2-5`, `rect:1-8,6-9`, broken qubits 17, 55, 62 out), the size of its two ACE patches, each an exact `QrackSimulator` with a ghost of every partner it is linked to (37 and 38 qubits), joined by `nighthawk_cloverfield.py`'s ghost loom on the 7 rows where `nighthawk61r2` has a coupler between columns 5 and 6: the field has `nighthawk61r2`'s coupling graph. Layer S's code on the loom is the loom's authority: a unit is read on the chip that holds copies of all its qubits; without it every qubit is read on its own chip, the chips sampled apart (the cloverfield's estimator). 3-qubit chains as in §10.9, 256 shots × K = 4 per setting, exact reference 128 shots; Layer 2 dd, L = 3, the §8 model; container CPU, PyQrack 2.29.5 / Qrack 10.13.2. Mean F_GHZ:

| Units | Seam class, plan | Pass 1 (own chips) | C (authority read) | E (+ dd), gates | E (+ dd), inside ZZ dropped | Exact, E |
|---|---|---|---|---|---|---|
| off the seam (47) | bulk | 1.000 | 1.000 (pass 1) | 1.000 | 1.000 | 1.000–1.001 |
| holding a link anchor, on one chip (22 / 21) | replica, clean | 1.000–1.002 | (pass 1) | 1.000 | 0.998–1.000 | 0.997–1.001 |
| across a link, read inside a ghost band (13) | replica, code | **0.265** (0.250–0.280) | **1.000** | **0.996** (0.990–1.002) | **1.000** (0.996–1.005) | 0.999 |
| no chip holds it | cut | — (none) | — | — | — | — |

| Across a link (13), pass 1 → E | P0 + P1 | C = 2\|c\| |
|---|---|---|
| pass 1, every qubit on its own chip | 0.507 | 0.023 |
| E, read on the covering chip, gates | — | 0.992 |
| E, read on the covering chip, inside ZZ dropped | — | 1.000 |

(The run with the inside ZZ dropped planned one more unit for the code, [11, 7, 15], a column chain on one chip at 1.001 in pass 1 that the 3σ test against its exact value flagged; its pass-1 mean over the 13 link-crossing units is 0.264. E with the inside ZZ as gates took 193 s, dropped 58 s, on 2 cores; pass 1 15 s, C 1 s.)

| ACE (§10.9, §10.11) against the loom, `nighthawk61r2`, class means | Pass 1 | After the code | After code + echo, gates | … inside ZZ dropped |
|---|---|---|---|---|
| ACE, replica (17) | 0.76 | 1.00 | 1.00 | 1.00 |
| ACE, cut (7) | 0.36 | 0.36 | 0.22 | 0.35 |
| loom (ghost band), across a link (13) | 0.27 | 1.00 (read inside the band) | 1.00 | 1.00 |
| loom (ghost band), cut | none | none | none | none |
| loom as a hard cut, across a link (13; §10.14) | 0.50 | 0.50 | 0.50 | 0.50 |

- **No 3-chain is cut, because the ghost band is two columns wide.** Each chip holds a ghost of every link anchor across from it, so a 3-chain through a link has two qubits on one side and the third as a ghost there: chip 0 simulates [col 4, col 5, ghost col 6], chip 1 [ghost col 5, col 6, col 7]. On ACE the seam is one column, replicated into both patches, and a chain [patch 0, seam, patch 1] has no simulator holding it (§10.7). Chains wider than the band are cut on the loom too (§10.14).
- **Reading inside the band is not a correction across the loom.** Read on the chip whose band holds it, every link-crossing unit is at the exact value (1.000), where reading each qubit on its own chip gives 0.265 (P0 + P1 0.51, C 0.02). The 1.000 is the covering chip's own simulation of the whole unit: its ghost received every gate, the link CZ included, so the unit never crossed the loom. It says how wide the replica band is, not what the loom carries; the loom carries nothing (§10.14).
- **The echo inside the band.** E stays at 0.996 with the inside ZZ as gates and 1.000 with it dropped: on the covering chip every inside coupling is exact, and the mean-field phases land only on copies that are not read.
- **Against ACE.** ACE's seam is a slope: replica units keep part of their coherence through the reconciled replicas (0.55–0.95 before the code). The loom is a cliff: read on their own chips the crossing units have none (0.265 here, 0.50 as a hard cut, §10.14). The ghost band hides the cliff for the units it covers by simulating them whole on one chip, at the cost of each chip carrying its partners' anchors (37 and 38 qubits here).

### 10.13 Two-patch ACE chips joined by the loom, ghost band (E30)

> **Read with §10.14.** The loom here is the ghost loom (`--loom-read ghost`): "across the loom" units read 1.00 because one chip's ghost band simulates them whole, not because anything crosses the loom. ACE's numbers inside each chip are unaffected by the loom's mode.

`nighthawk_ghz.py stack --engine loom --loom-chip-backend ace --loom-chips nighthawk61r2 nighthawk61r2 --order code-echo` (README §3.6, METHODOLOGY §6.8): two copies of `nighthawk61r2`, each on its stored two-patch ACE preset (the cloverleaf), side by side as a 122-qubit field, joined by `nighthawk_cloverfield.py`'s ghost loom on the 7 rows whose facing border pair are lattice neighbours (rows 1–5, 7, 8; row 6 of the second chip starts at the broken qubit 62). Each chip holds a ghost of each partner it is linked to on a free ACE site coupled to the anchor (simulator widths with ghosts 42 / 34 and 35 / 41, against 35 / 34 without). Two seams in one field: ACE's inside each chip and the loom's between them. Layer S's code is both decoders, ACE's boundary repetition code inside each chip (lineage authority, the code's schedule) and the loom's authority across; without it ACE runs uncorrected and every qubit is read on its own chip. 3-qubit chains, 256 shots × K = 4 per setting, exact reference 128 shots; Layer 2 dd, L = 3, the §8 model; container CPU, PyQrack 2.29.5 / Qrack 10.13.2. Mean F_GHZ (range of E in brackets):

| Units | Pass 1 | C (code) | E (code + dd), gates | E (code + dd), inside ZZ dropped | Exact, E |
|---|---|---|---|---|---|
| off the seam (94) | 1.000 | 1.000 | 1.001 (0.991–1.010) | 1.001 (0.991–1.011) | 1.000 |
| ACE replica, inside a chip (56; 34 coded, 22 on their host simulator) | 0.847 / 0.843 | 0.999 | **1.001** (0.988–1.011) | **1.000** (0.989–1.007) | 1.000–1.001 |
| across the loom, read inside a ghost band (14) | 0.258 / 0.264 | 0.999 / 1.001 | **1.001** (0.995–1.006) | **0.999** (0.992–1.006) | 0.998–1.001 |
| ACE cut (14: 7 per chip) | 0.356 | 0.360 / 0.356 | **0.205** (0.074–0.293) | **0.358** (0.346–0.375) | 1.000 |

(Pass 1 and C per run, gates / dropped where they differ. Coherence C = 2\|c\| in E: 1.00 on every replica class, 0.04 on the cut units in both runs. Seconds on 2 cores: pass 1 392 / 390, C 253 / 219, E 1277 / 547.)

| Same layers, `nighthawk61r2` class means: one ACE chip (§10.9, §10.11), two exact chips on the loom (§10.12), two ACE chips on the loom | Pass 1 | After the code | E, gates | E, inside ZZ dropped |
|---|---|---|---|---|
| one ACE chip, replica | 0.76 | 1.00 | 1.00 | 1.00 |
| one ACE chip, cut | 0.36 | 0.36 | 0.22 | 0.35 |
| two exact chips, across the ghost loom | 0.27 | 1.00 (inside the band) | 1.00 | 1.00 |
| two ACE chips, ACE replica | 0.85 | 1.00 | 1.00 | 1.00 |
| two ACE chips, across the ghost loom | 0.26 | 1.00 (inside the band) | 1.00 | 1.00 |
| two ACE chips, cut | 0.36 | 0.36 | 0.21 | 0.36 |

- **The two seams add.** Inside each chip ACE's seam behaves as on one chip (§10.9: 0.85 with the hosted replica units, 1.00 under the code); between the chips the ghost loom behaves as with exact chips (§10.12: 0.26 read apart, 1.00 read inside a ghost band, which is a whole-unit simulation on one chip, not a transfer across the loom). The ghost on an ACE chip sits on a free site coupled to its anchor, so the link CZ is a real gate inside one ACE simulator of that chip.
- **ACE's cut class, once per chip.** No 3-chain across the loom is cut here, only because each is held by one chip's ghost band (§10.12, §10.14). The 14 cut units are `nighthawk61r2`'s 7 per chip: no ACE simulator holds them, coherence 0.04, F_GHZ ≤ ½. They lose 0.15 to the drift's inside ZZ under the echo (0.36 → 0.21), and keep their no-drift level with it dropped (0.358), as on one chip (§10.11).
- **The echo is exact on every corrected unit, with or without seam protection.** E − C is +0.003 / 0.000 on the coded units; seam protection matters only for the cut units, and cuts pass E's time from 1277 s to 547 s.
- **What the field shows and what it does not.** Twice the qubits and two ACE seams: every ACE class is where it is on one chip. Across the loom the 1.00 is the ghost band's; as a hard cut the same units are separable across the loom (§10.14).

### 10.14 The loom as a hard cut, and what the ghost band does (E31)

`nighthawk_ghz.py stack --engine loom` now models the loom as what it is between independent chip processes, a hard cut (`--loom-read cut`, the default): each chip an independent simulator with no copy of any qubit of the other, a link CZ reaching each side only as the mean-field phase π P(1) of the other (the cloverfield's `--loom mean`), every qubit read on its own chip. The ghost loom of §10.12–10.13 stays available as `--loom-read ghost`. `--order code-echo`, 3-qubit chains, 256 shots × K = 4 per setting, exact reference 128 shots (5-chains: 128 × 2, exact 64); Layer 2 dd, L = 3, the §8 model; container CPU, PyQrack 2.29.5 / Qrack 10.13.2.

| Ghost loom, 5-qubit chains, two exact chips (`rect:1-8,2-5` + `rect:1-8,6-9`) | Units | Class | Pass 1 | C ("code") | E | Exact |
|---|---|---|---|---|---|---|
| across a link, 4 qubits on one side + 1 | 12 | replica: inside one chip's ghost band | 0.271 | **0.999** | **1.002** | 1.000 |
| across a link, 2 + 3 | 12 | cut: no ghost band holds them | 0.272 | **0.265** | **0.263** (0.24–0.31) | 1.004 |
| on one chip | 24 | bulk, replica | 1.000–1.001 | — | 1.000–1.001 | 1.002–1.003 |

| Hard cut, two exact chips of 30 + 31 (`nighthawk61r2`'s qubits) | Units | Pass 1 | C | E, gates | E, inside ZZ dropped | Exact |
|---|---|---|---|---|---|---|
| on one chip | 69 | 1.001 / 1.000 | (pass 1) | 1.000 | 1.001 | 1.000 |
| across a link | 13 | **0.502 / 0.506** | **0.499 / 0.500** | **0.499** | **0.502** | 1.000–1.001 |

| Hard cut, two `nighthawk61r2` ACE chips (122 qubits) | Units | Pass 1 | C (ACE's code) | E, gates | E, inside ZZ dropped | Exact |
|---|---|---|---|---|---|---|
| off the seams | 116 | 1.001 / 0.999 | (pass 1) | 1.000 (0.99–1.01) | 1.001 (0.99–1.01) | 1.000 |
| ACE replica, inside a chip | 34 | 0.749 / 0.755 | 1.001 / 1.000 | **1.000** (0.99–1.01) | **1.000** (0.99–1.01) | 0.998–1.006 |
| ACE cut, inside a chip (7 per chip) | 14 | 0.361 / 0.354 | 0.364 / 0.355 | **0.201** (0.08–0.29) | **0.354** (0.33–0.38) | 1.000 |
| across the loom | 14 | **0.501 / 0.499** | **0.504 / 0.499** | **0.498** (0.48–0.51) | **0.502** (0.48–0.53) | 0.997–0.999 |

(Pass 1 and C per run: gates / dropped. Seconds on 2 cores, gates / dropped: pass 1 370 / 381, C 241 / 235, E 1218 / 507.)

| A slope and a cliff: pass 1 per unit, two ACE chips, hard cut | F_GHZ | sd over units | Range | P0 + P1 | C = 2\|c\| | Under the drift (E, no protection) |
|---|---|---|---|---|---|---|
| ACE replica (the seam inside a chip) | 0.749 | 0.132 | 0.56–0.93 | 0.846 | 0.653 | 1.000 under the code |
| ACE cut | 0.361 | 0.015 | 0.34–0.39 | 0.670 | 0.052 | 0.201 (0.08–0.29) |
| across the loom | 0.501 | 0.008 | 0.48–0.51 | 0.502 | 0.500 | 0.498 (0.48–0.51) |

- **Why the ghost loom looked like a perfect correction.** With ghosts, each chip also simulates its partners' anchor qubits and the loom replays every gate on them, the link CZ included, exactly. Each chip so simulates a band of two columns beyond its own edge, and a 3-chain across a link always lies inside one chip's band: read there, it is that chip's own exact simulation of the whole unit. Nothing crossed the loom. 5-chains test it: those with four qubits on one side lie in a band and read 0.999; those split 2 + 3 do not, and stay at 0.26 in every pass.
- **A hard cut caps every unit across it at ½.** No entanglement crosses, so a unit across the loom is separable there, and F_GHZ ≤ ½ for any state separable across a cut (F_GHZ > ½ witnesses genuine multipartite entanglement). The hard cut sits exactly on the bound: each side is prepared perfectly, a Bell pair on one chip and a definite single-qubit state on the other, so P0 + P1 = ½ and C = ½ (the product of the two sides' own coherences), F_GHZ = 0.50 with a spread of shot noise (sd 0.008). No code lifts it, and the echo cannot lower it: across the cut the drift is a local phase on each side, which the fidelity's best phase θ absorbs.
- **ACE's seam is a slope, the loom a cliff.** ACE keeps part of a unit's coherence through its reconciled replicas, so its seam loss is graded and unit-dependent (replica 0.56–0.93, sd 0.13), sensitive to the drift (the cut class 0.36 → 0.20) and partly correctable (replica → 1.00). The loom keeps none and loses none further: every unit across it is at 0.50, in every pass, with or without protection. The heatmaps show it as a neutral band two columns either side of the links, flat through all three layers, next to ACE's red slope around each chip's seam.
- **Seam protection matters only inside ACE.** It holds ACE's cut units at their no-drift level (0.354 against 0.201) as on one chip (§10.11); across the hard cut there is nothing for the drift to damage.
- **What would carry a GHZ state across the loom.** A transport gadget: a pair entangled across the two chips, consumed by measurement and classically controlled corrections (a teleported CZ). The measurement outcomes are classical and the loom can carry them, but the shared pair is a joint state of two chips, which independent chip simulators cannot hold. The ghost loom holds it by simulating the far side on the near chip, which is a wider replica band, not a channel. Until a joint state is simulated somewhere, every GHZ unit across the loom is capped at ½.

### 10.15 QrackBridge in place of the loom (E32)

`nighthawk_ghz.py stack --engine bridge` (README §3.6, METHODOLOGY §6.8) joins the same two exact chips as §10.14 (`rect:1-8,2-5` + `rect:1-8,6-9`, `nighthawk61r2`'s 61 qubits, 7 links) with PyQrack's `QrackBridge` (Claude and Dan Strano; `qrack_bridge.py` of PyQrack 2.30.2, loaded onto PyQrack 2.29.5 / Qrack 10.13.2) instead of the loom: one patch per chip, a gate across the cut teleported through a pair of bond qubits, the cut recompressed to at most `max_bonds` bond qubits (greedy truncation, eager). Each circuit's bridge holds its units' qubits (widest: 30, 15 per patch). `--order code-echo`, 3-qubit chains, 256 shots × K = 4 per setting, exact reference 128 shots; Layer 2 dd, L = 3, the §8 model; container CPU. Every unit across the bridge is class bridge (no code; C reruns it as the control). Mean F_GHZ:

| `max_bonds` | Off the seam (69): pass 1 / E | Across the bridge (13): pass 1 | C | E | Exact | Peak bonds per cut | Truncating circuits (pass 1) | `fidelity_estimate`, truncating circuits |
|---|---|---|---|---|---|---|---|---|
| 6 | 1.000 / 1.000 | **1.000** | 1.000 | **1.001** (0.99–1.01) | 1.001 | 4 (5 in E) | 0 of 684 | — |
| 2 | 1.000 / 1.000 | **0.849** | 0.851 | **0.851** (0.51–1.01) | 1.000 | 3 | 108 of 684 | down to 0.25 |
| 1 | 1.000 / 1.000 | **0.736** | 0.735 | **0.735** (0.50–1.00) | 0.999 | 2 | 108 of 684 | down to 0.125 |
| 1, inside ZZ dropped | 1.000 / 1.000 | 0.736 | 0.735 | 0.734 (0.50–1.00) | 0.998 | 2 | 108 of 684 | down to 0.125 |
| 0 | 1.000 / 1.000 | **0.509** | 0.508 | **0.511** (0.50–0.52) | 0.999 | 1 | 216 of 684 | down to 0.0625 |
| loom, hard cut (§10.14) | 1.001 / 1.000 | 0.502 | 0.499 | 0.499 | 1.001 | — | — | — |

| E per unit, by how many units cross the cut in its circuit | 1 per circuit (3 units) | 3 per circuit (6) | 4 per circuit (4) |
|---|---|---|---|
| bonds an exact cut needs | 1 | 3 | 4 |
| `max_bonds` 6 | 0.999 | 1.001 | 1.001 |
| `max_bonds` 2 | 0.999 | 0.839 = (2 · 1 + ½) / 3 | 0.758 = (2 · 1 + 2 · ½) / 4 |
| `max_bonds` 1 | 0.999 | 0.673 = (1 + 2 · ½) / 3 | 0.629 = (1 + 3 · ½) / 4 |
| `max_bonds` 0 | 0.511 | 0.511 | 0.511 |

(Seconds on 2 cores, pass 1 / E: 47 / 392 at 6, 40 / 294 at 2, 40 / 273 at 1, 43 / 81 at 1 with the inside ZZ dropped, 40 / 238 at 0.)

- **The bridge carries entanglement across the cut.** At `max_bonds` 6 every unit across it is at the exact value through all three layers (1.000, 1.001 in E), with no truncation: a GHZ unit across the cut needs one bond qubit, a circuit with k such units k of them (Schmidt rank 2^k), here at most 4 (5 transiently while a drift coupling crosses). That is what the loom cannot do (§10.14), and what the ghost loom only imitated by simulating whole units on one chip (§10.12).
- **Its capacity rations whole units.** With the cut's Schmidt values all equal, the greedy truncation keeps `max_bonds` units' worth of entanglement and drops the rest: per circuit, `max_bonds` units stay at 1.00 and the others become separable across the cut, at the hard-cut value ½. A unit's fidelity is so either 1.00 or 0.50, and the class mean is the share carried: (carried + ½ · dropped) / units, which the per-circuit table reproduces to 0.01. Units alone on the cut in their circuit are carried at any capacity ≥ 1.
- **At zero capacity the bridge is the hard cut.** `max_bonds` 0 keeps one Schmidt value per cut: every unit across it at 0.51 in every pass, as the loom's hard cut (0.50).
- **The echo and seam protection.** E equals pass 1 at every capacity: the drift's couplings across the cut are bridged gates, exact within the capacity, and a unit already dropped is separable, where the drift is a local phase. Dropping the inside ZZ changes no fidelity (0.734 against 0.735 at `max_bonds` 1); it saves the bridged gates (E 81 s against 273 s).
- **Three seams, three shapes.** ACE's seam inside a chip is a slope: every replica unit keeps part of its coherence (0.56–0.93), the cut class none, and the drift erodes them (§10.14). The loom between independent chips is a cliff at ½. The bridge is a budget: entanglement crosses exactly up to `max_bonds` bond qubits per cut, so its fidelity falls in steps of whole units as the demand on the cut exceeds the capacity, from 1.00 to the cliff at 0.


### 10.16 A closed box: six faces on a QrackBridge ring (E33)

`nighthawk_ghz.py stack --engine bridge --box 5 5` (README §3.6, METHODOLOGY §6.8): the 98 lattice points on the surface of a 5 × 5 × 5 block, 192 couplers at unit distance, six circuits (base 25, front 12, right 12, lid 25, back 12, left 12 qubits; base and lid whole, the walls sharing the vertical edges round the box), each a QrackBridge patch, the six patches closed into a ring (`wrap=True`) in the order base → front → right → lid → back → left → base. Opposite faces sit three apart on the ring and share no edge, so of the 12 box edges six are a ring cut (base–front, front–right, right–lid, lid–back, back–left, left–base, the last one the ring's wrap cut) and six are relayed through the face between (base–right via front, front–lid via right, lid–left via back, base–back via left, front–left via base, right–back via lid), one bond on each of two cuts. 3-qubit chains along straight lines in x, y and z: 144 units in 18 circuits, 100 inside a face, 22 across a direct edge, 22 across a relayed edge (every one of the 44 cross-face couplers in one unit). Bipartite under x + y + z. `--order code-echo`, 256 shots × K = 4 per setting, exact reference 128 shots on one 98-qubit `QrackSimulator`; Layer 2 dd, L = 3, the §8 model; greedy truncation, eager; container CPU, `qrack_bridge.py` of PyQrack 2.30.2 on PyQrack 2.29.5 / Qrack 10.13.2. Each circuit's bridge holds its units' qubits, 9 per patch at most (widest bridge 54). Mean F_GHZ, pass 1 / C / E (exact E):

| `max_bonds`, inside ZZ | Inside a face (100) | Direct edge (22) | Relayed edge (22) | E range across edges | Peak bonds per cut | Truncating circuits (pass 1 / E of 648) | `fidelity_estimate` min (E) | Seconds, pass 1 / C / E |
|---|---|---|---|---|---|---|---|---|
| 6, gate | 1.000 / 1.000 / 1.000 (1.001) | **0.998 / 1.001 / 1.001** (1.001) | **1.000 / 0.999 / 1.000** (0.999) | 0.99–1.01 | 4 (5 in E) | 0 / 0 | 1.000 | 136 / 86 / 819 |
| 6, dropped | 1.000 / 1.000 / 1.000 (1.000) | 0.999 / 1.000 / 1.001 (0.999) | 0.999 / 0.999 / 1.001 (1.001) | 0.99–1.01 | 4 | 0 / 0 | 1.000 | 139 / 84 / 194 |
| 2, gate | 1.000 / 1.000 / 1.000 | **0.763 / 0.777 / 0.768** (1.000) | **0.843 / 0.835 / 0.812** (1.001) | 0.50–1.01 | 3 | 144 / 180 | 0.036 | 67 / 16 / 473 |
| 2, dropped | 1.000 / 1.000 / 1.000 | 0.772 / 0.761 / 0.773 (1.001) | 0.840 / 0.847 / 0.840 (1.000) | 0.50–1.01 | 3 | 144 / 144 | 0.063 | 70 / 16 / 138 |
| 1, gate | stopped in pass E: the ring's truncation zeroed the state (0 / 0, then an SVD that does not converge) | | | | 2 | 288 / — | — | — |
| 1, dropped | 1.000 / 1.000 / 1.000 | **0.667 / 0.657 / 0.668** (1.000) | **0.691 / 0.692 / 0.696** (1.000) | 0.50–1.01 | 2 | 288 / 288 | 0.016 | 84 / 15 / 129 |
| 0, gate | stopped in pass E, as at 1 | | | | 1 | 432 / — | — | — |
| 0, dropped | 1.000 / 1.000 / 1.000 | **0.509 / 0.509 / 0.508** (1.001) | **0.509 / 0.509 / 0.510** (0.999) | 0.50–0.52 | 1 | 432 / 432 | 0.004 | 61 / 11 / 127 |

E per box edge (units):

| Edge | Ring cuts | Units | 6 | 6, dropped | 2 | 2, dropped | 1, dropped | 0, dropped |
|---|---|---|---|---|---|---|---|---|
| back–left | 1 | 3 | 1.002 | 1.002 | 1.002 | 1.002 | 0.835 | 0.510 |
| front–right | 1 | 3 | 1.000 | 1.001 | 1.001 | 1.001 | 0.837 | 0.507 |
| base–front | 1 | 4 | 1.000 | 1.001 | 0.973 | 0.960 | 0.755 | 0.511 |
| base–left (wrap cut) | 1 | 4 | 1.000 | 1.000 | 0.731 | 0.774 | 0.646 | 0.505 |
| right–lid | 1 | 4 | 1.001 | 1.003 | 0.510 | 0.509 | 0.508 | 0.509 |
| lid–back | 1 | 4 | 1.000 | 1.001 | 0.509 | 0.508 | 0.510 | 0.509 |
| right–back via lid | 2 | 3 | 0.999 | 1.001 | 1.001 | 1.001 | 0.836 | 0.506 |
| front–left via base (and the wrap cut) | 2 | 3 | 1.000 | 0.997 | 0.778 | 0.999 | 0.819 | 0.509 |
| front–lid via right | 2 | 4 | 1.002 | 1.004 | 1.003 | 1.003 | 0.754 | 0.509 |
| lid–left via back | 2 | 4 | 0.997 | 1.002 | 0.999 | 0.995 | 0.753 | 0.512 |
| base–right via front | 2 | 4 | 0.998 | 0.998 | 0.534 | 0.522 | 0.512 | 0.509 |
| base–back via left (and the wrap cut) | 2 | 4 | 1.002 | 1.002 | 0.598 | 0.598 | 0.567 | 0.513 |

The demand on a cut is the number of units of one circuit crossing it. The z-chain circuits load four cuts with four units each, two direct and two relayed: base|front (base–front, base–right via front), right|lid (right–lid, front–lid via right), lid|back (lid–back, lid–left via back) and the wrap cut left|base (base–left, base–back via left). Every other cut of every circuit carries at most 2. E over the 8 units at each demand-4 cut, against the budget (min(d, max_bonds) · 1 + the rest · ½) / d of §10.15:

| Cut (demand 4) | `max_bonds` 2, gate | 2, dropped | 1, dropped | Budget at 2 / at 1 |
|---|---|---|---|---|
| base \| front | 0.754 | 0.741 | 0.633 | 0.750 / 0.625 |
| right \| lid | 0.756 | 0.756 | 0.631 | 0.750 / 0.625 |
| lid \| back | 0.754 | 0.752 | 0.632 | 0.750 / 0.625 |
| left \| base, the wrap cut | **0.665** | **0.686** | **0.607** | 0.750 / 0.625 |
| front–left via base, 2 units, demand 2 on base \| front and the wrap (pass 1 → E) | 1.002 → **0.666** | 1.005 → 1.000 | — | 1 at 2 |

`bridge_ring_repro.py` (in this folder) takes the last row apart: the two front–left chains of that circuit, gate for gate, on a 24-qubit bridge of six patches, ring against open chain (where the same gates relay the long way, through four cuts):

| `max_bonds` | Ring: `fidelity_estimate`, F_GHZ of the two chains (`out_ket`) | Open chain |
|---|---|---|
| 0 | the state vanishes in some runs (0 / 0 in the recompression); else 0.016, 0.50 / 0.50 | 0.250, 0.50 / 0.50 |
| 1 | 0.250, **0.50 / 0.50** | 0.500, **1.00** / 0.50 |
| 2 | **0.499** | 1.000 |
| 3 | 1.000 | 1.000 |

- **The box closes, and at capacity every edge is exact.** With `max_bonds` 6 (the peak demand is 4 bond qubits per cut, 5 while a drift coupling crosses) all 44 units across a box edge stay at the exact value through the three layers, 1.000 and 1.001 in E, the 22 relayed through a third face as the 22 on a direct cut; the bridge truncates nothing. Relaying costs bonds, one on each cut it passes, and no fidelity while they are there.
- **Below capacity the inner cuts ration whole units, as between two chips.** At the three demand-4 inner cuts the mean is the §10.15 budget, 0.754 / 0.756 / 0.754 against 0.750 at `max_bonds` 2 and 0.63 against 0.625 at 1: per circuit `max_bonds` units keep their GHZ at 1.00 and the rest drop to the separable ½. Which keep it is the greedy truncation's choice between equal Schmidt values: here the relayed units at right|lid and lid|back (front–lid and lid–left stay at 1.00, right–lid and lid–back drop to 0.51), the direct one at base|front (base–front 0.97, base–right via front 0.53). Units alone or in pairs on their cuts keep 1.00 at 2 and 0.75–0.84 at 1.
- **The ring's wrap cut is below the budget.** left|base: 0.665 / 0.686 at 2 (budget 0.750), 0.607 at 1 (0.625). QrackBridge truncates the wrap cut by a local SVD of its two patches, which on a ring is not the projection on the kept Schmidt vectors (a periodic MPS has no canonical form; the bridge's notes say its truncation there is locally optimal only and its `fidelity_estimate` an estimate). Within capacity it should cost nothing and it does: the two front–left chains through the wrap keep 1.00 in pass 1 and fall to 0.67 in E at `max_bonds` 2, when the drift's CX · Rz · CX cross the wrap again. Dropping the inside ZZ keeps those gates off the bridge and the chains at 1.00. The repro shows the same at 1 (the ring carries neither chain where the open chain carries one) and at 2 (`fidelity_estimate` 0.5 against 1.0).
- **At 0 and 1 the ring can zero the state.** With the inside ZZ as gates, pass E stopped at `max_bonds` 1 and 0: a truncation of the wrap cut and of an inner cut kept opposite branches of tied GHZ units, the contracted state was zero, its renormalisation divided 0 / 0 and the NaN state broke the next SVD. Which branch a tie keeps follows rounding, so the same circuit vanishes in some runs and not in others. `nighthawk_ghz.py` now raises numpy's divide and invalid errors inside every bridge circuit and stops with the circuit named, so a NaN state is never sampled. With `--inside-zz stabilizer` no bridged gate runs in E and both capacities complete.
- **At 0 the ring is the hard cut.** Every unit across an edge at 0.51 in every pass and on every edge, relayed or not, as the two-chip bridge at 0 (0.511) and the loom's hard cut (0.50).
- **The echo and seam protection.** E equals pass 1 wherever the ring's wrap is not involved: inside the faces 1.000 in every run, across the edges within 0.01 of pass 1 at 6 and at the inner cuts. Dropping the inside ZZ saves the bridged gates in E (194 s against 819 s at 6, 138 s against 473 s at 2).


### 10.17 Shor's code across QrackBridge (E34)

`nighthawk_ghz.py shor` (README §3.6, METHODOLOGY §6.8) puts Shor's [[m², 1, m]] code across QrackBridge in place of a GHZ chain, encoded as `nighthawk_qrack.py shor9` encodes it (§10.3, E20). Every m × m placement whose encoder CXs are couplers runs in its own circuit, with all four logical states, and raw and decoded success are measured against an exact `QrackSimulator` of the unit. There are two geometries:

- Nighthawk's 117 sites as two exact chips, `rect:0-11,0-5` + `rect:0-11,6-9` (70 + 47 qubits, 11 links on rows 0–4 and 6–11).
- The 5 × 5 × 5 box of §10.16 (six patches on a ring), where a unit bends over the box edge it crosses.

The cut meets a unit in one of three ways:

- inside one circuit;
- between its blocks: only leader-to-leader CXs cross, as E20's `patch_blocks`;
- through its blocks: every block's own CXs cross, as E20's `seam_leaders`.

Runs used 1024 shots per logical state (Shor-25: 256), exact 256 (128), greedy truncation, eager recompression, container CPU. Shor-9 and Shor-16 take 1.5–11 s per run on 2 cores; Shor-25 across the cut takes 194–323 s, almost all of it in the bridge's sampler over 2^20 outcomes for the X-basis readout. The default two chips (`rect:1-8,2-5` + `rect:1-8,6-9`, 61 qubits, 5 + 5 crossing units) gave the same numbers as the 117-qubit pair to 0.01.

Shor-9, decoded logical Z / logical X (logical Z is the mean of \|0_L⟩ and \|1_L⟩; logical X of \|+_L⟩ and \|−_L⟩). Raw logical Z of the through-blocks units is in brackets:

| `max_bonds` | Chips: inside (88) | Chips: between (12) | Chips: through (12) | Box: inside (48) | Box: between (24) | Box: through, not the wrap (18) | Box: through the wrap cut (6) |
|---|---|---|---|---|---|---|---|
| 6 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 (1.000) / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 (1.000) / 1.000 | 1.000 / 1.000 |
| 3 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 (1.000) / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 (1.000) / 1.000 | 1.000 / 1.000 |
| 2 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 (0.506)** / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 (0.502)** / 1.000 | **0.807** / 1.000 |
| 1 | 1.000 / 1.000 | 1.000 / 1.000 | **0.745 (0.502)** / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **0.752 (0.495)** / 1.000 | 0.728 / 1.000 |
| 0 | 1.000 / 1.000 | 1.000 / **0.495** | **0.495 (0.497)** / 1.000 | 1.000 / 1.000 | 1.000 / **0.498** | **0.503 (0.497)** / 1.000 | 0.501 / 1.000 |

At `max_bonds` 0 on the box, 3 of the 384 logical-state circuits vanished on the ring's truncation (§10.16): two through the wrap, one between blocks. They are not scored. Exact reference: 1.000 for every unit and state.

Bond qubits per cut that the prepared state keeps at full capacity (the most over the units):

| Code | Between blocks: \|0/1_L⟩ / \|±_L⟩ | Through blocks: \|0/1_L⟩ / \|±_L⟩ |
|---|---|---|
| Shor-9 | 0 / 1 | 3 / 2 |
| Shor-16 | 0 / 1 | 4 / 3 |
| Shor-25 (one unit each, `max_bonds` 6) | 0 / 1 | 5 / 4 |
| Shor-9 on the box, a unit through the wrap cut | 1 / 1 | 3 / 3 |

Through blocks, decoded logical Z against the majority vote over m block signs, with min(m, `max_bonds`) carried and the rest coin flips (ties ½). Raw is in brackets: for odd m the parity of all outcomes, for even m the first block's sign:

| `max_bonds` | Shor-9, chips (12) | Majority | Shor-16, chips (11) | Majority | Shor-25, chips (8) | Majority |
|---|---|---|---|---|---|---|
| 4 | — | — | 1.000 (1.000) | 1 | — | — |
| 3 | 1.000 (1.000) | 1 | 1.000 (1.000) | 1 | **1.000** (0.495) | 1 |
| 2 | 1.000 (0.506) | 1 | **0.874** (1.000) | 0.875 | **0.875** (0.501) | 0.875 |
| 1 | **0.745** (0.502) | 0.750 | **0.688** (1.000) | 0.688 | **0.677** (0.497) | 0.688 |
| 0 | 0.495 (0.497) | 0.500 | 0.502 (0.502) | 0.500 | — | 0.500 |

Against ACE's seam (§10.3, E20, the 3 × 3 register):

| Seam | Between blocks: logical Z / X | Through blocks: logical Z / X |
|---|---|---|
| ACE, `long_range_columns = 1` (E20) | 1.000 / 0.486 | 0.499 / 0.669 |
| QrackBridge, `max_bonds` 0 | 1.000 / 0.495 | 0.495 / 1.000 |
| QrackBridge, `max_bonds` 1 | 1.000 / 1.000 | 0.745 / 1.000 |
| QrackBridge, `max_bonds` ≥ m | 1.000 / 1.000 | 1.000 / 1.000 |

- **What crosses is countable.**
  - Cut between blocks: \|0_L⟩ and \|1_L⟩ are products of block states, so they need no bond. \|±_L⟩ is their superposition and needs one, which is the logical coherence itself.
  - Cut through blocks: every split block's GHZ needs one bond for \|0/1_L⟩ (m in all), and m − 1 for \|±_L⟩.
  - The bridge keeps exactly these at capacity, and every unit stays at 1.000.
- **Short of capacity, the bridge drops whole blocks, and the code corrects that.**
  - A dropped block keeps its bits and loses its sign. \|±_L⟩, read in Z, stays at 1.000 at any capacity; \|0/1_L⟩, read in X, sees m − `max_bonds` random block signs.
  - The decoder's majority vote then gives what counting predicts: Shor-9 0.745 at 1 (0.750), Shor-16 0.874 / 0.688 at 2 / 1 (0.875 / 0.688), Shor-25 0.875 / 0.677 at 2 / 1 (0.875 / 0.688).
  - Up to ⌊(m − 1)/2⌋ blocks short, the raw parity is 0.50 and the decoded value 1.000: Shor-9 at 2, Shor-16 at 3, Shor-25 at 3 (two blocks short). The bridge's truncation is the error the code is built for: a block-sign error confined to whole blocks.
  - ACE's seam is different (E20: decoding cannot repair it). There, every block through the seam loses part of its coherence at once, and part of its bits too (logical X 0.67).
- **At 0 the bridge is ACE's seam on the side that needs coherence.**
  - Between blocks, logical X is 0.495 (ACE 0.486); through blocks, logical Z is 0.495 (ACE 0.499).
  - On the side that needs only bits, the bridge keeps 1.000 where ACE drops to 0.669: a hard cut destroys coherence across it, while ACE's seam also scrambles populations.
- **On the box, direct and relayed edges are alike; the wrap cut is again the exception.**
  - Units across a direct or a relayed edge give the chips' numbers.
  - Through-blocks units across the ring's wrap cut fall below the count at 2 (0.807 against 1.000), and at 1 (0.728 against 0.750). They keep a bond more than they need (3 for \|±_L⟩, 1 for \|0/1_L⟩ between blocks), and at 0 some circuits vanish (§10.16).


### 10.18 IBM's two-module Nighthawk R1 on the bridge (E35)

The paper is N. M. Sundaresan et al. (IBM), *Meter-scale quantum interconnects between large quantum processors*, arXiv:2610.29986 (21 October 2026). It joins two 120-qubit IBM Nighthawk R1 modules with four L-couplers: about 80 cm of superconducting aluminium coax each, with a tunable L-qubit at each end coupled to one lattice qubit of row 11. Its measured numbers:

- state transfer, network-benchmarked: mean 99.00 %;
- distant CZ, randomized-benchmarked: mean 94.75 %, best 96.42 %;
- Bell pairs, GHZ-4 and GHZ-6 across the modules, raw and readout-mitigated (Table 1, Extended Data Table 3).

`nighthawk_ghz.py dual` (README §3.6, METHODOLOGY §6.8) builds that setup:

- the two modules are the two patches of a QrackBridge;
- the cable is a state transfer, a SWAP across the bridge;
- the distant CZ is the paper's sequence: CNOT into the L-qubit, transfer, CZ, transfer back, CNOT.

It runs the paper's circuits under five models. Noise comes from 20,000 Pauli trajectories per circuit, and every fidelity is computed from the trajectory-averaged density matrix with the paper's own estimators. The models carry no readout error, so the column to compare with is IBM's readout-mitigated one. The run took 290 s on 2 cores.

| State (lattice qubits, L-coupler) | Hard cut (bridge, `max_bonds` 0) | Bridge, exact (= QrackSimulator) | Bridge + constituent budget | Bridge + measured distant CZ | IBM mitigated | IBM raw |
|---|---|---|---|---|---|---|
| Bell 111–237 (240–247) | 0.500 | 1.000 | 0.966 ± 0.001 | 0.963 ± 0.001 | — | — |
| Bell 113–235 (241–246) | 0.500 | 1.000 | 0.952 ± 0.001 | 0.913 ± 0.002 | — | — |
| Bell 115–233 (242–245) | 0.500 | 1.000 | 0.955 ± 0.001 | **0.954 ± 0.001** | **0.9394 ± 0.0027** | 0.8591 |
| Bell 117–231 (243–244) | 0.500 | 1.000 | 0.960 ± 0.002 | **0.956 ± 0.001** | **0.8698 ± 0.0041** | 0.7693 |
| Bell 115–233, simultaneous | 0.500 | 1.000 | 0.957 ± 0.001 | 0.954 ± 0.001 | 0.8756 ± 0.0038 | 0.8040 |
| Bell 117–231, simultaneous | 0.500 | 1.000 | 0.958 ± 0.001 | 0.954 ± 0.001 | 0.8614 ± 0.0058 | 0.7617 |
| GHZ-4: 115, 116, 232, 233 | 0.500 | 1.000 | 0.949 ± 0.002 | **0.944 ± 0.002** | **0.9170 ± 0.0070** | 0.7723 |
| GHZ-6: 115–117, 231–233 | 0.500 | 1.000 | 0.941 ± 0.001 | **0.938 ± 0.002** | **0.9087 ± 0.0074** | 0.6767 |

Model inputs, all from the paper unless marked:

| Input | Value | Source |
|---|---|---|
| state transfer F per link (240–247, 241–246, 242–245, 243–244) | 99.29, 98.87, 98.99, 98.86 % | Extended Data Table 2 |
| distant CZ F per link | 96.42, 91.2, 95.54, 95.8 % | text (96.42, 95.54); Fig. 3b read (91.2, 95.8), mean 94.75 as in the text |
| L-qubit single-qubit error | 0.04–0.30 % | Extended Data Table 1 |
| local CZ inside a module | 0.355 % | the paper's budget for CZ111-237 (three local CZs = 40 % of 2.66 %); no lattice CZ benchmark is given |
| lattice single-qubit error | 0.05 % | assumed |
| readout, idle decoherence, crosstalk, leakage | not modelled | — |

Sensitivity to the local CZ error (bridge + measured distant CZ, 10,000 trajectories):

| Local CZ error | GHZ-4 | GHZ-6 | GHZ-4 → GHZ-6 | Bell 115–233 → GHZ-4 |
|---|---|---|---|---|
| 0 | 0.953 ± 0.002 | 0.955 ± 0.002 | +0.002 | −0.002 |
| 0.355 % (default) | 0.944 ± 0.002 | 0.938 ± 0.002 | −0.006 | −0.010 |
| 0.71 % | 0.937 ± 0.002 | 0.921 ± 0.003 | −0.016 | −0.018 |
| IBM, mitigated | 0.917 | 0.909 | −0.008 ± 0.010 | −0.022 ± 0.008 |

Over the whole dual chip (`dualmap`, 2000 trajectories per lattice chain, 4000 per link chain, 442 s): 3-qubit GHZ chains along every row and column of both modules (392) and every chain through an L-coupler (24: the link's pair and a neighbour of either end), mean F_GHZ:

| Chains | Hard cut | Constituent budget | Measured distant CZ |
|---|---|---|---|
| module A (196) | 1.000 | 0.992 | 0.992 |
| module B (196) | 1.000 | 0.992 | 0.992 |
| through 240–247, 111–237 (6) | 0.500 | 0.967 | 0.953 |
| through 241–246, 113–235 (6) | 0.500 | 0.947 | **0.894** |
| through 242–245, 115–233 (6) | 0.500 | 0.953 | 0.945 |
| through 243–244, 117–231 (6) | 0.500 | 0.955 | 0.947 |

Inside a module the model is uniform (one local CZ error for every coupler; the paper gives no lattice calibration), so the map's structure is the links: every chain through a cable sits 4–10 points below the lattice, 113–235 lowest. Under the hard cut, the qubits those chains touch (row 11 and the row above it at the link columns) are at ½.

Bond qubits per cut, at their peak: 3 for one distant CZ (the SWAP's ship-and-return takes two, the Bell pair one), 4 for the two simultaneous pairs.

- **The interconnect is a quantum channel, not a loom.** Every state in IBM's table is above ½ without readout mitigation, which for the GHZ states certifies genuine multipartite entanglement across the modules. The hard cut gives 0.500 for every circuit, exactly the bound (§10.14), so a loom-like model of the cable cannot reproduce the paper. The bridge carries each of these states with a bond or two to spare.
- **The benchmarked links explain most of what was measured.**
  - With each link's distant CZ as a single benchmarked channel, the model predicts:

    | State | Model | IBM mitigated | Gap |
    |---|---|---|---|
    | Bell 115–233 | 0.954 | 0.939 | 1.5 points |
    | GHZ-4 | 0.944 | 0.917 | 2.7 points |
    | GHZ-6 | 0.938 | 0.909 | 2.9 points |

  - The step from GHZ-4 to GHZ-6 (two more local CNOTs) is −0.006 in the model and −0.008 ± 0.010 measured, consistent with the paper's local CZ budget.
  - The extra loss sits earlier: Bell → GHZ-4 costs −0.010 in the model and −0.022 measured. This is consistent with the paper's own attribution: residual L-qubit population in the distant gate, and idling during it, neither of which a gate-level model has.
- **The constituent budget is optimistic where the paper says so.** The budget model sits above the measured-link model on every link, 0.952 against 0.913 on 113–235. It is the same upper bound the paper's budget gives (97.34 % against 96.42 % for CZ111-237): the constituents compose to less error than the benchmarked gate shows.
- **Two rows fall outside the model, and the paper names both causes.**
  - Bell 117–231 is 0.870 mitigated against 0.956 predicted, though its link is as good as 115–233's. The paper ties its numbers to Q231, the worst-readout qubit of the ring; mitigation evidently does not remove all of it.
  - Firing both pairs together drops 115–233 by 0.064 (0.876 against 0.939). The gate model has no crosstalk and predicts no change.
- **How the bridge would scale.** One distant CZ peaks at 3 bond qubits on QrackBridge: two for the transfer's SWAP, one for the entanglement left behind. A Bell pair per link uses one bond each, so the four L-couplers between two modules match a bridge of `max_bonds` 4. That is also the peak demand our 3-chain tilings put on a cut (§10.15–10.16), and enough for one Shor-9 unit cut through its blocks (3 bonds) or four cut between them (1 each, §10.17).

### 10.19 Layer 3, Shor order finding, across the configurations (E36)

The formal proof's Layer 3 is Shor order finding for N = 15, a = 7, with 4 counting and 4 work qubits, scored by the per-shot factoring success (`shrs_nighthawk_stealth.py`, `Shor`). `nighthawk_ghz.py shoralg` (README §3.6, METHODOLOGY §6.8) runs it as gates on this round's engines:

- the two exact chips joined by QrackBridge;
- the 5 × 5 × 5 box, counting register on the base and work register on a wall: front (a direct ring cut), right (relayed through front) or left (the ring's wrap cut);
- the dual Nighthawk R1, routed by SWAPs on each module's lattice, with the distant CNOTs through cable 242–245.

It also widens the case. The textbook counting width t = 2n is run next to the paper's t = 4, and N = 21 covers three orders: a = 8 (r = 2), a = 4 (r = 3) and a = 2 (r = 6). Each controlled multiplication by a^(2^k) mod N is synthesised as the paper's unitary on every basis state (selftest [20]).

Five layouts place the cut against the algorithm:

| Layout | Where the cut runs | CX across the cut, N 15 t 4 / N 21 a 2 |
|---|---|---|
| `inside` | no cut: one circuit | 0 / 0 |
| `split` | between the registers; each multiplication controlled through a fan-out ancilla on the work side | 4 / 20 |
| `direct` | between the registers; every gate a counting qubit controls crosses | 32 / 984 |
| `countsplit` | through the counting register (the inverse QFT crosses) | 12 / 60 |
| `worksplit` | through the work register (the multiplications cross) | 46 / 1866 |

On the bridge, every circuit runs at `max_bonds` 0–5 with 4 realisations (seeds). P(y) comes from the bridge's ket where it is at most 22 qubits wide, else from 4096 shots, whose expected TVD floor is 0.006–0.045 by case. Run times on 2 cores: 11 s (N 15) and 1203 s (N 21) on the chips, 742 s on the box, 179 s (N 15) and 324 s (N 21) on the dual.

Cells are success (or the order found where r is odd, which gives no factor), the range over the 4 realisations where it exceeds 0.02, and in brackets the TVD to the ideal distribution.

**Two chips, N = 21, a = 2 (r = 6, t = 10):** ideal success 0.322, uniform guess 0.051; the rule needs 4 bonds for the split layouts.

| Layout | 0 | 1 | 2 | 3 | 4 | 5 | Peak bonds |
|---|---|---|---|---|---|---|---|
| inside | 0.328 [0.02] | 0.322 [0.02] | 0.326 [0.02] | 0.324 [0.02] | 0.318 [0.02] | 0.325 [0.02] | 0 |
| split | 0.051 [0.94] | 0.003 [0.67] | 0.112 (0.03–0.20) [0.43] | 0.251 (0.18–0.33) [0.22] | **0.322 [0.00]** | 0.322 [0.00] | 5 |
| direct | 0.054 [0.94] | 0.091 [0.86] | 0.076 (0.03–0.17) [0.63] | 0.255 (0.16–0.34) [0.30] | **0.322 [0.00]** | 0.322 [0.00] | 5 |
| countsplit | 0.064 [0.94] | 0.060 (0.00–0.23) [0.55] | 0.147 (0.10–0.25) [0.37] | 0.230 (0.18–0.31) [0.25] | **0.318 [0.02]** | 0.328 [0.02] | 6 |
| worksplit | 0.051 [0.94] | 0.052 [0.94] | 0.112 (0.09–0.14) [0.73] | **0.319 [0.03]** | 0.322 [0.02] | 0.318 [0.02] | 5 |

**Two chips, N = 21, a = 4 (r = 3):** order found; ideal 0.665, uniform 0.299; the rule needs 3 bonds.

| Layout | 0 | 1 | 2 | 3 | 4 | 5 | Peak bonds |
|---|---|---|---|---|---|---|---|
| inside | 0.663 [0.01] | 0.661 [0.01] | 0.667 [0.02] | 0.655 [0.02] | 0.665 [0.01] | 0.656 [0.02] | 0 |
| split | 0.299 [0.96] | 0.022 [0.67] | 0.259 (0.17–0.31) [0.54] | **0.665 [0.00]** | 0.665 [0.00] | 0.665 [0.00] | 4 |
| direct | 0.262 [0.90] | 0.284 [0.92] | 0.442 (0.41–0.51) [0.57] | **0.665 [0.00]** | 0.665 [0.00] | 0.665 [0.00] | 4 |
| countsplit | 0.255 [0.97] | 0.147 (0.00–0.57) [0.59] | 0.394 (0.30–0.45) [0.43] | **0.662 [0.01]** | 0.664 [0.01] | 0.663 [0.02] | 6 |
| worksplit | 0.292 [0.96] | 0.302 [0.96] | 0.515 [0.73] | **0.670 [0.02]** | 0.662 [0.02] | 0.666 [0.01] | 4 |

**Power-of-two orders, every layout, every capacity:**

| Case | Ideal | Uniform guess | Two chips, `max_bonds` 0–5, all five layouts | Box, 0–5, split and direct, all three placements |
|---|---|---|---|---|
| N 15, a 7 (r 4), t 4: the paper's | 0.500 | 0.125 | 0.500, TVD 0.00 | 0.494–0.507, TVD ≤ 0.01 (shots) |
| N 15, a 7 (r 4), t 8 | 0.500 | 0.086 | 0.500, TVD 0.00 (inside: shots, ≤ 0.01) | 0.494–0.506, TVD ≤ 0.01 |
| N 21, a 8 (r 2), t 10 | 0.500 | 0.134 | 0.494–0.507, TVD ≤ 0.01 | — |

**The box, N = 21, split | direct at each placement:** success for a = 2 (need 4), order for a = 4 (need 3).

| Placement | a 2, `max_bonds` 3 | a 2, 4 | a 4, 2 | a 4, 3 |
|---|---|---|---|---|
| base / front (direct) | 0.242 \| 0.153 [0.22 \| 0.36] | **0.319 \| 0.317** [0.02 \| 0.02] | 0.219 \| 0.460 [0.58 \| 0.41] | **0.664 \| 0.664** [0.01 \| 0.02] |
| base / right (relayed through front) | 0.181 \| 0.273 [0.25 \| 0.33] | **0.326 \| 0.320** [0.02 \| 0.02] | 0.202 \| 0.517 [0.56 \| 0.38] | **0.664 \| 0.663** [0.02 \| 0.02] |
| base / left (the wrap cut) | 0.267 \| 0.190 [0.22 \| 0.36] | **0.329 \| 0.321** [0.03 \| 0.02] | 0.221 \| 0.426 [0.53 \| 0.56] | **0.664 \| 0.670** [0.01 \| 0.01] |

At `max_bonds` 0 every N = 21 a = 2 placement gives 0.00–0.06 against the uniform 0.051. No state vanished on the ring (0 of 576 box bridge circuits).

**The dual Nighthawk R1, N = 15, a = 7, link 242–245, 400 trajectories:** ideal 0.500; uniform 0.125 (t 4) and 0.086 (t 8). Physical CX count = local CX + 3 × SWAPs.

| t | Layout | Distant CNOTs | Physical CX | Hard cut | Ideal / exact | Constituent budget | Measured distant CZ | Classical fidelity (measured) |
|---|---|---|---|---|---|---|---|---|
| 4 | inside one module | 0 | 374 | 0.500 | 0.500 | 0.375 [0.17] | 0.375 [0.16] | 0.833 |
| 4 | split | 4 | 338 | 0.500 | 0.500 | 0.417 [0.09] | **0.414** [0.09] | 0.908 |
| 4 | direct | 32 | 327 | 0.500 | 0.500 | 0.441 [0.10] | 0.392 [0.12] | 0.882 |
| 8 | inside one module | 0 | 742 | 0.500 | 0.500 | 0.191 [0.72] | 0.191 [0.72] | 0.277 |
| 8 | split | 4 | 652 | 0.500 | 0.500 | 0.238 [0.61] | 0.222 [0.69] | 0.307 |
| 8 | direct | 32 | 641 | 0.500 | 0.500 | 0.228 [0.68] | 0.217 [0.67] | 0.334 |

N = 21, a = 2, split on the dual: 20 distant CNOTs and 24,544 physical CX (6,298 SWAPs). Exact 0.322; measured model 0.050 [0.94] over 24 trajectories, against the uniform guess's 0.051.

- **The bonds a cut needs follow the order.** With the cut between the registers, the output is exact from ⌈log₂ r⌉ + 1 bond qubits per cut. That is the final state's Schmidt rank r across the cut, plus one for the counting bit in flight. The values are 4 for r = 6 and 3 for r = 3, on the chips and at every box placement, fan-out or direct, whatever the number of crossing CXs (20 or 984). The peak the bridge holds is one more again (5, 4). A cut through the work register reaches the exact output at 3 for both orders, one bond early for r = 6. A cut through the counting register needs the split value (4, 3) but peaks at 6.
- **Power-of-two orders need no bond at all, so the paper's N = 15 is blind to the cut.** For r = 2^k the work register's record of x mod r is x's own low bits. A truncation that keeps any of its branches, as the hard cut does, is then a measurement of the work register, which Shor's algorithm tolerates at any point. Through the inverse QFT the counting register stays a product state.
  - N = 15, a = 7 (the paper's case at t = 4 and t = 8) and N = 21, a = 8 give the exact distribution at `max_bonds` 0 in every layout, on the box's relays and its wrap cut, and on the dual's hard cut. That includes the layouts where 1866 CXs cross the cut.
  - A factoring success of ½ at N = 15 therefore certifies nothing about the link between the circuits: two circuits with no quantum channel score it too. Only an order that is not a power of two (here N = 21, a = 2 or 4) makes the link carry entanglement.
- **Below capacity the success is not a measure of the run.** Under the bond limit the truncation chooses among equal Schmidt values, so the output depends on the seed. Two examples:
  - split at 3 bonds ranges 0.18–0.33 over 4 realisations;
  - one relayed box realisation found the order with 0.84 against the ideal 0.665, from a distribution 0.38 away in TVD.

  Success above the ideal, or near it, can come from a wrong distribution. The TVD to the ideal distribution, or the classical fidelity, is the measure that only the exact output passes. At `max_bonds` 0 an r = 6 run is the uniform guess (0.051–0.064, TVD 0.94).
- **The box's ring adds nothing to the rule.** The direct cut, a relay through a third face and the wrap cut all reach the exact output at the same capacity (4 / 3). Below it they differ, by realisation, as two chips do. Unlike the GHZ drift gates of §10.16, Layer 3 never zeroed the ring's state.
- **On the dual R1 the cable is cheaper than the lattice.**
  - With the benchmarked distant CZ, the paper's Layer 3 (t = 4) keeps 0.414 of the ideal 0.500 split across the modules, but 0.375 on one module alone. One module puts both registers on one line and costs more SWAPs (374 physical CX against 338).
  - The hard cut scores 0.500 again: r = 4. So on IBM's dual R1 this benchmark cannot tell the cable from no cable.
  - At t = 8 the extra QFT routing (~650 CX) drops the success to 0.19–0.24 (uniform 0.086).
  - N = 21's 24,544 physical CX at 0.355 % per local CZ leave exactly the uniform guess (0.050).

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
| a noise knob ACE's seam loss follows (E23): ACE's own thresholds (`_epsilon`, `_rot_epsilon`) | not tried: the gate-level amplifiers (CX, ZZ-echo and global folding) do not scale it; the boundary repetition code is not a noise level but a correction (§10.7) |
| Layer 2 at the paper's K = 32 (E22); Layer 1b and the OTOC on ACE | not run here: K = 4; 1b and OTOC only in the numpy reference |
| ACE's lineage authority after a seam-to-seam CX (E24, E25) | the crossbar pair becomes the sole authority and a partner stays there when the other qubit couples into a patch; `nighthawk_ghz.py` schedules around it (`--code-schedule`) or moves the partner along (`--code-authority colocated`). To report upstream: the co-located step in `_update_lineage`, with `nighthawk61r2` [48, 40, 56] (0.463 → 1.003) |
| GHZ across two ACE simulators (E24) | no coherence by construction (replicas and classical shadows only): a chip-wide GHZ on a register of two or more patches is not correctable on ACE |
| GHZ chip-wide on `nighthawk61r2` (E21) | stabilizer patches leave the tableau in ACE's sampler on PyQrack 2.29.5 / Qrack 10.13.2: test on the EPYC build |
| Hamming ladder n = 45–57 and `nighthawk61r2` (E18) | EPYC: README §7 recipe; 57 needs 64 GiB patches, 61 RAM + swap (≈ 385 GiB) |
| VQE ladder n = 28–61 (E19) | EPYC: README §7 recipe; numpy circuit reference to 28–30 there (`--vqe-exact-max`, 4–16 GiB), Lanczos E₀ to 26–28 |
| `QrackSimulator.measure_shots` beyond 31 qubits (PyQrack 2.28.1) | returns bits 0–31 only (a qubit above 31 reads 0; `QrackAceBackend` returns all); reproducer: `QrackSimulator(40)`, `x(35)`, `measure_shots(range(40), 3)` gives 0. `shor9` samples per unit; the exact engine now refuses to draw shots above 32 qubits instead of returning truncated ones. To report upstream |
| ACE sampler, unit inside one patch (E20) | odd block parities (whole-block flips) ≈ 5–7 % with all-exact CXs, in PyQrack 2.28.1 and 2.29.4, boundary code on or off; to report upstream with the `shor9 --register nighthawk61r2` units 0, 1 |
| E20 with PyQrack 2.29's boundary code | `shor9 --ace-boundary-rep` under PyQrack ≥ 2.29 on the EPYC, Shor-9 to Shor-36 and the registers, `--repeats 10` (each backend lands at 1.000 or 0.5) |
| `--ace-boundary-rep` on a register with a stored placement | until 22 October it made `run` and `shor9 --register` search a new placement instead of the register's stored one (the boundary code changes no sites or patches); fixed, no result in this file used it |
| `shor9 --register --jobs` on a saved register | until 22 October a spawned worker did not load `$NIGHTHAWK_REGISTERS`, so `fav*` and other saved registers stopped with `unknown --register`; fixed |
| `run` under PyQrack ≥ 2.29 | earlier versions of the script passed only non-default ACE switches, so under 2.29 a `run` without `--ace-boundary-rep` would have had the boundary code on; every ladder point so far is PyQrack 2.28.1 (off) |
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
