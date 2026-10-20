# ER=EPR on the cloverleaf: `rcs/nighthawk/nighthawk_erepr.py`

**The 2022 Sycamore wormhole, re-derived from its stated Hamiltonian, compiled onto Nighthawk lattice registers, and run on `nighthawk_qrack.py`'s engines and `nighthawk_cloverfield.py`'s chips and loom: the two sides of the wormhole are two chips, the thermofield double that joins them is the EPR, and the loom is the only way it crosses**

| | |
|---|---|
| Script | `rcs/nighthawk/nighthawk_erepr.py` |
| Builds on | `nighthawk_qrack.py`: `ExactEngine`, `AceEngine`, `two_patch_config`, `layout_row`, `ace_tiling` (v3), `placement_stats`, `u_angles`, `gate`, `find_repo` · `nighthawk_cloverfield.py`: `parse_chip`, `chip_layout`, `Field`, `ChipProc` workers, `Loom` (`ghost`, `mean`, `sample`), `worker_env_for` |
| Data | `erepr/`: `reference.json` (§4), `exact.jsonl`, `ace.jsonl`, `loom.jsonl`, `loom_links.jsonl`, `knit.jsonl` (§5–6), `tiles.json` (the tiling cache), `run_all.sh` |
| Target | Jafferis et al., *Traversable wormhole dynamics on a quantum processor*, Nature 612, 51 (2022) [[1]](#ref-1) |
| Exchange | Kobrin, Schuster, Yao, Comment, arXiv:2302.07897 [[2]](#ref-2) · Jafferis et al., Reply, arXiv:2303.15423 [[3]](#ref-3) |
| Circuit source | S. Davis, Caltech thesis ch. 14: Jordan–Wigner table, swaps, 9-qubit layout [[4]](#ref-4) |
| Lattice | the release's 12 × 10 Nighthawk lattice and `rect_matchings` ([`BlueQubitDev/rcs-nighthawk`](https://github.com/BlueQubitDev/rcs-nighthawk)), as for every nighthawk tool |
| Acceleration | CPU (every number below); OpenCL optional. **No CUDA.** |
| Versions | Python 3.13, numpy 2.5.3, PyQrack 2.28.2, Qiskit 2.6.0 (compiler only) |

---

## Overview

The experiment's claim is a traversable wormhole: a message inserted into one side of a thermofield double (TFD), scrambled by the learned Hamiltonian, re-emerges on the other side when a negative-energy coupling (μ < 0) joins them, and not when μ > 0. On the Nighthawk tools the two sides become two halves of a register: two ACE patches on the cloverleaf (§5), or two chips on the cloverfield (§6). The TFD's L|R entanglement (3.21 bits) and the coupling's 7 ZZ rotations are all that crosses: 18 of the circuit's ≈ 1040 CZs.

| Part | Does | Nighthawk machinery |
|---|---|---|
| reference (`derive`) | the published claims re-derived in dense numpy, no PyQrack | — |
| circuit | Majorana algebra → Pauli strings → `u` + CZ, routed onto lattice couplers and links | `rect` chips and `rect_matchings` via `cf.parse_chip` / `chip_layout`, `cf.Field` links |
| exact | the routed field register | `nq.ExactEngine` |
| cloverleaf | one register, two ACE patches, four placements | `nq.AceEngine`, `two_patch_config`, `layout_row`, `ace_tiling` v3, `placement_stats` |
| cloverfield | two chips in isolated workers, every CZ woven by the loom | `cf.ChipProc`, `cf.Loom` (`ghost`, `mean`, `sample`), ghost plans, `--links K` |
| knit | the cut resolved exactly, Schmidt components kept or dropped | `nq.ExactEngine` per chip |

### Key results

| Finding | Numbers | § |
|---|---|---|
| The learned Hamiltonian is the only 5-term commuting quartic model on 7 Majoranas | 630 commuting sets, 1 relabeling class | 4.1 |
| The teleportation peak sits at the training time, for the trained pair only | ψ¹ψ²: I_PT 0.241 bits at t = 2.80 (μ = −12) vs 0.100 (+12); untrained first peaks 2.5–4.7 | 4.4 |
| The sign asymmetry fails for 8 of 21 Majorana pairs; SYK N = 10 keeps it on all 9 tested | peak I(+12) ≥ 0.9 × peak I(−12); SYK ratios 0.56–0.68 | 4.4 |
| "Perfect size winding" holds for ψ¹ only; random coefficients reach Model 1's slope fit 27.8 % of the time | ψ² r₅ = 0.78; ψ⁴, ψ⁷ χ = 0.07, 0.18 | 4.3 |
| As a channel the ideal wormhole is 0.011 above the classical bound | best F_avg 0.6774 vs 2/3 | 4.4 |
| Routed onto Nighthawk fields, the circuit is exact on `nq.ExactEngine` | I_PT within 2.6 × 10⁻⁷ of the reference at 124 points, both fields | 3.4 |
| The ghost loom carries the wormhole exactly when every side qubit has a ghost | 4 links: 0.2410 at t = 2.8, drift 0, 4 round trips | 6.1 |
| One link fewer and the peak is gone; the mean-field loom closes it at any link count | 3 / 2 / 1 links: 0.023 / 0.041 / 0.041; mean field ≤ 0.0065 | 6.1 |
| The peak scales with the TFD entanglement kept across the cut; the t = 0 signal does not | knit: 0.006 → 0.241 bits as 0 → 3.21 bits are kept; 0.128 at t = 0 throughout | 6.2 |
| On two ACE patches no placement keeps the peak, and the coupler-weighted tiling loses everything | `tiling` 960 exact CZs, I_PT 0; `lr` ≤ 0.002; `seam-r` 0.013 at t = 2.8 | 5.2 |

---

## 1. Installation and quick start

```bash
cd thereminq-examples/rcs/nighthawk
git clone https://github.com/BlueQubitDev/rcs-nighthawk          # lattice and matchings, as for nighthawk_qrack
pip install numpy pyqrack qiskit
python3 nighthawk_erepr.py selftest                              # ≈ 10 s
python3 nighthawk_erepr.py plan                                  # fields, links, routing, ACE placements
./erepr/run_all.sh                                               # every number of this file; resumable
python3 nighthawk_erepr.py table erepr/*.jsonl                   # every simulation, side by side
python3 nighthawk_erepr.py aceprobe                              # Bell pairs across the two-patch register
```

| Subcommand | Does |
|---|---|
| `derive` | the exact reference in numpy (no PyQrack): structure, operator growth, correlators, size winding, random-coefficient control, teleportation (21 pairs, both signs, V normalisation, the Reply's H₁, SYK N = 10) → JSON |
| `plan` | fields, links, roles, routed gate counts; ACE placements scored with `placement_stats` |
| `run --backend exact` | `nq.ExactEngine` on a routed field register |
| `run --backend ace` | `nq.AceEngine` on the 16-site register; `--placement grid tiling lr seam-r`; `--reps` |
| `run --backend loom` | cloverfield: one `ChipProc` worker per chip, the `Loom` weaving every CZ; `--field wide narrow`, `--links K`, `--loom ghost mean sample` |
| `run --backend knit` | the cut resolved exactly (TFD Schmidt sum × coupling branches), `--keep k` Schmidt components |
| `table`, `aceprobe`, `selftest` | tables of JSONL runs; Bell pairs per site class; checks of §3.4 |

`nighthawk_erepr.py` carries `QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"` and points `NIGHTHAWK_TILE_CACHE` at `erepr/tiles.json` unless it is set.

---

## 2. Inputs and conventions

### 2.1 Inputs (as published)

| Input | Value | Source |
|---|---|---|
| Learned Hamiltonian, per side | −0.36 ψ¹ψ²ψ⁴ψ⁵ + 0.19 ψ¹ψ³ψ⁴ψ⁷ − 0.71 ψ¹ψ³ψ⁵ψ⁶ + 0.22 ψ²ψ³ψ⁴ψ⁶ + 0.49 ψ²ψ³ψ⁵ψ⁷ | [1], [2] Eq. 1, [3] Eq. 3, [4] Eq. 14.7 |
| Majorana normalisation | {ψⁱ, ψʲ} = δᵢⱼ (ψ² = ½) | [2] |
| Coupling | V = (1/qN) Σⱼ i ψ_Lʲ ψ_Rʲ, q = 4, N = 10, applied as e^{iμV} | [2], [4] |
| μ, β, t | ±12, 4, 2.8 | [2], [3] |
| Message pair | (ψ¹, ψ²) of L in, of R out | [3], [4] |
| Reply's perturbation H₁ | 0.3 ψ¹ψ²ψ³ψ⁵ | [3] Eq. 4 |
| SYK reference | q = 4, N = 10, couplings N(0, 3! J²/N³), J = 1.125 (winding), 1.25 (teleportation), 3 seeds | [2] |
| Device encoding | Jordan–Wigner table, 7 qubits + P, Q; T read from qubit 1 | [4] Eq. 14.3 |

### 2.2 Conventions

| Item | Definition |
|---|---|
| Split encoding | L: 8 Majoranas (ψ⁸ pads) on L0–L3; R on R0–R3 with the parity of L in its strings; P reference, Q message |
| Pairing state \|I⟩ | ground state of Σⱼ i ψ_Lʲ ψ_Rʲ (gap 1.000) |
| TFD | e^{−βH_L/2}\|I⟩, normalised; H_L\|I⟩ = H_R\|I⟩ to 3.4 × 10⁻¹⁷; L\|R entanglement 3.21 bits, Schmidt rank 16 |
| Protocol | \|TFD⟩\|Φ⁺⟩_PQ → e^{+iH_L t₀} → swap(Q, ψ_L¹ψ_L²) → e^{−iH_L t₀} → e^{iμV} → e^{−iH_R t₁}; symmetric t₀ = t₁ = t unless stated |
| TFD as a circuit | amplitude load Σ sₙ\|n⟩ on L, four copy CNOTs L_k → R_k, then W_L ⊗ W_R (the Schmidt vectors, compiled by Qiskit) |
| Coupling as a circuit | per Majorana pair j one ZZ rotation between the ports (L_k, R_k), k = (j − 1)/2 rounded down, after a ladder on each side |
| Readout | ρ_PT from the 16 correlators ⟨σ_P τ⟩, τ = (√2 ψ_R¹, √2 ψ_R², −2i ψ_R¹ψ_R²): R0 with the parity of L0–L3 |
| I_PT | S_P + S_T − S_PT in bits (2 = perfect) |
| F_avg | best average fidelity of the P→T channel over a decoding rotation on T; classical bound 2/3 |
| Trotterisation | one step is exact: the five terms commute |

---

## 3. Registers, routing and checks

### 3.1 Registers

Nighthawk lattice coordinates; chips are `nighthawk_cloverfield` `rect` chips, arranged 2 × 1 (chip A above chip B), so links join each column's lowest A site to its highest B site.

| Register | Sites | Roles | Links |
|---|---|---|---|
| field `wide` | chip A `rect:0-1,0-3` (8), chip B `rect:0-0,0-3` (4) | row 0: Q P · ·, row 1: L0 L1 L2 L3; chip B: R0 R1 R2 R3 | 4: L_k–R_k |
| field `narrow` | chip A as above, chip B `rect:0-1,0-1` (4) | chip B: R0 R1 / R2 R3 | 2: L0–R0, L1–R1 |
| field `wide --links K` | as `wide` | as `wide` | K of the 4, evenly spaced (cloverfield's rule) |
| single | `rect:0-3,0-3` (16) | the wide field as one register, row 3 idle | — |

### 3.2 Routing

Every two-qubit gate is routed onto lattice couplers or links: a gate between non-neighbours on one chip moves its first qubit along a shortest path with SWAPs and back; a gate between chips moves both ends to the nearest link, crosses there, and moves back; CNOT = H·CZ·H; consecutive single-qubit gates are fused. The layout is the role map after every gate.

| Register | CZ inside chips | CZ on links | SWAPs | single-qubit gates (fused) |
|---|---|---|---|---|
| field `wide` | 1026 | 18 | 242 | 1750 |
| field `narrow` | 1008 | 18 | 236 | 1738 |
| single (`rect:0-3,0-3`) | 1044 | — | 242 | — |

Counts at t = 2.8. The 18 link CZs are the protocol's whole L\|R traffic: 4 TFD copy CNOTs and 7 coupling ZZ rotations of 2 CZs each. The SWAPs come from the Qiskit-compiled TFD unitaries W_L, W_R and the Hamiltonian ladders on a line of four qubits.

### 3.3 ACE register and placements

`nq.AceEngine` on `rect:0-3,0-3` with `lrc`/`lrr` = `auto`: `two_patch_config(16)` gives lrc 2, lrr 3, internal simulator widths 11, 11 and 5 (`ace_sim_widths`), PyQrack's default ACE options (error detection and crossbars on), CPU.

| Sites | ACE indices |
|---|---|
| bulk A | 0, 1, 7, 8, 9, 15 |
| bulk B | 3, 4, 5, 11, 12, 13 |
| boundary (replicated) | 2, 6, 10, 14 |

The placement replaces `eng.idx`, as `nighthawk_cloverfield` does for its ghosts. Each is scored with `nq.placement_stats` on the routed circuit's 1044 CZs (a multiset: a coupler counts once per CZ):

| Placement | How | CZs exact / replica / cross | Seam sites used | Simulators with bulk qubits |
|---|---|---|---|---|
| `grid` | the lattice as ACE lays it out (`nq.place` without tiling) | 603 / 425 / 16 | 4 | 2 |
| `tiling` | `nq.ace_tiling` v3 on this circuit's CZ multiset (cached in `erepr/tiles.json`; ≈ 9 min on 2 cores) | 960 / 84 / 0 | 4 | 2 |
| `lr` | L, P, Q in bulk A, R in bulk B: the wormhole across the seam | 1026 / 0 / 18 | 4 | 2 |
| `seam-r` | L, P, Q in bulk A, R on the four boundary sites | 545 / 499 / 0 | 4 | 2 |

Idle sites fill the remaining places, so every placement uses all four seam sites.

### 3.4 Checks (`selftest`)

| Check | Result |
|---|---|
| Reference: H_L\|I⟩ = H_R\|I⟩; readout algebra | 3.4 × 10⁻¹⁷; τ_k² = 1 |
| Device encoding (Davis table) and split encoding: Majorana algebra | 14 and 16 Majoranas, max ‖{ψᵃ, ψᵇ}‖ = 0 |
| Both logical circuits on `nq.ExactEngine` | I_PT 0.24107 at t = 2.8, as the reference |
| ZXZ angles for the loom's `g` op (`nq.gate`) | matrix reproduced up to phase |
| Fields `wide`, `narrow` routed, on `nq.ExactEngine` | I_PT 0.24107 |
| Single register routed: state vector vs chain-rule estimator (prob, force_m on clones) | 0.24107 / 0.24107 |
| Wide field, ghost loom (every side qubit a link anchor: chip A holds a replica of the whole state) | 0.24103, ghost drift 4 × 10⁻⁸, product view 0 |
| Wide field, mean-field loom | 0.00109 |
| Knit (2048 branches) | 0.24107 |

Across the full grids: exact `wide` and `narrow` match the reference to 2.6 × 10⁻⁷ at all 124 points (31 times × 2 signs × 2 fields); the knit with 16 components to 7 × 10⁻⁸, the 4-link ghost loom to 3.7 × 10⁻⁵.

---

## 4. Reference (`derive`)

### 4.1 Structure

| Quantity | Value |
|---|---|
| Term pairs commuting (support overlap even) | 10 of 10; max ‖[A, B]‖ = 0 |
| 5-term commuting sets of quartic monomials on 7 Majoranas | 630, all touching all 7, in 1 relabeling class (Model 1 in it): the Reply's uniqueness statement holds |
| Spectrum (one side) | 8 levels, each doubly degenerate through ψ⁸; 24 distinct transition frequencies |

### 4.2 Operator growth (t = 2.8) and correlators (β = 4)

| Operator | Majorana strings | Sizes |
|---|---|---|
| ψ¹ … ψ⁵ (t) | 8 | 1, 3, 5 |
| ψ⁶, ψ⁷ (t) | 4 | 1, 3, 5 |
| SYK N = 10, ψ¹(t) | 511 of 512 odd strings | up to 9 |

G_i(t) = 2 Re tr ρ ψⁱ(t)ψⁱ, G_i(0) = 1, 0 ≤ t ≤ 30:

| i | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| min G_i | −0.580 | −0.602 | −0.873 | −0.680 | −0.576 | −0.791 | −0.892 |
| max G_i after t = 5 | 0.582 | 0.720 | 0.609 | 0.574 | 0.719 | 0.988 | 0.539 |
| at t | 10.2 | 23.3 | 15.5 | 30.0 | 16.2 | 26.9 | 18.3 |

Every fermion revives (ψ⁶ to 0.988); the average including ψ⁸ ≡ 1 stays between −0.091 and 0.330 after t = 5. The Comment's averaging artefact, reproduced.

### 4.3 Size winding (β = 4, t = 2.8)

ρ^{1/2}ψⁱ(t) = Σ_P c_P Γ_P over the 256 Hermitian strings; q(l) = Σ_{|P|=l} c_P², r_l = |q(l)| / Σ|c_P|², r = Σ_l |q(l)|, r̄ = (r − W)/(1 − W), χ the Comment's slope-linearity metric; only odd sizes carry weight.

| Model | i | r₁ | r₃ | r₅ | weight l = 1 / 3 / 5 | r | r̄ | χ |
|---|---|---|---|---|---|---|---|---|
| Model 1 | 1 | 1.000 | 0.962 | 0.956 | 0.18 / 0.59 / 0.23 | 0.968 | 0.901 | 0.997 |
| | 2 | 1.000 | 0.987 | 0.783 | 0.28 / 0.55 / 0.17 | 0.955 | 0.819 | 0.993 |
| | 3 | 1.000 | 0.789 | 0.992 | 0.12 / 0.43 / 0.45 | 0.906 | 0.751 | 0.980 |
| | 4 | 1.000 | 0.996 | 0.631 | 0.41 / 0.53 / 0.05 | 0.979 | 0.875 | **0.073** |
| | 5 | 1.000 | 0.808 | 0.997 | 0.12 / 0.51 / 0.37 | 0.901 | 0.747 | 0.994 |
| | 6 | 1.000 | 0.993 | 0.528 | 0.24 / 0.70 / 0.06 | 0.966 | 0.873 | 0.955 |
| | 7 | 1.000 | 0.999 | 0.632 | 0.41 / 0.52 / 0.07 | 0.973 | 0.831 | **0.178** |
| SYK N = 10, 3 seeds | 1, 2 | — | — | — | — | 0.900–0.934 | 0.515–0.593 | 0.929–0.973 |
| Model 1 + H₁ | 1, 2 | 0.858, 0.870 | 0.950, 0.973 | 0.954, 0.788 | — | 0.935, 0.914 | 0.814, 0.682 | 0.986, 0.997 |

**Random-coefficient control** (the Comment's Appendix E: Model 1's terms, coefficients N(0, 0.438²), 1000 instances, seed 7)

| Measure | Fraction of instances reaching Model 1 |
|---|---|
| χ, two best fermions ≥ min(χ₁, χ₂) = 0.993 | 27.8 % (Reply, by χ: 25 %) |
| χ, all seven fermions ≥ 0.993 | 0.0 % (Model 1 itself fails on ψ⁴, ψ⁷; Reply: 2 %, threshold not stated) |
| r̄ of ψ¹ ≥ 0.901 and of ψ² ≥ 0.819 (Model 1's) | 30.7 % (median r̄ over all fermions 0.888) |

### 4.4 Teleportation (V normalised with N = 10)

Checks: I_PT = 0 at μ = 0 (no signalling); readout τ_k² = 1 to 4.4 × 10⁻¹⁶.

| Trained pair (ψ¹, ψ²) | μ = −12 | μ = +12 |
|---|---|---|
| symmetric, I_PT at t = 2.8 | **0.2411** | 0.1002 |
| symmetric, peak 0 ≤ t ≤ 10 | 0.2411 at t = 2.80 | 0.1536 at t = 7.25 |
| fixed t₀ = 2.8, peak over t₁ | 0.2504 at t₁ = 3.30 | 0.1106 at t₁ = 2.10 |
| F_avg at t = 2.8; best | 0.6773; 0.6774 at t = 2.85 | 0.5789; 0.6389 at t = 7.70 |

| All 21 pairs, 0 ≤ t ≤ 10 | Model 1 | SYK N = 10, pairs (1,2), (3,7), (5,9), 3 seeds |
|---|---|---|
| peak I(−12) | 0.128–0.407 | 0.272–0.330 |
| time of first peak | 2.5–4.7 (2.80 for the trained pair only) | 2.8–3.3 |
| global peak after t = 6 | 8 of 20 untrained pairs | 0 of 9 |
| peak I(+12) ≥ 0.9 × peak I(−12) | 8 of 21 pairs: (1,5), (2,6), (2,7), (3,5), (3,6), (4,7), (5,6), (6,7) | 0 of 9 (ratios 0.56–0.68) |

| Sensitivity (trained pair) | peak I(−12) at t | peak I(+12) |
|---|---|---|
| V with N = 10 / 8 / 7 | 0.241 at 2.80 / 0.340 at 2.70 / 0.404 at 2.65 | 0.154 / 0.208 / 0.263 |
| Model 1 + H₁ | 0.226 at 2.70 (0.225 at t = 2.8) | 0.139 (0.097 at t = 2.8) |

The ideal trained wormhole beats the classical teleportation bound by 0.011. The experiment's evidence is the μ-asymmetry of I_PT; F_avg states what that asymmetry amounts to as a channel.

---

## 5. Cloverleaf: one register, two ACE patches

One 16-site register, `nq.AceEngine` with nighthawk's two-patch rule (§3.3); the routed circuit (1044 CZs, all on lattice couplers) is applied through the engine's `g1` / `cz`, read by the chain-rule estimator on clones (prob, force_m). ACE draws random outcomes: 8 repetitions per point, mean ± sd; t on a 0.4 grid, 0 ≤ t ≤ 6.

### 5.1 Which site pairs hold a Bell pair (`aceprobe`, 16 runs)

H, CNOT (H·CZ·H), then ZZ and XX; exact = +1, +1.

| Pair | Sites | ⟨ZZ⟩ | ⟨XX⟩ |
|---|---|---|---|
| bulk A – bulk A | 0, 1 | +1.000 ± 0.000 | +1.000 ± 0.000 |
| bulk B – bulk B | 3, 4 | +1.000 ± 0.000 | +1.000 ± 0.000 |
| bulk A – bulk B | 0, 3 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| bulk A (control) – boundary | 0, 2 | +1.000 ± 0.000 | +0.287 ± 0.274 |
| boundary (control) – bulk B | 6, 4 | +1.000 ± 0.000 | +1.000 ± 0.000 |
| boundary – boundary | 2, 6 | +0.750 ± 0.661 | +0.750 ± 0.661 |

A CZ between bulk sites of different patches leaves no correlation; through the boundary it is exact in one direction, partial or sign-flipped (2 of 16 runs) in the others.

### 5.2 The wormhole on the register

| Placement | Roles on sites | CZs exact / replica / cross | I_PT at t = 0 (μ = −12) | I(−12) at t = 2.8 | I(+12) at t = 2.8 | max I_PT over t, μ (mean) | best F_avg(−12) |
|---|---|---|---|---|---|---|---|
| exact (reference) | — | — | 0.1279 | **0.2411** | 0.1002 | 0.2411 | 0.6773 |
| `grid` | Q P in bulk A; L0 L1 bulk B, L2 boundary, L3 bulk A; R0 R1 bulk A, R2 boundary, R3 bulk B | 603 / 425 / 16 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| `tiling` | P in bulk B, Q on the boundary, L0 L1 L2 in bulk A, L3 on the boundary, R0 R1 R2 in bulk A, R3 on the boundary | 960 / 84 / 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| `lr` | L, P, Q in bulk A, R in bulk B | 1026 / 0 / 18 | 0.0000 | 0.0003 ± 0.0006 | 0.0007 ± 0.0009 | 0.0009 | 0.5045 |
| `seam-r` | L, P, Q in bulk A, R on the boundary | 545 / 499 / 0 | 0.0974 ± 0.0341 | 0.0127 ± 0.0066 | 0.0147 ± 0.0082 | 0.0977 | 0.5978 |

| Observation | Numbers |
|---|---|
| Nighthawk's coupler-weighted tiling minimises approximated CZs, and loses the whole signal | 960 exact CZs, 0 cross, I_PT 0 at every t: the P–Q Bell pair sits across the seam (P bulk B, Q boundary), the one pair §5.1 shows ACE cannot hold. Coupler counts weigh every CZ alike; the observable depends on a few |
| The lattice placement fails for the same reason | `grid` has Q P in bulk A and the message's first Majorana mode (L0) in bulk B: the message swap is a cross CZ |
| The wormhole across the seam is closed | `lr`: 18 cross CZs, exactly the L\|R traffic; I_PT ≤ 0.0022 in every run |
| R on the boundary keeps the coupling echo, not the peak | `seam-r`: 0.097 at t = 0 (exact 0.128), 0.013 at t = 2.8 (exact 0.241), sign asymmetry gone |
| Reconstructed ρ_PT stays physical | smallest eigenvalue ≥ 0.08 in every ACE run |

---

## 6. Cloverfield: two chips and the loom

### 6.1 The loom

The routed circuit runs on the field through `nighthawk_cloverfield` unchanged: one `ChipProc` worker per chip (isolated processes, exact chips), single-qubit gates as `g` ops to every copy of a qubit (`Loom._emit1`), every CZ through `Loom._layer` (exact where both qubits are held, the loom's mode where one is absent). Readout:

| View | Reads | Applies to |
|---|---|---|
| product | each chip's part of every string, multiplied: cloverfield's own estimator (correlation between the chips' outcomes dropped) | all modes |
| ghost (chip A) | P, the L string and the ghost of R0, all on chip A: parity collected onto P's copy with CNOTs, read as a marginal, undone | `ghost`, when R0 is a link anchor |

The I_PT of a `ghost` run is the ghost view, of `mean` and `sample` the product view. t on a 0.4 grid, 0 ≤ t ≤ 6. Fewer links mean longer routes: at t = 2.8 the circuit has 1026 / 1062 / 1098 / 1206 CZs inside chips and 242 / 254 / 266 / 302 SWAPs for 4 / 3 / 2 / 1 links; the 18 link CZs are the same in every case.

| Field, loom | Links | I_PT at t = 0 (μ = −12) | I(−12) at t = 2.8 | I(+12) at t = 2.8 | max I_PT over t, μ | best F_avg(−12) | ghost drift | round trips per run |
|---|---|---|---|---|---|---|---|---|
| exact (reference) | — | 0.1279 | **0.2411** | 0.1002 | 0.2411 | 0.6773 | — | — |
| `wide`, ghost | 4 | 0.1279 | **0.2410** | 0.1002 | 0.2410 | 0.6773 | 0.0000 | 4 |
| `wide --links 3`, ghost | 3 | 0.0848 | 0.0230 | 0.0248 | 0.0928 | 0.5937 | 0.0595 | 985 |
| `wide --links 2`, ghost | 2 | 0.0917 | 0.0407 | 0.0313 | 0.0917 | 0.5918 | 0.0534 | 755 |
| `wide --links 1`, ghost | 1 | 0.0792 | 0.0408 | 0.0380 | 0.0792 | 0.5678 | 0.0709 | 702 |
| `narrow`, ghost | 2 | 0.0264 | 0.0164 | 0.0235 | 0.1097 | 0.5559 | 0.0630 | 317 |
| any field, mean | 1–4 | 0.0027 | 0.0011 | 0.0001 | 0.0065 | 0.5138 | — | 19 |
| `wide`, `narrow`, sample | 4, 2 | 0.0000 | 0.0000 | 0.0009 | 0.0055 | 0.5146 | — | 19 |

| Observation | Numbers |
|---|---|
| With every side qubit a link anchor, the ghosts replicate the whole far side on chip A and the wormhole crosses exactly | `wide`, 4 links: 0.2410 bits at t = 2.8 (reference 0.2411), drift 0, 4 round trips |
| Remove one link and the t = 2.8 peak is gone | 3, 2, 1 links: 0.023, 0.041, 0.041 bits at t = 2.8; the sign asymmetry is lost or inverted |
| The `narrow` chip B (2 links, R2 and R3 two SWAPs from a link) loses the peak as well | 0.0164 vs 0.0235 at t = 2.8 |
| What the ghosts still carry is the early, coupling-driven signal | max I_PT 0.079–0.110, every one at t ≤ 0.4 |
| The product view of a ghost run is zero | the chips' own states carry no P–T correlation; it lives only in the replica |
| Mean field and sampled kicks close the wormhole on every field and link count | max I_PT ≤ 0.0065 bits; the mean-field result is independent of the geometry (chip A sees the same phases) |

### 6.2 The exact cut (`knit`)

The cut resolved exactly, as a benchmark for what a loom would have to carry: chip A = L, P, Q, chip B = R, each one `nq.ExactEngine` with a local branch register (4 Schmidt-index qubits plus one per coupling term, in \|+⟩; Qrack does not keep global phases across runs, so the branches of a chip share one run). Branch weights sₙ Πⱼ (cos θⱼ | −i sin θⱼ) are applied afterwards. `--keep k` keeps the k largest Schmidt components of the TFD.

| Keep | TFD weight kept | L\|R entanglement kept (bits) | Branches | I_PT at t = 0 | I(−12) at t = 2.8 | I(+12) at t = 2.8 | peak I(−12), at t | best F_avg(−12) |
|---|---|---|---|---|---|---|---|---|
| 16 (exact) | 1.000 | 3.21 | 2048 | 0.1279 | **0.2411** | 0.1002 | 0.2411, 2.8 | 0.6773 |
| 8 | 0.852 | 2.56 | 1024 | 0.1279 | 0.2059 | 0.0558 | 0.2184, 3.2 | 0.6699 |
| 4 | 0.658 | 1.73 | 512 | 0.1279 | 0.0922 | 0.0060 | 0.1279, 0.0 | 0.6131 |
| 2 | 0.524 | 1.00 | 256 | 0.1279 | 0.0503 | 0.0116 | 0.1279, 0.0 | 0.6131 |
| 1 (product TFD) | 0.262 | 0 | 128 | 0.1279 | 0.0059 | 0.0059 | 0.1279, 0.0 | 0.6131 |

| Observation | Numbers |
|---|---|
| The t = 0 signal is the coupling gate acting on the message and readout modes, not the wormhole | 0.1279 bits at every truncation, product TFD included |
| The t = 2.8 peak scales with the entanglement kept across the cut | 0.006 → 0.050 → 0.092 → 0.206 → 0.241 bits for 0 → 1.00 → 1.73 → 2.56 → 3.21 bits |
| The μ asymmetry needs most of the Schmidt spectrum | I(+12)/I(−12) at t = 2.8: 0.42 (16), 0.27 (8), 0.07 (4), 0.23 (2), 1.0 (1) |
| Cost of an exact cut | Schmidt rank × 2^links = 16 × 128 branches, chips of 17 and 15 qubits, against 12 for the field register; the ghost loom pays instead with replicas, and is exact only when every side qubit has one |
| keep = 1 | the top Schmidt value is doubly degenerate (ψ⁸); keep 1 drops one of the pair |

---

## 7. Claims ledger

| # | Claim | Source | Verdict | Evidence |
|---|---|---|---|---|
| 1 | The learned Hamiltonian is fully commuting | [2] | confirmed | §4.1 |
| 2 | It is the unique 7-fermion, 5-term commuting model up to relabeling | [3] | confirmed | §4.1 |
| 3 | A teleported fermion spreads onto 8 strings, the largest of size 5 | [3] | confirmed | §4.2 |
| 4 | Individual correlators revive; agreement with SYK is an averaging artefact | [2] | confirmed | §4.2 |
| 5 | Model 1 shows perfect size winding (r_l ≳ 0.95 for all l) | [1], [2] | ψ¹ only; ψ² r₅ = 0.78 | §4.3 |
| 6 | N = 10 SYK has damped winding, r_l ~ 0.75 | [2] | ordering holds in r̄ (0.52–0.59 vs 0.82–0.90); level not reproduced (r 0.90–0.93) | §4.3 |
| 7 | ψ⁴, ψ⁷ wind poorly at t = 2.8 | [3] | confirmed, χ 0.07, 0.18 | §4.3 |
| 8 | Random-coefficient commuting models wind about as well | [2]; 25 % / 2 % in [3] | confirmed: 27.8 % by χ, 30.7 % by r̄ | §4.3 |
| 9 | I_PT peaks near t = 2.8 for μ < 0 and is suppressed for μ > 0 | [1] | confirmed for ψ¹ψ²: 0.241 vs 0.100 bits | §4.4 |
| 10 | Untrained pairs: oscillations, extra peaks at longer times | [2] | confirmed: 8 of 20 peak after t = 6; 8 of 21 lose the sign asymmetry | §4.4 |
| 11 | Teleportation persists under the non-commuting H₁ | [3] | confirmed at t ≈ 2.8: 0.225 vs 0.097 | §4.4 |
| 12 | One Trotter step implements the dynamics | [4] | exact: the terms commute | §2.2 |
| 13 | The 9-qubit encoding represents 7 + 7 Majoranas | [4] | confirmed; same I_PT as the split encoding | §3.4 |
| 14 | The signal is the wormhole, carried by the TFD's entanglement | [1] | the t = 2.8 peak is (knit, ghost loom); the t = 0 transfer of 0.128 bits is the coupling gate alone | §6 |

---

## 8. Limitations

| Item | Effect |
|---|---|
| β, μ, the V normalisation (N = 10) and t come from the Comment and the thesis; the Nature paper's hardware β is not stated | absolute I_PT scales with the normalisation (0.24–0.40 bits for N = 10…7); every comparison is at one stated setting |
| The TFD is prepared exactly (Schmidt-loaded), not by the experiment's VQE | hardware preparation error is outside this study |
| Routing by SWAP chains is not optimised for gate count | ≈ 1000 CZs per run; the loom and ACE see every one of them |
| Loom chips are exact simulators | the loom's error alone, as in cloverfield selftest 8; ACE chips (`--chip-ace`) are not used here |
| The ghost readout reads P, the L string and the ghost of R0 on chip A | it measures what the loom's replica knows; the product readout is cloverfield's own estimator |
| ACE draws random outcomes | 8 repetitions per point, mean ± sd |
| SYK normalisation | {ψ, ψ} = δ with variance 3! J²/N³ does not reproduce the Comment's r_l ~ 0.75 level; the Comment reports it could not reproduce [1]'s SYK dynamics exactly either |
| χ thresholds | the Reply's "as good as Model 1" threshold is not stated; §4.3 uses min(χ₁, χ₂) of Model 1 |

---

## 9. References

<a id="ref-1"></a>[1] D. Jafferis, A. Zlokapa, J. D. Lykken, D. K. Kolchmeyer, S. I. Davis, N. Lauk, H. Neven, M. Spiropulu, *Traversable wormhole dynamics on a quantum processor*, Nature 612, 51–55 (2022). [doi:10.1038/s41586-022-05424-3](https://www.nature.com/articles/s41586-022-05424-3)

<a id="ref-2"></a>[2] B. Kobrin, T. Schuster, N. Y. Yao, *Comment on "Traversable wormhole dynamics on a quantum processor"*, [arXiv:2302.07897](https://arxiv.org/abs/2302.07897)

<a id="ref-3"></a>[3] D. Jafferis et al., *Comment on "Comment on 'Traversable wormhole dynamics on a quantum processor'"*, [arXiv:2303.15423](https://arxiv.org/abs/2303.15423)

<a id="ref-4"></a>[4] S. I. Davis, PhD thesis, Caltech, ch. 14 *Teleportation through the wormhole*. [thesis.caltech.edu/17699](https://thesis.caltech.edu/17699/)

<a id="ref-5"></a>[5] J. Maldacena, L. Susskind, *Cool horizons for entangled black holes*, [arXiv:1306.0533](https://arxiv.org/abs/1306.0533) · P. Gao, D. L. Jafferis, A. C. Wall, *Traversable wormholes via a double trace deformation*, [arXiv:1608.05687](https://arxiv.org/abs/1608.05687)

<a id="ref-6"></a>[6] [README.md](README.md) (nighthawk_qrack, cloverfield §3.5), [METHODOLOGY.md](METHODOLOGY.md), [RESULTS.md](RESULTS.md)
