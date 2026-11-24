# ER=EPR on the cloverleaf: `rcs/nighthawk/nighthawk_erepr.py`

**The 2022 Sycamore wormhole, re-derived from its stated Hamiltonian, compiled onto Nighthawk lattice registers, and run on `nighthawk_qrack.py`'s engines and `nighthawk_cloverfield.py`'s chips and loom: the two sides of the wormhole are two chips, the thermofield double that joins them is the EPR, and the loom is the only way it crosses**

| | |
|---|---|
| Script | `rcs/nighthawk/nighthawk_erepr.py` |
| Builds on | `nighthawk_qrack.py`: `ExactEngine`, `AceEngine`, `two_patch_config`, `layout_row`, `ace_tiling` (v3), `placement_stats`, `u_angles`, `gate`, `find_repo` · `nighthawk_cloverfield.py`: `parse_chip`, `chip_layout`, `Field`, `ChipProc` workers, `Loom` (`ghost`, `mean`, `sample`), `worker_env_for` |
| Data | `erepr/`: `reference.json` (§4), `exact.jsonl`, `ace.jsonl`, `loom.jsonl`, `loom_links.jsonl`, `knit.jsonl` (§5–6), `ds.json` (§4.5), `pairs*.json` (§6.3), `pairs_route*.json` and `qasm/` (§6.4), `tiles.json` (the tiling cache), `run_all.sh` |
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
| In the dS limit (β → 0, the double-scaled SYK state) the sign asymmetry vanishes; no β favours μ > 0 | peak ratio I(+12)/I(−12) 0.59–0.68 at β = 4, 1.005–1.030 at β = 0; teleportation works both ways (SYK F_avg 0.715–0.720) | 4.5 |
| With the coupling dressed by the constraint H_L = H_R, the traversal time and the sign preference both vanish; the sign of μ is the direction of relative time | I_PT depends on t₁ − t₀ alone; I(−μ, τ) = I(+μ, −τ) to 10⁻¹⁵; peak ratio 1.000 at every β | 4.5 |
| An AdS side joined to a dS side with one Hamiltonian is the TFD at the mean temperature | β_L = 4, β_R = 0 equals β = 2 to 4 × 10⁻¹⁶ | 4.5 |
| Thermal mode pairs split across two chips match the single register, and a chip is the side register plus log₂(keep) qubits | max \|d\| ≤ 1.1 × 10⁻⁶ with every branch; 6 pairs on 16–20 qubits per chip against 24 | 6.3 |
| The pair chips route onto rectangles of the Nighthawk lattice without changing the result; the default layout needs no SWAPs | 2 × 6 sites and 16 CZs per chip; max \|d\| ≤ 2.4 × 10⁻⁶ over every routed 64-branch case | 6.4 |
| Independent scramblers hide the pairs without touching the cut entropy; a mirrored scrambler (U* ⊗ U) undoes itself as the pairs get hotter | return ≤ 0.0004 against 0.003 → 0.9998 for r = 0.3 → 3 | 6.3 |
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
| `run --beta b` | any backend with the TFD at inverse temperature b (default 4); 0 is the dS limit |
| `pairs` | thermal mode pairs across an exact cut (§6.3): `--modes`, `--mode-qubits`, `--r` or `--horizon T`, `--keep`, `--ghost none independent mirror`, `--depth`; no release data needed |
| `pairs --route` | both chips on rectangles of the lattice through `Router` (§6.4): `--chips`, `--lattice release grid auto`, `--broken`, `--index-bits`, `--qasm DIR` |
| `ds` | the dS limit (§4.5): β sweep, both signs, Model 1 and SYK N = 10, sign verdict per β; `--dress` adds the constraint-dressed coupling scanned over τ = t₁ − t₀; `--qrack` repeats Model 1 through `knit` |
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
| Pairs: 2 mode pairs (r = 0.6, 0.9), depth 3, all 16 branches, independent and mirror scramblers vs the single register | max \|d\| 5.7 × 10⁻⁷ and 2.7 × 10⁻⁷; return 0.0046 and 0.3770 |
| Pairs routed: the same pairs on 3 × 3 rectangles of the ideal lattice, one site broken, mirror scrambler vs the single register | max \|d\| 1.7 × 10⁻⁶; CZ 63 / 21, SWAPs 18 / 4 |

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

### 4.5 The de Sitter limit (`ds`)

The AdS wrapper: the protocol, V and both Hamiltonians are unchanged, and only the TFD temperature moves. β → 0 is the state of the double-scaled SYK proposal for dS holography [[7]](#ref-7): two SYK systems at infinite temperature joined by H_L = H_R, which every TFD here meets to 7 × 10⁻¹⁷. In AdS the wormhole opens for negative injected energy (μ < 0). In dS a positive-energy shockwave gives a time advance that brings the two poles into causal contact [[8]](#ref-8), so a dS wormhole would favour μ > 0.

| Rule | Value |
|---|---|
| Grid | β ∈ {4, 2, 1, 0.5, 0.25, 0}, 0 ≤ t ≤ 10 at dt = 0.1, symmetric t₀ = t₁, μ = ±12 |
| Ratio | peak I(+12) / peak I(−12) over t |
| Sign | AdS if the ratio is below 0.9, dS if above 1/0.9 = 1.11, blind between |
| Models | Model 1 (trained pair ψ¹ψ²), SYK N = 10, J = 1.25, pair (1,2), seeds 0–2 |

| β | L\|R entanglement, Model 1 / SYK (bits) | Model 1: peak I(−12) / I(+12), ratio | SYK ratio, 3 seeds | Sign, Model 1 / SYK |
|---|---|---|---|---|
| 4 | 3.21 / 4.06–4.20 | 0.241 / 0.154, 0.637 | 0.590–0.678 | AdS / AdS |
| 2 | 3.80 / 4.70–4.73 | 0.268 / 0.248, 0.926 | 0.763–0.806 | blind / AdS |
| 1 | 3.95 / 4.92–4.93 | 0.269 / 0.278, 1.035 | 0.872–0.895 | blind / AdS |
| 0.5 | 3.99 / 4.98 | 0.277 / 0.285, 1.029 | 0.932–0.952 | blind / blind |
| 0.25 | 3.997 / 4.995 | 0.280 / 0.287, 1.025 | 0.968–0.990 | blind / blind |
| 0 | 4.00 / 5.00 | 0.281 / 0.287, 1.020 | 1.005–1.030 | blind / blind |

| Finding | Numbers |
|---|---|
| No temperature gives the dS sign | largest ratio 1.035 (Model 1, β = 1), threshold 1.11 |
| The sign asymmetry fades into a channel that works both ways | max over t of \|I(−12) − I(+12)\|: Model 1 0.150 → 0.055, SYK 0.117–0.122 → 0.008–0.013 from β = 4 to 0 |
| Teleportation gets stronger as the asymmetry goes | SYK peak I(−12) 0.27–0.30 bits at β = 4, I(±12) 0.36–0.38 at β = 0; best F_avg 0.715–0.720 for both signs at β = 0, against 2/3 |
| The training time is a β = 4 feature | Model 1's peak moves from t = 2.80 to 8.2–8.7 for β ≤ 1 |
| A fixed-time read keeps a sign the peaks lose | Model 1 at t = 2.8, β = 0: I(−12) 0.247, I(+12) 0.192 |
| Residual lean at β = 0 | all four models have ratio 1.005–1.030, toward μ > 0; below the threshold, and not separable from finite size at N ≤ 10 |

Reading: at high temperature the circuit teleports by operator spreading under generic scrambling (peaked-size teleportation [[9]](#ref-9)), which needs no gravity and no sign. The wrapper therefore lands in a regime that cannot pick a dS sign. A dS test of the sign needs a coupling or a Hamiltonian derived for a dS horizon, not a temperature.

`--qrack` runs Model 1 through `knit` (§6.2, all 16 Schmidt components, 2048 branches) on `nq.ExactEngine` at t = 2.8:

| β | L\|R (bits) | I(−12) knit / reference | I(+12) knit / reference | max \|d\| |
|---|---|---|---|---|
| 4 | 3.210 | 0.24107 / 0.24107 | 0.10021 / 0.10021 | 5.6 × 10⁻⁸ |
| 2 | 3.802 | 0.26486 / 0.26486 | 0.16103 / 0.16103 | 5.4 × 10⁻⁸ |
| 1 | 3.953 | 0.26035 / 0.26035 | 0.18100 / 0.18100 | 3.2 × 10⁻⁸ |
| 0.5 | 3.989 | 0.25449 / 0.25449 | 0.18753 / 0.18753 | 7.6 × 10⁻⁸ |
| 0.25 | 3.997 | 0.25085 / 0.25085 | 0.19002 / 0.19001 | 5.0 × 10⁻⁶ |
| 0 | 4.000 | 0.24683 / 0.24683 | 0.19202 / 0.19202 | 3.1 × 10⁻⁸ |

**Constraint-dressed coupling (`--dress`).** V does not commute with H_L − H_R (max element of the commutator 0.009–0.012), so under the constraint of [[7]](#ref-7) it is not a physical operator. V_d keeps only the blocks of V that conserve E_L − E_R, which is V averaged over the relative time generated by H_L − H_R; [V_d, H_L − H_R] = 0 to 2 × 10⁻¹⁷. The dressing is this file's construction, not a protocol of [7]. Since V_d and the TFD are both invariant, I_PT depends on τ = t₁ − t₀ alone, and the scan is over τ (−10 ≤ τ ≤ 10, dt = 0.1).

| Check | Model 1 | SYK N = 10, 3 seeds |
|---|---|---|
| Coupling loop of `TwoSided.run` vs e^{iμV} of `V_matrix` | 9.9 × 10⁻¹⁶ | 2.3 × 10⁻¹⁶ |
| Share of \|V\|² that V_d keeps | 16.1 % | 4.3–5.0 % |
| Spread of I_PT along t₀ = t₁ (boost invariance) | ≤ 6.7 × 10⁻¹⁶ | ≤ 8.9 × 10⁻¹⁶ |
| max over τ of \|I(−12, τ) − I(+12, −τ)\| | ≤ 6.7 × 10⁻¹⁶ | ≤ 1.1 × 10⁻¹⁵ |

| β | Model 1: peak I, at τ for μ = −12 / +12 | Model 1: I at τ = 0 | SYK: peak I, at τ for μ = −12 / +12 | SYK: max \|I(−12) − I(+12)\| over τ |
|---|---|---|---|---|
| 4 | 0.170, +0.4 / −0.4 | 0.168 | 0.236–0.255, +0.4 / −0.4 | 0.070–0.075 |
| 2 | 0.251, +0.2 / −0.2 | 0.250 | 0.363–0.372, +0.2 / −0.2 | 0.055–0.059 |
| 1 | 0.276, +0.1 / −0.1 | 0.276 | 0.418–0.428, +0.1 / −0.1 | 0.032–0.035 |
| 0.5 | 0.283, 0 / 0 | 0.283 | 0.434–0.444, 0 to +0.1 / 0 to −0.1 | 0.017–0.019 |
| 0 | 0.285, 0 / 0 | 0.285 | 0.440–0.449, 0 / 0 | 0.001–0.002 |

| Finding | Numbers |
|---|---|
| The coupling sign is the direction of relative time | I(−μ, τ) = I(+μ, −τ) to 10⁻¹⁵, every model and β |
| No traversal time is left | the symmetric protocol is flat in t; the peak sits at τ ≈ ±β/10 (grid 0.1) and at τ = 0 for β = 0 |
| The peaks of the two signs are equal at every β, and the curves coincide at β = 0 | ratio 1.000 throughout; max difference over τ 0.042–0.075 at β = 4, ≤ 0.002 at β = 0 |
| The channel is stronger than with V in the dS limit | SYK I_PT 0.440–0.449 bits at β = 0 against 0.36–0.38 with V |

Reading: once the coupling obeys the constraint, the wormhole's two marks (a traversal time and a preferred sign) are both gone, and what remains is transfer between operators at nearly equal proper time, with the sign of μ equivalent to swapping which side is later. The dressed coupling is not compiled to a circuit: it is a dense 2^(N/2) × 2^(N/2) block projection, numpy reference only.

**AdS on one side, dS on the other.** With the same Hamiltonian on both sides, (e^{−β_L H_L/4} ⊗ e^{−β_R H_R/4})\|I⟩ is the TFD at the mean of β_L and β_R, because H_L\|I⟩ = H_R\|I⟩. For β_L = 4, β_R = 0 the overlap with the β = 2 TFD is 1 to 4 × 10⁻¹⁶, so this hybrid is the β = 2 row above: Model 1 peak 0.268 / 0.248 (blind), SYK peak I(−12) 0.351–0.365 with ratio 0.763–0.806 (AdS sign). A junction that is not a mean temperature needs different Hamiltonians on the two sides, where H_L\|I⟩ ≠ H_R\|I⟩; that is not built here.

```bash
python3 nighthawk_erepr.py ds --dress --qrack --out erepr/ds.json         # ≈ 400 s on 2 vCPU
python3 nighthawk_erepr.py run --backend knit --beta 0 --out erepr/knit_ds.jsonl
python3 nighthawk_erepr.py run --backend exact --field wide --beta 0 --out erepr/exact_ds.jsonl
```

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

### 6.3 Mode pairs across the cut (`pairs`)

The knit applied to a state that is Schmidt-diagonal by construction: M pairs of bosonic modes, one mode of each pair per side (G and M; ghost and matter in the Strano model's pair picture). No release data is needed.

| Item | Value |
|---|---|
| Pair k | two-mode squeezed state cut at 2ⁿ quanta, Σ_m c_m \|m⟩_G \|m⟩_M with c_m ∝ tanhᵐ(r_k); each side alone is thermal at T/ω = −1 / ln tanh² r_k |
| Registers | n qubits per mode, nM per side; mode k on bits [nk, nk + n) |
| Chip | one `nq.ExactEngine` per side: nM qubits and an index register of log₂(keep) qubits, holding Σ_j \|j⟩ ⊗ U\|b_j⟩ for the `keep` heaviest branches b_j; branch weights are applied afterwards, as in §6.2 |
| Scramblers (`--ghost`) | `none`; `independent`: each side its own circuit; `mirror`: G runs the complex conjugate of M's circuit, U* ⊗ U. Circuits: `--depth` layers of Haar single-qubit gates and CZs on alternating bonds of a line |
| return | \|⟨Ψ₀\|Ψ⟩\|², the weight of the scrambled state on the pair state it started from |
| pair I | mutual information between G qubit q and M qubit q, mean and max over q (bits) |
| Reference | the same state and circuits on one register of 2nM qubits, when 2nM ≤ `--single-max` (22) |
| `--horizon T` | one temperature across modes of energy E_k = k + 1: tanh² r_k = e^(−E_k/T) |

Three pairs, n = 2, r = 0.6 (T/ω = 0.804), depth 4, all 64 branches (chips and single register both 12 qubits):

| Scrambler | return | pair I mean / max | Cut entropy (bits) | max \|d\| vs single register |
|---|---|---|---|---|
| none | 1.0000 | 1.158 / 1.534 | 3.474 | 4.6 × 10⁻⁷ |
| independent | 0.0002 | 0.068 / 0.192 | 3.474 | 1.6 × 10⁻⁷ |
| mirror | 0.0977 | 0.469 / 0.820 | 3.474 | 3.7 × 10⁻⁷ |

Squeezing sweep, same register, all branches:

| r | T/ω | Cut entropy (bits) | independent: return, pair I mean | mirror: return, pair I mean |
|---|---|---|---|---|
| 0.3 | 0.405 | 1.371 | 0.0000, 0.040 | 0.0033, 0.101 |
| 0.6 | 0.804 | 3.474 | 0.0002, 0.068 | 0.0977, 0.469 |
| 1.0 | 1.836 | 5.287 | 0.0003, 0.057 | 0.5746, 1.283 |
| 2.0 | 13.65 | 5.986 | 0.0004, 0.047 | 0.9894, 1.970 |
| 3.0 | 100.9 | 6.000 | 0.0004, 0.047 | 0.9998, 1.999 |

Truncation, four pairs at one temperature (`--modes 4 --horizon 1`; single register 16 qubits):

| keep | Chip qubits | Weight kept | Cut entropy (bits) | max \|d\| of pair correlators vs single register: none / independent / mirror |
|---|---|---|---|---|
| 256 (all) | 16 | 1.000000 | 2.459 | 4.8 × 10⁻⁷ / 1.1 × 10⁻⁶ / 5.2 × 10⁻⁷ |
| 64 | 14 | 0.999823 | 2.456 | 5.4 × 10⁻³ / 1.1 × 10⁻³ / 4.5 × 10⁻³ |
| 16 | 12 | 0.984453 | 2.307 | 9.9 × 10⁻² / 1.8 × 10⁻² / 7.2 × 10⁻² |

Six pairs, r = 0.3 (`--modes 6 --r 0.3 --keep 16 64 256`); the single register is 24 qubits and was not run:

| keep | Chip qubits | Weight kept | Cut entropy (bits) | Seconds per chip pair (2 vCPU) |
|---|---|---|---|---|
| 16 | 16 | 0.924815 | 2.099 | 0.10–0.27 |
| 64 | 18 | 0.988520 | 2.607 | 0.36–0.37 |
| 256 | 20 | 0.999493 | 2.734 | 1.48–1.51 |

| Finding | Numbers |
|---|---|
| With every branch kept the two chips reproduce the single register | max \|d\| ≤ 1.1 × 10⁻⁶ over every case above (Qrack single precision) |
| Scrambled, not lost | independent scramblers: return ≤ 0.0004 and pair I 0.04–0.07 bits (1.158 unscrambled at r = 0.6), with the cut entropy unchanged |
| The mirror scrambler undoes itself as the pairs get hotter | return 0.003 → 0.098 → 0.575 → 0.989 → 0.9998 for r = 0.3 → 3; pair I → 2 bits. U* ⊗ U leaves Σ_b \|b⟩\|b⟩ invariant, the statement H_L = H_R of §4.5 in circuit form |
| A chip is the side register plus log₂(keep) qubits | 6 pairs: 16–20 qubits per chip against 24, carrying 92.5–99.95 % of the weight |
| Truncation error tracks the weight dropped | correlators off by 10⁻³–10⁻² at 0.02 % dropped, 10⁻²–10⁻¹ at 1.6 % |
| With every branch kept a chip is as wide as the single register | the index register then has nM qubits; the saving comes from truncation, so it is largest at low squeezing |

```bash
python3 nighthawk_erepr.py pairs --out erepr/pairs.json
python3 nighthawk_erepr.py pairs --r 2.0 --keep 0 --ghost independent mirror --out erepr/pairs_r2.0.json
python3 nighthawk_erepr.py pairs --modes 4 --horizon 1.0 --keep 0 64 16 --out erepr/pairs_horizon.json
python3 nighthawk_erepr.py pairs --modes 6 --r 0.3 --keep 16 64 256 --out erepr/pairs_6.json
```

### 6.4 Pairs on the Nighthawk lattice (`pairs --route`)

Each chip of §6.3 placed on a rectangle of the 12 × 10 lattice and routed with the `Router` of §3.2, so the circuit is single-qubit gates and CZs on couplers only.

| Item | Value |
|---|---|
| Chips (`--chips`) | two `rect:r0-r1,c0-c1` tokens, G then M; default two stacked rectangles, min(nM, 10) wide |
| Lattice (`--lattice`) | `release`: the release's sites and couplers for the rectangle, broken sites dropped; `grid`: the ideal square lattice, `--broken r,c` to leave sites out; `auto`: `release` when the release is found |
| Placement | side qubit q on the q-th site of a boustrophedon from the chip's top row (line neighbours are couplers wherever the lattice is unbroken); each index bit on the free site nearest its partner side qubit |
| Branch set (`--index-bits b`) | the lowest 2ᵇ Fock levels of every mode: index bit i of mode k is copied to side bit i of mode k by one CX; b = n is every branch. The lookup of `--keep` (multi-controlled X) is not routed |
| Circuit | H on the index register, one CX per index bit, the scrambler; a gate between non-neighbours walks its first qubit along a shortest path with SWAPs and back |
| `--qasm DIR` | `pairs_<scrambler>_b<bits>_<G or M>.qasm` in the release's gate set (rz, rx, cz), with the site table (lattice position, physical index, role) as comments; each file is read back through `nq.qasm_ops` and run again |

Three pairs, n = 2, r = 0.6, depth 4, on the ideal lattice (all checks against the 12-qubit single register):

| Chips | Index bits (branches) | Sites per chip | CZs per chip: none / scrambled | SWAPs per chip | Weight kept | max \|d\| vs single register |
|---|---|---|---|---|---|---|
| `rect:0-1,0-5`, `rect:2-3,0-5` (default) | 2 (64) | 12 | 6 / 16 | 0 | 1.000000 | ≤ 1.3 × 10⁻⁶ |
| same | 1 (8) | 12 | 3 / 13 | 0 | 0.786845 | truncation: 0.09–0.53 |
| `rect:0-2,0-3`, `rect:5-7,2-5` | 2 (64) | 12 | 42 / 52 | 12 | 1.000000 | ≤ 1.7 × 10⁻⁶ |
| `rect:0-2,0-4`, `rect:3-5,0-4`, `--broken 0,2 4,1` | 2 (64) | 14 | G 76, M 52 (scrambled) | G 20, M 12 | 1.000000 | ≤ 2.4 × 10⁻⁶ |

Six pairs, r = 0.3, one index bit per mode (`--modes 6 --r 0.3 --route --index-bits 1`); the single register is 24 qubits and was not run:

| Chips | Branches | Sites per chip | CZs per chip | SWAPs per chip | Weight kept | Cut entropy (bits) |
|---|---|---|---|---|---|---|
| `rect:0-1,0-9`, `rect:2-3,0-9` | 64 | 20 | 52 | 8 | 0.957858 | 2.375 |

| Finding | Numbers |
|---|---|
| The default layout needs no SWAPs | side register on one row, each index bit on the site below its partner: 6 CX copies and 10 scrambler CZs per chip |
| Routing does not change the result | every 64-branch case matches the single register to ≤ 2.4 × 10⁻⁶; weight on unused sites ≤ 2.3 × 10⁻⁶ |
| The QASM files reproduce the routed run | 1 − overlap of the read-back state with the routed state ≤ 8.8 × 10⁻⁷ over the 12 default files |
| A broken site costs SWAPs, not exactness | the boustrophedon breaks at the hole: 20 SWAPs on that chip against 12 |

```bash
python3 nighthawk_erepr.py pairs --route --index-bits 2 1 --qasm erepr/qasm --out erepr/pairs_route.json
python3 nighthawk_erepr.py pairs --route --chips rect:0-2,0-3 rect:5-7,2-5 --out erepr/pairs_route_3x4.json
python3 nighthawk_erepr.py pairs --route --lattice grid --chips rect:0-2,0-4 rect:3-5,0-4 --broken 0,2 4,1 \
    --ghost independent mirror --out erepr/pairs_route_broken.json
python3 nighthawk_erepr.py pairs --modes 6 --r 0.3 --route --index-bits 1 --ghost independent mirror \
    --out erepr/pairs_route_6.json
```

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
| dS is a temperature and a constraint here, not a geometry | §4.5 moves β and, with `--dress`, projects V onto the constraint H_L = H_R; H_L and H_R are the AdS protocol's, and no coupling derived from a dS horizon is in the script |
| `pairs --route` on the release lattice | run here on the ideal lattice only (`--lattice grid`, with and without `--broken`); the `release` branch builds its geometry with `cf.chip_layout` as the other backends do and was not run |
| `pairs --keep` is logical, not routed | the top-`keep` lookup uses multi-controlled X on the whole index register; `--route` takes the product branch set of `--index-bits` |
| Recombining the two chips | the branch sum uses each chip's state vector; on hardware the same matrix elements are Pauli expectation values on the index register, 4^(index bits) strings per chip, which is not implemented |
| `pairs` prepares the cut squeezed state directly | c_m ∝ tanhᵐ r renormalised over 2ⁿ levels; it is not the truncated-generator evolution, which differs near the top level |
| The dressed coupling is reference only | V_d is a dense block projection, not a sum of ZZ rotations; `run` and the Qrack backends use V |
| Size winding in the dS limit | χ and r̄ are normalised by 1 − W and are undefined at β = 0; `ds.json` stores them, §4.5 does not use them |
| System size | N ≤ 10 Majoranas per side; the β = 0 lean toward μ > 0 (ratio ≤ 1.030) is within finite-size reach |

---

## 9. References

<a id="ref-1"></a>[1] D. Jafferis, A. Zlokapa, J. D. Lykken, D. K. Kolchmeyer, S. I. Davis, N. Lauk, H. Neven, M. Spiropulu, *Traversable wormhole dynamics on a quantum processor*, Nature 612, 51–55 (2022). [doi:10.1038/s41586-022-05424-3](https://www.nature.com/articles/s41586-022-05424-3)

<a id="ref-2"></a>[2] B. Kobrin, T. Schuster, N. Y. Yao, *Comment on "Traversable wormhole dynamics on a quantum processor"*, [arXiv:2302.07897](https://arxiv.org/abs/2302.07897)

<a id="ref-3"></a>[3] D. Jafferis et al., *Comment on "Comment on 'Traversable wormhole dynamics on a quantum processor'"*, [arXiv:2303.15423](https://arxiv.org/abs/2303.15423)

<a id="ref-4"></a>[4] S. I. Davis, PhD thesis, Caltech, ch. 14 *Teleportation through the wormhole*. [thesis.caltech.edu/17699](https://thesis.caltech.edu/17699/)

<a id="ref-5"></a>[5] J. Maldacena, L. Susskind, *Cool horizons for entangled black holes*, [arXiv:1306.0533](https://arxiv.org/abs/1306.0533) · P. Gao, D. L. Jafferis, A. C. Wall, *Traversable wormholes via a double trace deformation*, [arXiv:1608.05687](https://arxiv.org/abs/1608.05687)

<a id="ref-6"></a>[6] [README.md](README.md) (nighthawk_qrack, cloverfield §3.5), [METHODOLOGY.md](METHODOLOGY.md), [RESULTS.md](RESULTS.md)

<a id="ref-7"></a>[7] V. Narovlansky, H. Verlinde, *Double-scaled SYK and de Sitter holography*, JHEP 2025(5), 32. [doi:10.1007/JHEP05(2025)032](https://doi.org/10.1007/JHEP05(2025)032)

<a id="ref-8"></a>[8] L. Aalsma, G. Shiu, *Chaos and complementarity in de Sitter space*, JHEP 2020(5), 152. [arXiv:2002.01326](https://arxiv.org/abs/2002.01326)

<a id="ref-9"></a>[9] T. Schuster, B. Kobrin, P. Gao et al., *Many-body quantum teleportation via operator spreading in the traversable wormhole protocol*, Phys. Rev. X (2022). [arXiv:2102.00010](https://arxiv.org/abs/2102.00010)
