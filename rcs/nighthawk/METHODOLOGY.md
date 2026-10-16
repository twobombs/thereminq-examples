# Methodology: `rcs/nighthawk`

How `nighthawk_qrack.py`, `nighthawk_cloverfield.py` and the experiments of the [README](README.md) work, step by step. [RESULTS.md](RESULTS.md) holds the results; this file holds the procedure that produced them. Constants are quoted from the code (5026 lines, 16 October 2026).

---

## 0. Principles

| Principle | How it is applied |
|---|---|
| The release is ground truth | circuits, layout, partitions, seeds, inputs and hardware counts are read from `BlueQubitDev/rcs-nighthawk`; nothing is re-derived from the paper's text |
| Ideal qubits only | no noise model; every loss of fidelity is attributed to the simulation method or to sampling |
| The paper's own proxies | mirror survival and patched XEB as defined by the release; forward XEB only where an exact reference exists |
| Ground truth before proxy | wherever the size allows (n ≤ 34–36), ACE is scored against an exact reference (`fxeb`) |
| Controls first | the exact backend must give F = 1, and `verify` / `selftest` must pass before a result is used |
| Deterministic inputs | circuits, twirls, measurement orders, placements and tilings are seeded or cached; the known non-deterministic parts are listed in §10.3 |
| Report what was measured | plain means over circuits with instance-scatter errors; inverse-variance means are labelled as such |
| One configuration, one tag | every setting that can change a number enters the configuration hash (§10.1) |

---

## 1. Inputs and environment

| Input | Source | Used for |
|---|---|---|
| `data/layout.json` | release | 61 logical qubits, `logical_to_physical`, colour matchings A / B / C / D (29 / 22 / 29 / 22 couplers), schedule A B C D, 5 partitions each for K = 3 and K = 4, base seed 2025, 3 instances, depths, 10 mirror inputs of 30 ones in 61 bits |
| `data/circuits/**.qasm`, `manifest.json` | release | `verify` |
| `data/counts/*` | release | `hwxeb`, device mirror points (`danmodel`, summaries) |
| `data/results/*.json` | release | the paper's fit F(d) = 0.326 × 0.8717^d, patch XEB |
| `scripts/` (`layout`, `patching` modules) | release | sub-register construction (§2.5) |
| `contraction_cost/results/depth_sweep.json` | release | verification cost (README §5.7) |
| Dan Strano's ACE spreadsheet export | user upload | calculator constants (§8) |

| Machine | Role |
|---|---|
| 96-thread EPYC, 320 GB RAM, 256 GB NVMe RAID0 swap, 6 × AMD Radeon Pro V340 dies (8 GB) | CPU production runs (61-qubit mirrors, `fxeb` to 34 qubits), clvk / rusticl GPU runs |
| NVIDIA CMP 50HX (10 GB), NVIDIA OpenCL | GPU validation of tiled ACE and references |
| PoCL (CPU OpenCL) | OpenCL defect reproduction and fix validation |
| 2-core, 7 GB cloud container | selftests, CP-SAT searches, small and shallow runs (ladder ≤ 45 qubits, d = 4–6 at 61 qubits) |

| Software | Version |
|---|---|
| PyQrack | 2.28.1 and the release before it; `QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"` exported as `PYQRACK_SHARED_LIB_PATH` |
| Qrack | 10.13.1 / 10.13.2 (`cf53586`) with `qrack_ocl_async_fix.patch` for GPU runs |
| OR-Tools CP-SAT | placement and register search |
| Qiskit | `--ace-transpile` coupling-map handling |
| NumPy | all estimators and fits (sums in float64) |

---

## 2. Circuit construction

### 2.1 Single-qubit angles

Every angle comes from a counter-based hash, so any gate of any circuit can be regenerated alone:

| Step | Definition |
|---|---|
| mixer | SplitMix64: x ← x + 0x9E3779B97F4A7C15; z ← (x ⊕ x≫30)·0xBF58476D1CE4E5B9; z ← (z ⊕ z≫27)·0x94D049BB133111EB; out = z ⊕ z≫31 (mod 2⁶⁴) |
| uniform | u(seed, instance, cycle, qubit, k): fold the five parts through the mixer, keep the top 53 bits / 2⁵³ |
| φ, λ | 2π · u(…, 1), 2π · u(…, 3) |
| θ, Haar (released) | z = 1 − 2u(…, 2); θ = arccos z (cos θ uniform on [−1, 1]) |
| θ, `nnqab` (variant) | θ = arcsin z (sin θ uniform): same hash stream, so a variant differs from the released circuit in θ only |
| gate | rz(φ) then rx(θ) then rz(λ); matrix rz(λ)·rx(θ)·rz(φ) |
| on ACE | converted to a Qrack `u(θ′, φ′, λ′)` up to global phase (`u_angles`, pole-safe) |

### 2.2 Seeds per family

| Family | Seed | Instance argument |
|---|---|---|
| mirror | 2025 | instance k |
| patched, full, `fxeb` | 2025 + k · 1 000 003 | instance k |
| twirl Paulis | `numpy.default_rng([2025, d, k, input s, twirl r])` | — |
| `prob` measurement order | `numpy.default_rng([2025, d, k, s, r, 7])` | — |

### 2.3 Cycles

| Step | Rule |
|---|---|
| cycle c | single-qubit gate on every qubit, then the CZs of colour `schedule[c mod 4]` |
| truncation to n qubits | qubits ≥ n dropped; couplers kept only with both ends < n; mirror inputs truncated to their first n bits |
| patched | every coupler in the partition's boundary set removed in every cycle |
| mirror pseudo-patching | in forward cycle c the couplers of K = 3 boundary set `c mod 5` are removed (`rotate_every = 1`) |
| mirror depth d | d/2 forward cycles, then their exact inverse: reversed cycle order, CZs in reversed order, each single-qubit gate replaced by its conjugate transpose |
| mirror input | X on every qubit whose input bit is 1, from \|0…0⟩ |
| full / `fxeb` | unpatched forward circuit, d cycles |

The CZ pattern of a mirror circuit depends on the depth only, not on the instance. Boundary-CZ counts (§7.4) are therefore one number per register and depth.

### 2.4 Verification of the regeneration

`verify` rebuilds every released circuit (mirror, patched K = 3 and 4 over all partitions and instances, full at d = 4, 8, …, 40) and compares it with the released OpenQASM operation by operation:

| Check | Criterion |
|---|---|
| operation sequence | same gate type and qubits at every position |
| angles | equal modulo 2π within `--tol` = 10⁻⁹ rad |
| CZ counts | equal to `manifest.json` |
| coverage | every manifest entry regenerated |
| exit status | 0 only if all of the above hold |

### 2.5 Sub-registers

Registers other than the release follow the release's construction code:

| Element | Procedure |
|---|---|
| region | a window of the 12 × 10 lattice plus a list of dropped device sites; logical order row-major |
| couplers | the release's `rect_matchings(R, C)` then `remove_qubits` for the dropped sites |
| partitions | `make_patches(n, edges, K, seed=2025, balance_tol=1, top_n=5)` for K = 3 and 4 |
| inputs | ten strings, each with ⌊n/2⌋ ones at positions drawn by `numpy.default_rng(2025).choice(n, n//2, replace=False)` |
| check | selftest [9]: on the release's own window the recipe reproduces the release's inputs and partitions exactly |
| release prefixes | `release:N` = the bounding window of the first N logical qubits with every other window site dropped; at N = 61 identical to the release (circuits, angles and inputs verified equal) |

---

## 3. Simulation engines

| Engine | Construction | Gates |
|---|---|---|
| exact | `QrackSimulator(n, is_gpu = not --cpu)` | `mtrx` for single-qubit gates, `mcz` for CZ |
| ACE | `QrackAceBackend(size, long_range_columns = lrc, long_range_rows = lrr, is_torus, is_gpu, is_host_pointer, …)`; only non-default switches are passed (`is_boundary_repetition_code`, `is_error_detection`, `is_crossbars`) | logical qubit q acts on ACE site `site[q]` from the placement (§4); `u` for single-qubit gates, `cz`, Paulis by name |
| ACE reset | measure all sites (`m_all`), apply X where the outcome was 1: back to \|0…0⟩ without reallocating device buffers | — |
| ACE rebuild | after a C++ exception the backend is deleted and constructed again with the same arguments | — |
| exact reference (`fxeb`) | a separate `QrackSimulator(n)` running the same gates; all 2ⁿ probabilities copied into one NumPy buffer of Qrack's real type via `OutProbs` | — |

OpenCL is used only when asked for (`--ace-gpu`, `--ref-gpu`). CPU runs set `OCL_ICD_VENDORS` to an empty directory, so no OpenCL platform initialises.

---

## 4. Placement

A placement maps each logical qubit to an ACE site. Sites are classed by the set of simulators holding them: one patch's bulk, or a seam shared by two patches. A coupler is *exact* inside one bulk, *replica* when a seam site is involved and the ends share a simulator, and *cross* when they share none.

### 4.1 Tiling (`--ace-tiling`)

| | v2 | v3 (default) |
|---|---|---|
| cost | 10 · cross + 1 · replica | 20 · cross + 1 · replica + 6 · (qubits on seam sites) + 2 · (simulators holding bulk) |
| moves | swap two qubits, or move one to a free site | 85 % swaps, 15 % moves to a free site |
| schedule | 200 000 iterations, seed 2 | 150 000 iterations per restart, temperature 8 → 0.1 geometric |
| restarts | one, from the device placement | three (seeds 3, 1003, 2003): from the device placement and from the best (n−1)-qubit tiling extended by the cheapest free site |
| cache | — | `nighthawk_tiles.json`, keyed by a SHA-1 over weights, iterations, seeds, ACE layout, start and couplers |

### 4.2 Exact placement on a given register (`--ace-transpile`, engine `exact`)

| Element | Definition |
|---|---|
| groups | sites grouped by simulator set; sites within a group are interchangeable (a patch is a dense, fully connected state) |
| variables | x[q, g] = 1 if logical qubit q goes to group g |
| constraints | each qubit in one group; group capacities; every circuit coupler on a group pair that ACE's coupling map connects (`--ace-exclude` sites removed first) |
| width caps | `--transpile-max-width W`: every patch ≤ W − 1 qubits (one ancilla), at most `--transpile-max-wide N` patches at W − 1, the others ≤ W − 2 |
| objective | minimise 10⁶ · (qubits on seams) + 10³ · (widest − narrowest patch) + (couplers touching a seam) |
| solver | CP-SAT, 120 s, up to 8 workers, random seed 0 |
| result | qubits filled into the sites of their group in site order; cached per configuration; registers store the result (`ace.site`) |

`anneal` minimises the v3 cost on the same register and weights a cross coupler between patches without a shared seam by 1000. `vf2` uses Qiskit's VF2Layout with seam error weights 2 × 10⁻² per coupler and 10⁻² per qubit (bulk 10⁻⁴), bounded at 2000 layouts and 2 × 10⁶ calls. A placement containing a coupler between patches with no shared seam is refused unless `--transpile-allow-cross`.

### 4.3 Joint register and placement search (`layoutsearch`)

| Element | Definition |
|---|---|
| cells | the lattice window (default 12 × 10); `inc[v]` = cell in the register; broken qubits 17, 55, 62 forced out with `--broken disable` |
| groups | x[v, g] = 1 if cell v holds a qubit of group g; Σ_g x[v, g] = inc[v]; group capacities |
| coupling | for every lattice coupler (u, v) and every group pair ACE does not connect: ¬(x[u, a] ∧ x[v, b]) |
| size | Σ inc = n |
| region `rect` | row and column indicator runs without gaps; a cell inside the rectangle is used unless broken or one of at most `--trim` sites on the rectangle's border |
| region `convex` | every row and every column of the region is one unbroken run; a disabled broken site only in the interior |
| region `free` | any set; connectivity by a single-commodity flow of n units from one root |
| region `fixed` | exactly the cells of `--window` or `--fixed-register` |
| terms | `seam` = boundary qubits; `bcz` = couplers in use touching a seam; `cz` = Σ w_e · (coupler e in use and touching a seam), w_e from §7.4; `couplers` = couplers kept (maximised); `balance` = widest − narrowest patch |
| objective | lexicographic: weighted sum with each term scaled to exceed the full range of every later term (512 per step; `cz` scaled by 2 · Σ w_e + 1) |
| caps | as §4.2, widths including the ancilla |
| diversity | each later candidate differs from every earlier one in ≥ `--diversity` seam cells (no-good cuts) |
| solver | CP-SAT, `--cpsat-time` (300 s), up to 16 workers, random seed = candidate index; FEASIBLE = best found, not proven |
| verification | each candidate rebuilt through `RegisterLayout` and the normal placement path: one qubit per site, no cross coupler, widths within caps, coupler counts consistent |
| test | `run --register <candidate> --mirror-estimator prob` at `--test-depths`; score = geometric mean over depths of the plain-mean survival over instances |
| ranking | by score; by the objective when untested |

Two caveats follow from the model. Regions without the `rect` rule admit one- or two-site necks, i.e. nearly separate circuits. A width cap at or above n lets the solver put everything in one exact patch, so the two-patch ladder caps the widest patch at n/2 + 5.

---

## 5. Twirling

Each CZ(a, b) of a twirled run is applied as:

| Step | Operation |
|---|---|
| 1 | draw P_a, P_b uniformly from {I, X, Y, Z} |
| 2 | apply P_a on a, P_b on b |
| 3 | CZ(a, b) |
| 4 | apply P_a on a, P_b on b again |
| 5 | Z on a if P_b ∈ {X, Y}; Z on b if P_a ∈ {X, Y} |

The sequence equals CZ up to global phase (selftest [4]). One generator per (depth, instance, input, twirl) serves the forward and the inverse half, so the two halves carry independent Paulis. Twirling is off on the exact backend and for `fxeb`, whose unmirrored circuit has no inverse half for a coherent error to cancel against.

---

## 6. Estimators

### 6.1 Mirror, `shots`

| Step | Rule |
|---|---|
| budget | the paper's Appendix-D shots per depth (30 000 at d = 4–8 … 360 000 at d = 40), or `--shots N` |
| split | per input: budget / 10; per twirl: an even share, remainder to the first twirls |
| run | reset, prepare input, forward, inverse, sample `measure_shots` over the logical qubits |
| score | survival p = hits / shots over all inputs and twirls; σ = √(p(1−p)/shots) |
| exact backend | probability of the input string read directly (`prob_perm`) when `--exact-probs` |

### 6.2 Mirror, `prob` (ACE)

| Step | Rule |
|---|---|
| runs | 10 inputs × `--twirls` (at least 1) |
| per run | reset, prepare, forward, inverse; then, over the qubits in a random order (seeded, §2.2): p_q = P(qubit = wanted bit), multiply, force that outcome with ACE's `force_m` |
| result per run | Π p_q: the probability ACE's sampler returns the input; its expectation over the order equals the survival `measure_shots` estimates |
| floor | a factor ≤ 10⁻⁶ ends the run with 0 (below fp32 resolution, where `force_m` would throw) |
| failure | a C++ exception counts the run as 0, records the partial product as an upper bound (`prob_errors`, `prob_error_bound`) and rebuilds the backend |
| point | F = mean over runs; σ = sample std / √runs; per-input means kept in `survival` |

### 6.3 Forward XEB (`fxeb`)

| Step | Rule |
|---|---|
| samples | one forward run, 4096 shots (`measure_shots`) |
| reference | exact probabilities p of the same gates; Σp and Σp² in float64 |
| score | XEB = (N · mean_s p̂(x_s) − 1) / (N Σ p̂² − 1), p̂ = p / Σp, N = 2ⁿ |
| error | σ = std_s(N p̂(x_s) − 1) / √shots / (N Σ p̂² − 1) |
| also | linear XEB N · mean p̂ − 1, ideal XEB N Σ p̂² − 1, HOG = fraction of samples with p̂ > ln 2 / N |
| shared reference | with `--variants`, all configurations sample first, then one reference scores them all |

### 6.4 Patched and full

| Family | Rule |
|---|---|
| patched | F = Π_r mean_s [(2^{n_r} p_r(x_r) − 1) / (2^{n_r} Σ p_r² − 1)], patch distributions exact and renormalised in float64 (cacheable); σ by a second-order delta method on the product of means (gradient and Hessian terms) |
| full | samples stored, not scored |

---

## 7. Statistics and derived quantities

### 7.1 Aggregation

| Quantity | Definition | Where |
|---|---|---|
| point | one (register, family, depth, instance) record | JSONL |
| inverse-variance mean | Σ w F / Σ w, w = 1/σ², error √(1/Σ w); plain mean when any σ = 0 | `run --summarize` |
| plain mean over instances | mean of the instance values; error = std (ddof = 1) / √instances | graph tool, `danmodel`, README results |
| paired difference | per-circuit difference between variants on the same circuits and reference | variant comparisons |

The plain mean is the quantity of interest (an average over circuits). The inverse-variance mean weights instances by shot noise, which grows with their own XEB, and is biased low when instances differ.

### 7.2 Decay fits

| Fit | Definition |
|---|---|
| exponential | log₁₀ F = a + d · log₁₀ f by least squares; F(d) = A · f^d; b = −ln f |
| floor | mirror depths with F ≤ 3 / shots are dropped from the fit (`shots` estimator) |
| b(N) model | b = u · N + v · (CZs per cycle), least squares, constrained non-negative (best single term if the joint fit goes negative); extrapolated to N = 61; N and CZs per cycle are nearly collinear, so only b(61) is interpreted |
| power law (Dan's form) | log(−ln F) = c + p · log(d / d₀), least squares over points with 0 < F < 1 and d ≥ 6, d₀ = 12; gives the fitted p and XEB(12) = exp(−e^c) |
| per boundary CZ | Δ(−ln F) / Δ(boundary CZs) between depths |

### 7.3 Device reference

| Quantity | Source |
|---|---|
| per-depth device survival | release counts: hits / shots per instance and input, inverse-variance mean over instances |
| device fit | release `fidelity_vs_depth.json`: F(d) = 0.326 × 0.8717^d, b_dev = 0.137 |

### 7.4 Boundary-CZ count

For a register, placement and depth d: the number of CZs in the mirror's forward half (d/2 cycles, pseudo-patching applied) whose coupler is not exact under the placement. It is instance-independent (§2.3). The counter reproduced the three-patch column of RESULTS §6 exactly before it was used for new registers. `layoutsearch --cz-weights mirror[:D…]` sums the same counts per coupler over the depths D (default 4, 6, 8, 10, 12, 16, 20) as objective weights.

---

## 8. Dan Strano's calculator

| Element | Definition |
|---|---|
| source | "XEB prediction calculator" sheet: QrackAceBackend, PyQrack 2.28.2, error detection on, torus, SDRP off, native SWAP; `nn_qab` circuits of 10–27 qubits, 2–3 patches, 200 circuits per series |
| constants | power 2/3, slope 0.1653638846, intercept −0.0378757892, depth power 1.60569, d₀ = 12 |
| prediction | x = (bulk / (boundary · patches))^(2/3); XEB(12) = tanh(slope · x + intercept); XEB(d) = XEB(12)^((d/12)^p) |
| edge cases | boundary 0 or one patch → 1; XEB(12) ≤ 0 → 0 |
| inputs from a placement | boundary = logical qubits on seam sites (`ace_seam_used`); bulk = n − boundary; patches = simulators holding bulk (`ace_sims_used`) |
| reproduction | the sheet's worked example (52 / 7 / 2, d = 36) gives 0.0019774 in both; the depth power was refitted from the sheet's depth series (d = 14–22) to 1.610, with leave-one-depth-out values 1.58–1.66 |
| known inconsistency | the sheet's depth fit raises the atanh-space value to the power, while its calculator raises tanh of it; at d = 36 the two differ by ≈ 27 % (0.00252 vs 0.00198); `danmodel` follows the calculator |

---

## 9. Experiments

Each row is one experiment as run. "Runs" is per point: instances × (inputs × twirls) for mirrors, instances × shots for `fxeb`.

| # | Question | Register / placement | Family, θ, estimator | Depths | Runs | Hardware | RESULTS § |
|---|---|---|---|---|---|---|---|
| E1 | are the circuits the released ones? | release | all, Haar | all | — | any | 1 |
| E2 | does PyQrack reproduce the paper's hardware numbers? | release | patched + mirror counts | all | release data | CPU | 1 |
| E3 | layout or placement? | 8 × 8 window, five lrc/lrr, device vs tiled v2 | `fxeb`, Haar | 8, 12 | 3 × 4096 | CPU | 2 |
| E4 | does tiling v3 beat v2, and how does XEB scale with n and d? | 4/4, v2 and v3 on the same circuits, one shared reference | `fxeb`, Haar | 12, 16, 20 | 3 × 4096 | CMP 50HX (`--ace-gpu --ref-gpu`) | 3 |
| E5 | do ACE's switches change the seam cost? | 4/4 v3 with `rep`, `noed`, `noxbar` variants | `fxeb`, Haar | 12, 16 | 5 × 4096 | CMP 50HX | 3 |
| E6 | does the ensemble matter? | 4/4 tiled v3 vs device placement, n = 27 | `fxeb`, `nn_qab` | 12 | 20 × 4096 | CPU ACE, GPU reference (`--ref-gpu`) | 4 |
| E7 | which 61-qubit placement? | 75 sites, lrc 4 lrr 5, four exclusion / engine settings | mirror, `nn_qab`, shots | 4, 6 | 1 × 1000 shots, 4 twirls | CPU | 6 |
| E8 | how does ACE decay at the full register? | three patches, exact placement, 11 exclusions | mirror, `nn_qab`, `prob` | 4–20 | 3 × (10 × 64) | EPYC, 6 CPU workers | 6 |
| E9 | two or three patches at the same qubits? | 49-qubit sub-register, 55 vs 75 sites | mirror, `nn_qab`, `prob` | 4, 6 | 1 × 20 | CPU | 7 |
| E10 | do disabled sites on the seams help at 88 qubits? | `nighthawk88`, `88d`, `88e` | mirror, `nn_qab`, `prob` | 4, 6 | 3 × (10 × 4) | CPU | 8 |
| E11 | two patches at 61 qubits? | `nighthawk61r2` (error detection on / off), `nighthawk61w`; 45-qubit waist analogue | mirror, `nn_qab` and Haar, `prob` | 4, 6 (61); 8–20 (45) | 3 × (10 × 8); 1 × 10 | container CPU | 9 |
| E12 | does Dan's calculator scale? | `nighthawk2p28` … `2p45` | mirror, `nn_qab` and Haar, `prob`; `fxeb` at 28 | 4–20, 36 | 3 × 10 (≤ 36), 1 × 10 (40, 45); `fxeb` 1 × 4096 | container CPU | 10 |
| E13 | does a ghost link carry entanglement? | two exact 9-qubit chips; two `nighthawk88e` chips | mirror, `nn_qab`, `prob` per chip | 8–16; 4, 6 | selftest; 3 × (10 × 2) | CPU | 11 |
| E14 | why do deep GPU points fail? | 16-qubit reproducer; ACE mirror | — | 20 layers; 4, 6 | 5 runs; 3 × 640 | PoCL | 12 |
| E15 | does `--gtt` lift the allocation ceiling? | 24-qubit entangled state | — | — | 1 | PoCL | 12 |

### 9.1 Design rules applied to every experiment

| Rule | Practice |
|---|---|
| same circuits across arms | arms of a comparison share seeds, so they run identical gates; `fxeb` variants also share one reference |
| controls | exact backend at the same settings gives F = 1 (selftest [3], [4]) |
| GPU vs CPU | one point per new device or driver is rerun on the CPU before GPU results are used |
| ensemble labelling | `--theta nnqab` runs are tagged and never mixed with Haar runs in one table or fit |
| depth window | ranking on d ≥ 12 for `fxeb` (outputs concentrated below, ideal XEB ≫ 1) |
| memory planning | widths and peak memory computed before launch (§10.4); points too large for the budget are marked failed, not attempted |
| register fairness | registers other than the release are compared with the device only together with their narrowest cut (RESULTS §9) |

---

## 10. Execution and reproducibility

### 10.1 Configuration tag

`<backend>-<first 8 hex of SHA-1>` over a sorted JSON of: backend, shots, twirls, θ (if not Haar), lrc, lrr, exact-probs, torus, ACE layout, max width and torus search, non-default ACE switches, transpile settings (engine, width, exclusions, seed, trials, calls, allow-cross, caps), tiling version, mirror estimator (if not shots), register name (if not the release). Not included: register size (sweeps extend per n), GPU/CPU placement, `--gtt`, `--swap-gb`, worker counts. Records carry the full geometry (`ace_*` fields, `ace_map` = SHA-1 of the placement).

### 10.2 Work queue and records

| Element | Protocol |
|---|---|
| jobs | every (n, family, K, d, partition, instance) of the request; ordered `deep` (most expensive first) or `shallow` |
| claim | file created with `O_CREAT` + `O_EXCL`, holding `pid:attempt`; `done` is final |
| dead worker | its claim is taken over up to `--max-attempts` (2) workers, then marked `failed: …` and skipped |
| record order | bitstrings written atomically first, then the JSONL record, then the claim marked `done` |
| resume | records are read per tag; finished points are skipped; worker files are folded in |
| watchdog | `--max-rss-gb`: a worker over the cap marks its point failed and exits |
| progress | per-worker JSON progress file (runs, cycles, ETA) read by `--status` |

### 10.3 Determinism

| Deterministic | Not deterministic |
|---|---|
| circuits, inputs, partitions (hash and seeds) | Qrack's internal RNG in ACE measurement and reset (unseeded): two processes differ by ≈ 10⁻⁴ on the same `prob` point |
| twirl draws and `prob` measurement orders (seeded generators) | `measure_shots` samples |
| tilings (seeded annealing, cached) | CP-SAT with several workers can return a different solution of the same objective value; stored placements remove this from runs |
| stored placements of every register | GPU floating-point order |

### 10.4 Memory planning

| Quantity | Model |
|---|---|
| patch | 8 · 2^w bytes (fp32), w = qubits on the patch's sites + ancilla; unused sites stay separable and unallocated |
| compose transient | + half the widest patch's size (old state held while the wider one is built) |
| reference | 12 · 2ⁿ bytes (state + probability buffer) |
| point estimate | 1 GiB + engines + reference |
| ledger | one file per worker in a shared directory; a point is taken only if Σ reservations + estimate ≤ `--mem-budget-gb` and estimate + others' unallocated part ≤ 0.95 × (MemAvailable [+ swap up to `--swap-gb`]) |
| GPU die | a state stays in VRAM if 3 × size ≤ device memory; the preflight's `die_split` sums patches and compose buffer into VRAM and host parts, minus a 0.5 GiB reserve |
| Qrack ceilings | for CPU patches over 28 qubits, `QRACK_MAX_CPU_QB` and `QRACK_MAX_PAGING_QB` are raised to max(widest + 2, 40) unless set by the user |

### 10.5 Device isolation

| Platform | Each GPU worker gets |
|---|---|
| ordinary OpenCL | `QRACK_OCL_DEFAULT_DEVICE`, `QRACK_QUNITMULTI_DEVICES`, `QRACK_QPAGER_DEVICES` = its device |
| clvk | only clvk's ICD, the Vulkan device-select layer forced to its card's PCI address, device index 0 |
| any | OpenCL initialisations serialised under a machine-wide lock, `--init-gap` (10 s) apart; worker starts `--stagger` (20 s) apart |

---

## 11. Validation

| Check | Method | Outcome |
|---|---|---|
| circuit regeneration | `verify` (§2.4) | every released circuit identical |
| conventions | selftest [1]–[10] | all pass (RESULTS §1) |
| hardware re-scoring | `hwxeb` against the release | agreement within 0.05 σ; ideal XEB within 10⁻⁵ |
| ACE on GPU | tiled n = 27 points, CPU vs CMP 50HX | equal within shot noise (RESULTS §13) |
| OpenCL fix | 16-qubit reproducer and ACE mirror points, patched build vs CPU | exact, and within 3 × 10⁻⁴ (RESULTS §12) |
| loom | op-stream identity, worker vs in-process, CZ algebra, exact-chip mirrors | cloverfield selftest [1]–[5] (RESULTS §11) |
| layoutsearch | joint model vs exact engine on the release; saved-register round trip | selftest [10]; identical survival |
| sub-registers at N = 61 | `release:61` and `nighthawk61r2` against the release | identical mirror circuits, angles and inputs |
| boundary-CZ counter | three-patch register | reproduces RESULTS §6's column |
| calculator | sheet example | 0.0019774 in both |
| swap planning | preflight on EPYC-sized memory figures; d = 4–6 with error detection on | plan printed, config tag unchanged |

---

## 12. Cloverfield

| Step per mirror cycle | Operation |
|---|---|
| 1 | the loom builds the field's global circuit by the release's recipe on the combined register (colours, K = 3 cuts of the field's graph, inputs) and splits every cycle into per-chip batches |
| 2 | single-qubit gates and twirl Paulis go to every copy of a qubit: the real one and its ghosts |
| 3 | before a CZ layer the loom queries the marginals P(1) of the qubits whose partner lives on a chip without a copy of them (CZ layers are diagonal, so one query holds for the whole layer); ghost marginals are read at the same moment for `ghost_drift` |
| 4 | a CZ whose two ends both have a copy on a chip is applied exactly there: on the link, a with ghost(b) on one chip and ghost(a) with b on the other |
| 5 | a CZ with only one end on a chip becomes a phase π · P(1) of the other end on that chip (`ghost`, `mean`), with P(1) → 1 − P(1) if the twirl put X or Y on the other end; `sample` applies Z with probability P(1) instead |
| 6 | after the inverse half each chip returns the `prob` estimate of its part of the input; the field's survival is the product |

| Element | Rule |
|---|---|
| links | facing border sites of neighbouring chips, evenly spaced, each on a CZ colour free at both ends, so a cycle applies one colour and every qubit meets at most one CZ |
| ghost room | a ghost needs a free ACE site coupled to its anchor; admitted only within the chip's width caps, or within the die split with `--gtt` |
| isolation | one worker process per chip, pinned to one device; the loom has no simulator and no OpenCL; messages are length-prefixed pickles over stdin/stdout, and Qrack's own output goes to stderr |

---

## 13. Threats to validity

| Threat | Effect | Mitigation |
|---|---|---|
| `nn_qab` θ instead of Haar | weaker rotations, concentrated outputs: XEB and survival not comparable with the device's | labelled per record; Haar runs on the same registers where the device comparison matters |
| mirror survival vs XEB | the device's headline is XEB-based; mirrors are a proxy, and for ACE they differ from forward XEB | `fxeb` where an exact reference fits |
| `prob` vs `shots` | `prob` is the expectation of the sampler's survival, not a sample | equal in expectation by construction |
| registers other than the release | different circuit graph and narrowest cut | compared together with their narrowest cut |
| few instances | instance scatter 2–5× at depth | plain means with instance errors; instance-level tables |
| PyQrack version | ACE layout and seam handling change between versions | version noted per run |
| unseeded Qrack RNG | ≈ 10⁻⁴ between processes | below every reported difference |
| float32 states | rounding of 10⁻⁶–10⁻⁵ per probability | float64 sums; well below shot noise |
| error detection off at width | lowers survival at 61 qubits | runs with error detection on use swap rather than disabling it |
| calculator scope | fitted on other circuits, sizes and ACE geometry | compared in its own variable (XEB) at 28 qubits before extrapolating |
