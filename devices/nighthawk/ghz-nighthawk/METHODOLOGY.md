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
| ACE | `QrackAceBackend(size, long_range_columns = lrc, long_range_rows = lrr, is_torus, is_gpu, is_host_pointer, …)`; all three switches passed explicitly (`is_boundary_repetition_code` off, `is_error_detection` on, `use_crossbars` on unless a flag says otherwise; PyQrack 2.29 changed its own boundary-code default to on); only non-default ones enter the config tag | logical qubit q acts on ACE site `site[q]` from the placement (§4); `u` for single-qubit gates, `cz`, Paulis by name |
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
| `--protocol nnqab` | as `nn_qab.py`: θ sin-uniform; each coupler's two-qubit gate drawn uniformly from SWAP, CZ·SWAP, CZ·iSWAP, CZ·iSWAP†, iSWAP, iSWAP†, CX, CY, CZ, anti-CX, anti-CY, anti-CZ by the angle hash (u(seed, instance, cycle, a, b, 4), control a); 2^min(13, n+2) shots; the exact reference applies the same gates. Control: exact backend XEB 1.00 ± 0.02 |
| reference | exact probabilities p of the same gates; Σp and Σp² in float64 |
| score | XEB = (N · mean_s p̂(x_s) − 1) / (N Σ p̂² − 1), p̂ = p / Σp, N = 2ⁿ |
| error | σ = std_s(N p̂(x_s) − 1) / √shots / (N Σ p̂² − 1) |
| also | linear XEB N · mean p̂ − 1, ideal XEB N Σ p̂² − 1, HOG = fraction of samples with p̂ > ln 2 / N |
| shared reference | with `--variants`, all configurations sample first, then one reference scores them all |

### 6.4 Mirror Hamming weight (`hamming`, Dan Strano's observable)

The observable of `qem/richardson_nn_hamming_weight.py`, the only one of Dan's estimators that needs no 2ⁿ reference, so the only one that reaches 61 qubits.

| Step | Rule |
|---|---|
| circuit | U = the `fxeb` forward circuit of d cycles (same seed, θ and, under `--protocol nnqab`, the same twelve-gate draw as E17), then U†: each cycle's gates inverted in reverse order (iSWAP ↔ iSWAP†; (CZ·iSWAP)⁻¹ = iSWAP† then CZ; (CZ·iSWAP†)⁻¹ = iSWAP then CZ; the other nine self-inverse), then the single-qubit gates' adjoints |
| folding | M = U U†; scale factor λ = 2k + 1 runs M (M† M)^k = (U U†)^λ: 2·d·λ layers; `--fold` lists λ |
| input, twirl | \|0…0⟩; no twirling (as his script) |
| h | mean fraction of ones = Σ_q P(q = 1) / n, read from the backend after the circuit (`prob`, through the ACE register map): the shot-free value of his Hamming-weight estimator. Shots are not drawn because ACE's `measure_shots` clones the backend per shot |
| score | signal 1 − 2h; ideal 1, fully scrambled 0 |
| error | se 0 per circuit (no shot noise); spread over instances only |
| zero-noise | `danmodel`: per (register, d) with ≥ 2 folds, Richardson on ln(signal) in u = λ^p, p = 1.60569, extrapolated to u = 0: y₀ = Σ_i w_i ln s_i, w_i = Π_{j≠i} u_j / (u_j − u_i); mitigated = e^{y₀} |
| comparison with the calculator | λ = 1 rows at the depth d of U, the depth argument of his script and of his calculator; `dan_2d` in the CSV gives the stricter pairing at the mirror's 2·d layers; folded rows get no prediction, since the calculator models unfolded circuits |
| control | exact backend: signal 1.000000 at λ = 1, 3 (selftest [11]); every gate's inverse within 1.2 × 10⁻⁷ of identity |

### 6.5 VQE on the patched register (`vqe`)

A variational energy has a reference that no simulator supplies: the ground energy, and below it nothing. Every physical state has ⟨H⟩ ≥ E₀, so an ACE energy below E₀ (or below a rigorous lower bound) shows the two patches no longer form one quantum state.

| Step | Rule |
|---|---|
| Hamiltonian | transverse-field Ising H = −J Σ_⟨uv⟩ Z_u Z_v − h Σ_q X_q on the register's couplers (the four colour matchings A–D); J = 1, h = 3.04 (≈ the square lattice's critical point, h/J = 3.044) by default |
| ansatz | Hamiltonian-variational, p layers: \|+⟩ⁿ, then per layer RZZ(γ_l) = exp(−iγ_l Z Z / 2) on every coupler, colour by colour in the RCS order A, B, C, D, then RX(β_l) on every qubit; RZZ = CX · RZ_target(γ) · CX |
| angles | optimised once, exactly, in numpy on `--vqe-opt-register` (default `nighthawk2p16`): p = 1 from a 15 × 15 grid, each further p from the previous one interpolated (Zhou et al.) and refined by BFGS; the same angles at every size, so the ladder holds the circuit family fixed and measures the simulator |
| readout | shot-free: ⟨X_q⟩ = 1 − 2 P(q = 1) after H on every qubit; ⟨Z_u Z_v⟩ = 1 − 2 P(v = 1) after CX(u → v) on all couplers of one colour (a matching); one circuit pass for X and one per colour: 5 passes per energy. On ACE a seam coupler's readout CX is a seam gate too |
| circuit reference | the same circuit's energy from a dense numpy state vector (no Qrack; qubit q = bit q), n ≤ `--vqe-exact-max` (26) |
| ground reference | E₀ by Lanczos on the same numpy H (three vectors; lowest Ritz value to 10⁻¹¹), n ≤ `--vqe-ed-max` (24) |
| bound at any n | Anderson: H = Σ_q H_q with H_q = −h X_q − (J/2) Z_q Σ_{u∼q} Z_u, so E₀ ≥ Σ_q λ_min(H_q) = −Σ_q √(h² + (J k_q / 2)²), k_q the couplers at q |
| scores | energy per site; correlation energy captured: (E_prod − E) / (E_prod − E_ref), E_prod = −h n the energy of \|+⟩ⁿ (p = 0), E_ref the exact circuit's energy (backend / circuit) or E₀ (circuit / E₀); flags `below_e0`, `below_bound` |
| record | `fidelity` holds the energy per site (the framework's generic score: finite, lower is better) |
| control | selftest [12]: exact engine vs numpy within 10⁻³ relative (Qrack is float32), p = 0 gives −h n, Lanczos vs dense diagonalisation within 10⁻⁸, bound ≤ E₀ ≤ every energy |

### 6.6 Shor's code on a two-patch register (`shor9`)

By Claude and Dan Strano: one logical qubit of Shor's [[m², 1, m]] code, the outer m-qubit phase-flip code over m inner m-qubit bit-flip blocks, encoded across an ACE seam. m = 3 (Shor-9) is the default and the original test; `--size m` scales it (Shor-16, -25, -36, -49).

| Step | Rule |
|---|---|
| code | \|0_L⟩ = (\|0…0⟩ + \|1…1⟩)^⊗m / 2^(m/2), \|1_L⟩ = (\|0…0⟩ − \|1…1⟩)^⊗m / 2^(m/2); distance m |
| encoder | input on leader a; CXs along a tree of couplings copy it to the other leaders; H on every leader; each leader copied to its block along a tree of couplings (m = 3: stars, 8 CX and 3 H as the original) |
| register | `QrackAceBackend(m², …)`, flat, two equal patches joined by one seam column of m (replicated in both patches and the crossbar; seam-to-seam couplings exact): m = 3 `long_range_columns = 1`; larger m the first (lrc, lrr) giving that split (4: 2, 3; 5: 2, 4; 6: 3, 5; 7: 3, 6) |
| layouts | `seam_leaders`: leaders the seam column, every inner block straddling the seam; `patch_blocks` (odd m): the seam column one block, the others inside one patch each, only the outer layer crossing; trees found by breadth-first search in ACE's coupling map |
| readout | logical X: Z-basis majority per block, parity over blocks; logical Z: X-basis parity per block, majority over blocks; a tied majority (even m) counts 0.5; raw: no correction (odd m the parity of all outcomes; even m the leaders' parity, or the first block's sign) |
| score | success rate per logical state over `--shots`; 0.5 = no logical information; logical-Z (X) success = mean over \|0_L⟩, \|1_L⟩ (\|+_L⟩, \|−_L⟩) |
| repeats | each state on `--repeats` fresh backends: ACE's result changes from backend to backend by more than shot noise (± 0.5 / √(2 · shots) per logical basis) |
| parallel | `--jobs`: each fresh backend (one logical state, one layout or rotation) is one task in a spawn process pool, its shots kept together; the exact reference's shots are sampled in chunks on fresh simulators of the same state. Serial and parallel runs agree within the backend-to-backend spread (Shor-9, 4 backends: X 0.875 both; `nighthawk61r2` units within 0.01) |
| control | exact `QrackSimulator`: 1.0 everywhere, raw and decoded (selftest [13], and printed with every run) |
| whole register (`--register`) | units on disjoint m × m squares of the register's sites, leaders on the middle column or row, trees along rows and columns from the middle outwards (all encoder CXs nearest-neighbour couplers of the register); CP-SAT maximises their number; all in one circuit on the register's ACE preset; states rotated over units in 4 runs; each unit sampled on its own qubits; classified by the regions of its qubits and the ACE class (exact, replica, cross) of each encoder CX |
| distribution | each unit's outcomes against the exact distribution, uniform over the outcomes the state can give (Z basis: 2^(m−1), every block all-equal with the right parity of block values; X basis: 2^(m(m−1)), every block's parity the logical sign), checked against a numpy state vector for m = 3, 4 (selftest [13]): TVD = ½ Σ \|q − p\|, its floor = mean TVD of 40 exact samples of the same size, the share of outcomes with p = 0, and the most frequent of those |
| OpenCL | `--gpu` runs ACE and the reference on the OpenCL device `QRACK_OCL_DEFAULT_DEVICE` selects: a clvk die, or PoCL's CPU device (`OCL_ICD_VENDORS` pointing at `pocl.icd`); `--ocl-sync` sets `QRACK_OCL_SYNC=1` (Qrack ≥ 10.13.2). A wrong exact reference stops the run and nothing is recorded |
| versions | every result carries PyQrack's and the Qrack library's version (`qrack`) and `boundary_rep`. ACE is PyQrack's Python code: its results follow the PyQrack version and the boundary code, not the library |
| boundary code | `--ace-boundary-rep`: `is_boundary_repetition_code=True` (default off, passed explicitly) |
| sampling width | `QrackSimulator.measure_shots` and `m_all` return bits 0–31 only (PyQrack 2.28.1): above 32 qubits the exact reference draws each shot from a clone, qubit by qubit; registers are sampled per unit |

### 6.7 Patched and full

| Family | Rule |
|---|---|
| patched | F = Π_r mean_s [(2^{n_r} p_r(x_r) − 1) / (2^{n_r} Σ p_r² − 1)], patch distributions exact and renormalised in float64 (cacheable); σ by a second-order delta method on the product of means (gradient and Hessian terms) |
| full | samples stored, not scored |

### 6.8 GHZ Layer 1a, Layer 2 and the seam layers on a register (`nighthawk_ghz.py`)

The formal proof's calibration and refocusing layers (its Definitions 2–6, Lemmas 1, 3, 4, Theorem 1; numpy reference `shrs_nighthawk_stealth.py`) on a register's ACE preset placement. Gates exact; departures from the paper's model: ACE's seam approximation and sampler, and no global depolarising channel (A2).

| Step | Rule |
|---|---|
| units | `chip`: one GHZ over the register, H on a root, CX along a spanning tree (bfs from the centre; seam: Kruskal with in-simulator couplers first, fewest seam CXs = seam qubits + 1). `tiles`: k-qubit chains along lattice rows and columns, root in the middle, every orientation and offset; with Layer 2 a line's chains are spread into groups with no coupler between two chains |
| sublattices | A = {r + c even}, B = {r + c odd} of the device lattice; every coupler A–B is checked (Lemma 1) |
| drift | H_D = Σ J_ij Z_i Z_j + Σ (δω_i + ε_i) Z_i, η = 0; J_ij = 0.065 × U(0.75, 1.25) per coupler from `default_rng(seed)`; δω_q = 0.022 ((q mod 3) − 1); ε ~ N(0, 0.015²) per realisation from `default_rng([seed, 7919, r])`, shared by ACE and the exact reference |
| gates | inside a unit e^{−iφ Z_i Z_j} = CX_ij · u(0, 0, 2φ)_j · CX_ij, or with `--inside-zz stabilizer` not applied (Z_iZ_j ψ = ±ψ for the unit's GHZ and its Walsh-flipped images, so the gate is a global phase; the seam-crossing count of the drift is then 0); to an outside qubit e^{−iφ z_j Z_i} on the unit qubit, z_j its tracked Z eigenvalue (the same unitary: selftest [7]); between outside qubits a global phase, omitted |
| windows | `--blocks` L after the preparation, each of length T: `unmitigated` e^{−iTH}; `compensated` e^{+iTĤ} e^{−iTH} with Ĥ the true J, δω (oracle Layer 1b, Proposition 1 leaves ε); `dd` U(T/4) X_B U(T/4) X_A U(T/4) X_B U(T/4) X_A (Definition 5) |
| readout | `z`; MQC `qm`: u(π/2, 0, π − φ_m) = H Rz(−φ_m) on every unit qubit, φ_m = πm/(k + 1), m = 0 … 2k + 1 (Lemma 4); witness `xrA`: I, S†, Z, S on the root, H on all |
| estimators | ĉ = (1 / (2k + 2)) Σ_m ⟨M(φ_m)⟩ e^{−ikφ_m}; C_parity = 2\|ĉ\|, Phi_coh = arg ĉ, F_GHZ = (P0 + P1 + C)/2, F0 = (P0 + P1)/2 + Re ĉ; λ per block = [(P0 + P1 − 2/d)/(1 − 2/d)]^{1/L} (Lemma 3, L = 1 without windows); Σδω estimate = −Phi/(2LT) (Corollary 4.1); witness F ≥ P0 + P1 + (1 + ⟨X^⊗k⟩)/2 − 1 |
| realisations | `--repeats` K: a fresh ACE backend and ε draw each; shots pooled over realisations (the average of ρ over μ), per-realisation values kept as `*_runs` |
| control | exact `QrackSimulator` per realisation: Z shots all-equal per unit always (diagonal drift keeps populations); deterministic parities (`q0`, `q(k+1)`, `xr0`, `xr2`) ideal on every shot when the ideal window is I (no Layer 2, `dd`, `compensated` with σ = 0). Otherwise its numbers are recorded as `exact_*`, the clean-physics baseline |
| patches | stabilizer hybrid only for `chip` with Clifford readout and no Layer 2; dense otherwise. Exact reference shots: per-shot clones measured qubit by qubit on a tableau (`measure_shots` would densify), `measure_shots` dense up to 32 qubits |
| checks | selftest [1]–[16]: Lemma 4 and the witness against numpy ρ, Lemma 4 on 20 random ρ (\|ĉ − c\| < 3·10⁻⁶), Lemma 3, readout gates vs numpy on Qrack (≤ 1·10⁻⁷), Theorem 1 on a 2 × 3 grid with random J, δω, ε and state (\|⟨ψ\|W_T\|ψ⟩\|² = 1 ± 1·10⁻⁷), Corollaries 2.1 and 4.1 (arg c = −2LTΣδω to 10⁻⁶), the outside-coupling identity, fold = I, the ZNE amplification = I (zz and cx, K = 2, random state), both ZNE models on synthetic decays, Layer S's seam classes, plan and the code's schedule on a stub register, the co-located authority step on a stub backend, `--inside-zz stabilizer` against the gates on a 3-qubit GHZ unit in a 2 × 3 grid after two windows of each mode (\|⟨ψ_gate\|ψ_stab⟩\|² = 1 ± 1·10⁻⁶) and, as a control, on a product state (< 0.999); `probe`'s CX target on the seam or bulk qubit against the gates on a product state (1 ± 1·10⁻⁶); `probe`'s chain re-rooting of [11, 10, 12]; `LoomGates`' u, zrot, s, sdg, h, x, z against Qrack's on an entangled 2-qubit state (1 − fid < 10⁻⁶); with the cloverfield and the release, a 3-chain across a link of two rect:3×3 chips: with ghosts class replica, F_GHZ > 0.95 inside the ghost band and < 0.55 from the real copies; as a hard cut class cut and < 0.55 even when the authority read is asked for (1024 shots); with QrackBridge, two GHZ units [2, 1 \| 3], [0 \| 4, 5] on a 6-qubit bridge of two patches, F_GHZ from `out_ket`: max_bonds 0 both < 0.51, 1 one < 0.51 and one > 0.99, 2 both > 0.99 |
| bridge engine (`stack --engine bridge`) | the loom engine's field, links, cells and tiling, with the chips joined by PyQrack's `QrackBridge` (or `qrack_bridge.py` from `--qrack-bridge`, loaded as a PyQrack submodule) instead of the loom. Per circuit one bridge of P patches (P = chips): the circuit's unit qubits, chip c's at indices c·m … c·m + (its count − 1), m the largest per-chip count (padding \|0⟩ qubits, never touched), so the bridge's equal contiguous split puts each chip in its own patch. Gates through the bridge's API (u, h, x, s, adjs, mcx; Z = u(0, 0, π), zrot(φ) = u(0, 0, 2φ)); `max_bonds`, `truncation`, `lazy_bonds` from the options; the bridge's rng seeded per (realisation, setting, group). Readout: `measure_shots` over the unit qubits (exact sampling of the stored state). Classes: a unit on two chips is class bridge, planned control (no code; C reruns it). Logged per pass: peak bonds, truncating circuits, `fidelity_estimate` (mean, min). The exact reference: the field circuit on one `QrackSimulator` |
| loom engine (`stack --engine loom`) | the field: `nighthawk_cloverfield.Field` of the `--loom-chips` side by side (1 × K; chip c + 1 placed right of chip c, its columns shifted to follow), links on the facing border pairs of `--loom-link-rows` (default: the rows whose pair are lattice neighbours after the shift), a CZ colour free at both ends (asked-for links without one: refused). Per circuit, one chip engine per chip: exact (`ChipSim`, `QrackSimulator`, the chip's qubits + one ghost per partner it is linked to) or, with `--loom-chip-backend ace`, nighthawk_qrack's `AceEngine` on the register's stored preset with the boundary repetition code on for the code passes and off otherwise (the same sites), ghosts on free ACE sites coupled to their anchors (`place_ghosts`, deterministic), the authority rule installed per worker; `--loom-read cut` (default): no ghosts and the cloverfield's `Loom` in mean mode (a link CZ is the phase π P(1) of the other side's real copy, on each side), copies = real qubits only; `--loom-read ghost`: the cloverfield's `Loom` in ghost mode: a one-qubit gate on every copy (as a 2 × 2 matrix, op `m`), a CX as H · CZ · H on the target, the CZ woven by the loom (exact on a chip holding both copies, the mean-field phase π P(1) of the absent qubit's real copy elsewhere). Drift, Walsh pulses and readout rotations as on ACE, on every copy. Reading: per unit, `real` (each qubit's own copy) or `authority` (the chip holding copies of all the unit's qubits, the most real copies first; real where none). A unit's copies on one chip are sampled together (`measure_shots`, ≤ 32 bits), each sample shuffled with its own seeded permutation, chips paired shot by shot. Classes: sids = the (chip, simulator) pairs holding a copy (exact chip: one simulator; ACE chip: the simulators of the copy's ACE site), then as on ACE. Layer S's code = `authority` and, on ACE chips, the boundary repetition code; without it `real`, no code. Samples: a chip's read copies in calls of up to 32 qubits. The exact reference: the same field circuit on one `QrackSimulator` over all qubits |
| `sweep`, `probe` | `sweep`: `tiles` per window count L as above (L = 0: no Layer 2), units classed by `seam_class`, means per (boundary code or not, `inside_zz`, L, class). `probe`: the chosen class's units of every spread tiling, run per variant through `run_tasks` (fresh backend and ε per realisation, the same draws for every variant); `tseam` / `tbulk` swap a coupling's CX control and target so the target is the seam / bulk qubit (Z_aZ_b is symmetric: the same unitary); `chain-*` prepare H on the lower-index end and CX along the chain; scored as above per unit, the exact reference where asked |

#### Seam layers S and V (`stack`)

| Step | Rule |
|---|---|
| passes 1, 2 | the same units (Layer 2's spread groups in both passes), pass 1 without windows, pass 2 in the chosen mode; scored as above, ACE and exact |
| seam crossings n | preparation: CXs of class replica or cross; drift: 2 × (couplings inside the unit through a seam) × segments per window (unmitigated 1, compensated 2, dd 4) × L |
| extraction, fits | per unit with n > 0 and pass p: kept_pop = (P0 + P1)_ACE / (P0 + P1)_exact; kept_coh = C_ACE / C_exact (MQC) or ⟨X…X⟩_ACE / ⟨X…X⟩_exact (witness), exact values above 0.5 only. Least squares log(kept) = a + b·n over all such (unit, pass): first-crossing share e^a, per-crossing share e^b. Per seam coupler: the mean Z break rate over the units holding it, per pass |
| extraction, syndrome | per unit and pass: kept_pop, kept_coh as above; loss = F_exact − F_ACE and loss / se(F_ACE); P1 − P0 (ACE); the unit qubit with the highest rate of leaving the unit's Z majority (ties: the lower index) and whether it is a seam qubit |
| seam class | hosts = ∩ over the unit's qubits of the ACE simulators holding them (a seam qubit: every simulator with one of its replicas). bulk: hosts ≠ ∅, no seam qubit; replica: hosts ≠ ∅ with a seam qubit; cut: hosts = ∅ |
| plan | replica with loss > 0.02 and loss / se > 3 in pass 1 or 2: code; replica otherwise: clean (not rerun); cut: control; bulk: none (not rerun) |
| correction | code units: `is_boundary_repetition_code` on (ACE's authority decoder; its per-coupling error-detection gadget is off with it, as ACE does), and with `--code-schedule on` the code's schedule: the tree's CXs reordered, from every prepared qubit first into children held by one simulator, then into seam children, ties by the parent's position; the same tree, so the same state. Authority rule (`--code-authority`, installed on `QrackAceBackend._update_lineage` in every worker before its backend is built): `lineage` = PyQrack 2.29.5's; `colocated` = after it, for each participant with a recorded authority, every boundary qubit it has coupled with (both multi-replica) since the backend was built gets the replica slots in the same simulators as its authority, where it has any. The register is placed identically with and without the code (checked: qubit map, sites, simulators) |
| order (`--order`) | `echo-code`: pass 2, Layer S, Layer V as in these rows. `code-echo`: Layer S's syndrome, classes and plan from pass 1 alone (the fits over pass 1's crossings); pass C: code and control units rerun pass 1's circuit (no windows), realisations 1000 … 1000 + K − 1, ACE only, code units grouped apart from the rest, the code as below; bulk and clean units keep pass 1. Pass E: every unit, Layer 2 in the chosen mode, realisations 0 … K − 1 (pass 2's ε draws), code units under the code with its schedule, the rest without; ACE and exact |
| Layer V | code and control units rerun pass 2's circuit (Layer 2 mode, L, the model), realisations 1000 … 1000 + K − 1 (fresh ε, fresh backends), ACE only, code units grouped apart from control units in each spread group; compared per unit with passes 1, 2 and the exact pass-2 reference. V − pass 2 of the code units is reported against \|V − pass 2\| of the control units, the rerun spread |
| amplification (`--zne-k`) | `AmpGates`: every CX of class replica or cross, in the preparation and the drift, followed by K logical identities. `zz`: CX_ct Rz_t(φ) CX_ct · CX_tc Rz_c(−φ) CX_tc (four crossings, a rotation between each pair); `cx`: CX CX. λ = 1 + 4K (zz) or 1 + 2K (cx): every seam crossing of the circuit scaled alike, nothing else touched |
| ZNE (`--zne-k`) | at λ = 1 (pass 2) and each 1 + 4K: P0 + P1, Re c, Im c (MQC) or P0 + P1, ⟨X…X⟩ (witness), the observables linear in ρ. Linear: a weighted least-squares line per observable (weights 1/se²), read at λ = 0, F from the extrapolated observables. Exponential: weighted lines through log((P0 + P1 − 2/d)/(1 − 2/d)) and log\|c\| (or log⟨X…X⟩), read at 0; undefined where a value is ≤ 0. Flag: P0 + P1 > 1 or \|c\| > (P0 + P1)/2 by more than 2 se |

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
| E16 | why do Dan's calculator and our numbers differ? | his `nn_qab.py` circuits on his strips (10–26 qubits, his lrc/lrr, torus); our release prefixes 16–28 with a 3-qubit seam | XEB, twirled mirror (exact Clifford conjugation), cut-sampler XEB; his gates and CZ only | 12, 20 | 20 circuits (8 at 26); 3 instances / 3 × 10 runs | container CPU | 14 |
| E17 | do the Nighthawk ladder's numbers conform to Dan's in his own protocol? | `nighthawk2p16`, `2p20`, `2p24`, `2p28` | `fxeb`, `--protocol nnqab` | 12, 16, 20 | 10 circuits (28: 3) × 2^min(13, n+2) shots | container CPU | 10 |
| E18 | Dan's circuits past the XEB ceiling, up to 61 qubits | `nighthawk2p16` … `2p57`, `nighthawk61r2` | `hamming`, `--protocol nnqab`, λ = 1, 3, 5 | 4–12 cycles (8–120 layers) | 6 circuits (container, 16–36; 4 at 40); EPYC recipe: 10 (≤ 57), 3 (61) | container CPU; EPYC for 45–61 | 10 |
| E19 | does a variational energy survive the seam, against references no simulator supplies? | `nighthawk2p16` … `2p57`, `nighthawk61r2` | `vqe`, transverse-field Ising, h = 3.04, angles from `nighthawk2p16` | p = 1–4 (6) layers | 1 deterministic circuit per point | container CPU (16–28); EPYC | 10.2 |
| E20 | which logical information survives an ACE seam? | m × m ACE grids (m = 3–6), two patches, one m-qubit seam column; units tiled over `nighthawk61r2` and `fav5p112` | `shor9`, Shor's code m = 3–6 (`--size`), `seam_leaders` and `patch_blocks`; `--register` | encoder only | 512–8192 shots × 4 logical states × 1–6 fresh backends | container CPU | 10.3 |
| E21 | does a GHZ state survive the seam (Layer 1a)? | `nighthawk2p16`, `2p28`, `2p57`, `nighthawk61r2` | `nighthawk_ghz.py` `tiles` (3-qubit chains, MQC) and `chip` (witness / MQC); `fold:1` | — | 256–512 shots × 2–4 backends per setting | container CPU | 10.4 |
| E22 | does Layer 2 (Walsh echo, compensation) refocus across the seam? | `nighthawk2p16`, `2p28`, `nighthawk61r2` | `tiles`, `chip`; `--layer2` unmitigated, compensated, dd; the §8 model | L = 3 windows | 256 shots × K = 4 realisations | container CPU | 10.5 |
| E23 | can the seam's error be extracted from passes 1–2 and removed by ZNE? | `nighthawk2p28`, `nighthawk61r2` (tiles), `nighthawk2p16` (chip) | `stack`: Layer S (extraction, ZNE at λ = 1, 5, 9), Layer V; amplifier check `fold`, `zzfold`, `gfold` | L = 3 windows (dd) | 256 shots × K = 4 per setting and λ | container CPU | 10.6 |
| E24 | can a code correct the seam's error that passes 1–2 show, and which part? | `nighthawk2p28`, `nighthawk61r2` (tiles), `nighthawk2p16` (chip) | `stack`: Layer S (syndrome, seam class, replica repetition code + its schedule), Layer V (code units corrected, cut units as the control); tomography of two `nighthawk2p28` units | L = 3 windows (dd) | 256 shots × K = 4 per setting; tomography 27 Pauli settings × 2048 | container CPU | 10.7 |
| E25 | what does PyQrack's current ACE source suggest, and does the co-located authority rule make the code's schedule unnecessary? | `nighthawk2p28`, `nighthawk61r2` (tiles); seam-only pairs; the error-detection gadget on / off | `stack` with `--code-authority lineage / colocated` × `--code-schedule on / off`; in-process probes of `_update_lineage` | L = 3 windows (dd) | 256 shots × K = 4 per setting | container CPU | 10.8 |
| E26 | does the order of Layer 2 and Layer S matter: the code before the echo? | `nighthawk61r2` (tiles) | `stack --order code-echo` (lineage authority, the code's schedule) | L = 3 windows (dd) | 256 shots × K = 4 per setting | container CPU | 10.9 |
| E27 | how does the GHZ fidelity follow the number of Walsh-echo windows, per seam class? | `nighthawk2p28` (tiles) | `sweep`: `tiles`, L = 0, 1, 2, 3, 4, 5, 6, 8 (dd), L = 8 again under the boundary code | L as listed, T = 4.4 | 256 shots × K = 4 per setting, ACE only | container CPU | 10.10 |
| E28 | can the cut units be protected from the drop under Layer 2? | `nighthawk61r2`, `nighthawk2p28` (tiles) | `probe` on the cut units (8 variants of preparation and inside ZZ); `stack --inside-zz stabilizer` | L = 3 windows (dd) | 256 shots × K = 4 per setting; exact 128 | container CPU | 10.11 |
| E29 | the loom as the seam: two exact chips of the patches' size, the code as the loom's authority read | `nighthawk61r2`'s qubits as `rect:1-8,2-5` + `rect:1-8,6-9`, 7 links (tiles) | `stack --engine loom --order code-echo`, `--inside-zz gate / stabilizer` | L = 3 windows (dd) | 256 shots × K = 4 per setting; exact 128 | container CPU | 10.12 |
| E30 | the loom between two-patch ACE chips: both seams in one field | two `nighthawk61r2` chips (122 qubits, 7 links on rows 1–5, 7, 8) (tiles) | `stack --engine loom --loom-chip-backend ace --order code-echo`, `--inside-zz gate / stabilizer`; lineage authority with the code's schedule | L = 3 windows (dd) | 256 shots × K = 4 per setting; exact 128 | container CPU | 10.13 |
| E31 | the loom as a hard cut, and what the ghost band does | `nighthawk61r2`'s qubits as two exact chips (61), two `nighthawk61r2` ACE chips (122) (tiles) | `stack --engine loom --loom-read cut --order code-echo`, `--inside-zz gate / stabilizer`; `--loom-read ghost` with 5-qubit chains | L = 3 windows (dd) | 256 shots × K = 4 per setting (5-chains: 128 × 2); exact 128 / 64 | container CPU | 10.14 |
| E32 | QrackBridge in place of the loom: its capacity against GHZ fidelity | `nighthawk61r2`'s qubits as two exact chips (61), 7 links (tiles) | `stack --engine bridge --order code-echo`, `--bridge-max-bonds` 6, 2, 1, 0 (`--inside-zz gate`), 1 (`stabilizer`); greedy, eager | L = 3 windows (dd) | 256 shots × K = 4 per setting; exact 128 | container CPU, PyQrack 2.29.5 + `qrack_bridge.py` of PyQrack 2.30.2 | 10.15 |

### 9.1 E16 cross-check procedure

| Step | Procedure (scripts in `dan_crosscheck/`) |
|---|---|
| his circuits | `nn_qab.py`'s construction verbatim (`factor_width`, sequence 0 3 2 1 2 1 0 3, torus wrap, random gate from the 12, θ = arcsin(u/π)), seeded per circuit (`random.Random(1000 · seed + c)`) |
| his XEB | his ACE call (`QrackAceBackend(width, lrc, lrr)`, torus default), 2^min(13, n+2) shots, exact reference, his estimator |
| twirled mirror for Clifford gates | P_a ⊗ P_b before each gate G, then G (P_a ⊗ P_b) G† after, found by numerical conjugation of the 16 Pauli pairs; inverse half with the inverse gates; survival by the `prob` chain rule; checked: survival 1.000 on the exact simulator for both gate sets |
| CZ-only arm | the same circuits with every two-qubit gate replaced by CZ |
| cut sampler | ideal probabilities p; qubits split by the patch that owns them in ACE (seam qubits to their first patch); q = product of the two exact marginals; XEB = (N Σ q p − 1)/(N Σ p² − 1), exact, no sampling |
| our side | `release:N` registers with `--min-boundary 3`, `run --families fxeb` and `--families mirror --mirror-estimator prob`, and the cut sampler on the same forward circuits |

### 9.2 Design rules applied to every experiment

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
| conventions | selftest [1]–[13] | all pass (RESULTS §1) |
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
