# HOWTO: `rcs/nighthawk/nighthawk_qrack.py`

**A clean-qubit PyQrack companion to arXiv:2609.28657 (random-circuit sampling on IBM Nighthawk r2), with an exact-reference study of seam-partitioned approximate simulation**

| | |
|---|---|
| Script | [`rcs/nighthawk/nighthawk_qrack.py`](https://github.com/twobombs/thereminq-examples/tree/main/rcs/nighthawk) (ThereminQ examples) |
| Analysis | `rcs/nighthawk/nighthawk_graph.py` — harvest, comparison views, CSV export |
| Companion data | [`BlueQubitDev/rcs-nighthawk`](https://github.com/BlueQubitDev/rcs-nighthawk) |
| Paper under study | Sedrakyan et al., [arXiv:2609.28657](https://arxiv.org/abs/2609.28657) (preprint, v1) [[1]](#ref-1) |
| Simulator | [Qrack / PyQrack](https://github.com/unitaryfund/pyqrack) — `QrackSimulator`, `QrackAceBackend` [[10]](#ref-10) |
| Acceleration | OpenCL or CPU. **No CUDA.** |
| Companion | [`rcs/nn_qab/`](https://github.com/twobombs/thereminq-examples/tree/main/rcs/nn_qab) — ACE bulk-to-boundary sweeps on `nn_qab`'s torus circuits, same estimator |

---

## Abstract

We provide a reproducible, simulator-side companion to the 61-qubit random-circuit sampling (RCS) experiment of Ref. [[1]](#ref-1). The released circuits are regenerated bit-for-bit, the hardware bitstrings are re-scored against PyQrack ideal distributions, and the circuits are emulated on ideal qubits with an exact backend (a pipeline control) and with Qrack's approximate, seam-partitioned `QrackAceBackend` (ACE). For first-n truncations of the register (n ≤ 34) every ACE sample is scored by forward linear XEB against an exact 2ⁿ-amplitude reference. On the 8×8 Nighthawk window we find that ACE's native chunk numbering places most couplers across simulators, and that re-placing logical qubits onto ACE sites as compact tiles raises the forward XEB at n = 27, d = 8 from 0.008 to 0.32 at unchanged memory. Across ten layout/placement configurations the XEB is predicted best by the number of couplers across simulators and of simulators in use (rank correlations −0.80 and −0.84), and, among crossing-free tilings, by the number of logical qubits on seam sites. Even so, at depths where the output is close to Porter–Thomas (d ≥ 12), the tiled emulator's per-cycle decay at n = 27–29 (b ≈ 0.24–0.31) exceeds the device's b = 0.137 at n = 61, and grows with depth. Under the weaker single-qubit ensemble of the `nn_qab` benchmark (sin θ uniform), tiled ACE on the Nighthawk window reaches a mean XEB of 0.435 ± 0.037 at n = 27, d = 12 over 20 circuits, against 0.153 ± 0.014 with the device placement and 0.111–0.241 for `nn_qab`'s own register geometries (Sec. 9.7). At 61 qubits no XEB can be scored directly; the release's own contraction-cost estimates put one exact amplitude of the 36-cycle circuit at ≈ 10²² multiply-adds (Sec. 8.6). A shared-reference comparison of two tiling heuristics confirms the seam-qubit reading at n = 27–29: where the improved tiling halves the number of logical qubits on seam sites (n = 28, 8 → 4) the d = 12 XEB rises 3.6-fold, where the seam counts agree (n = 27) the two coincide (Sec. 9.5). Tiled ACE runs on GPU through NVIDIA's OpenCL stack and reproduces the CPU path (Sec. 8.3). None of ACE's built-in settings — boundary repetition code, seam error detection, crossbars — lowers the per-cycle cost of a seam; disabling error detection raises the XEB by a constant offset of ≈ 0.01 (Sec. 9.5). At the full 61 qubits the paper's own mirror proxy now runs on ACE with a CP-SAT placement on a three-patch register (9 boundary qubits): under the `nn_qab` single-qubit ensemble its survival lies above `ibm_phoenix` at d = 4–6, at 0.3–0.4× the device at d = 8–12, and 160× below it at d = 20; the losses follow the cumulative number of CZs through a boundary, not depth (≈ 0.26 nats per boundary CZ past d = 12; Sec. 9.8). Sub-registers of the Nighthawk lattice up to 88 qubits fit one 8 GB GPU as four patches (`--register`, Sec. 9.10), and placing disabled sites on the seams cuts their boundary from 17 to 11 qubits. Two host-memory lifetime defects in Qrack's OpenCL engine, which corrupted or crashed deep ACE points on GPU, were traced and fixed without serialising dispatch (Sec. 8.7).

---

## 1. Purpose and scope

Ref. [[1]](#ref-1) reports forward RCS on 61 qubits of the 120-qubit IBM Nighthawk r2 processor (`ibm_phoenix`), with native CZ gates on a square lattice, and estimates the circuit fidelity at 36 cycles from two proxy families: *patched* circuits scored with a normalised linear cross-entropy benchmark (XEB), and *mirror* circuits scored by echo survival. BlueQubit released the circuits, layout, partitions and measured bitstrings as `rcs-nighthawk`.

`nighthawk_qrack.py` does the following against that release, without a QPU:

1. **Regenerates** every released circuit bit-for-bit from `data/layout.json` and checks it against the released OpenQASM (`verify`).
2. **Re-scores** the measured `ibm_phoenix` bitstrings with PyQrack ideal patch distributions and rebuilds the paper's F(d) points, exponential fit and collision ratios (`hwxeb`).
3. **Emulates** the released circuits on ideal ("clean") qubits and scores them with the paper's estimators (`run --families mirror,patched,full`). With `--backend ace` the only error left is shot noise plus ACE's seam approximation; with `--backend exact` the result must give F = 1.
4. **Measures ACE directly** (`run --families fxeb`): the unpatched forward circuit on the first n logical qubits, sampled by ACE and scored by forward linear XEB against an exact reference. This is the only family in which the approximation is measured against ground truth rather than through a proxy.
5. **Chooses and places the ACE register** (`aceplan`, `--ace-max-width`, `--ace-tiling`): every distinct ACE layout is enumerated, and logical qubits can be re-placed onto ACE sites to keep couplers inside one simulator.
6. Runs a **data-only** analysis of the release (`seamgap`).
7. Runs the mirror family on **larger sub-registers** of the Nighthawk lattice built with the release's own recipe (`--register`, up to 88 qubits on one 8 GB GPU).

There is **no noise model**. Any deviation from F = 1 under `run` is attributable to the simulator backend or to sampling, not to an assumed device error.

> **Version note.** This HOWTO describes `nighthawk_qrack.py` at 3964 lines and `nighthawk_graph.py` at 1329 lines (13 October 2026). Earlier revisions contained a noise-model audit (`--preset r2`, `--stochastic`, `--edge-layers`); those flags no longer exist. Tiling version 2 (`--ace-tiling-version 2`) reproduces the config tags of runs made before tiling version 3 became the default.

---

## 2. Background

### 2.1 Random-circuit sampling and linear XEB

Each cycle applies a Haar-random single-qubit gate to every qubit, then a layer of CZ gates on one of four coupler colourings `A, B, C, D` in fixed order. The linear XEB was introduced as a fidelity proxy for such circuits by Boixo et al. [[2]](#ref-2) and used for the 53-qubit Sycamore experiment [[3]](#ref-3). Its relationship to the true fidelity, and the regimes in which it is and is not a reliable estimator, are analysed in [[4]](#ref-4) and [[5]](#ref-5).

### 2.2 Patched circuits (Eq. (1)–(2) of Ref. [1])

A patched circuit removes every CZ that crosses a partition boundary, so the 61-qubit register factorises into K independent patches whose ideal output distributions p_r are exactly computable. The estimator is the product over patches of the normalised patch XEB:

```
F_patched = Π_r  mean_s [ (2^{n_r} · p_r(x_r^{(s)}) − 1) / (2^{n_r} · Σ_x p_r(x)² − 1) ]
```

where x_r^{(s)} is the restriction of shot s to patch r. The standard error uses a second-order delta method on the product of means (`_delta_se`). Patch circuits as a fidelity proxy trace back to the "patch" and "elided" circuits of [[3]](#ref-3).

### 2.3 Mirror circuits

A mirror circuit runs U for d/2 cycles and then U† for d/2 cycles, so the ideal output is the input string; the fidelity proxy is the survival probability. This is the circuit mirroring of Proctor et al. [[6]](#ref-6), developed into scalable randomized benchmarking in [[7]](#ref-7) and extended to universal gate sets in [[8]](#ref-8). In the release the forward half is **pseudo-patched**: at each cycle the CZs crossing *one* of the five K=3 boundary sets are removed, rotating to the next partition every cycle (`rotate_every=1`); `mirror_cycles` reproduces this exactly.

### 2.4 Pauli-frame twirling

Optional gate twirling surrounds each CZ with random Paulis P_a ⊗ P_b before and the propagated correction `(P_a Z^[P_b∈{X,Y}]) ⊗ (P_b Z^[P_a∈{X,Y}])` after, an exact identity for ideal gates (`cz_layer`); this is the randomized-compiling construction of Wallman and Emerson [[9]](#ref-9). On ACE its role is to randomise the *coherent* seam error so that the forward and inverse halves of a mirror cannot cancel it. On the exact backend it is disabled.

### 2.5 Forward XEB against an exact reference (`fxeb`)

For first-n truncations small enough to hold exactly, the forward circuit is sampled by the backend under test and every sample is scored against the exact output distribution p of the same gates:

```
XEB = (N · mean_s p(x^{(s)}) − 1) / (N · Σ_x p(x)² − 1),     N = 2^n
```

i.e. Eq. (1) with the whole register as a single patch. The denominator is the ideal XEB (collision ratio): ≈ 1 for Porter–Thomas outputs, larger for concentrated ones. No twirling is applied, so a coherent approximation keeps whatever XEB it earns. The reference is a dense `QrackSimulator`; its probabilities are read into one float buffer (`out_probs_np`) and every sum over them is taken in float64, so float32 storage does not bias the estimator (Sec. 8.4).

### 2.6 ACE: seams, simulators and placement

`QrackAceBackend` partitions a rectangular register into bulk *chunks*, each held exactly by its own simulator, separated by *seam* sites whose qubits are replicated in the adjacent simulators (plus, on builds with crossbars, a shared boundary simulator). `long_range_columns` / `long_range_rows` (`--lrc`, `--lrr`) and `is_torus` set the layout. A CZ falls into one of three classes:

| Class | Definition | Treatment in ACE |
|---|---|---|
| *exact* | both ends are bulk qubits of the same simulator | applied exactly |
| *replica* | a seam qubit is involved, but the ends share a simulator | applied on the replica, reconciled at the seam |
| *cross* | the ends share no simulator | approximated across simulators |

Two properties of the Nighthawk window matter here. First, a simulator's width is its bulk **plus every seam replica it carries plus one**: a two-chunk split of the 8×8 register with one seam column yields two 37-qubit simulators, not two 32-qubit halves, and is out of reach of a dense state on a 310 GiB host. Second, ACE numbers its chunks along a folded one-dimensional chain, so on the device grid a chunk is a set of row strips; with the device-geometry placement most vertical couplers become *cross* couplers even where no seam separates them. For the default `--lrc 4 --lrr 4` at n = 61 the split is 33 exact / 30 replica / 39 cross out of 102 couplers.

**Tiling** (`--ace-tiling`) keeps ACE's layout and memory unchanged and instead searches, by deterministic simulated annealing, for the placement of logical qubits onto ACE sites that minimises a weighted coupler/site cost. For the same 4/4 layout at n = 61, tiling version 2 reaches 61 / 41 / 0. Couplers are then no longer ACE-grid neighbours; ACE applies them exactly inside a simulator and through its seam machinery otherwise.

| Version | Cost | Search |
|---|---|---|
| 2 | 10 × cross + 1 × replica | one annealing run from the device placement |
| 3 (default) | 20 × cross + 1 × replica + 6 × (logical qubits on seam sites) + 2 × (simulators holding bulk qubits) | three restarts: one from the device placement, two from the best (n−1)-qubit tiling extended by one qubit; results cached in `nighthawk_tiles.json` |

The version-3 weights follow the measurements of Sec. 9.3–9.4. A trial weighting that favoured fewer simulators over fewer seam qubits placed all 18 qubits of an n = 18 register in one simulator using four seam sites, and lowered the d = 12 XEB from 0.48 to 0.06 (single instance, 1000 shots); the seam term was raised accordingly.

### 2.7 Relation to approximate-simulation baselines

The quantity compared throughout is a per-cycle decay b = −d ln F / dd. A classical method that loses fidelity at a controlled rate per gate, and is matched against a device on that rate, is the framing of Zhou, Stoudenmire and Waintal for truncated matrix-product-state simulation [[11]](#ref-11). The exact alternative to ACE's approximate seams — Schrödinger–Feynman path summation across a cut, with simulation cost scaling linearly in the fraction of paths kept and hence in the target fidelity — is that of Markov et al. [[12]](#ref-12). On this lattice roughly two CZs cross the best bipartition per cycle, so the number of paths grows as about 2^(2d); this is used only as a shallow-depth reference, not at d = 36.

### 2.8 Bespoke registers: `--ace-transpile`

Inside a patch the bulk is a dense state vector, so every pair of its qubits is a coupler; only the boundary has geometry. Following `rcs/mirror_nighthawk.py` in [`vm6502q/pyqrack-examples`](https://github.com/vm6502q/pyqrack-examples), `--ace-transpile` takes the coupling map of an ACE register of any size (`get_logical_coupling_map()`, or the pairs sharing a simulator), removes chosen sites (`--ace-exclude`, to shape the boundary), and places the circuit's qubits on it. The placement is routing-free — every circuit coupler is a coupler of the register — so the released gates, Pauli twirling, `--variants` and the estimators apply unchanged; routing would insert SWAPs and change the circuit, and is not done. For these circuits (CZ between fresh random single-qubit gates) Qiskit's gate-level optimisations have nothing to cancel; their gain applies to circuits with SWAP-type gates such as the `nn_qab` set.

Three placement engines are provided. `exact` (default; needs OR-Tools) solves the placement with CP-SAT: inside a patch every site is equivalent, so the sites are grouped by the simulators that hold them (one patch's bulk, or the seam between two patches), and each logical qubit is given a group within the group capacities such that every circuit coupler lands on a pair of groups the register couples (ACE's own coupling map, which couples a seam site to both its patches and to seam sites of the same boundary). The objective is lexicographic: fewest qubits on seam sites (the smallest boundaries), then the most even patch widths, then the fewest circuit couplers touching a seam. `anneal` applies the tiling cost of Sec. 2.6 to the same register and exclusions, starting from the VF2 layout when Qiskit finds one; it is not guaranteed optimal. `vf2` is Qiskit's VF2Layout with a target that charges couplers and qubits on seam sites; on ACE's dense coupling graph its search must be bounded and it ranks only the layouts it reaches. Measured placements of the 61-qubit circuit:

| Register (flat) | Engine | exact / replica / cross CZ | qubits on seams | active dense memory |
|---|---|---|---|---|
| 64 sites, lrc 4 lrr 4 | vf2 | no routing-free layout within the search bound | — | — |
| 64 sites, lrc 4 lrr 4 | anneal | 63 / 38 / 1 | 12 | 0.05 GiB |
| 64 sites, lrc 4 lrr 7, error detection off | anneal | 84 / 18 / 0 | 6 | 192 GiB |
| 75 sites, lrc 4 lrr 5 (`mirror_nighthawk.py`) | anneal | 74 / 28 / 0 | 9 | 0.56 GiB |
| 75 sites, lrc 4 lrr 5, its first 7 exclusions | anneal, `--transpile-allow-cross` | 76 / 24 / 2 | 8 | 0.66 GiB |
| 75 sites, lrc 4 lrr 5, its current 11 exclusions (64 sites left) | anneal, `--transpile-allow-cross` | 76 / 25 / 1 | 8 | 0.44 GiB |
| 75 sites, lrc 4 lrr 5, its current 11 exclusions | anneal | refused: the annealer ends with one coupler (34–41) joining two patches with no seam | — | — |
| 75 sites, lrc 4 lrr 5, its 11 exclusions with seam site 69 reopened | anneal | 72 / 30 / 0 | 9 | 0.50 GiB |
| 75 sites, lrc 4 lrr 5, its current 11 exclusions | vf2 | none found within the time bound | — | — |
| **75 sites, lrc 4 lrr 5, its current 11 exclusions** | **exact (default)** | **74 / 28 / 0, proven optimal; boundaries 4 / 3 / 2, bulk 18 / 16 / 18** | **9** | **0.50 GiB** |
| 75 sites, lrc 4 lrr 5, no exclusions | exact | 73 / 29 / 0 (8 seam qubits, fewer than the balanced optimum of 9) | 8 | 0.77 GiB |

At n = 27 VF2 placed the circuit with 9–10 qubits on seams (17 exact / 24 replica CZ on 64 sites) where the tiling uses 3, and in the n = 18 smoke test it scored 0.26 against 0.60–0.69 for the annealed placements. The 75-site register (three 31-site simulators) holds up to ≈ 30 qubits in a single exact simulator, so its `fxeb` scores below n ≈ 30 are trivially exact; its comparisons are meaningful at the full 61 qubits, where it reaches 24–28 replica couplers at under 1 GiB, against 18 at 192 GiB for the two-patch register. `mirror_nighthawk.py` builds its backend with Qrack's default `is_torus=True`; add the `torus` token to match it (the torus does not change these placements).

The current `mirror_nighthawk.py` excludes 11 sites (74, 59, 44, 73, 69, 54, 68, 67, 64, 63, 62: six of the fifteen seam sites and five bulk sites), leaving exactly 64 sites for its 64-wire circuit, whose three dead qubits are idle wires the transpiler places like any other. On that register Qiskit's VF2 search stops at its time bound without a layout (its `NO_SOLUTION_FOUND` is not a proof), and `optimization_level=3` routes; on that script's gate set (SWAP, iSWAP and their CZ products) absorbs SWAPs into the qubit permutation. Measured on its own circuits, transpiling took 64–72 s per circuit and reduced the two-qubit gates from 141 to 83 at d = 4 and from 432 to 271 at d = 12, with none between simulators and 49–56 % touching a seam replica; all nine remaining seam sites hold a qubit. With `--ace-transpile` the 61 live qubits are placed once per register (cached), the three spare sites are free, and twirled circuits are not re-transpiled.

A placement may not join two patches that share no seam site: such a coupler is a contact between patches for which ACE keeps no boundary replica. The `exact` engine never produces one (it is outside the register's coupling map); the annealer weights it 1000 (against 20 in the tiling cost), and a placement that still contains one is refused, naming the coupler and the excluded seam sites that could be reopened; `--transpile-allow-cross` (token `xok`) accepts it. On the 11-exclusion register the annealer stops at such a placement, but CP-SAT shows the register is not the cause: the balanced three-way split of the Nighthawk graph needs at least 9 boundary qubits (proven, any register), and the register's open seam sites — four between patches 0 and 1, three between 1 and 2, two between 2 and 0 — hold exactly such a split, with 28 replica couplers and none across.

First mirror comparison at n = 61 (nnqab θ, one instance, 1000 shots, 4 twirls per point; placement effects beyond the coupler counts are visible, so treat it as indicative):

| Placement | exact / replica / cross | F, d = 4 | F, d = 6 |
|---|---|---|---|
| anneal, 11 exclusions, `xok` | 76 / 25 / 1 | 0.897 ± 0.010 | 0.118 ± 0.010 |
| anneal, 11 exclusions with 69 reopened | 72 / 30 / 0 | 0.799 ± 0.013 | 0.086 ± 0.009 |
| exact, 11 exclusions | 74 / 28 / 0 | 0.646 ± 0.015 | 0.388 ± 0.015 |
| exact, no exclusions | 73 / 29 / 0 | 0.121 ± 0.010 | 0.056 ± 0.007 |

The full mirror sweep on the exact 11-exclusion placement is in Sec. 9.8.

### 2.9 Width caps and larger registers: `--transpile-max-width`, `--register`

A patch is a dense state, so its width (bulk + seam replicas + ACE's error-detection ancilla) sets the memory: 8 bytes per amplitude in fp32, 1 GiB at 27 qubits, 2 GiB at 28. `--transpile-max-width W [--transpile-max-wide N]` adds this as a hard constraint to the `exact` engine: no patch wider than W, at most N patches at W, the others ≤ W − 1. Together with an ACE register of more sites than qubits, it lets the engine trade boundary qubits against memory.

`--register NAME` replaces the release's 61-qubit register with a larger sub-register of the 12 × 10 Nighthawk lattice, built with the release's own code and seed: `rect_matchings` and `remove_qubits` for the coupling graph, `make_patches(seed=2025, balance_tol=1, top_n=5)` for the K = 3 and K = 4 partitions the mirror circuits rotate through, and the release's recipe for the ten input strings (selftest [9] checks that this recipe reproduces the release). Only the `mirror` family runs on these registers, and there is no device data for them. With `--backend ace` and no explicit `--ace-transpile`, each register applies a stored ACE preset that fits one 8 GB GPU:

| Register | Sites used | ACE register | Patch widths (incl. ancilla) | Boundary qubits | exact / replica / cross CZ | Peak memory |
|---|---|---|---|---|---|---|
| `release` | the experiment's 61 | per flags | — | — | — | — |
| `nighthawk88` | rows 0–9 × cols 1–9 (88 of 90; the three broken qubits used) | 102 sites, lrc 4 lrr 5, flat | 28 / 27 / 27 / 27 | 17 | 102 / 55 / 0 | ≈ 7 GiB |
| `nighthawk88d` | whole lattice, 32 sites disabled (broken 17, 55, 62 + 29 corner sites) | 114 sites, lrc 4 lrr 5, flat | 28 / 23 / 25 / 27 | 11 | 108 / 42 / 0 | ≈ 5.3 GiB |

In `nighthawk88d` the region and the four-way split were chosen together by CP-SAT, under two constraints: every row and every column of the region is one unbroken run, so the only holes are broken qubits; and seams join only ring neighbours (0–1, 1–2, 2–3, 3–0), as in ACE's flat four-patch registers. Without the first constraint the optimum is trivial: four patches joined by three single qubits (B = 3, the minimum for a connected register), i.e. four nearly independent circuits. The cuts run along the lattice diagonals and all four patches meet at broken qubit 55. Per full A–B–C–D cycle 42 CZs pass through a boundary, against 55 in `nighthawk88` (11/11/10/10 against 14/14/14/13 per colour). Both placements are stored in the script, since CP-SAT needs minutes to find them; `RegisterLayout` accepts a window plus a list of dropped device sites, so further registers are a dictionary entry.

```bash
python3 nighthawk_qrack.py run --backend ace --register nighthawk88d --theta nnqab \
    --depths 4 6 8 10 12 16 20 --mirror-estimator prob --order shallow \
    --ace-gpu --gpus 0-5 --per-gpu 1 --stagger 20 --max-rss-gb 16 --out mirror88d_clvk.jsonl
```

---

## 3. Requirements

| Component | Notes |
|---|---|
| Python | ≥ 3.9 (uses `argparse.BooleanOptionalAction`) |
| numpy | required by every subcommand |
| matplotlib | `nighthawk_graph.py` only (Tk for the interactive viewer) |
| OR-Tools | `--ace-transpile` with the default `exact` engine, and `--register` presets (`python3 -m pip install ortools`) |
| PyQrack | required by `hwxeb`, `run`, `selftest`, `aceplan`; **not** by `verify` or `seamgap`. The `ace` backend needs a PyQrack build that ships `QrackAceBackend`; exact references above 32 qubits need Qrack built in >32-qubit mode |
| Qrack shared library | if `/usr/local/lib/qrack/libqrack_pinvoke.so` exists, the script exports it as `PYQRACK_SHARED_LIB_PATH` |
| Qiskit | only for `--ace-transpile` |
| GPU | OpenCL (select with `QRACK_OCL_DEFAULT_DEVICE`) or `--cpu`. No CUDA path is used. See Sec. 8.3 for ACE on GPU and Sec. 8.7 for the Qrack build to use |

The header block every PyQrack script in this repository carries:

```python
QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):
    os.environ["PYQRACK_SHARED_LIB_PATH"] = QRACK_LIB_PATH
```

### 3.1 Environment variables

| Variable | Effect |
|---|---|
| `QRACK_OCL_DEFAULT_DEVICE` | OpenCL device index; the launcher sets it per worker from `--gpus` |
| `QRACK_QUNITMULTI_DEVICES`, `QRACK_QPAGER_DEVICES` | devices Qrack may spread units and pages over; the launcher pins both to the worker's device unless set by the user, so ACE patches do not land on other cards of a mixed host. For single-process runs set all three by hand: Qrack's own default device is not necessarily device 0 |
| `QRACK_MAX_CPU_QB` | upper bound on qubits for CPU state vectors |
| `QRACK_MAX_ALLOC_MB` | Qrack's memory ceiling |
| `NIGHTHAWK_REPO` | location of the `rcs-nighthawk` checkout if `--repo` is not given |
| `NIGHTHAWK_TILE_CACHE` | tiling cache file (default `nighthawk_tiles.json` next to the script) |
| `NIGHTHAWK_MEM_LEDGER` | memory-reservation directory shared by all runs (default `/tmp/nighthawk_mem_ledger`) |
| `QRACK_OCL_SYNC` | Qrack ≥ 10.13.2: `1` finishes the OpenCL queue after every dispatch. Avoids the defects of Sec. 8.7 by serialising; not needed with the async fix |
| `OCL_ICD_VENDORS` | for CPU runs, point it at an empty directory (`/tmp/no-icd`) so that no OpenCL platform is initialised and no GPU memory is taken |

---

## 4. Installation

```bash
git clone https://github.com/twobombs/thereminq-examples
cd thereminq-examples/rcs/nighthawk
git clone https://github.com/BlueQubitDev/rcs-nighthawk     # circuits, layout, hardware counts
pip install numpy pyqrack matplotlib                        # present in the ThereminQ container
```

The release is found via `--repo`, `$NIGHTHAWK_REPO`, or a `data/layout.json` (or `rcs-nighthawk/` clone) in the working directory, next to the script, or up to three parents up.

| Path in the release | Used by |
|---|---|
| `data/layout.json` | all (qubit map, matchings, schedule, partitions, seeds, depths, mirror input strings) |
| `data/circuits/manifest.json`, `data/circuits/**.qasm` | `verify`, `selftest`, `seamgap` |
| `data/counts/patched_K{K}_d{d}.npz`, `data/counts/mirror_survival.json` | `hwxeb`, `selftest` |
| `data/results/patch_xeb.json`, `data/results/fidelity_vs_depth.json` | `hwxeb`, `run --summarize`, `selftest`, `seamgap`, graph tool |

---

## 5. Recommended workflow

```bash
# 1  circuits are regenerated exactly (no Qrack needed)
python3 nighthawk_qrack.py verify
# 2  conventions and exactness checks
python3 nighthawk_qrack.py selftest
# 3  the paper's hardware numbers, re-scored with PyQrack as reference
python3 nighthawk_qrack.py hwxeb --cpu --cache patch_cache
# 4  which ACE layouts exist, ranked by couplers across simulators (after tiling)
python3 nighthawk_qrack.py aceplan --n 27 --tiling
# 5  layout scan at one size against the exact reference
for L in "4 4" "2 7" "1 7" "4 3" "2 2"; do set -- $L
  python3 nighthawk_qrack.py run --backend ace --families fxeb --n 27 --depths 8 12 \
    --lrc $1 --lrr $2 --ace-tiling --cpu --out tile_${1}_${2}.jsonl &
done; wait
# 6  size sweep, two tilings against one shared reference, memory-aware workers
python3 nighthawk_qrack.py run --backend ace --families fxeb --sizes 27-34 --depths 12 16 20 \
  --variants 4/4:t3 4/4:t2 --gpus 0 --per-gpu 4 --mem-budget-gb 280 --max-rss-gb 280 \
  --out big_var.jsonl
# 7  everything collected so far, side by side
python3 nighthawk_graph.py --table harvest.csv
python3 nighthawk_graph.py                  # interactive viewer
```

Mirror and patched emulation of the released 61-qubit circuits (the proxy families of Ref. [[1]](#ref-1)) remain available as before, e.g. `run --backend ace --families mirror --sizes 27-36,61 --depths 4 6 8 12` and `run --backend exact --families patched --depths 20 36`.

---

## 6. Subcommand reference

### 6.1 `verify` — circuit regeneration

Rebuilds every released circuit from `layout.json` exactly as `rcs/circuits.py` does and compares it gate-by-gate with the released QASM: SplitMix64 hash of `(seed, instance, cycle, qubit, k)` → Haar angles (φ, θ, λ) applied as `rz(φ)·rx(θ)·rz(λ)`; seeds `2025 + k·1000003` (full, patched) and `2025` with instance `k` (mirror); mirror, patched K=3/K=4 over all partitions and instances, and full circuits at d = 4, 8, …, 40. Angles are compared modulo 2π within `--tol` (default `1e-9` rad) and CZ counts against `manifest.json`. Exit status is 0 only if every circuit is identical and none in the manifest was left unregenerated.

### 6.2 `selftest` — conventions and exactness

| # | Check | Pass criterion |
|---|---|---|
| 1 | angle hash + pseudo-patch rotation vs `mirror/d08_instance1.qasm` | identical |
| 2 | `u_angles` → Qrack `u` equals the Haar matrix | overlap within 1e-5 of 1 |
| 3 | exact mirror, first `--n` qubits, d = 12, all 10 inputs | \|1 − F\| < 1e-4 |
| 4 | Pauli-frame twirl is an identity (exact backend) | \|1 − F\| < 1e-4 |
| 5 | PyQrack patch ideal XEB vs the release (Aer, double) | max diff < 1e-3 |
| 6 | Eq. (1)–(2) on the hardware counts vs the release | within 0.05 σ |
| 7 | ideal samples score F = 1 | within 4 σ |
| 8 | ACE register geometry | grid size, `is_torus`, nearest-neighbour couplers |
| 9 | `--register` recipe applied to the release's own register | inputs and K = 3, 4 partitions identical to `layout.json` |

### 6.3 `hwxeb` — the paper's numbers with PyQrack as reference

Loads the measured `ibm_phoenix` bitstrings, computes every patch's ideal distribution with `QrackSimulator` (renormalised in float64, also when read from `--cache`), and applies Eq. (1)–(2); per (K, d) it prints the inverse-variance fidelity and mean collision ratio next to the release's, then refits the mirror survival data and reports F(36) and error per qubit per cycle. PyQrack's float32 builds are expected to agree with the release to about 1e-5 in ideal XEB.

### 6.4 `aceplan` — ACE layouts of the register

Enumerates every distinct ACE layout of the register holding the first `--n` logical qubits (all `lrc`, `lrr`; non-torus by default, `--torus any|true` to include torus layouts) and prints, per layout, the couplers across simulators, couplers on a seam, seam qubits, simulator count, simulator widths and worst-case dense memory, ranked by couplers across simulators. `--tiling` scores each layout after tiling (version 3; about 1–2 s per layout); `--max-width W` marks the layout `run --ace-max-width W` would pick.

### 6.5 `run` — clean-qubit emulation

**Families** (`--families`, comma list):

| Family | What runs | Score |
|---|---|---|
| `mirror` | pseudo-patched U(d/2) then exact inverse, ten released inputs, shots split across inputs (and twirls on ACE) | survival, binomial σ |
| `patched` | K-patch circuit on the 61-qubit register (n = 61 only) | Eq. (1)–(2), delta-method σ |
| `full` | unpatched forward circuit, release seed, `--pubs` pubs | none (stored only) |
| `fxeb` | unpatched forward circuit on the first n qubits, release seed per instance | forward XEB vs exact reference (Sec. 2.5); also linear XEB, HOG fraction, ideal XEB |

**Backend and layout flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--backend` | `ace` | `exact` or `ace` |
| `--lrc`, `--lrr` | `4`, `4` | ACE `long_range_columns`, `long_range_rows`, or `auto` (two-patch rule of `nn_qab.py`) |
| `--ace-torus` | off | `is_torus=True` (the device patch is not a torus) |
| `--geometry` | `manual` | `nnqab`: rows whole, two patches, highest bulk-to-boundary ratio, torus |
| `--ace-max-width W` | none | fewest couplers across simulators with every internal simulator ≤ W qubits |
| `--ace-torus-search` | `flat` | layouts `--ace-max-width` may use: `flat`, `torus`, `any` |
| `--ace-layout` | `grid` | `grid` (8×8 for every n) or `strip` (smallest R×8 strip holding the first n) |
| `--ace-tiling` | off | place logical qubits as compact tiles (Sec. 2.6) |
| `--ace-transpile` | off | place logical qubits on a register and coupling map of your choosing (Sec. 2.8); needs Qiskit |
| `--ace-width W`, `--ace-exclude a,b,…` | — | with `--ace-transpile`: ACE register size in sites, and sites removed from its coupling map |
| `--transpile-engine` | `exact` | `exact`: CP-SAT placement, fewest seam qubits, then even patches, then fewest seam couplers (needs OR-Tools); `anneal`: the tiling cost annealed on that register; `vf2`: Qiskit's VF2Layout as is |
| `--transpile-allow-cross` | off | accept placements in which a coupler joins two patches that share no seam site (refused by default) |
| `--transpile-seed`, `--transpile-trials`, `--transpile-calls` | 0, 2000, 2·10⁶ | VF2Layout seed, layouts scored, isomorphism-search bound |
| `--ace-tiling-version` | `3` | `2` reproduces earlier tiled runs and their config tags |
| `--ace-gpu`, `--ace-host-pointer` | off | ACE on OpenCL; keep simulator states in host memory. See Sec. 8.3 |
| `--ace-boundary-rep`, `--no-ace-error-detection`, `--no-ace-crossbars` | Qrack defaults (off, on, on) | ACE's boundary repetition code, seam error detection (detect-and-post-select) and crossbar simulator; non-default settings enter the config tag and the record |
| `--theta` | `haar` | `nnqab`: sin θ uniform — a variant ensemble, not the released circuits |
| `--register` | `release` | a larger Nighthawk sub-register with its ACE preset (Sec. 2.9); mirror only |
| `--transpile-max-width W`, `--transpile-max-wide N` | off | `exact` engine: patch width cap incl. the ancilla, and how many patches may reach it (Sec. 2.9) |
| `--mirror-estimator` | `shots` | `prob`: per circuit run, the probability ACE returns the prepared string (chain rule with forced measurement), so there is no shot noise and no shot floor; `shots`: sample with `measure_shots` at the paper's budgets |
| `--ace-gpu-shots` | off | with `--ace-gpu`: draw shots on the GPU instead of a CPU copy of the final state |

**Sampling and scope:** `--depths`, `--K`, `--instances` (3), `--partitions`, `--shots` (`paper` = Appendix-D budgets for mirror/patched, 4096 for `fxeb`, 100 000 per pub for `full`; or an integer), `--twirls` (64), `--exact-probs`, `--sizes` (`27-34`, `20,27-36`), `--n`, `--pubs`, `--cache`, `--cpu`, `--ref-gpu` (exact `fxeb` reference on OpenCL).

**Shared reference.** `--variants 4/4:t3 4/4:t2 4/3:t3 …` (lrc/lrr, then `torus`, `untiled`, `tiled`, `t2`, `t3`, the ACE options `rep`, `noed`, `noxbar`, and the transpile placements `tx` (exact), `txa` (anneal) or `vf2`, with `w<N>`, `ex<a+b+…>` and `xok` for `--transpile-allow-cross`) samples each `fxeb` point with every listed ACE configuration and scores all of them against **one** exact reference. Records, bitstrings and config tags stay per variant; only the claims are shared under a group tag. At n = 33–34, where the reference dominates the cost, this compares configurations for the price of one.

**Workers and memory.**

| Flag | Default | Meaning |
|---|---|---|
| `--gpus`, `--per-gpu` | none, `3` | launch `len(gpus) × per_gpu` workers through a claim-file queue |
| `--stagger`, `--init-gap` | `20`, `10` s | spacing of worker starts and of OpenCL initialisations |
| `--max-attempts`, `--retry-failed` | `2` | take-overs of a point whose worker died; clear failed marks |
| `--max-rss-gb` | off | per-worker resident-memory cap (watchdog marks the point failed and exits) |
| `--mem-budget-gb` | off | machine-wide budget shared by all workers and runs using the same ledger |
| `--mem-ledger` | `/tmp/nighthawk_mem_ledger` | reservation directory |
| `--order` | `deep` | `deep`: most expensive points first (no lone tail); `shallow`: cheapest first, for early results |
| `--status` | — | list every point of the configuration (done with F, running with progress and ETA, failed, pending); reads files only, safe while a run is going; give the run's own flags |
| `--gpu-isolation` | `auto` | every GPU worker sees exactly one OpenCL device. Required for clvk (one device per context): its workers get only clvk's ICD and the Vulkan device-select layer pinned to their card's PCI address |

A point marked failed (twice-died workers, or an exception) is skipped by every worker; `--retry-failed` clears the marks once, at launcher start. To drop pending points deliberately, write `failed: …` into their claim files under `<out>_shots/<cfg>/claims/`. With `--mirror-estimator prob`, a forced outcome whose probability is below 10⁻⁶ counts the run as failed (survival 0) and rebuilds the backend; the number of such runs is recorded (`prob_errors`).

With a budget, a worker takes a point only if its estimated peak fits both the budget (all reservations plus this point) and `MemAvailable` after the not-yet-allocated part of other reservations; otherwise it takes a smaller point or waits. The estimate is 1 GiB of overhead, the engines' worst case, and for `fxeb` the reference at 1.5 × 2ⁿ amplitudes (state + probability buffer). For fp32 builds this is 12 · 2ⁿ bytes: 1.5 GiB at n = 27, 48 GiB at n = 32, 193 GiB at n = 34. The ledger tracks host RAM only: with `--ref-gpu` or `--ace-gpu`, size `--per-gpu` to the card's memory yourself (≈ 8 · 2ⁿ bytes per reference while it is built, ≈ 0.15 GiB of ACE engines and ≈ 0.3 GiB of driver context per worker).

**Resumability.** Every configuration is tagged `<backend>-<sha1[:8]>` over backend, shots, twirls, ACE parameters, tiling version and exact-probs mode, but *not* the register size, so a sweep can be extended and resumes per (n, point). Bitstrings are written atomically *before* the JSONL record, and the claim is marked done last.

### 6.6 `seamgap` — data-only analysis

NumPy only: CZ counts per family and depth, the paper's fit at d = 36 and at the gate-count-equivalent depth, the fit corrected for missing CZs at three Pauli error rates, a weighted regression ln F = a + β·cycles + γ·CZ, the K=3 vs K=4 ratio and its implied ε_CZ, and a per-cycle error budget.

---

## 7. Outputs

### 7.1 JSONL record (`run`)

Schematic `fxeb` record from a `--variants` run (values illustrative):

```json
{"cfg": "ace-fd29e9a4", "n": 30, "family": "fxeb", "K": 0, "depth": 12, "partition": 0, "instance": 1,
 "seconds": 612.4, "fidelity": 0.0689, "se": 0.0085, "xeb_linear": 0.104, "ideal_xeb": 1.39, "hog": 0.494,
 "shots": 4096, "ace_seconds": 41.3, "ref_seconds": 1142.2, "ref_shared": 2,
 "ace_register": 64, "ace_lrc": 4, "ace_lrr": 4, "ace_torus": false, "ace_widths": [23, 23, 20, 19, 16],
 "ace_couplers": {"exact": 34, "replica": 13, "cross": 0}, "ace_seam_used": 5, "ace_sims_used": 2,
 "ace_tiling": true, "ace_tiling_version": 3, "ace_map": "…", "shots_file": "…"}
```

Every record carries `theta` (`haar` or `nnqab`), the single-qubit ensemble of its circuits. `ace_map` is a hash of the placement; the graph tool uses it to verify recomputed coupler splits. Mirror records carry `hits`, `shots_per_input`; patched records `patch_fidelity`, `patch_ideal_xeb`.

### 7.2 Bitstrings in the release's own layout

Per-point files under `<shots-dir>/<cfg>/points/n{n}/` are merged into `<shots-dir>/<cfg>/release/` (`counts/patched_K{K}_d{d}.npz`, `counts/mirror_survival.json`, `counts/mirror_shots_d{dd}.npz`, `samples/full_d{d}.npz`) so BlueQubit's analysis scripts read them unchanged. Truncated registers get an `_n{n}` suffix; bit q is logical qubit q (little-endian).

### 7.3 Summary tables (`run --summarize`)

Per n and family: F_sim ± σ (and F_hardware at n = 61); the mirror fit and per-cycle decay b(N) with a non-negative fit b = u·N + v·CZ/cycle extrapolated to N = 61; for `fxeb`, XEB (inverse-variance over instances), linear XEB, HOG, ideal XEB and the device fit at the same depth. With `--variants`, one block per variant.

**Weighting.** The inverse-variance mean weights each instance by its shot-noise 1/σ², and for forward XEB that σ grows with the instance's own XEB and output concentration. When instances differ strongly, the weighted mean is therefore biased low: for the 20-circuit `nn_qab`-ensemble run of Sec. 9.7 it gives 0.343 (tiled) and 0.125 (untiled), against plain means over circuits of 0.435 ± 0.037 and 0.153 ± 0.014. The quantity of interest is the average over circuits, as in `nn_qab`; quote the plain mean with its instance-scatter error (`nighthawk_graph.py --table`, views 8–11).

### 7.4 `nighthawk_graph.py`

With no `--out`, every `*.jsonl` in the working directory is harvested (worker files folded in) and configurations are labelled from their records (`ace 4/4 tiled v3`, `ace 2/7 untiled`, …).

| View | Content |
|---|---|
| 1–7 | device fidelity, mirror decay, b(N), bitstring diagnostics, release data, run cost |
| 8 | `fxeb`: XEB vs depth, one panel per n, every configuration |
| 9 | `fxeb`: XEB vs n, one panel per depth |
| 10 | `fxeb`: per-cycle decay b(N) per configuration, extrapolated to 61, against the device |
| 11 | `fxeb`: XEB at one (n, d) against exact/replica/cross couplers, seam qubits used, simulators used and widest simulator, with rank correlations |

Views 8–11 and the `--table` comparison show one single-qubit ensemble at a time (`--theta haar|nnqab|all`, default `haar`), since XEB values of different ensembles are not comparable; configurations of the `nn_qab` ensemble are labelled `[θ nnqab]`. Records written before the `theta` field existed are classified by their ideal XEB at d ≥ 8 (> 5 for the concentrated `nn_qab` outputs). `--table FILE` writes every point as CSV (coupler splits recomputed for older records where the placement hash matches) and prints a per-(n, d) ranking and each configuration's b. In views 8–11, points are plain means over instances with the standard error from the instance scatter, which exceeds shot noise; `run --summarize` uses inverse-variance weights.

---

## 8. Interpretation and practical notes

### 8.1 Expected outcomes

| Backend / family | Expected | A deviation means |
|---|---|---|
| `exact`, any | F = 1 (b = 0) up to shot noise | a pipeline or convention error; run `selftest` |
| `ace`, mirror | F < 1, decaying with depth | ACE's seam error per cycle; compare b_ace to the device's b |
| `ace`, patched | ≤ 1 | ACE's seams are set by the layout, not by the partitions |
| `ace`, fxeb | XEB < 1, decaying with depth | measured approximation error against ground truth |
| `hwxeb` | matches the release to ≪ 1 σ | Qrack and the release's reference agree |

A clean-qubit simulator whose decay is *slower* than the device's is a statement about that backend's approximation cost on the paper's own proxies, not a spoofing result; the limits of these proxies as certificates are discussed in [[4]](#ref-4) and [[5]](#ref-5).

### 8.2 Depth window

At d ≤ 8 the n = 27–30 outputs are still concentrated (ideal XEB 3–4 at d = 8, ≈ 66 at d = 4), the normalised XEB is dominated by few heavy bitstrings and is non-monotonic in depth; rank configurations on d ≥ 12, where the ideal XEB is 1.0–1.6. Because ACE's decay is not a single exponential (Sec. 9.4), b must be compared at matched depth windows across sizes.

### 8.3 ACE and the reference on GPU

Both engines run on OpenCL devices: the exact `fxeb` reference with `--ref-gpu` (an ordinary dense state vector) and ACE with `--ace-gpu`. Tiled ACE was validated on an NVIDIA CMP 50HX (TU102, 10 GB, NVIDIA OpenCL): the v2-tiled n = 27 points reproduce the CPU run on the same circuits (d = 12, 16, 20: 0.146, 0.063, 0.033 on GPU against 0.158, 0.063, 0.024 on CPU; inverse-variance means, ±0.009). Before using another OpenCL device for tiled ACE, run one point against a CPU run of the same circuits.

GPU execution of ACE is not faster here: its simulators are small (≤ 23 qubits for tiled 4/4), its work is many short kernels with frequent read-backs and host-side seam logic, and workers sharing one card queue on it, whereas CPU workers run on separate cores. On the CMP 50HX an n = 27–28 `fxeb` point with two variants took ≈ 10 min per worker with three workers, an n = 29 point ≈ 13 min with two; the n = 27 reference alone took ≈ 10 s. The useful split is ACE on CPU workers and the reference on a GPU, unless CPU cores are the scarcer resource.

**Device memory.** Device memory is not tracked by the memory ledger. Concurrent references on one 10 GB card failed with `QrackSimulator C++ library raised exception` (ten workers at n = 27; three at n = 28) and completed when rerun with one worker. Per worker, budget ≈ 8 · 2ⁿ bytes for the reference while it is built plus ≈ 0.5 GiB of ACE engines and driver context: on a 10 GB card ≤ 4 workers at n = 27, ≤ 2 at n = 28–29, 1 at n = 30, none beyond.

**Selecting a device.** Indices follow Qrack's start-up listing (all OpenCL platforms in one list; check before long runs, as driver changes reorder it). `--gpus N` pins each worker fully to device N (Sec. 3.1); for single-process runs set `QRACK_OCL_DEFAULT_DEVICE`, `QRACK_QUNITMULTI_DEVICES` and `QRACK_QPAGER_DEVICES` together, since Qrack's own default device need not be 0.

### 8.4 Precision

The exact reference is held in float32. Floating-point storage keeps ~7 significant digits at any magnitude, so 2⁻³⁴-scale probabilities are not degraded by their size; CZs are sign flips and add no rounding; accumulated single-qubit rounding at d = 20, n = 34 is of order 10⁻⁶–10⁻⁵ relative per probability, ~10⁻⁵ in XEB, against a per-instance shot noise of ~0.015 at 4096 shots. All sums over 2ⁿ probabilities are taken in float64 and sampled probabilities are divided by the float64 total. The measured ideal XEB at d = 20 (1.02–1.05) is consistent with Porter–Thomas.

### 8.5 Shot floor

At 4096 shots the per-instance standard error is ≈ 0.015, so XEB below ≈ 0.03 is unresolved; d = 16–20 points at n ≥ 30 need ≥ 16 384 shots to enter a decay fit.

### 8.6 What can be verified at 61 qubits

Scoring a sample by XEB needs the ideal probability of each sampled bitstring. The release contains every bitstring of the 61-qubit circuits that were sampled — all 10⁶ samples of the full d = 36 circuit (all distinct) and every shot of every patched circuit, as 61-bit integers, stored sorted so that shot order is not recoverable — but for mirror circuits only the number of shots that returned the prepared string. No probability of the full 61-qubit circuit is computed in Ref. [[1]](#ref-1) or here: the headline fidelity is inferred from the patched and mirror proxies under the assumption that the full circuit decays at the same rate per cycle, and every 61-qubit statement in this document is a proxy measurement or an extrapolation of b(N) (Sec. 10).

The release's `contraction_cost/results/depth_sweep.json` gives BlueQubit's best-found single-amplitude contraction cost of the full circuits (cotengra/kahypar, unlimited memory, following the methodology of [[4]](#ref-4)); the authors note every value is an upper bound on the optimal path cost:

| Depth | 20 | 28 | 32 | 36 | 40 |
|---|---|---|---|---|---|
| log₁₀ C_amp (complex multiply-adds) | 13.0 | 17.6 | 19.8 (repeat 20.0) | 21.8 (repeat 22.0) | 22.0 (repeat 22.1) |
| largest intermediate tensor | 2³⁴ | 2⁴⁹ | 2⁵⁶ | 2⁶⁴ | 2⁶⁵ |

Their sampling-cost model W = 8 κ N_s F C_amp (κ = 10, frugal rejection sampling), on Frontier at 20 % of 1.685 × 10¹⁸ FLOP/s, gives — evaluated here for N_s = 10⁶ and F(36) = 0.326 × 0.8717³⁶ ≈ 2.3 × 10⁻³ — W ≈ 1.2 × 10²⁷ FLOPs, ≈ 115 years (≈ 160 years with the repeat search). The intermediate of 2⁶⁴ entries (≈ 128 EiB in complex64) exceeds any machine, so a real contraction must slice, which raises the cost. Verification is costlier still in kind: it needs exact amplitudes (≈ 5 × 10²² FLOPs, ≈ 2 days on Frontier each, before batching) for ≈ 1/F² ≈ 2 × 10⁵ sampled bitstrings to resolve F. Within the cost model of the release, the 61-qubit, 36-cycle samples cannot be scored directly.

### 8.7 Qrack's OpenCL engine: host-memory defects and the fix

Deep ACE mirror points on GPU (61 qubits, d ≥ 8) crashed or returned wrong values on several OpenCL platforms while the same points ran cleanly on CPU. A 16-qubit reproducer without ACE (`qrack_ocl_repro.py`: one `QrackSimulator`, 20 layers of U + CZ, `prob()` on two qubits, CPU and OpenCL in separate processes) shows it on PoCL with Qrack 10.13.1:

| Engine | prob(q3) / prob(q10) |
|---|---|
| CPU | 0.414847 / 0.430348 |
| OpenCL, stock | NaN / NaN, NaN / NaN, segfault (three runs) |
| OpenCL, `DISABLE_OOO_OCL` | 1.0 / 1.0 (wrong) |
| OpenCL, synchronous dispatch | 0.414847 / 0.430348 |

Tracing `QEngineOCL` (`src/qengine/opencl.cpp`) found three defects, all on the application side of the OpenCL API:

| | Defect | Where |
|---|---|---|
| A | non-blocking buffer writes (`DISPATCH_WRITE`) read from stack arrays that go out of scope before the write executes; the OpenCL specification requires the host pointer to stay valid until completion | `Dispose(perm)`, `DecomposeDispose` (its argument block is overwritten while the first write is queued) |
| B | buffers created with `CL_MEM_USE_HOST_PTR` (CPU devices and integrated GPUs) have their host memory freed while kernels using them are still queued | `Dispose`, `Compose` |
| C | the `_PopQueue` completion callback is registered on an empty `cl::Event` before the kernel is enqueued; `clSetEventCallback` returns `CL_INVALID_EVENT` every time (1099 of 1099 registrations), so the callback never runs | `DispatchQueue` |

Fixing A and B together, with dispatch still asynchronous, restores correct results; either alone does not, nor does `DISABLE_OOO_OCL`. Upstream Qrack 10.13.2 (unitaryfoundation/qrack `cf53586`) adds an opt-in `QRACK_OCL_SYNC=1`, which finishes the queue after every dispatch; the maintainer does not intend to change the asynchronous path further. ThereminQ therefore carries `qrack_ocl_async_fix.patch` (one file, applies to 10.13.1 and 10.13.2): each asynchronous write copies its argument block to the heap and frees it from the write's completion callback, a `USE_HOST_PTR` buffer owns its host memory through a `clSetMemObjectDestructorCallback` holder, and the dead callback is removed. Validated on PoCL on top of 10.13.2: the reproducer exact in 5 of 5 runs, ACE mirror points d = 4 and 6 × 3 instances at 640 runs each equal to the CPU on the same build within 3 × 10⁻⁴, and 185 s against 198 s with `QRACK_OCL_SYNC=1`.

```bash
git clone https://github.com/unitaryfoundation/qrack && cd qrack && git checkout cf53586f
git apply /path/to/qrack_ocl_async_fix.patch
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF -DENABLE_OPENCL=ON -DQBCAPPOW=7 ..
make -j$(nproc) qrack_pinvoke && sudo make install     # /usr/local/lib/qrack/libqrack_pinvoke.so
```

Before a long GPU run, run the reproducer and one ACE point on the target device against a CPU run of the same circuits.

---

## 9. Results to date

Sections 9.1–9.7 are clean-qubit ACE emulations of the released circuit ensemble (Haar single-qubit gates, release seeds), scored by forward XEB against an exact reference, 3 instances, 4096 shots unless noted; Sections 9.8–9.10 are mirror circuits (the `nn_qab` ensemble, `prob` estimator) at 49–88 qubits. The device figure of merit for comparison is the paper's mirror fit F(d) = 0.326 × 0.8717^d, i.e. b_dev = 0.137 per cycle at n = 61. Values from `run --summarize` are inverse-variance means and may lie below the plain mean over circuits (Sec. 7.3); values from the graph tool are plain means. Each table states which.

### 9.1 Layout choice without tiling

Ranking the non-torus layouts of the 8×8 window by couplers on a seam, the best layout with all simulators ≤ 33 qubits (5/4) improves on the default 4/4 by one coupler (68 vs 69 of 102). Larger simulators help only on a torus (2/7 torus: 57 of 102, simulators of 33, 33, 25, 25 qubits, 128 GiB dense worst case); layouts with fewer seam qubits on the flat register alternate chunk ownership by row and place every vertical coupler across simulators.

### 9.2 Placement dominates layout (n = 27)

Inverse-variance means over 3 instances (`run --summarize`):

| lrc/lrr | XEB d=8, device placement | XEB d=8, tiled (v2) | XEB d=12, tiled (v2) |
|---|---|---|---|
| 4/4 | 0.008 | **0.321** | **0.182** |
| 4/3 | 0.036 | 0.227 | 0.153 |
| 2/7 | 0.046 | 0.154 | 0.136 |
| 1/7 | 0.077 | 0.145 | 0.109 |
| 2/2 | −0.032 | 0.117 | 0.091 |

Tiling removed every cross coupler in all five layouts at unchanged memory. The widest simulator does not predict the outcome: 2/7 (28-qubit simulators) is below 4/4 (23-qubit simulators).

### 9.3 What predicts the XEB

Over the ten configurations of Sec. 9.2 at n = 27, d = 8 (view 11), Spearman rank correlations of XEB with: simulators holding bulk qubits −0.84, cross couplers −0.80, exact couplers +0.74, logical qubits on seam sites −0.40, replica couplers −0.21, widest simulator −0.15. Removing cross couplers is necessary; among crossing-free tilings the ordering follows seam qubits used (Sec. 9.4).

### 9.4 Size and depth dependence (4/4, tiling v2)

| n | seam qubits / simulators | d = 12 | d = 16 | d = 20 |
|---|---|---|---|---|
| 27 | 3 / 2 | 0.158 | 0.063 | 0.024 |
| 28 | 8 / 3 | −0.015 | 0.023 | 0.005 |
| 29 | 5 / 3 | 0.101 | 0.043 | 0.008 |
| 30 | 6 / 3 | 0.069 | 0.001 | −0.001 |

(inverse-variance means from `run --summarize`; σ ≈ 0.009 shot noise, instance scatter larger; plain means over circuits may be higher, Sec. 7.3.) At d = 12 the XEB falls monotonically with seam qubits used (3, 5, 6, 8 → 0.158, 0.101, 0.069, −0.015), so the size-to-size variation is dominated by tiling quality rather than by n. The decay is not exponential in depth: at n = 27, b ≈ 0.19 over d = 8–12 and ≈ 0.25 over d = 16–20; over d = 12–20, b ≈ 0.24 at n = 27 and ≈ 0.31 at n = 29, against b_dev = 0.137 at n = 61. At the headline depth the tiled emulator therefore loses fidelity faster per cycle than the device does at more than twice the register size, and the gap widens with depth.

### 9.5 Tiling version 3

Version 2 occasionally tiled a larger register worse than a smaller one (n = 28: 8 seam qubits after 3 at n = 27). Version 3 (Sec. 2.6) gives, for 4/4:

| n | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 |
|---|---|---|---|---|---|---|---|---|
| v2 exact/replica/cross, seam/sims | 31/10/0, 3/2 | 23/20/0, 8/3 | 27/18/0, 5/3 | 28/19/0, 6/3 | 29/20/0, 7/3 | 22/28/0, 11/3 | 34/18/0, 6/3 | 35/19/0, 6/3 |
| v3 exact/replica/cross, seam/sims | 32/9/0, 3/2 | 30/13/0, 4/3 | 33/12/0, 4/2 | 34/13/0, 5/2 | 35/14/0, 5/3 | 35/15/0, 6/3 | 35/17/0, 5/3 | 36/18/0, 5/3 |

The placements are deterministic and were reproduced identically on two machines. Both tilings were sampled on the same circuits and scored against one shared reference per point (`--variants 4/4:t3 4/4:t2`, 3 instances; n = 27–29 on the NVIDIA CMP 50HX with `--ace-gpu --ref-gpu`):

| n | Seam qubits / simulators, v3 vs v2 | d = 12: v3 / v2 | ratio | d = 16: v3 / v2 | d = 20: v3 / v2 |
|---|---|---|---|---|---|
| 27 | 3/2 vs 3/2 | 0.165 / 0.146 | 1.1 | 0.062 / 0.063 | 0.026 / 0.033 |
| 29 | 4/2 vs 5/3 | 0.140 / 0.114 | 1.2 | 0.030 / 0.034 | 0.011 / 0.015 |
| 28 | 4/3 vs 8/3 | **0.147 / 0.040** | **3.6** | **0.051 / 0.012** | 0.016 / 0.021 |

(inverse-variance means from `run --summarize`; σ ≈ 0.009 shot noise; d = 20 is at the 4096-shot floor.) The prediction holds: where the seam counts agree (n = 27) the tilings coincide; one seam qubit fewer (n = 29) gives a small gain at d = 12 and none resolvable at d = 16; halving the seam qubits (n = 28) raises the XEB 3.6–4.3-fold. With v3 the d = 12 XEB falls smoothly with size (0.165, 0.147, 0.140 at n = 27, 28, 29), so the n = 28 dip of the earlier runs (Sec. 9.4) was a tiling artefact. The per-cycle decay over d = 12–20 rises with n (b ≈ 0.23, 0.27, 0.32) and stays above b_dev = 0.137.

**ACE settings.** The three switches of `QrackAceBackend` were compared on the same circuits as `--variants 4/4:t3 4/4:t3:rep 4/4:t3:noed 4/4:t3:noxbar` (n = 27–28, d = 12 and 16, 5 circuits per point, CMP 50HX, updated PyQrack). Paired per-circuit differences from the default:

| Setting | n=27, d=12 | n=27, d=16 | n=28, d=12 | n=28, d=16 |
|---|---|---|---|---|
| boundary repetition code on (`rep`) | −0.007 ± 0.010 | +0.009 ± 0.008 | −0.012 ± 0.011 | −0.001 ± 0.008 |
| seam error detection off (`noed`) | +0.008 ± 0.006 | +0.007 ± 0.009 | +0.029 ± 0.014 | +0.011 ± 0.005 |
| crossbars off (`noxbar`) | +0.001 ± 0.012 | +0.002 ± 0.009 | −0.020 ± 0.009 | −0.002 ± 0.005 |

The repetition code and the crossbars have no measurable effect. Disabling ACE's detect-and-post-select step raises the XEB in all four cells, by ≈ +0.010 ± 0.003 combined (an upper bound on significance, since the d = 12 and d = 16 circuits of an instance share their first twelve cycles), most at n = 28 where more qubits sit on seams. The gain is a constant offset: the per-cycle decay from d = 12 to 16 is unchanged (b ≈ 0.27 → 0.25 at n = 27, 0.26 → 0.26 at n = 28). No ACE setting reduces the cost per seam coupler; the remaining lever is structural (Sec. 9.9).

Because both variants share circuits, the decisive statistic is the *paired* per-circuit difference v3 − v2, in which circuit-to-circuit variation cancels; the CSV from `nighthawk_graph.py --table` carries the per-instance values for it. Sizes n = 30–32 (seam differences 1, 2, 5) are pending on the CPU path; n = 32, with 11 seam qubits under v2 against 6 under v3, is expected to show the largest gap.

### 9.6 Cost

The run cost is dominated by the exact reference: about 6.2 × 10³ s for one n = 34, d = 20 point on a 96-thread EPYC host, 1–4 × 10² s per tiled 4/4 point at n = 27. Wider ACE simulators also cost time (tiled 2/7 ≈ 1.8 × 10³ s per point at n = 27). On the CMP 50HX (`--ace-gpu --ref-gpu`, both variants per point) a point took ≈ 10 min at n = 27–28 and ≈ 13 min at n = 29 per worker. Mirror circuits at n = 61 cost ≈ 0.65–4.5 × 10³ s per point untiled (d = 4–20); their cost grows with depth (every twirl runs d/2 forward and d/2 inverse cycles) and is nearly independent of the shot count. On the three-patch register of Sec. 9.8 with the `prob` estimator (640 runs per point, one point per worker) a point took ≈ 6 h at d = 16 and ≈ 11 h at d = 20 on the EPYC; there a three-patch instance can use about 16 threads, so CPU workers stack well for batches but deep points take long.

### 9.7 Cross-check with the `nn_qab` ensemble

The `nn_qab` benchmark ([`rcs/nn_qab/`](https://github.com/twobombs/thereminq-examples/tree/main/rcs/nn_qab)) scores ACE with the same estimator on circuits whose single-qubit θ is drawn with sin θ uniform (`--theta nnqab` here), on ACE's own torus registers with a twelve-gate two-qubit set. Running that θ ensemble on the Nighthawk window (CZ couplers, flat register) separates ensemble, geometry and placement:

```bash
python3 nighthawk_qrack.py run --backend ace --families fxeb --n 27 --depths 12 \
  --theta nnqab --instances 20 --variants 4/4:t3 4/4:untiled --ref-gpu --gpus 0 --out nnqab_cross.jsonl
```

| Configuration (d = 12) | Circuits | Mean XEB (plain mean ± instance-scatter se) | HOG |
|---|---|---|---|
| Nighthawk window, 4/4 tiled v3 | 20 | **0.435 ± 0.037** | 0.665 |
| Nighthawk window, 4/4 device placement | 20 | 0.153 ± 0.014 | 0.500 |
| `nn_qab` 26 qubits, B-to-B 5.5 (best series) | 100 | 0.241 | — |
| `nn_qab` 27 qubits, two patches, B-to-B 3.5 | 100 | 0.185 | — |
| `nn_qab` 27 qubits, three patches, B-to-B 2.0 | 100 | 0.111 | — |
| Nighthawk window, 4/4 tiled v2, Haar ensemble (Sec. 9.2; inverse-variance mean) | 3 | 0.182 | — |

Three observations. (i) The ensemble matters: at d = 12 the `nn_qab` outputs are far from Porter–Thomas (ideal XEB ≈ 16 against ≈ 1.5 for the Haar ensemble), and the same tiled layout scores about twice its Haar value. XEB values from the two ensembles are not comparable. (ii) Within the ensemble, placement alone raises the XEB by a factor of 2.85, and tiled ACE on the Nighthawk geometry exceeds every `nn_qab` register geometry by a factor ≥ 1.8; part of this may be the gate set (CZ only, against `nn_qab`'s set including SWAP-type gates that move amplitude across seams), which this run does not separate. (iii) With the device placement the HOG is at chance (0.500) while the XEB is positive: in a concentrated distribution, a sampler that reaches a few very heavy bitstrings raises the probability-weighted score without carrying signal across the rest of the distribution; the tiled sampler has both.

The effective bulk-to-boundary ratio of the placed qubits (bulk qubits used over seam qubits used; view 11) carries `nn_qab`'s B-to-B variable over to placed registers; for the five tiled layouts of Sec. 9.2 the d = 12 XEB (inverse-variance means) falls monotonically with it (8.0, 8.0, 5.8, 3.5, 0.7 → 0.182, 0.153, 0.136, 0.109, 0.091).

### 9.8 Mirror at the full register (61 qubits)

Mirror circuits need no reference, so ACE runs the paper's own proxy on the experiment's register: same 61 qubits, same pseudo-patched circuits, same estimator, no extrapolation. The first runs (Haar θ, shot estimator, device-geometry and untiled 4/4 layouts) fell below the shot floor by d = 6, with flips at d = 4 confined to the layout's seam column. The configuration below uses the three-patch register of Sec. 2.8 (75 sites, lrc 4 lrr 5, torus, the 11 exclusions of `mirror_nighthawk.py`, exact placement: 74 / 28 / 0, 9 boundary qubits), the `nn_qab` single-qubit ensemble and the `prob` estimator, 640 runs per point (10 inputs × 64 twirls), on a 96-thread EPYC (CPU):

```bash
python3 nighthawk_qrack.py run --backend ace --families mirror --sizes 61 --theta nnqab \
  --depths 4 6 8 10 12 16 20 --lrc 4 --lrr 5 --ace-torus --ace-transpile --ace-width 75 \
  --ace-exclude 74,59,44,73,69,54,68,67,64,63,62 --mirror-estimator prob --order shallow \
  --cpu --gpus 0-5 --per-gpu 1 --mem-budget-gb 300 --max-rss-gb 16 --out mirror61_dan_prob_cpu.jsonl
```

| d | i0 | i1 | i2 | weighted mean (`--summarize`) | plain mean | `ibm_phoenix` mirror | boundary CZs (cumulative, forward half) |
|---|---|---|---|---|---|---|---|
| 4 | 0.630 | 0.292 | 0.550 | 0.495 ± 0.007 | 0.491 | 0.214 | 11 |
| 6 | 0.364 | 0.260 | 0.388 | 0.338 ± 0.006 | 0.337 | 0.154 | 18 |
| 8 | 0.0126 | 0.0726 | 0.0461 | 0.0309 ± 0.0016 | 0.0438 | 0.109 | 24 |
| 10 | 0.0119 | 0.0590 | 0.0471 | 0.0305 ± 0.0016 | 0.0393 | 0.081 | 27 |
| 12 | 0.0108 | 0.0632 | 0.0452 | 0.0238 ± 0.0013 | 0.0397 | 0.062 | 27 |
| 16 | failed | 1.38 × 10⁻³ | 5.3 × 10⁻⁴ | 5.9 × 10⁻⁴ | 9.5 × 10⁻⁴ | 0.030 | 40 |
| 20 | 4.6 × 10⁻⁵ | 2.1 × 10⁻⁴ | pending | — | 1.3 × 10⁻⁴ (i0, i1) | 0.021 | 49 |

![61-qubit mirror, every instance](mirror61_dan_instances_d4-20.png)

- ACE lies above the device at d = 4–6 (about 2.2×), crosses it between d = 6 and 8, sits at 0.3–0.4× the device from d = 8 to 12 (instance 1 at 0.7–1.0× of the device points), and falls to 1/30–1/50 at d = 16 and about 1/160 at d = 20.
- The loss follows the number of CZs through a boundary, not depth. From d = 8 to 12 only three boundary CZs are added and the survival is flat; from d = 12 to 20, 22 more cost a factor of ≈ 310, about 0.26 nats per boundary CZ (i1: 0.29 over d = 12–16, 0.21 over d = 16–20). A single exponential in d is therefore a poor description: the summary's fit, 5.30 × 0.586^d, exceeds 1 at d = 0.
- Instance 0 is a different curve from d = 8 on (≈ 0.011–0.013 against 0.045–0.073), and it pulls the inverse-variance mean below the plain mean (Sec. 7.3).
- Points: d = 16 i0 failed with a C++ exception on CPU (cause not yet traced); d = 24 was dropped deliberately, since a d = 20 point already costs ≈ 11 h (d = 16 ≈ 6 h). 8 of the points ran under PyQrack 2.28.1, the others under the version before it (Sec. 10).
- The device points are the paper's Haar circuits; these are `nn_qab`-ensemble circuits on the same register, so the comparison is of shape, not one-to-one.

### 9.9 Outlook: how close can ACE come at 61 qubits

Across n = 27–29 the tiled v3 decay per replica coupler (a CZ through a seam replica) is nearly constant: b / replica couplers = 0.23/9, 0.27/13, 0.32/12 ≈ 0.025 per cycle. At matched size the device's error rate (2.25 × 10⁻³ per qubit per cycle) corresponds to b ≈ 0.06 at n = 27–29, so tiled ACE already loses fidelity 4–5 times faster per cycle than the hardware would on the same number of qubits. Extrapolated with that rate:

| Split of the 8×8 window | couplers between simulators | estimated b at n = 61 | simulator width |
|---|---|---|---|
| tiled 4/4 (five simulators) | ≈ 40 | ≈ 1.0 | ≤ 23 |
| three large tiles | ≈ 16 | ≈ 0.4 | ≈ 22–25 |
| two halves, one seam line | ≈ 8 | ≈ 0.2 | ≈ 31–33 |
| device | — | 0.137 | — |

No balanced cut of an eight-wide grid crosses fewer than ≈ 8 couplers, so the device's rate (≲ 5 crossing couplers at 0.025 each) is out of reach of any seam-partitioned layout that fits in memory; a two-way split could come within ≈ 1.5× of it, with two dense ≈ 32-qubit simulators (32–64 GiB each). These estimates rest on three sizes and three to five circuits per point, and b itself grows with depth; mirror runs at the full register test them directly (Sec. 9.8). More shots only lower the measurement floor, and post-selection on ideal probabilities is the spoofing of [[5]](#ref-5), not a fidelity gain.

Measured since: the three-patch register with 9 boundary qubits (Sec. 9.8) stays above the shot-limited runs to d = 20, but its decay is a staircase in the boundary-CZ count rather than one rate; the summary's single exponential gives b ≈ 0.53 against the ≈ 0.4 estimated above for three large tiles. A two-way split of the 61-qubit register needs a 35-qubit patch (256 GiB in fp32) and does not fit a GPU. The two-patch effect was measured instead on a 49-qubit sub-register (the largest that fits two patches on one 8 GB die: 55 sites, lrc 5 lrr 4, flat, patches 28 / 27, 4 boundary qubits, 64 / 15 / 0) against a three-patch split of the same qubits (75 sites, lrc 4 lrr 5, torus, 23 / 23 / 12, 6 boundary qubits, 58 / 21 / 0): mirror survival 0.760 / 0.706 at d = 4 / 6 against 0.206 / 0.234 (CPU, `nn_qab` θ, 20 runs per point).

### 9.10 Larger registers on one 8 GB GPU

How far each patch count reaches on one 8 GB die (fp32, all patches fully entangled, plus Qrack's compose buffer the size of the widest patch), with splits found by CP-SAT over row-major sub-registers of the 12 × 10 lattice:

| Patches | Logical qubits | Boundary qubits | Widest patch (incl. ancilla) |
|---|---|---|---|
| 2 | 49 | 4 | 28 |
| 3 (experiment's register) | 61 | 9 | 25 |
| 3 | 68 | 11 | 28 |
| 4, widths ≤ 27 | 87 | 16 | 27 |
| 4 (`nighthawk88`) | 88 | 17 | 28 |
| 4, disabled sites on the seams (`nighthawk88d`) | 88 | 11 | 28 |

![nighthawk88d](nighthawk88d_four_patch_seams.png)

Under clvk each worker sees one die, so six dies run six such registers in parallel. A first CPU check of the two 88-qubit registers (d = 4 and 6, 3 instances × 40 runs, inverse-variance means) does not separate them yet: 0.047 ± 0.009 / 0.0082 ± 0.0020 for `nighthawk88` against 0.030 ± 0.005 / 0.0073 ± 0.0018 for `nighthawk88d`, with instance scatter from 0.018 to 0.20 at d = 4. Following Sec. 9.8 the difference should appear at depth, where 13 fewer boundary CZs per cycle compound; the full run on six V340 dies is next (command in Sec. 2.9).

---

verification and simulation at runtime:

<img width="970" height="403" alt="image" src="https://github.com/user-attachments/assets/717d8289-2a12-4de4-892d-77d5a161f955" />
<img width="1114" height="327" alt="image" src="https://github.com/user-attachments/assets/b84b32ce-4fae-434b-b06a-62b38d32c4b2" />

## 10. Limitations

- `fxeb` needs an exact 2ⁿ reference and is limited to n ≲ 34–36; statements about n = 61 are extrapolations of b(N).
- The `full` family cannot be scored at 61 qubits; samples are stored for downstream use only.
- Patched circuits use the released 61-qubit partitions and run only at n = 61.
- `QrackAceBackend` is an approximate method whose layout, chunk numbering and seam treatment depend on the PyQrack version; record the version, `--lrc`/`--lrr`, torus setting and tiling version with every published number (all are in each record).
- Tiling optimises a proxy cost; its weights are calibrated on the 4/4 runs of Sec. 9 and are not claimed optimal.
- Three instances per point; instance scatter exceeds shot noise, and differences between configurations below ~2σ of the instance-scatter error are not resolved.
- The ACE-settings comparison ran on an updated PyQrack; compare within one PyQrack version, as ACE's seam treatment and layout can change between versions (the PyQrack version enters neither the config tag nor the records: note it with each run, or use a new `--out` after an upgrade). On the 61-qubit mirror (Sec. 9.8) moving to PyQrack 2.28.1 shifted d = 6 instances 1 and 2 by −0.007 and +0.005 on CPU alone.
- GPU results depend on the Qrack build (Sec. 8.7): use Qrack with the async fix or `QRACK_OCL_SYNC=1`, and check one point against the CPU on every new device or driver.
- The `--register` sub-registers have no device data; their statements are about ACE alone. `nighthawk88d` changes the circuit graph (32 disabled sites, 150 couplers), so its fidelity is not comparable one-to-one with `nighthawk88`'s.
- The v3/v2 comparison (Sec. 9.5) rests on 3 circuits per point; its effect size is approximate until the paired differences and the n = 30–32 points are in.
- Sections 9.2 and 9.4–9.5 report inverse-variance means from `run --summarize`; the plain means over circuits can be higher (Sec. 7.3) and should be taken from the graph tool for publication.
- arXiv:2609.28657 is a preprint (v1) and had not been peer reviewed at the time of writing.

---

## 11. References

All identifiers below were checked against the arXiv abstract pages and, where given, the published versions.

<a id="ref-1"></a>**[1]** T. Sedrakyan *et al.*, "Quantum computational advantage in random-circuit sampling on IBM superconducting quantum computers," [arXiv:2609.28657](https://arxiv.org/abs/2609.28657) (2026). — *The experiment, estimators (Eq. (1)–(2)), partitions, pseudo-patching, shot budgets.*

<a id="ref-2"></a>**[2]** S. Boixo *et al.*, "Characterizing Quantum Supremacy in Near-Term Devices," [arXiv:1608.00263](https://arxiv.org/abs/1608.00263); *Nature Physics* **14**, 595 (2018). — *Cross-entropy benchmarking as a fidelity proxy.*

<a id="ref-3"></a>**[3]** F. Arute *et al.*, "Quantum supremacy using a programmable superconducting processor," [arXiv:1910.11333](https://arxiv.org/abs/1910.11333); *Nature* **574**, 505 (2019). — *Linear XEB on 53 qubits; patch and elided verification circuits.*

<a id="ref-4"></a>**[4]** A. Morvan *et al.*, "Phase transitions in random circuit sampling," [arXiv:2304.11119](https://arxiv.org/abs/2304.11119); *Nature* **634**, 328 (2024). — *Regimes in which XEB tracks fidelity; weak-link model.*

<a id="ref-5"></a>**[5]** X. Gao, M. Kalinowski, C.-N. Chou, M. D. Lukin, B. Barak, S. Choi, "Limitations of Linear Cross-Entropy as a Measure for Quantum Advantage," [arXiv:2112.01657](https://arxiv.org/abs/2112.01657); *PRX Quantum* **5**, 010334 (2024). — *When XEB and fidelity diverge; spoofing by cutting weak links.*

<a id="ref-6"></a>**[6]** T. Proctor, K. Rudinger, K. Young, E. Nielsen, R. Blume-Kohout, "Measuring the Capabilities of Quantum Computers," [arXiv:2008.11294](https://arxiv.org/abs/2008.11294); *Nature Physics* **18**, 75 (2022). — *Circuit mirroring.*

<a id="ref-7"></a>**[7]** T. Proctor, S. Seritan, K. Rudinger, E. Nielsen, R. Blume-Kohout, K. Young, "Scalable randomized benchmarking of quantum computers using mirror circuits," [arXiv:2112.09853](https://arxiv.org/abs/2112.09853); *Phys. Rev. Lett.* **129**, 150502 (2022). — *Mirror-circuit RB.*

<a id="ref-8"></a>**[8]** J. Hines *et al.*, "Demonstrating Scalable Randomized Benchmarking of Universal Gate Sets," [arXiv:2207.07272](https://arxiv.org/abs/2207.07272); *Phys. Rev. X* **13**, 041030 (2023). — *Mirror RB with continuously parametrised single-qubit gates.*

<a id="ref-9"></a>**[9]** J. J. Wallman, J. Emerson, "Noise tailoring for scalable quantum computation via randomized compiling," [arXiv:1512.01098](https://arxiv.org/abs/1512.01098); *Phys. Rev. A* **94**, 052325 (2016). — *Pauli-frame twirling.*

<a id="ref-10"></a>**[10]** D. Strano, B. Bollay, A. Blaauw, N. Shammah, W. J. Zeng, A. Mari, "Exact and approximate simulation of large quantum circuits on a single GPU," [arXiv:2304.14969](https://arxiv.org/abs/2304.14969); *Proc. IEEE QCE 2023*, pp. 949–958. — *Qrack's exact and approximate simulation methods.*

<a id="ref-11"></a>**[11]** Y. Zhou, E. M. Stoudenmire, X. Waintal, "What limits the simulation of quantum computers?," [arXiv:2002.07730](https://arxiv.org/abs/2002.07730); *Phys. Rev. X* **10**, 041038 (2020). — *Approximate simulation with a controlled error per gate, matched against device fidelity.*

<a id="ref-12"></a>**[12]** I. L. Markov, A. Fatima, S. V. Isakov, S. Boixo, "Quantum Supremacy Is Both Closer and Farther than It Appears," [arXiv:1807.10749](https://arxiv.org/abs/1807.10749) (2018). — *Trading circuit fidelity for simulation cost; cost linear in target fidelity.*

**Software and data (not on arXiv):** [BlueQubitDev/rcs-nighthawk](https://github.com/BlueQubitDev/rcs-nighthawk) (including `contraction_cost/`) · [unitaryfund/pyqrack](https://github.com/unitaryfund/pyqrack) · [vm6502q/pyqrack-examples](https://github.com/vm6502q/pyqrack-examples) (upstream `nn_qab.py`) · [twobombs/thereminq-examples](https://github.com/twobombs/thereminq-examples) (`rcs/nighthawk`, `rcs/nn_qab`)
