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

We provide a reproducible, simulator-side companion to the 61-qubit random-circuit sampling (RCS) experiment of Ref. [[1]](#ref-1). The released circuits are regenerated bit-for-bit, the hardware bitstrings are re-scored against PyQrack ideal distributions, and the circuits are emulated on ideal qubits with an exact backend (a pipeline control) and with Qrack's approximate, seam-partitioned `QrackAceBackend` (ACE). For first-n truncations of the register (n ≤ 34) every ACE sample is scored by forward linear XEB against an exact 2ⁿ-amplitude reference. On the 8×8 Nighthawk window we find that ACE's native chunk numbering places most couplers across simulators, and that re-placing logical qubits onto ACE sites as compact tiles raises the forward XEB at n = 27, d = 8 from 0.008 to 0.32 at unchanged memory. Across ten layout/placement configurations the XEB is predicted best by the number of couplers across simulators and of simulators in use (rank correlations −0.80 and −0.84), and, among crossing-free tilings, by the number of logical qubits on seam sites. Even so, at depths where the output is close to Porter–Thomas (d ≥ 12), the tiled emulator's per-cycle decay at n = 27–29 (b ≈ 0.24–0.31) exceeds the device's b = 0.137 at n = 61, and grows with depth. Under the weaker single-qubit ensemble of the `nn_qab` benchmark (sin θ uniform), tiled ACE on the Nighthawk window reaches a mean XEB of 0.435 ± 0.037 at n = 27, d = 12 over 20 circuits, against 0.153 ± 0.014 with the device placement and 0.111–0.241 for `nn_qab`'s own register geometries (Sec. 9.7). At 61 qubits no XEB can be scored directly; the release's own contraction-cost estimates put one exact amplitude of the 36-cycle circuit at ≈ 10²² multiply-adds (Sec. 8.6). A shared-reference comparison of two tiling heuristics confirms the seam-qubit reading at n = 27–29: where the improved tiling halves the number of logical qubits on seam sites (n = 28, 8 → 4) the d = 12 XEB rises 3.6-fold, where the seam counts agree (n = 27) the two coincide (Sec. 9.5). Tiled ACE runs on NVIDIA's OpenCL stack and agrees with the CPU path, but resets AMD Vega 10 and Vega 20 devices under Mesa rusticl (Sec. 8.3).

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

There is **no noise model**. Any deviation from F = 1 under `run` is attributable to the simulator backend or to sampling, not to an assumed device error.

> **Version note.** This HOWTO describes `nighthawk_qrack.py` at 2805 lines and `nighthawk_graph.py` at 1282 lines. Earlier revisions contained a noise-model audit (`--preset r2`, `--stochastic`, `--edge-layers`); those flags no longer exist. Tiling version 2 (`--ace-tiling-version 2`) reproduces the config tags of runs made before tiling version 3 became the default.

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

---

## 3. Requirements

| Component | Notes |
|---|---|
| Python | ≥ 3.9 (uses `argparse.BooleanOptionalAction`) |
| numpy | required by every subcommand |
| matplotlib | `nighthawk_graph.py` only (Tk for the interactive viewer) |
| PyQrack | required by `hwxeb`, `run`, `selftest`, `aceplan`; **not** by `verify` or `seamgap`. The `ace` backend needs a PyQrack build that ships `QrackAceBackend`; exact references above 32 qubits need Qrack built in >32-qubit mode |
| Qrack shared library | if `/usr/local/lib/qrack/libqrack_pinvoke.so` exists, the script exports it as `PYQRACK_SHARED_LIB_PATH` |
| Qiskit | **not** required |
| GPU | OpenCL (select with `QRACK_OCL_DEFAULT_DEVICE`) or `--cpu`. No CUDA path is used. See Sec. 8.3 for ACE on GPU |

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
| `--ace-tiling-version` | `3` | `2` reproduces earlier tiled runs and their config tags |
| `--ace-gpu`, `--ace-host-pointer` | off | ACE on OpenCL; simulator states in host RAM (GTT). See Sec. 8.3 |
| `--theta` | `haar` | `nnqab`: sin θ uniform — a variant ensemble, not the released circuits |

**Sampling and scope:** `--depths`, `--K`, `--instances` (3), `--partitions`, `--shots` (`paper` = Appendix-D budgets for mirror/patched, 4096 for `fxeb`, 100 000 per pub for `full`; or an integer), `--twirls` (64), `--exact-probs`, `--sizes` (`27-34`, `20,27-36`), `--n`, `--pubs`, `--cache`, `--cpu`, `--ref-gpu` (exact `fxeb` reference on OpenCL).

**Shared reference.** `--variants 4/4:t3 4/4:t2 4/3:t3 …` (lrc/lrr, then `torus`, `untiled`, `tiled`, `t2`, `t3`) samples each `fxeb` point with every listed ACE configuration and scores all of them against **one** exact reference. Records, bitstrings and config tags stay per variant; only the claims are shared under a group tag. At n = 33–34, where the reference dominates the cost, this compares configurations for the price of one.

**Workers and memory.**

| Flag | Default | Meaning |
|---|---|---|
| `--gpus`, `--per-gpu` | none, `3` | launch `len(gpus) × per_gpu` workers through a claim-file queue |
| `--stagger`, `--init-gap` | `20`, `10` s | spacing of worker starts and of OpenCL initialisations |
| `--max-attempts`, `--retry-failed` | `2` | take-overs of a point whose worker died; clear failed marks |
| `--max-rss-gb` | off | per-worker resident-memory cap (watchdog marks the point failed and exits) |
| `--mem-budget-gb` | off | machine-wide budget shared by all workers and runs using the same ledger |
| `--mem-ledger` | `/tmp/nighthawk_mem_ledger` | reservation directory |

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

`ace_map` is a hash of the placement; the graph tool uses it to verify recomputed coupler splits. Mirror records carry `hits`, `shots_per_input`; patched records `patch_fidelity`, `patch_ideal_xeb`.

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

`--table FILE` writes every point as CSV (coupler splits recomputed for older records where the placement hash matches) and prints a per-(n, d) ranking and each configuration's b. In views 8–11, points are plain means over instances with the standard error from the instance scatter, which exceeds shot noise; `run --summarize` uses inverse-variance weights.

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

### 8.3 ACE on OpenCL

| Device | OpenCL stack | Exact reference (`--ref-gpu`) | ACE, device placement | ACE, tiled (`--ace-gpu --ace-tiling`) |
|---|---|---|---|---|
| Radeon Pro V340 (Vega 10) | Mesa rusticl / radeonsi | not tested | runs | **resets the device** (VRAM and `--ace-host-pointer` alike) |
| Radeon Pro VII (Vega 20) | Mesa rusticl / radeonsi | not tested | not tested | **workers abort (SIGABRT)** |
| NVIDIA CMP 50HX (TU102, 10 GB, link at Gen 1 x1) | NVIDIA OpenCL | runs | runs | **runs; agrees with CPU** |

On rusticl the fault surfaced at the first read-back after the circuit, so the offending operation is one of the queued cross-simulator operations of a tiled placement; host-pointer mode (states in host RAM via GTT) did not avoid it, so it is not a VRAM-capacity effect — tiled 4/4 simulators are at most 23 qubits wide. On the NVIDIA stack the v2-tiled n = 27 points reproduce the CPU run on the same circuits (d = 12, 16, 20: 0.146, 0.063, 0.033 on GPU against 0.158, 0.063, 0.024 on CPU; inverse-variance means, ±0.009).

GPU execution of ACE is not faster here: its simulators are small, its work is many short kernels with frequent read-backs and host-side seam logic, and workers sharing one card queue on it, whereas CPU workers run on separate cores. On the CMP 50HX an n = 27–28 `fxeb` point with both variants took ≈ 10 min per worker with three workers, an n = 29 point ≈ 13 min with two; the n = 27 reference alone took ≈ 10 s. The useful split is ACE on CPU workers and the reference on a GPU, unless CPU cores are the scarcer resource.

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

---

## 9. Results to date

All results below are clean-qubit ACE emulations of the released circuit ensemble (Haar single-qubit gates, release seeds), scored by forward XEB against an exact reference, 3 instances, 4096 shots unless noted. The device figure of merit for comparison is the paper's mirror fit F(d) = 0.326 × 0.8717^d, i.e. b_dev = 0.137 per cycle at n = 61. Values from `run --summarize` are inverse-variance means and may lie below the plain mean over circuits (Sec. 7.3); values from the graph tool are plain means. Each table states which.

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

Because both variants share circuits, the decisive statistic is the *paired* per-circuit difference v3 − v2, in which circuit-to-circuit variation cancels; the CSV from `nighthawk_graph.py --table` carries the per-instance values for it. Sizes n = 30–32 (seam differences 1, 2, 5) are pending on the CPU path; n = 32, with 11 seam qubits under v2 against 6 under v3, is expected to show the largest gap.

### 9.6 Cost

The run cost is dominated by the exact reference: about 6.2 × 10³ s for one n = 34, d = 20 point on a 96-thread EPYC host, 1–4 × 10² s per tiled 4/4 point at n = 27. Wider ACE simulators also cost time (tiled 2/7 ≈ 1.8 × 10³ s per point at n = 27). On the CMP 50HX (`--ace-gpu --ref-gpu`, both variants per point) a point took ≈ 10 min at n = 27–28 and ≈ 13 min at n = 29 per worker. Mirror circuits at n = 61 cost ≈ 0.65–4.5 × 10³ s per point untiled (d = 4–20); their cost grows with depth (every twirl runs d/2 forward and d/2 inverse cycles) and is nearly independent of the shot count.

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

### 9.8 Planned: tiled mirror at the full register

Mirror circuits need no reference, so ACE can run the paper's own proxy at n = 61: the first like-for-like comparison with `ibm_phoenix` (same register, circuits and estimator, no extrapolation). Untiled ACE fell to the shot floor by d = 6 (survival 0.01–0.08 at d = 4), with per-qubit flip rates confined to the layout's seam column; tiled mirror has not yet been run. The release's mirror depths are 4–20 in steps of 2 and 24–40 in steps of 4 (patched: 20–40; full-circuit samples only at d = 36); the device fit gives F(40) ≈ 1.3 × 10⁻³. With N = 3 instances × the Appendix-D budget, survival p is resolved to ≈ ±33 % down to p_min ≈ 9/N:

| d | shots / instance | total N | p_min | device F(d) | largest ACE b resolvable at d (prefactor 0.3) |
|---|---|---|---|---|---|
| 4, 8 | 30 000 | 90 000 | 1.0 × 10⁻⁴ | 0.188, 0.109 | — |
| 12 | 45 000 | 135 000 | 6.7 × 10⁻⁵ | 0.063 | 0.64 |
| 16 | 60 000 | 180 000 | 5.0 × 10⁻⁵ | 0.036 | 0.54 |
| 20 | 72 000 | 216 000 | 4.2 × 10⁻⁵ | 0.021 | 0.44 |
| 24 | 96 000 | 288 000 | 3.1 × 10⁻⁵ | 0.012 | 0.38 |
| 28 | 120 000 | 360 000 | 2.5 × 10⁻⁵ | 0.0070 | 0.34 |
| 32 | 120 000 | 360 000 | 2.5 × 10⁻⁵ | 0.0040 | 0.29 |
| 36 | 240 000 | 720 000 | 1.25 × 10⁻⁵ | 0.0023 | 0.28 |
| 40 | 360 000 | 1 080 000 | 8.3 × 10⁻⁶ | 0.0013 | 0.26 |

The device is resolved at every depth (≈ 2.6 % relative at d = 40); ACE is resolved to d = 40 only if its 61-qubit decay is ≲ 0.26 per cycle, against 0.23–0.32 measured by `fxeb` at n = 27–29. The run is therefore staged, both stages writing the same tag:

```bash
# stage 1 (27 points; ≈ 6–12 h with 6 CPU workers)
python3 nighthawk_qrack.py run --backend ace --families mirror --sizes 61 \
  --depths 4 6 8 10 12 14 16 18 20 --lrc 4 --lrr 4 --ace-tiling \
  --cpu --gpus 0 --per-gpu 6 --mem-budget-gb 300 --out mirror_tiled.jsonl
# stage 2 (15 points; ≈ 10–20 h), only if stage 1 is well above p_min at d = 20
python3 nighthawk_qrack.py run --backend ace --families mirror --sizes 61 \
  --depths 24 28 32 36 40 --lrc 4 --lrr 4 --ace-tiling \
  --cpu --gpus 0 --per-gpu 6 --mem-budget-gb 300 --out mirror_tiled.jsonl
```

Each worker needs ≈ 1.1 GiB. Larger budgets (`--shots 1000000`) lower p_min about threefold at the deep end at almost no extra cost, under their own config tag. In `fxeb`, by contrast, depth 40 is out of reach: at b ≈ 0.25 the XEB at d = 40 is ≈ 5 × 10⁻⁵, which would need ~10⁹ samples per circuit.

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
