# HOWTO: `rcs/nighthawk/nighthawk_qrack.py`

**A clean-qubit PyQrack companion to arXiv:2609.28657 (random-circuit sampling on IBM Nighthawk r2), with an exact-reference study of seam-partitioned approximate simulation**

| | |
|---|---|
| Script | [`rcs/nighthawk/nighthawk_qrack.py`](https://github.com/twobombs/thereminq-examples/tree/main/rcs/nighthawk) (ThereminQ examples), 5026 lines, 16 October 2026 |
| Companions | `nighthawk_graph.py` (harvest, views, CSV) · `nighthawk_cloverfield.py` (several chips, one virtual chip) · [`rcs/nn_qab/`](https://github.com/twobombs/thereminq-examples/tree/main/rcs/nn_qab) |
| Methodology | [`METHODOLOGY.md`](METHODOLOGY.md): every procedure behind the code and the experiments |
| Data | [`BlueQubitDev/rcs-nighthawk`](https://github.com/BlueQubitDev/rcs-nighthawk), the authors' release |
| Paper | Sedrakyan et al., [arXiv:2609.28657](https://arxiv.org/abs/2609.28657) (preprint, v1) [[1]](#ref-1) |
| Simulator | [Qrack / PyQrack](https://github.com/unitaryfund/pyqrack): `QrackSimulator` (exact), `QrackAceBackend` (ACE, approximate) [[10]](#ref-10) |
| Acceleration | OpenCL or CPU. **No CUDA.** |

---

## Overview

BlueQubit ran 61-qubit random circuits on IBM's `ibm_phoenix` and estimated their fidelity from two proxies, patched XEB and mirror survival. This script runs the released circuits on ideal qubits in PyQrack, either exactly or with ACE, Qrack's seam-partitioned approximate simulator. There is no noise model: every loss of fidelity comes from the simulation method, which is the object of study.

| Tool | Does |
|---|---|
| `verify` | regenerates every released circuit bit for bit and compares it with the released QASM |
| `selftest` | conventions and exactness checks |
| `hwxeb` | re-scores the measured hardware bitstrings with PyQrack's ideal distributions |
| `run` | emulates mirror, patched, full and forward-XEB (`fxeb`) circuits, exactly or with ACE |
| `aceplan` | lists every ACE layout of a register |
| `layoutsearch` | CP-SAT search for a register and its ACE placement, verified and tested |
| `danmodel` | Dan Strano's ACE XEB calculator, alone or against measured points |
| `seamgap` | data-only analysis of the release |
| `nighthawk_cloverfield.py` | chips in isolated workers, joined by a coordinator (the *loom*) |
| `nighthawk_graph.py` | harvests every run, views 1–11, CSV export |

No XEB can be scored on the full 61-qubit circuits (§5.7), so every 61-qubit statement is a proxy measurement or an extrapolation.

### Key results

| Finding | Numbers | § |
|---|---|---|
| Placement matters more than layout | n = 27, d = 8 forward XEB: 0.008 device placement → 0.321 tiled, same memory | 6.1 |
| A seam costs fidelity every cycle, and no ACE setting removes it | decay b = 0.24–0.31 per cycle at n = 27–29 against the device's 0.137 at n = 61 | 6.2 |
| At 61 qubits the loss follows CZs through a boundary, not depth | three patches: 2.3× the device at d = 4–6, 1/160 at d = 20; ≈ 0.26 nats per boundary CZ | 6.5 |
| Two patches on IBM's own register and circuits match the device at shallow depth | Haar θ, d = 4 / 6: 0.226 / 0.128 against 0.214 / 0.154 | 6.8 |
| A register with a 3-qubit waist keeps the boundary nearly closed | 45-qubit version, d = 8 / 12 / 20: 0.90 / 0.88 / 0.12 against the device's 0.109 / 0.062 / 0.021; classically easier register | 6.8 |
| Dan's calculator holds at the edge of its data and overestimates beyond it | 28 qubits: fxeb within 10 %; 36–45 qubits at d = 20: 0.002–0.05 of the prediction | 6.9 |
| 88 qubits fit one 8 GB GPU; disabled sites on the seams shrink the boundary | boundary 17 → 11 qubits; `nighthawk88e` 10–20× `nighthawk88d` at d = 4–6 | 6.7 |
| A ghost link carries entanglement between chips | loom-only error, d = 8 / 12 / 16: ghost 1.000 / 0.989 / 0.656, mean-field 0.773 / 0.773 / 0.067 | 3.5 |
| Qrack's OpenCL engine had three host-memory defects | fixed without serialising dispatch; GPU reproduces CPU | 5.3 |

---

## 1. Installation and requirements

```bash
git clone https://github.com/twobombs/thereminq-examples
cd thereminq-examples/rcs/nighthawk
git clone https://github.com/BlueQubitDev/rcs-nighthawk     # circuits, layout, hardware counts
pip install numpy pyqrack matplotlib ortools qiskit
```

| Component | Needed for |
|---|---|
| Python ≥ 3.9 | everything (`argparse.BooleanOptionalAction`) |
| numpy | every subcommand |
| PyQrack | `hwxeb`, `run`, `selftest`, `aceplan`, `danmodel`; not `verify` or `seamgap`. ACE needs a build with `QrackAceBackend`; exact references above 32 qubits need Qrack built for > 32 qubits |
| OR-Tools | the `exact` placement engine, `--register` presets, `layoutsearch` |
| Qiskit | `--ace-transpile` |
| matplotlib | `nighthawk_graph.py` (Tk for the viewer) |
| GPU | OpenCL (`QRACK_OCL_DEFAULT_DEVICE`) or `--cpu`; use the Qrack build of §5.3 |

Every PyQrack script in the repository carries:

```python
QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):
    os.environ["PYQRACK_SHARED_LIB_PATH"] = QRACK_LIB_PATH
```

The release is found via `--repo`, `$NIGHTHAWK_REPO`, or a `data/layout.json` / `rcs-nighthawk/` in the working directory, next to the script, or up to three parents up.

| Release path | Used by |
|---|---|
| `data/layout.json` | all: qubit map, matchings, schedule, partitions, seeds, depths, mirror inputs |
| `data/circuits/manifest.json`, `data/circuits/**.qasm` | `verify`, `selftest`, `seamgap` |
| `data/counts/patched_K{K}_d{d}.npz`, `data/counts/mirror_survival.json` | `hwxeb`, `selftest`, `danmodel` |
| `data/results/patch_xeb.json`, `data/results/fidelity_vs_depth.json` | `hwxeb`, `run --summarize`, `selftest`, `seamgap`, graph tool |

| Environment variable | Effect |
|---|---|
| `QRACK_OCL_DEFAULT_DEVICE`, `QRACK_QUNITMULTI_DEVICES`, `QRACK_QPAGER_DEVICES` | device index, and the devices Qrack may spread units and pages over; the launcher pins all three per worker. For single-process runs set all three: Qrack's default device need not be 0 |
| `QRACK_MAX_CPU_QB`, `QRACK_MAX_PAGING_QB` | qubit ceilings of the CPU engine and QPager; raised automatically for patches over 28 qubits (§5.5) |
| `QRACK_MAX_ALLOC_MB` | per-device allocation ceiling (§5.4) |
| `QRACK_OCL_SYNC` | Qrack ≥ 10.13.2: `1` finishes the queue after every dispatch (§5.3) |
| `OCL_ICD_VENDORS` | for CPU runs, an empty directory (`/tmp/no-icd`): no OpenCL platform, no GPU memory taken |
| `NIGHTHAWK_REPO` | release location |
| `NIGHTHAWK_TILE_CACHE` | tiling cache (default `nighthawk_tiles.json` next to the script) |
| `NIGHTHAWK_MEM_LEDGER` | memory-reservation directory shared by all runs (default `/tmp/nighthawk_mem_ledger`) |
| `NIGHTHAWK_REGISTERS` | extra register files, besides `nighthawk_registers.json` next to the script |

### 1.1 Quick start

```bash
python3 nighthawk_qrack.py verify                                   # circuits vs released QASM
python3 nighthawk_qrack.py selftest                                 # ≈ 1 min
python3 nighthawk_qrack.py hwxeb --cpu --cache patch_cache          # the paper's numbers, PyQrack reference
python3 nighthawk_qrack.py aceplan --n 27 --tiling                  # ACE layouts, ranked
for L in "4 4" "2 7" "1 7" "4 3" "2 2"; do set -- $L                # layouts against the exact reference
  python3 nighthawk_qrack.py run --backend ace --families fxeb --n 27 --depths 8 12 \
    --lrc $1 --lrr $2 --ace-tiling --cpu --out tile_${1}_${2}.jsonl &
done; wait
python3 nighthawk_qrack.py run --backend ace --families fxeb --sizes 27-34 --depths 12 16 20 \
  --variants 4/4:t3 4/4:t2 --gpus 0 --per-gpu 4 --mem-budget-gb 280 --out big_var.jsonl
python3 nighthawk_graph.py --table harvest.csv                      # everything, side by side
```

---

## 2. Method

### 2.1 Circuits and estimators

Each cycle applies a Haar-random single-qubit gate to every qubit (`rz(φ)·rx(θ)·rz(λ)`), then CZs on one of four coupler colourings `A, B, C, D` in fixed order. Linear XEB as a fidelity proxy follows [[2]](#ref-2), [[3]](#ref-3); its limits are analysed in [[4]](#ref-4), [[5]](#ref-5).

| Family | Circuit | Score |
|---|---|---|
| `patched` | every CZ crossing a K-patch partition removed; patches independent, n = 61 only | F = Π_r mean_s [(2^{n_r} p_r(x_r) − 1) / (2^{n_r} Σ p_r² − 1)], delta-method σ (Eq. (1)–(2) of [[1]](#ref-1)) |
| `mirror` | U for d/2 cycles, then U† for d/2: depth d counts both halves. The forward half is pseudo-patched: each cycle drops the CZs crossing one of the five K = 3 boundary sets, rotating ([[6]](#ref-6)–[[8]](#ref-8)) | survival of the prepared input string |
| `fxeb` | unpatched forward circuit on the first n qubits | XEB = (N · mean_s p(x_s) − 1) / (N Σ p² − 1), N = 2^n, against an exact reference; sums in float64 |
| `full` | unpatched forward circuit, release seed | stored only |

`--theta nnqab` replaces the Haar θ with the `nn_qab` ensemble (sin θ uniform): a variant, not the released circuits.

**Pauli-frame twirling** [[9]](#ref-9) surrounds each CZ with random Paulis and their propagated correction, an identity for ideal gates. On ACE it randomises the coherent seam error, so the forward and inverse halves of a mirror cannot cancel it. It is off on the exact backend and in `fxeb`.

**Mirror estimators.** `shots` samples at the paper's budgets. `prob` takes, per run, the probability ACE returns the prepared string (chain rule with forced measurement): no shot noise, no shot floor. A forced outcome below 10⁻⁶ counts the run as survival 0 and rebuilds the backend (`prob_errors` in the record).

### 2.2 ACE: seams, simulators, placement

`QrackAceBackend` splits a rectangular register into bulk *patches*, each held exactly by its own dense simulator, joined by *seam* sites whose qubits are replicated in both neighbouring patches (plus, with crossbars, a shared boundary simulator). `--lrc`, `--lrr` (`long_range_columns`, `long_range_rows`) and `is_torus` set the layout.

| CZ class | Definition | Treatment |
|---|---|---|
| exact | both ends bulk qubits of one simulator | applied exactly |
| replica | a seam qubit involved, ends share a simulator | applied on the replica, reconciled at the seam |
| cross | ends share no simulator | approximated across simulators |

A patch's width is its bulk plus every seam replica it carries plus one error-detection ancilla, and its memory is 8 bytes × 2^width (§5.1). ACE numbers patches along a folded chain, so on the device grid a patch is a set of row strips. With the device placement most vertical couplers become cross couplers, which is why the placement engines exist:

| Engine | Flag | Method |
|---|---|---|
| tiling v3 (default tiling) | `--ace-tiling` | deterministic annealing of 20 × cross + 1 × replica + 6 × (qubits on seams) + 2 × (simulators with bulk); three restarts; cached in `nighthawk_tiles.json` |
| tiling v2 | `--ace-tiling-version 2` | 10 × cross + 1 × replica, one run; reproduces older config tags |
| exact (default transpile engine) | `--ace-transpile` | CP-SAT: sites grouped by the simulators holding them; lexicographic: fewest seam qubits, even patches, fewest seam couplers; proven optimal |
| anneal | `--transpile-engine anneal` | the tiling cost on the transpile register |
| vf2 | `--transpile-engine vf2` | Qiskit VF2Layout, bounded search |
| joint | `layoutsearch` (§3.4) | register and placement in one CP-SAT model |

Placements are routing-free: every circuit coupler is a coupler of the register, so the released gates, twirling and estimators apply unchanged. A coupler joining two patches that share no seam is refused unless `--transpile-allow-cross` (token `xok`). `--ace-exclude` removes sites from the coupling map, and `--transpile-max-width W [--transpile-max-wide N]` caps patch widths (incl. ancilla) for the exact engine.

### 2.3 Registers

`--register NAME` replaces the release's register with a sub-register of the 12 × 10 Nighthawk lattice. The circuits are built by the release's own code and seed: `rect_matchings`, `remove_qubits`, `make_patches(seed=2025, balance_tol=1, top_n=5)` for the K = 3, 4 partitions, and the release's input-string recipe (selftest [9]). Registers have no device data; they run `mirror`, and `fxeb` up to 36 qubits. Each carries a stored ACE preset (geometry, caps, placement).

| Register | Qubits | Region | ACE geometry | Patch widths incl. ancilla | Boundary qubits | exact / replica / cross | Couplers through a boundary | Dense memory |
|---|---|---|---|---|---|---|---|---|
| `release` (three-patch preset, §6.5) | 61 | the experiment's 8 × 8 window | 75 sites, lrc 4 lrr 5, torus, 11 exclusions | ≤ 25 | 9 | 74 / 28 / 0 | 28 | 0.50 GiB |
| `nighthawk88` | 88 | rows 0–9 × cols 1–9, broken qubits used | 102 sites, lrc 4 lrr 5 | 28 / 27 / 27 / 27 | 17 | 102 / 55 / 0 | 55 per A–D cycle (14 / 14 / 14 / 13) | ≈ 7 GiB |
| `nighthawk88d` | 88 | whole lattice, 32 sites disabled (17, 55, 62 + 29 corners); every row and column one run; ring seams | 114 sites, lrc 4 lrr 5 | 28 / 23 / 25 / 27 | 11 | 108 / 42 / 0 | 42 (11 / 11 / 10 / 10) | ≈ 5.3 GiB |
| `nighthawk88e` | 88 | as 88d, seams moved by `layoutsearch` | 114 sites, lrc 4 lrr 5 | 25 / 28 / 25 / 25 | 11 | 112 / 36 / 0 | 36 (8 / 9 / 10 / 9) | ≈ 4.8 GiB |
| `nighthawk61r2` | 61 | the release's own qubits: identical mirror circuits, angles, inputs | 77 sites, lrc 5 lrr 6 | 35 / 34 (34 / 33 without error detection) | 6 (proven minimum, balanced) | 83 / 19 / 0 | 19 | 384 GiB (192) |
| `nighthawk61w` | 61 | rows 0–10 × cols 2–7 minus 17, 52, 55, 57, 62: a 3-qubit waist on row 5 | 75 sites, lrc 8 lrr 4 | 33 / 33 | 3 | 93 / 7 / 0 | 7 | 128 GiB |
| `nighthawk2p28` … `nighthawk2p57` | 28–57 | the release's first n qubits | §6.9 | §6.9 | 3–5 | §6.9 | 11–18 | 2 MiB – 64 GiB |

The 88-qubit, 61-qubit and ladder registers were found with `layoutsearch`. In `nighthawk88d` all four patches meet at broken qubit 55. Without the one-run-per-row constraint the optimum is four patches joined by three single qubits: nearly independent circuits.

### 2.4 Dan Strano's ACE XEB calculator

The calculator sheet of Dan Strano's ACE spreadsheet predicts XEB from three counts and the depth. It was fitted on `nn_qab` circuits of 10–27 qubits with 2–3 patches (PyQrack 2.28.2, error detection on, torus, SDRP off):

```
x       = (bulk / (boundary · patches))^(2/3)
XEB(12) = tanh(0.1653638846 · x − 0.0378757892)
XEB(d)  = XEB(12) ^ ((d / 12)^1.60569)
```

The depth exponent p = 1.60569 is the "gain" exponent his zero-noise extrapolation needs (`qem/richardson_nn_hamming_weight.py`): unitary folding by λ multiplies depth by λ, so the Richardson fit is done in λ^p. `danmodel` reproduces his worked example exactly (52 bulk, 7 boundary, 2 patches, d = 36 → 0.0019774).

### 2.5 Relation to other approximate methods

Throughout, the figure compared with the device is a per-cycle decay b = −d ln F / dd, as in matching a truncated simulation's error per gate to a device [[11]](#ref-11). The exact alternative to ACE's seams is Schrödinger–Feynman path summation across a cut, whose cost scales with the fidelity kept [[12]](#ref-12). Two CZs cross the best bipartition of this lattice per cycle, so paths grow as 2^(2d): a shallow-depth reference only.

---

## 3. Commands

### 3.1 `verify`, `selftest`, `hwxeb`, `aceplan`, `seamgap`

`verify` rebuilds every circuit as `rcs/circuits.py` does: SplitMix64 hash of (seed, instance, cycle, qubit, k) → (φ, θ, λ); seeds 2025 + k·1000003 (full, patched) and 2025 with instance k (mirror); all families and partitions, full circuits at d = 4, 8, …, 40. Angles are compared modulo 2π within `--tol` (1e-9 rad) and CZ counts against `manifest.json`. It exits 0 only if every circuit is identical.

| selftest | Check | Pass |
|---|---|---|
| 1 | angle hash + pseudo-patch rotation vs `mirror/d08_instance1.qasm` | identical |
| 2 | `u_angles` → Qrack `u` equals the Haar matrix | overlap within 1e-5 of 1 |
| 3 | exact mirror, first `--n` qubits, d = 12, 10 inputs | \|1 − F\| < 1e-4 |
| 4 | twirl is an identity (exact) | \|1 − F\| < 1e-4 |
| 5 | PyQrack patch ideal XEB vs the release | max diff < 1e-3 |
| 6 | Eq. (1)–(2) on the hardware counts vs the release | within 0.05 σ |
| 7 | ideal samples score F = 1 | within 4 σ |
| 8 | ACE register geometry | grid, torus, nearest-neighbour couplers |
| 9 | `--register` recipe on the release's own register | inputs and partitions identical |
| 10 | `layoutsearch` joint model vs the exact engine, release register, 75 sites | same boundary count, no cross coupler |

| Command | Does |
|---|---|
| `hwxeb` | ideal patch distributions with `QrackSimulator` (renormalised in float64, cacheable with `--cache`), Eq. (1)–(2) per (K, d) next to the release's values, mirror refit, F(36), error per qubit per cycle |
| `aceplan` | every distinct ACE layout of the first `--n` qubits (`--torus any\|true`), with cross and seam couplers, seam qubits, widths, worst-case memory; `--tiling` scores after tiling, `--max-width W` marks `run --ace-max-width W`'s pick |
| `seamgap` | NumPy only: CZ counts per family and depth, the fit at d = 36 and at the gate-count-equivalent depth, corrections for missing CZs, ln F = a + β·cycles + γ·CZ, K = 3 vs K = 4 ratio and its ε_CZ, per-cycle error budget |

### 3.2 `run`

| Flag | Default | Meaning |
|---|---|---|
| `--backend` | `ace` | `exact` or `ace` |
| `--families` | `mirror,patched` | comma list of §2.1 families |
| `--register` | `release` | a register of §2.3 with its ACE preset |
| `--theta` | `haar` | `nnqab`: variant ensemble |
| `--depths`, `--instances`, `--K`, `--partitions` | release's, 3, 3 4, all 5 | scope |
| `--sizes`, `--n` | — | first-n truncations, e.g. `27-34`, `20,27-36` |
| `--shots` | `paper` | Appendix-D budgets for mirror and patched, 4096 for `fxeb`, 100 000 per pub for `full`; or an integer |
| `--twirls`, `--exact-probs`, `--pubs`, `--cache` | 64, on | twirls per input on ACE; exact patch probabilities; full-circuit pubs; patch cache |
| `--mirror-estimator` | `shots` | `prob`: §2.1 |
| `--lrc`, `--lrr`, `--ace-torus` | 4, 4, off | ACE layout; `auto` = `nn_qab.py`'s two-patch rule |
| `--geometry` | `manual` | `nnqab`: rows whole, two patches, highest bulk-to-boundary ratio, torus |
| `--ace-max-width W`, `--ace-torus-search`, `--ace-layout` | —, `flat`, `grid` | fewest cross couplers with every simulator ≤ W; layouts it may use; `strip` = smallest R × 8 strip |
| `--ace-tiling`, `--ace-tiling-version` | off, 3 | §2.2 |
| `--ace-transpile`, `--ace-width`, `--ace-exclude`, `--transpile-engine` | off, —, —, `exact` | §2.2 |
| `--transpile-allow-cross`, `--transpile-max-width`, `--transpile-max-wide` | off | §2.2 |
| `--transpile-seed`, `--transpile-trials`, `--transpile-calls` | 0, 2000, 2·10⁶ | VF2 seed and bounds |
| `--ace-boundary-rep`, `--no-ace-error-detection`, `--no-ace-crossbars` | off, on, on | ACE switches; non-defaults enter the config tag |
| `--variants SPEC …` | — | several ACE configurations scored against one exact reference (below) |
| `--cpu`, `--ace-gpu`, `--ref-gpu`, `--ace-gpu-shots`, `--ace-host-pointer` | — | engine placement (§5.2) |
| `--gtt GB`, `--vram-gb` | off, 8 | VRAM oversubscription (§5.4) |
| `--swap-gb GB` | off | CPU runs larger than RAM (§5.5) |
| `--gpus`, `--per-gpu` | —, 3 | `len(gpus) × per_gpu` workers through a claim-file queue |
| `--stagger`, `--init-gap` | 20 s, 10 s | spacing of worker starts and OpenCL inits |
| `--max-attempts`, `--retry-failed` | 2 | take-overs of a dead worker's point; clear failed marks |
| `--max-rss-gb` | off | per-worker memory cap: the point is marked failed |
| `--mem-budget-gb`, `--mem-ledger` | off, `/tmp/nighthawk_mem_ledger` | machine-wide budget shared by all runs (§5.1) |
| `--order` | `deep` | `shallow`: cheapest points first |
| `--gpu-isolation` | `auto` | every GPU worker sees one OpenCL device; required for clvk |
| `--status`, `--summarize`, `--pack` | — | point list with progress and ETA; summary tables; rebuild release-format files |

**Shared reference.** `--variants 4/4:t3 4/4:t2 …` samples each `fxeb` point with every listed configuration and scores all of them against one exact reference. The tokens are lrc/lrr, then `torus`, `untiled`, `tiled`, `t2`, `t3`, `rep`, `noed`, `noxbar`, `tx` (exact), `txa` (anneal), `vf2`, `w<N>`, `ex<a+b+…>`, `xok`. Records stay per variant.

**Resumability.** A configuration is tagged `<backend>-<sha1[:8]>` over backend, shots, twirls, ACE parameters, tiling version and exact-probs mode, not the register size, so sweeps extend and resume per point. `--gtt` and `--swap-gb` do not change the tag. Bitstrings are written before the JSONL record, and the claim is marked done last. To drop a pending point, write `failed: …` into its claim file under `<out>_shots/<cfg>/claims/`.

### 3.3 `danmodel`

| Mode | Command | Output |
|---|---|---|
| prediction | `danmodel [--registers …] [--depths …]` | bulk, boundary, patches, x and predicted XEB per register and depth; default the two-patch ladder, d = 4 … 36 |
| comparison | `danmodel --in run.jsonl …` | every measured mirror or `fxeb` point next to the prediction at its own seam and patch counts; plain mean over instances, worker files included |

Both modes add `ibm_phoenix`'s mirror survival at n = 61 (`--no-device` to omit) and write CSV with `--csv`.

### 3.4 `layoutsearch`

One CP-SAT model chooses n sites of the 12 × 10 lattice and an ACE group (a patch's bulk or a seam) for each:

| Step | |
|---|---|
| propose | group capacities, ACE's coupling map, width caps, connectivity, region rule; lexicographic `--objective`; `--candidates` with no-good cuts at `--diversity` boundary sites; `--hint` warm start |
| verify | each candidate rebuilt through `RegisterLayout`: one qubit per site, no cross coupler, widths within caps |
| test | `run --register <candidate> --mirror-estimator prob` at `--test-depths`; score = geometric mean of the plain-mean survival |
| select | ranked by score (by the objective without tests); `--save-best NAME` writes `nighthawk_registers.json` |

| Flag | Default | Meaning |
|---|---|---|
| `--n` | required | logical qubits |
| `--geometries`, `--patches`, `--torus`, `--max-geometries` | `auto`, —, `flat`, 4 | `SIZE:LRC:LRR[:t]`, or every geometry of n … n+40 sites with that many patches |
| `--max-width`, `--max-wide` | 28, 1 | patch width cap incl. ancilla, patches allowed at the cap |
| `--region` | `rect` | `rect` (holes only at broken qubits plus `--trim` border sites), `convex`, `free`, `fixed` |
| `--fixed-register`, `--window` | — | with `fixed`: a register's sites, `release`, or `release:N` (the release's first N qubits) |
| `--broken`, `--min-boundary` | `disable`, 0 | broken qubits 17, 55, 62 unused or used; floor on boundary qubits |
| `--objective` | `seam,bcz,couplers` | order from `seam` (boundary qubits), `bcz` (couplers through a boundary), `cz` (circuit CZs through a boundary), `couplers` (kept), `balance` |
| `--cz-weights` | — | `mirror[:D,…]`: weight couplers by the CZs the register's mirror circuits apply on them (depths 4–20 by default); `cz` replaces `bcz` |
| `--candidates`, `--diversity`, `--hint` | 2, 4, — | |
| `--cpsat-time`, `--workers` | 300 s, ≤ 16 | a FEASIBLE result is the best found, not proven |
| `--test-depths`, `--test-instances`, `--test-twirls`, `--test-jobs`, `--theta`, `--cpu` | `4 8`, 1, 4, 1, `nnqab`, off | the circuit test; empty `--test-depths` skips it |
| `--gtt`, `--vram-gb` | — | judge candidates by the die split (§5.4) |
| `--outdir`, `--prefix`, `--save-best` | `layoutsearch`, `ls<n>`, — | outputs |

`convex` and `free` regions let the solver join patches through one- or two-site necks, i.e. nearly separate circuits that survive a mirror trivially. `rect` forbids necks except where trimmed border sites meet a seam. The solver also puts everything in one patch when the width cap allows it: cap the widest patch below n.

| Tool check (CPU) | Result |
|---|---|
| release register, 75 sites, against the exact engine (selftest [10]) | both 8 boundary qubits, proven optimal, ≈ 30 s; 26 vs 29 couplers through a boundary |
| n = 88, `convex`, warm start from `nighthawk88d` | 11 boundary qubits kept, 42 → 36 couplers through a boundary (`nighthawk88e`) |
| n = 49, two vs three patches, `rect --trim 2`, d = 4, 6 | 3 vs 6 boundary qubits, score 0.94 vs 0.59 |
| `--save-best` round trip | stored placement applied, survival identical (1.000, 0.8867) |
| balanced two-way split of the release register | 6 boundary qubits proven minimal; 128 circuit CZs through the boundary over d = 4–20 (196 for three patches) |

### 3.5 `nighthawk_cloverfield.py`

The cloverfield puts two or more chips side by side, joined by inter-chip couplers, as IBM plans with l-couplers: one register, one circuit, one survival.

| Part | Role |
|---|---|
| chip worker | one isolated process per chip and device; its register's ACE preset (or an exact simulator); executes gate batches and answers marginal and probability queries over a pipe |
| loom | the coordinator; builds no simulator (OpenCL hidden); generates the field's circuit by the release's recipe and does all IO between chips; ≈ 45 MiB |
| `--loom ghost` (default) | each chip holds a ghost of every partner its links touch; single-qubit gates and twirls go to every copy; the link CZ is exact on both chips; the partner's other CZs reach the ghost as a mean-field phase π·P(1). Records carry `ghost_drift` |
| `--loom mean`, `--loom sample` | mean-field phases only (chips in a product state); Z kicks drawn from P(1) |
| estimator | `prob` per chip; the field's survival is the product |

Options: `--chips` (`nighthawk88e*2`, `release,nighthawk88d`, `rect:r0-r1,c0-c1`, any register), `--arrangement RxC`, `--links K|all`, `--link-rows`, `--link-cols`, `--loom`, `--ghost-max-width`, `--gtt`, `--vram-gb`, `--backend`, `--chip-ace`, `--cpu`, `--ace-gpu`, `--gpus`, `--chips-per-gpu`, `--theta`, `--twirls`. Subcommands: `plan` (links, widths, memory; `--png`), `run`, `linksearch` (survival against link count), `layoutsearch` (passed to nighthawk_qrack), `selftest`.

| selftest | Check | Result |
|---|---|---|
| 1 | field `nighthawk88e*2`, 4 links: colours are matchings, deterministic cuts | 176 qubits, 300 couplers |
| 2a | one chip, no links: loom op stream vs nighthawk_qrack's own calls | 20 618 ops, identical |
| 2b | the same through a worker process | 0.375971 vs 0.375882 (Qrack's RNG unseeded) |
| 3 | woven CZ on \|+⟩ against \|0⟩, \|1⟩, \|+⟩ | ⟨X⟩ = +1, −1, 0 |
| 4 | CZ·CZ = I across a link | ghost +1; mean-field −1 |
| 5 | loom error alone: two exact 9-qubit chips, 2 links, d = 8 / 12 / 16 | no links 1 / 1 / 1; mean-field 0.773 / 0.773 / 0.067; ghost 1.000 / 0.989 / 0.656 |

---

## 4. Outputs

`fxeb` record from a `--variants` run (values illustrative):

```json
{"cfg": "ace-fd29e9a4", "n": 30, "family": "fxeb", "depth": 12, "instance": 1, "theta": "haar",
 "fidelity": 0.0689, "se": 0.0085, "xeb_linear": 0.104, "ideal_xeb": 1.39, "hog": 0.494, "shots": 4096,
 "ace_seconds": 41.3, "ref_seconds": 1142.2, "ref_shared": 2, "ace_widths": [23, 23, 20, 19, 16],
 "ace_couplers": {"exact": 34, "replica": 13, "cross": 0}, "ace_seam_used": 5, "ace_sims_used": 2,
 "ace_tiling": true, "ace_tiling_version": 3, "ace_map": "…", "shots_file": "…"}
```

Mirror records add `survival`, `hits` and `shots_per_input`; patched records add `patch_fidelity` and `patch_ideal_xeb`. `ace_map` hashes the placement. Bitstrings land per point under `<shots-dir>/<cfg>/points/n{n}/` and are merged into the release's own layout under `<cfg>/release/` (`counts/…`, `samples/…`), so BlueQubit's scripts read them unchanged; bit q is logical qubit q.

**Means.** `run --summarize` reports inverse-variance means. These weight each instance by its shot noise, which grows with its own XEB, so they sit below the average over circuits when instances differ. The graph tool and `danmodel` report plain means with the instance-scatter error, which is the quantity to quote.

| Run (§6.3) | inverse-variance mean | plain mean |
|---|---|---|
| tiled, 20 circuits | 0.343 | 0.435 ± 0.037 |
| untiled, 20 circuits | 0.125 | 0.153 ± 0.014 |

| `nighthawk_graph.py` view | Content |
|---|---|
| 1–7 | device fidelity, mirror decay, b(N), bitstring diagnostics, release data, run cost |
| 8, 9 | `fxeb` XEB against depth (per n) and against n (per depth) |
| 10 | per-cycle decay b(N) per configuration, extrapolated to 61 |
| 11 | XEB at one (n, d) against coupler classes, seam qubits, simulators, widest simulator, with rank correlations |

Views 8–11 and `--table FILE` (CSV of every point) show one ensemble at a time (`--theta haar|nnqab|all`), since XEB values of different ensembles are not comparable.

---

## 5. Hardware and memory

### 5.1 Memory rule

| Item | Size (fp32) |
|---|---|
| dense patch of w qubits | 8 · 2^w bytes: 1 GiB at 27, 2 at 28, 4 at 29, 8 at 30, 64 at 33, 128 at 34, 256 at 35 |
| compose transient | Qrack rebuilds the widest patch one qubit wider while holding the old state: + half the new size |
| exact `fxeb` reference | 12 · 2^n bytes (state + probability buffer): 1.5 GiB at 27, 48 at 32, 193 at 34 |
| point estimate for the ledger | 1 GiB + engines' worst case + reference |

With `--mem-budget-gb`, a worker takes a point only if all reservations plus this one fit the budget, and the point fits `MemAvailable` (plus allowed swap, §5.5) after the unallocated part of other reservations. Otherwise it takes a smaller point or waits. A point larger than the budget is marked failed. Runs sharing one ledger must use the same budget. The ledger tracks host memory only.

### 5.2 GPU

| Check | CPU | GPU (NVIDIA CMP 50HX, OpenCL) |
|---|---|---|
| tiled v2, n = 27, d = 12 / 16 / 20 (± 0.009) | 0.158 / 0.063 / 0.024 | 0.146 / 0.063 / 0.033 |

| Cost on the CMP 50HX (10 GB) | Value |
|---|---|
| `fxeb` point, two variants, n = 27–28 | ≈ 10 min per worker (3 workers) |
| same, n = 29 | ≈ 13 min per worker (2 workers) |
| exact reference alone, n = 27 | ≈ 10 s |
| device memory per worker | ≈ 8 · 2^n bytes for the reference while built + ≈ 0.5 GiB engines and context |
| workers per card | ≤ 4 at n = 27, ≤ 2 at 28–29, 1 at 30, none beyond (10 at n = 27 and 3 at n = 28 failed) |

ACE is not faster on GPU: its simulators are small, its work is many short kernels with read-backs and host-side seam logic, and workers on one card queue. The useful split is ACE on CPU workers and `--ref-gpu` for the reference. Device indices follow Qrack's start-up listing and can change with drivers. Check one point against the CPU on every new device.

VRAM does not help patches larger than a die. Six V340 dies hold ≈ 45 GiB, a sixth of a 35-qubit patch, and the overflow sits in host RAM behind PCIe 3.0 x8 links (≈ 20 GB/s in total), no faster than NVMe swap.

### 5.3 Qrack's OpenCL engine: defects and fix

A 16-qubit reproducer (`qrack_ocl_repro.py`: 20 layers of U + CZ, `prob()` on two qubits), PoCL, Qrack 10.13.1:

| Engine | prob(q3) / prob(q10) |
|---|---|
| CPU | 0.414847 / 0.430348 |
| OpenCL, stock | NaN / NaN, NaN / NaN, segfault (three runs) |
| OpenCL, `DISABLE_OOO_OCL` | 1.0 / 1.0 (wrong) |
| OpenCL, synchronous dispatch | 0.414847 / 0.430348 |

| | Defect in `QEngineOCL` (`src/qengine/opencl.cpp`) | Where |
|---|---|---|
| A | non-blocking writes read stack arrays that go out of scope before the write runs | `Dispose(perm)`, `DecomposeDispose` |
| B | `CL_MEM_USE_HOST_PTR` buffers have their host memory freed while kernels are queued | `Dispose`, `Compose` |
| C | the completion callback is registered on an empty event: `CL_INVALID_EVENT` in 1099 of 1099 registrations | `DispatchQueue` |

Fixing A and B together restores correct results with asynchronous dispatch; neither alone does. Upstream Qrack 10.13.2 (`cf53586`) adds opt-in `QRACK_OCL_SYNC=1` and keeps the asynchronous path as is. `qrack_ocl_async_fix.patch` copies each write's arguments to the heap, gives `USE_HOST_PTR` buffers a destructor-callback owner, and removes the dead callback.

| Validation of the patch (PoCL, 10.13.2) | Result |
|---|---|
| reproducer | exact in 5 of 5 runs |
| ACE mirror, d = 4 and 6, 3 instances × 640 runs | equal to the CPU within 3 × 10⁻⁴ |
| wall time against `QRACK_OCL_SYNC=1` | 185 s vs 198 s |

```bash
git clone https://github.com/unitaryfoundation/qrack && cd qrack && git checkout cf53586f
git apply /path/to/qrack_ocl_async_fix.patch
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF -DENABLE_OPENCL=ON -DQBCAPPOW=7 ..
make -j$(nproc) qrack_pinvoke && sudo make install     # /usr/local/lib/qrack/libqrack_pinvoke.so
```

### 5.4 VRAM oversubscription: `--gtt`

Qrack keeps a state in device memory when three times its size fits, else in host RAM mapped over PCIe (the GTT on amdgpu). Every allocation counts against `QRACK_MAX_ALLOC_MB`, which defaults to the device's memory, so a spilling patch is refused unless the ceiling is raised. On an 8 GB V340 die, states up to 28 qubits stay in VRAM.

`--gtt GB` (with `--ace-gpu`) raises each worker's ceiling to `--vram-gb` + GB and sets `QRACK_QPAGER_DEVICES_HOST_POINTER=1`. The preflight prints each worker's VRAM / host split and stops if the VRAM part exceeds the die (less 0.5 GiB) or the host part exceeds GB. `layoutsearch --gtt` and `nighthawk_cloverfield.py --gtt` use the same split.

| Two `nighthawk88e` chips | Links |
|---|---|
| ghost, VRAM only | 3 (5.8 GiB per chip) |
| ghost, `--gtt 8` | 5 (chip at 25 / 28 / 29 / 26: 2.8 GiB VRAM + 8 GiB GTT) |
| ghost, `--gtt 16` | 6 (a 30-wide patch: 2.8 GiB VRAM + 16 GiB GTT) |
| mean-field | 8 |
| **Four chips, 2 × 2:** ghost, VRAM only / `--gtt 8` | 10 / 14 |

Verified on PoCL: a 128 MiB state fails under a 61 MB ceiling and runs under 563 MB in the loom's worker environment. Host-mapped patches run at PCIe speed rather than HBM speed, and they use the `USE_HOST_PTR` path of defect B, so use the patched build. `cat /sys/class/drm/card*/device/mem_info_gtt_total` shows the GTT; `amdgpu.gttsize` and `ttm.pages_limit` raise it.

### 5.5 Larger than RAM: `--swap-gb`

| `--swap-gb GB` | Effect |
|---|---|
| ledger | counts up to GB of free swap as available |
| preflight | prints dense states, compose transient, RAM and swap; warns, never stops |
| Qrack ceilings | for CPU patches over 28 qubits, `QRACK_MAX_CPU_QB` and `QRACK_MAX_PAGING_QB` set to 40 (also without `--swap-gb`); an explicit value wins. Past the ceiling QUnit throws "needed to engage automatic circuit elision" |

| `nighthawk61r2`, error detection on, EPYC | GiB |
|---|---|
| dense states, 35 / 34 | 384 |
| compose transient | 128 |
| worst case per worker | 513 |
| RAM + swap (320 GB + 256 GB NVMe RAID0) | 298 + 238 = 536 |
| state in swap at full width | ≈ 90 |

```bash
sudo sysctl vm.swappiness=100 vm.page-cluster=6   # swap out eagerly; 64 pages per swap-in
swapon --show && ulimit -v                         # swap active; must be unlimited
```

### 5.6 Precision and shot floor

| Quantity | Value |
|---|---|
| reference storage | float32; sums over 2^n in float64; sampled probabilities divided by the float64 total |
| accumulated rounding, d = 20, n = 34 | 10⁻⁶–10⁻⁵ per probability, ≈ 10⁻⁵ in XEB |
| shot noise per instance, 4096 shots | ≈ 0.015 → XEB below ≈ 0.03 unresolved |
| shots for d = 16–20 at n ≥ 30 | ≥ 16 384 |
| ideal XEB at d = 20 | 1.02–1.05 (Porter–Thomas) |
| ideal XEB at n = 27–30 | ≈ 66 at d = 4, 3–4 at d = 8, 1.0–1.6 at d ≥ 12: rank configurations on d ≥ 12 |

### 5.7 What can be verified at 61 qubits

The release holds every sampled 61-qubit bitstring of the full and patched circuits, but only hit counts for mirrors. No full-circuit probability is computed in [[1]](#ref-1) or here. BlueQubit's single-amplitude contraction cost (`contraction_cost/results/depth_sweep.json`, cotengra/kahypar, upper bounds):

| Depth | 20 | 28 | 32 | 36 | 40 |
|---|---|---|---|---|---|
| log₁₀ C_amp (multiply-adds) | 13.0 | 17.6 | 19.8 (20.0) | 21.8 (22.0) | 22.0 (22.1) |
| largest intermediate | 2³⁴ | 2⁴⁹ | 2⁵⁶ | 2⁶⁴ | 2⁶⁵ |

| Cost at d = 36 (W = 8 κ N_s F C_amp, κ = 10, N_s = 10⁶, F = 2.3 × 10⁻³, Frontier at 20 %) | Value |
|---|---|
| sampling | ≈ 1.2 × 10²⁷ FLOPs, ≈ 115 years (≈ 160 with the repeat search) |
| one exact amplitude | ≈ 5 × 10²² FLOPs, ≈ 2 days |
| bitstrings needed to resolve F | ≈ 1/F² ≈ 2 × 10⁵ |
| largest intermediate | 2⁶⁴ entries ≈ 128 EiB: slicing required, raising the cost |

---

## 6. Results

| Convention | |
|---|---|
| device reference | the paper's mirror fit F(d) = 0.326 × 0.8717^d, b_dev = 0.137 per cycle at n = 61; per-depth device points from the release's counts |
| §6.1–6.4 | released Haar ensemble unless noted, `fxeb` against the exact reference, 3 instances, 4096 shots, inverse-variance means |
| §6.5–6.9 | mirror circuits, `prob` estimator, plain means unless noted |
| PyQrack | 2.28.1 and the version before it; ACE behaviour can change between versions |

### 6.1 Layout and placement at n = 27

| lrc/lrr | XEB d = 8, device placement | XEB d = 8, tiled v2 | XEB d = 12, tiled v2 | effective bulk / seam qubits |
|---|---|---|---|---|
| 4/4 | 0.008 | **0.321** | **0.182** | 8.0 |
| 4/3 | 0.036 | 0.227 | 0.153 | 8.0 |
| 2/7 | 0.046 | 0.154 | 0.136 | 5.8 |
| 1/7 | 0.077 | 0.145 | 0.109 | 3.5 |
| 2/2 | −0.032 | 0.117 | 0.091 | 0.7 |

| Spearman ρ with XEB (10 configurations, d = 8) | ρ |
|---|---|
| simulators holding bulk qubits | −0.84 |
| cross couplers | −0.80 |
| exact couplers | +0.74 |
| logical qubits on seams | −0.40 |
| replica couplers | −0.21 |
| widest simulator | −0.15 |

| Layout alone, 8 × 8 window, no tiling | Couplers on a seam (of 102) | Simulators |
|---|---|---|
| 4/4 (default) | 69 | ≤ 23 qubits |
| 5/4 (best with all simulators ≤ 33) | 68 | ≤ 33 |
| 2/7 torus | 57 | 33 / 33 / 25 / 25, 128 GiB |
| device placement, 4/4, n = 61 | exact / replica / cross 33 / 30 / 39 | — |
| tiled v2, 4/4, n = 61 | 61 / 41 / 0 | — |

Tiling removes every cross coupler at unchanged memory, and the widest simulator does not predict the outcome. Removing cross couplers is necessary; after that the ordering follows the qubits on seams.

### 6.2 Size, depth and tiling version (4/4)

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

Both tilings ran on the same circuits against one shared reference. σ ≈ 0.009, and d = 20 is at the 4096-shot floor. Halving the seam qubits (n = 28) raises the XEB 3.6–4.3-fold, and where the seam counts agree the tilings coincide. Over d = 8–12 at n = 27, b ≈ 0.19; over d = 16–20, 0.25: the decay is not a single exponential.

| ACE switch (paired difference from default, 5 circuits) | n = 27, d = 12 | n = 27, d = 16 | n = 28, d = 12 | n = 28, d = 16 |
|---|---|---|---|---|
| boundary repetition code on | −0.007 ± 0.010 | +0.009 ± 0.008 | −0.012 ± 0.011 | −0.001 ± 0.008 |
| seam error detection off | +0.008 ± 0.006 | +0.007 ± 0.009 | +0.029 ± 0.014 | +0.011 ± 0.005 |
| crossbars off | +0.001 ± 0.012 | +0.002 ± 0.009 | −0.020 ± 0.009 | −0.002 ± 0.005 |

Only switching error detection off shifts the XEB, by a constant ≈ +0.010 ± 0.003 at n = 27–28. The decay per cycle is unchanged, so no switch lowers the cost of a seam. At 61 qubits error detection matters in the other direction (§6.8).

### 6.3 The `nn_qab` ensemble (n = 27, d = 12, plain means)

| Configuration | Circuits | XEB | HOG |
|---|---|---|---|
| Nighthawk window, 4/4 tiled v3 | 20 | **0.435 ± 0.037** | 0.665 |
| Nighthawk window, 4/4 device placement | 20 | 0.153 ± 0.014 | 0.500 |
| `nn_qab` 26 qubits, B-to-B 5.5 (best series) | 100 | 0.241 | — |
| `nn_qab` 27 qubits, two patches, B-to-B 3.5 | 100 | 0.185 | — |
| `nn_qab` 27 qubits, three patches, B-to-B 2.0 | 100 | 0.111 | — |
| Haar ensemble, 4/4 tiled v2 (inverse-variance) | 3 | 0.182 | — |

- **Ensembles don't compare.** `nn_qab` outputs are far from Porter–Thomas at d = 12 (ideal XEB ≈ 16 against ≈ 1.5), so the two ensembles' XEB values cannot be compared directly.
- **Placement and gate set.** Placement alone gives ×2.85. Tiled ACE on the Nighthawk window beats every `nn_qab` geometry by ≥ ×1.8, which may partly reflect the CZ-only gate set.
- **HOG at chance.** With the device placement the HOG is at chance while the XEB stays positive: a few very heavy bitstrings carry the score.

### 6.4 Cost

| Point | Hardware | Time |
|---|---|---|
| `fxeb`, n = 34, d = 20 (reference-dominated) | 96-thread EPYC | ≈ 6.2 × 10³ s |
| `fxeb`, tiled 4/4, n = 27 | EPYC | 1–4 × 10² s |
| `fxeb`, tiled 2/7, n = 27 | EPYC | ≈ 1.8 × 10³ s |
| mirror, n = 61, untiled, d = 4–20, shots | EPYC | 0.65–4.5 × 10³ s |
| mirror, n = 61, three patches, `prob`, 640 runs, d = 16 / 20 | EPYC, ≈ 16 threads per point | ≈ 6 h / 11 h |
| mirror, 45-qubit two-patch, 10 runs, d = 8 / 12 / 16 / 20 | 2-core container | 64 s / 340 s / 671 s / 1361 s |

### 6.5 61 qubits, three patches

Placements of the release circuit on the `mirror_nighthawk.py` register (75 sites, lrc 4 lrr 5):

| Exclusions | Engine | exact / replica / cross | Seam qubits | Memory | Mirror d = 4 | Mirror d = 6 |
|---|---|---|---|---|---|---|
| none, 64 sites lrc 4 lrr 4 | anneal | 63 / 38 / 1 | 12 | 0.05 GiB | — | — |
| none | exact | 73 / 29 / 0 | 8 | 0.77 GiB | 0.121 ± 0.010 | 0.056 ± 0.007 |
| first 7 | anneal, `xok` | 76 / 24 / 2 | 8 | 0.66 GiB | — | — |
| 11 | anneal, `xok` | 76 / 25 / 1 | 8 | 0.44 GiB | 0.897 ± 0.010 | 0.118 ± 0.010 |
| 11, site 69 reopened | anneal | 72 / 30 / 0 | 9 | 0.50 GiB | 0.799 ± 0.013 | 0.086 ± 0.009 |
| **11** | **exact, proven optimal** | **74 / 28 / 0** | **9 (4 / 3 / 2 per boundary)** | **0.50 GiB** | **0.646 ± 0.015** | **0.388 ± 0.015** |
| 11 | vf2 / plain anneal | none within the bound / refused (cross coupler 34–41) | — | — | — | — |

(Mirror columns: `nn_qab` θ, one instance, 1000 shots, 4 twirls.) The 11 exclusions are 74, 59, 44, 73, 69, 54, 68, 67, 64, 63, 62. Any balanced three-way split of the Nighthawk graph needs ≥ 9 boundary qubits.

Sweep on the exact placement (`nn_qab` θ, `prob`, 10 inputs × 64 twirls per point, CPU):

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

| Loss per boundary CZ | Value |
|---|---|
| d = 8 → 12 | +3 boundary CZs, survival flat |
| d = 12 → 20 | +22 boundary CZs, factor ≈ 310: 0.26 nats per CZ |
| instance 1, d = 12–16 / 16–20 | 0.29 / 0.21 nats per CZ |
| single-exponential fit | 5.30 × 0.586^d (exceeds 1 at d = 0) |
| power-law fit, log(−ln F) vs log d | p = 1.59 (d = 4–20), 1.28 (d ≥ 8); exponent 1.81 against boundary CZs |

The survival follows the boundary-CZ count in steps, not depth. Instance 0 runs on a lower curve from d = 8 on and pulls the inverse-variance mean below the plain mean. These are `nn_qab`-ensemble circuits on the device's register, so the comparison with the Haar-angle device points is one of shape.

### 6.6 Seam cost extrapolated to 61 qubits

| n (tiled v3) | b over d = 12–20 | replica couplers | b per replica coupler |
|---|---|---|---|
| 27 | 0.23 | 9 | 0.026 |
| 28 | 0.27 | 13 | 0.021 |
| 29 | 0.32 | 12 | 0.027 |
| device at n = 27–29 (2.25 × 10⁻³ per qubit-cycle) | ≈ 0.06 | — | — |

| Split of the 8 × 8 window | Couplers between simulators | Estimated b at 61 | Measured |
|---|---|---|---|
| tiled 4/4, five simulators | ≈ 40 | ≈ 1.0 | — |
| three large patches | ≈ 16 | ≈ 0.4 | single-exponential b ≈ 0.53 (§6.5) |
| two halves | ≈ 8 | ≈ 0.2 | §6.8 |
| device | — | 0.137 | — |

| 49-qubit sub-register, same qubits | Patches | Boundary qubits | exact / replica / cross | Mirror d = 4 / 6 (`nn_qab`, 20 runs) |
|---|---|---|---|---|
| 55 sites, lrc 5 lrr 4, flat | 28 / 27 | 4 | 64 / 15 / 0 | 0.760 / 0.706 |
| 75 sites, lrc 4 lrr 5, torus | 23 / 23 / 12 | 6 | 58 / 21 / 0 | 0.206 / 0.234 |

No balanced cut of an eight-wide grid crosses fewer than ≈ 8 couplers. More shots lower only the floor, and post-selecting on ideal probabilities is the spoofing of [[5]](#ref-5), not a fidelity gain.

### 6.7 Larger registers on one 8 GB GPU

| Patches | Logical qubits | Boundary qubits | Widest patch incl. ancilla |
|---|---|---|---|
| 2 | 49 | 4 | 28 |
| 3 (experiment's register) | 61 | 9 | 25 |
| 3 | 68 | 11 | 28 |
| 4, widths ≤ 27 | 87 | 16 | 27 |
| 4 (`nighthawk88`) | 88 | 17 | 28 |
| 4, disabled sites on the seams (`nighthawk88d`, `nighthawk88e`) | 88 | 11 | 28 |

![nighthawk88d](nighthawk88d_four_patch_seams.png)

| CPU check, `nn_qab` θ, 3 instances × 40 runs, inverse-variance | d = 4 | d = 6 |
|---|---|---|
| `nighthawk88` | 0.047 ± 0.009 | 0.0082 ± 0.0020 |
| `nighthawk88d` | 0.030 ± 0.005 | 0.0073 ± 0.0018 |
| `nighthawk88e` | **0.287 ± 0.020** | **0.143 ± 0.012** |

`nighthawk88e` has 14 % fewer boundary couplers than `nighthawk88d` and scores 10–20× higher. Where the seams sit relative to the mirrors' rotating pseudo-patch cuts matters as well. Instance scatter at d = 4 spans 0.018–0.20.

### 6.8 Two patches at 61 qubits

| Boundary CZs per mirror (forward half), d | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 36 |
|---|---|---|---|---|---|---|---|---|
| three patches (§6.5) | 11 | 18 | 24 | 27 | 27 | 40 | 49 | 93 |
| `nighthawk61r2` | 8 | 12 | 14 | 16 | 19 | 25 | 34 | 67 |
| `nighthawk61w` | 1 | 3 | 6 | 6 | 7 | 12 | 13 | 27 |

| Register | θ | Error detection | Runs per point | d = 4 | d = 6 |
|---|---|---|---|---|---|
| three patches (§6.5) | `nn_qab` | on | 640 | 0.495 | 0.338 |
| `nighthawk61r2` | `nn_qab` | on | 240 | 0.522 ± 0.019 | 0.354 ± 0.016 |
| `nighthawk61r2` | `nn_qab` | off | 240 | 0.448 ± 0.017 | 0.207 ± 0.012 |
| `nighthawk61r2` | Haar | on | 240 | **0.226 ± 0.021** (plain 0.254) | **0.128 ± 0.014** (plain 0.147) |
| `nighthawk61w` | both | on | 240 | 1.00000 | 1.00000 |
| `ibm_phoenix` | Haar | — | — | 0.214 | 0.154 |

(Inverse-variance means unless noted.) Two patches on IBM's register, with error detection, match the device at d = 4–6 on its own circuits. A 6-qubit seam is the floor for any balanced split of the 8 × 8 window, so the boundary losses remain. With error detection the run needs swap (§5.5).

`nighthawk61w` absorbs its 1–3 boundary CZs at d = 4–6 exactly. Its deeper points need the 64 GiB patches. The same construction at 45 qubits (rows 2–11 × cols 0–4, 3-qubit seam, widths 26 / 24, Haar θ, 1 instance × 10 runs) gives:

| d | 8 | 12 | 16 | 20 |
|---|---|---|---|---|
| boundary CZs | 5 | 6 | 10 | 11 |
| 45-qubit waist register | 0.899 ± 0.035 | 0.883 ± 0.036 | 0.164 ± 0.10 | 0.124 ± 0.074 |
| `ibm_phoenix`, 61 qubits | 0.109 | 0.062 | 0.030 | 0.021 |

| Comparison | `nighthawk61r2` | `nighthawk61w` |
|---|---|---|
| qubits, CZs at d = 20 | 61, 221 | 61, 221 |
| narrowest cut | 8 (IBM's window) | 3 |
| what a win shows | ACE against the device on the device's register | ACE on a two-chip register with a narrow link |

The device's fidelity follows its gate count, so it would score about the same on either register. The waist only makes the classical simulation easier. The 45-qubit runs at d = 16–20 are bimodal (inputs survive nearly intact or near 0), so ten runs fix the mean only to a factor of ≈ 2.

### 6.9 Two-patch scaling ladder against Dan's calculator

The release's first n qubits, each split in two by `layoutsearch --fixed-register release:N --cz-weights mirror --objective seam,balance,cz`, with the widest patch capped at n/2 + 5:

| n | 28 | 32 | 36 | 40 | 45 | 49 | 53 | 57 | 61 |
|---|---|---|---|---|---|---|---|---|---|
| register | `2p28` | `2p32` | `2p36` | `2p40` | `2p45` | `2p49` | `2p53` | `2p57` | `61r2` |
| boundary qubits (proven minimum) | 3 | 3 | 3 | 3 | 4 | 4 | 5 | 5 | 6 |
| patch widths incl. ancilla | 15/18 | 18/19 | 20/21 | 22/23 | 26/25 | 27/28 | 30/30 | 32/32 | 35/34 |
| couplers through the boundary | 11 | 11 | 12 | 12 | 12 | 15 | 15 | 18 | 19 |
| boundary CZs at d = 20 | 18 | 20 | 25 | 21 | 20 | 29 | 34 | 34 | 34 |
| Dan's XEB(12) | 0.372 | 0.409 | 0.444 | 0.476 | 0.425 | 0.450 | 0.408 | 0.429 | 0.396 |
| Dan's XEB at d = 4 / 20 / 36 | 0.844 / 0.106 / 0.0031 | 0.858 / 0.132 / 0.0054 | 0.870 / 0.158 / 0.0088 | 0.881 / 0.186 / 0.0132 | 0.864 / 0.143 / 0.0068 | 0.872 / 0.163 / 0.0095 | 0.857 / 0.130 / 0.0053 | 0.865 / 0.146 / 0.0071 | 0.853 / 0.122 / 0.0045 |

Measured mirror survival, with Dan's prediction in brackets (CPU, error detection on; 3 instances × 10 runs to n = 36, 1 × 10 at 40–45):

| n | d = 4 | 8 | 12 | 16 | 20 | 36 | fitted p | fitted XEB(12) |
|---|---|---|---|---|---|---|---|---|
| **`nn_qab` θ** | | | | | | | | |
| 28 | 0.593 (0.844) | 0.543 (0.597) | 0.462 (0.372) | 0.354 (0.208) | 0.058 (0.106) | 1.9 × 10⁻⁴ (0.0031) | 1.68 | 0.360 |
| 32 | 0.730 (0.858) | 0.260 (0.628) | 0.212 (0.409) | 0.135 (0.242) | 0.082 (0.132) | 2.4 × 10⁻⁴ (0.0054) | 1.44 | 0.217 |
| 36 | 0.765 (0.870) | 0.259 (0.655) | 0.058 (0.444) | 0.019 (0.276) | 2.8 × 10⁻⁴ (0.158) | 8.9 × 10⁻⁵ (0.0088) | 1.68 | 0.100 |
| 40 | 0.989 (0.881) | 0.319 (0.679) | 0.092 (0.476) | 0.034 (0.308) | 0.0088 (0.186) | 1.6 × 10⁻⁵ (0.0132) | 2.87 | 0.306 |
| 45 | 0.748 (0.864) | 0.600 (0.640) | 0.103 (0.425) | 0.067 (0.257) | 0.0010 (0.143) | — | 2.89 | 0.166 |
| **Haar θ** | | | | | | | | |
| 28 | 0.278 | 0.087 | 0.056 | 0.056 | 9.3 × 10⁻⁴ | 2.7 × 10⁻⁵ | 1.02 | 0.039 |
| 32 | 0.327 | 0.024 | 0.013 | 0.026 | 0.0037 | 2.7 × 10⁻⁵ | 0.81 | 0.021 |
| 36 | 0.397 | 0.052 | 0.0041 | 4.0 × 10⁻⁵ | 4.7 × 10⁻⁴ | 2.7 × 10⁻⁶ | 1.01 | 0.005 |
| 40 | 0.849 | 0.152 | 0.0046 | 0.0012 | 1.5 × 10⁻⁶ | 4.2 × 10⁻⁸ | 2.14 | 0.043 |
| 45 | 0.319 | 0.0078 | 0.0028 | 0.0055 | — | — | 0.86 | 0.003 |
| **Dan's calculator** | | | | | | | 1.606 | 0.37–0.48 |

(Fitted p and XEB(12): Dan's form, log(−ln F) linear in log(d/12), from d = 6 on.)

| Forward XEB, `nighthawk2p28`, `nn_qab` θ, 1 instance | d = 4 | d = 12 |
|---|---|---|
| `fxeb` against the exact reference | 0.885 ± 0.044 | 0.338 ± 0.016 |
| Dan's calculator | 0.844 | 0.372 |
| mirror survival, same register | 0.593 | 0.462 |

- **At 28 qubits the calculator holds.** That is the edge of its data, and the XEB it was fitted on agrees within 10 %.
- **Above 28 it overestimates, more with size and depth.** At a fixed 3-qubit seam its x grows with the bulk, so the prediction improves. The measurement worsens instead: the fitted XEB(12) falls and p rises to ≈ 2.9. Its single predictor counts seam qubits, not the CZs through them.
- **Haar losses are 10–100× larger.** A fit on `nn_qab` circuits does not transfer to the device's ensemble.
- **The runs are short.** Instance scatter is 2–5× at depth, so single points are indicative; the trend holds across sizes in both ensembles.

### 6.10 Cloverfield

| Two `nighthawk88e` chips, 176 qubits, 3 ghost links (CPU, `nn_qab`, 3 instances × 20 runs) | d = 4 | d = 6 |
|---|---|---|
| ghost | 0.079 | 0.017 |
| mean-field | 0.056 | 0.015 |

Each chip carries 3 ghosts at widths 25 / 28 / 27 / 26 (≈ 5.8 GiB, one chip per die). At these depths the chips' own seam losses dominate; selftest [5] isolates the loom's share (§3.5). Each link count rebuilds the global circuit, since the K = 3 cuts follow the field's coupling graph.

Runtime screenshots:

<img width="970" height="403" alt="image" src="https://github.com/user-attachments/assets/717d8289-2a12-4de4-892d-77d5a161f955" />
<img width="1114" height="327" alt="image" src="https://github.com/user-attachments/assets/b84b32ce-4fae-434b-b06a-62b38d32c4b2" />

---

## 7. Run recipes

```bash
# 61 qubits, three patches (§6.5): EPYC, CPU, six workers
python3 nighthawk_qrack.py run --backend ace --families mirror --sizes 61 --theta nnqab \
  --depths 4 6 8 10 12 16 20 --lrc 4 --lrr 5 --ace-torus --ace-transpile --ace-width 75 \
  --ace-exclude 74,59,44,73,69,54,68,67,64,63,62 --mirror-estimator prob --order shallow \
  --cpu --gpus 0-5 --per-gpu 1 --mem-budget-gb 300 --max-rss-gb 16 --out mirror61_dan_prob_cpu.jsonl

# 88 qubits on six V340 dies (§6.7)
python3 nighthawk_qrack.py run --backend ace --register nighthawk88e --theta nnqab \
  --depths 4 6 8 10 12 16 20 --mirror-estimator prob --order shallow \
  --ace-gpu --gpus 0-5 --per-gpu 1 --stagger 20 --max-rss-gb 16 --out mirror88e_clvk.jsonl

# the waist register (§6.8): 128 GiB, fits RAM
python3 nighthawk_qrack.py run --backend ace --register nighthawk61w --families mirror --theta haar \
  --depths 4 6 8 10 12 16 20 --instances 3 --twirls 1 --mirror-estimator prob --order shallow \
  --cpu --gpus 0 --per-gpu 1 --mem-budget-gb 290 --out mirror61w_haar.jsonl

# IBM's register as two patches, error detection on (§6.8): RAM + swap, alone on the machine
python3 nighthawk_qrack.py run --backend ace --register nighthawk61r2 --families mirror --theta haar \
  --depths 4 6 8 10 12 16 20 --instances 3 --twirls 1 --mirror-estimator prob --order shallow \
  --cpu --gpus 0 --per-gpu 1 --swap-gb 238 --mem-budget-gb 530 --out mirror61r2_haar.jsonl

# the two-patch ladder (§6.9), then 61 with swap, then the comparison
for n in 28 32 36 40 45 49 53 57; do for th in nnqab haar; do
  python3 nighthawk_qrack.py run --backend ace --register nighthawk2p$n --families mirror --theta $th \
    --depths 4 6 8 10 12 16 20 36 --instances 3 --twirls 1 --mirror-estimator prob --order shallow \
    --cpu --gpus 0 --per-gpu 1 --mem-budget-gb 290 --out ladder_${n}_$th.jsonl
done; done
for th in nnqab haar; do
  python3 nighthawk_qrack.py run --backend ace --register nighthawk61r2 --families mirror --theta $th \
    --depths 4 6 8 10 12 16 20 36 --instances 3 --twirls 1 --mirror-estimator prob --order shallow \
    --cpu --gpus 0 --per-gpu 1 --swap-gb 238 --mem-budget-gb 530 --out ladder_61_$th.jsonl
done
python3 nighthawk_qrack.py danmodel --in ladder_*.jsonl --csv ladder_vs_dan.csv

# forward XEB on the ladder (§6.9): references of 3 and 48 GiB
for n in 28 32; do
  python3 nighthawk_qrack.py run --backend ace --register nighthawk2p$n --families fxeb --theta nnqab \
    --depths 4 6 8 10 12 16 20 --instances 3 --order shallow \
    --cpu --gpus 0 --per-gpu 1 --mem-budget-gb 290 --out fxeb_ladder_${n}_nnqab.jsonl &
done; wait

# a register search (§3.4)
python3 nighthawk_qrack.py layoutsearch --n 88 --geometries 114:4:5 102:4:5 --region convex \
  --hint nighthawk88e --candidates 3 --cpsat-time 600 --test-depths 4 8 --test-jobs 3 --cpu --save-best nighthawk88x

# the cloverfield (§3.5)
python3 nighthawk_cloverfield.py plan --chips nighthawk88e*2 --links 4 --png field.png
python3 nighthawk_cloverfield.py run --chips nighthawk88e*2 --links 4 --theta nnqab --depths 4 6 8 10 12 \
  --ace-gpu --gpus 0-5 --out field88e2.jsonl
python3 nighthawk_cloverfield.py linksearch --chips nighthawk88e*2 --link-counts 0 2 4 all --theta nnqab \
  --depths 4 8 --twirls 4 --cpu
```

Progress of any run: the same command with `--status`. Deep points at 57–61 qubits are the slowest; time one d = 8 point before committing them.

---

## 8. Limitations

- **Verification.** `fxeb` needs an exact 2^n reference (n ≲ 34–36). The full 61-qubit circuits cannot be scored, and statements at 61 qubits are proxies or extrapolations.
- **Families.** Patched circuits use the released partitions and run only at n = 61. Registers have no device data.
- **ACE versions.** ACE's layout, numbering and seam treatment depend on the PyQrack version, which enters neither the config tag nor the records: note it with each run, or use a new `--out` after an upgrade. Moving to 2.28.1 shifted the 61-qubit d = 6 instances by −0.007 and +0.005.
- **Statistics.** Three instances per point at most; instance scatter exceeds shot noise, and differences below ≈ 2σ of it are not resolved. The ladder, two-patch and waist results rest on 10–240 runs per point.
- **Weights.** Tiling and layoutsearch optimise proxy costs; their weights are not claimed optimal.
- **GPU builds.** GPU results depend on the Qrack build (§5.3): check one point against the CPU on every new device or driver.
- **Comparability.** Registers with disabled sites or waists change the circuit graph, so their fidelities are not comparable one-to-one with the release's.
- **Open failure.** A CPU-side C++ exception at d = 16 (61-qubit i0, 45-qubit `nn_qab`) is not yet traced.
- **Preprint.** arXiv:2609.28657 is a preprint (v1), not peer reviewed at the time of writing.

---

## 9. References

All identifiers were checked against the arXiv abstract pages and, where given, the published versions.

<a id="ref-1"></a>**[1]** T. Sedrakyan *et al.*, "Quantum computational advantage in random-circuit sampling on IBM superconducting quantum computers," [arXiv:2609.28657](https://arxiv.org/abs/2609.28657) (2026). — *The experiment, estimators, partitions, pseudo-patching, shot budgets.*

<a id="ref-2"></a>**[2]** S. Boixo *et al.*, "Characterizing Quantum Supremacy in Near-Term Devices," [arXiv:1608.00263](https://arxiv.org/abs/1608.00263); *Nature Physics* **14**, 595 (2018). — *Cross-entropy benchmarking as a fidelity proxy.*

<a id="ref-3"></a>**[3]** F. Arute *et al.*, "Quantum supremacy using a programmable superconducting processor," [arXiv:1910.11333](https://arxiv.org/abs/1910.11333); *Nature* **574**, 505 (2019). — *Linear XEB on 53 qubits; patch and elided circuits.*

<a id="ref-4"></a>**[4]** A. Morvan *et al.*, "Phase transitions in random circuit sampling," [arXiv:2304.11119](https://arxiv.org/abs/2304.11119); *Nature* **634**, 328 (2024). — *Regimes in which XEB tracks fidelity.*

<a id="ref-5"></a>**[5]** X. Gao, M. Kalinowski, C.-N. Chou, M. D. Lukin, B. Barak, S. Choi, "Limitations of Linear Cross-Entropy as a Measure for Quantum Advantage," [arXiv:2112.01657](https://arxiv.org/abs/2112.01657); *PRX Quantum* **5**, 010334 (2024). — *When XEB and fidelity diverge; spoofing by cutting weak links.*

<a id="ref-6"></a>**[6]** T. Proctor, K. Rudinger, K. Young, E. Nielsen, R. Blume-Kohout, "Measuring the Capabilities of Quantum Computers," [arXiv:2008.11294](https://arxiv.org/abs/2008.11294); *Nature Physics* **18**, 75 (2022). — *Circuit mirroring.*

<a id="ref-7"></a>**[7]** T. Proctor, S. Seritan, K. Rudinger, E. Nielsen, R. Blume-Kohout, K. Young, "Scalable randomized benchmarking of quantum computers using mirror circuits," [arXiv:2112.09853](https://arxiv.org/abs/2112.09853); *Phys. Rev. Lett.* **129**, 150502 (2022). — *Mirror-circuit RB.*

<a id="ref-8"></a>**[8]** J. Hines *et al.*, "Demonstrating Scalable Randomized Benchmarking of Universal Gate Sets," [arXiv:2207.07272](https://arxiv.org/abs/2207.07272); *Phys. Rev. X* **13**, 041030 (2023). — *Mirror RB with parametrised single-qubit gates.*

<a id="ref-9"></a>**[9]** J. J. Wallman, J. Emerson, "Noise tailoring for scalable quantum computation via randomized compiling," [arXiv:1512.01098](https://arxiv.org/abs/1512.01098); *Phys. Rev. A* **94**, 052325 (2016). — *Pauli-frame twirling.*

<a id="ref-10"></a>**[10]** D. Strano, B. Bollay, A. Blaauw, N. Shammah, W. J. Zeng, A. Mari, "Exact and approximate simulation of large quantum circuits on a single GPU," [arXiv:2304.14969](https://arxiv.org/abs/2304.14969); *Proc. IEEE QCE 2023*, pp. 949–958. — *Qrack's methods.*

<a id="ref-11"></a>**[11]** Y. Zhou, E. M. Stoudenmire, X. Waintal, "What limits the simulation of quantum computers?," [arXiv:2002.07730](https://arxiv.org/abs/2002.07730); *Phys. Rev. X* **10**, 041038 (2020). — *Controlled error per gate, matched against device fidelity.*

<a id="ref-12"></a>**[12]** I. L. Markov, A. Fatima, S. V. Isakov, S. Boixo, "Quantum Supremacy Is Both Closer and Farther than It Appears," [arXiv:1807.10749](https://arxiv.org/abs/1807.10749) (2018). — *Cost linear in target fidelity.*

**Software and data:** [BlueQubitDev/rcs-nighthawk](https://github.com/BlueQubitDev/rcs-nighthawk) (incl. `contraction_cost/`) · [unitaryfund/pyqrack](https://github.com/unitaryfund/pyqrack) · [vm6502q/pyqrack-examples](https://github.com/vm6502q/pyqrack-examples) (`nn_qab.py`, `qem/richardson_nn_hamming_weight.py`) · [twobombs/thereminq-examples](https://github.com/twobombs/thereminq-examples) (`rcs/nighthawk`, `rcs/nn_qab`)
