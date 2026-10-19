# HOWTO: `rcs/nighthawk/nighthawk_qrack.py`

**A clean-qubit PyQrack companion to arXiv:2609.28657 (random-circuit sampling on IBM Nighthawk r2), with an exact-reference study of seam-partitioned approximate simulation**

| | |
|---|---|
| Script | [`rcs/nighthawk/nighthawk_qrack.py`](https://github.com/twobombs/thereminq-examples/tree/main/rcs/nighthawk) (ThereminQ examples), 5026 lines, 16 October 2026 |
| Companions | `nighthawk_graph.py` (harvest, views, CSV) · `nighthawk_cloverfield.py` (several chips, one virtual chip) · [`rcs/nn_qab/`](https://github.com/twobombs/thereminq-examples/tree/main/rcs/nn_qab) |
| Methodology | [`METHODOLOGY.md`](METHODOLOGY.md): every procedure behind the code and the experiments |
| Results | [`RESULTS.md`](RESULTS.md): every measured number, by experiment |
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

| Finding | Numbers | [RESULTS](RESULTS.md) § |
|---|---|---|
| Placement matters more than layout | n = 27, d = 8 forward XEB: 0.008 device placement → 0.321 tiled, same memory | 2 |
| A seam costs fidelity every cycle, and no ACE setting removes it | decay b = 0.24–0.31 per cycle at n = 27–29 against the device's 0.137 at n = 61 | 3 |
| At 61 qubits the loss follows CZs through a boundary, not depth | three patches: 2.3× the device at d = 4–6, 1/160 at d = 20; ≈ 0.26 nats per boundary CZ | 6 |
| Two patches on IBM's own register and circuits match the device at shallow depth | Haar θ, d = 4 / 6: 0.226 / 0.128 against 0.214 / 0.154 | 9 |
| A register with a 3-qubit waist keeps the boundary nearly closed | 45-qubit version, d = 8 / 12 / 20: 0.90 / 0.88 / 0.12 against the device's 0.109 / 0.062 / 0.021; classically easier register | 9 |
| In Dan's own protocol (`--protocol nnqab`) the Nighthawk ladder meets his calculator at 16 qubits and falls below it with size | XEB(12) 1.03× at 16 qubits, 0.40× at 28; depth power 0.89–1.70 vs his 1.606 | 10 |
| Dan's mirror Hamming weight carries his protocol past the XEB ceiling | at U's depth d: within 0.86–1.33× of his calculator at d ≤ 6 at every size; at d = 12 0.37–0.63× at 28–40 qubits, 0.91× at 45 (4-qubit seam); tracks the forward XEB where both exist; folding exponent 0.04–0.48 against his 1.606 | 10 |
| Mirror survival is a different observable from his XEB | ACE's XEB equals a seam-cut sampler's XEB at 16–28 qubits; mirror survival falls with size | 14 |
| 88 qubits fit one 8 GB GPU; disabled sites on the seams shrink the boundary | boundary 17 → 11 qubits; `nighthawk88e` 10–20× `nighthawk88d` at d = 4–6 | 8 |
| A ghost link carries entanglement between chips | loom-only error, d = 8 / 12 / 16: ghost 1.000 / 0.989 / 0.656, mean-field 0.773 / 0.773 / 0.067 | 11 |
| Qrack's OpenCL engine had three host-memory defects | fixed without serialising dispatch; GPU reproduces CPU | 12 |

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
| `hamming` | Dan Strano's mirror (`qem/richardson_nn_hamming_weight.py`): the `fxeb` forward circuit U of d cycles, then U†, folded to (U U†)^λ for λ = `--fold`, from \|0…0⟩, no twirling; any n, 61 included | signal 1 − 2h, h = Σ_q P(q = 1) / n taken exactly from the backend (ideal 1) |

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

`--register NAME` replaces the release's register with a sub-register of the 12 × 10 Nighthawk lattice. The circuits are built by the release's own code and seed: `rect_matchings`, `remove_qubits`, `make_patches(seed=2025, balance_tol=1, top_n=5)` for the K = 3, 4 partitions, and the release's input-string recipe (selftest [9]). Registers have no device data; they run `mirror` and `hamming` at any size, and `fxeb` up to 36 qubits. Each carries a stored ACE preset (geometry, caps, placement).

| Register | Qubits | Region | ACE geometry | Patch widths incl. ancilla | Boundary qubits | exact / replica / cross | Couplers through a boundary | Dense memory |
|---|---|---|---|---|---|---|---|---|
| `release` (three-patch preset, RESULTS §6) | 61 | the experiment's 8 × 8 window | 75 sites, lrc 4 lrr 5, torus, 11 exclusions | ≤ 25 | 9 | 74 / 28 / 0 | 28 | 0.50 GiB |
| `nighthawk88` | 88 | rows 0–9 × cols 1–9, broken qubits used | 102 sites, lrc 4 lrr 5 | 28 / 27 / 27 / 27 | 17 | 102 / 55 / 0 | 55 per A–D cycle (14 / 14 / 14 / 13) | ≈ 7 GiB |
| `nighthawk88d` | 88 | whole lattice, 32 sites disabled (17, 55, 62 + 29 corners); every row and column one run; ring seams | 114 sites, lrc 4 lrr 5 | 28 / 23 / 25 / 27 | 11 | 108 / 42 / 0 | 42 (11 / 11 / 10 / 10) | ≈ 5.3 GiB |
| `nighthawk88e` | 88 | as 88d, seams moved by `layoutsearch` | 114 sites, lrc 4 lrr 5 | 25 / 28 / 25 / 25 | 11 | 112 / 36 / 0 | 36 (8 / 9 / 10 / 9) | ≈ 4.8 GiB |
| `nighthawk61r2` | 61 | the release's own qubits: identical mirror circuits, angles, inputs | 77 sites, lrc 5 lrr 6 | 35 / 34 (34 / 33 without error detection) | 6 (proven minimum, balanced) | 83 / 19 / 0 | 19 | 384 GiB (192) |
| `nighthawk61w` | 61 | rows 0–10 × cols 2–7 minus 17, 52, 55, 57, 62: a 3-qubit waist on row 5 | 75 sites, lrc 8 lrr 4 | 33 / 33 | 3 | 93 / 7 / 0 | 7 | 128 GiB |
| `nighthawk2p28` … `nighthawk2p57` | 28–57 | the release's first n qubits | RESULTS §10 | RESULTS §10 | 3–5 | RESULTS §10 | 11–18 | 2 MiB – 64 GiB |

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
| 11 | the twelve `nn_qab` gates undone by their inverses; exact folded mirror, λ = 1, 3 | overlap within 1e-5; signal within 1e-4 of 1 |

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
| `--protocol` | `nighthawk` | `nnqab`: forward XEB scored as `nn_qab.py` scores it (sin θ-uniform single-qubit gates, each coupler's gate drawn from its twelve two-qubit gates, 2^min(13, n+2) shots); points compare one-to-one with Dan's calculator; `fxeb` (n ≤ 36) and `hamming` (any n) |
| `--fold` | 1 | `hamming`: odd folding factors λ, e.g. `1 3 5` for the zero-noise extrapolation |
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

**Resumability.** A configuration is tagged `<backend>-<sha1[:8]>` over backend, shots, twirls, ACE parameters, tiling version and exact-probs mode, not the register size, so sweeps extend and resume per point. `--gtt` and `--swap-gb` do not change the tag. Bitstrings are written before the JSONL record, and the claim is marked done last. A record with a non-finite score (NaN) is ignored everywhere and its point reopened on the next run; `hamming` refuses to write one and fails the point instead. Records carry `device` (`cpu` / `gpu`), outside the tag. To drop a pending point, write `failed: …` into its claim file under `<out>_shots/<cfg>/claims/`.

### 3.3 `danmodel`

| Mode | Command | Output |
|---|---|---|
| prediction | `danmodel [--registers …] [--depths …]` | bulk, boundary, patches, x and predicted XEB per register and depth; default the two-patch ladder, d = 4 … 36 |
| comparison | `danmodel --in run.jsonl …` | every measured mirror, `fxeb` or `hamming` point next to the prediction at its own seam and patch counts (`hamming` at the depth d of U, as his script labels it, λ = 1 only; the CSV adds `dan_2d` at the mirror's 2·d layers); plain mean over instances, worker files included |
| zero-noise | the same, on `hamming` runs with ≥ 2 folds | Dan's Richardson extrapolation: ln(signal) extrapolated to λ = 0 in u = λ^1.60569 |

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

Validation of the tool: [RESULTS.md §13](RESULTS.md#13-tool-validation).

### 3.5 `nighthawk_cloverfield.py`

The cloverfield puts two or more chips side by side, joined by inter-chip couplers, as IBM plans with l-couplers: one register, one circuit, one score (mirror survival, or the Hamming signal).

| Part | Role |
|---|---|
| chip worker | one isolated process per chip and device; its register's ACE preset (or an exact simulator); executes gate batches and answers marginal and probability queries over a pipe |
| loom | the coordinator; builds no simulator (OpenCL hidden); generates the field's circuit by the release's recipe and does all IO between chips; ≈ 45 MiB |
| `--loom ghost` (default) | each chip holds a ghost of every partner its links touch; single-qubit gates and twirls go to every copy; the link CZ is exact on both chips; the partner's other CZs reach the ghost as a mean-field phase π·P(1). Records carry `ghost_drift` |
| `--loom mean`, `--loom sample` | mean-field phases only (chips in a product state); Z kicks drawn from P(1) |
| estimator | `prob` per chip; the field's survival is the product |
| `--family hamming` | Dan's observable on the field: (U U†)^λ from \|0…0⟩ (`--fold`), signal 1 − 2h with h the mean P(1) over all field qubits, from the chips' marginals in one round trip; linear, so exact given the chips' states. Records add `chip_signal` and `signal_bulk` / `signal_seam` / `signal_anchor` |
| `--protocol nnqab` | Dan's twelve gates (and θ), with cz / acz on links and on couplers touching a link anchor: the only gates the loom can weave |
| `--cut-links` | the same links, anchors and gate draws, links left out of the circuit: the matched no-link baseline |
| split run (`--cpu-looms N` with `--gpus`) | one queue, cheapest first by cost 2 d λ; GPU looms take from the expensive end, CPU looms from the cheap end, until they meet. `--gpu-min-layers L` keeps GPUs off points below L layers, `--cpu-max-layers L` keeps the CPU off points above L. A point that breaks on a GPU (twice an exception, or non-probability marginals) goes to the CPU looms once, whatever its cost. Records carry `placement` and `devices`; the configuration tag does not, so a split run and a CPU-only run share records |

Options: `--chips` (`nighthawk88e*2`, `nighthawk2p45*4`, `release,nighthawk88d`, `rect:r0-r1,c0-c1`, any register), `--arrangement RxC`, `--links K|all`, `--link-rows`, `--link-cols`, `--cut-links`, `--loom`, `--ghost-max-width`, `--ghost-max-wide`, `--gtt`, `--vram-gb`, `--backend`, `--chip-ace`, `--cpu`, `--ace-gpu`, `--gpus`, `--chips-per-gpu`, `--theta`, `--twirls`; `run` and `linksearch` add `--family`, `--fold`, `--protocol`; `run` adds `--cpu-looms`, `--gpu-min-layers`, `--cpu-max-layers`. Subcommands: `plan` (links, widths, seam qubits and link anchors per chip, memory: dies, or host RAM with `--cpu`; `--png`), `run`, `linksearch` (score against link count; with `--family hamming` also the anchor-region signal), `layoutsearch` (passed to nighthawk_qrack), `selftest` (`--quick` skips [7]).

| selftest | Check |
|---|---|
| 1 | field `nighthawk88e*2`, 4 links: colours are matchings, cuts deterministic |
| 2a | one chip, no links: loom op stream vs nighthawk_qrack's own calls |
| 2b | the same through a worker process |
| 3 | woven CZ on \|+⟩ against \|0⟩, \|1⟩, \|+⟩ |
| 4 | CZ·CZ = I across a link |
| 5 | loom error alone: two exact 9-qubit chips, 2 links, d = 8 / 12 / 16 |
| 6 | Hamming, one chip, no links: loom op stream vs `run_hamming`, CZ and nnqab protocols, d = 4, λ = 3 |
| 7 | the same on ACE (`nighthawk2p45`) through a worker process vs in-process |
| 8 | exact chips: signal 1 with 0 links and with `--cut-links`; the loom's error alone with woven links (mean-field and ghost) |
| 9 | `--protocol nnqab` on `nighthawk2p45*4`, 2 × 2: every anchor coupler cz / acz, Dan's draw elsewhere |
| 10 | woven acz against exact basis cases; acz · acz = I across a ghost link |
| 11 | split queue: GPU looms from the expensive end, CPU from the cheap end, thresholds, hand-back above `--cpu-max-layers`, termination |

Results: [RESULTS.md §11](RESULTS.md#11-cloverfield-e13).

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

An example of the gap is in [RESULTS.md §4](RESULTS.md#4-the-nn_qab-ensemble-e6-n--27-d--12).

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

| Device memory, one 10 GB card | Value |
|---|---|
| per worker | ≈ 8 · 2^n bytes for the reference while built + ≈ 0.5 GiB engines and context |
| workers per card | ≤ 4 at n = 27, ≤ 2 at 28–29, 1 at 30, none beyond |

ACE is not faster on GPU: its simulators are small, its work is many short kernels with read-backs and host-side seam logic, and workers on one card queue. Measured on the V340 dies with the 45-qubit `hamming` ladder (patches 26 / 25, 18 workers): 4–9 W of a 100 W card, so the ladder runs on CPU at every size. The useful split is ACE on CPU workers and `--ref-gpu` for the reference. Device indices follow Qrack's start-up listing and can change with drivers. Check one point against the CPU on every new device (the CMP 50HX check: RESULTS §13).

VRAM does not help patches larger than a die. Six V340 dies hold ≈ 45 GiB, a sixth of a 35-qubit patch, and the overflow sits in host RAM behind PCIe 3.0 x8 links (≈ 20 GB/s in total), no faster than NVMe swap.

### 5.3 Qrack's OpenCL engine: defects and fix

A 16-qubit reproducer (`qrack_ocl_repro.py`: 20 layers of U + CZ, `prob()` on two qubits) returns NaN, wrong values or a segfault on stock OpenCL builds and correct values with synchronous dispatch (RESULTS §12).

| | Defect in `QEngineOCL` (`src/qengine/opencl.cpp`) | Where |
|---|---|---|
| A | non-blocking writes read stack arrays that go out of scope before the write runs | `Dispose(perm)`, `DecomposeDispose` |
| B | `CL_MEM_USE_HOST_PTR` buffers have their host memory freed while kernels are queued | `Dispose`, `Compose` |
| C | the completion callback is registered on an empty event: `CL_INVALID_EVENT` in 1099 of 1099 registrations | `DispatchQueue` |

Fixing A and B together restores correct results with asynchronous dispatch; neither alone does. Upstream Qrack 10.13.2 (`cf53586`) adds opt-in `QRACK_OCL_SYNC=1` and keeps the asynchronous path as is. `qrack_ocl_async_fix.patch` copies each write's arguments to the heap, gives `USE_HOST_PTR` buffers a destructor-callback owner, and removes the dead callback.

Validation of the patch: RESULTS §12.

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

Links that fit under each setting: RESULTS §11.

Host-mapped patches run at PCIe speed rather than HBM speed, and they use the `USE_HOST_PTR` path of defect B, so use the patched build. The ceiling test is in RESULTS §12. `cat /sys/class/drm/card*/device/mem_info_gtt_total` shows the GTT; `amdgpu.gttsize` and `ttm.pages_limit` raise it.

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

Every measured number is in [RESULTS.md](RESULTS.md), grouped by experiment (E1–E18 of [METHODOLOGY.md](METHODOLOGY.md) §9), with the points not yet measured in its §15.

---

## 7. Run recipes

```bash
# 61 qubits, three patches (RESULTS §6): EPYC, CPU, six workers
python3 nighthawk_qrack.py run --backend ace --families mirror --sizes 61 --theta nnqab \
  --depths 4 6 8 10 12 16 20 --lrc 4 --lrr 5 --ace-torus --ace-transpile --ace-width 75 \
  --ace-exclude 74,59,44,73,69,54,68,67,64,63,62 --mirror-estimator prob --order shallow \
  --cpu --gpus 0-5 --per-gpu 1 --mem-budget-gb 300 --max-rss-gb 16 --out mirror61_dan_prob_cpu.jsonl

# 88 qubits on six V340 dies (RESULTS §8)
python3 nighthawk_qrack.py run --backend ace --register nighthawk88e --theta nnqab \
  --depths 4 6 8 10 12 16 20 --mirror-estimator prob --order shallow \
  --ace-gpu --gpus 0-5 --per-gpu 1 --stagger 20 --max-rss-gb 16 --out mirror88e_clvk.jsonl

# the waist register (RESULTS §9): 128 GiB, fits RAM
python3 nighthawk_qrack.py run --backend ace --register nighthawk61w --families mirror --theta haar \
  --depths 4 6 8 10 12 16 20 --instances 3 --twirls 1 --mirror-estimator prob --order shallow \
  --cpu --gpus 0 --per-gpu 1 --mem-budget-gb 290 --out mirror61w_haar.jsonl

# IBM's register as two patches, error detection on (RESULTS §9): RAM + swap, alone on the machine
python3 nighthawk_qrack.py run --backend ace --register nighthawk61r2 --families mirror --theta haar \
  --depths 4 6 8 10 12 16 20 --instances 3 --twirls 1 --mirror-estimator prob --order shallow \
  --cpu --gpus 0 --per-gpu 1 --swap-gb 238 --mem-budget-gb 530 --out mirror61r2_haar.jsonl

# the two-patch ladder (RESULTS §10), then 61 with swap, then the comparison
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

# the ladder in Dan's protocol (RESULTS §10): his depths 12-22, 20 circuits per point; references up to 48 GiB (n = 32)
for n in 16 20 24 28 32; do
  python3 nighthawk_qrack.py run --backend ace --register nighthawk2p$n --families fxeb --protocol nnqab \
    --depths 12 14 16 18 20 22 --instances 20 --order shallow \
    --cpu --gpus 0 --per-gpu 1 --mem-budget-gb 290 --out dan_protocol_${n}.jsonl
done
python3 nighthawk_qrack.py danmodel --in dan_protocol_*.jsonl --csv dan_protocol_vs_calculator.csv

# Dan's circuits from 16 to 61 qubits (RESULTS §10): his mirror Hamming weight. Folds 3, 5 only at d <= 8:
# ACE's signal does not change with folding (RESULTS §10), and they cost 3x / 5x. 61 = IBM's register, RAM + swap
for n in 16 20 24 28 32 36 40 45 49 53 57; do
  [ $n -le 40 ] && k=10 || k=6
  python3 nighthawk_qrack.py run --backend ace --register nighthawk2p$n --families hamming --protocol nnqab \
    --depths 4 6 8 10 12 --fold 1 --instances $k --order shallow \
    --cpu --gpus 0 --per-gpu 1 --mem-budget-gb 290 --out dan_hamming_${n}.jsonl
  python3 nighthawk_qrack.py run --backend ace --register nighthawk2p$n --families hamming --protocol nnqab \
    --depths 4 6 8 --fold 3 5 --instances $k --order shallow \
    --cpu --gpus 0 --per-gpu 1 --mem-budget-gb 290 --out dan_hamming_${n}.jsonl
done
python3 nighthawk_qrack.py run --backend ace --register nighthawk61r2 --families hamming --protocol nnqab \
  --depths 4 6 8 10 12 --fold 1 --instances 3 --order shallow \
  --cpu --gpus 0 --per-gpu 1 --swap-gb 238 --mem-budget-gb 530 --out dan_hamming_61.jsonl
python3 nighthawk_qrack.py run --backend ace --register nighthawk61r2 --families hamming --protocol nnqab \
  --depths 4 6 --fold 3 5 --instances 3 --order shallow \
  --cpu --gpus 0 --per-gpu 1 --swap-gb 238 --mem-budget-gb 530 --out dan_hamming_61.jsonl
python3 nighthawk_qrack.py danmodel --in dan_hamming_*.jsonl --csv dan_hamming_vs_calculator.csv
python3 dan_crosscheck/analyse_hamming.py dan_hamming_*.jsonl

# a register search (§3.4)
python3 nighthawk_qrack.py layoutsearch --n 88 --geometries 114:4:5 102:4:5 --region convex \
  --hint nighthawk88e --candidates 3 --cpsat-time 600 --test-depths 4 8 --test-jobs 3 --cpu --save-best nighthawk88x

# the cloverfield (§3.5)
python3 nighthawk_cloverfield.py plan --chips nighthawk88e*2 --links 4 --png field.png
python3 nighthawk_cloverfield.py run --chips nighthawk88e*2 --links 4 --theta nnqab --depths 4 6 8 10 12 \
  --ace-gpu --gpus 0-5 --out field88e2.jsonl
python3 nighthawk_cloverfield.py linksearch --chips nighthawk88e*2 --link-counts 0 2 4 all --theta nnqab \
  --depths 4 8 --twirls 4 --cpu

# the 45-qubit chip as a virtual multi-chip Nighthawk, Hamming ladder (§3.5): 2 x 2 = 180 qubits, CPU
python3 nighthawk_cloverfield.py plan --chips nighthawk2p45*4 --arrangement 2x2 --links all \
  --ghost-max-width 28 --ghost-max-wide 2 --cpu --png field2x2.png
python3 nighthawk_cloverfield.py linksearch --chips nighthawk2p45*4 --arrangement 2x2 --link-counts 0 4 8 all \
  --ghost-max-width 28 --ghost-max-wide 2 --family hamming --theta nnqab --depths 4 8 --instances 3 --cpu
# calibration: one chip, no links, both ends of the ladder, GPU against CPU (seconds, card power)
python3 nighthawk_cloverfield.py run --chips nighthawk2p45 --links 0 --family hamming --theta nnqab \
  --depths 6 12 --fold 1 5 --instances 1 --ace-gpu --gpus 0 --out cal_gpu.jsonl
python3 nighthawk_cloverfield.py run --chips nighthawk2p45 --links 0 --family hamming --theta nnqab \
  --depths 6 12 --fold 1 5 --instances 1 --cpu --out cal_cpu.jsonl
# the ladder as one split run: 4 dies (one chip each) from the deep end, 4 CPU looms from the shallow end;
# L = the cost 2 d λ above which the GPU won the calibration (48: d >= 8 at λ = 3, d >= 12 at λ = 2 ...)
python3 nighthawk_cloverfield.py run --chips nighthawk2p45*4 --arrangement 2x2 --links all \
  --ghost-max-width 28 --ghost-max-wide 2 --family hamming --theta nnqab --depths 4 6 8 10 12 --fold 1 3 5 \
  --ace-gpu --gpus 0-3 --chips-per-gpu 1 --cpu-looms 4 --gpu-min-layers 48 --out field2p45x4.jsonl
# the matched baseline (same links and anchors, links out of the circuit), then the mean-field control
python3 nighthawk_cloverfield.py run ... --cut-links --out field2p45x4.jsonl
python3 nighthawk_cloverfield.py run ... --loom mean --out field2p45x4.jsonl
# Dan's gates instead of CZ: add --protocol nnqab to all three
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
