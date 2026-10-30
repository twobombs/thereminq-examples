# HOWTO: `rcs/nighthawk/nighthawk_qrack.py`

**A clean-qubit PyQrack companion to arXiv:2609.28657 (random-circuit sampling on IBM Nighthawk r2), with an exact-reference study of seam-partitioned approximate simulation**

| | |
|---|---|
| Script | [`rcs/nighthawk/nighthawk_qrack.py`](https://github.com/twobombs/thereminq-examples/tree/main/rcs/nighthawk) (ThereminQ examples), 5026 lines, 16 October 2026 |
| Companions | `nighthawk_graph.py` (harvest, views, CSV) · `nighthawk_3dgraph.py` (interactive 3D, dark mode) · `nighthawk_cloverfield.py` (several chips, one virtual chip) · [`rcs/nn_qab/`](https://github.com/twobombs/thereminq-examples/tree/main/rcs/nn_qab) |
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
| `hamming` | the Hamming ladder against his calculator: per register, a measured ÷ Dan grid by coupler load, Richardson over the folds |
| `shor9` | one unit of Shor's code (9 qubits, or `--size m`: Shor-16, Shor-36, …) on a two-patch ACE register, or units tiled over a whole register: logical success, raw and decoded, for the four logical states, against the exact distribution |
| `seamgap` | data-only analysis of the release |
| `nighthawk_cloverfield.py` | chips in isolated workers, joined by a coordinator (the *loom*) |
| `nighthawk_ghz.py` | the formal proof's Layer 1a (GHZ populations + MQC) and Layer 2 (Walsh echo, compensation) on a register under ACE: F_GHZ per state, seam maps, refocusing across the seams; `stack`: seam layers S (syndrome + repetition code) and V (the corrected measurement), in either order; `sweep`: F_GHZ against the number of Walsh-echo windows; `probe`: the drift's inside ZZ put through ACE several ways; `stackmap`: a stack run as three heatmaps on the chip layout; `stack --engine bridge --box L H`: the stack on a closed box of six QrackBridge-joined faces; `shor` / `shormap`: Shor's [[m², 1, m]] code across QrackBridge circuits, every placement, logical Z and X against the bridge's capacity; `dual`: IBM's two-module Nighthawk R1 experiment (arXiv:2610.29986) on the bridge, its Bell and GHZ states under four models against its Table 1; `shoralg`: Layer 3, the formal proof's Shor order finding, as gates across the bridge (two chips, the box) and the dual R1, factoring success and distance to the ideal distribution against the bridge's capacity or the dual's models |
| `bridge_ring_repro.py` | QrackBridge on a ring (`wrap=True`) against an open chain: two GHZ chains through the wrap cut, the truncation that carries fewer units than the open chain or zeroes the state (RESULTS §10.16) |
| `nighthawk_graph.py` | harvests every run, `shor9`, `ghz`, `shor_bridge`, `dual_r1`, `dual_map` and `shor_alg` record, views 1–28, CSV export, `shor9` registers in 3D (`--grid3d`); views 18–21: GHZ tiles on the lattice (chips, loom, bridge, box net), by placement × Layer 2 mode, engine vs exact, chip-wide break maps; 22–25: placements against QrackBridge `max_bonds`, Shor across the bridge, dual Nighthawk R1 models and heatmap; 26–28: Layer 3's factoring success and TVD against `max_bonds`, and on the dual R1; GHZ registers in 3D too |
| `nighthawk_3dgraph.py` | interactive 3D in Python (matplotlib, no browser), dark mode by default: every shape the tools write (run landscapes and bits on the lattice, `shor9` registers, GHZ registers on ACE patches, loom and bridge chips, the folded box, Shor across QrackBridge, the dual Nighthawk R1 with its cables, Layer 3's landscapes), one data set and measure at a time, telemetry panel, site picking (§4.1) |

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
| Dan's mirror Hamming weight carries his protocol past the XEB ceiling | at U's depth d: within 0.86–1.33× of his calculator at d ≤ 6 at every size; at d = 12 0.37–0.63× at 28–40 qubits, 0.91× at 45 (4-qubit seam); tracks the forward XEB where both exist; folding exponent 0.04–0.41 against his 1.606 | 10 |
| Mirror survival is a different observable from his XEB | ACE's XEB equals a seam-cut sampler's XEB at 16–28 qubits; mirror survival falls with size | 14 |
| 88 qubits fit one 8 GB GPU; disabled sites on the seams shrink the boundary | boundary 17 → 11 qubits; `nighthawk88e` 10–20× `nighthawk88d` at d = 4–6 | 8 |
| ACE's seam loss is set at the first crossing, so ZNE cannot undo it | coherence kept after the first crossing 0.391 (28 qubits), 0.390 (61); ×17 crossings move F by ≤ 0.08; ZNE − pass 2 −0.02 … +0.01 against a rerun's 0.04–0.06 | 10.6 |
| A ghost link carries entanglement between chips | loom-only error, d = 8 / 12 / 16: ghost 1.000 / 0.989 / 0.656, mean-field 0.773 / 0.773 / 0.067 | 11 |
| The Walsh echo's seam loss on ACE is a one-time step, and comes from the drift's ZZ inside a unit, which the GHZ stabiliser removes | replica units 0.78 → 0.72 by L = 2, then flat to L = 8; dropping the inside ZZ (a global phase on the GHZ) keeps the echo pass at the no-drift level: replica 0.764, cut 0.355 against 0.683, 0.228 | 10.10, 10.11 |
| The loom between independent chips is a hard cut: a GHZ unit across it is separable there | every unit across the loom at 0.50 (the separable bound, sd 0.008) in every pass, code, echo and protection alike; ACE's seam is a slope by contrast (replica 0.56–0.93 before its code, 1.00 after). The ghost loom's 1.00 is a two-column replica band simulating a unit whole on one chip; 5-chains wider than the band stay at 0.26 | 10.14 |
| QrackBridge carries GHZ states across the chip cut up to its bond capacity | `max_bonds` 6: every unit across it 1.00; 2 / 1: per circuit max_bonds units carried at 1.00, the rest separable at 0.50 (means 0.85 / 0.74); 0: the hard cut (0.51) | 10.15 |
| A closed box of six QrackBridge-joined faces carries GHZ chains across all twelve edges; its ring's wrap cut is the weak point | 5 × 5 × 5, 98 qubits: `max_bonds` 6 every unit across an edge 1.00, relayed through a third face or not; 2 / 1: the inner cuts ration whole units to the budget (0.75 / 0.63), the wrap cut 0.67 / 0.61; at 1 and 0 with the drift's ZZ as bridged gates the ring zeroed the state; 0: the hard cut (0.51) | 10.16 |
| Shor's code across QrackBridge needs a countable number of bonds, and its decoder repairs a cut short of them | between blocks 1 bond (for \|±_L⟩), through blocks one per block split; short of that the bridge drops whole blocks and decoding follows the majority count: Shor-9 / -16 / -25 at 1 bond 0.745 / 0.688 / 0.677 (0.750 / 0.688 / 0.688), 1.000 up to ⌊(m − 1)/2⌋ blocks short; at 0 E20's ACE-seam pattern (0.50 on the side that needs coherence) | 10.17 |
| IBM's two-module Nighthawk R1 (arXiv:2610.29986): its benchmarked links explain its Bell and GHZ states to 1.5–3 points, except where the paper names a cause | the hard cut caps every state at 0.50; the bridge with the measured distant CZ predicts Bell 115–233 0.954 (IBM mitigated 0.939), GHZ-4 0.944 (0.917), GHZ-6 0.938 (0.909); Bell 117–231 0.956 against 0.870 (the paper's worst-readout qubit Q231) and the simultaneous drop (−0.064) are outside the model | 10.18 |
| Shor order finding across a cut needs ⌈log₂ r⌉ + 1 bonds; a power-of-two order needs none, so the formal proof's N = 15 cannot show a quantum link | N 21: exact from 4 bonds (r = 6) and 3 (r = 3) on two chips and every box placement, uniform guess at 0; N 15 a 7 and N 21 a 8: the ideal ½ at `max_bonds` 0 in every layout, and through the dual R1's hard cut; dual R1 with its benchmarked distant CZ: 0.414 of ½ (t = 4) | 10.19 |
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
| PyQrack | `hwxeb`, `run`, `selftest`, `aceplan`, `danmodel`, `shor9`; not `verify`, `seamgap` or `hamming`. ACE needs a build with `QrackAceBackend`; exact references above 32 qubits need Qrack built for > 32 qubits. `nighthawk_ghz.py stack --engine bridge` needs `QrackBridge` (PyQrack ≥ 2.30), or its `qrack_bridge.py` given with `--qrack-bridge` (runs on PyQrack 2.29.5 / Qrack 10.13.2) |
| OR-Tools | the `exact` placement engine, `--register` presets, `layoutsearch` |
| Qiskit | `--ace-transpile` |
| matplotlib | `nighthawk_graph.py` (Tk for the viewer); `nighthawk_3dgraph.py` (any interactive backend: TkAgg, QtAgg, GTK, macOS); `--grid3d` pages need a browser with WebGL, and three.js 0.160 from jsDelivr or a local npm `three` package (`--three DIR --serve`) |
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
python3 nighthawk_graph.py --grid3d nighthawk_3d --open             # shor9 registers in 3D (three.js from jsDelivr)
python3 nighthawk_graph.py --grid3d nighthawk_3d --three node_modules/three --serve   # offline
python3 nighthawk_3dgraph.py --repo ../rcs-nighthawk                # every shape in 3D, dark mode, one window
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
| `vqe` | transverse-field Ising H = −J Σ Z_uZ_v − h Σ X on the register's couplers; Hamiltonian-variational circuit: \|+⟩ⁿ, then per layer RZZ(γ_l) on colours A–D and RX(β_l) on every qubit, `--depths` = layers p; angles optimised once in numpy on `--vqe-opt-register` and used unchanged at every size | energy ⟨H⟩ shot-free: ⟨X_q⟩ after H on every qubit, ⟨Z_uZ_v⟩ after CX(u→v) on one colour's couplers (1 + 4 passes); next to the same circuit's energy and E0 in numpy and Anderson's lower bound (METHODOLOGY §6.5) |

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

`--register NAME` replaces the release's register with a sub-register of the 12 × 10 Nighthawk lattice. The circuits are built by the release's own code and seed: `rect_matchings`, `remove_qubits`, `make_patches(seed=2025, balance_tol=1, top_n=5)` for the K = 3, 4 partitions, and the release's input-string recipe (selftest [9]). Registers have no device data; they run `mirror`, `hamming` and `vqe` at any size, and `fxeb` up to 36 qubits. Each carries a stored ACE preset (geometry, caps, placement).

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
| 12 | `vqe`: HVA energy on the exact engine vs the numpy state vector (12 qubits, p = 3); p = 0 gives −h n; Lanczos E0 vs dense diagonalisation (10 qubits); bound ≤ E0 ≤ every energy | 1e-3 relative (Qrack is float32); 1e-8 |
| 13 | `shor9`: both 3 × 3 layouts' encoders inside ACE's coupling map, units tiled on the release register on its couplers; exact decoder, all of them, four logical states | every success 1.0, raw and decoded |

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
| `--vqe-h`, `--vqe-j` | 3.04, 1 | `vqe`: field and coupling; 3.04 ≈ the square lattice's critical point h/J = 3.044 |
| `--vqe-opt-register` | `nighthawk2p16` | `vqe`: register the angles are optimised on (exact, numpy); `self` = the run's own first `--vqe-opt-n` (16) qubits |
| `--vqe-exact-max`, `--vqe-ed-max` | 26, 24 | `vqe`: largest n for the numpy circuit energy (16 B × 2ⁿ) and for the Lanczos E0 |
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
| `--ace-boundary-rep`, `--no-ace-error-detection`, `--no-ace-crossbars` | off, on, on | ACE switches, always passed explicitly (PyQrack 2.29 turned its own boundary-code default on); non-defaults enter the config tag |
| `--variants SPEC …` | — | several ACE configurations scored against one exact reference (below) |
| `--cpu`, `--ace-gpu`, `--ref-gpu`, `--ace-gpu-shots`, `--ace-host-pointer` | — | engine placement (§5.2) |
| `--gtt GB`, `--vram-gb` | off, 8 | VRAM oversubscription (§5.4) |
| `--swap-gb GB` | off | CPU runs larger than RAM (§5.5) |
| `--gpus`, `--per-gpu` | —, 3 | `len(gpus) × per_gpu` workers through a claim-file queue |
| `--stagger`, `--init-gap` | 20 s, 10 s | spacing of worker starts and OpenCL inits |
| `--max-attempts`, `--retry-failed` | 2 | take-overs of a dead worker's point; clear failed marks |
| `--max-rss-gb` | off | per-worker memory cap: the point is marked failed |
| `--mem-budget-gb`, `--mem-ledger` | off, `/tmp/nighthawk_mem_ledger` | machine-wide budget shared by all runs (§5.1) |
| `--mem-per-point-gb GB` | worst case | book this much per point in the ledger instead of both patches fully entangled; for shallow points, which stay well below it (57 qubits, d = 4–6: ~40 GiB against ~81); pair with `--max-rss-gb` |
| `--order` | `deep` | `shallow`: cheapest points first |
| `--gpu-isolation` | `auto` | every GPU worker sees one OpenCL device; required for clvk |
| `--status`, `--summarize`, `--pack` | — | point list with progress and ETA; summary tables; rebuild release-format files |

**Shared reference.** `--variants 4/4:t3 4/4:t2 …` samples each `fxeb` point with every listed configuration and scores all of them against one exact reference. The tokens are lrc/lrr, then `torus`, `untiled`, `tiled`, `t2`, `t3`, `rep`, `noed`, `noxbar`, `tx` (exact), `txa` (anneal), `vf2`, `w<N>`, `ex<a+b+…>`, `xok`. Records stay per variant.

**Resumability.** A configuration is tagged `<backend>-<sha1[:8]>` over backend, shots, twirls, ACE parameters, tiling version and exact-probs mode, not the register size, so sweeps extend and resume per point. `--gtt` and `--swap-gb` do not change the tag. Bitstrings are written before the JSONL record, and the claim is marked done last. A record with a non-finite score (NaN) is ignored everywhere and its point reopened on the next run; `hamming` refuses to write one and fails the point instead. Records carry `device` (`cpu` / `gpu`), outside the tag. `vqe` adds J, h and the angle source to the tag and keeps its optimised angles and E0 values in `<out>_vqe.json` (file-locked, shared by workers), so run it in its own `--out`. To drop a pending point, write `failed: …` into its claim file under `<out>_shots/<cfg>/claims/`.

### 3.3 `danmodel`

| Mode | Command | Output |
|---|---|---|
| prediction | `danmodel [--registers …] [--depths …]` | bulk, boundary, patches, x and predicted XEB per register and depth; default the two-patch ladder, d = 4 … 36 |
| comparison | `danmodel --in run.jsonl …` | every measured mirror, `fxeb` or `hamming` point next to the prediction at its own seam and patch counts (`hamming` at the depth d of U, as his script labels it, λ = 1 only; the CSV adds `dan_2d` at the mirror's 2·d layers); plain mean over instances, worker files included |
| zero-noise | the same, on `hamming` runs with ≥ 2 folds | Dan's Richardson extrapolation: ln(signal) extrapolated to λ = 0 in u = λ^1.60569 |

Both modes add `ibm_phoenix`'s mirror survival at n = 61 (`--no-device` to omit) and write CSV with `--csv`.

### 3.3.1 `hamming`

`hamming [--in dan_hamming_*.jsonl …]` reads `run --families hamming` outputs (worker files included; default every `dan_hamming_*.jsonl` in the working directory). Files only, no Qrack: safe beside running jobs.

| Section | Content |
|---|---|
| per register | seam qubits, patches, couplers through the seam and per seam qubit (from the records' `ace_couplers`); per depth at λ = 1: plain mean ± circuit scatter, circuits, Dan at d and at 2d, ratio; fitted depth power and signal(12) from `--fit-from` (default d = 6), flagged when 3 depths or fewer |
| grid | measured ÷ Dan per depth, registers ordered by couplers per seam qubit; `(k)` marks cells with fewer than `--min-circuits` (default 10) circuits |
| folding | signal at every λ per depth with circuit counts, and Dan's Richardson extrapolation to λ = 0 in u = λ^1.60569 |

| Option | Default | Effect |
|---|---|---|
| `--protocol` | `nnqab` | records of this protocol only; `any` for all |
| `--fit-from` | 6 | first depth of the depth-power fit |
| `--min-circuits` | 10 | cells below this get their circuit count |
| `--csv FILE` | — | λ = 1 rows with coupler load, ratios, every λ's signal and the Richardson value |

A circuit rerun (retry, second pass) replaces its earlier record; records of several configurations for one register are merged and the header says so.

### 3.3.2 `shor9`

One unit of Shor's code (by Claude and Dan Strano) on a two-patch `QrackAceBackend`: m blocks of m qubits, [[m², 1, m]]. `--size 3` (default) is Shor-9 on the original 3 × 3 register; `--size 4` Shor-16, `5` Shor-25, `6` Shor-36, `7` Shor-49. The register is an m × m grid of two patches joined by one seam column of m qubits; the seam set and coupling map are read from the backend and every CX of the encoder is checked against the map (selftest [13]).

| m | Code | ACE geometry (flat) | Patches (bulk) + seam |
|---|---|---|---|
| 3 | Shor-9 | `long_range_columns = 1` (the original) | 3 + 3, seam 3 |
| 4 | Shor-16 | lrc 2, lrr 3 | 6 + 6, seam 4 |
| 5 | Shor-25 | lrc 2, lrr 4 | 10 + 10, seam 5 |
| 6 | Shor-36 | lrc 3, lrr 5 | 15 + 15, seam 6 |
| 7 | Shor-49 | lrc 3, lrr 6 | 21 + 21, seam 7 |

| Layout | Blocks | What crosses the seam |
|---|---|---|
| `seam_leaders` | the m seam qubits lead the blocks; each block is a seam qubit plus bulk qubits of both patches | every inner bit-flip block; the outer phase-flip layer is seam-to-seam (exact) |
| `patch_blocks` (odd m) | the seam column is one block, the others are cut from each patch's bulk | only the outer phase-flip layer. For even m the m(m − 1)/2 bulk qubits of a patch do not split into whole blocks: `seam_leaders` only |

| Encoder | Rule |
|---|---|
| outer | the input on the first leader, copied to the other leaders by CXs along a tree of couplings; then H on every leader |
| inner | each leader copied to its block by CXs along a tree of couplings |
| m = 3 | both trees are stars: CX a→b, a→c; H on a, b, c; each leader CX to its two members (the original script) |

| Readout | Rule |
|---|---|
| logical X (\|+_L⟩, \|−_L⟩) | all in Z; a block's bit is the majority of its m bits; logical X = parity of the block bits |
| logical Z (\|0_L⟩, \|1_L⟩) | H on all, then Z; a block's sign is the parity of its m bits; logical Z = majority of the m signs |
| tie (even m) | a tied majority is a coin flip, counted 0.5, so 0.5 stays "no logical information" |
| raw (no correction) | odd m: the parity of all m² outcomes (the original's); even m, where that parity is always even: Z basis the parity of the leaders' bits, X basis the first block's sign |

Success 1.0 is perfect, 0.5 carries no logical information; all four logical states run, so a sampler biased toward one answer cannot score.

Every layout (and every unit with `--register`) is also compared outcome by outcome with the exact distribution: uniform over the outcomes the state can give (Z basis 2^(m−1): every block all-equal with the right parity of block values; X basis 2^(m(m−1)): every block's parity the logical sign), checked against a numpy state vector for m = 3 and 4 (selftest [13]).

| Column | Meaning |
|---|---|
| TVD | total variation distance between ACE's histogram (all repeats pooled) and the exact distribution |
| floor | the same distance for exact samples of the same size: the shot-noise level. Over 2^(m(m−1)) X-basis outcomes it reaches 1 from m = 5 on (every shot is new), and only the impossible share says anything there |
| impossible | share of shots on outcomes the exact state never gives |
| top wrong | the most frequent of those, as the blocks' bits (leader first); `x` = the decoder fails on it, `tie` = tied majority |

| Option | Default | Effect |
|---|---|---|
| `--size` | 3 | m: Shor-m² (3 = Shor-9, 4 = Shor-16, 6 = Shor-36) |
| `--shots` | 8192 | shots per logical state |
| `--repeats` | 1 | fresh ACE backends per state, averaged; results differ from backend to backend by more than shot noise, so 5–10 for a mean (the spread is printed) |
| `--gpu` | off | ACE and the reference on OpenCL: the device `QRACK_OCL_DEFAULT_DEVICE` names among the ICDs the loader sees (a clvk V340 die, or PoCL's CPU device with `OCL_ICD_VENDORS` pointing at a directory holding `pocl.icd`) |
| `--ocl-sync` | off | with `--gpu`: `QRACK_OCL_SYNC=1` (Qrack ≥ 10.13.2, §5.3). An OpenCL run whose exact reference is not 1.0 everywhere stops with an error and records nothing |
| `--no-exact` | off | skip the exact `QrackSimulator` reference |
| `--ace-boundary-rep` | off | ACE's `is_boundary_repetition_code`, passed explicitly either way; PyQrack ≥ 2.29's version lifts the seam (RESULTS §10.3) |
| `--progress` | 30 | seconds between heartbeats: a start line (versions, settings), the plan (tasks, processes), a line per finished task (label, seconds, share done, running, ETA), and when nothing finishes for this long the longest-running task; 0 = off. Follow a log with `tail -f` |
| `--jobs` | 1 | worker processes, one core each; 0 = every core. One task per logical state × fresh backend (× layout, or × rotation with `--register`); a backend's shots stay in its task. The exact reference's shots split into chunks over fresh simulators (same state, same distribution). Workers inherit `QRACK_OCL_SYNC` and the clvk isolation; with `--gpu` they share one device |
| `--out FILE` | — | append one JSON record per layout (per unit with `--register`), with the distribution comparison per logical state; with `--register` also one `shor9_map` record (each qubit's lattice row, column, ACE region and simulators, each coupler's class) for `nighthawk_graph.py` view 17 |
| `--register NAME` | — | tile units over a whole register instead (below) |
| `--max-tiles` | all | with `--register`: at most this many units |

**Over a whole register** (`--register release|nighthawk61r2|nighthawk88e|fav5p112|…`): units are disjoint m × m squares of the register's lattice sites, leaders on the middle column or row, the outer tree along the leaders' line and each inner tree along its block, both from the middle outwards, so every encoder CX is one of the register's own nearest-neighbour couplers (checked). CP-SAT picks the most units. All units run in one circuit on the register's ACE preset; the logical states rotate over the units in four runs, so every unit is read in all four. Each unit is sampled on its own qubits (units are independent). Per unit: lattice position, the regions its qubits sit in (patch bulk or seam), its encoder CXs as exact / replica / cross, the four success rates and the distribution comparison; then means by placement and the exact reference over the whole register. Qrack keeps unentangled units apart and the encoder is a Clifford circuit, so memory stays small at any size (61 qubits: 0.05 GB peak).

**Sampling above 32 qubits.** `QrackSimulator.measure_shots` and `m_all` return bits 0–31 only (PyQrack 2.28.1). The exact reference of a unit wider than 32 qubits (Shor-36, Shor-49) therefore draws each shot from a clone, qubit by qubit; ACE's own shots carry every bit.

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

The cloverfield puts two or more chips side by side, joined by inter-chip couplers, as IBM plans with l-couplers: one register, one circuit, one survival.

| Part | Role |
|---|---|
| chip worker | one isolated process per chip and device; its register's ACE preset (or an exact simulator); executes gate batches and answers marginal and probability queries over a pipe |
| loom | the coordinator; builds no simulator (OpenCL hidden); generates the field's circuit by the release's recipe and does all IO between chips; ≈ 45 MiB |
| `--loom ghost` (default) | each chip holds a ghost of every partner its links touch; single-qubit gates and twirls go to every copy; the link CZ is exact on both chips; the partner's other CZs reach the ghost as a mean-field phase π·P(1). Records carry `ghost_drift` |
| `--loom mean`, `--loom sample` | mean-field phases only (chips in a product state); Z kicks drawn from P(1) |
| estimator | `prob` per chip; the field's survival is the product |

Options: `--chips` (`nighthawk88e*2`, `release,nighthawk88d`, `rect:r0-r1,c0-c1`, any register), `--arrangement RxC`, `--links K|all`, `--link-rows`, `--link-cols`, `--loom`, `--ghost-max-width`, `--gtt`, `--vram-gb`, `--backend`, `--chip-ace`, `--cpu`, `--ace-gpu`, `--gpus`, `--chips-per-gpu`, `--theta`, `--twirls`. Subcommands: `plan` (links, widths, memory; `--png`), `run`, `linksearch` (survival against link count), `layoutsearch` (passed to nighthawk_qrack), `selftest`.

| selftest | Check |
|---|---|
| 1 | field `nighthawk88e*2`, 4 links: colours are matchings, cuts deterministic |
| 2a | one chip, no links: loom op stream vs nighthawk_qrack's own calls |
| 2b | the same through a worker process |
| 3 | woven CZ on \|+⟩ against \|0⟩, \|1⟩, \|+⟩ |
| 4 | CZ·CZ = I across a link |
| 5 | loom error alone: two exact 9-qubit chips, 2 links, d = 8 / 12 / 16 |

Results: [RESULTS.md §11](RESULTS.md#11-cloverfield-e13).

### 3.6 `nighthawk_ghz.py`

Layers 1a and 2 of *Exact Refocusing of Diagonal Crosstalk for Shor Order Finding on a 10 × 12 Square-Lattice Processor: Analysis and Simulation* (numpy reference `shrs_nighthawk_stealth.py`, the 2 × 4 plaquette Π), run on a Nighthawk register's ACE preset placement, so the paper's exact statements meet ACE's seams. Standalone; imports `nighthawk_qrack.py` for registers, placement, coupler classes and the worker pool. Gates are exact; the departures from the paper's model are ACE's seam approximation and its sampler. An exact `QrackSimulator` reference runs beside every ACE run, with the same quasi-static draws.

| Paper | Here |
|---|---|
| Lemma 1: bipartite, A = {r + c even} | checked on every register (`sublattice` in the map record) |
| Def. 2: H_D = Σ J_ij Z_i Z_j + Σ (δω_i + ε_i) Z_i, η = 0 (A4) | J_ij = `--j-zz` × U(0.75, 1.25) per coupler (`--seed`), δω_q = 0.022 ((q mod 3) − 1), ε_i ~ N(0, `--sigma-qs`²) per realisation (`--repeats`); defaults = §8 |
| Def. 3, A2: global depolarising D_p | not applied: ACE has no global D_p. Lemma 3's λ then reports ACE's population loss as an effective survival (exact only under A2) |
| Def. 4: block = G, V, D_p | GHZ preparation, then `--blocks` L windows with G = I (Lemma 3's probe) |
| Def. 5: Walsh sequence, route B | `--layer2 dd` (aliases `walsh`, `anti-otoc`): U(T/4) X_B U(T/4) X_A U(T/4) X_B U(T/4) X_A in time order |
| Def. 6: compensation, route A | `--layer2 compensated`: e^{+iTĤ} e^{−iTH}, Ĥ from the true J, δω (oracle Layer 1b) |
| — | `--layer2 unmitigated`: e^{−iTH} |
| Lemma 3: λ = [(P0 + P1 − 2/d) / (1 − 2/d)]^{1/L} | `lambda_per_block` |
| Lemma 4: MQC at φ_m = πm/(n + 1), m = 0 … 2n + 1; c = DFT coefficient at n | settings `q0`–`q(2k+1)`; `C_parity` = 2\|c\|, `Phi_coh` = arg c, `F_GHZ` = (P0 + P1 + C)/2 = max_θ ⟨GHZ_θ\|ρ\|GHZ_θ⟩; also `F0` = ⟨GHZ_0\|ρ\|GHZ_0⟩ |
| Cor. 4.1: Σ δω = −Φ / (2LT) | `sum_dw_estimate` (a register-wide state; a tile also sees its outside couplings) |
| Cor. 2.1: GHZ blind to ZZ inside it | selftest [6] |
| Theorem 1: W_T(H_D) = I | selftest [5]; the exact reference under `dd` must be ideal on every deterministic shot |
| Def. 8, Lemmas 6–7, Cor. 7.1: OTOC | not estimated here (the reference computes it for Π); `dd` tests Cor. 7.1 through the GHZ numbers |
| Layer 1b, §6–7: conditional Ramsey, Shor N = 15 | not here (reference) |

Not in the paper: the Clifford witness F ≥ P0 + P1 + P(even X parity) − 1 (two commuting stabiliser projectors whose product is \|GHZ⟩⟨GHZ\|), used where MQC is too wide; the root-phase coherence `C_x`, `phase_x` (`xr0`–`xr3`, every x / x̄ pair, not only 0…0 / 1…1); the seam maps.

| Drift on ACE | Rule |
|---|---|
| coupling inside a GHZ unit | CX · e^{−iφZ_j} · CX: across a seam it goes through ACE's seam machinery, as the circuit's CXs do. `--inside-zz stabilizer`: not applied. Z_iZ_j stabilises the unit's GHZ and every Walsh-flipped image of it, so the gate is a global phase there (Cor. 2.1; selftest [13]); the exact reference runs the same circuit. GHZ units only |
| coupling to a qubit outside every unit | e^{−iφ z_j Z_i} on the unit qubit, z_j = ±1 that qubit's Z eigenvalue tracked through the X pulses: the same unitary (selftest [7]) without merging units |
| coupling between two units | none: under `--layer2` the tiles of a line are spread into groups with no coupler between units (a ZZ between two GHZ units entangles them, and the register would no longer factorise) |

| Command | State | Readout |
|---|---|---|
| `chip` | one GHZ over all n qubits: H on a root, CX along a spanning tree (`--trees bfs`: breadth-first from the centre; `seam`: Kruskal, fewest seam CXs = seam qubits + 1) | witness (`z`, `xr0`–`xr3`, Clifford) above `--mqc-max`, MQC below |
| `tiles` | disjoint k-qubit chains (`--unit k`) along lattice rows and columns, every orientation and offset | MQC: `z`, `q0`–`q(2k+1)`, 2k + 3 settings |
| `stack` | pass 1 (Layer 1a), pass 2 (Layer 2), Layer S (seam syndrome + repetition code), Layer V (the corrected measurement) on `tiles` or one `chip` state (`--mode`): below | as `tiles` / `chip` |
| `plot` | per-site loss map, tree and coupler classes, fidelity by seam distance (`--in`, `--layer2`, `--png`) | — |
| `selftest` | [1] Lemma 4 and the witness against numpy, [2] Lemma 4 on random ρ, [3] Lemma 3, [4] readout gates on Qrack, [5] Theorem 1, [6] Cor. 2.1 and 4.1, [7] outside-coupling identity, [8] fold = I, [9] ZNE amplification = I, [10] ZNE models on synthetic decays, [11] Layer S's seam classes, plan and the code's schedule, [12] the co-located authority step, [13] `--inside-zz stabilizer` = a global phase on a GHZ unit (dd, unmitigated, compensated), not the identity on a product state, and `probe`'s CX-target choices = the same unitary, [14] `probe`'s chain re-rooting, [16] `--engine bridge`: two GHZ units across one cut at `max_bonds` 0 / 1 / 2 (both ≤ ½, one carried, both exact), when `QrackBridge` is found; [20] `shoralg`: every controlled multiplication = the paper's U on every basis state (N 15 structured and generic, N 21 generic), the gate circuit = the analytic distribution, the paper's good bins {4, 12} and success ½, with `QrackBridge` r = 4 exact at `max_bonds` 0 and r = 6 exact only from 4, the dual router noiseless; [19] `dual`: 248 qubits and 444 couplers; the distant CZ's decomposition = CZ on every link (L-qubits back in \|0⟩); Bell 115–233 1 on the bridge, ½ at `max_bonds` 0; measured-link noise alone leaves Bell 111–237 at its distant-CZ fidelity; [18] `shor`: Shor-9 placements on the 5 × 5 × 5 box (96: 48 inside, 24 between blocks, 24 through) and, with `QrackBridge`, a unit on one direct edge: between blocks \|0_L⟩ 1 at `max_bonds` 0 and \|+_L⟩ ≈ ½ at 0, 1 at 1; through blocks \|+_L⟩ 1 at 0 and \|0_L⟩ ≈ ½ / ¾ / 1 at 0 / 1 / 3; [17] `--box`: the 5 × 5 × 5 box's geometry (98 qubits, 192 couplers, bipartite, 6 direct and 6 relayed edges, opposite faces never adjacent; 144 chains, 44 across an edge, covering every cross-face coupler) and, with `QrackBridge`, a chain across a relayed edge through the ring's wrap cut on the 3 × 3 × 3 box (≤ ½ at `max_bonds` 0, exact at 1); [15] `--engine loom`: LoomGates' matrices against Qrack's gates; with `nighthawk_cloverfield.py` and the release, a 3-chain across a link: on the ghost loom inside a ghost band ≈ 1 and on its own chips ≤ ½, and as a hard cut (the default) class cut and ≤ ½ whatever the read | — |
| `sweep` | `tiles` once per window count (`--windows 0,1,2,3,4,5,6,8`; 0 = no Layer 2, dd otherwise, overriding `--layer2`, `--blocks`), again under the boundary code at `--code-at`; then the means per (setup, L, seam class): units, F_GHZ, P0 + P1, C, \|P1 − P0\|. `--summarise FILE…`: the means of records already written, no runs; `--summary FILE`: the means as JSON. Records: `tiles`' `ghz`, to `--out` (default `ghz_sweep_<register>.jsonl`) | as `tiles` |
| `probe` | the `--seam-class` units (default `cut`) of every spread tiling through Layer 2 (dd unless `--layer2`), per `--variants`: `none` (no Layer 2), `gate` (inside ZZ as CX · Rz · CX), `tseam` / `tbulk` (the CX's target on the seam / bulk qubit), `stab` (dropped), and `chain-none`, `chain-gate`, `chain-stab` with the GHZ prepared from one end of the chain; `--exact-variants` (default `gate,stab,chain-stab`) also on the exact simulator. Prints means and per-unit ranges; records `ghz_probe` (not read by `nighthawk_graph.py`) | as `tiles` |
| `stackmap` | the latest `stack` run in `--in` as three heatmaps on the chip layout (pass 1, 2, V, or 1, C, E): per site the lowest F_GHZ of the chains through it, seam qubits ringed, the loom's links as bars, class means under each panel (`--png`); side by side up to 10 columns, stacked above that. A `--box` run: three unfolded nets of the box (base in the middle, walls folded out, lid beyond the back wall), bars on every coupler across a box edge (solid: the two faces share a ring cut; dashed: relayed through the face between), the edges the net cuts open lettered on both faces, means per unit inside a face / across a direct cut / relayed | — |
| `shor` | Shor's [[m², 1, m]] code instead of GHZ, across QrackBridge circuits (`--engine bridge`: `--loom-chips`, or the box with `--box L H`): every placement of an m × m unit (`--size m`) whose encoder CXs are couplers of the field, each in its own circuit (a bridge holding the unit's qubits, one patch per chip), all four logical states encoded as `nighthawk_qrack.py shor9` encodes them (its `shor9_prepare`); raw and decoded success, the outcome distribution against the exact one (`shor9_compare`), the bond qubits per cut the prepared state keeps, the exact reference (`QrackSimulator` of the unit); a table by how the cut meets the code (inside one circuit, between its blocks, through them; on the box direct or relayed, and the wrap cut) | logical Z: X basis, logical X: Z basis, as `shor9` |
| `shormap` | the latest `shor` run in `--in` as heatmaps: logical Z and logical X side by side, one row per way the cut meets the code (between blocks, through blocks); per site the lowest decoded success among the units through it of that layout or inside one circuit; the box on its net (`--png`) | — |
| `dual` | the dual Nighthawk R1 of arXiv:2610.29986 (Sundaresan et al., IBM): two 120-qubit modules (12 × 10 square lattices, 218 couplers each; 10 r + c on module A, 120 + 10 r + c on B) and the paper's four L-couplers (L-qubits 240–243 on A, 244–247 on B, each coupled to a row-11 lattice qubit, cables 240–247, 241–246, 242–245, 243–244). The distant CZ as the paper builds it: CNOT lattice → L-qubit, state transfer across the cable (a SWAP across the bridge), CZ with the far lattice qubit, transfer back, CNOT. The paper's circuits: a Bell pair per L-coupler (H, one distant CNOT), both pairs on 242–245 and 243–244 at once, GHZ-4 (115, 116, 232, 233) and GHZ-6 (115–117, 231–233) rooted at Q115. Models: `hardcut` (bridge at `max_bonds` 0), `ideal` (bridge, exact), `exact` (QrackSimulator), `budget` (a depolarising channel after every constituent at the paper's benchmarks: transfer 1 − F per link, L-qubit EPG, `--local-cz-error`, `--lattice-1q-error`), `measured` (the distant CZ exact, then a two-qubit depolarising channel at its benchmarked error); noise as Pauli trajectories, fidelities from the trajectory-averaged density matrix (no readout, no shot noise): Bell ⟨Φ⁺\|ρ\|Φ⁺⟩, GHZ (P0 + P1 + 2\|ρ₀,₁\|)/2, the paper's estimators. Printed and recorded against IBM's raw and readout-mitigated values (`--out`, `--png`) | exact from the state |
| `dualmap` | GHZ fidelity heatmaps over both modules of the dual Nighthawk R1 and their L-couplers: 3-qubit chains tiled along every row and column of each 120-qubit lattice (392) plus every chain through an L-coupler (the link's lattice pair and a lattice neighbour of either end: 24), each its own circuit on the two-patch bridge, under the `dual` models (`--models`, default `hardcut,budget,measured`; `--trajectories` 400 per lattice chain, `--trajectories-link` 4000 per link chain); per site the lowest F_GHZ of the chains through it, module B drawn rotated by 180° with its row 11 facing A's, the L-qubits as grey cells and the cables labelled with their distant CZ (`--png`, `--out`) | MQC estimator from the exact density matrix |
| `dualplot` | `dual`'s plot from its records (`--in`, `--png`) | — |
| `shoralg` | Layer 3, the formal proof's Shor order finding (`shrs_nighthawk_stealth.py`: N = 15, a = 7, 4 counting + 4 work qubits, per-shot factoring success), as gates (H, X, phase, CX; each controlled multiplication by a^(2^k) mod N synthesised as the paper's unitary on every basis state), on every engine: QrackSimulator, QrackBridge over the two chips (`--loom-chips`), the box's faces (`--box L H`) or two bare patches (`--pair`) at every `--bonds` capacity, or the dual Nighthawk R1 (`--dual`: routed by SWAPs on each module's lattice, distant CNOTs through one cable, `dual`'s models). Layouts (`--layouts`): `inside` one circuit; `split` counting \| work with a fan-out ancilla (2 CX across per multiplication); `direct` counting \| work, every controlled gate across; `countsplit` the cut through the counting register (the inverse QFT crosses); `worksplit` through the work register. Per circuit: success by the paper's rule, the order found, the distance and classical fidelity to the analytic distribution, the ideal circuit's and a uniform guess's success, bonds the rule needs, peak bonds, truncations; `shor_alg` records | P(y) from the ket (bridge ≤ `--ket-max` qubits) or shots |

| Option | Default | Effect |
|---|---|---|
| `--register` | `nighthawk61r2` | any register of `nighthawk_qrack.py` or saved by its layoutsearch |
| `--shots`, `--repeats` | 4096, 3 | per setting and realisation; a realisation is a fresh ACE backend and ε draw (the paper's K) |
| `--estimator`, `--mqc-max` | `auto`, 12 | MQC up to 12 qubits, witness above; `both`; `mqc`; `witness` |
| `--layer2`, `--blocks`, `--t-idle`, `--j-zz`, `--sigma-qs`, `--seed` | none, 3, 4.4, 0.065, 0.015, 42 | Layer 2 mode and the §8 model |
| `--inside-zz` | `gate` | Layer 2: the drift's ZZ between two qubits of one unit as CX · Rz · CX (`gate`), or dropped (`stabilizer`: a global phase on the GHZ, no seam crossings in the drift). Records carry `layer2.inside_zz`; `nighthawk_graph.py` shows the setup as ", inside ZZ by stabiliser" |
| `--layers` | none | per-unit hook, logical identities that add seam crossings: `fold:F[:all]` F CX pairs after every seam CX (every CX); `zzfold:K[:all[:φ]]` K × ZZ(φ) ZZ(−φ); `gfold:K` the preparation undone and redone K times |
| `--stabilizer` | `auto` | tableau patches only for `chip` with Clifford readout and no Layer 2 |
| `--exact-shots`, `--no-exact` | 128 | the reference's shots per setting and realisation |
| `--force`, `--ace-boundary-rep`, `--gpu`, `--ocl-sync`, `--jobs`, `--progress`, `--out` | | as `shor9`; `--force`: dense chip above 30 qubits |
| `--code-authority` | `lineage` | the boundary code's authority rule wherever the code runs (`--ace-boundary-rep`, `stack`'s Layer V): `lineage` = PyQrack 2.29.5's own; `colocated` = lineage, and a seam qubit whose authority moves to a simulator takes its seam partners' authority with it (below) |

| Record field | Meaning |
|---|---|
| `P0`, `P1`, `C_parity`, `Phi_coh`, `F_GHZ`, `lambda_per_block`, `sum_dw_estimate` | Layer 1a, the reference implementation's names |
| `F0`, `F_witness`, `C_x`, `phase_x`, `c_re`, `c_im` | fidelity to GHZ_0, the witness, root-phase coherence, c |
| `exact_*` | the same from the exact reference |
| `edge_break`, `coupler_break`, `site_break` | Z-basis maps: per tree CX, per coupler, per qubit (against the unit's majority) |
| `layer2` | mode, blocks, T, J scale, σ, seed, η = 0 |

**Seam layers (`stack`).** Two layers after Layer 2, named S and V to stay clear of the paper's Layer 3 (the Shor payload). The paper has no seams; Layer S reads ACE's seam error from the first two passes and corrects what a code can correct, Layer V measures the GHZ states a third time.

| Step | Does | New circuits |
|---|---|---|
| pass 1 | Layer 1a: the units' GHZ states, no drift | every group, ACE + exact |
| pass 2 | Layer 2 in `--layer2` (`dd` when none is given) on the same units | every group, ACE + exact |
| Layer S, extraction | per unit and pass the seam syndrome against the exact reference: the share of P0 + P1 and of the coherence (\|c\|, or ⟨X…X⟩) ACE keeps, the fidelity lost (and in standard errors), the qubit whose Z value leaves the unit's majority most often and whether it is a seam qubit, which branch loses (P1 − P0); per unit its seam class (below); the seam-crossing fits (log(kept) = a + b·n: e^a what the first crossing leaves, e^b what each further one keeps); per seam coupler its Z break rate | none |
| Layer S, correction | every replica unit that lost F (> 0.02 and 3σ below the exact reference, pass 1 or 2) gets the replica repetition code with authority decoding (ACE's `is_boundary_repetition_code`: zero ancillae, the replica that received the real gate is the decoder's ground truth, stale replicas are flipped or rotated to it) and the code's schedule (from every qubit its CXs into the host simulator first, into seam qubits after), or `--code-schedule off` with `--code-authority colocated` (below). Cut units: no code | — |
| Layer V | the third GHZ measurement: the replica units under the correction, the cut units without it (the control: what a rerun moves), pass 2's Layer 2 and fresh ε draws, printed unit by unit beside pass 1, pass 2 and the exact reference | the units planned code or control, ACE |
| ZNE (`--zne-k`) | for comparison: the seam units rerun pass 2 with every seam crossing scaled by λ (`AmpGates`: K × ZZ(φ) ZZ(−φ) after each, λ = 1 + 4K; `--zne-amp cx`: K × CX CX, λ = 1 + 2K); P0 + P1, Re c, Im c (or ⟨X…X⟩) extrapolated to λ = 0, linearly and exponentially | seam units × each K, ACE |

| The seam between chips | ACE seam (inside a chip) | Loom, `--loom-read cut` (default) | Loom, `--loom-read ghost` | QrackBridge (`--engine bridge`) |
|---|---|---|---|---|
| what crosses | replicas in both patches and the crossbar, reconciled per coupling: part of the coherence | nothing: chips are independent simulators | nothing; each chip also simulates its partners' anchor qubits | entanglement, through bond qubits (virtual Bell pairs), up to `--bridge-max-bonds` bond qubits per cut |
| a CX through the seam | real where a simulator holds both copies, classical shadow elsewhere | the mean-field phase π P(1) of the other side, on each side | exact between a qubit and the partner's ghost on both chips; a neighbour's CZ reaches a ghost as the mean-field phase | gate teleportation: CNOT onto a fresh bond, H and the controlled gate on the far side, then an MPS recompression of the cut (exact when only zero Schmidt values are dropped) |
| a 3-chain across the seam, before any code | 0.55–0.95 (replica), 0.25–0.40 (cut): a slope | 0.50: separable across the cut, the bound for any such state | 0.26 read on its own chips | 1.00 within capacity; units beyond it lose their share (`max_bonds` 0 = the hard cut) |
| Layer S's code | the boundary repetition code, authority decoding: replica units to 1.00 | nothing to correct across the loom (ACE's code inside ACE chips) | reading a unit on the chip whose ghost band holds it: 1.00, because that chip simulated it whole; a band of two columns covers every 3-chain, not every 5-chain | none: the bridge is exact up to its truncation, which it reports (`fidelity_estimate`) |
| unit classes | bulk, replica, cut by the simulators holding every qubit | by the (chip, simulator) pairs holding the real qubit: every unit across the loom is cut | by the (chip, simulator) pairs holding a copy, ghosts included | units on two chips: class bridge (rerun as the control) |

A unit across a hard cut is separable there, so its F_GHZ is at most ½ (F_GHZ > ½ witnesses genuine multipartite entanglement); the ghost band does not change that, it only moves the cut for the units it covers (RESULTS §10.14).

| Seam class | Unit | Seam error | Layer S |
|---|---|---|---|
| bulk | one simulator holds it, no seam qubit | none | — |
| replica | one simulator holds it; its seam qubits enter through their replica there | a seam qubit's replicas disagree: a bit flip of the seam qubit against its unit, often one-sided (one branch loses) | code |
| cut | no simulator holds it | the GHZ needs coherence between two simulators; ACE joins them only through replicas and classical shadows, so c = 0 there and F_GHZ ≤ (P0 + P1)/2 ≤ ½ for any decoder | control |

A chip-wide GHZ over a register of two or more patches is always cut.

| The code's authority (`--code-authority`) | After a CX between two seam qubits | Needs the schedule |
|---|---|---|
| `lineage` (PyQrack 2.29.5) | the CX runs in every simulator and the crossbar pair becomes the sole authority of both; when one of them then couples into a patch its authority moves there, its partner's stays on the crossbar, whose copy is correlated with a stale copy | yes: CXs into the host simulator first |
| `colocated` | as `lineage`, then the partner's authority moves with it, to its copy in the same simulator, which shares the whole history | no |


| Option | Default | Effect |
|---|---|---|
| `--mode` | `tiles` | `tiles` (with `--unit`, `--orient`, `--offsets`) or `chip` (with `--trees`, `--root`) |
| `--code-schedule` | `on` | the code's schedule for the code units, or `off`: the preparation's order |
| `--engine` | `ace` | `ace`: the register on ACE's patches. `loom`: chips joined by `nighthawk_cloverfield.py`'s ghost loom as the seam; `tiles` only, no ZNE |
| `--loom-chips`, `--loom-link-rows` | `rect:1-8,2-5 rect:1-8,6-9`; auto | the chips (cloverfield `--chips` tokens, side by side in a row) and the rows that get a link. Default rows: every row whose facing border pair are lattice neighbours once the chips sit side by side. The default chips are `nighthawk61r2`'s 61 qubits as two chips of 30 and 31, the size of its two ACE patches, linked on rows 1–4, 6–8 (row 5: qubit 55 is broken), so the field has `nighthawk61r2`'s coupling graph |
| `--box L H` | — | with `--engine bridge`: a closed box instead of `--loom-chips`. The lattice points on the surface of an L × L × H block (L, H ≥ 3; 5 5: 98 qubits), couplers between points at unit distance; six faces, each one circuit and one QrackBridge patch: the base (z = 0) and lid (z = H − 1) whole, the four walls sharing the vertical edges round the box (front y = 0 holds x = 0 … L − 2, right x = L − 1 holds y = 0 … L − 2, back y = L − 1 holds x = 1 … L − 1, left x = 0 holds y = 1 … L − 1). The patches close into a ring (`wrap=True`) in the order base, front, right, lid, back, left: opposite faces three apart, so six box edges are a ring cut (base–front, front–right, right–lid, lid–back, back–left, left–base) and six are relayed through the face between (base–right via front, front–lid via right, lid–left via back, base–back via left, front–left via base, right–back via lid), one bond on each of the two cuts. Bipartite under x + y + z (the Walsh echo's sublattices). Chains of `--unit` run along straight lines in x, y and z (`--orient rows,cols` means all three here), so a unit crosses a box edge where its line runs off one face onto the next. Records carry `loom.box` (L, H, the ring, every edge's couplers, ring distance and relay face) and per unit `box` (faces, edge, ring cuts); the stack prints F_GHZ per box edge. The ring's wrap cut is truncated by a local SVD, not a projection: below capacity it carries less than the inner cuts and can zero the state; every bridge circuit runs with numpy's divide / invalid errors raised and stops with the circuit named rather than sample a NaN state (RESULTS §10.16) |
| `--size`, `--shor-units` (`shor`) | 3, `all` | m of Shor-m² (3 = Shor-9, 4 = Shor-16, 5 = Shor-25); `cross` keeps only the units across a cut (a unit inside one chip runs on one patch and never meets the bridge; Shor-16 and larger inside units are slow to sample there). Placements: on chips every m × m square of lattice cells, blocks along rows or columns; on the box m consecutive points along an axis inside the box's extent times m consecutive points round its cross-section, so a unit bends over the edges it crosses (the 8 corner points are in none). `shor`'s defaults: 1024 shots, 1 realisation (the bridge is deterministic up to its sampler), exact 256 |
| `--models`, `--trajectories`, `--batch`, `--local-cz-error`, `--lattice-1q-error`, `--seed` (`dual`) | `hardcut,ideal,budget,measured`, 2000, 100, 0.00355, 0.0005, 7 | the models above; Pauli trajectories per noisy circuit, split into batches (the standard error is over batches); the error of a CZ inside a module (default: the paper's budget for CZ111-237, its three local CZs 40 % of 2.66 %; the paper gives no lattice CZ benchmark) and of a lattice single-qubit gate (not in the paper; L-qubits take Extended Data Table 1's EPG). Per link: transfer F (Extended Data Table 2) and distant CZ F (Fig. 3b: 96.42 and 95.54 from the text, 91.2 for 113–235 and 95.8 for 117–231 read from the figure, together the text's mean 94.75) |
| `--N`, `--a`, `--counting` (`shoralg`) | 15 21; 7 for N = 15, else the smallest base of every order r > 1; 4 and 2n for N = 15, else 2n | the numbers, bases and counting widths, every combination (gcd(a, N) must be 1) |
| `--layouts`, `--bonds`, `--faces`, `--repeats` (`shoralg`) | all five; `--bridge-max-bonds`; the two chips, or `base/front base/right base/left` on the box; 1 | layouts as above; `max_bonds` values swept; side pairs A/B (counting side first; on the box direct, relayed through front, the ring's wrap cut); bridge realisations per circuit (below capacity a greedy truncation picks among equal Schmidt values, so the outcome depends on the seed; P is their mean, per-realisation values in `success_runs`, `tvd_runs`) |
| `--synth`, `--shots`, `--ket-max`, `--no-exact`, `--pair` (`shoralg`) | `auto`, 4096, 22, off, off | multiplication synthesis: `generic` (the permutation's transpositions, multi-controlled X on n − 2 ancillas), `structured` (N = 2^n − 1, a^(2^k) = ±2^j: a bit rotation by controlled SWAPs, the same unitary), `auto` (structured where it applies); shots when the bridge is wider than `--ket-max`; skip the QrackSimulator runs; two bare patches instead of chips |
| `--dual`, `--models`, `--trajectories`, `--link`, `--local-cz-error`, `--lattice-1q-error`, `--seed` (`shoralg`) | off, `hardcut,ideal,budget,measured`, 200, `242-245`, 0.00355, 0.0005, 7 | on the dual Nighthawk R1 under `dual`'s models (layouts inside, split, direct); Pauli trajectories per noisy circuit; the L-coupler every distant CNOT uses; as `dual` |
| `--loom-read` | `cut` | `cut`: the loom as a hard cut (the cloverfield's `--loom mean`, no ghosts, every qubit read on its own chip). `ghost`: the cloverfield's ghost loom, a replica band two columns wide; a unit inside one chip's band is read there. Records carry `loom.read`; `stackmap` labels the mode |
| `--engine bridge`, `--bridge-max-bonds`, `--bridge-truncation`, `--bridge-lazy`, `--qrack-bridge` | —, 6, `greedy`, 0, auto | the chips of `--loom-chips` (exact) joined by PyQrack's `QrackBridge`, one patch per chip, instead of the loom. Each circuit's bridge holds the qubits its units touch, the patches padded to equal size (the bridge splits its qubits into equal contiguous blocks); idle qubits stay \|0⟩ and in product, their couplings classical rotations, as on ACE. `max_bonds`: bond qubits kept per cut (capacity 2^max_bonds Schmidt values; 0 is the hard cut); `greedy` or `born` truncation; `lazy_bonds` buffered before one recompression. Per pass the log and a `ghz_bridge` record give the peak bond count, the truncating circuits and the bridge's `fidelity_estimate` |
| `--loom-chip-backend` | `exact` | `exact`: each chip a `QrackSimulator` (no schedule). `ace`: each chip its register's two-patch ACE preset (chips must be registers, e.g. `nighthawk61r2 nighthawk61r2`), ghosts on free ACE sites coupled to their anchor (the cloverfield's `place_ghosts`): ACE's seam inside each chip, the loom between them; Layer S's code is ACE's boundary repetition code inside (with `--code-authority`, `--code-schedule`) and the loom's authority across |
| `--cloverfield` | auto | `nighthawk_cloverfield.py`: `$NIGHTHAWK_CLOVERFIELD`, next to this script, or `../rcs-nighthawk` |
| `--order` | `echo-code` | `echo-code`: pass 2 (Layer 2), then Layer S from passes 1 and 2, then Layer V. `code-echo`: Layer S from pass 1 alone, pass C (pass 1's circuit, the replica units under the code, the cut units without), then pass E (Layer 2 on every unit, the code kept where it ran). The drift arrives with its echo in both orders, so pass E runs Layer V's circuit; `--zne-k` needs `echo-code` |
| `--zne-k` | none | ZNE identities per crossing, e.g. `1,2`: λ = 1, 5, 9 with `zz` |
| `--zne-amp`, `--zne-phi` | `zz`, 0.3 | ZNE amplification `zz` or `cx`; the ZZ angle (about one window's J·T) |

`--ace-boundary-rep` is refused: passes 1 and 2 measure the seam uncorrected, and Layer S switches the code on itself.

| Record (`stack --out`) | Holds |
|---|---|
| `ghz` with `stack_pass` 1, 2, V (`--order code-echo`: 1, C, E) and `stack_order` | the three measurements as ordinary records, with `seam_class`, `seam_plan`; V, C, E: `boundary_rep` true where the code ran, `code_authority`, `code_schedule`, the `edges` as run; V the exact fields of pass 2, C those of pass 1, E its own. `nighthawk_graph.py` reads 1, 2, E and the corrected V and C (setup ", boundary code"), not the V and C controls, which repeat pass 2 and pass 1 |
| `ghz_stack`, per unit | `stack_order`, `seam_class`, `seam_plan`, `syndrome` (pass 1, pass 2; code-echo: pass 1), `seam_crossings` (prep, drift), `pass1`, `pass2`, `passV` (with `boundary_rep`, `code_authority`, `code_schedule`, `edges`), `exact1`, `exact2`; code-echo: `pass1`, `passC` (seam units), `passE`, `exact1`, `exactE`; with `--zne-k`: `layerS` (λ, the observables at each λ, `F_at`, both fits), `F_zne_linear`, `F_zne_exp` |
| `ghz_seam_model` | Layer S's extraction and plan: `plan` (per class: units, plans, P0 + P1 and coherence kept, P1 − P0, breaks on a seam qubit, per pass), `fit` (`pop`, `coh`), `couplers` (a, b, class, break rate pass 1, pass 2) |

A code corrects ACE's seam where ZNE cannot: the loss is set at the first crossing, so no amplifier scales it (RESULTS §10.6), while the replica code removes it from every replica unit and leaves the cut units, which no code reaches, where a rerun leaves them (§10.7).

**Stabilizer patches.** ACE's default dense patches hold a chip-wide GHZ as 2^width amplitudes per patch. Preparation and Clifford readout keep a stabilizer-hybrid patch a tableau, but ACE's seam corrections are not Clifford: on PyPI's PyQrack 2.29.5 / Qrack 10.13.2 the tableau holds on the two-patch ladder up to `nighthawk2p57` (internal simulators 41/41/9, < 1 s per 8 shots), while `nighthawk61r2` (43/43/8) leaves it in sampling (Z: a zero-probability forced replica measurement; X: no shot in 15 min). Not yet traced. MQC (not Clifford) and Layer 2 (Z rotations by arbitrary angles) run dense; `tiles` stay small because QUnit keeps every chain separate.

**Views.** `nighthawk_ghz.py stackmap --in FILE`: one stack run, three heatmaps. `nighthawk_graph.py --ghz FILE` (or harvest): 18 tiles on the lattice per register and Layer 2 mode, 19 fidelity by placement with one bar per mode and the exact reference, 20 ACE against exact per unit, 21 chip-wide break maps per tree, 22–25 as in the views table below; `--table FILE` writes `FILE_ghz.csv` (with the engine columns `engine`, `loom_read`, `chip_backend`, `chips`, `max_bonds`, `truncation`, `lazy_bonds`, `box`, `box_edge`, `box_cuts`, `chips_spanned`, `inside_zz`, `stack_order`, `stack_pass`, `seam_class`, `seam_plan`), `FILE_shorbridge.csv` (per unit: layout, hops, wrap, bonds for Z and X, decoded and raw per logical state), `FILE_dual.csv` (per state and model, with IBM's values), `FILE_dualmap.csv` (per chain and model), `FILE_shoralg.csv` (per circuit: layout, placement, `max_bonds` or model, success, order, TVD, peak bonds, routing) and prints the summaries. Loom, bridge and box registers also get a `--grid3d` page, the patches being the chips. `--grid3d DIR` (or the viewer's 3D button) also writes `<register>_ghz_3d.html` per register: every site a column of height (v − 0.5)/0.5, v the mean F_GHZ of the chains through it or a chip-wide state's agreement with its majority; one switch per Layer 2 mode and chip-wide tree, ACE or the exact reference, the chains through a clicked site, the patches pulled apart into ACE's simulators as on the shor9 pages.

**Estimator bias.** C = 2\|ĉ\| is biased upward at finite shots, so F_GHZ of a perfect unit scatters around and above 1 (± 0.013 at 256 × 2 shots per setting); `F0` and the witness are linear and are not.

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
| 12 | `hamming`: Dan's Hamming signal against depth per register, folds λ = 1 / 3 / 5, with his calculator |
| 13 | `hamming`: measured ÷ calculator against register size per depth; effective folding exponent q against his 1.606 |
| 14 | `shor9`: decoded logical Z and X against code size per layout and setup (PyQrack version, boundary code, OpenCL); backend-to-backend standard error, every fresh backend as a dot |
| 15 | `shor9`: per logical state, impossible-outcome share and distance to the exact distribution against its shot-noise floor |
| 16 | `shor9 --register`: each unit's decoded Z / X and Z-basis impossible share against its encoder CXs through the seam; units matched by their square, pooled over runs |
| 17 | `shor9 --register` on the lattice, per register, code size and setup: qubits by ACE patch, seam qubits and the seam lines (a band through neighbouring seam qubits, labelled with the two patches it joins), couplers by class (exact / replica / cross), block leaders, each unit's square filled by min(Z, X) and labelled Z / X / impossible share. Map from the run's `shor9_map` record, else rebuilt (release + PyQrack), else the bare lattice |
| 18 | `nighthawk_ghz.py tiles` / `stack` on the lattice, per register, unit size, Layer 2 mode and setup: each site the mean F_GHZ of the units holding it; seam qubits and lines, couplers by class, the links between chips (loom / bridge) in black; box runs on the unfolded net. The subtitle names the engine's parameters (`max_bonds`, box ring, loom hard cut or ghost band, code first, inside ZZ by stabiliser) |
| 19 | mean fidelity by placement, one bar per Layer 2 mode, the exact reference as a tick; placements include across the loom, across the bridge, across a box edge (direct / relayed, wrap cut) |
| 20 | every unit's engine read (ACE, loom or QrackBridge) against its exact reference, per Layer 2 mode, by placement and register |
| 21 | `chip`: chip-wide break maps per tree |
| 22 | engines: mean F_GHZ per placement against QrackBridge `max_bonds`, one panel per geometry (box, two chips), the loom hard cut at the left |
| 23 | `shor` across QrackBridge: decoded logical Z (dots) and X (squares) against `max_bonds` per geometry and code size, by how the cut meets the code (inside one circuit, between blocks, through blocks, through the box ring's wrap cut), with the majority prediction for min(m, `max_bonds`) carried block signs |
| 24 | `dual`: every Dual Nighthawk R1 state under hard cut, exact bridge, constituent budget and measured distant CZ, with IBM's raw and readout-mitigated fidelities |
| 25 | `dualmap`: both modules (B turned 180°) with L-qubits and cables (labelled with distant-CZ fidelity), each site the lowest F_GHZ of its 3-qubit chains, one panel per model |
| 26 | `shoralg` on the bridge: per geometry, N, a and t, the per-shot factoring success (the order found where r is odd) of every layout and placement against `max_bonds`, the ideal circuit and a uniform guess as lines, the bonds the rule needs as a band |
| 27 | the same, TVD to the ideal distribution (shot-noise floors dotted) |
| 28 | `shoralg --dual`: per case, every layout's success under each model, with the distant / local CX and SWAP counts of its routing |

Views 8–11 and `--table FILE` (CSV of every point) show one ensemble at a time (`--theta haar|nnqab|all`), since XEB values of different ensembles are not comparable. `--table FILE` also writes `FILE_shor9.csv` (one row per code, layout and setup, one per register unit, with the top wrong patterns) and prints the `shor9` summary. Views 14–16 read `shor9` records from the harvested `*.jsonl` or from `--shor FILE` and need no BlueQubit release; records without `boundary_rep` are taken as PyQrack's own default (on from 2.29).

`--grid3d DIR` (or the viewer's *Open shor9 registers in 3D*) writes view 17 as a rotatable page per register and code size, with `index.html`:

| Element | Shows |
|---|---|
| lattice | the 12 × 10 Nighthawk grid, broken qubits 17, 55, 62, sites outside the register |
| qubits, couplers | colour by ACE patch; seam qubits split in the two patches they join; couplers exact (patch colour), replica (black), cross (red); seam lines |
| unit columns | one per unit, height (decoded success − 0.5) / 0.5 of logical X, Z or the weaker one; colour red 0.5 → green 1.0; grey outline at 1.0 |
| setups | one button per setup in the records (PyQrack version, boundary code, OpenCL); units matched by their square |
| *Pull the patches apart* | each patch on its own layer (its simulator); seam qubits stay in the crossbar and reappear as copies in every patch they join, joined by pillars |

| Option | Effect |
|---|---|
| `--grid3d [DIR]` | write the pages (default `./nighthawk_3d`) and exit |
| `--open` | open them in the default browser |
| `--three DIR_OR_URL` | three.js from a local npm `three` package (copied next to the pages) or another URL; default jsDelivr |
| `--serve [PORT]` | serve the pages on 127.0.0.1 (default 8017) and open them; needed with a local `--three` (module scripts do not load from `file://`) |

### 4.1 `nighthawk_3dgraph.py`: interactive 3D

One matplotlib window (any interactive backend), no browser: a scene per shape, its data sets and measures stepped with ◀ ▶ or the keys. It reads records the way `nighthawk_graph.py` does (it imports it): every `*.jsonl` in the working directory, or the files given with `--out`, `--shor` and `--ghz`. Run records need `--repo` for their lattice. Without a display, or with `--save DIR`, it writes PNGs.

| Shape | Geometry | Data sets | Measures (per site, or per cell of a landscape) |
|---|---|---|---|
| runs, landscape | depth × register size, one column per cell | per family (mirror, patched, fxeb, hamming, …), configuration and K | every numeric field of the records, averaged over instances: `fidelity`, `se`, `seconds` (and log10), `hog`, `xeb_linear`, `survival`, `ace_seam_used`, `ace_sims_used`, … |
| runs, bits on the lattice | the release's 61-qubit lattice and its couplers | per configuration, family, n and depth whose shots file exists (read when shown) | mirror bit-flip rate against the prepared string; P(1) and \|P(1) − ½\| |
| `shor9` register | lattice by ACE patch, seam qubits, couplers by class | per setup | logical Z, logical X, the weaker of the two, Z-basis impossible share |
| GHZ register | ACE patches; loom or bridge chips (each outlined, pulled apart sideways); the box folded into a cube, columns standing out of its faces | per unit size, Layer 2 mode and setup (engine, `max_bonds`, box ring, stack order); chip-wide states per tree | F_GHZ of the engine's read and of the exact reference, their difference, lowest F through the site, coherence C, population, units through the site; chip-wide: agreement with the majority. Coupler telemetry: each coupler's break rate |
| Shor across QrackBridge | as the GHZ register (chips or box) | per code size, `max_bonds`, truncation | decoded and raw logical Z and X, the weaker, bonds kept for Z and X, peak bonds, impossible share |
| Shor algorithm (Layer 3) | landscape: `max_bonds` (or the dual's model) × layout and placement, per geometry | per (N, a, t) | factoring success, order found, TVD to the ideal, peak bonds, truncations, CX across the cut, the bridge's fidelity estimate |
| dual Nighthawk R1 | both 12 × 10 modules, the L-qubits, the four cables as arcs labelled with their distant-CZ fidelity | the `dualmap` chains per model; every `dual` state | chains: lowest and mean F_GHZ through the site, peak bonds; states: each model's F on the state's qubits, IBM raw and mitigated |

The telemetry panel lists, per data set: records, runs, PyQrack / Qrack versions, shots × realisations, the Layer 2 parameters, the estimator, mean fidelity per placement class, and QrackBridge's passes (peak bonds against `max_bonds`, truncating circuits, fidelity estimate, widest chain). For Shor it lists each placement class with the majority prediction; for the dual, the cables (dCZ, transfer, gate time) and each model's F ± se with peak bonds; for runs, the configuration, total and per-point seconds, device, shots and ACE geometry. Clicking a site lists every measure there and the units holding it, and draws those units' edges: chain CXs, or Shor's encoder CXs.

| Control | Key | Effect |
|---|---|---|
| Shape ◀ ▶ | ↑ ↓ | the scene |
| Data set ◀ ▶ | ← → | the data set (the measure is kept where it exists) |
| Measure ◀ ▶ | m / M | the measure; colour bar and statistics follow |
| columns (else dots) | c | columns along each site's outward direction, or dots at the same height |
| value labels | v | numbers on the columns (by default when ≤ 64 sites or ≤ 48 cells) |
| group names | n | chips, faces, patches, modules |
| couplers, links and cables | | couplers by class (exact, replica, cross, L); links between chips; the dual's cables |
| coupler telemetry | k | couplers coloured and thickened by their break rate |
| pull apart | e | chips sideways, box faces along their normals, modules away from the cables (0–4) |
| height | | column scale (0.2–3) |
| reset view, top, side | 0, 1, 2 | camera; drag rotates, scroll zooms |
| light / dark | t | the theme, at any time |
| reload | r | re-read every file, keeping shape, data set and measure |
| save PNG | p | `./nighthawk_3d_png` (or `--save DIR`) |
| click | Esc clears | pick a site or a landscape cell |

| Option | Default | Effect |
|---|---|---|
| `--out`, `--shor`, `--ghz FILE` | every `*.jsonl` here | record files, as for `nighthawk_graph.py` |
| `--repo DIR` | auto | the BlueQubit release: the lattice of the run records |
| `--shots-dir DIR`, `--cfg TAG` | | as for `nighthawk_graph.py` |
| `--theme dark\|light\|auto` | `dark` | colours; `auto` reads the desktop (GNOME / GTK, KDE, macOS, Windows), else dark |
| `--scene`, `--dataset`, `--measure` | first | the first one shown: a number or part of its name |
| `--list` | | print every shape, data set and measure found, and exit |
| `--save DIR` | | PNGs instead of the window: the first data set of every shape, or the chosen ones |
| `--all-datasets`, `--all-measures` | | with `--save`: every data set, every measure |
| `--view ELEV AZIM`, `--pull X`, `--dots` | per shape | camera, pull-apart, dots |
| `--size W H`, `--dpi N` | 15.5 9, 130 | window / PNG size |
| `--follow SEC` | | re-read the records every SEC seconds (a sweep still running) |

Colours are a single-hue blue ramp for magnitudes, dark to light on the dark surface and light to dark on the light one; differences (engine − exact) use red ↔ grey ↔ blue. Chips, faces and modules take the categorical slots in a fixed order.

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

Every measured number is in [RESULTS.md](RESULTS.md), grouped by experiment (E1–E36 of [METHODOLOGY.md](METHODOLOGY.md) §9), with the points not yet measured in its §15.

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
python3 nighthawk_qrack.py hamming --csv dan_hamming_ladder.csv

# Shor's 9-qubit code on a 3 x 3 two-patch ACE register (RESULTS §10.3): 10 fresh backends per state
python3 nighthawk_qrack.py shor9 --shots 8192 --repeats 10 --jobs 0 --out shor9.jsonl      # every core
for m in 4 5 6; do python3 nighthawk_qrack.py shor9 --size $m --shots 4096 --repeats 5 --out shor9.jsonl; done
# the same on PoCL (OpenCL on the CPU; Qrack >= 10.13.2): a directory holding pocl.icd
OCL_ICD_VENDORS=/path/to/pocl-icd-dir python3 nighthawk_qrack.py shor9 --gpu --ocl-sync --size 4 --shots 4096 --repeats 5 --out shor9_pocl.jsonl
for r in nighthawk2p45 nighthawk61r2 nighthawk88e; do
  python3 nighthawk_qrack.py shor9 --register $r --shots 2048 --repeats 5 --out shor9_tiles.jsonl
done
NIGHTHAWK_REGISTERS=nighthawk_registers_favourable.json \
  python3 nighthawk_qrack.py shor9 --register fav5p112 --shots 2048 --repeats 5 --out shor9_tiles.jsonl

# VQE on the two-patch ladder: transverse-field Ising at h = 3.04, angles from nighthawk2p16 (cached in
# vqe_ladder_vqe.json on the first point), numpy references up to 26 qubits, Anderson's bound at every size
for r in nighthawk2p16 nighthawk2p20 nighthawk2p24 nighthawk2p28 nighthawk2p32 nighthawk2p36 nighthawk2p40 \
         nighthawk2p45 nighthawk2p49 nighthawk2p53 nighthawk2p57; do
  python3 nighthawk_qrack.py run --backend ace --register $r --families vqe --depths 1 2 3 4 6 \
    --order shallow --cpu --gpus 0 --per-gpu 1 --mem-budget-gb 290 --out vqe_ladder.jsonl
done
python3 nighthawk_qrack.py run --backend ace --register nighthawk61r2 --families vqe --depths 1 2 3 4 \
  --order shallow --cpu --gpus 0 --per-gpu 1 --swap-gb 238 --mem-budget-gb 530 --out vqe_ladder.jsonl

# a register search (§3.4)
python3 nighthawk_qrack.py layoutsearch --n 88 --geometries 114:4:5 102:4:5 --region convex \
  --hint nighthawk88e --candidates 3 --cpsat-time 600 --test-depths 4 8 --test-jobs 3 --cpu --save-best nighthawk88x

# GHZ (§3.6): Layer 1a tiles over 61 qubits; Layer 2 per mode (the paper's K = 32 realisations); chip-wide; the map
python3 nighthawk_ghz.py selftest
python3 nighthawk_ghz.py tiles --register nighthawk61r2 --unit 3 --shots 4096 --repeats 5 --jobs 16 --out ghz.jsonl
for m in unmitigated compensated dd; do
  python3 nighthawk_ghz.py tiles --register nighthawk61r2 --unit 3 --layer2 $m --blocks 3 --shots 512 --repeats 32 \
    --jobs 16 --out ghz_l2.jsonl
done
python3 nighthawk_ghz.py chip  --register nighthawk2p57 --shots 4096 --repeats 5 --jobs 16 --out ghz.jsonl
python3 nighthawk_ghz.py chip  --register nighthawk2p16 --layer2 dd --mqc-max 16 --repeats 32 --out ghz_l2.jsonl
python3 nighthawk_ghz.py plot  --in ghz_l2.jsonl --register nighthawk61r2 --layer2 dd --png ghz61_dd.png
# the seam layers (§3.6): passes 1 and 2, Layer S (syndrome + repetition code), Layer V (the corrected measurement)
python3 nighthawk_ghz.py stack --register nighthawk61r2 --unit 3 --shots 256 --repeats 4 --jobs 16 --out stack61.jsonl
python3 nighthawk_ghz.py stack --register nighthawk2p16 --mode chip --mqc-max 16 --repeats 8 --jobs 16 --out stack16.jsonl
# the same with ZNE beside the code (lambda 1, 5, 9)
python3 nighthawk_ghz.py stack --register nighthawk2p28 --unit 3 --zne-k 1,2 --out stack28.jsonl
# the code before the echo (E26), the inside ZZ by the GHZ stabiliser (E28), each run as three heatmaps
python3 nighthawk_ghz.py stack --register nighthawk61r2 --unit 3 --shots 256 --repeats 4 --order code-echo --jobs 16 \
  --out stack61_ce.jsonl
python3 nighthawk_ghz.py stack --register nighthawk61r2 --unit 3 --shots 256 --repeats 4 --inside-zz stabilizer \
  --jobs 16 --out stack61_zz.jsonl
python3 nighthawk_ghz.py stackmap --in stack61_ce.jsonl --png stack61_ce.png
# the loom as the seam: nighthawk61r2's 61 qubits as two exact chips of its patches' size; a hard cut (E31), and
# the ghost band (E29)
python3 nighthawk_ghz.py stack --engine loom --order code-echo --shots 256 --repeats 4 --jobs 16 --out cut61.jsonl
python3 nighthawk_ghz.py stack --engine loom --loom-read ghost --order code-echo --shots 256 --repeats 4 --jobs 16 \
  --out loom61.jsonl
python3 nighthawk_ghz.py stack --engine loom --order code-echo --shots 256 --repeats 4 --inside-zz stabilizer \
  --jobs 16 --out loom61_zz.jsonl
python3 nighthawk_ghz.py stackmap --in loom61.jsonl --png loom61.png
# the same two chips joined by QrackBridge instead of the loom (E32), its capacity from 6 bond qubits down to 0
for mb in 6 2 1 0; do
  python3 nighthawk_ghz.py stack --engine bridge --bridge-max-bonds $mb --order code-echo --shots 256 --repeats 4 \
    --jobs 16 --out bridge61_mb$mb.jsonl
done
python3 nighthawk_ghz.py stackmap --in bridge61_mb1.jsonl --png bridge61_mb1.png
# a closed 5 x 5 x 5 box (E33): six faces on a QrackBridge ring, chains across all twelve box edges; at max_bonds
# 1 and 0 the drift's ZZ as bridged gates can zero the ring's state (RESULTS 10.16), so those run with it dropped
for mb in 6 2; do
  python3 nighthawk_ghz.py stack --engine bridge --box 5 5 --bridge-max-bonds $mb --order code-echo --shots 256 \
    --repeats 4 --jobs 16 --out box_mb${mb}_gate.jsonl
done
for mb in 6 2 1 0; do
  python3 nighthawk_ghz.py stack --engine bridge --box 5 5 --bridge-max-bonds $mb --inside-zz stabilizer \
    --order code-echo --shots 256 --repeats 4 --jobs 16 --out box_mb${mb}_stabilizer.jsonl
done
python3 nighthawk_ghz.py stackmap --in box_mb2_gate.jsonl --png box_mb2_gate.png
python3 bridge_ring_repro.py      # the ring's wrap cut against an open chain
# Shor's code across QrackBridge (E34): Shor-9 on Nighthawk's sites as two chips and on the box; Shor-16, -25 across
for mb in 6 3 2 1 0; do
  python3 nighthawk_ghz.py shor --loom-chips rect:0-11,0-5 rect:0-11,6-9 --bridge-max-bonds $mb --jobs 16 \
    --out shor9_big_mb$mb.jsonl
  python3 nighthawk_ghz.py shor --box 5 5 --bridge-max-bonds $mb --jobs 16 --out shor9_box5_mb$mb.jsonl
done
for mb in 6 4 3 2 1 0; do
  python3 nighthawk_ghz.py shor --loom-chips rect:0-11,0-5 rect:0-11,6-9 --size 4 --shor-units cross \
    --bridge-max-bonds $mb --jobs 16 --out shor16_big_mb$mb.jsonl
done
for mb in 3 2 1; do
  python3 nighthawk_ghz.py shor --loom-chips rect:0-11,0-5 rect:0-11,6-9 --size 5 --shor-units cross \
    --bridge-max-bonds $mb --shots 256 --exact-shots 128 --jobs 16 --out shor25_big_mb$mb.jsonl
done
python3 nighthawk_ghz.py shormap --in shor9_box5_mb1.jsonl --png shor9_box5_mb1.png
# IBM's two-module Nighthawk R1 (arXiv:2610.29986) on the bridge (E35), and its sensitivity to the local CZ error
python3 nighthawk_ghz.py dual --models hardcut,ideal,exact,budget,measured --trajectories 20000 --batch 500 \
  --jobs 16 --out dual_r1.jsonl --png dual_r1.png
python3 nighthawk_ghz.py dualmap --trajectories 2000 --trajectories-link 4000 --jobs 16 --out dualmap.jsonl \
  --png dualmap.png
for e in 0 0.0071; do
  python3 nighthawk_ghz.py dual --models measured --local-cz-error $e --trajectories 10000 --batch 500 --jobs 16 \
    --out dual_r1_cz$e.jsonl
done
# Layer 3, Shor order finding, across the configurations (E36)
python3 nighthawk_ghz.py shoralg --N 15 --counting 4 8 --repeats 4 --bonds 0 1 2 3 4 5 --jobs 16 --out shoralg_chips.jsonl
python3 nighthawk_ghz.py shoralg --N 21 --repeats 4 --bonds 0 1 2 3 4 5 --jobs 16 --out shoralg_chips.jsonl
python3 nighthawk_ghz.py shoralg --box 5 5 --N 15 --counting 4 8 --layouts split,direct --repeats 4 --bonds 0 1 2 3 4 5 \
  --jobs 16 --out shoralg_box.jsonl
python3 nighthawk_ghz.py shoralg --box 5 5 --N 21 --a 2 4 --layouts split,direct --repeats 4 --bonds 0 1 2 3 4 5 \
  --jobs 16 --out shoralg_box.jsonl
python3 nighthawk_ghz.py shoralg --dual --N 15 --counting 4 8 --layouts inside,split,direct \
  --models hardcut,ideal,exact,budget,measured --trajectories 400 --jobs 16 --out shoralg_dual.jsonl
python3 nighthawk_ghz.py shoralg --dual --N 21 --a 2 --layouts split --models exact,measured --trajectories 24 \
  --jobs 16 --out shoralg_dual.jsonl
python3 nighthawk_graph.py --ghz shoralg_chips.jsonl --ghz shoralg_box.jsonl --ghz shoralg_dual.jsonl --figs 26-28 --save plots
# two-patch ACE chips joined by the loom (E30): two nighthawk61r2 cloverleaves, 122 qubits
python3 nighthawk_ghz.py stack --engine loom --loom-chip-backend ace --loom-chips nighthawk61r2 nighthawk61r2 \
  --order code-echo --shots 256 --repeats 4 --jobs 16 --out loomace61x2.jsonl
python3 nighthawk_ghz.py stackmap --in loomace61x2.jsonl --png loomace61x2.png
python3 nighthawk_ghz.py stackmap --in stack61_zz.jsonl --png stack61_zz.png
# the Walsh echo window by window (E27), the cut units under Layer 2 (E28)
python3 nighthawk_ghz.py sweep --register nighthawk2p28 --shots 256 --repeats 4 --no-exact --code-at 8 --jobs 16 \
  --out sweep28.jsonl --summary sweep28.json
python3 nighthawk_ghz.py probe --register nighthawk61r2 --shots 256 --repeats 4 --jobs 16 --out probe61.jsonl

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
- **GHZ on stabilizer patches.** `nighthawk_ghz.py chip` on `nighthawk61r2` does not sample on stabilizer-hybrid patches with PyQrack 2.29.5 / Qrack 10.13.2 (§3.6); its `tiles` run dense and are unaffected.
- **GHZ across simulators.** ACE joins its simulators through replicas and classical shadows, so a GHZ state no single simulator holds (a chip-wide state on two or more patches, a chain through a seam qubit into both patches) keeps no coherence and F_GHZ ≤ ½: `stack`'s Layer S corrects the replica class only (RESULTS §10.7).
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
