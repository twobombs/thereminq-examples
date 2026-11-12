# Nighthawk, consolidated: `nighthawk_qrack.py`

One script for every Nighthawk simulation in `devices/nighthawk`, plus IBM Quantum Nighthawk r3 as a field of six 120-qubit chiplets.

| | |
|---|---|
| Script | `nighthawk_qrack.py`, 18,683 lines |
| Viewers | `nighthawk_graph.py` (the `dual-nighthawk` version plus view 29, `--poster` for `field run` records and `--ghz-poster` for GHZ per chiplet, sections 3.4 and 3.5), `nighthawk_3dgraph.py` (unchanged) |
| Stubs | `nighthawk_ghz.py`, `nighthawk_cloverfield.py`, `nighthawk_erepr.py`: forward to the script |
| Data | [`BlueQubitDev/rcs-nighthawk`](https://github.com/BlueQubitDev/rcs-nighthawk) next to the script, `--repo`, or `$NIGHTHAWK_REPO` |
| Simulator | PyQrack: `QrackSimulator`, `QrackAceBackend`, `QrackBridge` |
| Acceleration | OpenCL or CPU. No CUDA. |
| Detail | `METHODOLOGY.md`, `RESULTS.md` and the long README of `dual-nighthawk` and `erepr-nighthawk` stay valid; only the command prefix changes (section 2) |

## 1. What went into it

| Source folder | Source file | Lines | Where it is now |
|---|---|---|---|
| `dual-nighthawk` | `nighthawk_qrack.py` | 14,085 | the base, unchanged apart from the hooks in section 5 |
| `rcs-nighthawk` | `nighthawk_qrack.py` | 5,476 | covered by the base: all 240 top-level names present |
| `vqe-nighthawk` | `nighthawk_qrack.py` | 5,829 | covered by the base: all 251 present |
| `shrs-nighthawk` | `nighthawk_qrack.py` | 6,741 | covered by the base: all 295 present |
| `ghz-nighthawk`, `shrs-nighthawk` | `nighthawk_ghz.py` | 4,505 | covered by the base (`ghz` command): all 138 present |
| `rcs-nighthawk` | `nighthawk_cloverfield.py` | 2,179 | merged in as the `field` command |
| `erepr-nighthawk` | `nighthawk_erepr.py` | 2,068 | merged in as the `erepr` command |
| new | | | device presets `r2`, `r1x2`, `r3`, `r3x2`; registers `nighthawk120`, `nighthawk120w`; `field --no-ace-error-detection` |

Where an older folder's function differs from the base, the base's newer version is the one kept (16 functions against `rcs`, 11 against `vqe`, 3 against `shrs`). `bro-nighthawk` is a different toolkit (DCS XEB, `tnsweep.py`) and is not part of this file.

## 2. Commands

| Old command line | New command line |
|---|---|
| `nighthawk_qrack.py <cmd> ...` | the same |
| `nighthawk_ghz.py <cmd> ...` | `nighthawk_qrack.py ghz <cmd> ...` |
| `nighthawk_cloverfield.py <cmd> ...` | `nighthawk_qrack.py field <cmd> ...` |
| `nighthawk_erepr.py <cmd> ...` | `nighthawk_qrack.py erepr <cmd> ...` |

| Group | Commands |
|---|---|
| main | `verify` `selftest` `hwxeb` `run` `aceplan` `layoutsearch` `danmodel` `hamming` `shor9` `bridgebench` `seamgap` |
| `ghz` | `chip` `tiles` `stack` `sweep` `probe` `stackmap` `shor` `shormap` `shoralg` `dual` `dualmap` `dualplot` `plot` `selftest` |
| `field` | `plan` `run` `linksearch` `layoutsearch` `selftest` |
| `erepr` | `derive` `plan` `run` `ds` `pairs` `table` `aceprobe` `selftest` |

`-h` works per group (`ghz -h`, `field -h`, `erepr -h`) and per command.

## 3. Nighthawk r3

### 3.1 Device presets (`field --device NAME`)

| Preset | Chips | Grid | l-couplers per adjacent pair | Qubits | Stands for |
|---|---|---|---|---|---|
| `r2` | `nighthawk120` | 1 x 1 | 0 | 120 | one Nighthawk chip |
| `r1x2` | `nighthawk120*2` | 1 x 2 | 4 | 240 | IBM's two-module r1 demonstration |
| `r3` | `nighthawk120*6` | 2 x 3 | 10 | 720 | one r3 processor as six chiplets |
| `r3x2` | `nighthawk120*12` | 2 x 6 | 10 | 1,440 | two r3 processors side by side |

`--chips`, `--arrangement` and `--links` still work and override the preset. Without `--device` the defaults are the cloverfield's own (`nighthawk88e*2`, 4 links).

```bash
python3 nighthawk_qrack.py field plan --device r3 --png r3.png
python3 nighthawk_qrack.py field run  --device r3 --family hamming --protocol nnqab --fold 1 3 \
        --depths 4 6 8 --ace-gpu --gpus 0-5 --out r3.jsonl
python3 nighthawk_qrack.py field run  --device r3 --cut-links --family hamming --protocol nnqab --fold 1 3 \
        --depths 4 6 8 --ace-gpu --gpus 0-5 --out r3.jsonl          # matched baseline: same anchors, no links
python3 nighthawk_qrack.py field run  --device r3 --chips nighthawk120w*6 --loom mean --ace-gpu --gpus 0-5 --out r3w.jsonl
```

One chip worker per chiplet: `--gpus 0-5` puts each chiplet on its own die.

### 3.2 IBM's figures against this model

| Item | IBM (QDC 2026) | This model |
|---|---|---|
| r2 qubits | 120 | 120 (`nighthawk120`) |
| r2 two-qubit gates | over 7,500 | no limit; `plan` prints the cycle count that budget buys |
| r3 qubits per processor | 520 | 720 (6 x 120) |
| r3 lattice | square, as r2 | 12 x 10 square per chiplet |
| r3 l-couplers | 10 per processor, chip to chip | 10 per adjacent chiplet pair, 7 pairs, up to 70 |
| r3 two-qubit gates | 10,000 | as r2 |
| Linked system | two processors, 1,040 qubits, 2027 | `r3x2`, 1,440 qubits |
| On-chip reset | kept from r2 | not modelled (no mid-circuit reset in these circuits) |
| Broken qubits | not published for r3 | none on a chiplet (17, 55, 62 are `ibm_phoenix`'s) |

The six-chiplet reading and the chiplet lattice are the model's, not IBM's. For IBM's own numbers use `--chips` with a rectangle or a saved register: two 520-qubit processors need a lattice larger than the release's 12 x 10, which `rect:` cannot express yet.

### 3.3 The chiplet registers

| Register | ACE sites, lrc, lrr | Patches | Widths incl. ancilla | Boundary qubits | Couplers inside / through a boundary / between patches | Dense fp32 | Ghost room on 8 GB |
|---|---|---|---|---|---|---|---|
| `nighthawk120` | 184, 2, 7, flat | 8 | 20 / 23 / 22 / 22 / 21 / 23 / 16 / 20 | 39 | 92 / 126 / 0 | 0.3 GiB | caps 27 x 2: yes |
| `nighthawk120w` | 170, 2, 9, flat | 6 | 26 / 27 / 26 / 27 / 25 / 26 | 31 | 118 / 100 / 0 | 4.75 GiB | none |

Both were found by annealing the seams on the lattice for an ACE ring of patches and assigning sites from `QrackAceBackend._unpack` (PyQrack 2.30.6). They were not searched with `layoutsearch`: a CP-SAT search (`layoutsearch --n 120 --region fixed`) may find a shorter boundary, and its result can be saved over these names.

Check of `nighthawk120` on PyQrack 2.30.6 from pip, CPU, outside the script: mirror circuits (random single-qubit gates, CZ colours A B C D, then the inverse), Hamming signal 1 - 2h, three circuits per point.

| Cycles in U | Error detection | `nighthawk120` | Row-major on ACE's own 12 x 10 grid (120 sites, lrc 3, lrr 4) |
|---|---|---|---|
| 4 | on | 0.576 | 0.136 |
| 8 | on | NaN in 2 of 3 circuits, 0.194 in the third | -0.013 |
| 4 | off | 0.793 | 0.769 |
| 8 | off | 0.305 | 0.176 |
| 12 | off | 0.072 | 0.034 |

The NaN comes from `QrackAceBackend` with `is_error_detection=True` (first seen after a seam-to-bulk CZ in cycle 5; the same with `use_crossbars=False`; gone with error detection off). Until that is settled upstream, run chiplets deeper than about 4 cycles with `field ... --no-ace-error-detection`. Your own Qrack build may behave differently from the pip wheel.

### 3.4 r3 measured: mean loom on the CPU

`field run --device r3 --loom mean [--cut-links] --family hamming --protocol nnqab --fold 1 --cpu --no-ace-error-detection`, on 2 CPU cores with about 8 GB RAM (one loom peaked at 1.8 GB; the ghost loom does not fit). Signal is 1 - 2h; means over instances.

| Depth | Instances | Woven | Cut links | Bulk, woven / cut | Seam, woven / cut | Anchor, woven / cut | s per point |
|---|---|---|---|---|---|---|---|
| 4 | 3 | 0.730 | 0.843 | 0.81 / 0.84 | 0.75 / 0.77 | 0.49 / 0.98 | 48 / 33 |
| 6 | 3 | 0.419 | 0.631 | 0.51 / 0.62 | 0.42 / 0.49 | 0.16 / 0.90 | 191 / 171 |
| 8 | 2 | 0.219 | 0.422 | 0.26 / 0.38 | 0.25 / 0.31 | 0.04 / 0.72 | 310 / 295 |

| Field | Count |
|---|---|
| Qubits: bulk / seam / link anchor | 370 / 218 / 132 |
| l-couplers placed (mean loom; the ghost plan places 64) | 70 |
| Couplers: intra-chip / link CZ | 1308 / 70 |

The poster of these records, and of any other `field run` output:

```
python3 nighthawk_graph.py --out r3_mean.jsonl --poster r3_poster.png --poster-note "Host: ..."
python3 nighthawk_graph.py --out r3_mean.jsonl --save plots --figs 29      # view 29 alone
```

| `--poster` option | Default | Effect |
|---|---|---|
| `--poster FILE` | | write the one-page poster (PNG, PDF or SVG by extension), then exit |
| `--poster-note TEXT` | | a line for the "How this was measured" box (repeatable) |
| `--poster-dpi N` | 150 | resolution |
| `--vram-gb G` | 8 | one die's memory, the line on the ghost-loom memory panel |
| `--cfg TAG` | largest pair | which woven / cut-links pair to draw |

### 3.5 GHZ on every r3 chiplet

`ghz tiles --register nighthawk120` (3-qubit GHZ chains along rows and columns, 196 units, MQC readout, 1024 shots per setting) once per chiplet, each chiplet alone on its QrackAceBackend preset (9 patches: 57 and 8 x 31 qubits, 39 seam qubits). Layer 2 is the Walsh echo (`--layer2 dd`, 3 windows, T 4.4, J 0.065, sigma 0.015) with the chiplet's own drift draw (`--seed` = chip index); "no drift" is one `--layer2 none` run, the same circuit on every chiplet. Anchors are r3's link anchors on the chiplet (70 l-couplers). The exact reference (QrackSimulator, 120 qubits) is 1.00 on every unit.

| Chiplet | Anchors (on seam) | F all | F seam units | F off-seam units | Seam units at or below 1/2 | F anchor units / rest | s |
|---|---|---|---|---|---|---|---|
| no drift | | | 0.600 | 1.000 | 39 % | | 509 |
| 0 | 19 (2) | 0.619 | 0.502 | 1.001 | 55 % | 0.809 / 0.576 | 832 |
| 1 | 28 (3) | 0.607 | 0.486 | 1.001 | 55 % | 0.773 / 0.543 | 826 |
| 2 | 19 (3) | 0.598 | 0.475 | 1.000 | 54 % | 0.759 / 0.562 | 791 |
| 3 | 19 (2) | 0.600 | 0.477 | 1.001 | 53 % | 0.761 / 0.564 | 814 |
| 4 | 28 (3) | 0.597 | 0.473 | 1.000 | 55 % | 0.785 / 0.526 | 802 |
| 5 | 19 (3) | 0.610 | 0.491 | 0.999 | 53 % | 0.750 / 0.579 | 824 |

| Finding | |
|---|---|
| Off-seam units | 1.000 with and without drift on all six chiplets: the Walsh echo refocuses the drift exactly |
| Seam units | 0.60 without drift, 0.47 to 0.50 with drift and echo; lowest unit 0.007 |
| Anchor units | higher than the rest because few anchors sit on ACE seams (2 or 3 per chiplet), not because of the links: no loom runs here |
| Chip-wide GHZ | `ghz chip --register nighthawk120` fails: stabilizer patches raise in ACE's sampler (as on `nighthawk61r2`, PyQrack 2.30.6 / Qrack 10.14.2 too), dense patches of 57 and 31 qubits do not fit |

```
for c in 0 1 2 3 4 5; do
  python3 nighthawk_qrack.py ghz tiles --register nighthawk120 --layer2 dd --seed $c --jobs 2 --out chip$c.jsonl > chip$c.log
done
python3 nighthawk_qrack.py ghz tiles --register nighthawk120 --jobs 2 --out layer1a.jsonl
python3 nighthawk_graph.py --ghz-poster r3_ghz.png --ghz-chip chip0.jsonl ... --ghz-chip chip5.jsonl \
        --ghz-base layer1a.jsonl --field-device r3
```

### 3.6 Chip topologies: `--topology`, `--hub`, `--pairs`, `--long-links`

| Option | Chip pairs that get l-couplers |
|---|---|
| `--topology grid` (default) | every pair of grid neighbours; fields and config tags as before |
| `--topology ring` | the chips in a cycle around the grid centre, each to the next |
| `--topology hub [--hub C]` | one router chip (default the one nearest the centre) to every other chip, nothing else |
| `--topology none` | only `--pairs` |
| `--pairs 0-5,2-3` | added to any of these: long-range shortcuts |
| `--long-links K` | l-couplers per pair that are not grid neighbours (default `--links`) |

Grid neighbours are placed first, as before (facing borders, `--links` per pair); long-range pairs then take the sites nearest the other chip that still have a coupler colour free at both ends. `field plan` prints the pairs, the links per pair and the chip-to-chip hops; records of a non-grid field carry `topology`, `chip_pairs`, `hub` and `long_links`.

r3, mean loom on the CPU, Hamming nnqab fold 1, signal 1 - 2h (grid: 3 instances; the others 2):

| Topology | Links (long) | Anchors | Hops mean / max | d4 woven / cut | d6 woven / cut | Cost per link, d4 | Lowest chip, d6 |
|---|---|---|---|---|---|---|---|
| grid | 70 (0) | 132 | 1.67 / 3 | 0.730 / 0.843 | 0.419 / 0.631 | 0.0016 | chip 4, 0.325 |
| ring | 60 (0) | 116 | 1.80 / 3 | 0.748 / 0.841 | 0.422 / 0.602 | 0.0016 | chip 4, 0.332 |
| hub 1, `--links 8` | 40 (16) | 78 | 1.67 / 2 | 0.761 / 0.820 | 0.452 / 0.580 | 0.0015 | chip 1 (hub), 0.295 |
| grid + 0-5, 2-3 | 90 (20) | 172 | 1.40 / 2 | 0.677 / 0.843 | 0.364 / 0.616 | 0.0018 | chip 4, 0.281 |

Ghost-loom plan of the same topologies (`field plan --loom ghost`, per-chip caps of one 8 GB die):

| Topology | Links placed | Ghosts per chip | Peak per chip (GiB) | Pairs left unconnected |
|---|---|---|---|---|
| grid | 64 | 18 to 27 | 3.0 to 4.4 | 0 |
| ring | 57 | 18 to 20 | 2.4 to 3.5 | 0 |
| hub 1, `--links 8` | 36 (1-5 gets 4 of 8) | hub 36, others 4 to 8 | hub 5.3, others 0.3 to 2.3 | 0 |
| hub 1, `--links 10` | 35 (1-5 gets 0) | hub 35 | hub 5.3 | 5 |
| grid + 0-5, 2-3 | 84 | 27 to 29 | 3.9 to 4.4 | 0 |

| Finding | |
|---|---|
| Mean loom | the woven signal falls with the number of link anchors, about 0.0016 per link at d4 whatever the topology; hops do not enter, since no entanglement crosses a link |
| Hub | the router chip is the hot spot: lowest chip signal (0.295 at d6), and in the ghost plan 36 ghosts and 5.3 GiB against 0.3 to 2.3 GiB on the others |
| Hub ports | only border sites of a 12 x 10 chiplet have a coupler colour free; with five pairs of 10 the placement finds 42 links on the hub, so a hub needs `--links 8` or fewer |
| Long-range sites | the facing borders are taken by the grid links, so shortcuts land on the outer borders (chip 0's top row and left column for 0-5) |
| Next | the hop penalty itself needs the ghost loom (six 8 GB dies) and a cross-chip observable (GHZ or Bell pairs at 1, 2, 3 hops) |

```
python3 nighthawk_qrack.py field run --device r3 --topology hub --links 8 --loom mean [--cut-links] \
        --family hamming --protocol nnqab --fold 1 --depths 4 6 --instances 2 --cpu --no-ace-error-detection --out r3_topo.jsonl
python3 nighthawk_graph.py --out r3_mean.jsonl --out r3_topo.jsonl --poster r3_hub1.png --poster-topology "hub 1"
```

The field poster names the topology in its title, draws long-range l-couplers as dashed arcs, shows the chip-to-chip hops as a table and a tile, adds the hops of the ghost plan when it leaves pairs unconnected, and draws the other topologies of the same field as grey lines in the signal panel. `--poster-topology LABEL` (grid, ring, hub N, grid + I-J, ...) picks which one.

## 4. Renamed names (for code that imported the old modules)

| Old module | Old name | Name in `nighthawk_qrack` |
|---|---|---|
| `nighthawk_cloverfield` | `cfg_tag` | `field_cfg_tag` |
| | `cmd_run`, `cmd_plan`, `cmd_selftest` | `cmd_field_run`, `cmd_field_plan`, `cmd_field_selftest` |
| | `summarize` | `field_summarize` |
| | `main` | `field_main` |
| `nighthawk_erepr` | `I2` | `ER_I2` |
| | `circuit_couplers`, `run_loom` | `er_circuit_couplers`, `er_run_loom` |
| | `build_field`, `start_chips` | `er_build_field`, `er_start_chips` |
| | `cmd_run`, `cmd_plan`, `cmd_selftest` | `cmd_erepr_run`, `cmd_erepr_plan`, `cmd_erepr_selftest` |
| | `main` | `erepr_main` |
| `nighthawk_ghz` | `main`, `make_parser`, `cmd_selftest` | `ghz_main`, `ghz_make_parser`, `cmd_ghz_selftest` (as before) |

Every other name is unchanged. `import nighthawk_cloverfield` and `import nighthawk_erepr` return the `nighthawk_qrack` module.

## 5. Changes to merged code

| Place | Change |
|---|---|
| `main()` | dispatches `field`, `erepr`, `chipworker` before the main parser |
| `cloverfield()` | returns this module instead of loading `nighthawk_cloverfield.py`; `--cloverfield` and `$NIGHTHAWK_CLOVERFIELD` are accepted and ignored |
| chip workers | started as `nighthawk_qrack.py chipworker` |
| `erepr` tile cache | `$NIGHTHAWK_TILE_CACHE` default set when `erepr` runs, not at import |
| `field` options | `--device`; `--no-ace-error-detection` (enters the config tag only when given); `--chips` and `--links` defaults applied after parsing |
| `field plan` | one extra line with `--device`: CZs per cycle against the device's gate budget |
| Config tags | unchanged for every existing run, so old JSONL files resume |

## 6. Verification status

Run on CPU (OpenCL hidden), Python 3.13, PyQrack 2.30.6 from pip, Qiskit 2.6.2, OR-Tools 9.15, numpy 2.5.

| Check | Result |
|---|---|
| Parses, compiles | yes |
| Top-level names of the three merged files present | 530 / 530, 50 / 50, 118 / 118 |
| Names bound twice at top level | none |
| Undefined names (pyflakes) | none; 18 messages, the same 18 as the three sources together |
| `nq.` / `cf.` references rewritten | 84 in `field`, 33 in `erepr`; none shadowed by a local |
| Placements `nighthawk120`, `nighthawk120w` | 120 distinct sites each, no coupler between patches without a seam |
| `selftest` | 14 / 14, 70 s |
| `ghz selftest` | 20 / 20, 9 s |
| `field selftest` | 12 / 12, 30 s ([12]: `--topology`, `--pairs`, hops) |
| `erepr selftest` | all, 7 s |
| `field plan --device r3` | 720 qubits, 64 l-couplers, every chiplet fits one 8 GB die (below) |

### 6.1 PyQrack 2.30 and `QrackBridge`

From PyQrack 2.30 `QrackBridge` wraps Qrack's native bridge (`init_qbridge`) instead of the numpy class `HBridge` mirrors. States, truncations and fidelity estimates still match `HBridge` exactly; two things changed.

| Change | Effect before the fix | Fix |
|---|---|---|
| Sampler draws from Qrack's C++ RNG | `selftest` [14] and `bridgebench --check` failed: shots never bit-identical | with the native bridge, both shot sets are compared with the contracted state: per-qubit P(1), largest \|z\| below 5 (measured 2.8 over 4,000 shots); identical shots are still required against the Python bridge |
| Seed must be one 64-bit integer | `ghz selftest` [18] and every `ghz shor`, `ghz stack --engine bridge`, `ghz shoralg` run stopped with `TypeError` | `bridge_seed()` hashes the seed lists to one integer (the same list always gives the same seed) |

Shots from bridge runs under PyQrack 2.30 are therefore a different draw from those of earlier versions, with the same distribution. `--bridge-truncation born` also draws from Qrack's RNG now, so its states are no longer reproducible against `HBridge` either.

### 6.2 `field plan --device r3`

| Chiplet pair | Grid direction | l-couplers placed of 10 |
|---|---|---|
| 0-1, 1-2, 3-4, 4-5 | horizontal (colour B) | 10 each |
| 0-3 | vertical (colour D) | 8 |
| 1-4 | vertical (colour D) | 7 |
| 2-5 | vertical (colour D) | 9 |

| Chiplet | Ghosts | Patch widths incl. ancilla | Peak fp32 |
|---|---|---|---|
| 0 | 18 | 20 / 26 / 26 / 27 / 25 / 25 / 16 / 20 | 3.5 GiB |
| 1 | 27 | 20 / 26 / 26 / 27 / 25 / 27 / 19 / 24 | 4.4 GiB |
| 2 | 19 | 20 / 23 / 22 / 26 / 25 / 27 / 19 / 24 | 3.0 GiB |
| 3 | 18 | 25 / 27 / 27 / 25 / 21 / 23 / 16 / 21 | 3.6 GiB |
| 4 | 27 | 24 / 27 / 27 / 25 / 21 / 26 / 19 / 25 | 4.1 GiB |
| 5 | 19 | 24 / 27 / 22 / 22 / 21 / 26 / 19 / 25 | 3.0 GiB |

64 of 70 possible l-couplers (8 facing pairs skipped: no CZ colour free at both ends, or no ghost room under the caps 27 x 2). About 343 CZs per cycle, so IBM's 10,000-gate budget is about 29 cycles of this field.

## 7. First run

```bash
git clone https://github.com/BlueQubitDev/rcs-nighthawk
python3 nighthawk_qrack.py selftest
python3 nighthawk_qrack.py ghz selftest
python3 nighthawk_qrack.py field selftest
python3 nighthawk_qrack.py erepr selftest
python3 nighthawk_qrack.py field plan --device r3
```

`field plan --device r3` lists every l-coupler and how many facing pairs were skipped for lack of ghost room; with `--loom mean` none are skipped.
