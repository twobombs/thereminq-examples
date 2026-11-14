# Nighthawk, consolidated: `nighthawk_qrack.py`

One script for every Nighthawk simulation in `devices/nighthawk`, plus IBM Quantum Nighthawk r3 as a field of six 120-qubit chiplets.

| | |
|---|---|
| Script | `nighthawk_qrack.py`, 20,150 lines |
| Viewers | `nighthawk_graph.py` (the `dual-nighthawk` version plus view 29, `--poster` for `field run` records and `--ghz-poster` for GHZ per chiplet and `--lab-poster` for the configuration lab, sections 3.4, 3.5 and 3.7), `nighthawk_3dgraph.py` (unchanged) |
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
| `--topology wheel` | that ring as the rim plus every chord across it: each chip linked directly to each other chip (15 pairs for six chips, one hop everywhere) |
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
| wheel, `--links 6` | 90 (48) | 172 | 1.00 / 1 | 0.680 / 0.845 | 0.363 / 0.622 | 0.0018 | chip 4, 0.253 |

Ghost-loom plan of the same topologies (`field plan --loom ghost`, per-chip caps of one 8 GB die):

| Topology | Links placed | Ghosts per chip | Peak per chip (GiB) | Pairs left unconnected |
|---|---|---|---|---|
| grid | 64 | 18 to 27 | 3.0 to 4.4 | 0 |
| ring | 57 | 18 to 20 | 2.4 to 3.5 | 0 |
| hub 1, `--links 8` | 36 (1-5 gets 4 of 8) | hub 36, others 4 to 8 | hub 5.3, others 0.3 to 2.3 | 0 |
| hub 1, `--links 10` | 35 (1-5 gets 0) | hub 35 | hub 5.3 | 5 |
| grid + 0-5, 2-3 | 84 | 27 to 29 | 3.9 to 4.4 | 0 |
| wheel, `--links 4` | 60 (4 per pair) | 20 | 2.2 to 4.3 | 0 |
| wheel, `--links 6` | 90 (6 per pair) | 30 | 4.0 to 5.0 | 0 |
| wheel, `--links 8` | 102 (2 to 8 per pair) | 30 to 37 | 4.8 to 5.5 | 0 |
| wheel, `--links 10` | 96 (0-4, 0-5, 1-3 get 0) | 28 to 35 | 4.6 to 5.3 | 3, reached in 2 hops |

| Finding | |
|---|---|
| Mean loom | the woven signal falls with the number of link anchors, about 0.0016 per link at d4 whatever the topology; hops do not enter, since no entanglement crosses a link |
| Hub | the router chip is the hot spot: lowest chip signal (0.295 at d6), and in the ghost plan 36 ghosts and 5.3 GiB against 0.3 to 2.3 GiB on the others |
| Hub ports | only border sites of a 12 x 10 chiplet have a coupler colour free; with five pairs of 10 the placement finds 42 links on the hub, so a hub needs `--links 8` or fewer |
| Wheel | 6 links per pair is the most that gives all 15 pairs the same count on the 8 GB dies; on the mean loom it costs the same per link as any other topology (90 links, the signal of grid + 0-5, 2-3) |
| Wheel, four-chip test | four 2 x 3 exact chips, one link per pair, depth 8, classical fidelity of the output: ghost loom 0.72 (wheel, 6 pairs) against 0.74 (grid, 4 pairs); mean loom 0.42 against 0.52; links cut 0.35 against 0.42. Direct links add couplers to approximate, they do not raise random-circuit fidelity; they remove the relay through a third chip |
| Long-range sites | the facing borders are taken by the grid links, so shortcuts land on the outer borders (chip 0's top row and left column for 0-5) |
| Next | the hop penalty itself needs the ghost loom (six 8 GB dies) and a cross-chip observable (GHZ or Bell pairs at 1, 2, 3 hops) |

```
python3 nighthawk_qrack.py field run --device r3 --topology hub --links 8 --loom mean [--cut-links] \
        --family hamming --protocol nnqab --fold 1 --depths 4 6 --instances 2 --cpu --no-ace-error-detection --out r3_topo.jsonl
python3 nighthawk_graph.py --out r3_mean.jsonl --out r3_topo.jsonl --poster r3_hub1.png --poster-topology "hub 1"
```

The field poster names the topology in its title, tints every ACE patch inside the chips and outlines it in the patch's colour (labels p0, p1, ...; the seam qubits between the outlines stay untinted), rings the port qubits of a `--link-port` field, draws long-range l-couplers as dashed arcs, shows the chip-to-chip hops as a table and a tile, adds the hops of the ghost plan when it leaves pairs unconnected, and draws the other topologies of the same field as grey lines in the signal panel. `--poster-topology LABEL` (grid, ring, wheel, hub N, grid + I-J, ...) picks which one.

```
python3 nighthawk_qrack.py field plan --device r3 --topology wheel --links 6 --loom ghost
python3 nighthawk_qrack.py lab --exp wheel --instances 4 --out lab_wheel.jsonl
python3 nighthawk_qrack.py lab --r3plan lab_r3.json
python3 nighthawk_graph.py --lab lab_wheel.jsonl --lab-r3 lab_r3.json --lab-poster posters/lab     # posters/lab_wheel.png
python3 nighthawk_graph.py --out r3_mean.jsonl --out r3_topo.jsonl --poster posters/r3_wheel.png --poster-topology wheel
```

### 3.7 Configuration lab: `lab`

`nighthawk_qrack.py lab` runs fields of at most 24 qubits whose whole register is also simulated exactly, so each way of building the device is scored against the truth of the same circuit (Haar single-qubit gates, CZ on the four coupler colours; `fxeb_cycles`).

| Score | Meaning | Available for |
|---|---|---|
| `fclass` | classical fidelity (sum sqrt(p q))^2 of the whole output distribution | looms and QrackBridge (exact or bridge chips) |
| `fb_intra`, `fb_cross` | the same on 8-qubit blocks inside a chip / around a link | everything (from 8192 samples for ACE) |
| `fxeb` | linear XEB over the ideal XEB | everything; exceeds 1 for a truncated bridge, so not used for ranking |
| `signal` | mirror observable 1 - 2h | everything; 1.00 with the links cut, so not a fidelity between interconnects |
| `width` | qubits of the widest simulator | everything |

| Option | Effect |
|---|---|
| `--exp inter links rows inside mixed holes numa` | which experiments (table below) |
| `--depths`, `--instances`, `--shots`, `--out` | defaults 4 6 8 12, 5, 4096, `lab.jsonl`; resumable |
| `--only TEXT`, `--list` | select / list configurations |
| `--r3plan JSON` | the full-size r3 ghost-loom plan, option by option |
| `--shor JSONL`, `--repeats` | Shor's code across the link (section 3.8) |
| `--selftest` | 8 checks: 6 on 16 qubits (exact scores 1, bridge with spare bonds scores 1, cut keeps signal 1) |

Between two exact 3 x 4 chips, `fclass` (3 to 4 instances):

| Interconnect | 1 link, d8 | 1 link, d12 | 3 links, d8 | 3 links, d12 | Widest simulator (3 links) |
|---|---|---|---|---|---|
| ghost loom | 0.80 | 0.72 | 0.75 | 0.74 | 15 |
| mean-field loom | 0.69 | 0.59 | 0.49 | 0.49 | 12 |
| sample loom | 0.69 | 0.58 | 0.49 | 0.47 | 12 |
| links cut | 0.67 | 0.57 | 0.49 | 0.49 | 12 |
| ghost loom, stitched readout | 0.66 | 0.58 | 0.58 | 0.50 | 15 |
| QrackBridge, 1 bond | 0.91 | 0.80 | 0.59 | 0.50 | 13 |
| QrackBridge, 2 bonds | 1.00 | 0.92 | 0.71 | 0.58 | 14 |
| QrackBridge, 3 bonds | 1.00 | 1.00 | 0.83 | 0.66 | 15 |
| QrackBridge, 4 bonds | 1.00 | 1.00 | 0.93 | 0.73 | 15 |

Inside one 4 x 5 chip (a row cut crosses 5 couplers):

| Engine | Widest simulator | `fb_intra` d6 | d8 | d12 | `fclass` d8 |
|---|---|---|---|---|---|
| ACE 20:2:4 | 13 | 0.950 | 0.960 | 0.979 | n/a |
| ACE 20:2:4, no error detection | 12 | 0.956 | 0.965 | 0.979 | n/a |
| ACE 20:1:4 | 11 | 0.914 | 0.925 | 0.958 | n/a |
| QrackBridge 2 patches, 2 bonds | 12 | 0.947 | 0.704 | 0.712 | 0.25 |
| QrackBridge 2 patches, 3 bonds | 13 | 0.981 | 0.848 | 0.812 | 0.39 |
| QrackBridge 2 patches, 4 bonds | 14 | 0.995 | 0.926 | 0.900 | 0.52 |
| QrackBridge 2 patches, 6 bonds | 16 | 1.000 | 0.989 | 0.971 | 0.75 |

Inside x between (two 3 x 4 chips, 3 links, depth 8), `fb_intra` / `fb_cross`:

| Inside | Ghost loom | Mean-field loom | Links cut |
|---|---|---|---|
| exact | 0.994 / 0.985 | 0.964 / 0.940 | 0.965 / 0.941 |
| QrackBridge 2 patches, 3 bonds for 3 couplers | 0.970 / 0.967 | 0.943 / 0.911 | 0.946 / 0.911 |
| QrackBridge 2 patches, 2 bonds | 0.937 / 0.943 | 0.912 / 0.881 | 0.912 / 0.881 |
| ACE 24:3:3 | 0.917 / 0.931 | 0.888 / 0.887 | 0.894 / 0.885 |
| ACE 20:2:4 | 0.924 / 0.948 | 0.887 / 0.885 | 0.892 / 0.888 |
| ACE 16:2:4 (room for 2 of 3 ghosts) | 0.913 / 0.927 | 0.886 / 0.881 | 0.892 / 0.888 |
| QrackBridge everywhere (4 patches), 1 / 2 / 3 / 4 bonds | 0.875 / 0.832, 0.942 / 0.933, 0.986 / 0.986, 0.997 / 0.998 | | |

Placement, disabled qubits and topology, `fclass` at depth 8, exact chips:

| Case | Ghost | Mean | Bridge 1 bond | Bridge 2 bonds | Cut |
|---|---|---|---|---|---|
| 1 link, corner row / centre row | 0.80 / 0.79 | 0.69 / 0.72 | 0.91 / 0.94 | 1.00 / 1.00 | 0.67 / 0.65 |
| 2 links, rows 0,1 / rows 0,2 | 0.76 / 0.73 | 0.58 / 0.52 | 0.76 / 0.66 | 0.89 / 0.81 | 0.56 / 0.52 |
| 1 link, border corner / chip centre | 0.87 / 0.86 | 0.81 / 0.77 | 0.90 / 0.98 | 1.00 / 1.00 | 0.79 / 0.69 |
| 2 links, disabled at the links / far | 0.74 / 0.75 | 0.54 / 0.60 | 0.75 / 0.63 | 0.91 / 0.81 | 0.51 / 0.54 |
| six 2 x 2 chips, grid / ring / grid + 0-5 | 0.57 / 0.58 / 0.58 | 0.30 / 0.25 / 0.25 | chain 0.23 / 0.36 / 0.21 | chain 0.39 / 0.66 / 0.32 | 0.18 / 0.18 / 0.17 |
| the same, bridge ring 0-1-2-5-4-3 | | | 0.35 / 0.65 / 0.29 | 0.71 / 1.00 / 0.50 | |

r3 at full size on the ghost loom (`lab --r3plan`, six 8 GB dies):

| Option | Links with a ghost | Skipped | Ghosts per chip | Peak per chip (GiB) |
|---|---|---|---|---|
| 4 links per pair | 28 | 0 | 8 to 12 | 0.5 to 1.0 |
| 6 links per pair | 42 | 0 | 12 to 18 | 1.0 to 2.7 |
| 8 links per pair | 56 | 4 | 16 to 24 | 1.6 to 3.8 |
| 10 links per pair | 64 | 8 | 18 to 27 | 3.0 to 4.4 |
| every facing row and column | 70 | 8 | 19 to 30 | 3.1 to 4.8 |
| 10 links, `--gtt 16` | 68 | 2 | 20 to 28 | 3.1 to 18.3 |
| 10 links, ring | 57 | 3 | 18 to 20 | 2.4 to 3.5 |
| 8 links, hub | 36 | 1 | 4 to 36 | 0.3 to 5.3 |

| Finding | |
|---|---|
| Between chiplets | ghost loom: a plateau near 0.75 that depth does not erode; mean-field and sample looms are no better than cutting the links; a bridge needs at least as many bond qubits as links, and more with depth |
| Inside a chiplet | ACE's block fidelity is flat with depth (0.95 to 0.98); a bridge cut with fewer bonds than couplers is exact for one cut layer and then falls below ACE |
| ACE + loom | ACE inside is the larger loss (0.99 to 0.92); the ghost loom keeps a 0.05 lead across the link over mean-field and cut |
| Stitched ghost readout | better at depth 4, worse from depth 6: not adopted |
| Link placement | row on the border, and border against chip centre, make no difference to the ghost loom |
| Disabled qubits | no gain from placing them at the links; a ghost needs a free ACE site coupled to its anchor, which is the register's business |
| Topology | ghost loom 0.57 to 0.58 on grid, ring and grid + long link; a bridge chain only wins where the chips form its chain or ring |
| Mirror signal | 1.00 with the links cut and 0.27 for the mean-field loom at the same fidelity 0.49: it ranks chips, not interconnects |
| Scale | 24 qubits, chips of 4 to 20 qubits: trends and mechanisms, not r3 numbers |

```
python3 nighthawk_qrack.py lab --exp inter mixed links --instances 3 --shots 8192 --out lab_inter.jsonl
python3 nighthawk_qrack.py lab --exp inside rows holes numa --instances 3 --shots 8192 --out lab_place.jsonl
python3 nighthawk_qrack.py lab --r3plan lab_r3.json
python3 nighthawk_graph.py --lab lab_inter.jsonl --lab lab_place.jsonl --lab-r3 lab_r3.json --lab-poster posters/lab
```

### 3.8 Shor's code across a link: `lab --shor`

One [[9,1,3]] unit on the two-chip lab field (chip A's two border columns, chip B's border column), the four logical states, encoder only, exact chips. Z_L is read from block parities in X, X_L from block majorities in Z; "decoded" is after the majority vote; "ghosts" reads chip B's qubits from their replicas in chip A's simulator. 4096 shots, 3 seeds; 1.000 is the exact result.

| Layout | Interconnect | Z_L raw / decoded | X_L raw / decoded | Through the ghosts, Z_L / X_L decoded |
|---|---|---|---|---|
| cut between blocks (1 link) | ghost loom | 1.000 / 1.000 | 0.513 / 0.513 | 1.000 / 1.000 |
| | mean, sample loom | 1.000 / 1.000 | 0.50 / 0.50 | |
| | links cut | 0.500 / 1.000 | 0.50 / 0.50 | |
| | QrackBridge, 1 to 3 bonds | 1.000 / 1.000 | 1.000 / 1.000 | |
| cut through blocks (3 links) | ghost loom | 0.130 / 0.512 | 0.979 / 1.000 | 1.000 / 1.000 |
| | mean-field loom | 0.113 / 0.514 | 0.487 / 1.000 | |
| | sample loom, links cut | 0.126 / 0.50 | 0.126 / 1.000 | |
| | QrackBridge, 1 bond | 0.252 / 0.748 | 1.000 / 1.000 | |
| | QrackBridge, 2 bonds | 0.500 / 1.000 | 1.000 / 1.000 | |
| | QrackBridge, 3 bonds | 1.000 / 1.000 | 1.000 / 1.000 | |

| Finding | |
|---|---|
| Per-chip readout | the logical operator that needs coherence across the link is a coin flip on every loom |
| Readout through the ghosts | the cross-link check is evaluated in one simulator: 1.000 in both layouts |
| Layout | between blocks Z_L survives any interconnect; through blocks X_L survives after the majority vote |
| Scope | encoder only: no syndrome rounds, no noise beyond the interconnect's own |

```
python3 nighthawk_qrack.py lab --shor lab_shor.jsonl [--shots 4096 --repeats 3]
python3 nighthawk_graph.py --lab lab_shor.jsonl --lab-poster posters/lab          # posters/lab_shor.png
```

`lab --selftest` checks [7] and [9] cover it.

### 3.9 QrackBridge across the seam with syndrome measurement: `lab --rounds`

The unit of 3.8 with repeated syndrome extraction, two patches cut at the link, exact against `bridge` with 1 to 4 bond qubits. Each round injects one random Pauli error and extracts the eight stabiliser checks to ancillas (a CX crosses the link where a check straddles it), then returns each ancilla to 0 with the predicted syndrome. Final readout is decoded with the Pauli frame. 1024 shots, 2 seeds, rounds 0, 1, 2, 4, 8, 16; 1.000 is the exact result.

| Layout | CXs across the link per round | Bonds | Z_L decoded, rounds 1 to 16 | X_L decoded, rounds 1 to 16 | Truncating sweeps per run |
|---|---|---|---|---|---|
| cut between blocks | 3 | 1 | 1.00 | 0.98 to 0.99 | 1, all in round 1 |
| | | 2, 3, 4 | 1.00 | 1.00 | 0 |
| cut through blocks | 11 | 1 | 0.50 | 1.00 | 3.5 |
| | | 2 | 0.63 to 0.76 | 1.00 | 2.5 |
| | | 3 | 0.78 to 0.83 | 1.00 | 2 to 3.8 |
| | | 4 | 1.00 | 1.00 | 0 |

| Workload across the link | Bonds for an exact result | Below that |
|---|---|---|
| syndrome rounds, cut between blocks | 2, at any number of rounds | 0.99, flat in rounds |
| syndrome rounds, cut through blocks | 4, at any number of rounds | 0.79 at 3 bonds, flat in rounds |
| random circuit, 3 links (3.7) | 3 at depth 4, 6 at depth 6 and 8 | fidelity falls with depth: 0.94, 0.64, 0.71, 0.58 at 2 bonds |

| Limit | |
|---|---|
| Measurement | `QrackBridge` has no measure, force-measure or reset; the ancilla reset here uses the syndrome predicted from the injected errors, so it covers known errors only |
| Patch size | patches are dense: construction fails with `std::bad_alloc` at 40 qubits per patch on this 8 GB container, so a patch cannot hold a 120-qubit chiplet |
| Use | one bridge per code unit that straddles a link (as `ghz shor --engine bridge` does per unit), widest patch 14 to 16 qubits, with the chiplets on ACE |

```
python3 nighthawk_qrack.py lab --rounds lab_rounds.jsonl [--shots 1024 --repeats 2]
python3 nighthawk_graph.py --lab lab_rounds.jsonl --lab lab_inter.jsonl --lab-poster posters/lab   # posters/lab_rounds.png
```

`lab --selftest` check [8] covers it.

### 3.10 One port qubit per chip: `--link-port`

Every l-coupler of a chip on one qubit, one coupler per chip pair of `--topology` / `--pairs` (`--links` and `--long-links` are not used). A qubit takes one two-qubit gate per layer, so the port couplers get layers of their own after A, B, C, D. A port is a router qubit with a coupler to every partner chip, not the point-to-point l-couplers of the device.

| Option | Round of the cycle (six-chip wheel, 15 pairs in 5 matchings) | Each coupler fires |
|---|---|---|
| `--link-port layers` | A B C D P1 P2 P3 P4 P5 | once per round |
| `--link-port mux` | A B C D P, P carrying one matching per round, round robin | once per 5 rounds |
| `--port-site inner` (default) | the chip's site nearest the centre of the field | |
| `--port-site centre` | the chip's site nearest its own centre | |

`--cut-links` keeps the port layers (empty), so the baseline has the same single-qubit layers. On the ghost loom every chip holds a ghost of every partner port, and a port-to-port gate is applied on every chip that holds both ghosts.

r3 as a wheel on the ghost loom (`field plan`, 8 GB dies):

| Link placement | l-couplers | Pairs linked | Link qubits | Ghosts per chip | GiB per chip |
|---|---|---|---|---|---|
| `--links 6` (3.6) | 90 | 15 of 15 | 172 | 30 | 4.0 to 5.0 |
| `--link-port layers` or `mux` | 14 | 14 of 15 (2-3 dropped: no ACE site left beside the port) | 6 | 4 to 5 | 0.7 to 2.2 |
| the same, `--port-site centre` | 15 | 15 of 15 | 6 | 5 | 0.3 |

Exact chips on a wheel, classical fidelity of the output distribution against the exact state of the same circuit (`lab --exp port`, 3 instances). Round 1 is left out: where the l-couplers sit in the last layer they cannot change the output distribution.

| Field | Links | Rounds | Layers | l-coupler gates | Ghost loom | Mean-field loom | Links cut |
|---|---|---|---|---|---|---|---|
| 4 chips (2 x 3) | a link qubit per partner | 2 | 8 | 12 | 0.71 | 0.41 | 0.37 |
| | | 3 | 12 | 18 | 0.66 | 0.31 | 0.29 |
| | port, layers | 2 | 14 | 12 | 0.72 | 0.38 | 0.42 |
| | | 3 | 21 | 18 | 0.70 | 0.33 | 0.31 |
| | port, mux | 2 | 10 | 4 | 0.71 | 0.59 | 0.46 |
| | | 3 | 15 | 6 | 0.69 | 0.44 | 0.38 |
| | | 6 | 30 | 12 | 0.69 | 0.32 | 0.33 |
| 6 chips (2 x 2) | port, layers | 2 | 18 | 30 | 0.75 | 0.25 | 0.25 |
| | port, mux | 2 | 10 | 6 | 0.75 | 0.58 | 0.49 |
| | | 5 | 25 | 15 | 0.69 | 0.20 | 0.21 |

r3 at full size, mean loom, Hamming family, nnqab, fold 1, CPU (`r3_port.jsonl`); signal woven / links cut:

| Schedule | Depth (rounds) | l-coupler gates | Instances | Signal | Port qubits | Cost per l-coupler gate |
|---|---|---|---|---|---|---|
| layers | 9 (1) | 15 | 1 | 0.785 / 0.819 | -0.38 / 0.99 | 0.0022 |
| layers | 18 (2) | 30 | 2 | 0.317 / 0.337 | -0.08 / 0.77 | 0.0007 |
| mux | 10 (2) | 6 | 1 | 0.383 / 0.402 | 0.14 / 0.75 | 0.0031 |

| Finding | |
|---|---|
| Fidelity | at equal couplers and equal gates per round, one port and a link qubit per partner score the same on the ghost loom (0.72 against 0.71 at 2 rounds, within the spread) |
| Memory | 5 ghosts per chip in place of 30; a centre port fits all 15 pairs at 0.3 GiB per chip |
| Multiplexing | the ghost loom holds 0.69 over 6 rounds for the same 12 gates; the looms without ghosts decay with the extra rounds |
| Port qubit | on the mean loom the port's own signal is gone after one round (-0.38 against 0.99 with the links cut): the damage of all the links lands on one qubit per chip |
| Field signal | a port field loses 0.02 to 0.03 to its links where 90 dedicated links lose 0.26 (3.6, depth 6): it fires 15 link gates per round, not 90 per four layers. Compare fields at equal link gates |

```
python3 nighthawk_qrack.py field plan --device r3 --topology wheel --link-port layers --loom ghost [--port-site centre]
python3 nighthawk_qrack.py field run  --device r3 --topology wheel --link-port layers --loom mean [--cut-links] --family hamming --protocol nnqab --fold 1 --depths 9 18 --cpu --out r3_port.jsonl
python3 nighthawk_qrack.py lab --exp port --instances 3 --out lab_port.jsonl
python3 nighthawk_graph.py --lab lab_port.jsonl --lab-r3 lab_r3.json --lab-poster posters/lab            # posters/lab_port.png
python3 nighthawk_graph.py --out r3_port.jsonl --poster posters/r3_port.png --poster-topology "wheel, port layers"
```

`field selftest` check [13] covers the placement and both schedules.

### 3.11 Quantum NUMA as a number: `lab --bell`

A Bell pair between every pair of six 3 x 4 chips on a 2 x 3 grid, one l-coupler per chip pair of the topology. H on the link anchor A, CXs down the shortest path to the anchor B (through relay chips where the two chips are not linked), then the CXs back, which return every relay qubit to 0 and leave (|00> + |11>) on A and B. Fidelity 1.00 is the Bell pair, 0.50 the most a state without entanglement reaches, 0.25 no correlation. 4096 shots.

By hops (the same on every topology that has the distance):

| Hops | Relay chips | CX (mean) | On l-couplers | Exact simulator | Ghost loom, the two chips read apart | Ghost loom, read in an end chip | Ghost loom, best simulator | Mean-field loom |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 1 | 1 | 1.00 | 0.25 | 1.00 | 1.00 (end chip) | 0.50 |
| 2 | 1 | 8.3 to 9.7 | 3 | 1.00 | 0.25 | none holds both ends | 1.00 (relay chip) | 0.50 |
| 3 | 2 | 15 to 21 | 5 | 1.00 | 0.25 | none | 0.25 (no simulator holds both ends) | 0.50 |

All 15 chip pairs per topology (ghost loom):

| Topology | l-couplers | Hops mean / max | CX per pair | On l-couplers | Read in an end chip | Read in any chip | Chips 0 and 5 |
|---|---|---|---|---|---|---|---|
| grid | 7 | 1.67 / 3 | 6.7 | 2.3 | 7 of 15 | 13 of 15 | 3 hops, 15 CX, 0.25 |
| ring | 6 | 1.80 / 3 | 7.5 | 2.6 | 6 of 15 | 12 of 15 | 3 hops, 15 CX, 0.25 |
| grid + 0-5 | 8 | 1.53 / 3 | 5.8 | 2.1 | 8 of 15 | 14 of 15 | 1 hop, 1 CX, 1.00 |
| wheel | 15 | 1.00 / 1 | 1.0 | 1.0 | 15 of 15 | 15 of 15 | 1 hop, 1 CX, 1.00 |

| Finding | |
|---|---|
| Gates | the far corners cost 15 CX (5 on l-couplers) through 2 relay chips on the grid and 1 CX on the wheel; a noiseless exact simulator gives 1.00 for both, so on hardware the hops are paid in gate error |
| Loom reach | the ghost loom carries entanglement one hop: a chip holds a ghost of its link partners and of nothing further. Linked chips read the pair in either end chip; at 2 hops only the relay chip holds both ends; at 3 hops no simulator does |
| Loom output | the loom's own output (chips read independently) is 0.25 at any distance; the pair is only visible inside one chip's simulator |
| Mean-field loom | 0.50 at any distance: correlation without entanglement |
| Not measured | the poster's survival curve is the grid's gate counts as (1 - p) per gate under assumed error rates, not a Nighthawk figure |

```
python3 nighthawk_qrack.py lab --bell lab_bell.jsonl [--shots 4096]
python3 nighthawk_graph.py --lab lab_bell.jsonl --lab-poster posters/lab          # posters/lab_bell.png
```

`lab --selftest` check [10] covers it.

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
