# Nighthawk, consolidated: `nighthawk_qrack.py`

One script for every Nighthawk simulation in `devices/nighthawk`, plus IBM Quantum Nighthawk r3 as a field of six 120-qubit chiplets.

| | |
|---|---|
| Script | `nighthawk_qrack.py`, 18,445 lines |
| Viewers | `nighthawk_graph.py`, `nighthawk_3dgraph.py` (the `dual-nighthawk` versions, unchanged) |
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
| `field selftest` | 11 / 11, 28 s |
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
