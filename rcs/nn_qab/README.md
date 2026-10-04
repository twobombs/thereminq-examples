# HOWTO: `rcs/nn_qab/`

**Nearest-neighbour random-circuit sampling with automatic circuit elision: measuring the bulk-to-boundary dependence of Qrack's ACE backend against an exact reference**

| | |
|---|---|
| Directory | [`rcs/nn_qab/`](https://github.com/twobombs/thereminq-examples/tree/main/rcs/nn_qab) (ThereminQ examples) |
| Upstream | `rcs/nn_qab.py` in [`vm6502q/pyqrack-examples`](https://github.com/vm6502q/pyqrack-examples), by Dan Strano and (Anthropic) Claude |
| Simulator | [Qrack / PyQrack](https://github.com/unitaryfund/pyqrack) — `QrackAceBackend` (method under test), `QrackSimulator` (reference) [[6]](#ref-6) |
| Acceleration | OpenCL or CPU. No CUDA path is used. |
| Licence | Rights and licence of the original code remain with Dan Strano *et al.*; the ThereminQ modifications concern environment handling, device pinning, sweep orchestration and analysis |
| Companion | [`rcs/nighthawk/`](https://github.com/twobombs/thereminq-examples/tree/main/rcs/nighthawk) — the same estimator applied to the 61-qubit IBM Nighthawk RCS circuits |

---

## Abstract

This directory measures how well Qrack's approximate, seam-partitioned `QrackAceBackend` (ACE) reproduces nearest-neighbour random quantum circuits, as a function of the geometry of its partition. A circuit of width n and depth d is run on ACE, sampled, and scored by a normalised linear cross-entropy benchmark (XEB) and the heavy-output generation (HOG) probability against the exact 2ⁿ output distribution of the same circuit. The independent variable is the *bulk-to-boundary ratio* (B-to-B) of the ACE register — qubits held exactly inside one patch over qubits replicated across a seam — set through `long_range_columns` and `long_range_rows`. Sweeps of 100 seeded circuits per configuration at depth 12 are orchestrated in two passes (ACE on a small device or CPU; exact reference on the device with the most memory) through lock-file work queues, resumable per seed. An extended reference path reaches widths 30–36 on a 320 GB host with an explicit memory guard, a linear (Porter–Thomas) estimator fall-back and swap-traffic accounting. A separate adversarial harness scores ACE against a seam-severed control, an ideal-sampling ceiling and a p-blind null, following the spoofing analysis of Gao *et al.* [[4]](#ref-4), and pools repeated runs with a random-effects model [[7]](#ref-7), [[8]](#ref-8). Across seven configurations from 10 to 27 qubits the mean XEB at depth 12 rises from 0.033 at B-to-B 1.0 to 0.241 at B-to-B 5.5, and at fixed width 27 from 0.111 (three patches) to 0.185 (two patches); between-seed standard deviations of 0.07–0.14 make 100 seeds per series a requirement rather than a luxury.

---

## 1. Purpose and scope

Qrack's ACE backend approximates a wide register by splitting it into patches that are each simulated exactly, joined by boundary (seam) qubits that are replicated in the adjacent patches and reconciled after gates that touch them [[6]](#ref-6). The quality of that approximation is not a single number: it depends on how much of the register sits on seams. This directory makes that dependence measurable:

1. **`nn_qab.py`** builds a seeded nearest-neighbour random circuit, runs it on ACE, runs the identical circuit on an exact `QrackSimulator`, and scores the ACE samples (single run, or a two-pass sweep).
2. **`nn_qab.sh`** regenerates the bulk-to-boundary series set (widths 10–28) under one PyQrack version, 100 seeds per series.
3. **`nn_qab32plus.py` / `nn_qab32plus.sh`** extend the series to widths 30–36, where the exact probability vector no longer fits comfortably in host memory.
4. **`nn_qab-adversarial.py`** separates *approximate simulation* from *XEB spoofing* with matched control arms and per-region error metrics.
5. **`nn_qab_graph.py`** plots XEB against B-to-B and width in three dimensions, fits line and plane models with cross-validation, and exports tidy CSVs.

There is no noise model anywhere: the ACE backend runs ideal gates, so every deviation of its XEB from the sampling ceiling is attributable to its partition and seam treatment.

---

## 2. Background

### 2.1 Circuit family

The register is laid out as a `row_len × col_len` rectangle obtained by `factor_width(width)` (the factorisation closest to square), with wrap-around in both directions. Each of `depth` layers applies:

- **Single-qubit gates:** `u(θ, φ, λ)` on every qubit, with φ, λ uniform on [−π, π) and θ = arcsin(r/π) for r uniform on [−π, π), i.e. **sin θ uniform on [−1, 1]**, θ ∈ [−π/2, π/2]. This distribution is concentrated near θ = 0 and is *not* the Haar measure on SU(2) (for which cos θ is uniform on [−1, 1], θ ∈ [0, π]); outputs are correspondingly less anticoncentrated than those of Haar-random circuits at equal depth.
- **Two-qubit gates:** one coupler colouring per layer, cycling through an eight-step activation sequence `[0, 3, 2, 1, 2, 1, 0, 3]` in the style of the Sycamore `ABCDCDAB` pattern [[2]](#ref-2). On each active coupler a gate is drawn uniformly from twelve: `swap`, `pswap`, `mswap`, `nswap`, `iswap`, `iiswap` (adjoint iSWAP), `cx`, `cy`, `cz` and their anti-controlled forms `acx`, `acy`, `acz`.
- **Swap realisation:** `--swap-mode swap` uses ACE's native `swap()`/`iswap()`; `cnot` decomposes every swap into three CNOTs through the ordinary controlled-Pauli path; `auto` (default) chooses `cnot` when the measured B-to-B ratio is ≥ 7.

Circuits are reproducible from `--seed`; the two-pass sweep serialises the circuit with the ACE counts so that the reference provably runs the same gates.

### 2.2 ACE geometry and the bulk-to-boundary ratio

`QrackAceBackend(width, long_range_columns=lrc, long_range_rows=lrr, is_torus=True)` partitions the register into patches. Its geometry is read from the backend itself (`_unpack`), not re-derived: a logical qubit with more than one physical location is a boundary (seam) qubit, the rest are bulk, and

```
B-to-B = (bulk qubits) / (boundary qubits)
```

(`bulk_to_boundary_ratio`, `inf` for a register without seams). The series are chosen as the geometrically optimal two-patch configuration for each width (`nn_qab.sh`), so that the ratio, not the patch count, varies between series.

### 2.3 Estimators

With p the exact output distribution, q the empirical frequency of the ACE samples, N = 2ⁿ and μ = 1/N, `calc_stats` computes

```
XEB = Σ_x (p(x) − μ)(q(x) − μ) / Σ_x (p(x) − μ)²  =  (N Σ_x q(x) p(x) − 1) / (N Σ_x p(x)² − 1)
```

i.e. linear XEB normalised by the circuit's own second moment rather than by its Porter–Thomas value; it equals 1 for ideal sampling and 0 for uniform noise. Linear XEB as a fidelity proxy is due to Boixo *et al.* [[1]](#ref-1) and was used at scale in [[2]](#ref-2); its limits as a certificate are analysed in [[4]](#ref-4) and [[5]](#ref-5). **HOG** is the sampled probability mass on bitstrings whose ideal probability exceeds the median of p, the heavy-output test of Aaronson and Chen [[3]](#ref-3). Only the ≤ 1024 sampled amplitudes and two float64 reductions over p are needed, so the estimator's peak memory is the probability vector alone.

Above the width where 2ⁿ probabilities fit, `nn_qab32plus.py` switches to the **linear (Porter–Thomas) estimator**, using only the sampled amplitudes (read in O(1) each with `prob_perm`) and the Porter–Thomas values Σp² → 2/N and median → ln 2 / N:

```
XEB_linear = N Σ_x q(x) p(x) − 1,        HOG_linear: threshold ln 2 / N
```

This is the standard linear XEB of [[2]](#ref-2) but a *different estimator*; exact-mode rows therefore record both, so overlapping widths calibrate the change.

### 2.4 Approximate simulation versus spoofing

An approximate simulator can score a high XEB either by tracking the circuit's amplitudes or by exploiting a weakness of the estimator (e.g. preserving only within-patch correlations, as in the cutting attacks of [[4]](#ref-4)). `nn_qab-adversarial.py` distinguishes the two with arms scored against the same ideal distribution and the same circuit: `ace` (method under test), `severed` (ACE with seam reconciliation disabled — elision without repair), `ceiling` (`shots` samples drawn from the exact distribution, the best any sampler can score at that budget) and `null` (ACE counts against an *independent* circuit's ideal distribution: a sampler that does not exploit knowledge of p scores ≈ 0 here, one that does would not). Correlator errors ⟨Z_iZ_j⟩ are reported separately for coupled pairs touching a seam and pairs inside a patch, optionally also in the X and Y bases, since a Z-only measurement cannot see phase repair.

### 2.5 Statistics across seeds and runs

Each sweep series is summarised by n, mean and standard deviation over seeds. In the adversarial harness each metric carries a bootstrap 95 % confidence interval per arm, ace-versus-severed is assessed as a *paired* difference, and repeated independent runs are pooled with the DerSimonian–Laird random-effects estimator [[7]](#ref-7), reporting Cochran's Q, τ² and I² [[8]](#ref-8). The harness documents why: two runs of an identical n = 20 command gave paired ace−severed differences of +0.138 (t = +3.98) and −0.041 (t = −1.59), I² = 94 %; the within-run standard error is anticonservative and the run, not the repetition, is the unit of analysis.

---

## 3. Requirements and environment

| Component | Notes |
|---|---|
| Python ≥ 3.9 | numpy; matplotlib for `nn_qab_graph.py` |
| PyQrack with `QrackAceBackend` | the exact reference above 32 qubits needs Qrack built in > 32-qubit mode |
| bash | for the sweep scripts |

All PyQrack-invoking scripts resolve the shared library **before** importing PyQrack, because `PYQRACK_SHARED_LIB_PATH` is read at import time:

```python
QRACK_LIB_PATH = os.environ.get("QRACK_LIB_PATH", "/usr/local/lib/qrack/libqrack_pinvoke.so")
if os.path.isfile(QRACK_LIB_PATH):
    os.environ["PYQRACK_SHARED_LIB_PATH"] = QRACK_LIB_PATH
```

If that library links against siblings in the same directory, the dynamic loader must find them at process start (`-Wl,-rpath,/usr/local/lib/qrack`, or `LD_LIBRARY_PATH` exported by the container entrypoint); setting it from Python is too late.

**Device pinning.** Qrack uses every detected OpenCL device by default, which on a mixed host scatters patch simulators across cards. All modes accept `--device N` (precedence `--device` > `$QRACK_DEVICE` > `0`), which sets `QRACK_OCL_DEFAULT_DEVICE`, `QRACK_QPAGER_DEVICES` and `QRACK_QUNITMULTI_DEVICES`; variables set explicitly by the user are left alone. In `nn_qab32plus.py` a comma list (`--device 1,2,3,4,5`) pages one simulator across several devices, its first entry becoming the default device.

---

## 4. Files

| File | Lines | Role |
|---|---|---|
| `nn_qab.py` | 934 | single run (`run`, or legacy positional form) and two-pass sweep (`ace`, `ideal`, `merge`) |
| `nn_qab.sh` | 250 | series sweep, widths 10–28, one device, resumable |
| `nn_qab32plus.py` | 1172 | `nn_qab.py` plus `--stats exact|linear|auto`, host-memory guard, swap accounting, multi-device paging |
| `nn_qab32plus.sh` | 388 | series sweep, widths 10–36, three reference profiles by width |
| `nn_qab-adversarial.py` | 1415 | `bench`, `run`, `pool`: control arms, per-region metrics, random-effects pooling |
| `nn_qab_graph.py` | 531 | 3-D XEB × B-to-B × width view, line/plane fits with LOOCV, CSV export (its docstring still names it `plot3d.py`) |

---

## 5. Recommended workflow

```bash
cd thereminq-examples/rcs/nn_qab

# 1  one self-contained run (legacy form: WIDTH DEPTH [LRC] [LRR] [SWAP_MODE])
python3 nn_qab.py 14 12 3 2
python3 nn_qab.py run --width 14 --depth 12 --lrc 3 --lrr 2 --seed 7

# 2  smoke test of the series sweep, then the full set (100 seeds, depth 12)
SEEDS=0-9 ./nn_qab.sh 10
./nn_qab.sh

# 3  widths 30-36 (point SCRIPT at the 32plus driver; one seed at 36 first)
SCRIPT=./nn_qab32plus.py SEEDS=0 ./nn_qab32plus.sh 36
SCRIPT=./nn_qab32plus.py ./nn_qab32plus.sh

# 4  approximate simulation or spoofing?
python3 nn_qab-adversarial.py run --runs 4 --reps 50 14 12 3 2 --bases zxy --csv summary.csv

# 5  analysis
python3 nn_qab_graph.py --runs runs --show
```

---

## 6. Reference

### 6.1 `nn_qab.py`

| Mode | Function |
|---|---|
| legacy | `nn_qab.py WIDTH DEPTH [LRC=4] [LRR=4] [SWAP_MODE=auto]` — ACE, then exact reference, then statistics; output format unchanged from the upstream script |
| `run` | the same with named flags: `--width`, `--depth`, `--lrc`, `--lrr`, `--swap-mode`, `--seed`, `--engine`, `--sdrp` |
| `ace` | sweep pass A: build the circuit from each seed, run ACE, save counts **and** the serialised circuit to `<out>/ace/<seed>.json` |
| `ideal` | sweep pass B: reload the same circuit, run the reference, score, write `<out>/xeb/<seed>.json` |
| `merge` | collect `<out>/xeb/*.json` into one CSV; print n, mean, stdev |

Shots per circuit are `2^min(10, width + 2)`, i.e. 1024 at every width ≥ 8. Sweep workers coordinate through `O_EXCL` lock files: launch N copies of the same command and they divide the seed pool; finished seeds are skipped, so an interrupted sweep resumes. Pass A is dominated by Python-side seam reconciliation (`_correct()`/`prob()`) and suits a small card or the CPU; pass B is about 90 % of wall time and belongs on the device with the most memory.

**Reference engines** (`--engine`): `statevector` (default), `cpu`, `qbdd` (binary decision tree), `sparse`, `stabilizer`. Measured on this circuit family at depth 12 on CPU, QBDD was 216–294× slower than the state vector at widths 12–16, scaled worse, and deviated from state-vector amplitudes by up to 6.3 % per permutation (median 0.5 %): random circuits are built to have none of the structure such engines exploit. **`--sdrp R`** enables Qrack's Schmidt-decomposition rounding, making the reference approximate but self-reporting (`unitary_fidelity` is recorded); an XEB scored against an SDRP reference is only interpretable together with that fidelity.

### 6.2 `nn_qab.sh`

Regenerates the series set under one PyQrack version, one series at a time, cheapest first, overlapping the ACE and reference passes of the same series. Series (`width lrc lrr`): `10 2 2`, `14 3 2`, `22 5 2`, `26 6 2`, `27 4 3`, `28 3 4`; the width-28 series repeats the width-14 B-to-B ratio (2.5) and breaks the width/ratio collinearity in the fit.

| Variable | Default | Meaning |
|---|---|---|
| `SEEDS` | `0-99` | seed range (`0-9,20,30-39` syntax) |
| `DEPTH` | `12` | circuit depth |
| `OUT_ROOT` | `runs` | output directory |
| `DEVICE`, `ACE_DEVICE`, `IDEAL_DEVICE` | `0` | OpenCL device for both / each pass |
| `ACE_JOBS`, `IDEAL_JOBS` | `1`, `2` | concurrent workers per pass |
| `PY`, `SCRIPT` | `python3`, `./nn_qab.py` | interpreter, driver |

At width 28 a reference worker needs ≈ 3.6 GiB of device memory (2 GiB state vector plus context) and ≈ 5.3 GiB of host RAM; an ACE worker ≈ 0.4 GiB (patch simulators ≤ 20 qubits). The defaults (1 ACE + 2 reference workers) sit near 7.6 GiB of a 10 GiB card.

### 6.3 `nn_qab32plus.py` (additions)

| Flag | Meaning |
|---|---|
| `--stats auto|exact|linear` | full-vector estimator (needs 2ⁿ probabilities) or linear Porter–Thomas estimator from sampled amplitudes; `auto` tries exact first |
| `--no-mem-check` | disable the host-memory guard |
| `--allow-swap` | count `SwapFree` in the guard's budget |

The guard estimates the host bytes a reference needs (12 B per amplitude for exact statistics in an fp32 build — 8 B state, 4 B probabilities — plus 1 GiB base; doubled for fp64) and **refuses with exit code 3** if neither estimator fits; the sweep then skips that series' reference while its ACE pass still completes, and each `ace/*.json` retains circuit and counts for a later reference. Every reference records its swap-in/swap-out traffic (`swap_in_gib`, `swap_out_gib`) from `/proc/vmstat`, as an NVMe-wear figure. Exact-mode rows also carry `xeb_linear`, `hog_linear` and `stats_mode`.

### 6.4 `nn_qab32plus.sh`

Target host: 96-thread EPYC, six Radeon Pro V340 dies (8 GB each, two per card on a shared PCIe 3.0 x8 link) through Mesa rusticl, 320 GB RAM, swap on an 8-drive NVMe RAID0. Series add `30 2 5`, `32 3 4`, `33 5 3`, `34 8 2`, `35 3 5`, `36 2 6` to the six above, adding equal-ratio groups (2.0: 30/36; 2.5: 14/28/35; 4.5: 22/33). Reference profiles by width:

| Width | Profile |
|---|---|
| ≤ `SMALL_MAX` (28) | one reference worker per die in `IDEAL_DEVICES` (default `1,2,3,4,5`), ACE on `ACE_DEVICES` (default `0`) |
| `SMALL_MAX` < w ≤ `PAGER_MAX` (30), `PAGER_DEVICES` set | one GPU simulator paged across those dies (opt-in; time one seed first) |
| otherwise | one worker on Qrack's CPU engine (`BIG_ENGINE`) |

| Width | State vector | + probabilities (exact) | Lives in |
|---|---|---|---|
| 30 | 8 GiB | 4 GiB | RAM |
| 32 | 32 GiB | 16 GiB | RAM |
| 34 | 128 GiB | 64 GiB | RAM |
| 35 | 256 GiB | 128 GiB | state in RAM, probabilities partly in swap |
| 36 | 512 GiB | 256 GiB | mostly swap |

Further variables: `ALLOW_SWAP` (1), `STATS` (`auto`), `SDRP` and `SDRP_FROM` (36), `IDEAL_ENGINE`, `BIG_ENGINE`. **`SCRIPT` defaults to `./nn_qab.py`, which lacks `--stats`/`--allow-swap`; the script's preflight stops with an error unless `SCRIPT=./nn_qab32plus.py` is set.** Run one seed at width 36 and compare `swap_out_gib` against the drives' rated endurance before launching more.

### 6.5 `nn_qab-adversarial.py`

```
nn_qab-adversarial.py bench WIDTH DEPTH [LRC] [LRR] [--reps N] [--bases zxy] [--shots N] [--seed N]
                            [--consensus N] [--sever-mode correct|edetect|both] [--json OUT]
nn_qab-adversarial.py run   --runs R --reps N WIDTH DEPTH [LRC] [LRR] [bench flags...]
nn_qab-adversarial.py pool  FILES... [--metric M] [--arm-a ace] [--arm-b severed] [--csv OUT]
```

| Metric | Definition / reading |
|---|---|
| `xeb` | the estimator of Sec. 2.3 |
| `xeb_google` | 2ⁿ Σ p q − 1, comparable to the literature [[2]](#ref-2) |
| `xeb_tail` | `xeb` restricted to bitstrings with p below the median; head-tilted spoofers have no tail structure |
| `hog` | heavy-output probability [[3]](#ref-3) |
| `hellinger`, `tvd` | classical fidelity Σ√(pq) and total variation; shot-biased, read relative to the `ceiling` arm |
| `zz_err_seam`, `zz_err_bulk` | mean \|⟨Z_iZ_j⟩_ace − ⟨Z_iZ_j⟩_ideal\| over coupled pairs touching a seam / inside a patch; if seam ≈ bulk, the severed-circuit picture does not apply |
| `z_err_seam`, `z_err_bulk` | single-qubit analogues |

`--sever-mode` selects which repair mechanism the `severed` arm removes: `correct` (no-op `_correct()`, seam reconciliation only), `edetect` (`is_error_detection=False`, the detect-and-post-select gadget), or `both`. Use more than one: if `severed` tracks `ace` under one mode, that mode may simply not be the one carrying the load. `--consensus N` scores N index-shifted instances of the circuit individually. `run` launches R independently seeded `bench` processes and pools them (≥ 3 runs before between-run heterogeneity is meaningful); quote the random-effects line.

### 6.6 `nn_qab_graph.py`

Reads `runs/w*.csv` and plots XEB (or HOG, or their linear variants) against B-to-B (x) and width (y), with per-series means and error bars, equal-ratio connectors and line (XEB ~ ratio) and plane (XEB ~ ratio + width) fits with leave-one-out cross-validation. Width is drawn as its own axis because in the original five series corr(ratio, width) = 0.84, so a two-dimensional XEB-versus-ratio plot cannot separate the two; the 30–36 extension lowers that correlation to ≈ 0.28. Series scored with the linear estimator are drawn hollow. Writes `<prefix>-runs.csv` (every run, tidy long form) and `<prefix>-summary.csv` (per series: n, mean, sd, se, 95 % CI, residual against each model). Flags: `--runs`, `--prefix`, `--show`, `--metric xeb|hog|xeb_linear|hog_linear`, `--min-n`, `--elev`, `--azim`, `--dpi`.

---

## 7. Outputs

Per seed, `<out>/ace/<seed>.json` holds geometry (`width`, `depth`, `long_range_columns`, `long_range_rows`, `boundary_qubits`, `bulk_qubits`, `bulk_to_boundary`), swap mode, shots, timing, device, the counts and the serialised circuit; `<out>/xeb/<seed>.json` adds `xeb_ace`, `hog_ace`, reference timing, engine, SDRP and `unitary_fidelity`, library and device provenance (and, for `nn_qab32plus.py`, `stats_mode`, `xeb_linear`, `hog_linear`, `swap_in_gib`, `swap_out_gib`). `merge` writes one CSV per series; the sweep scripts collect `summary.tsv` (`width, lrc, lrr, B_to_B, n, mean_xeb, stdev_xeb`).

---

## 8. Results to date

Mean and between-seed standard deviation of the XEB at depth 12, 100 seeds per series, from the accompanying analysis sheet (geometry read from the backend):

| Width | lrc | lrr | Boundary | Bulk | B-to-B | Patches | Mean XEB | SD |
|---|---|---|---|---|---|---|---|---|
| 18 | 1 | 3 | 9 | 9 | 1.0 | 3 | 0.0326 | 0.0838 |
| 10 | 2 | 2 | 4 | 6 | 1.5 | 2 | 0.1386 | 0.1403 |
| 27 | 2 | 3 | 9 | 18 | 2.0 | 3 | 0.1108 | 0.1006 |
| 14 | 3 | 2 | 4 | 10 | 2.5 | 2 | 0.1490 | 0.0977 |
| 27 | 4 | 3 | 6 | 21 | 3.5 | 2 | 0.1850 | 0.0735 |
| 22 | 5 | 2 | 4 | 18 | 4.5 | 2 | 0.2322 | 0.1088 |
| 26 | 6 | 2 | 4 | 22 | 5.5 | 2 | 0.2408 | 0.0757 |

The mean XEB increases with the bulk-to-boundary geometry across widths, and the two 27-qubit series isolate the effect at fixed width: the two-patch configuration (B-to-B 3.5) scores 0.185 against 0.111 for the three-patch one (B-to-B 2.0). In the sheet, atanh(XEB) regressed linearly on a composite geometric regressor built from the B-to-B ratio and the patch structure gives an in-sample R² of 0.845; leave-one-series-out cross-validation gives a mean training R² of 0.853 (CV-RMSE 0.039) and leave-two-out 0.872 (CV-RMSE 0.041). With seven series these fits establish a direction and rough magnitude, not a functional form. The between-seed standard deviations (0.07–0.14) are comparable to the means: with 100 seeds the standard error of each series mean is ≈ 0.01, with fewer the series are not separable.

**Relation to `rcs/nighthawk/`.** On the 61-qubit IBM Nighthawk circuits (Haar single-qubit gates, CZ couplers, flat register), the same estimator was applied to ACE with the logical qubits placed on ACE sites as compact tiles. Counting the B-to-B ratio over the sites the circuit actually occupies (bulk qubits used / seam qubits used), the n = 27, d = 12 XEB fell monotonically with that effective ratio across five layouts (8.0 → 0.182, 8.0 → 0.153, 5.8 → 0.136, 3.5 → 0.109, 0.7 → 0.091; three instances each), the same direction as the series above, on a harder circuit ensemble and without the torus.

---

## 9. Practical notes

- **Pin the device.** On the development host, letting Qrack spread one reference across all detected cards turned an ≈ 18 s run into ≈ 227 s by paging the state vector over a PCIe 1.0 x4 link; this is why everything defaults to device 0.
- **Two passes, two kinds of hardware.** ACE's cost is Python-side seam traffic; the reference's cost is the dense state. Keep them on different resources.
- **Seeds, not repetitions, are the unit** in the sweeps; **runs, not repetitions,** in the adversarial harness (Sec. 2.5).
- **Exact and linear XEB are different estimators.** Compare across the width at which a series falls back only through the overlapping widths where both are recorded.
- **Swap** keeps the widest references alive but rewrites most of a swapped state vector on every two-qubit gate pass; account for drive endurance.

---

## 10. Limitations

- The single-qubit ensemble is not Haar (Sec. 2.1), the register is a torus, and the two-qubit gate is drawn from twelve types; results do not transfer quantitatively to Haar/CZ circuits on a flat lattice without re-measurement (see `rcs/nighthawk/`).
- 1024 shots per circuit at every width; the XEB's shot noise is small next to the between-seed spread, but tail metrics (`xeb_tail`) and correlators are noisier.
- `QrackAceBackend` layout, seam treatment and error-detection gadget depend on the PyQrack version; regenerate whole series under one version (the purpose of the sweep scripts) and record it.
- Seven geometric configurations constrain a monotone trend, not a model; width and ratio remain partly confounded until the 30–36 series are complete.
- SDRP references are approximate by construction and must be read with their reported fidelity.

---

## 11. References

All identifiers were checked against the arXiv abstract pages and, where given, the published versions.

<a id="ref-1"></a>**[1]** S. Boixo *et al.*, "Characterizing Quantum Supremacy in Near-Term Devices," [arXiv:1608.00263](https://arxiv.org/abs/1608.00263); *Nature Physics* **14**, 595 (2018). — *Cross-entropy benchmarking as a fidelity proxy.*

<a id="ref-2"></a>**[2]** F. Arute *et al.*, "Quantum supremacy using a programmable superconducting processor," [arXiv:1910.11333](https://arxiv.org/abs/1910.11333); *Nature* **574**, 505 (2019). — *Linear XEB at 53 qubits; the `ABCDCDAB` coupler pattern.*

<a id="ref-3"></a>**[3]** S. Aaronson, L. Chen, "Complexity-Theoretic Foundations of Quantum Supremacy Experiments," [arXiv:1612.05903](https://arxiv.org/abs/1612.05903); *32nd Computational Complexity Conference (CCC 2017)*, LIPIcs **79**, 22, [doi:10.4230/LIPIcs.CCC.2017.22](https://doi.org/10.4230/LIPIcs.CCC.2017.22). — *Heavy-output generation.*

<a id="ref-4"></a>**[4]** X. Gao, M. Kalinowski, C.-N. Chou, M. D. Lukin, B. Barak, S. Choi, "Limitations of Linear Cross-Entropy as a Measure for Quantum Advantage," [arXiv:2112.01657](https://arxiv.org/abs/2112.01657); *PRX Quantum* **5**, 010334 (2024). — *XEB spoofing by cutting weak links; the reference point for the adversarial controls.*

<a id="ref-5"></a>**[5]** A. Morvan *et al.*, "Phase transitions in random circuit sampling," [arXiv:2304.11119](https://arxiv.org/abs/2304.11119); *Nature* **634**, 328 (2024). — *When XEB tracks fidelity.*

<a id="ref-6"></a>**[6]** D. Strano, B. Bollay, A. Blaauw, N. Shammah, W. J. Zeng, A. Mari, "Exact and approximate simulation of large quantum circuits on a single GPU," [arXiv:2304.14969](https://arxiv.org/abs/2304.14969); *Proc. IEEE QCE 2023*, pp. 949–958. — *Qrack's exact and approximate simulation methods.*

<a id="ref-7"></a>**[7]** R. DerSimonian, N. Laird, "Meta-analysis in clinical trials," *Controlled Clinical Trials* **7**, 177–188 (1986), [doi:10.1016/0197-2456(86)90046-2](https://doi.org/10.1016/0197-2456(86)90046-2). — *Random-effects pooling with a moment estimate of the between-study variance.*

<a id="ref-8"></a>**[8]** J. P. T. Higgins, S. G. Thompson, "Quantifying heterogeneity in a meta-analysis," *Statistics in Medicine* **21**, 1539–1558 (2002), [doi:10.1002/sim.1186](https://doi.org/10.1002/sim.1186). — *The I² heterogeneity statistic.*

**Software:** [vm6502q/pyqrack-examples](https://github.com/vm6502q/pyqrack-examples) (upstream `rcs/nn_qab.py`) · [unitaryfund/pyqrack](https://github.com/unitaryfund/pyqrack) · [twobombs/thereminq-examples](https://github.com/twobombs/thereminq-examples)
