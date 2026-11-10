# SDRP-Parameterized Zero-Noise Extrapolation (`sdrp/`)

## Overview

This directory implements a Zero-Noise Extrapolation (ZNE) verification pipeline ([`sdrp_zne_batch.py`](./sdrp_zne_batch.py)) that treats PyQrack's **Schmidt-Decomposition Rounding Parameter (SDRP)** as a continuous, controllable approximate-simulation "noise knob" within `QrackAceBackend` (Automatic Circuit Elision) [[3]](#ref-3). Using Mitiq's classical inference factories [[1]](#ref-1), [[2]](#ref-2), the pipeline evaluates whether series-mean linear Cross-Entropy Benchmarking (XEB) fidelities measured across finite SDRP truncation thresholds can be extrapolated back to the zero-truncation limit ($\text{SDRP} \to 0$).

---

## Methodology

### 1. Circuit Ensemble & Patch Geometry Series
The circuit generator constructs nearest-neighbor coupler Random Circuit Sampling (RCS) instances on rectangular lattices of width $w$ factored via `factor_width(w)`. Each layer applies random single-qubit rotations $U(\theta, \phi, \lambda)$ with uniform angles in $[0, 2\pi)$ followed by two-qubit entangling gates drawn uniformly from a 12-gate set (`swap`, `pswap`, `mswap`, `nswap`, `iswap`, `iiswap`, `cx`, `cy`, `cz`, `acx`, `acy`, `acz`) following the eight-step coupler activation sequence `[0, 3, 2, 1, 2, 1, 0, 3]`.

Five geometric patch series are defined via `QrackAceBackend(width, long_range_columns=lrc, long_range_rows=lrr)` with bulk-to-boundary proxy ratio $\text{ratio} = 2 \cdot \text{lrc}$:

| Series Width ($w$) | `long_range_columns` (`lrc`) | `long_range_rows` (`lrr`) | Bulk-to-Boundary Proxy Ratio |
| :---: | :---: | :---: | :---: |
| 10 | 2 | 2 | 4 |
| 14 | 3 | 2 | 6 |
| 27 | 4 | 3 | 8 |
| 22 | 5 | 2 | 10 |
| 26 | 6 | 2 | 12 |

### 2. Paired-Ensemble SDRP Scaling & ZNE Inference
To eliminate circuit-to-circuit variance across truncation levels and smooth the single-circuit step-function ("staircase") response of Schmidt-decomposition rounding, the **identical seeded ensemble** of $N_{\text{circuits}}$ random circuits is replayed at every SDRP scale factor:
*   **Base ("Golden") SDRP Threshold:**
    $$g = \frac{1 - 1/\sqrt{2}}{2} \approx 0.146447$$
*   **Default Scale Factors:** $s \in \{1/3, 2/3, 1.0\}$, corresponding to $\text{SDRP} \in \{0.048816, 0.097631, 0.146447\}$.
*   **Observable (Shot-Based Linear XEB):** For each circuit and SDRP level, `shots` bitstrings $\{x_i\}$ are drawn from `QrackAceBackend` and scored sparsely against an exact $\text{SDRP} = 0$ `QrackSimulator` reference using `prob_perm` [[4]](#ref-4), [[5]](#ref-5):
    $$F_{\text{XEB}} = 2^w \left( \frac{1}{N_{\text{shots}}} \sum_{i=1}^{N_{\text{shots}}} P_{\text{ideal}}(x_i) \right) - 1$$
*   **Zero-Truncation Extrapolation:** The ensemble-mean XEB at each scale factor is passed to Mitiq's `LinearFactory`, `RichardsonFactory`, and `ExpFactory` via `run_classical` to extrapolate $\hat{F}_{\text{XEB}}(\text{SDRP} \to 0)$, which is directly benchmarked against an exact $\text{SDRP} = 0$ ACE validation pass (`with_exact=True`).

---

## Usage

```bash
# Run default sweep across all five series (widths 10, 14, 27, 22, 26)
python3 sdrp_zne_batch.py

# Custom series subset, circuit count, depth, and CPU execution
python3 sdrp_zne_batch.py --series 10,14,22 --circuits 100 --depth 12 --shots 1024 --seed 42 --cpu --out sdrp_zne
```

### CLI Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--series` | `"10,14,27,22,26"` | Comma-separated widths from `SERIES_TABLE` |
| `--circuits` | `100` | Number of seeded RCS circuits per series |
| `--depth` | `12` | Circuit depth (layers of single-qubit and two-qubit gates) |
| `--shots` | `1024` | Measurement shots per circuit per SDRP point |
| `--seed` | `42` | Base RNG seed for deterministic paired-ensemble generation |
| `--scales` | `"0.333333,0.666667,1.0"` | Scale factors applied to `--base-sdrp` |
| `--base-sdrp` | `0.146447` | Base ("golden") SDRP parameter $g = (1 - 1/\sqrt{2})/2$ |
| `--cpu` | `False` | Run `QrackAceBackend` with `is_gpu=False` |
| `--no-exact` | `False` | Skip the direct `sdrp=0` validation column |
| `--out` | `"sdrp_zne"` | Output prefix for generated CSV files |

### Outputs
*   Writes per-series CSV logs (`<out>_w<width>_r<ratio>.csv`) containing per-circuit linear XEB across all SDRP scale points and the `sdrp=0 (exact)` validation column.
*   Prints series summary statistics (mean, median, standard error of the mean) and ZNE estimates with absolute error against the `sdrp=0` target.

---

## Core Quantum Mechanical Concepts & Verified ArXiv References

* <a id="ref-1"></a>**[1]** K. Temme, S. Bravyi, and J. M. Gambetta, "Error mitigation for short-depth quantum circuits," *Physical Review Letters* **119**, 180509 (2017), [arXiv:1612.02058](https://arxiv.org/abs/1612.02058).
* <a id="ref-2"></a>**[2]** R. LaRose, A. Mari, S. Kaiser *et al.*, "Mitiq: A software package for error mitigation on noisy quantum computers," *Quantum* **6**, 774 (2022), [arXiv:2009.04417](https://arxiv.org/abs/2009.04417).
* <a id="ref-3"></a>**[3]** D. Strano, B. Bollay, A. Blaauw, N. Shammah, W. J. Zeng, and A. Mari, "Exact and approximate simulation of large quantum circuits on a single GPU," *Proceedings of the IEEE International Conference on Quantum Computing and Engineering (QCE)*, pp. 949–958 (2023), [arXiv:2304.14969](https://arxiv.org/abs/2304.14969).
* <a id="ref-4"></a>**[4]** S. Boixo, S. V. Isakov, V. N. Smelyanskiy *et al.*, "Characterizing Quantum Supremacy in Near-Term Devices," *Nature Physics* **14**, 595–600 (2018), [arXiv:1608.00263](https://arxiv.org/abs/1608.00263).
* <a id="ref-5"></a>**[5]** F. Arute *et al.*, "Quantum supremacy using a programmable superconducting processor," *Nature* **574**, 505–510 (2019), [arXiv:1910.11333](https://arxiv.org/abs/1910.11333).

