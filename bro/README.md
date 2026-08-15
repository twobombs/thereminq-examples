## Exhaustive Guide to the ThereminQ DCS Toolkit: Cross-Entropy Benchmarking and Verification

######  This document has been generated; claims and/or facts therein could be totally false - emptor caveat

## Abstract
This document provides a highly formalized, comprehensive guide to the ThereminQ Doped Clifford Sampling (DCS) toolkit, a collection of Python and Shell utilities designed for evaluating, manipulating, and visualizing sampled quantum circuits through Cross-Entropy Benchmarking (XEB). This guide details the methodology, inputs, commands, orchestration scripts, and interactive visualization mechanisms provided by the toolkit, strongly grounded in the theoretical framework of [arXiv:2607.25941](https://arxiv.org/abs/2607.25941).

## 1. Introduction
With the advent of quantum advantage and supremacy protocols utilizing random circuit sampling (RCS), tools capable of dissecting the fidelity of sampling runs have become indispensable. Cross-Entropy Benchmarking (XEB) is a prominent metric to gauge quantum processor health. It requires computing the ideal probabilities of sampled bitstrings, which are fundamentally distinct from the samples themselves.

The ThereminQ toolkit resolves the complexities of evaluating such metrics by providing a monolithic approach. The tool parses sampling data, evaluates theoretical distributions such as the Porter-Thomas distribution, performs temporal-boundary tensor network contractions, handles multi-GPU workload distributions, and offers interactive 3D viewers to explore metrics like Heavy Output Generation (HOG) and Linear XEB.

As outlined by Martiel et al., verifying these experiments typically suffers a scaling problem, where resolving fidelity requires strong classical simulation. This toolkit implements the verification of DCS circuits encoded in spacetime codes, utilizing code-preserving $T$-gate doping to evaluate simulation hardness while retaining error-detection properties.

## 2. Methodology
The underlying methodology of this toolkit relies on three key estimators under a global depolarising error model. The sampling distribution is postulated as $F p_{ideal} + \frac{1-F}{D}$, where $F$ is the fidelity and $D$ is the Hilbert space dimension $2^N$.
- **Linear XEB**: $F = E[D \cdot p] - 1$
- **Logarithmic XEB**: $F = E[\ln(D \cdot p)] + \gamma$ (where $\gamma \approx 0.577$ is the Euler-Mascheroni constant).
- **Heavy Output Generation (HOG)**: $F = \frac{\text{HOG} - 1/2}{(\ln 2) / 2}$

Validating agreement between these estimators confirms the viability of a depolarising noise model, whereas divergence can signal coherent or leakage errors.

Furthermore, the toolkit includes mechanisms to bound the state fidelity after $T$-doping, isolating the fidelity drop to the conversion of previously-harmless faults. It features tools to evaluate the *elision fidelity* ($F_{elide}$), comparing patched amplitude probabilities against exact monolithic statevector probabilities.

## 3. Inputs
The toolkit requires outputs from sampling computations, primarily a JSON file describing the samples and optionally a NumPy (`.npy`) file describing ideal amplitudes.
- **`.npy` arrays**: Can be paired as `<prefix>_xeb_amplitudes.npy` (shape $(S,)$) and `<prefix>_xeb_bitstrings.npy` (shape $(S, N)$).
- **`.json` file**: The structure usually resembles `{"accepted_samples": [[0,1,...], ...], "ideal_probs": [...]}`. If probabilities are absent, the viewer restricts fidelity readouts.

## 4. Python Toolkit Components

### 4.1. `bro-xeb-viz.py`
A monolithic toolkit for manipulating and visualizing DCS circuits.
- **`view`**: Interactive 3D visualization.
- **`stats`**: Prints the fidelity table and health statistics.
- **`tcount`**: Parses a given QASM file and counts non-Clifford gates, aiding feasibility checks for strong simulation.
- **`probs`**: Uses PyQrack to strong-simulate ideal probabilities required for XEB.
- **Additional Subcommands**: `noisy`, `ladder`, `dedope`, `compare`, `structure`, `cut`, `patch`, `probe`, `selftest`, `doctor`, `env`, `budget`, `sample`, `idealsample`, `synth`.

### 4.2. `dcs_post_select_paralel.py`
Performs post-selected weak simulation of a DCS circuit on `QrackStabilizer` using parallel processes.
- **Purpose**: Generates samples from an empty tableau, independently evaluating the stochastic rounding decisions of $T$-gates. The circuit is code-preserving, meaning $T$-gates commute with checks, and acceptance should approach 100% in a noiseless simulated environment.
- **Concurrency**: Employs process-level parallelism (forking) over threading to prevent allocator contention within the `QrackStabilizer` library.

### 4.3. `experiment2-xeb-dash-UI.py`
An interactive 3D viewer for cross-entropy benchmarking sample dumps. Reads `.npy` pairs to plot the S shots embedded in 3D by PCA of the centred bit matrix, or via other mappings, enabling real-time fidelity inspection and theoretical curve overlaying.

### 4.4. `tnsweep.py`
Implements deterministic temporal-boundary contraction for 1D brickwork circuits, following the methodology of Manabe, Gu, and Pan (arXiv:2608.13110).
- **Methodology**: Instead of evolving a $2^N$ state vector forward in time (costing $O(2^{N+T})$), it sweeps a boundary state along the *spatial* direction. Because CZ gates have operator Schmidt rank 2, a spatial cut bounds the rank to $\lceil d/2 \rceil$, where $d$ is depth.
- **Usage**: Used to verify exact amplitudes, evaluate width/time scaling, and emit amplitude arrays for dense reference verification, fundamentally exploiting the circuit's open quasi-one-dimensional geometry.

## 5. Shell Automation Scripts

The `bro/scripts/` directory orchestrates large-scale evaluations, building upon the Python layer.

- **`check.sh`**: Validates QASM files by verifying the maximum gate-referenced qubit index against the declared register size.
- **`genrungs.sh`**: Generates a doping "ladder" (nested subsets of $T$-gate locations). Ensures that doping level $t$ is the only variable, isolating drift in $F_{elide}$ from doping placement variances. It strictly verifies graph topologies and bounds runs by memory budgets (e.g., stopping when exact references exceed $2^{27}$).
- **`felide.sh`**: Evaluates the elision fidelity $F_{elide}$ across the generated circuit family. It compares patched probabilities against exact probabilities using the *same* sample sets, ensuring that the underlying $F_{true}$ cancels out. It is the primary heuristic for deciding whether patched evaluations extrapolate successfully to full undoped limits.
- **`sweep.sh` / `synth_sweep.sh`**: Measures $F_{elide}$ concurrently for both monolithic exact arms and patched arms across varied $T$-counts.
- **`tnvalidate.sh`**: Proves that the temporal-boundary contraction path in `tnsweep.py` scales correctly with depth and produces exact amplitudes compared against a dense statevector reference before committing large HPC clusters.
- **`full_run.sh` / `ladder.sh` / `time.sh` / `trend.sh`**: Helper pipelines to trigger the ladder generation, amplitude fetch profiling, and patching trend evaluations.

## 6. OpenCL and PyQrack Multi-GPU Environment Configuration

For massive scale validation runs distributed across GPUs, PyQrack relies on explicit environment variables to constrain and target device contexts.
- **`QRACK_QPAGER_DEVICES`**: Defines the target indices for the QPager framework to enforce distributed boundary memory limits.
- **`QRACK_QUNITMULTI_DEVICES`**: Directs Qrack's multi-device orchestrator, bounding exactly which OpenCL accelerators form the unified compute fabric for a given worker.
These are typically populated inside the orchestration shell scripts (or inherited from ThereminQ's container definitions) to strictly enforce GPU isolation for concurrent ladder evaluations.

## 7. Interactive Visualization
The `view` subcommand summons a matplotlib-based 3D viewer featuring several rendering modes to dissect the sample.
- **Lattice**: Embeds qubits on an $n_x \times n_y \times n_z$ grid, coloured by either the $P(1)$ marginal probability or by the specific bit pattern of a selected shot.
- **Cloud**: Plots samples embedded by PCA of the centred bit matrix, illuminating potential deviations from the isotropic null.
- **Corr**: Generates a connected $\langle Z_i Z_j \rangle - \langle Z_i \rangle\langle Z_j \rangle$ surface mesh to showcase pair correlations against shot-noise boundaries.
- **Porter**: Displays the Porter-Thomas histogram of $x = D \cdot p$ and overlays the theoretical depolarising model (or Binomial distribution if run in samples-only mode).
- **Landscape**: Constructs a joint 2D histogram plotting Hamming weight against $\log_{10}(D \cdot p)$.

Controls are exposed via matplotlib widgets, allowing users to select shots, swap views, and toggle theoretical overlays interactively.

## 8. References
1. Arute, F., et al. "Quantum supremacy using a programmable superconducting processor." Nature 574.7779 (2019): 505-510.
2. Barak, B., Chou, C., and Gao, X. "Spoofing linear cross-entropy benchmarking in shallow quantum circuits." [arXiv:2005.02421](https://arxiv.org/abs/2005.02421) (2020).
3. Gao, X., Kalinowski, M., Chou, C., Lukin, M. D., Barak, B., and Choi, S. "Limitations of linear cross-entropy as a measure of quantum advantage." PRX Quantum 5 (2024): 010334.
4. Ware, B., Deshpande, A., Hangleiter, D., Niroula, P., Fefferman, B., Gorshkov, A. V., and Gullans, M. J. "A sharp phase transition in linear cross-entropy benchmarking." [arXiv:2305.04954](https://arxiv.org/abs/2305.04954) (2023).
5. Martiel, S., Chung, J., Seif, A., Ghosh, S., Hincks, I., Deshpande, A., Fefferman, B., Gambetta, J. M., and Javadi-Abhari, A. "Sampling hard circuits with verifiably high fidelity." [arXiv:2607.25941](https://arxiv.org/abs/2607.25941) (2026).
6. Manabe, Gu, and Pan. "Deterministic temporal-boundary contraction." [arXiv:2608.13110](https://arxiv.org/abs/2608.13110) (2026).

-----------------------------

From and for the 'Trust me, BRO' - paper

https://zenodo.org/records/21633064

-------------------------------

<img width="4290" height="2307" alt="xeb3d_landscape" src="https://github.com/user-attachments/assets/3773cb89-b899-46bd-b4ea-9b523dda637e" />
<img width="4335" height="2338" alt="xeb3d_corr" src="https://github.com/user-attachments/assets/bfcb247c-3e5e-4d38-848a-ad82ff347249" />
<img width="4206" height="2307" alt="xeb3d_porter" src="https://github.com/user-attachments/assets/8d027464-5b33-4be2-bb64-4fba1fabd331" />
<img width="4319" height="2338" alt="xeb3d_cloud" src="https://github.com/user-attachments/assets/a69274e4-fab5-4b44-ac74-59aba7f37e56" />
<img width="4314" height="2307" alt="xeb3d_lattice" src="https://github.com/user-attachments/assets/0ba5cc3b-304f-42a3-ae5e-92e98525f5b3" />