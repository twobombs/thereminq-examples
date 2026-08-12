## Exhaustive Guide to `bro-xeb-viz.py`: Cross-Entropy Benchmarking Toolkit

######  This document has been generated; claims and/or facts therein could be totally false - emptor caveat

## Abstract
This document provides a highly formalized, comprehensive guide to the `bro-xeb-viz.py` utility, a one-file toolkit designed for evaluating, manipulating, and visualizing sampled quantum circuits through Cross-Entropy Benchmarking (XEB). This guide details the methodology, inputs, commands, and interactive visualization mechanisms provided by the script.

## 1. Introduction
With the advent of quantum advantage and supremacy protocols utilizing random circuit sampling (RCS), tools capable of dissecting the fidelity of sampling runs have become indispensable. Cross-Entropy Benchmarking (XEB) is a prominent metric to gauge quantum processor health. It requires computing the ideal probabilities of sampled bitstrings, which are fundamentally distinct from the samples themselves.

The `bro-xeb-viz.py` toolkit resolves the complexities of evaluating such metrics by providing a monolithic approach. The tool parses sampling data, evaluates theoretical distributions such as the Porter-Thomas distribution, and offers an interactive 3D viewer to explore metrics like Heavy Output Generation (HOG) and Linear XEB.

## 2. Methodology
The underlying methodology of `bro-xeb-viz.py` relies on three key estimators under a global depolarising error model. The sampling distribution is postulated as $F p_{ideal} + \frac{1-F}{D}$, where $F$ is the fidelity and $D$ is the Hilbert space dimension $2^N$.
- **Linear XEB**: $F = E[D \cdot p] - 1$
- **Logarithmic XEB**: $F = E[\ln(D \cdot p)] + \gamma$ (where $\gamma \approx 0.577$ is the Euler-Mascheroni constant).
- **Heavy Output Generation (HOG)**: $F = \frac{\text{HOG} - 1/2}{(\ln 2) / 2}$

Validating agreement between these estimators confirms the viability of a depolarising noise model, whereas divergence can signal coherent or leakage errors.

## 3. Inputs
The toolkit requires outputs from sampling computations, primarily a JSON file describing the samples and optionally a NumPy (`.npy`) file describing ideal amplitudes.
- **`.npy` arrays**: Can be paired as `<prefix>_xeb_amplitudes.npy` (shape $(S,)$) and `<prefix>_xeb_bitstrings.npy` (shape $(S, N)$).
- **`.json` file**: The structure usually resembles `{"accepted_samples": [[0,1,...], ...], "ideal_probs": [...]}`. If probabilities are absent, the viewer restricts fidelity readouts.

## 4. Commands and Usage
The script is invoked via `python3 bro-xeb-viz.py [subcommand] [arguments]`.

### 4.1. `view`
**Purpose**: Interactive 3D visualization. If no subcommand is specified, the toolkit defaults to `view`.
- **Usage**:
  - `python3 bro-xeb-viz.py view result.json` (Samples only)
  - `python3 bro-xeb-viz.py view result.json --amps amps.npy`
  - `python3 bro-xeb-viz.py view --save-all out/` (Headless rendering to PNG)

### 4.2. `stats`
**Purpose**: Prints the fidelity table and health statistics of the sample.
- **Usage**: `python3 bro-xeb-viz.py stats result.json`

### 4.3. `tcount`
**Purpose**: Parses a given QASM file and counts non-Clifford gates, aiding feasibility checks for strong simulation.
- **Usage**: `python3 bro-xeb-viz.py tcount circuit.qasm`

### 4.4. `probs`
**Purpose**: Uses PyQrack to strong-simulate ideal probabilities required for XEB, given a QASM file and sample outputs.
- **Usage**: `python3 bro-xeb-viz.py probs circuit.qasm result.json --limit 1`

### 4.5. Additional Tools
- `noisy`: Simulates post-selected sampling on ACE with a given noise parameter.
- `ladder`: Sweeps a set of circuits and tabulates their respective fidelities.
- `dedope`: Reduces non-Clifford gate doping to construct intermediate ladder rungs.
- `compare`: Performs a structural, layer-by-layer comparison between two QASM files.
- `structure`: Provides a component decomposition report for the interaction graph of a QASM circuit.
- `cut`: Evaluates the interaction graph to find patch cuts, facilitating modular circuit analysis.
- `patch`: Emits patched sub-circuits derived from calculated interaction graph cuts.
- `probe`: Diagnoses and identifies a PyQrack configuration capable of supporting the chain rule for exact probability evaluation.
- `selftest`: Validates the stability and accuracy of the implemented chain rule against a dense statevector reference.
- `doctor`: Reports the installed PyQrack build configuration and systematically determines the architectural width at which simulation constraints induce failure.

## 5. Interactive Visualization
The `view` subcommand summons a matplotlib-based 3D viewer featuring several rendering modes to dissect the sample.
- **Lattice**: Embeds qubits on an $n_x \times n_y \times n_z$ grid, coloured by either the $P(1)$ marginal probability or by the specific bit pattern of a selected shot.
- **Cloud**: Plots samples embedded by PCA of the centred bit matrix, illuminating potential deviations from the isotropic null.
- **Corr**: Generates a connected $\langle Z_i Z_j \rangle - \langle Z_i \rangle\langle Z_j \rangle$ surface mesh to showcase pair correlations against shot-noise boundaries.
- **Porter**: Displays the Porter-Thomas histogram of $x = D \cdot p$ and overlays the theoretical depolarising model (or Binomial distribution if run in samples-only mode).
- **Landscape**: Constructs a joint 2D histogram plotting Hamming weight against $\log_{10}(D \cdot p)$.

Controls are exposed via matplotlib widgets, allowing users to select shots, swap views, and toggle theoretical overlays interactively.

## 6. References
1. Arute, F., et al. "Quantum supremacy using a programmable superconducting processor." Nature 574.7779 (2019): 505-510.
2. Aaronson, S., and Gunn, S. "On the Classical Hardness of Spoofing Linear Cross-Entropy Benchmarking." arXiv preprint arXiv:1910.12085 (2019).
3. Zlokapa, A., Boixo, S., and Lidar, D. "Boundaries of quantum supremacy via random circuit sampling." arXiv preprint arXiv:2005.02464 (2020).
4. Pan, F., and Zhang, P. "Simulating the Sycamore quantum supremacy circuits." arXiv preprint arXiv:2103.03074 (2021).


-----------------------------


From and for the 'Trust me, BRO' - paper

https://zenodo.org/records/21633064

-------------------------------

<img width="4290" height="2307" alt="xeb3d_landscape" src="https://github.com/user-attachments/assets/3773cb89-b899-46bd-b4ea-9b523dda637e" />
<img width="4335" height="2338" alt="xeb3d_corr" src="https://github.com/user-attachments/assets/bfcb247c-3e5e-4d38-848a-ad82ff347249" />
<img width="4206" height="2307" alt="xeb3d_porter" src="https://github.com/user-attachments/assets/8d027464-5b33-4be2-bb64-4fba1fabd331" />
<img width="4319" height="2338" alt="xeb3d_cloud" src="https://github.com/user-attachments/assets/a69274e4-fab5-4b44-ac74-59aba7f37e56" />
<img width="4314" height="2307" alt="xeb3d_lattice" src="https://github.com/user-attachments/assets/0ba5cc3b-304f-42a3-ae5e-92e98525f5b3" />
