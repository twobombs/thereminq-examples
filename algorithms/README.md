# Quantum Algorithms & Classical Simulation Methodologies (`algorithms/`)

## Abstract

The `algorithms/` domain of the **ThereminQ-examples** repository houses reference implementations, verification pipelines, and classical simulation methodologies spanning foundational quantum algorithms, variational hybrid architectures, holographic scrambling protocols, and high-performance random circuit sampling (RCS). Designed primarily around the GPU-accelerated **PyQrack** framework (`QrackSimulator`, `QrackAceBackend`, QrackNeuron layers and Quantum Binary Decision Diagrams) alongside interoperability layers for **Qiskit**, **PennyLane**, **Cirq**, **Quimb**, **OpenFermion** and **Mitiq**, these modules evaluate both exact statevector dynamics and controlled approximate simulation techniques, such as Automatic Circuit Elision (ACE) and Schmidt-Decomposition Rounding Parameter (SDRP) truncation, against analytical and exact numerical references.

---

## 1. Taxonomic Overview & Directory Index

The nineteen modules within `algorithms/` are organized into five thematic pillars according to their computational paradigm and physical domain:

| Directory | Algorithmic Domain | Primary Simulation Backend(s) | Key Theoretical & Verified References |
| :--- | :--- | :--- | :--- |
| [`er-epr/`](./er-epr/) | Holographic Traversable Wormhole Teleportation | `pyqrack` | [[1]](#ref-1) `1306.0533`, [[2]](#ref-2) `1608.05687`, [[3]](#ref-3) `1911.06314` |
| [`hhl/`](./hhl/) | Harrow–Hassidim–Lloyd Linear System Solver | `qiskit`, `qiskit_algorithms` | [[4]](#ref-4) `0811.3171`, [[5]](#ref-5) `1301.2340`, [[6]](#ref-6) `1802.08227` |
| [`noisy-shors/`](./noisy-shors/) | Integer Factorization in Resource-Scaled Regimes | `pyqrack` (optional, illustrative fallback) | [[7]](#ref-7) `quant-ph/9508027`, [[8]](#ref-8) `quant-ph/9605011`, [[9]](#ref-9) `1905.09749` |
| [`positron/`](./positron/) | Positronium Metrology via Unitary Inversion | `pyqrack` (Qiskit provider), `qiskit` | [[10]](#ref-10) `2506.04315` |
| [`qaoa/`](./qaoa/) | Adaptive Quantum Approximate Optimization (ADAPT-QAOA) | `qiskit.quantum_info` (density matrix), `scipy`, `networkx` | [[11]](#ref-11) `1411.4028`, [[12]](#ref-12) `2005.10258` |
| [`qec/`](./qec/) | Quantum Error Correction & Mitigation Wrappers | `pyqrack`, `mitiq` | [[13]](#ref-13) `1612.02058`, [[14]](#ref-14) `2009.04417` |
| [`qecho/`](./qecho/) | Out-of-Time-Order Correlators (OTOCs) & Loschmidt Echoes | `pyqrackising`, `pyqrack`, `vedo` | [[15]](#ref-15) `1503.01409`, [[16]](#ref-16) `1602.06271`, [[17]](#ref-17) `1806.02807`, [[18]](#ref-18) `2510.19550` |
| [`qft/`](./qft/) | Semiclassical Quantum Fourier Transform & QFT Condensates | `pyqrack`, `numpy` | [[19]](#ref-19) `quant-ph/9511007`, [[20]](#ref-20) `quant-ph/0001066`, [[21]](#ref-21) `quant-ph/0205095`, [[22]](#ref-22) `2304.14969` |
| [`qhrf/`](./qhrf/) | QHRF Coherence-Enhancement Models (exploratory) | `pyqrack`, `qiskit_aer` | [[8]](#ref-8) `quant-ph/9605011`, [[22]](#ref-22) `2304.14969` |
| [`qllm-audit/`](./qllm-audit/) | Quantum & Cryptographic Auditing of Large Language Models | `torch`, `scipy`, `qiskit` | [[7]](#ref-7) `quant-ph/9508027`, [[23]](#ref-23) `2204.06974`, [[24]](#ref-24) `2302.12173`, [[25]](#ref-25) `2306.05499` |
| [`qvml/`](./qvml/) | Quantum Variational Machine Learning & Tensor Contraction | `quimb`, `pyqrackising`, `pyqrack` (QrackNeuron), `vedo` | [[26]](#ref-26) `quant-ph/0511069`, [[27]](#ref-27) `1811.12926`, [[28]](#ref-28) `2002.01935` |
| [`qwscatter/`](./qwscatter/) | 1D & 2D Quantum Walk Scattering Dynamics | `pyqrack`, `qiskit_aer` | [[29]](#ref-29) `quant-ph/0303081`, [[30]](#ref-30) `0806.1972` |
| [`rcs/`](./rcs/) | Distributed Automatic Circuit Elision (ACE) for RCS | `pyqrack` (`QrackAceBackend`) | [[31]](#ref-31) `1608.00263`, [[32]](#ref-32) `1910.11333`, [[33]](#ref-33) `2005.02464`, [[34]](#ref-34) `2112.01657`, [[35]](#ref-35) `2211.03999`, [[36]](#ref-36) `2211.13267`, [[37]](#ref-37) `2304.11119`, [[38]](#ref-38) `2306.16659`, [[39]](#ref-39) `2406.18889` |
| [`rcsqbdd/`](./rcsqbdd/) | RCS via Quantum Binary Decision Diagrams & MPS | `pyqrack` (QBDD), `qiskit`, `quimb`, `torch` | [[40]](#ref-40) `1707.00865`, [[41]](#ref-41) `1905.08394`, [[42]](#ref-42) `2002.07730` |
| [`sdrp/`](./sdrp/) | SDRP-Parameterized Zero-Noise Extrapolation (ZNE) | `pyqrack` (`QrackAceBackend`), `mitiq` | [[13]](#ref-13) `1612.02058`, [[14]](#ref-14) `2009.04417`, [[22]](#ref-22) `2304.14969` |
| [`vqe/`](./vqe/) | Multi-Backend Variational Quantum Eigensolver | `pennylane` (`qrack.simulator`, `cirq.simulator`, `default.qubit`, IBM Quantum), `openfermion` | [[43]](#ref-43) `1304.3061`, [[44]](#ref-44) `1701.02691` |
| [`vqe-qml/`](./vqe-qml/) | VQE with Learnable Hermitian Observables for QML | `pyqrack`, `torch`, `qiskit` | [[45]](#ref-45) `2505.13525` |
| [`vqe-qml-dissipate/`](./vqe-qml-dissipate/) | Dissipative VQAs for Barren Plateau Avoidance | `qiskit` (`Estimator`), `qiskit_pyqrack` | [[46]](#ref-46) `1803.11173`, [[47]](#ref-47) `2007.14384`, [[48]](#ref-48) `2010.15968`, [[49]](#ref-49) `2507.02043` |
| [`vqls/`](./vqls/) | Variational Quantum Linear Solver for FEM/RCS Systems | `qiskit`, `qiskit_aer`, `qiskit_algorithms` | [[5]](#ref-5) `1301.2340`, [[50]](#ref-50) `1909.03898`, [[51]](#ref-51) `1909.05820`, [[52]](#ref-52) `2004.06252`, [[54]](#ref-54) EuRAD 2024 |

---

## 2. Detailed Module Specifications

### 2.1 Fault-Tolerant & Semiclassical Primitives

#### `qft/`: Semiclassical Quantum Fourier Transform & QFT Condensates
*   **Scripts:** [`qft-condensate-semiclassical.py`](./qft/qft-condensate-semiclassical.py), [`qft-cosmos-classical-condensate.py`](./qft/qft-cosmos-classical-condensate.py)
*   **Methodology:** Implements the observation of Griffiths and Niu [[19]](#ref-19) that a Quantum Fourier Transform (QFT) immediately preceding a computational-basis measurement can be decomposed into single-qubit measurements with classically fed-forward phase rotations.
    *   In `qft-condensate-semiclassical.py`, Shor's order-finding routine is executed using the single recycled control-qubit layout of Parker and Plenio [[20]](#ref-20) paired with an $n$-qubit work register in `QrackSimulator` [[22]](#ref-22), where modular multiplication $x \mapsto a^{2^j} x \bmod N$ is applied via pre-packed permutation tables (`QrackSimulator.hash()`) rather than gate-level arithmetic [[21]](#ref-21). Outputs are validated against exact classical order checks ($a^r \equiv 1 \pmod N$) and log-likelihood gap statistics.
    *   In `qft-cosmos-classical-condensate.py`, a ring of $W$ qubits partitioned into $p$-qubit patches separated by measure-and-reprepare classical seams is sampled in $O(W \cdot L \cdot \text{shots})$ time via exact single-qubit conditionals, with optional `QrackSimulator` spot-checks (`--check K`) and fault-injection diagnostics.
*   **Scope:** Neither program demonstrates a classical-simulation advantage; every output is checked against an exact reference (see [`qft/README.md`](./qft/README.md)).
*   **Verified References:** [[19]](#ref-19) [arXiv:quant-ph/9511007](https://arxiv.org/abs/quant-ph/9511007), [[20]](#ref-20) [arXiv:quant-ph/0001066](https://arxiv.org/abs/quant-ph/0001066), [[21]](#ref-21) [arXiv:quant-ph/0205095](https://arxiv.org/abs/quant-ph/0205095), [[22]](#ref-22) [arXiv:2304.14969](https://arxiv.org/abs/2304.14969).

#### `noisy-shors/`: Integer Factorization in Resource-Scaled Regimes
*   **Scripts:** [`noisy-big-shors.py`](./noisy-shors/noisy-big-shors.py), [`noisy-smol-shors.py`](./noisy-shors/noisy-smol-shors.py)
*   **Methodology:** Evaluates Shor's factoring algorithm [[7]](#ref-7) across small diagnostic composites (e.g., $N = 15$) and larger register widths using `QrackSimulator`. The controlled modular multiplier is built as an explicit permutation of controlled-X gates over all $N$ register states, which is the deliberate scaling bottleneck for large $N$; continued-fraction post-processing recovers the period. When PyQrack is not importable the quantum steps run as an illustrative classical fallback. No explicit noise model is defined in either script; the "noisy" label refers to the resource context of fault-tolerant thresholds [[8]](#ref-8) and physical resource estimates [[9]](#ref-9).
*   **Verified References:** [[7]](#ref-7) [arXiv:quant-ph/9508027](https://arxiv.org/abs/quant-ph/9508027), [[8]](#ref-8) [arXiv:quant-ph/9605011](https://arxiv.org/abs/quant-ph/9605011), [[9]](#ref-9) [arXiv:1905.09749](https://arxiv.org/abs/1905.09749).

#### `hhl/`: Harrow–Hassidim–Lloyd Linear System Solver
*   **Scripts:** [`hhl.py`](./hhl/hhl.py)
*   **Methodology:** Encapsulates the Harrow–Hassidim–Lloyd (HHL) algorithm [[4]](#ref-4) for preparing a quantum state $|x\rangle \propto A^{-1}|b\rangle$ corresponding to a Hermitian positive-definite system $Ax = b$. The implementation (`HHLProblem`) validates spectral positivity and Hermiticity ($A = A^\dagger$), constructs the quantum phase estimation and controlled-rotation circuit via `qiskit_algorithms.linear_solvers.hhl.HHL`, and benchmarks the normalized quantum statevector against the exact classical solution and its Euclidean norm $\|x\|_2$ [[6]](#ref-6). Includes a discussion of 1-bit Quantum Phase Estimation (QPE) for reduced resource requirements.
*   **Verified References:** [[4]](#ref-4) [arXiv:0811.3171](https://arxiv.org/abs/0811.3171), [[5]](#ref-5) [arXiv:1301.2340](https://arxiv.org/abs/1301.2340), [[6]](#ref-6) [arXiv:1802.08227](https://arxiv.org/abs/1802.08227).

---

### 2.2 Variational Quantum Algorithms & Quantum Machine Learning

#### `vqe/`: Multi-Backend Variational Quantum Eigensolver
*   **Scripts:**

| Script | Role | Execution target |
| :--- | :--- | :--- |
| [`import_clifford_vqe_min.py`](./vqe/import_clifford_vqe_min.py) | Hartree–Fock / Jordan–Wigner Hamiltonian preparation from the molecule CSV | `openfermion`, `openfermionpyscf` |
| [`import_clifford_vqe_entangled.py`](./vqe/import_clifford_vqe_entangled.py) | Entangled-ansatz VQE sweep over the molecule CSV | `pennylane`, `openfermion` |
| [`import_clifford_vqe_entangled-gpu-multi.py`](./vqe/import_clifford_vqe_entangled-gpu-multi.py) | Multi-process, multi-GPU variant of the sweep | `pennylane`, multiprocessing |
| [`import-clifford_vqe_entangled-mesa.py`](./vqe/import-clifford_vqe_entangled-mesa.py) | Multi-GPU variant refactored for PyQrack OpenCL via Mesa `rusticl` (RADV), with hardware polling, guarded CSV parsing and HF fallback | `pennylane`, OpenCL |
| [`run-multi-vqe-qrack-h2.py`](./vqe/run-multi-vqe-qrack-h2.py) | $\text{H}_2$ VQE benchmark | `qrack.simulator` |
| [`run-multi-vqe-pennylane-h2.py`](./vqe/run-multi-vqe-pennylane-h2.py) | $\text{H}_2$ VQE benchmark | PennyLane devices |
| [`run-multi-vqe-cirq-h2.py`](./vqe/run-multi-vqe-cirq-h2.py) | $\text{H}_2$ VQE benchmark | `cirq.simulator` |
| [`run-multi-vqe-ibmheron-h2.py`](./vqe/run-multi-vqe-ibmheron-h2.py) | $\text{H}_2$/6-31G (8 qubits) on real IBM Quantum hardware, compared with PySCF RHF | IBM Quantum (`ibm_brisbane`) |
| [`vqe-results-3dviz.py`](./vqe/vqe-results-3dviz.py) | 3D visualization of sweep results | `matplotlib` |
| [`import_clifford_vqe_entangled.sh`](./vqe/import_clifford_vqe_entangled.sh), [`import_clifford_vqe_min.csv`](./vqe/import_clifford_vqe_min.csv) | Launcher and molecule input table | |

*   **Methodology:** Implements the Variational Quantum Eigensolver (VQE) [[43]](#ref-43) to bound molecular ground-state energies $E_0 \le \langle \psi(\boldsymbol{\theta}) | \hat{H} | \psi(\boldsymbol{\theta}) \rangle$. Includes Jordan–Wigner fermionic-to-qubit mappings and Unitary Coupled Cluster Singles and Doubles (UCCSD) ansätze [[44]](#ref-44) for $\text{H}_2$ benchmarked against classical reference energies, as well as multi-process GPU-distributed sweeps over molecular datasets through the PennyLane-Qrack (`qrack.simulator`, OpenCL) device. The entangled sweeps derive from Dan Strano's `clifford_vqe_entangled.py` in `vm6502q/pyqrack-examples`.
*   **Verified References:** [[43]](#ref-43) [arXiv:1304.3061](https://arxiv.org/abs/1304.3061), [[44]](#ref-44) [arXiv:1701.02691](https://arxiv.org/abs/1701.02691), [[22]](#ref-22) [arXiv:2304.14969](https://arxiv.org/abs/2304.14969).

#### `vqe-qml/`: Learnable Hermitian Observables for Quantum Machine Learning
*   **Scripts:** [`hermitian-matrices-pyqrack.py`](./vqe-qml/hermitian-matrices-pyqrack.py), [`hermitian-matrices-pyqrack-oai.py`](./vqe-qml/hermitian-matrices-pyqrack-oai.py), [`hermitian-matrices-qiskit.py`](./vqe-qml/hermitian-matrices-qiskit.py), [`hermitian-matrices-pyqrack.sh`](./vqe-qml/hermitian-matrices-pyqrack.sh); paper copy [`2505.13525v1.pdf`](./vqe-qml/2505.13525v1.pdf)
*   **Methodology:** Implements the differentiable quantum measurement framework of Chen *et al.* [[45]](#ref-45), wherein both the variational circuit parameters $\boldsymbol{\theta}$ and the Hermitian measurement observable $\hat{M}(\boldsymbol{\phi}) = \hat{M}(\boldsymbol{\phi})^\dagger$ are co-optimized. In the PyTorch/PyQrack implementations, a classical controller network ("Fast Weight Programmer") dynamically parameterizes the unitary gates and the Hermitian readout matrix via a custom `torch.autograd.Function` backed by `QrackSimulator` (with a stub fallback when PyQrack is absent).
*   **Verified References:** [[45]](#ref-45) [arXiv:2505.13525](https://arxiv.org/abs/2505.13525), [[22]](#ref-22) [arXiv:2304.14969](https://arxiv.org/abs/2304.14969).

#### `vqe-qml-dissipate/`: Dissipative Variational Quantum Algorithms
*   **Scripts:** [`dissipate-ancilaries.py`](./vqe-qml-dissipate/dissipate-ancilaries.py)
*   **Methodology:** Investigates non-unitary, dissipative ansatz constructions designed to circumvent the exponential concentration of cost-function gradients known as the *barren plateau* phenomenon [[46]](#ref-46), [[47]](#ref-47), [[48]](#ref-48). Following Zapusek *et al.* [[49]](#ref-49), ancillary qubits are entangled with the system register and periodically reset mid-circuit to shed entropy and bound subsystem entanglement, with the aim of preserving polynomially scaling gradient variances $\text{Var}[\partial_k C]$ across increasing qubit widths. Expectation values are evaluated through Qiskit's `Estimator` on the `PyQrackSimulator` backend.
*   **Verified References:** [[46]](#ref-46) [arXiv:1803.11173](https://arxiv.org/abs/1803.11173), [[47]](#ref-47) [arXiv:2007.14384](https://arxiv.org/abs/2007.14384), [[48]](#ref-48) [arXiv:2010.15968](https://arxiv.org/abs/2010.15968), [[49]](#ref-49) [arXiv:2507.02043](https://arxiv.org/abs/2507.02043).

#### `vqls/`: Variational Quantum Linear Solver for Finite-Element Scattering
*   **Scripts:** [`vqls.py`](./vqls/vqls.py); paper copy [`1731241185612.pdf`](./vqls/1731241185612.pdf) [[54]](#ref-54)
*   **Methodology:** Implements the Variational Quantum Linear Solver (VQLS) [[50]](#ref-50), [[51]](#ref-51) to variationally minimize the cost Hamiltonian $\hat{H}_G = A^\dagger (\mathbb{I} - |b\rangle\langle b|) A$ using a `RealAmplitudes` ansatz and Simultaneous Perturbation Stochastic Approximation (`SPSA`). Motivated by Finite Element Method (FEM) discretizations of the Helmholtz equation for Radar Cross Section (RCS) estimation [[5]](#ref-5), [[54]](#ref-54), it evaluates state overlap $|\langle x_{\text{exact}} | \psi(\boldsymbol{\theta}^*) \rangle|^2$ on banded Hermitian matrices with Pauli-string decompositions [[52]](#ref-52), using `qiskit_aer` and `NumPyLinearSolver` as the exact reference.
*   **Verified References:** [[5]](#ref-5) [arXiv:1301.2340](https://arxiv.org/abs/1301.2340), [[50]](#ref-50) [arXiv:1909.03898](https://arxiv.org/abs/1909.03898), [[51]](#ref-51) [arXiv:1909.05820](https://arxiv.org/abs/1909.05820), [[52]](#ref-52) [arXiv:2004.06252](https://arxiv.org/abs/2004.06252), [[54]](#ref-54) EuRAD 2024.

#### `qaoa/`: Adaptive Quantum Approximate Optimization Algorithm (ADAPT-QAOA)
*   **Scripts:** [`maxcut-qaoa-gpt.py`](./qaoa/maxcut-qaoa-gpt.py)
*   **Methodology:** Extends the standard Quantum Approximate Optimization Algorithm [[11]](#ref-11) to the adaptive mixer selection framework (ADAPT-QAOA) of Zhu *et al.* [[12]](#ref-12) for the Max-Cut combinatorial optimization problem. At each layer $k$, the operator $\hat{A}_k$ from a predefined mixer pool $\mathcal{P}$ that maximizes the energy gradient magnitude $|\langle \psi^{(k-1)} | [\hat{H}_C, \hat{A}_j] | \psi^{(k-1)} \rangle|$ is appended to the ansatz. States are propagated as `qiskit.quantum_info.DensityMatrix` objects (allowing an optional depolarizing noise probability), compared against a standard QAOA ansatz and the brute-force optimal cut, and plotted as approximation ratios, mixer gradients and optimal parameters.
*   **Verified References:** [[11]](#ref-11) [arXiv:1411.4028](https://arxiv.org/abs/1411.4028), [[12]](#ref-12) [arXiv:2005.10258](https://arxiv.org/abs/2005.10258).

---

### 2.3 Random Circuit Sampling, Tensor Networks & Approximate Simulation

#### `rcs/` and `rcs/nn_qab/`: Distributed Automatic Circuit Elision (ACE) for Random Circuit Sampling
*   **Scripts:** [`nn_qab/nn_qab.py`](./rcs/nn_qab/nn_qab.py), [`nn_qab/nn_qab32plus.py`](./rcs/nn_qab/nn_qab32plus.py), [`nn_qab/nn_qab-adversarial.py`](./rcs/nn_qab/nn_qab-adversarial.py), [`nn_qab/nn_qab_graph.py`](./rcs/nn_qab/nn_qab_graph.py), [`nn_qab/nn_qab.sh`](./rcs/nn_qab/nn_qab.sh), [`nn_qab/nn_qab32plus.sh`](./rcs/nn_qab/nn_qab32plus.sh)
*   **Methodology:** A multi-pass benchmark suite quantifying the accuracy of PyQrack's seam-partitioned `QrackAceBackend` (Automatic Circuit Elision) [[22]](#ref-22) on nearest-neighbor random quantum circuits (torus topology, default depth 12, seeds 0-99) as a function of the geometric **bulk-to-boundary ratio** ($\text{B-to-B} = n_{\text{bulk}} / n_{\text{boundary}}$).
    *   **Two-Pass Orchestration:** Pass A executes low-memory local patch sampling via `QrackAceBackend`; Pass B computes exact $2^n$ reference amplitudes via `QrackSimulator` on high-memory GPUs or paged host RAM/NVMe swap, scoring normalized linear Cross-Entropy Benchmarking (XEB) [[31]](#ref-31), [[32]](#ref-32) and Heavy Output Generation (HOG) [[53]](#ref-53):
        $$\text{XEB} = \frac{2^n \sum_{x} q(x) p_{\text{ideal}}(x) - 1}{2^n \sum_{x} p_{\text{ideal}}(x)^2 - 1}$$
        Both passes are resumable per seed and pinned to a single OpenCL device by default.
    *   **Width Series:** `nn_qab.sh` covers widths 10 to 28; `nn_qab32plus.sh` extends to 36 via `QPager` paging across devices. Prime widths (11, 13, 17, 19, 23, 29, 31) are excluded because `factor_width()` collapses them to a 1-D ring.

        | Width | `lrc` | `lrr` | B-to-B | Sweep | Note |
        | ---: | ---: | ---: | ---: | :--- | :--- |
        | 10 | 2 | 2 | 1.5 | `nn_qab.sh` | |
        | 14 | 3 | 2 | 2.5 | `nn_qab.sh` | |
        | 22 | 5 | 2 | 4.5 | `nn_qab.sh` | |
        | 26 | 6 | 2 | 5.5 | `nn_qab.sh` | |
        | 27 | 4 | 3 | 3.5 | `nn_qab.sh` | |
        | 28 | 3 | 4 | 2.5 | `nn_qab.sh` | replicates 14 (breaks width/ratio collinearity) |
        | 30 | 2 | 5 | 2.0 | `nn_qab32plus.sh` | |
        | 32 | 3 | 4 | 3.0 | `nn_qab32plus.sh` | |
        | 33 | 5 | 3 | 4.5 | `nn_qab32plus.sh` | replicates 22 |
        | 34 | | | 7.5 | `nn_qab32plus.sh` | highest ratio in the set |
        | 35 | | | 2.5 | `nn_qab32plus.sh` | replicates 14 and 28 |
        | 36 | | | 2.0 | `nn_qab32plus.sh` | replicates 30 |

    *   **Adversarial Controls:** `nn_qab-adversarial.py` separates genuine approximate simulation from XEB spoofing attacks [[34]](#ref-34), [[37]](#ref-37) by comparing `ace` against seam-severed (`severed`), ideal-sampling (`ceiling`), and independent-circuit (`null`) control arms, pooling multi-run heterogeneity via DerSimonian–Laird random-effects meta-analysis.
    *   **Related:** device-specific RCS and postselection studies (e.g. Nighthawk) live under [`devices/nighthawk/`](../devices/nighthawk/).
*   **Verified References:** [[31]](#ref-31) [arXiv:1608.00263](https://arxiv.org/abs/1608.00263), [[53]](#ref-53) [arXiv:1612.05903](https://arxiv.org/abs/1612.05903), [[32]](#ref-32) [arXiv:1910.11333](https://arxiv.org/abs/1910.11333), [[33]](#ref-33) [arXiv:2005.02464](https://arxiv.org/abs/2005.02464), [[34]](#ref-34) [arXiv:2112.01657](https://arxiv.org/abs/2112.01657), [[35]](#ref-35) [arXiv:2211.03999](https://arxiv.org/abs/2211.03999), [[36]](#ref-36) [arXiv:2211.13267](https://arxiv.org/abs/2211.13267), [[22]](#ref-22) [arXiv:2304.14969](https://arxiv.org/abs/2304.14969), [[37]](#ref-37) [arXiv:2304.11119](https://arxiv.org/abs/2304.11119), [[38]](#ref-38) [arXiv:2306.16659](https://arxiv.org/abs/2306.16659), [[39]](#ref-39) [arXiv:2406.18889](https://arxiv.org/abs/2406.18889).

#### `rcsqbdd/`: RCS with Quantum Binary Decision Diagrams & Matrix Product States
*   **Scripts:**

| Script | Role |
| :--- | :--- |
| [`fcrcsqbdd.py`](./rcsqbdd/fcrcsqbdd.py) | Patch-circuit XEB: full grid vs. split-patch ideal |
| [`fcrcsqbdd-tensor.py`](./rcsqbdd/fcrcsqbdd-tensor.py) | Patch-circuit XEB with probability tensors (optional `torch` GPU path) |
| [`fcrcsqbdd-tensor-noisy.py`](./rcsqbdd/fcrcsqbdd-tensor-noisy.py) | Noisy variant of the tensor benchmark |
| [`fcrcsqbdd-tensor-print.py`](./rcsqbdd/fcrcsqbdd-tensor-print.py) | Dump pickled probability tensors |
| [`fcrcsqbdd-tensor-loadplot.py`](./rcsqbdd/fcrcsqbdd-tensor-loadplot.py) | Load and plot pickled tensors |
| [`fcrcsqbdd-mps-inspector.py`](./rcsqbdd/fcrcsqbdd-mps-inspector.py) | MPS patch simulation and probability-tensor inspection |
| [`qvqbdd.py`](./rcsqbdd/qvqbdd.py) | Quantum Volume sampling and heavy-output extraction via QBDD |
| [`qvqbdd-tensor.py`](./rcsqbdd/qvqbdd-tensor.py) | QV with `quimb` tensor-network reference |

*   **Methodology:** Evaluates Patch-Circuit XEB and Quantum Volume (QV) circuits [[27]](#ref-27) by combining PyQrack's Quantum Binary Decision Diagram (QBDD) simulation engine [[40]](#ref-40), [[22]](#ref-22) with Matrix Product State (MPS) tensor-network contractions in `quimb` and `torch` [[41]](#ref-41), [[42]](#ref-42). Inspects bond-dimension truncation fidelity and heavy-output extraction across full and split-patch topologies. Usage: `python <script> [width] [depth]`.
*   **Verified References:** [[40]](#ref-40) [arXiv:1707.00865](https://arxiv.org/abs/1707.00865), [[27]](#ref-27) [arXiv:1811.12926](https://arxiv.org/abs/1811.12926), [[41]](#ref-41) [arXiv:1905.08394](https://arxiv.org/abs/1905.08394), [[42]](#ref-42) [arXiv:2002.07730](https://arxiv.org/abs/2002.07730), [[22]](#ref-22) [arXiv:2304.14969](https://arxiv.org/abs/2304.14969).

#### `qvml/`: Quantum Variational Machine Learning & Tensor Network Contraction
*   **Scripts:** [`qvml.py`](./qvml/qvml.py), [`qmvl_overview.py`](./qvml/qmvl_overview.py), [`qvml_heatmap.py`](./qvml/qvml_heatmap.py), [`qvml_spheres.py`](./qvml/qvml_spheres.py), [`qvml.sh`](./qvml/qvml.sh), [`qvml_csv.sh`](./qvml/qvml_csv.sh), [`hybrid/qml-qneuron.py`](./qvml/hybrid/qml-qneuron.py)
*   **Methodology:** Converts randomized Quantum Volume circuits [[27]](#ref-27) into tensor networks via `quimb` [[26]](#ref-26), [[28]](#ref-28) and formulates contraction-path optimization as an equivalent Traveling Salesperson Problem (TSP) solved via `pyqrackising`, then contracts the network iteratively. `qvml_heatmap.py` renders contraction cost against width and depth from the CSV output of `qvml_csv.sh`; `qmvl_overview.py` and `qvml_spheres.py` provide 3D topological visualizations of contraction graphs using `networkx` and `vedo`. `hybrid/qml-qneuron.py` pairs a classical `nn.Linear` low-rank path with a `QrackNeuronTorchLayer` residual path for high-rank correlations.
*   **Verified References:** [[26]](#ref-26) [arXiv:quant-ph/0511069](https://arxiv.org/abs/quant-ph/0511069), [[27]](#ref-27) [arXiv:1811.12926](https://arxiv.org/abs/1811.12926), [[28]](#ref-28) [arXiv:2002.01935](https://arxiv.org/abs/2002.01935).

---

### 2.4 Quantum Dynamics, Scrambling & Holographic Duality

#### `er-epr/`: ER=EPR Traversable Wormhole Teleportation
*   **Scripts:** [`pyqrack.py`](./er-epr/pyqrack.py)
*   **Methodology:** Simulates the holographic traversable wormhole teleportation protocol grounded in the ER=EPR conjecture of Maldacena and Susskind [[1]](#ref-1) and the double-trace deformation mechanism of Gao, Jafferis, and Wall [[2]](#ref-2), [[3]](#ref-3). Prepares a Thermofield Double (TFD) state from Bell pairs across Left ($L$) and Right ($R$) boundary registers, evolves backward and forward under chaotic Ising scrambling dynamics, injects a message qubit into $L$, applies a bilateral negative-energy shockwave coupling $e^{i g V_{LR}}$, and measures the unscrambled expectation value at boundary $R$.
*   **Verified References:** [[1]](#ref-1) [arXiv:1306.0533](https://arxiv.org/abs/1306.0533), [[2]](#ref-2) [arXiv:1608.05687](https://arxiv.org/abs/1608.05687), [[3]](#ref-3) [arXiv:1911.06314](https://arxiv.org/abs/1911.06314).

#### `qecho/`: Out-of-Time-Order Correlators (OTOCs) & Quantum Echoes
*   **Scripts:**

| Location | Files | Role |
| :--- | :--- | :--- |
| `qecho/` | [`otoc_statevector_simulation.py`](./qecho/otoc_statevector_simulation.py) | Minimal `generate_otoc_samples` example at 56 qubits |
| `qecho/` | [`otoc_validation_isingonly_cpu.py`](./qecho/otoc_validation_isingonly_cpu.py), [`otoc_validation_isingonly_cpu.sh`](./qecho/otoc_validation_isingonly_cpu.sh) | $(n+1)$-dimensional TFIM reduction vs. sampled marginals |
| `qecho/` | [`otoc_validation_isingonly_graph.py`](./qecho/otoc_validation_isingonly_graph.py) | Parallel log parsing and interpolated 3D surfaces (`vedo`) |
| `qecho/` | [`otocs-prediction-512.py`](./qecho/otocs-prediction-512.py) | Linear runtime fit from `qXX_d512.log` files, extrapolated to 10k qubits at depth 512 |
| [`Prototyping/`](./qecho/Prototyping/) | `otocs-pyqrack.py`, `otocs-pyqrack2d.py`, `quecho-pyqrack.py`, `otac-claude.py` | Direct `QrackSimulator` OTOC and echo circuits (1D, 2D) |
| [`Prototyping/`](./qecho/Prototyping/) | `otoc_validation_isingonly.py` / `.sh`, `otoc_validation_isingonly_graph_*.py`, `otocs-prediction.py` | Earlier validation, plotting and prediction iterations |
| [`plateau/`](./qecho/plateau/) | `qrackneuron.py` | Echo (time-reversal) initialization of `QrackNeuronTorchLayer` landscapes |
| [`Docs/`](./qecho/Docs/) | `2510.19550v1.pdf`, `PyQrack OTOC OCL TN .pdf` | Paper copy [[18]](#ref-18) and PyQrack OTOC notes |

*   **Methodology:** Simulates and validates Out-of-Time-Order Correlators
    $$C(t) = \langle \hat{W}^\dagger(t) \hat{V}^\dagger(0) \hat{W}(t) \hat{V}(0) \rangle$$
    and Loschmidt time-reversal echo protocols to probe operator growth, butterfly velocities, and quantum information scrambling [[15]](#ref-15), [[16]](#ref-16), [[17]](#ref-17), [[18]](#ref-18). Leverages `pyqrackising`'s `generate_otoc_samples` kernel up to 56+ qubits and benchmarks $(n+1)$-dimensional Transverse-Field Ising Model (TFIM) approximations against full $2^n$ statevector evolutions across deep circuit regimes ($d = 512$).
*   **Verified References:** [[15]](#ref-15) [arXiv:1503.01409](https://arxiv.org/abs/1503.01409), [[16]](#ref-16) [arXiv:1602.06271](https://arxiv.org/abs/1602.06271), [[17]](#ref-17) [arXiv:1806.02807](https://arxiv.org/abs/1806.02807), [[18]](#ref-18) [arXiv:2510.19550](https://arxiv.org/abs/2510.19550).

#### `qwscatter/`: 1D & 2D Quantum Walk Scattering Dynamics
*   **Scripts:** [`qwscatter.py`](./qwscatter/qwscatter.py), [`qwscatter-3d.py`](./qwscatter/qwscatter-3d.py), [`qwscatter-large-qrack-multi.py`](./qwscatter/qwscatter-large-qrack-multi.py)
*   **Methodology:** Simulates discrete-time coined quantum walks on 1D chains with potential barriers and 2D lattices ($64 \times 64$ and scaled multi-GPU grids) [[29]](#ref-29), [[30]](#ref-30). Visualizes ballistic wavepacket dispersion ($\sigma(t) \propto t$), constructive/destructive interference fringes, and scattering reflection/transmission amplitudes using `Qiskit Aer` (1D, 2D) and multi-GPU `PyQrack` (large 2D).
*   **Verified References:** [[29]](#ref-29) [arXiv:quant-ph/0303081](https://arxiv.org/abs/quant-ph/0303081), [[30]](#ref-30) [arXiv:0806.1972](https://arxiv.org/abs/0806.1972).

---

### 2.5 Quantum Error Mitigation, Metrology & Cryptographic Auditing

#### `sdrp/`: SDRP-Parameterized Zero-Noise Extrapolation (ZNE)
*   **Scripts:** [`sdrp_zne_batch.py`](./sdrp/sdrp_zne_batch.py)
*   **Methodology:** Treats PyQrack's **Schmidt-Decomposition Rounding Parameter (SDRP)** [[22]](#ref-22) inside `QrackAceBackend` (`ace.set_sdrp(sdrp)`) as a continuous, controllable truncation-noise parameter for Zero-Noise Extrapolation (ZNE) [[13]](#ref-13), [[14]](#ref-14). Extrapolates series-mean linear XEB to the zero-truncation limit ($\text{SDRP} \to 0$) via Mitiq's `LinearFactory`, `RichardsonFactory`, and `ExpFactory`, validating directly against exact $\text{SDRP} = 0$ patch executions.

    | Parameter | Default | CLI flag |
    | :--- | :--- | :--- |
    | Width series | 10, 14, 27, 22, 26 | `--series` |
    | Circuits per series | 100 (seeded) | `--circuits`, `--seed 42` |
    | Depth | 12 | `--depth` |
    | Shots | 1024 | `--shots` |
    | Golden SDRP $g = \frac{1 - 1/\sqrt{2}}{2}$ | 0.146447 | `--base-sdrp` |
    | Scale points | $g \cdot \{1/3,\ 2/3,\ 1\} \approx \{0.049,\ 0.098,\ 0.146\}$ | `--scales` |
    | Exact reference | on | `--no-exact` disables |

*   **Verified References:** [[13]](#ref-13) [arXiv:1612.02058](https://arxiv.org/abs/1612.02058), [[14]](#ref-14) [arXiv:2009.04417](https://arxiv.org/abs/2009.04417), [[22]](#ref-22) [arXiv:2304.14969](https://arxiv.org/abs/2304.14969).

#### `qec/`: Quantum Error Correction & Mitigation Wrappers
*   **Scripts:** [`mitiq-run.sh`](./qec/mitiq-run.sh)
*   **Methodology:** Shell wrapper that sets `QRACK_MAX_PAGING_QB=27`, `QRACK_QTENSORNETWORK_THRESHOLD_QB=1` and `QRACK_OCL_DEFAULT_DEVICE=0`, then runs `mitiq_repetition_code.py` at 36 qubits from the external `vm6502q/pyqrack-examples` checkout (`/notebook/qrack/pyqrack-examples/qec/`, present in the ThereminQ container). The target script combines repetition-code error correction with Mitiq error mitigation [[13]](#ref-13), [[14]](#ref-14).
*   **Verified References:** [[13]](#ref-13) [arXiv:1612.02058](https://arxiv.org/abs/1612.02058), [[14]](#ref-14) [arXiv:2009.04417](https://arxiv.org/abs/2009.04417).

#### `positron/`: Positronium Metrology via Unitary Inversion
*   **Scripts:** [`unitary-inverse-metrology-example.py`](./positron/unitary-inverse-metrology-example.py)
*   **Methodology:** Simulates the phase estimation protocol of Song *et al.* [[10]](#ref-10), which pairs a probe qubit evolving under an unknown SU(2) rotation $U(\theta, \hat{\mathbf{n}})$ with an entangled "antiqubit" evolving under the inverted unitary $U^\dagger(\theta, \hat{\mathbf{n}})$. Benchmarks the resulting Fisher Information and axis-independent sensitivity against separable (entanglement-free) and standard parallel-entangled (agnostic) probe configurations, running Qiskit circuits on the PyQrack Qiskit provider and plotting against theoretical predictions.
*   **Verified References:** [[10]](#ref-10) [arXiv:2506.04315](https://arxiv.org/abs/2506.04315).

#### `qhrf/`: QHRF Coherence-Enhancement Models (exploratory)
*   **Scripts:** [`qhrf.py`](./qhrf/qhrf.py), [`qhrf-qrack.py`](./qhrf/qhrf-qrack.py), [`qhrf-genesis-pyqrack.py`](./qhrf/qhrf-genesis-pyqrack.py)
*   **Methodology:** Exploratory reproductions of the third-party QHRF coherence-stabilization claims (see links in [`qhrf/README.md`](./qhrf/README.md)); not a peer-reviewed method.

    | Script | Backend | Model |
    | :--- | :--- | :--- |
    | `qhrf.py` | `qiskit_aer` | $T_1 / T_2$ thermal-relaxation noise model; fidelity decay with and without a 5x $T_1 / T_2$ improvement |
    | `qhrf-qrack.py` | `pyqrack` | Same comparison; native noise methods unavailable, so no decay is shown |
    | `qhrf-genesis-pyqrack.py` | `pyqrack` | Interpretive 6-qubit circuit of "QHRF GENESIS V6" (Musselwhite) multi-register echo interference |

*   **Verified References:** [[8]](#ref-8) [arXiv:quant-ph/9605011](https://arxiv.org/abs/quant-ph/9605011), [[22]](#ref-22) [arXiv:2304.14969](https://arxiv.org/abs/2304.14969).

#### `qllm-audit/`: Quantum & Cryptographic Auditing of Large Language Models
*   **Subdirectories:**

| Directory | Files | Role |
| :--- | :--- | :--- |
| [`audit/`](./qllm-audit/audit/) | `quantum_audit.py` | Shor-style factoring audit of an RSA modulus (`qiskit`) |
| [`audit/`](./qllm-audit/audit/) | `forensic_audit.py`, `stenography_audit.py` | Kolmogorov–Smirnov / LSB statistical scans of weights (`torch`, `scipy.stats`) |
| [`audit/`](./qllm-audit/audit/) | `infected_nanogpt.py` | NanoGPT-style model with an embedded backdoor (`torch`) |
| [`kit/`](./qllm-audit/kit/) | `vulnerable_model.py`, `exploit_kit.py` | Hash-triggered backdoor model and matching trigger kit |
| [`llmfw/`](./qllm-audit/llmfw/) | `llm_firewall.py` | Canonicalizing input firewall |

*   **Methodology:** Investigates theoretical supply-chain vulnerabilities arising from cryptographic backdoors planted in neural network weights [[23]](#ref-23) and prompt-injection triggers [[24]](#ref-24), [[25]](#ref-25). Demonstrates the tension between public-key backdoors (such as RSA moduli embedded via least-significant-bit steganography, which remain statistically covert under weight-distribution scans but are vulnerable to quantum factorization via Shor's algorithm [[7]](#ref-7)) and Post-Quantum Cryptographic (PQC / Learning With Errors) backdoors (which resist quantum factorization but induce uniform-distribution statistical anomalies detectable via Kolmogorov–Smirnov forensic tests), paired with canonicalizing input-firewall countermeasures (`llmfw/`).
*   **Verified References:** [[7]](#ref-7) [arXiv:quant-ph/9508027](https://arxiv.org/abs/quant-ph/9508027), [[23]](#ref-23) [arXiv:2204.06974](https://arxiv.org/abs/2204.06974), [[24]](#ref-24) [arXiv:2302.12173](https://arxiv.org/abs/2302.12173), [[25]](#ref-25) [arXiv:2306.05499](https://arxiv.org/abs/2306.05499).

---

## 3. Execution Environment & Hardware Acceleration

Scripts across `algorithms/` are designed to run inside the **ThereminQ** container environment or on native Linux hosts equipped with OpenCL-capable GPUs (including multi-GPU setups via Mesa `rusticl`, AMD ROCm, or NVIDIA OpenCL) and high-core-count CPUs:

```bash
# Launch the ThereminQ Jupyter & OpenCL runtime container
docker run --gpus all --device=/dev/dri:/dev/dri -d twobombs/thereminq-tensors:jupyter [-m 16g --memory-swap 320g]

# Or install core Python dependencies in a local virtual environment
pip install pyqrack pyqrackising numpy scipy qiskit qiskit-aer qiskit-algorithms pennylane pennylane-qrack cirq quimb mitiq torch openfermion openfermionpyscf networkx pandas seaborn matplotlib vedo tqdm
```

| Setting | Scope | Effect |
| :--- | :--- | :--- |
| `QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"` | `qft/`, `rcs/nn_qab/` | If the file exists, exported as `PYQRACK_SHARED_LIB_PATH` before `pyqrack` is imported (e.g. a build with `-DQBCAPPOW=7` for $>32$-qubit registers); `nn_qab*` also accept a `QRACK_LIB_PATH` environment override. Other modules use the library bundled with the installed `pyqrack` wheel. |
| `QRACK_OCL_DEFAULT_DEVICE` | all PyQrack scripts | Primary OpenCL device |
| `QRACK_QPAGER_DEVICES`, `QRACK_QUNITMULTI_DEVICES` | multi-GPU runs | Device lists for paging and multi-device QUnit |
| `QRACK_MAX_PAGING_QB`, `QRACK_QTENSORNETWORK_THRESHOLD_QB` | `qec/` | Paging ceiling and tensor-network threshold |

---

## 4. Verified Bibliography

Every arXiv identifier below was checked against its official arXiv record (title and author list) on 2026-11-10; journal references are given where published. [[54]](#ref-54) is not on arXiv and is checked against the paper copy in `vqls/`.

<a id="ref-1"></a>**[1]** J. Maldacena and L. Susskind, "Cool horizons for entangled black holes," *Fortschritte der Physik* **61**, 781–811 (2013), [arXiv:1306.0533](https://arxiv.org/abs/1306.0533).

<a id="ref-2"></a>**[2]** P. Gao, D. L. Jafferis, and A. C. Wall, "Traversable Wormholes via a Double Trace Deformation," *Journal of High Energy Physics* **2017**(12), 151 (2017), [arXiv:1608.05687](https://arxiv.org/abs/1608.05687).

<a id="ref-3"></a>**[3]** A. R. Brown, H. Gharibyan, S. Leichenauer, H. W. Lin, S. Nezami, G. Salton, L. Susskind, B. Swingle, and M. Walter, "Quantum Gravity in the Lab: Teleportation by Size and Traversable Wormholes," *PRX Quantum* **4**, 010320 (2023), [arXiv:1911.06314](https://arxiv.org/abs/1911.06314).

<a id="ref-4"></a>**[4]** A. W. Harrow, A. Hassidim, and S. Lloyd, "Quantum algorithm for solving linear systems of equations," *Physical Review Letters* **103**, 150502 (2009), [arXiv:0811.3171](https://arxiv.org/abs/0811.3171).

<a id="ref-5"></a>**[5]** B. D. Clader, B. C. Jacobs, and C. R. Sprouse, "Preconditioned quantum linear system algorithm," *Physical Review Letters* **110**, 250504 (2013), [arXiv:1301.2340](https://arxiv.org/abs/1301.2340).

<a id="ref-6"></a>**[6]** D. Dervovic, M. Herbster, P. Mountney, S. Severini, N. Usher, and L. Wossnig, "Quantum linear systems algorithms: a primer," (2018), [arXiv:1802.08227](https://arxiv.org/abs/1802.08227).

<a id="ref-7"></a>**[7]** P. W. Shor, "Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer," *SIAM Journal on Computing* **26**(5), 1484–1509 (1997), [arXiv:quant-ph/9508027](https://arxiv.org/abs/quant-ph/9508027).

<a id="ref-8"></a>**[8]** P. W. Shor, "Fault-tolerant quantum computation," *Proceedings of the 37th Annual Symposium on Foundations of Computer Science (FOCS)*, pp. 56–65 (1996), [arXiv:quant-ph/9605011](https://arxiv.org/abs/quant-ph/9605011).

<a id="ref-9"></a>**[9]** C. Gidney and M. Ekerå, "How to factor 2048 bit RSA integers in 8 hours using 20 million noisy qubits," *Quantum* **5**, 433 (2021), [arXiv:1905.09749](https://arxiv.org/abs/1905.09749).

<a id="ref-10"></a>**[10]** X. Song, S. S. Borjigin, F. Salvati, Y.-X. Wang, N. Yunger Halpern, D. R. M. Arvidsson-Shukur, and K. Murch, "Superconducting-antiqubit simulation enables optimal phase estimation via unitary inversion," (2025), [arXiv:2506.04315](https://arxiv.org/abs/2506.04315). (v1 title: "Superconducting antiqubits achieve optimal phase estimation via unitary inversion".)

<a id="ref-11"></a>**[11]** E. Farhi, J. Goldstone, and S. Gutmann, "A Quantum Approximate Optimization Algorithm," (2014), [arXiv:1411.4028](https://arxiv.org/abs/1411.4028).

<a id="ref-12"></a>**[12]** L. Zhu, H. L. Tang, G. S. Barron, F. A. Calderon-Vargas, N. J. Mayhall, E. Barnes, and S. E. Economou, "An adaptive quantum approximate optimization algorithm for solving combinatorial problems on a quantum computer," *Physical Review Research* **4**, 033029 (2022), [arXiv:2005.10258](https://arxiv.org/abs/2005.10258).

<a id="ref-13"></a>**[13]** K. Temme, S. Bravyi, and J. M. Gambetta, "Error mitigation for short-depth quantum circuits," *Physical Review Letters* **119**, 180509 (2017), [arXiv:1612.02058](https://arxiv.org/abs/1612.02058).

<a id="ref-14"></a>**[14]** R. LaRose, A. Mari, S. Kaiser, P. J. Karalekas, A. A. Alves, P. Czarnik, M. El Mandouh, M. H. Gordon, Y. Hindy, A. Robertson, P. Thakre, M. Wahl, D. Samuel, R. Mistri, M. Tremblay, N. Gardner, N. T. Stemen, N. Shammah, and W. J. Zeng, "Mitiq: A software package for error mitigation on noisy quantum computers," *Quantum* **6**, 774 (2022), [arXiv:2009.04417](https://arxiv.org/abs/2009.04417).

<a id="ref-15"></a>**[15]** J. Maldacena, S. H. Shenker, and D. Stanford, "A bound on chaos," *Journal of High Energy Physics* **2016**(08), 106 (2016), [arXiv:1503.01409](https://arxiv.org/abs/1503.01409).

<a id="ref-16"></a>**[16]** B. Swingle, G. Bentsen, M. Schleier-Smith, and P. Hayden, "Measuring the scrambling of quantum information," *Physical Review A* **94**, 040302(R) (2016), [arXiv:1602.06271](https://arxiv.org/abs/1602.06271).

<a id="ref-17"></a>**[17]** K. A. Landsman, C. Figgatt, T. Schuster, N. M. Linke, B. Yoshida, N. Y. Yao, and C. Monroe, "Verified Quantum Information Scrambling," *Nature* **567**, 61–65 (2019), [arXiv:1806.02807](https://arxiv.org/abs/1806.02807).

<a id="ref-18"></a>**[18]** C. Zhang, R. G. Cortiñas, A. H. Karamlou *et al.*, "Quantum computation of molecular geometry via many-body nuclear spin echoes," (2025), [arXiv:2510.19550](https://arxiv.org/abs/2510.19550).

<a id="ref-19"></a>**[19]** R. B. Griffiths and C.-S. Niu, "Semiclassical Fourier Transform for Quantum Computation," *Physical Review Letters* **76**, 3228–3231 (1996), [arXiv:quant-ph/9511007](https://arxiv.org/abs/quant-ph/9511007).

<a id="ref-20"></a>**[20]** S. Parker and M. B. Plenio, "Efficient factorization with a single pure qubit and $\log N$ mixed qubits," *Physical Review Letters* **85**, 3049–3052 (2000), [arXiv:quant-ph/0001066](https://arxiv.org/abs/quant-ph/0001066).

<a id="ref-21"></a>**[21]** S. Beauregard, "Circuit for Shor's algorithm using $2n+3$ qubits," *Quantum Information and Computation* **3**(2), 175–185 (2003), [arXiv:quant-ph/0205095](https://arxiv.org/abs/quant-ph/0205095).

<a id="ref-22"></a>**[22]** D. Strano, B. Bollay, A. Blaauw, N. Shammah, W. J. Zeng, and A. Mari, "Exact and approximate simulation of large quantum circuits on a single GPU," *Proceedings of the IEEE International Conference on Quantum Computing and Engineering (QCE)*, pp. 949–958 (2023), [arXiv:2304.14969](https://arxiv.org/abs/2304.14969).

<a id="ref-23"></a>**[23]** S. Goldwasser, M. P. Kim, V. Vaikuntanathan, and O. Zamir, "Planting Undetectable Backdoors in Machine Learning Models," *Proceedings of the 63rd Annual IEEE Symposium on Foundations of Computer Science (FOCS)*, pp. 931–942 (2022), [arXiv:2204.06974](https://arxiv.org/abs/2204.06974).

<a id="ref-24"></a>**[24]** K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz, "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection," *Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security (AISec)*, pp. 79–90 (2023), [arXiv:2302.12173](https://arxiv.org/abs/2302.12173).

<a id="ref-25"></a>**[25]** Y. Liu, G. Deng, Y. Li, K. Wang, Z. Wang, X. Wang, T. Zhang, Y. Liu, H. Wang, Y. Zheng, L. Y. Zhang, and Y. Liu, "Prompt Injection attack against LLM-integrated Applications," (2023), [arXiv:2306.05499](https://arxiv.org/abs/2306.05499).

<a id="ref-26"></a>**[26]** I. L. Markov and Y. Shi, "Simulating quantum computation by contracting tensor networks," *SIAM Journal on Computing* **38**(3), 963–981 (2008), [arXiv:quant-ph/0511069](https://arxiv.org/abs/quant-ph/0511069).

<a id="ref-27"></a>**[27]** A. W. Cross, L. S. Bishop, S. Sheldon, P. D. Nation, and J. M. Gambetta, "Validating quantum computers using randomized model circuits," *Physical Review A* **100**, 032328 (2019), [arXiv:1811.12926](https://arxiv.org/abs/1811.12926).

<a id="ref-28"></a>**[28]** J. Gray and S. Kourtis, "Hyper-optimized tensor network contraction," *Quantum* **5**, 410 (2021), [arXiv:2002.01935](https://arxiv.org/abs/2002.01935).

<a id="ref-29"></a>**[29]** J. Kempe, "Quantum random walks - an introductory overview," *Contemporary Physics* **44**(4), 307–327 (2003), [arXiv:quant-ph/0303081](https://arxiv.org/abs/quant-ph/0303081).

<a id="ref-30"></a>**[30]** A. M. Childs, "Universal computation by quantum walk," *Physical Review Letters* **102**, 180501 (2009), [arXiv:0806.1972](https://arxiv.org/abs/0806.1972).

<a id="ref-31"></a>**[31]** S. Boixo, S. V. Isakov, V. N. Smelyanskiy, R. Babbush, N. Ding, Z. Jiang, M. J. Bremner, J. M. Martinis, and H. Neven, "Characterizing Quantum Supremacy in Near-Term Devices," *Nature Physics* **14**, 595–600 (2018), [arXiv:1608.00263](https://arxiv.org/abs/1608.00263).

<a id="ref-32"></a>**[32]** F. Arute *et al.*, "Quantum supremacy using a programmable superconducting processor," *Nature* **574**, 505–510 (2019); supplementary information at [arXiv:1910.11333](https://arxiv.org/abs/1910.11333).

<a id="ref-33"></a>**[33]** A. Zlokapa, S. Boixo, and D. Lidar, "Boundaries of quantum supremacy via random circuit sampling," *npj Quantum Information* **9**, 36 (2023), [arXiv:2005.02464](https://arxiv.org/abs/2005.02464).

<a id="ref-34"></a>**[34]** X. Gao, M. Kalinowski, C.-N. Chou, M. D. Lukin, B. Barak, and S. Choi, "Limitations of Linear Cross-Entropy as a Measure for Quantum Advantage," *PRX Quantum* **5**, 010334 (2024), [arXiv:2112.01657](https://arxiv.org/abs/2112.01657).

<a id="ref-35"></a>**[35]** D. Aharonov, X. Gao, Z. Landau, Y. Liu, and U. Vazirani, "A polynomial-time classical algorithm for noisy random circuit sampling," *Proceedings of the 55th Annual ACM Symposium on Theory of Computing (STOC)*, pp. 945–957 (2023), [arXiv:2211.03999](https://arxiv.org/abs/2211.03999).

<a id="ref-36"></a>**[36]** S. Oh and S. Kais, "Race for Quantum Advantage using Random Circuit Sampling," (2022), [arXiv:2211.13267](https://arxiv.org/abs/2211.13267).

<a id="ref-37"></a>**[37]** A. Morvan *et al.*, "Phase transitions in random circuit sampling," *Nature* **634**, 328–333 (2024), [arXiv:2304.11119](https://arxiv.org/abs/2304.11119).

<a id="ref-38"></a>**[38]** B. Fefferman, S. Ghosh, M. Gullans, K. Kuroiwa, and K. Sharma, "Effect of non-unital noise on random circuit sampling," *PRX Quantum* **5**, 030317 (2024), [arXiv:2306.16659](https://arxiv.org/abs/2306.16659).

<a id="ref-39"></a>**[39]** X.-H. Zhao, H.-S. Zhong, F. Pan, Z. Chen, R. Fu, Z. Su, X. Xie, C. Zhao, P. Zhang, W. Ouyang, C.-Y. Lu, J.-W. Pan, and M.-C. Chen, "Leapfrogging Sycamore: Harnessing 1432 GPUs for $7\times$ Faster Quantum Random Circuit Sampling," *National Science Review* (2025), [arXiv:2406.18889](https://arxiv.org/abs/2406.18889).

<a id="ref-40"></a>**[40]** A. Zulehner and R. Wille, "Advanced Simulation of Quantum Computations," *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems* **38**(5), 848–859 (2019), [arXiv:1707.00865](https://arxiv.org/abs/1707.00865).

<a id="ref-41"></a>**[41]** C. Guo, Y. Liu, M. Xiong, S. Xue, X. Fu, A. Huang, X. Qiang, P. Xu, J. Liu, M. Zheng, H.-L. Huang, M. Deng, D. Poletti, W.-S. Bao, and J. Wu, "General-purpose quantum circuit simulator with Projected Entangled-Pair States and the quantum supremacy frontier," *Physical Review Letters* **123**, 190501 (2019), [arXiv:1905.08394](https://arxiv.org/abs/1905.08394).

<a id="ref-42"></a>**[42]** Y. Zhou, E. M. Stoudenmire, and X. Waintal, "What limits the simulation of quantum computers?," *Physical Review X* **10**, 041038 (2020), [arXiv:2002.07730](https://arxiv.org/abs/2002.07730).

<a id="ref-43"></a>**[43]** A. Peruzzo, J. McClean, P. Shadbolt, M.-H. Yung, X.-Q. Zhou, P. J. Love, A. Aspuru-Guzik, and J. L. O'Brien, "A variational eigenvalue solver on a photonic quantum processor," *Nature Communications* **5**, 4213 (2014), [arXiv:1304.3061](https://arxiv.org/abs/1304.3061).

<a id="ref-44"></a>**[44]** J. Romero, R. Babbush, J. R. McClean, C. Hempel, P. J. Love, and A. Aspuru-Guzik, "Strategies for quantum computing molecular energies using the unitary coupled cluster ansatz," *Quantum Science and Technology* **4**(1), 014008 (2018), [arXiv:1701.02691](https://arxiv.org/abs/1701.02691).

<a id="ref-45"></a>**[45]** S. Y.-C. Chen, H.-H. Tseng, H.-Y. Lin, and S. Yoo, "Learning to Program Quantum Measurements for Machine Learning," (2025), [arXiv:2505.13525](https://arxiv.org/abs/2505.13525).

<a id="ref-46"></a>**[46]** J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, "Barren plateaus in quantum neural network training landscapes," *Nature Communications* **9**, 4812 (2018), [arXiv:1803.11173](https://arxiv.org/abs/1803.11173).

<a id="ref-47"></a>**[47]** S. Wang, E. Fontana, M. Cerezo, K. Sharma, A. Sone, L. Cincio, and P. J. Coles, "Noise-Induced Barren Plateaus in Variational Quantum Algorithms," *Nature Communications* **12**, 6961 (2021), [arXiv:2007.14384](https://arxiv.org/abs/2007.14384).

<a id="ref-48"></a>**[48]** C. Ortiz Marrero, M. Kieferová, and N. Wiebe, "Entanglement Induced Barren Plateaus," *PRX Quantum* **2**, 040316 (2021), [arXiv:2010.15968](https://arxiv.org/abs/2010.15968).

<a id="ref-49"></a>**[49]** E. Zapusek, I. Rojkov, and F. Reiter, "Scaling Quantum Algorithms via Dissipation: Avoiding Barren Plateaus," (2025), [arXiv:2507.02043](https://arxiv.org/abs/2507.02043).

<a id="ref-50"></a>**[50]** X. Xu, J. Sun, S. Endo, Y. Li, S. C. Benjamin, and X. Yuan, "Variational algorithms for linear algebra," *Science Bulletin* **66**(21), 2181–2188 (2021), [arXiv:1909.03898](https://arxiv.org/abs/1909.03898).

<a id="ref-51"></a>**[51]** C. Bravo-Prieto, R. LaRose, M. Cerezo, Y. Subasi, L. Cincio, and P. J. Coles, "Variational Quantum Linear Solver," *Quantum* **7**, 1188 (2023), [arXiv:1909.05820](https://arxiv.org/abs/1909.05820).

<a id="ref-52"></a>**[52]** A. Arrasmith, L. Cincio, R. D. Somma, and P. J. Coles, "Operator Sampling for Shot-frugal Optimization in Variational Algorithms," (2020), [arXiv:2004.06252](https://arxiv.org/abs/2004.06252).

<a id="ref-53"></a>**[53]** S. Aaronson and L. Chen, "Complexity-Theoretic Foundations of Quantum Supremacy Experiments," *Proceedings of the 32nd Computational Complexity Conference (CCC)*, LIPIcs **79**, 22 (2017), [arXiv:1612.05903](https://arxiv.org/abs/1612.05903).

<a id="ref-54"></a>**[54]** T. Pellerin, R. van Gaalen, and R. I. A. Harmanny, "Radar Cross Section Estimation using the Variational Quantum Linear Solver Algorithm," *Proceedings of the 21st European Radar Conference (EuRAD)* (2024); copy in [`vqls/1731241185612.pdf`](./vqls/1731241185612.pdf).
