<img width="1016" height="443" alt="Screenshot from 2026-01-01 15-36-30" src="https://github.com/user-attachments/assets/b5c8d613-5dac-484a-baef-0032dfd8e484" />

# ThereminQ-examples

Experimental configurations & development examples on the ThereminQ platform.

```bash
docker run --gpus all --device=/dev/dri:/dev/dri -d twobombs/thereminq-tensors:jupyter [-m 16g --memory-swap 320g]
```

## Repository Documentation

This repository is a collection of experimental configurations, simulation frameworks, and development examples operating on the multi-GPU ThereminQ platform. The project has been refactored into five main domains: `algorithms`, `applications`, `architecture`, `devices`, and `graphs`.

Below is a summary of each directory and its contents, describing the methodologies and providing links to foundational and verified arXiv preprints where applicable.

---

### `algorithms`

This directory houses various quantum algorithm prototypes, verification pipelines, and simulation methodologies optimized for classical execution.

*   **`algorithms/er-epr`**: Implementations relating to the holographic ER=EPR hypothesis and related simulations.
*   **`algorithms/hhl`**: Experimental code for the Harrow-Hassidim-Lloyd (HHL) algorithm used for solving linear systems of equations.
*   **`algorithms/noisy-shors`**: Simulation of Shor's integer factorization algorithm operating within a noisy environment context.
*   **`algorithms/positron`**: An experiment related to simulating anti-qubits in a quantum circuit.
*   **`algorithms/qaoa`**: ADAPT-QAOA implementations targeting combinatorial optimization problems such as Max-Cut.
*   **`algorithms/qec`**: Shell scripts for running quantum error correction (QEC) experiments utilizing the Mitiq library.
*   **`algorithms/qecho`**: Prototyping and simulation of Out-of-Time-Order Correlators (OTOCs) and Loschmidt echo protocols, used to study quantum chaos and information scrambling.
*   **`algorithms/qft`**: Implementations of the Quantum Fourier Transform (QFT), featuring semiclassical methodologies on ThereminQ/PyQrack and benchmarking circuits. Methodologies built upon exact GPU statevector simulations detailed in [[arXiv:2304.14969](https://arxiv.org/abs/2304.14969)].
*   **`algorithms/qhrf`**: Highly experimental code targeting Quantum Harmonic Resonance Frequency (QHRF).
*   **`algorithms/qllm-audit`**: Scripts and toolkits for automated analysis, auditing, and generation via LLM integration (e.g., Gemini).
*   **`algorithms/qvml`**: Quantum Variational Machine Learning (QVML) demonstrations using tensor network contractions. Transforms random quantum circuits into equivalent Traveling Salesperson Problems (TSP) solved via `pyqrackising`.
*   **`algorithms/qwscatter`**: 3D scatter plot visualization tooling.
*   **`algorithms/rcs`**: Highly optimized Random Circuit Sampling (RCS) benchmarks. Includes distributed multi-pass architectures (`nn_qeb.py`) to evaluate nearest-neighbor sampling fidelity via Automatic Circuit Elision (ACE). Includes references to Cross-Entropy Benchmarking (XEB) scaling frameworks from [[arXiv:2005.02464](https://arxiv.org/abs/2005.02464)], [[arXiv:2211.03999](https://arxiv.org/abs/2211.03999)], [[arXiv:2211.13267](https://arxiv.org/abs/2211.13267)], [[arXiv:2306.16659](https://arxiv.org/abs/2306.16659)], and GPU scaling properties from [[arXiv:2406.18889](https://arxiv.org/abs/2406.18889)].
*   **`algorithms/rcsqbdd`**: RCS benchmarks implemented using `pyqrack` for QBDD decision-diagram simulations paired with Matrix Product States (MPS) via `quimb` and `torch`.
*   **`algorithms/vqe`**: Implementations and variations of the Variational Quantum Eigensolver (VQE).
*   **`algorithms/vqe-qml`**: Experimental models fusing VQE with Quantum Machine Learning (QML) for Hermitian matrices.
*   **`algorithms/vqe-qml-dissipate`**: Explores dissipation methodologies to circumvent barren plateaus within Variational Quantum Algorithms.
*   **`algorithms/vqls`**: Implementations of the Variational Quantum Linear Solver.

### `applications`

This directory bridges low-level frameworks with domain-specific applications and benchmarks.

*   **`applications/findafactor`**: Shell benchmarks focused on factorization execution loops.
*   **`applications/hpc`**: High-Performance Computing benchmarks running on PyQrack across multi-GPU setups.
*   **`applications/ising`**: A broad suite for solving the Transverse-Field Ising Model (TFIM) and generalized Ising interactions. Includes pure C/OpenCL implementations, Python integrations, hardware testing for Quantinuum, and PyQrack frameworks mapping problems like protein folding and the TSP to Ising formulations.
*   **`applications/peaked`**: Codebases detailing solvers, simulations, and validation pipelines for peaked distributions and circuit properties.
*   **`applications/weed`**: AdS/CFT-related methodologies.

### `architecture`

This directory explores structural topologies, exact statevector distributions, and classical approximation boundaries.

*   **`architecture/sk-qd-early-cloverfield`**: High-performance pipeline implementations such as the Stochastic Krylov Quantum Dynamics for large spin models.
*   **`architecture/cloverleaf-fences`**: Contains the ThereminQ Distributed Automatic Circuit Elision (ACE) systems and the Holographic Distributed Engine. Projects focus on circumventing standard statevector memory limits via patch-and-kick routines, multi-GPU boundary messaging, and tensor network partition embedding for large (e.g., 729-qubit to 4096-qubit) multi-body SU(2)/XYZ simulations. Includes embeddings mapping topologies into maximal abelian covers (Hyperoctagon lattice) using constraint programming (CP-SAT) to mitigate Belief Propagation loop errors. Methodologies are grounded in variational scaling rules from [[arXiv:1704.05018](https://arxiv.org/abs/1704.05018)] and [[arXiv:2012.09265](https://arxiv.org/abs/2012.09265)].

### `devices`

This directory targets classical validation systems interacting directly with real quantum hardware sampling limits.

*   **`devices/bro-nighthawk`**: A unified Python and OpenCL toolkit executing the 'Trust me, BRO' Doped Clifford Sampling (DCS) pipeline. Handles Cross-Entropy Benchmarking (XEB), multi-GPU OpenCL orchestration via ThereminQ variables (`QRACK_QUNITMULTI_DEVICES`), temporal-boundary contractions, and 3D Porter-Thomas verification for circuit verification hardness, validating spoofing theories like those proposed in [[arXiv:2005.02421](https://arxiv.org/abs/2005.02421)] and sharp phase transitions evaluated in [[arXiv:2305.04954](https://arxiv.org/abs/2305.04954)].
*   **`devices/rcs-nighthawk`**: An exact-reference evaluation of approximate seam-partitioned RCS on hardware topologies (e.g., IBM Nighthawk), addressing classical simulation frontiers and advantage metrics seen in [[arXiv:2609.28657](https://arxiv.org/abs/2609.28657)].

### `graphs`

Contains automated generation scripts mapping output data (e.g., `quantum_data.csv`) into heatmaps, latency metrics, and interactive multi-dimensional views.

*   Includes tools mapping the computation time of patched executions against circuit width/depth (`run-draw-patched-sycamore256-time.py`) and standard rendering shell wrappers.

---

## Output Visualizations

**VQE H2 FCI**
![h2_sto-3g_vqe_convergence_seed42_ManualUCCSD_JW_default_qubit](https://github.com/user-attachments/assets/00f881b6-73e5-4554-a258-b4de190abf00)

**Willow 106x106**
![402325505-a887d0c6-be0b-4c94-abd0-8ee185d26e76 (1)](https://github.com/user-attachments/assets/991f546c-2b40-483a-9d14-4fa19a8a5ee8)

**QHRF**
![Screenshot from 2025-04-26 09-17-50](https://github.com/user-attachments/assets/01a3b824-ab49-4d34-af55-48ce7585f216)

**VQE-QML on Hermitian matrices**
![Screenshot from 2025-05-21 14-50-11](https://github.com/user-attachments/assets/c42ea5a2-cd38-4b9b-bd0d-d4e45715236d)

**ThereminQ Doped Clifford Sampling (XEB Tools)**
<img width="4290" height="2307" alt="xeb3d_landscape" src="https://github.com/user-attachments/assets/3773cb89-b899-46bd-b4ea-9b523dda637e" />

### Legacy Output: ThereminQ-Bonsai

![230887892-1490ddb4-959c-45bb-98ac-28fe682f92a9](https://github.com/user-attachments/assets/a093653a-e7c1-4747-a85a-6cc24f271923)
![141675948-cd953c5b-5bdc-4e67-82bd-649530596788 (1)](https://github.com/user-attachments/assets/4d1ffde1-e3e2-4238-b66f-2ef3258d6d5d)

*Note: Qrack was initially developed by [Dan Strano](https://github.com/unitaryfund/qrack) — Bonsai by [Jeroen Bedorf](https://github.com/treecode/Bonsai) — ThereminQ by [Aryan Blaauw](https://github.com/twobombs).*
