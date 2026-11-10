<img width="1016" height="443" alt="Screenshot from 2026-01-01 15-36-30" src="https://github.com/user-attachments/assets/b5c8d613-5dac-484a-baef-0032dfd8e484" />

# ThereminQ-examples

Experimental configurations, simulation frameworks and development examples for the multi-GPU **ThereminQ** platform, built around the OpenCL-accelerated **Qrack / PyQrack** simulator [[1]](#ref-1).

```bash
docker run --gpus all --device=/dev/dri:/dev/dri -d twobombs/thereminq-tensors:jupyter [-m 16g --memory-swap 320g]
```

---

## Repository Layout

| Path | Domain | Detailed documentation |
| :--- | :--- | :--- |
| [`algorithms/`](./algorithms/) | Quantum algorithm prototypes, verification pipelines and classical simulation methods (19 modules) | [`algorithms/README.md`](./algorithms/README.md) |
| [`applications/`](./applications/) | Domain applications and benchmarks: factoring, HPC, Ising models, peaked circuits, holographic tensors | per-subdirectory READMEs |
| [`architecture/`](./architecture/) | Distributed patch architectures: Cloverleaf/Cloverfield ACE engines and SKQD pipelines | per-subdirectory READMEs |
| [`devices/`](./devices/) | Exact-reference studies of published hardware experiments (IBM Nighthawk, Google Sycamore, Willow) | per-subdirectory READMEs |
| [`graphs/`](./graphs/) | Plotting and rendering scripts for simulation output | [`graphs/README.md`](./graphs/README.md) |
| [`fp16-validation.py`](./fp16-validation.py) | FP16 validation harness for Mesa `rusticl` OpenCL targets (`cl_khr_fp16` presence, arithmetic, MAD throughput, half/float round-trip, `half2` to `half16` vectors) | inline docstring |
| [`requirements.txt`](./requirements.txt) | Unpinned Python dependency set for the whole repository | |

---

## `algorithms/`

Full specifications, backends and a verified bibliography are in [`algorithms/README.md`](./algorithms/README.md).

| Module | Content |
| :--- | :--- |
| [`er-epr`](./algorithms/er-epr/) | Traversable wormhole teleportation (ER=EPR) with a thermofield-double state and Ising scrambling in PyQrack |
| [`hhl`](./algorithms/hhl/) | Harrow–Hassidim–Lloyd linear system solver via `qiskit_algorithms` |
| [`noisy-shors`](./algorithms/noisy-shors/) | Shor factoring at small and large register widths with an explicit permutation oracle |
| [`positron`](./algorithms/positron/) | Qubit/antiqubit phase estimation via unitary inversion |
| [`qaoa`](./algorithms/qaoa/) | ADAPT-QAOA for Max-Cut |
| [`qec`](./algorithms/qec/) | Repetition-code QEC with Mitiq error mitigation (wrapper) |
| [`qecho`](./algorithms/qecho/) | Out-of-Time-Order Correlators and Loschmidt echoes via `pyqrackising` and PyQrack |
| [`qft`](./algorithms/qft/) | Semiclassical QFT: single-control-qubit Shor order finding and the QFT condensate |
| [`qhrf`](./algorithms/qhrf/) | Exploratory reproductions of third-party QHRF coherence claims |
| [`qllm-audit`](./algorithms/qllm-audit/) | Cryptographic backdoors in LLM weights: Shor-style RSA audit, statistical forensics, input firewall |
| [`qvml`](./algorithms/qvml/) | Quantum Volume tensor-network contraction with path finding as TSP via `pyqrackising`; QrackNeuron hybrid layer |
| [`qwscatter`](./algorithms/qwscatter/) | 1D and 2D coined quantum walk scattering (Qiskit Aer, multi-GPU PyQrack) |
| [`rcs`](./algorithms/rcs/) | Nearest-neighbor RCS benchmarks of `QrackAceBackend` versus bulk-to-boundary ratio (`nn_qab.py`, widths 10 to 36) |
| [`rcsqbdd`](./algorithms/rcsqbdd/) | RCS and Quantum Volume via QBDD and MPS (`quimb`, `torch`) |
| [`sdrp`](./algorithms/sdrp/) | Zero-Noise Extrapolation with the ACE Schmidt-Decomposition Rounding Parameter as the noise knob |
| [`vqe`](./algorithms/vqe/) | VQE across PennyLane devices (`qrack.simulator`, Cirq, IBM Quantum) with OpenFermion Hamiltonians |
| [`vqe-qml`](./algorithms/vqe-qml/) | Learnable Hermitian observables for QML |
| [`vqe-qml-dissipate`](./algorithms/vqe-qml-dissipate/) | Dissipative ancilla resets against barren plateaus |
| [`vqls`](./algorithms/vqls/) | Variational Quantum Linear Solver for FEM radar cross section systems |

---

## `applications/`

| Path | Content |
| :--- | :--- |
| [`findafactor/`](./applications/findafactor/) | `run-findafactor.sh`: generates growing semiprimes and times the external `FindAFactor` factoring tool |
| [`hpc/`](./applications/hpc/) | Multi-GPU and cloud runners: Amazon Braket hybrid jobs with a custom Qrack image (`run-aws-byoc-qrack.py`), QRNG service, cosmology N-body on 4 GPUs, QFT cube 32+ multi-GPU, Sycamore patch-quadrant timing, RCS NN-49 on CPU, Dask FQA test |
| [`ising/ising-c/`](./applications/ising/ising-c/) | C/OpenCL 2D Ising and TFIM samplers (`main.c`, `ising_kernel.cl`, `tfim-c/`) |
| [`ising/ising-python/`](./applications/ising/ising-python/) | ACE island depth tests and visualizations |
| [`ising/pyqrackising/`](./applications/ising/pyqrackising/) | `PyQrackIsing` benchmarks: `MaxCUT`, `TSP`, `GPS` routing, `LABS` [[2]](#ref-2), protein `folding`, `TFIM-base` magnetisation |
| [`ising/quantinuum/`](./applications/ising/quantinuum/) | TFIM validation against Quantinuum-scale runs: island runs, `run352`, parameter scans, tensor plots |
| [`ising/tfim-qrack/`](./applications/ising/tfim-qrack/) | TFIM with Qrack: OpenCL and Python implementations, phase-diagram grid |
| [`peaked/`](./applications/peaked/) | The `P1_little_dimple.qasm` peaked-circuit challenge: `bruteforce` (PyQrack), `discovery` (tensor layout visuals), `simulation` (circuit resizing and connectivity), `solvers` (clustered angles, Haar deviation, holographic bulk, OTOC maps), `verification` (generated full-density and mirror challenges) |
| [`weed/`](./applications/weed/) | Holographic tensor-network ingestion and bulk reconstruction with the Weed tensor library (C++ and Python), training loop and sparse density data generators [[3]](#ref-3), [[4]](#ref-4) |

---

## `architecture/`

| Path | Content |
| :--- | :--- |
| [`cloverleaf-fences/`](./architecture/cloverleaf-fences/) | ThereminQ distributed Automatic Circuit Elision (ACE) and the Holographic Distributed Engine: patch-and-kick routines, multi-GPU boundary messaging and tensor-network partitioning to avoid monolithic statevector memory. Paper: [`Cloverleaf_Fences_Scaling.pdf`](./architecture/cloverleaf-fences/Cloverleaf_Fences_Scaling.pdf). Variational methods follow [[5]](#ref-5), [[6]](#ref-6). |
| [`cloverleaf-fences/concept/`](./architecture/cloverleaf-fences/concept/) | 72- and 108-qubit multi-node entanglement-forging prototypes |
| [`cloverleaf-fences/lhv-ace-xyz/`](./architecture/cloverleaf-fences/lhv-ace-xyz/) | SU(2)/XYZ ground-state estimation with local-hidden-variable kicks and ACE, by topology: `NN1D`, `NN2D`, `NN3D` (27 qubits), `NN25D` (4x4x4 branes, up to 4096 qubits), `FC3D`, `FC-concepts` (HOG/XEB, hadrons), and `NN4K` (K4 hyperoctagon embeddings via CP-SAT, belief-propagation loop mitigation) |
| [`cloverleaf-fences/syk-rcs/`](./architecture/cloverleaf-fences/syk-rcs/) | SYK / RCS concepts on 27x27 (729-qubit) cloverfields, NN and FC variants (concept stage) |
| [`sk-qd-early-cloverfield/`](./architecture/sk-qd-early-cloverfield/) | SKQD Krylov-subspace ground-state refinement for TFIM: ZeroMQ orchestrator/worker pipeline (`Braided-PyQrackIsing.PY`), 27x27 multi-cloverfield runner and dashboard |

---

## `devices/`

### `devices/nighthawk/`

Exact-reference PyQrack studies of IBM Nighthawk experiments. The `*-nighthawk` engine folders share `nighthawk_qrack.py` (release circuits rebuilt bit for bit, `exact` or `ace` backends) with `METHODOLOGY.md` and `RESULTS.md`; each folder carries the variant built for its experiment.

| Path | Focus | Main entry points |
| :--- | :--- | :--- |
| [`bro-nighthawk/`](./devices/nighthawk/bro-nighthawk/) | Doped Clifford Sampling (DCS) toolkit for IBM's verifiable-fidelity experiment [[7]](#ref-7), [[8]](#ref-8): XEB scoring, postselection, tensor-network sweeps, 3D Porter–Thomas views, multi-GPU OpenCL via `QRACK_QUNITMULTI_DEVICES`; spoofing and phase-transition context [[9]](#ref-9), [[10]](#ref-10) | `dcs_post_select_paralel.py`, `tnsweep.py`, `bro-xeb-viz.py`, `experiment2-xeb-dash-UI.py` |
| [`rcs-nighthawk/`](./devices/nighthawk/rcs-nighthawk/) | 61-qubit random circuit sampling on `ibm_phoenix` [[11]](#ref-11): mirror, patched, full and forward XEB under exact and ACE simulation; multi-chip cloverfield with the loom | `nighthawk_qrack.py`, `nighthawk_cloverfield.py`, `nighthawk_sweep.sh` |
| [`vqe-nighthawk/`](./devices/nighthawk/vqe-nighthawk/) | VQE on two-patch ACE circuits at variable register size | `nighthawk_qrack.py` |
| [`ghz-nighthawk/`](./devices/nighthawk/ghz-nighthawk/) | Chip-wide GHZ fidelity and GHZ-calibrated crosstalk refocusing for Shor order finding on a square lattice | `nighthawk_ghz.py`, paper PDF |
| [`shrs-nighthawk/`](./devices/nighthawk/shrs-nighthawk/) | Shor order finding (`shor9`) on Nighthawk registers across sizes and GPUs | `nighthawk_qrack.py`, `shrs-sweep.sh`, `shrs-sweep-cpu.sh` |
| [`dual-nighthawk/`](./devices/nighthawk/dual-nighthawk/) | Combined engine: dual-chip maps, GHZ, Shor, QrackBridge benchmark, 3D graphs | `nighthawk_qrack.py`, `nighthawk_3dgraph.py` |
| [`erepr-nighthawk/`](./devices/nighthawk/erepr-nighthawk/) | Sycamore wormhole Hamiltonian re-derived and run as two Nighthawk chips joined by the loom | `nighthawk_erepr.py` |

### Other devices

| Path | Content |
| :--- | :--- |
| [`sycamore56/`](./devices/sycamore56/) | Pointer to [`twobombs/thereminq-classiq`](https://github.com/twobombs/thereminq-classiq) |
| [`willow/`](./devices/willow/) | Early patched simulation of larger circuits: patch-quadrant timing CSV, runner script and 28-qubit FP32 `nn_qab` results |

---

## `graphs/`

| Script | Output |
| :--- | :--- |
| `run-draw-patched-sycamore256-time.py` | Heatmap of patched execution time versus circuit width and depth from `quantum_data.csv` |
| `metriq2025estimate.py` | Qubits versus logical depth: achieved systems against estimated requirements per application class |
| `sycamore_rings.sh`, `sycamore_spheres.sh`, `weave.sh`, `makeqftipsy.sh` | Rendering wrappers (Bonsai-style views) |
| `shors_rsaloop.sh`, `shors_winloop.sh`, `shors_failloop.sh` | Shor run loops for plotting |
| `supreme28q14d.sh`, `convert_measured_values.sh` | 28-qubit, depth-14 run and measured-value conversion |

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

---

## References

Identifiers checked against their arXiv records on 2026-11-10. Module-level references for `algorithms/` are in [`algorithms/README.md`](./algorithms/README.md#4-verified-bibliography).

<a id="ref-1"></a>**[1]** D. Strano, B. Bollay, A. Blaauw, N. Shammah, W. J. Zeng, and A. Mari, "Exact and approximate simulation of large quantum circuits on a single GPU," *Proc. IEEE QCE*, pp. 949–958 (2023), [arXiv:2304.14969](https://arxiv.org/abs/2304.14969).

<a id="ref-2"></a>**[2]** A. Gomez Cadavid, P. Chandarana, S. V. Romero, J. Trautmann, E. Solano *et al.*, "Scaling advantage with quantum-enhanced memetic tabu search for LABS," (2025), [arXiv:2511.04553](https://arxiv.org/abs/2511.04553).

<a id="ref-3"></a>**[3]** B. Swingle, "Entanglement Renormalization and Holography," *Physical Review D* **86**, 065007 (2012), [arXiv:0905.1317](https://arxiv.org/abs/0905.1317).

<a id="ref-4"></a>**[4]** F. Pastawski, B. Yoshida, D. Harlow, and J. Preskill, "Holographic quantum error-correcting codes: Toy models for the bulk/boundary correspondence," *Journal of High Energy Physics* **2015**(06), 149 (2015), [arXiv:1503.06237](https://arxiv.org/abs/1503.06237).

<a id="ref-5"></a>**[5]** A. Kandala, A. Mezzacapo, K. Temme, M. Takita, M. Brink, J. M. Chow, and J. M. Gambetta, "Hardware-efficient Variational Quantum Eigensolver for Small Molecules and Quantum Magnets," *Nature* **549**, 242–246 (2017), [arXiv:1704.05018](https://arxiv.org/abs/1704.05018).

<a id="ref-6"></a>**[6]** M. Cerezo, A. Arrasmith, R. Babbush, S. C. Benjamin, S. Endo, K. Fujii, J. R. McClean, K. Mitarai, X. Yuan, L. Cincio, and P. J. Coles, "Variational Quantum Algorithms," *Nature Reviews Physics* **3**, 625–644 (2021), [arXiv:2012.09265](https://arxiv.org/abs/2012.09265).

<a id="ref-7"></a>**[7]** S. Martiel, J.-U. Chung, A. Seif, S. Ghosh, I. Hincks *et al.*, "Sampling hard circuits with verifiably high fidelity," (2026), [arXiv:2607.25941](https://arxiv.org/abs/2607.25941).

<a id="ref-8"></a>**[8]** H. Manabe, H. Gu, and F. Pan, "Classical Simulation and Design Frontiers for IBM's Doped Clifford Sampling Experiment," (2026), [arXiv:2608.13110](https://arxiv.org/abs/2608.13110).

<a id="ref-9"></a>**[9]** B. Barak, C.-N. Chou, and X. Gao, "Spoofing Linear Cross-Entropy Benchmarking in Shallow Quantum Circuits," (2020), [arXiv:2005.02421](https://arxiv.org/abs/2005.02421).

<a id="ref-10"></a>**[10]** B. Ware, A. Deshpande, D. Hangleiter, P. Niroula, B. Fefferman *et al.*, "A sharp phase transition in linear cross-entropy benchmarking," (2023), [arXiv:2305.04954](https://arxiv.org/abs/2305.04954).

<a id="ref-11"></a>**[11]** T. Sedrakyan, Y. Zhang, H. Karapetyan, J. D. Baktay, H. Gharibyan *et al.*, "Quantum computational advantage in random-circuit sampling on IBM superconducting quantum computers," (2026), [arXiv:2609.28657](https://arxiv.org/abs/2609.28657).

---

*Qrack was initially developed by [Dan Strano](https://github.com/unitaryfund/qrack); Bonsai by [Jeroen Bédorf](https://github.com/treecode/Bonsai); ThereminQ by [Aryan Blaauw](https://github.com/twobombs).*
