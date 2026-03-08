# ThereminQ-Examples Project Structure Documentation

## Overview

This repository contains experimental configurations and development examples on the ThereminQ platform, built on top of the Qrack quantum computing framework. The project demonstrates various quantum algorithms, benchmarks, and simulations.

## Repository Structure

```
thereminq-examples/
├── README.md                    # Main project documentation
├── LICENSE                      # License file
├── .gitattributes               # Git attributes
│
├── agentics/                    # AI/LLM integration examples
│   ├── gemini-query.py          # Gemini API interaction script
│   ├── simple-gemini-query.py   # Simplified Gemini query example
│   └── README.md
│
├── er-epr/                      # Entanglement-Robust EPR experiments
│   ├── pyqrack.py               # PyQrack implementation
│   └── README.md
│
├── findafactor/                 # Quantum factorization benchmark
│   ├── README.md
│   └── run-findafactor.sh       # Factorization benchmark runner
│
├── graphs/                      # Visualization and graph generation
│   ├── convert_measured_values.sh
│   ├── makeqftipsy.sh
│   ├── metriq2025estimate.py
│   ├── readme.md
│   ├── run-draw-patched-sycamore256-time.py
│   ├── shors_failloop.sh
│   ├── shors_rsaloop.sh
│   ├── shors_winloop.sh
│   ├── supreme28q14d.sh
│   ├── sycamore_rings.sh
│   ├── sycamore_rings.tar.gz
│   ├── sycamore_spheres.sh
│   └── weave.sh
│
├── hhl/                         # Harrow-Hassidim-Lloyd algorithm
│   ├── hhl.py                   # HHL algorithm implementation
│   └── readme.md
│
├── hpc/                         # High-Performance Computing benchmarks
│   ├── README.md
│   ├── run-aws-byoc-qrack.py
│   ├── run-cosmos-nbody-QuadGPU.sh
│   ├── run-findafactor.sh
│   ├── run-fqa-dask
│   ├── run-qft-cube32plus-multi
│   ├── run-qrng-aws-service.sh
│   ├── run-rcs-nn-49-cpu
│   └── run-sycamore-patch-quadrant-time
│
├── ising/                       # Ising model experiments
│   ├── README.md
│   ├── ising-c/                 # C/OpenCL implementations
│   │   ├── ising_kernel.cl
│   │   ├── ising_sim
│   │   ├── main.c
│   │   ├── README.md
│   │   └── tfim-c/              # Transverse Field Ising Model
│   │       ├── ising_sampler
│   │       ├── ising_sampler_opencl
│   │       ├── ising_sampler_opencl.c
│   │       ├── ising_sampler.c
│   │       ├── Makefile
│   │       └── sampler_kernel.cl
│   ├── tfim-qrack/              # TFIM with Qrack
│   │   ├── README.md
│   │   └── opencl/
│   │       └── ising-c/
│   │           └── (OpenCL files)
│   ├── ising-python/            # Python implementations
│   │   ├── readme.md
│   │   ├── README.md
│   │   └── ising_ace_islands-check/
│   │       ├── ising_ace_depth_scan.sh
│   │       ├── ising_ace_depth_series-high.sh
│   │       ├── ising_ace_depth_series-low.sh
│   │       ├── ising_depth_series_loop.sh
│   │       ├── ising_depth_series.sh
│   │       ├── README.md
│   │       ├── graphs/            # Visualization scripts
│   │       │   ├── island-deltas.py
│   │       │   ├── magcostsheatmap.py
│   │       │   ├── magcurveheatmap.py
│   │       │   ├── magcurveheatmap3d.py
│   │       │   ├── magcurveheatmap3dark.py
│   │       │   ├── magsqrcurveheatmap.py
│   │       │   └── visualisation.py
│   │       └── measurements/
│   │           ├── 3dtime.py
│   │           └── fullogfiltered.txt
│   ├── pyqrackising/            # PyQrack Ising benchmarks
│   │   ├── README.md
│   │   ├── GPS/
│   │   │   └── RouteIsing.py
│   │   ├── LABS/
│   │   │   ├── 2511.04553v1.pdf
│   │   │   ├── full-labs-graph.py
│   │   │   ├── full-labs.py
│   │   │   ├── full-labs.sh
│   │   │   ├── full-labs16fast.sh
│   │   │   ├── full-labs16mid.sh
│   │   │   ├── full-labs32.sh
│   │   │   ├── full-labs32high.sh
│   │   │   ├── full-labs32mid.sh
│   │   │   ├── full-labs64.sh
│   │   │   ├── labs_logs.tar.gz
│   │   │   ├── log-parse-labs-graph.py
│   │   │   └── log-parse-labs.py
│   │   ├── MaxCUT/              # Max-Cut problem benchmarks
│   │   │   ├── maxcut_benchmarks.py
│   │   │   ├── maxcut_benchmarks.sh
│   │   │   ├── maxcut_gpu_perf_viz.py
│   │   │   ├── maxcut_quad_graph.py
│   │   │   ├── maxcut_random_opts.py
│   │   │   ├── maxcut_random.py
│   │   │   ├── maxcut_random.sh
│   │   │   ├── maxcut_stereo_graph.py
│   │   │   ├── README.md
│   │   │   └── results.tar.gz
│   │   ├── paretofront/
│   │   │   ├── 2511.01762v1.pdf
│   │   │   └── pyqrack-multisolver.py
│   │   ├── srohs/               # Shor's algorithm variants
│   │   │   ├── factorizing_wrapper_graph.py
│   │   │   ├── factorizing_wrapper.py
│   │   │   ├── factorizing_wrapper.sh
│   │   │   └── readme.md
│   │   ├── srohs/PyQrackIsing795_or_less/
│   │   │   ├── factorizing_wrapper_graph.py
│   │   │   ├── factorizing_wrapper.py
│   │   │   └── factorizing_wrapper.sh
│   │   ├── TFIM-base/           # TFIM base implementation
│   │   │   ├── README.md
│   │   │   ├── sqr_magnetisation-iterations_cli.py
│   │   │   ├── sqr_magnetisation-iterations_ui.py
│   │   │   ├── sqr_magnetisation-iterations.py
│   │   │   ├── sqr_magnetisation-iterations.sh
│   │   │   ├── start_ui.sh
│   │   │   └── !old/            # Deprecated files
│   │   └── TSP/                 # Traveling Salesman Problem
│   │       ├── parse_results_2d.py
│   │       ├── pyqrackising_benchmark_iterations.py
│   │       ├── pyqrackising_benchmark_iterations.sh
│   │       ├── README.md
│   │       ├── tsp_monte_carlo.py
│   │       ├── tsp_monte_carlo.sh
│   │       ├── tsp.py
│   │       ├── tsp.sh
│   │       ├── visualize_3d_mc.py
│   │       └── visualize_3d.py
│   └── quantinuum/              # Quantinuum hardware experiments
│       ├── README.md
│       ├── tfim_validation_tensor_plot.py
│       └── island-run/
│       └── scans/
│
├── noisy-shors/                 # Noisy Shor's algorithm
│   ├── noisy-big-shors.py
│   ├── noisy-smol-shors.py
│   └── README.md
│
├── peaked/                      # Quantum Phase Estimation experiments
│   ├── P1_little_dimple.qasm
│   ├── README.md
│   ├── bruteforce/
│   │   ├── aggregate.py
│   │   ├── qpepyqrack.py
│   │   ├── qpepyqrack.sh
│   │   ├── qpepyqrackqbd_viz.py
│   │   ├── qpepyqrackqbdd-transpiled.py
│   │   ├── qpepyqrackqbdd.py
│   │   └── README.md
│   ├── discovery/
│   │   ├── qpe.qasm
│   │   ├── qpetensorvizphyz.py
│   │   ├── qpetensorvizphyztagged.py
│   │   ├── qpetensorviztube.py
│   │   └── README.md
│   ├── simulation/
│   │   ├── connectivityplot.py
│   │   ├── hide+steer_qrack.py
│   │   ├── README.md
│   │   └── resize.py
│   ├── solvers/
│   │   ├── README.md
│   │   ├── clustered_angles_solver_P1/
│   │   │   ├── qpetensor.py
│   │   │   ├── README.md
│   │   │   ├── render_landscape.py
│   │   │   └── solve_dimple.py
│   │   ├── haar_solver_P1/
│   │   │   ├── haar-deviation.py
│   │   │   ├── purified-viz.py
│   │   │   ├── purify-deviation.py
│   │   │   ├── README.md
│   │   │   └── transpile-deviation.py
│   │   ├── holographic_solver_P1/
│   │   │   ├── analyse_stabilizers.py
│   │   │   ├── extract_peak.py
│   │   │   ├── holographic_bulk.py
│   │   │   └── README.md
│   │   │   └── solve_pyqrack.py
│   │   └── otoc_maps_solver_P1/
│   │       ├── README.md
│   │       ├── solution_otoc_p1b.py
│   │       └── solution_p1_optimize-hybrid.py
│   └── verification/
│       ├── peaked_generation_pyqrack_mirrored.py
│       ├── peaked_generation_pyqrack_p1-hybrid-sparse-36.py
│       ├── peaked_generation_pyqrack_p1-hybrid-sparse.py
│       ├── peaked_generation_pyqrack_p1-hybrid.py
│       ├── peaked_generation_pyqrack_p1.py
│       ├── peaked_generation_pyqrack.py
│       └── README.md
│
├── positron/                    # Anti-qubit simulation (Tenet-inspired)
│   ├── readme.md
│   └── unitary-inverse-metrology-example.py
│
├── qaoa/                        # Quantum Approximate Optimization Algorithm
│   ├── maxcut-qaoa-gpt.py       # ADAPT-QAOA for Max-Cut
│   └── README.md
│
├── qec/                         # Quantum Error Correction
│   ├── mitiq-run.sh             # Mitiq library QEC experiment
│   └── README.md
│
├── qecho/                       # Out-of-Time-Order Correlators (OTOC)
│   ├── README.md
│   ├── otoc_statevector_simulation.py
│   ├── otoc_validation_isingonly_cpu.py
│   ├── otoc_validation_isingonly_cpu.sh
│   ├── otoc_validation_isingonly_graph.py
│   ├── otocs-prediction-512.py
│   ├── Docs/                    # Documentation PDFs
│   │   ├── 2510.19550v1.pdf
│   │   ├── PyQrack OCL TN .pdf
│   │   └── README.md
│   ├── plateau/
│   │   └── qrackneuron.py
│   └── Prototyping/             # Experimental OTOC implementations
│       ├── otac-claude.py
│       ├── otoc_validation_isingonly_graph_highest.py
│       ├── otoc_validation_isingonly_graph_mpio_zero_shape_echo.py
│       ├── otoc_validation_isingonly_graph_mpio_zero_shape.py
│       ├── otoc_validation_isingonly_graph_mpio_zero.py
│       ├── otoc_validation_isingonly_graph_mpio.py
│       ├── otoc_validation_isingonly.py
│       ├── otoc_validation_isingonly.sh
│       ├── otocs-prediction.py
│       ├── otocs-pyqrack.py
│       ├── otocs-pyqrack2d.py
│       ├── quecho-pyqrack.py
│       └── README.md
│
├── qhrf/                        # Quantum Harmonic Resonance Frequency
│   ├── qhrf-genesis-pyqrack.py
│   ├── qhrf-qrack.py
│   ├── qhrf.py
│   └── readme.md
│
├── qllm-audit/                  # Quantum Large Language Model Audit
│   ├── README.md
│   ├── audit/                   # Audit implementations
│   │   ├── forensic_audit.py
│   │   ├── infected_nanogpt.py
│   │   ├── quantum_audit.py
│   │   ├── README.md
│   │   └── stenography_audit.py
│   ├── kit/                     # Exploit kit
│   │   ├── exploit_kit.py
│   │   ├── README.md
│   │   └── vulnerable_model.py
│   └── llmfw/                   # LLM firewall
│       ├── llm_firewall.py
│       └── README.md
│
├── qvml/                        # Quantum Variational Machine Learning
│   ├── qmvl_overview.py
│   ├── qvml_csv.sh
│   ├── qvml_heatmap.py
│   ├── qvml_spheres.py
│   ├── qvml.py
│   ├── qvml.sh
│   ├── readme.md
│   └── hybrid/
│       └── qml-qneuron.py
│
├── qwscatter/                   # Quantum Scatter Plot Visualization
│   ├── qwscatter-3d.py
│   ├── qwscatter-large-qrack-multi.py
│   ├── qwscatter.py
│   └── readme.md
│
├── rcs/                         # Random Circuit Sampling
│   ├── 2024_willow_patch_quadrant_time_all.csv
│   ├── README.md
│   └── run-sycamore-patch-quadrant-time
│
├── rcsqbdd/                     # RCS with QBDD simulation
│   ├── fcrcsqbdd-mps-inspector.py
│   ├── fcrcsqbdd-tensor-loadplot.py
│   ├── fcrcsqbdd-tensor-noisy.py
│   ├── fcrcsqbdd-tensor-print.py
│   ├── fcrcsqbdd-tensor.py
│   ├── fcrcsqbdd.py
│   ├── qvqbdd-tensor.py
│   ├── qvqbdd.py
│   └── README.md
│
├── vqe/                         # Variational Quantum Eigensolver
│   ├── import_clifford_vqe_entangled-gpu-multi.py
│   ├── import_clifford_vqe_entangled.py
│   ├── import_clifford_vqe_entangled.sh
│   ├── import_clifford_vqe_min.csv
│   ├── import_clifford_vqe_min.py
│   ├── readme.md
│   ├── run-multi-vqe-cirq-h2.py
│   ├── run-multi-vqe-ibmheron-h2.py
│   ├── run-multi-vqe-pennylane-h2.py
│   ├── run-multi-vqe-qrack-h2.py
│   └── vqe-results-3dviz.py
│
├── vqe-qml/                     # VQE with Quantum Machine Learning
│   ├── 2505.13525v1.pdf
│   ├── hermitian-matrices-pyqrack-oai.py
│   ├── hermitian-matrices-pyqrack.py
│   ├── hermitian-matrices-pyqrack.sh
│   ├── hermitian-matrices-qiskit.py
│   └── readme.md
│
├── vqe-qml-dissipate/           # VQE-QML with dissipation (barren plateau mitigation)
│   ├── dissipate-ancilaries.py
│   └── README.md
│
├── vqls/                        # Variational Quantum Linear Solver
│   ├── 1731241185612.pdf
│   ├── readme.md
│   └── vqls.py
│
└── weed/                        # Holographic/AdS-CFT experiments
    ├── feed_weed.py
    ├── generate_sparse_density_data.py
    ├── adscft-weed/
    │   ├── compile.sh
    │   ├── holographic_ingest
    │   ├── holographic_ingest_check
    │   ├── holographic_ingest_check.cpp
    │   ├── holographic_ingest.cpp
    │   ├── holographic_ingest.py
    │   └── README.md
    └── weed-training/
        ├── build-loop.sh
        ├── generate-data.py
        ├── run-loop.sh
        └── weed-training-loop.cpp
```

## Project Categories

### 1. Quantum Algorithms
- **[`hhl/`](hhl/)** - Harrow-Hassidim-Lloyd algorithm for solving linear systems
- **[`noisy-shors/`](noisy-shors/)** - Noisy implementations of Shor's factoring algorithm
- **[`qaoa/`](qaoa/)** - Quantum Approximate Optimization Algorithm for Max-Cut
- **[`vqe/`](vqe/)** - Variational Quantum Eigensolver implementations
- **[`vqe-qml/`](vqe-qml/)** - VQE combined with Quantum Machine Learning
- **[`vqe-qml-dissipate/`](vqe-qml-dissipate/)** - VQE-QML with dissipation to avoid barren plateaus
- **[`vqls/`](vqls/)** - Variational Quantum Linear Solver

### 2. Quantum Simulation & Benchmarking
- **[`ising/`](ising/)** - Ising model experiments (C, Python, OpenCL, PyQrack)
- **[`rcs/`](rcs/)** - Random Circuit Sampling benchmarks
- **[`rcsqbdd/`](rcsqbdd/)** - RCS with QBDD, MPS, and tensor simulations
- **[`graphs/`](graphs/)** - Visualization and graph generation scripts
- **[`hpc/`](hpc/)** - High-Performance Computing benchmarks

### 3. Quantum Chaos & Information Scrambling
- **[`qecho/`](qecho/)** - Out-of-Time-Order Correlators (OTOC) simulations
- **[`peaked/`](peaked/)** - Quantum Phase Estimation experiments

### 4. Quantum Machine Learning
- **[`qvml/`](qvml/)** - Quantum Variational Machine Learning demos
- **[`qllm-audit/`](qllm-audit/)** - Quantum LLM audit and security experiments

### 5. Specialized Experiments
- **[`agentics/`](agentics/)** - AI/LLM integration with Gemini API
- **[`er-epr/`](er-epr/)** - Entanglement-Robust EPR experiments
- **[`positron/`](positron/)** - Anti-qubit simulation (Tenet-inspired)
- **[`qhrf/`](qhrf/)** - Quantum Harmonic Resonance Frequency
- **[`qwscatter/`](qwscatter/)** - 3D quantum scatter plot visualization
- **[`qec/`](qec/)** - Quantum Error Correction with Mitiq
- **[`findafactor/`](findafactor/)** - Quantum factorization benchmark
- **[`weed/`](weed/)** - Holographic/AdS-CFT correspondence experiments

## Key Technologies Used

- **Qrack** - Quantum computing simulator framework
- **PyQrack** - Python bindings for Qrack
- **OpenCL** - GPU acceleration for quantum simulations
- **Mitiq** - Quantum error mitigation library
- **Tensor Networks** - Matrix Product States (MPS), Tensor Networks
- **QBDD** - Quantum Binary Decision Diagrams
- **Quantinuum** - Quantum hardware integration

## Credits

- **Qrack**: Dan Strano ([@unitaryfund/qrack](https://github.com/unitaryfund/qrack))
- **Bonsai**: Jeroen Bedorf ([@treecode/Bonsai](https://github.com/treecode/Bonsai))
- **ThereminQ**: Aryan Blaauw ([@twobombs](https://github.com/twobombs))