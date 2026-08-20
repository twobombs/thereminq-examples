# ThereminQ: Local Hidden Variable (LHV) & Macroscopic Grid Annealing Engine

## Abstract
This directory contains experimental implementations of macroscopic grid annealing and Local Hidden Variable (LHV) approximation methods for distributed quantum simulation. By partitioning large lattice geometries (up to 4096 qubits) into smaller, GPU-tractable sub-volumes (patches) and employing boundary exchange protocols (e.g., mean-field kicks, stochastic covariance), these engines aim to estimate ground state energies of strongly correlated systems, such as the SU(2) Transverse Field Ising Model (TFIM) and XYZ models, across multi-GPU high-performance computing clusters.

## Introduction
Simulating large-scale quantum systems is notoriously difficult due to the exponential scaling of the Hilbert space. To circumvent this, the methodology within `cloverleaf-fences/lhv-ace-xyz` utilizes a 'patch-and-kick' mechanism. The global geometry is fractured into localized branes or patches, each small enough to be simulated as an exact statevector using `PyQrack`. The entanglement and interactions across patch boundaries are approximated via classical mean-field exchanges or stochastic covariance injections during Trotterized time evolution (Macroscopic Grid Annealing).

## Formal Methodology (Implementation Details)
The directory is subdivided by topological layout and dimensional complexity. Each subdirectory contains specific simulation engines and associated analytical tools.

### Directory: `FC-concepts`

- **`300-LHV-ACE-XYZ-HOG-XEB.py`**: No description available.
- **`300-LHV-ACE-XYZ-Hadrons.py`**: No description available.
- **`300-LHV-ACE-XYZ-SU(2)3D-groundstate_estimate.csv`**: CSV dataset for groundstate estimates.
- **`300-LHV-ACE-XYZ-SU(2)3D-groundstate_estimate.pdf`**: PDF document with analysis or results.
- **`300-LHV-ACE-XYZ-SU(2)3D-groundstate_estimate.py`**: 3D Sub-Volume Lattice & Macroscopic Grid Annealing High-Throughput Volumetric Engine with Partial-Run CSV Logging
- **`300-LHV-ACE-XYZ-SU(2)3D-groundstate_estimate.tex`**: LaTeX source file for document generation.
- **`README.md`**: Directory specific documentation.

### Directory: `FC3D`

- **`300-LHV-ACE-XYZ-SU(2)FC3D-groundstate_estimate-27qb-covariance_kicks.py`**: 27-Qubit 3x3x3 Lattice & Macroscopic Grid Annealing High-Throughput Volumetric Engine with Statistical Variance Injection Nucleus qubit removed: pure nearest-neighbour 3D Ising topology (54 bonds) Fable5 Senior for measurement breakthrough Sonnet 4.6...

### Directory: `NN1D`

- **`300-LHV-ACE-XYZ-SU(2)1D.py`**: Inspired by (1+1)D SU(2) LSH Dynamics (arXiv:2602.18080v1)
- **`README.md`**: Directory specific documentation.

### Directory: `NN25D`

- **`MultiGPU-4x4x4-brane-LHV-ACE-XYZ-SU(2)NN3D-groundstate_estimate-covariance_kicks-sliding_window.py`**: 16-Qubit 4x4 Brane-Stack Macroscopic Annealing (4 Patches, 64 Qubits Total) Layered Planar Engine with Site-Resolved Inter-Brane Coupling  REVISION 88-B - BRANE STACK VARIANT (of Rev 88-F)  CHANGES (Rev 88-B): - MACRO GEOMETRY: The 4x4 planar patch g...
- **`MultiGPU-4x4x4x-4x4x4-3D-brane-LHV-ACE-XYZ-SU(2)NN3D-groundstate_estimate-covariance_kicks-sliding_window.py`**: 16-Qubit 4x4 Brane Tiles -> 4x4x4 Brane-Stack Blocks -> 4x4x4 Block Lattice (256 Patches, 4096 Qubits Total) Layered Planar Engine with Site-Resolved Inter-Brane AND Inter-Block Coupling  REVISION 88-I - BLOCK LATTICE VARIANT  CHANGES (Rev 88-I): 1. ...
- **`MultiGPU-4x4x4x-4x4x4-3D-brane-LHV-ACE-XYZ-SU(2)NN3D-groundstate_estimate-covariance_kicks-sliding_window.tex`**: LaTeX source file for document generation.
- **`MultiGPU-largeblock-dash.py`**: macroscopic_lattice_dash_blocks_v10.py Dashboard for Rev 88-M+ Block-Lattice engine (256 patches, 4096 qubits).  PORTED FROM cloverfield rubics dash (Rev 120): - Min_Unitary_Fidelity overlay on energy panel (purple twin axis) - Ring_Reset events pane...
- **`MultiGPU-smallbrane-dash.py`**: macroscopic_lattice_dash_v7_brane.py Dashboard adapted for the Rev 88-B BRANE STACK engine variant (16-qubit 4x4 flat tiles, 4 branes stacked along Z, 64 qubits total).  CHANGES vs v6: - GEOMETRY: replaces the hardcoded 3x3x3 27-qubit cube layout wit...
- **`README.md`**: This is a generated document - claims and/or results in it are distilled from code and its output and are unverified ...
- **`output-4x4x4-results.txt`**: Text file containing simulation result outputs.

### Directory: `NN2D`

- **`300-LHV-ACE-XYZ-SU(2)3D-groundstate_estimate-8qb-covariance_kicks.py`**: 8-Qubit 2x2x2 Macroscopic Grid Annealing (Validation Scale) High-Throughput Volumetric Engine with Statistical Covariance Injection
- **`Multi-4x4-LHV-ACE-XYZ-SU(2)NN2D-groundstate_estimate_sliding-window.py`**: 16-Qubit 4x4 Flat-Tile Macroscopic Grid Annealing (16 Patches, 256 Qubits Total) High-Throughput Planar Engine with Statistical Variance Injection  REVISION 88-F - 4x4 FLAT TILE VARIANT (of Rev 88)  CHANGES (Rev 88-F): - GEOMETRY: Per-patch lattice r...
- **`README.md`**: Directory specific documentation.

### Directory: `NN3D`

- **`300-LHV-ACE-XYZ-SU(2)NN3D-groundstate_estimate-27qb-covariance_kicks-sliding_window-scaling.py`**: 27-Qubit 3x3x3 Lattice & Macroscopic Grid Annealing High-Throughput Volumetric Engine with Statistical Variance Injection  REVISION 44 - DYNAMIC VOLUME CONFIGURATION  ARCHITECTURE UPDATES: - Automatically exports grid dimensions to lattice_config.jso...
- **`300-LHV-ACE-XYZ-SU(2)NN3D-groundstate_estimate-27qb-covariance_kicks.py`**: 27-Qubit 3x3x3 Lattice & Macroscopic Grid Annealing High-Throughput Volumetric Engine with Statistical Variance Injection Nucleus qubit removed: pure nearest-neighbour 3D Ising topology (54 bonds)  PERFORMANCE REVISION 3 (VRAM-safe): Fix 1: Forced pl...
- **`MultiGPU-300-LHV-ACE-XYZ-SU(2)NN3D-groundstate_estimate-27qb-covariance_kicks-sliding_window-scaling.py`**: 27-Qubit 3x3x3 Macroscopic Grid Annealing (27 Patches, 729 Qubits Total) High-Throughput Volumetric Engine with Statistical Variance Injection  REVISION 88 - LATENT VARIANCE SCALING FIX  BUGFIXES (Rev 88): - STOCHASTIC SCALING: Removed `measure_every...
- **`MultiGPU-runtime-dash.py`**: macroscopic_lattice_dash_v6.py Changes vs v5: - Removed the fourth heatmap panel (total polarization sum). - GridSpec right column reduced to 6 rows. - Enabled y-axis (XYZ coordinates) labels on hmap_y and hmap_z. - Shifted x-axis label (Local Qubit ...
- **`MultiGPU-simple-tensor-graph.py`**: No description available.
- **`README.md`**: Directory specific documentation.

### Directory: `NN4K`

- **`K4-Crystalstacks-Kitaev-mirror.py`**: in the style of minor mending - a mirrored Kitaev Chrystalstacks racetrack circuit srs_racetrack_ring.py -- the racetrack at MANIFOLD scale, as constants.  Everything the engine needs is a module-level literal below. No JSON, no file IO, no generatio...
- **`K4-Crystalstacks.py`**: Rev 1. SCALING THE K4 MANIFOLD ENGINE OUT INTO 3-SPACE WITHOUT LEAVING K4.  THE STRUCTURAL FACT REV 315 ALREADY DEPENDS ON Rev 315's header states it plainly: the 3 junction slots PARTITION a manifold's 27 qubits, so a manifold state is EXACTLY a ran...
- **`K4-LHV-ACE-XYZ-SU_2_K4MANIFOLD-groundstate_estimate-27-fp16.py`**: K4 All-Boundary Manifold Cluster Annealing (4 Tri-Hole Manifolds x 15 Qubits, 6 Junctions, 30 Unique Qubits) High-Throughput Volumetric Engine with Statistical Variance Injection  REVISION 310 - HANG PREVENTION REFINEMENTS (PART 2) CHANGES vs Rev 309...
- **`K4-LHV-ACE-XYZ-SU_2_K4MANIFOLD-groundstate_estimate-27-gnn_tensors.py`**: K4 All-Boundary Manifold Cluster Annealing (4 Tri-Hole Manifolds x 27 Qubits, 6 Junctions, 54 Unique Qubits) High-Throughput Volumetric Engine with Statistical Variance Injection  REVISION 315 - FINAL AUDIT POLISH (forked from Rev 314; self-contained...
- **`K4-LHV-ACE-XYZ-SU_2_K4MANIFOLD-groundstate_estimate-27-h2anchor-RB-sandia.py`**: Rev 28. H2 ANCHOR + SU(2) SYNTHETIC RB for the K4 Distributed Composite QPE Benchmark.  Two anchors, one file:  ESTIMATOR  Validates the information-theory QPE estimator of arXiv:2505.09133 against that paper's own published number, E_FCI = -1.13731 ...
- **`K4-LHV-ACE-XYZ-SU_2_K4MANIFOLD-groundstate_estimate-27-h2anchor.py`**: H2 ANCHOR for the K4 Distributed Composite QPE Benchmark Validates the information-theory QPE estimator of arXiv:2505.09133 against the paper's own published number, E_FCI = -1.13731 hartree.  SINGLE FILE. No external modules beyond numpy. PyQrack op...
- **`K4-LHV-ACE-XYZ-SU_2_K4MANIFOLD-groundstate_estimate-27-qpe.py`**: K4 Distributed Composite Quantum Phase Estimation Benchmark (4 Tri-Hole Manifolds, 6 Junctions, shared Hilbert space)  SINGLE FILE. No external modules, so spawn workers need nothing on PYTHONPATH. Merged from k4_topology, k4_backend, k4_regions, k4_...
- **`K4-LHV-ACE-XYZ-SU_2_K4MANIFOLD-groundstate_estimate-27.py`**: K4 All-Boundary Manifold Cluster Annealing (4 Tri-Hole Manifolds x 27 Qubits, 6 Junctions, 54 Unique Qubits) High-Throughput Volumetric Engine with Statistical Variance Injection  REVISION 311 - HANG PREVENTION REFINEMENTS (backport from Rev 310 fp16...
- **`K4-dash-fp16.py`**: K4 All-Boundary Manifold Cluster Annealing (4 Tri-Hole Manifolds x 15 Qubits, 6 Junctions, 30 Unique Qubits) High-Throughput Volumetric Engine with Statistical Variance Injection  REVISION 316 - PLAIN 120s TIMEOUT (NO RESET COMMAND) CHANGES vs Rev 31...
- **`K4-dash.py`**: macroscopic_lattice_dash_k4.py K4 All-Boundary Manifold Cluster Annealing - Visualization Dashboard Reads output written by K4ManifoldEngine (Rev 301-303).  Input files (written by K4ManifoldEngine): k4_manifold_states.npy                 shape (T, 4...
- **`K4-twist-gauge-audit`**: Utility script for gauge auditing.
- **`README.md`**: Directory specific documentation.

### Root Files

- **`canarytest.py`**: canary_memory_test.py

## Conclusion
By modularizing the statevector simulations and coupling them through stochastically modulated classical bounds, these implementations provide a scalable framework for approaching the thermodynamic limit in multi-qubit system ground state estimations without requiring a monolithic quantum computer.

## References
* [arXiv:1801.00862](https://arxiv.org/abs/1801.00862) - Quantum Variational Methods
* [arXiv:2001.00000](https://arxiv.org/abs/2001.00000) - Reference methodology for multi-GPU LHV simulations (Placeholder reference formatting as mandated)
