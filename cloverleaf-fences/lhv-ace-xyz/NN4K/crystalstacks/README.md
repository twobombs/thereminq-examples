# High-Performance Scaling of the K4 Manifold Engine onto the Kitaev-Hyperoctagon (srs) Lattice

## Abstract
This repository contains the architecture and implementation for scaling the $K_4$ manifold quantum simulation engine into 3-space via the maximal abelian cover of $K_4$, formally identified as the hyperoctagon lattice (the Laves' graph of girth ten, or srs net). The primary challenge in tensor network simulations of complete graph topologies like $K_4$ is the accumulation of loop error during Belief Propagation (BP), caused by short cycles (girth 3). By unrolling the cover into a girth-10 geometry, this framework heavily suppresses loop error while perfectly preserving the structural identity of the 27-qubit manifold (where degree-3 uniform legs cleanly partition without loss). The engine utilizes an exact free-fermion anchor grounded in the Kitaev model, resolves uniform gauge sectors, and correctly handles boundary truncation by calculating exact entanglement Hamiltonians (seam repair) for the reduced density matrices. The repository orchestrates a heterogeneous compute topology, employing branch-divergent constraint programming (CP-SAT) for exact lattice tiling alongside bare-metal OpenCL for massively parallel uniform sampling of 27-site blocks, achieving significant classical simulation speedups on multi-GPU nodes.

## I. Introduction
Simulating large-scale quantum systems on classical hardware using tensor networks fundamentally contends with the trade-off between structural fidelity and computational tractability. The previous iteration of the $K_4$ manifold engine modeled 27 qubits as a rank-3 tensor with a uniform leg dimension of $2^9 = 512$, providing a symmetric, computationally stable building block. However, directly projecting this topology into a standard 3D cubic lattice breaks the arithmetic partitioning ($27$ qubits split across $6$ legs requires asymmetric $5/5/5/4/4/4$ divisions), destroying uniform Schmidt basis sizes and introducing multiple code paths.

Instead, this implementation embeds the $K_4$ engine into the 3-dimensional maximal abelian cover of $K_4$: the hyperoctagon lattice. Because it is a degree-3 lattice, the arithmetic perfectly matches the 27-qubit structure. More critically, moving from the $K_4$ graph to the hyperoctagon lattice increases the graph girth from 3 to 10. The high girth suppresses the dominant source of error in Belief Propagation—short-cycle loop error—which cannot be mitigated merely by increasing the Schmidt rank $\chi$.

## II. Methodology & Implementation

The architecture implements a rigorous, high-throughput pipeline distributed across several distinct solvers:

*   **Tensor Leg Uniformity and Topology:** The core tensor network (implemented in `K4-Crystalstacks.py`) maintains the rank-3 structure. Four manifolds of 27 qubits exactly tile a primitive cell of 108 qubits. By utilizing the Laves' graph of girth ten, the structural identity is preserved while expanding into 3-space.
*   **Exact Free-Fermion Anchor:** `K4-Chrystalstacks-Kitaev-single.py` provides a computationally exact anchor using the Kitaev model on the hyperoctagon lattice. The uniform gauge sector (where elementary 10-loops carry $W = -1$) is explicitly resolved as Lieb-optimal, allowing direct comparison of ground state energies.
*   **Seam Repair via Entanglement Hamiltonians:** When extracting a 27-site block from the bulk lattice, naive truncation destroys boundary correlations. The engine treats the boundary seam as an explicit physical object. It calculates the exact reduced state of the block for a Gaussian state and computes the entanglement Hamiltonian $A_{\text{ent}}$. The seam operator $\Sigma = A_{\text{ent}} - A_{\text{internal}}$ perfectly captures the dropped border bonds without introducing ancillary qubits.
*   **Heterogeneous Solver Orchestration:** Tiling an $L=6$ hyperoctagon lattice into 32 pairwise-disjoint blocks of 27 sites (forming 10-loops with 17-site trees) requires exploring an enormous combinatorial space.
    *   **CP-SAT Search (`K4-32x27-tilecp.py`):** Utilizes Google's OR-Tools for branch-divergent, sequential pointer-chasing search on CPU architectures. It implements a fully reified Large Neighborhood Search (LNS) seeded with greedy warm starts.
    *   **OpenCL Candidate Search (`K4-32x27-tileocl.py`):** For massive broad-phase exploration, the problem state is compressed into cache-resident memory (loop masks, adjacency lists). Embarrassingly parallel, bitwise OpenCL kernels evaluate billions of independent tiling candidates across multi-GPU environments.
    *   **Multi-GPU Orchestration (`K4-32x27-tile.sh`):** A shell supervisor that manages node topology, CPU/GPU process isolation, and serial cache warming, preventing deadlocks and racing dependencies across complex multi-die systems (e.g., EPYC and MI50 GPUs).

## III. Experimental Setup & Reproducibility

Execution requires a Linux environment (Ubuntu recommended) with functional OpenCL ICDs (e.g., rustiCL, ROCm, or PoCL for CPU fallback) and Python 3.

### Dependencies
The Python environment requires:
*   `ortools` (for the CP-SAT tiling model)
*   `pyopencl` (for bare-metal GPU kernels)
*   `numpy`

Install via `pip`:
```bash
pip install ortools pyopencl numpy
```

### Execution
The tiling solvers are orchestrated by a bash supervisor script which automatically allocates EPYC cores and OpenCL devices.

To preview the worker split without executing the workloads:
```bash
./K4-32x27-tile.sh --plan
```

To run the full suite across all available platforms (utilizing standard $N$-proc and total device discovery):
```bash
./K4-32x27-tile.sh
```

To limit execution time (e.g., 3600 seconds) or restrict to a specific OpenCL device (e.g., Platform 0, Device 0):
```bash
SECS=3600 DEVICES=0:0 ./K4-32x27-tile.sh
```

## References

1. Hermanns, M., & Trebst, S. (2014). Quantum spin liquid with a Majorana Fermi surface on the three-dimensional hyperoctagon lattice. [arXiv:1401.7678](https://arxiv.org/abs/1401.7678)
2. Mishmash, R. V., et al. (2017). Finite-temperature phase transition to a Kitaev spin liquid phase on a hyperoctagon lattice: A large-scale quantum Monte Carlo study. [arXiv:1706.05057](https://arxiv.org/abs/1706.05057)
