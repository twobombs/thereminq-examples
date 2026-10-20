# High-Throughput Volumetric Quantum Simulation on K4 All-Boundary Manifold Clusters via Automatic Circuit Elision

**Abstract:** We present a robust computational framework for the simulation of highly entangled, multi-component topological quantum systems. This work introduces the $K_4$ All-Boundary Manifold Cluster Annealing Engine, a classical simulation architecture mapping a Transverse Field Ising Model (TFIM) with XYZ Nearest-Neighbor (NN) interactions onto a completely bounded $K_4$ topology. To surmount the exponential memory costs of simulating heavily connected geometric structures, we leverage a multi-GPU environment mediated by an Automatic Circuit Elision (ACE) backend. The framework implements Local Hidden Variable (LHV) approximations alongside advanced pluggable boundary modes—including Graph Neural Networks (GNN) and loopy Belief Propagation (BP) for truncated Tensor Networks—to stitch disparate manifold components. This architecture severely reduces inter-process communication constraints, retaining loop structures and simulating up to 108 highly entangled qubits efficiently.

## 1. Introduction

The near-term trajectory of quantum computing is defined by Noisy Intermediate-Scale Quantum (NISQ) devices, burdened by stringent limitations on coherence and connectivity. In exploring complex physical systems, mapping lattice models to restricted topologies inevitably creates profound routing overheads. To circumvent this, "Classical Quantum Advantage" seeks to exploit classical High-Performance Computing (HPC) hardware to probe intricate topological geometries, establishing analytical foundations before physical deployment.

Simulating non-trivial manifolds with extensive boundary interactions faces immediate bottlenecks when employing exact statevector simulation. The memory complexity $\mathcal{O}(2^N)$ fundamentally caps single-node monolithic statevectors. In response, this work establishes the ThereminQ pipeline for a specialized $K_4$ complete graph topology. The system is modeled as a set of four 27-qubit sub-volumes comprising exclusively of boundary qubits ("Cloverleaf Fences"). By avoiding bulk qubits entirely, we intensify the inter-manifold edge phenomena. The resulting computational complexity is managed via Automatic Circuit Elision (ACE) [1], distributing independent manifolds across parallel GPUs and coupling them iteratively through synthesized boundary conditions.

## 2. Theoretical Framework

The governing dynamics of the Cloverleaf Fences rely on a modified continuous-time Hamiltonian with XYZ Pauli nearest-neighbor interactions across the manifold geometry:
$$ H(t) = - \sum_{\langle i, j \rangle} J(t) (\sigma_i^x \sigma_j^x + \sigma_i^y \sigma_j^y + \sigma_i^z \sigma_j^z) - \sum_i h_x(t) \sigma_i^x - \sum_i h_z(t) \sigma_i^z $$
We integrate this Hamiltonian using a second-order Strang splitting (Suzuki-Trotter expansion) to advance the system dynamically through time $t$.

### The $K_4$ Manifold Topology & Cloverleaf Fences
The topology features four disjoint 27-qubit manifolds connected along a $K_4$ complete graph structure. Each 27-qubit sub-volume is a "tri-hole manifold" formed of 3 portals (rims) containing 18 qubits each (3 corner and 6 mid-edge qubits). Every qubit is intrinsically a boundary component, localized exactly between two portals. Across the four manifolds, six inter-manifold junctions exist, collectively coupling 54 unique junction qubits.

### Automatic Circuit Elision (ACE) & Local Hidden Variables (LHV)
Rather than contracting a global 108-qubit state, the ACE framework "cuts" the circuit along the topological boundaries. The inter-manifold operations are replaced with Local Hidden Variable (LHV) approximations and classical channels. Stochastic variance (shot noise) is injected to emulate projective quantum measurements, scaling proportionally with $\sqrt{\Delta t / N_{\text{shots}}}$.

When updating states at the boundary, the local updates $\Delta w$ take the form of Pauli rotations parametrized by the hidden variables. To preserve holonomy and loop-momentum spectra, we apply a discrete Möbius half-twist at junctions, which acts as a parity constraint observable during simulated evolution [3].

## 3. Algorithmic & Code Architecture

The implementation in the `NN4K` directory directly implements the theoretical framework into a highly-parallel multi-worker ecosystem based on PyQrack.

### Core Components
*   **`K4ManifoldEngine` (Orchestrator):** Defined primarily in `K4-LHV-ACE-XYZ-SU_2_K4MANIFOLD-groundstate_estimate-27-gnn_tensors.py`, it governs the high-level annealing schedule, boundary mode selection, and IPC (Inter-Process Communication) coordination.
*   **`gpu_worker_process`:** A dedicated GPU worker handling the exact statevector evolution of a single 27-qubit manifold using `PyQrack`. The Trotterization applies exact 2-qubit gates internally while managing bounded communication windows (`GATHER_TIMEOUT_S`) with the orchestrator to prevent simulation stalling.
*   **Pluggable Boundary Managers:** These classes dictate how boundary elision interactions manifest:
    *   `MeanFieldBoundary`: A direct bond-resolved mean-field baseline ($\chi = 1$). It communicates explicit Bloch vector rotation angles across junctions.
    *   `GNNBoundary`: Instantiates an `EdgeMLP` neural architecture. It utilizes PyTorch/NumPy-based inference to predict residual boundary dynamics from local operator averages, enriching standard mean-field interactions [4].
    *   `BPBoundary` (Belief Propagation): Enables a truncated Tensor Network model via loopy belief propagation [2, 5, 6] with bond dimension $\chi > 1$. Employs the `K4BoundaryBP` object for resolving messages.
*   **GPU-Bound Schmidt Decomposition:** Within `BPBoundary`, the method `leg_isometries` directly computes the top-$\chi$ Schmidt bases on the GPU worker. Consequently, `truncate_tensor` collapses the payload locally, allowing the system to transmit ~100 KB payload tensors per worker rather than raw multi-gigabyte statevectors.

### Dashboards and Tooling
The output generated by the Volumetric Engine is visualized utilizing interactive interfaces like `K4-dash.py` and `K4-dash-fp16.py`, enabling real-time topological phase observation and loop momentum Fourier spectrum analysis.

## 4. Execution & Benchmarking Expectations

The simulation engine is inherently structured for distributed execution across multi-GPU environments.
1.  **Hardware Requirements:** The optimal run setup scales efficiently up to 4 parallel GPUs, dedicating a single GPU worker to each 27-qubit $K_4$ manifold component. Environment variables like `QRACK_OCL_DEFAULT_DEVICE` are automatically orchestrated to bind CPU worker threads to distinct GPU indices.
2.  **Execution Command:**
    ```bash
    python3 K4-LHV-ACE-XYZ-SU_2_K4MANIFOLD-groundstate_estimate-27-gnn_tensors.py
    ```
3.  **Benchmarking & Output Details:** Upon execution, the master process generates periodic data structures reflecting the boundary metrics:
    *   `k4_manifold_states.npy`: Full history of subsystem metrics (Total, Bulk, Junction Energies).
    *   `k4_exact_ground_state_energy_curve.csv`: A unified CSV detailing Total / Bulk / Junction energies versus Trotter time step $t$.
    *   `k4_loop_momentum_spectrum.csv`: Loop momentum amplitudes corresponding to the Discrete Fourier Transform over the $K_4$ racetrack junctions.

    Metrics like the Swap Asymmetry (RMS variance between symmetrically paired manifolds) and Bond Consistency Error serve as primary diagnostic invariants.

## 5. Verified References

[1] Piveteau, C., & Sutter, D. (2022). Circuit Knitting With Classical Communication. [arXiv:2205.00016](https://arxiv.org/abs/2205.00016)

[2] Orús, R. (2013). A practical introduction to tensor networks: Matrix product states and projected entangled pair states. [arXiv:1306.2164](https://arxiv.org/abs/1306.2164)

[3] Brown, B. J., et al. (2018). Symmetry defects and their application to topological quantum computing. [arXiv:1811.02143](https://arxiv.org/abs/1811.02143)

[4] Daskin, A. (2024). A unifying primary framework for quantum graph neural networks from quantum graph states. [arXiv:2402.13001](https://arxiv.org/abs/2402.13001)

[5] Tindall, J. & Fishman, M. T. (2023). Gauging tensor networks with belief propagation. [arXiv:2306.17837](https://arxiv.org/abs/2306.17837)

[6] Alkabetz, R. & Arad, I. (2020). Tensor Networks contraction and the Belief Propagation algorithm. [arXiv:2008.04433](https://arxiv.org/abs/2008.04433)