# High-Performance Macroscopic Grid Annealing via LHV and ACE on K4 Manifold Clusters

**Authors:** [ThereminQ Collaboration]

## Abstract
Simulating the dynamics and ground-state properties of large-scale quantum many-body systems remains computationally prohibitive due to the exponential dimensionality of the Hilbert space. In this work, we present a highly scalable distributed quantum simulation engine based on the ThereminQ and PyQrack frameworks. Our architecture partitions a global interaction graph into localized sub-volumes, specifically utilizing a Cloverfield $K_4$ All-Boundary Manifold Cluster topology. By employing Local Hidden Variable (LHV) approximations combined with Automatic Circuit Elision (ACE), we enable bounded-error classical simulations of XYZ Pauli interactions that bypass traditional memory constraints. This framework relies on Schmidt decomposition to truncate cross-boundary entanglement while preserving intra-manifold coherence. Engineered for multi-GPU compute nodes, this `patch-and-kick` volumetric engine successfully estimates the ground states of thousands of qubits through massively parallel Trotterized time evolution and stochastic variance injection.

## Introduction
The accurate simulation of complex quantum systems, such as the Transverse Field Ising Model (TFIM) or generalized Heisenberg XYZ models, is essential for exploring quantum chemistry, material science, and high-energy physics. However, representing the full state vector of an $N$-qubit system requires tracking $2^N$ complex amplitudes, rendering exact classical simulations infeasible for $N > 50$ on modern hardware.

To overcome this, various approximation schemes have been proposed, including Tensor Networks (e.g., MPS, PEPS) and Quantum Monte Carlo methods. Our approach diverges by implementing Macroscopic Grid Annealing through a distributed sub-volume architecture. Instead of maintaining a global coherent state, we fracture the target geometry into localized "patches" (branes or tri-hole manifolds) that are small enough ($N \le 28$) to fit entirely within the memory of a single GPU as an exact statevector.

Interactions across the boundaries of these patches are approximated classically. We utilize a combination of tensor network patching, stochastic variance injection, and Local Hidden Variable (LHV) modeling to communicate correlations between adjacent sub-volumes without violating the memory constraints of individual compute nodes. This methodology allows us to synthesize global ground states from purely local dynamics and iterative classical boundary exchanges.

## Theoretical Framework (Methodology)

### Cloverfield $K_4$ Manifold Cluster Geometry
Our primary topological structure is the Cloverfield $K_4$ Manifold Cluster. The complete graph $K_4$ consists of four nodes, where every node is connected to every other node (six edges total). In our volumetric engine, each "node" is a Tri-Hole Manifold containing $27$ qubits.

The $27$ qubits within a single manifold are partitioned into three identical "junctions" (or portals) of $9$ qubits each ($3$ corner qubits and $6$ mid-edge qubits). These junctions serve as the interface for inter-manifold coupling. Because the $3$ junction slots exactly partition the $27$ qubits, a manifold state can be perfectly represented as a rank-3 tensor $T_{ijk}$ with leg dimension $d = 2^9 = 512$.

The global simulation connects four such tensors across six copy-tensor bonds, forming a $K_4$ topology with zero structural loss.

### Automatic Circuit Elision (ACE) and Schmidt Truncation
To manage entanglement growth across the junctions, we employ Automatic Circuit Elision (ACE) implemented via `PyQrack`. ACE dynamically simplifies quantum circuits by eliding gates that operate on separable or weakly entangled subsystems.

When a manifold state must be communicated across a GPU boundary, transmitting a full 27-qubit statevector ($2^{27}$ amplitudes, $\approx 1$ GB in single-precision floating point) is highly inefficient and creates significant IPC bottlenecks. Instead, we perform in-worker Schmidt decomposition on the state tensor:

$$ T_{ijk} \approx \sum_{\alpha=1}^{\chi} s_\alpha U_{i\alpha} \otimes V_{j\alpha} \otimes W_{k\alpha} $$

where $\chi$ is the truncation dimension (typically $\chi \ll 512$). By transmitting only the top $\chi$ singular values and corresponding basis vectors, we drastically reduce the inter-process communication overhead. This process is governed by configuration limits such as `QRACK_MAX_PAGING_QB`, which defines the maximum subsystem size allowed to be maintained cohesively before Schmidt truncation is enforced by the PyQrack backend to page state data out of GPU VRAM. This is critical for scaling approximate simulation via tunable fidelity reduction [1].

### Local Hidden Variable (LHV) Approximation for XYZ Models
The Hamiltonian for the XYZ Heisenberg model with local fields is given by:

$$ H = - \sum_{\langle i, j \rangle} \left( J_x X_i X_j + J_y Y_i Y_j + J_z Z_i Z_j \right) - \sum_i \left( h_x X_i + h_y Y_i + h_z Z_i \right) $$

For qubits residing within the same manifold, these interactions are applied exactly using two-qubit unitary gates during Trotterized time evolution. However, for qubits $a$ and $b$ spanning a junction between manifolds $A$ and $B$, applying an exact entangling gate is impossible.

Instead, we approximate the inter-manifold bond using a Local Hidden Variable (LHV) mean-field approach. Following the theoretical basis for LHV models in reproducing quantum correlations [2], and leveraging advanced classical simulations of multi-qubit non-local correlations [3], we compute the expectation values (local fields) of the boundary qubits on side $A$ and transmit them to side $B$ as classical information (a "kick").

The effective kick Hamiltonian applied to boundary qubit $b$ in manifold $B$ due to the state of manifold $A$ is:

$$ H_{kick}^{(b)} = -g \left( \langle X_a \rangle X_b + \langle Y_a \rangle Y_b + \langle Z_a \rangle Z_b \right) $$

where $g$ is the coupling strength. This classical boundary exchange prevents entanglement generation across the junction while effectively steering the independent local statevectors toward a globally consistent minimum energy state.

## Implementation Details

The core implementations reside within `cloverleaf-fences/lhv-ace-xyz`. The distributed engine architecture is organized as follows:

1. **Statevector Initialization:** Independent Python processes are spawned for each manifold. Each process initializes an exact $27$-qubit statevector on a dedicated GPU using `QrackSimulator`.
2. **Trotterized Evolution:** The time evolution operator $e^{-i H dt}$ is decomposed using Strang splitting. Single-qubit field rotations ($R_x, R_y, R_z$) and intra-manifold entangling gates (e.g., $ZZ$ rotations derived from $MCX$ and $R_z$) are applied sequentially.
3. **Measurement and Tomography:** At specified intervals, each GPU worker calculates the expectation values $\langle X \rangle, \langle Y \rangle, \langle Z \rangle$ for all local qubits. If `bp` (Belief Propagation) mode is active, the worker also performs the SVD to export the truncated tensor representation.
4. **IPC Synchronization:** The measurement data is sent to a Master Orchestrator process via Python `multiprocessing.Pipe`.
5. **Boundary Exchange:** The Master calculates the necessary corrective "kicks" based on the boundary mode (Mean-field, Graph Neural Network, or Belief Propagation). For mean-field, it calculates the Bloch vector rotations required to align the boundary qubits.
6. **State Update:** The calculated kick angles are sent back to the respective GPU workers, which apply them as single-qubit rotations before beginning the next Trotter step.

## Benchmarking & Multi-GPU Scaling

The framework is explicitly designed for massive parallelism across multi-GPU compute nodes. The primary scaling bottlenecks in distributed quantum simulation are VRAM capacity and inter-node communication latency.

*   **VRAM Isolation:** Multi-GPU distribution is enforced by pinning individual `multiprocessing` worker processes to specific physical GPUs. This is achieved by exporting OpenCL environment variables (`QRACK_OCL_DEFAULT_DEVICE`, `QRACK_QPAGER_DEVICES`, and `QRACK_QUNITMULTI_DEVICES`) prior to initializing the `QrackSimulator` instance. Memory allocation is strictly capped (e.g., 2 GB per manifold) using `QRACK_MAX_ALLOC_MB`.
*   **NUMA-Aware Execution:** To minimize PCIe bus saturation, worker processes are mapped to CPU cores belonging to the same NUMA node as the assigned GPU, ensuring that IPC `Pipe` transfers and local SVD computations do not traverse the slower QPI links between CPU sockets.
*   **Zero-Padded Output Tracking:** To maintain numerical stability and consistent checkpointing during scaling tests, output artifacts (like `k4_manifold_states.npy` and CSV logs) utilize zero-padded time steps and strictly typed, pre-allocated numpy arrays to prevent memory fragmentation and I/O stalls during prolonged annealing runs.

## References

[1] D. Strano et al., "Exact and approximate simulation of large quantum circuits on a single GPU," [arXiv:2304.14969](https://arxiv.org/abs/2304.14969), 2023.

[2] D. Kaszlikowski et al., "A local hidden variable model of quantum correlation," [arXiv:quant-ph/9905018](https://arxiv.org/abs/quant-ph/9905018), 1999.

[3] N. Miklin et al., "Simulations of quantum nonlocality with local negative bits," [arXiv:2106.07945](https://arxiv.org/abs/2106.07945), 2021.
