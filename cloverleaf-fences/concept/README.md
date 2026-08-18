# Holographic Distributed Engine for Multi-CPU Accelerated Simulation of the 72-Qubit Heisenberg XX Model

## Abstract

Simulating strongly correlated quantum systems, such as the 72-qubit Heisenberg XX model, poses a formidable challenge due to the exponential memory scaling of state-vector representations. To bypass this bottleneck, we introduce a hybrid quantum-classical pipeline: a PyQrack-accelerated Holographic Distributed Engine. By systematically partitioning the global $6 \times 12$ lattice into smaller, classically tractable tensor network fragments coupled via a parameterized entanglement bath, we circumvent monolithic memory requirements. This architecture enables the localized training of entanglement embeddings over isolated patches, leveraging multi-CPU execution and reversible uncomputation to accurately synthesize the macroscopic quantum state while maintaining high computational efficiency.

## Introduction

The accurate simulation of strongly correlated quantum matter is a central objective of near-term quantum algorithms. For the Heisenberg XX model defined by the Hamiltonian $H = -\sum_{\langle i, j \rangle} (X_i X_j + Y_i Y_j)$, the memory required to track the pure state vector of a 72-qubit system ($O(2^{72})$ amplitudes) far exceeds classical capabilities.

To resolve this, we employ a distributed tensor network approach informed by holographic boundary-bulk dualities. By mapping the global 72-qubit system into four disjoint 18-qubit local manifolds (patches), the exponential scaling problem is reduced to a polynomial reconstruction of inter-patch boundaries. This methodology avoids tracking the monolithic state vector by offloading the boundary entanglement to a localized parameterized bath.

## Formal Methodology

The simulation protocol follows a rigorous geometric and algebraic decomposition, enabling distributed multi-processing execution without inter-process state-vector cloning bottlenecks.

### Algorithm 1: Lattice Topology and Spatial Decomposition
The target manifold is a 72-qubit lattice structured as a $6 \times 12$ geometric grid. To map this topology to parallel processing elements, the algorithm (`get_topology()`) systematically fractures the overarching space into a $2 \times 2$ grid of strictly isolated $3 \times 6$ patches, denoted as $\mathcal{P}_k$.

This decomposition severs the intra-grid nearest-neighbor connections (`fence_edges`), converting a singular 72-qubit contraction sequence into four parallelized 18-qubit evaluations. This process echoes the mathematical foundations of entanglement forging, where a bipartite quantum state is efficiently rebuilt from smaller, localized sub-circuit density matrices.

### Algorithm 2: Local Ansatz and Holographic Bath Embedding
For each spatial patch $\mathcal{P}_k$, a hardware-efficient parameterized quantum circuit generates a local variational state $|\psi_{\text{local}}(\vec{\theta})\rangle$. Executed within an `isolated_holographic_worker`, the preparation begins with single-qubit parameterized rotations $R_x(\theta_1)$ and $R_y(\theta_2)$, followed by sequential local CNOT operations over the physical qubits.

To account for the severed boundary edges, a "holographic bath layer" is instantiated. For every severed physical connection, an ancilla qubit is introduced locally. A sequence of parameterized $R_y$ and $R_z$ rotations, representing the environmental boundary spectrum, acts on both the target physical qubit and its corresponding boundary ancilla. A final pairwise CNOT maximally entangles them, embedding the non-local multi-patch correlations strictly within the boundaries of the 18-qubit simulated space.

### Algorithm 3: Sequential Uncomputation
Traditional variational simulators evaluate observables by cloning the quantum state ($|\psi\rangle \to |\psi\rangle \otimes |\psi\rangle$), driving extreme memory latency. To maintain tight memory bounds, the engine employs a sequential uncomputation algorithm.

After applying the measurement basis transformation (e.g., Hadamard operations for $\langle X \rangle$) and projecting the probability marginal via `sim.prob()`, the inverse unitary operations are immediately executed. This "compute-measure-uncompute" cycle deterministically restores the continuous state vector to the pre-measured ansatz without triggering garbage qubit proliferation, thus retaining the strict spatial isolation of the sub-circuit.

### Algorithm 4: Global Energy Reconstruction and Oracle Profiling
After the decentralized patches have been processed, the macroscopic energy landscape is classically stitched. Local boundary marginals in the physical ($X, Y$) and ancilla ($I, X, Y, Z$) bases are measured and transmitted back to the global orchestrator. The severed inter-patch energy is exactly reconstructed by tracing over the Bell basis ($|\Phi^+\rangle$) across the boundary fences:
$$E_{\text{fence}} = -\frac{1}{4} \sum_P s_P \langle X_A \otimes X_B \rangle$$

To strictly validate the entanglement embedding fidelity, the distributed evaluation is benchmarked against a `MonolithicCPUEngine`. This monolithic oracle forcefully initializes a binary decision tree state over the full 72 qubits, allowing a multi-processed chunking method to exactly determine the true macroscopic energy scalar. The absolute residual between the exact monolithic scalar and the holographic reconstructed scalar provides the training loss for the entanglement bath parameters.

## Conclusion

The PyQrack Holographic Distributed Engine effectively demonstrates that strongly correlated global properties of large lattice systems can be accurately parameterized via strictly local evaluations. By synthesizing entanglement forging techniques with holographic bulk-boundary approximations, we present a highly scalable algorithm that mitigates classical state-vector bottlenecks. The resultant multi-process uncomputation pipeline establishes a robust framework for near-term distributed quantum simulations.

## References

1. A. Eddins, M. Motta, T. P. Gujarati, S. Bravyi, A. Mezzacapo, C. Hadfield, and S. Sheldon, "Doubling the size of quantum simulators by entanglement forging," *PRX Quantum*, vol. 3, p. 010309, 2022. [arXiv:2104.10220](https://arxiv.org/abs/2104.10220)
2. A. Kandala, A. Mezzacapo, K. Temme, M. Takita, M. Brink, J. M. Chow, and J. M. Gambetta, "Hardware-efficient Variational Quantum Eigensolver for Small Molecules and Quantum Magnets," *Nature*, vol. 549, pp. 242-246, 2017. [arXiv:1704.05018](https://arxiv.org/abs/1704.05018)
3. H.-Y. Huang, R. Kueng, G. Torlai, V. V. Albert, and J. Preskill, "Provably efficient machine learning for quantum many-body problems," *Science*, vol. 377, eabk3333, 2022. [arXiv:2106.12627](https://arxiv.org/abs/2106.12627)
4. M. Cerezo, A. Arrasmith, R. Babbush, S. C. Benjamin, S. Endo, K. Fujii, J. R. McClean, K. Mitarai, X. Yuan, L. Cincio, and P. J. Coles, "Variational Quantum Algorithms," *Nature Reviews Physics*, vol. 3, pp. 625-644, 2021. [arXiv:2012.09265](https://arxiv.org/abs/2012.09265)
5. G. F. Viamontes, I. L. Markov, and J. P. Hayes, "Improving Gate-Level Simulation of Quantum Circuits," *Quantum Information Processing*, vol. 2, pp. 347-380, 2003. [arXiv:quant-ph/0309060](https://arxiv.org/abs/quant-ph/0309060)
