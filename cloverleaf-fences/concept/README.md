# High-Throughput Volumetric Engine for Distributed Adiabatic Ground State Estimation of 3D Lattices via a Patch-and-Kick Mechanism

## Abstract

The classical simulation of large-scale, strongly correlated quantum systems is profoundly limited by the exponential scaling of the Hilbert space, generally bounded as $\mathcal{O}(2^N)$. Exact state-vector evaluation of a 3D Transverse-Field Ising Model (TFIM) rapidly becomes intractable for standard single-node architectures. In this work, we present a High-Throughput Volumetric Engine designed to circumvent these memory and computational constraints. By physically partitioning the global SU(2) 3D lattice into isolated sub-volumes (patches), we enable native simulation on multi-GPU architectures using PyQrack. To account for the severed inter-patch entanglement, we introduce an iterative "patch-and-kick" mechanism. This approach approximates boundary interactions via classical macro-spin exchanges—kicks—acting on the boundary qubits. This framework, inspired by mean-field decouplings and holographic traversable wormhole protocols, provides a highly scalable classical proxy for quantum correlations, unlocking multi-node quantum circuit simulation capabilities.

## 1. Introduction

The quest to find the ground state of generic local Hamiltonians is a central problem in condensed matter physics, quantum chemistry, and computational complexity, often classified as QMA-hard. Quantum Adiabatic Evolution (QAE) provides a physical framework to reach these ground states by slowly evolving the system from a trivial, easily prepared state to the target Hamiltonian's ground state [1]. However, on near-term hardware or full-state classical simulators, system size is strictly bottlenecked by qubit counts and exponentially growing memory requirements.

To address these constraints in simulating a 3D Transverse-Field Ising Model (TFIM) coupled to a longitudinal field, we introduce a distributed multi-GPU architecture. Drawing connections between condensed matter models and holographic dualities—such as Sachdev-Ye-Kitaev (SYK) models—we utilize classical multi-qubit interactions as channels for information exchange across boundaries [2]. Our ThereminQ High-Performance Computing (HPC) Volumetric Engine physically partitions a 3D lattice into smaller SU(2) sub-volumes, simulating intra-patch entanglement exactly while approximating inter-patch entanglement dynamically using a mean-field surrogate, enabling large-scale adiabatic annealing simulations.

## 2. Formal Methodology

We consider the task of determining the ground state properties of a 3D TFIM coupled to a longitudinal field, governed by the Hamiltonian:
$$ H = -J \sum_{\langle i, j \rangle} Z_i Z_j - h_x \sum_i X_i - h_z \sum_i Z_i $$
where $X_i, Z_i$ are Pauli matrices, $J$ is the exchange coupling, and $h_x, h_z$ are the transverse and longitudinal magnetic fields, respectively.

### 2.1 Lattice Topology and Volumetric Partitioning

To distribute the computational load, the global 3D lattice is spatially decomposed into isolated sub-volumes or patches, $P_k$, with dimensions $L_x \times L_y \times L_z$ (e.g., $3 \times 3 \times 2 = 18$ qubits). Each patch is assigned to an independent GPU worker maintaining a persistent PyQrack simulator state.

Because the total Hamiltonian is strictly nonseparable, the inter-patch boundary Hamiltonian $H_{\text{boundary}} = -g_{\text{face}} \sum_{i \in \partial P_a, j \in \partial P_b} \sigma_i \cdot \sigma_j$ is severed. This spatial fragmentation transforms an exponentially complex global contraction into parallelized, tractable sub-volume evaluations.

### 2.2 Adiabatic Evolution and Strang Splitting

To reach the ground state, we employ an adiabatic schedule parameterized by $s(t) = t/T \in [0, 1]$, interpolating between an initial Hamiltonian $H_0$ (driven by a strong transverse field $h_x(0) \gg 1$) and the target Hamiltonian $H_{\text{target}}$:
$$ H(s) = (1 - s)H_0 + sH_{\text{target}} $$

The distributed simulation uses a numerical Strang-splitting integration to perform the evolution. The algorithm alternates between exact unitary evolution within the isolated patches (intra-patch entanglement generation) and classical boundary interactions.

### 2.3 Patch-and-Kick Mean-Field Boundary Coupling

To approximate the severed non-local quantum correlations, we utilize a mean-field surrogate. For a boundary qubit $i \in \partial P_a$, the interaction with the adjacent face $\partial P_b$ is replaced by a coupling to the macroscopic magnetization of $\partial P_b$:
$$ \langle \sigma_{\partial P_b} \rangle = \frac{1}{|\partial P_b|} \sum_{j \in \partial P_b} \langle \sigma_j \rangle $$

The boundary interaction energy is thus minimized via iterative classical kicks on the boundary qubits. These kicks, parameterized by the coupling constant $g_{\text{face}}$ and the macroscopic magnetization of the adjacent face, apply classical rotational updates to the boundary qubits. This "patch-and-kick" mechanism acts as a classical proxy for quantum entanglement [3], dynamically injecting boundary conditions into the simulated patches without requiring coherent quantum links.

## 3. Conclusion

The "patch-and-kick" High-Throughput Volumetric Engine provides a robust numerical framework for scaling the classical simulation of quantum systems. By partitioning a global 3D lattice into tractable sub-volumes and utilizing classical macro-spin exchanges to approximate boundary interactions, we bypass the $O(2^N)$ memory barrier for calculating ground-state energies. This distributed architecture, synthesizing exact intra-patch evolution with mean-field boundary proxies, offers a critical pathway for the exascale evaluation of complex quantum many-body systems.

## 4. References

1. Chakrabarti, B. K. et al. "Tranverse Ising Model, Glass and Quantum Annealing", *Quantum Annealing and Related Optimization Methods* (2005). [arXiv:cond-mat/0312611](https://arxiv.org/abs/cond-mat/0312611)
2. Lykken, J. D. et al. "Long-range wormhole teleportation" (2024). [arXiv:2405.07876](https://arxiv.org/abs/2405.07876)
3. Matsuura, S. et al. "Mean Field Analysis of Quantum Annealing Correction", *Phys. Rev. Lett.* 116, 220501 (2016). [arXiv:1510.07709](https://arxiv.org/abs/1510.07709)
