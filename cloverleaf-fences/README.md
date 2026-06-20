# Cloverleaf Fences

This directory contains experimental implementations leveraging distributed quantum simulation, variational algorithms, and holographic tensor networks. The scripts demonstrate how deep-tech optimizations—such as OpenCL GPU offloading via Mesa/Rusticl and multi-process architectures—can be utilized to evaluate complex quantum mechanical phenomena and quantum chemistry problems on constrained hardware.

## Implementations

### Scaled Holographic Distributed Simulation (`4x4x6.py`)
This script expands the holographic embedding technique to a 96-qubit system (arranged in an 8x12 grid). The global grid is divided into a 2x3 grid of 4x4 patches (6 total patches), distributed across 6 independent GPUs.

It calculates an exact target global energy using a Monolithic CPU Oracle (QBDD) and approximates it using a Holographic Distributed Engine. This script serves as a robust demonstration of scaling the holographic boundary approach to larger quantum systems that are highly prohibitive to simulate directly without tensor network fracturing and boundary approximations.

### Holographic Distributed Simulation (`3x3x4.py`)
This script demonstrates an advanced topological embedding technique for large-scale circuit execution. A 36-qubit system (arranged in a 6x6 grid) is logically fractured into four 9-qubit (3x3) patches. Instead of simulating the full $2^{36}$ state space—which is computationally prohibitive for dense statevectors—the system is distributed across 4 independent GPUs.

The critical deep-tech innovation is the implementation of a **"Holographic Bath"** (using ancilla qubits) to emulate the cross-boundary entanglement (the "fences") between adjacent patches. The algorithm trains boundary parameters to minimize the energetic difference between the local isolated patches and a monolithic CPU oracle. This effectively utilizes boundary degrees of freedom to reconstruct the global entangled bulk state using only local, parallelized hardware resources.

### OpenCL-Accelerated VQE (`import-clifford_vqe_entangled-mesa.py`)
A highly parallelized Variational Quantum Eigensolver (VQE) pipeline designed to compute the ground-state energy of various molecular configurations. This implementation orchestrates PennyLane in conjunction with a custom PyQrack OpenCL backend via Mesa/Rusticl.

Key technical features include:
*   **Dynamic Hardware Polling:** Dynamically detects and distributes tasks across available GPUs using PyOpenCL and PyQrack environment variables.
*   **Fermionic Mapping:** Employs the OpenFermion library to compute molecular Hamiltonians and maps them to qubit representations using Jordan-Wigner transformations.
*   **Dynamic Ansatz Routing:** Automatically routes between hardware-efficient ansatze (Strongly Entangling Layers) and chemically-inspired Unitary Coupled Cluster with Singles and Doubles (UCCSD) ansatze depending on input parameters and basis set constraints.

## Core Quantum Mechanical Concepts & ArXiv References

The code in this repository relies on several foundational quantum computing and theoretical physics paradigms:

*   **Holographic Principle & Bulk-Boundary Correspondence:**
    Leveraging boundary degrees of freedom to reconstruct the bulk state, directly inspired by tensor network holography and AdS/CFT models. The `3x3x4.py` script attempts to recover global entanglement via local boundary baths.
    *   *Reference:* Swingle, B. (2009). "Entanglement Renormalization and Holography." [arXiv:0905.1317](https://arxiv.org/abs/0905.1317)
    *   *Reference:* Pastawski, F., Yoshida, B., Harlow, D., & Preskill, J. (2015). "Holographic quantum error-correcting codes." [arXiv:1503.06237](https://arxiv.org/abs/1503.06237)

*   **Variational Quantum Eigensolver (VQE):**
    A hybrid quantum-classical algorithmic approach to finding the lowest eigenvalue of a parameterized Hamiltonian, heavily utilized in near-term quantum chemistry.
    *   *Reference:* Peruzzo, A., et al. (2013). "A variational eigenvalue solver on a quantum processor." [arXiv:1304.3061](https://arxiv.org/abs/1304.3061)

*   **Unitary Coupled Cluster (UCCSD) & Fermionic Mappings:**
    An ansatz for VQE that maps classical coupled-cluster theory to a unitary operator suitable for quantum execution, capturing single and double electron excitations. The Jordan-Wigner transform is used to map Fermionic operators to Pauli strings.
    *   *Reference (UCCSD):* Romero, J., et al. (2017). "Strategies for quantum computing molecular energies using the unitary coupled cluster ansatz." [arXiv:1701.02691](https://arxiv.org/abs/1701.02691)
    *   *Reference (Fermionic to Qubit Mappings):* Seeley, J. T., Richard, M. J., & Love, P. J. (2012). "The Bravyi-Kitaev transformation for quantum computation of electronic structure." [arXiv:1208.5986](https://arxiv.org/abs/1208.5986)

*   **Circuit Cutting & Distributed Quantum Simulation:**
    Techniques for fracturing large entangling circuits into smaller, independent sub-circuits that can be executed on distributed or smaller quantum hardware, classically recombining the results via marginals or entanglement forging.
    *   *Reference:* Peng, W., et al. (2019). "Simulating Large Quantum Circuits on a Small Quantum Computer." [arXiv:1904.08690](https://arxiv.org/abs/1904.08690)
    *   *Reference:* Eddins, A., et al. (2021). "Doubling the size of quantum simulators by entanglement forging." [arXiv:2104.10220](https://arxiv.org/abs/2104.10220)
