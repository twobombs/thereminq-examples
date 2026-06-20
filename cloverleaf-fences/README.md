# Cloverleaf Fences

This directory contains experimental implementations leveraging distributed quantum simulation, variational algorithms, and holographic tensor networks. The scripts demonstrate how deep tech optimizations (like Rusticl/OpenCL GPU offloading and multi-process architecture) can be utilized to evaluate complex quantum mechanical phenomena and quantum chemistry problems on constrained hardware.

## Implementations

### Holographic Distributed Simulation (`3x3x4.py`)
This script demonstrates an advanced topological embedding technique. A 36-qubit system (arranged in a 6x6 grid) is logically fractured into four 9-qubit (3x3) patches. Instead of simulating the full 36-qubit space—which is computationally expensive—the system is distributed across 4 independent GPUs. The critical innovation is the use of a "Holographic Bath" (ancilla qubits) to emulate the cross-boundary entanglement (the "fences") between adjacent patches. By minimizing the energetic difference between the local patches and a monolithic CPU oracle, this code effectively trains the boundary parameters to recover the global entangled state using only local resources.

### OpenCL-Accelerated VQE (`import-clifford_vqe_entangled-mesa.py`)
A highly parallelized Variational Quantum Eigensolver (VQE) designed to compute the ground-state energy of various molecular configurations. This implementation utilizes PennyLane in conjunction with a custom PyQrack OpenCL backend via Mesa/Rusticl, dynamically polling and distributing tasks across available GPUs. It dynamically routes between hardware-efficient ansatze and chemically-inspired Unitary Coupled Cluster (UCCSD) ansatze depending on the input parameters, handling fermionic-to-qubit mapping via Jordan-Wigner transformations.

## Core Quantum Mechanical Concepts & ArXiv References

The code in this repository relies on several foundational quantum computing and theoretical physics paradigms:

*   **Holographic Principle & Bulk-Boundary Correspondence:**
    Leveraging boundary degrees of freedom to reconstruct the bulk state, directly inspired by tensor network holography and AdS/CFT models.
    *   *Reference:* Swingle, B. (2009). "Entanglement Renormalization and Holography." [arXiv:0905.1317](https://arxiv.org/abs/0905.1317)
    *   *Reference:* Pastawski, F., Yoshida, B., Harlow, D., & Preskill, J. (2015). "Holographic quantum error-correcting codes." [arXiv:1503.06237](https://arxiv.org/abs/1503.06237)

*   **Variational Quantum Eigensolver (VQE):**
    A hybrid quantum-classical algorithmic approach to finding the lowest eigenvalue of a parameterized Hamiltonian, heavily utilized in near-term quantum chemistry.
    *   *Reference:* Peruzzo, A., et al. (2013). "A variational eigenvalue solver on a quantum processor." [arXiv:1304.3061](https://arxiv.org/abs/1304.3061)

*   **Unitary Coupled Cluster (UCCSD):**
    An ansatz for VQE that maps classical coupled-cluster theory to a unitary operator suitable for quantum execution, capturing single and double electron excitations.
    *   *Reference:* Romero, J., et al. (2017). "Strategies for quantum computing molecular energies using the unitary coupled cluster ansatz." [arXiv:1701.02691](https://arxiv.org/abs/1701.02691)

*   **Circuit Cutting & Distributed Quantum Simulation:**
    Techniques for fracturing large entangling circuits into smaller, independent sub-circuits that can be executed on distributed or smaller quantum hardware, classically recombining the results.
    *   *Reference:* Peng, W., et al. (2019). "Simulating Large Quantum Circuits on a Small Quantum Computer." [arXiv:1904.08690](https://arxiv.org/abs/1904.08690)
