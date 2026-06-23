# Vqe

This directory hosts multiple implementations of the Variational Quantum Eigensolver (VQE) targeting quantum chemistry (e.g., H2 molecule ground states). Scripts evaluate various ansatze (Hardware-Efficient, UCCSD) and backend accelerators (Cirq, PennyLane, Qrack, IBM Heron). By dynamically adjusting parameterized Pauli strings to minimize the Hamiltonian expectation value, these scripts benchmark the variational landscape across diverse classical HPC backends and simulated noise environments.

## Core Quantum Mechanical Concepts & ArXiv References

* [1304.3061] A variational eigenvalue solver on a quantum processor (https://arxiv.org/abs/1304.3061)
* [1701.02691] Strategies for quantum computing molecular energies using the unitary coupled cluster ansatz (https://arxiv.org/abs/1701.02691)
