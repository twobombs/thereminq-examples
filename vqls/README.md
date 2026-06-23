# Vqls

This directory implements the Variational Quantum Linear Solver (VQLS) using Qiskit (`vqls.py`). VQLS circumvents the deep-circuit requirements of the exact HHL algorithm by using a parameterized `RealAmplitudes` ansatz and the SPSA optimizer to variationally project the quantum state towards the solution vector $x$ for the linear system $Ax = b$. It represents a NISQ-friendly alternative for linear algebra applications.

## Core Quantum Mechanical Concepts & ArXiv References

* [1909.05820] Variational Quantum Linear Solver (https://arxiv.org/abs/1909.05820)
* [2011.01938] Variational Quantum Algorithms (https://arxiv.org/abs/2011.01938)
