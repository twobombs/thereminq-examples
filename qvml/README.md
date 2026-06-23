# Qvml

This directory contains implementations of Quantum Variational Machine Learning (QVML). Scripts like `qvml.py` map multi-qubit parameterized quantum circuits to tensor networks using `quimb`. By solving for optimal contraction paths (sometimes mapped to equivalent Traveling Salesperson Problems via PyQrack's Ising solver), these tools efficiently simulate gradients and evaluate the cost landscape of quantum neural networks for classification or regression tasks.

## Core Quantum Mechanical Concepts & ArXiv References

* [1804.11326] Parameterized quantum circuits as machine learning models (https://arxiv.org/abs/1804.11326)
* [1810.03787] Tensor Network Quantum Virtual Machine (https://arxiv.org/abs/1810.03787)
