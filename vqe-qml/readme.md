# VQE-QML: Hermitian Matrices for Quantum Machine Learning

This directory contains experimental code for a Quantum Machine Learning (QML) technique that uses a "learnable" Hermitian observable.

## Files

- `2505.13525v1.pdf`: The PDF of the arXiv paper "[2505.13525](https://arxiv.org/abs/2505.13525)" which the experiments are based on.
- `hermitian-matrices-pyqrack-oai.py`: A PyTorch-based implementation of the QML concept using PyQrack, likely containing optimizations or variations (possibly "OAI" refers to OpenAI or specific optimizations).
- `hermitian-matrices-pyqrack.py`: A PyTorch-based implementation of the QML concept using PyQrack. It uses a classical neural network to generate parameters for the VQC and Hermitian observable.
- `hermitian-matrices-pyqrack.sh`: A shell script for running the `hermitian-matrices-pyqrack-oai.py` script with specific Qrack configurations.
- `hermitian-matrices-qiskit.py`: A Qiskit-based implementation demonstrating the creation of a learnable Hermitian observable and a VQC with data encoding.
- `readme.md`: This file, describing the contents of the `vqe-qml/` directory.

## Overview

The scripts in this directory demonstrate a powerful QML technique where both the quantum circuit and the observable are parameterized and trained. This approach allows the model to learn not just the best quantum state for a given task, but also the best measurement to perform on that state.

## Visualization

![Screenshot from 2025-05-21 14-50-11](https://github.com/user-attachments/assets/36f27cd1-4596-4ad0-83bd-a5bcb37b5edc)
