# VQE-QML: Hermitian Matrices for Quantum Machine Learning

This directory contains experimental code for a Quantum Machine Learning (QML) technique that uses a "learnable" Hermitian observable. The approach is based on the concepts presented in the arXiv paper "[2505.13525](https://arxiv.org/abs/2505.13525)".

## Scripts

*   **`hermitian-matrices-qiskit.py`**: A Qiskit-based implementation of the concept, demonstrating how to create a learnable Hermitian observable and a Variational Quantum Circuit (VQC) with data encoding.
*   **`hermitian-matrices-pyqrack.py` / `hermitian-matrices-pyqrack-oai.py`**: PyTorch-based implementations of the same concept, designed as fully trainable QML models. These scripts use a classical neural network (a "Fast Weight Programmer" or "ControllerNN") to generate the parameters for both the VQC and the Hermitian observable. The quantum part of the model is implemented as a custom `torch.autograd.Function` with a PyQrack backend.
*   **`hermitian-matrices-pyqrack.sh`**: A shell script for running the `hermitian-matrices-pyqrack-oai.py` script with specific Qrack configurations.

## Overview

The scripts in this directory demonstrate a powerful QML technique where both the quantum circuit and the observable are parameterized and trained. This approach allows the model to learn not just the best quantum state for a given task, but also the best measurement to perform on that state.

## Visualization

![Screenshot from 2025-05-21 14-50-11](https://github.com/user-attachments/assets/36f27cd1-4596-4ad0-83bd-a5bcb37b5edc)
