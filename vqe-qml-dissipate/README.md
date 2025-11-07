# VQE-QML with Dissipation

This directory contains a Python script that demonstrates a technique for avoiding barren plateaus in Variational Quantum Algorithms (VQAs) by using dissipation. The approach is based on the concepts presented in the arXiv paper "[2507.02043](https://arxiv.org/abs/2507.02043)".

## Files

*   **`dissipate-ancilaries.py`**: A Qiskit-based script that simulates and compares a standard unitary VQA with a dissipative VQA. The dissipative VQA uses ancillary qubits that are periodically reset to remove entropy from the system.

## Overview

The script in this directory provides a clear and well-commented implementation of the dissipative VQA concept. It demonstrates how the use of dissipation can prevent the gradients of the cost function from vanishing exponentially with the system size, which is a common problem in VQAs known as the "barren plateau" problem. The script calculates and plots the gradient variances for both the unitary and dissipative VQAs, showing the effectiveness of the dissipative approach.
