# Variational Quantum Eigensolver (VQE) Examples

This directory contains a diverse collection of scripts for implementing and analyzing the Variational Quantum Eigensolver (VQE) algorithm. The examples cover different quantum software frameworks (Cirq, PennyLane, Qrack), different molecular systems (H2, and a variety from a CSV), and various aspects of the VQE algorithm.

## Scripts

### VQE for Molecular Ground State Energy

*   **`import_clifford_vqe_entangled.py`**: A PennyLane-based script that performs VQE calculations for a list of molecules defined in a CSV file. It compares the calculated ground state energy to the expected value.
*   **`import_clifford_vqe_entangled-gpu-multi.py`**: A multi-GPU, enhanced version of the previous script that uses `multiprocessing` to distribute the calculations across multiple GPUs.
*   **`run-multi-vqe-cirq-h2.py`**: A comprehensive VQE simulation for the H2 molecule using PennyLane with the Cirq simulator as the backend. It includes explicit FCI calculation for benchmarking, a manual UCCSD ansatz, and detailed convergence analysis.
*   **`run-multi-vqe-pennylane-h2.py`**: A version of the H2 VQE simulation that is specifically adapted to use the PennyLane-Qrack plugin with GPU acceleration.
*   **`run-multi-vqe-qrack-h2.py`**: This script is identical to the PennyLane version, demonstrating the use of the PennyLane-Qrack plugin.
*   **`run-multi-vqe-ibmheron-h2.py`**: (Not analyzed in detail, but likely a version of the H2 VQE simulation for the IBM Heron processor.)

### Visualization

*   **`vqe-results-3dviz.py`**: A visualization tool that parses the log files from the VQE simulations and creates a 3D scatter plot of the results, showing the relationship between the number of qubits, the final energy, and the accuracy of the calculation.

## Overview

The scripts in this directory provide a rich set of tools for exploring the VQE algorithm. They demonstrate how to use different quantum software frameworks to solve a fundamental problem in quantum chemistry, and they include advanced features such as multi-GPU acceleration, detailed analysis, and sophisticated visualization.

## Visualizations

![h2_sto-3g_vqe_convergence_seed42_ManualUCCSD_JW_default_qubit](https://github.com/user-attachments/assets/00f881b6-73e5-4554-a258-b4de190abf00)

<img width="1096" height="853" alt="Screenshot from 2025-08-23 19-58-47" src="https://github.com/user-attachments/assets/b05f3a42-7381-41da-812d-450e4cad1324" />
