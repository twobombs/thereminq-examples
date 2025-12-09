# Variational Quantum Eigensolver (VQE) Examples

This directory contains a diverse collection of scripts for implementing and analyzing the Variational Quantum Eigensolver (VQE) algorithm.

## Files

- `import_clifford_vqe_entangled-gpu-multi.py`: A multi-GPU, enhanced version of the VQE script that distributes calculations across multiple GPUs using `multiprocessing`.
- `import_clifford_vqe_entangled.py`: A PennyLane-based script that performs VQE calculations for a list of molecules defined in `import_clifford_vqe_min.csv`.
- `import_clifford_vqe_entangled.sh`: A shell script to run the `import_clifford_vqe_entangled.py` script.
- `import_clifford_vqe_min.csv`: A CSV file containing molecule data (names, symbols, geometries) used by the VQE scripts.
- `import_clifford_vqe_min.py`: Likely a script to generate or process the `import_clifford_vqe_min.csv` file.
- `readme.md`: This file, describing the contents of the `vqe/` directory.
- `run-multi-vqe-cirq-h2.py`: A comprehensive VQE simulation for H2 using PennyLane with the Cirq simulator backend.
- `run-multi-vqe-ibmheron-h2.py`: A version of the H2 VQE simulation, likely targeting or simulating the IBM Heron processor.
- `run-multi-vqe-pennylane-h2.py`: A version of the H2 VQE simulation adapted to use the PennyLane-Qrack plugin with GPU acceleration.
- `run-multi-vqe-qrack-h2.py`: A script identical to the PennyLane version, demonstrating the use of the PennyLane-Qrack plugin.
- `vqe-results-3dviz.py`: A visualization tool that parses log files and creates a 3D scatter plot of the results.

## Overview

The scripts in this directory provide a rich set of tools for exploring the VQE algorithm. They demonstrate how to use different quantum software frameworks to solve a fundamental problem in quantum chemistry, and they include advanced features such as multi-GPU acceleration, detailed analysis, and sophisticated visualization.

## Visualizations

![h2_sto-3g_vqe_convergence_seed42_ManualUCCSD_JW_default_qubit](https://github.com/user-attachments/assets/00f881b6-73e5-4554-a258-b4de190abf00)

<img width="1096" height="853" alt="Screenshot from 2025-08-23 19-58-47" src="https://github.com/user-attachments/assets/b05f3a42-7381-41da-812d-450e4cad1324" />
