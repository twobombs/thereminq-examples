# Quantum Echoes (OTOC) Simulator

This directory contains a suite of scripts for simulating and analyzing Out-of-Time-Order Correlators (OTOCs), which are used to study quantum chaos and information scrambling.

## Files

- `Docs/`: Directory containing documentation related to the OTOC simulations.
- `Prototyping/`: Directory containing prototyping code and experimental scripts.
- `otoc_statevector_simulation.py`: A script demonstrating how to use the `generate_otoc_samples` function from `pyqrackising`.
- `otoc_validation_isingonly_cpu.py`: A validation tool that generates OTOC samples and prints the resulting probabilities.
- `otoc_validation_isingonly_cpu.sh`: A shell script to run the `otoc_validation_isingonly_cpu.py` script.
- `otoc_validation_isingonly_graph.py`: A visualization tool that parses log files and creates a 3D surface plot of execution time vs. qubits and depth.
- `otocs-prediction-512.py`: A tool that uses linear regression to predict the execution time for OTOC simulations with a large number of qubits (512).
- `readme.md`: This file, describing the contents of the `qecho/` directory.

## Overview

The scripts in this directory provide a comprehensive set of tools for simulating, analyzing, and visualizing OTOCs. They are designed to be used with the `pyqrackising` library and can be run on both CPUs and GPUs. The visualizations produced by these scripts provide valuable insights into the behavior of OTOCs in quantum systems.

<img width="7680" height="4167" alt="otoc_sweep_3d_plot (3)" src="https://github.com/user-attachments/assets/f741752b-b39a-487d-9cdf-32ad0e4eff50" />
