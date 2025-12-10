# Quantum Echoes (OTOC) Simulator

This directory contains a suite of scripts for simulating and analyzing Out-of-Time-Order Correlators (OTOCs), which are used to study quantum chaos and information scrambling. The name "Quantum Echoes" and the reference to a Google blog post suggest that these simulations are related to the experiments performed on Google's Sycamore processor.

## Key Scripts

*   **`otoc_statevector_simulation.py`**: A simple example of how to use the `generate_otoc_samples` function from the `pyqrackising` library.
*   **`otoc_validation_isingonly_cpu.py`**: A validation tool for the OTOC simulations that uses the `generate_otoc_samples` function to generate samples and then prints the resulting probabilities.
*   **`otoc_validation_isingonly_cpu.sh`**: Shell script to run the CPU-based OTOC validation.
*   **`otoc_validation_isingonly_graph.py`**: A sophisticated visualization tool that parses the log files from the OTOC simulations and creates a 3D surface plot of the execution time as a function of the number of qubits and the depth.
*   **`otocs-prediction-512.py`**: A prediction tool that uses linear regression to estimate the time required to run OTOC simulations for a large number of qubits.

## Subdirectories

*   **`Docs/`**: Contains documentation related to the OTOC simulations.
*   **`Prototyping/`**: Contains prototyping code and experimental scripts.

## Overview

The scripts in this directory provide a comprehensive set of tools for simulating, analyzing, and visualizing OTOCs. They are designed to be used with the `pyqrackising` library and can be run on both CPUs and GPUs. The visualizations produced by these scripts provide valuable insights into the behavior of OTOCs in quantum systems.

<img width="7680" height="4167" alt="otoc_sweep_3d_plot (3)" src="https://github.com/user-attachments/assets/f741752b-b39a-487d-9cdf-32ad0e4eff50" />
