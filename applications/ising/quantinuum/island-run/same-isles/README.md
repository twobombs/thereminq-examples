# Transverse Field Ising Model (TFIM) Benchmarks

This directory contains a suite of scripts for running, validating, and analyzing simulations of the Transverse Field Ising Model (TFIM) on the `QrackAceBackend`.

## Key Scripts

*   **`ising_ace_depth_series.py`**: A core script for simulating the TFIM on a 2D lattice. It uses a Trotterization approach to approximate the time evolution of the system and calculates the magnetization and square magnetization at each step.
*   **`ising_ace_validation.py`**: A script designed to validate the results of the `QrackAceBackend` against the `AerSimulator` from Qiskit. It runs the same TFIM circuit on both backends and then calculates a variety of statistics to compare the results.
*   **`ising_ace_free_energy.py`**: An extension of the TFIM simulation that includes the calculation of free energy. It estimates the entropy of the system and computes the energy contributions from both the Z and X terms to calculate the free energy at each Trotter step.

## Other Files

*   **Shell Scripts (`*.sh`)**: These scripts are used to automate the execution of the Python scripts with various parameters and configurations.
*   **Jupyter Notebooks (`*.ipynb`)**: These notebooks are likely used for interactive exploration, analysis, and visualization of the simulation data.
*   **Log Files (`*.log`, `*.txt`)**: These files contain the output and results of the simulation runs.

## Overview

The scripts in this directory are used to perform a comprehensive analysis of the TFIM model on the Qrack simulator. They cover a wide range of qubit widths and depths, and they include advanced analysis techniques such as free energy calculation and validation against other simulators. The results of these simulations are used to generate the various plots and visualizations seen in the `README.md` files in the parent directories.
