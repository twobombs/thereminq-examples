# TFIM-Qrack Benchmarks

This directory contains a collection of benchmarks and experiments for the Transverse Field Ising Model (TFIM) using the Qrack framework. It includes both OpenCL-accelerated quantum simulations and classical samplers.

## Subdirectories

*   **`opencl/`**: Contains C and OpenCL source code for simulations of the 2D Ising model and the Transverse Field Ising Model (TFIM).
    *   **`ising-c/`**: A full quantum circuit simulator that uses a Trotterization approach.
    *   **`tfim-c/`**: A classical sampler that uses a probabilistic model to generate samples directly, with both CPU and OpenCL-accelerated versions.
*   **`python/`**: Contains a suite of Python scripts for generating and visualizing data from a classical TFIM sampler.

## Overview

The scripts in this directory are used to generate and analyze data for TFIM phase diagrams. The `opencl` directory contains the low-level simulation and sampling code, while the `python` directory provides a set of tools for generating data and creating sophisticated 3D visualizations of the results.

## Visualization

<img width="7287" height="4309" alt="tfim_phase_diagram_grid" src="https://github.com/user-attachments/assets/344758b8-1260-4af2-b1a8-331465e8e20a" />
