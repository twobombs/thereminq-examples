# OpenCL Ising and TFIM Simulations

This directory contains C and OpenCL source code for simulations of the 2D Ising model and the Transverse Field Ising Model (TFIM).

## Subdirectories

*   **`ising-c/`**: Contains the C and OpenCL source code for a simulation of the 2D Ising model. This is a full quantum circuit simulator that uses a Trotterization approach.
*   **`tfim-c/`**: Contains the C and OpenCL source code for a TFIM sampler. This is a classical sampler that uses a probabilistic model to generate samples directly, rather than simulating the quantum circuit. It includes both a CPU-based and an OpenCL-accelerated version.
