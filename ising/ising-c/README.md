# C/OpenCL Ising Simulations

This directory contains C and OpenCL source code for simulations of the 2D Ising model and the Transverse Field Ising Model (TFIM).

## Subdirectories

*   **`tfim-c/`**: Contains the C and OpenCL source code for a TFIM sampler. This is a classical sampler that uses a probabilistic model to generate samples directly, rather than simulating the quantum circuit. It includes both a CPU-based and an OpenCL-accelerated version.

## Files

*   **`main.c`**: The host-side C code for a full quantum circuit simulator of the 2D Ising model.
*   **`ising_kernel.cl`**: The OpenCL kernel code for the 2D Ising model simulator.
*   **`ising_sim`**: The compiled executable for the 2D Ising model simulator.
