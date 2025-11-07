# TFIM Sampler (C and OpenCL)

This directory contains the C and OpenCL source code for a Transverse Field Ising Model (TFIM) sampler. It includes both a CPU-based and an OpenCL-accelerated version of the sampler.

## Files

*   **`ising_sampler.c`**: The CPU-based version of the TFIM sampler. It uses a probabilistic model to generate samples directly, rather than simulating the quantum circuit.
*   **`ising_sampler_opencl.c`**: The OpenCL-accelerated version of the TFIM sampler. It offloads the most computationally intensive part of the sampling process to the GPU.
*   **`sampler_kernel.cl`**: The OpenCL kernel code that calculates the "closeness-of-like-bits" metric for all possible permutations of a given Hamming weight.
*   **`Makefile`**: A makefile for compiling the C code.
*   **`generate_tfim_samples_multi_delta.sh` / `generate_tfim_samples_multi_delta_cpu.sh`**: Shell scripts for running the sampler with various parameters.
*   **`generate_tfim_samples_vars_graph_auto_c.py` / `generate_tfim_samples_vars_graph_auto_big_legenda_c.py`**: Python scripts for visualizing the results.

## How to Use

1.  **Compile the code**:
    ```bash
    make
    ```
    This will create two executables: `ising_sampler` and `ising_sampler_opencl`.

2.  **Run the sampler**:
    *   **CPU version**:
        ```bash
        ./ising_sampler [n_qubits] [depth] [dt] [shots] [J] [h] [theta] [delta_theta]
        ```
    *   **OpenCL version**:
        ```bash
        ./ising_sampler_opencl --list-devices  # List available OpenCL devices
        ./ising_sampler_opencl --device [device_index] [n_qubits] [depth] [dt] [shots] [J] [h]
        ```
3.  **Visualize the results**:
    *   Use the `generate_tfim_samples_vars_graph_auto_c.py` scripts to visualize the output of the samplers.
