# OpenCL Ising Model Simulation

This directory contains the C and OpenCL source code for a simulation of the 2D Ising model.

## Files

*   **`main.c`**: The host-side C code that sets up the OpenCL environment, loads and builds the kernel, creates the necessary buffers, and runs the simulation loop. It calculates and prints the magnetization and square magnetization at each Trotter step.
*   **`ising_kernel.cl`**: The OpenCL kernel code that defines the quantum operations executed on the device (GPU or CPU). It includes kernels for initializing the state vector and applying RX and RZZ gates.
*   **`install.sh`**: A simple shell script that compiles the `main.c` file to create the `ising_sim` executable.
*   **`ising_sim`**: The compiled executable.

## How to Use

1.  **Compile the code**:
    ```bash
    bash install.sh
    ```
2.  **Run the simulation**:
    ```bash
    ./ising_sim [n_qubits] [depth] [dt] [t1] [shots] [trials]
    ```
    For example:
    ```bash
    ./ising_sim 16 20
    ```
