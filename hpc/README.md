# High-Performance Computing (HPC) Scripts

This directory contains a collection of scripts for running high-performance computing (HPC) benchmarks and simulations using the Qrack framework. These scripts are designed to run on multi-GPU systems, cloud platforms like AWS, and other high-performance environments.

## Scripts

*   **`run-aws-byoc-qrack.py`**: A Python script that demonstrates how to use the Qrack simulator with Amazon Braket Hybrid Jobs, including how to use a custom Docker image for GPU acceleration.
*   **`run-cosmos-nbody-QuadGPU.sh`**: A shell script for running a cosmological N-body simulation benchmark on a quad-GPU system.
*   **`run-findafactor.sh`**: A shell script for running a factorization benchmark, similar to the one in the `findafactor` directory.
*   **`run-fqa-dask`**: A Python script that demonstrates how to use Dask with CuPy for GPU-accelerated scientific computing.
*   **`run-qft-cube32plus-multi`**: A shell script for running a QFT cosmology benchmark on a multi-GPU system.
*   **`run-qrng-aws-service.sh`**: A shell script that demonstrates how to use the Qrack benchmark executable as a Quantum Random Number Generator (QRNG) service.
*   **`run-rcs-nn-49-cpu`**: A shell script for running a Random Circuit Sampling (RCS) benchmark on the CPU.
*   **`run-sycamore-patch-quadrant-time`**: A shell script for running a Sycamore patch quadrant simulation benchmark.
