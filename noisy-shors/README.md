# Noisy Shor's Algorithm

This directory contains Python scripts for implementing Shor's algorithm for factoring integers, with a focus on running the algorithm in a simulated noisy environment.

## Scripts

*   **`noisy-big-shors.py`**: A detailed implementation of Shor's algorithm for factoring large numbers, using PyQrack as the quantum simulator. This script is designed to demonstrate the scaling of the algorithm to a large number of qubits, but it is not practical for actual factorization due to the naive and computationally expensive implementation of the quantum oracle.
*   **`noisy-smol-shors.py`**: A version of the Shor's algorithm script for factoring small numbers (e.g., 15). This script is likely used for testing, debugging, and educational purposes.

## Overview

The scripts in this directory provide a clear and well-commented implementation of Shor's algorithm, including both the classical and quantum components. The "noisy" aspect of the directory name suggests that these scripts may be used in conjunction with a noise model to study the performance of Shor's algorithm in the presence of errors, although the noise model itself is not explicitly defined in these scripts.
