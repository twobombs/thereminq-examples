# RCS with QBDD and Tensor Networks

This directory contains experiments and benchmarks focusing on Random Circuit Sampling (RCS) using Quantum Binary Decision Diagrams (QBDD) and Tensor Networks.

## Key Features

*   **Patch Circuit Benchmarking**: Scripts like `fcrcsqbdd.py` and `fcrcsqbdd-tensor.py` implement "Patch Circuit" XEB (Cross-Entropy Benchmarking), comparing full grid simulations against split-patch "ideal" simulations to calculate fidelity.
*   **QBDD Simulation**: `qvqbdd.py` leverages `PyQrack`'s QBDD simulator to efficiently sample from random circuits and identify heavy output strings.
*   **Tensor Network Inspection**: `fcrcsqbdd-mps-inspector.py` uses Matrix Product States (MPS) via `quimb` to simulate patches and visualize the resulting probability tensors.
*   **GPU Acceleration**: Several scripts are designed to utilize GPU resources (via PyTorch or PyQrack) for handling large probability tensors and simulations.

## Usage

Most scripts can be run directly with Python, optionally accepting width and depth arguments.

```bash
python fcrcsqbdd.py [width] [depth]
python qvqbdd.py [width] [depth]
```

## Requirements

*   `pyqrack`
*   `qiskit`
*   `quimb`
*   `torch` (for tensor operations on GPU)
*   `matplotlib` (for visualization)


## Core Quantum Mechanical Concepts & ArXiv References

* [1905.08394] Classical Simulation of Quantum Supreme Circuits (https://arxiv.org/abs/1905.08394)
* [2002.07730] Tensor network simulation of the Sycamore quantum supremacy circuits (https://arxiv.org/abs/2002.07730)
