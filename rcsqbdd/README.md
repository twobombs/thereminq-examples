# RCS with QBDD and Tensor Networks

This directory contains experiments and benchmarks focusing on Random Circuit Sampling (RCS) using Quantum Binary Decision Diagrams (QBDD) and Tensor Networks.

## Files

- `README.md`: This file, describing the contents of the `rcsqbdd/` directory.
- `fcrcsqbdd-mps-inspector.py`: A script that uses Matrix Product States (MPS) via `quimb` to simulate patches and visualize probability tensors.
- `fcrcsqbdd-tensor-loadplot.py`: A script for loading and plotting tensor data, possibly from previous runs.
- `fcrcsqbdd-tensor-noisy.py`: A script related to "Patch Circuit" XEB with tensor networks, likely incorporating noise models.
- `fcrcsqbdd-tensor-print.py`: A utility script for printing tensor information.
- `fcrcsqbdd-tensor.py`: A script implementing "Patch Circuit" XEB using tensor networks.
- `fcrcsqbdd.py`: A script implementing "Patch Circuit" XEB simulations.
- `qvqbdd-tensor.py`: A script combining QBDD simulations with tensor operations.
- `qvqbdd.py`: A script using `PyQrack`'s QBDD simulator for random circuit sampling.

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
