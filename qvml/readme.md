# Quantum Variational Machine Learning (QVML) Demo

This directory contains a small demonstration of Quantum Variational Machine Learning (QVML) using tensor network contraction. The scripts in this directory show how to generate a random quantum circuit, convert it to a tensor network, find an optimal contraction path, and then visualize the results.

## Scripts

*   **`qvml.py`**: A Python script that demonstrates the core logic of the QVML demo. It generates a random quantum volume (QV) circuit, converts it to a tensor network using `quimb`, finds an optimal contraction path by converting the problem to a Traveling Salesperson Problem (TSP) and solving it with `pyqrackising`, and then iteratively contracts the tensor network.
*   **`qmvl_overview.py`**: A visualization tool that parses the output of `qvml.py`, builds a `networkx` graph representation of the tensor network, and then uses `vedo` to create a 3D visualization. It can display multiple tensor networks in a grid.
*   **`qvml_heatmap.py`**: A script that reads data from a CSV file and generates a heatmap of the computational cost versus the number of qubits and the circuit depth.
*   **`qvml_spheres.py`**: Another visualization tool for the tensor networks, similar to `qmvl_overview.py`, but designed to visualize a single network at a time with different visualization options.
*   **`qvml.sh` / `qvml_csv.sh`**: Shell scripts for running the `qvml.py` script and generating CSV files from the output.

## Overview

The scripts in this directory provide a complete workflow for generating, simulating, and visualizing tensor networks. This is a powerful technique for simulating quantum circuits that are too large to be simulated with traditional statevector methods.

## Visualizations

<img width="12800" height="6945" alt="tensor_network_screenshot" src="https://github.com/user-attachments/assets/9652bd67-1e4a-4c0e-86b8-e8914b08fff4" />

<img width="1062" height="885" alt="Screenshot from 2025-10-03 15-05-31" src="https://github.com/user-attachments/assets/6c5845ef-ba16-4d8a-8fae-a7764a920580" />
