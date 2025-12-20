# Peaked: QASM to 3D Tensor Visualization and Analysis

This project explores various methods for analyzing, simulating, and visualizing Quantum Assembly (QASM) circuits, with a focus on solving the `P1_little_dimple.qasm` challenge. It is divided into three main approaches: brute-force simulation, structural discovery/visualization, and analytical landscape solving.

## Directory Structure

### `bruteforce/`
Contains scripts for direct simulation of the quantum circuit.
- **`qpepyqrackqbdd.py`**: Uses the `pyqrack` simulator to execute the circuit and decode the phase.

### `discovery/`
Focuses on 3D visualization of the circuit structure to identify patterns or anomalies.
- **`qpetensorvizphyz.py`**: Renders an interactive 3D model of the circuit tensor using `vedo`, featuring "extreme" scaling layouts.
- **`qpetensorviztube.py`**, **`qpetensorvizphyztagged.py`**: Variations of the visualization tool with different layouts or highlighting.
- **`qpe.qasm`**: A test QASM file.

### `solver_1/`
Implements an analytical approach to find hidden signals ("dimples") in the circuit parameters.
- **`qpetensor.py`**: Converts QASM files into PyTorch tensors for analysis.
- **`render_landscape.py`**: Visualizes the circuit as a 3D landscape of rotation angles (Time vs. Space vs. Angle) to highlight the signal.
- **`solve_dimple.py`**: A clustering-based solver that extracts the hidden phase by analyzing the distribution of rotation angles.

## Root Files
- **`P1_little_dimple.qasm`**: The target QASM file containing the circuit to be analyzed/solved.
