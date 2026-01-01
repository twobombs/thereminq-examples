# Peaked: QASM to 3D Tensor Visualization and Analysis

This project explores various methods for analyzing, simulating, and visualizing Quantum Assembly (QASM) circuits, with a focus on solving the `P1_little_dimple.qasm` challenge. It is divided into multiple approaches ranging from brute-force simulation to advanced holographic/stabilizer analysis and physics-inspired OTOC solvers.

## Directory Structure

### `bruteforce/`
Contains scripts for direct simulation of the quantum circuit using high-performance simulators.
- **`qpepyqrack.py`**: Uses the `pyqrack` simulator (GPU-accelerated) to execute the circuit and decode the phase.

### `discovery/`
Focuses on 3D visualization of the circuit structure to identify patterns or anomalies.
- **`qpetensorvizphyz.py`**: Renders an interactive 3D model of the circuit tensor using `vedo`, featuring "extreme" scaling layouts.

### `clustered_angles_solver_P1/` (formerly `solver_1`)
Implements an analytical approach to find hidden signals ("dimples") in the circuit parameters.
- **`render_landscape.py`**: Visualizes the circuit as a 3D landscape of rotation angles (Time vs. Space vs. Angle).
- **`solve_dimple.py`**: A clustering-based solver that extracts the hidden phase by analyzing the distribution of rotation angles.

### `holographic_solver_P1/`
A solver that exploits "stabilizer" structures and "holographic" principles to purify the circuit.
- **`extract_peak.py`**: Snaps circuit parameters to Clifford gates to remove noise and finding the "peaked" bitstring deterministically.
- **`analyse_stabilizers.py`**: Analyzes the Clifford tableau to determine the theoretical probability and superposition dimensions.

### `otoc_maps_solver_P1/`
Uses physics-inspired **Out-of-Time-Order Correlators (OTOCs)** to find the solution.
- **`solution_p1_optimize-hybrid.py`**: Scans for "Stability Plateaus" in the scrambling dynamics (time evolution).
- **`solution_otoc_p1b.py`**: Performs robust ensemble averaging in the identified time window to verify the attractor bitstring.

## Root Files
- **`P1_little_dimple.qasm`**: The target QASM file containing the circuit to be analyzed/solved.
