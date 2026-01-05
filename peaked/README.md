# Peaked: QASM to 3D Tensor Visualization and Analysis

This project explores various methods for analyzing, simulating, and visualizing Quantum Assembly (QASM) circuits, with a focus on solving the `P1_little_dimple.qasm` challenge. It is divided into multiple approaches ranging from brute-force simulation to advanced holographic/stabilizer analysis and physics-inspired OTOC solvers.

## Directory Structure

### `bruteforce/`
Contains scripts for direct simulation of the quantum circuit using high-performance simulators.
- **`qpepyqrack.py`**: Uses the `pyqrack` simulator (GPU-accelerated) to execute the circuit and decode the phase.
- **`qpepyqrackqbdd.py`**: A variant utilizing the Quantum Binary Decision Diagram (QBDD) engine for potentially improved performance.
- **`qpepyqrackqbdd-transpiled.py`**: A transpiled version for basis-set optimization before simulation.

### `discovery/`
Focuses on 3D visualization of the circuit structure to identify patterns or anomalies.
- **`qpetensorvizphyz.py`**: Renders an interactive 3D model of the circuit tensor using `vedo`, featuring "extreme" scaling layouts.
- **`qpetensorvizphyztagged.py`**: Adds rank-based heatmap coloring to the visualization to highlight interaction hotspots.
- **`qpetensorviztube.py`**: A clean "Dark Mode" visualization with simplified aesthetics and manual legend.

### `simulation/`
Contains scripts for circuit manipulation, resizing, and generating steered circuits.
- **`connectivityplot.py`**: Visualizes the connectivity of the quantum circuit.
- **`hide+steer_qrack.py`**: Generates random circuits steered towards specific bitstrings.
- **`resize.py`**: Rebuilds the 'little dimple' circuit structure at variable bitwidths, preserving the original connection texture and depth.

### `solvers/`
Contains specialized solvers developed to tackle the challenge using different theoretical frameworks.

- **`clustered_angles_solver_P1/`**:
    - **Method**: Analytical clustering of rotation angles.
    - **Key Scripts**: `render_landscape.py` (visualizes angle distribution), `solve_dimple.py` (clusters angles to find hidden signals).

- **`haar_solver_P1/`**:
    - **Method**: Analysis of statistical deviations from Haar randomness.
    - **Key Scripts**: `haar-deviation.py` (plots "magic angle" deviations), `purify-deviation.py` (purifies circuit structure).

- **`holographic_solver_P1/`**:
    - **Method**: Stabilizer purification and holographic bulk analysis.
    - **Key Scripts**: `extract_peak.py` (snaps parameters to Clifford gates), `analyse_stabilizers.py` (analyzes superposition dimensions).

- **`otoc_maps_solver_P1/`**:
    - **Method**: Physics-inspired Out-of-Time-Order Correlators (OTOCs) and scrambling stability.
    - **Key Scripts**: `solution_p1_optimize-hybrid.py` (scans for stability plateaus), `solution_otoc_p1b.py` (robust ensemble averaging).

### `verification/`
Contains tools for generating custom challenge circuits (Full Density, Mirror) and verifying solver robustness.
- **`peaked_generation_pyqrack.py`**: Generates and solves dense random circuits with hidden bitstrings.
- **`peaked_generation_pyqrack_p1.py`**: A consensus-based solver script for P1-style challenges.
- **`peaked_generation_pyqrack_p1-hybrid-sparse-36.py`**: A highly optimized solver for large circuits (36 qubits) using custom C-based atomic filtering and memory management.

## Root Files
- **`P1_little_dimple.qasm`**: The target QASM file containing the circuit to be analyzed/solved.
