# Quantum Echoes (OTOC) Simulator

This directory contains a suite of scripts for simulating and analyzing Out-of-Time-Order Correlators (OTOCs), which are used to study quantum chaos and information scrambling. The name "Quantum Echoes" and the reference to a Google blog post suggest that these simulations are related to the experiments performed on Google's Sycamore processor.

## Key Scripts

### `otoc_statevector_simulation.py`
A simple example script demonstrating how to use the `generate_otoc_samples` function from the `pyqrackising` library.
*   **Parameters**: Uses hardcoded parameters (`J`, `h`, `z`, `theta`, `t`) and `n_qubits=56`.
*   **Method**: Simulates OTOC using a specified Pauli string (`X` on the first qubit, `I` elsewhere) and measures in the Z basis.
*   **Usage**: Run directly with python.

### `otoc_validation_isingonly_cpu.py`
A validation tool for OTOC simulations. It reduces the transverse field Ising model from a $2^n$-dimensional problem to an $(n+1)$-dimensional approximation to check against Trotter error.
*   **Method**: Calculates experiment probabilities using `generate_otoc_samples` and compares marginal probabilities.
*   **Arguments**:
    1.  `n_qubits` (default: 16)
    2.  `depth` (default: 16)
    3.  `cycles` (default: 3)
    4.  `butterfly_fraction` (optional, default: 1/n_qubits)

### `otoc_validation_isingonly_cpu.sh`
A shell script designed to execute the CPU-based OTOC validation. It likely wraps the python script with specific arguments or environment settings.

### `otoc_validation_isingonly_graph.py`
A visualization tool that parses log files (expected in `otoc_sweep_log/`) from OTOC simulations to analyze performance scaling.
*   **Features**:
    *   Parses logs for execution time and probability of the zero state.
    *   Generates a 3D surface plot of execution time vs. qubits and depth using `vedo`.
    *   Handles data cleaning and interpolation for smoother visualization.
    *   Saves the plot as `otoc_sweep_3d_plot.png`.

### `otocs-prediction-512.py`
A predictive analysis tool that estimates simulation times for large qubit counts (up to 10,000) at a fixed depth (default: 512).
*   **Method**: Parses logs matching `q*_d512.log` in `otoc_sweep_log/` and performs linear regression on execution times.
*   **Output**: Prints predicted times for various qubit milestones and saves a trend plot as `qubit_time_prediction_linear_10k_d512.png`.

## Subdirectories

*   **`Docs/`**: Contains documentation and references related to OTOC simulations.
*   **`Prototyping/`**: Contains prototyping code and experimental scripts.

## Usage

To run the basic statevector simulation:
```bash
python3 otoc_statevector_simulation.py
```

To run the validation script with custom parameters (e.g., 20 qubits, depth 10, 5 cycles):
```bash
python3 otoc_validation_isingonly_cpu.py 20 10 5
```

To visualize results (ensure you have log files in `otoc_sweep_log/`):
```bash
python3 otoc_validation_isingonly_graph.py
```

## Overview

The scripts in this directory provide a comprehensive set of tools for simulating, analyzing, and visualizing OTOCs. They are designed to be used with the `pyqrackising` library and can be run on both CPUs and GPUs. The visualizations produced by these scripts provide valuable insights into the behavior of OTOCs in quantum systems.

<img width="7680" height="4167" alt="otoc_sweep_3d_plot (3)" src="https://github.com/user-attachments/assets/f741752b-b39a-487d-9cdf-32ad0e4eff50" />
