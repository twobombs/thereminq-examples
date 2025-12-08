# Quantinuum Run 352 Benchmarks

This directory contains a collection of benchmarks and experiments related to the Transverse Field Ising Model (TFIM), likely a specific run or set of experiments labeled "352".

## Subdirectories

*   **`ising/`**: Contains a suite of scripts for running, analyzing, and visualizing ACE Ising depth tests. This directory is very similar to the `ising` directory in the `island-run` parent directory, but it contains a few additional scripts.

## Key Scripts (in `ising/`)

*   **`3dtime.py`**: A script that creates a 3D log heatmap of the computation time versus the width and depth of the quantum circuit.
*   **`island-deltas.py`**: A script that reads a log file, pairs up entries with the same `width` and `depth`, and then calculates the difference in their `magnetization`, `square_magnetization`, and `seconds`.
*   **`magcostsheatmap.py`**: Reads log data and creates a `plotly` heatmap of average magnetization versus width and binned seconds.
*   **`magcurveheatmap.py`**: Reads log data and creates a `seaborn` heatmap of magnetization versus qubit width and circuit depth.
*   **`visualisation.py`**: A comprehensive visualization script that generates a variety of 2D plots to analyze the performance of the ACE Ising model.

This directory appears to be a snapshot of the `island-run` directory, with a focus on a particular run ("352") and some additional analysis scripts.
