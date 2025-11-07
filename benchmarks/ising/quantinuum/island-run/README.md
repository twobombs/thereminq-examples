# Quantinuum Island Run Benchmarks

This directory contains a collection of benchmarks and experiments related to the Transverse Field Ising Model (TFIM), likely run on or for Quantinuum hardware. The "island run" name suggests that these experiments may be related to the "islands" of qubits on a quantum device.

## Subdirectories

*   **`ising/`**: Contains a suite of scripts for running, analyzing, and visualizing ACE Ising depth tests.
*   **`max-isles/`**: Contains data and scripts related to a "maximum islands" patching strategy.
*   **`min-isles/`**: Contains data and scripts related to a "minimum islands" patching strategy.
*   **`same-isles/`**: Contains a variety of scripts for running, validating, and analyzing TFIM simulations, including free energy calculations and comparisons with Qiskit's Aer simulator.

## Visualization Scripts

*   **`magcurveheatmap3d.py`**: A Python script that reads quantum simulation log data and generates 3D surface plots of magnetization and square magnetization.
*   **`magcurveheatmap3dark.py`**: A dark-mode version of the 3D plotting script.
*   **`magsqrcurveheatmap.py`**: A script that generates a 2D heatmap of square magnetization versus qubit width and circuit depth.
