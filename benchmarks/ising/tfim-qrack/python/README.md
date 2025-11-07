# Python TFIM Sampler and Visualizations

This directory contains a suite of Python scripts for generating and visualizing data from a classical TFIM sampler.

## Sampler Scripts

*   **`generate_tfim_samples.py`**: A CPU-based TFIM sampler that uses a probabilistic model to generate samples directly, without simulating a quantum circuit.
*   **`generate_tfim_samples_vars.py` / `generate_tfim_samples_vars_delta.py`**: Variations of the sampler script with more advanced command-line argument handling.
*   **Shell Scripts (`*.sh`)**: A collection of shell scripts for running the Python samplers with various parameters and configurations.

## Visualization Scripts

*   **`generate_tfim_samples_vars_graph.py`**: A 3D visualization tool that parses the log files from the samplers and generates a 3D surface plot of the average magnetization.
*   **`generate_tfim_samples_vars_graph_auto.py`**: An automated version of the visualization script that generates a separate plot for each qubit width.
*   **`generate_tfim_samples_vars_graph_auto_big.py`**: A further enhancement that creates a single figure with a grid of 3D plots, with rows corresponding to qubit width and columns to theta.
*   **`generate_tfim_samples_vars_graph_auto_big_legenda.py`**: The final version of the visualization script, which includes a shared color bar for the entire grid of plots.

## How to Use

1.  **Generate Data**:
    *   Use one of the `generate_tfim_samples*.sh` scripts to run the samplers and generate log files in a `tfim_results` directory.
2.  **Visualize the Data**:
    *   Use one of the `generate_tfim_samples_vars_graph*.py` scripts to visualize the data from the `tfim_results` directory. The `auto_big_legenda` version is recommended for a comprehensive overview.
