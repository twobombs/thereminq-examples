# ACE Ising Depth Tests

This directory contains a suite of scripts and visualizations for conducting and analyzing depth tests of the ACE Ising model. The experiments cover a wide range of qubit widths and depths, and the results are visualized in various 2D and 3D plots.

## Scripts

### Core Logic and Benchmarking

*   **`ising_ace.py` / `ising_ace_depth_series.py`**: (These scripts are not present in this directory, but are called by the shell scripts.) These are the core Python scripts for running the ACE Ising simulations. They likely take parameters such as qubit width, depth, and patching strategy as command-line arguments.
*   **`3GPU-ACE-run.sh`**: A shell script that runs `ising_ace.py` with a nested loop of parameters, distributing the jobs across three GPUs.
*   **`ising_ace_depth_scan.sh`**: A script for running a series of benchmarks with different parameters, pausing for user input between each run.
*   **`ising_ace_depth_series-high.sh`**: A comprehensive benchmark script that runs simulations for qubit widths from 4 to 1024, using a "maximum islands" patching strategy.
*   **`ising_ace_depth_series-low.sh`**: Similar to the "high" version, but this script uses a "minimum islands" patching strategy.

### Visualization and Analysis

*   **`magcostsheatmap.py`**: Reads log data and creates a `plotly` heatmap of average magnetization versus width and binned seconds.
*   **`magcurveheatmap.py`**: Reads log data and creates a `seaborn` heatmap of magnetization versus qubit width and circuit depth.
*   **`visualisation.py`**: A comprehensive visualization script that generates a variety of 2D plots to analyze the performance of the ACE Ising model, including:
    *   Magnetization vs. Depth
    *   Computation Time vs. Depth
    *   Magnetization vs. Computation Time
    *   Average Magnetization vs. Width
    *   Box Plot of Magnetization by Width

## How to Use

1.  **Run Benchmarks**:
    *   Use one of the `*.sh` scripts to run the ACE Ising simulations. This will generate `.log` or `.txt` files containing the simulation data.
2.  **Analyze and Visualize**:
    *   Use the `magcostsheatmap.py`, `magcurveheatmap.py`, or `visualisation.py` scripts to generate plots from the log files. Make sure to place the log files in the same directory as the scripts and name them appropriately (e.g., `fullog.log`).

## Visualizations

![mag_vs_sec_heatmap](https://github.com/user-attachments/assets/784a2115-1009-40c4-a18e-ed134fe0ae95)
![Screenshot from 2025-06-09 14-39-43](https://github.com/user-attachments/assets/18ae217c-b28b-42ab-8949-2a6a4c4ebcd7)
![magnetization_heatmap](https://github.com/user-attachments/assets/25e3922f-a12a-4c6b-be4d-0926196a17f5)
![magnetization_vs_depth](https://github.com/user-attachments/assets/0d47151d-fb24-44b9-b09b-24e138148149)
![seconds_vs_depth](https://github.com/user-attachments/assets/bc7327e7-b58a-442e-a158-d2e15954f764)
![avg_magnetization_vs_width](https://github.com/user-attachments/assets/5040cc81-f8f9-4377-a0ba-fea8c379f061)
![magnetization_vs_seconds](https://github.com/user-attachments/assets/2d2cf00f-2762-4b2d-899d-2187601bdee1)
![boxplot_magnetization_vs_width](https://github.com/user-attachments/assets/ab8bc839-486f-4969-b3e2-e347f5d0e048)
