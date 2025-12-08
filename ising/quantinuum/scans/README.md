# Quantinuum Scans

This directory contains a collection of benchmark runs, likely related to the Stochastic Drift-Diffusion Process (SDRP) of the Qrack simulator. The subdirectories are named with different SDRP parameter settings, and they contain the results and analysis scripts for each run.

## Subdirectories

The subdirectories in this directory are named with different SDRP parameter settings, for example:

*   `0146SDRP200over20/`
*   `noSDRP150over20/`

Each of these subdirectories contains a snapshot of a benchmark run, including:

*   **Log Files (`*.txt`)**: The raw output of the simulation runs.
*   **Python Scripts (`*.py`)**: Scripts for analyzing and visualizing the results, such as `magnetization_costs.py`.
*   **PNG Images (`*.png`)**: The output of the visualization scripts.

## Scripts

This directory also contains the `ising_ace_depth_series-high.sh` and `ising_ace_depth_series-low.sh` scripts, which are used to run the benchmarks with different patching strategies.

## Overview

This directory is a collection of benchmark runs with different SDRP parameter settings. Each subdirectory is a self-contained experiment with its own data, analysis scripts, and results.
