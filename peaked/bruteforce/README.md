# Peaked: Bruteforce

This directory contains a brute-force approach to solving the `P1_little_dimple.qasm` challenge using `pyqrack`.

## Files

- **`qpepyqrack.py`**: The core simulation script. It reads `P1_little_dimple.qasm`, initializes a `QrackSimulator`, executes the gates (mapping QASM gates like `u`, `cz`, `cx` to PyQrack), and measures the final state to estimate the phase. It saves the results (Separability, Duration, Phase, Bitstring) to `qrack_results.csv`.
- **`qpepyqrackqbdd.py`**: A variant or legacy version of the main simulation script, also using `pyqrack` for QPE simulation.
- **`qpepyqrackqbd_viz.py`**: A visualization script that reads the CSV output (`qrack_results.csv` or similar) and generates a 3D scatter plot (Timestamp vs. Phase vs. Probability) to visualize the convergence of the simulation.
- **`qpepyqrack.sh`**: A shell script wrapper to execute the `qpepyqrack.py` solver.

## Usage
Run the shell script to start the solver:
```bash
./qpepyqrack.sh
```
Or run the python script directly:
```bash
python qpepyqrack.py
```

<img width="1600" height="800" alt="qrack_benchmark_3d" src="https://github.com/user-attachments/assets/eeced444-c28d-4d33-84b4-f7e8cea70cbd" />
