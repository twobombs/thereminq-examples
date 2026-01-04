# Peaked: Bruteforce

This directory contains a brute-force approach to solving the `P1_little_dimple.qasm` challenge using `pyqrack`.

## Files

- **`qpepyqrack.py`**: The core simulation script. It reads `P1_little_dimple.qasm`, initializes a `QrackSimulator`, executes the gates (mapping QASM gates like `u`, `cz`, `cx` to PyQrack), and measures the final state to estimate the phase. It saves the results (Separability, Duration, Phase, Bitstring) to `qrack_results.csv`.
- **`qpepyqrackqbdd.py`**: A variant of the main simulation script that explicitly enables the Quantum Binary Decision Diagram (QBDD) engine (`isBinaryDecisionTree=True`) in the `QrackSimulator` for potentially enhanced performance on certain circuit structures.
- **`qpepyqrackqbd_viz.py`**: A visualization script that reads the CSV output (`qrack_results.csv`) and generates a 3D scatter plot (Max Paging Qubits vs. Separability Threshold vs. Duration/Phase) to visualize the performance and convergence of the simulation sweeps.
- **`qpepyqrack.sh`**: A shell script wrapper to execute a parameter sweep of the `qpepyqrack.py` solver. It iterates through different `QRACK_MAX_PAGING_QB` and `QRACK_QUNIT_SEPARABILITY_THRESHOLD` values, logging the results to `qpepyqrack.log` and `qrack_results.csv`.

## Usage
Run the shell script to start the solver sweep:
```bash
./qpepyqrack.sh
```
Or run the python script directly:
```bash
python qpepyqrack.py
```

<img width="1600" height="800" alt="qrack_benchmark_3d" src="https://github.com/user-attachments/assets/eeced444-c28d-4d33-84b4-f7e8cea70cbd" />
