# Peaked: Bruteforce

This directory contains a brute-force approach to solving the `P1_little_dimple.qasm` challenge using `pyqrack`.

## Files

- **`qpepyqrackqbdd.py`**: Runs a full Quantum Phase Estimation (QPE) simulation using the PyQrack simulator. It parses the QASM file, executes the gates (mapping generic QASM gates to PyQrack's API), measures the qubits, and attempts to decode the phase from the measurement results.


<img width="1600" height="800" alt="qrack_benchmark_3d" src="https://github.com/user-attachments/assets/eeced444-c28d-4d33-84b4-f7e8cea70cbd" />
