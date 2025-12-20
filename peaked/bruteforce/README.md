# Peaked: Bruteforce

This directory contains a brute-force approach to solving the `P1_little_dimple.qasm` challenge using `pyqrack`.

## Files

- **`qpepyqrackqbdd.py`**: Runs a full Quantum Phase Estimation (QPE) simulation using the PyQrack simulator. It parses the QASM file, executes the gates (mapping generic QASM gates to PyQrack's API), measures the qubits, and attempts to decode the phase from the measurement results.
