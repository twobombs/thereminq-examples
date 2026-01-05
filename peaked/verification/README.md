# Peaked: Verification

This directory contains scripts for generating custom quantum circuit challenges and verifying solver performance against them. These tools allow testing of solver robustness on "full density" and "mirror" circuit topologies.

## Files

- **`peaked_generation_pyqrack.py`**:
    - **Purpose**: Generates "Full Density" random circuits with a hidden target bitstring and attempts to recover it.
    - **Method**: Embeds a secret bitstring, applies dense layers of random unitaries and barriers to obfuscate it, and then solves it using `pyqrack` with specific environment tuning (`QRACK_QUNIT_SEPARABILITY_THRESHOLD`). It uses a consensus voting mechanism over multiple rounds to filter out noise.

- **`peaked_generation_pyqrack_mirrored.py`**:
    - **Purpose**: Generates and solves Linear Nearest Neighbor (LNN) "Brickwork Mountain" mirror circuits.
    - **Method**: Constructs a circuit that evolves forward in time and then reverses (mirrors) itself. This creates a deep, entangled identity operation ideal for testing simulator memory management and precision.

- **`peaked_generation_pyqrack_p1.py`**:
    - **Purpose**: A dedicated solver script for `P1_little_dimple.qasm` (or similar files) using the consensus approach.
    - **Method**: Runs the circuit multiple times on the `pyqrack` simulator, collecting measurement shots. It aggregates the results to form a bitwise consensus, providing a confidence score for each qubit's value. This statistical approach helps overcome the inherent noise/approximation in the high-performance simulation settings.

- **`peaked_generation_pyqrack_p1-hybrid.py`**:
    - **Purpose**: A hybrid solver script that combines Qiskit optimization with PyQrack simulation.
    - **Method**: Loads the circuit using Qiskit, performs Level 3 optimization (transpilation) to reduce gate count and depth, and then executes the optimized circuit on `pyqrack`. It uses "Oracle Mode" (direct state vector inspection) to find the theoretical peak probability, which is useful for verifying solutions against a known target or ground truth.

<img width="841" height="920" alt="Screenshot from 2026-01-04 20-11-43" src="https://github.com/user-attachments/assets/6405c92c-8676-4fca-b8fe-58a448f50cda" />
