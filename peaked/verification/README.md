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
