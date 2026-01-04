# Peaked: Simulation

This directory contains scripts for generating, manipulating, and analyzing the connectivity of "Little Dimple" style quantum circuits.

## Files

- **`resize.py`**: A script to rebuild the "Little Dimple" circuit structure with a variable number of qubits (width) and depth. It preserves the original random connection texture and depth characteristics. It also generates a connectivity heatmap of the resulting circuit.
- **`hide+steer_qrack.py`**: A generator script that builds a random circuit and then "steers" the final state towards a specific target bitstring by applying corrective single-qubit rotations in the final layer. It uses `pyqrack` for high-performance simulation to determining the "natural winner" before steering.
- **`connectivityplot.py`**: A standalone tool to visualize the connectivity matrix of any given QASM file as a heatmap. It includes robust QASM loading logic to handle missing gate definitions (like `u` gates).

## Usage

**Resize a circuit:**
```bash
python resize.py -n 50 -d 800 -o custom_dimple.qasm
```

**Generate a steered circuit:**
```bash
python hide+steer_qrack.py -q 24 -d 200 --qasm
```

**Plot connectivity:**
```bash
python connectivityplot.py custom_dimple.qasm
```
<img width="1352" height="557" alt="Screenshot from 2026-01-04 11-59-53" src="https://github.com/user-attachments/assets/90385408-dec7-4e9f-8efe-16048c0a63a9" />
