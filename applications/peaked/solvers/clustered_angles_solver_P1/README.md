# Peaked: Solver 1

This directory contains a set of tools for analyzing the "landscape" of a quantum circuit to find hidden signals or "dimples" in the rotation angles.

## Files

- **`qpetensor.py`**: A utility script that converts a QASM file into a structural PyTorch tensor. The tensor representation `[GateID, Q1, Q2, P1, P2, P3]` facilitates easy processing for visualization or analysis tools.
- **`render_landscape.py`**: Generates a 3D scatter plot of the circuit's gates, mapping Time (Sequence) vs. Space (Qubit) vs. Energy (Rotation Angle). It highlights specific "signal" angles in red against a background of "noise" to visually identify the hidden information.
- **`solve_dimple.py`**: An analytical solver that attempts to extract the hidden signal (the "dimple") by clustering the rotation angles found in the QASM file. It filters out structural gates (like $\pi$, $\pi/2$) and looks for dominant clusters of non-standard angles to decode the hidden phase/flag.
