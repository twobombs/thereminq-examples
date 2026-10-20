# Peaked: Solvers

This directory contains specialized solvers developed to tackle the `P1_little_dimple.qasm` challenge using different theoretical frameworks. Each subdirectory represents a distinct approach.

## Subdirectories

### `clustered_angles_solver_P1/`
- **Method**: Analytical clustering of rotation angles. This approach analyzes the distribution of rotation parameters in the circuit to identify non-random clusters that may hint at the solution.
- **Key Scripts**:
    - `qpetensor.py`: Parses QASM to a tensor representation.
    - `render_landscape.py`: Visualizes the angle distribution landscape.
    - `solve_dimple.py`: The main solver script that clusters angles to find hidden signals.

### `haar_solver_P1/`
- **Method**: Statistical deviation from Haar randomness. This approach assumes that the "hidden" structure manifests as deviations from the expected random distribution of quantum gates (Haar measure), specifically looking for "magic angles" or "crystal" structures.
- **Key Scripts**:
    - `haar-deviation.py`: Analyzes the statistical distribution of rotation angles to detect deviations from randomness and plots the location of "magic" angles (like pi/2).
    - `purified-viz.py`: Visualizes the "purified" circuit structure.
    - `purify-deviation.py`: Attempts to "purify" the circuit by snapping near-Clifford gates to exact Cliffords.
    - `transpile-deviation.py`: Transpilation utilities to normalize the circuit for deviation analysis.

### `holographic_solver_P1/`
- **Method**: Stabilizer purification and holographic bulk analysis. This method treats the circuit as a tensor network or a holographic system, attempting to extract the "peak" by analyzing stabilizer properties and "snapping" parameters to Clifford gates.
- **Key Scripts**:
    - `extract_peak.py`: Attempts to "snap" rotation parameters to the nearest Clifford gates to simplify the circuit and find the peak.
    - `analyse_stabilizers.py`: Analyzes the dimensions of superposition to guide the solver.
    - `holographic_bulk.py`: Core logic for the holographic analysis.
    - `solve_pyqrack.py`: A solver leveraging `pyqrack` in the context of this holographic approach.

### `otoc_maps_solver_P1/`
- **Method**: Physics-inspired Out-of-Time-Order Correlators (OTOCs) and scrambling stability. This solver uses the concept of scrambling and chaos to find the solution, assuming the "hidden" bitstring corresponds to a stable plateau in the OTOC landscape.
- **Key Scripts**:
    - `solution_p1_optimize-hybrid.py`: Scans for stability plateaus using optimization techniques.
    - `solution_otoc_p1b.py`: Uses robust ensemble averaging of OTOCs to identify the solution.
