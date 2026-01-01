# Peaked: Holographic Solver P1

This directory contains tools for a **Holographic / Stabilizer** approach to solving quantum circuit challenges. It focuses on simplifying the circuit by identifying and exploiting stabilizer structures, essentially "purifying" the circuit to finding exact solutions in the presence of noise or complexity.

## Key Concepts
- **Stabilizer States**: Many quantum circuits are dominated by "Clifford" gates (H, S, CNOT, CZ) which can be simulated efficiently. This solver attempts to map the circuit to a stabilizer state.
- **Circuit Purification**: The `extract_peak.py` script "snaps" continuous rotation parameters to the nearest Clifford angles (multiples of $\pi/2$) to remove "thermal noise" and reveal the underlying structural core.
- **Holographic Bulk**: The naming suggests an approach inspired by the holographic principle (AdS/CFT), likely treating the circuit as a tensor network or bulk geometry where the solution lives on the boundary.

## Files

- **`extract_peak.py`**:
    - **Purpose**: Extracts the "peaked" bitstring by purifying the circuit.
    - **Method**: It parses the QASM file, keeps native stabilizer gates (CZ, CX, H, S, etc.), and converts parameterized 'u' gates to exact Clifford gates if they are close to multiples of $\pi/2$. It then simulates this "clean" circuit using `AerSimulator(method='stabilizer')` to find the deterministic outcome.

- **`analyse_stabilizers.py`**:
    - **Purpose**: Analyzes the stabilizer structure of the circuit to determine its complexity and theoretical output probability.
    - **Method**: It calculates the Clifford Tableau and counts the number of deterministic Z-constraints versus X/Y components. This allows it to compute the "Superposition Dimensions" and the exact theoretical probability of the output bitstring (e.g., verifying if the solution is unique or one of many).

- **`solve_pyqrack.py`**:
    - **Purpose**: A direct simulator for the purified or core circuit using `PyQrack`.
    - **Method**: It maps the QASM gates to PyQrack's highly optimized GPU-accelerated simulator to find the peaked bitstring directly. It's useful for verifying the results of the stabilizer analysis.

- **`holographic_bulk.py`**:
    - **Status**: Binary file.
    - **Description**: Likely contains serialized data, a pre-computed model, or a core binary component related to the holographic tensor network representation.

## Usage
1.  **Analyze Structure**: Run `python analyse_stabilizers.py` to understand the circuit's theoretical properties (determinism vs. superposition).
2.  **Extract Solution**: Run `python extract_peak.py` to attempt to find the solution by purifying the circuit.
3.  **Verify**: Run `python solve_pyqrack.py` to confirm the result with a full state vector simulation (if feasible) or optimized stabilizer simulation.
