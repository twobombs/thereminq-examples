# Peaked: OTOC Maps Solver P1

This directory implements a physics-inspired solver based on **Out-of-Time-Order Correlators (OTOCs)** and **Quantum Echoes**. It treats the quantum circuit as a physical many-body system (specifically an Ising model) and looks for "Stability Plateaus" in the scrambling dynamics to identify the correct solution bitstring.

## Key Concepts
- **OTOC (Out-of-Time-Order Correlator)**: A measure of how information spreads (scrambles) in a quantum system.
- **Quantum Scrambling**: The solver assumes that the "correct" solution acts as a stable attractor in the chaotic scrambling dynamics of the circuit.
- **Stability Plateau**: A region in the time evolution ($t$) where the system settles into a stable state (the solution) before eventually thermalizing or scrambling completely.

## Files

- **`solution_p1_optimize-hybrid.py`**:
    - **Purpose**: Performs a "Micro-Time Scan" (Time $t$ from 0.01 to 0.55) to find the optimal scrambling time.
    - **Method**: It uses a hybrid CPU/GPU architecture to simulate the OTOC dynamics across a range of time parameters. It looks for a time $t$ where a dominant bitstring signal emerges (a peak in the distribution). This identifies the "Fast Scrambler" regime.

- **`solution_otoc_p1b.py`**:
    - **Purpose**: Verification and robust extraction of the solution.
    - **Method**: Once a candidate time window is found (e.g., $t \in [0.30, 0.45]$), this script performs a massive "Robust Stability Scan" (10 million+ shots) with ensemble averaging. It samples $t$ continuously within the window to ensure the solution is a physical attractor and not a transient artifact. It outputs the top 10 candidate bitstrings.

## Workflow
1.  **Scan for Signal**: Run `solution_p1_optimize-hybrid.py` to sweep through time parameters and find the "Stability Plateau" where a signal appears.
2.  **Robust Extraction**: Run `solution_otoc_p1b.py` focused on that time window to rigorously verify the solution and distinguish it from noise.
