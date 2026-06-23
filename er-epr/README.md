# ER=EPR Wormhole Teleportation

This directory implements a quantum simulation of the "Wormhole Teleportation" protocol, inspired by the ER=EPR conjecture, using the high-performance `pyqrack` simulator.

## Files

- **`pyqrack.py`**: A Python script that simulates the traversal of a quantum signal through a wormhole. The simulation steps include:
    1.  **TFD Preparation**: creating a Thermofield Double state using entangled Bell pairs between two subsystems (Left and Right).
    2.  **Scrambling**: evolving the system forward and backward in time under an Ising Hamiltonian to mimic the chaotic dynamics of a black hole.
    3.  **Message Injection**: inserting a message qubit into the Left boundary.
    4.  **Shockwave**: applying a coupling interaction ("negative-energy shockwave") between the two sides to render the wormhole traversable.
    5.  **Decoding & Verification**: evolving the system to unscramble the information and measuring the expectation value on the Right boundary to confirm successful teleportation.


## Core Quantum Mechanical Concepts & ArXiv References

* [1306.0533] Cool horizons for entangled black holes (ER=EPR) (https://arxiv.org/abs/1306.0533)
* [1608.05687] Traversable wormholes via a double trace deformation (https://arxiv.org/abs/1608.05687)
* [2211.12038] Traversable wormhole dynamics on a quantum processor (https://arxiv.org/abs/2211.12038)
