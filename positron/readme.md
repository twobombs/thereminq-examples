# Positronium Metrology: Unitary Inversion

This directory contains a Python script that implements and simulates the concepts presented in the arXiv paper "[2506.04315](https://arxiv.org/abs/2506.04315)", which proposes a novel quantum sensing protocol called "Positronium Metrology". The core idea is to use an entangled pair of a qubit and an "anti-qubit" to achieve superior sensitivity. The "anti-qubit" is simulated by applying the inverse of the unitary operation applied to the qubit.

## Files

*   **`unitary-inverse-metrology-example.py`**: A Python script that simulates and compares three different quantum sensing protocols:
    1.  **Positronium Metrology**: An entangled qubit-antiqubit pair.
    2.  **Entanglement-Free Sensing**: A separable qubit-antiqubit pair.
    3.  **Agnostic Sensing**: A standard entangled qubit pair.

    The script uses PyQrack to simulate the protocols and then plots the results, comparing them to the theoretical predictions. It also includes an analysis of the Fisher Information for each protocol.

## Overview

The simulation demonstrates the key advantage of the Positronium Metrology protocol: it is independent of the rotation axis of the unitary operation, and it achieves a higher Fisher Information than the other two protocols, indicating a higher sensitivity. The creative framing of this concept with the movie "Tenet" highlights the idea of "inverting" the evolution of one of the entangled particles.

![tenet-tenet-movie](https://github.com/user-attachments/assets/6aa146b1-d995-43f9-944b-aa223480eed4)
