# Hhl

This directory implements the Harrow-Hassidim-Lloyd (HHL) algorithm for solving linear systems of equations (`Ax = b`) exponentially faster than classical counterparts, under certain conditions. The `hhl.py` script constructs the requisite quantum subroutines: state preparation to load `b`, Quantum Phase Estimation (QPE) to extract eigenvalues of the Hermitian matrix `A`, controlled rotations for eigenvalue inversion, and uncomputation. This algorithm is foundational for quantum machine learning and differential equation solving.

## Core Quantum Mechanical Concepts & ArXiv References

* [0811.3171] Quantum algorithm for linear systems of equations (https://arxiv.org/abs/0811.3171)
* [1302.1210] Quantum algorithms for curve fitting (https://arxiv.org/abs/1302.1210)
