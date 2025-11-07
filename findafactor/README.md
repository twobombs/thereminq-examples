# Find a Factor

This directory contains a shell script for running a factorization benchmark.

## Files

*   **`run-findafactor.sh`**: A shell script that acts as a wrapper for a Python script located at `/FindAFactor/find_a_factor`. It generates pairs of large prime numbers, calculates their product, and then runs the Python script to factor the product. The script iteratively increases the size of the numbers to be factored, providing a benchmark for the factorization script.
