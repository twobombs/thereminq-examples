# High-Performance Quantum Fourier Transform Implementation in the ThereminQ Ecosystem

**Authors:** ThereminQ Contributors (twobombs/thereminq-examples)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract
This repository directory contains advanced, high-performance implementations of the Quantum Fourier Transform (QFT) and its semiclassical variants, built upon the ThereminQ simulation ecosystem and the PyQrack framework. The QFT is a fundamental primitive in quantum algorithms, enabling exponential speedups in applications such as integer factorization (Shor's algorithm) and Quantum Phase Estimation. However, classically simulating dense, highly entangled QFT circuits is notoriously challenging due to the exponential growth of the state vector. By leveraging ThereminQ's distributed architecture and PyQrack's GPU-accelerated state vector simulation, these implementations demonstrate state-of-the-art classical emulation. We provide a semiclassical Shor factoring engine that minimizes qubit requirements, and a novel "condensate" simulation engine that executes a massive circuit of circuits using rotating classical seams, circumventing the need for a full global state vector.

## 1. Introduction
The Quantum Fourier Transform (QFT) is arguably the most critical subroutine in the quantum computing algorithmic toolkit. It serves as the backbone for Quantum Phase Estimation (QPE) and Shor's algorithm for factoring large integers, tasks for which quantum computers offer a theoretical exponential speedup over the best known classical algorithms.

The primary challenge in simulating the QFT on classical hardware stems from its capability to generate massive amounts of entanglement across all participating qubits. As the number of qubits $n$ increases, a full state vector simulation requires $\mathcal{O}(2^n)$ memory, rendering large-scale direct emulation intractable. To address this, optimized compilation strategies and semiclassical approximations are heavily utilized. The ThereminQ ecosystem, powered by the Qrack framework, is designed to push the boundaries of what is possible on classical hardware by employing GPU acceleration, tensor network techniques, and algorithmic optimizations such as the semiclassical QFT.

## 2. Theoretical Background
The Quantum Fourier Transform acts on a quantum state by mapping the computational basis state $|x\rangle$ to a superposition of all basis states, weighted by complex phases. The formal mathematical definition for an $n$-qubit system is given by:

$$
\text{QFT} |x\rangle = \frac{1}{\sqrt{2^n}} \sum_{y=0}^{2^n-1} \exp\left( -2 \pi i \frac{x \cdot \text{rev}(y)}{2^n} \right) |y\rangle
$$

where $\text{rev}(y)$ denotes the bit-reversed integer of $y$. Similarly, the Inverse Quantum Fourier Transform (IQFT) is defined as:

$$
\text{IQFT} |z\rangle = \frac{1}{\sqrt{2^n}} \sum_{x=0}^{2^n-1} \exp\left( +2 \pi i \frac{x \cdot \text{rev}(z)}{2^n} \right) |x\rangle
$$

The standard quantum circuit for the QFT decomposes this transformation into a sequence of Hadamard ($H$) gates and controlled-phase rotation ($R_k$) gates. The $R_k$ gate applies a phase shift of $2\pi / 2^k$.

## 3. Implementation Architecture (The Deep Dive)

The `qft` directory contains two primary implementations that showcase different classical simulation strategies.

### 3.1 Semiclassical Shor Factoring Engine (`qft-condensate-semiclassical.py`)
This script implements a space-optimized variant of Shor's algorithm using the Beauregard layout. Standard Shor's algorithm requires a $2n$-qubit counting register to hold the QFT state. This implementation drastically reduces the memory footprint by utilizing a **semiclassical inverse QFT**.

- **Resource Allocation:** Instead of a full $2n$-qubit register, the circuit recycles a **single control qubit** $2n$ times. The measurement outcome of this qubit is fed forward to determine subsequent classically-controlled rotation corrections. This ensures that the massive counting register never exists as a full state vector.
- **Work Register:** An $n$-qubit register ($q$) and an $n$-qubit scratchpad ($o$) are maintained. Controlled modular multiplication $a^{2^j} \bmod N$ is applied in-place using PyQrack's highly optimized permutation-level arithmetic functions (e.g., `sim.mcmuln`, `sim.mcdivn`).

```python
# Snippet demonstrating classical feed-forward semiclassical QFT
for j in range(t - 1, -1, -1):
    if sim.m(ctrl): sim.x(ctrl) # Recycle qubit
    sim.h(ctrl)
    # ... modular exponentiation logic ...
    if R: sim.u(ctrl, 0.0, 0.0, -2 * math.pi * R) # Semiclassical phase correction
    sim.h(ctrl)
    b = sim.m(ctrl)
    R = (R + b / 2) / 2
```

### 3.2 QFT-Cosmos Condensate (`qft-cosmos-classical-condensate.py`)
This engine simulates an enormously wide quantum circuit by fracturing it into a "condensate" of smaller $p$-qubit patches. It operates as a circuit of circuits with rotating seams.

- **Classical Seams:** Between layers, qubits are measured and re-prepared based on the classical outcomes (classical feed-forward). Because every patch begins with a product state and ends in a $Z$-basis measurement, the global output distribution factorizes perfectly into a chain of exact conditional probabilities.
- **Scalability:** By breaking the entanglement across seams, this script achieves exact sampling at arbitrary width and depth with a computational cost of $\mathcal{O}(W \cdot L \cdot \text{shots})$, avoiding a global state vector entirely.

```python
# Snippet demonstrating exact local patch simulation without global state vector
bits, lnp = chain_sample(A, B, rng, kind)
bits = bits.reshape(p, len(ms), -1).transpose(1, 0, 2).reshape(len(ms) * p, -1)
new_y[q] = bits
z_total += (p * math.log(2) + lnp).reshape(len(ms), -1).sum(axis=0)
```

## 4. Usage and Reproducibility

### Environment Setup
The simulation relies on PyQrack, which requires the underlying Qrack C++ library (and ideally an OpenCL runtime for GPU acceleration).

```bash
pip install pyqrack numpy
```

### Running the Examples
**Semiclassical Shor's Algorithm:**
To factor integers utilizing the semiclassical optimization (with default GPU acceleration):
```bash
python3 qft-condensate-semiclassical.py 15 21 35 143 221 --shots 32
```
This will output the log-likelihood gap against ideal draws, order finding success rates, and the discovered prime factors.

**QFT Condensate Simulation:**
To run a massive distributed condensate simulation (e.g., width 100,000 qubits cut into 8-qubit patches):
```bash
python3 qft-cosmos-classical-condensate.py --width 100000 --patch 8 --layers 10 --shots 64
```

## 5. References
1. Strano, D. et al. (2023). "Exact and approximate simulation of large Quantum Circuits." [arXiv:2304.14969](https://arxiv.org/abs/2304.14969)
2. Coppersmith, D. (2002). "An approximate Fourier transform useful in quantum factoring." [arXiv:quant-ph/0201067](https://arxiv.org/abs/quant-ph/0201067)
3. Beauregard, S. (2003). "Circuit for Shor's algorithm using 2n+3 qubits." [arXiv:quant-ph/0205095](https://arxiv.org/abs/quant-ph/0205095)
4. Farhi, E., Goldstone, J., Gosset, D., Gutmann, S., Meyer, H. B., & Shor, P. (2009). "Quantum Adiabatic Algorithms, Small Gaps, and Different Paths." [arXiv:0909.4766](https://arxiv.org/abs/0909.4766)