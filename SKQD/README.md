# SKQD (Stochastic Krylov Quantum Dynamics) Module Analysis

## Overview
The `Braided-PyQrackIsing-SKQD.PY` script is a high-performance, distributed computing pipeline designed to estimate the ground state energy of a 32-qubit Transverse-Field Ising Model (TFIM). It achieves this by employing a hybrid quantum-classical algorithm that partitions the large quantum system into smaller subsystems, simulates them in parallel, and uses the combined results to seed a classical subspace eigenvalue solver.

## Architecture & Communication
The system is built on a highly concurrent architecture leveraging Python's `multiprocessing` and the `ZeroMQ` (ZMQ) messaging library.
- **Orchestrator-Worker Model:** The `SKQDOrchestrator` acts as the master node, dispatching tasks to multiple worker processes (daemons) and collecting their results.
- **ZeroMQ Topology:**
  - **Ventilator (PUSH):** Distributes parameters (basis permutations, grid ID, time step) to workers.
  - **Sink (PULL):** Collects the computed probability distributions from the workers.
  - **Control & Sync (PUB/SUB, PULL/PUSH):** Handles deterministic handshake barriers for startup synchronization and broadcasts shutdown ("KILL") signals to gracefully terminate the workers.
- **Local Quantum Simulation:** Each worker utilizes `PyQrack` (a high-performance quantum simulator) to simulate local 9-qubit subsystems, avoiding the exponential memory cost of directly simulating 32 qubits.

## Algorithmic Pipeline

### 1. Quantum Probability Distribution (QPD) & Topology Cut
The 32-qubit system is divided into four smaller sections (Grids 0, 1, 2, and 3). The entanglement between these grids is severed ("cut") and represented by a sum over Pauli bases (I, X, Y, Z). With 4 cuts, this generates $4^4 = 256$ different basis permutations.

### 2. Multi-Pass Krylov Seeding
To identify the most critical quantum states (the "heavy" states) that contribute to the ground state, the orchestrator performs multiple passes at varying time evolution steps (`dt = 0.0, 0.1, 0.25, 0.5`).
- For every permutation, time step, and grid, a worker simulates the local grid.
- **Evolution Kernel:** The worker applies the corresponding basis rotations, evolves the state using a true Krylov evolution via `pyqrack_ising`'s `IsingModel1D` (or a Trotterized CX/RX fallback if missing), and applies reverse rotations.
- **Heavy State Selection (HSS):** To save memory and network bandwidth, the worker prunes the output state vector, sending back only the probabilities of the top 32 basis states.

### 3. Tensor Contraction & Subspace Extraction
The orchestrator aggregates the pruned probabilities received from the Sink into four 3D tensors (`T0, T1, T2, T3`).
- Using `opt_einsum`, it contracts these four local tensors to reconstruct a sparse representation of the global 32-qubit state.
- It then extracts a "seed subspace"—the bitstrings (up to 500) corresponding to the highest probabilities in the contracted global tensor.

### 4. Classical Configuration Recovery
With the initial seed subspace identified, the pipeline transitions to a classical refinement phase to find the exact ground state energy.
- **Sparse Hamiltonian Construction:** For the active subspace of 32-qubit bitstrings, a localized, sparse Hamiltonian matrix is constructed. The diagonal elements represent the ZZ coupling energy (Ferromagnetic), and off-diagonal elements represent the X transverse field connectivity (single bit flips).
- **Subspace Diagonalization:** The pipeline calculates the lowest eigenvalue ($E_0$) and eigenvector of this sparse matrix using SciPy's `eigsh` (or dense `eigh` if the subspace is tiny).
- **Subspace Expansion:** The highest probability configurations from the resulting eigenvector are identified. Their single-bit-flip neighbors are added to the active subspace (mimicking the transverse field), expanding the support for the next iteration.
- **Convergence:** The Hamiltonian is rebuilt and re-diagonalized with the expanded subspace. This iteratively continues until the ground state energy change falls below a strict tolerance (`1e-5`), or a hard memory limit on the subspace size is reached.

## Dependencies
- `numpy` & `scipy` (Sparse matrix construction and eigenvalue solvers)
- `opt_einsum` (Optimized tensor contraction)
- `zmq` (ZeroMQ networking)
- `multiprocessing` (Parallel worker processes)
- `pyqrack` (High-performance GPU-accelerated quantum simulation)
- `pyqrack_ising` (Optional native module for optimized Ising model Krylov evolution)
