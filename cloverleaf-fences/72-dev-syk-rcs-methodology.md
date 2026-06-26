# Methodology

## A. Lattice Partitioning and Persistent Multiprocessing (The Holographic Bulk)

In this work, the simulated physical system consists of a $N = 72$ qubit register representing the holographic bulk, which is partitioned into four identical $18$-qubit subsystems, or "patches." To efficiently simulate the expansive Hilbert space and temporal dynamics, we employ a distributed tensor network approach mapped to isolated, persistent GPU workers via `pyqrack`. The $18$-qubit patches are structurally organized into $3 \times 6$ sub-lattices.

To avoid the exponential overhead of a monolithic state-vector simulation, the PyQrack engine initiates separate parallel processes, effectively casting each patch into an isolated computational universe residing persistently in GPU VRAM. Inter-Process Communication (IPC) pipes coordinate the evolution of these patches, synchronizing local operations and mediating interactions at the boundary fences, echoing distributed quantum simulation techniques that utilize clustered processing to overcome single-node memory barriers [arXiv:2407.19348]. This architecture enables the continuous simulation of highly entangled states without the constant overhead of memory reallocation, providing a faithful representation of separated conformal boundaries in the dual gravity theory.

## B. Random Circuit Sampling (RCS) as the Scrambling Mechanism (SYK-like dynamics)

A core requirement of traversable wormhole protocols is the presence of chaotic, fast-scrambling dynamics, traditionally modeled by the Sachdev-Ye-Kitaev (SYK) model. In our framework, we substitute explicit fermionic Hamiltonians with deep Random Circuit Sampling (RCS), a standard proxy for geometric information scrambling [arXiv:1806.02807]. Within each time step of the bulk evolution, a synchronized layer of pseudo-random unitary operations is applied independently to each isolated $18$-qubit patch.

Specifically, each qubit undergoes a local rotation chosen uniformly from $\{R_X(\theta), R_Y(\theta), R_Z(\theta)\}$, where the angle $\theta \in [-\pi, \pi]$ is drawn from a uniform distribution initialized by a synchronized global seed. This is immediately followed by a layer of fixed topological entanglement, where Control-NOT (CNOT) operations are applied across the specified intra-patch $(3 \times 6)$ edges. By varying the depth of this RCS sequence per discrete time step, the protocol induces rapid operator growth and multi-partite entanglement, serving as a computationally scalable analogue to the highly chaotic, all-to-all SYK scrambling dynamics.

## C. Boundary Operator Measurement and Left-Right Correlations

To implement the inter-boundary coupling necessary for wormhole traversability, we measure local observables at the conformal boundaries of the entangled patches. At the conclusion of each scrambling interval, the exact expectation value of the Pauli Z operator, $\langle Z \rangle = 1.0 - 2.0 P(1)$, is computed for all qubits designated as boundary nodes. This readout approach closely parallels techniques used in evaluating the boundaries of traversable wormhole protocols [arXiv:2102.00010].

These local expectation values are relayed to the central orchestration engine (the "Bulk Space") via IPC pipes. This measurement collapses the boundary state locally while preserving the bulk entanglement structure. The orchestrator subsequently calculates the mean-field left-right correlations between adjacent boundary pairs $(q_A, q_B)$ residing in neighboring patches $(p_A, p_B)$.

## D. The Mean-Field ER=EPR Protocol (Wormhole Coupling Kicks via Z-rotations)

The defining phase of the protocol is the application of a double-trace deformation linking the left and right conformal boundaries, rendering the corresponding Einstein-Rosen bridge (ER=EPR) traversable. As proposed by Gao, Jafferis, and Wall [arXiv:1608.05687], this requires applying a coupling unitary $U = \exp(i g Z_L Z_R)$ between the two sides.

Due to the isolated GPU environments, we approximate this two-body interaction using a mean-field kick protocol. For a boundary qubit $q_A$ with a target partner $q_B$, the coupling is applied via a conditional local phase rotation:
$$ R_Z(\theta) = \exp(-i \frac{\theta}{2} Z_A) $$
where the rotation angle is classically calculated by the orchestrator as $\theta = 2 g \langle Z_B \rangle$, with $g$ denoting the dimensionless coupling strength. This local kick mimics the effect of the full non-local double-trace deformation without requiring a global entangling gate across the IPC barrier. As described in the theoretical studies of Maldacena, Stanford, and Yang [arXiv:1704.05333], this deformation injects negative energy shockwaves into the bulk, altering the causal structure and allowing quantum information to pass between the previously isolated patches.

## E. Observable Tracking: Bulk Magnetization and Signal Propagation

To verify the successful teleportation of information through the simulated traversable wormhole, we continuously track global and boundary observables. The primary order parameter is the bulk magnetization, defined as the sum of all local $\langle Z \rangle$ expectation values across the entire $72$-qubit lattice.

Simultaneously, the orchestrator computes the average connected boundary correlation $\langle Z_A Z_B \rangle$ prior to the application of the coupling kicks. The temporal evolution of these observables allows us to monitor the propagation of a signal injected at one boundary as it scrambles into the bulk, traverses the ER bridge via the double-trace coupling, and refocuses at the distant boundary. The periodic measurement (e.g., every 5 time steps) of these macroscopic quantities provides the necessary signature of holographic teleportation, analogous to the transmission signals measured in experimental quantum simulations of traversable wormhole dynamics by teleportation by size [arXiv:1911.06314].