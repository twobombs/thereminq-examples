# Exact Free-Fermion Anchors and CP-SAT Tiling for the Hyperoctagon Kitaev Model

## Abstract
This repository contains the simulation engine and formal mathematical framework for executing the Kitaev model on the three-dimensional hyperoctagon lattice, also known as the $(10,3)\text{-}a$ or $K_4$ crystal. By leveraging CP-SAT constraint programming, we identify $32 \times 27$ equivariant tilings, extracting 27-qubit patch manifolds (racetracks) suitable for distributed quantum simulation. We present a methodology for exact free-fermion anchor validation and Gaussian seam repair using the exact entanglement Hamiltonian $A_{\text{ent}}$ to generate the boundary correction $\Sigma$. We also detail state-vector execution on these 27-qubit blocks using the `k4_pyqrack_driver.py` native driver.

## 1. Introduction and Model
The Kitaev model on the hyperoctagon lattice represents a three-dimensional exactly solvable spin model that harbors a quantum spin liquid ground state. Crucially, the gapless Majorana modes form an extended two-dimensional Majorana Fermi surface, inherently protected by lattice symmetries.

The Hamiltonian is given by:
$$ H = -J_x \sum_{\langle i,j \rangle_x} \sigma_i^x \sigma_j^x - J_y \sum_{\langle i,j \rangle_y} \sigma_i^y \sigma_j^y - J_z \sum_{\langle i,j \rangle_z} \sigma_i^z \sigma_j^z $$
where $\sigma^\alpha$ are Pauli matrices, and the summation runs over the nearest-neighbor bonds of the $x, y, z$ types on the hyperoctagon lattice. In the thermodynamic limit, the ground state energy is analytically bound (e.g., benchmarked at optimal $-0.769884$).

## 2. Methods: Equivariant Tiling and Seam Repair
### 2.1 Equivariant Tiling
To partition the infinite lattice for finite-size tensor network or state-vector simulation, we employ a constraint programming approach (CP-SAT) to identify $32 \times 27$ equivariant tilings. The translation symmetry of the hyperoctagon lattice allows us to reduce the problem down to 8 distinct block types through a $\mathbb{Z}_2 \times \mathbb{Z}_2$ subgroup quotient graph lifting.

### 2.2 Manifold Extraction and Seam Repair
The extraction of 27-qubit patch manifolds (racetracks) introduces artificial boundaries (seams) that break translational invariance. To validate these patches, we compute an exact free-fermion anchor.

The seam repair methodology utilizes the exact entanglement Hamiltonian $A_{\text{ent}}$ derived from the free-fermion spectrum. We generate a Gaussian boundary correction $\Sigma$:
$$ \Sigma = \int e^{i A_{\text{ent}}} d\tau $$
which effectively repairs the quotient graph boundaries.

### 2.3 Flux Operators and Coils
We utilize a Hadamard test measurement to evaluate the 10-qubit flux word $W$, ensuring the system remains in the physical vortex-free sector:
$$ W = \prod_{p \in \text{loop}} \sigma_p^\alpha $$
Furthermore, non-contractible tree-site coils are formulated. The 3D winding vectors of these coils are computed to extract the twist-response tensor of the Majorana Fermi surface, providing order parameters for the topological phase.

## 3. Execution and Simulation Instructions

The simulation pipeline involves two primary components: the QASM generator `K4-qasm.py` and the native Python execution driver `k4_pyqrack_driver.py`.

### 3.1 Running the Simulation
To execute the simulation, first generate the appropriate QASM representations or leverage the pre-compiled manifests, then invoke the native driver:

```bash
python3 K4-qasm.py
python3 k4_pyqrack_driver.py --block 0 --theta 0.1 --steps 4
```

### 3.2 Technical Limitations and the 5-Gate Decomposition
A known technical limitation exists in the current pyqrack build utilized by `k4_pyqrack_driver.py`. While Qrack possesses a highly optimized native method `exp(b, ph, q)` that applies the Kitaev bond $e^{-i\theta P \otimes P}$ in a single call, a ctypes array type mismatch (passing `ulonglong*` instead of `int*`) prevents its use.
Consequently, the driver explicitly falls back to a 5-gate decomposition using standard Pauli basis changes for all Trotter steps.

### 3.2 Environment Configuration
The standard deployment environment relies strictly on Ubuntu, utilizing open GPU drivers via Mesa. Proprietary drivers may induce unpredictable behavior in the OpenCL state management.

### 3.3 Hardware Isolation
For multi-GPU cluster execution, strict hardware isolation must be enforced. You are mandated to explicitly assign the OpenCL environment variables alongside standard device variables to ensure correct OpenCL isolation:
```bash
export QRACK_QPAGER_DEVICES=0
export QRACK_QUNITMULTI_DEVICES=0
```
Failure to assign these will result in memory collision across GPU contexts.

### 3.4 State Management
The simulator environment must be explicitly reset between consecutive simulation runs. Stale state-vector data from previous Trotter steps will accumulate if the `QrackSimulator` instance is not torn down and re-instantiated per block or run.

## 4. References
1. [arXiv:cond-mat/0506438](https://arxiv.org/abs/cond-mat/0506438) - Kitaev, A. "Anyons in an exactly solved model and beyond". *Annals of Physics* 321, 2 (2006).
2. [arXiv:1401.7678](https://arxiv.org/abs/1401.7678) - Hermanns, M. and Trebst, S. "Quantum spin liquid with a Majorana Fermi surface on the three-dimensional hyperoctagon lattice". *Phys. Rev. B* 89, 235102 (2014).
3. [arXiv:quant-ph/9707021](https://arxiv.org/abs/quant-ph/9707021) - Kitaev, A. Yu. "Fault-tolerant quantum computation by anyons". *Annals of Physics* 303, 2-30 (2003).
