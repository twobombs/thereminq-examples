# Investigating the Kitaev Model and XYZ Hamiltonian Dynamics on Tricoordinated Spherical Shells within the Cloverleaf-Fences Topology

<img width="2816" height="1536" alt="Gemini_Generated_Image_tqdzautqdzautqdz" src="https://github.com/user-attachments/assets/bd0cea36-aa14-43c4-97e4-99aea16b3292" />

**Authors:** ThereminQ Development Team / Ai generated

**Abstract:**
This module presents a high-throughput computational pipeline designed to simulate quantum spin models—specifically the Kitaev honeycomb model and the Heisenberg XYZ model—on a stack of closed spherical shells (dodecahedra). Operating within the "cloverleaf-fences" multi-boundary topology, the framework leverages Local Hidden Variable (LHV) approximations combined with a Neural Network (NN4K) ansatz to analyze strongly correlated quantum states across highly connected manifolds. The core implementation, `shell_dodec_kitaev.py`, constructs a robust tricoordinated, bipartite 3-space graph of arbitrary shell depths, orchestrating exact free-fermion diagonalization for any given gauge sector. Unlike the standard hyperoctagon lattice where an exact uniform anchor exists, the radial couplings in this spherical geometry induce geometric frustration in the flux sectors, necessitating stochastic annealing to estimate the true ground state. Powered by the ThereminQ ecosystem and PyQrack for GPU-accelerated tensor contraction and statevector scaling, this pipeline enables the exploration of loop-error suppression, Automatic Circuit Elision (ACE), and boundary mean-field constraints. The orchestrated shell scripts provide a reproducible benchmark suite for evaluating finite-size scaling, simulated annealing performance, and multi-GPU tensor network distributions.

## 1. Introduction
The simulation of macroscopic quantum spin networks presents a significant computational challenge, particularly when dealing with highly frustrated geometries and long-range entanglement. Within the ThereminQ ecosystem, the `cloverleaf-fences` project investigates boundary-driven embedding methods, where the "cloverleaf-fences" topology acts as a multi-boundary manifold cluster designed to suppress loop errors and stabilize tensor contractions. This directory extends that work into closed spherical topologies, specifically stacks of tricoordinated dodecahedra.

By mapping the Heisenberg XYZ spin interactions onto this geometry, we explore how local hidden variable (LHV) constraints can approximate the macroscopic limit of the model. Furthermore, Neural Network Quantum States (NQS), specifically the NN4K (Neural Network for K4-manifolds) architecture, are employed to parameterize the complex wavefunctions of these highly correlated systems. This approach bridges the exact free-fermion solvability of the Kitaev model with the general applicability of machine learning techniques in predicting ground-state energies across frustrated flux sectors.

## 2. Theoretical Framework
The foundational mathematical model of this pipeline is the Heisenberg XYZ Hamiltonian on a structured graph:
$$ H = \sum_{\langle i,j \rangle} (J_x X_i X_j + J_y Y_i Y_j + J_z Z_i Z_j) $$
where $X, Y, Z$ are the standard Pauli operators and $J_x, J_y, J_z$ represent anisotropic coupling strengths. When mapped onto the dodecahedral shell stack, the degree-3 coordination allows a natural 3-edge coloring, mapping directly to the Kitaev interactions.

**Local Hidden Variables (LHV):**
In large-scale limits where exact statevector simulations become intractable, the system boundaries are approximated using LHV models. The LHV constraint posits that the marginal distributions of boundary measurements can be sampled using a classical probability distribution $\rho(\lambda)$, effectively bounding the quantum correlations:
$$ P(a, b | x, y) = \int d\lambda \rho(\lambda) P(a | x, \lambda) P(b | y, \lambda) $$

**Neural Network Ansatz (NN4K):**
To circumvent the exponential scaling of the Hilbert space, the NN4K architecture employs a generalized multi-layer perceptron architecture to encode the probability amplitudes of the quantum state:
$$ \Psi(S) = \exp \left( \sum_i a_i S_i + \sum_{i, j} W_{ij} S_i h_j \right) $$
where $S_i$ are the physical spin configurations and $h_j$ are the hidden layer variables. This ansatz is variationally optimized to minimize the energy expectation value $\langle \Psi | H | \Psi \rangle$.

## 3. Pipeline Implementation (The Shell Scripts)
The execution pipeline is driven by shell scripts that interface directly with the underlying Python engines.

### `benchtest.sh`
This script serves as the primary orchestration sequence, executing multiple configurations of the spherical shell lattice:
1. **Default Execution:** `python3 shell_dodec_kitaev.py`
   Builds the base lattice (4 shells, closed stack, 200 sites), verifies the 3-edge coloring and bipartite structure, and computes the exact free-fermion ground-state energy for the uniform gauge.
2. **Simulated Annealing:** `python3 shell_dodec_kitaev.py --anneal 3000`
   Introduces stochastic gauge optimization to search for the true ground state. Because the radial bonds create 8-loops that frustrate the uniform 10-loop gauge, this step utilizes 3000 annealing steps to find a minimal energy configuration.
3. **Scaled Capped Execution:** `python3 shell_dodec_kitaev.py --shells 8 --core`
   Expands the geometry to 8 shells and caps the inner boundary with core joints. This closes the internal boundary, reducing the girth from 8 to 6 at the core but ensuring a structurally distinct open outer surface.
4. **Hyperoctagon Cross-Check:** `python3 shell_dodec_kitaev.py --hyperoctagon ../coils/K4-Chrystalstacks-Kitaev-single.py`
   Validates the solver by running the exact free-fermion diagonalization on the established hyperoctagon lattice ($L=6$), verifying that the energy matches published finite-size scaling anchors.

## 4. Usage & Reproducibility
To reproduce the experiments in this repository, ensure that the ThereminQ Python environment is correctly configured.

**Prerequisites:**
- Python 3.8+ with `numpy` installed.
- (Optional but recommended) `PyQrack` and OpenCL drivers for GPU-accelerated operations in broader ThereminQ pipelines.
- Multi-GPU setups should have `QRACK_OCL_DEFAULT_DEVICE` and related environment variables configured to ensure proper isolation.

**Execution Commands:**
Navigate to the directory and run the benchmark suite:
```bash
cd cloverleaf-fences/lhv-ace-xyz/NN4K/shell
bash benchtest.sh
```

To export the lattice and the optimized gauge configuration to a JSON manifest for further tensor network (ACE) processing:
```bash
python3 shell_dodec_kitaev.py --shells 6 --anneal 4000 -o output_lattice.json
```

<img width="1544" height="666" alt="image" src="https://github.com/user-attachments/assets/5a9f0b9d-01d8-4775-80d9-1804d682dfdc" />

## 5. References
[1] Carleo, G., & Troyer, M. "Solving the quantum many-body problem with artificial neural networks." *Science* 355.6325 (2017): 602-606. [arXiv:1606.02318](https://arxiv.org/abs/1606.02318)

