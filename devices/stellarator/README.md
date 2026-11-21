# W7-X Stellarator Kitaev $\mathbb{Z}_2$ Lattice Gauge Emulation on $40 \times 27$-Qubit Manifolds

## This document has been generated - statements, results and conclusions could be totally hallucinated; emptor caveat

## Abstract
This directory contains, in a single file `stellarator.py`, the mathematical, geometric, free-fermion, and native PyQrack state-vector implementation of a **Wendelstein 7-X (W7-X)-like 5-field-period helias stellarator** mapped onto an exact 3-valent, 3-edge-colourable Kitaev $\mathbb{Z}_2$ lattice gauge model on $N_{\text{sites}} = 1080$ sites ($1620$ bonds across $N_{\text{shell}} = 5$ nested toroidal flux surfaces). Using CP-SAT module partitioning and $D_5$ stellarator symmetry, the $1080$-site machine decomposes cleanly into **$40$ connected $27$-qubit patch manifolds** ($10$ half-period modules $\times$ $4$ manifolds per module arranged as an all-to-all $K_4$ intra-module cluster), reducing to **$4$ canonical block equivalence classes** under the $D_5$ group. Each **$27$-qubit manifold** is simulated without any ancilla qubit ($27$ qubits per block via in-place Clifford parity folding, executing all `45/45` local `Sigma_keep` seam-repair terms) by the PyQrack driver and worker (Parts G and I of `stellarator.py`), validated against an exact $1080 \times 1080$ Majorana free-fermion anchor and a sparse block-pair $40 \times 27$ propagator (Part H) across validation rungs **F1 to F5**, and benchmarked against all **7 W7-X physics optimization criteria** (Part J) driven purely by 3D Biot-Savart coil filaments.

---

## 1. Architecture & Sizing Mathematics

### 1.1 W7-X Helias Symmetry and 5 Nested Flux Shells
The physical Wendelstein 7-X stellarator is a $5$-field-period helias with discrete symmetry group:
$$G_{\text{W7-X}} = \mathbb{Z}_5 \rtimes \mathbb{Z}_2 \cong D_5 \quad (|D_5| = 10)$$
consisting of $5$ field periods ($p = 0,\dots,4$) split into $10$ stellarator-symmetric half-period modules ($m = 2p + h$, $h \in \{0,1\}$), where module $m=0$ begins at the bean-shaped cross-section ($\varphi = 0$) and transitions to the triangular cross-section at the half-period boundary ($\varphi = \pi/5$).

To maintain exact degree-$3$ coordination and proper $(X, Y, Z)$ $3$-edge-colouring across $N_{\text{shell}} = 5$ nested flux surfaces ($s \in \{0, 1, 2, 3, 4\}$, where $s=0$ is the inner core shell surrounding the helical magnetic axis and $s=4$ is the Last Closed Flux Surface / LCFS with the $10$-unit island divertor):
- Each half-period module on shell $s$ is built from a base $8$-vertex cubic cell ($12$ base edges) by subdividing a fixed $4$-element vertex subset $F \subset \{0,\dots,7\}$ (branch sites, degree $3$ in-shell) and inserting $12$ degree-$2$ connector sites on the subdivided edges ($20$ regular shell sites per module per shell, or $200$ regular shell sites per shell).
- Each connector site receives its third bond from a radial rung connecting adjacent shells $s \leftrightarrow s+1$, with $20$ magnetic-axis core cap sites closing the inner boundary on shell $s=0$ ($220$ sites total on $s=0$) and $60$ island-divertor scrape-off-layer (SOL) target sites closing the outer boundary on shell $s=4$ ($260$ sites total on $s=4$, $6$ poloidal targets per module).
- **Candidate H (Default, Subdivided Hexagon Torus)**: Base cell has four $6$-cycles, lifting to four $12$-cycles per module ($200$ in-shell plaquettes + $1200$ radial $8$-cycles + $50$ axis/divertor cap $4$-cycles).
- **Candidate P (Subdivided $5/7$ Nanotorus)**: Azulene $5/7$-ring base cell lifting to two $10$-cycles and two $14$-cycles per module ($200$ in-shell plaquettes + $1200$ radial $8$-cycles + $20$ cap $4$-cycles). Both Candidate H and Candidate P are bipartite, have **full-graph girth $4$** (from the axis/divertor boundary cap $4$-cycles) and **bulk shell+radial girth $8$**, and admit exact $\text{GF}(2)$ uniform-plaquette and true minimum-energy ground-state $\mathbb{Z}_2$ gauge realisations.

| Quantity | Formula | Value ($N_{\text{shell}} = 5$) |
| :--- | :--- | :--- |
| Field periods / Half-period modules | $N_{\text{fp}} = 5, \; N_{\text{mod}} = 2 N_{\text{fp}}$ | $5$ periods / $10$ modules |
| Regular shell sites per module | $8\text{ (branch)} + 12\text{ (connector)}$ | $20$ regular sites (+ $2$ axis cap on $s=0$, + $6$ SOL cap on $s=4$) |
| Per-shell site breakdown ($s = 0..4$) | $220 + 200 + 200 + 200 + 260$ | **$1080$ sites ($108$ / module)** |
| Bonds per shell / Radial bonds / Total | $1140\text{ in-shell} + 480\text{ radial}$ | **$1620$ bonds ($3 \times 540$)** |
| Manifolds per module / Total manifolds | $4\text{ blocks/module} \times 10\text{ modules}$ | **$40 \times 27$-qubit manifolds ($1100$ internal + $320$ $K_4$ + $200$ toroidal seam bonds)** |
| Independent internal cycles | $(2 + 1 + 2 + 1)\text{ cycles/module} \times 10$ | **$60$ internal cycles (matches exact cycle rank $1100 - 40 \times 26 = 60$)** |
| Unique manifold equivalence classes | $40 / |D_5| = 40 / 10$ | **$4$ canonical block types (`type0..3`)** |
| Qubits per PyQrack worker instance | $27\text{ data qubits (ancilla-free parity fold)}$ | **$27$ qubits ($2^{27} \times 8\text{ B} = 1\text{ GiB}$ dense `complex64` SV)** |

> **Memory & V340 Multi-Die Arithmetic**: A dense $27$-qubit `complex64` (`fp32` real + `fp32` imag = $8$ bytes per amplitude) state vector occupies $2^{27} \times 8\text{ B} = 1\text{ GiB}$ (whereas a $28$-qubit dense state vector requires $2^{28} \times 8\text{ B} = 2\text{ GiB}$ in `complex64` or $4\text{ GiB}$ in `complex128`). Eliminating the $28$th Hadamard-test ancilla via in-place Clifford parity folding halves the dense state-vector footprint to $1\text{ GiB}$ per block. Across the $6$ AMD Radeon Pro V340 GPU dies ($8\text{ GiB}$ VRAM per die, $48\text{ GiB}$ aggregate VRAM), the $40$ manifold blocks distribute evenly at $6$ to $7$ blocks per die ($6$ to $7\text{ GiB}$ dense SV per die, fitting directly in VRAM with Qrack's stabilizer-hybrid / SDRP / host-device paging available for CPU or constrained-VRAM runs).

---

## 2. Exact Free-Fermion Anchor & Gaussian Seam Repair

In Kitaev's Majorana representation $\sigma_j^\alpha = i b_j^\alpha c_j$ with link operators $\hat{u}_{\langle jk\rangle_\alpha} = i b_j^\alpha b_k^\alpha = \pm 1$:
$$H(\{u\}) = \frac{i}{4} \sum_{j,k=1}^{1080} A_{jk} c_j c_k, \qquad A_{jk} = 2 J_{jk} u_{jk}$$

- **Hyperoctagon $L=6$ Benchmark Anchor**: Verified at $E_0/\text{site} = -0.7691433388$ (periodic) and $-0.7702703039$ (antiperiodic).
- **Candidate H Ground Energy ($N=1080$)**:
  - Uniform $u=+1$: $E_0/\text{site} = -0.7607314268$
  - $\text{GF}(2)$ uniform-plaquette gauge across all cycle lengths (`{4: +1, 6: -1, 8: -1, 12: -1}`): $E_0/\text{site} = -0.7626032522$
  - True minimum-energy ground flux sector (`u_ground`, where **$0 / 1620$ single-bond flips lower the energy**): **$E_0/\text{site} = -0.7731725623$**
- **Candidate P Ground Energy ($N=1080$)**:
  - Uniform $u=+1$: $E_0/\text{site} = -0.7607124383$
  - $\text{GF}(2)$ uniform-plaquette gauge across all cycle lengths (`{4: -1, 8: +1, 10: +1, 14: -1}`): $E_0/\text{site} = -0.7637626512$
  - True minimum-energy ground flux sector (`u_ground`, where **$0 / 1620$ single-bond flips lower the energy**): **$E_0/\text{site} = -0.7714457327$**
- **Edge-Localized Majorana Zero-Mode Pair (Parity-Fixed Pure Ground State)**: On the open $1080$-site machine (Candidate H, `u_ground`), the Majorana Hamiltonian $i A$ exhibits an exact pair of zero-energy modes ($|\varepsilon| = 5.1 \times 10^{-16}, 1.3 \times 10^{-15}$) separated from the bulk spectrum by a gap of **$0.211$**. The zero-mode subspace sits **$95.4\%$** on the two outermost flux surfaces (`s4 = 77.8%`, `s3 = 17.6%`, `s2 = 3.1%`, `s1 = 1.2%`, `s0 = 0.3%`). `MajoranaFlowPropagator` explicitly resolves this degenerate pair into a deterministic pure parity eigenstate (`G0^2 = -I`, `[A, G0] = 0`, `S_vN(G0) = 1.5e-11` nats rather than the $\ln 2$ mixture), and Rung F5 reports and gates the mode localization and purity directly.
- **Local Bipartite Gaussian Entanglement Seam Repair (`45/45` Executed on PyQrack)**: Restricting the $1080 \times 1080$ ground-state correlation matrix $G_{jk} = \frac{1}{2i}\langle [c_j, c_k]\rangle$ to each $27$-site manifold $B$ defines the exact entanglement Hamiltonian $A_{\text{ent}} = -2i\,\text{arctanh}(i G_B)$ and seam self-energy $\Sigma = A_{\text{ent}} - A_{\text{int}}$. Retaining the top `keep = 45` local bipartite off-diagonal pairs (`path_len <= 3`: $20$ to $22$ two-qubit internal bonds + $23$ to $25$ four-qubit Kitaev Pauli strings, all algebraically commuting with all internal cycles and executed at exact $0.5 \cdot \Sigma_{ab} \cdot \text{string\_sign}$ spin units by the PyQrack driver, the worker and the OpenQASM emitter) reduces the open-manifold bond correlation error from **$33.90\%$** down to **$2.463\% \le 2.50\%$** (`2.536%`, `1.943%`, `3.264%`, `2.110%` across `type0..3`).

---

## 3. W7-X 7-Criteria Optimization Scoreboard & Validation Rungs

## 3. Validation Rungs, W7-X Criteria & Presets

All figures below come from `python3 stellarator.py bench` (CPU PyQrack build, 16 min 46 s end to end) and are stored in `scoreboard.json`.

### 3.1 Validation Rungs F1 to F4

| Rung | System | Primary metric | Bare (uncoupled) comparison | Conservation / flux check | Gate |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **F1** | Single manifold: `16q` ACE-free 10 steps + full `27q` 6 steps, PyQrack vs Majorana (`ace_max_qb = 23`) | Givens `1.20e-03`; `16q` max RMS `4.31e-05`; `27q` mean RMS `0.0009` (max `0.0013`) | Bell-dimer + missing-colour Gaussian state | `<W_0> = +1.000000` | **PASS** |
| **F2** | `108`-qubit module 0 (`4 x 27` sparse block-pair $K_4$ seam Givens) | Kick-change error `0.35%`, RMS `0.0014` | Kick error `72.9%`, RMS `0.1546` | Strang `2.71e-03` | **PASS** |
| **F3** | `200`-qubit 10-module shell ring (`10 x 20`, 2-hop sparse: `20/45` pairs stored) | Module-1 transit peak step `7` vs `7`, kick error `3.5%` | Kick error `88.9%` | Exact toroidal seam arrival | **PASS** |
| **F4** | `1080`-qubit machine (`40 x 27`, 2-hop sparse: `352/780` pairs stored) | `blk40_dev = 0.0015`, kick error `1.4%`, RMS `0.0029` | Kick error `109.2%`, RMS `0.1728` | `E_drift = 2.0e-13` | **PASS** |

### 3.2 Rung F5: Baths, Zero Modes and Coupled PyQrack Blocks

| Component | Result | Comparison | Gate |
| :--- | :--- | :--- | :---: |
| Edge Majorana zero modes | `n = 2`, bulk gap `0.211`, `95.4%` on `s3 + s4` (`77.8%` on `s4`) | Parity-fixed pure `G0`, `S_vN = 1.5e-11` | **PASS** |
| Zero-T 60-site Lindblad CPTP divertor bath | Bravyi $\lambda_{\min} = -2.4 \times 10^{-15}$, `diag(M_R)` `98.9%` on `s4` | Source-free `dE = 7.8e-15`, ledger `2.9e-14` | **PASS** |
| Dual CPTP core heating bath | Bravyi $\lambda_{\min} = -1.5 \times 10^{-15}$ | 1-step core tail `-1.1e-06` | **PASS** |
| Heisenberg perturbation $g$ | `eps_W(0.04) = 2.25e-09`, `2g/g = 15.97x` ($O(g^4)$) | `eps_W(0) = 0` | **PASS** |
| Seam-loop commutation | `40` seam loops x `318` ops, `0` vison violations | | **PASS** |
| Joint `16q` $K_4$ seam pairs (all 6, 8 steps) | Max RMS `2.95e-05`, cross-seam signal error `0.186%` | Uncoupled `7.89e-02` (`3527x`), `100%` | **PASS** |
| `4 x 27q` all-seam LO-QPD (32/32 seams, 142 bonds, 1 Strang step, deterministic branches) | RMS `5.44e-04`, remote blocks `3.57%, 0.001%, 0.004%` | Uncoupled `3.26e-02` (`60.0x`) | **PASS** |
| `2 x 27q` shot LO-QPD (2 Strang steps, 3 cuts, `25,000` seeded shots) | `+0.4992 ± 0.0197` vs exact `+0.4631` (`z = 1.83`) | Uncoupled `+0.5801` (`z_unc = 4.1`), $\Gamma_{\text{LO}} = 3.16$ | **PASS** |

### 3.3 Cat-Entangler (1-Ebit Gate Teleportation) Seams

Each cross-seam rotation $e^{-i\theta P_a P_b}$ runs as a Bell pair, two mid-circuit measurements and Pauli feed-forward. Both halves share one classical state vector, so the identity runs check the circuit algebra, not a physical link. Running the same circuit on an exact `complex128` state vector (`ExactStateVector`) and on PyQrack, with the seams applied directly or teleported, separates the circuit identity from simulator numerics.

| Simulator | Qubits (data + comm) | Seams | Seams applied | RMS vs Majorana reference | Uncoupled RMS |
| :--- | :---: | :---: | :--- | :---: | :---: |
| `ExactStateVector` (`complex128`) | `20 + 2` | `2` | direct | `9.2e-14` | `9.7e-02` |
| `ExactStateVector` (`complex128`) | `20 + 2` | `2` | teleported | `2.2e-14` | `9.7e-02` |
| PyQrack | `20 + 2` | `2` | direct | `2.2e-05` | `9.7e-02` |
| PyQrack | `20 + 2` | `2` | teleported | `2.7e-04` | `9.7e-02` |
| PyQrack | `26 + 2` | `2` | teleported | `3.3e-02` | `1.0e-01` |

The teleported circuit is exact. The PyQrack error grows with width and with mid-circuit measurements and varies with the measurement branch taken. A one-off probe of the direct `28q` circuit gave `7.7e-03`, so the PyQrack runs measure simulator precision, not the protocol.

**Repeated end-to-end runs** (`10q`: `2 x 4` data + `2` comm, 1 seam, 6 steps, `1,000` runs each, fresh Bell pairs and real mid-circuit measurements, `seed = 2026`):

| Ebit fidelity $F$ | Shots mean | Exact value (all `4^6 = 4096` Pauli error patterns) | Exact bias vs ideal | Bias / (cut x (1 - F)) |
| :---: | :---: | :---: | :---: | :---: |
| `1.00` | `-0.1080 ± 0.0315` (`z = 0.54`, `z_unc = 11.8`) | `-0.0912` | `0` | |
| `0.99` | | `-0.0995` | `0.0083` | `0.139` |
| `0.95` | | `-0.1311` | `0.0400` | `0.133` |
| `0.90` | | `-0.1670` | `0.0759` | `0.126` |
| `0.80` | `-0.1840 ± 0.0311` (`z = 1.41` vs exact) | `-0.2277` | `0.1365` | `0.114` |

The bias grows close to linearly in $N_{\text{cuts}} (1 - F)$, with a coefficient of about `0.13` for this observable. The resource budget below assumes a coefficient of `1`, so it is an upper bound.

### 3.4 LO-QPD vs Distilled-Ebit Link Resources

Unentangled LO-QPD pays a $\Gamma_{\text{LO}}^2$ shot overhead, $\Gamma_{\text{LO}} = \prod_c (|\cos 2\theta_c| + 3|\sin 2\theta_c|)$. Ebit links keep $N_{\text{shots}} \approx 10^4$ at $\varepsilon = 10^{-2}$, provided each of the $N_{\text{ebits}}$ consumed ebits stays within $\epsilon_e^{\text{req}} = 0.01 / N_{\text{ebits}}$. The budget is split evenly between the DEJMPS distillation residual (ideal local gates) and storage decoherence while an ebit waits $r + 1$ classical rounds ($r$ DEJMPS parity exchanges plus the feed-forward). A depolarising pair memory then needs $T_{\text{mem}} / t_{\text{round}} \ge 3 (r + 1) / (2 \epsilon_e^{\text{req}})$. Below $\epsilon_e^{\text{req}} = 10^{-3}$ (`FT`), physical CNOT errors exceed the target, so fault-tolerant logical Bell pairs are needed and the raw-ebit counts are lower bounds.

| Regime | Ebits / shot | $\log_{10} N_{\text{shots}}$ LO-QPD | $\epsilon_e^{\text{req}}$ | DEJMPS $F_0 = 0.99$: rounds, raw / ebit, raw / $10^4$ shots | $T_{\text{mem}} / t_{\text{round}}$ ($F_0 = 0.99$) | DEJMPS $F_0 = 0.95$: rounds, raw / ebit, raw / $10^4$ shots | Feed-forward |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `2 x 27q`, 1 bridge, 2 Strang steps | `3` | `5.00` | `3.33e-03` | `2`, `4.1`, `1.23e+05` | `1.4e+03` | `3`, `9.2`, `2.76e+05` | `2` rounds, `2` bits / step |
| `22q / 28q` identity, 2 seams, 6 steps | `12` | `8.07` | `8.33e-04` (`FT`) | `2`, `4.1`, `4.93e+05` | `5.4e+03` | `3`, `9.2`, `1.10e+06` | `6` rounds, `4` bits / step |
| `4 x 27q`, 32 seams, 1 Strang step | `64` | `17.76` | `1.56e-04` (`FT`) | `3`, `8.2`, `5.26e+06` | `3.8e+04` | `4`, `18.4`, `1.18e+07` | `2` rounds, `64` bits / step |
| `4 x 27q`, 32 seams, 10 steps | `320` | `54.88` | `3.13e-05` (`FT`) | `3`, `8.2`, `2.63e+07` | `1.9e+05` | `4`, `18.4`, `5.89e+07` | `10` rounds, `64` bits / step |
| `40 x 27q` full `1080q`, 520 seams, 10 steps | `5200` | `830.72` | `1.92e-06` (`FT`) | `3`, `8.2`, `4.27e+08` | `3.1e+06` | `5`, `36.8`, `1.92e+09` | `10` rounds, `1040` bits / step |

At a `1 µs` classical round, the full machine needs a pair memory coherence time of about `3.1 s`.

### 3.5 W7-X 7 Optimization Criteria (`flow --optimise`)

Bounded Powell search (`12` iterations, `1531` evaluations, converged) over the 7 global coil-current knobs $(I_1..I_5, I_A, I_B)$ from the uniform baseline `(1, 1, 1, 1, 1, 0, 0)`, mapped through 3D Biot-Savart coil filaments at separately matched means $\langle J_{\text{rad}} \rangle = \langle J_{\text{in-shell}} \rangle = 1.000000$. Optimum (strict interior): `I1 = 1.1977, I2 = 0.8614, I3 = 1.1449, I4 = 0.8294, I5 = 0.9091, IA = -0.0397, IB = -0.0807`.

| # | Criterion | Optimised | Uniform baseline | Gate |
| :---: | :--- | :--- | :--- | :---: |
| 1 | Vacuum flux-surface integrity ($\varepsilon_W$, islands, trim residual) | `0`, `5` islands, `0` | `0`, `5` islands, `0` | **PASS** |
| 2 | Shafranov shift $\Delta\langle\rho\rangle$ at $\beta = 5\%$ | `0.03338` (`-21.6%`) | `0.04260` | **PASS** |
| 3 | Lattice MHD limit: bisected load $\lambda_{\text{crit}}$, mapped to $\beta$ via $\alpha_{\beta = 5\%} = 4.5$ | $\lambda_{\text{crit}} = 5.331$ (`1.87x`), $\beta_{\text{crit}} = 5.92\%$, gaps `d5 = +0.00014`, `d0 = +0.00011`, `0` unstable | $\lambda_{\text{crit}} = 2.843$, $\beta_{\text{crit}} = 3.16\%$, `d5 = -0.00332`, `1` unstable | **PASS** |
| 4 | Neoclassical radial leakage per step (shape only) | `1.1333e-02` (`-6.6%`), outboard `3.179e-02` (`-12.7%`), $\varepsilon_B = 0.0520$ | `1.2129e-02`, outboard `3.641e-02`, $\varepsilon_B = 0.0757$ | **PASS** |
| 5 | Fast-ion survival at $\beta = 2.5\%$ | `0.5325` (`+2.6%`), trapping `0.9022` | `0.5190`, trapping `0.7386` | **PASS** |
| 6 | Geodesic / bootstrap asymmetry | `0.0745` (`-14.9%`), geodesic `0.0626` | `0.0876`, geodesic `0.0903` | **PASS** |
| 7 | Coil knob economy | `7` global knobs, `0` bond tweaks | `0` tweaks | **PASS** |

The `1.87x` ratio of $\lambda_{\text{crit}}$ does not depend on $\alpha$. Crossing `5.0%` in $\beta$ does, and the smallest single-flip gaps are `O(1e-4)` (verified to `1.1e-13` by direct `eigvalsh`).

### 3.6 Core-to-Divertor Lindblad NESS Presets

135 steps (`dt = 1.20`, `gamma_heat = 0.14`, `gamma_div = 0.95`), matched means $\langle J_{\text{rad}} \rangle = \langle J_{\text{in-shell}} \rangle = 1.000$ on every clean preset, core ECRH heating on `20` axis sites, zero-T 60-site divertor bath on `s4`, and no direct `s4` kicks. The error field is an external Biot-Savart busbar dipole pair (`amp = 15.0`). The trim is a per-preset vacuum least-squares fit (`reg = 0.5`) over 5 trim + 10 control coils, combining the `s3 + s4` vector field with a parallel bond-field term weighted by $\lambda_\parallel = 1.5$. With this weight the fit reduces the edge vector field by `73.5%` (`71.9%` on `s4`) and the parallel field by `67.6%` to `67.9%`.

| Preset | Islands | $\tau_E$ | Steady `ex/in` | Driven $S_{\text{vN}}$ | Core fraction | 5-period P/M (clean, error, trim) | 10-unit P/M (clean, error, trim) | 10-unit $L_2$ recovery | Ledger |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `standard` | `5` | `120.46` | `0.887` | `131.7` | `0.749` | `1.128, 1.175, 1.123` | `1.217, 1.315, 1.195` | `77.2%` | `5.3e-12` |
| `high-iota` | `4` | `117.00` | `0.888` | `135.8` | `0.728` | `1.139, 1.180, 1.149` | `1.205, 1.234, 1.183` | `76.4%` | `3.9e-12` |
| `low-iota` | `6` | `124.99` | `0.877` | `131.1` | `0.758` | `1.102, 1.175, 1.137` | `1.181, 1.290, 1.209` | `65.2%` | `5.7e-12` |
| `high-mirror` | `5` | `126.57` | `0.866` | `138.0` | `0.713` | `1.115, 1.246, 1.113` | `1.251, 1.405, 1.256` | `82.2%` | `1.6e-12` |

The preset gate checks only conservation, matched means, convergence and a uniform `> 50%` field and profile recovery threshold; it does not check $\tau_E$ ordering or spread.

**$\lambda_\parallel$ trade-off sweep** (edge RMS reduction / parallel-field reduction / 10-unit $L_2$ recovery / trimmed 5-period P/M):

| Preset | $\lambda_\parallel = 0$ | $0.5$ | $1.5$ (used) | $4.0$ |
| :--- | :---: | :---: | :---: | :---: |
| `standard` (clean P/M `1.128`) | `74.6% / 63.2% / 69.2% / 1.100` | `74.4% / 65.0% / 72.0% / 1.110` | `73.5% / 67.8% / 77.2% / 1.123` | `70.0% / 72.5% / 80.7% / 1.136` |
| `high-iota` (clean `1.139`) | `74.6% / 63.2% / 69.5% / 1.139` | `74.4% / 64.9% / 73.3% / 1.142` | `73.5% / 67.7% / 76.4% / 1.149` | `70.0% / 72.4% / 76.9% / 1.164` |
| `low-iota` (clean `1.102`) | `74.6% / 63.3% / 56.8% / 1.163` | `74.4% / 65.0% / 60.6% / 1.154` | `73.5% / 67.9% / 65.2% / 1.137` | `69.9% / 72.5% / 69.9% / 1.126` |
| `high-mirror` (clean `1.115`) | `74.6% / 63.1% / 72.1% / 1.103` | `74.4% / 64.8% / 76.2% / 1.109` | `73.5% / 67.6% / 82.2% / 1.113` | `70.0% / 72.3% / 88.6% / 1.106` |

Raising $\lambda_\parallel$ trades edge-field reduction for exhaust-profile recovery on every preset. At $\lambda_\parallel = 1.5$ the edge field gives up about 1 point against an average `+8` points of $L_2$ recovery, while the 5-period P/M stays within `0.035` of clean on every preset.

**Island ablation** (fixed `standard` weights, only the edge island chain varies):

| Islands | $\tau_E$ | Change vs 0 islands | 10-unit exhaust std | 10-unit $L_2$ shift vs 0 islands |
| :---: | :---: | :---: | :---: | :---: |
| `0` | `123.80` | | `0.1477` | |
| `4` | `121.59` | `-1.8%` | `0.1265` | `0.280` |
| `5` | `120.46` | `-2.7%` | `0.1154` | `0.276` |
| `6` | `122.79` | `-0.8%` | `0.1248` | `0.314` |

**Trotter and steady-state audit**:

| Check | Result |
| :--- | :--- |
| `dt = 1.20 -> 0.60` at matched horizon `T = 108` | $\tau_E$ `119.02 -> 119.48` (`0.39%`), stored energy `1.59%`, profile `L2 = 0.0080` |
| Continuous-time Lyapunov NESS ($T \to \infty$), residual `1.2e-12` | $\tau_E^{\text{ss}} = 127.49$; the `5.5%` gap to `120.46` at 135 steps is finite pulse horizon |

---

## 4. Code Layout & Quick Start

The physics lives in `stellarator.py` and the native dashboard in `stellarator_ui.py` (one file, which imports the physics from `stellarator.py` in the same folder); data files (`equiv.json`, `manifest.json`, `qasm/`, `scoreboard.json`, `flow_telemetry.npz`, `stellarator_dashboard.png`) are generated by it. The hyperoctagon anchor still loads `../coils/K4-Chrystalstacks-Kitaev-single.py` from the cloverleaf-fences zoo.

| Part | Subcommand | Content |
| :---: | :--- | :--- |
| A | `geometry` | Parametric 3D W7-X surface, `50` non-planar + `20` planar + `5` trim + `10` control coils, $D_5$ checks, Biot-Savart Poincare tracer |
| B | `flux` | Loop stabilisers, toroidal/poloidal Wilson lines, `FluxMonitor`, island-chain detector, $n = 1$ trim cancellation |
| C | `kitaev` | `1080`-site lattice builder (Candidates H and P), exact Majorana solver, GF(2) gauge solver, ground flux finder, hyperoctagon anchor, seam-repair primitives |
| D | `equiv` | CP-SAT `40 x 27`-qubit partitioner and $D_5$ quotient (`equiv.json`) |
| E | `manifest` | Manifest compiler (`manifest.json`): cycles, `45` `sigma_repair_bonds`, $K_4$ and toroidal seams, divertor bonds |
| F | `qasm` | OpenQASM 3.0 emitter (`qasm/type{0..3}-{trotter,flux,full}.qasm`) |
| G | `driver` | PyQrack `27`-qubit driver: 5-gate bond decomposition, Pauli-string seam repair, Gaussian state preparation, `<W_p>` readout |
| H | `reference` | Exact `1080 x 1080` Majorana propagator and 2-hop sparse `40 x 27` block propagator (rungs F1 to F4) |
| I | `worker` | Persistent `27`-qubit manifold worker and seam-loop commutation check |
| J | `flow` | Flow orchestrator: baths, zero modes, rung F5 (QPD, Cat-entangler, `ExactStateVector`, resource table), 7 criteria, presets, $\lambda_\parallel$ sweep, ablation, Trotter audit, optimiser |
| K | `dash` | 6-panel dashboard (`stellarator_dashboard.png`) |
| L | `bench` | All Phase 0 to 8 gates in order |
| M | `ui` | Launcher for the native live dashboard in `stellarator_ui.py` (Section 5) |

```bash
python3 stellarator.py bench                     # full verification, about 17 min on CPU
python3 stellarator.py flow --manifest manifest.json --check --optimise \
    --telemetry flow_telemetry.npz --scoreboard scoreboard.json
python3 stellarator_ui.py                        # native live dashboard window (needs PySide6)
python3 stellarator.py ui                        # same, launched from the main file
python3 stellarator.py <subcommand> --help       # options of each part
```

Requirements: `numpy`, `scipy`, `pyqrack`, `ortools` (for `equiv`), `matplotlib` (for `dash` and `ui`), `PySide6` (for `ui`). The Qrack library path is set by `QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"` at the top of the file.

---

## 5. Live Dashboard (`stellarator_ui.py`)

`python3 stellarator_ui.py` (or `python3 stellarator.py ui`) opens a native desktop window (PySide6 / Qt widgets, matplotlib charts and a QPainter 3D view; `pip install PySide6`). Pick a job, set its parameters and modes, tick **sweep** on one parameter (the coil landscape also takes a second, **Y** axis), set its range, and press **Run**. The sweep runs in a background thread; the window polls it ten times a second and shows every step while it is computed. **Stop** ends the run at the next step.

![Live dashboard during a gamma_div sweep, with the 3D machine view on top](stellarator_ui.png)

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `--manifest` | `manifest.json` | Manifest used for lattice and blocks |
| `--scoreboard` | `scoreboard.json` | Source of the optimised coil knobs (`coils = optimised` and knob defaults) |
| `--selftest PNG` | off | Headless check: run a short NESS sweep and save a window screenshot (uses `QT_QPA_PLATFORM=offscreen` when no display is present) |
| `--selftest-all` | off | With `--selftest`: also run coil, PyQrack and link jobs, one screenshot each |

### Jobs

| Job | Calculation | Modes | Sweepable parameters | Live per step | Per point | Cost (CPU) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| NESS pulse | Core-to-divertor Lindblad NESS on 1080 sites (`_run_divertor_pulse`) | preset, error field (clean / error / trimmed), islands (preset / none), coils (optimised / baseline / custom) | steps, dt, gamma_heat, gamma_div, error amp, lambda_par, I1 to IB | stored, core, exhaust, injected, cumulative ledger, shell energies, 10 divertor-unit loads | tau_E, steady energy, core fraction, ex/in, S_vN, 10-unit / 5-period / poloidal peak-to-mean, ledger error, trim field reductions | about 0.25 to 0.4 s per step |
| Coil landscape | 3D Biot-Savart vacuum metrics (`physics_metrics`) vs uniform baseline | preset | I1 to I5, IA, IB (X and Y: heatmap) | metric profile along X | magnetic well, ripple, geodesic, trapping, radial asymmetries (relative %), mean improvement score | about 2 ms per point |
| 7 criteria | Full `evaluate_w7x_seven_criteria` | preset, coils | I1 to IB | phase progress (polish, C1 to C6) | Shafranov shift, beta_crit, lambda_crit, flip gap, unstable bonds, ripple, leakage, fast-ion survival, trapping, PS dipole, geodesic | about 60 s per point |
| PyQrack manifold | Gaussian state on one manifest block, 5-gate bond Trotter steps, every bond read out vs exact Majorana | width (16 / 27 qubits) | block, steps, dt, initial kick, ACE max qb, SDRP | RMS and max bond error, PyQrack vs exact bond scatter | final / max / mean RMS, `<W_0>` | 16q about 0.06 s, 27q about 5 s per step |
| Ebit link budget | `dejmps_link_budget` vs LO-QPD shot count | | ebits per shot, F0, delta, storage share, classical round, cut angle, shots | | DEJMPS rounds, raw ebits, T_mem / t_round, T_mem, eps_req, LO-QPD shots, FT flag | instant |

The **mean improvement score** of the coil landscape is the plain mean of the six relative improvements (deeper well, lower ripple, geodesic and radial asymmetries, higher trapping); it is a quick guide, not the Powell objective of Section 3.5. A single-point 7-criteria run at the optimised knobs reproduces Section 3.5 exactly (`beta_crit = 5.923%`, Shafranov shift `0.0334`, fast-ion survival `0.5325`).

### 3D machine view

The top panel draws the whole machine in 3D: the 1080 lattice sites and 1620 Kitaev bonds, all 85 coil filaments (50 non-planar, 20 planar, 5 trim, 10 control) and the 10 divertor units. It is drawn with QPainter, so it needs no OpenGL. Every job streams per-bond data to it while it runs.

| Job | Bonds coloured by (live) | Coils coloured by | Overlays |
| :--- | :--- | :--- | :--- |
| NESS pulse | Excess bond energy each step (energy flowing out from the heated axis to the divertor) | Current vs preset nominal, including trim and control coils in `trimmed` mode | Edge-island flux defects (flipped gauge bonds); divertor units sized by their exhaust share |
| Coil landscape | Coupling J of each bond for the knob setting being evaluated | Knob currents vs nominal | |
| 7 criteria | Coupling J at the evaluated knobs | Knob currents vs nominal | |
| PyQrack manifold | PyQrack vs exact error on each bond of the simulated block, each step | | The block's qubits |
| Ebit link budget | | | The 520 inter-block seam bonds, one teleported cut each per Trotter step |

| Control | Effect |
| :--- | :--- |
| Colour | Live job data, coupling J, flux shell, 27-qubit block type (seams white), half-period module, Kitaev bond colour x/y/z |
| Shells `s0` to `s4` | Show or hide each flux shell |
| Cut start, Toroidal span | Show only a toroidal sector (cutaway) |
| bonds, sites, coils, divertor, overlays | Layer toggles |
| rotate | Slow automatic rotation |
| Focus job region | Zoom to the PyQrack block, an edge-island defect or the hottest divertor unit |
| Mouse | Drag to rotate, wheel to zoom, shift-drag or right-drag to pan, double-click to reset; hover a site for a tooltip with its shell, module, block and value |

Only the latest per-bond frame is kept in the job's event log, so long runs do not accumulate per-bond arrays in memory; static overlays (islands, block qubits, seams, coil currents) are always kept.

### Panels

| Panel | Content |
| :--- | :--- |
| Header | Status, point and step progress bars, elapsed time and ETA |
| 3D machine view | See above |
| Live step trace | Selected step metric for every finished point (coloured by sweep value) and the running point; the 7-criteria job shows its phases, the instant jobs a metric profile |
| Sweep result | Selected summary metric vs the sweep value, a heatmap for X x Y sweeps, or a single number |
| Detail | NESS: shell energies and divertor-unit loads; PyQrack: bond-by-bond scatter; others: the last point's full result |
| Results table | One row per point; **Export CSV** and **Export JSON** save it |
| Log | Job start, per-point summaries, errors |

### Scripting sweeps without the window

The job engine in `stellarator_ui.py` only imports Qt when the window opens, so the same sweeps can be run from Python:

```python
import stellarator_ui as U
m = U.LiveMachine("manifest.json", "scoreboard.json")
r = U.LiveJobRunner(m)
r.start({"job": "ness", "params": {"steps": 135},
         "sweep": {"key": "gamma_div", "min": 0.5, "max": 1.5, "n": 5, "scale": "lin"}})
r.thread.join()
print(r.last_results["points"])
```

| Event (`runner.events`) | Content |
| :--- | :--- |
| `job_start` | Job, parameters, sweep axes, number of points |
| `point_start`, `point_done` | Sweep coordinates; summary metrics and run time |
| `step` | Per-step metrics of the running point |
| `frame` | 3D view data: per-bond values, coil currents, overlays |
| `log`, `job_done`, `job_cancelled`, `job_error` | Messages and the end of the run |
