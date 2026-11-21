# A Kitaev $\mathbb{Z}_2$ Lattice-Gauge Emulation of a W7-X-like Helias Stellarator on $40 \times 27$-Qubit PyQrack Manifolds: Methods, Validation Procedures and Outcomes

*Technical report on `twobombs/thereminq-examples`, `devices/stellarator` (commit `ede6167`)*  
*21 November 2026*

## Abstract

We document the methods, validation procedures and reported outcomes of the `devices/stellarator` code in the ThereminQ examples repository. The code maps a five-field-period, Wendelstein 7-X (W7-X)-like helias stellarator onto a three-valent, three-edge-colourable Kitaev honeycomb-type spin model with $N = 1080$ sites and $1620$ bonds on five nested toroidal flux shells. A CP-SAT partition under the $D_5$ stellarator symmetry splits the machine into $40$ connected $27$-site blocks belonging to $4$ symmetry classes. Each block is simulated as a dense $27$-qubit PyQrack state vector without ancillae, and the full machine is checked against an exact $1080 \times 1080$ Majorana free-fermion propagator and a sparse block-pair propagator. Inter-block seams are handled by Gaussian seam repair, local-operation quasiprobability decomposition (LO-QPD) circuit knitting, and cat-entangler gate teleportation, with a DEJMPS entanglement-distillation resource budget. Coil currents, mapped to bond couplings by 3D Biot-Savart filaments, are tuned by a bounded Powell search against seven W7-X-inspired lattice criteria, and a core-to-divertor Lindblad steady state is studied over four magnetic presets. All five validation rungs (F1 to F5) and all seven criteria pass in the archived `scoreboard.json`. In the coupled $108$-qubit module, seam repair lowers the bond-correlation RMS error from $0.155$ to $1.4 \times 10^{-3}$. The optimised coil set raises the lattice flux-sector stability load by a factor of $1.87$ and reduces the $\beta = 5\%$ Shafranov-shift proxy by $21.6\%$. We list the literature behind each method with verified arXiv identifiers, recompute the derived quantities that can be recomputed from the archived data, and record two discrepancies between the README and the code or data. The stellarator quantities are lattice proxies inspired by stellarator physics. They are not magnetohydrodynamic or kinetic plasma calculations.


## 1. Introduction

Stellarators confine plasma with a three-dimensional magnetic field produced by external, non-planar coils, which makes them steady-state devices but also gives a large coil and field-shaping design space [1]. W7-X, a five-period helias with an island divertor, is the leading optimised example. Its reduced neoclassical transport was demonstrated experimentally in 2021 [2], and its edge island topology depends strongly on the coil currents [3]. Modern coil design works directly with 3D filament curves [4] and with numerical targets such as quasisymmetry [5].

The `devices/stellarator` code is not a plasma code. It uses the W7-X geometry, coil set, island chains and design criteria as the *scaffolding* for a quantum many-body benchmark. The scaffolding is a Kitaev $\mathbb{Z}_2$ lattice gauge model [6] laid out on nested flux surfaces. The model is exactly solvable through Majorana fermions, so a $1080$-site instance has an exact classical reference against which distributed $27$-qubit state-vector simulations can be graded. This continues earlier ThereminQ work on three-dimensional Kitaev lattices such as the hyperoctagon lattice [7, 8, 9].

The repository README carries the header *"This document has been generated - statements, results and conclusions could be totally hallucinated; emptor caveat"*. For that reason every number in this report is taken from the archived `scoreboard.json` produced by `python3 stellarator.py bench`, or is marked as reported only in the README. Section 6 lists the checks we ran.

### 1.1 Repository contents

| File | Size | Role |
| :--- | ---: | :--- |
| `stellarator.py` | 10 625 lines | All physics, Parts A to M, 152 top-level functions and classes |
| `stellarator_ui.py` | 2 465 lines | PySide6 live dashboard and headless job engine |
| `stellarator_colab.ipynb` | 83 kB | Browser dashboard for Google Colab |
| `manifest.json` | 2.2 MB | Compiled lattice, blocks, seams, gauge, coil data |
| `scoreboard.json` | 58 kB | Results of the full `bench` run |
| `stellarator-with-3d-ui.tar.gz` | 1.7 MB | Snapshot incl. `qasm/`, `equiv.json`, telemetry, figures |

## 2. Model construction

### 2.1 Geometry, symmetry and coils (Part A)

The plasma boundary is a low-order, stellarator-symmetric Fourier surface with major radius $R_0 = 5.50$ m, minor radius $a = 0.53$ m and $N_{\text{fp}} = 5$. The cross-section changes from bean-shaped at $\varphi = 0$ to triangular at $\varphi = \pi/5$, and the magnetic axis follows a pentagonal helix. The discrete symmetry group is
$$G = \mathbb{Z}_5 \rtimes \mathbb{Z}_2 \cong D_5, \qquad |D_5| = 10,$$
which splits the torus into $10$ half-period modules $m = 2p + h$, with $p \in \{0,\dots,4\}$ and $h \in \{0,1\}$.

The coil set has $85$ filaments: $50$ non-planar modular coils (five types), $20$ planar coils (two types), $5$ trim coils for $n = 1$ error-field control and $10$ divertor control coils. A Biot-Savart evaluator (with an optional PyOpenCL kernel) and an RK4 field-line tracer build Poincaré sections and rotational-transform profiles $\iota(\rho)$. Four presets set the edge island chain:

| Preset | Edge $\iota$ | Islands |
| :--- | :---: | :---: |
| `standard` | $5/5$ | 5 |
| `high-iota` | $5/4$ | 4 |
| `low-iota` | $5/6$ | 6 |
| `high-mirror` | $5/5$ | 5 |

### 2.2 Lattice (Part C)

Each module and shell starts from an $8$-vertex cubic cell. Four branch vertices are kept, and $12$ connector sites of degree $2$ are inserted on subdivided edges. Each connector gets its third bond from a radial rung to the neighbouring shell. The inner shell ($s = 0$) is closed by $20$ magnetic-axis cap sites, and the outer shell ($s = 4$, the last closed flux surface) by $60$ island-divertor target sites, six per module. Every site then has degree $3$ and the bonds admit a proper $(x, y, z)$ colouring.

| Quantity | Value |
| :--- | :--- |
| Sites per shell $s = 0..4$ | $220 + 200 + 200 + 200 + 260 = 1080$ |
| Bonds | $1620 = 3 \times 540$ ($1140$ in-shell, $480$ radial) |
| Blocks | $40 \times 27$ sites ($10$ modules $\times$ $4$ blocks) |
| Bond partition | $1100$ internal, $320$ $K_4$ intra-module seams, $200$ toroidal seams |
| Internal cycle rank | $1100 - 40 \times 26 = 60$ |
| Block classes under $D_5$ | $40 / 10 = 4$ (`type0` to `type3`) |

Two base cells are implemented. Candidate H (subdivided hexagon torus, default) lifts four $6$-cycles to $12$-cycles. Candidate P (subdivided azulene $5/7$ nanotorus) gives two $10$-cycles and two $14$-cycles. Both are bipartite, with full-graph girth $4$ (from the cap cycles) and bulk girth $8$.

### 2.3 Kitaev Hamiltonian and Majorana solution

The spin Hamiltonian is $H = -\sum_{\langle jk \rangle_\alpha} J_{jk}\, \sigma^\alpha_j \sigma^\alpha_k$ with bond colour $\alpha \in \{x, y, z\}$. Following Kitaev [6], $\sigma^\alpha_j = i b^\alpha_j c_j$ and $\hat u_{jk} = i b^\alpha_j b^\alpha_k = \pm 1$ give, in each gauge sector,
$$H(\{u\}) = \frac{i}{4} \sum_{j,k=1}^{1080} A_{jk}\, c_j c_k, \qquad A_{jk} = 2 J_{jk} u_{jk},$$
a free-fermion problem solved by diagonalising the real antisymmetric matrix $A$. Three gauge sectors are compared: uniform $u = +1$, a GF(2) gauge that fixes a uniform flux per cycle length (motivated by Lieb's flux-phase theorem [10]), and a "ground" sector found by greedy single-bond flips until none lowers the energy. The coupling $J_{jk}$ of each bond is set from the 3D Biot-Savart field of the coils, with the radial and in-shell means matched to $1$.

### 2.4 Partition, manifest and circuit export (Parts D to F)

An OR-Tools CP-SAT model assigns the $108$ sites of module $0$ to $4$ connected blocks of $27$ sites, seeded on four disjoint radial cycles and bounded in graph depth. The $D_5$ images of module $0$ give all $40$ blocks, so only $4$ inequivalent block types remain (`equiv.json`). The manifest compiler lists for every block its internal cycles, $45$ seam-repair terms (Section 3.3), $K_4$ and toroidal seams, and divertor bonds. An emitter writes OpenQASM 3.0 [11] for each block type: one Trotter layer, a flux readout, and the full circuit (`qasm/type{0..3}-{trotter,flux,full}.qasm`).

## 3. Simulation methods

### 3.1 Exact free-fermion reference (Part H)

A bond rotation $\exp(-i\theta\, \sigma^\alpha_j \sigma^\alpha_k)$ acts in the Majorana picture as an exact $2 \times 2$ Givens rotation in the $(c_j, c_k)$ plane. The reference therefore evolves the full $1080 \times 1080$ covariance matrix $G_{jk} = \tfrac{1}{2i}\langle [c_j, c_k] \rangle$ exactly. A second propagator, `BlockDecomposedFlowPropagator`, stores only the $40$ intra-block blocks of $G$ and the cross-block blocks within two hops on the block graph. It applies Strang splitting, $U_{\text{seam}}(\Delta t/2)\, U_{\text{int}}(\Delta t)\, U_{\text{seam}}(\Delta t/2)$, with exact Givens rotations across seam bonds. Product-formula error is discussed in [12].

### 3.2 Gaussian seam repair

Cutting a block out of the machine changes its correlations. For block $B$ the restricted covariance $G_B$ defines an entanglement Hamiltonian [13, 14]
$$A_{\text{ent}} = -2i\, \operatorname{arctanh}(i G_B), \qquad \Sigma = A_{\text{ent}} - A_{\text{int}},$$
where $\Sigma$ is a seam self-energy. The code keeps the $45$ largest off-diagonal entries of $\Sigma$ whose internal Kitaev path length is at most $3$. These become $20$ to $22$ two-qubit bond terms and $23$ to $25$ four-qubit Pauli strings that commute with all internal cycle operators, applied at strength $0.5\, \Sigma_{ab}$ times the string sign.

### 3.3 PyQrack 27-qubit driver and worker (Parts G and I)

Each block is a dense $27$-qubit state vector in Qrack [15]. With `complex64` amplitudes this takes $2^{27} \times 8\ \text{B} = 1$ GiB. Every bond rotation uses a five-gate pattern: a basis change on both qubits, $\text{CX}$, $R_z(2\theta)$, $\text{CX}$, and the inverse basis change. Plaquette operators $W_p$ are read out without an ancilla. A Clifford basis rotation and a CNOT chain fold the parity of $W_p$ onto one qubit, which is then measured with `prob`, and the fold is undone. Removing the $28$th (Hadamard-test) qubit halves the memory per block. On the documented target, $6$ AMD Radeon Pro V340 dies with $8$ GiB each, this places $6$ or $7$ blocks per die. Qrack's automatic circuit elision (ACE) and Schmidt-decomposition rounding (SDRP) can be enabled; the reference run uses `ace_max_qb = 23` for the $27$-qubit rung.

### 3.4 Open-system baths

Divertor cooling is a zero-temperature, quasi-local Lindblad bath on the $60$ divertor sites, and core heating is a dual creation bath on the $20$ axis sites. Both are quadratic in Majoranas, so the state stays Gaussian [16, 17]. The code checks complete positivity through Bravyi's Gaussian condition [18], the smallest eigenvalue of $(I - X^2)(I - i G_0)$, and tracks an energy ledger (stored + exhausted $-$ injected). The infinite-time non-equilibrium steady state (NESS) is solved separately as a continuous Lyapunov equation with SciPy.

### 3.5 Inter-block coupling with quantum resources

**LO-QPD circuit knitting.** A seam rotation $e^{-i\theta P_a P_b}$ is replaced by a quasiprobability mixture of local operations [19, 20, 21]. The code samples six local channels with coefficients $(\cos^2\theta, \sin^2\theta, \pm\tfrac12 \sin 2\theta, \pm\tfrac12 \sin 2\theta)$, so each cut costs
$$\gamma(\theta) = 1 + 2\,|\sin 2\theta|, \qquad \Gamma_{\text{LO}} = \prod_c \gamma(\theta_c),$$
the optimal value for local operations without classical communication [22]. The shot count for error $\varepsilon$ scales as $\Gamma_{\text{LO}}^2 / \varepsilon^2$.

**Cat-entangler teleportation.** With one shared Bell pair, a seam rotation can be performed exactly with two mid-circuit measurements and Pauli feed-forward [23, 24]. The code checks this identity on an exact `complex128` state vector and on PyQrack. It also runs $1000$ end-to-end circuits with Werner-noisy ebits and compares them with an exact enumeration of all $4^6$ Pauli error patterns.

**Distillation budget.** If each of $N_{\text{ebits}}$ ebits may contribute error $\epsilon^{\text{req}}_e = 10^{-2}/N_{\text{ebits}}$, half of this budget goes to the DEJMPS residual [25, 26] and half to storage. A depolarising memory then needs
$$\frac{T_{\text{mem}}}{t_{\text{round}}} \ge \frac{3(r+1)}{2\,\epsilon^{\text{req}}_e},$$
where $r$ is the number of DEJMPS rounds. Below $\epsilon^{\text{req}}_e = 10^{-3}$ the code flags the regime as needing fault-tolerant Bell pairs.

### 3.6 Coil optimisation and the seven criteria (Part J)

Seven global knobs $(I_1, \dots, I_5, I_A, I_B)$ scale the five non-planar and two planar coil types. A bounded Powell search in SciPy starts from $(1,1,1,1,1,0,0)$. It minimises a weighted sum of relative improvements in seven vacuum lattice metrics (magnetic well, helical ripple, geodesic asymmetry, outboard trapping, radial asymmetry, outboard leakage, weight spread), plus a quadratic current-stress term. After the search, the seven criteria below are evaluated for both the optimum and the uniform baseline:

| # | Criterion | Lattice proxy |
| :---: | :--- | :--- |
| 1 | Flux-surface integrity | Interior Wilson-loop error $\varepsilon_W$, island count, trim residual |
| 2 | Shafranov stiffness | Outward shift of the energy centroid $\Delta\langle\rho\rangle$ at a $\beta = 5\%$ source |
| 3 | MHD limit | Critical curvature load $\lambda_{\text{crit}}$ before a single-bond flip lowers $E_0$; $\beta_{\text{crit}}$ via $\alpha = 4.5$ |
| 4 | Neoclassical transport | Radial and outboard energy leakage per step; ripple $\varepsilon_B$ |
| 5 | Fast-ion confinement | Inner-shell survival and outboard trapping at $\beta = 2.5\%$ |
| 6 | Bootstrap current | Geodesic and Pfirsch-Schlüter seam-dipole asymmetry |
| 7 | Knob economy | Global knobs used; arbitrary bond tweaks (must be $0$) |

### 3.7 Core-to-divertor NESS and error-field trim

The NESS pulse runs $135$ steps ($\Delta t = 1.20$, $\gamma_{\text{heat}} = 0.14$, $\gamma_{\text{div}} = 0.95$) with heating on the $20$ axis sites and the divertor bath on shell $4$. An external $n = 1$ error field from a Biot-Savart busbar dipole pair (amplitude $15$) is applied. Per preset, a regularised vacuum least-squares fit over $5$ trim and $10$ control coils (regulariser $0.5$) cancels the $s_3 + s_4$ vector field plus a parallel bond-field term weighted by $\lambda_\parallel = 1.5$.

## 4. Validation procedures

The `bench` subcommand runs the gates of phases 0 to 8 in order. The reported run took 16 min 46 s on a CPU PyQrack build. The rungs grow in system size:

| Rung | System | Reference | Steps, $\Delta t$ |
| :---: | :--- | :--- | :--- |
| F1 | One block: $16$q ACE-free and full $27$q PyQrack | Exact Majorana | 10, 0.04 (27q: 6) |
| F2 | Module 0, $4 \times 27 = 108$ sites, $K_4$ seams | Exact Majorana | 8, 0.06 |
| F3 | Shell ring, $10 \times 20 = 200$ sites, 2-hop sparse ($20/45$ pairs) | Exact Majorana | 16 |
| F4 | Full machine, $40 \times 27 = 1080$ sites, 2-hop sparse ($352/780$ pairs) | Exact Majorana | 10 |
| F5 | Baths, zero modes, perturbation, coupled PyQrack blocks | Exact Gaussian / enumeration | varies |

Every rung compares the repaired or coupled simulation with an uncoupled ("bare") one, so that a pass cannot come from a trivially small signal.

## 5. Outcomes

All values in this section come from `scoreboard.json` unless marked *(README)*.

### 5.1 Rungs F1 to F4

| Rung | Coupled / repaired | Bare (uncoupled) | Conservation |
| :---: | :--- | :--- | :--- |
| F1 | Givens bond err $1.20 \times 10^{-3}$; 16q max RMS $4.31 \times 10^{-5}$; 27q mean RMS $8.7 \times 10^{-4}$, max $1.29 \times 10^{-3}$ | n/a | $\langle W_0 \rangle = 1.000000$ |
| F2 | RMS $1.36 \times 10^{-3}$; kick error $0.35\%$ | RMS $0.155$; kick error $72.9\%$ | Strang RMS $2.71 \times 10^{-3}$ |
| F3 | RMS $3.58 \times 10^{-3}$; kick error $3.5\%$; transit peak at step 7 (exact: 7) | RMS $0.106$; kick error $88.9\%$ | toroidal arrival exact |
| F4 | RMS $3.14 \times 10^{-3}$; kick error $1.4\%$; max shell deviation $1.5 \times 10^{-3}$ | RMS $0.173$; kick error $109.2\%$ | $\Delta E = 2.0 \times 10^{-13}$ |

### 5.2 Ground-state energies *(README; not stored in `scoreboard.json`)*

| Lattice, gauge | $E_0 / N$ |
| :--- | :---: |
| Hyperoctagon $L = 6$ anchor, periodic / antiperiodic | $-0.7691433388$ / $-0.7702703039$ |
| Candidate H, uniform $u = +1$ | $-0.7607314268$ |
| Candidate H, GF(2) uniform plaquette | $-0.7626032522$ |
| Candidate H, ground sector ($0/1620$ lowering flips) | $-0.7731725623$ |
| Candidate P, uniform / GF(2) / ground | $-0.7607124383$ / $-0.7637626512$ / $-0.7714457327$ |

With Biot-Savart couplings, `scoreboard.json` records ground-sector energies of $-0.77938$ (uniform-coil baseline, $104$ flips) and $-0.77587$ (optimised coils, $26$ flips). Both have no lowering single flip. The README reports that seam repair lowers the open-block bond correlation error from $33.90\%$ to a mean of $2.463\%$ (per type: $2.536$, $1.943$, $3.264$, $2.110\%$).

### 5.3 Rung F5

| Component | Result | Comparison |
| :--- | :--- | :--- |
| Edge Majorana zero modes | $2$ modes, $|\varepsilon| \sim 10^{-15}$, bulk gap $0.211$; $95.4\%$ on $s_3 + s_4$ ($77.8\%$ on $s_4$) | Parity-fixed pure state, $S_{\text{vN}} = 1.6 \times 10^{-11}$ |
| Divertor bath CPTP | $\lambda_{\min} = -2.4 \times 10^{-15}$; $98.9\%$ weight on $s_4$ | Source-free $\Delta E = 1.1 \times 10^{-13}$; ledger $1.6 \times 10^{-13}$ |
| Core heating bath CPTP | $\lambda_{\min} = -1.6 \times 10^{-15}$ | One-step core tail $-1.1 \times 10^{-6}$ |
| Heisenberg perturbation $g$ | $\varepsilon_W(0.04) = 2.25 \times 10^{-9}$; $\varepsilon_W(2g)/\varepsilon_W(g) = 15.97$ | Consistent with $O(g^4)$; $\varepsilon_W(0) = 0$ |
| Seam-loop commutation | $40$ loops $\times$ $318$ operators, $0$ violations | n/a |
| Joint 16q $K_4$ seam pairs (6 pairs, 8 steps) | Max RMS $2.95 \times 10^{-5}$; cross-seam error $\le 0.186\%$ | Uncoupled mean RMS $7.89 \times 10^{-2}$ ($3527\times$) |
| $4 \times 27$q LO-QPD (32/32 seams, 142 bonds, 1 Strang step) | RMS vs 108-site reference $5.44 \times 10^{-4}$ | Uncoupled $3.26 \times 10^{-2}$ ($60.0\times$) |
| $2 \times 27$q shot LO-QPD (2 steps, 3 cuts, 25 000 shots) | $0.4992 \pm 0.0197$ vs exact $0.4631$ ($z = 1.83$); empirical/theory SE $= 0.998$ | Uncoupled $0.5801$ ($z = 4.10$); $\Gamma_{\text{LO}} = 3.16$ |

### 5.4 Cat-entangler seams

| Simulator | Qubits | Seams | RMS vs Majorana | Uncoupled RMS |
| :--- | :---: | :--- | :---: | :---: |
| Exact `complex128` | 20 + 2 | direct | $9.2 \times 10^{-14}$ | $9.7 \times 10^{-2}$ |
| Exact `complex128` | 20 + 2 | teleported | $2.2 \times 10^{-14}$ | $9.7 \times 10^{-2}$ |
| PyQrack | 20 + 2 | direct | $2.2 \times 10^{-5}$ | $9.7 \times 10^{-2}$ |
| PyQrack | 20 + 2 | teleported | $2.7 \times 10^{-4}$ | $9.7 \times 10^{-2}$ |
| PyQrack | 26 + 2 | teleported | $3.3 \times 10^{-2}$ | $1.0 \times 10^{-1}$ |

The teleported circuit is exact to machine precision. The PyQrack error grows with width and with the number of mid-circuit measurements, so the 28-qubit figure reflects simulator precision, not the protocol. In $1000$ repeated $10$-qubit runs (6 cuts, seed 2026), ideal ebits give $-0.108 \pm 0.031$ against the exact $-0.0912$ ($z = 0.54$; uncoupled distance $z = 11.8$). Werner ebits with $F = 0.80$ give $-0.184 \pm 0.031$ against the enumerated $-0.2277$ ($z = 1.41$). The exact bias grows almost linearly in $N_{\text{cuts}}(1 - F)$, with a coefficient of about $0.13$ for this observable.

### 5.5 Link resource budget

| Regime | Ebits/shot | $\log_{10} N_{\text{shots}}$ LO-QPD | $\epsilon^{\text{req}}_e$ | DEJMPS $F_0 = 0.99$: rounds, raw/ebit | $T_{\text{mem}}/t_{\text{round}}$ |
| :--- | ---: | ---: | :---: | :---: | ---: |
| $2 \times 27$q, 2 steps | 3 | 5.00 | $3.33 \times 10^{-3}$ | 2, 4.1 | $1.4 \times 10^{3}$ |
| 22q/28q identity, 6 steps | 12 | 8.07 | $8.33 \times 10^{-4}$ (FT) | 2, 4.1 | $5.4 \times 10^{3}$ |
| $4 \times 27$q, 32 seams, 1 step | 64 | 17.76 | $1.56 \times 10^{-4}$ (FT) | 3, 8.2 | $3.8 \times 10^{4}$ |
| $4 \times 27$q, 32 seams, 10 steps | 320 | 54.88 | $3.13 \times 10^{-5}$ (FT) | 3, 8.2 | $1.9 \times 10^{5}$ |
| Full machine, 520 seams, 10 steps | 5200 | 830.72 | $1.92 \times 10^{-6}$ (FT) | 3, 8.2 | $3.1 \times 10^{6}$ |

With entanglement, the shot count stays near $10^4$, whereas unentangled LO-QPD for the full machine would need about $10^{831}$ shots. At a $1\ \mu$s classical round, the full machine needs a pair-memory coherence time of about $3.1$ s.

### 5.6 Coil optimisation

The Powell search converged after $12$ iterations and $1531$ evaluations to a strict interior optimum, $I = (1.1977, 0.8614, 1.1449, 0.8294, 0.9091, -0.0397, -0.0807)$, with objective value $-3.637$.

| # | Criterion | Optimised | Uniform baseline | Change |
| :---: | :--- | :--- | :--- | :---: |
| 1 | $\varepsilon_W$, islands, trim residual | $0$, $5$, $0$ | $0$, $5$, $0$ | equal |
| 2 | Shafranov shift at $\beta = 5\%$ | $0.03338$ | $0.04260$ | $-21.6\%$ |
| 3 | $\lambda_{\text{crit}}$ ($\beta_{\text{crit}}$) | $5.331$ ($5.92\%$), 0 unstable bonds | $2.843$ ($3.16\%$), 1 unstable bond | $\times 1.87$ |
| 4 | Radial / outboard leakage; $\varepsilon_B$ | $0.01133$ / $0.03179$; $0.052$ | $0.01213$ / $0.03641$; $0.076$ | $-6.6\%$ / $-12.7\%$ |
| 5 | Fast-ion survival; trapping | $0.5325$; $0.902$ | $0.5190$; $0.739$ | $+2.6\%$ |
| 6 | Bootstrap proxy; geodesic | $0.0745$; $0.0626$ | $0.0876$; $0.0903$ | $-14.9\%$ |
| 7 | Global knobs; bond tweaks | $7$; $0$ | n/a; $0$ | pass |

The ratio $\lambda_{\text{crit}}^{\text{opt}} / \lambda_{\text{crit}}^{\text{base}} = 1.87$ does not depend on the calibration constant $\alpha = 4.5$, but whether $\beta_{\text{crit}}$ crosses $5\%$ does. The smallest single-flip gaps are $O(10^{-4})$ (optimised: $+1.44 \times 10^{-4}$ at $\beta = 5\%$).

### 5.7 NESS presets

| Preset | Islands | $\tau_E$ | Ex/in | $S_{\text{vN}}$ | Core frac. | 10-unit P/M (clean, err, trim) | $L_2$ recovery | Ledger |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| standard | 5 | 120.46 | 0.887 | 131.7 | 0.749 | 1.217, 1.315, 1.195 | 77.2% | $5.3 \times 10^{-12}$ |
| high-iota | 4 | 117.00 | 0.888 | 135.8 | 0.728 | 1.205, 1.234, 1.183 | 76.4% | $3.9 \times 10^{-12}$ |
| low-iota | 6 | 124.99 | 0.877 | 131.1 | 0.758 | 1.181, 1.290, 1.209 | 65.2% | $5.7 \times 10^{-12}$ |
| high-mirror | 5 | 126.57 | 0.866 | 138.0 | 0.713 | 1.251, 1.405, 1.256 | 82.2% | $1.6 \times 10^{-12}$ |

The trim fit reduces the edge error field by $73.5\%$ ($71.9\%$ on $s_4$) and the parallel bond field by $67.6$ to $67.9\%$ on every preset. Raising $\lambda_\parallel$ from $0$ to $4$ trades edge-field reduction ($74.6 \to 70.0\%$) for exhaust-profile recovery (for example `standard`: $69.2 \to 80.7\%$).

| Islands (standard weights) | $\tau_E$ | $\Delta\tau_E$ vs 0 | Exhaust std | $L_2$ shift vs 0 |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 123.80 | n/a | 0.1477 | n/a |
| 4 | 121.59 | $-1.8\%$ | 0.1265 | 0.280 |
| 5 | 120.46 | $-2.7\%$ | 0.1154 | 0.276 |
| 6 | 122.79 | $-0.8\%$ | 0.1248 | 0.314 |

**Time-step and horizon audit.** Halving $\Delta t$ from $1.20$ to $0.60$ at a fixed horizon $T = 108$ changes $\tau_E$ from $119.02$ to $119.48$ ($0.39\%$), stored energy by $1.59\%$ and the unit profile by $L_2 = 0.0080$. The continuous Lyapunov NESS (residual $1.2 \times 10^{-12}$) gives $\tau_E^{\text{ss}} = 127.49$. The $5.5\%$ gap to $120.46$ reflects the finite $135$-step pulse.

![Six-panel summary dashboard produced by `stellarator.py dash` (from the repository archive): 3D geometry, $\iota(\rho)$ per preset, Shafranov-shift proxy, NESS energy traces, divertor-unit loads with and without $n = 1$ trim, and the criteria and gates table.](stellarator_dashboard.png)

![Live dashboard `stellarator_ui.py` during a $\gamma_{\text{div}}$ sweep of the NESS pulse (from the repository archive). The 3D view colours bonds by excess energy and marks edge-island flux defects and divertor units.](stellarator_ui.png)

## 6. Verification performed for this report

The PyQrack shared library could not be loaded in our environment, so we did **not** re-run the benchmark. We ran the following checks on the archived data instead.

| Check | Outcome |
| :--- | :--- |
| `scoreboard.json` in the directory vs the copy in `stellarator-with-3d-ui.tar.gz` | Byte-identical |
| README tables F1 to F5, criteria, presets, ablation, Trotter audit vs `scoreboard.json` | Agree to the printed precision, except item 2 below |
| $2^{27} \times 8$ B; cycle rank $1100 - 40 \cdot 26$; $3 \cdot 1080 / 2$ bonds | $1$ GiB; $60$; $1620$ |
| Independent DEJMPS recursion for all 10 resource-table cells | Rounds, raw ebits per distilled ebit, raw per $10^4$ shots and $T_{\text{mem}}/t_{\text{round}}$ reproduced |
| $z$-score of the 25 000-shot LO-QPD estimate | $(0.49920 - 0.46309)/0.01974 = 1.83$ |
| Relative changes for criteria 2, 4, 5, 6 and the $\lambda_{\text{crit}}$ ratio | $-21.6$, $-6.6/-12.7$, $+2.6$, $-14.9\%$; $1.875$ |

**Discrepancies found.**

1. *LO-QPD overhead formula.* README section 3.4 gives $\Gamma_{\text{LO}} = \prod_c (|\cos 2\theta_c| + 3|\sin 2\theta_c|)$, but the code (resource table and Monte Carlo sampler) uses $\prod_c (1 + 2|\sin 2\theta_c|)$. The tabulated shot counts follow the code. The README formula should be corrected.
2. *Divertor-bath source-free check.* README rung F5 reports $dE = 7.8 \times 10^{-15}$ and ledger $2.9 \times 10^{-14}$; `scoreboard.json` stores $1.1 \times 10^{-13}$ and $1.6 \times 10^{-13}$. Both are at machine precision and the gate passes either way, but the README values come from a different run.
3. The hyperoctagon anchor, the Candidate H and P energies and the $33.90\% \to 2.463\%$ seam-repair figure are not stored in `scoreboard.json` and could not be checked against data.

## 7. Discussion and limitations

**What is established.** Within the archived run, the distributed scheme reproduces exact free-fermion dynamics of a $1080$-site, three-dimensional Kitaev lattice. Bare cuts give $10$ to $17\%$ RMS bond errors and $73$ to $109\%$ kick errors. With seam repair and exact seam Givens rotations these fall to $0.1$ to $0.4\%$ RMS and $0.35$ to $3.5\%$ kick errors, while storing only $45\%$ of the cross-block pairs. Real PyQrack $27$-qubit blocks follow the exact reference to $\sim 10^{-3}$ RMS, and QPD-knitted blocks to $5.4 \times 10^{-4}$. The teleportation identity holds to $10^{-14}$, and shot statistics agree with theory (SE ratio $0.998$).

**What is a proxy.** The stellarator side changes only the bond couplings $J_{jk}$ (through Biot-Savart coil fields), the boundary closures and the baths. "Shafranov shift", "$\beta_{\text{crit}}$", "neoclassical leakage", "fast-ion survival" and "bootstrap current" are lattice observables named after the plasma quantities. No MHD equilibrium, drift-kinetic or transport equation is solved, and the numbers should not be compared with W7-X design values [1, 2]. The mapping $\beta_{\text{crit}} = \lambda_{\text{crit}}/\alpha$ uses a calibration constant ($\alpha = 4.5$).

**Further caveats.**

- Several Powell objective terms (ripple, geodesic asymmetry, trapping, radial leakage) share inputs with criteria 4 to 6. "Beats all seven criteria" is therefore partly by construction. Criteria 2 and 3, which are evaluated only afterwards, are the more independent tests.
- The criterion-3 margin at the optimum is small: the smallest single-flip gap is $1.4 \times 10^{-4}$.
- The preset gate checks conservation, matched means, convergence and a uniform $> 50\%$ recovery threshold. It does not test the ordering or spread of $\tau_E$.
- All quantum-resource numbers come from classical simulation. Both halves of each teleported seam share one state vector, so no physical link was used. The $1080$-qubit rung F4 is a free-fermion calculation, not a $1080$-qubit state vector.
- The reported run used CPU PyQrack. The V340 multi-die layout is a memory-arithmetic plan, not a measured run.

## 8. Reproducibility

```bash
git clone https://github.com/twobombs/thereminq-examples
cd thereminq-examples/devices/stellarator
python3 stellarator.py bench        # all phase 0 to 8 gates, about 17 min on CPU
python3 stellarator.py flow --manifest manifest.json --check --optimise \
    --telemetry flow_telemetry.npz --scoreboard scoreboard.json
python3 stellarator_ui.py           # live dashboard (PySide6)
```

Requirements: `numpy`, `scipy`, `pyqrack`, `ortools`, `matplotlib`, and `PySide6` for the dashboard. The script sets `QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"`. The hyperoctagon anchor also loads `../coils/K4-Chrystalstacks-Kitaev-single.py`. This report covers commit `ede616730adf2fc21fb1839cc7998cfb8d053593`.

## References

1. Per Helander, "Theory of plasma confinement in non-axisymmetric magnetic fields", *Reports on Progress in Physics* **77**, 087001 (2014). No arXiv version located; [ADS 2014RPPh...77h7001H](https://ui.adsabs.harvard.edu/abs/2014RPPh...77h7001H)
2. C. D. Beidler et al., "Demonstration of reduced neoclassical energy transport in Wendelstein 7-X", *Nature* **596**, 221-226 (2021). No arXiv version; [doi:10.1038/s41586-021-03687-w](https://doi.org/10.1038/s41586-021-03687-w)
3. Robert Davies, Christopher B. Smiet, Charlotte Batzdorf, J. Geiger, J. Loizu, and S. A. Henneberg, "Characterisation of X- and O-points in Wendelstein 7-X with respect to coil currents" (2025). [arXiv:2511.15641](https://arxiv.org/abs/2511.15641)
4. Caoxiang Zhu, Stuart R. Hudson, Yuntao Song, and Yuanxi Wan, "New method to design stellarator coils without the winding surface" (2017). [arXiv:1705.02333](https://arxiv.org/abs/1705.02333)
5. Matt Landreman and Elizabeth Paul, "Magnetic fields with precise quasisymmetry for plasma confinement" (2021). [arXiv:2108.03711](https://arxiv.org/abs/2108.03711)
6. Alexei Kitaev, "Anyons in an exactly solved model and beyond", *Annals of Physics* **321**, 2-111 (2006). [arXiv:cond-mat/0506438](https://arxiv.org/abs/cond-mat/0506438)
7. M. Hermanns and S. Trebst, "Quantum spin liquid with a Majorana Fermi surface on the three-dimensional hyperoctagon lattice" (2014). [arXiv:1401.7678](https://arxiv.org/abs/1401.7678)
8. Kevin O'Brien, Maria Hermanns, and Simon Trebst, "Classification of gapless $Z_2$ spin liquids in three-dimensional Kitaev models", *Physical Review B* **93**, 085101 (2016). [arXiv:1511.05569](https://arxiv.org/abs/1511.05569)
9. Maria Hermanns, Kevin O'Brien, and Simon Trebst, "Weyl spin liquids", *Physical Review Letters* **114**, 157202 (2015). [arXiv:1411.7379](https://arxiv.org/abs/1411.7379)
10. E. H. Lieb, "The flux-phase of the half-filled band" (1994). [arXiv:cond-mat/9410025](https://arxiv.org/abs/cond-mat/9410025)
11. Andrew W. Cross, Ali Javadi-Abhari, Thomas Alexander, Niel de Beaudrap, Lev S. Bishop, Steven Heidel, Colm A. Ryan, Prasahnt Sivarajah, John Smolin, Jay M. Gambetta, and Blake R. Johnson, "OpenQASM 3: A broader and deeper quantum assembly language" (2021). [arXiv:2104.14722](https://arxiv.org/abs/2104.14722)
12. Andrew M. Childs, Yuan Su, Minh C. Tran, Nathan Wiebe, and Shuchen Zhu, "A theory of Trotter error", *Physical Review X* **11**, 011020 (2021). [arXiv:1912.08854](https://arxiv.org/abs/1912.08854)
13. Ingo Peschel, "Calculation of reduced density matrices from correlation functions" (2002). [arXiv:cond-mat/0212631](https://arxiv.org/abs/cond-mat/0212631)
14. Ingo Peschel and Viktor Eisler, "Reduced density matrices and entanglement entropy in free lattice models" (2009). [arXiv:0906.1663](https://arxiv.org/abs/0906.1663)
15. Daniel Strano, Benn Bollay, Aryan Blaauw, Nathan Shammah, William J. Zeng, and Andrea Mari, "Exact and approximate simulation of large quantum circuits on a single GPU" (2023). [arXiv:2304.14969](https://arxiv.org/abs/2304.14969)
16. Tomaž Prosen, "Third quantization: a general method to solve master equations for quadratic open Fermi systems", *New Journal of Physics* **10**, 043026 (2008). [arXiv:0801.1257](https://arxiv.org/abs/0801.1257)
17. Sergey Bravyi and Robert König, "Classical simulation of dissipative fermionic linear optics" (2011). [arXiv:1112.2184](https://arxiv.org/abs/1112.2184)
18. Sergey Bravyi, "Lagrangian representation for fermionic linear optics" (2004). [arXiv:quant-ph/0404180](https://arxiv.org/abs/quant-ph/0404180)
19. Kosuke Mitarai and Keisuke Fujii, "Constructing a virtual two-qubit gate by sampling single-qubit operations" (2019). [arXiv:1909.07534](https://arxiv.org/abs/1909.07534)
20. Kosuke Mitarai and Keisuke Fujii, "Overhead for simulating a non-local channel with local channels by quasiprobability sampling" (2020). [arXiv:2006.11174](https://arxiv.org/abs/2006.11174)
21. Tianyi Peng, Aram W. Harrow, Maris Ozols, and Xiaodi Wu, "Simulating large quantum circuits on a small quantum computer", *Physical Review Letters* **125** (2020). [arXiv:1904.00102](https://arxiv.org/abs/1904.00102)
22. Christophe Piveteau and David Sutter, "Circuit knitting with classical communication" (2022). [arXiv:2205.00016](https://arxiv.org/abs/2205.00016)
23. J. Eisert, K. Jacobs, P. Papadopoulos, and M. B. Plenio, "Optimal local implementation of non-local quantum gates" (2000). [arXiv:quant-ph/0005101](https://arxiv.org/abs/quant-ph/0005101)
24. Anocha Yimsiriwattana and Samuel J. Lomonaco Jr., "Generalized GHZ states and distributed quantum computing" (2004). [arXiv:quant-ph/0402148](https://arxiv.org/abs/quant-ph/0402148)
25. D. Deutsch, A. Ekert, R. Jozsa, C. Macchiavello, S. Popescu, and A. Sanpera, "Quantum privacy amplification and the security of quantum cryptography over noisy channels" (1996). [arXiv:quant-ph/9604039](https://arxiv.org/abs/quant-ph/9604039)
26. Charles H. Bennett, Gilles Brassard, Sandu Popescu, Benjamin Schumacher, John A. Smolin, and William K. Wootters, "Purification of noisy entanglement and faithful teleportation via noisy channels" (1995). [arXiv:quant-ph/9511027](https://arxiv.org/abs/quant-ph/9511027)
