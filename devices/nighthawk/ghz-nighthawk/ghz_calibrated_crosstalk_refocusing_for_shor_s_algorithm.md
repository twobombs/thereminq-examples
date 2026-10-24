# GHZ-Calibrated Crosstalk Refocusing for Shor's Algorithm on a Nighthawk-Class Square Lattice

## Exact Refocusing of Diagonal Crosstalk for Shor Order Finding on a $10 \times 12$ Square-Lattice Processor: Analysis and Simulation

**Date:** Oct 24, 2026  
**Author:** Generated
**Concept:** Aryan Blaauw

---

## Abstract

On a bipartite square-lattice register whose idle drift is diagonal ($ZZ$ couplings plus static and quasi-static $Z$ detunings) and whose incoherent noise is global depolarising with survival $\lambda$ per circuit block, a four-segment Walsh echo in each idle window makes Shor order finding for $N=15$, $a=7$ output exactly $\lambda^B p_{\text{ideal}} + (1-\lambda^B) u$, for every value of the couplings and detunings. With $B=5$ blocks and $\lambda=0.94$, the per-shot factoring success is $0.40021$; no refocusing that restores the ideal unitary can raise it, because the depolarising factor commutes with every unitary layer.

GHZ population measurements identify $\lambda$ exactly, a conditional Ramsey protocol identifies every coupling $J_{ij}$, and a monotone readout map recovers $p_{\text{ideal}}$ without changing per-shot success. Sections 2 to 7 prove these statements. Section 8 reports an 8-qubit density-matrix simulation that reproduces the predicted success to four decimals and measures the effect of exchange coupling, which lies outside the hypotheses. Section 9 runs Layers 1a and 2 on 16- to 61-qubit Nighthawk registers split by Qrack's ACE simulator: off the seams the echo restores $F_{\text{GHZ}} = 1.000$; on them every Layer 2 mode lowers it.

---

## 1. Setting

The stack has three layers: Layer 1 calibrates the drift and the noise ($\S 3$), Layer 2 refocuses each idle window ($\S 4$), and Layer 3 runs the order-finding circuit ($\S 6$). This section fixes the objects the proofs use.

**Definition 1 (lattice and register).** $\Gamma = (V, E)$ is the $10 \times 12$ square grid, vertex $(r, c)$ adjacent to $(r \pm 1, c)$ and $(r, c \pm 1)$. The register $\Pi$ is the $2 \times 4$ sub-grid on rows $3\text{--}4$ and columns $4\text{--}7$, with $n=8$ qubits and coupling set $E_\Pi$. It splits as a counting register $R_C$ (the four qubits of row 3) and a work register $R_W$ (row 4), so $d = 2^8$ and $d_C = 2^4$.

**Definition 2 (drift).** During an idle window of length $T$ the register evolves under $H = H_D + H_X$ with

$$H_D = \sum_{(i,j) \in E_\Pi} J_{ij} Z_i Z_j + \sum_i (\delta\omega_i + \epsilon_i) Z_i, \quad H_X = \eta \sum_{(i,j) \in E_\Pi} J_{ij} (X_i X_j + Y_i Y_j)$$

The $J_{ij}$ and $\delta\omega_i$ are fixed real numbers. The vector $\epsilon$ is quasi-static: constant within one shot and drawn from a distribution $\mu$ between shots.

**Definition 3 (depolarising channel).** $D_p(\rho) = (1-p)\rho + p I/d$, with survival $\lambda = 1-p$.

**Definition 4 (block).** A block applies an exact gate $G$, an idle unitary $V$, then $D_p$:

$$\mathcal{E}_{G,V}(\rho) = D_p(V G \rho G^\dagger V^\dagger)$$

A circuit is $B$ blocks in sequence; its output state is averaged over $\mu$.

**Definition 5 (Walsh sequence).** Let $A$ and $B$ be the sublattices of Lemma 1, let $X_A$ and $X_B$ be the products of $X$ over $A$ and over $B$, and let $\tau = T/4$. The refocused idle unitary is

$$W_T(H) = X_A \cdot \exp(-i\tau H) \cdot X_B \cdot \exp(-i\tau H) \cdot X_A \cdot \exp(-i\tau H) \cdot X_B \cdot \exp(-i\tau H)$$

**Definition 6 (compensation).** Given a diagonal estimate $\hat{H} = \sum \hat{J}_{ij} Z_i Z_j + \sum \delta\hat{\omega}_i Z_i$, the compensated idle unitary is $\exp(+iT\hat{H})\exp(-iTH)$.

**Definition 7 (per-shot success).** $S \subset \{0, \dots, 15\}$ is the set of counting outcomes $y$ from which continued fractions on $y/16$ return a non-trivial factor pair. For a counting distribution $p$, $P(p) = \sum_{y \in S} p(y)$.

**Definition 8 (OTOC).** For Hermitian unitary operators $W, V$ and idle unitary $U$, with $W(t) = (U^t)^\dagger W U^t$:

$$C(t) = 2^{-n} \cdot \|[W(t), V]\|^2, \quad \text{with } \|\cdot\| \text{ the Frobenius norm}$$

**Assumptions.**
* **A1.** Echo pulses are instantaneous and exact.
* **A2.** Incoherent noise in a block is the global depolarising channel $D_p$, applied after the idle window.
* **A3.** Crosstalk acting during a gate is absorbed into the adjacent idle window; gates $G$ are exact.
* **A4.** $\eta = 0$ (no exchange) wherever a statement is called exact; $\eta > 0$ is treated numerically in $\S 8$.

### Figure 1: Architecture of the Three-Layer Stack
> **Calibration feeds compensation and rescaling; the Walsh echo needs no input**
>
> * **Layer 1a $\cdot$ GHZ populations + MQC sweep**
>   * Measures $\lambda$ per block and $\sum \delta\omega$
>   * Blind to $ZZ$ and exchange (exact)
>   * Feeds $\lambda$ per block directly to **Readout rescaling (post-processing)**
> * **Layer 1b $\cdot$ Conditional Ramsey**
>   * Measures every $J_{ij}$ and $\delta\omega_i$
>   * $28$ settings per $2 \times 4$ plaquette
>   * Feeds $\hat{J}_{ij}, \delta\hat{\omega}_i$ to **Layer 2, route A $\cdot$ Compensation**
> * **Layer 2, route B $\cdot$ Walsh echo**
>   * $X$ on $B, A, B, A$ sublattices, $T/4$ apart
>   * Cancels $Z$ and $ZZ$ to first order; no input
>   * Feeds into **Layer 3**
> * **Layer 2, route A $\cdot$ Compensation**
>   * Applies $\exp(+iT\hat{H})$ after each idle window
>   * Misses exchange and shot-to-shot drift
>   * Feeds into **Layer 3**
> * **Layer 3 $\cdot$ Shor $N = 15, a = 7$**
>   * $5$ blocks, each followed by a refocused idle window
>   * Outputs counting histogram to **Readout rescaling (post-processing)**
> * **Readout rescaling (post-processing)**
>   * Rescales the histogram; per-shot success unchanged
>
> *Figure 1. The three-layer stack: calibration ($\S 3$), refocusing ($\S 4$), payload ($\S 6$), post-processing ($\S 7$). Route B is the refocusing that $\S 4$ proves exact; route A is analysed for comparison.*

---

## 2. Lattice

**Lemma 1.** The $10 \times 12$ square grid has $218$ edges and is bipartite under the colouring $A = \{(r, c) : r + c \text{ even}\}$, $B = \{(r, c) : r + c \text{ odd}\}$. Its maximum degree is $4$ and its mean degree is $436/120 \approx 3.633$. The same holds for every rectangular sub-grid, in particular for $\Pi$.

*Proof.* Each of the $10$ rows has $11$ horizontal edges and each of the $12$ columns has $9$ vertical edges, so $|E| = 10 \cdot 11 + 9 \cdot 12 = 110 + 108 = 218$. Every edge changes $r + c$ by exactly $1$, so it joins $A$ to $B$. An interior vertex has the four neighbours $(r \pm 1, c)$, $(r, c \pm 1)$, and no vertex has more. The degree sum is $2|E| = 436$ over $120$ vertices. A sub-grid inherits the colouring. $\blacksquare$

The counts match the published Nighthawk layout: $120$ qubits joined by $218$ tunable couplers in a square lattice (Help Net Security). For $\Pi$, $|E_\Pi| = 2 \cdot 3 + 4 = 10$.

---

## 3. Calibration (Layer 1)

The GHZ state identifies the depolarising survival $\lambda$ and the detuning sum exactly, and is provably blind to every $ZZ$ coupling; a conditional Ramsey protocol identifies each $J_{ij}$ exactly. Write $|0\rangle$ for the $n$-qubit string $0\dots 0$ and $|1\rangle$ for $1\dots 1$ where no confusion arises.

**Lemma 2 (GHZ endpoints are eigenvectors).** For all $J, \delta\omega, \epsilon$ and $\eta$, $H|0\dots 0\rangle = E_0|0\dots 0\rangle$ and $H|1\dots 1\rangle = E_1|1\dots 1\rangle$, with

$$E_0 = \sum_{(i,j)} J_{ij} + \sum_i (\delta\omega_i + \epsilon_i), \quad E_1 = \sum_{(i,j)} J_{ij} - \sum_i (\delta\omega_i + \epsilon_i)$$

$$E_0 - E_1 = 2 \sum_i (\delta\omega_i + \epsilon_i)$$

*Proof.* $Z_i$ and $Z_i Z_j$ are diagonal, with $Z_i Z_j = +1$ on both strings and $Z_i = +1$ on $0\dots 0$, $-1$ on $1\dots 1$. On a pair of equal bits, $X_i X_j |00\rangle = |11\rangle$ and $Y_i Y_j |00\rangle = (i)(i)|11\rangle = -|11\rangle$, so $(X_i X_j + Y_i Y_j)|00\rangle = 0$ and likewise on $|11\rangle$. Summing the terms gives $E_0$ and $E_1$. $\blacksquare$

**Corollary 2.1 (cat states cannot see $ZZ$).** Let $x \in \{0,1\}^n$ and $\bar{x}$ its complement. Under $H_D$, the relative phase between $|x\rangle$ and $|\bar{x}\rangle$ does not depend on any $J_{ij}$. Hence no measurement on any state in $\text{span}\{|x\rangle, |\bar{x}\rangle\}$ after diagonal drift carries information about the couplings.

*Proof.* $Z_i Z_j$ has eigenvalue $(-1)^{x_i + x_j}$ on $|x\rangle$ and $(-1)^{(1-x_i)+(1-x_j)} = (-1)^{x_i + x_j}$ on $|\bar{x}\rangle$. The $ZZ$ part of the energy is therefore equal on both strings and cancels in the phase difference; populations are invariant because $H_D$ is diagonal. $\blacksquare$

**Lemma 3 (population identity).** Start from $|\text{GHZ}\rangle = (|0\dots 0\rangle + |1\dots 1\rangle)/\sqrt{2}$ and apply $L$ blocks with $G = I$ and $V = \exp(-iTH)$ (no echo). Let $P_0, P_1$ be the output populations of $0\dots 0$ and $1\dots 1$. Then, for every $\mu$,

$$P_0 + P_1 = \lambda^L + (1 - \lambda^L) \cdot \frac{2}{d}, \quad \lambda = \left[ \frac{P_0 + P_1 - 2/d}{1 - 2/d} \right]^{1/L}$$

*Proof.* By Lemma 2, $V$ acts on $\text{span}\{|0\dots 0\rangle, |1\dots 1\rangle\}$ as $\text{diag}(\exp(-iE_0 T), \exp(-iE_1 T))$, so $V\rho V^\dagger$ keeps $P_0 + P_1 = 1$ for $\rho$ in that span. Since $V I V^\dagger = I$, induction gives the $L$-block output $\lambda^L V^L \rho (V^\dagger)^L + (1 - \lambda^L)I/d$. The maximally mixed term puts $2/d$ on the two strings. Averaging over $\mu$ is linear and leaves the identity unchanged. Solve for $\lambda$; the map is invertible because $d > 2$. $\blacksquare$

**Lemma 4 (multiple-quantum-coherence readout).** Let $c = \langle 0\dots 0 | \rho | 1\dots 1 \rangle$ and $M(\varphi) = \bigotimes_q (\cos\varphi X_q + \sin\varphi Y_q)$. For any $\rho$,

$$\text{Tr}[M(\varphi)\rho] = \sum_{a \in \{0,1\}^n} \exp(i\varphi(2|a| - n)) \cdot \rho_{\bar{a}, a}$$

so the coefficient of $\exp(in\varphi)$ is exactly $c$. If the only non-zero coherence of the form $\rho_{\bar{a}, a}$ is $c$ and its conjugate, as for every output of Lemma 3, then $\text{Tr}[M(\varphi)\rho] = C \cos(n\varphi + \phi)$ with $C = 2|c|$, $\phi = \arg c$. Moreover

$$\frac{1}{2}(P_0 + P_1 + C) = \max_{\theta} \langle \text{GHZ}_\theta | \rho | \text{GHZ}_\theta \rangle, \quad |\text{GHZ}_\theta\rangle = \frac{|0\dots 0\rangle + \exp(i\theta)|1\dots 1\rangle}{\sqrt{2}}$$

*Proof.* $\cos\varphi X + \sin\varphi Y = \exp(-i\varphi)|0\rangle\langle 1| + \exp(i\varphi)|1\rangle\langle 0|$, so $M(\varphi)$ has non-zero entries only at $(a, \bar{a})$, equal to the product over qubits of $\exp(-i\varphi)$ where $a_q = 0$ and $\exp(+i\varphi)$ where $a_q = 1$, that is $\exp(i\varphi(2|a| - n))$. Then $\text{Tr}[M\rho] = \sum_a M_{a, \bar{a}} \rho_{\bar{a}, a}$. Frequency $n$ arises only from $a = 1\dots 1$, with coefficient $\rho_{0\dots 0, 1\dots 1} = c$; frequency $-n$ gives $c^*$. For the fidelity, $\langle \text{GHZ}_\theta | \rho | \text{GHZ}_\theta \rangle = \frac{1}{2}(P_0 + P_1 + 2\text{Re}(\exp(i\theta)c))$, maximal at $\theta = -\arg c$. $\blacksquare$

**Sampling.** With $\varphi_m = \pi m / (n + 1)$, $m = 0, \dots, 2n + 1$, the $2n + 2$ samples separate all frequencies in $\{-n, \dots, n\}$ (span $2n < 2n + 2$), so the discrete Fourier coefficient at $n$ returns $c$ exactly for any $\rho$.

**Corollary 4.1 (detuning sum).** If $\mu$ is symmetric with a positive characteristic function (for example Gaussian), the output of Lemma 3 has $\arg c = -2LT \sum_i \delta\omega_i \pmod{2\pi}$, so $\sum_i \delta\omega_i = -\phi / (2LT)$ whenever $2LT|\sum_i \delta\omega_i| < \pi$.

*Proof.* By Lemmas 2 and 3, $c = (\lambda^L / 2) \exp(-2iLT \sum \delta\omega_i) \mathbb{E}_\mu[\exp(-2iLT \sum \epsilon_i)]$, and the expectation is real and positive by hypothesis. $\blacksquare$

**Lemma 5 (conditional Ramsey, $\eta = 0$).** Prepare qubit $i$ in $|+\rangle$ and every other qubit $k$ in a computational state with sign $z_k = +1$ for $|0\rangle$, $-1$ for $|1\rangle$. After time $\tau$ under $H_D$,

$$\langle X_i \rangle + i \langle Y_i \rangle = \exp(2i\tau h_i(z)) \cdot \mathbb{E}_\mu[\exp(2i\tau \epsilon_i)], \quad h_i(z) = \delta\omega_i + \sum_{k \sim i} J_{ik} z_k$$

Under the hypothesis of Corollary 4.1 and $2\tau|h_i| < \pi$, the estimators

$$\hat{J}_{ij} = \frac{1}{2}[h_i(z_j = +1) - h_i(z_j = -1)] \quad (\text{other } z_k = +1), \quad \delta\hat{\omega}_i = h_i(+1, \dots, +1) - \sum_{k \sim i} \hat{J}_{ik}$$

return $J_{ij}$ and $\delta\omega_i$ exactly.

*Proof.* $H_D$ is diagonal, so the product state stays a product and qubit $i$ sees the field $h_i(z) + \epsilon_i : |0\rangle \mapsto \exp(-i\tau(h + \epsilon))|0\rangle$, $|1\rangle \mapsto \exp(i\tau(h + \epsilon))|1\rangle$. For $a|0\rangle + b|1\rangle$, $\langle X \rangle + i \langle Y \rangle = 2a^* b = \exp(2i\tau(h + \epsilon))$. Averaging over $\mu$ multiplies by a positive real factor, so the argument is $2\tau h_i$. Flipping $z_j$ changes $h_i$ by $-2J_{ij}$. $\blacksquare$

One register needs $n + 2|E_\Pi| = 28$ settings when each coupling is read from both ends. With $\eta > 0$ the product state is no longer preserved and the estimators acquire an $O(\eta)$ bias, measured in $\S 8$.

---

## 4. Refocusing (Layer 2)

The Walsh sequence of Definition 5 cancels every diagonal drift term exactly, to all orders and for any unknown quasi-static detuning; compensation cancels only what was measured.

**Theorem 1 (exact Walsh refocusing).** Assume A1 and $\eta = 0$. For all couplings $J$, detunings $\delta\omega$, realisations $\epsilon$ and window lengths $T$,

$$W_T(H_D) = I$$

*Proof.* Write $U = \exp(-i\tau H_D)$ and $Q_1 = X_B$, $Q_2 = X_A$, $Q_3 = X_B$, $Q_4 = X_A$, so $W_T = Q_4 U Q_3 U Q_2 U Q_1 U$. Define the frames $F_0 = I$ and $F_k = Q_k \cdots Q_1$. Since $X_A$ and $X_B$ have disjoint supports, commute and square to $I$, $F_4 = I$. Because $F_k F_{k-1}^\dagger = Q_k$,

$$W_T = F_4 (F_3^\dagger U F_3)(F_2^\dagger U F_2)(F_1^\dagger U F_1)(F_0^\dagger U F_0) = \exp(-i\tau H_3)\exp(-i\tau H_2)\exp(-i\tau H_1)\exp(-i\tau H_0), \quad H_k = F_k^\dagger H_D F_k$$

The frames are $F_0 = I$, $F_1 = X_B$, $F_2 = X_A X_B$, $F_3 = X_A$. Conjugation by $X$ flips the sign of $Z$ on that qubit, and by Lemma 1 every coupling joins $A$ to $B$, so

$$H_k = \sum_{(i,j)} a_k b_k J_{ij} Z_i Z_j + \sum_{i \in A} a_k (\delta\omega_i + \epsilon_i) Z_i + \sum_{i \in B} b_k (\delta\omega_i + \epsilon_i) Z_i$$

$$a = (+, +, -, -), \quad b = (+, -, -, +)$$

All $H_k$ are diagonal and therefore commute, so the product equals $\exp(-i\tau \sum_k H_k)$. Since $\sum_k a_k = \sum_k b_k = \sum_k a_k b_k = 0$, $\sum_k H_k = 0$ and $W_T = I$. $\blacksquare$

**Remark on Theorem 1.** No condition on $T\|H\|$ is needed: the toggled Hamiltonians commute, so there are no higher-order terms. With $\eta > 0$, $X_i X_j$ is invariant under every frame and $Y_i Y_j$ picks up the sign $a_k b_k$, so the first-order average Hamiltonian is $\frac{1}{4}\sum_k H_k = \eta \sum J_{ij} X_i X_j$ plus commutator terms of order $\tau^2 \|H\|^2$. The echo therefore leaves an $XX$ residual.

**Proposition 1 (compensation residual).** For diagonal $H_D$ and $\hat{H}$,

$$\exp(iT\hat{H})\exp(-iTH_D) = \exp\left(-iT\left[\sum_{(i,j)} (J_{ij} - \hat{J}_{ij}) Z_i Z_j + \sum_i (\delta\omega_i - \delta\hat{\omega}_i + \epsilon_i) Z_i\right]\right)$$

This is $I$ for every realisation only if $\hat{J} = J$, $\delta\hat{\omega} = \delta\omega$ and $\epsilon = 0$ almost surely. If the $\epsilon_i$ are independent $\mathcal{N}(0, \sigma^2)$, each window multiplies the coherence between two strings that differ only at qubit $i$ by $\exp(-2T^2 \sigma^2)$, in addition to the deterministic residual phase.

*Proof.* The exponents commute, so they add. The $\epsilon$-dependent phase difference between the two strings is $2T\epsilon_i$, and $\mathbb{E}[\exp(2iT\epsilon_i)] = \exp(-2T^2 \sigma^2)$ for a centred Gaussian. $\blacksquare$

With $T = 4.4$ and $\sigma = 0.015$ this factor is $0.9913$ per window and per qubit, which the echo of Theorem 1 removes and compensation cannot.

**Lemma 6 (OTOC identity).** For Hermitian unitary $W$ and $V$, with $F(t) = 2^{-n}\text{Tr}[W(t) V W(t) V]$,

$$C(t) = 2(1 - F(t)), \quad F(t) \text{ is real}, \quad 0 \le C(t) \le 4$$

*Proof.* $W(t)$ is again Hermitian and unitary. $[W(t), V]^\dagger [W(t), V] = (VW - WV)(WV - VW) = 2I - VWVW - WVWV$, with $W = W(t)$. By cyclicity $\text{Tr}[VWVW] = \text{Tr}[WVWV]$, and that trace equals its own conjugate $\text{Tr}[(WVWV)^\dagger] = \text{Tr}[VWVW]$, so $F$ is real. $WVWV$ is unitary, so $|F| \le 1$. $\blacksquare$

**Lemma 7 (short-time growth).** If $[W, V] = 0$ and $U = \exp(-itH)$, then

$$C(t) = t^2 \cdot 2^{-n} \cdot \|[[H, W], V]\|^2 + O(t^3)$$

*Proof.* $W(t) = W + it[H, W] + O(t^2)$, so $[W(t), V] = it[[H, W], V] + O(t^2)$; take the squared Frobenius norm. $\blacksquare$

**Corollary 7.1 (refocused OTOC).** Under the hypotheses of Theorem 1, with one Walsh window per time step, $C(t) = C(0)$ for all $t$; in particular $C \equiv 0$ when $[W, V] = 0$.

*Proof.* Each step is $W_T(H_D) = I$, so $W(t) = W$. $\blacksquare$

The OTOC measures coherent operator spreading only. It is a diagnostic of Layer 2, not a protection mechanism, and it is insensitive to the depolarising part of the channel, which $\S 5$ treats.

---

## 5. Depolarising Invariance

The survival factor $\lambda^B$ is fixed by the noise alone: every unitary layer, refocusing included, commutes with it and can only change the noiseless part of the output.

**Lemma 8 (commutation and composition).** For every unitary $U$ and $p, q \in [0, 1]$,

$$U D_p(\rho) U^\dagger = D_p(U \rho U^\dagger), \quad D_p \circ D_q = D_r \text{ with } r = 1 - (1 - p)(1 - q)$$

*Proof.* $U I U^\dagger = I$ gives the first identity. For the second, $(1 - p)[(1 - q)\rho + qI/d] + pI/d = (1 - p)(1 - q)\rho + [1 - (1 - p)(1 - q)]I/d$. $\blacksquare$

**Theorem 2 (fixed depolarising factor).** For any gates $G_1, \dots, G_B$ and any idle unitaries $V_1, \dots, V_B$, which may depend on the realisation $\epsilon$, the $B$-block output averaged over $\mu$ is

$$\rho_{\text{out}} = \lambda^B \cdot \mathbb{E}_\mu[U_V \rho_0 U_V^\dagger] + (1 - \lambda^B) \cdot I/d, \quad U_V = V_B G_B \cdots V_1 G_1$$

The counting marginal is therefore $p = \lambda^B p_V + (1 - \lambda^B) u$, with $u$ uniform on the $d_C$ outcomes, and

$$P(p) = \lambda^B P(p_V) + (1 - \lambda^B) \cdot \frac{|S|}{d_C}$$

*Proof.* By Lemma 8 each block's $D_p$ can be moved past all later unitaries, and the $B$ copies compose to $D_{1 - \lambda^B}$. Averaging over $\mu$ is linear. The partial trace of $I/d$ over $R_W$ is $I/d_C$. $P$ is linear in $p$. $\blacksquare$

**Corollary of Theorem 2.** No choice of idle unitaries changes $\lambda^B$. A refocusing that restores the ideal circuit for every $\epsilon$, $U_V = U_{\text{ideal}}$, attains $P = \lambda^B P(p_{\text{ideal}}) + (1 - \lambda^B)|S|/d_C$, and this is the value of every such refocusing.

---

## 6. Shor Payload (Layer 3)

For $N = 15$ and $a = 7$ the ideal counting distribution is uniform on $\{0, 4, 8, 12\}$, and exactly two of the sixteen outcomes factor $N$, so one ideal shot succeeds with probability $1/2$.

**Definition 9 (controlled multipliers).** For $k = 0, \dots, 3$ let $m_k = 7^{2^k} \bmod 15$ and

$$C\text{-}U_k : |c\rangle|y\rangle \mapsto \begin{cases} |c\rangle|m_k y \bmod 15\rangle & \text{if } c_k = 1 \text{ and } 0 < y < 15, \\ |c\rangle|y\rangle & \text{otherwise} \end{cases}$$

where $c_k$ is bit $k$ of the counting value $c$. Each $C\text{-}U_k$ is a permutation of basis states, hence unitary, because $\gcd(m_k, 15) = 1$ makes $y \mapsto m_k y \bmod 15$ a bijection of $\{1, \dots, 14\}$.

The circuit is $B = 5$ blocks: $C\text{-}U_0, \dots, C\text{-}U_3$, then the inverse QFT on $R_C$, from the input $|+\rangle^{\otimes 4}|1\rangle$.

**Lemma 9 (ideal counting distribution).** $p_{\text{ideal}}(y) = 1/4$ for $y \in \{0, 4, 8, 12\}$ and $0$ otherwise.

*Proof.* The four multipliers compose to $|c\rangle|1\rangle \mapsto |c\rangle|7^c \bmod 15\rangle$. The multiplicative order of $7 \bmod 15$ is $4$ ($7, 4, 13, 1$), so writing $c = 4j + s$,

$$\frac{1}{4}\sum_{c=0}^{15} |c\rangle|7^c\rangle = \frac{1}{4}\sum_{s=0}^3 \left(\sum_{j=0}^3 |4j + s\rangle\right)|7^s\rangle$$

The inverse QFT sends $|x\rangle$ to $\frac{1}{4}\sum_y \exp(-2\pi i xy / 16)|y\rangle$, so the amplitude on $|y\rangle|7^s\rangle$ is $\frac{1}{16}\exp(-2\pi i sy / 16)\sum_j \exp(-2\pi i jy / 4)$. The inner sum is $4$ if $4$ divides $y$ and $0$ otherwise. Hence $p(y) = \sum_s (1/4)^2 = 1/4$ for $y \equiv 0 \pmod 4$. $\blacksquare$

**Lemma 10 (success set).** With continued fractions taken to denominators at most $15$, $S = \{4, 12\}$, so $|S| = 2$ and $P(p_{\text{ideal}}) = 1/2$.

*Proof.* By enumeration of all sixteen outcomes. A candidate period $r$ is accepted when $r$ is even, $7^r \equiv 1 \pmod{15}$ and $x = 7^{r/2} \bmod 15 \notin \{1, 14\}$; then $\gcd(x \pm 1, 15)$ is the factor pair.

| $y$ | Convergent | $r$ | $7^r \bmod 15$ | $x = 7^{r/2} \bmod 15$ | In $S$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $0$ | $0$ | $1$ | $7$ | — | no |
| $4, 12$ | $1/4, 3/4$ | $4$ | $1$ | $4$ | yes: $\gcd(3, 15) = 3, \gcd(5, 15) = 5$ |
| $8$ | $1/2$ | $2$ | $4$ | — | no |
| $2, 6, 10, 14$ | $1/8, 3/8, 5/8, 7/8$ | $8$ | $1$ | $1$ | no |
| $1, 15$ | $1/15, 14/15$ | $15$ | $13$ | — | no |
| $3, 13$ | $2/11, 9/11$ | $11$ | $13$ | — | no |
| $5, 11$ | $4/13, 9/13$ | $13$ | $7$ | — | no |
| $7, 9$ | $4/9, 5/9$ | $9$ | $7$ | — | no |

Only $y = 4$ and $12$ pass, and each carries ideal probability $1/4$. $\blacksquare$

---

## 7. Main Result

**Theorem 3.** Assume A1–A4 with $\eta = 0$. Run the circuit of $\S 6$ on $\Pi$ with every idle window replaced by the Walsh sequence $W_T(H)$ of Definition 5. Then for all couplings $J$, detunings $\delta\omega$, window lengths $T$ and distributions $\mu$,

$$p = \lambda^5 p_{\text{ideal}} + (1 - \lambda^5) u, \quad P(p) = \frac{\lambda^5}{2} + \frac{1 - \lambda^5}{8}$$

For $\lambda = 0.94$, $\lambda^5 = 0.733904$ and $P = 0.366952 + 0.033262 = 0.400214$.

*Proof.* Lemma 1 gives the bipartition that Theorem 1 needs, so every idle window equals $I$ for every $\epsilon$, and $U_V = U_{\text{ideal}}$. Theorem 2 with $B = 5$ gives $p = \lambda^5 p_{\text{ideal}} + (1 - \lambda^5) u$. Lemmas 9 and 10 give $P(p_{\text{ideal}}) = 1/2$ and $|S|/d_C = 2/16$. Substitute. $\blacksquare$

By the corollary of Theorem 2, $0.400214$ is the value of every refocusing that restores the ideal circuit; refocusing cannot exceed it by lowering the depolarising contribution.

**Corollary A (calibration transfer).** When the calibration blocks of Lemma 3 use the same $D_p$ as the circuit (A2), the survival $\lambda^5$ needed in Theorem 3 is identified from GHZ populations alone:

$$\Lambda = \left[ \frac{P_0 + P_1 - 2/d}{1 - 2/d} \right]^{5/L}$$

*Proof.* Lemma 3 identifies the per-block $\lambda$; Theorem 2 raises it to the number of blocks. $\blacksquare$

**Corollary B (readout rescaling).** Let $\Pi_\Delta$ be Euclidean projection onto the probability simplex and

$$R_\Lambda(p) = \Pi_\Delta\left[ \frac{p - (1 - \Lambda) u}{\Lambda} \right]$$

1. Under Theorem 3 with $\Lambda = \lambda^5$, $R_\Lambda(p) = p_{\text{ideal}}$ exactly.
2. For every $p$, $R_\Lambda$ preserves the weak order of the outcome probabilities, so it cannot promote an outcome over another.
3. $R_\Lambda$ acts on an estimated histogram, not on device samples, so the per-shot success $P(p)$ of the device is unchanged by it.
4. Each rescaled entry has variance $\text{Var}(\hat{p}(y))/\Lambda^2$, a factor $1/\Lambda^2 \approx 1.857$ at $\Lambda = 0.733904$.

*Proof.* (1) The argument of $\Pi_\Delta$ equals $p_{\text{ideal}}$, which lies in $\Delta$ and is fixed by $\Pi_\Delta$. (2) The affine map $v \mapsto (v - (1 - \Lambda)/d_C)/\Lambda$ is strictly increasing in each coordinate, and Euclidean projection onto $\Delta$ has the form $v \mapsto \max(v - \theta, 0)$ for one scalar $\theta$, which is non-decreasing. (3) is immediate. (4) Scaling a random variable by $1/\Lambda$ scales its variance by $1/\Lambda^2$. $\blacksquare$

---

## 8. Numerical Corroboration

An 8-qubit density-matrix simulation of $\Pi$ returns per-shot success $0.4002$ for the Walsh echo whenever the drift is diagonal, matching Theorem 3's $0.400214$ to four decimals; adding exchange, which the theorem excludes, lowers it to $0.3912$.

| Model parameter | Value |
| :--- | :--- |
| $J_{ij}$ | $0.065 \times U(0.75, 1.25)\text{ rad}$ per time unit |
| Exchange $\eta$ | $0.15$ ($0$ where stated) |
| $\delta\omega_q$ | $0.022 \times ((q \bmod 3) - 1)$ |
| Quasi-static $\epsilon_q$ | $\mathcal{N}(0, 0.015^2)$ per shot ($0$ where stated) |
| Depolarising $p$ | $0.06$ per block, all modes |
| Window $T$, Ramsey wait $\tau$, GHZ blocks $L$ | $4.4, 2.0, 3$ |
| Shots per setting | $4000$ |
| Noise realisations | $256$ per calibration, $32$ per circuit run |

**Calibration (Lemmas 2–5), default parameters:**

| Quantity | Estimate | True | Statement |
| :--- | :--- | :--- | :--- |
| GHZ relative phase from $ZZ$, $10$ units | $0.000\text{ rad}$ | $0$ | Lemma 2 |
| GHZ relative phase from $XX + YY$, $10$ units | $0.000\text{ rad}$ | $0$ | Lemma 2 |
| GHZ relative phase from $Z$ detuning, $10$ units | $0.440\text{ rad}$ | $0.440\text{ rad}$ | Lemma 2 |
| $\lambda$ per block | $0.9401$ | $0.9400$ | Lemma 3 |
| $\sum \delta\omega_i$ | $-0.0227$ | $-0.0220$ | Corollary 4.1 |
| Largest relative error in $\hat{J}_{ij}$ | $5.0\%$ | $0$ at $\eta = 0$ | Lemma 5 |
| $\Lambda = \lambda^5$ | $0.7344$ | $0.7339$ | Corollary A |

**Per-shot success by noise content.** The first two rows satisfy the hypotheses of Theorem 3; the last two do not.

| Drift terms present | Theorem 3 applies | Unmitigated | Compensated | Walsh echo |
| :--- | :--- | :--- | :--- | :--- |
| $Z + ZZ$ only | yes | $0.098$ | $0.393$ | $0.4002$ |
| $Z + ZZ + \text{quasi-static }\epsilon$ | yes | $0.099$ | $0.353$ | $0.4002$ |
| $Z + ZZ + \text{exchange}$ | no | $0.085$ | $0.337$ | $0.391$ |
| All terms (default) | no | $0.088$ | $0.305$ | $0.391$ |

The echo is insensitive to $\epsilon$, as Theorem 1 states; compensation loses $0.007$ to estimation error and a further $0.040$ to $\epsilon$, as Proposition 1 predicts. Under the hypotheses, $R_\Lambda$ returns fidelity $0.9995$ to $p_{\text{ideal}}$ (Corollary B). With all terms the fidelities to $p_{\text{ideal}}$ are $0.792$ (echo), $0.626$ (compensated) and $0.197$ (unmitigated); after $R_\Lambda$ they read $0.988$, $0.762$ and $0.177$, with per-shot success unchanged.

**OTOC after $t$ idle windows ($W = X$ on the first qubit, $V = X$ on the second, $\epsilon = 0$, $\eta = 0.15$):**

| Pair | Mode | $t = 1$ | $t = 3$ | $t = 5$ |
| :--- | :--- | :--- | :--- | :--- |
| Neighbours $(1, 5)$ | Unmitigated | $1.371$ | $3.189$ | $1.053$ |
| Neighbours $(1, 5)$ | Compensated | $0.037$ | $0.305$ | $0.730$ |
| Neighbours $(1, 5)$ | Walsh echo | $0.001$ | $0.007$ | $0.018$ |
| Distance 4 $(0, 7)$ | Unmitigated | $0.000$ | $0.005$ | $0.079$ |
| Distance 4 $(0, 7)$ | Compensated | $0.000$ | $0.000$ | $0.003$ |
| Distance 4 $(0, 7)$ | Walsh echo | $0.000$ | $0.000$ | $0.000$ |

The echo's non-zero neighbour values come from the $XX$ residual of the Remark on Theorem 1; at $\eta = 0$ Corollary 7.1 makes them exactly zero. The unmitigated oscillation reflects the short recurrence time of $8$ qubits.

### Sweep Plot Summary: Per-Shot Factoring Success vs. $ZZ$ and Exchange Strength
> **Walsh echo stays within $0.024$ of the $0.400$ ceiling up to $2\times ZZ$**  
> *Per-shot factoring success, Shor $N = 15, a = 7$; equal depolarisation ($p = 0.06$ per block) in every mode.*
> * **Horizontal Reference:** Depolarising ceiling at $P(\text{success}) = 0.400$.
> * **Sweep Range ($x$-axis):** $ZZ$ and exchange strength as a multiple of the default model ($0\times, 0.25\times, 0.5\times, 1\times \text{ [default]}, 1.5\times, 2\times$).
> * **End-point values at $2\times$ strength:**
>   * **Walsh echo:** starts at $0.4002$ ($0\times$), $0.391$ ($1\times$), ends at $0.377$ ($2\times$).
>   * **Compensated:** starts at $\approx 0.353$ ($0\times$), $0.305$ ($1\times$), ends at $0.227$ ($2\times$).
>   * **Unmitigated:** starts at $\approx 0.304$ ($0\times$), dips to $0.088$ ($1\times$), ends at $0.115$ ($2\times$).
> * *Source:* `shrs_nighthawk_stealth.py` sweep, seed $42$, $6\text{ strengths} \times 3\text{ modes}$, $32$ noise realisations per point, $\eta = 0.15$ throughout.

Scaling the $ZZ$ and exchange strengths together moves the echo only from $0.4002$ to $0.3766$ over $0$ to $2\times$, all of it due to exchange. The full default run takes $2\text{ min } 35\text{ s}$ on one CPU core.

---

## 9. Chip-Scale Implementation

`nighthawk_ghz.py` carries Layers 1a and 2 from the 8-qubit register $\Pi$ to whole Nighthawk-class registers of $16$ to $61$ qubits, simulated with Qrack's ACE backend. Off the seams it reproduces $\S 3$ and $\S 4$ exactly; on the seams no Layer 2 mode helps.

ACE splits a register into patches, each held by its own simulator. A seam qubit belongs to more than one patch, and a coupler between patches (a seam $CX$) is applied through replicas rather than as an exact two-qubit gate. Every run is checked against an exact reference (`QrackSimulator`) with the same gates and the same noise draws.

| Object in the proof | Implementation in `nighthawk_ghz.py` |
| :--- | :--- |
| Register $\Pi$ (Definition 1) | 3-qubit chains tiled over the register (rows and columns, offsets $0$ to $2$), or one GHZ over the whole register along a spanning tree: `bfs`, or `seam` (fewest seam crossings) |
| Drift $H_D$ (Definition 2, $\eta = 0$) | Per idle window: $Z$ rotations for $\delta\omega + \epsilon$; each coupling inside a unit as $CX \cdot R_z \cdot CX$; each coupling to an idle neighbour as a $Z$ rotation conditioned on that neighbour |
| Walsh sequence $W_T$ (Definition 5) | $X$ on sublattice $B, A, B, A$ between four segments of $\tau = T/4$; units placed so that no two share a coupler |
| Compensation (Definition 6) | Oracle $\hat{H} = H_D$ without $\epsilon$ |
| $\lambda$ (Lemma 3) | From $P_0 + P_1$ per unit |
| MQC readout (Lemma 4) | $2k + 2$ phases $\varphi_m = \pi m / (k + 1)$ on a $k$-qubit unit; $C = 2|c|$, $F_{\text{GHZ}} = (P_0 + P_1 + C)/2$; used up to $12$ qubits |
| Fidelity above $12$ qubits | Clifford witness $F \ge P_0 + P_1 + P(\text{even } X \text{ parity}) - 1$ |
| $\S 8$ noise model | $J = 0.065 \times U(0.75, 1.25)$, $\delta\omega_q = 0.022((q \bmod 3) - 1)$, $\sigma = 0.015$, $T = 4.4$, $L = 3$, seed $42$; $K = 4$ noise realisations per setting ($256$ in $\S 8$) |

The registers are `nighthawk2p16`, `nighthawk2p28` and `nighthawk2p57` (two patches each) and `nighthawk61r2` ($61$ qubits, two patches, $6$ seam qubits). The release lattice is stored as $12\text{ rows} \times 10\text{ columns}$, the transpose of Definition 1; Lemma 1 is unchanged. All runs use PyQrack 2.29.5 with Qrack 10.13.2 on CPU.

### 9.1 Layer 1a Across the Seams

Without drift, a 3-qubit GHZ chain keeps $F_{\text{GHZ}} = 1.000$ everywhere except on chains that hold a seam qubit, where it falls to $0.77$ with one seam $CX$ and to $0.35\text{--}0.43$ with two.

| Register | Units | Off the seam | Holding a seam qubit, 1 seam $CX$ | Holding a seam qubit, 2 seam $CX$s |
| :--- | :--- | :--- | :--- | :--- |
| `nighthawk2p28` ($3$ seam qubits) | $30$ | $1.002$ ($16$ units) | $0.767$ ($9$) | $0.353$ ($5$) |
| `nighthawk61r2` ($6$ seam qubits) | $82$ | $1.000$ ($58$) | $0.773$ ($15$) | $0.426$ ($9$) |

Mean $F_{\text{GHZ}}$ from Lemma 4; $256\text{ shots} \times 4\text{ backends}$ per setting on `nighthawk61r2`, $\times 2$ on `nighthawk2p28`. The exact reference reads $0.996\text{--}1.002$ in every row. Lemma 3 gives $\lambda$ per block $1.000$, $0.80\text{--}0.83$ and $0.55\text{--}0.62$ for the three columns. Values above $1$ are shot noise in $C$.

* The loss sits on the seam qubits, not around them. Every chain that only touches a seam qubit's neighbour is at $1.000$ with phase $0.00 \pm 0.01$.
* It is a decorrelation at the crossing, not an error per gate. Folding each seam $CX$ three times (`fold:1`) moves the seam units only from $0.638$ to $0.622$ in $F_0$ and leaves every other unit at $1.000$.

### 9.2 Layer 2 Across the Seams

Off the seam the Walsh echo restores $F_{\text{GHZ}} = 1.001$, as Theorem 1 predicts; on chains holding a seam qubit every Layer 2 mode lowers the fidelity below the no-drift value, and the exact reference stays at $0.88\text{--}1.00$.

| `nighthawk61r2` units | No drift | Unmitigated | Compensated (oracle $\hat{H}$) | Walsh echo | Exact: unmitigated / compensated / echo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Off the seam ($58$) | $1.000$ | $0.901$ ($F_0\ 0.581$) | $0.900$ ($F_0\ 0.877$) | $1.001$ | $0.901 / 0.900 / 1.002$ |
| Holding a seam qubit, 1 seam $CX$ ($15$) | $0.773$ | $0.650$ | $0.627$ | $0.671$ | $0.880 / 0.884 / 0.999$ |
| Holding a seam qubit, 2 seam $CX$s ($9$) | $0.426$ | $0.304$ | $0.274$ | $0.240$ | $0.886 / 0.887 / 1.000$ |

Mean $F_{\text{GHZ}}$, $L = 3$ windows, $256\text{ shots} \times K = 4$ realisations per setting, the same $\epsilon$ draws for ACE and the exact reference. `nighthawk2p28` repeats the pattern: $1.000$ off the seam under the echo, $0.183$ on the two-seam-$CX$ units.

* Off the seam, ACE reproduces $\S 4$. Unmitigated drift leaves $F_{\text{GHZ}}$ at $0.90$ (the $\epsilon$ dephasing of Proposition 1 over $K = 4$ draws) and $F_0$ at $0.58$ (the deterministic phase). Compensation restores the phase ($F_0\ 0.88$) but not $\epsilon$. The echo removes both, phase $0.00 \pm 0.01$ (Theorem 1, Corollary 7.1).
* On the seam, the drift itself crosses the seam. A coupling $J_{ij}$ between two patches is one more seam crossing per segment, so the echo's four segments cost most: on the two-seam-$CX$ units it falls to $0.240$, below unmitigated ($0.304$).
* Corollary 4.1 needs the calibration's $K$. At $k = 3$ the $\epsilon$ phase spread over $LT = 13.2$ is $2LT\sigma\sqrt{k} \approx 0.7\text{ rad}$ on a 3-qubit unit, larger than the $\delta\omega$ signal; the detuning-sum estimate needs $K \approx 256$, as in $\S 8$.

### 9.3 One GHZ Over the Whole Register

A single GHZ over the register does not survive the seam on ACE: the coherence $C$ and $\langle X\dots X \rangle$ are $\approx 0$ in every chip-wide run, while the exact reference is ideal.

| Register | Tree | Depth | $CX$s exact / seam | $P_0 + P_1$ | Fidelity | Seam $CX$s that break |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `nighthawk2p16`, no drift | `bfs` | $5$ | $10 / 5$ | $0.583$ | $F_{\text{GHZ}}\ 0.293$ | — |
| `nighthawk2p16`, Walsh echo | `bfs` | $5$ | $10 / 5$ | $0.000$ | — | — |
| `nighthawk2p16`, no drift | `seam` | $6$ | $11 / 4$ | $0.456$ | $F_{\text{GHZ}}\ 0.238$ | — |
| `nighthawk2p28`, no drift | `bfs` | $5$ | $21 / 6$ | $0.293$ | $F \ge -0.19$ | $3\text{ of } 6$ |
| `nighthawk2p28`, no drift | `seam` | $6$ | $23 / 4$ | $0.513$ | $F \ge 0.02$ | $2\text{ of } 4$ |
| `nighthawk2p57`, no drift | `bfs` | $7$ | $44 / 12$ | $0.267$, then $0.000$ | $F \ge -0.24$ | $4\text{ of } 12$ |
| `nighthawk2p57`, no drift | `seam` | $10$ | $50 / 6$ | $0.000$, then $0.005$ | $F \ge -0.51$ | $3\text{ of } 6$ |

`nighthawk2p16` uses Lemma 4 ($256\text{ shots} \times 4\text{ backends}$); the larger registers use the Clifford witness ($512\text{ shots} \times 2$). A seam $CX$ that breaks does so with probability $0.45\text{--}0.52$ and takes its whole subtree with it; the other seam $CX$s break at $0.000\text{--}0.010$, and no $CX$ inside one patch breaks. The two trees differ less than repeated runs do, so fewer seam crossings are not shown to help. `nighthawk61r2` has no chip-wide result on this build: its stabilizer patches fail in ACE's sampler.

**Remark (the seam lies outside A2).** In the same run, units inside one patch keep $P_0 + P_1 = 1.000$ while units holding a seam qubit lose it, so the seam acts as a channel on part of the register, not as the global $D_p$ of Definition 3. Lemma 8 needs a channel that commutes with every unitary; a channel acting on one side of the cut does not commute with gates that cross it. Theorems 2 and 3 therefore say nothing about seam units, and $\S 9.2$ shows that refocusing there can lower the fidelity.

### Figure 2: 3D Visualization of `nighthawk61r2` Under the Walsh Echo
> *Figure 2. `nighthawk61r2` under the Walsh echo, patches pulled apart: the crossbar of seam qubits on top with $F_{\text{GHZ}}$ columns, then the patch 0 (blue nodes) and patch 1 (orange nodes) simulators, joined by a vertical green pillar per seam qubit. The short columns ($0.5 \to 1.0$ scale) sit over the seam.*

The records are read by `nighthawk_graph.py`: views 18–21 draw the per-site fidelity map, the Layer 2 modes, ACE against the exact reference and the chip-wide break maps, and `grid3d` writes one 3D page per register.

---

## 10. Scope of the Hypotheses

Theorem 3 is exact under A1–A4 and says nothing outside them. The excluded effects are:

* **Exchange coupling ($\eta > 0$).** The Walsh sequence leaves $\eta \sum J_{ij} X_i X_j$ at first order (Remark on Theorem 1); $\S 8$ measures the cost at $\eta = 0.15$. A sequence mixing $X$ and $Y$ pulses is the natural extension.
* **Finite-width or imperfect echo pulses (A1).**
* **Incoherent noise that is not global depolarising (A2).** Theorem 2 uses unitary invariance of $D_p$; amplitude damping, local noise and leakage break it. Pauli twirling produces a Pauli channel, not $D_p$; global depolarising needs Clifford twirling or direct validation, for example a flat floor over the outcomes outside $\{0, 4, 8, 12\}$.
* **Crosstalk during gates (A3).**
* **Finite sampling.** Lemmas 3–5 are exact in expectation; finite shots and finite noise realisations add estimation error, measured in $\S 8$.
* **Device values.** The topology of Lemma 1 is the published one; the noise values in $\S 8$ are model parameters at a stress level, not device calibration data.
* **Seam-partitioned simulation.** On Qrack's ACE backend the seam acts as a channel on part of the register (Remark in $\S 9.3$), outside A2; Theorems 2 and 3 do not cover units that hold a seam qubit, and $\S 9.2$ measures refocusing lowering their fidelity.

---

## References

* IBM Quantum Nighthawk processor announcement, Help Net Security, 12 Nov 2025: $120$ qubits, $218$ tunable couplers, square lattice.
* Reference implementation: `shrs_nighthawk_stealth.py`, numpy/scipy density-matrix simulation, seed $42$.
* Chip-scale implementation: `nighthawk_ghz.py` (Layers 1a and 2 on ACE, with an exact `QrackSimulator` reference) and `nighthawk_graph.py` (views 18–21, 3D pages), `shrs-nighthawk` folder of `twobombs/thereminq-examples`; PyQrack 2.29.5, Qrack 10.13.2, seed $42$.
* Method literature, cited from memory and not re-opened for this document:
  * K. X. Wei et al., *Verifying multipartite entangled GHZ states via multiple quantum coherences*, Phys. Rev. A 101, 032343 (2020).
  * H. Hayes, K. Khodjasteh, L. Viola, M. J. Biercuk, *Reducing sequencing complexity in dynamical quantum error suppression by Walsh modulation*, Phys. Rev. A 84, 062323 (2011).
  * J. J. Wallman, J. Emerson, *Noise tailoring for scalable quantum computation via randomized compiling*, Phys. Rev. A 94, 052325 (2016).
  * B. Swingle, *Unscrambling the physics of out-of-time-order correlators*, Nature Physics 14, 988 (2018).
  * J. Duchi et al., *Efficient projections onto the $\ell_1$-ball for learning in high dimensions*, ICML 2008.
