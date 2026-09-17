# Quantum Fourier Transform examples: semiclassical Shor and the QFT condensate

**Authors:** ThereminQ Contributors (twobombs/thereminq-examples). Ai generated; claims, methods and results could be false and should not be taken as anything other than a PoC at best. Caveat emptor.

## Abstract
This directory holds two small, self-checking programs built around the semiclassical Quantum Fourier Transform (QFT), the observation that a QFT followed by a computational-basis measurement can be carried out one qubit at a time with classically controlled single-qubit phases (Griffiths & Niu 1996).

`qft-condensate-semiclassical.py` runs Shor order finding with a single recycled control qubit and an n-qubit work register simulated in PyQrack. The modular multiplier is an oracle permutation, not a gate-level circuit. `qft-cosmos-classical-condensate.py` samples a ring of QFT/IQFT patches joined by classical (measure and re-prepare) seams. That process is exactly and cheaply sampleable at any width and depth, and optionally cross-checks random patches against PyQrack.

Neither program demonstrates a classical-simulation advantage. Their value is that every output is checked against an exact reference.

## 1. Background
For an $n$-qubit register, and in the convention Qrack uses (verified against PyQrack to a total variation distance of about $10^{-7}$):

$$
\text{QFT} |x\rangle = \frac{1}{\sqrt{2^n}} \sum_{y=0}^{2^n-1} \exp\left( -2 \pi i \frac{x \cdot \text{rev}(y)}{2^n} \right) |y\rangle,
\qquad
\text{IQFT} |z\rangle = \frac{1}{\sqrt{2^n}} \sum_{x=0}^{2^n-1} \exp\left( +2 \pi i \frac{x \cdot \text{rev}(z)}{2^n} \right) |x\rangle
$$

where $\text{rev}(y)$ is the bit-reversed integer of $y$. The sign and bit order differ from the usual textbook form. The gate-level QFT uses Hadamards and controlled phase rotations $R_k$ with phase $2\pi/2^k$.

When the QFT is the last step before measurement, Griffiths and Niu showed that every controlled rotation can be replaced by a single-qubit rotation conditioned on bits already measured. Both programs rely on this, and it is also what makes the condensate classically easy (Section 3).

## 2. Semiclassical Shor (`qft-condensate-semiclassical.py`)

**Layout.**
- **Counting register.** A single control qubit is measured and recycled $t = 2n$ times. After each measurement, the accumulated phase $R$ is fed forward as a $-2\pi R$ phase correction. The step using $U^{2^{t-1}}$ reads the least significant bit of $m$ first. This is the single-control arrangement of Parker & Plenio (2000).
- **Work register.** The work register holds $n$ qubits as a full state vector, starting in $|1\rangle$.
- **Modular multiplication.** Controlled $U^{2^j}$ is applied as one permutation on (control, work) through `QrackSimulator.hash()`. The permutation leaves the state unchanged when the control is 0 and maps $x \mapsto A x \bmod N$ when it is 1, with $A = a^{2^j} \bmod N$; values $x \ge N$ are left fixed. The tables are built once with numpy and passed to Qrack pre-packed. A start-up self-test falls back to plain `hash()` if the pre-packed path does not reproduce it.

**What "qubits" means.** The output reports $1+n$ qubits. That is the simulated width, made possible because modular multiplication is an oracle. A reversible gate-level multiplier needs ancillae; Beauregard's construction uses $2n+3$ qubits in total. The $1+n$ figure is therefore not a hardware resource count.

**Why not `mcmuln`/`mcdivn`.** On pyqrack 2.25.2, `muln(11, 21, …)` applied to $x=0$ yields 11 rather than 0. This may be a semantics mismatch rather than a bug, but either way the arithmetic calls were replaced by the explicit permutation.

**Checks, per base $a$.**
- **Classical.** Any order candidate must satisfy $a^r \equiv 1 \pmod N$, and any reported factor must divide $N$. This check is exact at any size.
- **Statistical, when $r \nmid 2^t$.** The script computes the mean log-likelihood gap between the measured $m$ and the same number of draws from the exact QPE distribution for the true order, plus a z-score. The z-score is only indicative below about 30 shots. Deleting the phase correction is caught at $|z| > 3$ in every trial even at 32 shots.
- **Statistical, when $r \mid 2^t$ (for example $N=15$).** The ideal distribution is uniform on the $r$ multiples of $2^t/r$, so the likelihood gap is identically zero and tells you nothing. The script reports the number of off-support outcomes and a $\chi^2$ uniformity statistic instead.

**Bases and edge cases.**
- **Unusable bases.** A base with odd order, or with $a^{r/2} \equiv -1$, cannot yield a factor even from a perfect simulation. Such bases are reported as `base_unusable` and the next base is tried (`--max-a`, default 8). The final `status` separates `factored`, `simulation_ok_all_bases_unusable` and `not_factored`.
- **Inputs handled classically.** Primes, even numbers, perfect powers and $N<4$ are dealt with before any quantum step.
- **Factors.** The returned factors are a nontrivial split of $N$ and are not necessarily prime.

**Limits.**
- **Table memory.** The oracle tables hold $2^{n+1}$ entries for each distinct multiplier.
- **Validation cost.** The true order is computed by brute force, in time $O(N)$.
- **Measured costs.** On one CPU core with the PyPI wheel:

| N | work qubits | seconds per shot | table memory |
|---|---|---|---|
| 3127 | 12 | 0.02 | 2 MB |
| 55687 | 16 | 0.35 | 44 MB |

- **Reproducibility.** `--seed` fixes the bases and the ideal draws. It fixes the measured values only on Qrack builds without a hardware RNG; the PyPI wheel uses RDRAND.

## 3. QFT condensate (`qft-cosmos-classical-condensate.py`)

**Process.** A ring of $W$ qubits is cut into $p$-qubit patches. Each layer does four things:
1. It rotates the patch boundaries by `--shift`.
2. It re-prepares every qubit as $U(\theta,\phi,\lambda)|y\rangle$, where $y$ is that qubit's measured bit from the previous layer.
3. It runs a random QFT or IQFT on every patch.
4. It measures every patch.

**Why it is easy.** Each patch starts from a product state and ends in a $Z$ measurement, which is exactly the semiclassical-QFT setting. Its output distribution is therefore a chain of exact single-qubit conditionals (`chain_sample`), costing $O(p)$ per patch per shot and $O(W \cdot L \cdot \text{shots})$ overall.

**The seams are not what makes it tractable.** A single unseamed patch as wide as the ring is just as cheap:

| run | seconds | peak memory |
|---|---|---|
| `--width 100000 --patch 8 --layers 10 --shots 64` | 5.2 | 307 MB |
| `--width 100000 --patch 100000 --layers 10 --shots 64` | 15.7 | 469 MB |

Both runs used numpy on one CPU core, and PyQrack is never imported unless `--check` is used. The runs demonstrate an exact reference sampler that can be validated at scale. They do not demonstrate large-scale quantum emulation, and they say nothing about patching strategies.

**What is scored.**
- **Unchecked patches.** These are drawn from the same exact model that would score them. Any aggregate score over them measures the entropy of the ideal distribution rather than fidelity, and it comes out the same for every correct sampler. The script therefore reports no score for them.
- **Checked patches.** `--check K` runs $K$ random patches per layer on PyQrack, with one $p$-qubit simulator per shot. It feeds their outcomes into the next layer and reports the mean log-likelihood gap against paired ideal draws, split into clean and faulty shots.
- **Fault injection.** `--fault-rate` applies a random $R_Y$ error before measurement. At $W=96$, $p=8$, 6 layers, 256 shots and $K=3$, clean patches score $z \approx -0.9$ and faulty ones $z \approx -15$.

**Where the interesting physics would be.** With coherent seams (no mid-layer measurement, or entangling operations between layers), entanglement grows with depth. Exact checking is then limited to light-cone spot checks whose reference size grows with depth. That regime is not implemented here.

## 4. Usage
```bash
pip install pyqrack numpy        # an OpenCL runtime is optional; the qubit counts here are small
```
Both scripts point PyQrack at `QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"` when that file exists.

```bash
# Semiclassical Shor, several N, 32 shots per base
python3 qft-condensate-semiclassical.py 15 21 35 143 221 --shots 32 --no-gpu
python3 qft-condensate-semiclassical.py 3127 --shots 16 --max-a 4

# Condensate: exact sampling only (no Qrack, nothing scored)
python3 qft-cosmos-classical-condensate.py --width 100000 --patch 8 --layers 10 --shots 64
# Condensate with Qrack cross-checks and fault injection
python3 qft-cosmos-classical-condensate.py --width 96 --patch 8 --layers 6 --shots 256 --check 3 --fault-rate 0.5 --no-gpu
```

## 5. References
1. R. B. Griffiths, C.-S. Niu (1996). "Semiclassical Fourier transform for quantum computation." *Phys. Rev. Lett.* 76, 3228. [arXiv:quant-ph/9511007](https://arxiv.org/abs/quant-ph/9511007)
2. S. Parker, M. B. Plenio (2000). "Efficient factorization with a single pure qubit and log N mixed qubits." *Phys. Rev. Lett.* 85, 3049. [arXiv:quant-ph/0001066](https://arxiv.org/abs/quant-ph/0001066)
3. S. Beauregard (2003). "Circuit for Shor's algorithm using 2n+3 qubits." [arXiv:quant-ph/0205095](https://arxiv.org/abs/quant-ph/0205095). This is the gate-level contrast to the oracle used here.
4. D. Strano, B. Bollay, A. Blaauw, N. Shammah, W. J. Zeng, A. Mari (2023). "Exact and approximate simulation of large quantum circuits on a single GPU." [arXiv:2304.14969](https://arxiv.org/abs/2304.14969)
