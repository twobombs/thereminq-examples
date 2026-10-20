# -*- coding: us-ascii -*-
# H2 ANCHOR for the K4 Distributed Composite QPE Benchmark
# Validates the information-theory QPE estimator of arXiv:2505.09133
# against the paper's own published number, E_FCI = -1.13731 hartree.
#
# SINGLE FILE. No external modules beyond numpy. PyQrack optional.
#
# ===================================================================
# WHY THIS FILE EXISTS
# ===================================================================
# The K4 benchmark reproduces the paper's ESTIMATOR (Eqs 4-7, 15) but runs
# it on a transverse-field Ising model on the twisted K4 rim graph, not on
# the paper's Hamiltonian. At (3,6) there is no exact reference, so the only
# self-consistency signal is product-vs-bethe divergence -- which says nothing
# about whether the estimator itself is correct.
#
# This file closes that gap. It runs the SAME estimator code path on the
# paper's own 1-qubit H2 Hamiltonian, where the answer is published:
#
#     H = h1 Z + h2 X + h3 I,  (h1,h2,h3) = (0.79605, -0.18092, -0.32096)
#     t = pi/(8 h1),  U^k = (e^{-i h1 Z t} e^{-i h2 X t})^k
#     E_FCI = -1.13731 hartree,  paper's experiment: -1.136(13)
#
# If estimate_energy() returns -1.137 here, the estimator half of the K4
# pipeline is verified against an external number. If it does not, every
# K4 result is suspect regardless of how well the region decomposition
# verifies.
#
# ===================================================================
# THE PHASE CONVENTION, PINNED
# ===================================================================
# The paper drops the h3 I term as a global phase: U approximates
# e^{-i(H - h3 I)kt}. So U has eigenvalues e^{i phi_j} with
#
#     phi_j = -(E_j - h3) t          =>      E_j = -phi_j / t + h3
#
# NOT E = -phi/t. Forgetting the h3 shift costs exactly 0.32096 hartree and
# is the single most likely way to get a plausible-looking wrong answer.
# Verified numerically in selftest(): exact U eigenphase 0.402199 against
# analytic target 0.402713, the 5.1e-4 gap being the Trotter error, which is
# 1.0e-3 hartree and matches the paper's stated ~10^-3 systematic.
#
# ALIASING: |phi| < pi requires |E - h3| t < pi. Here (E-h3)t = 0.403, so
# there is ~7.8x headroom. check_aliasing() is still called for the general
# molecular path where this is NOT automatic.
#
# ===================================================================
# WHAT THIS ADDS TO THE K4 FILE
# ===================================================================
# 1. pauli_exp(): e^{-i theta P} for an ARBITRARY Pauli string P, replacing
#    the ZZ-only trotter_step(). This is the prerequisite for any molecular
#    Hamiltonian, since Jordan-Wigner produces X/Y strings with Z tails.
#
#    On the Qrack path this maps to a SINGLE call: QrackSimulator.exp(b, ph, q)
#    implements e^{i*ph*(b_0 tensor b_1 ...)} natively (PyQrack v2.7.1). No
#    CNOT ladder, no basis-change sandwich. The sign and angle scale are
#    autodetected the same way Rev 311 detects the Pauli codes, because
#    neither is documented.
#
# 2. PauliHamiltonian: a weighted Pauli-string container with exact
#    diagonalization for small n, so any future molecular system gets a
#    reference energy for free.
#
# 3. The anchor itself.
#
# ===================================================================
# WHAT THIS DOES NOT DO
# ===================================================================
# No color code, no Steane gadgets, no recursive gate teleportation. The
# paper's actual contribution is QEC-encoded QPE and none of that is
# simulated here; only the logical-level algorithm is. The decoherence model
# is the phenomenological q(k) = 1 - e^{-ak-b} of Eq. 14 with the paper's
# fitted a = 0.17(2), b = -0.17(9), not the memory/two-qubit channels of
# Table III.
#
# Consequently a match here validates the ESTIMATOR, not the error model.

import os
import sys
import math
import json
import argparse
import numpy as np


# =====================================================================
# PAULI ALGEBRA
# =====================================================================
_SQ2 = 1.0 / math.sqrt(2.0)
_H = np.array([[_SQ2, _SQ2], [_SQ2, -_SQ2]], dtype=np.complex128)
_PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def _rot(pauli, angle):
    """exp(-i * angle * P / 2)."""
    c = math.cos(0.5 * angle)
    s = math.sin(0.5 * angle)
    return c * np.eye(2, dtype=np.complex128) - 1j * s * _PAULI[pauli]


class NumpyStatevector(object):
    """Dense reference simulator, little-endian: bit q of the amplitude
    index is qubit q. Memory 2**n * 16 bytes."""

    def __init__(self, n, dtype=np.complex128):
        self.n = int(n)
        self.dtype = dtype
        self.psi = np.zeros((2,) * self.n, dtype=dtype)
        self.psi[(0,) * self.n] = 1.0

    def _axis(self, q):
        return self.n - 1 - int(q)

    def _apply1(self, u, q):
        ax = self._axis(q)
        self.psi = np.moveaxis(
            np.tensordot(u.astype(self.dtype), self.psi, axes=([1], [ax])),
            0, ax)

    def h(self, q):
        self._apply1(_H, q)

    def x(self, q):
        self._apply1(_PAULI["X"], q)

    def rx(self, a, q):
        self._apply1(_rot("X", a), q)

    def ry(self, a, q):
        self._apply1(_rot("Y", a), q)

    def rz(self, a, q):
        self._apply1(_rot("Z", a), q)

    def pauli_exp(self, theta, term):
        """exp(-i * theta * P) where term maps qubit -> 'X'/'Y'/'Z'.

        Diagonalized directly: P has eigenvalues +-1 and the eigenvalue on a
        computational basis state is the parity of the bits under the Z-part
        after the basis change. Implemented as basis-change / phase / undo so
        that it is manifestly the same operator the Qrack path applies.
        """
        term = dict((int(q), p.upper()) for q, p in term.items()
                    if p.upper() != "I")
        if not term:
            # identity term: pure global phase, drop it (see h3 note above)
            return
        for q, p in term.items():
            if p == "X":
                self.h(q)
            elif p == "Y":
                self.rx(0.5 * math.pi, q)
        qs = sorted(term.keys())
        
        # Parity tracking array built directly to target axes
        par = np.ones([2 if a in [self._axis(q) for q in qs] else 1
                       for a in range(self.n)])
        for q in qs:
            sh = [1] * self.n
            sh[self._axis(q)] = 2
            par = par * np.array([1.0, -1.0]).reshape(sh)
            
        self.psi = self.psi * np.exp(-1j * theta * par)
        
        for q, p in term.items():
            if p == "X":
                self.h(q)
            elif p == "Y":
                self.rx(-0.5 * math.pi, q)

    def expect_pauli(self, term):
        """<P> for a Pauli string given as {qubit: 'X'/'Y'/'Z'}."""
        term = dict((int(q), p.upper()) for q, p in term.items()
                    if p.upper() != "I")
        if not term:
            return 1.0
        out = self.psi
        for q, p in term.items():
            ax = self._axis(q)
            out = np.moveaxis(
                np.tensordot(_PAULI[p].astype(self.dtype), out,
                             axes=([1], [ax])), 0, ax)
        return float(np.real(np.vdot(self.psi, out)))

    def ket(self):
        return self.psi.reshape(-1).copy()

    def set_ket(self, v):
        v = np.asarray(v, dtype=self.dtype).reshape((2,) * self.n)
        nrm = np.linalg.norm(v)
        if nrm <= 0.0:
            raise ValueError("set_ket got a zero vector")
        self.psi = v / nrm


class QrackStatevector(object):
    """PyQrack backend exposing the NumpyStatevector interface.

    Autodetects the Pauli integer codes, the r() angle scale and the exp()
    sign, because none of the three are documented and all three have
    changed across Qrack releases. Rev 311 already did the first two; the
    third is new and is what makes arbitrary Pauli strings a single call.
    """

    def __init__(self, n, gpu=True, device=None):
        if device is not None:
            os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(device)
            os.environ["QRACK_QPAGER_DEVICES"] = str(device)
            os.environ["QRACK_QUNITMULTI_DEVICES"] = str(device)
        from pyqrack import QrackSimulator
        self._QS = QrackSimulator
        self.n = int(n)
        self._detect()
        self.sim = QrackSimulator(qubit_count=self.n,
                                  is_binary_decision_tree=False,
                                  is_stabilizer_hybrid=False, is_gpu=gpu)

    def _detect(self):
        QS = self._QS
        thresh = 0.5

        def find(prep, flip, exclude):
            p = QS(qubit_count=1, is_binary_decision_tree=False)
            prep(p)
            v0 = {}
            for code in range(8):
                if code in exclude:
                    continue
                try:
                    v0[code] = p.pauli_expectation([0], [code])
                except Exception:
                    pass
            flip(p)
            for code, a in v0.items():
                try:
                    b = p.pauli_expectation([0], [code])
                except Exception:
                    continue
                if abs(a) > thresh and abs(b) > thresh and a * b < 0:
                    del p
                    return code, (1.0 if a > 0 else -1.0)
            del p
            return None, None

        self.PZ, self.SZ = find(lambda p: None, lambda p: p.x(0), set())
        if self.PZ is None:
            raise RuntimeError("Fatal: could not autodetect PZ code")
        self.PX, self.SX = find(lambda p: p.h(0), lambda p: p.z(0), {self.PZ})
        if self.PX is None:
            raise RuntimeError("Fatal: could not autodetect PX code")
        c, s = math.cos(math.pi / 4.0), math.sin(math.pi / 4.0)
        self.PY, self.SY = find(
            lambda p: p.mtrx([complex(c, 0), complex(0, s),
                              complex(0, s), complex(c, 0)], 0),
            lambda p: p.z(0), {self.PX, self.PZ})
        if self.PY is None:
            raise RuntimeError("Fatal: could not autodetect PY code")

        m = QS(qubit_count=1, is_binary_decision_tree=False)
        m.r(self.PX, math.pi, 0)
        corr = self.SZ * m.pauli_expectation([0], [self.PZ])
        if abs(corr + 1.0) < 0.1:
            self.ANGLE_SCALE = 1.0
        elif abs(corr - 1.0) < 0.1:
            self.ANGLE_SCALE = 0.5
        else:
            raise RuntimeError(
                "Fatal: r(PX,pi) gave ambiguous SIGN_Z*<Z> = %.6f" % corr)
        del m

        self._code = {"X": self.PX, "Y": self.PY, "Z": self.PZ}
        self._sign = {"X": self.SX, "Y": self.SY, "Z": self.SZ}
        self._detect_exp()

    def _detect_exp(self):
        """Pin the sign and scale of exp(b, ph, q).

        Documented as e^{i*ph*(b_0 tensor b_1 ...)}, but neither the sign nor
        whether ph is the exponent coefficient or a half-angle is guaranteed.
        Probe: prepare |+>, apply the candidate for e^{-i(pi/4) Z}, and read
        <Y>. For |+>, e^{-i a Z} gives <Y> = -sin(2a), so a = pi/4 must give
        <Y> = -1. Fall back to the CNOT/RZ decomposition if exp() is absent
        or behaves unexpectedly rather than silently producing wrong physics.
        """
        self.EXP_OK = False
        self.EXP_SIGN = -1.0
        self.EXP_SCALE = 1.0
        QS = self._QS
        for sign in (-1.0, 1.0):
            for scale in (1.0, 0.5, 2.0):
                try:
                    p = QS(qubit_count=1, is_binary_decision_tree=False)
                    p.h(0)
                    p.exp([self.PZ], sign * scale * (math.pi / 4.0), [0])
                    y = self._sign["Y"] * p.pauli_expectation([0], [self.PY])
                    del p
                except Exception:
                    return
                if abs(y + 1.0) < 1e-3:
                    self.EXP_OK = True
                    self.EXP_SIGN = sign
                    self.EXP_SCALE = scale
                    return

    def h(self, q):
        self.sim.h(int(q))

    def x(self, q):
        self.sim.x(int(q))

    def _r(self, pauli, a, q):
        self.sim.r(self._code[pauli], float(a) * self.ANGLE_SCALE, int(q))

    def rx(self, a, q):
        self._r("X", a, q)

    def ry(self, a, q):
        self._r("Y", a, q)

    def rz(self, a, q):
        self._r("Z", a, q)

    def pauli_exp(self, theta, term):
        """exp(-i * theta * P). One native call when exp() is trustworthy,
        else the CNOT-ladder decomposition."""
        term = dict((int(q), p.upper()) for q, p in term.items()
                    if p.upper() != "I")
        if not term:
            return
        qs = sorted(term.keys())
        if self.EXP_OK:
            self.sim.exp([self._code[term[q]] for q in qs],
                         self.EXP_SIGN * self.EXP_SCALE * float(theta),
                         [int(q) for q in qs])
            return
        for q in qs:
            if term[q] == "X":
                self.sim.h(int(q))
            elif term[q] == "Y":
                self.rx(0.5 * math.pi, q)
        for i in range(len(qs) - 1):
            self.sim.mcx([int(qs[i])], int(qs[i + 1]))
        self.rz(2.0 * float(theta), qs[-1])
        for i in range(len(qs) - 2, -1, -1):
            self.sim.mcx([int(qs[i])], int(qs[i + 1]))
        for q in qs:
            if term[q] == "X":
                self.sim.h(int(q))
            elif term[q] == "Y":
                self.rx(-0.5 * math.pi, q)

    def expect_pauli(self, term):
        term = dict((int(q), p.upper()) for q, p in term.items()
                    if p.upper() != "I")
        if not term:
            return 1.0
        qs = sorted(term.keys())
        return float(self.sim.pauli_expectation(
            [int(q) for q in qs], [self._code[term[q]] for q in qs]))

    def ket(self):
        return np.asarray(self.sim.out_ket(), dtype=np.complex128)

    def set_ket(self, v):
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        nrm = np.linalg.norm(v)
        if nrm <= 0.0:
            raise ValueError("set_ket got a zero vector")
        self.sim.in_ket((v / nrm).tolist())


def make_backend(n, kind="numpy", **kw):
    if kind == "numpy":
        return NumpyStatevector(n, **kw)
    if kind == "qrack":
        return QrackStatevector(n, **kw)
    raise ValueError("unknown backend %r" % (kind,))


# =====================================================================
# WEIGHTED PAULI HAMILTONIAN
# =====================================================================
class PauliHamiltonian(object):
    """H = sum_a coeff_a * P_a, each P_a a dict {qubit: 'X'/'Y'/'Z'}.

    The identity term is kept in .identity and excluded from .terms, because
    it is a pure global phase under time evolution and must be added back by
    hand at readout. This is exactly the paper's h3.
    """

    def __init__(self, n_qubits, terms, name=""):
        self.n = int(n_qubits)
        self.name = name
        self.terms = []
        self.identity = 0.0
        for coeff, term in terms:
            t = dict((int(q), p.upper()) for q, p in dict(term).items()
                     if p.upper() != "I")
            if not t:
                self.identity += float(coeff)
            else:
                self.terms.append((float(coeff), t))

    def matrix(self):
        """Dense 2**n matrix. Only sane for n <= ~14."""
        d = 2 ** self.n
        if self.n > 14:
            raise RuntimeError("refusing to densify %d qubits" % self.n)
        m = self.identity * np.eye(d, dtype=np.complex128)
        for coeff, term in self.terms:
            op = np.array([[1.0]], dtype=np.complex128)
            # qubit 0 is least significant, so build most-significant first
            for q in range(self.n - 1, -1, -1):
                op = np.kron(op, _PAULI[term.get(q, "I")])
            m = m + coeff * op
        return m

    def eigenvalues(self):
        return np.linalg.eigvalsh(self.matrix())

    def ground_energy(self):
        return float(self.eigenvalues()[0])

    def expectation(self, sim):
        e = self.identity
        for coeff, term in self.terms:
            e += coeff * sim.expect_pauli(term)
        return float(e)

    def energy_scale_bound(self):
        return sum(abs(c) for c, _ in self.terms)

    def summary(self):
        return {"name": self.name, "n_qubits": self.n,
                "n_terms": len(self.terms), "identity": self.identity,
                "l1_norm": self.energy_scale_bound()}


def trotter_first_order(sim, ham, dt, order=None):
    """U ~ prod_a exp(-i coeff_a P_a dt), first order Lie-Trotter.

    Order is the term order as given unless overridden. The identity term is
    NOT applied: it is a global phase and is added back at readout.
    """
    seq = ham.terms if order is None else [ham.terms[i] for i in order]
    for coeff, term in seq:
        sim.pauli_exp(coeff * dt, term)


# =====================================================================
# THE PAPER'S H2 SYSTEM (arXiv:2505.09133 Sec. II A, Eq 1)
# =====================================================================
H2_H1 = 0.79605
H2_H2 = -0.18092
H2_H3 = -0.32096
H2_EFCI = -1.13731


def h2_hamiltonian():
    """H = h1 Z + h2 X + h3 I, minimal basis H2 at equilibrium geometry.

    Coefficients from InQuanto v4.2 / PySCF v2.4 Hartree-Fock, quoted in the
    paper. Exact ground state is -1.1373102 hartree, matching the paper's
    stated E_FCI = -1.13731.
    """
    return PauliHamiltonian(1, [(H2_H1, {0: "Z"}),
                                (H2_H2, {0: "X"}),
                                (H2_H3, {})], name="H2/STO-3G")


def h2_evolution_time():
    """t = pi/(8 h1), chosen so that e^{-i h1 Z t} is exactly the T gate."""
    return math.pi / (8.0 * H2_H1)


def truncate_binary_fraction(x, nbits=None):
    """Round x to nbits FRACTIONAL binary bits: round(x * 2**nbits) / 2**nbits.

    WARNING -- the paper's convention is ambiguous and the literal reading is
    a trap. Sec. II A says h2*t/pi is approximated by a 5-bit binary fraction
    with theta/pi = b0 + b1/2 + ... + b_nb/2**(nb-1), i.e. resolution 2**-4 =
    1/16. (Note: this corresponds exactly to `nbits=4` here, as this parameter 
    counts only fractional bits. E.g., truncate_binary_fraction(-0.0284, nbits=4) == 0.0).
    But h2*t/pi = -0.0284, which ROUNDS TO EXACTLY ZERO at that resolution. 
    U then degenerates to e^{-i h1 Z t} = T, of which the Hartree-Fock state 
    |1> is an exact eigenstate, and the whole benchmark silently reports the 
    wrong energy with perfect confidence.

    The diagnostic is unmistakable: |a_k| = 1.0000 for every k and arg(a_k)/k
    is flat at exactly pi/8 = 0.392699. If you see that, the X term has been
    truncated away, not simulated.

    Their quoted ~1e-3 hartree systematic is reproduced by the Trotter error
    alone (measured here: 1.04e-3 hartree), so the default is nbits=None.
    Reproducing their exact truncation would need ~10 fractional bits, which
    suggests the 5 bits are counted in some other normalization.
    """
    if nbits is None:
        return x
    scale = float(2 ** int(nbits))
    return round(x * scale) / scale


def h2_step(sim, t=None, nbits=None):
    """One application of U = e^{-i h1 Z t} e^{-i h2 X t}, Eq. 8.

    Rightmost factor acts first, so the X rotation is applied before the Z
    rotation. pauli_exp(theta, P) is exp(-i theta P), so the coefficients go
    in directly with no factor of two.
    """
    if t is None:
        t = h2_evolution_time()
    a2 = truncate_binary_fraction(H2_H2 * t / math.pi, nbits) * math.pi
    a1 = H2_H1 * t
    sim.pauli_exp(a2, {0: "X"})
    sim.pauli_exp(a1, {0: "Z"})


def h2_input_state(sim, alpha0=0.0, alpha1=0.0):
    """|Phi(a0,a1)> = RZ(a1) RY(a0) |1>, Eq. 9.

    (0,0) is the Hartree-Fock state |1>, used for the energy estimation.
    (-0.274220, -0.785398) is the calibration state, which approximates the
    true ground state up to Trotter error.
    """
    sim.x(0)
    sim.ry(alpha0, 0)
    sim.rz(alpha1, 0)


# =====================================================================
# LOSCHMIDT AMPLITUDE
# =====================================================================
def loschmidt_series(n, step_fn, prep_fn, kmax, backend="numpy", **bk):
    """a_k = <Phi|U^k|Phi> for k = 1..kmax.

    step_fn(sim) applies one U. prep_fn(sim) prepares |Phi> from |0...0>.
    Everything the QPE needs is a function of this array alone -- there is no
    ancilla to simulate and no controlled-U, which is the whole reduction the
    K4 file is built on.
    """
    sim = make_backend(n, backend, **bk)
    prep_fn(sim)
    psi0 = sim.ket().copy()
    out = np.zeros(kmax, dtype=complex)
    for k in range(kmax):
        step_fn(sim)
        out[k] = complex(np.vdot(psi0, sim.ket()))
    return out


# =====================================================================
# INFORMATION-THEORY QPE (Eqs 4-7, 15)
# =====================================================================
def sample_outcomes(a_k, kmax, ns, rng, decoh_a=0.0, decoh_b=0.0):
    """P(m|k,beta) = (1 + (-1)^m Re[e^{i beta} a_k]) / 2, Eq. 4, optionally
    damped by the paper's e^{-ak-b} decoherence model."""
    ks = rng.integers(1, kmax + 1, size=ns)
    betas = rng.choice([0.0, math.pi / 2.0], size=ns)
    damp = np.exp(-decoh_a * ks - decoh_b) if (decoh_a or decoh_b) else 1.0
    amp = a_k[ks - 1]
    p0 = 0.5 * (1.0 + damp * np.real(np.exp(1j * betas) * amp))
    p0 = np.clip(p0, 0.0, 1.0)
    ms = (rng.random(ns) > p0).astype(int)
    return ks, betas, ms


def phase_posterior(ks, betas, ms, grid, decoh_a=0.0, decoh_b=0.0):
    """log Q(phi), Eq. 6 plain or Eq. 15 noise-aware."""
    phi = grid[None, :]
    k = ks[:, None]
    arg = k * phi + betas[:, None] - ms[:, None] * math.pi
    damp = (np.exp(-decoh_a * k - decoh_b) if (decoh_a or decoh_b) else 1.0)
    q = 0.5 * (1.0 + damp * np.cos(arg))
    return np.log(np.clip(q, 1e-300, None)).sum(axis=0)


def estimate_phase(a_k, kmax, ns, rng, grid_n=4001, decoh_a=0.0, decoh_b=0.0):
    grid = np.linspace(0.0, 2.0 * math.pi, grid_n, endpoint=False)
    ks, betas, ms = sample_outcomes(a_k, kmax, ns, rng, decoh_a, decoh_b)
    lq = phase_posterior(ks, betas, ms, grid, decoh_a, decoh_b)
    return float(grid[int(np.argmax(lq))])


def estimate_phase_circular(a_k, kmax, ns, rng, repeats=1000, **kw):
    """Circular mean over bootstrap repeats, plus Holevo variance, Sec III B.

    Returns (phase in (-pi,pi], holevo std). The paper uses R = 1000.
    """
    phis = np.array([estimate_phase(a_k, kmax, ns, rng, **kw)
                     for _ in range(repeats)])
    r = np.mean(np.exp(1j * phis))
    mean = float(np.angle(r))
    holevo = 1.0 / max(abs(r) ** 2, 1e-300) - 1.0
    return mean, math.sqrt(max(holevo, 0.0))


def energy_from_phase(phi, t, identity=0.0):
    """E = -phi/t + identity.

    The identity shift is the paper's h3 and is NOT optional: U approximates
    e^{-i(H - h3 I)t}, so the eigenphase carries no information about h3 and
    it has to be added back classically.
    """
    return -phi / t + identity


def energy_from_slope(a_k, t, identity=0.0):
    """Sampling-free readout: unwrap arg(a_k) and fit the slope. Separates
    Trotter/boundary error from shot noise. Unwrapping means this survives
    past the aliasing limit where the posterior readout does not."""
    kk = np.arange(1, len(a_k) + 1) * t
    slope = float(np.polyfit(kk, np.unwrap(np.angle(a_k)), 1)[0])
    return -slope + identity


def check_aliasing(energy_scale, t, label=""):
    """The phase readout wraps at |E - identity| * t = pi."""
    t_max = math.pi / max(abs(energy_scale), 1e-12)
    if t > t_max:
        print("[ALIAS] %st=%.4f exceeds pi/|E|=%.4f; the posterior readout "
              "will resolve the wrong branch (the slope readout still "
              "unwraps)." % (label + " " if label else "", t, t_max),
              file=sys.stderr)
    return t_max


# =====================================================================
# THE ANCHOR
# =====================================================================
def run_h2_anchor(kmax=12, ns=2186, repeats=1000, seed=1337,
                  backend="numpy", nbits=None, noisy=True,
                  calibration_state=False, grid_n=4001):
    """Reproduce the paper's ground-state energy estimate.

    Defaults are the paper's own: kmax = 12, Ns = 2186 (post-discard),
    R = 1000, decoherence a = 0.17, b = -0.17 from their Sec III A fit.
    """
    ham = h2_hamiltonian()
    t = h2_evolution_time()
    rng = np.random.default_rng(seed)
    e_exact = ham.ground_energy()

    print("[H2] %s" % json.dumps(ham.summary()))
    print("[H2] t = pi/(8 h1) = %.6f   h1*t/pi = %.4f   h2*t/pi = %.6f"
          % (t, H2_H1 * t / math.pi, H2_H2 * t / math.pi))
    print("[H2] exact ground energy      %+.6f hartree "
          "(paper E_FCI = %.5f)" % (e_exact, H2_EFCI))
    print("[H2] target eigenphase        %+.6f rad"
          % (-(e_exact - ham.identity) * t))
    check_aliasing(abs(e_exact - ham.identity), t, "H2")

    a0, a1 = ((-0.274220, -0.785398) if calibration_state else (0.0, 0.0))
    prep = lambda s: h2_input_state(s, a0, a1)
    step = lambda s: h2_step(s, t, nbits)

    a_k = loschmidt_series(1, step, prep, kmax, backend=backend)

    print("\n  k   |a_k|      arg(a_k)   arg/k (should be flat at phi0)")
    ph = np.unwrap(np.angle(a_k))
    for k in range(kmax):
        print("  %2d   %7.4f   %+9.5f   %+9.5f"
              % (k + 1, abs(a_k[k]), ph[k], ph[k] / (k + 1)))

    e_slope = energy_from_slope(a_k, t, ham.identity)

    da, db = ((0.17, -0.17) if noisy else (0.0, 0.0))
    phi, sig = estimate_phase_circular(a_k, kmax, ns, rng, repeats=repeats,
                                       grid_n=grid_n,
                                       decoh_a=da, decoh_b=db)
    e_qpe = energy_from_phase(phi, t, ham.identity)
    e_sig = sig / t

    print("\n%-26s %12s %12s" % ("quantity", "hartree", "err vs exact"))
    print("%-26s %+12.5f %12s" % ("exact (diagonalization)", e_exact, "--"))
    print("%-26s %+12.5f %+12.5f" % ("slope readout, no shots",
                                     e_slope, e_slope - e_exact))
    print("%-26s %+12.5f %+12.5f" % ("QPE posterior + circular",
                                     e_qpe, e_qpe - e_exact))
    print("%-26s %12.5f %12s" % ("  statistical sigma", e_sig, ""))
    print("%-26s %+12.5f %12s" % ("paper, hardware H2-2", -1.136, "0.001(13)"))

    if abs(abs(a_k).min() - 1.0) < 1e-9:
        print("\n[WARN] |a_k| == 1 for every k: |Phi> is an exact eigenstate "
              "of U. If nbits is set, the X term was almost certainly "
              "truncated to zero -- see truncate_binary_fraction().",
              file=sys.stderr)

    sys_err = e_slope - e_exact
    shot_err = e_qpe - e_slope
    print("\ninterpretation:")
    print("  Trotter + %s systematic      %+8.5f hartree"
          % (("%d-bit" % nbits) if nbits else "no truncation", sys_err))
    print("  shot noise on top of that     %+8.5f hartree" % shot_err)
    print("  chemical precision            %8.5f hartree" % 0.0016)
    ok_sys = abs(sys_err) < 5e-3
    ok_shot = abs(shot_err) < max(3.0 * e_sig, 1e-4)
    print("  systematic under 5e-3:        %s (%s)"
          % ("YES" if ok_sys else "NO", "Trotter only" if not nbits
             else "Trotter + truncation"))
    print("  shot error within 3 sigma:    %s" % ("YES" if ok_shot else "NO"))
    ok = ok_sys and ok_shot
    return {"exact": e_exact, "slope": e_slope, "qpe": e_qpe, "sigma": e_sig,
            "a_k": a_k, "t": t, "ok": ok}


# =====================================================================
# SELF-TEST
# =====================================================================
def selftest():
    print("pauli_exp against dense matrix exponentials:")
    rng = np.random.default_rng(7)
    worst = 0.0
    for n, term in [(1, {0: "X"}), (1, {0: "Y"}), (1, {0: "Z"}),
                    (2, {0: "Z", 1: "Z"}), (2, {0: "X", 1: "Y"}),
                    (3, {0: "X", 1: "Y", 2: "Z"}),
                    (4, {0: "Y", 2: "X", 3: "Y"})]:
        theta = float(rng.normal())
        v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        
        ref = NumpyStatevector(n)
        ref.set_ket(v)
        op = np.array([[1.0]], dtype=np.complex128)
        for q in range(n - 1, -1, -1):
            op = np.kron(op, _PAULI[term.get(q, "I")])
            
        s2 = NumpyStatevector(n)
        s2.set_ket(v)
        s2.pauli_exp(theta, term)
        # exp(-i theta P) = cos(theta) I - i sin(theta) P  since P**2 = I
        u = (math.cos(theta) * np.eye(2 ** n) - 1j * math.sin(theta) * op)
        want = u @ ref.ket()
        worst = max(worst, float(np.abs(s2.ket() - want).max()))
    print("  max |pauli_exp - closed form| = %.2e" % worst)
    assert worst < 1e-12, "pauli_exp is wrong"

    print("expect_pauli against dense quadratic form:")
    worst = 0.0
    for n, term in [(2, {0: "Z", 1: "Z"}), (3, {0: "X", 2: "Y"})]:
        s = NumpyStatevector(n)
        v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        s.set_ket(v)
        op = np.array([[1.0]], dtype=np.complex128)
        for q in range(n - 1, -1, -1):
            op = np.kron(op, _PAULI[term.get(q, "I")])
        want = float(np.real(np.vdot(s.ket(), op @ s.ket())))
        worst = max(worst, abs(s.expect_pauli(term) - want))
    print("  max |expect_pauli - reference| = %.2e" % worst)
    assert worst < 1e-12

    print("H2 Hamiltonian spectrum:")
    ham = h2_hamiltonian()
    ev = ham.eigenvalues()
    print("  eigenvalues %s" % np.array2string(ev, precision=7))
    print("  ground %.7f vs paper E_FCI %.5f" % (ev[0], H2_EFCI))
    assert abs(ev[0] - H2_EFCI) < 1e-4, "H2 ground energy does not match paper"

    print("U eigenphase against the analytic target:")
    t = h2_evolution_time()
    cols = []
    for b in range(2):
        s = NumpyStatevector(1)
        if b:
            s.x(0)
        h2_step(s, t, nbits=None)
        cols.append(s.ket())
    u = np.array(cols).T
    w = np.linalg.eigvals(u)
    
    # We rely on the traceless nature of h1*Z + h2*X here, meaning 
    # the two eigenphases are +phi_0 and -phi_0. np.max picks the positive 
    # branch corresponding to the ground state phase.
    got = float(np.max(np.angle(w)))
    want = -(ham.ground_energy() - ham.identity) * t
    print("  exact-angle U eigenphase %+.6f  analytic target %+.6f  "
          "(Trotter %.2e rad = %.2e hartree)"
          % (got, want, abs(got - want), abs(got - want) / t))
    assert abs(got - want) < 1e-2

    print("identity term must NOT enter the phase:")
    bare = PauliHamiltonian(1, [(H2_H1, {0: "Z"}), (H2_H2, {0: "X"})])
    print("  h3 = %+.5f hartree is added back classically; omitting it "
          "would report %+.5f instead of %+.5f"
          % (ham.identity, ham.ground_energy() - ham.identity,
             ham.ground_energy()))
    assert abs(bare.ground_energy() - (ham.ground_energy() - ham.identity)) \
        < 1e-12

    print("\nrunning the anchor (noiseless, then with the paper's a/b):")
    r1 = run_h2_anchor(noisy=False, repeats=200, seed=11)
    print()
    r2 = run_h2_anchor(noisy=True, repeats=200, seed=11)
    assert r1["ok"] and r2["ok"], "anchor did not land within 3 sigma"
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="H2 anchor for the K4 distributed composite QPE benchmark")
    sub = ap.add_subparsers(dest="cmd")

    a = sub.add_parser("anchor", help="reproduce the paper's energy estimate")
    a.add_argument("--kmax", type=int, default=12)
    a.add_argument("--ns", type=int, default=2186)
    a.add_argument("--repeats", type=int, default=1000)
    a.add_argument("--seed", type=int, default=1337)
    a.add_argument("--backend", default="numpy")
    a.add_argument("--nbits", type=int, default=0,
                   help="binary-fraction bits for h2*t/pi; 0 disables")
    a.add_argument("--no-noise", action="store_true",
                   help="drop the paper's a=0.17 b=-0.17 decoherence model")
    a.add_argument("--calibration-state", action="store_true",
                   help="use the near-eigenstate instead of Hartree-Fock")

    s = sub.add_parser("selftest", help="verify every claim in this file")

    ns = ap.parse_args()
    if ns.cmd == "anchor":
        run_h2_anchor(kmax=ns.kmax, ns=ns.ns, repeats=ns.repeats,
                      seed=ns.seed, backend=ns.backend,
                      nbits=(ns.nbits if ns.nbits else None),
                      noisy=not ns.no_noise,
                      calibration_state=ns.calibration_state)
    elif ns.cmd == "selftest":
        selftest()
    else:
        ap.print_help()
