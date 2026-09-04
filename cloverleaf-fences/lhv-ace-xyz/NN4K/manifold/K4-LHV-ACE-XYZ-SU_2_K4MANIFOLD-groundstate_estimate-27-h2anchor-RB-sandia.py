# -*- coding: us-ascii -*-
# Rev 28.
# H2 ANCHOR + SU(2) SYNTHETIC RB for the K4 Distributed Composite QPE
# Benchmark.
#
# Two anchors, one file:
#
#   ESTIMATOR  Validates the information-theory QPE estimator of
#              arXiv:2505.09133 against that paper's own published
#              number, E_FCI = -1.13731 hartree.   [cmd: anchor]
#
#   ERROR MODEL  Validates and then REPLACES the phenomenological
#              decoherence model q(k) = 1 - e^{-ak-b} with a measured
#              SU(2) gate fidelity, using the synthetic randomized
#              benchmarking of Fan, Murray, Ladd, Young & Blume-Kohout,
#              Quantum Sci. Technol. 11 (2026) 035052. That half is
#              anchored against the paper's Tables 1-2, Eq. (48) and the
#              appendix J rate table.   [cmds: rb-tables, rb-rates,
#              rb-run, k4-rb]
#
# Rev 27 verified the estimator and left a = 0.17(2), b = -0.17(9) as
# imported hardware-fit constants with no derivation. Rev 28 closes that:
# k4-rb benchmarks the Trotter step as an SU(2) channel (infidelity
# 7.90e-4), converts f_1 into a = -ln f_1, and shows that the fitted
# a = 0.17 is asserting a 7.8% per-step infidelity -- 99x larger. The
# systematic and the decoherence are separate budgets.
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
        par = np.ones((1,) * self.n, dtype=float)
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
        old_env = {}
        if device is not None:
            for k in ["QRACK_OCL_DEFAULT_DEVICE", "QRACK_QPAGER_DEVICES", "QRACK_QUNITMULTI_DEVICES"]:
                old_env[k] = os.environ.get(k)
                os.environ[k] = str(device)
        from pyqrack import QrackSimulator
        self._QS = QrackSimulator
        self.n = int(n)
        self._detect()
        self.sim = QrackSimulator(qubit_count=self.n,
                                  is_binary_decision_tree=False,
                                  is_stabilizer_hybrid=False, is_gpu=gpu)
        if device is not None:
            for k, v in old_env.items():
                if v is None:
                    if k in os.environ:
                        del os.environ[k]
                else:
                    os.environ[k] = v

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
                    
        print("[WARN] QrackSimulator.exp() probe completed but no valid sign/scale matched. Falling back to CNOT decomposition.", file=sys.stderr)

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
    past the aliasing limit where the posterior readout does not.
    Unwrapping can fail to detect wraps for very short arrays (e.g., kmax < 3), 
    but succeeds for the target kmax=12.
    """
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
                  calibration_state=False, grid_n=4001, decoh=None):
    """Reproduce the paper's ground-state energy estimate.

    Defaults are the paper's own: kmax = 12, Ns = 2186 (post-discard),
    R = 1000, decoherence a = 0.17, b = -0.17 from their Sec III A fit.
    """
    ham = h2_hamiltonian()
    t = h2_evolution_time()
    
    if nbits is not None and abs(truncate_binary_fraction(H2_H2 * t / math.pi, nbits)) < 1e-12:
        print("\n[WARN] PRE-FLIGHT: nbits=%d truncates the X term exactly to zero. U degenerates to Z rotation." % nbits, file=sys.stderr)
        
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

    if decoh is not None:
        da, db = (float(decoh[0]), float(decoh[1]))
        print("\n[H2] decoherence model a = %.6f  b = %.6f  "
              "(supplied, e.g. by SU(2) synthetic RB)" % (da, db))
    else:
        da, db = ((0.17, -0.17) if noisy else (0.0, 0.0))
        if noisy:
            print("\n[H2] decoherence model a = %.4f  b = %.4f  "
                  "(the QPE paper's HARDWARE FIT, not derived; run "
                  "'k4-rb' to replace it with a benchmarked number)"
                  % (da, db))
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
# SU(2) SYNTHETIC RANDOMIZED BENCHMARKING
# Fan, Murray, Ladd, Young, Blume-Kohout,
# Quantum Sci. Technol. 11 (2026) 035052, doi:10.1088/2058-9565/ae8b56
# =====================================================================
# WHY THIS IS IN THE K4 FILE
# ---------------------------------------------------------------------
# The anchor above verifies the ESTIMATOR. It says nothing about the
# ERROR MODEL, which is imported wholesale from the QPE paper as the
# phenomenological q(k) = 1 - e^{-ak-b} with fitted a = 0.17(2),
# b = -0.17(9). Those two numbers are hardware-fit constants with no
# derivation, and every noisy K4 number inherits them.
#
# Fan et al give the missing half: a benchmarking protocol whose
# benchmarking group is SU(2) itself -- exactly the group the K4 patches
# rotate under -- and which returns, per irrep k of the superoperator
# representation, a quality parameter f_k and an SU(2)-invariant error
# rate p_k. For a spin-j patch the superoperator rep decomposes as
# j (x) j* = 0 + 1 + ... + 2j, so p_k is the rate of random weight-k
# SU(2) errors: k=0 no error, k=1 dipole, k=2 quadrupole, and so on.
#
# Two things this buys the K4 pipeline:
#
#   1. a and b stop being fitted. The Loschmidt interference term of a
#      qubit (j=1/2) patch damps by exactly the k=1 quality parameter
#      per step, so a = -ln f_1 and b = -ln A_1 with A_1 the SPAM
#      coefficient. rb_to_decoherence() / decoherence_to_rb() are the
#      two directions of that bridge.
#
#   2. For j > 1/2 patches the single number a is replaced by the whole
#      spectrum p_k. Spin codes correct weight k = 1 and 2 only, so
#      sum_{k<=2} p_k is the fraction of the error budget that the code
#      can actually absorb -- a figure of merit the scalar decoherence
#      model cannot express at all.
#
# WHY *SYNTHETIC* RB AND NOT ORDINARY CHARACTER RB
# ---------------------------------------------------------------------
# SU(2) on a j > 1/2 system is not a 2-design, so its superoperator rep
# is highly reducible with irreps of dimension up to 4j+1, and plain
# character RB pays a variance that grows with (dim phi_k)^2. The paper's
# fix is to replace physical SPAM with SYNTHETIC SPAM: prepare all 2j+1
# J_z eigenstates, measure J_z, and take a classical linear combination
# of the survival probabilities with the coefficients M_{k,l} of (29).
# That linear combination behaves as if we had prepared and measured the
# unphysical diagonal spherical tensor operator T^(k)_0, which lies
# entirely inside irrep k. Three protocols are implemented:
#
#   ssrb   pure synthetic SPAM. Zero variance at zero noise, but not
#          SPAM-robust: SPAM error leaks signal between irreps.
#   sschi  synthetic SPAM + character weighting (2k+1) chi_k(g).
#          SPAM-robust, highest variance.
#   ssr1   synthetic SPAM + rank-1 weighting (2k+1) d^k_00(g).
#          SPAM-robust and, per the paper, the best of the three: one to
#          two orders of magnitude fewer shots than character RB.
#
# WHAT IS VERIFIED AGAINST THE PAPER (see rb_selftest)
# ---------------------------------------------------------------------
#   * M and F for j = 7/2 against the printed matrices (I1), (I2).
#   * Zero-noise estimator variances against Tables 1 and 2, all four
#     protocols, to every printed digit.
#   * SU(2) error rates of the coherent exp(-i g J_z^2) channel against
#     (48), and of the energy-dependent dephasing channel (J3) against
#     the appendix J table.
#   * The projector identity (12)/(G16): the character weighting really
#     does synthesize Pi_k.
#
# NOT DONE HERE: finite frames (FFRB). The paper leaves the frame
# optimization open and declines to compare it quantitatively; so do we.


# ---------------------------------------------------------------------
# ANGULAR MOMENTUM RECOUPLING
# ---------------------------------------------------------------------
def _ifact(x):
    """Exact integer factorial.
    Relies on IEEE 754 exactness for small half-integer sums (safe for j <= 15).
    """
    n = int(round(x))
    if abs(x - n) > 1e-9 or n < 0:
        raise ValueError("factorial of %r" % (x,))
    return math.factorial(n)


def _is_int(x):
    return abs(x - round(x)) < 1e-9


def clebsch_gordan(j1, m1, j2, m2, j3, m3):
    """<j1 m1; j2 m2 | j3 m3>, Racah's closed form, exact integer
    factorials. Half-integer j are fine; every factorial argument is an
    integer whenever the triangle and projection rules hold."""
    if abs(m1 + m2 - m3) > 1e-9:
        return 0.0
    if j3 < abs(j1 - j2) - 1e-9 or j3 > j1 + j2 + 1e-9:
        return 0.0
    for j, m in ((j1, m1), (j2, m2), (j3, m3)):
        if abs(m) > j + 1e-9 or not _is_int(j - m):
            return 0.0
    pref = math.sqrt(
        (2 * j3 + 1)
        * _ifact(j3 + j1 - j2) * _ifact(j3 - j1 + j2) * _ifact(j1 + j2 - j3)
        / _ifact(j1 + j2 + j3 + 1))
    pref *= math.sqrt(
        _ifact(j3 + m3) * _ifact(j3 - m3)
        * _ifact(j1 - m1) * _ifact(j1 + m1)
        * _ifact(j2 - m2) * _ifact(j2 + m2))
    zmin = int(round(max(0.0, -(j3 - j2 + m1), -(j3 - j1 - m2))))
    zmax = int(round(min(j1 + j2 - j3, j1 - m1, j2 + m2)))
    total = 0.0
    for z in range(zmin, zmax + 1):
        total += (-1.0) ** z / (
            _ifact(z) * _ifact(j1 + j2 - j3 - z) * _ifact(j1 - m1 - z)
            * _ifact(j2 + m2 - z) * _ifact(j3 - j2 + m1 + z)
            * _ifact(j3 - j1 - m2 + z))
    return pref * total


def _delta_tri(a, b, c):
    return math.sqrt(_ifact(a + b - c) * _ifact(a - b + c)
                     * _ifact(-a + b + c) / _ifact(a + b + c + 1))


def _tri_ok(a, b, c):
    return (c >= abs(a - b) - 1e-9 and c <= a + b + 1e-9
            and _is_int(a + b + c))


def wigner_6j(j1, j2, j3, j4, j5, j6):
    """{j1 j2 j3; j4 j5 j6}, Racah. Needed only for the SU(2) 'Fourier'
    matrix F of (B13), which converts quality parameters to error rates."""
    for tr in ((j1, j2, j3), (j1, j5, j6), (j4, j2, j6), (j4, j5, j3)):
        if not _tri_ok(*tr):
            return 0.0
    pref = (_delta_tri(j1, j2, j3) * _delta_tri(j1, j5, j6)
            * _delta_tri(j4, j2, j6) * _delta_tri(j4, j5, j3))
    tmin = int(round(max(j1 + j2 + j3, j1 + j5 + j6,
                         j4 + j2 + j6, j4 + j5 + j3)))
    tmax = int(round(min(j1 + j2 + j4 + j5, j2 + j3 + j5 + j6,
                         j1 + j3 + j4 + j6)))
    total = 0.0
    for t in range(tmin, tmax + 1):
        total += (-1.0) ** t * _ifact(t + 1) / (
            _ifact(t - j1 - j2 - j3) * _ifact(t - j1 - j5 - j6)
            * _ifact(t - j4 - j2 - j6) * _ifact(t - j4 - j5 - j3)
            * _ifact(j1 + j2 + j4 + j5 - t) * _ifact(j2 + j3 + j5 + j6 - t)
            * _ifact(j1 + j3 + j4 + j6 - t))
    return pref * total


def su2_spin_ops(j):
    """Jx, Jy, Jz and the m values, in the DESCENDING basis
    m = j, j-1, ..., -j. For j = 1/2 this is exactly the computational
    basis of the rest of this file: index 0 is |0>, index 1 is |1>, and
    2*Jz is the Pauli Z of _PAULI."""
    d = int(round(2 * j)) + 1
    ms = np.array([j - i for i in range(d)], dtype=float)
    jp = np.zeros((d, d), dtype=np.complex128)
    for i in range(1, d):
        m = ms[i]
        jp[i - 1, i] = math.sqrt(j * (j + 1) - m * (m + 1))
    jm = jp.conj().T
    return (0.5 * (jp + jm), -0.5j * (jp - jm),
            np.diag(ms).astype(np.complex128), ms)


def su2_axis_angle(j, axis, theta):
    """exp(-i theta n.J) in the spin-j irrep."""
    jx, jy, jz, _ = su2_spin_ops(j)
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    a = theta * (n[0] * jx + n[1] * jy + n[2] * jz)
    w, v = np.linalg.eigh(a)
    return (v * np.exp(-1j * w)) @ v.conj().T


def haar_su2(rng):
    """Euler angles of a Haar-random SU(2) element, (A25): alpha and gamma
    uniform on [0,2pi), cos(beta) uniform on [-1,1]."""
    a, b, c = rng.random(3)
    return (2.0 * math.pi * a, math.acos(1.0 - 2.0 * b), 2.0 * math.pi * c)


# ---------------------------------------------------------------------
# SPHERICAL TENSOR BASIS FOR A SPIN-j PATCH
# ---------------------------------------------------------------------
class SpinBasis(object):
    """Everything that depends only on j: the spherical tensor operators
    T^(k)_q of (A6), the state/synthetic-state matrix M of (29), and the
    'Fourier' matrix F of (B13).

    Superoperators are (2j+1)^2 x (2j+1)^2 matrices in the T basis, with
    the basis ordered as (E5): k = 0..2j and, within each k, q descending
    from +k to -k. In that ordering a global SU(2) rotation is block
    diagonal with the blocks being the Wigner D-matrices D^k, (20).

    CONVENTION TRAP: T^(k)_q is orthonormal but NOT Hermitian, so the
    superket components of a Hermitian rho are complex. vec()/unvec()
    keep the Hilbert-Schmidt inner product <<A|B>> = tr(A^dag B), which
    is what makes <<E|S|rho>> a probability.
    """

    def __init__(self, j):
        self.j = float(j)
        if not _is_int(2 * self.j) or self.j < 0:
            raise ValueError("j must be a non-negative (half-)integer")
        self.d = int(round(2 * self.j)) + 1
        self.D2 = self.d * self.d
        self.ells = [self.j - i for i in range(self.d)]
        self.ks = list(range(int(round(2 * self.j)) + 1))
        self.index = []
        self.offsets = []
        off = 0
        for k in self.ks:
            self.offsets.append(off)
            for q in range(k, -k - 1, -1):
                self.index.append((k, q))
            off += 2 * k + 1
        self.pos = dict((kq, i) for i, kq in enumerate(self.index))
        self.T = np.zeros((self.D2, self.d, self.d), dtype=np.complex128)
        for a, (k, q) in enumerate(self.index):
            self.T[a] = self._tensor(k, q)
        self._Tflat = self.T.reshape(self.D2, -1)
        self._Tc = self._Tflat.conj()
        self.M = np.array(
            [[math.sqrt((2 * k + 1) / (2 * self.j + 1))
              * clebsch_gordan(self.j, l, k, 0, self.j, l)
              for l in self.ells] for k in self.ks])
        # F as printed in (I2): the 1/(2j+1) prefactor of the 6j form is
        # multiplied back in so that f = F p holds with sum_k p_k = 1.
        self.F = np.array(
            [[(2 * self.j + 1) * (-1.0) ** (2 * self.j + k + kk)
              * wigner_6j(k, self.j, self.j, kk, self.j, self.j)
              for kk in self.ks] for k in self.ks])
        self.Finv = np.linalg.inv(self.F)
        self._rot_cache = {}
        for k in self.ks:
            self._rot_cache[k] = self._rot_prep(k)
        self._rot_cache[self.j] = self._rot_prep(self.j)

    # -- construction -------------------------------------------------
    def _tensor(self, k, q):
        t = np.zeros((self.d, self.d), dtype=np.complex128)
        pref = math.sqrt((2 * k + 1) / (2 * self.j + 1))
        for a, m in enumerate(self.ells):
            for b, mp in enumerate(self.ells):
                cg = clebsch_gordan(self.j, mp, k, q, self.j, m)
                if cg:
                    t[a, b] = pref * cg
        return t

    def _rot_prep(self, j):
        jx, jy, jz, ms = su2_spin_ops(j)
        w, v = np.linalg.eigh(jy)
        return (ms, w, v)

    # -- vectorization ------------------------------------------------
    def vec(self, a):
        return self._Tc @ np.asarray(a, dtype=np.complex128).reshape(-1)

    def unvec(self, v):
        return np.tensordot(np.asarray(v), self.T, axes=([0], [0]))

    def superop_from_kraus(self, kraus):
        """S_{(kq),(k'q')} = tr(T^(k)dag_q  sum_a K_a T^(k')_q' K_a^dag)."""
        b = np.zeros((self.D2, self.d, self.d), dtype=np.complex128)
        for kop in kraus:
            kop = np.asarray(kop, dtype=np.complex128)
            b += np.einsum('ij,ajk,lk->ail', kop, self.T, kop.conj())
        return self._Tc @ b.reshape(self.D2, -1).T

    def superop_from_map(self, func):
        """Same, for a linear map given as a python callable on matrices."""
        b = np.array([func(self.T[a]) for a in range(self.D2)])
        return self._Tc @ b.reshape(self.D2, -1).T

    # -- rotations ----------------------------------------------------
    def wigner_D(self, j, alpha, beta, gamma):
        """D^j_{m'm}(a,b,c) = e^{-i m' a} d^j_{m'm}(b) e^{-i m c}, (A18),
        built from a cached eigendecomposition of Jy rather than Wigner's
        factorial sum. A Haar sample then costs dense linear algebra, not
        thousands of python-level factorials, which is the difference
        between a 3-second and a 30-minute RB run."""
        ms, w, v = self._rot_cache[j]
        dy = (v * np.exp(-1j * beta * w)) @ v.conj().T
        return (np.exp(-1j * alpha * ms)[:, None] * dy
                * np.exp(-1j * gamma * ms)[None, :])

    def rotation_superop(self, alpha, beta, gamma, want_blocks=False):
        s = np.zeros((self.D2, self.D2), dtype=np.complex128)
        blocks = []
        for k, o in zip(self.ks, self.offsets):
            dk = self.wigner_D(k, alpha, beta, gamma)
            s[o:o + 2 * k + 1, o:o + 2 * k + 1] = dk
            if want_blocks:
                blocks.append(dk)
        return (s, blocks) if want_blocks else s

    def irrep_projector(self, k):
        p = np.zeros((self.D2, self.D2), dtype=np.complex128)
        o = self.offsets[k]
        for i in range(2 * k + 1):
            p[o + i, o + i] = 1.0
        return p

    # -- readout ------------------------------------------------------
    def quality_parameters(self, chan):
        """f_k = Tr(Pi_k E)/Tr(Pi_k), (J2). This is the exact, infinite
        sample size answer that any SPAM-robust protocol must converge
        to; the RB simulation below is checked against it."""
        f = np.zeros(len(self.ks))
        for k in self.ks:
            o = self.offsets[k]
            n = 2 * k + 1
            f[k] = float(np.real(np.trace(chan[o:o + n, o:o + n]))) / n
        return f

    def error_rates(self, f):
        """p = F^{-1} f, (26)/(28). p_k is the rate of random weight-k
        SU(2) errors and sum_k p_k = 1."""
        return self.Finv @ np.asarray(f, dtype=float)

    def average_gate_fidelity(self, f):
        """F_ave = (d^{-1} sum_k Tr(Pi_k) f_k + 1)/(d+1), (C8)."""
        s = sum((2 * k + 1) * f[k] for k in self.ks)
        return float((s / self.d + 1.0) / (self.d + 1.0))

    def entanglement_fidelity(self, f):
        """p_0, the zero-error rate, which equals the process fidelity."""
        return float(sum((2 * k + 1) * f[k] for k in self.ks) / self.d ** 2)

    def state_superkets(self, unitaries=None):
        """Superkets of the 2j+1 J_z eigenstate projectors |l><l|,
        optionally each conjugated by its own unitary (SPAM error)."""
        out = np.zeros((self.D2, self.d), dtype=np.complex128)
        for i in range(self.d):
            p = np.zeros((self.d, self.d), dtype=np.complex128)
            p[i, i] = 1.0
            if unitaries is not None:
                v = unitaries[i]
                p = v @ p @ v.conj().T
            out[:, i] = self.vec(p)
        return out


# ---------------------------------------------------------------------
# ERROR CHANNELS
# ---------------------------------------------------------------------
def channel_unitary(basis, u):
    return basis.superop_from_kraus([u])


def channel_coherent_jz2(basis, gamma):
    """rho -> U rho U^dag with U = exp(-i gamma J_z^2), the paper's
    numerical example. Supported only in even irreps; perturbatively it
    is a pure quadrupole (k=2) error."""
    _, _, _, ms = su2_spin_ops(basis.j)
    return channel_unitary(basis, np.diag(np.exp(-1j * gamma * ms ** 2)))


def channel_dephasing(basis, gamma):
    """Energy-dependent dephasing, (J3):
    <l|rho|l'> -> exp(-gamma (l-l')^2) <l|rho|l'>.
    """
    _, _, _, ms = su2_spin_ops(basis.j)
    mask = np.exp(-gamma * (ms[:, None] - ms[None, :]) ** 2) # NOTE: paper prose has stray i; (J3) itself is the real form
    return basis.superop_from_map(lambda a: a * mask)


def channel_depolarizing(basis, p):
    """rho -> (1-p) rho + p I/d. Twirl-invariant, so f_k = 1-p for all
    k > 0; useful as a closed-form check on the protocols."""
    d = basis.d
    ident = np.eye(basis.D2, dtype=np.complex128)
    proj0 = basis.irrep_projector(0)
    return (1.0 - p) * ident + p * proj0


def channel_compose(*chans):
    out = np.eye(chans[0].shape[0], dtype=np.complex128)
    for c in chans:
        out = c @ out
    return out


# ---------------------------------------------------------------------
# ZERO-NOISE SAMPLE COMPLEXITY, (G26) (G37) (G53) (G55)
# ---------------------------------------------------------------------
def _c_factor(k, kk, protocol):
    """C(k,k') of (46): 1 for character weighting, the squared
    Clebsch-Gordan (C^{k',0}_{k,0;k,0})^2 for rank-1 weighting. Since
    that square is at most the character sum, R1 variance is manifestly
    below chi variance -- the paper's central claim."""
    if protocol in ("chi", "sschi"):
        return 1.0
    return clebsch_gordan(k, 0, k, 0, kk, 0) ** 2


def zero_noise_variance_physical(basis, k, ell_index, protocol):
    """(45)/(G53)/(G55): normalized variance for xRB with physical J_z
    eigenstate SPAM. Diverges as M_{k,l} -> 0, which is why the best
    choice of SPAM depends on k."""
    m = basis.M
    if abs(m[k, ell_index]) < 1e-12:
        return float("inf")
    s = 0.0
    for kk in range(0, min(2 * k, int(round(2 * basis.j))) + 1):
        s += _c_factor(k, kk, protocol) * m[kk, ell_index] ** 2 / (2 * kk + 1)
    return (2 * k + 1) ** 2 / m[k, ell_index] ** 4 * s - 1.0


def zero_noise_variance_synthetic(basis, k, protocol):
    """(47)/(G26)/(G37): variance for SSxRB. Pure SSRB is zero, (G3)."""
    if protocol == "ssrb":
        return 0.0
    m = basis.M
    s = 0.0
    for kk in range(0, min(2 * k, int(round(2 * basis.j))) + 1):
        s += (_c_factor(k, kk, protocol) / (2 * kk + 1)
              * float(np.sum(m[k] ** 2 * m[kk])) ** 2)
    return float((2 * k + 1) ** 2 * s - np.sum(m[k] ** 4))


def best_physical_spam(basis, k, protocol):
    """The J_z eigenstate that minimizes the xRB variance for irrep k."""
    vs = [(zero_noise_variance_physical(basis, k, i, protocol), i)
          for i in range(basis.d)]
    v, i = min(vs)
    return basis.ells[i], v


def shots_for_precision(variance, sigma):
    """Central limit theorem: N = Var / sigma^2 samples of the estimator
    for a target standard error sigma on the quality parameter."""
    return variance / max(sigma, 1e-300) ** 2


# ---------------------------------------------------------------------
# THE PROTOCOLS
# ---------------------------------------------------------------------
def _fit_single_exponential(ms, ys):
    """Fit m -> A f^m. Log-linear least squares on the positive points,
    which is deterministic and needs no optimizer; returns (A, f,
    sigma_f) with sigma_f the propagated standard error of the slope.

    Points that have gone negative (finite-sample fluctuation in a
    high-k irrep whose signal has already decayed into the noise) are
    dropped rather than clipped, because clipping biases f upward."""
    ms = np.asarray(ms, dtype=float)
    ys = np.asarray(ys, dtype=float)
    ok = ys > 1e-9
    if int(ok.sum()) < 2:
        return float("nan"), float("nan"), float("nan")
    x, y = ms[ok], np.log(ys[ok])
    xb = x.mean()
    sxx = float(((x - xb) ** 2).sum())
    if sxx <= 0.0:
        return float("nan"), float("nan"), float("nan")
    slope = float(((x - xb) * (y - y.mean())).sum() / sxx)
    inter = float(y.mean() - slope * xb)
    res = y - (inter + slope * x)
    sig = (math.sqrt(max(float((res ** 2).sum()) / (len(x) - 2) / sxx, 0.0))
           if len(x) > 2 else 0.0)
    f = math.exp(slope)
    return math.exp(inter), f, f * sig


def make_spam(basis, phi=0.0, meas_mode="coherent", rng=None):
    """State preparations and measurement effects, with the paper's
    adversarial SPAM error models (Sec. 6).

    phi = 0 is perfect SPAM. Otherwise every prepared |l><l| is rotated
    by exp(-i phi n_l . J) with its OWN fixed random axis -- per-prep,
    not per-shot, because random-per-shot SU(2) error averages out and
    would not test anything. Measurement error is either a single
    coherent rotation of all effects ('coherent') or a fixed random
    permutation of them ('permute'); both preserve sum_l E_l = 1."""
    if rng is None:
        rng = np.random.default_rng()
    d = basis.d
    preps = None
    if phi > 0.0:
        preps = [su2_axis_angle(basis.j, rng.normal(size=3), phi)
                 for _ in range(d)]
    rho = basis.state_superkets(preps)
    if phi > 0.0 and meas_mode == "coherent":
        v = su2_axis_angle(basis.j, rng.normal(size=3), phi)
        eff = basis.state_superkets([v] * d)
    elif phi > 0.0 and meas_mode == "permute":
        perm = rng.permutation(d)
        eff = basis.state_superkets()[:, perm]
    else:
        eff = basis.state_superkets()
    return rho, eff


def synthetic_rb(basis, err, protocol="ssr1", seq_lengths=(1, 2, 4, 8, 16, 32),
                 n_circuits=200, rng=None, phi=0.0, meas_mode="coherent",
                 shots=None, progress=False):
    """Simulate an SU(2) synthetic RB experiment and return f_k, p_k.

    err is the gate-independent error superoperator in the T basis: every
    SU(2) rotation is implemented as err @ G(g). protocol is one of
    'ssrb', 'sschi', 'ssr1'.

    For each sequence length m and each of n_circuits random circuits we
    apply g_1..g_m followed by the inversion gate g_inv = (g_m...g_1)^dag,
    so a perfect circuit is the identity. The synthetic-gate protocols
    additionally draw one more Haar element g, compile it with g_1 into a
    single gate (so it costs only ONE application of the noise channel,
    as in (34)), and weight the outcome by (2k+1) chi_k(g) for sschi or
    (2k+1) d^k_00(g) for ssr1. Because g_inv still inverts only
    g_1..g_m, the net operation of the circuit is g, which is exactly the
    'ending gate' picture of filtered RB.

    shots=None means infinitely many shots per circuit, i.e. the exact
    outcome probabilities are used; this is what the paper does for its
    figures and it isolates circuit-sampling variance from shot noise.
    Setting shots to an integer draws that many J_z outcomes per circuit
    per initial state instead.

    Post-processing follows Sec. 4.3.1: d_{k,m} is the (k,k) entry of
    M P_m M^T, fitted to A_k f_k^m, and p = F^{-1} f.
    """
    if rng is None:
        rng = np.random.default_rng()
    if protocol not in ("ssrb", "sschi", "ssr1"):
        raise ValueError("unknown protocol %r" % (protocol,))
    seq_lengths = [int(m) for m in seq_lengths]
    d, nk = basis.d, len(basis.ks)
    rho, eff = make_spam(basis, phi, meas_mode, rng)
    effc = eff.conj().T
    dkm = np.zeros((nk, len(seq_lengths)))
    for mi, m in enumerate(seq_lengths):
        acc = np.zeros((nk, d, d), dtype=np.complex128)
        for _ in range(n_circuits):
            gtot = np.eye(basis.D2, dtype=np.complex128)
            s = np.eye(basis.D2, dtype=np.complex128)
            w = np.ones(nk)
            for gi in range(m):
                al, be, ga = haar_su2(rng)
                g = basis.rotation_superop(al, be, ga)
                gtot = g @ gtot
                if gi == 0 and protocol != "ssrb":
                    al2, be2, ga2 = haar_su2(rng)
                    gx, blocks = basis.rotation_superop(
                        al2, be2, ga2, want_blocks=True)
                    for k in basis.ks:
                        if protocol == "sschi":
                            w[k] = (2 * k + 1) * float(
                                np.real(np.trace(blocks[k])))
                        else:
                            # d^k_00 is the middle diagonal entry of D^k;
                            # in the q-descending ordering that is [k,k].
                            w[k] = (2 * k + 1) * float(
                                np.real(blocks[k][k, k]))
                    s = (err @ (g @ gx)) @ s
                else:
                    s = (err @ g) @ s
            s = (err @ gtot.conj().T) @ s
            p = np.real(effc @ s @ rho)
            if shots is not None:
                p = _sample_shots(p, shots, rng)
            for k in basis.ks:
                acc[k] += w[k] * p
        acc /= float(n_circuits)
        for k in basis.ks:
            dkm[k, mi] = float(np.real((basis.M @ acc[k] @ basis.M.T)[k, k]))
        if progress:
            print("    m=%-4d d_k = %s" % (m, np.array2string(
                dkm[:, mi], precision=4, suppress_small=True)),
                file=sys.stderr)
    a = np.zeros(nk)
    f = np.zeros(nk)
    sf = np.zeros(nk)
    for k in basis.ks:
        a[k], f[k], sf[k] = _fit_single_exponential(seq_lengths, dkm[k])
        if np.isnan(f[k]) or f[k] < 1e-9:
            print("[WARN] Irrep k=%d signal decayed below threshold; fitting returned zero/nan." % k, file=sys.stderr)
    p = basis.error_rates(np.nan_to_num(f, nan=0.0))
    sp = error_rate_sigma(basis, sf)
    return {"protocol": protocol, "A": a, "f": f, "sigma_f": sf,
            "p": p, "sigma_p": sp, "signal": dkm,
            "seq_lengths": seq_lengths, "n_circuits": n_circuits}


def _sample_shots(probs, shots, rng):
    """Draw `shots` J_z outcomes per initial state. probs[:, i] is the
    outcome distribution for initial state i (columns = init)."""
    out = np.zeros_like(probs)
    for i in range(probs.shape[1]):
        col = np.clip(probs[:, i], 0.0, None)
        tot = col.sum()
        if tot <= 0.0:
            continue
        out[:, i] = rng.multinomial(shots, col / tot) / float(shots) * tot
    return out


def error_rate_sigma(basis, sigma_f):
    """(J1): propagate the quality-parameter uncertainties through
    p = F^{-1} f."""
    sf = np.nan_to_num(np.asarray(sigma_f, dtype=float), nan=0.0)
    cov = basis.Finv @ np.diag(sf ** 2) @ basis.Finv.T
    return np.sqrt(np.clip(np.diag(cov), 0.0, None))


def su2_error_report(basis, f, sigma_f=None, label=""):
    """Print the irrep-resolved error budget of a channel."""
    p = basis.error_rates(np.asarray(f, dtype=float))
    sp = error_rate_sigma(basis, sigma_f) if sigma_f is not None else None
    print("  irrep  f_k          p_k (weight-k SU(2) error rate)")
    for k in basis.ks:
        line = "   %2d    %-11.6f  %+.6e" % (k, f[k], p[k])
        if sp is not None:
            line += "  +- %.1e" % sp[k]
        print(line)
    fav = basis.average_gate_fidelity(f)
    print("  %saverage gate fidelity  %.6f   infidelity %.3e"
          % (label + " " if label else "", fav, 1.0 - fav))
    print("  entanglement/process fidelity p_0 = %.6f" % p[0])
    if len(basis.ks) > 2:
        corr = float(sum(p[k] for k in basis.ks if 1 <= k <= 2))
        unc = float(sum(p[k] for k in basis.ks if k > 2))
        print("  spin-code correctable weight k<=2: %.6e" % corr)
        print("  uncorrectable weight k>2:          %.6e" % unc)
    return p


# ---------------------------------------------------------------------
# BRIDGE TO THE K4 DECOHERENCE MODEL
# ---------------------------------------------------------------------
def rb_to_decoherence(f1, a1=1.0):
    """(f_1, A_1) -> (a, b) of the K4/QPE model q(k) = 1 - e^{-ak-b}.

    JUSTIFICATION. The K4 readout is an interference measurement:
    P(m|k,beta) = (1 + damp(k) Re[e^{i beta} a_k])/2 with a_k =
    <Phi|U^k|Phi>. Under a gate-independent, SU(2)-twirled error channel
    the traceless part of the system state is multiplied by f_1 per
    application of U -- that is precisely what the k=1 quality parameter
    IS -- so the interference term carries a factor f_1^k, and the
    remaining constant A_1 is SPAM. Hence damp(k) = A_1 f_1^k =
    e^{-ak-b} with a = -ln f_1 and b = -ln A_1.

    This is exact for a qubit patch (j=1/2), where k=1 is the only
    nontrivial irrep. For j > 1/2 a single scalar cannot represent the
    2j nontrivial decay rates and the decomposition p_k should be used
    directly; a = -ln f_1 is then only the dipole component."""
    f1 = float(f1)
    if not (0.0 < f1 <= 1.0):
        raise ValueError("f1 must lie in (0,1], got %r" % (f1,))
    a1 = float(a1)
    if a1 <= 0.0:
        raise ValueError("A1 must be positive, got %r" % (a1,))
    return (-math.log(f1), -math.log(a1))


def decoherence_to_rb(a, b, j=0.5):
    """The inverse map: what per-gate error rate does a fitted (a,b)
    actually assert? Returns a dict with f_1, the implied average gate
    fidelity, and the SPAM coefficient. Use this to sanity-check a
    phenomenological fit against a benchmarking number."""
    basis = SpinBasis(j)
    f1 = math.exp(-float(a))
    f = np.ones(len(basis.ks))
    for k in basis.ks:
        if k > 0:
            f[k] = f1 ** k if j > 0.5 else f1
    if j == 0.5:
        f = np.array([1.0, f1])
    fav = basis.average_gate_fidelity(f)
    return {"f1": f1, "A1": math.exp(-float(b)),
            "F_ave": fav, "infidelity": 1.0 - fav,
            "note": ("exact for j=1/2; for j>1/2 the scalar model is "
                     "assumed to act as f_k = f_1^k, which is a guess")}


def h2_ideal_unitary(t=None):
    """exp(-i (h1 Z + h2 X) t), the operator h2_step is TRYING to be."""
    if t is None:
        t = h2_evolution_time()
    hn = math.hypot(H2_H1, H2_H2)
    hop = (H2_H1 * _PAULI["Z"] + H2_H2 * _PAULI["X"]) / hn
    return (math.cos(hn * t) * np.eye(2, dtype=np.complex128)
            - 1j * math.sin(hn * t) * hop)


def h2_trotter_unitary(t=None, nbits=None):
    """The operator h2_step ACTUALLY applies, extracted by running the
    reference simulator on both basis states. Deriving it from the code
    rather than re-deriving it by hand means the two cannot drift."""
    cols = []
    for bit in range(2):
        s = NumpyStatevector(1)
        if bit:
            s.x(0)
        h2_step(s, t, nbits=nbits)
        cols.append(s.ket())
    return np.array(cols).T


def k4_step_error_channel(basis=None, t=None, nbits=None, extra=None):
    """The SU(2) error channel of ONE K4/H2 Trotter step.

    V = U_trotter U_ideal^dag is the coherent residual left over by the
    first-order Lie-Trotter splitting -- the very error whose energy cost
    the anchor above measures as ~1.0e-3 hartree. Benchmarking it as an
    SU(2) channel converts that into a gate infidelity, which is the
    quantity the RB literature and the K4 error budget both speak.

    MODELING ASSUMPTION, stated plainly: RB benchmarks a GROUP with a
    gate-independent error channel, while the K4 step is one fixed gate.
    What this function returns is the error of that gate, promoted to the
    error channel attached to every SU(2) rotation. That is the standard
    RB idealization and it is an assumption, not a derivation.

    extra, if given, is an additional channel superoperator composed
    after the coherent residual -- use it to fold in incoherent hardware
    noise (channel_depolarizing, channel_dephasing) on top of Trotter."""
    if basis is None:
        basis = SpinBasis(0.5)
    v = h2_trotter_unitary(t, nbits) @ h2_ideal_unitary(t).conj().T
    chan = basis.superop_from_kraus([v])
    if extra is not None:
        chan = channel_compose(chan, extra)
    return basis, chan, v


# ---------------------------------------------------------------------
# PAPER REFERENCE NUMBERS (for rb_selftest)
# ---------------------------------------------------------------------
PAPER_TABLE1 = {
    "chi": [7.0, 28.6816, 91.8386, 308.139, 268.103, 514.734, 404.56,
            381.656],
    "r1": [7.0, 7.52245, 12.5807, 42.3744, 21.0241, 32.779, 23.2173,
           21.6442],
    "sschi": [0.0, 1.07619, 3.23842, 6.15572, 10.4498, 15.668, 23.0531,
              34.0697],
    "ssr1": [0.0, 0.269048, 0.540816, 0.773292, 1.02387, 1.28994, 1.62223,
             2.11888],
}
PAPER_TABLE2 = {
    0.0: (0.0, 0.0, 0.0, 0.0),
    0.5: (23.0, 5.0, 4.0, 1.0),
    1.0: (25.25, 4.89286, 8.66667, 1.40476),
    1.5: (91.1811, 9.9465, 13.408, 1.63867),
    2.0: (95.25, 11.163, 18.4047, 1.80578),
    2.5: (209.672, 15.5894, 23.5132, 1.9322),
    3.0: (215.636, 18.0822, 28.7441, 2.03407),
    3.5: (381.656, 21.6442, 34.0697, 2.11888),
}
PAPER_COHERENT_P = [0.9668, 0.0, 0.03301, 0.0, 1.434e-4, 0.0, 1.110e-7, 0.0]
PAPER_DEPHASING_P = [0.9068, 0.08787, 0.005118, 1.991e-4, 5.315e-6,
                     9.504e-8, 1.039e-9, 5.297e-12]


# ---------------------------------------------------------------------
# DRIVERS
# ---------------------------------------------------------------------
def run_rb_tables(j=3.5):
    """Reproduce Tables 1 and 2: zero-noise estimator variance, i.e. the
    sample complexity of each protocol before a single circuit is run."""
    basis = SpinBasis(j)
    print("Table 1 -- zero-noise variance vs irrep k at j = %g" % j)
    print("  %3s %12s %12s %12s %12s   %s"
          % ("k", "chiRB", "R1RB", "SSchiRB", "SSR1RB", "best chiRB SPAM"))
    for k in basis.ks:
        _, v1 = best_physical_spam(basis, k, "chi")
        lb, v2 = best_physical_spam(basis, k, "r1")
        v3 = zero_noise_variance_synthetic(basis, k, "sschi")
        v4 = zero_noise_variance_synthetic(basis, k, "ssr1")
        print("  %3d %12.6g %12.6g %12.6g %12.6g   l = +-%s"
              % (k, v1, v2, v3, v4, abs(lb)))
    print("\nTable 2 -- zero-noise variance of the top irrep k = 2j")
    print("  %5s %12s %12s %12s %12s" % ("j", "chiRB", "R1RB", "SSchiRB",
                                         "SSR1RB"))
    for jj in sorted(PAPER_TABLE2):
        bb = SpinBasis(jj)
        k = int(round(2 * jj))
        if k == 0:
            print("  %5g %12.6g %12.6g %12.6g %12.6g" % (jj, 0, 0, 0, 0))
            continue
        _, v1 = best_physical_spam(bb, k, "chi")
        _, v2 = best_physical_spam(bb, k, "r1")
        print("  %5g %12.6g %12.6g %12.6g %12.6g"
              % (jj, v1, v2,
                 zero_noise_variance_synthetic(bb, k, "sschi"),
                 zero_noise_variance_synthetic(bb, k, "ssr1")))
    k = int(round(2 * j))
    _, vchi = best_physical_spam(basis, k, "chi")
    vssr1 = zero_noise_variance_synthetic(basis, k, "ssr1")
    print("\n  TOTAL shots for sigma = 0.05 on f_%d: chiRB %.3g vs SSR1RB %.3g "
          "(%.0fx)" % (k, shots_for_precision(vchi, 0.05),
                       shots_for_precision(vssr1, 0.05) * basis.d,
                       shots_for_precision(vchi, 0.05)
                       / max(shots_for_precision(vssr1, 0.05) * basis.d, 1e-30)))
    print("  (SSR1RB total shots includes the 2j+1 = %d multiplier, since one synthetic "
          "circuit costs one physical shot per initial state)" % basis.d)


def run_rb_rates(j=3.5, gamma_coh=0.04, gamma_deph=0.01):
    """Exact irrep-resolved error rates of the paper's two noise channels,
    the infinite sample size answer that the protocols must converge to."""
    basis = SpinBasis(j)
    print("coherent  U = exp(-i %.3f J_z^2)" % gamma_coh)
    f = basis.quality_parameters(channel_coherent_jz2(basis, gamma_coh))
    p = su2_error_report(basis, f, label="coherent")
    if abs(j - 3.5) < 1e-9:
        print("  paper (48): p0,p2,p4,p6 = 0.967, 0.0330, 1.43e-4, 1.11e-7")
        print("  this run:                 %.3f, %.4f, %.2e, %.2e"
              % (p[0], p[2], p[4], p[6]))
    print("\ndephasing  <l|rho|l'> -> exp(-%.3f (l-l')^2) <l|rho|l'>"
          % gamma_deph)
    f2 = basis.quality_parameters(channel_dephasing(basis, gamma_deph))
    su2_error_report(basis, f2, label="dephasing")
    return f, f2


def run_rb_experiment(j=3.5, protocol="ssr1", gamma=0.04, phi=0.0,
                      meas_mode="coherent", n_circuits=200,
                      seq_lengths=(1, 2, 4, 8, 12, 16, 24, 32),
                      shots=None, seed=1337, channel="coherent"):
    """Run a full synthetic RB experiment against a known channel and
    compare the recovered rates to the exact ones."""
    basis = SpinBasis(j)
    if channel == "coherent":
        err = channel_coherent_jz2(basis, gamma)
    elif channel == "dephasing":
        err = channel_dephasing(basis, gamma)
    elif channel == "depolarizing":
        err = channel_depolarizing(basis, gamma)
    else:
        raise ValueError("unknown channel %r" % (channel,))
    f_exact = basis.quality_parameters(err)
    p_exact = basis.error_rates(f_exact)
    rng = np.random.default_rng(seed)
    res = synthetic_rb(basis, err, protocol=protocol,
                       seq_lengths=seq_lengths, n_circuits=n_circuits,
                       rng=rng, phi=phi, meas_mode=meas_mode, shots=shots)
    print("%s  j=%g  %s channel gamma=%.4g  phi=%.3g (%s)  "
          "%d circuits x %d lengths%s"
          % (protocol.upper(), j, channel, gamma, phi, meas_mode,
             n_circuits, len(seq_lengths),
             "" if shots is None else "  %d shots/circuit" % shots))
    print("  %3s %12s %12s %10s %14s %14s"
          % ("k", "f_k (RB)", "f_k (exact)", "A_k", "p_k (RB)", "p_k (exact)"))
    for k in basis.ks:
        print("  %3d %12.6f %12.6f %10.4f %+14.6e %+14.6e"
              % (k, res["f"][k], f_exact[k], res["A"][k],
                 res["p"][k], p_exact[k]))
    fav_rb = basis.average_gate_fidelity(np.nan_to_num(res["f"], nan=1.0))
    fav_ex = basis.average_gate_fidelity(f_exact)
    print("  F_ave  RB %.6f   exact %.6f   difference %+.2e"
          % (fav_rb, fav_ex, fav_rb - fav_ex))
    res["f_exact"] = f_exact
    res["p_exact"] = p_exact
    return res


def run_k4_rb_anchor(t=None, nbits=None, run_experiment=True,
                     n_circuits=400, seed=1337, depol=0.0,
                     seq_lengths=(4, 8, 16, 32, 64, 128, 256),
                     rerun_anchor=True):
    """THE INTEGRATION POINT.

    1. Benchmark the K4/H2 Trotter step as an SU(2) channel on the j=1/2
       patch: exact quality parameters, error rates, gate fidelity.
    2. Optionally run the actual SSR1RB protocol against that channel and
       confirm the protocol recovers the exact numbers.
    3. Convert f_1 into the (a,b) of the K4 decoherence model, and invert
       the paper's fitted (0.17, -0.17) into the gate infidelity it is
       secretly asserting.
    4. Re-run the H2 anchor with the RB-derived decoherence instead of
       the fitted one.

    SEQUENCE LENGTHS. The Trotter residual is a ~1e-3 infidelity, so a
    sequence of 32 gates decays the signal by only a few percent and the
    fit is dominated by circuit-sampling noise. The default lengths run
    out to 256, where f_1^m has fallen to ~0.67; the rule of thumb is
    m_max of order a few times 1/(1-f_1). Short sequences do not give a
    wrong answer, they give a noisy one, and at 1e-3 infidelity the
    difference is the whole measurement.
    (Note: For the pure Trotter channel, the infidelity is so small (~1e-3)
    that the sequence length warning will always fire for the default 256 max
    length. This is expected behavior.)
    """
    if t is None:
        t = h2_evolution_time()
    basis, err, v = k4_step_error_channel(t=t, nbits=nbits)
    extra = None
    if depol > 0.0:
        extra = channel_depolarizing(basis, depol)
        basis, err, v = k4_step_error_channel(t=t, nbits=nbits, extra=extra)

    print("=" * 68)
    print("K4 STEP AS AN SU(2) CHANNEL  (j = 1/2 patch)")
    print("=" * 68)
    print("  U_trotter U_ideal^dag, principal angle %.6f rad"
          % (2.0 * math.acos(min(1.0, abs(float(np.real(np.trace(v))) / 2.0)))))
    if depol > 0.0:
        print("  plus depolarizing p = %.4g composed after it" % depol)
    f_exact = basis.quality_parameters(err)
    su2_error_report(basis, f_exact, label="K4 step")

    f1 = float(f_exact[1])
    a_rb, b_rb = rb_to_decoherence(f1, 1.0)
    print("\n  f_1 = %.8f  ->  decoherence exponent a = -ln f_1 = %.6f"
          % (f1, a_rb))

    a_meas, b_meas = a_rb, b_rb
    if run_experiment:
        print("\n  SSR1RB against the same channel: %d circuits, "
              "lengths %s" % (n_circuits, list(seq_lengths)))
        
        if seq_lengths[-1] < 1.0 / max(1.0 - f1, 1e-9):
            print("[WARN] Max sequence length %d is too short for f_1 = %.6f; fit may be degraded." % (seq_lengths[-1], f1), file=sys.stderr)
            
        rng = np.random.default_rng(seed)
        res = synthetic_rb(basis, err, protocol="ssr1",
                           seq_lengths=seq_lengths, n_circuits=n_circuits,
                           rng=rng)
        a_meas, b_meas = rb_to_decoherence(min(res["f"][1], 1.0),
                                           min(res["A"][1], 1.0))
        print("    f_1  RB %.8f +- %.1e   exact %.8f   difference %+.2e"
              % (res["f"][1], res["sigma_f"][1], f1, res["f"][1] - f1))
        print("    A_1  RB %.6f" % res["A"][1])
        print("    a    RB %.6f               exact %.6f" % (a_meas, a_rb))
        print("    b    RB %.6f" % b_meas)
        out_exp = res
    else:
        out_exp = None

    print("\n" + "=" * 68)
    print("THE FITTED DECOHERENCE MODEL, INVERTED")
    print("=" * 68)
    inv = decoherence_to_rb(0.17, -0.17, j=0.5)
    print("  the QPE paper fits a = 0.17(2), b = -0.17(9). That asserts")
    print("    f_1     = %.6f" % inv["f1"])
    print("    F_ave   = %.6f  (infidelity %.4f per step)"
          % (inv["F_ave"], inv["infidelity"]))
    print("    A_1     = %.6f  (SPAM coefficient, b = -ln A_1)"
          % inv["A1"])
    print("  the Trotter residual alone gives infidelity %.3e, i.e. %.0fx"
          % (1.0 - basis.average_gate_fidelity(f_exact),
             inv["infidelity"] / max(1.0 - basis.average_gate_fidelity(f_exact),
                                     1e-30)))
    print("  smaller. The fitted a is therefore almost entirely hardware,")
    print("  not algorithmic: the K4 systematic and the K4 decoherence are")
    print("  independent error budgets and must not be conflated.")

    out = {"basis": basis, "channel": err, "f": f_exact,
           "a_rb": a_rb, "b_rb": b_rb, "a_meas": a_meas, "b_meas": b_meas,
           "fitted_inverse": inv, "experiment": out_exp}

    if rerun_anchor:
        print("\n" + "=" * 68)
        print("H2 ANCHOR WITH RB-DERIVED DECOHERENCE  (a = %.6f, b = %.6f)"
              % (a_meas, b_meas))
        print("=" * 68)
        out["anchor"] = run_h2_anchor(seed=seed, repeats=200,
                                      decoh=(a_meas, b_meas))
    return out


def rb_selftest():
    """Every claim of the RB section, checked against the paper."""
    print("SU(2) recoupling coefficients:")
    assert abs(clebsch_gordan(0.5, 0.5, 0.5, -0.5, 1, 0)
               - 1.0 / math.sqrt(2.0)) < 1e-12
    assert abs(clebsch_gordan(0.5, 0.5, 0.5, 0.5, 1, 1) - 1.0) < 1e-12
    assert abs(clebsch_gordan(1, 0, 1, 0, 2, 0) - math.sqrt(2.0 / 3.0)) < 1e-12
    assert abs(wigner_6j(1, 1, 1, 1, 1, 1) - 1.0 / 6.0) < 1e-12
    assert abs(wigner_6j(1, 2, 3, 1, 2, 3) - 1.0 / 105.0) < 1e-12
    # structural checks, which do not depend on a tabulated value:
    # {j1 j2 j3; 0 j3 j2} = (-1)^(j1+j2+j3) / sqrt((2j2+1)(2j3+1))
    for a_, b_, c_ in [(1, 2, 3), (2, 2, 2), (1.5, 1.5, 1), (3.5, 3.5, 2)]:
        want = (-1.0) ** (a_ + b_ + c_) / math.sqrt((2 * b_ + 1)
                                                    * (2 * c_ + 1))
        assert abs(wigner_6j(a_, b_, c_, 0, c_, b_) - want) < 1e-12
    # sum_x (2x+1) {a b x; c d p}{a b x; c d q} = delta_pq/(2p+1)
    for p_ in range(5):
        for q_ in range(5):
            tot = sum((2 * x + 1) * wigner_6j(2, 2, x, 2, 2, p_)
                      * wigner_6j(2, 2, x, 2, 2, q_) for x in range(5))
            want = (1.0 / (2 * p_ + 1)) if p_ == q_ else 0.0
            assert abs(tot - want) < 1e-10, (p_, q_, tot, want)
    print("  Clebsch-Gordan and 6j closed forms OK "
          "(tabulated values, the j=0 identity and 6j orthogonality)")

    b = SpinBasis(3.5)
    print("spherical tensor basis at j = 7/2:")
    ortho = np.abs(np.einsum('aij,bij->ab', b.T.conj(), b.T)
                   - np.eye(b.D2)).max()
    print("  |<<T_a|T_b>> - delta| = %.2e" % ortho)
    assert ortho < 1e-10
    print("  |M M^T - I| = %.2e" % np.abs(b.M @ b.M.T - np.eye(b.d)).max())
    assert np.abs(b.M @ b.M.T - np.eye(b.d)).max() < 1e-10
    # spot-check M against the printed (I1); rows 0-2 survive extraction
    assert abs(b.M[0, 0] - 0.5 / math.sqrt(2.0)) < 1e-9
    assert abs(b.M[1, 0] - 0.5 * math.sqrt(7.0 / 6.0)) < 1e-9
    assert abs(b.M[2, 1] - 0.5 / math.sqrt(42.0)) < 1e-9
    for k in b.ks:
        for i, l in enumerate(b.ells):
            assert abs(b.M[k, b.d - 1 - i] - (-1) ** k * b.M[k, i]) < 1e-10
    print("  M matches (I1) and obeys M_{k,-l} = (-1)^k M_{k,l}")
    # F against the printed (I2)
    assert np.abs(b.F - b.F.T).max() < 1e-10
    for kk in b.ks:
        assert abs(b.F[0, kk] - 1.0) < 1e-10
    for want, got in [(59.0 / 63.0, b.F[1, 1]), (17.0 / 21.0, b.F[1, 2]),
                      (-7.0 / 9.0, b.F[1, 7]), (7.0 / 15.0, b.F[2, 2]),
                      (-31.0 / 77.0, b.F[3, 3]), (-33.0 / 91.0, b.F[5, 5]),
                      (-1.0 / 6435.0, b.F[7, 7])]:
        assert abs(want - got) < 1e-9, (want, got)
    print("  F matches (I2) entry by entry")

    print("rotation superoperator:")
    rng = np.random.default_rng(3)
    al, be, ga = haar_su2(rng)
    uj = b.wigner_D(b.j, al, be, ga)
    err = np.abs(b.rotation_superop(al, be, ga)
                 - b.superop_from_kraus([uj])).max()
    print("  |blockdiag(D^k) - superop(D^j)| = %.2e" % err)
    assert err < 1e-9
    leg = max(abs(b.wigner_D(k, al, be, ga)[k, k].real
                  - np.polynomial.legendre.Legendre.basis(k)(math.cos(be)))
              for k in b.ks)
    print("  |d^k_00(beta) - P_k(cos beta)| = %.2e   ((F75))" % leg)
    assert leg < 1e-10

    print("character weighting really synthesizes Pi_k, (12)/(E9):")
    rng = np.random.default_rng(17)
    acc = np.zeros((b.D2, b.D2), dtype=np.complex128)
    n = 4000
    k = 2
    for _ in range(n):
        al, be, ga = haar_su2(rng)
        g, blocks = b.rotation_superop(al, be, ga, want_blocks=True)
        acc += (2 * k + 1) * np.conj(np.trace(blocks[k])) * g
    acc /= n
    dev = np.abs(acc - b.irrep_projector(k)).max()
    print("  |(2k+1)<chi_k(g)* G> - Pi_%d| = %.3f at %d samples "
          "(Monte Carlo, O(1/sqrt N))" % (k, dev, n))
    assert dev < 0.15

    print("Tables 1 and 2 against the paper:")
    worst = 0.0
    for k in b.ks:
        got = (best_physical_spam(b, k, "chi")[1],
               best_physical_spam(b, k, "r1")[1],
               zero_noise_variance_synthetic(b, k, "sschi"),
               zero_noise_variance_synthetic(b, k, "ssr1"))
        for name, g in zip(["chi", "r1", "sschi", "ssr1"], got):
            want = PAPER_TABLE1[name][k]
            worst = max(worst, abs(g - want) / max(abs(want), 1e-6))
    print("  Table 1 worst relative deviation %.2e" % worst)
    assert worst < 2e-5
    worst = 0.0
    for jj, want in PAPER_TABLE2.items():
        bb = SpinBasis(jj)
        kk = int(round(2 * jj))
        if kk == 0:
            continue
        got = (best_physical_spam(bb, kk, "chi")[1],
               best_physical_spam(bb, kk, "r1")[1],
               zero_noise_variance_synthetic(bb, kk, "sschi"),
               zero_noise_variance_synthetic(bb, kk, "ssr1"))
        for g, w in zip(got, want):
            worst = max(worst, abs(g - w) / max(abs(w), 1e-6))
    print("  Table 2 worst relative deviation %.2e" % worst)
    assert worst < 2e-5
    assert zero_noise_variance_synthetic(b, 3, "ssr1") \
        < zero_noise_variance_synthetic(b, 3, "sschi"), \
        "(G38) says rank-1 variance must be below character variance"

    print("error rates of the paper's channels:")
    f = b.quality_parameters(channel_coherent_jz2(b, 0.04))
    p = b.error_rates(f)
    print("  coherent  p = %s" % np.array2string(p, precision=6))
    assert abs(p.sum() - 1.0) < 1e-9
    for k in b.ks:
        want = PAPER_COHERENT_P[k]
        if want == 0.0:
            assert abs(p[k]) < 1e-9, "odd irreps must vanish for J_z^2"
        else:
            assert abs(p[k] - want) <= 5e-4 * max(abs(want), 1e-3) + 1e-6, \
                (k, p[k], want)
    f2 = b.quality_parameters(channel_dephasing(b, 0.01))
    p2 = b.error_rates(f2)
    print("  dephasing p = %s" % np.array2string(p2, precision=8))
    assert abs(p2.sum() - 1.0) < 1e-9
    for k in b.ks:
        assert abs(p2[k] - PAPER_DEPHASING_P[k]) <= 1e-3 * \
            abs(PAPER_DEPHASING_P[k]) + 1e-12, (k, p2[k])
    print("  both match (48) and the appendix J table")

    print("depolarizing closed form (f_k = 1-p for all k>0):")
    bq = SpinBasis(0.5)
    fd = bq.quality_parameters(channel_depolarizing(bq, 0.1))
    assert abs(fd[0] - 1.0) < 1e-12 and abs(fd[1] - 0.9) < 1e-12
    for jj in (0.5, 1.5, 3.5):
        bj = SpinBasis(jj)
        fj = bj.quality_parameters(channel_depolarizing(bj, 0.1))
        assert np.abs(fj[1:] - 0.9).max() < 1e-12
        want = 1.0 - 0.1 * (bj.d - 1.0) / bj.d
        assert abs(bj.average_gate_fidelity(fj) - want) < 1e-12
        pj = bj.error_rates(fj)
        assert abs(pj.sum() - 1.0) < 1e-9
    print("  f_k = 1-p for every k>0 and F_ave = 1 - p(d-1)/d, "
          "j = 1/2, 3/2, 7/2 OK")

    print("protocols recover the exact rates (small run, j = 1/2):")
    err = channel_coherent_jz2(bq, 0.30)
    fx = bq.quality_parameters(err)
    for proto in ("ssrb", "sschi", "ssr1"):
        rng = np.random.default_rng(5)
        r = synthetic_rb(bq, err, protocol=proto,
                         seq_lengths=(1, 2, 4, 8, 16, 32),
                         n_circuits=300, rng=rng)
        print("  %-6s f_1 = %.5f  exact %.5f  A_1 = %.4f"
              % (proto, r["f"][1], fx[1], r["A"][1]))
        assert abs(r["f"][1] - fx[1]) < 0.02, proto

    print("SPAM-robustness ordering at j = 1/2, phi = 0.25:")
    rng = np.random.default_rng(9)
    devs = {}
    for proto in ("ssrb", "ssr1"):
        r = synthetic_rb(bq, err, protocol=proto,
                         seq_lengths=(1, 2, 4, 8, 16, 32), n_circuits=300,
                         rng=np.random.default_rng(9), phi=0.25,
                         meas_mode="permute")
        devs[proto] = abs(r["f"][1] - fx[1])
        print("  %-6s |f_1 - exact| = %.4f" % (proto, devs[proto]))
    print("  (SSRB is not SPAM-robust; SSR1RB is. Ordering is statistical "
          "at 300 circuits and is reported, not asserted.)")

    print("decoherence bridge:")
    a, bb_ = rb_to_decoherence(math.exp(-0.17), math.exp(0.17))
    assert abs(a - 0.17) < 1e-12 and abs(bb_ + 0.17) < 1e-12
    inv = decoherence_to_rb(0.17, -0.17)
    print("  a=0.17 <-> f_1=%.6f, F_ave=%.6f, infidelity %.4f"
          % (inv["f1"], inv["F_ave"], inv["infidelity"]))
    assert abs(inv["f1"] - math.exp(-0.17)) < 1e-12

    print("K4 Trotter step benchmarked:")
    bas, chan, v = k4_step_error_channel()
    fk = bas.quality_parameters(chan)
    fav = bas.average_gate_fidelity(fk)
    print("  f_1 = %.8f   F_ave = %.8f   infidelity = %.3e"
          % (fk[1], fav, 1.0 - fav))
    assert 0.0 < 1.0 - fav < 1e-3, "Trotter residual is out of range"
    assert abs(np.abs(v @ v.conj().T - np.eye(2)).max()) < 1e-12
    print("  the residual is unitary, so p_0 + p_1 = 1 with no leakage: "
          "%.12f" % bas.error_rates(fk).sum())
    print("\nRB CHECKS PASSED")


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
    assert abs(got - want) < 5e-3

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

    print("\n" + "=" * 68)
    print("SU(2) SYNTHETIC RANDOMIZED BENCHMARKING (Fan et al 2026)")
    print("=" * 68)
    rb_selftest()

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

    rt = sub.add_parser("rb-tables",
                        help="zero-noise RB sample complexity, Tables 1-2")
    rt.add_argument("-j", type=float, default=3.5)

    rr = sub.add_parser("rb-rates",
                        help="exact SU(2) error rates of the paper channels")
    rr.add_argument("-j", type=float, default=3.5)
    rr.add_argument("--gamma-coherent", type=float, default=0.04)
    rr.add_argument("--gamma-dephasing", type=float, default=0.01)

    re_ = sub.add_parser("rb-run", help="simulate a synthetic RB experiment")
    re_.add_argument("-j", type=float, default=3.5)
    re_.add_argument("--protocol", default="ssr1",
                     choices=["ssrb", "sschi", "ssr1"])
    re_.add_argument("--channel", default="coherent",
                     choices=["coherent", "dephasing", "depolarizing"])
    re_.add_argument("--gamma", type=float, default=0.04)
    re_.add_argument("--phi", type=float, default=0.0,
                     help="SPAM error angle; 0 is perfect SPAM")
    re_.add_argument("--meas-mode", default="coherent",
                     choices=["coherent", "permute"])
    re_.add_argument("--circuits", type=int, default=200)
    re_.add_argument("--shots", type=int, default=0,
                     help="shots per circuit per input state; 0 = infinite")
    re_.add_argument("--seed", type=int, default=1337)

    rk = sub.add_parser("k4-rb",
                        help="benchmark the K4 step and recalibrate the "
                             "decoherence model from it")
    rk.add_argument("--circuits", type=int, default=400)
    rk.add_argument("--seed", type=int, default=1337)
    rk.add_argument("--nbits", type=int, default=0)
    rk.add_argument("--depol", type=float, default=0.0,
                    help="extra depolarizing noise on top of Trotter")
    rk.add_argument("--no-experiment", action="store_true",
                    help="exact twirl only, skip the simulated RB run")
    rk.add_argument("--no-anchor", action="store_true",
                    help="skip re-running the H2 anchor")

    ns = ap.parse_args()
    if ns.cmd == "anchor":
        run_h2_anchor(kmax=ns.kmax, ns=ns.ns, repeats=ns.repeats,
                      seed=ns.seed, backend=ns.backend,
                      nbits=(ns.nbits if ns.nbits else None),
                      noisy=not ns.no_noise,
                      calibration_state=ns.calibration_state)
    elif ns.cmd == "selftest":
        selftest()
    elif ns.cmd == "rb-tables":
        run_rb_tables(j=getattr(ns, "j"))
    elif ns.cmd == "rb-rates":
        run_rb_rates(j=getattr(ns, "j"),
                     gamma_coh=ns.gamma_coherent,
                     gamma_deph=ns.gamma_dephasing)
    elif ns.cmd == "rb-run":
        run_rb_experiment(j=getattr(ns, "j"), protocol=ns.protocol,
                          channel=ns.channel, gamma=ns.gamma, phi=ns.phi,
                          meas_mode=ns.meas_mode, n_circuits=ns.circuits,
                          shots=(ns.shots if ns.shots else None),
                          seed=ns.seed)
    elif ns.cmd == "k4-rb":
        run_k4_rb_anchor(nbits=(ns.nbits if ns.nbits else None),
                         run_experiment=not ns.no_experiment,
                         n_circuits=ns.circuits, seed=ns.seed,
                         depol=ns.depol, rerun_anchor=not ns.no_anchor)
    else:
        ap.print_help()
