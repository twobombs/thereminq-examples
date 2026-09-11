# -*- coding: us-ascii -*-
# qasm_qrack.py -- OpenQASM on PyQrack: backend, seam cutter, self-tests.
#
# One file, four parts:
#   1. QASM BACKEND      parse OpenQASM 2/3, drive QrackSimulator directly
#   2. SEAM RUNNER       cut a circuit at patch seams, run every chunk on its
#                        own PyQrack instance, compare with QrackAceBackend
#                        and an exact reference, synthesise the results
#   3. BACKEND SELFTEST  against Qiskit reference states   (needs qiskit)
#   4. SEAM SELFTEST     knit vs exact, truncation bound, ACE lowering
#
#   python3 qasm_qrack.py run   circuit.qasm --shots 4096      (part 1)
#   python3 qasm_qrack.py       circuit.qasm                   (same: 'run' is the default)
#   python3 qasm_qrack.py seams circuit.qasm --pauli "ZZ@0,5"  (part 2)
#   python3 qasm_qrack.py selftest [backend|seams|all] [-v|-q] (parts 3, 4)
#       progress on stderr: live line on a terminal, periodic lines in logs
#
# Runtime dependencies: pyqrack for part 1; pyqrack + numpy for part 2;
# qiskit additionally for part 3. Missing optional packages only disable
# the parts that need them.
#
# =====================================================================
# =====================================================================
# PART 1 -- QASM BACKEND
# =====================================================================
# =====================================================================
#
#
# WHY THIS EXISTS
# =====================================================================
# PyQrack has no QASM reader. The usual route is
#
#   .qasm -> qiskit.qasm3.loads -> QrackCircuit.in_from_qiskit_circuit
#
# which drags in Qiskit plus the ANTLR importer, and lands on a
# QrackCircuit that cannot measure. This file parses QASM itself and
# drives a live QrackSimulator, so measurement, reset and classical
# feed-forward all work, and the only runtime dependency is pyqrack.
# numpy is not needed for this part.
#
# WHAT IS SUPPORTED
# =====================================================================
#   headers        OPENQASM 2.0 / 3.0, include "stdgates.inc"/"qelib1.inc"
#                  (built in), other includes resolved from disk
#   declarations   qubit[n] q; qubit q; qreg q[n]; bit[n] c; creg c[n];
#                  $n physical qubits; const/float/int/angle scalars;
#                  input scalars, bound from params=
#   gates          stdgates.inc and the usual qelib1 extras, U, CX,
#                  gphase, user 'gate' definitions (recursive)
#   modifiers      ctrl(n) @, negctrl(n) @, inv @, pow(k) @ -- in any
#                  order, on any gate incl. user gates and gphase
#   operands       q, q[i], q[-1], q[a:b], q[a:s:b], q[{i,j}];
#                  register broadcast
#   classical      measure (both syntaxes), reset, barrier,
#                  if (c == n) / if (c[i]) / if (!c[i]) with else
#   ignored        delay, box (inlined), #pragma, cal/defcal blocks
#   not supported  def, for, while, switch, classical arithmetic --
#                  these raise with the line number, never skip silently
#
# CONVENTIONS -- pinned against exact matrices by test_qasm_qrack.py
# =====================================================================
#   U(t,p,l)   [[cos t/2, -e^il sin t/2], [e^ip sin t/2, e^i(p+l) cos t/2]]
#   rz(l)      diag(e^-il/2, e^il/2)  == Qrack r(PauliZ, l), exactly
#   p(l)       diag(1, e^il)
#   rzz(t)     exp(-i t/2 Z(x)Z), likewise rxx / ryy / rzx
#   gphase     unobservable alone; under ctrl it becomes a phase gate,
#              so ctrl @ (user gate containing gphase) is exact
#   qubits     global index = declaration order; out_ket little-endian
#   counts     Qiskit style: registers joined by spaces, last-declared
#              leftmost, bit 0 of each register rightmost
#
# HOW IT RUNS
# =====================================================================
# The circuit compiles to a flat list of primitive ops. Gates keep a
# name tag where one exists, so h / s / t / cx / rz / ... reach Qrack
# as native calls (keeping the Clifford+RZ stabilizer-hybrid path
# live) and only genuinely arbitrary unitaries go through mtrx.
#
# The leading unitary part runs ONCE. Then:
#   * only measurements remain  -> exact P(1) per measured bit, and all
#                                  shots from one measure_shots() call
#   * anything dynamic remains  -> clone() the prepared state per shot
#                                  and run the tail on the clone
#
# seed= makes sampling reproducible. Qrack's own seed() does not govern
# its measurement RNG (m() and measure_shots() differ run to run even
# when seeded), so seeded runs draw each outcome in Python from the
# exact prob() and commit it with force_m(). That is exact sampling,
# just per shot rather than one measure_shots() call.
#
# Qrack's exp() would be the natural primitive for rxx/ryy/rzz but is
# broken in current wheels (int* vs ulonglong* in the ctypes binding;
# passing the right array type segfaults), so those decompose into
# basis change + CX ladder + RZ.
#
#   python3 qasm_qrack.py run circuit.qasm --shots 4096
#   python3 qasm_qrack.py run qasm/*.qasm --shots 0   # exact probabilities only
#   python3 qasm_qrack.py run c.qasm --param theta=0.1 --device 2 --json

import argparse
import cmath
import heapq
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict, deque

try:                                    # parts 2-4 only
    import numpy as np
except ImportError:
    np = None

__all__ = ["QasmError", "Program", "Result", "compile_qasm", "QrackBackend",
           "run_qasm", "SeamSplit", "knit", "AceRunner", "reference",
           "parse_pauli"]


class QasmError(Exception):
    pass


# =====================================================================
# 2x2 MATRICES, pure python, row-major (a, b, c, d)
# =====================================================================

R2 = 1.0 / math.sqrt(2.0)
MX = (0, 1, 1, 0)
MY = (0, -1j, 1j, 0)
MZ = (1, 0, 0, -1)
MH = (R2, R2, R2, -R2)
MS = (1, 0, 0, 1j)
MSDG = (1, 0, 0, -1j)
MT = (1, 0, 0, cmath.exp(1j * math.pi / 4))
MTDG = (1, 0, 0, cmath.exp(-1j * math.pi / 4))
MSX = (0.5 + 0.5j, 0.5 - 0.5j, 0.5 - 0.5j, 0.5 + 0.5j)
MSXDG = (0.5 - 0.5j, 0.5 + 0.5j, 0.5 + 0.5j, 0.5 - 0.5j)

FIXED = {"x": MX, "y": MY, "z": MZ, "h": MH, "s": MS, "sdg": MSDG,
         "t": MT, "tdg": MTDG, "sx": MSX, "sxdg": MSXDG}
INV_TAG = {"x": "x", "y": "y", "z": "z", "h": "h", "s": "sdg", "sdg": "s",
           "t": "tdg", "tdg": "t", "sx": "sxdg", "sxdg": "sx"}


def m_U(t, p, l):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return (c, -cmath.exp(1j * l) * s, cmath.exp(1j * p) * s,
            cmath.exp(1j * (p + l)) * c)


def m_rx(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return (c, -1j * s, -1j * s, c)


def m_ry(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return (c, -s, s, c)


def m_rz(t):
    return (cmath.exp(-0.5j * t), 0, 0, cmath.exp(0.5j * t))


def m_p(l):
    return (1, 0, 0, cmath.exp(1j * l))


ROT = {"rx": m_rx, "ry": m_ry, "rz": m_rz, "p": m_p}


def m_dag(m):
    a, b, c, d = m
    return (complex(a).conjugate(), complex(c).conjugate(),
            complex(b).conjugate(), complex(d).conjugate())


def m_close(m, n, tol=1e-12):
    return all(abs(complex(x) - complex(y)) < tol for x, y in zip(m, n))


def m_pow(m, k):
    """Principal-branch power of a 2x2 unitary."""
    a, b, c, d = (complex(x) for x in m)
    tr, det = a + d, a * d - b * c
    disc = cmath.sqrt(tr * tr / 4 - det)
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    if abs(l1 - l2) < 1e-12:
        f = l1 ** k
        return (f, 0, 0, f)
    f1, f2 = l1 ** k, l2 ** k
    # M^k = f1 (M - l2 I)/(l1 - l2) + f2 (M - l1 I)/(l2 - l1)
    g = 1 / (l1 - l2)
    return (g * (f1 * (a - l2) - f2 * (a - l1)), g * (f1 - f2) * b,
            g * (f1 - f2) * c, g * (f1 * (d - l2) - f2 * (d - l1)))


def recognise(m):
    for nm, f in FIXED.items():
        if m_close(m, f):
            return nm
    return None


# =====================================================================
# IR
#   ("g",  tag, M, target, ctrls, cvals)   single-target unitary
#   ("gp", theta, ctrls, cvals)            global phase e^{i theta}
#   ("sw", a, b, ctrls, cvals)             swap
#   ("m",  qubit, clbit)                   clbit may be None
#   ("r",  qubit)                          reset
#   ("if", bits, op, value, then, else)    classical condition
# tag: fixed name, (rot, theta) for rx/ry/rz/p, or None
# =====================================================================

def G(tag, m, t):
    return ("g", tag, m, t, (), ())


def G_fixed(nm, t):
    return ("g", nm, FIXED[nm], t, (), ())


def G_rot(nm, th, t):
    return ("g", (nm, th), ROT[nm](th), t, (), ())


def G_gen(m, t):
    return ("g", recognise(m), m, t, (), ())


def add_ctrl(ops, cs, vs):
    cs, vs = tuple(cs), tuple(vs)
    out = []
    for op in ops:
        k = op[0]
        if k == "g":
            out.append(("g", op[1], op[2], op[3], cs + op[4], vs + op[5]))
        elif k == "gp":
            out.append(("gp", op[1], cs + op[2], vs + op[3]))
        elif k == "sw":
            out.append(("sw", op[1], op[2], cs + op[3], vs + op[4]))
        else:
            raise QasmError("cannot control a non-unitary operation")
    return out


def inverse(ops):
    out = []
    for op in reversed(ops):
        k = op[0]
        if k == "g":
            tag = op[1]
            if isinstance(tag, tuple):
                tag = (tag[0], -tag[1])
            elif tag is not None:
                tag = INV_TAG[tag]
            out.append(("g", tag, m_dag(op[2]), op[3], op[4], op[5]))
        elif k == "gp":
            out.append(("gp", -op[1], op[2], op[3]))
        elif k == "sw":
            out.append(op)
        else:
            raise QasmError("cannot invert a non-unitary operation")
    return out


def power(ops, k, line):
    if abs(k - round(k)) < 1e-12:
        k = int(round(k))
        base = ops if k >= 0 else inverse(ops)
        return base * abs(k)
    if len(ops) != 1 or ops[0][0] not in ("g", "gp"):
        raise QasmError("line %d: non-integer pow() is only supported on "
                        "gates that reduce to one single-qubit unitary" % line)
    op = ops[0]
    if op[0] == "gp":
        return [("gp", op[1] * k, op[2], op[3])]
    tag = op[1]
    if isinstance(tag, tuple):
        return [("g", (tag[0], tag[1] * k), ROT[tag[0]](tag[1] * k),
                 op[3], op[4], op[5])]
    m = m_pow(op[2], k)
    return [("g", recognise(m), m, op[3], op[4], op[5])]


# =====================================================================
# BUILT-IN GATES   name -> (n_params, n_qubits, lower(params, qubits))
# =====================================================================

def _c(ops, *cs):
    return add_ctrl(ops, cs, (1,) * len(cs))


def _cx(a, b):
    return _c([G_fixed("x", b)], a)


def _rzz(t, a, b):
    return _cx(a, b) + [G_rot("rz", t, b)] + _cx(a, b)


def _rxx(t, a, b):
    hh = [G_fixed("h", a), G_fixed("h", b)]
    return hh + _rzz(t, a, b) + hh


def _ryy(t, a, b):
    pre = [G_rot("rx", math.pi / 2, a), G_rot("rx", math.pi / 2, b)]
    return pre + _rzz(t, a, b) + inverse(pre)


def _rzx(t, a, b):
    return [G_fixed("h", b)] + _rzz(t, a, b) + [G_fixed("h", b)]


def _fixed(nm):
    return (0, 1, lambda p, q: [G_fixed(nm, q[0])])


def _rot(nm):
    return (1, 1, lambda p, q: [G_rot(nm, p[0], q[0])])


def _crot(nm):
    return (1, 2, lambda p, q: _c([G_rot(nm, p[0], q[1])], q[0]))


def _cfixed(nm, n=1):
    return (0, n + 1, lambda p, q: _c([G_fixed(nm, q[n])], *q[:n]))


PI = math.pi
BUILTIN = {
    "id": (0, 1, lambda p, q: []),
    "x": _fixed("x"), "y": _fixed("y"), "z": _fixed("z"), "h": _fixed("h"),
    "s": _fixed("s"), "sdg": _fixed("sdg"), "t": _fixed("t"),
    "tdg": _fixed("tdg"), "sx": _fixed("sx"), "sxdg": _fixed("sxdg"),
    "rx": _rot("rx"), "ry": _rot("ry"), "rz": _rot("rz"),
    "p": _rot("p"), "phase": _rot("p"), "u1": _rot("p"),
    "U": (3, 1, lambda p, q: [G_gen(m_U(*p), q[0])]),
    "u": (3, 1, lambda p, q: [G_gen(m_U(*p), q[0])]),
    "u3": (3, 1, lambda p, q: [G_gen(m_U(*p), q[0])]),
    "u2": (2, 1, lambda p, q: [G_gen(m_U(PI / 2, p[0], p[1]), q[0])]),
    "r": (2, 1, lambda p, q: [G_gen(m_U(p[0], p[1] - PI / 2,
                                       PI / 2 - p[1]), q[0])]),
    "cx": _cfixed("x"), "CX": _cfixed("x"), "cnot": _cfixed("x"),
    "cy": _cfixed("y"), "cz": _cfixed("z"), "ch": _cfixed("h"),
    "cs": _cfixed("s"), "csdg": _cfixed("sdg"),
    "csx": (0, 2, lambda p, q: _c([G_fixed("sx", q[1])], q[0])),
    "crx": _crot("rx"), "cry": _crot("ry"), "crz": _crot("rz"),
    "cp": _crot("p"), "cphase": _crot("p"), "cu1": _crot("p"),
    "cu3": (3, 2, lambda p, q: _c([G_gen(m_U(*p), q[1])], q[0])),
    "cu": (4, 2, lambda p, q: _c([("gp", p[3], (), ()),
                                  G_gen(m_U(p[0], p[1], p[2]), q[1])], q[0])),
    "ccx": _cfixed("x", 2), "toffoli": _cfixed("x", 2),
    "ccz": _cfixed("z", 2), "c3x": _cfixed("x", 3), "c4x": _cfixed("x", 4),
    "swap": (0, 2, lambda p, q: [("sw", q[0], q[1], (), ())]),
    "cswap": (0, 3, lambda p, q: [("sw", q[1], q[2], (q[0],), (1,))]),
    "fredkin": (0, 3, lambda p, q: [("sw", q[1], q[2], (q[0],), (1,))]),
    "iswap": (0, 2, lambda p, q: [G_fixed("s", q[0]), G_fixed("s", q[1]),
                                  G_fixed("h", q[0])] + _cx(q[0], q[1])
              + _cx(q[1], q[0]) + [G_fixed("h", q[1])]),
    "dcx": (0, 2, lambda p, q: _cx(q[0], q[1]) + _cx(q[1], q[0])),
    "ecr": (0, 2, lambda p, q: _rzx(PI / 4, q[0], q[1])
            + [G_fixed("x", q[0])] + _rzx(-PI / 4, q[0], q[1])),
    "rzz": (1, 2, lambda p, q: _rzz(p[0], q[0], q[1])),
    "rxx": (1, 2, lambda p, q: _rxx(p[0], q[0], q[1])),
    "ryy": (1, 2, lambda p, q: _ryy(p[0], q[0], q[1])),
    "rzx": (1, 2, lambda p, q: _rzx(p[0], q[0], q[1])),
    "gphase": (1, 0, lambda p, q: [("gp", p[0], (), ())]),
}


# =====================================================================
# LEXER
# =====================================================================

_TOK = re.compile(r"""
   (?P<ws>\s+)
  |(?P<lc>//[^\n]*)
  |(?P<bc>/\*.*?\*/)
  |(?P<str>"[^"]*")
  |(?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  |(?P<id>\$\d+|[A-Za-z_\u0080-\uffff][A-Za-z0-9_\u0080-\uffff]*)
  |(?P<op>->|==|!=|<=|>=|\*\*|&&|\|\||[@;,()\[\]{}+\-*/%^=<>!~&|:])
""", re.X | re.S)


def lex(src):
    src = re.sub(r"(?m)^[ \t]*#.*$", "", src)      # pragmas / annotations
    toks, pos, line = [], 0, 1
    while pos < len(src):
        m = _TOK.match(src, pos)
        if not m:
            raise QasmError("line %d: unexpected character %r"
                            % (line, src[pos]))
        kind, text = m.lastgroup, m.group()
        if kind not in ("ws", "lc", "bc"):
            toks.append((kind, text, line))
        line += text.count("\n")
        pos = m.end()
    toks.append(("eof", "", line))
    return toks


# =====================================================================
# EXPRESSIONS
# =====================================================================

CONSTS = {"pi": math.pi, "\u03c0": math.pi, "tau": 2 * math.pi,
          "\u03c4": 2 * math.pi, "euler": math.e, "\u2107": math.e}
FUNCS = {"sin": math.sin, "cos": math.cos, "tan": math.tan,
         "arcsin": math.asin, "arccos": math.acos, "arctan": math.atan,
         "asin": math.asin, "acos": math.acos, "atan": math.atan,
         "exp": math.exp, "ln": math.log, "log": math.log,
         "sqrt": math.sqrt, "abs": abs, "floor": math.floor,
         "ceiling": math.ceil, "ceil": math.ceil}


def ev(e, env):
    k = e[0]
    if k == "num":
        return e[1]
    if k == "var":
        if e[1] in env:
            return env[e[1]]
        if e[1] in CONSTS:
            return CONSTS[e[1]]
        raise QasmError("line %d: unknown identifier '%s'" % (e[2], e[1]))
    if k == "neg":
        return -ev(e[1], env)
    if k == "call":
        return FUNCS[e[1]](*[ev(a, env) for a in e[2]])
    a, b = ev(e[2], env), ev(e[3], env)
    op = e[1]
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a / b
    if op == "%":
        return math.fmod(a, b)
    return a ** b


# =====================================================================
# PROGRAM / COMPILER
# =====================================================================

class Program:
    """Compiled circuit: flat op list plus register layout."""

    def __init__(self):
        self.ops = []
        self.qregs = {}          # name -> list of global qubit indices
        self.cregs = {}          # name -> list of global clbit indices
        self.creg_order = []
        self.num_qubits = 0
        self.num_clbits = 0
        self.version = None

    def gate_counts(self):
        c = Counter()

        def walk(ops):
            for op in ops:
                if op[0] == "if":
                    walk(op[4])
                    walk(op[5])
                    continue
                if op[0] == "g":
                    t = op[1]
                    nm = t[0] if isinstance(t, tuple) else (t or "mtrx")
                    c["c" * len(op[4]) + nm] += 1
                else:
                    c[{"gp": "gphase", "sw": "swap", "m": "measure",
                       "r": "reset"}[op[0]]] += 1
        walk(self.ops)
        return dict(c)


class _Compiler:
    KEYWORDS_UNSUPPORTED = {"def", "for", "while", "switch", "extern", "let",
                            "return", "break", "continue", "end"}
    SCALAR_TYPES = {"float", "int", "uint", "angle", "bool", "complex"}

    def __init__(self, src, params, path, depth=0):
        self.t = lex(src)
        self.i = 0
        self.params = dict(params or {})
        self.path = path
        self.depth = depth
        self.prog = Program()
        self.env = {}
        self.gates = {}          # name -> (params, args, body)
        self.opaque = set()
        self.phys = {}

    # ---- token helpers ---------------------------------------------
    def peek(self, k=0):
        return self.t[self.i + k]

    def nxt(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def accept(self, text):
        if self.t[self.i][1] == text and self.t[self.i][0] != "str":
            self.i += 1
            return True
        return False

    def expect(self, text):
        tok = self.nxt()
        if tok[1] != text:
            raise QasmError("line %d: expected '%s', got '%s'"
                            % (tok[2], text, tok[1]))
        return tok

    def ident(self):
        tok = self.nxt()
        if tok[0] != "id":
            raise QasmError("line %d: expected identifier, got '%s'"
                            % (tok[2], tok[1]))
        return tok[1]

    def skip_to(self, text):
        while self.peek()[1] != text and self.peek()[0] != "eof":
            self.i += 1
        self.expect(text)

    def skip_block(self):
        self.skip_to("{")
        depth = 1
        while depth:
            tok = self.nxt()
            if tok[0] == "eof":
                raise QasmError("unterminated block")
            depth += {"{": 1, "}": -1}.get(tok[1], 0)

    # ---- expressions -------------------------------------------------
    def expr(self):
        e = self.term()
        while self.peek()[1] in ("+", "-"):
            op = self.nxt()[1]
            e = ("bin", op, e, self.term())
        return e

    def term(self):
        e = self.unary()
        while self.peek()[1] in ("*", "/", "%"):
            op = self.nxt()[1]
            e = ("bin", op, e, self.unary())
        return e

    def unary(self):
        if self.accept("-"):
            return ("neg", self.unary())
        if self.accept("+"):
            return self.unary()
        return self.powr()

    def powr(self):
        b = self.atom()
        if self.peek()[1] in ("**", "^"):
            self.nxt()
            return ("bin", "**", b, self.unary())
        return b

    def atom(self):
        tok = self.nxt()
        if tok[0] == "num":
            return ("num", float(tok[1]))
        if tok[1] == "(":
            e = self.expr()
            self.expect(")")
            return e
        if tok[0] == "id":
            if tok[1] in FUNCS and self.peek()[1] == "(":
                self.nxt()
                args = [self.expr()]
                while self.accept(","):
                    args.append(self.expr())
                self.expect(")")
                return ("call", tok[1], args)
            return ("var", tok[1], tok[2])
        raise QasmError("line %d: bad expression at '%s'" % (tok[2], tok[1]))

    def cexpr(self, env=None):
        return ev(self.expr(), self.env if env is None else env)

    # ---- operands ----------------------------------------------------
    def operand(self):
        """-> (name, index_spec or None, line)."""
        tok = self.nxt()
        if tok[0] != "id":
            raise QasmError("line %d: expected operand, got '%s'"
                            % (tok[2], tok[1]))
        spec = None
        if self.accept("["):
            if self.accept("{"):
                items = [self.expr()]
                while self.accept(","):
                    items.append(self.expr())
                self.expect("}")
                spec = ("set", items)
            else:
                a = self.expr()
                if self.accept(":"):
                    b = self.expr()
                    if self.accept(":"):
                        spec = ("slice", a, b, self.expr())
                    else:
                        spec = ("slice", a, ("num", 1.0), b)
                else:
                    spec = ("idx", a)
            self.expect("]")
        return (tok[1], spec, tok[2])

    def resolve(self, opnd, regs, what, local=None):
        name, spec, line = opnd
        if local is not None:
            if spec is not None or name not in local:
                raise QasmError("line %d: gate bodies may only use their "
                                "own qubit arguments ('%s')" % (line, name))
            return [local[name]]
        if what == "qubit" and name.startswith("$"):
            if name not in self.phys:
                self.phys[name] = self._alloc_q(name, 1)[0]
            return [self.phys[name]]
        if name not in regs:
            raise QasmError("line %d: unknown %s register '%s'"
                            % (line, what, name))
        reg = regs[name]
        if spec is None:
            return list(reg)

        def ix(e):
            v = int(round(ev(e, self.env)))
            if v < 0:
                v += len(reg)
            if not 0 <= v < len(reg):
                raise QasmError("line %d: index %d out of range for '%s'[%d]"
                                % (line, v, name, len(reg)))
            return v
        if spec[0] == "idx":
            return [reg[ix(spec[1])]]
        if spec[0] == "set":
            return [reg[ix(e)] for e in spec[1]]
        a, st, b = ix(spec[1]), int(round(ev(spec[2], self.env))), ix(spec[3])
        return [reg[k] for k in range(a, b + (1 if st > 0 else -1), st)]

    # ---- allocation --------------------------------------------------
    def _alloc_q(self, name, n):
        if name in self.prog.qregs:
            raise QasmError("qubit register '%s' declared twice" % name)
        base = self.prog.num_qubits
        self.prog.qregs[name] = list(range(base, base + n))
        self.prog.num_qubits += n
        return self.prog.qregs[name]

    def _alloc_c(self, name, n):
        if name in self.prog.cregs:
            raise QasmError("bit register '%s' declared twice" % name)
        base = self.prog.num_clbits
        self.prog.cregs[name] = list(range(base, base + n))
        self.prog.creg_order.append(name)
        self.prog.num_clbits += n

    # ---- statements --------------------------------------------------
    def run(self):
        while self.peek()[0] != "eof":
            self.prog.ops += self.statement()
        return self.prog

    def block_or_stmt(self):
        if self.accept("{"):
            ops = []
            while not self.accept("}"):
                ops += self.statement()
            return ops
        return self.statement()

    def statement(self):
        tok = self.peek()
        w, line = tok[1], tok[2]
        if tok[0] != "id":
            if w == ";":
                self.nxt()
                return []
            raise QasmError("line %d: unexpected '%s'" % (line, w))

        if w == "OPENQASM":
            self.nxt()
            self.prog.version = self.nxt()[1]
            self.expect(";")
            return []
        if w == "include":
            self.nxt()
            fn = self.nxt()[1].strip('"')
            self.expect(";")
            return self.include(fn, line)
        if w in ("qubit", "qreg", "bit", "creg"):
            return self.declare(w)
        if w == "const" or w in self.SCALAR_TYPES:
            return self.scalar_decl()
        if w == "input":
            self.nxt()
            self.skip_type()
            name = self.ident()
            self.expect(";")
            if name not in self.params:
                raise QasmError("line %d: input '%s' has no value; pass "
                                "params={'%s': ...}" % (line, name, name))
            self.env[name] = float(self.params[name])
            return []
        if w == "output":
            raise QasmError("line %d: 'output' is not supported" % line)
        if w == "gate":
            self.gate_def()
            return []
        if w == "opaque":
            self.nxt()
            self.opaque.add(self.ident())
            self.skip_to(";")
            return []
        if w in ("barrier", "delay"):
            self.skip_to(";")
            return []
        if w == "box":
            self.nxt()
            if self.accept("["):
                self.skip_to("]")
            return self.block_or_stmt()
        if w in ("cal", "defcal", "defcalgrammar"):
            if w == "defcalgrammar":
                self.skip_to(";")
            else:
                self.skip_block()
            return []
        if w == "measure":
            self.nxt()
            q = self.operand()
            if self.accept("->"):
                c = self.operand()
                self.expect(";")
                return self.measure(q, c, line)
            self.expect(";")
            return self.measure(q, None, line)
        if w == "reset":
            self.nxt()
            qs = self.resolve(self.operand(), self.prog.qregs, "qubit")
            self.expect(";")
            return [("r", q) for q in qs]
        if w == "if":
            return self.if_stmt()
        if w in self.KEYWORDS_UNSUPPORTED:
            raise QasmError("line %d: '%s' (classical control flow / "
                            "subroutines) is not supported" % (line, w))
        # assignment form: c = measure q;  c[i] = measure q[j];
        if w in self.prog.cregs:
            save = self.i
            c = self.operand()
            if self.accept("="):
                if not self.accept("measure"):
                    raise QasmError("line %d: classical assignment is not "
                                    "supported (only '= measure')" % line)
                q = self.operand()
                self.expect(";")
                return self.measure(q, c, line)
            self.i = save
        return self.gate_call(None)

    def skip_type(self):
        self.nxt()
        if self.accept("["):
            self.skip_to("]")

    def declare(self, w):
        line = self.nxt()[2]
        if w in ("qubit", "bit"):
            n = 1
            if self.accept("["):
                n = int(self.cexpr())
                self.expect("]")
            name = self.ident()
        else:
            name = self.ident()
            self.expect("[")
            n = int(self.cexpr())
            self.expect("]")
        if self.peek()[1] == "=":
            raise QasmError("line %d: initialised declarations are not "
                            "supported" % line)
        self.expect(";")
        (self._alloc_q if w in ("qubit", "qreg") else self._alloc_c)(name, n)
        return []

    def scalar_decl(self):
        line = self.peek()[2]
        self.accept("const")
        self.skip_type()
        name = self.ident()
        if not self.accept("="):
            raise QasmError("line %d: scalar '%s' needs an initialiser"
                            % (line, name))
        self.env[name] = self.cexpr()
        self.expect(";")
        return []

    def include(self, fn, line):
        if os.path.basename(fn) in ("stdgates.inc", "qelib1.inc"):
            return []
        base = os.path.dirname(self.path) if self.path else "."
        full = fn if os.path.isabs(fn) else os.path.join(base, fn)
        if not os.path.exists(full):
            raise QasmError("line %d: include '%s' not found" % (line, fn))
        if self.depth > 16:
            raise QasmError("include nesting too deep")
        sub = _Compiler(open(full).read(), self.params, full, self.depth + 1)
        sub.prog, sub.env, sub.gates = self.prog, self.env, self.gates
        sub.opaque, sub.phys = self.opaque, self.phys
        ops = []
        while sub.peek()[0] != "eof":
            ops += sub.statement()
        return ops

    def gate_def(self):
        line = self.nxt()[2]
        name = self.ident()
        ps = []
        if self.accept("("):
            if not self.accept(")"):
                ps.append(self.ident())
                while self.accept(","):
                    ps.append(self.ident())
                self.expect(")")
        args = [self.ident()]
        while self.accept(","):
            args.append(self.ident())
        self.expect("{")
        body = []
        while not self.accept("}"):
            tok = self.peek()
            if tok[1] == "barrier":
                self.skip_to(";")
                continue
            if tok[0] == "eof":
                raise QasmError("line %d: unterminated gate '%s'"
                                % (line, name))
            body.append(self.parse_call())
        self.gates[name] = (ps, args, body)

    def parse_call(self):
        """-> (mods, name, param_exprs, operands, line)."""
        mods = []
        line = self.peek()[2]
        while self.peek()[1] in ("ctrl", "negctrl", "inv", "pow") and \
                self.peek(1)[1] in ("@", "("):
            m = self.nxt()[1]
            if m == "inv":
                mods.append(("inv",))
            elif m == "pow":
                self.expect("(")
                mods.append(("pow", self.expr()))
                self.expect(")")
            else:
                n = ("num", 1.0)
                if self.accept("("):
                    n = self.expr()
                    self.expect(")")
                mods.append((m, n))
            self.expect("@")
        name = self.ident()
        pe = []
        if self.accept("("):
            if not self.accept(")"):
                pe.append(self.expr())
                while self.accept(","):
                    pe.append(self.expr())
                self.expect(")")
        if self.accept("["):                   # duration designator
            self.skip_to("]")
        opnds = []
        if self.peek()[1] != ";":
            opnds.append(self.operand())
            while self.accept(","):
                opnds.append(self.operand())
        self.expect(";")
        return (mods, name, pe, opnds, line)

    def gate_call(self, _):
        return self.expand_call(self.parse_call(), self.env, None)

    def expand_call(self, call, env, local):
        mods, name, pe, opnds, line = call
        params = [ev(e, env) for e in pe]
        lists = [self.resolve(o, self.prog.qregs, "qubit", local)
                 for o in opnds]
        width = {len(x) for x in lists if len(x) != 1}
        if len(width) > 1:
            raise QasmError("line %d: broadcast over registers of different "
                            "sizes" % line)
        n = width.pop() if width else 1
        out = []
        for k in range(n):
            qs = [x[k] if len(x) > 1 else x[0] for x in lists]
            if len(set(qs)) != len(qs):
                raise QasmError("line %d: repeated qubit in '%s'"
                                % (line, name))
            out += self.apply_mods(mods, name, params, qs, env, line)
        return out

    def apply_mods(self, mods, name, params, qs, env, line):
        pos, bound = 0, []
        for m in mods:
            if m[0] in ("ctrl", "negctrl"):
                k = int(round(ev(m[1], env)))
                bound.append((m[0], qs[pos:pos + k]))
                pos += k
            elif m[0] == "pow":
                bound.append(("pow", ev(m[1], env)))
            else:
                bound.append(("inv",))
        ops = self.base_gate(name, params, qs[pos:], line)
        for m in reversed(bound):
            if m[0] == "inv":
                ops = inverse(ops)
            elif m[0] == "pow":
                ops = power(ops, m[1], line)
            else:
                v = 1 if m[0] == "ctrl" else 0
                ops = add_ctrl(ops, m[1], [v] * len(m[1]))
        return ops

    def base_gate(self, name, params, qs, line):
        if name in self.gates:
            ps, args, body = self.gates[name]
            if len(params) != len(ps) or len(qs) != len(args):
                raise QasmError("line %d: gate '%s' takes %d params and %d "
                                "qubits, got %d and %d" % (
                                    line, name, len(ps), len(args),
                                    len(params), len(qs)))
            if self.depth > 64:
                raise QasmError("line %d: gate recursion too deep" % line)
            env = dict(self.env)
            env.update(zip(ps, params))
            local = dict(zip(args, qs))
            self.depth += 1
            try:
                ops = []
                for c in body:
                    ops += self.expand_call(c, env, local)
            finally:
                self.depth -= 1
            return ops
        if name in BUILTIN:
            np_, nq, fn = BUILTIN[name]
            if len(params) != np_ or len(qs) != nq:
                raise QasmError("line %d: '%s' takes %d params and %d "
                                "qubits, got %d and %d" % (
                                    line, name, np_, nq, len(params), len(qs)))
            return fn(params, qs)
        if name in self.opaque:
            raise QasmError("line %d: opaque gate '%s' has no definition"
                            % (line, name))
        raise QasmError("line %d: unknown gate '%s'" % (line, name))

    def measure(self, q, c, line):
        qs = self.resolve(q, self.prog.qregs, "qubit")
        if c is None:
            return [("m", x, None) for x in qs]
        cs = self.resolve(c, self.prog.cregs, "bit")
        if len(qs) != len(cs):
            raise QasmError("line %d: measure width mismatch (%d qubits, "
                            "%d bits)" % (line, len(qs), len(cs)))
        return [("m", a, b) for a, b in zip(qs, cs)]

    def if_stmt(self):
        self.nxt()
        self.expect("(")
        neg = self.accept("!")
        bits = self.resolve(self.operand(), self.prog.cregs, "bit")
        op, val = "!=", 0.0
        if self.peek()[1] in ("==", "!=", "<", ">", "<=", ">="):
            op = self.nxt()[1]
            val = self.cexpr()
        if self.peek()[1] in ("true", "false"):
            val = float(self.nxt()[1] == "true")
        self.expect(")")
        if neg:
            op, val = "==", 0.0
        then = self.block_or_stmt()
        other = []
        if self.accept("else"):
            other = self.block_or_stmt()
        return [("if", tuple(bits), op, int(round(val)), then, other)]


def compile_qasm(src, params=None, path=None):
    """QASM text -> Program. path is used only to resolve includes."""
    c = _Compiler(src, params, path)
    prog = c.run()
    if prog.num_qubits == 0:
        raise QasmError("program declares no qubits")
    return prog


# =====================================================================
# EXECUTION
# =====================================================================

_SIMPLE = {"x": "x", "y": "y", "z": "z", "h": "h", "s": "s", "sdg": "adjs",
           "t": "t", "tdg": "adjt", "sx": "sx", "sxdg": "adjsx"}
_MC = {"x": "mcx", "y": "mcy", "z": "mcz", "h": "mch", "s": "mcs",
       "sdg": "mcadjs", "t": "mct", "tdg": "mcadjt"}
_MAC = {"x": "macx", "y": "macy", "z": "macz", "h": "mach", "s": "macs",
        "sdg": "macadjs", "t": "mact", "tdg": "macadjt"}
_UNITARY = ("g", "gp", "sw")


class Result:
    def __init__(self):
        self.counts = {}
        self.probabilities = {}      # "c[0]" -> exact P(1), terminal only
        self.statevector = None
        self.statevector_is_final = None
        self.shots = 0
        self.mode = None
        self.num_qubits = 0
        self.gate_counts = {}
        self.timings = {}
        self.simulator = None        # prepared pre-measurement state

    def expectation_z(self):
        """<Z> = 1 - 2 P(1) per measured bit; for a Hadamard-test ancilla
        this is Re<W> directly, with no shot noise."""
        return {k: 1.0 - 2.0 * p for k, p in self.probabilities.items()}

    def as_dict(self):
        return {"counts": self.counts, "probabilities": self.probabilities,
                "expectation_z": self.expectation_z(), "shots": self.shots,
                "mode": self.mode, "num_qubits": self.num_qubits,
                "gate_counts": self.gate_counts, "timings": self.timings}


class QrackBackend:
    """Run QASM on QrackSimulator.

    sim_kwargs go straight to QrackSimulator(...), e.g. is_gpu=False,
    is_stabilizer_hybrid=True. device selects the OpenCL device index.
    """

    def __init__(self, device=None, **sim_kwargs):
        from pyqrack import QrackSimulator, Pauli
        self._QS = QrackSimulator
        self._P = {"rx": Pauli.PauliX, "ry": Pauli.PauliY,
                   "rz": Pauli.PauliZ}
        self.device = device
        self.sim_kwargs = sim_kwargs

    # ---- primitive dispatch -----------------------------------------
    def _gate(self, sim, tag, m, t, cs, vs):
        rot = isinstance(tag, tuple) and tag[0] in self._P
        if not cs:
            if tag in _SIMPLE:
                getattr(sim, _SIMPLE[tag])(t)
            elif rot:
                sim.r(self._P[tag[0]], tag[1], t)
            else:
                sim.mtrx(list(m), t)
            return
        cl = list(cs)
        if all(vs):
            if tag in _MC:
                getattr(sim, _MC[tag])(cl, t)
            elif rot:
                sim.mcr(self._P[tag[0]], tag[1], cl, t)
            else:
                sim.mcmtrx(cl, list(m), t)
        elif not any(vs):
            if tag in _MAC:
                getattr(sim, _MAC[tag])(cl, t)
            else:
                sim.macmtrx(cl, list(m), t)
        else:
            perm = sum(v << k for k, v in enumerate(vs))
            sim.ucmtrx(cl, list(m), t, perm)

    @staticmethod
    def _meas(sim, q, rng):
        if rng is None:
            return int(sim.m(q))
        r = rng.random() < sim.prob(q)
        sim.force_m(q, r)
        return int(r)

    def _exec(self, sim, ops, cbits, rng=None):
        for op in ops:
            k = op[0]
            if k == "g":
                self._gate(sim, op[1], op[2], op[3], op[4], op[5])
            elif k == "m":
                r = self._meas(sim, op[1], rng)
                if op[2] is not None:
                    cbits[op[2]] = r
            elif k == "gp":
                cs, vs = op[2], op[3]
                if cs:
                    e = cmath.exp(1j * op[1])
                    m = (1, 0, 0, e) if vs[-1] else (e, 0, 0, 1)
                    self._gate(sim, None, m, cs[-1], cs[:-1], vs[:-1])
            elif k == "sw":
                a, b, cs, vs = op[1:]
                if not cs:
                    sim.swap(a, b)
                elif all(vs):
                    sim.cswap(list(cs), a, b)
                elif not any(vs):
                    sim.acswap(list(cs), a, b)
                else:
                    for x, y in ((a, b), (b, a), (a, b)):
                        self._gate(sim, "x", MX, y, cs + (x,), vs + (1,))
            elif k == "r":
                if self._meas(sim, op[1], rng):
                    sim.x(op[1])
            elif k == "if":
                bits, cmp_, val = op[1], op[2], op[3]
                v = sum(cbits[b] << j for j, b in enumerate(bits))
                hit = {"==": v == val, "!=": v != val, "<": v < val,
                       ">": v > val, "<=": v <= val, ">=": v >= val}[cmp_]
                self._exec(sim, op[4] if hit else op[5], cbits, rng)

    def _new_sim(self, n):
        sim = self._QS(n, **self.sim_kwargs)
        if self.device is not None:
            sim.set_device(self.device)
        return sim

    def _key(self, prog, cbits):
        regs = []
        for nm in prog.creg_order:
            idx = prog.cregs[nm]
            regs.append("".join(str(cbits[i]) for i in reversed(idx)))
        return " ".join(reversed(regs))

    def _labels(self, prog):
        lab = {}
        for nm in prog.creg_order:
            idx = prog.cregs[nm]
            for j, b in enumerate(idx):
                lab[b] = nm if len(idx) == 1 else "%s[%d]" % (nm, j)
        return lab

    # ---- public ------------------------------------------------------
    def run(self, program, shots=1024, seed=None, params=None,
            statevector=False, path=None):
        """program: Program, QASM text, or a path to a .qasm file."""
        res = Result()
        t0 = time.perf_counter()
        if not isinstance(program, Program):
            if "\n" not in program and os.path.exists(program):
                path = program
                program = open(program).read()
            program = compile_qasm(program, params, path)
        prog = program
        res.num_qubits = prog.num_qubits
        res.gate_counts = prog.gate_counts()
        t1 = time.perf_counter()

        ops = prog.ops
        split = 0
        while split < len(ops) and ops[split][0] in _UNITARY:
            split += 1
        prefix, tail = ops[:split], ops[split:]

        rng = random.Random(seed) if seed is not None else None
        sim = self._new_sim(prog.num_qubits)
        self._exec(sim, prefix, None)
        t2 = time.perf_counter()
        res.simulator = sim

        terminal = all(op[0] == "m" for op in tail)
        res.statevector_is_final = not tail
        if statevector:
            res.statevector = list(sim.out_ket())

        lab = self._labels(prog)
        res.shots = shots
        if terminal:
            meas = [(q, c) for _, q, c in tail if c is not None]
            last = {}
            for q, c in meas:
                last[c] = q                    # later measure wins
            for c in sorted(last):
                res.probabilities[lab[c]] = float(sim.prob(last[c]))
            qubits = sorted(set(last.values()))
            res.mode = "sampled"
            if shots and qubits and len(qubits) <= 64 and rng is None:
                pos = {q: j for j, q in enumerate(qubits)}
                cnt = Counter(sim.measure_shots(qubits, shots))
                counts = Counter()
                for perm, n in cnt.items():
                    cb = [0] * prog.num_clbits
                    for c, q in last.items():
                        cb[c] = (perm >> pos[q]) & 1
                    counts[self._key(prog, cb)] += n
                res.counts = dict(counts)
            elif shots and qubits:
                terminal = False       # >64 bits or seeded: per-shot path
            elif shots:
                res.counts = {self._key(prog, [0] * prog.num_clbits): shots}
        if not terminal:
            res.mode = "per-shot"
            counts = Counter()
            for s in range(max(shots, 1)):
                w = sim.clone() if shots > 1 else sim
                cb = [0] * prog.num_clbits
                self._exec(w, tail, cb, rng)
                counts[self._key(prog, cb)] += 1
            res.counts = dict(counts) if shots else {}
        t3 = time.perf_counter()
        res.timings = {"compile_s": t1 - t0, "unitary_s": t2 - t1,
                       "measure_s": t3 - t2}
        return res


def run_qasm(source, shots=1024, seed=None, params=None, **sim_kwargs):
    """One-shot convenience: text or path in, Result out."""
    return QrackBackend(**sim_kwargs).run(source, shots=shots, seed=seed,
                                          params=params)


# =====================================================================
# CLI
# =====================================================================

def _kv(s):
    k, v = s.split("=", 1)
    lv = v.lower()
    if lv in ("true", "false"):
        return k, lv == "true"
    try:
        return k, float(v) if any(ch in v for ch in ".eE") else int(v)
    except ValueError:
        return k, v


def main_run(argv=None):
    ap = argparse.ArgumentParser(prog="qasm_qrack.py run",
                                 description="Run OpenQASM on PyQrack")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--shots", type=int, default=1024,
                    help="0 = exact probabilities only, no sampling")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--param", action="append", default=[], metavar="K=V",
                    help="value for an 'input' scalar")
    ap.add_argument("--cpu", action="store_true", help="is_gpu=False")
    ap.add_argument("--device", type=int, help="OpenCL device index")
    ap.add_argument("--sim", action="append", default=[], metavar="K=V",
                    help="extra QrackSimulator kwarg, e.g. "
                         "is_stabilizer_hybrid=true")
    ap.add_argument("--top", type=int, default=16,
                    help="show this many most frequent outcomes")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    kw = dict(_kv(s) for s in a.sim)
    if a.cpu:
        kw["is_gpu"] = False
    params = {k: float(v) for k, v in (_kv(s) for s in a.param)}
    be = QrackBackend(device=a.device, **kw)
    out, rc = {}, 0
    for f in a.files:
        try:
            r = be.run(f, shots=a.shots, seed=a.seed, params=params)
        except QasmError as e:
            print("%s: %s" % (f, e), file=sys.stderr)
            rc = 1
            continue
        if a.json:
            out[f] = r.as_dict()
            continue
        gates = sum(v for k, v in r.gate_counts.items()
                    if k not in ("measure", "reset"))
        print("%s: %d qubits, %d gates, %s, %.3f s"
              % (f, r.num_qubits, gates, r.mode, sum(r.timings.values())))
        for k, p in r.probabilities.items():
            print("  P(%s=1) = %.9f   <Z> = %+.9f" % (k, p, 1 - 2 * p))
        for k, n in sorted(r.counts.items(), key=lambda x: -x[1])[:a.top]:
            print("  %s  %d" % (k, n))
        if len(r.counts) > a.top:
            print("  ... %d more outcomes" % (len(r.counts) - a.top))
    if a.json:
        json.dump(out, sys.stdout, indent=1)
        print()
    return rc

# =====================================================================
# =====================================================================
# PART 2 -- SEAM RUNNER
# =====================================================================
# =====================================================================
#
# its own PyQrack instance, run the same circuit through QrackAceBackend's
# seam gadget on its own instances, and synthesise both against an exact
# single-instance reference.
#
# THREE ROUTES, ONE CIRCUIT
# =====================================================================
#   knit   Qubits are partitioned into patches. Every gate inside a
#          patch stays in that patch's chunk. Every gate across a seam
#          is CUT: replaced by a weighted sum of purely local operations
#          on each side. Each (patch, local choice) is rendered as a
#          standalone OPENQASM 3 chunk and run on its own QrackSimulator
#          -- optionally in its own process, on its own OpenCL device.
#          The chunk results are recombined classically. Exact, or
#          truncated with a rigorous bound.
#   ace    The whole circuit on QrackAceBackend, whose patches are
#          separate QrackSimulator instances and whose seam qubits are
#          replicated across neighbouring patches plus a shared crossbar
#          instance, kept consistent by ancilla-based detection and
#          replica reconciliation. Approximate, by design.
#   ref    One ordinary QrackSimulator, exact. Only feasible when the
#          whole register fits, which is exactly what makes it the
#          yardstick at the sizes where it does.
#
# With --partition ace (the default) the knit route cuts at ACE's own
# seams: each logical qubit goes to its ACE home patch. Both routes then
# face identical seams, so ref-vs-knit-vs-ace isolates what the seam
# gadget costs in accuracy, gate for gate.
#
# WHY THE CUT IS A QUASI-PROBABILITY SUM, NOT A SUM OF STATES
# =====================================================================
# Qrack gives every instance a random global phase and exposes no way
# to pin it. Splitting e^{-i t/2 A(x)B} = cI - is A(x)B into two branch
# STATES and adding them back needs those phases to agree across
# instances, so it cannot work here. Cutting the CHANNEL instead is
# phase-free: every term is a local channel whose outputs are
# probabilities. With c = cos(t/2), s = sin(t/2), A and B any Hermitian
# involutions (A^2 = B^2 = I), and verified to 1e-16 numerically:
#
#   U rho U+ =   c^2        rho
#              + s^2        (A (x) B) rho (A (x) B)
#              - cs [ M_A (x) R+_B  -  M_A (x) R-_B
#                   + R+_A (x) M_B  -  R-_A (x) M_B ] (rho)
#
#   R+-_X(rho) = e^{+-i pi/4 X} rho e^{-+i pi/4 X}
#   M_X(rho)   = P+ rho P+  -  P- rho P-      P+- = (I +- X)/2
#
# M_X is "measure X, weight by its eigenvalue". Chunks enumerate both
# outcomes exactly on cloned instances rather than sampling, so there
# is no shot noise anywhere in the knit route. The price is the
# one-norm gamma = 1 + 2|sin t| per cut: 1.4 for a Kitaev bond at
# theta = 0.1, 3 for a CNOT. Terms are enumerated best-first by
# |coefficient|. The excluded one-norm bounds the error of every
# reported probability and <P>, since each local chunk value is at most
# 1 in magnitude.
#
# WHAT GETS CUT, AND HOW
# =====================================================================
#   cx a,b ; diag(b)... ; cx a,b   across a seam  -> one ZZ cut
#       (this is the K4 bond: basis changes stay local, so XX/YY/ZZ
#        bonds all cost one cut with t = 2*theta)
#   any controlled single-qubit gate, controls in one patch, target in
#       another -> local phases + one cut: A = I - 2*Pi(controls),
#       B = the SU(2) axis of the target gate
#       (several controls: the M channel measures Pi through a single
#        per-patch ancilla, computed in, measured, computed out)
#   swap across a seam -> three cuts (XX, YY, ZZ at t = pi/2)
# Controls split over several patches, or shared between the target's
# patch and another, are refused with the op index: move the seam.
#
# INPUT
# =====================================================================
# A unitary circuit, optionally followed by terminal measurements.
# Observables: the measured bits (exact joint distribution up to 16
# bits, per-bit marginals always), and any Pauli strings via --pauli.
#
#   python3 qasm_qrack.py seams trotter.qasm --pauli "ZYYYZYYYZZ@0,5,3,9,7,4,1,11,10,6"
#   python3 qasm_qrack.py seams c.qasm --partition auto:3 --workers 6 --devices 0,1,2,3,4,5
#   python3 qasm_qrack.py seams c.qasm --lrc 2 --lrr 2 --emit chunks/ --json


# ---------------------------------------------------------------------
# PROGRESS  (selftest, and chunk dispatch in 'seams')
# ---------------------------------------------------------------------

def fmt_s(t):
    return "%.1fs" % t if t < 60 else "%dm%02ds" % (t // 60, t % 60)


class Progress:
    """Progress on stderr, so stdout stays clean for results.

    Terminal: one live line per group, rewritten in place (at most 10
    updates/s), closed by a summary line. Pipe (logs, CI): groups that
    finish within 2 s print only their summary; longer ones add a line
    at every 10% and at least every 5 s, so a long case never leaves the
    log silent. verbose: every step on its own line with its time.
    quiet: group summaries only. Failures are counted live from the
    suite's own 'fails' list.
    """

    def __init__(self, suite, n_groups=1, fails=None, verbose=False,
                 quiet=False, stream=None):
        self.out = stream or sys.stderr
        self.tty = getattr(self.out, "isatty", lambda: False)()
        self.suite, self.n_groups, self.gi = suite, n_groups, 0
        self.fails = fails if fails is not None else []
        self.verbose, self.quiet = verbose, quiet
        self.t_suite = time.perf_counter()
        self.name = None

    def start(self, name, total):
        if self.name:
            self.end()
        self.gi += 1
        self.name, self.total, self.done = name, max(int(total), 1), 0
        self.detail = self.sub = ""
        self.t0 = self._t_step = self._last = time.perf_counter()
        self._tick = -1
        self.f0 = len(self.fails)
        self._emit(force=True)

    def step(self, detail="", n=1):
        now = time.perf_counter()
        self.done += n
        self.detail, self.sub = detail, ""
        if self.verbose and not self.quiet:
            self._clear()
            self._write("  %s %3d/%d  %-60s %7s%s\n" % (
                self._tag(), self.done, self.total, detail[:60],
                fmt_s(now - self._t_step),
                "  FAILS %d" % self._nf() if self._nf() else ""))
        self._t_step = now
        self._emit()

    def running(self, text):
        """Label the step now in progress (replaces the last finished
        step's label, so heartbeats never pair one case's label with
        another case's counters)."""
        self.detail, self.sub = "now: " + text, ""
        self._emit()

    def substep(self, text):
        """Movement inside a slow step, e.g. chunk counts."""
        self.sub = text
        self._emit()

    def note(self, text):
        """A result line for stdout, without tearing the live line."""
        self._clear()
        print(text, flush=True)
        self._emit(force=True)

    def end(self):
        if not self.name:
            return
        self._clear()
        nf = self._nf()
        self._write("%s %-22s %4d/%-4d %-10s %8s\n" % (
            self._tag(), self.name, self.done, self.total,
            "ok" if not nf else "%d FAILED" % nf,
            fmt_s(time.perf_counter() - self.t0)))
        self.name = None

    def finish(self):
        self.end()
        return time.perf_counter() - self.t_suite

    def _nf(self):
        return len(self.fails) - self.f0

    def _tag(self):
        return "[%s %d/%d]" % (self.suite, self.gi, self.n_groups)

    def _line(self):
        el = time.perf_counter() - self.t0
        eta = ("  eta %s" % fmt_s(el / self.done * (self.total - self.done))
               if self.done and self.done < self.total else "")
        d = " | ".join(x for x in (self.detail, self.sub) if x)
        return "%s %s  %d/%d %3d%%  %s%s%s%s" % (
            self._tag(), self.name, self.done, self.total,
            100 * self.done // self.total, fmt_s(el), eta,
            "  FAILS %d" % self._nf() if self._nf() else "",
            "  | " + d if d else "")

    def _emit(self, force=False):
        if self.quiet or not self.name or (self.verbose and not force):
            return
        now = time.perf_counter()
        if self.tty:
            if not force and now - self._last < 0.1:
                return
            self._last = now
            w = shutil.get_terminal_size((100, 20)).columns - 1
            self._write("\r" + self._line()[:w] + "\x1b[K")
        else:
            # logs: fast groups get only their summary line; slow ones a
            # line per 10% and a heartbeat at least every 5 s
            if now - self.t0 < 2.0:
                return
            tick = 10 * self.done // self.total
            if tick != self._tick or now - self._last >= 5:
                self._tick, self._last = tick, now
                self._write(self._line() + "\n")

    def _clear(self):
        if self.tty and not self.quiet:
            self._write("\r\x1b[K")

    def _write(self, text):
        self.out.write(text)
        self.out.flush()


EPS = 1e-12
# Outcomes below this are treated as impossible. In single precision a
# repeated measurement's impossible outcome comes back as ~1e-6, and
# force_m() on it either raises or silently normalises a zero vector
# into NaNs. So the floor tracks the library's float width (the same
# fppow ladder QrackAceBackend uses), and non-finite branches are
# discarded as a second line of defence.
def _pfloor():
    try:
        from pyqrack.qrack_system import Qrack
        fp = Qrack.fppow
    except Exception:
        fp = 5
    return 2.0 ** -8 if fp < 5 else (2.0 ** -14 if fp == 5 else 1e-10)


PFLOOR = _pfloor()
if np is not None:
    PX = np.array([[0, 1], [1, 0]], complex)
    PY = np.array([[0, -1j], [1j, 0]], complex)
    PZ = np.array([[1, 0], [0, -1]], complex)
    I2 = np.eye(2, dtype=complex)
PAULI_INT = {"X": 1, "Y": 3, "Z": 2}         # pyqrack Pauli enum values


def tup(M):
    return tuple(complex(x) for x in np.asarray(M).reshape(-1))


def Gm(M, q):
    t = tup(M)
    return ("g", recognise(t), t, q, (), ())


def Gs(M, q):
    """Uncontrolled local op, dropped when it is a pure phase: every use
    is inside a channel, where global phase cancels."""
    M = np.asarray(M).reshape(2, 2)
    if abs(M[0, 1]) < EPS and abs(M[1, 0]) < EPS and \
            abs(M[0, 0] - M[1, 1]) < EPS:
        return []
    return [Gm(M, q)]


def op_qubits(op):
    k = op[0]
    if k == "g":
        return (op[3],) + tuple(op[4])
    if k == "gp":
        return tuple(op[2])
    if k == "sw":
        return (op[1], op[2]) + tuple(op[3])
    if k in ("m", "r"):
        return (op[1],)
    raise QasmError("classical control flow cannot be cut at seams")


def su2_split(M):
    """M = e^{i phi} (cos g I + i sin g N), N a Hermitian involution.
    Returns (phi, g, N); N is None when M is a pure phase."""
    M = np.asarray(M, complex).reshape(2, 2)
    phi = cmath.phase(np.linalg.det(M)) / 2
    V = M * cmath.exp(-1j * phi)
    cg = max(-1.0, min(1.0, ((V[0, 0] + V[1, 1]) / 2).real))
    g = math.acos(cg)
    if math.sin(g) < 1e-9:
        return phi + (math.pi if cg < 0 else 0.0), 0.0, None
    return phi, g, (V - cg * I2) / (1j * math.sin(g))


def u_params(M):
    """M = e^{i alpha} U(theta, phi, lambda), OpenQASM U convention."""
    a, b, c, d = (complex(x) for x in np.asarray(M).reshape(-1))
    th = 2 * math.atan2(abs(c), abs(a))
    if abs(a) > 1e-12:
        al = cmath.phase(a)
        if abs(c) > 1e-12:
            ph, lm = cmath.phase(c) - al, cmath.phase(-b) - al
        else:
            ph, lm = 0.0, cmath.phase(d) - al
    else:
        lm, al = 0.0, cmath.phase(-b)
        ph = cmath.phase(c) - al
    return th, ph, lm, al


def zyz(M):
    """M = e^{i a} Rz(b) Ry(g) Rz(d)."""
    M = np.asarray(M, complex).reshape(2, 2)
    a = cmath.phase(np.linalg.det(M)) / 2
    V = M * cmath.exp(-1j * a)
    g = 2 * math.atan2(abs(V[1, 0]), abs(V[0, 0]))
    s1 = 2 * cmath.phase(V[1, 1]) if abs(V[0, 0]) > 1e-12 else 0.0
    d1 = 2 * cmath.phase(V[1, 0]) if abs(V[1, 0]) > 1e-12 else 0.0
    return a, (s1 + d1) / 2, g, (s1 - d1) / 2


# =====================================================================
# PARTITIONS
# =====================================================================

def partition_ace(n, lrc, lrr, transpose):
    from pyqrack import QrackAceBackend
    a = QrackAceBackend(n, long_range_columns=lrc, long_range_rows=lrr,
                        is_transpose=transpose, is_gpu=False)
    home = [a._qubits[q][0][0] for q in range(n)]
    reps = [len(a._qubits[q]) for q in range(n)]
    return home, reps


def interaction(prog):
    w = defaultdict(int)
    for op in prog.ops:
        if op[0] in ("g", "sw", "gp"):
            qs = op_qubits(op)
            for x, y in itertools.combinations(sorted(set(qs)), 2):
                w[(x, y)] += 1
    return w


def partition_auto(prog, k):
    n = prog.num_qubits
    w = interaction(prog)
    adj = defaultdict(dict)
    for (x, y), c in w.items():
        adj[x][y] = adj[y][x] = c
    order, seen = [], set()
    for s in sorted(range(n), key=lambda q: len(adj[q])):
        if s in seen:
            continue
        dq = deque([s])
        seen.add(s)
        while dq:
            v = dq.popleft()
            order.append(v)
            for u in sorted(adj[v], key=lambda u: -adj[v][u]):
                if u not in seen:
                    seen.add(u)
                    dq.append(u)
    part = [0] * n
    for i, q in enumerate(order):
        part[q] = min(k - 1, i * k // n)
    lo, hi = n // k - 1, -(-n // k) + 1
    for _ in range(8):                          # greedy cut refinement
        moved = False
        for q in range(n):
            size = [part.count(p) for p in range(k)]
            here = part[q]
            if size[here] - 1 < lo:
                continue
            gain = defaultdict(int)
            for u, c in adj[q].items():
                gain[part[u]] += c
            best = max(range(k), key=lambda p: (gain[p] - gain[here]
                                                 if size[p] + 1 <= hi or
                                                 p == here else -1e9))
            if best != here and gain[best] > gain[here]:
                part[q] = best
                moved = True
        if not moved:
            break
    return part


def partition_spec(spec, prog, lrc, lrr, transpose, ace_n):
    n = prog.num_qubits
    if spec == "ace":
        home, reps = partition_ace(ace_n, lrc, lrr, transpose)
        return home[:n], reps[:n]
    if spec.startswith("auto:"):
        return partition_auto(prog, int(spec[5:])), None
    if spec == "regs":
        part = [0] * n
        for i, idx in enumerate(prog.qregs.values()):
            for q in idx:
                part[q] = i
        return part, None
    part = [None] * n
    for i, grp in enumerate(spec.split("/")):
        for item in grp.split(","):
            a, _, b = item.partition("-")
            for q in range(int(a), int(b or a) + 1):
                part[q] = i
    if None in part:
        raise QasmError("partition does not cover qubit(s) %s"
                        % [q for q, p in enumerate(part) if p is None])
    return part, None


def compact(part):
    ids = {p: i for i, p in enumerate(sorted(set(part)))}
    return [ids[p] for p in part]


# =====================================================================
# SEAM SPLIT
# =====================================================================

TERMS = (("I", "I"), ("U", "U"), ("M", "R+"), ("M", "R-"), ("R+", "M"),
         ("R-", "M"))


class Cut:
    def __init__(self, j, theta, pa, sa, pb, sb, origin):
        self.j, self.theta, self.origin = j, theta, origin
        self.patch = (pa, pb)
        self.side = (sa, sb)

    def terms(self):
        c, s = math.cos(self.theta / 2), math.sin(self.theta / 2)
        w = (c * c, s * s, -c * s, c * s, -c * s, c * s)
        return [(wi, ch) for wi, ch in zip(w, TERMS) if abs(wi) > 1e-15]

    def gamma(self):
        return sum(abs(w) for w, _ in self.terms())

    def describe(self):
        def sd(s):
            if s[0] == "mat":
                return "%s on q%d" % (pauli_name(s[1]), s[2])
            return "I-2Pi(ctrl %s=%s)" % (list(s[1]), list(s[2]))
        return {"cut": self.j, "origin": self.origin,
                "theta": self.theta, "gamma": self.gamma(),
                "patch_a": self.patch[0], "side_a": sd(self.side[0]),
                "patch_b": self.patch[1], "side_b": sd(self.side[1])}


def pauli_name(N):
    for nm, P in (("X", PX), ("Y", PY), ("Z", PZ), ("-Z", -PZ)):
        if np.allclose(N, P, atol=1e-9):
            return nm
    n = [np.trace(N @ P).real / 2 for P in (PX, PY, PZ)]
    return "n.sigma(%.3f,%.3f,%.3f)" % tuple(n)


class SeamSplit:
    def __init__(self, prog, part):
        self.prog = prog
        self.part = part
        self.npatch = max(part) + 1
        ops = prog.ops
        k = 0
        while k < len(ops) and ops[k][0] in ("g", "gp", "sw"):
            k += 1
        prefix, tail = ops[:k], ops[k:]
        if any(op[0] != "m" for op in tail):
            raise QasmError("seam cutting needs a unitary circuit followed "
                            "only by terminal measurements")
        last = {}
        for _, q, c in tail:
            if c is not None:
                last[c] = q
        self.meas = sorted(last)                # global clbits
        self.meas_q = [last[c] for c in self.meas]
        self.chunks = [[] for _ in range(self.npatch)]
        self.cuts = []
        self.needs_anc = [False] * self.npatch
        i = 0
        while i < len(prefix):
            nxt = self._zz(prefix, i)
            if nxt is not None:
                i = nxt
                continue
            self._handle(prefix[i], i)
            i += 1
        self.cuts_of = [[c.j for c in self.cuts if p in c.patch]
                        for p in range(self.npatch)]
        self.qubits_of = [[q for q in range(prog.num_qubits) if part[q] == p]
                          for p in range(self.npatch)]

    def _local(self, p, op):
        self.chunks[p].append(("op", op))

    def _cut(self, theta, pa, sa, pb, sb, origin):
        j = len(self.cuts)
        self.cuts.append(Cut(j, theta, pa, sa, pb, sb, origin))
        self.chunks[pa].append(("cut", j, 0))
        self.chunks[pb].append(("cut", j, 1))
        if sa[0] == "proj":
            self.needs_anc[pa] = True

    def _zz(self, ops, i):
        op = ops[i]
        if op[0] != "g" or op[1] != "x" or len(op[4]) != 1 or not op[5][0]:
            return None
        a, b = op[4][0], op[3]
        if self.part[a] == self.part[b]:
            return None
        d, j = [1 + 0j, 1 + 0j], i + 1
        while j < len(ops):
            o = ops[j]
            if o[0] == "g" and o[3] == b and not o[4] and \
                    abs(o[2][1]) < EPS and abs(o[2][2]) < EPS:
                d = [d[0] * o[2][0], d[1] * o[2][3]]
                j += 1
                continue
            break
        if j >= len(ops) or ops[j] != op:
            return None
        nu = (cmath.phase(d[1]) - cmath.phase(d[0])) / 2
        if abs(math.sin(nu)) > 1e-12:
            self._cut(2 * nu, self.part[a], ("mat", PZ, a), self.part[b],
                      ("mat", PZ, b), "zz@op%d" % i)
        return j + 1

    def _handle(self, op, idx):
        qs = op_qubits(op)
        ps = {self.part[q] for q in qs}
        if len(ps) <= 1:
            if ps:
                self._local(ps.pop(), op)
            return
        if op[0] == "gp":
            cs, vs = op[2], op[3]
            e = cmath.exp(1j * op[1])
            m = (1, 0, 0, e) if vs[-1] else (e, 0, 0, 1)
            return self._handle(("g", None, m, cs[-1], cs[:-1], vs[:-1]), idx)
        if op[0] == "sw":
            a, b, cs, vs = op[1:]
            if cs:
                for x, y in ((a, b), (b, a), (a, b)):
                    self._handle(("g", "x", MX, y, cs + (x,), vs + (1,)),
                                 idx)
                return
            for P in (PX, PY, PZ):
                self._cut(math.pi / 2, self.part[a], ("mat", P, a),
                          self.part[b], ("mat", P, b), "swap@op%d" % idx)
            return
        _, tag, m, t, cs, vs = op
        tp = self.part[t]
        cps = {self.part[c] for c in cs}
        if tp in cps or len(cps) != 1:
            raise QasmError(
                "op %d couples patches %s with controls spread over %s; this "
                "cutter needs all controls in one patch other than the "
                "target's -- adjust the partition" % (idx, sorted(ps),
                                                      sorted(cps)))
        pa = cps.pop()
        phi, g, N = su2_split(m)
        if abs(cmath.exp(1j * phi) - 1) > EPS:
            self._local(pa, ("gp", phi, cs, vs))
        if N is None:
            return
        for o in Gs(math.cos(g / 2) * I2 + 1j * math.sin(g / 2) * N, t):
            self._local(tp, o)
        if len(cs) == 1:
            sa = ("mat", PZ if vs[0] else -PZ, cs[0])
        else:
            sa = ("proj", tuple(cs), tuple(vs))
        self._cut(g, pa, sa, tp, ("mat", N, t), "op%d" % idx)

    # ---- chunk rendering --------------------------------------------
    def _chan(self, side, chan, lmap, anc):
        if chan == "I":
            return []
        if side[0] == "mat":
            N, q = side[1], lmap[side[2]]
            if chan == "U":
                return Gs(N, q)
            if chan in ("R+", "R-"):
                sg = 1 if chan == "R+" else -1
                r = 1 / math.sqrt(2)
                return Gs(r * I2 + 1j * sg * r * N, q)
            w, v = np.linalg.eigh(N)            # ascending: -1, +1
            W = np.column_stack([v[:, 1], v[:, 0]])
            return Gs(W.conj().T, q) + [("m", q, "sgn")] + Gs(W, q)
        cs = tuple(lmap[c] for c in side[1])
        vs = side[2]
        if chan == "U":
            return [("gp", math.pi, cs, vs)]
        if chan in ("R+", "R-"):
            return [("gp", -math.pi / 2 if chan == "R+" else math.pi / 2,
                     cs, vs)]
        mcx = ("g", "x", MX, anc, cs, vs)
        return [mcx, ("m", anc, "sgn"), mcx]

    def render(self, p, choice):
        """(p, local channel per cut touching p) -> OPENQASM 3 text."""
        qs = self.qubits_of[p]
        lmap = {q: i for i, q in enumerate(qs)}
        anc = len(qs)
        nq = len(qs) + (1 if self.needs_anc[p] else 0)
        pos = {j: i for i, j in enumerate(self.cuts_of[p])}
        ops = []
        for item in self.chunks[p]:
            if item[0] == "op":
                ops.append(remap(item[1], lmap))
            else:
                _, j, s = item
                ops += self._chan(self.cuts[j].side[s], choice[pos[j]],
                                  lmap, anc)
        obs = [lmap[q] for q in self.meas_q if self.part[q] == p]
        hdr = ["patch %d: global qubits %s" % (p, qs),
               "seam channels: %s" % ", ".join(
                   "cut%d=%s" % (j, ch) for j, ch in zip(self.cuts_of[p],
                                                         choice)) or "none"]
        if self.needs_anc[p]:
            hdr.append("q[%d] is the projector-measurement ancilla" % anc)
        return write_qasm(ops, nq, obs, hdr)


def remap(op, lm):
    k = op[0]
    if k == "g":
        return ("g", op[1], op[2], lm[op[3]], tuple(lm[c] for c in op[4]),
                op[5])
    if k == "gp":
        return ("gp", op[1], tuple(lm[c] for c in op[2]), op[3])
    if k == "sw":
        return ("sw", lm[op[1]], lm[op[2]], tuple(lm[c] for c in op[3]),
                op[4])
    raise QasmError("unexpected op in chunk")


def write_qasm(ops, nq, obs, header=()):
    f = lambda x: repr(float(x))
    n_sgn = sum(1 for o in ops if o[0] == "m")
    out = ["OPENQASM 3.0;"] + ["// " + h for h in header] + [
        'include "stdgates.inc";', "qubit[%d] q;" % nq]
    if n_sgn:
        out.append("bit[%d] sgn;" % n_sgn)
    if obs:
        out.append("bit[%d] obs;" % len(obs))
    k = 0
    for op in ops:
        if op[0] == "m":
            out.append("measure q[%d] -> sgn[%d];" % (op[1], k))
            k += 1
            continue
        if op[0] == "g":
            tag, M, t, cs, vs = op[1:]
        elif op[0] == "gp":
            tag, M, t, cs, vs = "gphase", None, None, op[2], op[3]
        else:
            tag, M, t, cs, vs = "swap", None, None, op[3], op[4]
        mods = "".join("ctrl @ " if v else "negctrl @ " for v in vs)
        ctl = ["q[%d]" % c for c in cs]
        if op[0] == "gp":
            out.append("%sgphase(%s)%s;" % (mods, f(op[1]),
                                            " " + ", ".join(ctl) if ctl else ""))
            continue
        if op[0] == "sw":
            out.append("%sswap %s;" % (mods, ", ".join(
                ctl + ["q[%d]" % op[1], "q[%d]" % op[2]])))
            continue
        args = ", ".join(ctl + ["q[%d]" % t])
        if isinstance(tag, tuple):
            out.append("%s%s(%s) %s;" % (mods, tag[0], f(tag[1]), args))
        elif tag:
            out.append("%s%s %s;" % (mods, tag, args))
        else:
            th, ph, lm, al = u_params(M)
            if cs and abs(cmath.exp(1j * al) - 1) > 1e-15:
                out.append("%sgphase(%s) %s;" % (mods, f(al), ", ".join(ctl)))
            out.append("%sU(%s, %s, %s) %s;" % (mods, f(th), f(ph), f(lm),
                                                args))
    for i, q in enumerate(obs):
        out.append("obs[%d] = measure q[%d];" % (i, q))
    return "\n".join(out) + "\n"


# =====================================================================
# CHUNK WORKER  (one QrackSimulator per chunk; one process per device)
# =====================================================================

_W = {}


def _winit(devq, sim_kwargs):
    dev = devq.get() if devq is not None else None
    _W["be"] = QrackBackend(device=dev, **sim_kwargs)
    _W["dev"] = dev


def run_chunk(job):
    """job = (key, qasm, paulis, joint).  Exact enumeration of every signed
    measurement branch; returns weighted local observables."""
    key, text, paulis, joint = job
    if "be" not in _W:
        _winit(None, {})
    from pyqrack import Pauli
    be = _W["be"]
    prog = compile_qasm(text)
    sgn = set(prog.cregs.get("sgn", []))
    obs_bits = prog.cregs.get("obs", [])
    obs_q = [None] * len(obs_bits)
    body = []
    for op in prog.ops:
        if op[0] == "m" and op[2] not in sgn:
            obs_q[obs_bits.index(op[2])] = op[1]
        else:
            body.append(op)
    m = len(obs_q)
    acc_w, leaves = 0.0, 0
    acc_j = np.zeros(1 << m) if joint else None
    acc_m = np.zeros(m)
    acc_p = np.zeros(len(paulis))
    pz = [([q for q, _ in pl], [Pauli(b) for _, b in pl]) for pl in paulis]
    stack = [(be._new_sim(prog.num_qubits), 0, 1.0)]
    while stack:
        s, k, w = stack.pop()
        j = k
        while j < len(body) and body[j][0] != "m":
            j += 1
        be._exec(s, body[k:j], None)
        if j < len(body):
            q = body[j][1]
            p1 = float(s.prob(q))
            outs = [(r, p) for r, p in ((0, 1.0 - p1), (1, p1)) if p > PFLOOR]
            sims = [s] + [s.clone() for _ in outs[1:]]
            for ss, (r, p) in zip(sims, outs):
                try:
                    ss.force_m(q, bool(r))
                except RuntimeError:        # Qrack deems it impossible
                    continue
                if not math.isfinite(float(ss.prob(q))):
                    continue
                stack.append((ss, j + 1, w * p * (-1.0 if r else 1.0)))
            continue
        leaves += 1
        acc_w += w
        if joint and m:
            acc_j += w * np.array([s.prob_perm(obs_q, [bool(x >> i & 1)
                                                       for i in range(m)])
                                   for x in range(1 << m)])
        elif joint:
            acc_j += w
        for i, q in enumerate(obs_q):
            acc_m[i] += w * float(s.prob(q))
        for i, (qs, bs) in enumerate(pz):
            acc_p[i] += w * (float(s.pauli_expectation(qs, bs)) if qs else 1.0)
    return key, {"w": acc_w, "joint": acc_j, "marg": acc_m, "pauli": acc_p,
                 "leaves": leaves, "device": _W.get("dev")}


# =====================================================================
# KNIT: enumerate, dispatch, synthesise
# =====================================================================

def top_terms(cuts, budget):
    lists = [c.terms() for c in cuts]
    total = 1
    for l in lists:
        total *= len(l)
        if total > budget:
            break
    if total <= budget:
        for combo in itertools.product(*[range(len(l)) for l in lists]):
            yield combo, math.prod(lists[j][i][0] for j, i in enumerate(combo))
        return
    order = [sorted(range(len(l)), key=lambda i: -abs(l[i][0])) for l in lists]
    mag = lambda idx: math.prod(abs(lists[j][order[j][i]][0])
                                for j, i in enumerate(idx))
    start = (0,) * len(lists)
    heap, seen, n = [(-mag(start), start)], {start}, 0
    while heap and n < budget:
        _, idx = heapq.heappop(heap)
        combo = tuple(order[j][i] for j, i in enumerate(idx))
        yield combo, math.prod(lists[j][i][0] for j, i in enumerate(combo))
        n += 1
        for j in range(len(idx)):
            if idx[j] + 1 < len(lists[j]):
                nx = idx[:j] + (idx[j] + 1,) + idx[j + 1:]
                if nx not in seen:
                    seen.add(nx)
                    heapq.heappush(heap, (-mag(nx), nx))


def knit(split, paulis, budget, workers, devices, sim_kwargs, emit=None,
         progress=None):
    """progress(done, total) is called after every chunk instance."""
    t0 = time.perf_counter()
    cuts = split.cuts
    gamma = math.prod(c.gamma() for c in cuts) if cuts else 1.0
    lists = [c.terms() for c in cuts]
    joint = len(split.meas) <= 16
    combos, incl = defaultdict(float), 0.0
    nterms = 0
    for combo, coef in top_terms(cuts, budget):
        keys = []
        for p in range(split.npatch):
            keys.append((p, tuple(lists[j][combo[j]][1][cuts[j].patch.index(p)]
                                  for j in split.cuts_of[p])))
        combos[tuple(keys)] += coef
        incl += abs(coef)
        nterms += 1
    excluded = max(0.0, gamma - incl)

    # paulis restricted to each patch, in local indices
    loc_p = []
    for p in range(split.npatch):
        lmap = {q: i for i, q in enumerate(split.qubits_of[p])}
        loc_p.append([[(lmap[q], b) for q, b in pl if split.part[q] == p]
                      for pl in paulis])
    need = sorted({k for kt in combos for k in kt})
    jobs = [(k, split.render(k[0], k[1]), loc_p[k[0]], joint) for k in need]
    if emit:
        os.makedirs(emit, exist_ok=True)
        for k, text, _, _ in jobs[:500]:
            nm = "patch%d_%s.qasm" % (k[0], "-".join(
                c.replace("+", "p").replace("-", "m") for c in k[1]) or "bulk")
            open(os.path.join(emit, nm), "w").write(text)
        json.dump({"partition": split.part,
                   "cuts": [c.describe() for c in cuts]},
                  open(os.path.join(emit, "seams.json"), "w"), indent=1)
    t1 = time.perf_counter()

    res = {}
    if workers > 1:
        ctx = mp.get_context("spawn")           # OpenCL does not survive fork
        devq = ctx.Queue()
        for i in range(workers):
            devq.put(devices[i % len(devices)] if devices else None)
        with ctx.Pool(workers, initializer=_winit,
                      initargs=(devq, sim_kwargs)) as pool:
            for k, v in pool.imap_unordered(run_chunk, jobs, chunksize=4):
                res[k] = v
                if progress:
                    progress(len(res), len(jobs))
    else:
        _W["be"] = QrackBackend(device=devices[0] if devices else None,
                                   **sim_kwargs)
        _W["dev"] = devices[0] if devices else None
        for job in jobs:
            k, v = run_chunk(job)
            res[k] = v
            if progress:
                progress(len(res), len(jobs))
    t2 = time.perf_counter()

    # synthesis
    obs_pos = []                                   # per measured bit
    for i, q in enumerate(split.meas_q):
        p = split.part[q]
        local = [qq for qq in split.meas_q if split.part[qq] == p]
        obs_pos.append((p, local.index(q)))
    marg = np.zeros(len(split.meas))
    pauli = np.zeros(len(paulis))
    M = len(split.meas)
    dist = np.zeros(1 << M) if joint else None
    if joint:
        gidx = []
        for p in range(split.npatch):
            bits = [i for i, q in enumerate(split.meas_q)
                    if split.part[q] == p]
            x = np.arange(1 << M)
            li = np.zeros(1 << M, dtype=np.int64)
            for k, gi in enumerate(bits):
                li |= ((x >> gi) & 1) << k
            gidx.append(li)
    for kt, coef in combos.items():
        vals = [res[k] for k in kt]
        ws = [v["w"] for v in vals]
        for i, (p, li) in enumerate(obs_pos):
            others = math.prod(w for pp, w in enumerate(ws) if pp != p)
            marg[i] += coef * vals[p]["marg"][li] * others
        for o, pl in enumerate(paulis):
            sup = {split.part[q] for q, _ in pl}
            pauli[o] += coef * math.prod(
                vals[p]["pauli"][o] if p in sup else ws[p]
                for p in range(split.npatch))
        if joint:
            term = np.full(1 << M, coef)
            for p in range(split.npatch):
                term = term * vals[p]["joint"][gidx[p]]
            dist += term
    t3 = time.perf_counter()
    return {"marg": marg, "pauli": pauli, "dist": dist,
            "trace": sum(c * math.prod(res[k]["w"] for k in kt)
                         for kt, c in combos.items()),
            "cuts": len(cuts), "gamma": gamma, "terms": nterms,
            "unique_products": len(combos), "excluded": excluded,
            "chunks": len(jobs), "leaves": sum(v["leaves"] for v in
                                                 res.values()),
            "devices": sorted({str(v["device"]) for v in res.values()}),
            "timings": {"split_render_s": t1 - t0, "chunks_s": t2 - t1,
                        "synthesis_s": t3 - t2}}


# =====================================================================
# ACE ROUTE
# =====================================================================

_ACE_FIX = {"h": "h", "x": "x", "y": "y", "z": "z", "s": "s", "sdg": "adjs",
            "t": "t", "tdg": "adjt", "sx": "sx", "sxdg": "adjsx"}


class AceRunner:
    def __init__(self, n, lrc, lrr, transpose, **kw):
        from pyqrack import QrackAceBackend, Pauli
        self.args = dict(long_range_columns=lrc, long_range_rows=lrr,
                         is_transpose=transpose, **kw)
        self.n = n
        self.P = {"rx": Pauli.PauliX, "ry": Pauli.PauliY, "rz": Pauli.PauliZ}
        self.ace = QrackAceBackend(n, **self.args)

    def fresh(self):
        from pyqrack import QrackAceBackend
        self.ace = QrackAceBackend(self.n, **self.args)

    def _one(self, q, tag, M):
        a = self.ace
        if tag in _ACE_FIX:
            getattr(a, _ACE_FIX[tag])(q)
        elif isinstance(tag, tuple) and tag[0] in self.P:
            a.r(self.P[tag[0]], tag[1], q)
        else:
            th, ph, lm, _ = u_params(M)
            a.u(q, th, ph, lm)

    def _ctl(self, c, v, t, tag, M):
        a = self.ace
        if tag in ("x", "y", "z"):
            getattr(a, ("c" if v else "ac") + tag)(c, t)
            return
        if not v:
            a.x(c)
        al, be, ga, de = zyz(M)
        Z, Y = self.P["rz"], self.P["ry"]
        a.r(Z, (de - be) / 2, t)
        a.cx(c, t)
        a.r(Z, -(de + be) / 2, t)
        a.r(Y, -ga / 2, t)
        a.cx(c, t)
        a.r(Y, ga / 2, t)
        a.r(Z, be, t)
        a.u(c, 0.0, 0.0, al)
        if not v:
            a.x(c)

    def execute(self, ops):
        for i, op in enumerate(ops):
            k = op[0]
            if k == "g":
                tag, M, t, cs, vs = op[1:]
                if not cs:
                    self._one(t, tag, M)
                elif len(cs) == 1:
                    self._ctl(cs[0], vs[0], t, tag, M)
                else:
                    raise QasmError("ACE route: op %d has %d controls; "
                                    "QrackAceBackend takes one" % (i, len(cs)))
            elif k == "gp":
                cs, vs = op[2], op[3]
                if len(cs) > 1:
                    raise QasmError("ACE route: multi-controlled phase")
                if cs:
                    if not vs[0]:
                        self.ace.x(cs[0])
                    self.ace.u(cs[0], 0.0, 0.0, op[1])
                    if not vs[0]:
                        self.ace.x(cs[0])
            elif k == "sw":
                if op[3]:
                    raise QasmError("ACE route: controlled swap")
                self.ace.swap(op[1], op[2])
            elif k == "m":
                continue
            else:
                raise QasmError("ACE route: '%s' not supported" % k)

    def layout(self):
        a = self.ace
        return {"patch_sims": len(a.sim) - (1 if a._boundary_sim_id is not
                                            None else 0),
                "crossbar": a._boundary_sim_id is not None,
                "sim_widths": [s.num_qubits() for s in a.sim],
                "seam_qubits": [q for q in range(a.num_qubits())
                                if len(a._qubits[q]) > 1]}

    def coupling(self, ops, paulis):
        """Two-qubit interactions on / off ACE's native coupling map (pairs
        sharing a simulator through their replicas). ACE targets
        nearest-neighbour circuits; off-map gates are its worst case."""
        cm = {frozenset(e) for e in self.ace.get_logical_coupling_map()}
        on = off = 0
        for op in ops:
            if op[0] not in ("g", "gp", "sw"):
                continue
            qs = sorted(set(op_qubits(op)))
            if len(qs) < 2:
                continue
            if all(frozenset(e) in cm for e in itertools.combinations(qs, 2)):
                on += 1
            else:
                off += 1
        anc = self.n - 1
        h_on = sum(1 for pl in paulis for q, _ in pl
                   if frozenset((anc, q)) in cm)
        h_all = sum(len(pl) for pl in paulis)
        return {"gates_on_map": on, "gates_off_map": off,
                "hadamard_test_on_map": h_on,
                "hadamard_test_off_map": h_all - h_on}

    def run(self, split, paulis, shots):
        prefix = [op for op in split.prog.ops if op[0] != "m"]
        cpl = self.coupling(prefix, paulis)
        self.execute(prefix)
        out = {"marg": [float(self.ace.prob(q)) for q in split.meas_q],
               "layout": dict(self.layout(), **cpl)}
        if shots and split.meas_q:
            out["shots"] = self.ace.measure_shots(split.meas_q, shots)
        vals = []
        for pl in paulis:                       # Hadamard test, last qubit
            self.fresh()
            self.execute(prefix)
            anc = self.n - 1
            self.ace.h(anc)
            for q, b in pl:
                getattr(self.ace, "c" + "?xzy"[b])(anc, q)
            self.ace.h(anc)
            vals.append(1.0 - 2.0 * float(self.ace.prob(anc)))
        out["pauli"] = vals
        return out


# =====================================================================
# REFERENCE
# =====================================================================

def reference(split, paulis, sim_kwargs):
    from pyqrack import Pauli
    be = QrackBackend(**sim_kwargs)
    r = be.run(split.prog, shots=0)
    s = r.simulator
    mq = split.meas_q
    out = {"marg": [float(s.prob(q)) for q in mq],
           "pauli": [float(s.pauli_expectation([q for q, _ in pl],
                                               [Pauli(b) for _, b in pl]))
                     for pl in paulis]}
    if len(mq) <= 16:
        out["dist"] = np.array([s.prob_perm(mq, [bool(x >> i & 1)
                                                 for i in range(len(mq))])
                                for x in range(1 << len(mq))])
    return out


# =====================================================================
# CLI
# =====================================================================

def parse_pauli(spec, n):
    spec = spec.strip()
    if "@" in spec:
        word, qs = spec.split("@")
        qs = [int(x) for x in qs.split(",")]
        if len(word) != len(qs):
            raise QasmError("pauli '%s': word and qubit list differ in length"
                            % spec)
        pl = list(zip(qs, word))
    else:
        pl = [(int(t[1:]), t[0]) for t in spec.split()]
    out = []
    for q, b in pl:
        b = b.upper()
        if b == "I":
            continue
        if b not in PAULI_INT or not 0 <= q < n:
            raise QasmError("bad Pauli factor %s%d" % (b, q))
        out.append((q, PAULI_INT[b]))
    return out


def main_seams(argv=None):
    if np is None:
        raise SystemExit("seams needs numpy")
    ap = argparse.ArgumentParser(prog="qasm_qrack.py seams",
                                 description="Cut QASM at seams; knit across "
                                 "PyQrack instances; compare with ACE")
    ap.add_argument("file")
    ap.add_argument("--partition", default="ace",
                    help="ace | auto:K | regs | explicit '0-5/6-12'")
    ap.add_argument("--lrc", type=int, default=4, help="ACE long_range_columns")
    ap.add_argument("--lrr", type=int, default=4, help="ACE long_range_rows")
    ap.add_argument("--transpose", action="store_true")
    ap.add_argument("--pauli", action="append", default=[],
                    help="WORD@q,q,.. or 'Z0 Y5 ..'; repeatable")
    ap.add_argument("--max-terms", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--devices", help="OpenCL device ids, e.g. 0,1,2,3,4,5")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--shots", type=int, default=0, help="ACE shots (joint)")
    ap.add_argument("--ace-repeats", type=int, default=5,
                    help="independent ACE runs; its seam gadget is "
                         "stochastic, so mean and spread are reported")
    ap.add_argument("--no-ace", action="store_true")
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--emit", help="write chunk QASM + seams.json here")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true",
                    help="no chunk progress on stderr")
    a = ap.parse_args(argv)

    kw = {"is_gpu": False} if a.cpu else {}
    devices = [int(x) for x in a.devices.split(",")] if a.devices else None
    prog = compile_qasm(open(a.file).read(), path=a.file)
    paulis = [parse_pauli(s, prog.num_qubits) for s in a.pauli]
    ace_n = prog.num_qubits + (1 if paulis else 0)
    part, reps = partition_spec(a.partition, prog, a.lrc, a.lrr,
                                a.transpose, ace_n)
    part = compact(part)
    split = SeamSplit(prog, part)

    report = {"file": a.file, "qubits": prog.num_qubits,
              "partition": part,
              "patch_sizes": [len(q) for q in split.qubits_of],
              "cuts": [c.describe() for c in split.cuts]}
    pr = Progress("seams", 1, quiet=a.quiet)

    def chunk_cb(done, total):
        if pr.name is None:
            pr.start("chunk instances", total)
            pr.done = done - 1
        pr.step("%d worker%s" % (a.workers, "s" if a.workers > 1 else ""))
    kn = knit(split, paulis, a.max_terms, a.workers, devices, kw, a.emit,
              progress=chunk_cb)
    pr.finish()
    report["knit"] = kn
    ace = ref = None
    if not a.no_ace:
        t = time.perf_counter()
        runs = [AceRunner(ace_n, a.lrc, a.lrr, a.transpose, **kw).run(
            split, paulis, a.shots if r == 0 else 0)
            for r in range(max(1, a.ace_repeats))]
        ace = dict(runs[0])
        for f in ("marg", "pauli"):
            v = np.array([r[f] for r in runs], float).reshape(len(runs), -1)
            ace[f] = v.mean(axis=0).tolist()
            ace[f + "_std"] = v.std(axis=0).tolist()
        ace["repeats"] = len(runs)
        ace["time_s"] = time.perf_counter() - t
        report["ace"] = ace
    if not a.no_ref:
        t = time.perf_counter()
        ref = reference(split, paulis, kw)
        ref["time_s"] = time.perf_counter() - t
        report["ref"] = ref

    if a.json:
        json.dump(report, sys.stdout, indent=1, default=lambda o: (
            o.tolist() if isinstance(o, np.ndarray) else str(o)))
        print()
        return 0

    lab = QrackBackend.__new__(QrackBackend)._labels(prog)
    print("%s: %d qubits -> %d patches %s, %d seam cuts" % (
        a.file, prog.num_qubits, split.npatch, report["patch_sizes"],
        kn["cuts"]))
    print("knit: %d chunk instances (%d measurement branches), %d of %s "
          "terms, gamma %.4g, excluded one-norm %.3g" % (
              kn["chunks"], kn["leaves"], kn["terms"],
              "all" if kn["excluded"] < 1e-12 else "more", kn["gamma"],
              kn["excluded"]))
    if ace:
        L = ace["layout"]
        print("ace:  %d patch instances + %s, widths %s, seam qubits %s" % (
            L["patch_sims"], "crossbar" if L["crossbar"] else "no crossbar",
            L["sim_widths"], L["seam_qubits"]))
        print("      2q gates on/off ACE coupling map: %d/%d%s" % (
            L["gates_on_map"], L["gates_off_map"],
            "; Hadamard-test couplings on/off: %d/%d" % (
                L["hadamard_test_on_map"], L["hadamard_test_off_map"])
            if paulis else ""))
    bound = kn["excluded"]
    print("  %-14s %11s %11s   %s" % ("observable", "ref", "knit",
                                        "ace mean +- sd (%d runs)" %
                                        ace["repeats"] if ace else "ace"))
    fa = lambda f, i: ("%+.6f +- %.6f" % (ace[f][i], ace[f + "_std"][i])
                       if ace else "-")
    for i, c in enumerate(split.meas):
        print("  %-14s %11s %+11.6f   %s" % (
            "P(%s=1)" % lab[c], "%+.6f" % ref["marg"][i] if ref else "-",
            kn["marg"][i], fa("marg", i)))
    for o, s in enumerate(a.pauli):
        print("  %-14s %11s %+11.6f   %s" % (
            "<%s>" % (s if len(s) <= 12 else s[:9] + "..."),
            "%+.6f" % ref["pauli"][o] if ref else "-", kn["pauli"][o],
            fa("pauli", o)))
    if ref:
        dm = max([abs(x - y) for x, y in zip(kn["marg"], ref["marg"])] +
                 [abs(x - y) for x, y in zip(kn["pauli"], ref["pauli"])] +
                 [0.0])
        line = "  knit - ref: max %.2e (truncation bound %.2e)" % (dm, bound)
        if kn["dist"] is not None and "dist" in ref:
            line += ", TVD %.2e" % (0.5 * np.abs(kn["dist"] - ref["dist"]).sum())
        print(line)
        if ace:
            da = max([abs(x - y) for x, y in zip(ace["marg"], ref["marg"])] +
                     [abs(x - y) for x, y in zip(ace["pauli"], ref["pauli"])] +
                     [0.0])
            print("  ace  - ref: max %.2e (of the mean)" % da)
            seam = set(ace["layout"]["seam_qubits"])
            es = [abs(x - y) for q, x, y in zip(split.meas_q, ace["marg"],
                                                 ref["marg"]) if q in seam]
            eb = [abs(x - y) for q, x, y in zip(split.meas_q, ace["marg"],
                                                 ref["marg"]) if q not in seam]
            if es and eb:
                print("  ace  - ref by qubit class: seam max %.2e mean %.2e "
                      "(%d) | bulk max %.2e mean %.2e (%d)" % (
                          max(es), sum(es) / len(es), len(es), max(eb),
                          sum(eb) / len(eb), len(eb)))
    tt = kn["timings"]
    print("time: knit %.2fs (render %.2f, chunks %.2f, synth %.2f)%s%s" % (
        sum(tt.values()), tt["split_render_s"], tt["chunks_s"],
        tt["synthesis_s"], ", ace %.2fs" % ace["time_s"] if ace else "",
        ", ref %.2fs" % ref["time_s"] if ref else ""))
    return 0


# =====================================================================
# =====================================================================
# PARTS 3, 4 -- SELF-TESTS
# =====================================================================
# =====================================================================


def selftest_backend(verbose=False, quiet=False):
    """Backend vs Qiskit reference states. Returns 0 on success."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.circuit import library as L
    from qiskit.quantum_info import Statevector

    # test_qasm_qrack.py -- pin qasm_qrack.py against Qiskit reference states.
    #
    # Every case is built twice: as QASM text for the backend, and as a
    # Qiskit circuit object for the reference. The object route never goes
    # through a QASM parser, so parser bugs cannot agree with themselves.
    # Each circuit starts with a generic U layer so relative phases between
    # control branches are visible; states are compared by |<a|b>|, since
    # Qrack's global phase is arbitrary between runs.
    #
    # Qiskit is needed for this file only, never for the backend.
    #
    #   python3 test_qasm_qrack.py            (CPU, deterministic seed)




    BE = QrackBackend(is_gpu=False)
    TOL = 2e-5            # pyqrack wheels run single precision
    fails = []
    PR = Progress("backend", 6, fails, verbose, quiet)


    def fid(a, b):
        a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
        return abs(np.vdot(a / np.linalg.norm(a), b / np.linalg.norm(b)))


    def check(name, qasm, qc):
        try:
            _check(name, qasm, qc)
        finally:
            PR.step(name)

    def _check(name, qasm, qc):
        try:
            r = BE.run(qasm, shots=0, statevector=True)
        except Exception as e:
            fails.append("%s: raised %r" % (name, e))
            return
        f = fid(r.statevector, Statevector(qc).data)
        if f < 1 - TOL:
            fails.append("%s: fidelity %.9f" % (name, f))


    def prep(n, rng):
        ang = [[rng.uniform(0, 2 * math.pi) for _ in range(3)] for _ in range(n)]
        qasm = "".join("U(%r,%r,%r) q[%d];\n" % (*a, i) for i, a in enumerate(ang))
        qc = QuantumCircuit(n)
        for i, a in enumerate(ang):
            qc.u(*a, i)
        return qasm, qc


    def header(n):
        return 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[%d] q;\n' % n


    # name: (n_params, n_qubits, qiskit gate factory)
    GATES = {
        "id": (0, 1, lambda: L.IGate()), "x": (0, 1, lambda: L.XGate()),
        "y": (0, 1, lambda: L.YGate()), "z": (0, 1, lambda: L.ZGate()),
        "h": (0, 1, lambda: L.HGate()), "s": (0, 1, lambda: L.SGate()),
        "sdg": (0, 1, lambda: L.SdgGate()), "t": (0, 1, lambda: L.TGate()),
        "tdg": (0, 1, lambda: L.TdgGate()), "sx": (0, 1, lambda: L.SXGate()),
        "sxdg": (0, 1, lambda: L.SXdgGate()),
        "rx": (1, 1, L.RXGate), "ry": (1, 1, L.RYGate), "rz": (1, 1, L.RZGate),
        "p": (1, 1, L.PhaseGate), "phase": (1, 1, L.PhaseGate),
        "u1": (1, 1, L.U1Gate), "u2": (2, 1, L.U2Gate), "u3": (3, 1, L.U3Gate),
        "U": (3, 1, L.UGate), "u": (3, 1, L.UGate), "r": (2, 1, L.RGate),
        "cx": (0, 2, lambda: L.CXGate()), "CX": (0, 2, lambda: L.CXGate()),
        "cy": (0, 2, lambda: L.CYGate()), "cz": (0, 2, lambda: L.CZGate()),
        "ch": (0, 2, lambda: L.CHGate()), "cs": (0, 2, lambda: L.CSGate()),
        "csdg": (0, 2, lambda: L.CSdgGate()), "csx": (0, 2, lambda: L.CSXGate()),
        "crx": (1, 2, L.CRXGate), "cry": (1, 2, L.CRYGate),
        "crz": (1, 2, L.CRZGate), "cp": (1, 2, L.CPhaseGate),
        "cphase": (1, 2, L.CPhaseGate), "cu1": (1, 2, L.CU1Gate),
        "cu3": (3, 2, L.CU3Gate), "cu": (4, 2, L.CUGate),
        "swap": (0, 2, lambda: L.SwapGate()),
        "iswap": (0, 2, lambda: L.iSwapGate()),
        "dcx": (0, 2, lambda: L.DCXGate()), "ecr": (0, 2, lambda: L.ECRGate()),
        "rxx": (1, 2, L.RXXGate), "ryy": (1, 2, L.RYYGate),
        "rzz": (1, 2, L.RZZGate), "rzx": (1, 2, L.RZXGate),
        "ccx": (0, 3, lambda: L.CCXGate()), "ccz": (0, 3, lambda: L.CCZGate()),
        "cswap": (0, 3, lambda: L.CSwapGate()),
        "c3x": (0, 4, lambda: L.C3XGate()), "c4x": (0, 5, lambda: L.C4XGate()),
    }


    def t_every_gate(rng):
        PR.start("every gate", len(GATES))
        for name, (np_, nq, mk) in GATES.items():
            n = nq + 1
            pq, pc = prep(n, rng)
            ps = [rng.uniform(-3, 3) for _ in range(np_)]
            qs = rng.sample(range(n), nq)          # scrambled operand order
            qasm = header(n) + pq + "%s%s %s;\n" % (
                name, "(%s)" % ",".join(map(repr, ps)) if ps else "",
                ", ".join("q[%d]" % x for x in qs))
            pc.append(mk(*ps), qs)
            check("gate " + name, qasm, pc)


    def t_modifiers(rng):
        cases = [
            # (qasm line, qiskit gate, qubits)
            ("ctrl @ h q[0], q[1];", L.HGate().control(1), [0, 1]),
            ("negctrl @ x q[0], q[1];", L.XGate().control(1, ctrl_state=0),
             [0, 1]),
            ("negctrl(2) @ s q[0], q[1], q[2];",
             L.SGate().control(2, ctrl_state=0), [0, 1, 2]),
            ("ctrl @ negctrl @ inv @ t q[0], q[1], q[2];",
             L.TdgGate().control(1, ctrl_state=0).control(1), [0, 1, 2]),
            ("negctrl @ ctrl @ ry(0.7) q[2], q[0], q[1];",
             L.RYGate(0.7).control(1).control(1, ctrl_state=0), [2, 0, 1]),
            ("ctrl(2) @ rz(1.1) q[1], q[2], q[0];",
             L.RZGate(1.1).control(2), [1, 2, 0]),
            ("ctrl(2) @ U(0.3,1.2,-0.4) q[0], q[1], q[2];",
             L.UGate(0.3, 1.2, -0.4).control(2), [0, 1, 2]),
            ("inv @ u3(0.3,1.2,-0.4) q[1];", L.U3Gate(0.3, 1.2, -0.4).inverse(),
             [1]),
            ("pow(0.5) @ x q[0];", L.SXGate(), [0]),
            ("pow(0.5) @ z q[0];", L.SGate(), [0]),
            ("pow(3) @ t q[2];", L.TGate().power(3), [2]),
            ("pow(-2) @ s q[1];", L.SGate().power(-2), [1]),
            ("pow(0.3) @ h q[1];", L.HGate().power(0.3), [1]),
            ("pow(0.25) @ rx(0.8) q[0];", L.RXGate(0.2), [0]),
            ("ctrl @ pow(0.5) @ x q[1], q[0];", L.SXGate().control(1), [1, 0]),
            ("ctrl @ swap q[2], q[0], q[1];", L.SwapGate().control(1), [2, 0, 1]),
            ("negctrl @ swap q[2], q[0], q[1];",
             L.SwapGate().control(1, ctrl_state=0), [2, 0, 1]),
            ("ctrl @ negctrl @ swap q[3], q[2], q[0], q[1];",
             L.SwapGate().control(1, ctrl_state=0).control(1), [3, 2, 0, 1]),
            ("ctrl @ gphase(0.9) q[1];", L.PhaseGate(0.9), [1]),
            ("negctrl @ gphase(0.9) q[1];", None, [1]),
            ("ctrl(2) @ gphase(0.9) q[0], q[2];", L.CPhaseGate(0.9), [0, 2]),
            ("ctrl @ cu(0.3,0.2,0.1,0.8) q[0], q[1], q[2];",
             L.CUGate(0.3, 0.2, 0.1, 0.8).control(1), [0, 1, 2]),
            ("inv @ ctrl @ rxx(0.6) q[0], q[1], q[2];",
             L.RXXGate(0.6).control(1).inverse(), [0, 1, 2]),
            ("ctrl @ iswap q[3], q[1], q[0];", L.iSwapGate().control(1),
             [3, 1, 0]),
        ]
        PR.start("modifiers", len(cases))
        for line, g, qs in cases:
            pq, pc = prep(4, rng)
            if g is None:                       # negctrl @ gphase == X P X
                pc.x(qs[0])
                pc.p(0.9, qs[0])
                pc.x(qs[0])
            else:
                pc.append(g, qs)
            check("mod: " + line, header(4) + pq + line + "\n", pc)


    def t_user_gates(rng):
        PR.start("user gates", 2)
        # A global phase inside a user gate must survive ctrl @.
        src = header(3) + """
    gate ph(t) a { gphase(t); rz(t) a; }
    gate twist(t, s) a, b { ph(t) a; cx a, b; ry(s) b; ctrl @ ph(s) a, b; }
    """
        sub = QuantumCircuit(1, global_phase=0.4)
        sub.rz(0.4, 0)
        g_ph = sub.to_gate()
        pq, pc = prep(3, rng)
        pc.append(g_ph.control(1), [2, 0])
        check("user: ctrl @ gate with gphase",
              src + pq + "ctrl @ ph(0.4) q[2], q[0];\n", pc)

        def twist(t, s):
            c = QuantumCircuit(2)
            p1 = QuantumCircuit(1, global_phase=t)
            p1.rz(t, 0)
            p2 = QuantumCircuit(1, global_phase=s)
            p2.rz(s, 0)
            c.append(p1.to_gate(), [0])
            c.cx(0, 1)
            c.ry(s, 1)
            c.append(p2.to_gate().control(1), [0, 1])
            return c.to_gate()
        pq, pc = prep(3, rng)
        pc.append(twist(0.3, -0.8).control(1, ctrl_state=0).inverse(), [1, 2, 0])
        check("user: inv @ negctrl @ nested user gate",
              src + pq + "inv @ negctrl @ twist(0.3, -0.8) q[1], q[2], q[0];\n",
              pc)


    def t_syntax(rng):
        PR.start("syntax", 7)
        pq, pc = prep(4, rng)
        src = ('OPENQASM 3.0;\ninclude "stdgates.inc";\n'
               "const float[64] th = pi / 7;\ninput float phi;\n"
               "qubit[2] a;\nqubit[2] b;\n")
        pq = pq.replace("q[0]", "a[0]").replace("q[1]", "a[1]") \
               .replace("q[2]", "b[0]").replace("q[3]", "b[1]")
        body = ("h a;\ncx a, b;\nrz(th * 2 + phi) b[-1];\n"
                "ry(-(th ** 2) / 3) a[0:1];\nx b[{1, 0}];\n"
                "/* block\ncomment */ ctrl @ p(sin(th)) a[1], b[0];\n")
        for q in (0, 1):
            pc.h(q)
        pc.cx(0, 2)
        pc.cx(1, 3)
        th = math.pi / 7
        pc.rz(th * 2 + 0.25, 3)
        pc.ry(-(th ** 2) / 3, 0)
        pc.ry(-(th ** 2) / 3, 1)
        pc.x(3)
        pc.x(2)
        pc.cp(math.sin(th), 1, 2)
        r = BE.run(src + pq + body, shots=0, statevector=True,
                   params={"phi": 0.25})
        f = fid(r.statevector, Statevector(pc).data)
        if f < 1 - TOL:
            fails.append("syntax: registers/slices/consts/input fidelity %.9f" % f)
        PR.step("registers, slices, consts, input")

        q2 = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\n'
              "gate maj a,b,c { cx c,b; cx c,a; ccx a,b,c; }\n"
              "u3(0.1,0.2,0.3) q[0]; U(1,2,3) q[1]; CX q[0],q[2];\n"
              "maj q[0],q[1],q[2];\n")
        check("syntax: QASM 2 with user gate", q2,
              QuantumCircuit.from_qasm_str(q2))

        for bad, why in (("qubit[2] q; foo q[0];", "unknown gate"),
                         ("qubit[2] q; h q[2];", "out of range"),
                         ("qubit[2] q; cx q[0], q[0];", "repeated"),
                         ("qubit[1] q; for int i in [0:2] { h q; }", "not supported"),
                         ("input float z; qubit q;", "has no value")):
            try:
                compile_qasm("OPENQASM 3.0;\n" + bad)
                fails.append("syntax: accepted bad program (%s)" % why)
            except QasmError as e:
                if why not in str(e):
                    fails.append("syntax: wrong error for %s: %s" % (why, e))
            PR.step("rejects: " + why)


    POOL = [n for n in GATES if n not in ("c4x",)]


    def t_random(rng, trials=12, n=7, depth=250):
        PR.start("random circuits", trials)
        for tr in range(trials):
            pq, pc = prep(n, rng)
            lines = []
            for _ in range(depth):
                name = rng.choice(POOL)
                np_, nq, mk = GATES[name]
                ps = [rng.uniform(-3, 3) for _ in range(np_)]
                nc = rng.choice((0, 0, 0, 1, 2)) if nq + 2 <= n else 0
                qs = rng.sample(range(n), nq + nc)
                vals = [rng.choice((0, 1)) for _ in range(nc)]
                inv = rng.random() < 0.2
                g = mk(*ps)
                if inv:
                    g = g.inverse()
                mods = "inv @ " if inv else ""
                for v in reversed(vals):
                    g = g.control(1, ctrl_state=v)
                mods = "".join("ctrl @ " if v else "negctrl @ " for v in vals) \
                    + mods
                lines.append("%s%s%s %s;" % (
                    mods, name, "(%s)" % ",".join(map(repr, ps)) if ps else "",
                    ", ".join("q[%d]" % x for x in qs)))
                pc.append(g, qs)
            check("circuit %d: %d qubits, %d gates" % (tr + 1, n, depth), header(n) + pq + "\n".join(lines)
                  + "\n", pc)


    def t_dynamic():
        PR.start("dynamic circuits", 7)
        # Teleport |1>: the output qubit must read 1 on every shot.
        tele = header(3) + """bit[2] m;
    bit out;
    x q[0];
    h q[1]; cx q[1], q[2];
    cx q[0], q[1]; h q[0];
    m[0] = measure q[0];
    m[1] = measure q[1];
    if (m[1] == 1) x q[2];
    if (m[0]) z q[2];
    out = measure q[2];
    """
        r = BE.run(tele, shots=200, seed=7)
        outs = {k.split()[0] for k in r.counts}
        if r.mode != "per-shot" or outs != {"1"}:
            fails.append("dynamic: teleport gave %s (%s)" % (r.counts, r.mode))
        if sum(r.counts.values()) != 200:
            fails.append("dynamic: teleport shot count")
        PR.step("teleport")

        ghz = header(3) + "bit[3] c;\nh q[0]; cx q[0], q[1]; cx q[1], q[2];\n" \
            "c = measure q;\n"
        r = BE.run(ghz, shots=4000)                  # measure_shots path
        if set(r.counts) - {"000", "111"} or r.mode != "sampled":
            fails.append("dynamic: GHZ outcomes %s" % r.counts)
        if abs(r.counts.get("111", 0) / 4000 - 0.5) > 0.04:
            fails.append("dynamic: GHZ imbalance %s" % r.counts)
        if any(abs(p - 0.5) > 1e-5 for p in r.probabilities.values()):
            fails.append("dynamic: GHZ exact marginals %s" % r.probabilities)
        PR.step("GHZ, measure_shots path")

        # seeded: reproducible, per-shot, same support
        r1, r2 = BE.run(ghz, shots=300, seed=3), BE.run(ghz, shots=300, seed=3)
        if r1.counts != r2.counts or set(r1.counts) - {"000", "111"}:
            fails.append("dynamic: seeded GHZ %s vs %s" % (r1.counts, r2.counts))
        PR.step("seeded reproducibility")

        # c = [1,1,0] reads "011"; d (declared later) prints leftmost
        order = header(3) + "bit[3] c;\nbit[1] d;\nx q[0]; x q[1];\n" \
            "c = measure q;\nmeasure q[0] -> d[0];\n"
        r = BE.run(order, shots=10)
        if r.counts != {"1 011": 10}:
            fails.append("dynamic: bit ordering %s" % r.counts)
        PR.step("bit ordering")

        rst = header(1) + "bit c;\nx q[0];\nreset q[0];\nh q[0];\nh q[0];\n" \
            "c = measure q[0];\n"
        r = BE.run(rst, shots=50)
        if r.counts != {"0": 50}:
            fails.append("dynamic: reset %s" % r.counts)
        PR.step("reset")

        ifelse = header(2) + "bit a;\nbit b;\nx q[0];\na = measure q[0];\n" \
            "if (a == 0) { z q[1]; } else { h q[1]; z q[1]; h q[1]; }\n" \
            "b = measure q[1];\n"
        r = BE.run(ifelse, shots=20)
        if r.counts != {"1 1": 20}:
            fails.append("dynamic: if/else %s" % r.counts)
        PR.step("if / else")

        # register-valued condition: r = [0,1] has value 2, not 1
        regcond = header(3) + "bit[2] r;\nbit f;\nx q[1];\nr = measure q[0:1];\n" \
            "if (r == 2) x q[2];\nf = measure q[2];\n"
        r = BE.run(regcond, shots=20)
        if r.counts != {"1 10": 20}:
            fails.append("dynamic: register condition %s" % r.counts)
        PR.step("register condition")


    def main():
        rng = random.Random(12345)
        for t in (t_every_gate, t_modifiers, t_user_gates, t_syntax):
            t(rng)
        t_random(rng)
        t_dynamic()
        el = PR.finish()
        if fails:
            print("%d FAILURES" % len(fails))
            for f in fails:
                print("  " + f)
            return 1
        print("all checks passed (%d gates, modifiers, user gates, syntax, "
              "12 random circuits, dynamic circuits) in %s"
              % (len(GATES), fmt_s(el)))
        return 0

    return main()


def selftest_seams(verbose=False, quiet=False):
    """Knit vs exact, truncation bound, ACE lowering. Returns 0 on success."""
    import numpy as np

    # test_seam_runner.py -- knit route against the exact single-instance
    # reference on random circuits that exercise every cut type; the
    # truncation bound; and the ACE lowering on a seam-free layout (where
    # ACE must be exact, so any error is in the lowering, not the gadget).



    KW = {"is_gpu": False}
    fails = []
    PR = Progress("seams", 3, fails, verbose, quiet)

    def chunks(done, total):
        PR.substep("chunks %d/%d" % (done, total))

    ONE = ["h q[{a}];", "x q[{a}];", "sx q[{a}];", "t q[{a}];",
           "rx({p}) q[{a}];", "ry({p}) q[{a}];", "rz({p}) q[{a}];",
           "U({p},{r},{s}) q[{a}];"]
    TWO = ["cx q[{a}], q[{b}];", "cz q[{a}], q[{b}];", "cy q[{a}], q[{b}];",
           "ch q[{a}], q[{b}];", "crx({p}) q[{a}], q[{b}];",
           "crz({p}) q[{a}], q[{b}];", "cp({p}) q[{a}], q[{b}];",
           "cu({p},{r},{s},{p}) q[{a}], q[{b}];", "negctrl @ ry({p}) q[{a}], q[{b}];",
           "swap q[{a}], q[{b}];", "rzz({p}) q[{a}], q[{b}];",
           "h q[{a}]; h q[{b}]; cx q[{a}], q[{b}]; rz({p}) q[{b}]; "
           "cx q[{a}], q[{b}]; h q[{a}]; h q[{b}];"]
    THREE = ["ccx q[{a}], q[{c}], q[{b}];",
             "ctrl @ negctrl @ U({p},{r},{s}) q[{a}], q[{c}], q[{b}];"]


    def rand_circuit(rng, n, depth, part, multi=True):
        lines = ['OPENQASM 3.0;', 'include "stdgates.inc";',
                 "qubit[%d] q;" % n, "bit[%d] c;" % n]
        for q in range(n):
            lines.append("U(%r,%r,%r) q[%d];" % (rng.uniform(0, 3), rng.uniform(0, 6),
                                                 rng.uniform(0, 6), q))
        for _ in range(depth):
            r = rng.random()
            f = dict(p=rng.uniform(-3, 3), r=rng.uniform(-3, 3),
                     s=rng.uniform(-3, 3))
            if r < 0.45:
                f["a"] = rng.randrange(n)
                lines.append(rng.choice(ONE).format(**f))
            elif r < 0.9 or not multi:
                f["a"], f["b"] = rng.sample(range(n), 2)
                lines.append(rng.choice(TWO).format(**f))
            else:
                # two controls in one patch, target elsewhere (projector path)
                pa = rng.choice(sorted(set(part)))
                same = [q for q in range(n) if part[q] == pa]
                other = [q for q in range(n) if part[q] != pa]
                if len(same) < 2 or not other:
                    continue
                f["a"], f["c"] = rng.sample(same, 2)
                f["b"] = rng.choice(other)
                lines.append(rng.choice(THREE).format(**f))
        lines.append("c = measure q;")
        return "\n".join(lines) + "\n"


    def rand_pauli(rng, n):
        qs = rng.sample(range(n), rng.randint(2, n))
        return " ".join(rng.choice("XYZ") + str(q) for q in qs)


    def t_random(rng, trials=40):
        PR.start("random cut circuits", trials)
        done = refused = skipped = 0
        while done < trials:
            n = rng.randint(4, 7)
            k = rng.choice((2, 3))
            part = [min(k - 1, q * k // n) for q in range(n)]
            rng.shuffle(part)
            part = compact(part)
            src = rand_circuit(rng, n, rng.randint(6, 14), part)
            prog = compile_qasm(src)
            try:
                sp = SeamSplit(prog, part)
            except QasmError:
                refused += 1
                continue
            if not 1 <= len(sp.cuts) <= 5:
                skipped += 1
                continue
            ps = [rand_pauli(rng, n) for _ in range(2)]
            pl = [parse_pauli(s, n) for s in ps]
            PR.running("case %d: n=%d, %d patches, %d cuts, gamma %.1f"
                       % (done + 1, n, k, len(sp.cuts),
                          math.prod(c.gamma() for c in sp.cuts)))
            kn = knit(sp, pl, 10 ** 6, 1, None, KW, progress=chunks)
            ref = reference(sp, pl, KW)
            tol = 2e-6 * kn["gamma"] + 2e-5
            err = max(np.abs(kn["marg"] - ref["marg"]).max(),
                      np.abs(kn["pauli"] - ref["pauli"]).max(),
                      0.5 * np.abs(kn["dist"] - ref["dist"]).sum(),
                      abs(kn["trace"] - 1))
            if kn["excluded"] > 1e-9 or err > tol:
                fails.append("random n=%d cuts=%d gamma=%.1f: err %.2e (tol %.1e)"
                             "\n%s" % (n, len(sp.cuts), kn["gamma"], err, tol, src))
            done += 1
            PR.step("case %d: %d cuts, %d chunks  [resampled: %d out of "
                    "range, %d refused]" % (done, len(sp.cuts), kn["chunks"],
                                            skipped, refused))
        return refused


    def t_truncation(rng):
        # many weak ZZ bonds across one seam: exact is expensive, truncation
        # must stay inside its own bound
        n = 8
        part = [0] * 4 + [1] * 4
        lines = ['OPENQASM 3.0;', 'include "stdgates.inc";', "qubit[8] q;",
                 "bit[8] c;"]
        for q in range(n):
            lines.append("ry(%r) q[%d];" % (rng.uniform(0.3, 2.8), q))
        for step in range(3):
            for a in range(4):
                b = 4 + (a + step) % 4
                lines.append("rzz(0.2) q[%d], q[%d];" % (a, b))
                lines.append("rx(0.3) q[%d]; rx(0.3) q[%d];" % (a, b))
        lines.append("c = measure q;")
        prog = compile_qasm("\n".join(lines) + "\n")
        sp = SeamSplit(prog, part)
        pl = [parse_pauli("Z0 Z4 Z1 Z5", n)]
        PR.start("truncation bound", 3)
        ref = reference(sp, pl, KW)
        for budget in (30, 300, 3000):
            PR.running("budget %d" % budget)
            kn = knit(sp, pl, budget, 1, None, KW, progress=chunks)
            err = max(np.abs(kn["marg"] - ref["marg"]).max(),
                      np.abs(kn["pauli"] - ref["pauli"]).max())
            if err > kn["excluded"] + 1e-4:
                fails.append("truncation: budget %d err %.3e exceeds bound %.3e"
                             % (budget, err, kn["excluded"]))
            PR.note("  truncation: %d cuts, budget %5d -> err %.2e, bound %.2e"
                    % (len(sp.cuts), budget, err, kn["excluded"]))
            PR.step("budget %d" % budget)


    def t_ace_lowering(rng, trials=8):
        PR.start("ACE lowering", trials)
        for tr in range(trials):
            n = 6
            part = [0] * n
            src = rand_circuit(rng, n, 30, part, multi=False)
            prog = compile_qasm(src)
            sp = SeamSplit(prog, part)
            pl = [parse_pauli(rand_pauli(rng, n), n)]
            ref = reference(sp, pl, KW)
            # 7 qubits (6 + Hadamard-test ancilla) is prime -> 1D chain; long
            # ranges past its length leave no seam at all
            a = AceRunner(n + 1, 16, 16, False, **KW)
            if a.layout()["seam_qubits"]:
                fails.append("ace lowering: layout unexpectedly has seams")
                return
            out = a.run(sp, pl, 0)
            err = max(max(abs(x - y) for x, y in zip(out["marg"], ref["marg"])),
                      abs(out["pauli"][0] - ref["pauli"][0]))
            if err > 1e-4:
                fails.append("ace lowering trial %d: err %.2e\n%s" % (tr, err, src))
            PR.step("trial %d, seam-free layout" % (tr + 1))


    def main():
        rng = random.Random(2026)
        refused = t_random(rng)
        t_truncation(rng)
        t_ace_lowering(rng)
        el = PR.finish()
        if fails:
            print("%d FAILURES" % len(fails))
            for f in fails[:6]:
                print(" ", f)
            return 1
        print("all checks passed (40 random cut circuits, %d refused as "
              "unsupported partitions; truncation bound; ACE lowering) in %s"
              % (refused, fmt_s(el)))
        return 0

    return main()


# =====================================================================
# ENTRY POINT
# =====================================================================

def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    cmd = argv[0] if argv and argv[0] in ("run", "seams", "selftest") else None
    rest = argv[1:] if cmd else argv
    if cmd == "seams":
        return main_seams(rest)
    if cmd == "selftest":
        flags = {x for x in rest if x.startswith("-")}
        words = [x for x in rest if not x.startswith("-")]
        which = words[0] if words else "all"
        if which not in ("backend", "seams", "all") or len(words) > 1 or \
                flags - {"-v", "--verbose", "-q", "--quiet"}:
            raise SystemExit("selftest [backend|seams|all] [-v|--verbose] "
                             "[-q|--quiet]")
        v = bool(flags & {"-v", "--verbose"})
        q = bool(flags & {"-q", "--quiet"})
        rc = 0
        if which in ("backend", "all"):
            print("== backend selftest (vs Qiskit)", flush=True)
            rc |= selftest_backend(v, q)
        if which in ("seams", "all"):
            print("== seam selftest", flush=True)
            rc |= selftest_seams(v, q)
        return rc
    return main_run(rest)


if __name__ == "__main__":
    sys.exit(main())
