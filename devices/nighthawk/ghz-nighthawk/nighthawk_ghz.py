#!/usr/bin/env python3
"""
nighthawk_ghz.py -- Layer 1a (GHZ populations + MQC) and Layer 2 (idle-window refocusing) of
"Exact Refocusing of Diagonal Crosstalk for Shor Order Finding on a 10 x 12 Square-Lattice Processor"
(the formal proof; numpy reference shrs_nighthawk_stealth.py), run on Nighthawk registers under
QrackAceBackend, so the paper's statements meet ACE's seams.

Companion of nighthawk_qrack.py, imported as a library (registers, ACE presets and placement, the
coupler classes, the worker pool). Gates are exact; the only departures from the paper's model are
ACE's seam approximation and its sampler. An exact QrackSimulator reference runs beside every ACE run.

Paper objects as implemented here (section numbers of the paper)
  Lemma 1     the register's couplers join the sublattices A = {r + c even}, B = {r + c odd} of the device
              lattice (checked for every register)
  Def. 2      idle drift H_D = sum J_ij Z_i Z_j + sum (dw_i + eps_i) Z_i, with eta = 0 (A4): J_ij = j_zz x
              U(0.75, 1.25) per coupler (seeded), dw_q = 0.022 ((q mod 3) - 1), eps_i ~ N(0, sigma_qs^2)
              quasi-static: one draw per realisation (--repeats); defaults are the paper's section 8 values
  Def. 4      a block with G = I (Lemma 3): one idle window V; --blocks L of them after the preparation.
              No depolarising channel: ACE has no global D_p (A2); lambda from Lemma 3 then measures the
              population loss ACE's seams cause, as an effective survival, exact only under A2
  Def. 5      Walsh sequence (--layer2 dd, the anti-OTOC layer, route B): U(T/4) X_B U(T/4) X_A U(T/4) X_B
              U(T/4) X_A in time order; Theorem 1: W_T(H_D) = I for every J, dw, eps
  Def. 6      compensation (--layer2 compensated, route A): e^{+iT H_hat} e^{-iT H}, H_hat from the true
              J, dw (an oracle Layer 1b: Proposition 1 then leaves the eps term only)
  Lemma 3     lambda per block = [(P0 + P1 - 2/d) / (1 - 2/d)]^(1/L)
  Lemma 4     MQC: Tr[M(phi) rho] at phi_m = pi m / (n + 1), m = 0 .. 2n + 1; the DFT coefficient at n is
              c = rho_{0..0,1..1}; C_parity = 2|c|, Phi_coh = arg c, F_GHZ = (P0 + P1 + C) / 2 = max over
              theta of <GHZ_theta|rho|GHZ_theta>. 2n + 2 settings, not Clifford: units up to --mqc-max
  Cor. 4.1    sum dw = -Phi / (2 L T) (when the unit is the whole register: no outside couplings)
  Cor. 2.1    a GHZ state is blind to every Z_i Z_j inside it
  Def. 8      the OTOC is not estimated here (shrs_nighthawk_stealth.py computes it for the 2 x 4 plaquette);
              Corollary 7.1 (refocused OTOC constant) is what --layer2 dd tests through the GHZ numbers
Not in the paper: the Clifford witness F >= P0 + P1 + P(even X parity) - 1, used for chip-wide states too
wide for MQC, and the seam maps.

Seam layers (stack; not in the paper, which has no seams; named S and V to stay clear of its Layer 3, the
Shor payload)
  pass 1      Layer 1a: the units' GHZ states, no drift
  pass 2      Layer 2: the same units through L idle windows in the chosen mode (default dd)
  Layer S     seam error correction, in two steps.
              Extraction, no new circuits: per unit and pass the seam syndrome against the exact reference (the
              share of P0 + P1 and of the coherence ACE keeps, the qubit whose Z value leaves the unit's majority
              most often and whether it is a seam qubit, which branch loses: P1 - P0), and per unit its seam class:
                replica  one ACE simulator holds the whole unit (its seam qubits through their replica there): a
                         seam error is a disagreement between a seam qubit's replicas, a bit flip of the seam qubit
                         against its unit. Correctable: the replicas are a repetition code
                cut      no simulator holds the whole unit: its GHZ needs coherence between two simulators, and ACE
                         carries only classical correlations between them (a replica or the classical shadow), so
                         c = 0 there and F_GHZ <= (P0 + P1) / 2 <= 1/2 whatever a decoder does. Not correctable
              Also the seam-crossing fits of the passes (what the first crossing keeps, what each further one
              keeps) and per seam coupler its Z break rate.
              Correction: every replica unit whose fidelity fell below the exact reference (by 0.02 and 3 sigma
              in pass 1 or 2) gets the replica repetition code with authority decoding: ACE's
              is_boundary_repetition_code, zero ancillae, the replica that received the real gate is the decoder's
              ground truth and the stale replicas are flipped or rotated to it; and the code's schedule: from every
              qubit its CXs into the host simulator first, its CXs into seam qubits after (a CX between two seam
              qubits is matched in every simulator and moves the decoder's authority to the crossbar, where it
              stays for the partner when the other qubit couples on; the same state, CXs from one control commute).
              --code-schedule off keeps the preparation's order; --code-authority colocated (any run with the
              code) moves a seam partner's authority along instead, and needs no schedule. Cut units get no code.
              --zne-k adds zero-noise extrapolation for comparison: the seam units rerun pass 2 with every seam
              crossing scaled by lambda (AmpGates: K identities ZZ(phi) ZZ(-phi) after each, lambda = 1 + 4K),
              P0 + P1, Re c, Im c (or <X..X>) extrapolated to lambda = 0, linearly and exponentially.
  Layer V     the third GHZ measurement: the seam units through pass 2's circuit again (new eps draws, new ACE
              backends), the replica units under the code, the cut units without (the control: a rerun measures
              how far pass 2 moves by itself), compared unit by unit with pass 1, pass 2 and the exact reference.
  --order     echo-code (default): the order above. code-echo flips Layers 2 and S: pass 1, then Layer S from
              pass 1's syndrome alone, pass C (the third letter of code: pass 1's circuit, no drift, the replica
              units under the code, the cut units without), then pass E (the drift and the Walsh echo on every
              unit, the code still on where it ran). The drift arrives with its echo in both orders, so pass E
              runs the same circuit as echo-code's Layer V; what changes is the middle measurement and the
              syndrome the plan is read from.

Drift on ACE. e^{-i phi Z_i Z_j} is applied as CX_ij e^{-i phi Z_j} CX_ij, so a coupling across a seam goes
through ACE's seam machinery, as the circuit's own CXs do. A coupling from a GHZ unit to a qubit outside every
unit is applied as e^{-i phi z_j Z_i} on the unit qubit, with z_j = +-1 that qubit's Z eigenvalue, tracked
through the Walsh pulses: the same unitary, exactly, without merging units. Couplings among outside qubits are
global phases. Units of one circuit never share a coupler under --layer2 (tiles are spread into groups),
because a ZZ between two GHZ units entangles them and the register would no longer factorise.
--inside-zz stabilizer drops the couplings inside a unit instead: Z_i Z_j stabilises the unit's GHZ state and
every Walsh-flipped image of it, so e^{-i phi Z_i Z_j} is a global phase there (Corollary 2.1; selftest [13]),
while on ACE it is two CXs through the seam wherever the coupling crosses one. dw, eps and the couplings to
outside qubits stay. Valid for GHZ units only, which is all this script prepares; the exact reference runs the
same circuit.

Commands
  chip      one GHZ state over every qubit of the register: H on a root, then CX along a spanning tree
            (--trees bfs: breadth-first from the centre; seam: fewest CXs through a seam, Kruskal). With
            the witness readout (Clifford) and no Layer 2 the patches run as stabilizer hybrids, so large
            registers fit (measured: the two-patch ladder up to nighthawk2p57; not nighthawk61r2 on
            PyQrack 2.29.5 / Qrack 10.13.2, whose sampler leaves the tableau). With MQC or Layer 2 the
            patches are dense: small registers only.
  tiles     disjoint k-qubit GHZ chains (--unit k) along lattice rows and columns, every orientation and
            offset; per unit Layer 1a numbers, per qubit the mean over its units, by seam distance.
  stack     pass 1, pass 2, Layer S (seam error correction), Layer V on tiles (--mode tiles) or one state (--mode
            chip).
  sweep     tiles once per Walsh-echo window count (--windows 0,1,2,3,4,5,6,8: 0 = no Layer 2, dd
            otherwise), again under the boundary code at --code-at; F_GHZ, P0 + P1, C and |P1 - P0| per
            (setup, L, seam class). --summarise FILE... reads records instead of running.
  probe     the --seam-class units (default cut) through Layer 2 with the drift's ZZ inside a unit put
            through ACE several ways (--variants: none, gate, tseam, tbulk, stab, and chain-* with the GHZ
            prepared from one end), against the exact reference (--exact-variants); ghz_probe records.
  stackmap  a stack run's three measurements (pass 1, 2, V or 1, C, E) as heatmaps on the chip layout (--in).
  plot      the map of a records file (--in).
  selftest  Lemma 3, Lemma 4 and the witness against numpy; the readout gates, Theorem 1, Corollaries 2.1
            and 4.1, the outside-coupling identity, fold and the ZNE amplification on the exact simulator;
            the ZNE models on synthetic decays; Layer S's seam classes, plan, the code's schedule, the
            co-located authority rule, --inside-zz stabilizer and the probe's target and chain rules.

Per-unit hook (--layers): pre / after_cx / post inside a unit's preparation. fold:F[:all] adds F CX pairs
after every seam CX (every CX); zzfold:K[:all[:phi]] K identities ZZ(phi) ZZ(-phi); gfold:K undoes and redoes
the whole preparation K times. All logical identities; on ACE they add seam crossings (README 3.6: ACE's
seam loss barely follows them).

Records (--out, JSONL): one ghz_map per run, one ghz per state (chip: per tree; tiles: per unit) with the
reference implementation's Layer 1a names (P0, P1, C_parity, Phi_coh, F_GHZ, lambda_per_block,
sum_dw_estimate) for ACE and, prefixed exact_, for the exact reference. stack adds stack_pass (1, 2, V; with
--order code-echo 1, C, E) and stack_order to its ghz records (V, C, E: boundary_rep true where the code ran,
seam_plan), one ghz_stack per unit (seam_class, seam_plan, syndrome per pass, pass1, pass2, passV, exact1,
exact2, seam_crossings; code-echo: pass1, passC, passE, exact1, exactE; with --zne-k layerS: the observables at
every lambda and both fits, F_zne_linear, F_zne_exp) and one ghz_seam_model (Layer S's extraction and plan).
probe writes one ghz_probe record per unit, variant and reference (variant, exact, seam_class and the Layer 1a
names); sweep writes tiles' ghz records.

Examples
  python3 nighthawk_ghz.py selftest
  python3 nighthawk_ghz.py tiles --register nighthawk61r2 --unit 3 --repeats 4 --jobs 8 --out ghz61.jsonl
  python3 nighthawk_ghz.py tiles --register nighthawk61r2 --unit 3 --layer2 dd --blocks 3 --repeats 8 \\
          --jobs 8 --out ghz61_dd.jsonl
  python3 nighthawk_ghz.py chip  --register nighthawk2p57 --shots 4096 --repeats 5 --jobs 8 --out ghz57.jsonl
  python3 nighthawk_ghz.py chip  --register nighthawk2p16 --layer2 unmitigated --out ghz16.jsonl
  python3 nighthawk_ghz.py stack --register nighthawk61r2 --unit 3 --shots 256 --repeats 4 --jobs 8 --out stack61.jsonl
  python3 nighthawk_ghz.py stack --register nighthawk2p16 --mode chip --out stack16.jsonl
  python3 nighthawk_ghz.py stack --register nighthawk2p28 --unit 3 --zne-k 1,2 --out stack28.jsonl
  python3 nighthawk_ghz.py stack --register nighthawk61r2 --unit 3 --shots 256 --repeats 4 --order code-echo \\
          --out stack61_ce.jsonl
  python3 nighthawk_ghz.py stackmap --in stack61_ce.jsonl --png stack61_ce.png
  python3 nighthawk_ghz.py sweep --register nighthawk2p28 --shots 256 --repeats 4 --no-exact --code-at 8 \\
          --out sweep28.jsonl --summary sweep28.json
  python3 nighthawk_ghz.py probe --register nighthawk61r2 --shots 256 --repeats 4 --out probe61.jsonl
  python3 nighthawk_ghz.py plot  --in ghz61.jsonl --register nighthawk61r2 --png ghz61.png

Environment: no CUDA. CPU by default; --gpu runs ACE and the reference on the OpenCL device
QRACK_OCL_DEFAULT_DEVICE names (--ocl-sync: QRACK_OCL_SYNC=1, Qrack >= 10.13.2).
"""

import os

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):
    os.environ["PYQRACK_SHARED_LIB_PATH"] = QRACK_LIB_PATH

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))          # nighthawk_qrack.py next to this file (also under python -I)

import nighthawk_qrack as nq                 # noqa: E402

COLUMNS = 10          # nighthawk_qrack's device lattice: 12 rows x 10 columns (the paper writes 10 x 12)
LAYER2_MODES = ("none", "unmitigated", "compensated", "dd")


# ================================================================== register, placement, classes

class Register:
    """A register as ACE places it: layout, couplers, per qubit its ACE simulators (two or more for a
    seam qubit), per coupler its class, lattice cell and sublattice, seam distances."""

    def __init__(self, repo, reg, brep=False):
        if reg != "release" and reg not in nq.REGISTERS:
            nq.load_register_files()
        self.repo, self.name, self.brep = repo, reg, bool(brep)
        self.lay = nq.Layout(repo) if reg == "release" else nq.RegisterLayout(repo, reg)
        self.n = self.lay.n
        ra = nq.shor9_run_args(reg, repo, False, self.brep)          # CPU settings: placement only
        nq.apply_register_preset(ra, self.lay)
        self.map, sids, idx, row = nq.shor9_register_map(self.lay, ra)
        probe = nq.AceEngine(self.lay, self.n, ra)
        self.size, self.lrc, self.lrr, self.torus, self.opts = probe.size, probe.lrc, probe.lrr, probe.torus, probe.opts
        self.widths = list(probe.widths)
        del probe
        self.idx = list(idx)
        self.sids = [sids[idx[q]] for q in range(self.n)]
        self.seam = {q for q in range(self.n) if len(self.sids[q]) > 1}
        self.couplers = [tuple(c) for c in nq.logical_couplers(self.lay, self.n)]
        self.adj = collections.defaultdict(set)
        for a, b in self.couplers:
            self.adj[a].add(b)
            self.adj[b].add(a)
        self.cell = {q: (p // COLUMNS, p % COLUMNS) for q, p in enumerate(self.lay.l2p[:self.n])}
        self.sub = [(r + c) % 2 for r, c in (self.cell[q] for q in range(self.n))]      # Lemma 1: 0 = A, 1 = B
        bad = [(a, b) for a, b in self.couplers if self.sub[a] == self.sub[b]]
        if bad:
            raise SystemExit(f"ghz: {reg} is not bipartite under r + c parity (Lemma 1): couplers {bad[:5]}")
        self.seam_dist = self._bfs_dist(self.seam) if self.seam else {q: math.inf for q in range(self.n)}

    def region(self, q):
        return "seam" if q in self.seam else f"p{next(iter(self.sids[q]))}"

    def cls(self, a, b):
        sa, sb = self.sids[a], self.sids[b]
        return "exact" if len(sa) == 1 and sa == sb else "replica" if sa & sb else "cross"

    def _bfs_dist(self, sources, nodes=None):
        nodes = set(range(self.n)) if nodes is None else set(nodes)
        dist = {q: 0 for q in sources if q in nodes}
        frontier = list(dist)
        while frontier:
            nxt = []
            for u in frontier:
                for v in sorted(self.adj[u]):
                    if v in nodes and v not in dist:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        return {q: dist.get(q, math.inf) for q in nodes}

    def centre(self, nodes=None, adj=None):
        """The node of least eccentricity (ties: not a seam qubit, then the lowest index)."""
        adj = self.adj if adj is None else adj
        nodes = sorted(range(self.n) if nodes is None else nodes)
        best = None
        for s in nodes:
            dist, frontier = {s: 0}, [s]
            while frontier:
                nxt = []
                for u in frontier:
                    for v in adj[u]:
                        if v not in dist:
                            dist[v] = dist[u] + 1
                            nxt.append(v)
                frontier = nxt
            if len(dist) < len(nodes):
                continue
            key = (max(dist.values()), s in self.seam, s)
            if best is None or key < best[0]:
                best = (key, s)
        if best is None:
            raise SystemExit(f"ghz: the couplers of {self.name} do not connect its qubits")
        return best[1]

    def backend(self, gpu=False, stab=True):
        """A fresh QrackAceBackend on the register's placement and ACE options."""
        return nq.ace_backend(self.size, long_range_columns=self.lrc, long_range_rows=self.lrr, is_torus=self.torus,
                              is_gpu=bool(gpu), is_stabilizer_hybrid=bool(stab), **self.opts)

    def describe(self):
        return (f"{self.name} ({self.n} qubits, {len(self.seam)} seam), ACE {self.lrc}x{self.lrr}"
                f"{' torus' if self.torus else ''} on {self.size} sites, simulator widths {self.widths}")


_REGS = {}


def register(repo, reg, brep=False):
    key = (repo, reg, bool(brep))
    if key not in _REGS:
        _REGS[key] = Register(repo, reg, brep)
    return _REGS[key]


# ================================================================== ACE's boundary code: the authority rule

AUTHORITY_RULES = ("lineage", "colocated")
_LINEAGE = {}


def _colocate(self, lq1, hq1, lq2, hq2):
    """The co-located rule's step after PyQrack's lineage update (self: a QrackAceBackend). Every boundary qubit
    whose authority now sits in some simulators pulls the authority of each boundary partner it has met to that
    partner's replicas in those simulators, where it has one. Why: a CX between two seam qubits runs in every
    simulator and the lineage rule then makes the crossbar pair the sole authority; when one of the two later
    couples into a patch, its authority moves there, but its partner's stays on the crossbar, whose copy is
    correlated with a stale copy of the first (nighthawk61r2 [48, 40, 56]: F 0.58 under the code and the echo,
    1.00 with this step). The partner's copy in the new simulator shares the whole history."""
    P = self.__dict__.setdefault("_seam_partners", collections.defaultdict(set))
    if len(hq1) > 1 and len(hq2) > 1:
        P[lq1].add(lq2)
        P[lq2].add(lq1)
    for lq, hq in ((lq1, hq1), (lq2, hq2)):
        if len(hq) < 2 or lq not in self._authority:
            continue
        sims = {hq[i][0] for i in self._authority[lq]}
        for p in sorted(P.get(lq, ())):
            hp = self._unpack(p)
            new = {j for j, b in enumerate(hp) if b[0] in sims}
            if new and new != set(self._authority.get(p, range(len(hp)))):
                self._authority[p] = new


def set_authority_rule(rule):
    """Install the authority rule of ACE's boundary repetition code in this process, before a backend is built:
    lineage (PyQrack's own lineage tracking, 2.29.5) or colocated (lineage, then _colocate). The rule is read
    from the class at every coupling, so it holds for every backend this process builds afterwards."""
    from pyqrack import qrack_ace_backend as qab
    cls = qab.QrackAceBackend
    if "orig" not in _LINEAGE:
        if not hasattr(cls, "_update_lineage"):
            if rule == "lineage":
                return
            raise SystemExit("ghz: this PyQrack's QrackAceBackend has no lineage authority (_update_lineage, "
                             "PyQrack 2.29.5); --code-authority colocated needs it")
        _LINEAGE["orig"] = cls._update_lineage
    orig = _LINEAGE["orig"]
    if rule == "lineage":
        cls._update_lineage = orig
        return
    if rule != "colocated":
        raise ValueError(rule)

    def colocated(self, lq1, hq1, lq2, hq2):
        orig(self, lq1, hq1, lq2, hq2)
        if self.is_boundary_repetition_code and not self._in_gadget_capture and lq1 is not None and lq2 is not None:
            _colocate(self, lq1, hq1, lq2, hq2)
    cls._update_lineage = colocated


# ================================================================== Layer 2 model: Definition 2 (eta = 0)

class Drift:
    """H_D of Definition 2 on a register: J_ij per coupler (j_zz x U(0.75, 1.25), seeded as the reference
    does), dw_q = 0.022 ((q mod 3) - 1), quasi-static eps ~ N(0, sigma_qs^2) per realisation, window T."""

    def __init__(self, couplers, n, sub, j_zz=0.065, sigma_qs=0.015, t_idle=4.4, seed=42):
        self.couplers, self.n, self.sub = [tuple(c) for c in couplers], n, list(sub)
        self.j_zz, self.sigma, self.T, self.seed = float(j_zz), float(sigma_qs), float(t_idle), int(seed)
        self.J = self.j_zz * np.random.default_rng(self.seed).uniform(0.75, 1.25, len(self.couplers))
        self.dw = np.array([0.022 * ((q % 3) - 1.0) for q in range(n)])

    def eps(self, realisation):
        """The quasi-static detunings of one realisation (the same for ACE and the exact reference)."""
        if self.sigma == 0:
            return np.zeros(self.n)
        return np.random.default_rng([self.seed, 7919, int(realisation)]).normal(0.0, self.sigma, self.n)

    def describe(self):
        return dict(j_zz=self.j_zz, sigma_qs=self.sigma, t_idle=self.T, seed=self.seed, eta=0.0)


class Probe:
    """What the drift acts on in one circuit: the units' qubits, couplings inside a unit (gates), couplings
    from a unit qubit to a qubit outside every unit (z-signed local rotations), none between two units."""

    def __init__(self, drift, units, inside_zz="gate", seam=()):
        self.drift, self.inside_zz, self.seam = drift, inside_zz, set(seam)
        owner = {}
        for k, u in enumerate(units):
            for q in u["qubits"]:
                owner[q] = k
        self.qubits = sorted(owner)
        self.inside, self.outside = [], []
        for (a, b), J in zip(drift.couplers, drift.J):
            ia, ib = owner.get(a), owner.get(b)
            if ia is not None and ib is not None:
                if ia != ib:
                    raise RuntimeError(f"ghz: units {ia} and {ib} share coupler {a}-{b}; Layer 2 needs them apart")
                self.inside.append((a, b, float(J)))
            elif ia is not None:
                self.outside.append((a, b, float(J)))
            elif ib is not None:
                self.outside.append((b, a, float(J)))
        self.idle = sorted({b for _, b, _ in self.outside})


def drift_segment(g, probe, dt, eps, z, sign=1.0, with_eps=True):
    """e^{-i sign dt H} on the probe: H = H_D with (dw + eps) or, with_eps False, with dw only (H_hat).
    z: the Z eigenvalue (+1 / -1) of every outside qubit, as the Walsh pulses leave it."""
    d = probe.drift
    rule = probe.inside_zz
    for a, b, J in (probe.inside if rule != "stabilizer" else ()):     # stabilizer: a global phase on the GHZ
        if rule == "target-seam" and a in probe.seam and b not in probe.seam:
            a, b = b, a                                  # the probe's variants: the CX's target on the seam qubit
        elif rule == "target-bulk" and b in probe.seam and a not in probe.seam:
            a, b = b, a                                  # ... or on the bulk qubit (Z_a Z_b is symmetric)
        g.cx(a, b)
        g.zrot(b, sign * J * dt)
        g.cx(a, b)
    for a, b, J in probe.outside:
        g.zrot(a, sign * J * dt * z[b])
    for q in probe.qubits:
        g.zrot(q, sign * (d.dw[q] + (eps[q] if with_eps else 0.0)) * dt)


def walsh_pulse(g, probe, z, lattice):
    """X on every qubit of sublattice `lattice` (0 = A, 1 = B): the unit qubits get the gate, the outside
    qubits (Z eigenstates throughout) get their sign flipped."""
    sub = probe.drift.sub
    for q in probe.qubits:
        if sub[q] == lattice:
            g.x(q)
    for b in probe.idle:
        if sub[b] == lattice:
            z[b] = -z[b]


def idle_block(g, probe, mode, eps):
    """One idle window V of Definition 4 (G = I) in the given Layer 2 mode."""
    T = probe.drift.T
    z = {b: 1.0 for b in probe.idle}                     # outside qubits start in |0>
    if mode == "unmitigated":
        drift_segment(g, probe, T, eps, z)
    elif mode == "compensated":                          # Definition 6: e^{+iT H_hat} e^{-iT H}
        drift_segment(g, probe, T, eps, z)
        drift_segment(g, probe, T, eps, z, sign=-1.0, with_eps=False)
    elif mode == "dd":                                   # Definition 5, time order U X_B U X_A U X_B U X_A
        for lat in (1, 0, 1, 0):
            drift_segment(g, probe, T / 4, eps, z)
            walsh_pulse(g, probe, z, lat)
        assert all(v == 1.0 for v in z.values())
    elif mode != "none":
        raise ValueError(mode)


# ================================================================== GHZ units: trees and chains

def unit_from_tree(R, root, parent_of, order):
    """A unit (a dict, JSON-ready): qubits, root, tree edges parent -> child in preparation order, each
    edge's coupler class, the tree's depth."""
    depth = {root: 0}
    edges = []
    for c in order:
        if c == root:
            continue
        p = parent_of[c]
        depth[c] = depth[p] + 1
        edges.append([p, c])
    qubits = [root] + [c for _, c in edges]
    return dict(qubits=qubits, root=root, edges=edges, classes=[R.cls(p, c) for p, c in edges],
                depth=max(depth.values()))


def tree_bfs(R, root):
    parent, order, frontier = {root: None}, [root], [root]
    while frontier:
        nxt = []
        for u in frontier:
            for v in sorted(R.adj[u]):
                if v not in parent:
                    parent[v] = u
                    order.append(v)
                    nxt.append(v)
        frontier = nxt
    if len(order) != R.n:
        raise SystemExit(f"ghz: {R.name} is not connected from qubit {root}")
    return unit_from_tree(R, root, parent, order)


def tree_seam(R, root=None):
    """Spanning tree with the fewest CXs through a seam: Kruskal with inside-patch couplers first (ties
    toward the register's centre), then seam couplers, then couplers between simulators with no seam.
    Rooted at the tree's own centre unless root is given."""
    weight = {"exact": 0, "replica": 1, "cross": 1000}
    dc = R._bfs_dist({R.centre()})
    parent = list(range(R.n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    tadj = collections.defaultdict(set)
    for a, b in sorted(R.couplers, key=lambda e: (weight[R.cls(*e)], max(dc[e[0]], dc[e[1]]), e)):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            tadj[a].add(b)
            tadj[b].add(a)
    root = R.centre(adj=tadj) if root is None else root
    par, order, frontier = {root: None}, [root], [root]
    while frontier:
        nxt = []
        for u in frontier:
            for v in sorted(tadj[u]):
                if v not in par:
                    par[v] = u
                    order.append(v)
                    nxt.append(v)
        frontier = nxt
    if len(order) != R.n:
        raise SystemExit(f"ghz: {R.name} is not connected")
    return unit_from_tree(R, root, par, order)


TREES = dict(bfs=lambda R, root: tree_bfs(R, R.centre() if root is None else root),
             seam=lambda R, root: tree_seam(R, root))


def chain_unit(R, qs):
    """A GHZ chain on consecutive coupled qubits qs: root in the middle, CXs outwards along the chain."""
    m = (len(qs) - 1) // 2
    order = [qs[m]] + [qs[i] for k in range(1, len(qs)) for i in (m - k, m + k) if 0 <= i < len(qs)]
    parent = {}
    for i in range(len(qs)):
        if i < m:
            parent[qs[i]] = qs[i + 1]
        elif i > m:
            parent[qs[i]] = qs[i - 1]
    return unit_from_tree(R, qs[m], parent, order)


def tilings(R, k, orients=("rows", "cols"), offsets=None):
    """[(orientation, offset, [units])]: k-qubit chains along every lattice row (or column) of the register,
    split where a site is missing or two neighbours are not coupled, cut into chunks of k from the offset."""
    pos = {q: R.cell[q] for q in range(R.n)}
    out = []
    for ori in orients:
        line = collections.defaultdict(list)
        for q, (r, c) in pos.items():
            line[r if ori == "rows" else c].append(q)
        runs = []
        for key in sorted(line):
            qs = sorted(line[key], key=lambda q: pos[q][1] if ori == "rows" else pos[q][0])
            run = [qs[0]]
            for a, b in zip(qs, qs[1:]):
                step = (pos[b][1] - pos[a][1]) if ori == "rows" else (pos[b][0] - pos[a][0])
                if step == 1 and b in R.adj[a]:
                    run.append(b)
                else:
                    runs.append(run)
                    run = [b]
            runs.append(run)
        for f in (range(k) if offsets is None else offsets):
            units = [chain_unit(R, run[i:i + k]) for run in runs for i in range(f, len(run) - k + 1, k)]
            if units:
                out.append((ori, f, units))
    return out


def spread(R, units):
    """Split units into groups with no coupler between two units of a group (greedy colouring), for Layer 2."""
    sets = [set(u["qubits"]) for u in units]
    touch = [set().union(*(R.adj[q] for q in s)) | s for s in sets]
    groups = []
    for i, s in enumerate(sets):
        for gr in groups:
            if all(not (touch[i] & sets[j]) for j in gr):
                gr.append(i)
                break
        else:
            groups.append([i])
    return [[units[i] for i in gr] for gr in groups]


def unit_info(R, u):
    regions = collections.Counter(R.region(q) for q in u["qubits"])
    d = min(R.seam_dist[q] for q in u["qubits"])
    kind = "seam" if d == 0 else "near seam" if d == 1 else "bulk"
    return dict(regions=dict(regions), cx_classes=dict(collections.Counter(u["classes"])),
                seam_distance=(None if d == math.inf else int(d)), kind=kind,
                cells=[list(R.cell[q]) for q in u["qubits"]])


# ================================================================== per-unit layers

class Gates:
    """Logical-qubit gates on a backend (ACE through its placement, or the exact simulator)."""

    def __init__(self, sim, mapq):
        self.sim, self.q = sim, mapq

    def h(self, q):
        self.sim.h(self.q[q])

    def x(self, q):
        self.sim.x(self.q[q])

    def z(self, q):
        self.sim.z(self.q[q])

    def s(self, q):
        self.sim.s(self.q[q])

    def sdg(self, q):
        self.sim.adjs(self.q[q])

    def u(self, q, th, ph, lm):
        self.sim.u(self.q[q], th, ph, lm)

    def zrot(self, q, phi):
        """e^{-i phi Z} up to a global phase: diag(1, e^{2 i phi})."""
        self.sim.u(self.q[q], 0.0, 0.0, 2.0 * phi)

    def cx(self, c, t):
        self.sim.mcx([self.q[c]], self.q[t])


class AmpGates(Gates):
    """Layer S noise amplification: every CX through a seam (coupler class replica or cross), in the
    preparation and in the drift alike, is followed by K logical identities, so every seam crossing of the
    circuit is multiplied by the same scale lambda = 1 + per * K and nothing else changes.
      zz  ZZ(phi) ZZ(-phi), each as CX Rz CX, the second with control and target swapped (per = 4): a rotation
          between every two crossings, as the drift's own couplings cross
      cx  CX CX (per = 2): plain local folding
    Exact on the reference simulator (selftest [9])."""
    PER = dict(zz=4, cx=2)

    def __init__(self, sim, mapq, R, kind="zz", K=1, phi=0.3):
        super().__init__(sim, mapq)
        if kind not in self.PER:
            raise ValueError(kind)
        self.R, self.kind, self.K, self.phi = R, kind, int(K), float(phi)

    @classmethod
    def scale(cls, kind, K):
        return 1 + cls.PER[kind] * int(K)

    def cx(self, c, t):
        Gates.cx(self, c, t)
        if self.K <= 0 or self.R.cls(c, t) == "exact":
            return
        for _ in range(self.K):
            if self.kind == "zz":
                Gates.cx(self, c, t)
                self.zrot(t, self.phi)
                Gates.cx(self, c, t)
                Gates.cx(self, t, c)
                self.zrot(c, -self.phi)
                Gates.cx(self, t, c)
            else:
                Gates.cx(self, c, t)
                Gates.cx(self, c, t)


class Layer:
    """Base per-unit layer: nothing. A layer acts on one unit at three points of its preparation."""
    name = "none"

    def pre(self, g, unit):
        pass

    def after_cx(self, g, unit, p, c, cls):
        pass

    def post(self, g, unit):
        pass


class Fold(Layer):
    """fold:F[:all]: F extra CX pairs after every seam CX (or every CX). Logically the identity; on ACE
    each pair passes the seam twice more, so its error is scaled 2F + 1 times."""

    def __init__(self, F=1, scope="seam"):
        self.F, self.scope = int(F), scope
        self.name = f"fold:{self.F}" + (":all" if scope == "all" else "")

    def after_cx(self, g, unit, p, c, cls):
        if self.scope == "all" or cls != "exact":
            for _ in range(2 * self.F):
                g.cx(p, c)


class ZZFold(Layer):
    """zzfold:K[:all[:phi]]: after every seam CX (or every CX) K identities ZZ(phi) ZZ(-phi), each ZZ as the
    drift applies one (CX Rz CX), the second with control and target swapped so no two equal CXs meet.
    Four extra crossings of that coupler per identity, each with a rotation between, as Layer 2's windows
    cross it."""

    def __init__(self, K=1, scope="seam", phi=0.3):
        self.K, self.scope, self.phi = int(K), scope, float(phi)
        self.name = f"zzfold:{self.K}" + (":all" if scope == "all" else "") + (f":{phi:g}" if self.phi != 0.3 else "")

    def after_cx(self, g, unit, p, c, cls):
        if self.scope == "all" or cls != "exact":
            for _ in range(self.K):
                g.cx(p, c)
                g.zrot(c, self.phi)
                g.cx(p, c)
                g.cx(c, p)
                g.zrot(p, -self.phi)
                g.cx(c, p)


class GlobalFold(Layer):
    """gfold:K: after the preparation U, K times U^-1 U (the whole tree undone and redone): every CX of the
    unit, seam or not, 2K + 1 times, with the state each CX meets in the circuit itself."""

    def __init__(self, K=1):
        self.K = int(K)
        self.name = f"gfold:{self.K}"

    def post(self, g, unit):
        for _ in range(self.K):
            for p, c in reversed(unit["edges"]):        # U^-1 ends with H on the root and U starts with it:
                g.cx(p, c)                              # the pair cancels exactly and is left out
            for p, c in unit["edges"]:
                g.cx(p, c)


LAYERS = {
    "none": lambda *args: Layer(),
    "fold": lambda F="1", scope="seam": Fold(int(F), scope),
    "zzfold": lambda K="1", scope="seam", phi="0.3": ZZFold(int(K), scope, float(phi)),
    "gfold": lambda K="1": GlobalFold(int(K)),
}


def parse_layers(spec):
    """'none', 'fold:2', 'fold:1:all+...' -> [Layer]."""
    out = []
    for part in (spec or "none").split("+"):
        name, *args = part.strip().split(":")
        if name not in LAYERS:
            raise SystemExit(f"ghz: unknown layer {name!r}; known: {', '.join(sorted(LAYERS))}")
        lay = LAYERS[name](*args)
        if lay.name != "none":
            out.append(lay)
    return out


def layers_name(layers):
    return "+".join(L.name for L in layers) or "none"


def parse_layer2(spec):
    s = (spec or "none").strip().lower()
    s = {"walsh": "dd", "anti-otoc": "dd", "echo": "dd", "compensation": "compensated"}.get(s, s)
    if s not in LAYER2_MODES:
        raise SystemExit(f"ghz: unknown --layer2 {spec!r}; known: {', '.join(LAYER2_MODES)} (walsh = anti-otoc = dd)")
    return s


# ================================================================== circuit: preparation and readout

def prepare(g, unit, layers):
    for L in layers:
        L.pre(g, unit)
    g.h(unit["root"])
    for (p, c), cls in zip(unit["edges"], unit["classes"]):
        g.cx(p, c)
        for L in layers:
            L.after_cx(g, unit, p, c, cls)
    for L in layers:
        L.post(g, unit)


def readout(g, unit, setting):
    """Basis change before a Z measurement of every qubit.
      z     none (populations P0, P1)
      qM    MQC (Lemma 4): M(phi) = cos(phi) X + sin(phi) Y on every qubit, phi = pi M / (k + 1),
            M = 0 .. 2k + 1: H Rz(-phi) = u(pi/2, 0, pi - phi). q0 is the X basis.
      xrA   phase A pi/2 on the root (I, S-dagger, Z, S), then H on all: Clifford, any size"""
    if setting == "z":
        return
    k = len(unit["qubits"])
    if setting.startswith("q"):
        phi = int(setting[1:]) * math.pi / (k + 1)
        for q in unit["qubits"]:
            g.u(q, math.pi / 2, 0.0, math.pi - phi)
        return
    if setting.startswith("xr"):
        a = int(setting[2:]) % 4
        r = unit["root"]
        (lambda: None, lambda: g.sdg(r), lambda: g.z(r), lambda: g.s(r))[a]()
        for q in unit["qubits"]:
            g.h(q)
        return
    raise ValueError(setting)


def settings_for(k, estimator, mqc_max):
    """Readout settings for a k-qubit unit: mqc (z, q0 .. q(2k+1)), witness (z, xr0 .. xr3), or both."""
    est = estimator if estimator != "auto" else ("mqc" if k <= mqc_max else "witness")
    s = ["z"]
    if est in ("mqc", "both"):
        s += [f"q{m}" for m in range(2 * k + 2)]
    if est in ("witness", "both"):
        s += ["xr0", "xr1", "xr2", "xr3"]
    return s


def resolve_stabilizer(a, mode, settings, layer2):
    """--stabilizer auto: tableau patches for a chip-wide state with Clifford readout and no Layer 2; dense
    otherwise (tiles: QUnit keeps every chain separate, so dense is small)."""
    if a.stabilizer != "auto":
        return a.stabilizer == "on"
    return mode == "chip" and layer2 == "none" and all(s == "z" or s.startswith("xr") or s == "q0" for s in settings)


def ideal_parity(setting, k):
    """The parity every shot of |GHZ_0> gives (+1 / -1), None where it is random."""
    if setting in ("xr0", "q0"):
        return 1
    if setting == "xr2":
        return -1
    if setting == f"q{k + 1}":
        return -1 if k % 2 else 1
    return None


def _task(spec):
    """One circuit: a fresh backend (ACE, or the exact simulator), every unit prepared, the per-unit layers,
    Layer 2 (blocks of idle windows on the probe), the readout setting, `shots` samples of all register
    qubits (bit i = logical qubit i)."""
    R = register(spec["repo"], spec["reg"], spec["brep"])
    layers = parse_layers(spec["layers"])
    if spec["exact"]:
        from pyqrack import QrackSimulator
        sim = QrackSimulator(R.n, is_gpu=spec["gpu"], is_stabilizer_hybrid=spec["stab"])
        mapq = list(range(R.n))
    else:
        set_authority_rule(spec.get("authority", "lineage"))
        sim = R.backend(spec["gpu"], spec["stab"])
        mapq = R.idx
    amp = spec.get("amp")
    g = AmpGates(sim, mapq, R, amp["kind"], amp["K"], amp["phi"]) if amp and amp.get("K", 0) > 0 else Gates(sim, mapq)
    for u in spec["units"]:
        prepare(g, u, layers)
    l2 = spec.get("layer2") or {}
    if l2.get("mode", "none") != "none":
        drift = Drift(R.couplers, R.n, R.sub, l2["j_zz"], l2["sigma_qs"], l2["t_idle"], l2["seed"])
        probe = Probe(drift, spec["units"], l2.get("inside_zz", "gate"), R.seam)
        eps = drift.eps(spec["realisation"])
        for _ in range(int(l2["blocks"])):
            idle_block(g, probe, l2["mode"], eps)
    for u in spec["units"]:
        readout(g, u, spec["setting"])
    if spec["exact"]:
        shots = exact_shots(sim, R.n, spec["shots"], spec["stab"])
    else:
        try:
            shots = sim.measure_shots(list(R.idx), spec["shots"])
        except RuntimeError as err:
            if spec["stab"]:
                raise RuntimeError(f"ghz: ACE's sampler failed on a stabilizer-hybrid backend ({err}); a forced "
                                   f"seam-replica measurement had probability 0, which a tableau refuses: "
                                   f"rerun with --stabilizer off (dense patches)") from err
            raise
    del sim
    return [int(x) for x in shots]


def exact_shots(sim, n, shots, stab):
    """Samples of the exact simulator. With the stabilizer hybrid every shot is a clone measured qubit by
    qubit, which stays a tableau: QrackSimulator.measure_shots converts a state that is random in the
    measured basis to a dense ket first (2^28 amplitudes took 3 GB at 28 qubits). Dense and n <= 32:
    measure_shots (it returns bits 0-31 only, PyQrack 2.28-2.29)."""
    if not stab and n <= 32:
        return sim.measure_shots(list(range(n)), shots)
    out = []
    for _ in range(shots):
        c = sim.clone()
        out.append(sum(int(c.m(q)) << q for q in range(n)))
        del c
    return out


# ================================================================== estimators (Layer 1a)

def bit_matrix(shots, n):
    """shots (ints, bit i = qubit i) -> uint8 array shots x n."""
    nb = max(1, (n + 7) // 8)
    buf = b"".join(int(x).to_bytes(nb, "little") for x in shots)
    a = np.frombuffer(buf, dtype=np.uint8).reshape(len(shots), nb)
    return np.unpackbits(a, axis=1, bitorder="little")[:, :n]


def parity_mean(b):
    """<(-1)^(sum of bits)> over the shots of a shots x k bit array, and its standard error."""
    p = 1.0 - 2.0 * (b.sum(axis=1) & 1)
    m = float(p.mean())
    return m, math.sqrt(max(0.0, 1.0 - m * m) / len(p))


def estimate(data, k, blocks=1, t_idle=None):
    """Layer 1a numbers of one k-qubit unit from {setting: shots x k bits} (unit order: root first), in the
    reference implementation's names. blocks: L of Lemma 3; t_idle: T for Corollary 4.1 (None: no windows)."""
    out = {}
    d = 2.0 ** k
    if "z" in data:
        b = data["z"]
        S = len(b)
        w = b.sum(axis=1)
        P0, P1 = float(np.mean(w == 0)), float(np.mean(w == k))
        pop = P0 + P1
        out.update(P0=P0, P1=P1, pop=pop, pop_se=math.sqrt(max(pop * (1 - pop), 0.0) / S),
                   hamming=np.bincount(w, minlength=k + 1).tolist())
        lam_total = max(1e-6, (pop - 2 / d) / (1 - 2 / d))                     # Lemma 3
        out["lambda_per_block"] = lam_total ** (1.0 / max(1, int(blocks)))
    par = {s: parity_mean(v) for s, v in data.items() if s != "z"}
    out["parity"] = {s: round(m, 6) for s, (m, _) in par.items()}
    M = 2 * k + 2
    if "pop" in out and all(f"q{m}" in par for m in range(M)):                 # Lemma 4
        phis = math.pi * np.arange(M) / (k + 1)
        sig = np.array([par[f"q{m}"][0] for m in range(M)])
        var = np.array([par[f"q{m}"][1] ** 2 for m in range(M)])
        c = complex(np.mean(sig * np.exp(-1j * k * phis)))
        se_c = math.sqrt(var.sum()) / M                                          # per real component, about
        out.update(c_re=c.real, c_im=c.imag, c_se=se_c, C_parity=2 * abs(c), Phi_coh=math.atan2(c.imag, c.real),
                   F_GHZ=0.5 * (out["P0"] + out["P1"] + 2 * abs(c)),
                   F0=0.5 * (out["P0"] + out["P1"]) + c.real)                    # fidelity to GHZ_0
        out["F_GHZ_se"] = 0.5 * math.hypot(out["pop_se"], 2 * se_c)
        if t_idle:
            out["sum_dw_estimate"] = -out["Phi_coh"] / (2 * max(1, int(blocks)) * t_idle)   # Corollary 4.1
    x = par.get("q0") or par.get("xr0")
    if x is not None and "pop" in out:                                          # Clifford witness (not in the paper)
        out["x_parity"], out["x_se"] = x[0], x[1]
        out["F_witness"] = out["pop"] + (1.0 + x[0]) / 2.0 - 1.0
        out["F_witness_se"] = math.hypot(out["pop_se"], x[1] / 2.0)
    if all(f"xr{a}" in par for a in range(4)):
        re_ = (par["xr0"][0] - par["xr2"][0]) / 2.0
        im_ = (par["xr1"][0] - par["xr3"][0]) / 2.0
        out["C_x"] = math.hypot(re_, im_)          # every x / x-bar coherence pair, not only 0..0 / 1..1
        out["phase_x"] = math.atan2(im_, re_)
    return out


def headline(e):
    """The unit's fidelity: F_GHZ (Lemma 4) when measured, else the witness bound."""
    return ("F_GHZ", e["F_GHZ"]) if "F_GHZ" in e else ("F_witness", e.get("F_witness"))


def z_breaks(R, unit, bz):
    """Z-basis map of a unit (bz: shots x unit qubits, unit order): per tree edge and per coupler inside
    the unit the share of shots whose two qubits differ, per qubit the share where it differs from the
    unit's majority (ties: the root's bit)."""
    pos = {q: i for i, q in enumerate(unit["qubits"])}
    k = len(unit["qubits"])
    w = bz.sum(axis=1)
    maj = np.where(2 * w == k, bz[:, 0], (2 * w > k).astype(np.uint8))
    site = {q: float(np.mean(bz[:, pos[q]] != maj)) for q in unit["qubits"]}
    edges = [[p, c, R.cls(p, c), float(np.mean(bz[:, pos[p]] != bz[:, pos[c]]))] for p, c in unit["edges"]]
    qs = set(unit["qubits"])
    cpl = [[a, b, R.cls(a, b), float(np.mean(bz[:, pos[a]] != bz[:, pos[b]]))]
           for a, b in R.couplers if a in qs and b in qs]
    return site, edges, cpl


def score_unit(R, unit, per_rep, k, blocks=1, t_idle=None, maps=True):
    """Pooled (the realisations averaged, as the paper averages rho over mu) and per-realisation numbers."""
    cols = unit["qubits"]
    pooled = {s: np.concatenate([d[s][:, cols] for d in per_rep]) for s in per_rep[0]}
    est = estimate(pooled, k, blocks, t_idle)
    reps = [estimate({s: d[s][:, cols] for s in d}, k, blocks, t_idle) for d in per_rep]
    for key in ("F_GHZ", "F0", "F_witness", "C_parity", "Phi_coh", "pop"):
        vals = [r[key] for r in reps if key in r]
        if len(vals) > 1:
            est[key + "_runs"] = [round(v, 6) for v in vals]
            est[key + "_spread"] = [round(min(vals), 6), round(max(vals), 6)]
    if maps and "z" in pooled:
        site, edges, cpl = z_breaks(R, unit, pooled["z"])
        est.update(site_break={str(q): round(v, 6) for q, v in site.items()},
                   edge_break=[[p, c, cl, round(v, 6)] for p, c, cl, v in edges],
                   coupler_break=[[a, b, cl, round(v, 6)] for a, b, cl, v in cpl])
    return est


def check_exact(units, setting, shots_bits, parity_ideal):
    """Shots of the exact reference that are not ideal: all-equal in Z per unit (any diagonal drift keeps
    populations, Lemma 3's proof), and, where the idle unitary is the identity (no Layer 2, Theorem 1's
    echo, oracle compensation without eps), the deterministic MQC / X parities."""
    bad = 0
    for u in units:
        b = shots_bits[:, u["qubits"]]
        k = len(u["qubits"])
        if setting == "z":
            w = b.sum(axis=1)
            bad += int(np.sum((w != 0) & (w != k)))
        elif parity_ideal:
            want = ideal_parity(setting, k)
            if want is not None:
                p = 1 - 2 * (b.sum(axis=1).astype(np.int64) & 1)      # signed: the unsigned sum would wrap
                bad += int(np.sum(p != want))
    return bad


# ================================================================== running

def layer2_spec(a):
    mode = parse_layer2(a.layer2)
    if mode == "none":
        return dict(mode="none", blocks=0)
    l2 = dict(mode=mode, blocks=int(a.blocks), j_zz=float(a.j_zz), sigma_qs=float(a.sigma_qs),
              t_idle=float(a.t_idle), seed=int(a.seed), eta=0.0)
    if getattr(a, "inside_zz", "gate") == "stabilizer":
        l2["inside_zz"] = "stabilizer"            # absent: the gates (every record before this option)
    return l2


def identity_idle(l2):
    """True when the ideal idle unitary is the identity: no Layer 2, the Walsh echo (Theorem 1), or oracle
    compensation without quasi-static spread (Proposition 1 with eps = 0)."""
    m = l2["mode"]
    return m in ("none", "dd") or (m == "compensated" and l2.get("sigma_qs", 0) == 0)


def run_tasks(a, R, groups, settings, label, l2, amp=None, with_exact=True, rep_offset=0, brep=None):
    """Every (group of units, setting, realisation) on ACE and on the exact simulator. groups: [(name,
    [units])]. amp: Layer S amplification (AmpGates) or None. with_exact False: ACE only. rep_offset: first
    realisation index (fresh eps draws for Layer V). brep: ACE's boundary repetition code for these circuits
    (None: --ace-boundary-rep). Returns ({name: [ {setting: bits} per realisation ]} for ACE and for exact, bad,
    seconds)."""
    reps = max(1, int(a.repeats))
    ex_reps = reps if l2["mode"] != "none" else 1
    no_exact = a.no_exact or not with_exact
    base = dict(repo=a.repo, reg=R.name, brep=bool(a.ace_boundary_rep if brep is None else brep), layers=a.layers,
                gpu=bool(a.gpu), stab=a.stab, layer2=l2, amp=amp, authority=getattr(a, "code_authority", "lineage"))
    calls, labels, keys = [], [], []
    for name, units in groups:
        for s in settings:
            for r in range(reps):
                calls.append((_task, (dict(base, units=units, setting=s, shots=a.shots, exact=False,
                                           realisation=r + rep_offset),)))
                labels.append(f"ACE {label} {name} {s} realisation {r + 1}/{reps}")
                keys.append((name, s, r, False))
            if not no_exact:
                for r in range(ex_reps):
                    calls.append((_task, (dict(base, units=units, setting=s, shots=a.exact_shots, exact=True,
                                               realisation=r + rep_offset),)))
                    labels.append(f"exact {label} {name} {s} realisation {r + 1}/{ex_reps}")
                    keys.append((name, s, r, True))
    pool, workers = nq.shor_pool(a.jobs, len(calls))
    every = float(a.progress or 0)
    if every:
        print(f"# ghz plan: {len(calls)} circuits ({len(groups)} groups x {len(settings)} settings x {reps} "
              f"ACE realisations" + (f" + exact x {ex_reps}" if not no_exact else "") + f"), {workers} process(es)"
              + (f" [{label}]" if label else ""), flush=True)
    t0 = time.time()
    try:
        res = nq.shor_pool_map(pool, calls, labels, every, what="ghz")
    finally:
        if pool is not None:
            pool.shutdown()
    data = {name: [dict() for _ in range(reps)] for name, _ in groups}
    exact = {name: [dict() for _ in range(ex_reps)] for name, _ in groups}
    unit_of = dict(groups)
    bad = 0
    pid = identity_idle(l2)
    for (name, s, r, ex), shots in zip(keys, res):
        bits = bit_matrix(shots, R.n)
        if ex:
            bad += check_exact(unit_of[name], s, bits, pid)
            exact[name][r][s] = bits
        else:
            data[name][r][s] = bits
    return data, (None if no_exact else exact), (None if no_exact else bad), time.time() - t0


def header(a, R, what, l2):
    stab = "stabilizer hybrid" if a.stab else "dense"
    dev = ("OpenCL" + (" (QRACK_OCL_SYNC=1)" if os.environ.get("QRACK_OCL_SYNC") == "1" else "")) if a.gpu else "CPU"
    lay2 = ("Layer 2 none" if l2["mode"] == "none" else
            f"Layer 2 {l2['mode']}: {l2['blocks']} blocks, T {l2['t_idle']}, J {l2['j_zz']} x U(0.75, 1.25), "
            f"sigma_qs {l2['sigma_qs']}, seed {l2['seed']}, eta 0"
            + (", ZZ inside a unit dropped (stabilizer)" if l2.get("inside_zz") == "stabilizer" else ""))
    return (f"{what} on {R.describe()}; {a.shots} shots x {a.repeats} realisation(s) per setting, {lay2}, "
            f"layers {a.layers or 'none'}, {stab}, {dev}"
            + (", boundary repetition code" if a.ace_boundary_rep else "")
            + (f", code authority {a.code_authority}" if getattr(a, "code_authority", "lineage") != "lineage" else "")
            + f"; {nq.qrack_versions()}")


def base_record(a, R, l2):
    return dict(register=R.name, n=R.n, shots=a.shots, repeats=a.repeats, layers=layers_name(parse_layers(a.layers)),
                layer2=l2, estimator=a.estimator, stabilizer=bool(a.stab), boundary_rep=bool(a.ace_boundary_rep),
                code_authority=getattr(a, "code_authority", "lineage"),
                **({"engine": "loom", "loom": R.map["loom"]} if getattr(R, "engine", "ace") == "loom" else {}),
                device="gpu" if a.gpu else "cpu", ocl_sync=os.environ.get("QRACK_OCL_SYNC") == "1",
                qrack=nq.qrack_versions(), ace=dict(lrc=R.lrc, lrr=R.lrr, torus=bool(R.torus), width=R.size,
                                                   widths=R.widths))


def exact_fields(R, unit, per_rep, k, l2):
    """The exact reference's Layer 1a numbers for a unit, prefixed exact_ (no maps)."""
    if not per_rep:
        return {}
    e = score_unit(R, unit, per_rep, k, max(1, l2["blocks"]), l2.get("t_idle") if l2["mode"] != "none" else None,
                   maps=False)
    keep = ("P0", "P1", "C_parity", "Phi_coh", "F_GHZ", "F0", "lambda_per_block", "sum_dw_estimate", "F_witness",
            "x_parity", "C_x", "phase_x")
    return {"exact_" + key: e[key] for key in keep if key in e}


def write_records(a, R, recs, bad):
    if not a.out or bad:
        return
    when = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(a.out, "a") as fh:
        fh.write(json.dumps(dict(test="ghz_map", register=R.name, when=when, seam=sorted(R.seam), sublattice=R.sub,
                                 seam_distance={str(q): (None if d == math.inf else int(d))
                                                for q, d in R.seam_dist.items()}, **R.map)) + "\n")
        for r in recs:
            fh.write(json.dumps(dict(r, when=when)) + "\n")
    print(f"  appended {len(recs)} records and the register map to {a.out}")


def stop_on_bad(a, bad):
    if bad:
        msg = (f"ghz: the exact reference gave {bad} non-ideal shots, so this run's numbers are not recorded")
        if a.gpu:
            msg += "; on OpenCL use Qrack >= 10.13.2 with --ocl-sync, or drop --gpu (nighthawk_qrack README 5.3)"
        raise SystemExit(msg)


def fmt(v, nd=3):
    return "  -  " if v is None else f"{v:.{nd}f}"


def ascii_map(R, value, nd=2):
    """The register on its lattice, one cell per site: the value, '*' after a seam qubit, '.' for no site."""
    cells = {R.cell[q]: q for q in range(R.n)}
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    w = nd + 3
    lines = ["      " + "".join(f"c{c:<{w}}" for c in range(min(cols), max(cols) + 1))]
    for r in range(min(rows), max(rows) + 1):
        s = f"  r{r:<3}"
        for c in range(min(cols), max(cols) + 1):
            q = cells.get((r, c))
            if q is None:
                s += " " * (w // 2) + "." + " " * (w - w // 2)
            else:
                v = value.get(q)
                s += (f"{v:.{nd}f}" if v is not None else "-" * (nd + 2)) + ("*" if q in R.seam else " ")
        lines.append(s)
    return "\n".join(lines)


def print_unit_line(name, e, ex):
    key, F = headline(e)
    s = (f"{name:<6} P0+P1 {fmt(e.get('pop'))}  ")
    if "F_GHZ" in e:
        s += (f"C {fmt(e['C_parity'])}  Phi {fmt(e['Phi_coh'], 2):>5}  F_GHZ {e['F_GHZ']:.3f} +/- {e['F_GHZ_se']:.3f}  "
              f"F0 {e['F0']:.3f}  lambda/block {fmt(e.get('lambda_per_block'), 4)}")
        if "sum_dw_estimate" in e:
            s += f"  sum dw est {e['sum_dw_estimate']:+.4f}"
    if "F_witness" in e:
        s += f"  F >= {e['F_witness']:.3f} (witness)"
    if "C_x" in e:
        s += f"  C_x {e['C_x']:.3f} phase_x {e['phase_x']:+.2f}"
    if ex:
        s += ("   | exact: " + "  ".join(f"{k_[6:]} {v:.3f}" for k_, v in ex.items()
                                         if k_ in ("exact_F_GHZ", "exact_Phi_coh", "exact_F_witness", "exact_P0")))
    print(s)


# ================================================================== chip

def cmd_chip(a):
    R = register(a.repo, a.register, a.ace_boundary_rep)
    l2 = layer2_spec(a)
    names = [t.strip() for t in a.trees.split(",") if t.strip()]
    for t in names:
        if t not in TREES:
            raise SystemExit(f"ghz: unknown tree {t!r}; known: {', '.join(TREES)}")
    trees = [(t, TREES[t](R, a.root)) for t in names]
    settings = settings_for(R.n, a.estimator, a.mqc_max)
    a.stab = resolve_stabilizer(a, "chip", settings, l2["mode"])
    if not a.stab and R.n > 30 and not a.force:
        raise SystemExit(f"ghz: {R.n} qubits with dense patches (MQC readout or Layer 2) do not fit: the GHZ state spans "
                         f"each patch ({R.widths}); use a register of <= 30 qubits, the witness readout without "
                         f"Layer 2, or --force")
    print(header(a, R, f"GHZ over all {R.n} qubits", l2))
    print("tree  root  depth  CX exact/replica/cross  readout settings: " + " ".join(settings))
    for t, u in trees:
        c = collections.Counter(u["classes"])
        print(f"{t:<5} {u['root']:>4}  {u['depth']:>5}  {c.get('exact', 0):>5}/{c.get('replica', 0)}/{c.get('cross', 0)}")
    data, exact, bad, secs = run_tasks(a, R, [(t, [u]) for t, u in trees], settings, "chip", l2)
    stop_on_bad(a, bad)
    L, T = max(1, l2["blocks"]), (l2.get("t_idle") if l2["mode"] != "none" else None)
    print()
    recs = []
    for t, u in trees:
        e = score_unit(R, u, data[t], R.n, L, T)
        ex = exact_fields(R, u, exact[t] if exact else None, R.n, l2)
        print_unit_line(t, e, ex)
        recs.append(dict(base_record(a, R, l2), test="ghz", mode="chip", tree=t, **u,
                         cx_classes=dict(collections.Counter(u["classes"])), **e, **ex))
    if l2["mode"] != "none":
        d = Drift(R.couplers, R.n, R.sub, l2["j_zz"], l2["sigma_qs"], l2["t_idle"], l2["seed"])
        print(f"  Corollary 4.1 reference: sum dw over the register {float(d.dw.sum()):+.4f} (eps averages to 0)")
    for t, u in trees:
        e = next(r for r in recs if r["tree"] == t)
        worst = sorted(e.get("edge_break", []), key=lambda x: -x[3])[:8]
        print(f"\n{t}: tree edges that break the Z pattern most (parent child class rate): "
              + ", ".join(f"{p}-{c} {cl} {v:.3f}" for p, c, cl, v in worst))
        by = collections.defaultdict(list)
        for p, c, cl, v in e.get("edge_break", []):
            by[cl].append(v)
        print("  mean break rate by edge class: " + ", ".join(f"{cl} {np.mean(v):.4f} ({len(v)})"
                                                         for cl, v in sorted(by.items())))
        print(f"  share of shots where each site differs from the majority ({t} tree; * seam qubit):")
        print(ascii_map(R, {int(q): v for q, v in e.get("site_break", {}).items()}, 3))
    if bad is not None:
        print(f"\nexact reference (QrackSimulator, {R.n} qubits, {a.exact_shots} shots per setting and realisation): "
              + ("every checked shot ideal" if bad == 0 else f"{bad} shots NOT ideal"))
    print(f"seconds: {secs:.1f}")
    write_records(a, R, recs, bad)


# ================================================================== tiles

def tile_groups(a, R, k, apart):
    """[(name, [units])]: k-qubit chains in every orientation and offset of --orient / --offsets, one group per
    tiling or, apart, split so no coupler joins two units of a group (Layer 2 needs that)."""
    if k < 2:
        raise SystemExit("ghz: --unit must be 2 or more")
    offs = None if a.offsets in (None, "all") else [int(x) for x in a.offsets.split(",")]
    tl = tilings(R, k, tuple(x.strip() for x in a.orient.split(",")), offs)
    if not tl:
        raise SystemExit(f"ghz: no chain of {k} coupled qubits along a lattice line of {R.name}")
    groups = []
    for ori, f, units in tl:
        parts = spread(R, units) if apart else [units]
        for j, part in enumerate(parts):
            groups.append((f"{ori}{f}" + (f".{j}" if len(parts) > 1 else ""), part))
    return groups


def cmd_tiles(a):
    R = register(a.repo, a.register, a.ace_boundary_rep)
    l2 = layer2_spec(a)
    k = int(a.unit)
    groups = tile_groups(a, R, k, l2["mode"] != "none")
    settings = settings_for(k, a.estimator, a.mqc_max)
    a.stab = resolve_stabilizer(a, "tiles", settings, l2["mode"])
    nunits = sum(len(u) for _, u in groups)
    print(header(a, R, f"{k}-qubit GHZ chains tiled", l2))
    print(f"{len(groups)} circuit groups ({', '.join(f'{g} {len(u)}' for g, u in groups)} units), {nunits} units; "
          f"readout settings: {' '.join(settings)}")
    data, exact, bad, secs = run_tasks(a, R, groups, settings, "tiles", l2)
    stop_on_bad(a, bad)
    L, T = max(1, l2["blocks"]), (l2.get("t_idle") if l2["mode"] != "none" else None)
    recs = []
    for g, units in groups:
        for j, u in enumerate(units):
            e = score_unit(R, u, data[g], k, L, T)
            ex = exact_fields(R, u, exact[g] if exact else None, k, l2)
            recs.append(dict(base_record(a, R, l2), test="ghz", mode="tiles", unit_size=k, tiling=g, unit=j, **u,
                             **unit_info(R, u), **e, **ex))
    key = headline(recs[0])[0]
    print(f"\nby seam distance ({key}; 0 = the unit holds a seam qubit): units, mean, min, max"
          + (", exact reference mean" if exact else ""))
    by = collections.defaultdict(list)
    for r in recs:
        by[r["seam_distance"]].append(r)
    for d in sorted(by, key=lambda x: (x is None, x)):
        v = [r[key] for r in by[d]]
        xv = [r.get("exact_" + key) for r in by[d] if r.get("exact_" + key) is not None]
        print(f"  {str(d):>4}  {len(v):>4}  {np.mean(v):.3f}  {min(v):.3f}  {max(v):.3f}"
              + (f"   {np.mean(xv):.3f}" if xv else ""))
    if "Phi_coh" in recs[0]:
        ph = [r["Phi_coh"] for r in recs if r["seam_distance"] not in (0, None)]
        xph = [r["exact_Phi_coh"] for r in recs if r["seam_distance"] not in (0, None) and "exact_Phi_coh" in r]
        if ph:
            print(f"  Phi_coh off the seam: ACE mean {np.mean(ph):+.3f}, spread {np.std(ph):.3f}"
                  + (f"; exact mean {np.mean(xph):+.3f}" if xph else ""))
    print(f"\nlowest {min(10, len(recs))} units ({key}, regions, CX exact/replica/cross):")
    for r in sorted(recs, key=lambda r: r[key])[:10]:
        c = r["cx_classes"]
        print(f"  {r['tiling']:>8} #{r['unit']:<3} qubits {r['qubits']}  {r[key]:.3f}  "
              f"{', '.join(f'{x} {y}' for x, y in sorted(r['regions'].items()))}  "
              f"{c.get('exact', 0)}/{c.get('replica', 0)}/{c.get('cross', 0)}")
    per_q = collections.defaultdict(list)
    for r in recs:
        for q in r["qubits"]:
            per_q[q].append(r[key])
    print(f"\nper site: mean {key} of the units holding it (* seam qubit):")
    print(ascii_map(R, {q: float(np.mean(v)) for q, v in per_q.items()}, 2))
    if bad is not None:
        print(f"\nexact reference (QrackSimulator, {R.n} qubits, {a.exact_shots} shots per setting, group and "
              f"realisation): " + ("every checked shot ideal" if bad == 0 else f"{bad} shots NOT ideal"))
    print(f"seconds: {secs:.1f}  (shot noise alone on {key}: about +/- "
          f"{np.mean([r.get(key + '_se', 0) for r in recs]):.3f} per unit)")
    write_records(a, R, recs, bad)


# ================================================================== stack: Layer S (seam error correction), Layer V

SEGMENTS = dict(none=0, unmitigated=1, compensated=2, dd=4)      # drift segments per idle window (idle_block)
KEEP = ("P0", "P1", "pop", "C_parity", "Phi_coh", "F_GHZ", "F_GHZ_se", "F0", "F_witness", "F_witness_se", "x_parity",
        "lambda_per_block")


def seam_crossings(R, u, l2):
    """Seam crossings of one unit's circuit: the preparation's CXs through a seam, and the drift's, two CXs per
    coupling inside the unit through a seam, per segment of every window (outside couplings are rotations)."""
    prep = sum(1 for c in u["classes"] if c != "exact")
    qs = set(u["qubits"])
    inside = sum(1 for a_, b_ in R.couplers if a_ in qs and b_ in qs and R.cls(a_, b_) != "exact")
    if l2.get("inside_zz") == "stabilizer":
        inside = 0                                   # the drift's couplings inside the unit are not applied
    return prep, 2 * inside * SEGMENTS[l2["mode"]] * max(0, int(l2.get("blocks") or 0))


def lin_obs(e):
    """The observables linear in rho that the fidelity is built from, with standard errors: P0 + P1 and
    c = rho_{0..0,1..1} (Lemma 4), or P0 + P1 and the X parity (witness)."""
    o = dict(pop=e["pop"], pop_se=e["pop_se"])
    if "c_re" in e:
        o.update(c_re=e["c_re"], c_im=e["c_im"], c_se=e["c_se"])
    else:
        o.update(x=e["x_parity"], x_se=e["x_se"])
    return o


def fidelity_from(o):
    """F_GHZ = (P0 + P1 + 2|c|) / 2 (Lemma 4), or the witness P0 + P1 + (1 + <X..X>) / 2 - 1."""
    if "c_re" in o:
        return 0.5 * (o["pop"] + 2 * abs(complex(o["c_re"], o["c_im"])))
    return o["pop"] + (1 + o["x"]) / 2 - 1


def wls(x, y, se):
    """Weighted least-squares line y = p0 + p1 x: (p0, se of p0, p1). One point: (y, se, 0)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = 1.0 / np.maximum(np.asarray(se, float), 1e-6) ** 2
    if len(x) == 1:
        return float(y[0]), float(1 / math.sqrt(w[0])), 0.0
    A = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    p = cov @ (A.T @ (w * y))
    return float(p[0]), float(math.sqrt(max(cov[0, 0], 0.0))), float(p[1])


def zne_extrapolate(lams, obs, k):
    """Zero-noise extrapolation of one unit's observables measured at seam scales lams (lambda = 1: the circuit
    itself, lambda = 0: no seam error).
      linear  a weighted line through each linear observable (P0 + P1, Re c, Im c or <X..X>), read at 0
      exp     A e^{-r lambda} through the population excess (P0 + P1 - 2/d) / (1 - 2/d) (Lemma 3's survival) and
              through |c| (or <X..X>), read at 0; None where a value is not positive
    Returns {model: dict(obs at 0, F, F_se, slopes, physical) or None}."""
    d = 2.0 ** k
    mqc = "c_re" in obs[0]
    out = {}
    pop0, pop_se, pop_sl = wls(lams, [o["pop"] for o in obs], [o["pop_se"] for o in obs])
    if mqc:
        re0, re_se, _ = wls(lams, [o["c_re"] for o in obs], [o["c_se"] for o in obs])
        im0, im_se, _ = wls(lams, [o["c_im"] for o in obs], [o["c_se"] for o in obs])
        o0 = dict(pop=pop0, c_re=re0, c_im=im0)
        cse = max(re_se, im_se)
        F_se = 0.5 * math.hypot(pop_se, 2 * cse)
        phys = pop0 <= 1 + 2 * pop_se and abs(complex(re0, im0)) <= pop0 / 2 + 2 * cse
    else:
        x0, x_se, _ = wls(lams, [o["x"] for o in obs], [o["x_se"] for o in obs])
        o0 = dict(pop=pop0, x=x0)
        F_se = math.hypot(pop_se, x_se / 2)
        phys = pop0 <= 1 + 2 * pop_se and x0 <= 1 + 2 * x_se
    out["linear"] = dict(obs=o0, F=fidelity_from(o0), F_se=F_se, pop_slope=pop_sl, physical=bool(phys))
    exc = np.array([(o["pop"] - 2 / d) / (1 - 2 / d) for o in obs])
    exc_se = np.array([o["pop_se"] / (1 - 2 / d) for o in obs])
    coh = np.array([abs(complex(o["c_re"], o["c_im"])) if mqc else o["x"] for o in obs])
    coh_se = np.array([o["c_se"] if mqc else o["x_se"] for o in obs])
    if exc.min() > 0 and coh.min() > 0:
        le0, le_se, le_sl = wls(lams, np.log(exc), exc_se / exc)
        lc0, lc_se, lc_sl = wls(lams, np.log(coh), coh_se / coh)
        e0, c0 = math.exp(le0), math.exp(lc0)
        p0 = 2 / d + (1 - 2 / d) * e0
        oe = dict(pop=p0, c_re=c0, c_im=0.0) if mqc else dict(pop=p0, x=c0)
        F_se = (0.5 * math.hypot((1 - 2 / d) * e0 * le_se, 2 * c0 * lc_se) if mqc
                else math.hypot((1 - 2 / d) * e0 * le_se, c0 * lc_se / 2))
        phys = p0 <= 1 and (c0 <= p0 / 2 if mqc else c0 <= 1)
        out["exp"] = dict(obs=oe, F=fidelity_from(oe), F_se=F_se, pop_rate=-le_sl, coh_rate=-lc_sl,
                          physical=bool(phys))
    else:
        out["exp"] = None
    return out


def seam_model(R, rows):
    """Layer S, extraction: the seam's error information in passes 1 and 2, no new circuits. Per unit with seam
    crossings and per pass, the share of P0 + P1 and of the coherence (|c|, or <X..X>) that ACE keeps against
    the exact reference, against the circuit's seam crossings n: log(kept) = a + b n by least squares. e^a is
    what the first crossing leaves (scaling the crossings cannot remove it), e^b what each further crossing
    keeps (the part ZNE can remove); at least three distinct crossing counts (pass 1 alone has 1 and 2, one per seam
    class, so --order code-echo reports no fit). Per seam coupler: the share of shots whose two ends differ in Z, per pass."""
    pts = dict(pop=[], coh=[])
    for r in rows:
        if r["prep"] + r["drift"] == 0:
            continue
        for e, x, n in ((r["e1"], r["x1"], r["prep"]), (r.get("e2"), r.get("x2"), r["prep"] + r["drift"])):
            if not x or e is None:
                continue
            xp = x["exact_P0"] + x["exact_P1"]
            if xp > 0.5:
                pts["pop"].append((n, e["pop"] / xp))
            if x.get("exact_C_parity", 0) > 0.5 and "C_parity" in e:
                pts["coh"].append((n, e["C_parity"] / x["exact_C_parity"]))
            elif x.get("exact_x_parity", 0) > 0.5 and "x_parity" in e:
                pts["coh"].append((n, e["x_parity"] / x["exact_x_parity"]))
    fit = {}
    for key, p in pts.items():
        ns = np.array([q[0] for q in p], float)
        if len(set(ns.tolist())) < 3:        # two counts: a line through two class means, not a decay
            fit[key] = None
            continue
        b, a_ = np.polyfit(ns, np.log(np.clip([q[1] for q in p], 1e-3, None)), 1)
        fit[key] = dict(first=float(math.exp(a_)), per_crossing=float(math.exp(b)), points=len(p),
                        crossings=[int(ns.min()), int(ns.max())])
    brk = collections.defaultdict(lambda: ([], []))
    for r in rows:
        for i, e in enumerate((r["e1"], r.get("e2"))):
            for a_, b_, cl, v in (e or {}).get("coupler_break", []):
                if cl != "exact":
                    brk[(min(a_, b_), max(a_, b_))][i].append(v)
    couplers = [[a_, b_, R.cls(a_, b_), (round(float(np.mean(v1)), 4) if v1 else None),
                 (round(float(np.mean(v2)), 4) if v2 else None)] for (a_, b_), (v1, v2) in sorted(brk.items())]
    return dict(fit=fit, couplers=couplers)


def print_seam_model(model, lams):
    print("\nLayer S, extraction (the passes so far against the exact reference, units with seam crossings n):")
    for key, name in (("pop", "P0 + P1"), ("coh", "coherence")):
        f = model["fit"].get(key)
        if f is None:
            print(f"  {name:<9}  not identifiable (fewer than three crossing counts, or no exact reference)")
            continue
        print(f"  {name:<9}  kept after the first crossing {f['first']:.3f}, per further crossing {f['per_crossing']:.4f}"
              f"  ({f['points']} unit-passes, n {f['crossings'][0]}..{f['crossings'][1]})")
    f = model["fit"].get("coh")
    if f is not None and lams:
        print(f"  ZNE ceiling under this model: lambda -> 0 restores the per-crossing part only; the first-crossing "
              f"share {f['first']:.3f} stays" + ("  [per-crossing loss < 1 %: little for ZNE to extrapolate]"
                                                 if f["per_crossing"] > 0.99 else ""))
    br = sorted((c for c in model["couplers"] if c[3] is not None), key=lambda c: -(c[3] or 0))
    if br:
        print("  seam couplers, Z break rate pass 1 / pass 2 (highest first): "
              + ", ".join(f"{a_}-{b_} {v1:.2f}/" + ("  - " if v2 is None else f"{v2:.2f}")
                          for a_, b_, _, v1, v2 in br[:8]))
    if lams:
        print(f"  ZNE (--zne-k) runs the units with seam crossings again with every crossing scaled by lambda {lams[1:]}")


SEAM_CLASSES = ("bulk", "replica", "cut")


def unit_hosts(R, u):
    """The ACE simulators that hold every qubit of a unit (a seam qubit is held by the simulator of each replica)."""
    return sorted(set.intersection(*(set(R.sids[q]) for q in u["qubits"])))


def seam_class(R, u):
    """bulk: one simulator holds the unit and none of its qubits is a seam qubit. replica: one simulator holds the
    unit, its seam qubits through their replica there; their seam error is a disagreement between replicas, which
    the replica repetition code corrects. cut: no simulator holds the whole unit; its GHZ needs coherence between
    two simulators, and ACE has none to give (c = 0, F_GHZ <= 1/2)."""
    if not unit_hosts(R, u):
        return "cut"
    return "replica" if any(q in R.seam for q in u["qubits"]) else "bulk"


def code_schedule(R, u):
    """The unit's tree in the order the replica repetition code needs: from every prepared qubit, its CXs into qubits
    of one simulator first, its CXs into seam qubits after. A CX between two seam qubits is matched in every simulator,
    which clears ACE's authority record, so a seam qubit that meets another seam qubit before its partner in the host
    simulator leaves the decoder no replica to trust (measured: nighthawk61r2 [48, 40, 56] at F 0.23-0.45 under the
    code in the preparation's order, 1.000 in this one). The same state: CXs from one control commute, and every
    child still follows its parent."""
    parent = {c: p for p, c in u["edges"]}
    kids = collections.defaultdict(list)
    for p, c in u["edges"]:
        kids[p].append(c)
    order, ready = [u["root"]], list(kids[u["root"]])
    while ready:
        ready.sort(key=lambda c: (len(R.sids[c]) > 1, order.index(parent[c])))
        c = ready.pop(0)
        order.append(c)
        ready += kids[c]
    return unit_from_tree(R, u["root"], parent, order)


def syndrome(R, e, x, key):
    """One pass's seam syndrome of a unit, read from its records: the shares of P0 + P1 and of the coherence ACE keeps
    against the exact reference, the fidelity it loses (and in standard errors), which branch loses population
    (P1 - P0 > 0: the 0..0 branch), and the qubit whose Z value leaves the unit's majority most often."""
    out = dict(branch=round(e["P1"] - e["P0"], 4))
    if x:
        xp = x.get("exact_P0", 0.0) + x.get("exact_P1", 0.0)
        if xp > 0.5:
            out["pop_kept"] = round(e["pop"] / xp, 4)
        if "C_parity" in e and x.get("exact_C_parity", 0.0) > 0.5:
            out["coh_kept"] = round(e["C_parity"] / x["exact_C_parity"], 4)
        elif "x_parity" in e and x.get("exact_x_parity", 0.0) > 0.5:
            out["coh_kept"] = round(e["x_parity"] / x["exact_x_parity"], 4)
        xF = x.get("exact_" + key)
        if xF is not None and e.get(key) is not None:
            out["loss"] = round(xF - e[key], 4)
            out["loss_sigma"] = round((xF - e[key]) / max(e.get(key + "_se", 0.0), 1e-3), 2)
    sb = e.get("site_break") or {}
    if sb:
        q, v = max(((int(q), v) for q, v in sb.items()), key=lambda t: (t[1], -t[0]))
        out.update(break_site=q, break_rate=round(v, 4), break_on_seam=q in R.seam)
    return out


def seam_plan(R, rows, key, min_loss=0.02, min_sigma=3.0):
    """Layer S's extraction and plan, per row (unit): seam class, the syndrome of passes 1 and 2, and the plan: code
    (a replica unit that lost fidelity in pass 1 or 2: rerun under the replica repetition code), clean (a replica
    unit with no measurable loss: nothing to correct), control (a cut unit: rerun without a code), None (bulk).
    Returns a summary per class."""
    for r in rows:
        r["cls"] = seam_class(R, r["u"])
        r["syn"] = [syndrome(R, r["e1"], r["x1"], key)] + ([syndrome(R, r["e2"], r["x2"], key)] if r.get("e2") else [])
        lost = any(s.get("loss", 0.0) > min_loss and s.get("loss_sigma", 0.0) > min_sigma for s in r["syn"])
        r["plan"] = (("code" if lost else "clean") if r["cls"] == "replica" else
                     "control" if r["cls"] == "cut" else None)
    summary = {}
    for cl in SEAM_CLASSES:
        rs = [r for r in rows if r["cls"] == cl]
        if not rs:
            continue
        s = dict(units=len(rs), plans=dict(collections.Counter(str(r["plan"]) for r in rs)))
        for i in range(len(rs[0]["syn"])):
            for f in ("pop_kept", "coh_kept", "branch"):
                v = [r["syn"][i][f] for r in rs if f in r["syn"][i]]
                if v:
                    s[f"{f}{i + 1}"] = round(float(np.mean(v)), 4)
            v = [r["syn"][i]["break_on_seam"] for r in rs if r["syn"][i].get("break_rate", 0) > 0.02]
            if v:
                s[f"break_on_seam{i + 1}"] = [int(sum(v)), len(v)]
        summary[cl] = s
    return summary


def print_seam_plan(summary, min_loss=0.02, passes="pass 1 / pass 2"):
    print(f"\nLayer S, syndrome by seam class (against the exact reference; {passes}):")
    print(f"  {'class':<9}{'units':>6}  {'P0+P1 kept':>13}  {'coherence kept':>15}  {'P1 - P0':>14}  "
          f"{'breaks on a seam qubit':>23}  plan")
    for cl in SEAM_CLASSES:
        s = summary.get(cl)
        if not s:
            continue

        def pair(f, nd=3):
            return f"{fmt(s.get(f + '1'), nd)} / {fmt(s.get(f + '2'), nd)}"

        def brk(i):
            v = s.get(f"break_on_seam{i}")
            return f"{v[0]}/{v[1]}" if v else "-"
        plans = ", ".join(f"{k} {v}" for k, v in sorted(s["plans"].items()) if k != "None") or "-"
        print(f"  {cl:<9}{s['units']:>6}  {pair('pop_kept'):>13}  {pair('coh_kept'):>15}  {pair('branch'):>14}  "
              f"{brk(1) + ' / ' + brk(2):>23}  {plans}")
    print(f"  (breaks: units whose most-breaking qubit, at > 0.02, is a seam qubit. replica: one simulator holds the "
          f"unit; code = it lost > {min_loss} and 3 sigma of F, rerun under the replica repetition code. cut: no "
          f"simulator holds it, ACE keeps no coherence between simulators, F_GHZ <= 1/2: rerun without a code)")


def cmd_stack(a):
    """Pass 1 Layer 1a, pass 2 Layer 2, Layer S (the seam's syndrome from passes 1 and 2, then the replica repetition
    code on the units it can correct; ZNE for comparison with --zne-k), Layer V (the third GHZ measurement: pass 2's
    circuit under the correction, the uncorrectable units as a control), compared unit by unit."""
    if a.ace_boundary_rep:
        raise SystemExit("ghz stack: passes 1 and 2 measure the seam uncorrected and Layer S applies the boundary "
                         "repetition code itself: drop --ace-boundary-rep")
    if a.engine == "loom":
        if a.mode != "tiles":
            raise SystemExit("ghz stack --engine loom: tiles only (a chip-wide GHZ over two chips is always split)")
        if a.zne_k:
            raise SystemExit("ghz stack --engine loom: no ZNE on the loom")
        a.loom_chips = a.loom_chips or list(LOOM_DEFAULT["chips"])
        if a.loom_link_rows is None and list(a.loom_chips) == list(LOOM_DEFAULT["chips"]):
            a.loom_link_rows = list(LOOM_DEFAULT["link_rows"])
        a.code_schedule = "off"                          # the loom reads, it does not reorder: no ACE schedule
        R = loom_register(a.repo, a.cloverfield, a.loom_chips, a.loom_link_rows)
        runner = run_loom
    else:
        R = register(a.repo, a.register, False)
        Rc = register(a.repo, a.register, True)
        if Rc.idx != R.idx or Rc.size != R.size or Rc.sids != R.sids:
            raise SystemExit(f"ghz stack: ACE places {R.name} differently with the boundary repetition code, so "
                             f"Layer V would not rerun the same circuits")
        runner = run_tasks
    if parse_layers(a.layers):
        raise SystemExit("ghz stack: --layers must be none: the passes compare the plain preparation")
    if parse_layer2(a.layer2) == "none":
        a.layer2 = "dd"
    l2, l2_0 = layer2_spec(a), dict(mode="none", blocks=0)
    ks = sorted({int(x) for x in str(a.zne_k or "").split(",") if x.strip()})
    if ks and ks[0] < 1:
        raise SystemExit("ghz stack: --zne-k needs K >= 1")
    lams = [1] + [AmpGates.scale(a.zne_amp, K) for K in ks] if ks else []
    if a.mode == "tiles":
        k = int(a.unit)
        groups = tile_groups(a, R, k, True)
    else:
        names = [t.strip() for t in a.trees.split(",") if t.strip()]
        for t in names:
            if t not in TREES:
                raise SystemExit(f"ghz: unknown tree {t!r}; known: {', '.join(TREES)}")
        groups = [(t, [TREES[t](R, a.root)]) for t in names]
        k = R.n
    settings = settings_for(k, a.estimator, a.mqc_max)
    a.stab = resolve_stabilizer(a, a.mode, settings, l2["mode"])
    if a.mode == "chip" and not a.stab and R.n > 30 and not a.force:
        raise SystemExit(f"ghz stack: {R.n} qubits with dense patches do not fit; use a register of <= 30 qubits")
    L, T = max(1, l2["blocks"]), l2["t_idle"]
    what = f"GHZ stack, {k}-qubit chains" if a.mode == "tiles" else f"GHZ stack over all {R.n} qubits"
    print(header(a, R, what, l2))
    if a.order == "code-echo":
        if lams:
            raise SystemExit("ghz stack: --zne-k extrapolates pass 2's circuit; with --order code-echo there is none")
        code = ("the loom's authority (each unit read on the chip holding copies of all its qubits)"
                if a.engine == "loom" else
                f"the replica repetition code (authority {a.code_authority}, schedule {a.code_schedule})")
        print(f"passes (order code-echo): 1 = Layer 1a (no drift); S = the seam's syndrome from pass 1, then {code} "
              f"where it corrects; C = pass 1's circuit under the correction; E = Layer 2 {l2['mode']} on every unit, "
              f"the code kept where it ran")
    else:
        print(f"passes: 1 = Layer 1a (no drift); 2 = Layer 2 {l2['mode']}; S = the seam's syndrome from 1 and 2, then "
              f"the replica repetition code where it corrects (authority {a.code_authority}, schedule "
              f"{a.code_schedule})" + (f", and ZNE ({a.zne_amp}, lambda {lams})" if lams else "")
              + "; V = the third measurement: pass 2's circuit under the correction")
    print(f"{len(groups)} circuit groups, {sum(len(u) for _, u in groups)} units; readout settings: {' '.join(settings)}")
    d1, x1, bad, t1 = runner(a, R, groups, settings, "pass 1", l2_0)
    stop_on_bad(a, bad)
    if a.order == "code-echo":
        return stack_code_echo(a, R, groups, settings, k, l2, l2_0, d1, x1, t1, runner)
    d2, x2, bad, t2 = runner(a, R, groups, settings, "pass 2", l2)
    stop_on_bad(a, bad)
    rows = []
    for g, units in groups:
        for j, u in enumerate(units):
            prep, drift = seam_crossings(R, u, l2)
            rows.append(dict(g=g, j=j, u=u, prep=prep, drift=drift,
                             e1=score_unit(R, u, d1[g], k, 1, None),
                             x1=exact_fields(R, u, x1[g] if x1 else None, k, l2_0),
                             e2=score_unit(R, u, d2[g], k, L, T),
                             x2=exact_fields(R, u, x2[g] if x2 else None, k, l2)))
    key = headline(rows[0]["e1"])[0]
    model = seam_model(R, rows)
    print_seam_model(model, lams)
    plan = seam_plan(R, rows, key)
    print_seam_plan(plan)
    if a.no_exact:
        print("  (--no-exact: no loss can be measured, so no unit is planned for the code)")
    tS = tV = 0.0
    sg = [(g, units) for g, units in groups if any(r["prep"] + r["drift"] for r in rows if r["g"] == g)]
    if lams and sg:
        dS = {}
        for K in ks:
            amp = dict(kind=a.zne_amp, K=K, phi=float(a.zne_phi))
            dS[K], _, _, t_ = run_tasks(a, R, sg, settings, f"ZNE lambda {AmpGates.scale(a.zne_amp, K)}", l2,
                                        amp=amp, with_exact=False)
            tS += t_
        names = {g for g, _ in sg}
        for r in rows:
            if r["g"] in names and r["prep"] + r["drift"]:
                r["eS"] = {K: score_unit(R, r["u"], dS[K][r["g"]], k, L, T, maps=False) for K in ks}
                r["obs"] = [lin_obs(r["e2"])] + [lin_obs(r["eS"][K]) for K in ks]
                r["zne"] = zne_extrapolate(lams, r["obs"], k)
    by_g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["plan"] in ("code", "control"):
            by_g[r["plan"]][r["g"]].append(r)
    for plan_, brep in (("code", True), ("control", False)):
        for rs in by_g[plan_].values():
            for r in rs:
                r["uV"] = code_schedule(R, r["u"]) if plan_ == "code" and a.code_schedule == "on" else r["u"]
        vg = [(f"{g}/{plan_}", [r["uV"] for r in rs]) for g, rs in by_g[plan_].items()]
        if not vg:
            continue
        dV, _, _, t_ = runner(a, R, vg, settings, f"Layer V, {plan_}", l2, with_exact=False, rep_offset=1000,
                                 brep=brep)
        tV += t_
        for g, rs in by_g[plan_].items():
            for r in rs:
                r["eV"] = score_unit(R, r["uV"], dV[f"{g}/{plan_}"], k, L, T)
    report_stack(R, rows, key, lams)
    print(f"seconds: pass 1 {t1:.1f}, pass 2 {t2:.1f}" + (f", ZNE {tS:.1f}" if lams else "") + f", Layer V {tV:.1f}")
    recs = []
    for r in rows:
        info = (unit_info(R, r["u"]) if a.mode == "tiles"
                else dict(cx_classes=dict(collections.Counter(r["u"]["classes"]))))
        info = dict(info, seam_class=r["cls"], seam_plan=r["plan"])
        where = (dict(mode="tiles", unit_size=k, tiling=r["g"], unit=r["j"]) if a.mode == "tiles"
                 else dict(mode="chip", tree=r["g"]))
        for p, l2p, e, x in ((1, l2_0, r["e1"], r["x1"]), (2, l2, r["e2"], r["x2"])):
            recs.append(dict(base_record(a, R, l2p), test="ghz", stack_pass=p, stack_order="echo-code", **where,
                             **r["u"], **info, **e, **x))
        if "eV" in r:
            recs.append(dict(base_record(a, R, l2), test="ghz", stack_pass="V", stack_order="echo-code", **where,
                             **r["uV"], **info, **r["eV"],
                             **r["x2"], boundary_rep=r["plan"] == "code",
                             code_schedule=(r["plan"] == "code" and a.code_schedule == "on")))
        st = dict(base_record(a, R, l2), test="ghz_stack", stack_order="echo-code", **where, **r["u"], **info,
                  seam_crossings=dict(prep=r["prep"], drift=r["drift"]), syndrome=r["syn"],
                  pass1={q: r["e1"][q] for q in KEEP if q in r["e1"]}, exact1=r["x1"],
                  pass2={q: r["e2"][q] for q in KEEP if q in r["e2"]}, exact2=r["x2"])
        if "eV" in r:
            st["passV"] = dict({q: r["eV"][q] for q in KEEP if q in r["eV"]}, boundary_rep=r["plan"] == "code",
                               code_authority=a.code_authority,
                               code_schedule=(r["plan"] == "code" and a.code_schedule == "on"), edges=r["uV"]["edges"])
        if "zne" in r:
            z = r["zne"]
            st.update(layerS=dict(amp=a.zne_amp, phi=float(a.zne_phi), K=ks, lambdas=lams, observables=r["obs"],
                                  F_at=[fidelity_from(o) for o in r["obs"]], linear=z["linear"], exp=z["exp"]),
                      F_zne_linear=z["linear"]["F"], F_zne_exp=(z["exp"]["F"] if z["exp"] else None))
        recs.append(st)
    recs.append(dict(base_record(a, R, l2), test="ghz_seam_model", mode=a.mode, plan=plan, lambdas=lams,
                     amp=(a.zne_amp if lams else None), phi=(float(a.zne_phi) if lams else None), **model))
    write_records(a, R, recs, None)


def stack_code_echo(a, R, groups, settings, k, l2, l2_0, d1, x1, t1, runner=None):
    """--order code-echo: Layer S from pass 1's syndrome, pass C (pass 1's circuit under the correction: replica
    units with the code, cut units without, as the control), pass E (Layer 2 on every unit, the code kept on the
    replica units it ran on). Bulk and clean units keep pass 1's value in C; every unit runs in E, with the exact
    reference of Layer 2."""
    runner = runner or run_tasks
    L, T = max(1, l2["blocks"]), l2["t_idle"]
    rows = []
    for g, units in groups:
        for j, u in enumerate(units):
            prep, drift = seam_crossings(R, u, l2)
            rows.append(dict(g=g, j=j, u=u, prep=prep, drift=drift, e1=score_unit(R, u, d1[g], k, 1, None),
                             x1=exact_fields(R, u, x1[g] if x1 else None, k, l2_0)))
    key = headline(rows[0]["e1"])[0]
    model = seam_model(R, rows)
    print_seam_model(model, [])
    plan = seam_plan(R, rows, key)
    print_seam_plan(plan, passes="pass 1")
    if a.no_exact:
        print("  (--no-exact: no loss can be measured, so no unit is planned for the code)")
    for r in rows:
        r["uV"] = code_schedule(R, r["u"]) if r["plan"] == "code" and a.code_schedule == "on" else r["u"]
    by_g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by_g[r["plan"] if r["plan"] == "code" else "rest"][r["g"]].append(r)
    tC = tE = 0.0
    for plan_, brep in (("code", True), ("control", False)):
        rs_g = collections.defaultdict(list)
        for r in rows:
            if r["plan"] == plan_:
                rs_g[r["g"]].append(r)
        if not rs_g:
            continue
        dC, _, _, t_ = runner(a, R, [(f"{g}/{plan_}", [r["uV"] for r in rs]) for g, rs in rs_g.items()], settings,
                                 f"pass C, {plan_}", l2_0, with_exact=False, rep_offset=1000, brep=brep)
        tC += t_
        for g, rs in rs_g.items():
            for r in rs:
                r["eC"] = score_unit(R, r["uV"], dC[f"{g}/{plan_}"], k, 1, None)
    for part, brep in (("code", True), ("rest", False)):
        vg = [(f"{g}/{part}", [r["uV"] for r in rs]) for g, rs in by_g[part].items()]
        if not vg:
            continue
        dE, xE, bad, t_ = runner(a, R, vg, settings, f"pass E, {part}", l2, brep=brep)
        stop_on_bad(a, bad)
        tE += t_
        for g, rs in by_g[part].items():
            for j, r in enumerate(rs):
                r["eE"] = score_unit(R, r["uV"], dE[f"{g}/{part}"], k, L, T)
                r["xE"] = exact_fields(R, r["uV"], xE[f"{g}/{part}"] if xE else None, k, l2)
    report_code_echo(rows, key)
    print(f"seconds: pass 1 {t1:.1f}, pass C {tC:.1f}, pass E {tE:.1f}")
    recs = []
    for r in rows:
        info = (unit_info(R, r["u"]) if a.mode == "tiles"
                else dict(cx_classes=dict(collections.Counter(r["u"]["classes"]))))
        info = dict(info, seam_class=r["cls"], seam_plan=r["plan"])
        where = (dict(mode="tiles", unit_size=k, tiling=r["g"], unit=r["j"]) if a.mode == "tiles"
                 else dict(mode="chip", tree=r["g"]))
        coded = r["plan"] == "code"
        cs = coded and a.code_schedule == "on"
        recs.append(dict(base_record(a, R, l2_0), test="ghz", stack_pass=1, stack_order="code-echo", **where,
                         **r["u"], **info, **r["e1"], **r["x1"]))
        if "eC" in r:
            recs.append(dict(base_record(a, R, l2_0), test="ghz", stack_pass="C", stack_order="code-echo", **where,
                             **r["uV"], **info, **r["eC"], **r["x1"], boundary_rep=coded, code_schedule=cs))
        recs.append(dict(base_record(a, R, l2), test="ghz", stack_pass="E", stack_order="code-echo", **where,
                         **r["uV"], **info, **r["eE"], **r["xE"], boundary_rep=coded, code_schedule=cs))
        st = dict(base_record(a, R, l2), test="ghz_stack", stack_order="code-echo", **where, **r["u"], **info,
                  seam_crossings=dict(prep=r["prep"], drift=r["drift"]), syndrome=r["syn"],
                  pass1={q: r["e1"][q] for q in KEEP if q in r["e1"]}, exact1=r["x1"],
                  passE=dict({q: r["eE"][q] for q in KEEP if q in r["eE"]}, boundary_rep=coded,
                             code_authority=a.code_authority, code_schedule=cs, edges=r["uV"]["edges"]),
                  exactE=r["xE"])
        if "eC" in r:
            st["passC"] = dict({q: r["eC"][q] for q in KEEP if q in r["eC"]}, boundary_rep=coded,
                               code_authority=a.code_authority, code_schedule=cs, edges=r["uV"]["edges"])
        recs.append(st)
    recs.append(dict(base_record(a, R, l2), test="ghz_seam_model", stack_order="code-echo", mode=a.mode, plan=plan,
                     lambdas=[], amp=None, phi=None, **model))
    write_records(a, R, recs, None)


def report_code_echo(rows, key):
    """--order code-echo: the three GHZ measurements (pass 1, C, E) by seam class and plan; the units C reran."""
    xk = "exact_" + key

    def mean(rs, f):
        v = [f(r) for r in rs]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else None
    by = collections.defaultdict(list)
    for r in rows:
        by[(SEAM_CLASSES.index(r["cls"]), r["prep"], str(r["plan"]))].append(r)
    print(f"\nOrder code-echo: {key} by class, mean over units (pass 1 Layer 1a | C: pass 1's circuit under the code "
          f"(code) or without (control); bulk and clean units keep pass 1 | E: Layer 2 on every unit, the code kept | "
          f"exact, E)")
    print(f"  {'seam class, seam CXs':<24}{'plan':<9}{'units':>6}{'pass 1':>9}{'C':>9}{'E':>9}{'exact':>9}")
    for c in sorted(by):
        rs = by[c]
        cl, prep, plan_ = SEAM_CLASSES[c[0]], c[1], c[2]
        cols = [mean(rs, lambda r: r["e1"].get(key)),
                mean(rs, lambda r: (r["eC"] if "eC" in r else r["e1"]).get(key)),
                mean(rs, lambda r: r["eE"].get(key)), mean(rs, lambda r: r["xE"].get(xk))]
        print(f"  {cl + ', ' + str(prep):<24}{('-' if plan_ == 'None' else plan_):<9}{len(rs):>6}"
              + "".join(f"{fmt(x):>9}" for x in cols))
    code = [r for r in rows if r["plan"] == "code"]
    ctrl = [r for r in rows if r["plan"] == "control"]
    if code:
        g1 = [r["eC"][key] - r["e1"][key] for r in code]
        g2 = [r["eE"][key] - r["eC"][key] for r in code]
        print(f"  code (replica units): C - pass 1 mean {np.mean(g1):+.3f}, range {min(g1):+.3f} .. {max(g1):+.3f}; "
              f"E - C mean {np.mean(g2):+.3f}; exact - E mean "
              f"{np.mean([r['xE'][xk] - r['eE'][key] for r in code if r['xE'].get(xk) is not None]):+.3f}")
    if ctrl:
        sp = [abs(r["eC"][key] - r["e1"][key]) for r in ctrl]
        print(f"  control (cut units, no code): |C - pass 1| mean {np.mean(sp):.3f} (what a rerun moves); E {key} "
              f"mean {np.mean([r['eE'][key] for r in ctrl]):.3f}, at most {max(r['eE'][key] for r in ctrl):.3f}")
    bulk = [r for r in rows if r["cls"] == "bulk"]
    if bulk:
        print(f"  bulk: pass 1 {mean(bulk, lambda r: r['e1'].get(key)):.3f}, E "
              f"{mean(bulk, lambda r: r['eE'].get(key)):.3f} (the echo on its own)")


def report_stack(R, rows, key, lams):
    """Layer V: the three GHZ measurements (pass 1, pass 2, V) side by side, by seam class, plan and seam CXs in the
    preparation; the units Layer V reran one by one; what the code changed against what a rerun changes."""
    xk = "exact_" + key

    def mean(rs, f):
        v = [f(r) for r in rs]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else None
    by = collections.defaultdict(list)
    for r in rows:
        by[(SEAM_CLASSES.index(r["cls"]), r["prep"], str(r["plan"]))].append(r)
    zcol = bool(lams)
    print(f"\nLayer V: {key} by class, mean over units (pass 1 Layer 1a | pass 2 Layer 2 | V: pass 2's circuit again, "
          f"under the code (code) or without (control) | exact, pass 2" + (" | ZNE linear" if zcol else "") + ")")
    print(f"  {'seam class, seam CXs':<24}{'plan':<9}{'units':>6}{'pass 1':>9}{'pass 2':>9}{'V':>9}{'exact':>9}"
          + (f"{'ZNE lin':>9}" if zcol else ""))
    for c in sorted(by):
        rs = by[c]
        cl, prep, plan_ = SEAM_CLASSES[c[0]], c[1], c[2]
        v = [r for r in rs if "eV" in r]
        cols = [mean(rs, lambda r: r["e1"].get(key)), mean(rs, lambda r: r["e2"].get(key)),
                mean(v, lambda r: r["eV"].get(key)) if v else None, mean(rs, lambda r: r["x2"].get(xk))]
        if zcol:
            z = [r for r in rs if "zne" in r]
            cols.append(mean(z, lambda r: r["zne"]["linear"]["F"]) if z else None)
        print(f"  {cl + ', ' + str(prep):<24}{('-' if plan_ == 'None' else plan_):<9}{len(rs):>6}"
              + "".join(f"{fmt(x):>9}" for x in cols))
    v = [r for r in rows if "eV" in r]
    if not v:
        print("  no unit to correct or to control: every unit is in the bulk, or no loss was measured")
        return
    print(f"\nunits Layer V reran ({key}; seam crossings prep + drift; syndrome of pass 2: the qubit that breaks most, "
          f"* = a seam qubit, and P1 - P0):")
    for r in sorted(v, key=lambda r: (r["plan"], r["prep"], r["e2"].get(key, 0))):
        s = r["syn"][1]
        site = (f"q{s['break_site']}{'*' if s.get('break_on_seam') else ' '} {s['break_rate']:.2f}"
                if "break_site" in s else "  -  ")
        print(f"  {r['plan']:<8}{r['g']:>9} #{r['j']:<3} {str(r['u']['qubits']):<14} n {r['prep']}+{r['drift']:<3} "
              f"p1 {fmt(r['e1'].get(key))}  p2 {fmt(r['e2'].get(key))}  V {fmt(r['eV'].get(key))}  "
              f"exact {fmt(r['x2'].get(xk))}   breaks {site}  P1-P0 {s['branch']:+.2f}"
              + (f"  ZNE lin {r['zne']['linear']['F']:.3f}" if "zne" in r else ""))
    code = [r for r in v if r["plan"] == "code"]
    ctrl = [r for r in v if r["plan"] == "control"]
    if ctrl:
        spread = [abs(r["eV"][key] - r["e2"][key]) for r in ctrl]
        coh = [r["eV"].get("C_parity", r["eV"].get("x_parity")) for r in ctrl]
        print(f"  control (cut units, no code): |V - pass 2| mean {np.mean(spread):.3f}, max {max(spread):.3f} (what a "
              f"rerun moves); V {key} at most {max(r['eV'][key] for r in ctrl):.3f}, coherence mean "
              f"{np.mean([c for c in coh if c is not None]):.3f} (no coherence between simulators: F <= 1/2)")
    if code:
        gain = [r["eV"][key] - r["e2"][key] for r in code]
        gap = [r["x2"][xk] - r["eV"][key] for r in code if r["x2"].get(xk) is not None]
        print(f"  code (replica units): V - pass 2 mean {np.mean(gain):+.3f}, range {min(gain):+.3f} .. {max(gain):+.3f}"
              + (f"; exact - V mean {np.mean(gap):+.3f}" if gap else "")
              + (f"; V - pass 1 mean {np.mean([r['eV'][key] - r['e1'][key] for r in code]):+.3f}"))
        if ctrl:
            print(f"  the code moves F by {np.mean(gain):+.3f} on average, a rerun by "
                  f"{np.mean([abs(r['eV'][key] - r['e2'][key]) for r in ctrl]):.3f}")
    z = [r for r in rows if "zne" in r]
    if z:
        g = [r["zne"]["linear"]["F"] - r["e2"][key] for r in z]
        print(f"  ZNE (linear) minus pass 2 on the seam units: mean {np.mean(g):+.3f}, range {min(g):+.3f} .. "
              f"{max(g):+.3f}")


# ================================================================== the loom as the seam (nighthawk_cloverfield.py)

LOOM_DEFAULT = dict(chips=("rect:1-8,2-5", "rect:1-8,6-9"), link_rows=(1, 2, 3, 4, 6, 7, 8))
_CF = {}


def cloverfield(path=None):
    """nighthawk_cloverfield.py, imported once: --cloverfield, $NIGHTHAWK_CLOVERFIELD, next to this script, or in
    ../rcs-nighthawk (the layout of thereminq-examples). It imports nighthawk_qrack, which is this one."""
    if "m" in _CF:
        return _CF["m"]
    here = Path(__file__).resolve().parent
    cands = [Path(x) for x in (path, os.environ.get("NIGHTHAWK_CLOVERFIELD")) if x]
    cands += [here / "nighthawk_cloverfield.py", here.parent / "rcs-nighthawk" / "nighthawk_cloverfield.py"]
    for c in cands:
        c = c / "nighthawk_cloverfield.py" if c.is_dir() else c
        if c.is_file():
            import importlib.util
            spec = importlib.util.spec_from_file_location("nighthawk_cloverfield", str(c))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["nighthawk_cloverfield"] = mod
            spec.loader.exec_module(mod)
            # it puts its own folder first on sys.path; workers would then import that folder's
            # nighthawk_qrack.py instead of this one: keep this folder first
            other = str(c.resolve().parent)
            if other != str(here):
                while other in sys.path:
                    sys.path.remove(other)
            if str(here) not in sys.path[:1]:
                sys.path.insert(0, str(here))
            _CF["m"], _CF["path"] = mod, str(c)
            return mod
    raise SystemExit("ghz --engine loom: nighthawk_cloverfield.py not found (give --cloverfield PATH, set "
                     "NIGHTHAWK_CLOVERFIELD, or put it next to this script or in ../rcs-nighthawk)")


class LoomRegister:
    """Two (or more) chips of the cloverfield as one register: the chips side by side on the device lattice,
    joined by the loom's links. Answers what Register answers for ACE, with chips in place of ACE simulators:
    sids[q] = the chips holding a copy of q (its own, and every chip holding a ghost of it), seam = the link
    anchors (a copy on both chips), cls of a coupler as for ACE. With the ghost loom a unit that one chip holds
    copies of is read there exactly (the loom's authority, Layer S's code); real copies on two chips are a
    product state."""

    def __init__(self, repo, cf_path, chips, link_rows):
        cf = cloverfield(cf_path)
        nq.load_register_files()
        self.repo = repo
        self.specs = [cf.parse_chip(t, repo) for t in chips]
        self.field = cf.Field(repo, self.specs, (1, len(self.specs)), "all", list(link_rows) if link_rows else None)
        f = self.field
        self.skipped = [list(x) for x in f.skipped]   # facing pairs with no CZ colour free at both ends
        if f.skipped and link_rows:
            raise SystemExit(f"ghz --engine loom: links asked for but without a free CZ colour: {f.skipped}")
        if not f.links:
            raise SystemExit("ghz --engine loom: no link between the chips")
        self.n = f.n
        self.name = "loom:" + "+".join(sp["name"] for sp in self.specs)
        self.chip_of, self.local = list(f.chip_of), list(f.local)
        self.ghosts = f.ghost_plan()                          # per chip [(partner global, anchor local, site)]
        hold = [{f.chip_of[q]} for q in range(self.n)]
        for c, plan in enumerate(self.ghosts):
            for q, _, _ in plan:
                hold[q].add(c)
        self.sids = [frozenset(h) for h in hold]
        self.seam = {q for q in range(self.n) if len(self.sids[q]) > 1}
        self.couplers = [tuple(e) for e in f.edges]
        self.links = [(u, v) for u, v, _ in f.links]
        self.adj = collections.defaultdict(set)
        for a_, b_ in self.couplers:
            self.adj[a_].add(b_)
            self.adj[b_].add(a_)
        self.cell = {q: tuple(f.cells[f.chip_of[q]][f.local[q]]) for q in range(self.n)}
        self.sub = [(r + c) % 2 for r, c in (self.cell[q] for q in range(self.n))]
        bad = [(a_, b_) for a_, b_ in self.couplers if self.sub[a_] == self.sub[b_]]
        if bad:
            raise SystemExit(f"ghz --engine loom: the field is not bipartite under r + c parity: {bad[:5]}")
        self.seam_dist = Register._bfs_dist(self, self.seam)
        self.size, self.lrc, self.lrr, self.torus = None, None, None, False
        self.widths = [f.chip_n[c] + len(self.ghosts[c]) for c in range(len(self.specs))]
        self.engine = "loom"
        self.map = dict(n=self.n, columns=COLUMNS,
                        sites=[[q, self.cell[q][0], self.cell[q][1], sorted(self.sids[q])] for q in range(self.n)],
                        couplers=[[a_, b_, self.cls(a_, b_)] for a_, b_ in self.couplers], ace=None,
                        loom=dict(chips=[sp["name"] for sp in self.specs], chip_n=list(f.chip_n),
                                  links=[list(x) for x in self.links],
                                  ghosts=[[g[0] for g in plan] for plan in self.ghosts], widths=self.widths))

    region = Register.region
    cls = Register.cls

    def describe(self):
        return (f"{self.name} ({self.n} qubits; exact chips of {', '.join(str(x) for x in self.field.chip_n)} "
                f"qubits + ghosts, widths {self.widths}; {len(self.links)} links, loom ghost"
                + (f"; {len(self.skipped)} facing pairs without a free CZ colour left unlinked" if self.skipped else "")
                + ")")

    def unit_reads(self, u, policy):
        """Where each qubit of a unit is read: (chip, local index) per qubit. real: its own copy. authority: on
        the chip holding copies of every qubit of the unit (most real copies first), real where none does."""
        f = self.field
        if policy == "authority":
            hosts = set.intersection(*(set(self.sids[q]) for q in u["qubits"]))
            if hosts:
                c = min(hosts, key=lambda c_: (-sum(1 for q in u["qubits"] if f.chip_of[q] == c_), c_))
                out = []
                for q in u["qubits"]:
                    if f.chip_of[q] == c:
                        out.append((c, f.local[q]))
                    else:
                        j = [g[0] for g in self.ghosts[c]].index(q)
                        out.append((c, f.chip_n[c] + j))
                return out
        return [(f.chip_of[q], f.local[q]) for q in u["qubits"]]


_LOOM_REGS = {}


def loom_register(repo, cf_path, chips, link_rows):
    key = (repo, cf_path, tuple(chips), tuple(link_rows or ()))
    if key not in _LOOM_REGS:
        _LOOM_REGS[key] = LoomRegister(repo, cf_path, chips, link_rows)
    return _LOOM_REGS[key]


class LoomChip:
    """One chip for the loom, in this process: the cloverfield's ChipSim on an exact simulator with the chip's
    ghosts, plus one op the loom's stream lacks, ("m", q, M): a 2 x 2 matrix on local qubit q."""

    def __init__(self, cf, spec, repo, anchors):
        self.sim = cf.ChipSim(spec, repo, dict(backend="exact", cpu=True, ace_gpu=False, ghost_anchors=anchors))
        self._r = None

    def put(self, msg):
        ops, query = msg[1], msg[2]
        e = self.sim.eng
        run = []
        for op in ops:
            if op[0] == "m":
                if run:
                    self.sim.batch(run, [])
                    run = []
                e.g1(op[1], op[2])
            else:
                run.append(op)
        if run:
            self.sim.batch(run, [])
        self._r = ("m", [self.sim.marg(q) for q in query])

    def get(self, want):
        return self._r


_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


class LoomGates:
    """Gates (the interface of Gates) through the loom: a one-qubit gate on every copy of the qubit, a CX as
    H CZ H on the target, the CZ woven by the loom (exact where a chip holds both qubits, the mean-field phase
    where it holds one)."""

    def __init__(self, loom, queues):
        self.loom, self.qs = loom, queues

    def _m(self, q, M):
        self.loom._emit1(self.qs, q, "m", np.asarray(M, dtype=complex))

    def h(self, q):
        self._m(q, _H)

    def x(self, q):
        self._m(q, [[0, 1], [1, 0]])

    def z(self, q):
        self._m(q, [[1, 0], [0, -1]])

    def s(self, q):
        self._m(q, [[1, 0], [0, 1j]])

    def sdg(self, q):
        self._m(q, [[1, 0], [0, -1j]])

    def u(self, q, th, ph, lm):
        c, s_ = math.cos(th / 2), math.sin(th / 2)
        self._m(q, [[c, -np.exp(1j * lm) * s_], [np.exp(1j * ph) * s_, np.exp(1j * (ph + lm)) * c]])

    def zrot(self, q, phi):
        self._m(q, [[1, 0], [0, np.exp(2j * phi)]])

    def cx(self, c, t):
        self.h(t)
        self.loom._layer([(c, t)], None, self.qs, None)
        self.h(t)


def _loom_task(spec):
    """One circuit on the loom (or, exact, on one QrackSimulator over the field): every unit prepared, Layer 2,
    the readout setting; shots of every unit qubit, read where spec['policy'] says (LoomRegister.unit_reads),
    as ints with bit q = field qubit q. Copies on different chips are sampled apart and paired shot by shot after
    an independent shuffle of each sample (the chips' samplers share a stream otherwise)."""
    R = loom_register(spec["repo"], spec["cf"], spec["chips"], spec["link_rows"])
    cf = cloverfield(spec["cf"])
    units = spec["units"]
    if spec["exact"]:
        from pyqrack import QrackSimulator
        sim = QrackSimulator(R.n, is_gpu=False)
        g = Gates(sim, list(range(R.n)))
    else:
        f = R.field
        chips = [LoomChip(cf, sp, spec["repo"], [anc for _, anc, _ in R.ghosts[c]]) for c, sp in enumerate(R.specs)]
        loom = cf.Loom(f, chips, "ghost")
        queues = [[] for _ in chips]
        g = LoomGates(loom, queues)
    for u in units:
        prepare(g, u, [])
    l2 = spec.get("layer2") or {}
    if l2.get("mode", "none") != "none":
        drift = Drift(R.couplers, R.n, R.sub, l2["j_zz"], l2["sigma_qs"], l2["t_idle"], l2["seed"])
        probe = Probe(drift, units, l2.get("inside_zz", "gate"), R.seam)
        eps = drift.eps(spec["realisation"])
        for _ in range(int(l2["blocks"])):
            idle_block(g, probe, l2["mode"], eps)
    for u in units:
        readout(g, u, spec["setting"])
    if spec["exact"]:
        return [int(x) for x in exact_shots(sim, R.n, spec["shots"], False)]
    loom._flush(queues, [[] for _ in chips])
    S = spec["shots"]
    out = np.zeros(S, dtype=np.uint64)
    pol = spec["policy"]
    for ui, u in enumerate(units):
        reads = R.unit_reads(u, pol.get(tuple(u["qubits"]), "real"))
        per = collections.defaultdict(list)
        for q, (c, l_) in zip(u["qubits"], reads):
            per[c].append((q, l_))
        for c, ql in per.items():
            w = np.array(chips[c].sim.eng.sim.measure_shots([l_ for _, l_ in ql], S), dtype=np.uint64)
            w = w[np.random.default_rng([spec["realisation"], ui, c, zlib_key(spec["setting"])]).permutation(S)]
            for j, (q, _) in enumerate(ql):
                out |= ((w >> np.uint64(j)) & np.uint64(1)) << np.uint64(q)
    return [int(x) for x in out]


def zlib_key(s):
    import zlib
    return zlib.crc32(s.encode())


def run_loom(a, R, groups, settings, label, l2, amp=None, with_exact=True, rep_offset=0, brep=None):
    """run_tasks for --engine loom: the same groups, settings and realisations on the loom. brep True: every
    unit is read by the loom's authority (Layer S's code); otherwise every unit from its real copies (the
    cloverfield's estimator)."""
    if amp:
        raise SystemExit("ghz --engine loom: no ZNE amplification on the loom")
    reps = max(1, int(a.repeats))
    ex_reps = reps if l2["mode"] != "none" else 1
    no_exact = a.no_exact or not with_exact
    pol_name = "authority" if brep else "real"
    base = dict(repo=a.repo, cf=a.cloverfield, chips=list(a.loom_chips), link_rows=list(a.loom_link_rows or ()),
                layer2=l2)
    calls, labels, keys = [], [], []
    for name, units in groups:
        pol = {tuple(u["qubits"]): pol_name for u in units}
        for st in settings:
            for r in range(reps):
                calls.append((_loom_task, (dict(base, units=units, setting=st, shots=a.shots, exact=False,
                                                realisation=r + rep_offset, policy=pol),)))
                labels.append(f"loom {label} {name} {st} realisation {r + 1}/{reps}")
                keys.append((name, st, r, False))
            if not no_exact:
                for r in range(ex_reps):
                    calls.append((_loom_task, (dict(base, units=units, setting=st, shots=a.exact_shots, exact=True,
                                                    realisation=r + rep_offset, policy=pol),)))
                    labels.append(f"exact {label} {name} {st} realisation {r + 1}/{ex_reps}")
                    keys.append((name, st, r, True))
    pool, workers = nq.shor_pool(a.jobs, len(calls))
    every = float(a.progress or 0)
    if every:
        print(f"# ghz plan: {len(calls)} circuits on the loom ({len(groups)} groups x {len(settings)} settings x "
              f"{reps} realisations" + (f" + exact x {ex_reps}" if not no_exact else "") + f"), {workers} "
              f"process(es) [{label}]", flush=True)
    t0 = time.time()
    try:
        res = nq.shor_pool_map(pool, calls, labels, every, what="ghz")
    finally:
        if pool is not None:
            pool.shutdown()
    data = {name: [dict() for _ in range(reps)] for name, _ in groups}
    exact = {name: [dict() for _ in range(ex_reps)] for name, _ in groups}
    unit_of = dict(groups)
    bad = 0
    pid = identity_idle(l2)
    for (name, st, r, ex), shots in zip(keys, res):
        bits = bit_matrix(shots, R.n)
        if ex:
            bad += check_exact(unit_of[name], st, bits, pid)
            exact[name][r][st] = bits
        else:
            data[name][r][st] = bits
    return data, (None if no_exact else exact), (None if no_exact else bad), time.time() - t0


# ================================================================== sweep, probe, stackmap

def chain_from_end(u):
    """The same GHZ as a chain from one end of a chain unit: H on the end, CX along the chain (the probe's chain
    variants; nighthawk_ghz.py's tiles root a chain in its middle)."""
    adj = collections.defaultdict(set)
    for p_, c_ in u["edges"]:
        adj[p_].add(c_)
        adj[c_].add(p_)
    end = min(q for q in u["qubits"] if len(adj[q]) == 1)
    order, prev = [end], None
    while len(order) < len(u["qubits"]):
        nxt = [q for q in adj[order[-1]] if q != prev][0]
        prev = order[-1]
        order.append(nxt)
    edges = [[x, y] for x, y in zip(order, order[1:])]
    return dict(u, root=end, edges=edges)


def cmd_sweep(a):
    """tiles once per Walsh-echo window count L (0: no Layer 2), optionally again under the boundary code, then
    the means per (setup, L, seam class); or (--summarise) the means of records already written."""
    import copy
    if not a.summarise:
        if not a.out:
            a.out = f"ghz_sweep_{a.register}.jsonl"
        for L, code in ([(int(x), False) for x in a.windows.split(",") if x.strip()]
                        + [(int(x), True) for x in str(a.code_at or "").split(",") if x.strip()]):
            b = copy.copy(a)
            b.layer2, b.blocks, b.ace_boundary_rep = ("none" if L == 0 else "dd"), max(L, 1), code
            print(f"\n# sweep: L = {L}" + (", boundary code" if code else ""), flush=True)
            cmd_tiles(b)
    sweep_summary(a.summarise or [a.out], a.register, a.repo, a.summary)


def sweep_summary(paths, reg, repo, out=None):
    R = register(repo, reg, False)
    acc = collections.defaultdict(list)
    for path in paths:
        for r in read_records(path):
            if r.get("test") != "ghz" or r.get("register") != reg or r.get("mode") != "tiles" or "stack_pass" in r:
                continue
            l2 = r.get("layer2") or {}
            L = int(l2.get("blocks", 0)) if l2.get("mode") == "dd" else 0
            setup = ("code" if r.get("boundary_rep") else "no code") + (
                ", zz stab" if l2.get("inside_zz") == "stabilizer" else "")
            acc[(setup, L, seam_class(R, dict(qubits=r["qubits"])))].append(r)
    if not acc:
        raise SystemExit(f"ghz sweep: no tiles records of {reg} in {', '.join(paths)}")
    rows = []
    for (setup, L, cl), rs in sorted(acc.items(), key=lambda t: (t[0][0], t[0][1], SEAM_CLASSES.index(t[0][2]))):
        def m(f):
            return round(float(np.mean([f(r) for r in rs])), 4)
        rows.append(dict(setup=setup, windows=L, seam_class=cl, units=len(rs), F_GHZ=m(lambda r: r["F_GHZ"]),
                         pop=m(lambda r: r["pop"]), C_parity=m(lambda r: r.get("C_parity", float("nan"))),
                         abs_branch=m(lambda r: abs(r["P1"] - r["P0"]))))
    print(f"\nsweep, {reg}: mean over units per window count L (4 Walsh segments per window; C = 2|c|)")
    print(f"  {'setup':<18}{'L':>3}  {'class':<9}{'units':>6}{'F_GHZ':>8}{'P0+P1':>8}{'C':>8}{'|P1-P0|':>9}")
    for r in rows:
        print(f"  {r['setup']:<18}{r['windows']:>3}  {r['seam_class']:<9}{r['units']:>6}{r['F_GHZ']:>8.3f}"
              f"{r['pop']:>8.3f}{r['C_parity']:>8.3f}{r['abs_branch']:>9.3f}")
    if out:
        with open(out, "w") as fh:
            json.dump(dict(register=reg, records=list(paths), rows=rows), fh, indent=1)
        print(f"  summary: {out}")
    return rows


PROBE_VARIANTS = dict(none=("tree", None), gate=("tree", "gate"), tseam=("tree", "target-seam"),
                      tbulk=("tree", "target-bulk"), stab=("tree", "stabilizer"), **{
                          "chain-none": ("chain", None), "chain-gate": ("chain", "gate"),
                          "chain-stab": ("chain", "stabilizer")})


def cmd_probe(a):
    """One seam class's k-qubit chains through Layer 2 with the drift's ZZ inside a unit put through ACE in
    several ways (PROBE_VARIANTS: preparation tree or chain from one end; inside ZZ none (no Layer 2), as gates,
    with the CX's target on the seam or the bulk qubit, or dropped by the GHZ stabiliser), against the exact
    reference for --exact-variants."""
    if parse_layer2(a.layer2) == "none":
        a.layer2 = "dd"
    R = register(a.repo, a.register, False)
    k = int(a.unit)
    groups = [(g, [u for u in units if seam_class(R, u) == a.seam_class]) for g, units in tile_groups(a, R, k, True)]
    groups = [(g, us) for g, us in groups if us]
    if not groups:
        raise SystemExit(f"ghz probe: no {k}-qubit chain of class {a.seam_class} on {R.name}")
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    exact_v = {v.strip() for v in a.exact_variants.split(",") if v.strip()}
    for v in variants + sorted(exact_v):
        if v not in PROBE_VARIANTS:
            raise SystemExit(f"ghz probe: unknown variant {v!r}; known: {', '.join(PROBE_VARIANTS)}")
    settings = settings_for(k, a.estimator, a.mqc_max)
    a.stab = False
    l2 = layer2_spec(a)
    l2.pop("inside_zz", None)
    print(header(a, R, f"probe: {k}-qubit {a.seam_class} chains", l2))
    print(f"{sum(len(u) for _, u in groups)} units in {len(groups)} circuit groups; variants {', '.join(variants)}"
          f" (exact reference: {', '.join(v for v in variants if v in exact_v) or 'none'})")
    rows, means = [], []
    for v in variants:
        prep, rule = PROBE_VARIANTS[v]
        vg = [(g, [chain_from_end(u) if prep == "chain" else u for u in us]) for g, us in groups]
        lv = dict(mode="none", blocks=0) if rule is None else dict(l2, inside_zz=rule)
        data, exact, bad, _ = run_tasks(a, R, vg, settings, f"probe {v}", lv, with_exact=v in exact_v, brep=False)
        stop_on_bad(a, bad)
        L, T = max(1, lv["blocks"]), (lv.get("t_idle") if lv["mode"] != "none" else None)
        for g, us in vg:
            for j, u in enumerate(us):
                for ex, per in ((False, data[g]), (True, exact[g] if exact else None)):
                    if per is None:
                        continue
                    e = score_unit(R, u, per, k, L, T, maps=False)
                    rows.append(dict(base_record(a, R, lv), test="ghz_probe", variant=v, exact=ex, seam_class=a.seam_class,
                                     mode="tiles", unit_size=k, tiling=g, unit=j, **u,
                                     **{q: e[q] for q in KEEP if q in e}))
    key = "F_GHZ"
    print(f"\nprobe, {R.name}, {a.seam_class} units: mean (per unit min .. max)")
    print(f"  {'variant':<12}{'ref':<7}{'units':>6}{key:>8}{'P0+P1':>8}{'C':>8}{'F0':>8}   per unit")
    for v in variants:
        for ex in (False, True):
            rs = [r for r in rows if r["variant"] == v and r["exact"] == ex]
            if not rs:
                continue
            f = [r[key] for r in rs]
            mm = {q: round(float(np.mean([r.get(q, np.nan) for r in rs])), 4) for q in (key, "pop", "C_parity", "F0")}
            means.append(dict(variant=v, exact=ex, units=len(rs), **mm))
            print(f"  {v:<12}{'exact' if ex else 'ACE':<7}{len(rs):>6}{mm[key]:>8.3f}{mm['pop']:>8.3f}"
                  f"{mm['C_parity']:>8.3f}{mm['F0']:>8.3f}   {min(f):.2f} .. {max(f):.2f}")
    write_records(a, R, rows, None)


def cmd_stackmap(a):
    """The three GHZ measurements of a stack run as heatmaps on the chip layout, side by side: per site the
    lowest F_GHZ among the chains through it; a unit a pass did not rerun keeps the previous pass's value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    recs = read_records(a.inp)
    st = [r for r in recs if r.get("test") == "ghz_stack" and r.get("mode") == "tiles"
          and (not a.register or r["register"] == a.register)]
    if not st:
        raise SystemExit(f"ghz stackmap: no tiles ghz_stack record{' for ' + a.register if a.register else ''} in {a.inp}")
    when = max(r["when"] for r in st)                     # the latest stack run in the file
    st = [r for r in st if r["when"] == when]
    reg = st[0]["register"]
    order = st[0].get("stack_order", "echo-code")
    mp = [r for r in recs if r.get("test") == "ghz_map" and r["register"] == reg and r.get("when") == when][-1]
    ghz = [r for r in recs if r.get("test") == "ghz" and r.get("when") == when and "stack_pass" in r]
    P = (1, 2, "V") if order == "echo-code" else (1, "C", "E")
    back = {2: 1, "V": 2, "C": 1, "E": None}
    by = {p: {(r["tiling"], r["unit"]): r for r in ghz if r["stack_pass"] == p} for p in P}
    layers = []
    for p in P:
        acc = collections.defaultdict(list)
        for k_ in by[1]:
            r, q_ = by[p].get(k_), p
            while r is None:
                q_ = back[q_]
                r = by[q_][k_]
            for q in r["qubits"]:
                acc[q].append(min(1.05, max(0.0, r["F_GHZ"])))
        layers.append({q: float(np.min(v)) for q, v in acc.items()})
    fields = ("pass1", "pass2", "passV") if order == "echo-code" else ("pass1", "passC", "passE")
    counts = collections.Counter(r["seam_class"] for r in st)

    def cls_mean(c, i):
        v = [(r.get(fields[i]) or (r["pass1"] if order == "code-echo" and i == 1 else None)) for r in st
             if r["seam_class"] == c]
        v = [d["F_GHZ"] for d in v if d]
        return float(np.mean(v)) if v else None
    sites = {s_[0]: (s_[1], s_[2]) for s_ in mp["sites"]}
    seam = set(mp.get("seam", []))
    loom = mp.get("loom")
    cmap = LinearSegmentedColormap.from_list("div", [(0.0, "#a32c2b"), (0.25, "#e98584"), (0.5, "#e4e2dc"),
                                                     (0.75, "#6da7ec"), (1.0, "#184f95")])
    norm = Normalize(0.0, 1.0)
    ink, ink2, muted, surf = "#1d1d1b", "#4a4944", "#6b6a64", "#fcfcfb"
    names = (["Pass 1", "Pass 2", "V"] if order == "echo-code" else ["Pass 1", "C", "E"])
    code_name = "Layer S: the loom's authority read" if loom else "Layer S repetition code"
    subs = (["Layer 1a, no drift", "Layer 2, anti-OTOC Walsh echo", "echo + " + code_name]
            if order == "echo-code" else
            ["Layer 1a, no drift", code_name, "code + Layer 2, anti-OTOC Walsh echo"])
    l2 = next((r["layer2"] for r in ghz if r["stack_pass"] != 1), {})
    rows_ = sorted({rc[0] for q, rc in sites.items() if q in layers[0]})
    cols_ = sorted({rc[1] for q, rc in sites.items() if q in layers[0]})
    fig = plt.figure(figsize=(17, 8.2), dpi=200, facecolor=surf)
    W, X0, GAP = 0.265, 0.035, 0.035
    for li, vals in enumerate(layers):
        ax = fig.add_axes([X0 + li * (W + GAP), 0.20, W, 0.62], facecolor=surf)
        for q, F in vals.items():
            r_, c_ = sites[q]
            ax.add_patch(Rectangle((c_ - 0.5, r_ - 0.5), 1, 1, facecolor=cmap(norm(F)), edgecolor=surf, lw=2.0))
            dark = F > 0.82 or F < 0.12
            ax.text(c_, r_ + 0.06, f"{F:.2f}", ha="center", va="center", fontsize=9.5, fontweight="bold",
                    color="#ffffff" if dark else ink)
            ax.text(c_ - 0.40, r_ - 0.32, str(q), ha="left", va="center", fontsize=6.5,
                    color="#ffffff" if dark else muted)
            if q in seam:
                ax.add_patch(Rectangle((c_ - 0.44, r_ - 0.44), 0.88, 0.88, facecolor="none", edgecolor=ink, lw=2.0))
        for u_, v_ in (loom or {}).get("links", []):           # the loom's links: a bar across the chip boundary
            (ru, cu), (rv, cv) = sites[u_], sites[v_]
            if ru == rv:
                ax.plot([(cu + cv) / 2] * 2, [ru - 0.42, ru + 0.42], color=ink, lw=3.2, solid_capstyle="butt")
            else:
                ax.plot([cu - 0.42, cu + 0.42], [(ru + rv) / 2] * 2, color=ink, lw=3.2, solid_capstyle="butt")
        ax.set_xlim(cols_[0] - 0.55, cols_[-1] + 0.55)
        ax.set_ylim(rows_[-1] + 0.55, rows_[0] - 0.55)
        ax.set_aspect("equal")
        ax.axis("off")
        x = X0 + li * (W + GAP)
        fig.text(x, 0.815, names[li], fontsize=15, fontweight="bold", color=ink, ha="left")
        fig.text(x + (0.048 if names[li].startswith("Pass") else 0.020), 0.816, subs[li], fontsize=11, color=ink2,
                 ha="left")
        m = [cls_mean(c, li) for c in SEAM_CLASSES]
        fig.text(x, 0.195, "class means   " + "   ".join(
            f"{lab} {v:.2f}" if v is not None else f"{lab} {'none' if not counts.get(c_, 0) else 'not rerun'}"
            for lab, v, c_ in zip(("off seam", "replica", "cut"), m, SEAM_CLASSES)), fontsize=10.5, color=ink,
            ha="left",
            family="monospace")
    cax = fig.add_axes([0.945, 0.26, 0.010, 0.50])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0", "0.25", "½", "0.75", "1"])
    cb.ax.tick_params(labelsize=9.5, colors=ink2)
    cb.outline.set_visible(False)
    fig.text(0.935, 0.785, "F_GHZ", fontsize=11, fontweight="bold", color=ink)
    k = st[0].get("unit_size", 3)
    prot = l2.get("inside_zz") == "stabilizer"
    reg_label = (f"loom, {len(loom['chips'])} exact chips of {' + '.join(str(x) for x in loom['chip_n'])} qubits"
                 if loom else reg)
    fig.text(0.02, 0.955, f"{reg_label}: GHZ fidelity per measurement layer"
             + ("  ·  code first, echo second" if order == "code-echo" else "  ·  echo first, code second")
             + ("  ·  with seam protection" if prot else "  ·  without seam protection"), fontsize=19,
             fontweight="bold", color=ink, ha="left")
    import textwrap
    lines = [f"Each cell is the lowest F_GHZ among the {k}-qubit GHZ chains through that qubit (small number: qubit "
             f"index). " + ("Ringed cells are link anchors (a ghost copy on the other chip); bars are the loom's links. "
                            "Below ½: no coherence between chips." if loom else
                            "Ringed cells are ACE seam qubits. Below ½: no coherence between simulators.")]
    if loom:
        lines += textwrap.wrap(f"The seam is nighthawk_cloverfield.py's ghost loom between chips "
                               f"{' and '.join(loom['chips'])}: the link CZ runs exactly on both chips, a neighbour's CZ "
                               f"reaches a ghost as a mean-field phase. Layer S's code on the loom: a unit is read on the "
                               f"chip that holds copies of all its qubits; without it, every qubit is read on its own "
                               f"chip.", 205)
    if prot:
        lines.append("Seam protection: the drift's ZZ inside a GHZ unit is not put through the "
                     + ("loom" if loom else "ACE seam") + " (a global phase on the GHZ, Corollary 2.1; --inside-zz "
                     "stabilizer).")
    for i, line in enumerate(lines):
        fig.text(0.02, 0.925 - 0.023 * i, line, fontsize=10.5, color=ink2, ha="left")
    host = "chip" if loom else "simulator"
    fig.text(0.02, 0.10, f"Class means over {k}-qubit chains: off seam {counts.get('bulk', 0)} units, replica "
             f"{counts.get('replica', 0)} (one {host} holds every qubit, the seam's through a "
             f"{'ghost' if loom else 'replica'}), cut {counts.get('cut', 0)} (no {host} holds them all). The code runs "
             f"on the replica units Layer S plans it for; cut units are the control.", fontsize=9.5, color=ink2,
             ha="left")
    zz = ", inside ZZ by stabiliser" if l2.get("inside_zz") == "stabilizer" else ""
    fig.text(0.02, 0.065, f"{st[0]['shots']} shots × {st[0]['repeats']} realisations per setting · Layer 2 "
             f"{l2.get('mode', 'none')}, L = {l2.get('blocks')}, T = {l2.get('t_idle')}, J = {l2.get('j_zz')} × "
             f"U(0.75, 1.25), σ = {l2.get('sigma_qs')}{zz} · "
             + ("V reruns the seam units on fresh draws; others keep pass 2" if order == "echo-code" else
                "C reruns the seam units on fresh backends (others keep pass 1); E runs every unit"),
             fontsize=8.5, color=muted, ha="left")
    fig.text(0.02, 0.042, f"{os.path.basename(a.inp)} · run {when} · {st[0].get('qrack', '')} · nighthawk_ghz.py stack --order {order}"
             + (" --engine loom" if loom else ""),
             fontsize=8.5, color=muted, ha="left")
    png = a.png or f"ghz_stack_{reg}_{order}.png"
    fig.savefig(png, facecolor=surf)
    plt.close(fig)
    print(f"stackmap: {reg}, order {order}, run {when}: {png}")
    for c in SEAM_CLASSES:
        print(f"  {c:<8} {counts.get(c, 0):>3} units  " + "  ".join(fmt(cls_mean(c, i)) for i in range(3)))


# ================================================================== plot

# one blue ramp, light -> dark, for magnitude (the reference palette's sequential blue, steps 100-700)
LOSS_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8984"
CLASS_COL = {"exact": "#b9b8b2", "replica": "#eb6834", "cross": "#8f4fd1"}    # inside / through a seam / no seam


def read_records(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


def _l2name(r):
    l2 = r.get("layer2") or {}
    return l2.get("mode", "none") if isinstance(l2, dict) else str(l2)


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    recs = read_records(a.inp)
    maps = [r for r in recs if r.get("test") == "ghz_map" and (not a.register or r["register"] == a.register)]
    if not maps:
        raise SystemExit(f"ghz plot: no ghz_map record{' for ' + a.register if a.register else ''} in {a.inp}")
    mp = maps[-1]
    reg = mp["register"]
    ghz = [r for r in recs if r.get("test") == "ghz" and r["register"] == reg
           and (not a.layers or r.get("layers") == a.layers) and (not a.layer2 or _l2name(r) == parse_layer2(a.layer2))]
    if not ghz:
        raise SystemExit(f"ghz plot: no ghz records for {reg}")
    tiles = [r for r in ghz if r["mode"] == "tiles"]
    chips = {}
    for r in ghz:
        if r["mode"] == "chip":
            chips[r["tree"]] = r                               # the latest run per tree
    sites = {s[0]: s for s in mp["sites"]}
    seam = set(mp.get("seam", []))
    xy = {q: (s[2], -s[1]) for q, s in sites.items()}
    cmap = LinearSegmentedColormap.from_list("loss", LOSS_RAMP)

    panels = []                                                # (title, {qubit: loss}, colour-bar label, tree edges)
    if tiles:
        key = headline(tiles[0])[0]
        per_q = collections.defaultdict(list)
        for r in tiles:
            for q in r["qubits"]:
                per_q[q].append(r[key])
        panels.append((f"{tiles[0]['unit_size']}-qubit GHZ chains, Layer 2 {_l2name(tiles[0])}: loss per site\n"
                       f"1 - mean {key} of the units holding it", {q: 1.0 - float(np.mean(v)) for q, v in per_q.items()},
                       f"1 - {key}", None))
    for t, r in sorted(chips.items()):
        k_, F = headline(r)
        panels.append((f"{r['n']}-qubit GHZ, {t} tree (thick: its CXs), {k_} {F:.3f}"
                       f"\nshare of Z shots where a site differs from the majority",
                       {int(q): v for q, v in r.get("site_break", {}).items()}, "break rate",
                       {tuple(sorted(e)) for e in r["edges"]}))
    npan = len(panels) + (1 if tiles else 0)
    ncol = min(2, npan)
    nrow = -(-npan // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 5.0 * nrow), squeeze=False)
    axes = list(axes.flat)
    for ax in axes[npan:]:
        ax.axis("off")
    if tiles:                                                  # the box plot goes last
        axes = axes[:len(panels)] + [axes[npan - 1]]
    vmax = max([max(v.values() or [0]) for _, v, _, _ in panels] + [0.05])
    for ax, (title, val, clabel, tree) in zip(axes, panels):
        segs, cols, lws = [], [], []
        for a_, b_, cl in mp["couplers"]:
            segs.append([xy[a_], xy[b_]])
            cols.append(CLASS_COL.get(cl, INK_MUTED))
            on_tree = tree is None or tuple(sorted((a_, b_))) in tree
            lws.append((3.0 if cl != "exact" else 2.0) if on_tree else 0.8)
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, zorder=1, capstyle="round"))
        qs = sorted(val)
        sc = ax.scatter([xy[q][0] for q in qs], [xy[q][1] for q in qs], c=[val[q] for q in qs], cmap=cmap,
                        vmin=0.0, vmax=vmax, s=300, zorder=2,
                        edgecolors=[INK if q in seam else "#c9c8c2" for q in qs],
                        linewidths=[2.4 if q in seam else 1.0 for q in qs])
        for q in qs:
            ax.text(xy[q][0], xy[q][1], f"{max(val[q], 0.0):.2f}", ha="center", va="center", fontsize=6.5, zorder=3,
                    color="#fcfcfb" if val[q] / vmax > 0.5 else INK)
        cb = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.02)
        cb.set_label(clabel, color=INK_2, fontsize=8)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, colors=INK_2)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_aspect("equal")
        ax.margins(0.08)
        ax.axis("off")
    axes[0].legend(handles=[Line2D([], [], color=CLASS_COL["exact"], lw=2, label="coupler inside one simulator"),
                            Line2D([], [], color=CLASS_COL["replica"], lw=3, label="coupler through a seam"),
                            Line2D([], [], marker="o", ls="", markersize=9, markerfacecolor="#fcfcfb",
                                   markeredgecolor=INK, markeredgewidth=2, label="seam qubit")],
                   loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=7.5)
    if tiles:
        ax = axes[-1]
        key = headline(tiles[0])[0]
        by = collections.defaultdict(list)
        for r in tiles:
            by[r["seam_distance"]].append(r[key])
        ds = sorted(d for d in by if d is not None)
        bp = ax.boxplot([by[d] for d in ds], positions=range(len(ds)), widths=0.45, patch_artist=True,
                        medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=INK_2),
                        capprops=dict(color=INK_2), flierprops=dict(marker="o", markersize=4,
                                                                   markerfacecolor=INK_MUTED, markeredgecolor="none"))
        for b in bp["boxes"]:
            b.set(facecolor="#9ec5f4", edgecolor=INK_2)
        ax.set_xticks(range(len(ds)), [f"{d}\n({len(by[d])} units)" for d in ds], fontsize=8, color=INK_2)
        ax.set_xlabel("unit's distance to the nearest seam qubit (couplers)", color=INK_2, fontsize=8.5)
        ax.set_ylabel(key, color=INK_2, fontsize=8.5)
        ax.axhline(0.5, color=INK_MUTED, lw=0.8, ls="--")
        ax.text(len(ds) - 0.5, 0.51, "0.5: genuine entanglement above", ha="right", va="bottom", fontsize=7,
                color=INK_MUTED)
        ax.set_ylim(min(0.0, min(min(v) for v in by.values()) - 0.05), 1.05)
        ax.set_title(f"{key} by seam distance", fontsize=8.5, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(INK_MUTED)
        ax.tick_params(colors=INK_2, labelsize=8)
    fig.suptitle(f"{reg} on ACE ({mp['ace']['lrc']}x{mp['ace']['lrr']}), Layer 2 {_l2name(ghz[-1])}, per-unit layers "
                 f"{ghz[-1].get('layers', 'none')}; loss below 0 (F_GHZ above 1: shot noise) shown as 0; "
                 f"{ghz[-1].get('qrack', '')}", fontsize=9, color=INK)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    out = a.png or f"ghz_{reg}.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


# ================================================================== selftest

def _np_ghz_rho(k, theta=0.0, p_flip=0.0, p_deph=0.0):
    """numpy density matrix: GHZ with phase theta, then an independent bit flip on qubit 0 with p_flip and a
    global dephasing of the 0..0 / 1..1 coherence by (1 - p_deph)."""
    d = 2 ** k
    psi = np.zeros(d, complex)
    psi[0], psi[-1] = 1 / math.sqrt(2), np.exp(1j * theta) / math.sqrt(2)
    rho = np.outer(psi, psi.conj())
    rho[0, -1] *= (1 - p_deph)
    rho[-1, 0] *= (1 - p_deph)
    X0 = np.kron(np.array([[0, 1], [1, 0]]), np.eye(d // 2))
    return (1 - p_flip) * rho + p_flip * X0 @ rho @ X0


def _np_readout_probs(rho, k, setting):
    """Outcome probabilities (index bit i = qubit i) after readout(setting) on a numpy rho (qubit 0 = root,
    the most significant factor of the kron products)."""
    I2, H = np.eye(2), np.array([[1, 1], [1, -1]]) / math.sqrt(2)

    def u(th, ph, lm):
        return np.array([[math.cos(th / 2), -np.exp(1j * lm) * math.sin(th / 2)],
                         [np.exp(1j * ph) * math.sin(th / 2), np.exp(1j * (ph + lm)) * math.cos(th / 2)]])
    if setting == "z":
        ops = [I2] * k
    elif setting.startswith("xr"):
        ph = [np.eye(2), np.diag([1, -1j]), np.diag([1, -1]), np.diag([1, 1j])][int(setting[2:])]
        ops = [H @ ph] + [H] * (k - 1)
    else:
        phi = int(setting[1:]) * math.pi / (k + 1)
        ops = [u(math.pi / 2, 0, math.pi - phi)] * k
    U = ops[0]
    for o in ops[1:]:
        U = np.kron(U, o)
    pr = np.real(np.diag(U @ rho @ U.conj().T))
    out = np.zeros(2 ** k)
    for idx_ in range(2 ** k):                          # kron index: qubit 0 is the high bit; ours: bit 0
        b = [(idx_ >> (k - 1 - i)) & 1 for i in range(k)]
        out[sum(bit << i for i, bit in enumerate(b))] = pr[idx_]
    return out


def _bits_from_probs(p, k, S=0):
    """An exact 'bit matrix' stand-in: every outcome repeated in proportion, for the estimators' arithmetic."""
    S = S or 400_000
    counts = np.round(p * S).astype(int)
    rows = np.repeat(np.arange(len(p)), counts)
    return ((rows[:, None] >> np.arange(k)) & 1).astype(np.uint8)


class _Lattice:
    """A small rows x cols grid for the exact-simulator checks (no ACE, no release needed)."""

    def __init__(self, rows, cols):
        self.n = rows * cols
        self.couplers = [(r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1)] + \
                        [(r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols)]
        self.sub = [(q // cols + q % cols) % 2 for q in range(self.n)]


def _ket(sim, n):
    return np.array(sim.out_ket(), dtype=complex)[: 2 ** n]


def cmd_selftest(a):
    from pyqrack import QrackSimulator
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    print("[1] Lemma 4 (MQC) and the witness against numpy density matrices (k = 3, 4, 5)")
    for k in (3, 4, 5):
        for th, pf, pd in ((0.0, 0.0, 0.0), (0.7, 0.0, 0.0), (0.0, 0.1, 0.0), (0.3, 0.05, 0.4)):
            rho = _np_ghz_rho(k, th, pf, pd)
            ghz = np.zeros(2 ** k)
            ghz[0] = ghz[-1] = 1 / math.sqrt(2)
            F0_true = float(np.real(ghz @ rho @ ghz))
            thetas = np.linspace(-math.pi, math.pi, 2001)
            Fmax = max(0.5 * float(np.real(rho[0, 0] + rho[-1, -1] + 2 * np.exp(1j * t) * rho[0, -1])) for t in thetas)
            data = {s: _bits_from_probs(_np_readout_probs(rho, k, s), k) for s in settings_for(k, "both", 99)}
            e = estimate(data, k)
            c = rho[0, -1]
            check(f"k={k} theta={th} flip={pf} dephase={pd}",
                  abs(e["c_re"] - c.real) < 2e-3 and abs(e["c_im"] - c.imag) < 2e-3
                  and abs(e["F_GHZ"] - Fmax) < 2e-3 and abs(e["F0"] - F0_true) < 2e-3
                  and e["F_witness"] <= F0_true + 2e-3,
                  f"c {e['c_re']:+.4f}{e['c_im']:+.4f}i / {c.real:+.4f}{c.imag:+.4f}i, F_GHZ {e['F_GHZ']:.4f} / "
                  f"max_theta {Fmax:.4f}, F0 {e['F0']:.4f} / {F0_true:.4f}, witness {e['F_witness']:.4f}")

    print("[2] Lemma 4 on any rho: the DFT at n returns rho_{0..0,1..1} (random states, k = 4)")
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(20):
        A = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
        rho = A @ A.conj().T
        rho /= np.trace(rho)
        data = {s: _bits_from_probs(_np_readout_probs(rho, 4, s), 4) for s in settings_for(4, "mqc", 99)}
        e = estimate(data, 4)
        worst = max(worst, abs(complex(e["c_re"], e["c_im"]) - rho[0, -1]))
    check("20 random density matrices", worst < 2e-3, f"largest |c_hat - c| {worst:.1e}")

    print("[3] Lemma 3: lambda from the populations of a depolarised GHZ (k = 3, L = 3)")
    for lam in (0.94, 0.7):
        d = 8
        rho = _np_ghz_rho(3)
        for _ in range(3):
            rho = lam * rho + (1 - lam) * np.eye(d) / d
        e = estimate({"z": _bits_from_probs(_np_readout_probs(rho, 3, "z"), 3)}, 3, blocks=3)
        check(f"lambda {lam}", abs(e["lambda_per_block"] - lam) < 2e-3, f"estimate {e['lambda_per_block']:.4f}")

    print("[4] readout gates on Qrack against numpy (k = 3, GHZ with phase 0.5)")
    k = 3
    rho = _np_ghz_rho(k, 0.5)
    for s in settings_for(k, "both", 99):
        sim = QrackSimulator(k, is_gpu=False)
        g = Gates(sim, list(range(k)))
        u = dict(qubits=[0, 1, 2], root=0, edges=[[0, 1], [0, 2]], classes=["exact", "exact"], depth=1)
        prepare(g, u, [])
        sim.u(0, 0.0, 0.0, 0.5)                      # phase 0.5 on |1..1>
        readout(g, u, s)
        pq = np.array([sim.prob_perm([0, 1, 2], [bool((x >> i) & 1) for i in range(k)]) for x in range(2 ** k)])
        pn = _np_readout_probs(rho, k, s)
        check(f"setting {s}", np.max(np.abs(pq - pn)) < 1e-5, f"max |dp| {np.max(np.abs(pq - pn)):.2e}")

    print("[5] Theorem 1 on the exact simulator: W_T(H_D) = I (2 x 3 grid, random state, random J, dw, eps)")
    lat = _Lattice(2, 3)
    for seed in (1, 2, 3):
        drift = Drift(lat.couplers, lat.n, lat.sub, j_zz=0.3, sigma_qs=0.2, t_idle=4.4, seed=seed)
        unit = dict(qubits=list(range(lat.n)))
        probe = Probe(drift, [unit])
        r_ = np.random.default_rng(seed)
        sim = QrackSimulator(lat.n, is_gpu=False)
        g = Gates(sim, list(range(lat.n)))
        for q in range(lat.n):
            g.u(q, *r_.uniform(0, math.pi, 3))
        for a_, b_ in lat.couplers[:4]:
            g.cx(a_, b_)
        before = _ket(sim, lat.n)
        idle_block(g, probe, "dd", drift.eps(seed))
        after = _ket(sim, lat.n)
        fid = abs(np.vdot(before, after)) ** 2
        check(f"seed {seed}", abs(fid - 1) < 1e-6, f"|<psi|W_T|psi>|^2 = {fid:.9f}")
        idle_block(g, probe, "unmitigated", drift.eps(seed))
        check(f"seed {seed}: unmitigated is not the identity (control)", abs(np.vdot(before, _ket(sim, lat.n))) ** 2 < 0.999)

    print("[6] Corollary 2.1 and 4.1 on the exact simulator: a register-wide GHZ under unmitigated drift")
    drift = Drift(lat.couplers, lat.n, lat.sub, j_zz=0.3, sigma_qs=0.0, t_idle=4.4, seed=5)
    unit = dict(qubits=list(range(lat.n)))
    probe = Probe(drift, [unit])
    for with_dw in (False, True):
        if not with_dw:
            drift.dw = np.zeros(lat.n)
        else:                                            # the paper's formula sums to 0 on 2 x 3: shift it
            drift.dw = np.array([0.022 * ((q % 3) - 1.0) + 0.011 for q in range(lat.n)])
        sim = QrackSimulator(lat.n, is_gpu=False)
        g = Gates(sim, list(range(lat.n)))
        g.h(0)
        for q in range(1, lat.n):
            g.cx(0, q)
        L = 3
        for _ in range(L):
            idle_block(g, probe, "unmitigated", drift.eps(0))
        v = _ket(sim, lat.n)
        c = v[0] * np.conj(v[-1])
        want = -2 * L * drift.T * float(drift.dw.sum())
        dphi = (np.angle(c) - want + math.pi) % (2 * math.pi) - math.pi
        check("ZZ only: coherence untouched" if not with_dw else "ZZ + dw: arg c = -2 L T sum dw",
              abs(abs(c) - 0.5) < 1e-6 and abs(dphi) < 1e-6,
              f"|c| {abs(c):.6f}, arg c {np.angle(c):+.6f}, expected {((want + math.pi) % (2 * math.pi)) - math.pi:+.6f}")

    print("[7] the outside-coupling identity: e^{-i phi Z_a Z_b} with b in |0> or |1> = e^{-i phi z_b Z_a}")
    for zb in (0, 1):
        sims = []
        for exact_gate in (True, False):
            sim = QrackSimulator(2, is_gpu=False)
            g = Gates(sim, [0, 1])
            g.u(0, 1.1, 0.3, 0.7)
            if zb:
                g.x(1)
            if exact_gate:
                g.cx(0, 1)
                g.zrot(1, 0.37)
                g.cx(0, 1)
            else:
                g.zrot(0, 0.37 * (1 - 2 * zb))
            sims.append(_ket(sim, 2))
        fid = abs(np.vdot(sims[0], sims[1])) ** 2
        check(f"z_b = {1 - 2 * zb:+d}", abs(fid - 1) < 1e-6, f"overlap {fid:.9f}")      # fp32 Qrack

    print("[8] fold is the identity on the exact simulator")
    sim = QrackSimulator(4, is_gpu=False)
    g = Gates(sim, list(range(4)))
    u = dict(qubits=[1, 0, 2, 3], root=1, edges=[[1, 0], [1, 2], [2, 3]], classes=["exact", "replica", "exact"],
             depth=2)
    prepare(g, u, parse_layers("fold:2:all"))
    p = [sim.prob_perm([0, 1, 2, 3], [bool((x >> i) & 1) for i in range(4)]) for x in range(16)]
    check("fold:2:all keeps the GHZ populations", abs(p[0] - 0.5) < 1e-6 and abs(p[15] - 0.5) < 1e-6)

    print("[9] Layer S amplification (AmpGates) is the identity on the exact simulator (random 3-qubit state)")

    class _SeamStub:                                     # coupler 0-1 through a seam, the rest inside a patch
        @staticmethod
        def cls(a_, b_):
            return "replica" if {a_, b_} == {0, 1} else "exact"
    for kind in ("zz", "cx"):
        kets = []
        for K in (0, 2):
            sim = QrackSimulator(3, is_gpu=False)
            g = AmpGates(sim, [0, 1, 2], _SeamStub, kind, K, 0.3) if K else Gates(sim, [0, 1, 2])
            r_ = np.random.default_rng(7)
            for q in range(3):
                g.u(q, *r_.uniform(0, math.pi, 3))
            g.cx(0, 1)
            g.cx(1, 2)
            g.cx(1, 0)
            kets.append(_ket(sim, 3))
        fid = abs(np.vdot(kets[0], kets[1])) ** 2
        check(f"{kind}, K = 2 (lambda {AmpGates.scale(kind, 2)})", abs(fid - 1) < 1e-6, f"overlap {fid:.9f}")

    print("[10] ZNE: each model returns its own zero-noise value (k = 3, lambda 1, 5, 9)")
    lams = [1, 5, 9]
    obs = [dict(pop=0.25 + 0.75 * 0.9 * math.exp(-0.05 * x), pop_se=0.01, c_re=0.45 * math.exp(-0.08 * x), c_im=0.0,
                c_se=0.01) for x in lams]
    z = zne_extrapolate(lams, obs, 3)
    want = 0.5 * (0.25 + 0.75 * 0.9 + 2 * 0.45)
    check("exponential decay, exp model", z["exp"] is not None and abs(z["exp"]["F"] - want) < 1e-9,
          f"{z['exp']['F']:.6f} / {want:.6f}")
    obs = [dict(pop=0.95 - 0.02 * x, pop_se=0.01, c_re=0.45 - 0.01 * x, c_im=0.0, c_se=0.01) for x in lams]
    z = zne_extrapolate(lams, obs, 3)
    check("linear decay, linear model", abs(z["linear"]["F"] - 0.5 * (0.95 + 0.9)) < 1e-9, f"{z['linear']['F']:.6f}")

    print("[11] Layer S: seam classes and the correction plan (two patches 0, 1, crossbar 2, seam qubits 1, 4)")

    class _RegStub:
        sids = [{0}, {0, 1, 2}, {1}, {0}, {0, 1, 2}]
        seam = {1, 4}

        @classmethod
        def cls(cls, a_, b_):
            sa, sb = cls.sids[a_], cls.sids[b_]
            return "exact" if len(sa) == 1 and sa == sb else "replica" if sa & sb else "cross"
    cl = {tuple(qs): seam_class(_RegStub, dict(qubits=list(qs), classes=[])) for qs in ((0, 3), (0, 1, 3), (1, 2), (0, 1, 2))}
    check("bulk / replica / replica / cut", list(cl.values()) == ["bulk", "replica", "replica", "cut"], str(cl))

    def _row(qs, F, xF, se=0.01):
        e = dict(P0=0.5 * F, P1=0.5 * F, pop=F, F_GHZ=F, F_GHZ_se=se, site_break={str(qs[0]): 0.1})
        x = dict(exact_P0=0.5, exact_P1=0.5, exact_F_GHZ=xF)
        return dict(u=dict(qubits=list(qs), classes=[]), e1=e, x1=x, e2=e, x2=x)
    rows = [_row((0, 1, 3), 0.75, 1.0), _row((0, 1, 3), 0.995, 1.0), _row((0, 1, 2), 0.4, 1.0), _row((0, 3), 1.0, 1.0)]
    seam_plan(_RegStub, rows, "F_GHZ")
    check("code / clean / control / none", [r["plan"] for r in rows] == ["code", "clean", "control", None],
          str([r["plan"] for r in rows]))
    us = code_schedule(_RegStub, dict(qubits=[1, 4, 2], root=1, edges=[[1, 4], [1, 2]], classes=["replica"] * 2))
    check("the code's schedule: into the host simulator first, seam to seam after", us["edges"] == [[1, 2], [1, 4]],
          str(us["edges"]))

    print("[12] the co-located authority rule (stub backend): two seam qubits meet, the crossbar holds both "
          "authorities; one couples into patch 1, its partner's authority follows")

    class _AceStub:
        def __init__(self):
            self.hq = {0: [(0, 5), (1, 5), (2, 0)], 1: [(0, 6), (1, 6), (2, 1)], 7: [(1, 9)]}
            self._authority = {0: {2}, 1: {2}}           # after the seam-to-seam CX: the crossbar pair

        def _unpack(self, lq):
            return self.hq[lq]
    st = _AceStub()
    _colocate(st, 0, st.hq[0], 1, st.hq[1])               # records the partners, moves nothing
    unmoved = st._authority == {0: {2}, 1: {2}}
    st._authority[0] = {1}                                # lineage: qubit 0's CX into patch 1 moves it there
    _colocate(st, 0, st.hq[0], 7, st.hq[7])
    check("partner 1 follows qubit 0 into patch 1; nothing moves before", unmoved and st._authority[1] == {1},
          str(st._authority))

    print("[13] --inside-zz stabilizer on the exact simulator: a 3-qubit GHZ unit on a 2 x 3 grid, dropping the ZZ "
          "inside it changes only a global phase (dd, unmitigated, compensated)")
    lat = _Lattice(2, 3)
    unit = dict(qubits=[0, 1, 2], root=0, edges=[[0, 1], [1, 2]], classes=["exact", "exact"])
    for mode in ("dd", "unmitigated", "compensated"):
        drift = Drift(lat.couplers, lat.n, lat.sub, j_zz=0.3, sigma_qs=0.2, t_idle=4.4, seed=11)
        kets = []
        for rule in ("gate", "stabilizer"):
            sim = QrackSimulator(lat.n, is_gpu=False)
            g = Gates(sim, list(range(lat.n)))
            prepare(g, unit, [])
            probe = Probe(drift, [unit], rule)
            for _ in range(2):
                idle_block(g, probe, mode, drift.eps(3))
            kets.append(_ket(sim, lat.n))
        fid = abs(np.vdot(kets[0], kets[1])) ** 2
        check(f"{mode}: |<gate|stabilizer>|^2 = 1", abs(fid - 1) < 1e-6, f"{fid:.9f}")
    sim_a, sim_b = QrackSimulator(lat.n, is_gpu=False), QrackSimulator(lat.n, is_gpu=False)
    r_ = np.random.default_rng(4)
    angles = r_.uniform(0, math.pi, (3, 3))
    drift = Drift(lat.couplers, lat.n, lat.sub, j_zz=0.3, sigma_qs=0.0, t_idle=4.4, seed=11)
    for sim, rule in ((sim_a, "gate"), (sim_b, "stabilizer")):
        g = Gates(sim, list(range(lat.n)))
        for q in range(3):
            g.u(q, *angles[q])
        idle_block(g, Probe(drift, [unit], rule), "unmitigated", drift.eps(0))
    fid = abs(np.vdot(_ket(sim_a, lat.n), _ket(sim_b, lat.n))) ** 2
    check("control: on a product state that is not a GHZ, dropping the ZZ is not the identity", fid < 0.999,
          f"{fid:.6f}")
    kets = []
    for rule in ("gate", "target-seam", "target-bulk"):
        sim = QrackSimulator(lat.n, is_gpu=False)
        g = Gates(sim, list(range(lat.n)))
        for q in range(3):
            g.u(q, *angles[q])
        idle_block(g, Probe(drift, [unit], rule, seam={1}), "dd", drift.eps(0))
        kets.append(_ket(sim, lat.n))
    fids = [abs(np.vdot(kets[0], kt)) ** 2 for kt in kets[1:]]
    check("probe: the CX's target on the seam or the bulk qubit is the same unitary (product state, dd)",
          all(abs(f_ - 1) < 1e-6 for f_ in fids), ", ".join(f"{f_:.9f}" for f_ in fids))
    print("[14] probe: a chain unit re-rooted at one end keeps its qubits and its tree's edges as one path")
    u = dict(qubits=[11, 10, 12], root=11, edges=[[11, 10], [11, 12]], classes=["replica", "replica"])
    c = chain_from_end(u)
    check("[11, 10, 12] from the middle -> 10 -> 11 -> 12", c["root"] == 10 and c["edges"] == [[10, 11], [11, 12]],
          f"root {c['root']}, edges {c['edges']}")

    print("[15] --engine loom: LoomGates' matrices against Qrack's gates; a GHZ across a link on the ghost loom")

    class _MatQ:                                          # LoomGates' matrices, applied with mtrx on one simulator
        def __init__(self, sim):
            self.sim = sim

        def _emit1(self, qs, q, head, M):
            self.sim.mtrx([complex(x) for x in np.asarray(M).reshape(-1)], q)
    r_ = np.random.default_rng(9)
    worst = 0.0
    for name, args in (("u", tuple(r_.uniform(0, 2 * math.pi, 3))), ("zrot", (0.37,)), ("s", ()), ("sdg", ()),
                       ("h", ()), ("x", ()), ("z", ())):
        sims = [QrackSimulator(2, is_gpu=False) for _ in range(2)]
        for sim in sims:
            sim.u(0, 1.1, 0.4, 2.3)
            sim.u(1, 0.7, 1.9, 0.2)
            sim.mcx([0], 1)
        getattr(Gates(sims[0], [0, 1]), name)(1, *args)
        getattr(LoomGates(_MatQ(sims[1]), None), name)(1, *args)
        worst = max(worst, 1 - abs(np.vdot(_ket(sims[0], 2), _ket(sims[1], 2))) ** 2)
    check("u, zrot, s, sdg, h, x, z equal Qrack's on an entangled state", worst < 1e-6, f"worst 1 - fid {worst:.2e}")
    try:
        cloverfield(None)
        repo = str(nq.find_repo(None))
    except (SystemExit, Exception) as err:
        print(f"  [skip] the loom part needs nighthawk_cloverfield.py and the rcs-nighthawk release ({err})")
    else:
        Rl = loom_register(repo, None, ["rect:8-10,0-2", "rect:8-10,3-5"], None)
        u0, v0 = Rl.links[0]
        x0 = next(q for q in sorted(Rl.adj[u0]) if Rl.chip_of[q] == Rl.chip_of[u0] and q != v0)
        unit = chain_unit(Rl, [x0, u0, v0])
        ok_cls = seam_class(Rl, unit) == "replica"
        res = {}
        for pol in ("authority", "real"):
            per = [dict()]
            for st in settings_for(3, "auto", 12):
                sh = _loom_task(dict(repo=repo, cf=None, chips=["rect:8-10,0-2", "rect:8-10,3-5"], link_rows=[],
                                     units=[unit], setting=st, shots=1024, exact=False, realisation=0,
                                     layer2=dict(mode="none", blocks=0), policy={tuple(unit["qubits"]): pol}))
                per[0][st] = bit_matrix(sh, Rl.n)
            res[pol] = score_unit(Rl, unit, per, 3, 1, None, maps=False)["F_GHZ"]
        check(f"chain {unit['qubits']} across link {u0}-{v0}: class replica; read on the covering chip F ~ 1, "
              f"on the real copies F <= 1/2", ok_cls and res["authority"] > 0.95 and res["real"] < 0.55,
              f"authority {res['authority']:.3f}, real {res['real']:.3f}")
    print("selftest " + ("passed" if ok else "FAILED"))
    if not ok:
        raise SystemExit(1)


# ================================================================== command line

def make_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo", default=None, help="rcs-nighthawk clone (default: auto-detect, $NIGHTHAWK_REPO)")
    common.add_argument("--register", default="nighthawk61r2",
                        help="release, any register of nighthawk_qrack.py (nighthawk61r2, nighthawk2p45, nighthawk88e, "
                             "...) or one saved by its layoutsearch (fav5p112, ...); its ACE preset placement is used")
    common.add_argument("--shots", type=int, default=4096, help="shots per readout setting and realisation")
    common.add_argument("--repeats", type=int, default=3,
                        help="realisations per setting: a fresh ACE backend and, with Layer 2, a fresh eps draw each "
                             "(the paper's K)")
    common.add_argument("--estimator", choices=["auto", "mqc", "witness", "both"], default="auto",
                        help="auto: Lemma 4's MQC up to --mqc-max qubits, the Clifford witness above")
    common.add_argument("--mqc-max", dest="mqc_max", type=int, default=12,
                        help="largest unit the MQC readout (2k + 2 settings, not Clifford) runs on with --estimator auto")
    common.add_argument("--layer2", default="none",
                        help="idle-window refocusing after the preparation: none, unmitigated, compensated (route A, "
                             "oracle H_hat), dd (route B, Walsh echo: the anti-OTOC layer; also 'walsh', 'anti-otoc')")
    common.add_argument("--blocks", type=int, default=3, help="Layer 2: idle windows L (the paper's GHZ blocks)")
    common.add_argument("--t-idle", dest="t_idle", type=float, default=4.4, help="Layer 2: window T (time units)")
    common.add_argument("--j-zz", dest="j_zz", type=float, default=0.065, help="Layer 2: mean J_ij (rad per time unit)")
    common.add_argument("--sigma-qs", dest="sigma_qs", type=float, default=0.015,
                        help="Layer 2: quasi-static detuning spread (rad per time unit)")
    common.add_argument("--seed", type=int, default=42, help="Layer 2: seed of the J_ij draw and the eps realisations")
    common.add_argument("--layers", default="none",
                        help="per-unit hook: none, fold:F[:all], zzfold:K[:all[:phi]], gfold:K, joined by '+'")
    common.add_argument("--exact-shots", dest="exact_shots", type=int, default=128,
                        help="shots of the exact reference per setting and realisation")
    common.add_argument("--no-exact", dest="no_exact", action="store_true", help="skip the exact reference")
    common.add_argument("--stabilizer", choices=["auto", "on", "off"], default="auto",
                        help="ACE's patch simulators as stabilizer hybrids. auto: on for chip with Clifford-only readout "
                             "and no Layer 2 (a GHZ across a patch is otherwise 2^width amplitudes), off otherwise")
    common.add_argument("--force", action="store_true", help="chip: run dense patches above 30 qubits anyway")
    common.add_argument("--ace-boundary-rep", dest="ace_boundary_rep", action="store_true",
                        help="ACE's is_boundary_repetition_code (off unless given, as in nighthawk_qrack.py)")
    common.add_argument("--code-authority", dest="code_authority", choices=AUTHORITY_RULES, default="lineage",
                        help="the boundary code's authority rule, wherever the code runs (--ace-boundary-rep; stack's "
                             "Layer V): lineage = PyQrack 2.29.5's own; colocated = lineage, and a seam qubit whose "
                             "authority moves to a simulator takes its seam partners' authority with it")
    common.add_argument("--inside-zz", dest="inside_zz", choices=["gate", "stabilizer"], default="gate",
                        help="Layer 2: the drift's ZZ between two qubits of one GHZ unit as CX Rz CX (gate), or dropped "
                             "(stabilizer: a global phase on the GHZ, Corollary 2.1; on ACE it saves the seam "
                             "crossings it would make)")
    common.add_argument("--gpu", action="store_true", help="ACE and the reference on OpenCL (QRACK_OCL_DEFAULT_DEVICE)")
    common.add_argument("--ocl-sync", dest="ocl_sync", action="store_true", help="with --gpu: QRACK_OCL_SYNC=1")
    common.add_argument("--jobs", type=int, default=1, help="worker processes, one circuit each (0 = every core)")
    common.add_argument("--progress", type=float, default=30.0, help="seconds between heartbeats (0 = off)")
    common.add_argument("--out", default=None, help="append JSONL records (ghz_map once, then ghz per state)")

    c = sub.add_parser("chip", parents=[common], help="one GHZ state over the whole register")
    c.add_argument("--trees", default="bfs,seam", help="comma list of bfs, seam")
    c.add_argument("--root", type=int, default=None, help="root qubit (default: the tree's centre)")

    t = sub.add_parser("tiles", parents=[common], help="k-qubit GHZ chains tiled over the register")
    t.add_argument("--unit", type=int, default=3, help="qubits per GHZ chain")
    t.add_argument("--orient", default="rows,cols", help="rows, cols or both")
    t.add_argument("--offsets", default="all", help="chain offsets along a line: all, or a comma list")

    s = sub.add_parser("stack", parents=[common],
                       help="pass 1 Layer 1a, pass 2 Layer 2, Layer S (seam syndrome + repetition code), Layer V "
                            "(the corrected measurement)")
    s.add_argument("--mode", choices=["tiles", "chip"], default="tiles", help="GHZ chains tiled, or one over the register")
    s.add_argument("--unit", type=int, default=3, help="tiles: qubits per GHZ chain")
    s.add_argument("--orient", default="rows,cols", help="tiles: rows, cols or both")
    s.add_argument("--offsets", default="all", help="tiles: chain offsets along a line: all, or a comma list")
    s.add_argument("--trees", default="bfs,seam", help="chip: comma list of bfs, seam")
    s.add_argument("--root", type=int, default=None, help="chip: root qubit (default: the tree's centre)")
    s.add_argument("--zne-amp", dest="zne_amp", choices=sorted(AmpGates.PER), default="zz",
                   help="ZNE (with --zne-k): amplification of every seam crossing: zz (ZZ(phi) ZZ(-phi) after each, "
                        "lambda = 1 + 4K) or cx (CX CX, lambda = 1 + 2K)")
    s.add_argument("--zne-k", dest="zne_k", default="",
                   help="ZNE for comparison with the code: comma list of K (identities per seam crossing), e.g. 1,2 "
                        "(lambda 1, 5, 9); default none")
    s.add_argument("--zne-phi", dest="zne_phi", type=float, default=0.3, help="ZNE, zz: the ZZ angle phi")
    s.add_argument("--code-schedule", dest="code_schedule", choices=["on", "off"], default="on",
                   help="Layer S: reorder each code unit's CXs, into the host simulator before into seam qubits (on), "
                        "or keep the preparation's order (off; with --code-authority colocated the code needs no "
                        "schedule)")
    s.add_argument("--engine", choices=["ace", "loom"], default="ace",
                   help="ace: the register on ACE's patches (default). loom: two exact chips joined by "
                        "nighthawk_cloverfield.py's ghost loom as the seam (--loom-chips, --loom-link-rows); Layer S's "
                        "code is then the loom's authority (a unit read on the chip that holds copies of all of it)")
    s.add_argument("--loom-chips", dest="loom_chips", nargs="+", default=None,
                   help="loom: the chips, nighthawk_cloverfield.py --chips tokens (default rect:1-8,2-5 rect:1-8,6-9: "
                        "nighthawk61r2's 61 qubits as two chips of 30 and 31, its two ACE patches' size)")
    s.add_argument("--loom-link-rows", dest="loom_link_rows", type=int, nargs="+", default=None,
                   help="loom: the rows that get a link (default with the default chips: 1 2 3 4 6 7 8, every coupler "
                        "of nighthawk61r2 between columns 5 and 6; otherwise every facing pair)")
    s.add_argument("--cloverfield", default=None,
                   help="loom: nighthawk_cloverfield.py (default: $NIGHTHAWK_CLOVERFIELD, next to this script, or "
                        "../rcs-nighthawk)")
    s.add_argument("--order", choices=["echo-code", "code-echo"], default="echo-code",
                   help="echo-code: pass 2 = Layer 2 (the anti-OTOC echo), then Layer S and Layer V; code-echo: Layer S "
                        "from pass 1, pass C = the code on pass 1's circuit, then pass E = Layer 2 with the code kept")

    tiles_opts = argparse.ArgumentParser(add_help=False)
    tiles_opts.add_argument("--unit", type=int, default=3, help="qubits per GHZ chain")
    tiles_opts.add_argument("--orient", default="rows,cols", help="rows, cols or both")
    tiles_opts.add_argument("--offsets", default="all", help="chain offsets along a line: all, or a comma list")

    w = sub.add_parser("sweep", parents=[common, tiles_opts],
                       help="tiles over a range of Walsh-echo window counts, by seam class (RESULTS 10.10)")
    w.add_argument("--windows", default="0,1,2,3,4,5,6,8", help="window counts L, comma list (0 = no Layer 2; dd "
                                                               "otherwise, overriding --layer2 and --blocks)")
    w.add_argument("--code-at", dest="code_at", default="",
                   help="window counts to run again under the boundary repetition code (--code-authority applies)")
    w.add_argument("--summarise", nargs="+", default=None, metavar="JSONL",
                   help="no runs: the means of the tiles records in these files")
    w.add_argument("--summary", default=None, help="write the means as JSON here")

    pr = sub.add_parser("probe", parents=[common, tiles_opts],
                        help="one seam class's chains through Layer 2 (dd unless --layer2), the drift's inside ZZ "
                             "put through ACE several ways (RESULTS 10.11)")
    pr.add_argument("--seam-class", dest="seam_class", choices=SEAM_CLASSES, default="cut")
    pr.add_argument("--variants", default=",".join(PROBE_VARIANTS),
                    help="comma list of " + ", ".join(PROBE_VARIANTS) + " (chain-*: the GHZ prepared from one end; "
                         "none: no Layer 2; gate, tseam, tbulk: the inside ZZ as CX Rz CX with the CX's target as "
                         "listed, on the seam qubit, on the bulk qubit; stab: dropped)")
    pr.add_argument("--exact-variants", dest="exact_variants", default="gate,stab,chain-stab",
                    help="variants also run on the exact simulator")

    m = sub.add_parser("stackmap", help="a stack run's three measurements as heatmaps on the chip layout")
    m.add_argument("--in", dest="inp", required=True, help="JSONL written by stack --out (its latest run is drawn)")
    m.add_argument("--register", default=None, help="which register (default: the latest stack run in the file)")
    m.add_argument("--png", default=None, help="output image (default ghz_stack_<register>_<order>.png)")

    p = sub.add_parser("plot", help="the map of a records file")
    p.add_argument("--in", dest="inp", required=True, help="JSONL written by chip / tiles --out")
    p.add_argument("--register", default=None, help="which register (default: the last map in the file)")
    p.add_argument("--layers", default=None, help="only records with this per-unit layer stack")
    p.add_argument("--layer2", default=None, help="only records with this Layer 2 mode")
    p.add_argument("--png", default=None, help="output image (default ghz_<register>.png)")

    sub.add_parser("selftest", help="paper identities against numpy and the exact simulator")
    return ap


def main():
    a = make_parser().parse_args()
    if getattr(a, "ocl_sync", False):
        os.environ["QRACK_OCL_SYNC"] = "1"
    if a.cmd in ("chip", "tiles", "stack", "sweep", "probe"):
        nq.ensure_isolated(a)
    dict(chip=cmd_chip, tiles=cmd_tiles, stack=cmd_stack, sweep=cmd_sweep, probe=cmd_probe, stackmap=cmd_stackmap,
         plot=cmd_plot, selftest=cmd_selftest)[a.cmd](a)


if __name__ == "__main__":
    main()
