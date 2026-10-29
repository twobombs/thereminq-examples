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
  shor      Shor's [[m^2, 1, m]] code instead of GHZ, across QrackBridge circuits (--engine bridge: two chips, or
            the box with --box L H): every placement of an m x m unit (--size m) whose encoder CXs are couplers,
            each in its own circuit, all four logical states encoded as nighthawk_qrack.py shor9 encodes them;
            raw and decoded success, the outcome distribution against the exact one, the bond qubits per cut
            the prepared state keeps, per unit and by how the cut meets the code (inside one circuit, between
            its blocks, through them; on the box direct or relayed, and the wrap cut).
  shormap   a shor run as heatmaps (--in): logical Z and X side by side, one row per way the cut meets the code.
  dual      the dual Nighthawk R1 of arXiv:2610.29986 (IBM): two 120-qubit modules joined by four L-couplers, the
            distant CZ built as the paper builds it (CNOT into the L-qubit, transfer, CZ, transfer back, CNOT), its
            Bell and GHZ states under four models (the hard cut, the bridge exact, the bridge with the paper's
            constituent budget or with the benchmarked distant CZ as Pauli trajectories) against Table 1.
  dualmap   GHZ fidelity heatmaps over both modules of the dual Nighthawk R1 and their L-couplers: 3-qubit chains
            tiled over each lattice plus every chain through a link, one panel per model (default hard cut, budget,
            measured), module B drawn rotated to face A, the L-qubits and cables between them.
  dualplot  dual's model-against-measurement plot from its records (--in).
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
shor writes one shor_map (the register's map) and one shor_bridge record per unit (blocks, outer and inner trees,
layout, chips, crossing CXs, per logical state raw, decoded, tvd, impossible, bonds kept, peak bonds, truncations,
fidelity_estimate, vanished and the exact reference's decoded success; logical_Z, logical_X). probe writes one
ghz_probe record per unit, variant and reference (variant, exact, seam_class and the Layer 1a
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
  python3 nighthawk_ghz.py shor  --box 5 5 --bridge-max-bonds 1 --jobs 2 --out shor9_box5_mb1.jsonl
  python3 nighthawk_ghz.py shormap --in shor9_box5_mb1.jsonl --png shor9_box5_mb1.png
  python3 nighthawk_ghz.py dual  --models hardcut,ideal,exact,budget,measured --trajectories 20000 --jobs 2 \\
          --out dual_r1.jsonl --png dual_r1.png

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
                cells=[list(R.cell[q]) for q in u["qubits"]],
                **({"box": R.unit_box(u)} if hasattr(R, "unit_box") else {}))


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
    orients = tuple(x.strip() for x in a.orient.split(","))
    if hasattr(R, "tilings"):                            # the box: straight lines along x, y and z
        tl = R.tilings(k, ("x", "y", "z") if a.orient == "rows,cols" else orients, offs)
    else:
        tl = tilings(R, k, orients, offs)
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


SEAM_CLASSES = ("bulk", "replica", "cut", "bridge")


def unit_hosts(R, u):
    """The ACE simulators that hold every qubit of a unit (a seam qubit is held by the simulator of each replica)."""
    return sorted(set.intersection(*(set(R.sids[q]) for q in u["qubits"])))


def seam_class(R, u):
    """bulk: one simulator holds the unit and none of its qubits is a seam qubit. replica: one simulator holds the
    unit, its seam qubits through their replica there; their seam error is a disagreement between replicas, which
    the replica repetition code corrects. cut: no simulator holds the whole unit; its GHZ needs coherence between
    two simulators, and ACE has none to give (c = 0, F_GHZ <= 1/2). bridge (--engine bridge): the unit's qubits sit
    on two chips joined by QrackBridge, which carries its entanglement up to its bond capacity."""
    if getattr(R, "read", None) == "bridge" and len({R.chip_of[q] for q in u["qubits"]}) > 1:
        return "bridge"
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
                     "control" if r["cls"] in ("cut", "bridge") else None)
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
          f"simulator holds it, ACE keeps no coherence between simulators, F_GHZ <= 1/2: rerun without a code"
          + ("; bridge: across QrackBridge, no code, rerun as the control" if "bridge" in summary else "") + ")")


def cmd_stack(a):
    """Pass 1 Layer 1a, pass 2 Layer 2, Layer S (the seam's syndrome from passes 1 and 2, then the replica repetition
    code on the units it can correct; ZNE for comparison with --zne-k), Layer V (the third GHZ measurement: pass 2's
    circuit under the correction, the uncorrectable units as a control), compared unit by unit."""
    if a.ace_boundary_rep:
        raise SystemExit("ghz stack: passes 1 and 2 measure the seam uncorrected and Layer S applies the boundary "
                         "repetition code itself: drop --ace-boundary-rep")
    if getattr(a, "box", None) and a.engine != "bridge":
        raise SystemExit("ghz stack --box: the box's faces are joined by QrackBridge: add --engine bridge")
    if a.engine == "bridge":
        if a.loom_chip_backend != "exact":
            raise SystemExit("ghz stack --engine bridge: QrackBridge patches are exact simulators; drop "
                             "--loom-chip-backend ace")
        a.engine, a.loom_read = "loom", "bridge"
        qrack_bridge(a.qrack_bridge)                     # fail early if it cannot be found
    elif a.engine == "loom" and a.loom_read == "bridge":
        raise SystemExit("ghz stack: --loom-read bridge is --engine bridge")
    if a.engine == "loom":
        if a.mode != "tiles":
            raise SystemExit("ghz stack --engine loom: tiles only (a chip-wide GHZ over two chips is always split)")
        if a.zne_k:
            raise SystemExit("ghz stack --engine loom: no ZNE on the loom")
        if a.loom_chip_backend == "exact":
            a.code_schedule = "off"                      # exact chips: the loom reads, nothing to reorder
        if getattr(a, "box", None):
            if a.loom_chips or a.loom_link_rows:
                raise SystemExit("ghz stack --box: the faces are the chips; drop --loom-chips / --loom-link-rows")
            R = box_register(*a.box)
            a.loom_chips, a.loom_link_rows = list(BOX_FACES), []
        else:
            a.loom_chips = a.loom_chips or list(LOOM_DEFAULT["chips"])
            R = loom_register(a.repo, a.cloverfield, a.loom_chips, a.loom_link_rows, a.loom_chip_backend, a.loom_read)
            a.loom_link_rows = list(R.link_rows)
        if R.read == "bridge":
            R.map["loom"]["bridge"] = dict(max_bonds=a.bridge_max_bonds, truncation=a.bridge_truncation,
                                           lazy_bonds=a.bridge_lazy, source=_QB.get("src"))
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
        code = ("no code (the bridge is exact up to its truncation; its units rerun as the control)"
                if getattr(a, "loom_read", "") == "bridge" else
                "the loom's authority (each unit read on the chip holding copies of all its qubits)"
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
    report_box(R, rows, (("pass 1", "e1"), ("pass 2", "e2"), ("V", "eV")))
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
    if getattr(a, "_bridge_log", None):
        recs.append(dict(base_record(a, R, l2), test="ghz_bridge", passes=a._bridge_log))
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
    report_box(R, rows, (("pass 1", "e1"), ("C", "eC"), ("E", "eE")))
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
    if getattr(a, "_bridge_log", None):
        recs.append(dict(base_record(a, R, l2), test="ghz_bridge", passes=a._bridge_log))
    write_records(a, R, recs, None)


def report_box(R, rows, passes):
    """--box: F_GHZ per box edge (the face pair a unit crosses, relayed or not) and pass; a pass that did not rerun
    a unit keeps the previous pass's value, as in the stackmap."""
    if not hasattr(R, "unit_box"):
        return
    by = collections.defaultdict(list)
    for r in rows:
        b = R.unit_box(r["u"])
        by[(len(b["cuts"]), b["edge"] or "inside one face")].append(r)
    print("\nBox: F_GHZ by box edge crossed (ring cuts per unit: 0 inside a face, 1 a direct cut, 2 relayed)")
    print(f"  {'edge':<26}{'cuts':>5}{'units':>6}" + "".join(f"{n_:>9}" for n_, _ in passes))
    for (nc, edge), rs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        vals = []
        for i, (_, key_) in enumerate(passes):
            v = []
            for r in rs:
                e = r.get(key_) or next((r[k_] for _, k_ in reversed(passes[:i]) if r.get(k_)), None)
                v.append(e["F_GHZ"])
            vals.append(float(np.mean(v)))
        print(f"  {edge:<26}{nc:>5}{len(rs):>6}" + "".join(f"{x:>9.3f}" for x in vals))


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

LOOM_DEFAULT = dict(chips=("rect:1-8,2-5", "rect:1-8,6-9"))
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
    """Two (or more) chips of the cloverfield as one register, side by side in a row (chip c + 1 drawn and tiled
    to the right of chip c), joined by the loom's links. Answers what Register answers for ACE: sids[q] = the
    (chip, simulator) pairs holding a copy of q (an exact chip is one simulator, an ACE chip its patches and
    crossbar), seam = every qubit with more than one, cls of a coupler as for ACE. Default links: the facing
    border pairs that are lattice neighbours once the chips sit side by side.
      read cut (default)  the loom as a hard cut: chips are independent simulators, no ghosts; a link CZ reaches
                          each side as the mean-field phase pi P(1) of the other (the cloverfield's --loom mean):
                          no entanglement crosses, a unit across the loom is a product of its chips' parts
                          (F_GHZ <= 1/2) whatever is done to it. Copies are the real qubits only, so such a unit
                          is class cut. Layer S's code: ACE's boundary repetition code inside an ACE chip.
      read ghost          the cloverfield's --loom ghost: every chip also simulates a ghost of each partner its
                          links touch, every gate replayed on it, the link CZ exact between a qubit and the
                          partner's ghost. The ghosts are a replica band two columns wide (both anchor columns on
                          both chips); a unit inside one chip's band is simulated there whole, and reading it there
                          (Layer S's 'code' in this mode) reads a circuit that never crossed the loom. Units wider
                          than the band are cut, as without ghosts."""

    def __init__(self, repo, cf_path, chips, link_rows, backend="exact", read="cut"):
        cf = cloverfield(cf_path)
        nq.load_register_files()
        self.repo, self.backend = repo, backend
        self.specs = [cf.parse_chip(t, repo) for t in chips]
        if backend == "ace":
            bad = [sp["name"] for sp in self.specs if sp["kind"] != "register"]
            if bad:
                raise SystemExit(f"ghz --engine loom --loom-chip-backend ace: chips need a register with an ACE "
                                 f"preset (nighthawk61r2, nighthawk2p28, ...), not {bad}")
        lays = [cf.chip_layout(sp, repo) for sp in self.specs]
        cells = [[(p_ // COLUMNS, p_ % COLUMNS) for p_ in lay.l2p[:lay.n]] for lay in lays]
        self.shift, right = [], None
        for cl in cells:
            sh = 0 if right is None else right + 1 - min(c_ for _, c_ in cl)
            self.shift.append(sh)
            right = max(c_ for _, c_ in cl) + sh
        explicit = bool(link_rows)
        if not link_rows:
            link_rows = []
            for r in sorted(set.intersection(*({r_ for r_, _ in cl} for cl in cells))):
                ok = True
                for c in range(len(cells) - 1):
                    ra = max(c_ for r_, c_ in cells[c] if r_ == r) + self.shift[c]
                    rb = min(c_ for r_, c_ in cells[c + 1] if r_ == r) + self.shift[c + 1]
                    ok = ok and rb == ra + 1
                if ok:
                    link_rows.append(r)
        self.link_rows = list(link_rows)
        self.field = cf.Field(repo, self.specs, (1, len(self.specs)), "all", self.link_rows)
        f = self.field
        self.skipped = [list(x) for x in f.skipped]       # facing pairs with no CZ colour free at both ends
        if f.skipped and explicit:
            raise SystemExit(f"ghz --engine loom: links asked for but without a free CZ colour: {f.skipped}")
        self.link_rows = [r for r in self.link_rows
                          if any(f.cells[f.chip_of[u]][f.local[u]][0] == r for u, _, _ in f.links)]
        if not f.links:
            raise SystemExit("ghz --engine loom: no link between the chips")
        self.n = f.n
        self.name = (("bridge:" if read == "bridge" else "loom:") + "+".join(sp["name"] for sp in self.specs)
                     + ("" if backend == "exact" else ":ace") + ("" if read == "bridge" else f":{read}"))
        self.chip_of, self.local = list(f.chip_of), list(f.local)
        self.read = read
        self.ghosts = f.ghost_plan() if read == "ghost" else [[] for _ in self.specs]   # [(partner, anchor, site)]
        self.ghost_index = [{g[0]: f.chip_n[c] + j for j, g in enumerate(plan)} for c, plan in enumerate(self.ghosts)]
        hold = [set() for _ in range(self.n)]
        self.chip_widths = []
        for c in range(len(self.specs)):
            if backend == "ace":
                eng = ace_chip_engine(cf, self.specs[c], repo, self.anchors(c), False)
                row = nq.layout_row(eng.size, eng.lrc, eng.lrr, eng.torus, eng.opts)
                site_sims = [frozenset(s_ for s_, _ in u_) for u_ in row["unpack"]]
                idx = list(eng.idx)
                sims = lambda l_, idx=idx, ss=site_sims: ss[idx[l_]]          # noqa: E731
                self.chip_widths.append(list(eng.ghost_widths))
                del eng
            else:
                sims = lambda l_: frozenset([0])                  # noqa: E731
                self.chip_widths.append([f.chip_n[c] + len(self.ghosts[c])])
            for q in range(self.n):
                if f.chip_of[q] == c:
                    hold[q] |= {(c, s_) for s_ in sims(f.local[q])}
                elif q in self.ghost_index[c]:
                    hold[q] |= {(c, s_) for s_ in sims(self.ghost_index[c][q])}
        self.sids = [frozenset(h) for h in hold]
        self.seam = {q for q in range(self.n) if len(self.sids[q]) > 1}
        self.couplers = [tuple(e) for e in f.edges]
        self.links = [(u, v) for u, v, _ in f.links]
        self.adj = collections.defaultdict(set)
        for a_, b_ in self.couplers:
            self.adj[a_].add(b_)
            self.adj[b_].add(a_)
        self.cell = {q: (cells[f.chip_of[q]][f.local[q]][0], cells[f.chip_of[q]][f.local[q]][1] + self.shift[f.chip_of[q]])
                     for q in range(self.n)}
        self.sub = [(r + c) % 2 for r, c in (self.cell[q] for q in range(self.n))]
        bad = [(a_, b_) for a_, b_ in self.couplers if self.sub[a_] == self.sub[b_]]
        if bad:
            raise SystemExit(f"ghz --engine loom: the field is not bipartite under r + c parity once the chips sit "
                             f"side by side: couplers {bad[:5]} (choose --loom-link-rows)")
        self.seam_dist = Register._bfs_dist(self, self.seam)
        self.size, self.lrc, self.lrr, self.torus = None, None, None, False
        self.widths = [w for ws in self.chip_widths for w in ws]
        self.engine = "loom"
        self.map = dict(n=self.n, columns=COLUMNS,
                        sites=[[q, self.cell[q][0], self.cell[q][1], sorted([list(x) for x in self.sids[q]])]
                               for q in range(self.n)],
                        couplers=[[a_, b_, self.cls(a_, b_)] for a_, b_ in self.couplers], ace=None,
                        loom=dict(chips=[sp["name"] for sp in self.specs], chip_n=list(f.chip_n), backend=backend,
                                  read=read,
                                  links=[list(x) for x in self.links], link_rows=self.link_rows,
                                  ghosts=[[g[0] for g in plan] for plan in self.ghosts], widths=self.chip_widths))

    region = Register.region
    cls = Register.cls

    def anchors(self, c):
        return [anc for _, anc, _ in self.ghosts[c]]

    def describe(self):
        kind = ("ACE chips (two-patch presets)" if self.backend == "ace" else "exact chips")
        how = ("loom ghost: a replica band, not a channel" if self.read == "ghost" else
               "QrackBridge across the links: gate teleportation through bond qubits" if self.read == "bridge" else
               "loom a hard cut: mean-field across the links, no ghosts")
        return (f"{self.name} ({self.n} qubits; {kind} of {', '.join(str(x) for x in self.field.chip_n)} qubits"
                + (" + ghosts" if self.read == "ghost" else "") + f", widths {self.chip_widths}; {len(self.links)} "
                f"links on rows {self.link_rows}, {how}"
                + (f"; {len(self.skipped)} facing pairs without a free CZ colour left unlinked" if self.skipped else "")
                + ")")

    def unit_reads(self, u, policy):
        """Where each qubit of a unit is read: (chip, local index) per qubit. real: its own copy. authority: on
        the chip holding copies of every qubit of the unit (most real copies first), real where none does."""
        f = self.field
        if policy == "authority":
            hosts = set.intersection(*({c for c, _ in self.sids[q]} for q in u["qubits"]))
            if hosts:
                c = min(hosts, key=lambda c_: (-sum(1 for q in u["qubits"] if f.chip_of[q] == c_), c_))
                return [(c, f.local[q]) if f.chip_of[q] == c else (c, self.ghost_index[c][q]) for q in u["qubits"]]
        return [(f.chip_of[q], f.local[q]) for q in u["qubits"]]


_LOOM_REGS = {}


def loom_register(repo, cf_path, chips, link_rows, backend="exact", read="cut"):
    key = (repo, cf_path, tuple(chips), tuple(link_rows or ()), backend, read)
    if key not in _LOOM_REGS:
        _LOOM_REGS[key] = LoomRegister(repo, cf_path, chips, link_rows, backend, read)
    return _LOOM_REGS[key]


BOX_FACES = ("base", "front", "right", "lid", "back", "left")          # the bridge ring, in this order
BOX_NORMAL = dict(base=(0, 0, -1), lid=(0, 0, 1), front=(0, -1, 0), back=(0, 1, 0), left=(-1, 0, 0), right=(1, 0, 0))


class BoxRegister:
    """A closed box: the lattice points on the surface of an L x L x H block (x, y in 0..L-1, z in 0..H-1),
    couplers between points at unit distance, six circuits, one per face: the base (z = 0), four walls and the
    lid (z = H-1), each a QrackBridge patch. A point on a box edge belongs to one face: base and lid keep their
    whole square; the walls share the vertical edges round the box (front y = 0 holds x = 0..L-2, right x = L-1
    holds y = 0..L-2, back y = L-1 holds x = 1..L-1, left x = 0 holds y = 1..L-1). Qubits are numbered face by
    face in the ring order base, front, right, lid, back, left, and the six patches close into a ring
    (QrackBridge wrap=True): opposite faces sit three apart on it, so every box edge joins faces one apart (six
    edges, a direct cut) or two apart (six edges, relayed through the face between, one bond on each of the two
    cuts). Bipartite under x + y + z parity, so the Walsh echo's sublattices are exact. cell[q] places q on the
    unfolded net: the base in the middle, the walls folded out round it, the lid beyond the back wall."""

    def __init__(self, L=5, H=5):
        if L < 3 or H < 3:
            raise SystemExit("ghz --box: L and H must be 3 or more")
        self.L, self.H = L, H

        def face(p):
            x, y, z = p
            if z == 0:
                return "base"
            if z == H - 1:
                return "lid"
            if y == 0 and x <= L - 2:
                return "front"
            if x == L - 1 and y <= L - 2:
                return "right"
            if y == L - 1 and x >= 1:
                return "back"
            return "left"
        pts = [(x, y, z) for x in range(L) for y in range(L) for z in range(H)
               if x in (0, L - 1) or y in (0, L - 1) or z in (0, H - 1)]
        pts.sort(key=lambda p: (BOX_FACES.index(face(p)), p[2], p[1], p[0]))
        self.xyz = pts
        pos = {p: q for q, p in enumerate(pts)}
        self.n = len(pts)
        self.face = [face(p) for p in pts]
        self.chip_of = [BOX_FACES.index(f) for f in self.face]
        self.chip_n = [self.chip_of.count(c) for c in range(len(BOX_FACES))]
        self.local = [q - sum(self.chip_n[:self.chip_of[q]]) for q in range(self.n)]
        self.couplers = []
        for q, (x, y, z) in enumerate(pts):
            for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                p = (x + d[0], y + d[1], z + d[2])
                if p in pos:
                    self.couplers.append((q, pos[p]))
        self.adj = collections.defaultdict(set)
        for a_, b_ in self.couplers:
            self.adj[a_].add(b_)
            self.adj[b_].add(a_)
        self.links = [(a_, b_) for a_, b_ in self.couplers if self.chip_of[a_] != self.chip_of[b_]]
        self.sub = [(x + y + z) % 2 for x, y, z in pts]
        self.cell = {q: self.net(p, self.face[q]) for q, p in enumerate(pts)}
        assert len(set(self.cell.values())) == self.n
        self.sids = [frozenset({(self.chip_of[q], 0)}) for q in range(self.n)]
        self.seam = set()                                 # no simulator holds a copy of another face's qubit
        self.seam_dist = {q: math.inf for q in range(self.n)}
        self.specs = [dict(name=f, kind="face") for f in BOX_FACES]
        self.backend, self.read, self.engine = "exact", "bridge", "loom"
        self.name = f"box{L}x{L}x{H}"
        self.size, self.lrc, self.lrr, self.torus = None, None, None, False
        self.widths = list(self.chip_n)
        self.link_rows, self.skipped = [], []
        P = len(BOX_FACES)
        pairs = collections.Counter(tuple(sorted((self.chip_of[a_], self.chip_of[b_]))) for a_, b_ in self.links)
        self.edges = []
        for (i, j), cnt in sorted(pairs.items()):
            d = (j - i) % P
            step = 1 if d <= P - d else -1
            via = [BOX_FACES[(i + step * s_) % P] for s_ in range(1, min(d, P - d))]
            self.edges.append(dict(faces=[BOX_FACES[i], BOX_FACES[j]], couplers=cnt, ring_distance=min(d, P - d),
                                   via=via))
        self.map = dict(n=self.n, columns=None,
                        sites=[[q, self.cell[q][0], self.cell[q][1], [[self.chip_of[q], 0]], list(self.xyz[q]),
                                self.face[q]] for q in range(self.n)],
                        couplers=[[a_, b_, self.cls(a_, b_)] for a_, b_ in self.couplers], ace=None,
                        loom=dict(chips=list(BOX_FACES), chip_n=list(self.chip_n), backend="exact", read="bridge",
                                  links=[list(x) for x in self.links], link_rows=[], ghosts=[[] for _ in BOX_FACES],
                                  widths=[[x] for x in self.chip_n],
                                  box=dict(L=L, H=H, ring=list(BOX_FACES), wrap=True, edges=self.edges)))

    region = Register.region
    cls = Register.cls

    def net(self, p, f):
        """(row, column) of point p (on face f) on the unfolded net."""
        L, H = self.L, self.H
        x, y, z = p
        return dict(base=(H - 1 + y, H - 1 + x), front=(H - 1 - z, H - 1 + x), back=(H + L - 2 + z, H - 1 + x),
                    left=(H - 1 + y, H - 1 - z), right=(H - 1 + y, H + L - 2 + z),
                    lid=(2 * H + 2 * L - 4 - y, H - 1 + x))[f]

    def describe(self):
        direct = sum(1 for e in self.edges if e["ring_distance"] == 1)
        return (f"{self.name} ({self.n} qubits on the surface of a {self.L} x {self.L} x {self.H} box, "
                f"{len(self.couplers)} couplers; six faces "
                + ", ".join(f"{f} {n_}" for f, n_ in zip(BOX_FACES, self.chip_n))
                + f", one QrackBridge patch each on a ring in that order; {len(self.links)} couplers across the "
                f"{len(self.edges)} box edges, {direct} edges on a ring cut, {len(self.edges) - direct} relayed "
                f"through one face)")

    def ring_cuts(self, i, j):
        """The ring cuts (cut c joins patches c and c + 1, cut P-1 the wrap) a bridged gate from patch i to patch
        j crosses, the shorter way round, as QrackBridge routes it."""
        P = len(BOX_FACES)
        d = (j - i) % P
        step = 1 if d <= P - d else -1
        out, p = [], i
        while p != j:
            nx = (p + step) % P
            out.append(p if step == 1 else nx)
            p = nx
        return out

    def unit_box(self, u):
        """A unit's place on the box: its faces (ring order), the box edge it crosses (one line: 'base-right via
        front'), the ring cuts its preparation's bridged CXs cross (one bond each per crossing unit)."""
        faces = sorted({self.chip_of[q] for q in u["qubits"]})
        cuts = set()
        for p_, c_ in u["edges"]:
            if self.chip_of[p_] != self.chip_of[c_]:
                cuts.update(self.ring_cuts(self.chip_of[p_], self.chip_of[c_]))
        names = [BOX_FACES[f] for f in faces]
        via = []
        for p_, c_ in u["edges"]:
            a_, b_ = self.chip_of[p_], self.chip_of[c_]
            if a_ != b_:
                cs = self.ring_cuts(a_, b_)
                if len(cs) > 1:
                    P = len(BOX_FACES)
                    d = (b_ - a_) % P
                    step = 1 if d <= P - d else -1
                    via += [BOX_FACES[(a_ + step * s_) % P] for s_ in range(1, len(cs))]
        edge = "-".join(names) + (" via " + "+".join(sorted(set(via), key=BOX_FACES.index)) if via else "")
        return dict(faces=names, edge=(edge if len(names) > 1 else None), cuts=sorted(cuts),
                    hops=max((len(self.ring_cuts(self.chip_of[p_], self.chip_of[c_])) for p_, c_ in u["edges"]),
                             default=0))

    def tilings(self, k, orients=("x", "y", "z"), offsets=None):
        """[(orientation, offset, [units])]: k-qubit chains along straight lattice lines of the surface (x, y or
        z), split where the line leaves the surface, cut into chunks of k from the offset. A chain can turn no
        corner, so the units across a box edge are the ones whose line runs off one face onto the next."""
        pos = {p: q for q, p in enumerate(self.xyz)}
        ax = dict(x=0, y=1, z=2)
        out = []
        for ori in orients:
            i = ax[ori]
            line = collections.defaultdict(list)
            for p in self.xyz:
                line[tuple(v for j, v in enumerate(p) if j != i)].append(p)
            runs = []
            for key in sorted(line):
                ps = sorted(line[key], key=lambda p: p[i])
                run = [ps[0]]
                for a_, b_ in zip(ps, ps[1:]):
                    if b_[i] == a_[i] + 1:
                        run.append(b_)
                    else:
                        runs.append(run)
                        run = [b_]
                runs.append(run)
            for f in (range(k) if offsets is None else offsets):
                units = [chain_unit(self, [pos[p] for p in run[j:j + k]]) for run in runs
                         for j in range(f, len(run) - k + 1, k)]
                if units:
                    out.append((ori, f, units))
        return out


_BOX_REGS = {}


def box_register(L=5, H=5):
    if (L, H) not in _BOX_REGS:
        _BOX_REGS[(L, H)] = BoxRegister(L, H)
    return _BOX_REGS[(L, H)]


def ace_chip_engine(cf, spec, repo, anchors, brep):
    """nighthawk_qrack's AceEngine on a register chip's ACE preset, with the boundary repetition code on or off
    (the same sites and patches either way) and the chip's ghosts on free ACE sites coupled to their anchors
    (the cloverfield's place_ghosts, deterministic): logical n + j is ghost j."""
    lay = cf.chip_layout(spec, repo)
    a = cf.chip_engine_args(spec, dict(backend="ace", cpu=True, ace_gpu=False))
    a.ace_boundary_rep = bool(brep)
    eng = nq.AceEngine(lay, lay.n, a)
    sites, widths = cf.place_ghosts(eng, anchors, None) if anchors else ([], list(eng.widths))
    eng.idx = list(eng.idx) + list(sites)
    eng.ghost_widths = widths
    return eng


class LoomChip:
    """One chip for the loom, in this process: an exact simulator (the cloverfield's ChipSim) or the chip
    register's ACE preset (ace_chip_engine), each with the chip's ghosts, driven by the loom's op stream plus one
    op it lacks, ("m", q, M): a 2 x 2 matrix on local qubit q."""

    def __init__(self, cf, spec, repo, anchors, backend="exact", brep=False):
        self.backend = backend
        if backend == "ace":
            self.eng = ace_chip_engine(cf, spec, repo, anchors, brep)
        else:
            self.eng = cf.ChipSim(spec, repo, dict(backend="exact", cpu=True, ace_gpu=False,
                                                   ghost_anchors=anchors)).eng
        self._r = None

    def site(self, l_):
        return self.eng.idx[l_] if self.backend == "ace" else l_

    def put(self, msg):
        e = self.eng
        for op in msg[1]:
            k = op[0]
            if k == "m":
                e.g1(op[1], op[2])
            elif k == "p":
                e.pauli(op[1], op[2])
            elif k == "cz":
                e.cz(op[1], op[2])
            elif k == "ph":
                e.g1(op[1], np.diag([1.0, np.exp(1j * op[2])]))
            elif k == "reset":
                e.reset()
            else:
                raise ValueError(f"loom chip: op {k}")
        self._r = ("m", [float(e.sim.prob(self.site(q))) for q in msg[2]])

    def get(self, want):
        return self._r

    def sample(self, locals_, shots):
        return np.array(self.eng.sim.measure_shots([self.site(l_) for l_ in locals_], shots), dtype=np.uint64)


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


_QB = {}


def qrack_bridge(path=None):
    """PyQrack's QrackBridge (PyQrack >= 2.30), or qrack_bridge.py loaded from --qrack-bridge /
    $NIGHTHAWK_QRACK_BRIDGE as a submodule of the installed PyQrack (it needs only QrackSimulator and the
    library's OutKet / InKet, which PyQrack 2.29.5 / Qrack 10.13.2 have)."""
    if "cls" in _QB:
        return _QB["cls"]
    try:
        from pyqrack import QrackBridge
        _QB["cls"], _QB["src"] = QrackBridge, "pyqrack"
        return QrackBridge
    except ImportError:
        pass
    path = path or os.environ.get("NIGHTHAWK_QRACK_BRIDGE")
    if not path or not Path(path).is_file():
        raise SystemExit("ghz --engine bridge: this PyQrack has no QrackBridge (PyQrack >= 2.30 has it); give "
                         "--qrack-bridge PATH to its qrack_bridge.py, or set NIGHTHAWK_QRACK_BRIDGE")
    import importlib.util
    import pyqrack
    spec = importlib.util.spec_from_file_location("pyqrack.qrack_bridge", str(path))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pyqrack.__name__
    sys.modules["pyqrack.qrack_bridge"] = mod
    spec.loader.exec_module(mod)
    _QB["cls"], _QB["src"] = mod.QrackBridge, str(path)
    return mod.QrackBridge


class BridgeGates:
    """Gates (the interface of Gates) on a QrackBridge: field qubit q at bridge index idx[q]. A CX between two
    patches is the bridge's gate teleportation through a bond pair, recompressed to at most max_bonds bond
    qubits per cut."""

    def __init__(self, br, idx):
        self.b, self.q = br, idx

    def h(self, q):
        self.b.h(self.q[q])

    def x(self, q):
        self.b.x(self.q[q])

    def z(self, q):
        self.b.u(self.q[q], 0.0, 0.0, math.pi)

    def s(self, q):
        self.b.s(self.q[q])

    def sdg(self, q):
        self.b.adjs(self.q[q])

    def u(self, q, th, ph, lm):
        self.b.u(self.q[q], th, ph, lm)

    def zrot(self, q, phi):
        self.b.u(self.q[q], 0.0, 0.0, 2.0 * phi)

    def cx(self, c, t):
        self.b.mcx([self.q[c]], self.q[t])


def _bridge_layout(R, units):
    """One bridge patch per chip, holding the chip's qubits of these units (idle qubits stay |0> and in product,
    and a coupling to one is a classical rotation, so they need no place), padded to equal size because
    QrackBridge splits its qubits into equal contiguous patches. Returns (bridge width, patches, idx)."""
    per = collections.defaultdict(list)
    for u in units:
        for q in u["qubits"]:
            per[R.chip_of[q]].append(q)
    P = len(R.specs)
    m = max(1, max((len(v) for v in per.values()), default=1))
    idx = {}
    for c in range(P):
        for j, q in enumerate(sorted(per.get(c, []))):
            idx[q] = c * m + j
    return m * P, P, idx


def _loom_task(spec):
    """_loom_task_once, on QrackBridge with numpy's divide / invalid errors raised: on a ring (the box) the bridge
    truncates the wrap cut by a local SVD of its two patches, which on a ring is not a projection of the state (a
    periodic MPS has no canonical form), and where a GHZ unit's branches tie the wrap cut and an inner cut can keep
    opposite branches. The contracted state is then zero: its renormalisation divides 0 / 0 and the NaN state
    would be sampled or break a later SVD. Here it stops the run with the circuit named (RESULTS 10.16; the
    repro is bridge_ring_repro.py)."""
    if spec["exact"] or spec.get("read") != "bridge":
        return _loom_task_once(spec)
    try:
        with np.errstate(divide="raise", invalid="raise"):
            return _loom_task_once(spec)
    except (np.linalg.LinAlgError, FloatingPointError) as err:
        raise RuntimeError(f"ghz --engine bridge: the bridge's state vanished ({err}) on group {spec.get('group')}, "
                           f"setting {spec['setting']}, realisation {spec['realisation']}, max_bonds "
                           f"{spec.get('max_bonds')}" + (": a truncation on the ring (wrap cut) zeroed it; raise "
                                                         "--bridge-max-bonds, or keep bridged gates out of Layer 2 "
                                                         "with --inside-zz stabilizer" if spec.get("box") else "")
                           ) from err


def _loom_task_once(spec):
    """One circuit on the loom (or, exact, on one QrackSimulator over the field): every unit prepared, Layer 2,
    the readout setting; shots of every unit qubit, read where spec['policy'] says (LoomRegister.unit_reads),
    as ints with bit q = field qubit q. Each chip's copies are sampled in calls of up to 32 qubits (Qrack returns
    32 bits per shot), each sample shuffled by its own seeded permutation and the samples paired shot by shot:
    chips (and calls) are independent, as the units are. ACE chips run with the boundary repetition code where
    spec['brep'] (Layer S's code) and the authority rule spec['authority']."""
    if spec.get("box"):
        R, cf = box_register(*spec["box"]), None
    else:
        R = loom_register(spec["repo"], spec["cf"], spec["chips"], spec["link_rows"], spec.get("backend", "exact"),
                          spec.get("read", "cut"))
        cf = cloverfield(spec["cf"])
    units = spec["units"]
    bridge = None
    if spec["exact"]:
        from pyqrack import QrackSimulator
        sim = QrackSimulator(R.n, is_gpu=False)
        g = Gates(sim, list(range(R.n)))
    elif R.read == "bridge":
        QB = qrack_bridge(spec.get("qrack_bridge"))
        width, P, idx = _bridge_layout(R, units)
        bridge = QB(width, max_bonds=int(spec["max_bonds"]), patches=P, truncation=spec.get("truncation", "greedy"),
                    lazy_bonds=int(spec.get("lazy_bonds", 0)), wrap=bool(spec.get("box")),
                    seed=[int(spec["realisation"]), zlib_key(spec["setting"]), zlib_key(spec.get("group", ""))])
        g = BridgeGates(bridge, idx)
    else:
        if R.backend == "ace":
            set_authority_rule(spec.get("authority", "lineage"))
        chips = [LoomChip(cf, sp, spec["repo"], R.anchors(c), R.backend, spec.get("brep", False))
                 for c, sp in enumerate(R.specs)]
        loom = cf.Loom(R.field, chips, "ghost" if R.read == "ghost" else "mean")
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
    if bridge is not None:
        qs = sorted(idx)
        w = np.array(bridge.measure_shots([idx[q] for q in qs], spec["shots"]), dtype=object)
        bits = np.zeros((spec["shots"], R.n), dtype=np.uint8)
        for j, q in enumerate(qs):
            bits[:, q] = [(int(x) >> j) & 1 for x in w]
        packed = np.packbits(bits, axis=1, bitorder="little")
        return ([int.from_bytes(row.tobytes(), "little") for row in packed],
                dict(fidelity=float(bridge.fidelity_estimate), truncations=int(bridge.truncations),
                     peak_bonds=int(bridge.peak_bonds), width=int(width)))
    loom._flush(queues, [[] for _ in chips])
    S = spec["shots"]
    bits = np.zeros((S, R.n), dtype=np.uint8)
    pol = spec["policy"]
    per = collections.defaultdict(list)                  # chip -> [(field qubit, local index)]
    for u in units:
        for q, (c, l_) in zip(u["qubits"], R.unit_reads(u, pol.get(tuple(u["qubits"]), "real"))):
            per[c].append((q, l_))
    for c, ql in sorted(per.items()):
        for k0 in range(0, len(ql), 32):
            part = ql[k0:k0 + 32]
            w = chips[c].sample([l_ for _, l_ in part], S)
            w = w[np.random.default_rng([spec["realisation"], c, k0, zlib_key(spec["setting"])]).permutation(S)]
            for j, (q, _) in enumerate(part):
                bits[:, q] = (w >> np.uint64(j)) & np.uint64(1)
    packed = np.packbits(bits, axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), "little") for row in packed]


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
                box=([R.L, R.H] if isinstance(R, BoxRegister) else None), layer2=l2, backend=R.backend, read=R.read, brep=bool(brep),
                authority=getattr(a, "code_authority", "lineage"),
                **(dict(max_bonds=a.bridge_max_bonds, truncation=a.bridge_truncation, lazy_bonds=a.bridge_lazy,
                        qrack_bridge=a.qrack_bridge) if R.read == "bridge" else {}))
    calls, labels, keys = [], [], []
    for name, units in groups:
        pol = {tuple(u["qubits"]): pol_name for u in units}
        for st in settings:
            for r in range(reps):
                calls.append((_loom_task, (dict(base, units=units, setting=st, shots=a.shots, exact=False,
                                                realisation=r + rep_offset, policy=pol, group=name),)))
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
    stats = []
    for (name, st, r, ex), shots in zip(keys, res):
        if isinstance(shots, tuple):
            shots, info = shots
            stats.append(info)
        bits = bit_matrix(shots, R.n)
        if ex:
            bad += check_exact(unit_of[name], st, bits, pid)
            exact[name][r][st] = bits
        else:
            data[name][r][st] = bits
    if stats:
        fid = [x["fidelity"] for x in stats]
        summ = dict(label=label, circuits=len(stats), max_bonds=a.bridge_max_bonds, truncation=a.bridge_truncation,
                    lazy_bonds=a.bridge_lazy, peak_bonds=max(x["peak_bonds"] for x in stats),
                    truncating=sum(1 for x in stats if x["truncations"]), fidelity_mean=round(float(np.mean(fid)), 4),
                    fidelity_min=round(float(min(fid)), 4), max_width=max(x["width"] for x in stats))
        a._bridge_log = getattr(a, "_bridge_log", []) + [summ]
        print(f"  bridge [{label}]: {summ['circuits']} circuits, peak {summ['peak_bonds']} bond qubits per cut "
              f"(max_bonds {summ['max_bonds']}), {summ['truncating']} truncated, fidelity_estimate mean "
              f"{summ['fidelity_mean']:.3f}, min {summ['fidelity_min']:.3f}; widest bridge {summ['max_width']} qubits")
    return data, (None if no_exact else exact), (None if no_exact else bad), time.time() - t0


# ================================================================== Shor's code across QrackBridge circuits

def shor_grid_unit(at, m):
    """Shor's [[m^2, 1, m]] unit on a grid at(i, j) (block i, position j), built as nighthawk_qrack's shor9_tiles
    builds it: leaders at the middle position, the outer tree along the leaders from the middle outwards, each
    inner tree along its block from the leader outwards."""
    L = (m - 1) // 2
    order = [L] + [i for k in range(1, m) for i in (L - k, L + k) if 0 <= i < m]
    chain = [(i, i - 1) for i in range(L, 0, -1)] + [(i, i + 1) for i in range(L, m - 1)]
    return dict(blocks=[[at(i, j) for j in order] for i in order],
                outer=[[at(i, L), at(k, L)] for i, k in chain],
                inner=[[at(i, j), at(i, k)] for i in order for j, k in chain])


def box_cross_ring(R, ax):
    """The box's cross-section perpendicular to axis ax (0 = x, 1 = y, 2 = z) as a closed ring of (a, b) points in
    the other two coordinates, in order round it."""
    ext = (R.L, R.L, R.H)
    o = [i for i in range(3) if i != ax]
    A, B = ext[o[0]], ext[o[1]]
    return o, ([(a_, 0) for a_ in range(A)] + [(A - 1, b_) for b_ in range(1, B)]
               + [(a_, B - 1) for a_ in range(A - 2, -1, -1)] + [(0, b_) for b_ in range(B - 2, 0, -1)])


def shor_candidates(R, m):
    """[(unit, where)]: every placement of an m x m Shor unit whose encoder CXs are couplers of R, in both
    orientations. On the box: m consecutive points along an axis, inside the box's extent (not on its end faces),
    times m consecutive points round the box's cross-section there, so a unit bends over the box edges it crosses;
    blocks round the cross-section (orientation ring) or along the axis (axis). Elsewhere: m x m squares of lattice
    cells, blocks along rows or along columns."""
    grids = []
    if isinstance(R, BoxRegister):
        pos = {p: q for q, p in enumerate(R.xyz)}
        ext = (R.L, R.L, R.H)
        for ax in range(3):
            o, ring = box_cross_ring(R, ax)
            for s0 in range(1, ext[ax] - m):
                for t0 in range(len(ring)):
                    def at(i, j, ax=ax, o=o, ring=ring, s0=s0, t0=t0):
                        p = [0, 0, 0]
                        p[ax] = s0 + i
                        p[o[0]], p[o[1]] = ring[(t0 + j) % len(ring)]
                        return pos[tuple(p)]
                    grids.append((at, dict(axis="xyz"[ax], start=s0, ring_start=t0), ("ring", "axis")))
    else:
        cell = {rc: q for q, rc in R.cell.items()}
        for (r0, c0) in sorted(cell):
            if all((r0 + i, c0 + j) in cell for i in range(m) for j in range(m)):
                grids.append((lambda i, j, r0=r0, c0=c0: cell[(r0 + i, c0 + j)], dict(row=r0, col=c0),
                              ("rows", "cols")))
    out = []
    for at, where, names in grids:
        for name, f in ((names[0], at), (names[1], lambda i, j, at=at: at(j, i))):
            u = shor_grid_unit(f, m)
            if all(c in R.adj[p] for p, c in u["outer"] + u["inner"]):
                out.append((u, dict(where, blocks=name)))
    return out


def shor_unit_info(R, u):
    """Where a Shor unit sits against the bridge's cuts: the chips (faces) it spans, its encoder CXs that cross
    between chips (outer: leader to leader; inner: inside a block), the blocks the cut splits, and its layout:
    inside (one circuit), blocks whole (only outer CXs cross: the cut runs between blocks, as nighthawk_qrack's
    patch_blocks) or blocks split (inner CXs cross: the cut runs through blocks, as its seam_leaders). On the box
    also the ring cuts the bridged CXs cross, the most cuts one CX crosses (hops: 1 direct, 2 relayed) and whether
    one is the ring's wrap cut."""
    ch = R.chip_of
    names = [sp["name"] for sp in R.specs]
    qs = [q for b in u["blocks"] for q in b]
    xo = sum(1 for p, c in u["outer"] if ch[p] != ch[c])
    xi = sum(1 for p, c in u["inner"] if ch[p] != ch[c])
    split = sum(1 for b in u["blocks"] if len({ch[q] for q in b}) > 1)
    info = dict(chips=[names[c] for c in sorted({ch[q] for q in qs})], cross_outer=xo, cross_inner=xi,
                blocks_split=split, layout="inside" if not (xo or xi) else "blocks whole" if not split else
                "blocks split", qubits=qs, cells=[list(R.cell[q]) for q in qs])
    if isinstance(R, BoxRegister):
        cuts, hops = set(), 0
        for p, c in u["outer"] + u["inner"]:
            if ch[p] != ch[c]:
                cs = R.ring_cuts(ch[p], ch[c])
                cuts.update(cs)
                hops = max(hops, len(cs))
        info.update(cuts=sorted(cuts), hops=hops, wrap=(len(BOX_FACES) - 1) in cuts)
    return info


def shor_class(info):
    """The report's class of a unit: its layout, blocks split and, on the box, direct or relayed (and the wrap)."""
    c = info["layout"] + (f", {info['blocks_split']} split" if info["blocks_split"] else "")
    if info.get("hops"):
        c += (", relayed" if info["hops"] > 1 else ", direct") + (", wrap" if info.get("wrap") else "")
    return c


def _shor_register(spec):
    if spec.get("box"):
        return box_register(*spec["box"])
    return loom_register(spec["repo"], spec["cf"], spec["chips"], spec["link_rows"], "exact", "bridge")


def _shor_task(spec):
    """One logical state of one Shor unit: nighthawk_qrack's shor9_prepare (input, encoder, readout basis) on a
    QrackBridge holding the unit's qubits, one patch per chip (a ring on the box), or on a QrackSimulator of the
    unit (exact); shots over the unit's qubits in block order (bit i = i-th qubit of the blocks in turn). Returns
    (shots, info); a bridge whose truncation zeroed the state (0 / 0 in a recompression, possible on a ring)
    returns (None, info with vanished)."""
    u = spec["unit"]
    qs = [q for b in u["blocks"] for q in b]
    if spec["exact"]:
        from pyqrack import QrackSimulator
        sim = QrackSimulator(len(qs), is_gpu=False)
        nq.shor9_prepare(sim, u, spec["state"], {q: i for i, q in enumerate(qs)})
        return [int(x) for x in sim.measure_shots(list(range(len(qs))), spec["shots"])], {}
    R = _shor_register(spec)
    QB = qrack_bridge(spec.get("qrack_bridge"))
    width, P, idx = _bridge_layout(R, [dict(qubits=qs)])
    br = QB(width, max_bonds=int(spec["max_bonds"]), patches=P, truncation=spec.get("truncation", "greedy"),
            lazy_bonds=int(spec.get("lazy_bonds", 0)), wrap=bool(spec.get("box")),
            seed=[int(spec["realisation"]), zlib_key(spec["state"]), int(spec["index"])])
    kept = []
    stats = lambda: dict(fidelity=float(br.fidelity_estimate), truncations=int(br.truncations),  # noqa: E731
                         peak_bonds=int(br.peak_bonds), width=int(width), bonds=(kept[0] if kept else None))
    try:
        with np.errstate(divide="raise", invalid="raise"):
            nq.shor9_prepare(br, u, spec["state"], idx)
            br.flush()
            kept.append([int(x) for x in br.bond])      # bond qubits per cut the prepared state keeps
            w = br.measure_shots([idx[q] for q in qs], spec["shots"])
    except (np.linalg.LinAlgError, FloatingPointError) as err:
        return None, dict(stats(), vanished=str(err))
    return [int(x) for x in w], stats()


def cmd_shor(a):
    """Shor's [[m^2, 1, m]] code across QrackBridge circuits: every placement of a unit on the chips (or the box),
    each in its own circuit, all four logical states, raw and decoded success against the exact reference."""
    if a.engine != "bridge":
        raise SystemExit("ghz shor: --engine bridge (Shor's code across ACE's seam is nighthawk_qrack.py shor9)")
    m = int(a.size)
    if m < 3:
        raise SystemExit("ghz shor: --size m >= 3")
    QB_src = qrack_bridge(a.qrack_bridge) and _QB.get("src")
    if a.box:
        R = box_register(*a.box)
        a.loom_chips, a.loom_link_rows = list(BOX_FACES), []
    else:
        a.loom_chips = a.loom_chips or list(LOOM_DEFAULT["chips"])
        R = loom_register(a.repo, a.cloverfield, a.loom_chips, a.loom_link_rows, "exact", "bridge")
        a.loom_link_rows = list(R.link_rows)
    R.map["loom"]["bridge"] = dict(max_bonds=a.bridge_max_bonds, truncation=a.bridge_truncation,
                                   lazy_bonds=a.bridge_lazy, source=QB_src)
    cands = shor_candidates(R, m)
    if a.shor_units == "cross":
        cands = [c for c in cands if shor_unit_info(R, c[0])["layout"] != "inside"]
    if not cands:
        raise SystemExit(f"ghz shor: no {m} x {m} Shor unit fits on {R.name}")
    infos = [shor_unit_info(R, u) for u, _ in cands]
    reps = max(1, int(a.repeats))
    states = nq.SHOR9_STATES
    base = dict(repo=a.repo, cf=a.cloverfield, chips=list(a.loom_chips), link_rows=list(a.loom_link_rows or ()),
                box=([R.L, R.H] if isinstance(R, BoxRegister) else None), max_bonds=a.bridge_max_bonds,
                truncation=a.bridge_truncation, lazy_bonds=a.bridge_lazy, qrack_bridge=a.qrack_bridge)
    calls, labels, keys = [], [], []
    for j, (u, _) in enumerate(cands):
        for s in states:
            for r in range(reps):
                calls.append((_shor_task, (dict(base, unit=u, state=s, shots=a.shots, exact=False, realisation=r,
                                                index=j),)))
                labels.append(f"shor unit {j} |{s}_L> realisation {r + 1}/{reps}")
                keys.append((j, s, False))
            if not a.no_exact:
                calls.append((_shor_task, (dict(base, unit=u, state=s, shots=a.exact_shots, exact=True,
                                                realisation=0, index=j),)))
                labels.append(f"exact shor unit {j} |{s}_L>")
                keys.append((j, s, True))
    print(f"Shor-{m * m} ([[{m * m}, 1, {m}]]) across QrackBridge on {R.describe()}; max_bonds {a.bridge_max_bonds} "
          f"({a.bridge_truncation}, lazy {a.bridge_lazy}); {len(cands)} units ("
          + ", ".join(f"{k} {v}" for k, v in sorted(collections.Counter(shor_class(i) for i in infos).items()))
          + f"), each in its own circuit; {a.shots} shots x {reps} realisation(s) per logical state"
          + ("" if a.no_exact else f", exact {a.exact_shots}") + f"; {nq.qrack_versions()}")
    pool, workers = nq.shor_pool(a.jobs, len(calls))
    every = float(a.progress or 0)
    if every:
        print(f"# ghz shor plan: {len(calls)} circuits, {workers} process(es)", flush=True)
    t0 = time.time()
    try:
        res = nq.shor_pool_map(pool, calls, labels, every, what="ghz shor")
    finally:
        if pool is not None:
            pool.shutdown()
    secs = time.time() - t0
    hist = collections.defaultdict(collections.Counter)
    stat = collections.defaultdict(list)
    for (j, s, ex), (shots, info) in zip(keys, res):
        if shots is None:
            stat[(j, s)].append(info)
            continue
        hist[(j, s, ex)].update(shots)
        if not ex:
            stat[(j, s)].append(info)
    recs = []
    when = time.strftime("%Y-%m-%dT%H:%M:%S")
    for j, ((u, where), info) in enumerate(zip(cands, infos)):
        _, local = nq._local_unit(u)
        per = {}
        for s in states:
            h = hist.get((j, s, False))
            st_ = stat[(j, s)]
            kb = [max(x["bonds"]) for x in st_ if x.get("bonds")]
            d = dict(peak_bonds=max(x["peak_bonds"] for x in st_), bonds=(max(kb) if kb else None),
                     bonds_per_cut=next((x["bonds"] for x in st_ if x.get("bonds")), None),
                     truncations=max(x["truncations"] for x in st_),
                     fidelity_estimate=min(x["fidelity"] for x in st_),
                     vanished=sum(1 for x in st_ if x.get("vanished")))
            if h:
                n_ = sum(h.values())
                ok = [(nq.shor9_decode(k, local, s), v) for k, v in h.items()]
                d.update(raw=sum(v * r_ for (r_, _), v in ok) / n_, decoded=sum(v * d_ for (_, d_), v in ok) / n_,
                         shots=n_, **{k_: v_ for k_, v_ in nq.shor9_compare(h, local, s).items()
                                      if k_ in ("tvd", "tvd_floor", "impossible", "top_wrong")})
            hx = hist.get((j, s, True))
            if hx:
                n_ = sum(hx.values())
                d["exact_decoded"] = sum(v * nq.shor9_decode(k, local, s)[1] for k, v in hx.items()) / n_
            per[s] = d
        lz = [per[s]["decoded"] for s in ("0", "1") if "decoded" in per[s]]
        lx = [per[s]["decoded"] for s in ("+", "-") if "decoded" in per[s]]
        recs.append(dict(test="shor_bridge", register=R.name, when=when, size=m, unit=j, where=where, **info,
                         shor_class=shor_class(info), blocks=u["blocks"], outer=u["outer"], inner=u["inner"],
                         states=per, logical_Z=(float(np.mean(lz)) if len(lz) == 2 else None),
                         logical_X=(float(np.mean(lx)) if len(lx) == 2 else None),
                         bonds_Z=max((per[s]["bonds"] or 0) for s in ("0", "1")),
                         bonds_X=max((per[s]["bonds"] or 0) for s in ("+", "-")),
                         peak_bonds=max(per[s]["peak_bonds"] for s in states),
                         shots=a.shots, repeats=reps, exact_shots=(None if a.no_exact else a.exact_shots),
                         loom=R.map["loom"], qrack=nq.qrack_versions()))
    report_shor(recs)
    print(f"seconds: {secs:.1f}")
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(json.dumps(dict(test="shor_map", register=R.name, when=when, **R.map)) + "\n")
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        print(f"  appended {len(recs)} records and the register map to {a.out}")


def report_shor(recs):
    """Decoded success per class and logical state, logical Z and X, and the bond qubits per cut the prepared
    state kept (the most over the class's units; Z states / X states)."""
    by = collections.defaultdict(list)
    for r in recs:
        by[r["shor_class"]].append(r)
    print(f"\n{'class':<40}{'units':>6}" + "".join(f"{'|' + s + '_L>':>8}" for s in nq.SHOR9_STATES)
          + f"{'log Z':>8}{'log X':>8}{'exact':>7}{'bonds Z/X':>11}{'vanished':>9}")
    for c in sorted(by, key=lambda c_: (c_.split(",")[0] != "inside", c_)):
        rs = by[c]
        cell = []
        for s in nq.SHOR9_STATES:
            v = [r["states"][s]["decoded"] for r in rs if "decoded" in r["states"][s]]
            cell.append(f"{np.mean(v):>8.3f}" if v else f"{'-':>8}")
        lz = [r["logical_Z"] for r in rs if r["logical_Z"] is not None]
        lx = [r["logical_X"] for r in rs if r["logical_X"] is not None]
        ex = [r["states"][s]["exact_decoded"] for r in rs for s in nq.SHOR9_STATES if "exact_decoded" in r["states"][s]]
        van = sum(r["states"][s]["vanished"] for r in rs for s in nq.SHOR9_STATES)
        print(f"{c:<40}{len(rs):>6}" + "".join(cell) + (f"{np.mean(lz):>8.3f}" if lz else f"{'-':>8}")
              + (f"{np.mean(lx):>8.3f}" if lx else f"{'-':>8}") + (f"{np.mean(ex):>7.3f}" if ex else f"{'-':>7}")
              + f"{max(r['bonds_Z'] for r in rs):>6}/{max(r['bonds_X'] for r in rs)}" + f"{van:>9}")


def cmd_shormap(a):
    """A shor run as heatmaps, logical Z (|0_L>, |1_L>) and logical X (|+_L>, |-_L>) side by side, one row per way
    a cut meets the code (between its blocks, through them): per site the lowest decoded success among the units
    through it of that layout or inside one circuit; on the box's unfolded net or the chips side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle
    import textwrap
    recs = read_records(a.inp)
    sr = [r for r in recs if r.get("test") == "shor_bridge"]
    if not sr:
        raise SystemExit(f"ghz shormap: no shor_bridge record in {a.inp}")
    when = max(r["when"] for r in sr)
    sr = [r for r in sr if r["when"] == when]
    mp = [r for r in recs if r.get("test") == "shor_map" and r.get("when") == when][-1]
    reg, m, loom = sr[0]["register"], sr[0]["size"], mp["loom"]
    bp = loom.get("bridge") or {}
    box = loom.get("box")
    sites = {s_[0]: (s_[1], s_[2]) for s_ in mp["sites"]}
    lays = [x for x in ("blocks whole", "blocks split") if any(r["layout"] == x for r in sr)] or ["inside"]
    panels = []
    for lay in lays:
        for key in ("logical_Z", "logical_X"):
            d = collections.defaultdict(list)
            for r in sr:
                if r["layout"] in ("inside", lay) and r[key] is not None:
                    for q in r["qubits"]:
                        d[q].append(r[key])
            panels.append((lay, key, {q: float(min(v)) for q, v in d.items()}))
    cmap = LinearSegmentedColormap.from_list("div", [(0.0, "#a32c2b"), (0.25, "#e98584"), (0.5, "#e4e2dc"),
                                                     (0.75, "#6da7ec"), (1.0, "#184f95")])
    norm = Normalize(0.0, 1.0)
    ink, ink2, muted, surf = "#1d1d1b", "#4a4944", "#6b6a64", "#fcfcfb"
    if box:
        R = box_register(box["L"], box["H"])
        dec = box_net_decor(R, loom, sites)
        rows_, cols_ = dec["rows"], dec["cols"]
    else:
        rows_ = sorted({rc[0] for rc in sites.values()})
        cols_ = sorted({rc[1] for rc in sites.values()})
    nrow, ncol = rows_[-1] - rows_[0] + 1, cols_[-1] - cols_[0] + 1
    classes = ["inside", "blocks whole", "blocks split"]
    if box:
        classes = ["inside", "blocks whole, direct", "blocks whole, relayed", "blocks split, direct",
                   "blocks split, relayed"]

    def cls_of(r):
        c = r["layout"]
        if box and c != "inside":
            c += ", relayed" if r["hops"] > 1 else ", direct"
        return c
    counts = collections.Counter(cls_of(r) for r in sr)

    def mean_of(c, key):
        v = [r[key] for r in sr if cls_of(r) == c and r[key] is not None]
        return float(np.mean(v)) if v else None
    van = sum(r["states"][s]["vanished"] for r in sr for s in nq.SHOR9_STATES)
    cross_only = not any(r["layout"] == "inside" for r in sr)
    what = (f"a closed {box['L']} × {box['L']} × {box['H']} box, six faces on a QrackBridge ring ({' → '.join(box['ring'])})"
            if box else f"{' + '.join(loom['chips'])}, {len(loom['chips'])} exact chips joined by QrackBridge")
    lines = textwrap.wrap(
        f"Shor's [[{m * m}, 1, {m}]] code on {what}: every placement of a {m} × {m} unit whose encoder CXs are "
        f"couplers ({len(sr)} units" + (" across the cut; units inside one chip not run"
                                                                   if cross_only else "") + ", each in its own circuit"
        + (", bent over the box edges it crosses; the box's 8 corner points are in none" if box else "")
        + f"), all four logical states. Each cell is the "
        f"lowest decoded success among the units through that qubit that sit inside one circuit or meet the cut as "
        f"the row says (1: perfect, ½: no logical information). Cut between blocks: only leader-to-leader CXs "
        f"cross (nighthawk_qrack shor9's patch_blocks); cut through blocks: CXs inside the blocks cross "
        f"(seam_leaders). Cuts kept to max_bonds "
        f"{bp.get('max_bonds')} bond qubits ({bp.get('truncation', 'greedy')})."
        + (f" {van} logical-state circuits vanished on the ring's truncation (not counted)." if van else ""), 150)
    if box:
        lines += textwrap.wrap("Net: base in the middle, walls folded out, lid beyond the back wall; bars on the "
                               "couplers across box edges, solid on a ring cut, dashed relayed through the face "
                               "between; edges the net cuts open lettered on both faces: "
                               + ", ".join(f"{ch} {'–'.join(kk)}" for kk, ch in sorted(dec["letters"].items(),
                                                                                        key=lambda x: x[1])) + ".", 150)
    else:
        lines.append("Bars: the couplers between the chips, each a bridged gate (teleported through a bond pair).")
    FW = 17.0
    nr = len(lays)
    cell = min(0.75, 6.8 / ncol, (8.5 if nr == 1 else 5.6 if not box else 7.8) / nrow)
    PW, PH = cell * ncol, cell * nrow
    X0, GAP, RG = 0.45, 0.9, 1.15                          # left margin, gap between columns, between rows
    Hh = 1.0 + 0.22 * len(lines) + 0.4
    FH = Hh + nr * (0.45 + PH + 0.55) + (nr - 1) * (RG - 1.0) + 0.6
    fig = plt.figure(figsize=(FW, FH), dpi=170 if FH > 14 else 200, facecolor=surf)
    sub = dict(logical_Z=("Logical Z", "|0_L⟩, |1_L⟩: read in X, a block's sign = parity of its bits"),
               logical_X=("Logical X", "|+_L⟩, |−_L⟩: read in Z, a block's bit = majority of its bits"))
    row_name = {"blocks whole": "Cut between blocks", "blocks split": "Cut through blocks", "inside": "Inside"}
    for pi, (lay, key, vals_) in enumerate(panels):
        ri, ci = divmod(pi, 2)
        x0 = X0 + ci * (PW + GAP)
        y0 = FH - Hh - (ri + 1) * (0.45 + PH) - ri * (0.55 + RG - 1.0)
        ax = fig.add_axes([x0 / FW, y0 / FH, PW / FW, PH / FH], facecolor=surf)
        fs = 7.0 if cell < 0.5 else 9.5
        for q, F in vals_.items():
            r_, c_ = sites[q]
            ax.add_patch(Rectangle((c_ - 0.5, r_ - 0.5), 1, 1, facecolor=cmap(norm(F)), edgecolor=surf, lw=1.4))
            dark = F > 0.82 or F < 0.12
            ax.text(c_, r_ + 0.08, f"{F:.2f}", ha="center", va="center", fontsize=fs, fontweight="bold",
                    color="#ffffff" if dark else ink)
            ax.text(c_ - 0.42, r_ - 0.30, str(q), ha="left", va="center", fontsize=fs * 0.64,
                    color="#ffffff" if dark else muted)
        if box:
            draw_box_net(ax, dec, ink, ink2, surf)
        else:
            for u_, v_ in loom.get("links", []):
                (ru, cu), (rv, cv) = sites[u_], sites[v_]
                xs, ys = (([(cu + cv) / 2] * 2, [ru - 0.42, ru + 0.42]) if ru == rv else
                          ([cu - 0.42, cu + 0.42], [(ru + rv) / 2] * 2))
                ax.plot(xs, ys, color=surf, lw=5.5, solid_capstyle="butt")
                ax.plot(xs, ys, color=ink, lw=2.8, solid_capstyle="butt")
            ax.set_xlim(cols_[0] - 0.6, cols_[-1] + 0.6)
            ax.set_ylim(rows_[-1] + 0.6, rows_[0] - 0.6)
            ax.set_aspect("equal")
            ax.axis("off")
        ytop = (y0 + PH + 0.25) / FH
        fig.text(x0 / FW, ytop, f"{row_name[lay]} · {sub[key][0]}", fontsize=13, fontweight="bold", color=ink,
                 ha="left")
        fig.text(x0 / FW, ytop - 0.24 / FH, sub[key][1], fontsize=9, color=ink2, ha="left")
        txt = [f"{c_.split(', ')[-1]} {mean_of(c_, key):.2f}" for c_ in classes if counts.get(c_)
               and (c_ == "inside" or c_.startswith(lay))]
        fig.text(x0 / FW, (y0 - 0.32) / FH, "   ".join(txt), fontsize=9.5, color=ink, ha="left", family="monospace")
    yc = FH - Hh - nr * (0.45 + PH) - (nr - 1) * (0.55 + RG - 1.0)
    hc = FH - Hh - 0.45 - yc
    cax = fig.add_axes([(X0 + 2 * PW + GAP + 0.35) / FW, (yc + 0.25 * hc) / FH, 0.13 / FW, 0.5 * hc / FH])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0", "0.25", "½", "0.75", "1"])
    cb.ax.tick_params(labelsize=9, colors=ink2)
    cb.outline.set_visible(False)
    fig.text((X0 + 2 * PW + GAP + 0.15) / FW, (yc + 0.78 * hc) / FH, "success", fontsize=10.5, fontweight="bold",
             color=ink)
    title = (f"Shor-{m * m} across QrackBridge: " + (f"box {box['L']}×{box['L']}×{box['H']}" if box else
                                                     f"{len(loom['chips'])} chips") + f", max_bonds {bp.get('max_bonds')}")
    fig.text(0.02, (FH - 0.5) / FH, title, fontsize=18, fontweight="bold", color=ink, ha="left")
    for i, line in enumerate(lines):
        fig.text(0.02, (FH - 1.0 - 0.22 * i) / FH, line, fontsize=9.8, color=ink2, ha="left")
    fig.text(0.02, 0.42 / FH, f"{os.path.basename(a.inp)} · run {when} · {sr[0]['shots']} shots × {sr[0]['repeats']} "
             f"per logical state · {sr[0].get('qrack', '')} · nighthawk_ghz.py shor --size {m}"
             + (f" --box {box['L']} {box['H']}" if box else
                "" if tuple(loom["chips"]) == LOOM_DEFAULT["chips"] else " --loom-chips " + " ".join(loom["chips"]))
             + (" --shor-units cross" if cross_only else "") + f" --bridge-max-bonds {bp.get('max_bonds')}",
             fontsize=8.3, color=muted, ha="left")
    png = a.png or f"shor_{reg}_mb{bp.get('max_bonds')}.png"
    fig.savefig(png, facecolor=surf)
    plt.close(fig)
    print(f"shormap: {reg}, run {when}: {png}")
    for c in classes:
        if counts.get(c):
            print(f"  {c:<24} {counts[c]:>3} units  logical Z {fmt(mean_of(c, 'logical_Z'))}  "
                  f"logical X {fmt(mean_of(c, 'logical_X'))}")


# ================================================================== dual Nighthawk R1 (arXiv:2610.29986)

DUAL_PAPER = ("N. M. Sundaresan et al. (IBM), Meter-scale quantum interconnects between large quantum processors, "
              "arXiv:2610.29986 (October 2026)")
# One row per L-coupler: lattice qubit on module A, its L-qubit, the L-qubit on module B, its lattice qubit; the state
# transfer's network-benchmarking fidelity and duration (Extended Data Table 2); the distant CZ's randomized-
# benchmarking fidelity (Fig. 3b: 111-237 96.42 and 115-233 95.54 from the text, which says three links reach 95.54
# or better; 113-235 and 117-231 read from the figure, error 0.088 and 0.042, so the four average the text's 94.75).
DUAL_LINKS = [dict(name="240-247", a=111, la=240, lb=247, b=237, transfer=0.9929, tg=312, dcz=0.9642, src="text"),
              dict(name="241-246", a=113, la=241, lb=246, b=235, transfer=0.9887, tg=296, dcz=0.912, src="Fig. 3b"),
              dict(name="242-245", a=115, la=242, lb=245, b=233, transfer=0.9899, tg=256, dcz=0.9554, src="text"),
              dict(name="243-244", a=117, la=243, lb=244, b=231, transfer=0.9886, tg=312, dcz=0.958, src="Fig. 3b")]
DUAL_LQ_EPG = {240: 0.00058, 241: 0.00301, 242: 0.00178, 243: 0.00040, 244: 0.00053, 245: 0.00060, 246: 0.00092,
               247: 0.00044}                               # L-qubit single-qubit error per gate (Extended Data Table 1)
# Table 1 and Extended Data Table 3: F (%) without and with readout mitigation, and their uncertainties
DUAL_MEASURED = {"bell 115-233": ((85.91, 0.40), (93.94, 0.27)), "bell 117-231": ((76.93, 0.49), (86.98, 0.41)),
                 "simultaneous: bell 115-233": ((80.40, 0.46), (87.56, 0.38)),
                 "simultaneous: bell 117-231": ((76.17, 0.49), (86.14, 0.58)),
                 "ghz4 115 116 232 233": ((77.23, 0.67), (91.70, 0.70)),
                 "ghz6 115-117 231-233": ((67.67, 0.66), (90.87, 0.74))}


class DualR1:
    """Two IBM Nighthawk R1 modules joined by the paper's four L-couplers. Each module a 12 x 10 square lattice, 120
    qubits, 218 couplers: qubit 10 r + c on module A, 120 + 10 r + c on module B (the paper's numbering: 111-117 on
    A and 231-237 on B are row 11 of each). Each L-coupler: an L-qubit on each module (240-243 on A, 244-247 on B),
    coupled to one lattice qubit of row 11 by its tunable coupler, the two joined by 80 cm of coax (Fig. 1e)."""

    def __init__(self):
        self.n = 248
        self.module = [0] * 120 + [1] * 120 + [0] * 4 + [1] * 4
        self.couplers = []
        for m in (0, 1):
            for r in range(12):
                for c in range(10):
                    q = 120 * m + 10 * r + c
                    if c < 9:
                        self.couplers.append((q, q + 1))
                    if r < 11:
                        self.couplers.append((q, q + 10))
        for L in DUAL_LINKS:
            self.couplers += [(L["a"], L["la"]), (L["lb"], L["b"])]
        self.cables = [(L["la"], L["lb"]) for L in DUAL_LINKS]
        self.adj = collections.defaultdict(set)
        for a_, b_ in self.couplers:
            self.adj[a_].add(b_)
            self.adj[b_].add(a_)

    def describe(self):
        return (f"dual Nighthawk R1 ({self.n} qubits: 2 x 120 lattice, 218 couplers each, 8 L-qubits; "
                f"{len(self.couplers)} on-module couplers, 4 cables {', '.join(L['name'] for L in DUAL_LINKS)})")


def _dual_link(a, b):
    for i, L in enumerate(DUAL_LINKS):
        if {a, b} == {L["a"], L["b"]}:
            return i
    raise SystemExit(f"ghz dual: no L-coupler joins {a} and {b}")


def dual_states():
    """The paper's circuits (Table 1, Methods), as (ops, target groups, kind): a Bell pair is H and one distant CNOT;
    the GHZ states are rooted at Q115 at one end of the 242-245 bridge, the distant gate first, the two chains fanning
    out in parallel. Plus Bell pairs on the two L-couplers the paper did not use for states (111-237, 113-235)."""
    def bell(a, b):
        return [("h", a), ("dcx", a, b)]
    out = {}
    for a, b in ((111, 237), (113, 235), (115, 233), (117, 231)):
        out[f"bell {a}-{b}"] = (bell(a, b), [[a, b]], "bell")
    out["simultaneous"] = (bell(115, 233) + bell(117, 231), [[115, 233], [117, 231]], "bell")
    g4 = [("h", 115), ("dcx", 115, 233), ("cx", 115, 116), ("cx", 233, 232)]
    out["ghz4 115 116 232 233"] = (g4, [[115, 116, 232, 233]], "ghz")
    out["ghz6 115-117 231-233"] = (g4 + [("cx", 116, 117), ("cx", 232, 231)], [[115, 116, 117, 231, 232, 233]], "ghz")
    return out


class DualGates:
    """The paper's gate set on a simulator (QrackBridge over the two modules, or QrackSimulator), with optional Pauli
    noise drawn per trajectory. A distant CZ between lattice qubits a (module A) and b (module B) through L-coupler
    (la, lb), as in Fig. 3b and Heya et al.: CNOT a -> la (Ry(-pi/2) . CZ . Ry(pi/2) on la), state transfer la -> lb
    (a SWAP with lb in |0>, across the bridge), CZ lb - b, transfer back, CNOT a -> la again: three local CZs, two
    transfers, four single-qubit gates; the virtual phase correction is the identity here.
    noise none: exact. budget: the paper's constituent budget, a depolarising channel after every constituent at
    its benchmarked error (transfer: 1 - F on the arriving L-qubit; local CZ: --local-cz-error on its pair; L-qubit
    rotations: Extended Data Table 1's EPG; lattice single-qubit gates: --lattice-1q-error). measured: the distant
    CZ exact, then a two-qubit depolarising channel on (a, b) at its benchmarked error; local gates as in budget.
    A depolarising channel of average error r is a uniformly drawn non-identity Pauli with probability 3r/2 (one
    qubit) or 5r/4 (two qubits)."""

    def __init__(self, sim, idx, noise="none", r_cz=0.00355, r_1q=0.0005, rng=None):
        self.s, self.q, self.noise, self.r_cz, self.r_1q = sim, idx, noise, r_cz, r_1q
        self.rng = rng or np.random.default_rng()

    def _pauli(self, q, k):
        if k == 1:
            self.s.x(self.q[q])
        elif k == 2:
            self.s.u(self.q[q], math.pi, math.pi / 2, math.pi / 2)
        elif k == 3:
            self.s.u(self.q[q], 0.0, 0.0, math.pi)

    def _dep1(self, q, r):
        if self.noise != "none" and r > 0 and self.rng.random() < 1.5 * r:
            self._pauli(q, int(self.rng.integers(1, 4)))

    def _dep2(self, a, b, r):
        if self.noise != "none" and r > 0 and self.rng.random() < 1.25 * r:
            k = int(self.rng.integers(1, 16))
            self._pauli(a, k // 4)
            self._pauli(b, k % 4)

    def _r1(self, q):
        return DUAL_LQ_EPG.get(q, self.r_1q)

    def h(self, q):
        self.s.h(self.q[q])
        self._dep1(q, self._r1(q))

    def ry(self, q, th):
        self.s.u(self.q[q], th, 0.0, 0.0)
        self._dep1(q, self._r1(q))

    def cz(self, a, b):
        self.s.mcz([self.q[a]], self.q[b])
        self._dep2(a, b, self.r_cz)

    def cx(self, c, t):
        self.s.mcx([self.q[c]], self.q[t])
        self._dep2(c, t, self.r_cz)

    def transfer(self, src, dst, L):
        self.s.swap(self.q[src], self.q[dst])
        if self.noise == "budget":
            self._dep1(dst, 1.0 - L["transfer"])

    def dcz(self, a, b):
        L = DUAL_LINKS[_dual_link(a, b)]
        if L["a"] != a:
            a, b = b, a
        exact = self.noise == "measured"
        keep, self.noise = self.noise, ("none" if exact else self.noise)
        for step in ("cnot", "go", "cz", "back", "cnot"):
            if step == "cnot":
                self.ry(L["la"], -math.pi / 2)
                self.cz(a, L["la"])
                self.ry(L["la"], math.pi / 2)
            elif step == "go":
                self.transfer(L["la"], L["lb"], L)
            elif step == "back":
                self.transfer(L["lb"], L["la"], L)
            else:
                self.cz(L["lb"], b)
        self.noise = keep
        if exact:
            self._dep2(a, b, 1.0 - L["dcz"])

    def dcx(self, c, t):
        self.h(t)
        self.dcz(c, t)
        self.h(t)


def _dual_used(ops):
    qs = set()
    for op in ops:
        if op[0] == "dcx":
            L = DUAL_LINKS[_dual_link(op[1], op[2])]
            qs |= {L["a"], L["la"], L["lb"], L["b"]}
        else:
            qs |= set(op[1:])
    return sorted(qs)


def _dual_rho(v, qs, n):
    psi = v.reshape([2] * n)
    ax = [n - 1 - q for q in qs]
    t = np.transpose(psi, ax + [x for x in range(n) if x not in ax]).reshape(2 ** len(qs), -1)
    return t @ t.conj().T


def _dual_task(spec):
    """A batch of trajectories of one circuit and model: the mean density matrix of each target group (from the
    simulator's ket, so no readout and no shot noise), and the bridge's peak bond count."""
    R1 = DualR1()
    ops, groups, _ = (spec["ops"], spec["groups"], None) if "ops" in spec else dual_states()[spec["state"]]
    used = _dual_used(ops)
    A = [q for q in used if R1.module[q] == 0]
    B = [q for q in used if R1.module[q] == 1]
    m = max(len(A), len(B))
    idx = {q: i for i, q in enumerate(A)}
    idx.update({q: m + i for i, q in enumerate(B)})
    rng = np.random.default_rng(spec["seed"])
    acc = [np.zeros((2 ** len(g), 2 ** len(g)), dtype=complex) for g in groups]
    peak = 0
    for _ in range(int(spec["trajectories"])):
        if spec["engine"] == "exact":
            from pyqrack import QrackSimulator
            sim = QrackSimulator(2 * m, is_gpu=False)
        else:
            sim = qrack_bridge(spec.get("qrack_bridge"))(2 * m, max_bonds=int(spec["max_bonds"]), patches=2)
        g = DualGates(sim, idx, spec["noise"], spec["r_cz"], spec["r_1q"], rng)
        for op in ops:
            getattr(g, op[0])(*op[1:])
        v = np.asarray(sim.out_ket(), dtype=complex)
        if spec["engine"] != "exact":
            peak = max(peak, int(sim.peak_bonds))
        v = v / np.linalg.norm(v)
        for k, grp in enumerate(groups):
            acc[k] += _dual_rho(v, [idx[q] for q in grp], 2 * m)
    return [a / int(spec["trajectories"]) for a in acc], peak


def dual_fidelity(rho, kind):
    """Bell: <Phi+|rho|Phi+> = (1 + <XX> - <YY> + <ZZ>) / 4 (the paper's Eq. 4). GHZ: (P0..0 + P1..1 + A) / 2 with A =
    2 |rho_{0..0,1..1}|, the parity oscillation's amplitude at nu = N (the paper's estimator, our Lemma 4)."""
    if kind == "bell":
        return float(np.real(rho[0, 0] + rho[3, 3] + rho[0, 3] + rho[3, 0]) / 2)
    return float(np.real(rho[0, 0] + rho[-1, -1]) / 2 + abs(rho[0, -1]))


DUAL_MODELS = dict(hardcut=dict(engine="bridge", noise="none", mb=0),
                   ideal=dict(engine="bridge", noise="none", mb=None),
                   budget=dict(engine="bridge", noise="budget", mb=None),
                   measured=dict(engine="bridge", noise="measured", mb=None),
                   exact=dict(engine="exact", noise="none", mb=None))


def dual_units(R1, k=3):
    """k-qubit GHZ chains over the dual Nighthawk R1: every row and column of each module cut into chunks of k from
    every offset (rooted in the middle, CX outwards), and the chains through each L-coupler: the link's lattice pair
    (a, b) with every lattice neighbour of a, or of b, as the third qubit (rooted at the link's end that holds two)."""
    units = []
    for m in (0, 1):
        for ori in ("rows", "cols"):
            for line in range(12 if ori == "rows" else 10):
                qs = ([120 * m + 10 * line + c for c in range(10)] if ori == "rows" else
                      [120 * m + 10 * r + line for r in range(12)])
                for f in range(k):
                    for i in range(f, len(qs) - k + 1, k):
                        ch = qs[i:i + k]
                        mid = (k - 1) // 2
                        edges = [[ch[j + 1], ch[j]] for j in range(mid - 1, -1, -1)] + \
                                [[ch[j - 1], ch[j]] for j in range(mid + 1, k)]
                        units.append(dict(qubits=ch, root=ch[mid], edges=edges, link=None,
                                          tiling=f"{'AB'[m]} {ori}{f}"))
    for L in DUAL_LINKS:
        for end, other in ((L["a"], L["b"]), (L["b"], L["a"])):
            for nb in sorted(q for q in R1.adj[end] if q < 240):
                units.append(dict(qubits=[nb, end, other], root=end, edges=[[end, nb], [end, other]],
                                  link=L["name"], tiling=f"link {L['name']}"))
    return units


def dual_unit_ops(u):
    links = {frozenset((L["a"], L["b"])) for L in DUAL_LINKS}
    return [("h", u["root"])] + [("dcx" if frozenset((p_, c_)) in links else "cx", p_, c_) for p_, c_ in u["edges"]]


def dual_cell(q):
    """(row, column) of q on the drawing: module A as it is (rows 0-11), its L-qubits in row 12 under their lattice
    qubit, module B rotated by 180 degrees below (its row 11 facing A's, rows 15-26), its L-qubits in row 14."""
    if q >= 240:
        L = next(L_ for L_ in DUAL_LINKS if q in (L_["la"], L_["lb"]))
        return (12, L["a"] % 10) if q == L["la"] else (14, 9 - (L["b"] - 120) % 10)
    if q < 120:
        return (q // 10, q % 10)
    r, c = divmod(q - 120, 10)
    return (15 + 11 - r, 9 - c)


def cmd_dualmap(a):
    """GHZ fidelity over both modules of the dual Nighthawk R1 and their four L-couplers: 3-qubit chains tiled over each
    lattice plus every chain through a link, each its own circuit on the two-patch bridge, under the hard cut, the
    constituent budget and the measured distant CZ; heatmaps of the lowest F_GHZ per site with the cables drawn."""
    R1 = DualR1()
    qrack_bridge(a.qrack_bridge)
    units = dual_units(R1, 3)
    models = [x.strip() for x in a.models.split(",") if x.strip()]
    calls, keys = [], []
    for mo in models:
        M = DUAL_MODELS[mo]
        noisy = M["noise"] != "none"
        for j0 in range(0, len(units), a.chunk):
            for j, u in enumerate(units[j0:j0 + a.chunk], j0):
                K = (a.trajectories_link if u["link"] else a.trajectories) if noisy else 1
                calls.append((_dual_task, (dict(ops=dual_unit_ops(u), groups=[u["qubits"]], engine=M["engine"],
                                                noise=M["noise"], max_bonds=(a.bridge_max_bonds if M["mb"] is None
                                                                             else M["mb"]),
                                                trajectories=K, seed=[a.seed, j, zlib_key(mo)], r_cz=a.local_cz_error,
                                                r_1q=a.lattice_1q_error, qrack_bridge=a.qrack_bridge),)))
                keys.append((j, mo))
    print(f"{R1.describe()}; {len(units)} GHZ chains ({sum(1 for u in units if u['link'])} through a link); models "
          f"{', '.join(models)}; trajectories {a.trajectories} (lattice), {a.trajectories_link} (link); local CZ "
          f"error {a.local_cz_error}; {nq.qrack_versions()}")
    pool, workers = nq.shor_pool(a.jobs, len(calls))
    t0 = time.time()
    try:
        res = nq.shor_pool_map(pool, calls, [f"dualmap unit {k[0]} {k[1]}" for k in keys], float(a.progress or 0),
                               what="ghz dualmap")
    finally:
        if pool is not None:
            pool.shutdown()
    secs = time.time() - t0
    F = {k: dual_fidelity(r_[0][0], "ghz") for k, r_ in zip(keys, res)}
    peak = {k: r_[1] for k, r_ in zip(keys, res)}
    when = time.strftime("%Y-%m-%dT%H:%M:%S")
    recs = [dict(test="dual_map", when=when, unit=j, **u, F={mo: F[(j, mo)] for mo in models},
                 peak_bonds={mo: peak[(j, mo)] for mo in models}, local_cz_error=a.local_cz_error,
                 trajectories=(a.trajectories_link if u["link"] else a.trajectories), qrack=nq.qrack_versions())
            for j, u in enumerate(units)]
    print(f"\n{'chains':<34}" + "".join(f"{mo:>11}" for mo in models))
    groups = [("module A", lambda u: not u["link"] and u["qubits"][0] < 120),
              ("module B", lambda u: not u["link"] and u["qubits"][0] >= 120)] + \
             [(f"through {L['name']} ({L['a']}-{L['b']}, CZ {100 * L['dcz']:.2f} %)",
               lambda u, n_=L["name"]: u["link"] == n_) for L in DUAL_LINKS]
    for lab, fn in groups:
        rs = [r for r in recs if fn(r)]
        print(f"{lab:<34}" + "".join(f"{np.mean([r['F'][mo] for r in rs]):>11.3f}" for mo in models)
              + f"   ({len(rs)})")
    print(f"seconds: {secs:.1f}")
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(json.dumps(dict(test="dual_map_layout", when=when, models=models, links=DUAL_LINKS,
                                     sites=[[q, *dual_cell(q)] for q in range(R1.n)], cables=R1.cables)) + "\n")
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        print(f"  appended {len(recs)} records to {a.out}")
    if a.png:
        dualmap_plot(recs, models, a)


def dualmap_plot(recs, models, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle
    import textwrap
    cmap = LinearSegmentedColormap.from_list("div", [(0.0, "#a32c2b"), (0.25, "#e98584"), (0.5, "#e4e2dc"),
                                                     (0.75, "#6da7ec"), (1.0, "#184f95")])
    norm = Normalize(0.0, 1.0)
    ink, ink2, muted, surf, lq = "#1d1d1b", "#4a4944", "#6b6a64", "#fcfcfb", "#d6d4cd"
    names = dict(hardcut=("Hard cut", "the bridge at max_bonds 0: no entanglement through the cables"),
                 ideal=("Bridge, exact", "no noise"),
                 budget=("Constituent budget", "transfers, local CZs, L-qubit gates at their benchmarks"),
                 measured=("Measured distant CZ", "each link's benchmarked distant CZ as one channel"),
                 exact=("QrackSimulator", "no noise"))
    cell, X0, GAP = 0.40, 0.45, 0.75
    ncol, nrow = 10, 27
    PW, PH = cell * ncol, cell * nrow
    lines = textwrap.wrap(
        "Two IBM Nighthawk R1 modules (120 qubits each) joined by the four L-couplers of arXiv:2610.29986, each module "
        "one patch of a QrackBridge. Module B is drawn rotated by 180°, its row 11 facing module A's; grey cells are "
        "the L-qubits, lines the 80 cm cables (labelled with the link's benchmarked distant CZ). Each cell is the "
        "lowest F_GHZ among the 3-qubit GHZ chains through that qubit: chains tiled along every row and column of "
        "each module, and every chain through an L-coupler (the link's lattice pair plus a neighbour of either end), "
        "each its own circuit; the distant CZ built as the paper builds it. Noise as Pauli trajectories (local CZ "
        f"error {a.local_cz_error}); no readout error.", 168)
    FW = X0 + len(models) * (PW + GAP) + 0.9
    Hh = 0.95 + 0.21 * len(lines) + 0.35
    FH = Hh + 0.55 + PH + 0.6 + 0.6
    fig = plt.figure(figsize=(FW, FH), dpi=200, facecolor=surf)
    links = {L["name"]: L for L in DUAL_LINKS}
    for li, mo in enumerate(models):
        vals = collections.defaultdict(list)
        for r in recs:
            for q in r["qubits"]:
                vals[q].append(r["F"][mo])
        x0 = X0 + li * (PW + GAP)
        ax = fig.add_axes([x0 / FW, 1.2 / FH, PW / FW, PH / FH], facecolor=surf)
        for q in range(248):
            r_, c_ = dual_cell(q)
            if q >= 240:
                ax.add_patch(Rectangle((c_ - 0.42, r_ - 0.42), 0.84, 0.84, facecolor=lq, edgecolor=surf, lw=1.0))
                ax.text(c_, r_, f"L{q}", ha="center", va="center", fontsize=4.8, color=ink2)
                continue
            Fq = min(vals[q])
            ax.add_patch(Rectangle((c_ - 0.5, r_ - 0.5), 1, 1, facecolor=cmap(norm(Fq)), edgecolor=surf, lw=0.9))
            dark = Fq > 0.82 or Fq < 0.12
            ax.text(c_, r_ + 0.10, f"{Fq:.2f}", ha="center", va="center", fontsize=5.6, fontweight="bold",
                    color="#ffffff" if dark else ink)
            ax.text(c_ - 0.44, r_ - 0.30, str(q), ha="left", va="center", fontsize=3.4, color="#ffffff" if dark
                    else muted)
        for L in DUAL_LINKS:
            (ra, ca), (rb, cb) = dual_cell(L["la"]), dual_cell(L["lb"])
            ax.plot([ca, cb], [ra + 0.42, rb - 0.42], color=ink, lw=2.2, solid_capstyle="round", zorder=3)
            ax.text((ca + cb) / 2 + 0.18, 13.0, f"{100 * L['dcz']:.1f}", fontsize=5.2, color=ink2, va="center",
                    ha="left")
        ax.text(-0.55, 5.5, "module A", rotation=90, ha="right", va="center", fontsize=9, fontweight="bold", color=ink)
        ax.text(-0.55, 20.5, "module B", rotation=90, ha="right", va="center", fontsize=9, fontweight="bold", color=ink)
        ax.set_xlim(-1.3, 9.6)
        ax.set_ylim(26.6, -0.6)
        ax.set_aspect("equal")
        ax.axis("off")
        nm, sb = names[mo]
        fig.text(x0 / FW, (1.2 + PH + 0.42) / FH, nm, fontsize=13, fontweight="bold", color=ink)
        fig.text(x0 / FW, (1.2 + PH + 0.18) / FH, sb, fontsize=8.5, color=ink2)
        lat = [r["F"][mo] for r in recs if not r["link"]]
        lk = {n_: np.mean([r["F"][mo] for r in recs if r["link"] == n_]) for n_ in links}
        fig.text(x0 / FW, 0.95 / FH, f"lattice chains {np.mean(lat):.3f}", fontsize=8.5, color=ink,
                 family="monospace")
        fig.text(x0 / FW, 0.72 / FH, "through links " + "  ".join(f"{v:.2f}" for v in lk.values()), fontsize=8.5,
                 color=ink, family="monospace")
        fig.text(x0 / FW, 0.52 / FH, "(" + ", ".join(lk) + ")", fontsize=6.8, color=muted, family="monospace")
    cax = fig.add_axes([(FW - 0.55) / FW, (1.2 + 0.3 * PH) / FH, 0.12 / FW, 0.4 * PH / FH])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0", "0.25", "½", "0.75", "1"])
    cb.ax.tick_params(labelsize=8, colors=ink2)
    cb.outline.set_visible(False)
    fig.text((FW - 0.75) / FW, (1.2 + 0.73 * PH) / FH, "F_GHZ", fontsize=9.5, fontweight="bold", color=ink)
    fig.text(0.02 * 17 / FW, (FH - 0.45) / FH, "Dual Nighthawk R1: GHZ fidelity over both modules and their L-couplers",
             fontsize=16, fontweight="bold", color=ink)
    for i, line in enumerate(lines):
        fig.text(0.02 * 17 / FW, (FH - 0.95 - 0.21 * i) / FH, line, fontsize=8.6, color=ink2)
    fig.text(0.02 * 17 / FW, 0.22 / FH, f"{recs[0]['when']} · {recs[0]['qrack']} · nighthawk_ghz.py dualmap --models "
             f"{','.join(models)}", fontsize=7.5, color=muted)
    fig.savefig(a.png, facecolor=surf)
    plt.close(fig)
    print(f"  plot: {a.png}")


def cmd_dual(a):
    """The paper's two-module experiment on the dual Nighthawk R1: every circuit under each model, against Table 1."""
    R1 = DualR1()
    qrack_bridge(a.qrack_bridge)
    states = dual_states()
    models = [x.strip() for x in a.models.split(",") if x.strip()]
    for x in models:
        if x not in DUAL_MODELS:
            raise SystemExit(f"ghz dual: unknown model {x!r}; known: {', '.join(DUAL_MODELS)}")
    K, batch = int(a.trajectories), int(a.batch)
    calls, keys = [], []
    for name in states:
        for mo in models:
            M = DUAL_MODELS[mo]
            noisy = M["noise"] != "none"
            nb = max(1, K // batch) if noisy else 1
            for j in range(nb):
                calls.append((_dual_task, (dict(state=name, engine=M["engine"], noise=M["noise"],
                                                max_bonds=(a.bridge_max_bonds if M["mb"] is None else M["mb"]),
                                                trajectories=(batch if noisy else 1), seed=[a.seed, j, zlib_key(name),
                                                                                         zlib_key(mo)],
                                                r_cz=a.local_cz_error, r_1q=a.lattice_1q_error,
                                                qrack_bridge=a.qrack_bridge),)))
                keys.append((name, mo))
    print(f"{R1.describe()}; {DUAL_PAPER}")
    print(f"models {', '.join(models)}; noisy models {K} trajectories in batches of {batch}; local CZ error "
          f"{a.local_cz_error}, lattice 1q error {a.lattice_1q_error}, max_bonds {a.bridge_max_bonds}; "
          f"{nq.qrack_versions()}")
    pool, workers = nq.shor_pool(a.jobs, len(calls))
    t0 = time.time()
    try:
        res = nq.shor_pool_map(pool, calls, [f"dual {k[0]} {k[1]}" for k in keys], float(a.progress or 0),
                               what="ghz dual")
    finally:
        if pool is not None:
            pool.shutdown()
    secs = time.time() - t0
    acc = collections.defaultdict(list)
    peaks = collections.defaultdict(int)
    for k, (rhos, pk) in zip(keys, res):
        acc[k].append(rhos)
        peaks[k] = max(peaks[k], pk)
    rows = []
    for name, (ops, groups, kind) in states.items():
        for gi, grp in enumerate(groups):
            label = name if len(groups) == 1 else f"simultaneous: bell {grp[0]}-{grp[1]}"
            row = dict(state=label, kind=kind, qubits=grp, circuit=name, ops=[list(o) for o in ops])
            for mo in models:
                batches = [r[gi] for r in acc[(name, mo)]]
                F = dual_fidelity(sum(batches) / len(batches), kind)
                fb = [dual_fidelity(b, kind) for b in batches]
                row[mo] = dict(F=F, se=(float(np.std(fb, ddof=1) / math.sqrt(len(fb))) if len(fb) > 1 else 0.0),
                               peak_bonds=peaks[(name, mo)])
            if label in DUAL_MEASURED:
                (u, us), (mm, ms) = DUAL_MEASURED[label]
                row["ibm"] = dict(unmitigated=u / 100, unmitigated_se=us / 100, mitigated=mm / 100,
                                  mitigated_se=ms / 100)
            rows.append(row)
    hdr = f"{'state':<30}" + "".join(f"{mo:>16}" for mo in models) + f"{'IBM mitigated':>15}{'IBM raw':>10}"
    print("\n" + hdr)
    for r in rows:
        line = f"{r['state']:<30}"
        for mo in models:
            d = r[mo]
            line += f"{d['F']:>9.3f}" + (f" ±{d['se']:.3f}" if d["se"] else "       ")
        if "ibm" in r:
            line += f"{r['ibm']['mitigated']:>15.4f}{r['ibm']['unmitigated']:>10.4f}"
        print(line)
    print("bond qubits per cut (peak): " + ", ".join(
        f"{r['state']} {max(r[mo]['peak_bonds'] for mo in models)}" for r in rows))
    print(f"seconds: {secs:.1f}")
    when = time.strftime("%Y-%m-%dT%H:%M:%S")
    if a.out:
        with open(a.out, "a") as fh:
            for r in rows:
                fh.write(json.dumps(dict(test="dual_r1", when=when, paper=DUAL_PAPER, links=DUAL_LINKS,
                                         local_cz_error=a.local_cz_error, lattice_1q_error=a.lattice_1q_error,
                                         trajectories=K, max_bonds=a.bridge_max_bonds, qrack=nq.qrack_versions(),
                                         **r)) + "\n")
        print(f"  appended {len(rows)} records to {a.out}")
    if a.png:
        dual_plot(rows, models, a)


def cmd_dualplot(a):
    """dual's plot from its records (the latest run in --in)."""
    recs = [r for r in read_records(a.inp) if r.get("test") == "dual_r1"]
    if not recs:
        raise SystemExit(f"ghz dualplot: no dual_r1 record in {a.inp}")
    when = max(r["when"] for r in recs)
    rows = [r for r in recs if r["when"] == when]
    models = [m_ for m_ in DUAL_MODELS if m_ in rows[0]]
    a.local_cz_error, a.trajectories = rows[0]["local_cz_error"], rows[0]["trajectories"]
    dual_plot(rows, models, a)


def dual_plot(rows, models, a):
    """Model against measurement, one row per state: a dot per model, IBM's mitigated and raw values beside them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, muted, surf, grid = "#1d1d1b", "#4a4944", "#6b6a64", "#fcfcfb", "#e4e2dc"
    style = dict(hardcut=("hard cut (bridge, max_bonds 0)", "#8a8984", "o", 7),
                 ideal=("bridge, exact", "#b9b8b2", "o", 7),
                 exact=("QrackSimulator, exact", "#b9b8b2", "s", 6),
                 budget=("bridge + constituent budget", "#eb6834", "o", 8),
                 measured=("bridge + measured distant CZ", "#2a78d6", "o", 8))
    import textwrap
    if "ideal" in models and "exact" in models and all(abs(r["ideal"]["F"] - r["exact"]["F"]) < 1e-6 for r in rows):
        models = [m_ for m_ in models if m_ != "exact"]          # the same dot: say so in the legend
        style["ideal"] = ("bridge, exact (= QrackSimulator)",) + style["ideal"][1:]
    n = len(rows)
    fig = plt.figure(figsize=(11.5, 1.8 + 0.52 * n + 1.2), dpi=200, facecolor=surf)
    ax = fig.add_axes([0.30, 1.2 / (1.8 + 0.52 * n + 1.2), 0.66, 0.52 * n / (1.8 + 0.52 * n + 1.2)], facecolor=surf)
    for i, r in enumerate(rows):
        y = n - 1 - i
        ax.axhline(y, color=grid, lw=0.8, zorder=0)
        for mo in models:
            lab, col, mk, ms = style[mo]
            d = r[mo]
            ax.errorbar(d["F"], y, xerr=d["se"] or None, fmt=mk, color=col, ms=ms, mec=surf, mew=1.2, zorder=3,
                        label=lab if i == 0 else None)
        if "ibm" in r:
            ib = r["ibm"]
            ax.errorbar(ib["mitigated"], y, xerr=ib["mitigated_se"], fmt="D", color=ink, ms=7, mec=surf, mew=1.2,
                        zorder=4, label="IBM, readout-mitigated" if not any("ibm" in q for q in rows[:i]) else None)
            ax.errorbar(ib["unmitigated"], y, xerr=ib["unmitigated_se"], fmt="D", mfc=surf, color=ink, ms=7, mew=1.4,
                        zorder=4, label="IBM, raw" if not any("ibm" in q for q in rows[:i]) else None)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["state"] for r in rows][::-1], fontsize=10, color=ink)
    ax.axvline(0.5, color=muted, lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax.text(0.505, n - 0.45, "½: separable bound", fontsize=8.5, color=muted, va="bottom")
    ax.set_xlim(0.45, 1.01)
    ax.set_ylim(-0.6, n - 0.2)
    ax.set_xlabel("state fidelity F", fontsize=10, color=ink2)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color(muted)
    ax.tick_params(axis="x", colors=ink2, labelsize=9)
    ax.tick_params(axis="y", length=0)
    H = 1.8 + 0.52 * n + 1.2
    fig.text(0.02, 1 - 0.45 / H, "Dual Nighthawk R1: the paper's states against the bridge's models", fontsize=15,
             fontweight="bold", color=ink)
    sub = textwrap.wrap(
        "Two 120-qubit modules joined by four L-couplers (arXiv:2610.29986). Distant CZ = CNOT into the L-qubit, "
        "transfer, CZ, transfer back, CNOT, as in the paper; noise as Pauli trajectories from its benchmarks (local CZ "
        f"error {a.local_cz_error}, {rows[0]['trajectories'] if 'trajectories' in rows[0] else a.trajectories} "
        "trajectories). Models carry no readout error: compare them with IBM's readout-mitigated values. Error bars: "
        "trajectory sampling (models), the paper's uncertainties (IBM).", 150)
    for i, line in enumerate(sub):
        fig.text(0.02, 1 - (0.80 + 0.24 * i) / H, line, fontsize=9, color=ink2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.10), ncol=4, frameon=False, fontsize=8.5,
              handletextpad=0.3, columnspacing=1.2)
    fig.savefig(a.png, facecolor=surf)
    plt.close(fig)
    print(f"  plot: {a.png}")


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
    loom = mp.get("loom")
    if loom and loom.get("box"):
        return stackmap_box(a, st, layers, mp, order, fields, ghz, when)
    if loom:                                              # a unit whose qubits sit on two chips: its own class
        ends = list(np.cumsum(loom["chip_n"]))
        chip_of = lambda q: next(i for i, e in enumerate(ends) if q < e)        # noqa: E731
    classes = ("bulk", "replica", "cut") + (("loom",) if loom else ())
    labels = dict(bulk="off seam", replica="replica", cut="cut",
                  loom="across bridge" if (loom or {}).get("read") == "bridge" else "across loom")

    def cls_of(r):
        if loom and len({chip_of(q) for q in r["qubits"]}) > 1:
            return "loom"
        return r["seam_class"]
    counts = collections.Counter(cls_of(r) for r in st)

    def cls_mean(c, i):
        v = [(r.get(fields[i]) or (r["pass1"] if order == "code-echo" and i == 1 else None)) for r in st
             if cls_of(r) == c]
        v = [d["F_GHZ"] for d in v if d]
        return float(np.mean(v)) if v else None
    sites = {s_[0]: (s_[1], s_[2]) for s_ in mp["sites"]}
    seam = set(mp.get("seam", []))
    cmap = LinearSegmentedColormap.from_list("div", [(0.0, "#a32c2b"), (0.25, "#e98584"), (0.5, "#e4e2dc"),
                                                     (0.75, "#6da7ec"), (1.0, "#184f95")])
    norm = Normalize(0.0, 1.0)
    ink, ink2, muted, surf = "#1d1d1b", "#4a4944", "#6b6a64", "#fcfcfb"
    names = (["Pass 1", "Pass 2", "V"] if order == "echo-code" else ["Pass 1", "C", "E"])
    ace_chips = bool(loom) and loom.get("backend") == "ace"
    ghost = bool(loom) and loom.get("read", "ghost") == "ghost"         # records before --loom-read: ghost
    bridged = bool(loom) and loom.get("read") == "bridge"
    bp = (loom or {}).get("bridge") or {}
    if not loom:
        code_name = "Layer S repetition code"
    elif bridged:
        code_name = "Layer S: nothing to correct (the bridge is exact up to truncation)"
    elif ghost:
        code_name = ("Layer S: ACE's code + reading inside a ghost band" if ace_chips else
                     "Layer S: reading inside a ghost band")
    else:
        code_name = ("Layer S: ACE's code inside each chip" if ace_chips else
                     "Layer S: nothing to correct across a hard cut")
    subs = (["Layer 1a, no drift", "Layer 2, anti-OTOC Walsh echo", "echo + " + code_name]
            if order == "echo-code" else
            ["Layer 1a, no drift", code_name, "code + Layer 2, anti-OTOC Walsh echo"])
    l2 = next((r["layer2"] for r in ghz if r["stack_pass"] != 1), {})
    rows_ = sorted({rc[0] for q, rc in sites.items() if q in layers[0]})
    cols_ = sorted({rc[1] for q, rc in sites.items() if q in layers[0]})
    k = st[0].get("unit_size", 3)
    prot = l2.get("inside_zz") == "stabilizer"
    import textwrap
    if loom:
        kind = "two-patch ACE chips" if ace_chips else "exact chips"
        reg_label = (f"{'QrackBridge' if bridged else 'loom'}, {len(loom['chips'])} {kind} of "
                     f"{' + '.join(str(x) for x in loom['chip_n'])} qubits"
                     + (f", max_bonds {bp.get('max_bonds')}" if bridged else ""))
        ring = (("Ringed cells are ACE seam qubits and link anchors (a ghost copy on the other chip)" if ace_chips else
                 "Ringed cells are link anchors (a ghost copy on the other chip)") if ghost else
                ("Ringed cells are ACE seam qubits" if ace_chips else "No cell has a copy on another chip"))
        links_word = "bridged" if bridged else "loom's"
        lines = textwrap.wrap(f"Each cell is the lowest F_GHZ among the {k}-qubit GHZ chains through that qubit (small "
                              f"number: qubit index). {ring}; bars are the {links_word} links. Below ½: no coherence between "
                              f"simulators.", 205)
        chips_txt = (f"{' and '.join(loom['chips'])}"
                     + (", each on its register's two-patch ACE preset," if ace_chips else ", each an exact simulator,"))
        if bridged:
            lines += textwrap.wrap(
                f"QrackBridge between {chips_txt[:-1]}, one patch per chip: a gate across the cut is teleported "
                f"through a pair of bond qubits (a virtual Bell pair), then the cut is recompressed to at most "
                f"{bp.get('max_bonds')} bond qubits ({bp.get('truncation', 'greedy')}). Entanglement crosses up to that "
                f"capacity: 2^max_bonds Schmidt values per cut; 0 is a hard cut. Each circuit's bridge holds the qubits "
                f"its units touch (idle qubits are |0> and in product).", 205)
        elif ghost:
            lines += textwrap.wrap(
                f"Ghost loom between {chips_txt[:-1]}: each chip also simulates its partners' anchor qubits, every gate "
                f"replayed on them, a replica band two columns wide. A unit inside one chip's band is simulated there "
                f"whole and read there (Layer S here" + (", with ACE's repetition code inside each chip" if ace_chips
                                                          else "") +
                "): not a correction across the loom, which nothing crosses; units wider than the band stay at or "
                "below ½ (5-qubit chains, RESULTS §10.14).", 205)
        else:
            lines += textwrap.wrap(
                f"The loom between {chips_txt} is a hard cut: no ghosts, a link CZ reaches each side only as the "
                f"mean-field phase of the other, every qubit read on its own chip. No entanglement crosses, so a unit "
                f"across the loom is separable there: F_GHZ ≤ ½ whatever is done to it. Layer S's code: "
                + ("ACE's boundary repetition code inside each chip; nothing across the loom." if ace_chips else
                   "none (exact chips have no seam inside; nothing corrects across the loom)."), 205)
    else:
        reg_label = reg
        lines = [f"Each cell is the lowest F_GHZ among the {k}-qubit GHZ chains through that qubit (small number: qubit "
                 f"index). Ringed cells are ACE seam qubits. Below ½: no coherence between simulators."]
    if prot:
        lines.append("Seam protection: the drift's ZZ inside a GHZ unit is not put through the "
                     + ("seams" if loom else "ACE seam") + " (a global phase on the GHZ, Corollary 2.1; --inside-zz "
                     "stabilizer).")
    # geometry: three panels side by side (up to 10 columns), or stacked at full width (wider fields)
    ncol, nrow = cols_[-1] - cols_[0] + 1, rows_[-1] - rows_[0] + 1
    order_txt = (("code first, echo second" if order == "code-echo" else "echo first, code second")
                 + ("  ·  with seam protection" if prot else "  ·  without seam protection"))
    wide = ncol > 10
    two_line = wide or len(f"{reg_label}: GHZ fidelity per measurement layer  ·  {order_txt}") > 100
    if not wide:
        FW = 17.0
        W, X0, GAP, AH = 0.265, 0.035, 0.035, 5.1            # panel width (fraction), axes height (inches)
        Hh = 0.62 + 0.19 * len(lines) + 0.25 + (0.35 if two_line else 0.0)
        FH = Hh + 0.45 + AH + 0.45 + 1.0
        bot = (FH - Hh - 0.45 - AH) / FH
        panels = [dict(ax=[X0 + li * (W + GAP), bot, W, AH / FH], name=(X0 + li * (W + GAP), (FH - Hh - 0.2) / FH),
                       means=(X0 + li * (W + GAP), bot - 0.2 / FH)) for li in range(3)]
        cbar, cbar_lab = [0.945, bot + 0.1 * AH / FH, 0.010, 0.8 * AH / FH], (0.935, bot + 0.93 * AH / FH)
        title_y, line_y, line_dy = (FH - 0.37) / FH, (FH - 0.62 - (0.35 if two_line else 0.0)) / FH, 0.19 / FH
        foot = (0.82 / FH, 0.53 / FH, 0.34 / FH)
    else:
        FW = 17.0
        cell = min(0.80 * FW / ncol, 0.95)
        ph, Hh, Hf = cell * nrow, 1.15 + 0.32 * len(lines), 1.3
        block = 0.55 + ph + 0.55
        FH = Hh + 3 * block + Hf
        panels = []
        for li in range(3):
            top = FH - Hh - li * block
            panels.append(dict(ax=[0.6 / FW, (top - 0.55 - ph) / FH, cell * ncol / FW, ph / FH],
                               name=(0.6 / FW, (top - 0.38) / FH), means=(0.6 / FW, (top - 0.55 - ph - 0.40) / FH)))
        mid = panels[1]["ax"]
        cbar, cbar_lab = [(0.6 + cell * ncol + 0.5) / FW, mid[1], 0.16 / FW, mid[3]], \
            ((0.6 + cell * ncol + 0.4) / FW, mid[1] + mid[3] + 0.25 / FH)
        title_y, line_y, line_dy = (FH - 0.55) / FH, (FH - 1.35) / FH, 0.32 / FH
        foot = (1.0 / FH, 0.65 / FH, 0.40 / FH)
    fig = plt.figure(figsize=(FW, FH), dpi=200 if FH < 12 else 160, facecolor=surf)
    for li, vals in enumerate(layers):
        P = panels[li]
        ax = fig.add_axes(P["ax"], facecolor=surf)
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
        x, y = P["name"]
        fig.text(x, y, names[li], fontsize=15, fontweight="bold", color=ink, ha="left")
        fig.text(x + (0.048 if names[li].startswith("Pass") else 0.020) * (17.0 / FW), y + 0.001, subs[li], fontsize=11,
                 color=ink2, ha="left")
        fig.text(*P["means"], "class means   " + "   ".join(
            f"{labels[c_]} {v:.2f}" if v is not None else f"{labels[c_]} not rerun"
            for c_, v in ((c_, cls_mean(c_, li)) for c_ in classes if counts.get(c_, 0))), fontsize=10.5, color=ink,
            ha="left", family="monospace")
    cax = fig.add_axes(cbar)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0", "0.25", "½", "0.75", "1"])
    cb.ax.tick_params(labelsize=9.5, colors=ink2)
    cb.outline.set_visible(False)
    fig.text(*cbar_lab, "F_GHZ", fontsize=11, fontweight="bold", color=ink)
    if two_line:
        fig.text(0.02, title_y, f"{reg_label}: GHZ fidelity per measurement layer", fontsize=19, fontweight="bold",
                 color=ink, ha="left")
        fig.text(0.02, title_y - (0.40 if wide else 0.33) / FH, order_txt, fontsize=15, fontweight="bold", color=ink2,
                 ha="left")
    else:
        fig.text(0.02, title_y, f"{reg_label}: GHZ fidelity per measurement layer  ·  {order_txt}", fontsize=19,
                 fontweight="bold", color=ink, ha="left")
    for i, line in enumerate(lines):
        fig.text(0.02, line_y - line_dy * i, line, fontsize=10.5, color=ink2, ha="left")
    host = "simulator" if (ace_chips or not loom) else "chip"
    via = ("replica or ghost" if (ace_chips and ghost) else "ghost" if ghost else "replica")
    desc = dict(bulk=f"off seam {counts.get('bulk', 0)}",
                replica=f"replica {counts.get('replica', 0)} (one {host} holds every qubit, a seam qubit through its {via})",
                cut=f"cut {counts.get('cut', 0)} (no {host} holds them all)",
                loom=f"{labels['loom']} {counts.get('loom', 0)} (qubits on two chips"
                     + ("; read inside a ghost band" if ghost else "; bridged" if bridged else
                        "; separable across the hard cut") + ")")
    fig.text(0.02, foot[0], f"Class means over {k}-qubit chains: " + ", ".join(desc[c_] for c_ in classes if counts.get(c_, 0))
             + "." + (" The code runs on the replica units Layer S plans it for; cut units are the control."
                      if counts.get("replica") or counts.get("cut") else ""), fontsize=9.5, color=ink2, ha="left")
    zz = ", inside ZZ by stabiliser" if l2.get("inside_zz") == "stabilizer" else ""
    fig.text(0.02, foot[1], f"{st[0]['shots']} shots × {st[0]['repeats']} realisations per setting · Layer 2 "
             f"{l2.get('mode', 'none')}, L = {l2.get('blocks')}, T = {l2.get('t_idle')}, J = {l2.get('j_zz')} × "
             f"U(0.75, 1.25), σ = {l2.get('sigma_qs')}{zz} · "
             + ("V reruns the seam units on fresh draws; others keep pass 2" if order == "echo-code" else
                "C reruns the seam units on fresh backends (others keep pass 1); E runs every unit"),
             fontsize=8.5, color=muted, ha="left")
    fig.text(0.02, foot[2], f"{os.path.basename(a.inp)} · run {when} · {st[0].get('qrack', '')} · nighthawk_ghz.py stack "
             f"--order {order}" + (" --engine loom" if loom and not bridged else "")
             + (" --loom-chip-backend ace" if ace_chips else "")
             + ((f" --engine bridge --bridge-max-bonds {bp.get('max_bonds')}" if bridged else
                 " --loom-read " + ("ghost" if ghost else "cut")) if loom else ""),
             fontsize=8.5, color=muted, ha="left")
    png = a.png or f"ghz_stack_{reg}_{order}.png"
    fig.savefig(png, facecolor=surf)
    plt.close(fig)
    print(f"stackmap: {reg}, order {order}, run {when}: {png}")
    for c in classes:
        print(f"  {c:<8} {counts.get(c, 0):>3} units  " + "  ".join(fmt(cls_mean(c, i)) for i in range(3)))


def box_net_decor(R, loom, sites):
    """The box net's decoration for a heatmap: a bar on every coupler across a box edge (on the cell side facing
    the other face; relayed edges flagged), a letter on each edge the net cuts open (on both faces), and each
    face's name where it clears every cell and letter."""
    ring = loom["box"]["ring"]
    hops = {tuple(sorted(e["faces"], key=ring.index)): e for e in loom["box"]["edges"]}
    # every coupler across a box edge: one bar per end, on the cell side facing the other face
    letters, bars = {}, []
    for u_, v_ in loom["links"]:
        fu, fv = R.face[u_], R.face[v_]
        key = tuple(sorted((fu, fv), key=ring.index))
        relayed = hops[key]["ring_distance"] > 1
        rc_u, rc_v = sites[u_], sites[v_]
        folded = abs(rc_u[0] - rc_v[0]) + abs(rc_u[1] - rc_v[1]) == 1
        if not folded and key not in letters:
            letters[key] = "abcdefghijklmnop"[len(letters)]
        for q, f_other in ((u_, fv), (v_, fu)):
            p = R.xyz[q]
            nrm = BOX_NORMAL[f_other]
            r0, c0 = R.net(p, R.face[q])
            r1, c1 = R.net(tuple(p[i] + nrm[i] for i in range(3)), R.face[q])
            bars.append((r0, c0, r1 - r0, c1 - c0, relayed, None if folded else key))
            if folded:
                break
    rows_ = sorted({rc[0] for rc in sites.values()})
    cols_ = sorted({rc[1] for rc in sites.values()})
    nrow, ncol = rows_[-1] - rows_[0] + 1, cols_[-1] - cols_[0] + 1
    at = {rc: q for q, rc in sites.items()}
    letter_at = []
    for key, ch in letters.items():
        for f_ in key:
            pts = [(r0 + dr * 0.8, c0 + dc * 0.8) for r0, c0, dr, dc, _, kk in bars
                   if kk == key and R.face[at[(r0, c0)]] == f_]
            if pts:
                letter_at.append(((float(np.mean([p_[0] for p_ in pts])), float(np.mean([p_[1] for p_ in pts]))), ch))
    name_at = {}                                         # each face's name where it clears every cell and letter
    for f_ in ring:
        qs = [q for q in range(R.n) if R.face[q] == f_]
        r_lo, r_hi = min(sites[q][0] for q in qs), max(sites[q][0] for q in qs)
        c_lo, c_hi = min(sites[q][1] for q in qs), max(sites[q][1] for q in qs)
        r_mid, c_mid = (r_lo + r_hi) / 2, (c_lo + c_hi) / 2
        hw = 0.09 * len(f_) + 0.05                       # half the label's width, in cells (9 pt bold, 0.44 in cells)
        obst = ([(rc[0], rc[1], 0.5, 0.5) for rc in sites.values()] + [(p_[0], p_[1], 0.2, 0.15) for p_, _ in letter_at]
                + [(v[3], v[4], 0.2, v[5]) for v in name_at.values()])
        best = fallback = None
        if f_ == "base":                                 # boxed in by the walls: the notch under its first column
            name_at[f_] = (r_hi + 0.9, c_lo, "center", r_hi + 0.9, c_lo, hw)
            continue
        for rr, cc in ((r_lo - 0.85, c_lo - 0.2 - hw), (r_lo - 0.85, c_hi + 0.2 + hw), (r_hi + 0.85, c_lo - 0.2 - hw),
                       (r_hi + 0.85, c_hi + 0.2 + hw), (r_mid, c_lo - 1.25 - hw), (r_mid, c_hi + 1.25 + hw),
                       (r_lo - 1.4, c_mid), (r_hi + 1.4, c_mid)):
            if not (rows_[0] - 1.3 <= rr <= rows_[-1] + 1.0 and cols_[0] - 0.5 <= cc - hw and cc + hw <= cols_[-1] + 0.5):
                continue
            fallback = fallback or (rr, cc, "center", rr, cc, hw)
            near = sorted(((rr - q_r) ** 2 + (cc - q_c) ** 2, R.face[at[(q_r, q_c)]]) for q_r, q_c in sites.values())
            if near[0][1] != f_ or any(d_ < near[0][0] + 0.6 and g_ != f_ for d_, g_ in near[1:]):
                continue                                 # nearer (or as near) another face: ambiguous
            clear = min(max(abs(rr - o[0]) - o[2] - 0.2, abs(cc - o[1]) - o[3] - hw) for o in obst)
            if best is None or clear > best[0] + 0.05:
                best = (clear, (rr, cc, "center", rr, cc, hw))
        name_at[f_] = best[1] if best else (fallback or (r_lo - 0.85, c_mid, "center", r_lo - 0.85, c_mid, hw))
    return dict(bars=bars, letters=letters, letter_at=letter_at, name_at=name_at, rows=rows_, cols=cols_,
                nrow=nrow, ncol=ncol)


def draw_box_net(ax, dec, ink, ink2, surf):
    for r0, c0, dr, dc, relayed, key in dec["bars"]:   # a surface halo under each bar keeps it legible on dark cells
        xs, ys = (([c0 - 0.40, c0 + 0.40], [r0 + dr / 2] * 2) if dr else ([c0 + dc / 2] * 2, [r0 - 0.40, r0 + 0.40]))
        ax.plot(xs, ys, color=surf, lw=5.0, solid_capstyle="butt")
        ax.plot(xs, ys, color=ink, lw=2.4, ls=(0, (2.0, 1.2)) if relayed else "solid", solid_capstyle="butt",
                dash_capstyle="butt")
    for (rr, cc), ch in dec["letter_at"]:              # one letter beside each cut-open side
        ax.text(cc, rr, ch, ha="center", va="center", fontsize=8.5, fontweight="bold", color=ink2, style="italic")
    for f_, (rr, cc, ha, _, _, _) in dec["name_at"].items():
        ax.text(cc, rr, f_, ha=ha, va="center", fontsize=9, fontweight="bold", color=ink)
    ax.set_xlim(dec["cols"][0] - 0.6, dec["cols"][-1] + 0.6)
    ax.set_ylim(dec["rows"][-1] + 0.6, dec["rows"][0] - 1.4)
    ax.set_aspect("equal")
    ax.axis("off")


def stackmap_box(a, st, layers, mp, order, fields, ghz, when):
    """cmd_stackmap for --box: the three GHZ measurements on the box's unfolded net, side by side. Bars mark the
    couplers across a box edge: solid where the edge's faces share a ring cut of the bridge, dashed where the
    bridge relays through the face between; an edge the net cuts open is drawn on both faces with one letter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle
    import textwrap
    loom, box = mp["loom"], mp["loom"]["box"]
    L, H = box["L"], box["H"]
    reg = st[0]["register"]
    R = box_register(L, H)
    bp = loom.get("bridge") or {}
    sites = {s_[0]: (s_[1], s_[2]) for s_ in mp["sites"]}
    ring = box["ring"]

    def cls_of(r):
        b = r.get("box") or R.unit_box(r)
        n_ = len(b["cuts"])
        return "face" if n_ == 0 else "direct" if n_ == 1 else "relayed"
    classes = ("face", "direct", "relayed")
    labels = dict(face="face", direct="direct", relayed="relayed")
    counts = collections.Counter(cls_of(r) for r in st)

    def mean_of(rs, i):
        v = []
        for r in rs:
            d = r.get(fields[i]) or (r.get(fields[i - 1]) if i else None) or r["pass1"]
            v.append(d["F_GHZ"])
        return float(np.mean(v)) if v else None
    cmap = LinearSegmentedColormap.from_list("div", [(0.0, "#a32c2b"), (0.25, "#e98584"), (0.5, "#e4e2dc"),
                                                     (0.75, "#6da7ec"), (1.0, "#184f95")])
    norm = Normalize(0.0, 1.0)
    ink, ink2, muted, surf = "#1d1d1b", "#4a4944", "#6b6a64", "#fcfcfb"
    names = (["Pass 1", "Pass 2", "V"] if order == "echo-code" else ["Pass 1", "C", "E"])
    code_name = "Layer S: nothing to correct (the bridge is exact up to truncation)"
    subs = (["Layer 1a, no drift", "Layer 2, anti-OTOC Walsh echo", "echo + " + code_name]
            if order == "echo-code" else ["Layer 1a, no drift", "Layer S: bridged units rerun (control)",
                                          "Layer 2, anti-OTOC Walsh echo"])
    l2 = next((r["layer2"] for r in ghz if r["stack_pass"] != 1), {})
    prot = l2.get("inside_zz") == "stabilizer"
    k = st[0].get("unit_size", 3)
    dec = box_net_decor(R, loom, sites)
    letters, nrow, ncol = dec["letters"], dec["nrow"], dec["ncol"]
    direct = [e for e in box["edges"] if e["ring_distance"] == 1]
    relay = [e for e in box["edges"] if e["ring_distance"] > 1]
    lines = textwrap.wrap(
        f"A closed {L} × {L} × {H} box: the {R.n} lattice points on its surface, couplers at unit distance, "
        f"the base, four walls and the lid each one circuit and one QrackBridge patch, the six patches on a ring "
        f"({' → '.join(ring)} → {ring[0]}). Opposite faces sit three apart on the ring, so every box edge joins "
        f"faces one apart (solid bars: "
        + ", ".join("–".join(e["faces"]) for e in direct) + ") or two apart (dashed: "
        + ", ".join(f"{'–'.join(e['faces'])} via {e['via'][0]}" for e in relay)
        + "), where a bridged gate is teleported through the face between, one bond on each of the two cuts. "
        f"Cuts kept to max_bonds {bp.get('max_bonds')} bond qubits ({bp.get('truncation', 'greedy')}); the wrap cut "
        f"{ring[-1]}|{ring[0]} is truncated by a local SVD, on a ring not a projection (below capacity it carries "
        f"less than the inner cuts).", 172)
    lines += textwrap.wrap(
        f"The net: base in the middle, walls folded out round it, the lid beyond the back wall. Each cell is the "
        f"lowest F_GHZ among the {k}-qubit chains through that qubit (chains run along straight lines in x, y and "
        f"z, so they cross a box edge where a line runs off one face onto the next; small number: qubit index). "
        f"Edges the net cuts open are drawn on both faces with one letter: "
        + ", ".join(f"{letters[kk]} {'–'.join(kk)}" for kk in sorted(letters, key=letters.get)) + ".", 172)
    if prot:
        lines.append("Seam protection: the drift's ZZ inside a GHZ unit is not put through the bridge (a global "
                     "phase on the GHZ, Corollary 2.1; --inside-zz stabilizer).")
    order_txt = (("code first, echo second" if order == "code-echo" else "echo first, code second")
                 + ("  ·  with seam protection" if prot else "  ·  without seam protection"))
    reg_label = f"Box {L}×{L}×{H}, six faces on a QrackBridge ring, max_bonds {bp.get('max_bonds')}"
    FW, cell, X0, GAP = 17.0, 0.44, 0.35, 0.55
    PW, PH = cell * ncol, cell * nrow
    Hh = 1.25 + 0.21 * len(lines) + 0.35
    FH = Hh + 0.35 + PH + 0.55 + 0.95
    fig = plt.figure(figsize=(FW, FH), dpi=200, facecolor=surf)
    for li, vals in enumerate(layers):
        x0 = X0 + li * (PW + GAP)
        ax = fig.add_axes([x0 / FW, (0.95 + 0.55) / FH, PW / FW, PH / FH], facecolor=surf)
        for q, F in vals.items():
            r_, c_ = sites[q]
            ax.add_patch(Rectangle((c_ - 0.5, r_ - 0.5), 1, 1, facecolor=cmap(norm(F)), edgecolor=surf, lw=1.4))
            dark = F > 0.82 or F < 0.12
            ax.text(c_, r_ + 0.08, f"{F:.2f}", ha="center", va="center", fontsize=7.2, fontweight="bold",
                    color="#ffffff" if dark else ink)
            ax.text(c_ - 0.42, r_ - 0.30, str(q), ha="left", va="center", fontsize=4.6,
                    color="#ffffff" if dark else muted)
        draw_box_net(ax, dec, ink, ink2, surf)
        ytop = (0.95 + 0.55 + PH + 0.30) / FH
        fig.text(x0 / FW, ytop, names[li], fontsize=14, fontweight="bold", color=ink, ha="left")
        fig.text((x0 + (0.80 if names[li].startswith("Pass") else 0.30)) / FW, ytop, subs[li], fontsize=9.5,
                 color=ink2, ha="left")
        fig.text(x0 / FW, (0.95 + 0.20) / FH, "   ".join(
            f"{labels[c_]} {mean_of([r for r in st if cls_of(r) == c_], li):.2f}" for c_ in classes
            if counts.get(c_)), fontsize=9.5, color=ink, ha="left", family="monospace")
    cax = fig.add_axes([(X0 + 3 * PW + 2 * GAP + 0.25) / FW, (0.95 + 0.55 + 0.15 * PH) / FH, 0.13 / FW,
                        0.7 * PH / FH])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0", "0.25", "½", "0.75", "1"])
    cb.ax.tick_params(labelsize=9, colors=ink2)
    cb.outline.set_visible(False)
    fig.text((X0 + 3 * PW + 2 * GAP + 0.10) / FW, (0.95 + 0.55 + 0.88 * PH) / FH, "F_GHZ", fontsize=10.5,
             fontweight="bold", color=ink)
    fig.text(0.02, (FH - 0.45) / FH, f"{reg_label}: GHZ fidelity per measurement layer", fontsize=18,
             fontweight="bold", color=ink, ha="left")
    fig.text(0.02, (FH - 0.85) / FH, order_txt, fontsize=14, fontweight="bold", color=ink2, ha="left")
    for i, line in enumerate(lines):
        fig.text(0.02, (FH - 1.25 - 0.21 * i) / FH, line, fontsize=9.8, color=ink2, ha="left")
    desc = dict(face=f"inside a face {counts.get('face', 0)}",
                direct=f"direct cut {counts.get('direct', 0)} (one ring cut, one bond per unit)",
                relayed=f"relayed {counts.get('relayed', 0)} (two ring cuts, a bond on each)")
    fig.text(0.02, 0.78 / FH, f"Means over {k}-qubit chains: " + ", ".join(desc[c_] for c_ in classes
                                                                           if counts.get(c_)) + ".",
             fontsize=9, color=ink2, ha="left")
    zz = ", inside ZZ by stabiliser" if prot else ""
    fig.text(0.02, 0.52 / FH, f"{st[0]['shots']} shots × {st[0]['repeats']} realisations per setting · Layer 2 "
             f"{l2.get('mode', 'none')}, L = {l2.get('blocks')}, T = {l2.get('t_idle')}, J = {l2.get('j_zz')} × "
             f"U(0.75, 1.25), σ = {l2.get('sigma_qs')}{zz} · "
             + ("V reruns the bridged units on fresh draws; others keep pass 2" if order == "echo-code" else
                "C reruns the bridged units (others keep pass 1); E runs every unit"),
             fontsize=8.3, color=muted, ha="left")
    fig.text(0.02, 0.30 / FH, f"{os.path.basename(a.inp)} · run {when} · {st[0].get('qrack', '')} · nighthawk_ghz.py "
             f"stack --engine bridge --box {L} {H} --bridge-max-bonds {bp.get('max_bonds')} --order {order}",
             fontsize=8.3, color=muted, ha="left")
    png = a.png or f"ghz_stack_{reg}_{order}.png"
    fig.savefig(png, facecolor=surf)
    plt.close(fig)
    print(f"stackmap: {reg}, order {order}, run {when}: {png}")
    for c in classes:
        rs = [r for r in st if cls_of(r) == c]
        if rs:
            print(f"  {c:<8} {len(rs):>3} units  " + "  ".join(fmt(mean_of(rs, i)) for i in range(3)))


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
        Rl = loom_register(repo, None, ["rect:8-10,0-2", "rect:8-10,3-5"], None, "exact", "ghost")
        lr = list(Rl.link_rows)
        u0, v0 = Rl.links[0]
        x0 = next(q for q in sorted(Rl.adj[u0]) if Rl.chip_of[q] == Rl.chip_of[u0] and q != v0)
        unit = chain_unit(Rl, [x0, u0, v0])
        ok_cls = seam_class(Rl, unit) == "replica"
        res = {}
        for read, pol in (("ghost", "authority"), ("ghost", "real"), ("cut", "authority")):
            per = [dict()]
            for st in settings_for(3, "auto", 12):
                sh = _loom_task(dict(repo=repo, cf=None, chips=["rect:8-10,0-2", "rect:8-10,3-5"], link_rows=lr,
                                     read=read, units=[unit], setting=st, shots=1024, exact=False, realisation=0,
                                     layer2=dict(mode="none", blocks=0), policy={tuple(unit["qubits"]): pol}))
                per[0][st] = bit_matrix(sh, Rl.n)
            res[(read, pol)] = score_unit(Rl, unit, per, 3, 1, None, maps=False)["F_GHZ"]
        Rc = loom_register(repo, None, ["rect:8-10,0-2", "rect:8-10,3-5"], lr, "exact", "cut")
        check(f"ghost read: chain {unit['qubits']} across link {u0}-{v0} lies in chip {Rl.chip_of[u0]}'s ghost band "
              f"(class replica), simulated whole there: F ~ 1; read on its own chips F <= 1/2",
              ok_cls and res[("ghost", "authority")] > 0.95 and res[("ghost", "real")] < 0.55,
              f"band {res[('ghost', 'authority')]:.3f}, own chips {res[('ghost', 'real')]:.3f}")
        check("cut read (the default): the same chain is class cut and stays <= 1/2 (the bound of any state "
              "separable across the loom) even when asked for the authority read", seam_class(Rc, unit) == "cut" and res[("cut", "authority")] < 0.55,
              f"class {seam_class(Rc, unit)}, F {res[('cut', 'authority')]:.3f}")
    print("[16] --engine bridge: QrackBridge's capacity, two GHZ units across one cut of a 6-qubit bridge "
          "(patches [0 1 2] [3 4 5]), max_bonds 0 / 1 / 2")
    try:
        QB = qrack_bridge(None)
    except SystemExit as err:
        print(f"  [skip] {err}")
    else:
        def fghz(v, qs, n):
            psi = v.reshape([2] * n)
            ax = [n - 1 - q for q in qs]
            t = np.transpose(psi, ax + [x for x in range(n) if x not in ax]).reshape(2 ** len(qs), -1)
            gv = np.zeros(2 ** len(qs))
            gv[0] = gv[-1] = 1 / math.sqrt(2)
            return float(np.real(gv @ (t @ t.conj().T) @ gv))
        got = []
        for mb in (0, 1, 2):
            b = QB(6, max_bonds=mb, patches=2)
            g = BridgeGates(b, list(range(6)))
            prepare(g, dict(qubits=[2, 1, 3], root=2, edges=[[2, 1], [2, 3]], classes=["", ""]), [])
            prepare(g, dict(qubits=[4, 5, 0], root=4, edges=[[4, 5], [4, 0]], classes=["", ""]), [])
            v = np.asarray(b.out_ket())
            got.append(sorted([fghz(v, [1, 2, 3], 6), fghz(v, [0, 4, 5], 6)]))
        check("max_bonds 0: both <= 1/2 (a hard cut); 1: one unit carried; 2: both exact",
              got[0][1] < 0.51 and got[1][0] < 0.51 and got[1][1] > 0.99 and got[2][0] > 0.99,
              "; ".join(f"{mb}: {a_:.3f} {b_:.3f}" for mb, (a_, b_) in zip((0, 1, 2), got)))
    print("[17] --box: the 5 x 5 x 5 box's geometry; on the 3 x 3 x 3 box, a GHZ chain across a relayed edge "
          "through the ring's wrap cut, max_bonds 0 / 1")
    Rb = BoxRegister(5, 5)
    ring_d = {tuple(e["faces"]): e["ring_distance"] for e in Rb.edges}
    check("98 qubits (base and lid 25, walls 12), 192 couplers, bipartite under x + y + z, 12 box edges: 6 on a "
          "ring cut, 6 relayed; opposite faces never share an edge",
          Rb.n == 98 and Rb.chip_n == [25, 12, 12, 25, 12, 12] and len(Rb.couplers) == 192
          and all(Rb.sub[a_] != Rb.sub[b_] for a_, b_ in Rb.couplers) and len(ring_d) == 12
          and sorted(ring_d.values()) == [1] * 6 + [2] * 6
          and not any(set(k_) in ({"base", "lid"}, {"front", "back"}, {"left", "right"}) for k_ in ring_d),
          f"n {Rb.n}, faces {Rb.chip_n}, couplers {len(Rb.couplers)}, ring distances {sorted(ring_d.values())}")
    units = [u for _, _, us in Rb.tilings(3) for u in us]
    cross = [u for u in units if Rb.unit_box(u)["edge"]]
    check("3-qubit chains along x, y and z: 144 units, 44 across a box edge, every cross-face coupler in one",
          len(units) == 144 and len(cross) == 44
          and {tuple(sorted(e)) for u in cross for e in u["edges"] if Rb.chip_of[e[0]] != Rb.chip_of[e[1]]}
          == {tuple(sorted(x)) for x in Rb.links}, f"{len(units)} units, {len(cross)} across an edge")
    try:
        QB = qrack_bridge(None)
    except SystemExit as err:
        print(f"  [skip] {err}")
    else:
        R3 = BoxRegister(3, 3)
        u = next(u for _, _, us in R3.tilings(3) for u in us
                 if R3.unit_box(u)["hops"] == 2 and len(BOX_FACES) - 1 in R3.unit_box(u)["cuts"])
        cuts = R3.unit_box(u)["cuts"]
        got = []
        for mb in (0, 1):
            width, P, idx = _bridge_layout(R3, [u])
            b = QB(width, max_bonds=mb, patches=P, wrap=True)
            prepare(BridgeGates(b, idx), u, [])
            v = np.asarray(b.out_ket())
            got.append(fghz(v, [idx[q] for q in u["qubits"]], width))
        check(f"chain {u['qubits']} ({R3.unit_box(u)['edge']}, ring cuts {cuts}): max_bonds 0 <= 1/2, 1 exact",
              got[0] < 0.51 and got[1] > 0.99, f"{got[0]:.3f} / {got[1]:.3f}")
    print("[18] shor: Shor-9 placements on the 5 x 5 x 5 box; across one direct edge (not the wrap), the cut between "
          "blocks and through them at max_bonds 0 / 1 / 3 (256 shots)")
    Rb = BoxRegister(5, 5)
    cands = shor_candidates(Rb, 3)
    cl = collections.Counter(shor_unit_info(Rb, u)["layout"] for u, _ in cands)
    check("96 placements: 48 inside a face, 24 with the cut between blocks, 24 through them; every encoder CX a coupler",
          len(cands) == 96 and cl == {"inside": 48, "blocks whole": 24, "blocks split": 24},
          f"{len(cands)}: {dict(cl)}")
    try:
        qrack_bridge(None)
    except SystemExit as err:
        print(f"  [skip] {err}")
    else:
        got = {}
        for lay in ("blocks whole", "blocks split"):
            u = next(u for u, _ in cands if shor_unit_info(Rb, u)["layout"] == lay
                     and shor_unit_info(Rb, u)["hops"] == 1 and not shor_unit_info(Rb, u)["wrap"])
            _, local = nq._local_unit(u)
            for mb in (0, 1, 3):
                for st in ("0", "+"):
                    sh, _ = _shor_task(dict(unit=u, state=st, shots=256, exact=False, realisation=0, index=0,
                                            box=[5, 5], max_bonds=mb))
                    got[(lay, mb, st)] = float(np.mean([nq.shor9_decode(x, local, st)[1] for x in sh]))
        w, sp = (lambda mb, st: got[("blocks whole", mb, st)]), (lambda mb, st: got[("blocks split", mb, st)])
        check("cut between blocks: |0_L> needs no bond (1 at max_bonds 0), |+_L> one (~1/2 at 0, 1 at 1)",
              w(0, "0") > 0.99 and abs(w(0, "+") - 0.5) < 0.12 and w(1, "+") > 0.99,
              f"|0_L> {w(0, '0'):.3f}; |+_L> {w(0, '+'):.3f} / {w(1, '+'):.3f}")
        check("cut through blocks: |+_L> survives any capacity; |0_L> ~1/2 at 0, ~3/4 at 1 (the decoder's majority over "
              "one carried and two random block signs), 1 at 3",
              sp(0, "+") > 0.99 and abs(sp(0, "0") - 0.5) < 0.12 and abs(sp(1, "0") - 0.75) < 0.12
              and sp(3, "0") > 0.99, f"|+_L> {sp(0, '+'):.3f}; |0_L> {sp(0, '0'):.3f} / {sp(1, '0'):.3f} / "
                                     f"{sp(3, '0'):.3f}")
    print("[19] dual: the dual Nighthawk R1 (arXiv:2610.29986); the distant CZ's decomposition against a CZ; Bell pairs "
          "across the cable on the bridge and the hard cut; the measured-link noise against its benchmark")
    D = DualR1()
    check("248 qubits, 2 x 218 lattice couplers + 8 lattice-L couplers, 4 cables, every lattice qubit of row 11 that "
          "an L-coupler serves", D.n == 248 and len(D.couplers) == 444 and len(D.cables) == 4
          and all(L["a"] // 10 == 11 and (L["b"] - 120) // 10 == 11 for L in DUAL_LINKS), f"{len(D.couplers)} couplers")
    from pyqrack import QrackSimulator
    worst = 1.0
    rng = np.random.default_rng(3)
    for L in DUAL_LINKS:
        idx = {L["a"]: 0, L["la"]: 1, L["lb"]: 2, L["b"]: 3}
        th = rng.uniform(0, math.pi, 4)
        s1, s2 = QrackSimulator(4), QrackSimulator(4)
        for sim in (s1, s2):
            sim.u(0, th[0], th[1], 0.0)
            sim.u(3, th[2], th[3], 0.0)
        DualGates(s1, idx).dcz(L["a"], L["b"])
        s2.mcz([0], 3)
        v1, v2 = np.asarray(s1.out_ket()), np.asarray(s2.out_ket())
        worst = min(worst, abs(np.vdot(v1, v2)) ** 2)
    check("distant CZ (CNOT into the L-qubit, transfer, CZ, transfer back, CNOT) = CZ on the lattice pair, L-qubits back "
          "in |0>, every link", worst > 1 - 1e-5, f"worst overlap {worst:.7f}")
    try:
        qrack_bridge(None)
    except SystemExit as err:
        print(f"  [skip] {err}")
    else:
        got = {}
        for mb in (0, 1):
            rh, _ = _dual_task(dict(state="bell 115-233", engine="bridge", noise="none", max_bonds=mb, trajectories=1,
                                    seed=[0], r_cz=0.0, r_1q=0.0))
            got[mb] = dual_fidelity(rh[0], "bell")
        rh, _ = _dual_task(dict(state="bell 111-237", engine="bridge", noise="measured", max_bonds=6, trajectories=3000,
                                seed=[1], r_cz=0.0, r_1q=0.0))
        fm = dual_fidelity(rh[0], "bell")
        check("Bell 115-233 across the cable: 1 on the bridge at max_bonds 1, 1/2 at 0 (the hard cut)",
              got[1] > 0.999 and abs(got[0] - 0.5) < 1e-3, f"{got[1]:.4f} / {got[0]:.4f}")
        check("measured-link noise alone: Bell 111-237 at the link's distant-CZ fidelity 0.9642 (3000 trajectories, "
              "+- 0.012; a two-qubit depolarising channel of average error r leaves a Bell state at 1 - r)",
              abs(fm - DUAL_LINKS[0]["dcz"]) < 0.012, f"{fm:.4f}")
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

    bridge_opts = argparse.ArgumentParser(add_help=False)
    bridge_opts.add_argument("--bridge-max-bonds", dest="bridge_max_bonds", type=int, default=6,
                   help="bridge: bond qubits kept per cut after each recompression (QrackBridge max_bonds): its "
                        "capacity, 2^max_bonds Schmidt values; 0 keeps none (a hard cut)")
    bridge_opts.add_argument("--bridge-truncation", dest="bridge_truncation", choices=["greedy", "born"], default="greedy",
                   help="bridge: keep the largest Schmidt values (greedy) or draw a group by the Born rule")
    bridge_opts.add_argument("--bridge-lazy", dest="bridge_lazy", type=int, default=0,
                   help="bridge: lazy_bonds, bond qubits buffered before one recompression (0: after every gate)")
    bridge_opts.add_argument("--qrack-bridge", dest="qrack_bridge", default=None,
                   help="bridge: qrack_bridge.py to load when this PyQrack has no QrackBridge (PyQrack >= 2.30 has it; "
                        "default $NIGHTHAWK_QRACK_BRIDGE)")
    bridge_opts.add_argument("--box", type=int, nargs=2, metavar=("L", "H"), default=None,
                   help="bridge: a closed box instead of chips: the lattice points on the surface of an L x L x H "
                        "block, six faces (base, four walls, lid) as six QrackBridge patches on a ring (base, front, "
                        "right, lid, back, left); chains along x, y and z, so units cross every box edge")
    bridge_opts.add_argument("--loom-chips", dest="loom_chips", nargs="+", default=None,
                   help="loom: the chips, nighthawk_cloverfield.py --chips tokens (default rect:1-8,2-5 rect:1-8,6-9: "
                        "nighthawk61r2's 61 qubits as two chips of 30 and 31, its two ACE patches' size)")
    bridge_opts.add_argument("--loom-link-rows", dest="loom_link_rows", type=int, nargs="+", default=None,
                   help="loom: the rows that get a link (default: every row whose facing border pair are lattice "
                        "neighbours once the chips sit side by side; the default chips: 1 2 3 4 6 7 8, every coupler "
                        "of nighthawk61r2 between columns 5 and 6)")
    bridge_opts.add_argument("--cloverfield", default=None,
                   help="loom: nighthawk_cloverfield.py (default: $NIGHTHAWK_CLOVERFIELD, next to this script, or "
                        "../rcs-nighthawk)")

    s = sub.add_parser("stack", parents=[common, bridge_opts],
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
    s.add_argument("--engine", choices=["ace", "loom", "bridge"], default="ace",
                   help="ace: the register on ACE's patches (default). loom: two exact chips joined by "
                        "nighthawk_cloverfield.py's ghost loom as the seam (--loom-chips, --loom-link-rows); Layer S's "
                        "code is then the loom's authority (a unit read on the chip that holds copies of all of it)")
    s.add_argument("--loom-read", dest="loom_read", choices=["cut", "ghost"], default="cut",
                   help="loom: cut (default) = a hard cut, chips independent, a link CZ as the mean-field phase on "
                        "each side, every qubit read on its own chip: a unit across the loom is a product state; "
                        "ghost = the cloverfield's ghost loom, each chip also simulating its partners' anchor qubits "
                        "(a replica band two columns wide): a unit inside one chip's band is read there, simulated "
                        "whole; that is not a correction across the loom")
    s.add_argument("--loom-chip-backend", dest="loom_chip_backend", choices=["exact", "ace"], default="exact",
                   help="loom: each chip an exact QrackSimulator (default), or its register's two-patch ACE preset "
                        "(chips must be registers, e.g. nighthawk61r2 nighthawk61r2): ACE's seam inside each chip, "
                        "the loom between them; Layer S's code is then ACE's boundary repetition code inside and the "
                        "loom's authority across")
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
    pr.add_argument("--seam-class", dest="seam_class", choices=SEAM_CLASSES[:3], default="cut")
    pr.add_argument("--variants", default=",".join(PROBE_VARIANTS),
                    help="comma list of " + ", ".join(PROBE_VARIANTS) + " (chain-*: the GHZ prepared from one end; "
                         "none: no Layer 2; gate, tseam, tbulk: the inside ZZ as CX Rz CX with the CX's target as "
                         "listed, on the seam qubit, on the bulk qubit; stab: dropped)")
    pr.add_argument("--exact-variants", dest="exact_variants", default="gate,stab,chain-stab",
                    help="variants also run on the exact simulator")

    sh = sub.add_parser("shor", parents=[common, bridge_opts],
                        help="Shor's [[m^2, 1, m]] code across QrackBridge circuits (two chips, or the box)")
    sh.add_argument("--engine", choices=["bridge"], default="bridge", help="the chips joined by QrackBridge")
    sh.add_argument("--size", type=int, default=3, help="m: Shor-m^2 (3 = Shor-9)")
    sh.add_argument("--shor-units", dest="shor_units", choices=["all", "cross"], default="all",
                    help="every placement of a unit, or only those across a cut")
    sh.set_defaults(shots=1024, repeats=1, exact_shots=256)

    dm = sub.add_parser("dualmap", help="GHZ fidelity heatmaps over both modules of the dual Nighthawk R1 and their "
                                        "L-couplers")
    dm.add_argument("--models", default="hardcut,budget,measured", help="as dual, one panel each")
    dm.add_argument("--trajectories", type=int, default=400, help="Pauli trajectories per chain inside a module")
    dm.add_argument("--trajectories-link", dest="trajectories_link", type=int, default=4000,
                    help="Pauli trajectories per chain through a link")
    dm.add_argument("--chunk", type=int, default=1000, help="(task order)")
    dm.add_argument("--local-cz-error", dest="local_cz_error", type=float, default=0.00355)
    dm.add_argument("--lattice-1q-error", dest="lattice_1q_error", type=float, default=0.0005)
    dm.add_argument("--bridge-max-bonds", dest="bridge_max_bonds", type=int, default=6)
    dm.add_argument("--qrack-bridge", dest="qrack_bridge", default=None)
    dm.add_argument("--seed", type=int, default=7)
    dm.add_argument("--jobs", type=int, default=1)
    dm.add_argument("--progress", type=float, default=30.0)
    dm.add_argument("--out", default=None, help="append dual_map JSONL records")
    dm.add_argument("--png", default=None, help="the heatmaps")

    dp = sub.add_parser("dualplot", help="dual's model-against-measurement plot from its records")
    dp.add_argument("--in", dest="inp", required=True, help="JSONL written by dual --out (its latest run is drawn)")
    dp.add_argument("--png", required=True, help="output image")

    sm = sub.add_parser("shormap", help="a shor run's logical Z and X success as heatmaps on the chips or the box net")
    sm.add_argument("--in", dest="inp", required=True, help="JSONL written by shor --out (its latest run is drawn)")
    sm.add_argument("--png", default=None, help="output image (default shor_<register>_mb<max_bonds>.png)")

    du = sub.add_parser("dual", help="the dual Nighthawk R1 of arXiv:2610.29986: two 120-qubit modules, four "
                                     "L-couplers; its Bell and GHZ states under the bridge's models")
    du.add_argument("--models", default="hardcut,ideal,budget,measured",
                    help="comma list of hardcut (bridge at max_bonds 0), ideal (bridge, exact), budget (constituent "
                         "noise), measured (the benchmarked distant CZ as one channel), exact (QrackSimulator)")
    du.add_argument("--trajectories", type=int, default=2000, help="Pauli trajectories per circuit for noisy models")
    du.add_argument("--batch", type=int, default=100, help="trajectories per task (the standard error is over batches)")
    du.add_argument("--local-cz-error", dest="local_cz_error", type=float, default=0.00355,
                    help="average error of a CZ inside a module (default: the paper's budget for CZ111-237, three "
                         "local CZs 40%% of 2.66%%)")
    du.add_argument("--lattice-1q-error", dest="lattice_1q_error", type=float, default=0.0005,
                    help="average error of a single-qubit gate on a lattice qubit (not in the paper; L-qubits use "
                         "Extended Data Table 1)")
    du.add_argument("--bridge-max-bonds", dest="bridge_max_bonds", type=int, default=6, help="bond qubits per cut")
    du.add_argument("--qrack-bridge", dest="qrack_bridge", default=None, help="qrack_bridge.py (PyQrack < 2.30)")
    du.add_argument("--seed", type=int, default=7, help="trajectory seed")
    du.add_argument("--jobs", type=int, default=1, help="worker processes (0 = every core)")
    du.add_argument("--progress", type=float, default=30.0, help="seconds between heartbeats (0 = off)")
    du.add_argument("--out", default=None, help="append dual_r1 JSONL records")
    du.add_argument("--png", default=None, help="model against measurement, one row per state")

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
    if a.cmd in ("chip", "tiles", "stack", "sweep", "probe", "shor"):
        nq.ensure_isolated(a)
    dict(chip=cmd_chip, tiles=cmd_tiles, stack=cmd_stack, sweep=cmd_sweep, probe=cmd_probe, stackmap=cmd_stackmap,
         shor=cmd_shor, shormap=cmd_shormap, dual=cmd_dual, dualmap=cmd_dualmap, dualplot=cmd_dualplot, plot=cmd_plot, selftest=cmd_selftest)[a.cmd](a)


if __name__ == "__main__":
    main()
