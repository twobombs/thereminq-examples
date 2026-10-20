#!/usr/bin/env python3
"""
nighthawk_erepr.py -- ER=EPR on the cloverleaf and the cloverfield.

The 2022 Sycamore wormhole (Jafferis et al., Nature 612, 51) re-derived from its stated learned
Hamiltonian, then compiled onto Nighthawk lattice registers and run with nighthawk_qrack's engines
(ExactEngine, AceEngine and its placement machinery) and nighthawk_cloverfield's chips, workers and
loom. The two sides of the wormhole become two chips; the thermofield double (TFD) that joins them is
the EPR, and the loom is the only way it can cross.

Circuit (split encoding)
  L and R: 7 + 7 Majoranas of the learned Hamiltonian (+ psi^8 per side) by Jordan-Wigner on 4 + 4
  qubits, the R strings carrying the parity of L; P (reference) and Q (message). Protocol:
  |TFD> |Phi+>_PQ -> e^{+i H_L t0} -> swap(Q, psi_L^1 psi_L^2) -> e^{-i H_L t0} -> e^{i mu V}
  -> e^{-i H_R t1}; readout of (psi_R^1, psi_R^2) on R0 with the L parity string; I_PT and the best
  average teleportation fidelity from the 16 correlators <sigma_P tau>.
  The TFD is Schmidt-loaded: sum_n s_n |n>|n> by amplitude loading on L and four L->R copy CNOTs,
  then W_L (x) W_R; the coupling is one ZZ rotation per Majorana pair between ports (L_k, R_k),
  k = (j - 1) // 2. Nothing else crosses L|R.

Registers (Nighthawk 12 x 10 lattice coordinates, release's rect_matchings)
  field wide     chip A rect:0-1,0-3 (row 0: Q P - -, row 1: L0 L1 L2 L3) over
                 chip B rect:0-0,0-3 (R0 R1 R2 R3): 4 links L_k - R_k, every side qubit a link anchor
  field narrow   chip A as above over chip B rect:0-1,0-1 (R0 R1 / R2 R3): 2 links (L0-R0, L1-R1);
                 L2, L3, R2, R3 reach the links through SWAPs
  single         rect:0-3,0-3: the wide field as one 16-site register (row 3 idle), for ACE
  Every two-qubit gate is routed onto lattice couplers or links with SWAP chains that restore the
  layout, CNOT = H CZ H, so the circuit is the CZ + single-qubit form the cloverleaf and the loom run.

Subcommands
  derive    exact reference in numpy (no PyQrack): structure, operator growth, correlators, size
            winding, random-coefficient control, teleportation (all pairs, both signs, V normalisation,
            the Reply's H1, SYK N = 10) -> JSON
  plan      field, links, roles, routed gate counts; ACE placements with nighthawk's coupler classes
  run       --backend exact   nq.ExactEngine on the routed field register
            --backend ace     nq.AceEngine on the single register, lrc/lrr by nq.two_patch_config,
                              --placement grid | tiling (nq.ace_tiling, circuit-weighted) | lr | seam-r
            --backend loom    cloverfield: one ChipProc worker per chip (isolated processes, exact
                              chips), the Loom weaving every CZ: --loom ghost | mean | sample
            --backend knit    the cut resolved exactly: TFD Schmidt sum x 2^7 coupling branches, each
                              chip one nq.ExactEngine with a local branch register
  table     JSONL files side by side (mean +- sd over repetitions)
  aceprobe  Bell pairs across the two-patch register's site classes
  selftest  algebra, gates, routing, engines, ghost loom on the wide field, knit, against the reference

Release data (layout.json, rcs/layout.py) as for nighthawk_qrack: --repo, $NIGHTHAWK_REPO, or an
rcs-nighthawk clone next to this script.

Examples
  python3 nighthawk_erepr.py selftest
  python3 nighthawk_erepr.py derive --out erepr/reference.json
  python3 nighthawk_erepr.py plan --field narrow
  python3 nighthawk_erepr.py run --backend ace --placement grid tiling lr seam-r --reps 8 --out erepr/ace.jsonl
  python3 nighthawk_erepr.py run --backend loom --field narrow --loom ghost mean --out erepr/loom.jsonl
  python3 nighthawk_erepr.py table erepr/*.jsonl
"""
import argparse
import functools
import itertools
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH) and "PYQRACK_SHARED_LIB_PATH" not in os.environ:
    os.environ["PYQRACK_SHARED_LIB_PATH"] = QRACK_LIB_PATH

HERE = Path(__file__).resolve().parent
os.environ.setdefault("NIGHTHAWK_TILE_CACHE", str(HERE / "erepr" / "tiles.json"))
sys.path.insert(0, str(HERE))
import nighthawk_qrack as nq            # noqa: E402
import nighthawk_cloverfield as cf      # noqa: E402

# ================================================================== inputs (as published)
MODEL1 = [(-0.36, (1, 2, 4, 5)), (0.19, (1, 3, 4, 7)), (-0.71, (1, 3, 5, 6)),
          (0.22, (2, 3, 4, 6)), (0.49, (2, 3, 5, 7))]          # Jafferis et al.; Comment Eq. 1; Reply Eq. 3
H1_REPLY = [(0.3, (1, 2, 3, 5))]                                  # Reply Eq. 4
Q_BODY, BETA, MU, T_WIND, N_V = 4, 4.0, 12.0, 2.8, 10
QN = Q_BODY * N_V
SQ2 = math.sqrt(2.0)
I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.diag([1.0 + 0j, -1.0])
PAULI = (I2, PX, PY, PZ)
H_M = np.array([[1, 1], [1, -1]], dtype=complex) / SQ2
SDG = np.diag([1, -1j])
BASIS = {"X": H_M, "Y": H_M @ SDG, "Z": I2}                      # B P B^dagger = Z

FIELDS = {"wide": ("rect:0-1,0-3", "rect:0-0,0-3"), "narrow": ("rect:0-1,0-3", "rect:0-1,0-1")}
SINGLE = "rect:0-3,0-3"
# field / single-register index of each role: Q, P on row 0, L_k on row 1, R_k next (row 2 or chip B)
ROLE = {"Q": 0, "P": 1, **{f"L{k}": 4 + k for k in range(4)}, **{f"R{k}": 8 + k for k in range(4)}}


def kron(*ops):
    return functools.reduce(np.kron, ops)


# ================================================================== reference (numpy, dense, exact)
class Side:
    """n Majoranas on n/2 qubits (Jordan-Wigner, psi^2 = 1/2), the Hamiltonian h and its eigensystem."""

    def __init__(self, n, terms):
        n += n % 2
        self.n, self.nq, self.d = n, n // 2, 2 ** (n // 2)
        self.g = [kron(*([PZ] * k + [P] + [I2] * (self.nq - k - 1))) / SQ2 for k in range(self.nq) for P in (PX, PY)]
        self.par = kron(*([PZ] * self.nq))
        self.h = self.ham(terms)
        self.e, self.w = np.linalg.eigh(self.h)

    def ham(self, terms):
        h = np.zeros((self.d, self.d), dtype=complex)
        for c, idx in terms:
            m = np.eye(self.d, dtype=complex)
            for j in idx:
                m = m @ self.g[j - 1]
            h += c * m
        return h

    def U(self, t):
        return (self.w * np.exp(-1j * self.e * t)) @ self.w.conj().T

    def rho(self, beta):
        p = np.exp(-beta * (self.e - self.e.min()))
        p /= p.sum()
        return (self.w * np.sqrt(p)) @ self.w.conj().T, (self.w * p) @ self.w.conj().T

    def heis(self, op, t):
        u = self.U(t)
        return u.conj().T @ op @ u


def syk_terms(n, J, seed):
    rng = np.random.default_rng(seed)
    sd = math.sqrt(6.0 * J * J / n ** 3)
    return [(float(rng.normal(0, sd)), idx) for idx in itertools.combinations(range(1, n + 1), 4)]


def majorana_commute(a, b):
    return len(set(a) & set(b)) % 2 == 0


@functools.lru_cache(maxsize=4)
def string_basis(n):
    s = Side(n, [])
    subsets = [S for k in range(n + 1) for S in itertools.combinations(range(n), k)]
    G = np.empty((len(subsets), s.d, s.d), dtype=complex)
    for i, S in enumerate(subsets):
        k = len(S)
        m = np.eye(s.d, dtype=complex)
        for j in S:
            m = m @ s.g[j]
        G[i] = (1j ** (k * (k - 1) // 2)) * (2 ** (k / 2)) * m / math.sqrt(s.d)     # Hermitian, orthonormal
    return subsets, np.array([len(S) for S in subsets]), G


def coeffs(n, O):
    _, sizes, G = string_basis(n)
    return np.einsum("sij,ji->s", G, O), sizes


def winding_metrics(c, sizes, n):
    q = np.array([np.sum(c[sizes == l] ** 2) for l in range(n + 1)])
    p = np.array([np.sum(np.abs(c[sizes == l]) ** 2) for l in range(n + 1)])
    r_l = np.where(p > 1e-12, np.abs(q) / np.maximum(p, 1e-300), np.nan)
    r, W = float(np.sum(np.abs(q))), float(np.real(np.sum(q)))
    rbar = (r - W) / (1 - W) if abs(1 - W) > 1e-12 else float("nan")
    ls = np.arange(n + 1)
    mus = np.linspace(0, 2 * math.pi, 8192, endpoint=False)
    C = np.abs(np.exp(-1j * np.outer(mus, ls)) @ q)
    k = int(np.argmax(C))
    for _ in range(3):
        mus = np.linspace(mus[k] - 2 * math.pi / 8192, mus[k] + 2 * math.pi / 8192, 2001)
        C = np.abs(np.exp(-1j * np.outer(mus, ls)) @ q)
        k = int(np.argmax(C))
    top = np.argsort(-np.abs(q))[:2]
    L = max(abs(W), 2 * abs(q[top[0]]) + 2 * abs(q[top[1]]) - r)
    chi = (float(C[k]) - L) / (r - L) if r - L > 1e-12 else float("nan")
    return dict(r_l=r_l, p=p, r=r, W=W, rbar=float(rbar), chi=float(chi), slope=float(mus[k]))


def winding_side(s, i, beta=BETA, t=T_WIND):
    rs, _ = s.rho(beta)
    c, sizes = coeffs(s.n, SQ2 * rs @ s.heis(s.g[i - 1], t))
    return winding_metrics(c, sizes, s.n)


def apL(M, psi):
    return np.tensordot(M, psi, axes=(1, 0))


def apR(M, psi):
    return np.moveaxis(np.tensordot(M, psi, axes=(1, 1)), 0, 1)


def apP(M, psi):
    return np.moveaxis(np.tensordot(M, psi, axes=(1, 2)), 0, 2)


def apQ(M, psi):
    return np.moveaxis(np.tensordot(M, psi, axes=(1, 3)), 0, 3)


class TwoSided:
    """|TFD> (x) |Phi+>_PQ as a tensor psi[l, r, p, q]; psi_L^j = g_j (x) 1, psi_R^j = Par_L (x) g_j."""

    def __init__(self, side, beta=BETA, n_used=7, qN=QN):
        self.s, self.qN, self.used = side, qN, n_used
        d = side.d
        self.A = [gj @ side.par for gj in side.g]
        K = sum(1j * np.kron(self.A[j], side.g[j]) for j in range(side.n))
        ev, ew = np.linalg.eigh(K)
        self.k_gap = float(ev[1] - ev[0])
        hl, hr = np.kron(side.h, np.eye(d)), np.kron(np.eye(d), side.h)
        self.hlhr_err = float(np.abs((hl - hr) @ ew[:, 0]).max())
        M = (side.w * np.exp(-beta * (side.e - side.e.min()) / 2)) @ side.w.conj().T
        tfd = M @ ew[:, 0].reshape(d, d)
        self.tfd = tfd / np.linalg.norm(tfd)

    def run(self, mu, t0, t1, pair=(1, 2)):
        s, (a_, b_) = self.s, pair
        psi = np.einsum("lr,pq->lrpq", self.tfd, I2 / SQ2)
        psi = apL(s.U(-t0), psi)
        Xt, Yt = SQ2 * s.g[a_ - 1], SQ2 * s.g[b_ - 1]
        new = psi.copy()
        for Pq, M in ((PX, Xt), (PY, Yt), (PZ, -1j * Xt @ Yt)):
            new += apL(M, apQ(Pq, psi))
        psi = apL(s.U(t0), 0.5 * new)
        th = mu / (2 * self.qN)
        for j in range(self.used):
            psi = math.cos(th) * psi - 2 * math.sin(th) * apL(self.A[j], apR(s.g[j], psi))
        psi = apR(s.U(t1), psi)
        d = s.d
        tau = [(np.eye(d), np.eye(d)), (s.par, SQ2 * s.g[a_ - 1]), (s.par, SQ2 * s.g[b_ - 1]),
               (np.eye(d), -2j * s.g[a_ - 1] @ s.g[b_ - 1])]
        self.tau_err = max(float(np.abs(np.kron(L_ @ L_, R_ @ R_) - np.eye(d * d)).max()) for L_, R_ in tau)
        Tm = np.zeros((4, 4))
        for j, (L_, R_) in enumerate(tau):
            w = apL(L_, apR(R_, psi))
            for i, sp in enumerate(PAULI):
                Tm[i, j] = float(np.real(np.vdot(psi, apP(sp, w))))
        return metrics(Tm)


def metrics(Tm):
    """rho_PT from <sigma_i (x) tau_j>; I_PT in bits; best average fidelity over a decoding rotation on T
    (classical bound 2/3); smallest eigenvalue of the reconstructed rho_PT."""
    Tm = np.asarray(Tm, dtype=float)
    rho = sum(Tm[i, j] * np.kron(PAULI[i], PAULI[j]) for i in range(4) for j in range(4)) / 4
    rho = (rho + rho.conj().T) / 2

    def S(r):
        ev = np.clip(np.linalg.eigvalsh(r), 0, None)
        ev = ev / ev.sum()
        ev = ev[ev > 1e-14]
        return float(-np.sum(ev * np.log2(ev)))
    r4 = rho.reshape(2, 2, 2, 2)
    I = S(np.einsum("ijkj->ik", r4)) + S(np.einsum("ijil->jl", r4)) - S(rho)
    M = np.diag([1.0, -1.0, 1.0]) @ Tm[1:, 1:]
    sv = np.linalg.svd(M, compute_uv=False)
    Fe = (1 + sv[0] + sv[1] + (np.sign(np.linalg.det(M)) or 1.0) * sv[2]) / 4
    return dict(I=max(I, 0.0), Favg=float((2 * Fe + 1) / 3), min_eig=float(np.linalg.eigvalsh(rho).min()))


def peak(ts, ys):
    k = int(np.argmax(ys))
    return float(ts[k]), float(ys[k])


def derive_structure(out):
    s = Side(8, MODEL1)
    terms = [t for _, t in MODEL1]
    mats = [s.ham([(1.0, t)]) for t in terms]
    comm = max(np.abs(mats[i] @ mats[j] - mats[j] @ mats[i]).max() for i, j in itertools.combinations(range(5), 2))
    quads = list(itertools.combinations(range(1, 8), 4))
    adj = {q: {p for p in quads if p != q and majorana_commute(p, q)} for q in quads}
    cliques = []

    def grow(cl, cand):
        if len(cl) == 5:
            cliques.append(tuple(cl))
            return
        for q in sorted(cand):
            if not cl or q > cl[-1]:
                grow(cl + [q], cand & adj[q])
    grow([], set(quads))
    perms = list(itertools.permutations(range(1, 8)))
    canon = lambda c: min(tuple(sorted(tuple(sorted(p[i - 1] for i in t)) for t in c)) for p in perms)
    classes = {canon(c) for c in cliques}
    levels = sorted(set(np.round(s.e, 6).tolist()))
    freqs = sorted(set(np.round(np.abs(np.subtract.outer(levels, levels)).ravel(), 6).tolist()) - {0.0})
    out["structure"] = dict(max_commutator=float(comm), commuting_5term_sets=len(cliques),
                            touching_all7=sum(1 for c in cliques if len(set().union(*c)) == 7),
                            relabeling_classes=len(classes), model1_in_class=canon(terms) in classes,
                            spectrum_levels=levels, transition_frequencies=len(freqs))
    print(f"structure: max |[A,B]| = {comm:.1e}; {len(cliques)} commuting 5-term sets, {len(classes)} relabeling "
          f"class(es), Model 1 in it: {canon(terms) in classes}; {len(levels)} levels, {len(freqs)} frequencies")


def derive_growth(out):
    s = Side(8, MODEL1)
    rows = []
    for i in range(1, 8):
        c, sizes = coeffs(8, SQ2 * s.heis(s.g[i - 1], T_WIND))
        nz = np.abs(c) > 1e-9
        rows.append(dict(i=i, strings=int(nz.sum()), sizes=sorted({int(x) for x in sizes[nz]})))
    syk = Side(10, syk_terms(10, 1.125, 0))
    c, sizes = coeffs(10, SQ2 * syk.heis(syk.g[0], T_WIND))
    out["growth"] = dict(model1=rows, syk10_psi1=int((np.abs(c) > 1e-9).sum()))
    print("growth at t = 2.8: " + ", ".join(f"psi^{r['i']} {r['strings']}" for r in rows)
          + f"; SYK N = 10 psi^1: {out['growth']['syk10_psi1']} strings")


def derive_correlators(out):
    s = Side(8, MODEL1)
    _, rho = s.rho(BETA)
    ts = np.arange(0, 30.0001, 0.1)
    rows, Gs = [], []
    for i in range(1, 9):
        G = np.array([2 * np.real(np.trace(rho @ s.heis(s.g[i - 1], t) @ s.g[i - 1])) for t in ts])
        Gs.append(G)
        late = ts > 5
        rows.append(dict(i=i, G_min=float(G.min()), G_max_after_t5=float(G[late].max()),
                         t_late_max=float(ts[late][np.argmax(G[late])])))
    Ga = np.mean(Gs, axis=0)
    out["correlators"] = dict(rows=rows, Gavg_min=float(Ga.min()), Gavg_max_after_t5=float(Ga[ts > 5].max()))
    print("correlators: " + ", ".join(f"G{r['i']} back to {r['G_max_after_t5']:.3f} at t = {r['t_late_max']:.1f}"
                                      for r in rows[:7]))


def derive_winding(out, syk_seeds=3):
    res = {"model1": [], "model1_plus_H1": [], "syk10": []}
    for name, terms in (("model1", MODEL1), ("model1_plus_H1", MODEL1 + H1_REPLY)):
        side = Side(8, terms)
        for i in range(1, 8):
            m = winding_side(side, i)
            res[name].append(dict(i=i, r_l=[None if np.isnan(x) else float(x) for x in m["r_l"]],
                                  p_l=[float(x) for x in m["p"]], r=m["r"], rbar=m["rbar"], chi=m["chi"]))
    for seed in range(syk_seeds):
        side = Side(10, syk_terms(10, 1.125, seed))
        for i in (1, 2):
            m = winding_side(side, i)
            res["syk10"].append(dict(seed=seed, i=i, r=m["r"], rbar=m["rbar"], chi=m["chi"]))
    out["winding"] = res
    print("winding (chi): Model 1 " + " ".join(f"{r['chi']:.3f}" for r in res["model1"])
          + "; SYK " + " ".join(f"{r['chi']:.3f}" for r in res["syk10"]))


def derive_random(out, instances=1000, seed=7):
    rms = math.sqrt(np.mean([c * c for c, _ in MODEL1]))
    base = [winding_side(Side(8, MODEL1), i) for i in range(1, 8)]
    thr = min(base[0]["chi"], base[1]["chi"])
    rb = (base[0]["rbar"], base[1]["rbar"])
    rng = np.random.default_rng(seed)
    chis, rbars = [], []
    for _ in range(instances):
        side = Side(8, [(float(rng.normal(0, rms)), idx) for _, idx in MODEL1])
        ms = [winding_side(side, i) for i in range(1, 8)]
        chis.append([m["chi"] for m in ms])
        rbars.append([m["rbar"] for m in ms])
    chis, rbars = np.nan_to_num(np.array(chis), nan=-1), np.array(rbars)
    srt = -np.sort(-chis, axis=1)
    out["random"] = dict(instances=instances, seed=seed, coef_sd=rms, threshold=thr,
                         frac_best_two=float(np.mean(srt[:, 1] >= thr)), frac_all_seven=float(np.mean(srt[:, 6] >= thr)),
                         frac_rbar=float(np.mean((rbars[:, 0] >= rb[0]) & (rbars[:, 1] >= rb[1]))),
                         median_rbar=float(np.nanmedian(rbars)))
    r = out["random"]
    print(f"random ({instances}): best two >= chi {thr:.3f}: {r['frac_best_two']:.3f}; all seven: "
          f"{r['frac_all_seven']:.3f}; r-bar of psi^1, psi^2 >= Model 1's: {r['frac_rbar']:.3f}")


def derive_teleport(out, tmax=10.0, dt=0.05, syk_seeds=3):
    ts = np.round(np.arange(0, tmax + 1e-9, dt), 6)
    k28 = int(np.argmin(np.abs(ts - T_WIND)))
    side = Side(8, MODEL1)
    tw = TwoSided(side)
    res = dict(checks=dict(K_gap=tw.k_gap, HL_minus_HR_on_I=tw.hlhr_err,
                           I_mu0=max(tw.run(0.0, t, t)["I"] for t in (0.0, 1.3, 2.8, 4.1)), tau_err=tw.tau_err))
    sym = {}
    for mu in (-MU, MU):
        rr = [tw.run(mu, t, t) for t in ts]
        I = [r["I"] for r in rr]
        F = [r["Favg"] for r in rr]
        sym[str(mu)] = dict(t=ts.tolist(), I=I, Favg=F, peak=peak(ts, I), Fpeak=peak(ts, F), I28=I[k28], F28=F[k28])
    res["symmetric"] = sym
    res["fixed_t0"] = {str(mu): dict(peak=peak(ts, [tw.run(mu, T_WIND, t)["I"] for t in ts])) for mu in (-MU, MU)}
    rows = []
    for pr in itertools.combinations(range(1, 8), 2):
        Im = np.array([tw.run(-MU, t, t, pr)["I"] for t in ts])
        Ip = np.array([tw.run(MU, t, t, pr)["I"] for t in ts])
        lm = [k for k in range(1, len(ts) - 1) if Im[k] >= Im[k - 1] and Im[k] > Im[k + 1] and Im[k] > 0.1 * Im.max()]
        rows.append(dict(pair=list(pr), peak_neg=peak(ts, Im), peak_pos=peak(ts, Ip), maxima=[float(ts[k]) for k in lm]))
    res["pairs"] = rows
    res["V_norm"] = {}
    for Nv in (7, 8, 10):
        tt = TwoSided(side, qN=Q_BODY * Nv)
        res["V_norm"][str(Nv)] = dict(neg=peak(ts, [tt.run(-MU, t, t)["I"] for t in ts]),
                                      pos=peak(ts, [tt.run(MU, t, t)["I"] for t in ts]))
    tp = TwoSided(Side(8, MODEL1 + H1_REPLY))
    Im, Ip = [tp.run(-MU, t, t)["I"] for t in ts], [tp.run(MU, t, t)["I"] for t in ts]
    res["H1"] = dict(neg=peak(ts, Im), pos=peak(ts, Ip), at28=(Im[k28], Ip[k28]))
    tsy = np.round(np.arange(0, tmax + 1e-9, max(dt, 0.1)), 6)
    res["syk10"] = []
    for seed in range(syk_seeds):
        tt = TwoSided(Side(10, syk_terms(10, 1.25, seed)), n_used=10)
        for pr in ((1, 2), (3, 7), (5, 9)):
            res["syk10"].append(dict(seed=seed, pair=list(pr),
                                     neg=peak(tsy, [tt.run(-MU, t, t, pr)["I"] for t in tsy]),
                                     pos=peak(tsy, [tt.run(MU, t, t, pr)["I"] for t in tsy])))
    out["teleport"] = res
    print(f"teleport: (1,2) I_PT at 2.8 {sym[str(-MU)]['I28']:.4f} (mu -12) / {sym[str(MU)]['I28']:.4f} (+12), "
          f"peak {sym[str(-MU)]['peak']}; best F_avg {sym[str(-MU)]['Fpeak']}; pairs losing the sign: "
          f"{sum(r['peak_pos'][1] >= 0.9 * r['peak_neg'][1] for r in rows)} of 21")


def cmd_derive(a):
    out = dict(inputs=dict(model1=MODEL1, H1=H1_REPLY, q=Q_BODY, beta=BETA, mu=MU, t=T_WIND, N_V=N_V))
    t0 = time.time()
    parts = a.parts or ["structure", "growth", "correlators", "winding", "random", "teleport"]
    for p in parts:
        {"structure": derive_structure, "growth": derive_growth, "correlators": derive_correlators,
         "winding": derive_winding, "random": derive_random, "teleport": derive_teleport}[p](out)
    out["seconds"] = round(time.time() - t0, 1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
        print(f"wrote {a.out} ({out['seconds']} s)")


# ================================================================== circuit: Pauli strings and encodings
_PM = {("X", "Y"): (1j, "Z"), ("Y", "X"): (-1j, "Z"), ("Y", "Z"): (1j, "X"), ("Z", "Y"): (-1j, "X"),
       ("Z", "X"): (1j, "Y"), ("X", "Z"): (-1j, "Y")}


class PS:
    """coef * prod_q ops[q] (qubit q = bit q of the amplitude index, as PyQrack and Qiskit)."""

    def __init__(self, coef=1.0, ops=None):
        self.c, self.ops = complex(coef), dict(ops or {})

    def __mul__(self, o):
        c, ops = self.c * o.c, dict(self.ops)
        for q, b in o.ops.items():
            a_ = ops.get(q)
            if a_ is None:
                ops[q] = b
            elif a_ == b:
                del ops[q]
            else:
                ph, r = _PM[(a_, b)]
                c *= ph
                ops[q] = r
        return PS(c, ops)

    def scale(self, k):
        return PS(self.c * k, self.ops)

    def relabel(self, m):
        return PS(self.c, {m[q]: p for q, p in self.ops.items()})

    def matrix(self, n):
        mm = {"X": PX, "Y": PY, "Z": PZ}
        out = np.array([[1.0 + 0j]])
        for q in range(n - 1, -1, -1):
            out = np.kron(out, mm[self.ops[q]] if q in self.ops else I2)
        return self.c * out


def maj_product(ms, idx):
    out = PS()
    for j in idx:
        out = out * ms[j - 1]
    return out


class Encoding:
    """split (logical L0-3 = 0-3, R0-3 = 4-7, P = 8, Q = 9) or device (Davis thesis Eq. 14.3: 7 qubits,
    L and R interleaved; P = 7, Q = 8; T is qubit 1)."""

    def __init__(self, name="split"):
        self.name = name
        if name == "device":
            Lt = {1: (1, "X"), 2: (1, "Y"), 3: (5, "X"), 4: (2, "X"), 5: (4, "X"), 6: (3, "X"), 7: (6, "X")}
            Rt = {1: (0, "X"), 2: (0, "Y"), 3: (5, "Y"), 4: (2, "Y"), 5: (4, "Y"), 6: (3, "Y"), 7: (6, "Y")}
            mk = lambda k, p: PS(1 / SQ2, {**{q: "Z" for q in range(k)}, k: p})
            self.L, self.R = [mk(*Lt[j]) for j in range(1, 8)], [mk(*Rt[j]) for j in range(1, 8)]
            self.nf, self.P, self.Q, self.Lq, self.Rq = 7, 7, 8, None, None
            self.tau = [PS(1, {0: "X"}), PS(1, {0: "Y"}), PS(1, {0: "Z"})]
        else:
            jw = lambda off, k, p, pre: PS(1 / SQ2, {**pre, **{off + q: "Z" for q in range(k)}, off + k: p})
            parL = {q: "Z" for q in range(4)}
            self.L = [jw(0, (j - 1) // 2, "XY"[(j - 1) % 2], {}) for j in range(1, 9)]
            self.R = [jw(4, (j - 1) // 2, "XY"[(j - 1) % 2], parL) for j in range(1, 9)]
            self.nf, self.P, self.Q, self.Lq, self.Rq = 8, 8, 9, [0, 1, 2, 3], [4, 5, 6, 7]
            self.tau = [PS(1, {**parL, 4: "X"}), PS(1, {**parL, 4: "Y"}), PS(1, {4: "Z"})]
        self.n = self.nf + 2

    def side(self, q):
        return "B" if self.Rq and q in self.Rq else "A"

    def roles(self):
        """logical qubit -> role name (split encoding)"""
        return {**{k: f"L{k}" for k in range(4)}, **{4 + k: f"R{k}" for k in range(4)}, 8: "P", 9: "Q"}

    def algebra(self):
        ms = self.L + self.R
        anti = max(abs((a_ * b).c + (b * a_).c) for a_, b in itertools.combinations(ms, 2))
        sq = max(abs((m * m).c - 0.5) + len((m * m).ops) for m in ms)
        return dict(majoranas=len(ms), max_anticommutator=float(anti), square_err=float(sq))

    def H(self, side):
        out = []
        for c, idx in MODEL1:
            p = maj_product(self.L if side == "L" else self.R, idx)
            out.append(PS(c * p.c.real, p.ops))
        return out

    def V_terms(self):
        return [PS(((self.L[j] * self.R[j]).scale(1j / QN)).c.real, (self.L[j] * self.R[j]).ops) for j in range(7)]

    def tfd(self, beta=BETA):
        n = self.nf
        K = sum((self.L[j] * self.R[j]).scale(1j).matrix(n) for j in range(len(self.L)))
        ev, ew = np.linalg.eigh(K)
        HL = sum(p.matrix(n) for p in self.H("L"))
        HR = sum(p.matrix(n) for p in self.H("R"))
        e, w = np.linalg.eigh(HL)
        v = (w * np.exp(-beta * (e - e.min()) / 2)) @ w.conj().T @ ew[:, 0]
        return v / np.linalg.norm(v), dict(K_gap=float(ev[1] - ev[0]),
                                           HL_minus_HR=float(np.abs((HL - HR) @ ew[:, 0]).max()))


def u3(th, ph, la):
    return np.array([[math.cos(th / 2), -np.exp(1j * la) * math.sin(th / 2)],
                     [np.exp(1j * ph) * math.sin(th / 2), np.exp(1j * (ph + la)) * math.cos(th / 2)]])


def qiskit_ops(qc, qubits):
    """Qiskit circuit -> logical ops ('u', q, M) / ('cx', c, t), transpiled to u + cx."""
    from qiskit import transpile
    tq = transpile(qc, basis_gates=["u", "cx"], optimization_level=1, seed_transpiler=1)
    out = []
    for ins in tq.data:
        qs = [qubits[tq.find_bit(x).index] for x in ins.qubits]
        nm = ins.operation.name
        if nm == "u":
            out.append(("u", qs[0], u3(*[float(x) for x in ins.operation.params])))
        elif nm == "cx":
            out.append(("cx", qs[0], qs[1]))
        elif nm not in ("barrier", "global_phase"):
            raise RuntimeError(f"unexpected gate {nm}")
    return out


class TFDPrep:
    """split: amplitude load sum_n s_n |n> on L, copy CNOTs L_k -> R_k, then W_L (x) W_R (Schmidt vectors).
    device: one 7-qubit state preparation."""

    def __init__(self, enc, beta=BETA):
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import StatePreparation, UnitaryGate
        self.enc = enc
        self.vec, self.info = enc.tfd(beta)
        if enc.name == "device":
            qc = QuantumCircuit(enc.nf)
            qc.append(StatePreparation(self.vec), range(enc.nf))
            self.full = qiskit_ops(qc, list(range(enc.nf)))
            return
        U, s, Vh = np.linalg.svd(self.vec.reshape(16, 16).T)          # M[l, r], index l + 16 r
        self.s, self.U, self.Wr = s, U, Vh.T
        p = s[s > 1e-12] ** 2
        self.entropy = float(-np.sum(p * np.log2(p)))

        def unitary(W, qs):
            qc = QuantumCircuit(4)
            qc.append(UnitaryGate(W), range(4))
            return qiskit_ops(qc, qs)
        self.WL, self.WR = unitary(U, enc.Lq), unitary(self.Wr, enc.Rq)
        self._load = {}

    def load(self, keep=16):
        if keep not in self._load:
            from qiskit import QuantumCircuit
            from qiskit.circuit.library import StatePreparation
            s = self.s.copy()
            s[keep:] = 0
            qc = QuantumCircuit(4)
            qc.append(StatePreparation((s / np.linalg.norm(s)).astype(complex)), range(4))
            self._load[keep] = qiskit_ops(qc, self.enc.Lq)
        return self._load[keep]

    def ops(self, keep=16):
        if self.enc.name == "device":
            return list(self.full)
        return list(self.load(keep)) + [("cx", l, r) for l, r in zip(self.enc.Lq, self.enc.Rq)] + self.WL + self.WR

    def entropy_kept(self, keep):
        p = self.s[:keep] ** 2
        p = p[p > 1e-14] / p.sum()
        return float(-np.sum(p * np.log2(p))), float(np.sum(self.s[:keep] ** 2))


def pauli_rot(ps, theta, enc, out, port=None):
    """exp(-i theta P). One-sided: a ladder to the last qubit, a Z rotation there. Across L|R (split): a
    ladder on each side ending at the ports (L_k, R_k) and one ('zz', L_k, R_k, theta) between them."""
    sup = sorted(ps.ops)
    for q in sup:
        if ps.ops[q] != "Z":
            out.append(("u", q, BASIS[ps.ops[q]]))
    A = [q for q in sup if enc.side(q) == "A"]
    B = [q for q in sup if enc.side(q) == "B"]
    if A and B:
        pa, pb = port
        A = sorted(A, key=lambda q: -1 if q == pa else q, reverse=True)      # ... -> pa
        B = sorted(B, key=lambda q: 99 if q == pb else q)                    # ... -> pb
    lad = [("cx", x, y) for chain in (A, B) for x, y in zip(chain, chain[1:])]
    out.extend(lad)
    if A and B:
        out.append(("zz", A[-1], B[-1], theta))
    else:
        out.append(("u", (A or B)[-1], np.diag([np.exp(-1j * theta), np.exp(1j * theta)])))
    out.extend(reversed(lad))
    for q in sup:
        if ps.ops[q] != "Z":
            out.append(("u", q, BASIS[ps.ops[q]].conj().T))


def evolve(enc, side, t, out):
    for p in enc.H(side):
        pauli_rot(PS(1, p.ops), p.c.real * t, enc, out)


def message_swap(enc, out):
    """Q <-> (psi_L^1, psi_L^2): device CZ(0,1) SWAP(1,Q) CZ(0,1); split SWAP(L0, Q)."""
    cx3 = lambda a_, b: out.extend([("cx", a_, b), ("cx", b, a_), ("cx", a_, b)])
    if enc.name == "device":
        out.extend([("u", 1, H_M), ("cx", 0, 1), ("u", 1, H_M)])
        cx3(1, enc.Q)
        out.extend([("u", 1, H_M), ("cx", 0, 1), ("u", 1, H_M)])
    else:
        cx3(0, enc.Q)


def logical_program(enc, prep, mu, t0, t1, keep=16):
    """(tfd ops, protocol ops) on logical qubits: ('u', q, M), ('cx', c, t), ('zz', a, b, theta)."""
    pre = prep.ops(keep) + [("u", enc.P, H_M), ("cx", enc.P, enc.Q)]
    post = []
    evolve(enc, "L", -t0, post)
    message_swap(enc, post)
    evolve(enc, "L", t0, post)
    for j, p in enumerate(enc.V_terms()):
        k = j // 2
        pauli_rot(PS(1, p.ops), -mu * p.c.real, enc, post,
                  port=(enc.Lq[k], enc.Rq[k]) if enc.name == "split" else None)
    evolve(enc, "R", t1, post)
    return pre, post


def logical_flat(ops):
    """('zz') expanded to cx + phase + cx (single-register runs of the unrouted program)."""
    out = []
    for o in ops:
        if o[0] == "zz":
            _, a_, b, th = o
            out += [("cx", a_, b), ("u", b, np.diag([np.exp(-1j * th), np.exp(1j * th)])), ("cx", a_, b)]
        else:
            out.append(o)
    return out


# ================================================================== Nighthawk registers and routing
def repo_of(a):
    try:
        return str(nq.find_repo(getattr(a, "repo", None)))
    except SystemExit:
        raise SystemExit("release data not found: clone https://github.com/BlueQubitDev/rcs-nighthawk next to this "
                         "script, or pass --repo / set NIGHTHAWK_REPO")


def build_field(repo, name, links="all", ghost=False):
    specs = [cf.parse_chip(t, repo) for t in FIELDS[name]]
    return cf.Field(repo, specs, arrangement=(2, 1), links=links)


def single_layout(repo):
    return cf.chip_layout(cf.parse_chip(SINGLE, repo), repo)


class Router:
    """Routes logical ops onto a register's couplers. edges: intra-chip couplers and links (global
    indices); chip_of: chip of every site. A two-qubit gate between non-neighbours on one chip moves its
    first qubit along a shortest path with SWAPs and back; a gate between chips moves both ends to the
    nearest link, crosses there, and moves back. Every op returns the layout to the role map."""

    def __init__(self, n, edges, chip_of, links):
        self.n, self.chip_of = n, list(chip_of)
        self.adj = {q: set() for q in range(n)}
        for u, v in edges:
            if self.chip_of[u] == self.chip_of[v]:
                self.adj[u].add(v)
                self.adj[v].add(u)
        self.links = [(u, v) if self.chip_of[u] < self.chip_of[v] else (v, u) for u, v in links]
        self.ops = []
        self.stats = dict(cz_chip=0, cz_link=0, swaps=0)

    def path(self, a_, b):
        prev, dq = {a_: None}, deque([a_])
        while dq:
            x = dq.popleft()
            if x == b:
                break
            for y in sorted(self.adj[x]):
                if y not in prev:
                    prev[y] = x
                    dq.append(y)
        if b not in prev:
            raise RuntimeError(f"no path {a_} -> {b} on its chip")
        p = [b]
        while prev[p[-1]] is not None:
            p.append(prev[p[-1]])
        return p[::-1]

    def m(self, q, M):
        self.ops.append(("m", q, np.asarray(M, dtype=complex)))

    def cz(self, a_, b):
        link = self.chip_of[a_] != self.chip_of[b]
        self.stats["cz_link" if link else "cz_chip"] += 1
        self.ops.append(("cz", a_, b))

    def cx_adj(self, c, t):
        self.m(t, H_M)
        self.cz(c, t)
        self.m(t, H_M)

    def swap_adj(self, a_, b):
        self.stats["swaps"] += 1
        self.cx_adj(a_, b)
        self.cx_adj(b, a_)
        self.cx_adj(a_, b)

    def _walk(self, p):
        for x, y in zip(p, p[1:]):
            self.swap_adj(x, y)

    def _unwalk(self, p):
        for x, y in reversed(list(zip(p, p[1:]))):
            self.swap_adj(x, y)

    def two(self, a_, b, fn):
        """fn(u, v) on neighbours u, v holding the contents of a_, b."""
        if self.chip_of[a_] == self.chip_of[b]:
            if b in self.adj[a_]:
                return fn(a_, b)
            p = self.path(a_, b)
            self._walk(p[:-1])
            fn(p[-2], b)
            self._unwalk(p[:-1])
            return
        best = None
        for u, v in self.links:
            if self.chip_of[u] == self.chip_of[a_] and self.chip_of[v] == self.chip_of[b]:
                uu, vv = u, v
            elif self.chip_of[v] == self.chip_of[a_] and self.chip_of[u] == self.chip_of[b]:
                uu, vv = v, u
            else:
                continue
            pa, pb = self.path(a_, uu), self.path(b, vv)
            if best is None or len(pa) + len(pb) < len(best[0]) + len(best[1]):
                best = (pa, pb)
        if best is None:
            raise RuntimeError(f"no link between the chips of {a_} and {b}")
        pa, pb = best
        self._walk(pa)
        self._walk(pb)
        fn(pa[-1], pb[-1])
        self._unwalk(pb)
        self._unwalk(pa)

    def route(self, ops, role_of):
        for o in ops:
            if o[0] == "u":
                self.m(role_of[o[1]], o[2])
            elif o[0] == "cx":
                self.two(role_of[o[1]], role_of[o[2]], self.cx_adj)
            elif o[0] == "zz":
                th = o[3]

                def zz(u, v, th=th):
                    self.cx_adj(u, v)
                    self.m(v, np.diag([np.exp(-1j * th), np.exp(1j * th)]))
                    self.cx_adj(u, v)
                self.two(role_of[o[1]], role_of[o[2]], zz)
            else:
                raise RuntimeError(o)
        return self


def fuse(ops, n):
    """Merge consecutive single-qubit gates per qubit; drop identities (up to phase)."""
    pend = [None] * n
    out = []

    def flush(q):
        M = pend[q]
        pend[q] = None
        if M is None or (abs(M[0, 1]) < 1e-12 and abs(M[1, 0]) < 1e-12 and abs(M[0, 0] - M[1, 1]) < 1e-12):
            return
        out.append(("m", q, M))
    for o in ops:
        if o[0] == "m":
            pend[o[1]] = o[2] if pend[o[1]] is None else o[2] @ pend[o[1]]
        else:
            flush(o[1])
            flush(o[2])
            out.append(o)
    for q in range(n):
        flush(q)
    return out


def role_map(enc):
    return {q: ROLE[r] for q, r in enc.roles().items()}


def routed_program(enc, prep, mu, t0, t1, n, edges, chip_of, links, keep=16):
    pre, post = logical_program(enc, prep, mu, t0, t1, keep)
    r = Router(n, edges, chip_of, links).route(pre + post, role_map(enc))
    return fuse(r.ops, n), r.stats


def field_geometry(f):
    return f.n, f.edges, f.chip_of, [(u, v) for u, v, _ in f.links]


def single_geometry(lay):
    edges = sorted({tuple(sorted(e)) for es in lay.matchings.values() for e in es})
    return lay.n, edges, [0] * lay.n, []


def observables(enc_P, tau, idx):
    """16 strings sigma_P^i tau^j relabelled to register indices."""
    sig = [PS(), PS(1, {enc_P: "X"}), PS(1, {enc_P: "Y"}), PS(1, {enc_P: "Z"})]
    return [[(sig[i] * ([PS()] + tau)[j]).relabel(idx) for j in range(4)] for i in range(4)]


def register_obs(enc):
    """Observables on field / single-register indices (split encoding)."""
    return observables(enc.P, enc.tau, role_map(enc))


# ================================================================== engines
def pauli_expect(ket, O):
    """<ket| O |ket> for a Pauli string without forming its matrix: P|x> = phase(x) |x ^ flip>."""
    idx = np.arange(ket.size, dtype=np.int64)
    flip = sum(1 << q for q, p in O.ops.items() if p in "XY")
    zm = sum(1 << q for q, p in O.ops.items() if p in "YZ")
    ny = sum(1 for p in O.ops.values() if p == "Y")
    bits = idx & zm
    par = np.zeros(ket.size, dtype=np.int64)
    while zm:
        par ^= bits & 1
        bits >>= 1
        zm >>= 1
    phase = (1j ** ny) * (1 - 2 * par)
    return complex(O.c * np.vdot(ket[idx ^ flip], phase * ket))


def exact_correlators(ket, obs, n):
    return np.array([[float(np.real(pauli_expect(ket, O))) for O in row] for row in obs])


def zxz(M):
    """Angles (phi, theta, lam) with nq.gate(angles) equal to M up to a global phase (the loom's 'g' op)."""
    th, ph, la = nq.u_angles(M)
    for cand in ((la - math.pi / 2, th, ph + math.pi / 2), (la + math.pi / 2, th, ph - math.pi / 2),
                 (la + math.pi / 2, -th, ph - math.pi / 2), (la - math.pi / 2, -th, ph + math.pi / 2)):
        G = nq.gate(cand)
        k = np.argmax(np.abs(M.ravel()))
        if abs(G.ravel()[k]) > 1e-9 and np.allclose(G * (M.ravel()[k] / G.ravel()[k]), M, atol=1e-9):
            return tuple(float(x) for x in cand)
    raise RuntimeError("no ZXZ angles found")


def run_exact(enc, prep, mu, t0, t1, a, geom):
    ops, stats = routed_program(enc, prep, mu, t0, t1, *geom)
    n = geom[0]
    eng = nq.ExactEngine(n, True)
    for o in ops:
        eng.g1(o[1], o[2]) if o[0] == "m" else eng.cz(o[1], o[2])
    ket = np.array(eng.sim.out_ket(), dtype=complex)
    r = metrics(exact_correlators(ket, register_obs(enc), n))
    r.update(stats)
    return r


def run_logical_exact(enc, prep, mu, t0, t1):
    """The unrouted logical program on nq.ExactEngine (device or split encoding)."""
    pre, post = logical_program(enc, prep, mu, t0, t1)
    eng = nq.ExactEngine(enc.n, True)
    for o in logical_flat(pre + post):
        if o[0] == "u":
            eng.g1(o[1], o[2])
        else:
            eng.g1(o[2], H_M)
            eng.cz(o[1], o[2])
            eng.g1(o[2], H_M)
    ket = np.array(eng.sim.out_ket(), dtype=complex)
    return metrics(exact_correlators(ket, observables(enc.P, enc.tau, {q: q for q in range(enc.n)}), enc.n))


def chain_dist(sim, qs, eps=1e-7):
    if not qs:
        return {(): 1.0}
    p1, out = float(sim.prob(qs[0])), {}
    for b in (0, 1):
        pb = p1 if b else 1.0 - p1
        if pb < eps:
            continue
        c = sim.clone()
        c.force_m(qs[0], bool(b))
        for k, v in chain_dist(c, qs[1:], eps).items():
            out[(b,) + k] = out.get((b,) + k, 0.0) + pb * v
    return out


def sampled_correlators(sim, obs, site):
    """<O> for the 16 strings from 9 measurement settings: on a clone of the final state rotate P and T
    into Z, read the joint distribution of every qubit a setting needs by the chain rule (prob, force_m)."""
    Tm = np.zeros((4, 4))
    Tm[0, 0] = 1.0
    acc = {}
    for i, j in itertools.product((1, 2, 3), (1, 2, 3)):
        O = obs[i][j]
        c = sim.clone()
        for q, p in O.ops.items():
            if p != "Z":
                c.u(site[q], *nq.u_angles(BASIS[p]))
        qs = sorted(O.ops)
        d = chain_dist(c, [site[q] for q in qs])
        Pq = [q for q in qs if q in obs[i][0].ops]
        par = lambda k, sub: np.prod([1 - 2 * k[qs.index(q)] for q in sub]) if sub else 1.0
        Tm[i, j] = float(np.real(O.c)) * sum(v * par(k, qs) for k, v in d.items())
        acc.setdefault(("P", i), []).append(sum(v * par(k, Pq) for k, v in d.items()))
        acc.setdefault(("T", j), []).append(float(np.real(obs[0][j].c)) *
                                             sum(v * par(k, [q for q in qs if q not in Pq]) for k, v in d.items()))
    for i in (1, 2, 3):
        Tm[i, 0] = float(np.mean(acc[("P", i)]))
        Tm[0, i] = float(np.mean(acc[("T", i)]))
    return Tm


def ace_args(lay, placement):
    a = argparse.Namespace(cpu=True, ace_gpu=False, ace_host_pointer=False, ace_gpu_shots=False, ace_layout="grid",
                           ace_tiling=False, ace_max_width=None, lrc="auto", lrr="auto", ace_torus=False,
                           ace_transpile=False, ace_width=None, ace_exclude=None)
    return a


class AceRegister:
    """nq.AceEngine on the single register (rect:0-3,0-3, 16 sites), lrc/lrr from nq.two_patch_config; the
    placement replaces eng.idx: grid (the lattice as ACE lays it out), tiling (nq.ace_tiling v3 weighted by
    this circuit's CZ couplers), lr (L, P, Q in one patch's bulk, R in the other's), seam-r (R on the
    boundary sites)."""

    def __init__(self, repo, couplers=None):
        self.lay = single_layout(repo)
        self.n = self.lay.n
        self.args = ace_args(self.lay, "grid")
        eng = nq.AceEngine(self.lay, self.n, self.args)
        self.size, self.lrc, self.lrr, self.torus = eng.size, eng.lrc, eng.lrr, eng.torus
        self.row = nq.layout_row(self.size, self.lrc, self.lrr, self.torus, eng.opts)
        self.grid = list(eng.idx)
        self.widths = eng.widths
        sids = [frozenset(s for s, _ in u) for u in self.row["unpack"]]
        self.sids = sids
        self.boundary = [x for x in range(self.size) if len(sids[x]) > 1]
        bulk = {}
        for x in range(self.size):
            if len(sids[x]) == 1:
                bulk.setdefault(next(iter(sids[x])), []).append(x)
        (pa, A), (pb, B) = sorted(bulk.items())[:2]
        self.bulkA, self.bulkB = A, B
        self.couplers = couplers or []
        del eng

    def placement(self, name):
        if name == "grid":
            return list(self.grid)
        if name == "tiling":
            return list(nq.ace_tiling(self.row, self.grid, self.couplers, self.size, 3))
        roles_A = [ROLE[r] for r in ("L0", "L1", "L2", "L3", "P", "Q")]
        roles_B = [ROLE[f"R{k}"] for k in range(4)]
        site = {}
        if name == "lr":
            site.update(zip(roles_A, self.bulkA))
            site.update(zip(roles_B, self.bulkB))
        elif name == "seam-r":
            site.update(zip(roles_A, self.bulkA))
            site.update(zip(roles_B, self.boundary))
        else:
            raise SystemExit(f"unknown placement {name}")
        free = [x for x in range(self.size) if x not in site.values()]
        for q in range(self.n):
            if q not in site:
                site[q] = free.pop(0)
        return [site[q] for q in range(self.n)]

    def stats(self, site):
        return nq.placement_stats(self.row, site, self.couplers)

    def engine(self, site):
        eng = nq.AceEngine(self.lay, self.n, self.args)
        eng.idx = list(site)
        return eng


def circuit_couplers(ops):
    """The routed circuit's CZs as a multiset of (u, v), u < v: the weights nighthawk's tiling minimises."""
    return [tuple(sorted((o[1], o[2]))) for o in ops if o[0] == "cz"]


def run_ace(enc, prep, mu, t0, t1, a, reg, site):
    ops, stats = routed_program(enc, prep, mu, t0, t1, *single_geometry(reg.lay))
    eng = reg.engine(site)
    for o in ops:
        eng.g1(o[1], o[2]) if o[0] == "m" else eng.cz(o[1], o[2])
    r = metrics(sampled_correlators(eng.sim, register_obs(enc), eng.idx))
    r.update(stats)
    return r


# ---------------------------------------------------------------- cloverfield
def start_chips(field, mode, repo):
    """One isolated worker per chip (nighthawk_cloverfield's ChipProc, CPU, exact chips), ghosts as the
    field's ghost plan places them."""
    plan = field.ghost_plan() if mode == "ghost" else [[] for _ in field.specs]
    env = cf.worker_env_for(argparse.Namespace(cpu=True), None, cpu=True)
    chips = []
    for c, spec in enumerate(field.specs):
        opts = dict(backend="exact", cpu=True, ace_gpu=False, ghost_anchors=[anc for _, anc, _ in plan[c]],
                    ghost_sites=[s for _, _, s in plan[c]])
        chips.append(cf.ChipProc(c, spec, repo, opts, env, nq.THETA_DIST["mode"]))
    return chips


def loom_readout(loom, field, enc, chip, holder):
    """Correlators read on one chip: every qubit of each string is held there (holder: global -> local),
    its parity collected onto P's copy with CNOTs (H CZ H), read as a marginal, the collection undone."""
    f = loom.f
    obs = register_obs(enc)
    Tm = np.zeros((4, 4))
    Tm[0, 0] = 1.0
    for i, j in itertools.product(range(4), range(4)):
        if i == 0 and j == 0:
            continue
        O = obs[i][j]
        qs = sorted(O.ops)
        if any(q not in holder for q in qs):
            return None
        do = []
        for q in qs:
            if O.ops[q] != "Z":
                do.append(("g", holder[q], zxz(BASIS[O.ops[q]]), False))
        col = holder[qs[0]]
        for q in qs[1:]:
            t_ = holder[q]
            do += [("g", col, zxz(H_M), False), ("cz", t_, col), ("g", col, zxz(H_M), False)]
        undo = []
        for op in reversed(do):
            undo.append(op if op[0] == "cz" else ("g", op[1], op[2], not op[3]))
        queues = [[] for _ in loom.chips]
        queues[chip] = do
        res = loom._flush(queues, [[col] if c == chip else [] for c in range(len(loom.chips))])
        p1 = res[chip][0]
        queues[chip] = undo
        loom._flush(queues, [[] for _ in loom.chips])
        Tm[i, j] = float(np.real(O.c)) * (1 - 2 * p1)
    return Tm


def product_readout(loom, field, enc):
    """Correlators as products over chips of each chip's part of the string (the cloverfield estimator:
    correlation between the chips' outcomes is dropped)."""
    obs = register_obs(enc)
    Tm = np.zeros((4, 4))
    Tm[0, 0] = 1.0
    for i, j in itertools.product(range(4), range(4)):
        if i == 0 and j == 0:
            continue
        O = obs[i][j]
        val = float(np.real(O.c))
        for c in range(len(loom.chips)):
            part = PS(1, {q: p for q, p in O.ops.items() if field.chip_of[q] == c})
            if not part.ops:
                continue
            val *= _chip_parity(loom, c, part, {q: field.local[q] for q in part.ops})
        Tm[i, j] = val
    return Tm


def _chip_parity(loom, chip, O, holder):
    qs = sorted(O.ops)
    do = []
    for q in qs:
        if O.ops[q] != "Z":
            do.append(("g", holder[q], zxz(BASIS[O.ops[q]]), False))
    col = holder[qs[0]]
    for q in qs[1:]:
        do += [("g", col, zxz(H_M), False), ("cz", holder[q], col), ("g", col, zxz(H_M), False)]
    undo = [op if op[0] == "cz" else ("g", op[1], op[2], not op[3]) for op in reversed(do)]
    queues = [[] for _ in loom.chips]
    queues[chip] = do
    p1 = loom._flush(queues, [[col] if c == chip else [] for c in range(len(loom.chips))])[chip][0]
    queues[chip] = undo
    loom._flush(queues, [[] for _ in loom.chips])
    return 1 - 2 * p1


def run_loom(enc, prep, mu, t0, t1, a, field, mode, repo, chips=None):
    ops, stats = routed_program(enc, prep, mu, t0, t1, *field_geometry(field))
    own = chips is None
    chips = chips or start_chips(field, mode, repo)
    try:
        loom = cf.Loom(field, chips, mode=mode)
        srng = np.random.default_rng([nq.BASE_SEED, int(round(1000 * t1)), int(round(mu)) + 1000, 17])
        queues = [[("reset",)] for _ in chips]
        for o in ops:
            if o[0] == "m":
                loom._emit1(queues, o[1], "g", zxz(o[2]), False)
            else:
                loom._layer([(o[1], o[2])], None, queues, srng)
        loom._flush(queues, [[] for _ in chips])
        out = dict(stats, round_trips=loom.round_trips,
                   ghost_drift=float(np.mean(loom.drift)) if loom.drift else None)
        Tp = product_readout(loom, field, enc)
        out["product"] = metrics(Tp)
        if mode == "ghost":                               # chip A's view: P, L and the ghost of R0
            gh = {q: field.local[q] for q in range(field.n) if field.chip_of[q] == 0}
            for c_, l_ in loom.ghost_of.get(ROLE["R0"], []):
                if c_ == 0:
                    gh[ROLE["R0"]] = l_
            Tg = loom_readout(loom, field, enc, 0, gh)
            if Tg is not None:
                out["ghost_view"] = metrics(Tg)
        main = out.get("ghost_view", out["product"])
        out.update(I=main["I"], Favg=main["Favg"], min_eig=main["min_eig"],
                   view="ghost (chip A)" if "ghost_view" in out else "product")
        return out
    finally:
        if own:
            for ch in chips:
                ch.close()


# ---------------------------------------------------------------- knit (exact loom)
def run_knit(enc, prep, mu, t0, t1, a, keep):
    """Chip A = L, P, Q; chip B = R; each one nq.ExactEngine with a branch register (4 Schmidt-index qubits
    copied onto its own side, one qubit per coupling ZZ that puts Z on its port) in |+>, so one run per chip
    holds every branch with consistent phases (Qrack does not keep global phases across runs). Branch
    weights s_n prod_j (cos th_j | -i sin th_j) are applied afterwards; correlators are double sums over
    branch pairs of chip-A and chip-B overlaps."""
    _, post = logical_program(enc, prep, mu, t0, t1, keep)
    zz = [o for o in post if o[0] == "zz"]
    nl = len(zz)
    s = prep.s.copy()
    s[keep:] = 0
    s /= np.linalg.norm(s)
    chips = {"A": enc.Lq + [enc.P, enc.Q], "B": enc.Rq}
    kets = {}
    for side, qubits in chips.items():
        own = enc.Lq if side == "A" else enc.Rq
        loc = {q: i for i, q in enumerate(qubits)}
        nq_ = len(qubits)
        nreg, lreg = [nq_ + k for k in range(4)], [nq_ + 4 + j for j in range(nl)]
        eng = nq.ExactEngine(nq_ + 4 + nl, True)
        for r in nreg + lreg:
            eng.g1(r, H_M)
        for k in range(4):
            eng.g1(loc[own[k]], H_M)
            eng.cz(nreg[k], loc[own[k]])
            eng.g1(loc[own[k]], H_M)
        body = (prep.WL + [("u", enc.P, H_M), ("cx", enc.P, enc.Q)]) if side == "A" else list(prep.WR)
        j = 0
        for o in body + post:
            if o[0] == "zz":
                eng.cz(lreg[j], loc[o[1] if side == "A" else o[2]])
                j += 1
            elif o[0] == "u":
                if enc.side(o[1]) == side:
                    eng.g1(loc[o[1]], o[2])
            elif enc.side(o[1]) == side:
                eng.g1(loc[o[2]], H_M)
                eng.cz(loc[o[1]], loc[o[2]])
                eng.g1(loc[o[2]], H_M)
        kets[side] = np.array(eng.sim.out_ket(), dtype=complex).reshape(2 ** nl * 16, 2 ** nq_)
    th = [o[3] for o in zz]
    w = np.zeros(2 ** nl * 16, dtype=complex)
    for path in range(2 ** nl):
        f_ = 1.0 + 0j
        for j in range(nl):
            f_ *= (-1j * math.sin(th[j])) if (path >> j) & 1 else math.cos(th[j])
        w[path * 16: path * 16 + 16] = s * f_
    live = np.abs(w) > 1e-14
    KA, KB, w = kets["A"][live], kets["B"][live], w[live]
    locA = {q: i for i, q in enumerate(chips["A"])}
    locB = {q: i for i, q in enumerate(chips["B"])}
    obs = observables(enc.P, enc.tau, {q: q for q in range(enc.n)})
    Tm = np.zeros((4, 4))
    GB = {}
    for i in range(4):
        for j in range(4):
            O = obs[i][j]
            oa = PS(O.c, {locA[q]: p for q, p in O.ops.items() if enc.side(q) == "A"}).matrix(len(locA))
            key = tuple(sorted((q, p) for q, p in O.ops.items() if enc.side(q) == "B"))
            if key not in GB:
                GB[key] = KB.conj() @ PS(1, {locB[q]: p for q, p in key}).matrix(len(locB)) @ KB.T
            Tm[i, j] = float(np.real(np.einsum("b,c,bc,bc->", w.conj(), w, KA.conj() @ oa @ KA.T, GB[key])))
    norm = float(np.real(np.einsum("b,c,bc,bc->", w.conj(), w, KA.conj() @ KA.T, KB.conj() @ KB.T)))
    r = metrics(Tm / norm)
    S_, wk = prep.entropy_kept(keep)
    r.update(branches=int(live.sum()), schmidt_kept=int((s > 1e-12).sum()), entanglement_kept=S_, weight_kept=wk,
             chip_qubits=[len(locA) + 4 + nl, len(locB) + 4 + nl])
    return r


# ================================================================== commands
def times(a):
    return [round(x, 6) for x in np.arange(a.tmin, a.tmax + 1e-9, a.dt)]


def cmd_plan(a):
    repo = repo_of(a)
    enc = Encoding("split")
    prep = TFDPrep(enc)
    for name in FIELDS:
        f = build_field(repo, name, a.links)
        ops, st = routed_program(enc, prep, -MU, T_WIND, T_WIND, *field_geometry(f))
        print(f"field {name}: chips {[s['name'] for s in f.specs]} ({f.chip_n} sites), links {[(u, v) for u, v, _ in f.links]}")
        print(f"  roles: Q {ROLE['Q']}, P {ROLE['P']}, L {[ROLE[f'L{k}'] for k in range(4)]}, R {[ROLE[f'R{k}'] for k in range(4)]}; "
              f"ghost plan {[[(q, anc) for q, anc, _ in g] for g in f.ghost_plan()]}")
        print(f"  routed circuit (t = 2.8): {st['cz_chip']} CZ inside chips, {st['cz_link']} on links, {st['swaps']} SWAPs, "
              f"{sum(1 for o in ops if o[0] == 'm')} single-qubit gates after fusion")
    reg = AceRegister(repo)
    ops, st = routed_program(enc, prep, -MU, T_WIND, T_WIND, *single_geometry(reg.lay))
    reg.couplers = circuit_couplers(ops)
    print(f"single register {SINGLE}: {reg.n} sites, ACE lrc {reg.lrc} lrr {reg.lrr}, widths {reg.widths}, "
          f"boundary {reg.boundary}, bulk A {reg.bulkA}, bulk B {reg.bulkB}; routed: {st['cz_chip']} CZ, {st['swaps']} SWAPs")
    print("  placement   circuit CZs exact / replica / cross   seam sites used   simulators")
    for p in ("grid", "tiling", "lr", "seam-r"):
        site = reg.placement(p)
        s_ = reg.stats(site)
        print(f"  {p:10s}  {s_['exact']:5d} / {s_['replica']:5d} / {s_['cross']:5d}              {s_['seam_used']}               {s_['sims_used']}")


def cmd_run(a):
    repo = repo_of(a) if a.backend in ("exact", "ace", "loom") else None
    enc = Encoding("split")
    prep = TFDPrep(enc)
    done = set()
    if a.out and os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["backend"], r.get("field"), r.get("variant"), str(r.get("links")), r["mu"], r["t0"], r["t1"],
                      r.get("rep", 0)))
    fh = None
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        fh = open(a.out, "a")
    reg = None
    if a.backend == "ace":
        reg = AceRegister(repo)
        ops, _ = routed_program(enc, prep, -MU, T_WIND, T_WIND, *single_geometry(reg.lay))
        reg.couplers = circuit_couplers(ops)
        variants = a.placement
    elif a.backend == "loom":
        variants = a.loom
    elif a.backend == "knit":
        variants = [f"keep{k}" for k in a.keep]
    else:
        variants = [None]
    fields = a.field if a.backend in ("exact", "loom") else [None]
    reps = a.reps if a.backend == "ace" else 1
    for fname, var, rep in itertools.product(fields, variants, range(reps)):
        field = build_field(repo, fname, a.links) if fname else None
        site = reg.placement(var) if a.backend == "ace" else None
        chips = start_chips(field, var, repo) if a.backend == "loom" else None
        try:
            for mu in a.mu:
                for t in times(a):
                    t0, t1 = (a.t0, t) if a.t0 is not None else (t, t)
                    key = (a.backend, fname, var, str(a.links if a.backend in ("exact", "loom") else None), mu, t0, t1, rep)
                    if key in done:
                        continue
                    tic = time.time()
                    if a.backend == "exact":
                        r = run_exact(enc, prep, mu, t0, t1, a, field_geometry(field))
                    elif a.backend == "ace":
                        r = run_ace(enc, prep, mu, t0, t1, a, reg, site)
                        r.update(ace=dict(size=reg.size, lrc=reg.lrc, lrr=reg.lrr, widths=reg.widths),
                                 placement_stats=reg.stats(site))
                    elif a.backend == "loom":
                        r = run_loom(enc, prep, mu, t0, t1, a, field, var, repo, chips)
                    else:
                        r = run_knit(enc, prep, mu, t0, t1, a, int(var[4:]))
                    rec = dict(backend=a.backend, field=fname, variant=var, mu=mu, t0=t0, t1=t1, rep=rep,
                               protocol="fixed_t0" if a.t0 is not None else "symmetric",
                               links=a.links if a.backend in ("exact", "loom") else None,
                               seconds=round(time.time() - tic, 2), **r)
                    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()
                                      if k not in ("ace", "product", "ghost_view", "placement_stats")}), flush=True)
                    if fh:
                        fh.write(json.dumps(rec, default=float) + "\n")
                        fh.flush()
        finally:
            for ch in chips or []:
                ch.close()


def cmd_table(a):
    recs = []
    for f in a.files:
        recs += [json.loads(x) for x in open(f)]
    groups = {}
    for r in recs:
        mode = "sym" if r.get("protocol", "symmetric") == "symmetric" else f"t0={r['t0']}"
        groups.setdefault((r["backend"], str(r.get("field")), str(r.get("variant")), str(r.get("links")), mode), []).append(r)
    print("backend  field   variant  links t0     reps  I(-12)@2.8        I(+12)@2.8        peak I(-12) @t   "
          "peak I(+12) @t   best F(-12)  extra")
    for key, rs in sorted(groups.items()):
        curve = {}
        for r in rs:
            curve.setdefault((r["mu"], r["t1"]), []).append(r)

        def series(sign):
            return sorted((t, np.mean([x["I"] for x in v]), np.std([x["I"] for x in v]), np.mean([x["Favg"] for x in v]))
                          for (mu, t), v in curve.items() if np.sign(mu) == sign)
        neg, pos = series(-1), series(1)
        at = lambda S: next(((m, sd) for tt, m, sd, _ in S if abs(tt - 2.8) < 1e-6), (float("nan"), 0.0))
        pk = lambda S: max(S, key=lambda x: x[1]) if S else (float("nan"), float("nan"))
        n28, p28 = at(neg), at(pos)
        extra = ""
        if rs[0].get("ghost_drift") is not None:
            extra = f"ghost drift {np.mean([r['ghost_drift'] for r in rs if r.get('ghost_drift') is not None]):.4f}"
        if rs[0].get("product") and rs[0].get("ghost_view"):
            pr = [r["product"]["I"] for r in rs if abs(r["t1"] - 2.8) < 1e-6 and r["mu"] < 0]
            extra += f"; product view I(-12)@2.8 {np.mean(pr):.4f}" if pr else ""
        b, f, v, l_, mode = key
        print(f"{b:8s} {f:7s} {v:8s} {l_:5s} {mode:6s} {max(len(x) for x in curve.values()):4d}  "
              f"{n28[0]:.4f} +- {n28[1]:.4f}  {p28[0]:.4f} +- {p28[1]:.4f}  {pk(neg)[1]:.4f} {pk(neg)[0]:4.1f}      "
              f"{pk(pos)[1]:.4f} {pk(pos)[0]:4.1f}      {max((x[3] for x in neg), default=float('nan')):.4f}       {extra}")


def cmd_aceprobe(a):
    reg = AceRegister(repo_of(a))
    A, B, S = reg.bulkA, reg.bulkB, reg.boundary
    print(f"ACE register {reg.size} sites ({SINGLE}), lrc {reg.lrc}, lrr {reg.lrr}: bulk A {A}, bulk B {B}, "
          f"boundary {S}, widths {reg.widths}")
    print(f"  pair                    sites    <ZZ>             <XX>             (exact Bell: 1, 1; {a.reps} runs, mean +- sd)")
    for name, x, y in (("bulk A - bulk A", A[0], A[1]), ("bulk B - bulk B", B[0], B[1]), ("bulk A - bulk B", A[0], B[0]),
                       ("bulk A - boundary", A[0], S[0]), ("boundary - bulk B", S[1], B[1]), ("boundary - boundary", S[0], S[1])):
        out = {"Z": [], "X": []}
        for _, basis in itertools.product(range(a.reps), "ZX"):
            eng = reg.engine(list(range(reg.size)))
            eng.g1(x, H_M)
            eng.g1(y, H_M)
            eng.cz(x, y)
            eng.g1(y, H_M)
            if basis == "X":
                eng.g1(x, H_M)
                eng.g1(y, H_M)
            d = chain_dist(eng.sim, [x, y])
            out[basis].append(sum(v * (1 - 2 * (k[0] ^ k[1])) for k, v in d.items()))
        print(f"  {name:22s} {x:2d},{y:2d}    {np.mean(out['Z']):+.4f} +- {np.std(out['Z']):.4f}  "
              f"{np.mean(out['X']):+.4f} +- {np.std(out['X']):.4f}")


def cmd_selftest(a):
    repo = repo_of(a)
    ok = True

    def check(name, good, detail):
        nonlocal ok
        ok &= bool(good)
        print(f"  {name}: {detail}  {'ok' if good else 'FAIL'}")
    print("== reference and encodings ==")
    tw = TwoSided(Side(8, MODEL1))
    ref = {mu: tw.run(mu, T_WIND, T_WIND)["I"] for mu in (-MU, MU)}
    check("reference", tw.hlhr_err < 1e-12 and tw.tau_err < 1e-12,
          f"I_PT(t = 2.8) {ref[-MU]:.5f} / {ref[MU]:.5f}; H_L|I> = H_R|I> to {tw.hlhr_err:.1e}")
    for name in ("device", "split"):
        enc = Encoding(name)
        al = enc.algebra()
        check(f"{name} Majoranas", al["max_anticommutator"] < 1e-12 and al["square_err"] < 1e-12,
              f"{al['majoranas']}, max |{{a,b}}| {al['max_anticommutator']:.1e}")
        r = run_logical_exact(enc, TFDPrep(enc), -MU, T_WIND, T_WIND)
        check(f"{name} logical circuit on nq.ExactEngine", abs(r["I"] - ref[-MU]) < 1e-4, f"I_PT {r['I']:.5f}")
    print("== Nighthawk registers ==")
    enc = Encoding("split")
    prep = TFDPrep(enc)
    M = np.array([[0.6, 0.8j], [0.8j, 0.6]]) @ np.diag([1, np.exp(0.7j)])
    check("ZXZ angles of the loom's 'g' op", True, f"{zxz(M)}")
    for name in FIELDS:
        f = build_field(repo, name)
        r = run_exact(enc, prep, -MU, T_WIND, T_WIND, a, field_geometry(f))
        check(f"field {name} routed on nq.ExactEngine", abs(r["I"] - ref[-MU]) < 1e-4,
              f"I_PT {r['I']:.5f}; {r['cz_chip']} chip CZ, {r['cz_link']} link CZ, {r['swaps']} SWAPs")
    reg = AceRegister(repo)
    lay = reg.lay
    eng_ops, st = routed_program(enc, prep, -MU, T_WIND, T_WIND, *single_geometry(lay))
    eng = nq.ExactEngine(lay.n, True)
    for o in eng_ops:
        eng.g1(o[1], o[2]) if o[0] == "m" else eng.cz(o[1], o[2])
    ket = np.array(eng.sim.out_ket(), dtype=complex)
    Tm = exact_correlators(ket, register_obs(enc), lay.n)
    Ts = sampled_correlators(eng.sim, register_obs(enc), list(range(lay.n)))
    check("single register routed, state vector vs chain-rule estimator",
          abs(metrics(Tm)["I"] - ref[-MU]) < 1e-4 and abs(metrics(Ts)["I"] - ref[-MU]) < 1e-4,
          f"I_PT {metrics(Tm)['I']:.5f} / {metrics(Ts)['I']:.5f}")
    print("== cloverfield ==")
    f = build_field(repo, "wide")
    r = run_loom(enc, prep, -MU, T_WIND, T_WIND, a, f, "ghost", repo)
    check("wide field, ghost loom (every side qubit anchored: chip A holds the whole state)",
          abs(r["ghost_view"]["I"] - ref[-MU]) < 1e-3, f"I_PT {r['ghost_view']['I']:.5f}, product view "
          f"{r['product']['I']:.5f}, drift {r['ghost_drift']}")
    r = run_loom(enc, prep, -MU, T_WIND, T_WIND, a, f, "mean", repo)
    check("wide field, mean-field loom", r["I"] < 0.01, f"I_PT {r['I']:.5f}")
    k = run_knit(enc, prep, -MU, T_WIND, T_WIND, a, 16)
    check("knit (exact cut)", abs(k["I"] - ref[-MU]) < 1e-4, f"I_PT {k['I']:.5f}, {k['branches']} branches")
    print("selftest", "PASSED" if ok else "FAILED")
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("derive")
    d.add_argument("--parts", nargs="+", choices=["structure", "growth", "correlators", "winding", "random", "teleport"])
    d.add_argument("--out", default=None)
    p = sub.add_parser("plan")
    p.add_argument("--repo", default=None)
    p.add_argument("--links", default="all")
    r = sub.add_parser("run")
    r.add_argument("--repo", default=None)
    r.add_argument("--backend", choices=["exact", "ace", "loom", "knit"], default="exact")
    r.add_argument("--field", nargs="+", choices=list(FIELDS), default=["wide"])
    r.add_argument("--links", default="all", help="cloverfield --links: all, or K per chip pair")
    r.add_argument("--placement", nargs="+", default=["grid", "tiling", "lr", "seam-r"])
    r.add_argument("--loom", nargs="+", choices=["ghost", "mean", "sample"], default=["ghost", "mean"])
    r.add_argument("--keep", type=int, nargs="+", default=[16])
    r.add_argument("--reps", type=int, default=1)
    r.add_argument("--mu", type=float, nargs="+", default=[-MU, MU])
    r.add_argument("--tmin", type=float, default=0.0)
    r.add_argument("--tmax", type=float, default=6.0)
    r.add_argument("--dt", type=float, default=0.4)
    r.add_argument("--t0", type=float, default=None)
    r.add_argument("--out", default=None)
    t = sub.add_parser("table")
    t.add_argument("files", nargs="+")
    q = sub.add_parser("aceprobe")
    q.add_argument("--repo", default=None)
    q.add_argument("--reps", type=int, default=8)
    s = sub.add_parser("selftest")
    s.add_argument("--repo", default=None)
    a = ap.parse_args()
    dict(derive=cmd_derive, plan=cmd_plan, run=cmd_run, table=cmd_table, aceprobe=cmd_aceprobe,
         selftest=cmd_selftest)[a.cmd](a)


if __name__ == "__main__":
    main()
