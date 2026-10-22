#!/usr/bin/env python3
"""
nighthawk_qrack.py -- clean-qubit PyQrack companion to arXiv:2609.28657
(BlueQubit, random-circuit sampling on IBM Nighthawk r2 / ibm_phoenix, 61 qubits).

No noise model. The simulator stands in for the QPU with ideal gates; the only
error left in any number this script prints is shot noise and, for the ace
backend, QrackAceBackend's own seam approximation. Everything else is aligned
gate-for-gate with the paper and the BlueQubitDev/rcs-nighthawk release:

  circuits   regenerated from data/layout.json exactly as rcs/circuits.py does:
             SplitMix64 hash of (seed, instance, cycle, qubit, k) -> Haar angles,
             applied as rz(phi) rx(theta) rz(lambda); colours A,B,C,D in the fixed
             order; seeds 2025 + k*1000003 (full, patched) and 2025/instance k
             (mirror); `verify` checks every one of the 233 released QASM files.
  patches    the five released K=3 and K=4 partitions (layout.json), not re-searched.
  mirror     forward d/2 cycles pseudo-patched through the five K=3 boundary sets,
             advancing one partition per cycle (rotate_every=1), then the exact
             inverse; the ten released input strings (char q -> logical qubit q);
             shots split evenly over the inputs; optional Pauli-frame gate
             twirling (64 randomisations in the paper). Twirling is a gate identity,
             so it adds no noise: on the ace backend it only randomises ACE's
             coherent seam error so the two mirror halves cannot cancel it.
  patched    Eq. (1)-(2): per-patch linear XEB normalised by its ideal XEB, product
             over patches, second-order delta-method standard error.
  combine    inverse-variance mean over instances; unweighted log10 exponential fit
             of the mirror points (analysis/fidelity_vs_depth.py).

Subcommands
  verify    regenerate all circuits and compare with data/circuits/*.qasm (no Qrack).
  hwxeb     re-score the measured ibm_phoenix bitstrings against PyQrack ideal patch
            distributions and rebuild the paper's F(d) points, fit and collision
            ratios; every number is checked against data/results/*.json.
  run       clean-qubit emulation of the released circuits (--backend exact|ace),
            scored with the same estimators; the summary sets the simulator's
            per-cycle fidelity next to the device's 0.872. --sizes sweeps the
            register (first-n truncation, the release's `e < n` convention),
            resumable per (n, point), and fits b(N) = u*N + v*CZ/cycle to 61.
            Every bitstring drawn is kept: one npz per point under
            <out>_shots/<cfg>/points, merged after each run into .../release/ in the release's data/
            layout, so BlueQubit's analysis scripts read it unchanged. --families full
            draws the 10 x 100k samples of the d36 circuit (stored, not scorable).
            --families fxeb scores the backend's samples of the unpatched forward
            circuit against an exact 2^n reference (nn_qab.py's estimator), for
            first-n truncations small enough to simulate exactly.
            --gpus 0-5 --per-gpu 3 spreads the points over 18 workers through a
            claim-file queue (QRACK_OCL_DEFAULT_DEVICE per worker); a killed worker's
            point is taken over, and the same command resumes.
  selftest  conventions and exactness checks (seconds).
  aceplan   every distinct ACE layout of the register: seams, simulator widths and
            worst-case dense memory, ranked as --ace-max-width ranks them.
  seamgap   data-only analysis of the release (numpy), unchanged.
  layoutsearch
            iterate to a register + ACE placement: one CP-SAT model picks which lattice
            sites to use (--region rect|convex|free|fixed, broken qubits disabled by
            default) and which patch bulk or seam each goes to, under the width caps,
            for each ACE geometry (--geometries SIZE:LRC:LRR[:t] ... or auto with
            --patches); --candidates diverse layouts per geometry; every candidate is
            re-checked through RegisterLayout (no coupler between patches, caps, counts),
            then scored by a short mirror run (--test-depths, --test-instances,
            --test-twirls, geometric mean of the plain-mean survival); --save-best NAME
            writes the winner to nighthawk_registers.json for --register NAME.
  danmodel  Dan Strano's ACE XEB calculator: predictions per register, or against measured records.
  shor9     one unit of Shor's 9-qubit code (--size m: [[m^2, 1, m]], Shor-16, Shor-36, ...) on an m x m
            two-patch ACE register (Claude and Dan Strano):
            logical success raw / decoded for the four logical states, seam-leader and patch-block
            layouts, against an exact reference.
  hamming   his mirror Hamming-weight ladder (run --families hamming): per register the signal
            against his calculator at d and 2d, the seam's coupler load, a fitted depth power,
            a measured / Dan grid sorted by coupler load, and his Richardson extrapolation over
            the folds. Reads files only (no Qrack): safe beside running jobs.

Backends
  exact   QrackSimulator. Patched circuits factorise, and QUnit keeps the patches
          separate, so 61-qubit patched circuits run exactly. Mirror circuits
          entangle all 61 qubits after one ABCD sweep: use --n to truncate to the
          first n logical qubits (the release's own `e < n` convention).
  ace     QrackAceBackend, noise=0. The 61 logical qubits are placed at their true
          positions on an 8x8 ACE register (rows 1-8, cols 2-9 of the device), so
          every coupler is nearest-neighbour in ACE's grid; the three dropped sites
          idle in |0>. Default is_torus=False (the device patch is not a torus).
          --ace-torus, and --geometry nnqab (nn_qab.py's rule), switch to
          is_torus=True; the run header and every record say which was used.
          --ace-max-width W: bigger patches. Of every distinct ACE layout of the
          register (all lrc, lrr; non-torus only by default, as the device patch is
          not a torus -- --ace-torus-search torus|any widens it), take the one that puts the
          fewest of this circuit's CZ couplers on a seam (the only gates ACE
          approximates) while its widest internal simulator has at most W qubits,
          i.e. the least approximation that still fits in memory. Widths are read from the
          installed QrackAceBackend itself (patch bulk + boundary replicas + crossbar),
          so the choice follows whatever your Qrack build does. `aceplan` lists them.
          --ace-tiling: ACE numbers its chunks along a folded 1D chain, so on the
          device grid a chunk is strips of different rows and most vertical couplers
          cross simulators. Tiling instead searches (deterministic annealing) for the
          placement of logical qubits on ACE sites that keeps the most couplers inside
          one simulator: compact tiles on the device grid, the same simulators and
          memory. Couplers are then no longer ACE-grid neighbours; ACE still applies
          them (exactly inside a simulator, through its seam machinery otherwise).

Environment: no CUDA. OpenCL via QRACK_OCL_DEFAULT_DEVICE, or --cpu.
For big exact states: QRACK_MAX_CPU_QB, QRACK_MAX_ALLOC_MB.
GPU isolation (--gpu-isolation auto): every GPU process sees exactly one OpenCL device.
          clvk (OpenCL on Vulkan/RADV) allows one device per context and Qrack opens all of a
          platform's devices, so a clvk worker gets only clvk's ICD plus Mesa's device-select
          layer pinned to its GPU's PCI address (it is device 0 inside); workers on other
          platforms get clvk's ICD hidden. Single-process runs re-exec themselves isolated to
          QRACK_OCL_DEFAULT_DEVICE. --gpus numbers are global (`clinfo -l` order). Without
          a clvk ICD installed this is the plain QRACK_OCL_DEFAULT_DEVICE pinning.
          With --ace-gpu, gates run on the GPU and shots are drawn on a CPU copy of the final
          state (measure_shots clones the backend per shot); --ace-gpu-shots keeps them on the GPU.

Larger registers (--register NAME, mirror family only): a Nighthawk sub-register beyond the
          experiment's 61 qubits, built with the release's own code and seed (rect_matchings,
          make_patches(seed=2025, balance_tol=1, top_n=5), the input-string recipe; selftest [9]
          checks that this recipe reproduces the release). nighthawk88: 88 qubits, rows 0-9 x
          columns 1-9 of the 12 x 10 lattice. With --backend ace and no explicit --ace-transpile it
          applies a four-patch preset that fits one 8 GB GPU in fp32: 102 sites, lrc 4 lrr 5, flat,
          stored exact placement, patch widths 28/27/27/27 incl. the error-detection ancilla,
          17 boundary qubits, couplers 102 inside / 55 through a boundary / 0 between patches.
          nighthawk88d: the same 88 qubits' worth on the whole 12 x 10 lattice with 32 sites disabled
          (the device's broken qubits 17, 55, 62 and corner sites), chosen so the four patches meet
          in short diagonal seams, all four at broken qubit 55: 114 sites, lrc 4 lrr 5, flat, widths
          28/23/25/27, 11 boundary qubits, couplers 108 inside / 42 through a boundary / 0 between.
          nighthawk88e: found by layoutsearch from nighthawk88d: also 11 boundary qubits, but 36
          couplers through a boundary instead of 42; first CPU check d4/d6 0.287/0.143 against
          0.030/0.0071 for nighthawk88d. Registers saved by layoutsearch (nighthawk_registers.json
          next to the script, or files in $NIGHTHAWK_REGISTERS) are loaded like built-in ones.
          No device data exists for any of them: summaries show survival and the fitted decay only.
          --transpile-max-width W [--transpile-max-wide N]: the same width caps for any exact placement.

VRAM oversubscription (--gtt GB, with --ace-gpu): Qrack keeps a state vector in device memory
          when 3x its size fits the device, else in host RAM the GPU maps over PCIe (GTT), but
          counts both against one ceiling, QRACK_MAX_ALLOC_MB (default: the device's memory).
          --gtt GB raises that ceiling to --vram-gb + GB per worker and puts paged states in host
          memory, so patches wider than an 8 GB V340 die holds (29: 4 GiB, 30: 8 GiB) run with
          the excess in host RAM. The preflight prints each worker's VRAM / GTT split and stops if
          it cannot fit; layoutsearch --gtt judges candidates by the same split. Same numbers as
          VRAM-only runs, so the config tag is unchanged; the records carry gtt_gb.

Examples
  python nighthawk_qrack.py verify   --repo rcs-nighthawk
  python nighthawk_qrack.py selftest --repo rcs-nighthawk
  python nighthawk_qrack.py hwxeb    --repo rcs-nighthawk --cpu
  python nighthawk_qrack.py run --backend ace --register nighthawk88 --theta nnqab --depths 4 6 8 10 12 \
         --mirror-estimator prob --ace-gpu --gpus 0-5 --per-gpu 1 --out mirror88_clvk.jsonl
  python nighthawk_qrack.py layoutsearch --n 88 --geometries 114:4:5 102:4:5 --region convex \
         --hint nighthawk88e --candidates 3 --cpsat-time 600 --test-depths 4 8 --test-jobs 3 --cpu \
         --save-best nighthawk88x
  python nighthawk_qrack.py run      --repo rcs-nighthawk --backend ace --families mirror \\
         --depths 4 8 16 24 36 --out clean_ace.jsonl
  python nighthawk_qrack.py run      --repo rcs-nighthawk --backend ace --families mirror \\
         --sizes 27-36,61 --depths 4 6 8 12 --out clean_ace.jsonl
  python nighthawk_qrack.py run      --repo rcs-nighthawk --backend exact --families patched \\
         --depths 20 36 --out clean_exact.jsonl
  python nighthawk_qrack.py run      --repo rcs-nighthawk --summarize --out clean_ace.jsonl
  python nighthawk_qrack.py aceplan  --repo rcs-nighthawk --max-width 33
  python nighthawk_qrack.py run      --repo rcs-nighthawk --backend ace --families fxeb \
         --sizes 27-34 --ace-max-width 33 --out big_ace.jsonl
"""

import os

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):
    os.environ["PYQRACK_SHARED_LIB_PATH"] = QRACK_LIB_PATH

import argparse
import collections
import sys
import hashlib
import json
import math
import re
import statistics as st
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ================================================================== constants of the release
BASE_SEED = 2025
INSTANCE_SEED_STRIDE = 1000003
MIRROR_SHOTS = {4: 30_000, 6: 30_000, 8: 30_000, 10: 45_000, 12: 45_000, 14: 45_000, 16: 60_000,
                18: 60_000, 20: 72_000, 24: 96_000, 28: 120_000, 32: 120_000, 36: 240_000, 40: 360_000}
SAMPLE_PUBS, SAMPLE_SHOTS_PER_PUB = 10, 100_000
PATCHED_SHOTS = {20: 2_400, 24: 4_800, 28: 12_000, 32: 24_000, 36: 36_000, 40: 54_000}
GATE_TWIRLS = 64
FULL_DEPTHS = list(range(4, 41, 4))
_MASK64 = (1 << 64) - 1


# ================================================================== angle hash (rcs/circuits.py)
def _splitmix64(x):
    x = (int(x) + 0x9E3779B97F4A7C15) & _MASK64
    z = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def u01_from_key(*parts):
    x = 0
    for p in parts:
        x = _splitmix64(x ^ (int(p) & _MASK64))
    return ((x >> 11) & ((1 << 53) - 1)) / float(1 << 53)


THETA_DIST = {"mode": "haar"}

# --protocol nnqab: forward XEB scored the way rcs/nn_qab.py (vm6502q/pyqrack-examples) scores it, so
# measured points are on the same footing as Dan Strano's ACE calculator: sin(theta)-uniform single-qubit
# gates (THETA_DIST nnqab), each coupler's two-qubit gate drawn uniformly from nn_qab.py's twelve,
# 2^min(13, n + 2) shots, no twirling. The couplers, colour schedule and register stay Nighthawk's.
PROTOCOL = {"mode": "nighthawk"}
NNQAB_GATES = ("swap", "pswap", "mswap", "nswap", "iswap", "iiswap", "cx", "cy", "cz", "acx", "acy", "acz")


def nnqab_gate(seed, instance, cycle, a, b):
    """nn_qab.py's random two-qubit gate for coupler (a, b) of a cycle, from the same hash stream."""
    return NNQAB_GATES[min(11, int(12 * u01_from_key(seed, instance, cycle, a, b, 4)))]


def qrack_g2(sim, g, a, b):
    """nn_qab.py's two-qubit gates (control a, target b), on a QrackSimulator or QrackAceBackend."""
    if g == "cz":
        sim.mcz([a], b)
    elif g == "cx":
        sim.mcx([a], b)
    elif g == "cy":
        sim.mcy([a], b)
    elif g == "acx":
        sim.macx([a], b)
    elif g == "acy":
        sim.macy([a], b)
    elif g == "acz":
        sim.macz([a], b)
    elif g == "swap":
        sim.swap(a, b)
    elif g == "iswap":
        sim.iswap(a, b)
    elif g == "iiswap":
        sim.adjiswap(a, b)
    elif g == "pswap":
        sim.mcz([a], b)
        sim.swap(a, b)
    elif g == "mswap":
        sim.mcz([a], b)
        sim.iswap(a, b)
    elif g == "nswap":
        sim.mcz([a], b)
        sim.adjiswap(a, b)
    else:
        raise ValueError(f"unknown two-qubit gate {g}")


def qrack_g2_inv(sim, g, a, b):
    """Inverse of qrack_g2: iSWAP <-> iSWAP^dagger, CZ.iSWAP -> iSWAP^dagger then CZ, CZ.iSWAP^dagger ->
    iSWAP then CZ; the other nine are their own inverses (CZ.SWAP included: CZ commutes with SWAP)."""
    if g == "iswap":
        sim.adjiswap(a, b)
    elif g == "iiswap":
        sim.iswap(a, b)
    elif g == "mswap":
        sim.adjiswap(a, b)
        sim.mcz([a], b)
    elif g == "nswap":
        sim.iswap(a, b)
        sim.mcz([a], b)
    else:
        qrack_g2(sim, g, a, b)


def haar_angles(seed, instance, cycle, qubit):
    """(phi, theta, lambda), applied as rz(phi), rx(theta), rz(lambda).
    haar (the paper): cos(theta) uniform on [-1, 1], theta in [0, pi].
    nnqab (variant, --theta nnqab): sin(theta) uniform on [-1, 1] as nn_qab.py draws it,
    theta in [-pi/2, pi/2]: weaker rotations, outputs far from Porter-Thomas. Same hash
    stream, so a variant circuit differs from the released one only in theta."""
    phi = 2.0 * math.pi * u01_from_key(seed, instance, cycle, qubit, 1)
    z = 1.0 - 2.0 * u01_from_key(seed, instance, cycle, qubit, 2)
    if THETA_DIST["mode"] == "nnqab":
        theta = math.asin(min(1.0, max(-1.0, z)))
        lam = 2.0 * math.pi * u01_from_key(seed, instance, cycle, qubit, 3)
        return phi, theta, lam
    theta = math.acos(min(1.0, max(-1.0, z)))
    lam = 2.0 * math.pi * u01_from_key(seed, instance, cycle, qubit, 3)
    return phi, theta, lam


def instance_seed(instance, kind, base=BASE_SEED):
    return base + instance * INSTANCE_SEED_STRIDE if kind in ("full", "patched") else base


# ================================================================== one-qubit algebra
I2 = np.eye(2, dtype=complex)


def rz(t):
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])


def rx(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def gate(ang):
    phi, th, lam = ang
    return rz(lam) @ rx(th) @ rz(phi)            # matrix of rz(phi) then rx then rz(lam)


def u_angles(M, tol=1e-12):
    """(theta, phi, lambda) of the U gate equal to M up to global phase; pole-safe."""
    c, s = abs(M[0, 0]), abs(M[1, 0])
    th = 2 * np.arctan2(s, c)
    g = np.angle(M[0, 0]) if c > tol else np.angle(M[1, 0])
    ph = np.angle(M[1, 0]) - g if s > tol else 0.0
    la = np.angle(-M[0, 1]) - g if s > tol else np.angle(M[1, 1]) - g - ph
    return float(th), float(ph), float(la)


def flat(m):
    return [complex(x) for x in m.reshape(-1)]


# ================================================================== layout
REPO_NAME = "rcs-nighthawk"
REPO_URL = "https://github.com/BlueQubitDev/rcs-nighthawk"


def _is_repo(p):
    return p is not None and (Path(p) / "data/layout.json").is_file()


def find_repo(repo=None):
    """--repo if given; else $NIGHTHAWK_REPO; else a folder holding data/layout.json, or an
    rcs-nighthawk clone, in the working dir, next to this script, or up to 3 parents up."""
    if repo:
        if _is_repo(repo):
            return Path(repo)
        raise SystemExit(f"--repo {repo}: no data/layout.json there. Clone it with\n  git clone {REPO_URL} {repo}")
    here, script = Path.cwd(), Path(__file__).resolve().parent
    cands = [os.environ.get("NIGHTHAWK_REPO")]
    for base in (here, script):
        for p in [base, *list(base.parents)[:3]]:
            cands += [p, p / REPO_NAME]
    for c in cands:
        if _is_repo(c):
            return Path(c)
    raise SystemExit(f"cannot find the BlueQubit release (data/layout.json). Either clone it next to this script:\n"
                     f"  git clone {REPO_URL} {script / REPO_NAME}\n"
                     f"or pass --repo /path/to/{REPO_NAME}, or set NIGHTHAWK_REPO.")


class Layout:
    def __init__(self, repo):
        self.repo = find_repo(repo)
        L = json.loads((self.repo / "data/layout.json").read_text())
        self.n = L["num_qubits"]
        self.l2p = L["logical_to_physical"]
        self.schedule = L["schedule"]
        self.matchings = {k: [tuple(e) for e in v] for k, v in L["matchings"].items()}
        self.base_seed = L["base_seed"]
        self.instances = L["instances"]
        self.depths = L["depths"]
        self.inputs = L["mirror_input_strings"]
        self.partitions = {}
        for K, plist in L["partitions"].items():
            self.partitions[int(K)] = [dict(
                boundary=frozenset(tuple(sorted(e)) for e in p["boundary_edges"]),
                patches=[sorted(v) for _, v in sorted(p["patch_qubits"].items(), key=lambda kv: int(kv[0]))])
                for p in plist]
        cols = L["device_lattice"]["cols"]
        (r0, r1), (c0, c1) = L["subgrid"]["rows"], L["subgrid"]["cols"]
        self.grid = (r1 - r0 + 1, c1 - c0 + 1)
        self.pos = [(p // cols - r0, p % cols - c0) for p in self.l2p]
        assert self.base_seed == BASE_SEED


# Nighthawk sub-registers beyond the experiment's 61 qubits, built with the release's own recipe
# (scripts/run_experiment.py: rect_matchings + remove_qubits, make_patches(seed=base_seed,
# balance_tol=1, top_n=5) for K = 3 and 4, ten half-filled mirror inputs from
# default_rng(base_seed)). Region: rows/cols of the 12 x 10 device lattice, filled row by row
# up to n qubits. 'ace' is the ACE preset applied when --backend ace is used without an
# explicit --ace-transpile: an ACE register the exact engine maps this register onto within
# the width caps (max width incl. the error-detection ancilla, how many patches may reach it).
REGISTERS = {
    "nighthawk88": dict(
        n=88, rows=(0, 9), cols=(1, 9),
        about="88 qubits, rows 0-9 x columns 1-9 of the 12 x 10 Nighthawk lattice (last row 7/9); "
              "contains the experiment's 8 x 8 window, its three dropped sites included",
        # four patches 28/27/27/27 wide (incl. ancilla) on one 8 GB die, 17 boundary qubits,
        # couplers inside / through a boundary / between patches 102 / 55 / 0. Found by the exact
        # engine with caps (28, 1) in ~5 min; stored so every run uses the same placement at once.
        ace=dict(width=102, lrc=4, lrr=5, torus=False, max_width=28, max_wide=1,
                 site=[10, 11, 12, 9, 5, 6, 7, 8, 27, 13, 32, 33, 31, 28, 29, 30, 49, 50, 34, 35, 36, 55, 51, 52, 53, 4, 26, 37, 56, 57, 77, 54, 73, 48, 0, 1, 58, 59, 78, 99, 74, 72, 2, 3, 22, 79, 80, 14, 38, 94, 23, 24, 25, 44, 60, 82, 15, 16, 21, 45, 46, 47, 66, 17, 18, 19, 20, 39, 43, 67, 68, 69, 40, 41, 42, 61, 62, 63, 65, 70, 71, 64, 83, 84, 85, 86, 87, 88])),
    "nighthawk88d": dict(
        n=88, rows=(0, 11), cols=(0, 9), drop=[0, 1, 2, 7, 8, 9, 10, 11, 17, 18, 19, 55, 62, 80, 81, 82, 83, 90, 91, 92, 93, 99, 100, 101, 102, 103, 109, 110, 111, 112, 113, 119],
        about="88 qubits on the whole 12 x 10 Nighthawk lattice with 32 sites disabled: the device's broken "
              "qubits 17, 55, 62 and corner sites chosen so the four ACE patches meet in short diagonal seams "
              "(all four at broken qubit 55); every row and column of the region is one unbroken run",
        # four patches 28/23/25/27 wide (incl. ancilla), 11 boundary qubits, couplers inside / through a
        # boundary / between patches 108 / 42 / 0 (150 couplers; nighthawk88 has 157). Region and split
        # found with CP-SAT under ACE's ring of seams (114 sites, lrc 4 lrr 5, flat: seams 0-1, 1-2,
        # 2-3, 3-0 only); the placement puts each qubit straight into its patch's bulk or seam group.
        ace=dict(width=114, lrc=4, lrr=5, torus=False, max_width=28, max_wide=1,
                 site=[10, 11, 12, 13, 34, 35, 36, 37, 38, 15, 16, 14, 39, 40, 41, 62, 63, 9, 5, 17, 18, 19, 42, 64, 65, 86, 33, 6, 7, 20, 21, 22, 43, 66, 87, 61, 8, 29, 30, 44, 45, 46, 67, 68, 31, 32, 53, 54, 69, 70, 91, 23, 0, 4, 55, 56, 57, 92, 93, 94, 95, 47, 1, 2, 28, 58, 59, 3, 24, 25, 26, 52, 60, 27, 48, 49, 50, 51, 72, 73, 74, 75, 76, 77, 78, 79, 100, 101])),
    "nighthawk88e": dict(
        n=88, rows=(0, 11), cols=(0, 9), drop=[0, 1, 8, 9, 10, 11, 17, 20, 21, 55, 62, 79, 80, 81, 82, 83, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103, 109, 110, 111, 112, 113, 119],
        about="88 qubits, 11 boundary qubits like nighthawk88d but 36 couplers through a boundary instead of 42 "
              "(per colour 8/9/10/9); found by layoutsearch warm-started from nighthawk88d (convex region, "
              "broken qubits disabled, all four patches meet at broken qubit 55)",
        # 114 sites, lrc 4 lrr 5, flat; widths 25/28/25/25 incl. ancilla; couplers 112 / 36 / 0 (148).
        # First CPU check (nnqab, prob, 3 instances x 40 runs): d4 0.287, d6 0.143 against 0.030, 0.0071
        # for nighthawk88d on the same settings.
        ace=dict(width=114, lrc=4, lrr=5, torus=False, max_width=28, max_wide=1,
                 site=[15, 16, 17, 18, 19, 20, 21, 22, 43, 44, 45, 10, 11, 46, 67, 68, 69, 70, 14, 12, 13, 0, 1, 23, 47, 91, 92, 42, 34, 35, 36, 2, 3, 24, 25, 71, 66, 37, 38, 39, 40, 26, 27, 48, 49, 50, 41, 62, 63, 64, 51, 72, 73, 74, 4, 9, 65, 86, 87, 75, 76, 77, 78, 28, 5, 6, 33, 61, 7, 8, 29, 30, 31, 32, 53, 54, 55, 56, 57, 58, 59, 60, 81, 82, 83, 84, 105, 106])),
    "nighthawk61r2": dict(
        n=61, rows=(1, 8), cols=(2, 9), drop=[17, 55, 62],
        about="the release's own 61 qubits (bit-for-bit the same mirror circuits) as two patches: 6 boundary "
              "qubits, the proven minimum of any balanced two-way split, placed for the fewest circuit CZs "
              "through the boundary (layoutsearch --cz-weights mirror)",
        # 77 sites, lrc 5 lrr 6, flat; widths 35/34 incl. ancilla (256 + 128 GiB fp32: too large for a
        # 320 GB host), 34/33 with --no-ace-error-detection (128 + 64 GiB). Couplers 83 / 19 / 0; boundary
        # CZs per mirror circuit (forward half) d4..d20: 8 12 14 16 19 25 34 (three-patch: 11 18 24 27 27 40 49).
        ace=dict(width=77, lrc=5, lrr=6, torus=False, max_width=35, max_wide=1,
                 site=[6, 7, 8, 9, 10, 0, 1, 11, 12, 13, 14, 5, 2, 3, 4, 15, 28, 29, 16, 17, 18, 19, 20, 30, 31, 32,
                       27, 21, 22, 23, 24, 33, 34, 35, 25, 26, 39, 40, 36, 37, 38, 41, 42, 43, 44, 50, 51, 52, 49, 45,
                       46, 47, 48, 53, 54, 55, 56, 60, 61, 62, 63])),
    "nighthawk61w": dict(
        n=61, rows=(0, 10), cols=(2, 7), drop=[17, 52, 55, 57, 62],
        about="61 qubits, 11 x 6 window, two patches joined by a 3-qubit waist on row 5 (broken qubits 55 and "
              "62 plus border sites 52 and 57 left out); same recipe and CZ count as the release (221 CZs at "
              "d = 20), but a different register: its narrowest cut is 3 qubits, the 8 x 8 window's is 8",
        # 75 sites, lrc 8 lrr 4, flat; widths 33/33 incl. ancilla (64 + 64 GiB fp32), couplers 93 / 7 / 0.
        # Boundary CZs per mirror circuit (forward half) d4..d20, d36: 1 3 6 6 7 12 13, 27.
        ace=dict(width=75, lrc=8, lrr=4, torus=False, max_width=34, max_wide=1,
                 site=[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                       48, 49, 50, 51, 52, 69, 8, 23, 38, 0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31,
                       32, 33, 34, 35, 36, 37, 54, 55, 56, 57, 58, 59, 60])),
    "nighthawk2p16": dict(
        n=16, rows=(1, 3), cols=(2, 9), drop=[17, 33, 34, 35, 36, 37, 38, 39],
        about="two-patch ladder below 28: the release's first 16 logical qubits, held to a 3-qubit seam "
              "(--min-boundary 3), placed for the fewest circuit CZs through the boundary",
        # 28 sites, lrc 3 lrr 3, flat; widths 10/11 incl. ancilla; 6 couplers through the boundary.
        ace=dict(width=28, lrc=3, lrr=3, torus=False, max_width=13, max_wide=1,
                 site=[4, 5, 6, 7, 3, 0, 1, 8, 9, 18, 10, 2, 11, 12, 13, 17])),
    "nighthawk2p20": dict(
        n=20, rows=(1, 3), cols=(2, 9), drop=[17, 37, 38, 39],
        about="two-patch ladder below 28: the release's first 20 logical qubits, held to a 3-qubit seam "
              "(--min-boundary 3), placed for the fewest circuit CZs through the boundary",
        # 32 sites, lrc 4 lrr 3, flat; widths 13/12 incl. ancilla; 9 couplers through the boundary.
        ace=dict(width=32, lrc=4, lrr=3, torus=False, max_width=15, max_wide=1,
                 site=[5, 6, 7, 4, 0, 1, 2, 8, 9, 10, 12, 3, 13, 14, 15, 11, 21, 20, 16, 17])),
    "nighthawk2p24": dict(
        n=24, rows=(1, 4), cols=(2, 9), drop=[17, 43, 44, 45, 46, 47, 48, 49],
        about="two-patch ladder below 28: the release's first 24 logical qubits, held to a 3-qubit seam "
              "(--min-boundary 3), placed for the fewest circuit CZs through the boundary",
        # 38 sites, lrc 8 lrr 1, torus; widths 16/12 incl. ancilla; 8 couplers through the boundary.
        ace=dict(width=38, lrc=8, lrr=1, torus=True, max_width=17, max_wide=1,
                 site=[0, 1, 2, 3, 4, 9, 10, 5, 6, 7, 19, 8, 11, 12, 13, 20, 21, 22, 27, 14, 15, 16, 28, 18])),
    "nighthawk2p28": dict(
        n=28, rows=(1, 4), cols=(2, 9), drop=[17, 47, 48, 49],
        about="two-patch ladder: the release's first 28 logical qubits, 3 boundary qubits (proven minimum "
              "under the width cap), placed for the fewest circuit CZs through the boundary",
        # 39 sites, lrc 6 lrr 2, flat; widths 15/18 incl. ancilla; couplers 32 / 11 / 0.
        # Boundary CZs per mirror circuit (forward half) d4..d20: 5 7 8 9 10 13 18.
        ace=dict(width=39, lrc=6, lrr=2, torus=False, max_width=19, max_wide=1,
                 site=[7, 8, 9, 10, 11, 0, 1, 12, 13, 14, 15, 6, 2, 3, 4, 16, 17, 18, 19, 5, 20, 21, 22, 33, 34, 32, 23, 24])),
    "nighthawk2p32": dict(
        n=32, rows=(1, 5), cols=(2, 9), drop=[17, 53, 54, 55, 56, 57, 58, 59],
        about="two-patch ladder: the release's first 32 logical qubits, 3 boundary qubits (proven minimum "
              "under the width cap), placed for the fewest circuit CZs through the boundary",
        # 44 sites, lrc 5 lrr 3, flat; widths 18/19 incl. ancilla; couplers 39 / 11 / 0.
        # Boundary CZs per mirror circuit (forward half) d4..d20: 5 8 10 10 12 15 20.
        ace=dict(width=44, lrc=5, lrr=3, torus=False, max_width=21, max_wide=1,
                 site=[6, 7, 8, 9, 10, 0, 1, 11, 12, 13, 14, 5, 2, 3, 4, 15, 28, 29, 16, 17, 18, 19, 20, 30, 31, 27, 21, 22, 23, 24, 25, 32])),
    "nighthawk2p36": dict(
        n=36, rows=(1, 5), cols=(2, 9), drop=[17, 55, 58, 59],
        about="two-patch ladder: the release's first 36 logical qubits, 3 boundary qubits (proven minimum "
              "under the width cap), placed for the fewest circuit CZs through the boundary",
        # 50 sites, lrc 5 lrr 4, flat; widths 20/21 incl. ancilla; couplers 45 / 12 / 0.
        # Boundary CZs per mirror circuit (forward half) d4..d20: 5 8 11 12 14 20 25.
        ace=dict(width=50, lrc=5, lrr=4, torus=False, max_width=23, max_wide=1,
                 site=[6, 7, 8, 9, 10, 0, 1, 11, 12, 13, 14, 5, 2, 3, 4, 26, 27, 28, 15, 16, 17, 18, 19, 29, 30, 25, 20, 21, 22, 23, 24, 31, 32, 33, 36, 37])),
    "nighthawk2p40": dict(
        n=40, rows=(1, 6), cols=(2, 9), drop=[17, 55, 62, 65, 66, 67, 68, 69],
        about="two-patch ladder: the release's first 40 logical qubits, 3 boundary qubits (proven minimum "
              "under the width cap), placed for the fewest circuit CZs through the boundary",
        # 51 sites, lrc 8 lrr 2, flat; widths 22/23 incl. ancilla; couplers 52 / 12 / 0.
        # Boundary CZs per mirror circuit (forward half) d4..d20: 4 4 7 10 13 19 21.
        ace=dict(width=51, lrc=8, lrr=2, torus=False, max_width=25, max_wide=1,
                 site=[9, 10, 11, 12, 13, 0, 1, 14, 15, 16, 17, 8, 2, 3, 4, 18, 19, 20, 25, 5, 6, 7, 26, 21, 22, 42, 27, 28, 29, 30, 31, 23, 24, 43, 32, 33, 34, 35, 44, 45])),
    "nighthawk2p45": dict(
        n=45, rows=(1, 6), cols=(2, 9), drop=[17, 55, 62],
        about="two-patch ladder: the release's first 45 logical qubits, 4 boundary qubits (proven minimum "
              "under the width cap), placed for the fewest circuit CZs through the boundary",
        # 60 sites, lrc 5 lrr 5, flat; widths 26/25 incl. ancilla; couplers 61 / 12 / 0.
        # Boundary CZs per mirror circuit (forward half) d4..d20: 7 8 8 10 11 15 20.
        ace=dict(width=60, lrc=5, lrr=5, torus=False, max_width=27, max_wide=1,
                 site=[6, 7, 8, 9, 10, 0, 1, 11, 12, 13, 14, 5, 2, 3, 4, 26, 27, 28, 15, 16, 17, 18, 19, 29, 30, 31, 25, 20, 21, 22, 23, 32, 33, 34, 24, 36, 37, 38, 46, 47, 35, 39, 40, 41, 42])),
    "nighthawk2p49": dict(
        n=49, rows=(1, 7), cols=(2, 9), drop=[17, 55, 62, 76, 77, 78, 79],
        about="two-patch ladder: the release's first 49 logical qubits, 4 boundary qubits (proven minimum "
              "under the width cap), placed for the fewest circuit CZs through the boundary",
        # 63 sites, lrc 4 lrr 6, flat; widths 27/28 incl. ancilla; couplers 64 / 15 / 0.
        # Boundary CZs per mirror circuit (forward half) d4..d20: 6 9 12 13 17 21 29.
        ace=dict(width=63, lrc=4, lrr=6, torus=False, max_width=29, max_wide=1,
                 site=[5, 6, 7, 8, 9, 0, 1, 10, 11, 12, 23, 4, 2, 3, 14, 24, 25, 26, 13, 15, 16, 17, 18, 27, 28, 22, 19, 20, 21, 32, 33, 29, 30, 41, 34, 35, 36, 37, 42, 43, 31, 38, 39, 50, 51, 44, 45, 46, 47])),
    "nighthawk2p53": dict(
        n=53, rows=(1, 7), cols=(2, 9), drop=[17, 55, 62],
        about="two-patch ladder: the release's first 53 logical qubits, 5 boundary qubits (proven minimum "
              "under the width cap), placed for the fewest circuit CZs through the boundary",
        # 65 sites, lrc 6 lrr 4, flat; widths 30/30 incl. ancilla; couplers 72 / 15 / 0.
        # Boundary CZs per mirror circuit (forward half) d4..d20: 8 10 12 15 19 24 34.
        ace=dict(width=65, lrc=6, lrr=4, torus=False, max_width=31, max_wide=1,
                 site=[7, 8, 9, 10, 11, 0, 1, 12, 13, 14, 15, 6, 2, 3, 4, 16, 17, 18, 33, 19, 5, 20, 21, 34, 35, 36, 32, 22, 23, 24, 25, 37, 38, 39, 26, 27, 28, 29, 40, 41, 45, 30, 31, 46, 47, 42, 43, 44, 58, 48, 49, 50, 51])),
    "nighthawk2p57": dict(
        n=57, rows=(1, 8), cols=(2, 9), drop=[17, 55, 62, 86, 87, 88, 89],
        about="two-patch ladder: the release's first 57 logical qubits, 5 boundary qubits (proven minimum "
              "under the width cap), placed for the fewest circuit CZs through the boundary",
        # 72 sites, lrc 4 lrr 7, flat; widths 32/32 incl. ancilla; couplers 76 / 18 / 0.
        # Boundary CZs per mirror circuit (forward half) d4..d20: 7 10 12 17 20 25 34.
        ace=dict(width=72, lrc=4, lrr=7, torus=False, max_width=33, max_wide=1,
                 site=[5, 6, 7, 8, 9, 0, 1, 10, 11, 12, 23, 4, 2, 3, 14, 24, 25, 26, 13, 15, 16, 17, 18, 27, 28, 22, 19, 20, 21, 32, 33, 29, 30, 41, 34, 35, 36, 37, 42, 43, 31, 38, 39, 50, 51, 44, 45, 46, 40, 52, 53, 54, 55, 47, 48, 59, 60])),
}

BUILTIN_REGISTERS = frozenset(REGISTERS)


def _load_release_module(repo, name):
    """rcs/<name>.py of the release by file path (stdlib only; rcs/__init__ is not run)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"_nighthawk_rcs_{name}", repo / "rcs" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                 # dataclasses look their module up while loading
    spec.loader.exec_module(mod)
    return mod


class RegisterLayout(Layout):
    """A Layout for REGISTERS[name], generated deterministically from the release's code and
    its base seed, so the circuits follow the experiment's construction exactly: four-colour
    schedule A B C D, the mirror circuits' rotating K = 3 pseudo-patch cuts, the same depths,
    instances, angle hash and input-string recipe. There is no device data for it."""

    def __init__(self, repo, name):
        if name not in REGISTERS:
            raise SystemExit(f"unknown --register {name}; known: release, {', '.join(sorted(REGISTERS))}")
        spec = REGISTERS[name]
        self.repo = find_repo(repo)
        L = json.loads((self.repo / "data/layout.json").read_text())
        rl, pt = _load_release_module(self.repo, "layout"), _load_release_module(self.repo, "patching")
        (r0, r1), (c0, c1) = spec["rows"], spec["cols"]
        R, C, n = r1 - r0 + 1, c1 - c0 + 1, spec["n"]
        cols_dev = L["device_lattice"]["cols"]
        if not (0 < n <= R * C):
            raise SystemExit(f"register {name}: {n} qubits do not fit rows {r0}-{r1} x cols {c0}-{c1}")
        if spec.get("drop") is not None:
            # device sites (row * cols + col) left out of the window, as the release drops 17, 55, 62
            gone = sorted({(p // cols_dev - r0) * C + (p % cols_dev - c0) for p in spec["drop"]})
            if any(not (0 <= g < R * C) for g in gone) or R * C - len(gone) != n:
                raise SystemExit(f"register {name}: drop list does not leave {n} sites in the window")
        else:
            gone = list(range(n, R * C))
        keep = [i for i in range(R * C) if i not in set(gone)]
        m = rl.remove_qubits(rl.rect_matchings(R, C), gone)
        edges = sorted({tuple(sorted(e)) for es in m.values() for e in es})
        base = L["base_seed"]
        parts = {k: pt.make_patches(n, edges, k, seed=base, balance_tol=1, top_n=5) for k in (3, 4)}
        rng = np.random.default_rng(seed=base)
        strings = []
        for _ in range(len(L["mirror_input_strings"])):
            bits = np.array(["0"] * n)
            bits[rng.choice(n, size=n // 2, replace=False)] = "1"
            strings.append("".join(bits))
        self.name = name
        self.n = n
        self.l2p = [(r0 + i // C) * cols_dev + (c0 + i % C) for i in keep]
        self.schedule = L["schedule"]
        self.matchings = {k: [tuple(e) for e in v] for k, v in m.items()}
        self.base_seed = base
        self.instances = L["instances"]
        self.depths = L["depths"]
        self.inputs = strings
        self.partitions = {K: [dict(boundary=frozenset(tuple(sorted(e)) for e in b),
                                    patches=[sorted(v) for _, v in sorted(pq.items(), key=lambda kv: int(kv[0]))])
                               for b, pq in parts[K]] for K in (3, 4)}
        self.grid = (R, C)
        self.pos = [(i // C, i % C) for i in keep]
        assert self.base_seed == BASE_SEED


def make_layout(a):
    """The release's 61-qubit layout, or --register NAME."""
    name = getattr(a, "register", "release") or "release"
    return Layout(getattr(a, "repo", None)) if name == "release" else RegisterLayout(getattr(a, "repo", None), name)


def layout_id(lay):
    return getattr(lay, "name", "release")


def is_release(lay):
    return layout_id(lay) == "release"


def apply_register_preset(a, lay):
    """--register NAME with --backend ace and no explicit --ace-transpile: the register's ACE
    preset (geometry, exact engine, width caps). Explicit --ace-transpile settings win."""
    if is_release(lay) or a.backend != "ace" or getattr(a, "ace_transpile", False):
        return False
    ace = REGISTERS[layout_id(lay)]["ace"]
    a.ace_transpile, a.transpile_engine = True, "exact"
    a.ace_width, a.lrc, a.lrr, a.ace_torus = ace["width"], ace["lrc"], ace["lrr"], ace["torus"]
    a.ace_exclude = None
    a.transpile_max_width, a.transpile_max_wide = ace["max_width"], ace["max_wide"]
    return True


def preset_placement(lay, n, a, row, size):
    """The stored placement of the register's ACE preset, when this run uses exactly that preset.
    Switching seam error detection off keeps it: the sites and patches are unchanged, each patch
    only loses its ancilla (one qubit narrower, half the memory)."""
    if is_release(lay):
        return None
    ace = REGISTERS[layout_id(lay)].get("ace")
    if not ace or n != lay.n or size != ace["width"] or (row["lrc"], row["lrr"]) != (ace["lrc"], ace["lrr"]):
        return None
    opts = {k: v for k, v in ace_opts(a).items() if k != "is_error_detection"}
    if bool(getattr(a, "ace_torus", False)) != ace["torus"] or parse_exclude(getattr(a, "ace_exclude", None)) \
            or opts or transpile_caps(a) != (ace["max_width"], ace["max_wide"]) \
            or (getattr(a, "transpile_engine", "exact") or "exact") != "exact":
        return None
    return list(ace["site"])


def transpile_caps(a):
    """(max width incl. ancilla, patches allowed at that width) for the exact engine, or None."""
    w = getattr(a, "transpile_max_width", None)
    return (int(w), getattr(a, "transpile_max_wide", None)) if w else None


def build_cycles(lay, ncyc, seed, inst, removed=frozenset(), rotate=None, n=None):
    """[(angles[q], cz_edges)] per cycle, as build_rcs_circuit with removed_edges /
    make_pseudo_patch_edge_filter(rotate, rotate_every=1)."""
    n = n or lay.n
    out = []
    for c in range(ncyc):
        ang = [haar_angles(seed, inst, c, q) for q in range(n)]
        col = lay.schedule[c % len(lay.schedule)]
        es = [e for e in lay.matchings[col] if e[0] < n and e[1] < n and tuple(sorted(e)) not in removed]
        if rotate:
            cut = rotate[c % len(rotate)]
            es = [e for e in es if tuple(sorted(e)) not in cut]
        out.append((ang, es))
    return out


def mirror_cycles(lay, depth, inst, n=None):
    rot = [p["boundary"] for p in lay.partitions[3]]
    return build_cycles(lay, depth // 2, instance_seed(inst, "mirror"), inst, rotate=rot, n=n)


def patched_cycles(lay, K, depth, j, inst):
    return build_cycles(lay, depth, instance_seed(inst, "patched"), inst,
                        removed=lay.partitions[K][j]["boundary"])


def full_cycles(lay, depth):
    return build_cycles(lay, depth, instance_seed(0, "full"), 0)


# ================================================================== verify (no Qrack)
_OP = re.compile(r"^\s*(rz|rx|cz)\s*(?:\(([^)]*)\))?\s+q\[(\d+)\](?:\s*,\s*q\[(\d+)\])?\s*;", re.M)


def qasm_ops(text):
    out = []
    for g, arg, a, b in _OP.findall(text):
        if g == "cz":
            out.append(("cz", int(a), int(b)))
        else:
            try:
                out.append((g, int(a), float(arg)))
            except ValueError:           # rx(_theta_q_): mirror preparation parameter
                continue
    return out


def ops_forward(cyc):
    out = []
    for ang, es in cyc:
        for q, (phi, th, lam) in enumerate(ang):
            out += [("rz", q, phi), ("rx", q, th), ("rz", q, lam)]
        out += [("cz", a, b) for a, b in es]
    return out


def ops_inverse(cyc):
    out = []
    for g in reversed(ops_forward(cyc)):
        out.append(g if g[0] == "cz" else (g[0], g[1], -g[2]))
    return out


def same_ops(A, B, tol):
    if len(A) != len(B):
        return False, f"length {len(A)} vs {len(B)}"
    for i, (x, y) in enumerate(zip(A, B)):
        if x[0] != y[0] or x[1] != y[1]:
            return False, f"op {i}: {x} vs {y}"
        if x[0] == "cz":
            if x[2] != y[2]:
                return False, f"op {i}: {x} vs {y}"
        elif abs(math.remainder(x[2] - y[2], 2 * math.pi)) > tol:
            return False, f"op {i}: angle {x[2]!r} vs {y[2]!r}"
    return True, ""


def released_circuits(lay):
    for d in lay.depths["mirror"]:
        for i in range(lay.instances):
            cyc = mirror_cycles(lay, d, i)
            yield f"mirror/d{d:02d}_instance{i}.qasm", ops_forward(cyc) + ops_inverse(cyc)
    for K in (3, 4):
        for d in lay.depths["patched"]:
            for j in range(len(lay.partitions[K])):
                for i in range(lay.instances):
                    yield (f"patched/K{K}/d{d}/partition{j}_instance{i}.qasm",
                           ops_forward(patched_cycles(lay, K, d, j, i)))
    for d in FULL_DEPTHS:
        yield f"full/d{d:02d}_logical.qasm", ops_forward(full_cycles(lay, d))


def cmd_verify(a):
    lay = Layout(a.repo)
    root = lay.repo / "data/circuits"
    man = {c["qasm3"]: c for c in json.loads((root / "manifest.json").read_text())["circuits"]}
    n_ok = n_bad = 0
    for rel, mine in released_circuits(lay):
        theirs = qasm_ops((root / rel).read_text())
        ok, why = same_ops(mine, theirs, a.tol)
        ncz = sum(1 for g in mine if g[0] == "cz")
        if ok and rel in man and man[rel]["cz"] != ncz:
            ok, why = False, f"CZ count {ncz} vs manifest {man[rel]['cz']}"
        n_ok += ok
        n_bad += not ok
        if not ok:
            print(f"DIFFERENT {rel}: {why}")
    in_repo = {k for k in man if not k.endswith("_executed.qasm")}
    missed = in_repo - {rel for rel, _ in released_circuits(lay)}
    print(f"{n_ok} identical, {n_bad} different (angle tol {a.tol:g} rad)"
          + (f"; not regenerated: {sorted(missed)}" if missed else ""))
    raise SystemExit(1 if n_bad or missed else 0)


# ================================================================== engines
PAULIS = ("i", "x", "y", "z")


class ExactEngine:
    """QrackSimulator on the logical register."""

    def __init__(self, n, cpu):
        from pyqrack import QrackSimulator
        self.n = n
        self.sim = QrackSimulator(n, is_gpu=not cpu)

    def geometry(self):
        return {}

    def reset(self):
        self.sim.reset_all()

    def g1(self, q, M):
        self.sim.mtrx(flat(M), q)

    def pauli(self, p, q):
        if p != "i":
            getattr(self.sim, p)(q)

    def cz(self, a, b):
        self.sim.mcz([a], b)

    def cx(self, a, b):
        self.sim.mcx([a], b)

    def g2(self, g, a, b):
        qrack_g2(self.sim, g, a, b)

    def g2_inv(self, g, a, b):
        qrack_g2_inv(self.sim, g, a, b)

    def prob1(self, q):
        return float(self.sim.prob(q))

    def shots(self, s):
        if self.n > 32:      # PyQrack 2.28.1: QrackSimulator.measure_shots returns bits 0-31 only
            raise RuntimeError(f"exact engine: measure_shots drops every bit above 31 (n = {self.n}); "
                               "use a prob-based estimator or the ace backend")
        return np.array(self.sim.measure_shots(list(range(self.n)), s), dtype=np.uint64)

    def prob_bits(self, bits):
        return float(self.sim.prob_perm(list(range(self.n)), [bool(b) for b in bits]))


_TWO_PATCH = {}


# the script's defaults, passed explicitly by ace_backend (PyQrack's own defaults differ between versions)
ACE_OPTS = (("ace_boundary_rep", "is_boundary_repetition_code", False),
            ("ace_error_detection", "is_error_detection", True),
            ("ace_crossbars", "use_crossbars", True))


_ACE_PARAMS = {}


def ace_backend(*args, **kw):
    """QrackAceBackend with every ACE_OPTS option passed explicitly (the script's default unless the caller
    gives one), for each option this PyQrack build has. PyQrack changes these defaults between versions
    (2.29 turned is_boundary_repetition_code on), so a run means the same thing on any version; config tags
    keep naming only the options that differ from ACE_OPTS."""
    from pyqrack import QrackAceBackend
    if "p" not in _ACE_PARAMS:
        import inspect
        _ACE_PARAMS["p"] = set(inspect.signature(QrackAceBackend.__init__).parameters)
    for _, k, d in ACE_OPTS:
        if k in _ACE_PARAMS["p"]:
            kw.setdefault(k, d)
    return QrackAceBackend(*args, **kw)


def ace_opts(a):
    """QrackAceBackend keyword arguments that differ from the script's defaults (ACE_OPTS) (only those
    are passed, so builds without an option work as long as it is not requested)."""
    return {kw: bool(getattr(a, attr, dflt)) for attr, kw, dflt in ACE_OPTS
            if a is not None and bool(getattr(a, attr, dflt)) != dflt}


def check_ace_opts(configs):
    """Stop before any work if a requested ACE option is not a parameter of this PyQrack
    build's QrackAceBackend (the options appeared in different PyQrack versions)."""
    import inspect
    from pyqrack import QrackAceBackend
    have = set(inspect.signature(QrackAceBackend.__init__).parameters)
    bad = [(name, kw) for name, a in configs for kw in ace_opts(a) if kw not in have]
    if bad:
        import pyqrack
        ver = getattr(pyqrack, "__version__", "unknown version")
        raise SystemExit("this PyQrack build (" + ver + ") has no QrackAceBackend parameter "
                         + ", ".join(sorted({kw for _, kw in bad})) + "; drop "
                         + ", ".join(sorted({name for name, _ in bad})) + " or update PyQrack")


def ace_register_geometry(size, lrc, lrr, torus, opts=None):
    """(row_length, column_length, patch sizes, boundary sites, bulk/boundary) of an ACE
    register, read from its own _unpack() like nn_qab.py's bulk_to_boundary_ratio()."""
    from collections import Counter
    from pyqrack import QrackAceBackend
    s = ace_backend(size, long_range_columns=lrc, long_range_rows=lrr, is_torus=torus, is_gpu=False,
                        **(opts or {}))
    sizes = Counter(sid for lq in range(s.num_qubits()) for sid, _ in s._unpack(lq))
    bnd = sum(1 for lq in range(s.num_qubits()) if len(s._unpack(lq)) > 1)
    patches = sorted(v for k, v in sizes.items() if not (bnd and k == max(sizes)))
    return (s.get_row_length(), s.get_column_length(), patches, bnd,
            (size - bnd) / bnd if bnd else float("inf"))


def two_patch_config(size, torus):
    """nn_qab.py's rule: exactly two patches at the highest finite bulk-to-boundary
    ratio, ties broken toward the most balanced patch sizes, then the smallest lrc/lrr."""
    key = (size, torus)
    if key not in _TWO_PATCH:
        best = None
        for lrr in range(0, 9):
            for lrc in range(0, 9):
                R, C, p, b, ratio = ace_register_geometry(size, lrc, lrr, torus)
                if len(p) == 2 and b:
                    score = (ratio, -abs(p[0] - p[1]), -lrc, -lrr)
                    if best is None or score > best[0]:
                        best = (score, lrc, lrr, p, b, ratio)
        if best is None:
            raise SystemExit(f"no two-patch ACE configuration for a {size}-site register (torus={torus})")
        _TWO_PATCH[key] = best[1:]
    return _TWO_PATCH[key]


def ace_sim_widths(s):
    """Qubit width of every simulator inside a QrackAceBackend, widest first: the patch
    simulators (bulk + boundary replicas) and, on builds with crossbars, the shared
    boundary simulator. Falls back to counting _unpack() entries per simulator."""
    sims = getattr(s, "sim", None)
    if sims:
        try:
            return sorted((int(x.num_qubits()) for x in sims), reverse=True)
        except Exception:
            pass
    return sorted(collections.Counter(sid for lq in range(s.num_qubits()) for sid, _ in s._unpack(lq)).values(),
                  reverse=True)


def amp_bytes():
    """Bytes per complex amplitude of this Qrack build (fp16 4, fp32 8, fp64 16)."""
    from pyqrack.qrack_system import Qrack
    return 2 ** (Qrack.fppow - 2)


# ================================================================== VRAM + GTT
# Where Qrack keeps an OpenCL state vector (QEngineOCL::SetDevice): in device memory when
# three times its size fits the device's global memory, otherwise in host RAM that the GPU
# maps over PCIe (CL_MEM_USE_HOST_PTR; on amdgpu the GTT). Every allocation, host-mapped
# ones included, counts against one per-device ceiling, QRACK_MAX_ALLOC_MB (default: the
# device's global memory), so without raising it a state that would spill to the GTT is
# refused ("VRAM limits exceeded"). --gtt GB raises the ceiling to VRAM + GB and puts paged
# states (QPager) in host memory too.
GTT_RESERVE_GB = 0.5          # driver, kernels and Qrack's small buffers on the die


def die_split(widths, vram_gb=8.0, gtt_gb=0.0, reserve_gb=GTT_RESERVE_GB):
    """One worker's ACE patches, fully entangled, on one OpenCL die: every patch state plus
    the compose buffer (a state the size of the widest patch), each placed by Qrack's rule.
    Returns dict(vram, gtt, total GiB, fits, limit_mb) for a die of vram_gb with gtt_gb of
    host memory allowed."""
    gib = 2 ** 30
    states = [2.0 ** w * amp_bytes() / gib for w in widths] + ([2.0 ** max(widths) * amp_bytes() / gib] if widths else [])
    vram = sum(x for x in states if 3 * x <= vram_gb)
    gtt = sum(x for x in states if 3 * x > vram_gb)
    return dict(vram=vram, gtt=gtt, total=vram + gtt, fits=vram <= vram_gb - reserve_gb and gtt <= gtt_gb,
                limit_mb=int((vram_gb + gtt_gb) * 1024))


def gtt_env(env, vram_gb, gtt_gb):
    """Environment additions for --gtt: Qrack's per-device allocation ceiling at VRAM + GTT,
    paged states in host memory. Variables the user set explicitly win."""
    if gtt_gb:
        env.setdefault("QRACK_MAX_ALLOC_MB", str(int((vram_gb + gtt_gb) * 1024)))
        env.setdefault("QRACK_QPAGER_DEVICES_HOST_POINTER", "1")
        env["NIGHTHAWK_GTT_GB"] = str(gtt_gb)
    return env


def describe_split(sp, vram_gb, gtt_gb):
    return (f"VRAM {sp['vram']:.2f} of {vram_gb:g} GiB (-{GTT_RESERVE_GB:g} reserve), GTT {sp['gtt']:.2f} of "
            f"{gtt_gb:g} GiB -> {'fits' if sp['fits'] else 'DOES NOT FIT'}")


def dense_gb(widths):
    """Worst-case memory if every simulator were one dense state vector. QUnit keeps
    unentangled qubits apart, so shallow or truncated points use far less."""
    return sum(2.0 ** w for w in widths) * amp_bytes() / 2 ** 30


_ACE_PLAN = {}


def ace_plan(size, toruses=(False, True)):
    """Every distinct ACE layout of a size-site register over lrc = 0..row length,
    lrr = 0..column length and the given torus settings. Layouts that place every
    qubit identically are listed once, under the smallest lrc + lrr."""
    key = (size, tuple(toruses))
    if key in _ACE_PLAN:
        return _ACE_PLAN[key]
    from pyqrack import QrackAceBackend
    probe = ace_backend(size, is_gpu=False)
    C, R = probe.get_row_length(), probe.get_column_length()
    del probe
    cands = sorted(((t, lrc, lrr) for t in toruses for lrr in range(R + 1) for lrc in range(C + 1)),
                   key=lambda x: (x[1] + x[2], x[1], x[0]))
    rows, seen = [], set()
    for torus, lrc, lrr in cands:
        s = ace_backend(size, long_range_columns=lrc, long_range_rows=lrr, is_torus=torus, is_gpu=False)
        unpack = [tuple(tuple(e) for e in s._unpack(q)) for q in range(s.num_qubits())]
        sig = (torus, tuple(unpack))
        if sig not in seen:
            seen.add(sig)
            widths = ace_sim_widths(s)
            rows.append(dict(torus=torus, lrc=lrc, lrr=lrr, boundary=sum(1 for u in unpack if len(u) > 1),
                             sims=len(widths), widths=widths, max_width=widths[0], unpack=unpack))
        del s
    _ACE_PLAN[key] = rows
    return rows


def circuit_couplers(lay, n, layout):
    """The circuit's CZ couplers among the first n logical qubits, in ACE register indices
    (every colour is used equally often, so each coupler counts once)."""
    _, _, C = ace_register(lay, n, layout)
    idx = [r * C + c for r, c in lay.pos[:n]]
    return sorted({tuple(sorted((idx[u], idx[v]))) for es in lay.matchings.values() for u, v in es if u < n and v < n})


def seam_couplers(r, couplers):
    """Couplers ACE cannot apply inside one simulator: either end is a seam (replicated)
    qubit, or the two ends live in different simulators. These are where ACE
    approximates; every other CZ is exact."""
    un = r["unpack"]
    return sum(1 for u, v in couplers if len(un[u]) > 1 or len(un[v]) > 1 or un[u][0][0] != un[v][0][0])


def with_seams(rows, couplers, tiling=None):
    """Layout rows (with seams) annotated with their seam-coupler count for this circuit;
    layouts with identical numbers listed once, under the smallest lrc + lrr.
    tiling = (lay, n, start sites): count after ace_tiling instead of on the device grid."""
    out, seen = [], set()
    for r in rows:
        if not r["boundary"]:
            continue
        if tiling:
            lay, n, start, *ver = tiling
            lc = logical_couplers(lay, n)
            site = ace_tiling(r, start, lc, len(r["unpack"]), ver[0] if ver else TILING_VERSION)
            sids = [frozenset(sid for sid, _ in u) for u in r["unpack"]]
            ex, rp, cr = coupler_classes(sids, site, lc)
            r = dict(r, seam_cz=rp + cr, cross_cz=cr)
        else:
            sids = [frozenset(sid for sid, _ in u) for u in r["unpack"]]
            ex, rp, cr = coupler_classes(sids, list(range(len(sids))), couplers)
            r = dict(r, seam_cz=rp + cr, cross_cz=cr)
        sig = (r["torus"], r["boundary"], r["seam_cz"], r["cross_cz"], tuple(r["widths"]))
        if sig not in seen:
            seen.add(sig)
            out.append(r)
    return out


def ace_rank(r):
    """Fewest couplers across simulators first (ends that share no simulator: the n=27
    fxeb layout scan showed these, not seam-replica couplers, track the XEB loss),
    then fewest couplers on a seam, then fewest simulators, seam qubits, footprint;
    a non-torus layout wins a tie (the device patch is not a torus)."""
    return (r["cross_cz"], r["seam_cz"], r["sims"], r["boundary"], r["max_width"],
            sum(2 ** w for w in r["widths"]), r["torus"], r["lrc"] + r["lrr"], r["lrc"])


def widest_ace_config(size, max_width, couplers, torus=None, tiling=None):
    """The --ace-max-width pick: the layout that approximates the fewest of this circuit's
    couplers with no internal simulator wider than max_width. Layouts without seams
    (one simulator, i.e. exact) are left to --backend exact."""
    toruses = (False, True) if torus is None else (bool(torus),)
    rows = with_seams(ace_plan(size, toruses), couplers, tiling)
    ok = [r for r in rows if r["max_width"] <= max_width]
    if not ok:
        narrow = min((r["max_width"] for r in rows), default=None)
        raise SystemExit(f"no ACE layout of a {size}-site register has all simulators <= {max_width} qubits"
                         + (f" (narrowest with seams: {narrow})" if narrow else ""))
    return min(ok, key=ace_rank)


TILING_VERSION = 3          # default for --ace-tiling; --ace-tiling-version 2 keeps the old placement
# v2: couplers only, 10 x across simulators + 1 x through a seam replica, one annealing run
# v3: couplers across simulators 20 (was 10), through a seam replica 1, + 6 per logical
#     qubit on a seam site + 2 per simulator holding bulk qubits; three restarts, one from
#     the device grid and two from the best (n-1)-qubit tiling extended by one qubit, so
#     adding a qubit never starts from scratch; results cached on disk
#     (nighthawk_tiles.json next to this script, or $NIGHTHAWK_TILE_CACHE).
#     Seam qubits dominate: in the 4/4 tiled big run XEB at d=12 fell monotonically with
#     seam qubits used (3, 5, 6, 8 -> 0.158, 0.101, 0.069, -0.015), and a trial weighting
#     that traded seam qubits for fewer simulators (n=18: 4 seam / 1 simulator against
#     1 seam / 2) dropped XEB from 0.48 to 0.06 and ran 4x slower
TILING_W3 = dict(cross=20, replica=1, sim=2, seam=6)
TILING_ITERS3, TILING_SEEDS3 = 150_000, (3, 1003, 2003)


def tiling_version(a):
    return int(getattr(a, "ace_tiling_version", None) or TILING_VERSION)


def coupler_classes(sids, site, couplers):
    """(exact, replica, cross) counts for logical couplers placed on ACE sites:
    exact   both ends bulk qubits of the same simulator (applied exactly)
    replica a seam qubit is involved, but the two ends share a simulator
    cross   the two ends share no simulator at all"""
    ex = rp = cr = 0
    for u, v in couplers:
        a, b = sids[site[u]], sids[site[v]]
        if len(a) == 1 and len(b) == 1 and a == b:
            ex += 1
        elif a & b:
            rp += 1
        else:
            cr += 1
    return ex, rp, cr


_TILES = {}


def ace_tiling(row, start, couplers, size, version=TILING_VERSION):
    """Placement of logical qubits 0..n-1 on ACE sites (see TILING_VERSION)."""
    if int(version) == 2:
        return _tiling_v2(row, start, couplers, size)
    if int(version) == 3:
        return _tiling_v3(row, start, couplers, size)
    raise SystemExit(f"unknown tiling version {version} (2 or 3)")


def placement_stats(row, site, couplers):
    """exact / replica / cross couplers, logical qubits on seam sites, simulators holding
    bulk qubits -- of one placement."""
    sids = [frozenset(sid for sid, _ in u) for u in row["unpack"]]
    ex, rp, cr = coupler_classes(sids, site, couplers)
    bulk = {next(iter(sids[x])) for x in site if len(sids[x]) == 1}
    return dict(exact=ex, replica=rp, cross=cr, seam_used=sum(1 for x in site if len(sids[x]) > 1),
                sims_used=len(bulk))


def _tile_cache_path():
    p = os.environ.get("NIGHTHAWK_TILE_CACHE")
    return Path(p) if p else Path(__file__).resolve().parent / "nighthawk_tiles.json"


def _tile_cache_get(key):
    try:
        return json.loads(_tile_cache_path().read_text()).get(key)
    except (OSError, ValueError):
        return None


def _tile_cache_put(key, site):
    """Merge one entry into the shared cache file (atomic replace; a lost race only costs a
    deterministic recomputation)."""
    p = _tile_cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            d = {}
        d[key] = site
        tmp = p.with_name(p.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(d))
        os.replace(tmp, p)
    except OSError:
        pass


def _cost3_parts(sids, bulk_of):
    W = TILING_W3

    def w(su, sv):
        a, b = sids[su], sids[sv]
        if len(a) == 1 and len(b) == 1 and a == b:
            return 0
        return W["replica"] if a & b else W["cross"]

    def total(site, couplers):
        c = sum(w(site[u], site[v]) for u, v in couplers)
        bulk = {bulk_of[x] for x in site if bulk_of[x] >= 0}
        return c + W["sim"] * len(bulk) + W["seam"] * sum(1 for x in site if bulk_of[x] < 0)
    return w, total


def _anneal3(sids, couplers, site0, size, iters, seed, exclude=()):
    """Simulated annealing of the v3 cost from site0 over sites not in exclude; returns
    (best placement, its cost)."""
    import random
    W = TILING_W3
    bulk_of = [next(iter(s)) if len(s) == 1 else -1 for s in sids]
    w, total = _cost3_parts(sids, bulk_of)
    n = len(site0)
    site = list(site0)
    used = set(site)
    excl = set(exclude)
    free = [x for x in range(size) if x not in used and x not in excl]
    adj = [[] for _ in range(n)]
    for u, v in couplers:
        adj[u].append(v)
        adj[v].append(u)
    cnt = collections.Counter(bulk_of[x] for x in site if bulk_of[x] >= 0)

    def local(q, sq, skip=-1):
        return sum(w(sq, site[x]) for x in adj[q] if x != skip)

    c = total(site, couplers)
    best_c, best = c, site[:]
    rng = random.Random(seed)
    T0, T1 = 8.0, 0.1
    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / iters)
        if free and rng.random() < 0.15:
            q, k = rng.randrange(n), rng.randrange(len(free))
            sq, sf = site[q], free[k]
            b1, b2 = bulk_of[sq], bulk_of[sf]
            dsim = 0
            if b1 != b2:
                dsim -= 1 if b1 >= 0 and cnt[b1] == 1 else 0
                dsim += 1 if b2 >= 0 and cnt[b2] == 0 else 0
            d = (local(q, sf) - local(q, sq) + W["sim"] * dsim
                 + W["seam"] * ((b2 < 0) - (b1 < 0)))
            if d <= 0 or rng.random() < math.exp(-d / T):
                site[q], free[k] = sf, sq
                if b1 >= 0:
                    cnt[b1] -= 1
                if b2 >= 0:
                    cnt[b2] += 1
                c += d
        elif n > 1:
            a_, b_ = rng.sample(range(n), 2)
            sa, sb = site[a_], site[b_]
            before = local(a_, sa, b_) + local(b_, sb, a_)
            after = local(a_, sb, b_) + local(b_, sa, a_)
            d = after - before
            if d <= 0 or rng.random() < math.exp(-d / T):
                site[a_], site[b_] = sb, sa
                c += d
        if c < best_c:
            best_c, best = c, site[:]
    return best, best_c


def _tiling_v3(row, start, couplers, size):
    n = len(start)
    couplers = [tuple(e) for e in couplers]
    key = ("v3", row["torus"], row["lrc"], row["lrr"], size, tuple(start), tuple(couplers))
    if key in _TILES:
        return _TILES[key]
    dkey = hashlib.sha1(json.dumps([3, TILING_W3, TILING_ITERS3, list(TILING_SEEDS3), row["unpack"],
                                    list(start), couplers]).encode()).hexdigest()
    hit = _tile_cache_get(dkey)
    if hit is not None:
        _TILES[key] = hit
        return hit
    sids = [frozenset(sid for sid, _ in u) for u in row["unpack"]]
    bulk_of = [next(iter(s)) if len(s) == 1 else -1 for s in sids]
    _, total = _cost3_parts(sids, bulk_of)
    starts = [list(start)]
    if n > 2:                                    # warm start: best (n-1) tiling + one qubit
        prev = _tiling_v3(row, start[:n - 1], [e for e in couplers if e[1] < n - 1], size)
        free = [x for x in range(size) if x not in set(prev)]
        ext = min((prev + [x] for x in free), key=lambda s_: total(s_, couplers))
        starts = [ext, list(start)]
    best = None
    for k, seed in enumerate(TILING_SEEDS3):
        site, c = _anneal3(sids, couplers, starts[k % len(starts)], size, TILING_ITERS3, seed)
        if best is None or c < best[1]:
            best = (site, c)
    _TILES[key] = best[0]
    _tile_cache_put(dkey, best[0])
    return best[0]


def _tiling_v2(row, start, couplers, size, iters=200_000, seed=2):
    """Placement of logical qubits 0..n-1 on ACE sites minimising 10 x (couplers across
    simulators) + (couplers through a seam replica), by simulated annealing over swaps and moves
    to free sites. Starts from `start` (the device-geometry placement), so it is never
    worse than that; deterministic for a given layout, circuit and seed."""
    key = (row["torus"], row["lrc"], row["lrr"], size, tuple(start), tuple(couplers), iters, seed)
    if key in _TILES:
        return _TILES[key]
    import random
    sids = [frozenset(sid for sid, _ in u) for u in row["unpack"]]
    n = len(start)
    site = list(start)
    used = set(site)
    free = [x for x in range(size) if x not in used]
    adj = [[] for _ in range(n)]
    for u, v in couplers:
        adj[u].append(v)
        adj[v].append(u)

    def w(su, sv):
        a, b = sids[su], sids[sv]
        if len(a) == 1 and len(b) == 1 and a == b:
            return 0
        return 1 if a & b else 10

    def local(q, sq, skip=-1):
        return sum(w(sq, site[x]) for x in adj[q] if x != skip)

    c = sum(w(site[u], site[v]) for u, v in couplers)
    best_c, best = c, site[:]
    rng = random.Random(seed)
    T0, T1 = 8.0, 0.1
    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / iters)
        if free and rng.random() < 0.1:
            q, k = rng.randrange(n), rng.randrange(len(free))
            sq, sf = site[q], free[k]
            d = local(q, sf) - local(q, sq)
            if d <= 0 or rng.random() < math.exp(-d / T):
                site[q], free[k] = sf, sq
                c += d
        elif n > 1:
            a_, b_ = rng.sample(range(n), 2)
            sa, sb = site[a_], site[b_]
            before = local(a_, sa, b_) + local(b_, sb, a_)
            after = local(a_, sb, b_) + local(b_, sa, a_)
            d = after - before
            if d <= 0 or rng.random() < math.exp(-d / T):
                site[a_], site[b_] = sb, sa
                c += d
        if c < best_c:
            best_c, best = c, site[:]
    _TILES[key] = best
    return best


def place(lay, n, a, row, size):
    """ACE site of each of the first n logical qubits: the device geometry, with
    --ace-tiling the annealed tiling of that geometry, with --ace-transpile the layout
    Qiskit's VF2Layout finds on ACE's own logical coupling map."""
    if getattr(a, "ace_transpile", False):
        return transpile_layout(lay, n, a, row, size)
    _, _, C = ace_register(lay, n, getattr(a, "ace_layout", "grid"))
    start = [r * C + c for r, c in lay.pos[:n]]
    if not getattr(a, "ace_tiling", False):
        return start
    return ace_tiling(row, start, logical_couplers(lay, n), size, tiling_version(a))


def ace_size(lay, n, a):
    """Number of ACE sites: --ace-width, else the device window (or strip) of ace_register."""
    w = getattr(a, "ace_width", None)
    return int(w) if w else ace_register(lay, n, getattr(a, "ace_layout", "grid"))[0]


def parse_exclude(spec):
    """'74,59,44' or '74+59+44' -> sorted tuple of ACE sites."""
    if not spec:
        return ()
    return tuple(sorted({int(x) for x in re.split(r"[,+ ]+", str(spec).strip()) if x}))


TRANSPILE_SEAM_EDGE_ERR, TRANSPILE_BULK_EDGE_ERR = 2e-2, 1e-4
# VF2 search bounds: on ACE's dense coupling graph (every pair inside a patch) the
# isomorphism search is unbounded without them; the defaults are set from timings on
# the 64- and 75-site registers (see README)
TRANSPILE_CROSS_W = 1000          # anneal weight of a coupler between patches with no seam between them
TRANSPILE_TRIALS, TRANSPILE_CALLS, TRANSPILE_TIME_S = 2000, 2_000_000, 120
TRANSPILE_SEAM_QUBIT_ERR, TRANSPILE_BULK_QUBIT_ERR = 1e-2, 1e-4


def ace_coupling_edges(row, size, torus, opts, exclude):
    """Undirected couplers of the ACE register: the backend's own get_logical_coupling_map()
    where this PyQrack has it (as mirror_nighthawk.py uses it), else every pair of sites
    sharing a simulator -- inside a patch the bulk is a dense state vector, i.e. fully
    connected. Couplers touching an excluded site are dropped (Dan's coupler_exclusions)."""
    ex = set(exclude)
    edges = None
    try:
        from pyqrack import QrackAceBackend
        b = ace_backend(size, long_range_columns=row["lrc"], long_range_rows=row["lrr"], is_torus=torus,
                            is_gpu=False, **(opts or {}))
        if hasattr(b, "get_logical_coupling_map"):
            edges = {tuple(sorted(map(int, e))) for e in b.get_logical_coupling_map() if e[0] != e[1]}
        del b
    except Exception:
        edges = None
    if edges is None:
        sids = [frozenset(sid for sid, _ in u) for u in row["unpack"]]
        edges = {(i, j) for i in range(size) for j in range(i + 1, size) if sids[i] & sids[j]}
    return sorted(e for e in edges if not (ex & set(e)))


_TX = {}


def _exact_on_register(row, size, couplers, edges, exclude, time_s, max_width=None, max_wide=None):
    """Exact placement by CP-SAT (OR-Tools). Sites are grouped by the set of simulators that
    hold them (a patch's bulk; a seam between two patches); sites of one group are
    interchangeable because a patch simulator is a dense, fully connected state vector.
    Each logical qubit gets a group, within the group capacities, such that every circuit
    coupler lands on a pair of groups the register couples (`edges`, ACE's own coupling map
    when PyQrack has it). Lexicographic objective: fewest qubits on seam sites (the three
    smallest boundaries), then the most even patch widths (bulk + seam replicas), then the
    fewest circuit couplers touching a seam. Returns (site list, status) or (None, status).
    max_width (with max_wide): hard memory caps, widths counted as ACE allocates them (logical
    qubits on the patch's sites plus its error-detection ancilla): every patch <= max_width,
    and at most max_wide patches at exactly max_width (the others <= max_width - 1)."""
    from ortools.sat.python import cp_model
    sids = [frozenset(sid for sid, _ in u) for u in row["unpack"]]
    ex = set(exclude)
    patch = sorted({next(iter(x)) for i, x in enumerate(sids) if len(x) == 1 and i not in ex}
                   | {next(iter(x)) for x in sids if len(x) == 1})
    groups = {}
    for i in range(size):
        if i not in ex:
            groups.setdefault(sids[i], []).append(i)
    G = sorted(groups, key=lambda g: (len(g), sorted(g)))
    eset = {tuple(sorted(e)) for e in edges}
    ok = {}
    for a in range(len(G)):
        for b in range(a, len(G)):
            ga, gb = groups[G[a]], groups[G[b]]
            ok[a, b] = ok[b, a] = any((min(i, j), max(i, j)) in eset for i in ga for j in gb if i != j)
    n = 1 + max(max(e) for e in couplers)
    m = cp_model.CpModel()
    x = {(q, g): m.NewBoolVar("") for q in range(n) for g in range(len(G))}
    for q in range(n):
        m.AddExactlyOne(x[q, g] for g in range(len(G)))
    for g in range(len(G)):
        m.Add(sum(x[q, g] for q in range(n)) <= len(groups[G[g]]))
    for u, v in couplers:
        for a in range(len(G)):
            for b in range(len(G)):
                if not ok[a, b]:
                    m.AddBoolOr([x[u, a].Not(), x[v, b].Not()])
    seamg = [g for g in range(len(G)) if len(G[g]) > 1]
    on_seam = [sum(x[q, g] for g in seamg) for q in range(n)]
    B = sum(on_seam)
    rep = []
    for u, v in couplers:
        r = m.NewBoolVar("")
        m.Add(r >= on_seam[u])
        m.Add(r >= on_seam[v])
        rep.append(r)
    widths = [sum(x[q, g] for q in range(n) for g in range(len(G)) if p in G[g]) for p in patch]
    if max_width:
        cap = int(max_width) - 1                 # - error-detection ancilla
        for w in widths:
            m.Add(w <= cap)
        if max_wide is not None:
            big = [m.NewBoolVar("") for _ in widths]
            for w, b_ in zip(widths, big):
                m.Add(w <= cap - 1 + b_)
            m.Add(sum(big) <= int(max_wide))
    wmax, wmin = m.NewIntVar(0, n, ""), m.NewIntVar(0, n, "")
    m.AddMaxEquality(wmax, widths)
    m.AddMinEquality(wmin, widths)
    m.Minimize(1_000_000 * B + 1_000 * (wmax - wmin) + sum(rep))
    sol = cp_model.CpSolver()
    sol.parameters.max_time_in_seconds = float(time_s)
    sol.parameters.num_workers = max(1, min(8, os.cpu_count() or 1))
    sol.parameters.random_seed = 0
    st = sol.Solve(m)
    name = sol.StatusName(st)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, name
    free = {g: list(groups[G[g]]) for g in range(len(G))}
    site = [0] * n
    for q in range(n):
        g = next(g for g in range(len(G)) if sol.Value(x[q, g]))
        site[q] = free[g].pop(0)
    return site, name


def _anneal_on_register(row, size, couplers, exclude, start=None):
    """Tiling cost (version 3 weights) annealed on an arbitrary ACE register with excluded
    sites: from the VF2 layout when Qiskit found one, else from a greedy fill (logical qubits
    in breadth-first order of the circuit graph onto allowed sites, bulk patch by patch,
    seam sites last). Three restarts; the best placement is kept."""
    sids = [frozenset(sid for sid, _ in u) for u in row["unpack"]]
    n = 1 + max(max(e) for e in couplers)
    excl = set(exclude)
    starts = []
    if start is not None:
        starts.append(list(start))
    adj = [[] for _ in range(n)]
    for u, v in couplers:
        adj[u].append(v)
        adj[v].append(u)
    order, seen = [], set()
    for root in range(n):
        if root in seen:
            continue
        queue = [root]
        seen.add(root)
        while queue:
            x = queue.pop(0)
            order.append(x)
            for y in sorted(adj[x]):
                if y not in seen:
                    seen.add(y)
                    queue.append(y)
    sites = sorted((x for x in range(size) if x not in excl),
                   key=lambda x: (len(sids[x]) > 1, min(sids[x]), x))
    if len(sites) < n:
        raise SystemExit(f"--ace-transpile: {n} qubits do not fit on {len(sites)} usable ACE sites")
    greedy = [0] * n
    for q, x in zip(order, sites):
        greedy[q] = x
    starts.append(greedy)
    best = None
    for k, seed in enumerate(TILING_SEEDS3):
        site, c = _anneal3(sids, couplers, starts[k % len(starts)], size, TILING_ITERS3, seed, excl)
        if best is None or c < best[1]:
            best = (site, c)
    return best[0]


def transpile_layout(lay, n, a, row, size):
    """Placement by Qiskit's VF2Layout pass on ACE's coupling map: a perfect layout, so every
    circuit coupler is a coupler of the ACE register and no routing (SWAP) is needed.
    Among the layouts it tries, VF2Layout keeps the one with the best error score; the
    target charges couplers that touch a seam site (and seam sites themselves) more than
    bulk ones, so it prefers placements with few seam crossings. Routing is deliberately not
    done: it would change the circuit, and the shared-circuit variants, twirling and
    estimators all assume the released gates. Deterministic for a given seed; cached in the
    tiling cache file."""
    exclude = parse_exclude(getattr(a, "ace_exclude", None))
    engine = getattr(a, "transpile_engine", "exact") or "exact"
    seed = int(getattr(a, "transpile_seed", 0) or 0)
    trials = int(getattr(a, "transpile_trials", 0) or 0) or TRANSPILE_TRIALS
    calls = int(getattr(a, "transpile_calls", 0) or 0) or TRANSPILE_CALLS
    couplers = logical_couplers(lay, n)
    torus = bool(getattr(a, "ace_torus", False))
    opts = ace_opts(a)
    strict = not getattr(a, "transpile_allow_cross", False)
    caps = transpile_caps(a)
    key = (engine, n, row["lrc"], row["lrr"], torus, size, exclude, seed, trials, calls, tuple(sorted(opts.items())),
           strict, caps, layout_id(lay))
    if key in _TX:
        return _TX[key]
    pre = preset_placement(lay, n, a, row, size)
    if pre is not None:
        _TX[key] = _check_no_cross(row, pre, couplers, True, size, exclude)
        return pre
    if engine == "exact":
        try:
            import ortools
        except ImportError:
            raise SystemExit("--transpile-engine exact needs OR-Tools: python3 -m pip install ortools "
                             "(or use --transpile-engine anneal)")
        edges = ace_coupling_edges(row, size, torus, opts, exclude)
        dkey = hashlib.sha1(json.dumps(["tx-exact", ortools.__version__, row["unpack"], list(exclude), couplers,
                                        edges, sorted(opts.items()), TRANSPILE_TIME_S]
                                       + ([list(caps)] if caps else [])).encode()).hexdigest()
        hit = _tile_cache_get(dkey)
        if hit is None:
            hit, status = _exact_on_register(row, size, couplers, edges, exclude, TRANSPILE_TIME_S, *(caps or ()))
            if hit is None:
                raise SystemExit(f"--ace-transpile exact: no placement of the {n}-qubit circuit on this ACE register "
                                 f"({size} sites, lrc {row['lrc']} lrr {row['lrr']}, {len(exclude)} excluded) "
                                 f"puts every coupler on a register coupler"
                                 + (f" within the width caps {caps}" if caps else "")
                                 + f" (CP-SAT: {status}); reopen excluded sites" + (" or relax the caps" if caps else ""))
            print(f"# ace transpile exact: CP-SAT {status}", flush=True)
            _tile_cache_put(dkey, hit)
        _TX[key] = _check_no_cross(row, hit, couplers, True, size, exclude)
        return hit
    try:
        import qiskit
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import CZGate, RZGate, SXGate
        from qiskit.circuit import Parameter
        from qiskit.transpiler import Target, InstructionProperties, PassManager
        from qiskit.transpiler.passes import VF2Layout
    except ImportError:
        raise SystemExit("--ace-transpile needs Qiskit: python3 -m pip install qiskit")
    W = dict(TILING_W3, cross=TRANSPILE_CROSS_W) if strict else TILING_W3
    dkey = hashlib.sha1(json.dumps(["tx", engine, W, TILING_ITERS3, list(TILING_SEEDS3), qiskit.__version__, row["unpack"], list(exclude), seed, trials, calls,
                                    couplers, sorted(opts.items())]).encode()).hexdigest()
    hit = _tile_cache_get(dkey)
    if hit is not None:
        _TX[key] = _check_no_cross(row, hit, couplers, strict, size, exclude)
        return hit
    edges = ace_coupling_edges(row, size, torus, opts, exclude)
    seam = {q for q, u in enumerate(row["unpack"]) if len(u) > 1}
    tgt = Target(num_qubits=size)
    tgt.add_instruction(CZGate(), {
        e2: InstructionProperties(error=TRANSPILE_SEAM_EDGE_ERR if (seam & set(e)) else TRANSPILE_BULK_EDGE_ERR)
        for e in edges for e2 in (e, e[::-1])})
    q_err = {q: InstructionProperties(error=TRANSPILE_SEAM_QUBIT_ERR if q in seam else TRANSPILE_BULK_QUBIT_ERR)
             for q in range(size)}
    tgt.add_instruction(SXGate(), {(q,): q_err[q] for q in range(size)})
    tgt.add_instruction(RZGate(Parameter("t")), {(q,): InstructionProperties(error=0.0) for q in range(size)})
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.sx(q)
    for u, v in couplers:
        qc.cz(u, v)
    kw = dict(target=tgt, seed=seed, strict_direction=False, max_trials=trials, call_limit=calls,
              time_limit=TRANSPILE_TIME_S)
    pm = PassManager([VF2Layout(**kw)])
    pm.run(qc)
    layout = pm.property_set.get("layout")
    vf2 = None
    if layout is not None:
        virt = layout.get_virtual_bits()
        vf2 = [int(virt[qc.qubits[q]]) for q in range(n)]
    if engine == "vf2":
        if vf2 is None:
            why = pm.property_set.get("VF2Layout_stop_reason")
            raise SystemExit(f"--ace-transpile vf2: no routing-free layout of the {n}-qubit circuit found on this ACE "
                             f"register ({size} sites, lrc {row['lrc']} lrr {row['lrr']}, {len(exclude)} excluded; "
                             f"VF2: {why}). Raise --transpile-calls, or use --transpile-engine anneal")
        site = vf2
    else:
        saved = dict(TILING_W3)
        TILING_W3.update(W)
        try:
            site = _anneal_on_register(row, size, couplers, exclude, vf2)
        finally:
            TILING_W3.clear()
            TILING_W3.update(saved)
    _tile_cache_put(dkey, site)
    _TX[key] = _check_no_cross(row, site, couplers, strict, size, exclude)
    return site


def _check_no_cross(row, site, couplers, strict, size, exclude):
    """A coupler between two patches that share no seam site is a contact ACE has no boundary
    for. With --ace-transpile it is refused unless --transpile-allow-cross is given."""
    st = placement_stats(row, site, couplers)
    if strict and st["cross"]:
        sids = [frozenset(sid for sid, _ in u) for u in row["unpack"]]
        bad = [(u, v) for u, v in couplers if not (sids[site[u]] & sids[site[v]])]
        seams = sorted(x for x in exclude if len(sids[x]) > 1)
        raise SystemExit(
            f"--ace-transpile: every placement found on this register ({size} sites, lrc {row['lrc']} "
            f"lrr {row['lrr']}, {len(exclude)} excluded) leaves {st['cross']} coupler(s) between patches with no "
            f"seam between them (logical {bad}). Reopen an excluded seam site"
            f"{' (one of ' + ','.join(map(str, seams)) + ')' if seams else ''}, or pass --transpile-allow-cross")
    return site


def logical_couplers(lay, n):
    return sorted({tuple(sorted(e)) for es in lay.matchings.values() for e in es if e[0] < n and e[1] < n})


def layout_row(size, lrc, lrr, torus, opts=None):
    """ace_plan's entry for one layout (the identical layout with the smallest lrc+lrr
    if this one was folded into it), for the given non-default ACE options."""
    from pyqrack import QrackAceBackend
    s = ace_backend(size, long_range_columns=lrc, long_range_rows=lrr, is_torus=torus, is_gpu=False,
                        **(opts or {}))
    unpack = [tuple(tuple(e) for e in s._unpack(q)) for q in range(s.num_qubits())]
    widths = ace_sim_widths(s)
    del s
    return dict(torus=torus, lrc=lrc, lrr=lrr, boundary=sum(1 for u in unpack if len(u) > 1),
                sims=len(widths), widths=widths, max_width=widths[0], unpack=unpack)


def torus_search(a):
    """Torus settings the --ace-max-width search may use: False (non-torus, the default,
    as the Nighthawk patch is not a torus), True, or None for both."""
    mode = getattr(a, "ace_torus_search", "flat")
    if getattr(a, "ace_torus", False):
        mode = "torus"
    return {"flat": False, "torus": True, "any": None}[mode]


def ace_register(lay, n, layout):
    """(sites, rows, cols) of the ACE register holding the first n logical qubits."""
    R, C = lay.grid
    if layout == "strip":
        R = max(4, max(r for r, _ in lay.pos[:n]) + 1)
    return R * C, R, C


class AceEngine:
    """QrackAceBackend on the device geometry: logical q -> its (row, col) of the
    subgrid, so couplers are nearest-neighbour in ACE's grid. One backend per worker and
    size, reset in place between runs.

    --ace-layout grid   the full 8x8 subgrid for every size (the original layout)
    --ace-layout strip  the smallest R x 8 strip of the subgrid holding the first n
                        logical qubits (R >= 4, so ACE lays it out as 8 columns), the
                        register nn_qab.py would use for that width
    --lrc/--lrr auto    nn_qab.py's two-patch rule for that register
    --ace-max-width W   fewest seam couplers with every internal simulator <= W qubits,
                        over the torus settings --ace-torus-search allows (default: non-torus)
    is_torus            False unless --ace-torus, --geometry nnqab or --ace-torus-search torus|any"""

    def __init__(self, lay, n, a):
        self.lay, self.n, self.a = lay, n, a
        self.size, R, C = ace_register(lay, n, getattr(a, "ace_layout", "grid"))
        if getattr(a, "ace_width", None):
            self.size = int(a.ace_width)            # --ace-transpile on a register of its own size
        grid_idx = [r * C + c for r, c in lay.pos[:n]]
        self.idx = grid_idx
        self.torus = torus = getattr(a, "ace_torus", False)
        lrc, lrr = a.lrc, a.lrr
        mw = getattr(a, "ace_max_width", None)
        tiling = (lay, n, grid_idx, tiling_version(a)) if getattr(a, "ace_tiling", False) else None
        if mw:
            pick = widest_ace_config(self.size, mw, circuit_couplers(lay, n, getattr(a, "ace_layout", "grid")),
                                     torus_search(a), tiling)
            lrc, lrr, torus = pick["lrc"], pick["lrr"], pick["torus"]
            self.torus = torus
        elif "auto" in (str(lrc), str(lrr)):
            alrc, alrr, *_ = two_patch_config(self.size, torus)
            lrc = alrc if str(lrc) == "auto" else int(lrc)
            lrr = alrr if str(lrr) == "auto" else int(lrr)
        self.lrc, self.lrr = int(lrc), int(lrr)
        self.opts = ace_opts(a)
        _, _, self.patches, self.boundary, self.b2b = ace_register_geometry(self.size, self.lrc, self.lrr, torus,
                                                                            self.opts)
        row = layout_row(self.size, self.lrc, self.lrr, torus, self.opts)
        self.idx = place(lay, n, a, row, self.size)
        self.stats = placement_stats(row, self.idx, logical_couplers(lay, n))
        self.coupler_classes = (self.stats["exact"], self.stats["replica"], self.stats["cross"])
        from pyqrack import QrackAceBackend
        # ACE on rusticl/Vega10 hung compute rings and forced GPU resets: CPU unless --ace-gpu
        self._make_sim = lambda: ace_backend(
            self.size, long_range_columns=self.lrc, long_range_rows=self.lrr, is_torus=torus,
            is_gpu=(not a.cpu) and getattr(a, "ace_gpu", False),
            is_host_pointer=getattr(a, "ace_host_pointer", False), **self.opts)
        self.sim = self._make_sim()
        self.broken = False
        rl, cl = self.sim.get_row_length(), self.sim.get_column_length()
        if not getattr(a, "ace_width", None) and (rl, cl) != (C, R):
            raise SystemExit(f"ACE chose a {rl}x{cl} grid for {self.size} qubits, expected {C} columns x {R} rows")
        self.widths = ace_sim_widths(self.sim)

    def geometry(self):
        return dict(ace_register=self.size, ace_lrc=self.lrc, ace_lrr=self.lrr, ace_torus=bool(self.torus),
                    ace_patches=self.patches, ace_boundary=self.boundary,
                    ace_b2b=round(self.b2b, 3) if self.boundary else None, ace_widths=self.widths,
                    ace_couplers=dict(zip(("exact", "replica", "cross"), self.coupler_classes)),
                    ace_tiling=bool(getattr(self.a, "ace_tiling", False)),
                    **({"ace_tiling_version": tiling_version(self.a)} if getattr(self.a, "ace_tiling", False) else {}),
                    ace_seam_used=self.stats["seam_used"], ace_sims_used=self.stats["sims_used"],
                    **({"ace_transpile": True, "ace_transpile_engine": getattr(self.a, "transpile_engine", "exact"),
                        "ace_exclude": list(parse_exclude(getattr(self.a, "ace_exclude", None))),
                        "ace_transpile_seed": int(getattr(self.a, "transpile_seed", 0) or 0),
                        "ace_transpile_allow_cross": bool(getattr(self.a, "transpile_allow_cross", False))}
                       if getattr(self.a, "ace_transpile", False) else {}),
                    **{attr: bool(getattr(self.a, attr, dflt)) for attr, _, dflt in ACE_OPTS
                       if bool(getattr(self.a, attr, dflt)) != dflt},
                    **({"gtt_gb": float(self.a.gtt), "vram_gb": float(getattr(self.a, "vram_gb", 8.0) or 8.0)}
                       if getattr(self.a, "gtt", None) and getattr(self.a, "ace_gpu", False) and not self.a.cpu else {}),
                    ace_map=hashlib.sha1(json.dumps(self.idx).encode()).hexdigest()[:10])

    def reset(self):
        """Back to |0...0> in place: measure all, flip the ones. No reallocation, so no
        new device buffers or uploads per twirl (checked exact: every P(1) = 0 after)."""
        v = self.sim.m_all()
        for q in range(self.size):
            if (v >> q) & 1:
                self.sim.x(q)

    def g1(self, q, M):
        self.sim.u(self.idx[q], *u_angles(M))

    def pauli(self, p, q):
        if p != "i":
            getattr(self.sim, p)(self.idx[q])

    def cz(self, a, b):
        self.sim.cz(self.idx[a], self.idx[b])

    def cx(self, a, b):
        self.sim.cx(self.idx[a], self.idx[b])

    def g2(self, g, a, b):
        qrack_g2(self.sim, g, self.idx[a], self.idx[b])

    def g2_inv(self, g, a, b):
        qrack_g2_inv(self.sim, g, self.idx[a], self.idx[b])

    def prob1(self, q):
        return float(self.sim.prob(self.idx[q]))

    def shots(self, s):
        """s samples of the logical qubits. QrackAceBackend.measure_shots clones the whole
        backend per shot; on the GPU that is new device buffers and copies for every shot
        (4096 shots of a tiled register: hours, at near-zero GPU load). With --ace-gpu the
        samples are drawn on a CPU copy of the final state instead, unless --ace-gpu-shots."""
        sim = self.sim
        if getattr(self.a, "ace_gpu", False) and not self.a.cpu and not getattr(self.a, "ace_gpu_shots", False):
            sim = self._cpu_twin()
        return np.array(sim.measure_shots(self.idx, s), dtype=np.uint64)

    def _cpu_twin(self):
        """CPU QrackAceBackend holding the GPU backend's state: same constructor arguments
        (so the same internal simulators, seams and layout), one out_ket/in_ket per internal
        simulator."""
        from pyqrack import QrackAceBackend
        widest = max(g.num_qubits() for g in self.sim.sim)
        if widest > CPU_TWIN_MAX_QB:
            # out_ket() returns every amplitude of an internal simulator as a Python list, unused
            # sites included: 2^31 complex numbers for a 31-qubit patch is tens of GB per patch
            raise RuntimeError(f"--ace-gpu shots: internal simulators up to {widest} qubits are too wide to copy "
                               f"to the CPU (limit {CPU_TWIN_MAX_QB}); use --mirror-estimator prob (mirror) "
                               f"or --ace-gpu-shots")
        twin = ace_backend(self.size, long_range_columns=self.lrc, long_range_rows=self.lrr,
                               is_torus=self.torus, is_gpu=False, **self.opts)
        if len(twin.sim) != len(self.sim.sim):
            raise RuntimeError(f"CPU copy has {len(twin.sim)} simulators, the GPU backend {len(self.sim.sim)}")
        for g, c in zip(self.sim.sim, twin.sim):
            if g.num_qubits() != c.num_qubits():
                raise RuntimeError(f"CPU copy simulator width {c.num_qubits()} != GPU {g.num_qubits()}")
            c.in_ket(g.out_ket())
        # ACE keeps classical state next to the simulators, which ace_backend(to_clone=...)
        # copies and a fresh backend does not: the LHV proxy of every boundary qubit (used by
        # prob/m/force_m through _correct) and the logical -> (simulator, index) map
        for k, lhv in getattr(self.sim, "_lhv", {}).items():
            if k in twin._lhv:
                twin._lhv[k].bloch = list(lhv.bloch)
        if hasattr(self.sim, "_qubits"):
            twin._qubits = [list(q) if isinstance(q, (list, tuple)) else q for q in self.sim._qubits] \
                if isinstance(self.sim._qubits, list) else \
                {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in self.sim._qubits.items()}
        return twin

    prob_bits = None

    def prob_target(self, bits, rng):
        """Probability that ACE's sampler returns `bits`, by the chain rule on the live state:
        in a random qubit order (as QrackAceBackend.m_all draws one), multiply the marginal of
        the wanted outcome and force it (ACE's own force_m, with the same seam correction as
        m). One pass per circuit run instead of measure_shots, which clones the whole backend
        for every single shot. Its expectation over the order is the survival measure_shots
        estimates; it collapses the state, which the next reset() clears."""
        P = 1.0
        for q in rng.permutation(self.n):
            lq, b = self.idx[q], bool(bits[q])
            try:
                p = float(self.sim.prob(lq))
                f = p if b else 1.0 - p
                if f <= PROB_TARGET_EPS:
                    return 0.0                  # below single-precision resolution: Qrack's force_m throws
                P *= f
                self.sim.force_m(lq, b)
            except RuntimeError:
                self.broken = True              # the backend's state is unreliable now: rebuild before reuse
                raise ProbTargetError(P)
        return P

    def rebuild(self):
        """Fresh backend after a C++ exception (same register, layout and options)."""
        try:
            del self.sim
        except AttributeError:
            pass
        self.sim = self._make_sim()
        self.broken = False


_GATE = {"a": None}


def gated(fn):
    """Run an OpenCL initialisation (first context + JIT, backend allocation) under one
    machine-wide lock, at least --init-gap seconds after the previous one finished, so
    workers never initialise at the same time over the cards' shared PCIe links."""
    a = _GATE["a"]
    if a is None or a.cpu or not a.init_gap:
        return fn()
    import fcntl
    lock = shots_root(a) / "initgate.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            txt = fh.read().strip()
            wait = (float(txt) if txt else 0.0) + a.init_gap - time.time()
            if wait > 0:
                time.sleep(wait)
            out = fn()
            fh.seek(0)
            fh.truncate()
            fh.write(f"{time.time():.3f}")
            fh.flush()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    return out


def make_engine(lay, n, a):
    return gated(lambda: AceEngine(lay, n, a) if a.backend == "ace" else ExactEngine(n, a.cpu))


def cz_layer(eng, es, rng):
    """CZ layer, Pauli-frame twirled when rng is given: P_a x P_b before, and after it
    CZ (P_a x P_b) CZ = (P_a Z^[P_b in XY]) x (P_b Z^[P_a in XY]) up to phase. Exact."""
    for a_, b_ in es:
        if rng is None:
            eng.cz(a_, b_)
            continue
        pa, pb = PAULIS[rng.integers(4)], PAULIS[rng.integers(4)]
        eng.pauli(pa, a_)
        eng.pauli(pb, b_)
        eng.cz(a_, b_)
        eng.pauli(pa, a_)
        eng.pauli(pb, b_)
        if pb in "xy":
            eng.pauli("z", a_)
        if pa in "xy":
            eng.pauli("z", b_)


def apply_forward(eng, cyc, rng=None, tick=None):
    for ang, es in cyc:
        for q, t in enumerate(ang):
            eng.g1(q, gate(t))
        cz_layer(eng, es, rng)
        if tick:
            tick()


def apply_inverse(eng, cyc, rng=None, tick=None):
    for ang, es in reversed(cyc):
        cz_layer(eng, list(reversed(es)), rng)
        for q in reversed(range(len(ang))):
            eng.g1(q, gate(ang[q]).conj().T)
        if tick:
            tick()


def progress_dir(a):
    """<shots-dir>/<cfg tag>/progress: one JSON file per busy worker (pid), read by --status."""
    return shots_root(a) / "progress"


PROB_TARGET_EPS = 1e-6   # marginal below which a forced outcome is treated as probability 0 (fp32-safe)


CPU_TWIN_MAX_QB = 26     # widest internal simulator the --ace-gpu CPU shot copy accepts (2^26 amplitudes)


class ProbTargetError(Exception):
    """Qrack raised inside prob_target; .args[0] is the partial product reached (an upper bound)."""


class Progress:
    """Per-point progress of a long point (mirror: inputs x twirls circuit runs, each of
    d cycles). Counts cycles, so even a single slow circuit run shows movement. Writes
    <progress dir>/<pid>.json at most every PROGRESS_FILE_S and prints a log line at most
    every PROGRESS_LOG_S (and after the first circuit run, which fixes the speed)."""

    def __init__(self, a, label, runs, cycles_per_run):
        self.a, self.label, self.runs, self.cpr = a, label, max(1, runs), max(1, cycles_per_run)
        self.total = self.runs * self.cpr
        self.done = 0
        self.t0 = self.t_file = self.t_log = time.time()
        self.first_logged = False
        self.path = None
        try:
            d = progress_dir(a)
            d.mkdir(parents=True, exist_ok=True)
            self.path = d / f"{os.getpid()}.json"
        except Exception:                          # no --out (selftest) or unwritable: log only
            self.path = None
        self._write(force=True)

    def state(self):
        el = time.time() - self.t0
        f = self.done / self.total
        eta = el / f - el if f > 0 else None
        return dict(point=self.label, pid=os.getpid(), host=os.uname().nodename, runs=self.runs,
                    run=min(self.runs, self.done // self.cpr), cycles_done=self.done, cycles_total=self.total,
                    fraction=f, elapsed_s=el, eta_s=eta, started=self.t0, updated=time.time())

    def _write(self, force=False):
        now = time.time()
        if self.path is None or (not force and now - self.t_file < PROGRESS_FILE_S):
            return
        self.t_file = now
        try:
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self.state()))
            os.replace(tmp, self.path)
        except OSError:
            pass

    def tick(self):
        self.done += 1
        self._write()
        now = time.time()
        first = not self.first_logged and self.done >= self.cpr
        if first or now - self.t_log >= PROGRESS_LOG_S:
            self.t_log, self.first_logged = now, self.first_logged or first
            print(f"# progress {self.label}: {progress_line(self.state())}", flush=True)

    def close(self):
        if self.path is not None:
            try:
                self.path.unlink()
            except OSError:
                pass


PROGRESS_FILE_S, PROGRESS_LOG_S = 20.0, 300.0


def _hms(s):
    if s is None:
        return "?"
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def progress_line(st):
    return (f"circuit run {st['run']}/{st['runs']}, {100 * st['fraction']:.1f}% of cycles, "
            f"elapsed {_hms(st['elapsed_s'])}, ETA {_hms(st['eta_s'])}")


# ================================================================== estimators (rcs/estimators.py)
def extract_bits(shots, qubits):
    shots = np.asarray(shots, dtype=np.uint64)
    out = np.zeros(shots.shape, dtype=np.int64)
    for j, q in enumerate(qubits):
        out |= ((shots >> np.uint64(q)) & np.uint64(1)).astype(np.int64) << j
    return out


def ideal_xeb(p):
    return float(p.size * np.dot(p, p) - 1.0)


def _delta_se(means, cov):
    k = len(means)
    grad = np.array([np.prod(np.delete(means, i)) for i in range(k)])
    hess = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            hess[i, j] = hess[j, i] = np.prod(np.delete(means, [i, j]))
    a = cov @ hess
    return math.sqrt(max(float(grad @ cov @ grad) + 0.5 * float(np.trace(a @ a)), 0.0))


def patched_xeb(shots, probs, patches):
    """Eq. (1)-(2): prod_r mean_s[(2^n_r p_r(x_r) - 1) / (2^n_r sum p_r^2 - 1)]."""
    if len(shots) < 2:
        raise ValueError("need at least two shots")
    norms = [ideal_xeb(p) for p in probs]
    scores = np.array([(p.size * p[extract_bits(shots, qs)] - 1.0) / nz
                       for p, qs, nz in zip(probs, patches, norms)])
    means = scores.mean(axis=1)
    cov = np.atleast_2d(np.cov(scores, bias=True)) / len(shots)
    return dict(fidelity=float(np.prod(means)), se=_delta_se(means, cov),
                patch_fidelity=[float(m) for m in means], patch_ideal_xeb=norms, shots=int(len(shots)))


def inverse_variance_mean(v, se):
    v, se = np.asarray(v, float), np.maximum(np.asarray(se, float), 1e-12)
    w = 1 / se ** 2
    return float(np.sum(w * v) / np.sum(w)), float(math.sqrt(1 / np.sum(w)))


def fit_exp(depths, F):
    s, i = np.polyfit(np.asarray(depths, float), np.log10(np.asarray(F, float)), 1)
    return float(10 ** i), float(10 ** s)


# ================================================================== ideal patch distributions
def patch_probs(cyc, patches, cpu):
    """Exact distribution of every patch sub-circuit (qubit j of patch = sorted[j]),
    little-endian as Qiskit / out_probs; renormalised in float64."""
    from pyqrack import QrackSimulator
    out = []
    for qs in patches:
        loc = {q: j for j, q in enumerate(qs)}
        if _GATE.get("warm"):
            sim = QrackSimulator(len(qs), is_gpu=not cpu)
        else:
            sim = gated(lambda: QrackSimulator(len(qs), is_gpu=not cpu))
            _GATE["warm"] = True
        for ang, es in cyc:
            for q in qs:
                sim.mtrx(flat(gate(ang[q])), loc[q])
            for a_, b_ in es:
                if a_ in loc and b_ in loc:
                    sim.mcz([loc[a_]], loc[b_])
                elif (a_ in loc) != (b_ in loc):
                    raise RuntimeError(f"CZ {a_}-{b_} crosses a patch boundary")
        p = np.asarray(sim.out_probs(), dtype=np.float64)
        out.append(p / p.sum())
        del sim
    return out


def cached_patch_probs(cache, key, cyc, patches, cpu):
    """Ideal patch distributions, from the float32 cache when present. Cached arrays are
    renormalised in float64 on load, exactly as freshly computed ones are, so a cached
    and an uncached run score identically."""
    if cache:
        f = Path(cache) / (key.replace("/", "_") + ".npz")
        if f.exists():
            out = []
            with np.load(f) as z:
                for r in range(len(patches)):
                    p = z[f"p{r}"].astype(np.float64)
                    out.append(p / p.sum())
            return out
    probs = patch_probs(cyc, patches, cpu)
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, **{f"p{r}": p.astype(np.float32) for r, p in enumerate(probs)})
    return probs


# ================================================================== hwxeb: the paper's numbers, Qrack as reference
def cmd_hwxeb(a):
    lay = Layout(a.repo)
    ref = {(r["K"], r["depth"], r["partition"], r["instance"]): r
           for r in json.loads((lay.repo / "data/results/patch_xeb.json").read_text())}
    fvd = json.loads((lay.repo / "data/results/fidelity_vs_depth.json").read_text())
    depths = a.depths or lay.depths["patched"]
    recs, worst_f, worst_i, t0 = [], 0.0, 0.0, time.time()
    for K in a.K:
        for d in depths:
            shots = np.load(lay.repo / f"data/counts/patched_K{K}_d{d}.npz")
            done = []
            for j, part in enumerate(lay.partitions[K]):
                for i in range(lay.instances):
                    if a.limit and len(recs) >= a.limit:
                        break
                    probs = cached_patch_probs(a.cache, f"K{K}_d{d}_p{j}_i{i}", patched_cycles(lay, K, d, j, i),
                                               part["patches"], a.cpu)
                    r = patched_xeb(shots[f"partition{j}_instance{i}"], probs, part["patches"])
                    rr = ref.get((K, d, j, i))
                    if rr:
                        worst_f = max(worst_f, abs(r["fidelity"] - rr["fidelity"]) / max(rr["se"], 1e-12))
                        ideal_ref = sorted(rr["patch_ideal_xeb"][str(k)] for k in range(K))
                        worst_i = max(worst_i, max(abs(x - y) for x, y in zip(sorted(r["patch_ideal_xeb"]), ideal_ref)))
                    rec = dict(K=K, depth=d, partition=j, instance=i, **r)
                    recs.append(rec)
                    done.append(rec)
            if done:
                F, se = inverse_variance_mean([r["fidelity"] for r in done], [r["se"] for r in done])
                cr = np.mean([x for r in done for x in r["patch_ideal_xeb"]])
                pr = fvd["points"][f"{K}-patch"].get(str(d), {})
                print(f"K={K} d={d:>2}: {len(done):>2} circuits  F = {F:.3e} +/- {se:.1e}"
                      f"  (release {pr.get('fidelity', float('nan')):.3e})  collision ratio {cr:.4f}"
                      f"  (release {fvd['collision_ratio'][f'K{K}'].get(str(d), float('nan')):.4f})"
                      f"  [{time.time() - t0:.0f}s]", flush=True)
    print(f"\nworst |F_qrack - F_release| = {worst_f:.3f} sigma; worst |ideal XEB diff| = {worst_i:.2e}"
          f"  (PyQrack float32 wheels agree to ~1e-5; {'OK' if worst_f < 0.05 and worst_i < 1e-3 else 'CHECK'})")

    ms = json.loads((lay.repo / "data/counts/mirror_survival.json").read_text())["depths"]
    pts = {}
    for d, e in ms.items():
        h, s = np.asarray(e["hits"], float).sum(-1), np.asarray(e["shots"], float).sum(-1)
        p = h / s
        pts[int(d)] = inverse_variance_mean(p, np.sqrt(np.maximum(p * (1 - p), 1e-12) / s))
    dd = sorted(pts)
    A, f = fit_exp(dd, [pts[d][0] for d in dd])
    print(f"\nmirror fit from the hardware counts: F(d) = {A:.4f} x {f:.4f}^d "
          f"(release {fvd['fit']['prefactor']:.4f} x {fvd['fit']['fidelity_per_cycle']:.4f}^d); "
          f"F(36) = {A * f ** 36:.3e}; error/qubit/cycle {-math.log(f) / lay.n:.2e}")
    if a.out:
        Path(a.out).write_text(json.dumps(recs, indent=1))
        print("wrote", a.out)


# ================================================================== danmodel: Dan Strano's ACE XEB calculator
# The "XEB prediction calculator" sheet of Dan Strano's ACE spreadsheet (QrackAceBackend, PyQrack
# 2.28.2, error detection on, torus, SDRP off, native SWAP; nn_qab circuits of 10-27 qubits, 2-3
# patches, fitted at depth 12 and over depths 14-22): x = (bulk / (boundary * patches))^(2/3),
# XEB_12 = tanh(slope * x + intercept), XEB(d) = XEB_12 ^ ((d / 12)^p).
DAN_MODEL = dict(power=2 / 3, slope=0.1653638846, intercept=-0.0378757892, depth_power=1.60569, d0=12)
LADDER = ("nighthawk2p16", "nighthawk2p20", "nighthawk2p24", "nighthawk2p28", "nighthawk2p32", "nighthawk2p36", "nighthawk2p40", "nighthawk2p45", "nighthawk2p49",
          "nighthawk2p53", "nighthawk2p57", "nighthawk61r2")


def dan_xeb(bulk, boundary, patches, depth, m=DAN_MODEL):
    """Dan's calculator. No boundary (one exact patch): 1. A non-positive depth-12 XEB: 0."""
    if boundary <= 0 or patches <= 1:
        return 1.0
    x = (bulk / (boundary * patches)) ** m["power"]
    x12 = math.tanh(m["slope"] * x + m["intercept"])
    return 0.0 if x12 <= 0 else x12 ** ((depth / m["d0"]) ** m["depth_power"])


def device_mirror(repo=None):
    """ibm_phoenix mirror survival per depth from the release's counts (inverse-variance mean over
    instances, as hwxeb), or {} without the release."""
    try:
        ms = json.loads((find_repo(repo) / "data/counts/mirror_survival.json").read_text())["depths"]
    except (OSError, ValueError, SystemExit):
        return {}
    out = {}
    for d, e in ms.items():
        h, s = np.asarray(e["hits"], float).sum(-1), np.asarray(e["shots"], float).sum(-1)
        p = h / s
        out[int(d)] = inverse_variance_mean(p, np.sqrt(np.maximum(p * (1 - p), 1e-12) / s))[0]
    return out


def register_dan_inputs(name, repo=None, error_detection=True):
    """(n, bulk, boundary, patches, widths) of a register's stored ACE placement."""
    spec = REGISTERS[name]
    ace = spec["ace"]
    row = layout_row(ace["width"], ace["lrc"], ace["lrr"], ace["torus"], {})
    sids = [frozenset(s for s, _ in u) for u in row["unpack"]]
    site = ace["site"]
    B = sum(1 for x in site if len(sids[x]) > 1)
    patches = {next(iter(sids[x])) for x in site if len(sids[x]) == 1}
    widths = [sum(1 for x in site if p in sids[x]) + (1 if error_detection else 0) for p in sorted(patches)]
    return spec["n"], spec["n"] - B, B, len(patches), widths


def cmd_danmodel(a):
    depths = a.depths or [4, 6, 8, 10, 12, 16, 20, 36]
    dev = device_mirror(a.repo) if a.device else {}
    rows = []
    nonfinite = 0
    if a.inputs:
        groups, seen = {}, {}
        for path in a.inputs:
            for f in record_files(path):
                try:
                    lines = f.read_text().splitlines()
                except OSError:
                    continue
                for line in lines:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if (r.get("family") not in ("mirror", "fxeb", "hamming") or "fidelity" not in r
                            or "ace_seam_used" not in r):
                        continue
                    if not math.isfinite(float(r["fidelity"])):
                        nonfinite += 1
                        continue
                    fold = r.get("fold", 1) if r["family"] == "hamming" else 0
                    seen[(path, r["cfg"], r["n"], r["family"], r["depth"], fold, r["instance"])] = r
        for (path, cfg, n, fam, d, fold, i), r in seen.items():
            k = (path, cfg, r.get("theta", "haar"), fam, n, r["ace_seam_used"], r.get("ace_sims_used", 2), d, fold)
            groups.setdefault(k, []).append(r)
        for (path, cfg, theta, fam, n, B, P, d, fold), rs in sorted(
                groups.items(), key=lambda kv: (kv[0][4], kv[0][0], kv[0][3], kv[0][7], kv[0][8])):
            v = np.array([r["fidelity"] for r in rs], float)
            se = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float(rs[0].get("se", float("nan")))
            # hamming: d = the depth of U (his script's depth argument), the calculator's depth; the mirror
            # runs 2 d lambda layers (dan_2d: the calculator at the mirror's 2 d layers, a stricter pairing)
            rows.append(dict(source=Path(path).name, cfg=cfg, theta=theta, family=fam if fam != "hamming" else f"ham{fold}",
                             n=n, bulk=n - B, boundary=B, patches=P, depth=d, fold=fold,
                             cycles=d if fam == "hamming" else None, layers=2 * d * fold if fam == "hamming" else d,
                             instances=len(v), measured=float(v.mean()), se=se,
                             # the calculator models an unfolded circuit: folded rows (lambda > 1) get none
                             dan=dan_xeb(n - B, B, P, d) if fold <= 1 else None,
                             dan_2d=dan_xeb(n - B, B, P, 2 * d) if fam == "hamming" and fold == 1 else None,
                             device=dev.get(2 * d if fam == "hamming" else d) if n == 61 and fam != "fxeb" and fold <= 1
                             else None))
    else:
        for name in a.registers or LADDER:
            if name not in REGISTERS or "ace" not in REGISTERS[name]:
                raise SystemExit(f"danmodel: {name} is not a register with a stored ACE placement")
            n, bulk, B, P, w = register_dan_inputs(name, a.repo)
            for d in depths:
                rows.append(dict(source=name, n=n, bulk=bulk, boundary=B, patches=P, widths=w, depth=d,
                                 dan=dan_xeb(bulk, B, P, d), device=dev.get(d) if n == 61 else None))
    if nonfinite:
        print(f"# {nonfinite} record(s) with a non-finite score skipped")
    if not rows:
        raise SystemExit("danmodel: no mirror/fxeb/hamming ACE records in the given files")
    x_of = lambda r: (r["bulk"] / (r["boundary"] * r["patches"])) ** DAN_MODEL["power"] if r["boundary"] else float("inf")
    print("# Dan's calculator: x = (bulk/(boundary*patches))^(2/3); XEB_12 = tanh(%.6f x %+.6f); "
          "XEB(d) = XEB_12^((d/12)^%.5f)" % (DAN_MODEL["slope"], DAN_MODEL["intercept"], DAN_MODEL["depth_power"]))
    hdr = f"{'source':<26} {'n':>3} {'bulk':>4} {'bnd':>3} {'P':>2} {'x':>6} {'d':>3}  {'Dan XEB':>10}"
    if a.inputs:
        hdr += f"  {'theta':>5} {'fam':>6} {'inst':>4}  {'measured':>18}  {'meas/Dan':>8}"
    if dev:
        hdr += f"  {'ibm_phoenix':>11}"
    print(hdr)
    last = None
    for r in rows:
        if last is not None and (r["source"], r["n"]) != last:
            print()
        last = (r["source"], r["n"])
        s = (f"{r['source'][:26]:<26} {r['n']:>3} {r['bulk']:>4} {r['boundary']:>3} {r['patches']:>2} "
             f"{x_of(r):>6.3f} {r['depth']:>3}  " + (f"{r['dan']:>10.4g}" if r["dan"] is not None else f"{'-':>10}"))
        if a.inputs:
            ratio = r["measured"] / r["dan"] if r["dan"] else float("nan")
            s += (f"  {r['theta']:>5} {r['family']:>6} {r['instances']:>4}  "
                  f"{r['measured']:>9.4g} +/- {r['se']:<6.2g}  {ratio:>8.3g}")
        if dev:
            s += f"  {r['device']:>11.4g}" if r.get("device") is not None else f"  {'':>11}"
        print(s)
    zne = defaultdict(dict)
    for r in rows:
        if r.get("fold"):
            zne[(r["source"], r["cfg"], r["n"], r["cycles"])][r["fold"]] = r["measured"]
    zne = {k: v for k, v in zne.items() if len(v) >= 2}
    if zne:
        p = DAN_MODEL["depth_power"]
        print(f"\n# Dan's zero-noise extrapolation of the Hamming-weight signal (qem/richardson_nn_hamming_weight.py): "
              f"Richardson in u = lambda^{p:g} on ln(signal); ideal 1")
        print(f"{'source':<26} {'n':>3} {'cycles':>6}  {'signal at lambda = ...':<34} {'mitigated':>9}")
        for (src, cfg, n, d), by in sorted(zne.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][3])):
            lams = sorted(by)
            pts = "  ".join(f"{l}:{by[l]:.4f}" for l in lams)
            if all(by[l] > 0 for l in lams):
                y0 = math.exp(richardson_lambda_p(lams, [math.log(by[l]) for l in lams], p)[0])
                mit = f"{y0:>9.4f}"
            else:
                mit = f"{'(<= 0)':>9}"
            print(f"{src[:26]:<26} {n:>3} {d:>6}  {pts:<34} {mit}")
            for r in rows:
                if (r["source"], r["cfg"], r["n"], r.get("cycles")) == (src, cfg, n, d) and r.get("fold") == lams[0]:
                    r["mitigated"] = float(mit) if mit.strip() != "(<= 0)" else None
    if a.csv:
        import csv
        keys = sorted({k for r in rows for k in r})
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("wrote", a.csv)


# ================================================================== hamming: the E18 ladder against Dan's calculator
def hamming_inputs(paths):
    """Main --out files to read: the given ones, or dan_hamming_*.jsonl in the working directory.
    Worker files (<stem>.w<pid>.jsonl) are not listed here: record_files() adds them per main file."""
    import glob
    worker = re.compile(r"\.w\d+\.jsonl$")
    if paths:
        return [p for p in paths if not worker.search(p)]
    return sorted(p for p in glob.glob("dan_hamming_*.jsonl") if not worker.search(p))


def cmd_hamming(a):
    """Dan Strano's mirror Hamming-weight signal (run --families hamming) per register, against his
    calculator at the depth d of U (and at the mirror's 2 d layers), with the coupler load of the seam,
    a fitted depth power, a measured / Dan grid and his Richardson extrapolation over the folds."""
    files = hamming_inputs(a.inputs)
    if not files:
        raise SystemExit("hamming: no input files (give --in, or run where dan_hamming_*.jsonl are)")
    recs, geo, cfgs, nonfinite = {}, {}, defaultdict(set), 0
    for path in files:
        for f in record_files(path):
            try:
                lines = f.read_text().splitlines()
            except OSError:
                continue
            for line in lines:
                if not line.startswith("{"):
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("family") != "hamming" or "fidelity" not in r or "ace_seam_used" not in r:
                    continue
                if a.protocol != "any" and r.get("protocol", "native") != a.protocol:
                    continue
                if not math.isfinite(float(r["fidelity"])):
                    nonfinite += 1
                    continue
                n = r["n"]
                cpl = r.get("ace_couplers") or {}
                geo[n] = (r["ace_seam_used"], r.get("ace_sims_used", 2), cpl.get("replica", 0) + cpl.get("cross", 0))
                cfgs[n].add(r["cfg"])
                # one record per circuit; a later line (rerun, retry) replaces an earlier one
                recs[(n, r["depth"], r.get("fold", r["K"]), r["instance"])] = float(r["fidelity"])
    if nonfinite:
        print(f"# {nonfinite} record(s) with a non-finite score skipped")
    if not recs:
        raise SystemExit("hamming: no hamming records" + (f" with --protocol {a.protocol}" if a.protocol != "any" else ""))
    data = defaultdict(lambda: defaultdict(list))
    for (n, d, lam, _), v in recs.items():
        data[n][(d, lam)].append(v)
    p0 = DAN_MODEL["depth_power"]
    rows = []

    def mean_se(v):
        v = np.array(v, float)
        return float(v.mean()), (float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan"))

    print(f"# signal 1 - 2h at lambda = 1, plain mean over circuits; Dan's calculator at the depth d of U "
          f"(and at the mirror's 2d layers); {len(files)} file(s), protocol {a.protocol}")
    for n in sorted(data):
        B, P, nc = geo[n]
        note = f"; {len(cfgs[n])} configurations merged" if len(cfgs[n]) > 1 else ""
        print(f"n {n}: seam {B}, patches {P}, couplers through the seam {nc} ({nc / B:.2f} per seam qubit){note}")
        pts = []
        for d in sorted(d for (d, lam) in data[n] if lam == 1):
            m, se = mean_se(data[n][(d, 1)])
            k = len(data[n][(d, 1)])
            dn, d2 = dan_xeb(n - B, B, P, d), dan_xeb(n - B, B, P, 2 * d)
            ratio = m / dn if dn else float("nan")
            print(f"   d {d:>2}  {m:.3f} ± {se:.3f}  ({k:>2})  Dan(d) {dn:.3f}  ratio {ratio:.2f}  Dan(2d) {d2:.4f}")
            rows.append(dict(n=n, seam=B, patches=P, seam_couplers=nc, couplers_per_seam=round(nc / B, 3), depth=d,
                             fold=1, circuits=k, signal=m, se=se, dan=dn, ratio=ratio, dan_2d=d2))
            if 0 < m < 1 and d >= a.fit_from:
                pts.append((d, math.log(d / DAN_MODEL["d0"]), math.log(-math.log(m))))
        if len(pts) >= 2:
            _, X, Y = (np.array(c) for c in zip(*pts))
            p, c = np.polyfit(X, Y, 1)
            print(f"   fitted p {p:.2f}, signal(12) {math.exp(-math.exp(c)):.3f} over d = {pts[0][0]}-{pts[-1][0]}"
                  f" ({len(pts)} depths)  (Dan: {p0}, {dan_xeb(n - B, B, P, 12):.3f})"
                  + ("  [3 depths or fewer: indicative only]" if len(pts) <= 3 else ""))

    # measured / Dan grid, registers sorted by coupler load
    order = sorted(data, key=lambda n: (geo[n][2] / geo[n][0], n))
    depths = sorted({d for n in data for (d, lam) in data[n] if lam == 1})
    by = {(r["n"], r["depth"]): r for r in rows}
    print(f"\n# measured / Dan, lambda = 1, registers by couplers per seam qubit (Dan's strips: {a.threshold:.1f}); "
          f"(k) = only k circuits, fewer than {a.min_circuits}")
    print(f"{'n':>9} " + " ".join(f"{n:>8}" for n in order))
    print(f"{'cpl/seam':>9} " + " ".join(f"{geo[n][2] / geo[n][0]:>8.2f}" for n in order))
    for d in depths:
        cells = []
        for n in order:
            r = by.get((n, d))
            if r is None:
                cells.append(f"{'-':>8}")
            else:
                s = f"{r['ratio']:.2f}" + (f"({r['circuits']})" if r["circuits"] < a.min_circuits else "")
                cells.append(f"{s:>8}")
        print(f"{'d = ' + str(d):>9} " + " ".join(cells))

    # folding: Dan's qem/richardson_nn_hamming_weight.py, ln(signal) to lambda = 0 in u = lambda^p
    print(f"\n# folding: signal at lambda = 1 / 3 / 5 ... and Richardson (ln signal, u = lambda^{p0:g}) to lambda = 0")
    for n in sorted(data):
        for d in sorted({d for (d, _) in data[n]}):
            lams = sorted(l for (dd, l) in data[n] if dd == d)
            ms = [mean_se(data[n][(d, l)])[0] for l in lams]
            ks = [len(data[n][(d, l)]) for l in lams]
            mit = (math.exp(richardson_lambda_p(lams, [math.log(x) for x in ms], p0)[0])
                   if len(lams) > 1 and min(ms) > 0 else float("nan"))
            print(f"n {n:>3} d {d:>2}  lambda " + "/".join(map(str, lams)) + ":  " + " / ".join(f"{x:.3f}" for x in ms)
                  + f"   mitigated {mit:.3f}   circuits " + " / ".join(map(str, ks)))
            for r in rows:
                if (r["n"], r["depth"]) == (n, d):
                    r.update({f"signal_l{l}": x for l, x in zip(lams, ms)}, mitigated=mit)
    if a.csv:
        import csv
        keys = sorted({k for r in rows for k in r})
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("wrote", a.csv)


# ================================================================== run: clean-qubit emulation
def ace_geometry(lay, lrc, lrr, torus):
    """Boundary qubits, bulk components and bulk-to-boundary ratio of the ACE register
    as built, read from QrackAceBackend._unpack() the way nn_qab.py measures it."""
    from pyqrack import QrackAceBackend
    R, C = lay.grid
    sim = ace_backend(R * C, long_range_columns=lrc, long_range_rows=lrr, is_torus=torus, is_gpu=False)
    bnd = {q for q in range(R * C) if len(sim._unpack(q)) > 1}
    del sim
    bulk, seen, comps = set(range(R * C)) - bnd, set(), []
    for s0 in sorted(bulk):
        if s0 in seen:
            continue
        stack, size = [s0], 0
        seen.add(s0)
        while stack:
            v = stack.pop()
            size += 1
            r, c = divmod(v, C)
            for rr, cc in ((r, c + 1), (r + 1, c), (r, c - 1), (r - 1, c)):
                w = rr * C + cc
                if 0 <= rr < R and 0 <= cc < C and w in bulk and w not in seen:
                    seen.add(w)
                    stack.append(w)
        comps.append(size)
    ratio = len(bulk) / len(bnd) if bnd else float("inf")
    return dict(boundary=len(bnd), patches=sorted(comps, reverse=True), b_to_b=ratio)


def resolve_geometry(lay, a):
    """--geometry nnqab: nn_qab.py's rule on this register. Rows left whole (lrr = column
    length), lrc chosen for exactly two patches at the highest finite bulk-to-boundary
    ratio, ties broken toward the most balanced patches, torus. On the 8x8 Nighthawk
    window this is lrc=3, lrr=8: seam columns 3 and 7, two 24-qubit patches, B-to-B 3."""
    if getattr(a, "geometry", "manual") != "nnqab" or a.backend != "ace":
        return None
    R, C = lay.grid
    best = None
    for lrc in range(1, C):
        g = ace_geometry(lay, lrc, R, True)
        if len(g["patches"]) != 2 or g["b_to_b"] == float("inf"):
            continue
        key = (g["b_to_b"], -abs(g["patches"][0] - g["patches"][1]))
        if best is None or key > best[0]:
            best = (key, lrc, g)
    if best is None:
        raise SystemExit("nnqab geometry: no lrc gives exactly two patches on this register")
    _, a.lrc, g = best
    a.lrr, a.ace_torus = R, True
    return g


def cfg_tag(a):
    """Everything but the register size: records of every n share one tag, so a sweep
    can be extended with more sizes and resumes per (n, point).

    With --lrc/--lrr auto the tag holds the literal 'auto' (kept that way so existing
    runs still resume); the lrc/lrr actually used are written into every record
    (ace_lrc, ace_lrr, ace_torus, ...) and summarize() warns if records under one tag
    and one size disagree, e.g. after a change to two_patch_config()."""
    keys = dict(backend=a.backend, shots=a.shots, twirls=a.twirls,
                **({"theta": a.theta} if getattr(a, "theta", "haar") != "haar" else {}),
                **({"protocol": a.protocol} if getattr(a, "protocol", "nighthawk") != "nighthawk" else {}),
                lrc=a.lrc if a.backend == "ace" else None, lrr=a.lrr if a.backend == "ace" else None,
                exact_probs=a.exact_probs and a.backend == "exact")
    if getattr(a, "ace_torus", False) and a.backend == "ace":
        keys["torus"] = True                     # absent for the original non-torus runs
    if getattr(a, "ace_layout", "grid") != "grid" and a.backend == "ace":
        keys["layout"] = a.ace_layout             # absent for the original 8x8 runs
    if getattr(a, "ace_max_width", None) and a.backend == "ace":
        keys["max_width"] = a.ace_max_width       # absent unless --ace-max-width; lrc/lrr then unused
        if getattr(a, "ace_torus_search", "flat") != "any":
            keys["torus_search"] = a.ace_torus_search   # 'any' keeps the tag of the first runs
    if a.backend == "ace":
        for attr, kw, dflt in ACE_OPTS:                # only non-defaults: existing tags unchanged
            if bool(getattr(a, attr, dflt)) != dflt:
                keys[kw] = bool(getattr(a, attr, dflt))
    if getattr(a, "ace_transpile", False) and a.backend == "ace":
        keys["transpile"] = dict(engine=getattr(a, "transpile_engine", "exact") or "exact",
                                 width=getattr(a, "ace_width", None),
                                 exclude=list(parse_exclude(getattr(a, "ace_exclude", None))),
                                 seed=int(getattr(a, "transpile_seed", 0) or 0),
                                 trials=int(getattr(a, "transpile_trials", 0) or 0),
                                 calls=int(getattr(a, "transpile_calls", 0) or 0),
                                 **({"allow_cross": True} if getattr(a, "transpile_allow_cross", False) else {}),
                                 **({"caps": list(transpile_caps(a))} if transpile_caps(a) else {}))
    if getattr(a, "ace_tiling", False) and a.backend == "ace":
        keys["tiling"] = tiling_version(a)        # absent for device-geometry placement; 2 = the v2 tags
    if getattr(a, "mirror_estimator", "shots") != "shots" and a.backend == "ace":
        keys["mirror_estimator"] = a.mirror_estimator   # absent for sampled (shots) runs
    if (getattr(a, "register", "release") or "release") != "release":
        keys["register"] = a.register             # absent for the release's 61-qubit register
    if "vqe" in str(getattr(a, "families", "")).split(","):
        src = getattr(a, "vqe_opt_register", None) or "nighthawk2p16"
        keys["vqe"] = dict(J=float(a.vqe_j), h=float(a.vqe_h), angles=src,
                           **({"opt_n": int(a.vqe_opt_n)} if src == "self" else {}))
    return a.backend + "-" + hashlib.sha1(json.dumps(keys, sort_keys=True).encode()).hexdigest()[:8]


def parse_sizes(s):
    """'36', '27-36', '27,30,33-36' -> sorted unique sizes."""
    out = set()
    for part in str(s).split(","):
        lo, _, hi = part.strip().partition("-")
        out.update(range(int(lo), int(hi) + 1) if hi else [int(lo)])
    return sorted(out)


def mirror_czpc(lay, n, d):
    """CZ gates per cycle of the truncated pseudo-patched mirror (both halves)."""
    cyc = mirror_cycles(lay, d, 0, n=n)
    return sum(len(es) for _, es in cyc) / max(len(cyc), 1)


SHOT_TABLES = {"mirror": MIRROR_SHOTS, "patched": PATCHED_SHOTS}


def n_shots(a, family, d):
    if a.shots == "paper":
        table = SHOT_TABLES[family]
        if d not in table:
            raise SystemExit(f"--shots paper has no {family} budget for depth {d} "
                             f"(Appendix D lists {sorted(table)}); pass --shots N for other depths")
        return table[d]
    return int(a.shots)


def check_shot_budgets(a, jobs):
    """Fail before any worker starts, not on the first unbudgeted point mid-run."""
    if a.shots != "paper":
        return
    bad = sorted({(fam, d) for _, fam, _, d, _, _ in jobs if fam in SHOT_TABLES and d not in SHOT_TABLES[fam]})
    if bad:
        lines = "\n".join(f"  {fam} d={d}: Appendix D budgets exist for {sorted(SHOT_TABLES[fam])}" for fam, d in bad)
        raise SystemExit(f"--shots paper has no budget for:\n{lines}\npass --shots N, or use the listed depths")


def run_mirror(lay, eng, d, inst, a):
    """Survival over the ten released inputs, shots split evenly; twirled runs split
    each input's shots further over the randomisations (as SamplerV2 does)."""
    n = eng.n
    cyc = mirror_cycles(lay, d, inst, n=n)
    per_input = max(1, n_shots(a, "mirror", d) // len(lay.inputs))
    R = a.twirls if a.backend == "ace" else 0       # twirling is an identity for exact gates
    hits, shots, keep = [], [], {}
    if a.backend == "ace" and getattr(a, "mirror_estimator", "shots") == "prob":
        return run_mirror_prob(lay, eng, d, inst, a, n, cyc, R)
    runs = sum(1 for _ in lay.inputs for r in range(max(R, 1))
               if per_input // max(R, 1) + (1 if r < per_input % max(R, 1) else 0) > 0)
    prog = Progress(a, f"n{n} mirror d{d} i{inst}", runs, 2 * len(cyc))
    try:
        return _run_mirror_body(lay, eng, d, inst, a, n, cyc, per_input, R, hits, shots, keep, prog)
    finally:
        prog.close()


def run_mirror_prob(lay, eng, d, inst, a, n, cyc, R):
    """--mirror-estimator prob (ACE): every input x twirl circuit run contributes the
    probability that ACE's sampler returns the prepared string (AceEngine.prob_target),
    instead of measure_shots samples. No shot noise and no per-shot backend clone; the
    error bar is the spread over the runs. No bitstrings are kept."""
    R = max(R, 1)
    prog = Progress(a, f"n{n} mirror d{d} i{inst}", len(lay.inputs) * R, 2 * len(cyc))
    per_run, per_input, errors = [], [], []
    try:
        for s, string in enumerate(lay.inputs):
            bits = [int(c) for c in string[:n]]
            ps = []
            for r in range(R):
                rng = np.random.default_rng([BASE_SEED, d, inst, s, r]) if a.twirls else None
                eng.reset()
                for q, b in enumerate(bits):
                    if b:
                        eng.pauli("x", q)
                apply_forward(eng, cyc, rng, prog.tick)
                apply_inverse(eng, cyc, rng, prog.tick)
                try:
                    ps.append(eng.prob_target(bits, np.random.default_rng([BASE_SEED, d, inst, s, r, 7])))
                except ProbTargetError as e:
                    # Qrack's C++ side refused a forced outcome (fp32 edge case): count the run as 0,
                    # record the partial product as its upper bound, rebuild the backend, carry on
                    errors.append(float(e.args[0]))
                    ps.append(0.0)
                    print(f"# prob_target: Qrack exception in n{n} d{d} i{inst} input {s} twirl {r} "
                          f"(partial product {e.args[0]:.3g}); run counted as 0, backend rebuilt", flush=True)
                    eng.rebuild()
            per_run += ps
            per_input.append(float(np.mean(ps)))
    finally:
        prog.close()
    x = np.array(per_run)
    se = float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0
    out = dict(fidelity=float(x.mean()), se=se, survival=per_input, shots=0, runs=len(x), estimator="prob")
    if errors:   # runs counted as 0 after a Qrack exception; their partial products bound the bias above
        out.update(prob_errors=len(errors), prob_error_bound=float(sum(errors) / len(x)))
    return out


def _run_mirror_body(lay, eng, d, inst, a, n, cyc, per_input, R, hits, shots, keep, prog):
    for s, string in enumerate(lay.inputs):
        bits = [int(c) for c in string[:n]]
        target = sum(b << q for q, b in enumerate(bits))
        h = tot = 0
        pr = 0.0
        got, tw = [], []
        for r in range(max(R, 1)):
            rng = np.random.default_rng([BASE_SEED, d, inst, s, r]) if R else None
            k = per_input // max(R, 1) + (1 if r < per_input % max(R, 1) else 0)
            if k == 0:
                continue
            eng.reset()
            for q, b in enumerate(bits):
                if b:
                    eng.pauli("x", q)
            apply_forward(eng, cyc, rng, prog.tick)
            apply_inverse(eng, cyc, rng, prog.tick)
            if eng.prob_bits and a.exact_probs:
                pr += eng.prob_bits(bits) / max(R, 1)
            else:
                sh = eng.shots(k)
                got.append(sh)
                tw.append(np.full(k, r, dtype=np.uint16))
                h += int(np.count_nonzero(sh == np.uint64(target)))
                tot += k
        if eng.prob_bits and a.exact_probs:
            hits.append(pr)
            shots.append(0)
        else:
            hits.append(h)
            shots.append(tot)
            keep[f"input{s}"] = np.concatenate(got) if got else np.zeros(0, np.uint64)
            keep[f"input{s}_twirl"] = np.concatenate(tw) if tw else np.zeros(0, np.uint16)
    if eng.prob_bits and a.exact_probs:
        return dict(fidelity=float(np.mean(hits)), se=0.0, survival=hits, shots=0)
    p = sum(hits) / sum(shots)
    return dict(fidelity=p, se=math.sqrt(max(p * (1 - p), 1e-12) / sum(shots)),
                hits=hits, shots=int(sum(shots)), shots_per_input=shots, _shots=keep)


def run_patched(lay, eng, K, d, j, inst, a):
    part = lay.partitions[K][j]
    cyc = patched_cycles(lay, K, d, j, inst)
    probs = cached_patch_probs(a.cache, f"K{K}_d{d}_p{j}_i{inst}", cyc, part["patches"], a.cpu)
    eng.reset()
    apply_forward(eng, cyc)
    sh = eng.shots(n_shots(a, "patched", d))
    return dict(**patched_xeb(sh, probs, part["patches"]), _shots={"shots": sh})


FXEB_SHOTS = 4096


def out_probs_np(sim):
    """All 2^n probabilities straight into one numpy buffer of Qrack's real1 type
    (as in nn_qab.py): no Python float list, so 34 qubits needs 64 GiB, not ~10x that."""
    import ctypes
    from pyqrack.qrack_system import Qrack
    c_type, np_type = (ctypes.c_float, np.float32) if Qrack.fppow < 6 else (ctypes.c_double, np.float64)
    buf = np.empty(1 << sim.num_qubits(), dtype=np_type)
    Qrack.qrack_lib.OutProbs(sim.sid, buf.ctypes.data_as(ctypes.POINTER(c_type)))
    sim._throw_if_error()
    return buf


def fxeb_cycles(lay, d, inst, n):
    """The unpatched forward circuit on the first n logical qubits, release seed
    convention for full circuits (2025 + k*1000003, instance k)."""
    return build_cycles(lay, d, instance_seed(inst, "full"), inst, n=n)


def fxeb_gates(cyc, inst):
    """--protocol nnqab: each coupler's two-qubit gate per cycle; None for the CZ circuits."""
    if PROTOCOL["mode"] != "nnqab":
        return None
    seed = instance_seed(inst, "full")
    return [[nnqab_gate(seed, inst, c, a_, b_) for a_, b_ in es] for c, (_, es) in enumerate(cyc)]


def fxeb_shot_count(a, n):
    if a.shots != "paper":
        return int(a.shots)
    return 1 << min(13, n + 2) if PROTOCOL["mode"] == "nnqab" else FXEB_SHOTS


def fxeb_sample(eng, cyc, a, gates=None):
    eng.reset()
    if gates is None:
        apply_forward(eng, cyc)
    else:
        for (ang, es), gs in zip(cyc, gates):
            for q, t in enumerate(ang):
                eng.g1(q, gate(t))
            for (a_, b_), g in zip(es, gs):
                eng.g2(g, a_, b_)
    return eng.shots(fxeb_shot_count(a, eng.n))


def fxeb_reference(n, cyc, a, gates=None):
    """Exact 2^n output probabilities of the forward circuit (one float buffer; n=34:
    128 GiB state + 64 GiB probabilities), with their sum and collision sum."""
    from pyqrack import QrackSimulator
    ref = QrackSimulator(n, is_gpu=not a.cpu and a.ref_gpu)
    for c, (ang, es) in enumerate(cyc):
        for q, t in enumerate(ang):
            ref.mtrx(flat(gate(t)), q)
        for k, (a_, b_) in enumerate(es):
            if gates is None:
                ref.mcz([a_], b_)
            else:
                qrack_g2(ref, gates[c][k], a_, b_)
    p = out_probs_np(ref)
    del ref
    sum_p = float(p.sum(dtype=np.float64))
    sum_sq = float(np.einsum("i,i->", p, p, dtype=np.float64)) / (sum_p * sum_p)
    return p, sum_p, sum_sq


def fxeb_score(ref, sh):
    """(N sum_s q p - 1) / (N sum p^2 - 1) of samples sh against the reference."""
    p, sum_p, sum_sq = ref
    N = float(p.size)
    ps = p[sh.astype(np.int64)].astype(np.float64) / sum_p
    norm = N * sum_sq - 1.0                      # ideal XEB (collision ratio), ~1 if Porter-Thomas
    x = N * ps - 1.0
    return dict(fidelity=float(x.mean() / norm), se=float(x.std(ddof=1) / math.sqrt(len(x)) / norm),
                xeb_linear=float(x.mean()), ideal_xeb=norm, hog=float(np.mean(ps > math.log(2.0) / N)),
                shots=int(len(x)), _shots={"shots": sh})


def run_fxeb(lay, eng, d, inst, a):
    """Forward XEB of the backend's samples against the exact reference, as nn_qab.py
    does: (N sum_s q p - 1) / (N sum p^2 - 1), which is Eq. (1) with the whole register
    as one patch. No twirling: the samples come from one plain forward run, so a
    structured (coherent) approximation keeps whatever XEB it earns. The reference is
    an exact QrackSimulator of the same gates."""
    cyc = fxeb_cycles(lay, d, inst, eng.n)
    gates = fxeb_gates(cyc, inst)
    sh = fxeb_sample(eng, cyc, a, gates)
    return fxeb_score(fxeb_reference(eng.n, cyc, a, gates), sh)


def run_fxeb_group(lay, engs, d, inst, a):
    """Several engines' samples of the same circuit scored against ONE exact reference.
    Samples are drawn first, so only bitstrings are held while the reference is built."""
    n = engs[0].n
    cyc = fxeb_cycles(lay, d, inst, n)
    gates = fxeb_gates(cyc, inst)
    shots, ace_s = [], []
    for eng in engs:
        t0 = time.time()
        shots.append(fxeb_sample(eng, cyc, a, gates))
        ace_s.append(time.time() - t0)
    t0 = time.time()
    ref = fxeb_reference(n, cyc, a, gates)
    ref_s = time.time() - t0
    out = []
    for sh, s in zip(shots, ace_s):
        r = fxeb_score(ref, sh)
        r.update(ace_seconds=round(s, 2), ref_seconds=round(ref_s, 2), ref_shared=len(engs))
        out.append(r)
    return out


def run_hamming(lay, eng, d, lam, inst, a):
    """Dan Strano's mirror observable (qem/richardson_nn_hamming_weight.py): the forward circuit U of d
    cycles (fxeb_cycles; nn_qab.py's gates under --protocol nnqab), then U^dagger, folded to scale factor
    lam = 2k + 1 as M (M^dagger M)^k = (U U^dagger)^lam, from |0...0>, no twirling. h = mean fraction of
    ones, taken exactly as sum_q P(q = 1) / n (the shot-free value of his Hamming-weight estimator; ACE's
    measure_shots would clone the backend per shot). Score: the signal 1 - 2h (ideal 1)."""
    n = eng.n
    cyc = fxeb_cycles(lay, d, inst, n)
    gates = fxeb_gates(cyc, inst)
    # progress for --status: one "circuit run" per U U^dagger pass, ticked per cycle (2 d per pass);
    # ACE's later, more entangled cycles are slower, so the ETA runs optimistic early in a pass
    prog = Progress(a, f"n{n} hamming K{int(lam)} d{d} p0 i{inst}", int(lam), 2 * len(cyc))   # --status label
    try:
        eng.reset()
        for _ in range(int(lam)):
            if gates is None:
                apply_forward(eng, cyc, tick=prog.tick)
                apply_inverse(eng, cyc, tick=prog.tick)
                continue
            for (ang, es), gs in zip(cyc, gates):
                for q, t in enumerate(ang):
                    eng.g1(q, gate(t))
                for (a_, b_), g in zip(es, gs):
                    eng.g2(g, a_, b_)
                prog.tick()
            for (ang, es), gs in zip(reversed(cyc), reversed(gates)):
                for (a_, b_), g in zip(reversed(es), reversed(gs)):
                    eng.g2_inv(g, a_, b_)
                for q in reversed(range(len(ang))):
                    eng.g1(q, gate(ang[q]).conj().T)
                prog.tick()
    finally:
        prog.close()
    ps = [eng.prob1(q) for q in range(n)]
    bad = [q for q, x in enumerate(ps) if not math.isfinite(x) or x < -1e-6 or x > 1 + 1e-6]
    if bad:                                  # a broken state (seen on GPU and CPU): fail, never record
        raise RuntimeError(f"hamming: P(q = 1) not a probability on {len(bad)} of {n} qubits "
                           f"(first: q{bad[0]} = {ps[bad[0]]!r})")
    h = float(np.mean(ps))
    return dict(fidelity=1.0 - 2.0 * h, se=0.0, hamming_weight=float(n * h), h=h, fold=int(lam),
                layers=2 * int(d) * int(lam), estimator="prob")


# ================================================================== vqe: transverse-field Ising on the register
# H = -J sum_<uv> Z_u Z_v - h sum_q X_q over the register's couplers (its four colour matchings A-D).
# Ansatz (Hamiltonian variational, p layers): |+>^n, then per layer RZZ(gamma_l) on every coupler, colour by
# colour (A, B, C, D, as the RCS cycles), then RX(beta_l) on every qubit. RZZ(t) = exp(-i t/2 ZZ) =
# CX . RZ_target(t) . CX, RX(t) = exp(-i t/2 X). The angles are optimised once, exactly in numpy, on a small
# register (--vqe-opt-register, default nighthawk2p16) and used unchanged at every size: one circuit family,
# so the ladder measures how the simulator scales, not how an optimiser fares.
# Energy without shots: <X_q> = 1 - 2 P(q = 1) after H on every qubit; <Z_u Z_v> = 1 - 2 P(v = 1) after
# CX(u -> v) on the couplers of one colour (a matching: all its pairs at once). 1 + 4 circuit passes per
# energy. On ACE a seam coupler's readout CX is itself a seam gate.
# References, all in numpy (no Qrack): the same circuit's energy (n <= --vqe-exact-max), the ground energy E0
# by Lanczos (n <= --vqe-ed-max), and at any n Anderson's bound E0 >= -sum_q sqrt(h^2 + (J k_q / 2)^2)
# (k_q couplers at qubit q; H split into stars, each coupler shared by its two ends). No quantum state has
# an energy below E0, so an energy below E0 or below the bound means the patches do not form one state.
VQE_HAD = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


def vqe_colours(lay, n):
    """[(colour, [(u, v), ...])] in the schedule's order, couplers among the first n logical qubits."""
    return [(c, [tuple(e) for e in lay.matchings[c] if e[0] < n and e[1] < n]) for c in lay.schedule]


def vqe_energy(eng, n, cols, gam, bet, J, h, tick=None):
    """<H> of the HVA state on an engine (exact or ACE) from 1 + (colours with couplers) passes."""
    def prepare():
        eng.reset()
        for q in range(n):
            eng.g1(q, VQE_HAD)
        for g, b in zip(gam, bet):
            Rz = rz(g)
            for _, es in cols:
                for u, v in es:
                    eng.cx(u, v)
                    eng.g1(v, Rz)
                    eng.cx(u, v)
            Rx = rx(b)
            for q in range(n):
                eng.g1(q, Rx)
            if tick:
                tick()

    def probs(qs):
        ps = [eng.prob1(q) for q in qs]
        bad = [q for q, x in zip(qs, ps) if not math.isfinite(x) or x < -1e-6 or x > 1 + 1e-6]
        if bad:                              # a broken state: fail the point, never record it
            raise RuntimeError(f"vqe: P(q = 1) not a probability on {len(bad)} of {len(qs)} qubits "
                               f"(first: q{bad[0]})")
        return ps

    prepare()
    for q in range(n):
        eng.g1(q, VQE_HAD)
    xs = [1.0 - 2.0 * p for p in probs(list(range(n)))]
    zz = []
    for _, es in cols:
        if not es:
            continue
        prepare()
        for u, v in es:
            eng.cx(u, v)
        zz += [1.0 - 2.0 * p for p in probs([v for _, v in es])]
    sx, szz = float(sum(xs)), float(sum(zz))
    return dict(energy=-J * szz - h * sx, sum_zz=szz, sum_x=sx, min_zz=float(min(zz)) if zz else None,
                min_x=float(min(xs)))


class TfimNumpy:
    """The same H and circuit as a dense numpy state vector (no Qrack): logical qubit q = bit q of the
    index. Memory: 16 B x 2^n for the state, 2 B x 2^n for the coupling diagonal."""

    def __init__(self, n, cols, J, h):
        self.n, self.J, self.h = n, float(J), float(h)
        N = 1 << n
        idx = np.arange(N, dtype=np.uint64 if n > 31 else np.uint32)
        D = np.zeros(N, dtype=np.int16)
        for _, es in cols:
            for u, v in es:
                D += (1 - 2 * (((idx >> u) ^ (idx >> v)) & 1)).astype(np.int16)
        self.D = D                                   # sum over couplers of z_u z_v, per basis state

    def _rx_all(self, psi, b):
        c, s = math.cos(b / 2), math.sin(b / 2)
        for q in range(self.n):
            v = psi.reshape(-1, 2, 1 << q)
            a0 = v[:, 0, :].copy()
            v[:, 0, :] *= c
            v[:, 0, :] += -1j * s * v[:, 1, :]
            v[:, 1, :] *= c
            v[:, 1, :] += -1j * s * a0

    def state(self, gam, bet):
        N = 1 << self.n
        psi = np.full(N, 1.0 / math.sqrt(N), dtype=np.complex128)
        for g, b in zip(gam, bet):
            psi *= np.exp(-0.5j * float(g) * self.D)
            self._rx_all(psi, b)
        return psi

    def energy(self, psi):
        szz = float(np.dot(np.abs(psi) ** 2, self.D))
        sx = 0.0
        for q in range(self.n):
            v = psi.reshape(-1, 2, 1 << q)
            sx += 2.0 * float(np.real(np.vdot(v[:, 0, :], v[:, 1, :])))
        return -self.J * szz - self.h * sx

    def ground(self, tol=1e-11, maxit=600):
        """E0 by Lanczos (three vectors; the lowest Ritz value converges without reorthogonalisation)."""
        from scipy.linalg import eigh_tridiagonal
        D = self.D.astype(np.float64)

        def H(v):
            w = -self.J * D * v
            for q in range(self.n):
                w -= self.h * v.reshape(-1, 2, 1 << q)[:, ::-1, :].reshape(-1)
            return w
        v = np.random.default_rng(1).standard_normal(1 << self.n)
        v /= np.linalg.norm(v)
        v_prev, beta, al, be, e_old = np.zeros_like(v), 0.0, [], [], math.inf
        for k in range(maxit):
            w = H(v) - beta * v_prev
            a_ = float(np.dot(w, v))
            w -= a_ * v
            al.append(a_)
            e = float(eigh_tridiagonal(np.array(al), np.array(be), eigvals_only=True,
                                       select="i", select_range=(0, 0))[0]) if be else a_
            if k > 8 and abs(e - e_old) < tol * max(1.0, abs(e)):
                return e
            e_old = e
            beta = float(np.linalg.norm(w))
            if beta < 1e-14:
                return e
            be.append(beta)
            v_prev, v = v, w / beta
        return e_old


def tfim_anderson_bound(n, cols, J, h):
    k = [0] * n
    for _, es in cols:
        for u, v in es:
            k[u] += 1
            k[v] += 1
    return -sum(math.sqrt(h * h + (J * kq / 2.0) ** 2) for kq in k)


def vqe_cache_path(a):
    out = getattr(a, "out", None)
    return Path(out).with_name(Path(out).stem + "_vqe.json") if out else None


def _vqe_cache(path, update=None):
    """Read (update=None) or merge-and-write the vqe cache next to --out, under a file lock."""
    if path is None:
        return {}
    import fcntl
    lock = path.with_name(path.name + ".lock")
    with open(lock, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            cur = json.loads(path.read_text()) if path.exists() else {}
            if update:
                for k, v in update.items():
                    cur.setdefault(k, {}).update(v)
                tmp = path.with_name(path.name + ".tmp")
                tmp.write_text(json.dumps(cur, indent=1))
                os.replace(tmp, path)
            return cur
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def vqe_optimise(cols, n, J, h, pmax, log=print):
    """HVA angles for p = 1 .. pmax on an n-qubit numpy reference: p = 1 from a grid, then each p from
    the previous one interpolated (Zhou et al., PRX 10, 021067) and refined with BFGS."""
    from scipy.optimize import minimize
    ref = TfimNumpy(n, cols, J, h)

    def f(x):
        p = len(x) // 2
        return ref.energy(ref.state(x[:p], x[p:]))
    out, prev = {}, None
    for p in range(1, pmax + 1):
        if prev is None:
            grid = [(f(np.array([g, b])), g, b) for g in np.linspace(0.05, math.pi - 0.05, 15)
                    for b in np.linspace(0.05, math.pi - 0.05, 15)]
            _, g0, b0 = min(grid)
            starts = [np.array([g0, b0])]
        else:
            q = p - 1

            def interp(x):
                x = np.concatenate([[0.0], x, [0.0]])
                return np.array([(i / q) * x[i] + ((q - i) / q) * x[i + 1] for i in range(p)])
            gp, bp = prev[:q], prev[q:]
            starts = [np.concatenate([interp(gp), interp(bp)]),
                      np.concatenate([gp, [gp[-1] * 0.5], bp, [bp[-1] * 0.5]])]
        best = min((minimize(f, x0, method="BFGS", options=dict(gtol=1e-7, maxiter=2000)) for x0 in starts),
                   key=lambda r: r.fun)
        prev = best.x
        out[str(p)] = dict(gamma=[float(x) for x in best.x[:p]], beta=[float(x) for x in best.x[p:]],
                           energy=float(best.fun))
        log(f"# vqe angles, p = {p}: E = {best.fun:.6f} on {n} qubits", flush=True)
    return out, ref


def vqe_angles(lay, a, p):
    """(gamma, beta, source) for p layers: optimised once on --vqe-opt-register (or the run's own first
    --vqe-opt-n qubits with 'self') and cached in <out>_vqe.json, so every size uses the same angles."""
    J, h = float(a.vqe_j), float(a.vqe_h)
    src = getattr(a, "vqe_opt_register", None) or "nighthawk2p16"
    if src == "self":
        slay, sn = lay, min(int(a.vqe_opt_n), lay.n)
    else:
        slay = Layout(a.repo) if src == "release" else RegisterLayout(a.repo, src)
        sn = slay.n
    key = f"angles|{layout_id(slay)}|n{sn}|J{J:g}|h{h:g}"
    path = vqe_cache_path(a)
    cur = _vqe_cache(path).get(key, {})
    if str(p) not in cur:
        pmax = max([p] + [int(x) for x in (getattr(a, "depths", None) or [])])
        print(f"# vqe: optimising HVA angles p = 1..{pmax} on {layout_id(slay)} ({sn} qubits, numpy), "
              f"J {J:g} h {h:g}", flush=True)
        ang, ref = vqe_optimise(vqe_colours(slay, sn), sn, J, h, pmax)
        if sn <= 26:
            ang["e0"] = ref.ground()
        _vqe_cache(path, {key: ang})
        cur = ang
    d = cur[str(p)]
    return d["gamma"], d["beta"], dict(register=layout_id(slay), n=sn, energy=d["energy"], e0=cur.get("e0"))


def run_vqe(lay, eng, p, a):
    """One VQE point: the energy of the p-layer HVA state on this engine, with its references."""
    n = eng.n
    J, h = float(a.vqe_j), float(a.vqe_h)
    cols = vqe_colours(lay, n)
    gam, bet, src = vqe_angles(lay, a, p)
    passes = 1 + sum(1 for _, es in cols if es)
    prog = Progress(a, f"n{n} vqe K0 d{p} p0 i0", passes, max(1, p))       # --status label
    try:
        res = vqe_energy(eng, n, cols, gam, bet, J, h, tick=prog.tick)
    finally:
        prog.close()
    E = res["energy"]
    out = dict(res, energy_per_site=E / n, vqe_j=J, vqe_h=h, layers=int(p), gamma=gam, beta=bet,
               angles_from=src, couplers=sum(len(es) for _, es in cols),
               anderson_bound=tfim_anderson_bound(n, cols, J, h), estimator="prob")
    if n <= int(a.vqe_exact_max):
        t0 = time.time()
        ref = TfimNumpy(n, cols, J, h)
        out["exact_energy"] = ref.energy(ref.state(gam, bet))
        out["exact_seconds"] = round(time.time() - t0, 2)
        if n <= int(a.vqe_ed_max):
            key = f"e0|{layout_id(lay)}|n{n}|J{J:g}|h{h:g}"
            path = vqe_cache_path(a)
            e0 = _vqe_cache(path).get(key, {}).get("e0")
            if e0 is None:
                t0 = time.time()
                e0 = ref.ground()
                _vqe_cache(path, {key: {"e0": e0, "seconds": round(time.time() - t0, 2)}})
            out["e0"] = e0
        del ref
    tol = 1e-9 * max(1.0, abs(E))
    out["below_bound"] = bool(E < out["anderson_bound"] - tol)
    out["below_e0"] = bool("e0" in out and E < out["e0"] - tol)
    out["fidelity"] = E / n          # the framework's generic score: energy per site (lower is better)
    out["se"] = 0.0
    return out


def vqe_table(vq, backend="ace"):
    """Summary of vqe records: energy per site of the backend against the same circuit in numpy, the ground
    energy and Anderson's bound (all per site), and the two variational checks."""
    r0 = vq[0]
    print(f"\n== vqe: transverse-field Ising, J {r0['vqe_j']:g}, h {r0['vqe_h']:g}; HVA angles from "
          f"{r0['angles_from']['register']} ({r0['angles_from']['n']} qubits); energies per site ==")
    print("(correlation energy: what the circuit gains below the product state |+>^n, E_prod = -h n; "
          "backend/circuit = the share of the exact circuit's gain the backend keeps)")
    print(f"n    p   E_{backend:<8}  E_circuit  backend-circuit  E0        bound      "
          f"corr. backend/circuit  corr. circuit/E0  checks")
    for r in sorted(vq, key=lambda r: (r["n"], r["layers"])):
        n = r["n"]
        ec, e0, ep = r.get("exact_energy"), r.get("e0"), -float(r["vqe_h"]) * n
        flags = ("BELOW E0 " if r.get("below_e0") else "") + ("BELOW BOUND" if r.get("below_bound") else "")
        cb = (ep - r["energy"]) / (ep - ec) if ec is not None and ep != ec else None
        c0 = (ep - ec) / (ep - e0) if ec is not None and e0 is not None and ep != e0 else None
        print(f"{n:<4} {r['layers']:>2}  {r['energy'] / n:>10.5f}  "
              + (f"{ec / n:>9.5f}  {(r['energy'] - ec) / n:>+15.5f}" if ec is not None else f"{'-':>9}  {'-':>15}")
              + (f"  {e0 / n:>8.5f}" if e0 is not None else f"  {'-':>8}")
              + f"  {r['anderson_bound'] / n:>9.5f}"
              + (f"  {cb:>21.3f}" if cb is not None else f"  {'-':>21}")
              + (f"  {c0:>16.3f}" if c0 is not None else f"  {'-':>16}")
              + f"  {flags or 'ok'}")


# ================================================================== shor9: Shor's code on ACE (9 qubits, and larger)
# By (Anthropic) Claude and Dan Strano. Shor's code concatenates an m-qubit phase-flip code (outer) with m
# m-qubit bit-flip codes (inner blocks): [[m^2, 1, m]]; m = 3 is Shor-9, --size 4 Shor-16, --size 6 Shor-36.
#     |0_L> = (|0..0> + |1..1>)^m / 2^(m/2),    |1_L> = (|0..0> - |1..1>)^m / 2^(m/2)
# Encoder: input on leader a; CXs copy it to the other leaders along a tree of couplings (outer); H on every
# leader; each leader copies its bit to its block along a tree of couplings (inner). For m = 3 both trees are
# stars: CX a->b, a->c; H on a, b, c; each leader CX to its two members (the original script).
# Readout with decoding:
#   logical X (|+_L>, |-_L>): all in Z; a block's bit is the majority of its m bits; logical X = parity of
#     the block bits.
#   logical Z (|0_L>, |1_L>): H on all, then Z; a block's sign is the parity of its m outcomes; logical Z =
#     majority of the m block signs.
#   For even m a majority can tie: a tie is a coin flip (counted 0.5), so 0.5 stays 'no information'.
#   raw: the readout without correction. Odd m: the parity of all m^2 outcomes (the original script's). Even
#     m, where that parity carries no information: Z basis the parity of the leaders' bits, X basis the first
#     block's sign.
# All four logical states run, so a sampler biased toward one answer cannot score; 1.0 perfect, 0.5 nothing.
# Register: an m x m ACE grid of two patches joined by one seam column of m qubits (replicated in both patches
# and the crossbar; seam-to-seam couplings exact): m = 3 long_range_columns = 1 as the original; larger m the
# first (lrc, lrr) that gives that split (16: 2, 3; 25: 2, 4; 36: 3, 5; 49: 3, 6). The seam set and coupling
# map are read from the backend and every CX is checked against the map.
#   seam_leaders: the block leaders are the seam qubits (outer code seam-to-seam, exact); every inner block is
#                 a seam qubit plus bulk qubits of both patches, so each straddles the seam.
#   patch_blocks: one block is the seam column, the others are cut from the patches' bulk (inner blocks
#                 exact); the outer layer crosses the seam. Odd m only: for even m the m (m - 1) / 2 bulk
#                 qubits of a patch do not split into whole blocks.
SHOR9_WIDTH = 9
SHOR9_ACE = dict(long_range_columns=1, is_torus=False)
SHOR9_STATES = ("0", "1", "+", "-")
_SHOR_GEOM = {3: dict(SHOR9_ACE)}


def _shor9_replicas(sim, q):
    for attr in ("qubits", "_qubits"):
        if hasattr(sim, attr):
            return getattr(sim, attr)[q]
    return sim._unpack(q)


def _ace_split(sim, n):
    seams = [q for q in range(n) if len(_shor9_replicas(sim, q)) > 1]
    patches = {}
    for q in range(n):
        if len(_shor9_replicas(sim, q)) == 1:
            patches.setdefault(_shor9_replicas(sim, q)[0][0], []).append(q)
    return seams, patches


def shor_ace_geometry(m):
    """QrackAceBackend keyword arguments for an m x m register of two equal patches joined by one seam column."""
    if m in _SHOR_GEOM:
        return dict(_SHOR_GEOM[m])
    from pyqrack import QrackAceBackend
    for lrc in range(1, m + 2):
        for lrr in range(1, m + 4):
            try:
                s = ace_backend(m * m, long_range_columns=lrc, long_range_rows=lrr, is_torus=False, is_gpu=False)
            except Exception:
                continue
            seams, patches = _ace_split(s, m * m)
            ok = (s.get_row_length() == m and len(seams) == m and len({q % m for q in seams}) == 1
                  and len(patches) == 2 and all(len(v) == m * (m - 1) // 2 for v in patches.values()))
            del s
            if ok:
                _SHOR_GEOM[m] = dict(long_range_columns=lrc, long_range_rows=lrr, is_torus=False)
                return dict(_SHOR_GEOM[m])
    raise SystemExit(f"shor9: no ACE geometry of {m * m} qubits gives two patches joined by one seam column")


def shor_star(blocks):
    """The original encoder's trees (m = 3): leader a to the other leaders, each leader to its members."""
    a = blocks[0][0]
    return dict(blocks=blocks, outer=[(a, b[0]) for b in blocks[1:]],
                inner=[(b[0], x) for b in blocks for x in b[1:]])


def shor_bfs_unit(blocks, cpl):
    """Encoder trees by breadth-first search over the couplings cpl (a set of sorted pairs): outer over the
    leaders from blocks[0][0], inner inside each block from its leader."""
    def tree(root, nodes):
        nodes, seen, out, frontier = set(nodes), {root}, [], [root]
        while frontier:
            nxt = []
            for u in frontier:
                for v in sorted(nodes - seen):
                    if tuple(sorted((u, v))) in cpl:
                        seen.add(v)
                        out.append((u, v))
                        nxt.append(v)
            frontier = nxt
        if seen != nodes:
            raise RuntimeError(f"shor9: qubits {sorted(nodes - seen)} not reachable from {root} through couplings")
        return out
    return dict(blocks=blocks, outer=tree(blocks[0][0], [b[0] for b in blocks]),
                inner=[e for b in blocks for e in tree(b[0], b)])


def shor_unit(u):
    return u if isinstance(u, dict) else shor_star(u)


def shor9_encoder(u):
    u = shor_unit(u)
    return ([("cx", p, c) for p, c in u["outer"]] + [("h", b[0]) for b in u["blocks"]]
            + [("cx", p, c) for p, c in u["inner"]])


def shor9_layouts(ace, m=3):
    n = m * m
    seams, patches = _ace_split(ace, n)
    if len(seams) != m or len(patches) != 2:
        raise RuntimeError(f"shor9: unexpected ACE layout: seams {seams}, patches {patches}")
    p0, p1 = [sorted(v) for _, v in sorted(patches.items())]
    if len(p0) != m * (m - 1) // 2 or len(p1) != m * (m - 1) // 2:
        raise RuntimeError(f"shor9: unexpected ACE layout: seams {seams}, patches {p0} {p1}")
    if m == 3:
        return {"seam_leaders": shor_star([[seams[i], p0[i], p1[i]] for i in range(3)]),
                "patch_blocks": shor_star([[seams[1], seams[0], seams[2]], [p0[0], p0[1], p0[2]], [p1[0], p1[1], p1[2]]])}
    cpl = {tuple(sorted(p)) for p in ace.get_logical_coupling_map()}
    i0 = i1 = 0
    sl = []
    for i in range(m):                                     # each block: one seam qubit + (m - 1) bulk, both patches
        k0 = (m - 1 + (i % 2)) // 2 if (m - 1) % 2 else (m - 1) // 2
        sl.append([seams[i]] + p0[i0:i0 + k0] + p1[i1:i1 + (m - 1 - k0)])
        i0, i1 = i0 + k0, i1 + m - 1 - k0
    out = {"seam_leaders": shor_bfs_unit(sl, cpl)}
    if m % 2:            # each patch holds (m - 1) / 2 whole blocks; for even m a block would have to span both
        mid = m // 2
        pb = [[seams[mid]] + seams[:mid] + seams[mid + 1:]] + [pp[j * m:(j + 1) * m] for pp in (p0, p1)
                                                                 for j in range((m - 1) // 2)]
        out["patch_blocks"] = shor_bfs_unit(pb, cpl)
    return out


def shor9_check_coupling(ace, ops):
    cm = {tuple(p) for p in ace.get_logical_coupling_map()}
    bad = [(c, t) for g, c, *rest in ops if g == "cx" for t in rest if (c, t) not in cm and (t, c) not in cm]
    if bad:
        raise RuntimeError(f"shor9: encoder uses couplings outside the logical coupling map: {bad}")


def shor9_prepare(sim, u, state, mapq=None):
    """Logical state on one unit, encoder, and the basis change of its readout (H on all for Z states)."""
    u = shor_unit(u)
    mq = (lambda q: q) if mapq is None else (lambda q: mapq[q])
    a = u["blocks"][0][0]
    if state in ("1", "-"):
        sim.x(mq(a))
    if state in ("+", "-"):
        sim.h(mq(a))
    for g, *qs in shor9_encoder(u):
        if g == "h":
            sim.h(mq(qs[0]))
        else:
            sim.mcx([mq(qs[0])], mq(qs[1]))
    if state in ("0", "1"):                     # logical Z is read in the X basis
        for b in u["blocks"]:
            for q in b:
                sim.h(mq(q))


def shor9_decode(x, blocks, state):
    """(raw ok, decoded ok) for one shot x (int over the qubits the blocks name) of one unit. A tied majority
    (even m) is a coin flip: decoded ok = 0.5, so 0.5 stays 'no logical information'."""
    m = len(blocks)
    bits = [[(x >> q) & 1 for q in b] for b in blocks]
    if state in ("+", "-"):
        expect = 0 if state == "+" else 1
        sums = [sum(b) for b in bits]
        ties = [2 * s_ == len(b) for s_, b in zip(sums, blocks)]
        if any(ties):            # a tied block bit: the logical parity is a coin flip
            dec_ok = 0.5
        else:
            dec_ok = float((sum(1 for s_, b in zip(sums, blocks) if 2 * s_ > len(b)) & 1) == expect)
        raw = (sum(sums) & 1) if m % 2 else (sum(b[0] for b in bits) & 1)
    else:
        expect = 0 if state == "0" else 1
        signs = [sum(b) & 1 for b in bits]
        dec_ok = 0.5 if 2 * sum(signs) == m else float((1 if 2 * sum(signs) > m else 0) == expect)
        raw = (sum(sum(b) for b in bits) & 1) if m % 2 else signs[0]
    return raw == expect, dec_ok


def shots_wide(sim, qubits, shots):
    """measure_shots that keeps every bit: QrackSimulator's measure_shots and m_all return bits 0-31 only
    (PyQrack 2.28.1), so above 32 qubits each shot is a clone measured qubit by qubit. QrackAceBackend composes
    its shots from the patches and returns all bits."""
    from pyqrack import QrackSimulator
    if len(qubits) <= 32 or not isinstance(sim, QrackSimulator):
        return sim.measure_shots(list(qubits), shots)
    out = []
    for _ in range(shots):
        c = sim.clone()
        out.append(sum(int(c.m(q)) << i for i, q in enumerate(qubits)))
        del c
    return out


def shor9_run(sim, u, state, shots, hist=None):
    """(raw, decoded) success rate of one logical state of one unit (qubits 0..m^2-1) on one simulator."""
    u = shor_unit(u)
    shor9_prepare(sim, u, state)
    nq = sum(len(b) for b in u["blocks"])
    counts = collections.Counter(shots_wide(sim, list(range(nq)), shots))
    if hist is not None:
        hist.update(counts)
    raw_ok = dec_ok = 0
    for k, v in counts.items():
        r, d = shor9_decode(k, u["blocks"], state)
        raw_ok += v * r
        dec_ok += v * d
    return raw_ok / shots, dec_ok / shots


# --jobs: the pieces of a shor9 run are independent, so they run in a pool of processes, one core each
# (Qrack threads only states far larger than a Shor unit, and the shot loop is Python):
#   ACE      one task per (layout or rotation, logical state, fresh backend); a backend's shots stay in one
#            task, because backends differ from one another by more than shot noise
#   exact    the reference state is the same in every simulator, so its shots split into chunks over fresh
#            simulators: the same distribution, sampled in parallel
# Workers are started with 'spawn' (no fork of a process holding Qrack or OpenCL state) and inherit the
# environment: QRACK_OCL_SYNC, the OpenCL isolation of a clvk die, OCL_ICD_VENDORS.

def shor_pool(jobs, ntasks):
    """(pool, workers): a spawn process pool of min(jobs, ntasks) workers (jobs 0 = every core), or (None, 1)."""
    n = (os.cpu_count() or 1) if int(jobs) <= 0 else int(jobs)
    n = max(1, min(n, int(ntasks)))
    if n == 1:
        return None, 1
    import concurrent.futures as cf
    import multiprocessing as mp
    return cf.ProcessPoolExecutor(n, mp_context=mp.get_context("spawn")), n


def shor_pool_map(pool, calls):
    """[f(*args) for f, args in calls], in order: in the pool, or in this process without one."""
    if pool is None:
        return [f(*args) for f, args in calls]
    futs = [pool.submit(f, *args) for f, args in calls]
    return [f_.result() for f_ in futs]


def shor_chunks(shots, k):
    """shots split into at most k chunks of at least 32 (the exact reference's parallel sampling)."""
    k = max(1, min(int(k), int(shots) // 32 or 1))
    return [shots // k + (1 if i < shots % k else 0) for i in range(k)]


def _shor9_ace_task(n, geo, gpu, u, s, shots, brep=False):
    h = collections.Counter()
    r = shor9_run(ace_backend(n, is_gpu=gpu, is_boundary_repetition_code=brep, **geo), u, s, shots, h)
    return r, h


def _shor9_exact_task(n, gpu, u, s, shots):
    from pyqrack import QrackSimulator
    return shor9_run(QrackSimulator(n, is_gpu=gpu), u, s, shots), shots


def shor9_bench(shots, gpu=False, exact=True, repeats=1, m=3, jobs=1, brep=False):
    """{layout: {"unit", state: (raw, decoded), "runs", "hist"}} for both ACE layouts, each state on `repeats`
    fresh backends (mean over them), plus the exact reference (QrackSimulator, seam_leaders), where every
    entry must be 1.0. jobs > 1 (0 = every core): the backends, and chunks of the exact shots, in a process
    pool. brep: ACE's is_boundary_repetition_code. Returns (results, geo, workers)."""
    geo, n = shor_ace_geometry(m), m * m
    probe = ace_backend(n, is_gpu=gpu, **geo)
    lays = shor9_layouts(probe, m)
    for u in lays.values():
        shor9_check_coupling(probe, shor9_encoder(u))
    del probe
    R = max(1, int(repeats))
    keys = [(name, s, r) for name in lays for s in SHOR9_STATES for r in range(R)]
    pool, workers = shor_pool(jobs, len(keys) + (len(SHOR9_STATES) if exact else 0))
    k_ex = max(1, -(-workers // len(SHOR9_STATES)))
    ex_keys = [(s, c) for s in SHOR9_STATES for c in shor_chunks(shots, k_ex)] if exact else []
    try:
        out = shor_pool_map(pool, [(_shor9_ace_task, (n, geo, gpu, lays[name], s, shots, brep)) for name, s, _ in keys]
                            + [(_shor9_exact_task, (n, gpu, lays["seam_leaders"], s, c)) for s, c in ex_keys])
    finally:
        if pool is not None:
            pool.shutdown()
    got = dict(zip(keys, out[:len(keys)]))
    results = {}
    for name, u in lays.items():
        row = {"unit": u, "blocks": u["blocks"], "runs": {}, "hist": {}}
        for s in SHOR9_STATES:
            h = collections.Counter()
            runs = []
            for r in range(R):
                rr, hh = got[(name, s, r)]
                runs.append(tuple(rr))
                h.update(hh)
            row["runs"][s] = runs
            row["hist"][s] = h
            row[s] = (float(np.mean([r[0] for r in runs])), float(np.mean([r[1] for r in runs])))
        results[name] = row
    if exact:
        u = lays["seam_leaders"]
        ref = {"unit": u, "blocks": u["blocks"]}
        for s in SHOR9_STATES:
            parts = [o for (s_, _), o in zip(ex_keys, out[len(keys):]) if s_ == s]
            tot = sum(c for _, c in parts)
            ref[s] = (sum(r[0] * c for r, c in parts) / tot, sum(r[1] * c for r, c in parts) / tot)
        results["exact_reference"] = ref
    return results, geo, workers


def shor9_exact_dist(u, state):
    """Exact outcome distribution of one unit on qubits 0..m^2-1 from a numpy state vector (no Qrack); used to
    check shor_possible (selftest [13]) for m = 3 and 4."""
    u = shor_unit(u)
    nq = sum(len(b) for b in u["blocks"])
    psi = np.zeros(1 << nq, dtype=complex)
    psi[0] = 1.0
    idx = np.arange(1 << nq)

    class V:
        def h(self, q):
            v = psi.reshape(-1, 2, 1 << q)
            a0, a1 = v[:, 0, :].copy(), v[:, 1, :].copy()
            v[:, 0, :], v[:, 1, :] = (a0 + a1) / math.sqrt(2), (a0 - a1) / math.sqrt(2)

        def x(self, q):
            v = psi.reshape(-1, 2, 1 << q)
            v[:, [0, 1], :] = v[:, [1, 0], :]

        def mcx(self, cs, t):
            sel = idx[(idx >> cs[0]) & 1 == 1]
            psi[sel] = psi[sel ^ (1 << t)].copy()
    shor9_prepare(V(), u, state)
    p = np.abs(psi) ** 2
    p[p < 1e-12] = 0.0
    return p / p.sum()


def shor_possible(x, blocks, state):
    """Whether the exact state can give outcome x: Z basis (|+-_L>) every block all-equal with the parity of
    the block values the logical one; X basis (|0_L>, |1_L>) every block's parity the logical sign."""
    bits = [[(x >> q) & 1 for q in b] for b in blocks]
    if state in ("+", "-"):
        return all(len(set(b)) == 1 for b in bits) and (sum(b[0] for b in bits) & 1) == (0 if state == "+" else 1)
    return all((sum(b) & 1) == (0 if state == "0" else 1) for b in bits)


def shor_support(blocks, state):
    """Number of outcomes the exact state gives, all equally likely: 2^(m - 1) (Z basis), 2^(sum(len - 1)) (X)."""
    m = len(blocks)
    return 2 ** (m - 1) if state in ("+", "-") else 2 ** sum(len(b) - 1 for b in blocks)


def shor9_compare(hist, blocks, state, top=3, draws=40, seed=9):
    """ACE's outcome histogram of one unit against the exact distribution (uniform over shor_possible's
    outcomes): total variation distance, the same distance for exact samples of the same size (the floor),
    the share of impossible outcomes and the most frequent of them, with whether the decoder recovers them."""
    K = shor_support(blocks, state)
    p = 1.0 / K
    N = sum(hist.values())
    ok = {int(k): v for k, v in hist.items() if shor_possible(int(k), blocks, state)}
    bad = sorted(((v, int(k)) for k, v in hist.items() if int(k) not in ok), reverse=True)
    tvd = 0.5 * (sum(abs(v / N - p) for v in ok.values()) + (K - len(ok)) * p + sum(v for v, _ in bad) / N)
    rng = np.random.default_rng(seed)
    fl = []
    for _ in range(draws):
        c = np.unique(rng.integers(0, K, size=N), return_counts=True)[1]
        fl.append(0.5 * (float(np.abs(c / N - p).sum()) + (K - len(c)) * p))
    fmt = lambda k: " ".join("".join(str((k >> q_) & 1) for q_ in b) for b in blocks)
    return dict(tvd=tvd, tvd_floor=float(np.mean(fl)), impossible=sum(v for v, _ in bad) / N, support=K, shots=N,
                top_wrong=[dict(pattern=fmt(k), share=v / N, decodes=float(shor9_decode(k, blocks, state)[1]))
                           for v, k in bad[:top]])


def shor9_print_dist(label, cmp_by_state):
    """One line per logical state: distance to exact (floor), impossible share, top wrong patterns."""
    for s in SHOR9_STATES:
        c = cmp_by_state.get(s)
        if c is None:
            continue
        tops = ", ".join(f"{t['pattern']} {100 * t['share']:.1f}%"
                         + ("" if t["decodes"] == 1 else " x" if t["decodes"] == 0 else " tie") for t in c["top_wrong"])
        print(f"  {label:<14} |{s}_L>  TVD {c['tvd']:.3f} (floor {c['tvd_floor']:.3f})  impossible "
              f"{100 * c['impossible']:5.1f}%  " + (tops if tops else "-"))


_RUN_PARSER = {}


def shor9_tiles(lay, max_tiles=None, m=3):
    """Disjoint units of Shor's [[m^2, 1, m]] code on the register's own couplers: m x m squares of lattice
    sites inside the register, leaders on the middle column (blocks = rows) or the middle row (blocks =
    columns); the outer tree runs along the leaders' line from the middle outwards, each inner tree along its
    block from the leader outwards, so every CX is a nearest-neighbour coupler. Most units first (CP-SAT;
    greedy without OR-Tools). Returns [(unit, (r0, c0), orientation)]."""
    C = 10
    cell = {(p // C, p % C): q for q, p in enumerate(lay.l2p)}
    L = (m - 1) // 2
    order = [L] + [i for k in range(1, m) for i in (L - k, L + k) if 0 <= i < m]      # middle, then outwards
    chain = lambda: [(i, i - 1) for i in range(L, 0, -1)] + [(i, i + 1) for i in range(L, m - 1)]

    def unit(at):                                   # at(i, j): qubit of block i, position j (leader at j = L)
        blocks = [[at(i, j) for j in order] for i in order]
        outer = [(at(i, L), at(k, L)) for i, k in chain()]
        inner = [(at(i, j), at(i, k)) for i in order for j, k in chain()]
        return dict(blocks=blocks, outer=outer, inner=inner)

    cand = []
    for (r0, c0) in sorted(cell):
        sq = [(r0 + i, c0 + j) for i in range(m) for j in range(m)]
        if not all(x in cell for x in sq):
            continue
        cand += [(unit(lambda i, j: cell[(r0 + i, c0 + j)]), (r0, c0), "rows"),
                 (unit(lambda i, j: cell[(r0 + j, c0 + i)]), (r0, c0), "cols")]
    sites = lambda t: {x for b in t[0]["blocks"] for x in b}
    try:
        from ortools.sat.python import cp_model
        md = cp_model.CpModel()
        xs = [md.NewBoolVar("") for _ in cand]
        for q in {x for t in cand for x in sites(t)}:
            md.AddAtMostOne(x for x, t in zip(xs, cand) if q in sites(t))
        for i in range(0, len(cand), 2):                    # one orientation per square
            md.AddAtMostOne([xs[i], xs[i + 1]])
        if max_tiles:
            md.Add(sum(xs) <= int(max_tiles))
        md.Maximize(sum(xs))
        sv = cp_model.CpSolver()
        sv.parameters.max_time_in_seconds = 30
        sv.parameters.num_workers = 2
        sv.Solve(md)
        chosen = [t for x, t in zip(xs, cand) if sv.Value(x)]
    except ImportError:
        chosen, used = [], set()
        for t in cand:
            if not (sites(t) & used) and (not max_tiles or len(chosen) < max_tiles):
                chosen.append(t)
                used |= sites(t)
    cpl = set(logical_couplers(lay, lay.n))
    for u, _, _ in chosen:
        for g, c, *rest in shor9_encoder(u):
            if g == "cx" and tuple(sorted((c, rest[0]))) not in cpl:
                raise RuntimeError(f"shor9: tile CX {c}-{rest[0]} is not a coupler of {layout_id(lay)}")
    return sorted(chosen, key=lambda t: t[1])


def _local_unit(u):
    """The unit renumbered onto 0..m^2-1 in its sampling order (blocks in turn)."""
    qs = [q for b in u["blocks"] for q in b]
    loc = {q: i for i, q in enumerate(qs)}
    return qs, [[loc[q] for q in b] for b in u["blocks"]]


def _shor9_one_run(sim, mapq, tiles, rot, shots):
    """Every unit of the register in one circuit, logical states rotated by rot; each unit then sampled on
    its own qubits: [(state, raw, decoded, histogram)] per unit."""
    states = [SHOR9_STATES[(k + rot) % 4] for k in range(len(tiles))]
    for (u, _, _), st_ in zip(tiles, states):
        shor9_prepare(sim, u, st_, mapq)
    out = []
    for (u, _, _), st_ in zip(tiles, states):
        # units are independent; shots_wide keeps bits above 31, which QrackSimulator's measure_shots drops
        # (PyQrack 2.28.1)
        qs, local = _local_unit(u)
        sh = shots_wide(sim, [mapq[q] for q in qs], shots)
        ok = [shor9_decode(x, local, st_) for x in sh]
        out.append((st_, sum(r for r, _ in ok) / len(ok), sum(d for _, d in ok) / len(ok), collections.Counter(sh)))
    return out


_SHOR_LAYOUTS = {}


def _shor9_reg_task(repo, reg, ra, tiles, rot, shots, exact, gpu):
    """One run of the register (a fresh ACE engine, or the exact simulator), in a pool worker or here."""
    lay = _SHOR_LAYOUTS.get((repo, reg))
    if lay is None:
        lay = _SHOR_LAYOUTS[(repo, reg)] = Layout(repo) if reg == "release" else RegisterLayout(repo, reg)
    if exact:
        from pyqrack import QrackSimulator
        return _shor9_one_run(QrackSimulator(lay.n, is_gpu=gpu), list(range(lay.n)), tiles, rot, shots)
    THETA_DIST["mode"] = ra.theta
    eng = AceEngine(lay, lay.n, ra)
    out = _shor9_one_run(eng.sim, eng.idx, tiles, rot, shots)
    del eng
    return out


def cmd_shor9_register(a):
    """Units tiled over a whole register, every unit in one circuit on the register's own ACE placement;
    logical states rotate over the units in four runs, so every unit is read in all four."""
    reg, m = a.register, int(a.size)
    lay = Layout(a.repo) if reg == "release" else RegisterLayout(a.repo, reg)
    _SHOR_LAYOUTS[(a.repo, reg)] = lay
    n = lay.n
    tiles = shor9_tiles(lay, a.max_tiles, m)
    if not tiles:
        raise SystemExit(f"shor9: no {m} x {m} square of {reg} is free of holes")
    ra = _RUN_PARSER["r"].parse_args(["--backend", "ace", "--register", reg] + (["--repo", a.repo] if a.repo else [])
                                     + (["--ace-gpu"] if a.gpu else ["--cpu"])
                                     + (["--ace-boundary-rep"] if getattr(a, "ace_boundary_rep", False) else []))
    ra.cmd = "run"
    THETA_DIST["mode"] = ra.theta
    apply_register_preset(ra, lay)
    probe = AceEngine(lay, n, ra)
    row = layout_row(probe.size, probe.lrc, probe.lrr, probe.torus, probe.opts)
    sids = [frozenset(s for s, _ in u) for u in row["unpack"]]
    idx = list(probe.idx)
    patch_of = lambda q: ("seam" if len(sids[idx[q]]) > 1 else f"p{next(iter(sids[idx[q]]))}")
    info = []
    for k, (u, (r0, c0), ori) in enumerate(tiles):
        qs = [q for b in u["blocks"] for q in b]
        regions = collections.Counter(patch_of(q) for q in qs)
        cls = collections.Counter()
        for g, c, *rest in shor9_encoder(u):
            if g == "cx":
                sa, sb = sids[idx[c]], sids[idx[rest[0]]]
                cls["exact" if len(sa) == 1 and sa == sb else "replica" if sa & sb else "cross"] += 1
        kind = "touches seam" if "seam" in regions else "one patch" if len(regions) == 1 else "two patches"
        info.append(dict(unit=k, rows=[r0, r0 + m - 1], cols=[c0, c0 + m - 1], leaders=ori, blocks=u["blocks"],
                         regions=dict(regions), cx_classes=dict(cls), kind=kind))
    del probe
    ver = qrack_versions()
    R = max(1, a.repeats)
    keys = [(rot, r) for rot in range(4) for r in range(R)]
    pool, workers = shor_pool(getattr(a, "jobs", 1), len(keys) + (0 if a.no_exact else 4))
    print(f"Shor-{m * m} units across {reg}: {len(tiles)} units on {m * m * len(tiles)} of {n} qubits, ACE "
          f"{row['lrc']}x{row['lrr']}{' torus' if row['torus'] else ''} preset placement, {a.shots} shots, "
          f"{a.repeats} fresh backend(s) per rotation, "
          f"{'OpenCL' + (' (QRACK_OCL_SYNC=1)' if os.environ.get('QRACK_OCL_SYNC') == '1' else '') if a.gpu else 'CPU'}"
          f"{f', {workers} processes' if workers > 1 else ''}"
          f"{', boundary repetition code' if getattr(a, 'ace_boundary_rep', False) else ''}; {ver}")
    print("unit  rows   cols  leaders  regions                       encoder CX exact/replica/cross")
    for d in info:
        reg_s = ", ".join(f"{k} {v}" for k, v in sorted(d["regions"].items()))
        c = d["cx_classes"]
        print(f"{d['unit']:>4}  {d['rows'][0]:>2}-{d['rows'][1]:<2} {d['cols'][0]:>2}-{d['cols'][1]:<2} "
              f"{d['leaders']:>6}   {reg_s:<29} {c.get('exact', 0)}/{c.get('replica', 0)}/{c.get('cross', 0)}")

    acc = [defaultdict(list) for _ in tiles]
    hists = [defaultdict(collections.Counter) for _ in tiles]
    t0 = time.time()
    gpu = bool(a.gpu)
    ex_keys = [] if a.no_exact else [(rot, c) for rot in range(4) for c in shor_chunks(a.shots, -(-workers // 4))]
    try:
        out = shor_pool_map(pool, [(_shor9_reg_task, (a.repo, reg, ra, tiles, rot, a.shots, False, gpu))
                                   for rot, _ in keys]
                            + [(_shor9_reg_task, (a.repo, reg, ra, tiles, rot, c, True, gpu)) for rot, c in ex_keys])
    finally:
        if pool is not None:
            pool.shutdown()
    for res in out[:len(keys)]:
        for k, (st, raw, dec, h) in enumerate(res):
            acc[k][st].append((raw, dec))
            hists[k][st].update(h)
    exact_ok = None
    if not a.no_exact:
        exact_ok = all(raw == 1.0 and dec == 1.0 for res in out[len(keys):] for _, raw, dec, _ in res)
    print("\nunit  kind           |0_L>          |1_L>          |+_L>          |-_L>          logical Z  logical X")
    recs = []
    cmps = [{s: shor9_compare(hists[k][s], _local_unit(u)[1], s) for s in SHOR9_STATES}
            for k, (u, _, _) in enumerate(tiles)]
    for d, ac, cm in zip(info, acc, cmps):
        m_ = {s: (float(np.mean([r for r, _ in ac[s]])), float(np.mean([x for _, x in ac[s]]))) for s in SHOR9_STATES}
        z, x = (m_["0"][1] + m_["1"][1]) / 2, (m_["+"][1] + m_["-"][1]) / 2
        print(f"{d['unit']:>4}  {d['kind']:<13} " + " ".join(f"{m_[s][0]:.3f}/{m_[s][1]:.3f}" for s in SHOR9_STATES)
              + f"   {z:.3f}      {x:.3f}")
        recs.append(dict(test="shor9", size=m, code=f"shor{m * m}", register=reg, shots=a.shots, repeats=a.repeats,
                         qrack=ver, ocl_sync=os.environ.get("QRACK_OCL_SYNC") == "1",
                         boundary_rep=bool(getattr(a, "ace_boundary_rep", False)),
                         device="gpu" if a.gpu else "cpu", logical_z=z, logical_x=x, **d,
                         distribution={s: cm[s] for s in SHOR9_STATES},
                         **{f"raw_{s}": m_[s][0] for s in SHOR9_STATES},
                         **{f"decoded_{s}": m_[s][1] for s in SHOR9_STATES}))
    print("\noutcome distribution vs exact, per unit and logical state: total variation distance (floor: exact "
          "samples of the same size), share of outcomes the exact state never gives, the most frequent of those as "
          "blocks (leader first), x = the decoder fails on it")
    for d, cm in zip(info, cmps):
        shor9_print_dist(f"unit {d['unit']} " + ("seam" if d["kind"] == "touches seam" else "patch"), cm)
    by = defaultdict(list)
    for r in recs:
        by[r["kind"]].append(r)
    print("\nby placement: units, mean decoded logical Z / X")
    for kind, rs in sorted(by.items()):
        print(f"  {kind:<13} {len(rs):>2}   {np.mean([r['logical_z'] for r in rs]):.3f} / "
              f"{np.mean([r['logical_x'] for r in rs]):.3f}")
    if exact_ok is not None:
        print(f"  exact reference (QrackSimulator, {n} qubits): " + ("every unit and state 1.000" if exact_ok
                                                                       else "NOT 1.0 everywhere"))
    print(f"  seconds: {time.time() - t0:.1f}  (shot noise alone +/- {0.5 / math.sqrt(2 * a.shots * max(1, a.repeats)):.3f} "
          f"per logical basis)")
    if a.out and exact_ok is not False:
        with open(a.out, "a") as fh:
            for r in recs:
                fh.write(json.dumps(dict(r, when=time.strftime("%Y-%m-%dT%H:%M:%S"))) + "\n")
        print(f"  appended {len(recs)} records to {a.out}")
    if exact_ok is False:
        shor9_bad_reference(bool(a.gpu))


def cmd_shor9(a):
    if getattr(a, "ocl_sync", False):
        os.environ["QRACK_OCL_SYNC"] = "1"           # read by Qrack when its first OpenCL engine starts
    if getattr(a, "register", None):
        return cmd_shor9_register(a)
    m = int(a.size)
    if m < 3:
        raise SystemExit("shor9: --size must be 3 or more (3 = Shor-9)")
    gpu = bool(a.gpu)
    t0 = time.time()
    results, geo, workers = shor9_bench(a.shots, gpu=gpu, exact=not a.no_exact, repeats=a.repeats, m=m,
                                        jobs=getattr(a, "jobs", 1), brep=bool(getattr(a, "ace_boundary_rep", False)))
    gs = ", ".join(f"{k}={v}" for k, v in geo.items())
    if m % 2 == 0:
        print(f"# --size {m}: patch_blocks needs whole blocks inside each patch, and {m * (m - 1) // 2} bulk "
              f"qubits per patch do not split into blocks of {m}: seam_leaders only")
    ver = qrack_versions()
    print(f"Shor-{m * m} on ACE ({m}x{m}, {gs}), {a.shots} shots per logical state, "
          f"{'OpenCL' + (' (QRACK_OCL_SYNC=1)' if os.environ.get('QRACK_OCL_SYNC') == '1' else '') if gpu else 'CPU'}"
          + (f", {a.repeats} fresh backends per state (means)" if a.repeats > 1 else "")
          + (f", {workers} processes" if workers > 1 else "")
          + (", boundary repetition code" if getattr(a, "ace_boundary_rep", False) else "") + f"; {ver}")
    print("success rate per logical state as raw / decoded  (1.0 perfect, 0.5 no logical information); "
          "'vs exact': total variation distance to the exact distribution (floor: exact samples of the same size), "
          "share of impossible outcomes, the most frequent of those (blocks, leader first; x = decoder fails, "
          "tie = tied majority, even m)")
    recs = []
    for name, row in results.items():
        print(f"  {name:14s} blocks {row['blocks']}")
        print("      " + "  ".join(f"|{s}_L>: {row[s][0]:.3f} / {row[s][1]:.3f}" for s in SHOR9_STATES))
        z = (row["0"][1] + row["1"][1]) / 2
        x = (row["+"][1] + row["-"][1]) / 2
        spread = ""
        if len(row.get("runs", {}).get("0", [])) > 1:
            zs = [(r0[1] + r1[1]) / 2 for r0, r1 in zip(row["runs"]["0"], row["runs"]["1"])]
            xs = [(rp[1] + rm[1]) / 2 for rp, rm in zip(row["runs"]["+"], row["runs"]["-"])]
            spread = (f"  (backend to backend: Z {min(zs):.3f}-{max(zs):.3f}, X {min(xs):.3f}-{max(xs):.3f}; "
                      f"shot noise alone +/- {0.5 / math.sqrt(2 * a.shots):.3f})")
        print(f"      decoded logical-Z success {z:.3f}, logical-X success {x:.3f}{spread}")
        cm = {s: shor9_compare(row["hist"][s], row["blocks"], s) for s in SHOR9_STATES} if "hist" in row else None
        if cm:
            shor9_print_dist("vs exact", cm)
        recs.append(dict(test="shor9", size=m, code=f"shor{m * m}", ace=geo, layout=name, blocks=row["blocks"],
                         qrack=ver, ocl_sync=os.environ.get("QRACK_OCL_SYNC") == "1",
                         boundary_rep=bool(getattr(a, "ace_boundary_rep", False)), shots=a.shots, repeats=a.repeats, runs=row.get("runs"), **({"distribution": cm} if cm else {}),
                         device="gpu" if gpu else "cpu", logical_z=z, logical_x=x,
                         **{f"raw_{s}": row[s][0] for s in SHOR9_STATES},
                         **{f"decoded_{s}": row[s][1] for s in SHOR9_STATES}))
    secs = time.time() - t0
    print(f"  seconds: {secs:.2f}")
    bad_ref = False
    if "exact_reference" in results:
        ref = results["exact_reference"]
        bad_ref = any(abs(ref[s][1] - 1.0) > 1e-12 or abs(ref[s][0] - 1.0) > 1e-12 for s in SHOR9_STATES)
    if a.out and not bad_ref:
        with open(a.out, "a") as fh:
            for r in recs:
                fh.write(json.dumps(dict(r, seconds=round(secs, 2), when=time.strftime("%Y-%m-%dT%H:%M:%S"))) + "\n")
        print(f"  appended {len(recs)} records to {a.out}")
    if bad_ref:
        shor9_bad_reference(gpu)


def qrack_versions():
    """'pyqrack X, Qrack Y' (the library from its versioned file name, when there is one)."""
    try:
        from importlib.metadata import version
        pv = version("pyqrack")
    except Exception:
        pv = "?"
    lib = "?"
    try:
        import pyqrack.qrack_system as qs
        path = Path(qs.Qrack.qrack_lib._name).resolve()
        cand = sorted(path.parent.glob(path.name + ".*")) or [path]
        m_ = re.search(r"\.so\.([\d.]+)$", str(cand[-1]))
        lib = m_.group(1) if m_ else path.name
    except Exception:
        pass
    return f"pyqrack {pv}, Qrack {lib}"


def shor9_bad_reference(gpu):
    """The exact simulator must decode every state at 1.0; when it does not, nothing of the run is trusted."""
    msg = "shor9: the exact reference is not 1.0 everywhere, so this run's numbers are not recorded"
    if gpu:
        msg += ("; on OpenCL that is Qrack's asynchronous dispatch defect (README section 5.3): use Qrack >= 10.13.2 "
                "with --ocl-sync (QRACK_OCL_SYNC=1), or the patched build, or drop --gpu")
    raise SystemExit(msg)


def richardson_lambda_p(lams, ys, p):
    """Richardson extrapolation to lambda = 0 in u = lambda^p (Dan's qem script; p = 1 is ordinary
    Richardson): returns (y0, weights)."""
    u = [float(l) ** p for l in lams]
    w = []
    for i, ui in enumerate(u):
        wi = 1.0
        for j, uj in enumerate(u):
            if j != i:
                wi *= uj / (uj - ui)
        w.append(wi)
    return float(sum(wi * yi for wi, yi in zip(w, ys))), w


def run_full(lay, eng, d, pub, a):
    """One pub of the unpatched forward circuit (release seed 2025, instance 0), as the
    10 x 100k-shot pubs of the 10^6-sample run. Not scorable at 61 qubits: stored only."""
    cyc = build_cycles(lay, d, instance_seed(0, "full"), 0, n=eng.n)
    eng.reset()
    apply_forward(eng, cyc)
    k = SAMPLE_SHOTS_PER_PUB if a.shots == "paper" else int(a.shots)
    sh = eng.shots(k)
    return dict(fidelity=None, se=None, shots=int(k), _shots={"shots": sh})


# ================================================================== bitstring storage
def shots_base(a):
    return Path(a.shots_dir) if a.shots_dir else Path(a.out).with_name(Path(a.out).stem + "_shots")


def shots_root(a):
    """<shots-dir or out stem_shots>/<cfg tag>: configurations never mix in one release/."""
    return shots_base(a) / cfg_tag(a)


def claim_dir(a):
    """Claims of a single configuration, or of a --variants group (one claim per point
    covers every variant scored against that point's reference)."""
    return shots_base(a) / (getattr(a, "claim_tag", None) or cfg_tag(a)) / "claims"


def point_file(a, n, fam, K, d, j, i):
    return shots_root(a) / "points" / f"n{n}" / f"{fam}_K{K}_d{d:02d}_p{j}_i{i}.npz"


def save_point(path, arrays):
    """Atomic write, so an interrupted run never leaves a half file behind a record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in arrays.items()})
    os.replace(tmp, path)


def pack_shots(recs, lay, a):
    """Merge the per-point files into the release's data/ layout under <shots>/release:
    counts/patched_K{K}_d{d}.npz (partition{j}_instance{i}, sorted uint64),
    counts/mirror_survival.json (hits[i][s] / shots[i][s]) plus mirror_shots.npz with every
    raw mirror shot and its twirl index, samples/full_d{d}.npz (shots, sorted).
    Truncated registers get an _n{n} suffix; bit q is logical qubit q throughout."""
    rel = shots_root(a) / "release"
    groups, missing = defaultdict(dict), 0
    for (n, fam, K, d, j, i), r in sorted(recs.items(), key=lambda kv: kv[0]):
        if fam in ("hamming", "vqe") or r.get("estimator") == "prob":
            continue                               # shot-free estimators keep no bitstrings
        f = point_file(a, n, fam, K, d, j, i)
        if not f.exists():
            missing += fam != "mirror" or not a.exact_probs or a.backend == "ace"
            continue
        groups[(n, fam, K, d)][(j, i)] = (f, r)
    written = []
    for (n, fam, K, d), pts in groups.items():
        sfx = "" if n == lay.n else f"_n{n}"
        if fam == "patched":
            out = rel / "counts" / f"patched_K{K}_d{d}{sfx}.npz"
            arrays = {}
            for (j, i), (f, _) in pts.items():
                with np.load(f) as z:
                    arrays[f"partition{j}_instance{i}"] = np.sort(z["shots"].astype(np.uint64))
        elif fam == "full":
            out = rel / "samples" / f"full_d{d}{sfx}.npz"
            parts = []
            for _, (f, _) in sorted(pts.items()):
                with np.load(f) as z:
                    parts.append(z["shots"].astype(np.uint64))
            arrays = {"shots": np.sort(np.concatenate(parts))}
        else:
            out = rel / "counts" / f"mirror_shots_d{d:02d}{sfx}.npz"
            arrays = {}
            for (_, i), (f, _) in pts.items():
                with np.load(f) as z:
                    arrays.update({f"instance{i}_{k}": z[k] for k in z.files})
        save_point(out, arrays)
        written.append(out)
    mirror = defaultdict(dict)
    for (n, fam, K, d), pts in groups.items():
        if fam == "mirror":
            for (_, i), (_, r) in sorted(pts.items()):
                mirror[n].setdefault(str(d), {"instances": [], "hits": [], "shots": []})
                e = mirror[n][str(d)]
                e["instances"].append(i)
                e["hits"].append(r["hits"])
                e["shots"].append(r["shots_per_input"])
    for n, depths in mirror.items():
        sfx = "" if n == lay.n else f"_n{n}"
        out = rel / "counts" / f"mirror_survival{sfx}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"description": "Clean-qubit emulation (" + a.backend + "): hits[i][s] of "
                                   "shots[i][s] executions of instance instances[i] prepared in input string s "
                                   "returned that string. Depth = cycles of U U^dagger.", "depths": depths}, indent=1))
        written.append(out)
    if written:
        print(f"# bitstrings: {len(written)} release-format files under {rel}")
    if missing:
        print(f"# note: {missing} records have no stored bitstrings (written before storage existed)")


def record_files(out):
    """The main --out file plus every worker's <stem>.w<pid>.jsonl beside it."""
    o = Path(out)
    return [o] + sorted(o.parent.glob(o.stem + ".w*.jsonl"))


_JSONL = {}     # (file, tag) -> [byte offset read so far, inode, {key: record}]
_BAD_RECORDS = set()   # records skipped for a non-finite score: (file, n, family, K, depth, partition, instance)


def load_jsonl(path, tag, n_default):
    """All records of this cfg tag across the main and worker files. Incremental: each
    file is read only from where the previous call stopped, and only up to its last
    complete line, so polling every point (worker loop) or every 30 s (launcher) costs
    the new bytes, not the whole history. A file that shrank or was replaced is re-read
    from the start. Returns a fresh dict each call; callers may modify it."""
    recs = {}
    for f in record_files(path) if path else []:
        try:
            stt = f.stat()
        except FileNotFoundError:
            continue
        ck = (str(f), tag)
        ent = _JSONL.get(ck)
        if ent is None or stt.st_size < ent[0] or stt.st_ino != ent[1]:
            ent = _JSONL[ck] = [0, stt.st_ino, {}]
        if stt.st_size > ent[0]:
            with open(f, "rb") as fh:
                fh.seek(ent[0])
                chunk = fh.read()
            end = chunk.rfind(b"\n") + 1            # an unfinished last line waits for next time
            for line in chunk[:end].splitlines():
                try:
                    r = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):   # a line cut by a killed worker
                    continue
                if r.get("cfg") == tag:
                    f_ = r.get("fidelity")
                    if isinstance(f_, float) and not math.isfinite(f_):
                        _BAD_RECORDS.add((str(f), r["n"], r["family"], r["K"], r["depth"], r["partition"], r["instance"]))
                        continue
                    r.setdefault("n", n_default)
                    ent[2][(r["n"], r["family"], r["K"], r["depth"], r["partition"], r["instance"])] = r
            ent[0] += end
        recs.update(ent[2])
    return recs


# ================================================================== work queue (claim files)
def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def claim_file(a, key):
    n, fam, K, d, j, i = key
    return claim_dir(a) / f"n{n}_{fam}_K{K}_d{d:02d}_p{j}_i{i}"


def _write_atomic(path, text):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def try_claim(path, max_attempts=2, retry_failed=False):
    """O_EXCL claim holding 'pid:attempt'. 'done' claims are final. A claim whose worker
    died (OOM kill, memory cap, segfault) is taken over, up to max_attempts workers in
    total; after that the point is marked 'failed: ...' so the remaining workers do not
    crash on it one after another. --retry-failed clears that."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}:{getattr(try_claim, 'next_attempt', 1)}".encode())
            os.close(fd)
            try_claim.next_attempt = 1
            return True
        except FileExistsError:
            pass
        try:
            txt = path.read_text().strip()
        except FileNotFoundError:
            continue
        if txt == "done":
            return False
        if txt.startswith("failed"):
            if not retry_failed:
                return False
            att = 0
        else:
            pid_s, _, att_s = txt.partition(":")
            if not pid_s.isdigit():
                return False                        # being written right now
            if _pid_alive(int(pid_s)):
                return False
            att = int(att_s) if att_s.isdigit() else 1
            if att >= max_attempts:
                _write_atomic(path, f"failed: {att} worker(s) died on this point (memory cap, OOM or crash)")
                return False
        try:
            os.rename(path, path.with_name(path.name + f".stale{os.getpid()}"))
        except FileNotFoundError:
            pass
        try_claim.next_attempt = att + 1
    return False


def failed_points(a):
    root = claim_dir(a)
    out = []
    if root.exists():
        for f in sorted(root.iterdir()):
            if f.is_file() and "." not in f.name:
                try:
                    t = f.read_text().strip()
                except OSError:
                    continue
                if t.startswith("failed"):
                    out.append((f.name, t))
    return out


def mark_done(path):
    _write_atomic(path, "done")


def _point(v):
    """Inverse-variance mean of (F, se) pairs; plain mean if any se is 0 (exact probs)."""
    if all(s > 0 for _, s in v):
        return inverse_variance_mean(*zip(*v))
    return float(np.mean([x for x, _ in v])), 0.0


def _decay(pts, floor):
    """Per-cycle decay b = -dlnF/dd from a log-linear fit over depths with F > floor.
    Returns (b or None, prefactor, number of depths dropped)."""
    good = [(d, F) for d, F in sorted(pts.items()) if F > floor]
    dropped = len(pts) - len(good)
    if len(good) < 2:
        return None, None, dropped
    A, f = fit_exp([d for d, _ in good], [F for _, F in good])
    return -math.log(f), A, dropped


def report_failed(a):
    bad = failed_points(a)
    if bad:
        print(f"\n# {len(bad)} point(s) marked failed (rerun with --retry-failed to try again):")
        for name, why in bad:
            print(f"#   {name}: {why}")


GEO_KEYS = ("ace_register", "ace_lrc", "ace_lrr", "ace_torus", "ace_map")


def report_geometry(recs, a):
    """ACE geometry actually used, per size, from the records. More than one under a tag
    means the auto rule (or Qrack's layout) changed between runs: those points mix
    different seam placements and should not be fitted together."""
    if a.backend != "ace":
        return
    seen = defaultdict(set)
    for r in recs.values():
        if "ace_lrc" in r:
            seen[r["n"]].add(tuple(r.get(k) for k in GEO_KEYS))
    for n in sorted(seen):
        if len(seen[n]) > 1:
            opts = "; ".join(", ".join(f"{k[4:]}={v}" for k, v in zip(GEO_KEYS, g)) for g in sorted(seen[n], key=str))
            print(f"# WARNING n={n}: records under this tag used {len(seen[n])} ACE geometries ({opts}). "
                  f"Use a fresh --out, or explicit --lrc/--lrr, to keep them apart.")


def summarize(recs, lay, a):
    report_failed(a)
    report_geometry(recs, a)
    fvd = json.loads((lay.repo / "data/results/fidelity_vs_depth.json").read_text())
    by = defaultdict(list)
    min_shots = {}
    n_full = defaultdict(int)
    fx = defaultdict(list)
    hm = defaultdict(list)
    vq = []
    for r in recs.values():
        if r["family"] == "vqe":
            vq.append(r)
            continue
        if r["family"] == "fxeb":
            fx[(r["n"], r["depth"])].append(r)
            continue
        if r["family"] == "full":
            n_full[(r["n"], r["depth"])] += r["shots"]
            continue
        if r["family"] == "hamming":
            hm[(r["n"], r["depth"], r["K"])].append(r["fidelity"])
            continue
        fam = "mirror" if r["family"] == "mirror" else f"{r['K']}-patch"
        by[(r["n"], fam, r["depth"])].append((r["fidelity"], r["se"]))
        if r["family"] == "mirror" and r.get("shots"):
            min_shots[r["n"]] = min(min_shots.get(r["n"], r["shots"]), r["shots"])
    for (n, d), k in sorted(n_full.items()):
        print(f"full circuit n={n} d={d}: {k:,} stored samples (no score: not verifiable at this size)")
    if fx:
        A_hw, f_hw = fvd["fit"]["prefactor"], fvd["fit"]["fidelity_per_cycle"]
        print(f"\n== forward XEB of the {a.backend} samples vs the exact reference (nn_qab estimator)"
              f"{', theta ' + a.theta + ' VARIANT' if getattr(a, 'theta', 'haar') != 'haar' else ''} ==")
        print("(ideal XEB = collision ratio of the exact output: 1 = Porter-Thomas, as the paper's circuits;"
              " large = concentrated output, easier to score on)")
        print("n    d   XEB (Eq.1, one patch)   linear XEB   HOG     ideal XEB   instances"
              "   [ibm_phoenix 61q fit at d]")
        for (n, d) in sorted(fx):
            v = fx[(n, d)]
            F, se = inverse_variance_mean([r["fidelity"] for r in v], [r["se"] for r in v])
            print(f"{n:<4} {d:>2}  {F:8.5f} +/- {se:.1e}      {np.mean([r['xeb_linear'] for r in v]):8.5f}"
                  f"   {np.mean([r['hog'] for r in v]):.3f}   {np.mean([r['ideal_xeb'] for r in v]):.4f}"
                  f"      {len(v)}          [{A_hw * f_hw ** d:.3e}]")
    if hm:
        print(f"\n== mirror Hamming-weight signal 1 - 2h of the {a.backend} backend (Dan Strano's observable; "
              f"ideal 1; plain means over instances) ==")
        print("n    d   fold  layers  signal              instances")
        for (n, d, lam) in sorted(hm):
            v = np.array(hm[(n, d, lam)])
            se = v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float("nan")
            print(f"{n:<4} {d:>2}  {lam:>4}  {2 * d * lam:>6}  {v.mean():8.5f} +/- {se:.1e}   {len(v)}")
    if vq:
        vqe_table(vq, a.backend)
    sizes = sorted({k[0] for k in by})
    b_rows, fit_depths = [], set()
    for n in sizes:
        fams = [f for f in ("mirror", "3-patch", "4-patch") if any(k[0] == n and k[1] == f for k in by)]
        full = n == lay.n and is_release(lay)
        print(f"\n== clean-qubit emulation ({a.backend}), n = {n}"
              + (" -- the experiment's register, compared with ibm_phoenix ==" if full else
                 f" -- register {layout_id(lay)} (no device data) ==" if not is_release(lay) and n == lay.n
                 else " (first-n truncation) =="))
        print("family    d   F_sim" + ("            F_hardware     F_sim/F_hw" if full else ""))
        mpts = {}
        for fam in fams:
            for d in sorted(k[2] for k in by if k[0] == n and k[1] == fam):
                F, se = _point(by[(n, fam, d)])
                if fam == "mirror":
                    mpts[d] = F
                hw = fvd["points"][fam].get(str(d), {}).get("fidelity") if full else None
                print(f"{fam:8s} {d:>3}  {F:.5f}+/-{se:.1e}  " + (f"{hw:.3e}      {F / hw:10.1f}" if hw else ""))
        floor = 3.0 / min_shots[n] if n in min_shots else 0.0
        b, A, dropped = _decay(mpts, floor)
        if b is not None:
            czpc = float(np.mean([mirror_czpc(lay, n, d) for d in mpts]))
            fit_depths.update(mpts)
            b_rows.append((n, czpc, b, dropped))
            print(f"mirror fit F(d) = {A:.4f} x {math.exp(-b):.5f}^d"
                  + (f"  vs device {fvd['fit']['prefactor']:.4f} x {fvd['fit']['fidelity_per_cycle']:.4f}^d" if full else "")
                  + (f"  ({dropped} depth(s) at or below 3/shots dropped)" if dropped else ""))

    if not b_rows:
        return
    b_dev = -math.log(fvd["fit"]["fidelity_per_cycle"])
    print(f"\n== per-cycle decay b(N) of the {a.backend} simulator (mirror) ==")
    print("N    CZ/cycle   b(N)       per-cycle F   err/qubit/cycle")
    for n, c, b, dr in b_rows:
        print(f"{n:<4} {c:8.2f}   {b:9.5f}{'*' if dr else ' '} {math.exp(-b):10.5f}    {b / n:.2e}")
    print(f"device (ibm_phoenix, N=61): b = {b_dev:.5f}, per-cycle {math.exp(-b_dev):.4f}, err/qubit/cycle {b_dev / 61:.2e}")
    if any(dr for *_, dr in b_rows):
        print("(* fitted with deep depths dropped at or below 3/shots: shallow depths only)")
    if a.backend == "exact":
        print("(exact backend: b = 0 up to shot noise by construction -- a pipeline control)")
        return
    if len(b_rows) >= 2 and is_release(lay):
        X = np.array([[n, c] for n, c, _, _ in b_rows])
        y = np.array([b for _, _, b, _ in b_rows])
        (u, v), *_ = np.linalg.lstsq(X, y, rcond=None)
        if u < 0 or v < 0:                      # non-negative: best single-term fit
            cands = []
            for col in (0, 1):
                cj = max(float(X[:, col] @ y / (X[:, col] @ X[:, col])), 0.0)
                cands.append((float(np.sum((y - cj * X[:, col]) ** 2)), col, cj))
            _, col, cj = min(cands)
            u, v = (cj, 0.0) if col == 0 else (0.0, cj)
        c61 = float(np.mean([mirror_czpc(lay, lay.n, d) for d in sorted(fit_depths)]))
        b61 = u * lay.n + v * c61
        meas = next((b for n, _, b, _ in b_rows if n == lay.n), None)
        print(f"\nfit b = u*N + v*CZpc (non-negative): u {u:.3e}/qubit, v {v:.3e}/CZ"
              "  (N and CZ/cycle are nearly collinear: trust b(61), not u and v separately)")
        print(f"at N=61 ({c61:.2f} CZ/cycle): b {b61:.5f}, per-cycle {math.exp(-b61):.5f}, "
              f"F(36) {math.exp(-36 * b61):.3e}" + (f"; measured at 61: b {meas:.5f}" if meas is not None else "")
              + f";  sim/device decay ratio {b61 / b_dev:.2f} ({'cleaner' if b61 < b_dev else 'noisier'} than ibm_phoenix)")
        if len(b_rows) < 5 or max(n for n, *_ in b_rows) - min(n for n, *_ in b_rows) < 6:
            print("(few or narrowly spaced sizes: the 61-qubit numbers are indicative only)")


_WATCH = {"claim": None}


def start_rss_watchdog(limit_gb, period=2.0):
    """Real memory cap: a daemon thread reads this process's resident set from
    /proc/self/status; above limit_gb it marks the point being worked on as failed and
    exits, before the kernel OOM killer picks a victim. Unlike ulimit -v it does not
    count the address space Qrack's thread pool reserves but never touches."""
    import threading

    def rss_gb():
        try:
            for line in open("/proc/self/status"):
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1048576
        except OSError:
            pass
        return 0.0

    def loop():
        while True:
            time.sleep(period)
            r = rss_gb()
            if r > limit_gb:
                cf = _WATCH["claim"]
                msg = f"failed: resident memory {r:.2f} GB above --max-rss-gb {limit_gb:g}"
                if cf is not None:
                    try:
                        _write_atomic(cf, msg)
                    except OSError:
                        pass
                print(f"# {msg}; worker exits", flush=True)
                os._exit(3)

    threading.Thread(target=loop, daemon=True).start()


def parse_gpus(s):
    return parse_sizes(s) if s else []


def build_jobs(lay, a, sizes, fams):
    insts = range(a.instances if a.instances is not None else lay.instances)
    jobs = []
    for n in sizes:
        for fam in fams:
            if fam == "patched" and n != lay.n:
                continue
            ds = a.depths or ([d for d in FULL_DEPTHS if d <= 20] if fam == "fxeb" else
                              [4, 6, 8, 10, 12] if fam == "hamming" else [1, 2, 3, 4] if fam == "vqe" else
                              lay.depths[{"mirror": "mirror", "patched": "patched", "full": "full_sampled"}[fam]])
            for d in ds:
                if fam == "fxeb":
                    jobs += [(n, "fxeb", 0, d, 0, i) for i in insts]
                elif fam == "vqe":                          # depth slot = HVA layers p; deterministic: one point
                    jobs.append((n, "vqe", 0, d, 0, 0))
                elif fam == "hamming":                      # K slot = folding scale factor lambda
                    jobs += [(n, "hamming", int(lam), d, 0, i) for lam in (a.fold or [1]) for i in insts]
                elif fam == "full":
                    jobs += [(n, "full", 0, d, p, 0) for p in range(a.pubs)]
                elif fam == "mirror":
                    jobs += [(n, "mirror", 0, d, 0, i) for i in insts]
                else:
                    for K in a.K:
                        parts = range(len(lay.partitions[K])) if a.partitions is None else range(a.partitions)
                        jobs += [(n, "patched", K, d, j, i) for j in parts for i in insts]
    # longest first (default), so the big points do not end up as a lone tail; --order shallow
    # takes the cheapest first, so results arrive early when few workers face very deep points
    sign = 1 if getattr(a, "order", "deep") == "shallow" else -1
    return sorted(jobs, key=lambda k: sign * (k[0] * k[3] * (MIRROR_SHOTS.get(k[3], 1) if k[1] == "mirror" else 1)))


def _strip_launch_args(argv):
    out, skip = [], False
    for t in argv:
        if skip:
            skip = False
            continue
        if t in ("--gpus", "--per-gpu"):
            skip = True
            continue
        if t.startswith("--gpus=") or t.startswith("--per-gpu=") or t == "--retry-failed":
            continue                                # the launcher clears failed marks once
        out.append(t)
    return out


# ================================================================== OpenCL device isolation
# clvk (OpenCL on Vulkan) accepts one device per context, while Qrack opens one context per
# platform with all of its devices: with more than one GPU visible through clvk, Qrack aborts
# at start-up ("Only one device per context is supported"). So every process that uses the
# GPU sees exactly one device:
#   clvk device   -> only clvk's ICD loaded, and Mesa's Vulkan device-select layer shows only
#                    that GPU (DRI_PRIME=pci-..., MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1);
#                    inside the process it is OpenCL device 0
#   other device  -> clvk's ICD hidden (it would still abort Qrack's init), Qrack pinned to
#                    the device's index among the remaining platforms
#   no clvk       -> unchanged: Qrack pinned to the global index (QRACK_OCL_DEFAULT_DEVICE)
# Device numbers given to --gpus / QRACK_OCL_DEFAULT_DEVICE are global: the order of
# `clinfo -l` (all platforms) in the launching environment.

_CL_PROBE = r'''
import ctypes, json, sys
try:
    cl = ctypes.CDLL("libOpenCL.so.1")
except OSError:
    print("[]"); sys.exit(0)
vp, u32, sz = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t
cl.clGetPlatformIDs.argtypes = [u32, ctypes.POINTER(vp), ctypes.POINTER(u32)]
cl.clGetPlatformInfo.argtypes = [vp, u32, sz, vp, ctypes.POINTER(sz)]
cl.clGetDeviceIDs.argtypes = [vp, ctypes.c_uint64, u32, ctypes.POINTER(vp), ctypes.POINTER(u32)]
cl.clGetDeviceInfo.argtypes = [vp, u32, sz, vp, ctypes.POINTER(sz)]
def info(fn, obj, param):
    n = sz()
    if fn(obj, param, 0, None, ctypes.byref(n)) or not n.value:
        return ""
    b = ctypes.create_string_buffer(n.value)
    fn(obj, param, n, b, None)
    return b.value.decode(errors="replace").strip()
np_ = u32()
if cl.clGetPlatformIDs(0, None, ctypes.byref(np_)) or not np_.value:
    print("[]"); sys.exit(0)
plats = (vp * np_.value)()
cl.clGetPlatformIDs(np_.value, plats, None)
out = []
for p in plats:
    pname = info(cl.clGetPlatformInfo, p, 0x0902)                 # CL_PLATFORM_NAME
    nd = u32()
    if cl.clGetDeviceIDs(p, 0xFFFFFFFF, 0, None, ctypes.byref(nd)) or not nd.value:
        continue
    devs = (vp * nd.value)()
    cl.clGetDeviceIDs(p, 0xFFFFFFFF, nd.value, devs, None)
    for d in devs:
        bus = (u32 * 4)()                                          # cl_khr_pci_bus_info
        ok = cl.clGetDeviceInfo(d, 0x410F, ctypes.sizeof(bus), bus, None) == 0
        out.append(dict(platform=pname, device=info(cl.clGetDeviceInfo, d, 0x102B),
                        pci=list(bus) if ok else None))
print(json.dumps(out))
'''

_CL_PLAN = {}
_ISO_KEYS = ("OCL_ICD_VENDORS", "DRI_PRIME", "MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE",
             "QRACK_OCL_DEFAULT_DEVICE", "QRACK_QUNITMULTI_DEVICES", "QRACK_QPAGER_DEVICES")


def base_env():
    """The environment as the user started the run: an isolated process (launcher or
    worker) restores the variables isolation changed, so device numbers stay global."""
    env = dict(os.environ)
    orig = env.pop("NIGHTHAWK_ORIG_ENV", None)
    env.pop("NIGHTHAWK_ISOLATED", None)
    if orig:
        for k in _ISO_KEYS:
            env.pop(k, None)
        env.update({k: v for k, v in json.loads(orig).items() if v is not None})
    return env


def _icd_files():
    """The .icd files the ICD loader reads in the user's environment."""
    src = base_env().get("OCL_ICD_VENDORS") or "/etc/OpenCL/vendors"
    p = Path(src)
    if p.is_file():
        return [p]
    return sorted(p.glob("*.icd")) if p.is_dir() else []


def _is_clvk_icd(f):
    try:
        return "clvk" in f.name.lower() or "clvk" in f.read_text().lower()
    except OSError:
        return False


def _icd_dir(name, files):
    """A private vendors directory holding exactly these .icd files."""
    import tempfile
    d = Path(tempfile.gettempdir()) / f"nighthawk_icd_{os.getuid()}" / name
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.icd"):
        old.unlink()
    for f in files:
        (d / f.name).write_text(f.read_text())
    return str(d)


def opencl_plan():
    """Global OpenCL device list of this environment and how to isolate each device.
    Enumerated in a child process, so the caller never loads an OpenCL driver itself.
    Without a clvk ICD nothing is probed: the old pinning applies unchanged."""
    if "plan" in _CL_PLAN:
        return _CL_PLAN["plan"]
    icds = _icd_files()
    clvk_icds = [f for f in icds if _is_clvk_icd(f)]
    plan = dict(devices=None, clvk=bool(clvk_icds))
    if clvk_icds:
        import subprocess
        try:
            res = subprocess.run([sys.executable, "-c", _CL_PROBE], capture_output=True, text=True, timeout=120,
                                 env=base_env())
            plan["devices"] = json.loads(res.stdout.strip().splitlines()[-1]) if res.stdout.strip() else []
        except (subprocess.SubprocessError, ValueError, IndexError) as err:
            raise SystemExit(f"OpenCL device probe failed ({err}); give --gpu-isolation off to skip it")
        plan["clvk_dir"] = _icd_dir("clvk", clvk_icds)
        plan["other_dir"] = _icd_dir("noclvk", [f for f in icds if f not in clvk_icds])
    _CL_PLAN["plan"] = plan
    return plan


def isolated_env(g, base=None):
    """Environment that shows a process only global OpenCL device g (see above)."""
    env = base_env() if base is None else dict(base)
    env["NIGHTHAWK_ORIG_ENV"] = json.dumps({k: env.get(k) for k in _ISO_KEYS})
    env["NIGHTHAWK_ISOLATED"] = str(g)
    plan = opencl_plan()
    devs = plan["devices"]
    if not plan["clvk"] or devs is None:
        env["QRACK_OCL_DEFAULT_DEVICE"] = str(g)
        for k in ("QRACK_QUNITMULTI_DEVICES", "QRACK_QPAGER_DEVICES"):
            env.setdefault(k, str(g))
        return env
    if not 0 <= g < len(devs):
        raise SystemExit(f"OpenCL device {g} does not exist here: {len(devs)} device(s) "
                         + ", ".join(f"#{i} {d['device']} [{d['platform']}]" for i, d in enumerate(devs)))
    d = devs[g]
    if d["platform"].lower() == "clvk":
        if not d["pci"]:
            raise SystemExit(f"clvk device #{g} ({d['device']}) reports no PCI address "
                             "(cl_khr_pci_bus_info): cannot isolate it")
        dom, bus, dev, fn = d["pci"]
        env.update(OCL_ICD_VENDORS=plan["clvk_dir"], DRI_PRIME=f"pci-{dom:04x}_{bus:02x}_{dev:02x}_{fn:x}",
                   MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE="1", QRACK_OCL_DEFAULT_DEVICE="0",
                   QRACK_QUNITMULTI_DEVICES="0", QRACK_QPAGER_DEVICES="0")
        return env
    local = sum(1 for x in devs[:g] if x["platform"].lower() != "clvk")
    env.update(OCL_ICD_VENDORS=plan["other_dir"], QRACK_OCL_DEFAULT_DEVICE=str(local),
               QRACK_QUNITMULTI_DEVICES=str(local), QRACK_QPAGER_DEVICES=str(local))
    return env


def describe_device(g):
    plan = opencl_plan()
    if not plan["clvk"] or plan["devices"] is None or not 0 <= g < len(plan["devices"]):
        return f"OpenCL device {g}"
    d = plan["devices"][g]
    pci = "{:04x}:{:02x}:{:02x}.{:x}".format(*d["pci"]) if d["pci"] else "?"
    return f"#{g} {d['device']} [{d['platform']}, {pci}]"


def ensure_isolated(a):
    """Qrack initialises OpenCL on its first simulator, CPU-only ones included, and aborts
    when clvk shows it more than one device. So when clvk has several devices, this process
    re-execs itself seeing one: QRACK_OCL_DEFAULT_DEVICE (default 0) for a single-process
    run, the first --gpus device for a launcher (which only builds CPU backends itself;
    its workers are isolated one by one from the original environment)."""
    if getattr(a, "gpu_isolation", "auto") == "off" or getattr(a, "cmd", "") in ("seamgap", "hamming"):
        return
    if os.environ.get("NIGHTHAWK_ISOLATED") is not None or getattr(a, "worker", False):
        return
    plan = opencl_plan()
    if not plan["clvk"] or not plan["devices"]:
        return
    if sum(1 for d in plan["devices"] if d["platform"].lower() == "clvk") <= 1:
        return                                      # Qrack copes with one clvk device
    gpus = parse_gpus(getattr(a, "gpus", None))
    g = gpus[0] if gpus else int(base_env().get("QRACK_OCL_DEFAULT_DEVICE", "0") or 0)
    role = "launcher" if gpus else "this process"
    print(f"# OpenCL isolation: {role} uses {describe_device(g)} only", flush=True)
    os.execve(sys.executable, [sys.executable] + sys.argv, isolated_env(g))


def worker_env(g, a=None):
    """Environment of a worker pinned to OpenCL device g. Qrack otherwise spreads a
    simulator's units and pages over every detected device, which on a mixed host
    (different vendors, drivers, PCIe links) puts ACE patches on cards the worker was
    not given; and with clvk it would abort outright (one device per context).
    --gpu-isolation off: the old pinning only (QRACK_OCL_DEFAULT_DEVICE=g; variables the
    user set explicitly are left alone)."""
    if a is not None and getattr(a, "gpu_isolation", "auto") == "off":
        env = dict(base_env(), QRACK_OCL_DEFAULT_DEVICE=str(g))
        for k in ("QRACK_QUNITMULTI_DEVICES", "QRACK_QPAGER_DEVICES"):
            env.setdefault(k, str(g))
        return env
    return isolated_env(g)


def launch(a, lay, tag, jobs, recs, done_keys=None, finish=None):
    """Spawn len(gpus) x per_gpu workers (QRACK_OCL_DEFAULT_DEVICE per worker), watch the
    record files, then pack the bitstrings and summarise once they are all done.
    done_keys/finish override how progress is read and what runs at the end (--variants)."""
    if done_keys is None:
        done_keys = lambda: load_jsonl(a.out, tag, lay.n)
    import subprocess
    gpus = parse_gpus(a.gpus)
    logdir = Path(a.out).with_name(Path(a.out).stem + "_logs")
    logdir.mkdir(parents=True, exist_ok=True)
    child = [sys.executable, os.path.abspath(__file__)] + _strip_launch_args(sys.argv[1:]) + ["--worker"]
    procs = []
    order = [(g, k) for k in range(a.per_gpu) for g in gpus]      # round-robin over the cards
    print(f"# launching {len(order)} workers (GPUs {gpus} x {a.per_gpu}), one every {a.stagger:g} s; "
          f"OpenCL inits serialised {a.init_gap:g} s apart; logs in {logdir}/", flush=True)
    if getattr(a, "gpu_isolation", "auto") != "off":
        plan = opencl_plan()
        if plan["clvk"]:
            for g in gpus:
                print(f"#   gpu{g} = {describe_device(g)}, isolated per worker", flush=True)
                if getattr(a, "ace_host_pointer", False) and plan["devices"] and \
                        plan["devices"][g]["platform"].lower() == "clvk":
                    print("#   note: clvk ignores host-pointer buffers: --ace-host-pointer has no effect "
                          "on clvk devices", flush=True)
    for idx, (g, k) in enumerate(order):
        if idx:
            time.sleep(a.stagger)
        env = worker_env(g, a)
        log = open(logdir / f"gpu{g}_job{k}.txt", "a")
        procs.append((g, k, subprocess.Popen(child, env=env, stdout=log, stderr=subprocess.STDOUT)))
        print(f"#   started gpu{g} job{k}", flush=True)
    total = len(jobs)
    t0, last, respawns = time.time(), -1, 0

    def open_points(have):
        """Points neither recorded nor marked done/failed: still worth a worker."""
        out = []
        for k in jobs:
            if k in have:
                continue
            try:
                t = claim_file(a, k).read_text().strip()
            except OSError:
                t = ""
            if t == "done" or t.startswith("failed"):
                continue
            out.append(k)
        return out

    try:
        while True:
            have = done_keys()
            alive = [p for *_, p in procs if p.poll() is None]
            # a worker the memory watchdog stopped (exit 3) is replaced while points remain;
            # other exits (crash loops, 3 failures in a row) are not
            for idx, (g, k, p) in enumerate(procs):
                if p.poll() == 3 and respawns < total and open_points(have):
                    env = worker_env(g, a)
                    log = open(logdir / f"gpu{g}_job{k}.txt", "a")
                    procs[idx] = (g, k, subprocess.Popen(child, env=env, stdout=log, stderr=subprocess.STDOUT))
                    respawns += 1
                    print(f"#   gpu{g} job{k} stopped by the memory watchdog; replacement started", flush=True)
                    alive.append(procs[idx][2])
            if not alive:
                break
            time.sleep(30)
            have = done_keys()
            done = sum(1 for k in jobs if k in have)
            if done != last:
                el = time.time() - t0
                new = done - len([k for k in jobs if k in recs])
                eta = f", ~{(total - done) * el / new / 3600:.1f} h left" if new > 0 else ""
                print(f"# {done}/{total} points done, {sum(p.poll() is None for *_, p in procs)} workers alive"
                      f" [{el / 60:.0f} min{eta}]", flush=True)
                last = done
    except KeyboardInterrupt:
        print("# stopping workers (finished points are kept; rerun the same command to resume)")
        for *_, p in procs:
            p.terminate()
        raise SystemExit(130)
    bad = [(g, k, p.returncode) for g, k, p in procs if p.returncode and p.returncode != 3]
    if bad:
        print(f"# workers with errors (see logs): {bad}")
    if finish is not None:
        return finish()
    recs = load_jsonl(a.out, tag, lay.n)
    pack_shots(recs, lay, a)
    summarize(recs, lay, a)


def mem_total_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return None


def gpu_die_check(lay, n, a, row, size):
    """--ace-gpu: where the patches go on one die (VRAM / GTT) when fully entangled; stop
    before any work if they cannot fit even with --gtt."""
    if a.cpu or not getattr(a, "ace_gpu", False) or getattr(a, "ace_host_pointer", False):
        return
    vram, gtt = float(getattr(a, "vram_gb", 8.0) or 8.0), float(getattr(a, "gtt", 0) or 0)
    w = active_widths(lay, n, a, row, size)
    sp = die_split(w, vram, gtt)
    print(f"# die n={n}: patches {w} fully entangled: {describe_split(sp, vram, gtt)}"
          + (f"; QRACK_MAX_ALLOC_MB={sp['limit_mb']}" if gtt else ""), flush=True)
    if not sp["fits"] and not getattr(a, "worker", False):
        need = max(0.0, sp["gtt"])
        raise SystemExit(f"n={n}: the patches need {sp['vram']:.2f} GiB of VRAM and {need:.2f} GiB of host-mapped "
                         f"memory on one {vram:g} GB die: " + (f"give --gtt {math.ceil(need)} (or more)" if sp["vram"] <=
                         vram - GTT_RESERVE_GB else "the VRAM part alone is too large; use narrower patches "
                         "(--transpile-max-width) or --ace-host-pointer (every state in host memory)"))


def ace_preflight(lay, a, sizes):
    """--ace-max-width: print the layout each register size gets and check the
    worst-case dense memory of all workers against this machine's RAM.
    --ace-tiling with fixed lrc/lrr: print the coupler counts the tiling reaches."""
    if a.backend != "ace":
        return
    if getattr(a, "ace_transpile", False):
        f = lambda x: f"{x['exact']}/{x['replica']}/{x['cross']}, {x['seam_used']} seam, {x['sims_used']} sims"
        for n in sizes:
            size = ace_size(lay, n, a)
            row = layout_row(size, int(a.lrc), int(a.lrr), bool(a.ace_torus), ace_opts(a))
            lc = logical_couplers(lay, n)
            t = placement_stats(row, transpile_layout(lay, n, a, row, size), lc)
            ex = parse_exclude(getattr(a, "ace_exclude", None))
            print(f"# ace transpile ({getattr(a, 'transpile_engine', 'exact')}) n={n}, {size} sites, "
                  f"lrc {a.lrc} lrr {a.lrr}{' torus' if a.ace_torus else ''}, "
                  f"{len(ex)} sites excluded: exact/replica/cross {f(t)}, simulator widths {row['widths']}",
                  flush=True)
            gpu_die_check(lay, n, a, row, size)
            host_plan(lay, n, a, row, size)
        return
    if getattr(a, "ace_tiling", False) and not a.ace_max_width:
        for n in sizes:
            size, _, C = ace_register(lay, n, a.ace_layout)
            row = layout_row(size, int(a.lrc), int(a.lrr), bool(a.ace_torus), ace_opts(a))
            sids = [frozenset(sid for sid, _ in u) for u in row["unpack"]]
            start = [r_ * C + c_ for r_, c_ in lay.pos[:n]]
            lc = logical_couplers(lay, n)
            g = placement_stats(row, start, lc)
            t = placement_stats(row, ace_tiling(row, start, lc, size, tiling_version(a)), lc)
            f = lambda x: f"{x['exact']}/{x['replica']}/{x['cross']}, {x['seam_used']} seam, {x['sims_used']} sims"
            print(f"# ace tiling v{tiling_version(a)} n={n}, lrc {a.lrc} lrr {a.lrr}: exact/replica/cross "
                  f"{f(g)} on the device grid -> {f(t)} tiled", flush=True)
        return
    if not getattr(a, "ace_max_width", None):
        return
    workers = len(parse_gpus(a.gpus)) * a.per_gpu if a.gpus else 1
    worst = 0.0
    for n in sizes:
        size, _, C = ace_register(lay, n, a.ace_layout)
        cp = circuit_couplers(lay, n, a.ace_layout)
        tiling = (lay, n, [r_ * C + c_ for r_, c_ in lay.pos[:n]], tiling_version(a)) if a.ace_tiling else None
        r = widest_ace_config(size, a.ace_max_width, cp, torus_search(a), tiling)
        gb = dense_gb(r["widths"])
        worst = max(worst, gb)
        print(f"# ace n={n} ({size} sites), max width {a.ace_max_width}: lrc={r['lrc']} lrr={r['lrr']} "
              f"torus={r['torus']}, {r['cross_cz']}/{len(cp)} couplers across simulators, "
              f"{r['seam_cz']} on a seam, {r['boundary']} seam qubits, "
              f"simulator widths {r['widths']}, dense worst case {gb:,.1f} GB per worker", flush=True)
    ram = mem_total_gb()
    if ram and worst * workers > 0.85 * ram:
        print(f"# WARNING: {workers} worker(s) x {worst:,.1f} GB dense worst case > 85% of {ram:,.0f} GB RAM. "
              f"Shallow and truncated points stay far smaller (QUnit keeps unentangled qubits apart), "
              f"but deep full-register ones will not: lower --per-gpu, or set --max-rss-gb so an "
              f"oversized point is marked failed instead of tripping the OOM killer.", flush=True)
    if a.ace_gpu:
        print("# note: --ace-gpu with wide patches: each V340 die has 8 GB; patches over ~29 qubits "
              "(fp32) will not fit on one", flush=True)


# ================================================================== memory-aware scheduling
def mem_available_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return None


def swap_free_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("SwapFree:"):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return None


def host_available_gb(swap_gb=None):
    """MemAvailable, plus up to swap_gb of free swap when --swap-gb is given."""
    avail = mem_available_gb()
    if avail is None or not swap_gb:
        return avail
    return avail + min(float(swap_gb), swap_free_gb() or 0.0)


# Qrack refuses (or tries to elide) a QEngineCPU / QPager wider than QRACK_MAX_CPU_QB /
# QRACK_MAX_PAGING_QB; the build-time defaults are not documented, and no run here has gone past
# 28 qubits per patch. Wider CPU patches get both raised (setdefault: an explicit value wins).
QRACK_CPU_CAP_FROM = 28


def lift_qrack_cpu_caps(widest):
    cap = str(max(int(widest) + 2, 40))
    set_ = {}
    for var in ("QRACK_MAX_CPU_QB", "QRACK_MAX_PAGING_QB"):
        if var not in os.environ:
            os.environ[var] = cap
            set_[var] = cap
    return set_


def host_plan(lay, n, a, row, size):
    """CPU runs: dense states of the active patches, Qrack's compose transient (the widest
    patch is rebuilt one qubit wider while its old state is still held: + half its new size),
    against RAM and the swap --swap-gb allows. Prints only; it never stops the run."""
    if getattr(a, "ace_gpu", False):
        return
    w = active_widths(lay, n, a, row, size)
    if not w:
        return
    if max(w) > QRACK_CPU_CAP_FROM:
        lifted = lift_qrack_cpu_caps(max(w))
        if lifted:
            print("# qrack: " + ", ".join(f"{k}={v}" for k, v in lifted.items())
                  + f" (patches up to {max(w)} qubits on the CPU engine)", flush=True)
    states = dense_gb(w)
    compose = 0.5 * dense_gb([max(w)])
    worst = 1.0 + states + compose
    ram = mem_total_gb() or 0.0
    swap_cap = float(getattr(a, "swap_gb", 0) or 0.0)
    swap_now = swap_free_gb() or 0.0
    use = min(swap_cap, swap_now)
    print(f"# host: patches {w}: dense {states:,.0f} GiB + compose transient {compose:,.0f} GiB = worst case "
          f"{worst:,.0f} GiB per worker; RAM {ram:,.0f} GiB" + (f" + swap {use:,.0f} GiB (--swap-gb {swap_cap:g}, "
          f"{swap_now:,.0f} GiB free)" if swap_cap else ""), flush=True)
    if worst > 0.9 * ram:
        if not swap_cap:
            print(f"# WARNING: the worst case exceeds RAM; give --swap-gb GB to plan with swap, or deep points will "
                  f"reach the OOM killer", flush=True)
        elif worst > 0.95 * (ram + use):
            print(f"# WARNING: the worst case exceeds RAM + the swap allowed ({ram + use:,.0f} GiB): deep points may "
                  f"still be killed", flush=True)
        else:
            print(f"# note: about {max(0.0, states - 0.9 * ram):,.0f} GiB of the dense states will live in swap at "
                  f"full width; every gate pass over the widest patch then streams through it", flush=True)


def rss_gb_of(pid):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return 0.0


_ENGINE_GB = {}


def engine_gb(lay, n, a):
    """Worst-case memory of one simulation engine at size n, in GiB."""
    key = (n, a.backend, str(a.lrc), str(a.lrr), bool(getattr(a, "ace_torus", False)),
           tuple(sorted(ace_opts(a).items())), bool(getattr(a, "ace_tiling", False)), tiling_version(a),
           bool(getattr(a, "ace_transpile", False)), getattr(a, "ace_width", None),
           parse_exclude(getattr(a, "ace_exclude", None)), getattr(a, "transpile_seed", 0),
           getattr(a, "ace_layout", "grid"), getattr(a, "ace_max_width", None),
           getattr(a, "register", "release"), transpile_caps(a))
    if key in _ENGINE_GB:
        return _ENGINE_GB[key]
    amp, gib = amp_bytes(), 2 ** 30
    full = 2.0 ** n * amp / gib
    if a.backend != "ace":
        gb = full
    elif getattr(a, "ace_max_width", None) or "auto" in (str(a.lrc), str(a.lrr)):
        gb = min(full, 4 * 2.0 ** (getattr(a, "ace_max_width", None) or 33) * amp / gib)
    else:
        size = ace_size(lay, n, a)
        row = layout_row(size, int(a.lrc), int(a.lrr), bool(getattr(a, "ace_torus", False)), ace_opts(a))
        gb = min(active_dense_gb(lay, n, a, row, size), 2 * full)
    _ENGINE_GB[key] = gb
    return gb


def active_dense_gb(lay, n, a, row, size):
    """Dense worst case of the ACE simulators counting only the sites the circuit uses:
    replicas of unused seam sites (and unused bulk sites) stay in |0>, separable, and are
    never allocated by QUnit. Each simulator keeps its non-site qubits (error-detection
    ancilla, repetition code) as in the full register."""
    idx = place(lay, n, a, row, size)
    used, total = collections.Counter(), collections.Counter()
    for q, entries in enumerate(row["unpack"]):
        for sid, _ in entries:
            total[sid] += 1
            if q in set(idx):
                used[sid] += 1
    extra = (1 if bool(getattr(a, "ace_error_detection", True)) else 0) + \
            (1 if bool(getattr(a, "ace_boundary_rep", False)) else 0)
    widths = [used[sid] + extra for sid in total if used[sid]]
    return dense_gb(widths) if widths else 0.0


def active_widths(lay, n, a, row, size):
    """Widths of the ACE simulators that hold circuit qubits (+ ancilla), as active_dense_gb
    counts them; the crossbar simulator is left out (it holds seam replicas only)."""
    idx = set(place(lay, n, a, row, size))
    used = collections.Counter(sid for q, entries in enumerate(row["unpack"]) if q in idx for sid, _ in entries)
    patches = {next(iter({sid for sid, _ in e})) for e in row["unpack"] if len(e) == 1}
    extra = (1 if bool(getattr(a, "ace_error_detection", True)) else 0) + \
            (1 if bool(getattr(a, "ace_boundary_rep", False)) else 0)
    return [used[p] + extra for p in sorted(patches) if used[p]]


def point_mem_gb(lay, a, key, variants=None):
    """Estimated peak resident memory of one point, in GiB: 1 GiB of process overhead, the
    engine(s), and for fxeb the exact reference (state + probability buffer, 1.5 x 2^n
    amplitudes). Patched points at 61 qubits keep their patches apart (QUnit), so only
    their largest patch counts."""
    n, fam, K, d, j, i = key
    amp, gib = amp_bytes(), 2 ** 30
    if getattr(a, "mem_per_point_gb", None):          # measured peak, given by the user (shallow points)
        return float(a.mem_per_point_gb)
    if fam == "patched":
        return 1.0 + 2 * 2.0 ** max(len(p) for p in lay.partitions[K][j]["patches"]) * amp / gib
    engines = sum(engine_gb(lay, n, v) for v in (variants or [a]))
    ref = 1.5 * 2.0 ** n * amp / gib if fam == "fxeb" else 0.0
    return 1.0 + engines + ref


class MemLedger:
    """Machine-wide memory reservations: one file per worker pid holding the GiB its
    current point needs, in a directory every run on this machine shares (default
    /tmp/nighthawk_mem_ledger, or --mem-ledger / $NIGHTHAWK_MEM_LEDGER). A point is
    taken only if (a) all reservations plus it stay within --mem-budget-gb and (b) it fits
    in MemAvailable after the part of other reservations not yet allocated -- so
    processes outside the ledger (an older run, the OS) are respected through (b).
    Dead workers' reservations are dropped automatically."""

    def __init__(self, budget_gb, path=None, swap_gb=None):
        self.budget = float(budget_gb)
        self.swap_gb = swap_gb                      # --swap-gb: free swap counts as available, up to this
        self.dir = Path(path or os.environ.get("NIGHTHAWK_MEM_LEDGER") or "/tmp/nighthawk_mem_ledger")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mine = self.dir / str(os.getpid())
        self.noted = False

    def _entries(self):
        out = {}
        for f in self.dir.iterdir():
            if not f.name.isdigit():
                continue
            pid = int(f.name)
            if not _pid_alive(pid):
                f.unlink(missing_ok=True)
                continue
            try:
                out[pid] = float(f.read_text())
            except (OSError, ValueError):
                pass
        return out

    def reserve(self, gb):
        import fcntl
        with open(self.dir / ".lock", "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                ent = self._entries()
                ent.pop(os.getpid(), None)
                pending = sum(max(0.0, g - rss_gb_of(pid)) for pid, g in ent.items())
                avail = host_available_gb(self.swap_gb)
                if sum(ent.values()) + gb > self.budget:
                    return False
                if avail is not None and gb + pending > 0.95 * avail:
                    return False
                _write_atomic(self.mine, f"{gb:.3f}")
                return True
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def release(self):
        self.mine.unlink(missing_ok=True)


def make_ledger(a):
    if not getattr(a, "mem_budget_gb", 0):
        return None
    import atexit
    led = MemLedger(a.mem_budget_gb, getattr(a, "mem_ledger", None), getattr(a, "swap_gb", None))
    atexit.register(led.release)
    return led


def show_status(a, jobs, recs):
    """--status: every point of this configuration -- done (with F), running (worker pid,
    progress, ETA from the worker's progress file), failed, or pending. Reads files only;
    safe while workers run."""
    prog = {}
    pdir = progress_dir(a)
    if pdir.exists():
        for f in pdir.glob("*.json"):
            try:
                st = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if _pid_alive(int(st.get("pid", 0))):
                prog[st["point"]] = st
    rows, counts = [], collections.Counter()
    for k in sorted(jobs, key=lambda k: (k[0], k[1], k[3], k[2], k[4], k[5])):
        n, fam, K, d, j, i = k
        label = f"n{n} {fam} d{d} i{i}" if fam == "mirror" else f"n{n} {fam} K{K} d{d} p{j} i{i}"
        if k in recs:
            r = recs[k]
            state, info = "done", (f"F = {r['fidelity']:.4g}" if isinstance(r, dict) and "fidelity" in r else "")
        else:
            try:
                t = claim_file(a, k).read_text().strip()
            except OSError:
                t = ""
            pid_s = t.split(":")[0]
            if t.startswith("failed"):
                state, info = "failed", t
            elif t == "done":
                state, info = "done", "(record in a worker file)"
            elif pid_s.isdigit() and _pid_alive(int(pid_s)):
                st = prog.get(label)
                state = "running"
                info = f"pid {pid_s}" + (f", {progress_line(st)}" if st else ", no progress file yet")
            else:
                state, info = "pending", ""
        counts[state] += 1
        rows.append(f"  {label:<26} {state:<8} {info}")
    print(f"# status {cfg_tag(a)} ({a.out}): " + ", ".join(f"{counts[s]} {s}" for s in
                                                          ("done", "running", "failed", "pending")))
    print("\n".join(rows))
    if counts["running"] and not prog:
        print("# no progress files: these workers were started by a version without progress reporting")


def claim_peek(cf):
    """False if this point is done, failed, or held by a live worker."""
    try:
        t = cf.read_text().strip()
    except OSError:
        return True
    if t == "done" or t.startswith("failed"):
        return False
    pid_s = t.partition(":")[0]
    return not (pid_s.isdigit() and _pid_alive(int(pid_s)))


def next_point(a, remaining, cur, est, ledger):
    """Claim the next point: the current size first, else queue order (longest first).
    With a ledger only points whose memory estimate fits are taken -- a big point that does
    not fit right now is passed over for smaller ones and retried later; if nothing fits,
    the worker waits. Returns (key, claim file) or None when nothing is left for it."""
    while remaining:
        waiting = False
        for key in sorted(remaining, key=lambda k: k[0] != cur):
            cf = claim_file(a, key)
            if ledger is None:
                remaining.remove(key)
                if try_claim(cf, a.max_attempts):
                    return key, cf
                continue
            if not claim_peek(cf):
                remaining.remove(key)
                continue
            need = est(key)
            if need > ledger.budget:
                remaining.remove(key)
                if try_claim(cf, a.max_attempts):
                    _write_atomic(cf, f"failed: needs ~{need:.0f} GiB, above --mem-budget-gb {ledger.budget:g}")
                    print(f"# n = {key[0]} {key[1]} d{key[3]}: needs ~{need:.0f} GiB > budget; marked failed",
                          flush=True)
                continue
            if not ledger.reserve(need):
                waiting = True
                continue
            remaining.remove(key)
            if try_claim(cf, a.max_attempts):
                ledger.noted = False
                return key, cf
            ledger.release()
        if waiting and remaining:
            if not ledger.noted:
                print(f"# waiting for memory (budget {ledger.budget:g} GiB, available "
                      f"{host_available_gb(ledger.swap_gb) or 0:.0f} GiB"
                      + (" incl. swap" if ledger.swap_gb else "") + ")", flush=True)
                ledger.noted = True
            time.sleep(15)
    return None


def mem_preflight(lay, a, jobs, variants=None):
    if not getattr(a, "mem_budget_gb", 0):
        return
    per = {}
    for k in jobs:
        per.setdefault(k[0], point_mem_gb(lay, a, k, variants))
    print(f"# memory budget {a.mem_budget_gb:g} GiB (ledger {getattr(a, 'mem_ledger', None) or '/tmp/nighthawk_mem_ledger'}); "
          f"per-point estimate: " + ", ".join(f"n={n} {g:.1f}" for n, g in sorted(per.items())) + " GiB", flush=True)
    over = [n for n, g in per.items() if g > a.mem_budget_gb]
    if over:
        print(f"# WARNING: sizes {sorted(over)} exceed the budget and will be marked failed", flush=True)


def check_transpile_args(a):
    if getattr(a, "ace_transpile", False):
        if getattr(a, "ace_tiling", False):
            raise SystemExit("give --ace-transpile or --ace-tiling, not both")
        if getattr(a, "ace_max_width", None) or "auto" in (str(a.lrc), str(a.lrr)) or \
                getattr(a, "geometry", "manual") == "nnqab":
            raise SystemExit("--ace-transpile needs explicit --lrc/--lrr")
    elif getattr(a, "ace_width", None) or getattr(a, "ace_exclude", None) or getattr(a, "transpile_allow_cross", False):
        raise SystemExit("--ace-width, --ace-exclude and --transpile-allow-cross apply to --ace-transpile placements only")


def parse_variant(spec, a):
    """'4/4:t3', '4/4:t2', '4/3:tiled', '2/7:torus:t3', '4/4:untiled', '4/4:t3:rep' -> run namespace.
    ACE options: rep = boundary repetition code on, noed = error detection off,
    noxbar = crossbars off (Qrack defaults: off, on, on)."""
    import copy
    toks = spec.split(":")
    try:
        lrc, lrr = (int(x) for x in toks[0].split("/"))
    except ValueError:
        raise SystemExit(f"--variants {spec}: start with lrc/lrr, e.g. 4/4:t3")
    v = copy.copy(a)
    v.lrc, v.lrr, v.ace_torus, v.ace_tiling = lrc, lrr, False, False
    v.ace_tiling_version, v.ace_max_width, v.geometry, v.variants = TILING_VERSION, None, "manual", None
    v.ace_boundary_rep, v.ace_error_detection, v.ace_crossbars = False, True, True
    v.ace_transpile, v.ace_width, v.ace_exclude, v.transpile_engine = False, None, None, "exact"
    v.transpile_allow_cross = False
    for t in toks[1:]:
        if t == "torus":
            v.ace_torus = True
        elif t in ("untiled", "grid"):
            v.ace_tiling = False
        elif t == "tiled":
            v.ace_tiling = True
        elif t == "rep":
            v.ace_boundary_rep = True               # is_boundary_repetition_code=True
        elif t == "noed":
            v.ace_error_detection = False           # is_error_detection=False
        elif t == "noxbar":
            v.ace_crossbars = False                 # use_crossbars=False
        elif t in ("tx", "txa", "vf2"):
            v.ace_transpile, v.ace_tiling = True, False   # register + coupling-map placement
            v.transpile_engine = {"tx": "exact", "txa": "anneal", "vf2": "vf2"}[t]
        elif t == "xok":
            v.transpile_allow_cross = True          # allow couplers between patches with no seam (with tx)
        elif re.fullmatch(r"w\d+", t):
            v.ace_width = int(t[1:])                # ACE register of this many sites (with tx)
        elif re.fullmatch(r"ex[\d+]+", t):
            v.ace_exclude = t[2:]                   # sites excluded from the coupling map (with tx)
        elif re.fullmatch(r"t\d+", t):
            v.ace_tiling, v.ace_tiling_version = True, int(t[1:])
        else:
            raise SystemExit(f"--variants {spec}: unknown token {t!r} "
                             f"(torus, tiled, untiled, t2, t3, rep, noed, noxbar, tx, txa, vf2, xok, w<N>, ex<a+b+...>)")
    v.ace_torus_search = "torus" if v.ace_torus else "flat"
    check_transpile_args(v)
    return v


def cmd_run_group(a, lay):
    """--variants: every fxeb point is sampled by each listed ACE configuration and all of
    them are scored against one exact reference. Records and bitstrings go under each
    variant's own config tag, exactly as separate runs would write them (so --summarize
    with that variant's flags, the graph tool and resuming all work per variant); only
    the claims are shared, under a group tag, one per point."""
    if a.backend != "ace" or a.families != "fxeb":
        raise SystemExit("--variants scores several ACE configurations against one exact reference: "
                         "use it with --backend ace --families fxeb")
    if a.ace_max_width or a.geometry == "nnqab" or a.ace_tiling or a.ace_torus or a.ace_transpile \
            or a.ace_width or a.ace_exclude:
        raise SystemExit("--variants sets the layout per variant: drop --ace-max-width, --geometry, "
                         "--ace-tiling, --ace-torus, --ace-transpile, --ace-width and --ace-exclude "
                         "(use the tokens tx, w<N>, ex<a+b+...>)")
    vas = [parse_variant(sp, a) for sp in a.variants]
    tags = [cfg_tag(v) for v in vas]
    if len(set(tags)) != len(tags):
        raise SystemExit("two --variants describe the same configuration")
    a.claim_tag = "grp-" + hashlib.sha1("|".join(sorted(tags)).encode()).hexdigest()[:8]
    names = [f"{sp} [{t}]" for sp, t in zip(a.variants, tags)]
    if not (a.summarize or a.pack):
        check_ace_opts(list(zip(a.variants, vas)))

    def done_keys():
        rr = [set(load_jsonl(a.out, t, lay.n)) for t in tags]
        return set.intersection(*rr)

    def finish():
        for v, t, nm in zip(vas, tags, names):
            print(f"\n######## variant {nm}")
            recs_v = load_jsonl(a.out, t, lay.n)
            pack_shots(recs_v, lay, v)
            summarize(recs_v, lay, v)

    if a.summarize or a.pack:
        for v, t, nm in zip(vas, tags, names):
            print(f"\n######## variant {nm}")
            recs_v = load_jsonl(a.out, t, lay.n)
            if a.pack:
                pack_shots(recs_v, lay, v)
            if a.summarize:
                summarize(recs_v, lay, v)
        return
    if a.sizes and a.n:
        raise SystemExit("give --sizes or --n, not both")
    sizes = parse_sizes(a.sizes) if a.sizes else [a.n or lay.n]
    if max(sizes) > 36 or min(sizes) < 2:
        raise SystemExit("fxeb needs an exact 2^n reference: sizes 2..~34-36")
    jobs = build_jobs(lay, a, sizes, ["fxeb"])
    done = done_keys()
    todo = [k for k in jobs if k not in done]
    if a.retry_failed and not a.worker:
        cleared = 0
        for k in todo:
            cf = claim_file(a, k)
            try:
                if cf.read_text().strip().startswith("failed"):
                    os.rename(cf, cf.with_name(cf.name + f".stale{os.getpid()}"))
                    cleared += 1
            except OSError:
                pass
        print(f"# --retry-failed: {cleared} failed point(s) cleared for another attempt", flush=True)
    if not a.worker:
        print(f"# variants group {a.claim_tag}: {len(vas)} ACE configurations, one exact reference per point")
        for v, nm in zip(vas, names):
            print(f"#   {nm}", flush=True)
            if v.ace_tiling or getattr(v, "ace_transpile", False):
                ace_preflight(lay, v, sizes)
        print(f"# {len(jobs) - len(todo)} of {len(jobs)} points done for every variant", flush=True)
        mem_preflight(lay, a, jobs, vas)
        if a.gpus:
            return launch(a, lay, None, jobs, done, done_keys, finish) if todo else finish()
    out = Path(a.out)
    fh = open(out.with_name(f"{out.stem}.w{os.getpid()}.jsonl") if a.worker else out, "a")
    _GATE["a"] = a
    if a.max_rss_gb:
        start_rss_watchdog(a.max_rss_gb)
    ledger = make_ledger(a)
    engines, cur, fails, remaining = {}, None, 0, list(todo)
    while remaining:
        if ledger:
            ledger.release()
        got = next_point(a, remaining, cur, lambda k: point_mem_gb(lay, a, k, vas), ledger)
        if got is None:
            break
        key, cf = got
        n, fam, K, d, j, i = key
        need = [vi for vi, t in enumerate(tags) if key not in load_jsonl(a.out, t, lay.n)]
        if not need:
            mark_done(cf)
            continue
        _WATCH["claim"] = cf
        try:
            if n != cur:
                engines.clear()
                cur = n
            for vi in need:
                if vi not in engines:
                    engines[vi] = make_engine(lay, n, vas[vi])
            t0 = time.time()
            res = run_fxeb_group(lay, [engines[vi] for vi in need], d, i, a)
        except Exception as err:
            print(f"# n = {n} fxeb d{d} inst {i}: failed ({err}); marked failed", flush=True)
            _write_atomic(cf, f"failed: {type(err).__name__}: {err}"[:500])
            engines.clear()
            cur = None
            fails += 1
            if a.worker and fails >= 3:
                raise SystemExit(f"# worker stops after {fails} failures in a row")
            continue
        fails = 0
        for vi, r in zip(need, res):
            arrays = r.pop("_shots", None)
            if arrays:
                f = point_file(vas[vi], n, fam, K, d, j, i)
                save_point(f, arrays)
                r["shots_file"] = str(f)
            rec = dict(cfg=tags[vi], n=n, family=fam, K=K, depth=d, partition=j, instance=i, theta=a.theta,
                       **({"protocol": a.protocol} if getattr(a, "protocol", "nighthawk") != "nighthawk" else {}),
                       seconds=round(r["ace_seconds"] + r["ref_seconds"] / len(need), 2),
                       **engines[vi].geometry(), **r)
            fh.write(json.dumps(rec) + "\n")
            print(f"n {n:>2} fxeb d{d:>3} inst {i}  {a.variants[vi]:>14s}  F {r['fidelity']:.5f} +/- {r['se']:.1e}"
                  f"  (ace {r['ace_seconds']}s, ref {r['ref_seconds']}s shared by {len(need)})", flush=True)
        fh.flush()
        os.fsync(fh.fileno())
        mark_done(cf)
    if ledger:
        ledger.release()
    if not a.worker:
        finish()


def cmd_run(a):
    if getattr(a, "protocol", "nighthawk") == "nnqab":
        if any(f not in ("fxeb", "hamming") for f in a.families.split(",")):
            raise SystemExit("--protocol nnqab runs Dan Strano's circuits: --families fxeb (nn_qab.py's XEB, "
                             "n <= 36) and/or hamming (his mirror Hamming weight, any n)")
        if a.theta != "nnqab":
            print("# --protocol nnqab: single-qubit ensemble set to nnqab (as nn_qab.py)", flush=True)
            a.theta = "nnqab"
    PROTOCOL["mode"] = getattr(a, "protocol", "nighthawk")
    THETA_DIST["mode"] = a.theta
    lay = make_layout(a)
    if not is_release(lay):
        if a.variants:
            raise SystemExit("--register runs one configuration: drop --variants")
        fams_ = a.families.split(",")
        if a.families == "mirror,patched":           # the default: patched circuits need exact patch references
            a.families = "mirror"
        elif any(f not in ("mirror", "fxeb", "hamming", "vqe") for f in fams_) or ("fxeb" in fams_ and lay.n > 36):
            raise SystemExit(f"--register {layout_id(lay)}: only --families mirror, hamming, vqe"
                             + (", fxeb" if lay.n <= 36 else f" (no exact reference for fxeb at {lay.n} qubits)")
                             + " (registers have no device data)")
        if apply_register_preset(a, lay) and not a.worker and not (a.summarize or a.pack or getattr(a, "status", False)):
            ace = REGISTERS[layout_id(lay)]["ace"]
            print(f"# register {layout_id(lay)}: {REGISTERS[layout_id(lay)]['about']}", flush=True)
            print(f"# ACE preset: {ace['width']} sites, lrc {ace['lrc']} lrr {ace['lrr']}"
                  f"{' torus' if ace['torus'] else ''}, exact placement (stored), patch widths <= {ace['max_width']} "
                  f"(at most {ace['max_wide']} at {ace['max_width']}, incl. the error-detection ancilla)", flush=True)
    if a.variants:
        return cmd_run_group(a, lay)
    if a.backend == "ace" and not (a.summarize or a.pack):
        check_ace_opts([("this run", a)])
    geo = resolve_geometry(lay, a)
    if geo and not a.worker:
        print(f"# nnqab geometry: lrc={a.lrc} lrr={a.lrr} torus, {geo['boundary']} boundary qubits, "
              f"patches {geo['patches']}, bulk-to-boundary {geo['b_to_b']:.2f}")
    tag = cfg_tag(a)
    recs = load_jsonl(a.out, tag, lay.n)
    if a.summarize or a.pack:
        if a.pack:
            pack_shots(recs, lay, a)
        if a.summarize:
            summarize(recs, lay, a)
        return
    if a.sizes and a.n:
        raise SystemExit("give --sizes or --n, not both")
    if a.ace_max_width and (a.geometry == "nnqab" or "auto" in (str(a.lrc), str(a.lrr))):
        raise SystemExit("--ace-max-width picks lrc/lrr itself: drop --geometry nnqab and --lrc/--lrr auto")
    if a.ace_torus and a.ace_torus_search != "flat":
        raise SystemExit("give --ace-torus or --ace-torus-search, not both")
    if a.ace_torus:
        a.ace_torus_search = "torus"
    if a.ace_tiling and ("auto" in (str(a.lrc), str(a.lrr)) or a.geometry == "nnqab"):
        raise SystemExit("--ace-tiling needs explicit --lrc/--lrr or --ace-max-width")
    check_transpile_args(a)
    if a.theta != "haar" and not a.worker:
        print(f"# VARIANT: single-qubit theta drawn as {a.theta}, not Haar -- these are not the paper's "
              f"circuits (records kept under their own config tag)", flush=True)
    sizes = parse_sizes(a.sizes) if a.sizes else [a.n or lay.n]
    if not all(2 <= n <= lay.n for n in sizes):
        raise SystemExit(f"sizes must lie in 2..{lay.n}")
    fams = a.families.split(",")
    if "fxeb" in fams and max(sizes) > 36:
        raise SystemExit("fxeb needs an exact 2^n reference: sizes up to ~34-36 (memory), not 61")
    jobs = build_jobs(lay, a, sizes, fams)
    check_shot_budgets(a, jobs)
    if getattr(a, "status", False):
        return show_status(a, jobs, recs)
    if not a.worker:
        ace_preflight(lay, a, sizes)
    todo = [k for k in jobs if k not in recs]
    if not a.worker and not getattr(a, "summarize", False):
        reopened = 0
        for k in todo:                              # 'done' but no usable record: the record was NaN
            cf = claim_file(a, k)
            try:
                if cf.read_text().strip() == "done":
                    os.rename(cf, cf.with_name(cf.name + f".stale{os.getpid()}"))
                    reopened += 1
            except OSError:
                pass
        if reopened:
            print(f"# {reopened} point(s) marked done had no usable record (non-finite score, or moved aside): reopened", flush=True)
    if _BAD_RECORDS and not a.worker:
        print(f"# {len(_BAD_RECORDS)} record(s) with a non-finite score ignored (rerun to replace them)", flush=True)
    if a.retry_failed and not a.worker:            # once, here; workers never retry failed points
        cleared = 0
        for k in todo:
            cf = claim_file(a, k)
            try:
                if cf.read_text().strip().startswith("failed"):
                    os.rename(cf, cf.with_name(cf.name + f".stale{os.getpid()}"))
                    cleared += 1
            except OSError:
                pass
        print(f"# --retry-failed: {cleared} failed point(s) cleared for another attempt", flush=True)
    if not a.worker:
        if a.backend == "exact" and a.exact_probs and "mirror" in fams:
            print("# note: exact mirror reads survival with prob_perm, so it draws no bitstrings; "
                  "--no-exact-probs samples (and stores) them")
        if "patched" in fams and any(n != lay.n for n in sizes):
            print("# note: patched circuits use the released 61-qubit partitions; run only at n = 61")
        print(f"# cfg {tag}: backend {a.backend}, sizes {sizes}, shots {a.shots}, "
              f"twirls {a.twirls if a.backend == 'ace' else 0}"
              f"{', lrc/lrr ' + str(a.lrc) + '/' + str(a.lrr) if a.backend == 'ace' else ''}"
              f"{(', is_torus ' + str(bool(getattr(a, 'ace_torus', False)))) if a.backend == 'ace' and not a.ace_max_width else ''}"
              f"{(', max width ' + str(a.ace_max_width) + ' (layout per size above)') if a.backend == 'ace' and a.ace_max_width else ''}")
        print(f"# {len(jobs) - len(todo)} of {len(jobs)} points already done", flush=True)
        mem_preflight(lay, a, jobs)
        if a.gpus:
            if todo:
                return launch(a, lay, tag, jobs, recs)
            return summarize(recs, lay, a)
    out = Path(a.out)
    out_file = out.with_name(f"{out.stem}.w{os.getpid()}.jsonl") if a.worker else out
    fh = open(out_file, "a")
    _GATE["a"] = a
    if a.max_rss_gb:
        start_rss_watchdog(a.max_rss_gb)
    ledger = make_ledger(a)
    engines, fails, remaining = {}, 0, list(todo)
    while remaining:
        if ledger:
            ledger.release()                        # the previous point's reservation
        cur = next(iter(engines), None)          # stay on the current size: no re-initialisation
        got = next_point(a, remaining, cur, lambda k: point_mem_gb(lay, a, k), ledger)
        if got is None:
            break
        key, cf = got
        n, fam, K, d, j, i = key
        if key in load_jsonl(a.out, tag, lay.n):   # finished by another worker since we started
            mark_done(cf)
            continue
        _WATCH["claim"] = cf
        if n not in engines:
            engines.clear()                         # one engine alive per worker
            try:
                engines[n] = make_engine(lay, n, a)
            except Exception as err:                # e.g. exact backend beyond QRACK_MAX_CPU_QB
                print(f"# n = {n}: cannot build the {a.backend} engine ({err}); skipped", flush=True)
                cf.unlink(missing_ok=True)
                fails += 1
                if a.worker and fails >= 3:
                    raise SystemExit(f"# worker stops after {fails} failures in a row (GPU state?)")
                continue
        eng = engines[n]
        t0 = time.time()
        try:
            r = (run_mirror(lay, eng, d, i, a) if fam == "mirror" else
                 run_fxeb(lay, eng, d, i, a) if fam == "fxeb" else
                 run_hamming(lay, eng, d, K, i, a) if fam == "hamming" else
                 run_vqe(lay, eng, d, a) if fam == "vqe" else
                 run_full(lay, eng, d, j, a) if fam == "full" else run_patched(lay, eng, K, d, j, i, a))
        except Exception as err:                    # deterministic (e.g. QRACK_MAX_CPU_QB): do not retry
            import traceback
            print(f"# n = {n} {fam} d{d} inst {i}: failed ({err}); marked failed", flush=True)
            traceback.print_exc(file=sys.stdout)        # where it failed, for the worker log
            sys.stdout.flush()
            _write_atomic(cf, f"failed: {type(err).__name__}: {err}"[:500])
            engines.clear()                         # do not reuse a backend that failed mid-run
            fails += 1
            if a.worker and fails >= 3:
                raise SystemExit(f"# worker stops after {fails} failures in a row (GPU state?)")
            continue
        fails = 0
        arrays = r.pop("_shots", None)
        if arrays:                                  # bitstrings first, record second, claim last
            f = point_file(a, n, fam, K, d, j, i)
            save_point(f, arrays)
            r["shots_file"] = str(f)
        rec = dict(cfg=tag, n=n, family=fam, K=K, depth=d, partition=j, instance=i, theta=a.theta,
                       **({"protocol": a.protocol} if getattr(a, "protocol", "nighthawk") != "nighthawk" else {}),
                   seconds=round(time.time() - t0, 2),
                   device="gpu" if (not a.cpu and getattr(a, "ace_gpu", False) and a.backend == "ace") else "cpu",
                   **eng.geometry(), **r)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
        mark_done(cf)
        recs[key] = rec
        score = f"F {r['fidelity']:.5f} +/- {r['se']:.1e}" if r["fidelity"] is not None else f"{r['shots']:,} samples"
        print(f"n {n:>2} {fam:7s} K{K} d{d:>3} part {j} inst {i}  {score}  ({rec['seconds']}s)", flush=True)
    if ledger:
        ledger.release()
    if a.worker:
        return
    recs = load_jsonl(a.out, tag, lay.n)
    pack_shots(recs, lay, a)
    summarize(recs, lay, a)


# ================================================================== aceplan
def cmd_aceplan(a):
    lay = Layout(a.repo)
    n = a.n or lay.n
    size, R, C = ace_register(lay, n, a.ace_layout)
    toruses = {"any": (False, True), "false": (False,), "true": (True,)}[a.torus]
    cp = circuit_couplers(lay, n, a.ace_layout)
    tiling = (lay, n, [r_ * C + c_ for r_, c_ in lay.pos[:n]]) if a.tiling else None
    if tiling:
        print(f"# tiling every layout (annealing, ~1-2 s each)...", flush=True)
    rows = sorted(with_seams(ace_plan(size, toruses), cp, tiling), key=ace_rank)
    pick = widest_ace_config(size, a.max_width, cp, None if a.torus == "any" else a.torus == "true", tiling) \
        if a.max_width else None
    print(f"ACE layouts of the {R}x{C} register ({size} sites) holding the first {n} logical qubits "
          f"({len(cp)} circuit couplers); {amp_bytes()} bytes/amplitude; ranked by couplers across simulators"
          + (", after tiling" if tiling else ", device-grid placement"))
    print("torus  lrc lrr  cross CZ  seam CZ  seam qb  sims  widest  dense GB (worst)  simulator widths")
    for r in rows[:a.top] if not a.max_width else rows:
        if a.max_width and r["max_width"] > a.max_width and not a.all:
            continue
        mark = "  <- pick" if pick is not None and (r["lrc"], r["lrr"], r["torus"]) == \
            (pick["lrc"], pick["lrr"], pick["torus"]) else ""
        print(f"{str(r['torus']):5s}  {r['lrc']:>3} {r['lrr']:>3}  {r['cross_cz']:>8}  {r['seam_cz']:>7}  {r['boundary']:>7}  "
              f"{r['sims']:>4}  {r['max_width']:>6}  {dense_gb(r['widths']):>16,.1f}  {r['widths']}{mark}")
    ram = mem_total_gb()
    if ram:
        print(f"(this machine: {ram:,.0f} GB RAM)")


# ================================================================== layoutsearch
# Joint search for a register and its ACE placement: which sites of the 12 x 10 Nighthawk
# lattice to use (n of them), and which ACE group (a patch's bulk, or a seam) each one goes to,
# in one CP-SAT model. Candidates are verified, registered and scored by short mirror runs.

LS_OBJECTIVES = ("seam", "bcz", "cz", "couplers", "balance")


def release_prefix_spec(L, N, C):
    """REGISTERS entry for the release's first N logical qubits: the bounding window of those
    sites, with every other window site (the broken qubits among them) dropped. At N = 61 it is
    the release itself; below, RegisterLayout builds that register's own circuits by the
    release's recipe (its own K = 3, 4 partitions)."""
    l2p = L["logical_to_physical"]
    if not 0 < N <= len(l2p):
        raise SystemExit(f"release:N needs 1 <= N <= {len(l2p)}")
    cells = {(p // C, p % C) for p in l2p[:N]}
    r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
    c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
    drop = [r * C + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in cells]
    return dict(n=N, rows=[r0, r1], cols=[c0, c1], drop=drop,
                about=f"the release's first {N} logical qubits")


def mirror_cz_weights(lay, depths, C):
    """{frozenset({cell_u, cell_v}): CZs the register's mirror circuits apply on that coupler},
    summed over the forward halves of `depths` (the inverse half repeats them). The CZ pattern
    (colour schedule plus the rotating pseudo-patch cuts) does not depend on the instance."""
    w = defaultdict(int)
    cell = [(p // C, p % C) for p in lay.l2p]
    for d in depths:
        for _, es in mirror_cycles(lay, d, 0):
            for u, v in es:
                w[frozenset((cell[u], cell[v]))] += 1
    return dict(w)


def ace_groups(row, edges):
    """Sites grouped by the set of simulators that hold them (as _exact_on_register), the
    patches (simulators that own a bulk group), and which group pairs ACE couples."""
    sids = [frozenset(sid for sid, _ in u) for u in row["unpack"]]
    groups = {}
    for i, s_ in enumerate(sids):
        groups.setdefault(s_, []).append(i)
    G = sorted(groups, key=lambda g: (len(g), sorted(g)))
    patch = sorted({next(iter(g)) for g in G if len(g) == 1})
    eset = {tuple(sorted(e)) for e in edges}
    ok = {}
    for a_ in range(len(G)):
        for b_ in range(a_, len(G)):
            ga, gb = groups[G[a_]], groups[G[b_]]
            ok[a_, b_] = ok[b_, a_] = any((min(i, j), max(i, j)) in eset for i in ga for j in gb if i != j)
    return G, groups, patch, ok


def lattice_info(repo=None):
    L = json.loads((find_repo(repo) / "data/layout.json").read_text())
    R, C = L["device_lattice"]["rows"], L["device_lattice"]["cols"]
    return R, C, {(p // C, p % C) for p in L["dropped_physical_qubits"]}, L


def search_layout(row, edges, n, *, R=12, C=10, broken=frozenset(), broken_policy="disable",
                  region="convex", window=None, fixed=None, max_width=None, max_wide=None,
                  objective=("seam", "bcz", "couplers"), nogoods=(), diversity=4, hint=None,
                  time_s=120.0, workers=None, seed=0, trim=0, min_boundary=0, cz_weight=None):
    """One CP-SAT solve of the joint model. region: rect (a rectangle of the lattice, chosen by
    the solver; holes only at broken sites, plus up to `trim` sites taken off its border),
    convex (every row and column of the region is one unbroken run; holes only at broken
    sites), free (any connected set), fixed (exactly the cells in `fixed`). nogoods: earlier solutions' seam-cell sets; a new solution differs
    from each in at least `diversity` seam cells. hint: {cell: group key} warm start.
    Returns a dict (cells, groups per cell, metrics, solver status) or None."""
    from ortools.sat.python import cp_model
    G, groups, patch, ok = ace_groups(row, edges)
    r0, r1, c0, c1 = window or (0, R - 1, 0, C - 1)
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    idx = {p: i for i, p in enumerate(cells)}
    V, NG = len(cells), len(G)
    E = [(idx[(r, c)], idx[(r + dr, c + dc)]) for (r, c) in cells for dr, dc in ((0, 1), (1, 0))
         if (r + dr, c + dc) in idx]
    seamg = [g for g in range(NG) if len(G[g]) > 1]
    m = cp_model.CpModel()
    reg = [m.NewBoolVar("") for _ in range(V)]
    inc = [m.NewBoolVar("") for _ in range(V)]
    for i, p in enumerate(cells):
        if p in broken and broken_policy == "disable":
            m.Add(inc[i] == 0)
        elif region != "rect":
            m.Add(inc[i] == reg[i])
        if region == "fixed":
            m.Add(inc[i] == int(p in fixed))
    m.Add(sum(inc) == n)
    if region == "rect":
        rows_, cols_ = list(range(r0, r1 + 1)), list(range(c0, c1 + 1))
        inr = {r: m.NewBoolVar("") for r in rows_}
        inc_c = {c: m.NewBoolVar("") for c in cols_}
        for line in (rows_, cols_):
            var = inr if line is rows_ else inc_c
            for a_ in range(len(line)):
                for b_ in range(a_ + 2, len(line)):
                    for mid in range(a_ + 1, b_):
                        m.AddBoolOr([var[line[a_]].Not(), var[line[b_]].Not(), var[line[mid]]])
        edge_r = {r: m.NewBoolVar("") for r in rows_}     # first or last row of the rectangle
        edge_c = {c: m.NewBoolVar("") for c in cols_}
        for var, ed, line in ((inr, edge_r, rows_), (inc_c, edge_c, cols_)):
            for k, r in enumerate(line):
                outs = [var[line[k - 1]].Not()] if k > 0 else []
                outs2 = [var[line[k + 1]].Not()] if k + 1 < len(line) else []
                # ed <=> var and (previous out or next out); at the window edge the line is a border
                if k == 0 or k + 1 == len(line):
                    m.Add(ed[r] == var[r])
                else:
                    m.AddBoolOr([outs[0], outs2[0]]).OnlyEnforceIf(ed[r])
                    m.AddImplication(ed[r], var[r])
                    m.AddBoolOr([var[r].Not(), var[line[k - 1]], var[line[k + 1]], ed[r]])
                    m.AddBoolOr([var[r].Not(), var[line[k - 1]].Not(), var[line[k + 1]].Not(), ed[r].Not()])
        holes = []
        for i, (r, c) in enumerate(cells):
            m.AddBoolAnd([inr[r], inc_c[c]]).OnlyEnforceIf(reg[i])
            m.AddBoolOr([inr[r].Not(), inc_c[c].Not()]).OnlyEnforceIf(reg[i].Not())
            if not ((r, c) in broken and broken_policy == "disable"):
                h = m.NewBoolVar("")                         # a trimmed site: in the rectangle, unused
                m.Add(inc[i] == reg[i] - h)
                m.AddBoolOr([edge_r[r], edge_c[c]]).OnlyEnforceIf(h)
                holes.append(h)
        m.Add(sum(holes) <= int(trim))
    if region == "convex":
        lines = [[idx[(r, c)] for c in range(c0, c1 + 1)] for r in range(r0, r1 + 1)] + \
                [[idx[(r, c)] for r in range(r0, r1 + 1)] for c in range(c0, c1 + 1)]
        for L_ in lines:
            for a_ in range(len(L_)):
                for b_ in range(a_ + 2, len(L_)):
                    for mid in range(a_ + 1, b_):
                        m.AddBoolOr([reg[L_[a_]].Not(), reg[L_[b_]].Not(), reg[L_[mid]]])
        for i, p in enumerate(cells):                       # a disabled broken site only inside
            if p in broken and broken_policy == "disable":
                r, c = p
                for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if q in idx:
                        m.AddImplication(reg[i], reg[idx[q]])
                    else:
                        m.Add(reg[i] == 0)
    x = {(v, g): m.NewBoolVar("") for v in range(V) for g in range(NG)}
    for v in range(V):
        m.Add(sum(x[v, g] for g in range(NG)) == inc[v])
    for g in range(NG):
        m.Add(sum(x[v, g] for v in range(V)) <= len(groups[G[g]]))
    bad = [(a_, b_) for a_ in range(NG) for b_ in range(NG) if not ok[a_, b_]]
    for u, v in E:
        for a_, b_ in bad:
            m.AddBoolOr([x[u, a_].Not(), x[v, b_].Not()])
    seam = [sum(x[v, g] for g in seamg) for v in range(V)]
    B = sum(seam)
    if min_boundary:
        m.Add(B >= int(min_boundary))
    used_e, bcz = [], []
    for u, v in E:
        b_ = m.NewBoolVar("")
        m.AddBoolAnd([inc[u], inc[v]]).OnlyEnforceIf(b_)
        m.AddBoolOr([inc[u].Not(), inc[v].Not()]).OnlyEnforceIf(b_.Not())
        used_e.append(b_)
        t_ = m.NewBoolVar("")                               # coupler in use and touching a seam
        m.Add(t_ <= b_)
        m.Add(t_ >= seam[u] + b_ - 1)
        m.Add(t_ >= seam[v] + b_ - 1)
        m.Add(t_ <= seam[u] + seam[v])
        bcz.append(t_)
    czw = [int((cz_weight or {}).get(frozenset((cells[u], cells[v])), 0)) for u, v in E]
    if "cz" in objective and not cz_weight:
        raise SystemExit("objective cz needs circuit weights (--cz-weights)")
    widths = [sum(x[v, g] for v in range(V) for g in range(NG) if p in G[g]) for p in patch]
    if max_width:
        cap = int(max_width) - 1
        for w in widths:
            m.Add(w <= cap)
        if max_wide is not None:
            big = [m.NewBoolVar("") for _ in widths]
            for w, b_ in zip(widths, big):
                m.Add(w <= cap - 1 + b_)
            m.Add(sum(big) <= int(max_wide))
    if region != "fixed":                                   # connected: single-commodity flow
        root = [m.NewBoolVar("") for _ in range(V)]
        m.Add(sum(root) == 1)
        for v in range(V):
            m.AddImplication(root[v], inc[v])
        f = {}
        for u, v in E:
            for a_, b_ in ((u, v), (v, u)):
                f[a_, b_] = m.NewIntVar(0, n, "")
                m.Add(f[a_, b_] <= n * inc[a_])
                m.Add(f[a_, b_] <= n * inc[b_])
        inn, out = defaultdict(list), defaultdict(list)
        for (a_, b_), var in f.items():
            out[a_].append(var)
            inn[b_].append(var)
        for v in range(V):
            m.Add(sum(inn[v]) - sum(out[v]) == inc[v] - n * root[v])
    for S in nogoods:
        Sv = {idx[p] for p in S if p in idx}
        m.Add(sum(1 - seam[v] for v in Sv) + sum(seam[v] for v in range(V) if v not in Sv) >= int(diversity))
    wmax, wmin = m.NewIntVar(0, n, ""), m.NewIntVar(0, n, "")
    m.AddMaxEquality(wmax, widths)
    m.AddMinEquality(wmin, widths)
    terms = dict(seam=B, bcz=sum(bcz), cz=sum(w_ * t_ for w_, t_ in zip(czw, bcz)), couplers=-sum(used_e),
                 balance=wmax - wmin)
    span = {"cz": max(512, 2 * sum(czw) + 1)}               # each term must stay below one step of the previous
    obj, scale = 0, 1
    for name in reversed(objective):
        obj += scale * terms[name]
        scale *= span.get(name, 512)
    m.Minimize(obj)
    if hint:
        key_of = {G[g]: g for g in range(NG)}
        for v, p in enumerate(cells):
            gk = hint.get(p)
            m.AddHint(inc[v], int(gk is not None))
            for g in range(NG):
                m.AddHint(x[v, g], int(gk is not None and key_of.get(gk) == g))
    sol = cp_model.CpSolver()
    sol.parameters.max_time_in_seconds = float(time_s)
    sol.parameters.num_workers = int(workers or max(1, min(16, os.cpu_count() or 1)))
    sol.parameters.random_seed = int(seed)
    st_ = sol.Solve(m)
    if st_ not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None if st_ != cp_model.INFEASIBLE else dict(status="INFEASIBLE")
    gof = {}
    for v, p in enumerate(cells):
        if sol.Value(inc[v]):
            gof[p] = G[next(g for g in range(NG) if sol.Value(x[v, g]))]
    return dict(status=sol.StatusName(st_), objective=sol.ObjectiveValue(), bound=sol.BestObjectiveBound(),
                cells=gof, seam=sum(1 for g in gof.values() if len(g) > 1),
                bcz=int(sum(sol.Value(t_) for t_ in bcz)), couplers=int(sum(sol.Value(b_) for b_ in used_e)),
                cz=int(sum(w_ * sol.Value(t_) for w_, t_ in zip(czw, bcz))),
                widths=[sum(1 for g in gof.values() if p in g) + 1 for p in patch],
                seconds=round(sol.WallTime(), 1))


def register_from_cells(gof, row, edges, cols_dev):
    """REGISTERS entry for a solved layout: the bounding window, the device sites dropped from
    it, and the ACE site of every logical qubit (logical order = row-major, as RegisterLayout)."""
    G, groups, patch, ok = ace_groups(row, edges)
    used = sorted(gof)
    r0, r1 = min(r for r, _ in used), max(r for r, _ in used)
    c0, c1 = min(c for _, c in used), max(c for _, c in used)
    drop = [r * cols_dev + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in gof]
    free = {k: list(v) for k, v in groups.items()}
    site = [free[gof[p]].pop(0) for p in used]
    return dict(n=len(used), rows=[r0, r1], cols=[c0, c1], drop=drop), site


def verify_register(name, spec, repo=None):
    """Independent check of a registered layout through the normal code path: RegisterLayout
    rebuilds the register and its couplers from the release's code; every coupler must share a
    simulator (no 'cross'), the widths must respect the caps, and the counts must match."""
    ace = spec["ace"]
    lay = RegisterLayout(repo, name)
    row = layout_row(ace["width"], ace["lrc"], ace["lrr"], ace["torus"], {})
    cp = logical_couplers(lay, lay.n)
    site = ace["site"]
    st_ = placement_stats(row, site, cp)
    w = collections.Counter(sid for x_ in site for sid, _ in row["unpack"][x_])
    patches = sorted({next(iter(frozenset(s for s, _ in u))) for u in row["unpack"] if len(u) == 1})
    widths = [w[p] + 1 for p in patches]
    problems = []
    if len(set(site)) != len(site):
        problems.append("two logical qubits on one ACE site")
    if st_["cross"]:
        problems.append(f"{st_['cross']} coupler(s) between patches")
    W, N = ace.get("max_width"), ace.get("max_wide")
    if W and (max(widths) > W or (N is not None and sum(1 for x_ in widths if x_ == W) > N)):
        problems.append(f"widths {widths} break the caps {W}/{N}")
    if lay.n != spec["n"] or len(site) != lay.n:
        problems.append("qubit count mismatch")
    return dict(ok=not problems, problems=problems, exact=st_["exact"], replica=st_["replica"],
                cross=st_["cross"], seam=st_["seam_used"], widths=widths, couplers=len(cp),
                partitions_ok=all(len(lay.partitions[K]) > 0 for K in (3, 4)))


def _registry_path():
    return Path(__file__).resolve().parent / "nighthawk_registers.json"


def _register_files():
    files = [_registry_path()]
    env = os.environ.get("NIGHTHAWK_REGISTERS")
    if env:
        files += [Path(p) for p in env.split(os.pathsep) if p]
    return files


def load_register_files():
    """REGISTERS entries from nighthawk_registers.json next to the script and the files in
    $NIGHTHAWK_REGISTERS (os.pathsep-separated). Built-in names are never overridden."""
    for f in _register_files():
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        for k, v in data.items():
            if k not in BUILTIN_REGISTERS:
                REGISTERS[k] = v


def save_register(path, name, entry):
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    data[name] = entry
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def parse_geometries(spec_list, n, max_width, patches, torus_mode, limit):
    """'SIZE:LRC:LRR[:t]' entries, or 'auto': every ACE register of n..n+40 sites with the
    requested patch count whose patches can hold n qubits under the width cap, one per
    distinct group structure, most bulk capacity first."""
    out = []
    for tok in spec_list:
        if tok != "auto":
            f = tok.split(":")
            out.append((int(f[0]), int(f[1]), int(f[2]), len(f) > 3 and f[3] in ("t", "torus")))
            continue
        seen, cands = set(), []
        tor = {"flat": (False,), "torus": (True,), "any": (False, True)}[torus_mode]
        for size in range(n, n + 41):
            for torus in tor:
                for lrc in range(1, 9):
                    for lrr in range(1, 9):
                        try:
                            row = layout_row(size, lrc, lrr, torus, {})
                        except Exception:
                            continue
                        sids = collections.Counter(frozenset(s for s, _ in u) for u in row["unpack"])
                        bulk = {next(iter(k)): v for k, v in sids.items() if len(k) == 1}
                        if patches and len(bulk) != patches:
                            continue
                        cap = (max_width - 1) if max_width else 10 ** 9
                        width = {p: sum(v for k, v in sids.items() if p in k) for p in bulk}
                        if sum(min(cap, width[p]) for p in bulk) < n or size < n:
                            continue
                        sig = (torus, tuple(sorted((len(k), v) for k, v in sids.items())))
                        if sig in seen:
                            continue
                        seen.add(sig)
                        cands.append((-sum(min(cap, v) for v in bulk.values()), size, row["lrc"], row["lrr"], torus))
        cands.sort()
        out += [(s_, a_, b_, t_) for _, s_, a_, b_, t_ in cands[:limit]]
    return out


def score_records(path, depths):
    """Plain mean survival over instances per depth, and the score: geometric mean over the
    test depths (floor 1e-9, so a zero cannot hide behind others)."""
    per = defaultdict(list)
    for fn in [Path(path)] + sorted(Path(path).parent.glob(Path(path).stem + ".w*.jsonl")):
        try:
            for line in fn.read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("family") == "mirror" and r.get("depth") in depths:
                    per[r["depth"]].append(float(r["fidelity"]))
        except OSError:
            pass
    means = {d: (sum(v) / len(v)) for d, v in per.items() if v}
    if len(means) < len(depths):
        return means, None
    return means, math.exp(sum(math.log(max(means[d], 1e-9)) for d in depths) / len(depths))


def cmd_layoutsearch(a):
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    if not a.max_width or a.max_width < 2:
        raise SystemExit("--max-width must be given (patch width cap incl. the ancilla)")
    R, C, broken, L = lattice_info(a.repo)
    objective = tuple(a.objective.split(","))
    if any(o not in LS_OBJECTIVES for o in objective):
        raise SystemExit(f"--objective: choose from {','.join(LS_OBJECTIVES)}")
    window = None
    if a.window:
        rr, cc = a.window.split(",")
        window = tuple(map(int, rr.split("-"))) + tuple(map(int, cc.split("-")))
    fixed = None
    if a.region == "fixed":
        if a.fixed_register and a.fixed_register.startswith("release:"):
            # the release's first N logical qubits (row-major in its window), as a register of its own
            N = int(a.fixed_register.split(":", 1)[1])
            name = f"_release_first_{N}"
            REGISTERS[name] = release_prefix_spec(L, N, C)
            a.fixed_register = name
        if a.fixed_register:
            if a.fixed_register != "release" and a.fixed_register not in REGISTERS:
                raise SystemExit(f"--fixed-register: unknown register {a.fixed_register}")
            spec = REGISTERS[a.fixed_register] if a.fixed_register != "release" else None
            if spec is None:
                fixed = {(p // C, p % C) for p in L["logical_to_physical"]}
            else:
                (fr0, fr1), (fc0, fc1) = spec["rows"], spec["cols"]
                dr = set(spec.get("drop") or [])
                fixed = {(r, c) for r in range(fr0, fr1 + 1) for c in range(fc0, fc1 + 1) if r * C + c not in dr}
                if spec.get("drop") is None:
                    fixed = set(sorted(fixed)[:spec["n"]])
        elif window:
            fixed = {(r, c) for r in range(window[0], window[1] + 1) for c in range(window[2], window[3] + 1)
                     if (r, c) not in broken}
        else:
            raise SystemExit("--region fixed needs --window or --fixed-register")
        if len(fixed) != a.n:
            raise SystemExit(f"--region fixed: the region has {len(fixed)} sites, not --n {a.n}")
    cz_weight = None
    if a.cz_weights:
        if a.region != "fixed" or not a.fixed_register:
            raise SystemExit("--cz-weights needs --region fixed --fixed-register (the circuits must be known)")
        kind, _, ds = a.cz_weights.partition(":")
        if kind != "mirror":
            raise SystemExit("--cz-weights: mirror[:D1,D2,...]")
        lay_w = Layout(a.repo) if a.fixed_register == "release" else RegisterLayout(a.repo, a.fixed_register)
        wd = [int(x) for x in ds.split(",")] if ds else [4, 6, 8, 10, 12, 16, 20]
        cz_weight = mirror_cz_weights(lay_w, wd, C)
        if "cz" not in objective:
            objective = tuple(o if o != "bcz" else "cz" for o in objective)
            if "cz" not in objective:
                objective = objective[:1] + ("cz",) + objective[1:]
        print(f"# circuit weights: mirror depths {wd}, {sum(cz_weight.values())} CZs (forward halves)", flush=True)
    hint = None
    if a.hint:
        hs = REGISTERS[a.hint]
        lay_h = RegisterLayout(a.repo, a.hint)
        row_h = layout_row(hs["ace"]["width"], hs["ace"]["lrc"], hs["ace"]["lrr"], hs["ace"]["torus"], {})
        hint = {(p // C, p % C): frozenset(s for s, _ in row_h["unpack"][x_])
                for p, x_ in zip(lay_h.l2p, hs["ace"]["site"])}
    geoms = parse_geometries(a.geometries, a.n, a.max_width, a.patches, a.torus, a.max_geometries)
    if not geoms:
        raise SystemExit("no ACE geometry fits; relax --max-width or --patches, or give --geometries")
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    regfile = outdir / "registers.json"
    log = outdir / "layoutsearch.jsonl"
    prefix = a.prefix or f"ls{a.n}"
    print(f"# layoutsearch n={a.n}, region {a.region}" + (f" (trim {a.trim})" if a.region == "rect" else "")
          + (f" window {window}" if window else "")
          + f", broken qubits {a.broken}, caps {a.max_width}/{a.max_wide}, objective {','.join(objective)}, "
            f"{len(geoms)} geometr{'y' if len(geoms) == 1 else 'ies'}, {a.candidates} candidate(s) each", flush=True)
    cands = []
    for size, lrc, lrr, torus in geoms:
        row = layout_row(size, lrc, lrr, torus, {})
        lrc, lrr = row["lrc"], row["lrr"]                   # identical layouts fold to the smallest lrc+lrr
        edges = ace_coupling_edges(row, size, torus, {}, ())
        prev = []
        for c in range(a.candidates):
            res = search_layout(row, edges, a.n, R=R, C=C, broken=broken, broken_policy=a.broken,
                                region=a.region, window=window, fixed=fixed, max_width=a.max_width,
                                max_wide=a.max_wide, objective=objective, nogoods=prev, diversity=a.diversity,
                                hint=hint if c == 0 else None, time_s=a.cpsat_time, workers=a.workers, seed=c,
                                trim=a.trim, min_boundary=a.min_boundary, cz_weight=cz_weight)
            geo = f"{size} sites lrc {lrc} lrr {lrr}{' torus' if torus else ''}"
            if res is None or "cells" not in res:
                print(f"  {geo}: {'infeasible' if res else 'no solution in time'}"
                      + (" (no further distinct layout)" if c else ""), flush=True)
                break
            prev.append({p for p, g in res["cells"].items() if len(g) > 1})
            spec, site = register_from_cells(res["cells"], row, edges, C)
            name = f"{prefix}-{size}-{lrc}{lrr}{'t' if torus else ''}-c{c}"
            entry = dict(spec, about=f"layoutsearch: n={a.n}, {geo}, region {a.region}, objective {','.join(objective)}",
                         ace=dict(width=size, lrc=lrc, lrr=lrr, torus=torus, max_width=a.max_width,
                                  max_wide=a.max_wide, site=site))
            REGISTERS[name] = entry
            save_register(regfile, name, entry)
            ver = verify_register(name, entry, a.repo)
            if a.gtt is not None or a.vram_gb is not None:
                vg = a.vram_gb or 8.0
                sp = die_split(ver["widths"], vg, a.gtt or 0.0)
                ver["die"] = dict(sp, vram_gb=vg, gtt_gb=a.gtt or 0.0)
                if not sp["fits"]:
                    ver["ok"] = False
                    ver["problems"].append(f"die: {describe_split(sp, vg, a.gtt or 0.0)}")
            rec = dict(name=name, geometry=dict(size=size, lrc=lrc, lrr=lrr, torus=torus), cpsat=dict(
                status=res["status"], seconds=res["seconds"], seam=res["seam"], bcz=res["bcz"], cz=res["cz"],
                couplers=res["couplers"], widths=res["widths"]), verify=ver, spec=spec)
            cands.append(rec)
            print(f"  {name}: {res['status'].lower()} in {res['seconds']} s -- {res['seam']} boundary qubits, "
                  f"{res['bcz']} of {res['couplers']} couplers through a boundary"
                  + (f" ({res['cz']} circuit CZs)" if cz_weight else "") + f", widths {res['widths']}"
                  + (f", VRAM {ver['die']['vram']:.1f} + GTT {ver['die']['gtt']:.1f} GiB" if "die" in ver else "") + "; "
                  f"verify {'OK' if ver['ok'] else 'FAILED: ' + '; '.join(ver['problems'])} "
                  f"({ver['exact']}/{ver['replica']}/{ver['cross']})", flush=True)
    good = [c for c in cands if c["verify"]["ok"]]
    if a.test_depths and good:
        env = dict(os.environ)
        env["NIGHTHAWK_REGISTERS"] = os.pathsep.join([str(regfile.resolve())] + ([env["NIGHTHAWK_REGISTERS"]]
                                                     if env.get("NIGHTHAWK_REGISTERS") else []))
        if a.cpu:
            os.makedirs("/tmp/no-icd", exist_ok=True)
            env.setdefault("OCL_ICD_VENDORS", "/tmp/no-icd")
        print(f"# circuit tests: mirror, depths {a.test_depths}, {a.test_instances} instance(s) x "
              f"{a.test_twirls} twirls x 10 inputs, theta {a.theta}, {'CPU' if a.cpu else 'OpenCL'}", flush=True)

        def test(c):
            out = outdir / f"{c['name']}.jsonl"
            cmd = [sys.executable, os.path.abspath(__file__), "run", "--backend", "ace", "--register", c["name"],
                   "--theta", a.theta, "--mirror-estimator", "prob", "--instances", str(a.test_instances),
                   "--twirls", str(a.test_twirls), "--out", str(out), "--depths", *map(str, a.test_depths)]
            if a.repo:
                cmd += ["--repo", a.repo]
            cmd += ["--cpu"] if a.cpu else ["--ace-gpu"] + (["--gtt", str(a.gtt), "--vram-gb", str(a.vram_gb or 8.0)] if a.gtt else [])
            t0 = time.time()
            p = subprocess.run(cmd, env=env, capture_output=True, text=True)
            c["test"] = dict(seconds=round(time.time() - t0, 1), returncode=p.returncode)
            means, score = score_records(out, a.test_depths)
            c["test"].update(means={str(k): v for k, v in sorted(means.items())}, score=score)
            if p.returncode != 0:
                c["test"]["stderr"] = p.stderr[-2000:]
            return c

        with ThreadPoolExecutor(max_workers=max(1, a.test_jobs)) as ex:
            for c in ex.map(test, good):
                t_ = c["test"]
                print(f"  {c['name']}: " + (f"score {t_['score']:.4g}  " if t_["score"] is not None else "no score  ")
                      + "  ".join(f"d{d} {v:.4g}" for d, v in t_["means"].items()) + f"  ({t_['seconds']} s)"
                      + ("" if t_["returncode"] == 0 else f"  exit {t_['returncode']}"), flush=True)
    for c in cands:
        with open(log, "a") as fh:
            fh.write(json.dumps(c) + "\n")

    def key(c):
        sc = c.get("test", {}).get("score")
        return (sc is None, -(sc or 0), c["cpsat"]["seam"], c["cpsat"]["cz"] if cz_weight else 0,
                c["cpsat"]["bcz"], -c["cpsat"]["couplers"])
    ranked = sorted(good, key=key)
    if not ranked:
        raise SystemExit("no verified candidate")
    print(f"\n== ranking ({'measured score' if a.test_depths else 'CP-SAT objective'}) ==")
    print("name                             boundary  bnd CZ/couplers  widths            "
          + ("circuit CZs  " if cz_weight else "") + "score")
    for c in ranked:
        sc = c.get("test", {}).get("score")
        print(f"{c['name']:<32} {c['cpsat']['seam']:>8}  {c['cpsat']['bcz']:>6}/{c['cpsat']['couplers']:<8}  "
              f"{str(c['cpsat']['widths']):<16}  " + (f"{c['cpsat']['cz']:>11}  " if cz_weight else "")
              + f"{'' if sc is None else f'{sc:.4g}'}")
    best = ranked[0]
    print(f"\nbest: {best['name']}  (registers in {regfile}, log in {log})")
    if a.save_best:
        if a.save_best in BUILTIN_REGISTERS:
            raise SystemExit(f"--save-best: {a.save_best} is a built-in register name")
        entry = dict(REGISTERS[best["name"]])
        entry["about"] = entry["about"] + f"; selected from {len(cands)} candidates as {best['name']}"
        save_register(_registry_path(), a.save_best, entry)
        print(f"saved as --register {a.save_best} in {_registry_path()}")


# ================================================================== selftest
def cmd_selftest(a):
    from pyqrack import QrackSimulator
    lay = Layout(a.repo)
    ok = True

    def report(name, good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"{name}: {msg}  {'OK' if good else 'FAIL'}")

    rel = "mirror/d08_instance1.qasm"
    cyc = mirror_cycles(lay, 8, 1)
    good, why = same_ops(ops_forward(cyc) + ops_inverse(cyc),
                         qasm_ops((lay.repo / "data/circuits" / rel).read_text()), 1e-9)
    report("[1] angle hash + pseudo-patch rotation vs released " + rel, good, why or "identical")

    M = gate(haar_angles(BASE_SEED, 0, 0, 0))
    s1, s2 = QrackSimulator(1, is_gpu=False), QrackSimulator(1, is_gpu=False)
    s1.u(0, *u_angles(M))
    s2.mtrx(flat(M), 0)
    ov = abs(np.vdot(np.array(s1.out_ket()), np.array(s2.out_ket())))
    report("[2] u_angles -> Qrack u == Haar matrix", abs(1 - ov) < 1e-5, f"overlap {ov:.7f}")

    ns = argparse.Namespace(backend="exact", cpu=True, exact_probs=True, shots="1000", twirls=0, n=a.n)
    eng = ExactEngine(a.n, True)
    F = run_mirror(lay, eng, 12, 0, ns)["fidelity"]
    report(f"[3] exact mirror, first {a.n} logical qubits, d=12, 10 inputs", abs(1 - F) < 1e-4, f"F = {F:.7f}")

    eng.reset()
    rng = np.random.default_rng(3)
    bits = [int(c) for c in lay.inputs[0][:a.n]]
    for q, b in enumerate(bits):
        if b:
            eng.pauli("x", q)
    cyc = mirror_cycles(lay, 12, 0, n=a.n)
    apply_forward(eng, cyc, rng)
    apply_inverse(eng, cyc, rng)
    F = eng.prob_bits(bits)
    report("[4] Pauli-frame twirl is an identity (exact)", abs(1 - F) < 1e-4, f"F = {F:.7f}")

    K, d, j, i = 3, 20, 0, 0
    part = lay.partitions[K][j]
    probs = patch_probs(patched_cycles(lay, K, d, j, i), part["patches"], True)
    rr = next(r for r in json.loads((lay.repo / "data/results/patch_xeb.json").read_text())
              if (r["K"], r["depth"], r["partition"], r["instance"]) == (K, d, j, i))
    mine, theirs = sorted(ideal_xeb(p) for p in probs), sorted(rr["patch_ideal_xeb"].values())
    diff = max(abs(x - y) for x, y in zip(mine, theirs))
    report("[5] PyQrack patch ideal XEB vs release (Aer, double)", diff < 1e-3, f"max diff {diff:.1e}")

    shots = np.load(lay.repo / f"data/counts/patched_K{K}_d{d}.npz")[f"partition{j}_instance{i}"]
    r = patched_xeb(shots, probs, part["patches"])
    report("[6] Eq.(1)-(2) on the hardware counts vs release", abs(r["fidelity"] - rr["fidelity"]) < 0.05 * rr["se"],
           f"{r['fidelity']:.5f} vs {rr['fidelity']:.5f} (se {rr['se']:.1e})")

    samp = np.zeros(4000, dtype=np.uint64)
    g = np.random.default_rng(9)
    for qs, p in zip(part["patches"], probs):
        x = g.choice(p.size, size=samp.size, p=p)
        for jj, q in enumerate(qs):
            samp |= ((x >> jj) & 1).astype(np.uint64) << np.uint64(q)
    r = patched_xeb(samp, probs, part["patches"])
    report("[7] ideal samples score F = 1 within 4 se", abs(1 - r["fidelity"]) < 4 * r["se"],
           f"F = {r['fidelity']:.4f} +/- {r['se']:.4f}")

    try:
        ae = AceEngine(lay, lay.n, argparse.Namespace(lrc=a.lrc, lrr=a.lrr, cpu=True))
        print(f"[8] ACE register {ae.size} qubits, grid {ae.sim.get_row_length()}x{ae.sim.get_column_length()}, "
              f"is_torus={ae.torus}; logical->ACE map is nearest-neighbour for all couplers: "
              f"{all(abs(lay.pos[u][0] - lay.pos[v][0]) + abs(lay.pos[u][1] - lay.pos[v][1]) == 1 for es in lay.matchings.values() for u, v in es)}")
    except ImportError as err:
        print(f"[8] ACE not available: {err}")
    # [9] the --register recipe, applied to the release's own register, reproduces the release
    pt = _load_release_module(lay.repo, "patching")
    rng = np.random.default_rng(seed=lay.base_seed)
    strings = []
    for _ in range(len(lay.inputs)):
        bits = np.array(["0"] * lay.n)
        bits[rng.choice(lay.n, size=lay.n // 2, replace=False)] = "1"
        strings.append("".join(bits))
    same = strings == lay.inputs
    for K in (3, 4):
        parts = pt.make_patches(lay.n, logical_couplers(lay, lay.n), K, seed=lay.base_seed, balance_tol=1, top_n=5)
        same &= [frozenset(tuple(sorted(e)) for e in b) for b, _ in parts] == [p["boundary"] for p in lay.partitions[K]]
    report("[9] --register recipe reproduces the release's inputs and K = 3, 4 partitions", same, "identical" if same else "differs")
    # [10] layoutsearch's joint model, held to the release's own sites, reaches the exact engine's
    # optimum, and its placement passes the independent check (no coupler between patches)
    try:
        import ortools  # noqa: F401
        R_, C_, broken, _ = lattice_info(lay.repo)
        row = layout_row(75, 4, 5, True, {})
        edges = ace_coupling_edges(row, 75, True, {}, ())
        cp = logical_couplers(lay, lay.n)
        res = search_layout(row, edges, lay.n, R=R_, C=C_, broken=broken, region="fixed",
                            fixed={(p // C_, p % C_) for p in lay.l2p}, max_width=32, max_wide=3, time_s=90)
        site_e, _ = _exact_on_register(row, 75, cp, edges, (), 90, 32, 3)
        se = placement_stats(row, site_e, cp)
        sj = placement_stats(row, register_from_cells(res["cells"], row, edges, C_)[1], cp) if res and "cells" in res else None
        good = bool(sj) and sj["cross"] == 0 and sj["seam_used"] == se["seam_used"] == res["seam"]
        report("[10] layoutsearch joint model vs exact engine, release register on 75 sites (lrc 4 lrr 5, torus)", good,
               f"{sj['seam_used'] if sj else '-'} / {se['seam_used']} boundary qubits, "
               f"{sj['replica'] if sj else '-'} / {se['replica']} couplers through a boundary")
    except ImportError:
        print("[10] OR-Tools not installed: layoutsearch check skipped")
    # [11] --protocol nnqab hamming: each of nn_qab.py's 12 gates undone by qrack_g2_inv, and the
    # folded mirror (U U^dagger)^lambda returns the exact engine to |0...0> (signal 1)
    rng = np.random.default_rng(7)
    worst = 0.0
    for g in NNQAB_GATES:
        sim = QrackSimulator(2, is_gpu=False)
        for q in range(2):
            sim.u(q, *rng.uniform(0, 2 * math.pi, 3))
        k0 = np.array(sim.out_ket())
        qrack_g2(sim, g, 0, 1)
        qrack_g2_inv(sim, g, 0, 1)
        worst = max(worst, 1 - abs(np.vdot(k0, np.array(sim.out_ket()))))
    old = PROTOCOL["mode"]
    PROTOCOL["mode"] = "nnqab"
    try:
        sig = [run_hamming(lay, ExactEngine(a.n, True), 8, lam, 0, ns)["fidelity"] for lam in (1, 3)]
    finally:
        PROTOCOL["mode"] = old
    report(f"[11] nnqab gate inverses (12 gates) and exact folded mirror, first {a.n} qubits, d=8, lambda 1, 3",
           worst < 1e-5 and all(abs(1 - x) < 1e-4 for x in sig),
           f"worst 1 - overlap {worst:.1e}, signal {sig[0]:.6f} / {sig[1]:.6f}")
    # [12] vqe: the HVA circuit's energy read out shot-free on the exact engine equals the independent numpy
    # state vector; p = 0 gives -h n; Lanczos E0 equals dense diagonalisation; bound <= E0 <= every energy
    nv = min(a.n, 12)
    cols = vqe_colours(lay, nv)
    gam, bet = rng.uniform(0, 2, 3), rng.uniform(0, 2, 3)
    ref = TfimNumpy(nv, cols, 1.0, 3.04)
    e_np = ref.energy(ref.state(gam, bet))
    e_q = vqe_energy(ExactEngine(nv, True), nv, cols, gam, bet, 1.0, 3.04)["energy"]
    e_p0 = ref.energy(ref.state([], []))
    nd = min(nv, 10)
    rd = TfimNumpy(nd, vqe_colours(lay, nd), 1.0, 3.04)
    Hd = np.diag(-rd.D.astype(float))
    ii = np.arange(1 << nd)
    for q in range(nd):
        Hd[ii ^ (1 << q), ii] -= 3.04
    e0_l, e0_d = rd.ground(), float(np.linalg.eigvalsh(Hd)[0])
    e0 = ref.ground()
    bnd = tfim_anderson_bound(nv, cols, 1.0, 3.04)
    report(f"[12] vqe: HVA energy, exact engine vs numpy, first {nv} qubits, p = 3; Lanczos vs dense ({nd}); "
           f"bound <= E0 <= E",
           abs(e_np - e_q) < 1e-3 * abs(e_np) and abs(e_p0 + 3.04 * nv) < 1e-9 and abs(e0_l - e0_d) < 1e-8
           and bnd <= e0 <= min(e_np, e_p0) + 1e-9,
           f"{e_q:.6f} vs {e_np:.6f}; p = 0 {e_p0:.4f}; E0 {e0_l:.8f} vs {e0_d:.8f}; bound {bnd:.3f} <= {e0:.4f}")
    # [13] shor9: both layouts read off ACE's own seam set use only couplings in its logical coupling map, and on
    # the exact simulator every logical state decodes (and reads raw) with success 1.0
    from pyqrack import QrackAceBackend
    probe = ace_backend(SHOR9_WIDTH, is_gpu=False, **SHOR9_ACE)
    lays9 = shor9_layouts(probe)
    for blocks in lays9.values():
        shor9_check_coupling(probe, shor9_encoder(blocks))
    del probe
    ex9 = {(nm, st): shor9_run(QrackSimulator(SHOR9_WIDTH, is_gpu=False), bl, st, 256)
           for nm, bl in lays9.items() for st in SHOR9_STATES}
    tl = shor9_tiles(lay)                   # the release's own register: units on its couplers (checked inside)
    sim = QrackSimulator(lay.n, is_gpu=False)
    for (u, _, _), st in zip(tl, SHOR9_STATES * len(tl)):
        shor9_prepare(sim, u, st)
    for (u, _, _), st in zip(tl, SHOR9_STATES * len(tl)):
        qs, loc = _local_unit(u)
        ex9[("release", st, qs[0])] = tuple(float(np.mean(v)) for v in zip(
            *[shor9_decode(x, loc, st) for x in sim.measure_shots(qs, 128)]))
    # the exact distribution: shor_possible / shor_support against a numpy state vector (Shor-9, Shor-16), and
    # exact Qrack samples against it (no impossible outcome, distance within 3x the floor)
    worst9 = 0.0
    for mm in (3, 4):
        blocks_m = [[i * mm + j for j in range(mm)] for i in range(mm)]
        um = shor_bfs_unit(blocks_m, {(i * mm + j, i * mm + j + 1) for i in range(mm) for j in range(mm - 1)}
                           | {(i * mm, (i + 1) * mm) for i in range(mm - 1)})
        for st in SHOR9_STATES:
            pnp = shor9_exact_dist(um, st)
            sup = {x for x in range(1 << (mm * mm)) if shor_possible(x, blocks_m, st)}
            worst9 = max(worst9, float(abs(len(sup) - shor_support(blocks_m, st))),
                         float(np.abs(pnp[sorted(sup)] - 1.0 / len(sup)).max()), float(pnp.sum() - pnp[sorted(sup)].sum()))
    for nm, u in lays9.items():
        for st in SHOR9_STATES:
            h9 = collections.Counter()
            shor9_run(QrackSimulator(SHOR9_WIDTH, is_gpu=False), u, st, 4096, h9)
            c9 = shor9_compare(h9, u["blocks"], st)
            worst9 = max(worst9, c9["impossible"], c9["tvd"] - 3 * c9["tvd_floor"])
    ex9[("dist", "numpy vs analytic vs exact samples", 0)] = (1.0, 1.0) if worst9 <= 1e-12 else (0.0, 0.0)
    report(f"[13] shor9: encoder within ACE's coupling map; exact decoder, both 3x3 layouts and {len(tl)} units "
           f"tiled on the release's {lay.n} qubits, 4 logical states; exact distribution (Shor-9, Shor-16)",
           all(r == (1.0, 1.0) for r in ex9.values()),
           f"{sum(r == (1.0, 1.0) for r in ex9.values())} / {len(ex9)} at 1.0 / 1.0")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


# ================================================================== seamgap (numpy only, unchanged)
def cmd_seamgap(a):
    MED_CZ = 1.9e-3 * 5 / 4          # RB -> Pauli channel, (d+1)/d with d = 4
    MED_SX = 2.5e-4 * 3 / 2          # d = 2
    repo = find_repo(a.repo)
    man = json.loads((repo / "data/circuits/manifest.json").read_text())["circuits"]
    tab = collections.defaultdict(list)
    for c in man:
        kind = c["kind"] if c["kind"] != "patched" else "K" + c["file"].split("/")[1].lstrip("K")
        n = len(re.findall(r"^\s*cz\b", (repo / "data/circuits" / c["qasm3"]).read_text(), re.M))
        assert n == c["cz"], f"manifest mismatch in {c['file']}"
        tab[(kind, int(re.search(r"d(\d+)", c["file"]).group(1)))].append(n)
    mean = {k: st.mean(v) for k, v in tab.items()}
    kinds = ["full", "mirror", "K3", "K4"]
    print("== CZ counts (mean [min-max]); gaps relative to the sampled full circuit ==")
    print("depth\t" + "\t".join(kinds) + "\tfull-mirror\tfull-K3")
    for d in sorted({d for _, d in tab}):
        cells = [f"{st.mean(tab[(k, d)]):.0f}[{min(tab[(k, d)])}-{max(tab[(k, d)])}]"
                 if (k, d) in tab else "-" for k in kinds]
        gap = lambda k: (f"{mean[('full', d)] - mean[(k, d)]:.0f} "
                         f"({(mean[('full', d)] - mean[(k, d)]) / mean[('full', d)]:.1%})"
                         if (k, d) in mean and ("full", d) in mean else "-")
        print(f"{d}\t" + "\t".join(cells) + f"\t{gap('mirror')}\t{gap('K3')}")

    res = json.loads((repo / "data/results/fidelity_vs_depth.json").read_text())
    pts, fit = res["points"], res["fit"]
    A, f = fit["prefactor"], fit["fidelity_per_cycle"]
    d0 = 36
    czpc = mean[("mirror", d0)] / d0
    deq = mean[("full", d0)] / czpc
    missing = mean[("full", d0)] - mean[("mirror", d0)]
    print(f"\n== fit F(d) = {A:.4f} x {f:.4f}^d ==")
    print(f"at d={d0}: {A * f ** d0:.3e} (headline);  at gate-count-equivalent d={deq:.1f}: "
          f"{A * f ** deq:.3e} (all decay charged to CZ)")
    for eps in (1e-3, MED_CZ, 3e-3):
        print(f"  {missing:.0f} missing CZ at Pauli eps {eps:.2e}: {A * f ** d0 * (1 - eps) ** missing:.3e}")

    fam = {"mirror": "mirror", "3-patch": "K3", "4-patch": "K4"}
    rows = [(fk, int(ds), mean[(fam[fk], int(ds))], np.log(v["fidelity"]), v["se"] / v["fidelity"])
            for fk, pk in pts.items() for ds, v in pk.items() if int(ds) >= a.dmin]
    fams = sorted({r[0] for r in rows})
    print(f"\n== ln F = a + beta*cycles + gamma*CZ, d >= {a.dmin} ==")
    for off in (False, True):
        X = np.array([[d, c] + ([float(fk == g) for g in fams] if off else [1.0]) for fk, d, c, _, _ in rows])
        w = np.array([1 / s for *_, s in rows])
        y = np.array([lf for *_, lf, _ in rows])
        Xw, yw = X * w[:, None], y * w
        b, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        r = yw - Xw @ b
        cov = np.linalg.inv(Xw.T @ Xw) * (r @ r / max(len(y) - X.shape[1], 1))
        e = np.sqrt(np.diag(cov))
        print(f"{'family offsets' if off else 'common intercept':17s} beta {b[0]:+.4f}+/-{e[0]:.4f}  "
              f"gamma {b[1]:+.5f}+/-{e[1]:.5f}   (Pauli CZ error would give {np.log(1 - MED_CZ):+.5f})")
    print("\n== K=3 vs K=4 at equal depth ==\ndepth\tF3/F4\tdCZ\timplied eps_CZ")
    for d in sorted(int(x) for x in pts["3-patch"] if x in pts["4-patch"]):
        r3 = pts["3-patch"][str(d)]["fidelity"] / pts["4-patch"][str(d)]["fidelity"]
        dc = mean[("K3", d)] - mean[("K4", d)]
        print(f"{d}\t{r3:.3f}\t{dc:.0f}\t{-np.log(r3) / dc:+.5f}")
    dec = -np.log(f)
    bcz, b1q = czpc * MED_CZ, 61 * 2 * MED_SX
    print(f"\n== per-cycle budget (Pauli rates) ==\nfitted {dec:.4f} | CZ {bcz:.4f} | 1q {b1q:.4f} | "
          f"unexplained {dec - bcz - b1q:.4f} ({(dec - bcz - b1q) / dec:.0%})")


# ================================================================== CLI
def main():
    load_register_files()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seamgap", help="data-only analysis of the released repo")
    s.add_argument("--repo", default=None, help="rcs-nighthawk clone (default: auto-detect)")
    s.add_argument("--dmin", type=int, default=20)

    v = sub.add_parser("verify", help="regenerate all circuits and compare with the released QASM")
    v.add_argument("--repo", default=None, help="rcs-nighthawk clone (default: auto-detect)")
    v.add_argument("--tol", type=float, default=1e-9, help="angle tolerance, rad")

    h = sub.add_parser("hwxeb", help="re-score the ibm_phoenix bitstrings with PyQrack ideal patches")
    h.add_argument("--repo", default=None, help="rcs-nighthawk clone (default: auto-detect)")
    h.add_argument("--K", type=int, nargs="+", default=[3, 4])
    h.add_argument("--depths", type=int, nargs="+", default=None)
    h.add_argument("--limit", type=int, default=0, help="stop after this many circuits (0 = all 180)")
    h.add_argument("--cache", default=None, help="directory for ideal patch distributions (float32 npz)")
    h.add_argument("--cpu", action="store_true")
    h.add_argument("--out", default=None, help="write per-circuit records (patch_xeb.json format)")

    r = sub.add_parser("run", help="clean-qubit emulation of the released circuits")
    _RUN_PARSER["r"] = r
    r.add_argument("--repo", default=None, help="rcs-nighthawk clone (default: auto-detect)")
    r.add_argument("--backend", choices=["exact", "ace"], default="ace")
    r.add_argument("--register", default="release", choices=["release"] + sorted(REGISTERS),
                   help="release (default): the experiment's 61-qubit register; nighthawk88: an 88-qubit "
                        "Nighthawk sub-register built with the release's own recipe (mirror only; with --backend ace "
                        "the four-patch preset that fits one 8 GB GPU: 102 sites, lrc 4 lrr 5, widths 28/27/27/27); "
                        "nighthawk88d: 88 qubits with disabled sites placed on the seams, 11 boundary qubits "
                        "instead of 17 (114 sites, lrc 4 lrr 5, widths 28/23/25/27); nighthawk88e: as 88d with 36 "
                        "couplers through a boundary instead of 42 (found by layoutsearch); plus any register "
                        "saved by layoutsearch")
    r.add_argument("--transpile-max-width", dest="transpile_max_width", type=int, default=None,
                   help="exact engine: no ACE patch wider than this (logical qubits + error-detection ancilla)")
    r.add_argument("--transpile-max-wide", dest="transpile_max_wide", type=int, default=None,
                   help="with --transpile-max-width W: at most this many patches at W, the others <= W-1")
    r.add_argument("--families", default="mirror,patched",
                   help="comma list of mirror, patched, full, fxeb, hamming, vqe (full: unscored d36 samples; "
                        "fxeb: forward XEB vs an exact reference, n <= ~34; vqe: transverse-field Ising energy of "
                        "an HVA circuit, --depths = layers p)")
    r.add_argument("--theta", choices=["haar", "nnqab"], default="haar",
                   help="single-qubit theta: haar (the paper) or nnqab (sin(theta) uniform, as nn_qab.py; "
                        "a variant, not the released circuits)")
    r.add_argument("--ref-gpu", dest="ref_gpu", action="store_true",
                   help="fxeb: exact reference on OpenCL (default CPU)")
    r.add_argument("--pubs", type=int, default=SAMPLE_PUBS, help="full: pubs of --shots each (paper 10 x 100k)")
    r.add_argument("--shots-dir", dest="shots_dir", default=None,
                   help="where bitstrings go (default: <out stem>_shots/); a <cfg> subfolder holds points/ + release/")
    r.add_argument("--pack", action="store_true", help="only rebuild the release-format files from stored points")
    r.add_argument("--depths", type=int, nargs="+", default=None, help="default: the release's depths")
    r.add_argument("--K", type=int, nargs="+", default=[3, 4])
    r.add_argument("--instances", type=int, default=None, help="default 3, as released")
    r.add_argument("--partitions", type=int, default=None, help="use the first j partitions (default all 5)")
    r.add_argument("--shots", default="paper",
                   help="'paper' (Appendix D budgets, listed depths only; full: 100k per pub) "
                        "or shots per circuit / pub")
    r.add_argument("--twirls", type=int, default=GATE_TWIRLS, help="ace mirror Pauli-frame randomisations")
    r.add_argument("--exact-probs", dest="exact_probs", action=argparse.BooleanOptionalAction, default=True,
                   help="exact backend mirror: read survival with prob_perm instead of sampling")
    r.add_argument("--sizes", default=None,
                   help="register sizes to sweep, first-n truncation: 36, 27-36 or 20,27-36 (mirror; patched at 61 only)")
    r.add_argument("--n", type=int, default=None, help="single size, same as --sizes n")
    r.add_argument("--geometry", choices=["manual", "nnqab"], default="manual",
                   help="ace: manual = --lrc/--lrr as given, not a torus (default); nnqab = nn_qab.py's "
                        "rule: rows whole, two patches, highest bulk-to-boundary, torus (overrides --lrc/--lrr "
                        "and sets is_torus=True)")
    r.add_argument("--ace-torus", dest="ace_torus", action="store_true", help="ace: is_torus=True")
    r.add_argument("--lrc", default=4, type=lambda v: v if v == "auto" else int(v),
                   help="ace: long_range_columns, or auto (nn_qab.py two-patch rule)")
    r.add_argument("--lrr", default=4, type=lambda v: v if v == "auto" else int(v),
                   help="ace: long_range_rows, or auto")
    r.add_argument("--ace-max-width", dest="ace_max_width", type=int, default=None,
                   help="ace: bigger patches -- the layout with the fewest circuit couplers on a seam whose "
                        "widest internal simulator has at most this many qubits (see aceplan); overrides --lrc/--lrr")
    r.add_argument("--ace-tiling-version", dest="ace_tiling_version", type=int, default=TILING_VERSION,
                   choices=[2, 3], help=f"tiling search (default {TILING_VERSION}): 2 = couplers only, one run; "
                   "3 = + simulators used and seam qubits, restarts, warm start from n-1 (see TILING_VERSION)")
    r.add_argument("--variants", nargs="+", default=None, metavar="SPEC",
                   help="fxeb only: score several ACE configurations against ONE exact reference per point, "
                        "e.g. --variants 4/4:t3 4/4:t2 4/3:t3 (lrc/lrr, then torus / untiled / tiled / t2 / t3). "
                        "Each variant keeps its own config tag and records")
    r.add_argument("--ace-transpile", dest="ace_transpile", action="store_true",
                   help="ace: place logical qubits with Qiskit's VF2Layout on ACE's logical coupling map (bulk "
                        "fully connected inside a patch), as in pyqrack-examples rcs/mirror_nighthawk.py; "
                        "routing-free, so twirling, variants and estimators apply unchanged. Needs qiskit")
    r.add_argument("--ace-width", dest="ace_width", type=int, default=None,
                   help="with --ace-transpile: ACE register size in sites (default: the 8x8 device window); "
                        "e.g. 75 for mirror_nighthawk.py's 15x5 register")
    r.add_argument("--ace-exclude", dest="ace_exclude", default=None,
                   help="with --ace-transpile: ACE sites removed from the coupling map, e.g. '74,59,44,69,54,68,64' "
                        "(mirror_nighthawk.py's coupler_exclusions) to shape the boundary")
    r.add_argument("--transpile-engine", dest="transpile_engine", choices=["exact", "anneal", "vf2"], default="exact",
                   help="with --ace-transpile: exact (default) = CP-SAT: fewest qubits on seams, then even patches, then "
                        "fewest seam couplers (needs ortools); anneal = the tiling cost annealed on the register; "
                        "vf2 = Qiskit's VF2Layout as is")
    r.add_argument("--transpile-allow-cross", dest="transpile_allow_cross", action="store_true",
                   help="with --ace-transpile: accept placements with couplers between patches that share no seam "
                        "(by default refused: ACE has no boundary for that contact)")
    r.add_argument("--transpile-seed", dest="transpile_seed", type=int, default=0, help="VF2Layout seed")
    r.add_argument("--transpile-trials", dest="transpile_trials", type=int, default=0,
                   help=f"VF2Layout max_trials: layouts scored before keeping the best (0 = {TRANSPILE_TRIALS})")
    r.add_argument("--transpile-calls", dest="transpile_calls", type=int, default=0,
                   help=f"VF2Layout call_limit: bound on the isomorphism search (0 = {TRANSPILE_CALLS})")
    r.add_argument("--ace-boundary-rep", dest="ace_boundary_rep", action="store_true",
                   help="ace: is_boundary_repetition_code=True (the seam's real-gated replica decides; default off, passed "
                        "explicitly, since PyQrack 2.29 made it its own default)")
    r.add_argument("--no-ace-error-detection", dest="ace_error_detection", action="store_false",
                   help="ace: is_error_detection=False (Qrack default on: detect-and-post-select at the seams)")
    r.add_argument("--no-ace-crossbars", dest="ace_crossbars", action="store_false",
                   help="ace: use_crossbars=False (Qrack default on)")
    r.add_argument("--ace-tiling", dest="ace_tiling", action="store_true",
                   help="ace: place logical qubits on ACE sites as compact tiles (annealed) instead of the "
                        "device grid, so most couplers stay inside one simulator; same simulators and memory")
    r.add_argument("--ace-torus-search", dest="ace_torus_search", choices=["flat", "torus", "any"], default="flat",
                   help="--ace-max-width: search non-torus layouts only (flat, default: Nighthawk is not a torus), "
                        "torus only, or any (the first runs' behaviour and config tag)")
    r.add_argument("--ace-layout", dest="ace_layout", choices=["grid", "strip"], default="grid",
                   help="grid: full 8x8 subgrid for every size; strip: smallest R x 8 strip holding "
                        "the first n qubits (nn_qab-style register)")
    r.add_argument("--cache", default=None, help="directory for ideal patch distributions")
    r.add_argument("--cpu", action="store_true", help="is_gpu=False everywhere")
    r.add_argument("--ace-gpu", dest="ace_gpu", action="store_true",
                   help="run QrackAceBackend on OpenCL too (off by default: tiled placements reset Vega10/20 "
                        "dies under rusticl/radeonsi; they run on clvk/RADV). Shots are drawn on a CPU copy "
                        "unless --ace-gpu-shots")
    r.add_argument("--gtt", "-gtt", dest="gtt", type=float, default=None, metavar="GB",
                   help="--ace-gpu: allow VRAM oversubscription with up to GB of host memory per worker that the GPU "
                        "maps over PCIe (GTT): states Qrack places in host memory (> 1/3 of the die) or pages no longer "
                        "hit the die's allocation ceiling (QRACK_MAX_ALLOC_MB = VRAM + GB). Slower than VRAM; same "
                        "numbers, so not part of the config tag")
    r.add_argument("--vram-gb", dest="vram_gb", type=float, default=8.0,
                   help="device memory per worker's die for the --gtt plan (default 8: one V340 die)")
    r.add_argument("--fold", type=int, nargs="+", default=None, metavar="LAMBDA",
                   help="hamming: unitary folding scale factors (odd), e.g. 1 3 5 for Dan's zero-noise "
                        "extrapolation (danmodel then extrapolates in lambda^p, p = 1.60569); default 1")
    r.add_argument("--vqe-h", dest="vqe_h", type=float, default=3.04,
                   help="vqe: transverse field h of H = -J sum ZZ - h sum X (default 3.04, the square lattice's "
                        "critical point h/J = 3.044)")
    r.add_argument("--vqe-j", dest="vqe_j", type=float, default=1.0, help="vqe: coupling J (default 1)")
    r.add_argument("--vqe-opt-register", dest="vqe_opt_register", default="nighthawk2p16",
                   help="vqe: register the HVA angles are optimised on (numpy, exact), then used at every size "
                        "(default nighthawk2p16; 'self' = the run's own first --vqe-opt-n qubits)")
    r.add_argument("--vqe-opt-n", dest="vqe_opt_n", type=int, default=16, help="vqe: qubits for --vqe-opt-register self")
    r.add_argument("--vqe-exact-max", dest="vqe_exact_max", type=int, default=26,
                   help="vqe: numpy energy of the same circuit up to this many qubits (16 B x 2^n; default 26: 1 GiB)")
    r.add_argument("--vqe-ed-max", dest="vqe_ed_max", type=int, default=24,
                   help="vqe: Lanczos ground energy E0 up to this many qubits (default 24)")
    r.add_argument("--protocol", choices=["nighthawk", "nnqab"], default="nighthawk",
                   help="nnqab: score forward XEB exactly as rcs/nn_qab.py does (sin-theta-uniform single-qubit gates, "
                        "each coupler's two-qubit gate drawn from nn_qab.py's twelve, 2^min(13, n+2) shots), so points "
                        "compare one-to-one with Dan Strano's ACE calculator; --families fxeb only")
    r.add_argument("--swap-gb", dest="swap_gb", type=float, default=None, metavar="GB",
                   help="CPU runs larger than RAM: count up to GB of free swap as available memory (the "
                        "--mem-budget-gb ledger and the preflight plan), and lift Qrack's CPU qubit ceilings. "
                        "Slower once states page; same numbers, so not part of the config tag")
    r.add_argument("--ace-host-pointer", dest="ace_host_pointer", action="store_true",
                   help="ace with --ace-gpu: keep simulator states in host RAM, read by the GPU over PCIe "
                        "(GTT) instead of VRAM; same numbers, so not part of the config tag")
    r.add_argument("--out", default="nighthawk_clean.jsonl")
    r.add_argument("--summarize", action="store_true")
    r.add_argument("--mirror-estimator", dest="mirror_estimator", choices=["shots", "prob"], default="shots",
                   help="ace mirror: shots = sample with QrackAceBackend.measure_shots (clones the backend per "
                        "shot; paper budgets); prob = per circuit run, the probability ACE returns the prepared "
                        "string (chain rule with force_m): no shot noise, one pass per run, no bitstrings kept")
    r.add_argument("--order", choices=["deep", "shallow"], default="deep",
                   help="point order: deep = most expensive first (default, no lone tail); shallow = "
                        "cheapest first (early results with few workers)")
    r.add_argument("--status", action="store_true",
                   help="list every point of this configuration: done (F), running (progress, ETA), failed, "
                        "pending; give the same flags as the run. Reads files only, safe while it runs")
    r.add_argument("--gpus", default=None,
                   help="launch workers on these OpenCL devices, e.g. 0-5 or 0,2,4; pairs with --per-gpu")
    r.add_argument("--per-gpu", dest="per_gpu", type=int, default=3, help="workers per GPU (default 3)")
    r.add_argument("--stagger", type=float, default=20.0, help="seconds between worker starts (default 20)")
    r.add_argument("--init-gap", dest="init_gap", type=float, default=10.0,
                   help="min seconds between any two OpenCL initialisations, machine-wide (default 10; 0 = off)")
    r.add_argument("--max-attempts", dest="max_attempts", type=int, default=2,
                   help="workers that may die on one point before it is marked failed (default 2)")
    r.add_argument("--retry-failed", dest="retry_failed", action="store_true",
                   help="try points marked failed again")
    r.add_argument("--max-rss-gb", dest="max_rss_gb", type=float, default=0,
                   help="per-worker resident-memory cap: above it the point is marked failed and the worker "
                        "exits (0 = off). Use this instead of ulimit -v or QRACK_MAX_CPU_QB")
    r.add_argument("--mem-budget-gb", dest="mem_budget_gb", type=float, default=0,
                   help="machine-wide memory budget (GiB) shared by every worker and run using the same "
                        "ledger: a worker takes a point only if its estimated peak fits, else it takes a "
                        "smaller one or waits (0 = off: workers take points regardless of memory)")
    r.add_argument("--mem-per-point-gb", dest="mem_per_point_gb", type=float, default=None, metavar="GB",
                   help="book this many GiB per point in the --mem-budget-gb ledger instead of the worst-case "
                        "estimate (both patches fully entangled). For shallow points, which stay well below it "
                        "(57 qubits, d = 4-6: ~40 GiB measured against ~81 estimated); pair with --max-rss-gb so a "
                        "point that grows past it is stopped and can be retried alone. Not in the config tag")
    r.add_argument("--mem-ledger", dest="mem_ledger", default=None,
                   help="reservation directory shared by all runs on this machine "
                        "(default /tmp/nighthawk_mem_ledger, or $NIGHTHAWK_MEM_LEDGER)")
    r.add_argument("--gpu-isolation", dest="gpu_isolation", choices=["auto", "off"], default="auto",
                   help="auto (default): every GPU process sees exactly one OpenCL device; required for "
                        "clvk (one device per context), whose ICD is hidden from processes on other "
                        "platforms; without a clvk ICD this is the plain QRACK_OCL_DEFAULT_DEVICE pinning. "
                        "off: QRACK_OCL_DEFAULT_DEVICE pinning only")
    r.add_argument("--ace-gpu-shots", dest="ace_gpu_shots", action="store_true",
                   help="with --ace-gpu: draw shots on the GPU backend itself. Default: sample on a CPU copy "
                        "of the final state (one out_ket per internal simulator), because measure_shots "
                        "clones the whole backend per shot, i.e. new device buffers and copies per shot")
    r.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    p = sub.add_parser("aceplan", help="list ACE layouts: seams, simulator widths, memory")
    p.add_argument("--repo", default=None, help="rcs-nighthawk clone (default: auto-detect)")
    p.add_argument("--n", type=int, default=None, help="logical register (default 61)")
    p.add_argument("--ace-layout", dest="ace_layout", choices=["grid", "strip"], default="grid")
    p.add_argument("--torus", choices=["any", "false", "true"], default="false",
                   help="layouts to list: false (default, the device patch is not a torus), true or any")
    p.add_argument("--max-width", dest="max_width", type=int, default=None,
                   help="mark the --ace-max-width pick and hide wider layouts")
    p.add_argument("--all", action="store_true", help="with --max-width, still list the wider layouts")
    p.add_argument("--top", type=int, default=25, help="without --max-width, list this many (default 25)")
    p.add_argument("--tiling", action="store_true", help="count seam couplers after --ace-tiling placement")

    ls = sub.add_parser("layoutsearch", help="CP-SAT search for a register + ACE placement, verified and "
                                              "ranked by short mirror runs; the best is saved as a --register")
    ls.add_argument("--repo", default=None, help="rcs-nighthawk clone (default: auto-detect)")
    ls.add_argument("--n", type=int, required=True, help="logical qubits")
    ls.add_argument("--geometries", nargs="+", default=["auto"],
                    help="ACE registers to try, SIZE:LRC:LRR[:t] (t = torus), or auto (default): every distinct "
                         "geometry of n..n+40 sites with --patches patches that can hold n qubits under the cap")
    ls.add_argument("--patches", type=int, default=None, help="with auto: number of patches (2, 3, 4, ...)")
    ls.add_argument("--torus", choices=["flat", "torus", "any"], default="flat", help="with auto (default flat)")
    ls.add_argument("--max-geometries", dest="max_geometries", type=int, default=4,
                    help="with auto: try this many geometries, most bulk capacity first (default 4)")
    ls.add_argument("--max-width", dest="max_width", type=int, default=28,
                    help="patch width cap incl. the error-detection ancilla (default 28: 2 GiB fp32)")
    ls.add_argument("--gtt", "-gtt", dest="gtt", type=float, default=None, metavar="GB",
                    help="judge candidates by the die split instead of 8 GB VRAM alone: --vram-gb of device memory plus "
                         "GB of host memory mapped over PCIe (use with a larger --max-width, e.g. 29 or 30)")
    ls.add_argument("--vram-gb", dest="vram_gb", type=float, default=None,
                    help="judge candidates against one GPU die of this size (default 8 when --gtt is given; "
                         "no die check otherwise, e.g. for CPU runs)")
    ls.add_argument("--max-wide", dest="max_wide", type=int, default=1,
                    help="patches allowed at --max-width, the others <= max-width - 1 (default 1)")
    ls.add_argument("--region", choices=["rect", "convex", "free", "fixed"], default="rect",
                    help="rect (default): a rectangle of the lattice chosen by the solver, holes only at broken "
                         "qubits plus up to --trim sites off its border; convex: every row and column one unbroken "
                         "run (allows thin necks: patches joined by one or two qubits, i.e. nearly separate "
                         "circuits, when n is small against the lattice); free: any connected set (same, more so); "
                         "fixed: exactly --window's sites, or --fixed-register's (placement only)")
    ls.add_argument("--trim", type=int, default=0,
                    help="with rect: sites that may be left out along the rectangle's border (default 0)")
    ls.add_argument("--min-boundary", dest="min_boundary", type=int, default=0,
                    help="at least this many boundary qubits (a floor against trivially decoupled patches)")
    ls.add_argument("--window", default=None, help="restrict the region to rows-cols, e.g. 0-9,1-9")
    ls.add_argument("--fixed-register", dest="fixed_register", default=None,
                    help="with --region fixed: use this register's sites (release, release:N = the release's first N "
                         "logical qubits, nighthawk88, ...)")
    ls.add_argument("--broken", choices=["disable", "use"], default="disable",
                    help="the device's broken qubits (17, 55, 62): disabled, as on ibm_phoenix (default), or used")
    ls.add_argument("--objective", default="seam,bcz,couplers",
                    help="lexicographic, from " + ",".join(LS_OBJECTIVES) + ": seam = boundary qubits, bcz = "
                         "couplers through a boundary, couplers = couplers kept (maximised), balance = widest minus "
                         "narrowest patch, cz = CZs through a boundary in the circuits (needs --cz-weights) "
                         "(default seam,bcz,couplers)")
    ls.add_argument("--cz-weights", dest="cz_weights", default=None, metavar="mirror[:D,...]",
                    help="with --region fixed: weight each coupler by the CZs the register's mirror circuits apply on "
                         "it (forward halves, summed over depths D; default 4,6,8,10,12,16,20) and rank by those "
                         "(objective term cz, replacing bcz)")
    ls.add_argument("--candidates", type=int, default=2,
                    help="distinct layouts per geometry (each later one differs from every earlier one in at "
                         "least --diversity boundary sites; default 2)")
    ls.add_argument("--diversity", type=int, default=4)
    ls.add_argument("--hint", default=None, help="warm-start the first solve from this register's layout")
    ls.add_argument("--cpsat-time", dest="cpsat_time", type=float, default=300.0,
                    help="seconds per CP-SAT solve (default 300; FEASIBLE results are best found, not proven)")
    ls.add_argument("--workers", type=int, default=None, help="CP-SAT threads (default: cores, at most 16)")
    ls.add_argument("--test-depths", dest="test_depths", type=int, nargs="*", default=[4, 8],
                    help="mirror depths of the circuit test (default 4 8; none = rank by the CP-SAT objective only)")
    ls.add_argument("--test-instances", dest="test_instances", type=int, default=1)
    ls.add_argument("--test-twirls", dest="test_twirls", type=int, default=4,
                    help="twirls per input (10 inputs) in the circuit test (default 4: 40 runs per point)")
    ls.add_argument("--test-jobs", dest="test_jobs", type=int, default=1, help="candidates tested in parallel")
    ls.add_argument("--theta", choices=["haar", "nnqab"], default="nnqab")
    ls.add_argument("--cpu", action="store_true", help="circuit tests on CPU (OpenCL hidden)")
    ls.add_argument("--outdir", default="layoutsearch", help="candidate registers, test records and the log")
    ls.add_argument("--prefix", default=None, help="candidate name prefix (default ls<n>)")
    ls.add_argument("--save-best", dest="save_best", default=None,
                    help="save the best candidate as this --register name in nighthawk_registers.json")

    dm = sub.add_parser("danmodel", help="Dan Strano's ACE XEB calculator: predictions per register and depth, "
                                         "or against measured mirror/fxeb records")
    dm.add_argument("--repo", default=None, help="rcs-nighthawk clone (default: auto-detect)")
    dm.add_argument("--in", dest="inputs", nargs="+", default=None, metavar="JSONL",
                    help="run outputs: each measured point is compared with the calculator at its own bulk, "
                         "boundary and patch counts (worker files included)")
    dm.add_argument("--registers", nargs="+", default=None,
                    help="without --in: registers to predict (default: the two-patch ladder nighthawk2p28 ... "
                         "nighthawk2p57, nighthawk61r2)")
    dm.add_argument("--depths", type=int, nargs="+", default=None, help="without --in (default 4 6 8 10 12 16 20 36)")
    dm.add_argument("--device", action=argparse.BooleanOptionalAction, default=True,
                    help="add ibm_phoenix's mirror survival at n = 61 (default on)")
    dm.add_argument("--csv", default=None, help="also write the rows as CSV")

    hm = sub.add_parser("hamming", help="Dan's mirror Hamming-weight ladder (run --families hamming) against his "
                                        "calculator: per register, a measured / Dan grid by coupler load, Richardson")
    hm.add_argument("--in", dest="inputs", nargs="+", default=None, metavar="JSONL",
                    help="run outputs (worker files included; default: dan_hamming_*.jsonl in the working directory)")
    hm.add_argument("--protocol", default="nnqab", help="records of this protocol only (default nnqab; any = all)")
    hm.add_argument("--fit-from", dest="fit_from", type=int, default=6,
                    help="fit the depth power from this depth on (default 6)")
    hm.add_argument("--min-circuits", dest="min_circuits", type=int, default=10,
                    help="mark grid cells resting on fewer circuits (default 10)")
    hm.add_argument("--threshold", type=float, default=3.0,
                    help="coupler load named in the grid header (default 3.0, Dan's strips)")
    hm.add_argument("--csv", default=None, help="also write the lambda = 1 rows as CSV")

    s9 = sub.add_parser("shor9", help="one unit of Shor's 9-qubit code on a 3x3 two-patch ACE register: logical "
                                      "success raw / decoded for |0_L>, |1_L>, |+_L>, |-_L>, seam-leader and "
                                      "patch-block layouts, exact reference")
    s9.add_argument("--shots", type=int, default=8192, help="shots per logical state (default 8192)")
    s9.add_argument("--gpu", action="store_true",
                    help="QrackAceBackend and the reference on OpenCL (default CPU): the device QRACK_OCL_DEFAULT_DEVICE "
                         "names among the ICDs the loader sees, a clvk GPU die or PoCL's CPU device "
                         "(OCL_ICD_VENDORS=<dir with pocl.icd>)")
    s9.add_argument("--ocl-sync", dest="ocl_sync", action="store_true",
                    help="with --gpu: QRACK_OCL_SYNC=1, finish the OpenCL queue after every dispatch (Qrack >= 10.13.2; "
                         "without it, or the patched build, OpenCL results can be wrong: the exact reference stops the run)")
    s9.add_argument("--size", type=int, default=3,
                    help="code size m: Shor's [[m^2, 1, m]] code, m blocks of m qubits (3 = Shor-9, default; 4 = Shor-16, "
                         "5 = Shor-25, 6 = Shor-36, 7 = Shor-49); an m x m ACE grid, or m x m squares with --register")
    s9.add_argument("--repeats", type=int, default=1,
                    help="fresh ACE backends per logical state (default 1, as the original script); results "
                         "differ from backend to backend by more than shot noise, so use 5-10 for a mean")
    s9.add_argument("--no-exact", dest="no_exact", action="store_true", help="skip the exact QrackSimulator reference")
    s9.add_argument("--ace-boundary-rep", dest="ace_boundary_rep", action="store_true",
                    help="ACE's is_boundary_repetition_code=True (default off, passed explicitly: PyQrack 2.29 made "
                         "it its own default; with it on, ACE skips its seam error detection)")
    s9.add_argument("--jobs", type=int, default=1,
                    help="worker processes, one core each (default 1; 0 = every core): one task per logical state "
                         "and fresh backend (x layouts, or x rotations with --register), and the exact reference's "
                         "shots in chunks; a backend's shots stay in one task. With --gpu all workers share the "
                         "one OpenCL device (a clvk die: keep it to a few)")
    s9.add_argument("--out", default=None, help="append one JSON record per layout (per unit with --register)")
    s9.add_argument("--register", default=None,
                    help="tile Shor-9 units over this whole register (release, nighthawk2p45, nighthawk61r2, "
                         "nighthawk88e, any saved register) on its own couplers and ACE preset, all units in one "
                         "circuit; default: the 3x3 two-patch test")
    s9.add_argument("--max-tiles", dest="max_tiles", type=int, default=None, help="with --register: at most this many units")
    s9.add_argument("--repo", default=None, help="rcs-nighthawk clone (with --register; default: auto-detect)")

    t = sub.add_parser("selftest", help="conventions and exactness checks")
    t.add_argument("--repo", default=None, help="rcs-nighthawk clone (default: auto-detect)")
    t.add_argument("--n", type=int, default=14, help="register for the exact mirror checks")
    t.add_argument("--lrc", type=int, default=4)
    t.add_argument("--lrr", type=int, default=4)

    if len(sys.argv) == 1:
        ap.print_help()
        print("""
quick start (the BlueQubit release is found next to this script, in the working dir,
up to 3 parents up, via $NIGHTHAWK_REPO or --repo):
  git clone https://github.com/BlueQubitDev/rcs-nighthawk
  python3 nighthawk_qrack.py verify                                 # circuits vs released QASM
  python3 nighthawk_qrack.py selftest                               # conventions, ~1 min
  python3 nighthawk_qrack.py hwxeb    --cache xeb_cache
  python3 nighthawk_qrack.py run      --backend ace --families mirror \\
          --sizes 27-36,61 --depths 4 6 8 12 --out clean_ace.jsonl
help per command: python3 nighthawk_qrack.py <command> -h""")
        return
    a = ap.parse_args()
    if getattr(a, "gtt", None) and not getattr(a, "cpu", False):
        gtt_env(os.environ, float(getattr(a, "vram_gb", 8.0) or 8.0), float(a.gtt))   # inherited by re-exec and workers
    ensure_isolated(a)
    dict(seamgap=cmd_seamgap, verify=cmd_verify, hwxeb=cmd_hwxeb, run=cmd_run, selftest=cmd_selftest,
         aceplan=cmd_aceplan, layoutsearch=cmd_layoutsearch, danmodel=cmd_danmodel, hamming=cmd_hamming,
         shor9=cmd_shor9)[a.cmd](a)


if __name__ == "__main__":
    main()
