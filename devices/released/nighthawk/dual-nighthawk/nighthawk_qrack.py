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
  bridgebench
            the horizontal QrackBridge (HBridge): every bridge patch on a patch server (its own PyQrack
            instance, process, device or CPUs), the recompression and readout on a glue server (numpy only);
            random circuits on it and on PyQrack's own QrackBridge, states and shots compared, wall times and
            each workload's profile (gates, ket export / load, glue sweeps / sampling, coordinator wait).
            nighthawk_qrack.py ghz --bridge-engine horizontal runs every bridge command on it.
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

# ghz's loom imports nighthawk_cloverfield.py, which imports nighthawk_qrack: let that be this module
if __name__ in sys.modules:                 # (absent when a loader executes this file before registering it)
    sys.modules.setdefault("nighthawk_qrack", sys.modules[__name__])

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
    only loses its ancilla (one qubit narrower, half the memory). So does the boundary repetition
    code: the same sites and patches, with one more ancilla per patch in PyQrack 2.28 (twice the
    memory) and none from 2.29."""
    if is_release(lay):
        return None
    ace = REGISTERS[layout_id(lay)].get("ace")
    if not ace or n != lay.n or size != ace["width"] or (row["lrc"], row["lrr"]) != (ace["lrc"], ace["lrr"]):
        return None
    opts = {k: v for k, v in ace_opts(a).items() if k not in ("is_error_detection", "is_boundary_repetition_code")}
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


def _shor_timed(f, args):
    t = time.time()
    return f(*args), time.time() - t


def shor_pool_map(pool, calls, labels=None, every=30.0, what="shor9"):
    """[f(*args) for f, args in calls], in order: in the pool, or in this process without one. every > 0:
    a line per finished task (its label, seconds, share done, running tasks, elapsed and ETA) and, when
    nothing finishes for `every` seconds, a heartbeat naming the longest-running task."""
    n = len(calls)
    labels = labels or [f"task {i + 1}" for i in range(n)]
    t0 = time.time()

    def clock(done):
        el = time.time() - t0
        return f"{el / 60:.1f} min elapsed" + (f", ~{el * (n - done) / done / 60:.1f} min left" if 0 < done < n else "")
    out = [None] * n
    if pool is None:
        for i, (f, args) in enumerate(calls):
            if every:
                print(f"# {what} {i + 1}/{n} start: {labels[i]}", flush=True)
            t = time.time()
            out[i] = f(*args)
            if every:
                print(f"# {what} {i + 1}/{n} done ({100 * (i + 1) // n} %): {labels[i]} in {time.time() - t:.1f} s; "
                      f"{clock(i + 1)}", flush=True)
        return out
    import concurrent.futures as cf
    futs = {pool.submit(_shor_timed, f, args): i for i, (f, args) in enumerate(calls)}
    pending, seen, done_n, last = set(futs), {}, 0, time.time()
    cap = getattr(pool, "_max_workers", None) or n           # the executor marks a queued call or two running too
    nrun = lambda: min(cap, sum(1 for x in pending if x.running()))
    while pending:
        fin, pending = cf.wait(pending, timeout=5, return_when=cf.FIRST_COMPLETED)
        now = time.time()
        for f_ in pending:
            if f_ not in seen and f_.running():
                seen[f_] = now
        for f_ in sorted(fin, key=lambda x: futs[x]):
            i = futs[f_]
            out[i], dt = f_.result()
            done_n += 1
            last = now
            if every:
                print(f"# {what} {done_n}/{n} done ({100 * done_n // n} %): {labels[i]} in {dt:.1f} s; "
                      f"{nrun()} running; {clock(done_n)}", flush=True)
        if every and not fin and now - last >= every:
            run = [x for x in pending if x.running()]
            old = min(run, key=lambda x: seen.get(x, now)) if run else None
            print(f"# {what} {done_n}/{n} done; {nrun()} running"
                  + (f", longest {(now - seen.get(old, now)) / 60:.1f} min: {labels[futs[old]]}" if old else "")
                  + f"; {clock(done_n)}", flush=True)
            last = now
    return out


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


def shor9_bench(shots, gpu=False, exact=True, repeats=1, m=3, jobs=1, brep=False, every=30.0):
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
    nch = len(ex_keys) // len(SHOR9_STATES) if ex_keys else 0
    labels = ([f"{name} |{s}_L> backend {r + 1}/{R}" for name, s, r in keys]
              + [f"exact |{s}_L> chunk {k % nch + 1}/{nch} ({c} shots)" for k, (s, c) in enumerate(ex_keys)])
    if every:
        print(f"# shor9 plan: Shor-{n}, {len(lays)} layout(s) x {len(SHOR9_STATES)} states x {R} backend(s) = {len(keys)} "
              f"ACE tasks of {shots} shots" + (f", exact reference in {len(ex_keys)} chunks" if ex_keys else "")
              + f"; {workers} process(es)", flush=True)
    try:
        out = shor_pool_map(pool, [(_shor9_ace_task, (n, geo, gpu, lays[name], s, shots, brep)) for name, s, _ in keys]
                            + [(_shor9_exact_task, (n, gpu, lays["seam_leaders"], s, c)) for s, c in ex_keys],
                            labels, every)
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
        if reg != "release" and reg not in REGISTERS:
            load_register_files()               # a spawned worker: saved registers ($NIGHTHAWK_REGISTERS, ...)
        lay = _SHOR_LAYOUTS[(repo, reg)] = Layout(repo) if reg == "release" else RegisterLayout(repo, reg)
    if exact:
        from pyqrack import QrackSimulator
        return _shor9_one_run(QrackSimulator(lay.n, is_gpu=gpu), list(range(lay.n)), tiles, rot, shots)
    THETA_DIST["mode"] = ra.theta
    eng = AceEngine(lay, lay.n, ra)
    out = _shor9_one_run(eng.sim, eng.idx, tiles, rot, shots)
    del eng
    return out


def shor9_run_args(reg, repo=None, gpu=False, brep=False):
    """The `run` settings shor9 --register uses on a register: its ACE preset placement."""
    if "r" not in _RUN_PARSER:
        make_parser()
    ra = _RUN_PARSER["r"].parse_args(["--backend", "ace", "--register", reg] + (["--repo", repo] if repo else [])
                                     + (["--ace-gpu"] if gpu else ["--cpu"])
                                     + (["--ace-boundary-rep"] if brep else []))
    ra.cmd = "run"
    THETA_DIST["mode"] = ra.theta
    return ra


def shor9_register_map(lay, ra):
    """The register on the lattice as ACE places it: per logical qubit (row, column, region: p<patch> or
    seam) and per coupler its class (exact inside one simulator, replica through a seam qubit's copy,
    cross between simulators that share no seam), and for a seam qubit the patches it joins. Returns (map,
    sids, idx, row); map is what the shor9_map
    record stores and nighthawk_graph.py draws."""
    probe = AceEngine(lay, lay.n, ra)
    row = layout_row(probe.size, probe.lrc, probe.lrr, probe.torus, probe.opts)
    sids = [frozenset(s_ for s_, _ in u) for u in row["unpack"]]
    idx = list(probe.idx)
    del probe
    region = lambda q: "seam" if len(sids[idx[q]]) > 1 else f"p{next(iter(sids[idx[q]]))}"

    def cls(c, t):
        sa, sb = sids[idx[c]], sids[idx[t]]
        return "exact" if len(sa) == 1 and sa == sb else "replica" if sa & sb else "cross"
    C = 10
    # sites: [qubit, row, column, region, the ACE simulators holding it (two or more for a seam qubit)]
    mp = dict(n=lay.n, columns=C, sites=[[q, p // C, p % C, region(q), sorted(int(x) for x in sids[idx[q]])]
                                         for q, p in enumerate(lay.l2p[:lay.n])],
              couplers=[[a_, b_, cls(a_, b_)] for a_, b_ in logical_couplers(lay, lay.n)],
              ace=dict(lrc=row["lrc"], lrr=row["lrr"], torus=bool(row["torus"]), width=len(row["unpack"])))
    return mp, sids, idx, row


def cmd_shor9_register(a):
    """Units tiled over a whole register, every unit in one circuit on the register's own ACE placement;
    logical states rotate over the units in four runs, so every unit is read in all four."""
    reg, m = a.register, int(a.size)
    lay = Layout(a.repo) if reg == "release" else RegisterLayout(a.repo, reg)
    _SHOR_LAYOUTS[(a.repo, reg)] = lay
    n = lay.n
    every = float(a.progress or 0)
    if every:
        print(f"# shor9 start {time.strftime('%Y-%m-%d %H:%M:%S')}: Shor-{m * m} units on {reg} ({n} qubits), {a.shots} "
              f"shots, {a.repeats} backend(s) per rotation, {'OpenCL' if a.gpu else 'CPU'}, --jobs {a.jobs}"
              f"{', boundary repetition code' if a.ace_boundary_rep else ''}; {qrack_versions()}", flush=True)
        print(f"# shor9 placing {m} x {m} units (CP-SAT, up to 30 s)", flush=True)
    tiles = shor9_tiles(lay, a.max_tiles, m)
    if not tiles:
        raise SystemExit(f"shor9: no {m} x {m} square of {reg} is free of holes")
    ra = shor9_run_args(reg, a.repo, bool(a.gpu), bool(getattr(a, "ace_boundary_rep", False)))
    apply_register_preset(ra, lay)
    lmap, sids, idx, row = shor9_register_map(lay, ra)
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
    nch = len(ex_keys) // 4 if ex_keys else 0
    labels = ([f"ACE rotation {rot + 1}/4 backend {r + 1}/{R}" for rot, r in keys]
              + [f"exact rotation {rot + 1}/4 chunk {k % nch + 1}/{nch} ({c} shots)" for k, (rot, c) in enumerate(ex_keys)])
    if every:
        print(f"# shor9 plan: {len(tiles)} units, 4 rotations x {R} backend(s) = {len(keys)} ACE tasks of {a.shots} shots"
              + (f", exact reference ({n} qubits) in {len(ex_keys)} chunks" if ex_keys else "") + f"; {workers} process(es)",
              flush=True)
    try:
        out = shor_pool_map(pool, [(_shor9_reg_task, (a.repo, reg, ra, tiles, rot, a.shots, False, gpu))
                                   for rot, _ in keys]
                            + [(_shor9_reg_task, (a.repo, reg, ra, tiles, rot, c, True, gpu)) for rot, c in ex_keys],
                            labels, every)
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
        when = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(a.out, "a") as fh:
            # the register as placed (sites, regions, coupler classes), once per run: nighthawk_graph.py view 17
            fh.write(json.dumps(dict(test="shor9_map", register=reg, size=m, qrack=ver,
                                     boundary_rep=bool(getattr(a, "ace_boundary_rep", False)), when=when, **lmap)) + "\n")
            for r in recs:
                fh.write(json.dumps(dict(r, when=when)) + "\n")
        print(f"  appended {len(recs)} records and the register map to {a.out}")
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
    if a.progress:
        print(f"# shor9 start {time.strftime('%Y-%m-%d %H:%M:%S')}: Shor-{m * m}, {a.shots} shots, {a.repeats} backend(s) "
              f"per state, {'OpenCL' if gpu else 'CPU'}, --jobs {a.jobs}"
              f"{', boundary repetition code' if a.ace_boundary_rep else ''}; {qrack_versions()}", flush=True)
    results, geo, workers = shor9_bench(a.shots, gpu=gpu, exact=not a.no_exact, repeats=a.repeats, m=m,
                                        jobs=getattr(a, "jobs", 1), brep=bool(getattr(a, "ace_boundary_rep", False)),
                                        every=float(getattr(a, "progress", 30.0) or 0))
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

    # [14] the horizontal bridge: exact against QrackSimulator at full capacity; = PyQrack's QrackBridge (state and
    # shots, same seed) while truncating, on a chain, a ring and with lazy bonds, patch p on server p mod 2
    worst_ex, same, cases = 1.0, True, 0
    QB = local_qrack_bridge()
    for n_, P_, wrap_, mb_, lazy_ in ((10, 2, False, 6, 0), (12, 4, True, 6, 0), (12, 2, False, 1, 0),
                                      (12, 4, True, 1, 0), (12, 3, False, 2, 2)):
        ops = _bb_circuit(n_, P_, 5, 0.3, 5 + n_ + P_, wrap_)
        hb = HBridge(n_, max_bonds=mb_, patches=P_, wrap=wrap_, lazy_bonds=lazy_, seed=3, hbridge="devices=cpu,cpu")
        _bb_apply(hb, ops)
        vh = np.asarray(hb.out_ket())
        if mb_ >= 6:
            ref = QrackSimulator(n_, is_gpu=False)
            for op in ops:
                if op[0] == "u":
                    ref.u(op[1], *op[2:])
                else:
                    ref.mcx([op[1]], op[2])
            worst_ex = min(worst_ex, abs(np.vdot(vh, np.asarray(ref.out_ket()))) ** 2)
        if QB is not None:
            lb = QB(n_, max_bonds=mb_, patches=P_, wrap=wrap_, lazy_bonds=lazy_, seed=3)
            _bb_apply(lb, ops)
            ov = abs(np.vdot(vh, np.asarray(lb.out_ket()))) ** 2
            same &= ov > 1 - 1e-5 and hb.measure_shots(list(range(n_)), 200) == lb.measure_shots(list(range(n_)), 200)
            same &= abs(hb.fidelity_estimate - lb.fidelity_estimate) < 1e-6 and hb.truncations == lb.truncations
        hb.close()
        cases += 1
    report("[14] horizontal QrackBridge (patch servers + glue server): exact at full capacity (chain, ring); "
           "= PyQrack's QrackBridge while truncating (state, shots, fidelity estimate, truncations)",
           worst_ex > 1 - 1e-5 and same, f"{cases} circuits; worst overlap with QrackSimulator {worst_ex:.7f}; "
           + ("same as QrackBridge" if (QB is not None and same) else "QrackBridge not found: not compared"
              if QB is None else "DIFFERS from QrackBridge"))
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


# ================================================================== horizontal QrackBridge (HBridge)
#
# QrackBridge (PyQrack >= 2.30, qrack_bridge.py; by Claude and Dan Strano) keeps every patch as one QrackSimulator
# and does its glue -- the per-cut recompression and the readout's contraction -- with numpy, all in the caller's
# process. HBridge is the same algorithm split by workload over separate processes:
#
#   patch servers   each its own PyQrack instance (its own process, environment, OpenCL device or CPU, CPU
#                   affinity), holding any number of patch simulators: data qubits, bond qubits and scratch
#                   qubits. Every gate of the bridge is local to one patch (a gate between patches is teleported
#                   through fresh bond pairs whose Bell projection is applied later, at contraction), so the
#                   servers run their op streams in parallel and never talk to each other.
#   glue server     one process with no Qrack at all: numpy / BLAS only (its own threads and affinity). It holds
#                   the truncation and sampling RNG of every bridge and does the recompression sweeps, the
#                   ring's wrap-cut step, the sampling and the contraction from the patch kets.
#   coordinator     the caller: the bridge's bookkeeping (layout, bonds per cut, orthogonality centre, touched
#                   patches), batching ops per server; at a recompression it has the touched patches export
#                   their kets into shared memory, the glue rewrite them in place, the patches reload them.
#
# The algorithm, its order of operations and its use of the RNG are QrackBridge's, so a run with the same seed
# gives the same state and the same shots (selftest, bridgebench --check). Servers are started once per process and
# configuration (spec) and shared by every bridge made there; each server records what it spent its time on, so
# the gate workload, the ket traffic and the linear algebra can be measured and tuned apart (bridgebench).

HB_TOL = 1e-6
HB_SPEC_DEFAULT = "devices=cpu,cpu;glue_threads=1;batch=512"


def _hb_bits(dim):
    return (max(int(dim), 1) - 1).bit_length()


def _hb_gram_svd(m):
    lam, u = np.linalg.eigh(m @ m.conj().T)
    order = np.argsort(lam)[::-1]
    return u[:, order], np.sqrt(np.clip(lam[order], 0, None))


def _hb_left_factor(a):
    lam, v = np.linalg.eigh(a.conj().T @ a)
    keep = lam > (HB_TOL * HB_TOL) * lam[-1]
    keep[-1] = True
    s = np.sqrt(lam[keep])
    v = v[:, keep]
    return (a @ v) / s[None, :], s[:, None] * v.conj().T


def parse_hbridge(spec=None):
    """The horizontal bridge's configuration from a spec (default $NIGHTHAWK_HBRIDGE, else HB_SPEC_DEFAULT):
    'key=value;...' or JSON. Keys:
      devices       patch servers, one per entry: cpu or gpu:N (OpenCL device N, isolated as run --gpus isolates
                    its workers); patch p of every bridge on server p mod S (default cpu,cpu)
      patch_cpus    CPU affinity per patch server, '/'-separated, each a list such as 0-3 or 0,2 (Linux)
      glue_cpus     CPU affinity of the glue server
      glue_threads  BLAS / OpenMP threads of the glue server (default 1)
      patch_threads OpenMP threads of each patch server (Qrack's CPU engine), unset by default
      host_pointer  1: gpu servers' simulators with is_host_pointer (state in host memory)
      batch         ops queued per server before they are sent (default 512)
      glue_fast     1: the readout's per-prefix contraction as one BLAS matmul (threaded by glue_threads) instead
                    of QrackBridge's einsum; same algorithm, sums in another order, so shots can differ from
                    QrackBridge's in the last bits of a probability (default 0: QrackBridge's own arithmetic)
    """
    spec = spec if spec is not None else os.environ.get("NIGHTHAWK_HBRIDGE") or HB_SPEC_DEFAULT
    if isinstance(spec, dict):
        d = dict(spec)
    elif spec.strip().startswith("{"):
        d = json.loads(spec)
    else:
        d = {}
        for part in spec.split(";"):
            if part.strip():
                k, _, v = part.partition("=")
                d[k.strip()] = v.strip()
    cfg = dict(devices=["cpu", "cpu"], patch_cpus=[], glue_cpus=None, glue_threads=1, patch_threads=None,
               host_pointer=False, batch=512, glue_fast=False)
    for k, v in d.items():
        if k not in cfg:
            raise SystemExit(f"hbridge spec: unknown key {k!r} (keys: {', '.join(cfg)})")
        if k == "devices":
            cfg[k] = [x.strip() for x in (v.split(",") if isinstance(v, str) else v) if x.strip()]
        elif k == "patch_cpus":
            cfg[k] = [x.strip() for x in (v.split("/") if isinstance(v, str) else v)]
        elif k in ("glue_threads", "batch", "patch_threads"):
            cfg[k] = None if v in (None, "", "none") else int(v)
        elif k in ("host_pointer", "glue_fast"):
            cfg[k] = str(v).lower() in ("1", "true", "yes", "on")
        else:
            cfg[k] = v
    for dev in cfg["devices"]:
        if dev != "cpu" and not re.fullmatch(r"gpu:\d+", dev):
            raise SystemExit(f"hbridge spec: device {dev!r} (cpu or gpu:N)")
    if not cfg["devices"]:
        raise SystemExit("hbridge spec: at least one patch server")
    return cfg


def _hb_cpus(s):
    out = set()
    for part in str(s).split(","):
        if "-" in part:
            a_, b_ = part.split("-")
            out.update(range(int(a_), int(b_) + 1))
        elif part.strip():
            out.add(int(part))
    return out


def _hb_attach(name):
    """Attach to a coordinator's shared-memory segment without letting this process's resource tracker unlink it."""
    from multiprocessing import shared_memory
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:                                   # Python < 3.13
        shm = shared_memory.SharedMemory(name=name)
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
        return shm


# ------------------------------------------------------------------ servers
def hbridge_server(argv):
    """Entry of a server process: nighthawk_qrack.py _hbridge-server ROLE INDEX ADDRESS AUTHKEY CONFIG."""
    from multiprocessing.connection import Client
    role, index, address, key, cfg = argv[0], int(argv[1]), argv[2], bytes.fromhex(argv[3]), json.loads(argv[4])
    cpus = cfg.get("cpus")
    if cpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, _hb_cpus(cpus))
    conn = Client(address, family="AF_UNIX", authkey=key)
    try:
        (_hb_patch_server if role == "patch" else _hb_glue_server)(conn, index, cfg)
    except (EOFError, ConnectionError, KeyboardInterrupt):
        pass


def _hb_patch_server(conn, index, cfg):
    import ctypes
    from pyqrack import QrackSimulator
    from pyqrack.qrack_system import Qrack
    real = np.float32 if Qrack.fppow < 6 else np.float64
    creal = ctypes.c_float if real is np.float32 else ctypes.c_double
    kw = dict(is_gpu=cfg["device"] != "cpu")
    if cfg.get("host_pointer") and kw["is_gpu"]:
        kw["is_host_pointer"] = True
    sims = {}
    st = dict(role="patch", index=index, pid=os.getpid(), device=cfg["device"], cpus=cfg.get("cpus"), ops=0,
              op_s=0.0, by_op={}, exports=0, export_s=0.0, export_bytes=0, loads=0, load_s=0.0, load_bytes=0,
              idle_s=0.0, sims=0, peak_qubits=0, ocl=os.environ.get("QRACK_OCL_DEFAULT_DEVICE"))
    conn.send(("hello", "patch", index, dict(pid=os.getpid(), fppow=int(Qrack.fppow), device=cfg["device"])))
    err = None
    while True:
        t0 = time.perf_counter()
        msg = conn.recv()
        t1 = time.perf_counter()
        st["idle_s"] += t1 - t0
        kind = msg[0]
        if kind == "quit":
            return
        if kind == "ops":
            if err is not None:
                continue
            try:
                for op in msg[1]:
                    key, name = op[0], op[1]
                    a0 = time.perf_counter()
                    if name == "new":
                        sims[key] = QrackSimulator(op[2], **kw)
                        st["sims"] += 1
                    elif name == "free":
                        sims.pop(key, None)
                    elif name == "clone":
                        sims[op[2]] = sims[key].clone()
                    elif name == "alloc":
                        sims[key].allocate_qubit(op[2])
                        st["peak_qubits"] = max(st["peak_qubits"], int(op[2]) + 1)
                    else:
                        getattr(sims[key], name)(*op[2:])
                    dt = time.perf_counter() - a0
                    st["ops"] += 1
                    st["op_s"] += dt
                    b = st["by_op"].setdefault(name, [0, 0.0])
                    b[0] += 1
                    b[1] += dt
            except Exception:
                import traceback
                err = traceback.format_exc()
            continue
        try:
            if err is not None:
                raise RuntimeError("an earlier op failed:\n" + err)
            if kind == "export":                       # [(key, shm name)]: each patch ket into its segment
                out = []
                for key, name in msg[1]:
                    a0 = time.perf_counter()
                    s = sims[key]
                    n = s.num_qubits()
                    shm = _hb_attach(name)
                    buf = np.ndarray((2 << n,), dtype=real, buffer=shm.buf)
                    Qrack.qrack_lib.OutKet(s.sid, buf.ctypes.data_as(ctypes.POINTER(creal)))
                    s._throw_if_error()
                    del buf
                    shm.close()
                    st["exports"] += 1
                    st["export_bytes"] += (2 << n) * np.dtype(real).itemsize
                    st["export_s"] += time.perf_counter() - a0
                    out.append(n)
                conn.send(("ok", out))
            elif kind == "load":                       # [(key, shm name, qubits)]: a fresh patch simulator from it
                for key, name, n in msg[1]:
                    a0 = time.perf_counter()
                    shm = _hb_attach(name)
                    buf = np.ndarray((2 << n,), dtype=real, buffer=shm.buf)
                    s = QrackSimulator(n, **kw)
                    Qrack.qrack_lib.InKet(s.sid, buf.ctypes.data_as(ctypes.POINTER(creal)))
                    s._throw_if_error()
                    sims[key] = s
                    del buf
                    shm.close()
                    st["loads"] += 1
                    st["load_bytes"] += (2 << n) * np.dtype(real).itemsize
                    st["load_s"] += time.perf_counter() - a0
                    st["peak_qubits"] = max(st["peak_qubits"], n)
                conn.send(("ok", None))
            elif kind == "stats":
                conn.send(("ok", dict(st, live=len(sims), by_op={k: list(v) for k, v in st["by_op"].items()})))
            elif kind == "ping":
                conn.send(("ok", None))
            else:
                raise ValueError(f"patch server: {kind}")
        except Exception:
            import traceback
            conn.send(("error", traceback.format_exc()))
            err = None


def _hb_tensor(v, lay, bl, br, nd):
    """A patch ket as A[l, d, r] (QrackBridge._tensor)."""
    k = len(lay)
    order = ([("L", j) for j in reversed(range(bl))] + [("d", i) for i in reversed(range(nd))]
             + [("R", j) for j in reversed(range(br))])
    t = v.reshape([2] * k)
    if any(tag[0] == "0" for tag in lay):
        t = t[tuple(0 if lay[k - 1 - x][0] == "0" else slice(None) for x in range(k))]
        lay = [tag for tag in lay if tag[0] != "0"]
        k = len(lay)
    pos = {tag: x for x, tag in enumerate(lay)}
    t = t.transpose([k - 1 - pos[tag] for tag in order])
    return t.reshape(1 << bl, 1 << nd, 1 << br)


class _HBGlue:
    """The glue's work for one call: tensors in from shared memory, QrackBridge's sweeps, tensors out."""

    def __init__(self, rng, max_bonds, truncation):
        self.rng, self.max_bonds, self.truncation = rng, max_bonds, truncation
        self.fid, self.trunc, self.fast = 1.0, 0, False

    def select(self, sv, exact_only):
        if sv[0] <= 0:
            return np.arange(1)
        live = int(np.sum(sv > HB_TOL * sv[0]))
        b_exact = _hb_bits(live)
        b_new = b_exact if exact_only else min(b_exact, self.max_bonds)
        keep = min(live, 1 << b_new)
        if keep >= live:
            return np.arange(keep)
        w = sv[:live] ** 2
        self.trunc += 1
        if self.truncation == "greedy":
            self.fid *= float(np.sum(w[:keep]) / np.sum(w))
            return np.arange(keep)
        groups = [np.arange(i, min(i + keep, live)) for i in range(0, live, keep)]
        pg = np.array([np.sum(w[g]) for g in groups]) / np.sum(w)
        self.fid *= float(np.sum(pg ** 2))
        return groups[int(self.rng.choice(len(groups), p=pg))]


def _hb_read(pt, cdt):
    shm = _hb_attach(pt["shm"])
    v = np.ndarray((1 << len(pt["lay"]),), dtype=cdt, buffer=shm.buf).astype(complex)
    shm.close()
    return _hb_tensor(v, [tuple(x) for x in pt["lay"]], pt["bl"], pt["br"], pt["nd"])


def _hb_write(pt, a, bl, br, cdt):
    nd = pt["nd"]
    full = np.zeros((1 << bl, 1 << nd, 1 << br), dtype=complex)
    full[: a.shape[0], :, : a.shape[2]] = a
    v = full.transpose(2, 0, 1).reshape(-1)
    v = v / np.linalg.norm(v)
    shm = _hb_attach(pt["shm"])
    out = np.ndarray((v.size,), dtype=cdt, buffer=shm.buf)
    out[:] = v.astype(cdt)
    del out
    shm.close()
    return nd + bl + br


def _hb_glue_server(conn, index, cfg):
    rngs = {}
    st = dict(role="glue", index=index, pid=os.getpid(), cpus=cfg.get("cpus"), threads=cfg.get("threads"),
              calls={}, idle_s=0.0, read_bytes=0)
    conn.send(("hello", "glue", index, dict(pid=os.getpid())))
    while True:
        t0 = time.perf_counter()
        msg = conn.recv()
        t1 = time.perf_counter()
        st["idle_s"] += t1 - t0
        kind = msg[0]
        if kind == "quit":
            return
        try:
            if kind == "seed":
                rngs[msg[1]] = np.random.default_rng(msg[2])
                continue
            if kind == "drop":
                rngs.pop(msg[1], None)
                continue
            if kind == "rngclone":
                rngs[msg[2]] = np.random.default_rng(int(rngs[msg[1]].integers(1 << 62)))
                continue
            if kind == "stats":
                conn.send(("ok", dict(st, live=len(rngs), calls={k: list(v) for k, v in st["calls"].items()})))
                continue
            key, pl = msg[1], msg[2]
            cdt = np.complex64 if pl["dtype"] == "f4" else np.complex128
            for pt in pl.get("patches", []):
                st["read_bytes"] += (1 << len(pt["lay"])) * np.dtype(cdt).itemsize
            g = _HBGlue(rngs[key], pl.get("max_bonds"), pl.get("truncation", "greedy"))
            g.fast = bool(cfg.get("fast"))
            if kind == "sweep":
                res = _hb_glue_sweep(g, pl, cdt)
            elif kind == "wrap":
                res = _hb_glue_wrap(g, pl, cdt)
            elif kind == "sample":
                res = _hb_glue_sample(g, pl, cdt)
            elif kind == "contract":
                res = _hb_glue_contract(pl, cdt)
            else:
                raise ValueError(f"glue server: {kind}")
            c = st["calls"].setdefault(kind, [0, 0.0])
            c[0] += 1
            c[1] += time.perf_counter() - t1
            conn.send(("ok", res))
        except Exception:
            import traceback
            conn.send(("error", traceback.format_exc()))


def _hb_glue_sweep(g, pl, cdt):
    """QrackBridge._sweep over patches start..end, the centre ending at lo; tensors written back in place."""
    start, end, lo, exact = pl["start"], pl["end"], pl["lo"], pl["exact_only"]
    pts = {pt["p"]: pt for pt in pl["patches"]}
    T = {k: _hb_read(pts[k], cdt) for k in range(start, end + 1)}
    bonds = {}
    for k in range(start, end):
        l, d, r = T[k].shape
        q, rr = _hb_left_factor(T[k].reshape(l * d, r))
        T[k] = q.reshape(l, d, -1)
        nl, nd_, nr = T[k + 1].shape
        T[k + 1] = (rr @ T[k + 1].reshape(nl, nd_ * nr)).reshape(-1, nd_, nr)
        bonds[k] = _hb_bits(q.shape[1])
    for k in range(end, lo, -1):
        l, d, r = T[k].shape
        m = T[k].reshape(l, d * r)
        u, sv = _hb_gram_svd(m)
        idx = g.select(sv, exact)
        us = u[:, idx]
        T[k] = ((us.conj().T @ m) / sv[idx][:, None]).reshape(len(idx), d, r)
        pl_, pd, pr = T[k - 1].shape
        T[k - 1] = (T[k - 1].reshape(pl_ * pd, pr) @ (us * sv[idx][None, :])).reshape(pl_, pd, -1)
        bonds[k - 1] = _hb_bits(len(idx))
    nq_ = {}
    for k, a in T.items():
        pt = pts[k]
        bl = bonds.get(k - 1, pt["bl"]) if k - 1 >= start else pt["bl"]
        br = bonds.get(k, pt["br"]) if k <= end - 1 else pt["br"]
        nq_[k] = _hb_write(pt, a, bl, br, cdt)
    return dict(bonds=bonds, nq=nq_, fid=g.fid, trunc=g.trunc)


def _hb_glue_wrap(g, pl, cdt):
    """QrackBridge._compress_wrap: the ring's wrap cut between patches P-1 (a) and 0 (b)."""
    pa, pb = pl["patches"]
    a, b = _hb_read(pa, cdt), _hb_read(pb, cdt)
    l, d, w = a.shape
    _, d0, r0 = b.shape
    qa, ra = np.linalg.qr(a.reshape(l * d, w))
    qb, rb = np.linalg.qr(b.reshape(w, d0 * r0).T)
    u, sv, vh = np.linalg.svd(ra @ rb.T)
    idx = g.select(sv, pl["exact_only"])
    wb = _hb_bits(len(idx))
    na = _hb_write(pa, ((qa @ u[:, idx]) * sv[idx][None, :]).reshape(l, d, -1), pa["bl"], wb, cdt)
    nb = _hb_write(pb, (qb @ vh[idx, :].T).T.reshape(-1, d0, r0), wb, pb["br"], cdt)
    return dict(wrap=wb, nq={pa["p"]: na, pb["p"]: nb}, fid=g.fid, trunc=g.trunc)


def _hb_glue_sample(g, pl, cdt):
    """QrackBridge.measure_shots after its compression: sequential conditional sampling, patch by patch."""
    T = [_hb_read(pt, cdt) for pt in pl["patches"]]
    P, shots, starts, qubits, ring = len(T), pl["shots"], pl["starts"], pl["qubits"], pl["ring"]
    rng = g.rng
    R = None
    if ring:
        cw = T[0].shape[0]
        eye = np.eye(cw)
        R = [None] * (P + 1)
        R[P] = np.einsum("rw,sv->rwsv", eye, eye)
        for k in range(P - 1, 0, -1):
            R[k] = np.einsum("ldr,rwsv,mds->lwmv", T[k], R[k + 1], T[k].conj(), optimize=True)
        groups = [(eye, shots, 0)]
    else:
        groups = [(np.ones((1, 1)), shots, 0)]
    for k in range(P):
        nxt = []
        for X, cnt, bits in groups:
            if g.fast:
                M = (X @ T[k].reshape(T[k].shape[0], -1)).reshape(X.shape[0], T[k].shape[1], T[k].shape[2])
            else:
                M = np.einsum("wl,ldr->wdr", X, T[k])
            if R is None:
                p = np.sum(np.abs(M) ** 2, axis=(0, 2))
            else:
                p = np.einsum("wdr,rwsv,vds->d", M, R[k + 1], M.conj(), optimize=True).real
            p = np.clip(p, 0, None)
            p /= p.sum()
            for d, c in enumerate(rng.multinomial(cnt, p)):
                if c:
                    Y = M[:, d, :]
                    nxt.append((Y / np.linalg.norm(Y), int(c), bits | (d << starts[k])))
        groups = nxt
    out = []
    for _, c, full in groups:
        out += [sum(((full >> q) & 1) << j for j, q in enumerate(qubits))] * c
    rng.shuffle(out)
    return out


def _hb_glue_contract(pl, cdt):
    C = None
    for pt in pl["patches"]:
        A = _hb_read(pt, cdt)
        if C is None:
            C = np.einsum("wl,ldr->wdr", np.eye(A.shape[0]), A)
        else:
            w, D, _ = C.shape
            C = np.einsum("wDr,rdq->wdDq", C, A).reshape(w, -1, A.shape[2])
    v = np.einsum("wDw->D", C)
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------ service (the servers of one configuration)
_HB_SERVICES = {}


class HBService:
    """The patch servers and the glue server of one configuration, started on first use in this process and shared
    by every HBridge made here; stopped at exit."""

    def __init__(self, cfg):
        import secrets
        import subprocess
        import tempfile
        from multiprocessing.connection import Listener
        self.cfg = cfg
        self.dir = tempfile.mkdtemp(prefix="nh-hb-", dir="/tmp" if os.path.isdir("/tmp") else None)
        self.address = os.path.join(self.dir, "s")
        key = secrets.token_bytes(16)
        self.listener = Listener(self.address, family="AF_UNIX", authkey=key)
        me = str(Path(__file__).resolve())
        self.procs = []
        roles = [("patch", i) for i in range(len(cfg["devices"]))] + [("glue", 0)]
        for role, i in roles:
            if role == "patch":
                dev = cfg["devices"][i]
                env = dict(os.environ) if dev == "cpu" else isolated_env(int(dev.split(":")[1]))
                if cfg.get("patch_threads"):
                    env["OMP_NUM_THREADS"] = str(cfg["patch_threads"])
                c = dict(device=dev, host_pointer=cfg.get("host_pointer"),
                         cpus=(cfg["patch_cpus"][i] if i < len(cfg["patch_cpus"]) else None))
            else:
                env = dict(os.environ)
                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
                    env[k] = str(cfg.get("glue_threads") or 1)
                c = dict(cpus=cfg.get("glue_cpus"), threads=cfg.get("glue_threads"), fast=cfg.get("glue_fast"))
            env["NIGHTHAWK_ISOLATED"] = env.get("NIGHTHAWK_ISOLATED", "hbridge")
            self.procs.append(subprocess.Popen([sys.executable, me, "_hbridge-server", role, str(i), self.address,
                                                key.hex(), json.dumps(c)], env=env))
        self.conn, self.info = {}, {}
        for _ in roles:
            cn = self.listener.accept()
            hello = cn.recv()
            self.conn[(hello[1], hello[2])] = cn
            self.info[(hello[1], hello[2])] = hello[3]
        fp = {v["fppow"] for k, v in self.info.items() if k[0] == "patch"}
        if len(fp) != 1:
            raise SystemExit(f"hbridge: patch servers disagree on Qrack's precision {fp}")
        self.real = "f4" if fp.pop() < 6 else "f8"
        self.isz = 4 if self.real == "f4" else 8
        self.patch = [self.conn[("patch", i)] for i in range(len(cfg["devices"]))]
        self.glue = self.conn[("glue", 0)]
        self.q = [[] for _ in self.patch]
        self.next_key = 0
        self.wait_s = 0.0
        self.sent_ops = 0
        self.batches = 0
        import atexit
        atexit.register(self.close)

    def new_key(self):
        self.next_key += 1
        return self.next_key

    def pick(self, P):
        """Servers for a bridge's P patches: patch p on server p mod S, the same for every bridge, so a workload that
        always sits on one patch (one chip, one side of a cut) always meets the same server and device."""
        return [p % len(self.patch) for p in range(P)]

    def op(self, i, op):
        q = self.q[i]
        q.append(op)
        if len(q) >= self.cfg["batch"]:
            self.flush(i)

    def flush(self, i):
        if self.q[i]:
            self.patch[i].send(("ops", self.q[i]))
            self.sent_ops += len(self.q[i])
            self.batches += 1
            self.q[i] = []

    def request(self, i, msg):
        self.flush(i)
        self.patch[i].send(msg)

    def _recv(self, cn):
        t0 = time.perf_counter()
        r = cn.recv()
        self.wait_s += time.perf_counter() - t0
        if r[0] == "error":
            raise RuntimeError("hbridge server failed:\n" + r[1])
        return r[1]

    def reply(self, i):
        return self._recv(self.patch[i])

    def glue_call(self, kind, key, payload):
        self.glue.send((kind, key, payload))
        return self._recv(self.glue)

    def glue_note(self, msg):
        self.glue.send(msg)

    def stats(self):
        out = []
        for i in range(len(self.patch)):
            self.request(i, ("stats",))
            out.append(self.reply(i))
        self.glue.send(("stats",))
        out.append(self._recv(self.glue))
        return dict(servers=out, coordinator=dict(wait_s=self.wait_s, sent_ops=self.sent_ops, batches=self.batches))

    def close(self):
        for cn in list(self.conn.values()):
            try:
                cn.send(("quit",))
                cn.close()
            except Exception:
                pass
        for p in self.procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
        self.conn = {}
        try:
            self.listener.close()
            import shutil
            shutil.rmtree(self.dir, ignore_errors=True)
        except Exception:
            pass


def hbridge_service(spec=None):
    cfg = parse_hbridge(spec)
    k = json.dumps(cfg, sort_keys=True)
    if k not in _HB_SERVICES:
        _HB_SERVICES[k] = HBService(cfg)
    return _HB_SERVICES[k]


class _HBPatch:
    """A patch simulator on a patch server, as the bridge's bookkeeping sees it: every call is queued there."""
    __slots__ = ("svc", "i", "key")

    def __init__(self, svc, i, key):
        self.svc, self.i, self.key = svc, i, key

    def _op(self, name, *args):
        self.svc.op(self.i, (self.key, name) + args)

    def allocate_qubit(self, qid):
        self._op("alloc", qid)

    def __getattr__(self, name):
        if name in ("u", "x", "h", "s", "adjs", "swap", "iswap", "adjiswap", "mcx", "macx", "mcmtrx", "macmtrx"):
            return lambda *args: self._op(name, *args)
        raise AttributeError(name)


class HBridge:
    """QrackBridge with its patches on patch servers and its glue on the glue server (see the section above). Same
    constructor and interface: u, x, h, s, adjs, mcx / mcy / mcz / macx / macy / macz, cx / cy / cz, swap, iswap,
    adjiswap, flush, measure_shots, out_ket, clone, and fidelity_estimate, truncations, flushes, peak_bonds, bond.
    hbridge: a spec (parse_hbridge) or None for $NIGHTHAWK_HBRIDGE."""

    def __init__(self, qubit_count, max_bonds=6, seed=None, truncation="greedy", patches=2, wrap=False,
                 lazy_bonds=0, max_patch_qubits=None, hbridge=None, _clone=None):
        if _clone is not None:
            src = _clone
            for k in ("n", "P", "ring", "max_bonds", "lazy_bonds", "max_patch_qubits", "truncation",
                      "fidelity_estimate", "truncations", "flushes", "peak_bonds", "center", "_wrap_touched", "svc",
                      "srv"):
                setattr(self, k, getattr(src, k))
            self.part = [list(x) for x in src.part]
            self.start = list(src.start)
            self.side = dict(src.side)
            self.layout = [list(x) for x in src.layout]
            self.bond = list(src.bond)
            self._touched = set(src._touched)
            self.key = self.svc.new_key()
            self.sim = []
            for p in range(self.P):
                self.svc.op(self.srv[p], ((src.key, p), "clone", (self.key, p)))
                self.sim.append(_HBPatch(self.svc, self.srv[p], (self.key, p)))
            self.svc.glue_note(("rngclone", src.key, self.key))
            self._t0 = time.perf_counter()
            return
        if truncation not in ("greedy", "born"):
            raise ValueError("truncation must be 'greedy' or 'born'")
        if not 1 <= patches <= qubit_count:
            raise ValueError("need 1 <= patches <= qubit_count")
        self.svc = hbridge_service(hbridge)
        self.key = self.svc.new_key()
        self.n, self.P = qubit_count, patches
        self.ring = bool(wrap) and patches >= 3
        base, rem = divmod(qubit_count, patches)
        sizes = [base + (1 if p >= patches - rem else 0) for p in range(patches)]
        self.start = [sum(sizes[:p]) for p in range(patches)]
        self.part = [list(range(self.start[p], self.start[p] + sizes[p])) for p in range(patches)]
        self.side = {q: (p, i) for p, qs in enumerate(self.part) for i, q in enumerate(qs)}
        self.layout = [[("d", i) for i in range(len(qs))] for qs in self.part]
        self.srv = self.svc.pick(patches)
        self.sim = []
        for p in range(patches):
            self.svc.op(self.srv[p], ((self.key, p), "new", len(self.part[p])))
            self.sim.append(_HBPatch(self.svc, self.srv[p], (self.key, p)))
        self.svc.glue_note(("seed", self.key, seed))
        self.bond = [0] * (patches if self.ring else patches - 1)
        self.max_bonds = max_bonds
        if lazy_bonds < 0:
            raise ValueError("lazy_bonds must be >= 0")
        self.lazy_bonds, self.max_patch_qubits = lazy_bonds, max_patch_qubits
        self.flushes, self.truncation, self.fidelity_estimate = 0, truncation, 1.0
        self.truncations, self.peak_bonds, self.center = 0, 0, 0
        self._touched, self._wrap_touched = set(), False
        self._t0 = time.perf_counter()

    def clone(self):
        return HBridge(None, _clone=self)

    def close(self):
        """Free this bridge's simulators and RNG on the servers (also on garbage collection)."""
        if getattr(self, "svc", None) is None or not self.svc.conn:
            return
        for p in range(self.P):
            self.svc.op(self.srv[p], ((self.key, p), "free"))
        self.svc.glue_note(("drop", self.key))
        self.svc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # --- chain geometry (QrackBridge's) ------------------------------------
    def _nd(self, p):
        return len(self.part[p])

    def _lcut(self, p):
        if p > 0:
            return p - 1
        return self.P - 1 if self.ring else None

    def _rcut(self, p):
        return p if (p < self.P - 1 or self.ring) else None

    def _dims(self, p):
        lc, rc = self._lcut(p), self._rcut(p)
        return (0 if lc is None else self.bond[lc]), (0 if rc is None else self.bond[rc])

    def _path(self, pa, pb):
        step = 1 if pb > pa else -1
        if self.ring:
            right = (pb - pa) % self.P
            step = 1 if right <= self.P - right else -1
        path = [pa]
        while path[-1] != pb:
            path.append((path[-1] + step) % self.P)
        return path

    def _new_bond(self, here, there):
        rightward = there == ((here + 1) % self.P if self.ring else here + 1)
        c = here if rightward else there
        j = self.bond[c]
        self.bond[c] += 1
        self.peak_bonds = max(self.peak_bonds, self.bond[c])
        if self.ring and c == self.P - 1:
            self._wrap_touched = True
        ids = {}
        for p, tag in ((c, ("R", j)), ((c + 1) % self.P, ("L", j))):
            qid = len(self.layout[p])
            self.sim[p].allocate_qubit(qid)
            self.layout[p].append(tag)
            ids[p] = qid
        return ids[here], ids[there]

    def _new_scratch(self, p):
        qid = len(self.layout[p])
        self.sim[p].allocate_qubit(qid)
        self.layout[p].append(("0", qid))
        return qid

    def _ship(self, path, src):
        out, inc = self._new_bond(path[0], path[1])
        self.sim[path[0]].swap(src, out)
        for k in range(1, len(path) - 1):
            s = self.sim[path[k]]
            s.h(inc)
            out, nxt = self._new_bond(path[k], path[k + 1])
            s.mcx([inc], out)
            inc = nxt
        s = self.sim[path[-1]]
        z = self._new_scratch(path[-1])
        s.h(inc)
        s.mcx([inc], z)
        return z

    def _unship(self, path, z, dst):
        out, inc = self._new_bond(path[0], path[1])
        self.sim[path[0]].swap(z, out)
        for k in range(1, len(path) - 1):
            s = self.sim[path[k]]
            s.h(inc)
            out, nxt = self._new_bond(path[k], path[k + 1])
            s.mcx([inc], out)
            inc = nxt
        s = self.sim[path[-1]]
        s.h(inc)
        s.mcx([inc], dst)

    def _loc(self, q):
        p, i = self.side[q]
        return self.sim[p], p, i

    # --- gates (QrackBridge's) ----------------------------------------------
    def u(self, q, th, ph, lm):
        s, _, i = self._loc(q)
        s.u(i, th, ph, lm)

    def x(self, q):
        s, _, i = self._loc(q)
        s.x(i)

    def h(self, q):
        s, _, i = self._loc(q)
        s.h(i)

    def s(self, q):
        s, _, i = self._loc(q)
        s.s(i)

    def adjs(self, q):
        s, _, i = self._loc(q)
        s.adjs(i)

    def _controlled(self, c, t, mtrx, anti):
        sc, pc, ic = self._loc(c)
        st_, pt, it = self._loc(t)
        if pc == pt:
            (sc.macmtrx if anti else sc.mcmtrx)([ic], mtrx, it)
            return
        path = self._path(pc, pt)
        out, inc = self._new_bond(path[0], path[1])
        (sc.macx if anti else sc.mcx)([ic], out)
        for k in range(1, len(path) - 1):
            s = self.sim[path[k]]
            s.h(inc)
            out, nxt = self._new_bond(path[k], path[k + 1])
            s.mcx([inc], out)
            inc = nxt
        st_.h(inc)
        st_.mcmtrx([inc], mtrx, it)
        self._touched.update(path)
        self._maybe_compress()

    def mcx(self, c, t):
        self._controlled(c[0], t, [0, 1, 1, 0], False)

    def mcy(self, c, t):
        self._controlled(c[0], t, [0, -1j, 1j, 0], False)

    def mcz(self, c, t):
        self._controlled(c[0], t, [1, 0, 0, -1], False)

    def macx(self, c, t):
        self._controlled(c[0], t, [0, 1, 1, 0], True)

    def macy(self, c, t):
        self._controlled(c[0], t, [0, -1j, 1j, 0], True)

    def macz(self, c, t):
        self._controlled(c[0], t, [1, 0, 0, -1], True)

    def cx(self, a, b):
        self.mcx([a], b)

    def cy(self, a, b):
        self.mcy([a], b)

    def cz(self, a, b):
        self.mcz([a], b)

    def _pair_gate(self, a, b, op):
        sa, pa, ia = self._loc(a)
        sb, pb, ib = self._loc(b)
        if pa == pb:
            getattr(sa, op)(ia, ib)
            return
        path = self._path(pa, pb)
        z = self._ship(path, ia)
        getattr(sb, op)(z, ib)
        self._unship(path[::-1], z, ia)
        self._touched.update(path)
        self._maybe_compress()

    def swap(self, a, b):
        self._pair_gate(a, b, "swap")

    def iswap(self, a, b):
        self._pair_gate(a, b, "iswap")

    def adjiswap(self, a, b):
        self._pair_gate(a, b, "adjiswap")

    # --- the glue, remote ------------------------------------------------------
    def _export(self, ps):
        """Patch kets of ps into fresh shared-memory segments (the servers in parallel); {p: segment}."""
        from multiprocessing import shared_memory
        shms, by = {}, collections.defaultdict(list)
        for p in ps:
            k = len(self.layout[p])
            shms[p] = shared_memory.SharedMemory(create=True, size=max(16, (2 << k) * self.svc.isz))
            by[self.srv[p]].append(((self.key, p), shms[p].name))
        for i, items in by.items():
            self.svc.request(i, ("export", items))
        for i in by:
            self.svc.reply(i)
        return shms

    def _payload(self, ps, shms):
        return [dict(p=p, shm=shms[p].name, lay=self.layout[p], bl=self._dims(p)[0], br=self._dims(p)[1],
                     nd=self._nd(p)) for p in ps]

    def _load(self, shms, nq):
        by = collections.defaultdict(list)
        for p, n in nq.items():
            by[self.srv[p]].append(((self.key, p), shms[p].name, int(n)))
        for i, items in by.items():
            self.svc.request(i, ("load", items))
        for i in by:
            self.svc.reply(i)
        self._free(shms)

    @staticmethod
    def _free(shms):
        for s in shms.values():
            s.close()
            s.unlink()

    def _relayout(self, p):
        bl, br = self._dims(p)
        self.layout[p] = ([("d", i) for i in range(self._nd(p))] + [("L", j) for j in range(bl)]
                          + [("R", j) for j in range(br)])

    def _sweep(self, lo, hi, exact_only):
        start, end = min(lo, self.center), max(hi, self.center)
        if start == end:
            self.center = lo
            return
        ps = list(range(start, end + 1))
        shms = self._export(ps)
        try:
            r = self.svc.glue_call("sweep", self.key, dict(
                start=start, end=end, lo=lo, exact_only=exact_only, max_bonds=self.max_bonds,
                truncation=self.truncation, dtype=self.svc.real, patches=self._payload(ps, shms)))
        except Exception:
            self._free(shms)
            raise
        for k, b in r["bonds"].items():
            self.bond[int(k)] = int(b)
        self.fidelity_estimate *= r["fid"]
        self.truncations += r["trunc"]
        for p in ps:
            self._relayout(p)
        self._load(shms, {int(k): v for k, v in r["nq"].items()})
        self.center = lo

    def _compress_wrap(self, exact_only):
        P = self.P
        shms = self._export([P - 1, 0])
        try:
            r = self.svc.glue_call("wrap", self.key, dict(exact_only=exact_only, max_bonds=self.max_bonds,
                                                          truncation=self.truncation, dtype=self.svc.real,
                                                          patches=self._payload([P - 1, 0], shms)))
        except Exception:
            self._free(shms)
            raise
        self.bond[P - 1] = int(r["wrap"])
        self.fidelity_estimate *= r["fid"]
        self.truncations += r["trunc"]
        self._relayout(P - 1)
        self._relayout(0)
        self._load(shms, {int(k): v for k, v in r["nq"].items()})

    def _over_budget(self):
        if any(b > self.max_bonds + self.lazy_bonds for b in self.bond):
            return True
        if self.max_patch_qubits is not None:
            return any(len(lay) > self.max_patch_qubits for lay in self.layout)
        return False

    def _maybe_compress(self):
        if not self.lazy_bonds:
            self._compress()
            return
        if self._over_budget():
            self._compress()

    def flush(self):
        self._compress()

    def _compress(self, exact_only=False):
        if not self._touched:
            return
        if self._wrap_touched:
            self._compress_wrap(exact_only)
            self._touched.update((0, self.P - 1))
        self._sweep(min(self._touched), max(self._touched), exact_only)
        if not exact_only:
            self.flushes += 1
            self._touched.clear()
            self._wrap_touched = False

    def measure_shots(self, qubits, shots):
        self._compress()
        self._sweep(0, 0, exact_only=True)
        ps = list(range(self.P))
        shms = self._export(ps)
        try:
            return self.svc.glue_call("sample", self.key, dict(
                dtype=self.svc.real, patches=self._payload(ps, shms), shots=int(shots), starts=self.start,
                qubits=[int(q) for q in qubits], ring=self.ring))
        finally:
            self._free(shms)

    def out_ket(self):
        self._compress()
        ps = list(range(self.P))
        shms = self._export(ps)
        try:
            return self.svc.glue_call("contract", self.key, dict(dtype=self.svc.real,
                                                                 patches=self._payload(ps, shms)))
        finally:
            self._free(shms)

    def profile(self):
        """The service's counters: per patch server, the glue server and the coordinator (cumulative over every
        bridge of this configuration in this process)."""
        return self.svc.stats()


def hbridge_profile_delta(a, b):
    """Counters of b minus a (two HBService.stats() snapshots), per server."""
    keep = {"index", "pid", "device", "cpus", "threads", "role", "ocl", "peak_qubits", "live"}

    def sub(x, y):
        if isinstance(x, dict):
            return {k: (x[k] if k in keep else sub(x[k], y.get(k))) if k in y else x[k] for k in x}
        if isinstance(x, list) and y is not None and len(x) == len(y) and all(isinstance(v, (int, float)) for v in x):
            return [u - v for u, v in zip(x, y)]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            return x - y
        return x
    out = dict(servers=[sub(x, y) for x, y in zip(b["servers"], a["servers"])],
               coordinator=sub(b["coordinator"], a["coordinator"]))
    return out


def hbridge_condensed(delta):
    """A profile delta reduced to what a record keeps: per patch server its device and seconds in gates, ket export
    and load; the glue's seconds per call kind; the coordinator's wait."""
    out = dict(patch=[], glue={}, wait_s=round(delta["coordinator"]["wait_s"], 4))
    for s in delta["servers"]:
        if s["role"] == "patch":
            out["patch"].append(dict(device=s["device"], ops=s["ops"], op_s=round(s["op_s"], 4),
                                     export_s=round(s["export_s"], 4), load_s=round(s["load_s"], 4)))
        else:
            out["glue"] = {k: round(v[1], 4) for k, v in s["calls"].items() if v[0]}
    out["glue_s"] = round(sum(out["glue"].values()), 4)
    return out


def hbridge_sum(items):
    """The sum of hbridge_condensed profiles (same servers)."""
    t = dict(patch=[dict(p) for p in items[0]["patch"]], glue=dict(items[0]["glue"]), wait_s=items[0]["wait_s"])
    for x in items[1:]:
        for a_, b_ in zip(t["patch"], x["patch"]):
            for k in ("ops", "op_s", "export_s", "load_s"):
                a_[k] += b_[k]
        for k, v in x["glue"].items():
            t["glue"][k] = t["glue"].get(k, 0.0) + v
        t["wait_s"] += x["wait_s"]
    t["glue_s"] = sum(t["glue"].values())
    return t


def local_qrack_bridge(path=None):
    """PyQrack's own QrackBridge (PyQrack >= 2.30), or its qrack_bridge.py from path / $NIGHTHAWK_QRACK_BRIDGE loaded as
    a PyQrack submodule; None if neither is there."""
    try:
        from pyqrack import QrackBridge
        return QrackBridge
    except ImportError:
        pass
    path = path or os.environ.get("NIGHTHAWK_QRACK_BRIDGE")
    if not path or not Path(path).is_file():
        return None
    import importlib.util
    import pyqrack
    if "pyqrack.qrack_bridge" in sys.modules:
        return sys.modules["pyqrack.qrack_bridge"].QrackBridge
    spec = importlib.util.spec_from_file_location("pyqrack.qrack_bridge", str(path))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pyqrack.__name__
    sys.modules["pyqrack.qrack_bridge"] = mod
    spec.loader.exec_module(mod)
    return mod.QrackBridge


# ------------------------------------------------------------------ bridgebench
def _bb_circuit(n, P, depth, cross, seed, wrap):
    """A random circuit on QrackBridge's patch split: per layer a random U3 on every qubit and CX on a random
    matching, a share `cross` of its pairs between neighbouring patches (on a ring also P-1 | 0), the rest inside
    a patch."""
    rng = np.random.default_rng(seed)
    base, rem = divmod(n, P)
    sizes = [base + (1 if p >= P - rem else 0) for p in range(P)]
    start = [sum(sizes[:p]) for p in range(P)]
    part = [list(range(start[p], start[p] + sizes[p])) for p in range(P)]
    ops = []
    for _ in range(depth):
        for q in range(n):
            ops.append(("u", q) + tuple(float(x) for x in rng.uniform(0, 2 * np.pi, 3)))
        free = set(range(n))
        for _ in range(n // 2):
            if len(free) < 2:
                break
            if rng.random() < cross and P > 1:
                cuts = list(range(P if (wrap and P >= 3) else P - 1))
                rng.shuffle(cuts)
                pair = None
                for c in cuts:
                    a_ = [q for q in part[c] if q in free]
                    b_ = [q for q in part[(c + 1) % P] if q in free]
                    if a_ and b_:
                        pair = (int(rng.choice(a_)), int(rng.choice(b_)))
                        break
            else:
                ps = [p for p in range(P) if sum(1 for q in part[p] if q in free) >= 2]
                pair = None
                if ps:
                    p = int(rng.choice(ps))
                    c_ = [q for q in part[p] if q in free]
                    pair = tuple(int(x) for x in rng.choice(c_, 2, replace=False))
            if pair is None:
                break
            if rng.random() < 0.5:
                pair = pair[::-1]
            free -= set(pair)
            ops.append(("cx",) + pair)
    return ops


def _bb_apply(b, ops):
    for op in ops:
        if op[0] == "u":
            b.u(op[1], *op[2:])
        else:
            b.mcx([op[1]], op[2])


def cmd_bridgebench(a):
    """Horizontal QrackBridge against PyQrack's own: the same random circuits on both (and the exact simulator), the
    states and shots compared, wall times, and the horizontal one's profile per workload."""
    cfg = parse_hbridge(a.hbridge)
    QB = local_qrack_bridge(a.qrack_bridge)
    print(f"bridgebench: n {a.n}, {a.patches} patches{' (ring)' if a.wrap else ''}, depth {a.depth}, cross share "
          f"{a.cross}, max_bonds {a.max_bonds}, lazy {a.lazy}, {a.reps} circuit(s); horizontal: patch servers "
          f"{', '.join(cfg['devices'])}, glue threads {cfg['glue_threads']}, batch {cfg['batch']}; local: "
          + ("PyQrack's QrackBridge" if QB else "not found (PyQrack < 2.30 and no $NIGHTHAWK_QRACK_BRIDGE)"))
    t0 = time.perf_counter()
    svc = hbridge_service(a.hbridge)
    print(f"  servers up in {time.perf_counter() - t0:.2f} s: "
          + ", ".join(f"{r} {i} pid {v['pid']}" for (r, i), v in sorted(svc.info.items())))
    rows = []
    worst = 1.0
    same_shots = True
    snap0 = svc.stats()
    for rep in range(a.reps):
        ops = _bb_circuit(a.n, a.patches, a.depth, a.cross, a.seed + rep, a.wrap)
        kw = dict(max_bonds=a.max_bonds, patches=a.patches, wrap=a.wrap, lazy_bonds=a.lazy, seed=a.seed + rep)
        res = {}
        for eng in ("horizontal", "local"):
            if eng == "local" and QB is None:
                continue
            t1 = time.perf_counter()
            b = HBridge(a.n, hbridge=a.hbridge, **kw) if eng == "horizontal" else QB(a.n, **kw)
            _bb_apply(b, ops)
            b.flush()
            t2 = time.perf_counter()
            shots = b.measure_shots(list(range(a.n)), a.shots)
            t3 = time.perf_counter()
            v = np.asarray(b.out_ket()) if a.n <= a.ket_max else None
            res[eng] = dict(gates=t2 - t1, shots_s=t3 - t2, shots=shots, v=v, fid=b.fidelity_estimate,
                            trunc=b.truncations, peak=b.peak_bonds, flushes=b.flushes)
            if eng == "horizontal":
                b.close()
        h, l_ = res["horizontal"], res.get("local")
        ov = abs(np.vdot(h["v"], l_["v"])) ** 2 if l_ and h["v"] is not None and l_["v"] is not None else None
        if ov is not None:
            worst = min(worst, ov)
        same = l_ is not None and h["shots"] == l_["shots"]
        same_shots &= same or l_ is None
        rows.append((rep, len(ops), h, l_, ov, same))
        print(f"  circuit {rep}: {len(ops)} ops ({sum(1 for o in ops if o[0] == 'cx')} CX); horizontal "
              f"{h['gates']:.2f} s gates + {h['shots_s']:.2f} s shots" + (f"; local {l_['gates']:.2f} + "
                                                                       f"{l_['shots_s']:.2f} s" if l_ else "")
              + f"; truncations {h['trunc']}, peak bonds {h['peak']}, fidelity estimate {h['fid']:.4f}"
              + (f" (local {l_['fid']:.4f})" if l_ else "")
              + (f"; |<h|l>|^2 {ov:.7f}" if ov is not None else "") + (f"; shots {'identical' if same else 'DIFFER'}"
                                                                       if l_ else ""), flush=True)
    prof = hbridge_profile_delta(snap0, svc.stats())
    print("\nhorizontal profile (this run):")
    print(f"  {'server':<14}{'where':<18}{'busy s':>9}{'gates':>9}{'gate s':>9}{'ket out':>9}{'out s':>8}{'ket in':>8}"
          f"{'in s':>8}{'MB':>9}{'idle s':>9}")
    for s in prof["servers"]:
        if s["role"] == "patch":
            busy = s["op_s"] + s["export_s"] + s["load_s"]
            print(f"  {'patch ' + str(s['index']):<14}{s['device'] + (' cpus ' + s['cpus'] if s.get('cpus') else ''):<18}"
                  f"{busy:>9.2f}{s['ops']:>9}{s['op_s']:>9.2f}{s['exports']:>9}{s['export_s']:>8.2f}{s['loads']:>8}"
                  f"{s['load_s']:>8.2f}{(s['export_bytes'] + s['load_bytes']) / 1e6:>9.1f}{s['idle_s']:>9.2f}")
        else:
            busy = sum(v[1] for v in s["calls"].values())
            calls = ", ".join(f"{k} {v[0]} / {v[1]:.2f} s" for k, v in sorted(s["calls"].items()))
            print(f"  {'glue':<14}{('threads ' + str(s.get('threads'))):<18}{busy:>9.2f}  {calls}; read "
                  f"{s['read_bytes'] / 1e6:.1f} MB; idle {s['idle_s']:.2f} s")
    c = prof["coordinator"]
    print(f"  coordinator: {c['sent_ops']} ops in {c['batches']} batches, {c['wait_s']:.2f} s waiting on servers")
    tot_h = sum(r[2]["gates"] + r[2]["shots_s"] for r in rows)
    tot_l = sum(r[3]["gates"] + r[3]["shots_s"] for r in rows if r[3]) if QB else None
    print(f"\nwall: horizontal {tot_h:.2f} s" + (f", local {tot_l:.2f} s ({tot_l / tot_h:.2f}x)" if tot_l else ""))
    if QB:
        ok = worst > 1 - 1e-5 and same_shots
        print(f"check: horizontal = local QrackBridge (worst state overlap {worst:.7f}; shots "
              f"{'identical' if same_shots else 'differ'}): {'OK' if ok else 'FAIL'}")
        if a.check and not ok:
            raise SystemExit(1)
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(json.dumps(dict(test="bridgebench", when=time.strftime("%Y-%m-%dT%H:%M:%S"), n=a.n,
                                     patches=a.patches, wrap=a.wrap, depth=a.depth, cross=a.cross,
                                     max_bonds=a.max_bonds, lazy=a.lazy, reps=a.reps, hbridge=cfg,
                                     horizontal_s=tot_h, local_s=tot_l, worst_overlap=worst if QB else None,
                                     shots_identical=same_shots if QB else None, profile=prof,
                                     qrack=qrack_versions())) + "\n")
        print(f"  appended to {a.out}")


# ================================================================== ghz: the formal proof's layers, the seam engines, Layer 3 (formerly nighthawk_ghz.py)
GHZ_DOC = """
nighthawk_qrack.py ghz -- Layer 1a (GHZ populations + MQC) and Layer 2 (idle-window refocusing) of
"Exact Refocusing of Diagonal Crosstalk for Shor Order Finding on a 10 x 12 Square-Lattice Processor"
(the formal proof; numpy reference shrs_nighthawk_stealth.py), run on Nighthawk registers under
QrackAceBackend, so the paper's statements meet ACE's seams.

The ghz command of nighthawk_qrack.py (formerly nighthawk_ghz.py, whose stub forwards here), on its
registers, ACE presets and placement, coupler classes and worker pool. Gates are exact; the only
departures from the paper's model are ACE's seam approximation and its sampler. An exact
QrackSimulator reference runs beside every ACE run.

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
  shoralg   Layer 3, the paper's Shor order finding (N = 15, a = 7, 4 counting + 4 work qubits; also 2n counting
            qubits and N = 21, every order), as gates (H, X, phase, CX; the controlled multiplications synthesised
            exactly as the paper's U), across QrackBridge circuits (two chips, the box's faces with --box L H, two
            bare circuits with --pair) at every --bonds capacity, or on the dual Nighthawk R1 (--dual: routed on
            the module lattices, distant CNOTs through one cable, the dual's models). Layouts: inside one circuit,
            the cut between the registers (fan-out ancilla, or direct), through the counting register, through
            the work register. Per circuit the per-shot factoring success (the paper's rule), the order found, the
            distance to the ideal distribution, bonds and truncations; shor_alg records.
  plot      the map of a records file (--in).
  selftest  Lemma 3, Lemma 4 and the witness against numpy; the readout gates, Theorem 1, Corollaries 2.1
            and 4.1, the outside-coupling identity, fold and the ZNE amplification on the exact simulator;
            the ZNE models on synthetic decays; Layer S's seam classes, plan, the code's schedule, the
            co-located authority rule, --inside-zz stabilizer and the probe's target and chain rules; Layer 3's
            multiplications, circuit, success rule, hard-cut rule and dual router.

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
  python3 nighthawk_qrack.py ghz selftest
  python3 nighthawk_qrack.py ghz tiles --register nighthawk61r2 --unit 3 --repeats 4 --jobs 8 --out ghz61.jsonl
  python3 nighthawk_qrack.py ghz tiles --register nighthawk61r2 --unit 3 --layer2 dd --blocks 3 --repeats 8 \\
          --jobs 8 --out ghz61_dd.jsonl
  python3 nighthawk_qrack.py ghz chip  --register nighthawk2p57 --shots 4096 --repeats 5 --jobs 8 --out ghz57.jsonl
  python3 nighthawk_qrack.py ghz chip  --register nighthawk2p16 --layer2 unmitigated --out ghz16.jsonl
  python3 nighthawk_qrack.py ghz stack --register nighthawk61r2 --unit 3 --shots 256 --repeats 4 --jobs 8 --out stack61.jsonl
  python3 nighthawk_qrack.py ghz stack --register nighthawk2p16 --mode chip --out stack16.jsonl
  python3 nighthawk_qrack.py ghz stack --register nighthawk2p28 --unit 3 --zne-k 1,2 --out stack28.jsonl
  python3 nighthawk_qrack.py ghz stack --register nighthawk61r2 --unit 3 --shots 256 --repeats 4 --order code-echo \\
          --out stack61_ce.jsonl
  python3 nighthawk_qrack.py ghz stackmap --in stack61_ce.jsonl --png stack61_ce.png
  python3 nighthawk_qrack.py ghz sweep --register nighthawk2p28 --shots 256 --repeats 4 --no-exact --code-at 8 \\
          --out sweep28.jsonl --summary sweep28.json
  python3 nighthawk_qrack.py ghz probe --register nighthawk61r2 --shots 256 --repeats 4 --out probe61.jsonl
  python3 nighthawk_qrack.py ghz plot  --in ghz61.jsonl --register nighthawk61r2 --png ghz61.png
  python3 nighthawk_qrack.py ghz shor  --box 5 5 --bridge-max-bonds 1 --jobs 2 --out shor9_box5_mb1.jsonl
  python3 nighthawk_qrack.py ghz shormap --in shor9_box5_mb1.jsonl --png shor9_box5_mb1.png
  python3 nighthawk_qrack.py ghz dual  --models hardcut,ideal,exact,budget,measured --trajectories 20000 --jobs 2 \\
          --out dual_r1.jsonl --png dual_r1.png
  python3 nighthawk_qrack.py ghz shoralg --N 15 21 --bonds 0 1 2 3 4 5 --jobs 2 --out shoralg_chips.jsonl
  python3 nighthawk_qrack.py ghz shoralg --box 5 5 --layouts split,direct --bonds 0 1 2 3 4 5 --jobs 2 --out shoralg_box.jsonl
  python3 nighthawk_qrack.py ghz shoralg --dual --N 15 --models hardcut,ideal,exact,budget,measured --trajectories 400 \\
          --jobs 2 --out shoralg_dual.jsonl

Environment: no CUDA. CPU by default; --gpu runs ACE and the reference on the OpenCL device
QRACK_OCL_DEFAULT_DEVICE names (--ocl-sync: QRACK_OCL_SYNC=1, Qrack >= 10.13.2).
"""

COLUMNS = 10          # nighthawk_qrack's device lattice: 12 rows x 10 columns (the paper writes 10 x 12)
LAYER2_MODES = ("none", "unmitigated", "compensated", "dd")


# ================================================================== register, placement, classes

class Register:
    """A register as ACE places it: layout, couplers, per qubit its ACE simulators (two or more for a
    seam qubit), per coupler its class, lattice cell and sublattice, seam distances."""

    def __init__(self, repo, reg, brep=False):
        if reg != "release" and reg not in REGISTERS:
            load_register_files()
        self.repo, self.name, self.brep = repo, reg, bool(brep)
        self.lay = Layout(repo) if reg == "release" else RegisterLayout(repo, reg)
        self.n = self.lay.n
        ra = shor9_run_args(reg, repo, False, self.brep)          # CPU settings: placement only
        apply_register_preset(ra, self.lay)
        self.map, sids, idx, row = shor9_register_map(self.lay, ra)
        probe = AceEngine(self.lay, self.n, ra)
        self.size, self.lrc, self.lrr, self.torus, self.opts = probe.size, probe.lrc, probe.lrr, probe.torus, probe.opts
        self.widths = list(probe.widths)
        del probe
        self.idx = list(idx)
        self.sids = [sids[idx[q]] for q in range(self.n)]
        self.seam = {q for q in range(self.n) if len(self.sids[q]) > 1}
        self.couplers = [tuple(c) for c in logical_couplers(self.lay, self.n)]
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
        return ace_backend(self.size, long_range_columns=self.lrc, long_range_rows=self.lrr, is_torus=self.torus,
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
    pool, workers = shor_pool(a.jobs, len(calls))
    every = float(a.progress or 0)
    if every:
        print(f"# ghz plan: {len(calls)} circuits ({len(groups)} groups x {len(settings)} settings x {reps} "
              f"ACE realisations" + (f" + exact x {ex_reps}" if not no_exact else "") + f"), {workers} process(es)"
              + (f" [{label}]" if label else ""), flush=True)
    t0 = time.time()
    try:
        res = shor_pool_map(pool, calls, labels, every, what="ghz")
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
            + f"; {qrack_versions()}")


def base_record(a, R, l2):
    return dict(register=R.name, n=R.n, shots=a.shots, repeats=a.repeats, layers=layers_name(parse_layers(a.layers)),
                layer2=l2, estimator=a.estimator, stabilizer=bool(a.stab), boundary_rep=bool(a.ace_boundary_rep),
                code_authority=getattr(a, "code_authority", "lineage"),
                **({"engine": "loom", "loom": R.map["loom"]} if getattr(R, "engine", "ace") == "loom" else {}),
                device="gpu" if a.gpu else "cpu", ocl_sync=os.environ.get("QRACK_OCL_SYNC") == "1",
                qrack=qrack_versions(), ace=dict(lrc=R.lrc, lrr=R.lrr, torus=bool(R.torus), width=R.size,
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
    ../rcs-nighthawk (the layout of thereminq-examples). It imports nighthawk_qrack, this file."""
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
        load_register_files()
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
                row = layout_row(eng.size, eng.lrc, eng.lrr, eng.torus, eng.opts)
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
    eng = AceEngine(lay, lay.n, a)
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
    library's OutKet / InKet, which PyQrack 2.29.5 / Qrack 10.13.2 have). With --bridge-engine horizontal
    ($NIGHTHAWK_BRIDGE_ENGINE): nighthawk_qrack.py's HBridge, the same algorithm with every patch on a patch server
    (its own PyQrack instance, device and CPUs) and the glue on a glue server (--hbridge / $NIGHTHAWK_HBRIDGE)."""
    if os.environ.get("NIGHTHAWK_BRIDGE_ENGINE", "local") == "horizontal":
        _QB["src"] = "nighthawk_qrack.HBridge (" + json.dumps(parse_hbridge(None)) + ")"
        return HBridge
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
    pool, workers = shor_pool(a.jobs, len(calls))
    every = float(a.progress or 0)
    if every:
        print(f"# ghz plan: {len(calls)} circuits on the loom ({len(groups)} groups x {len(settings)} settings x "
              f"{reps} realisations" + (f" + exact x {ex_reps}" if not no_exact else "") + f"), {workers} "
              f"process(es) [{label}]", flush=True)
    t0 = time.time()
    try:
        res = shor_pool_map(pool, calls, labels, every, what="ghz")
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
        shor9_prepare(sim, u, spec["state"], {q: i for i, q in enumerate(qs)})
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
            shor9_prepare(br, u, spec["state"], idx)
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
    states = SHOR9_STATES
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
          + ("" if a.no_exact else f", exact {a.exact_shots}") + f"; {qrack_versions()}")
    pool, workers = shor_pool(a.jobs, len(calls))
    every = float(a.progress or 0)
    if every:
        print(f"# ghz shor plan: {len(calls)} circuits, {workers} process(es)", flush=True)
    t0 = time.time()
    try:
        res = shor_pool_map(pool, calls, labels, every, what="ghz shor")
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
        _, local = _local_unit(u)
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
                ok = [(shor9_decode(k, local, s), v) for k, v in h.items()]
                d.update(raw=sum(v * r_ for (r_, _), v in ok) / n_, decoded=sum(v * d_ for (_, d_), v in ok) / n_,
                         shots=n_, **{k_: v_ for k_, v_ in shor9_compare(h, local, s).items()
                                      if k_ in ("tvd", "tvd_floor", "impossible", "top_wrong")})
            hx = hist.get((j, s, True))
            if hx:
                n_ = sum(hx.values())
                d["exact_decoded"] = sum(v * shor9_decode(k, local, s)[1] for k, v in hx.items()) / n_
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
                         loom=R.map["loom"], qrack=qrack_versions()))
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
    print(f"\n{'class':<40}{'units':>6}" + "".join(f"{'|' + s + '_L>':>8}" for s in SHOR9_STATES)
          + f"{'log Z':>8}{'log X':>8}{'exact':>7}{'bonds Z/X':>11}{'vanished':>9}")
    for c in sorted(by, key=lambda c_: (c_.split(",")[0] != "inside", c_)):
        rs = by[c]
        cell = []
        for s in SHOR9_STATES:
            v = [r["states"][s]["decoded"] for r in rs if "decoded" in r["states"][s]]
            cell.append(f"{np.mean(v):>8.3f}" if v else f"{'-':>8}")
        lz = [r["logical_Z"] for r in rs if r["logical_Z"] is not None]
        lx = [r["logical_X"] for r in rs if r["logical_X"] is not None]
        ex = [r["states"][s]["exact_decoded"] for r in rs for s in SHOR9_STATES if "exact_decoded" in r["states"][s]]
        van = sum(r["states"][s]["vanished"] for r in rs for s in SHOR9_STATES)
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
    van = sum(r["states"][s]["vanished"] for r in sr for s in SHOR9_STATES)
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
             f"per logical state · {sr[0].get('qrack', '')} · nighthawk_qrack.py ghz shor --size {m}"
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

    def x(self, q):
        self.s.x(self.q[q])
        self._dep1(q, self._r1(q))

    def u(self, q, th, ph, lm):
        """A general rotation; with th = 0 a phase gate, which IBM runs as a virtual Z (a frame change): no error."""
        self.s.u(self.q[q], th, ph, lm)
        if th != 0.0:
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
          f"error {a.local_cz_error}; {qrack_versions()}")
    pool, workers = shor_pool(a.jobs, len(calls))
    t0 = time.time()
    try:
        res = shor_pool_map(pool, calls, [f"dualmap unit {k[0]} {k[1]}" for k in keys], float(a.progress or 0),
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
                 trajectories=(a.trajectories_link if u["link"] else a.trajectories), qrack=qrack_versions())
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
    fig.text(0.02 * 17 / FW, 0.22 / FH, f"{recs[0]['when']} · {recs[0]['qrack']} · nighthawk_qrack.py ghz dualmap --models "
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
          f"{qrack_versions()}")
    pool, workers = shor_pool(a.jobs, len(calls))
    t0 = time.time()
    try:
        res = shor_pool_map(pool, calls, [f"dual {k[0]} {k[1]}" for k in keys], float(a.progress or 0),
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
                                         trajectories=K, max_bonds=a.bridge_max_bonds, qrack=qrack_versions(),
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


# ================================================================== Layer 3: Shor order finding across the configurations

# The paper's Layer 3 (shrs_nighthawk_stealth.py, class Shor): order finding for N = 15, a = 7 with 4 counting and 4
# work qubits; the counting register starts in |+>^t, the work register in |1>, counting bit k controls the
# multiplication by a^(2^k) mod N (on 0 < y < N; y = 0 and y >= N are left alone), then the inverse QFT and a
# readout of the counting register; figure of merit the per-shot factoring success. Here the same circuit as gates
# (H, X, phase, CX), so it runs on every engine of this script: QrackSimulator, QrackBridge over two chips or the
# box's six faces, and the dual Nighthawk R1's two modules with their cables.

SHA_LAYOUTS = ("inside", "split", "direct", "countsplit", "worksplit")
SHA_LAYOUT_TEXT = dict(
    inside="both registers on one circuit",
    split="counting | work: the cut between the registers, each controlled multiplication through a fan-out "
          "ancilla on the work side (two CXs across per multiplication)",
    direct="counting | work, no fan-out: every gate the counting qubit controls crosses the cut",
    countsplit="the cut through the counting register (low half | high half + work): the inverse QFT crosses it",
    worksplit="the cut through the work register (counting + low work bits | high work bits): the multiplications "
              "cross it")


def sha_order(a, N):
    x, r = a % N, 1
    while x != 1:
        x, r = x * a % N, r + 1
    return r


def sha_bases(N):
    """For every multiplicative order r > 1 mod N, the smallest a with gcd(a, N) = 1 of that order (a = N - 1, of
    order 2 and no use, left out)."""
    seen = {}
    for a in range(2, N - 1):
        if math.gcd(a, N) == 1:
            seen.setdefault(sha_order(a, N), a)
    return [seen[r] for r in sorted(seen) if r > 1]


def sha_convergents(y, Q, N):
    """Denominators of the continued-fraction convergents of y / Q, up to N."""
    out, (p, q), h, k = [], (y, Q), (0, 1), (1, 0)
    while q:
        c = p // q
        p, q = q, p - c * q
        h, k = (h[1], c * h[1] + h[0]), (k[1], c * k[1] + k[0])
        if k[1] > N:
            break
        out.append(k[1])
    return out


def sha_factor(N, a, d):
    """The factor pair from a candidate order d (d even, a^(d/2) not -1), else None."""
    if d % 2 or pow(a, d, N) != 1:
        return None
    x = pow(a, d // 2, N)
    if x in (1, N - 1):
        return None
    f = sorted((math.gcd(x - 1, N), math.gcd(x + 1, N)))
    return tuple(f) if 1 < f[0] < N else None


def sha_outcome(y, t, N, a, r):
    """Per counting outcome y: (factoring success by the paper's rule: the denominator of y / 2^t limited to N gives
    the factors, as shrs_nighthawk_stealth.py's Shor._factors_from; the order found: the first convergent
    denominator d of y / 2^t with a^d = 1 mod N equals r)."""
    from fractions import Fraction
    if y == 0:
        return False, False
    d = Fraction(y, 1 << t).limit_denominator(N).denominator
    ok = next((dd for dd in sha_convergents(y, 1 << t, N) if pow(a, dd, N) == 1), None)
    return sha_factor(N, a, d) is not None, ok == r


def sha_analytic(N, a, t):
    """P(y) of the ideal circuit: the work register's r cosets of the counting values, each an arithmetic progression
    of step r, through the inverse QFT."""
    Q, r = 1 << t, sha_order(a, N)
    y = np.arange(Q)
    P = np.zeros(Q)
    for s in range(r):
        xs = np.arange(s, Q, r)
        P += np.abs(np.exp(-2j * np.pi * np.outer(y, xs) / Q).sum(axis=1) / Q) ** 2
    return P


def sha_score(P, t, N, a, ideal=None):
    """The measures of one output distribution P over the counting register."""
    r = sha_order(a, N)
    succ = order = 0.0
    for y in np.nonzero(P > 1e-12)[0]:
        s, o = sha_outcome(int(y), t, N, a, r)
        succ += P[y] * s
        order += P[y] * o
    out = dict(success=float(succ), order=float(order), zero=float(P[0]))
    if ideal is not None:
        out.update(tvd=float(0.5 * np.abs(P - ideal).sum()),
                   fidelity=float(np.sum(np.sqrt(np.clip(P, 0, None) * ideal)) ** 2),
                   support=float(P[ideal > 1e-9].sum()))
        nz = P.copy()
        nz[0] = -1
        out["argmax_ok"] = bool(sha_outcome(int(np.argmax(nz)), t, N, a, r)[0])
    return out


def sha_bonds_needed(r, layout):
    """Bond qubits per cut the split layouts need for the exact output: none when r is a power of two (the work
    register's record of x mod r is then x's own low bits, so a truncation that keeps any of its cosets, as a hard cut
    does, changes nothing the counting register shows); otherwise the final state's Schmidt rank r across the cut,
    ceil(log2 r) bond qubits, plus one for the counting bit in flight (the fan-out ancilla, or the controlled gate)."""
    if layout == "inside":
        return 0
    if r & (r - 1) == 0:
        return 0
    if layout in ("split", "direct"):
        return int(math.ceil(math.log2(r))) + 1
    return None


class ShaOps:
    """The circuit as ops on logical qubits: (h, q), (x, q), (p, q, lambda) = diag(1, e^{i lambda}), (cx, c, t)."""

    def __init__(self):
        self.ops = []

    def h(self, q):
        self.ops.append(("h", q))

    def x(self, q):
        self.ops.append(("x", q))

    def p(self, q, lam):
        self.ops.append(("p", q, float(lam)))

    def cx(self, c, t):
        self.ops.append(("cx", c, t))

    def ccx(self, a, b, t):
        """Toffoli: 6 CX, 7 T / T^dagger, 2 H (exact)."""
        T = math.pi / 4
        self.h(t)
        self.cx(b, t)
        self.p(t, -T)
        self.cx(a, t)
        self.p(t, T)
        self.cx(b, t)
        self.p(t, -T)
        self.cx(a, t)
        self.p(b, T)
        self.p(t, T)
        self.h(t)
        self.cx(a, b)
        self.p(a, T)
        self.p(b, -T)
        self.cx(a, b)

    def cswap(self, c, a, b):
        self.cx(b, a)
        self.ccx(c, a, b)
        self.cx(b, a)

    def cp(self, a, b, th):
        self.p(a, th / 2)
        self.cx(a, b)
        self.p(b, -th / 2)
        self.cx(a, b)
        self.p(b, th / 2)

    def mcx(self, ctl, t, anc):
        """X on t when every (qubit, value) of ctl holds; k >= 3 controls through a V-chain of k - 2 clean ancillas."""
        neg = [q for q, v in ctl if not v]
        for q in neg:
            self.x(q)
        cs = [q for q, _ in ctl]
        k = len(cs)
        if k == 1:
            self.cx(cs[0], t)
        elif k == 2:
            self.ccx(cs[0], cs[1], t)
        else:
            if len(anc) < k - 2:
                raise ValueError(f"mcx: {k} controls need {k - 2} ancillas")
            chain = [(cs[0], cs[1], anc[0])] + [(cs[i], anc[i - 2], anc[i - 1]) for i in range(2, k - 1)]
            for a_, b_, t_ in chain:
                self.ccx(a_, b_, t_)
            self.ccx(cs[k - 1], anc[k - 3], t)
            for a_, b_, t_ in reversed(chain):
                self.ccx(a_, b_, t_)
        for q in neg:
            self.x(q)


def sha_transpositions(perm):
    """A permutation of 0..len-1 as transpositions in gate order (a cycle e0 -> e1 -> ... as swaps (e_i, e_i+1) from
    the end of the cycle down)."""
    seen, out = set(), []
    for s0 in range(len(perm)):
        if s0 in seen or perm[s0] == s0:
            seen.add(s0)
            continue
        cyc, x = [], s0
        while x not in seen:
            seen.add(x)
            cyc.append(x)
            x = perm[x]
        out += [(cyc[i], cyc[i + 1]) for i in range(len(cyc) - 2, -1, -1)]
    return out


def _sha_ctrans(o, ctrl, W, Z, x, y):
    """Controlled transposition of basis states x and y of the work register W: CXs from one bit where they differ
    onto the others (so the two then differ there only), an X on that bit controlled by ctrl and every other bit at
    its value, the CXs undone."""
    n = len(W)
    D = [i for i in range(n) if (x >> i & 1) != (y >> i & 1)]
    p_ = D[0]
    for d in D[1:]:
        o.cx(W[p_], W[d])
    x2 = x if not (x >> p_ & 1) else y
    o.mcx([(ctrl, 1)] + [(W[i], x2 >> i & 1) for i in range(n) if i != p_], W[p_], Z)
    for d in reversed(D[1:]):
        o.cx(W[p_], W[d])


def sha_cmul(o, ctrl, W, Z, c, N, synth="auto"):
    """Multiplication by c mod N of the work register W (little-endian) on 0 < y < N, controlled by ctrl; y = 0 and
    y >= N unchanged, as the paper's U. generic: the permutation's transpositions, each one multi-controlled X
    (needs len(W) - 2 clean ancillas Z). structured, for N = 2^n - 1 and c = +-2^k mod N: a cyclic rotation of the
    bits by k (controlled SWAPs), for -2^k then every bit flipped (CXs from ctrl), and y = 0 / y = N swapped back
    (one controlled transposition), so it is the same unitary. auto: structured where it applies. Returns the
    synthesis used."""
    n = len(W)
    perm = [(c * y) % N if 0 < y < N else y for y in range(1 << n)]
    if synth in ("auto", "structured") and N == (1 << n) - 1:
        for k in range(n):
            for sgn in (1, -1):
                if (sgn * (1 << k)) % N == c % N:
                    for a_, b_ in sha_transpositions([(i + k) % n for i in range(n)]):
                        o.cswap(ctrl, W[a_], W[b_])
                    if sgn < 0:
                        for q in W:
                            o.cx(ctrl, q)
                        _sha_ctrans(o, ctrl, W, Z, 0, N)
                    return "structured"
        if synth == "structured":
            raise SystemExit(f"shoralg --synth structured: {c} is not +-2^k mod {N}")
    for x, y in sha_transpositions(perm):
        _sha_ctrans(o, ctrl, W, Z, x, y)
    return "generic"


def sha_iqft(o, qs):
    """The inverse QFT on qs (qs[j] = counting bit j): the swap-free QFT's gates reversed with negated phases; the
    output's bit i is on qs[t - 1 - i] (no SWAPs: the readout is relabelled)."""
    t = len(qs)
    f = ShaOps()
    for i in range(t):
        f.h(qs[i])
        for j in range(i + 1, t):
            f.cp(qs[j], qs[i], math.pi / 2 ** (j - i))
    for op in reversed(f.ops):
        o.ops.append(("p", op[1], -op[2]) if op[0] == "p" else op)


def sha_circuit(N, a, t, layout, synth="auto"):
    """The logical circuit and its sides: counting qubits C (C_k controls a^(2^k)), work W (little-endian, starts at
    |1>), the fan-out ancilla F (split, countsplit, worksplit), the multi-controlled-X ancillas Z (generic synthesis).
    side[q] is 0 or 1: the circuit (chip, face, module) a qubit sits on. ybits[i] is the qubit holding bit i of the
    outcome y."""
    n = N.bit_length()
    fan = layout in ("split", "countsplit", "worksplit")
    C = list(range(t))
    W = list(range(t, t + n))
    q = t + n
    F = q if fan else None
    q += 1 if fan else 0
    Z = list(range(q, q + max(0, n - 2)))
    nq = q + len(Z)
    side = {x: 0 for x in range(nq)}
    if layout in ("split", "direct"):
        side.update({x: 1 for x in range(t, nq)})
    elif layout == "countsplit":
        side.update({x: 1 for x in range(t // 2, nq)})
    elif layout == "worksplit":
        side.update({x: 1 for x in W[n // 2:] + ([F] if fan else []) + Z})
    o = ShaOps()
    for c in C:
        o.h(c)
    o.x(W[0])
    synths = []
    for k in range(t):
        c = pow(a, 1 << k, N)
        if c == 1:
            continue                          # a^(2^k) = 1: the multiplication is the identity (repeated squaring)
        if fan:
            o.cx(C[k], F)
        synths.append(sha_cmul(o, F if fan else C[k], W, Z, c, N, synth))
        if fan:
            o.cx(C[k], F)
    sha_iqft(o, C)
    return dict(ops=o.ops, nq=nq, side=side, C=C, W=W, F=F, Z=Z, ybits=[C[t - 1 - i] for i in range(t)],
                synth=sorted(set(synths)), mults=len(synths))


def sha_counts(ops, side):
    cx = [op for op in ops if op[0] == "cx"]
    return dict(ops=len(ops), cx=len(cx), cross=sum(1 for op in cx if side[op[1]] != side[op[2]]),
                oneq=sum(1 for op in ops if op[0] in ("h", "x")))


def sha_apply(g, ops):
    """ops on a gate object with h, x, u, cx (Gates, BridgeGates, DualGates, or a QrackSimulator adapter)."""
    for op in ops:
        k = op[0]
        if k == "h":
            g.h(op[1])
        elif k == "x":
            g.x(op[1])
        elif k == "p":
            g.u(op[1], 0.0, 0.0, op[2])
        elif k == "cx":
            g.cx(op[1], op[2])
        elif k == "dcx":
            g.dcx(op[1], op[2])
        else:
            raise ValueError(f"shoralg: op {k}")


class _SimGates:
    def __init__(self, sim, idx):
        self.s, self.q = sim, idx

    def h(self, q):
        self.s.h(self.q[q])

    def x(self, q):
        self.s.x(self.q[q])

    def u(self, q, th, ph, lm):
        self.s.u(self.q[q], th, ph, lm)

    def cx(self, c, t):
        self.s.mcx([self.q[c]], self.q[t])


def sha_marginal(v, ybits_idx, t):
    """P(y) from a ket (bit q of the basis index = simulator qubit q)."""
    p = np.abs(np.asarray(v, dtype=complex)) ** 2
    p /= p.sum()
    i = np.arange(p.size, dtype=np.int64)
    y = np.zeros(p.size, dtype=np.int64)
    for b, q in enumerate(ybits_idx):
        y |= ((i >> q) & 1) << b
    return np.bincount(y, weights=p, minlength=1 << t)


def sha_dual_lines(link):
    """The two module lines the dual router uses: from the link's lattice qubit on each module along row 11 to
    column 0, then row 10 left to right, row 9 right to left, ...: each step a coupler."""
    def snake(start, base):
        r0, c0 = (start - base) // 10, (start - base) % 10
        path = [base + 10 * r0 + c for c in range(c0, -1, -1)]
        for i, r in enumerate(range(r0 - 1, -1, -1)):
            path += [base + 10 * r + c for c in (range(10) if i % 2 == 0 else range(9, -1, -1))]
        return path
    L = DUAL_LINKS[[x["name"] for x in DUAL_LINKS].index(link)]
    return snake(L["a"], 0), snake(L["b"], 120), L


def sha_dual_route(circ, link="242-245"):
    """The logical circuit compiled onto the dual Nighthawk R1: side 0 on module A, side 1 on module B, each side's
    qubits on its line (sha_dual_lines) in order, F (if any) first on the work side. A CX between two qubits of one
    module moves its control along the line by SWAPs (3 CXs each) until the two are neighbours; a CX across moves
    both to the link's ends and runs as the paper's distant CNOT (dcx). Returns (physical ops, ybits as physical
    qubits, counts)."""
    lines = sha_dual_lines(link)[:2]
    order = {0: [], 1: []}
    for q in ([circ["F"]] if circ["F"] is not None else []) + circ["C"] + circ["W"] + circ["Z"]:
        order[circ["side"][q]].append(q)
    if circ["F"] is not None:
        s = circ["side"][circ["F"]]
        order[s] = [circ["F"]] + [q for q in order[s] if q != circ["F"]]
    pos, occ = {}, ({}, {})
    for s in (0, 1):
        if len(order[s]) > len(lines[s]):
            raise SystemExit(f"shoralg --dual: {len(order[s])} qubits do not fit a module line")
        for i, q in enumerate(order[s]):
            pos[q] = (s, i)
            occ[s][i] = q
    out, n = [], dict(swaps=0, local_cx=0, distant=0)

    def phys(q):
        s, i = pos[q]
        return lines[s][i]

    def swap(s, i):                          # exchange line positions i and i + 1 of module s
        a_, b_ = lines[s][i], lines[s][i + 1]
        out.extend([("cx", a_, b_), ("cx", b_, a_), ("cx", a_, b_)])
        n["swaps"] += 1
        qa, qb = occ[s].pop(i, None), occ[s].pop(i + 1, None)
        if qa is not None:
            occ[s][i + 1], pos[qa] = qa, (s, i + 1)
        if qb is not None:
            occ[s][i], pos[qb] = qb, (s, i)

    def bring(q, j):
        s, i = pos[q]
        while i > j:
            swap(s, i - 1)
            i -= 1
        while i < j:
            swap(s, i)
            i += 1
    for op in circ["ops"]:
        if op[0] != "cx":
            out.append((op[0], phys(op[1])) + tuple(op[2:]))
            continue
        c, t_ = op[1], op[2]
        if pos[c][0] == pos[t_][0]:
            while abs(pos[c][1] - pos[t_][1]) > 1:
                bring(c, pos[c][1] + (1 if pos[t_][1] > pos[c][1] else -1))
            out.append(("cx", phys(c), phys(t_)))
            n["local_cx"] += 1
        else:
            bring(c, 0)
            bring(t_, 0)
            out.append(("dcx", phys(c), phys(t_)))
            n["distant"] += 1
    return out, [phys(q) for q in circ["ybits"]], n


SHA_DUAL_MODELS = ("hardcut", "ideal", "exact", "budget", "measured")


def _sha_dual_task(spec):
    """Trajectories of one circuit on the dual R1 under one model: the mean P(y) over trajectories (from the ket),
    the bridge's peak bonds."""
    circ = sha_circuit(spec["N"], spec["a"], spec["t"], spec["layout"], spec.get("synth", "auto"))
    ops, ybits, _ = sha_dual_route(circ, spec.get("link", "242-245"))
    used = sorted({q for op in ops for q in (op[1:3] if op[0] in ("cx", "dcx") else op[1:2])} | set(ybits))
    for op in ops:
        if op[0] == "dcx":
            L = DUAL_LINKS[_dual_link(op[1], op[2])]
            used = sorted(set(used) | {L["la"], L["lb"]})
    R1 = DualR1()
    A = [q for q in used if R1.module[q] == 0]
    B = [q for q in used if R1.module[q] == 1]
    m = max(len(A), len(B))
    one = not A or not B                     # nothing crosses (inside): one module, one simulator
    if one:
        A, B, m = A + B, [], len(A) + len(B)
    idx = {q: i for i, q in enumerate(A)}
    idx.update({q: m + i for i, q in enumerate(B)})
    mo = DUAL_MODELS[spec["model"]]
    rng = np.random.default_rng(spec["seed"])
    P, peak = np.zeros(1 << spec["t"]), 0
    T = int(spec["trajectories"]) if mo["noise"] != "none" else 1
    for _ in range(T):
        if mo["engine"] == "exact" or one:
            from pyqrack import QrackSimulator
            sim = QrackSimulator(m if one else 2 * m, is_gpu=False)
        else:
            sim = qrack_bridge(spec.get("qrack_bridge"))(2 * m, patches=2, truncation=spec.get("truncation", "greedy"),
                                                       max_bonds=int(spec["max_bonds"] if mo["mb"] is None else mo["mb"]))
        g = DualGates(sim, idx, mo["noise"], spec["r_cz"], spec["r_1q"], rng)
        sha_apply(g, ops)
        if mo["engine"] != "exact" and not one:
            sim.flush()
            peak = max(peak, int(sim.peak_bonds))
        P += sha_marginal(sim.out_ket(), [idx[q] for q in ybits], spec["t"])
    return P / T, dict(peak_bonds=peak, width=(m if one else 2 * m), trajectories=T)


def _sha_register(spec):
    if spec.get("box"):
        return box_register(*spec["box"])
    return loom_register(spec["repo"], spec["cf"], spec["chips"], spec["link_rows"], "exact", "bridge")


def sha_place(R, circ, faces):
    """Logical qubit -> field qubit: side s on chip (face) faces[s], in order of the chip's qubits."""
    free = {c: [q for q in range(len(R.chip_of)) if R.chip_of[q] == c] for c in set(faces)}
    phys = {}
    for q in range(circ["nq"]):
        c = faces[circ["side"][q]]
        if not free[c]:
            raise SystemExit(f"shoralg: chip {R.specs[c]['name']} has too few qubits for this layout")
        phys[q] = free[c].pop(0)
    return phys


def _sha_task(spec):
    """One circuit: exact (QrackSimulator over the logical qubits) or on QrackBridge over the chips or the box's faces
    (one patch per chip, a ring on the box). P(y) from the ket when the bridge is at most spec['ket_max'] qubits
    wide, else from spec['shots'] shots. A bridge whose truncation zeroes the state (possible on the ring) returns
    P None with 'vanished'."""
    circ = sha_circuit(spec["N"], spec["a"], spec["t"], spec["layout"], spec.get("synth", "auto"))
    t = spec["t"]
    if spec["engine"] == "exact":
        from pyqrack import QrackSimulator
        sim = QrackSimulator(circ["nq"], is_gpu=False)
        sha_apply(_SimGates(sim, list(range(circ["nq"]))), circ["ops"])
        return sha_marginal(sim.out_ket(), circ["ybits"], t), dict(readout="ket", width=circ["nq"])
    QB = qrack_bridge(spec.get("qrack_bridge"))
    if spec.get("pair"):                        # two bare circuits, one patch each (no chip geometry)
        per = [[q for q in range(circ["nq"]) if circ["side"][q] == s_] for s_ in (0, 1)]
        if spec["layout"] == "inside":
            per = [per[0] + per[1], []]
        m = max(1, max(len(v) for v in per))
        phys = {q: s_ * m + i for s_ in (0, 1) for i, q in enumerate(per[s_])}
        width, P_, idx = 2 * m, 2, {q: q for q in phys.values()}
    else:
        R = _sha_register(spec)
        faces = [[sp["name"] for sp in R.specs].index(f) for f in spec["faces"]]
        phys = sha_place(R, circ, faces)
        width, P_, idx = _bridge_layout(R, [dict(qubits=list(phys.values()))])
    br = QB(width, max_bonds=int(spec["max_bonds"]), patches=P_, truncation=spec.get("truncation", "greedy"),
            lazy_bonds=int(spec.get("lazy_bonds", 0)), wrap=bool(spec.get("box")),
            seed=[int(spec.get("realisation", 0)), int(spec["N"]), int(spec["a"]), int(spec["max_bonds"]), int(t),
                  zlib_key(spec["layout"] + "/" + "/".join(spec.get("faces") or []))])
    lidx = {q: idx[phys[q]] for q in phys}
    hz = isinstance(br, HBridge)
    snap = br.svc.stats() if hz else None
    stats = lambda: dict(fidelity=float(br.fidelity_estimate), truncations=int(br.truncations),  # noqa: E731
                         peak_bonds=int(br.peak_bonds), width=int(width),
                         **({"hbridge": hbridge_condensed(hbridge_profile_delta(snap, br.svc.stats()))}
                            if hz else {}))
    try:
        with np.errstate(divide="raise", invalid="raise"):
            sha_apply(BridgeGates(br, lidx), circ["ops"])
            br.flush()
            kept = [int(x) for x in br.bond]
            if width <= int(spec.get("ket_max", 22)):
                P = sha_marginal(br.out_ket(), [lidx[q] for q in circ["ybits"]], t)
                return P, dict(stats(), readout="ket", bonds=kept)
            w = br.measure_shots([lidx[q] for q in circ["ybits"]], int(spec["shots"]))
    except (np.linalg.LinAlgError, FloatingPointError) as err:
        return None, dict(stats(), vanished=str(err))
    P = np.bincount(np.array([int(x) for x in w], dtype=np.int64), minlength=1 << t) / float(spec["shots"])
    return P, dict(stats(), readout="shots", bonds=kept)


def sha_cases(a):
    """[(N, a, t)] from --N, --a and --counting (default: the paper's N = 15, a = 7, t = 4, and t = 2n)."""
    out = []
    for N in a.N:
        n = N.bit_length()
        for b in (a.a or ([7] if N == 15 else sha_bases(N))):
            if math.gcd(b, N) != 1:
                raise SystemExit(f"shoralg: gcd({b}, {N}) != 1 (a classical factor; nothing to find)")
            for t in (a.counting or ([4, 2 * n] if N == 15 else [2 * n])):
                out.append((N, b, int(t)))
    return out


def cmd_shoralg(a):
    """Layer 3 across the configurations: every (N, a, t), layout and bridge capacity (or dual model) as its own
    circuit; per circuit the counting register's distribution, the per-shot factoring success, the order found, the
    distance to the ideal distribution, the bridge's bonds and truncations."""
    cases = sha_cases(a)
    layouts = [x for x in a.layouts.split(",") if x]
    for lay in layouts:
        if lay not in SHA_LAYOUTS:
            raise SystemExit(f"shoralg --layouts: {lay} (one of {', '.join(SHA_LAYOUTS)})")
    when = time.strftime("%Y-%m-%dT%H:%M:%S")
    calls, labels, keys = [], [], []
    if a.dual:
        models = [m for m in a.models.split(",") if m]
        layouts = [x for x in layouts if x in ("inside", "split", "direct")]
        geo = dict(name="dual Nighthawk R1", link=a.link)
        for (N, b, t) in cases:
            for lay in layouts:
                for mo in models:
                    if mo not in SHA_DUAL_MODELS:
                        raise SystemExit(f"shoralg --models: {mo}")
                    calls.append((_sha_dual_task, (dict(N=N, a=b, t=t, layout=lay, synth=a.synth, model=mo,
                                                        trajectories=a.trajectories, seed=[a.seed, N, b, t],
                                                        r_cz=a.local_cz_error, r_1q=a.lattice_1q_error,
                                                        max_bonds=max(a.bonds or [a.bridge_max_bonds]),
                                                        link=a.link, qrack_bridge=a.qrack_bridge),)))
                    labels.append(f"dual N {N} a {b} t {t} {lay} {mo}")
                    keys.append((N, b, t, lay, "dual", mo, None))
        R = None
    elif a.pair:
        R, geo = None, dict(name="two circuits")
        base = dict(pair=True, truncation=a.bridge_truncation, lazy_bonds=a.bridge_lazy, qrack_bridge=a.qrack_bridge,
                    shots=a.shots, ket_max=a.ket_max, synth=a.synth)
        for (N, b, t) in cases:
            for lay in layouts:
                if not a.no_exact:
                    calls.append((_sha_task, (dict(base, N=N, a=b, t=t, layout=lay, engine="exact"),)))
                    labels.append(f"exact N {N} a {b} t {t} {lay}")
                    keys.append((N, b, t, lay, "exact", None, None))
                for mb in (a.bonds or [a.bridge_max_bonds]):
                    for rl in range(max(1, a.repeats)):
                        calls.append((_sha_task, (dict(base, N=N, a=b, t=t, layout=lay, engine="bridge", max_bonds=mb,
                                                       realisation=rl),)))
                        labels.append(f"bridge N {N} a {b} t {t} {lay} max_bonds {mb} realisation {rl + 1}")
                        keys.append((N, b, t, lay, "bridge", "A/B" if lay != "inside" else "A", mb))
    else:
        if a.box:
            R = box_register(*a.box)
            a.loom_chips, a.loom_link_rows = list(BOX_FACES), []
            pairs = [p.split("/") for p in (a.faces or ["base/front", "base/right", "base/left"])]
        else:
            a.loom_chips = a.loom_chips or list(LOOM_DEFAULT["chips"])
            R = loom_register(a.repo, a.cloverfield, a.loom_chips, a.loom_link_rows, "exact", "bridge")
            a.loom_link_rows = list(R.link_rows)
            pairs = [p.split("/") for p in (a.faces or [f"{a.loom_chips[0]}/{a.loom_chips[1]}"])]
        names = [sp["name"] for sp in R.specs]
        for p in pairs:
            if len(p) != 2 or any(x not in names for x in p):
                raise SystemExit(f"shoralg --faces: {'/'.join(p)} (chips / faces: {', '.join(names)})")
        geo = dict(name=R.name)
        base = dict(repo=a.repo, cf=a.cloverfield, chips=list(a.loom_chips), link_rows=list(a.loom_link_rows or ()),
                    box=([R.L, R.H] if isinstance(R, BoxRegister) else None), truncation=a.bridge_truncation,
                    lazy_bonds=a.bridge_lazy, qrack_bridge=a.qrack_bridge, shots=a.shots, ket_max=a.ket_max,
                    synth=a.synth)
        bonds = a.bonds or [a.bridge_max_bonds]
        for (N, b, t) in cases:
            for lay in layouts:
                if not a.no_exact:
                    calls.append((_sha_task, (dict(base, N=N, a=b, t=t, layout=lay, engine="exact"),)))
                    labels.append(f"exact N {N} a {b} t {t} {lay}")
                    keys.append((N, b, t, lay, "exact", None, None))
                for pr in (pairs if lay != "inside" else pairs[:1]):
                    for mb in bonds:
                        for rl in range(max(1, a.repeats)):
                            calls.append((_sha_task, (dict(base, N=N, a=b, t=t, layout=lay, engine="bridge",
                                                           max_bonds=mb, realisation=rl,
                                                           faces=list(pr) if lay != "inside" else [pr[0], pr[0]]),)))
                            labels.append(f"bridge N {N} a {b} t {t} {lay} {'/'.join(pr)} max_bonds {mb} "
                                          f"realisation {rl + 1}")
                            keys.append((N, b, t, lay, "bridge", "/".join(pr) if lay != "inside" else pr[0], mb))
    print(f"Layer 3 (Shor order finding) across {geo['name']}: " + ", ".join(
        f"N {N} a {b} (r {sha_order(b, N)}) t {t}" for N, b, t in cases) + f"; layouts {', '.join(layouts)}; "
        + (f"models {a.models}, {a.trajectories} trajectories, link {a.link}" if a.dual else
           f"max_bonds {', '.join(str(x) for x in (a.bonds or [a.bridge_max_bonds]))} ({a.bridge_truncation})")
        + f"; synthesis {a.synth}; {len(calls)} circuits; {qrack_versions()}", flush=True)
    pool, workers = shor_pool(a.jobs, len(calls))
    t0 = time.time()
    try:
        res = shor_pool_map(pool, calls, labels, float(a.progress or 0), what="ghz shoralg")
    finally:
        if pool is not None:
            pool.shutdown()
    secs = time.time() - t0
    grouped = {}
    for k, (P, info) in zip(keys, res):
        grouped.setdefault(k, []).append((P, info))
    recs = []
    for (N, b, t, lay, eng, where, mb), runs in grouped.items():
        Ps = [P for P, _ in runs if P is not None]
        P = np.mean(Ps, axis=0) if Ps else None
        info = dict(runs[0][1])
        if len(runs) > 1:
            ideal_ = sha_analytic(N, b, t)
            sc_ = [sha_score(x, t, N, b, ideal_) for x in Ps]
            info.update(realisations=len(runs), vanished=sum(1 for x, _ in runs if x is None),
                        success_runs=[round(x["success"], 4) for x in sc_], order_runs=[round(x["order"], 4) for x in sc_],
                        tvd_runs=[round(x["tvd"], 4) for x in sc_],
                        peak_bonds=max(i_.get("peak_bonds", 0) for _, i_ in runs),
                        truncations=max(i_.get("truncations", 0) for _, i_ in runs),
                        fidelity=min(i_.get("fidelity", 1.0) for _, i_ in runs))
        r = sha_order(b, N)
        circ = sha_circuit(N, b, t, lay, a.synth)
        ideal = sha_analytic(N, b, t)
        if len(runs) > 1 and any(i_.get("hbridge") for _, i_ in runs):
            info["hbridge"] = hbridge_sum([i_["hbridge"] for _, i_ in runs if i_.get("hbridge")])
        rec = dict(test="shor_alg", when=when, geometry=geo["name"], engine=eng, N=N, a=b, r=r, t=t,
                   bridge_engine=os.environ.get("NIGHTHAWK_BRIDGE_ENGINE", "local"),
                   n_work=N.bit_length(), qubits=circ["nq"], layout=lay, synth=circ["synth"], mults=circ["mults"],
                   **sha_counts(circ["ops"], circ["side"]), bonds_needed=sha_bonds_needed(r, lay),
                   exact=sha_score(ideal, t, N, b, ideal),
                   uniform=sha_score(np.full(1 << t, 1.0 / (1 << t)), t, N, b, ideal),
                   qrack=qrack_versions())
        if eng == "dual":
            _, _, rn = sha_dual_route(circ, a.link)
            rec.update(model=where, link=a.link, route=rn, r_cz=a.local_cz_error, r_1q=a.lattice_1q_error,
                       max_bonds=(DUAL_MODELS[where]["mb"] if DUAL_MODELS[where]["mb"] is not None else
                                  max(a.bonds or [a.bridge_max_bonds])) if DUAL_MODELS[where]["engine"] != "exact"
                       else None)
        elif eng == "bridge":
            f = where.split("/")
            rec.update(faces=f, max_bonds=mb, truncation=a.bridge_truncation, lazy_bonds=a.bridge_lazy,
                       **({"loom": R.map["loom"]} if R is not None else {}))
            if isinstance(R, BoxRegister) and len(f) == 2:
                cuts = R.ring_cuts(BOX_FACES.index(f[0]), BOX_FACES.index(f[1]))
                rec.update(cuts=cuts, hops=len(cuts), wrap=(len(BOX_FACES) - 1) in cuts)
        rec.update({("fidelity_estimate" if k == "fidelity" else k): v for k, v in info.items()})
        if P is not None:
            rec.update(sha_score(P, t, N, b, ideal))
            rec["dist"] = {str(y): round(float(P[y]), 6) for y in np.nonzero(P > 1e-6)[0]}
            if info.get("readout") == "shots":
                rec["tvd_floor"] = float(0.5 * np.sum(np.sqrt(2 * ideal * (1 - ideal) / (math.pi * a.shots))))
        recs.append(rec)
    report_shoralg(recs)
    print(f"seconds: {secs:.1f}")
    if a.out:
        with open(a.out, "a") as fh:
            if R is not None:
                fh.write(json.dumps(dict(test="shor_alg_map", register=R.name, when=when, **R.map)) + "\n")
            for r_ in recs:
                fh.write(json.dumps(r_) + "\n")
        print(f"  appended {len(recs)} records to {a.out}")
    return recs


def report_shoralg(recs):
    """Per (geometry, N, a, t): the ideal circuit's and a uniform guess's success; per layout and placement, the
    success (and the distance to the ideal distribution) at every bridge capacity or model, the bonds the layout needs
    by the rule and the peak the bridge held."""
    groups = collections.defaultdict(list)
    for r in recs:
        groups[(r["geometry"], r["N"], r["a"], r["t"])].append(r)
    for (geo, N, b, t), rs in sorted(groups.items()):
        r0 = rs[0]
        key = "success" if r0["exact"]["success"] > 0 else "order"
        print(f"\n{geo}: N {N}, a {b}, order r {r0['r']}, {t} counting + {r0['n_work']} work qubits; ideal: success "
              f"{r0['exact']['success']:.3f}, order found {r0['exact']['order']:.3f}; uniform guess: success "
              f"{r0['uniform']['success']:.3f}, order {r0['uniform']['order']:.3f}"
              + ("" if key == "success" else "; odd r gives no factor: cells show the order found"))
        cols = sorted({(r.get("model") or "", -1 if r.get("max_bonds") is None else r["max_bonds"]) for r in rs
                       if r["engine"] != "exact"}, key=lambda c: (SHA_DUAL_MODELS.index(c[0]) if c[0] else 0, c[1]))
        head = [c[0] if c[0] else f"mb {c[1]}" for c in cols]
        print(f"  {'layout':<11}{'placement':<22}{'CX':>6}{'cross':>6}{'need':>5}" + "".join(f"{h:>15}" for h in head)
              + f"{'peak':>6}{'exact':>7}")
        rows = collections.defaultdict(dict)
        for r in rs:
            if r["engine"] == "exact":
                rows[(r["layout"], "")]["exact"] = r
                continue
            pl = f"link {r.get('link')}" if r["engine"] == "dual" else " / ".join(r.get("faces") or [])
            if r.get("hops"):
                pl += " (" + ("relayed" if r["hops"] > 1 else "direct") + (", wrap" if r.get("wrap") else "") + ")"
            rows[(r["layout"], pl)][(r.get("model") or "", -1 if r.get("max_bonds") is None else r["max_bonds"])] = r
        for (lay, pl), cell in sorted(rows.items(), key=lambda kv: (SHA_LAYOUTS.index(kv[0][0]), kv[0][1])):
            if not pl:
                continue
            any_ = next(iter(v for k, v in cell.items() if k != "exact"))
            ex = rows.get((lay, ""), {}).get("exact")
            line = (f"  {lay:<11}{pl[:21]:<22}{any_['cx']:>6}{any_['cross']:>6}"
                    + f"{'-' if any_['bonds_needed'] is None else any_['bonds_needed']:>5}")
            for c in cols:
                r = cell.get(c)
                if r is None:
                    line += f"{'':>15}"
                elif "success" not in r:
                    line += f"{'vanished':>15}"
                else:
                    line += f"{r[key]:>7.3f} ({r['tvd']:.2f})"
            line += f"{max((r.get('peak_bonds') or 0) for k, r in cell.items() if k != 'exact'):>6}"
            line += f"{ex['tvd']:>7.0e}" if ex and "tvd" in ex else f"{'':>7}"
            print(line)
    hp = [r["hbridge"] for r in recs if r.get("hbridge")]
    if hp:
        t = hbridge_sum(hp)
        print("\n  horizontal bridge, all circuits: " + "; ".join(
            f"patch server {i} ({p['device']}) gates {p['op_s']:.1f} s, ket out {p['export_s']:.1f} s, in "
            f"{p['load_s']:.1f} s" for i, p in enumerate(t["patch"])) + f"; glue {t['glue_s']:.1f} s ("
            + ", ".join(f"{k} {v:.1f} s" for k, v in sorted(t["glue"].items())) + f"); coordinator waiting "
            f"{t['wait_s']:.1f} s")
    print("\n  cells: per-shot factoring success (the paper's rule; the order found where r is odd) and, in brackets, the "
          "total variation distance to the ideal distribution (shots: above the floor tvd_floor in the records); need: bond qubits per cut the rule says the exact output takes; exact: the "
          "QrackSimulator run's distance to the analytic distribution")


# ================================================================== sweep, probe, stackmap

def chain_from_end(u):
    """The same GHZ as a chain from one end of a chain unit: H on the end, CX along the chain (the probe's chain
    variants; nighthawk_qrack.py ghz's tiles root a chain in its middle)."""
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
    fig.text(0.02, foot[2], f"{os.path.basename(a.inp)} · run {when} · {st[0].get('qrack', '')} · nighthawk_qrack.py ghz stack "
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
    fig.text(0.02, 0.30 / FH, f"{os.path.basename(a.inp)} · run {when} · {st[0].get('qrack', '')} · nighthawk_qrack.py ghz "
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


def cmd_ghz_selftest(a):
    from pyqrack import QrackSimulator
    ok = True

    failed = []

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        if not cond:
            failed.append(name[:70])
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
        repo = str(find_repo(None))
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
            _, local = _local_unit(u)
            for mb in (0, 1, 3):
                for st in ("0", "+"):
                    sh, _ = _shor_task(dict(unit=u, state=st, shots=256, exact=False, realisation=0, index=0,
                                            box=[5, 5], max_bonds=mb))
                    got[(lay, mb, st)] = float(np.mean([shor9_decode(x, local, st)[1] for x in sh]))
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
    print("[20] shoralg: Layer 3 as gates; the controlled multiplications against the paper's U; the circuit against "
          "the analytic distribution; the paper's success rule; the hard cut on QrackBridge for r = 4 and r = 6; the "
          "dual router")
    from fractions import Fraction
    bad = 0
    for N, c, synth in ((15, 7, "structured"), (15, 13, "structured"), (15, 2, "structured"), (15, 7, "generic"),
                        (21, 2, "generic"), (21, 4, "generic")):
        n = N.bit_length()
        o = ShaOps()
        sha_cmul(o, 0, list(range(1, n + 1)), list(range(n + 1, 2 * n - 1)), c, N, synth)
        for y in range(1 << n):
            for cv in (0, 1):
                sim = QrackSimulator(2 * n - 1, is_gpu=False)
                for q in range(2 * n - 1):
                    if ((y << 1) | cv) >> q & 1:
                        sim.x(q)
                sha_apply(_SimGates(sim, list(range(2 * n - 1))), o.ops)
                v = sim.m_all()
                want = ((c * y) % N if 0 < y < N else y) if cv else y
                bad += int(((v >> 1) & ((1 << n) - 1)) != want or (v & 1) != cv or (v >> (n + 1)) != 0)
    check("controlled multiplication = the paper's U (y -> c y mod N on 0 < y < N, else y; ancillas back to 0) on every "
          "basis state: N 15 c 7, 13, 2 structured, 7 generic; N 21 c 2, 4 generic", bad == 0, f"{bad} wrong")
    tv = {}
    for N, b, t in ((15, 7, 4), (21, 2, 6)):
        for lay in ("split", "direct"):
            P, _ = _sha_task(dict(N=N, a=b, t=t, layout=lay, engine="exact"))
            tv[(N, lay)] = 0.5 * np.abs(P - sha_analytic(N, b, t)).sum()
    check("the gate circuit's counting distribution = the analytic one (N 15 a 7 t 4, N 21 a 2 t 6; split, direct)",
          max(tv.values()) < 1e-5, f"worst TVD {max(tv.values()):.1e}")
    paper = [y for y in range(16) if y and Fraction(y, 16).limit_denominator(15).denominator % 2 == 0
             and pow(7, Fraction(y, 16).limit_denominator(15).denominator, 15) == 1
             and sha_factor(15, 7, Fraction(y, 16).limit_denominator(15).denominator)]
    sc = sha_score(sha_analytic(15, 7, 4), 4, 15, 7)
    check("the paper's figure of merit: N 15 a 7 t 4 good bins 4 and 12, ideal per-shot success 1/2",
          paper == [4, 12] and abs(sc["success"] - 0.5) < 1e-9, f"bins {paper}, success {sc['success']:.4f}")
    try:
        qrack_bridge(None)
    except SystemExit as err:
        print(f"  [skip] {err}")
    else:
        bx = dict(pair=True, engine="bridge", shots=1, ket_max=40)
        got = {}
        for N, b, t, mb in ((15, 7, 6, 0), (21, 2, 6, 0), (21, 2, 6, 4)):
            P, _ = _sha_task(dict(bx, N=N, a=b, t=t, layout="split", max_bonds=mb))
            got[(N, mb)] = 0.5 * np.abs(P - sha_analytic(N, b, t)).sum()
        check("QrackBridge, counting | work on two circuits: r = 4 needs no bond (exact at max_bonds 0, the hard cut); "
              "r = 6 needs ceil(log2 6) + 1 = 4 (far off at 0, exact at 4)",
              got[(15, 0)] < 1e-6 and got[(21, 0)] > 0.5 and got[(21, 4)] < 1e-6,
              f"TVD {got[(15, 0)]:.1e} / {got[(21, 0)]:.2f} / {got[(21, 4)]:.1e}")
        dv = {}
        for mo in ("exact", "ideal"):
            P, _ = _sha_dual_task(dict(N=15, a=7, t=4, layout="split", model=mo, trajectories=1, seed=[0], r_cz=0.0,
                                       r_1q=0.0, max_bonds=6))
            dv[mo] = 0.5 * np.abs(P - sha_analytic(15, 7, 4)).sum()
        check("dual router (module lines, SWAPs, the distant CNOT through 242-245), noiseless: N 15 a 7 t 4 exact on "
              "QrackSimulator and on the bridge", max(dv.values()) < 1e-5, f"TVD {dv['exact']:.1e} / {dv['ideal']:.1e}")
    print("selftest " + ("passed" if ok else "FAILED: " + " | ".join(failed)))
    if not ok:
        raise SystemExit(1)


# ================================================================== command line

def ghz_make_parser():
    ap = argparse.ArgumentParser(prog="nighthawk_qrack.py ghz", description=GHZ_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
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
    bridge_opts.add_argument("--bridge-engine", dest="bridge_engine", choices=["local", "horizontal"], default=None,
                             help="local: PyQrack's QrackBridge in this process (default); horizontal: "
                                  "nighthawk_qrack.py's HBridge, each patch on its own patch server (PyQrack "
                                  "instance), the recompression and readout on a glue server")
    bridge_opts.add_argument("--hbridge", default=None,
                             help="horizontal bridge spec (nighthawk_qrack.py parse_hbridge): devices=cpu,gpu:0,...;"
                                  "patch_cpus=0-3/4-7;glue_cpus=8-11;glue_threads=4;glue_fast=1;batch=512")
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
    dm.add_argument("--bridge-engine", dest="bridge_engine", choices=["local", "horizontal"], default=None,
                    help="as stack: local QrackBridge, or the horizontal HBridge")
    dm.add_argument("--hbridge", default=None, help="as stack: the horizontal bridge spec")
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
    du.add_argument("--bridge-engine", dest="bridge_engine", choices=["local", "horizontal"], default=None,
                    help="as stack: local QrackBridge, or the horizontal HBridge")
    du.add_argument("--hbridge", default=None, help="as stack: the horizontal bridge spec")
    du.add_argument("--qrack-bridge", dest="qrack_bridge", default=None, help="qrack_bridge.py (PyQrack < 2.30)")
    du.add_argument("--seed", type=int, default=7, help="trajectory seed")
    du.add_argument("--jobs", type=int, default=1, help="worker processes (0 = every core)")
    du.add_argument("--progress", type=float, default=30.0, help="seconds between heartbeats (0 = off)")
    du.add_argument("--out", default=None, help="append dual_r1 JSONL records")
    du.add_argument("--png", default=None, help="model against measurement, one row per state")

    sa = sub.add_parser("shoralg", parents=[bridge_opts],
                        help="Layer 3, the paper's Shor order finding (N = 15, a = 7), as gates across QrackBridge "
                             "circuits (two chips, the box) or the dual Nighthawk R1")
    sa.add_argument("--repo", default=None, help="rcs-nighthawk clone (the chips; default: auto-detect)")
    sa.add_argument("--N", type=int, nargs="+", default=[15, 21], help="numbers to factor (default 15 21)")
    sa.add_argument("--a", type=int, nargs="+", default=None,
                    help="bases (default: 7 for N = 15, the paper's; else the smallest base of every order r > 1)")
    sa.add_argument("--counting", type=int, nargs="+", default=None,
                    help="counting qubits t (default: 4, the paper's, and 2n for N = 15; 2n otherwise)")
    sa.add_argument("--layouts", default=",".join(SHA_LAYOUTS),
                    help="comma list of " + ", ".join(f"{k} ({v})" for k, v in SHA_LAYOUT_TEXT.items()))
    sa.add_argument("--bonds", type=int, nargs="+", default=None,
                    help="max_bonds values to sweep (default: --bridge-max-bonds alone)")
    sa.add_argument("--faces", nargs="+", default=None,
                    help="side pairs A/B, counting side first (default: the two chips; with --box base/front "
                         "(direct), base/right (relayed through front), base/left (the ring's wrap cut))")
    sa.add_argument("--synth", choices=["auto", "generic", "structured"], default="auto",
                    help="controlled modular multiplication: generic (the permutation's transpositions), structured "
                         "(N = 2^n - 1, a^(2^k) = +-2^j: a bit rotation), auto (structured where it applies)")
    sa.add_argument("--shots", type=int, default=4096, help="shots when the bridge is wider than --ket-max")
    sa.add_argument("--repeats", type=int, default=1,
                    help="bridge runs per circuit: below capacity a greedy truncation picks among equal Schmidt values, "
                         "a tie that Qrack's rounding breaks differently on every run (the seed does not fix it); P is "
                         "their mean, success_runs each one")
    sa.add_argument("--ket-max", dest="ket_max", type=int, default=22,
                    help="bridges up to this many qubits are read from their ket (no shot noise)")
    sa.add_argument("--no-exact", dest="no_exact", action="store_true", help="skip the QrackSimulator runs")
    sa.add_argument("--dual", action="store_true", help="on the dual Nighthawk R1 (arXiv:2610.29986) instead")
    sa.add_argument("--pair", action="store_true",
                    help="on two bare QrackBridge circuits (one patch per side, no chip geometry, no release needed)")
    sa.add_argument("--models", default="hardcut,ideal,budget,measured", help="--dual: as dual")
    sa.add_argument("--trajectories", type=int, default=200, help="--dual: Pauli trajectories per noisy circuit")
    sa.add_argument("--link", default="242-245", help="--dual: the L-coupler the distant CNOTs use")
    sa.add_argument("--local-cz-error", dest="local_cz_error", type=float, default=0.00355)
    sa.add_argument("--lattice-1q-error", dest="lattice_1q_error", type=float, default=0.0005)
    sa.add_argument("--seed", type=int, default=7, help="--dual: trajectory seed")
    sa.add_argument("--jobs", type=int, default=1, help="worker processes (0 = every core)")
    sa.add_argument("--progress", type=float, default=30.0, help="seconds between heartbeats (0 = off)")
    sa.add_argument("--out", default=None, help="append shor_alg JSONL records")

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


def ghz_main(argv=None):
    a = ghz_make_parser().parse_args(argv)
    if getattr(a, "bridge_engine", None):                 # inherited by the worker pool
        os.environ["NIGHTHAWK_BRIDGE_ENGINE"] = a.bridge_engine
    if getattr(a, "hbridge", None):
        parse_hbridge(a.hbridge)
        os.environ["NIGHTHAWK_HBRIDGE"] = a.hbridge
    if getattr(a, "ocl_sync", False):
        os.environ["QRACK_OCL_SYNC"] = "1"
    if a.cmd in ("chip", "tiles", "stack", "sweep", "probe", "shor", "shoralg"):
        ensure_isolated(a)
    dict(chip=cmd_chip, tiles=cmd_tiles, stack=cmd_stack, sweep=cmd_sweep, probe=cmd_probe, stackmap=cmd_stackmap,
         shor=cmd_shor, shormap=cmd_shormap, shoralg=cmd_shoralg, dual=cmd_dual, dualmap=cmd_dualmap, dualplot=cmd_dualplot, plot=cmd_plot, selftest=cmd_ghz_selftest)[a.cmd](a)



# ================================================================== CLI
def make_parser():
    """The command-line parser (also fills _RUN_PARSER, which shor9 and nighthawk_graph.py use to build a
    run's ACE settings for a register)."""
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
    s9.add_argument("--progress", type=float, default=30.0, metavar="SECONDS",
                    help="progress lines (start, plan, every finished task with its time and an ETA) and a heartbeat "
                         "naming the longest-running task when nothing finishes for this many seconds (default 30; 0 = off)")
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

    bb = sub.add_parser("bridgebench", help="horizontal QrackBridge (patch servers + glue server) against PyQrack's "
                                            "own on random circuits: states, shots, wall times, per-workload profile")
    bb.add_argument("--n", type=int, default=20, help="qubits")
    bb.add_argument("--patches", type=int, default=2, help="bridge patches")
    bb.add_argument("--wrap", action="store_true", help="patches on a ring (3+)")
    bb.add_argument("--depth", type=int, default=12, help="layers of U3 on every qubit + CX on a random matching")
    bb.add_argument("--cross", type=float, default=0.2, help="share of each layer's CXs between neighbouring patches")
    bb.add_argument("--max-bonds", dest="max_bonds", type=int, default=4, help="bond qubits per cut")
    bb.add_argument("--lazy", type=int, default=0, help="lazy_bonds")
    bb.add_argument("--reps", type=int, default=3, help="circuits (seeds)")
    bb.add_argument("--seed", type=int, default=11)
    bb.add_argument("--shots", type=int, default=1024)
    bb.add_argument("--ket-max", dest="ket_max", type=int, default=24, help="compare kets up to this many qubits")
    bb.add_argument("--hbridge", default=None,
                    help="horizontal bridge spec (default $NIGHTHAWK_HBRIDGE or '" + HB_SPEC_DEFAULT + "'): "
                         "devices=cpu,gpu:0,...;patch_cpus=0-3/4-7;glue_cpus=8-11;glue_threads=4;patch_threads=N;"
                         "host_pointer=0|1;batch=512")
    bb.add_argument("--qrack-bridge", dest="qrack_bridge", default=None,
                    help="PyQrack's qrack_bridge.py for the local reference (PyQrack < 2.30)")
    bb.add_argument("--check", action="store_true", help="exit 1 unless horizontal = local")
    bb.add_argument("--out", default=None, help="append a bridgebench record (JSONL)")

    sub.add_parser("ghz", add_help=False,
                   help="the formal proof's Layer 1a / 2, seam layers, the loom / bridge / box engines, Shor's code "
                        "and Layer 3 across them, the dual Nighthawk R1 (formerly nighthawk_ghz.py; "
                        "nighthawk_qrack.py ghz -h)")

    t = sub.add_parser("selftest", help="conventions and exactness checks")
    t.add_argument("--repo", default=None, help="rcs-nighthawk clone (default: auto-detect)")
    t.add_argument("--n", type=int, default=14, help="register for the exact mirror checks")
    t.add_argument("--lrc", type=int, default=4)
    t.add_argument("--lrr", type=int, default=4)
    return ap


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_hbridge-server":     # a horizontal bridge's server process
        hbridge_server(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "ghz":                 # the ghz commands, with their own parser
        ghz_main(sys.argv[2:])
        return
    load_register_files()
    ap = make_parser()
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
         shor9=cmd_shor9, bridgebench=cmd_bridgebench)[a.cmd](a)


if __name__ == "__main__":
    main()
