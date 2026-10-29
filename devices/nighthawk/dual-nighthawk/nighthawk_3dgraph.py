#!/usr/bin/env python3
"""
nighthawk_3dgraph.py -- interactive 3D views, in Python (matplotlib, no browser), of the telemetry and
measurements of every shape the nighthawk tools write: the Nighthawk lattice of nighthawk_qrack.py's runs and
shor9 registers, nighthawk_ghz.py's registers (ACE patches, two chips joined by the loom or QrackBridge, the box
of six faces on a ring), Shor's code across QrackBridge, and the dual Nighthawk R1 of arXiv:2610.29986.

Dark mode by default (--theme light, or auto to follow the desktop where it can tell; T switches at any time).

Shapes (one scene per register or record family; each scene has data sets and measures):
  runs, landscape      nighthawk_qrack.py run: per family (mirror, patched, fxeb, hamming, ...) and configuration,
                       one column per (register size, depth); measures are every numeric field the records carry:
                       fidelity, se, seconds (and log10), hog, xeb_linear, survival, ace_seam_used, ...
  runs, bits           the same runs' bitstrings (<out>_shots/), per qubit on the lattice: mirror bit-flip rate
                       against the prepared string, P(1) and its bias for the other families
  shor9 register       nighthawk_qrack.py shor9 --register: each unit's square on the lattice, by ACE patch;
                       logical Z, X, the weaker of the two, the Z-basis impossible share
  GHZ register         nighthawk_ghz.py tiles / stack / chip: every site's F_GHZ (the engine's read: ACE, loom or
                       QrackBridge), the exact reference, their difference, coherence C, population, lowest F,
                       units through the site; chip-wide states' agreement with the majority. Coupler telemetry:
                       each coupler's break rate. Loom and bridge chips sit apart, the box is folded into a cube
                       with the columns standing out of its faces; QrackBridge passes (peak bonds, truncations,
                       fidelity estimates) in the telemetry panel
  Shor across bridge   nighthawk_ghz.py shor: per code size and max_bonds, decoded and raw logical Z and X, bonds
                       kept for Z and X, peak bonds, impossible share; the encoder CXs of a picked unit
  dual Nighthawk R1    nighthawk_ghz.py dualmap / dual: both 12 x 10 modules, the L-qubits and the four cables
                       (arcs labelled with their distant-CZ fidelity); per model the lowest / mean F_GHZ of the
                       3-qubit chains through each site and their peak bonds; every paper state per model with
                       IBM's raw and mitigated values

Mouse: drag rotates, scroll zooms, a click on a column picks the site (its units are drawn and listed).
Keys:  up / down shape · left / right data set · m / M measure · c columns or dots · v values · n names
       e pull the chips apart · k coupler telemetry · t theme · r reload · p save PNG · home / 0 reset view
       1 top view · 2 side view

Records are found as nighthawk_graph.py finds them (it is imported for its readers): every *.jsonl in the
working directory, or --out / --shor / --ghz files; --repo for the BlueQubit release (the lattice of the run
records). Without a display, or with --save DIR, it writes PNGs instead of opening the window.

Usage
  python3 nighthawk_3dgraph.py                                   # every *.jsonl here, window
  python3 nighthawk_3dgraph.py --ghz box_mb2_gate.jsonl --ghz dualmap.jsonl --ghz dual_r1.jsonl
  python3 nighthawk_3dgraph.py --out run.jsonl --repo ../rcs-nighthawk --scene runs
  python3 nighthawk_3dgraph.py --list                            # the scenes and data sets found
  python3 nighthawk_3dgraph.py --save shots3d --all-datasets     # PNGs, no window
  python3 nighthawk_3dgraph.py --theme light --scene box --measure exact
"""

import os

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):
    os.environ["PYQRACK_SHARED_LIB_PATH"] = QRACK_LIB_PATH

import argparse
import importlib.util
import json
import math
import re
import subprocess
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np


# ================================================================== the graph module (readers, labels)
def load_graph():
    p = Path(__file__).resolve().parent / "nighthawk_graph.py"
    if not p.exists():
        raise SystemExit(f"nighthawk_graph.py not found next to this script ({p.parent})")
    spec = importlib.util.spec_from_file_location("nighthawk_graph", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nighthawk_graph"] = mod
    spec.loader.exec_module(mod)
    return mod


G = None                                    # nighthawk_graph, loaded in main() after the backend is chosen


# ================================================================== themes
# Sequential blue ramp, 100 -> 700, and the categorical slots in their validated order (dataviz reference palette).
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
       "#1c5cab", "#184f95", "#104281", "#0d366b"]
THEMES = {
    "dark": dict(
        name="dark", surface="#1a1a19", panel="#212120", control="#2d2d2b", hover="#3b3b38", text="#f0efec",
        text2="#c3c2b7", muted="#8f8e84", faint="#45443f", grid="#383835", coupler="#5e5d57", link="#f0efec",
        seam="#9085e9", replica="#d95926", cross="#e66767", nodata="#4f4e49", accent="#3987e5",
        cat=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
        seq=SEQ[10::-1], div=["#e66767", "#383835", "#3987e5"], light=0.62),
    "light": dict(
        name="light", surface="#fcfcfb", panel="#f4f3f0", control="#e8e7e3", hover="#d9d8d3", text="#0b0b0b",
        text2="#52514e", muted="#7c7b75", faint="#d4d3cd", grid="#e4e3df", coupler="#b4b3ac", link="#0b0b0b",
        seam="#6250d6", replica="#eb6834", cross="#e34948", nodata="#d4d3cd", accent="#2a78d6",
        cat=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
        seq=SEQ[1:], div=["#e34948", "#f0efec", "#2a78d6"], light=0.55),
}


def desktop_theme():
    """'dark' or 'light' from the desktop where it can be read (GNOME / GTK, KDE, macOS, Windows), else 'dark'."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["defaults", "read", "-g", "AppleInterfaceStyle"], capture_output=True, text=True,
                                 timeout=2)
            return "dark" if "dark" in out.stdout.lower() else "light"
        if sys.platform == "win32":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                               r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
            return "light" if winreg.QueryValueEx(k, "AppsUseLightTheme")[0] else "dark"
        gt = os.environ.get("GTK_THEME", "")
        if gt:
            return "dark" if "dark" in gt.lower() else "light"
        for cmd in (["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
                    ["gsettings", "get", "org.gnome.desktop.interface", "gtk-theme"]):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=2).stdout.lower()
            except (OSError, subprocess.TimeoutExpired):
                continue
            if "dark" in out:
                return "dark"
            if out.strip() and "default" not in out:
                return "light"
        kde = Path.home() / ".config/kdeglobals"
        if kde.exists() and "dark" in kde.read_text(errors="ignore").lower():
            return "dark"
    except Exception:
        pass
    return "dark"


# ================================================================== scene model
class Measure:
    """One quantity on a scene's sites (or a landscape's cells): values, colour range, seq / div scale."""

    def __init__(self, name, values, lo=None, hi=None, kind="seq", fmt="{:.3f}", note=""):
        self.name, self.kind, self.fmt, self.note = name, kind, fmt, note
        self.values = {k: float(v) for k, v in values.items() if v is not None and math.isfinite(float(v))}
        vs = list(self.values.values())
        if kind == "div":
            m = max([abs(v) for v in vs] + [0.05])
            lo, hi = (-m, m) if lo is None else (lo, hi)
        if lo is None:
            lo = min(vs) if vs else 0.0
        if hi is None:
            hi = max(vs) if vs else 1.0
        if hi <= lo:
            hi = lo + (abs(lo) * 0.1 or 1.0)
        self.lo, self.hi = float(lo), float(hi)

    def frac(self, v):
        return min(1.0, max(0.0, (v - self.lo) / (self.hi - self.lo)))


class Dataset:
    """One data set of a scene: telemetry lines, measures, units (for picks), coupler values, marked qubits."""

    def __init__(self, label, short=None):
        self.label, self.short = label, short or label
        self.info = []                      # (key, value) telemetry lines
        self.measures = {}                  # name -> Measure
        self.units = []                     # dict(qubits, edges, text)
        self.edges = {}                     # (a, b) -> value, coupler telemetry
        self.edge_name = ""
        self.mark = set()
        self.cells = {}                     # landscape: (i, j) -> detail text
        self.lazy = None                    # callable filling measures on first use

    def ready(self):
        if self.lazy is not None:
            f, self.lazy = self.lazy, None
            try:
                f(self)
            except Exception as err:        # a missing or broken shots file: the data set stays empty
                self.info.append(("error", f"{type(err).__name__}: {err}"))
        return self


class Scene:
    """A shape: sites (position, outward direction, group), couplers, links, groups, data sets."""

    def __init__(self, family, name, kind="lattice"):
        self.family, self.name, self.kind = family, name, kind
        self.sites = {}                     # q -> dict(p, up, group)
        self.couplers = []                  # (a, b, cls)
        self.links = []                     # (a, b, label, arc)
        self.groups = []
        self.group_dir = {}                 # group -> pull-apart direction
        self.height = 2.0
        self.datasets = []
        self.notes = []
        self.xs, self.ys, self.xname, self.yname = [], [], "", ""     # landscape axes
        self.view = (34, -62)
        self.plates = False                 # draw each group's footprint (chips, faces, modules)
        self.half = 0.3                     # column half-width

    @property
    def title(self):
        return f"{self.family} · {self.name}"

    def add_site(self, q, p, group, up=(0.0, 0.0, 1.0)):
        self.sites[int(q)] = dict(p=np.asarray(p, float), up=np.asarray(up, float), group=group)
        if group not in self.groups:
            self.groups.append(group)

    def finish(self, mode="radial"):
        """Pull-apart directions: radial (from the centre to each group's centre, in the plane), or normal (the
        groups' own outward direction: box faces)."""
        if not self.sites:
            return self
        P = np.array([s["p"] for s in self.sites.values()])
        c = P.mean(axis=0)
        for g in self.groups:
            if g == "seam":
                self.group_dir[g] = np.zeros(3)
                continue
            qs = [s for s in self.sites.values() if s["group"] == g]
            if mode == "normal":
                d = np.mean([s["up"] for s in qs], axis=0)
            else:
                d = np.mean([s["p"] for s in qs], axis=0) - c
                d[2] = 0.0
            nd = np.linalg.norm(d)
            self.group_dir[g] = d / nd if nd > 1e-9 and len(self.groups) > 1 else np.zeros(3)
        return self


TEXT_Z = 1e6                                # labels above the depth-sorted 3D collections (Axes3D numbers those from 1 up)


def r4(v):
    return None if v is None or not math.isfinite(float(v)) else float(v)


def mean_by_site(units, field):
    acc = defaultdict(list)
    for u in units:
        v = u.get(field)
        if v is None or not math.isfinite(float(v)):
            continue
        for q in u["qubits"]:
            acc[int(q)].append(float(v))
    return acc


def edge_key(a, b):
    a, b = int(a), int(b)
    return (a, b) if a < b else (b, a)


# ================================================================== lattice geometry from the map records
def map_geometry(sc, mp):
    """Sites, groups and couplers of a ghz_map / shor_map / shor9_map record. Loom and bridge maps: one group per
    chip; the box: the xyz each site carries and its face; ACE maps: the patches and the seam."""
    lo = mp.get("loom") or {}
    chips = list(lo.get("chips") or [])
    ends = list(np.cumsum(lo.get("chip_n") or []))
    chip = (lambda q: next((i for i, e in enumerate(ends) if q < e), len(ends) - 1)) if ends else None
    box = any(len(st) > 5 and isinstance(st[5], str) and isinstance(st[4], (list, tuple)) and len(st[4]) == 3
              for st in mp["sites"])          # box maps: [q, row, col, regions, xyz, face]
    for st in mp["sites"]:
        q, row, col = int(st[0]), int(st[1]), int(st[2])
        if chip is not None:
            g = chips[chip(q)] if chip(q) < len(chips) else f"chip {chip(q)}"
        elif len(st) > 3 and isinstance(st[3], str):
            g = st[3]
        else:
            g = "lattice"
        if box and len(st) > 4 and isinstance(st[4], (list, tuple)):
            sc.add_site(q, st[4], st[5] if len(st) > 5 and isinstance(st[5], str) else g)
        else:
            sc.add_site(q, (col, -row, 0.0), g)
    if box:                                         # outward normals: from the cube's centre to each face's centre
        P = np.array([s["p"] for s in sc.sites.values()])
        c = (P.min(axis=0) + P.max(axis=0)) / 2
        for g in sc.groups:
            qs = [s for s in sc.sites.values() if s["group"] == g]
            d = np.mean([s["p"] for s in qs], axis=0) - c
            n = np.zeros(3)
            n[int(np.argmax(np.abs(d)))] = np.sign(d[int(np.argmax(np.abs(d)))]) or 1.0
            for s in qs:
                s["up"] = n
        sc.height, sc.half = 1.0, 0.17
        sc.view = (24, -52)
    links = {edge_key(a, b) for a, b in (lo.get("links") or [])}
    for cp in mp.get("couplers") or []:
        a, b, cls = int(cp[0]), int(cp[1]), (cp[2] if len(cp) > 2 else "exact")
        if a not in sc.sites or b not in sc.sites:
            continue
        if edge_key(a, b) in links:
            sc.links.append((a, b, None, False))
        else:
            sc.couplers.append((a, b, cls))
    sc.finish("normal" if box else "radial")
    sc.plates = bool(chips) or box
    if lo:
        read = lo.get("read", "ghost")
        sc.notes.append("joined by " + ("QrackBridge" if read == "bridge" else "the loom, hard cut" if read == "cut"
                                         else "the loom, ghost band") + f"; chip backend {lo.get('backend', '?')}")
    return sc


# ================================================================== GHZ registers (nighthawk_ghz.py)
def ghz_scenes(store, files):
    out = []
    regs = sorted({r["register"] for r in store.ghz if r.get("register")})
    bridges = G.read_tests(files, "ghz_bridge", lambda r: (r.get("when"), r.get("register"),
                                                             json.dumps(r.get("passes"))[:200]))
    tiles = G.ghz_tiles(store.ghz)
    chips = G.ghz_chips(store.ghz)
    for reg in regs:
        mp = G.ghz_register_map(store, reg)
        if not mp:
            continue
        sc = map_geometry(Scene("GHZ", reg), mp)
        for key in sorted((k for k in tiles if k[0] == reg), key=lambda k_: (k_[1], k_[2], G._setup_rank(k_[3]))):
            _, k, layers, setup = key
            for mode in G.GHZ_MODES:
                rs = tiles[key].get(mode)
                if rs:
                    sc.datasets.append(ghz_tiles_dataset(reg, k, layers, setup, mode, rs, bridges))
        for key in sorted((k for k in chips if k[0] == reg), key=lambda k_: (k_[1], str(k_[2]))):
            sc.datasets.append(ghz_chip_dataset(key, chips[key]))
        if sc.datasets:
            out.append(sc)
    return out


def _setup_text(r):
    lng = G.shor_setup(r)[0]
    return lng or "ACE"


def ghz_tiles_dataset(reg, k, layers, setup, mode, rs, bridges):
    mlab = G.GHZ_MODE_LABEL.get(mode, mode)
    ds = Dataset(f"{k}-qubit chains, {mlab}" + (f", {layers}" if layers not in (None, "none") else "")
                 + (f" · {setup}" if setup else ""), short=f"{k}q · {G.GHZ_MODE_SHORT.get(mode, mode)}"
                 + (f" · {G.shor_setup(rs[0])[1]}" if setup else ""))
    units = []
    for r in rs:
        F, xF = G.ghz_F(r)[1], G.ghz_F(r, "exact_")[1]
        units.append(dict(qubits=[int(q) for q in r.get("qubits", [])], edges=[tuple(e) for e in r.get("edges", [])],
                          F=F, xF=xF, d=(F - xF) if F is not None and xF is not None else None,
                          C=r4(r.get("C_parity", r.get("coherence"))), pop=r4(r.get("pop")), place=G.ghz_place(r)[1],
                          text=f"{r.get('tiling', '')} unit {r.get('unit')}: F {F:.3f}"
                               + (f" (exact {xF:.3f})" if xF is not None else "")
                               + (f", C {r.get('C_parity'):.3f}" if r.get("C_parity") is not None else "")
                               + f" · {G.ghz_place(r)[1]}"))
    key = G.ghz_F(rs[0])[0] or "F"
    per = mean_by_site(units, "F")
    ds.measures[f"{key}, engine read"] = Measure(f"{key}, engine read", {q: np.mean(v) for q, v in per.items()},
                                                0.5, 1.0, note="mean over the units holding the site")
    xper = mean_by_site(units, "xF")
    if xper:
        ds.measures[f"{key}, exact reference"] = Measure(f"{key}, exact reference",
                                                        {q: np.mean(v) for q, v in xper.items()}, 0.5, 1.0)
        dper = mean_by_site(units, "d")
        ds.measures["engine − exact"] = Measure("engine − exact", {q: np.mean(v) for q, v in dper.items()},
                                                kind="div", fmt="{:+.3f}")
    ds.measures["lowest F through the site"] = Measure("lowest F through the site",
                                                       {q: min(v) for q, v in per.items()}, 0.5, 1.0)
    cper = mean_by_site(units, "C")
    if cper:
        ds.measures["coherence C"] = Measure("coherence C", {q: np.mean(v) for q, v in cper.items()}, 0.0, 1.0)
    pper = mean_by_site(units, "pop")
    if pper:
        ds.measures["population P0 + P1"] = Measure("population P0 + P1", {q: np.mean(v) for q, v in pper.items()},
                                                    0.5, 1.0)
    ds.measures["units through the site"] = Measure("units through the site", {q: len(v) for q, v in per.items()},
                                                    0, None, fmt="{:.0f}")
    acc = defaultdict(list)
    for r in rs:
        for cb in r.get("coupler_break") or []:
            acc[edge_key(cb[0], cb[1])].append(float(cb[3]))
    if acc:
        ds.edges = {e: float(np.mean(v)) for e, v in acc.items()}
        ds.edge_name = "coupler break rate"
    ds.units = units
    r0 = rs[0]
    whens = sorted({r.get("when", "") for r in rs})
    l2 = r0.get("layer2") if isinstance(r0.get("layer2"), dict) else {}
    ds.info += [("records", f"{len(rs)} units, {len(whens)} run{'s' if len(whens) != 1 else ''} (newest {whens[-1]})"),
                ("setup", _setup_text(r0)), ("qrack", r0.get("qrack", "?")),
                ("shots × realisations", f"{r0.get('shots')} × {r0.get('repeats', 1)}"),
                ("Layer 2", mlab + (", " + ", ".join(f"{a} {l2[a]}" for a in ("blocks", "j_zz", "sigma_qs", "t_idle",
                                                                                 "inside_zz") if a in l2)
                                    if mode != "none" else "")),
                ("estimator", f"{r0.get('estimator', '?')} ({key})"), ("device", r0.get("device", "?"))]
    if r0.get("stack_order"):
        ds.info.append(("stack", f"order {r0['stack_order']}, pass {r0.get('stack_pass', '?')}"))
    pl = defaultdict(list)
    for u in units:
        pl[u["place"]].append(u["F"])
    for p_, v in sorted(pl.items(), key=lambda kv: -len(kv[1])):
        ds.info.append((p_, f"{np.mean(v):.3f} ({len(v)} units)"))
    for b in bridges:
        if b.get("register") == reg and b.get("when") in whens:
            for ps in b.get("passes") or []:
                ds.info.append((f"bridge {ps.get('label', '')}",
                                f"peak bonds {ps.get('peak_bonds')}/{ps.get('max_bonds')}, truncating "
                                f"{ps.get('truncating')}/{ps.get('circuits')} circuits, fidelity estimate "
                                f"{ps.get('fidelity_mean')} (min {ps.get('fidelity_min')}), widest "
                                f"{ps.get('max_width')} qubits"))
    return ds


def ghz_chip_dataset(key, recs):
    _, tree, mode, layers, setup = key
    r = recs[-1]
    kF, F = G.ghz_F(r)
    xF = G.ghz_F(r, "exact_")[1]
    ds = Dataset(f"chip-wide GHZ, {tree} tree, {G.GHZ_MODE_LABEL.get(mode, mode)}" + (f" · {setup}" if setup else ""),
                 short=f"chip {tree} · {G.GHZ_MODE_SHORT.get(mode, mode)}")
    ds.measures["agreement with the majority"] = Measure(
        "agreement with the majority", {int(q): 1.0 - float(v) for q, v in (r.get("site_break") or {}).items()},
        0.5, 1.0, note="share of Z shots where the site agrees with the state's majority")
    ds.edges = {edge_key(e[0], e[1]): float(e[3]) for e in r.get("edge_break") or []}
    ds.edge_name = "tree CX break rate"
    cx = r.get("cx_classes") or {}
    ds.info += [("runs", f"{len(recs)} (newest {r.get('when')})"), ("setup", _setup_text(r)),
                ("qrack", r.get("qrack", "?")), ("tree", f"{tree}, root {r.get('root')}, depth {r.get('depth')}"),
                (kF or "F", f"{F:.3f}" + (f" (exact {xF:.3f})" if xF is not None else "")),
                ("population", f"{r.get('pop')}"), ("CX classes", ", ".join(f"{k_} {v}" for k_, v in cx.items())),
                ("shots × realisations", f"{r.get('shots')} × {r.get('repeats', 1)}")]
    return ds


# ================================================================== shor9 registers (nighthawk_qrack.py)
def shor9_scenes(store):
    out = []
    regs = sorted({(r["register"], int(r.get("size", 3))) for r in store.shor
                   if r.get("register") and r.get("unit") is not None})
    for reg, m in regs:
        try:
            pay = G.shor3d_payload(store, reg, m)
        except Exception as err:
            print(f"# shor9 {reg}: {err}")
            pay = None
        if not pay:
            continue
        sc = Scene("shor9", f"{reg} · Shor-{m * m}")
        for st in pay["sites"]:
            sc.add_site(st[0], (int(st[2]), -int(st[1]), 0.0), st[3] if isinstance(st[3], str) else "lattice")
        sc.couplers = [(int(a), int(b), c) for a, b, c, *_ in pay["couplers"] if int(a) in sc.sites and int(b) in sc.sites]
        sc.finish()
        sc.notes.append(f"ACE lrc {pay['ace'].get('lrc')}, lrr {pay['ace'].get('lrr')}")
        rc = {q: (-int(s["p"][1]), int(s["p"][0])) for q, s in sc.sites.items()}
        for su in pay["setups"]:
            ds = Dataset(su["label"] or "PyQrack default", short=su["label"] or "default")
            units = []
            for u in su["units"]:
                qs = [q for q, (r_, c_) in rc.items() if u["rows"][0] <= r_ <= u["rows"][1]
                      and u["cols"][0] <= c_ <= u["cols"][1]]
                units.append(dict(qubits=qs, edges=[], z=u["z"], x=u["x"], w=min(u["z"], u["x"]), imp=u["imp"],
                                  text=f"{u['id']} rows {u['rows'][0]}-{u['rows'][1]} cols {u['cols'][0]}-{u['cols'][1]}: "
                                       f"Z {u['z']:.3f}, X {u['x']:.3f} · {u['kind']} · CX exact/replica/cross "
                                       f"{'/'.join(str(c) for c in u['cx'])}"))
            for f_, name, lo in (("z", "logical Z", 0.5), ("x", "logical X", 0.5), ("w", "weaker of Z and X", 0.5),
                                 ("imp", "Z-basis impossible share", 0.0)):
                per = mean_by_site(units, f_)
                if per:
                    ds.measures[name] = Measure(name, {q: np.mean(v) for q, v in per.items()}, lo,
                                                1.0 if lo else None)
            ds.units = units
            ds.info += [("units", str(len(units))), ("backends", str(su.get("backends"))),
                        ("shots", str(su.get("shots")))]
            sc.datasets.append(ds)
        out.append(sc)
    return out


# ================================================================== Shor's code across QrackBridge (nighthawk_ghz.py shor)
def shor_bridge_scenes(store, files):
    if not store.shor_bridge:
        return []
    maps = {}
    for r in G.read_tests(files, "shor_map", lambda r: (r.get("when"), r.get("register"))):
        if r.get("register") not in maps or r.get("when", "") > maps[r["register"]].get("when", ""):
            maps[r["register"]] = r
    out = []
    for reg in sorted({r["register"] for r in store.shor_bridge}):
        mp = maps.get(reg) or G.ghz_register_map(store, reg)
        if not mp:
            print(f"# shor across the bridge on {reg}: no shor_map record, skipped")
            continue
        sc = map_geometry(Scene("Shor across QrackBridge", reg), mp)
        groups = defaultdict(list)
        for r in store.shor_bridge:
            if r["register"] == reg:
                br = (r.get("loom") or {}).get("bridge") or {}
                groups[(int(r.get("size", 3)), br.get("max_bonds"), br.get("truncation"), r.get("qrack"))].append(r)
        for (m, mb, tr, qv), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], -(kv[0][1] or 0))):
            sc.datasets.append(shor_bridge_dataset(m, mb, tr, qv, rs))
        out.append(sc)
    return out


def shor_bridge_dataset(m, mb, tr, qv, rs):
    ds = Dataset(f"Shor-{m * m}, QrackBridge max_bonds {mb}" + (f", {tr}" if tr else ""),
                 short=f"Shor-{m * m} · max_bonds {mb}")
    units = []
    for r in rs:
        st = r.get("states") or {}
        raw = [st[s]["raw"] for s in ("0", "1") if s in st and "raw" in st[s]]
        rawx = [st[s]["raw"] for s in ("+", "-") if s in st and "raw" in st[s]]
        imp = [st[s]["impossible"] for s in st if isinstance(st[s], dict) and "impossible" in st[s]]
        cls = G.shor_bridge_class(r, True)
        units.append(dict(qubits=[int(q) for q in r.get("qubits", [])],
                          edges=[tuple(e) for e in (r.get("inner") or []) + (r.get("outer") or [])],
                          z=r.get("logical_Z"), x=r.get("logical_X"),
                          w=min(r.get("logical_Z", 1), r.get("logical_X", 1)),
                          rz=np.mean(raw) if raw else None, rx=np.mean(rawx) if rawx else None,
                          bz=r.get("bonds_Z"), bx=r.get("bonds_X"), pk=r.get("peak_bonds"),
                          imp=np.mean(imp) if imp else None, cls=cls,
                          text=f"unit {r.get('unit')} · {cls}: Z {r.get('logical_Z', float('nan')):.3f}, "
                               f"X {r.get('logical_X', float('nan')):.3f} · bonds Z/X {r.get('bonds_Z')}/{r.get('bonds_X')}"
                               f" · peak {r.get('peak_bonds')}"))
    spec = (("z", "decoded logical Z", 0.5, 1.0), ("x", "decoded logical X", 0.5, 1.0),
            ("w", "weaker of Z and X", 0.5, 1.0), ("rz", "raw logical Z", 0.5, 1.0), ("rx", "raw logical X", 0.5, 1.0),
            ("bz", "bonds kept, Z", 0, None), ("bx", "bonds kept, X", 0, None), ("pk", "peak bonds", 0, None),
            ("imp", "impossible share", 0.0, None))
    for f_, name, lo, hi in spec:
        per = mean_by_site(units, f_)
        if per:
            ds.measures[name] = Measure(name, {q: np.mean(v) for q, v in per.items()}, lo, hi,
                                        fmt="{:.0f}" if f_ in ("bz", "bx", "pk") else "{:.3f}")
    ds.units = units
    r0 = rs[0]
    br = (r0.get("loom") or {}).get("bridge") or {}
    ds.info += [("records", f"{len(rs)} units (run {r0.get('when')})"), ("qrack", qv or "?"),
                ("bridge", f"max_bonds {mb}, truncation {tr}, lazy bonds {br.get('lazy_bonds', 0)}"),
                ("shots", f"{r0.get('shots')} × {r0.get('repeats', 1)}, exact {r0.get('exact_shots')}")]
    by = defaultdict(list)
    for u in units:
        by[u["cls"]].append(u)
    for c, us in sorted(by.items()):
        z, x = np.mean([u["z"] for u in us]), np.mean([u["x"] for u in us])
        pred = (f", majority predicts {G.shor_majority(m, min(m, mb or 0)):.3f}"
                if c.startswith("cut through blocks") and mb is not None else "")
        ds.info.append((c, f"Z {z:.3f}, X {x:.3f} ({len(us)} units){pred}"))
    return ds


# ================================================================== dual Nighthawk R1 (nighthawk_ghz.py dualmap / dual)
def dual_scenes(store):
    if not store.dual_layout:
        return []
    lay = max(store.dual_layout, key=lambda r: r.get("when", ""))
    sc = Scene("dual Nighthawk R1", "two modules, four cables")
    pos = {int(st[0]): (int(st[1]), int(st[2])) for st in lay["sites"]}

    def group(q):
        r_ = pos[q][0]
        return "module A" if r_ <= 11 else "L-qubits A" if r_ <= 12 else "L-qubits B" if r_ <= 14 else "module B"
    for q, (r_, c_) in pos.items():
        sc.add_site(q, (c_, -r_, 0.0), group(q))
    for g in sc.groups:
        sc.group_dir[g] = np.array([0.0, 1.0 if g.endswith("A") else -1.0, 0.0])
    lattice = {q: rc for q, rc in pos.items() if q < 240}
    by_rc = {rc: q for q, rc in lattice.items()}
    for q, (r_, c_) in lattice.items():
        for dr, dc in ((0, 1), (1, 0)):
            p = by_rc.get((r_ + dr, c_ + dc))
            if p is not None and group(p) == group(q):
                sc.couplers.append((q, p, "exact"))
    links = lay.get("links") or []
    for L in links:
        sc.couplers += [(int(L["a"]), int(L["la"]), "L"), (int(L["lb"]), int(L["b"]), "L")]
        sc.links.append((int(L["la"]), int(L["lb"]), f"{100 * L['dcz']:.1f}", True))
    sc.height, sc.plates = 2.4, True
    sc.view = (38, -58)
    cables = ", ".join(f"{L['name']} dCZ {L['dcz']:.4f} (transfer {L.get('transfer')}, {L.get('tg')} ns)"
                       for L in links)
    sc.notes.append("cables: " + cables)
    # the GHZ chains of dualmap, one data set per model
    recs = [r for r in store.dual_map if r.get("when") == lay.get("when")]
    for mo in lay.get("models") or (list(recs[0]["F"]) if recs else []):
        ds = Dataset(f"3-qubit chains, {mo} model (run {lay.get('when')})", short=f"chains · {mo}")
        units = [dict(qubits=[int(q) for q in r["qubits"]], edges=[tuple(e) for e in r.get("edges", [])],
                      F=r["F"].get(mo), pk=(r.get("peak_bonds") or {}).get(mo), link=r.get("link"),
                      text=f"{r.get('tiling', '')} unit {r.get('unit')}: F {r['F'].get(mo):.3f}"
                           + (f" · through cable {r['link']}" if r.get("link") else ""))
                 for r in recs]
        per = mean_by_site(units, "F")
        ds.measures["lowest F_GHZ through the site"] = Measure("lowest F_GHZ through the site",
                                                               {q: min(v) for q, v in per.items()}, 0.5, 1.0)
        ds.measures["mean F_GHZ through the site"] = Measure("mean F_GHZ through the site",
                                                             {q: np.mean(v) for q, v in per.items()}, 0.5, 1.0)
        pk = mean_by_site(units, "pk")
        if pk:
            ds.measures["peak bonds"] = Measure("peak bonds", {q: max(v) for q, v in pk.items()}, 0, None, fmt="{:.0f}")
        ds.units = units
        lat = [u["F"] for u in units if not u["link"]]
        r0 = recs[0] if recs else {}
        ds.info += [("chains", f"{len(units)} ({len(lat)} inside a module)"),
                    ("inside a module", f"{np.mean(lat):.3f}" if lat else "-"),
                    ("trajectories", str(r0.get("trajectories"))), ("local CZ error", str(r0.get("local_cz_error"))),
                    ("qrack", r0.get("qrack", "?"))]
        for L in links:
            v = [u["F"] for u in units if u["link"] == L["name"]]
            if v:
                ds.info.append((f"through {L['name']}", f"{np.mean(v):.3f} ({len(v)} chains), dCZ {L['dcz']:.4f}"))
        sc.datasets.append(ds)
    # the paper's states (dual), newest run: one data set per state, one measure per model
    if store.dual:
        when = max(r["when"] for r in store.dual)
        for r in [r for r in store.dual if r["when"] == when]:
            qs = [int(q) for q in r.get("qubits", [])]
            if not all(q in sc.sites for q in qs):
                continue
            ds = Dataset(f"{r['state']} (run {when})", short=r["state"])
            for mo in G.DUAL_MODELS:
                if isinstance(r.get(mo), dict) and r[mo].get("F") is not None:
                    ds.measures[f"{mo} model"] = Measure(f"{mo} model", {q: r[mo]["F"] for q in qs}, 0.5, 1.0)
            ib = r.get("ibm")
            if ib:
                ds.measures["IBM, readout-mitigated"] = Measure("IBM, readout-mitigated",
                                                                {q: ib["mitigated"] for q in qs}, 0.5, 1.0)
                ds.measures["IBM, raw"] = Measure("IBM, raw", {q: ib["unmitigated"] for q in qs}, 0.5, 1.0)
            ds.mark = set(qs)
            ds.units = [dict(qubits=qs, edges=[(o[1], o[2]) for o in r.get("ops", []) if len(o) == 3],
                             text=f"{r['state']}: " + ", ".join(f"{mo} {r[mo]['F']:.3f}" for mo in G.DUAL_MODELS
                                                                 if isinstance(r.get(mo), dict)))]
            ds.info += [("circuit", " · ".join(" ".join(str(x) for x in o) for o in r.get("ops", []))),
                        ("trajectories", str(r.get("trajectories"))),
                        ("local CZ error", str(r.get("local_cz_error"))),
                        ("lattice 1q error", str(r.get("lattice_1q_error"))), ("max_bonds", str(r.get("max_bonds")))]
            for mo in G.DUAL_MODELS:
                if isinstance(r.get(mo), dict):
                    ds.info.append((mo, f"F {r[mo]['F']:.4f} ± {r[mo].get('se', 0):.4f}, peak bonds "
                                        f"{r[mo].get('peak_bonds')}"))
            if ib:
                ds.info.append(("IBM", f"mitigated {ib['mitigated']:.4f} ± {ib['mitigated_se']}, raw "
                                       f"{ib['unmitigated']:.4f} ± {ib['unmitigated_se']}"))
            sc.datasets.append(ds)
    return [sc] if sc.datasets else []


# ================================================================== runs (nighthawk_qrack.py run)
SKIP_FIELDS = {"n", "K", "depth", "partition", "instance", "ace_register", "ace_patches", "ace_lrc", "ace_lrr"}


def run_scenes(store):
    if not store.data:
        return []
    fams = defaultdict(lambda: defaultdict(list))
    for cfg, recs in store.data.items():
        for (n, fam, K, d, j, i), r in recs.items():
            fams[fam][(cfg, K)].append(r)
    out = []
    for fam in sorted(fams):
        sc = Scene("runs", f"{fam} landscape", kind="landscape")
        allr = [r for v in fams[fam].values() for r in v]
        sc.xs = sorted({int(r["depth"]) for r in allr})
        sc.ys = sorted({int(r["n"]) for r in allr})
        sc.xname, sc.yname = "depth", "register size n"
        sc.view = (28, -128)
        for (cfg, K), rs in sorted(fams[fam].items()):
            ds = Dataset(f"{G.lab(cfg)}" + (f", K = {K}" if K else ""), short=f"{cfg}" + (f" K{K}" if K else ""))
            cells = defaultdict(list)
            for r in rs:
                cells[(sc.xs.index(int(r["depth"])), sc.ys.index(int(r["n"])))].append(r)
            fields = sorted({k for r in rs for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                             and k not in SKIP_FIELDS})
            first = [f_ for f_ in ("fidelity", "survival", "se", "seconds") if f_ in fields]
            for f_ in first + [f_ for f_ in fields if f_ not in first]:
                vals = {c: np.mean([r[f_] for r in v if isinstance(r.get(f_), (int, float))]) for c, v in cells.items()
                        if any(isinstance(r.get(f_), (int, float)) for r in v)}
                if vals:
                    ds.measures[f_] = Measure(f_, vals, 0.0 if f_ in ("fidelity", "survival") else None,
                                              None, kind="seq")
                if f_ == "seconds" and vals:
                    ds.measures["log10 seconds"] = Measure("log10 seconds", {c: math.log10(max(v, 1e-3))
                                                                             for c, v in vals.items()})
            for c, v in cells.items():
                fid = [r.get("fidelity") for r in v if r.get("fidelity") is not None]
                ds.cells[c] = (f"n {sc.ys[c[1]]}, depth {sc.xs[c[0]]}: {len(v)} points; F "
                               + (", ".join(f"{x:.4f}" for x in fid[:6]) + ("…" if len(fid) > 6 else "") if fid else "-")
                               + f"; {sum(r.get('seconds', 0) or 0 for r in v):.1f} s")
            r0 = rs[0]
            ace = {k: r0[k] for k in ("ace_lrc", "ace_lrr", "ace_widths", "ace_couplers", "ace_b2b") if k in r0}
            ds.info += [("configuration", cfg), ("points", str(len(rs))),
                        ("seconds", f"{sum(r.get('seconds', 0) or 0 for r in rs):.1f} in all, "
                                    f"{np.mean([r.get('seconds', 0) or 0 for r in rs]):.2f} per point"),
                        ("device", str(r0.get("device", "?"))), ("theta", str(r0.get("theta", "?"))),
                        ("shots", str(r0.get("shots", "?")))]
            if ace:
                ds.info.append(("ACE", ", ".join(f"{k[4:]} {v}" for k, v in ace.items())))
            sc.datasets.append(ds)
        out.append(sc)
    bits = run_bits_scene(store)
    if bits:
        out.append(bits)
    return out


def run_bits_scene(store):
    """Per-qubit bit statistics of the runs' bitstrings on the lattice; one lazily computed data set per
    (configuration, family, n, depth) whose shots file exists."""
    lay = store.lay
    if lay is None:
        return None
    sc = Scene("runs", "bits on the lattice")
    for q in range(lay.n):
        r_, c_ = lay.pos[q]
        sc.add_site(q, (c_, -r_, 0.0), "lattice")
    for pairs in lay.matchings.values():
        for a, b in pairs:
            sc.couplers.append((int(a), int(b), "exact"))
    sc.finish()
    sc.view = (52, -72)
    keys = []
    for cfg, recs in store.data.items():
        base = store.bases.get(cfg)
        if base is None:
            continue
        for k in recs:
            if k[1] in ("hamming", "vqe"):
                continue
            if G.point_npz(base, cfg, k).exists():
                keys.append((cfg, k[1], k[2], k[0], k[3]))
    for cfg, fam, K, n, d in sorted(set(keys)):
        ds = Dataset(f"{G.lab(cfg)} · {fam}" + (f" K{K}" if K else "") + f" · n {n} · depth {d}",
                     short=f"{cfg} · {fam} · n{n} d{d}")
        ds.lazy = (lambda ds_, cfg=cfg, fam=fam, K=K, n=n, d=d: _fill_bits(ds_, store, cfg, fam, K, n, d))
        sc.datasets.append(ds)
    return sc if sc.datasets else None


def _fill_bits(ds, store, cfg, fam, K, n, d):
    lay, base = store.lay, store.bases[cfg]
    ks = [k for k in store.data[cfg] if k[0] == n and k[1] == fam and k[2] == K and k[3] == d]
    err, tot, ones, cnt, hits = np.zeros(n), 0, np.zeros(n), 0, 0
    targets = [sum(int(c) << q for q, c in enumerate(s[:n])) for s in lay.inputs]
    for k in ks:
        with np.load(G.point_npz(base, cfg, k)) as z:
            if fam == "mirror":
                for s, t in enumerate(targets):
                    if f"input{s}" not in z.files:
                        continue
                    x = z[f"input{s}"].astype(np.uint64)
                    if x.size == 0:
                        continue
                    diff = x ^ np.uint64(t)
                    err += G.bit_matrix(diff, n).sum(axis=0)
                    tot += x.size
                    hits += int((diff == 0).sum())
            for f_ in z.files:
                if f_.startswith("input") or f_.endswith("_twirl"):
                    continue
                x = z[f_].astype(np.uint64).ravel()
                if x.size:
                    ones += G.bit_matrix(x, n).sum(axis=0)
                    cnt += x.size
    if tot:
        ds.measures["bit-flip rate"] = Measure("bit-flip rate", {q: err[q] / tot for q in range(n)}, 0.0, None,
                                               note="share of shots where the bit differs from the prepared string")
        ds.info.append(("survival", f"{hits / tot:.4f} of {tot} shots"))
    if cnt:
        p1 = ones / cnt
        ds.measures["P(1)"] = Measure("P(1)", {q: p1[q] for q in range(n)}, 0.0, 1.0)
        ds.measures["|P(1) − ½|"] = Measure("|P(1) − ½|", {q: abs(p1[q] - 0.5) for q in range(n)}, 0.0, None)
        ds.info.append(("samples", str(cnt)))
    ds.info += [("points", str(len(ks))), ("register size", str(n)), ("depth", str(d))]


# ================================================================== all scenes
def build_scenes(store, files):
    sc = []
    for f in (lambda: ghz_scenes(store, files), lambda: shor_bridge_scenes(store, files), lambda: dual_scenes(store),
              lambda: shor9_scenes(store), lambda: run_scenes(store)):
        try:
            sc += f()
        except Exception as err:            # one broken family must not take the others down
            import traceback
            traceback.print_exc()
            print(f"# skipped a family: {type(err).__name__}: {err}")
    return [s for s in sc if s.datasets]


# ================================================================== geometry for drawing
def frame(u):
    u = np.asarray(u, float)
    u = u / (np.linalg.norm(u) or 1.0)
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    v = np.cross(u, a)
    v /= np.linalg.norm(v)
    return u, v, np.cross(u, v)


def cuboid_faces(p, up, h, half):
    """The five visible faces of a column standing on p along up, height h (negative: the other way)."""
    if h < 0:
        up, h = -np.asarray(up, float), -h
    u, v, w = frame(up)
    b = [p + half * (s1 * v + s2 * w) for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    t = [x + h * u for x in b]
    faces, normals = [t], [u]
    for i in range(4):
        j = (i + 1) % 4
        faces.append([b[i], b[j], t[j], t[i]])
        m = (b[i] + b[j]) / 2 - p
        normals.append(m / (np.linalg.norm(m) or 1.0))
    return faces, normals


def shade(rgb, n, amount):
    L = np.array([-0.35, -0.55, 0.76])
    L /= np.linalg.norm(L)
    f = (1 - amount) + amount * max(0.0, float(np.dot(n, L)))
    return tuple(min(1.0, c * f) for c in rgb[:3]) + (1.0,)


def bezier(a, b, lift, k=24):
    c = (a + b) / 2 + np.array([0, 0, lift])
    t = np.linspace(0, 1, k)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b


# ================================================================== the viewer
class Viewer:
    def __init__(self, fig, store, files, a):
        self.fig, self.store, self.files, self.a = fig, store, files, a
        self.theme = THEMES[a.theme]
        self.scenes = build_scenes(store, files)
        self.si, self.di, self.mi = 0, 0, 0
        self.columns, self.values, self.names, self.edge_tel = True, None, True, False
        self.pull, self.hscale, self.zoom = 0.0, 1.0, 1.0
        self.couplers_on, self.links_on = True, True
        self.view = None
        self.pick, self._cand, self._press = None, None, None
        self.status = store.summary()
        self.widgets = {}
        self.select(a.scene, a.dataset, a.measure)
        self.build()
        fig.canvas.mpl_connect("key_press_event", self.on_key)
        fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        fig.canvas.mpl_connect("pick_event", self.on_pick)
        fig.canvas.mpl_connect("button_press_event", self.on_press)
        fig.canvas.mpl_connect("button_release_event", self.on_release)
        if a.follow:
            self.timer = fig.canvas.new_timer(interval=int(a.follow * 1000))
            self.timer.add_callback(self.reload)
            self.timer.start()

    # ---------------------------------------------------------- selection
    @property
    def scene(self):
        return self.scenes[self.si] if self.scenes else None

    @property
    def dataset(self):
        sc = self.scene
        return sc.datasets[self.di].ready() if sc and sc.datasets else None

    @property
    def measure(self):
        ds = self.dataset
        if not ds or not ds.measures:
            return None
        names = list(ds.measures)
        return ds.measures[names[self.mi % len(names)]]

    def select(self, scene=None, dataset=None, measure=None):
        def find(items, want, name):
            if want is None:
                return 0
            if str(want).isdigit() and int(want) < len(items):
                return int(want)
            for i, it in enumerate(items):
                if want.lower() in name(it).lower():
                    return i
            print(f"# no match for {want!r}; first one shown")
            return 0
        if not self.scenes:
            return
        self.si = find(self.scenes, scene, lambda s: s.title)
        self.di = find(self.scene.datasets, dataset, lambda d: d.label)
        ds = self.dataset
        self.mi = find(list(ds.measures), measure, lambda m: m) if ds and ds.measures else 0

    def step(self, what, k):
        if not self.scenes:
            return
        if what == "scene":
            self.si = (self.si + k) % len(self.scenes)
            self.di, self.mi, self.pick, self.view = 0, 0, None, None
            self.values = None
        elif what == "dataset":
            n = len(self.scene.datasets)
            name = self.measure.name if self.measure else None
            self.di = (self.di + k) % n
            ms = list(self.dataset.measures)
            self.mi = ms.index(name) if name in ms else 0
            self.pick = None
        elif what == "measure":
            n = max(1, len(self.dataset.measures))
            self.mi = (self.mi + k) % n
        self.draw()

    # ---------------------------------------------------------- layout
    def build(self):
        """(Re)create every axes and widget in the current theme; draw()."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons, Slider
        T, fig = self.theme, self.fig
        keep = None
        if getattr(self, "ax", None) is not None:
            keep = (self.ax.elev, self.ax.azim)
        fig.clf()
        fig.set_facecolor(T["surface"])
        self.ax = fig.add_axes([0.205, 0.02, 0.555, 0.93], projection="3d")
        if keep:
            self.view = keep
        self.cax = fig.add_axes([0.785, 0.905, 0.2, 0.014])
        self.iax = fig.add_axes([0.775, 0.02, 0.22, 0.86])
        self.iax.set_axis_off()
        self.lax = fig.add_axes([0.0, 0.0, 0.195, 1.0])
        self.lax.set_axis_off()
        self.lax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=self.lax.transAxes, color=T["panel"], zorder=-1))
        W = self.widgets = {}
        fig.text(0.012, 0.968, "nighthawk 3D", color=T["text"], fontsize=13, fontweight="bold")
        fig.text(0.012, 0.950, "telemetry and measurements", color=T["muted"], fontsize=8)

        def button(key, rect, label, cb, fs=9):
            bax = fig.add_axes(rect)
            b = Button(bax, label, color=T["control"], hovercolor=T["hover"])
            b.label.set_color(T["text"])
            b.label.set_fontsize(fs)
            for s in bax.spines.values():
                s.set_edgecolor(T["faint"])
            b.on_clicked(cb)
            W[key] = b
            return b
        y = 0.895
        self.sel_text = {}
        for what, title in (("scene", "Shape"), ("dataset", "Data set"), ("measure", "Measure")):
            fig.text(0.012, y + 0.024, title, color=T["muted"], fontsize=8)
            button(f"{what}-", [0.012, y - 0.012, 0.022, 0.032], "◀", lambda e, w=what: self.step(w, -1), 8)
            button(f"{what}+", [0.168, y - 0.012, 0.022, 0.032], "▶", lambda e, w=what: self.step(w, 1), 8)
            self.sel_text[what] = fig.text(0.101, y + 0.004, "", color=T["text"], fontsize=8.2, ha="center",
                                           va="center")
            y -= 0.072
        # toggles
        cax = fig.add_axes([0.012, y - 0.15, 0.178, 0.17])
        cax.set_facecolor(T["panel"])
        for s in cax.spines.values():
            s.set_visible(False)
        labels = ["columns (else dots)", "value labels", "group names", "couplers", "links and cables",
                  "coupler telemetry"]
        acts = [self.columns, bool(self.values), self.names, self.couplers_on, self.links_on, self.edge_tel]
        cb = CheckButtons(cax, labels, acts, label_props=dict(color=[T["text"]] * 6, fontsize=[8.5] * 6),
                          frame_props=dict(edgecolor=T["muted"], facecolor=T["control"]),
                          check_props=dict(facecolor=T["accent"]))
        cb.on_clicked(self.on_check)
        W["checks"] = cb
        y -= 0.2
        for key, label, lo, hi, val in (("pull", "pull apart", 0.0, 4.0, self.pull),
                                        ("height", "height", 0.2, 3.0, self.hscale)):
            sax = fig.add_axes([0.075, y, 0.085, 0.022])
            sax.set_facecolor(T["control"])
            s = Slider(sax, label, lo, hi, valinit=val, color=T["accent"], track_color=T["control"],
                       handle_style=dict(facecolor=T["text"], edgecolor=T["muted"]))
            s.label.set_color(T["text2"])
            s.label.set_fontsize(8.5)
            s.valtext.set_color(T["text2"])
            s.valtext.set_fontsize(8)
            s.on_changed(lambda v, k=key: self.on_slider(k, v))
            W[key] = s
            y -= 0.04
        y -= 0.012
        for i, (key, label, cb_) in enumerate((("reset", "reset view", lambda e: self.reset_view()),
                                               ("top", "top", lambda e: self.set_view(90, -90)),
                                               ("side", "side", lambda e: self.set_view(4, -90)),
                                               ("theme", "light" if T["name"] == "dark" else "dark",
                                                lambda e: self.toggle_theme()),
                                               ("reload", "reload", lambda e: self.reload()),
                                               ("save", "save PNG", lambda e: self.save()))):
            col, row = i % 3, i // 3
            button(key, [0.012 + col * 0.06, y - row * 0.042, 0.056, 0.034], label, cb_, 8.2)
        y -= 0.1
        help_ = ("drag: rotate · scroll: zoom · click: pick a site\n↑ ↓ shape  ← → data set  m measure\n"
                 "c columns  v values  n names  e pull apart\nk coupler telemetry  t theme  r reload\n"
                 "p save  0 reset  1 top  2 side")
        fig.text(0.012, y - 0.02, help_, color=T["muted"], fontsize=7.4, va="top", linespacing=1.5)
        self.status_text = fig.text(0.012, 0.012, "", color=T["muted"], fontsize=6.6, va="bottom", wrap=True)
        try:
            fig.canvas.manager.set_window_title("nighthawk 3D")
        except Exception:
            pass
        self.draw()

    # ---------------------------------------------------------- drawing
    def colormap(self, m):
        from matplotlib.colors import LinearSegmentedColormap
        T = self.theme
        return LinearSegmentedColormap.from_list("nh", T["div"] if m.kind == "div" else T["seq"])

    def positions(self, sc):
        """Base position and outward direction of every site with the current pull-apart."""
        out = {}
        for q, s in sc.sites.items():
            out[q] = (s["p"] + self.pull * sc.group_dir.get(s["group"], np.zeros(3)), s["up"])
        return out

    def draw(self):
        T, ax = self.theme, self.ax
        if self.view is None and self.scene is not None:
            self.view = self.scene.view
        elev, azim = (self.view if self.view else (30, -60))
        ax.cla()
        ax.set_facecolor(T["surface"])
        sc, ds, m = self.scene, self.dataset, self.measure
        for what, txt in (("scene", f"{self.si + 1}/{len(self.scenes)}  {sc.title if sc else '-'}"),
                          ("dataset", f"{self.di + 1}/{len(sc.datasets)}  {ds.short}" if ds else "-"),
                          ("measure", f"{self.mi % max(1, len(ds.measures)) + 1}/{len(ds.measures)}  {m.name}"
                           if m else "-")):
            self.sel_text[what].set_text(txt if len(txt) <= 34 else txt[:33] + "…")
        if sc is None:
            ax.set_axis_off()
            ax.text2D(0.5, 0.5, "no records with a shape to draw\n(see --help)", transform=ax.transAxes,
                      ha="center", color=T["text2"])
            self.fig.canvas.draw_idle()
            return
        self._pickmap = []
        if sc.kind == "landscape":
            pts = self.draw_landscape(sc, ds, m)
        else:
            pts = self.draw_lattice(sc, ds, m)
        self.frame_axes(sc, pts)
        ax.view_init(elev=elev, azim=azim)
        self.draw_panel(sc, ds, m)
        self.status_text.set_text(textwrap.fill(self.status, 58))
        self.fig.canvas.draw_idle()

    def frame_axes(self, sc, pts):
        ax = self.ax
        P = np.array(pts) if len(pts) else np.zeros((1, 3))
        lo, hi = P.min(axis=0), P.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        if sc.kind == "landscape":
            ax.set_xlim(lo[0] - 0.6, hi[0] + 0.6)
            ax.set_ylim(lo[1] - 0.6, hi[1] + 0.6)
            ax.set_zlim(min(0, lo[2]), hi[2] * 1.05 if hi[2] > 0 else 1)
            ax.set_box_aspect((span[0] + 1.2, span[1] + 1.2, max(span[0], span[1], 3) * 0.55), zoom=self.zoom)
            return
        pad = 0.8
        lo, hi = lo - pad, hi + pad
        c, r = (lo + hi) / 2, (hi - lo) / 2
        r = np.maximum(r, 0.3 * r.max())
        ax.set_xlim(c[0] - r[0], c[0] + r[0])
        ax.set_ylim(c[1] - r[1], c[1] + r[1])
        ax.set_zlim(c[2] - r[2], c[2] + r[2])
        ax.set_box_aspect(tuple(2 * r), zoom=self.zoom)
        ax.set_axis_off()

    def draw_lattice(self, sc, ds, m):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
        from matplotlib.colors import to_rgba
        T, ax = self.theme, self.ax
        pos = self.positions(sc)
        pts = [p for p, _ in pos.values()]
        H = sc.height * self.hscale
        cmap = self.colormap(m) if m else None
        gcol = {g: (T["seam"] if g == "seam" else T["cat"][i % len(T["cat"])]) for i, g in
                enumerate([g for g in sc.groups if g != "seam"])}
        gcol["seam"] = T["seam"]
        vals = m.values if m else {}

        def top(q):
            p, u = pos[q]
            v = vals.get(q)
            h = max(0.04, m.frac(v) * H) if v is not None else 0.0
            if m is not None and m.kind == "div" and v is not None:
                h = max(0.04, abs(v) / m.hi * H * 0.6)
            return p + h * u
        # couplers
        if self.couplers_on and sc.couplers:
            segs, cols, lws = [], [], []
            etel = self.edge_tel and ds is not None and ds.edges
            emax = max(list(ds.edges.values()) + [1e-9]) if etel else 1.0
            from matplotlib.colors import LinearSegmentedColormap
            ecmap = LinearSegmentedColormap.from_list("e", [T["coupler"], T["cross"]])
            for a, b, cls in sc.couplers:
                segs.append([pos[a][0], pos[b][0]])
                if etel:
                    v = ds.edges.get(edge_key(a, b))
                    cols.append(ecmap(v / emax) if v is not None else to_rgba(T["faint"]))
                    lws.append(1.0 + 2.4 * (v / emax) if v is not None else 0.6)
                else:
                    cols.append(to_rgba({"replica": T["replica"], "cross": T["cross"], "L": T["text2"]}.get(cls, T["coupler"])))
                    lws.append(0.9 if cls == "exact" else 1.5)
            ax.add_collection3d(Line3DCollection(segs, colors=cols, linewidths=lws, zorder=1))
        # links: chips joined (loom / bridge / box edges) and the dual's cables
        if self.links_on:
            for a, b, label, arc in sc.links:
                pa, pb = pos[a][0], pos[b][0]
                if arc:
                    lift = 1.2 + 0.15 * np.linalg.norm(pa - pb)
                    C = bezier(pa, pb, lift)
                    ax.plot(C[:, 0], C[:, 1], C[:, 2], color=T["link"], lw=2.6, zorder=3)
                    pts.append(C[len(C) // 2])
                    if label:
                        ap = C[len(C) // 2]
                        ax.text(ap[0], ap[1], ap[2] + 0.25, label, color=T["text"], fontsize=8, ha="center",
                                zorder=TEXT_Z)
                else:
                    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color=T["link"], lw=2.2, zorder=3)
        # each group's footprint: chips, box faces, modules (outlined)
        if sc.plates:
            for g in sc.groups:
                gq = [q for q in sc.sites if sc.sites[q]["group"] == g]
                if len(gq) < 3:
                    continue
                Q = np.array([pos[q][0] for q in gq])
                u = sc.sites[gq[0]]["up"]
                lo_, hi_ = Q.min(axis=0) - 0.42, Q.max(axis=0) + 0.42
                k = int(np.argmax(np.abs(u)))
                lo_[k] = hi_[k] = Q[:, k].mean() - 0.03 * np.sign(u[k])
                a_, b_ = [i for i in range(3) if i != k]
                corners = []
                for ca, cb_ in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    c = lo_.copy()
                    c[a_] = (lo_ if ca == 0 else hi_)[a_]
                    c[b_] = (lo_ if cb_ == 0 else hi_)[b_]
                    corners.append(c)
                # outline only: a filled plate is depth-sorted as one polygon and would tint the columns
                ax.add_collection3d(Line3DCollection([[corners[i], corners[(i + 1) % 4]] for i in range(4)],
                                                     colors=[to_rgba(gcol.get(g, T["muted"]), 0.7)], linewidths=1.0))
        # base sites
        P = np.array([pos[q][0] for q in sc.sites])
        C = [to_rgba(gcol.get(sc.sites[q]["group"], T["muted"]), 0.9) for q in sc.sites]
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=9 if len(sc.sites) > 150 else 14, c=C, depthshade=False,
                   edgecolors="none", zorder=2)
        # the measure
        qs = [q for q in sc.sites if q in vals]
        tops = {q: top(q) for q in qs}
        if qs and self.columns:                     # one collection per column: depth-sorted against each other
            for q in qs:
                p, u = pos[q]
                h = float(np.dot(tops[q] - p, u))
                rgb = cmap(m.frac(vals[q]))
                fs, ns = cuboid_faces(p, u, h, sc.half)
                ax.add_collection3d(Poly3DCollection(fs, facecolors=[shade(rgb, n_, T["light"]) for n_ in ns],
                                                     edgecolors=(0, 0, 0, 0), linewidths=0))
        elif qs:
            Tp = np.array([tops[q] for q in qs])
            ax.scatter(Tp[:, 0], Tp[:, 1], Tp[:, 2], s=46, c=[cmap(m.frac(vals[q])) for q in qs], depthshade=False,
                       edgecolors=T["surface"], linewidths=0.8, zorder=5)
            for q in qs:
                p, _ = pos[q]
                ax.plot([p[0], tops[q][0]], [p[1], tops[q][1]], [p[2], tops[q][2]], color=T["faint"], lw=0.7)
        pts += list(tops.values())
        # marked qubits (a dual state) and the picked site's units
        if ds is not None and ds.mark:
            for q in ds.mark:
                p = tops.get(q, pos[q][0])
                ax.scatter([p[0]], [p[1]], [p[2] + 0.05], s=150, facecolors="none", edgecolors=T["text"],
                           linewidths=1.4, depthshade=False, zorder=7)
        if self.pick is not None and ds is not None:
            for u in ds.units:
                if self.pick in u["qubits"]:
                    for a_, b_ in (u.get("edges") or []):
                        if a_ in pos and b_ in pos:
                            pa, pb = tops.get(a_, pos[a_][0]), tops.get(b_, pos[b_][0])
                            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2] + 0.06, pb[2] + 0.06], color=T["text"],
                                    lw=2.0, zorder=8)
                    U = np.array([tops.get(q, pos[q][0]) for q in u["qubits"] if q in pos])
                    ax.scatter(U[:, 0], U[:, 1], U[:, 2] + 0.06, s=40, c=T["text"], depthshade=False, zorder=8)
            p = tops.get(self.pick, pos.get(self.pick, (np.zeros(3), None))[0])
            ax.scatter([p[0]], [p[1]], [p[2] + 0.08], s=200, facecolors="none", edgecolors=T["accent"],
                       linewidths=2.2, depthshade=False, zorder=9)
        # value labels and group names
        show_values = self.values if self.values is not None else len(qs) <= 64
        if show_values and m:
            for q in qs:
                t = tops[q] + 0.18 * pos[q][1]
                ax.text(t[0], t[1], t[2], m.fmt.format(vals[q]), color=T["text2"], fontsize=6.2, ha="center",
                        va="bottom", zorder=TEXT_Z)
        if self.names and len(sc.groups) > 1:
            for g in sc.groups:
                gq = [q for q in sc.sites if sc.sites[q]["group"] == g]
                c = np.mean([pos[q][0] for q in gq], axis=0)
                u = sc.sites[gq[0]]["up"]
                d = sc.group_dir.get(g, np.zeros(3))
                if np.allclose(u, (0, 0, 1)):     # flat: outside the group, away from the centre
                    ext = np.ptp(np.array([pos[q][0] for q in gq]), axis=0)
                    t = c + d * (0.5 * float(np.dot(np.abs(d), ext)) + 1.2) + np.array([0, 0, 0.2])
                else:
                    t = c + u * (H + 0.9)
                ax.text(t[0], t[1], t[2], g, color=gcol.get(g, T["text2"]), fontsize=8.5, ha="center",
                        fontweight="bold", zorder=TEXT_Z,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=T["surface"], edgecolor="none", alpha=0.75))
        # pick targets: the column tops
        if qs:
            Tp = np.array([tops[q] for q in qs])
            art = ax.scatter(Tp[:, 0], Tp[:, 1], Tp[:, 2], s=60, c=[(0, 0, 0, 0)] * len(qs), picker=True,
                             pickradius=6, depthshade=False, zorder=10)
            self._pickmap.append((art, qs))
        return pts

    def draw_landscape(self, sc, ds, m):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        T, ax = self.theme, self.ax
        pts = [(0, 0, 0), (len(sc.xs) - 1, len(sc.ys) - 1, 0)]
        if m is None:
            return pts
        cmap = self.colormap(m)
        keys, tops = [], []
        for (i, j), v in sorted(m.values.items()):
            p = np.array([i, j, 0.0])
            fs, ns = cuboid_faces(p, (0, 0, 1), v if abs(v) > 1e-12 else 1e-6, 0.33)
            rgb = cmap(m.frac(v))
            ax.add_collection3d(Poly3DCollection(fs, facecolors=[shade(rgb, n_, T["light"]) for n_ in ns],
                                                 edgecolors=(0, 0, 0, 0), linewidths=0))
            keys.append((i, j))
            tops.append((i, j, v))
        pts += tops
        show = self.values if self.values is not None else len(tops) <= 48
        if show:
            for i, j, v in tops:
                ax.text(i, j, v, " " + m.fmt.format(v), color=T["text2"], fontsize=7, ha="center", va="bottom",
                        zorder=TEXT_Z)
        if self.pick is not None and self.pick in m.values:
            i, j = self.pick
            ax.scatter([i], [j], [m.values[self.pick]], s=180, facecolors="none", edgecolors=T["accent"],
                       linewidths=2, depthshade=False)
        Tp = np.array(tops)
        art = ax.scatter(Tp[:, 0], Tp[:, 1], Tp[:, 2], s=80, c=[(0, 0, 0, 0)] * len(tops), picker=True,
                         pickradius=8, depthshade=False)
        self._pickmap.append((art, keys))
        ax.set_xticks(range(len(sc.xs)), [str(x) for x in sc.xs])
        ax.set_yticks(range(len(sc.ys)), [str(y) for y in sc.ys])
        ax.set_xlabel(sc.xname, color=T["text2"])
        ax.set_ylabel(sc.yname, color=T["text2"])
        ax.set_zlabel(m.name, color=T["text2"])
        ax.tick_params(colors=T["text2"], labelsize=8)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color(T["panel"])
            axis.pane.set_edgecolor(T["grid"])
            try:
                axis._axinfo["grid"].update(color=T["grid"], linewidth=0.6)
                axis.line.set_color(T["faint"])
            except Exception:
                pass
        return pts

    def draw_panel(self, sc, ds, m):
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.colors import Normalize
        T = self.theme
        self.cax.cla()
        self.iax.cla()
        self.iax.set_axis_off()
        if m is not None:
            cb = ColorbarBase(self.cax, cmap=self.colormap(m), norm=Normalize(m.lo, m.hi), orientation="horizontal")
            cb.outline.set_edgecolor(T["faint"])
            self.cax.tick_params(colors=T["text2"], labelsize=7.5, length=2)
            self.cax.set_title(m.name, color=T["text"], fontsize=9, loc="left", pad=3)
        else:
            self.cax.set_axis_off()
        lines = [(sc.title, T["text"], 10.5, "bold")]
        lines += [(t, T["text2"], 8, "normal") for t in textwrap.wrap(ds.label, 48)] if ds else []
        for note in sc.notes:
            lines += [(t, T["muted"], 7.4, "normal") for t in textwrap.wrap(note, 54)]
        if m is not None and m.values:
            v = np.array(list(m.values.values()))
            what = "cells" if sc.kind == "landscape" else "sites"
            lines.append((f"{m.name}: mean {m.fmt.format(v.mean())}, min {m.fmt.format(v.min())}, "
                          f"max {m.fmt.format(v.max())} over {len(v)} {what}", T["text"], 8, "normal"))
            if m.note:
                lines += [(t, T["muted"], 7.4, "normal") for t in textwrap.wrap(m.note, 54)]
        if ds is not None and ds.edges and sc.kind != "landscape":
            ev = np.array(list(ds.edges.values()))
            lines.append((f"{ds.edge_name}: mean {ev.mean():.3f}, max {ev.max():.3f} over {len(ev)} couplers"
                          + ("" if self.edge_tel else "  (k shows it)"), T["text2"], 7.6, "normal"))
        lines.append(("", T["text"], 4, "normal"))
        lines.append(("telemetry", T["muted"], 8, "bold"))
        for k, v in (ds.info if ds else []):
            wrapped = textwrap.wrap(f"{k}: {v}", 56, subsequent_indent="   ")
            lines += [(t, T["text2"], 7.4, "normal") for t in wrapped]
        if self.pick is not None and ds is not None:
            lines.append(("", T["text"], 4, "normal"))
            if sc.kind == "landscape":
                lines.append(("picked", T["muted"], 8, "bold"))
                lines += [(t, T["text"], 7.6, "normal") for t in textwrap.wrap(ds.cells.get(self.pick, ""), 54)]
            else:
                q = self.pick
                s = sc.sites.get(q, {})
                vs = [f"{mm.name} {mm.fmt.format(mm.values[q])}" for mm in ds.measures.values() if q in mm.values]
                lines.append((f"site {q} · {s.get('group', '')}", T["accent"], 8.4, "bold"))
                for t in textwrap.wrap(" · ".join(vs), 56):
                    lines.append((t, T["text"], 7.4, "normal"))
                us = [u for u in ds.units if q in u["qubits"]]
                for u in us[:10]:
                    lines += [(t, T["text2"], 7.2, "normal") for t in textwrap.wrap(u["text"], 58,
                                                                                    subsequent_indent="   ")]
                if len(us) > 10:
                    lines.append((f"… and {len(us) - 10} more units", T["muted"], 7.2, "normal"))
        width_in = self.iax.get_position().width * self.fig.get_figwidth()
        wrapped = []
        for text, col, fs, fw in lines:
            n = max(20, int(width_in * 72 / (fs * (0.60 if fw == "bold" else 0.53))))
            parts = textwrap.wrap(text, n, subsequent_indent="   ") if len(text) > n else [text]
            wrapped += [(t, col, fs, fw) for t in parts]
        y = 1.0
        for text, col, fs, fw in wrapped:
            if y < 0.0:
                self.iax.text(0, y, "…", color=T["muted"], fontsize=8, transform=self.iax.transAxes)
                break
            self.iax.text(0, y, text, color=col, fontsize=fs, fontweight=fw, va="top", transform=self.iax.transAxes,
                          family="DejaVu Sans")
            y -= fs * 0.0019 + 0.006

    # ---------------------------------------------------------- events
    def on_check(self, label):
        st = self.widgets["checks"].get_status()
        self.columns, vals, self.names, self.couplers_on, self.links_on, self.edge_tel = st
        self.values = vals
        self.draw()

    def on_slider(self, key, v):
        if key == "pull":
            self.pull = float(v)
        else:
            self.hscale = float(v)
        self.draw()

    def on_scroll(self, ev):
        if ev.inaxes is not self.ax:
            return
        self.zoom = min(6.0, max(0.3, self.zoom * (1.12 if ev.button == "up" else 1 / 1.12)))
        self.ax.set_box_aspect(self.ax.get_box_aspect(), zoom=self.zoom)
        self.fig.canvas.draw_idle()

    def on_pick(self, ev):
        for art, keys in self._pickmap:
            if ev.artist is art and len(ev.ind):
                self._cand = keys[int(ev.ind[0])]

    def on_press(self, ev):
        self._press = (ev.x, ev.y) if ev.inaxes is self.ax else None
        self.view = (self.ax.elev, self.ax.azim)

    def on_release(self, ev):
        if self._press is None or ev.inaxes is not self.ax:
            self._cand = None
            return
        moved = math.hypot(ev.x - self._press[0], ev.y - self._press[1])
        self.view = (self.ax.elev, self.ax.azim)
        if moved < 4:
            self.pick = self._cand
            self.draw()
        self._cand = None

    def on_key(self, ev):
        k = ev.key or ""
        acts = {"up": lambda: self.step("scene", -1), "down": lambda: self.step("scene", 1),
                "left": lambda: self.step("dataset", -1), "right": lambda: self.step("dataset", 1),
                "m": lambda: self.step("measure", 1), "M": lambda: self.step("measure", -1),
                "c": lambda: self.toggle(0), "v": lambda: self.toggle(1), "n": lambda: self.toggle(2),
                "k": lambda: self.toggle(5), "e": self.toggle_pull, "t": self.toggle_theme, "r": self.reload,
                "p": self.save, "home": self.reset_view, "0": self.reset_view, "1": lambda: self.set_view(90, -90),
                "2": lambda: self.set_view(4, -90), "escape": self.clear_pick}
        if k in acts:
            acts[k]()

    def toggle(self, i):
        cb = self.widgets["checks"]
        if i == 1 and self.values is None:          # 'auto' becomes the opposite of what auto showed
            m = self.measure
            auto = m is not None and len(m.values) <= (48 if self.scene.kind == "landscape" else 64)
            if cb.get_status()[1] == auto:
                cb.set_active(1)
                return
        cb.set_active(i)

    def toggle_pull(self):
        self.widgets["pull"].set_val(0.0 if self.pull > 0 else 2.0)

    def clear_pick(self):
        self.pick = None
        self.draw()

    def set_view(self, elev, azim):
        self.view = (elev, azim)
        self.draw()

    def reset_view(self):
        self.view, self.zoom = (self.scene.view if self.scene else None), 1.0
        self.draw()

    def toggle_theme(self):
        self.theme = THEMES["light" if self.theme["name"] == "dark" else "dark"]
        self.build()

    def reload(self):
        keep = (self.scene.title if self.scene else None, self.dataset.label if self.dataset else None,
                self.measure.name if self.measure else None)
        self.store.reload()
        self.scenes = build_scenes(self.store, self.files)
        self.status = self.store.summary()
        self.si = next((i for i, s in enumerate(self.scenes) if s.title == keep[0]), 0)
        if self.scene:
            self.di = next((i for i, d in enumerate(self.scene.datasets) if d.label == keep[1]), 0)
            ms = list(self.dataset.measures)
            self.mi = ms.index(keep[2]) if keep[2] in ms else 0
        self.draw()

    def save(self, folder=None, name=None):
        folder = Path(folder or self.a.save or "nighthawk_3d_png")
        folder.mkdir(parents=True, exist_ok=True)
        stem = name or re.sub(r"[^A-Za-z0-9._-]+", "_", f"{self.si + 1:02d}_{self.scene.title}_{self.di + 1:02d}"
                                                           f"_{self.measure.name if self.measure else ''}")[:150]
        p = folder / f"{stem}.png"
        self.fig.savefig(p, dpi=self.a.dpi, facecolor=self.theme["surface"])
        print("saved", p)
        return p


# ================================================================== main
def make_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", action="append", default=[], help="record file(s); default every *.jsonl here")
    ap.add_argument("--shor", action="append", default=[], help="nighthawk_qrack.py shor9 --out file(s)")
    ap.add_argument("--ghz", action="append", default=[],
                    help="nighthawk_ghz.py --out file(s): tiles, stack, chip, shor, dual, dualmap")
    ap.add_argument("--repo", default=None, help="rcs-nighthawk clone (the lattice of the run records)")
    ap.add_argument("--shots-dir", dest="shots_dir", default=None, help="as given to run --shots-dir")
    ap.add_argument("--cfg", action="append", default=None, help="only these run configuration tags")
    ap.add_argument("--theme", choices=["dark", "light", "auto"], default="dark",
                    help="colours (default dark); auto follows the desktop where it can tell, else dark")
    ap.add_argument("--scene", default=None, help="the first shape: number (0-based) or part of its name")
    ap.add_argument("--dataset", default=None, help="the first data set: number or part of its label")
    ap.add_argument("--measure", default=None, help="the first measure: number or part of its name")
    ap.add_argument("--list", action="store_true", help="print the shapes, data sets and measures found; exit")
    ap.add_argument("--save", default=None, metavar="DIR", help="write PNGs to DIR instead of opening the window")
    ap.add_argument("--all-datasets", action="store_true", help="--save: every data set, not only the first")
    ap.add_argument("--all-measures", action="store_true", help="--save: every measure, not only the first")
    ap.add_argument("--view", nargs=2, type=float, default=None, metavar=("ELEV", "AZIM"),
                    help="camera elevation and azimuth in degrees")
    ap.add_argument("--pull", type=float, default=0.0, help="pull the chips / faces / modules apart by this much")
    ap.add_argument("--dots", action="store_true", help="dots instead of columns")
    ap.add_argument("--dpi", type=int, default=130, help="PNG resolution (default 130)")
    ap.add_argument("--size", nargs=2, type=float, default=(15.5, 9.0), metavar=("W", "H"),
                    help="window / PNG size in inches (default 15.5 9)")
    ap.add_argument("--follow", type=float, default=None, metavar="SEC",
                    help="re-read the records every SEC seconds (a sweep still running)")
    return ap


def harvest(a):
    """Every *.jsonl here (worker files folded into their stem), unless files were named."""
    if a.out or a.shor or a.ghz:
        return
    stems = set()
    for p in Path(".").glob("*.jsonl"):
        mt = re.match(r"^(.*)\.w\d+\.jsonl$", p.name)
        stems.add(mt.group(1) + ".jsonl" if mt else p.name)
    a.out = sorted(stems)
    if a.out:
        print(f"harvesting {len(a.out)} record files: {', '.join(a.out)}")


def main():
    global G
    a = make_args().parse_args()
    harvest(a)
    if not (a.out or a.shor or a.ghz):
        make_args().print_help()
        return
    a.theme = desktop_theme() if a.theme == "auto" else a.theme
    import matplotlib
    display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) or sys.platform in ("win32", "darwin")
    headless = bool(a.save or a.list) or not display
    if headless:
        matplotlib.use("Agg")
    for k in ("keymap.back", "keymap.forward", "keymap.home", "keymap.pan", "keymap.zoom", "keymap.save",
              "keymap.xscale", "keymap.yscale", "keymap.grid", "keymap.grid_minor", "keymap.copy"):
        if k in matplotlib.rcParams:
            matplotlib.rcParams[k] = ["ctrl+s"] if k == "keymap.save" else []
    G = load_graph()
    a.hwxeb, a.theta, a.figs, a.n, a.depth = [], "haar", None, None, None
    G.THETA["show"] = "all"
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):            # the graph's note about views 1-13 does not apply here
        store = G.Store(a)
    if store.lay is None and store.data:
        print("# no BlueQubit release found: the run records' landscapes are drawn, their bits on the lattice "
              "are not (give --repo)")
    files = list(a.out) + list(a.shor) + list(a.ghz)
    import matplotlib.pyplot as plt
    if a.list:
        for i, sc in enumerate(build_scenes(store, files)):
            print(f"[{i}] {sc.title}: {len(sc.sites) or len(sc.xs) * len(sc.ys)} "
                  f"{'cells' if sc.kind == 'landscape' else 'sites'}, {len(sc.couplers)} couplers, {len(sc.links)} links")
            for j, ds in enumerate(sc.datasets):
                ms = list(ds.measures) if ds.lazy is None else ["(computed when shown)"]
                print(f"     {j:>2}  {ds.label}  ->  {', '.join(ms)}")
        return
    fig = plt.figure(figsize=tuple(a.size))
    v = Viewer(fig, store, files, a)
    v.columns = not a.dots
    v.pull = a.pull
    if a.view:
        v.view = tuple(a.view)
    if a.dots or a.pull:
        v.build()
    if not v.scenes:
        print("no records with a shape to draw: give --out / --ghz / --shor files (and --repo for run records)")
    if headless:
        if not a.save:
            a.save = "nighthawk_3d_png"
            print("no display -> writing PNGs to ./nighthawk_3d_png")
        for si in ([v.si] if a.scene is not None else range(len(v.scenes))):
            v.si, v.pick = si, None
            v.select(si, a.dataset, a.measure)
            v.view = tuple(a.view) if a.view else None
            dis = [v.di] if a.dataset is not None else range(len(v.scene.datasets) if a.all_datasets else 1)
            for di in dis:
                v.di = di
                ds = v.dataset
                mi0 = v.mi if a.measure is not None else 0
                for mi in ([mi0] if not a.all_measures else range(len(ds.measures))):
                    v.mi = mi
                    v.draw()
                    v.save()
        return
    plt.show()


if __name__ == "__main__":
    main()
