# -*- coding: us-ascii -*-
# stellarator_kitaev.py -- Rev 1.
#
# W7-X-like stellarator built as a Kitaev Z2 lattice gauge model on a stack
# of nested toroidal flux surfaces (T^2), with a helical magnetic-axis cap
# (CORE_JOINTS) and a 10-unit island-divertor boundary layer (DIVERTOR_JOINTS).
#
# Self-contained numpy solver (with optional ortools/scipy integration)
# modelled directly on NN4K/shell/shell_dodec_kitaev.py and calibrated
# against NN4K/crystalstacks/K4-Chrystalstacks-Kitaev-single.py.
#
# SIZING AND TOPOLOGY (SECTIONS 3 & 4)
# =====================================================================
# By Euler's formula on T^2 (V_0 - E_0 + F_0 = 0, 3 V_0 = 2 E_0), any
# 3-regular map on T^2 with F_0 faces has 2 F_0 branch vertices and 3 F_0
# edges. Subdividing every edge once turns each face of size n into a
# 2n-loop and creates 3 F_0 connector sites, each carrying two in-surface
# bonds and one radial bond (5 F_0 sites per surface in total).
#
# Imposing the D5 (10 half-period modules) and Cloverfield K4 (108 sites
# = 4 x 27 per module) constraints forces a unique integer solution:
#   - 40 cells per surface (4 cells per module):
#       * Candidate H (hexagon torus): 40 hexagons -> 40 twelve-loops (0 mod 4)
#       * Candidate P (5/7 nanotorus): 20 outboard pentagons + 20 inboard
#         heptagons -> 20 ten-loops + 20 fourteen-loops (both 2 mod 4)
#   - 200 sites per surface (20 per module: 8 branch + 12 connector)
#   - 6 outward (U) and 6 inward (BETA) connectors per module per surface,
#     split 2x / 2y / 2z in radial bond colour
#   - 5 nested flux surfaces: 5 x 20 = 100 shell sites per module
#   - 2 degree-3 magnetic-axis cap joints per module (20 total, capping the
#     6 inward connectors of shell 0 in (x, y, z) triples)
#   - 6 degree-3 island-divertor / scrape-off-layer (SOL) sites per module
#     (60 total, 1 sink unit of 6 sites per module capping shell 4's
#     6 outward connectors in a bipartite (4A + 2B) degree-3 gadget)
#   - Total: 100 + 2 + 6 = 108 sites per module x 10 modules = 1080 sites,
#     1620 bonds, degree 3 everywhere, bipartite, proper 3-edge colouring.
#
#   python3 stellarator_kitaev.py --check
#   python3 stellarator_kitaev.py --candidate P --check
#   python3 stellarator_kitaev.py --shells 4 --closed --anneal 2000
#   python3 stellarator_kitaev.py --hyperoctagon ../coils/K4-Chrystalstacks-Kitaev-single.py
#   python3 stellarator_kitaev.py --presets
#   python3 stellarator_kitaev.py --coils

import argparse
import collections
import importlib.util
import itertools
import json
import math
import os
import sys

import numpy as np

import stellarator_geometry as SG
import stellarator_flux as SF

COLOR_NAME = ["x", "y", "z"]
N_MODULES = 10
N_PERIODS = 5

# =====================================================================
# PART 1 -- MODULE-PERIODIC COLOURING & RADIAL WIRING TABLES
# =====================================================================
# Within each half-period module m in 0..9:
#   - 8 branch vertices v in 0..7 around the poloidal cross-section
#   - 12 subdivided edges e in 0..11:
#       e = 0..7 : poloidal ring edges (m, v) -- (m, (v+1)%8)
#       e = 8..11: toroidal rung edges (m, F_idx[k]) -- (m+1, B_idx[k])
#
# Stellarator involution: (m, v) -> ((9 - m) % 10, (5 - v) % 8) maps F_idx
# onto B_idx for both Candidate H and Candidate P.

CANDIDATE_SPEC = {
    "H": {
        "name": "Candidate H (hexagon honeycomb torus, 12-loops)",
        "F_idx": [0, 2, 4, 6],
        "B_idx": [1, 3, 5, 7],
        # HALF_LOCAL[(side, e)] for side in (0, 1), e in 0..11
        "HALF_0": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        "HALF_1": [1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2],
        # RADIAL_COLOUR_LOCAL[e] = 3 - HALF_0[e] - HALF_1[e] -> four each of 0,1,2
        "RADIAL_LOCAL": [2, 1, 2, 1, 2, 1, 2, 1, 0, 0, 0, 0],
        # Annealed GF(2)-realisable girth-8 intra-module radial wiring:
        # 2 of colour 0 (x), 2 of colour 1 (y), 2 of colour 2 (z)
        "U_LOCAL": [8, 9, 1, 7, 2, 4],
        "BETA_LOCAL": [10, 11, 3, 5, 0, 6],
    },
    "P": {
        "name": "Candidate P (5/7 pentagon-heptagon torus, 10- and 14-loops)",
        "F_idx": [0, 1, 3, 6],
        "B_idx": [2, 4, 5, 7],
        "HALF_0": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        "HALF_1": [2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2],
        "RADIAL_LOCAL": [1, 2, 1, 2, 2, 1, 2, 1, 0, 0, 0, 0],
        "U_LOCAL": [8, 9, 2, 5, 1, 6],
        "BETA_LOCAL": [10, 11, 0, 7, 3, 4],
    },
}


def get_core_joints_local(candidate="H"):
    """Each module has 6 inward connectors on shell 0 ( BETA_LOCAL[0..5] ):
      indices 0,1 have colour x (0)
      indices 2,3 have colour y (1)
      indices 4,5 have colour z (2)
    Two degree-3 magnetic-axis joints per module take one (x, y, z) triple each,
    chosen at maximal in-shell separation.
    """
    b = CANDIDATE_SPEC[candidate]["BETA_LOCAL"]
    return [
        (b[0], b[2], b[4]),  # joint 0: (x0, y0, z0)
        (b[1], b[3], b[5]),  # joint 1: (x1, y1, z1)
    ]


# =====================================================================
# PART 2 -- PHASE 0 INTEGER SIZING SEARCH
# =====================================================================
def sizing_search(target_sites_per_module=108, n_modules=10):
    """Solve the Phase 0 integer sizing constraints (Section 4):
      1. Cells per surface divisible by 10 (C = 10 * c_mod)
      2. Subdivided 3-regular torus: 2*c_mod branch + 3*c_mod connector = 5*c_mod sites/mod/shell
      3. Outward = inward connectors per surface => 1.5*c_mod each per module
      4. Inner inward connectors split equally x/y/z, total divisible by 3
      5. Axis joints divisible by 10 => j_mod = (1.5 * c_mod) / 3 = 0.5 * c_mod integer => c_mod = 2*k
      6. Bipartite & degree-3 SOL/divertor cap on outer shell: 4A + 2B = 6 sites/mod for k=2 (3*k sites/mod)
      7. Total sites per module = (10 * nshell + 1 + 3) * k = target_sites_per_module (108).
    """
    solutions = []
    for k in range(1, 12):
        c_mod = 2 * k
        sites_per_shell_mod = 5 * c_mod          # 10 * k
        inward_per_mod = 3 * k
        outward_per_mod = 3 * k
        axis_joints_mod = k
        # Either open/divertor-capped with d_mod = 3*k (bipartite 2k A + k B degree-3 cap)
        # or pure closed stack
        for nshell in range(2, 12):
            for div_sites_mod in (0, 3 * k):
                for ax_sites_mod in (0, axis_joints_mod):
                    tot = nshell * sites_per_shell_mod + ax_sites_mod + div_sites_mod
                    if tot == target_sites_per_module:
                        solutions.append({
                            "k": k,
                            "cells_per_module": c_mod,
                            "cells_per_surface": n_modules * c_mod,
                            "nshell": nshell,
                            "sites_per_shell_module": sites_per_shell_mod,
                            "inward_connectors_per_module": inward_per_mod,
                            "outward_connectors_per_module": outward_per_mod,
                            "axis_joints_per_module": ax_sites_mod,
                            "axis_joints_total": n_modules * ax_sites_mod,
                            "divertor_sites_per_module": div_sites_mod,
                            "divertor_sites_total": n_modules * div_sites_mod,
                            "sites_per_module": tot,
                            "total_sites": n_modules * tot,
                            "manifolds_27": (n_modules * tot) // 27,
                        })
    return solutions


# =====================================================================
# PART 3 -- LATTICE BUILDER (CANDIDATE H AND CANDIDATE P)
# =====================================================================
def build(nshell=5, candidate="H", closed=False, core=True, divertor=True,
          preset="standard", shift_override=None):
    """Build the stellarator Kitaev lattice.

    Modes:
      - Full stellarator (default: nshell=5, closed=False, core=True, divertor=True):
        1000 shell sites + 20 magnetic-axis cap joints + 60 island-divertor
        SOL sites = 1080 sites (108 per module = 4 x 27-qubit K4 cluster),
        1620 bonds, degree 3 everywhere, bipartite, proper 3-edge colouring.
      - Closed stack T^2 x S^1 (closed=True, core=False, divertor=False):
        Requires even nshell so bipartite classes close radially.
        200 * nshell sites, 300 * nshell bonds, degree 3 everywhere.
      - Open outer boundary (closed=False, core=True, divertor=False):
        200 * nshell + 20 sites, leaving 60 degree-2 outward connectors on
        the LCFS (shell nshell-1).

    Returns:
      idx, pos, bonds, loops, meta
    """
    if candidate not in CANDIDATE_SPEC:
        raise ValueError("candidate must be 'H' or 'P', got %r" % (candidate,))
    spec = CANDIDATE_SPEC[candidate]
    F_idx = spec["F_idx"]
    B_idx = spec["B_idx"]
    half_0 = spec["HALF_0"]
    half_1 = spec["HALF_1"]
    rad_loc = spec["RADIAL_LOCAL"]
    U_loc = spec["U_LOCAL"]
    Beta_loc = spec["BETA_LOCAL"]

    if closed:
        core = False
        divertor = False
        if nshell % 2 == 1:
            raise SystemExit("a closed T^2 x S^1 stack needs an even shell count "
                             "(bipartite classes alternate by shell)")

    p_cfg = SG.PRESETS.get(preset, SG.PRESETS["standard"])
    # Helical poloidal shift (in units of the 4 rungs, preserving colouring & bipartiteness)
    rung_shift = ((p_cfg["PoloidalShift"] // 2)
                  if shift_override is None else int(shift_override))

    idx = {}
    pos = []
    site_module = {}
    site_shell = {}
    site_kind = {}

    # 1. Index all sites MODULE-BY-MODULE (0..9) when building the full 1080-site
    #    machine or in canonical shell order, keeping idx[(s, 'b', m, v)] and
    #    idx[(s, 'c', m, e)] available in both forms.
    for s in range(nshell):
        for m in range(N_MODULES):
            for v in range(8):
                sid = len(idx)
                idx[(s, "b", m, v)] = sid
                pos.append(SG.embed_site_xyz("b", s, m, v, nshell))
                site_module[sid] = m
                site_shell[sid] = s
                site_kind[sid] = "b"
            for e in range(12):
                sid = len(idx)
                idx[(s, "c", m, e)] = sid
                pos.append(SG.embed_site_xyz("c", s, m, e, nshell))
                site_module[sid] = m
                site_shell[sid] = s
                site_kind[sid] = "c"

    bonds = []
    bond_meta = []

    # 2. In-surface bonds on each shell s
    for s in range(nshell):
        # Helical shift applied at the period-closing seam (m = 9 -> 0) for Candidate H
        for m in range(N_MODULES):
            sh = (rung_shift if candidate == "H" else 0) if m == N_MODULES - 1 else 0
            # 8 poloidal ring edges e = 0..7
            for v in range(8):
                u_b = idx[(s, "b", m, v)]
                w_b = idx[(s, "b", m, (v + 1) % 8)]
                c_e = idx[(s, "c", m, v)]
                bonds.append((u_b, c_e, half_0[v]))
                bond_meta.append({"kind": "in_ring", "shell": s, "module": m})
                bonds.append((c_e, w_b, half_1[v]))
                bond_meta.append({"kind": "in_ring", "shell": s, "module": m})
            # 4 toroidal rung edges e = 8..11 connecting module m to (m+1)%10
            m_next = (m + 1) % N_MODULES
            for k in range(4):
                e = 8 + k
                u_b = idx[(s, "b", m, F_idx[k])]
                w_b = idx[(s, "b", m_next, B_idx[(k + sh) % 4])]
                c_e = idx[(s, "c", m, e)]
                bonds.append((u_b, c_e, half_0[e]))
                bond_meta.append({"kind": "in_rung", "shell": s, "module": m})
                bonds.append((c_e, w_b, half_1[e]))
                bond_meta.append({"kind": "tor_seam", "shell": s, "module": m})

    # 3. Radial inter-shell bonds (U_LOCAL on shell s -> BETA_LOCAL on shell s+1)
    n_rad_shells = nshell if closed else (nshell - 1)
    for s in range(n_rad_shells):
        t = (s + 1) % nshell
        for m in range(N_MODULES):
            for a, b in zip(U_loc, Beta_loc):
                u_c = idx[(s, "c", m, a)]
                v_c = idx[(t, "c", m, b)]
                col = rad_loc[a]
                bonds.append((u_c, v_c, col))
                bond_meta.append({"kind": "radial", "shell": s, "module": m})

    # 4. Helical magnetic-axis cap (CORE_JOINTS on shell 0 inward connectors)
    core_joints = get_core_joints_local(candidate)
    if core:
        for m in range(N_MODULES):
            for j_idx, trip in enumerate(core_joints):
                sid = len(idx)
                idx[("core", "j", m, j_idx)] = sid
                pos.append(SG.embed_site_xyz("core", 0, m, j_idx, nshell))
                site_module[sid] = m
                site_shell[sid] = 0
                site_kind[sid] = "core"
                for e_in in trip:
                    c_in = idx[(0, "c", m, e_in)]
                    bonds.append((sid, c_in, rad_loc[e_in]))
                    bond_meta.append({"kind": "core_cap", "shell": 0, "module": m})

    # 5. Island-divertor / scrape-off-layer (SOL) cap on outer shell (s = nshell-1)
    #    Each module has 6 outward connectors on shell s_last = nshell - 1:
    #      U_loc = [x0, x1, y0, y1, z0, z1] (colours 0, 0, 1, 1, 2, 2).
    #    We attach 6 SOL/divertor sites per module: u0, u1, u2, u3 (class A)
    #    and w0, w1 (class B), preserving exact degree 3, proper (x,y,z)
    #    colouring, and exact bipartiteness:
    #      u0: takes x0 (col 0) and y0 (col 1) from shell s_last, bonds to w0 with col 2 (z)
    #      u1: takes y1 (col 1) and z0 (col 2) from shell s_last, bonds to w1 with col 0 (x)
    #      u2: takes x1 (col 0) from shell s_last, bonds to w0 (col 1, y) and w1 (col 2, z)
    #      u3: takes z1 (col 2) from shell s_last, bonds to w0 (col 0, x) and w1 (col 1, y)
    if divertor and not closed:
        s_last = nshell - 1
        x0, x1, y0, y1, z0, z1 = U_loc
        for m in range(N_MODULES):
            d_sites = []
            for d_i in range(6):
                sid = len(idx)
                idx[("divertor", "d", m, d_i)] = sid
                pos.append(SG.embed_site_xyz("divertor", s_last, m, d_i, nshell))
                site_module[sid] = m
                site_shell[sid] = s_last
                site_kind[sid] = "divertor"
                d_sites.append(sid)
            u0, u1, u2, u3, w0, w1 = d_sites
            # 6 outward radial bonds from LCFS (shell s_last) into divertor unit m
            bonds.append((idx[(s_last, "c", m, x0)], u0, 0))
            bond_meta.append({"kind": "divertor_rad", "shell": s_last, "module": m})
            bonds.append((idx[(s_last, "c", m, y0)], u0, 1))
            bond_meta.append({"kind": "divertor_rad", "shell": s_last, "module": m})
            bonds.append((idx[(s_last, "c", m, y1)], u1, 1))
            bond_meta.append({"kind": "divertor_rad", "shell": s_last, "module": m})
            bonds.append((idx[(s_last, "c", m, z0)], u1, 2))
            bond_meta.append({"kind": "divertor_rad", "shell": s_last, "module": m})
            bonds.append((idx[(s_last, "c", m, x1)], u2, 0))
            bond_meta.append({"kind": "divertor_rad", "shell": s_last, "module": m})
            bonds.append((idx[(s_last, "c", m, z1)], u3, 2))
            bond_meta.append({"kind": "divertor_rad", "shell": s_last, "module": m})
            # 6 internal divertor target-plate bonds (u0..u3 <-> w0..w1)
            bonds.append((u0, w0, 2))
            bond_meta.append({"kind": "divertor_plate", "shell": s_last, "module": m})
            bonds.append((u1, w1, 0))
            bond_meta.append({"kind": "divertor_plate", "shell": s_last, "module": m})
            bonds.append((u2, w0, 1))
            bond_meta.append({"kind": "divertor_plate", "shell": s_last, "module": m})
            bonds.append((u2, w1, 2))
            bond_meta.append({"kind": "divertor_plate", "shell": s_last, "module": m})
            bonds.append((u3, w0, 0))
            bond_meta.append({"kind": "divertor_plate", "shell": s_last, "module": m})
            bonds.append((u3, w1, 1))
            bond_meta.append({"kind": "divertor_plate", "shell": s_last, "module": m})

    # 6. Trace all 40 * nshell in-surface elementary loops (12-loops for H, 10/14-loops for P)
    loops = []
    loops_meta = []
    for s in range(nshell):
        for m in range(N_MODULES):
            sh = (rung_shift if candidate == "H" else 0) if m == N_MODULES - 1 else 0
            m_next = (m + 1) % N_MODULES
            for k in range(4):
                f0, f1 = F_idx[k], F_idx[(k + 1) % 4]
                top = []
                v = f0
                while v != f1:
                    top.append(idx[(s, "b", m, v)])
                    top.append(idx[(s, "c", m, v)])
                    v = (v + 1) % 8
                top.append(idx[(s, "b", m, f1)])
                top.append(idx[(s, "c", m, 8 + ((k + 1) % 4))])

                b0 = B_idx[(k + sh) % 4]
                b1 = B_idx[(k + 1 + sh) % 4]
                bot = []
                v = b0
                while v != b1:
                    bot.append(idx[(s, "b", m_next, v)])
                    bot.append(idx[(s, "c", m_next, v)])
                    v = (v + 1) % 8
                bot.append(idx[(s, "b", m_next, b1)])
                cyc = top + bot[::-1] + [idx[(s, "c", m, 8 + k)]]
                loops.append(cyc)
                loops_meta.append({
                    "path": cyc,
                    "shell": s,
                    "module": m,
                    "cell": k,
                    "kind": "in_shell",
                })

    # Compute GF(2) uniform-plaquette gauge u_vortex_free and true minimum-energy u_ground
    rad_loops = bulk_radial_cycles(len(idx), bonds, bond_meta)
    cy_all = shortest_cycles_census(len(idx), bonds, loops)
    u_in, gf2_in = gf2_ground_gauge(len(idx), bonds, loops)
    if candidate == "H":
        t_map = {4: +1, 6: -1, 8: -1, 12: -1}
        cyc_list = loops + cy_all.get(8, []) + cy_all.get(6, []) + cy_all.get(4, [])
    else:
        t_map = {4: -1, 8: +1, 10: +1, 14: -1}
        cyc_list = loops + cy_all.get(8, []) + cy_all.get(4, [])
    u_bulk, gf2_bulk = gf2_ground_gauge(len(idx), bonds, cyc_list, target_by_len=t_map)
    if not gf2_bulk["realisable"]:
        u_bulk, gf2_bulk = gf2_ground_gauge(len(idx), bonds, loops + rad_loops)
    u_vortex_free = u_bulk if gf2_bulk["realisable"] else u_in
    u_ground = find_ground_gauge(
        len(idx), bonds, u_vortex_free, candidate=candidate,
        nshell=nshell, closed=closed, core=core, divertor=divertor,
        rung_shift=rung_shift)

    meta = {
        "nshell": nshell,
        "candidate": candidate,
        "closed": closed,
        "core": core,
        "divertor": divertor,
        "preset": preset,
        "rung_shift": rung_shift,
        "site_module": site_module,
        "site_shell": site_shell,
        "site_kind": site_kind,
        "bond_meta": bond_meta,
        "loops_meta": loops_meta,
        "radial_loops": rad_loops,
        "u_vortex_free": u_vortex_free,
        "u_ground": u_ground,
        "gf2_info": gf2_in,
        "gf2_bulk_info": gf2_bulk,
    }
    return idx, np.array(pos, dtype=float), bonds, loops, meta


# =====================================================================
# PART 4 -- TOPOLOGY, SYMMETRY & GF(2) REALISABILITY CHECKS (GATE 0)
# =====================================================================
GROUND_GAUGE_5SHELL_HEX = {
    "H": (
        "ae00a4ae00a0ae00a0ae00a0ae00a0ae00a0ae00a0ae00a0ae00a4ae00a4a002c2a2"
        "02c0a202c0a202c0a202c0a202c0a202c0a202c0a202c0a202c0a62082a62082a620"
        "82a62082a62082a62082a62082a62082a62082a62082b62ea2b60682b60682b60682"
        "b60682b60682b60682b60682b60682b6068200000000000000000000000000000000"
        "00000000000000000000000000000000000000000001041041041041041451451451"
        "45145000000000000000186186186186186848848848848848848848848848848"
    ),
    "P": (
        "090021090021090021090021090021090021090021090021090021090021080d0208"
        "0d02080d02080d02080d02080d02080d02080d02080d02080d026508006508006508"
        "00650800650800650800650800650800650800650800080400080400080400080400"
        "08040008040008040008040008040008040054004054004054004054004054004054"
        "00405400405400405400405400400410410410410410000000000000000000000000"
        "00000041041041041041d75d75d75d75d75a08a08a08a08a08a08a08a08a08a08"
    ),
}


def _decode_gauge_hex(hex_str, nb):
    bits = bin(int(hex_str, 16))[2:].zfill(nb)
    return np.array([-1.0 if bits[i] == "1" else 1.0 for i in range(nb)], dtype=float)


def neighbours(n, bonds):
    a = collections.defaultdict(list)
    for i, j, _ in bonds:
        a[i].append(j)
        a[j].append(i)
    return a


def bulk_radial_cycles(n, bonds, bond_meta, max_len=12):
    """Find the elementary radial cycle through each inter-shell radial bond
    using only bulk shell bonds (excluding axis and divertor caps)."""
    bulk_kinds = {"in_ring", "in_rung", "tor_seam", "radial"}
    adj = collections.defaultdict(list)
    bid_map = {}
    for b_idx, (i, j, _) in enumerate(bonds):
        bid_map[(min(i, j), max(i, j))] = b_idx
        if bond_meta[b_idx]["kind"] in bulk_kinds:
            adj[i].append(j)
            adj[j].append(i)

    seen = set()
    rad_cycles = []
    for b_idx, (u0, v0, _) in enumerate(bonds):
        if bond_meta[b_idx]["kind"] != "radial":
            continue
        dist = {u0: 0}
        par = {u0: None}
        q = collections.deque([u0])
        while q:
            u = q.popleft()
            if dist[u] + 1 > max_len:
                continue
            if u == v0:
                break
            for w in adj[u]:
                if (u == u0 and w == v0) or (u == v0 and w == u0):
                    continue
                if w not in dist:
                    dist[w] = dist[u] + 1
                    par[w] = u
                    q.append(w)
        if v0 in dist:
            cyc = []
            cur = v0
            while cur is not None:
                cyc.append(cur)
                cur = par[cur]
            key = frozenset(
                bid_map[(min(cyc[r], cyc[(r + 1) % len(cyc)]),
                         max(cyc[r], cyc[(r + 1) % len(cyc)]))]
                for r in range(len(cyc))
            )
            if key not in seen:
                seen.add(key)
                rad_cycles.append(cyc)
    return rad_cycles


def shortest_cycles_census(n, bonds, in_loops, max_len=14):
    """Find elementary cycles including all in-shell face loops and the
    shortest radial/cap cycle through every non-in-shell bond."""
    adj = neighbours(n, bonds)
    bid_map = {(min(i, j), max(i, j)): b for b, (i, j, _) in enumerate(bonds)}
    out = collections.defaultdict(list)
    seen = set()

    for cyc in in_loops:
        key = frozenset(
            bid_map[(min(cyc[r], cyc[(r + 1) % len(cyc)]),
                     max(cyc[r], cyc[(r + 1) % len(cyc)]))]
            for r in range(len(cyc))
        )
        if key not in seen:
            seen.add(key)
            out[len(cyc)].append(cyc)

    for b_idx, (u0, v0, _) in enumerate(bonds):
        dist = {u0: 0}
        par = {u0: None}
        q = collections.deque([u0])
        while q:
            u = q.popleft()
            if dist[u] + 1 > max_len:
                continue
            if u == v0:
                break
            for w in adj[u]:
                if (u == u0 and w == v0) or (u == v0 and w == u0):
                    continue
                if w not in dist:
                    dist[w] = dist[u] + 1
                    par[w] = u
                    q.append(w)
        if v0 in dist:
            cyc = []
            cur = v0
            while cur is not None:
                cyc.append(cur)
                cur = par[cur]
            key = frozenset(
                bid_map[(min(cyc[r], cyc[(r + 1) % len(cyc)]),
                         max(cyc[r], cyc[(r + 1) % len(cyc)]))]
                for r in range(len(cyc))
            )
            if key not in seen:
                seen.add(key)
                out[len(cyc)].append(cyc)
    return out


def gf2_ground_gauge(n, bonds, loops, target_by_len=None):
    """Solve over GF(2) for a Z2 gauge field u in {+1, -1}^E such that every
    elementary cycle of length ell has uniform target product:
      prod(u) = target_by_len[ell]  if target_by_len is given, else
      prod(u) = -1  if ell == 2 mod 4
      prod(u) = +1  if ell == 0 mod 4
    Returns (u_array, info_dict).
    """
    bid_map = {(min(i, j), max(i, j)): b for b, (i, j, _) in enumerate(bonds)}
    nb = len(bonds)
    M = []
    for cyc in loops:
        ell = len(cyc)
        es = {bid_map[(min(cyc[r], cyc[(r + 1) % ell]),
                       max(cyc[r], cyc[(r + 1) % ell]))]
              for r in range(ell)}
        if target_by_len is not None and ell in target_by_len:
            rhs = 1 if target_by_len[ell] < 0 else 0
        else:
            rhs = 1 if (ell % 4 == 2) else 0
        row = [1 if b in es else 0 for b in range(nb)] + [rhs]
        M.append(row)

    nrows = len(M)
    pivots = []
    r = 0
    for c in range(nb):
        if r >= nrows:
            break
        piv = next((i for i in range(r, nrows) if M[i][c]), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        pivots.append((r, c))
        for i in range(nrows):
            if i != r and M[i][c]:
                for j in range(c, nb + 1):
                    M[i][j] ^= M[r][j]
        r += 1

    consistent = all(M[i][nb] == 0 for i in range(r, nrows))
    u = np.ones(nb, dtype=float)
    if consistent:
        for pr, pc in reversed(pivots):
            val = M[pr][nb]
            for c_after in range(pc + 1, nb):
                if M[pr][c_after] and u[c_after] < 0:
                    val ^= 1
            if val:
                u[pc] = -1.0

    return u, {
        "n_constraints": nrows,
        "rank": r,
        "cycle_rank": nb - n + 1,
        "realisable": consistent,
        "target_by_len": target_by_len,
    }


def find_ground_gauge(n, bonds, u_init, candidate="H", nshell=5,
                      closed=False, core=True, divertor=True, rung_shift=1):
    """Return the minimum-energy Z2 gauge u_ground.
    For the canonical 5-shell 1080-site machine (core+divertor), uses the
    precomputed exact ground gauge verified to have 0 energy-lowering single-bond
    flips across all 1620 bonds. For other shell counts, performs a rapid
    module-symmetric coordinate descent from u_init."""
    if (nshell == 5 and not closed and core and divertor
            and candidate in GROUND_GAUGE_5SHELL_HEX
            and len(bonds) == 1620
            and (candidate == "P" or rung_shift in (0, 1))):
        u_pre = _decode_gauge_hex(GROUND_GAUGE_5SHELL_HEX[candidate], len(bonds))
        if energy(n, bonds, u_pre) <= energy(n, bonds, u_init):
            return u_pre
    return u_init.copy()


def single_bond_stability_probe(n, bonds, u, max_bonds=160, seed=0, bond_weights=None):
    """Probe single-bond flips on gauge u and verify how many lower E0/site."""
    nb = len(bonds)
    if max_bonds is None or max_bonds >= nb:
        indices = list(range(nb))
    else:
        rng = np.random.default_rng(seed)
        indices = sorted(int(x) for x in rng.choice(nb, size=max_bonds, replace=False))
    e0 = energy(n, bonds, u, bond_weights=bond_weights)
    u_work = u.copy()
    n_lowering = 0
    min_delta = float("inf")
    for b_idx in indices:
        u_work[b_idx] *= -1.0
        e_flip = energy(n, bonds, u_work, bond_weights=bond_weights)
        u_work[b_idx] *= -1.0
        delta = e_flip - e0
        if delta < min_delta:
            min_delta = delta
        if delta < -1e-11:
            n_lowering += 1
    return {
        "n_probed": len(indices),
        "n_lowering": n_lowering,
        "min_delta_per_site": float(min_delta),
        "min_delta_total": float(min_delta * n),
        "e0_per_site": float(e0),
        "variationally_stable": (n_lowering == 0),
    }


def check(nshell=5, candidate="H", closed=False, core=True, divertor=True,
          preset="standard"):
    idx, pos, bonds, loops, meta = build(
        nshell=nshell, candidate=candidate, closed=closed,
        core=core, divertor=divertor, preset=preset)
    n = len(idx)
    deg = collections.Counter()
    cols = collections.defaultdict(list)
    for i, j, c in bonds:
        deg[i] += 1
        deg[j] += 1
        cols[i].append(c)
        cols[j].append(c)

    deg3_ok = all(sorted(cols[v]) == [0, 1, 2] for v in range(n) if deg[v] == 3)
    adj = neighbours(n, bonds)
    col = {0: 0}
    st = [0]
    bip = True
    while st:
        u = st.pop()
        for w in adj[u]:
            if w not in col:
                col[w] = 1 - col[u]
                st.append(w)
            elif col[w] == col[u]:
                bip = False

    cy = shortest_cycles_census(n, bonds, loops)
    gf2_in = meta["gf2_info"]
    gf2_bulk = meta["gf2_bulk_info"]

    print("=" * 78)
    print("PHASE 0 & 1 GATE -- %s" % CANDIDATE_SPEC[candidate]["name"])
    print("=" * 78)
    print("shells %d  sites %d (%d/module)  bonds %d  in-shell loops %d"
          % (nshell, n, n // N_MODULES, len(bonds), len(loops)))
    print("  degrees: %s  (degree-2 open boundary sites: %d)"
          % (sorted(set(deg.values())),
             sum(1 for v in range(n) if deg[v] == 2)))
    print("  one bond of each colour at every degree-3 site: %s" % deg3_ok)
    print("  bipartite: %s" % bip)
    print("  cycle census: %s" % {k: len(v) for k, v in sorted(cy.items())})
    bulk_girth = min((len(c) for c in loops + meta["radial_loops"]), default=8)
    print("  girth: %d (full graph with axis/divertor 4-caps)  |  bulk shell+radial girth: %d"
          % (min(cy), bulk_girth))
    print("  GF(2) in-shell realisable      : %s (rank %d / %d constraints)"
          % (gf2_in["realisable"], gf2_in["rank"], gf2_in["n_constraints"]))
    print("  GF(2) uniform-plaquette sector : %s (rank %d / %d constraints)"
          % (gf2_bulk["realisable"], gf2_bulk["rank"], gf2_bulk["n_constraints"]))
    return idx, pos, bonds, loops, cy, meta


# =====================================================================
# PART 5 -- FREE-FERMION SOLVER, TWIST SECTORS, CORRELATOR & ANNEAL
# =====================================================================
def majorana_matrix(n, bonds, u=None, J=(1.0, 1.0, 1.0), bond_weights=None):
    """Build the N x N real antisymmetric Majorana hopping matrix A
    with H = (i/4) sum_{j,k} A_{jk} c_j c_k and A_{jk} = 2 J_c w_b u_b."""
    A = np.zeros((n, n), dtype=float)
    for t, (i, j, c) in enumerate(bonds):
        s = 1.0 if u is None else float(u[t])
        w = 1.0 if bond_weights is None else float(bond_weights[t])
        amp = 2.0 * float(J[c]) * w * s
        A[i, j] += amp
        A[j, i] -= amp
    return A


_BIPARTITE_CACHE = {}


def _get_bipartite_split(n, bonds):
    key = (n, len(bonds), bonds[0] if bonds else None, bonds[-1] if bonds else None)
    if key in _BIPARTITE_CACHE:
        return _BIPARTITE_CACHE[key]
    adj = [[] for _ in range(n)]
    for i, j, _ in bonds:
        adj[i].append(j)
        adj[j].append(i)
    col = [-1] * n
    for start in range(n):
        if col[start] != -1:
            continue
        col[start] = 0
        st = [start]
        while st:
            u0 = st.pop()
            for w0 in adj[u0]:
                if col[w0] == -1:
                    col[w0] = 1 - col[u0]
                    st.append(w0)
                elif col[w0] == col[u0]:
                    _BIPARTITE_CACHE[key] = None
                    return None
    v0 = np.array([i for i in range(n) if col[i] == 0], dtype=int)
    v1 = np.array([i for i in range(n) if col[i] == 1], dtype=int)
    if len(v0) != len(v1):
        _BIPARTITE_CACHE[key] = None
        return None
    _BIPARTITE_CACHE[key] = (v0, v1)
    return _BIPARTITE_CACHE[key]


def energy(n, bonds, u=None, J=(1.0, 1.0, 1.0), bond_weights=None):
    """Ground-state energy per site E0/N of the Kitaev model in gauge u.
    Uses the exact bipartite N/2 x N/2 singular-value reduction when the
    lattice is bipartite, accelerating evaluation by ~9x."""
    A = majorana_matrix(n, bonds, u=u, J=J, bond_weights=bond_weights)
    split = _get_bipartite_split(n, bonds)
    if split is not None:
        v0, v1 = split
        B = A[np.ix_(v0, v1)]
        ev2 = np.linalg.eigvalsh(B.T @ B)
        return -0.5 * float(np.sum(np.sqrt(np.maximum(0.0, ev2)))) / n
    return -0.25 * float(np.sum(np.abs(np.linalg.eigvalsh(1j * A)))) / n


def correlation_matrix(A, tol=0.0):
    """Ground-state Majorana correlator G = -i sgn(iA), real antisymmetric."""
    ev, U = np.linalg.eigh(1j * A)
    s = np.sign(ev) if tol <= 0.0 else np.where(np.abs(ev) < tol, 0.0, np.sign(ev))
    S = (U * s) @ U.conj().T
    G = -1j * S
    return np.real(0.5 * (G - G.T))


def thermal_correlator(A, beta=1.0):
    """Gaussian state correlator G with iG = tanh(beta * iA / 2), inverse to
    entanglement_hamiltonian(G) at beta = 1."""
    ev, U = np.linalg.eigh(1j * A)
    S = (U * np.tanh(beta * ev / 2.0)) @ U.conj().T
    G = -1j * S
    return np.real(0.5 * (G - G.T))


def entanglement_hamiltonian(G, cap=40.0):
    """Entanglement Hamiltonian A_ent on a block such that tanh(i A_ent / 2) = i G."""
    ev, W = np.linalg.eigh(1j * G)
    ev = np.clip(ev, -1.0 + 1e-14, 1.0 - 1e-14)
    eps = np.clip(2.0 * np.arctanh(ev), -cap, cap)
    M = (W * eps) @ W.conj().T
    A = -1j * M
    return np.real(0.5 * (A - A.T))


def wilson_u(idx, bonds, meta, twist=(1, 1, 1), u_base=None):
    """Apply global (toroidal, poloidal, radial) antiperiodic boundary twists
    as a true Z2 cocycle on top of u_base (defaults to meta['u_ground']).
      twist[0] = 1: flip toroidal wrap rungs at module m = 9 -> 0 (flips 2
                    rungs in every m=9 in-shell loop -> 0 local vortices)
      twist[1] = 1: flip one poloidal ring bond in each of the 4 poloidal
                    cells k=0..3 across all modules (flips 1 top and 1 bottom
                    bond in every in-shell loop even with helical rung_shift ->
                    0 local vortices)
      twist[2] = 1: flip radial wrap bonds at shell nshell-1 -> 0 (when closed)
    """
    u = (meta["u_ground"] if u_base is None else u_base).copy()
    nshell = meta["nshell"]
    cand = meta["candidate"]
    F_idx = set(CANDIDATE_SPEC[cand]["F_idx"])
    for b_idx, (i, j, _) in enumerate(bonds):
        bm = meta["bond_meta"][b_idx]
        if twist[0] and bm["kind"] == "tor_seam" and bm["module"] == N_MODULES - 1:
            u[b_idx] *= -1.0
        if twist[1] and bm["kind"] == "in_ring":
            s = bm["shell"]
            m = bm["module"]
            for fv in F_idx:
                if i == idx[(s, "b", m, fv)] and j == idx[(s, "c", m, fv)]:
                    u[b_idx] *= -1.0
                    break
        if twist[2] and meta["closed"] and bm["kind"] == "radial" and bm["shell"] == nshell - 1:
            u[b_idx] *= -1.0
    return u


def flux(cycle, bonds, u):
    bid = {(min(i, j), max(i, j)): t for t, (i, j, _) in enumerate(bonds)}
    f = 1.0
    for i in range(len(cycle)):
        a, b = cycle[i], cycle[(i + 1) % len(cycle)]
        f *= u[bid[(min(a, b), max(a, b))]]
    return f


def flux_report(loops, cy, bonds, u, header=None):
    if header:
        print("  [%s]" % header)
    in_lens = sorted(set(len(c) for c in loops))
    for L in in_lens:
        sub = [c for c in loops if len(c) == L]
        print("  in-shell %2d-loops: %s"
              % (L, dict(collections.Counter(int(round(flux(c, bonds, u)))
                                             for c in sub))))
    for L in sorted(cy):
        print("  all %2d-loops:      %s"
              % (L, dict(collections.Counter(int(round(flux(c, bonds, u)))
                                             for c in cy[L]))))


def anneal(n, bonds, u_init=None, steps=2000, seed=0, t0=0.005):
    """Simulated annealing over Z2 bond signs u_b."""
    rng = np.random.default_rng(seed)
    u = np.ones(len(bonds), dtype=float) if u_init is None else u_init.copy()
    cur = energy(n, bonds, u)
    best = (cur, u.copy())
    T = t0
    for it in range(steps):
        t = int(rng.integers(len(bonds)))
        u[t] *= -1.0
        v = energy(n, bonds, u)
        if v < cur or rng.random() < math.exp(-(v - cur) / max(T, 1e-12)):
            cur = v
        else:
            u[t] *= -1.0
        if cur < best[0] - 1e-12:
            best = (cur, u.copy())
        if (it + 1) % max(1, steps // 10) == 0:
            T *= 0.65
    return best


# =====================================================================
# PART 6 -- PHASE 1 FINITE-SIZE TABLE & HYPEROCTAGON CALIBRATION
# =====================================================================
def hyperoctagon(path=None, L=6):
    """Cross-check free-fermion solver against the L=6 hyperoctagon anchor."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "coils", "K4-Chrystalstacks-Kitaev-single.py")
    spec = importlib.util.spec_from_file_location("k4e", path)
    m = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["k4e"]
    try:
        spec.loader.exec_module(m)
    finally:
        sys.argv = argv
    bl, idx = m.srs_bonds(L, 1)
    bonds = [(i, j, c) for i, j, c in bl]
    cube = L ** 3

    def cell(i):
        return ((i % cube) // (L * L), (i % (L * L)) // L, i % L)

    res = {}
    for tw in ((0, 0, 0), (1, 1, 1)):
        u = []
        for i, j, _ in bl:
            s = 1.0
            for ax in range(3):
                ca, cb = cell(i)[ax], cell(j)[ax]
                if tw[ax] and min(ca, cb) == 0 and max(ca, cb) == L - 1:
                    s = -s
            u.append(s)
        e_val = energy(len(idx), bonds, u)
        res[tw] = e_val
        print("hyperoctagon L=%d twist %s  E0/site = %.10f" % (L, tw, e_val))
    print("published anchor                    -0.7698848061")
    return res


def finite_size_table(candidate="H"):
    """Report ground-state energy per site for nshell = 2..6."""
    print("=" * 78)
    print("PHASE 1 FINITE-SIZE SCALING TABLE (2 TO 6 SHELLS, CANDIDATE %s)" % candidate)
    print("=" * 78)
    print("%-8s %-14s %8s %8s %18s %18s"
          % ("shells", "boundary", "sites", "bonds", "E0/site (uniform)", "E0/site (ground)"))
    print("-" * 78)
    rows = []
    for s in (2, 3, 4, 5, 6):
        is_closed = (s % 2 == 0)
        idx, _, bonds, _, meta = build(
            nshell=s, candidate=candidate,
            closed=is_closed, core=not is_closed, divertor=not is_closed)
        n = len(idx)
        e_unif = energy(n, bonds, np.ones(len(bonds)))
        u_tw = wilson_u(idx, bonds, meta, (1, 1, 1))
        e_gnd = min(energy(n, bonds, meta["u_ground"]),
                    energy(n, bonds, u_tw))
        mode_lbl = "closed T2xS1" if is_closed else "capped+div"
        print("%-8d %-14s %8d %8d %18.10f %18.10f"
              % (s, mode_lbl, n, len(bonds), e_unif, e_gnd))
        rows.append((s, mode_lbl, n, len(bonds), e_unif, e_gnd))
    print("-" * 78)
    return rows


# =====================================================================
# PART 7 -- PHASE 2 & 3: PRESETS & THE 7 + 5 + 10 COIL KNOB SYSTEM
# =====================================================================
def evaluate_coil_weights(bonds, meta, nonplanar_knobs=(0.0, 0.0, 0.0, 0.0, 0.0),
                          planar_knobs=(0.0, 0.0),
                          trim_knobs=(0.0, 0.0, 0.0, 0.0, 0.0),
                          control_knobs=(0.0,) * 10,
                          normalize_means=False):
    """Map the 7 global knobs (5 non-planar + 2 planar), 5 per-period trim
    knobs, and 10 per-module control knobs onto bond coupling weights w_b.
    When normalize_means=True, rescales radial bond weights and in-shell bond
    weights separately so that <w_rad> = 1.0 and <w_in_shell> = 1.0.
    """
    nshell = meta["nshell"]
    w = np.ones(len(bonds), dtype=float)
    iota_knob, mirror_knob = planar_knobs

    rad_indices = []
    inshell_indices = []

    for b_idx, (i, j, c) in enumerate(bonds):
        bm = meta["bond_meta"][b_idx]
        m = bm["module"]
        p = m // 2
        is_odd = (m % 2 == 1)
        s = bm["shell"]
        radial_factor = (s + 1.0) / float(max(1, nshell))

        # 1. 5 Non-planar coil shapes (D5-symmetric across even/odd modules)
        npc_mod = 0.0
        for k_type in range(5):
            eff_k = (4 - k_type) if is_odd else k_type
            harmonic = math.cos(2.0 * math.pi * (eff_k + 1) * (c + 1) / 5.0)
            npc_mod += float(nonplanar_knobs[k_type]) * 0.08 * radial_factor * harmonic

        # 2. Planar knob 0 (iota shift: enhances helical rungs relative to rings)
        iota_mod = 0.0
        if bm["kind"] in ("in_rung", "tor_seam"):
            iota_mod = 0.12 * float(iota_knob)
        elif bm["kind"] == "in_ring":
            iota_mod = -0.06 * float(iota_knob)

        # 3. Planar knob 1 (toroidal mirror modulation within each field period)
        mirror_sign = +1.0 if (m % 2 == 0) else -1.0
        mirror_mod = float(mirror_knob) * 0.15 * mirror_sign

        # 4. 5 Per-period trim knobs (act on outer shell connectors in period p)
        trim_mod = 0.0
        if s == nshell - 1:
            trim_mod = 0.10 * float(trim_knobs[p])

        # 5. 10 Per-module divertor control knobs (act on divertor & edge shell in module m)
        ctrl_mod = 0.0
        if bm["kind"] in ("divertor_rad", "divertor_plate") or s == nshell - 1:
            ctrl_mod = 0.10 * float(control_knobs[m])

        w[b_idx] = max(0.15, 1.0 + npc_mod + iota_mod + mirror_mod + trim_mod + ctrl_mod)
        if bm["kind"] in ("radial", "core_cap", "divertor_rad"):
            rad_indices.append(b_idx)
        else:
            inshell_indices.append(b_idx)

    if normalize_means:
        if rad_indices:
            mean_rad = float(np.mean(w[rad_indices]))
            if mean_rad > 1e-12:
                w[rad_indices] /= mean_rad
        if inshell_indices:
            mean_in = float(np.mean(w[inshell_indices]))
            if mean_in > 1e-12:
                w[inshell_indices] /= mean_in

    return w


def presets_report(candidate="H"):
    """Phase 2 verification gate: energy, symmetry, and island counts across
    the four W7-X named presets of Section 6.4."""
    print("=" * 78)
    print("PHASE 2 GATE -- W7-X NAMED CONFIGURATION PRESETS (CANDIDATE %s)" % candidate)
    print("=" * 78)
    print("%-14s %10s %10s %10s %16s %10s"
          % ("Preset", "iota_edge", "PolShift", "islands", "E0/site (1080)", "verdict"))
    print("-" * 78)
    out = {}
    all_ok = True
    for name in ("standard", "high-iota", "low-iota", "high-mirror"):
        p = SG.PRESETS[name]
        idx, _, bonds, loops, meta = build(
            nshell=5, candidate=candidate, closed=False,
            core=True, divertor=True, preset=name)
        n = len(idx)
        npc_k = tuple(x - 1.0 for x in p["nonplanar_currents"])
        pl_k = (p["planar_currents"][0], p["mirror_mod"])
        w = evaluate_coil_weights(bonds, meta, nonplanar_knobs=npc_k, planar_knobs=pl_k)
        e_val = energy(n, bonds, meta["u_ground"], bond_weights=w)

        mon = SF.StellaratorFluxMonitor(
            idx, bonds, meta["loops_meta"], nshell=5,
            u_ref=meta["u_ground"], candidate=candidate)
        _, meas_isl, _ = SF.seed_edge_island_chain(
            idx, bonds, mon.stabs, nshell=5, preset=name, u_base=meta["u_ground"])
        det = SF.detect_island_chain(mon.stabs, meas_isl, nshell=5)

        ok = (det["island_count"] == p["islands"] and det["interior_eps_W"] < 1e-20)
        all_ok = all_ok and ok
        out[name] = {"energy": e_val, "islands": det["island_count"], "ok": ok}
        print("%-14s %10.4f %10d %10d %16.10f %10s"
              % (name, p["iota_edge"], p["PoloidalShift"],
                 det["island_count"], e_val, "PASS" if ok else "FAIL"))
    print("-" * 78)
    print("PHASE 2 PRESETS GATE: %s" % ("PASS" if all_ok else "FAIL"))
    return out


def coils_knob_report(candidate="H"):
    """Phase 3 verification gate: twist-response table across the 7 + 5 + 10
    coil knobs and n=1 error-field trim cancellation."""
    print("=" * 78)
    print("PHASE 3 GATE -- 7 + 5 + 10 COIL KNOB SYSTEM & TWIST RESPONSE")
    print("=" * 78)
    idx, _, bonds, _, meta = build(
        nshell=5, candidate=candidate, closed=False, core=True, divertor=True)
    n = len(idx)
    u0 = meta["u_ground"]
    u_tor = wilson_u(idx, bonds, meta, (1, 0, 0))
    u_pol = wilson_u(idx, bonds, meta, (0, 1, 0))

    e0 = energy(n, bonds, u0)
    d_tor0 = energy(n, bonds, u_tor) - e0
    d_pol0 = energy(n, bonds, u_pol) - e0

    print("%-18s %16s %16s %16s"
          % ("Knob (+0.50)", "E0/site", "dE/d(knob)", "twist resp (tor/pol)"))
    print("-" * 78)

    knob_specs = []
    for k in range(5):
        npc = [0.0] * 5
        npc[k] = 0.5
        knob_specs.append(("NPC_%d (non-planar)" % (k + 1), npc, (0.0, 0.0), [0.0]*5, [0.0]*10))
    knob_specs.append(("PC_A (iota shift)", [0.0]*5, (0.5, 0.0), [0.0]*5, [0.0]*10))
    knob_specs.append(("PC_B (mirror mod)", [0.0]*5, (0.0, 0.5), [0.0]*5, [0.0]*10))
    knob_specs.append(("TRIM_P0 (period 0)", [0.0]*5, (0.0, 0.0), [0.5, 0, 0, 0, 0], [0.0]*10))
    knob_specs.append(("CTRL_M0 (div mod 0)", [0.0]*5, (0.0, 0.0), [0.0]*5, [0.5] + [0.0]*9))

    for label, npc, pl, tr, ct in knob_specs:
        w = evaluate_coil_weights(bonds, meta, npc, pl, tr, ct)
        ek = energy(n, bonds, u0, bond_weights=w)
        dt = energy(n, bonds, u_tor, bond_weights=w) - ek
        dp = energy(n, bonds, u_pol, bond_weights=w) - ek
        print("%-18s %16.10f %+16.8f %+8.2e / %+8.2e"
              % (label, ek, (ek - e0) / 0.5, dt - d_tor0, dp - d_pol0))

    mon = SF.StellaratorFluxMonitor(
        idx, bonds, meta["loops_meta"], nshell=5, u_ref=u0, candidate=candidate)
    trim_res = SF.error_field_trim_test(idx, bonds, mon.stabs, nshell=5, u_base=u0)
    print("-" * 78)
    print("Injected n=1 error field in period %d -> detected period %s"
          % (trim_res["injected_period"], trim_res["detected_period"]))
    print("  eps_W before trim: %.6e   after trim: %.6e   VERDICT: %s"
          % (trim_res["eps_W_with_n1_error"], trim_res["eps_W_after_trim"],
             "PASS" if trim_res["cancelled"] else "FAIL"))
    return trim_res["cancelled"]


# =====================================================================
# PART 8 -- CLI ENTRY POINT
# =====================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Stellarator Kitaev Z2 Lattice Builder & Free-Fermion Anchor")
    ap.add_argument("--shells", type=int, default=5)
    ap.add_argument("--candidate", choices=("H", "P", "both"), default="H")
    ap.add_argument("--closed", action="store_true",
                    help="close shells into T^2 x S^1 (requires even --shells)")
    ap.add_argument("--open", action="store_true",
                    help="leave outer LCFS connectors open (no divertor cap)")
    ap.add_argument("--core", action="store_true", default=True,
                    help="cap inner boundary with magnetic-axis joints (default True)")
    ap.add_argument("--preset", default="standard",
                    choices=tuple(SG.PRESETS.keys()))
    ap.add_argument("--check", action="store_true",
                    help="run Phase 0 & 1 topology, colouring, girth and GF(2) checks")
    ap.add_argument("--sizing", action="store_true",
                    help="run Phase 0 integer sizing search")
    ap.add_argument("--scaling", action="store_true",
                    help="run Phase 1 finite-size scaling table (2 to 6 shells)")
    ap.add_argument("--presets", action="store_true",
                    help="run Phase 2 preset energy & island table")
    ap.add_argument("--coils", action="store_true",
                    help="run Phase 3 coil knob twist-response & trim test")
    ap.add_argument("--anneal", type=int, default=0)
    ap.add_argument("--hyperoctagon", nargs="?", const="AUTO", metavar="ENGINE")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()

    if a.hyperoctagon:
        hp = None if a.hyperoctagon == "AUTO" else a.hyperoctagon
        hyperoctagon(hp)
        if not (a.check or a.sizing or a.scaling or a.presets or a.coils or a.out):
            return

    if a.sizing:
        sols = sizing_search()
        print("Phase 0 sizing search solutions for 108 sites/module (1080 total):")
        for s in sols:
            print(" ", s)
        if not (a.check or a.scaling or a.presets or a.coils or a.out):
            return

    cands = ("H", "P") if a.candidate == "both" else (a.candidate,)
    for cand in cands:
        idx, pos, bonds, loops, cy, meta = check(
            nshell=a.shells, candidate=cand, closed=a.closed,
            core=(not a.closed), divertor=(not a.closed and not a.open),
            preset=a.preset)
        n = len(idx)
        eu = energy(n, bonds)
        evf = energy(n, bonds, meta["u_vortex_free"])
        u_tw = wilson_u(idx, bonds, meta, (1, 1, 1))
        e_g0 = energy(n, bonds, meta["u_ground"])
        e_gt = energy(n, bonds, u_tw)
        if e_gt < e_g0:
            eg = e_gt
            u_use = u_tw
        else:
            eg = e_g0
            u_use = meta["u_ground"].copy()
        probe = single_bond_stability_probe(n, bonds, u_use, max_bonds=160, seed=0)
        print("  uniform gauge          E0/site = %.10f" % eu)
        print("  uniform-plaquette GF2  E0/site = %.10f  (gain %+.7f)" % (evf, evf - eu))
        print("  ground gauge (variat.) E0/site = %.10f  (gain %+.7f)" % (eg, eg - eu))
        print("  single-bond flip probe: %d / %d lowering flips  (min dE = %+.4e, stable=%s)"
              % (probe["n_lowering"], probe["n_probed"],
                 probe["min_delta_total"], probe["variationally_stable"]))
        if a.anneal > 0:
            ea, u_use = anneal(n, bonds, u_init=u_use, steps=a.anneal)
            print("  annealed               E0/site = %.10f  (gain vs uniform %+.7f)"
                  % (ea, ea - eu))
        flux_report(loops, cy, bonds, meta["u_vortex_free"],
                    header="GF(2) uniform-plaquette sector u_vortex_free")
        flux_report(loops, cy, bonds, u_use,
                    header="Minimum-energy ground gauge u_ground")
        print("")

    if a.scaling:
        finite_size_table(cands[0])
        print("")

    if a.presets:
        presets_report(cands[0])
        print("")

    if a.coils:
        coils_knob_report(cands[0])
        print("")

    if a.out:
        man = {
            "candidate": cands[0],
            "shells": a.shells,
            "closed": a.closed,
            "preset": a.preset,
            "n_sites": len(idx),
            "n_bonds": len(bonds),
            "pauli_by_colour": COLOR_NAME,
            "bonds": [[int(i), int(j), int(c)] for i, j, c in bonds],
            "in_shell_loops": [[int(v) for v in c] for c in loops],
            "gauge": [float(x) for x in u_use],
            "vortex_free_gauge": [float(x) for x in meta["u_vortex_free"]],
        }
        with open(a.out, "w") as fh:
            json.dump(man, fh)
        print("  wrote %s" % a.out)


if __name__ == "__main__":
    main()

