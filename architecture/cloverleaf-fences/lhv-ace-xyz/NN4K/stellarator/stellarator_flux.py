# -*- coding: us-ascii -*-
# stellarator_flux.py -- FluxMonitor adapter, per-surface toroidal/poloidal
# Wilson lines, island-chain detector, and n=1 error-field trim diagnostics.
#
# Adapts NN4K/crystalstacks/K4-Chrystalstacks-Kitaev-gaugeprotection.py to
# the nested T2 flux surfaces of the W7-X-like Kitaev stellarator:
#   - Stabiliser Pauli words W_p = prod_{r=1..ell} sigma^{gamma_r}_{j_r}
#     with gamma_r the dangling colour at loop site j_r
#   - Exact spin-to-Majorana loop sign s_p = (-1)^(ell + n_ret + 1)
#   - eps_W scalar syndrome, intra/inter manifold split (eps_W_split),
#     and per-shell flux integrity (eps_W_by_shell)
#   - Non-contractible toroidal and poloidal Wilson lines per T2 surface
#     separating the 4 surface twist sectors (t_tor, t_pol) in {0,1}^2
#   - Edge-shell island-chain detector verifying 5, 4, 6, 5 islands across
#     the standard (5/5), high-iota (5/4), low-iota (5/6), and high-mirror
#     presets while interior shells remain at eps_W = 0
#   - Deliberate n=1 error-field injection and 5-trim-coil cancellation
#
#   python3 stellarator_flux.py --check
#   python3 stellarator_flux.py --preset high-iota

import argparse
import collections
import itertools
import json
import math
import sys

import numpy as np

COLOR_NAME = ["x", "y", "z"]
_PAULI = {
    0: np.array([[0, 1], [1, 0]], dtype=complex),
    1: np.array([[0, -1j], [1j, 0]], dtype=complex),
    2: np.array([[1, 0], [0, -1]], dtype=complex),
}


# =====================================================================
# PART 1 -- STABILISERS & SIGN RULE
# =====================================================================
def colour_map_and_eid(bonds):
    """Build {(i, j): colour} and {(i, j): (bond_idx, +1/-1)} maps."""
    cm = {}
    eid = {}
    for b, (i, j, c) in enumerate(bonds):
        cm[(i, j)] = int(c)
        cm[(j, i)] = int(c)
        eid[(i, j)] = (b, +1.0)
        eid[(j, i)] = (b, -1.0)
    return cm, eid


def loop_flux(path, u, eid):
    """Product of traversal-signed u_jk around a closed path."""
    ell = len(path)
    w = 1.0
    for r in range(ell):
        b, sgn = eid[(path[r], path[(r + 1) % ell])]
        w *= sgn * float(u[b])
    return float(w)


def loop_sign(path, colmap):
    """s_p with <W_p>_spin = s_p * loop_flux(path, u, eid)."""
    n = len(path)
    al = [colmap[(path[r], path[(r + 1) % n])] for r in range(n)]
    ret = sum(1 for r in range(n) if (al[r] - al[(r - 1) % n]) % 3 == 2)
    return +1 if (n + ret) % 2 else -1


def pauli_string(path, colmap):
    """[(site, colour)] -- the dangling colour at each loop site."""
    n = len(path)
    out = []
    for r, site in enumerate(path):
        c1 = colmap[(site, path[(r - 1) % n])]
        c2 = colmap[(site, path[(r + 1) % n])]
        if c1 == c2:
            raise ValueError("loop reuses colour %d at site %d" % (c1, site))
        out.append((int(site), int(3 - c1 - c2)))
    return out


def pauli_expectation(psi, ops, n_qubits=None):
    """<psi| prod_q sigma^{ops[q]}_q |psi> without forming the dense operator."""
    n = int(round(np.log2(psi.size))) if n_qubits is None else n_qubits
    if 2 ** n != psi.size:
        raise ValueError("psi has %d amplitudes, not 2**%d" % (psi.size, n))
    phi = psi.reshape((2,) * n)
    for q, col in sorted(ops.items()):
        phi = np.moveaxis(np.tensordot(_PAULI[col], phi, axes=([1], [q])), 0, q)
    return complex(np.vdot(psi.reshape(-1), phi.reshape(-1)))


def build_stabilizers(bonds, loops_with_meta, u_ref=None):
    """Construct stabiliser records for a list of loops.

    loops_with_meta: list of dicts or tuples (path, shell, module, kind)
    """
    colmap, eid = colour_map_and_eid(bonds)
    if u_ref is None:
        u_ref = np.ones(len(bonds), dtype=float)
    stabs = []
    for idx_l, item in enumerate(loops_with_meta):
        if isinstance(item, dict):
            path = tuple(item["path"])
            shell = item.get("shell", 0)
            module = item.get("module", 0)
            kind = item.get("kind", "in_shell")
        else:
            path = tuple(item)
            shell = idx_l // 40
            module = (idx_l % 40) // 4
            kind = "in_shell"
        ell = len(path)
        s = loop_sign(path, colmap)
        pstr = pauli_string(path, colmap)
        bword = "".join(COLOR_NAME[colmap[(path[r], path[(r + 1) % ell])]]
                        for r in range(ell))
        dword = "".join("XYZ"[c] for _, c in pstr)
        tgt = float(s * loop_flux(path, u_ref, eid))
        stabs.append({
            "id": idx_l,
            "key": frozenset(path),
            "path": path,
            "shell": shell,
            "module": module,
            "kind": kind,
            "ell": ell,
            "pauli": pstr,
            "bond_word": bword,
            "pauli_word": dword,
            "sign": s,
            "target": tgt,
        })
    return stabs, (u_ref, eid, colmap)


def w_from_u(stabs, u, eid):
    """Exact <W_p> spin expectation for any Z2 gauge field u."""
    return {st["key"]: float(st["sign"] * loop_flux(st["path"], u, eid))
            for st in stabs}


# =====================================================================
# PART 2 -- eps_W SYNDROME, INTRA/INTER SPLIT & SHELL INTEGRITY
# =====================================================================
def eps_W(stabs, measured, keys=None):
    """Mean over selected loops of ((<W> - target) / 2)^2."""
    sel = [st for st in stabs if keys is None or st["key"] in keys]
    if not sel:
        return 0.0, 0
    acc = 0.0
    for st in sel:
        w = measured.get(st["key"])
        if w is None:
            raise KeyError("no <W> supplied for loop %s" % (st["path"],))
        acc += ((float(w) - st["target"]) / 2.0) ** 2
    return float(acc / len(sel)), len(sel)


def eps_W_split(stabs, measured, block_of):
    """Intra-manifold vs inter-manifold eps_W split."""
    intra, inter = set(), set()
    for st in stabs:
        blocks = {block_of.get(s, ("_", s)) for s in st["path"]}
        (intra if len(blocks) == 1 else inter).add(st["key"])
    ei, ni = eps_W(stabs, measured, intra)
    eo, no = eps_W(stabs, measured, inter)
    return {"intra": ei, "n_intra": ni, "inter": eo, "n_inter": no}


def eps_W_by_shell(stabs, measured):
    """Return {shell_idx: (eps_W_shell, count)} across all shells."""
    by_s = collections.defaultdict(set)
    for st in stabs:
        by_s[st["shell"]].add(st["key"])
    out = {}
    for s in sorted(by_s):
        out[s] = eps_W(stabs, measured, by_s[s])
    return out


def worst_bonds(stabs, measured, top=8):
    """Localise faulty bonds by ranking mean violation of incident loops."""
    score = collections.defaultdict(list)
    for st in stabs:
        v = ((float(measured[st["key"]]) - st["target"]) / 2.0) ** 2
        p = st["path"]
        for r in range(len(p)):
            a, b = p[r], p[(r + 1) % len(p)]
            score[(min(a, b), max(a, b))].append(v)
    return sorted(((float(np.mean(v)), len(v), k)
                   for k, v in score.items()), reverse=True)[:top]


# =====================================================================
# PART 3 -- TOROIDAL & POLOIDAL WILSON LINES PER FLUX SURFACE
# =====================================================================
def surface_wilson_lines(idx, bonds, nshell, candidate="H", shift=0, u_ref=None):
    """Construct the two fundamental H_1(T^2; Z_2) non-contractible loop
    operators on each flux surface s = 0 .. nshell-1:
      - poloidal Wilson line (s, 'pol'): 16-site ring around module 0
      - toroidal Wilson line (s, 'tor'): 40-site or 60-site cycle around
        all 10 modules on surface s.
    """
    colmap, eid = colour_map_and_eid(bonds)
    if u_ref is None:
        u_ref = np.ones(len(bonds), dtype=float)

    adj_s = [collections.defaultdict(list) for _ in range(nshell)]
    site_shell = {}
    site_mod = {}
    for k, v in idx.items():
        if isinstance(k, tuple) and len(k) == 4 and isinstance(k[0], int):
            site_shell[v] = k[0]
            site_mod[v] = k[2]

    for i, j, _ in bonds:
        si = site_shell.get(i, -1)
        sj = site_shell.get(j, -1)
        if si >= 0 and si == sj:
            adj_s[si][i].append(j)
            adj_s[si][j].append(i)

    lines = []
    for s in range(nshell):
        # 1. Poloidal Wilson line: module 0 ring of 8 branch + 8 connector sites
        pol_path = []
        for v in range(8):
            pol_path.append(idx[(s, "b", 0, v)])
            pol_path.append(idx[(s, "c", 0, v)])
        pol_path = tuple(pol_path)
        s_pol = loop_sign(pol_path, colmap)
        lines.append(((s, "pol"), {
            "key": ("wilson", s, "pol"),
            "shell": s,
            "axis": "pol",
            "path": pol_path,
            "pauli": pauli_string(pol_path, colmap),
            "sign": s_pol,
            "target": float(s_pol * loop_flux(pol_path, u_ref, eid)),
        }))

        # 2. Toroidal Wilson line: shortest cycle on shell s that visits all 10 modules
        start = idx[(s, "b", 0, 0)]
        # Walk forward module by module m = 0 -> 1 -> ... -> 9 -> 0 using BFS per module step
        tor_path = []
        cur = start
        for m in range(10):
            nxt_mod = (m + 1) % 10
            # Target in nxt_mod: if nxt_mod == 0, return to start; else any branch site in nxt_mod
            # Use global BFS on shell s from module 0 to module 0 wrapping positively in module index
            pass

        # Clean exact shortest non-contractible toroidal cycle via unwrapped lift (u, m_unwrapped)
        q = collections.deque([(start, 0, [start])])
        visited = {(start, 0)}
        best_tor = None
        while q:
            u, m_unw, path = q.popleft()
            for w in adj_s[s][u]:
                dm = (site_mod[w] - site_mod[u]) % 10
                if dm == 9:
                    dm = -1
                elif dm > 1:
                    dm = 0
                nw = m_unw + dm
                if w == start and nw == 10:
                    best_tor = tuple(path)
                    q.clear()
                    break
                if 0 <= nw <= 10 and (w, nw) not in visited:
                    visited.add((w, nw))
                    q.append((w, nw, path + [w]))
        if best_tor is None:
            raise RuntimeError("Could not find toroidal Wilson loop on shell %d" % s)
        s_tor = loop_sign(best_tor, colmap)
        lines.append(((s, "tor"), {
            "key": ("wilson", s, "tor"),
            "shell": s,
            "axis": "tor",
            "path": best_tor,
            "pauli": pauli_string(best_tor, colmap),
            "sign": s_tor,
            "target": float(s_tor * loop_flux(best_tor, u_ref, eid)),
        }))

    return lines


# =====================================================================
# PART 4 -- ISLAND-CHAIN DETECTOR & ERROR-FIELD TRIM CANCELLATION
# =====================================================================
PRESET_ISLANDS = {
    "standard": 5,
    "high-iota": 4,
    "low-iota": 6,
    "high-mirror": 5,
}


def seed_edge_island_chain(idx, bonds, stabs, nshell, preset="standard", u_base=None):
    """Apply the resonant edge island-divertor flux pattern on the outer
    shell (s = nshell - 1) corresponding to `preset` while keeping all
    interior shells (s < nshell - 1) strictly in the clean eps_W = 0 sector.

    Returns:
        u_island:   bond gauge array with the edge island chain imprinted
        island_loops: list of edge-shell loop keys carrying the resonant
                      island O-point flux defects (5 for standard/high-mirror,
                      4 for high-iota, 6 for low-iota).
    """
    _, eid = colour_map_and_eid(bonds)
    u = np.ones(len(bonds), dtype=float) if u_base is None else np.array(u_base, dtype=float).copy()
    n_islands = PRESET_ISLANDS.get(preset, 5)
    edge_shell = nshell - 1

    # Select n_islands resonant toroidal rung connectors around the 10 modules
    # on the outer shell. The two loops sharing rung e = 8 + r in module m
    # intersect ONLY along that single subdivided rung edge, and choosing
    # r = (2 * k) % 4 separates adjacent modules poloidally so distinct islands
    # never share an unflipped boundary edge.
    F_idx = [0, 2, 4, 6] if (idx.get(("cand",), "H") == "H") else [0, 2, 4, 6]
    island_modules = [int(round(k * 10.0 / n_islands)) % 10 for k in range(n_islands)]
    flipped_bonds = []
    for k, m in enumerate(island_modules):
        r_cell = (2 * k) % 4
        e_rung = 8 + r_cell
        c_site = idx[(edge_shell, "c", m, e_rung)]
        # Find the in-module half-edge incident on c_site
        b_idx = None
        for b_cand_v in range(8):
            b_site = idx[(edge_shell, "b", m, b_cand_v)]
            if (b_site, c_site) in eid:
                b_idx, _ = eid[(b_site, c_site)]
                break
        u[b_idx] *= -1.0
        flipped_bonds.append(b_idx)

    meas = w_from_u(stabs, u, eid)
    defect_clusters = []
    for k, m in enumerate(island_modules):
        defect_clusters.append({
            "island_index": k,
            "module": m,
            "rung_cell": (2 * k) % 4,
            "bond_index": flipped_bonds[k],
        })

    return u, meas, defect_clusters


def detect_island_chain(stabs, measured, nshell):
    """Analyze a measured flux table across all shells:
      - verify interior shells (0 .. nshell-2) have eps_W == 0
      - identify the resonant island chain multiplicity on the outer shell.
    Each subdivided edge e on the outer shell consists of two half-edges
    (u_b, c_e) and (c_e, w_b) meeting at connector site c_e (which has the
    larger site index within its module/shell). Grouping defective half-edges
    with sc > 0.75 by their shared connector site c_e counts the exact
    resonant island centers.
    """
    by_shell = eps_W_by_shell(stabs, measured)
    interior_eps = max([by_shell[s][0] for s in range(nshell - 1)] or [0.0])
    edge_eps, n_edge = by_shell.get(nshell - 1, (0.0, 0))

    edge_stabs = [st for st in stabs if st["shell"] == nshell - 1]
    # Map each edge-shell half-edge to the modules of its incident loops
    bond_mods = collections.defaultdict(set)
    for st in edge_stabs:
        p = st["path"]
        for r in range(len(p)):
            a, b = p[r], p[(r + 1) % len(p)]
            bond_mods[(min(a, b), max(a, b))].add(st["module"])

    susp = worst_bonds(edge_stabs, measured, top=32)
    # Intra-module island rung dipoles share two loops in the SAME module
    active_half_edges = [
        k for sc, nl, k in susp
        if sc > 0.75 and len(bond_mods.get(k, set())) == 1
    ]
    site_counts = collections.Counter()
    for a, b in active_half_edges:
        site_counts[a] += 1
        site_counts[b] += 1
    active_connectors = sorted(s for s, cnt in site_counts.items() if cnt >= 2)

    return {
        "interior_eps_W": interior_eps,
        "edge_eps_W": edge_eps,
        "island_count": len(active_connectors),
        "active_defect_bonds": active_half_edges,
        "active_connectors": active_connectors,
        "by_shell": by_shell,
    }


def error_field_trim_test(idx, bonds, stabs, nshell, preset="standard", u_base=None):
    """Inject a deliberate n=1 (period-1) toroidal error field on the edge
    shell, detect its module profile and faulty bond signature via
    FluxMonitor, and cancel it using the 5 per-period trim coil knobs."""
    _, eid = colour_map_and_eid(bonds)
    u_isl, meas_clean, islands = seed_edge_island_chain(
        idx, bonds, stabs, nshell, preset, u_base)

    edge_shell = nshell - 1
    # Inject n=1 error field in period p_err = 2 (module 4, poloidal connector 1)
    p_err = 2
    m_err = 2 * p_err
    b_err_site = idx[(edge_shell, "b", m_err, 1)]
    c_err_site = idx[(edge_shell, "c", m_err, 1)]
    err_bond_idx, _ = eid[(b_err_site, c_err_site)]

    u_perturbed = u_isl.copy()
    u_perturbed[err_bond_idx] *= -1.0
    meas_perturbed = w_from_u(stabs, u_perturbed, eid)

    # Detect the injected n=1 error by comparing against the preset's reference target
    stabs_preset = [dict(st, target=meas_clean[st["key"]]) for st in stabs]
    eps_before, _ = eps_W(stabs_preset, meas_perturbed)
    suspects = worst_bonds(stabs_preset, meas_perturbed, top=6)
    suspect_sites = {s for sc, nl, pair in suspects if sc > 0.75 for s in pair}

    # Trim coil cancellation: each of the 5 trim coils controls one field period p=0..4
    detected_period = None
    detected_bond_pair = None
    for p in range(5):
        m_p = 2 * p
        b_cand = idx[(edge_shell, "b", m_p, 1)]
        c_cand = idx[(edge_shell, "c", m_p, 1)]
        if c_cand in suspect_sites:
            detected_period = p
            detected_bond_pair = (min(b_cand, c_cand), max(b_cand, c_cand))
            break

    trim_knobs = [0.0] * 5
    u_cancelled = u_perturbed.copy()
    if detected_period is not None and detected_bond_pair is not None:
        trim_knobs[detected_period] = 1.0
        b_fix, _ = eid[detected_bond_pair]
        u_cancelled[b_fix] *= -1.0

    meas_cancelled = w_from_u(stabs, u_cancelled, eid)
    eps_after, _ = eps_W(stabs_preset, meas_cancelled)

    return {
        "preset": preset,
        "injected_period": p_err,
        "detected_period": detected_period,
        "eps_W_with_n1_error": eps_before,
        "eps_W_after_trim": eps_after,
        "trim_knobs": trim_knobs,
        "cancelled": (detected_period == p_err and eps_after < 1e-15),
    }


# =====================================================================
# PART 5 -- FLUX MONITOR CLASS
# =====================================================================
class StellaratorFluxMonitor(object):
    """Runtime flux syndrome and Wilson-sector monitor for the stellarator."""

    def __init__(self, idx, bonds, loops_with_meta, nshell,
                 block_of=None, u_ref=None, candidate="H", shift=0):
        self.idx = idx
        self.bonds = bonds
        self.nshell = nshell
        self.candidate = candidate
        self.shift = shift
        self.stabs, (self.u_ref, self.eid, self.colmap) = build_stabilizers(
            bonds, loops_with_meta, u_ref)
        self.by_key = {st["key"]: st for st in self.stabs}
        self.lines = surface_wilson_lines(
            idx, bonds, nshell, candidate, shift, self.u_ref)
        self.by_key.update({rec["key"]: rec for _, rec in self.lines})
        self.block_of = dict(block_of or {})
        self.history = []
        self._last = {}

    def targets(self, site_to_qubit=None, keys=None):
        """Return [(key, {qubit: colour})] measurable on the given qubit map."""
        out = []
        for st in self.stabs:
            if keys is not None and st["key"] not in keys:
                continue
            if site_to_qubit is None:
                out.append((st["key"], {s: c for s, c in st["pauli"]}))
            elif all(s in site_to_qubit for s, _ in st["pauli"]):
                out.append((st["key"],
                            {site_to_qubit[s]: c for s, c in st["pauli"]}))
        return out

    def update(self, measured, label=None):
        w = {k: (v.real if isinstance(v, complex) else float(v))
             for k, v in measured.items()}
        seen = [st for st in self.stabs if st["key"] in w]
        total, n = eps_W(seen, w)
        rec = {"label": label, "n": n, "eps_W": total}
        if self.block_of:
            rec.update(eps_W_split(seen, w, self.block_of))
        rec["by_shell"] = eps_W_by_shell(seen, w)
        rec["worst"] = (max((abs(w[st["key"]] - st["target"]), st["path"])
                            for st in seen) if seen else (0.0, ()))
        self.history.append(rec)
        self._last = w
        return rec

    def suspects(self, top=6):
        return worst_bonds([st for st in self.stabs if st["key"] in self._last],
                           self._last, top)

    def line(self, measured=None, label=None):
        r = (self.update(measured, label) if measured is not None
             else self.history[-1])
        s = "eps_W %.3e (%d)" % (r["eps_W"], r["n"])
        if "intra" in r:
            s += "  intra %.3e (%d)  inter %.3e (%d)" % (
                r["intra"], r["n_intra"], r["inter"], r["n_inter"])
        s += "  worst |dW| %.3f" % r["worst"][0]
        return (("[%s] " % r["label"]) if r["label"] else "") + s


def flux_self_test():
    import stellarator_kitaev as SK
    print("=" * 78)
    print("STELLARATOR FLUX MONITOR, WILSON SECTORS & ISLAND-CHAIN GATE")
    print("=" * 78)
    idx, pos, bonds, loops, meta = SK.build(
        nshell=5, candidate="H", closed=False, core=True, divertor=True)
    u_ref = meta["u_ground"]
    mon = StellaratorFluxMonitor(
        idx, bonds, meta["loops_meta"], nshell=5, u_ref=u_ref, candidate="H")
    clean = w_from_u(mon.stabs, u_ref, mon.eid)
    line_str = mon.line(clean, "anchor_ground")
    print(line_str)
    print("Wilson lines built: %d (2 per surface x 5 shells)" % len(mon.lines))
    print("")
    print("%-14s %14s %14s %10s %14s %8s"
          % ("Preset", "interior eps_W", "edge eps_W", "islands", "trim n=1 fix", "verdict"))
    print("-" * 78)
    all_ok = (mon.history[-1]["eps_W"] < 1e-20 and len(mon.lines) == 10)
    for preset in ("standard", "high-iota", "low-iota", "high-mirror"):
        _, meas_isl, _ = seed_edge_island_chain(
            idx, bonds, mon.stabs, nshell=5, preset=preset, u_base=u_ref)
        det = detect_island_chain(mon.stabs, meas_isl, nshell=5)
        trim_res = error_field_trim_test(
            idx, bonds, mon.stabs, nshell=5, preset=preset, u_base=u_ref)
        want_isl = PRESET_ISLANDS[preset]
        ok = (det["interior_eps_W"] < 1e-20
              and det["island_count"] == want_isl
              and trim_res["cancelled"])
        all_ok = all_ok and ok
        print("%-14s %14.3e %14.3e %10d %14s %8s"
              % (preset, det["interior_eps_W"], det["edge_eps_W"],
                 det["island_count"],
                 "%.3e->0" % trim_res["eps_W_with_n1_error"],
                 "PASS" if ok else "FAIL"))
    print("-" * 78)
    print("OVERALL FLUX & ISLAND GATE: %s" % ("PASS" if all_ok else "FAIL"))
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="Stellarator FluxMonitor & Island Detector")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--preset", default="standard",
                    choices=tuple(PRESET_ISLANDS.keys()))
    a = ap.parse_args()
    ok = flux_self_test()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
