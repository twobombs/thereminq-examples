# -*- coding: us-ascii -*-
# stellarator-manifest.py -- Rev 1.
#
# Build the self-contained JSON manifest (`manifest.json`) for the 40 x 27-site
# W7-X-like stellarator Kitaev lattice from `equiv.json`, modelled on
# NN4K/coils/K4-32x27-manifest.py.
#
# Includes:
#   - Per-block qubit tables (q = 0..26 -> global site 0..1079), internal bonds,
#     and canonical intra-manifold loop stabilisers (Pauli words & signs)
#   - Intra-module K4 cluster junction bond tables (6 pairs per module)
#   - Inter-module toroidal seam bond tables (connecting module m to m+1)
#   - Radial inter-shell bond routing tables
#   - 7 + 5 + 10 W7-X coil winding paths and coupling weights
#
#   python3 stellarator-manifest.py --equiv equiv.json -o manifest.json
#   python3 stellarator-manifest.py --equiv equiv.json --validate-only
#   python3 stellarator-manifest.py --equiv equiv.json --coils-only

import argparse
import collections
import json
import sys

import numpy as np

import stellarator_flux as SF
import stellarator_geometry as SG
import stellarator_kitaev as SK


def build_coil_windings(idx, bonds, meta):
    """Construct lattice coil winding paths for the 50 non-planar coils
    (5 D5-symmetric types), 20 planar coils (2 types), 5 trim coils, and
    10 divertor control coils on the outer shell (s = nshell - 1)."""
    s_edge = meta["nshell"] - 1
    nonplanar = []
    for m in range(SK.N_MODULES):
        t_npc = min(m, 9 - m)
        # Poloidal ring path around module m on outer shell
        ring_sites = []
        for v in range(8):
            ring_sites.append(int(idx[(s_edge, "b", m, v)]))
            ring_sites.append(int(idx[(s_edge, "c", m, v)]))
        nonplanar.append({
            "coil_group": "nonplanar",
            "module": m,
            "period": m // 2,
            "type": t_npc,
            "sites": ring_sites,
        })

    planar = []
    for m in range(SK.N_MODULES):
        for p_type in range(2):
            planar.append({
                "coil_group": "planar",
                "module": m,
                "period": m // 2,
                "type": p_type,
                "sites": [int(idx[(s_edge, "c", m, 8 + k)]) for k in range(4)],
            })

    trim = []
    for p in range(SK.N_PERIODS):
        m_even = 2 * p
        trim.append({
            "coil_group": "trim",
            "period": p,
            "module": m_even,
            "sites": [
                int(idx[(s_edge, "b", m_even, 1)]),
                int(idx[(s_edge, "c", m_even, 1)]),
                int(idx[(s_edge, "b", m_even, 2)]),
            ],
        })

    control = []
    for m in range(SK.N_MODULES):
        div_sites = [int(idx[("divertor", "d", m, d)]) for d in range(6)]
        control.append({
            "coil_group": "control",
            "module": m,
            "period": m // 2,
            "sites": div_sites,
        })

    return {
        "nonplanar": nonplanar,
        "planar": planar,
        "trim": trim,
        "control": control,
    }


def find_block_internal_cycles(sites_g, internal_bonds, primary_loop_g,
                               colmap, u_ground, eid):
    """Find the minimum-length GF(2)-independent cycle basis of the 27-qubit
    block (exact cycle rank = len(internal_bonds) - 26), seeded with the
    canonical 8-cycle primary_loop_g."""
    g_to_q = {g: q for q, g in enumerate(sites_g)}
    adj = collections.defaultdict(list)
    edge_list = []
    edge_to_idx = {}
    for idx_e, ib in enumerate(internal_bonds):
        u, v = int(ib["ga"]), int(ib["gb"])
        adj[u].append(v)
        adj[v].append(u)
        e_key = (min(u, v), max(u, v))
        edge_list.append(e_key)
        edge_to_idx[e_key] = idx_e

    cycle_rank = max(1, len(internal_bonds) - (len(sites_g) - 1))

    # Enumerate all simple cycles in the 27-vertex induced subgraph (cycle rank <= 6)
    all_cycles = [list(primary_loop_g)]

    def dfs_cycles(start, cur, visited, path):
        for nxt in sorted(adj[cur]):
            if nxt == start and len(path) >= 4:
                all_cycles.append(list(path))
            elif nxt > start and nxt not in visited and len(path) < 14:
                visited.add(nxt)
                path.append(nxt)
                dfs_cycles(start, nxt, visited, path)
                path.pop()
                visited.remove(nxt)

    for v_start in sorted(sites_g):
        dfs_cycles(v_start, v_start, {v_start}, [v_start])

    # Sort candidate cycles after primary_loop_g by length so we get the shortest basis
    primary_cyc = all_cycles[0]
    other_cycs = sorted(all_cycles[1:], key=len)
    candidates = [primary_cyc] + other_cycs

    basis_vecs = []
    cycles_g = []
    for cyc in candidates:
        ell = len(cyc)
        mask = 0
        valid = True
        for r in range(ell):
            e_key = (min(cyc[r], cyc[(r + 1) % ell]),
                     max(cyc[r], cyc[(r + 1) % ell]))
            if e_key not in edge_to_idx:
                valid = False
                break
            mask ^= (1 << edge_to_idx[e_key])
        if not valid or mask == 0:
            continue
        # Reduce mask against current GF(2) basis
        rem = mask
        for piv_bit, b_vec in basis_vecs:
            if (rem >> piv_bit) & 1:
                rem ^= b_vec
        if rem != 0:
            piv = rem.bit_length() - 1
            basis_vecs.append((piv, rem))
            basis_vecs.sort(reverse=True)
            cycles_g.append(cyc)
            if len(cycles_g) >= cycle_rank:
                break

    loops_out = []
    for cyc_id, cyc_g in enumerate(cycles_g):
        pstr_glob = SF.pauli_string(cyc_g, colmap)
        pstr_loc = [[int(g_to_q[g]), int(col)] for g, col in pstr_glob]
        s_loop = int(SF.loop_sign(cyc_g, colmap))
        f_loop = int(SF.loop_flux(cyc_g, u_ground, eid))
        tgt_loop = float(s_loop * f_loop)
        pword = "".join("XYZ"[col] for _, col in pstr_loc)
        loops_out.append({
            "cycle_index": cyc_id,
            "sites": [int(g) for g in cyc_g],
            "qubits": [int(g_to_q[g]) for g in cyc_g],
            "length": len(cyc_g),
            "pauli_local": pstr_loc,
            "pauli_word": pword,
            "sign": s_loop,
            "flux": f_loop,
            "target": tgt_loop,
        })
    return loops_out


def _pauli_eps(c1, c2):
    """Levi-Civita sign for sigma^{c1} sigma^{c2} = i * eps * sigma^{c3}."""
    return +1.0 if (c1, c2) in ((0, 1), (1, 2), (2, 0)) else -1.0


def _shortest_path_pauli_string(qa, qb, adj):
    """Find shortest internal Kitaev path qa -> ... -> qb inside a 27-qubit
    block and return `(pauli_string, string_sign, path_len)` satisfying:
      i * c_qa * c_qb = string_sign * prod_{(q, c) in pauli_string} sigma^c_q
    so H_Sigma = (i/2) * Sigma_{qa,qb} * c_qa * c_qb = 0.5 * Sigma_{qa,qb} * string_sign * P."""
    q = collections.deque([(int(qa), [])])
    vis = {int(qa)}
    found = None
    while q:
        curr, path = q.popleft()
        if curr == int(qb):
            found = path
            break
        for nxt, col, u_dir in adj[curr]:
            if nxt not in vis:
                vis.add(nxt)
                q.append((nxt, path + [(curr, nxt, col, u_dir)]))
    if not found:
        return [], 1.0, 0
    k = len(found)
    u_prod = 1.0
    eps_prod = 1.0
    p_list = [[int(found[0][0]), int(found[0][2])]]
    for r in range(k):
        u_prod *= float(found[r][3])
        if r >= 1:
            c_prev = int(found[r - 1][2])
            c_curr = int(found[r][2])
            eps_prod *= _pauli_eps(c_prev, c_curr)
            p_list.append([int(found[r][0]), int(3 - c_prev - c_curr)])
    p_list.append([int(found[-1][1]), int(found[-1][2])])
    s_str = float(((-1) ** k) * eps_prod * u_prod)
    return p_list, s_str, k


def compute_sigma_repair_bonds(sites_g, internal_bonds, A_full, G_full,
                               seam_qubits_set, max_terms=45):
    """Compute the Gaussian seam self-energy Sigma_B = A_ent,B - A_int,B on a
    27-qubit block and extract the leading boundary repair terms together with
    their exact internal Kitaev Pauli strings (`N9`)."""
    A_int = A_full[np.ix_(sites_g, sites_g)]
    G_blk = G_full[np.ix_(sites_g, sites_g)]
    A_ent = SK.entanglement_hamiltonian(G_blk)
    Sigma = A_ent - A_int

    ib_map = {}
    n_loc = len(sites_g)
    adj = {q: [] for q in range(n_loc)}
    for ib in internal_bonds:
        qa, qb = int(ib["qa"]), int(ib["qb"])
        col = int(ib["colour"])
        u_dir = float(ib["u"])
        ib_map[(min(qa, qb), max(qa, qb))] = ib
        adj[qa].append((qb, col, u_dir))
        adj[qb].append((qa, col, -u_dir))

    candidates = []
    for qa in range(n_loc):
        for qb in range(qa + 1, n_loc):
            sig_val = float(Sigma[qa, qb])
            if abs(sig_val) < 1e-6:
                continue
            ib = ib_map.get((qa, qb))
            p_str, s_str, path_len = _shortest_path_pauli_string(qa, qb, adj)
            if not p_str or path_len > 3:
                continue
            if ib is not None:
                col = int(ib["colour"])
                pauli = ib["pauli"]
                is_bond = True
            else:
                col = int(p_str[0][1])
                pauli = "".join(SK.COLOR_NAME[c] for _, c in p_str)
                is_bond = False
            touches_seam = (qa in seam_qubits_set) or (qb in seam_qubits_set)
            candidates.append({
                "qa": int(qa),
                "qb": int(qb),
                "ga": int(sites_g[qa]),
                "gb": int(sites_g[qb]),
                "colour": int(col),
                "pauli": pauli,
                "sigma_ab": sig_val,
                "string_sign": float(s_str),
                "pauli_string": p_str,
                "path_length": int(path_len),
                "abs_sigma": abs(sig_val),
                "is_internal_bond": bool(is_bond),
                "touches_seam": bool(touches_seam),
            })

    candidates.sort(key=lambda d: d["abs_sigma"], reverse=True)
    kept = candidates[:max_terms]
    for item in kept:
        item.pop("abs_sigma", None)

    frob_total = float(np.linalg.norm(Sigma, "fro"))
    Sigma_keep = np.zeros_like(Sigma)
    for item in kept:
        qa, qb, sv = item["qa"], item["qb"], item["sigma_ab"]
        Sigma_keep[qa, qb] = sv
        Sigma_keep[qb, qa] = -sv
    frob_res = float(np.linalg.norm(Sigma - Sigma_keep, "fro"))
    captured = 1.0 - (frob_res / max(1e-12, frob_total))
    return kept, {
        "sigma_frob": frob_total,
        "residual_frob": frob_res,
        "captured_fraction": float(captured),
        "n_keep": len(kept),
    }


def generate_manifest(equiv_path="equiv.json", preset_override=None):
    with open(equiv_path, "r") as fh:
        eq = json.load(fh)

    cand = eq.get("candidate", "H")
    nshell = int(eq.get("shells", 5))
    preset = preset_override or eq.get("preset", "standard")

    idx, pos, bonds, loops, meta = SK.build(
        nshell=nshell, candidate=cand, closed=False,
        core=True, divertor=True, preset=preset)
    colmap, eid = SF.colour_map_and_eid(bonds)
    u_ground = meta["u_ground"]
    u_vortex_free = meta["u_vortex_free"]
    p_cfg = SG.PRESETS[preset]
    npc_k = tuple(x - 1.0 for x in p_cfg["nonplanar_currents"])
    pl_k = (p_cfg["planar_currents"][0], p_cfg["mirror_mod"])
    weights = SK.evaluate_coil_weights(bonds, meta, nonplanar_knobs=npc_k, planar_knobs=pl_k)

    # Precompute global Majorana matrix and ground-state correlator for Sigma_keep repair
    A_full = SK.majorana_matrix(len(idx), bonds, u=u_ground, bond_weights=weights)
    G_full = SK.correlation_matrix(A_full)

    # Map global site -> (block_id, local_qubit_idx)
    site_to_bq = {}
    blocks_out = []

    for b in eq["blocks"]:
        bid = int(b["block_id"])
        loop_g = [int(s) for s in b["loop"]]
        tree_g = [int(s) for s in b["tree"]]
        sites_g = loop_g + tree_g
        if len(sites_g) != 27:
            raise ValueError("Block %d has %d sites != 27" % (bid, len(sites_g)))
        for q_loc, g_site in enumerate(sites_g):
            site_to_bq[g_site] = (bid, q_loc)

    if len(site_to_bq) != len(idx):
        raise ValueError("Manifest site coverage %d != %d" % (len(site_to_bq), len(idx)))

    # Classify all 1620 bonds into internal, intra-module K4 junction, or toroidal seam
    internal_by_block = collections.defaultdict(list)
    seam_qubits_by_block = collections.defaultdict(set)
    k4_junctions = []
    toroidal_seams = []
    radial_bonds_info = []

    for b_idx, (i, j, c) in enumerate(bonds):
        bi, qi = site_to_bq[i]
        bj, qj = site_to_bq[j]
        mi, mj = bi // 4, bj // 4
        u_val = float(u_ground[b_idx])
        w_val = float(weights[b_idx])
        bm = meta["bond_meta"][b_idx]

        if bi == bj:
            internal_by_block[bi].append({
                "qa": int(qi),
                "qb": int(qj),
                "ga": int(i),
                "gb": int(j),
                "colour": int(c),
                "pauli": SK.COLOR_NAME[c],
                "bond_id": int(b_idx),
                "u": u_val,
                "weight": w_val,
                "kind": bm["kind"],
            })
        elif mi == mj:
            seam_qubits_by_block[bi].add(int(qi))
            seam_qubits_by_block[bj].add(int(qj))
            k4_junctions.append({
                "module": int(mi),
                "block_a": int(bi),
                "qa": int(qi),
                "ga": int(i),
                "block_b": int(bj),
                "qb": int(qj),
                "gb": int(j),
                "colour": int(c),
                "pauli": SK.COLOR_NAME[c],
                "bond_id": int(b_idx),
                "u": u_val,
                "weight": w_val,
                "kind": bm["kind"],
            })
        else:
            seam_qubits_by_block[bi].add(int(qi))
            seam_qubits_by_block[bj].add(int(qj))
            toroidal_seams.append({
                "module_a": int(mi),
                "block_a": int(bi),
                "qa": int(qi),
                "ga": int(i),
                "module_b": int(mj),
                "block_b": int(bj),
                "qb": int(qj),
                "gb": int(j),
                "colour": int(c),
                "pauli": SK.COLOR_NAME[c],
                "bond_id": int(b_idx),
                "u": u_val,
                "weight": w_val,
                "kind": bm["kind"],
            })

        if bm["kind"] in ("radial", "core_cap", "divertor_rad"):
            radial_bonds_info.append({
                "bond_id": int(b_idx),
                "kind": bm["kind"],
                "shell": int(bm["shell"]),
                "module": int(bm["module"]),
                "block_a": int(bi),
                "qa": int(qi),
                "block_b": int(bj),
                "qb": int(qj),
                "colour": int(c),
                "internal": bool(bi == bj),
            })

    # Build per-block manifest entries including all internal cycles and Sigma_keep repair
    for b in eq["blocks"]:
        bid = int(b["block_id"])
        loop_g = [int(s) for s in b["loop"]]
        tree_g = [int(s) for s in b["tree"]]
        sites_g = loop_g + tree_g

        internal_loops = find_block_internal_cycles(
            sites_g, internal_by_block[bid], loop_g, colmap, u_ground, eid)
        primary_lp = internal_loops[0]

        div_qs = [q for q, g in enumerate(sites_g) if meta["site_kind"][g] == "divertor"]
        div_q_set = set(div_qs)
        core_qs = [q for q, g in enumerate(sites_g) if meta["site_kind"][g] == "core"]
        shell_by_q = [int(meta["site_shell"][g]) for g in sites_g]
        seam_qs = sorted(seam_qubits_by_block[bid])
        seam_q_set = set(seam_qs)

        # 2-qubit internal bonds incident on divertor qubits (or edge shell 4) for bond-parity sinks
        div_bonds = [
            {
                "qa": int(ib["qa"]),
                "qb": int(ib["qb"]),
                "colour": int(ib["colour"]),
                "pauli": ib["pauli"],
                "bond_id": int(ib["bond_id"]),
                "u": float(ib["u"]),
                "kind": ib["kind"],
            }
            for ib in internal_by_block[bid]
            if (ib["qa"] in div_q_set or ib["qb"] in div_q_set
                or ib["kind"] in ("divertor_rad", "divertor_plate")
                or shell_by_q[ib["qa"]] == nshell - 1
                or shell_by_q[ib["qb"]] == nshell - 1)
        ]

        # 2-qubit internal bonds incident on seam boundary qubits for bond-type seam kicks
        seam_boundary_bonds = [
            {
                "qa": int(ib["qa"]),
                "qb": int(ib["qb"]),
                "colour": int(ib["colour"]),
                "pauli": ib["pauli"],
                "bond_id": int(ib["bond_id"]),
                "u": float(ib["u"]),
                "weight": float(ib["weight"]),
            }
            for ib in internal_by_block[bid]
            if (ib["qa"] in seam_q_set or ib["qb"] in seam_q_set)
        ]

        sigma_bonds, sigma_stats = compute_sigma_repair_bonds(
            sites_g, internal_by_block[bid], A_full, G_full, seam_q_set, max_terms=45)

        blocks_out.append({
            "block_id": bid,
            "module": int(b["module"]),
            "period": int(b["period"]),
            "parity": int(b["parity"]),
            "type": int(b["type"]),
            "loop": loop_g,
            "tree": tree_g,
            "sites": sites_g,
            "shells": shell_by_q,
            "core_qubits": core_qs,
            "divertor_qubits": div_qs,
            "seam_qubits": seam_qs,
            "internal_bonds": internal_by_block[bid],
            "seam_boundary_bonds": seam_boundary_bonds,
            "divertor_bonds": div_bonds,
            "sigma_repair_bonds": sigma_bonds,
            "sigma_repair_stats": sigma_stats,
            "internal_loops": internal_loops,
            "loop_pauli_local": primary_lp["pauli_local"],
            "loop_pauli_word": primary_lp["pauli_word"],
            "loop_sign": primary_lp["sign"],
            "loop_target": primary_lp["target"],
        })

    # Canonical 4 block types (from module 0's 4 blocks: block_id = 0, 1, 2, 3)
    block_types_out = []
    for k in range(4):
        b0 = blocks_out[k]
        block_types_out.append({
            "type_id": k,
            "loop_len": len(b0["loop"]),
            "tree_len": len(b0["tree"]),
            "n_internal_bonds": len(b0["internal_bonds"]),
            "n_internal_loops": len(b0["internal_loops"]),
            "internal_bonds": [
                {"qa": ib["qa"], "qb": ib["qb"], "colour": ib["colour"],
                 "pauli": ib["pauli"], "u": ib["u"], "weight": ib["weight"],
                 "kind": ib["kind"]}
                for ib in b0["internal_bonds"]
            ],
            "sigma_repair_bonds": b0["sigma_repair_bonds"],
            "sigma_repair_stats": b0["sigma_repair_stats"],
            "divertor_bonds": b0["divertor_bonds"],
            "seam_boundary_bonds": b0["seam_boundary_bonds"],
            "internal_loops": b0["internal_loops"],
            "loop_pauli_local": b0["loop_pauli_local"],
            "loop_pauli_word": b0["loop_pauli_word"],
            "loop_sign": b0["loop_sign"],
            "loop_target": b0["loop_target"],
            "core_qubits": b0["core_qubits"],
            "divertor_qubits": b0["divertor_qubits"],
            "seam_qubits": b0["seam_qubits"],
        })

    coils = build_coil_windings(idx, bonds, meta)

    manifest = {
        "candidate": cand,
        "shells": nshell,
        "preset": preset,
        "n_sites": len(idx),
        "n_bonds": len(bonds),
        "n_blocks": len(blocks_out),
        "block_size": 27,
        "n_modules": SK.N_MODULES,
        "blocks_per_module": 4,
        "pauli_by_colour": SK.COLOR_NAME,
        "bonds": [[int(i), int(j), int(c)] for i, j, c in bonds],
        "gauge": [float(x) for x in u_ground],
        "vortex_free_gauge": [float(x) for x in u_vortex_free],
        "weights": [float(x) for x in weights],
        "in_shell_loops": [
            {"path": [int(v) for v in lm["path"]],
             "shell": int(lm["shell"]),
             "module": int(lm["module"]),
             "cell": int(lm["cell"])}
            for lm in meta["loops_meta"]
        ],
        "block_types": block_types_out,
        "blocks": blocks_out,
        "k4_junctions": k4_junctions,
        "toroidal_seams": toroidal_seams,
        "radial_bonds": radial_bonds_info,
        "coils": coils,
    }
    return manifest


def validate_manifest(manifest, coils_only=False):
    print("=" * 78)
    print("PHASE 4 GATE (MANIFEST) -- 40 x 27 STELLARATOR MANIFEST & COIL WINDINGS")
    print("=" * 78)
    n_int = sum(len(b["internal_bonds"]) for b in manifest["blocks"])
    n_k4 = len(manifest["k4_junctions"])
    n_tor = len(manifest["toroidal_seams"])
    tot_bonds = n_int + n_k4 + n_tor
    tot_int_loops = sum(len(b["internal_loops"]) for b in manifest["blocks"])
    expected_int_loops = n_int - manifest["n_blocks"] * (manifest["block_size"] - 1)

    print("  candidate: %s   preset: %s   shells: %d"
          % (manifest["candidate"], manifest["preset"], manifest["shells"]))
    print("  blocks: %d x %d sites = %d sites   (total internal cycles: %d / %d cycle rank)"
          % (manifest["n_blocks"], manifest["block_size"],
             manifest["n_sites"], tot_int_loops, expected_int_loops))
    print("  bonds: %d total = %d internal + %d K4 junction + %d toroidal seam"
          % (tot_bonds, n_int, n_k4, n_tor))
    for bt in manifest["block_types"]:
        print("  block type %d: %2d-loop (%s, target %+d) | %d internal cycles | %2d bonds | Sigma_keep=%d (%.1f%%) | div_bonds=%d"
              % (bt["type_id"], bt["loop_len"], bt["loop_pauli_word"],
                 int(bt["loop_target"]), bt["n_internal_loops"],
                 bt["n_internal_bonds"],
                 bt["sigma_repair_stats"]["n_keep"],
                 100.0 * bt["sigma_repair_stats"]["captured_fraction"],
                 len(bt["divertor_bonds"])))
    c = manifest["coils"]
    print("  coil windings: %d non-planar | %d planar | %d trim | %d divertor control"
          % (len(c["nonplanar"]), len(c["planar"]), len(c["trim"]), len(c["control"])))
    ok = (tot_bonds == manifest["n_bonds"]
          and manifest["n_blocks"] == 40
          and manifest["n_sites"] == 1080
          and tot_int_loops == expected_int_loops)
    print("  MANIFEST GATE VERDICT: %s" % ("PASS" if ok else "FAIL"))
    print("=" * 78)
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Build stellarator 40x27 manifest.json from equiv.json")
    ap.add_argument("--equiv", default="equiv.json")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--coils-only", action="store_true")
    ap.add_argument("-o", "--out", default="manifest.json")
    a = ap.parse_args()

    manifest = generate_manifest(a.equiv, preset_override=a.preset)
    ok = validate_manifest(manifest, coils_only=a.coils_only)
    if not a.validate_only and not a.coils_only and a.out:
        with open(a.out, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print("  wrote %s" % a.out)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
