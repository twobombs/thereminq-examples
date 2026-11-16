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
    p_cfg = SG.PRESETS[preset]
    npc_k = tuple(x - 1.0 for x in p_cfg["nonplanar_currents"])
    pl_k = (p_cfg["planar_currents"][0], p_cfg["mirror_mod"])
    weights = SK.evaluate_coil_weights(bonds, meta, nonplanar_knobs=npc_k, planar_knobs=pl_k)

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

    # Build per-block manifest entries including canonical loop stabiliser words
    for b in eq["blocks"]:
        bid = int(b["block_id"])
        loop_g = [int(s) for s in b["loop"]]
        tree_g = [int(s) for s in b["tree"]]
        sites_g = loop_g + tree_g
        g_to_q = {g: q for q, g in enumerate(sites_g)}

        pstr_glob = SF.pauli_string(loop_g, colmap)
        pstr_loc = [[int(g_to_q[g]), int(col)] for g, col in pstr_glob]
        s_loop = int(SF.loop_sign(loop_g, colmap))
        f_loop = int(SF.loop_flux(loop_g, u_ground, eid))
        tgt_loop = float(s_loop * f_loop)
        pword = "".join("XYZ"[col] for _, col in pstr_loc)

        div_qs = [q for q, g in enumerate(sites_g) if meta["site_kind"][g] == "divertor"]
        core_qs = [q for q, g in enumerate(sites_g) if meta["site_kind"][g] == "core"]
        shell_by_q = [int(meta["site_shell"][g]) for g in sites_g]

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
            "internal_bonds": internal_by_block[bid],
            "loop_pauli_local": pstr_loc,
            "loop_pauli_word": pword,
            "loop_sign": s_loop,
            "loop_target": tgt_loop,
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
            "internal_bonds": [
                {"qa": ib["qa"], "qb": ib["qb"], "colour": ib["colour"],
                 "pauli": ib["pauli"], "u": ib["u"], "weight": ib["weight"]}
                for ib in b0["internal_bonds"]
            ],
            "loop_pauli_local": b0["loop_pauli_local"],
            "loop_pauli_word": b0["loop_pauli_word"],
            "loop_sign": b0["loop_sign"],
            "loop_target": b0["loop_target"],
            "core_qubits": b0["core_qubits"],
            "divertor_qubits": b0["divertor_qubits"],
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

    print("  candidate: %s   preset: %s   shells: %d"
          % (manifest["candidate"], manifest["preset"], manifest["shells"]))
    print("  blocks: %d x %d sites = %d sites"
          % (manifest["n_blocks"], manifest["block_size"], manifest["n_sites"]))
    print("  bonds: %d total = %d internal + %d K4 junction + %d toroidal seam"
          % (tot_bonds, n_int, n_k4, n_tor))
    for bt in manifest["block_types"]:
        print("  block type %d: %2d-loop (%s, target %+d) + %2d tree | %2d internal bonds | core=%s div=%s"
              % (bt["type_id"], bt["loop_len"], bt["loop_pauli_word"],
                 int(bt["loop_target"]), bt["tree_len"], bt["n_internal_bonds"],
                 bt["core_qubits"], bt["divertor_qubits"]))
    c = manifest["coils"]
    print("  coil windings: %d non-planar | %d planar | %d trim | %d divertor control"
          % (len(c["nonplanar"]), len(c["planar"]), len(c["trim"]), len(c["control"])))
    ok = (tot_bonds == manifest["n_bonds"]
          and manifest["n_blocks"] == 40
          and manifest["n_sites"] == 1080)
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
