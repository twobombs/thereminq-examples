# -*- coding: us-ascii -*-
# stellarator-equiv.py -- Rev 1.
#
# D5-equivariant partition of the 1080-site W7-X-like stellarator Kitaev
# lattice into 40 blocks of 27 sites = 10 half-period modules x 4 manifolds
# of 27 sites (one Cloverfield K4 cluster of 4 x 27 = 108 qubits per module).
#
# Modelled directly on NN4K/coils/K4-32x27-equiv.py.
#
# Because radial inter-shell wiring (U_LOCAL -> BETA_LOCAL), magnetic-axis
# caps (CORE_JOINTS), and island-divertor caps (DIVERTOR_JOINTS) are strictly
# intra-module, each half-period module m in 0..9 has an isomorphic 108-site
# induced subgraph. Partitioning the 108 module sites into 4 connected 27-site
# manifolds (Types 0, 1, 2, 3) yields:
#   - 4 canonical 27-site block types (Type 0..3), replicated across m=0..9
#     by D5 symmetry -> 40 blocks x 27 sites = 1080 sites total
#   - Every 27-site block is a single connected component
#   - Every 27-site block contains a canonical internal elementary cycle
#     (loop) plus spanning tree sites, enabling per-manifold FluxMonitor
#     stabiliser readout
#   - All 6 pairs of manifolds within each module share intra-module bonds,
#     forming a complete-graph K4 cluster per half-period module.
#
#   python3 stellarator-equiv.py --check
#   python3 stellarator-equiv.py --prune -o equiv.json

import argparse
import collections
import json
import sys

import numpy as np

import stellarator_kitaev as SK


def local_site_map(idx, nshell=5):
    """Map local module index loc in 0..107 to (role, shell, sub_index)
    and return per-module local-to-global table loc_to_glob[m][loc].

    Local site ordering (0..107) inside module m:
      0..19  : shell 0 (0..7 branch 'b', 8..19 connector 'c' 0..11)
      20..39 : shell 1
      40..59 : shell 2
      60..79 : shell 3
      80..99 : shell 4
      100..101: magnetic-axis core joints ('core', j=0..1)
      102..107: island-divertor SOL cap sites ('divertor', d=0..5)
    """
    loc_to_glob = [[0] * 108 for _ in range(SK.N_MODULES)]
    glob_to_mod_loc = {}
    for m in range(SK.N_MODULES):
        for s in range(nshell):
            base = 20 * s
            for v in range(8):
                g = idx[(s, "b", m, v)]
                loc_to_glob[m][base + v] = g
                glob_to_mod_loc[g] = (m, base + v)
            for e in range(12):
                g = idx[(s, "c", m, e)]
                loc_to_glob[m][base + 8 + e] = g
                glob_to_mod_loc[g] = (m, base + 8 + e)
        for j in range(2):
            g = idx[("core", "j", m, j)]
            loc_to_glob[m][100 + j] = g
            glob_to_mod_loc[g] = (m, 100 + j)
        for d in range(6):
            g = idx[("divertor", "d", m, d)]
            loc_to_glob[m][102 + d] = g
            glob_to_mod_loc[g] = (m, 102 + d)
    return loc_to_glob, glob_to_mod_loc


def module_adj_and_cycles(idx, bonds, meta):
    """Build the 108-vertex induced graph of module 0 and find elementary
    cycles wholly inside module 0."""
    loc_to_glob, glob_to_mod_loc = local_site_map(idx, meta["nshell"])
    g0_set = set(loc_to_glob[0])
    g_to_l = {g: l for l, g in enumerate(loc_to_glob[0])}

    adj = collections.defaultdict(set)
    mod_bonds = []
    for i, j, c in bonds:
        if i in g0_set and j in g0_set:
            li, lj = g_to_l[i], g_to_l[j]
            adj[li].add(lj)
            adj[lj].add(li)
            mod_bonds.append((li, lj, c))

    return adj, mod_bonds, loc_to_glob, glob_to_mod_loc


def shortest_internal_cycle(sub_nodes, adj):
    """Find the shortest elementary cycle (length >= 4) inside an induced
    subgraph `sub_nodes`. Prefer bulk cycles of length >= 8 when available."""
    sub_set = set(sub_nodes)
    sub_adj = {u: [w for w in adj[u] if w in sub_set] for u in sub_set}
    best_bulk = None
    best_any = None

    for u0 in sorted(sub_set):
        for v0 in sorted(sub_adj[u0]):
            if v0 <= u0:
                continue
            dist = {u0: 0}
            par = {u0: None}
            q = collections.deque([u0])
            while q:
                u = q.popleft()
                if u == v0:
                    break
                for w in sub_adj[u]:
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
                ell = len(cyc)
                if best_any is None or ell < len(best_any):
                    best_any = cyc
                if ell >= 8 and (best_bulk is None or ell < len(best_bulk)):
                    best_bulk = cyc
    return best_bulk if best_bulk is not None else best_any


def is_connected(nodes, adj):
    nodes_set = set(nodes)
    if not nodes_set:
        return True
    start = next(iter(nodes_set))
    vis = {start}
    q = collections.deque([start])
    while q:
        u = q.popleft()
        for w in adj[u]:
            if w in nodes_set and w not in vis:
                vis.add(w)
                q.append(w)
    return len(vis) == len(nodes_set)


def build_k4_module_partition(adj, rad_cycles_local, candidate="H", time_limit=3.0):
    """Solve for the exact partition of the 108 module sites into 4 connected
    27-site manifolds (Types 0..3) using OR-Tools CP-SAT such that:
      1. Every manifold has an 8-site bulk radial cycle (`loop_local`) + 19
         connected tree sites (`tree_local`) = 27 sites.
      2. Every manifold induces a single connected component.
      3. All 6 K4 inter-manifold pairs (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
         share at least 2 intra-module bonds.
      4. The total number of internal bonds across the 4 manifolds is maximised.
    """
    from ortools.sat.python import cp_model

    cycles8 = [c for c in rad_cycles_local if len(c) == 8]
    if len(cycles8) < 4:
        cycles8 = sorted(rad_cycles_local, key=len)

    disj = None
    for i in range(len(cycles8)):
        for j in range(i + 1, len(cycles8)):
            if set(cycles8[i]) & set(cycles8[j]):
                continue
            for k in range(j + 1, len(cycles8)):
                if (set(cycles8[i]) | set(cycles8[j])) & set(cycles8[k]):
                    continue
                for l in range(k + 1, len(cycles8)):
                    if (set(cycles8[i]) | set(cycles8[j]) | set(cycles8[k])) & set(cycles8[l]):
                        continue
                    disj = [cycles8[i], cycles8[j], cycles8[k], cycles8[l]]
                    break
                if disj:
                    break
            if disj:
                break
        if disj:
            break

    if disj is None:
        raise RuntimeError("Could not find 4 disjoint radial cycles in module 0")

    model = cp_model.CpModel()
    MAX_D = 19
    x = {(u, k): model.NewBoolVar("x_%d_%d" % (u, k))
         for u in range(108) for k in range(4)}
    depth = {(u, k, d): model.NewBoolVar("d_%d_%d_%d" % (u, k, d))
             for u in range(108) for k in range(4) for d in range(MAX_D)}

    seed_of = {}
    for k, cyc in enumerate(disj):
        for u in cyc:
            seed_of[u] = k

    for u in range(108):
        model.Add(sum(x[u, k] for k in range(4)) == 1)
        for k in range(4):
            model.Add(sum(depth[u, k, d] for d in range(MAX_D)) == x[u, k])
            if u in seed_of:
                if seed_of[u] == k:
                    model.Add(x[u, k] == 1)
                    model.Add(depth[u, k, 0] == 1)
                else:
                    model.Add(x[u, k] == 0)
            else:
                model.Add(depth[u, k, 0] == 0)
                for d in range(1, MAX_D):
                    model.AddBoolOr([depth[u, k, d].Not()] +
                                    [depth[w, k, d - 1] for w in adj[u]])

    for k in range(4):
        model.Add(sum(x[u, k] for u in range(108)) == 27)

    # All 6 K4 inter-manifold pairs share at least 2 bonds
    for a in range(4):
        for b in range(a + 1, 4):
            pair_edges = []
            for u in range(108):
                for w in adj[u]:
                    if u < w:
                        eb1 = model.NewBoolVar("e_%d_%d_%d_%d" % (u, w, a, b))
                        model.AddBoolAnd([x[u, a], x[w, b]]).OnlyEnforceIf(eb1)
                        eb2 = model.NewBoolVar("e_%d_%d_%d_%d" % (w, u, a, b))
                        model.AddBoolAnd([x[w, a], x[u, b]]).OnlyEnforceIf(eb2)
                        pair_edges.extend([eb1, eb2])
            model.Add(sum(pair_edges) >= 2)

    int_bonds = []
    for u in range(108):
        for w in adj[u]:
            if u < w:
                for k in range(4):
                    ib = model.NewBoolVar("ib_%d_%d_%d" % (u, w, k))
                    model.AddBoolAnd([x[u, k], x[w, k]]).OnlyEnforceIf(ib)
                    int_bonds.append(ib)
    model.Maximize(sum(int_bonds))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 4
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT failed to partition module 0: %s"
                           % solver.StatusName(status))

    block_types = []
    for k in range(4):
        cyc = [int(u) for u in disj[k]]
        cyc_set = set(cyc)
        nodes = [int(u) for u in range(108) if solver.Value(x[u, k])]
        # Order tree nodes by BFS distance from the 8-cycle
        tree = []
        vis = set(cyc)
        q = collections.deque(cyc)
        nodes_set = set(nodes)
        while q:
            u = q.popleft()
            for w in sorted(adj[u]):
                if w in nodes_set and w not in vis:
                    vis.add(w)
                    tree.append(int(w))
                    q.append(w)
        block_types.append({
            "type_id": k,
            "loop_local": cyc,
            "tree_local": tree,
            "sites_local": cyc + tree,
        })

    return block_types


def build_equiv(candidate="H", nshell=5, preset="standard"):
    """Build the complete 40-block x 27-site D5-equivariant partition."""
    idx, pos, bonds, loops, meta = SK.build(
        nshell=nshell, candidate=candidate, closed=False,
        core=True, divertor=True, preset=preset)
    adj, mod_bonds, loc_to_glob, glob_to_mod_loc = module_adj_and_cycles(idx, bonds, meta)
    g2l = {g: l for l, g in enumerate(loc_to_glob[0])}
    rad_cycles_local = [
        [g2l[u] for u in c]
        for c in meta["radial_loops"]
        if all(u in g2l for u in c)
    ]
    block_types = build_k4_module_partition(adj, rad_cycles_local, candidate=candidate)

    blocks = []
    for m in range(SK.N_MODULES):
        for bt in block_types:
            k = bt["type_id"]
            loop_g = [int(loc_to_glob[m][u]) for u in bt["loop_local"]]
            tree_g = [int(loc_to_glob[m][u]) for u in bt["tree_local"]]
            blocks.append({
                "block_id": 4 * m + k,
                "module": m,
                "period": m // 2,
                "parity": m % 2,
                "type": k,
                "loop": loop_g,
                "tree": tree_g,
            })

    return {
        "candidate": candidate,
        "shells": nshell,
        "preset": preset,
        "n_sites": len(idx),
        "n_blocks": len(blocks),
        "block_size": 27,
        "n_modules": SK.N_MODULES,
        "blocks_per_module": 4,
        "block_types": block_types,
        "blocks": blocks,
    }, (idx, pos, bonds, loops, meta)


def verify_equiv(equiv_data, idx, bonds, meta):
    """Gate 4a: Verify exact 40x27 partition, connectivity, internal loops,
    and K4 cluster structure."""
    blocks = equiv_data["blocks"]
    n_sites = len(idx)
    adj = SK.neighbours(n_sites, bonds)

    seen = collections.Counter()
    all_connected = True
    loop_lengths = collections.Counter()

    block_of = {}
    for b in blocks:
        bid = b["block_id"]
        sites = b["loop"] + b["tree"]
        if len(sites) != 27:
            raise AssertionError("Block %d has %d sites, expected 27" % (bid, len(sites)))
        for s in sites:
            seen[s] += 1
            block_of[s] = bid
        # Connectivity check on global graph
        s_set = set(sites)
        vis = {sites[0]}
        q = collections.deque([sites[0]])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w in s_set and w not in vis:
                    vis.add(w)
                    q.append(w)
        if len(vis) != 27:
            all_connected = False
        # Verify loop is a valid cycle in the global graph
        lp = b["loop"]
        for r in range(len(lp)):
            if lp[(r + 1) % len(lp)] not in adj[lp[r]]:
                raise AssertionError("Invalid cycle edge in block %d" % bid)
        loop_lengths[len(lp)] += 1

    exact_cover = (len(seen) == n_sites and set(seen.values()) == {1})

    # Count internal vs intra-module K4 vs inter-module toroidal seam bonds
    n_internal = 0
    n_k4_junction = 0
    n_tor_seam = 0
    k4_pairs_mod0 = set()
    for i, j, _ in bonds:
        bi, bj = block_of[i], block_of[j]
        mi, mj = bi // 4, bj // 4
        if bi == bj:
            n_internal += 1
        elif mi == mj:
            n_k4_junction += 1
            if mi == 0:
                ti, tj = bi % 4, bj % 4
                k4_pairs_mod0.add((min(ti, tj), max(ti, tj)))
        else:
            n_tor_seam += 1

    print("=" * 78)
    print("PHASE 4 GATE (EQUIV) -- D5-EQUIVARIANT 40 x 27 MANIFOLD PARTITION")
    print("=" * 78)
    print("  blocks: %d (10 modules x 4 manifolds/module)   sites/block: 27" % len(blocks))
    print("  exact partition of %d sites : %s" % (n_sites, exact_cover))
    print("  all 40 blocks connected       : %s" % all_connected)
    print("  canonical loop lengths        : %s" % dict(sorted(loop_lengths.items())))
    print("  K4 pairs active in module     : %d / 6  %s"
          % (len(k4_pairs_mod0), sorted(k4_pairs_mod0)))
    print("  bond split (total %d)       : %d internal | %d K4 intra-module | %d toroidal seam"
          % (len(bonds), n_internal, n_k4_junction, n_tor_seam))
    ok = exact_cover and all_connected and len(blocks) == 40
    print("  EQUIV GATE VERDICT            : %s" % ("PASS" if ok else "FAIL"))
    print("=" * 78)
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="D5-equivariant 40x27 block partitioner for the stellarator")
    ap.add_argument("--candidate", choices=("H", "P"), default="H")
    ap.add_argument("--shells", type=int, default=5)
    ap.add_argument("--preset", default="standard")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--prune", action="store_true",
                    help="compatibility flag with K4-32x27-equiv.py")
    ap.add_argument("-o", "--out", default="equiv.json")
    a = ap.parse_args()

    equiv_data, (idx, pos, bonds, loops, meta) = build_equiv(
        candidate=a.candidate, nshell=a.shells, preset=a.preset)
    ok = verify_equiv(equiv_data, idx, bonds, meta)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(equiv_data, fh, indent=2)
        print("  wrote %s" % a.out)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
