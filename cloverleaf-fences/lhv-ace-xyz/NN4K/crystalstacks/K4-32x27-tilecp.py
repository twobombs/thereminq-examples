# -*- coding: us-ascii -*-
# tilecp.py -- Rev 2.
#
# Can the L=6 hyperoctagon lattice be partitioned into 32 racetrack
# blocks of 27 sites, each a 10-loop with a 17-site tree hanging off it?
#
# The counting says maybe. 32 x 27 = 864 = every site, and
# 32 x 27 internal + (32 x 27)/2 shared = 864 + 432 = 1296 = every
# bond. Both close exactly. That is a necessary condition and not a
# construction, which is what this script is for.
#
# WHAT CHANGED IN REV 2
# =====================================================================
# Rev 1 asked a yes/no question with an all-or-nothing model and, after
# 900 seconds, had returned neither answer. Six changes, in rough order
# of how much they matter.
#
#   1. MAXIMISE, DO NOT DECIDE. Sites may now be left unassigned and
#      the objective is the number of complete blocks. A timeout now
#      yields a lower bound (blocks actually built, and verified) AND
#      an upper bound (CP-SAT's objective bound) instead of nothing.
#      Optimum 32 proves the tiling; a proven optimum below 32
#      disproves it; anything else brackets it. Rev 1 could only
#      report the middle case as silence, which is what it did.
#
#   2. AN INDEPENDENT VERIFIER. verify_blocks() re-derives everything
#      from the lattice: 27 sites, pairwise disjoint, exactly 27
#      induced edges, connected, exactly one cycle, that cycle of
#      length 10 and equal to a genuine elementary loop. It shares no
#      code with the model. A model this fiddly should not be trusted
#      on its own say-so.
#
#   3. A SELF-TEST WITH A KNOWN ANSWER. --selftest runs the identical
#      encoding on the induced subgraph of one known-good block with
#      K=1, where the only possible answer is that block. L=6 is the
#      smallest lattice whose site count 27 divides, so there is no
#      natural small instance; this manufactures one. Run it before
#      trusting any long run.
#
#   4. FULL REIFICATION OF ok[e]. Rev 1 constrained only
#      ok => justified and leaned on a global count for the converse.
#      Correct, but it propagates badly and it is exactly the kind of
#      cleverness that hides an off-by-one. Now an iff, both ways.
#
#   5. A COMPLETE WARM START, AND CHECKPOINTING. --hint seeds the
#      search from a greedy packing (25-28 blocks unaided). The hint
#      sets EVERY variable -- loop choice, parent pointers, depths,
#      edge justification -- not just the block labels: a partial hint
#      left the solver reconstructing the rest and it reported no
#      feasible solution at all inside 200s. A callback verifies and
#      writes every improving solution to disk, so a long run is
#      informative while it runs rather than only at the end.
#
#   6. A LOWER BOUND, AND DEAD WEIGHT REMOVED.
#      --lb pins the objective from below, by default at the greedy
#      hint size, so the search hunts for an improvement rather than
#      rediscovering 28 blocks from nothing. --lb 32 turns the whole
#      thing back into a decision problem, and there INFEASIBLE is a
#      proof that no tiling exists -- probably the mode you want for a
#      long run.
#      Rev 1's negative channelling, blk[v] != k enforced on
#      y[v][k].Not(), was 27648 constraints implied by ExactlyOne(y[v])
#      plus the positive direction; it is gone. Redundant-but-
#      propagating constraints replace it (loop sites per block,
#      internal edges), behind --no-redundant so you can measure
#      whether they earn their keep.
#
# REV 2.1 -- CRASH FIX
# =====================================================================
# Rev 2 set solver.parameters.repair_hint = True. On one core CP-SAT
# clamps its portfolio and never spawns the hint-repair subsolver, so
# the parameter was never exercised; on 48 workers it is spawned, asks
# for a fixed-search heuristic, finds none because the model declares
# no decision strategy, and aborts:
#
#     Check failed: heuristics.fixed_search != nullptr
#
# THE NEXT ENCODING QUESTION: --no-redundant
# =====================================================================
# With symmetry settled, the redundant constraints are the remaining
# unmeasured choice. They are not small: dropping them takes the model
# from 63,969 variables to 36,321, because the ly[v][k] reification is
# 864 x 32 booleans on its own. Presolve gets cheaper too -- probing
# runs in about 0.8s against 1.3s.
#
# Cheaper is not the same as better: the whole point of a redundant
# constraint is to propagate, and fewer variables that propagate worse
# can easily lose. This has NOT been measured to completion, which is
# why the launcher now spends its second proof slot on it rather than
# on the symmetry question that is already answered.
#
# REV 2.7 -- THE SYMMETRY BREAKING WAS HARMFUL; NOW OFF BY DEFAULT
# =====================================================================
# A 96-thread run put both encodings side by side for an hour, 40
# workers each, identical model otherwise. The result is not close.
#
#                        with my symmetry    --no-symmetry
#   [Probe] first round         29.5s              2.2s
#   variables probed           14,267           175,254
#   search began at            100.3s             14.9s
#   subtrees closed in 1h           0        2 (of 49)
#
# The cause is the value-precedence encoding itself. It builds an
# 864-long chain of AddMaxEquality -- mx[v] = max(mx[v-1], cc[v]) --
# and that sequential dependency is poison for probing: twelve times
# FEWER variables probed in thirteen times MORE time, about 440x worse
# per probe. Presolve then eats a hundred seconds before search starts.
#
# And it was never needed. CP-SAT finds the symmetry by itself -- 4
# generators, orbits of size 216 -- and applies orbit-based breaking
# during presolve. My constraints did not add information, they only
# obstructed the machinery that would have found it anyway.
#
# So --symmetry is now opt-in and off by default. Only the version
# without it made any proof progress at all: closed:1/47 at 454s and
# closed:2/49 at 680s, against zero subtrees closed by the other in a
# full hour.
#
# REV 2.6 -- SAFE TO RUN CONCURRENTLY
# =====================================================================
# The lattice cache was a single fixed path shared by every process and
# every --cycles setting. Two hazards, both real once you run the proof
# and a search side by side:
#   - different --cycles values thrash: each run finds the other's key,
#     rebuilds, and overwrites, forever.
#   - a reader can catch a half-written pickle mid-dump.
# Now one cache file per key, written to a pid-suffixed temp and moved
# into place with os.replace, which is atomic. Readers see the old file
# or the new one, never a partial. Any number of processes may share a
# machine.
#
# Still your job: give every concurrent run a distinct --out. They do
# not coordinate, and the default paths collide.
#
# REV 2.5 -- DO NOT HINT BELOW THE LOWER BOUND
# =====================================================================
# Running --lb 32 with the built-in 31-block packing produced
#
#     The solution hint is complete, but it is infeasible!
#
# which is exactly right: nb >= 32 and a 31-block hint cannot both
# hold. The hint is now dropped whenever it falls below the lower
# bound, with a printed reason. It only ever cost time and bias.
#
# Two other things that 12-hour run made visible, neither a bug:
#   - presolve takes ~260s, of which ~240s is three [Probe] rounds at
#     80s each. Irrelevant against a 12-hour budget; if you care,
#     cp_model_probing_level can be lowered.
#   - with lb == K the objective is pinned at [32,32], so this is a
#     pure SAT/UNSAT decision. There will be no #Bound movement to
#     watch: the only progress signal is #Model variable shaving, and
#     the answer arrives all at once or not at all.
#
# REV 2.4 -- THE PACKING IS NOW IN THE FILE
# =====================================================================
# The best construction so far, 31 of 32 blocks, is embedded as
# BUILTIN_PACKINGS and used automatically when --start finds nothing.
# One file, no companion JSON, and a run begins one block from the
# target rather than four.
#
# Site indices only mean anything relative to how the engine numbers
# sites, so the constant is VERIFIED against the freshly built lattice
# before use and declined with a printed reason if it does not hold. An
# embedded constant that silently disagreed with the lattice would be
# worse than no constant at all. --no-builtin skips it.
#
# Precedence: --start file, then the built-in, then greedy.

# REV 2.3 -- LARGE NEIGHBOURHOOD SEARCH
# =====================================================================
# The global model is the wrong instrument. Given 864 sites, 32
# interchangeable labels and a complete verified 28-block warm start,
# 48 workers spent 610 seconds without ever improving on the packing
# they were handed. But the leftover from a 28-block packing is 108
# sites -- exactly four blocks -- so the real question is local: can
# four more blocks be made to fit if a few neighbours may move?
#
# --lns freezes most of the solution, tears out a few blocks, and
# re-solves the induced subgraph of whatever is now free. That
# subproblem is ~200 sites and ~8 labels, and it closes to OPTIMAL in
# seconds. Sound, not a relaxation: a block's validity depends only on
# its own 27 sites, their induced edges and their cycle, so adjacency
# to a frozen block outside the region cannot affect it.
#
# 28 -> 31 blocks in 215s on 8 cores, against 28 in 610s on 48.
#
# TUNE IT BY WATCHING THE STATUS COLUMN, NOT THE REGION SIZE.
# A bigger neighbourhood is not a better one:
#
#   --lns-destroy 4   region 162-216   mostly OPTIMAL   28 -> 31
#   --lns-destroy 6   region 270       UNKNOWN always   28 -> 28
#
# At destroy 6 the subproblem stops being solvable inside the
# per-iteration budget, every iteration returns nothing, and the search
# never moves at all. A small neighbourhood solved to proven optimality
# beats a large one not solved. If you raise --lns-destroy, raise
# --lns-seconds with it and check the status column really does say
# OPTIMAL.
#
# LNS constructs; it cannot refute. Reaching 31 says nothing about
# whether 32 exists. Only --lb 32 returning INFEASIBLE proves
# impossibility, and that is the global model that stalls.
#
#   - repair_hint is now OFF by default, behind --repair-hint. That is
#     the fix. The complete warm start added in Rev 2 is what actually
#     mattered; repair_hint was belt-and-braces and it cost a core
#     dump. Passing --repair-hint still aborts and now warns you so.
#   - AddDecisionStrategy is declared as well. This was expected to be
#     an independent fix and IT IS NOT: with the strategy declared,
#     --repair-hint still aborts identically on 9.15.6755 at 8
#     workers. It is kept because branching on blk in site order,
#     smallest label first, matches the order the value-precedence
#     symmetry breaking wants anyway. --no-strategy disables it.
#
# REV 2.2 -- THE HINT WAS NEVER BEING ACCEPTED
# =====================================================================
# On 48 workers, 610 seconds, with a verified 28-block warm start, the
# solver reported best found 0. It had been discarding the hint every
# time.
#
# The value-precedence symmetry breaking requires block labels to
# appear in site order: the first assigned site must carry label 0, the
# next new label 1, and so on. The greedy packing numbers blocks in the
# order it happened to find them. Site 0 landed in greedy block 6, the
# precedence constraint rejected it, and the whole hint was infeasible.
# add_hint now sorts blocks by their smallest site before assigning
# labels. Fixing blk to the hint goes INFEASIBLE -> OPTIMAL.
#
# Worth recording how this got past Rev 2.1: the check that "proved"
# the encoding accepts a greedy packing was built with symmetry=False.
# It disabled the very constraint that made the hint infeasible, so it
# passed for the wrong reason and I reported the encoding as validated.
# A validation that turns off part of the model validates part of the
# model. hint_is_feasible() now runs against the REAL model before
# every solve, and a rejected hint is reported and dropped rather than
# silently wasting the run.
#
# A SECOND BUG, FOUND WHILE TESTING THE FIRST
# =====================================================================
# Rev 2 read solver.BestObjectiveBound() unconditionally. On a run that
# timed out at UNKNOWN having found nothing, that returned 0 and the
# verdict logic duly printed
#
#     NO TILING EXISTS: proven upper bound 0 < 32
#
# which is a false negative on a question this script exists to answer.
# The bound is only meaningful once the solver has proven something, so
# every verdict is now gated on the status: OPTIMAL or INFEASIBLE may
# conclude, FEASIBLE may claim only what it has built, and UNKNOWN
# concludes nothing at all.
#
# The ortools version is printed at startup; please report it with any
# further crash. 9.15.6755 reproduces the repair_hint abort exactly.
#
# A NOTE ON HARDWARE
# =====================================================================
# Rev 1's inconclusive 900-second run was made on ONE core. This model
# has roughly 10^5 booleans and CP-SAT parallelises well; --workers now
# defaults to the machine's core count. Do not read anything into a
# single-core timeout.
#
# STANDING CAVEATS
# =====================================================================
#   - A block is REQUIRED to contain a full elementary 10-loop. The
#     lattice has girth 10, but a unicyclic 27-site subgraph could
#     carry a 12- or 14-cycle instead. Those are excluded by fiat here
#     because the racetrack is the point. For the weaker question --
#     can it be tiled by unicyclic 27-blocks at all -- drop the loop
#     machinery and constrain induced edges to 27.
#   - Connectivity is not asserted, it is forced. Every non-loop site
#     carries a parent at strictly smaller depth, so its chain
#     terminates on a loop site of its own block. Cheaper and safer
#     than any flow or cut encoding.
#   - Symmetry across the K interchangeable labels is broken by value
#     precedence over sites. That is the standard linear-size
#     encoding, not necessarily the best one here; --no-symmetry turns
#     it off for comparison.
#   - Blocks are required to contain a loop, so a proven optimum below
#     K disproves THIS tiling, not every 27-site tiling.
#
# Requires ortools. The lattice comes from the K4 engine.

import argparse
import collections
import importlib.util
import json
import os
import pickle
import sys
import time

from ortools.sat.python import cp_model

ELL = 10
DEFAULT_B = 27
CACHE = "/tmp/k4_lattice_cache.pkl"


# =====================================================================
# PART 1 -- LATTICE
# =====================================================================

def find_engine(explicit):
    """The lattice engine, located by capability rather than filename.

    These files get renamed; hard-coding one name means the default
    invocation breaks the moment the directory convention changes.
    """
    import glob
    cands = [explicit] if explicit else []
    if not explicit:
        here = os.path.dirname(os.path.abspath(__file__))
        me = os.path.abspath(__file__)
        for pat in ("*Kitaev-single*.py", "*Kitaev*single*.py",
                    "*[CK]rystalstacks*Kitaev*.py", "*rystalstacks*.py"):
            for d in (here, os.getcwd()):
                for p in sorted(glob.glob(os.path.join(d, pat))):
                    if os.path.abspath(p) != me and p not in cands:
                        cands.append(p)
    for p in cands:
        if not os.path.exists(p):
            continue
        try:
            spec = importlib.util.spec_from_file_location("k4engine", p)
            m = importlib.util.module_from_spec(spec)
            argv, sys.argv = sys.argv, ["k4engine"]
            try:
                spec.loader.exec_module(m)
            finally:
                sys.argv = argv
        except Exception:
            continue
        if hasattr(m, "srs_bonds") and hasattr(m, "elementary_loops"):
            if not explicit:
                print("found engine: %s" % os.path.basename(p))
            return p
    raise SystemExit(
        "could not find the K4 lattice engine.\n"
        "  it must define srs_bonds and elementary_loops\n"
        "  examined: %s\n"
        "  pass it explicitly with --engine <file>"
        % (", ".join(os.path.basename(c) for c in cands) or "nothing"))


def load_lattice(engine, L, cache=CACHE, cycles=(ELL,)):
    """(adj, loops, n) for the L^3 hyperoctagon torus. Enumerating the
    elementary 10-loops is the expensive step, so it is cached."""
    key = "%s-L%d-c%s" % (os.path.basename(engine), L,
                          "_".join(str(c) for c in cycles))
    # One cache FILE per key, not one file shared by every key. With a
    # single path, concurrent runs with different --cycles each find
    # the other's key and rebuild over it forever; and a reader can
    # catch a half-written pickle from a writer. Per-key paths plus an
    # atomic rename remove both. Safe to run many processes at once.
    if cache:
        safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in key)
        cache = os.path.join(os.path.dirname(cache) or ".",
                             "k4_lattice_%s.pkl" % safe)
    if cache and os.path.exists(cache):
        try:
            blob = pickle.load(open(cache, "rb"))
            if blob.get("key") == key:
                return blob["adj"], blob["loops"], blob["n"]
        except Exception:
            pass
    import importlib.util
    spec = importlib.util.spec_from_file_location("k4engine", engine)
    k4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k4)
    bonds, idx = k4.srs_bonds(L, 1)
    a = collections.defaultdict(list)
    for i, j, c in bonds:
        a[i].append(j)
        a[j].append(i)
    adj = {v: sorted(a[v]) for v in range(len(idx))}
    loops = []
    for c in cycles:
        uniq, _, _, _ = k4.elementary_loops(L, c)
        got = [tuple(p) for p in uniq.values()]
        print("  %d-cycles: %d" % (c, len(got)))
        loops.extend(got)
    if cache:
        tmp = "%s.%d.tmp" % (cache, os.getpid())
        try:
            with open(tmp, "wb") as fh:
                pickle.dump({"key": key, "adj": adj, "loops": loops,
                             "n": len(idx)}, fh)
            os.replace(tmp, cache)          # atomic; readers see one or
        except Exception:                   # the other, never a partial
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return adj, loops, len(idx)


def edge_index(adj):
    edges = sorted({(min(u, v), max(u, v)) for u in adj for v in adj[u]})
    return edges, {e: i for i, e in enumerate(edges)}


# =====================================================================
# PART 1b -- BUILT-IN PACKING
# =====================================================================
# The best construction reached so far: 31 of the 32 blocks, 837 of 864
# sites, one block short. Embedded so the script is self-contained and
# a run starts one block from the target instead of four.
#
# These are site indices into the lattice as this engine builds it, so
# they mean something only if your engine numbers sites the same way.
# builtin_packing() therefore VERIFIES them against the lattice it has
# just constructed and declines, with a reason, if they do not hold up.
# A stale or mis-indexed constant must not quietly poison a run.
# --no-builtin skips it, which is what you want when measuring what the
# search reaches unaided.
#
# It is a construction, not a proof. 31 says nothing about whether 32
# exists.

BUILTIN_PACKINGS = {
    # (L, B): blocks
    (6, 27): (
    (0,1,5,30,31,66,180,185,221,246,247,282,396,401,432,437,462,463,498,648,649,653,679,680,828,834,839),
    (2,157,182,183,186,187,188,218,248,336,367,397,398,404,433,434,470,588,614,618,619,650,651,805,831,835,836),
    (3,9,39,45,219,225,255,261,286,399,435,440,441,471,476,477,482,513,620,652,657,663,687,688,693,699,837),
    (6,8,13,19,192,193,216,224,229,235,402,403,408,409,438,439,444,445,451,629,654,656,661,662,667,840,841),
    (7,36,37,38,43,44,73,140,217,223,253,260,319,468,469,473,475,505,571,655,684,685,686,691,692,721,788),
    (10,11,15,41,42,46,47,220,222,226,227,252,257,406,442,443,447,474,478,479,504,509,658,659,664,690,695),
    (12,18,24,49,50,55,228,234,240,259,265,271,414,450,455,461,480,481,486,492,660,666,672,678,697,698,703),
    (14,20,51,81,87,117,123,230,231,267,297,302,303,333,339,452,483,519,549,554,555,585,668,669,735,766,771),
    (16,52,53,58,59,88,89,196,232,238,263,268,269,412,448,453,484,485,490,520,521,670,696,701,706,707,736),
    (17,22,23,167,191,197,202,203,208,233,383,407,413,418,419,449,454,628,634,635,665,671,845,846,850,851,856),
    (21,27,170,200,201,206,207,237,380,385,386,416,417,423,458,459,489,602,632,637,638,639,675,823,849,854,855),
    (25,26,57,63,204,205,236,242,243,273,279,420,421,456,457,488,494,495,525,636,641,673,674,705,711,852,859),
    (28,29,32,33,34,35,239,244,245,250,281,424,430,460,464,465,466,467,491,496,502,676,677,681,682,683,719),
    (40,72,77,102,107,132,138,256,287,317,318,323,348,354,472,508,539,570,606,689,694,720,725,750,755,786,787),
    (48,54,84,85,120,121,122,158,264,295,300,301,337,368,516,517,547,552,553,589,702,733,739,768,769,770,806),
    (56,61,62,67,68,86,93,103,241,266,272,277,278,283,487,493,499,518,524,530,535,704,709,710,715,716,741),
    (60,65,71,90,96,97,251,270,275,276,306,308,497,503,522,528,529,533,558,708,713,714,738,744,745,746,775),
    (64,70,94,95,101,126,131,249,274,280,305,311,347,501,526,527,532,557,562,563,712,718,742,743,749,774,779),
    (69,74,99,104,105,135,254,284,285,290,315,320,321,351,500,506,536,537,567,572,717,722,747,752,753,784,789),
    (75,80,111,112,116,152,178,291,296,327,328,332,358,511,512,543,548,579,584,610,728,729,734,759,760,765,801),
    (76,113,137,143,149,174,179,292,322,353,359,395,544,545,569,574,611,647,724,730,761,780,791,797,822,827,858),
    (78,79,109,110,114,115,151,258,289,294,330,331,355,510,541,542,546,582,583,727,732,757,758,762,763,799,800),
    (82,108,144,145,181,210,211,288,298,324,329,360,361,390,391,427,540,550,576,581,613,642,643,767,792,793,829),
    (83,119,125,155,161,262,293,299,304,335,341,365,371,514,515,551,556,587,592,593,617,623,726,731,773,803,809),
    (91,127,128,153,159,160,307,338,343,344,369,375,376,523,559,560,590,591,596,621,627,776,777,807,808,813,844),
    (100,106,136,141,172,173,177,310,316,352,357,388,389,568,573,603,604,609,640,748,754,785,790,820,821,826,857),
    (118,148,154,184,190,209,215,334,363,364,370,394,400,425,431,580,586,615,616,646,772,796,802,832,833,838,863),
    (124,129,130,134,165,166,171,309,340,345,346,350,377,381,382,387,561,566,597,598,633,778,783,814,815,819,825),
    (133,168,169,198,199,312,313,349,378,379,384,410,415,564,565,600,601,605,630,631,781,816,817,818,847,848,853),
    (146,147,176,212,213,325,326,362,392,393,422,428,429,577,578,607,608,644,645,764,794,795,824,830,860,861,862),
    (150,156,162,163,164,189,194,195,342,366,372,373,374,405,411,594,595,624,625,626,798,804,810,811,812,842,843),
    ),
}


def builtin_packing(adj, loops, L, B):
    """The embedded packing if it verifies against this lattice, else []."""
    blocks = BUILTIN_PACKINGS.get((L, B))
    if not blocks:
        return []
    flat = [v for b in blocks for v in b]
    if flat and max(flat) >= len(adj):
        print("built-in packing does not fit this lattice (site %d of %d)"
              " -- ignoring it." % (max(flat), len(adj)))
        return []
    blocks = [list(b) for b in blocks]
    ok, bad = verify_blocks(blocks, adj, loops, B)
    if not ok:
        print("built-in packing failed verification against this lattice"
              " -- ignoring it.")
        for b in bad[:2]:
            print("  ! %s" % b)
        return []
    print("built-in packing: %d blocks, verifier pass" % len(blocks))
    return blocks


# =====================================================================
# PART 2 -- INDEPENDENT VERIFIER
# =====================================================================
# Shares nothing with the model. If the encoding is wrong, this is what
# catches it.

def verify_blocks(blocks, adj, loops, B=DEFAULT_B, ell=None):
    """(ok, [problems]). Each block must be 27 sites, disjoint from the
    others, with exactly 27 induced edges, connected, and carrying one
    cycle which is a genuine elementary ell-loop."""
    bad, seen = [], {}
    loopset = {frozenset(p) for p in loops}
    lens = {len(p) for p in loops} if ell is None else {ell}
    for bi, blk in enumerate(blocks):
        S = set(blk)
        if len(S) != B:
            bad.append("block %d: %d sites, want %d" % (bi, len(S), B))
            continue
        for v in S:
            if v in seen:
                bad.append("site %d in blocks %d and %d" % (v, seen[v], bi))
            seen[v] = bi
        ie = [(u, w) for u in S for w in adj[u] if w in S and u < w]
        if len(ie) != B:
            bad.append("block %d: %d induced edges, want %d"
                       % (bi, len(ie), B))
            continue
        nb = collections.defaultdict(list)
        for u, w in ie:
            nb[u].append(w)
            nb[w].append(u)
        start = next(iter(S))
        comp, stack = {start}, [start]
        while stack:
            v = stack.pop()
            for w in nb[v]:
                if w not in comp:
                    comp.add(w)
                    stack.append(w)
        if len(comp) != B:
            bad.append("block %d: disconnected, %d of %d reachable"
                       % (bi, len(comp), B))
            continue
        # 27 vertices, 27 edges, connected -> exactly one cycle.
        # Peel leaves; what survives is that cycle.
        deg = {v: len(nb[v]) for v in S}
        rem = set(S)
        leaves = [v for v in rem if deg[v] == 1]
        while leaves:
            v = leaves.pop()
            if v not in rem:
                continue
            rem.discard(v)
            for w in nb[v]:
                if w in rem:
                    deg[w] -= 1
                    if deg[w] == 1:
                        leaves.append(w)
        if len(rem) not in lens:
            bad.append("block %d: cycle length %d, allowed %s"
                       % (bi, len(rem), sorted(lens)))
        elif frozenset(rem) not in loopset:
            bad.append("block %d: its %d-cycle is not an admissible loop"
                       % (bi, len(rem)))
    return (not bad), bad


# =====================================================================
# PART 3 -- GREEDY PACKING (hint source, and a baseline)
# =====================================================================

def greedy_blocks(adj, loops, B, rng=None, limit=None):
    import random
    rng = rng or random.Random(0)
    order = list(loops)
    rng.shuffle(order)
    free, out = set(adj), []
    for p in order:
        if limit and len(out) >= limit:
            break
        if not set(p) <= free:
            continue
        cur = set(p)
        while len(cur) < B:
            fr = [w for v in cur for w in adj[v]
                  if w in free and w not in cur
                  and sum(1 for x in adj[w] if x in cur) == 1]
            if not fr:
                break
            cur.add(rng.choice(sorted(fr)))
        if len(cur) == B:
            out.append(sorted(cur))
            free -= cur
    return out


def best_greedy(adj, loops, B, restarts, target):
    import random
    best = []
    for s in range(restarts):
        g = greedy_blocks(adj, loops, B, random.Random(s), limit=target)
        if len(g) > len(best):
            best = g
            if len(best) >= target:
                break
    return best


# =====================================================================
# PART 4 -- THE MODEL
# =====================================================================

def build(adj, loops, K, B, ell=None, symmetry=True, redundant=True,
          strategy=True):
    # `ell` is per-loop now: loops may have different lengths when
    # --cycles admits more than one. Anything that used a single
    # constant length reads len(p) instead.
    """Blocks 0..K-1 plus an 'unassigned' class K. Maximises the count
    of complete blocks. Returns (model, vars)."""
    n = len(adj)
    edges, eid = edge_index(adj)
    loops_at = collections.defaultdict(list)
    loops_with = collections.defaultdict(list)
    lmin = min(len(p) for p in loops)
    for i, p in enumerate(loops):
        for v in p:
            loops_at[v].append(i)
        for r in range(len(p)):
            a, b = p[r], p[(r + 1) % len(p)]
            loops_with[eid[(min(a, b), max(a, b))]].append(i)

    m = cp_model.CpModel()
    x = [m.NewBoolVar("x%d" % i) for i in range(len(loops))]
    used = [m.NewBoolVar("u%d" % k) for k in range(K)]
    blk = [m.NewIntVar(0, K, "b%d" % v) for v in range(n)]
    y = [[m.NewBoolVar("y%d_%d" % (v, k)) for k in range(K + 1)]
         for v in range(n)]
    un = [y[v][K] for v in range(n)]
    lp = [m.NewBoolVar("lp%d" % v) for v in range(n)]
    d = [m.NewIntVar(0, B - lmin, "d%d" % v) for v in range(n)]
    ok = [m.NewBoolVar("ok%d" % e) for e in range(len(edges))]
    par = {v: [m.NewBoolVar("p%d_%d" % (v, j)) for j in range(len(adj[v]))]
           for v in range(n)}
    nb = m.NewIntVar(0, K, "nblocks")

    for v in range(n):
        m.AddExactlyOne(y[v])
        for k in range(K + 1):
            # positive direction only; ExactlyOne supplies the converse
            m.Add(blk[v] == k).OnlyEnforceIf(y[v][k])
        m.AddAtMostOne([x[i] for i in loops_at[v]])
        m.Add(lp[v] == sum(x[i] for i in loops_at[v]))
        m.AddImplication(lp[v], un[v].Not())
        m.Add(d[v] == 0).OnlyEnforceIf(lp[v])
        m.Add(sum(par[v]) == 0).OnlyEnforceIf(lp[v])
        m.Add(sum(par[v]) == 0).OnlyEnforceIf(un[v])
        m.Add(sum(par[v]) == 1).OnlyEnforceIf([lp[v].Not(), un[v].Not()])
        for j, u in enumerate(adj[v]):
            m.Add(blk[v] == blk[u]).OnlyEnforceIf(par[v][j])
            m.Add(d[v] == d[u] + 1).OnlyEnforceIf(par[v][j])

    m.Add(nb == sum(used))
    m.Add(sum(x) == nb)
    for k in range(K):
        m.Add(sum(y[v][k] for v in range(n)) == B * used[k])
    for k in range(K - 1):
        m.Add(used[k] >= used[k + 1])

    for i, p in enumerate(loops):
        for r in range(len(p)):
            m.Add(blk[p[r]] == blk[p[(r + 1) % len(p)]]).OnlyEnforceIf(x[i])

    for e, (u, v) in enumerate(edges):
        ju, jv = adj[v].index(u), adj[u].index(v)
        just = [x[i] for i in loops_with[e]] + [par[v][ju], par[u][jv]]
        m.AddBoolOr([ok[e].Not()] + just)              # ok => justified
        for t in just:
            m.AddImplication(t, ok[e])                 # justified => ok
        # a chord may not join two sites of the same assigned block
        m.Add(blk[u] != blk[v]).OnlyEnforceIf([ok[e].Not(), un[u].Not()])

    aux = {}
    mixed = len({len(p) for p in loops}) > 1
    if redundant:
        # with mixed cycle lengths the loop-site count is no longer
        # ell*nb: it is the total length of whichever loops got picked
        m.Add(sum(lp) == sum(x[i] * len(loops[i]) for i in range(len(loops))))
        m.Add(sum(ok) == B * nb)
        ly = [[m.NewBoolVar("ly%d_%d" % (v, k)) for k in range(K)]
              for v in range(n)]
        for v in range(n):
            for k in range(K):
                m.AddImplication(ly[v][k], y[v][k])
                m.AddImplication(ly[v][k], lp[v])
                m.AddBoolOr([y[v][k].Not(), lp[v].Not(), ly[v][k]])
        if not mixed:
            for k in range(K):
                m.Add(sum(ly[v][k] for v in range(n))
                      == len(loops[0]) * used[k])
            aux["ly"] = ly

    if symmetry and K > 1:
        cc = [m.NewIntVar(0, K - 1, "c%d" % v) for v in range(n)]
        mx = [m.NewIntVar(0, K - 1, "m%d" % v) for v in range(n)]
        for v in range(n):
            m.Add(cc[v] == blk[v]).OnlyEnforceIf(un[v].Not())
            m.Add(cc[v] == 0).OnlyEnforceIf(un[v])
        m.Add(mx[0] == cc[0])
        m.Add(blk[0] == 0).OnlyEnforceIf(un[0].Not())
        for v in range(1, n):
            m.Add(blk[v] <= mx[v - 1] + 1).OnlyEnforceIf(un[v].Not())
            m.AddMaxEquality(mx[v], [mx[v - 1], cc[v]])
        aux["cc"], aux["mx"] = cc, mx

    if strategy:
        # Declaring this is what keeps heuristics.fixed_search non-null,
        # which is what Rev 2 tripped over on a 48-worker portfolio.
        # Smallest label first also matches the value-precedence order.
        m.AddDecisionStrategy(blk, cp_model.CHOOSE_FIRST,
                              cp_model.SELECT_MIN_VALUE)

    m.Maximize(nb)
    return m, {"x": x, "blk": blk, "nb": nb, "used": used, "y": y,
               "lp": lp, "d": d, "ok": ok, "par": par, "eid": eid,
               "loops": loops, "adj": adj, "ell": None, "aux": aux,
               "n": n, "K": K, "B": B, "edges": edges}


def block_structure(S, adj):
    """(cycle, parent, depth) for one block: peel leaves to expose the
    unique cycle, then root a BFS forest on it."""
    S = set(S)
    nb = {v: [w for w in adj[v] if w in S] for v in S}
    deg = {v: len(nb[v]) for v in S}
    rem = set(S)
    leaves = [v for v in rem if deg[v] == 1]
    while leaves:
        v = leaves.pop()
        if v not in rem:
            continue
        rem.discard(v)
        for w in nb[v]:
            if w in rem:
                deg[w] -= 1
                if deg[w] == 1:
                    leaves.append(w)
    parent, depth = {}, {v: 0 for v in rem}
    fr = sorted(rem)
    while fr:
        nx = []
        for v in fr:
            for w in nb[v]:
                if w not in depth:
                    depth[w] = depth[v] + 1
                    parent[w] = v
                    nx.append(w)
        fr = nx
    return rem, parent, depth


def hint_is_feasible(adj, loops, K, B, blocks, seconds=60.0, workers=1):
    """Hard-fix blk to `blocks` and ask whether the model admits it.

    Cheap, and it makes the Rev 2.2 class of bug impossible to ship
    again: a hint the model cannot accept is worse than no hint, since
    it looks like it is helping while the solver quietly discards it.
    """
    m, V = build(adj, loops, K, B)
    where = {}
    for k, S in enumerate(sorted(blocks[:K], key=min)):
        for v in S:
            where[v] = k
    for v in range(V["n"]):
        m.Add(V["blk"][v] == where.get(v, K))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = seconds
    s.parameters.num_search_workers = workers
    st = s.Solve(m)
    return st in (cp_model.OPTIMAL, cp_model.FEASIBLE), s.StatusName(st)


def add_hint(m, V, blocks):
    """A COMPLETE warm start. Rev 1 hinted blk alone and left every
    auxiliary variable -- loop choice, parent pointers, depths, edge
    justification -- for the solver to reconstruct, which on one core
    it never managed inside the budget. Hint all of it or none."""
    n, K = V["n"], V["K"]
    loop_id = {frozenset(p): i for i, p in enumerate(V["loops"])}
    # Labels MUST appear in site order or the value-precedence symmetry
    # breaking rejects the hint outright -- see the Rev 2.2 note.
    blocks = sorted(blocks[:K], key=min)
    where, isloop, par_of, dep = {}, set(), {}, {}
    chosen = set()
    for k, S in enumerate(blocks):
        cyc, parent, depth = block_structure(S, V["adj"])
        i = loop_id.get(frozenset(cyc))
        if i is None:
            return 0                       # not a racetrack block; skip hint
        chosen.add(i)
        for v in S:
            where[v] = k
        isloop |= set(cyc)
        par_of.update(parent)
        dep.update(depth)
    for i in range(len(V["loops"])):
        m.AddHint(V["x"][i], 1 if i in chosen else 0)
    for k in range(K):
        m.AddHint(V["used"][k], 1 if k < len(blocks) else 0)
    just = set()
    for v in range(n):
        k = where.get(v, K)
        m.AddHint(V["blk"][v], k)
        for kk in range(K + 1):
            m.AddHint(V["y"][v][kk], 1 if kk == k else 0)
        m.AddHint(V["lp"][v], 1 if v in isloop else 0)
        m.AddHint(V["d"][v], dep.get(v, 0))
        p = par_of.get(v)
        for j, u in enumerate(V["adj"][v]):
            hit = 1 if (p is not None and u == p) else 0
            m.AddHint(V["par"][v][j], hit)
            if hit:
                just.add(V["eid"][(min(u, v), max(u, v))])
    for i in chosen:
        p = V["loops"][i]
        for r in range(len(p)):
            a, b = p[r], p[(r + 1) % len(p)]
            just.add(V["eid"][(min(a, b), max(a, b))])
    for e in range(len(V["edges"])):
        m.AddHint(V["ok"][e], 1 if e in just else 0)
    m.AddHint(V["nb"], len(blocks))

    # The auxiliary variables need hinting as well. Leaving cc, mx and
    # ly unset made the hint incomplete, and CP-SAT then failed to
    # complete it: 48 workers for 610s reported no solution at all
    # while holding a perfectly good 28-block packing. Hint everything
    # or the hint does nothing.
    aux = V.get("aux", {})
    if "cc" in aux:
        run = 0
        for v in range(n):
            c = where.get(v, None)
            c = 0 if c is None else c
            run = max(run, c)
            m.AddHint(aux["cc"][v], c)
            m.AddHint(aux["mx"][v], run)
    if "ly" in aux:
        for v in range(n):
            k = where.get(v, K)
            for kk in range(K):
                m.AddHint(aux["ly"][v][kk],
                          1 if (kk == k and v in isloop) else 0)
    return len(blocks)


def extract(solver, V):
    part = collections.defaultdict(list)
    for v in range(V["n"]):
        k = solver.Value(V["blk"][v])
        if k < V["K"]:
            part[k].append(v)
    return [sorted(p) for _, p in sorted(part.items())]


class Progress(cp_model.CpSolverSolutionCallback):
    def __init__(self, V, adj, loops, out):
        super().__init__()
        self.V, self.adj, self.loops, self.out = V, adj, loops, out
        self.best = -1

    def on_solution_callback(self):
        k = int(self.Value(self.V["nb"]))
        if k <= self.best:
            return
        self.best = k
        blocks = extract(self, self.V)
        ok, bad = verify_blocks(blocks, self.adj, self.loops, self.V["B"])
        print("  [%8.1fs] %3d blocks, %4d/%d sites, verifier %s"
              % (self.WallTime(), k, k * self.V["B"], self.V["n"],
                 "pass" if ok else "FAIL: " + "; ".join(bad[:2])))
        sys.stdout.flush()
        if self.out:
            json.dump({"n_blocks": k, "verified": ok, "problems": bad,
                       "wall": self.WallTime(), "blocks": blocks},
                      open(self.out, "w"), indent=1)


# =====================================================================
# PART 4b -- LARGE NEIGHBOURHOOD SEARCH
# =====================================================================
# The global model hands CP-SAT 864 sites and 32 interchangeable
# labels, and on 48 workers it spent 610 seconds without improving on
# the packing it started from. That is the wrong shape of question.
#
# The leftover from a 28-block packing is 108 sites -- exactly four
# blocks' worth -- so what is actually being asked is local: can four
# more blocks be made to fit if a few neighbouring ones are allowed to
# move? So freeze most of the solution, tear out a handful of blocks,
# and re-solve the induced subgraph of what is now free. That
# subproblem has ~200 sites and ~8 labels instead of 864 and 32, and
# it is solved to OPTIMALITY in seconds rather than approached for ten
# minutes.
#
# This is sound because a block's validity depends only on its own
# sites: its 27 induced edges, its cycle, its tree. Adjacency to a
# frozen block outside the region is irrelevant. So the subproblem on
# the induced subgraph is exactly the right subproblem, not a
# relaxation of one.

def load_checkpoint(path, adj, loops, B):
    """A packing from a --out checkpoint, or [] with a printed reason.

    Never raises: a missing or unusable checkpoint is a reason to build
    a fresh greedy packing, not to abort the run.
    """
    if not os.path.exists(path):
        print("no checkpoint at %s -- building a greedy packing instead."
              " (--start resumes a file an earlier --out wrote; there is"
              " nothing to resume on a first run.)" % path)
        return []
    try:
        blob = json.load(open(path))
        blocks = [list(b) for b in blob["blocks"]]
        pool = [[list(b) for b in p] for p in blob.get("pool", [])]
    except Exception as e:
        print("could not read %s (%s) -- building a greedy packing"
              " instead." % (path, e))
        return []
    ok, bad = verify_blocks(blocks, adj, loops, B)
    print("resumed %s: %d blocks, verifier %s%s"
          % (path, len(blocks), "pass" if ok else "FAIL",
             ", pool of %d" % len(pool) if pool else ""))
    if not ok:
        for b in bad[:3]:
            print("  ! %s" % b)
        print("  checkpoint rejected -- building a greedy packing instead.")
        return []
    return blocks if not pool else (blocks, pool)


def lns(adj, loops, blocks, B, iters, destroy, seconds, workers,
        out=None, seed=0, target=None):
    import random
    rng = random.Random(seed)
    best = [list(b) for b in blocks]
    n = len(adj)
    print("LNS: %d iterations, tearing out %d blocks at a time,"
          " %.0fs each" % (iters, destroy, seconds))
    for it in range(iters):
        if target and len(best) >= target:
            break
        assigned = set().union(*best) if best else set()
        idx = list(range(len(best)))
        rng.shuffle(idx)
        D = set(idx[:min(destroy, len(best))])
        keep = [best[i] for i in idx[min(destroy, len(best)):]]
        region = (set(adj) - assigned) | set().union(
            *[set(best[i]) for i in D]) if D else set(adj) - assigned
        K2 = len(region) // B
        if K2 < 1:
            continue
        ren = {v: i for i, v in enumerate(sorted(region))}
        inv = {i: v for v, i in ren.items()}
        sadj = {ren[v]: sorted(ren[w] for w in adj[v] if w in region)
                for v in region}
        sloops = [tuple(ren[v] for v in p) for p in loops
                  if set(p) <= region]
        m, V = build(sadj, sloops, K2, B)
        m.Add(V["nb"] >= len(D))          # never regress
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = seconds
        s.parameters.num_search_workers = workers
        st = s.Solve(m)
        got = 0
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            new = [sorted(inv[i] for i in blk) for blk in extract(s, V)]
            ok, bad = verify_blocks(keep + new, adj, loops, B)
            got = len(new)
            if ok and len(new) > len(D):
                best = keep + new
                print("  iter %3d: region %3d sites / %2d labels ->"
                      " %d blocks (was %d), TOTAL %d  [%s]"
                      % (it, len(region), K2, got, len(D), len(best),
                         s.StatusName(st)))
                if out:
                    json.dump({"n_blocks": len(best), "verified": True,
                               "blocks": best}, open(out, "w"), indent=1)
                continue
            if not ok:
                print("  iter %3d: REJECTED, verifier: %s" % (it, bad[:1]))
        if it % 10 == 0:
            print("  iter %3d: region %3d / %2d labels -> %d, no gain"
                  " [%s]" % (it, len(region), K2, got, s.StatusName(st)))
        sys.stdout.flush()
    return best


# =====================================================================
# PART 5 -- SELF-TEST
# =====================================================================

def selftest(adj, loops, B):
    """Run the identical encoding on the induced subgraph of one known
    block with K=1. The only valid answer is that block."""
    print("=" * 70)
    print("SELF-TEST: one known block, K=1, induced subgraph")
    print("=" * 70)
    g = greedy_blocks(adj, loops, B, limit=1)
    if not g:
        print("  no seed block available -- inconclusive")
        return False
    S = set(g[0])
    ren = {v: i for i, v in enumerate(sorted(S))}
    sadj = {ren[v]: sorted(ren[w] for w in adj[v] if w in S) for v in S}
    sloops = [tuple(ren[v] for v in p) for p in loops if set(p) <= S]
    print("  subgraph: %d sites, %d edges, %d elementary %d-loops"
          % (len(sadj), sum(len(a) for a in sadj.values()) // 2,
             len(sloops), ELL))
    m, V = build(sadj, sloops, 1, B)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = 120.0
    s.parameters.num_search_workers = max(1, os.cpu_count() or 1)
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("  status %s -- ENCODING IS BROKEN, this must be solvable"
              % s.StatusName(st))
        return False
    got = extract(s, V)
    ok, bad = verify_blocks(got, sadj, sloops, B)
    hit = len(got) == 1 and set(got[0]) == set(range(len(sadj)))
    print("  status %s, blocks %d, verifier %s, recovers the block: %s"
          % (s.StatusName(st), len(got), "pass" if ok else "FAIL",
             "yes" if hit else "no"))
    for b in bad[:3]:
        print("    ! %s" % b)
    return ok and hit


# =====================================================================
# PART 6 -- CLI
# =====================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Racetrack-block tiling of the hyperoctagon lattice")
    ap.add_argument("--engine", default=None,
                    help="path to the K4 engine. Found automatically by "
                         "the symbols it defines if not given")
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    ap.add_argument("--cycles", default=str(ELL),
                    help="comma-separated cycle lengths a block may "
                         "carry. The L=6 lattice has 1296 10-cycles, "
                         "none at 11/12/13/15, 1296 at 14 and 9072 at "
                         "16, so '10,14,16' is the meaningful relaxation")
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = every core on the machine")
    ap.add_argument("--out", default="tiling.json")
    ap.add_argument("--hint", type=int, default=200, metavar="RESTARTS",
                    help="greedy restarts for the warm start; 0 disables")
    ap.add_argument("--warm", action="store_true",
                    help="build and cache the lattice, then exit. Run "
                         "this before launching concurrent jobs so they "
                         "do not each enumerate the same cycles")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--symmetry", action="store_true",
                    help="enable the hand-written value-precedence "
                         "symmetry breaking. OFF by default: it is "
                         "measurably harmful, see REV 2.7")
    ap.add_argument("--no-symmetry", action="store_true",
                    help="deprecated; symmetry breaking is off by default")
    ap.add_argument("--no-redundant", action="store_true")
    ap.add_argument("--lb", type=int, default=-1, metavar="N",
                    help="require at least N blocks. Default: the greedy "
                         "hint size. --lb 32 makes it a pure decision "
                         "problem, where INFEASIBLE disproves the tiling")
    ap.add_argument("--repair-hint", action="store_true",
                    help="enable CP-SAT hint repair. OFF by default: it "
                         "aborts on some versions when the model declares "
                         "no decision strategy")
    ap.add_argument("--no-builtin", action="store_true",
                    help="ignore the embedded 31-block packing and start "
                         "from a fresh greedy one")
    ap.add_argument("--start", metavar="FILE",
                    help="resume from a checkpoint written by --out "
                         "instead of building a greedy packing")
    ap.add_argument("--lns", type=int, default=0, metavar="ITERS",
                    help="large-neighbourhood search instead of the "
                         "global solve. Recommended: --lns 500")
    ap.add_argument("--lns-destroy", type=int, default=4,
                    help="blocks torn out per LNS iteration")
    ap.add_argument("--pool-starts", type=int, default=16,
                    help="how many pooled packings to try when the "
                         "checkpoint carries a pool")
    ap.add_argument("--lns-seconds", type=float, default=30.0,
                    help="solver budget per LNS subproblem")
    ap.add_argument("--no-hintcheck", action="store_true",
                    help="skip the hint feasibility probe")
    ap.add_argument("--no-strategy", action="store_true",
                    help="do not declare a decision strategy")
    ap.add_argument("--log", action="store_true", help="CP-SAT search log")
    ns = ap.parse_args(argv)
    workers = ns.workers or max(1, os.cpu_count() or 1)

    try:
        import ortools
        print("ortools %s, python %s, %d core%s"
              % (getattr(ortools, "__version__", "unknown"),
                 sys.version.split()[0], workers,
                 "" if workers == 1 else "s"))
    except Exception:
        pass
    cycles = tuple(int(c) for c in ns.cycles.split(","))
    adj, loops, n = load_lattice(find_engine(ns.engine), ns.L, cycles=cycles)
    edges, _ = edge_index(adj)
    print("lattice L=%d: %d sites, %d bonds, %d admissible cycles %s"
          % (ns.L, n, len(edges), len(loops), list(cycles)))
    if n % ns.B:
        print("%d does not divide %d -- no exact tiling is possible"
              % (ns.B, n))
        return 1
    K, B = n // ns.B, ns.B
    print("target: %d blocks x %d sites; %d internal + %d shared bonds"
          % (K, B, K * B, K * B // 2))
    print("")

    if ns.warm:
        print("lattice cached.")
        return 0

    if ns.selftest:
        return 0 if selftest(adj, loops, B) else 2

    hint, pool = [], []
    if ns.start:
        hint = load_checkpoint(ns.start, adj, loops, B)
        if isinstance(hint, tuple):
            hint, pool = hint
    if not hint and not ns.no_builtin:
        hint = builtin_packing(adj, loops, ns.L, B)
    if not hint and ns.hint:
        t = time.time()
        hint = best_greedy(adj, loops, B, ns.hint, K)
        ok, bad = verify_blocks(hint, adj, loops, B)
        print("greedy hint: %d blocks in %.1fs, verifier %s"
              % (len(hint), time.time() - t, "pass" if ok else "FAIL"))
        for b in bad[:3]:
            print("  ! %s" % b)
        if not ok:
            hint = []
    if hint and not ns.no_hintcheck:
        t = time.time()
        good, why = hint_is_feasible(adj, loops, K, B, hint,
                                     min(120.0, ns.seconds / 4), workers)
        print("hint feasibility: %s (%s) in %.1fs"
              % ("accepted" if good else "REJECTED", why, time.time() - t))
        if not good:
            print("  the model cannot represent this hint; dropping it."
                  " Re-run with --no-symmetry to see whether the"
                  " symmetry breaking is the cause.")
            hint = []

    if ns.lns:
        if not hint:
            print("LNS needs a starting packing, and both --start and"
                  " --hint came up empty. Re-run with --hint 200.")
            return 1
        t0 = time.time()
        if pool:
            # A pool of distinct good packings beats one deep start:
            # LNS holds most of the loop selection fixed, so its reach
            # is bounded by where it began. Spend the budget across
            # starts rather than all of it on one.
            starts = sorted(pool, key=lambda p: -len(p))[:ns.pool_starts]
            per = max(1, ns.lns // max(1, len(starts)))
            print("pool: %d starts, %d LNS iterations each"
                  % (len(starts), per))
            best = hint
            for si, st0 in enumerate(starts):
                r = lns(adj, loops, st0, B, per, ns.lns_destroy,
                        ns.lns_seconds, workers, None, seed=si, target=K)
                if len(r) > len(best):
                    best = r
                    json.dump({"n_blocks": len(best), "verified": True,
                               "blocks": best}, open(ns.out, "w"), indent=1)
                print("  start %d/%d: began %d, reached %d, best so far %d"
                      % (si + 1, len(starts), len(st0), len(r), len(best)))
                sys.stdout.flush()
                if len(best) >= K:
                    break
        else:
            best = lns(adj, loops, hint, B, ns.lns, ns.lns_destroy,
                       ns.lns_seconds, workers, ns.out, target=K)
        ok, bad = verify_blocks(best, adj, loops, B)
        # Always write the result, improved or not. Writing only on
        # improvement means a run that gains nothing leaves no file,
        # and the next --start or --fix then dies on a missing path.
        if ok and ns.out:
            json.dump({"n_blocks": len(best), "verified": True,
                       "blocks": best}, open(ns.out, "w"), indent=1)
        print("")
        print("LNS finished in %.1fs: %d blocks, verifier %s, %d sites left"
              % (time.time() - t0, len(best), "pass" if ok else "FAIL",
                 n - len(best) * B))
        for b in bad[:3]:
            print("  ! %s" % b)
        if len(best) >= K and ok:
            print("TILING EXISTS. %s holds the %d blocks." % (ns.out, K))
        else:
            print("Best construction %d of %d. LNS cannot prove"
                  " impossibility -- use --lb %d for that."
                  % (len(best), K, K))
        return 0

    t0 = time.time()
    m, V = build(adj, loops, K, B, None,
                 symmetry=ns.symmetry and not ns.no_symmetry,
                 redundant=not ns.no_redundant,
                 strategy=not ns.no_strategy)
    lb = ns.lb if ns.lb >= 0 else len(hint)
    if lb > 0:
        m.Add(V["nb"] >= lb)
        print("lower bound: requiring at least %d blocks" % lb)
    # A hint smaller than the lower bound is INFEASIBLE by
    # construction -- CP-SAT reports "the solution hint is complete,
    # but it is infeasible" and burns time trying to repair it, while
    # hint-guided subsolvers get anchored to a region that provably
    # holds no solution. In decision mode (lb == K) drop it.
    if hint and lb > len(hint):
        print("hint has %d blocks but the lower bound is %d: dropping it"
              " (a hint below the bound is infeasible by construction)"
              % (len(hint), lb))
        hint = []
    hinted = add_hint(m, V, hint) if hint else 0
    if hint and not hinted:
        print("  hint rejected: greedy blocks are not racetrack blocks")
    print("model built in %.1fs (hinted %d blocks); solving up to %.0fs"
          " on %d worker%s"
          % (time.time() - t0, hinted, ns.seconds, workers,
             "" if workers == 1 else "s"))

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = ns.seconds
    s.parameters.num_search_workers = workers
    s.parameters.log_search_progress = ns.log
    if ns.repair_hint:
        print("WARNING: --repair-hint aborts on ortools 9.15 with"
              " 'Check failed: heuristics.fixed_search != nullptr'."
              " Declaring a decision strategy does not prevent it.")
        s.parameters.repair_hint = True
    cb = Progress(V, adj, loops, ns.out)
    st = s.Solve(m, cb)

    best = cb.best if cb.best >= 0 else 0
    proven = st in (cp_model.OPTIMAL, cp_model.INFEASIBLE)
    bound = int(s.BestObjectiveBound()) if proven else None
    print("")
    print("status %s after %.1fs" % (s.StatusName(st), s.WallTime()))
    print("blocks: best found %d, upper bound %s, target %d"
          % (best, bound if proven else "not proven", K))

    # Only OPTIMAL and INFEASIBLE license a conclusion. Rev 2 read the
    # bound on any status and announced "no tiling exists" off the back
    # of a timeout that had proven nothing.
    if st == cp_model.INFEASIBLE:
        print("INFEASIBLE at lb=%d: at most %d racetrack blocks fit."
              % (lb, lb - 1))
        if lb >= K:
            print("NO TILING EXISTS. This is a proof.")
    elif st == cp_model.OPTIMAL:
        if best >= K:
            print("TILING EXISTS. %s holds the %d blocks." % (ns.out, K))
        else:
            print("NO TILING EXISTS: proven optimum %d < %d." % (best, K))
    elif st == cp_model.FEASIBLE:
        print("PARTIAL: %d verified blocks in %s, %d sites left over."
              % (best, ns.out, n - best * B))
        print("Nothing is proven about %d. Run longer." % K)
    else:
        print("OPEN: no conclusion. Best construction %d block%s%s."
              % (best, "" if best == 1 else "s",
                 " (from the greedy hint)" if best == 0 and hint else ""))
        print("The solver proved nothing in the time given -- this is"
              " not evidence either way. Run longer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
