# -*- coding: us-ascii -*-
# k4_ocl_pack.py -- OpenCL candidate search for the 32 x 27 racetrack
# tiling of the L=6 hyperoctagon lattice.
#
# WHAT THIS IS AND IS NOT
# =====================================================================
# This does NOT replace tilecp.py. CP-SAT search is pointer-chasing,
# branch-divergent and sequential; it belongs on the EPYC. What moves
# to the GPU is the half that is embarrassingly parallel: sampling a
# set of 32 pairwise-disjoint 10-loops and greedily completing each one
# into a 27-site block. Every work-item is an independent candidate.
#
# The whole problem is tiny and cache-resident, which is why this is
# worth doing at all:
#
#   loop masks       1296 x 14 ulong   =  141 KB
#   loop site lists  1296 x 10 int     =   52 KB
#   adjacency         864 x  3 int     =   10 KB
#
# It fits in LDS/L2 with room to spare. No streaming, no allocation,
# fixed-size private state, pure bitwise work.
#
# WHY OPENCL AND NOT CUDA
# =====================================================================
# CUDA strands the AMD half of a mixed fleet. OpenCL runs on the V340s,
# the MI50 and the NVIDIA cards from one source tree, via rusticl or
# the vendor ICDs. This file is also validated against PoCL on CPU, so
# it can be checked without a GPU present.
#
# THE MEASUREMENT THAT MOTIVATES IT
# =====================================================================
# One CPU core does 203 candidates/sec. Over 20293 samples the scores
# fell out as 26 -> 7615, 27 -> 7391, 28 -> 2011, 29 -> 124, 30 -> 1:
# each extra block costs 16-120x more samples, so reaching 32 by
# sampling wants somewhere in 1e8 to 1e10 candidates. 48 cores give
# ~1e4/sec, which is 3 hours at the optimistic end and 12 days at the
# other. That span is exactly where a GPU changes the answer.
#
# Be clear-eyed though: uniform sampling is a WEAK strategy, and CPU
# LNS in tilecp.py already reaches 31. This is worth running because it
# explores loop SELECTION broadly, which LNS does badly -- it holds
# most of the selection fixed. The two are complements. Neither proves
# anything: only --lb 32 returning INFEASIBLE settles the question.
#
# COMPANION DISCOVERY
# =====================================================================
# The CP-SAT module and the lattice engine are located by the symbols
# they define, not by filename, and the globs cover the naming
# conventions these files actually get renamed into. Hard-coding
# "tilecp.py" broke the moment the file became K4-32x27-tilecp.py.
# --tilecp and --engine still override, and a failure lists what was
# looked for and what was examined instead of raising FileNotFoundError
# from inside the import machinery.
#
# READ THE host COLUMN BEFORE THEORISING
# =====================================================================
# Round times on a six-V340 host went erratic -- 1.2s rounds becoming
# 9s, 30s, then back -- and two plausible explanations both turned out
# to be wrong:
#
#   - "the round barrier makes big rounds worse": no. Reverting
#     --global-size from 4194304 to 1048576 changed nothing; the same
#     degradation appeared at both sizes.
#   - "the solvers are starving the host thread": no. Renicing 90
#     CP-SAT workers to +10 and giving the host six reserved threads
#     did not help either.
#
# So the round line now reports host time separately. prepare() for
# round r+1 runs between submitting r and collecting it, which means
# host cost lands INSIDE dt and masquerades as slow GPUs. If host
# approaches dt, the cards are waiting on Python; if host is near zero
# and dt is still large, they are not, and the variance is on the
# device side. Measure before theorising -- this file has now cost two
# wrong diagnoses that a single number would have settled.
#
# --precheck is off by default. It never reduced the zero rate (see
# feasible()), so it was paying host cost for nothing, though measured
# host time says it was not the cause of the stalls either.
#
# THE HOST WAS STARVING THE CARDS
# =====================================================================
# nvtop on six V340s showed 7-44 percent utilisation, memory clocks at
# 9-74 MHz and 3-6 W drawn of 110 W available. The cards were idling
# between bursts, and the reported candidates/sec -- measured over the
# kernel window only -- was hiding it.
#
# Two host-side causes, both here:
#
#   1. The score histogram was a Python loop over every candidate.
#      At 6.29M scores a round that is 1.8 seconds, against a 1.2
#      second kernel window: the host spent longer counting than the
#      GPUs spent computing. np.bincount does the same work in 0.044s,
#      41x faster.
#
#   2. Nothing was in flight during host phases. Round r+1 is now
#      prepared while round r is still running on the cards, and
#      submitted the moment r is collected, so the only GPU-idle
#      window is the collect-and-submit pair.
#
# Neither changes what is computed: replay still agrees work-item for
# work-item and the score distributions are unchanged.
#
# SIX DEVICES WERE RUNNING NEARLY IN SERIES
# =====================================================================
# Measured on a six-die V340 host:
#
#   one die   --devices 0:0     705,000 candidates/sec
#   six dies  --devices all   1,030,000 candidates/sec
#
# 1.46x for six times the hardware. Not the cards -- a single Vega10
# against the Vega20 Pro VII's 1.14M is exactly the ratio you would
# expect -- but this file.
#
# pyopencl enqueues lazily. Nothing forces a submit until something
# flushes the queue, and collect() calls finish() on each device in
# turn, so device k's kernel did not start until device k-1 had
# finished. Adding q.flush() at the end of enqueue() submits every
# device's work before any of them is waited on.
#
# The per-round host work was vectorised at the same time. Building the
# availability mask was 1296 Python set intersections PER DEVICE PER
# ROUND; with six devices and sub-second kernels that stops being free.
# It is one numpy AND-reduce against a precomputed incidence matrix now.
#
# MULTI-DEVICE  (--devices)
# =====================================================================
# Earlier revisions used exactly one device: devs[0], one context, one
# queue. On a host with a Vega and two NVIDIA cards that leaves most of
# the machine idle, and rusticl only enumerated the AMD card anyway.
#
# --devices takes 'auto' (first GPU, the old behaviour), 'all' (every
# GPU on every platform), or an explicit 'platform:device' list. Each
# device gets its own context, program and queue, because an OpenCL
# context cannot straddle two platforms and a mixed AMD/NVIDIA fleet is
# always two platforms. Rounds are enqueued on every device before any
# is collected, so they genuinely overlap rather than taking turns.
#
# Every device gets its own seed each round. replay() is a function of
# (gid, seed), so each work-item stays independently reproducible on
# the host and --validate checks all devices: the correctness story
# does not weaken as devices are added. Naming one device twice
# (--devices 0:0,0:0) is a legitimate way to exercise this path on a
# single-GPU box, and is how it was tested here.
#
# Work-group size is derived per device from its own local_mem_size, so
# a card with less LDS gets a smaller group rather than a build failure.
#
# Two things a six-device host made obvious:
#   - sel_out was written by every work-item on every candidate and
#     never read back. At --global-size 1048576 that is 134 MB of
#     pointless VRAM and 134 MB of pointless writes per round PER
#     DEVICE. replay() reconstructs the selection from (gid, seed), so
#     the buffer was always dead. Removed.
#   - in --fix mode every device used to sample the SAME torn region.
#     Six cards re-sampling one neighbourhood explores no more than one
#     card does; six cards on six neighbourhoods explores six times as
#     much. Tears are drawn per device now. With one device this is
#     bit-for-bit the old behaviour.
#
# GPU AS THE LNS REPAIR ENGINE  (--fix)
# =====================================================================
# LNS in tilecp.py stalled because --lns-destroy 8 returned UNKNOWN on
# every iteration: CP-SAT could not SOLVE a 243-site subproblem inside
# the per-iteration budget. Sampling does not have to solve it.
#
# --fix holds most blocks of a packing fixed and samples completions of
# the rest, which is destroy-and-repair with the repair done by brute
# force. Against the 31-block packing with --tear 6:
#
#   uniform sampling   42,000,000 candidates, never once reached 31
#   --fix repair          393,216 candidates, reached 31 eight times
#
# That is the difference between an instrument that plateaus at 30 and
# one that lives at 31. It has not produced a 32.
#
# Larger tears are fine here, unlike in CP-SAT LNS where 8 was already
# past the point of solvability. The limit is different: see feasible()
# on why 58 percent of work-items score zero in this mode.
#
# WHAT 42 MILLION SAMPLES SHOWED
# =====================================================================
# On a Radeon Pro VII the LDS and conflict-row changes below took this
# from 280k to 1.14M candidates/sec. Then 42 million candidates were
# drawn and every one of them scored at most 30.
#
# That matters more than the speed. Extrapolating the tail from 2.6M
# samples had suggested p(31) around 1.8e-7, i.e. roughly seven hits in
# 42M. Zero hits bounds p(31) below about 7e-8, so the tail falls
# FASTER than geometric and the earlier estimate of an hour to reach 32
# was wrong -- optimistic by at least a factor of a few, and plausibly
# by orders of magnitude.
#
# Uniform sampling appears to plateau at 30. CPU LNS in tilecp.py
# reaches 31 from a merely greedy 28, so past 30 sampling is the weaker
# instrument. What sampling is good for is BREADTH: it produces ~150
# distinct 30-block packings a second, and a diverse set of starts is
# worth more to LNS than one deep one, because LNS holds most of the
# loop selection fixed and so cannot travel far from where it began.
#
# Hence --keep: harvest a pool of distinct good packings rather than
# one winner per round, and hand the pool to tilecp.py --start, which
# runs LNS from each in turn. Note the pool EVICTS its weakest member
# rather than merely filling; a pool that fills with the first round's
# 29s and then ignores every later 30 is worse than keeping nothing.
#
# TUNING ON VEGA
# =====================================================================
# First run on a Radeon Pro VII (vega20, 60 CUs, rusticl) gave 280k
# candidates/sec against 16k on a single Xeon core. Only 17x for a
# cache-resident integer kernel on that card is poor, and the two
# suspects were both in this file rather than in the hardware:
#
#   - ~900 bytes of private arrays per work-item spilled to scratch,
#     so every cur[] bit test in the growth loop -- the hottest
#     operation there is -- became a global access. cur and freem now
#     live in LDS at 224 B/item, so a work-group of 64 uses 14 KB of
#     the 64 KB available.
#   - loop selection ANDed a candidate's 14-word mask against used[]
#     on every probe, streaming lmask from global thousands of times
#     per work-item. It now carries a 1296-bit availability set in
#     private memory and clears it with a precomputed conflict row:
#     32 row reads per candidate instead of MAXATT mask reads.
#
# The rewrite is behaviour-preserving, not merely equivalent-ish: the
# availability set is exactly the old disjointness test, both consume
# one RNG draw per probe, and the optimised kernel reproduces the same
# winning work-item at the same gid as the original.
#
# Worth sweeping --local (32, 64, 128) and --cap (48 is ample; the
# candidate list never exceeds about 44 entries). On CPU via PoCL this
# version is slightly SLOWER, which is expected -- LDS is just memory
# there and the conflict rows are extra work. The win is on GPU.
#
# CORRECTNESS
# =====================================================================
# The kernel's RNG is xorshift32 and the host reimplements the whole
# candidate procedure in Python against the same stream, so any
# work-item can be replayed exactly. --validate replays a sample and
# asserts the scores agree, which is what makes a kernel this fiddly
# trustable. Winning candidates are additionally checked by
# tilecp.verify_blocks before anything is written.
#
# Output is a checkpoint tilecp.py can resume from:
#     python3 k4_ocl_pack.py --out gpu.json
#     python3 tilecp.py --start gpu.json --lns 2000

import argparse
import collections
import importlib.util
import json
import os
import sys
import time

import numpy as np

ELL, B, K_TARGET = 10, 27, 32


KERNEL = r"""
// ---------------------------------------------------------------
// One work-item = one candidate tiling attempt.
// -D N -D W -D NL -D CW -D ELL -D B -D KT -D CAP -D MAXATT -D LS
//
// Two things dominate and both are addressed here:
//
//  1. cur[] and freem[] are the hot masks -- every degree test in the
//     growth loop touches cur. As private arrays they spilled to
//     scratch on Vega, turning each bit test into a global access.
//     They live in LDS now, indexed by local id.
//
//  2. Loop selection used to AND a candidate's 14-word mask against
//     used[] on every probe, streaming lmask from global memory
//     thousands of times per work-item. Instead we carry a 1296-bit
//     "still available" set in private memory and clear it with a
//     precomputed conflict row on each pick: 32 row reads per
//     candidate instead of MAXATT mask reads.
// ---------------------------------------------------------------
#define GET(m,v) ((m[(v)>>6] >> ((v)&63)) & 1UL)
#define SET(m,v) m[(v)>>6] |=  (1UL << ((v)&63))
#define CLR(m,v) m[(v)>>6] &= ~(1UL << ((v)&63))
#define LGET(m,b,v) ((m[(b)+((v)>>6)] >> ((v)&63)) & 1UL)
#define LSET(m,b,v) m[(b)+((v)>>6)] |=  (1UL << ((v)&63))
#define LCLR(m,b,v) m[(b)+((v)>>6)] &= ~(1UL << ((v)&63))

inline uint xs32(uint *s) {
    uint x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

__kernel __attribute__((reqd_work_group_size(LS, 1, 1)))
void search(__global const ulong *lmask,   // NL * W
            __global const int   *lsite,   // NL * ELL
            __global const int   *adjv,    // N * 3
            __global const ulong *conf,    // NL * CW, loop conflict rows
            __global const ulong *blocked, // W, sites held by fixed blocks
            __global const ulong *avail0,  // CW, loops still selectable
            const int nfix,                // fixed blocks, scored as-is
            const uint seed,
            __global int *score)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int base = lid * W;

    __local ulong lcur[LS * W];
    __local ulong lfree[LS * W];

    uint st = (uint)(gid * 2654435761u) ^ seed;
    st |= 1u;
    for (int i = 0; i < 8; i++) xs32(&st);

    ulong avail[CW], inc[W];
    int sel[KT], cand[CAP], added[B - ELL];
    int w, j, k = 0;

    // ---- pick the remaining pairwise-disjoint loops ---------------
    // With nfix > 0 the host has already cleared avail0 of every loop
    // that touches a fixed block, so sampling here can only ever
    // produce completions consistent with what is held fixed. This is
    // LNS repair by brute force: CP-SAT could not SOLVE a 243-site
    // subproblem inside a per-iteration budget, but it does not need
    // solving when it can be sampled a million times a second.
    int need = KT - nfix;
    for (w = 0; w < CW; w++) avail[w] = avail0[w];

    for (int a = 0; a < MAXATT && k < need; a++) {
        int r = (int)(xs32(&st) % (uint)NL);
        if (!GET(avail, r)) continue;
        sel[k++] = r;
        for (w = 0; w < CW; w++) avail[w] &= ~conf[r * CW + w];
    }
    if (k < need) { score[gid] = 0; return; }

    for (w = 0; w < W; w++) lfree[base + w] = ~blocked[w];
    for (j = 0; j < need; j++)
        for (w = 0; w < W; w++) lfree[base + w] &= ~lmask[sel[j] * W + w];

    // ---- grow each loop into a B-site block ----------------------
    int done = nfix;
    for (int b = 0; b < need; b++) {
        int i = sel[b];
        for (w = 0; w < W; w++) { lcur[base + w] = lmask[i * W + w]; inc[w] = 0UL; }
        int cnt = ELL, nc = 0, na = 0;

        for (int t = 0; t < ELL; t++) {
            int v = lsite[i * ELL + t];
            for (j = 0; j < 3; j++) {
                int u = adjv[v * 3 + j];
                if (LGET(lfree, base, u) && !LGET(lcur, base, u)
                        && !GET(inc, u) && nc < CAP) {
                    cand[nc++] = u; SET(inc, u);
                }
            }
        }

        while (cnt < B) {
            int vcount = 0, pick = -1;
            for (int c = 0; c < nc; c++) {
                int u = cand[c];
                if (u < 0) continue;
                if (!LGET(lfree, base, u) || LGET(lcur, base, u)) {
                    cand[c] = -1; continue;
                }
                int deg = 0;
                for (j = 0; j < 3; j++)
                    if (LGET(lcur, base, adjv[u * 3 + j])) deg++;
                if (deg > 1) { cand[c] = -1; continue; }
                if (deg != 1) continue;
                vcount++;
                if ((xs32(&st) % (uint)vcount) == 0u) pick = u;
            }
            if (pick < 0) break;
            LSET(lcur, base, pick); LCLR(lfree, base, pick);
            added[na++] = pick; cnt++;
            for (j = 0; j < 3; j++) {
                int u = adjv[pick * 3 + j];
                if (LGET(lfree, base, u) && !LGET(lcur, base, u)
                        && !GET(inc, u) && nc < CAP) {
                    cand[nc++] = u; SET(inc, u);
                }
            }
        }

        if (cnt == B) done++;
        else for (int t = 0; t < na; t++) LSET(lfree, base, added[t]);
    }
    score[gid] = done;
}
"""


# =====================================================================
# HOST-SIDE REPLAY -- the same procedure, same RNG stream
# =====================================================================

class XS32(object):
    """xorshift32, bit-identical to the kernel's."""

    def __init__(self, gid, seed):
        self.s = ((gid * 2654435761) ^ seed) & 0xFFFFFFFF
        self.s |= 1
        for _ in range(8):
            self.next()

    def next(self):
        x = self.s
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.s = x & 0xFFFFFFFF
        return self.s


def replay(gid, seed, loops, adj, n, maxatt, cap, conf,
           blocked=None, avail0=None, fixed=()):
    """Reproduce one work-item exactly. Returns (score, blocks)."""
    r = XS32(gid, seed)
    nl = len(loops)
    need = K_TARGET - len(fixed)
    avail = np.ones(nl, dtype=bool) if avail0 is None else avail0.copy()
    sel = []
    for _ in range(maxatt):
        if len(sel) >= need:
            break
        i = r.next() % nl
        if not avail[i]:
            continue
        sel.append(i)
        avail &= ~conf[i]
    if len(sel) < need:
        return 0, []
    free = np.ones(n, dtype=bool)
    if blocked is not None:
        free &= ~blocked
    for i in sel:
        free[list(loops[i])] = False
    blocks = [list(b) for b in fixed]
    for i in sel:
        cur = set(loops[i])
        inc = set()
        cand = []
        for v in loops[i]:
            for u in adj[v]:
                if free[u] and u not in cur and u not in inc and len(cand) < cap:
                    cand.append(u)
                    inc.add(u)
        added = []
        while len(cur) < B:
            vcount, pick = 0, -1
            for c in range(len(cand)):
                u = cand[c]
                if u < 0:
                    continue
                if not free[u] or u in cur:
                    cand[c] = -1
                    continue
                deg = sum(1 for x in adj[u] if x in cur)
                if deg > 1:
                    cand[c] = -1
                    continue
                if deg != 1:
                    continue
                vcount += 1
                if r.next() % vcount == 0:
                    pick = u
            if pick < 0:
                break
            cur.add(pick)
            free[pick] = False
            added.append(pick)
            for u in adj[pick]:
                if free[u] and u not in cur and u not in inc and len(cand) < cap:
                    cand.append(u)
                    inc.add(u)
        if len(cur) == B:
            blocks.append(sorted(cur))
        else:
            for v in added:
                free[v] = True
    return len(blocks), blocks


# =====================================================================
# DRIVER
# =====================================================================

def _try_import(path, name):
    """Import a file, with argv neutralised. None if it will not load."""
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        argv, sys.argv = sys.argv, [name]
        try:
            spec.loader.exec_module(m)
        finally:
            sys.argv = argv
        return m
    except Exception:
        return None


def find_companion(explicit, patterns, needs, label):
    """Locate a companion module by capability, not by filename.

    Filenames get renamed -- K4-32x27-tilecp.py, tileocl, whatever the
    directory convention is -- so match on the symbols the module has
    to provide and try the given globs in priority order. An explicit
    path always wins and is never second-guessed.
    """
    import glob
    if explicit and os.path.exists(explicit):
        m = _try_import(explicit, label)
        if m and all(hasattr(m, a) for a in needs):
            return m, explicit
        raise SystemExit("%s: %s does not provide %s"
                         % (label, explicit, ", ".join(needs)))
    here = os.path.dirname(os.path.abspath(__file__))
    me = os.path.abspath(__file__)
    seen, tried = set(), []
    for pat in patterns:
        for d in (here, os.getcwd()):
            for path in sorted(glob.glob(os.path.join(d, pat))):
                path = os.path.abspath(path)
                if path in seen or path == me:
                    continue
                seen.add(path)
                tried.append(os.path.basename(path))
                m = _try_import(path, label)
                if m and all(hasattr(m, a) for a in needs):
                    print("found %s: %s" % (label, os.path.basename(path)))
                    return m, path
    raise SystemExit(
        "could not find the %s module.\n"
        "  looked for: %s\n"
        "  needs to define: %s\n"
        "  examined: %s\n"
        "  pass it explicitly with --%s <file>"
        % (label, ", ".join(patterns), ", ".join(needs),
           ", ".join(tried) or "nothing", label))


def main(argv=None):
    ap = argparse.ArgumentParser(description="OpenCL racetrack packing search")
    ap.add_argument("--tilecp", default=None,
                    help="the CP-SAT module. Found automatically by the "
                         "symbols it defines if not given")
    ap.add_argument("--engine", default=None,
                    help="the K4 lattice engine. Also found automatically")
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--global-size", type=int, default=1 << 16)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="stop after this many seconds regardless of "
                         "--rounds. 0 = run all rounds. Lets the GPU job "
                         "share a wall-clock budget with the solvers")
    ap.add_argument("--maxatt", type=int, default=20000)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--local", type=int, default=64,
                    help="work-group size; LDS use is 2*local*W*8 bytes")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--devices", default="auto",
                    help="'auto' (first GPU), 'all' (every GPU across "
                         "every platform), or a list like '0:0,1:0' of "
                         "platform:device. Each device gets its own "
                         "context, so mixed AMD/NVIDIA fleets work")
    ap.add_argument("--platform", type=int, default=-1)
    ap.add_argument("--device", type=int, default=-1)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--fix", metavar="FILE",
                    help="hold most blocks of a packing fixed and sample "
                         "completions of the rest -- LNS repair, but by "
                         "brute force on the GPU instead of CP-SAT")
    ap.add_argument("--precheck", action="store_true",
                    help="reject tears that admit no completion before "
                         "launching. OFF by default: it never reduced "
                         "the zero rate and its failure path is a scan "
                         "of every available loop, retried up to 64 "
                         "times per device -- host cost that lands "
                         "inside the round and stalls the cards")
    ap.add_argument("--tear", type=int, default=6,
                    help="blocks torn out of --fix each round. Larger is "
                         "fine here: sampling does not have to SOLVE the "
                         "region, which is what capped CP-SAT at 4-5")
    ap.add_argument("--threshold", type=int, default=0,
                    help="keep every candidate scoring at least this. "
                         "0 = best seen minus one")
    ap.add_argument("--keep", type=int, default=64,
                    help="how many distinct good packings to collect")
    ap.add_argument("--validate", type=int, default=16,
                    help="work-items to replay on the host and check")
    ap.add_argument("--out", default="gpu_packing.json")
    ns = ap.parse_args(argv)

    import pyopencl as cl

    if ns.list_devices:
        for pi, p in enumerate(cl.get_platforms()):
            print("[%d] %s" % (pi, p.name))
            for di, d in enumerate(p.get_devices()):
                print("    [%d] %s -- %d CUs, %d KB local, wg %d"
                      % (di, d.name.strip(), d.max_compute_units,
                         d.local_mem_size // 1024, d.max_work_group_size))
        return 0

    t, tpath = find_companion(
        ns.tilecp, ["*tilecp*.py", "*tile*cp*.py", "*tiling*.py"],
        ["load_lattice", "verify_blocks", "build"], "tilecp")
    _, epath = find_companion(
        ns.engine, ["*Kitaev-single*.py", "*Kitaev*single*.py",
                    "*[CK]rystalstacks*Kitaev*.py", "*rystalstacks*.py"],
        ["srs_bonds", "elementary_loops"], "engine")
    adj, loops, n = t.load_lattice(epath, ns.L)
    nl, w = len(loops), (n + 63) // 64
    print("lattice %d sites, %d loops, %d words/mask" % (n, nl, w))

    lmask = np.zeros((nl, w), dtype=np.uint64)
    lsite = np.zeros((nl, ELL), dtype=np.int32)
    for i, p in enumerate(loops):
        for v in p:
            lmask[i, v >> 6] |= np.uint64(1) << np.uint64(v & 63)
        lsite[i] = p
    adjv = np.zeros((n, 3), dtype=np.int32)
    for v in range(n):
        adjv[v] = adj[v]
    # loop-loop conflict rows: bit j set iff loops i and j share a site
    cw = (nl + 63) // 64
    conf = np.zeros((nl, nl), dtype=bool)
    for i in range(nl):
        conf[i] = (lmask & lmask[i]).any(axis=1)
    confw = np.zeros((nl, cw), dtype=np.uint64)
    for i in range(nl):
        for j in np.nonzero(conf[i])[0]:
            confw[i, j >> 6] |= np.uint64(1) << np.uint64(j & 63)
    print("buffers: masks %d KB, sites %d KB, adjacency %d KB,"
          " conflict rows %d KB"
          % (lmask.nbytes // 1024, lsite.nbytes // 1024, adjv.nbytes // 1024,
             confw.nbytes // 1024))

    base_blocks = []
    if ns.fix:
        if not os.path.exists(ns.fix):
            raise SystemExit(
                "no packing at %s.\n"
                "  --fix needs a checkpoint to hold blocks from. Make one"
                " with:\n    python3 %s --lns 1 --out %s"
                % (ns.fix, os.path.basename(tpath), ns.fix))
        try:
            blob = json.load(open(ns.fix))
            base_blocks = [list(b) for b in blob["blocks"]]
        except Exception as e:
            raise SystemExit("could not read %s (%s)" % (ns.fix, e))
        ok, bad = t.verify_blocks(base_blocks, adj, loops, B)
        print("fix source %s: %d blocks, verifier %s"
              % (ns.fix, len(base_blocks), "pass" if ok else "FAIL"))
        for b in bad[:2]:
            print("  ! %s" % b)
        if not ok:
            return 1

    plats = cl.get_platforms()

    def pick_devices():
        """Resolve --devices into a list of (platform, device) pairs.

        Different vendors live on different platforms and an OpenCL
        context cannot straddle two, so each device gets its own
        context, program and queue. Duplicates are allowed -- naming
        the same device twice is a legitimate way to test the
        multi-device path on a single-GPU box.
        """
        spec = ns.devices
        if ns.platform >= 0 or ns.device >= 0:
            p = plats[ns.platform if ns.platform >= 0 else 0]
            ds = p.get_devices()
            return [(p, ds[ns.device if ns.device >= 0 else 0])]
        if spec and spec not in ("auto", "all"):
            out = []
            for tok in spec.split(","):
                pi, _, di = tok.partition(":")
                p = plats[int(pi)]
                out.append((p, p.get_devices()[int(di or 0)]))
            return out
        gpus = [(p, d) for p in plats for d in p.get_devices(cl.device_type.GPU)]
        if spec == "all":
            return gpus or [(plats[0], plats[0].get_devices()[0])]
        return [gpus[0]] if gpus else [(plats[0], plats[0].get_devices()[0])]

    picked = pick_devices()
    if not picked:
        raise SystemExit("no OpenCL devices; try --list-devices")

    opts_common = ("-D N=%d -D W=%d -D NL=%d -D CW=%d -D ELL=%d -D B=%d "
                   "-D KT=%d -D CAP=%d -D MAXATT=%d"
                   % (n, w, nl, cw, ELL, B, K_TARGET, ns.cap, ns.maxatt))

    class Unit(object):
        """One device: its own context, queue, program and buffers."""

        def __init__(self, idx, plat, dev):
            self.idx, self.plat, self.dev = idx, plat, dev
            self.ctx = cl.Context([dev])
            self.q = cl.CommandQueue(self.ctx)
            self.lsz = ns.local
            if 2 * self.lsz * w * 8 > dev.local_mem_size:
                self.lsz = max(8, int(dev.local_mem_size / (2 * w * 8)) & ~7)
            self.G = max(self.lsz, (ns.global_size // self.lsz) * self.lsz)
            t0 = time.time()
            prg = cl.Program(self.ctx, KERNEL).build(
                options=opts_common + " -D LS=%d" % self.lsz)
            self.krn = cl.Kernel(prg, "search")
            f = cl.mem_flags
            self.lmask = cl.Buffer(self.ctx, f.READ_ONLY | f.COPY_HOST_PTR,
                                   hostbuf=lmask)
            self.lsite = cl.Buffer(self.ctx, f.READ_ONLY | f.COPY_HOST_PTR,
                                   hostbuf=lsite)
            self.adjb = cl.Buffer(self.ctx, f.READ_ONLY | f.COPY_HOST_PTR,
                                  hostbuf=adjv)
            self.conf = cl.Buffer(self.ctx, f.READ_ONLY | f.COPY_HOST_PTR,
                                  hostbuf=confw)
            self.score = np.zeros(self.G, dtype=np.int32)
            self.dscore = cl.Buffer(self.ctx, f.WRITE_ONLY, self.score.nbytes)
            print("  [%d] %s (%s), %d CUs, wg %d, %d work-items,"
                  " built in %.2fs"
                  % (idx, dev.name.strip(), plat.name.strip(),
                     dev.max_compute_units, self.lsz, self.G,
                     time.time() - t0))

        def enqueue(self, bl, av, nfix, seed):
            f = cl.mem_flags
            self.dbl = cl.Buffer(self.ctx, f.READ_ONLY | f.COPY_HOST_PTR,
                                 hostbuf=bl)
            self.dav = cl.Buffer(self.ctx, f.READ_ONLY | f.COPY_HOST_PTR,
                                 hostbuf=av)
            self.krn(self.q, (self.G,), (self.lsz,), self.lmask, self.lsite,
                     self.adjb, self.conf, self.dbl, self.dav,
                     np.int32(nfix), np.uint32(seed), self.dscore)
            # Submit NOW. Without this the enqueue sits in the queue
            # until something forces a flush, and since collect() calls
            # finish() on each device in turn, device k's kernel would
            # not start until device k-1 had finished: six dies running
            # nearly in series. One die measured 705k candidates/sec;
            # six measured 1.03M, i.e. 1.46x for 6x the hardware.
            self.q.flush()

        def collect(self):
            cl.enqueue_copy(self.q, self.score, self.dscore)
            self.q.finish()
            return self.score

    print("devices:")
    units = [Unit(i, p, d) for i, (p, d) in enumerate(picked)]
    total_items = sum(u.G for u in units)
    print("work-items per round: %d across %d device%s, %d CUs total"
          % (total_items, len(units), "" if len(units) == 1 else "s",
             sum(u.dev.max_compute_units for u in units)))

    ones = np.array([np.uint64(0xFFFFFFFFFFFFFFFF)] * cw, dtype=np.uint64)
    if nl & 63:
        ones[cw - 1] = np.uint64((1 << (nl & 63)) - 1)
    loopset = {frozenset(p): i for i, p in enumerate(loops)}
    # loop-by-site incidence, built once. The per-round availability
    # mask used to be 1296 Python set intersections PER DEVICE per
    # round; with six devices and sub-second kernels that host work
    # stops being free.
    LSITE = np.zeros((nl, n), dtype=bool)
    for i, p in enumerate(loops):
        LSITE[i, list(p)] = True

    def feasible(av_bool, need):
        """Can `need` disjoint loops even be drawn from what is left?

        Cheap insurance against a tear that leaves no room at all, and
        it costs microseconds. Be clear about what it does NOT fix: in
        --fix mode roughly 58 percent of work-items still score zero,
        and adding this precheck did not move that figure. Raising
        --maxatt eightfold did not move it either, so the failures are
        not probe exhaustion and not dead tears. They are random
        selection order failing in a tight region -- greedy from a good
        start finds `need` disjoint loops where a random order paints
        itself into a corner after four picks. Inherent to uniform
        probing; a biased sampler would be the fix, not a bigger
        budget.
        """
        avail = av_bool.copy()
        got = 0
        for i in np.nonzero(avail)[0]:
            if not avail[i]:
                continue
            got += 1
            if got >= need:
                return True
            avail &= ~conf[i]
        return False

    def make_round(rng, tries=64):
        """Choose which blocks to hold fixed, and derive the masks."""
        if not base_blocks:
            return np.zeros(w, dtype=np.uint64), ones.copy(), [], 0
        if not ns.precheck:
            return _make_round(rng)[:4]
        for _ in range(tries):
            r = _make_round(rng)
            if feasible(r[4], K_TARGET - r[3]):
                return r[:4]
        return r[:4]

    def _make_round(rng):
        idx = list(range(len(base_blocks)))
        rng.shuffle(idx)
        keep = [base_blocks[i] for i in idx[ns.tear:]]
        bl = np.zeros(w, dtype=np.uint64)
        held = set()
        for blk in keep:
            for v in blk:
                bl[v >> 6] |= np.uint64(1) << np.uint64(v & 63)
                held.add(v)
        held_mask = np.zeros(n, dtype=bool)
        if held:
            held_mask[list(held)] = True
        avb = ~(LSITE & held_mask).any(axis=1)
        packed = np.packbits(avb.astype(np.uint8), bitorder="little")
        buf = np.zeros(cw * 8, dtype=np.uint8)
        buf[:len(packed)] = packed
        av = buf.view(np.uint64)
        return bl, av, keep, len(keep), avb

    import random as _random
    rng = _random.Random(ns.seed)
    hist = collections.Counter()
    best, best_blocks = 0, []
    pool, pool_keys = [], set()
    total, elapsed = 0, 0.0
    def prepare(rnd):
        """Host-side setup for one round: a tear PER DEVICE, not one
        shared by all. Six cards sampling six different neighbourhoods
        explores six times as much as six re-sampling one; with a
        single device this is exactly the old behaviour."""
        jobs = []
        for u in units:
            bl, av, keep, nfix = make_round(rng)
            blmask = np.zeros(n, dtype=bool)
            for blk in keep:
                blmask[list(blk)] = True
            avb = np.zeros(nl, dtype=bool)
            for i in range(nl):
                avb[i] = bool((av[i >> 6] >> np.uint64(i & 63))
                              & np.uint64(1))
            # replay() is a function of (gid, seed), so a distinct seed
            # per device keeps every work-item independently
            # reproducible: the correctness story does not weaken as
            # devices are added.
            sd = np.uint32((ns.seed + rnd * 7919 + u.idx * 104729)
                           & 0xFFFFFFFF)
            jobs.append((u, sd, bl, av, nfix, blmask, avb, keep))
        return jobs

    def submit(jobs):
        for u, sd, bl, av, nfix, _, _, _ in jobs:
            u.enqueue(bl, av, nfix, sd)      # all devices run concurrently

    # One round in flight while the host works on the next. Without
    # this the cards idle through every host phase, which is what the
    # 9 MHz clocks were showing.
    pending = prepare(0)
    submit(pending)
    t0 = time.time()
    wall0 = time.time()
    host_t = 0.0
    for rnd in range(ns.rounds):
        if ns.seconds and time.time() - wall0 >= ns.seconds:
            print("  budget of %.0fs reached after %d rounds"
                  % (ns.seconds, rnd))
            break
        jobs = pending
        th = time.time()
        nxt = prepare(rnd + 1) if rnd + 1 < ns.rounds else None
        host = time.time() - th
        host_t += host
        results = [(u, sd, u.collect().copy(), bm, ab, kp)
                   for u, sd, _, _, _, bm, ab, kp in jobs]
        dt = time.time() - t0
        if nxt is not None:
            submit(nxt)                      # GPUs busy again immediately
            pending = nxt
        t0 = time.time()
        elapsed += dt
        total += total_items
        top = 0
        for _, _, sc, _, _, _ in results:
            # np.bincount, not a Python loop. Counting 6.29M scores one
            # int at a time cost 1.8s per round against a 1.2s kernel
            # window -- the GPUs sat at idle clocks (9-74 MHz, 3-6 W of
            # 110 W) waiting for the host to finish counting.
            bc = np.bincount(sc, minlength=K_TARGET + 1)
            for v in np.nonzero(bc)[0]:
                hist[int(v)] += int(bc[v])
            top = max(top, int(sc.max()))
        # host time is reported separately: it happens between submit
        # and collect, so it lands inside dt and used to masquerade as
        # slow GPUs. If host approaches dt, the cards are waiting on
        # Python, not the other way round.
        print("  round %d: %d candidates in %.2fs (host %.2fs) = %.0f/sec,"
              " best %d" % (rnd, total_items, dt, host,
                            total_items / max(dt, 1e-9), top))
        sys.stdout.flush()

        thr = ns.threshold or max(best, top) - 1
        if len(pool) < ns.keep or top > min(len(p) for p in pool):
            floor = thr if len(pool) < ns.keep else min(len(p) for p in pool)
            for u, sd, sc, blmask, avb, keep in results:
                gids = np.nonzero(sc >= floor)[0]
                if len(gids) > 4 * ns.keep:
                    gids = gids[np.argsort(-sc[gids])[:4 * ns.keep]]
                for gid in gids:
                    s_, blocks = replay(int(gid), int(sd), loops, adj, n,
                                        ns.maxatt, ns.cap, conf,
                                        blmask, avb, keep)
                    if s_ != int(sc[gid]):
                        print("    MISMATCH dev %d gid %d: kernel %d, host %d"
                              % (u.idx, gid, int(sc[gid]), s_))
                        continue
                    ok, bad = t.verify_blocks(blocks, adj, loops, B)
                    if not ok:
                        print("    dev %d gid %d failed the verifier: %s"
                              % (u.idx, gid, bad[:1]))
                        continue
                    key = tuple(sorted(tuple(b) for b in blocks))
                    if key in pool_keys:
                        continue
                    pool_keys.add(key)
                    pool.append(blocks)
                    if len(pool) > ns.keep:
                        pool.sort(key=lambda p: -len(p))
                        pool.pop()
                    if s_ > best:
                        best, best_blocks = s_, blocks
                        print("    dev %d gid %d: %d blocks, verifier pass"
                              % (u.idx, gid, s_))
            if pool:
                print("    pool %d/%d, sizes %d..%d"
                      % (len(pool), ns.keep, min(len(p) for p in pool),
                         max(len(p) for p in pool)))
        last = results

    print("")
    print("%d candidates in %.1fs = %.0f/sec (host %.1fs, %.0f%%)"
          % (total, elapsed, total / max(elapsed, 1e-9), host_t,
             100 * host_t / max(elapsed, 1e-9)))
    print("score distribution: %s" % dict(sorted(hist.items())))

    if ns.validate:
        print("")
        print("validating %d work-items per device by exact host replay"
              % ns.validate)
        bad = tot = 0
        for u, sd, sc, blmask, avb, keep in last:
            for gid in range(min(ns.validate, len(sc))):
                tot += 1
                s_, _ = replay(gid, int(sd), loops, adj, n, ns.maxatt,
                               ns.cap, conf, blmask, avb, keep)
                if s_ != int(sc[gid]):
                    bad += 1
                    print("  MISMATCH dev %d gid %d: kernel %d, host %d"
                          % (u.idx, gid, int(sc[gid]), s_))
        print("  %d/%d agree" % (tot - bad, tot))
        if bad:
            print("  kernel and host disagree -- do not trust the output")
            return 2

    if best_blocks:
        pool.sort(key=lambda p: -len(p))
        json.dump({"n_blocks": len(best_blocks), "verified": True,
                   "blocks": best_blocks,
                   "pool": pool,
                   "pool_sizes": [len(p) for p in pool]},
                  open(ns.out, "w"), indent=1)
        print("")
        print("best %d blocks; pool of %d distinct packings -> %s"
              % (best, len(pool), ns.out))
        print("sizes: %s"
              % dict(sorted(collections.Counter(len(p) for p in pool).items())))
        print("resume with: python3 %s --start %s --lns 2000"
              % (os.path.basename(tpath), ns.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
