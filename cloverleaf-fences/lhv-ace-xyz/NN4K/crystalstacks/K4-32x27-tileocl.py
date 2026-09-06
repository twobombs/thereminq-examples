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
            const uint seed,
            __global int *score,
            __global int *sel_out)
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

    // ---- pick KT pairwise-disjoint loops -------------------------
    for (w = 0; w < CW; w++) avail[w] = ~0UL;
    if (NL & 63) avail[CW - 1] = (1UL << (NL & 63)) - 1UL;

    for (int a = 0; a < MAXATT && k < KT; a++) {
        int r = (int)(xs32(&st) % (uint)NL);
        if (!GET(avail, r)) continue;
        sel[k++] = r;
        for (w = 0; w < CW; w++) avail[w] &= ~conf[r * CW + w];
    }
    if (k < KT) { score[gid] = 0; return; }
    for (j = 0; j < KT; j++) sel_out[gid * KT + j] = sel[j];

    for (w = 0; w < W; w++) lfree[base + w] = ~0UL;
    for (j = 0; j < KT; j++)
        for (w = 0; w < W; w++) lfree[base + w] &= ~lmask[sel[j] * W + w];

    // ---- grow each loop into a B-site block ----------------------
    int done = 0;
    for (int b = 0; b < KT; b++) {
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


def replay(gid, seed, loops, adj, n, maxatt, cap, conf):
    """Reproduce one work-item exactly. Returns (score, blocks)."""
    r = XS32(gid, seed)
    nl = len(loops)
    avail = np.ones(nl, dtype=bool)
    sel = []
    for _ in range(maxatt):
        if len(sel) >= K_TARGET:
            break
        i = r.next() % nl
        if not avail[i]:
            continue
        sel.append(i)
        avail &= ~conf[i]
    if len(sel) < K_TARGET:
        return 0, []
    free = np.ones(n, dtype=bool)
    for i in sel:
        free[list(loops[i])] = False
    blocks = []
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
    ap.add_argument("--maxatt", type=int, default=20000)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--local", type=int, default=64,
                    help="work-group size; LDS use is 2*local*W*8 bytes")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--platform", type=int, default=-1)
    ap.add_argument("--device", type=int, default=-1)
    ap.add_argument("--list-devices", action="store_true")
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

    plats = cl.get_platforms()
    plat = plats[ns.platform] if ns.platform >= 0 else None
    if plat is None:
        for p in plats:
            if p.get_devices(cl.device_type.GPU):
                plat = p
                break
        plat = plat or plats[0]
    devs = plat.get_devices()
    dev = devs[ns.device] if ns.device >= 0 else devs[0]
    print("device: %s (%s), %d CUs"
          % (dev.name.strip(), plat.name.strip(), dev.max_compute_units))

    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    lsz = ns.local
    lds = 2 * lsz * w * 8
    if lds > dev.local_mem_size:
        lsz = max(8, int(dev.local_mem_size / (2 * w * 8)) & ~7)
        print("local size %d needs %d KB LDS, only %d KB available;"
              " dropping to %d" % (ns.local, lds // 1024,
                                   dev.local_mem_size // 1024, lsz))
    print("work-group %d, LDS %d B/group (%d B/item)"
          % (lsz, 2 * lsz * w * 8, 2 * w * 8))
    opts = ("-D N=%d -D W=%d -D NL=%d -D CW=%d -D ELL=%d -D B=%d -D KT=%d "
            "-D CAP=%d -D MAXATT=%d -D LS=%d"
            % (n, w, nl, cw, ELL, B, K_TARGET, ns.cap, ns.maxatt, lsz))
    t0 = time.time()
    prg = cl.Program(ctx, KERNEL).build(options=opts)
    krn = cl.Kernel(prg, "search")   # retrieve once; re-retrieving per
    print("kernel built in %.2fs" % (time.time() - t0))   # round is costly

    mf = cl.mem_flags
    d_lmask = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lmask)
    d_lsite = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lsite)
    d_adj = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=adjv)
    d_conf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=confw)
    G = (ns.global_size // lsz) * lsz
    score = np.zeros(G, dtype=np.int32)
    d_score = cl.Buffer(ctx, mf.WRITE_ONLY, score.nbytes)
    d_sel = cl.Buffer(ctx, mf.WRITE_ONLY, G * K_TARGET * 4)

    hist = collections.Counter()
    best, best_blocks = 0, []
    pool, pool_keys = [], set()
    total, elapsed = 0, 0.0
    for rnd in range(ns.rounds):
        seed = np.uint32((ns.seed + rnd * 7919) & 0xFFFFFFFF)
        t0 = time.time()
        krn(q, (G,), (lsz,), d_lmask, d_lsite, d_adj, d_conf, seed,
            d_score, d_sel)
        cl.enqueue_copy(q, score, d_score)
        q.finish()
        dt = time.time() - t0
        elapsed += dt
        total += G
        for s in score:
            hist[int(s)] += 1
        top = int(score.max())
        print("  round %d: %d candidates in %.2fs = %.0f/sec, best %d"
              % (rnd, G, dt, G / dt, top))
        sys.stdout.flush()
        # Harvest EVERY good candidate, not just the round's best.
        # A single winner per round throws away ~150 distinct 30-block
        # packings a second, and a diverse pool of starts is worth far
        # more to LNS than one deep one.
        # The pool must EVICT, not merely fill: filling it with the
        # first round's 29s and then ignoring every later 30 is worse
        # than keeping one winner. Accept anything beating the pool's
        # weakest member and drop that member.
        thr = ns.threshold or max(best, top) - 1
        if len(pool) < ns.keep or top > min(len(p) for p in pool):
            floor = thr if len(pool) < ns.keep else min(len(p) for p in pool)
            gids = np.nonzero(score >= floor)[0]
            if len(gids) > 4 * ns.keep:
                gids = gids[np.argsort(-score[gids])[:4 * ns.keep]]
            for gid in gids:
                sc, blocks = replay(int(gid), int(seed), loops, adj, n,
                                    ns.maxatt, ns.cap, conf)
                if sc != int(score[gid]):
                    print("    MISMATCH gid %d: kernel %d, host %d"
                          % (gid, int(score[gid]), sc))
                    continue
                ok, bad = t.verify_blocks(blocks, adj, loops, B)
                if not ok:
                    print("    gid %d failed the verifier: %s" % (gid, bad[:1]))
                    continue
                key = tuple(sorted(tuple(b) for b in blocks))
                if key in pool_keys:
                    continue
                pool_keys.add(key)
                pool.append(blocks)
                if len(pool) > ns.keep:
                    pool.sort(key=lambda p: -len(p))
                    pool.pop()
                if sc > best:
                    best, best_blocks = sc, blocks
                    print("    gid %d: %d blocks, verifier pass" % (gid, sc))
            if gids.size:
                print("    pool %d/%d, sizes %d..%d"
                      % (len(pool), ns.keep,
                         min(len(p) for p in pool) if pool else 0,
                         max(len(p) for p in pool) if pool else 0))

    print("")
    print("%d candidates in %.1fs = %.0f/sec" % (total, elapsed,
                                                 total / max(elapsed, 1e-9)))
    print("score distribution: %s" % dict(sorted(hist.items())))

    if ns.validate:
        print("")
        print("validating %d work-items by exact host replay" % ns.validate)
        seed = int(np.uint32((ns.seed + (ns.rounds - 1) * 7919) & 0xFFFFFFFF))
        bad = 0
        for gid in range(min(ns.validate, G)):
            sc, _ = replay(gid, seed, loops, adj, n, ns.maxatt,
                           ns.cap, conf)
            if sc != int(score[gid]):
                bad += 1
                print("  MISMATCH gid %d: kernel %d, host %d"
                      % (gid, int(score[gid]), sc))
        print("  %d/%d agree" % (min(ns.validate, G) - bad,
                                 min(ns.validate, G)))
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