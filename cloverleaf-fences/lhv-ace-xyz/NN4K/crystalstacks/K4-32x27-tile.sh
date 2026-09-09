#!/usr/bin/env bash
# K4-32x27-tile.sh -- run the three searches together, correctly.
#
# The naive version got three things wrong, all visible in its output:
#
#   - the GPU job never started, because it needs base.json and the job
#     writing base.json was still running. Dependencies are ordered
#     here, not raced.
#   - three processes wrote to one terminal and interleaved into
#     something unreadable. Each job gets its own timestamped log.
#   - the script ended (on a stray `pause`) and took the background
#     jobs with it. They run under setsid now, and the script waits on
#     them with a trap so Ctrl-C stops everything cleanly.
#
# Caches are warmed serially first: three processes each enumerating
# the same 1296 cycles is wasted work.
#
# Worker counts are derived from nproc unless you override them.
#
#   ./K4-32x27-tile.sh --plan                # show the split, run nothing
#   ./K4-32x27-tile.sh                       # 3 hours
#   SECS=43200 ./K4-32x27-tile.sh            # longer
#   GSIZE=1048576 ./K4-32x27-tile.sh         # smaller GPU rounds
#   MIXED_W=0 ./K4-32x27-tile.sh             # skip the mixed-cycle run
#   PROOF2_W=0 ./K4-32x27-tile.sh            # one proof, not two
#   DEVICES=0:0 ./K4-32x27-tile.sh           # one GPU instead of all
#   ./K4-32x27-tile.sh --detach              # keep running, free the prompt

set -u
cd "$(dirname "$0")" || exit 1

PY=${PY:-python3}
CP=${CP:-./K4-32x27-tilecp.py}
OCL=${OCL:-./K4-32x27-tileocl.py}
SECS=${SECS:-10800}
TEAR=${TEAR:-6}
# 'all' uses every GPU on every platform. Without this the OpenCL job
# defaults to the first device and leaves the rest of the fleet idle --
# on a six-die host that is five sixths of the throughput unused.
DEVICES=${DEVICES:-all}
# 4194304 work-items per device = 25.2M candidates per round across six
# dies. Measured: at 1048576 the six V340s ran 77-91 percent utilised at
# the 110 W cap; four times the work per round dilutes the remaining
# host time further, and the score buffer is still only 16.8 MB of the
# 7.98 GB each die has.
GSIZE=${GSIZE:-4194304}
# Rounds is now just a ceiling: the GPU job also honours --seconds, so
# it stops with the solvers instead of holding the launcher open for
# hours after they finish.
ROUNDS=${ROUNDS:-1000000}
LOGDIR=${LOGDIR:-logs}
DETACH=0
PLAN_ONLY=0
case "${1:-}" in
    --detach) DETACH=1 ;;
    --plan)   PLAN_ONLY=1 ;;
esac

# ---- core detection and allocation ------------------------------------
# nproc honours cgroup and affinity limits, which matters in a container:
# os.cpu_count() would report the host's threads and cheerfully
# oversubscribe. NTHREADS overrides everything, mostly for testing.
detect_threads() {
    if [ -n "${NTHREADS:-}" ]; then echo "$NTHREADS"; return; fi
    if command -v nproc >/dev/null 2>&1; then nproc; return; fi
    if command -v getconf >/dev/null 2>&1; then
        n=$(getconf _NPROCESSORS_ONLN 2>/dev/null) && [ -n "$n" ] &&             { echo "$n"; return; }
    fi
    $PY -c 'import os; print(os.cpu_count() or 1)'
}
T=$(detect_threads)
[ "$T" -lt 1 ] && T=1

# The OpenCL host process must keep six queues fed. Starve it and the
# cards idle: measured at GPU_W=2 with 94 solver workers, the host got
# 3 percent CPU and round times went from 1.2s to 9s as soon as the
# solvers left presolve and their search threads began spinning.
GPU_W=${GPU_W:-6}                       # host threads for the OpenCL job
# CP-SAT workers spin; the OpenCL host blocks on the GPU. At equal
# priority the spinners win every scheduling decision and the cards go
# idle. Niced down, the host thread preempts them -- it needs very
# little CPU, just not to wait for it.
SOLVER_NICE=${SOLVER_NICE:-10}
[ "$T" -le 4 ] && GPU_W=1
AVAIL=$((T - GPU_W))
[ "$AVAIL" -lt 1 ] && AVAIL=1

# The mixed-cycle run is the weakest of the three -- 2592 cycles against
# 1296, and it loses a redundant constraint that only holds at uniform
# cycle length -- so it gets a small share and nothing at all on a small
# machine. MIXED_W=0 skips it.
if [ -z "${MIXED_W:-}" ]; then
    if [ "$T" -ge 16 ]; then MIXED_W=$(( AVAIL * 15 / 100 )); else MIXED_W=0; fi
fi
[ "$MIXED_W" -lt 0 ] && MIXED_W=0

# Past roughly 48 workers CP-SAT's portfolio returns diminish: the
# subsolvers duplicate each other and shared-tree synchronisation starts
# to cost more than it buys. On a big machine two INDEPENDENT proofs --
# one with the hand-written symmetry breaking, one without -- explore
# more than one very wide portfolio, and they answer the same question,
# so whichever finishes first ends it.
if [ -z "${PROOF2_W:-}" ]; then
    if [ "$T" -ge 64 ]; then PROOF2_W=$(( (AVAIL - MIXED_W) / 2 )); else PROOF2_W=0; fi
fi
[ "$PROOF2_W" -lt 0 ] && PROOF2_W=0

PROOF_W=${PROOF_W:-$(( AVAIL - MIXED_W - PROOF2_W ))}
[ "$PROOF_W" -lt 1 ] && PROOF_W=1

mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d-%H%M%S)
PIDS=()

printf "=== %s threads detected ===\n" "$T"
printf "  %-22s %s\n" "proof (default)"     "$PROOF_W workers"
[ "$PROOF2_W" -gt 0 ] && \
printf "  %-22s %s\n" "proof (--no-redundant)" "$PROOF2_W workers"
[ "$MIXED_W" -gt 0 ] && \
printf "  %-22s %s\n" "mixed cycles 10,14"  "$MIXED_W workers"
printf "  %-22s %s\n" "gpu repair (host)"   "$GPU_W thread(s), devices=$DEVICES"
printf "  %-22s %s\n" "solver nice"         "+$SOLVER_NICE (gpu host runs at 0)"
printf "  %-22s %s\n" "budget"              "${SECS}s"
if [ "$PLAN_ONLY" = "1" ]; then exit 0; fi

cleanup() {
    echo ""
    echo "stopping jobs..."
    for p in ${PIDS[@]+"${PIDS[@]}"}; do
        kill -TERM "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null
    done
    exit 130
}
trap cleanup INT TERM

die() {                          # die <message> <logfile>
    echo ""
    echo "FAILED: $1"
    if [ -f "${2:-}" ]; then
        echo "--- last 20 lines of $2 ---"
        tail -20 "$2"
    fi
    exit 1
}

has_flag() {                     # has_flag <script> <--flag>
    # Word boundary required: a bare substring match reports --warm
    # present when the script only has --warmup, and the run then dies
    # at the first step with no output.
    $PY "$1" --help 2>/dev/null | grep -qE -- "(^|[^A-Za-z0-9-])$2([ ,=]|\$)"
}

launch() {
    local name=$1; shift
    local log="$LOGDIR/$name-$STAMP.log"
    local pri=()
    case "$name" in
        proof*|mixed) pri=(nice -n "$SOLVER_NICE") ;;
    esac
    setsid ${pri[@]+"${pri[@]}"} "$@" > "$log" 2>&1 &
    PIDS+=("$!")
    printf "  %-6s pid %-8s -> %s\n" "$name" "$!" "$log"
}

# Capability probes. These scripts get edited often and it is easy to
# update the launcher without updating what it launches; a missing flag
# then killed the run at the first step with no output at all.
WARMLOG="$LOGDIR/warm-$STAMP.log"
if ! has_flag "$CP" "--warm"; then
    echo "note: $CP has no --warm (older copy?); skipping the cache warm."
    echo "      each job will build the lattice itself -- duplicated work,"
    echo "      not fatal. Update $CP to avoid it."
else
    echo "=== warming lattice caches (serial, so each is built once) ==="
    $PY "$CP" --warm > "$WARMLOG" 2>&1 || die "lattice warm failed" "$WARMLOG"
    if [ "$MIXED_W" -gt 0 ]; then
        $PY "$CP" --cycles 10,14 --warm >> "$WARMLOG" 2>&1 \
            || die "mixed-cycle warm failed" "$WARMLOG"
    fi
    grep -h "cycles:" "$WARMLOG" | sed 's/^/  /'
fi

if ! has_flag "$OCL" "--devices"; then
    echo "note: $OCL has no --devices; it will use one GPU only."
    DEVICES=""
fi
if [ "$MIXED_W" -gt 0 ] && ! has_flag "$CP" "--cycles"; then
    echo "note: $CP has no --cycles; skipping the mixed-cycle job."
    MIXED_W=0
fi

echo "=== building base.json (the GPU job depends on it) ==="
$PY "$CP" --lns 1 --lns-seconds 10 --workers 2 --out base.json \
    > "$LOGDIR/base-$STAMP.log" 2>&1
[ -s base.json ] || die "base.json was not written" "$LOGDIR/base-$STAMP.log"
$PY -c 'import json;d=json.load(open("base.json"));print("  base.json: %d blocks, verified %s"%(d["n_blocks"],d["verified"]))'

echo "=== launching ==="
launch proof $PY "$CP" --lb 32 --seconds "$SECS" --workers "$PROOF_W" \
       --log --out proof.json
if [ "$PROOF2_W" -gt 0 ]; then
    # The symmetry question is settled (Rev 2.7: my value-precedence
    # encoding was 440x worse per probe and closed zero subtrees in an
    # hour, so it is off by default now). The second slot tests the
    # next open encoding question instead: whether the redundant
    # constraints -- loop sites per block, internal edge count -- earn
    # their keep or obstruct presolve the same way.
    launch proof2 $PY "$CP" --lb 32 --seconds "$SECS" --workers "$PROOF2_W" \
           --no-redundant --log --out proof-nored.json
fi
if [ "$MIXED_W" -gt 0 ]; then
    launch mixed $PY "$CP" --cycles 10,14 --lb 32 --seconds "$SECS" \
           --workers "$MIXED_W" --no-builtin --log --out proof-mixed.json
fi
if [ -n "$DEVICES" ]; then
    launch gpu $PY "$OCL" --fix base.json --tear "$TEAR" --devices "$DEVICES" \
           --global-size "$GSIZE" --rounds "$ROUNDS" --seconds "$SECS" \
           --keep 64 --out rep.json
else
    launch gpu $PY "$OCL" --fix base.json --tear "$TEAR" \
           --global-size "$GSIZE" --rounds "$ROUNDS" --seconds "$SECS" \
           --keep 64 --out rep.json
fi

sleep 3
for p in ${PIDS[@]+"${PIDS[@]}"}; do
    kill -0 "$p" 2>/dev/null || echo "  WARNING: pid $p already exited --" \
        "check its log above"
done

echo ""
echo "monitor:  tail -f $LOGDIR/*-$STAMP.log"
echo "verdicts: proof.json / proof-mixed.json / rep.json"

if [ "$DETACH" = "1" ]; then
    echo "detached. stop with:  kill ${PIDS[*]}"
    exit 0
fi

echo "waiting -- Ctrl-C stops all of them"
wait
echo ""
echo "=== finished ==="
for f in proof proof2 mixed gpu; do
    l="$LOGDIR/$f-$STAMP.log"
    [ -f "$l" ] && printf "%-6s %s\n" "$f" \
        "$(grep -E 'NO TILING|TILING EXISTS|^OPEN|^PARTIAL|best .* blocks;' "$l" | tail -1)"
done
