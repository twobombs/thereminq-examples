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
#   ./K4-32x27-tile.sh                       # 12 hours
#   SECS=3600 ./K4-32x27-tile.sh             # shorter
#   MIXED_W=0 ./K4-32x27-tile.sh             # skip the mixed-cycle run
#   PROOF2_W=0 ./K4-32x27-tile.sh            # one proof, not two
#   ./K4-32x27-tile.sh --detach              # keep running, free the prompt

set -u
cd "$(dirname "$0")" || exit 1

PY=${PY:-python3}
CP=${CP:-./K4-32x27-tilecp.py}
OCL=${OCL:-./K4-32x27-tileocl.py}
SECS=${SECS:-43200}
TEAR=${TEAR:-6}
GSIZE=${GSIZE:-1048576}
ROUNDS=${ROUNDS:-5000}
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

GPU_W=${GPU_W:-2}                       # host thread for the OpenCL job
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
printf "  %-22s %s\n" "proof (symmetry)"    "$PROOF_W workers"
[ "$PROOF2_W" -gt 0 ] && \
printf "  %-22s %s\n" "proof (--no-symmetry)" "$PROOF2_W workers"
[ "$MIXED_W" -gt 0 ] && \
printf "  %-22s %s\n" "mixed cycles 10,14"  "$MIXED_W workers"
printf "  %-22s %s\n" "gpu repair (host)"   "$GPU_W thread(s)"
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

launch() {
    local name=$1; shift
    local log="$LOGDIR/$name-$STAMP.log"
    setsid "$@" > "$log" 2>&1 &
    PIDS+=("$!")
    printf "  %-6s pid %-8s -> %s\n" "$name" "$!" "$log"
}

echo "=== warming lattice caches (serial, so each is built once) ==="
$PY "$CP" --warm > "$LOGDIR/warm-$STAMP.log" 2>&1 || exit 1
if [ "$MIXED_W" -gt 0 ]; then
    $PY "$CP" --cycles 10,14 --warm >> "$LOGDIR/warm-$STAMP.log" 2>&1 || exit 1
fi
grep -h "cycles:" "$LOGDIR/warm-$STAMP.log" | sed 's/^/  /'

echo "=== building base.json (the GPU job depends on it) ==="
$PY "$CP" --lns 1 --lns-seconds 10 --workers 2 --out base.json \
    > "$LOGDIR/base-$STAMP.log" 2>&1
if [ ! -s base.json ]; then
    echo "  base.json was not written; see $LOGDIR/base-$STAMP.log"
    exit 1
fi
$PY -c 'import json;d=json.load(open("base.json"));print("  base.json: %d blocks, verified %s"%(d["n_blocks"],d["verified"]))'

echo "=== launching ==="
launch proof $PY "$CP" --lb 32 --seconds "$SECS" --workers "$PROOF_W" \
       --log --out proof.json
if [ "$PROOF2_W" -gt 0 ]; then
    launch proof2 $PY "$CP" --lb 32 --seconds "$SECS" --workers "$PROOF2_W" \
           --no-symmetry --log --out proof-nosym.json
fi
if [ "$MIXED_W" -gt 0 ]; then
    launch mixed $PY "$CP" --cycles 10,14 --lb 32 --seconds "$SECS" \
           --workers "$MIXED_W" --no-builtin --log --out proof-mixed.json
fi
launch gpu $PY "$OCL" --fix base.json --tear "$TEAR" \
       --global-size "$GSIZE" --rounds "$ROUNDS" --keep 64 --out rep.json

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
