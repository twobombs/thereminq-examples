#!/usr/bin/env bash
#
# sweep.sh -- regenerate the whole B-to-B series set under one pyqrack version.
#
# Stage 1 runs the ACE pass for every geometry on the small card. Stage 2 runs
# the exact reference on the big card, in a loop that re-scans for work, so the
# two stages pipeline: ideal workers consume seeds as soon as ACE produces them
# instead of waiting for the whole ACE stage to finish. Stage 3 merges.
#
# Both stages are resumable. Seeds already finished are skipped, so killing
# this script and re-running it costs only the samples that were in flight.
#
# Usage:
#   ./sweep.sh                 # every series
#   ./sweep.sh 27 28           # only these widths
#   SEEDS=0-9 ./sweep.sh 10    # quick smoke test
#
# Tunables (environment):
#   SEEDS         seed range                        (default 0-99)
#   DEPTH         circuit depth                     (default 12)
#   OUT_ROOT      output directory                  (default runs)
#   ACE_DEVICE    OpenCL device for the ACE pass    (default 1)
#   IDEAL_DEVICE  OpenCL device for the reference   (default 0)
#   ACE_JOBS      concurrent ACE workers            (default 2)
#   IDEAL_JOBS    concurrent ideal workers          (default 2)
#   PY            python interpreter                (default python3)
#   SCRIPT        path to nn_qab.py                 (default ./nn_qab.py)

set -euo pipefail

SEEDS="${SEEDS:-0-99}"
DEPTH="${DEPTH:-12}"
OUT_ROOT="${OUT_ROOT:-runs}"
ACE_DEVICE="${ACE_DEVICE:-1}"
IDEAL_DEVICE="${IDEAL_DEVICE:-0}"
ACE_JOBS="${ACE_JOBS:-2}"
IDEAL_JOBS="${IDEAL_JOBS:-2}"
PY="${PY:-python3}"
SCRIPT="${SCRIPT:-./nn_qab.py}"

# width lrc lrr -- the geometrically optimal 2-patch config for each width.
# B-to-B ratio is 1.5, 2.5, 3.5, 4.5, 5.5, and 2.5 again at width 28, which
# is the replicate that breaks the width/ratio collinearity in the fit.
SERIES=(
    "10 2 2"
    "14 3 2"
    "27 4 3"
    "22 5 2"
    "26 6 2"
    "28 3 4"
)

# ---------------------------------------------------------------------------

log() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[ -f "$SCRIPT" ] || die "$SCRIPT not found (set SCRIPT=/path/to/nn_qab.py)"

# Expand the seed spec so we know when a series is complete. Accepts the same
# "0-9,20,30-39" syntax nn_qab.py takes.
seed_count() {
    local total=0 part lo hi
    IFS=',' read -ra parts <<< "$1"
    for part in "${parts[@]}"; do
        if [[ "$part" == *-* ]]; then
            lo="${part%%-*}"; hi="${part##*-}"
            total=$(( total + hi - lo + 1 ))
        else
            total=$(( total + 1 ))
        fi
    done
    printf '%d' "$total"
}

N_SEEDS="$(seed_count "$SEEDS")"

# Filter to the widths named on the command line, if any.
if [ "$#" -gt 0 ]; then
    WANTED=("$@")
    filtered=()
    for want in "${WANTED[@]}"; do
        for cfg in "${SERIES[@]}"; do
            # shellcheck disable=SC2086
            set -- $cfg
            [ "$1" = "$want" ] && filtered+=("$cfg")
        done
    done
    if [ "${#filtered[@]}" -eq 0 ]; then
        printf 'error: no series matched: %s\n' "${WANTED[*]}" >&2
        printf 'known widths:' >&2
        for cfg in "${SERIES[@]}"; do
            # shellcheck disable=SC2086
            set -- $cfg
            printf ' %s' "$1" >&2
        done
        printf '\n' >&2
        exit 1
    fi
    SERIES=("${filtered[@]}")
fi

mkdir -p "$OUT_ROOT/logs"

# Kill background workers if we're interrupted, rather than orphaning GPU jobs.
PIDS=()
cleanup() {
    local pid
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

count_done() {  # count_done <dir>
    local d="$1"
    [ -d "$d" ] || { printf '0'; return; }
    find "$d" -maxdepth 1 -name '*.json' -type f 2>/dev/null | wc -l | tr -d ' '
}

# ---------------------------------------------------------------------------
log "sweep: ${#SERIES[@]} series x $N_SEEDS seeds, depth $DEPTH"
log "ace -> device $ACE_DEVICE ($ACE_JOBS jobs)   ideal -> device $IDEAL_DEVICE ($IDEAL_JOBS jobs)"
START=$(date +%s)

# --- stage 1: ACE, backgrounded so stage 2 can start consuming --------------
for cfg in "${SERIES[@]}"; do
    set -- $cfg
    width="$1"; lrc="$2"; lrr="$3"
    out="$OUT_ROOT/w$width"
    for ((j = 0; j < ACE_JOBS; j++)); do
        "$PY" -u "$SCRIPT" ace \
            --device "$ACE_DEVICE" \
            --width "$width" --depth "$DEPTH" --lrc "$lrc" --lrr "$lrr" \
            --seeds "$SEEDS" --out "$out" \
            >> "$OUT_ROOT/logs/ace-w$width.log" 2>&1 &
        PIDS+=($!)
    done
done
log "stage 1: ${#PIDS[@]} ACE workers launched"

# --- stage 2: ideal, looping until each series is complete ------------------
for cfg in "${SERIES[@]}"; do
    set -- $cfg
    width="$1"
    out="$OUT_ROOT/w$width"
    log "stage 2: width $width"

    stall=0
    while :; do
        before="$(count_done "$out/xeb")"

        ipids=()
        for ((j = 0; j < IDEAL_JOBS; j++)); do
            "$PY" -u "$SCRIPT" ideal \
                --device "$IDEAL_DEVICE" \
                --seeds "$SEEDS" --out "$out" \
                >> "$OUT_ROOT/logs/ideal-w$width.log" 2>&1 &
            ipids+=($!)
        done
        wait "${ipids[@]}" || true

        after="$(count_done "$out/xeb")"
        log "  width $width: $after/$N_SEEDS"
        [ "$after" -ge "$N_SEEDS" ] && break

        # No progress this pass means we're waiting on the ACE stage -- or,
        # if ACE has finished too, that some seeds died and left a stale
        # lock. Give up rather than spin forever.
        if [ "$after" -le "$before" ]; then
            stall=$(( stall + 1 ))
            if [ "$stall" -ge 20 ]; then
                log "  width $width: stalled at $after/$N_SEEDS, moving on"
                log "  (check $OUT_ROOT/logs/, and for stale *.lock in $out)"
                break
            fi
            sleep 15
        else
            stall=0
        fi
    done
done

wait || true          # let any still-running ACE workers finish
PIDS=()

# --- stage 3: merge ---------------------------------------------------------
log "stage 3: merge"
SUMMARY="$OUT_ROOT/summary.tsv"
printf 'width\tlrc\tlrr\tB_to_B\tn\tmean_xeb\tstdev_xeb\n' > "$SUMMARY"

for cfg in "${SERIES[@]}"; do
    set -- $cfg
    width="$1"; lrc="$2"; lrr="$3"
    out="$OUT_ROOT/w$width"
    csv="$OUT_ROOT/w$width.csv"

    if ! "$PY" "$SCRIPT" merge --out "$out" --csv "$csv" > /dev/null 2>&1; then
        log "  width $width: no results"
        continue
    fi

    # bulk_to_boundary is column 8, xeb_ace column 12 (see FIELDS in nn_qab.py)
    awk -F',' -v w="$width" -v c="$lrc" -v r="$lrr" '
        NR > 1 { b = $8; n++; s += $12; q += $12 * $12 }
        END {
            if (n == 0) exit
            m = s / n
            sd = (n > 1) ? sqrt((q - n * m * m) / (n - 1)) : 0
            printf "%s\t%s\t%s\t%s\t%d\t%.10f\t%.10f\n", w, c, r, b, n, m, sd
        }' "$csv" >> "$SUMMARY"
done

ELAPSED=$(( $(date +%s) - START ))
log "done in $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo
if command -v column > /dev/null 2>&1; then
    column -t -s $'\t' "$SUMMARY"
else
    awk -F'\t' '{ for (i = 1; i <= NF; i++) printf "%-14s", $i; print "" }' "$SUMMARY"
fi
echo
echo "per-run rows: $OUT_ROOT/w*.csv"
echo "summary:      $SUMMARY"
