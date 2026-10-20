#!/usr/bin/env bash
#
# sweep.sh -- regenerate the whole B-to-B series set under one pyqrack version.
#
# Everything runs on OpenCL device 0 by default, matching the single-device
# directive: Qrack otherwise spreads work across every detected card, which is
# what turned an ~18s reference run into a ~227s one by scattering the state
# vector across a PCIe 1.0 x4 link.
#
# Because both stages share one card, series are processed one at a time, and
# only the ACE and ideal workers for the SAME series are ever in flight
# together. That overlap is still worth having: the ACE pass is dominated by
# Python-side _correct()/prob() traffic rather than GPU work, so it fills CPU
# while the reference pass has the GPU.
#
# Device memory at width 28, per worker:
#     ideal   ~3.6 GiB   (2 GiB state vector + CUDA context)
#     ace     ~0.4 GiB   (patch sims are <= 20 qubits, 8 MiB)
# Defaults of ACE_JOBS=1 + IDEAL_JOBS=2 therefore sit near 7.6 GiB of the
# CMP 50HX's 10 GiB. Host RAM is the other ceiling: ~5.3 GiB per ideal worker.
# Raise IDEAL_JOBS only after checking both.
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
#   DEVICE        OpenCL device for both stages     (default 0)
#   ACE_DEVICE    override for the ACE pass         (default $DEVICE)
#   IDEAL_DEVICE  override for the reference        (default $DEVICE)
#   ACE_JOBS      concurrent ACE workers            (default 1)
#   IDEAL_JOBS    concurrent ideal workers          (default 2)
#   PY            python interpreter                (default python3)
#   SCRIPT        path to nn_qab.py                 (default ./nn_qab.py)

set -euo pipefail

SEEDS="${SEEDS:-0-99}"
DEPTH="${DEPTH:-12}"
OUT_ROOT="${OUT_ROOT:-runs}"
DEVICE="${DEVICE:-0}"
ACE_DEVICE="${ACE_DEVICE:-$DEVICE}"
IDEAL_DEVICE="${IDEAL_DEVICE:-$DEVICE}"
ACE_JOBS="${ACE_JOBS:-1}"
IDEAL_JOBS="${IDEAL_JOBS:-2}"
PY="${PY:-python3}"
SCRIPT="${SCRIPT:-./nn_qab.py}"

# width lrc lrr -- the geometrically optimal 2-patch config for each width.
# B-to-B ratio is 1.5, 2.5, 4.5, 5.5, 3.5, and 2.5 again at width 28, which
# is the replicate that breaks the width/ratio collinearity in the fit.
# Ordered cheapest-first so a failure surfaces on a small series.
SERIES=(
    "10 2 2"
    "14 3 2"
    "22 5 2"
    "26 6 2"
    "27 4 3"
    "28 3 4"
)

# ---------------------------------------------------------------------------

log() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*"; }

[ -f "$SCRIPT" ] || {
    printf 'error: %s not found (set SCRIPT=...)\n' "$SCRIPT" >&2
    exit 1
}

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
ACE_PIDS=()
cleanup() {
    local pid
    for pid in "${ACE_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

count_done() {  # count_done <dir>
    local d="$1"
    [ -d "$d" ] || { printf '0'; return; }
    find "$d" -maxdepth 1 -name '*.json' -type f 2>/dev/null | wc -l | tr -d ' '
}

any_alive() {   # any_alive <pid...>
    local pid
    for pid in "$@"; do
        kill -0 "$pid" 2>/dev/null && return 0
    done
    return 1
}

# ---------------------------------------------------------------------------
log "sweep: ${#SERIES[@]} series x $N_SEEDS seeds, depth $DEPTH"
log "ace -> device $ACE_DEVICE ($ACE_JOBS jobs)   ideal -> device $IDEAL_DEVICE ($IDEAL_JOBS jobs)"
START=$(date +%s)

for cfg in "${SERIES[@]}"; do
    # shellcheck disable=SC2086
    set -- $cfg
    width="$1"; lrc="$2"; lrr="$3"
    out="$OUT_ROOT/w$width"
    log "series width=$width lrc=$lrc lrr=$lrr"

    # --- stage 1: ACE, backgrounded so stage 2 can consume as it goes -------
    ACE_PIDS=()
    for ((j = 0; j < ACE_JOBS; j++)); do
        "$PY" -u "$SCRIPT" ace \
            --device "$ACE_DEVICE" \
            --width "$width" --depth "$DEPTH" --lrc "$lrc" --lrr "$lrr" \
            --seeds "$SEEDS" --out "$out" \
            >> "$OUT_ROOT/logs/ace-w$width.log" 2>&1 &
        ACE_PIDS+=($!)
    done

    # --- stage 2: ideal, looping until this series is complete --------------
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
        log "  w$width: xeb $after/$N_SEEDS  ace $(count_done "$out/ace")/$N_SEEDS"
        [ "$after" -ge "$N_SEEDS" ] && break

        if [ "$after" -le "$before" ]; then
            # No progress. Either the ACE pass hasn't caught up yet, or it has
            # finished and some seeds died leaving stale locks behind.
            if any_alive "${ACE_PIDS[@]}"; then
                sleep 15
            else
                stall=$(( stall + 1 ))
                if [ "$stall" -ge 3 ]; then
                    log "  w$width: stalled at $after/$N_SEEDS, moving on"
                    log "  (see $OUT_ROOT/logs/, and for stale *.lock under $out)"
                    break
                fi
                sleep 5
            fi
        else
            stall=0
        fi
    done

    wait "${ACE_PIDS[@]}" 2>/dev/null || true
    ACE_PIDS=()
done

# --- stage 3: merge ---------------------------------------------------------
log "merge"
SUMMARY="$OUT_ROOT/summary.tsv"
printf 'width\tlrc\tlrr\tB_to_B\tn\tmean_xeb\tstdev_xeb\n' > "$SUMMARY"

for cfg in "${SERIES[@]}"; do
    # shellcheck disable=SC2086
    set -- $cfg
    width="$1"; lrc="$2"; lrr="$3"
    out="$OUT_ROOT/w$width"
    csv="$OUT_ROOT/w$width.csv"

    if ! "$PY" "$SCRIPT" merge --out "$out" --csv "$csv" > /dev/null 2>&1; then
        log "  w$width: no results"
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
