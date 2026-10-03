#!/usr/bin/env bash
#
# sweep.sh -- regenerate the whole B-to-B series set under one pyqrack version.
#
# Target: the 96-thread EPYC box -- 6 Radeon Pro V340 dies (8 GB each, two
# per card sharing one PCIe 3.0 x8 link) through Mesa rusticl, 320 GB RAM,
# swap on an 8-drive NVMe RAID0, Qrack built for > 32 qubits.
# Check the "OpenCL device #n:" list Qrack prints at startup; the defaults
# below assume the six V340 dies are devices 0-5.
#
# Series run one at a time, cheapest first. Within a series the ACE and
# ideal stages overlap: ACE is dominated by Python-side _correct()/prob()
# traffic, so it fills CPU while the reference has the devices.
#
# Three reference profiles, by width:
#
#   <= SMALL_MAX (28)   one ideal worker per die in IDEAL_DEVICES, each
#                       pinned to its own die (2 GiB state vector at 28).
#                       The ACE pass keeps ACE_DEVICES to itself.
#
#   > SMALL_MAX         one worker, Qrack's CPU engine (BIG_ENGINE). From
#                       here the state vector is >= 8 GiB and outgrows a
#                       die; spreading it across dies means global-qubit
#                       page swaps over shared PCIe 3.0 x8 links -- the
#                       same effect that turned an ~18 s run into ~227 s on
#                       the dev rig.
#
#   PAGER_DEVICES set   widths SMALL_MAX < w <= PAGER_MAX (30) instead page
#                       one GPU simulator across those dies. Opt-in: time
#                       one seed against the CPU engine before committing.
#
#     width  state vec  + probs (exact)  lives in
#       30      8 GiB       4 GiB        RAM
#       32     32 GiB      16 GiB        RAM
#       33     64 GiB      32 GiB        RAM
#       34    128 GiB      64 GiB        RAM
#       35    256 GiB     128 GiB        state in RAM, probs ~partly swap
#       36    512 GiB     256 GiB        mostly swap
#
# With ALLOW_SWAP=1 (default) nn_qab.py counts SwapFree in its memory
# budget, so --stats auto keeps exact stats through 36 and only falls back
# to linear (Porter-Thomas) XEB if even swap runs out. Every row records
# xeb_linear too, and summary.tsv carries both means.
#
# NVMe wear: at 36 roughly 200 GiB of the state vector sits outside RAM and
# each two-qubit gate pass rewrites it. Each seed's swap traffic is logged
# (swap_out_gib in the CSV, per-series total in summary.tsv). Run ONE seed
# at 36 first (SEEDS=0 ./sweep.sh 36) and compare swap_out_gib against the
# drives' rated TBW before launching more.
#
# nn_qab.py refuses up front (exit 2) when a width won't fit at all; the
# series is then skipped, but its ACE pass still runs to completion. Each
# ace/*.json holds the full circuit plus counts, so a reference can be
# produced for it later.
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
#   SEEDS          seed range                         (default 0-99)
#   DEPTH          circuit depth                      (default 12)
#   OUT_ROOT       output directory                   (default runs)
#   ACE_DEVICES    dies for ACE workers, round-robin  (default 0)
#   ACE_JOBS       concurrent ACE workers             (default 2)
#   IDEAL_DEVICES  dies for small-width ideal workers (default 1,2,3,4,5)
#   IDEAL_JOBS     ideal workers for <= SMALL_MAX     (default: one per die)
#   SMALL_MAX      last width on the per-die profile  (default 28)
#   IDEAL_ENGINE   reference engine, per-die profile  (default statevector)
#   BIG_ENGINE     reference engine above SMALL_MAX   (default cpu)
#   PAGER_DEVICES  comma list: page widths up to PAGER_MAX across these
#                  dies on the GPU instead            (default unset)
#   PAGER_MAX      last width PAGER_DEVICES covers    (default 30)
#   ALLOW_SWAP     1 = let the guard count swap       (default 1)
#   STATS          auto | exact | linear              (default auto)
#   SDRP           Schmidt rounding for the reference, approximate and
#                  fidelity-tagged                    (default unset)
#   SDRP_FROM      first width SDRP applies to        (default 36)
#   PY             python interpreter                 (default python3)
#   SCRIPT         path to nn_qab.py                  (default ./nn_qab.py)

set -euo pipefail

SEEDS="${SEEDS:-0-99}"
DEPTH="${DEPTH:-12}"
OUT_ROOT="${OUT_ROOT:-runs}"
ACE_DEVICES="${ACE_DEVICES:-0}"
ACE_JOBS="${ACE_JOBS:-2}"
IDEAL_DEVICES="${IDEAL_DEVICES:-1,2,3,4,5}"
IFS=',' read -ra IDEAL_DEV_LIST <<< "$IDEAL_DEVICES"
IFS=',' read -ra ACE_DEV_LIST <<< "$ACE_DEVICES"
IDEAL_JOBS="${IDEAL_JOBS:-${#IDEAL_DEV_LIST[@]}}"
SMALL_MAX="${SMALL_MAX:-28}"
IDEAL_ENGINE="${IDEAL_ENGINE:-statevector}"
BIG_ENGINE="${BIG_ENGINE:-cpu}"
PAGER_DEVICES="${PAGER_DEVICES:-}"
PAGER_MAX="${PAGER_MAX:-30}"
ALLOW_SWAP="${ALLOW_SWAP:-1}"
STATS="${STATS:-auto}"
SDRP="${SDRP:-}"
SDRP_FROM="${SDRP_FROM:-36}"
PY="${PY:-python3}"
SCRIPT="${SCRIPT:-./nn_qab.py}"

# width lrc lrr -- the geometrically optimal 2-patch config for each width.
# B-to-B ratio is 1.5, 2.5, 4.5, 5.5, 3.5, and 2.5 again at width 28, which
# is the replicate that breaks the width/ratio collinearity in the fit.
# Ordered cheapest-first so a failure surfaces on a small series.
#
# 30-36 follow the same rule as the rows above: rows left whole
# (lrr = column length), lrc chosen for exactly two patches at the highest
# finite B-to-B ratio, ties broken toward the most balanced patch sizes.
# Reproduces all six original rows. Patch sizes include each patch's
# detection ancilla.
#
#   width  grid   B-to-B  patches
#     30   6x5     2.0    21 + 21
#     32   8x4     3.0    21 + 21
#     33  11x3     4.5    22 + 19    replicates width 22
#     34  17x2     7.5    21 + 19    highest ratio in the set
#     35   7x5     2.5    26 + 21    replicates widths 14 and 28
#     36   6x6     2.0    25 + 25    replicates width 30
#
# 29 and 31 are prime: factor_width() collapses them to a 29x1 / 31x1 ring,
# a 1-D topology unlike every other row (B-to-B 13.5 / 14.5). Left out for
# the same reason 11, 13, 17, 19 and 23 are; uncomment to add them anyway.
SERIES=(
    "10 2 2"
    "14 3 2"
    "22 5 2"
    "26 6 2"
    "27 4 3"
    "28 3 4"
    "30 2 5"
    "32 3 4"
    "33 5 3"
    "34 8 2"
    "35 3 5"
    "36 2 6"
    # "29 14 1"
    # "31 15 1"
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

# Reference settings for one width: sets I_ENGINE, I_JOBS, I_EXTRA.
ideal_profile() {
    local w="$1"
    I_EXTRA=(--stats "$STATS")
    if [ "$w" -le "$SMALL_MAX" ]; then
        I_ENGINE="$IDEAL_ENGINE"; I_JOBS="$IDEAL_JOBS"
        I_DEVS=("${IDEAL_DEV_LIST[@]}")
    elif [ -n "$PAGER_DEVICES" ] && [ "$w" -le "$PAGER_MAX" ]; then
        I_ENGINE="$IDEAL_ENGINE"; I_JOBS=1
        I_DEVS=("$PAGER_DEVICES")             # one worker, paged across these
    else
        I_ENGINE="$BIG_ENGINE"; I_JOBS=1
        I_DEVS=("${IDEAL_DEV_LIST[0]}")       # CPU engine; device is moot
    fi
    # One worker above SMALL_MAX is deliberate: the memory guard reads
    # MemAvailable/SwapFree once per seed, and two workers checking at the
    # same moment would both pass and then both allocate.
    if [ "$w" -gt "$SMALL_MAX" ] && [ "$ALLOW_SWAP" = 1 ]; then
        I_EXTRA+=(--allow-swap)
    fi
    if [ -n "$SDRP" ] && [ "$w" -ge "$SDRP_FROM" ]; then
        I_EXTRA+=(--sdrp "$SDRP")
    fi
}

# ---------------------------------------------------------------------------
log "sweep: ${#SERIES[@]} series x $N_SEEDS seeds, depth $DEPTH"
log "ace -> devices $ACE_DEVICES ($ACE_JOBS jobs)   ideal <= $SMALL_MAX -> devices $IDEAL_DEVICES ($IDEAL_JOBS jobs)"
START=$(date +%s)

for cfg in "${SERIES[@]}"; do
    # shellcheck disable=SC2086
    set -- $cfg
    width="$1"; lrc="$2"; lrr="$3"
    out="$OUT_ROOT/w$width"
    ideal_profile "$width"
    log "series width=$width lrc=$lrc lrr=$lrr  ideal: engine=$I_ENGINE jobs=$I_JOBS devices=${I_DEVS[*]} ${I_EXTRA[*]}"

    # --- stage 1: ACE, backgrounded so stage 2 can consume as it goes -------
    ACE_PIDS=()
    for ((j = 0; j < ACE_JOBS; j++)); do
        "$PY" -u "$SCRIPT" ace \
            --device "${ACE_DEV_LIST[$(( j % ${#ACE_DEV_LIST[@]} ))]}" \
            --width "$width" --depth "$DEPTH" --lrc "$lrc" --lrr "$lrr" \
            --seeds "$SEEDS" --out "$out" \
            >> "$OUT_ROOT/logs/ace-w$width.log" 2>&1 &
        ACE_PIDS+=($!)
    done

    # --- stage 2: ideal, looping until this series is complete --------------
    stall=0
    refused=0
    while :; do
        before="$(count_done "$out/xeb")"

        ipids=()
        for ((j = 0; j < I_JOBS; j++)); do
            "$PY" -u "$SCRIPT" ideal \
                --device "${I_DEVS[$(( j % ${#I_DEVS[@]} ))]}" \
                --engine "$I_ENGINE" "${I_EXTRA[@]}" \
                --seeds "$SEEDS" --out "$out" \
                >> "$OUT_ROOT/logs/ideal-w$width.log" 2>&1 &
            ipids+=($!)
        done
        for pid in "${ipids[@]}"; do
            rc=0; wait "$pid" || rc=$?
            [ "$rc" -eq 2 ] && refused=1
        done

        if [ "$refused" -eq 1 ]; then
            log "  w$width: reference refused, not enough host RAM (see logs/ideal-w$width.log)"
            log "  w$width: letting the ACE pass finish; circuits + counts stay in $out/ace"
            break
        fi

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
printf 'width\tlrc\tlrr\tB_to_B\tn\tstats\tmean_xeb\tstdev_xeb\tmean_xeb_linear\tswap_out_gib\n' > "$SUMMARY"

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

    # bulk_to_boundary is column 8, xeb_ace 12, stats_mode 23, xeb_linear 24,
    # swap_out_gib 27
    # (see FIELDS in nn_qab.py). mean_xeb uses whichever estimator the row
    # was scored with; mean_xeb_linear is the same estimator at every width,
    # so it is the column to read across the exact -> linear switch.
    awk -F',' -v w="$width" -v c="$lrc" -v r="$lrr" '
        NR > 1 {
            sub(/\r$/, "")
            b = $8; n++; s += $12; q += $12 * $12
            if ($24 != "") { nl++; l += $24 }
            so += $27
            t = ($23 == "") ? "exact" : $23   # rows from before stats_mode
            mode = (mode == "" || mode == t) ? t : "mixed"
        }
        END {
            if (n == 0) exit
            m = s / n
            sd = (n > 1) ? sqrt((q - n * m * m) / (n - 1)) : 0
            lin = (nl == n) ? sprintf("%.10f", l / n) : "NA"
            printf "%s\t%s\t%s\t%s\t%d\t%s\t%.10f\t%.10f\t%s\t%.1f\n", \
                   w, c, r, b, n, mode, m, sd, lin, so
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