#!/usr/bin/env bash
# nighthawk_sweep.sh -- ACE mirror sweep at the experiment's 61 qubits, CPU only,
# sized for a 310 GiB / 96-thread EPYC.
#
# What changed since the untiled seam-budget sweep, and why:
#   * Untiled ACE is no longer swept. With the device placement, mirror survival at n = 61
#     was 0.01-0.08 at d = 4 and below the shot floor from d = 6, with every flip confined to
#     the layout's seam column: there is nothing left to measure. Every setting below is
#     TILED (--ace-tiling, version 3), which on the fxeb data at n = 27-29 removed every
#     coupler across simulators and raised the XEB 3-40x at unchanged memory.
#   * 4:4 is the reference setting (best tiled layout at n = 27); 5:5 and 6:6 stay as
#     neighbours. 4:4:noed repeats 4:4 with ACE's seam error detection off, which gave a
#     small constant XEB gain (+0.010 +/- 0.003) at n = 27-28 without changing the decay.
#   * Depths follow the release in two stages: stage 1 = 4..20 (step 2), stage 2 = the deep
#     tail 24..40, run only for settings whose pooled d = 20 survival is still well above the
#     shot floor (p_min = 9 / total shots; 4.2e-5 at d = 20 with the paper budgets).
#   * The old heavy untiled settings (lrc or lrr = 7, dense units of many GB) are replaced by
#     one structural candidate: the layout with the fewest couplers across simulators whose
#     internal simulators stay <= SPLIT_WIDTH qubits (--ace-max-width, tiled). Fewer couplers
#     through seams is the only lever left (no ACE setting lowers the cost per seam coupler);
#     a two-way split is estimated at b ~ 0.2 per cycle against ~1.0 for tiled 4:4.
#   * Memory is coordinated by the shared ledger (--mem-budget-gb): each worker reserves its
#     estimated peak before taking a point, so settings and phases no longer split a fixed
#     budget by hand. --max-rss-gb stays as the per-worker backstop (the watchdog marks the
#     point failed with its GB figure and exits before the kernel OOM killer acts).
#
# Measured per worker: tiled 4:4-6:6 simulators are <= 25 qubits wide, ~1-1.5 GiB per worker
# including Python. The split candidate is estimated by the ledger at ~257 GiB (4 x 2^33
# amplitudes worst case) and therefore runs alone.
#
# Not used, and why:
#   ulimit -v         caps reserved address space; Qrack's thread pool reserves far
#                     more than it touches -> std::future_error, failures at d=4
#   QRACK_MAX_CPU_QB  this host's Qrack build trips its fidelity guard under the cap
#                     -> RuntimeError on points that need ~40 MB
#
# Rerun to resume (config tags do not include depths or sizes).
#   Retry failed points:      EXTRA=--retry-failed ./nighthawk_sweep.sh
#   Other budget:             BUDGET_GB=250 ./nighthawk_sweep.sh
#   Only some phases:         PHASES="1 2" ./nighthawk_sweep.sh     (1 tiled stage 1, 2 deep tail, 3 split)
#   Other settings:           SETTINGS="4:4 4:4:noed 4:3" ./nighthawk_sweep.sh
#   Force the deep tail:      STAGE2=always ./nighthawk_sweep.sh
# Setting syntax: lrc:lrr[:noed][:rep][:torus][:untiled]
# The PyQrack version is in neither the config tag nor the records: after upgrading PyQrack,
# start a fresh set of output files (OUT_PREFIX=...) rather than resuming old ones.
set -u
cd "$(dirname "$0")"

mkdir -p /tmp/no-icd
export OCL_ICD_VENDORS=/tmp/no-icd           # no GPU contexts in any process
export MALLOC_ARENA_MAX=4                    # fewer glibc arenas across ~230 threads per worker
export NIGHTHAWK_TILE_CACHE=${NIGHTHAWK_TILE_CACHE:-$PWD/nighthawk_tiles.json}

EXTRA=${EXTRA:-}
BUDGET_GB=${BUDGET_GB:-300}                  # shared ledger budget, GiB (RAM is 310 GiB)
PHASES=${PHASES:-1 2 3}
SETTINGS=${SETTINGS:-4:4 4:4:noed 5:5 6:6}
WORKERS=${WORKERS:-12}                       # per setting; 27 points per setting in stage 1
CAP_GB=${CAP_GB:-8}                          # --max-rss-gb backstop for tiled workers
STAGE1="4 6 8 10 12 14 16 18 20"
STAGE2_DEPTHS="24 28 32 36 40"
STAGE2=${STAGE2:-auto}                       # auto | always | never
SPLIT_WIDTH=${SPLIT_WIDTH:-33}
SPLIT_TORUS=${SPLIT_TORUS:-any}              # torus only moves ACE's seams; the circuit is Nighthawk's
OUT_PREFIX=${OUT_PREFIX:-mirror61}

# flags for one setting (used for launching AND for --summarize, so the config tag matches)
flags() {
  local IFS=: f; set -- $1
  local c=$1 r=$2; shift 2
  local out="--lrc $c --lrr $r" tiled=1
  for f in "$@"; do
    case $f in
      noed)    out="$out --no-ace-error-detection" ;;
      rep)     out="$out --ace-boundary-rep" ;;
      torus)   out="$out --ace-torus" ;;
      untiled) tiled=0 ;;
      *) echo "unknown setting token '$f'" >&2; exit 2 ;;
    esac
  done
  [ $tiled = 1 ] && out="$out --ace-tiling"
  echo "$out"
}
name() { echo "${OUT_PREFIX}_c${1//:/_}" | sed 's/_c\([0-9]*\)_\([0-9]*\)/_c\1r\2/'; }

COMMON="--backend ace --families mirror --sizes 61 --cpu --mem-budget-gb $BUDGET_GB"

# launch NAME "FLAGS" "DEPTHS" WORKERS MAX_RSS_GB
launch() {
  nohup python3 nighthawk_qrack.py run $COMMON $2 --depths $3 --max-rss-gb "$5" $EXTRA \
      --out "$1.jsonl" --gpus 0 --per-gpu "$4" --stagger 2 >> "$1.txt" 2>&1 &
  echo "$1: $2 | depths [$3] | $4 workers, <= ${5} GiB resident each -> $1.txt"
  sleep 10
}

# pooled mirror survival at depth D from a run's records (main + worker files): "p N"
survival() {
  python3 - "$1" "$2" <<'EOF'
import glob, json, sys
stem, d = sys.argv[1], int(sys.argv[2])
h = n = 0.0
for f in [stem + ".jsonl"] + glob.glob(stem + ".w*.jsonl"):
    try:
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("family") == "mirror" and r.get("depth") == d and r.get("shots"):
                h += r["fidelity"] * r["shots"]; n += r["shots"]
    except OSError:
        pass
print(f"{h / n if n else 0.0:.3e} {int(n)}")
EOF
}

echo "== sweep: settings [$SETTINGS], phases [$PHASES], ledger budget $BUDGET_GB GiB =="

if [[ " $PHASES " == *" 1 "* ]]; then
  echo "== phase 1: tiled, stage-1 depths [$STAGE1] =="
  for st in $SETTINGS; do launch "$(name "$st")" "$(flags "$st")" "$STAGE1" "$WORKERS" "$CAP_GB"; done
fi

[ -s hwxeb.json ] || nohup python3 nighthawk_qrack.py hwxeb --cpu \
    --out hwxeb.json --cache xeb_cache > hwxeb.txt 2>&1 &

wait

if [[ " $PHASES " == *" 2 "* ]]; then
  echo "== phase 2: deep tail [$STAGE2_DEPTHS] (mode $STAGE2) =="
  for st in $SETTINGS; do
    nm=$(name "$st")
    read -r p n < <(survival "$nm" 20)
    keep=$(awk -v p="$p" -v n="$n" 'BEGIN{ print (n > 0 && p > 10 * 9 / n) ? 1 : 0 }')
    echo "$nm: pooled survival at d=20 = $p over $n shots (floor 9/N = $(awk -v n="$n" 'BEGIN{printf "%.1e", n ? 9/n : 0}'))"
    if [ "$STAGE2" = always ] || { [ "$STAGE2" = auto ] && [ "$keep" = 1 ]; }; then
      launch "$nm" "$(flags "$st")" "$STAGE2_DEPTHS" "$WORKERS" "$CAP_GB"
    else
      echo "  -> deep tail skipped: d=20 is not well above the floor (STAGE2=always to force)"
    fi
  done
  wait
fi

if [[ " $PHASES " == *" 3 "* ]]; then
  echo "== phase 3: two-way-split candidate, simulators <= $SPLIT_WIDTH qubits, tiled =="
  python3 nighthawk_qrack.py aceplan --n 61 --tiling --max-width "$SPLIT_WIDTH" --torus "$SPLIT_TORUS" \
      > "${OUT_PREFIX}_aceplan.txt" 2>&1
  grep -E "<- pick|^torus" "${OUT_PREFIX}_aceplan.txt"
  SPLIT_FLAGS="--ace-max-width $SPLIT_WIDTH --ace-torus-search $SPLIT_TORUS --ace-tiling"
  launch "${OUT_PREFIX}_split${SPLIT_WIDTH}" "$SPLIT_FLAGS" "$STAGE1" 1 "$BUDGET_GB"
  wait
fi

echo "== all phases finished =="
summ() {
  python3 nighthawk_qrack.py run --backend ace --families mirror $2 --out "$1.jsonl" --summarize \
      > "$1.summary.txt" 2>&1
  echo "--- $1"; grep -E "^mirror|marked failed|mirror fit|^#   " "$1.summary.txt" | head -24
}
for st in $SETTINGS; do summ "$(name "$st")" "$(flags "$st")"; done
[[ " $PHASES " == *" 3 "* ]] && summ "${OUT_PREFIX}_split${SPLIT_WIDTH}" \
    "--ace-max-width $SPLIT_WIDTH --ace-torus-search $SPLIT_TORUS --ace-tiling"
echo "compare with the device: python3 nighthawk_graph.py --hwxeb hwxeb.json   (views 1-4)"
