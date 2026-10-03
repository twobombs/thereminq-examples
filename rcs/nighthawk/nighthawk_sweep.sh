#!/usr/bin/env bash
# nighthawk_fullsweep.sh -- ACE seam-budget sweep at the experiment's 61 qubits, CPU only,
# sized for a 330 GB / 96-thread EPYC.
#
# Measured per worker (single ACE run, paper circuits):
#   lrc,lrr in {4,5,6}, any depth up to 20 ......... ~40 MB, no growth over many twirls
#   lrc or lrr = 7, depth >= 8 ...................... dense units of many GB (the OOM)
#
# Memory rule: BUDGET_GB (default 300) divided by the number of workers running at once
# is each worker's --max-rss-gb. A watchdog in each worker reads its resident memory;
# above the cap the point is marked failed (with the GB figure) and the worker exits,
# before the kernel OOM killer acts.
#
#   phase 1  cheap 5:5 6:6 (18 workers each) + heavy 7:7 7:4 4:7 (1 each) = 39 workers
#            -> 300/39 = 7.7 GB each
#   phase 2  heavy points that failed in phase 1, alone: 3 workers -> 100 GB each
# A heavy point that still fails in phase 2 marks where that seam budget stops fitting
# in memory, which is itself a result. A heavy worker spreads one dense unit over 10-36
# cores on its own, so one worker per heavy setting is enough.
#
# Not used, and why:
#   ulimit -v         caps reserved address space; Qrack's thread pool reserves far
#                     more than it touches -> std::future_error, failures at d=4
#   QRACK_MAX_CPU_QB  this host's Qrack build trips its fidelity guard under the cap
#                     on points that need ~40 MB -> RuntimeError on seam 5 and 6
# Stop any older sweep yourself before starting this one.
#
# Rerun to resume. Retry every failed point with: EXTRA=--retry-failed ./nighthawk_fullsweep.sh
# Other budget: BUDGET_GB=250 ./nighthawk_fullsweep.sh
set -u
cd "$(dirname "$0")"

mkdir -p /tmp/no-icd
export OCL_ICD_VENDORS=/tmp/no-icd           # no GPU contexts in any process
export MALLOC_ARENA_MAX=4                    # fewer glibc arenas across ~230 threads per worker
EXTRA=${EXTRA:-}

name() { [ "$1" = "$2" ] && echo "clean_ace_seam$1" || echo "clean_ace_c$1r$2"; }

# launch SETTING DEPTHS WORKERS MAX_RSS_GB
launch() {
  local c=${1%:*} r=${1#*:} out
  out=$(name "$c" "$r")
  nohup python3 nighthawk_qrack.py run --backend ace --families mirror \
      --sizes 61 --depths $2 --lrc "$c" --lrr "$r" --cpu --max-rss-gb "$4" $EXTRA \
      --out "$out.jsonl" --gpus 0 --per-gpu "$3" --stagger 2 > "$out.txt" 2>&1 &
  echo "lrc=$c lrr=$r: $3 workers, depths [$2], max ${4} GB resident each -> $out.txt"
  sleep 10
}

BUDGET_GB=${BUDGET_GB:-300}
CHEAP="5:5 6:6";      CHEAP_DEPTHS="4 6 8 12 16 20"; CHEAP_W=18
HEAVY="7:7 7:4 4:7";  HEAVY_DEPTHS="4 6 8 12";       HEAVY_W=1

n1=$(( $(echo $CHEAP | wc -w) * CHEAP_W + $(echo $HEAVY | wc -w) * HEAVY_W ))
cap1=$(awk -v b="$BUDGET_GB" -v n="$n1" 'BEGIN{printf "%.1f", b/n}')
echo "== phase 1: $n1 workers, $BUDGET_GB GB / $n1 = $cap1 GB resident each =="
for st in $CHEAP; do launch "$st" "$CHEAP_DEPTHS" "$CHEAP_W" "$cap1"; done
for st in $HEAVY; do launch "$st" "$HEAVY_DEPTHS" "$HEAVY_W" "$cap1"; done

[ -s hwxeb.json ] || nohup python3 nighthawk_qrack.py hwxeb --cpu \
    --out hwxeb.json --cache xeb_cache > hwxeb.txt 2>&1 &

wait

n2=$(( $(echo $HEAVY | wc -w) * HEAVY_W ))
cap2=$(awk -v b="$BUDGET_GB" -v n="$n2" 'BEGIN{printf "%.1f", b/n}')
echo "== phase 2: failed heavy points alone, $n2 workers, $BUDGET_GB GB / $n2 = $cap2 GB each =="
EXTRA="$EXTRA --retry-failed"
for st in $HEAVY; do launch "$st" "$HEAVY_DEPTHS" "$HEAVY_W" "$cap2"; done
wait
echo "== all settings finished =="
for st in $CHEAP $HEAVY; do
  c=${st%:*}; r=${st#*:}; out=$(name "$c" "$r")
  python3 nighthawk_qrack.py run --backend ace --lrc "$c" --lrr "$r" --out "$out.jsonl" --summarize \
      > "$out.summary.txt" 2>&1
  echo "--- lrc=$c lrr=$r"; grep -E "^mirror|marked failed|^#   " "$out.summary.txt" | head -20
done
