#!/usr/bin/env bash
# nighthawk_fullsweep.sh -- ACE seam-budget sweep at the experiment's 61 qubits, CPU only,
# sized for a 330 GB / 96-thread EPYC.
#
# Measured per worker (single ACE run, paper circuits):
#   lrc,lrr in {4,5,6}, any depth up to 20 ......... ~40 MB, no growth over many twirls
#   lrc or lrr = 7, depth >= 8 ...................... dense units of many GB (the OOM)
#
# So two pools:
#   cheap  5:5 6:6           depths 4 6 8 12 16 20   18 workers each, 4 GB each
#   heavy  7:7 7:4 4:7       depths 4 6 8 12          1 worker each, 50 GB each
# A heavy worker already spreads one dense unit over 10-36 cores on its own (btop:
# ~230 threads, 1000-3600% CPU), so a second worker per setting only adds contention.
#
# Memory control is --max-rss-gb: a watchdog in each worker reads its resident memory;
# above the cap the point is marked failed (with the GB figure) and the worker exits,
# before the kernel OOM killer acts. A failed heavy point marks where that seam budget
# stops fitting, which is itself a result. Worst case 36 x 4 + 3 x 50 = 294 GB.
# Not used, and why:
#   ulimit -v         caps reserved address space; Qrack's thread pool reserves far
#                     more than it touches -> std::future_error, failures at d=4
#   QRACK_MAX_CPU_QB  this host's Qrack build trips its fidelity guard under the cap
#                     on points that need ~40 MB -> RuntimeError on seam 5 and 6
# Stop any older sweep yourself before starting this one.
#
# Rerun to resume. Retry failed points with: EXTRA=--retry-failed ./nighthawk_fullsweep.sh
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

#        setting depths               workers max_rss_gb
launch   5:5     "4 6 8 12 16 20"     18      4
launch   6:6     "4 6 8 12 16 20"     18      4
launch   7:7     "4 6 8 12"           1       50
launch   7:4     "4 6 8 12"           1       50
launch   4:7     "4 6 8 12"           1       50

[ -s hwxeb.json ] || nohup python3 nighthawk_qrack.py hwxeb --cpu \
    --out hwxeb.json --cache xeb_cache > hwxeb.txt 2>&1 &

wait
echo "== all settings finished =="
for st in 5:5 6:6 7:7 7:4 4:7; do
  c=${st%:*}; r=${st#*:}; out=$(name "$c" "$r")
  python3 nighthawk_qrack.py run --backend ace --lrc "$c" --lrr "$r" --out "$out.jsonl" --summarize \
      > "$out.summary.txt" 2>&1
  echo "--- lrc=$c lrr=$r"; grep -E "^mirror|marked failed|^#   " "$out.summary.txt" | head -20
done
