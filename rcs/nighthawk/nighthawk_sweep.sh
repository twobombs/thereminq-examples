#!/usr/bin/env bash
# nighthawk_fullsweep.sh -- ACE seam-budget sweep at the experiment's 61 qubits, CPU only,
# sized for a 330 GB / 96-thread EPYC.
#
# Measured per worker (single ACE run, paper circuits):
#   lrc,lrr in {4,5,6}, any depth up to 20 ......... ~40 MB, no growth over many twirls
#   lrc or lrr = 7, depth >= 8 ...................... dense units of many GB (the OOM)
#
# So two pools:
#   cheap  5:5 6:6           depths 4 6 8 12 16 20   18 workers each
#   heavy  7:7 7:4 4:7       depths 4 6 8 12          1 worker each, capped
# A heavy worker already spreads one dense unit over 10-36 cores on its own (seen in
# btop: ~230 threads, 1000-3600% CPU), so a second worker per setting only adds
# contention. Stop any older sweep yourself before starting this one.
# A capped worker that needs more than QRACK_MAX_CPU_QB qubits in one dense unit gets
# a clean Qrack error and the point is marked failed (not retried): that marks where
# a seam budget stops fitting in memory, which is itself a result.
# ulimit -v is the hard backstop per worker, so one runaway cannot take the host down.
#
# Rerun to resume. Retry failed points with: EXTRA=--retry-failed ./nighthawk_fullsweep.sh
set -u
cd "$(dirname "$0")"

mkdir -p /tmp/no-icd
export OCL_ICD_VENDORS=/tmp/no-icd           # no GPU contexts in any process
export MALLOC_ARENA_MAX=4                    # keep reserved address space small under ulimit -v
EXTRA=${EXTRA:-}

name() { [ "$1" = "$2" ] && echo "clean_ace_seam$1" || echo "clean_ace_c$1r$2"; }

# launch SETTING DEPTHS WORKERS MAX_QB MEM_GB
launch() {
  local c=${1%:*} r=${1#*:} out
  out=$(name "$c" "$r")
  (
    ulimit -v $(( $5 * 1024 * 1024 ))
    QRACK_MAX_CPU_QB=$4 nohup python3 nighthawk_qrack.py run --backend ace --families mirror \
        --sizes 61 --depths $2 --lrc "$c" --lrr "$r" --cpu $EXTRA \
        --out "$out.jsonl" --gpus 0 --per-gpu "$3" --stagger 2 > "$out.txt" 2>&1
  ) &
  echo "lrc=$c lrr=$r: $3 workers, depths [$2], cap ${4} qubits / ${5} GB each -> $out.txt"
  sleep 10
}

#        setting depths               workers max_qb mem_gb
launch   5:5     "4 6 8 12 16 20"     18      28     8
launch   6:6     "4 6 8 12 16 20"     18      28     8
launch   7:7     "4 6 8 12"           1       31     40
launch   7:4     "4 6 8 12"           1       31     40
launch   4:7     "4 6 8 12"           1       31     40
# limits: 36 x 8 GB + 3 x 40 GB; real use is ~40 MB per cheap worker and 5-9 GB per
# heavy worker (measured), so about 30-40 GB in practice on the 330 GB box.

[ -s hwxeb.json ] || ( ulimit -v $(( 16 * 1024 * 1024 )); nohup python3 nighthawk_qrack.py hwxeb --cpu \
    --out hwxeb.json --cache xeb_cache > hwxeb.txt 2>&1 ) &

wait
echo "== all settings finished =="
for st in 5:5 6:6 7:7 7:4 4:7; do
  c=${st%:*}; r=${st#*:}; out=$(name "$c" "$r")
  python3 nighthawk_qrack.py run --backend ace --lrc "$c" --lrr "$r" --out "$out.jsonl" --summarize \
      > "$out.summary.txt" 2>&1
  echo "--- lrc=$c lrr=$r"; grep -E "^mirror|marked failed|^#   " "$out.summary.txt" | head -20
done
