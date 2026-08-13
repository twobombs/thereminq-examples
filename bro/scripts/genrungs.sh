#!/usr/bin/env bash
#
# genrungs.sh -- generate the doping ladder, then prove it is what it claims.
#
# Rungs are nested: each is a superset of the vendor's 75T doping and a subset
# of the full 468. That makes doping level the only variable along the chain,
# so any drift in F_elide or in certificate accuracy is attributable to t
# rather than to the keep-set jumping around.
#
# Replicates are independent draws at ONE doping level. The vendor's own files
# share no doped positions between levels, so that is how they built theirs;
# the spread across replicates measures sensitivity to doping *placement* at
# fixed doping *level*. Without it, drift along the rungs is confounded.
#
# Every generated file is verified, not assumed:
#   - t is exactly what was asked for
#   - the interaction graph is byte-identical to the source
#   - rungs contain all 75 of the vendor's doped positions
#
# Feasibility: rungs stop at the doping where an exact reference still fits in
# memory. Beyond that there is no exact column, so no F_elide, so no rung.
#
# Usage:
#   ./genrungs.sh                       # default ladder, verify, report
#   ./genrungs.sh --chi-max 24          # tighter memory budget
#   ./genrungs.sh --rungs 85,100,117 --rep-t 100 --seeds 3
#   ./genrungs.sh --verify-only         # check existing files, generate nothing
#
set -uo pipefail

VIZ="${VIZ:-python bro-xeb-viz.py}"
QDIR="${QDIR:-./qasm_circuits}"
SOURCE="$QDIR/nq70_depth70_checks27_doped_checks.qasm"
ANCHOR="$QDIR/nq70_depth70_checks27_doped_75T_checks.qasm"
CHI_MAX=27
RUNGS=""
REP_T=100
SEEDS=3
VERIFY_ONLY=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chi-max)     CHI_MAX="$2"; shift 2 ;;
    --rungs)       RUNGS="$2";   shift 2 ;;
    --rep-t)       REP_T="$2";   shift 2 ;;
    --seeds)       SEEDS="$2";   shift 2 ;;
    --source)      SOURCE="$2";  shift 2 ;;
    --anchor)      ANCHOR="$2";  shift 2 ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    --force)       FORCE=1;      shift ;;
    -h|--help)     sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

for f in "$SOURCE" "$ANCHOR"; do
  [[ -s "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done

ANCHOR_T=$($VIZ tcount "$ANCHOR" 2>/dev/null | awk '/count t =/ {print $NF; exit}')
SOURCE_T=$($VIZ tcount "$SOURCE" 2>/dev/null | awk '/count t =/ {print $NF; exit}')

# largest doping whose exact reference still fits the budget: 0.23t <= CHI_MAX
T_MAX=$(awk -v c="$CHI_MAX" 'BEGIN{printf "%d", c/0.23}')

if [[ -z "$RUNGS" ]]; then
  # evenly spaced between the anchor and the feasibility ceiling
  RUNGS=$(awk -v a="$ANCHOR_T" -v m="$T_MAX" 'BEGIN{
    n=3; for(i=1;i<=n;i++){ printf "%d%s", a+(m-a)*i/n, (i<n?",":"") }}')
fi

printf 'source  %s  (t=%s)\n' "$(basename "$SOURCE")" "$SOURCE_T"
printf 'anchor  %s  (t=%s)\n' "$(basename "$ANCHOR")" "$ANCHOR_T"
printf 'budget  2^%s  ->  t <= %s\n' "$CHI_MAX" "$T_MAX"
printf 'rungs   %s\n' "$RUNGS"
printf 'reps    t=%s x %s seeds\n\n' "$REP_T" "$SEEDS"

FAIL=0

# ---------------------------------------------------------------- generation
gen() {  # out_path  extra_args...
  local out="$1"; shift
  if [[ -s "$out" && $FORCE -eq 0 ]]; then
    echo "  exists, keeping $(basename "$out")"
    return 0
  fi
  $VIZ dedope "$SOURCE" --out "$out" "$@" >/dev/null 2>&1 \
    || { echo "  FAILED to generate $(basename "$out")"; FAIL=1; return 1; }
  echo "  wrote $(basename "$out")"
}

if [[ $VERIFY_ONLY -eq 0 ]]; then
  echo "--- rungs (nested on the anchor) ---"
  IFS=',' read -ra RUNG_LIST <<< "$RUNGS"
  for t in "${RUNG_LIST[@]}"; do
    if (( t > T_MAX )); then
      echo "  SKIP t=$t: 0.23t exceeds the 2^$CHI_MAX budget, no exact reference"
      continue
    fi
    gen "$QDIR/rung_${t}T_checks.qasm" --keep-from "$ANCHOR" --keep "$t"
  done

  echo
  echo "--- replicates (independent draws at t=$REP_T) ---"
  if (( REP_T > T_MAX )); then
    echo "  SKIP: t=$REP_T exceeds the budget"
  else
    for s in $(seq 1 "$SEEDS"); do
      gen "$QDIR/rep${REP_T}T_s${s}_checks.qasm" --keep "$REP_T" --seed "$s"
    done
  fi
  echo
fi

# -------------------------------------------------------------- verification
# One `compare` per pair, captured whole. Piping into `grep -q` would close
# the pipe on first match, kill the producer with SIGPIPE, and -o pipefail
# would then report a successful match as a failed check.
CMP=""
compare_pair() { CMP=$($VIZ compare "$1" "$2" 2>/dev/null); }
cmp_shared()   { awk '/doped positions/ {split($3,a,"/"); print a[1]; exit}' <<< "$CMP"; }
cmp_skeleton() { case "$CMP" in *IDENTICAL*) echo ok ;; *) echo DIFFER ;; esac; }

echo "--- verification ---"
printf '%-34s %6s %9s %9s %8s %s\n' file t skeleton has_anchor cost note

shopt -s nullglob
for f in "$QDIR"/rung_*_checks.qasm "$QDIR"/rep*_checks.qasm; do
  base=$(basename "$f")
  t=$($VIZ tcount "$f" 2>/dev/null | awk '/count t =/ {print $NF; exit}')

  compare_pair "$f" "$SOURCE"
  skel=$(cmp_skeleton)
  sh_src=$(cmp_shared)
  [[ "$skel" == "ok" ]] || FAIL=1

  compare_pair "$f" "$ANCHOR"
  sh=$(cmp_shared)
  note=""
  if [[ "$base" == rung_* ]]; then
    if [[ "$sh" == "$ANCHOR_T" ]]; then
      nested="${sh}/${ANCHOR_T}"
    else
      nested="${sh}/${ANCHOR_T}"
      note="NOT nested on the anchor -- regenerate with --keep-from"
      FAIL=1
    fi
  else
    nested="${sh}/${ANCHOR_T}"
    note="independent draw (expected)"
  fi

  # the source must contain every rung, or the rung is not from this family
  [[ "$sh_src" == "$t" ]] || { note="${note:+$note; }NOT a subset of the source"; FAIL=1; }

  # verifies fine but exceeds the budget: felide.sh will skip it, so say so
  cost=$(awk -v t="$t" 'BEGIN{printf "2^%.1f", 0.23*t}')
  if (( t > T_MAX )); then
    note="${note:+$note; }over budget, felide will skip"
  fi
  printf '%-34s %6s %9s %9s %8s %s\n' "$base" "$t" "$skel" "$nested" "$cost" "$note"
done

echo
if [[ $FAIL -eq 0 ]]; then
  echo "all generated circuits verified: same skeleton, rungs nested on the"
  echo "anchor, every rung a subset of the source."
  echo
  echo "next: ./felide.sh --dry-run"
else
  echo "VERIFICATION FAILED -- do not run the ladder on these files."
  echo "A rung that is not nested still measures something, but drift along"
  echo "the ladder would then confound doping level with doping placement,"
  echo "which is the one thing the nesting exists to separate."
  exit 1
fi
