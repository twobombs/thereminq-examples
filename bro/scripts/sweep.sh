#!/usr/bin/env bash
#
# synth_sweep.sh -- measure F_elide where BOTH routes run.
#
# The reference family cannot yield this number: its exact monolithic
# amplitude is 2^214. On a circuit small enough that n+t fits, exact and
# patched amplitudes are computed from the SAME samples, so
#
#   XEB_exact   = F_true
#   XEB_patched = F_true * F_elide
#   ratio       = F_elide
#
# F_true cancels, which is why each circuit is sampled once and the result
# reused for both arms. Never regenerate samples between them.
#
# Sizing matters more than anything else. Cost is 2^(n+t) in BOTH memory and
# time, and the exact arm is monolithic -- no patches to spread across
# workers, only samples. n+t=28 is ~25 min per sample; n+t=22 is ~23 s. The
# defaults sit near 22 for that reason.
#
# The doping levels are independent, so circuits run concurrently; within
# each, the exact arm must finish before its patched arms start.
#
# Usage:
#   ./synth_sweep.sh --dry-run          # cost estimate, runs nothing
#   ./synth_sweep.sh
#   ./synth_sweep.sh --shots 100 --t-list 6,9,11 --jobs 3
#
set -uo pipefail

VIZ="${VIZ:-python bro-xeb-viz.py}"
SAMPLER="${SAMPLER:-python dcs_post_select_paralel.py}"
QRACKLIB="${QRACKLIB:-}"
# No ancillas by default. The synth generator's checks measure Z-parity over
# a data span, but every layer applies H to the data qubits, so that parity is
# not a stabilizer of the state when the ancilla reads it -- the outcome is a
# coin flip and post-selection discards (1/2)^NA of the shots while projecting
# onto nothing meaningful. Measured: 3/20 accepted with NA=3, i.e. 12.5%.
#
# F_elide is a statement about elision error and does not involve
# post-selection at all, so the checks are simply dropped. Set --n-ancilla
# only if you also fix the generator to make its checks code-preserving.
ND=13; NA=0
T_LIST="6,9,11"             # doping -> n+t = 19, 22, 24
G_LIST="20,17,14"           # patch targets: shallow to deep elision
SHOTS=50
JOBS=3                      # concurrent circuits
WORKERS=8                   # workers for the patched arm (processes)
EXACT_WORKERS=1             # exact arm: threads, unvalidated -- keep at 1
VERIFY=1                    # recompute the exact arm once and compare
OUT="synth_runs"
MAXCPUQB=""                 # QRACK_MAX_CPU_QB: low value forces GPU handoff
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qrack-lib) QRACKLIB="$2"; shift 2 ;;
    --shots)     SHOTS="$2";    shift 2 ;;
    --t-list)    T_LIST="$2";   shift 2 ;;
    --g-list)    G_LIST="$2";   shift 2 ;;
    --n-data)    ND="$2";       shift 2 ;;
    --n-ancilla) NA="$2";       shift 2 ;;
    --jobs)      JOBS="$2";     shift 2 ;;
    --workers)   WORKERS="$2";  shift 2 ;;
    --exact-workers) EXACT_WORKERS="$2"; shift 2 ;;
    --out)       OUT="$2";      shift 2 ;;
    --max-cpu-qb) MAXCPUQB="$2"; shift 2 ;;
    --gpu)        MAXCPUQB="8";  shift ;;
    --no-verify) VERIFY=0;      shift ;;
    --dry-run)   DRY=1;         shift ;;
    -h|--help)   sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Pin the Qrack build before ANY child starts. pyqrack resolves
# libqrack_pinvoke.so through ctypes at import and silently falls back to the
# copy bundled in dist-packages when PYQRACK_SHARED_LIB_PATH is unset. The
# sampler is a separate process and inherits nothing not exported here; the
# bundled binary is built for instructions this hardware may lack, which
# showed up as SIGILL in every sampler worker until it was pinned.
# ---------------------------------------------------------------------------
resolve_qrack_lib() {
  local want="${1:-}"
  [[ -z "$want" && -n "${PYQRACK_SHARED_LIB_PATH:-}" ]] && want="$PYQRACK_SHARED_LIB_PATH"
  if [[ -z "$want" ]]; then
    for c in /usr/local/lib/qrack/libqrack_pinvoke.so \
             /usr/local/lib/libqrack_pinvoke.so \
             /usr/lib/qrack/libqrack_pinvoke.so \
             /usr/lib/libqrack_pinvoke.so; do
      [[ -f "$c" ]] && { want="$c"; break; }
    done
  fi
  [[ -d "$want" ]] && want="$want/libqrack_pinvoke.so"
  if [[ -n "$want" && -f "$want" ]]; then
    export PYQRACK_SHARED_LIB_PATH="$want"
    echo "qrack library: $PYQRACK_SHARED_LIB_PATH"
  else
    echo "WARNING: no libqrack_pinvoke.so found; pyqrack will load its" >&2
    echo "         bundled copy, which may not match this CPU." >&2
  fi
  export QRACK_MAX_PAGING_QB="${QRACK_MAX_PAGING_QB:--1}"

  # QRACK_MAX_CPU_QB is the widest state a CPU engine may hold; anything
  # wider is handed to the accelerator. So it cuts both ways:
  #   -1  unlimited CPU width -> the GPU is never used  (what NOT to set)
  #    8  narrow CPU  -> a 13-qubit patch goes to the GPU (--gpu)
  # Whether that is faster is an open question: the states here are ~64 KB
  # and the work is 2^t separate small operations, so per-kernel launch
  # overhead may swamp the transfer. Measure it, do not assume.
  if [[ -n "${2:-}" ]]; then
    export QRACK_MAX_CPU_QB="$2"
    echo "QRACK_MAX_CPU_QB=$2 (states wider than this go to the GPU)"
  elif [[ "${QRACK_MAX_CPU_QB:-}" == "-1" || "${QRACK_MAX_CPU_QB:-}" == "0" ]]; then
    echo "note: QRACK_MAX_CPU_QB=$QRACK_MAX_CPU_QB pins work to the CPU;" >&2
    echo "      unset it, or pass --gpu, to let Qrack use the accelerator." >&2
  fi
}

resolve_qrack_lib "$QRACKLIB" "$MAXCPUQB"
mkdir -p "$OUT"
N=$((ND + NA))

IFS=',' read -ra TS <<< "$T_LIST"
IFS=',' read -ra GS <<< "$G_LIST"

# --- cost estimate, calibrated on a measured point -------------------------
echo
printf '%-5s %-5s %-10s %-12s %s\n' "t" "n+t" "s/sample" "exact arm" "note"
for T in "${TS[@]}"; do
  NT=$((N + T))
  EST=$(awk -v nt="$NT" 'BEGIN{printf "%.2f", 375.0*2^(nt-26)}')
  TOT=$(awk -v e="$EST" -v s="$SHOTS" 'BEGIN{printf "%.1f", e*s/60}')
  FLAG=""
  awk -v t="$TOT" 'BEGIN{exit !(t>90)}' && FLAG="SLOW: lower --t-list or --shots"
  printf '%-5s %-5s %-10s %-12s %s\n' "$T" "$NT" "$EST" "${TOT} min" "$FLAG"
done
echo
echo "${#TS[@]} circuits x ${#GS[@]} cut depths = $(( ${#TS[@]} * ${#GS[@]} )) rows"
echo "$JOBS circuits concurrently, $WORKERS workers each; wall ~= slowest circuit"
echo "estimates are serial per-sample, calibrated on n+t=26 at 375 s/sample"
if [[ -n "$MAXCPUQB" ]]; then
  echo
  echo "GPU mode: states wider than $MAXCPUQB qubits are handed to the device."
  echo "At $N qubits a state is $(( (2**N * 8) / 1024 )) KB, and the work is 2^t such"
  echo "states, so this may well be slower than CPU. Compare against a run"
  echo "without --gpu on the same --t-list before drawing a conclusion."
fi

if [[ $DRY -eq 1 ]]; then
  echo; echo "dry run: nothing executed."
  exit 0
fi

# --- one circuit end to end, run as a background job -----------------------
one_circuit() {
  local T="$1"
  # one row set per doping level, even if the caller loops oddly
  local STAMP="$OUT/.done_t$T"
  [[ -e "$STAMP" ]] && { echo "t=$T: already recorded, skipping"; return; }
  : > "$STAMP"
  local Q="$OUT/synth_t$T.qasm"
  local R="$OUT/synth_t$T.result.json"
  local E="$OUT/synth_t$T.exact.npy"
  local L="$OUT/synth_t$T.log"

  $VIZ synth --n-data $ND --n-ancilla $NA --t-gates "$T" --out "$Q" >>"$L" 2>&1 \
    || { echo "t=$T: synth failed, see $L"; return; }
  [[ -s "$R" ]] || $SAMPLER "$Q" --n-data $ND --n-ancilla $NA --shots "$SHOTS" \
      --workers 16 --chunksize 8 --out "$R" >>"$L" 2>&1 \
    || { echo "t=$T: sampler failed, see $L"; return; }

  # A low acceptance rate silently shrinks the sample set, and a handful of
  # samples produces an XEB dominated by noise that still looks like a number.
  local ACC
  ACC=$(python -c "import json;d=json.load(open('$R'));print(d['accepted'])" 2>/dev/null)
  if [[ -n "$ACC" ]] && (( ACC < SHOTS )); then
    echo "t=$T: only $ACC/$SHOTS shots accepted -- XEB from $ACC samples has"
    echo "      standard error ~$(awk -v a="$ACC" 'BEGIN{printf "%.2f", 1.41/sqrt(a)}')."
    (( ACC < 10 )) && { echo "      too few to measure anything; skipping."; return; }
  fi

  # Exact arm: SERIAL by default. The monolithic path parallelises with
  # threads sharing one Qrack library instance, which has never been shown
  # safe -- and a corrupted exact reference is undetectable downstream except
  # as an impossible XEB. The patched path uses processes and is fine.
  [[ -s "$E" ]] || $VIZ probs "$Q" "$R" --n-data $ND --n-ancilla $NA \
      --method permutation --workers "$EXACT_WORKERS" --out "$E" >>"$L" 2>&1 \
    || { echo "t=$T: exact probs failed, see $L"; return; }
  # Determinism check. The exact arm must be a pure function of (circuit,
  # samples); if a second computation disagrees, every downstream ratio is
  # meaningless and no amount of averaging fixes it. Cheap relative to the
  # run itself, and it caught a real problem once.
  if [[ "$VERIFY" == "1" ]]; then
    local E2="$OUT/synth_t$T.exact.verify.npy"
    rm -f "$E2"
    $VIZ probs "$Q" "$R" --n-data $ND --n-ancilla $NA \
        --method permutation --workers "$EXACT_WORKERS" --out "$E2" \
        >>"$L" 2>&1
    if [[ -s "$E2" ]]; then
      local DEV
      DEV=$(python - "$E" "$E2" <<'EOF'
import sys, numpy as np
a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
if a.shape != b.shape:
    print("shape-mismatch"); sys.exit()
den = np.maximum(np.abs(a), 1e-300)
print(f"{float(np.abs(a - b).max() / den.max()):.3e}")
EOF
)
      echo "  t=$T: exact arm reproducibility, max rel dev = $DEV"
      if [[ "$DEV" != "shape-mismatch" ]] && \
         awk -v d="$DEV" 'BEGIN{exit !(d+0 > 1e-4)}'; then
        echo "t=$T: EXACT ARM IS NOT DETERMINISTIC (dev $DEV). Two computations"
        echo "      of the same amplitudes disagree, so every F_elide below is"
        echo "      meaningless. Stop here and diagnose."
        return
      fi
    fi
  fi

  local XE
  XE=$($VIZ stats "$R" --amps "$E" 2>/dev/null | awk '/linear XEB/{print $4;exit}')
  [[ -n "$XE" ]] || { echo "t=$T: could not read exact XEB"; return; }
  if awk -v e="$XE" 'BEGIN{exit !(e+0 < -0.05)}'; then
    echo "t=$T: EXACT XEB = $XE, which is negative. Samples scored against"
    echo "      their own ideal distribution cannot give mean D*p < 1, so the"
    echo "      exact arm is wrong, not the circuit. Check --exact-workers 1,"
    echo "      then the sampler's acceptance rate in $L."
  fi

  for TG in "${GS[@]}"; do
    local P="$OUT/p_t${T}_g${TG}"
    $VIZ patch "$Q" --outdir "$P" --target "$TG" --cost-model nt \
        --n-data $ND --n-ancilla $NA >"$P.log" 2>&1 || continue
    local CUTS ELID A XP FE
    CUTS=$(awk '/cuts,/{print $1;exit}' "$P.log")
    ELID=$(awk '/cuts,/{print $3;exit}' "$P.log")
    A="$OUT/synth_t${T}_g${TG}.patched.npy"
    [[ -s "$A" ]] || $VIZ probs "$Q" "$R" --n-data $ND --n-ancilla $NA \
        --patches "$P/patch_manifest.json" --method permutation \
        --workers "$WORKERS" --out "$A" >>"$L" 2>&1 || continue
    XP=$($VIZ stats "$R" --amps "$A" 2>/dev/null | awk '/linear XEB/{print $4;exit}')
    # F_true cancels in the ratio, but only when the exact arm is well away
    # from zero; otherwise the quotient is noise divided by noise
    FE=$(awk -v e="$XE" -v p="$XP" 'BEGIN{
           if (e+0 < 0.05) print "unreliable(F_true~0)";
           else printf "%.4f", p/e }')
    printf '%s,%s,%s,%s,%s,%s,%s\n' "$T" "$TG" "$CUTS" "$ELID" "$XE" "$XP" "$FE" \
        >> "$OUT/felide.csv"
    printf '  t=%-3s target=%-3s cuts=%-3s elided=%-4s F_elide=%s\n' \
        "$T" "$TG" "$CUTS" "$ELID" "$FE"
  done
}

echo "t_gates,target,cuts,elided,XEB_exact,XEB_patched,F_elide" > "$OUT/felide.csv"
echo

running=0
for T in "${TS[@]}"; do
  one_circuit "$T" &
  running=$((running + 1))
  if (( running >= JOBS )); then
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  fi
done
wait

echo
echo "=== $OUT/felide.csv ==="
if command -v column >/dev/null 2>&1; then
  column -s, -t < "$OUT/felide.csv"
else
  awk -F, '{for(i=1;i<=NF;i++) printf "%-14s", $i; print ""}' "$OUT/felide.csv"
fi
echo
echo "F_elide near 1 at shallow elision, decaying smoothly as elided grows,"
echo "means patch-based verification has a working regime. Negative even at"
echo "the shallowest cut means it does not, and the -0.83 measured on the"
echo "117T rung is the general case rather than a result of over-cutting."
