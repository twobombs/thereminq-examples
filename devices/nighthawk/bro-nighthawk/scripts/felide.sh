#!/usr/bin/env bash
#
# felide.sh -- measure the elision fidelity F_elide across a circuit family.
#
# For each circuit it runs the same samples against two references:
#
#   exact    p_ideal from full strong simulation
#   patched  prod_i p_i from the patched circuits
#
# Scoring one sample set against both gives
#
#   XEB_exact   = F_true
#   XEB_patched = F_true * F_elide
#
# so their ratio is F_elide with F_true cancelling out -- which is the point:
# it does not matter how good the sampler is, only that both columns use the
# SAME samples. That is why the script never regenerates samples for a circuit
# it already has.
#
# F_elide is the number that decides whether the patched route can reach the
# undoped circuit, where no exact reference exists. Watch whether it is stable
# across doping levels, not just whether it is large at one of them.
#
# Usage:
#   ./felide.sh                          # every *_checks.qasm in ./qasm_circuits
#   ./felide.sh --shots 2000
#   ./felide.sh --pattern '*75T*.qasm'
#   ./felide.sh --dry-run
#
set -uo pipefail

VIZ="${VIZ:-python bro-xeb-viz.py}"
SAMPLER="${SAMPLER:-python dcs_post_select_paralel.py}"
QDIR="${QDIR:-./qasm_circuits}"
OUTDIR="${OUTDIR:-./felide_runs}"
PATTERN='*_checks.qasm'
SHOTS=500
LIMIT=""
TARGET=27
REFCUT=""          # circuit whose cuts every rung reuses (default: the source)
SIMPRESET="near-clifford"
NOCLONE=""
SIMCLASS="QrackSimulator"
COSTMODEL="nt"
METHOD="chain"
QRACKLIB="${PYQRACK_SHARED_LIB_PATH:-}"
NDATA=70
NANC=27
DRY=0
CHI_MAX=27          # skip anything whose cheapest route exceeds 2^this

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pattern)  PATTERN="$2"; shift 2 ;;
    --shots)    SHOTS="$2";   shift 2 ;;
    --limit)    LIMIT="$2";   shift 2 ;;
    --target)   TARGET="$2";  shift 2 ;;
    --ref-cut)  REFCUT="$2";  shift 2 ;;
    --sim-preset) SIMPRESET="$2"; shift 2 ;;
    --no-clone)   NOCLONE="--no-clone"; shift ;;
    --sim-class)  SIMCLASS="$2"; shift 2 ;;
    --method)     METHOD="$2"; shift 2 ;;
    --qrack-lib)  QRACKLIB="$2"; shift 2 ;;
    --n-data)     NDATA="$2";    shift 2 ;;
    --n-ancilla)  NANC="$2";     shift 2 ;;
    --cost-model) COSTMODEL="$2"; shift 2 ;;
    --no-refcut)  REFCUT="-";    shift ;;
    --outdir)   OUTDIR="$2";  shift 2 ;;
    --chi-max)  CHI_MAX="$2"; shift 2 ;;
    --dry-run)  DRY=1;        shift ;;
    -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

resolve_qrack_lib "$QRACKLIB"

mkdir -p "$OUTDIR"

# --- one cut set for the whole ladder -------------------------------------
# Every rung shares the source's interaction graph exactly (genrungs.sh checks
# this), so the cuts found on the SOURCE apply unchanged everywhere. Holding
# them fixed is what makes F_elide comparable across rungs: the same edges, the
# same elided gates, only the doping changing. Letting each rung pick its own
# cost-balanced cuts would compare different elisions and the trend would mean
# nothing -- and a rung already under the target would get no cuts at all,
# reporting F_elide = 1 for a patching that never happened.
[[ -n "$REFCUT" ]] || REFCUT="$QDIR/nq70_depth70_checks27_doped_checks.qasm"
if [[ "$REFCUT" == "-" ]]; then
  # --no-refcut: cut each circuit on its own cost, rather than reusing a set
  # derived from a harder circuit. Right when there is no ladder to keep
  # comparable -- a single circuit measured against itself.
  REFMAN=""
fi
REFDIR="$OUTDIR/_refcut"
REFMAN="$REFDIR/patch_manifest.json"
if [[ -n "$REFMAN" && ! -s "$REFMAN" ]]; then
  if [[ ! -s "$REFCUT" ]]; then
    echo "reference circuit for cuts not found: $REFCUT" >&2
    echo "pass --ref-cut <qasm>" >&2
    exit 1
  fi
  echo "deriving the shared cut set from $(basename "$REFCUT") (target 2^$TARGET)"
  $VIZ patch "$REFCUT" --outdir "$REFDIR" --target "$TARGET" \
       --n-data "$NDATA" --n-ancilla "$NANC" || exit 1
  echo
fi
if [[ -n "$REFMAN" ]]; then
  CUTS=$(python -c "import json;print(len(json.load(open('$REFMAN'))['cuts']))" 2>/dev/null || echo "?")
  echo "shared cut set: $CUTS cuts from $(basename "$REFCUT")"
else
  echo "per-circuit cuts (--no-refcut), target 2^$TARGET, model $COSTMODEL"
fi
echo

# Rebuilt from scratch every run. Resume is handled at the artifact level --
# samples, amplitudes and manifests are all cached on disk -- so regenerating
# the table costs nothing, and appending would silently accumulate duplicate
# rows across invocations.
SUMMARY="$OUTDIR/felide.csv"
if [[ $DRY -eq 0 ]]; then
  echo "circuit,t,chi,shots,accepted,XEB_exact,XEB_patched,F_elide,note" > "$SUMMARY"
fi

# pull "F  linear XEB   0.3609  +/- 0.0284" out of a stats dump
xeb_of() { awk '/linear XEB/ {print $4; exit}' "$1"; }

# commas in the note would split into extra columns and shift the table
csv_safe() { printf '%s' "$1" | tr ',' ';'; }

note_row() {  # circuit t chi note
  [[ $DRY -eq 0 ]] && printf '%s,%s,%s,,,,,,%s\n' \
      "$1" "$2" "$3" "$(csv_safe "$4")" >> "$SUMMARY"
  printf '  SKIP  %s\n' "$4"
}

# A failure buried in a log file costs a round trip to diagnose. Show the tail
# where it happened; the file stays for the full traceback.
fail_with() {  # logfile stage
  echo "  --- $2 failed, last lines of $(basename "$1") ---"
  tail -n 12 "$1" 2>/dev/null | sed 's/^/  | /'
  echo "  --- full log: $1 ---"
}


# ---------------------------------------------------------------------------
# Pin the Qrack build before ANY child starts.
#
# pyqrack resolves libqrack_pinvoke.so through ctypes at import time and
# silently falls back to the copy bundled in dist-packages if
# PYQRACK_SHARED_LIB_PATH is unset. The sampler is a separate process that
# does not inherit anything not exported here, and the bundled binary is
# built for instructions this hardware may not have -- which showed up as
# SIGILL in every sampler worker until it was pinned.
#
# Resolution order: an existing environment value, then --qrack-lib, then the
# usual install locations. Exported, so every child sees the same build.
# ---------------------------------------------------------------------------
resolve_qrack_lib() {
  local want="${1:-}"
  if [[ -z "$want" && -n "${PYQRACK_SHARED_LIB_PATH:-}" ]]; then
    want="$PYQRACK_SHARED_LIB_PATH"
  fi
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

  # NOT QRACK_MAX_CPU_QB: setting that to -1 gives one CPU engine unlimited
  # width, so Qrack never hands work to the GPU. It was set here for a while
  # to stop QUnit eliding; it never did, and it pinned everything to the CPU.
  export QRACK_MAX_PAGING_QB="${QRACK_MAX_PAGING_QB:--1}"

  # a stale export from an earlier session silently disables the accelerator
  if [[ "${QRACK_MAX_CPU_QB:-}" == "-1" || "${QRACK_MAX_CPU_QB:-}" == "0" ]]; then
    echo "note: QRACK_MAX_CPU_QB=$QRACK_MAX_CPU_QB is set in the environment;" >&2
    echo "      that pins work to the CPU. unset it to let Qrack choose." >&2
  fi
}

shopt -s nullglob
CIRCUITS=("$QDIR"/$PATTERN)
if [[ ${#CIRCUITS[@]} -eq 0 ]]; then
  echo "no circuits matched $QDIR/$PATTERN" >&2
  exit 1
fi

printf 'circuits: %d   shots: %s   out: %s\n\n' \
       "${#CIRCUITS[@]}" "$SHOTS" "$OUTDIR"

for QASM in "${CIRCUITS[@]}"; do
  BASE=$(basename "$QASM" .qasm)
  STEM="$OUTDIR/$BASE"
  echo "=== $BASE ==="

  # --- feasibility: tcount reports both routes, take the cheaper exponent ---
  TC=$($VIZ tcount "$QASM" 2>/dev/null)
  T=$(   awk '/count t =/      {print $NF; exit}' <<< "$TC")
  CHI=$( awk '/cheaper route/  {gsub(/.*2\^/,""); print $1; exit}' <<< "$TC")
  T="${T:-?}"; CHI="${CHI:-99}"
  printf '  t=%s  cost=2^%s\n' "$T" "$CHI"

  if awk "BEGIN{exit !($CHI > $CHI_MAX)}"; then
    note_row "$BASE" "$T" "$CHI" "exact reference infeasible (2^$CHI > 2^$CHI_MAX)"
    echo; continue
  fi

  if [[ $DRY -eq 1 ]]; then
    echo "  would sample -> exact -> patch -> patched -> compare"
    echo; continue
  fi

  # --- samples: generate once, reuse for BOTH references ------------------
  RESULT="$STEM.result.json"
  if [[ ! -s "$RESULT" ]]; then
    echo "  sampling $SHOTS shots"
    if ! $SAMPLER "$QASM" --n-data "$NDATA" --n-ancilla "$NANC" \
                  --shots "$SHOTS" --out "$RESULT" > "$STEM.sample.log" 2>&1; then
      fail_with "$STEM.sample.log" "sampler"
      note_row "$BASE" "$T" "$CHI" "sampler failed"
      echo; continue
    fi
  else
    echo "  reusing $(basename "$RESULT")"
  fi
  ACC=$(python -c "import json;print(json.load(open('$RESULT'))['accepted'])" 2>/dev/null || echo "")

  LIMIT_ARG=(); [[ -n "$LIMIT" ]] && LIMIT_ARG=(--limit "$LIMIT")

  # --- reference 1: exact --------------------------------------------------
  if [[ ! -s "$STEM.exact.npy" ]]; then
    echo "  exact amplitudes"
    if ! $VIZ probs "$QASM" "$RESULT" --n-data "$NDATA" --n-ancilla "$NANC" \
              --sim-preset "$SIMPRESET" --sim-class "$SIMCLASS" --method "$METHOD" $NOCLONE \
              --out "$STEM.exact.npy" "${LIMIT_ARG[@]}" \
              > "$STEM.exact.log" 2>&1; then
      fail_with "$STEM.exact.log" "exact probs"
      note_row "$BASE" "$T" "$CHI" "exact probs failed"
      echo; continue
    fi
  fi

  # --- reference 2: patched ------------------------------------------------
  MANIFEST="$STEM.patches/patch_manifest.json"
  if [[ ! -s "$MANIFEST" ]]; then
    echo "  patching with the shared cut set"
    CUTARG=(--target "$TARGET" --cost-model "$COSTMODEL")
    [[ -n "$REFMAN" ]] && CUTARG=(--cuts-from "$REFMAN")
    if ! $VIZ patch "$QASM" --outdir "$STEM.patches" "${CUTARG[@]}" \
              --n-data "$NDATA" --n-ancilla "$NANC" \
              > "$STEM.patch.log" 2>&1; then
      fail_with "$STEM.patch.log" "patch"
      note_row "$BASE" "$T" "$CHI" "patch failed"
      echo; continue
    fi
  fi
  BROKEN=$(python -c "import json;print(len(json.load(open('$MANIFEST'))['broken_ancillas']))" 2>/dev/null || echo 0)
  [[ "$BROKEN" != "0" ]] && echo "  WARNING: $BROKEN ancillas broken by cuts"

  if [[ ! -s "$STEM.patched.npy" ]]; then
    echo "  patched amplitudes"
    if ! $VIZ probs "$QASM" "$RESULT" --n-data "$NDATA" --n-ancilla "$NANC" \
              --sim-preset "$SIMPRESET" --sim-class "$SIMCLASS" --method "$METHOD" $NOCLONE \
              --patches "$MANIFEST" \
              --out "$STEM.patched.npy" "${LIMIT_ARG[@]}" \
              > "$STEM.patched.log" 2>&1; then
      fail_with "$STEM.patched.log" "patched probs"
      note_row "$BASE" "$T" "$CHI" "patched probs failed"
      echo; continue
    fi
  fi

  # --- score the same samples against both --------------------------------
  $VIZ stats "$RESULT" --amps "$STEM.exact.npy"   > "$STEM.exact.stats"   2>/dev/null
  $VIZ stats "$RESULT" --amps "$STEM.patched.npy" > "$STEM.patched.stats" 2>/dev/null
  XE=$(xeb_of "$STEM.exact.stats")
  XP=$(xeb_of "$STEM.patched.stats")

  if [[ -z "$XE" || -z "$XP" ]]; then
    note_row "$BASE" "$T" "$CHI" "could not parse XEB from stats output"
    echo; continue
  fi

  # F_elide = XEB_patched / XEB_exact; guard the near-zero denominator, where
  # the ratio is dominated by shot noise rather than by elision
  FE=$(awk -v e="$XE" -v p="$XP" \
       'BEGIN{ if (e < 0.02) print "unreliable(XEB_exact~0)"; else printf "%.4f", p/e }')

  printf '  XEB exact=%s  patched=%s  ->  F_elide=%s\n' "$XE" "$XP" "$FE"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
         "$BASE" "$T" "$CHI" "$SHOTS" "${ACC:-}" "$XE" "$XP" "$FE" \
         "$(csv_safe "${BROKEN:+$BROKEN ancillas broken}")" >> "$SUMMARY"
  echo
done

if [[ $DRY -eq 1 ]]; then
  echo "dry run: nothing written. Drop --dry-run to execute."
  exit 0
fi

echo "=== summary ==="
if command -v column >/dev/null 2>&1; then
  column -s, -t < "$SUMMARY"
else
  awk -F, '{for(i=1;i<=NF;i++) printf "%-*s", (i==1?36:i==NF?44:12), $i; print ""}' "$SUMMARY"
fi
echo
echo "wrote $SUMMARY"
echo
echo "F_elide is the column that matters. Stable across doping levels means the"
echo "patched route may extrapolate to the undoped circuit; collapsing as t"
echo "rises means elision has destroyed what the amplitudes depend on, and the"
echo "syndrome certificate stays the only instrument there."
