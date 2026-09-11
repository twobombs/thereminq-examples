#!/usr/bin/env bash
# k4-build-manifest.sh [TILING] [THETA] [OUTDIR]
#
# Builds the relabelled manifest the K4 tools need, from files in this
# directory:
#
#   TILING (equiv.json)  --K4-32x27-manifest.py-->  manifest.json
#   manifest.json        --K4-qasm.py------------>  OUTDIR/manifest.json  (relabelled:
#                                                     canonical_bonds/loop/word)
#                                                   OUTDIR/qasm/type<k>-{trotter,flux,full}.qasm
#                                                   OUTDIR/k4_pyqrack_driver.py
#
# then runs w_commutes.py on the result, if it is here.
#
#   ./k4-build-manifest.sh                     # equiv.json, theta 0.1, into out/
#   ./k4-build-manifest.sh bury-2.json 0.05 out-b2
#   WITH_COILS=1 ./k4-build-manifest.sh        # also solve the coils (needs ortools)
#
# Note: ./manifest.json (the plain one) lacks the canonical_* fields.
# The driver that reads the relabelled one is OUTDIR/k4_pyqrack_driver.py,
# not the copy in this directory.
set -euo pipefail
cd "$(dirname "$0")"

TILING=${1:-equiv.json}
THETA=${2:-0.1}
OUT=${3:-out}

for f in K4-32x27-manifest.py K4-qasm.py "$TILING"; do
    [ -f "$f" ] || { echo "missing $f -- run this from the coils directory" >&2; exit 1; }
done

COILS=--no-coils
if [ "${WITH_COILS:-0}" = 1 ]; then
    if python3 -c "import ortools" 2>/dev/null; then
        COILS=""
    else
        echo "WITH_COILS=1 but ortools is not installed; building without coils" >&2
    fi
fi

echo "== 1/3  $TILING -> manifest.json"
python3 K4-32x27-manifest.py "$TILING" -o manifest.json $COILS

echo "== 2/3  manifest.json -> $OUT/ (theta $THETA)"
python3 K4-qasm.py manifest.json --theta "$THETA" -o "$OUT"

echo "== 3/3  W against every bond term"
CHECKER=""
for c in w_commutes.py w_communtes.py; do [ -f "$c" ] && { CHECKER=$c; break; }; done
if [ -n "$CHECKER" ]; then
    python3 "$CHECKER" "$OUT/manifest.json" | tail -n 1
else
    echo "w_commutes.py not in this directory; skipped"
fi

W=$(grep -o "W = [XYZ]* over loop qubits [0-9,]*" "$OUT/qasm/type0-flux.qasm" | awk '{print $3"@"$7}')
echo
echo "relabelled manifest: $OUT/manifest.json"
echo "type-0 flux word:    $W"
echo
echo "next:"
echo "  python3 $OUT/k4_pyqrack_driver.py --block 0 --theta $THETA --steps 4"
echo "  python3 qasm-pyqrack-seam-backend.py run $OUT/qasm/type0-trotter.qasm --shots 0 --pauli \"$W\""