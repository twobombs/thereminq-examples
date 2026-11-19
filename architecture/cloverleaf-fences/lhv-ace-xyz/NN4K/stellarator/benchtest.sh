#!/usr/bin/env bash
# benchtest.sh -- End-to-end verification suite for the W7-X Stellarator
# Kitaev Z2 Lattice Gauge Emulation (Phases 0 to 8 of stellarator-kitaev-plan.md).
set -euo pipefail

cd "$(dirname "$0")"

echo "=============================================================================="
echo "  W7-X STELLARATOR KITAEV Z2 LATTICE GAUGE EMULATION -- FULL BENCHTEST SUITE"
echo "=============================================================================="

echo "[1/8] Phase 1 & 7 Gate: W7-X Geometry, D5 Symmetry, Coils & Biot-Savart Poincare..."
python3 stellarator_geometry.py --check

echo "[2/8] Phase 2 & 3 Gate: Flux Stabilisers, Wilson Lines, Island Chains & Trim Coils..."
python3 stellarator_flux.py --check

echo "[3/8] Phase 0, 1, 2 & 3 Gate: Sizing, Hyperoctagon Anchor, Candidates H/P & Scaling..."
python3 stellarator_kitaev.py --sizing --hyperoctagon --check --candidate both --scaling --presets --coils

echo "[4/8] Phase 4a Gate: 40 x 27-Qubit Manifold Partition & 4 Block Equivalence Classes..."
python3 stellarator-equiv.py --check --prune -o equiv.json

echo "[5/8] Phase 4b & 4c Gate: Manifest Compiler & OpenQASM 3.0 Emitter..."
python3 stellarator-manifest.py --equiv equiv.json -o manifest.json
python3 stellarator-qasm.py --manifest manifest.json -o qasm

echo "[6/8] Phase 4d Gate: PyQrack 27-Qubit Manifold Driver & Seam-Repair Budget..."
python3 stellarator_pyqrack_driver.py --manifest manifest.json --check

echo "[7/8] Phase 5 & 6 Gate: Worker Self-Test, Rungs F1-F5 & W7-X 7-Criteria Scoreboard..."
python3 stellarator_worker.py --manifest manifest.json --check
python3 stellarator_flow.py --manifest manifest.json --check --optimise \
  --telemetry flow_telemetry.npz --scoreboard scoreboard.json

echo "[8/8] Phase 8 Gate: 6-Panel W7-X Stellarator Dashboard Render..."
python3 stellarator-dash.py --manifest manifest.json --scoreboard scoreboard.json \
  --telemetry flow_telemetry.npz --save stellarator_dashboard.png --check

echo "=============================================================================="
echo "  ALL PHASES (0 TO 8) PASSED SUCCESSFULLY"
echo "=============================================================================="
