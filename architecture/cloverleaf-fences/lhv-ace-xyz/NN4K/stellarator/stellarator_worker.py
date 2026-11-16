# -*- coding: us-ascii -*-
# stellarator_worker.py -- Rev 1.
#
# Persistent PyQrack manifold worker for the 40 x 27-qubit W7-X-like
# stellarator flow engine (Section 5.1 & 5.2).
#
# Each worker owns a subset of the 40 27-qubit manifolds (e.g. 6 to 7
# manifolds per V340 GPU die, or time-sliced across the 10 half-period
# modules) and executes the 7-stage flow loop per step:
#   1. Trotter step on internal manifold bonds (+ optional Heisenberg term)
#   2. Flux-preserving bond sources (edge gas puff, core pellet train,
#      10-channel ECRH gyrotron drive)
#   3. Seam-facing Pauli expectation readout (with startup sign probe)
#   4. (Orchestrator routes K4 junction, toroidal seam & radial kicks)
#   5. Incoming covariance kicks ('xyz' full covariance or 'z' boundary mode)
#   6. Mid-circuit measurement + conditional X reset on divertor sink qubits
#   7. Hadamard-test canonical loop stabiliser <W_p> readout into FluxMonitor
#
#   python3 stellarator_worker.py --manifest manifest.json --check

import argparse
import json
import math
import os
import sys

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):
    os.environ.setdefault("PYQRACK_SHARED_LIB_PATH", QRACK_LIB_PATH)

import numpy as np
from pyqrack import Pauli, QrackSimulator

import stellarator_pyqrack_driver as SPD


def probe_pauli_signs():
    """Startup Pauli expectation sign probe (Risk 9, Section 9):
    Verify PyQrack's prob(q) convention on |1> (Z=-1), |+> (X=+1), |+i> (Y=+1)
    and return the verified sign factors (sx, sy, sz)."""
    sim = QrackSimulator(1)
    # Z probe on |1>: 1 - 2*prob(0) should be -1
    sim.x(0)
    z_val = 1.0 - 2.0 * float(sim.prob(0))
    sz = -1.0 if z_val > 0 else +1.0
    sim.x(0)

    # X probe on |+>: H -> measure in X basis (H) -> |0> -> +1
    sim.h(0)
    sim.h(0)
    x_val = 1.0 - 2.0 * float(sim.prob(0))
    sx = +1.0 if x_val > 0 else -1.0

    # Y probe on |+i> = S H |0>: measure in Y basis (H S^dagger) -> |0> -> +1
    sim.h(0)
    sim.s(0)
    sim.adjs(0)
    sim.h(0)
    y_val = 1.0 - 2.0 * float(sim.prob(0))
    sy = +1.0 if y_val > 0 else -1.0

    return (sx, sy, sz)


class StellaratorManifoldWorker(object):
    """Persistent PyQrack worker managing one or more 27-qubit manifolds."""

    def __init__(self, block_specs, device_id=0,
                 max_ket_qubits=SPD.DEFAULT_MAX_KET_QUBITS,
                 kick_mode="xyz"):
        self.block_specs = {int(b["block_id"]): b for b in block_specs}
        self.device_id = int(device_id)
        self.max_ket_qubits = int(max_ket_qubits)
        self.kick_mode = kick_mode
        self.pauli_signs = probe_pauli_signs()

        self.sims = {}
        self.active_ket = {}
        self.exhaust_counts = {int(b["block_id"]): 0 for b in block_specs}

        for bid, b in self.block_specs.items():
            sim = QrackSimulator(28)
            SPD.prepare_loop_sector_state(
                sim, b["loop_pauli_local"], float(b["loop_target"]))
            self.sims[bid] = sim
            # Include canonical loop qubits + divertor qubits + inner core qubits
            # in the active non-Clifford ket set up to max_ket_qubits
            priority_qs = list(range(len(b["loop"])))
            for q in b.get("divertor_qubits", []) + b.get("core_qubits", []):
                if q not in priority_qs:
                    priority_qs.append(q)
            for q in range(27):
                if len(priority_qs) >= self.max_ket_qubits:
                    break
                if q not in priority_qs:
                    priority_qs.append(q)
            self.active_ket[bid] = set(priority_qs[:self.max_ket_qubits])

    def step_trotter(self, dt=0.06, heisenberg_g=0.0):
        """Step 1: Trotter step on internal manifold bonds (+ optional
        Heisenberg perturbation g * sum_{a=x,y,z} sigma^a_i sigma^a_j)."""
        for bid, b in self.block_specs.items():
            sim = self.sims[bid]
            act = self.active_ket[bid]
            for ib in b["internal_bonds"]:
                qa, qb = int(ib["qa"]), int(ib["qb"])
                col = int(ib["colour"])
                theta = -float(dt) * float(ib["weight"]) * float(ib["u"])
                non_cliff = (qa in act and qb in act)
                SPD.apply_bond_step(sim, qa, qb, col, theta, non_clifford=non_cliff)
                if heisenberg_g != 0.0 and non_cliff:
                    theta_h = -float(dt) * float(heisenberg_g)
                    for c_alt in (0, 1, 2):
                        if c_alt != col:
                            SPD.apply_bond_step(
                                sim, qa, qb, c_alt, theta_h, non_clifford=True)

    def apply_sources(self, bid, bond_indices, theta_drive):
        """Step 2: Apply flux-preserving bond rotations on selected internal
        bonds (core pellet injection, edge gas puff, or ECRH drive)."""
        if bid not in self.sims or abs(theta_drive) < 1e-15:
            return
        sim = self.sims[bid]
        b = self.block_specs[bid]
        act = self.active_ket[bid]
        ibs = b["internal_bonds"]
        for idx_ib in bond_indices:
            if 0 <= idx_ib < len(ibs):
                ib = ibs[idx_ib]
                qa, qb = int(ib["qa"]), int(ib["qb"])
                col = int(ib["colour"])
                non_cliff = (qa in act and qb in act)
                SPD.apply_bond_step(
                    sim, qa, qb, col, float(theta_drive) * float(ib["u"]),
                    non_clifford=non_cliff)

    def read_qubit_pauli(self, bid, q_loc, col):
        """Step 3: Read <sigma^col_q> on a seam-facing qubit using basis
        rotation + exact probability readout."""
        sim = self.sims[bid]
        q = int(q_loc)
        c = int(col)
        if c == 0:
            sim.h(q)
            val = 1.0 - 2.0 * float(sim.prob(q))
            sim.h(q)
            return self.pauli_signs[0] * val
        elif c == 1:
            sim.adjs(q)
            sim.h(q)
            val = 1.0 - 2.0 * float(sim.prob(q))
            sim.h(q)
            sim.s(q)
            return self.pauli_signs[1] * val
        else:
            val = 1.0 - 2.0 * float(sim.prob(q))
            return self.pauli_signs[2] * val

    def apply_seam_kicks(self, bid, kicks):
        """Step 5: Apply incoming seam kicks.
        To preserve the intra-manifold loop stabilisers W_p (which commute with
        all internal Kitaev bonds), kicks on qubits that belong to the canonical
        8-loop are routed through their incident internal bond rotation, while
        tree boundary qubits receive direct Z (or X/Y/Z covariance) rotations."""
        if bid not in self.sims:
            return
        sim = self.sims[bid]
        b = self.block_specs[bid]
        act = self.active_ket[bid]
        loop_q_set = {int(q) for q, _ in b["loop_pauli_local"]}

        for q_loc, col, angle in kicks:
            q = int(q_loc)
            c = int(col)
            phi = float(angle)
            if abs(phi) < 1e-12 or q not in act:
                continue
            if q in loop_q_set:
                # Route kick into the first internal bond incident on q so W_p
                # commutes with the kick operator (Risk 8 mitigation)
                for ib in b["internal_bonds"]:
                    qa, qb = int(ib["qa"]), int(ib["qb"])
                    if qa == q or qb == q:
                        if qa in act and qb in act:
                            SPD.apply_bond_step(
                                sim, qa, qb, int(ib["colour"]),
                                0.5 * phi * float(ib["u"]), non_clifford=True)
                        break
            else:
                if self.kick_mode == "z" or c == 2:
                    sim.r(Pauli.PauliZ, 2.0 * phi, q)
                elif c == 0:
                    sim.h(q)
                    sim.r(Pauli.PauliZ, 2.0 * phi, q)
                    sim.h(q)
                elif c == 1:
                    sim.adjs(q)
                    sim.h(q)
                    sim.r(Pauli.PauliZ, 2.0 * phi, q)
                    sim.h(q)
                    sim.s(q)

    def measure_and_reset_sinks(self, bid, drive_amplitude=0.0):
        """Step 6: Mid-circuit measurement and conditional X reset on the
        divertor target-plate qubits of block `bid`. Returns (exhaust_prob_sum,
        exhaust_bit_count)."""
        if bid not in self.sims:
            return 0.0, 0
        b = self.block_specs[bid]
        div_qs = b.get("divertor_qubits", [])
        if not div_qs:
            return 0.0, 0

        sim = self.sims[bid]
        act = self.active_ket[bid]
        # Couple edge excitations into the divertor plate bonds before readout
        if abs(drive_amplitude) > 1e-12:
            for ib in b["internal_bonds"]:
                qa, qb = int(ib["qa"]), int(ib["qb"])
                if (qa in div_qs or qb in div_qs) and (qa in act and qb in act):
                    SPD.apply_bond_step(
                        sim, qa, qb, int(ib["colour"]),
                        float(drive_amplitude), non_clifford=True)

        prob_sum = 0.0
        bit_count = 0
        for q in div_qs:
            q_int = int(q)
            p1 = float(sim.prob(q_int))
            prob_sum += p1
            outcome = sim.m(q_int)
            if outcome:
                sim.x(q_int)  # Conditional X reset back to |0>
                bit_count += 1
        self.exhaust_counts[bid] += bit_count
        return prob_sum, bit_count

    def measure_flux(self, bid):
        """Step 7: Measure canonical 8-loop stabiliser <W_p> via Hadamard test."""
        sim = self.sims[bid]
        b = self.block_specs[bid]
        w_val = SPD.measure_flux_hadamard(sim, b["loop_pauli_local"], anc=27)
        tgt = float(b["loop_target"])
        eps_w = ((w_val - tgt) / 2.0) ** 2
        return w_val, tgt, eps_w


def worker_self_test(manifest):
    print("=" * 78)
    print("PHASE 5 GATE (PYQRACK WORKER) -- SIGN PROBE, SOURCES, KICKS & SINK RESET")
    print("=" * 78)
    signs = probe_pauli_signs()
    print("  startup Pauli sign probe (X, Y, Z) : %s" % (signs,))

    # Instantiate a worker owning module 0's 4 manifolds (108 qubits = 1 K4 cluster)
    mod0_blocks = manifest["blocks"][:4]
    worker = StellaratorManifoldWorker(mod0_blocks, device_id=0, kick_mode="xyz")

    for step in range(3):
        worker.step_trotter(dt=0.06, heisenberg_g=0.0)
        # Core pellet source on block 0
        worker.apply_sources(0, [0, 1], theta_drive=0.12)
        # Apply seam kicks on all 4 blocks
        for bid in range(4):
            worker.apply_seam_kicks(bid, [(10, 2, 0.05), (11, 0, -0.04)])
        # Measure & reset divertor sinks on block 2 (which owns the 6 divertor qubits)
        ex_prob, ex_bits = worker.measure_and_reset_sinks(2, drive_amplitude=0.18)

    all_ok = (signs == (1.0, 1.0, 1.0))
    for bid in range(4):
        w_val, tgt, eps_w = worker.measure_flux(bid)
        ok = eps_w < 1e-6
        all_ok = all_ok and ok
        print("  block %d (type %d): <W_p> = %+.6f (target %+.1f)  eps_W = %.4e  div_exhaust=%.4f  %s"
              % (bid, bid % 4, w_val, tgt, eps_w,
                 ex_prob if bid == 2 else 0.0, "PASS" if ok else "FAIL"))

    print("-" * 78)
    print("WORKER SELF-TEST VERDICT: %s" % ("PASS" if all_ok else "FAIL"))
    print("=" * 78)
    return all_ok


def main():
    ap = argparse.ArgumentParser(
        description="Persistent PyQrack manifold worker for the stellarator")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--kick-mode", choices=("xyz", "z"), default="xyz")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    with open(a.manifest, "r") as fh:
        manifest = json.load(fh)

    ok = worker_self_test(manifest)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
