# -*- coding: us-ascii -*-
# stellarator_worker.py -- Rev 2.
#
# Persistent PyQrack 27-qubit (ancilla-free) manifold worker for the
# 40 x 27-qubit W7-X-like stellarator flow engine (Section 5.1 & 5.2).
#
# Each worker owns a subset of the 40 27-qubit manifolds (e.g. 6 to 7
# manifolds per V340 GPU die, or time-sliced across the 10 half-period
# modules) and executes the 7-stage flow loop per step:
#   1. Trotter step on all 27-qubit internal manifold bonds + Sigma_keep
#      boundary-repair bonds (+ optional Heisenberg perturbation)
#   2. Flux-preserving bond sources (edge gas puff on s=4, core pellet train,
#      10-channel ECRH gyrotron drive)
#   3. 2-qubit bond-expectation readout on seam-incident bonds (with startup
#      Pauli & bond-parity sign probes)
#   4. (Orchestrator routes K4 junction, toroidal seam & radial kicks)
#   5. Incoming 2-qubit bond-type seam kicks (flux-preserving across all cycles)
#   6. 2-qubit bond-parity measurement + conditional reset on divertor sink bonds
#   7. Ancilla-free in-place parity readout of ALL internal cycles in
#      `b["internal_loops"]` into FluxMonitor
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
os.environ.setdefault("QRACK_DISABLE_QUNIT_FIDELITY_GUARD", "1")

import numpy as np
from pyqrack import Pauli, QrackSimulator

import stellarator_pyqrack_driver as SPD


def probe_pauli_signs():
    """Startup Pauli & 2-qubit bond-parity expectation sign probe (Risk 9):
    Verify PyQrack's prob(q) convention on single-qubit |1> (Z=-1), |+> (X=+1),
    |+i> (Y=+1) and 2-qubit bond parity <sigma^c_0 sigma^c_1> on a dimer."""
    sim = QrackSimulator(2, is_gpu=False)
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

    # 2-qubit bond parity probe on Z0 Z1 = -1 (|01>)
    sim.x(1)
    zz_val = SPD.read_bond_expectation(sim, 0, 1, 2)
    sim.x(1)
    s_bond = +1.0 if zz_val < -0.99 else -1.0

    return (sx, sy, sz, s_bond)


class StellaratorManifoldWorker(object):
    """Persistent PyQrack worker managing one or more 27-qubit manifolds."""

    def __init__(self, block_specs, device_id=0,
                 kick_mode="xyz", use_sigma_repair=True,
                 shared_dimer=False,
                 ace_max_qb=None,
                 sdrp=0.0):
        self.block_specs = {int(b["block_id"]): b for b in block_specs}
        self.device_id = int(device_id)
        self.kick_mode = kick_mode
        self.use_sigma_repair = bool(use_sigma_repair)
        self.shared_dimer = bool(shared_dimer)
        self.pauli_signs = probe_pauli_signs()

        self.sims = {}
        self.q_counts = {}
        self.shared_G0 = {}
        self.sig_maps = {}
        self.last_cycles = {}
        self.exhaust_counts = {int(b["block_id"]): 0 for b in block_specs}

        for bid, b in self.block_specs.items():
            sim = QrackSimulator(27, is_gpu=SPD.HAS_OPENCL_SO)
            if self.shared_dimer:
                qc, g0, _ = SPD.prepare_shared_dimer_state(sim, b)
                self.q_counts[bid] = qc
                self.shared_G0[bid] = g0
            else:
                qc = SPD.prepare_all_internal_loops(sim, b)
                self.q_counts[bid] = qc
            n_core = sum(1 for v in self.q_counts[bid].values() if v > 0)
            ace_qb = (22 if n_core >= 24 else SPD.DEFAULT_ACE_MAX_QB) if ace_max_qb is None else int(ace_max_qb)
            SPD.configure_simulator_27q(sim, ace_max_qb=ace_qb, sdrp=sdrp)
            self.sims[bid] = sim
            smap = {}
            if self.use_sigma_repair:
                for sb in b.get("sigma_repair_bonds", []):
                    if sb.get("is_internal_bond", False):
                        qa, qb = int(sb["qa"]), int(sb["qb"])
                        smap[(min(qa, qb), max(qa, qb))] = float(
                            sb.get("sigma_ab", sb.get("sigma_val", 0.0)))
            self.sig_maps[bid] = smap

    def step_trotter(self, dt=0.06, heisenberg_g=0.0):
        """Step 1: Trotter step on all 27-qubit internal manifold bonds,
        fused internal Sigma_keep repair bonds, and optional Heisenberg
        perturbation g * sum_{a=x,y,z} sigma^a_i sigma^a_j."""
        for bid, b in self.block_specs.items():
            sim = self.sims[bid]
            smap = self.sig_maps[bid]
            n_core = sum(1 for v in self.q_counts[bid].values() if v > 0)
            for ib in b["internal_bonds"]:
                qa, qb = int(ib["qa"]), int(ib["qb"])
                col = int(ib["colour"])
                s_val = smap.get((min(qa, qb), max(qa, qb)), 0.0)
                theta = -float(dt) * (float(ib["weight"]) * float(ib["u"]) + s_val)
                SPD.apply_bond_step(sim, qa, qb, col, theta, non_clifford=True)
                if heisenberg_g != 0.0:
                    theta_h = -float(dt) * float(heisenberg_g)
                    for c_alt in (0, 1, 2):
                        if c_alt != col:
                            SPD.apply_bond_step(
                                sim, qa, qb, c_alt, theta_h, non_clifford=True)
            if not self.shared_dimer and n_core < 24:
                self.last_cycles[bid] = SPD.measure_all_internal_loops(
                    sim, b, q_counts=self.q_counts[bid])

    def apply_sources(self, bid, bond_indices, theta_drive):
        """Step 2: Apply flux-preserving 2-qubit bond rotations on selected
        internal bonds (core pellet injection, edge gas puff on s=4, or ECRH
        gyrotron drive)."""
        if bid not in self.sims or abs(theta_drive) < 1e-15:
            return
        sim = self.sims[bid]
        b = self.block_specs[bid]
        ibs = b["internal_bonds"]
        for idx_ib in bond_indices:
            if 0 <= idx_ib < len(ibs):
                ib = ibs[idx_ib]
                qa, qb = int(ib["qa"]), int(ib["qb"])
                col = int(ib["colour"])
                SPD.apply_bond_step(
                    sim, qa, qb, col, float(theta_drive) * float(ib["u"]),
                    non_clifford=True)

    def read_bond_expectation(self, bid, qa, qb, col):
        """Step 3: Read 2-qubit bond expectation <sigma^col_qa sigma^col_qb>
        using ancilla-free in-place parity folding."""
        sim = self.sims[bid]
        qc = self.q_counts.get(bid)
        return self.pauli_signs[3] * SPD.read_bond_expectation(
            sim, qa, qb, col, q_counts=qc)

    def read_internal_bond_expectations(self, bid, max_bonds=None):
        """Read <sigma^col_qa sigma^col_qb> across internal bonds of block `bid`."""
        sim = self.sims[bid]
        b = self.block_specs[bid]
        qc = self.q_counts.get(bid)
        ibs = b["internal_bonds"]
        if max_bonds is not None:
            ibs = ibs[:int(max_bonds)]
        out = []
        for ib in ibs:
            qa, qb = int(ib["qa"]), int(ib["qb"])
            col = int(ib["colour"])
            val = self.pauli_signs[3] * SPD.read_bond_expectation(
                sim, qa, qb, col, q_counts=qc)
            out.append({
                "qa": qa,
                "qb": qb,
                "colour": col,
                "u": float(ib["u"]),
                "weight": float(ib["weight"]),
                "exp_val": val,
            })
        return out

    def read_seam_bond_expectations(self, bid, max_bonds=1):
        """Step 3: Read 2-qubit bond expectations on internal bonds incident on
        seam boundary qubits of block `bid`, indexed by seam qubit `q_loc`."""
        sim = self.sims[bid]
        b = self.block_specs[bid]
        qc = self.q_counts.get(bid, {})
        seam_set = {int(q) for q in b.get("seam_qubits", [])}
        # Prioritize seam bonds touching non-cycle or private qubits for fast readout
        cand_bonds = [
            ib for ib in b["internal_bonds"]
            if int(ib["qa"]) in seam_set or int(ib["qb"]) in seam_set
        ]
        cand_bonds.sort(key=lambda ib: (
            min(qc.get(int(ib["qa"]), 0), qc.get(int(ib["qb"]), 0)),
            qc.get(int(ib["qa"]), 0) + qc.get(int(ib["qb"]), 0)
        ))
        seam_exp = {}
        count = 0
        for ib in cand_bonds:
            qa, qb = int(ib["qa"]), int(ib["qb"])
            col = int(ib["colour"])
            val = self.pauli_signs[3] * SPD.read_bond_expectation(
                sim, qa, qb, col, q_counts=qc)
            if qa in seam_set and qa not in seam_exp:
                seam_exp[qa] = (qa, qb, col, float(ib["u"]), val)
            elif qb in seam_set and qb not in seam_exp:
                seam_exp[qb] = (qa, qb, col, float(ib["u"]), val)
            count += 1
            if max_bonds is not None and count >= int(max_bonds):
                break
        return seam_exp

    def read_qubit_pauli(self, bid, q_loc, col):
        """Compatibility helper: if `q_loc` is incident on an internal bond,
        return that 2-qubit bond expectation; otherwise read single-qubit Pauli."""
        b = self.block_specs[bid]
        q = int(q_loc)
        for ib in b["internal_bonds"]:
            qa, qb = int(ib["qa"]), int(ib["qb"])
            if qa == q or qb == q:
                return self.read_bond_expectation(bid, qa, qb, int(ib["colour"]))
        sim = self.sims[bid]
        return SPD.read_pauli_word_inplace(sim, [(q, int(col))])

    def apply_seam_kicks(self, bid, kicks):
        """Step 5: Apply incoming 2-qubit bond-type seam kicks.
        Accepts either:
          - 4-tuples `(qa, qb, col, angle)` specifying an explicit internal bond
          - 3-tuples `(q_loc, col, angle)` specifying a seam qubit, routed into
            its incident internal Kitaev bond so all intra-block cycle
            stabilisers W_p commute with the kick operator (no vison creation)."""
        if bid not in self.sims:
            return
        sim = self.sims[bid]
        b = self.block_specs[bid]

        for item in kicks:
            if len(item) == 4:
                qa, qb, col, angle = int(item[0]), int(item[1]), int(item[2]), float(item[3])
                if abs(angle) < 1e-12:
                    continue
                SPD.apply_bond_step(sim, qa, qb, col, angle, non_clifford=True)
            else:
                q = int(item[0])
                col_req = int(item[1])
                phi = float(item[2])
                if abs(phi) < 1e-12:
                    continue
                chosen = None
                for ib in b["internal_bonds"]:
                    qa, qb = int(ib["qa"]), int(ib["qb"])
                    if qa == q or qb == q:
                        if int(ib["colour"]) == col_req:
                            chosen = ib
                            break
                        if chosen is None:
                            chosen = ib
                if chosen is not None:
                    SPD.apply_bond_step(
                        sim, int(chosen["qa"]), int(chosen["qb"]),
                        int(chosen["colour"]),
                        0.5 * phi * float(chosen["u"]), non_clifford=True)

    def measure_and_reset_sinks(self, bid, drive_amplitude=0.0, max_bonds=2):
        """Step 6: 2-qubit bond-parity divertor sink measurement and reset on
        `b['divertor_bonds']` (island-divertor target bonds on outer shell s=4).
        For tree divertor bonds with a non-cycle leaf qubit (`q_counts[q] == 0`),
        performs a projective 2-qubit bond-parity measurement + conditional reset
        to the bond ground state `sigma^col_qa sigma^col_qb = +u_b`. For cycle-
        touching divertor bonds, reads the 2-qubit bond expectation and applies
        a flux-preserving damping rotation toward `+u_b`.
        Returns `(exhaust_bond_excitation_sum, exhaust_reset_count)`."""
        if bid not in self.sims:
            return 0.0, 0
        b = self.block_specs[bid]
        div_bonds = b.get("divertor_bonds", [])
        if not div_bonds:
            return 0.0, 0

        sim = self.sims[bid]
        qc = self.q_counts[bid]

        deg = {}
        for ib in b["internal_bonds"]:
            qa_i, qb_i = int(ib["qa"]), int(ib["qb"])
            deg[qa_i] = deg.get(qa_i, 0) + 1
            deg[qb_i] = deg.get(qb_i, 0) + 1

        ordered_bonds = sorted(
            div_bonds,
            key=lambda db: (
                min(qc.get(int(db["qa"]), 0), qc.get(int(db["qb"]), 0)),
                min(deg.get(int(db["qa"]), 3), deg.get(int(db["qb"]), 3)),
                max(deg.get(int(db["qa"]), 3), deg.get(int(db["qb"]), 3)),
                -(int(db["qa"]) + int(db["qb"]))
            )
        )
        if max_bonds is not None:
            ordered_bonds = ordered_bonds[:int(max_bonds)]

        if abs(drive_amplitude) > 1e-12:
            for db in ordered_bonds:
                qa, qb = int(db["qa"]), int(db["qb"])
                col = int(db["colour"])
                u_b = float(db["u"])
                SPD.apply_bond_step(
                    sim, qa, qb, col, float(drive_amplitude) * u_b,
                    non_clifford=True)

        exhaust_sum = 0.0
        reset_count = 0
        for db in ordered_bonds:
            qa, qb = int(db["qa"]), int(db["qb"])
            col = int(db["colour"])
            u_b = float(db["u"])
            exp_val = self.pauli_signs[3] * SPD.read_bond_expectation(
                sim, qa, qb, col, q_counts=qc)
            p_ex = max(0.0, min(1.0, 0.5 * (1.0 - u_b * exp_val)))
            exhaust_sum += p_ex

            leaf_q = None
            other_q = None
            cand_a = (qc.get(qa, 0), deg.get(qa, 3))
            cand_b = (qc.get(qb, 0), deg.get(qb, 3))
            if min(cand_a[0], cand_b[0]) == 0:
                if cand_a <= cand_b:
                    leaf_q, other_q = qa, qb
                else:
                    leaf_q, other_q = qb, qa

            if leaf_q is not None:
                SPD.apply_basis_change(sim, col, leaf_q, other_q, inverse=False)
                sim.mcx([other_q], leaf_q)
                outcome = 1 if sim.m(leaf_q) else 0
                want = 1 if u_b < 0 else 0
                if outcome != want:
                    sim.x(leaf_q)
                    reset_count += 1
                sim.mcx([other_q], leaf_q)
                SPD.apply_basis_change(sim, col, leaf_q, other_q, inverse=True)
            else:
                if p_ex > 0.25:
                    reset_count += 1
                damp_theta = -0.25 * u_b * p_ex
                SPD.apply_bond_step(sim, qa, qb, col, damp_theta, non_clifford=True)

        self.exhaust_counts[bid] += reset_count
        return exhaust_sum, reset_count

    def measure_flux(self, bid):
        """Step 7: Measure ALL independent internal cycles in `b['internal_loops']`
        via ancilla-free in-place parity folding.
        Returns `(w_first, tgt_first, max_eps_w, cycles)`."""
        sim = self.sims[bid]
        b = self.block_specs[bid]
        cycles = SPD.measure_all_internal_loops(
            sim, b, q_counts=self.q_counts.get(bid))
        self.last_cycles[bid] = cycles
        max_eps = max(c["eps_W"] for c in cycles) if cycles else 0.0
        return cycles[0]["w_meas"], cycles[0]["target"], max_eps, cycles


def worker_self_test(manifest):
    print("=" * 78)
    print("PHASE 5 GATE (PYQRACK WORKER) -- 27Q MULTI-CYCLE, BOND KICKS & SINK RESET")
    print("=" * 78)
    signs = probe_pauli_signs()
    print("  startup Pauli & bond sign probe (X, Y, Z, bond) : %s" % (signs,))

    # Instantiate a worker owning module 0's 4 manifolds (108 qubits = 1 K4 cluster)
    mod0_blocks = manifest["blocks"][:4]
    worker = StellaratorManifoldWorker(mod0_blocks, device_id=0, kick_mode="xyz")

    for step in range(4):
        worker.step_trotter(dt=0.06, heisenberg_g=0.0)
        # Core pellet source on block 0 + edge gas-puff on block 2 (s=4)
        worker.apply_sources(0, [0, 1], theta_drive=0.12)
        worker.apply_sources(2, [0, 1], theta_drive=0.08)
        # Read 2-qubit seam bond expectations and apply 2-qubit bond-type seam kicks
        for bid in range(4):
            seam_exps = worker.read_seam_bond_expectations(bid, max_bonds=2)
            kicks = []
            for q_s, (qa, qb, col, u_b, val) in seam_exps.items():
                kicks.append((qa, qb, col, 0.04 * u_b * val))
            worker.apply_seam_kicks(bid, kicks)
        # Measure & reset 2-qubit divertor sink bonds on block 2 & 3
        ex_prob2, ex_cnt2 = worker.measure_and_reset_sinks(2, drive_amplitude=0.18, max_bonds=3)
        ex_prob3, ex_cnt3 = worker.measure_and_reset_sinks(3, drive_amplitude=0.18, max_bonds=3)

    all_ok = (signs == (1.0, 1.0, 1.0, 1.0))
    total_cycles = 0
    for bid in range(4):
        w_val, tgt, max_eps_w, cycles = worker.measure_flux(bid)
        total_cycles += len(cycles)
        ok = max_eps_w < 1e-6
        all_ok = all_ok and ok
        ex_val = ex_prob2 if bid == 2 else (ex_prob3 if bid == 3 else 0.0)
        print("  block %d (type %d): %d cycles  <W_0>=%+.6f (tgt %+.1f)  max_eps_W=%.4e  div_bond_ex=%.4f  %s"
              % (bid, bid % 4, len(cycles), w_val, tgt, max_eps_w,
                 ex_val, "PASS" if ok else "FAIL"))

    print("  total monitored cycles in module 0 : %d (across 40 blocks = %d)"
          % (total_cycles, total_cycles * 10))
    print("-" * 78)
    print("WORKER SELF-TEST VERDICT: %s" % ("PASS" if all_ok else "FAIL"))
    print("=" * 78)
    return all_ok


def main():
    ap = argparse.ArgumentParser(
        description="Persistent PyQrack 27-qubit manifold worker for the stellarator")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--kick-mode", choices=("xyz", "z"), default="xyz")
    ap.add_argument("--require-qrack-lib", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    if a.require_qrack_lib and not os.path.exists(QRACK_LIB_PATH):
        raise RuntimeError(
            "Required Qrack shared library missing at %s" % QRACK_LIB_PATH)

    with open(a.manifest, "r") as fh:
        manifest = json.load(fh)

    ok = worker_self_test(manifest)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
