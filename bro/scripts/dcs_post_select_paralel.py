###############################
#
# this is a modded and paralellized version of https://github.com/vm6502q/pyqrack-examples/blob/main/ibm/dcs_post_select.py
# copyrights, license, ownership remains effective in this file pertaining everything but the paralellization part for Dan Strano
# dont ask Dan about this script; ask me
################################

# Post-selected weak simulation of a doped-Clifford-sampling (DCS) circuit
# on QrackStabilizer -- parallel-shot version.
#
# Circuit source: Martiel, Chung, Seif, Ghosh, Hincks, Deshpande, Fefferman,
# Gambetta, Javadi-Abhari, "Sampling hard circuits with verifiably high
# fidelity," arXiv:2607.25941 (IBM Research / U. Chicago). Data and circuits
# from the paper are released under CC BY 4.0 via Zenodo,
# 10.5281/zenodo.21633064.
#
# IMPORTANT BUG NOTE: QrackStabilizer.m_all() silently corrupts results for
# systems above 64 qubits (verified directly: correct at n=64, wrong at
# n=65, confirmed at several points up to n=97). This script measures each
# qubit individually via .m(q) instead, which is correct at every size
# tested. QrackAceBackend.m_all() does NOT have this problem -- it's
# specific to QrackStabilizer. Worth fixing at the source; this is a
# workaround, not a fix.
#
# Expected behavior: this simulation has no physical hardware noise (no CZ
# error, no readout error, nothing) -- the only error source present is the
# T-gate weak-simulation rounding, which is specifically constructed
# (code-preserving doping) to commute with the checks. So acceptance here
# should be close to 100%, not the paper's 5.9e-4 -- that rate is almost
# entirely a statement about their physical hardware's noise, which this
# simulation doesn't model. 100% acceptance is the CORRECT result here, not
# a sign that post-selection isn't doing anything.
#
# PARALLELISM NOTES:
#   * Shots are embarrassingly parallel: each one is an independent fresh
#     instance from an empty tableau. The only cross-shot coupling is the
#     RNG, which is why seeding is handled explicitly below.
#   * Processes, not threads. QStabilizer work is tableau bit-twiddling
#     dispatched through ctypes; a thread pool would spend its life in
#     the same address space fighting over the same allocator with little
#     to show for it. Fork is cheap here -- the per-shot state is O(n^2)
#     bits, i.e. nothing at 97 qubits.
#   * pyqrack is imported *inside* the worker initializer, never in the
#     parent. That guarantees no Qrack library state (RNG included) exists
#     at fork time to be duplicated across workers.

import argparse
import json
import os
import time
import multiprocessing as mp

# --- worker-process globals, populated once per process by _init_worker ---
_QC = None
_N_DATA = None
_N_ANCILLA = None
_QrackStabilizer = None


def _init_worker(qasm_path, n_data, n_ancilla, pin_threads):
    """Runs once per worker process. Parses the QASM here rather than
    shipping a QuantumCircuit through pickle for every shot."""
    global _QC, _N_DATA, _N_ANCILLA, _QrackStabilizer

    if pin_threads:
        # Set before the pyqrack import so the native library sees them at
        # load time. We are already saturating cores with processes; any
        # intra-process threading underneath is pure oversubscription.
        # (Qrack's own concurrency level is not necessarily governed by
        # these -- verify against your build if you see contention.)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(var, "1")

    from pyqrack import QrackStabilizer
    from qiskit import QuantumCircuit

    _QrackStabilizer = QrackStabilizer
    _QC = QuantumCircuit.from_qasm_file(qasm_path)
    _N_DATA = n_data
    _N_ANCILLA = n_ancilla


def _run_one(task):
    """One independent weak-simulation trial. Returns (index, samples) on
    acceptance, (index, None) on rejection."""
    idx, seed = task
    n_qubits = _N_DATA + _N_ANCILLA

    # Fresh instance per shot: mandatory for weak simulation, since
    # every T-gate's stochastic rounding decision needs to be an
    # independent draw, not shared state reused across "shots" of a
    # single execution.
    sim = _QrackStabilizer(n_qubits)
    try:
        if seed is not None:
            # Must precede any gate application. Guarded because .seed()
            # is not present on every pyqrack revision; without it Qrack
            # seeds each new instance from std::random_device at
            # construction time, which is post-fork and therefore still
            # independent per worker -- just not reproducible.
            seed_fn = getattr(sim, "seed", None)
            if seed_fn is not None:
                seed_fn(seed)

        sim.run_qiskit_circuit(_QC, shots=0)

        syndrome = [sim.m(q) for q in range(_N_DATA, n_qubits)]
        if not all(s == 0 for s in syndrome):
            return idx, None
        return idx, [sim.m(q) for q in range(_N_DATA)]
    finally:
        # Release the simulator ID immediately rather than waiting on GC;
        # a long-lived worker chewing through thousands of shots should
        # not accumulate live sids.
        del sim


def run_post_selected(qasm_path, n_data, n_ancilla, shots, out_path=None,
                      workers=None, seed=None, chunksize=None,
                      pin_threads=True, progress_every=0):
    """Run `shots` independent, fresh, top-to-bottom weak-simulation trials
    of the circuit at qasm_path, keeping only shots where every ancilla
    measures 0 (the "zero syndrome" post-selection condition). Trials are
    distributed across `workers` processes.

    Assumes the convention used in the reference circuits: data qubits are
    indices [0, n_data), ancillas are the following n_ancilla indices. This
    is a real assumption about register layout, not something inferred
    automatically -- verify it holds for any new circuit (e.g. by checking
    that the low-index qubits have much higher per-qubit gate counts, as
    they do here: ~198 gates/qubit for data vs ~20 gates/qubit for
    ancillas in the reference circuit) before trusting results from a
    circuit built differently.

    If `seed` is given, shot i is seeded deterministically from it and the
    accepted-sample list is returned in shot order, so the run is
    reproducible and independent of worker count and scheduling. If it is
    None, each instance self-seeds from system entropy.
    """
    n_qubits = n_data + n_ancilla

    # Validate the circuit once, in the parent, before spinning up a pool.
    from qiskit import QuantumCircuit
    qc = QuantumCircuit.from_qasm_file(qasm_path)
    if qc.num_qubits != n_qubits:
        raise ValueError(
            f"circuit has {qc.num_qubits} qubits, expected {n_qubits} "
            f"(n_data={n_data} + n_ancilla={n_ancilla})"
        )
    del qc

    if workers is None:
        workers = os.cpu_count() or 1
    workers = max(1, min(workers, shots))

    if seed is None:
        tasks = [(i, None) for i in range(shots)]
    else:
        # Widely-spaced distinct seeds; avoids adjacent-seed correlation
        # in whatever PRNG the build happens to use.
        tasks = [(i, (seed + i * 0x9E3779B97F4A7C15) & 0xFFFFFFFF)
                 for i in range(shots)]

    if chunksize is None:
        # Big enough to amortize IPC, small enough that stragglers at the
        # tail don't leave cores idle.
        chunksize = max(1, shots // (workers * 8))

    accepted = {}
    done = 0
    t0 = time.perf_counter()

    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp
    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(qasm_path, n_data, n_ancilla, pin_threads),
    ) as pool:
        for idx, samples in pool.imap_unordered(_run_one, tasks, chunksize):
            done += 1
            if samples is not None:
                accepted[idx] = samples
            if progress_every and done % progress_every == 0:
                rate = done / (time.perf_counter() - t0)
                print(f"  {done}/{shots} shots, {len(accepted)} accepted, "
                      f"{rate:.2f} shots/s", flush=True)

    elapsed = time.perf_counter() - t0
    accepted_samples = [accepted[i] for i in sorted(accepted)]

    result = {
        "qasm_path": qasm_path,
        "shots": shots,
        "workers": workers,
        "seed": seed,
        "accepted": len(accepted_samples),
        "acceptance_rate": len(accepted_samples) / shots,
        "elapsed_seconds": elapsed,
        "seconds_per_shot": elapsed / shots,   # wall-clock, i.e. throughput
        "shots_per_second": shots / elapsed,
        "accepted_samples": accepted_samples,
    }

    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f)

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("qasm_path")
    p.add_argument("--n-data", type=int, default=70)
    p.add_argument("--n-ancilla", type=int, default=27)
    p.add_argument("--shots", type=int, default=500)
    p.add_argument("--workers", type=int, default=None,
                   help="worker processes (default: os.cpu_count())")
    p.add_argument("--seed", type=int, default=None,
                   help="base seed; makes the run reproducible and "
                        "independent of worker count")
    p.add_argument("--chunksize", type=int, default=None)
    p.add_argument("--no-pin-threads", dest="pin_threads",
                   action="store_false",
                   help="don't force *_NUM_THREADS=1 in workers")
    p.add_argument("--progress-every", type=int, default=0,
                   help="print running totals every N completed shots")
    p.add_argument("--out", default="dcs_post_select_result.json")
    args = p.parse_args()

    result = run_post_selected(
        args.qasm_path, args.n_data, args.n_ancilla, args.shots,
        out_path=args.out, workers=args.workers, seed=args.seed,
        chunksize=args.chunksize, pin_threads=args.pin_threads,
        progress_every=args.progress_every,
    )
    print(
        f"{result['accepted']}/{result['shots']} accepted "
        f"({100 * result['acceptance_rate']:.1f}%), "
        f"{result['elapsed_seconds']:.1f}s total on "
        f"{result['workers']} workers, "
        f"{result['shots_per_second']:.2f} shots/s"
    )
    print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
