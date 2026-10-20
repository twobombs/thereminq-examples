# -*- coding: us-ascii -*-
# 27-Qubit 3x3x3 Macroscopic Grid RCS Benchmark (27 Patches, 729 Qubits Total)
# High-Throughput Volumetric Engine with XEB + HOG Verification
#
# REVISION 87 - RANDOM CIRCUIT SAMPLING CONVERSION (from Rev 86 + CLONE FIX)
#
# CHANGES (Rev 87):
# - BUGFIX (Measurement Collapse): measure_shots is projective. If we sample
#   first, the state collapses, and subsequent prob_perm reads trivially return
#   1.0. If we pull prob_all, we move 536MB of amplitudes over PCIe per patch.
#   Fix: The Clone-and-Discard pattern. We clone the simulator in VRAM via
#   clone_sid, collapse the clone with measure_shots, immediately delete the
#   clone to free HBM, and query prob_perm on the pristine original.
# - CLONE SAFETY PROBE: A lightweight Pauli <Z> check on the clone verifies
#   that PyQrack actually duplicated the state instead of silently falling back
#   to a fresh |0> state due to API drift.
# - PHYSICS SWAP: TFIM Trotter annealing replaced with Random Circuit Sampling
#   (RCS). Each layer = Haar-random SU(2) on every qubit followed by a
#   staggered CZ pattern cycling through 6 edge partitions.
# - VERIFICATION: Per-patch linear XEB fidelity and HOG computed.
# - RUNTIME AUTODETECT EXTENDED: Pauli codes, ANGLE_SCALE, measure_shots
#   bit packing, and prob_perm ordering.
# - REPRODUCIBILITY: Per-patch circuits seeded via np.random.SeedSequence for
#   cryptographically safe independent entropy streams invariant under re-topology.

import os
import sys
import gc
import csv
import json
import time
import math
import numpy as np
import multiprocessing as mp
import multiprocessing.connection
from typing import List, Tuple, Dict, Any

# --- GLOBAL CONFIGURATION ---
GRID_X, GRID_Y, GRID_Z = 3, 3, 3
TOTAL_PATCHES = GRID_X * GRID_Y * GRID_Z
QUBITS_PER_PATCH = 27

# Topography tuning for raw statevectors
GPUS_AVAILABLE = 1
WORKERS_PER_GPU = 1
TOTAL_WORKERS = GPUS_AVAILABLE * WORKERS_PER_GPU

# Porter-Thomas ideal HOG score: (1 + ln 2) / 2
PT_IDEAL_HOG = (1.0 + math.log(2.0)) / 2.0

# =====================================================================
# ENVIRONMENT - set before pyqrack import
# =====================================================================
os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"

# =====================================================================
# PURE FUNCTIONS (Math & Topology)
# =====================================================================
def generate_27q_lattice_subvolume() -> Tuple[List[Tuple[int, int]], Dict[str, List[int]]]:
    lx, ly, lz = 3, 3, 3
    edges: List[Tuple[int, int]] = []
    boundaries: Dict[str, List[int]] = {
        "+X": [], "-X": [], "+Y": [], "-Y": [], "+Z": [], "-Z": []
    }

    for x in range(lx):
        for y in range(ly):
            for z in range(lz):
                idx = x * (ly * lz) + y * lz + z
                if x < lx - 1: edges.append((idx, (x + 1) * (ly * lz) + y * lz + z))
                if y < ly - 1: edges.append((idx, x * (ly * lz) + (y + 1) * lz + z))
                if z < lz - 1: edges.append((idx, x * (ly * lz) + y * lz + (z + 1)))

                if x == 0: boundaries["-X"].append(idx)
                if x == lx - 1: boundaries["+X"].append(idx)
                if y == 0: boundaries["-Y"].append(idx)
                if y == ly - 1: boundaries["+Y"].append(idx)
                if z == 0: boundaries["-Z"].append(idx)
                if z == lz - 1: boundaries["+Z"].append(idx)

    return edges, boundaries


def generate_cz_patterns() -> List[List[Tuple[int, int]]]:
    lx, ly, lz = 3, 3, 3

    def idx(x, y, z):
        return x * (ly * lz) + y * lz + z

    patterns: List[List[Tuple[int, int]]] = []
    for axis in range(3):
        for parity in range(2):
            edges: List[Tuple[int, int]] = []
            for x in range(lx):
                for y in range(ly):
                    for z in range(lz):
                        c = (x, y, z)
                        if c[axis] % 2 != parity:
                            continue
                        if axis == 0 and x < lx - 1:
                            edges.append((idx(x, y, z), idx(x + 1, y, z)))
                        elif axis == 1 and y < ly - 1:
                            edges.append((idx(x, y, z), idx(x, y + 1, z)))
                        elif axis == 2 and z < lz - 1:
                            edges.append((idx(x, y, z), idx(x, y, z + 1)))
            if edges:
                patterns.append(edges)
    return patterns

# =====================================================================
# WORKER PROCESS LOGIC
# =====================================================================
def gpu_worker_process(
    rank: int,
    workers_per_gpu: int,
    assigned_patches: List[int],
    conn: mp.connection.Connection,
    total_layers: int,
    shots: int,
    measure_every: int,
    master_seed: int
):
    os.environ["PYQRACK_SHARED_LIB_PATH"] = "/usr/local/lib/qrack/libqrack_pinvoke.so"
    os.environ["OCL_ICD_PLATFORM_SORT"] = "none"

    physical_gpu_index = rank // workers_per_gpu
    os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(physical_gpu_index)
    os.environ["QRACK_QPAGER_DEVICES"] = str(physical_gpu_index)
    os.environ["QRACK_QUNITMULTI_DEVICES"] = str(physical_gpu_index)
    os.environ["QRACK_MAX_ALLOC_MB"] = "64000"
    os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"

    import pyqrack
    from pyqrack import QrackSimulator

    sims = {}

    try:
        # --- PAULI CODE AUTODETECT ---
        _THRESH = 0.5

        # Z Probe
        _probe_z = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
        vals0_z = {}
        for _code in range(8):
            try: vals0_z[_code] = _probe_z.pauli_expectation([0], [_code])
            except Exception: pass
        _probe_z.x(0)
        PZ, SIGN_Z = None, None
        for _code, v0 in vals0_z.items():
            try: v1 = _probe_z.pauli_expectation([0], [_code])
            except Exception: continue
            if abs(v0) > _THRESH and abs(v1) > _THRESH and (v0 * v1) < 0:
                PZ = _code
                SIGN_Z = 1.0 if v0 > 0 else -1.0
                break
        if PZ is None:
            raise RuntimeError("Fatal: could not autodetect PZ code")
        del _probe_z

        # X Probe
        _probe_x = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
        _probe_x.h(0)
        vals0_x = {}
        for _code in range(8):
            if _code == PZ: continue
            try: vals0_x[_code] = _probe_x.pauli_expectation([0], [_code])
            except Exception: pass
        _probe_x.z(0)
        PX, SIGN_X = None, None
        for _code, v0 in vals0_x.items():
            try: v1 = _probe_x.pauli_expectation([0], [_code])
            except Exception: continue
            if abs(v0) > _THRESH and abs(v1) > _THRESH and (v0 * v1) < 0:
                PX = _code
                SIGN_X = 1.0 if v0 > 0 else -1.0
                break
        if PX is None:
            raise RuntimeError("Fatal: could not autodetect PX code")
        del _probe_x

        # Y Probe
        _probe_y = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
        _c, _s = math.cos(math.pi / 4.0), math.sin(math.pi / 4.0)
        _probe_y.mtrx([complex(_c, 0), complex(0, _s), complex(0, _s), complex(_c, 0)], 0)
        vals0_y = {}
        for _code in range(8):
            if _code in (PX, PZ): continue
            try: vals0_y[_code] = _probe_y.pauli_expectation([0], [_code])
            except Exception: pass
        _probe_y.z(0)
        PY, SIGN_Y = None, None
        for _code, v0 in vals0_y.items():
            try: v1 = _probe_y.pauli_expectation([0], [_code])
            except Exception: continue
            if abs(v0) > _THRESH and abs(v1) > _THRESH and (v0 * v1) < 0:
                PY = _code
                SIGN_Y = 1.0 if v0 > 0 else -1.0
                break
        if PY is None:
            raise RuntimeError("Fatal: could not autodetect PY code")
        del _probe_y

        # --- ANGLE CONVENTION AUTODETECT ---
        _sim_mag = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
        _sim_mag.r(PX, math.pi, 0)
        mag_check = _sim_mag.pauli_expectation([0], [PZ])
        _corrected = SIGN_Z * mag_check
        if abs(_corrected + 1.0) < 0.1:
            ANGLE_SCALE = 1.0
        elif abs(_corrected - 1.0) < 0.1:
            ANGLE_SCALE = 0.5
        else:
            raise RuntimeError(
                f"Fatal: r(PX,pi) returned ambiguous SIGN_Z*<Z> = {_corrected:.6f}; "
                f"expected ~+1.0 or ~-1.0"
            )
        del _sim_mag

        # --- SAMPLING API AUTODETECT ---
        _probe_bo = QrackSimulator(qubit_count=2, is_binary_decision_tree=False)
        _probe_bo.x(0)

        try:
            _shots_bo = list(_probe_bo.measure_shots([0, 1], 16))
        except Exception as e:
            raise RuntimeError(f"Fatal: measure_shots unavailable: {e}")
        if all(s == 1 for s in _shots_bo):
            SHOT_LSB_FIRST = True
        elif all(s == 2 for s in _shots_bo):
            SHOT_LSB_FIRST = False
        else:
            raise RuntimeError(f"Fatal: measure_shots bit-order probe ambiguous: {_shots_bo[:4]}")

        PERM_INDEX_ALIGNED = None
        try:
            _pp = float(_probe_bo.prob_perm([0, 1], [True, False]))
        except Exception as e:
            raise RuntimeError(f"Fatal: prob_perm unavailable: {e}")
        if abs(_pp - 1.0) < 0.01:
            PERM_INDEX_ALIGNED = True
        else:
            _pp_rev = float(_probe_bo.prob_perm([0, 1], [False, True]))
            if abs(_pp_rev - 1.0) < 0.01:
                PERM_INDEX_ALIGNED = False
            else:
                raise RuntimeError(
                    f"Fatal: prob_perm ordering probe ambiguous "
                    f"(aligned={_pp:.4f}, reversed={_pp_rev:.4f})"
                )
        del _probe_bo

        def apply_rx(sim, theta, q):
            sim.r(PX, float(theta) * ANGLE_SCALE, q)

        def apply_ry(sim, theta, q):
            sim.r(PY, float(theta) * ANGLE_SCALE, q)

        def apply_rz(sim, theta, q):
            sim.r(PZ, float(theta) * ANGLE_SCALE, q)

        def apply_haar_su2(sim, q, rng):
            alpha = rng.uniform(0.0, 2.0 * math.pi)
            beta  = math.acos(rng.uniform(-1.0, 1.0))
            gamma = rng.uniform(0.0, 2.0 * math.pi)
            apply_rz(sim, gamma, q)
            apply_ry(sim, beta, q)
            apply_rz(sim, alpha, q)

        def apply_cz_layer(sim, edges):
            for q1, q2 in edges:
                sim.mcz([q1], q2)

        _QLIST = list(range(QUBITS_PER_PATCH))
        _DIM = float(2 ** QUBITS_PER_PATCH)
        _HOG_THRESHOLD = math.log(2.0) / _DIM

        def bits_from_shot(s_val: int) -> List[bool]:
            if SHOT_LSB_FIRST:
                bits = [bool((s_val >> i) & 1) for i in range(QUBITS_PER_PATCH)]
            else:
                bits = [bool((s_val >> (QUBITS_PER_PATCH - 1 - i)) & 1)
                        for i in range(QUBITS_PER_PATCH)]
            if not PERM_INDEX_ALIGNED:
                bits = bits[::-1]
            return bits

        def xeb_hog_measure(sim) -> Dict[str, Any]:
            # The Clone-and-Discard Pattern with Sanity Probe
            try:
                sim_clone = QrackSimulator(clone_sid=sim.quid)
            except Exception as e:
                raise RuntimeError(f"Fatal: clone_sid constructor failed: {e}")

            # SANITY CHECK: Did it actually clone? 
            # If PyQrack silently fell back to initializing a fresh |0...0> state,
            # <Z> on q0 will be exactly +1.0 (or -1.0 depending on PZ parity).
            # Because we just applied a Haar SU(2) layer, the chance of this being
            # exactly 1.0 in a real random state is zero to float precision.
            z_mag = sim_clone.pauli_expectation([0], [PZ])
            if abs((SIGN_Z * z_mag) - 1.0) < 1e-6:
                raise RuntimeError(
                    "Fatal: clone_sid API failed silently. The clone returned a "
                    "fresh |0> state instead of duplicating the active statevector."
                )

            samples = list(sim_clone.measure_shots(_QLIST, shots))
            
            # Immediately free the clone from VRAM/host RAM
            del sim_clone
            
            # Query probabilities non-destructively on the original superposition
            probs = np.empty(len(samples), dtype=np.float64)
            for i, s_val in enumerate(samples):
                probs[i] = float(sim.prob_perm(_QLIST, bits_from_shot(int(s_val))))
                
            xeb = _DIM * float(np.mean(probs)) - 1.0
            hog = float(np.mean(probs > _HOG_THRESHOLD))
            mean_logp = float(np.mean(np.log(np.clip(probs, 1e-300, None))))
            
            return {
                "samples": np.array(samples, dtype=np.uint32),
                "xeb": xeb,
                "hog": hog,
                "mean_logp": mean_logp,
            }

        cz_patterns = generate_cz_patterns()

        rngs = {
            p: np.random.default_rng(np.random.SeedSequence([master_seed, p]))
            for p in assigned_patches
        }

        for p in assigned_patches:
            sim = QrackSimulator(
                qubit_count=QUBITS_PER_PATCH,
                is_binary_decision_tree=False,
                is_stabilizer_hybrid=False,
                is_gpu=True,
            )
            sims[p] = sim

            try:
                _ = sim.pauli_expectation([0], [PZ])
            except Exception as e:
                raise RuntimeError(f"Fatal: VRAM/PCIe paging allocation failed on patch {p}. Driver error: {e}")

        for t in range(total_layers):
            pattern = cz_patterns[t % len(cz_patterns)]
            is_measure = (t % measure_every == 0) or (t == total_layers - 1)
            patch_data_to_master = {}

            for p in assigned_patches:
                sim = sims[p]
                rng = rngs[p]

                t_start_layer = time.perf_counter()
                for q in range(QUBITS_PER_PATCH):
                    apply_haar_su2(sim, q, rng)
                apply_cz_layer(sim, pattern)
                t_lat_layer = time.perf_counter() - t_start_layer

                t_lat_sample = 0.0
                if is_measure:
                    t_start_sample = time.perf_counter()
                    result = xeb_hog_measure(sim)
                    t_lat_sample = time.perf_counter() - t_start_sample

                    patch_data_to_master[p] = {
                        "xeb": result["xeb"],
                        "hog": result["hog"],
                        "mean_logp": result["mean_logp"],
                        "samples": result["samples"],
                        "lat_layer_ms": t_lat_layer * 1000.0,
                        "lat_sample_ms": t_lat_sample * 1000.0,
                    }

            if is_measure:
                conn.send(patch_data_to_master)
                _ack = conn.recv()

    finally:
        for p in list(sims.keys()):
            _s = sims.pop(p)
            del _s
        sims.clear()
        gc.collect()
        conn.close()

# =====================================================================
# MASTER ORCHESTRATOR
# =====================================================================
class MultiGpuRcsEngine:
    def __init__(self, master_seed: int = 1337):
        self.master_seed = master_seed
        self.intra_edges, self.boundaries = generate_27q_lattice_subvolume()
        self.cz_patterns = generate_cz_patterns()

        self.patch_coords = {}
        idx = 0
        for x in range(GRID_X):
            for y in range(GRID_Y):
                for z in range(GRID_Z):
                    self.patch_coords[idx] = (x, y, z)
                    idx += 1

        self.sample_history: Dict[int, np.ndarray] = {}
        self.aggregate_csv   = "rcs_xeb_hog_curve_multi.csv"
        self.per_patch_csv   = "rcs_xeb_hog_per_patch_multi.csv"
        self.samples_file    = "rcs_sampled_bitstrings.npz"
        self.config_file     = "rcs_config.json"

        self._init_files()

        self.worker_assignments = [[] for _ in range(TOTAL_WORKERS)]
        for i in range(TOTAL_PATCHES):
            self.worker_assignments[i % TOTAL_WORKERS].append(i)

    def _init_files(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump({
                    "grid_x": GRID_X, "grid_y": GRID_Y, "grid_z": GRID_Z,
                    "num_patches": TOTAL_PATCHES,
                    "qubits_per_patch": QUBITS_PER_PATCH,
                    "master_seed": self.master_seed,
                    "benchmark": "RCS",
                    "cz_patterns": len(self.cz_patterns),
                    "pt_ideal_xeb": 1.0,
                    "pt_ideal_hog": PT_IDEAL_HOG,
                }, f)
            with open(self.aggregate_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Depth_Layers", "XEB_Mean", "XEB_Std",
                    "XEB_Min", "XEB_Max", "HOG_Mean", "HOG_Std"
                ]).writeheader()
            with open(self.per_patch_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Patch", "XEB_Fidelity", "HOG_Score", "Mean_LogP"
                ]).writeheader()
        except Exception as e:
            print(f"[CSV] Warning: Setup configuration write failed: {e}", flush=True, file=sys.stderr)

    def _log_csvs(self, step, xeb_arr, hog_arr, per_patch):
        try:
            with open(self.aggregate_csv, mode='a', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Depth_Layers", "XEB_Mean", "XEB_Std",
                    "XEB_Min", "XEB_Max", "HOG_Mean", "HOG_Std"
                ]).writerow({
                    "Step": step,
                    "Depth_Layers": step + 1,
                    "XEB_Mean": float(np.mean(xeb_arr)),
                    "XEB_Std": float(np.std(xeb_arr)),
                    "XEB_Min": float(np.min(xeb_arr)),
                    "XEB_Max": float(np.max(xeb_arr)),
                    "HOG_Mean": float(np.mean(hog_arr)),
                    "HOG_Std": float(np.std(hog_arr)),
                })
            with open(self.per_patch_csv, mode='a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=[
                    "Step", "Patch", "XEB_Fidelity", "HOG_Score", "Mean_LogP"
                ])
                for p in sorted(per_patch.keys()):
                    payload = per_patch[p]
                    w.writerow({
                        "Step": step, "Patch": p,
                        "XEB_Fidelity": payload["xeb"],
                        "HOG_Score": payload["hog"],
                        "Mean_LogP": payload["mean_logp"],
                    })
        except Exception as e:
            print(f"[CSV] Warning: Log write failed: {e}", flush=True, file=sys.stderr)

    def _save_samples(self):
        try:
            np.savez_compressed(
                self.samples_file,
                **{f"step_{k:04d}": v for k, v in self.sample_history.items()}
            )
        except Exception as e:
            print(f"[Checkpoint] Warning: Failed to save samples: {e}", flush=True, file=sys.stderr)

    def run(self, total_layers: int, shots: int = 500, measure_every: int = 1):
        if total_layers < 1:
            raise ValueError("total_layers must be at least 1")
        if measure_every < 1:
            raise ValueError("measure_every must be a positive integer")
        if shots < 1:
            raise ValueError("shots must be a positive integer")

        total_qubits = TOTAL_PATCHES * QUBITS_PER_PATCH
        print(f"[Engine] RCS/XEB/HOG | {TOTAL_PATCHES} patches, {total_qubits} qubits, "
              f"{GPUS_AVAILABLE} GPUs ({WORKERS_PER_GPU} workers/GPU), "
              f"{total_layers} layers, {shots} shots/patch/measure", flush=True)
        print(f"[Engine] Porter-Thomas targets: XEB -> 1.0000, HOG -> {PT_IDEAL_HOG:.4f}", flush=True)

        active_ranks = [r for r in range(TOTAL_WORKERS) if self.worker_assignments[r]]

        workers = []
        pipes   = []

        for rank in active_ranks:
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=gpu_worker_process,
                args=(rank, WORKERS_PER_GPU, self.worker_assignments[rank], child_conn,
                      total_layers, shots, measure_every, self.master_seed)
            )
            p.start()
            child_conn.close()
            workers.append(p)
            pipes.append(parent_conn)

        try:
            for t in range(total_layers):
                is_measure = (t % measure_every == 0) or (t == total_layers - 1)
                if not is_measure:
                    continue

                t0 = time.perf_counter()

                # --- GATHER ---
                per_patch = {}
                max_lat_layer = 0.0
                max_lat_sample = 0.0

                for conn in pipes:
                    try:
                        data = conn.recv()
                    except EOFError:
                        raise RuntimeError("Worker IPC connection lost.")
                    for p, payload in data.items():
                        per_patch[p] = payload
                        max_lat_layer = max(max_lat_layer, payload["lat_layer_ms"])
                        max_lat_sample = max(max_lat_sample, payload["lat_sample_ms"])

                if len(per_patch) != TOTAL_PATCHES:
                    raise RuntimeError(
                        f"Fatal: IPC gather incomplete. "
                        f"Expected {TOTAL_PATCHES} patches, got {len(per_patch)}."
                    )

                xeb_arr = np.array([per_patch[p]["xeb"] for p in sorted(per_patch)], dtype=np.float64)
                hog_arr = np.array([per_patch[p]["hog"] for p in sorted(per_patch)], dtype=np.float64)

                step_samples = np.stack(
                    [per_patch[p]["samples"] for p in sorted(per_patch)]
                )
                self.sample_history[t] = step_samples

                if len(self.sample_history) % 10 == 0:
                    self._save_samples()

                print(f"Step {t:03d} | XEB: {np.mean(xeb_arr):+.4f} "
                      f"+/- {np.std(xeb_arr):.4f} | "
                      f"HOG: {np.mean(hog_arr):.4f} (PT {PT_IDEAL_HOG:.4f}) | "
                      f"Lat(Layer/Sample): {max_lat_layer:5.1f}/{max_lat_sample:6.1f}ms | "
                      f"{time.perf_counter() - t0:.2f}s", flush=True)

                self._log_csvs(t, xeb_arr, hog_arr, per_patch)

                # --- SCATTER ---
                for conn in pipes:
                    conn.send(True)

        finally:
            for conn in pipes:
                try: conn.close()
                except Exception: pass

            if self.sample_history:
                self._save_samples()
                print(f"\n[Master] Dumped sampled bitstrings to {self.samples_file}", flush=True)

            for p in workers:
                try:
                    p.join(timeout=15)
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=3)
                        if p.is_alive():
                            p.kill()
                except Exception: 
                    pass

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    engine = MultiGpuRcsEngine(master_seed=1337)
    try:
        engine.run(
            total_layers=24,
            shots=500,
            measure_every=1
        )
    except KeyboardInterrupt:
        pass
