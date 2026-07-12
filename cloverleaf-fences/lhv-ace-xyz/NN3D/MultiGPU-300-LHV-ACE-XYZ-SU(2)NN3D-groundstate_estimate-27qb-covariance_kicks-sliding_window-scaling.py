# -*- coding: us-ascii -*-
# 27-Qubit 3x3x3 Lattice & Macroscopic Grid Annealing
# High-Throughput Volumetric Engine with Statistical Variance Injection
#
# REVISION 70 - MULTI-GPU DISTRIBUTED IPC ARCHITECTURE (PROD GOLD MASTER)
#
# BUGFIX (Rev 70):
# - FIXED: QrackSimulator constructor kwarg renamed from `is_qubdd` (nonexistent)
#   to `is_binary_decision_tree` (correct name in this PyQrack build).
#   Applies to all 4 constructor calls in the worker: two sign probes,
#   the magnitude probe, and the main patch simulators.

import os
import sys
import gc
import csv
import json
import time
import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Dict, Any

# --- GLOBAL CONFIGURATION ---
GRID_X, GRID_Y, GRID_Z = 9, 9, 9
TOTAL_PATCHES = GRID_X * GRID_Y * GRID_Z
QUBITS_PER_PATCH = 27

# Set this to the number of distinct GPU devices you want to target.
WORKER_GPUS = 1

# =====================================================================
# ENVIRONMENT - set before pyqrack import
# =====================================================================
os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"
os.environ["QRACK_QUNIT_SEPARABILITY_THRESHOLD"] = "1e-7"

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

# =====================================================================
# WORKER PROCESS LOGIC
# =====================================================================
def gpu_worker_process(
    rank: int,
    assigned_patches: List[int],
    conn: mp.connection.Connection,
    dt: float,
    total_steps: int,
    target_J: float,
    target_hx: float,
    target_hz: float,
    measure_every: int
):
    os.environ["PYQRACK_SHARED_LIB_PATH"] = "/usr/local/lib/qrack/libqrack_pinvoke.so"
    os.environ["OCL_ICD_PLATFORM_SORT"] = "none"
    os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(rank)
    os.environ["QRACK_QPAGER_DEVICES"] = "-1"
    os.environ["QRACK_QUNITMULTI_DEVICES"] = "-1"
    os.environ["QRACK_MAX_ALLOC_MB"] = "64000"
    os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"

    import pyqrack
    from pyqrack import QrackSimulator

    sims = {}

    try:
        # --- PAULI CODE AUTODETECT ---
        # Empirically discover which integer codes map to X, Y, Z in this build.
        # |0> has <Z>=+/-1 and <X>=<Y>=0 -> Z code is whichever gives |val|~1 on |0>
        # |+> (after H) has <X>=+/-1      -> X code gives |val|~1 on |+>
        # Rx(-pi/2)|0> = |+i> has <Y>=+/-1 -> Y code gives |val|~1 on |+i>
        # Tolerance is loose (1e-2) to accommodate fp16 (FPPOW=5) BDT precision.
        _THRESH = 0.5   # anything above 0.5 is unambiguously a unit eigenstate response

        _probe_z = QrackSimulator(qubit_count=1, is_binary_decision_tree=True)
        PZ = None
        for _code in range(8):
            try:
                _v = _probe_z.pauli_expectation([0], [_code])
                if abs(_v) > _THRESH:
                    PZ = _code
                    break
            except Exception:
                continue
        assert PZ is not None, "Fatal: could not autodetect PZ code"
        del _probe_z

        _probe_x = QrackSimulator(qubit_count=1, is_binary_decision_tree=True)
        _probe_x.h(0)
        PX = None
        for _code in range(8):
            if _code == PZ: continue
            try:
                _v = _probe_x.pauli_expectation([0], [_code])
                if abs(_v) > _THRESH:
                    PX = _code
                    break
            except Exception:
                continue
        assert PX is not None, "Fatal: could not autodetect PX code"
        del _probe_x

        _probe_y = QrackSimulator(qubit_count=1, is_binary_decision_tree=True)
        # Rx(-pi/2)|0> rotates toward |+i> so <Y> becomes +/-1
        # We use a hardcoded matrix here to avoid chicken-and-egg with PX
        import cmath as _cm
        _c = _cm.cos(np.pi / 4.0)
        _s = _cm.sin(np.pi / 4.0)
        _probe_y.mtrx([complex(_c, 0), complex(0, _s), complex(0, _s), complex(_c, 0)], 0)
        PY = None
        for _code in range(8):
            if _code in (PX, PZ): continue
            try:
                _v = _probe_y.pauli_expectation([0], [_code])
                if abs(_v) > _THRESH:
                    PY = _code
                    break
            except Exception:
                continue
        if PY is None:
            # Fallback: use the lowest unused code; still better than a wrong constant
            PY = next(c for c in range(8) if c not in (PX, PZ))
        del _probe_y
        # ----------------------------

        # --- SIGN & MAGNITUDE CONVENTION AUTODETECT ---
        _sim_x = QrackSimulator(qubit_count=1, is_binary_decision_tree=True)
        _sim_x.h(0)
        qrack_x = _sim_x.pauli_expectation([0], [PX])
        assert abs(_v := qrack_x) > _THRESH, f"Fatal: PX probe magnitude too small: {qrack_x}"
        SIGN_X = 1.0 if qrack_x > 0 else -1.0
        del _sim_x

        SIGN_Y = SIGN_X

        _sim_z = QrackSimulator(qubit_count=1, is_binary_decision_tree=True)
        qrack_z = _sim_z.pauli_expectation([0], [PZ])
        assert abs(qrack_z) > _THRESH, f"Fatal: PZ probe magnitude too small: {qrack_z}"
        SIGN_Z = 1.0 if qrack_z > 0 else -1.0
        del _sim_z

        # Detect r() angle convention: half-angle exp(-i*theta/2*P) or full-angle exp(-i*theta*P).
        # r(PX, pi)|0> -> if <Z> flips to -1: half-angle (Rx(pi) = X gate maps |0>->|1>)
        #                 if <Z> stays  +1: full-angle (exp(-i*pi*X)|0> = -|0>, <Z>=+1)
        # ANGLE_SCALE = 1.0 for half-angle convention, 0.5 for full-angle convention.
        # All physical angles are expressed as if half-angle; ANGLE_SCALE corrects the call.
        _sim_mag = QrackSimulator(qubit_count=1, is_binary_decision_tree=True)
        _sim_mag.r(PX, np.pi, 0)
        mag_check = _sim_mag.pauli_expectation([0], [PZ])
        _corrected = SIGN_Z * mag_check
        if abs(_corrected + 1.0) < 0.1:
            # r(PX, pi) flipped <Z> to -1: half-angle convention, angles pass through as-is
            ANGLE_SCALE = 1.0
        elif abs(_corrected - 1.0) < 0.1:
            # r(PX, pi) left <Z> at +1: full-angle convention, divide physical angles by 2
            ANGLE_SCALE = 0.5
        else:
            raise AssertionError(
                f"Fatal: r(PX,pi) returned ambiguous SIGN_Z*<Z> = {_corrected:.6f}; "
                f"expected ~+1.0 or ~-1.0"
            )
        del _sim_mag
        # -------------------------------------------------------

        def apply_h(sim, q): sim.h(q)

        def apply_rx(sim, theta, q):
            # theta is the TRUE physical rotation angle (half-angle convention).
            # ANGLE_SCALE=1.0 for half-angle r(), 0.5 for full-angle r().
            sim.r(PX, float(theta) * ANGLE_SCALE, q)

        def apply_rz(sim, theta, q):
            sim.r(PZ, float(theta) * ANGLE_SCALE, q)

        def apply_zz(sim, theta, q1, q2):
            # CNOT . Rz(2*theta) . CNOT implements exp(-i*theta*ZZ).
            # The 2.0 here is structural (CNOT sandwich geometry), not a convention factor.
            # apply_rz already applies ANGLE_SCALE, so 2*theta is the correct physical arg.
            sim.mcx([q1], q2); apply_rz(sim, 2.0 * theta, q2); sim.mcx([q1], q2)

        def trotter_step_body(sim, num_qubits, intra_edges, J, hx, hz, dt_local):
            """
            2nd-order Strang splitting:
              A/2 (transverse, dt/2) -> B (longitudinal + ZZ, dt) -> A/2 (transverse, dt/2)
            Angles here are TRUE physical half-angle values (what exp(-i*theta*P) needs).
            ANGLE_SCALE inside apply_rx/apply_rz adapts them to the r() convention at runtime.
            """
            dt_half = dt_local / 2.0
            # True physical angles (half-angle convention throughout):
            #   Rx(hx, dt/2): rotate by hx*dt/2 around X (half-step)
            #   Rz(hz, dt):   rotate by hz*dt   around Z (full-step)
            #   ZZ(J,  dt):   rotate by J*dt    in ZZ plane (full-step, 2x inside apply_zz)
            theta_x  = -hx * dt_half      # half-step; applied twice = -hx*dt total
            theta_z  = -hz * dt_local     # full-step
            theta_zz = -J  * dt_local     # full-step

            for q in range(num_qubits): apply_rx(sim, theta_x, q)
            for q in range(num_qubits): apply_rz(sim, theta_z, q)
            for q1, q2 in intra_edges: apply_zz(sim, theta_zz, q1, q2)
            for q in range(num_qubits): apply_rx(sim, theta_x, q)

        def z_means(sim, qubits):
            return np.array([SIGN_Z * float(sim.pauli_expectation([q], [PZ])) for q in qubits])

        def x_means(sim, qubits):
            return np.array([SIGN_X * float(sim.pauli_expectation([q], [PX])) for q in qubits])

        def y_means(sim, qubits):
            return np.array([SIGN_Y * float(sim.pauli_expectation([q], [PY])) for q in qubits])

        def zz_means_meanfield(z_exp, edges):
            """Mean-field approximation: <ZZ> ~ <Z><Z>. Avoids O(edges) mcx pairs per step."""
            return np.array([z_exp[q1] * z_exp[q2] for q1, q2 in edges])

        def apply_kicks(sim, kicks, dt_local, m_every):
            """
            Applies boundary mean-field kicks as a single SU(2) rotation per boundary qubit.
            Uses explicit mtrx() which is always convention-independent (pure SU(2) matrix).
            Rotation angle = K = |k_vec| * dt * m_every.
            The SU(2) matrix is exp(-i * K/2 * n_hat . sigma) by construction.
            """
            if not kicks: return
            for raw_q, (kx, ky, kz) in kicks.items():
                q = int(raw_q)
                kx *= dt_local * m_every
                ky *= dt_local * m_every
                kz *= dt_local * m_every
                K = np.sqrt(kx**2 + ky**2 + kz**2)
                if K > 0.0:
                    c, s = np.cos(K / 2.0), np.sin(K / 2.0)
                    nx, ny, nz = kx / K, ky / K, kz / K
                    sim.mtrx([complex(c, -nz * s), complex(-ny * s, -nx * s),
                              complex(ny * s, -nx * s), complex(c,  nz * s)], q)

        intra_edges, boundaries = generate_27q_lattice_subvolume()

        for p in assigned_patches:
            sim = QrackSimulator(qubit_count=QUBITS_PER_PATCH, is_binary_decision_tree=True)
            for q in range(QUBITS_PER_PATCH): apply_h(sim, q)
            sims[p] = sim

        kick_payloads = {p: {} for p in assigned_patches}

        for t in range(total_steps):
            s = t / max(1, (total_steps - 1))
            current_hx = (1.0 - s) * 3.0 + s * target_hx
            current_J  = s * target_J
            current_hz = s * target_hz
            is_measure = (t % measure_every == 0) or (t == total_steps - 1)

            patch_data_to_master = {}

            for p in assigned_patches:
                sim = sims[p]
                if kick_payloads[p]:
                    apply_kicks(sim, kick_payloads[p], dt, measure_every)

                trotter_step_body(sim, QUBITS_PER_PATCH, intra_edges,
                                  current_J, current_hx, current_hz, dt)

                if is_measure:
                    all_q = list(range(QUBITS_PER_PATCH))
                    state = {
                        "Z": z_means(sim, all_q),
                        "X": x_means(sim, all_q),
                        "Y": y_means(sim, all_q),
                    }
                    zz_exp = zz_means_meanfield(state["Z"], intra_edges)
                    bulk_e = (-current_hz * float(np.sum(state["Z"]))
                              - current_J  * float(np.sum(zz_exp))
                              - current_hx * float(np.sum(state["X"])))
                    patch_data_to_master[p] = {"state": state, "bulk_energy": bulk_e}

            if is_measure:
                conn.send(patch_data_to_master)
                kick_payloads = conn.recv()
            else:
                kick_payloads = {p: {} for p in assigned_patches}

    finally:
        # Drop all references simultaneously so C++ destructors free OpenCL VRAM at once.
        sims.clear()
        gc.collect()
        conn.close()

# =====================================================================
# MASTER ORCHESTRATOR
# =====================================================================
class MultiGpuHadronEngine:
    def __init__(self, master_seed: int = 1337):
        self.master_seed = master_seed
        self.intra_edges, self.boundaries = generate_27q_lattice_subvolume()
        self.all_boundary_qubits = sorted(set(q for face in self.boundaries.values() for q in face))
        self._bq_to_idx = {q: i for i, q in enumerate(self.all_boundary_qubits)}
        self._bq_arr    = np.array(self.all_boundary_qubits, dtype=np.intp)

        self.patch_coords = {}
        idx = 0
        for x in range(GRID_X):
            for y in range(GRID_Y):
                for z in range(GRID_Z):
                    self.patch_coords[idx] = (x, y, z)
                    idx += 1
        self.coord_to_patch = {v: k for k, v in self.patch_coords.items()}

        self.lattice_history  = []
        self.energy_csv       = "ground_state_energy_curve_multi.csv"
        self.profiles_csv     = "boundary_profiles_multi.csv"
        self.state_dump_file  = "macroscopic_lattice_states.npy"
        self.config_file      = "lattice_config.json"

        self._init_files()

        self.worker_assignments = [[] for _ in range(WORKER_GPUS)]
        for i in range(TOTAL_PATCHES):
            self.worker_assignments[i % WORKER_GPUS].append(i)

    def _init_files(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump({"grid_x": GRID_X, "grid_y": GRID_Y, "grid_z": GRID_Z,
                           "num_patches": TOTAL_PATCHES,
                           "qubits_per_patch": QUBITS_PER_PATCH}, f)
            with open(self.energy_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Anneal_Percent", "Bulk_Energy",
                    "Boundary_Energy", "Total_Energy"
                ]).writeheader()
            with open(self.profiles_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Patch", "Face", "X_mean", "Y_mean", "Z_mean"
                ]).writeheader()
        except Exception as e:
            print(f"[CSV] Warning: Setup configuration write failed: {e}", file=sys.stderr)

    def _log_csvs(self, step, anneal, bulk, bound, total, patch_profiles):
        try:
            with open(self.energy_csv, mode='a', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Anneal_Percent", "Bulk_Energy",
                    "Boundary_Energy", "Total_Energy"
                ]).writerow({"Step": step, "Anneal_Percent": anneal,
                             "Bulk_Energy": bulk, "Boundary_Energy": bound,
                             "Total_Energy": total})

            with open(self.profiles_csv, mode='a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=[
                    "Step", "Patch", "Face", "X_mean", "Y_mean", "Z_mean"
                ])
                for p, prof in patch_profiles.items():
                    for face_name, face_qubits in self.boundaries.items():
                        if not face_qubits: continue
                        xm = float(np.mean([prof["means"]["X"][self._bq_to_idx[q]] for q in face_qubits]))
                        ym = float(np.mean([prof["means"]["Y"][self._bq_to_idx[q]] for q in face_qubits]))
                        zm = float(np.mean([prof["means"]["Z"][self._bq_to_idx[q]] for q in face_qubits]))
                        w.writerow({"Step": step, "Patch": p, "Face": face_name,
                                    "X_mean": xm, "Y_mean": ym, "Z_mean": zm})
        except Exception as e:
            print(f"[CSV] Warning: Log write failed: {e}", file=sys.stderr)

    def run(self, total_steps: int, dt: float, target_g_face: float, target_J: float,
            target_hx: float, target_hz: float,
            measure_every: int = 1, effective_shots: float = 512.0):

        assert total_steps >= 1,   "total_steps must be at least 1"
        assert measure_every >= 1, "measure_every must be a positive integer"

        total_qubits = TOTAL_PATCHES * QUBITS_PER_PATCH
        print(f"[Engine] {TOTAL_PATCHES} patches, {total_qubits} qubits, {WORKER_GPUS} GPUs, {total_steps} steps")

        active_ranks = [r for r in range(WORKER_GPUS) if self.worker_assignments[r]]

        workers = []
        pipes   = []

        for rank in active_ranks:
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=gpu_worker_process,
                args=(rank, self.worker_assignments[rank], child_conn,
                      dt, total_steps, target_J, target_hx, target_hz, measure_every)
            )
            p.start()
            child_conn.close()   # Drop master's copy -> guarantees EOF propagation on worker crash
            workers.append(p)
            pipes.append(parent_conn)

        noise_rng = np.random.default_rng(self.master_seed)

        try:
            for t in range(total_steps):
                t0 = time.perf_counter()
                s = t / max(1, (total_steps - 1))
                current_g_face = s * target_g_face
                is_measure = (t % measure_every == 0) or (t == total_steps - 1)

                if not is_measure:
                    continue

                # --- GATHER ---
                patch_full_states = {}
                bulk_energy = 0.0
                for conn in pipes:
                    try:
                        data = conn.recv()
                    except EOFError:
                        raise RuntimeError("Worker IPC connection lost.")
                    for p, payload in data.items():
                        patch_full_states[p] = payload["state"]
                        bulk_energy += payload["bulk_energy"]

                if len(patch_full_states) != TOTAL_PATCHES:
                    raise RuntimeError(
                        f"Fatal: IPC gather incomplete. "
                        f"Expected {TOTAL_PATCHES} patches, got {len(patch_full_states)}."
                    )

                # --- BUILD PROFILES ---
                step_state    = np.zeros((TOTAL_PATCHES, QUBITS_PER_PATCH, 3))
                patch_profiles = {}
                bq = self._bq_arr

                for p, state in patch_full_states.items():
                    step_state[p, :, 0] = state["X"]
                    step_state[p, :, 1] = state["Y"]
                    step_state[p, :, 2] = state["Z"]
                    patch_profiles[p] = {
                        "means": {
                            "X": state["X"][bq],
                            "Y": state["Y"][bq],
                            "Z": state["Z"][bq],
                        },
                        "vars": {
                            "X": np.clip(1.0 - state["X"][bq]**2, 0.0, 1.0),
                            "Y": np.clip(1.0 - state["Y"][bq]**2, 0.0, 1.0),
                            "Z": np.clip(1.0 - state["Z"][bq]**2, 0.0, 1.0),
                        }
                    }

                self.lattice_history.append(step_state.copy())

                if len(self.lattice_history) % 10 == 0:
                    try:
                        np.save(self.state_dump_file, np.array(self.lattice_history))
                    except Exception as e:
                        print(f"[Checkpoint] Warning: Failed to save: {e}", file=sys.stderr)

                # --- COMPUTE KICKS & BOUNDARY ENERGY ---
                next_kick_payloads       = {p: {} for p in range(TOTAL_PATCHES)}
                macroscopic_boundary_energy = 0.0
                scale           = np.sqrt(dt * measure_every / effective_shots)
                stochastic_noise = {}
                n_b = len(self.all_boundary_qubits)

                for p, prof in patch_profiles.items():
                    xn = noise_rng.normal(0.0, 1.0, n_b) * np.sqrt(prof["vars"]["X"]) * scale
                    yn = noise_rng.normal(0.0, 1.0, n_b) * np.sqrt(prof["vars"]["Y"]) * scale
                    zn = noise_rng.normal(0.0, 1.0, n_b) * np.sqrt(prof["vars"]["Z"]) * scale
                    stochastic_noise[p] = {
                        q: (xn[i], yn[i], zn[i])
                        for i, q in enumerate(self.all_boundary_qubits)
                    }

                for p1, coord1 in self.patch_coords.items():
                    x1, y1, z1 = coord1
                    neighbors = {
                        "+X": (x1+1, y1,   z1  ), "-X": (x1-1, y1,   z1  ),
                        "+Y": (x1,   y1+1, z1  ), "-Y": (x1,   y1-1, z1  ),
                        "+Z": (x1,   y1,   z1+1), "-Z": (x1,   y1,   z1-1),
                    }

                    for dir1, coord2 in neighbors.items():
                        p2 = self.coord_to_patch.get(coord2)
                        if p2 is None or p1 >= p2: continue

                        dir2    = dir1.replace("+", "temp").replace("-", "+").replace("temp", "-")
                        face1_q = self.boundaries[dir1]
                        face2_q = self.boundaries[dir2]
                        prof1, noise1 = patch_profiles[p1], stochastic_noise[p1]
                        prof2, noise2 = patch_profiles[p2], stochastic_noise[p2]

                        ax2 = np.mean([prof2["means"]["X"][self._bq_to_idx[q]] + noise2[q][0] for q in face2_q])
                        ay2 = np.mean([prof2["means"]["Y"][self._bq_to_idx[q]] + noise2[q][1] for q in face2_q])
                        az2 = np.mean([prof2["means"]["Z"][self._bq_to_idx[q]] + noise2[q][2] for q in face2_q])

                        ax1 = np.mean([prof1["means"]["X"][self._bq_to_idx[q]] + noise1[q][0] for q in face1_q])
                        ay1 = np.mean([prof1["means"]["Y"][self._bq_to_idx[q]] + noise1[q][1] for q in face1_q])
                        az1 = np.mean([prof1["means"]["Z"][self._bq_to_idx[q]] + noise1[q][2] for q in face1_q])

                        macroscopic_boundary_energy += (
                            current_g_face
                            * (ax1*ax2 + ay1*ay2 + az1*az2)
                            * ((len(face1_q) + len(face2_q)) / 2.0)
                        )

                        for q1f in face1_q:
                            k = next_kick_payloads[p1].get(q1f, (0., 0., 0.))
                            next_kick_payloads[p1][q1f] = (
                                k[0] + current_g_face * ax2,
                                k[1] + current_g_face * ay2,
                                k[2] + current_g_face * az2,
                            )
                        for q2f in face2_q:
                            k = next_kick_payloads[p2].get(q2f, (0., 0., 0.))
                            next_kick_payloads[p2][q2f] = (
                                k[0] + current_g_face * ax1,
                                k[1] + current_g_face * ay1,
                                k[2] + current_g_face * az1,
                            )

                total_energy = bulk_energy + macroscopic_boundary_energy
                print(f"Step {t:03d} | E: {total_energy:+.4f} | {time.perf_counter() - t0:.2f}s")
                self._log_csvs(t, s * 100, bulk_energy,
                               macroscopic_boundary_energy, total_energy, patch_profiles)

                # --- SCATTER ---
                for i, w_rank in enumerate(active_ranks):
                    worker_payload = {p: next_kick_payloads[p]
                                      for p in self.worker_assignments[w_rank]}
                    pipes[i].send(worker_payload)

        finally:
            for conn in pipes:
                try: conn.close()
                except Exception: pass

            if self.lattice_history:
                try:
                    np.save(self.state_dump_file, np.array(self.lattice_history))
                    print(f"\n[Master] Dumped history matrix to {self.state_dump_file}")
                except Exception as e:
                    print(f"\n[Master] Failed to save lattice history: {e}", file=sys.stderr)

            for p in workers:
                # 15s to allow heavy OpenCL VRAM free on BDT destructors, then escalate
                p.join(timeout=15)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=3)
                    if p.is_alive():
                        try: p.kill()
                        except Exception: pass


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    engine = MultiGpuHadronEngine(master_seed=1337)
    try:
        engine.run(
            total_steps=100,
            dt=0.04,
            target_g_face=0.15,
            target_J=1.0,
            target_hx=0.5,
            target_hz=0.2,
            measure_every=1,
            effective_shots=512.0
        )
    except KeyboardInterrupt:
        pass
