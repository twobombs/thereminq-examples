# -*- coding: us-ascii -*-
# 27-Qubit 3x3x3 Macroscopic Grid Annealing (27 Patches, 729 Qubits Total)
# High-Throughput Volumetric Engine with Statistical Variance Injection
#
# REVISION 88.1 - SYNTAX FIX
#
# BUGFIXES (Rev 88.1):
# - Fixed a catastrophic SyntaxError in build_patch_skqd_hamiltonian's 
#   state_to_idx dictionary comprehension.
#
# NEW (Rev 88):
# - STRICT ENVIRONMENT ORDERING: Hoisted all os.environ OpenCL/Qrack configurations 
#   before `get_numa_node_for_opencl_device` is invoked. This ensures the ICD 
#   platform loader uses OCL_ICD_PLATFORM_SORT="none" when resolving devices 
#   for the NUMA mapping, preventing device misalignment on multi-platform nodes.
# - SYSFS BDF FORMATTING: Fixed AMD topology BDF string formatting from colons 
#   (00:00:0) to the standard sysfs dot notation (00:00.0) preventing silent 
#   fallback to NUMA Node 0 on multi-socket boards.
# - COMPLETE PROBE ASSERTIONS: Added the scalar type assertion to the X and Y 
#   Pauli probes for complete PyQrack API contract enforcement.
#
# NEW (Rev 87):
# - DYNAMIC NUMA PINNING: Worker processes now parse sysfs topology trees to 
#   resolve the physical PCIe root complex of their assigned OpenCL device.
#   CPU affinity is strictly locked to the corresponding NUMA node.
# - CONTINUOUS KICK ACCUMULATION: Fixed a latent physics bug where boundary 
#   kicks were applied as a lumped sum and then zeroed out when measure_every > 1. 
#
# DEPENDENCIES: numpy, scipy, pyqrack, pyopencl (for topology resolution).

import os
import sys
import gc
import csv
import json
import time
import math
import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Dict, Any

# --- GLOBAL CONFIGURATION ---
GRID_X, GRID_Y, GRID_Z = 3, 3, 3
TOTAL_PATCHES = GRID_X * GRID_Y * GRID_Z
QUBITS_PER_PATCH = 27

# Topography tuning for raw statevectors
GPUS_AVAILABLE = 1
WORKERS_PER_GPU = 1  # 4 workers handling ~7 patches each
TOTAL_WORKERS = GPUS_AVAILABLE * WORKERS_PER_GPU

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

def get_numa_node_for_opencl_device(physical_gpu_index: int) -> int:
    """
    Resolve the NUMA node of an OpenCL device by matching its PCI BDF
    against /sys/bus/pci/devices/*/numa_node.
    
    Falls back to node 0 if resolution fails.
    """
    import pyopencl as cl
    platforms = cl.get_platforms()
    devices = [d for p in platforms for d in p.get_devices(cl.device_type.GPU)]
    if physical_gpu_index >= len(devices):
        return 0
    
    try:
        # PCIe BDF string, e.g. "0000:03:00.0"
        bdf = devices[physical_gpu_index].pci_bus_id_nv  # NVIDIA extension
    except cl.LogicError:
        try:
            # Mesa/ROCm path: parse from device topology_str or name
            # Correct sysfs format: bus(2):device(2).function(1)
            topo = devices[physical_gpu_index].topology_amd
            bdf = f"{topo.bus:02x}:{topo.device:02x}.{topo.function:x}"
        except Exception:
            return 0
    
    numa_path = f"/sys/bus/pci/devices/0000:{bdf}/numa_node"
    try:
        with open(numa_path) as f:
            node = int(f.read().strip())
        return node if node >= 0 else 0  # -1 means "unknown", treat as 0
    except OSError:
        return 0

def get_cpu_set_for_numa_node(numa_node: int) -> set:
    """Read /sys/devices/system/node/nodeN/cpulist and expand to a set of ints."""
    cpulist_path = f"/sys/devices/system/node/node{numa_node}/cpulist"
    try:
        with open(cpulist_path) as f:
            raw = f.read().strip()  # e.g. "0-15,32-47"
        cores = set()
        for part in raw.split(','):
            if '-' in part:
                lo, hi = part.split('-')
                cores.update(range(int(lo), int(hi) + 1))
            else:
                cores.add(int(part))
        return cores
    except OSError:
        return set(range(os.cpu_count() or 1))  # fallback: unrestricted

# =====================================================================
# SKQD SUBSPACE REFINEMENT (Master-side classical kernels)
# =====================================================================
def build_patch_skqd_hamiltonian(subspace: List[int],
                                 edges: List[Tuple[int, int]],
                                 J: float,
                                 hx_site: np.ndarray,
                                 hy_site: np.ndarray,
                                 hz_site: np.ndarray,
                                 num_qubits: int):
    """Sparse subspace Hamiltonian over the sampled configuration basis."""
    import scipy.sparse as sp

    N = len(subspace)
    # Fixed the missing `bs in` syntax error here
    state_to_idx = {bs: i for i, bs in enumerate(subspace)}
    use_y = bool(np.any(np.abs(hy_site) > 1e-12))
    dtype = np.complex128 if use_y else np.float64

    rows: List[int] = []
    cols: List[int] = []
    data: List[Any] = []

    for i, bs in enumerate(subspace):
        diag = 0.0
        for q1, q2 in edges:
            z1 = 1.0 - 2.0 * ((bs >> q1) & 1)
            z2 = 1.0 - 2.0 * ((bs >> q2) & 1)
            diag += -J * z1 * z2
        for q in range(num_qubits):
            zq = 1.0 - 2.0 * ((bs >> q) & 1)
            diag += -hz_site[q] * zq
        rows.append(i); cols.append(i); data.append(diag)

        for q in range(num_qubits):
            j = state_to_idx.get(bs ^ (1 << q))
            if j is None:
                continue
            elem = -float(hx_site[q])
            if use_y:
                bit = (bs >> q) & 1
                elem = complex(elem, 0.0) + float(hy_site[q]) * (1j if bit == 0 else -1j)
            rows.append(i); cols.append(j); data.append(elem)

    return sp.coo_matrix(
        (np.asarray(data, dtype=dtype), (rows, cols)), shape=(N, N)
    ).tocsr()

def skqd_configuration_recovery(seed_subspace: List[int],
                                edges: List[Tuple[int, int]],
                                J: float,
                                hx_site: np.ndarray,
                                hy_site: np.ndarray,
                                hz_site: np.ndarray,
                                num_qubits: int,
                                max_iters: int = 15,
                                tolerance: float = 1e-6,
                                max_subspace_size: int = 8192,
                                support_frac: float = 0.90,
                                label: str = "") -> Tuple[float, int]:
    """Iterative SKQD loop: diagonalize in subspace, expand support via dominant configurations."""
    from scipy.sparse.linalg import eigsh

    current = list(dict.fromkeys(seed_subspace))
    prev_energy = np.inf
    E0 = np.inf
    v0 = None

    for iteration in range(max_iters):
        if not current:
            print(f"[SKQD{label}] Catastrophic subspace collapse.", file=sys.stderr)
            break

        H_eff = build_patch_skqd_hamiltonian(
            current, edges, J, hx_site, hy_site, hz_site, num_qubits)

        if len(current) < 16:
            evals, evecs = np.linalg.eigh(H_eff.toarray())
            E0, v0 = float(evals[0]), evecs[:, 0]
        else:
            try:
                evals, evecs = eigsh(H_eff, k=1, which='SA')
                E0, v0 = float(evals[0]), evecs[:, 0]
            except Exception as e:
                print(f"[SKQD{label}] eigsh failed ({e}); dense fallback.", file=sys.stderr)
                evals, evecs = np.linalg.eigh(H_eff.toarray())
                E0, v0 = float(evals[0]), evecs[:, 0]

        if abs(prev_energy - E0) < tolerance:
            prev_energy = E0
            break
        prev_energy = E0

        probabilities = np.abs(v0) ** 2
        order = np.argsort(probabilities)[::-1]

        current_set = set(current)
        new_configurations: set = set()
        cumulative_p = 0.0
        for idx in order:
            cumulative_p += probabilities[idx]
            bs = current[int(idx)]
            for q in range(num_qubits):
                fb = bs ^ (1 << q)
                if fb not in current_set:
                    new_configurations.add(fb)
            if cumulative_p > support_frac:
                break

        if not new_configurations:
            break

        ranked_existing = [current[int(idx)] for idx in order]
        expanded = ranked_existing + list(new_configurations)
        if len(expanded) > max_subspace_size:
            expanded = expanded[:max_subspace_size]
        current = expanded

    return prev_energy, len(current)

# =====================================================================
# WORKER PROCESS LOGIC
# =====================================================================
def gpu_worker_process(
    rank: int,
    workers_per_gpu: int,
    assigned_patches: List[int],
    conn: mp.connection.Connection,
    dt: float,
    total_steps: int,
    initial_hx: float,
    target_J: float,
    target_hx: float,
    target_hz: float,
    measure_every: int,
    skqd_shots: int
):
    # Map multiple ranks to the same physical GPU device index
    physical_gpu_index = rank // workers_per_gpu
    
    # --- 1. SET ALL ENVIRONMENT VARIABLES BEFORE ANY IMPORTS OR LOGIC ---
    os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(physical_gpu_index)
    os.environ["PYQRACK_SHARED_LIB_PATH"] = "/usr/local/lib/qrack/libqrack_pinvoke.so"
    os.environ["OCL_ICD_PLATFORM_SORT"] = "none"
    os.environ["QRACK_QPAGER_DEVICES"] = str(physical_gpu_index)
    os.environ["QRACK_QUNITMULTI_DEVICES"] = str(physical_gpu_index)
    os.environ["QRACK_MAX_ALLOC_MB"] = "64000"
    os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"

    # --- 2. ENFORCE NUMA LOCALITY USING THE CONFIGURED ENVIRONMENT ---
    numa_node = get_numa_node_for_opencl_device(physical_gpu_index)
    cpu_set   = get_cpu_set_for_numa_node(numa_node)
    try:
        os.sched_setaffinity(0, cpu_set)
    except PermissionError:
        print(f"[Worker {rank}] Warning: sched_setaffinity failed (no CAP_SYS_NICE?); "
              f"NUMA locality not enforced.", file=sys.stderr)

    # --- 3. IMPORT AND INITIALIZE PYQRACK ---
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
            try: 
                v0 = _probe_z.pauli_expectation([0], [_code])
            except Exception:
                pass
            else:
                assert np.ndim(v0) == 0, f"PyQrack API mismatch: pauli_expectation returned non-scalar type {type(v0)}"
                vals0_z[_code] = v0
                
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
            try: 
                v0 = _probe_x.pauli_expectation([0], [_code])
            except Exception:
                pass
            else:
                assert np.ndim(v0) == 0, f"PyQrack API mismatch: pauli_expectation returned non-scalar type {type(v0)}"
                vals0_x[_code] = v0
                
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
            try: 
                v0 = _probe_y.pauli_expectation([0], [_code])
            except Exception:
                pass
            else:
                assert np.ndim(v0) == 0, f"PyQrack API mismatch: pauli_expectation returned non-scalar type {type(v0)}"
                vals0_y[_code] = v0
                
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
        # ----------------------------

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
        # -------------------------------------------------------

        def apply_h(sim, q): sim.h(q)

        def apply_rx(sim, theta, q):
            sim.r(PX, float(theta) * ANGLE_SCALE, q)

        def apply_ry(sim, theta, q):
            sim.r(PY, float(theta) * ANGLE_SCALE, q)

        def apply_rz(sim, theta, q):
            sim.r(PZ, float(theta) * ANGLE_SCALE, q)

        def apply_zz(sim, theta, q1, q2):
            sim.mcx([q1], q2); apply_rz(sim, 2.0 * theta, q2); sim.mcx([q1], q2)

        def trotter_step_body(sim, num_qubits, intra_edges, J, hx, hz, dt_local):
            dt_half = dt_local / 2.0

            theta_x  = -2.0 * hx * dt_half
            theta_z  = -2.0 * hz * dt_local
            theta_zz = -J * dt_local

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
            return np.array([z_exp[q1] * z_exp[q2] for q1, q2 in edges])

        def apply_kicks(sim, kicks, dt_local):
            if not kicks: return
            for raw_q, (kx, ky, kz) in kicks.items():
                q = int(raw_q)
                
                # Continuous accumulation: apply uniformly per step
                coef = -2.0 * dt_local

                theta_x = kx * coef
                theta_y = ky * coef
                theta_z = kz * coef

                if abs(theta_x) > 1e-12: apply_rx(sim, theta_x, q)
                if abs(theta_y) > 1e-12: apply_ry(sim, theta_y, q)
                if abs(theta_z) > 1e-12: apply_rz(sim, theta_z, q)

        def sample_patch_counts(sim, num_qubits, shots):
            if shots <= 0:
                return {}
            try:
                samples = sim.measure_shots(list(range(num_qubits)), int(shots))
            except Exception as e:
                print(f"[Worker {rank}] Warning: measure_shots failed ({e}); "
                      f"SKQD seeding disabled for this patch.", file=sys.stderr)
                return {}
            counts: Dict[int, int] = {}
            for s_int in samples:
                if not isinstance(s_int, (int, np.integer)):
                    raise RuntimeError(f"PyQrack API mismatch: measure_shots returned non-integer bitstring representation {type(s_int)}")
                s_int = int(s_int)
                counts[s_int] = counts.get(s_int, 0) + 1
            return counts

        intra_edges, boundaries = generate_27q_lattice_subvolume()

        for p in assigned_patches:
            sim = QrackSimulator(
                qubit_count=QUBITS_PER_PATCH,
                is_binary_decision_tree=False,
                is_gpu=True, 
            )
            for q in range(QUBITS_PER_PATCH): apply_h(sim, q)
            sims[p] = sim

            # --- VRAM PAGING SMOKE TEST ---
            try:
                _ = sim.pauli_expectation([0], [PZ])
            except Exception as e:
                raise RuntimeError(f"Fatal: VRAM/PCIe paging allocation failed on patch {p}. Driver error: {e}")

        kick_payloads = {p: {} for p in assigned_patches}

        for t in range(total_steps):
            s = t / max(1, (total_steps - 1))
            current_hx = (1.0 - s) * initial_hx + s * target_hx
            current_J  = s * target_J
            current_hz = s * target_hz
            is_final   = (t == total_steps - 1)
            is_measure = (t % measure_every == 0) or is_final

            patch_data_to_master = {}

            for p in assigned_patches:
                sim = sims[p]
                
                if kick_payloads[p]:
                    apply_kicks(sim, kick_payloads[p], dt)

                t_start_trotter = time.perf_counter()
                trotter_step_body(sim, QUBITS_PER_PATCH, intra_edges,
                                  current_J, current_hx, current_hz, dt)
                t_lat_trotter = time.perf_counter() - t_start_trotter

                t_lat_tomo = 0.0
                if is_measure:
                    t_start_tomo = time.perf_counter()
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
                    t_lat_tomo = time.perf_counter() - t_start_tomo

                    payload = {
                        "state": state,
                        "meanfield_bulk_energy": bulk_e,
                        "lat_trotter_ms": t_lat_trotter * 1000.0,
                        "lat_tomo_ms": t_lat_tomo * 1000.0
                    }

                    if is_final:
                        t_smp = time.perf_counter()
                        payload["skqd_counts"] = sample_patch_counts(
                            sim, QUBITS_PER_PATCH, skqd_shots)
                        payload["lat_sample_ms"] = (time.perf_counter() - t_smp) * 1000.0

                    patch_data_to_master[p] = payload

            if is_measure:
                conn.send(patch_data_to_master)
                kick_payloads = conn.recv()

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
        self.energy_csv       = "meanfield_ground_state_energy_curve_multi.csv"
        self.profiles_csv     = "boundary_profiles_multi.csv"
        self.skqd_csv         = "skqd_refined_energies.csv"
        self.state_dump_file  = "macroscopic_lattice_states.npy"
        self.config_file      = "lattice_config.json"

        self._final_skqd_counts: Dict[int, Dict[int, int]] = {}
        self._final_kick_fields: Dict[int, Dict[int, Tuple[float, float, float]]] = {}
        self._final_boundary_energy: float = 0.0

        self._init_files()

        self.worker_assignments = [[] for _ in range(TOTAL_WORKERS)]
        for i in range(TOTAL_PATCHES):
            self.worker_assignments[i % TOTAL_WORKERS].append(i)

    def _init_files(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump({"grid_x": GRID_X, "grid_y": GRID_Y, "grid_z": GRID_Z,
                           "num_patches": TOTAL_PATCHES,
                           "qubits_per_patch": QUBITS_PER_PATCH}, f)
            with open(self.energy_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Anneal_Percent", "MeanField_Bulk_Energy",
                    "MeanField_Boundary_Energy", "MeanField_Total_Energy"
                ]).writeheader()
            with open(self.profiles_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Patch", "Face", "X_mean", "Y_mean", "Z_mean"
                ]).writeheader()
            with open(self.skqd_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Patch", "Seed_States", "Final_Subspace", "SKQD_E0",
                    "MeanField_Bulk_E", "Delta_E"
                ]).writeheader()
        except Exception as e:
            print(f"[CSV] Warning: Setup configuration write failed: {e}", file=sys.stderr)

    def _log_csvs(self, step, anneal, bulk, bound, total, patch_profiles):
        try:
            with open(self.energy_csv, mode='a', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Anneal_Percent", "MeanField_Bulk_Energy",
                    "MeanField_Boundary_Energy", "MeanField_Total_Energy"
                ]).writerow({"Step": step, "Anneal_Percent": anneal,
                             "MeanField_Bulk_Energy": bulk, "MeanField_Boundary_Energy": bound,
                             "MeanField_Total_Energy": total})

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

    def _run_skqd_refinement(self,
                             target_J: float,
                             target_hx: float,
                             target_hz: float,
                             final_bulk_energies: Dict[int, float],
                             top_seed: int,
                             max_subspace: int,
                             max_iters: int):
        if not self._final_skqd_counts:
            print("[SKQD] No samples captured; skipping refinement.", file=sys.stderr)
            return

        print(f"\n--- SKQD Subspace Refinement ({TOTAL_PATCHES} patches, "
              f"seed<= {top_seed}, budget {max_subspace}) ---")

        nq = QUBITS_PER_PATCH
        total_embedded = 0.0
        results = []

        for p in range(TOTAL_PATCHES):
            counts = self._final_skqd_counts.get(p, {})
            if not counts:
                print(f"[SKQD] Patch {p:02d}: no samples; skipped.", file=sys.stderr)
                continue

            seed = [bs for bs, _ in sorted(counts.items(),
                                           key=lambda kv: kv[1],
                                           reverse=True)[:top_seed]]
            if not seed:
                seed = [0, (1 << nq) - 1]

            kicks = self._final_kick_fields.get(p, {})
            hx_site = np.full(nq, target_hx, dtype=np.float64)
            hy_site = np.zeros(nq, dtype=np.float64)
            hz_site = np.full(nq, target_hz, dtype=np.float64)
            for q, (kx, ky, kz) in kicks.items():
                q = int(q)
                hx_site[q] += kx
                hy_site[q] += ky
                hz_site[q] += kz

            t0 = time.perf_counter()
            E0, sub_n = skqd_configuration_recovery(
                seed, self.intra_edges, target_J,
                hx_site, hy_site, hz_site, nq,
                max_iters=max_iters,
                max_subspace_size=max_subspace,
                label=f" P{p:02d}")
            elapsed = time.perf_counter() - t0

            mf_e = final_bulk_energies.get(p, float('nan'))
            delta = E0 - mf_e if np.isfinite(mf_e) else float('nan')
            total_embedded += E0
            results.append((p, len(seed), sub_n, E0, mf_e, delta))

            print(f"[SKQD] Patch {p:02d} | seed {len(seed):4d} -> support {sub_n:5d} "
                  f"| E0 {E0:+.6f} | MF {mf_e:+.6f} | dE {delta:+.6f} | {elapsed:.2f}s")

        try:
            with open(self.skqd_csv, mode='a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=[
                    "Patch", "Seed_States", "Final_Subspace", "SKQD_E0",
                    "MeanField_Bulk_E", "Delta_E"])
                for p, ns, sub_n, E0, mf_e, delta in results:
                    w.writerow({"Patch": p, "Seed_States": ns,
                                "Final_Subspace": sub_n, "SKQD_E0": E0,
                                "MeanField_Bulk_E": mf_e, "Delta_E": delta})
        except Exception as e:
            print(f"[SKQD] Warning: CSV write failed: {e}", file=sys.stderr)

        e_bnd = self._final_boundary_energy
        corrected = total_embedded - e_bnd
        mf_total = sum(v for v in final_bulk_energies.values()) + e_bnd

        print(f"\n[SKQD] Sum of embedded patch E0:      {total_embedded:+.6f}")
        print(f"[SKQD] Double-count correction (-E_b): {-e_bnd:+.6f}")
        print(f"[SKQD] SKQD-refined macroscopic E:     {corrected:+.6f}")
        print(f"[SKQD] Mean-field macroscopic E:       {mf_total:+.6f}")
        print(f"[SKQD] Refinement gain:                {corrected - mf_total:+.6f}")

    def run(self, total_steps: int, dt: float, initial_hx: float, target_g_face: float,
            target_J: float, target_hx: float, target_hz: float,
            measure_every: int = 1, effective_shots: float = 512.0,
            skqd_enable: bool = True, skqd_shots: int = 1024,
            skqd_top_seed: int = 256, skqd_max_subspace: int = 8192,
            skqd_max_iters: int = 15):

        if total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        if measure_every < 1:
            raise ValueError("measure_every must be a positive integer")

        total_qubits = TOTAL_PATCHES * QUBITS_PER_PATCH
        print(f"[Engine] {TOTAL_PATCHES} patches, {total_qubits} qubits, {GPUS_AVAILABLE} GPUs ({WORKERS_PER_GPU} workers/GPU), {total_steps} steps")

        active_ranks = [r for r in range(TOTAL_WORKERS) if self.worker_assignments[r]]

        workers = []
        pipes   = []

        effective_skqd_shots = int(skqd_shots) if skqd_enable else 0

        for rank in active_ranks:
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=gpu_worker_process,
                args=(rank, WORKERS_PER_GPU, self.worker_assignments[rank], child_conn,
                      dt, total_steps, initial_hx, target_J, target_hx, target_hz,
                      measure_every, effective_skqd_shots)
            )
            p.start()
            child_conn.close()
            workers.append(p)
            pipes.append(parent_conn)

        noise_rng = np.random.default_rng(self.master_seed)
        final_bulk_energies: Dict[int, float] = {}

        try:
            for t in range(total_steps):
                s = t / max(1, (total_steps - 1))
                current_g_face = s * target_g_face
                is_final   = (t == total_steps - 1)
                is_measure = (t % measure_every == 0) or is_final

                if not is_measure:
                    continue

                t0 = time.perf_counter()

                patch_full_states = {}
                bulk_energy = 0.0
                max_lat_trotter = 0.0
                max_lat_tomo = 0.0

                for conn in pipes:
                    try:
                        data = conn.recv()
                    except EOFError:
                        raise RuntimeError("Worker IPC connection lost.")
                    for p, payload in data.items():
                        patch_full_states[p] = payload["state"]
                        bulk_energy += payload["meanfield_bulk_energy"]
                        max_lat_trotter = max(max_lat_trotter, payload["lat_trotter_ms"])
                        max_lat_tomo = max(max_lat_tomo, payload["lat_tomo_ms"])
                        if is_final:
                            final_bulk_energies[p] = payload["meanfield_bulk_energy"]
                            if "skqd_counts" in payload:
                                self._final_skqd_counts[p] = payload["skqd_counts"]

                if len(patch_full_states) != TOTAL_PATCHES:
                    raise RuntimeError(
                        f"Fatal: IPC gather incomplete. "
                        f"Expected {TOTAL_PATCHES} patches, got {len(patch_full_states)}."
                    )

                step_state = np.zeros((TOTAL_PATCHES, QUBITS_PER_PATCH, 3))
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

                next_kick_payloads = {p: {} for p in range(TOTAL_PATCHES)}
                macroscopic_boundary_energy = 0.0

                scale = np.sqrt(dt * measure_every / effective_shots)
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
                            -current_g_face
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
                print(f"Step {t:03d} | E: {total_energy:+.4f} | Lat(Trot/Tomo): {max_lat_trotter:5.1f}/{max_lat_tomo:5.1f}ms | {time.perf_counter() - t0:.2f}s")
                self._log_csvs(t, s * 100, bulk_energy,
                               macroscopic_boundary_energy, total_energy, patch_profiles)

                if is_final:
                    self._final_kick_fields = {p: dict(k) for p, k in next_kick_payloads.items()}
                    self._final_boundary_energy = macroscopic_boundary_energy

                for i, w_rank in enumerate(active_ranks):
                    worker_payload = {p: next_kick_payloads[p]
                                      for p in self.worker_assignments[w_rank]}
                    pipes[i].send(worker_payload)

            if skqd_enable:
                try:
                    self._run_skqd_refinement(
                        target_J, target_hx, target_hz,
                        final_bulk_energies,
                        top_seed=skqd_top_seed,
                        max_subspace=skqd_max_subspace,
                        max_iters=skqd_max_iters)
                except Exception as e:
                    print(f"[SKQD] Refinement stage failed non-fatally: {e}",
                          file=sys.stderr)

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
            initial_hx=3.0,
            target_g_face=0.15,
            target_J=1.0,
            target_hx=0.5,
            target_hz=0.2,
            measure_every=1,
            effective_shots=512.0,
            skqd_enable=True,
            skqd_shots=1024,
            skqd_top_seed=256,
            skqd_max_subspace=8192,
            skqd_max_iters=15
        )
    except KeyboardInterrupt:
        pass
