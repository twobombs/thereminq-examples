# -*- coding: us-ascii -*-
# 27-Qubit 3x3x3 Lattice & Macroscopic Grid Annealing
# High-Throughput Volumetric Engine with Statistical Variance Injection
# Nucleus qubit removed: pure nearest-neighbour 3D Ising topology (54 bonds)
#
# REVISION 27 -- DRI_PRIME DEVICE ISOLATION (LIBDRM BAREMETAL FIX)
#
# This machine exposes TWO OpenCL devices under Mesa rusticl:
#   Device 0: AMD Radeon Pro VII / Instinct MI50 class (vega20, 16GB) -- compute
#   Device 1: NVIDIA CMP 50HX (10GB, nouveau)                         -- DO NOT USE
#
# ROOT CAUSE OF THE PERSISTENT 0x00008006fa682000 FAULT (Fix 40):
#   PyQrack's C++ extension probes PCI device IDs by opening /dev/dri file
#   descriptors directly via libdrm, completely bypassing the OpenCL ICD layer.
#   RUSTICL_ENABLE and CUDA_VISIBLE_DEVICES only filter the OpenCL platform --
#   they have no effect on libdrm's native DRM device enumeration. PyQrack was
#   opening the NVIDIA CMP 50HX DRI node and allocating its internal state
#   vector buffer in the nouveau VA space, which produces 57-bit pointers the
#   Vega20 IOMMU cannot translate.
#   FIX: `DRI_PRIME=pci-0000_44_00_0` is set at the top of every worker env
#   block. This is a per-process libdrm directive that redirects all DRI device
#   opens to the AMD card at PCI address 0000:44:00.0 before any fd is opened.
#   It is non-destructive: other processes on the host or in other containers
#   that do not have DRI_PRIME set continue using all cards normally.
#
# Fixes carried forward:
#   Fix 1-11: Trotter S2 symmetry, ZZ mcmtrx logic.
#   Fix 12: Single MI50 hardware config.
#   Fix 17: Zero-copy fast_in_ket/fast_out_ket via Qrack pinvoke.
#   Fix 20: Pin QRACK_FPPOW=5 to standardize library float32 contract.
#   Fix 26: Ephemeral `multiprocessing.Pool` (maxtasksperchild=1) teardown.
#   Fix 28-30: `alloc_dma_buf` via `MAP_POPULATE` and 64GB hint with 2GB strides.
#   Fix 31: `fast_out_ket` returns `np.array(copy=True)` to break mmap view binding.
#   Fix 32: `fast_out_ket` and `fast_in_ket` blocking queue drains.
#   Fix 35: 16-Patch memory clamp (17 GB static / 17 GB transient IPC).
#   Fix 36-37: Environment variable configuration locked before pyqrack import,
#              blinding the OpenCL ICD to the nouveau driver.
#   Fix 38: RUSTICL_ALLOW_SVM=0 forces explicit clCreateBuffer allocation.
#   Fix 39: Direct make_sim initialization, no TypeError fallback overhead.

import os
import gc
import csv
import time
import signal
import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Optional
import ctypes
import ctypes.util

# ==========================================
# 0. PYQRACK API SAFEGUARDS & GATES
# ==========================================
PX, PY, PZ = 1, 2, 3

def make_sim(QrackSimulator: Any, n: int) -> Any:
    # Fix 39: Direct initialization, PyQrack 2.5.1 does not accept kwargs
    return QrackSimulator(qubit_count=n)

def apply_h(sim: Any, q: int) -> None:
    if hasattr(sim, 'h'): sim.h(q)
    else: sim.mtrx([complex(1/np.sqrt(2),0), complex(1/np.sqrt(2),0),
                    complex(1/np.sqrt(2),0), complex(-1/np.sqrt(2),0)], q)

def apply_rx(sim: Any, theta: float, q: int) -> None:
    if hasattr(sim, 'r'): sim.r(PX, float(theta), q)
    else: sim.mtrx([complex(np.cos(theta/2),0), complex(0,-np.sin(theta/2)),
                    complex(0,-np.sin(theta/2)), complex(np.cos(theta/2),0)], q)

def apply_ry(sim: Any, theta: float, q: int) -> None:
    if hasattr(sim, 'r'): sim.r(PY, float(theta), q)
    else: sim.mtrx([complex(np.cos(theta/2),0), complex(-np.sin(theta/2),0),
                    complex(np.sin(theta/2),0), complex(np.cos(theta/2),0)], q)

def apply_rz(sim: Any, theta: float, q: int) -> None:
    if hasattr(sim, 'r'): sim.r(PZ, float(theta), q)
    else: sim.mtrx([complex(np.cos(theta/2),-np.sin(theta/2)), 0j,
                    0j, complex(np.cos(theta/2),np.sin(theta/2))], q)

def apply_cx(sim: Any, c: int, t: int) -> None:
    if hasattr(sim, 'cx'): sim.cx(c, t)
    elif hasattr(sim, 'mcx'):
        try: sim.mcx([c], t)
        except TypeError: sim.mcx([c], [t])
    else: raise RuntimeError("No CX gate available.")

def apply_zz(sim: Any, theta: float, q1: int, q2: int) -> None:
    if hasattr(sim, 'mcmtrx'):
        apply_rz(sim, theta, q1)
        apply_rz(sim, theta, q2)
        ph = complex(np.cos(2.0*theta), -np.sin(2.0*theta))
        try:    sim.mcmtrx([q1], [complex(1,0), 0j, 0j, ph], q2)
        except TypeError: sim.mcmtrx([q1], [complex(1,0), 0j, 0j, ph], [q2])
    else:
        apply_cx(sim, q1, q2)
        apply_rz(sim, theta, q2)
        apply_cx(sim, q1, q2)

def trotter_step_body(sim: Any, num_qubits: int,
                      intra_edges: List[Tuple[int, int]],
                      J: float, hx: float, hz: float, dt: float,
                      steps: int) -> None:
    for _ in range(steps):
        for q in range(num_qubits): apply_rx(sim, -hx*dt, q)
        for q in range(num_qubits): apply_rz(sim, -hz*dt, q)
        for q1, q2 in intra_edges:  apply_zz(sim, -2.0*J*dt, q1, q2)
        for q in range(num_qubits): apply_rz(sim, -hz*dt, q)
        for q in range(num_qubits): apply_rx(sim, -hx*dt, q)

def z_means(sim: Any, qubits: List[int]) -> np.ndarray:
    return np.array([1.0 - 2.0*sim.prob(q) for q in qubits])

def x_means(sim: Any, qubits: List[int]) -> np.ndarray:
    out = np.empty(len(qubits))
    for i, q in enumerate(qubits):
        apply_h(sim, q); out[i] = 1.0 - 2.0*sim.prob(q); apply_h(sim, q)
    return out

def y_means(sim: Any, qubits: List[int]) -> np.ndarray:
    out = np.empty(len(qubits))
    for i, q in enumerate(qubits):
        apply_rx(sim, -np.pi/2, q); out[i] = 1.0 - 2.0*sim.prob(q)
        apply_rx(sim, np.pi/2, q)
    return out

def zz_means_meanfield(z_exp: np.ndarray, edges: List[Tuple[int,int]]) -> np.ndarray:
    return np.array([z_exp[q1]*z_exp[q2] for q1,q2 in edges])


# ==========================================
# 0b. 48-BIT STRIDED ALLOCATOR & STATE TRANSFER
# ==========================================
_LIBC = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

_LIBC.mmap.restype = ctypes.c_void_p
_LIBC.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int, ctypes.c_size_t]
_LIBC.munmap.restype = ctypes.c_int
_LIBC.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

_QRACK_PINVOKE: Dict[str, Any] = {}

def _get_qrack_pinvoke() -> Optional[Any]:
    if "q" not in _QRACK_PINVOKE:
        try:
            from pyqrack.qrack_system import Qrack
            _QRACK_PINVOKE["q"] = Qrack
        except Exception:
            _QRACK_PINVOKE["q"] = None
    return _QRACK_PINVOKE["q"]

def _state_dtype_and_ptr(qrack: Any) -> Tuple[Any, Any]:
    if qrack.fppow < 6:
        return np.complex64, ctypes.POINTER(ctypes.c_float)
    return np.complex128, ctypes.POINTER(ctypes.c_double)

def alloc_dma_buf(num_qubits: int, worker_slot: int = 0) -> Tuple[np.ndarray, int, int]:
    qrack = _get_qrack_pinvoke()
    dtype = np.complex64 if (qrack is None or qrack.fppow < 6) else np.complex128
    nbytes = (1 << num_qubits) * np.dtype(dtype).itemsize

    hint_base = 0x1000000000
    hint_addr = ctypes.c_void_p(hint_base + worker_slot * (2 * 1024 * 1024 * 1024))

    ptr = _LIBC.mmap(
        hint_addr, nbytes,
        0x1 | 0x2,              # PROT_READ | PROT_WRITE
        0x02 | 0x20 | 0x08000,  # MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE
        -1, 0
    )

    if ptr == 0xffffffffffffffff or ptr == -1:
        raise MemoryError(f"Failed to mmap 48-bit DMA buffer. Errno: {ctypes.get_errno()}")

    c_type = ctypes.c_float if dtype == np.complex64 else ctypes.c_double
    c_array = (c_type * ((1 << num_qubits) * 2)).from_address(ptr)
    buf = np.ndarray((1 << num_qubits,), dtype=dtype, buffer=c_array)

    return buf, ptr, nbytes

def free_dma_buf(ptr: int, nbytes: int) -> None:
    try:
        _LIBC.munmap(ctypes.c_void_p(ptr), nbytes)
    except Exception:
        pass

def fast_in_ket(sim: Any, state: np.ndarray, dma_buf: np.ndarray) -> None:
    qrack = _get_qrack_pinvoke()
    if qrack is not None:
        _, cptr = _state_dtype_and_ptr(qrack)
        np.copyto(dma_buf, state)
        qrack.qrack_lib.InKet(sim.sid, dma_buf.ctypes.data_as(cptr))
        sim._throw_if_error()
        try:
            _ = sim.prob(0)
        except Exception:
            pass
    else:
        sim.in_ket(state.tolist())

def fast_out_ket(sim: Any, num_qubits: int, dma_buf: np.ndarray) -> np.ndarray:
    qrack = _get_qrack_pinvoke()
    if qrack is not None:
        _, cptr = _state_dtype_and_ptr(qrack)
        qrack.qrack_lib.OutKet(sim.sid, dma_buf.ctypes.data_as(cptr))
        sim._throw_if_error()
        try:
            _ = sim.prob(0)
        except Exception:
            pass
        return np.array(dma_buf, dtype=np.complex64, copy=True)
    return np.array(sim.out_ket(), dtype=np.complex64)


# ==========================================
# 1. TOPOLOGY
# ==========================================
def generate_27q_lattice_subvolume() -> Tuple[List[Tuple[int,int]], Dict[str,List[int]]]:
    lx, ly, lz = 3, 3, 3
    edges = []
    boundaries = {"+X":[], "-X":[], "+Y":[], "-Y":[], "+Z":[], "-Z":[]}
    for x in range(lx):
        for y in range(ly):
            for z in range(lz):
                idx = x*(ly*lz) + y*lz + z
                if x < lx-1: edges.append((idx, (x+1)*(ly*lz)+y*lz+z))
                if y < ly-1: edges.append((idx, x*(ly*lz)+(y+1)*lz+z))
                if z < lz-1: edges.append((idx, x*(ly*lz)+y*lz+(z+1)))
                if x == 0:    boundaries["-X"].append(idx)
                if x == lx-1: boundaries["+X"].append(idx)
                if y == 0:    boundaries["-Y"].append(idx)
                if y == ly-1: boundaries["+Y"].append(idx)
                if z == 0:    boundaries["-Z"].append(idx)
                if z == lz-1: boundaries["+Z"].append(idx)
    return edges, boundaries


# ==========================================
# 2. WORKER ENVIRONMENT ISOLATION BLOCK
# ==========================================
def _isolate_worker_env(args: Dict[str, Any]) -> None:
    """
    Fix 40: DRI_PRIME forces libdrm to open only the AMD DRI node.
    Fix 38: RUSTICL_ALLOW_SVM=0 forces explicit clCreateBuffer.
    Fix 37: RUSTICL_ENABLE=radeonsi blinds OpenCL ICD to nouveau.
    Fix 36: All env vars set before any pyqrack import.
    """
    os.environ["DRI_PRIME"]              = "pci-0000_44_00_0"
    os.environ["CUDA_VISIBLE_DEVICES"]   = ""
    os.environ["OCL_ICD_PLATFORM_SORT"]  = "none"
    os.environ["RUSTICL_ENABLE"]         = "radeonsi"
    os.environ["RUSTICL_ALLOW_SVM"]      = "0"
    os.environ["MESA_VK_DEVICE_SELECT"]  = "amd"
    os.environ["OMP_NUM_THREADS"]        = "1"
    os.environ["OPENBLAS_NUM_THREADS"]   = "1"
    os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"
    os.environ["QRACK_MAX_ALLOC_MB"]         = str(int(args["gpu_max_alloc_mb"]))
    os.environ["QRACK_OCL_DEFAULT_DEVICE"]   = str(args["gpu_device_id"])
    os.environ["QRACK_QPAGER_DEVICES"]       = "-1"
    os.environ["QRACK_QUNITMULTI_DEVICES"]   = "-1"
    os.environ["QRACK_FPPOW"]                = "5"


# ==========================================
# 3. EPHEMERAL WORKER TASKS
# ==========================================
def worker_init_state(args: Dict[str, Any]) -> Dict[str, Any]:
    _isolate_worker_env(args)

    num_qubits  = args["num_qubits"]
    seed        = args["seed"]
    worker_slot = args["worker_slot"]

    from pyqrack import QrackSimulator
    dma_buf, ptr, nbytes = alloc_dma_buf(num_qubits, worker_slot)
    sim = make_sim(QrackSimulator, num_qubits)

    rng = np.random.default_rng(seed)
    for q in range(num_qubits):
        apply_h(sim, q)
        apply_rx(sim, rng.normal(0, 1e-5), q)
        apply_rz(sim, rng.normal(0, 1e-5), q)
    state = fast_out_ket(sim, num_qubits, dma_buf)

    del sim
    gc.collect()
    time.sleep(0.01)
    free_dma_buf(ptr, nbytes)

    return {"patch_idx": args["patch_idx"], "state": state}


def worker_evolve_step(args: Dict[str, Any]) -> Dict[str, Any]:
    _isolate_worker_env(args)

    num_qubits  = args["num_qubits"]
    state       = args["state"]
    intra_edges = args["intra_edges"]
    J           = args["J"]
    hx          = args["hx"]
    hz          = args["hz"]
    dt          = args["dt"]
    steps       = args["steps"]
    worker_slot = args["worker_slot"]

    from pyqrack import QrackSimulator
    dma_buf, ptr, nbytes = alloc_dma_buf(num_qubits, worker_slot)
    sim = make_sim(QrackSimulator, num_qubits)

    fast_in_ket(sim, state, dma_buf)
    trotter_step_body(sim, num_qubits, intra_edges, J, hx, hz, dt, steps)

    stat_payload = None
    if args.get("measure_stats", False):
        bounds = args["boundaries"]
        all_boundary_qubits = sorted(list(set(
            q for face in bounds.values() for q in face)))
        Z_mean = z_means(sim, all_boundary_qubits)
        X_mean = x_means(sim, all_boundary_qubits)
        Y_mean = y_means(sim, all_boundary_qubits)
        Z_var = np.clip(1.0 - Z_mean**2, 0.0, 1.0)
        X_var = np.clip(1.0 - X_mean**2, 0.0, 1.0)
        Y_var = np.clip(1.0 - Y_mean**2, 0.0, 1.0)
        stat_payload = {
            "qubits": all_boundary_qubits,
            "means": {"X": X_mean, "Y": Y_mean, "Z": Z_mean},
            "vars":  {"X": X_var,  "Y": Y_var,  "Z": Z_var}
        }

    new_state = fast_out_ket(sim, num_qubits, dma_buf)

    del sim
    gc.collect()
    time.sleep(0.01)
    free_dma_buf(ptr, nbytes)

    return {"patch_idx": args["patch_idx"], "state": new_state, "stats": stat_payload}


def worker_apply_kicks(args: Dict[str, Any]) -> Dict[str, Any]:
    _isolate_worker_env(args)

    num_qubits  = args["num_qubits"]
    state       = args["state"]
    kicks       = args["kicks"]
    J_val       = args["J"]
    hx_val      = args["hx"]
    hz_val      = args["hz"]
    intra_edges = args["intra_edges"]
    worker_slot = args["worker_slot"]

    from pyqrack import QrackSimulator
    dma_buf, ptr, nbytes = alloc_dma_buf(num_qubits, worker_slot)
    sim = make_sim(QrackSimulator, num_qubits)

    fast_in_ket(sim, state, dma_buf)

    for raw_q, (kx, ky, kz) in kicks.items():
        q = int(raw_q)
        K = np.sqrt(kx**2 + ky**2 + kz**2)
        if K > 0.0:
            c, s = np.cos(K/2.0), np.sin(K/2.0)
            nx, ny, nz = kx/K, ky/K, kz/K
            sim.mtrx([complex(c, -nz*s), complex(-ny*s, -nx*s),
                      complex( ny*s, -nx*s), complex(c,  nz*s)], q)

    all_q  = list(range(num_qubits))
    z_exp  = z_means(sim, all_q)
    x_exp  = x_means(sim, all_q)

    new_state = fast_out_ket(sim, num_qubits, dma_buf)

    del sim
    gc.collect()
    time.sleep(0.01)
    free_dma_buf(ptr, nbytes)

    zz_exp = zz_means_meanfield(z_exp, intra_edges)
    local_energy = (
        -hz_val * float(np.sum(z_exp))
        - J_val  * float(np.sum(zz_exp))
        - hx_val * float(np.sum(x_exp))
    )

    return {"patch_idx": args["patch_idx"], "state": new_state, "energy": local_energy}


def worker_compute_benchmarks(args: Dict[str, Any]) -> Dict[str, Any]:
    _isolate_worker_env(args)

    num_qubits  = args["num_qubits"]
    state       = args["state"]
    worker_slot = args["worker_slot"]

    from pyqrack import QrackSimulator
    dma_buf, ptr, nbytes = alloc_dma_buf(num_qubits, worker_slot)
    sim = make_sim(QrackSimulator, num_qubits)

    all_q = list(range(num_qubits))
    fast_in_ket(sim, state, dma_buf)
    z_e = z_means(sim, all_q)
    x_e = x_means(sim, all_q)
    y_e = y_means(sim, all_q)

    del sim
    gc.collect()
    time.sleep(0.01)
    free_dma_buf(ptr, nbytes)

    avg_purity = float(np.mean(x_e**2 + y_e**2 + z_e**2))
    return {"patch_idx": args["patch_idx"], "purity": avg_purity}


# ==========================================
# 4. ORCHESTRATOR
# ==========================================
class VolumetricHadronEngine27Q:
    def __init__(
        self,
        gpu_allocation: List[int],
        semaphore_limits: Dict[int, int],
        gpu_alloc_caps_mb: Dict[int, int],
        grid: Tuple[int,int,int] = (2,2,2),
        master_seed: int = 42
    ):
        self.gpu_allocation = gpu_allocation
        self.grid_x, self.grid_y, self.grid_z = grid
        self.num_patches = self.grid_x * self.grid_y * self.grid_z

        if len(self.gpu_allocation) != self.num_patches:
            raise ValueError(
                f"GPU allocation list ({len(self.gpu_allocation)}) must match "
                f"total grid patches ({self.num_patches}).")

        self.qubits_per_patch = 27
        self.intra_edges, self.boundaries = generate_27q_lattice_subvolume()

        LIVE_MB = 2200
        residency: Dict[int,int] = {}
        for g in self.gpu_allocation:
            residency[g] = residency.get(g, 0) + 1
        for g, count in residency.items():
            cap = gpu_alloc_caps_mb.get(g)
            if cap is None:
                raise ValueError(f"No VRAM cap for GPU {g}.")
            concurrent = semaphore_limits.get(g, 1)
            need = concurrent * LIVE_MB
            if need > cap:
                raise ValueError(
                    f"GPU {g}: {concurrent} concurrent sims x {LIVE_MB}MB = "
                    f"{need}MB exceeds {cap}MB cap.")

        self.gpu_alloc_caps_mb = gpu_alloc_caps_mb

        self.patch_coords: Dict[int,Tuple[int,int,int]] = {}
        idx = 0
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                for z in range(self.grid_z):
                    self.patch_coords[idx] = (x, y, z)
                    idx += 1

        self.coord_to_patch = {v: k for k, v in self.patch_coords.items()}

        self.pool_size = sum(semaphore_limits.values())
        self.ctx  = mp.get_context('spawn')
        self.pool = self.ctx.Pool(processes=self.pool_size, maxtasksperchild=1)

        self.patch_states: Dict[int, np.ndarray] = {}
        self.energy_history: List[Dict[str,Any]] = []

        self.csv_filename = "ground_state_energy_curve_27q_16p.csv"
        self._init_csv()

        total_sites = self.num_patches * self.qubits_per_patch
        host_gb     = self.num_patches * 1.074

        print(f"Initializing 27Q Ephemeral-Pool Engine...")
        print(f"Grid: {grid} = {self.num_patches} patches x {self.qubits_per_patch}q "
              f"= {total_sites} total logical qubits")
        print(f"Host RAM: {self.num_patches} arrays x 1.07GB = {host_gb:.1f}GB")

        master_rng  = np.random.default_rng(master_seed)
        patch_seeds = master_rng.integers(0, 2**31-1, size=self.num_patches)

        print("Dispatching initial states to GPU Pool...")
        init_args = [
            {
                "patch_idx":       p,
                "gpu_device_id":   self.gpu_allocation[p],
                "num_qubits":      self.qubits_per_patch,
                "seed":            int(patch_seeds[p]),
                "gpu_max_alloc_mb": self.gpu_alloc_caps_mb[self.gpu_allocation[p]],
                "worker_slot":     p
            } for p in range(self.num_patches)
        ]

        results = self.pool.map(worker_init_state, init_args)
        for r in results:
            self.patch_states[r["patch_idx"]] = r["state"]
        print("Initialization complete.")

    def _init_csv(self) -> None:
        try:
            with open(self.csv_filename, mode='w', newline='') as f:
                csv.DictWriter(
                    f, fieldnames=["Step","Anneal_Percent","Energy"]
                ).writeheader()
        except Exception:
            pass

    def _append_to_csv(self, data: Dict[str,Any]) -> None:
        try:
            with open(self.csv_filename, mode='a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=["Step","Anneal_Percent","Energy"])
                w.writerow(data)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass

    def anneal_to_ground_state(
        self,
        total_steps: int,
        dt: float,
        target_g_face: float,
        target_J: float,
        target_hx: float,
        target_hz: float,
        measure_every: int = 1
    ) -> None:
        print(f"\nStarting Adiabatic Anneal with Stochastic Injection...")
        if measure_every > 1:
            print(f"Measurement cadence: every {measure_every} steps")
        self.energy_history.clear()
        noise_rng       = np.random.default_rng()
        effective_shots = 512.0

        for t in range(total_steps):
            s              = t / max(1, (total_steps - 1))
            current_hx     = (1.0-s)*3.0 + s*target_hx
            current_J      = s*target_J
            current_hz     = s*target_hz
            current_g_face = s*target_g_face
            is_measure     = (t % measure_every == 0) or (t == total_steps-1)

            step_args = [
                {
                    "patch_idx":       p,
                    "gpu_device_id":   self.gpu_allocation[p],
                    "num_qubits":      self.qubits_per_patch,
                    "state":           self.patch_states[p],
                    "intra_edges":     self.intra_edges,
                    "boundaries":      self.boundaries,
                    "J":               current_J,
                    "hx":              current_hx,
                    "hz":              current_hz,
                    "dt":              dt,
                    "steps":           1,
                    "gpu_max_alloc_mb": self.gpu_alloc_caps_mb[self.gpu_allocation[p]],
                    "measure_stats":   is_measure,
                    "worker_slot":     p
                } for p in range(self.num_patches)
            ]

            results = self.pool.map(worker_evolve_step, step_args)
            for r in results:
                self.patch_states[r["patch_idx"]] = r["state"]

            if not is_measure:
                print(f"Step {t:03d} | Anneal: {s*100:05.1f}% | (evolve-only)")
                continue

            patch_profiles = {r["patch_idx"]: r["stats"] for r in results}
            kick_payloads  = {p: {"kicks": {}} for p in range(self.num_patches)}

            macroscopic_boundary_energy = 0.0
            scale = np.sqrt(dt / effective_shots)

            stochastic_noise: Dict[int,Dict[int,Tuple[float,float,float]]] = {}
            for p, prof in patch_profiles.items():
                n_bounds = len(prof["qubits"])
                X_noise = noise_rng.normal(0.0,1.0,n_bounds)*np.sqrt(prof["vars"]["X"])*scale
                Y_noise = noise_rng.normal(0.0,1.0,n_bounds)*np.sqrt(prof["vars"]["Y"])*scale
                Z_noise = noise_rng.normal(0.0,1.0,n_bounds)*np.sqrt(prof["vars"]["Z"])*scale
                for arr in (X_noise, Y_noise, Z_noise):
                    if not np.all(np.isfinite(arr)): arr[:] = 0.0
                stochastic_noise[p] = {
                    q: (X_noise[i], Y_noise[i], Z_noise[i])
                    for i, q in enumerate(prof["qubits"])
                }

            for p1, coord1 in self.patch_coords.items():
                x1, y1, z1 = coord1
                neighbors = {
                    "+X": (x1+1,y1,z1), "-X": (x1-1,y1,z1),
                    "+Y": (x1,y1+1,z1), "-Y": (x1,y1-1,z1),
                    "+Z": (x1,y1,z1+1), "-Z": (x1,y1,z1-1),
                }
                for dir1, coord2 in neighbors.items():
                    if not (0<=coord2[0]<self.grid_x and
                            0<=coord2[1]<self.grid_y and
                            0<=coord2[2]<self.grid_z): continue
                    p2 = self.coord_to_patch.get(coord2)
                    if p2 is None or p1 >= p2: continue

                    dir2    = dir1.replace("+","temp").replace("-","+").replace("temp","-")
                    face1_q = self.boundaries[dir1]
                    face2_q = self.boundaries[dir2]

                    prof1, noise1 = patch_profiles[p1], stochastic_noise[p1]
                    prof2, noise2 = patch_profiles[p2], stochastic_noise[p2]

                    q_to_i1 = {q: i for i, q in enumerate(prof1["qubits"])}
                    q_to_i2 = {q: i for i, q in enumerate(prof2["qubits"])}

                    ax2=ay2=az2=0.0
                    for q2 in face2_q:
                        i2 = q_to_i2[q2]
                        ax2 += prof2["means"]["X"][i2] + noise2[q2][0]
                        ay2 += prof2["means"]["Y"][i2] + noise2[q2][1]
                        az2 += prof2["means"]["Z"][i2] + noise2[q2][2]
                    n2 = max(1,len(face2_q)); ax2/=n2; ay2/=n2; az2/=n2

                    ax1=ay1=az1=0.0
                    for q1 in face1_q:
                        i1 = q_to_i1[q1]
                        ax1 += prof1["means"]["X"][i1] + noise1[q1][0]
                        ay1 += prof1["means"]["Y"][i1] + noise1[q1][1]
                        az1 += prof1["means"]["Z"][i1] + noise1[q1][2]
                    n1 = max(1,len(face1_q)); ax1/=n1; ay1/=n1; az1/=n1

                    interaction_E = (
                        -current_g_face
                        * (ax1*ax2 + ay1*ay2 + az1*az2)
                        * ((len(face1_q)+len(face2_q)) / 2.0)
                    )
                    macroscopic_boundary_energy += interaction_E

                    for q1 in face1_q:
                        k = kick_payloads[p1]["kicks"].get(q1, (0.,0.,0.))
                        kick_payloads[p1]["kicks"][q1] = (
                            k[0]+current_g_face*ax2,
                            k[1]+current_g_face*ay2,
                            k[2]+current_g_face*az2)
                    for q2 in face2_q:
                        k = kick_payloads[p2]["kicks"].get(q2, (0.,0.,0.))
                        kick_payloads[p2]["kicks"][q2] = (
                            k[0]+current_g_face*ax1,
                            k[1]+current_g_face*ay1,
                            k[2]+current_g_face*az1)

            kick_args = [
                {
                    "patch_idx":       p,
                    "gpu_device_id":   self.gpu_allocation[p],
                    "num_qubits":      self.qubits_per_patch,
                    "state":           self.patch_states[p],
                    "kicks":           kick_payloads[p]["kicks"],
                    "J":               current_J,
                    "hx":              current_hx,
                    "hz":              current_hz,
                    "intra_edges":     self.intra_edges,
                    "gpu_max_alloc_mb": self.gpu_alloc_caps_mb[self.gpu_allocation[p]],
                    "worker_slot":     p
                } for p in range(self.num_patches)
            ]

            kick_results = self.pool.map(worker_apply_kicks, kick_args)
            bulk_energy  = 0.0
            for r in kick_results:
                self.patch_states[r["patch_idx"]] = r["state"]
                bulk_energy += r["energy"]

            total_energy = bulk_energy + macroscopic_boundary_energy

            print(f"Step {t:03d} | Anneal: {s*100:05.1f}% | "
                  f"Total Setup Potential Energy: {total_energy:+.4f}")

            step_data = {"Step": t, "Anneal_Percent": s*100, "Energy": total_energy}
            self.energy_history.append(step_data)
            self._append_to_csv(step_data)

            if t == total_steps-1:
                print(f"         +-- Calculating Final Benchmarks...")
                bench_args = [
                    {
                        "patch_idx":       p,
                        "gpu_device_id":   self.gpu_allocation[p],
                        "num_qubits":      self.qubits_per_patch,
                        "state":           self.patch_states[p],
                        "gpu_max_alloc_mb": self.gpu_alloc_caps_mb[self.gpu_allocation[p]],
                        "worker_slot":     p
                    } for p in range(self.num_patches)
                ]
                bench_res  = self.pool.map(worker_compute_benchmarks, bench_args)
                avg_purity = sum(r["purity"] for r in bench_res) / self.num_patches
                print(f"         +-- Avg Sub-Volume Purity: {avg_purity:.4f}")

    def shutdown(self) -> None:
        print("\nShutting down Ephemeral-Pool Engine...")
        try:
            self.pool.close()
            self.pool.join()
        except Exception:
            pass


# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    mp.freeze_support()

    gpu_0 = int(os.environ.get("WORMHOLE_GPU", "0"))

    # SAFE BASELINE MODE:
    # 4x4x1 = 16 patches (caps Host RAM to ~17 GB static + ~17 GB IPC)
    target_grid              = (4, 4, 1)
    explicit_gpu_allocation  = [gpu_0] * 16

    # Throttled Concurrency: 2 simultaneous workers.
    # Prevents PCIe Gen 2 bus saturation and drops active VRAM to ~2.2 GB.
    semaphore_caps = {gpu_0: 2}

    gpu_alloc_caps_mb = {gpu_0: 14000}

    engine = VolumetricHadronEngine27Q(
        gpu_allocation    = explicit_gpu_allocation,
        semaphore_limits  = semaphore_caps,
        gpu_alloc_caps_mb = gpu_alloc_caps_mb,
        grid              = target_grid,
        master_seed       = 1337
    )

    try:
        engine.anneal_to_ground_state(
            total_steps   = 100,
            dt            = 0.04,
            target_g_face = 0.15,
            target_J      = 1.0,
            target_hx     = 0.5,
            target_hz     = 0.2,
            measure_every = 1
        )
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    finally:
        engine.shutdown()
