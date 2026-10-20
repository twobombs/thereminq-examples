import os
import gc
import time
import queue
import threading
import numpy as np
import multiprocessing as mp

# ==========================================
# 0. PYQRACK API SAFEGUARDS
# ==========================================
try:
    from pyqrack import Pauli
    PX = getattr(Pauli, 'PauliX', getattr(Pauli, 'X', 1))
    PY = getattr(Pauli, 'PauliY', getattr(Pauli, 'Y', 2))
    PZ = getattr(Pauli, 'PauliZ', getattr(Pauli, 'Z', 3))
except ImportError:
    PX, PY, PZ = 1, 2, 3

def apply_h(sim, q):
    if hasattr(sim, 'h'):
        sim.h(q)
    else:
        c = complex(1/np.sqrt(2), 0)
        # PyQrack mtrx expects a list of targets for its C++ MultiTarget mapping
        sim.mtrx([c, c, c, -c], [q])

def apply_rx(sim, theta, q):
    if hasattr(sim, 'r'):
        sim.r(PX, float(theta), q)
    else:
        half_t = float(theta) / 2.0
        c = complex(np.cos(half_t), 0)
        s = complex(0, -np.sin(half_t))
        sim.mtrx([c, s, s, c], [q])

def apply_ry(sim, theta, q):
    if hasattr(sim, 'r'):
        sim.r(PY, float(theta), q)
    else:
        half_t = float(theta) / 2.0
        c = complex(np.cos(half_t), 0)
        s1 = complex(-np.sin(half_t), 0)
        s2 = complex(np.sin(half_t), 0)
        sim.mtrx([c, s1, s2, c], [q])

def apply_rz(sim, theta, q):
    if hasattr(sim, 'r'):
        sim.r(PZ, float(theta), q)
    else:
        half_t = float(theta) / 2.0
        c_m = complex(np.cos(-half_t), np.sin(-half_t))
        c_p = complex(np.cos(half_t), np.sin(half_t))
        sim.mtrx([c_m, 0j, 0j, c_p], [q])

def apply_cx(sim, c, t):
    if hasattr(sim, 'cx'):
        sim.cx(c, t)
    else:
        sim.mcx([c], t)

# ==========================================
# 1. TOPOLOGY DEFINITIONS (72-Qubit 6x12 Grid)
# ==========================================
def get_3x6_edges():
    """Intra-patch entanglement for a 3x6 physical sub-grid (18 qubits)"""
    edges = []
    for r in range(3):
        for c in range(6):
            idx = r * 6 + c
            if c < 5: edges.append((idx, idx + 1))
            if r < 2: edges.append((idx, idx + 6))
    return edges

def get_topology():
    """
    Creates a 6x12 global grid (72 qubits),
    divided into a 2x2 grid of 3x6 patches.
    """
    patches = [[] for _ in range(4)]
    fence_edges = []

    for r in range(6):
        for c in range(12):
            idx = r * 12 + c
            patch_r = r // 3
            patch_c = c // 6
            patch_idx = patch_r * 2 + patch_c
            local_idx = len(patches[patch_idx])
            patches[patch_idx].append((idx, local_idx))

    global_to_local = {}
    for p_idx, patch in enumerate(patches):
        for global_idx, local_idx in patch:
            global_to_local[global_idx] = (p_idx, local_idx)

    for r in range(6):
        for c in range(12):
            global_1 = r * 12 + c
            if c == 5:
                global_2 = r * 12 + (c + 1)
                fence_edges.append((global_to_local[global_1], global_to_local[global_2]))
            if r == 2:
                global_2 = (r + 1) * 12 + c
                fence_edges.append((global_to_local[global_1], global_to_local[global_2]))

    return patches, fence_edges

# ==========================================
# 2. ISOLATED GPU WORKER TASKS & INITIALIZERS
# ==========================================
def init_worker(device_queue, identical_devices=False):
    device_id = device_queue.get()
    try:
        os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(device_id)
        os.environ["QRACK_QPAGER_DEVICES"] = str(device_id)
        os.environ["QRACK_QUNITMULTI_DEVICES"] = str(device_id)

        from pyqrack import QrackSimulator

        _warmup = None
        try:
            _warmup = QrackSimulator(qubit_count=1)
        finally:
            if _warmup is not None:
                del _warmup

        if not identical_devices:
            device_queue.put(device_id)
    except Exception:
        if not identical_devices:
            device_queue.put(device_id)
        raise

def isolated_holographic_worker(patch_params, boundary_params, patch_idx, intra_edges, fence_qubits):
    """
    Evaluates the local patch energy contribution for the Heisenberg XX model: 
    H = -\\sum (XX + YY)
    """
    from pyqrack import QrackSimulator

    num_ancillas = len(fence_qubits)
    num_physical = len(patch_params) // 2
    total_qubits = num_physical + num_ancillas

    if len(boundary_params) != num_ancillas * 4:
        raise ValueError(f"Boundary parameter size mismatch: expected {num_ancillas * 4}, got {len(boundary_params)}")

    sim = QrackSimulator(qubit_count=total_qubits)

    # 1. Apply Main Physical Ansatz FIRST
    param_idx = 0
    for q in range(num_physical):
        apply_rx(sim, patch_params[param_idx], q)
        apply_ry(sim, patch_params[param_idx + 1], q)
        param_idx += 2

    for q1, q2 in intra_edges:
        apply_cx(sim, q1, q2)

    # 2. Apply Holographic Bath Layer
    b_idx = 0
    for q in fence_qubits:
        apply_ry(sim, boundary_params[b_idx], q)
        apply_rz(sim, boundary_params[b_idx + 1], q)
        b_idx += 2

    ancilla_indices = list(range(num_physical, total_qubits))
    for q in ancilla_indices:
        apply_ry(sim, boundary_params[b_idx], q)
        apply_rz(sim, boundary_params[b_idx + 1], q)
        b_idx += 2

    for i, f_q in enumerate(fence_qubits):
        apply_cx(sim, ancilla_indices[i], f_q)

    # 3. Evaluate Intra-Patch Energy (Using Uncomputing instead of Clone)
    intra_energy = 0.0
    for q1, q2 in intra_edges:
        for basis in ['X', 'Y']:
            if basis == 'X':
                apply_h(sim, q1); apply_h(sim, q2)
                apply_cx(sim, q1, q2)
                p_odd = sim.prob(q2)
                # Uncompute
                apply_cx(sim, q1, q2)
                apply_h(sim, q2); apply_h(sim, q1)
            else:
                apply_rx(sim, np.pi / 2, q1); apply_rx(sim, np.pi / 2, q2)
                apply_cx(sim, q1, q2)
                p_odd = sim.prob(q2)
                # Uncompute
                apply_cx(sim, q1, q2)
                apply_rx(sim, -np.pi / 2, q2); apply_rx(sim, -np.pi / 2, q1)
                
            intra_energy -= ((1.0 - p_odd) - p_odd)

    # 4. Evaluate Boundary Marginals (Using Uncomputing instead of Clone)
    boundary_expectations = {}
    for i, q in enumerate(fence_qubits):
        a_q = ancilla_indices[i]
        boundary_expectations[q] = {}

        for b_phys in ['X', 'Y']:
            boundary_expectations[q][b_phys] = {}
            if b_phys == 'X':
                apply_h(sim, q)
            elif b_phys == 'Y':
                apply_rx(sim, np.pi / 2, q)

            for b_anc in ['I', 'X', 'Y', 'Z']:
                if b_anc == 'X':
                    apply_h(sim, a_q)
                elif b_anc == 'Y':
                    apply_rx(sim, np.pi / 2, a_q)

                if b_anc == 'I':
                    p_odd = sim.prob(q)
                else:
                    apply_cx(sim, q, a_q)
                    p_odd = sim.prob(a_q)
                    # Uncompute inner entanglement
                    apply_cx(sim, q, a_q)

                boundary_expectations[q][b_phys][b_anc] = (1.0 - p_odd) - p_odd

                # Uncompute ancilla basis
                if b_anc == 'X':
                    apply_h(sim, a_q)
                elif b_anc == 'Y':
                    apply_rx(sim, -np.pi / 2, a_q)

            # Uncompute physical basis
            if b_phys == 'X':
                apply_h(sim, q)
            elif b_phys == 'Y':
                apply_rx(sim, -np.pi / 2, q)

    del sim
    return patch_idx, intra_energy, boundary_expectations

# ==========================================
# 3. CPU ORACLE POOL WORKERS
# ==========================================
def init_oracle_pool_worker():
    """Forces extreme process-level parallelization over internal thread parallelization."""
    os.environ["QRACK_QBDD_SEPARABILITY_THRESHOLD"] = "-1"
    os.environ["QRACK_MAX_PAGING_QB"] = "0"
    os.environ["QRACK_OCL_DEFAULT_DEVICE"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "1"  

    from pyqrack import QrackSimulator
    _warmup = None
    try:
        _warmup = QrackSimulator(
            qubit_count=1,
            is_gpu=False,
            is_binary_decision_tree=True
        )
    finally:
        if _warmup is not None:
            del _warmup

def oracle_evaluation_chunk(params, edges_chunk, num_qubits, patches, intra_patch_edges, fence_edges, p_l_to_global):
    """
    Rebuilds the state natively on the worker to avoid IPC SID Clone overhead.
    Evaluates the Heisenberg XX model Hamiltonian: H = -\\sum (XX + YY)
    """
    from pyqrack import QrackSimulator
    
    sim = QrackSimulator(
        qubit_count=num_qubits,
        is_gpu=False,
        is_binary_decision_tree=True
    )
    
    # 1. Deterministically Build State
    param_idx = 0
    for p_idx, patch in enumerate(patches):
        for g, l in patch:
            apply_rx(sim, params[param_idx], g)
            apply_ry(sim, params[param_idx + 1], g)
            param_idx += 2
        for q1_local, q2_local in intra_patch_edges:
            g1 = p_l_to_global[(p_idx, q1_local)]
            g2 = p_l_to_global[(p_idx, q2_local)]
            apply_cx(sim, g1, g2)

    for (pA, qA), (pB, qB) in fence_edges:
        apply_cx(sim, p_l_to_global[(pA, qA)], p_l_to_global[(pB, qB)])

    # 2. Evaluate Specific Chunk
    energy = 0.0
    for q1, q2 in edges_chunk:
        for basis in ['X', 'Y']:
            if basis == 'X':
                apply_h(sim, q1); apply_h(sim, q2)
                apply_cx(sim, q1, q2)
                p_odd = sim.prob(q2)
                # Uncompute
                apply_cx(sim, q1, q2)
                apply_h(sim, q2); apply_h(sim, q1)
            else:
                apply_rx(sim, np.pi / 2, q1); apply_rx(sim, np.pi / 2, q2)
                apply_cx(sim, q1, q2)
                p_odd = sim.prob(q2)
                # Uncompute
                apply_cx(sim, q1, q2)
                apply_rx(sim, -np.pi / 2, q2); apply_rx(sim, -np.pi / 2, q1)

            energy -= ((1.0 - p_odd) - p_odd)
            
    del sim
    gc.collect()
    return energy

# ==========================================
# 4. HOLOGRAPHIC ORCHESTRATOR
# ==========================================
class HolographicDistributedEngine:
    def __init__(self, device_ids):
        self.device_ids = device_ids
        self.patches, self.fence_edges = get_topology()
        self.intra_patch_edges = get_3x6_edges()
        
        self.identical_devices = len(set(device_ids)) == 1

        temp_reqs = {i: set() for i in range(4)}
        for (pA, qA), (pB, qB) in self.fence_edges:
            temp_reqs[pA].add(qA)
            temp_reqs[pB].add(qB)

        # Sorted list ensures deterministic ordering.
        # Set ensures no duplicate fence qubits per patch, preventing key collisions.
        self.patch_fence_reqs = {i: sorted(list(temp_reqs[i])) for i in range(4)}

        self.total_b_params = 0
        for p_idx in range(4):
            # 4 parameters per fence qubit: 
            # 2 for the physical boundary qubit (Ry, Rz) + 2 for the matching ancilla (Ry, Rz)
            self.total_b_params += len(self.patch_fence_reqs[p_idx]) * 4

        self.ctx = mp.get_context('spawn')
        self.device_queue = self.ctx.Queue()
        for d in self.device_ids:
            self.device_queue.put(d)

        self.pool = self.ctx.Pool(
            processes=len(self.device_ids),
            initializer=init_worker,
            initargs=(self.device_queue, self.identical_devices)
        )

    def _rebuild_pool(self):
        try:
            self.pool.terminate()
            self.pool.join()
        except Exception: 
            pass
        
        self.device_queue = self.ctx.Queue()
        for d in self.device_ids:
            self.device_queue.put(d)
        
        self.pool = self.ctx.Pool(
            processes=len(self.device_ids),
            initializer=init_worker,
            initargs=(self.device_queue, self.identical_devices)
        )

    def run(self, params, boundary_params):
        worker_args = []
        b_offset = 0
        p_offset = 0

        for p_idx in range(4):
            reqs = self.patch_fence_reqs[p_idx]
            n_patch_params = len(self.patches[p_idx]) * 2
            p_params = params[p_offset : p_offset + n_patch_params]
            p_offset += n_patch_params

            b_size = len(reqs) * 4
            b_params = boundary_params[b_offset : b_offset + b_size]
            b_offset += b_size

            worker_args.append((p_params, b_params, p_idx, self.intra_patch_edges, reqs))

        if p_offset != len(params):
            raise ValueError(f"Param count mismatch: consumed {p_offset}, expected {len(params)}")
        if b_offset != len(boundary_params):
            raise ValueError(f"Boundary param count mismatch: consumed {b_offset}, expected {len(boundary_params)}")

        try:
            results = self.pool.starmap_async(isolated_holographic_worker, worker_args).get(timeout=600)
        except mp.TimeoutError:
            self._rebuild_pool()
            raise RuntimeError("GPU worker timed out.")
        except Exception as exc:
            if "WorkerLost" in type(exc).__name__ or "RemoteTraceback" in str(exc):
                self._rebuild_pool()
            raise

        total_energy = 0.0
        patch_marginals = {}

        for p_idx, intra_energy, boundaries in results:
            total_energy += intra_energy
            patch_marginals[p_idx] = boundaries

        # Correlator signs for the |Φ+⟩ = (|00⟩+|11⟩)/√2 Bell State. 
        # Note: If targeting |Ψ+⟩, the sign of ⟨YY⟩ would flip to positive.
        bell_signs = {'I': 1.0, 'X': 1.0, 'Y': -1.0, 'Z': 1.0}

        for (pA, qA), (pB, qB) in self.fence_edges:
            margA = patch_marginals[pA][qA]
            margB = patch_marginals[pB][qB]

            x_term = 0.25 * sum(bell_signs[P] * margA['X'][P] * margB['X'][P] for P in ['I', 'X', 'Y', 'Z'])
            y_term = 0.25 * sum(bell_signs[P] * margA['Y'][P] * margB['Y'][P] for P in ['I', 'X', 'Y', 'Z'])

            total_energy -= (x_term + y_term)

        return total_energy

    def shutdown(self):
        self.pool.close()
        t = threading.Thread(target=self.pool.join, daemon=True)
        t.start()
        t.join(timeout=10)
        if t.is_alive():
            self.pool.terminate()
            self.pool.join()

# ==========================================
# 5. MULTI-PROCESSED CPU ORACLE
# ==========================================
class MonolithicCPUEngine:
    def __init__(self):
        self.num_qubits = 72
        self.patches, self.fence_edges = get_topology()
        self.intra_patch_edges = get_3x6_edges()

        self.global_edges = []
        for r in range(6):
            for c in range(12):
                idx = r * 12 + c
                if c < 11: self.global_edges.append((idx, idx + 1))
                if r < 5:  self.global_edges.append((idx, idx + 12))

        self.p_l_to_global = {}
        for p_idx, patch in enumerate(self.patches):
            for g_idx, l_idx in patch:
                self.p_l_to_global[(p_idx, l_idx)] = g_idx

        # Leave one core free for OS/GPU management
        self.num_workers = max(1, mp.cpu_count() - 1)
        self.ctx = mp.get_context('spawn')
        self.pool = self.ctx.Pool(
            processes=self.num_workers,
            initializer=init_oracle_pool_worker
        )

    def run(self, params):
        # Pure-Python fractional chunking perfectly divides the edges list 
        # without invoking NumPy arrays, preserving the raw Python `int` 
        # tuple types required by the PyQrack C++ bindings.
        k, m = divmod(len(self.global_edges), self.num_workers)
        chunks = [
            self.global_edges[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
            for i in range(self.num_workers)
        ]
        # Drop any empty chunks if num_workers > global_edges
        chunks = [c for c in chunks if c]
        
        worker_args = [
            (params, chunk, self.num_qubits, self.patches, self.intra_patch_edges, self.fence_edges, self.p_l_to_global)
            for chunk in chunks
        ]
        
        # Fire off the multi-processed evaluations and block until complete
        results = self.pool.starmap(oracle_evaluation_chunk, worker_args)
        
        return sum(results)

    def shutdown(self):
        self.pool.close()
        self.pool.join()

# ==========================================
# 6. EXECUTION & TRAINING TARGET
# ==========================================
if __name__ == "__main__":
    # Single-GPU Testing Mode: Pinging device 0 four times will serialize access. 
    # To scale, change to distinct physical device IDs (e.g., [0, 1, 2, 3]).
    AVAILABLE_GPUS = [0, 0, 0, 0]
    np.random.seed(42)

    holographic_engine = HolographicDistributedEngine(device_ids=AVAILABLE_GPUS)
    cpu_oracle = MonolithicCPUEngine()

    try:
        expected_n_params = sum(len(patch) * 2 for patch in holographic_engine.patches)
        test_params = np.random.uniform(-np.pi, np.pi, expected_n_params)
        test_boundary_params = np.random.uniform(-np.pi, np.pi, holographic_engine.total_b_params)

        """
        ====================================================================
        ARCHITECTURAL NOTE ON THE OBJECTIVE FUNCTION (LOSS):
        
        The Oracle evaluates the exact physical energy of the un-bathed 72-qubit 
        ansatz (bare CX connections across the patch boundaries). 
        
        The Holographic Engine attempts to mimic that global state locally 
        by cutting the lattice and wrapping each patch in a parameterized 
        entanglement bath layer (Ry, Rz rotations + CX to Ancillas). 
        
        Therefore, the "Loss" printed below is NOT a standard VQE gradient loss 
        converging to a chemical ground state. It is an ENTANGLEMENT EMBEDDING LOSS, 
        measuring how accurately the local bath parameters can approximate the true 
        global lattice state defined by the current physical parameters.
        ====================================================================
        """

        print(f"STEP 1: Calculating Ground Truth (Multi-Processed 72-qubit Oracle across {cpu_oracle.num_workers} cores)...")
        start_time = time.time()
        exact_energy = cpu_oracle.run(test_params)
        print(f"Oracle Execution complete in {time.time() - start_time:.2f}s")
        print(f"Target Global Energy: {exact_energy:.8f}\n")

        print("STEP 2: Calculating Holographic Approximation (Single-GPU 4-Patch)...")
        start_time = time.time()
        approx_energy = holographic_engine.run(test_params, test_boundary_params)
        print(f"Distributed Execution complete in {time.time() - start_time:.2f}s")
        print(f"Holographic Embedded Energy: {approx_energy:.8f}\n")

        print("-" * 50)
        print("TRAINING LOSS:")
        loss = abs(exact_energy - approx_energy)
        print(f"Current Error (Loss): {loss:.8f}")

    finally:
        print("\nShutting down engines...")
        holographic_engine.shutdown()
        cpu_oracle.shutdown()
