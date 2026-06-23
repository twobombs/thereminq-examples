import os
import gc
import time
import numpy as np
import multiprocessing as mp

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
            # Vertical boundary (middle column, c=5)
            if c == 5:
                global_2 = r * 12 + (c + 1)
                fence_edges.append((global_to_local[global_1], global_to_local[global_2]))
            # Horizontal boundary (middle row, r=2)
            if r == 2:
                global_2 = (r + 1) * 12 + c
                fence_edges.append((global_to_local[global_1], global_to_local[global_2]))

    return patches, fence_edges

# ==========================================
# 2. ISOLATED WORKER TASKS & INITIALIZERS
# ==========================================
def init_worker(device_queue):
    device_id = device_queue.get()
    try:
        os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(device_id)
        os.environ["QRACK_QPAGER_DEVICES"] = str(device_id)
        os.environ["QRACK_QUNITMULTI_DEVICES"] = str(device_id)

        from pyqrack import QrackSimulator

        _warmup = None
        try:
            _warmup = QrackSimulator(qubitCount=1)
        finally:
            del _warmup

        device_queue.put(device_id)
    except Exception:
        raise

def init_oracle_worker():
    os.environ["QRACK_QBDD_SEPARABILITY_THRESHOLD"] = "-1"
    os.environ["QRACK_MAX_PAGING_QB"] = "0"
    os.environ["QRACK_OCL_DEFAULT_DEVICE"] = "-1"

    from pyqrack import QrackSimulator
    _warmup = None
    try:
        _warmup = QrackSimulator(
            qubitCount=1,
            isOpenCL=False,
            isPaged=False,
            isBinaryDecisionTree=True,
            isTensorNetwork=False,
            isSchmidtDecompose=False,
        )
    finally:
        del _warmup

def isolated_holographic_worker(patch_params, boundary_params, patch_idx, intra_edges, fence_qubits):
    from pyqrack import QrackSimulator

    num_ancillas = len(fence_qubits)
    num_physical = len(patch_params) // 2
    total_qubits = num_physical + num_ancillas

    if len(boundary_params) != num_ancillas * 4:
        raise ValueError(f"Boundary parameter size mismatch: expected {num_ancillas * 4}, got {len(boundary_params)}")

    sim = QrackSimulator(qubitCount=total_qubits)
    sim_sid = getattr(sim, 'sid', None)
    if sim_sid is None: raise RuntimeError("PyQrack cloneSid unavailable.")

    # 1. Apply Holographic Bath Layer
    b_idx = 0
    for q in fence_qubits:
        sim.ry(boundary_params[b_idx], q)
        sim.rz(boundary_params[b_idx + 1], q)
        b_idx += 2

    ancilla_indices = list(range(num_physical, total_qubits))
    for q in ancilla_indices:
        sim.ry(boundary_params[b_idx], q)
        sim.rz(boundary_params[b_idx + 1], q)
        b_idx += 2

    for i, f_q in enumerate(fence_qubits):
        sim.cx(ancilla_indices[i], f_q)

    # 2. Apply Main Physical Ansatz
    param_idx = 0
    for q in range(num_physical):
        sim.rx(patch_params[param_idx], q)
        sim.ry(patch_params[param_idx + 1], q)
        param_idx += 2

    for q1, q2 in intra_edges:
        sim.cx(q1, q2)

    # 3. Evaluate Intra-Patch Energy
    intra_energy = 0.0
    for q1, q2 in intra_edges:
        for basis in ['X', 'Y']:
            s_clone = QrackSimulator(cloneSid=sim_sid)
            if basis == 'X':
                s_clone.h(q1); s_clone.h(q2)
            else:
                s_clone.rz(-np.pi / 2, q1); s_clone.rz(-np.pi / 2, q2)
                s_clone.h(q1); s_clone.h(q2)
            s_clone.cx(q1, q2)
            p_odd = s_clone.prob(q2)
            intra_energy += (1.0 - p_odd) - p_odd
            del s_clone

    # 4. Evaluate Boundary Marginals
    boundary_expectations = {}
    for i, q in enumerate(fence_qubits):
        a_q = ancilla_indices[i]
        boundary_expectations[q] = {}

        for b_phys in ['X', 'Y']:
            boundary_expectations[q][b_phys] = {}
            for b_anc in ['I', 'X', 'Y', 'Z']:
                s_clone = QrackSimulator(cloneSid=sim_sid)

                if b_phys == 'X':
                    s_clone.h(q)
                elif b_phys == 'Y':
                    s_clone.rz(-np.pi / 2, q)
                    s_clone.h(q)

                if b_anc == 'X':
                    s_clone.h(a_q)
                elif b_anc == 'Y':
                    s_clone.rz(-np.pi / 2, a_q)
                    s_clone.h(a_q)

                if b_anc == 'I':
                    p_odd = s_clone.prob(q)
                else:
                    s_clone.cx(q, a_q)
                    p_odd = s_clone.prob(a_q)

                boundary_expectations[q][b_phys][b_anc] = (1.0 - p_odd) - p_odd
                del s_clone

    del sim
    return patch_idx, intra_energy, boundary_expectations

def isolated_oracle_worker(params, num_qubits, patches, intra_patch_edges, fence_edges, p_l_to_global, global_edges):
    from pyqrack import QrackSimulator
    sim = QrackSimulator(
        qubitCount=num_qubits,
        isOpenCL=False,
        isPaged=False,
        isBinaryDecisionTree=True,
        isTensorNetwork=False,
        isSchmidtDecompose=False,
    )

    param_idx = 0
    for p_idx, patch in enumerate(patches):
        for g, l in patch:
            sim.rx(params[param_idx], g)
            sim.ry(params[param_idx + 1], g)
            param_idx += 2
        for q1_local, q2_local in intra_patch_edges:
            g1 = p_l_to_global[(p_idx, q1_local)]
            g2 = p_l_to_global[(p_idx, q2_local)]
            sim.cx(g1, g2)

    for (pA, qA), (pB, qB) in fence_edges:
        sim.cx(p_l_to_global[(pA, qA)], p_l_to_global[(pB, qB)])

    energy = 0.0
    sim_sid = getattr(sim, 'sid', None)

    for q1, q2 in global_edges:
        for basis in ['X', 'Y']:
            s_clone = QrackSimulator(cloneSid=sim_sid)
            if basis == 'X':
                s_clone.h(q1); s_clone.h(q2)
            else:
                s_clone.rz(-np.pi / 2, q1); s_clone.rz(-np.pi / 2, q2)
                s_clone.h(q1); s_clone.h(q2)

            s_clone.cx(q1, q2)
            p_odd = s_clone.prob(q2)
            energy += (1.0 - p_odd) - p_odd
            del s_clone

    del sim
    gc.collect()
    return energy

# ==========================================
# 3. HOLOGRAPHIC ORCHESTRATOR
# ==========================================
class HolographicDistributedEngine:
    def __init__(self, device_ids):
        self.device_ids = device_ids
        self.patches, self.fence_edges = get_topology()
        self.intra_patch_edges = get_3x6_edges()

        temp_reqs = {i: set() for i in range(4)}
        for (pA, qA), (pB, qB) in self.fence_edges:
            temp_reqs[pA].add(qA)
            temp_reqs[pB].add(qB)

        self.patch_fence_reqs = {i: sorted(list(temp_reqs[i])) for i in range(4)}

        self.total_b_params = 0
        for p_idx in range(4):
            self.total_b_params += len(self.patch_fence_reqs[p_idx]) * 4

        self.ctx = mp.get_context('spawn')
        self.device_queue = self.ctx.Queue()
        for d in self.device_ids:
            self.device_queue.put(d)

        self.pool = self.ctx.Pool(
            processes=len(self.device_ids),
            initializer=init_worker,
            initargs=(self.device_queue,)
        )

    def _rebuild_pool(self):
        try:
            self.pool.terminate()
            self.pool.join()
        except Exception: pass
        while not self.device_queue.empty():
            try: self.device_queue.get_nowait()
            except Exception: break
        for d in self.device_ids:
            self.device_queue.put(d)
        self.pool = self.ctx.Pool(len(self.device_ids), init_worker, (self.device_queue,))

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

        try:
            results = self.pool.starmap_async(isolated_holographic_worker, worker_args).get(timeout=300)
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

        bell_signs = {'I': 1.0, 'X': 1.0, 'Y': -1.0, 'Z': 1.0}

        for (pA, qA), (pB, qB) in self.fence_edges:
            margA = patch_marginals[pA][qA]
            margB = patch_marginals[pB][qB]

            x_term = 0.25 * sum(bell_signs[P] * margA['X'][P] * margB['X'][P] for P in ['I', 'X', 'Y', 'Z'])
            y_term = 0.25 * sum(bell_signs[P] * margA['Y'][P] * margB['Y'][P] for P in ['I', 'X', 'Y', 'Z'])

            total_energy += (x_term + y_term)

        return total_energy

    def shutdown(self):
        self.pool.close()
        self.pool.join()

# ==========================================
# 4. MONOLITHIC CPU ORACLE
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

        self.ctx = mp.get_context('spawn')
        self.pool = self.ctx.Pool(1, initializer=init_oracle_worker)

    def run(self, params):
        try:
            return self.pool.apply_async(
                isolated_oracle_worker,
                args=(params, self.num_qubits, self.patches, self.intra_patch_edges, self.fence_edges, self.p_l_to_global, self.global_edges)
            ).get(timeout=600)
        except mp.TimeoutError:
            raise RuntimeError("Oracle worker timed out.")

    def shutdown(self):
        self.pool.close()
        self.pool.join()

# ==========================================
# 5. EXECUTION & TRAINING TARGET
# ==========================================
if __name__ == "__main__":
    # We pass '0' four times. The Orchestrator spins up 4 workers,
    # and they will all map their OpenCL context to the single GPU (Device 0).
    AVAILABLE_GPUS = [0, 0, 0, 0]
    np.random.seed(42)

    holographic_engine = HolographicDistributedEngine(device_ids=AVAILABLE_GPUS)
    cpu_oracle = MonolithicCPUEngine()

    try:
        expected_n_params = sum(len(patch) * 2 for patch in holographic_engine.patches)
        test_params = np.random.uniform(-np.pi, np.pi, expected_n_params)
        test_boundary_params = np.random.uniform(-np.pi, np.pi, holographic_engine.total_b_params)

        print("STEP 1: Calculating Ground Truth (Monolithic 72-qubit Oracle)...")
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
