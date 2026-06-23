import os
import gc
import time
import queue
import threading
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
def init_worker(device_queue, identical_devices=False):
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
            if _warmup is not None:
                del _warmup

        if not identical_devices:
            device_queue.put(device_id)
    except Exception:
        if not identical_devices:
            device_queue.put(device_id)
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
        if _warmup is not None:
            del _warmup

def isolated_holographic_worker(patch_params, boundary_params, patch_idx, intra_edges, fence_qubits):
    from pyqrack import QrackSimulator

    num_ancillas = len(fence_qubits)
    num_physical = len(patch_params) // 2
    total_qubits = num_physical + num_ancillas

    if len(boundary_params) != num_ancillas * 4:
        raise ValueError(f"Boundary parameter size mismatch: expected {num_ancillas * 4}, got {len(boundary_params)}")

    sim = QrackSimulator(qubitCount=total_qubits)

    # 1. Apply Main Physical Ansatz FIRST
    param_idx = 0
    for q in range(num_physical):
        sim.rx(patch_params[param_idx], q)
        sim.ry(patch_params[param_idx + 1], q)
        param_idx += 2

    for q1, q2 in intra_edges:
        sim.cx(q1, q2)

    # 2. Apply Holographic Bath Layer
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

    # 3. Capture SIM_SID post-circuit
    sim_sid = getattr(sim, 'sid', None)
    if sim_sid is None: 
        raise RuntimeError("PyQrack cloneSid unavailable.")

    # 4. Evaluate Intra-Patch Energy
    intra_energy = 0.0
    for q1, q2 in intra_edges:
        for basis in ['X', 'Y']:
            s_clone = QrackSimulator(cloneSid=sim_sid)
            if basis == 'X':
                s_clone.h(q1); s_clone.h(q2)
            else:
                s_clone.rx(np.pi / 2, q1); s_clone.rx(np.pi / 2, q2)
                
            s_clone.cx(q1, q2)
            p_odd = s_clone.prob(q2)
            intra_energy -= ((1.0 - p_odd) - p_odd)
            del s_clone

    # 5. Evaluate Boundary Marginals
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
                    s_clone.rx(np.pi / 2, q)

                if b_anc == 'X':
                    s_clone.h(a_q)
                elif b_anc == 'Y':
                    s_clone.rx(np.pi / 2, a_q)

                if b_anc == 'I':
                    p_odd = s_clone.prob(q)
                else:
                    s_clone.cx(q, a_q)
                    p_odd = s_clone.prob(a_q)

                boundary_expectations[q][b_phys][b_anc] = (1.0 - p_odd) - p_odd
                del s_clone

    del sim
    return patch_idx, intra_energy, boundary_expectations

UNRECOVERABLE = ('cloneSid', 'unavailable', 'base init')

def persistent_oracle_worker(task_queue, result_queue, num_qubits, patches, intra_patch_edges, fence_edges, p_l_to_global, global_edges):
    """
    Maintains a persistent process to avoid OS-level fork/spawn overhead per VQE step.
    Note: The base simulator is initialized as an empty |0> state rather than a 
    pre-baked topology. Because the ansatz requires physical parameters to be applied 
    before the CX layers, we cannot pre-bake the CX network without altering the 
    expressivity of the VQE target state. 
    """
    try:
        init_oracle_worker()
        from pyqrack import QrackSimulator

        base_sim = QrackSimulator(
            qubitCount=num_qubits,
            isOpenCL=False,
            isPaged=False,
            isBinaryDecisionTree=True,
            isTensorNetwork=False,
            isSchmidtDecompose=False,
        )
        
        base_sim_sid = getattr(base_sim, 'sid', None)
        if base_sim_sid is None:
            raise RuntimeError("PyQrack cloneSid unavailable during Oracle base init.")
            
        result_queue.put("READY")
    except Exception as e:
        result_queue.put(e)
        return

    while True:
        try:
            params = task_queue.get()
            if params is None:  # Sentinel to kill worker
                break
                
            s_eval = QrackSimulator(cloneSid=base_sim_sid)

            param_idx = 0
            for p_idx, patch in enumerate(patches):
                for g, l in patch:
                    s_eval.rx(params[param_idx], g)
                    s_eval.ry(params[param_idx + 1], g)
                    param_idx += 2
                for q1_local, q2_local in intra_patch_edges:
                    g1 = p_l_to_global[(p_idx, q1_local)]
                    g2 = p_l_to_global[(p_idx, q2_local)]
                    s_eval.cx(g1, g2)

            for (pA, qA), (pB, qB) in fence_edges:
                s_eval.cx(p_l_to_global[(pA, qA)], p_l_to_global[(pB, qB)])

            energy = 0.0
            eval_sid = getattr(s_eval, 'sid', None)
            
            if eval_sid is None:
                raise RuntimeError("PyQrack cloneSid unavailable during Oracle evaluation.")

            for q1, q2 in global_edges:
                for basis in ['X', 'Y']:
                    s_clone = QrackSimulator(cloneSid=eval_sid)
                    if basis == 'X':
                        s_clone.h(q1); s_clone.h(q2)
                    else:
                        s_clone.rx(np.pi / 2, q1); s_clone.rx(np.pi / 2, q2)

                    s_clone.cx(q1, q2)
                    p_odd = s_clone.prob(q2)
                    energy -= ((1.0 - p_odd) - p_odd)
                    del s_clone

            result_queue.put(energy)
            
        except Exception as e:
            result_queue.put(e)
            if any(kw in str(e) for kw in UNRECOVERABLE):
                break  # Unrecoverable error; exit to prevent stale exception loop
            
        finally:
            try:
                del s_eval
            except NameError:
                pass
            gc.collect()
            
    try:
        del base_sim
    except NameError:
        pass

# ==========================================
# 3. HOLOGRAPHIC ORCHESTRATOR
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

            total_energy -= (x_term + y_term)

        return total_energy

    def shutdown(self):
        self.pool.close()
        t = threading.Thread(target=self.pool.join, daemon=True)
        t.start()
        t.join(timeout=10)
        
        self.pool.terminate()
        self.pool.join()

# ==========================================
# 4. MONOLITHIC CPU ORACLE (Persistent)
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
        self.task_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        
        self.worker = self.ctx.Process(
            target=persistent_oracle_worker,
            args=(
                self.task_queue, 
                self.result_queue, 
                self.num_qubits, 
                self.patches, 
                self.intra_patch_edges, 
                self.fence_edges, 
                self.p_l_to_global, 
                self.global_edges
            )
        )
        self.worker.start()
        
        # Readiness Handshake
        try:
            status = self.result_queue.get(timeout=120)
            if isinstance(status, Exception):
                raise status
            if status != "READY":
                raise RuntimeError(f"Oracle worker returned unexpected init status: {status!r}")
        except queue.Empty:
            self.worker.terminate()
            self.worker.join()
            raise RuntimeError("Oracle worker failed to initialize within timeout.")

    def run(self, params):
        if not self.worker.is_alive():
            raise RuntimeError("Oracle worker has exited. Recreate the engine to continue.")
            
        self.task_queue.put(params)
        try:
            res = self.result_queue.get(timeout=600)
            if isinstance(res, Exception):
                raise res
            return res
        except queue.Empty:
            raise RuntimeError("Oracle worker timed out.")
            # Note: A timeout expiry here implies worker death or heavy load.

    def shutdown(self):
        self.task_queue.put(None)
        self.worker.join(timeout=5)
        if self.worker.is_alive():
            self.worker.terminate()
            self.worker.join()

# ==========================================
# 5. EXECUTION & TRAINING TARGET
# ==========================================
if __name__ == "__main__":
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
