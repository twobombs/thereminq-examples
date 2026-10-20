import os
import gc
import time
import numpy as np
import multiprocessing as mp

# ==========================================
# 1. TOPOLOGY DEFINITIONS (108-Qubit 6x18 Grid)
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
    Creates a 6x18 global grid (108 qubits),
    divided into a 2x3 grid of 3x6 patches.
    Returns (patches, fence_edges) — global_to_local is dropped;
    no caller uses it.
    """
    patches = [[] for _ in range(6)]
    fence_edges = []

    for r in range(6):
        for c in range(18):
            idx = r * 18 + c
            patch_r = r // 3
            patch_c = c // 6
            patch_idx = patch_r * 3 + patch_c
            local_idx = len(patches[patch_idx])
            patches[patch_idx].append((idx, local_idx))

    # Build global→local only for fence edge construction, then discard.
    global_to_local = {}
    for p_idx, patch in enumerate(patches):
        for global_idx, local_idx in patch:
            global_to_local[global_idx] = (p_idx, local_idx)

    for r in range(6):
        for c in range(18):
            global_1 = r * 18 + c
            if c % 6 == 5 and c < 17:
                global_2 = r * 18 + (c + 1)
                fence_edges.append((global_to_local[global_1], global_to_local[global_2]))
            if r % 3 == 2 and r < 5:
                global_2 = (r + 1) * 18 + c
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

        # Force OpenCL context initialization and pin the device.
        # Use try/finally so the native handle is always released even
        # if QrackSimulator.__init__ raises after partial allocation.
        _warmup = None
        try:
            _warmup = QrackSimulator(qubitCount=1)
        finally:
            del _warmup

        # Only return the device ID to the pool on a *clean* exit.
        # If we reach here without raising, the worker is healthy.
        device_queue.put(device_id)

    except Exception:
        # Do NOT return device_id on crash — a replacement worker will
        # pull a fresh slot.  Let the pool shrink rather than double-
        # assign the same GPU to two workers.
        raise

def init_oracle_worker():
    """
    Forces the subprocess to use CPU/QBDD paths and suppresses OpenCL
    before any PyQrack module is loaded into memory.
    """
    os.environ["QRACK_QBDD_SEPARABILITY_THRESHOLD"] = "-1"
    os.environ["QRACK_MAX_PAGING_QB"] = "0"
    # Use "-1" (invalid index) rather than "" — some Qrack builds treat
    # an empty string as device 0 rather than "no device".
    os.environ["QRACK_OCL_DEFAULT_DEVICE"] = "-1"

    from pyqrack import QrackSimulator
    _warmup = None
    try:
        _warmup = QrackSimulator(qubitCount=1)
    finally:
        del _warmup

def isolated_holographic_worker(patch_params, boundary_params, patch_idx, intra_edges, fence_qubits):
    from pyqrack import QrackSimulator

    num_ancillas = len(fence_qubits)
    num_physical = len(patch_params) // 2
    total_qubits = num_physical + num_ancillas

    # Explicit validation guards (Optimization-safe)
    if len(boundary_params) != num_ancillas * 4:
        raise ValueError(
            f"Boundary parameter size mismatch: expected {num_ancillas * 4}, "
            f"got {len(boundary_params)}"
        )

    if intra_edges:
        max_local_idx = max(max(e) for e in intra_edges)
        if max_local_idx >= num_physical:
            raise ValueError(
                f"intra_patch_edges references out-of-bounds local qubits: {max_local_idx}"
            )

    sim = QrackSimulator(qubitCount=total_qubits)

    # FIX: Guard immediately after construction, before any GPU work.
    sim_sid = getattr(sim, 'sid', None)
    if sim_sid is None:
        raise RuntimeError("PyQrack cloneSid unavailable — cannot proceed with safe state cloning.")

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
                # FIX: Y-eigenbasis diagonalisation is S†H = rz(-π/2) then H.
                # rz(+π/2) was the conjugate rotation and produced sign-flipped
                # Y⊗Y expectation values.
                s_clone.rz(-np.pi / 2, q1); s_clone.rz(-np.pi / 2, q2)
                s_clone.h(q1); s_clone.h(q2)

            s_clone.cx(q1, q2)
            p_odd = s_clone.prob(q2)
            intra_energy += (1.0 - p_odd) - p_odd
            del s_clone

    # 4. Evaluate Boundary Marginals
    #
    # We collect single-site marginals for each fence qubit in the X and Y
    # Pauli bases, and for each ancilla in the I, X, Y, Z Pauli bases.
    # These are the building blocks for the mean-field Bell-state contraction
    # in the orchestrator.
    #
    # Layout of boundary_expectations[q]:
    #   { 'X': { 'I': <⟨X_q⊗I_a⟩>, 'X': <⟨X_q⊗X_a⟩>, 'Y': <⟨X_q⊗Y_a⟩>, 'Z': <⟨X_q⊗Z_a⟩> },
    #     'Y': { ... same for Y_q ... } }
    #
    # For the 'I' ancilla case we measure only the physical qubit (single-site
    # marginal).  For X/Y/Z ancilla cases we apply the appropriate basis
    # rotation to both qubits, entangle via CNOT, and read the parity from
    # the ancilla — giving the joint two-qubit expectation value.
    #
    # Z⊗Z parity:  rotate neither qubit (already in Z basis), CNOT q→a_q,
    #              read prob(a_q).  CNOT maps |00⟩→|00⟩, |11⟩→|11⟩ for even
    #              parity and |01⟩→|01⟩, |10⟩→|11⟩ for odd — so a_q==1 iff
    #              the two qubits had opposite Z values.  This is correct.
    boundary_expectations = {}
    for i, q in enumerate(fence_qubits):
        a_q = ancilla_indices[i]
        boundary_expectations[q] = {}

        for b_phys in ['X', 'Y']:
            boundary_expectations[q][b_phys] = {}
            for b_anc in ['I', 'X', 'Y', 'Z']:
                s_clone = QrackSimulator(cloneSid=sim_sid)

                # Rotate physical qubit into measurement basis
                if b_phys == 'X':
                    s_clone.h(q)
                elif b_phys == 'Y':
                    # FIX: S†H = rz(-π/2) then H
                    s_clone.rz(-np.pi / 2, q)
                    s_clone.h(q)

                # Rotate ancilla into measurement basis
                if b_anc == 'X':
                    s_clone.h(a_q)
                elif b_anc == 'Y':
                    # FIX: S†H = rz(-π/2) then H
                    s_clone.rz(-np.pi / 2, a_q)
                    s_clone.h(a_q)
                # b_anc == 'Z' or 'I': no rotation needed

                if b_anc == 'I':
                    # Single-site marginal on the physical qubit only
                    p_odd = s_clone.prob(q)
                else:
                    # Joint parity: CNOT physical→ancilla, read ancilla.
                    # Works for X, Y, and Z ancilla bases because all three
                    # are already rotated into the Z basis above.
                    s_clone.cx(q, a_q)
                    p_odd = s_clone.prob(a_q)

                boundary_expectations[q][b_phys][b_anc] = (1.0 - p_odd) - p_odd
                del s_clone

    del sim
    return patch_idx, intra_energy, boundary_expectations

def isolated_oracle_worker(params, num_qubits, patches, intra_patch_edges, fence_edges, p_l_to_global, global_edges):
    from pyqrack import QrackSimulator
    sim = QrackSimulator(qubitCount=num_qubits)

    # Circuit ordering:
    # (1) All patch-local RX/RY layers applied across all patches.
    # (2) All intra-patch CX gates across all patches.
    # (3) All cross-patch fence CX gates.
    # This is a simultaneous (not sequential) patch application — every patch's
    # local ansatz is fully built before any fence gate fires.  Document this
    # explicitly so callers do not assume a patch-sequential ordering.
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
        global_A = p_l_to_global[(pA, qA)]
        global_B = p_l_to_global[(pB, qB)]
        sim.cx(global_A, global_B)

    energy = 0.0

    sim_sid = getattr(sim, 'sid', None)
    if sim_sid is None:
        raise RuntimeError("PyQrack cloneSid unavailable — cannot proceed with safe state cloning.")

    for q1, q2 in global_edges:
        for basis in ['X', 'Y']:
            s_clone = QrackSimulator(cloneSid=sim_sid)

            if basis == 'X':
                s_clone.h(q1); s_clone.h(q2)
            else:
                # FIX: S†H = rz(-π/2) then H  (was rz(+π/2) — wrong sign)
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

        # Build requirements as deterministic sorted lists to ensure stable param mappings
        temp_reqs = {i: set() for i in range(6)}
        for (pA, qA), (pB, qB) in self.fence_edges:
            temp_reqs[pA].add(qA)
            temp_reqs[pB].add(qB)

        self.patch_fence_reqs = {i: sorted(list(temp_reqs[i])) for i in range(6)}

        self.total_b_params = 0
        for p_idx in range(6):
            reqs = len(self.patch_fence_reqs[p_idx])
            self.total_b_params += (reqs * 4)

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
        """Tear down and reconstruct the worker pool after a native crash."""
        try:
            self.pool.terminate()
            self.pool.join()
        except Exception:
            pass
        # Drain and refill device queue
        while not self.device_queue.empty():
            try:
                self.device_queue.get_nowait()
            except Exception:
                break
        for d in self.device_ids:
            self.device_queue.put(d)
        self.pool = self.ctx.Pool(
            processes=len(self.device_ids),
            initializer=init_worker,
            initargs=(self.device_queue,)
        )

    def run(self, params, boundary_params):
        worker_args = []
        b_offset = 0
        p_offset = 0

        for p_idx in range(6):
            reqs = self.patch_fence_reqs[p_idx]

            n_patch_params = len(self.patches[p_idx]) * 2
            p_params = params[p_offset : p_offset + n_patch_params]
            p_offset += n_patch_params

            b_size = len(reqs) * 4
            b_params = boundary_params[b_offset : b_offset + b_size]
            b_offset += b_size

            worker_args.append((p_params, b_params, p_idx, self.intra_patch_edges, reqs))

        try:
            # 5-minute hard timeout prevents orchestrator hangs on GPU segfaults/OOMs
            results = self.pool.starmap_async(isolated_holographic_worker, worker_args).get(timeout=300)
        except mp.TimeoutError:
            self._rebuild_pool()
            raise RuntimeError("A GPU worker timed out. Pool has been rebuilt for the next call.")
        except Exception as exc:
            # WorkerLostError or similar native crash — rebuild and re-raise
            if "WorkerLost" in type(exc).__name__ or "RemoteTraceback" in str(exc):
                self._rebuild_pool()
            raise

        total_energy = 0.0
        patch_marginals = {}

        for p_idx, intra_energy, boundaries in results:
            total_energy += intra_energy
            patch_marginals[p_idx] = boundaries

        # -------------------------------------------------------------------------
        # BOUNDARY CONTRACTION — Mean-Field Bell-State Approximation
        #
        # For each fence edge (qA in patch pA, qB in patch pB) we approximate
        # the cross-patch contribution to the Heisenberg-XX+YY Hamiltonian via
        # a mean-field product of local marginals.
        #
        # Target state: |Φ+⟩ = (|00⟩ + |11⟩)/√2
        # Stabilizers:  XX → +1,  YY → -1,  ZZ → +1
        #
        # For a two-qubit operator O_A ⊗ O_B the mean-field approximation gives:
        #   ⟨O_A ⊗ O_B⟩ ≈ ⟨O_A⟩ · ⟨O_B⟩
        #
        # We collect ⟨X_q⊗P_a⟩ and ⟨Y_q⊗P_a⟩ from the workers (P ∈ {I,X,Y,Z}).
        # The ancilla marginals serve as a holographic proxy for the inter-patch
        # entanglement that the mean-field cannot capture directly.
        #
        # bell_signs encodes the target Bell-state eigenvalue for each ancilla
        # Pauli basis used in the contraction:
        #   I → +1 (no constraint from identity),
        #   X → +1 (XX stabilizer),
        #   Y → -1 (YY stabilizer, sign from |Φ+⟩),
        #   Z → +1 (ZZ stabilizer)
        #
        # KNOWN LIMITATION: this is a separable (product-state) approximation
        # across the boundary.  It cannot capture true cross-patch entanglement.
        # Corner qubits appear in multiple fence edges; their marginal is
        # evaluated once and contracted against each neighbour independently.
        # -------------------------------------------------------------------------
        bell_signs = {'I': 1.0, 'X': 1.0, 'Y': -1.0, 'Z': 1.0}

        for (pA, qA), (pB, qB) in self.fence_edges:
            margA = patch_marginals[pA][qA]
            margB = patch_marginals[pB][qB]

            x_term = 0.25 * sum(
                bell_signs[P] * margA['X'][P] * margB['X'][P] for P in ['I', 'X', 'Y', 'Z']
            )
            y_term = 0.25 * sum(
                bell_signs[P] * margA['Y'][P] * margB['Y'][P] for P in ['I', 'X', 'Y', 'Z']
            )

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
        self.num_qubits = 108
        self.patches, self.fence_edges = get_topology()
        self.intra_patch_edges = get_3x6_edges()

        self.global_edges = []
        for r in range(6):
            for c in range(18):
                idx = r * 18 + c
                if c < 17: self.global_edges.append((idx, idx + 1))
                if r < 5:  self.global_edges.append((idx, idx + 18))

        # Tuple-keyed dictionary works natively in CPython Pickle.
        # local_idx convention matches get_3x6_edges() row-major order (0–17).
        # (Consider flattening to 1D array indices if transitioning to Arrow IPC later.)
        self.p_l_to_global = {}
        for p_idx, patch in enumerate(self.patches):
            for g_idx, l_idx in patch:
                self.p_l_to_global[(p_idx, l_idx)] = g_idx

        # Persistent pool to avoid high OpenCL init overheads per VQE step
        self.ctx = mp.get_context('spawn')
        self.pool = self.ctx.Pool(1, initializer=init_oracle_worker)

    def run(self, params):
        try:
            result = self.pool.apply_async(
                isolated_oracle_worker,
                args=(
                    params,
                    self.num_qubits,
                    self.patches,
                    self.intra_patch_edges,
                    self.fence_edges,
                    self.p_l_to_global,
                    self.global_edges
                )
            ).get(timeout=600)  # 10-minute timeout for the 108q Oracle
            return result
        except mp.TimeoutError:
            raise RuntimeError("The Oracle worker timed out (exceeded 600s).")

    def shutdown(self):
        self.pool.close()
        self.pool.join()

# ==========================================
# 5. EXECUTION & TRAINING TARGET
# ==========================================
if __name__ == "__main__":
    AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5]
    np.random.seed(42)

    holographic_engine = HolographicDistributedEngine(device_ids=AVAILABLE_GPUS)
    cpu_oracle = MonolithicCPUEngine()

    try:
        # Dynamically evaluate expected parameters to prevent scaling breaks
        expected_n_params = sum(len(patch) * 2 for patch in holographic_engine.patches)
        test_params = np.random.uniform(-np.pi, np.pi, expected_n_params)

        test_boundary_params = np.random.uniform(-np.pi, np.pi, holographic_engine.total_b_params)

        print("STEP 1: Calculating Ground Truth (Monolithic 108-qubit Oracle)...")
        start_time = time.time()
        exact_energy = cpu_oracle.run(test_params)
        print(f"Oracle Execution complete in {time.time() - start_time:.2f}s")
        print(f"Target Global Energy: {exact_energy:.8f}\n")

        print("STEP 2: Calculating Holographic Approximation (Distributed 6-GPU)...")
        start_time = time.time()
        approx_energy = holographic_engine.run(test_params, test_boundary_params)
        print(f"Distributed Execution complete in {time.time() - start_time:.2f}s")
        print(f"Holographic Embedded Energy: {approx_energy:.8f}\n")

        print("-" * 50)
        print("TRAINING LOSS:")
        loss = abs(exact_energy - approx_energy)
        print(f"Current Error (Loss): {loss:.8f}")

    finally:
        # Guarantee resource cleanup on exception
        print("\nShutting down engines...")
        holographic_engine.shutdown()
        cpu_oracle.shutdown()
