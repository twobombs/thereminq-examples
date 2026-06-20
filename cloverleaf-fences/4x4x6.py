import os
import gc
import time
import numpy as np
import multiprocessing as mp

# ==========================================
# 1. TOPOLOGY DEFINITIONS (96-Qubit Global Grid)
# ==========================================
def get_4x4_edges():
    """Intra-patch entanglement for a 4x4 physical sub-grid (16 qubits)"""
    edges = []
    for r in range(4):
        for c in range(4):
            idx = r * 4 + c
            if c < 3: edges.append((idx, idx + 1)) 
            if r < 3: edges.append((idx, idx + 4)) 
    return edges

def get_topology():
    """
    Creates an 8x12 global grid (96 qubits), 
    divided into a 2x3 grid of 4x4 patches (6 total patches).
    """
    patches = [[] for _ in range(6)]
    fence_edges = []
    
    for r in range(8):
        for c in range(12):
            idx = r * 12 + c
            patch_r = r // 4
            patch_c = c // 4
            patch_idx = patch_r * 3 + patch_c
            local_idx = len(patches[patch_idx])
            patches[patch_idx].append((idx, local_idx))

    global_to_local = {}
    for p_idx, patch in enumerate(patches):
        for global_idx, local_idx in patch:
            global_to_local[global_idx] = (p_idx, local_idx)

    for r in range(8):
        for c in range(12):
            global_1 = r * 12 + c
            if c % 4 == 3 and c < 11:
                global_2 = r * 12 + (c + 1)
                fence_edges.append((global_to_local[global_1], global_to_local[global_2]))
            if r % 4 == 3 and r < 7:
                global_2 = (r + 1) * 12 + c
                fence_edges.append((global_to_local[global_1], global_to_local[global_2]))

    return patches, fence_edges, global_to_local

# ==========================================
# 2. PERSISTENT WORKER INITIALIZATION
# ==========================================
def init_worker(device_queue):
    """
    Initializes OpenCL variables exactly once per spawned worker process, 
    guaranteeing strict isolation and avoiding context setup overhead per call.
    """
    device_id = device_queue.get()
    os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(device_id)
    os.environ["QRACK_QPAGER_DEVICES"] = str(device_id)
    os.environ["QRACK_QUNITMULTI_DEVICES"] = str(device_id)
    
    # Pre-load PyQrack to lock the OpenCL context to this specific GPU
    import pyqrack 

# ==========================================
# 3. ISOLATED GPU WORKER TASK
# ==========================================
def isolated_holographic_worker(patch_params, boundary_params, patch_idx, intra_edges, fence_qubits):
    from pyqrack import QrackSimulator
    
    # Due to set deduplication of corner boundary qubits, middle patches 
    # require 10 unique boundary qubits (16 + 10 = 26 qubits).
    num_ancillas = len(fence_qubits)
    total_qubits = 16 + num_ancillas
    
    if total_qubits > 28:
        print(f"WARNING [Patch {patch_idx}]: Allocating {total_qubits} qubits risks an Out-Of-Memory error.")

    sim = QrackSimulator(qubitCount=total_qubits)
    
    # 1. Apply Holographic Bath Layer
    b_idx = 0
    for q in fence_qubits:
        sim.ry(boundary_params[b_idx], q)
        sim.rz(boundary_params[b_idx + 1], q)
        b_idx += 2
        
    ancilla_indices = list(range(16, total_qubits))
    for q in ancilla_indices:
        sim.ry(boundary_params[b_idx], q)
        sim.rz(boundary_params[b_idx + 1], q)
        b_idx += 2
        
    # Bind bath to physical boundary
    for i, f_q in enumerate(fence_qubits):
        sim.cx(ancilla_indices[i], f_q)
        
    # 2. Apply Main Physical Ansatz (Qubits 0-15)
    param_idx = 0
    for q in range(16):
        sim.rx(patch_params[param_idx], q)
        sim.ry(patch_params[param_idx + 1], q)
        param_idx += 2
        
    for q1, q2 in intra_edges:
        sim.cx(q1, q2)
        
    # 3. Evaluate Intra-Patch Energy
    intra_energy = 0.0
    for q1, q2 in intra_edges:
        for basis in ['X', 'Y']:
            s_clone = QrackSimulator(cloneSid=sim.sid)
            if basis == 'X':
                s_clone.h(q1); s_clone.h(q2)
            else:
                s_clone.rz(-np.pi/2, q1); s_clone.rz(-np.pi/2, q2)
                s_clone.h(q1); s_clone.h(q2)
                
            s_clone.cx(q1, q2)
            p_odd = s_clone.prob(q2)
            intra_energy += (1.0 - p_odd) - p_odd
            del s_clone

    # 4. Evaluate Boundary Marginals
    boundary_expectations = {}
    for q in fence_qubits:
        boundary_expectations[q] = {}
        for basis in ['X', 'Y']:
            s_clone = QrackSimulator(cloneSid=sim.sid)
            if basis == 'X': 
                s_clone.h(q)
            else: 
                s_clone.rz(-np.pi/2, q)
                s_clone.h(q)
                
            p_odd = s_clone.prob(q)
            boundary_expectations[q][basis] = (1.0 - p_odd) - p_odd
            del s_clone
            
    del sim
    return patch_idx, intra_energy, boundary_expectations

# ==========================================
# 4. HOLOGRAPHIC ORCHESTRATOR (PERSISTENT)
# ==========================================
class HolographicDistributedEngine:
    def __init__(self, device_ids):
        self.device_ids = device_ids
        self.patches, self.fence_edges, self.mapping = get_topology()
        self.intra_patch_edges = get_4x4_edges()
        
        self.patch_fence_reqs = {i: set() for i in range(6)}
        for (pA, qA), (pB, qB) in self.fence_edges:
            self.patch_fence_reqs[pA].add(qA)
            self.patch_fence_reqs[pB].add(qB)
            
        self.total_b_params = 0
        for p_idx in range(6):
            reqs = len(self.patch_fence_reqs[p_idx])
            self.total_b_params += (reqs * 4) 
            
        # Initialize persistent pool for VQE optimization
        self.ctx = mp.get_context('spawn')
        self.device_queue = self.ctx.Queue()
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
        
        for p_idx in range(6):
            reqs = list(self.patch_fence_reqs[p_idx])
            p_params = params[p_idx * 32 : (p_idx + 1) * 32]
            
            b_size = len(reqs) * 4
            b_params = boundary_params[b_offset : b_offset + b_size]
            b_offset += b_size
            
            worker_args.append((p_params, b_params, p_idx, self.intra_patch_edges, reqs))
            
        results = self.pool.starmap(isolated_holographic_worker, worker_args)
            
        total_energy = 0.0
        patch_marginals = {}
        
        for p_idx, intra_energy, boundaries in results:
            total_energy += intra_energy
            patch_marginals[p_idx] = boundaries
            
        for (pA, qA), (pB, qB) in self.fence_edges:
            x_term = patch_marginals[pA][qA]['X'] * patch_marginals[pB][qB]['X']
            y_term = patch_marginals[pA][qA]['Y'] * patch_marginals[pB][qB]['Y']
            total_energy += (x_term + y_term)
            
        return total_energy
        
    def shutdown(self):
        self.pool.close()
        self.pool.join()

# ==========================================
# 5. MONOLITHIC CPU ORACLE (QBDD GROUND TRUTH)
# ==========================================
class MonolithicCPUEngine:
    def __init__(self):
        self.num_qubits = 96
        self.patches, self.fence_edges, self.mapping = get_topology()
        self.intra_patch_edges = get_4x4_edges()
        
        self.global_edges = []
        for r in range(8):
            for c in range(12):
                idx = r * 12 + c
                if c < 11: self.global_edges.append((idx, idx + 1))
                if r < 7: self.global_edges.append((idx, idx + 12))

    def run(self, params):
        from pyqrack import QrackSimulator
        
        sim = QrackSimulator(
            qubitCount=self.num_qubits, 
            isOpenCL=False, 
            isPaged=False,
            isBinaryDecisionTree=True 
        )
        
        # 1. Apply Local Patch Parameters & Intra-Patch Entanglement
        param_idx = 0
        for patch in self.patches:
            local_to_global = {l: g for g, l in patch}
            for g, l in patch:
                sim.rx(params[param_idx], g)
                sim.ry(params[param_idx + 1], g)
                param_idx += 2
                
            for q1_local, q2_local in self.intra_patch_edges:
                q1_global = local_to_global[q1_local]
                q2_global = local_to_global[q2_local]
                sim.cx(q1_global, q2_global)
                
        # 2. Apply Cross-Border Entanglement (The True Fences)
        for (pA, qA), (pB, qB) in self.fence_edges:
            global_A = [g for g, l in self.patches[pA] if l == qA][0]
            global_B = [g for g, l in self.patches[pB] if l == qB][0]
            sim.cx(global_A, global_B)

        # 3. Evaluate Global Energy
        energy = 0.0
        for q1, q2 in self.global_edges:
            for basis in ['X', 'Y']:
                s_clone = QrackSimulator(cloneSid=sim.sid)
                if basis == 'X':
                    s_clone.h(q1); s_clone.h(q2)
                else:
                    s_clone.rz(-np.pi/2, q1); s_clone.rz(-np.pi/2, q2)
                    s_clone.h(q1); s_clone.h(q2)
                    
                s_clone.cx(q1, q2)
                p_odd = s_clone.prob(q2)
                energy += (1.0 - p_odd) - p_odd
                del s_clone

        del sim
        gc.collect()
        return energy

# ==========================================
# 6. EXECUTION & TRAINING TARGET
# ==========================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5] 
    np.random.seed(42)
    test_params = np.random.uniform(-np.pi, np.pi, 192)
    
    holographic_engine = HolographicDistributedEngine(device_ids=AVAILABLE_GPUS)
    cpu_oracle = MonolithicCPUEngine()
    
    test_boundary_params = np.random.uniform(-np.pi, np.pi, holographic_engine.total_b_params) 
    
    print("STEP 1: Calculating Ground Truth (Monolithic 96-qubit QBDD)...")
    start_time = time.time()
    exact_energy = cpu_oracle.run(test_params)
    print(f"Oracle Execution complete in {time.time() - start_time:.2f}s")
    print(f"Target Global Energy: {exact_energy:.8f}\n")
    
    print("STEP 2: Calculating Holographic Approximation (Distributed 6-GPU)...")
    start_time = time.time()
    # Now optimized for VQE: we can call this repeatedly without respawning context!
    approx_energy = holographic_engine.run(test_params, test_boundary_params)
    print(f"Distributed Execution complete in {time.time() - start_time:.2f}s")
    print(f"Holographic Embedded Energy: {approx_energy:.8f}\n")
    
    print("-" * 50)
    print("TRAINING LOSS:")
    loss = abs(exact_energy - approx_energy)
    print(f"Current Error (Loss): {loss:.8f}")
    
    # Graceful exit
    holographic_engine.shutdown()

