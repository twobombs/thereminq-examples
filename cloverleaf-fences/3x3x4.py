import os
import gc
import time
import numpy as np
import multiprocessing as mp

# ==========================================
# 1. TOPOLOGY DEFINITIONS
# ==========================================
def get_3x3_edges():
    edges = []
    for r in range(3):
        for c in range(3):
            idx = r * 3 + c
            if c < 2: edges.append((idx, idx + 1)) 
            if r < 2: edges.append((idx, idx + 3)) 
    return edges

def get_topology():
    patches = [[] for _ in range(4)]
    fence_edges = []
    
    for r in range(6):
        for c in range(6):
            idx = r * 6 + c
            patch_idx = (0 if r < 3 else 2) + (0 if c < 3 else 1)
            local_idx = len(patches[patch_idx])
            patches[patch_idx].append((idx, local_idx))

    global_to_local = {}
    for p_idx, patch in enumerate(patches):
        for global_idx, local_idx in patch:
            global_to_local[global_idx] = (p_idx, local_idx)

    for r in range(6):
        for c in range(6):
            global_1 = r * 6 + c
            if c == 2:
                global_2 = r * 6 + (c + 1)
                fence_edges.append((global_to_local[global_1], global_to_local[global_2]))
            if r == 2:
                global_2 = (r + 1) * 6 + c
                fence_edges.append((global_to_local[global_1], global_to_local[global_2]))

    return patches, fence_edges, global_to_local

# ==========================================
# 2. ISOLATED GPU WORKER (WITH ANCILLA BATH)
# ==========================================
def isolated_holographic_worker(device_id, patch_params, boundary_params, patch_idx, intra_edges, fence_qubits, num_ancillas):
    # CORRECTED: Utilizing the proper PyQrack OpenCL device variable
    os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(device_id)
    
    from pyqrack import QrackSimulator
    import numpy as np
    
    total_qubits = 9 + num_ancillas
    sim = QrackSimulator(qubitCount=total_qubits)
    
    # 1. Apply Holographic Bath Layer
    b_idx = 0
    for q in fence_qubits:
        sim.ry(boundary_params[b_idx], q)
        sim.rz(boundary_params[b_idx + 1], q)
        b_idx += 2
        
    ancilla_indices = list(range(9, total_qubits))
    for q in ancilla_indices:
        sim.ry(boundary_params[b_idx], q)
        sim.rz(boundary_params[b_idx + 1], q)
        b_idx += 2
        
    for i, f_q in enumerate(fence_qubits):
        ancilla_q = ancilla_indices[i % num_ancillas]
        sim.cx(ancilla_q, f_q)
        
    # 2. Apply Main Physical Ansatz (Qubits 0-8)
    param_idx = 0
    for q in range(9):
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
# 3. HOLOGRAPHIC ORCHESTRATOR
# ==========================================
class HolographicDistributedEngine:
    def __init__(self, device_ids, num_ancillas=3):
        self.device_ids = device_ids
        self.num_ancillas = num_ancillas
        self.patches, self.fence_edges, self.mapping = get_topology()
        self.intra_patch_edges = get_3x3_edges()
        
        self.patch_fence_reqs = {i: set() for i in range(4)}
        for (pA, qA), (pB, qB) in self.fence_edges:
            self.patch_fence_reqs[pA].add(qA)
            self.patch_fence_reqs[pB].add(qB)
            
        self.total_b_params = 0
        for p_idx in range(4):
            reqs = len(self.patch_fence_reqs[p_idx])
            self.total_b_params += (reqs + self.num_ancillas) * 2
            
    def run(self, params, boundary_params):
        worker_args = []
        b_offset = 0
        
        for p_idx in range(4):
            dev_id = self.device_ids[p_idx % len(self.device_ids)]
            reqs = list(self.patch_fence_reqs[p_idx])
            
            p_params = params[p_idx * 18 : (p_idx + 1) * 18]
            
            b_size = (len(reqs) + self.num_ancillas) * 2
            b_params = boundary_params[b_offset : b_offset + b_size]
            b_offset += b_size
            
            worker_args.append((dev_id, p_params, b_params, p_idx, self.intra_patch_edges, reqs, self.num_ancillas))
            
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=4) as pool:
            results = pool.starmap(isolated_holographic_worker, worker_args)
            
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

# ==========================================
# 4. MONOLITHIC CPU ORACLE (GROUND TRUTH)
# ==========================================
class MonolithicCPUEngine:
    def __init__(self, use_qbdd=True):
        self.num_qubits = 36
        self.use_qbdd = use_qbdd
        self.patches, self.fence_edges, self.mapping = get_topology()
        self.intra_patch_edges = get_3x3_edges()
        
        self.global_edges = []
        for r in range(6):
            for c in range(6):
                idx = r * 6 + c
                if c < 5: self.global_edges.append((idx, idx + 1))
                if r < 5: self.global_edges.append((idx, idx + 6))

    def run(self, params):
        from pyqrack import QrackSimulator
        
        sim = QrackSimulator(
            qubitCount=self.num_qubits, 
            isOpenCL=False, 
            isPaged=(not self.use_qbdd),
            isBinaryDecisionTree=self.use_qbdd
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
            # Map patch indices back to global 36-qubit indices
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
# 5. EXECUTION & TRAINING TARGET
# ==========================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    AVAILABLE_GPUS = [0, 1, 2, 3] 
    NUM_ANCILLAS = 3
    
    np.random.seed(42)
    test_params = np.random.uniform(-np.pi, np.pi, 72)
    
    # Initialize Engines
    holographic_engine = HolographicDistributedEngine(device_ids=AVAILABLE_GPUS, num_ancillas=NUM_ANCILLAS)
    cpu_oracle = MonolithicCPUEngine(use_qbdd=True)
    
    test_boundary_params = np.random.uniform(-np.pi, np.pi, holographic_engine.total_b_params) 
    
    print("STEP 1: Calculating Ground Truth (Monolithic 36-qubit QBDD)...")
    start_time = time.time()
    exact_energy = cpu_oracle.run(test_params)
    print(f"Oracle Execution complete in {time.time() - start_time:.2f}s")
    print(f"Target Global Energy: {exact_energy:.8f}\n")
    
    print("STEP 2: Calculating Holographic Approximation (Distributed Multi-GPU)...")
    start_time = time.time()
    approx_energy = holographic_engine.run(test_params, test_boundary_params)
    print(f"Distributed Execution complete in {time.time() - start_time:.2f}s")
    print(f"Holographic Embedded Energy: {approx_energy:.8f}\n")
    
    print("-" * 50)
    print("TRAINING LOSS:")
    loss = abs(exact_energy - approx_energy)
    print(f"Current Error (Loss): {loss:.8f}")
    print("\nTo train this model, your classical optimizer must iteratively update")
    print("'test_boundary_params' to drive this Loss to 0.")
