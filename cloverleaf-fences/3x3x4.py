import os
import gc
import time
import numpy as np
import multiprocessing as mp

# ==========================================
# 1. TOPOLOGY DEFINITIONS
# ==========================================
def get_3x3_edges():
    """Returns nearest-neighbor edges for a local 3x3 grid."""
    edges = []
    for r in range(3):
        for c in range(3):
            idx = r * 3 + c
            if c < 2: edges.append((idx, idx + 1)) 
            if r < 2: edges.append((idx, idx + 3)) 
    return edges

def get_topology():
    """Returns intra-patch mappings and inter-patch fence edges for 6x6."""
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
# 2. ISOLATED GPU WORKER PROCESS
# ==========================================
def isolated_patch_worker(device_id, patch_params, patch_idx, intra_edges, fence_qubits_to_measure):
    """
    Runs in a completely separate OS process. 
    Imports PyQrack ONLY AFTER setting the environment variable to avoid static caching.
    Receives exactly 18 parameters specific to its patch.
    """
    os.environ["QRACK_DEFAULT_DEVICE"] = str(device_id)
    
    from pyqrack import QrackSimulator
    import numpy as np
    
    sim = QrackSimulator(qubitCount=9)
    
    # 1. Apply Ansatz
    param_idx = 0
    for q in range(9):
        sim.rx(patch_params[param_idx], q)
        sim.ry(patch_params[param_idx + 1], q)
        param_idx += 2
        
    for q1, q2 in intra_edges:
        sim.cx(q1, q2)
        
    # 2. Evaluate Intra-Patch Energy
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

    # 3. Evaluate Marginal Boundary Expectations for the Fence
    boundary_expectations = {}
    for q in fence_qubits_to_measure:
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
# 3. DISTRIBUTED ORCHESTRATOR
# ==========================================
class DistributedVQEEngine:
    def __init__(self, device_ids):
        self.device_ids = device_ids
        self.patches, self.fence_edges, self.mapping = get_topology()
        self.intra_patch_edges = get_3x3_edges()
        
        self.patch_fence_reqs = {i: set() for i in range(4)}
        for (pA, qA), (pB, qB) in self.fence_edges:
            self.patch_fence_reqs[pA].add(qA)
            self.patch_fence_reqs[pB].add(qB)
            
    def run(self, params):
        worker_args = []
        for p_idx in range(4):
            dev_id = self.device_ids[p_idx % len(self.device_ids)]
            reqs = list(self.patch_fence_reqs[p_idx])
            
            patch_params = params[p_idx * 18 : (p_idx + 1) * 18]
            worker_args.append((dev_id, patch_params, p_idx, self.intra_patch_edges, reqs))
            
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=4) as pool:
            results = pool.starmap(isolated_patch_worker, worker_args)
            
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
# 4. MONOLITHIC CPU VERIFICATION ENGINE
# ==========================================
class MonolithicCPUEngine:
    def __init__(self, use_qbdd=True):
        self.num_qubits = 36
        self.use_qbdd = use_qbdd
        self.patches, _, _ = get_topology()
        self.intra_patch_edges = get_3x3_edges()
        
        self.global_edges = []
        for r in range(6):
            for c in range(6):
                idx = r * 6 + c
                if c < 5: self.global_edges.append((idx, idx + 1))
                if r < 5: self.global_edges.append((idx, idx + 6))

    def run(self, params):
        from pyqrack import QrackSimulator
        
        # Initialize with QBDD flag if requested to prevent massive RAM/swap allocation
        sim = QrackSimulator(
            qubitCount=self.num_qubits, 
            isOpenCL=False, 
            isPaged=(not self.use_qbdd),
            isBinaryDecisionTree=self.use_qbdd
        )
        
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
# 5. EXECUTION & VERIFICATION N-TIMES
# ==========================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    N_RUNS = 5
    AVAILABLE_GPUS = [0, 1, 2, 3] 
    
    np.random.seed(42)
    test_params = np.random.uniform(-np.pi, np.pi, 72)
    
    gpu_engine = DistributedVQEEngine(device_ids=AVAILABLE_GPUS)
    
    # Toggle use_qbdd here to use Quantum Binary Decision Diagrams
    cpu_engine = MonolithicCPUEngine(use_qbdd=True)
    
    print(f"Starting Verification: {N_RUNS} runs...")
    print("-" * 50)
    
    start_time = time.time()
    gpu_results = []
    for i in range(N_RUNS):
        res = gpu_engine.run(test_params)
        gpu_results.append(res)
    gpu_time = time.time() - start_time
    
    print(f"Distributed Multi-Die Executions complete in {gpu_time:.2f}s")
    print(f"Average Energy: {np.mean(gpu_results):.8f}\n")
    
    print("Allocating monolithic 36-qubit states to CPU (QBDD mode active)...")
    start_time = time.time()
    cpu_results = []
    for i in range(N_RUNS):
        res = cpu_engine.run(test_params)
        cpu_results.append(res)
    cpu_time = time.time() - start_time
    
    print(f"Monolithic CPU Executions complete in {cpu_time:.2f}s")
    print(f"Average Energy: {np.mean(cpu_results):.8f}\n")
    
    print("-" * 50)
    print("VERIFICATION RESULTS:")
    
    diff = abs(np.mean(gpu_results) - np.mean(cpu_results))
    print(f"Absolute Energy Difference: {diff:.8e}")
    if diff < 1e-6:
        print("[SUCCESS] Distributed Marginal Evaluation perfectly matches Monolithic Global Evaluation.")
    else:
        print("[FAILED] State evaluation diverged.")
