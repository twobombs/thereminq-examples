import os
import gc
import time
import numpy as np
from pyqrack import QrackSimulator

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
# 2. MULTI-GPU DISTRIBUTED ENGINE
# ==========================================
class GPUPoolAllocator:
    def __init__(self, available_device_ids, memory_limit_mb=8176, bytes_per_amplitude=8):
        self.device_ids = available_device_ids
        self.memory_limit_mb = memory_limit_mb
        self.bytes_per_amp = bytes_per_amplitude
        self.device_usage_mb = {dev: 0.0 for dev in self.device_ids}
        
    def _get_sv_size_mb(self, num_qubits):
        return (2**num_qubits * self.bytes_per_amp) / (1024 * 1024)

    def allocate(self, num_qubits):
        req_mb = self._get_sv_size_mb(num_qubits)
        for dev_id in self.device_ids:
            if self.device_usage_mb[dev_id] + req_mb <= self.memory_limit_mb:
                self.device_usage_mb[dev_id] += req_mb
                return dev_id
        raise MemoryError("GPU memory limit exceeded.")

class DistributedVQEEngine:
    def __init__(self, device_ids):
        precision = int(os.environ.get("QRACK_FPPOW", 5))
        bytes_per_amp = 8 if precision == 5 else 16 
        self.allocator = GPUPoolAllocator(device_ids, bytes_per_amplitude=bytes_per_amp)
        self.patches, self.fence_edges, self.mapping = get_topology()
        self.intra_patch_edges = get_3x3_edges()
        
    def run(self, params):
        self.allocator.device_usage_mb = {dev: 0.0 for dev in self.allocator.device_ids}
        
        # Provision
        sims = []
        original_dev = os.environ.get("QRACK_DEFAULT_DEVICE", "0")
        for i in range(4):
            dev_id = self.allocator.allocate(9)
            os.environ["QRACK_DEFAULT_DEVICE"] = str(dev_id)
            sims.append(QrackSimulator(qubitCount=9))
        os.environ["QRACK_DEFAULT_DEVICE"] = original_dev

        # Apply Ansatz
        param_idx = 0
        for sim in sims:
            for q in range(9):
                sim.rx(params[param_idx], q)
                sim.ry(params[param_idx + 1], q)
                param_idx += 2
            for q1, q2 in self.intra_patch_edges:
                sim.cx(q1, q2)

        # Evaluate
        energy = 0.0
        
        # Intra-patch
        for p_idx in range(4):
            sim = sims[p_idx]
            for q1, q2 in self.intra_patch_edges:
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

        # Inter-patch (Fence)
        for (pA, qA), (pB, qB) in self.fence_edges:
            for basis in ['X', 'Y']:
                expectations = []
                for s, q in [(sims[pA], qA), (sims[pB], qB)]:
                    s_clone = QrackSimulator(cloneSid=s.sid)
                    if basis == 'X': s_clone.h(q)
                    else: s_clone.rz(-np.pi/2, q); s_clone.h(q)
                    p_odd = s_clone.prob(q)
                    expectations.append((1.0 - p_odd) - p_odd)
                    del s_clone
                energy += expectations[0] * expectations[1]
                
        for sim in sims: del sim
        gc.collect()
        
        return energy

# ==========================================
# 3. MONOLITHIC CPU VERIFICATION ENGINE
# ==========================================
class MonolithicCPUEngine:
    def __init__(self):
        self.num_qubits = 36
        self.patches, _, _ = get_topology()
        self.intra_patch_edges = get_3x3_edges()
        
        # Build global 6x6 lattice edges for Hamiltonian evaluation
        self.global_edges = []
        for r in range(6):
            for c in range(6):
                idx = r * 6 + c
                if c < 5: self.global_edges.append((idx, idx + 1))
                if r < 5: self.global_edges.append((idx, idx + 6))

    def run(self, params):
        # Force CPU execution and enable OS paging to handle the ~550GB allocation
        sim = QrackSimulator(qubitCount=self.num_qubits, isOpenCL=False, isPaged=True)
        
        # Apply Ansatz globally but strictly localized to patch logic
        param_idx = 0
        for patch in self.patches:
            local_to_global = {l: g for g, l in patch}
            
            # Rotations
            for g, l in patch:
                sim.rx(params[param_idx], g)
                sim.ry(params[param_idx + 1], g)
                param_idx += 2
                
            # Entanglement strictly on 3x3 sub-grids
            for q1_local, q2_local in self.intra_patch_edges:
                q1_global = local_to_global[q1_local]
                q2_global = local_to_global[q2_local]
                sim.cx(q1_global, q2_global)

        # Evaluate complete 6x6 XY Hamiltonian
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
# 4. EXECUTION & VERIFICATION N-TIMES
# ==========================================
if __name__ == "__main__":
    N_RUNS = 5
    AVAILABLE_GPUS = [0, 1, 2, 3] 
    
    # 72 parameters for 4 patches
    np.random.seed(42)
    test_params = np.random.uniform(-np.pi, np.pi, 72)
    
    gpu_engine = DistributedVQEEngine(device_ids=AVAILABLE_GPUS)
    cpu_engine = MonolithicCPUEngine()
    
    print(f"Starting Verification: {N_RUNS} runs...")
    print("-" * 50)
    
    # Evaluate Multi-GPU
    start_time = time.time()
    gpu_results = []
    for i in range(N_RUNS):
        res = gpu_engine.run(test_params)
        gpu_results.append(res)
    gpu_time = time.time() - start_time
    
    print(f"Distributed GPU Executions complete in {gpu_time:.2f}s")
    print(f"Average Energy: {np.mean(gpu_results):.8f}\n")
    
    # Evaluate Monolithic CPU
    print("Allocating monolithic 36-qubit statevectors (Watch RAM usage)...")
    start_time = time.time()
    cpu_results = []
    for i in range(N_RUNS):
        res = cpu_engine.run(test_params)
        cpu_results.append(res)
    cpu_time = time.time() - start_time
    
    print(f"Monolithic CPU Executions complete in {cpu_time:.2f}s")
    print(f"Average Energy: {np.mean(cpu_results):.8f}\n")
    
    # Assert equivalence
    print("-" * 50)
    print("VERIFICATION RESULTS:")
    
    # Statevector exact probability should guarantee deterministic float equivalence 
    # to roughly 1e-7 due to slight non-associativity in fp-math across backends.
    diff = abs(np.mean(gpu_results) - np.mean(cpu_results))
    print(f"Absolute Energy Difference: {diff:.8e}")
    if diff < 1e-6:
        print("✅ SUCCESS: Distributed Marginal Evaluation perfectly matches Monolithic Global Evaluation.")
    else:
        print("❌ FAILED: State evaluation diverged.")
