import os
import gc
import numpy as np
from scipy.optimize import minimize
from pyqrack import QrackSimulator

# ==========================================
# 1. DYNAMIC GPU ALLOCATOR
# ==========================================
class GPUPoolAllocator:
    def __init__(self, available_device_ids, memory_limit_mb=8176, bytes_per_amplitude=8):
        """
        Manages GPU selection. 
        Defaults to 8176 MB limit for standard 8GB VRAM dies.
        Bytes per amplitude defaults to 8 for single precision (QRACK_FPPOW=5).
        Set to 16 if using double precision.
        """
        self.device_ids = available_device_ids
        self.memory_limit_mb = memory_limit_mb
        self.bytes_per_amp = bytes_per_amplitude
        self.device_usage_mb = {dev: 0.0 for dev in self.device_ids}
        
    def _get_sv_size_mb(self, num_qubits):
        return (2**num_qubits * self.bytes_per_amp) / (1024 * 1024)

    def allocate(self, num_qubits):
        """Returns the assigned device_id based on available tracked VRAM."""
        req_mb = self._get_sv_size_mb(num_qubits)
        
        for dev_id in self.device_ids:
            if self.device_usage_mb[dev_id] + req_mb <= self.memory_limit_mb:
                self.device_usage_mb[dev_id] += req_mb
                return dev_id
                
        raise MemoryError(f"All GPUs have exceeded memory limits. Need {req_mb:.2f} MB.")

# ==========================================
# 2. TOPOLOGY & DISTRIBUTED MAPPING
# ==========================================
def get_3x3_edges():
    """Returns nearest-neighbor horizontal and vertical edges for a 3x3 grid."""
    edges = []
    for r in range(3):
        for c in range(3):
            idx = r * 3 + c
            if c < 2: edges.append((idx, idx + 1)) # Horizontal
            if r < 2: edges.append((idx, idx + 3)) # Vertical
    return edges

def get_topology():
    """Returns intra-patch clusters and inter-patch (fence) edges for a 6x6."""
    patches = [[] for _ in range(4)]
    fence_edges = []
    
    # 1. Map 3x3 patches
    for r in range(6):
        for c in range(6):
            idx = r * 6 + c
            patch_idx = (0 if r < 3 else 2) + (0 if c < 3 else 1)
            local_idx = len(patches[patch_idx])
            patches[patch_idx].append((idx, local_idx))

    # Global to (patch_idx, local_idx) mapping
    global_to_local = {}
    for p_idx, patch in enumerate(patches):
        for global_idx, local_idx in patch:
            global_to_local[global_idx] = (p_idx, local_idx)

    # 2. Define the fence (edges crossing the 3x3 boundaries)
    for r in range(6):
        for c in range(6):
            global_1 = r * 6 + c
            # Horizontal fence
            if c == 2:
                global_2 = r * 6 + (c + 1)
                fence_edges.append((global_to_local[global_1], global_to_local[global_2]))
            # Vertical fence
            if r == 2:
                global_2 = (r + 1) * 6 + c
                fence_edges.append((global_to_local[global_1], global_to_local[global_2]))

    return patches, fence_edges, global_to_local

# ==========================================
# 3. DISTRIBUTED EXECUTION ENGINE
# ==========================================
class DistributedVQEEngine:
    def __init__(self, device_ids):
        # Configure bytes based on build precision environment variable.
        # Defaults to 8 bytes for FP32.
        precision = int(os.environ.get("QRACK_FPPOW", 5))
        bytes_per_amp = 8 if precision == 5 else 16 

        self.allocator = GPUPoolAllocator(device_ids, bytes_per_amplitude=bytes_per_amp)
        self.patches, self.fence_edges, self.mapping = get_topology()
        self.num_qubits_per_patch = 9
        self.intra_patch_edges = get_3x3_edges()
        
    def _create_patch_simulators(self):
        """Initializes distributed simulators via environment variable routing."""
        sims = []
        # Stash original env var to restore later
        original_dev = os.environ.get("QRACK_DEFAULT_DEVICE", "0")
        
        for i in range(4):
            dev_id = self.allocator.allocate(self.num_qubits_per_patch)
            # Route next context initialization to specific device
            os.environ["QRACK_DEFAULT_DEVICE"] = str(dev_id)
            sim = QrackSimulator(qubitCount=self.num_qubits_per_patch)
            sims.append(sim)
            
        os.environ["QRACK_DEFAULT_DEVICE"] = original_dev
        return sims

    def apply_local_ansatz(self, sims, params):
        """Applies depth-heavy 2D grid operations locally."""
        param_idx = 0
        for sim in sims:
            # Local rotations
            for q in range(self.num_qubits_per_patch):
                sim.rx(params[param_idx], q)
                sim.ry(params[param_idx + 1], q)
                param_idx += 2
                
            # Local intra-patch entanglement matching the 3x3 physical grid
            for q1, q2 in self.intra_patch_edges:
                sim.cx(q1, q2)
                
        return param_idx

    def get_local_expectation(self, sim, q, basis='X'):
        """Computes exact <X> or <Y> for a single qubit."""
        sim_clone = QrackSimulator(cloneSid=sim.sid)
        if basis == 'X':
            sim_clone.h(q)
        elif basis == 'Y':
            sim_clone.rz(-np.pi/2, q)
            sim_clone.h(q)
            
        p_odd = sim_clone.prob(q)
        del sim_clone
        return (1.0 - p_odd) - p_odd

    def get_intra_patch_xy(self, sim, q1, q2):
        """Computes <X_i X_j> + <Y_i Y_j> locally within one GPU."""
        energy = 0.0
        for basis in ['X', 'Y']:
            sim_clone = QrackSimulator(cloneSid=sim.sid)
            if basis == 'X':
                sim_clone.h(q1)
                sim_clone.h(q2)
            else:
                sim_clone.rz(-np.pi/2, q1)
                sim_clone.rz(-np.pi/2, q2)
                sim_clone.h(q1)
                sim_clone.h(q2)
                
            sim_clone.cx(q1, q2)
            p_odd = sim_clone.prob(q2)
            energy += (1.0 - p_odd) - p_odd
            del sim_clone
            
        return energy

    def evaluate_cost(self, params):
        """Evaluates the XY Hamiltonian across the distributed architecture."""
        self.allocator.device_usage_mb = {dev: 0.0 for dev in self.allocator.device_ids}
        sims = self._create_patch_simulators()
        self.apply_local_ansatz(sims, params)
        
        energy = 0.0
        
        # 4. Evaluate Intra-Patch Energy (Strictly on 3x3 lattice edges)
        for p_idx in range(4):
            sim = sims[p_idx]
            for q1, q2 in self.intra_patch_edges:
                energy += self.get_intra_patch_xy(sim, q1, q2)
                
        # 5. Evaluate Inter-Patch Energy (The Fence)
        for (pA, qA), (pB, qB) in self.fence_edges:
            simA = sims[pA]
            simB = sims[pB]
            
            x_a = self.get_local_expectation(simA, qA, 'X')
            x_b = self.get_local_expectation(simB, qB, 'X')
            y_a = self.get_local_expectation(simA, qA, 'Y')
            y_b = self.get_local_expectation(simB, qB, 'Y')
            
            energy += (x_a * x_b) + (y_a * y_b)
            
        print(f"Distributed Current Energy: {energy:.6f}")
        
        # 6. Explicitly flush simulators to reclaim GPU VRAM 
        for sim in sims:
            del sim
        gc.collect()
        
        return energy

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    # Ensure memory usage limits and device pooling is accurate
    AVAILABLE_GPUS = [0, 1, 2, 3] 
    engine = DistributedVQEEngine(device_ids=AVAILABLE_GPUS)
    
    # 4 patches * 9 qubits * 2 params (Rx, Ry) = 72 params
    num_params = 72 
    np.random.seed(42)
    initial_params = np.random.uniform(-np.pi, np.pi, num_params)
    
    print("Starting Multi-GPU Distributed VQE Allocation...")
    
    result = minimize(
        engine.evaluate_cost, 
        initial_params, 
        method='COBYLA', 
        options={'maxiter': 50, 'disp': True}
    )
    
    print("\nOptimization Complete.")
    print(f"Final Optimized Energy: {result.fun:.6f}")
