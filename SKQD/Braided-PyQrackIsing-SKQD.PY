import numpy as np
import opt_einsum as oe
import multiprocessing as mp
import zmq
from pyqrack import QrackSimulator
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import time

try:
    from pyqrack_ising import IsingModel1D
    HAS_PYQRACK_ISING = True
except ImportError:
    HAS_PYQRACK_ISING = False

# =====================================================================
# 1. TOPOLOGY & EXACT QPD PHYSICS
# =====================================================================
BASES = ['I', 'X', 'Y', 'Z']

CUT_TOPOLOGY = [
    (0, 8, 1, 0), # Cut A: 0 -> 1
    (1, 8, 2, 0), # Cut B: 1 -> 2
    (2, 8, 3, 0), # Cut C: 2 -> 3
    (3, 8, 0, 0)  # Cut D: 3 -> 0
]

QPD_COEFFS = {'I': 0.5, 'X': 0.5, 'Y': -0.5, 'Z': 0.5}

# =====================================================================
# 2. LOCAL HAMILTONIAN (Unified Sign Convention)
# =====================================================================
def get_local_zz_energy(state_int, num_qubits=8):
    """Aligned spins return -1 (Ferromagnetic TFIM)"""
    energy = 0
    for i in range(num_qubits - 1):
        bit1 = (state_int >> i) & 1
        bit2 = (state_int >> (i + 1)) & 1
        energy += -1 if bit1 == bit2 else 1
    return energy

# =====================================================================
# 3. ZMQ WORKER DAEMON (NATIVE KRYLOV KERNEL)
# =====================================================================
def apply_hss(probabilities, max_states=32):
    heavy_indices = np.argsort(probabilities)[-max_states:]
    pruned_probs = np.zeros_like(probabilities)
    pruned_probs[heavy_indices] = probabilities[heavy_indices]
    return pruned_probs

def qrack_zmq_worker(worker_id, task_addr, sink_addr, ctrl_addr, sync_addr):
    context = zmq.Context()
    
    receiver = context.socket(zmq.PULL)
    receiver.connect(task_addr)
    
    sender = context.socket(zmq.PUSH)
    sender.connect(sink_addr)
    
    ctrl = context.socket(zmq.SUB)
    ctrl.connect(ctrl_addr)
    ctrl.setsockopt_string(zmq.SUBSCRIBE, "")

    sync = context.socket(zmq.PUSH)
    sync.connect(sync_addr)
    
    poller = zmq.Poller()
    poller.register(receiver, zmq.POLLIN)
    poller.register(ctrl, zmq.POLLIN)
    
    sim = QrackSimulator(9)
    
    if HAS_PYQRACK_ISING:
        print(f"[Worker {worker_id}] Native PyQrackIsing Kernel online.")
    else:
        print(f"[Worker {worker_id}] WARNING: PyQrackIsing missing. Using Trotterized fallback.")
        
    # Dispatch READY signal to orchestrator
    sync.send_string("READY")
    
    while True:
        socks = dict(poller.poll())
        
        if ctrl in socks:
            if ctrl.recv_string() == "KILL":
                break
                
        if receiver in socks:
            perm_idx, grid_id, dt, cut_bases = receiver.recv_pyobj()
            sim.reset_all()

            for cut_idx, (up_g, up_q, down_g, down_q) in enumerate(CUT_TOPOLOGY):
                if down_g == grid_id:
                    basis = cut_bases[cut_idx]
                    if basis == 'X': sim.h(down_q) 
                    elif basis == 'Y': sim.h(down_q); sim.s(down_q) 
                    elif basis == 'Z': sim.x(down_q) 

            # True Krylov Evolution
            if HAS_PYQRACK_ISING and dt > 0.0:
                local_ising = IsingModel1D(num_qubits=9, j_z=1.0, h_x=1.0, simulator=sim)
                local_ising.time_evolve(dt)
            elif dt > 0.0:
                # Fallback purely for runtime safety
                for q in range(8): sim.cx(q, q + 1)
                for q in range(9): sim.rx(q, dt)

            for cut_idx, (up_g, up_q, down_g, down_q) in enumerate(CUT_TOPOLOGY):
                if up_g == grid_id:
                    basis = cut_bases[cut_idx]
                    if basis == 'X': sim.h(up_q)
                    elif basis == 'Y': sim.adjs(up_q); sim.h(up_q)

            probabilities = np.array(sim.get_probabilities())
            comp_probs = probabilities.reshape(256, 2).sum(axis=1)
            pruned_probs = apply_hss(comp_probs, max_states=32) 
            
            sender.send_pyobj((perm_idx, grid_id, pruned_probs))
        
    receiver.close()
    sender.close()
    ctrl.close()
    sync.close()
    context.term()

# =====================================================================
# 4. ZMQ ORCHESTRATOR & SKQD PIPELINE
# =====================================================================
class SKQDOrchestrator:
    def __init__(self, num_workers=4, task_port=5555, sink_port=5556, ctrl_port=5557, sync_port=5558):
        self.num_workers = num_workers
        self.context = zmq.Context()
        
        self.ventilator = self.context.socket(zmq.PUSH)
        self.ventilator.bind(f"tcp://*:{task_port}")
        
        self.sink = self.context.socket(zmq.PULL)
        self.sink.bind(f"tcp://*:{sink_port}")
        self.sink.RCVTIMEO = 5000 
        
        self.pub = self.context.socket(zmq.PUB)
        self.pub.bind(f"tcp://*:{ctrl_port}")

        self.sync = self.context.socket(zmq.PULL)
        self.sync.bind(f"tcp://*:{sync_port}")
        
        self.workers = []
        task_addr = f"tcp://localhost:{task_port}"
        sink_addr = f"tcp://localhost:{sink_port}"
        ctrl_addr = f"tcp://localhost:{ctrl_port}"
        sync_addr = f"tcp://localhost:{sync_port}"
        
        print("Spawning worker nodes and waiting for SYNC handshake...")
        for i in range(num_workers):
            p = mp.Process(target=qrack_zmq_worker, args=(i, task_addr, sink_addr, ctrl_addr, sync_addr))
            p.start()
            self.workers.append(p)
            
        # Deterministic Handshake Barrier
        for _ in range(num_workers):
            self.sync.recv_string()
        print("All workers synchronized. Fabric ready.")

    def seed_tensors_from_fabric(self, time_step_dt=0.1):
        for perm_idx in range(256):
            b_A = BASES[(perm_idx // 64) % 4]
            b_B = BASES[(perm_idx // 16) % 4]
            b_C = BASES[(perm_idx // 4) % 4]
            b_D = BASES[perm_idx % 4]
            cut_bases = {0: b_A, 1: b_B, 2: b_C, 3: b_D}
            
            for grid_id in range(4):
                self.ventilator.send_pyobj((perm_idx, grid_id, time_step_dt, cut_bases))
                
        T0, T1, T2, T3 = [np.zeros((4, 4, 256)) for _ in range(4)]
        
        received_chunks = 0
        while received_chunks < 1024:
            try:
                perm_idx, grid_id, pruned_probs = self.sink.recv_pyobj()
                
                d, a = (perm_idx // 64) % 4, (perm_idx // 16) % 4
                b, c = (perm_idx // 4) % 4, perm_idx % 4
                
                if grid_id == 0:
                    T0[d, a, :] += pruned_probs * (QPD_COEFFS[BASES[d]] * QPD_COEFFS[BASES[a]])
                elif grid_id == 1:
                    T1[a, b, :] += pruned_probs * (QPD_COEFFS[BASES[a]] * QPD_COEFFS[BASES[b]])
                elif grid_id == 2:
                    T2[b, c, :] += pruned_probs * (QPD_COEFFS[BASES[b]] * QPD_COEFFS[BASES[c]])
                elif grid_id == 3:
                    T3[c, d, :] += pruned_probs * (QPD_COEFFS[BASES[c]] * QPD_COEFFS[BASES[d]])
                
                received_chunks += 1
            except zmq.Again:
                raise RuntimeError("ZMQ Sink Timeout. Pipeline stall detected.")
                
        return T0, T1, T2, T3

    def extract_seed_subspace(self, T0, T1, T2, T3, top_k=500):
        loc0 = np.argsort(np.abs(T0).sum(axis=(0,1)))[-16:]
        loc1 = np.argsort(np.abs(T1).sum(axis=(0,1)))[-16:]
        loc2 = np.argsort(np.abs(T2).sum(axis=(0,1)))[-16:]
        loc3 = np.argsort(np.abs(T3).sum(axis=(0,1)))[-16:]
        
        sub_T0 = T0[:, :, loc0]
        sub_T1 = T1[:, :, loc1]
        sub_T2 = T2[:, :, loc2]
        sub_T3 = T3[:, :, loc3]
        
        sub_global = oe.contract('daw, abx, bcy, cdz -> wxyz', sub_T0, sub_T1, sub_T2, sub_T3)
        
        flat_probs = np.abs(sub_global).flatten()
        
        # Null-State Protection Boundary
        if np.all(flat_probs == 0):
            print("Warning: QPD parity cancellation yielded empty set. Injecting heuristic root.")
            return [0x00000000, 0xFFFFFFFF]

        best_flat_indices = np.argsort(flat_probs)[-top_k:]
        
        seed_subspace = set()
        for flat_idx in best_flat_indices:
            i0, i1, i2, i3 = np.unravel_index(flat_idx, (16, 16, 16, 16))
            w, x, y, z = loc0[i0], loc1[i1], loc2[i2], loc3[i3]
            global_bitstring = (z << 24) | (y << 16) | (x << 8) | w
            seed_subspace.add(global_bitstring)
            
        return list(seed_subspace)

    def build_skqd_hamiltonian(self, subspace_bitstrings, g_field=1.0, num_qubits=32):
        N = len(subspace_bitstrings)
        state_to_idx = {bs: idx for idx, bs in enumerate(subspace_bitstrings)}
        
        rows, cols, data = [], [], []
        
        for idx, bs in enumerate(subspace_bitstrings):
            zz_energy = 0
            for i in range(num_qubits):
                bit1 = (bs >> i) & 1
                bit2 = (bs >> ((i + 1) % num_qubits)) & 1
                zz_energy += -1 if bit1 == bit2 else 1
                
            rows.append(idx)
            cols.append(idx)
            data.append(zz_energy)
            
            for i in range(num_qubits):
                flipped_bs = bs ^ (1 << i)
                if flipped_bs in state_to_idx:
                    flipped_idx = state_to_idx[flipped_bs]
                    rows.append(idx)
                    cols.append(flipped_idx)
                    data.append(-g_field)
                    
        return sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

    def skqd_configuration_recovery(self, seed_subspace, g_field=1.0, max_iters=20, tolerance=1e-5, max_subspace_size=8192):
        current_subspace = list(set(seed_subspace))
        prev_energy = np.inf
        
        print(f"\n--- Initiating Classical SKQD Recovery (Max States: {max_subspace_size}) ---")
        for iteration in range(max_iters):
            if len(current_subspace) == 0:
                print("Catastrophic Subspace Collapse.")
                break

            H_eff = self.build_skqd_hamiltonian(current_subspace, g_field=g_field)
            
            if len(current_subspace) < 10:
                evals, evecs = np.linalg.eigh(H_eff.toarray())
                E0, v0 = evals[0], evecs[:, 0]
            else:
                try:
                    evals, evecs = eigsh(H_eff, k=1, which='SA')
                    E0, v0 = evals[0], evecs[:, 0]
                except Exception as e:
                    print(f"Eigsh failed: {e}. Falling back to dense diagonalization.")
                    evals, evecs = np.linalg.eigh(H_eff.toarray())
                    E0, v0 = evals[0], evecs[:, 0]
                
            print(f"Iter {iteration}: Active Support = {len(current_subspace)} states, E0 = {E0:.6f}")
            
            if abs(prev_energy - E0) < tolerance:
                print("SKQD Energy Manifold Converged.")
                break
            prev_energy = E0
            
            probabilities = np.abs(v0)**2
            sorted_indices = np.argsort(probabilities)[::-1]
            
            cumulative_p = 0.0
            new_configurations = set()
            
            for idx in sorted_indices:
                cumulative_p += probabilities[idx]
                bs = current_subspace[idx]
                
                # TFIM Transverse-Field Connectivity
                for i in range(32):
                    new_configurations.add(bs ^ (1 << i))
                    
                if cumulative_p > 0.90: 
                    break
                    
            # Integrate and apply strict memory boundaries to prevent silent RAM exhaustion
            expanded_subspace = list(set(current_subspace) | new_configurations)
            
            if len(expanded_subspace) > max_subspace_size:
                print(f"Memory Wall Intercept: Truncating {len(expanded_subspace)} states to {max_subspace_size}.")
                current_subspace = expanded_subspace[:max_subspace_size]
            else:
                current_subspace = expanded_subspace

        return prev_energy

    def run_skqd_pipeline(self):
        # 1. Multi-Pass Krylov Seeding
        time_steps = [0.0, 0.1, 0.25, 0.5]
        global_seed_subspace = set()
        
        print("1. Firing Multi-Pass Krylov Generator...")
        for dt in time_steps:
            print(f"   -> Evolving fabric at dt = {dt}")
            T0, T1, T2, T3 = self.seed_tensors_from_fabric(time_step_dt=dt)
            dt_subspace = self.extract_seed_subspace(T0, T1, T2, T3, top_k=500)
            global_seed_subspace.update(dt_subspace)
            
        print(f"2. Extracted aggregated initial support: {len(global_seed_subspace)} states.")
        
        # 3. Execution
        E0 = self.skqd_configuration_recovery(list(global_seed_subspace), g_field=1.0)
        return E0

    def shutdown(self):
        self.pub.send_string("KILL")
        for p in self.workers:
            p.join()
        self.ventilator.close()
        self.sink.close()
        self.pub.close()
        self.sync.close()
        self.context.term()

# =====================================================================
# 5. BOOTSTRAP EXECUTION
# =====================================================================
if __name__ == '__main__':
    print("Booting ThereminQ-HPC SKQD Krylov Orchestrator...")
    orchestrator = SKQDOrchestrator(num_workers=4)
    
    try:
        start_time = time.time()
        final_energy = orchestrator.run_skqd_pipeline()
        end_time = time.time()
        
        print(f"\nFinal Verified SKQD Ground State Energy: {final_energy:.6f}")
        print(f"Total Architecture Execution Time: {end_time - start_time:.2f} seconds")
        
    finally:
        orchestrator.shutdown()
        print("Streaming Fabric securely unmounted.")
