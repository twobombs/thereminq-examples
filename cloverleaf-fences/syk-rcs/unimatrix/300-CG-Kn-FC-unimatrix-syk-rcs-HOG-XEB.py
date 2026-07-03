import os
import gc
import time
import signal
import sys
import json
import numpy as np
import multiprocessing as mp
from multiprocessing.connection import Connection, wait
from multiprocessing.synchronize import Semaphore
from collections import Counter
from typing import List, Tuple, Dict, Any

# ==========================================
# 0. HOLOGRAPHIC TOPOLOGY (BULK/BOUNDARY)
# ==========================================
def get_complete_intra_edges(num_qubits: int = 25) -> List[Tuple[int, int]]:
    """Fully connected intra-patch entanglement for scaled architecture"""
    edges = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            edges.append((i, j))
    return edges

def get_holographic_topology(
    num_patches: int = 12, 
    qubits_per_patch: int = 25, 
    boundary_size: int = 4
) -> Tuple[List[List[int]], List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    
    total_qubits = num_patches * qubits_per_patch
    patches = [[] for _ in range(num_patches)]
    fence_edges = []
    
    for idx in range(total_qubits):
        patch_idx = idx // qubits_per_patch
        patches[patch_idx].append(idx)

    for p1 in range(num_patches):
        for p2 in range(p1 + 1, num_patches):
            for b1 in range(boundary_size):
                for b2 in range(boundary_size):
                    fence_edges.append(((p1, b1), (p2, b2)))

    return patches, fence_edges

# ==========================================
# 1. ISOLATED PERSISTENT UNIVERSE (GPU WORKER)
# ==========================================
def persistent_universe_worker(
    device_id: int, 
    patch_idx: int, 
    num_qubits: int, 
    intra_edges: List[Tuple[int, int]], 
    boundary_qubits: List[int],
    cmd_pipe: Connection,
    ram_semaphore: Semaphore
) -> None:
    
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    
    sim = None
    try:
        if boundary_qubits:
            assert max(boundary_qubits) < num_qubits, f"Worker {patch_idx}: Boundary index {max(boundary_qubits)} out of bounds."

        os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(device_id)
        os.environ["QRACK_QPAGER_DEVICES"] = str(device_id)
        os.environ["QRACK_QUNITMULTI_DEVICES"] = str(device_id)

        from pyqrack import QrackSimulator
        
        sim = QrackSimulator(qubit_count=num_qubits)
        
        # PyQrack Pauli axis enum: X=1, Y=2, Z=3
        PX, PY, PZ = 1, 2, 3
        
        if hasattr(sim, 'r'):
            apply_rx = lambda theta, q: sim.r(PX, float(theta), q)
            apply_ry = lambda theta, q: sim.r(PY, float(theta), q)
            apply_rz = lambda theta, q: sim.r(PZ, float(theta), q)
        else:
            apply_rx = lambda theta, q: sim.mtrx([complex(np.cos(theta/2), 0), complex(0, -np.sin(theta/2)), 
                                                  complex(0, -np.sin(theta/2)), complex(np.cos(theta/2), 0)], q)
            apply_ry = lambda theta, q: sim.mtrx([complex(np.cos(theta/2), 0), complex(-np.sin(theta/2), 0), 
                                                  complex(np.sin(theta/2), 0), complex(np.cos(theta/2), 0)], q)
            apply_rz = lambda theta, q: sim.mtrx([complex(np.cos(-theta/2), np.sin(-theta/2)), 0j, 
                                                  0j, complex(np.cos(theta/2), np.sin(theta/2))], q)
            
        if hasattr(sim, 'cx'):
            apply_cx = lambda c, t: sim.cx(c, t)
        else:
            apply_cx = lambda c, t: sim.mcx([c], t)
            
        if hasattr(sim, 'h'):
            apply_h = lambda q: sim.h(q)
        else:
            apply_h = lambda q: sim.mtrx([complex(1/np.sqrt(2), 0), complex(1/np.sqrt(2), 0), 
                                          complex(1/np.sqrt(2), 0), complex(-1/np.sqrt(2), 0)], q)

        for q in range(num_qubits):
            apply_h(q)
            
        cmd_pipe.send({"status": "READY", "patch_idx": patch_idx})
        
        rotation_gates = (apply_rx, apply_ry, apply_rz)
        all_bits = list(range(num_qubits))

        while True:
            try:
                if cmd_pipe.poll(timeout=0.1):
                    cmd = cmd_pipe.recv()
                else:
                    continue
            except (EOFError, OSError, BrokenPipeError):
                break
                
            action = cmd.get("action")

            if action == "SHUTDOWN":
                break

            try:
                if action == "RCS_CHUNK":
                    seed = cmd.get("seed")
                    depth = cmd.get("depth", 1)
                    rng = np.random.default_rng(seed)

                    for _ in range(depth):
                        for q in range(num_qubits):
                            gate_idx = rng.integers(0, 3)
                            theta = rng.uniform(-np.pi, np.pi)
                            rotation_gates[gate_idx](theta, q)
                        
                        for q1, q2 in intra_edges:
                            apply_cx(q1, q2)
                    
                    cmd_pipe.send({"status": "CHUNK_COMPLETE", "patch_idx": patch_idx})

                elif action == "MEASURE_BOUNDARY_Z":
                    z_exp = {}
                    for q in boundary_qubits:
                        p_one = sim.prob(q)
                        z_exp[q] = 1.0 - 2.0 * p_one
                    cmd_pipe.send({"status": "Z_EXP_COMPUTED", "patch_idx": patch_idx, "data": z_exp})

                elif action == "APPLY_WORMHOLE_KICKS":
                    kicks = cmd.get("kicks", {})
                    for raw_q, theta in kicks.items():
                        q = int(raw_q)
                        if not (0 <= q < num_qubits):
                            raise IndexError(f"Kick qubit index {q} out of bounds")
                        apply_rz(theta, q)
                    cmd_pipe.send({"status": "KICKS_APPLIED", "patch_idx": patch_idx})

                elif action == "MEASURE_MAGNETIZATION":
                    total_z = sum((1.0 - 2.0 * sim.prob(q)) for q in range(num_qubits))
                    cmd_pipe.send({"status": "MAGNETIZATION_MEASURED", "patch_idx": patch_idx, "data": total_z})
                
                elif action == "SAMPLE_BITSTRINGS":
                    shots = cmd.get("shots", 10000)
                    with ram_semaphore:
                        try:
                            raw_shots = sim.measure_shots(all_bits, shots)
                            # Normalize keys to strings to prevent JSON serialization tuple crashes
                            shot_counts = {str(k): v for k, v in Counter(raw_shots).items()}
                            cmd_pipe.send({
                                "status": "BENCHMARKS_SAMPLED", 
                                "patch_idx": patch_idx, 
                                "data": {"shots": shots, "counts": shot_counts}
                            })
                        except Exception as e:
                            print(f"[WORKER {patch_idx} DEBUG] measure_shots() failed: {e}", file=sys.stderr)
                            cmd_pipe.send({
                                "status": "ERROR",
                                "patch_idx": patch_idx,
                                "msg": f"measure_shots() failed: {str(e)}"
                            })
                            
                        gc.collect() 

            except Exception as inner_e:
                try:
                    cmd_pipe.send({"status": "ERROR", "patch_idx": patch_idx, "msg": f"Failed during {action}: {str(inner_e)}"})
                except (EOFError, OSError, BrokenPipeError):
                    break
                continue
                
    except (EOFError, OSError, BrokenPipeError):
        pass
    except Exception as e:
        try:
            cmd_pipe.send({"status": "ERROR", "patch_idx": patch_idx, "msg": f"Worker {patch_idx} failed fatally: {str(e)}"})
        except (EOFError, OSError, BrokenPipeError):
            pass
    finally:
        if sim is not None:
            del sim
        gc.collect()

# ==========================================
# 2. WORMHOLE ORCHESTRATOR (THE BULK SPACE)
# ==========================================
class TraversableWormholeEngine:
    def __init__(self, device_ids: List[int], boundary_size: int = 4, qubits_per_patch: int = 25):
        self._is_shutdown = False
        self._mean_field_warned = False
        self._warned_at_step = 0
        
        self.device_ids = device_ids
        self.boundary_size = boundary_size
        self.qubits_per_patch = qubits_per_patch
        
        self.patches, self.fence_edges = get_holographic_topology(
            num_patches=12, 
            qubits_per_patch=self.qubits_per_patch, 
            boundary_size=self.boundary_size
        )
        
        self.intra_patch_edges = get_complete_intra_edges(num_qubits=self.qubits_per_patch)
        
        self.num_patches = len(self.patches)
        self.boundary_map = {i: {} for i in range(self.num_patches)}
        
        for (pA, qA), (pB, qB) in self.fence_edges:
            self.boundary_map[pA].setdefault(qA, []).append((pB, qB))
            self.boundary_map[pB].setdefault(qB, []).append((pA, qA))

        self.ctx = mp.get_context('spawn')
        self.workers = []
        self.pipes = []
        
        self.ram_semaphore = self.ctx.Semaphore(6)

        print(f"Initializing {self.num_patches} isolated GPU Universes ({self.qubits_per_patch} Qubits/Patch, Boundary={self.boundary_size})...")
        for p_idx in range(self.num_patches):
            
            assert len(self.patches[p_idx]) == self.qubits_per_patch, f"Patch {p_idx} invalid size."
            assert len(self.boundary_map[p_idx]) > 0, f"Patch {p_idx} has no boundary qubits -- wormhole kicks will be vacuous."
            assert all(q < len(self.patches[p_idx]) for q in self.boundary_map[p_idx].keys()), f"Patch {p_idx}: boundary qubit index exceeds patch size."
            
            parent_conn, child_conn = self.ctx.Pipe()
            boundary_qubits = list(self.boundary_map[p_idx].keys())
            
            p = self.ctx.Process(
                target=persistent_universe_worker,
                args=(
                    self.device_ids[p_idx % len(self.device_ids)], 
                    p_idx, 
                    len(self.patches[p_idx]), 
                    self.intra_patch_edges, 
                    boundary_qubits, 
                    child_conn,
                    self.ram_semaphore
                )
            )
            p.start()
            
            child_conn.close() 
            self.workers.append(p)
            self.pipes.append(parent_conn)
            
        for i, pipe in enumerate(self.pipes):
            if pipe.poll(timeout=45.0): 
                msg = pipe.recv()
                if msg.get("status") != "READY":
                    self.shutdown()
                    raise RuntimeError(f"Worker {i} initialized with bad state: {msg.get('msg', '')}")
                if msg.get("patch_idx") != i:
                    self.shutdown()
                    raise RuntimeError(f"Worker {i} returned mismatched patch_idx: expected {i}, got {msg.get('patch_idx')}")
            else:
                self.shutdown()
                raise TimeoutError(f"Worker {i} failed to initialize within 45 seconds.")

    def sync_broadcast(self, action: str, kwargs_list: List[Dict] = None, timeout_secs: float = 300.0, expected_status: str = None) -> Dict[int, Any]:
        if kwargs_list is None:
            kwargs_list = [{}] * self.num_patches
            
        for i, pipe in enumerate(self.pipes):
            payload = {"action": action}
            payload.update(kwargs_list[i])
            try:
                pipe.send(payload)
            except (BrokenPipeError, OSError) as e:
                self.shutdown()
                raise RuntimeError(f"Worker {i} pipe broken during send: {e}")
            
        results = {}
        pending = list(enumerate(self.pipes)) 
        deadline = time.monotonic() + timeout_secs
        
        while pending:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                self.shutdown()
                raise TimeoutError(f"Workers {[i for i, _ in pending]} timed out during {action}.")
                
            active_pipes = [p for _, p in pending]
            ready_pipes = wait(active_pipes, timeout=timeout)
            
            if not ready_pipes:
                continue
                
            still_pending = []
            for idx, pipe in pending:
                if pipe in ready_pipes:
                    try:
                        res = pipe.recv()
                        if res.get("status") == "ERROR":
                            self.shutdown()
                            raise RuntimeError(f"Worker {idx} reported an error: {res.get('msg')}")
                            
                        # Restored boundary checking loop
                        if res.get("patch_idx") != idx:
                            self.shutdown()
                            raise RuntimeError(f"Worker {idx} returned mismatched patch_idx during {action}: expected {idx}, got {res.get('patch_idx')}")
                            
                        if expected_status and res.get("status") != expected_status:
                            self.shutdown()
                            raise RuntimeError(f"Worker {idx} returned unexpected status '{res.get('status')}'. Expected '{expected_status}'.")
                            
                        results[idx] = res
                    except (EOFError, OSError, BrokenPipeError) as e:
                        self.shutdown()
                        raise RuntimeError(f"Worker {idx} pipe failed during recv: {e}")
                else:
                    still_pending.append((idx, pipe))
            
            pending = still_pending
            
        return results

    def evolve(self, total_time_steps: int, depth_per_step: int, coupling_strength: float, benchmark_interval: int = None):
        if benchmark_interval is None:
            benchmark_interval = total_time_steps

        print(f"\nStarting SYK Multi-Boundary Unimatrix Evolution...")
        print(f"Total Steps: {total_time_steps} | RCS Depth/Step: {depth_per_step} | g: {coupling_strength} | Benchmark Interval: {benchmark_interval}\n")
        
        main_rng = np.random.default_rng()

        for t in range(total_time_steps):
            seeds = main_rng.integers(0, 2**32, size=self.num_patches)
            self.sync_broadcast("RCS_CHUNK", [{"seed": int(seeds[i]), "depth": depth_per_step} for i in range(self.num_patches)], expected_status="CHUNK_COMPLETE")
            
            z_results = self.sync_broadcast("MEASURE_BOUNDARY_Z", expected_status="Z_EXP_COMPUTED")
            patch_z_exp = {p_idx: res["data"] for p_idx, res in z_results.items()}

            global_z_pool = sum(z for z_dict in patch_z_exp.values() for z in z_dict.values())
            total_boundary_qubits = sum(len(z_dict) for z_dict in patch_z_exp.values())
            bulk_mean_field = global_z_pool / total_boundary_qubits if total_boundary_qubits > 0 else 0.0
            
            if abs(bulk_mean_field) < 1e-4:
                if not self._mean_field_warned:
                    print(f"         +-- [NOTE] Bulk Mean Field thermalized near zero at Step {t:03d}. Kicks suppressed.")
                    self._mean_field_warned = True
                    self._warned_at_step = t
            elif abs(bulk_mean_field) > 1e-3 and self._mean_field_warned:
                if t - self._warned_at_step > 10:
                    self._mean_field_warned = False

            kick_payloads = [{"kicks": {}} for _ in range(self.num_patches)]
            
            for pA in range(self.num_patches):
                for qA in self.boundary_map[pA].keys():
                    unified_kick = 2.0 * coupling_strength * bulk_mean_field
                    kick_payloads[pA]["kicks"][qA] = unified_kick

            self.sync_broadcast("APPLY_WORMHOLE_KICKS", kick_payloads, expected_status="KICKS_APPLIED")

            mag_res = self.sync_broadcast("MEASURE_MAGNETIZATION", expected_status="MAGNETIZATION_MEASURED")
            mag_sum = sum(res["data"] for res in mag_res.values())
            
            cross_corr_connected = 0.0
            edge_count = 0
            seen = set()
            
            for pA in range(self.num_patches):
                for qA, neighbors in self.boundary_map[pA].items():
                    for (pB, qB) in neighbors:
                        endpointA = (pA, qA)
                        endpointB = (pB, qB)
                        key = (min(endpointA, endpointB), max(endpointA, endpointB))
                        
                        if key not in seen:
                            seen.add(key)
                            mean_zA = patch_z_exp[pA][qA]
                            mean_zB = patch_z_exp[pB][qB]
                            cross_corr_connected += (mean_zA * mean_zB) - (bulk_mean_field ** 2)
                            edge_count += 1
                        
            avg_corr_c = cross_corr_connected / edge_count if edge_count > 0 else 0.0

            print(f"Step {t:03d} | Bulk Mag: {mag_sum:+.4f} | Bulk Field <Z>: {bulk_mean_field:+.4f} | Classical MF Variance: {avg_corr_c:+.4f} | Kicks: {sum(len(k['kicks']) for k in kick_payloads)}")
            
            run_benchmarks = (t > 0 and t % benchmark_interval == 0) or (t == total_time_steps - 1)
            
            if run_benchmarks:
                shots = 10000
                bench_res = self.sync_broadcast("SAMPLE_BITSTRINGS", [{"shots": shots} for _ in range(self.num_patches)], timeout_secs=600.0, expected_status="BENCHMARKS_SAMPLED")
                
                total_unique_states = 0
                telemetry_payload = {
                    "step": t,
                    "depth_per_step": depth_per_step,
                    "coupling_strength": coupling_strength,
                    "shots_per_patch": shots,
                    "patches": {}
                }
                
                for p_idx, res in bench_res.items():
                    counts = res["data"]["counts"]
                    total_unique_states += len(counts)
                    # Explicit string conversion for JSON compatibility
                    telemetry_payload["patches"][str(p_idx)] = counts
                
                filename = f"wormhole_telemetry_step_{t}.json"
                with open(filename, 'w') as f:
                    json.dump(telemetry_payload, f, indent=2)
                    
                print(f"         +-- Sycamore Sampling -> {shots:,} shots per patch executed natively on GPU.")
                print(f"         +-- Telemetry Export  -> Saved {total_unique_states:,} unique basis states to {filename}")
                print(f"         +-- Next Step         -> Route JSON through Quantum Lemonade for ideal amplitude contraction to compute XEB/HOG.")

    def shutdown(self):
        if self._is_shutdown:
            return
        self._is_shutdown = True
        
        print("\nCollapsing the Wormhole (Shutting down GPU workers)...")
        for pipe in getattr(self, 'pipes', []):
            try:
                if not pipe.closed:
                    while pipe.poll(0.0):
                        try:
                            pipe.recv()
                        except (EOFError, OSError):
                            break
                    pipe.send({"action": "SHUTDOWN"})
            except (EOFError, OSError, BrokenPipeError):
                pass
                
        for p in getattr(self, 'workers', []):
            p.join(timeout=5)
            if p.is_alive(): 
                p.terminate()
                p.join(timeout=2)
                
        for pipe in getattr(self, 'pipes', []):
            try:
                if not pipe.closed:
                    pipe.close()
            except (EOFError, OSError, BrokenPipeError):
                pass

# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    gpu_env = os.environ.get("WORMHOLE_GPUS", "0,1") 
    base_gpus = [int(g.strip()) for g in gpu_env.split(',')]
    
    num_patches = 12
    AVAILABLE_GPUS = [base_gpus[i % len(base_gpus)] for i in range(num_patches)]
    
    wormhole_engine = TraversableWormholeEngine(
        device_ids=AVAILABLE_GPUS, 
        boundary_size=4, 
        qubits_per_patch=25
    )

    try:
        wormhole_engine.evolve(
            total_time_steps=10, 
            depth_per_step=1, 
            coupling_strength=1.5,
            benchmark_interval=None # Defaults to final step only
        )
    except KeyboardInterrupt:
        print("\n[SIGINT] Manual interrupt received.")
    except Exception as e:
        print(f"\n[FATAL ERROR] The orchestrator caught an exception:")
        print(str(e))
        raise
    finally:
        wormhole_engine.shutdown()
        sys.exit(0)
