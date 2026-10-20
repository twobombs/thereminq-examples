import math
import random
import sys
import numpy as np
import os

# Quantum Libraries
from pyqrack import QrackSimulator
from qiskit import QuantumCircuit

# Tensor/Visualization Libraries
try:
    import torch
    import matplotlib.pyplot as plt
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[!] Warning: 'torch' or 'matplotlib' not found. GPU tensor creation will be skipped.")

def bench_patch_xeb_tensor(width, depth, shots=5000):
    """
    Generates a Patch Circuit, runs Experiment vs Ideal, and returns data for Tensor creation.
    """
    
    # ----------------------------------------------------------------
    # 1. Configuration & Split Logic
    # ----------------------------------------------------------------
    mid = width // 2
    mask_lower = (1 << mid) - 1 
    
    print(f"[-] Configuration: {width} Qubits, Depth {depth}")
    print(f"[-] Patch Split: Left ({mid} qubits) / Right ({width - mid} qubits)")

    # ----------------------------------------------------------------
    # 2. Circuit Generation (Full & Split)
    # ----------------------------------------------------------------
    full_rcs = QuantumCircuit(width)
    left_rcs = QuantumCircuit(mid)
    right_rcs = QuantumCircuit(width - mid)

    for d in range(depth):
        # Single Qubit Gates
        for i in range(width):
            angle = random.uniform(0, 2 * math.pi)
            
            full_rcs.h(i)
            full_rcs.rz(angle, i)
            
            if i < mid:
                left_rcs.h(i)
                left_rcs.rz(angle, i)
            else:
                right_rcs.h(i - mid)
                right_rcs.rz(angle, i - mid)

        # 2-Qubit Couplers (Patch Logic)
        all_bits = list(range(width))
        random.shuffle(all_bits)
        
        while len(all_bits) > 1:
            u = all_bits.pop()
            v = all_bits.pop()
            
            u_side = u < mid
            v_side = v < mid
            
            # CRITICAL: Only apply gate if both qubits are on the SAME side
            if u_side == v_side:
                full_rcs.cx(u, v)
                if u_side:
                    left_rcs.cx(u, v)
                else:
                    right_rcs.cx(u - mid, v - mid)
            else:
                pass # Elided gate (crosses boundary)

    # ----------------------------------------------------------------
    # 3. The 'Experiment' (Full Grid Simulation)
    # ----------------------------------------------------------------
    print("[-] Running 'Experiment' (Sampling from Full Grid)...")
    sim_exp = QrackSimulator(width)
    sim_exp.run_qiskit_circuit(full_rcs)
    measured_shots = sim_exp.measure_shots(list(range(width)), shots)
    
    # ----------------------------------------------------------------
    # 4. The 'Verification' (Split Grid Simulation)
    # ----------------------------------------------------------------
    print(f"[-] Running Ideal Left Patch ({mid} qubits)...")
    sim_left = QrackSimulator(mid)
    sim_left.run_qiskit_circuit(left_rcs)
    # Get probability vector for Left Patch
    probs_left = np.array(sim_left.prob_all(list(range(mid))))
    
    print(f"[-] Running Ideal Right Patch ({width - mid} qubits)...")
    sim_right = QrackSimulator(width - mid)
    sim_right.run_qiskit_circuit(right_rcs)
    # Get probability vector for Right Patch
    probs_right = np.array(sim_right.prob_all(list(range(width - mid))))

    # ----------------------------------------------------------------
    # 5. Linear XEB Calculation
    # ----------------------------------------------------------------
    print("[-] Calculating Linear XEB...")
    sum_probs = 0.0
    
    for k in measured_shots:
        k_left = k & mask_lower
        k_right = k >> mid
        p_total = probs_left[k_left] * probs_right[k_right]
        sum_probs += p_total

    mean_prob = sum_probs / shots
    n_pow = 2 ** width
    xeb = (n_pow * mean_prob) - 1

    return {
        "xeb": xeb,
        "tensor_left": probs_left,
        "tensor_right": probs_right,
        "width": width
    }

if __name__ == "__main__":
    # 1. Parse Arguments
    w = 30
    d = 10
    if len(sys.argv) >= 3:
        w = int(sys.argv[1])
        d = int(sys.argv[2])

    try:
        # 2. Run Benchmark
        results = bench_patch_xeb_tensor(w, d)
        
        print("\n" + "="*60)
        print(f"  PATCH CIRCUIT RESULTS (Width={w}, Depth={d})")
        print("="*60)
        print(f"  Linear XEB Score : {results['xeb']:.5f}")
        print("-" * 60)
        
        # 3. GPU Tensor Creation (PyTorch)
        if HAS_TORCH and torch.cuda.is_available():
            print("[-] GPU DETECTED: Generating Full Tensor on CUDA...")
            
            # Move numpy arrays to GPU (float32 is sufficient)
            t_left = torch.from_numpy(results['tensor_left']).to('cuda').float()
            t_right = torch.from_numpy(results['tensor_right']).to('cuda').float()
            
            # Create Full Tensor via Outer Product
            # Shape: [2^15, 2^15] -> [32768, 32768]
            # VRAM: ~4 GB
            full_tensor_gpu = torch.ger(t_left, t_right)
            
            print(f"    Full Tensor Shape: {full_tensor_gpu.shape}")
            print(f"    Device: {full_tensor_gpu.device}")
            print(f"    Max Probability: {torch.max(full_tensor_gpu).item():.8f}")
            
            # 4. Generate Heatmap Visualization
            print("[-] Generating Heatmap (Top-Left 100x100 slice)...")
            # Grab a slice from GPU to CPU
            slice_cpu = full_tensor_gpu[:100, :100].cpu().numpy()
            
            plt.figure(figsize=(8, 6))
            plt.imshow(slice_cpu, cmap='inferno', interpolation='nearest')
            plt.title(f"Patch Circuit Interference (XEB={results['xeb']:.2f})")
            plt.colorbar(label="Probability Magnitude")
            plt.savefig("patch_heatmap.png")
            print("    [+] Saved visualization to 'patch_heatmap.png'")
            
            # Clean up GPU memory
            del full_tensor_gpu
            torch.cuda.empty_cache()
            
        elif HAS_TORCH:
             print("[!] GPU not detected. Skipping full tensor creation (too large for CPU).")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
