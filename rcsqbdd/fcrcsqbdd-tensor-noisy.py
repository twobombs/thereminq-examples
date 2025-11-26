import math
import random
import sys
import numpy as np
from pyqrack import QrackSimulator
from qiskit import QuantumCircuit

# Visualization Imports
try:
    import torch
    import matplotlib.pyplot as plt
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[!] Warning: torch/matplotlib not found. Images will be skipped.")

# 1. Generate the Circuit BLUEPRINT (Once)
def generate_circuit_params(width, depth):
    mid = width // 2
    layers = []
    
    for d in range(depth):
        layer = {}
        layer['angles'] = [random.uniform(0, 2 * math.pi) for _ in range(width)]
        
        all_bits = list(range(width))
        random.shuffle(all_bits)
        pairs = []
        while len(all_bits) > 1:
            u = all_bits.pop()
            v = all_bits.pop()
            if (u < mid and v < mid) or (u >= mid and v >= mid):
                pairs.append((u, v))
        layer['pairs'] = pairs
        layers.append(layer)
        
    return layers

def apply_random_pauli(circuit, idx):
    r = random.random()
    if r < 0.33: circuit.x(idx)
    elif r < 0.66: circuit.y(idx)
    else: circuit.z(idx)

# 2. Benchmark Function (Returns Score AND Vectors)
def bench_fixed_structure(width, blueprint, noise_rate, shots=5000):
    mid = width // 2
    mask_lower = (1 << mid) - 1 
    
    ideal_left = QuantumCircuit(mid)
    ideal_right = QuantumCircuit(width - mid)
    noisy_full = QuantumCircuit(width)
    
    # Reconstruct from Blueprint
    for layer in blueprint:
        for i, angle in enumerate(layer['angles']):
            # Ideal
            if i < mid:
                ideal_left.h(i)
                ideal_left.rz(angle, i)
            else:
                ideal_right.h(i - mid)
                ideal_right.rz(angle, i - mid)
            
            # Noisy
            noisy_full.h(i)
            noisy_full.rz(angle, i)
            if noise_rate > 0 and random.random() < noise_rate:
                apply_random_pauli(noisy_full, i)
        
        for (u, v) in layer['pairs']:
            if u < mid:
                ideal_left.cx(u, v)
            else:
                ideal_right.cx(u - mid, v - mid)
            
            noisy_full.cx(u, v)
            if noise_rate > 0:
                if random.random() < noise_rate: apply_random_pauli(noisy_full, u)
                if random.random() < noise_rate: apply_random_pauli(noisy_full, v)

    # Simulation
    sim_exp = QrackSimulator(width)
    sim_exp.run_qiskit_circuit(noisy_full)
    measured_shots = sim_exp.measure_shots(list(range(width)), shots)
    
    sim_left = QrackSimulator(mid)
    sim_left.run_qiskit_circuit(ideal_left)
    probs_left = np.array(sim_left.prob_all(list(range(mid))))
    
    sim_right = QrackSimulator(width - mid)
    sim_right.run_qiskit_circuit(ideal_right)
    probs_right = np.array(sim_right.prob_all(list(range(width - mid))))

    # XEB Calculation
    sum_probs = 0.0
    for k in measured_shots:
        k_left = k & mask_lower
        k_right = k >> mid
        sum_probs += probs_left[k_left] * probs_right[k_right]

    xeb = (2**width * (sum_probs / shots)) - 1
    
    # RETURN VECTORS NOW
    return xeb, probs_left, probs_right

if __name__ == "__main__":
    w, d = 30, 10
    print(f"Generating ONE circuit structure for {w} qubits...")
    blueprint = generate_circuit_params(w, d)
    
    # We will test these noise levels
    noise_levels = [0.000, 0.002, 0.005, 0.010]
    
    print("\nStarting Noise Sweep & Visualization...")
    print(f"{'Noise':<10} | {'XEB Score':<15} | {'Image Saved':<20}")
    print("-" * 50)
    
    for n in noise_levels:
        # Get data back from function
        score, p_left, p_right = bench_fixed_structure(w, blueprint, n)
        
        filename = "Skipped"
        
        # --- Visualization Logic ---
        if HAS_TORCH and torch.cuda.is_available():
            try:
                # 1. Create Tensor on GPU
                t_left = torch.from_numpy(p_left).to('cuda').float()
                t_right = torch.from_numpy(p_right).to('cuda').float()
                full_tensor = torch.ger(t_left, t_right)
                
                # 2. Grab a slice
                slice_cpu = full_tensor[:100, :100].cpu().numpy()
                
                # 3. Plot
                filename = f"heatmap_noise_{n*100:.1f}.png"
                plt.figure(figsize=(6, 6))
                plt.imshow(slice_cpu, cmap='inferno', interpolation='nearest')
                plt.title(f"Noise {n*100:.1f}% (XEB={score:.2f})")
                plt.colorbar(label="Prob")
                plt.savefig(filename)
                plt.close() # Important: Close memory to prevent leaks
                
                # 4. Clean GPU
                del full_tensor
                del t_left
                del t_right
                torch.cuda.empty_cache()
                
            except Exception as e:
                filename = f"Error: {e}"
        
        print(f"{n*100:5.2f}%    | {score:7.5f}         | {filename}")
