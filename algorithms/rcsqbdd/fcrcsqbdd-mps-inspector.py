import math
import random
import sys
import numpy as np
import matplotlib.pyplot as plt

# 1. Imports
try:
    import quimb.tensor as qtn
    import quimb
    import torch # Used for efficient Outer Product on GPU
except ImportError:
    print("[!] Error: Quimb or Torch not installed.")
    sys.exit(1)

# --- Gate Definitions ---
def get_H(): return np.array([[1, 1], [1, -1]]) / math.sqrt(2)
def get_RZ(theta): return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]])
def get_CNOT(): return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]).reshape(2,2,2,2)

def run_and_contract_patch(n_qubits, depth, seed):
    """
    Simulates a patch as MPS, then CONTRACTS it to a dense vector.
    """
    rng = random.Random(seed)
    print(f"[-] Simulating Patch ({n_qubits} qubits) as MPS...")
    
    # 1. Build MPS
    mps = qtn.MPS_computational_state("0" * n_qubits)
    
    for d in range(depth):
        # Singles
        for i in range(n_qubits):
            angle = rng.uniform(0, 2 * math.pi)
            mps.gate_(get_H(), i, contract='swap+split')
            mps.gate_(get_RZ(angle), i, contract='swap+split')
            
        # Doubles (Random internal connectivity)
        all_bits = list(range(n_qubits))
        rng.shuffle(all_bits)
        while len(all_bits) > 1:
            u = all_bits.pop()
            v = all_bits.pop()
            mps.gate_(get_CNOT(), (u, v), contract='swap+split')
    
    # 2. Contract to Dense Vector
    # This collapses the Tensor Network into a single array of size 2^N
    print(f"    ...Contracting MPS to Dense Vector (Size 2^{n_qubits})...")
    dense_state = mps.to_dense()
    
    # 3. Convert to Probability Vector (Real numbers)
    # P = |psi|^2
    prob_vector = np.abs(dense_state)**2
    
    return prob_vector

if __name__ == "__main__":
    width = 30
    depth = 10
    mid = width // 2
    
    print(f"[-] Configuration: Width {width}, Depth {depth}")
    
    # ---------------------------------------------------------
    # 1. Get Probability Vectors from MPS
    # ---------------------------------------------------------
    # Left Patch (15 Qubits)
    probs_left = run_and_contract_patch(mid, depth, seed=100)
    
    # Right Patch (15 Qubits)
    probs_right = run_and_contract_patch(width - mid, depth, seed=200)
    
    # ---------------------------------------------------------
    # 2. Construct Full Tensor on GPU
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        print("\n[-] GPU Detected. Constructing Full Tensor (Outer Product)...")
        device = torch.device("cuda")
        
        # Move to GPU
        t_left = torch.from_numpy(probs_left).float().to(device)
        t_right = torch.from_numpy(probs_right).float().to(device)
        
        # Calculate Tensor: T[i,j] = Left[i] * Right[j]
        # Shape: [32768, 32768]
        full_tensor = torch.ger(t_left, t_right)
        
        print(f"    Tensor Shape: {full_tensor.shape}")
        print(f"    Max Probability: {torch.max(full_tensor).item():.8f}")
        
        # ---------------------------------------------------------
        # 3. Visualization
        # ---------------------------------------------------------
        print("[-] Generating Heatmap from MPS Tensor...")
        
        # Grab a slice to visualize the "Plaid" pattern
        slice_cpu = full_tensor[:100, :100].cpu().numpy()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(slice_cpu, cmap='inferno', interpolation='nearest')
        plt.title("Probability Tensor (Derived from MPS)\nNote the 'Plaid' Product State Structure")
        plt.colorbar(label="Probability")
        plt.savefig("mps_tensor_heatmap.png")
        print("    [+] Saved 'mps_tensor_heatmap.png'")
        
        # ---------------------------------------------------------
        # 4. (Optional) Save the Tensor
        # ---------------------------------------------------------
        # Only save if you really need it (it's ~4GB uncompressed)
        # torch.save(full_tensor, "patch_probability_tensor.pt")
        # print("    [+] Saved full tensor to 'patch_probability_tensor.pt'")

    else:
        print("[!] No GPU detected. Skipping full tensor construction to avoid RAM overflow.")
        print("    (You have the left/right vectors in 'probs_left' and 'probs_right'.)")
