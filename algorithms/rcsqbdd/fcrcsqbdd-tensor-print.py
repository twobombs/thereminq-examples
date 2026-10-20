import math
import random
import sys
import numpy as np
import pickle  # Added for saving data

try:
    import quimb.tensor as qtn
    import quimb
except ImportError:
    print("[!] Error: Quimb not installed. Run 'pip install quimb'")
    sys.exit(1)

# ------------------------------------------------------------------
# 1. Gate Definitions
# ------------------------------------------------------------------
def get_H(): 
    return np.array([[1, 1], [1, -1]]) / math.sqrt(2)

def get_RZ(theta): 
    return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]])

def get_CNOT(): 
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]).reshape(2,2,2,2)

# ------------------------------------------------------------------
# 2. Patch Circuit Generator
# ------------------------------------------------------------------
def generate_patch_tn(width, depth, seed=42):
    """
    Generates a Tensor Network representing a Patch Circuit.
    Returns the raw uncontracted structure.
    """
    mid = width // 2
    rng = random.Random(seed)
    
    # Initialize the MPS in state |00...0>
    tn = qtn.MPS_computational_state("0" * width)
    
    # Tag the initial tensors
    for i, t in enumerate(tn.tensors):
        t.add_tag("PSI0")
        t.add_tag(f"I{i}") 

    gate_count = 0

    print(f"[-] Building Patch Circuit (Width={width}, Depth={depth})...")
    
    for d in range(depth):
        # --- Single Qubit Gates ---
        for i in range(width):
            angle = rng.uniform(0, 2 * math.pi)
            
            # Apply H 
            tn.gate_(get_H(), i, contract='swap+split', tags=f"GATE_{gate_count}")
            gate_count += 1
            
            # Apply RZ
            tn.gate_(get_RZ(angle), i, contract='swap+split', tags=f"GATE_{gate_count}")
            gate_count += 1
            
        # --- 2-Qubit Couplers (Patch Logic) ---
        all_bits = list(range(width))
        rng.shuffle(all_bits)
        
        while len(all_bits) > 1:
            u = all_bits.pop()
            v = all_bits.pop()
            
            # CHECK: Are they on the same side?
            u_side = u < mid
            v_side = v < mid
            
            if u_side == v_side:
                # Apply CNOT
                tn.gate_(get_CNOT(), (u, v), contract='swap+split', tags=f"GATE_{gate_count}")
                gate_count += 1
            else:
                pass # Elided gate

    return tn

# ------------------------------------------------------------------
# 3. Main Execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Use small width for readable print output, or larger for real data
    w = 6   
    d = 4
    
    if len(sys.argv) >= 3:
        w = int(sys.argv[1])
        d = int(sys.argv[2])

    # 1. Generate
    tn = generate_patch_tn(w, d)

    # 2. Print Structure (as requested previously)
    print("\n" + "="*80)
    print(" 1. OPTIMAL CONTRACTION PATH INFO")
    print("="*80)
    try:
        print(tn.contraction_info())
    except Exception as e:
        print(f"Info unavailable: {e}")

    print("\n" + "="*80)
    print(" 2. RAW TENSOR STRUCTURE")
    print("="*80)
    print(tn)
    
    print("\n" + "="*80)
    print(f" Structure Summary:")
    print(f"  - Total Tensors: {len(tn.tensors)}")
    print(f"  - Outer Indices: {tn.outer_inds()}")
    print("="*80)

    # 3. SAVE TO PICKLE
    filename = f"patch_circuit_w{w}_d{d}.pkl"
    print(f"\n[-] Saving Tensor Network object to '{filename}'...")
    
    try:
        with open(filename, 'wb') as f:
            pickle.dump(tn, f)
        print("    [+] Success! File saved.")
        print("    (You can reload this in Python using: tn = pickle.load(open('filename', 'rb')))")
    except Exception as e:
        print(f"    [!] Error saving pickle: {e}")
