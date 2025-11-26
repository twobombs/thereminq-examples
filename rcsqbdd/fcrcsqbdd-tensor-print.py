import math
import random
import sys
import numpy as np

try:
    import quimb.tensor as qtn
    import quimb
except ImportError:
    print("[!] Error: Quimb not installed. Run 'pip install quimb'")
    sys.exit(1)

# ------------------------------------------------------------------
# 1. Gate Definitions (Explicit Matrices)
# ------------------------------------------------------------------
def get_H(): 
    return np.array([[1, 1], [1, -1]]) / math.sqrt(2)

def get_RZ(theta): 
    return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]])

def get_CNOT(): 
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]).reshape(2,2,2,2)

# ------------------------------------------------------------------
# 2. Patch Circuit Generator (Returns Raw Tensor Network)
# ------------------------------------------------------------------
def generate_patch_tn(width, depth, seed=42):
    """
    Generates a Tensor Network representing a Patch Circuit.
    It DOES NOT contract it. It returns the raw structure with tags.
    """
    mid = width // 2
    rng = random.Random(seed)
    
    # Initialize the MPS in state |00...0>
    # We use 'MPS_computational_state' as the base, but we will treat it
    # as a generic Tensor Network to see the raw structure.
    tn = qtn.MPS_computational_state("0" * width)
    
    # Tag the initial tensors
    for i, t in enumerate(tn.tensors):
        t.add_tag("PSI0")
        t.add_tag(f"I{i}") # Initial index tag

    gate_count = 0

    print(f"[-] Building Patch Circuit (Width={width}, Depth={depth})...")
    
    for d in range(depth):
        # --- Single Qubit Gates ---
        for i in range(width):
            angle = rng.uniform(0, 2 * math.pi)
            
            # Apply H (Add unique tag for visualization)
            # contract='swap+split' preserves the TN structure while updating bonds
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
                # Elided gate (Crosses the cut) - We skip it
                pass

    return tn

# ------------------------------------------------------------------
# 3. Main Execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    # NOTE: Use a small width (e.g., 6 or 8) to keep the printout readable.
    # If you use 30, the terminal output will be thousands of lines long.
    w = 6   
    d = 4
    
    if len(sys.argv) >= 3:
        w = int(sys.argv[1])
        d = int(sys.argv[2])

    # 1. Generate the Tensor Network
    tn = generate_patch_tn(w, d)

    print("\n" + "="*80)
    print(" 1. OPTIMAL CONTRACTION PATH SEGMENTS AND COST (Top of Screenshot)")
    print("="*80)
    # This calculates the complexity of contracting the network
    # It matches the blue text "Optimal contraction path segments..."
    try:
        print(tn.contraction_info())
    except Exception as e:
        print(f"Could not calculate contraction info: {e}")

    print("\n" + "="*80)
    print(" 2. RAW TENSOR STRUCTURE (Bottom of Screenshot)")
    print("="*80)
    # This prints the list of Tensor objects, shapes, indices, and tags
    # exactly as shown in your "TensorNetworkGenVector" screenshot.
    print(tn)
    
    print("\n" + "="*80)
    print(f" Structure Summary:")
    print(f"  - Total Tensors: {len(tn.tensors)}")
    print(f"  - Outer Indices: {tn.outer_inds()}")
    print("="*80)
