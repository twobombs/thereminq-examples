import pickle
import matplotlib.pyplot as plt
import sys
import os

try:
    import quimb.tensor as qtn
    import quimb
except ImportError:
    print("[!] Error: Quimb must be installed to load the pickle.")
    sys.exit(1)

def plot_bond_dimensions(filename):
    print(f"[-] Loading '{filename}'...")
    
    if not os.path.exists(filename):
        print(f"[!] File not found: {filename}")
        return

    # 1. Load the Tensor Network
    with open(filename, 'rb') as f:
        tn = pickle.load(f)

    # FIX: Access the .tensors attribute directly
    # In an MPS, this list is ordered by site (0, 1, 2...)
    tensors = tn.tensors
    n_tensors = len(tensors)
    
    print(f"[-] Loaded Tensor Network with {n_tensors} tensors.")
    print("[-] Scanning bond dimensions...")
    
    bonds = []
    
    for i in range(n_tensors - 1):
        t_current = tensors[i]
        t_next = tensors[i+1]
        
        # Calculate shared indices manually (Set Intersection)
        shared_inds = set(t_current.inds).intersection(set(t_next.inds))
        
        if not shared_inds:
            # If no index connects them, the bond dim is 1 (disconnected)
            rank = 1
        else:
            # Calculate the product of dimensions of all shared bonds
            rank = 1
            for ind_name in shared_inds:
                # Get dimension of the index from the tensor itself
                # (ind_size lookup on the TN can sometimes be slow/complex)
                # We find which axis corresponds to this index name
                for j, name in enumerate(t_current.inds):
                    if name == ind_name:
                        rank *= t_current.shape[j]
                        break
            
        bonds.append(rank)

    # 3. Visualization
    print("[-] Generating Plot...")
    
    links = list(range(n_tensors - 1))
    mid = n_tensors // 2
    
    plt.figure(figsize=(10, 6))
    
    # Plot the line
    plt.plot(links, bonds, marker='o', linestyle='-', color='blue', linewidth=2, label='Bond Dimension')
    
    # Fill under the curve
    plt.fill_between(links, bonds, color='blue', alpha=0.1)
    
    # Highlight the Cut
    plt.axvline(x=mid-1, color='red', linestyle='--', linewidth=2, label=f'The Cut (Link {mid-1})')
    
    # Annotate the Valley
    cut_rank = bonds[mid-1]
    plt.annotate(f'Rank {cut_rank}\n(Zero Entanglement)', 
                 xy=(mid-1, cut_rank), 
                 xytext=(mid-1, max(bonds)/1.5),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 horizontalalignment='center')

    plt.title(f"Forensic Analysis of 'Patch' Circuit Structure\n(File: {filename})")
    plt.xlabel("Qubit Link Index")
    plt.ylabel("Bond Dimension (Rank)")
    plt.yscale('linear') 
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_img = "patch_forensics.png"
    plt.savefig(output_img)
    print(f"[+] Plot saved to '{output_img}'")
    
    # Validation Message
    if cut_rank == 1:
        print("\n[SUCCESS] Center Bond Dimension is 1.")
        print("          This mathematically proves the circuit is a Product State.")
    else:
        print(f"\n[WARNING] Center Bond Dimension is {cut_rank}.")

if __name__ == "__main__":
    target_file = "patch_circuit_w20_d20.pkl" 
    
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        
    plot_bond_dimensions(target_file)
