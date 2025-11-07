import numpy as np
import dimod
import sys
import os

# We assume pyqrackising is installed
# (e.g., via 'pip install pyqrackising')
from pyqrackising import spin_glass_solver

# --- Function to build the LABS HUBO ---
# This is the new, crucial component
def create_labs_hubo(N):
    """
    Creates the HUBO (Higher-Order Unconstrained Binary Optimization) model 
    for the LABS problem for a sequence of length N.
    Based on the formula E(s) = sum(Ck^2) from the article.
    """
    print(f"  Building symbolic LABS HUBO (N={N})...")
    
    # Create N symbolic *binary* variables (b0, b1, ..., b{N-1})
    bin_vars = [dimod.Binary(i) for i in range(N)]
    
    # Convert them to symbolic *spin* variables (s_i = 2*b_i - 1)
    spin_vars = [2 * b - 1 for b in bin_vars]
    
    E = 0.0 # This will become our total symbolic energy expression
    
    # Loop for k from 1 to N-1
    for k in range(1, N):
        Ck = 0.0 # Symbolic expression for Ck
        
        # Loop for i from 1 to N-k (but 0-indexed: 0 to N-k-1)
        # This implements Ck = sum(s_i * s_{i+k})
        for i in range(N - k):
            Ck += spin_vars[i] * spin_vars[i + k]
            
        # Add Ck^2 to the total energy. 
        # dimod expands this square symbolically, which creates the 4-body terms.
        E += Ck**2
        
    print("  Symbolic HUBO construction complete.")
    # E is now a large 'dimod.Poly' object containing all 2-body, 4-body
    # interactions, and an offset.
    return E

# --- Function to convert Ising to MaxCut graph ---
# (Taken directly from your factorization example)
def convert_ising_to_maxcut(h, J, num_vars, label_to_index):
    """Converts an Ising model (h, J) to a Max-Cut weight graph W."""
    graph_size = num_vars + 1
    ancilla_node_idx = num_vars
    W = np.zeros((graph_size, graph_size))
    
    # J-couplings (2-body)
    for (i_label, j_label), coupling in J.items():
        if i_label in label_to_index and j_label in label_to_index:
            i, j = label_to_index[i_label], label_to_index[j_label]
            W[i, j] = coupling
            W[j, i] = coupling
            
    # h-biases (1-body) become couplings to the ancilla node
    for i_label, bias in h.items():
        if i_label in label_to_index:
            i = label_to_index[i_label]
            W[i, ancilla_node_idx] = -bias
            W[ancilla_node_idx, i] = -bias
            
    return W

# --- Function to calculate the energy of a found sequence ---
# (Uses the original LABS formula for verification)
def calculate_labs_energy(sequence_spins):
    """Calculates the LABS energy for a given spin sequence."""
    N = len(sequence_spins)
    E = 0
    for k in range(1, N):
        Ck = 0
        for i in range(N - k):
            Ck += sequence_spins[i] * sequence_spins[i + k]
        E += Ck**2
    return E

# --- Custom Solver & Decoder ---
def solve_and_decode_labs(model_data, use_gpu, quality):
    """Runs the solver and decodes the result into a LABS sequence."""
    
    W = model_data['graph']
    N = model_data['N']
    label_to_index = model_data['labels']
    num_bqm_vars = model_data['num_bqm_vars']

    print(f"  Starting PyQrackIsing solver (GPU={use_gpu}, Quality={quality})...")
    
    # Step 1: Solve the reduced Max-Cut / Ising problem
    result_tuple = spin_glass_solver(
        W,
        quality=quality,
        is_combo_maxcut_gpu=use_gpu
    )
    bitstring = result_tuple[0]
    
    # Step 2: Convert the result back to normalized spins
    # (This logic is from your factorization wrapper, handling the ancilla)
    spins = {i: 1 - 2 * int(bit) for i, bit in enumerate(bitstring)}
    ancilla_node_idx = num_bqm_vars
    ancilla_spin = spins.get(ancilla_node_idx, 1)
    normalized_spins = {var: spin * ancilla_spin for var, spin in spins.items()}

    # Step 3: Extract the *original* N spins from the BQM result
    # We ignore all the ancilla variables that 'dimod' created.
    # Our original variables had labels 0, 1, ..., N-1.
    
    result_sequence_spins = []
    for i in range(N):
        # Find the index in the BQM for our original variable 'i'
        bqm_index = label_to_index[i]
        
        # Get the spin value (+1 or -1)
        spin_value = normalized_spins.get(bqm_index, 1) # Default to +1 if not found
        result_sequence_spins.append(int(spin_value))

    # Step 4: Calculate the energy of this sequence using the *original* HUBO formula
    energy = calculate_labs_energy(result_sequence_spins)
    
    return result_sequence_spins, energy

# --- Main execution block ---
if __name__ == '__main__':
    # --- Configuration ---
    N_to_solve = 8  # IMPORTANT: Even N=10 is HUGE after reduction!
    solver_quality = 4
    use_gpu = False # Set to True if you have a GPU with OpenCL

    print(f"Starting LABS solver for N = {N_to_solve} via HUBO-reduction.")
    
    # 1. Create the HUBO
    # This object contains 4-body terms
    hubo = create_labs_hubo(N_to_solve)

    # 2. Reduction: HUBO -> QUBO
    # THIS IS THE SCALING BOTTLENECK
    # 'dimod' adds ancilla variables and penalty terms to
    # convert 4-body terms into 2-body terms.
    print(f"Starting HUBO-to-QUBO reduction (this can take time)...")
    # The 'strength' is a penalty coefficient. Must be high enough.
    bqm = dimod.make_quadratic(hubo, strength=50.0, vartype='BINARY')
    print("  Reduction complete.")

    # 3. Prepare the BQM (QUBO) for the solver
    # We map the (potentially many) variables in the BQM to indices
    label_to_index = {label: i for i, label in enumerate(bqm.variables)}
    num_bqm_vars = len(label_to_index)
    
    print(f"--- Problem Size ---")
    print(f"Original variables (N): {N_to_solve}")
    print(f"Variables after QUBO reduction: {num_bqm_vars}")
    print(f"--------------------")

    # Convert BQM to Ising model (h, J)
    h, J, offset = bqm.to_ising()
    
    # Convert Ising model to Max-Cut graph (W)
    max_cut_graph = convert_ising_to_maxcut(h, J, num_bqm_vars, label_to_index)

    # Normalize the graph (good practice for the solver)
    max_abs_val = np.max(np.abs(max_cut_graph))
    if max_abs_val > 0:
        max_cut_graph /= max_abs_val

    # 4. Package data and solve
    model_data = {
        'graph': max_cut_graph,
        'N': N_to_solve,
        'labels': label_to_index,
        'num_bqm_vars': num_bqm_vars
    }
    
    try:
        sequence, energy = solve_and_decode_labs(model_data, use_gpu, solver_quality)
        
        print("\n--- Result ---")
        print(f"Found sequence (spins): {sequence}")
        print(f"Found energy: {energy}")

    except Exception as e:
        print(f"\n--- Error during solver execution ---")
        print(e)
        print("Please ensure 'pyqrackising' is installed (`pip install pyqrackising`).")

