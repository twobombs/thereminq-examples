import numpy as np
import dimod
import sys
import os
import argparse # Import argparse

# Add these imports
from dimod import BinaryPolynomial, BINARY

# We assume pyqrackising is installed
# (e.g., via 'pip install pyqrackising')
from pyqrackising import spin_glass_solver

# --- Function to build the LABS HUBO ---
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
   
    # We must explicitly use a Polynomial object to hold the 4th-order terms.
    E = BinaryPolynomial({}, BINARY)
   
    # Loop for k from 1 to N-1
    for k in range(1, N):
        Ck = 0.0 # Symbolic expression for Ck (will become a BQM)
       
        # Loop for i from 1 to N-k (but 0-indexed: 0 to N-k-1)
        # This implements Ck = sum(s_i * s_{i+k})
        for i in range(N - k):
            Ck += spin_vars[i] * spin_vars[i + k]
           
        # Add Ck^2 to the total energy. 
       
        # THE FIX (Part 1):
        # Manually convert the 'BinaryQuadraticModel' (Ck)
        # into a dictionary that 'BinaryPolynomial' understands.
       
        # 1. Initialize the polynomial terms dictionary
        poly_terms = {}
       
        # 2. Add linear terms from Ck
        if hasattr(Ck, 'linear'):
            for v, bias in Ck.linear.items():
                poly_terms[(v,)] = bias
       
        # 3. Add quadratic terms from Ck
        if hasattr(Ck, 'quadratic'):
            poly_terms.update(Ck.quadratic)
       
        # 4. Add the constant offset term
        if hasattr(Ck, 'offset'):
            poly_terms[()] = Ck.offset
       
        # 5. Create the BinaryPolynomial from the terms dict
        Ck_poly = BinaryPolynomial(poly_terms, vartype=BINARY)
       
        # THE FIX (Part 2):
        # Manually compute E += Ck_poly * Ck_poly
       
        ck_terms = list(Ck_poly.items())
        for i in range(len(ck_terms)):
            term_i_tuple, bias_i = ck_terms[i]
            # Use frozenset for hashable set operations
            term_i = frozenset(term_i_tuple) 
           
            # Multiply with itself (i == j term)
            key_i = tuple(sorted(term_i))
            E[key_i] = E.get(key_i, 0.0) + (bias_i * bias_i)

            # Multiply with other terms (i < j terms)
            for j in range(i + 1, len(ck_terms)):
                term_j_tuple, bias_j = ck_terms[j]
                term_j = frozenset(term_j_tuple)
               
                # new_term = term_i * term_j = union(term_i, term_j)
                new_term = term_i.union(term_j)
               
                # new_bias = 2 * bias_i * bias_j
                new_bias = 2 * bias_i * bias_j
               
                # Create the canonical tuple key, sorted
                key_j = tuple(sorted(new_term))
                E[key_j] = E.get(key_j, 0.0) + new_bias
       
    print("  Symbolic HUBO construction complete.")
    return E

# --- Function to convert Ising to MaxCut graph ---
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
        is_maxcut_gpu=use_gpu 
    )
    bitstring = result_tuple[0]
   
    # Step 2: Convert the result back to normalized spins
    spins = {i: 1 - 2 * int(bit) for i, bit in enumerate(bitstring)}
    ancilla_node_idx = num_bqm_vars
    ancilla_spin = spins.get(ancilla_node_idx, 1)
    normalized_spins = {var: spin * ancilla_spin for var, spin in spins.items()}

    # Step 3: Extract the *original* N spins from the BQM result
    result_sequence_spins = []
    for i in range(N):
        bqm_index = label_to_index[i]
        spin_value = normalized_spins.get(bqm_index, 1)
        result_sequence_spins.append(int(spin_value))

    # Step 4: Calculate the energy of this sequence
    energy = calculate_labs_energy(result_sequence_spins)
   
    return result_sequence_spins, energy

# --- Main execution block ---
if __name__ == '__main__':
    # --- Configuration ---
    # NEW: Add argument parsing
    parser = argparse.ArgumentParser(description="LABS HUBO-to-QUBO Solver Wrapper")
    parser.add_argument(
        "--N",
        type=int,
        default=8,  # Default N=8
        help="Sequence length N for the LABS problem."
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=3, # Default quality=3
        help="Solver quality parameter for PyQrackIsing."
    )
    parser.add_argument(
        "--gpu",
        action="store_true", # Default is False
        help="Use the GPU for the solver."
    )
    
    args = parser.parse_args()

    # Set parameters from args
    N_to_solve = args.N
    solver_quality = args.quality
    use_gpu = args.gpu

    print(f"Starting LABS solver via HUBO-reduction.")
    # This is the line that was cut off
    print(f"Config: N={N_to_solve}, Quality={solver_quality}, Use_GPU={use_gpu}")
   
    # 1. Create the HUBO
    hubo = create_labs_hubo(N_to_solve)

    # 2. Reduction: HUBO -> QUBO
    print(f"Starting HUBO-to-QUBO reduction (this can take time)...")
    # Using a high strength value for the reduction
    bqm = dimod.make_quadratic(hubo, strength=50.0, vartype='BINARY')
    print("  Reduction complete.")

    # 3. Prepare the BQM (QUBO) for the solver
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

    # Normalize the graph
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
