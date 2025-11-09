#
# LABS solver using correct HUBO-to-QUBO reduction
# This version includes a validity check on the solver's output
# and the fix for minimization.
#

import numpy as np
import dimod
import sys
import os
import argparse

from dimod import BinaryPolynomial, BINARY
from pyqrackising import spin_glass_solver

# --- Function to build the LABS HUBO ---
def create_labs_hubo(N):
    """
    Creates the HUBO (Higher-Order Unconstrained Binary Optimization) model 
    for the LABS problem for a sequence of length N.
    Based on the formula E(s) = sum(Ck^2) from the article.
    """
    print(f"  Building symbolic LABS HUBO (N={N})...")

    bin_vars = [dimod.Binary(i) for i in range(N)]
    spin_vars = [2 * b - 1 for b in bin_vars]
    E = BinaryPolynomial({}, BINARY)

    for k in range(1, N):
        Ck = 0.0
        for i in range(N - k):
            Ck += spin_vars[i] * spin_vars[i + k]

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

        # 6. Manually compute E += Ck_poly * Ck_poly
        ck_terms = list(Ck_poly.items())
        for i in range(len(ck_terms)):
            term_i_tuple, bias_i = ck_terms[i]
            term_i = frozenset(term_i_tuple)

            key_i = tuple(sorted(term_i))
            E[key_i] = E.get(key_i, 0.0) + (bias_i * bias_i)

            for j in range(i + 1, len(ck_terms)):
                term_j_tuple, bias_j = ck_terms[j]
                term_j = frozenset(term_j_tuple)
                new_term = term_i.union(term_j)
                new_bias = 2 * bias_i * bias_j
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

    for (i_label, j_label), coupling in J.items():
        if i_label in label_to_index and j_label in label_to_index:
            i, j = label_to_index[i_label], label_to_index[j_label]
            
            # --- THE FIX ---
            # Negate J to flip from maximization to minimization
            W[i, j] = -coupling
            W[j, i] = -coupling
            # --- END OF FIX ---

    for i_label, bias in h.items():
        if i_label in label_to_index:
            i = label_to_index[i_label]
            # This part is already correct for the conversion
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
    # 'spins' dict is keyed by INDEX (0..23)
    spins = {i: 1 - 2 * int(bit) for i, bit in enumerate(bitstring)}
    ancilla_node_idx = num_bqm_vars
    ancilla_spin = spins.get(ancilla_node_idx, 1)
    
    # 'normalized_spins' dict is keyed by INDEX (0..23)
    normalized_spins = {var: spin * ancilla_spin for var, spin in spins.items()
                        if var in label_to_index.values()} # Ensure only valid indices

    # Step 3: Extract the *original* N spins from the BQM result
    result_sequence_spins = []
    for i in range(N):
        # 'i' is the LABEL we are looking for
        bqm_index = label_to_index.get(i) 
        if bqm_index is not None:
             # Get the spin value from the INDEX
             spin_value = normalized_spins.get(bqm_index, 1)
             result_sequence_spins.append(int(spin_value))
        else:
             result_sequence_spins.append(1) 

    # Step 4: Calculate the energy of this sequence
    energy = calculate_labs_energy(result_sequence_spins)

    # MODIFIED: Return normalized_spins (keyed by INDEX)
    return result_sequence_spins, energy, normalized_spins

# --- Main execution block ---
if __name__ == '__main__':
    # --- Configuration ---
    parser = argparse.ArgumentParser(description="LABS HUBO-to-QUBO Solver")
    parser.add_argument(
        "--N", type=int, default=8, help="Sequence length N."
    )
    parser.add_argument(
        "--quality", type=int, default=3, help="Solver quality parameter."
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Use the GPU for the solver."
    )
    parser.add_argument(
        "--strength", type=float, default=5000.0, help="Penalty strength."
    )

    args = parser.parse_args()

    N_to_solve = args.N
    solver_quality = args.quality
    use_gpu = args.gpu
    penalty_strength = args.strength

    print(f"Starting LABS solver via HUBO-reduction.")
    print(f"Config: N={N_to_solve}, Quality={solver_quality}, Use_GPU={use_gpu}, Strength={penalty_strength}")

    # 1. Create the HUBO
    hubo = create_labs_hubo(N_to_solve)

    # 2. Reduction: HUBO -> QUBO
    print(f"Starting HUBO-to-QUBO reduction (this can take time)...")
    bqm = dimod.make_quadratic(hubo, strength=penalty_strength, vartype='BINARY')
    print("  Reduction complete.")

    # 3. Prepare BQM (QUBO) for the solver
    label_to_index = {label: i for i, label in enumerate(bqm.variables)}
    num_bqm_vars = len(label_to_index)

    print(f"--- Problem Size ---")
    print(f"Original variables (N): {N_to_solve}")
    print(f"Variables after QUBO reduction: {num_bqm_vars}")
    print(f"--------------------")

    h, J, offset = bqm.to_ising()
    
    max_cut_graph = convert_ising_to_maxcut(h, J, num_bqm_vars, label_to_index)

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
        # MODIFIED: Get normalized_spins back (keyed by INDEX)
        sequence, energy, normalized_spins = solve_and_decode_labs(
            model_data, use_gpu, solver_quality
        )

        # --- NEW: Solver Validity Check ---
        print("\n--- Solver Validity Check ---")
        
        # 1. Create the inverse map: {index: label}
        index_to_label = {i: label for label, i in label_to_index.items()}

        # 2. Build the spin sample {label: spin_value}
        # 'normalized_spins' is {index: spin}
        full_spin_sample = {
            index_to_label[index]: spin
            for index, spin in normalized_spins.items()
            if index in index_to_label # Make sure index is valid
        }
        
        # 3. Convert spin sample to binary sample {label: 0/1}
        full_binary_sample = {
            label: (spin + 1) / 2
            for label, spin in full_spin_sample.items()
        }

        # Calculate the energy of the sample on the reduced QUBO
        # This energy INCLUDES any penalties paid.
        qubo_energy = bqm.energy(full_binary_sample)
        
        # The 'energy' variable is the true LABS energy (from HUBO)
        print(f"  Verified LABS Energy (True): {energy}")
        print(f"  Solver's QUBO Energy (Raw): {qubo_energy}")

        is_valid = np.isclose(qubo_energy, energy)
        if is_valid:
            print("  [PASS] CHECK PASSED: Solution is valid (constraints satisfied).")
        else:
            penalty = qubo_energy - energy
            print(f"  [FAIL] CHECK FAILED: Solution is invalid. Penalty paid: {penalty}")
            print(f"     This likely means the --strength ({penalty_strength}) is too low.")
        # --- End of Check ---

        print("\n--- Result ---")
        print(f"Found sequence (spins): {sequence}")
        print(f"Found energy: {energy}")

    except Exception as e:
        print(f"\n--- Error during solver execution ---")
        print(e)
        print("Please ensure 'pyqrackising' is installed (`pip install pyqrackising`).")
