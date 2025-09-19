# this is a wrapper that leverages ising models to factor
# derived from: https://arxiv.org/abs/2301.06738
# gemini25 - initial version : this will fail because reasons

import numpy as np
import dimod
from collections import defaultdict
from pyqrackising import spin_glass_solver
import sys

def create_hubo_for_factorization(N, NQ):
    """
    Generates the HUBO model for factoring integer N.
    """
    hubo = defaultdict(float)
    for l1 in range(NQ):
        for l2 in range(NQ):
            p_qubit, q_qubit = l1, NQ + l2
            coeff = 2**(2 * (l1 + l2)) - 2**(l1 + l2 + 1) * N
            hubo[tuple(sorted((p_qubit, q_qubit)))] += coeff
    for l3 in range(NQ):
        for l1 in range(NQ - 1):
            for l2 in range(l1 + 1, NQ):
                coeff = 2**(l1 + l2 + 2*l3 + 1)
                hubo[tuple(sorted((l1, l2, NQ + l3)))] += coeff
                hubo[tuple(sorted((l3, NQ + l1, NQ + l2)))] += coeff
    for l3 in range(NQ - 1):
        for l4 in range(l3 + 1, NQ):
            for l1 in range(NQ - 1):
                for l2 in range(l1 + 1, NQ):
                    coeff = 2**(l1 + l2 + l3 + l4 + 2)
                    hubo[tuple(sorted((l1, l2, NQ + l3, NQ + l4)))] += coeff
    return hubo

def convert_ising_to_maxcut(h, J, num_vars, label_to_index):
    """
    Converts an Ising model to an equivalent Max-Cut problem graph.
    """
    graph_size = num_vars + 1
    ancilla_node_idx = num_vars
    W = np.zeros((graph_size, graph_size))
    
    for (i_label, j_label), coupling in J.items():
        i, j = label_to_index[i_label], label_to_index[j_label]
        W[i, j] = coupling
        W[j, i] = coupling

    for i_label, bias in h.items():
        i = label_to_index[i_label]
        W[i, ancilla_node_idx] = -bias
        W[ancilla_node_idx, i] = -bias
        
    return W

def solve_and_decode(max_cut_graph, NQ, label_to_index, quality, correction_quality):
    """
    A single run of the solver and result decoding.
    """
    num_total_vars = max_cut_graph.shape[0]
    
    result_tuple = spin_glass_solver(
        max_cut_graph, 
        quality=quality, 
        correction_quality=correction_quality
    )
    bitstring = result_tuple[0]

    spins = {i: 1 - 2 * int(bit) for i, bit in enumerate(bitstring)}
    ancilla_node_idx = num_total_vars - 1
    ancilla_spin = spins.get(ancilla_node_idx, 1)
    normalized_spins = {var: spin * ancilla_spin for var, spin in spins.items()}
    binary_solution = {i: (s + 1) // 2 for i, s in normalized_spins.items()}

    p, q = 0, 0
    for i in range(NQ):
        if binary_solution.get(label_to_index.get(i), 0) == 1:
            p += 2**i
    for i in range(NQ):
        if binary_solution.get(label_to_index.get(NQ + i), 0) == 1:
            q += 2**i
            
    return p, q

# --- Main execution block ---
if __name__ == '__main__':
    N_to_factor = 15
    num_qubits_per_factor = 3
    
    # Solver Tuning Parameters
    quality = 2
    correction_quality = 4
    num_runs = 10
    
    # --- Model Preparation (Done once) ---
    hubo = create_hubo_for_factorization(N_to_factor, num_qubits_per_factor)
    max_coeff = max(abs(c) for c in hubo.values()) if hubo else 1.0
    bqm = dimod.make_quadratic(hubo, strength=max_coeff * 5, vartype='BINARY')
    label_to_index = {label: i for i, label in enumerate(bqm.variables)}
    num_bqm_vars = len(label_to_index)
    h, J, offset = bqm.to_ising()
    max_cut_graph = convert_ising_to_maxcut(h, J, num_bqm_vars, label_to_index)
    
    print(f"Original problem variables: {2 * num_qubits_per_factor}")
    print(f"Total variables after HUBO->QUBO reduction: {num_bqm_vars}")
    print(f"Max-Cut graph size: {max_cut_graph.shape[0]}x{max_cut_graph.shape[1]}")
    print(f"Running solver {num_runs} times to find the best solution...\n")
    
    # --- Solver Loop ---
    best_solution = (0, 0)
    lowest_cost = sys.maxsize

    for i in range(num_runs):
        p, q = solve_and_decode(
            max_cut_graph, 
            num_qubits_per_factor,
            label_to_index,
            quality=quality,
            correction_quality=correction_quality
        )
        
        cost = (p * q - N_to_factor)**2
        print(f"Run {i+1}/{num_runs}: Found ({p}, {q}). Cost = {cost}")
        
        if cost < lowest_cost:
            lowest_cost = cost
            best_solution = (p, q)
    
    # --- Final Results ---
    print("\n--- Best Result Found ---")
    factors = sorted(best_solution)
    print(f"Found factors p = {factors[0]}, q = {factors[1]}")
    print(f"Verification: {factors[0]} * {factors[1]} = {factors[0] * factors[1]} (N={N_to_factor})")
    if lowest_cost == 0:
        print("Success: Optimal solution found!")
    else:
        print("Note: Solver did not find the optimal solution.")
      
