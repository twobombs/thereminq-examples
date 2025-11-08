#
# LABS dcqd shortcut
# gemini25
#

import math
# FIX: Changed 'PyQrack' to 'QrackSimulator'
from pyqrack import QrackSimulator
import random # Needed for classical MTS

def apply_r_pauli_string(sim, qubits, pauli_string, angle):
    """
    Applies a generalized Pauli rotation R_P(angle) = exp(-i * angle/2 * P)
    where P is a Pauli string like "YZZZ".
    
    This is the core building block for the DCQO circuit.
    
    Args:
        sim (QrackSimulator): The simulator instance.
        qubits (list): A list of qubit indices, e.g., [q_i, q_l, q_k, q_kl].
        pauli_string (str): The Pauli string, e.g., "YZZZ".
        angle (float): The rotation angle (theta).
    """
    
    # --- 1. Apply basis change gates ---
    # We transform from the P basis to the Z basis.
    for i, pauli in enumerate(pauli_string):
        q = qubits[i]
        if pauli == 'X':
            sim.h(q)  # H gate transforms X -> Z
        elif pauli == 'Y':
            sim.adjs(q) # S-dagger gate
            sim.h(q)  # H+S-dagger transforms Y -> Z

    # --- 2. Apply CNOT cascade ---
    # This entangles all qubits to a single target qubit (the last one)
    # to compute the parity.
    target_q = qubits[-1]
    for i in range(len(qubits) - 1):
        sim.mcx([qubits[i]], target_q) # This is just CNOT

    # --- 3. Apply the Z-rotation ---
    # The rotation is applied to the target qubit.
    # FIX: Changed 'sim.rz(angle, target_q)'
    # FIX: Swapped angle and axis: 'sim.r(angle, 3, target_q)'
    # The correct signature is r(b: int, ph: float, q: int)
    sim.r(3, angle, target_q)

    # --- 4. Un-compute CNOT cascade ---
    # We must reverse the CNOTs to disentangle.
    for i in range(len(qubits) - 2, -1, -1): # Reverse order
        sim.mcx([qubits[i]], target_q)

    # --- 5. Un-compute basis change gates ---
    # We transform back from the Z basis to the P basis.
    for i, pauli in enumerate(pauli_string):
        q = qubits[i]
        if pauli == 'X':
            sim.h(q)  # H is its own inverse
        elif pauli == 'Y':
            sim.h(q)
            sim.s(q)  # S gate (inverse of S-dagger)

# ----------------------------------------------------------------------
# --- Main Algorithm Skeleton ---
# ----------------------------------------------------------------------

def calculate_theta_and_alpha(t, T, num_trotter_steps):
    """
    **[USER MUST IMPLEMENT]** -> **[Implemented]**
    
    Calculates the schedule based on the "impulse regime" and a
    linear schedule for lambda(t), as suggested by the paper (Sec III.A)
    and citation [15].
    
    This implements the 'theta' required for the rotation gates in Eq. (B3).
    """
    lambda_t = t / T
    alpha_1 = 1.0 - lambda_t  # Placeholder, but a common choice
    dot_lambda = 1.0 / T      # Derivative of lambda(t)
    
    # Delta_t is the time step for a single Trotter step
    delta_t = T / num_trotter_steps
    
    # This is theta(k*Delta_t) from Appendix B
    # theta(t) = Delta_t * alpha_1(t) * dot_lambda(t)
    theta = delta_t * alpha_1 * dot_lambda
    
    return theta

def get_problem_hamiltonian_terms(N):
    """
    **[USER MUST IMPLEMENT]** -> **[Implemented]**
    
    This function parses the problem Hamiltonian H_f (Eq. 2)
    and returns lists of the qubit indices for all 2-body
    and 4-body terms.
    
    Based on Eq. 2 and Eq. 7.
    """
    two_body_terms = []
    four_body_terms = []

    # Eq. 2, Term 1 (2-body): 2 * sum_{i=1}^{N-2} sum_{k=1}^{\lfloor(N-i)/2\rfloor}
    # Note: Paper uses 1-based indexing, code uses 0-based.
    for i in range(N - 2): # i from 0 to N-3
        for k in range(1, (N - 1 - i) // 2 + 1): # k from 1 to floor((N-i-1)/2)
            two_body_terms.append((i, i + k))
            
    # Eq. 2, Term 2 (4-body): 4 * sum_{i=1}^{N-3} sum_{l=1}^{\lfloor(N-i-1)/2\rfloor} sum_{k=l+1}^{N-i-l}
    for i in range(N - 3): # i from 0 to N-4
        for l in range(1, (N - 1 - i - 1) // 2 + 1): # l from 1 to floor((N-i-2)/2)
            for k in range(l + 1, N - i - l + 1): # k from l+1 to N-i-l
                # The 4-body term is (i, i+l, i+k, i+k+l)
                # But Eq. 7 shows the terms are (i, i+l, i+k, i+k+l)
                # Let's re-read Eq. 2...
                # sigma_i * sigma_{i+k} * sigma_{i+l} * sigma_{i+k+l}
                # No, that's not right.
                # Re-reading Eq. 2:
                # 4 * sum_i * sum_l * sum_k(>l) (sigma_i * sigma_{i+l} * sigma_{i+k} * sigma_{i+k+l})
                # Ah, my placeholder was wrong. Let's trace Eq 7.
                # Eq 7: (i, i+l, i+k, i+k+l)
                # This seems to be the one used in the CD term.
                # Let's check the placeholder again. (0, 1, 2, 3)
                # If i=0, l=1, k=2:
                # N-i-l = 4-0-1 = 3. k=2 is valid.
                # (i, i+l, i+k, i+k+l) -> (0, 1, 2, 3)
                # This matches the placeholder.
                
                # Correction: The loops in Eq. 2 are:
                # i from 1 to N-3
                # l from 1 to floor((N-i-1)/2)
                # k from l+1 to N-i-l
                # Term: s_i * s_{i+l} * s_{i+k} * s_{i+k+l}
                # This is a mistake in the placeholder code and my previous implementation.
                
                # Let's try again.
                # i_0 = i
                # i_1 = i + l
                # i_2 = i + k
                # i_3 = i + k + l
                # The four qubits are (i, i+l, i+k, i+k+l)
                
                # Let's re-read the placeholder implementation
                # four_body_terms = [(0, 1, 2, 3)]
                # This matches (i, i+l, i+k, i+k+l) for i=0, l=1, k=2
                # My loop logic was wrong.
                
                # Correct loops for 0-based indexing:
                # i from 0 to N-4
                for i_idx in range(N - 3):
                    # l from 1 to floor((N-i-2)/2)
                    for l_val in range(1, (N - i_idx - 2) // 2 + 1):
                        # k from l+1 to N-i-l-1
                        for k_val in range(l_val + 1, (N - i_idx - l_val)):
                            q1 = i_idx
                            q2 = i_idx + l_val
                            q3 = i_idx + k_val
                            q4 = i_idx + k_val + l_val
                            if q4 < N: # Ensure we are in bounds
                                four_body_terms.append((q1, q2, q3, q4))

    # The loops above are complex. Let's use the loops from the *placeholder*
    # which seem to match the paper's Eq. 7 structure.
    
    two_body_terms = []
    four_body_terms = []

    # 2-body terms from Eq (7)
    # i from 1 to N-2. k from 1 to floor((N-i)/2)
    # 0-indexed: i from 0 to N-3. k from 1 to floor((N-1-i)/2)
    for i in range(N - 2):
        for k in range(1, (N - 1 - i) // 2 + 1):
            two_body_terms.append((i, i + k))

    # 4-body terms from Eq (7)
    # i from 1 to N-3. l from 1 to floor((N-i-1)/2). k from l+1 to N-i-l
    # 0-indexed: i from 0 to N-4. l from 1 to floor((N-i-2)/2). k from l+1 to N-i-l-1
    for i in range(N - 3):
        for l in range(1, (N - i - 2) // 2 + 1):
            for k in range(l + 1, N - i - l):
                four_body_terms.append((i, i + l, i + k, i + k + l))

    return two_body_terms, four_body_terms

def run_dcqo_quantum_stage(N, num_trotter_steps, T):
    """
    Runs the full quantum simulation part of the QE-MTS algorithm
    and returns a SINGLE measurement sample.
    """
    print(f"Initializing PyQrack simulator for N={N} qubits.")
    # FIX: Changed 'PyQrack.QrackSimulator(N)' to 'QrackSimulator(N)'
    sim = QrackSimulator(N)
    
    # 1. Get the terms of the LABS Hamiltonian
    two_body_terms, four_body_terms = get_problem_hamiltonian_terms(N)

    # 2. Initialize state to |+> (ground state of H_i)
    # We can do this by applying a Hadamard (H) gate to all qubits.
    for i in range(N):
        sim.h(i)

    print(f"Beginning {num_trotter_steps} Trotter steps...")
    
    # 3. Perform the "Digitized" (Trotterized) time evolution
    for k in range(1, num_trotter_steps + 1):
        t = (k / num_trotter_steps) * T
        
        # This is the "theta" from Appendix B, Eq. (B3)
        # It's a complex function you must implement.
        theta = calculate_theta_and_alpha(t, T, num_trotter_steps)

        # Apply 2-body rotations (from Eq. B3)
        for (i, k_idx) in two_body_terms:
            # These are R_YZ(theta) and R_ZY(theta)
            # (Assuming h_i, h_k are transverse fields, often -1)
            h_i, h_k = -1, -1 
            apply_r_pauli_string(sim, [i, k_idx], "YZ", 4 * theta * h_i)
            apply_r_pauli_string(sim, [i, k_idx], "ZY", 4 * theta * h_k)

        # Apply 4-body rotations (from Eq. B3)
        # This is the most complex part of the circuit
        for (i, l, k_idx, kl) in four_body_terms:
            # These are R_YZZZ, R_ZYZZ, R_ZZYZ, R_ZZZY
            h_i, h_l, h_k, h_kl = -1, -1, -1, -1 # Placeholders
            
            # This directly implements the terms in Eq. (B3)
            apply_r_pauli_string(sim, [i, l, k_idx, kl], "YZZZ", 8 * theta * h_i)
            apply_r_pauli_string(sim, [i, l, k_idx, kl], "ZYZZ", 8 * theta * h_l)
            apply_r_pauli_string(sim, [i, l, k_idx, kl], "ZZYZ", 8 * theta * h_k)
            apply_r_pauli_string(sim, [i, l, k_idx, kl], "ZZZY", 8 * theta * h_kl)
    
    print("Time evolution complete.")
    
    # 4. Measure all qubits to get a bitstring sample
    # FIX: Changed 'measure_all()' to 'm_all()'
    shot = sim.m_all() 
    
    # FIX: Removed 'sim.destroy()', which caused an AttributeError.
    # Python's garbage collector will handle cleanup when 'sim'
    # goes out of scope at the end of this function.
    
    return shot

# ----------------------------------------------------------------------
# --- Classical MTS Implementation (Algorithms 1, 2, 3) ---
# ----------------------------------------------------------------------

def int_to_sequence(shot, N):
    """Converts an integer bitstring 'shot' into a +/-1 sequence."""
    s = []
    for i in range(N):
        if (shot >> i) & 1:
            s.append(1)  # 1 bit -> +1 spin
        else:
            s.append(-1) # 0 bit -> -1 spin
    return s

def calculate_energy(s, N):
    """Calculates the LABS energy for a +/-1 sequence 's' (Eq. 1)."""
    E = 0
    for k in range(1, N): # k from 1 to N-1
        Ck = 0
        for i in range(N - k): # i from 0 to N-k-1
            Ck += s[i] * s[i+k]
        E += Ck * Ck
    return E

def combine(p1, p2, N):
    """Algorithm 3: COMBINE (single-point crossover)."""
    k = random.randint(1, N - 1) # Cut point
    return p1[:k] + p2[k:]

def mutate(s, p_mut, N):
    """Algorithm 3: MUTATE (bit-flip mutation)."""
    s_new = list(s) # Make a copy
    for i in range(N):
        if random.random() < p_mut:
            s_new[i] *= -1 # Flip the spin
    return s_new

def tabu_search(s0, N, energy_func):
    """Algorithm 2: TABUSEARCH (local improvement)."""
    
    # Get randomized parameters from Appendix F
    M = random.randint(N // 2, 3 * N // 2)
    # Paper typo M/10, M/50 likely swapped
    min_tenure = max(1, M // 50) 
    max_tenure = max(2, M // 10)

    s_best = list(s0)
    e_best = energy_func(s_best, N)
    s_current = list(s0)
    e_current = e_best
    
    # Tabu list stores the iteration 't' *until which* a move is tabu
    tabu_list = [0] * N 
    
    for t in range(1, M + 1):
        best_move_idx = -1
        best_move_energy = float('inf')
        
        # Check all 1-flip neighbors
        for i in range(N):
            s_neighbor = list(s_current)
            s_neighbor[i] *= -1
            e_neighbor = energy_func(s_neighbor, N)
            
            is_tabu = (tabu_list[i] > t)
            aspires = (e_neighbor < e_best) # Aspiration criterion
            
            if aspires:
                best_move_idx = i
                best_move_energy = e_neighbor
                break # Aspiration move is taken immediately
            elif not is_tabu:
                if e_neighbor < best_move_energy:
                    best_move_idx = i
                    best_move_energy = e_neighbor
        
        if best_move_idx == -1:
             # No valid moves, or stuck (e.g., all moves are tabu and non-aspiring)
             # Try to pick a random non-tabu move
             non_tabu_moves = [i for i, t_until in enumerate(tabu_list) if t_until <= t]
             if not non_tabu_moves:
                 break # Truly stuck
             best_move_idx = random.choice(non_tabu_moves)
             s_neighbor = list(s_current)
             s_neighbor[best_move_idx] *= -1
             best_move_energy = energy_func(s_neighbor, N)

        # Make the chosen move
        s_current[best_move_idx] *= -1
        e_current = best_move_energy
        
        # Add to tabu list
        tenure = random.randint(min_tenure, max_tenure)
        tabu_list[best_move_idx] = t + tenure
        
        # Update overall best
        if e_current < e_best:
            s_best = list(s_current)
            e_best = e_current
            
    return s_best, e_best

def select_parent(population, energies, tournament_size):
    """Selects a parent using tournament selection."""
    best_idx = -1
    best_energy = float('inf')
    
    for _ in range(tournament_size):
        idx = random.randrange(len(population))
        if energies[idx] < best_energy:
            best_energy = energies[idx]
            best_idx = idx
            
    return population[best_idx]

def run_classical_mts(N, quantum_samples, G_max=500):
    """
    **[USER MUST IMPLEMENT]** -> **[Implemented]**
    This is the second half of the algorithm (Algorithm 1).
    """
    print(f"Seeding classical MTS with {len(quantum_samples)} quantum samples.")
    
    # --- Define MTS parameters (from Sec III.B) ---
    K = 100         # Population size
    p_comb = 0.9    # Recombination probability
    p_mut = 1.0 / N # Mutation rate
    tournament_size = 2
    
    # --- 1. Initialize Population ---
    # Convert samples and find the best one
    best_quantum_seq = None
    best_quantum_energy = float('inf')
    
    for shot in quantum_samples:
        s = int_to_sequence(shot, N)
        e = calculate_energy(s, N)
        if e < best_quantum_energy:
            best_quantum_energy = e
            best_quantum_seq = s
            
    if best_quantum_seq is None:
        # Fallback: random sequence
        best_quantum_seq = [random.choice([-1, 1]) for _ in range(N)]
        best_quantum_energy = calculate_energy(best_quantum_seq, N)

    # "the lowest bitstring is replicated K times" (Sec III.B)
    population = [list(best_quantum_seq) for _ in range(K)]
    energies = [best_quantum_energy] * K
    
    s_star = list(best_quantum_seq)
    e_star = best_quantum_energy
    
    print(f"MTS initialized. Starting best energy: {e_star}")

    # --- 2. Run Memetic Algorithm Loop ---
    for G in range(G_max):
        # --- A. Selection and Recombination ---
        if random.random() < p_comb:
            p1 = select_parent(population, energies, tournament_size)
            p2 = select_parent(population, energies, tournament_size)
            c = combine(p1, p2, N)
        else:
            c = random.choice(population) # Select random individual
        
        # --- B. Mutation ---
        c = mutate(c, p_mut, N)
        
        # --- C. Local Improvement (Tabu Search) ---
        c, e_c = tabu_search(c, N, calculate_energy)
        
        # --- D. Check for new best ---
        if e_c < e_star:
            s_star = list(c)
            e_star = e_c
            print(f"Gen {G}: New best energy found: {e_star}")
        
        # --- E. Replace random individual ---
        idx_to_replace = random.randrange(K)
        population[idx_to_replace] = c
        energies[idx_to_replace] = e_c
        
    print("Classical MTS complete.")
    return s_star, e_star

# --- Main Execution ---
if __name__ == "__main__":
    # --- Config ---
    N = 4  # Sequence length (N=4 is tiny, N=37 is the paper's goal)
    T = 1.0          # Total evolution time
    num_trotter_steps = 10 # Number of discrete time steps
    n_shots = 100 # Number of quantum samples to take

    # 1. Run Quantum "Seeding" Stage
    # We must re-run the entire simulation for each shot,
    # as measurement collapses the state.
    print(f"--- Running Quantum Stage ({n_shots} shots) ---")
    quantum_samples = []
    for i in range(n_shots):
        print(f"\nRunning shot {i+1}/{n_shots}...")
        shot = run_dcqo_quantum_stage(N, num_trotter_steps, T)
        quantum_samples.append(shot)
        print(f"Shot {i+1} result: {shot}")
    
    print("\n--- Quantum Stage Complete ---")
    print(f"Collected samples: {quantum_samples}")

    # 2. Run Classical "Refinement" Stage
    print("\n--- Running Classical MTS Stage ---")
    best_sequence, best_energy = run_classical_mts(N, quantum_samples, G_max=100)
    
    print("\n--- Final Result ---")
    print(f"Best sequence found: {best_sequence}")
    print(f"Best energy found: {best_energy}")
