#
# LABS dcqd shortcut
# gemini25
#
# --- COMPLETE SCRIPT (Fix for live tqdm bar) ---
#
# Implements the Quantum-Enhanced Memetic Tabu Search (QE-MTS)
# from arXiv:2511.04553v1 [source: 2]
#
# This version uses pool.imap_unordered for a live-updating
# progress bar during parallel execution.
#

import math
import os # Used for os.devnull and os.dup2
from pyqrack import QrackSimulator
import random # Needed for classical MTS
import argparse # For CLI input
import multiprocessing
import itertools
from tqdm import tqdm # For a nice progress bar
import sys # Needed for C-level flush

def apply_r_pauli_string(sim, qubits, pauli_string, angle):
    """
    Applies a generalized Pauli rotation R_P(angle) = exp(-i * angle * P)
    where P is a Pauli string like "YZZZ". (Note paper's definition in Eq. B2)
    
    This is the core building block for the DCQO circuit.
    
    Args:
        sim (QrackSimulator): The simulator instance.
        qubits (list): A list of qubit indices, e.g., [q_i, q_l, q_k, q_kl].
        pauli_string (str): The Pauli string, e.g., "YZZZ".
        angle (float): The rotation angle (x, from paper's R_P(x)).
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
    #
    # *** CRITICAL FIX ***
    # The paper's definition (Eq. B2) is R_P(x) = exp(-ixP).
    # The standard PyQrack gate is R_Z(phi) = exp(-i*phi*Z/2).
    # To implement the paper's rotation, we must set phi = 2*x.
    # 'angle' here is 'x' from the paper.
    sim.r(3, 2 * angle, target_q)

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
    Calculates the schedule based on the "impulse regime" and a
    linear schedule for lambda(t), as suggested by the paper (Sec III.A)
    and citation [15].
    
    This implements the 'theta' required for the rotation gates in Eq. (B3).
    
    NOTE: This uses a simple linear ansatz for alpha_1(t) = 1 - lambda_t,
    as the derivation in Appendix B is non-trivial and appears
    to be incomplete (references undefined h^b term).
    This ansatz is a practical choice.
    """
    lambda_t = t / T
    dot_lambda = 1.0 / T      # Derivative of lambda(t)
    
    # This is the ansatz for the counterdiabatic coefficient
    alpha_1 = 1.0 - lambda_t  
    
    # Delta_t is the time step for a single Trotter step
    delta_t = T / num_trotter_steps
    
    # This is theta(k*Delta_t) from Appendix B
    # theta(t) = Delta_t * alpha_1(t) * dot_lambda(t)
    theta = delta_t * alpha_1 * dot_lambda
    
    return theta

def get_problem_hamiltonian_terms(N):
    """
    This function parses the problem Hamiltonian H_f (Eq. 2)
    and returns lists of the qubit indices for all 2-body
    and 4-body terms, as derived for the CD term O_1 in Eq. 7.
    
    All loops are converted from the paper's 1-based indexing to
    Python's 0-based indexing.
    """
    two_body_terms = []
    four_body_terms = []

    # 2-body terms from Eq (7)
    # Paper: i from 1 to N-2. k from 1 to floor((N-i)/2)
    # 0-indexed: i from 0 to N-3. k from 1 to floor((N-1-i)/2)
    for i in range(N - 2):
        for k in range(1, (N - 1 - i) // 2 + 1):
            two_body_terms.append((i, i + k))

    # 4-body terms from Eq (7)
    # Paper: i from 1 to N-3. l from 1 to floor((N-i-1)/2). k from l+1 to N-i-l
    # 0-indexed: i from 0 to N-4. l from 1 to floor((N-i-2)/2). k from l+1 to N-i-l-1
    # Note: range(start, stop) runs to stop-1.
    # The paper's k loop (1-indexed) runs up to N-i_p-l.
    # Our k loop (1-indexed) must run up to N-(i_c+1)-l.
    # So the 'stop' argument for range() should be N-i_c-l.
    for i in range(N - 3):
        for l in range(1, (N - i - 2) // 2 + 1):
            # This range is correct for the paper's logic
            for k in range(l + 1, N - i - l): 
                four_body_terms.append((i, i + l, i + k, i + k + l))

    return two_body_terms, four_body_terms

def run_dcqo_quantum_stage(N, num_trotter_steps, T):
    """
    Runs the full quantum simulation part of the QE-MTS algorithm
    and returns a SINGLE measurement sample.
    """
    
    # --- Start of output suppression ---
    # This is a low-level fix to silence C++ libraries
    # that print directly to file descriptors.
    
    # Open /dev/null
    fnull = os.open(os.devnull, os.O_WRONLY)
    # Save a copy of the original file descriptors
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    # Redirect stdout (1) and stderr (2) to /dev/null
    os.dup2(fnull, 1)
    os.dup2(fnull, 2)
    # ---

    try:
        # This call will now be silent
        sim = QrackSimulator(N)
    finally:
        # --- Restore original stdout/stderr ---
        # Flush any C-level buffers
        sys.stdout.flush()
        sys.stderr.flush()
        # Close the /dev/null file descriptor
        os.close(fnull)
        # Restore the original file descriptors
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        # Close the saved copies
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        # ---
    
    # 1. Get the terms of the LABS Hamiltonian
    two_body_terms, four_body_terms = get_problem_hamiltonian_terms(N)

    # 2. Initialize state to |+> (ground state of H_i)
    # We can do this by applying a Hadamard (H) gate to all qubits.
    for i in range(N):
        sim.h(i)
    
    # 3. Perform the "Digitized" (Trotterized) time evolution
    for k_step in range(1, num_trotter_steps + 1):
        t = (k_step / num_trotter_steps) * T
        
        # This is theta(k*Delta_t) from Appendix B, Eq. (B3)
        theta = calculate_theta_and_alpha(t, T, num_trotter_steps)
        
        # Set transverse fields h_j^x = -1 (from Sec III.A)
        h_x = -1.0

        # Apply 2-body rotations (from Eq. B3)
        for (i, i_k) in two_body_terms:
            # These are R_YZ(4*theta*h_i) and R_ZY(4*theta*h_k)
            apply_r_pauli_string(sim, [i, i_k], "YZ", 4 * theta * h_x)
            apply_r_pauli_string(sim, [i, i_k], "ZY", 4 * theta * h_x)

        # Apply 4-body rotations (from Eq. B3)
        # (i, i_l, i_k, i_kl) corresponds to (i, i+l, i+k, i+k+l)
        for (i, i_l, i_k, i_kl) in four_body_terms:
            qubits = [i, i_l, i_k, i_kl]
            
            # This directly implements the terms in Eq. (B3)
            # R_YZZZ(8*theta*h_i)
            apply_r_pauli_string(sim, qubits, "YZZZ", 8 * theta * h_x)
            # R_ZYZZ(8*theta*h_i+l)
            apply_r_pauli_string(sim, qubits, "ZYZZ", 8 * theta * h_x)
            # R_ZZYZ(8*theta*h_i+k)
            apply_r_pauli_string(sim, qubits, "ZZYZ", 8 * theta * h_x)
            # R_ZZZY(8*theta*h_i+k+l)
            apply_r_pauli_string(sim, qubits, "ZZZY", 8 * theta * h_x)
    
    # 4. Measure all qubits to get a bitstring sample
    shot = sim.m_all()  
    
    # No sim.destroy() needed, Python's GC will handle it
    
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
        for i in range(N - k): # i from 0 to N-k-1 (paper i=1 to N-k)
            Ck += s[i] * s[i+k]
        E += Ck * Ck
    return E

def combine(p1, p2, N):
    """Algorithm 3: COMBINE (single-point crossover)."""
    k = random.randint(1, N - 1) # Cut point k in {1, ..., N-1}
    # p1[0...k-1] + p2[k...N-1]
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
    M = random.randint(N // 2, 3 * N // 2) #
    
    # Note: Paper typo M/10, M/50 likely swapped
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
             # No valid moves, or stuck
             # Try to pick a random non-tabu move to escape
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
        
        # Add to tabu list (randomized tenure)
        tenure = random.randint(min_tenure, max_tenure)
        tabu_list[best_move_idx] = t + tenure
        
        # Update overall best (incumbent)
        if e_current < e_best:
            s_best = list(s_current)
            e_best = e_current
            
    return s_best, e_best

def select_parent(population, energies, tournament_size):
    """Selects a parent using tournament selection (from Sec III.B)."""
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
    This is the second half of the algorithm (Algorithm 1).
    It takes the quantum samples and uses them to seed the
    classical memetic tabu search.
    """
    print(f"Seeding classical MTS with {len(quantum_samples)} quantum samples.")
    
    # --- Define MTS parameters (from Sec III.B) ---
    K = 100       # Population size
    p_comb = 0.9  # Recombination probability
    p_mut = 1.0 / N # Mutation rate
    tournament_size = 2 #
    
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
        # Fallback: random sequence if no samples
        print("Warning: No quantum samples provided, using random seed.")
        best_quantum_seq = [random.choice([-1, 1]) for _ in range(N)]
        best_quantum_energy = calculate_energy(best_quantum_seq, N)

    # "the lowest bitstring is replicated K times" (Sec III.B)
    population = [list(best_quantum_seq) for _ in range(K)]
    energies = [best_quantum_energy] * K
    
    s_star = list(best_quantum_seq)
    e_star = best_quantum_energy
    
    print(f"MTS initialized. Starting best energy: {e_star}")

    # --- 2. Run Memetic Algorithm Loop (Alg 1) ---
    for G in range(G_max):
        # --- A. Selection and Recombination ---
        if random.random() < p_comb:
            p1 = select_parent(population, energies, tournament_size)
            p2 = select_parent(population, energies, tournament_size)
            c = combine(p1, p2, N)
        else:
            c = list(random.choice(population)) # Select random individual
        
        # --- B. Mutation ---
        c = mutate(c, p_mut, N) #
        
        # --- C. Local Improvement (Tabu Search) ---
        c, e_c = tabu_search(c, N, calculate_energy) #
        
        # --- D. Check for new best ---
        if e_c < e_star:
            s_star = list(c)
            e_star = e_c
            print(f"Gen {G+1}/{G_max}: New best energy found: {e_star}")
        
        # --- E. Replace random individual ---
        idx_to_replace = random.randrange(K)
        population[idx_to_replace] = c
        energies[idx_to_replace] = e_c
        
    print("Classical MTS complete.")
    return s_star, e_star

# --- Helper function for multiprocessing ---
# This wrapper is needed so that pool.imap_unordered
# can call our main function, which takes multiple arguments.
def worker_unpack_args(args):
    """Helper function to unpack arguments for pool.imap_unordered."""
    return run_dcqo_quantum_stage(*args)

# --- Main Execution ---
if __name__ == "__main__":
    # --- Config ---
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Quantum-Enhanced Memetic Tabu Search for LABS.")
    parser.add_argument('-N', '--N', type=int, default=4,
                        help='Sequence length (default: 4)')
    parser.add_argument('-s', '--n_shots', type=int, default=100,
                        help='Number of quantum shots to sample (default: 100)')
    parser.add_argument('-g', '--generations', type=int, default=100,
                        help='Max generations for classical MTS (default: 100)')
    
    args = parser.parse_args()

    # Use arguments from CLI or defaults
    N = args.N
    n_shots = args.n_shots
    G_max = args.generations
    T = 1.0       # Total evolution time (arbitrary for this schedule)
    num_trotter_steps = 10 # Number of discrete time steps
    
    # Basic validation
    if N < 4:
        print(f"Warning: N={N} is. Setting N=4.")
        N = 4
    
    # --- Calculate Hamiltonian Complexity ---
    print("Calculating Hamiltonian complexity...")
    two_body_terms, four_body_terms = get_problem_hamiltonian_terms(N)
    num_2_body = len(two_body_terms)
    num_4_body = len(four_body_terms)
    # ---
    
    print("--- QE-MTS Configuration ---")
    print(f"Sequence Length (N): {N}")
    print(f"Hamiltonian Complexity:")
    print(f"  - 2-Body Terms: {num_2_body}")
    print(f"  - 4-Body Terms: {num_4_body}")
    print(f"Quantum Shots (n_shots): {n_shots}")
    print(f"Classical Generations (G_max): {G_max}")
    print(f"Total Evolution Time (T): {T}")
    print(f"Trotter Steps: {num_trotter_steps}")
    print("------------------------------")

    # 1. Run Quantum "Seeding" Stage (IN PARALLEL)
    print(f"--- Running Quantum Stage ({n_shots} shots in parallel) ---")
    
    # Create an iterable of arguments for each shot.
    # All shots use the same (N, num_trotter_steps, T) arguments.
    args_iterable = itertools.repeat((N, num_trotter_steps, T), n_shots)

    # `multiprocessing.Pool()` automatically uses all available CPU cores.
    # The 'with' statement handles setup and cleanup.
    with multiprocessing.Pool() as pool:
        #
        # --- FIX ---
        # Changed from pool.starmap(...) to pool.imap_unordered(...)
        # starmap waits for ALL results (no live bar)
        # imap_unordered yields results as they finish (live bar)
        # We use the 'worker_unpack_args' helper to unpack the argument tuple.
        #
        quantum_samples = list(tqdm(
            pool.imap_unordered(worker_unpack_args, args_iterable), 
            total=n_shots
        ))
    
    print("\n--- Quantum Stage Complete ---")
    # print(f"Collected samples: {quantum_samples}") # Too verbose

    # 2. Run Classical "Refinement" Stage
    print("\n--- Running Classical MTS Stage ---")
    best_sequence, best_energy = run_classical_mts(N, quantum_samples, G_max=G_max)
    
    print("\n--- Final Result ---")
    print(f"Best sequence found: {best_sequence}")
    print(f"Best energy found: {best_energy}")
