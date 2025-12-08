#!/usr/bin/env python
#
# LABS dcqd shortcut
# gemini25
#
# --- COMPLETE SCRIPT (Fix for live tqdm bar) ---
#
# Implements the Quantum-Enhanced Memetic Tabu Search (QE-MTS)
# from arXiv:2511.04553v1 
#
# This version uses pool.imap_unordered for a live-updating
# progress bar during parallel execution.
#
# *** MODIFIED: Uses QBDD backend instead of statevector ***
#
# --- OPTIMIZATION UPGRADES ---
#
# 1.  **Numba JIT:** All classical functions (energy, tabu search, etc.)
#     are decorated with `@numba.njit` for massive C-speed execution.
# 2.  **Delta-E Calculation:** `tabu_search` no longer recalculates the
#     full O(N^2) energy for every neighbor. It now calculates the
#     O(N) "delta energy" (change in energy), making the core
#     tabu search O(N^3) instead of O(N^4).
# 3.  **Parallelized MTS:** The classical MTS stage is now parallelized.
#     Each generation runs its `tabu_search` calls in parallel using
#     the multiprocessing pool, just like the quantum stage.
# 4.  **Better Seeding:** Implements the Appendix D strategy. Instead of
#     replicating the *single* best quantum shot, the initial
#     population is seeded with the *Top K* best (unique) shots
#     for greater initial diversity.
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
import numpy as np # Numba works best with NumPy arrays

### OPTIMIZATION: Import Numba
import numba

# ----------------------------------------------------------------------
# --- Quantum Circuit Definitions (Unchanged) ---
# ----------------------------------------------------------------------

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
# --- Main Algorithm Skeleton (Unchanged) ---
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
        #
        # *** MODIFICATION HERE ***
        # Changed from QrackSimulator(N) to use the QBDD backend.
        sim = QrackSimulator(N, isBinaryDecisionTree=True)
        #
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
# --- Classical MTS Implementation (OPTIMIZED) ---
# ----------------------------------------------------------------------

### OPTIMIZATION: Add Numba JIT decorator
# `nopython=True` is faster and enforces pure, fast code.
# `cache=True` saves the compiled function for future runs.
@numba.njit(cache=True)
def int_to_sequence(shot, N):
    """Converts an integer bitstring 'shot' into a +/-1 sequence."""
    # Use a NumPy array for Numba compatibility and speed
    s = np.empty(N, dtype=np.int8) 
    for i in range(N):
        if (shot >> i) & 1:
            s[i] = 1  # 1 bit -> +1 spin
        else:
            s[i] = -1 # 0 bit -> -1 spin
    return s

### OPTIMIZATION: Add Numba JIT decorator
@numba.njit(cache=True)
def calculate_energy(s, N):
    """Calculates the LABS energy for a +/-1 sequence 's' (Eq. 1)."""
    E = 0.0 # Use float for Numba
    for k in range(1, N): # k from 1 to N-1
        Ck = 0.0
        for i in range(N - k): # i from 0 to N-k-1 (paper i=1 to N-k)
            Ck += s[i] * s[i+k]
        E += Ck * Ck
    return E

### OPTIMIZATION: NEW function to get all Ck values (for Delta-E)
@numba.njit(cache=True)
def get_all_autocorrelations(s, N):
    """Calculates and returns all Ck values for a sequence."""
    # Ck_list[0] will be C1, Ck_list[N-2] will be C(N-1)
    Ck_list = np.empty(N - 1, dtype=np.float64)
    for k in range(1, N): # k from 1 to N-1
        Ck = 0.0
        for i in range(N - k):
            Ck += s[i] * s[i+k]
        Ck_list[k - 1] = Ck
    return Ck_list

### OPTIMIZATION: NEW Delta-E calculation function
@numba.njit(cache=True)
def calculate_delta_energy(s_old, Ck_list_old, N, flip_index):
    """
    Calculates the new energy and new Ck list in O(N) time 
    after flipping a single spin at `flip_index`.
    """
    ### FIX: Use np.copy() instead of .copy()
    s_new = np.copy(s_old)
    
    ### FIX: (Proactive) Change from *= to = -
    # s_new[flip_index] *= -1 
    s_new[flip_index] = -s_new[flip_index]
    
    ### FIX: Use np.copy() instead of .copy()
    Ck_list_new = np.copy(Ck_list_old)
    E_new = 0.0
    
    # This loop is O(N)
    for k in range(1, N): # k from 1 to N-1
        Ck_old = Ck_list_old[k - 1]
        
        # Calculate the change (delta) for this Ck
        delta_Ck = 0.0
        
        # Check if the flipped spin contributes to Ck
        # This happens in two cases:
        
        # 1. As s_i (the first term in s_i * s_{i+k})
        #    This is when i = flip_index
        i = flip_index
        if i < N - k:
            # The old term was s_old[i] * s_old[i+k]
            # The new term is s_new[i] * s_new[i+k]
            # Since s_new[i+k] == s_old[i+k], the change is
            # (s_new[i] - s_old[i]) * s_old[i+k]
            # = (-2 * s_old[i]) * s_old[i+k]
            delta_Ck -= 2 * s_old[i] * s_old[i+k]
            
        # 2. As s_{i+k} (the second term in s_i * s_{i+k})
        #    This is when i+k = flip_index, so i = flip_index - k
        i = flip_index - k
        if i >= 0 and i < (N - k):
            # The old term was s_old[i] * s_old[i+k]
            # The new term is s_new[i] * s_new[i+k]
            # Since s_new[i] == s_old[i], the change is
            # s_old[i] * (s_new[i+k] - s_old[i+k])
            # = s_old[i] * (-2 * s_old[i+k])
            delta_Ck -= 2 * s_old[i] * s_old[i+k]

        Ck_new = Ck_old + delta_Ck
        Ck_list_new[k - 1] = Ck_new
        E_new += Ck_new * Ck_new
        
    return s_new, E_new, Ck_list_new

### OPTIMIZATION: Add Numba JIT decorator
@numba.njit(cache=True)
def combine(p1, p2, N):
    """Algorithm 3: COMBINE (single-point crossover)."""
    k = random.randint(1, N - 1) # Cut point k in {1, ..., N-1}
    # p1[0...k-1] + p2[k...N-1]
    
    # Numba-friendly array concatenation
    c = np.empty(N, dtype=np.int8)
    c[:k] = p1[:k]
    c[k:] = p2[k:]
    return c

### OPTIMIZATION: Add Numba JIT decorator
@numba.njit(cache=True)
def mutate(s, p_mut, N):
    """Algorithm 3: MUTATE (bit-flip mutation)."""
    ### FIX: Use np.copy() instead of .copy()
    s_new = np.copy(s) # Make a copy
    for i in range(N):
        if random.random() < p_mut:
            ### FIX: Change from *= to = -
            # s_new[i] *= -1 # Flip the spin
            s_new[i] = -s_new[i]
    return s_new

### OPTIMIZATION: Add Numba JIT decorator and use Delta-E
@numba.njit(cache=True)
def tabu_search(s0, N):
    """Algorithm 2: TABUSEARCH (local improvement)."""
    
    # Get randomized parameters from Appendix F
    M_min = N // 2
    M_max = 3 * N // 2
    # Numba-compatible randint: random.randrange(start, stop)
    M = random.randrange(M_min, M_max + 1) 
    
    # Note: Paper typo M/10, M/50 likely swapped
    min_tenure = max(1, M // 50)  
    max_tenure = max(2, M // 10)

    # --- Initial calculation ---
    ### FIX: Use np.copy() instead of .copy()
    s_best = np.copy(s0)
    # This is the *only* full O(N^2) energy calculation
    Ck_list_best = get_all_autocorrelations(s_best, N)
    e_best = 0.0
    for Ck in Ck_list_best:
        e_best += Ck * Ck
    
    ### FIX: Use np.copy() instead of .copy()
    s_current = np.copy(s_best)
    e_current = e_best
    ### FIX: Use np.copy() instead of .copy()
    Ck_list_current = np.copy(Ck_list_best)
    
    # Tabu list stores the iteration 't' *until which* a move is tabu
    tabu_list = np.zeros(N, dtype=np.int32) 
    
    for t in range(1, M + 1):
        best_move_idx = -1
        best_move_energy = np.inf
        # Store these to avoid re-calculation
        best_move_s_neighbor = s_current 
        best_move_Ck_list_neighbor = Ck_list_current

        # Check all 1-flip neighbors
        # ### OPTIMIZATION: This loop is now O(N*N) = O(N^2)
        # ### instead of O(N*N^2) = O(N^3)
        for i in range(N):
            
            # ### OPTIMIZATION: Calculate neighbor energy in O(N)
            s_neighbor, e_neighbor, Ck_list_neighbor = calculate_delta_energy(
                s_current, Ck_list_current, N, i
            )
            
            is_tabu = (tabu_list[i] > t)
            aspires = (e_neighbor < e_best) # Aspiration criterion
            
            if aspires:
                best_move_idx = i
                best_move_energy = e_neighbor
                best_move_s_neighbor = s_neighbor
                best_move_Ck_list_neighbor = Ck_list_neighbor
                break # Aspiration move is taken immediately
            elif not is_tabu:
                if e_neighbor < best_move_energy:
                    best_move_idx = i
                    best_move_energy = e_neighbor
                    best_move_s_neighbor = s_neighbor
                    best_move_Ck_list_neighbor = Ck_list_neighbor
            
        if best_move_idx == -1:
            # No valid moves, or stuck. Pick a random non-tabu move.
            
            ### FIX: Replace random.choice(list) with Numba-friendly loop
            # 1. Count non-tabu moves
            non_tabu_count = 0
            for i in range(N):
                if tabu_list[i] <= t:
                    non_tabu_count += 1
            
            if non_tabu_count == 0:
                break # Truly stuck
            
            # 2. Pick a random *index* from 0 to count-1
            target_choice_idx = random.randrange(non_tabu_count)
            
            # 3. Find the N-th non-tabu move
            current_choice_count = 0
            for i in range(N):
                if tabu_list[i] <= t:
                    if current_choice_count == target_choice_idx:
                        best_move_idx = i
                        break
                    current_choice_count += 1
            ### END FIX
            
            # We must have found an index, so now we calculate its properties
            s_neighbor, e_neighbor, Ck_list_neighbor = calculate_delta_energy(
                s_current, Ck_list_current, N, best_move_idx
            )
            best_move_energy = e_neighbor
            best_move_s_neighbor = s_neighbor
            best_move_Ck_list_neighbor = Ck_list_neighbor

        # Make the chosen move
        s_current = best_move_s_neighbor
        e_current = best_move_energy
        Ck_list_current = best_move_Ck_list_neighbor
        
        # Add to tabu list (randomized tenure)
        tenure = random.randrange(min_tenure, max_tenure + 1)
        tabu_list[best_move_idx] = t + tenure
        
        # Update overall best (incumbent)
        if e_current < e_best:
            ### FIX: Use np.copy() instead of .copy()
            s_best = np.copy(s_current)
            e_best = e_current
            ### FIX: Use np.copy() instead of .copy()
            Ck_list_best = np.copy(Ck_list_current) # Store Ck's for the best
            
    return s_best, e_best

### OPTIMIZATION: Add Numba JIT decorator
@numba.njit(cache=True)
def select_parent(population, energies, tournament_size):
    """Selects a parent using tournament selection (from Sec III.B)."""
    best_idx = -1
    best_energy = np.inf
    
    pop_size = len(population)
    for _ in range(tournament_size):
        idx = random.randrange(pop_size)
        if energies[idx] < best_energy:
            best_energy = energies[idx]
            best_idx = idx
            
    # Return a *copy* to avoid mutation issues in parallel code
    ### FIX: Use np.copy() instead of .copy()
    return np.copy(population[best_idx])

### OPTIMIZATION: NEW worker function for parallel MTS
def classical_worker(args):
    """
    A single-shot worker for the classical MTS stage.
    This function is designed to be called by pool.imap_unordered.
    
    It performs one full "child generation" step:
    1. Select parents
    2. Combine/Mutate
    3. Run Tabu Search (the expensive part)
    """
    # Unpack arguments
    population, energies, K, N, p_comb, p_mut, tournament_size = args
    
    # --- A. Selection and Recombination ---
    if random.random() < p_comb:
        p1 = select_parent(population, energies, tournament_size)
        p2 = select_parent(population, energies, tournament_size)
        c = combine(p1, p2, N)
    else:
        # Select random individual
        rand_idx = random.randrange(K)
        ### FIX: Use np.copy() instead of .copy()
        c = np.copy(population[rand_idx])
    
    # --- B. Mutation ---
    c = mutate(c, p_mut, N) #
    
    # --- C. Local Improvement (Tabu Search) ---
    # This is the main computational bottleneck, now run in parallel.
    c_improved, e_c = tabu_search(c, N) 
    
    return c_improved, e_c

### OPTIMIZATION: Modified MTS main function
def run_classical_mts(N, quantum_samples, G_max, pool):
    """
    This is the second half of the algorithm (Algorithm 1).
    It takes the quantum samples and uses them to seed the
    classical memetic tabu search.
    
    --- OPTIMIZED ---
    1. Uses "Top K" seeding strategy from Appendix D.
    2. Runs the main generation loop in parallel using the `pool`.
    """
    print(f"Seeding classical MTS with {len(quantum_samples)} quantum samples.")
    
    # --- Define MTS parameters (from Sec III.B) ---
    K = 100     # Population size
    p_comb = 0.9  # Recombination probability
    p_mut = 1.0 / N # Mutation rate
    tournament_size = 2 #
    
    # --- 1. Initialize Population (OPTIMIZED Seeding) ---
    
    # Calculate energies for all quantum samples
    # We must use the non-jitted `calculate_energy` here
    # because `pool.imap` can't send jobs to JIT-compiled functions
    # directly in the main thread.
    
    # Convert shots to sequences first
    quantum_sequences = [int_to_sequence(shot, N) for shot in quantum_samples]
    
    print("Calculating energies for initial quantum seeds...")
    initial_energies = [calculate_energy(s, N) for s in quantum_sequences]
    
    # Pair them up and find the best
    seq_energy_pairs = sorted(zip(initial_energies, quantum_sequences), key=lambda x: x[0])
    
    ### FIX: Use np.copy() instead of .copy()
    s_star = np.copy(seq_energy_pairs[0][1])
    e_star = seq_energy_pairs[0][0]
    
    # Build the Top-K population
    population_list = []
    energies_list = []
    
    # Use the best K unique sequences
    unique_seqs = {tuple(s): e for e, s in reversed(seq_energy_pairs)}
    best_unique_seqs = sorted(unique_seqs.items(), key=lambda x: x[1])
    
    print(f"Found {len(best_unique_seqs)} unique quantum seeds.")
    
    for i in range(K):
        # Loop over the best unique sequences
        
        ### FIX: Swapped e and s_tuple
        s_tuple, e = best_unique_seqs[i % len(best_unique_seqs)]
        
        population_list.append(np.array(s_tuple, dtype=np.int8))
        energies_list.append(e)

    # Convert to NumPy arrays for Numba
    population = np.array(population_list)
    energies = np.array(energies_list, dtype=np.float64)

    print(f"MTS initialized with Top-K seeds. Starting best energy: {e_star}")

    # --- 2. Run Memetic Algorithm Loop (Alg 1) ---
    # ### OPTIMIZATION: This loop is now parallelized
    
    for G in range(G_max):
        
        # Prepare arguments for the parallel workers
        # We must pass all data the workers need.
        # This creates K identical argument tuples.
        args_iterable = [
            (population, energies, K, N, p_comb, p_mut, tournament_size)
        ] * K
        
        new_population_list = []
        new_energies_list = []
        
        # Run K `classical_worker` jobs in parallel
        # We don't use tqdm here, as it conflicts with the quantum bar
        # A simple print is better.
        if (G+1) % 10 == 0 or G == 0: # Print on first and every 10th
            print(f"--- MTS Generation {G+1}/{G_max} ---")
            
        results = pool.map(classical_worker, args_iterable)
        
        # Process results and find the new best
        for s_new, e_new in results:
            new_population_list.append(s_new)
            new_energies_list.append(e_new)
            
            if e_new < e_star:
                ### FIX: Use np.copy() instead of .copy()
                s_star = np.copy(s_new)
                e_star = e_new
                print(f"Gen {G+1}/{G_max}: New best energy found: {e_star}")
        
        # The new generation replaces the old one
        population = np.array(new_population_list)
        energies = np.array(new_energies_list)

    print("Classical MTS complete.")
    return s_star, e_star

# --- Helper function for multiprocessing (Unchanged) ---
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
    parser.add_argument('-N', '--N', type=int, default=10,
                        help='Sequence length (default: 10)')
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
        print(f"Warning: N={N} is too small. Setting N=4.")
        N = 4
    
    # --- JIT WARM-UP ---
    # We must "warm up" the Numba functions before starting the pool
    # to avoid each parallel process recompiling them.
    print("Warming up JIT-compiled functions...")
    try:
        s_warm = int_to_sequence(0, N)
        e_warm = calculate_energy(s_warm, N)
        ck_warm = get_all_autocorrelations(s_warm, N)
        calculate_delta_energy(s_warm, ck_warm, N, 0)
        s_warm_2 = combine(s_warm, s_warm, N)
        s_warm_3 = mutate(s_warm, 0.1, N)
        tabu_search(s_warm, N)
        pop_warm = np.array([s_warm, s_warm_2])
        eng_warm = np.array([e_warm, e_warm])
        select_parent(pop_warm, eng_warm, 2)
        print("JIT warm-up complete.")
    except Exception as e:
        print(f"JIT warm-up failed: {e}")
        print("This can happen on first run. Subsequent runs may be faster.")

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

    # `multiprocessing.Pool()` automatically uses all available CPU cores.
    # The 'with' statement handles setup and cleanup.
    # We create the pool *once* and pass it to both stages.
    with multiprocessing.Pool() as pool:
    
        # 1. Run Quantum "Seeding" Stage (IN PARALLEL)
        print(f"--- Running Quantum Stage ({n_shots} shots in parallel) ---")
        
        # Create an iterable of arguments for each shot.
        # All shots use the same (N, num_trotter_steps, T) arguments.
        args_iterable = itertools.repeat((N, num_trotter_steps, T), n_shots)
        
        quantum_samples = list(tqdm(
            pool.imap_unordered(worker_unpack_args, args_iterable), 
            total=n_shots
        ))
        
        print("\n--- Quantum Stage Complete ---")

        # 2. Run Classical "Refinement" Stage
        # ### OPTIMIZATION: Pass the pool to the classical stage
        print("\n--- Running Classical MTS Stage (in parallel) ---")
        best_sequence, best_energy = run_classical_mts(
            N, quantum_samples, G_max=G_max, pool=pool
        )
        
    print("\n--- Final Result ---")
    # Convert best sequence back to a list for nice printing
    print(f"Best sequence found: {list(best_sequence)}")
    print(f"Best energy found: {best_energy}")
