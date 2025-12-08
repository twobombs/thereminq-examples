#!/usr/bin/env python3
#
# pareto-front PoC
# gemini25
#
# --- REFACTORED FOR SCALABILITY (CPU-BOUND) ---
# This version avoids the 2**N memory bottleneck by
# generating solutions on-the-fly in each worker.
#
# --- FIX (11/08/25): Corrected pool.map call to avoid TypeError ---

import numpy as np
import random
import argparse
import multiprocessing
# --- MODIFIED: We no longer need 'partial' ---
# from functools import partial

# --- 1. Define the Objective Functions ---
# (No changes in this section)

def objective_1_max_ones(solution):
    """F1: Counts the number of 1s in the bit string."""
    return sum(solution)

def objective_2_max_flips(solution):
    """F2: Counts the number of flips (01 or 10)"""
    flips = 0
    for i in range(len(solution) - 1):
        if solution[i] != solution[i+1]:
            flips += 1
    return flips

# --- 2. The Solver (MODIFIED FOR SCALABILITY) ---
# (No changes in this section)

def find_optimal_solution(weight_1, weight_2, problem_size):
    """
    --- MODIFIED: This is the solver ---
    It now loops from 0 to 2**N and generates solutions on the fly
    instead of reading from a global list.
    """
    best_solution = None
    best_score = -float('inf')

    # Loop through all possible 2**N solutions
    for i in range(2**problem_size):
        
        # Generate the solution on the fly
        bin_str = bin(i)[2:].zfill(problem_size)
        solution = [int(bit) for bit in bin_str]
        
        # Evaluate this solution
        f1_score = objective_1_max_ones(solution)
        f2_score = objective_2_max_flips(solution)
        combined_score = (weight_1 * f1_score) + (weight_2 * f2_score)

        if combined_score > best_score:
            best_score = combined_score
            best_solution = (tuple(solution), f1_score, f2_score)
    
    return best_solution


# --- 3. The Pareto Filtering Function ---
# (No changes in this section)

def find_pareto_front(all_found_solutions):
    """
    Filters a set of solutions to find the non-dominated Pareto front.
    """
    print(f"\nFiltering {len(all_found_solutions)} unique solutions to find the front...")
    
    solution_points = {}
    for sol_tuple, f1, f2 in all_found_solutions:
        solution_points[(f1, f2)] = sol_tuple

    pareto_front_points = set()

    for p1 in solution_points.keys():
        f1_p1, f2_p1 = p1
        is_dominated = False
        
        for p2 in solution_points.keys():
            if p1 == p2:
                continue
            
            f1_p2, f2_p2 = p2
            
            # Check for dominance
            if (f1_p2 >= f1_p1 and f2_p2 >= f2_p1) and \
               (f1_p2 > f1_p1 or f2_p2 > f2_p1):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_front_points.add(p1)
            
    return sorted(list(pareto_front_points)), solution_points

# --- 4. Helper Function for Parallelism ---
# (No changes in this section - still accepts problem_size)

def run_one_weighting(problem_size):
    """
    This function now correctly accepts 'problem_size'
    from the iterable provided to pool.map.
    """
    # Generate random, normalized weights
    w1 = np.random.random()
    w2 = 1.0 - w1
    
    # Run the solver by calling the top-level function
    # and passing problem_size to it.
    solution = find_optimal_solution(w1, w2, problem_size)
    return solution

# --- 5. The Main Workflow (MODIFIED) ---

def main(problem_size, num_weightings):
    """
    Main execution function, parallelized and memory-efficient.
    """
    
    print(f"--- Running with Problem Size: {problem_size}, Weightings: {num_weightings} ---")
    
    print(f"Running solver for {num_weightings} random weightings IN PARALLEL...")
    
    # Create a pool of worker processes.
    with multiprocessing.Pool() as pool:
        
        # --- MODIFIED: This is the fix ---
        # Instead of using 'partial', we create an iterable (a list)
        # that contains 'problem_size' repeated 'num_weightings' times.
        iterable = [problem_size] * num_weightings
        
        # 'map' will now call run_one_weighting(problem_size)
        # for each item in the list, which matches the
        # function's signature.
        all_found_solutions = pool.map(run_one_weighting, iterable)
        
    # Filter all found solutions to get the Pareto front
    valid_solutions = {s for s in all_found_solutions if s is not None}
    
    pareto_front, all_points = find_pareto_front(valid_solutions)

    print("\n--- Pareto Front Found ---")
    print("Format: (F1 Score, F2 Score)")
    for point in pareto_front:
        # Show an example solution for each point on the front
        print(f"  {point}  (e.g., solution: {all_points[point]})")

if __name__ == "__main__":
    # --- 6. Add CLI Argument Parsing ---
    # (No changes in this section)
    
    parser = argparse.ArgumentParser(
        description="Find the Pareto front for a toy multi-objective problem."
    )
    
    parser.add_argument(
        "-p", "--problem_size",
        type=int,
        default=10,
        help="The number of bits in the solution string (default: 10)."
    )
    
    parser.add_argument(
        "-w", "--num_weightings",
        type=int,
        default=200,
        help="The number of random weightings to sample (default: 200)."
    )
    
    args = parser.parse_args()
    
    # (Warning for large problem sizes remains)
    if args.problem_size > 22:
        print(f"WARNING: Problem size {args.problem_size} is very large.")
        print(f"Each of the {args.num_weightings} workers will have to check 2^{args.problem_size} solutions.")
        print(f"(That's ~{2**args.problem_size:,} solutions per worker!)")
        print("This may take a very long time!")
    
    main(args.problem_size, args.num_weightings)
