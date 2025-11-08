#!/usr/bin/env python3
#
# pareto-front PoC
# gemini25
#
# --- NOW WITH MULTIPROCESSING (FIXED) ---

import numpy as np
import random
import argparse
import multiprocessing
# We no longer need 'partial'
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

# --- 2. The Solver (MODIFIED FOR PARALLELISM) ---

# --- MODIFIED: 'all_solutions' is now a global variable ---
# This list will be created by the main process and
# inherited by the child worker processes.
all_solutions = []

def initialize_solver(problem_size):
    """
    --- NEW FUNCTION ---
    Populates the GLOBAL 'all_solutions' list.
    This is run once by the main process *before* forking.
    """
    global all_solutions
    
    for i in range(2**problem_size):
        bin_str = bin(i)[2:].zfill(problem_size)
        solution = [int(bit) for bit in bin_str]
        all_solutions.append(solution)
    
    print(f"Solver initialized with {len(all_solutions)} possible solutions.")

def find_optimal_solution_global(weight_1, weight_2):
    """
    --- MODIFIED: This is the solver, moved to the top level ---
    This function can now be pickled and sent to workers.
    It reads from the global 'all_solutions' list.
    """
    global all_solutions # Access the global list
    best_solution = None
    best_score = -float('inf')

    # The list was already populated by the main process
    for solution in all_solutions:
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
            
            if (f1_p2 >= f1_p1 and f2_p2 >= f2_p1) and \
               (f1_p2 > f1_p1 or f2_p2 > f2_p1):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_front_points.add(p1)
            
    return sorted(list(pareto_front_points)), solution_points

# --- 4. Helper Function for Parallelism (MODIFIED) ---

def run_one_weighting(_):
    """
    --- MODIFIED: This function no longer takes a solver argument ---
    This is the function that each worker process will run.
    The '_' is a placeholder for the item from range().
    """
    # Generate random, normalized weights
    w1 = np.random.random()
    w2 = 1.0 - w1
    
    # Run the solver by calling the new GLOBAL function
    solution = find_optimal_solution_global(w1, w2)
    return solution

# --- 5. The Main Workflow (MODIFIED) ---

def main(problem_size, num_weightings):
    """
    Main execution function, now PARALLELIZED correctly.
    """
    
    print(f"--- Running with Problem Size: {problem_size}, Weightings: {num_weightings} ---")
    
    # --- MODIFIED: Call the initializer function ONCE ---
    # This populates the global 'all_solutions' list
    initialize_solver(problem_size)
    
    print(f"Running solver for {num_weightings} random weightings IN PARALLEL...")
    
    # Create a pool of worker processes.
    with multiprocessing.Pool() as pool:
        
        # --- MODIFIED: Map the simplified 'run_one_weighting' function ---
        # Each worker will call this function, which in turn calls
        # the global solver.
        all_found_solutions = pool.map(run_one_weighting, range(num_weightings))
        
    # Filter all found solutions to get the Pareto front
    valid_solutions = {s for s in all_found_solutions if s is not None}
    
    pareto_front, all_points = find_pareto_front(valid_solutions)

    print("\n--- Pareto Front Found ---")
    print("Format: (F1 Score, F2 Score)")
    for point in pareto_front:
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
    
    main(args.problem_size, args.num_weightings)
