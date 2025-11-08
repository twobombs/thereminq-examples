#
# pareto-front PoC
# gemini25
#

import numpy as np
import random

# --- 1. Define the Objective Functions ---

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

# --- 2. The Solver (The "Quantum" Part) ---

def get_solver(problem_size):
    """
    Creates a simple *classical* solver for this toy problem.
    This function finds the *true* best solution for any given weighting.
    """
    # Generate all 2^N possible solutions for a small problem
    all_solutions = []
    for i in range(2**problem_size):
        # Format as a binary string, pad with 0s, convert to int list
        bin_str = bin(i)[2:].zfill(problem_size)
        solution = [int(bit) for bit in bin_str]
        all_solutions.append(solution)
    
    print(f"Solver initialized with {len(all_solutions)} possible solutions.")

    def find_optimal_solution(weight_1, weight_2):
        """
        ---!!! THIS IS THE STEP a quantum algorithm would perform !!!---
        
        Instead of this classical brute-force search, you would:
        1. Construct a Hamiltonian (Ising model) from the combined objective:
           H = - (weight_1 * F1 + weight_2 * F2)
        2. Use an algorithm (like QAOA or QA) to find its ground state (the solution).
        3. If using PyQrack, you would build the QAOA circuit here and
           run the simulation to find the most probable solution.
        
        We use a classical solver here to keep the example clear.
        """
        best_solution = None
        best_score = -float('inf')

        for solution in all_solutions:
            # Calculate the two objectives for this solution
            f1_score = objective_1_max_ones(solution)
            f2_score = objective_2_max_flips(solution)
            
            # This is the single weighted-sum objective
            combined_score = (weight_1 * f1_score) + (weight_2 * f2_score)

            if combined_score > best_score:
                best_score = combined_score
                # Store the solution and its *original* objective scores
                best_solution = (tuple(solution), f1_score, f2_score)
        
        return best_solution

    return find_optimal_solution

# --- 3. The Pareto Filtering Function ---

def find_pareto_front(all_found_solutions):
    """
    Filters a set of solutions to find the non-dominated Pareto front.
    This follows the definition from the paper.
    
    A solution (s) is "non-dominated" if no other solution (s')
    is better or equal on *all* objectives and strictly better on *at least one*.
    """
    print(f"\nFiltering {len(all_found_solutions)} unique solutions to find the front...")
    
    # Use a dictionary to store solutions by their (F1, F2) scores
    # This automatically handles duplicates
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
            
            # Check if p2 dominates p1
            if (f1_p2 >= f1_p1 and f2_p2 >= f2_p1) and \
               (f1_p2 > f1_p1 or f2_p2 > f2_p1):
                is_dominated = True
                break # Dominated, no need to check further
        
        if not is_dominated:
            pareto_front_points.add(p1)
            
    # Sort for easier reading
    return sorted(list(pareto_front_points)), solution_points

# --- 4. The Main Workflow ---

def main():
    PROBLEM_SIZE = 10
    NUM_WEIGHTINGS = 200 # Number of random weightings to sample
    
    # Get the solver function
    solver = get_solver(PROBLEM_SIZE)
    
    # Store all unique solutions found
    all_found_solutions = set() 

    print(f"Running solver for {NUM_WEIGHTINGS} random weightings...")
    
    for _ in range(NUM_WEIGHTINGS):
        # Generate random, normalized weights
        w1 = np.random.random()
        w2 = 1.0 - w1
        
        # Run the solver for this weighting
        # This is Step 2, the "quantum" part
        solution = solver(w1, w2)
        
        if solution:
            all_found_solutions.add(solution)

    # Filter all found solutions to get the Pareto front
    pareto_front, all_points = find_pareto_front(all_found_solutions)

    print("\n--- Pareto Front Found ---")
    print("Format: (F1 Score, F2 Score)")
    for point in pareto_front:
        print(f"  {point}  (e.g., solution: {all_points[point]})")

if __name__ == "__main__":
    main()
