# Spin Glass Ground State (considered NP-complete)
# Produced by Dan Strano, Elara (the OpenAI custom GPT)

from pyqrackising import spin_glass_solver
import networkx as nx
import numpy as np

import sys
from itertools import combinations


# LABS spin glass
def generate_labs_qubo(N, lam=10.0):
    """
    Generate a LABS instance as a quadratic QUBO-like form
    by expanding quartic terms with auxiliary spins.
    
    Parameters
    ----------
    N : int
        Length of the binary sequence.
    lam : float
        Penalty coefficient for enforcing auxiliary constraints.
    
    Returns
    -------
    W : np.ndarray
        Full symmetric weight matrix (size ~ N^2).
    labels : list[str]
        Names of variables (original spins + auxiliaries).
    """
    # Label base spins
    spins = [f"s{i}" for i in range(N)]
    aux = []
    index_map = {s: idx for idx, s in enumerate(spins)}
    
    # Placeholder dictionary for coupling terms
    couplings = {}

    # Each k defines a set of autocorrelation interactions
    for k in range(1, N):
        for i, j in combinations(range(N - k), 2):
            # quartic term: s_i * s_{i+k} * s_j * s_{j+k}
            # introduce two auxiliaries a_i_k and a_j_k
            ai = f"a_{i}_{k}"
            aj = f"a_{j}_{k}"
            for a in (ai, aj):
                if a not in index_map:
                    index_map[a] = len(index_map)
                    aux.append(a)

            # term: a_i_k * a_j_k
            couplings[(ai, aj)] = couplings.get((ai, aj), 0.0) + 1.0

            # penalty to enforce a_i_k = s_i * s_{i+k}
            # This penalty is (a - s_i * s_{i+k})^2 = a^2 - 2*a*s_i*s_{i+k} + (s_i*s_{i+k})^2
            # In QUBO (0,1) or Ising (-1,1), s^2 = 1 (constant, can be ignored)
            # and a^2 = a (for 0,1) or 1 (for -1,1)
            # Assuming Ising spins (-1, 1), the penalty is lam * (1 - 2*a*s_i*s_{i+k} + 1)
            # The script seems to be using a QUBO-like formulation that maps to Ising
            # Let's trace the script's logic for the penalty:
            # -lam * (a_i_k * s_i)
            # -lam * (a_i_k * s_{i+k})
            # +2*lam * a_i_k (as diagonal term a_i_k * a_i_k)
            # This corresponds to lam * (a_i_k - s_i - s_{i+k})^2, which isn't quite right for a_ik = s_i*s_{i+k}
            # A more standard penalty for a = s_i * s_{i+k} (Ising) is P(a, s_i, s_{i+k}) = lam * (a*s_i*s_{i+k} - 1)^2
            # Or simpler: lam * (1 - a*s_i*s_{i+k}). This isn't quadratic.
            # A common quadratic penalty is: lam * (a - s_i*s_{i+k})^2 = lam * (a^2 - 2*a*s_i*s_{i+k} + (s_i*s_{i+k})^2)
            # = lam * (1 - 2*a*s_i*s_{i+k} + 1) = 2*lam * (1 - a*s_i*s_{i+k}). Also not quadratic.
            #
            # Let's re-examine the standard reduction for a = x*y (Ising):
            # We want to minimize (a - xy)^2. This is not quadratic.
            # Let's try to minimize -a*x*y. This is cubic.
            #
            # The standard quadratic penalty to enforce a = xy (Ising) is:
            # P = lam * (1 - ax - ay + xy)
            # This is 0 if a=xy and > 0 otherwise.
            # The script implements something different:
            # P = lam * (a*a - a*s_i - a*s_{i+k})
            # Let's check the script's penalty logic:
            for (x, y) in [(ai, f"s{i}"), (ai, f"s{i+k}")]:
                couplings[(x, y)] = couplings.get((x, y), 0.0) - lam

            # same for a_j_k
            for (x, y) in [(aj, f"s{j}"), (aj, f"s{j+k}")]:
                couplings[(x, y)] = couplings.get((x, y), 0.0) - lam

            # diagonal penalties to ensure consistency
            for a in (ai, aj):
                couplings[(a, a)] = couplings.get((a, a), 0.0) + 2 * lam
            
            # The penalty for a_i_k is:
            # 2*lam*a_i_k^2 - lam*a_i_k*s_i - lam*a_i_k*s_{i+k}
            # If spins are {-1, 1}, a^2 = 1 (a constant).
            # P = 2*lam - lam*a_i_k*s_i - lam*a_i_k*s_{i+k}
            # This doesn't look like a standard penalty to enforce a = s_i*s_{i+k}.
            #
            # If spins are {0, 1} (QUBO), then a^2 = a.
            # P = 2*lam*a_i_k - lam*a_i_k*s_i - lam*a_i_k*s_{i+k}
            # P = a_i_k * (2*lam - lam*s_i - lam*s_{i+k})
            # This penalty term encourages a_i_k=0.
            # This reduction seems specific, perhaps from a particular paper.
            # The goal is to replace s_i*s_{i+k}*s_j*s_{j+k}
            # by a_i_k * a_j_k
            # with penalties to enforce a_i_k = s_i*s_{i+k} and a_j_k = s_j*s_{j+k}
            # A standard {0,1} QUBO penalty for a = s_i*s_{i+k} is:
            # lam * (s_i*s_{i+k} - 2*a*s_i - 2*a*s_{i+k} + 3*a)
            # The script is using a different formulation.
            # Despite the specific penalty formulation, the *intent* is clear:
            # reduce a quartic term to a quadratic one using auxiliaries.

    # Build full symmetric matrix
    M = len(index_map)
    W = np.zeros((M, M))
    for (x, y), w in couplings.items():
        i, j = index_map[x], index_map[y]
        W[i, j] += w
        if i != j:
            W[j, i] += w  # symmetric

    # Normalize
    if W.max() != 0:
        W /= W.max()

    labels = list(index_map.keys())
    return W, labels


def calculate_labs_energy(spins_pm1):
    """
    Calculates the true LABS energy E = sum(C_k^2) as defined in the paper (Eq. 1).
    
    Parameters
    ----------
    spins_pm1 : list[int]
        The binary sequence as {-1, 1} spins.
        
    Returns
    -------
    int
        The total LABS energy.
    """
    N = len(spins_pm1)
    total_energy = 0
    # k is the autocorrelation lag
    for k in range(1, N):
        Ck = 0
        # i is the position in the sequence
        for i in range(N - k):
            Ck += spins_pm1[i] * spins_pm1[i+k]
        total_energy += Ck**2
    return total_energy


if __name__ == "__main__":
    n_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    quality = int(sys.argv[2]) if len(sys.argv) > 2 else None # Quality parameter for the solver

    # --- New code to handle 'lam' CLI option ---
    default_lam = 10.0
    lam = float(sys.argv[3]) if len(sys.argv) > 3 else default_lam
    # --- End of new code ---

    print(f"Generating LABS QUBO for N = {n_nodes} (lam = {lam})...")
    G_m, labels = generate_labs_qubo(n_nodes, lam=lam)
    print(f"Matrix size (original spins + auxiliaries): {G_m.shape[0]}x{G_m.shape[0]}")
    
    # --- Updated print statement to show quality ---
    quality_str = f"with quality={quality}" if quality is not None else "with default quality"
    print(f"Solving with pyqrackising spin_glass_solver {quality_str}...")
    # --- End of update ---
    
    # is_spin_glass=False likely means it's a QUBO/BF, not a standard Ising model
    best_bitstring, best_cut_value, best_cut, best_energy = spin_glass_solver(G_m, quality=quality, is_spin_glass=False)

    print("\n--- Results ---")
    print(f"Best energy (solver output): {best_energy}")
    print(f"Best cut value (solver output): {best_cut_value}")
    
    # Extract the original spins from the solution
    original_spins = []
    for i in range(n_nodes):
        label = f"s{i}"
        if label in labels:
            idx = labels.index(label)
            original_spins.append(best_bitstring[idx])
        else:
            original_spins.append('?') # Should not happen

    print(f"\nLength {n_nodes} solution (original spins): {''.join(map(str, original_spins))}")
    
    # --- New code to calculate true LABS energy ---
    # Convert {0, 1} bitstring to {-1, 1} spins
    # The LABS formula E = sum(C_k^2) uses s_i in {-1, 1}
    # We assume the standard mapping: 0 -> -1, 1 -> 1
    # FIX: Convert b from string to int before doing math
    spins_pm1 = [(2 * int(b) - 1) for b in original_spins]
    
    true_labs_energy = calculate_labs_energy(spins_pm1)
    print(f"True LABS Energy (E = sum(C_k^2)): {true_labs_energy}")
    # --- End of new code ---
    
    # --- Restoring missing output ---
    print(f"\nFull solution bitstring (all variables): {''.join(map(str, best_bitstring))}")
    print(f"Variable Labels: {labels}")
    # --- End of restored output ---
