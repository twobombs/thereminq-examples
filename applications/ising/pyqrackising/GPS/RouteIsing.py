import numpy as np
import math
import networkx as nx
import random

# ==========================================
# 1. GPS & Geometry Helpers
# ==========================================

def haversine(lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two points (km)."""
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ==========================================
# 2. Hamiltonian Construction (The Physics)
# ==========================================

def generate_tsp_isings(gps_locations, penalty_strength=None):
    """
    Generates Ising model parameters (J, h) for the TSP.
    """
    num_cities = len(gps_locations)
    num_qubits = num_cities ** 2  # N cities * N time steps
    
    # Auto-tune penalty: Must be > max path cost to enforce constraints
    if penalty_strength is None:
        penalty_strength = 20000.0 

    # Initialize QUBO matrix Q
    Q = np.zeros((num_qubits, num_qubits))
    
    def get_idx(city, time):
        return city * num_cities + time

    # Precompute distances
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                dist_matrix[i][j] = haversine(*gps_locations[i], *gps_locations[j])

    # --- A. Constraint: One City per Time Step ---
    for t in range(num_cities):
        for i in range(num_cities):
            u = get_idx(i, t)
            Q[u, u] -= penalty_strength
            for j in range(i + 1, num_cities):
                v = get_idx(j, t)
                Q[u, v] += 2 * penalty_strength

    # --- B. Constraint: Each City Visited Exactly Once ---
    for i in range(num_cities):
        for t1 in range(num_cities):
            u = get_idx(i, t1)
            Q[u, u] -= penalty_strength
            for t2 in range(t1 + 1, num_cities):
                v = get_idx(i, t2)
                Q[u, v] += 2 * penalty_strength

    # --- C. Objective: Minimize Distance ---
    for t in range(num_cities):
        next_t = (t + 1) % num_cities
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    u = get_idx(i, t)
                    v = get_idx(j, next_t)
                    idx1, idx2 = min(u, v), max(u, v)
                    Q[idx1, idx2] += dist_matrix[i][j]

    # --- D. Convert QUBO to Ising (h, J) ---
    h = np.zeros(num_qubits)
    J = np.zeros((num_qubits, num_qubits))
    
    for i in range(num_qubits):
        term1 = Q[i, i] / 2
        term2 = 0
        for k in range(num_qubits):
            if k > i: term2 += Q[i, k]
            if k < i: term2 += Q[k, i]
        h[i] = term1 + term2 / 4
        
        for j in range(i + 1, num_qubits):
            J[i, j] = Q[i, j] / 4
            
    return J, h

# ==========================================
# 3. Solvers (PyQrackIsing & Fallback)
# ==========================================

def solve_with_pyqrack(J, h):
    """
    Solves using pyqrackising's spin_glass_solver.
    """
    try:
        from pyqrackising import spin_glass_solver
        print("\n[System] PyQrackIsing detected. Engaging Quantum-Inspired Solver...")
    except ImportError:
        print("\n[System] PyQrackIsing not found. Install via 'pip install pyqrackising'")
        return None

    # Construct Graph
    # Note: Pure spin glass solvers often focus on J (edges). 
    # To encode 'h' (bias), we use a 'ghost spin' trick or node weights if supported.
    # Here we map to a NetworkX graph with node weights.
    G = nx.Graph()
    num_qubits = len(h)
    
    for i in range(num_qubits):
        G.add_node(i, weight=h[i])
        
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            if J[i, j] != 0:
                G.add_edge(i, j, weight=J[i, j])

    # Run Solver
    # Returns list of spins (usually -1/1 or 0/1 depending on version)
    results = spin_glass_solver(G)
    
    # Normalize output to [-1, 1]
    return [1 if x > 0 else -1 for x in results]

def solve_with_simulated_annealing(J, h):
    """
    Classical fallback solver if pyqrack is missing.
    """
    print("\n[System] Falling back to Classical Simulated Annealing...")
    
    def get_energy(spins):
        E = 0
        # h terms
        E += np.dot(h, spins)
        # J terms (upper triangular)
        for i in range(len(h)):
            for j in range(i + 1, len(h)):
                if J[i, j] != 0:
                    E += J[i, j] * spins[i] * spins[j]
        return E

    num_qubits = len(h)
    current_spins = np.random.choice([-1, 1], size=num_qubits)
    current_energy = get_energy(current_spins)
    
    best_spins = current_spins.copy()
    best_energy = current_energy
    
    T = 150.0
    cooling = 0.99
    
    for step in range(150000):
        # Flip random spin
        k = np.random.randint(0, num_qubits)
        current_spins[k] *= -1
        new_energy = get_energy(current_spins)
        
        diff = new_energy - current_energy
        if diff < 0 or np.random.rand() < np.exp(-diff / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_spins = current_spins.copy()
        else:
            current_spins[k] *= -1 # Revert
            
        T *= cooling
        if T < 0.1: break
            
    return best_spins

# ==========================================
# 4. Decoder
# ==========================================

def decode_solution(spins, city_names):
    num_cities = len(city_names)
    # Map -1/+1 -> 0/1
    binary = [(s + 1) // 2 for s in spins]
    grid = np.array(binary).reshape((num_cities, num_cities))
    
    path = [-1] * num_cities
    valid = True
    
    print("\n--- Raw Grid (Rows=Cities, Cols=Time) ---")
    print(grid)
    
    for t in range(num_cities):
        cols = np.where(grid[:, t] == 1)[0]
        if len(cols) == 1:
            path[t] = cols[0]
        else:
            valid = False
            
    if not valid or len(set(path)) != num_cities:
        return None, "Constraints Violated (Invalid Path)"
    
    named_route = [city_names[i] for i in path]
    named_route.append(named_route[0]) # Close loop
    return named_route, "Valid"

# ==========================================
# 5. Main Execution
# ==========================================

if __name__ == "__main__":
    # 1. Define Cities
    cities = {
        "New York": (40.7128, -74.0060),
        "London": (51.5074, -0.1278),
        "Paris": (48.8566, 2.3522),
        "Tokyo": (35.6762, 139.6503),
        "Sydney": (-33.8688, 151.2093)
    }
    names = list(cities.keys())
    coords = list(cities.values())
    
    print(f"Initializing TSP for {len(names)} cities...")
    
    # 2. Build Hamiltonian
    J, h = generate_tsp_isings(coords)
    
    # 3. Solve (Try PyQrack, else SA)
    spins = solve_with_pyqrack(J, h)
    
    if spins is None:
        spins = solve_with_simulated_annealing(J, h)
        
    # 4. Decode
    route, status = decode_solution(spins, names)
    
    print(f"\nStatus: {status}")
    if route:
        print(f"Optimal Route: {' -> '.join(route)}")
        
        # Calculate final distance
        total_dist = 0
        for i in range(len(route) - 1):
            c1 = cities[route[i]]
            c2 = cities[route[i+1]]
            total_dist += haversine(*c1, *c2)
        print(f"Total Distance: {total_dist:.2f} km")
    else:
        print("Solver failed to find a valid configuration. Try increasing penalty_strength or annealing steps.")
