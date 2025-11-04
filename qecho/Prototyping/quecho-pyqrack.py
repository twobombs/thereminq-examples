# from https://www.nature.com/articles/s41586-025-09526-6
# genemi25 interpretation of a simple 1d OTOC string

import numpy as np
from pyqrack import QrackSimulator # Import QrackSimulator directly

# --- 1. Setup Parameters ---

n_qubits = 4

# Define the qubits for our operators, based on the paper's setup
q_m = 0  # We will measure M = Z on qubit 0
q_b = 2  # We will apply B = X on qubit 2

print(f"Native PyQrack simulation with {n_qubits} qubits:")
print(f"  M = Z gate on qubit {q_m}")
print(f"  B = X gate on qubit {q_b}")
print("  U = Minimal non-commuting unitary (CNOT_2,0)\n")

# --- 2. Define Unitaries as Functions ---

def apply_M(sim):
    """Applies the M operator (Z on q_m)"""
    sim.z(q_m)

def apply_B(sim):
    """Applies the B operator (X on q_b)"""
    sim.x(q_b)

def apply_U(sim):
    """Applies the scrambling unitary U to the simulator"""
    # This is a minimal U that makes B(t) and M(t) not commute.
    # CNOT(control=q_b, target=q_m)
    sim.mcx([q_b], q_m) 

def apply_U_dag(sim):
    """Applies the inverse unitary U_dag to the simulator"""
    # CNOT is its own inverse
    sim.mcx([q_b], q_m)


# --- 3. Run OTOC(1) (k=1) "Single-Path" Echo ---
# Sequence: |0> -> U -> B -> U_dag -> Measure M
sim_k1 = QrackSimulator(n_qubits)

# Apply the sequence
apply_U(sim_k1)
apply_B(sim_k1)
apply_U_dag(sim_k1)

# --- 4. Run OTOC(2) (k=2) "Double-Path" Echo ---
# Sequence: |0> -> U -> B -> U_dag -> M -> U -> B -> U_dag -> Measure M
sim_k2 = QrackSimulator(n_qubits)

# Apply first B(t) = U_dag B U
apply_U(sim_k2)
apply_B(sim_k2)
apply_U_dag(sim_k2)

# Apply M
apply_M(sim_k2)

# Apply second B(t) = U_dag B U
apply_U(sim_k2)
apply_B(sim_k2)
apply_U_dag(sim_k2)


# --- 5. Calculate and Display Results ---

# FIX: Calculate <Z> manually from probabilities,
# because pauli_expectation is bugged.
# <Z> = P(0) - P(1) = (1 - P(1)) - P(1) = 1 - 2*P(1)

# Get P(1) for qubit q_m for the k=1 simulation
prob_k1 = sim_k1.prob(q_m)
otoc_k1_val = 1.0 - (2.0 * prob_k1)

# Get P(1) for qubit q_m for the k=2 simulation
prob_k2 = sim_k2.prob(q_m)
otoc_k2_val = 1.0 - (2.0 * prob_k2)


print("\n--- Simulation Results (Exact) ---")
print(f"OTOC(1) (k=1) [C(2)] <Z> Value: {otoc_k1_val:.6f}")
print(f"OTOC(2) (k=2) [C(4)] <Z> Value: {otoc_k2_val:.6f}")

print("\nThis demonstrates the nested echo protocol using native pyqrack.")
print("The values are calculated exactly, without shots.")

