# -*- coding: utf-8 -*-
# Wormhole Teleportation Protocol via PyQrack
# Implements the "Negative-Energy Spike" transmission channel.
#
# System: Two entangled chains (Left & Right) of N qubits each.
# Mechanism: TFD State -> Scramble -> Shockwave (Coupling) -> Unscramble -> Measure

import math
import random
import os
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pyqrack import QrackSimulator, Pauli

# --- Configuration ---
config_path = os.path.join(os.path.dirname(__file__), "config.toml")
with open(config_path, "rb") as f:
    config = tomllib.load(f)["system"]

n_side = config["n_side"]
n_total = 2 * n_side
j = config["j"]
h = config["h"]
t_evolve = config["t_evolve"]
g_shock = config["g_shock"]
message_qubit = config["message_qubit"]

# --- 1. Initialize Simulator ---
sim = QrackSimulator(n_total)

# --- 2. Prepare Thermofield Double (TFD) State ---
# Ideally, TFD is e^(-beta*H/2) |MAX_ENTANGLED>.
# For simplicity/infinite temp limit, we start with perfect Bell pairs.
print(f"1. Preparing Entangled TFD State ({n_total} qubits)...")
for i in range(n_side):
    # Create Bell pair between L[i] and R[i]
    # L indices: 0 to n_side-1
    # R indices: n_side to 2*n_side-1
    left_q = i
    right_q = i + n_side
    
    sim.h(left_q)
    sim.mcnot([left_q], right_q)

# --- Helper: Time Evolution (Scrambling) ---
def evolve_system(sim, time, j_c, h_c, n_side):
    """
    Evolve both Left and Right systems independently under H_Ising.
    H = sum(J * Z_i Z_{i+1}) + sum(h * X_i)

    Args:
        sim: The QrackSimulator instance.
        time: The time duration for evolution.
        j_c: Ising Interaction Strength.
        h_c: Transverse Field Strength.
        n_side: Number of qubits per side.
    """
    # Trotter steps (simplified for demo)
    steps = 10 
    dt = time / steps
    
    for _ in range(steps):
        # Apply Transverse Field (X) on all qubits
        for i in range(2 * n_side):
            sim.r(Pauli.PauliX, 2 * h_c * dt, i)
            
        # Apply Ising Interaction (ZZ) on Left Chain
        for i in range(n_side - 1):
            # Exp(-i * J * dt * Z_i Z_{i+1})
            # Decomposed: CNOT -> RZ -> CNOT
            sim.mcnot([i], i+1)
            sim.r(Pauli.PauliZ, 2 * j_c * dt, i+1)
            sim.mcnot([i], i+1)
            
        # Apply Ising Interaction (ZZ) on Right Chain
        for i in range(n_side, 2 * n_side - 1):
            sim.mcnot([i], i+1)
            sim.r(Pauli.PauliZ, 2 * j_c * dt, i+1)
            sim.mcnot([i], i+1)

# --- 3. Backward Time Evolution (-t) ---
# We go "back in time" to insert the message earlier.
print(f"2. Evolving backwards by t = -{t_evolve}...")
evolve_system(sim, -t_evolve, j, h, n_side)

# --- 4. Insert The Message (The Input) ---
# We flip a qubit on the Left boundary. This is the "particle" entering the wormhole.
print("3. Injecting Message (Pauli Y) on Left Boundary...")
sim.y(message_qubit)

# --- 5. Forward Time Evolution (+t) ---
# The message scrambles into the bulk.
print(f"4. Evolving forward by t = +{t_evolve}...")
evolve_system(sim, t_evolve, j, h, n_side)

# --- 6. The Negative-Energy Shockwave (Coupling) ---
# This is the interaction connecting Left and Right: Exp(i * g * Sum(Z_L * Z_R))
# This correlates the two sides, enabling the traversal.
print(f"5. Applying Negative-Energy Shockwave (g = {g_shock})...")
for i in range(n_side):
    left_q = i
    right_q = i + n_side
    
    # Implement ZZ interaction between Left and Right
    sim.mcnot([left_q], right_q)
    sim.r(Pauli.PauliZ, 2 * g_shock, right_q) # Angle determines coupling strength
    sim.mcnot([left_q], right_q)

# --- 7. Final Evolution (Decoding) ---
# In the ideal protocol, the Right side evolves further to decode.
# For symmetry in this simple echo, we evolve forward again.
print(f"6. Final Evolution t = +{t_evolve}...")
evolve_system(sim, t_evolve, j, h, n_side)

# --- 8. Measure / Verify Teleportation ---
# If successful, the message (Y) should reappear on the RIGHT boundary
# at the corresponding qubit index (message_qubit + n_side).
target_qubit = message_qubit + n_side

# We measure the expectation value of Y on the target qubit.
# <Y> close to 1 (or -1 depending on sign conventions) means success.
# <Y> close to 0 means the info remains scrambled/lost.

expectation_y = sim.op_expectation(Pauli.PauliY, target_qubit)

print("-" * 30)
print(f"Teleportation Result (Expectation of Y on Right[{target_qubit}]):")
print(f"<Y> = {expectation_y:.4f}")
print("-" * 30)

if abs(expectation_y) > 0.1:
    print("SUCCESS: Signal Detected! The wormhole was traversable.")
else:
    print("FAILURE: Signal lost. Scrambling dominated or shockwave mistimed.")
