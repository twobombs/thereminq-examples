# -*- coding: utf-8 -*-
import time
from PyQrackIsing import generate_tfim_samples

# Define the simulation parameters
J = -1.0
h = 2.0
z = 4
theta = 0.17453292519943295 # pi/18 radians
t = 5
n_qubits = 56
shots = 100

# --- Print the parameters ---
print("## Simulation Parameters ##")
print(f"J (Coupling Strength): {J}")
print(f"h (Transverse Field): {h}")
print(f"z (Coordination Number): {z}")
print(f"theta (Variational Angle): {theta}")
print(f"t (Trotter Steps): {t}")
print(f"n_qubits (Number of Qubits): {n_qubits}")
print(f"shots (Number of Measurements): {shots}")
print("-" * 25) # Separator

# --- Time the calculation ---
start_time = time.perf_counter() # Start the clock

samples = generate_tfim_samples(
    J=J,
    h=h,
    z=z,
    theta=theta,
    t=t,
    n_qubits=n_qubits,
    shots=shots
)

end_time = time.perf_counter() # Stop the clock
elapsed_time = end_time - start_time
print(f"Calculation took: {elapsed_time:.4f} seconds")
print("-" * 25)

# --- Print the output samples as comma-separated decimals ---
print("\n## Output Samples (Decimal Comma-Separated) ##")
# The 'samples' variable is already a list of integers.
# This converts each integer to a string and joins them with a comma.
print(",".join(map(str, samples)))
