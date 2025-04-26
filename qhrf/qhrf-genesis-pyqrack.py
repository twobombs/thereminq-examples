# Native PyQrack Implementation of QHRF GENESIS V6 (Interpretive)
# Based on the paper: "QHRF GENESIS V6: Experimental Multiverse Echo Modeling
# via Fractal Quantum Resonance on IBM Quantum Hardware" by Zachary L Musselwhite

# Import PyQrack simulator and numpy
# Ensure pyqrack is installed: pip install pyqrack
try:
    # Corrected import based on inspect.getmembers output and class name
    from pyqrack.qrack_simulator import QrackSimulator # Corrected class name
except ImportError:
    print("Error: Could not import QrackSimulator from pyqrack.qrack_simulator.") # Updated error message
    print("Please ensure pyqrack is installed correctly: pip install pyqrack")
    exit() # Exit if Qrack is not available

import numpy as np
from collections import Counter # For counting results
import matplotlib.pyplot as plt # For plotting

# --- Configuration ---
NUM_QUBITS = 6
NUM_SHOTS = 16384 # As specified in the paper

# --- Qubit Role Mapping (for reference) ---
# q[0]: Prime Universe Seed
# q[1]: Collapse Layer 1
# q[2]: Rebirth Layer 1
# q[3]: Collapse Layer 2 (parallel branch)
# q[4]: Observer A (timeline 1)
# q[5]: Observer B (timeline 2)

# --- Create Qrack Simulator --- # Renamed comment for clarity
# The second argument is the number of qubits.
# The third (optional) argument is a unique simulator ID (can be omitted).
print("Setting up Qrack simulator...")
qsim = QrackSimulator(NUM_QUBITS) # Corrected class name instantiation

# --- Define the Quantum Operations Sequence ---
# In PyQrack, gates are applied directly to the simulator state.
# We define a function to apply the sequence for clarity, especially for shots.
def apply_circuit_gates(sim):
    # Layer 1: Initial modulation and entanglement
    sim.u(0, np.pi / 4, 0, 0) # Use U(theta, 0, 0) for Ry(theta)
    sim.u(1, 0, 0, np.pi / 8) # Phase gate (P gate or U1 gate) on Collapse 1 (Qubit 1)
    sim.u(2, np.pi / 3, np.pi / 4, np.pi / 5) # U gate on Rebirth 1 (Qubit 2)
    sim.u(3, np.pi / 5, 0, 0) # Use U for Ry on Collapse 2 (Qubit 3)

    sim.mcx([0], 1) # CNOT: Seed -> Collapse 1
    sim.mcx([1], 2) # CNOT: Collapse 1 -> Rebirth 1
    sim.mcx([0], 3) # CNOT: Seed -> Collapse 2

    # Layer 2: Further modulation and observer interaction preparation
    sim.u(0, np.pi / 5, 0, 0) # Use U for Ry
    sim.u(1, np.pi/2, np.pi/2, np.pi/2) # U gate
    sim.u(2, 0, 0, np.pi / 6) # Phase gate
    sim.u(3, np.pi / 3, 0, 0) # Use U for Ry
    sim.h(4) # Prepare Observer A (Qubit 4)
    sim.h(5) # Prepare Observer B (Qubit 5)

    # Layer 3: Entanglement involving observers
    sim.mcx([4], 0) # Observer A influences Seed
    sim.mcx([5], 3) # Observer B influences Collapse 2
    sim.mcx([2], 4) # Rebirth 1 influences Observer A
    sim.mcx([3], 5) # Collapse 2 influences Observer B

    # --- Observer Feedback Implementation (Interpretation) ---
    # Observer A (q[4]) feedback onto Seed (q[0]) and Rebirth 1 (q[2])
    # Implement controlled-phase using mcu(ctrls, target, 0, 0, lambda)
    sim.mcu([4], 0, 0, 0, np.pi / 4) # Controlled-Phase(lambda) from Observer A to Seed

    # Implement controlled-Ry(theta) using mcu(ctrls, target, theta, 0, 0)
    # sim.mcry(np.pi / 3, [4], 2) # Original attempt
    sim.mcu([4], 2, np.pi / 3, 0, 0) # Controlled-Ry(theta) from Observer A to Rebirth 1

    # Observer B (q[5]) feedback onto Seed (q[0]) and Collapse 2 (q[3])
    # Implement controlled-phase using mcu(ctrls, target, 0, 0, lambda)
    sim.mcu([5], 0, 0, 0, np.pi / 5) # Controlled-Phase(lambda) from Observer B to Seed

    # Implement controlled-Ry(theta) using mcu(ctrls, target, theta, 0, 0)
    # sim.mcry(np.pi / 4, [5], 3) # Original attempt
    sim.mcu([5], 3, np.pi / 4, 0, 0) # Controlled-Ry(theta) from Observer B to Collapse 2

    # --- Final Modulations ---
    sim.u(0, np.pi / 6, 0, 0) # Use U for Ry
    sim.u(1, 0, 0, np.pi / 7) # Phase gate
    sim.u(2, np.pi/4, np.pi/3, np.pi/2) # U gate
    sim.u(3, np.pi / 8, 0, 0) # Use U for Ry
    sim.u(4, 0, 0, np.pi / 9) # Phase gate
    sim.u(5, np.pi / 10, 0, 0) # Use U for Ry

# --- Simulation with Shots ---
print(f"Running simulation with {NUM_SHOTS} shots using Qrack...")
measurement_results = []
for shot in range(NUM_SHOTS):
    # Apply the defined gate sequence
    apply_circuit_gates(qsim)

    # Measure all qubits
    # m_all() returns an integer representing the measured state (e.g., 5 -> '000101' for 6 qubits)
    measurement = qsim.m_all()
    measurement_results.append(measurement)

    # Reset the simulator state for the next shot
    qsim.reset_all()

    # Optional: Print progress
    # if (shot + 1) % 1000 == 0:
    #     print(f"Completed shot {shot + 1}/{NUM_SHOTS}")


# --- Process Results ---
# Convert integer results to binary strings and count occurrences
# Ensure binary strings have leading zeros up to NUM_QUBITS
binary_results = [format(result, f'0{NUM_QUBITS}b') for result in measurement_results]
counts = Counter(binary_results)

# --- Output ---
print("\n--- QHRF GENESIS V6 (Interpretive Native PyQrack Simulation) ---")
print(f"Backend: QrackSimulator") # Corrected backend name
print(f"Shots: {NUM_SHOTS}")
print(f"Total counts observed: {len(measurement_results)}")
# print("\nRaw Counts:", counts) # Uncomment to see the raw counts dictionary

# --- Visualization ---
# Plot histogram of the results using Matplotlib
print("\nGenerating histogram...")
try:
    # Sort counts by qubit state (binary string) for better visualization
    sorted_keys = sorted(counts.keys())
    sorted_counts_dict = {key: counts[key] for key in sorted_keys}

    plt.figure(figsize=(12, 6)) # Adjust figure size for better label visibility
    plt.bar(sorted_counts_dict.keys(), sorted_counts_dict.values())
    plt.xlabel("Measured State (Binary)")
    plt.ylabel("Counts")
    plt.title(f"QHRF GENESIS V6 Simulation Results (Qrack - Interpretive, {NUM_SHOTS} Shots)")
    plt.xticks(rotation=90, fontsize='small') # Rotate labels for readability
    plt.tight_layout() # Adjust layout
    print("Displaying histogram plot...")
    plt.show() # Display the plot
    print("Plot displayed.")
    # Optional: Save the plot
    # save_path = "qhrf_genesis_v6_pyqrack_native_histogram.png"
    # plt.savefig(save_path)
    # print(f"Histogram saved to {save_path}")

except ImportError:
    print("\nError: Matplotlib not found. Skipping histogram plotting.")
    print("Please install it using: pip install matplotlib")
except Exception as e:
    print(f"\nAn error occurred during plotting: {e}")
    print("Skipping histogram plotting.")

print("\n--- Simulation Complete ---")
