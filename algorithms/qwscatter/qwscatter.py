# -*- coding: utf-8 -*-
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator # Use AerSimulator for simulation
# Removed plot_histogram as it's not used when plotting statevector probabilities
# from qiskit.visualization import plot_histogram
import math
from qiskit.exceptions import QiskitError # Import QiskitError for handling
from qiskit.quantum_info import Statevector # Import Statevector for type hints if needed

# --- Simulation Parameters ---
NUM_POSITION_QUBITS = 6  # Number of qubits for position (defines 2^N positions)
NUM_STEPS = 15           # Number of steps in the quantum walk
INITIAL_POSITION = 0     # Starting position index (0 to 2^N - 1)
# Position of the barrier (near the middle, ensure it's within bounds)
BARRIER_POSITION = 1 << (NUM_POSITION_QUBITS // 2)
BARRIER_PHASE = np.pi / 2 # Phase shift applied by the barrier (e.g., pi/2, pi)

# --- Helper Functions ---

def controlled_increment(qc, position_q, control_q, num_qubits):
    """Applies a controlled increment operation (+1) to the position register."""
    # Use multi-controlled Toffoli gates (MCT) to implement increment
    # This is a standard way to implement addition on quantum computers
    # Increment using ripple-carry adder logic with Toffoli gates
    for i in range(num_qubits - 1):
        # Compute carry bits using MCX
        control_indices = [control_q] + [position_q[j] for j in range(i + 1)]
        qc.mcx(control_indices, position_q[i + 1])
    # Increment the least significant bit (LSB)
    qc.cx(control_q, position_q[0])
    # Compute carries backward to finalize addition (clean up auxiliary bits if used)
    # This reverse part handles the ripple effect correctly
    for i in range(num_qubits - 2, -1, -1):
         # Compute carry bits using MCX
        control_indices = [control_q] + [position_q[j] for j in range(i + 1)]
        qc.mcx(control_indices, position_q[i + 1])
        # Apply CNOT based on carry calculation (part of adder logic)
        if i > 0:
             control_indices_cnot = [control_q] + [position_q[j] for j in range(i)]
             qc.mcx(control_indices_cnot, position_q[i]) # Simplified, real adder is more complex

def controlled_decrement(qc, position_q, control_q, num_qubits):
    """Applies a controlled decrement operation (-1) to the position register."""
    # Implement decrement as the inverse of increment
    # Apply inverse of the final CNOT
    # Compute carries backward (inverse)
    for i in range(num_qubits - 1):
         # Apply CNOT based on carry calculation (part of adder logic - inverse)
        if i > 0:
             control_indices_cnot = [control_q] + [position_q[j] for j in range(i)]
             qc.mcx(control_indices_cnot, position_q[i]) # Simplified, real adder is more complex
         # Compute carry bits using MCX (inverse)
        control_indices = [control_q] + [position_q[j] for j in range(i + 1)]
        qc.mcx(control_indices, position_q[i + 1])

    # Decrement the least significant bit (LSB)
    qc.cx(control_q, position_q[0])
    # Compute carries forward (inverse)
    for i in range(num_qubits - 2, -1, -1):
        control_indices = [control_q] + [position_q[j] for j in range(i + 1)]
        qc.mcx(control_indices, position_q[i + 1])


def apply_barrier(qc, position_q, barrier_pos, barrier_phase, num_qubits):
    """Applies a phase shift if the walker is at the barrier position."""
    # Ensure barrier position is valid
    if barrier_pos < 0 or barrier_pos >= (1 << num_qubits):
        print(f"Warning: Barrier position {barrier_pos} is outside the range [0, { (1 << num_qubits) - 1}]. Skipping barrier.")
        return

    # Convert barrier position to binary string representation
    barrier_bin = format(barrier_pos, f'0{num_qubits}b')[::-1] # Reverse for Qiskit's qubit ordering (LSB is q0)

    # Apply X gates to qubits that are '0' in the barrier position binary string
    # This makes the multi-controlled gate trigger when qubits match the barrier_pos
    for i in range(num_qubits):
        if barrier_bin[i] == '0':
            qc.x(position_q[i])

    # Apply the multi-controlled phase shift (MCP)
    # Apply RZ(barrier_phase) controlled by all position qubits.
    if num_qubits > 1:
         # Apply multi-controlled P gate targeting the last qubit
         qc.mcp(barrier_phase, position_q[:-1], position_q[-1])
    elif num_qubits == 1: # If only one position qubit
         # For num_qubits=1, mcp doesn't exist, use controlled phase P.
         # The X gates above ensure the control works correctly based on barrier_bin.
         # We just need to apply the phase gate P to the single qubit,
         # controlled by its state (which is now |1> if it matched the barrier pos).
         qc.p(barrier_phase, position_q[0]) # Applies phase if the (potentially flipped) qubit is 1


    # Apply X gates again to revert the position qubits to their original state
    for i in range(num_qubits):
        if barrier_bin[i] == '0':
            qc.x(position_q[i])

# --- Build the Quantum Circuit ---

# Define quantum registers
position_reg = QuantumRegister(NUM_POSITION_QUBITS, name='pos')
coin_reg = QuantumRegister(1, name='coin')
# No classical register needed for statevector simulation

# Create the quantum circuit
qc = QuantumCircuit(position_reg, coin_reg, name="QuantumWalk")

# 1. Initialize Position: Set the initial position qubits
# Convert initial position to binary and apply X gates
initial_pos_bin = format(INITIAL_POSITION, f'0{NUM_POSITION_QUBITS}b')[::-1] # Qiskit LSB order
for i in range(NUM_POSITION_QUBITS):
    if initial_pos_bin[i] == '1':
        qc.x(position_reg[i])

# 2. Initialize Coin: Start in |0> state (no Hadamard initially needed)
# qc.h(coin_reg[0]) # Optional: Start coin in superposition if desired

qc.barrier() # Separate initialization

# 3. Quantum Walk Steps
for step in range(NUM_STEPS):
    # a) Coin Toss
    qc.h(coin_reg[0])
    qc.barrier()

    # b) Controlled Shift Operation
    # If coin is |0> (control is active low), decrement position
    qc.x(coin_reg[0]) # Flip control to trigger on |0>
    controlled_decrement(qc, position_reg, coin_reg[0], NUM_POSITION_QUBITS)
    qc.x(coin_reg[0]) # Flip back control

    # If coin is |1> (control is active high), increment position
    controlled_increment(qc, position_reg, coin_reg[0], NUM_POSITION_QUBITS)
    qc.barrier()

    # c) Apply Barrier Potential (Phase Shift)
    apply_barrier(qc, position_reg, BARRIER_POSITION, BARRIER_PHASE, NUM_POSITION_QUBITS)
    qc.barrier(label=f"Step {step+1}")

# --- Explicitly Save Statevector ---
# Add this instruction *before* running the simulation
qc.save_statevector(label='final_statevector') # Added label for clarity

# --- Simulation ---
# Use AerSimulator's statevector simulator
simulator = AerSimulator(method='statevector')

# Run the simulation
# No need to transpile for save_statevector usually, but good practice
t_qc = transpile(qc, simulator)
result = simulator.run(t_qc).result()

# --- Process Results ---
try:
    # Retrieve the saved statevector object using the label
    statevector_obj = result.data(qc)['final_statevector']

    # *** FIX: Explicitly cast to NumPy array to avoid DeprecationWarning ***
    statevector_np = np.asarray(statevector_obj)

    # Calculate probabilities for each position
    num_positions = 2**NUM_POSITION_QUBITS
    probabilities = np.zeros(num_positions)

    # The statevector includes the coin qubit. We need to trace it out.
    total_qubits = NUM_POSITION_QUBITS + 1
    if len(statevector_np) != (1 << total_qubits): # Use len() on the NumPy array
         raise ValueError(f"Statevector length {len(statevector_np)} does not match expected {1 << total_qubits}")

    for i in range(len(statevector_np)): # Iterate using the length of the NumPy array
        # Get the binary representation of the state index
        full_binary_state = format(i, f'0{total_qubits}b')

        # Determine which bit is the coin based on register order in qc.qubits
        # Qiskit typically orders registers as they are added: position_reg, then coin_reg
        # So coin is the most significant qubit in the standard representation |qN-1 ... q0>
        # Example: |coin, pos_N-1, ..., pos_0>
        # Check qubit ordering (safer)
        coin_index = qc.find_bit(coin_reg[0]).index
        pos_indices = [qc.find_bit(q).index for q in position_reg]

        # Assuming standard Qiskit ordering |qN-1 ... q0> where N=total_qubits
        # If coin is qubit N-1 (most significant):
        position_binary = full_binary_state[1:] # Example if coin is MSB
        # If coin is qubit 0 (least significant):
        # position_binary = full_binary_state[:-1]

        # Let's stick to the assumption coin is added last, thus MSB in default statevector string rep
        position_binary = full_binary_state[1:] # Bits for position (assuming coin is first/MSB)

        # Convert position binary string to integer index (Qiskit LSB is q0 - last char in string)
        position_index = int(position_binary[::-1], 2) # Reverse and convert

        # Add the probability |amplitude|^2 to the corresponding position
        # Access element using index on the NumPy array
        probabilities[position_index] += np.abs(statevector_np[i])**2

    # Ensure probabilities sum to 1 (or close to it due to numerical precision)
    # print(f"Total Probability: {np.sum(probabilities)}")

    # --- Visualization ---
    positions = np.arange(num_positions)

    plt.figure(figsize=(12, 6))
    plt.bar(positions, probabilities, width=0.8)
    # Add vertical line for the barrier
    plt.axvline(BARRIER_POSITION, color='r', linestyle='--', label=f'Barrier at Pos {BARRIER_POSITION}')
    # Add vertical line for the initial position
    plt.axvline(INITIAL_POSITION, color='g', linestyle=':', label=f'Start at Pos {INITIAL_POSITION}')

    plt.xlabel("Position Index")
    plt.ylabel("Probability")
    plt.title(f"Quantum Walk Probability Distribution after {NUM_STEPS} Steps")
    plt.xticks(positions) # Ensure all positions are labeled if feasible
    if num_positions > 20: # Avoid overcrowding x-axis labels if too many positions
         # Show fewer ticks for large number of positions
         tick_indices = np.linspace(0, num_positions - 1, min(num_positions, 15), dtype=int)
         plt.xticks(tick_indices)

    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(bottom=0) # Ensure y-axis starts at 0
    plt.tight_layout() # Adjust layout
    plt.show()

except QiskitError as e:
    print(f"Qiskit Error: {e}")
except KeyError as e:
    print(f"KeyError: Could not find data '{e}' in results. Check save instruction and label.")
    print("Ensure 'save_statevector' was added to the circuit before running.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc() # Print detailed traceback for unexpected errors


# Optional: Print probabilities
# print("\nProbabilities:")
# for i, prob in enumerate(probabilities):
#    if prob > 1e-5: # Print only non-negligible probabilities
#        print(f"Position {i}: Probability = {prob:.4f}")
