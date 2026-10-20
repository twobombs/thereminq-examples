# -*- coding: utf-8 -*-
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
# Consider enabling GPU simulation if available and qiskit-aer is installed with GPU support
# from qiskit_aer.providers.aer_provider import AerProvider
# simulator = AerProvider().get_backend('aer_simulator_statevector_gpu') # Example
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Statevector # For type hints
import math
import time # To estimate runtime

# --- Simulation Parameters ---
# *** Increased qubits for 64x64 grid ***
NUM_X_QUBITS = 6  # Qubits for X position (2^6 = 64 positions) - Increased
NUM_Y_QUBITS = 6  # Qubits for Y position (2^6 = 64 positions) - Increased
NUM_POS_QUBITS = NUM_X_QUBITS + NUM_Y_QUBITS
NUM_COIN_QUBITS = 2 # Need 2 qubits for 4 directions (e.g., +/- X, +/- Y)
NUM_ANCILLA_QUBITS = 1 # Ancilla for doubly-controlled operations
TOTAL_QUBITS = NUM_POS_QUBITS + NUM_COIN_QUBITS + NUM_ANCILLA_QUBITS # Updated total (12+2+1=15)

# *** Reduced number of steps due to extreme computational cost ***
NUM_STEPS = 50           # Number of actual quantum walk steps simulated (Reduced significantly)
INTERPOLATION_FRAMES = 1 # Number of animation frames per simulation step (Reduced to minimum)

# Initial position (center of the larger grid) - dynamically calculated
INITIAL_X = 1 << (NUM_X_QUBITS // 2)
INITIAL_Y = 1 << (NUM_Y_QUBITS // 2)

# Barrier: A line along a specific X coordinate - dynamically calculated
# Adjusted barrier position slightly to give more room for spreading
BARRIER_X_COORD = (1 << NUM_X_QUBITS) - 8 # Example: Barrier further from center
# Reduced barrier phase for a more subtle perturbation
BARRIER_PHASE = np.pi / 4 # Phase shift applied by the barrier

# --- Helper Functions (Increment, Decrement, Barrier - Unchanged) ---

def controlled_increment(qc, target_q, control_q, num_target_qubits):
    """Applies a controlled increment operation (+1) to the target register."""
    for i in range(num_target_qubits - 1):
        control_indices = [control_q] + [target_q[j] for j in range(i + 1)]
        qc.mcx(control_indices, target_q[i + 1])
    qc.cx(control_q, target_q[0])
    for i in range(num_target_qubits - 2, -1, -1):
        control_indices = [control_q] + [target_q[j] for j in range(i + 1)]
        qc.mcx(control_indices, target_q[i + 1])
        if i > 0:
             control_indices_cnot = [control_q] + [target_q[j] for j in range(i)]
             qc.mcx(control_indices_cnot, target_q[i])

def controlled_decrement(qc, target_q, control_q, num_target_qubits):
    """Applies a controlled decrement operation (-1) to the target register."""
    for i in range(num_target_qubits - 1):
        if i > 0:
             control_indices_cnot = [control_q] + [target_q[j] for j in range(i)]
             qc.mcx(control_indices_cnot, target_q[i])
        control_indices = [control_q] + [target_q[j] for j in range(i + 1)]
        qc.mcx(control_indices, target_q[i + 1])
    qc.cx(control_q, target_q[0])
    for i in range(num_target_qubits - 2, -1, -1):
        control_indices = [control_q] + [target_q[j] for j in range(i + 1)]
        qc.mcx(control_indices, target_q[i + 1])

def apply_2d_barrier(qc, x_reg, barrier_x_coord, barrier_phase, num_x_qubits):
    """Applies a phase shift if the walker's X coordinate matches barrier_x_coord."""
    if barrier_x_coord < 0 or barrier_x_coord >= (1 << num_x_qubits): return
    barrier_bin = format(barrier_x_coord, f'0{num_x_qubits}b')[::-1]
    for i in range(num_x_qubits):
        if barrier_bin[i] == '0': qc.x(x_reg[i])
    # Use multi-controlled phase gate (mcp)
    if num_x_qubits > 1:
        # Ensure all qubits in x_reg are used as controls if possible
        controls = list(x_reg)
        target = controls.pop() # Use the last qubit as target for mcp
        if controls: # Check if there are control qubits left
             qc.mcp(barrier_phase, controls, target)
        else: # If only one qubit in x_reg, use single qubit phase gate
             qc.p(barrier_phase, target)
    elif num_x_qubits == 1:
         qc.p(barrier_phase, x_reg[0])

    for i in range(num_x_qubits):
        if barrier_bin[i] == '0': qc.x(x_reg[i])

# --- Build the Quantum Circuit ---
x_reg = QuantumRegister(NUM_X_QUBITS, name='x')
y_reg = QuantumRegister(NUM_Y_QUBITS, name='y')
coin_reg = QuantumRegister(NUM_COIN_QUBITS, name='coin')
anc_reg = QuantumRegister(NUM_ANCILLA_QUBITS, name='anc')
qc = QuantumCircuit(x_reg, y_reg, coin_reg, anc_reg, name="QuantumWalk2D_3DAnim_64x64")

# 1. Initialize Position (X and Y)
initial_x_bin = format(INITIAL_X, f'0{NUM_X_QUBITS}b')[::-1]
initial_y_bin = format(INITIAL_Y, f'0{NUM_Y_QUBITS}b')[::-1]
for i in range(NUM_X_QUBITS):
    if initial_x_bin[i] == '1': qc.x(x_reg[i])
for i in range(NUM_Y_QUBITS):
    if initial_y_bin[i] == '1': qc.x(y_reg[i])
qc.h(coin_reg)
qc.save_statevector(label='step_0')
qc.barrier()

# --- 2D Quantum Walk Step Definition (Corrected Shift) ---
def walk_step_2d(qc, x_reg, y_reg, coin_reg, anc_reg):
    # 1. Coin Operator (Grover diffusion operator)
    qc.h(coin_reg); qc.x(coin_reg); qc.h(coin_reg[1]); qc.cx(coin_reg[0], coin_reg[1])
    qc.h(coin_reg[1]); qc.x(coin_reg); qc.h(coin_reg); qc.barrier(label="Coin Op")
    # 2. Controlled Shift based on coin state |c1 c0> using ancilla
    c1, c0 = coin_reg[1], coin_reg[0]; anc = anc_reg[0]
    # |00> -> +Y
    qc.x(coin_reg); qc.ccx(c0, c1, anc); controlled_increment(qc, y_reg, anc, NUM_Y_QUBITS)
    qc.ccx(c0, c1, anc); qc.x(coin_reg); qc.barrier()
    # |01> -> -Y
    qc.x(c1); qc.ccx(c0, c1, anc); controlled_decrement(qc, y_reg, anc, NUM_Y_QUBITS)
    qc.ccx(c0, c1, anc); qc.x(c1); qc.barrier()
    # |10> -> -X
    qc.x(c0); qc.ccx(c0, c1, anc); controlled_decrement(qc, x_reg, anc, NUM_X_QUBITS)
    qc.ccx(c0, c1, anc); qc.x(c0); qc.barrier()
    # |11> -> +X
    qc.ccx(c0, c1, anc); controlled_increment(qc, x_reg, anc, NUM_X_QUBITS)
    qc.ccx(c0, c1, anc); qc.barrier(label="Shift Op")

# --- Apply Walk Steps ---
# Loop for the reduced number of steps
print(f"Building circuit with {NUM_STEPS} steps...")
build_start_time = time.time()
for step in range(NUM_STEPS):
    walk_step_2d(qc, x_reg, y_reg, coin_reg, anc_reg)
    apply_2d_barrier(qc, x_reg, BARRIER_X_COORD, BARRIER_PHASE, NUM_X_QUBITS)
    qc.save_statevector(label=f'step_{step+1}')
    qc.barrier(label=f"End Step {step+1}")
build_end_time = time.time()
print(f"Circuit build time: {build_end_time - build_start_time:.2f} seconds")

# --- Simulation ---
# Try to set simulation options for potentially better performance/memory handling
# These options might vary depending on the Aer version and system capabilities
# simulator.set_options(fusion_enable=True, fusion_max_qubit=10, fusion_threshold=14) # Example fusion options
# simulator.set_options(statevector_parallel_threshold=14) # Example parallelization option

simulator = AerSimulator(method='statevector')
print(f"Simulating 2D walk circuit ({TOTAL_QUBITS} qubits, {NUM_STEPS} steps)...")
print("--- WARNING: THIS WILL REQUIRE SIGNIFICANT MEMORY AND TIME ---")
print("--- Consider reducing NUM_X_QUBITS/NUM_Y_QUBITS or NUM_STEPS if it fails/takes too long ---")
sim_start_time = time.time()
try:
    result = simulator.run(qc).result()
    sim_end_time = time.time()
    print(f"Simulation complete. Time taken: {sim_end_time - sim_start_time:.2f} seconds")
except Exception as e:
    print(f"\n--- SIMULATION FAILED ---")
    print(f"Error: {e}")
    print("This likely failed due to insufficient memory or excessive runtime.")
    print(f"Required statevector size: 2^{TOTAL_QUBITS} complex numbers.")
    print("Try reducing NUM_X_QUBITS, NUM_Y_QUBITS, or NUM_STEPS.")
    exit() # Stop the script if simulation fails

# --- Process Results for Each Step (Adjusted for Ancilla) ---
raw_probabilities_2d = [] # Store results from simulation
num_x_positions = 1 << NUM_X_QUBITS
num_y_positions = 1 << NUM_Y_QUBITS
max_prob = 0

print("Processing simulation results...")
proc_start_time = time.time()
try:
    # Process results for the reduced number of steps
    for step in range(NUM_STEPS + 1):
        statevector_obj = result.data(qc)[f'step_{step}']
        statevector_np = np.asarray(statevector_obj)
        probabilities_2d = np.zeros((num_x_positions, num_y_positions))
        if len(statevector_np) != (1 << TOTAL_QUBITS):
             raise ValueError(f"Statevector length mismatch at step {step}")

        for i in range(len(statevector_np)):
            # Efficiently extract indices using bit manipulation (faster for large statevectors)
            # Assumes qubit order: |anc c1 c0 yN..y0 xN..x0>
            x_index = i & ((1 << NUM_X_QUBITS) - 1) # Get last NUM_X_QUBITS bits
            y_index = (i >> NUM_X_QUBITS) & ((1 << NUM_Y_QUBITS) - 1) # Get next NUM_Y_QUBITS bits

            probabilities_2d[x_index, y_index] += np.abs(statevector_np[i])**2

        raw_probabilities_2d.append(probabilities_2d)
        step_max_prob = probabilities_2d.max()
        if step_max_prob > max_prob: max_prob = step_max_prob

except (KeyError, ValueError) as e:
    print(f"Error processing results: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during processing: {e}")
    import traceback; traceback.print_exc(); exit()
proc_end_time = time.time()
print(f"Result processing complete. Time taken: {proc_end_time - proc_start_time:.2f} seconds")

# --- Generate Interpolated Frames ---
# With INTERPOLATION_FRAMES = 1, this just copies the list
print("Generating interpolated frames (INTERPOLATION_FRAMES=1)...")
interp_start_time = time.time()
interpolated_probabilities_2d = list(raw_probabilities_2d) # Direct copy if INTERPOLATION_FRAMES=1
total_animation_frames = len(interpolated_probabilities_2d)
interp_end_time = time.time()
print(f"Generated {total_animation_frames} frames for animation. Time taken: {interp_end_time - interp_start_time:.2f} seconds")


# --- Animation Setup ---
print("Setting up 3D animation...")
anim_setup_start_time = time.time()
fig = plt.figure(figsize=(12, 9)) # Slightly larger figure
ax = fig.add_subplot(111, projection='3d')
x_coords = np.arange(num_x_positions)
y_coords = np.arange(num_y_positions)
X, Y = np.meshgrid(x_coords, y_coords)

# Initial surface plot using the first frame
surf = ax.plot_surface(X, Y, interpolated_probabilities_2d[0].T, cmap='viridis', edgecolor='none', vmin=0, vmax=max_prob*1.1 if max_prob > 0 else 0.1)

# Styling
ax.set_xlabel("X Position"); ax.set_ylabel("Y Position"); ax.set_zlabel("Probability")
ax.set_zlim(0, max_prob * 1.1 if max_prob > 0 else 0.1)
title = ax.set_title("2D Quantum Walk - Frame 0")
# Adjust ticks for larger grid - show fewer labels to avoid clutter
tick_skip_x = max(1, num_x_positions // 8) # Show ~8 ticks
tick_skip_y = max(1, num_y_positions // 8)
ax.set_xticks(x_coords[::tick_skip_x])
ax.set_yticks(y_coords[::tick_skip_y])


# --- Update Function for Animation ---
def update_3d(frame, probabilities_list, title_obj, x_mesh, y_mesh, axis, max_p):
    """Updates the 3D surface plot for each animation frame."""
    axis.clear() # Clear previous surface
    probabilities = probabilities_list[frame].T # Get probabilities for the current frame
    # Redraw the surface
    new_surf = axis.plot_surface(x_mesh, y_mesh, probabilities, cmap='viridis', edgecolor='none', vmin=0, vmax=max_p*1.1 if max_p > 0 else 0.1)
    # Reset styling
    axis.set_xlabel("X Position"); axis.set_ylabel("Y Position"); axis.set_zlabel("Probability")
    axis.set_zlim(0, max_p * 1.1 if max_p > 0 else 0.1)
    # Adjust ticks for larger grid
    axis.set_xticks(x_coords[::tick_skip_x])
    axis.set_yticks(y_coords[::tick_skip_y])

    # Update title to show frame number (which is same as step number here)
    new_title = axis.set_title(f"2D Quantum Walk - Step {frame}")
    return new_surf, new_title

# --- Create and Display Animation ---
# Adjust interval based on interpolation frames for smoother visual speed
animation_interval = 150 # Slower interval as interpolation is off

# Animate over the number of steps
ani = animation.FuncAnimation(fig, update_3d, frames=total_animation_frames,
                              fargs=(interpolated_probabilities_2d, title, X, Y, ax, max_prob),
                              interval=animation_interval, blit=False, repeat=False)

plt.tight_layout()
anim_setup_end_time = time.time()
print(f"Animation setup complete. Time taken: {anim_setup_end_time - anim_setup_start_time:.2f} seconds")
print("Displaying plot (Rendering may take time)...")
plt.show()

# Saving options
# print("Saving animation (This may take a very long time)...")
# save_start_time = time.time()
# try:
#     # ani.save('quantum_walk_2d_3d_64x64.gif', writer='pillow', fps=15)
#     ani.save('quantum_walk_2d_3d_64x64.mp4', writer='ffmpeg', fps=15)
#     save_end_time = time.time()
#     print(f"Animation saved. Time taken: {save_end_time - save_start_time:.2f} seconds")
# except Exception as e:
#     print(f"Error saving animation: {e}")

print("Script finished.")
